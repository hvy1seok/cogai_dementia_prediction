"""
SigLIP2 기반 치매 진단 모델 (PyTorch Lightning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, List, Optional
from torchmetrics import Accuracy

# Lion Optimizer 라이브러리 임포트
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
    print("🦁 lion-pytorch 라이브러리 로드 성공")
except ImportError:
    LION_AVAILABLE = False
    print("⚠️ lion-pytorch 라이브러리를 찾을 수 없습니다. pip install lion-pytorch를 실행하세요.")

# SAM Optimizer 구현 (참조: https://github.com/davda54/sam)
class SAM(torch.optim.Optimizer):
    """SAM: Sharpness-Aware Minimization"""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    dtype=torch.float32
                )
        return norm.to(shared_device)
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class FocalLoss(nn.Module):
    """
    Focal Loss 구현 - 불균형 데이터셋에 효과적
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SigLIPDementiaClassifier(pl.LightningModule):
    """
    SigLIP2 기반 다국어 치매 진단 분류기
    - Base: SigLIP2 (google/siglip2-base-patch16-naflex)
    - Native: Multilingual vision-language understanding
    """
    
    def __init__(self, 
                 model_name: str = "google/siglip2-base-patch16-naflex",
                 num_classes: int = 2,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 100,
                 max_epochs: int = 10,
                 use_language_embedding: bool = True,
                 loss_type: str = "cross_entropy",  # "cross_entropy", "focal", "bce"
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 optimizer_type: str = "adamw",  # "adamw", "lion", "sam"
                 sam_rho: float = 0.05):
        
        super().__init__()
        self.save_hyperparameters()
        
        # SigLIP2 모델 로드 (사전훈련 가중치 사용)
        print("🔄 SigLIP2 모델 로드 시도...")
        self.siglip = AutoModel.from_pretrained(model_name)
        print(f"✅ SigLIP2 모델 로드 성공! 타입: {type(self.siglip)}")
        print(f"📊 모델 크기: {self.siglip.config.vision_config.hidden_size if hasattr(self.siglip.config, 'vision_config') else '알 수 없음'}")
        
        # SigLIP2는 네이티브 다국어 지원 - 추가 언어 임베딩 선택적 사용
        if use_language_embedding:
            # 선택적 언어별 fine-tuning을 위한 임베딩
            self.language_embedding = nn.Embedding(10, 512)  # SigLIP2 크기에 맞춤
            self.language_projection = nn.Linear(512, 768)
        else:
            self.language_embedding = None
            self.language_projection = None
        
        # 분류 헤드 - SigLIP2의 실제 출력 차원에 맞춤
        # SigLIP2의 출력은 dynamic하므로 실행 시점에서 결정
        # 일단 placeholder로 설정하고 첫 번째 forward에서 재조정
        self.classifier = None  # 동적으로 생성될 예정
        self.hidden_size_detected = False
        
        # 언어 ID 매핑
        self.language_to_id = {
            'English': 0, 'Greek': 1, 'Korean': 2, 'Spanish': 3, 'French': 4,
            'German': 5, 'Italian': 6, 'Portuguese': 7, 'Japanese': 8, 'Chinese': 9
        }
        
        # 손실 함수 초기화
        if loss_type == "focal":
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"🎯 Focal Loss 사용: alpha={focal_alpha}, gamma={focal_gamma}")
        elif loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
            print("⚖️ BCE Loss 사용")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("📊 Cross Entropy Loss 사용")
        
        # 메트릭 초기화
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
    def forward(self, input_ids, attention_mask=None, pixel_values=None, pixel_attention_mask=None, spatial_shapes=None, language_ids=None):
        """순전파 - SigLIP2 네이티브 다국어 지원"""
        # SigLIP2 모델 통과 (모든 필요한 입력 포함)
        model_inputs = {
            'input_ids': input_ids,
            'pixel_values': pixel_values
        }
        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask
        if pixel_attention_mask is not None:
            model_inputs['pixel_attention_mask'] = pixel_attention_mask
        if spatial_shapes is not None:
            model_inputs['spatial_shapes'] = spatial_shapes
            
        outputs = self.siglip(**model_inputs)
        
        # SigLIP2의 멀티모달 특징 추출 (고정 차원 사용)
        if hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
            # 이미지와 텍스트 임베딩 결합 (고정 차원!)
            multimodal_embeddings = (outputs.image_embeds + outputs.text_embeds) / 2
        elif hasattr(outputs, 'pooler_output'):
            multimodal_embeddings = outputs.pooler_output
        else:
            # 폴백: 마지막 히든 상태의 평균
            multimodal_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # logits_per_image는 가변 차원이므로 사용하지 않음!
        
        # 이제 고정 차원을 사용하므로 분류기 한 번만 생성
        if self.classifier is None:
            actual_hidden_size = multimodal_embeddings.shape[-1]
            self.classifier = nn.Sequential(
                nn.Linear(actual_hidden_size, actual_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(actual_hidden_size // 2, self.hparams.num_classes)
            ).to(multimodal_embeddings.device)
        
        # 언어 임베딩은 SigLIP2 네이티브 다국어 능력으로 대체
        
        # 분류
        logits = self.classifier(multimodal_embeddings)
        return logits
    
    def training_step(self, batch, batch_idx):
        """훈련 스텝"""
        # 언어 ID 변환
        language_ids = self._get_language_ids(batch['language'])
        
        # 안전한 입력 준비
        input_ids = batch['input_ids']
        pixel_values = batch['pixel_values']
        attention_mask = batch.get('attention_mask', None)
        pixel_attention_mask = batch.get('pixel_attention_mask', None)
        spatial_shapes = batch.get('spatial_shapes', None)
        
        # 순전파
        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            language_ids=language_ids
        )
        
        # 손실 계산
        if self.hparams.loss_type == "bce":
            # BCE는 이진 분류용이므로 라벨을 float으로 변환하고 로짓의 두 번째 클래스만 사용
            labels_bce = batch['labels'].float()
            logits_bce = logits[:, 1]  # 치매 클래스 확률만 사용
            loss = self.criterion(logits_bce, labels_bce)
        else:
            loss = self.criterion(logits, batch['labels'])
        
        # 정확도 계산
        acc = self.train_accuracy(logits.softmax(dim=-1), batch['labels'])
        
        # 로깅
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """검증 스텝"""
        # 언어 ID 변환
        language_ids = self._get_language_ids(batch['language'])
        
        # 안전한 입력 준비
        input_ids = batch['input_ids']
        pixel_values = batch['pixel_values']
        attention_mask = batch.get('attention_mask', None)
        pixel_attention_mask = batch.get('pixel_attention_mask', None)
        spatial_shapes = batch.get('spatial_shapes', None)
        
        # 순전파
        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            language_ids=language_ids
        )
        
        # 손실 계산
        if self.hparams.loss_type == "bce":
            # BCE는 이진 분류용이므로 라벨을 float으로 변환하고 로짓의 두 번째 클래스만 사용
            labels_bce = batch['labels'].float()
            logits_bce = logits[:, 1]  # 치매 클래스 확률만 사용
            loss = self.criterion(logits_bce, labels_bce)
        else:
            loss = self.criterion(logits, batch['labels'])
        
        # 정확도 계산
        acc = self.val_accuracy(logits.softmax(dim=-1), batch['labels'])
        
        # 예측값 저장 (나중에 메트릭 계산용)
        self.validation_step_outputs.append({
            'logits': logits,
            'labels': batch['labels'],
            'loss': loss
        })
        
        # 로깅
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """테스트 스텝"""
        # 언어 ID 변환
        language_ids = self._get_language_ids(batch['language'])
        
        # 안전한 입력 준비
        input_ids = batch['input_ids']
        pixel_values = batch['pixel_values']
        attention_mask = batch.get('attention_mask', None)
        pixel_attention_mask = batch.get('pixel_attention_mask', None)
        spatial_shapes = batch.get('spatial_shapes', None)
        
        # 순전파
        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            language_ids=language_ids
        )
        
        # 손실 계산
        if self.hparams.loss_type == "bce":
            # BCE는 이진 분류용이므로 라벨을 float으로 변환하고 로짓의 두 번째 클래스만 사용
            labels_bce = batch['labels'].float()
            logits_bce = logits[:, 1]  # 치매 클래스 확률만 사용
            loss = self.criterion(logits_bce, labels_bce)
        else:
            loss = self.criterion(logits, batch['labels'])
        
        # 정확도 계산
        acc = self.test_accuracy(logits.softmax(dim=-1), batch['labels'])
        
        # AUC 계산 (배치별)
        probs = F.softmax(logits, dim=-1)
        if logits.shape[1] == 2 and len(torch.unique(batch['labels'])) > 1:
            try:
                batch_auc = roc_auc_score(batch['labels'].cpu(), probs[:, 1].cpu())
                self.log('test_auc', batch_auc, prog_bar=True, sync_dist=True)
            except ValueError:
                # 배치에 한 클래스만 있는 경우 AUC 계산 불가
                pass
        
        # 예측값 저장
        self.test_step_outputs.append({
            'logits': logits,
            'labels': batch['labels'],
            'loss': loss
        })
        
        # 로깅
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_start(self):
        """검증 에포크 시작 시"""
        self.validation_step_outputs = []
    
    def on_validation_epoch_end(self):
        """검증 에포크 종료 시"""
        self._compute_validation_metrics()
    
    def on_test_epoch_start(self):
        """테스트 에포크 시작 시"""
        self.test_step_outputs = []
    
    def on_test_epoch_end(self):
        """테스트 에포크 종료 시"""
        self._compute_test_metrics()
    
    def _get_language_ids(self, languages: List[str]) -> torch.Tensor:
        """언어를 ID로 변환"""
        if self.language_embedding is None:
            return None
        
        language_ids = []
        for lang in languages:
            lang_id = self.language_to_id.get(lang, 0)  # 기본값은 English
            language_ids.append(lang_id)
        
        return torch.tensor(language_ids, device=self.device)
    
    def _compute_validation_metrics(self):
        """검증 메트릭 계산"""
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # 예측값
        probs = F.softmax(all_logits, dim=-1)
        preds = torch.argmax(all_logits, dim=-1)
        
        # 메트릭 계산
        accuracy = accuracy_score(all_labels.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels.cpu(), preds.cpu(), average='weighted'
        )
        
        # ROC AUC (이진 분류인 경우)
        if all_logits.shape[1] == 2:
            auc = roc_auc_score(all_labels.cpu(), probs[:, 1].cpu())
        else:
            auc = 0.0
        
        # 로깅
        self.log('val_accuracy_final', accuracy)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)
        self.log('val_auc', auc)
        
        # wandb에 상세 메트릭 로깅
        if self.logger:
            self.logger.experiment.log({
                'val/accuracy': accuracy,
                'val/precision': precision,
                'val/recall': recall,
                'val/f1': f1,
                'val/auc': auc
            })
    
    def _compute_test_metrics(self):
        """테스트 메트릭 계산"""
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        # 예측값
        probs = F.softmax(all_logits, dim=-1)
        preds = torch.argmax(all_logits, dim=-1)
        
        # 메트릭 계산
        accuracy = accuracy_score(all_labels.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels.cpu(), preds.cpu(), average='weighted'
        )
        
        # ROC AUC (이진 분류인 경우)
        if all_logits.shape[1] == 2:
            auc = roc_auc_score(all_labels.cpu(), probs[:, 1].cpu())
        else:
            auc = 0.0
        
        # 로깅
        self.log('test_accuracy_final', accuracy)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        self.log('test_auc', auc)
        
        # wandb에 상세 메트릭 로깅
        if self.logger:
            self.logger.experiment.log({
                'test/accuracy': accuracy,
                'test/precision': precision,
                'test/recall': recall,
                'test/f1': f1,
                'test/auc': auc
            })
    
    def configure_optimizers(self):
        """옵티마이저 설정"""
        # 가중치 감쇠를 적용하지 않을 파라미터들
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 파라미터 그룹 생성
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        # 옵티마이저 선택
        if self.hparams.optimizer_type == "lion":
            if not LION_AVAILABLE:
                print("⚠️ lion-pytorch 라이브러리가 없습니다. AdamW로 대체합니다.")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay
                )
                print(f"⚡ AdamW Optimizer 사용 (Lion 대체): lr={self.hparams.learning_rate}")
            else:
                optimizer = Lion(
                    optimizer_grouped_parameters,
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay
                )
                print(f"🦁 Lion Optimizer 사용 (lion-pytorch): lr={self.hparams.learning_rate}")
        elif self.hparams.optimizer_type == "sam":
            optimizer = SAM(
                self.parameters(),
                torch.optim.AdamW,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                rho=self.hparams.sam_rho
            )
            print(f"🎯 SAM Optimizer 사용: lr={self.hparams.learning_rate}, rho={self.hparams.sam_rho}")
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            print(f"⚡ AdamW Optimizer 사용: lr={self.hparams.learning_rate}")
        
        # 학습률 스케줄러
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

def create_model(config) -> SigLIPDementiaClassifier:
    """모델 생성"""
    return SigLIPDementiaClassifier(
        model_name=config.model_name,
        num_classes=2,  # 치매 여부 (0: 정상, 1: 치매)
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_epochs=config.num_epochs,
        use_language_embedding=True,  # 언어 무관 학습을 위해 활성화
        loss_type=config.loss_type,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        optimizer_type=config.optimizer_type,
        sam_rho=config.sam_rho
    ) 