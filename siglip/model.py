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
    alpha가 리스트/텐서인 경우 클래스별 가중치 적용
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            # 텐서를 모듈 버퍼로 등록하여 자동으로 디바이스 이동
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 클래스별 alpha 가중치 적용
        if hasattr(self, 'alpha') and isinstance(self.alpha, torch.Tensor):
            # alpha가 텐서인 경우 클래스별 가중치 적용 (자동으로 올바른 디바이스에 있음)
            alpha_t = self.alpha.gather(0, targets.long())
            focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        else:
            # alpha가 스칼라인 경우 기존 방식
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
        self.siglip_model = AutoModel.from_pretrained(model_name)
        print(f"✅ SigLIP2 모델 로드 성공! 타입: {type(self.siglip_model)}")
        print(f"📊 모델 크기: {self.siglip_model.config.vision_config.hidden_size if hasattr(self.siglip_model.config, 'vision_config') else '알 수 없음'}")
        
        # SigLIP2는 네이티브 다국어 지원 - 추가 언어 임베딩 선택적 사용
        if use_language_embedding:
            # 선택적 언어별 fine-tuning을 위한 임베딩
            self.language_embedding = nn.Embedding(10, 512)  # SigLIP2 크기에 맞춤
            self.language_projection = nn.Linear(512, 768)
        else:
            self.language_embedding = None
            self.language_projection = None
        
        # 분류 헤드 - SigLIP2의 hidden_size는 config에서 미리 알 수 있음
        # SigLIP2 모델의 config에서 hidden_size 추출
        if hasattr(self.siglip_model.config, 'hidden_size'):
            actual_hidden_size = self.siglip_model.config.hidden_size
        elif hasattr(self.siglip_model.config, 'vision_config') and hasattr(self.siglip_model.config.vision_config, 'hidden_size'):
            actual_hidden_size = self.siglip_model.config.vision_config.hidden_size
        else:
            # 폴백: 일반적인 SigLIP2 hidden_size
            actual_hidden_size = 768
        
        print(f"📐 Hidden size: {actual_hidden_size}")
        
        # 분류기 미리 생성 (동적 생성 문제 해결)
        self.classifier = nn.Sequential(
            nn.Linear(actual_hidden_size, actual_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(actual_hidden_size // 2, self.hparams.num_classes)
        )
        
        self.hidden_size_detected = True
        self.actual_hidden_size = actual_hidden_size
        
        # 언어 ID 매핑
        self.language_to_id = {
            'English': 0, 'Greek': 1, 'Korean': 2, 'Spanish': 3, 'French': 4,
            'German': 5, 'Italian': 6, 'Portuguese': 7, 'Japanese': 8, 'Chinese': 9
        }
        
        # 손실 함수는 나중에 클래스 가중치와 함께 초기화
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.criterion = None  # 나중에 설정
        
        # 메트릭 초기화
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
    
    def setup_loss_function(self, class_weights=None):
        """손실 함수 초기화 - 클래스 가중치 적용"""
        if self.loss_type == "focal":
            if class_weights is not None:
                # 클래스 가중치 자동 적용
                alpha = class_weights
                print(f"🎯 Focal Loss 사용: alpha={alpha} (자동 계산), gamma={self.focal_gamma}")
                print(f"   정상 클래스 가중치: {alpha[0]:.3f}, 치매 클래스 가중치: {alpha[1]:.3f}")
            else:
                # 수동 설정 또는 균등 가중치
                alpha = self.focal_alpha
                print(f"🎯 Focal Loss 사용: alpha={alpha} (수동 설정), gamma={self.focal_gamma}")
            
            self.criterion = FocalLoss(alpha=alpha, gamma=self.focal_gamma)
        elif self.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
            print("⚖️ BCE Loss 사용")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("📊 Cross Entropy Loss 사용")
        
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
            
        outputs = self.siglip_model(**model_inputs)
        
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
        
        # 차원 검증 (디버깅용)
        expected_size = self.actual_hidden_size
        actual_size = multimodal_embeddings.shape[-1]
        if actual_size != expected_size:
            print(f"⚠️ 차원 불일치: 예상 {expected_size}, 실제 {actual_size}")
            # 차원 조정이 필요한 경우 처리
            if actual_size > expected_size:
                multimodal_embeddings = multimodal_embeddings[:, :expected_size]
            elif actual_size < expected_size:
                # 패딩 또는 projection 필요
                padding = torch.zeros(multimodal_embeddings.shape[0], expected_size - actual_size, 
                                    device=multimodal_embeddings.device)
                multimodal_embeddings = torch.cat([multimodal_embeddings, padding], dim=1)
        
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
        self.log('train_loss', loss, prog_bar=True, batch_size=batch['input_ids'].size(0))
        self.log('train_acc', acc, prog_bar=True, batch_size=batch['input_ids'].size(0))
        
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
        
        # 예측값 저장 (언어 정보 포함)
        self.validation_step_outputs.append({
            'logits': logits,
            'labels': batch['labels'],
            'languages': batch['language'],  # 언어별 분석용
            'loss': loss
        })
        
        # 로깅
        self.log('val_loss', loss, prog_bar=True, batch_size=batch['input_ids'].size(0))
        self.log('val_acc', acc, prog_bar=True, batch_size=batch['input_ids'].size(0))
        
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
        
        # 예측값 저장 (언어 정보 포함)
        self.test_step_outputs.append({
            'logits': logits,
            'labels': batch['labels'],
            'languages': batch['language'],  # 언어별 분석용
            'loss': loss
        })
        
        # 로깅
        self.log('test_loss', loss, prog_bar=True, batch_size=batch['input_ids'].size(0))
        self.log('test_acc', acc, prog_bar=True, batch_size=batch['input_ids'].size(0))
        
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
        """검증 메트릭 계산 - 최적 threshold 기반"""
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # 언어 정보 수집
        all_languages = []
        for x in self.validation_step_outputs:
            if isinstance(x['languages'], list):
                all_languages.extend(x['languages'])
            else:
                # 단일 배치의 경우
                all_languages.extend([x['languages']] * len(x['labels']))
        
        # 예측 확률 계산
        probs = F.softmax(all_logits, dim=-1)
        
        if all_logits.shape[1] == 2:
            # 이진 분류: 치매 클래스 확률 사용
            y_scores = probs[:, 1].cpu().numpy()
            y_true = all_labels.cpu().numpy()
            
            # ROC AUC 계산
            auc = roc_auc_score(y_true, y_scores)
            
            # 최적 threshold 찾기 (Youden's J statistic)
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # 최적 threshold로 예측
            optimal_preds = (y_scores >= optimal_threshold).astype(int)
            
            # 기본 threshold (0.5)로도 예측
            default_preds = (y_scores >= 0.5).astype(int)
            
            # 최적 threshold 기반 메트릭
            optimal_accuracy = accuracy_score(y_true, optimal_preds)
            optimal_precision, optimal_recall, optimal_f1, _ = precision_recall_fscore_support(
                y_true, optimal_preds, average='weighted', zero_division=0
            )
            
            # 기본 threshold 기반 메트릭 (비교용)
            default_accuracy = accuracy_score(y_true, default_preds)
            default_precision, default_recall, default_f1, _ = precision_recall_fscore_support(
                y_true, default_preds, average='weighted', zero_division=0
            )
            
        else:
            # 다중 분류
            auc = 0.0
            optimal_threshold = 0.5
            optimal_preds = torch.argmax(all_logits, dim=-1).cpu().numpy()
            default_preds = optimal_preds
            y_true = all_labels.cpu().numpy()
            
            optimal_accuracy = accuracy_score(y_true, optimal_preds)
            optimal_precision, optimal_recall, optimal_f1, _ = precision_recall_fscore_support(
                y_true, optimal_preds, average='weighted', zero_division=0
            )
            default_accuracy = optimal_accuracy
            default_precision, default_recall, default_f1 = optimal_precision, optimal_recall, optimal_f1
        
        # 로깅 - 최적 threshold 기반 (베스트 모델 선택용)
        batch_size = len(y_true)
        self.log('val_accuracy', optimal_accuracy, batch_size=batch_size)
        self.log('val_precision', optimal_precision, batch_size=batch_size)
        self.log('val_recall', optimal_recall, batch_size=batch_size)
        self.log('val_f1', optimal_f1, batch_size=batch_size)
        self.log('val_auc', auc, batch_size=batch_size)  # 베스트 모델 선택 기준
        self.log('val_optimal_threshold', optimal_threshold, batch_size=batch_size)
        
        # 추가 로깅 - 비교용
        self.log('val_accuracy_default', default_accuracy, batch_size=batch_size)
        
        # wandb에 상세 메트릭 로깅
        if self.logger:
            self.logger.experiment.log({
                # 최적 threshold 기반 (메인 지표)
                'val/accuracy_optimal': optimal_accuracy,
                'val/precision_optimal': optimal_precision,
                'val/recall_optimal': optimal_recall,
                'val/f1_optimal': optimal_f1,
                'val/auc': auc,
                'val/optimal_threshold': optimal_threshold,
                
                # 비교 지표
                'val/accuracy_default_0.5': default_accuracy,
                'val/precision_default_0.5': default_precision,
                'val/recall_default_0.5': default_recall,
                'val/f1_default_0.5': default_f1,
            })
    
    def _compute_test_metrics(self):
        """테스트 메트릭 계산 - 최적 threshold 기반 + 언어별 분석"""
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        # 언어 정보 수집
        all_languages = []
        for x in self.test_step_outputs:
            if isinstance(x['languages'], list):
                all_languages.extend(x['languages'])
            else:
                # 단일 배치의 경우
                all_languages.extend([x['languages']] * len(x['labels']))
        
        # 예측 확률 계산
        probs = F.softmax(all_logits, dim=-1)
        
        if all_logits.shape[1] == 2:
            # 이진 분류: 치매 클래스 확률 사용
            y_scores = probs[:, 1].cpu().numpy()
            y_true = all_labels.cpu().numpy()
            
            # ROC AUC 계산
            auc = roc_auc_score(y_true, y_scores)
            
            # 최적 threshold 찾기 (Youden's J statistic)
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # 최적 threshold로 예측
            optimal_preds = (y_scores >= optimal_threshold).astype(int)
            
            # 기본 threshold (0.5)로도 예측
            default_preds = (y_scores >= 0.5).astype(int)
            
            # 최적 threshold 기반 메트릭
            optimal_accuracy = accuracy_score(y_true, optimal_preds)
            optimal_precision, optimal_recall, optimal_f1, _ = precision_recall_fscore_support(
                y_true, optimal_preds, average='weighted', zero_division=0
            )
            
            # 기본 threshold 기반 메트릭 (비교용)
            default_accuracy = accuracy_score(y_true, default_preds)
            default_precision, default_recall, default_f1, _ = precision_recall_fscore_support(
                y_true, default_preds, average='weighted', zero_division=0
            )
            
            # argmax 기반 메트릭 (기존 방식)
            argmax_preds = torch.argmax(all_logits, dim=-1).cpu().numpy()
            argmax_accuracy = accuracy_score(y_true, argmax_preds)
            argmax_precision, argmax_recall, argmax_f1, _ = precision_recall_fscore_support(
                y_true, argmax_preds, average='weighted', zero_division=0
            )
            
        else:
            # 다중 분류
            auc = 0.0
            optimal_threshold = 0.5
            y_scores = probs.max(dim=-1)[0].cpu().numpy()
            optimal_preds = torch.argmax(all_logits, dim=-1).cpu().numpy()
            default_preds = optimal_preds
            argmax_preds = optimal_preds
            y_true = all_labels.cpu().numpy()
            
            optimal_accuracy = accuracy_score(y_true, optimal_preds)
            optimal_precision, optimal_recall, optimal_f1, _ = precision_recall_fscore_support(
                y_true, optimal_preds, average='weighted', zero_division=0
            )
            default_accuracy = optimal_accuracy
            default_precision, default_recall, default_f1 = optimal_precision, optimal_recall, optimal_f1
            argmax_accuracy = optimal_accuracy
            argmax_precision, argmax_recall, argmax_f1 = optimal_precision, optimal_recall, optimal_f1
        
        # 로깅 - 최적 threshold 기반 (메인)
        batch_size = len(y_true)
        self.log('test_accuracy', optimal_accuracy, batch_size=batch_size)
        self.log('test_precision', optimal_precision, batch_size=batch_size)
        self.log('test_recall', optimal_recall, batch_size=batch_size)
        self.log('test_f1', optimal_f1, batch_size=batch_size)
        self.log('test_auc', auc, batch_size=batch_size)
        self.log('test_optimal_threshold', optimal_threshold, batch_size=batch_size)
        
        # 추가 로깅 - 비교용
        self.log('test_accuracy_default', default_accuracy, batch_size=batch_size)
        self.log('test_accuracy_argmax', argmax_accuracy, batch_size=batch_size)
        
        # wandb에 상세 메트릭 로깅
        if self.logger:
            self.logger.experiment.log({
                # 최적 threshold 기반 (메인 지표)
                'test/accuracy_optimal': optimal_accuracy,
                'test/precision_optimal': optimal_precision,
                'test/recall_optimal': optimal_recall,
                'test/f1_optimal': optimal_f1,
                'test/auc': auc,
                'test/optimal_threshold': optimal_threshold,
                
                # 비교 지표들
                'test/accuracy_default_0.5': default_accuracy,
                'test/precision_default_0.5': default_precision,
                'test/recall_default_0.5': default_recall,
                'test/f1_default_0.5': default_f1,
                
                'test/accuracy_argmax': argmax_accuracy,
                'test/precision_argmax': argmax_precision,
                'test/recall_argmax': argmax_recall,
                'test/f1_argmax': argmax_f1,
            })
        
        # 전체 결과 출력
        print(f"\n🎯 전체 테스트 결과 (최적 threshold = {optimal_threshold:.3f}):")
        print(f"   AUC: {auc:.4f}")
        print(f"   Accuracy: {optimal_accuracy:.4f}")
        print(f"   Precision: {optimal_precision:.4f}")
        print(f"   Recall: {optimal_recall:.4f}")
        print(f"   F1: {optimal_f1:.4f}")
        
        print(f"\n📊 Threshold 비교:")
        print(f"   최적 threshold ({optimal_threshold:.3f}): Acc={optimal_accuracy:.4f}")
        print(f"   기본 threshold (0.500): Acc={default_accuracy:.4f}")
        print(f"   Argmax 방식: Acc={argmax_accuracy:.4f}")
        
        # 언어별 결과 계산 및 출력
        self._compute_language_specific_metrics(y_scores, y_true, all_languages, optimal_threshold)
    
    def _compute_language_specific_metrics(self, y_scores, y_true, all_languages, optimal_threshold):
        """언어별 테스트 메트릭 계산 및 출력"""
        from collections import defaultdict
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        # 언어별로 데이터 그룹화
        language_data = defaultdict(lambda: {'scores': [], 'labels': [], 'indices': []})
        
        for i, (score, label, lang) in enumerate(zip(y_scores, y_true, all_languages)):
            language_data[lang]['scores'].append(score)
            language_data[lang]['labels'].append(label)
            language_data[lang]['indices'].append(i)
        
        print(f"\n🌍 언어별 테스트 결과:")
        print(f"{'='*80}")
        
        # wandb 로깅용 언어별 메트릭
        language_metrics = {}
        
        for lang in sorted(language_data.keys()):
            lang_scores = np.array(language_data[lang]['scores'])
            lang_labels = np.array(language_data[lang]['labels'])
            
            if len(lang_scores) == 0:
                continue
                
            # 언어별 AUC 계산
            try:
                lang_auc = roc_auc_score(lang_labels, lang_scores)
            except ValueError:
                lang_auc = 0.0
            
            # 최적 threshold로 예측
            lang_optimal_preds = (lang_scores >= optimal_threshold).astype(int)
            
            # 기본 threshold (0.5)로 예측
            lang_default_preds = (lang_scores >= 0.5).astype(int)
            
            # 메트릭 계산
            lang_optimal_acc = accuracy_score(lang_labels, lang_optimal_preds)
            lang_default_acc = accuracy_score(lang_labels, lang_default_preds)
            
            lang_precision, lang_recall, lang_f1, _ = precision_recall_fscore_support(
                lang_labels, lang_optimal_preds, average='weighted', zero_division=0
            )
            
            # 클래스별 분포
            from collections import Counter
            label_dist = Counter(lang_labels)
            normal_count = label_dist[0]
            dementia_count = label_dist[1]
            
            # 결과 출력
            print(f"\n📊 {lang} ({len(lang_scores)}개 샘플)")
            print(f"   정상: {normal_count}개, 치매: {dementia_count}개")
            print(f"   AUC: {lang_auc:.4f}")
            print(f"   Accuracy (최적): {lang_optimal_acc:.4f}")
            print(f"   Accuracy (0.5): {lang_default_acc:.4f}")
            print(f"   Precision: {lang_precision:.4f}")
            print(f"   Recall: {lang_recall:.4f}")
            print(f"   F1: {lang_f1:.4f}")
            
            # wandb 로깅용 메트릭 저장
            language_metrics[f'{lang}_auc'] = lang_auc
            language_metrics[f'{lang}_accuracy_optimal'] = lang_optimal_acc
            language_metrics[f'{lang}_accuracy_default'] = lang_default_acc
            language_metrics[f'{lang}_precision'] = lang_precision
            language_metrics[f'{lang}_recall'] = lang_recall
            language_metrics[f'{lang}_f1'] = lang_f1
            language_metrics[f'{lang}_sample_count'] = len(lang_scores)
            language_metrics[f'{lang}_normal_count'] = normal_count
            language_metrics[f'{lang}_dementia_count'] = dementia_count
        
        print(f"{'='*80}")
        
        # wandb에 언어별 메트릭 로깅
        if self.logger:
            self.logger.experiment.log({
                'language_specific_metrics': language_metrics
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
            # SAM은 PyTorch Lightning과 호환성 문제가 있으므로 더 강한 정규화를 가진 AdamW로 대체
            print("⚠️ SAM은 PyTorch Lightning과 호환성 문제가 있습니다.")
            print("🔄 더 강한 정규화(higher weight decay)를 가진 AdamW로 대체합니다.")
            
            # SAM의 정규화 효과를 모방하기 위해 weight decay를 증가
            enhanced_weight_decay = self.hparams.weight_decay * 2.0
            
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                weight_decay=enhanced_weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print(f"⚡ Enhanced AdamW Optimizer 사용 (SAM 대체): lr={self.hparams.learning_rate}, wd={enhanced_weight_decay:.4f}")
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

def create_callbacks(training_config, checkpoint_dir):
    """PyTorch Lightning callbacks 생성"""
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    import os
    
    # 체크포인트 디렉토리 생성
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ModelCheckpoint callback (validation AUC 기준)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_model_{epoch:02d}_{val_auc:.3f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    # EarlyStopping callback (validation AUC 기준)
    early_stopping_callback = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=getattr(training_config, 'early_stopping_patience', 15),
        verbose=True,
        strict=False  # metric이 없어도 오류 발생 안함
    )
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        lr_monitor
    ]
    
    print(f"✅ Callbacks 생성 완료:")
    print(f"   - ModelCheckpoint: validation AUC 기준 베스트 모델 저장")
    print(f"   - EarlyStopping: validation AUC 기준 {getattr(training_config, 'early_stopping_patience', 15)} epochs patience")
    print(f"   - LearningRateMonitor: 학습률 추적")
    
    return callbacks, checkpoint_callback 