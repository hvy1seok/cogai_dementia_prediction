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
                 use_language_embedding: bool = True):
        
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
        
        # 분류 헤드 - 다양한 config 속성 지원
        if hasattr(self.siglip.config, 'projection_dim'):
            hidden_size = self.siglip.config.projection_dim
        elif hasattr(self.siglip.config, 'hidden_size'):
            hidden_size = self.siglip.config.hidden_size
        elif hasattr(self.siglip.config, 'vision_config') and hasattr(self.siglip.config.vision_config, 'hidden_size'):
            hidden_size = self.siglip.config.vision_config.hidden_size
        else:
            hidden_size = 768  # 기본값
        
        print(f"🔧 분류 헤드 hidden_size: {hidden_size}")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 언어 ID 매핑
        self.language_to_id = {
            'English': 0, 'Greek': 1, 'Korean': 2, 'Spanish': 3, 'French': 4,
            'German': 5, 'Italian': 6, 'Portuguese': 7, 'Japanese': 8, 'Chinese': 9
        }
        
        # 메트릭 초기화
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        
    def forward(self, input_ids, attention_mask, pixel_values, language_ids=None):
        """순전파 - SigLIP2 네이티브 다국어 지원"""
        # SigLIP2 모델 통과
        outputs = self.siglip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # SigLIP2의 멀티모달 특징 추출
        if hasattr(outputs, 'logits_per_image'):
            multimodal_embeddings = outputs.logits_per_image  # [batch_size, projection_dim]
        elif hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
            # 이미지와 텍스트 임베딩 결합
            multimodal_embeddings = (outputs.image_embeds + outputs.text_embeds) / 2
        else:
            # 폴백: 풀링된 출력 사용
            multimodal_embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        
        # 선택적 언어별 fine-tuning
        if self.language_embedding is not None and language_ids is not None:
            lang_emb = self.language_embedding(language_ids)  # [batch_size, 512]
            lang_emb = self.language_projection(lang_emb)     # [batch_size, 768]
            
            # 언어별 특징 추가 (작은 가중치로)
            if multimodal_embeddings.shape[-1] == lang_emb.shape[-1]:
                multimodal_embeddings = multimodal_embeddings + lang_emb * 0.1
        
        # 분류
        logits = self.classifier(multimodal_embeddings)
        return logits
    
    def training_step(self, batch, batch_idx):
        """훈련 스텝"""
        # 언어 ID 변환
        language_ids = self._get_language_ids(batch['language'])
        
        # 순전파
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            language_ids=language_ids
        )
        
        # 손실 계산
        loss = F.cross_entropy(logits, batch['labels'])
        
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
        
        # 순전파
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            language_ids=language_ids
        )
        
        # 손실 계산
        loss = F.cross_entropy(logits, batch['labels'])
        
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
        
        # 순전파
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            language_ids=language_ids
        )
        
        # 손실 계산
        loss = F.cross_entropy(logits, batch['labels'])
        
        # 정확도 계산
        acc = self.test_accuracy(logits.softmax(dim=-1), batch['labels'])
        
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
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
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
        use_language_embedding=True  # 언어 무관 학습을 위해 활성화
    ) 