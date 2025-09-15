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

class SigLIP2DementiaClassifier(pl.LightningModule):
    """SigLIP2 기반 치매 진단 분류기"""
    
    def __init__(self, 
                 model_name: str = "google/siglip2-base-patch16-224",
                 num_classes: int = 2,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 100,
                 max_epochs: int = 10,
                 use_language_embedding: bool = True):
        
        super().__init__()
        self.save_hyperparameters()
        
        # SigLIP2 모델 로드 - 여러 방법 시도
        model_loaded = False
        
        # 방법 1: 직접 SigLIP2 로드
        try:
            from transformers import Siglip2Model, Siglip2Config
            print("🔄 SigLIP2 모델 직접 로드 시도...")
            self.siglip2 = Siglip2Model.from_pretrained(model_name)
            print("✅ SigLIP2 모델 로드 성공!")
            model_loaded = True
        except Exception as e:
            print(f"⚠️ 방법 1 실패: {e}")
        
        # 방법 2: Config만 가져와서 새 SigLIP2 모델 생성
        if not model_loaded:
            try:
                from transformers import Siglip2Model, Siglip2Config
                print("🔄 SigLIP2 Config로 새 모델 생성 시도...")
                config = Siglip2Config()  # 기본 설정 사용
                self.siglip2 = Siglip2Model(config)
                print("✅ 새 SigLIP2 모델 생성 성공! (사전훈련 가중치 없음)")
                model_loaded = True
            except Exception as e:
                print(f"⚠️ 방법 2 실패: {e}")
        
        # 방법 3: AutoModel 폴백
        if not model_loaded:
            print("🔄 AutoModel 폴백 사용...")
            self.siglip2 = AutoModel.from_pretrained(model_name)
            print(f"로드된 모델 타입: {type(self.siglip2)}")
            print(f"모델 설정: {self.siglip2.config.model_type if hasattr(self.siglip2.config, 'model_type') else 'Unknown'}")
        
        # 언어 임베딩 (언어 무관 학습을 위해)
        if use_language_embedding:
            self.language_embedding = nn.Embedding(10, 768)  # 최대 10개 언어 지원
        else:
            self.language_embedding = None
        
        # 분류 헤드
        hidden_size = self.siglip2.config.projection_dim
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
        """순전파"""
        # SigLIP2 모델 통과
        outputs = self.siglip2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # 멀티모달 임베딩 추출
        multimodal_embeddings = outputs.logits_per_image  # [batch_size, hidden_size]
        
        # 언어 임베딩 추가 (선택적)
        if self.language_embedding is not None and language_ids is not None:
            lang_embeddings = self.language_embedding(language_ids)
            multimodal_embeddings = multimodal_embeddings + lang_embeddings
        
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

def create_model(config) -> SigLIP2DementiaClassifier:
    """모델 생성"""
    return SigLIP2DementiaClassifier(
        model_name=config.model_name,
        num_classes=2,  # 치매 여부 (0: 정상, 1: 치매)
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_epochs=config.num_epochs,
        use_language_embedding=True  # 언어 무관 학습을 위해 활성화
    ) 