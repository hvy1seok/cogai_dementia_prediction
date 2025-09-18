"""
SigLIP 기반 대조군 모델들
SigLIP의 개별 컴포넌트를 분리하여 공정한 비교 제공
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from config import ControlGroupConfig, AudioOnlyConfig, TextOnlyConfig, ConcatConfig

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SigLIPAudioOnlyModel(nn.Module):
    """
    SigLIP-Audio-Only: SigLIP의 이미지 인코더만 사용
    멜스펙토그램 → SigLIP Vision Encoder → 분류기
    """
    
    def __init__(self, config: AudioOnlyConfig):
        super().__init__()
        self.config = config
        
        # SigLIP 모델 로드 (이미지 인코더만 사용)
        print(f"🔄 SigLIP 모델 로드: {config.siglip_model}")
        self.siglip_model = AutoModel.from_pretrained(config.siglip_model)
        print(f"✅ SigLIP 로드 완료!")
        
        # SigLIP 이미지 프로세서
        self.processor = AutoProcessor.from_pretrained(config.siglip_model)
        
        # SigLIP의 실제 이미지 임베딩 차원 확인
        self.hidden_size = self.siglip_model.config.vision_config.hidden_size
        print(f"📊 SigLIP 이미지 임베딩 차원: {self.hidden_size}")
        
        # 분류기 (SigLIP 이미지 특징 → 분류)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes)
        )
        
        # 손실 함수 설정
        self.setup_loss_function()
    
    def setup_loss_function(self, class_weights=None):
        """손실 함수 설정"""
        if self.config.loss_type == "focal":
            alpha = class_weights[1] if class_weights is not None else self.config.focal_alpha
            self.criterion = FocalLoss(
                alpha=alpha, 
                gamma=self.config.focal_gamma, 
                reduction='mean'
            )
            print(f"📊 Focal Loss 사용 (alpha={alpha:.3f}, gamma={self.config.focal_gamma})")
        else:
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
                print(f"📊 Weighted Cross Entropy Loss 사용")
            else:
                self.criterion = nn.CrossEntropyLoss()
                print("📊 Cross Entropy Loss 사용")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """순전파 - 이미지만 처리"""
        # SigLIP 이미지 인코더만 사용
        with torch.no_grad():
            # SigLIP의 이미지 인코더 통과
            outputs = self.siglip_model.get_image_features(pixel_values=pixel_values)
        
        # 분류기 통과
        logits = self.classifier(outputs)
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """손실 계산"""
        if self.config.num_classes == 2:
            # 이진 분류: logits을 스칼라로 변환
            logits = logits[:, 1] - logits[:, 0]  # AD - HC
            return self.criterion(logits, labels.float())
        else:
            return self.criterion(logits, labels)

class SigLIPTextOnlyModel(nn.Module):
    """
    SigLIP-Text-Only: SigLIP의 텍스트 인코더 + Gemma 토크나이저
    텍스트 → Gemma 토크나이저 → SigLIP Text Encoder → 분류기
    """
    
    def __init__(self, config: TextOnlyConfig):
        super().__init__()
        self.config = config
        
        # SigLIP 모델 로드 (텍스트 인코더만 사용)
        print(f"🔄 SigLIP 모델 로드: {config.siglip_model}")
        self.siglip_model = AutoModel.from_pretrained(config.siglip_model)
        print(f"✅ SigLIP 로드 완료!")
        
        # Gemma 토크나이저 (SigLIP과 동일)
        print(f"🔄 Gemma 토크나이저 로드: {config.text_tokenizer}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✅ Gemma 토크나이저 로드 완료! (vocab_size: {self.tokenizer.vocab_size})")
        
        # SigLIP의 실제 텍스트 임베딩 차원 확인
        self.hidden_size = self.siglip_model.config.text_config.hidden_size
        print(f"📊 SigLIP 텍스트 임베딩 차원: {self.hidden_size}")
        
        # 분류기 (SigLIP 텍스트 특징 → 분류)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes)
        )
        
        # 손실 함수 설정
        self.setup_loss_function()
    
    def setup_loss_function(self, class_weights=None):
        """손실 함수 설정"""
        if self.config.loss_type == "focal":
            alpha = class_weights[1] if class_weights is not None else self.config.focal_alpha
            self.criterion = FocalLoss(
                alpha=alpha, 
                gamma=self.config.focal_gamma, 
                reduction='mean'
            )
            print(f"📊 Focal Loss 사용 (alpha={alpha:.3f}, gamma={self.config.focal_gamma})")
        else:
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
                print(f"📊 Weighted Cross Entropy Loss 사용")
            else:
                self.criterion = nn.CrossEntropyLoss()
                print("📊 Cross Entropy Loss 사용")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """순전파 - 텍스트만 처리"""
        # SigLIP 텍스트 인코더만 사용
        with torch.no_grad():
            # SigLIP의 텍스트 인코더 통과
            outputs = self.siglip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # 분류기 통과
        logits = self.classifier(outputs)
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """손실 계산"""
        if self.config.num_classes == 2:
            # 이진 분류: logits을 스칼라로 변환
            logits = logits[:, 1] - logits[:, 0]  # AD - HC
            return self.criterion(logits, labels.float())
        else:
            return self.criterion(logits, labels)

class SigLIPConcatModel(nn.Module):
    """
    SigLIP-Concat: SigLIP의 이미지+텍스트 인코더 분리 후 연결
    멜스펙토그램 → SigLIP Vision + 텍스트 → SigLIP Text → Concat → 분류기
    """
    
    def __init__(self, config: ConcatConfig):
        super().__init__()
        self.config = config
        
        # SigLIP 모델 로드 (이미지+텍스트 인코더 모두 사용)
        print(f"🔄 SigLIP 모델 로드: {config.siglip_model}")
        self.siglip_model = AutoModel.from_pretrained(config.siglip_model)
        print(f"✅ SigLIP 로드 완료!")
        
        # SigLIP 이미지 프로세서
        self.processor = AutoProcessor.from_pretrained(config.siglip_model)
        
        # Gemma 토크나이저 (SigLIP과 동일)
        print(f"🔄 Gemma 토크나이저 로드: {config.text_tokenizer}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✅ Gemma 토크나이저 로드 완료! (vocab_size: {self.tokenizer.vocab_size})")
        
        # SigLIP의 실제 임베딩 차원들
        self.image_hidden_size = self.siglip_model.config.vision_config.hidden_size
        self.text_hidden_size = self.siglip_model.config.text_config.hidden_size
        self.fused_feature_dim = self.image_hidden_size + self.text_hidden_size
        
        print(f"📊 SigLIP 이미지 임베딩 차원: {self.image_hidden_size}")
        print(f"📊 SigLIP 텍스트 임베딩 차원: {self.text_hidden_size}")
        print(f"📊 융합된 특징 차원: {self.fused_feature_dim}")
        
        # 융합된 특징을 위한 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.fused_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes)
        )
        
        # 손실 함수 설정
        self.setup_loss_function()
    
    def setup_loss_function(self, class_weights=None):
        """손실 함수 설정"""
        if self.config.loss_type == "focal":
            alpha = class_weights[1] if class_weights is not None else self.config.focal_alpha
            self.criterion = FocalLoss(
                alpha=alpha, 
                gamma=self.config.focal_gamma, 
                reduction='mean'
            )
            print(f"📊 Focal Loss 사용 (alpha={alpha:.3f}, gamma={self.config.focal_gamma})")
        else:
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
                print(f"📊 Weighted Cross Entropy Loss 사용")
            else:
                self.criterion = nn.CrossEntropyLoss()
                print("📊 Cross Entropy Loss 사용")
    
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """순전파 - 이미지+텍스트 분리 처리 후 연결"""
        # SigLIP 이미지 인코더
        with torch.no_grad():
            image_features = self.siglip_model.get_image_features(pixel_values=pixel_values)
        
        # SigLIP 텍스트 인코더
        with torch.no_grad():
            text_features = self.siglip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # 특징 연결 (Late Fusion)
        fused_features = torch.cat([image_features, text_features], dim=1)
        
        # 분류기 통과
        logits = self.classifier(fused_features)
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """손실 계산"""
        if self.config.num_classes == 2:
            # 이진 분류: logits을 스칼라로 변환
            logits = logits[:, 1] - logits[:, 0]  # AD - HC
            return self.criterion(logits, labels.float())
        else:
            return self.criterion(logits, labels)

def compute_metrics(predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    """성능 지표 계산"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    
    # AUC 계산 (이진 분류만)
    auc = 0.0
    optimal_threshold = 0.5
    if len(set(labels)) == 2:
        try:
            auc = roc_auc_score(labels, probabilities)
            fpr, tpr, thresholds = roc_curve(labels, probabilities)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        except Exception as e:
            print(f"⚠️ AUC 계산 실패: {e}")
            auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'auc': auc,
        'optimal_threshold': optimal_threshold
    }

def compute_language_specific_metrics(predictions: np.ndarray, labels: np.ndarray, 
                                    probabilities: np.ndarray, languages: List[str],
                                    optimal_threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """언어별 성능 지표 계산"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    lang_metrics = {}
    unique_languages = list(set(languages))
    
    for lang in unique_languages:
        lang_indices = [i for i, l in enumerate(languages) if l == lang]
        if len(lang_indices) == 0:
            continue
        
        lang_preds = predictions[lang_indices]
        lang_labels = labels[lang_indices]
        lang_probs = probabilities[lang_indices]
        
        # 해당 언어에 두 클래스가 모두 있는지 확인
        if len(set(lang_labels)) < 2:
            print(f"⚠️ {lang}: 단일 클래스만 존재, 지표 계산 제한")
            lang_metrics[lang] = {
                'accuracy': accuracy_score(lang_labels, lang_preds),
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'macro_f1': 0.0,
                'auc': 0.0
            }
            continue
        
        accuracy = accuracy_score(lang_labels, lang_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(lang_labels, lang_preds, average='weighted', zero_division=0)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(lang_labels, lang_preds, average='macro', zero_division=0)
        
        # AUC 계산
        auc = 0.0
        try:
            if len(set(lang_labels)) == 2:
                auc = roc_auc_score(lang_labels, lang_probs)
        except Exception as e:
            print(f"⚠️ {lang} AUC 계산 실패: {e}")
        
        lang_metrics[lang] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_f1': macro_f1,
            'auc': auc
        }
    
    return lang_metrics
