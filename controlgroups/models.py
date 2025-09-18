"""
Control Group Models
대조군 모델 구현: Audio-only, Text-only, Concat Late Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor,
    ViTModel, XLMRobertaModel
)
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .config import AudioOnlyConfig, TextOnlyConfig, ConcatConfig

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

class AudioOnlyModel(nn.Module):
    """
    ViT-Spec (Audio-only, multilingual joint)
    오디오 스펙트로그램만 사용하는 멀티링궐 모델
    """
    
    def __init__(self, config: AudioOnlyConfig):
        super().__init__()
        self.config = config
        
        # ViT 오디오 인코더
        self.audio_encoder = ViTModel.from_pretrained(config.model_name)
        
        # 분류기
        layers = []
        input_dim = config.audio_feature_dim
        
        for hidden_dim in config.classifier_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, config.num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # 손실 함수
        if config.loss_type == "focal":
            self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, 3, 224, 224] - 스펙트로그램 이미지
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # ViT 인코딩
        audio_features = self.audio_encoder(pixel_values=pixel_values).last_hidden_state
        
        # [CLS] 토큰 사용 (첫 번째 토큰)
        cls_features = audio_features[:, 0]  # [batch_size, hidden_dim]
        
        # 분류
        logits = self.classifier(cls_features)
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """손실 계산"""
        if self.config.num_classes == 2:
            # 이진 분류: logits을 스칼라로 변환
            logits = logits[:, 1] - logits[:, 0]  # AD - HC
            return self.criterion(logits, labels.float())
        else:
            return self.criterion(logits, labels)

class TextOnlyModel(nn.Module):
    """
    Text-only (Gemma Encoder, multilingual joint)
    전사 → Gemma/XLM-R([CLS]) → FC(sigmoid)
    """
    
    def __init__(self, config: TextOnlyConfig):
        super().__init__()
        self.config = config
        
        # 텍스트 인코더
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
        
        # 분류기
        layers = []
        input_dim = config.text_feature_dim
        
        for hidden_dim in config.classifier_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, config.num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # 손실 함수
        if config.loss_type == "focal":
            self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # 텍스트 인코딩
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if self.config.use_cls_token:
            # [CLS] 토큰 사용
            text_features = text_outputs.last_hidden_state[:, 0]  # [batch_size, hidden_dim]
        else:
            # 평균 풀링
            text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        # 분류
        logits = self.classifier(text_features)
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """손실 계산"""
        if self.config.num_classes == 2:
            # 이진 분류: logits을 스칼라로 변환
            logits = logits[:, 1] - logits[:, 0]  # AD - HC
            return self.criterion(logits, labels.float())
        else:
            return self.criterion(logits, labels)

class ConcatModel(nn.Module):
    """
    Concat (ViT + XLM-R) Late Fusion
    두 임베딩 late fusion(concat) → 2층 FFN(sigmoid)
    """
    
    def __init__(self, config: ConcatConfig):
        super().__init__()
        self.config = config
        
        # 오디오 인코더 (ViT)
        self.audio_encoder = ViTModel.from_pretrained(config.audio_encoder)
        
        # 텍스트 인코더 (XLM-R 또는 Gemma)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
        
        # Late Fusion: 2층 FFN
        layers = []
        input_dim = config.fused_feature_dim
        
        for hidden_dim in config.ffn_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, config.num_classes))
        self.fusion_classifier = nn.Sequential(*layers)
        
        # 손실 함수
        if config.loss_type == "focal":
            self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, 
                pixel_values: torch.Tensor, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, 3, 224, 224] - 스펙트로그램 이미지
            input_ids: [batch_size, seq_len] - 토큰화된 텍스트
            attention_mask: [batch_size, seq_len] - 어텐션 마스크
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # 오디오 인코딩 (ViT)
        audio_outputs = self.audio_encoder(pixel_values=pixel_values)
        audio_features = audio_outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # 텍스트 인코딩 (XLM-R/Gemma)
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if self.config.use_cls_token:
            text_features = text_outputs.last_hidden_state[:, 0]  # [CLS] token
        else:
            text_features = text_outputs.last_hidden_state.mean(dim=1)  # 평균 풀링
        
        # Late Fusion: Concatenation
        if self.config.fusion_method == "concat":
            fused_features = torch.cat([audio_features, text_features], dim=-1)
        elif self.config.fusion_method == "add":
            # 차원이 같을 때만 가능
            fused_features = audio_features + text_features
        else:
            # 기본값: concat
            fused_features = torch.cat([audio_features, text_features], dim=-1)
        
        # 2층 FFN으로 분류
        logits = self.fusion_classifier(fused_features)
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """손실 계산"""
        if self.config.num_classes == 2:
            # 이진 분류: logits을 스칼라로 변환
            logits = logits[:, 1] - logits[:, 0]  # AD - HC
            return self.criterion(logits, labels.float())
        else:
            return self.criterion(logits, labels)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """평가 지표 계산"""
    
    # 기본 분류 지표
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # AUC 계산
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    # 정확도
    accuracy = (y_true == y_pred).mean()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': f1,
        'auc': auc
    }

def compute_language_specific_metrics(
    y_true: List[int], 
    y_pred: List[int], 
    y_prob: List[float], 
    languages: List[str]
) -> Dict[str, Dict[str, float]]:
    """언어별 성능 지표 계산"""
    
    language_metrics = {}
    unique_languages = list(set(languages))
    
    for lang in unique_languages:
        # 해당 언어의 인덱스 찾기
        lang_indices = [i for i, l in enumerate(languages) if l == lang]
        
        if len(lang_indices) > 0:
            lang_y_true = [y_true[i] for i in lang_indices]
            lang_y_pred = [y_pred[i] for i in lang_indices]
            lang_y_prob = [y_prob[i] for i in lang_indices]
            
            # 해당 언어의 지표 계산
            lang_metrics = compute_metrics(
                np.array(lang_y_true),
                np.array(lang_y_pred),
                np.array(lang_y_prob)
            )
            
            language_metrics[lang] = lang_metrics
    
    return language_metrics
