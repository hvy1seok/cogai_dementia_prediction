"""
SigLIP v2 커스텀 구현 - Language Agnostic 치매 진단을 위한
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, SiglipConfig

class SigLIP2CustomModel(nn.Module):
    """
    SigLIP v2 커스텀 구현
    - 개선된 Sigmoid Loss
    - 언어 무관 멀티모달 학습
    - 다국어 지원
    """
    
    def __init__(self, base_model_name: str = "google/siglip-base-patch16-224", num_languages: int = 10):
        super().__init__()
        
        # Base SigLIP 모델 로드
        self.siglip_base = SiglipModel.from_pretrained(base_model_name)
        
        # 언어별 임베딩
        self.language_embeddings = nn.Embedding(num_languages, 768)
        
        # 개선된 Cross-modal Attention
        self.cross_attention = nn.MultiheadAttention(768, 12, dropout=0.1)
        
        # Language Agnostic Feature Extractor
        self.language_agnostic_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 학습 가능한 바이어스 (SigLIP v2 특징)
        self.learnable_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, images, texts, language_ids=None):
        # Base SigLIP 특징 추출
        outputs = self.siglip_base(pixel_values=images, input_ids=texts)
        
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        # 언어별 임베딩 추가
        if language_ids is not None:
            lang_embeds = self.language_embeddings(language_ids)
            text_features = text_features + lang_embeds
        
        # Cross-modal Attention
        attended_features, _ = self.cross_attention(
            text_features.unsqueeze(0), 
            image_features.unsqueeze(0), 
            image_features.unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)
        
        # Language Agnostic 특징 추출
        agnostic_features = self.language_agnostic_layer(attended_features)
        
        return {
            'image_features': image_features,
            'text_features': text_features,
            'agnostic_features': agnostic_features,
            'bias': self.learnable_bias
        }

class ImprovedSigmoidLoss(nn.Module):
    """SigLIP v2의 개선된 Sigmoid Loss"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features, bias=None):
        # 정규화
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 유사도 계산
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 학습 가능한 바이어스 추가 (SigLIP v2 특징)
        if bias is not None:
            logits = logits + bias
        
        # Sigmoid Loss
        labels = torch.eye(logits.size(0), device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return loss
