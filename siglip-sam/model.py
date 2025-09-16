"""
SigLIP-SAM 치매 진단 모델
순수 PyTorch 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import numpy as np
from typing import Dict, Optional, Tuple
from sam_optimizer import SAM

# Lion Optimizer 라이브러리 임포트
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
    print("🦁 lion-pytorch 라이브러리 로드 성공")
except ImportError:
    LION_AVAILABLE = False
    print("⚠️ lion-pytorch 라이브러리를 찾을 수 없습니다. pip install lion-pytorch를 실행하세요.")

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

class SigLIPSAMDementiaClassifier(nn.Module):
    """
    SigLIP2 기반 치매 진단 분류기 (SAM 옵티마이저 지원)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # SigLIP2 모델 로드
        print(f"🔄 SigLIP2 모델 로드 중: {config.model_name}")
        self.siglip = AutoModel.from_pretrained(config.model_name)
        
        # 모델 설정 확인
        model_config = self.siglip.config
        if hasattr(model_config, 'projection_dim'):
            self.hidden_size = model_config.projection_dim
        elif hasattr(model_config, 'hidden_size'):
            self.hidden_size = model_config.hidden_size
        elif hasattr(model_config, 'vision_config') and hasattr(model_config.vision_config, 'hidden_size'):
            self.hidden_size = model_config.vision_config.hidden_size
        else:
            self.hidden_size = 768  # 기본값
        
        print(f"📐 Hidden size: {self.hidden_size}")
        
        # 언어 임베딩 (선택적)
        self.language_embedding = nn.Embedding(10, 512)  # 최대 10개 언어 지원
        self.language_projection = nn.Linear(512, self.hidden_size)
        
        # 분류기는 동적으로 생성
        self.classifier = None
        
        # 언어 ID 매핑
        self.language_to_id = {
            'English': 0, 'Greek': 1, 'Korean': 2, 'Spanish': 3, 'French': 4,
            'German': 5, 'Italian': 6, 'Portuguese': 7, 'Japanese': 8, 'Chinese': 9,
            'Mandarin': 9  # Chinese와 동일하게 처리
        }
        
        # 손실 함수 초기화
        if config.loss_type == "focal":
            self.criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
            print(f"🎯 Focal Loss 사용: alpha={config.focal_alpha}, gamma={config.focal_gamma}")
        elif config.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
            print("⚖️ BCE Loss 사용")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("📊 Cross Entropy Loss 사용")
    
    def forward(self, batch):
        """
        순전파
        """
        # SigLIP2 모델 통과
        outputs = self.siglip(
            input_ids=batch.get('input_ids'),
            pixel_values=batch.get('pixel_values'),
            attention_mask=batch.get('attention_mask'),
            pixel_attention_mask=batch.get('pixel_attention_mask'),
            spatial_shapes=batch.get('spatial_shapes'),
            return_dict=True
        )
        
        # 멀티모달 임베딩 생성 (고정 차원)
        multimodal_embeddings = (outputs.image_embeds + outputs.text_embeds) / 2
        
        # 언어 임베딩 추가 (선택적)
        if 'languages' in batch:
            language_ids = torch.tensor([
                self.language_to_id.get(lang, 0) for lang in batch['languages']
            ], device=multimodal_embeddings.device)
            
            lang_embeds = self.language_embedding(language_ids)
            lang_embeds = self.language_projection(lang_embeds)
            
            # 언어 임베딩과 멀티모달 임베딩 결합
            multimodal_embeddings = multimodal_embeddings + lang_embeds
        
        # 분류기 동적 생성 (첫 번째 호출 시에만)
        if self.classifier is None:
            input_dim = multimodal_embeddings.size(-1)
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 2)  # 이진 분류
            ).to(multimodal_embeddings.device)
            print(f"🏗️ 분류기 생성: {input_dim} -> 512 -> 2")
        
        # 분류
        logits = self.classifier(multimodal_embeddings)
        
        return logits
    
    def compute_loss(self, logits, labels):
        """
        손실 계산
        """
        if self.config.loss_type == "bce":
            # BCE는 이진 분류용이므로 라벨을 float으로 변환하고 로짓의 두 번째 클래스만 사용
            labels_bce = labels.float()
            logits_bce = logits[:, 1]  # 치매 클래스 확률만 사용
            loss = self.criterion(logits_bce, labels_bce)
        else:
            loss = self.criterion(logits, labels)
        
        return loss
    
    def create_optimizer(self, config):
        """
        옵티마이저 생성
        """
        # 가중치 감쇠를 적용하지 않을 파라미터들
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 파라미터 그룹 생성
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        # 옵티마이저 선택
        if config.optimizer_type == "lion":
            if not LION_AVAILABLE:
                print("⚠️ lion-pytorch 라이브러리가 없습니다. AdamW로 대체합니다.")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
                print(f"⚡ AdamW Optimizer 사용 (Lion 대체): lr={config.learning_rate}")
            else:
                optimizer = Lion(
                    optimizer_grouped_parameters,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
                print(f"🦁 Lion Optimizer 사용 (lion-pytorch): lr={config.learning_rate}")
        elif config.optimizer_type == "sam":
            optimizer = SAM(
                self.parameters(),
                torch.optim.AdamW,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                rho=config.sam_rho,
                adaptive=config.sam_adaptive
            )
            print(f"🎯 SAM Optimizer 사용: lr={config.learning_rate}, rho={config.sam_rho}, adaptive={config.sam_adaptive}")
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            print(f"⚡ AdamW Optimizer 사용: lr={config.learning_rate}")
        
        return optimizer
    
    def create_scheduler(self, optimizer, config, total_steps):
        """
        스케줄러 생성
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )
        
        return scheduler
