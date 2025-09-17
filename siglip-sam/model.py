"""
SigLIP-SAM 치매 진단 모델
순수 PyTorch 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import numpy as np
from typing import Dict, Optional, Tuple, List
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
    alpha가 리스트/텐서인 경우 클래스별 가중치 적용
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            # 텐서를 모듈 파라미터로 등록하여 자동으로 디바이스 이동
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
            # alpha가 텐서인 경우 클래스별 가중치 적용
            # 명시적으로 디바이스를 맞춰줌 (안전성을 위해)
            alpha_tensor = self.alpha.to(targets.device)
            alpha_t = alpha_tensor.gather(0, targets.long())
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

class SigLIP2ContrastiveLoss(nn.Module):
    """
    SigLIP2 스타일 Contrastive Loss 구현
    - Sigmoid matching (CLIP의 softmax 대신)
    - Same-patient audio-text pairs를 positive로 처리
    - Cross-modal representation alignment
    """
    def __init__(self, temperature: float = 0.07, use_sigmoid: bool = True):
        super(SigLIP2ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.use_sigmoid = use_sigmoid
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, audio_embeds: torch.Tensor, text_embeds: torch.Tensor, 
                patient_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_embeds: [batch_size, embed_dim] 오디오 임베딩
            text_embeds: [batch_size, embed_dim] 텍스트 임베딩  
            patient_ids: [batch_size] 환자 ID 리스트
        
        Returns:
            Dict containing contrastive loss and alignment metrics
        """
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device
        
        # L2 정규화
        audio_embeds = F.normalize(audio_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        
        # 유사도 행렬 계산: [batch_size, batch_size]
        similarity_matrix = torch.matmul(audio_embeds, text_embeds.T) / self.temperature
        
        # Positive pairs 마스크 생성 (같은 환자)
        positive_mask = torch.zeros(batch_size, batch_size, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                if patient_ids[i] == patient_ids[j]:
                    positive_mask[i, j] = 1.0
        
        # Negative pairs 마스크 (다른 환자)
        negative_mask = 1.0 - positive_mask
        
        if self.use_sigmoid:
            # SigLIP 스타일: Sigmoid matching
            # Positive pairs는 1에 가깝게, negative pairs는 0에 가깝게
            positive_loss = self.bce_with_logits(
                similarity_matrix * positive_mask, 
                positive_mask
            )
            
            negative_loss = self.bce_with_logits(
                similarity_matrix * negative_mask,
                torch.zeros_like(negative_mask)
            )
            
            contrastive_loss = (positive_loss + negative_loss) / 2
            
        else:
            # CLIP 스타일: Softmax contrastive
            # Audio-to-text direction
            a2t_loss = self.cross_entropy(similarity_matrix, torch.arange(batch_size, device=device))
            
            # Text-to-audio direction  
            t2a_loss = self.cross_entropy(similarity_matrix.T, torch.arange(batch_size, device=device))
            
            contrastive_loss = (a2t_loss + t2a_loss) / 2
        
        # 정렬 메트릭 계산
        with torch.no_grad():
            # Positive pairs 평균 유사도
            positive_similarities = similarity_matrix * positive_mask
            positive_count = positive_mask.sum()
            avg_positive_sim = positive_similarities.sum() / (positive_count + 1e-8)
            
            # Negative pairs 평균 유사도
            negative_similarities = similarity_matrix * negative_mask
            negative_count = negative_mask.sum()
            avg_negative_sim = negative_similarities.sum() / (negative_count + 1e-8)
            
            # 정렬 정도 (positive - negative)
            alignment_score = avg_positive_sim - avg_negative_sim
        
        return {
            'contrastive_loss': contrastive_loss,
            'avg_positive_similarity': avg_positive_sim,
            'avg_negative_similarity': avg_negative_sim,
            'alignment_score': alignment_score,
            'positive_pairs_count': positive_count,
            'negative_pairs_count': negative_count
        }

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
        
        # 손실 함수는 나중에 클래스 가중치와 함께 초기화
        self.config = config
        self.criterion = None  # 나중에 설정
        
        # SigLIP2 Contrastive Learning 설정
        self.use_contrastive = getattr(config, 'use_contrastive', True)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.5)
        self.contrastive_temperature = getattr(config, 'contrastive_temperature', 0.07)
        
        if self.use_contrastive:
            self.contrastive_loss = SigLIP2ContrastiveLoss(
                temperature=self.contrastive_temperature,
                use_sigmoid=True  # SigLIP2 스타일
            )
            print(f"🔗 SigLIP2 Contrastive Learning 활성화:")
            print(f"   가중치: {self.contrastive_weight}")
            print(f"   온도: {self.contrastive_temperature}")
        else:
            self.contrastive_loss = None
            print("⚪ Contrastive Learning 비활성화")
    
    def setup_loss_function(self, class_weights=None):
        """손실 함수 초기화 - 클래스 가중치 적용"""
        if self.config.loss_type == "focal":
            if class_weights is not None and self.config.auto_class_weights:
                # 클래스 가중치 자동 적용
                alpha = class_weights
                print(f"🎯 Focal Loss 사용: alpha={alpha} (자동 계산), gamma={self.config.focal_gamma}")
                print(f"   정상 클래스 가중치: {alpha[0]:.3f}, 치매 클래스 가중치: {alpha[1]:.3f}")
            else:
                # 수동 설정 또는 균등 가중치
                alpha = self.config.focal_alpha
                print(f"🎯 Focal Loss 사용: alpha={alpha} (수동 설정), gamma={self.config.focal_gamma}")
            
            self.criterion = FocalLoss(alpha=alpha, gamma=self.config.focal_gamma)
            
        elif self.config.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
            print("⚖️ BCE Loss 사용")
        else:
            if class_weights is not None and self.config.auto_class_weights:
                # CrossEntropy에도 클래스 가중치 적용
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
                self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
                print(f"📊 Cross Entropy Loss 사용: 클래스 가중치 {class_weights}")
            else:
                self.criterion = nn.CrossEntropyLoss()
                print("📊 Cross Entropy Loss 사용")
    
    def forward(self, batch, return_embeddings=False):
        """
        순전파 - SigLIP2 Contrastive Learning 포함
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
        
        # 개별 임베딩 추출 (contrastive learning용)
        audio_embeds = outputs.image_embeds  # 멜스펙토그램을 이미지로 처리
        text_embeds = outputs.text_embeds
        
        # 멀티모달 임베딩 생성 (분류용)
        multimodal_embeddings = (audio_embeds + text_embeds) / 2
        
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
        
        if return_embeddings:
            return {
                'logits': logits,
                'audio_embeds': audio_embeds,
                'text_embeds': text_embeds,
                'multimodal_embeds': multimodal_embeddings
            }
        
        return logits
    
    def compute_loss(self, model_outputs, labels, patient_ids=None):
        """
        통합 손실 계산 - Classification + Contrastive Learning
        """
        # 모델 출력 처리
        if isinstance(model_outputs, dict):
            logits = model_outputs['logits']
            audio_embeds = model_outputs.get('audio_embeds')
            text_embeds = model_outputs.get('text_embeds')
        else:
            logits = model_outputs
            audio_embeds = None
            text_embeds = None
        
        # 분류 손실 계산
        if self.config.loss_type == "bce":
            # BCE는 이진 분류용이므로 라벨을 float으로 변환하고 로짓의 두 번째 클래스만 사용
            labels_bce = labels.float()
            logits_bce = logits[:, 1]  # 치매 클래스 확률만 사용
            classification_loss = self.criterion(logits_bce, labels_bce)
        else:
            classification_loss = self.criterion(logits, labels)
        
        # Contrastive 손실 계산 (활성화된 경우)
        total_loss = classification_loss
        contrastive_metrics = {}
        
        if (self.use_contrastive and 
            audio_embeds is not None and 
            text_embeds is not None and 
            patient_ids is not None):
            
            contrastive_outputs = self.contrastive_loss(
                audio_embeds, text_embeds, patient_ids
            )
            
            contrastive_loss = contrastive_outputs['contrastive_loss']
            total_loss = (1 - self.contrastive_weight) * classification_loss + \
                        self.contrastive_weight * contrastive_loss
            
            # Contrastive 메트릭 저장
            contrastive_metrics = {
                'contrastive_loss': contrastive_loss.item(),
                'avg_positive_similarity': contrastive_outputs['avg_positive_similarity'].item(),
                'avg_negative_similarity': contrastive_outputs['avg_negative_similarity'].item(),
                'alignment_score': contrastive_outputs['alignment_score'].item(),
                'positive_pairs_count': contrastive_outputs['positive_pairs_count'].item(),
                'negative_pairs_count': contrastive_outputs['negative_pairs_count'].item()
            }
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'contrastive_metrics': contrastive_metrics
        }
    
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
