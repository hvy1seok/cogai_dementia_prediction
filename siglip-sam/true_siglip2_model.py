"""
진정한 SigLIP2 모델 구현
- EMA Teacher-Student 구조
- Self-Distillation (SILC/TIPS Loss)
- Auto-Regressive Decoder
- LoCa Loss (Captioning + Dense Captioning + Referring Expressions)
- Multi-Loss 통합 (SILC/TIPS 20% + Sigmoid 100% + LoCa 100%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import numpy as np
from typing import Dict, Optional, Tuple, List
import copy
import math
from sam_optimizer import SAM

class EMATeacher(nn.Module):
    """
    EMA (Exponential Moving Average) Teacher Model
    - Student 모델의 파라미터를 EMA로 업데이트
    - Stop gradient로 teacher 파라미터 고정
    - Self-distillation target 제공
    """
    def __init__(self, student_model, momentum: float = 0.999):
        super(EMATeacher, self).__init__()
        self.momentum = momentum
        
        # Student 모델을 복사하여 Teacher 생성
        self.teacher = copy.deepcopy(student_model)
        
        # Teacher 파라미터는 gradient 계산 안함
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        print(f"🧑‍🏫 EMA Teacher 초기화: momentum={momentum}")
    
    def update(self, student_model):
        """Student 모델로부터 EMA 업데이트"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), student_model.parameters()
            ):
                teacher_param.data = (
                    self.momentum * teacher_param.data + 
                    (1 - self.momentum) * student_param.data
                )
    
    def forward(self, *args, **kwargs):
        """Teacher forward pass (no gradient)"""
        with torch.no_grad():
            return self.teacher(*args, **kwargs)

class SILCTIPSLoss(nn.Module):
    """
    SILC/TIPS Loss - Self-Distillation + Masked Prediction
    - Student가 Teacher의 출력을 모방하도록 학습
    - Masked input에 대한 예측 loss
    - Feature alignment between student and teacher
    """
    def __init__(self, mask_ratio: float = 0.15, temperature: float = 0.07):
        super(SILCTIPSLoss, self).__init__()
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def create_random_mask(self, batch_size: int, seq_len: int, device: torch.device):
        """랜덤 마스크 생성"""
        mask = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio
        return mask
    
    def forward(self, 
                student_features: torch.Tensor,
                teacher_features: torch.Tensor,
                student_logits: torch.Tensor = None,
                teacher_logits: torch.Tensor = None,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        # Feature alignment loss (MSE)
        feature_loss = self.mse_loss(student_features, teacher_features.detach())
        
        # Logits alignment loss (KL divergence)
        logits_loss = 0.0
        if student_logits is not None and teacher_logits is not None:
            student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits.detach() / self.temperature, dim=-1)
            logits_loss = self.kl_div(student_probs, teacher_probs)
        
        # Masked prediction loss (if mask provided)
        masked_loss = 0.0
        if mask is not None:
            masked_student = student_features * mask.unsqueeze(-1)
            masked_teacher = teacher_features.detach() * mask.unsqueeze(-1)
            masked_loss = self.mse_loss(masked_student, masked_teacher)
        
        total_loss = feature_loss + logits_loss + masked_loss
        
        return {
            'silc_tips_loss': total_loss,
            'feature_loss': feature_loss,
            'logits_loss': logits_loss,
            'masked_loss': masked_loss
        }

class AutoRegressiveDecoder(nn.Module):
    """
    Auto-Regressive Decoder for Caption Generation
    - Cross-attention with image features
    - Sequential text generation
    - Dense captioning capability
    """
    def __init__(self, 
                 image_dim: int = 768,
                 text_dim: int = 768,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 vocab_size: int = 30522,
                 max_length: int = 77):
        super(AutoRegressiveDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Image feature projection
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                image_features: torch.Tensor,
                text_ids: torch.Tensor = None,
                max_length: int = None) -> torch.Tensor:
        
        batch_size = image_features.shape[0]
        device = image_features.device
        max_len = max_length or self.max_length
        
        # Image features as memory for cross-attention
        image_memory = self.image_proj(image_features)  # [batch, seq_len, hidden_dim]
        
        if text_ids is not None:
            # Training mode: use provided text
            seq_len = text_ids.shape[1]
            
            # Text embeddings + positional encoding
            text_embeds = self.text_embedding(text_ids)
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_embeds = self.pos_embedding(pos_ids)
            
            text_embeds = text_embeds + pos_embeds
            text_embeds = self.layer_norm(text_embeds)
            
            # Create causal mask for auto-regressive generation
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            
            # Transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=text_embeds,
                memory=image_memory,
                tgt_mask=causal_mask
            )
            
            # Output logits
            logits = self.output_proj(decoder_output)
            return logits
            
        else:
            # Inference mode: auto-regressive generation
            # Start with [CLS] token (assuming token_id = 101)
            generated_ids = torch.full((batch_size, 1), 101, device=device, dtype=torch.long)
            
            for step in range(max_len - 1):
                # Current sequence embeddings
                current_len = generated_ids.shape[1]
                text_embeds = self.text_embedding(generated_ids)
                pos_ids = torch.arange(current_len, device=device).unsqueeze(0).expand(batch_size, -1)
                pos_embeds = self.pos_embedding(pos_ids)
                
                text_embeds = text_embeds + pos_embeds
                text_embeds = self.layer_norm(text_embeds)
                
                # Causal mask
                causal_mask = torch.triu(torch.ones(current_len, current_len, device=device), diagonal=1).bool()
                
                # Decoder forward
                decoder_output = self.transformer_decoder(
                    tgt=text_embeds,
                    memory=image_memory,
                    tgt_mask=causal_mask
                )
                
                # Get next token logits
                next_token_logits = self.output_proj(decoder_output[:, -1, :])
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)
                
                # Stop if all sequences generated [SEP] token (assuming token_id = 102)
                if torch.all(next_token_ids.squeeze() == 102):
                    break
            
            return generated_ids

class LoCaLoss(nn.Module):
    """
    LoCa Loss - Localization + Captioning Loss
    - Standard captioning loss
    - Dense captioning loss (multiple regions)
    - Referring expressions loss
    """
    def __init__(self, vocab_size: int = 30522, ignore_index: int = -100):
        super(LoCaLoss, self).__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self,
                caption_logits: torch.Tensor,
                caption_targets: torch.Tensor,
                dense_caption_logits: torch.Tensor = None,
                dense_caption_targets: torch.Tensor = None,
                referring_logits: torch.Tensor = None,
                referring_targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        # Standard captioning loss
        caption_loss = self.cross_entropy(
            caption_logits.reshape(-1, self.vocab_size),
            caption_targets.reshape(-1)
        )
        
        # Dense captioning loss
        dense_caption_loss = 0.0
        if dense_caption_logits is not None and dense_caption_targets is not None:
            dense_caption_loss = self.cross_entropy(
                dense_caption_logits.reshape(-1, self.vocab_size),
                dense_caption_targets.reshape(-1)
            )
        
        # Referring expressions loss
        referring_loss = 0.0
        if referring_logits is not None and referring_targets is not None:
            referring_loss = self.cross_entropy(
                referring_logits.reshape(-1, self.vocab_size),
                referring_targets.reshape(-1)
            )
        
        total_loss = caption_loss + dense_caption_loss + referring_loss
        
        return {
            'loca_loss': total_loss,
            'caption_loss': caption_loss,
            'dense_caption_loss': dense_caption_loss,
            'referring_loss': referring_loss
        }

class TrueSigLIP2DementiaClassifier(nn.Module):
    """
    진정한 SigLIP2 치매 진단 분류기
    - EMA Teacher-Student 구조
    - Self-Distillation (SILC/TIPS Loss)  
    - Auto-Regressive Decoder
    - Multi-Loss: SILC/TIPS (20%) + Sigmoid (100%) + LoCa (100%)
    """
    def __init__(self, config):
        super(TrueSigLIP2DementiaClassifier, self).__init__()
        self.config = config
        
        print("🔥 진정한 SigLIP2 모델 초기화 시작...")
        
        # Base SigLIP2 모델 (Student)
        self.siglip_student = AutoModel.from_pretrained(config.model_name)
        
        # 모델 설정 확인
        model_config = self.siglip_student.config
        if hasattr(model_config, 'projection_dim'):
            self.hidden_size = model_config.projection_dim
        elif hasattr(model_config, 'hidden_size'):
            self.hidden_size = model_config.hidden_size
        elif hasattr(model_config, 'vision_config') and hasattr(model_config.vision_config, 'hidden_size'):
            self.hidden_size = model_config.vision_config.hidden_size
        else:
            self.hidden_size = 768
        
        print(f"📐 Hidden size: {self.hidden_size}")
        
        # EMA Teacher 초기화
        self.ema_teacher = EMATeacher(
            student_model=self.siglip_student,
            momentum=getattr(config, 'ema_momentum', 0.999)
        )
        
        # Auto-Regressive Decoder
        self.ar_decoder = AutoRegressiveDecoder(
            image_dim=self.hidden_size,
            text_dim=self.hidden_size,
            hidden_dim=getattr(config, 'decoder_hidden_dim', 512),
            num_heads=getattr(config, 'decoder_num_heads', 8),
            num_layers=getattr(config, 'decoder_num_layers', 6),
            vocab_size=getattr(config, 'vocab_size', 30522),
            max_length=getattr(config, 'max_caption_length', 77)
        )
        
        # 분류기 (기존 치매 진단용)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)  # 이진 분류
        )
        
        # Loss 함수들
        self.silc_tips_loss = SILCTIPSLoss(
            mask_ratio=getattr(config, 'mask_ratio', 0.15),
            temperature=getattr(config, 'silc_temperature', 0.07)
        )
        
        self.sigmoid_loss = SigLIP2ContrastiveLoss(
            temperature=getattr(config, 'contrastive_temperature', 0.07),
            use_sigmoid=True
        )
        
        self.loca_loss = LoCaLoss(
            vocab_size=getattr(config, 'vocab_size', 30522)
        )
        
        # Classification loss (기존)
        self.classification_criterion = None  # setup_loss_function에서 설정
        
        # Loss 가중치
        self.silc_weight = getattr(config, 'silc_weight', 0.2)  # 20%
        self.sigmoid_weight = getattr(config, 'sigmoid_weight', 1.0)  # 100%
        self.loca_weight = getattr(config, 'loca_weight', 1.0)  # 100%
        self.classification_weight = getattr(config, 'classification_weight', 1.0)  # 100%
        
        print(f"🔗 True SigLIP2 Loss 가중치:")
        print(f"   SILC/TIPS: {self.silc_weight} (20%)")
        print(f"   Sigmoid: {self.sigmoid_weight} (100%)")
        print(f"   LoCa: {self.loca_weight} (100%)")
        print(f"   Classification: {self.classification_weight} (100%)")
        
        # 언어 임베딩 (선택적)
        self.language_embedding = nn.Embedding(10, 512)
        self.language_projection = nn.Linear(512, self.hidden_size)
        
        # 언어 ID 매핑
        self.language_to_id = {
            'English': 0, 'Greek': 1, 'Korean': 2, 'Spanish': 3, 'French': 4,
            'German': 5, 'Italian': 6, 'Portuguese': 7, 'Japanese': 8, 'Chinese': 9,
            'Mandarin': 9
        }
        
        print("✅ 진정한 SigLIP2 모델 초기화 완료!")
        print("🧑‍🏫 EMA Teacher-Student 구조 활성화")
        print("🎯 Self-Distillation + Caption Generation + Contrastive Learning")
    
    def setup_loss_function(self, class_weights=None):
        """Classification loss 함수 초기화"""
        from model import FocalLoss  # 기존 FocalLoss 사용
        
        if self.config.loss_type == "focal":
            if class_weights is not None and self.config.auto_class_weights:
                alpha = class_weights
                print(f"🎯 Focal Loss 사용: alpha={alpha} (자동 계산), gamma={self.config.focal_gamma}")
            else:
                alpha = self.config.focal_alpha
                print(f"🎯 Focal Loss 사용: alpha={alpha} (수동 설정), gamma={self.config.focal_gamma}")
            
            self.classification_criterion = FocalLoss(alpha=alpha, gamma=self.config.focal_gamma)
            
        elif self.config.loss_type == "bce":
            self.classification_criterion = nn.BCEWithLogitsLoss()
            print("⚖️ BCE Loss 사용")
        else:
            if class_weights is not None and self.config.auto_class_weights:
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
                self.classification_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
                print(f"📊 Cross Entropy Loss 사용: 클래스 가중치 {class_weights}")
            else:
                self.classification_criterion = nn.CrossEntropyLoss()
                print("📊 Cross Entropy Loss 사용")
    
    def forward(self, 
                batch,
                return_embeddings: bool = False,
                generate_captions: bool = False,
                caption_targets: torch.Tensor = None,
                training: bool = True):
        """
        진정한 SigLIP2 Forward Pass
        - Student와 Teacher 모두 forward
        - Auto-regressive caption generation
        - Multi-modal feature extraction
        """
        # Student forward pass
        student_outputs = self.siglip_student(
            input_ids=batch.get('input_ids'),
            pixel_values=batch.get('pixel_values'),
            attention_mask=batch.get('attention_mask'),
            pixel_attention_mask=batch.get('pixel_attention_mask'),
            spatial_shapes=batch.get('spatial_shapes'),
            return_dict=True
        )
        
        # Teacher forward pass (no gradient)
        teacher_outputs = self.ema_teacher(
            input_ids=batch.get('input_ids'),
            pixel_values=batch.get('pixel_values'),
            attention_mask=batch.get('attention_mask'),
            pixel_attention_mask=batch.get('pixel_attention_mask'),
            spatial_shapes=batch.get('spatial_shapes'),
            return_dict=True
        )
        
        # 개별 임베딩 추출
        student_audio_embeds = student_outputs.image_embeds  # 멜스펙토그램을 이미지로 처리
        student_text_embeds = student_outputs.text_embeds
        
        teacher_audio_embeds = teacher_outputs.image_embeds
        teacher_text_embeds = teacher_outputs.text_embeds
        
        # 멀티모달 임베딩 생성 (분류용)
        student_multimodal = (student_audio_embeds + student_text_embeds) / 2
        teacher_multimodal = (teacher_audio_embeds + teacher_text_embeds) / 2
        
        # 언어 임베딩 추가 (선택적)
        if 'languages' in batch:
            language_ids = torch.tensor([
                self.language_to_id.get(lang, 0) for lang in batch['languages']
            ], device=student_multimodal.device)
            
            lang_embeds = self.language_embedding(language_ids)
            lang_embeds = self.language_projection(lang_embeds)
            
            student_multimodal = student_multimodal + lang_embeds
        
        # 분류 logits
        classification_logits = self.classifier(student_multimodal)
        
        # Caption generation (if requested)
        caption_logits = None
        if generate_captions or caption_targets is not None:
            caption_logits = self.ar_decoder(
                image_features=student_audio_embeds,
                text_ids=caption_targets if training else None
            )
        
        # Return 구성
        outputs = {
            'classification_logits': classification_logits,
            'student_audio_embeds': student_audio_embeds,
            'student_text_embeds': student_text_embeds,
            'student_multimodal_embeds': student_multimodal,
            'teacher_audio_embeds': teacher_audio_embeds,
            'teacher_text_embeds': teacher_text_embeds,
            'teacher_multimodal_embeds': teacher_multimodal,
            'caption_logits': caption_logits
        }
        
        if return_embeddings:
            return outputs
        else:
            return classification_logits
    
    def compute_loss(self, model_outputs, labels, patient_ids=None, caption_targets=None):
        """
        진정한 SigLIP2 Multi-Loss 계산
        - SILC/TIPS Loss (20%)
        - Sigmoid Contrastive Loss (100%)
        - LoCa Loss (100%)
        - Classification Loss (100%)
        """
        total_loss = 0.0
        loss_components = {}
        
        # 1. Classification Loss
        classification_loss = 0.0
        if self.classification_criterion is not None:
            if self.config.loss_type == "bce":
                labels_bce = labels.float()
                logits_bce = model_outputs['classification_logits'][:, 1]
                classification_loss = self.classification_criterion(logits_bce, labels_bce)
            else:
                classification_loss = self.classification_criterion(
                    model_outputs['classification_logits'], labels
                )
        
        total_loss += self.classification_weight * classification_loss
        loss_components['classification_loss'] = classification_loss
        
        # 2. SILC/TIPS Loss (Self-Distillation)
        silc_loss_dict = self.silc_tips_loss(
            student_features=model_outputs['student_multimodal_embeds'],
            teacher_features=model_outputs['teacher_multimodal_embeds'],
            student_logits=model_outputs['classification_logits'],
            teacher_logits=None  # Teacher도 classification head 필요시 추가
        )
        
        silc_loss = silc_loss_dict['silc_tips_loss']
        total_loss += self.silc_weight * silc_loss
        loss_components.update({f'silc_{k}': v for k, v in silc_loss_dict.items()})
        
        # 3. Sigmoid Contrastive Loss
        sigmoid_loss_dict = {}
        if patient_ids is not None:
            sigmoid_loss_dict = self.sigmoid_loss(
                audio_embeds=model_outputs['student_audio_embeds'],
                text_embeds=model_outputs['student_text_embeds'],
                patient_ids=patient_ids
            )
            
            sigmoid_loss = sigmoid_loss_dict['contrastive_loss']
            total_loss += self.sigmoid_weight * sigmoid_loss
            loss_components.update({f'sigmoid_{k}': v for k, v in sigmoid_loss_dict.items()})
        
        # 4. LoCa Loss (Caption Generation)
        loca_loss_dict = {}
        if (model_outputs['caption_logits'] is not None and 
            caption_targets is not None):
            
            loca_loss_dict = self.loca_loss(
                caption_logits=model_outputs['caption_logits'],
                caption_targets=caption_targets
            )
            
            loca_loss = loca_loss_dict['loca_loss']
            total_loss += self.loca_weight * loca_loss
            loss_components.update({f'loca_{k}': v for k, v in loca_loss_dict.items()})
        
        return {
            'total_loss': total_loss,
            'loss_components': loss_components
        }
    
    def update_teacher(self):
        """EMA Teacher 업데이트"""
        self.ema_teacher.update(self.siglip_student)
    
    def create_optimizer(self, config):
        """옵티마이저 생성 (기존 코드 재사용)"""
        # 가중치 감쇠를 적용하지 않을 파라미터들
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 파라미터 그룹 생성
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]
        
        # 옵티마이저 선택
        if config.optimizer_type == "sam":
            optimizer = SAM(
                self.parameters(),
                torch.optim.AdamW,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                rho=config.sam_rho,
                adaptive=config.sam_adaptive
            )
            print(f"🎯 SAM Optimizer 사용: lr={config.learning_rate}, rho={config.sam_rho}")
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            print(f"⚡ AdamW Optimizer 사용: lr={config.learning_rate}")
        
        return optimizer
    
    def create_scheduler(self, optimizer, config, total_steps):
        """스케줄러 생성"""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )
        return scheduler


# SigLIP2ContrastiveLoss는 기존 것 재사용
class SigLIP2ContrastiveLoss(nn.Module):
    """기존 SigLIP2 Contrastive Loss (재사용)"""
    def __init__(self, temperature: float = 0.07, use_sigmoid: bool = True):
        super(SigLIP2ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.use_sigmoid = use_sigmoid
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, audio_embeds: torch.Tensor, text_embeds: torch.Tensor, 
                patient_ids: List[str]) -> Dict[str, torch.Tensor]:
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device
        
        # L2 정규화
        audio_embeds = F.normalize(audio_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(audio_embeds, text_embeds.T) / self.temperature
        
        # Positive pairs 마스크 생성
        positive_mask = torch.zeros(batch_size, batch_size, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                if patient_ids[i] == patient_ids[j]:
                    positive_mask[i, j] = 1.0
        
        # Negative pairs 마스크
        negative_mask = 1.0 - positive_mask
        
        if self.use_sigmoid:
            # SigLIP 스타일: Sigmoid matching
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
            a2t_loss = self.cross_entropy(similarity_matrix, torch.arange(batch_size, device=device))
            t2a_loss = self.cross_entropy(similarity_matrix.T, torch.arange(batch_size, device=device))
            contrastive_loss = (a2t_loss + t2a_loss) / 2
        
        # 정렬 메트릭 계산
        with torch.no_grad():
            positive_similarities = similarity_matrix * positive_mask
            positive_count = positive_mask.sum()
            avg_positive_sim = positive_similarities.sum() / (positive_count + 1e-8)
            
            negative_similarities = similarity_matrix * negative_mask
            negative_count = negative_mask.sum()
            avg_negative_sim = negative_similarities.sum() / (negative_count + 1e-8)
            
            alignment_score = avg_positive_sim - avg_negative_sim
        
        return {
            'contrastive_loss': contrastive_loss,
            'avg_positive_similarity': avg_positive_sim,
            'avg_negative_similarity': avg_negative_sim,
            'alignment_score': alignment_score,
            'positive_pairs_count': positive_count,
            'negative_pairs_count': negative_count
        }
