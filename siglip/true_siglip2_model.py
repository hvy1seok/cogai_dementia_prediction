"""
진정한 SigLIP2 모델 구현 (PyTorch Lightning 버전)
- EMA Teacher-Student 구조
- Self-Distillation (SILC/TIPS Loss)
- Auto-Regressive Decoder
- LoCa Loss (Captioning + Dense Captioning + Referring Expressions)
- Multi-Loss 통합 (SILC/TIPS 20% + Sigmoid 100% + LoCa 100%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import numpy as np
from typing import Dict, Optional, Tuple, List
import copy
import math
from torchmetrics import Accuracy

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
        image_memory = self.image_proj(image_features)
        
        if text_ids is not None:
            # Training mode: use provided text
            seq_len = text_ids.shape[1]
            
            # Text embeddings + positional encoding
            text_embeds = self.text_embedding(text_ids)
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_embeds = self.pos_embedding(pos_ids)
            
            text_embeds = text_embeds + pos_embeds
            text_embeds = self.layer_norm(text_embeds)
            
            # Create causal mask
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
            generated_ids = torch.full((batch_size, 1), 101, device=device, dtype=torch.long)
            
            for step in range(max_len - 1):
                current_len = generated_ids.shape[1]
                text_embeds = self.text_embedding(generated_ids)
                pos_ids = torch.arange(current_len, device=device).unsqueeze(0).expand(batch_size, -1)
                pos_embeds = self.pos_embedding(pos_ids)
                
                text_embeds = text_embeds + pos_embeds
                text_embeds = self.layer_norm(text_embeds)
                
                causal_mask = torch.triu(torch.ones(current_len, current_len, device=device), diagonal=1).bool()
                
                decoder_output = self.transformer_decoder(
                    tgt=text_embeds,
                    memory=image_memory,
                    tgt_mask=causal_mask
                )
                
                next_token_logits = self.output_proj(decoder_output[:, -1, :])
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)
                
                if torch.all(next_token_ids.squeeze() == 102):
                    break
            
            return generated_ids

class LoCaLoss(nn.Module):
    """
    LoCa Loss - Localization + Captioning Loss
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

class TrueSigLIP2DementiaClassifier(pl.LightningModule):
    """
    진정한 SigLIP2 치매 진단 분류기 (PyTorch Lightning 버전)
    - EMA Teacher-Student 구조
    - Self-Distillation (SILC/TIPS Loss)  
    - Auto-Regressive Decoder
    - Multi-Loss: SILC/TIPS (20%) + Sigmoid (100%) + LoCa (100%)
    """
    def __init__(self, 
                 model_name: str = "google/siglip2-base-patch16-naflex",
                 num_classes: int = 2,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 100,
                 max_epochs: int = 10,
                 use_language_embedding: bool = True,
                 loss_type: str = "cross_entropy",
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 optimizer_type: str = "adamw",
                 sam_rho: float = 0.05,
                 # True SigLIP2 전용 파라미터
                 ema_momentum: float = 0.999,
                 silc_weight: float = 0.2,
                 sigmoid_weight: float = 1.0,
                 loca_weight: float = 1.0,
                 classification_weight: float = 1.0,
                 mask_ratio: float = 0.15,
                 decoder_hidden_dim: int = 512,
                 decoder_num_heads: int = 8,
                 decoder_num_layers: int = 6,
                 vocab_size: int = 30522,
                 max_caption_length: int = 77):
        
        super().__init__()
        self.save_hyperparameters()
        
        print("🔥 진정한 SigLIP2 모델 초기화 시작...")
        
        # Base SigLIP2 모델 (Student)
        self.siglip_student = AutoModel.from_pretrained(model_name)
        
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
            momentum=ema_momentum
        )
        
        # Auto-Regressive Decoder
        self.ar_decoder = AutoRegressiveDecoder(
            image_dim=self.hidden_size,
            text_dim=self.hidden_size,
            hidden_dim=decoder_hidden_dim,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            vocab_size=vocab_size,
            max_length=max_caption_length
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Loss 함수들
        self.silc_tips_loss = SILCTIPSLoss(
            mask_ratio=mask_ratio,
            temperature=0.07
        )
        
        self.sigmoid_loss = SigLIP2ContrastiveLoss(
            temperature=0.07,
            use_sigmoid=True
        )
        
        self.loca_loss = LoCaLoss(vocab_size=vocab_size)
        
        # Classification loss (나중에 설정)
        self.criterion = None
        
        # Loss 가중치
        self.silc_weight = silc_weight
        self.sigmoid_weight = sigmoid_weight
        self.loca_weight = loca_weight
        self.classification_weight = classification_weight
        
        print(f"🔗 True SigLIP2 Loss 가중치:")
        print(f"   SILC/TIPS: {self.silc_weight}")
        print(f"   Sigmoid: {self.sigmoid_weight}")
        print(f"   LoCa: {self.loca_weight}")
        print(f"   Classification: {self.classification_weight}")
        
        # 언어 임베딩
        if use_language_embedding:
            self.language_embedding = nn.Embedding(10, 512)
            self.language_projection = nn.Linear(512, self.hidden_size)
        else:
            self.language_embedding = None
            self.language_projection = None
        
        # 언어 ID 매핑
        self.language_to_id = {
            'English': 0, 'Greek': 1, 'Korean': 2, 'Spanish': 3, 'French': 4,
            'German': 5, 'Italian': 6, 'Portuguese': 7, 'Japanese': 8, 'Chinese': 9,
            'Mandarin': 9
        }
        
        # 메트릭 초기화
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
        # Step outputs 저장용
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        print("✅ 진정한 SigLIP2 모델 초기화 완료!")
    
    def setup_loss_function(self, class_weights=None):
        """Classification loss 함수 초기화"""
        from model import FocalLoss  # 기존 FocalLoss 사용
        
        if self.hparams.loss_type == "focal":
            if class_weights is not None:
                alpha = class_weights
                print(f"🎯 Focal Loss 사용: alpha={alpha} (자동 계산), gamma={self.hparams.focal_gamma}")
            else:
                alpha = self.hparams.focal_alpha
                print(f"🎯 Focal Loss 사용: alpha={alpha} (수동 설정), gamma={self.hparams.focal_gamma}")
            
            self.criterion = FocalLoss(alpha=alpha, gamma=self.hparams.focal_gamma)
            
        elif self.hparams.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
            print("⚖️ BCE Loss 사용")
        else:
            if class_weights is not None:
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
                self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
                print(f"📊 Cross Entropy Loss 사용: 클래스 가중치 {class_weights}")
            else:
                self.criterion = nn.CrossEntropyLoss()
                print("📊 Cross Entropy Loss 사용")
    
    def forward(self, 
                input_ids, attention_mask=None, pixel_values=None, 
                pixel_attention_mask=None, spatial_shapes=None, 
                language_ids=None, return_embeddings=False,
                generate_captions=False, caption_targets=None, training=True):
        """진정한 SigLIP2 Forward Pass"""
        
        # Student forward pass
        student_outputs = self.siglip_student(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            return_dict=True
        )
        
        # Teacher forward pass (training 시에만)
        teacher_outputs = None
        if training:
            teacher_outputs = self.ema_teacher(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                return_dict=True
            )
        
        # 개별 임베딩 추출
        student_audio_embeds = student_outputs.image_embeds
        student_text_embeds = student_outputs.text_embeds
        
        teacher_audio_embeds = None
        teacher_text_embeds = None
        if teacher_outputs is not None:
            teacher_audio_embeds = teacher_outputs.image_embeds
            teacher_text_embeds = teacher_outputs.text_embeds
        
        # 멀티모달 임베딩 생성
        student_multimodal = (student_audio_embeds + student_text_embeds) / 2
        teacher_multimodal = None
        if teacher_outputs is not None:
            teacher_multimodal = (teacher_audio_embeds + teacher_text_embeds) / 2
        
        # 언어 임베딩 추가
        if language_ids is not None and self.language_embedding is not None:
            lang_embeds = self.language_embedding(language_ids)
            lang_embeds = self.language_projection(lang_embeds)
            student_multimodal = student_multimodal + lang_embeds
        
        # 분류 logits
        classification_logits = self.classifier(student_multimodal)
        
        # Caption generation
        caption_logits = None
        if generate_captions or caption_targets is not None:
            caption_logits = self.ar_decoder(
                image_features=student_audio_embeds,
                text_ids=caption_targets if training else None
            )
        
        if return_embeddings:
            return {
                'classification_logits': classification_logits,
                'student_audio_embeds': student_audio_embeds,
                'student_text_embeds': student_text_embeds,
                'student_multimodal_embeds': student_multimodal,
                'teacher_audio_embeds': teacher_audio_embeds,
                'teacher_text_embeds': teacher_text_embeds,
                'teacher_multimodal_embeds': teacher_multimodal,
                'caption_logits': caption_logits
            }
        else:
            return classification_logits
    
    def compute_loss(self, model_outputs, labels, patient_ids=None, caption_targets=None):
        """진정한 SigLIP2 Multi-Loss 계산"""
        total_loss = 0.0
        loss_components = {}
        
        # 1. Classification Loss
        classification_loss = 0.0
        if self.criterion is not None:
            if self.hparams.loss_type == "bce":
                labels_bce = labels.float()
                logits_bce = model_outputs['classification_logits'][:, 1]
                classification_loss = self.criterion(logits_bce, labels_bce)
            else:
                classification_loss = self.criterion(
                    model_outputs['classification_logits'], labels
                )
        
        total_loss += self.classification_weight * classification_loss
        loss_components['classification_loss'] = classification_loss
        
        # 2. SILC/TIPS Loss (Self-Distillation) - training 시에만
        if (model_outputs['teacher_multimodal_embeds'] is not None and
            model_outputs['student_multimodal_embeds'] is not None):
            
            silc_loss_dict = self.silc_tips_loss(
                student_features=model_outputs['student_multimodal_embeds'],
                teacher_features=model_outputs['teacher_multimodal_embeds'],
                student_logits=model_outputs['classification_logits'],
                teacher_logits=None
            )
            
            silc_loss = silc_loss_dict['silc_tips_loss']
            total_loss += self.silc_weight * silc_loss
            loss_components.update({f'silc_{k}': v for k, v in silc_loss_dict.items()})
        
        # 3. Sigmoid Contrastive Loss
        if (patient_ids is not None and 
            model_outputs['student_audio_embeds'] is not None and
            model_outputs['student_text_embeds'] is not None):
            
            sigmoid_loss_dict = self.sigmoid_loss(
                audio_embeds=model_outputs['student_audio_embeds'],
                text_embeds=model_outputs['student_text_embeds'],
                patient_ids=patient_ids
            )
            
            sigmoid_loss = sigmoid_loss_dict['contrastive_loss']
            total_loss += self.sigmoid_weight * sigmoid_loss
            loss_components.update({f'sigmoid_{k}': v for k, v in sigmoid_loss_dict.items()})
        
        # 4. LoCa Loss (Caption Generation)
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
    
    def training_step(self, batch, batch_idx):
        """훈련 스텝"""
        # 환자 ID 추출
        if 'patient_id' in batch:
            patient_ids = batch['patient_id'] if isinstance(batch['patient_id'], list) else [batch['patient_id']]
        else:
            if 'language' in batch:
                languages = batch['language'] if isinstance(batch['language'], list) else ['Unknown'] * len(batch['labels'])
                patient_ids = [f"{lang}_{i // 2}" for i, lang in enumerate(languages)]
            else:
                patient_ids = [f"patient_{i // 2}" for i in range(len(batch['labels']))]
        
        # Caption targets (임시로 None)
        caption_targets = None
        
        # Forward pass
        model_outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            pixel_values=batch['pixel_values'],
            pixel_attention_mask=batch.get('pixel_attention_mask'),
            spatial_shapes=batch.get('spatial_shapes'),
            return_embeddings=True,
            training=True
        )
        
        # Loss 계산
        loss_dict = self.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
        loss = loss_dict['total_loss']
        
        # EMA Teacher 업데이트
        self.ema_teacher.update(self.siglip_student)
        
        # 메트릭 계산
        logits = model_outputs['classification_logits']
        self.train_accuracy(logits, batch['labels'])
        
        # 로깅
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        
        # Loss components 로깅
        for key, value in loss_dict['loss_components'].items():
            if isinstance(value, torch.Tensor):
                self.log(f'train_{key}', value)
            else:
                self.log(f'train_{key}', value)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """검증 스텝"""
        # 환자 ID 추출
        if 'patient_id' in batch:
            patient_ids = batch['patient_id'] if isinstance(batch['patient_id'], list) else [batch['patient_id']]
        else:
            if 'language' in batch:
                languages = batch['language'] if isinstance(batch['language'], list) else ['Unknown'] * len(batch['labels'])
                patient_ids = [f"{lang}_{i // 2}" for i, lang in enumerate(languages)]
            else:
                patient_ids = [f"patient_{i // 2}" for i in range(len(batch['labels']))]
        
        caption_targets = None
        
        # Forward pass (no teacher update during validation)
        model_outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            pixel_values=batch['pixel_values'],
            pixel_attention_mask=batch.get('pixel_attention_mask'),
            spatial_shapes=batch.get('spatial_shapes'),
            return_embeddings=True,
            training=False
        )
        
        # Loss 계산 (teacher 없이)
        loss_dict = self.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
        loss = loss_dict['total_loss']
        
        # 메트릭 계산
        logits = model_outputs['classification_logits']
        self.val_accuracy(logits, batch['labels'])
        
        # 결과 저장
        output = {
            'val_loss': loss,
            'logits': logits.detach(),
            'labels': batch['labels'],
            'language': batch.get('language', ['Unknown'] * len(batch['labels']))
        }
        
        # Loss components 추가
        for key, value in loss_dict['loss_components'].items():
            if isinstance(value, torch.Tensor):
                output[f'val_{key}'] = value.detach()
            else:
                output[f'val_{key}'] = value
        
        self.validation_step_outputs.append(output)
        
        # 로깅
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """테스트 스텝"""
        # 환자 ID 추출
        if 'patient_id' in batch:
            patient_ids = batch['patient_id'] if isinstance(batch['patient_id'], list) else [batch['patient_id']]
        else:
            if 'language' in batch:
                languages = batch['language'] if isinstance(batch['language'], list) else ['Unknown'] * len(batch['labels'])
                patient_ids = [f"{lang}_{i // 2}" for i, lang in enumerate(languages)]
            else:
                patient_ids = [f"patient_{i // 2}" for i in range(len(batch['labels']))]
        
        caption_targets = None
        
        # Forward pass
        model_outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            pixel_values=batch['pixel_values'],
            pixel_attention_mask=batch.get('pixel_attention_mask'),
            spatial_shapes=batch.get('spatial_shapes'),
            return_embeddings=True,
            training=False
        )
        
        # Loss 계산
        if self.criterion is not None:
            loss_dict = self.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
            loss = loss_dict['total_loss']
        else:
            loss = 0.0
            loss_dict = {'loss_components': {}}
        
        # 메트릭 계산
        logits = model_outputs['classification_logits']
        if self.criterion is not None:
            self.test_accuracy(logits, batch['labels'])
        
        # 결과 저장
        output = {
            'test_loss': loss,
            'logits': logits.detach(),
            'labels': batch['labels'],
            'language': batch.get('language', ['Unknown'] * len(batch['labels']))
        }
        
        # Loss components 추가
        for key, value in loss_dict['loss_components'].items():
            if isinstance(value, torch.Tensor):
                output[f'test_{key}'] = value.detach()
            else:
                output[f'test_{key}'] = value
        
        self.test_step_outputs.append(output)
        
        # 로깅
        if self.criterion is not None:
            self.log('test_loss', loss)
            self.log('test_acc', self.test_accuracy)
        
        return logits
    
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
    
    def _compute_validation_metrics(self):
        """검증 메트릭 계산"""
        if not self.validation_step_outputs:
            return
        
        # 기본 메트릭 계산 (기존 코드 재사용)
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_languages = []
        for x in self.validation_step_outputs:
            if isinstance(x['language'], list):
                all_languages.extend(x['language'])
            else:
                all_languages.extend(['Unknown'] * len(x['labels']))
        
        # AUC 계산
        try:
            from sklearn.metrics import roc_auc_score
            probs = torch.softmax(all_logits, dim=1)[:, 1].cpu().numpy()
            labels = all_labels.cpu().numpy()
            auc = roc_auc_score(labels, probs)
            self.log('val_auc', auc)
        except:
            self.log('val_auc', 0.0)
        
        self.validation_step_outputs.clear()
    
    def _compute_test_metrics(self):
        """테스트 메트릭 계산"""
        if not self.test_step_outputs:
            return
        
        # 언어별 메트릭 계산 (기존 코드 재사용)
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        all_languages = []
        for x in self.test_step_outputs:
            if isinstance(x['language'], list):
                all_languages.extend(x['language'])
            else:
                all_languages.extend(['Unknown'] * len(x['labels']))
        
        # AUC 계산
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            probs = torch.softmax(all_logits, dim=1)[:, 1].cpu().numpy()
            labels = all_labels.cpu().numpy()
            auc = roc_auc_score(labels, probs)
            
            # 최적 threshold 계산
            fpr, tpr, thresholds = roc_curve(labels, probs)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            self.log('test_auc', auc)
            self.log('test_optimal_threshold', optimal_threshold)
            
            # 언어별 메트릭 계산 및 로깅
            self._compute_language_specific_metrics(probs, labels, all_languages, optimal_threshold)
            
        except Exception as e:
            print(f"⚠️ 테스트 메트릭 계산 실패: {e}")
            self.log('test_auc', 0.0)
        
        self.test_step_outputs.clear()
    
    def _compute_language_specific_metrics(self, y_scores, y_true, all_languages, optimal_threshold):
        """언어별 테스트 메트릭 계산 및 출력 (기존 코드 재사용)"""
        from collections import defaultdict, Counter
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        # 언어별로 데이터 그룹화
        language_data = defaultdict(lambda: {'scores': [], 'labels': [], 'indices': []})
        
        for i, (score, label, lang) in enumerate(zip(y_scores, y_true, all_languages)):
            language_data[lang]['scores'].append(score)
            language_data[lang]['labels'].append(label)
            language_data[lang]['indices'].append(i)
        
        print(f"\n🌍 진정한 SigLIP2 언어별 테스트 결과:")
        print(f"{'='*80}")
        
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
            lang_default_preds = (lang_scores >= 0.5).astype(int)
            
            # 메트릭 계산
            lang_optimal_acc = accuracy_score(lang_labels, lang_optimal_preds)
            lang_default_acc = accuracy_score(lang_labels, lang_default_preds)
            
            lang_precision, lang_recall, lang_f1, _ = precision_recall_fscore_support(
                lang_labels, lang_optimal_preds, average='weighted', zero_division=0
            )
            
            # 클래스별 분포
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
            self.log(f'test_{lang}_auc', lang_auc)
            self.log(f'test_{lang}_accuracy_optimal', lang_optimal_acc)
            self.log(f'test_{lang}_accuracy_default', lang_default_acc)
            self.log(f'test_{lang}_precision', lang_precision)
            self.log(f'test_{lang}_recall', lang_recall)
            self.log(f'test_{lang}_f1', lang_f1)
            self.log(f'test_{lang}_sample_count', len(lang_scores))
        
        print(f"{'='*80}")
    
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
        if self.hparams.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            # SAM은 PyTorch Lightning에서 복잡하므로 AdamW로 대체
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        
        # 스케줄러 설정
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def create_true_siglip2_model(config) -> TrueSigLIP2DementiaClassifier:
    """진정한 SigLIP2 모델 생성 함수"""
    model = TrueSigLIP2DementiaClassifier(
        model_name=config.model_name,
        num_classes=2,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_epochs=getattr(config, 'num_epochs', 10),
        use_language_embedding=True,
        loss_type=config.loss_type,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        optimizer_type=config.optimizer_type,
        sam_rho=getattr(config, 'sam_rho', 0.05),
        # True SigLIP2 전용 파라미터
        ema_momentum=getattr(config, 'ema_momentum', 0.999),
        silc_weight=getattr(config, 'silc_weight', 0.2),
        sigmoid_weight=getattr(config, 'sigmoid_weight', 1.0),
        loca_weight=getattr(config, 'loca_weight', 1.0),
        classification_weight=getattr(config, 'classification_weight', 1.0),
        mask_ratio=getattr(config, 'mask_ratio', 0.15),
        decoder_hidden_dim=getattr(config, 'decoder_hidden_dim', 512),
        decoder_num_heads=getattr(config, 'decoder_num_heads', 8),
        decoder_num_layers=getattr(config, 'decoder_num_layers', 6),
        vocab_size=getattr(config, 'vocab_size', 30522),
        max_caption_length=getattr(config, 'max_caption_length', 77)
    )
    
    return model
