"""
Control Groups Configuration
대조군 모델들을 위한 설정 파일
"""

from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class ControlGroupConfig:
    """대조군 모델들의 공통 설정"""
    
    # 데이터 설정
    data_dir: str = "../../training_dset"
    languages: List[str] = None  # ["English", "Mandarin"] or ["English", "Greek", "Spanish", "Mandarin"]
    max_seq_length: int = 512
    
    # 훈련 설정
    batch_size: int = 64
    learning_rate: float = 2e-5
    num_epochs: int = 100
    early_stopping_patience: int = 15
    
    # 모델 공통 설정
    hidden_dim: int = 768
    dropout: float = 0.1
    num_classes: int = 2  # HC(0), AD(1)
    
    # 손실 함수 설정
    loss_type: str = "focal"  # "focal", "bce"
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    auto_class_weights: bool = True
    
    # 평가 설정
    best_model_metric: str = "avg_lang_macro_f1"  # "val_macro_f1", "avg_lang_macro_f1"
    target_languages: List[str] = None
    split_by_patient: bool = True
    
    # 디바이스 및 시드
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    
    # 출력 설정
    output_dir: str = "../modules/outputs/controlgroups"
    save_checkpoints: bool = True
    log_wandb: bool = True
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["English", "Mandarin"]  # 기본값: 영어 + 중국어
        if self.target_languages is None:
            self.target_languages = self.languages.copy()

@dataclass 
class AudioOnlyConfig(ControlGroupConfig):
    """ViT-Spec (Audio-only) 모델 설정"""
    
    # 오디오 전용 설정
    model_name: str = "google/vit-base-patch16-224"  # ViT 백본 (기존)
    siglip_model: str = "google/siglip-base-patch16-224"  # SigLIP 백본 (새로운 기본값)
    audio_feature_dim: int = 768
    mel_bins: int = 128
    max_audio_length: int = 1024
    
    # 분류기 설정
    classifier_hidden_dims: List[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.classifier_hidden_dims is None:
            self.classifier_hidden_dims = [512, 256]

@dataclass
class TextOnlyConfig(ControlGroupConfig):
    """Text-only (Gemma Encoder) 모델 설정"""
    
    # 텍스트 전용 설정
    text_encoder: str = "google/gemma-2b"  # Gemma-2b 유지 (기존)
    siglip_model: str = "google/siglip-base-patch16-224"  # SigLIP 백본 (새로운 기본값)
    text_tokenizer: str = "google/gemma-2b"  # Gemma 토크나이저
    text_feature_dim: int = 768  # SigLIP과 동일한 차원으로 제한
    use_cls_token: bool = True
    
    # 분류기 설정
    classifier_hidden_dims: List[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.classifier_hidden_dims is None:
            self.classifier_hidden_dims = [512, 256]

@dataclass
class ConcatConfig(ControlGroupConfig):
    """Concat (ViT + XLM-R) Late Fusion 모델 설정"""
    
    # 오디오 인코더 설정
    audio_encoder: str = "google/vit-base-patch16-224"  # ViT 백본 (기존)
    siglip_model: str = "google/siglip-base-patch16-224"  # SigLIP 백본 (새로운 기본값)
    audio_feature_dim: int = 768
    mel_bins: int = 128
    max_audio_length: int = 1024
    
    # 텍스트 인코더 설정
    text_encoder: str = "google/gemma-2b"  # Gemma-2b 유지 (기존)
    text_tokenizer: str = "google/gemma-2b"  # Gemma 토크나이저
    text_feature_dim: int = 768  # SigLIP과 동일한 차원으로 제한
    use_cls_token: bool = True
    
    # Late Fusion 설정
    fusion_method: str = "concat"  # "concat", "add", "attention"
    fused_feature_dim: int = 1536  # audio_dim(768) + text_dim(768)
    
    # 2층 FFN 설정
    ffn_hidden_dims: List[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.ffn_hidden_dims is None:
            self.ffn_hidden_dims = [1024, 512]  # 2층 FFN
        # SigLIP과 동일한 차원 사용
        self.fused_feature_dim = self.audio_feature_dim + self.text_feature_dim  # 768 + 768 = 1536
