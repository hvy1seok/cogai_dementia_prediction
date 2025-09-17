"""
SigLIP-SAM 치매 진단 모델 설정
순수 PyTorch 구현 (SAM 옵티마이저 지원)
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class SigLIPSAMConfig:
    """SigLIP-SAM 모델 설정"""
    # 모델 설정
    model_name: str = "google/siglip2-base-patch16-naflex"
    text_tokenizer: str = "google/gemma-2b"  # Gemma 토크나이저 (256K vocab, multilingual)
    max_length: int = 64  # SigLIP2 모델의 최대 텍스트 길이
    image_size: int = 224
    
    # 오디오 처리 설정
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: float = 0.0
    fmax: float = 8000.0
    
    # 학습 설정
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 100
    warmup_steps: int = 100
    weight_decay: float = 0.01
    early_stopping_patience: int = 15  # Validation AUC 기준 Early Stopping (15 에폭)
    
    # 손실 함수 설정
    loss_type: str = "focal"  # "cross_entropy", "focal", "bce"
    focal_alpha: float = 1.0          # Focal Loss alpha 파라미터 (또는 "auto"로 자동 계산)
    focal_gamma: float = 2.0          # Focal Loss gamma 파라미터
    auto_class_weights: bool = True   # 클래스 불균형 자동 보정
    
    # 옵티마이저 설정
    optimizer_type: str = "sam"       # "adamw", "lion", "sam"
    sam_rho: float = 0.05             # SAM rho 파라미터
    sam_adaptive: bool = False        # Adaptive SAM 사용 여부
    
    # 데이터 설정
    train_split: float = 0.7
    val_split: float = 0.1
    test_split: float = 0.2
    random_seed: int = 42
    split_by_patient: bool = True  # True: 환자 단위 분할, False: 샘플 단위 분할
    
    # 경로 설정
    data_dir: str = "../../training_dset"
    output_dir: str = "../modules/outputs/siglip-sam"
    checkpoint_dir: str = "../modules/outputs/siglip-sam/checkpoints"
    
    # 언어 설정 (언어 무관 학습을 위해)
    languages: List[str] = None
    
    # Cross-lingual 설정
    cross_lingual_mode: bool = False
    train_languages: List[str] = None
    test_languages: List[str] = None
    
    # 학습 설정
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # SigLIP2 Contrastive Learning 설정
    use_contrastive: bool = True          # Contrastive Learning 사용 여부
    contrastive_weight: float = 0.5       # Contrastive vs Classification 손실 가중치
    contrastive_temperature: float = 0.07 # Contrastive Learning 온도 파라미터
    
    # 진정한 SigLIP2 설정
    ema_momentum: float = 0.999           # EMA Teacher momentum
    silc_weight: float = 0.2              # SILC/TIPS Loss 가중치 (20%)
    sigmoid_weight: float = 1.0           # Sigmoid Contrastive Loss 가중치 (100%)
    loca_weight: float = 1.0              # LoCa Caption Loss 가중치 (100%)
    classification_weight: float = 1.0    # Classification Loss 가중치 (100%)
    mask_ratio: float = 0.15              # Masked prediction 비율
    decoder_hidden_dim: int = 512         # Auto-regressive decoder hidden dimension
    decoder_num_heads: int = 8            # Decoder attention heads
    decoder_num_layers: int = 6           # Decoder layers
    vocab_size: int = 30522               # Vocabulary size
    max_caption_length: int = 77          # Maximum caption length
    
    # 베스트 모델 선택 기준 설정
    best_model_metric: str = "val_auc"    # "val_auc" 또는 "avg_lang_auc"
    target_languages: List[str] = None    # avg_lang_auc 모드에서 평균을 계산할 타겟 언어들
    
    # 로깅 설정
    log_interval: int = 10
    save_interval: int = 5  # 에포크마다 저장
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["English", "Greek", "Spanish", "Mandarin"]
        
        if self.target_languages is None:
            self.target_languages = ["English", "Spanish", "Mandarin"]
        
        # 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
