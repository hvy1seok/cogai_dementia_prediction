"""
SigLIP2 기반 치매 진단 시스템 설정
"""
from dataclasses import dataclass
from typing import Optional, List
import os

@dataclass
class SigLIPConfig:
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
    num_epochs: int = 100  # Early stopping으로 실제로는 더 일찍 종료될 수 있음
    warmup_steps: int = 100
    weight_decay: float = 0.01
    early_stopping_patience: int = 15  # Validation AUC 기준 Early Stopping
    split_by_patient: bool = True  # True: 환자 단위 분할, False: 샘플 단위 분할
    
    # 손실 함수 설정
    loss_type: str = "cross_entropy"  # "cross_entropy", "focal", "bce"
    focal_alpha: float = 1.0          # Focal Loss alpha 파라미터
    focal_gamma: float = 2.0          # Focal Loss gamma 파라미터
    auto_class_weights: bool = True   # 클래스 불균형 자동 보정
    
    # 옵티마이저 설정
    optimizer_type: str = "adamw"     # "adamw", "lion", "sam"
    sam_rho: float = 0.05             # SAM rho 파라미터
    
    # SigLIP2 Contrastive Learning 설정
    use_contrastive: bool = False     # Contrastive learning 사용 여부
    contrastive_weight: float = 0.5   # Classification vs Contrastive loss 가중치
    contrastive_temperature: float = 0.07  # Contrastive learning 온도 파라미터
    
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
    
    # 데이터 설정
    train_split: float = 0.7
    val_split: float = 0.1
    test_split: float = 0.2
    random_seed: int = 42
    
    # 경로 설정
    data_dir: str = "../../training_dset"
    output_dir: str = "../modules/outputs/siglip"
    checkpoint_dir: str = "../modules/outputs/siglip/checkpoints"
    
    # 언어 설정 (언어 무관 학습을 위해)
    languages: List[str] = None
    
    # 베스트 모델 선택 기준 설정
    best_model_metric: str = "val_macro_f1"    # "val_auc", "val_macro_f1", "avg_lang_macro_f1"
    target_languages: List[str] = None    # avg_lang_macro_f1 모드에서 평균을 계산할 타겟 언어들
    
    # Cross-lingual 설정
    cross_lingual_mode: bool = False
    train_languages: List[str] = None
    test_languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["English", "Greek", "Spanish", "Mandarin"]
        
        if self.target_languages is None:
            self.target_languages = ["English", "Spanish", "Mandarin"]
        
        # 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # 조기 종료 설정
    early_stopping_patience: int = 10  # 10 에폭으로 변경
    early_stopping_threshold: float = 0.001
    
    # 혼합 정밀도 훈련
    fp16: bool = True
    bf16: bool = False
    
    # Cross-lingual 설정
    cross_lingual_mode: bool = False
    train_languages: List[str] = None
    test_languages: List[str] = None

@dataclass
class DataConfig:
    """데이터 처리 설정"""
    # 텍스트 전처리
    max_text_length: int = 512
    text_padding: str = "max_length"
    text_truncation: bool = True
    
    # 이미지 전처리
    image_mean: List[float] = None
    image_std: List[float] = None
    
    def __post_init__(self):
        if self.image_mean is None:
            self.image_mean = [0.485, 0.456, 0.406]
        if self.image_std is None:
            self.image_std = [0.229, 0.224, 0.225] 