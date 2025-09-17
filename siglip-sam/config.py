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
    
    # 로깅 설정
    log_interval: int = 10
    save_interval: int = 5  # 에포크마다 저장
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["English", "Greek", "Spanish", "Mandarin"]
        
        # 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
