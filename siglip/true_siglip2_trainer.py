"""
진정한 SigLIP2 치매 진단 모델 트레이너 (PyTorch Lightning 버전)
- EMA Teacher-Student 학습
- Multi-Loss: SILC/TIPS + Sigmoid + LoCa + Classification
- Caption generation 및 dense captioning
- PyTorch Lightning 기반 훈련
"""

import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from transformers import AutoProcessor
from datetime import datetime

from config import SigLIPConfig, TrainingConfig
from true_siglip2_model import create_true_siglip2_model
from data_processor import create_dataloaders

def setup_wandb_logger(config: SigLIPConfig, training_config: TrainingConfig):
    """wandb 로거 설정 - True SigLIP2 전용"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 언어 정보
    lang_info = "_".join(config.languages) if len(config.languages) <= 2 else f"{len(config.languages)}langs"
    
    model_info = "TrueSigLIP2"
    loss_info = f"{config.loss_type}_MultiLoss"
    opt_info = config.optimizer_type
    
    run_name = f"true-siglip2-lightning_{lang_info}_{model_info}_{loss_info}_{opt_info}_bs{config.batch_size}_lr{config.learning_rate}_{timestamp}"
    
    wandb_logger = WandbLogger(
        project="dementia-prediction-true-siglip2-lightning",
        name=run_name,
        tags=[
            f"loss_{config.loss_type}",
            f"optimizer_{config.optimizer_type}",
            f"batch_size_{config.batch_size}",
            f"languages_{len(config.languages)}",
            "true_siglip2",
            "ema_teacher_student",
            "multi_loss",
            "caption_generation",
            "pytorch_lightning"
        ],
        config={
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": getattr(config, 'num_epochs', 100),
            "languages": config.languages,
            "loss_type": config.loss_type,
            "optimizer_type": config.optimizer_type,
            "early_stopping_patience": training_config.early_stopping_patience,
            "ema_momentum": getattr(config, 'ema_momentum', 0.999),
            "silc_weight": getattr(config, 'silc_weight', 0.2),
            "sigmoid_weight": getattr(config, 'sigmoid_weight', 1.0),
            "loca_weight": getattr(config, 'loca_weight', 1.0),
            "classification_weight": getattr(config, 'classification_weight', 1.0),
        }
    )
    
    return wandb_logger

def create_callbacks(training_config: TrainingConfig, checkpoint_dir: str, config: SigLIPConfig):
    """콜백 생성 - True SigLIP2 전용"""
    callbacks = []
    
    # 베스트 모델 선택 기준에 따른 모니터링 메트릭 결정
    if hasattr(config, 'best_model_metric'):
        if config.best_model_metric == "avg_lang_auc":
            monitor_metric = 'val_avg_lang_auc'
            filename_template = 'true-siglip2-{epoch:02d}-{val_avg_lang_auc:.3f}'
            print(f"📊 베스트 모델 선택 기준: 언어별 평균 AUC ({config.target_languages})")
        elif config.best_model_metric == "val_macro_f1":
            monitor_metric = 'val_macro_f1'
            filename_template = 'true-siglip2-{epoch:02d}-{val_macro_f1:.3f}'
            print(f"📊 베스트 모델 선택 기준: 전체 Macro F1")
        elif config.best_model_metric == "avg_lang_macro_f1":
            monitor_metric = 'val_avg_lang_macro_f1'
            filename_template = 'true-siglip2-{epoch:02d}-{val_avg_lang_macro_f1:.3f}'
            print(f"📊 베스트 모델 선택 기준: 언어별 평균 Macro F1 ({config.target_languages})")
        else:
            monitor_metric = 'val_auc'
            filename_template = 'true-siglip2-{epoch:02d}-{val_auc:.3f}'
            print(f"📊 베스트 모델 선택 기준: 전체 AUC")
    else:
        monitor_metric = 'val_auc'
        filename_template = 'true-siglip2-{epoch:02d}-{val_auc:.3f}'
        print(f"📊 베스트 모델 선택 기준: 전체 AUC")
    
    # 모델 체크포인트 콜백
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename_template,
        monitor=monitor_metric,
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early Stopping 콜백
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode='max',
        patience=training_config.early_stopping_patience,
        verbose=True,
        min_delta=training_config.early_stopping_threshold
    )
    callbacks.append(early_stopping)
    
    print(f"⏳ Early Stopping: {monitor_metric} 기준 {training_config.early_stopping_patience} epochs patience")
    
    return callbacks

def train_model(config: SigLIPConfig, training_config: TrainingConfig):
    """메인 훈련 함수 - True SigLIP2 PyTorch Lightning"""
    print("=== 진정한 SigLIP2 치매 진단 모델 훈련 시작 (PyTorch Lightning) ===")
    
    # 시드 설정
    pl.seed_everything(config.random_seed, workers=True)
    
    # SigLIP2 프로세서 및 Gemma 토크나이저 로드
    print("SigLIP2 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained(config.model_name)
    
    print("Gemma 토크나이저 로드 중...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer)
    print(f"✅ Gemma 토크나이저 로드 완료! Vocab size: {tokenizer.vocab_size}")
    
    # 데이터로더 생성
    print("데이터로더 생성 중...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        processor=processor,
        tokenizer=tokenizer,
        config=config,
        cross_lingual_mode=config.cross_lingual_mode,
        train_languages=config.train_languages,
        test_languages=config.test_languages
    )
    
    print(f"훈련 데이터: {len(train_loader.dataset)} 샘플")
    print(f"검증 데이터: {len(val_loader.dataset)} 샘플")
    print(f"테스트 데이터: {len(test_loader.dataset)} 샘플")
    
    # True SigLIP2 모델 생성
    print("진정한 SigLIP2 모델 생성 중...")
    model = create_true_siglip2_model(config)
    
    # 클래스 가중치 계산 및 손실 함수 설정
    from data_processor import compute_class_weights
    
    if hasattr(train_loader.dataset, 'dataset'):
        original_dataset = train_loader.dataset.dataset
    else:
        original_dataset = train_loader.dataset
    
    class_weights = compute_class_weights(original_dataset, config)
    model.setup_loss_function(class_weights)
    
    # wandb 로거 설정
    wandb_logger = setup_wandb_logger(config, training_config)
    
    # 콜백 설정
    callbacks = create_callbacks(training_config, config.checkpoint_dir, config)
    
    # 트레이너 설정
    # 작은 데이터셋에서는 에폭 단위로 검증 수행
    val_check_interval = None if len(train_loader) < 50 else training_config.eval_steps
    
    trainer = pl.Trainer(
        max_epochs=getattr(config, 'num_epochs', 100),
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if training_config.fp16 else 32,
        gradient_clip_val=training_config.max_grad_norm,
        accumulate_grad_batches=training_config.gradient_accumulation_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=1 if val_check_interval is None else None,
        log_every_n_steps=min(training_config.logging_steps, len(train_loader)),
        deterministic=False,  # CUDA upsample 연산과의 호환성을 위해 비활성화
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 시드 설정으로 재현성 보장 (deterministic 알고리즘 대신)
    pl.seed_everything(42, workers=True)
    print("✅ 시드 설정 완료 (재현성 보장)")
    
    # 훈련 시작
    print("🚀 진정한 SigLIP2 훈련 시작...")
    print("⚡ EMA Teacher-Student + Multi-Loss로 최고 성능 달성!")
    trainer.fit(model, train_loader, val_loader)
    
    # 테스트
    print("🔍 최종 테스트 수행...")
    test_results = trainer.test(model, test_loader, ckpt_path='best')
    
    # 베스트 모델 경로 출력
    best_model_path = callbacks[0].best_model_path
    print(f"\n=== 진정한 SigLIP2 훈련 완료 ===")
    print(f"💾 베스트 모델: {best_model_path}")
    
    return model, best_model_path

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="진정한 SigLIP2 치매 진단 모델 훈련 (PyTorch Lightning)")
    
    # 기본 설정
    parser.add_argument("--data_dir", type=str, default="../../training_dset", help="데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default="../modules/outputs/siglip/true-siglip2", help="출력 디렉토리")
    parser.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-naflex", help="모델 이름")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="학습률")
    parser.add_argument("--num_epochs", type=int, default=100, help="에포크 수")
    
    # 언어별 파서 선택 옵션
    parser.add_argument("--parser", type=str, default="all", 
                       choices=["all", "English", "Greek", "Spanish", "Mandarin"],
                       help="사용할 언어 파서")
    parser.add_argument("--languages", nargs="+", default=None, help="특정 언어 목록")
    
    # 손실 함수 선택 옵션
    parser.add_argument("--loss_type", type=str, default="focal",
                       choices=["cross_entropy", "focal", "bce"],
                       help="손실 함수 타입")
    parser.add_argument("--focal_alpha", type=float, default=1.0, help="Focal Loss alpha 파라미터")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma 파라미터")
    parser.add_argument("--auto_class_weights", action="store_true", help="클래스 불균형 자동 보정")
    
    # 옵티마이저 선택 옵션
    parser.add_argument("--optimizer_type", type=str, default="adamw",
                       choices=["adamw"],  # PyTorch Lightning에서는 AdamW만 지원
                       help="옵티마이저 타입")
    
    # True SigLIP2 전용 옵션
    parser.add_argument("--ema_momentum", type=float, default=0.999, help="EMA Teacher momentum")
    parser.add_argument("--silc_weight", type=float, default=0.2, help="SILC/TIPS Loss 가중치")
    parser.add_argument("--sigmoid_weight", type=float, default=1.0, help="Sigmoid Loss 가중치")
    parser.add_argument("--loca_weight", type=float, default=1.0, help="LoCa Loss 가중치")
    parser.add_argument("--classification_weight", type=float, default=1.0, help="Classification Loss 가중치")
    
    # 베스트 모델 선택 기준 옵션
    parser.add_argument("--best_model_metric", type=str, default="val_macro_f1", 
                       choices=["val_auc", "val_macro_f1", "avg_lang_auc", "avg_lang_macro_f1"],
                       help="베스트 모델 선택 기준 (val_macro_f1: 전체 Macro F1, avg_lang_macro_f1: 언어별 평균 Macro F1)")
    parser.add_argument("--target_languages", nargs="+", default=["English", "Spanish", "Mandarin"],
                       help="avg_lang_auc 모드에서 평균을 계산할 타겟 언어들")
    
    # Cross-lingual 모드 옵션
    parser.add_argument("--train_languages", nargs="+", default=["English", "Spanish", "Mandarin"],
                       help="Cross-lingual 모드에서 훈련에 사용할 언어들")
    parser.add_argument("--test_languages", nargs="+", default=["Greek"],
                       help="Cross-lingual 모드에서 테스트에 사용할 언어들")
    
    # 데이터 분할 방식 옵션
    parser.add_argument("--split_by_patient", type=str, default="true", 
                       choices=["true", "false"],
                       help="데이터 분할 방식 (true: 환자 단위, false: 샘플 단위)")
    
    args = parser.parse_args()
    
    # 설정 생성
    config = SigLIPConfig()
    training_config = TrainingConfig()
    
    # 명령행 인수로 설정 덮어쓰기
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
        config.checkpoint_dir = f"{args.output_dir}/checkpoints"
    if args.model_name:
        config.model_name = args.model_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    
    # 손실 함수 설정
    if args.loss_type:
        config.loss_type = args.loss_type
    if args.focal_alpha:
        config.focal_alpha = args.focal_alpha
    if args.focal_gamma:
        config.focal_gamma = args.focal_gamma
    config.auto_class_weights = args.auto_class_weights
    
    # 옵티마이저 설정
    if args.optimizer_type:
        config.optimizer_type = args.optimizer_type
    
    # True SigLIP2 설정
    config.ema_momentum = args.ema_momentum
    config.silc_weight = args.silc_weight
    config.sigmoid_weight = args.sigmoid_weight
    config.loca_weight = args.loca_weight
    config.classification_weight = args.classification_weight
    
    # 베스트 모델 선택 기준 설정
    config.best_model_metric = args.best_model_metric
    config.target_languages = args.target_languages
    config.split_by_patient = args.split_by_patient.lower() == "true"
    
    # Cross-lingual 설정
    if args.parser == "cross_lingual":
        config.cross_lingual_mode = True
        config.train_languages = args.train_languages
        config.test_languages = args.test_languages
        config.languages = args.train_languages + args.test_languages
    else:
        config.cross_lingual_mode = False
    
    # 언어 파서 설정
    if args.parser == "all":
        if args.languages:
            config.languages = args.languages
        else:
            config.languages = ["English", "Greek", "Spanish", "Mandarin"]
    else:
        config.languages = [args.parser]
    
    print(f"선택된 언어: {config.languages}")
    print(f"데이터 디렉토리: {config.data_dir}")
    print(f"옵티마이저: {config.optimizer_type}")
    print(f"손실 함수: {config.loss_type}")
    
    # 디렉토리 생성
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 모델 훈련
    model, best_model_path = train_model(config, training_config)

if __name__ == "__main__":
    main()
