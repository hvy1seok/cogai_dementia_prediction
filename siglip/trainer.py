"""
SigLIP2 치매 진단 모델 훈련 스크립트
PyTorch Lightning과 wandb 사용
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb
from transformers import AutoProcessor  # SigLIP2 지원
import argparse
from datetime import datetime

from config import SigLIPConfig, TrainingConfig
from data_processor import create_dataloaders
from model import create_model
from language_parsers import get_language_parser, parse_all_languages

def setup_wandb(config: SigLIPConfig, training_config: TrainingConfig):
    """wandb 설정"""
    wandb.init(
        project="dementia-prediction-siglip2",
        name=f"siglip2-dementia-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "languages": config.languages,
            "sample_rate": config.sample_rate,
            "n_mels": config.n_mels,
            "image_size": config.image_size,
            "max_length": config.max_length,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "max_grad_norm": training_config.max_grad_norm,
            "fp16": training_config.fp16,
            "bf16": training_config.bf16,
        }
    )
    
    return WandbLogger(project="dementia-prediction-siglip2")

def create_callbacks(training_config: TrainingConfig, checkpoint_dir: str):
    """콜백 생성"""
    callbacks = []
    
    # 모델 체크포인트 (AUC 기준 베스트 모델)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="siglip2-dementia-best-auc-{test_auc:.3f}-epoch{epoch:02d}",
        monitor="test_auc",  # AUC 기준으로 모니터링
        mode="max",  # AUC 최대값 추적
        save_top_k=3,  # 상위 3개 모델만 저장
        save_last=True,  # 마지막 모델도 저장
        verbose=True,
        auto_insert_metric_name=False  # 파일명에 메트릭 이름 자동 추가 방지
    )
    callbacks.append(checkpoint_callback)
    
    # 조기 종료 비활성화 (validation 없으므로 제거)
    
    # 학습률 모니터링
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks

def train_model(config: SigLIPConfig, training_config: TrainingConfig):
    """모델 훈련"""
    print("=== SigLIP2 치매 진단 모델 훈련 시작 ===")
    
    # 시드 설정
    pl.seed_everything(config.random_seed)
    
    # GPU 확인
    if torch.cuda.is_available():
        print(f"GPU 사용 가능: {torch.cuda.get_device_name()}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
    else:
        print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")
    
    # SigLIP2 프로세서 로드
    print("SigLIP2 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained(config.model_name)  # SigLIP2 지원
    
    # 데이터로더 생성
    print("데이터로더 생성 중...")
    train_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        processor=processor,
        config=config,
        train_split=0.8,
        test_split=0.2
    )
    
    print(f"훈련 데이터: {len(train_loader.dataset)} 샘플")
    print(f"테스트 데이터: {len(test_loader.dataset)} 샘플")
    
    # 모델 생성
    print("모델 생성 중...")
    model = create_model(config)
    
    # wandb 설정
    print("wandb 설정 중...")
    wandb_logger = setup_wandb(config, training_config)
    
    # 콜백 생성
    print("콜백 생성 중...")
    callbacks = create_callbacks(training_config, config.checkpoint_dir)
    
    # 훈련 설정
    trainer_kwargs = {
        "max_epochs": config.num_epochs,
        "logger": wandb_logger,
        "callbacks": callbacks,
        "accelerator": "auto",
        "devices": "auto",
        "precision": "16-mixed" if training_config.fp16 else "32",
        "gradient_clip_val": training_config.max_grad_norm,
        "accumulate_grad_batches": training_config.gradient_accumulation_steps,
        "log_every_n_steps": 10,  # 32 배치보다 작게 설정
        "check_val_every_n_epoch": None,  # validation 비활성화
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    # 멀티 GPU 설정
    if torch.cuda.device_count() > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)  # 동적 분류기 때문에 True 필요
        print(f"멀티 GPU 훈련 설정: {torch.cuda.device_count()}개 GPU 사용")
    
    # 훈련기 생성
    trainer = pl.Trainer(**trainer_kwargs)
    
    # 훈련 시작 (validation 없이)
    print("훈련 시작...")
    trainer.fit(model, train_loader)
    
    # 베스트 모델 로드 및 테스트
    print("베스트 모델 로드 중...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"✅ 베스트 모델 경로: {best_model_path}")
        # 베스트 모델 로드
        model = SigLIPDementiaClassifier.load_from_checkpoint(best_model_path)
        print("🏆 베스트 AUC 모델로 테스트 실행...")
    else:
        print("⚠️ 베스트 모델을 찾을 수 없습니다. 마지막 모델로 테스트합니다.")
    
    trainer.test(model, test_loader)
    
    # wandb 종료
    wandb.finish()
    
    print("=== 훈련 완료 ===")
    return model, trainer

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="SigLIP2 치매 진단 모델 훈련")
    parser.add_argument("--config", type=str, default=None, help="설정 파일 경로")
    parser.add_argument("--data_dir", type=str, default="../training_dset", help="데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default="../modules/outputs/siglip", help="출력 디렉토리")
    parser.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224", help="모델 이름")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="학습률")
    parser.add_argument("--num_epochs", type=int, default=10, help="에포크 수")
    # 언어별 파서 선택 옵션
    parser.add_argument("--parser", type=str, default="all", 
                       choices=["all", "English", "Greek", "Spanish", "Mandarin"],
                       help="사용할 언어 파서 (all: 모든 언어, 개별 언어 선택 가능)")
    parser.add_argument("--languages", nargs="+", default=None, help="특정 언어 목록 (parser=all일 때 사용)")
    
    args = parser.parse_args()
    
    # 설정 생성
    config = SigLIPConfig()
    training_config = TrainingConfig()
    
    # 명령행 인수로 설정 덮어쓰기
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model_name:
        config.model_name = args.model_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    
    # 언어 파서 설정
    if args.parser == "all":
        if args.languages:
            config.languages = args.languages
        else:
            config.languages = ["English", "Greek", "Spanish", "Mandarin"]
    else:
        # 단일 언어 선택
        config.languages = [args.parser]
    
    print(f"선택된 언어: {config.languages}")
    print(f"데이터 디렉토리: {config.data_dir}")
    
    # 데이터 디렉토리 확인
    if not os.path.exists(config.data_dir):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {config.data_dir}")
        return
    
    # 선택된 언어별 데이터 확인
    print("\n=== 언어별 데이터 확인 ===")
    for language in config.languages:
        try:
            parser = get_language_parser(language, config.data_dir)
            data = parser.parse_data()
            print(f"{language}: {len(data)}개 샘플")
            if data:
                normal_count = sum(1 for d in data if d['label'] == 0)
                dementia_count = sum(1 for d in data if d['label'] == 1)
                print(f"  - 정상: {normal_count}개, 치매: {dementia_count}개")
        except Exception as e:
            print(f"{language}: 오류 - {e}")
    
    # 훈련 실행
    model, trainer = train_model(config, training_config)
    
    print(f"훈련 완료! 모델이 {config.checkpoint_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main() 