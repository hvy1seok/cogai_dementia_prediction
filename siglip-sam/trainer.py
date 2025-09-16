"""
SigLIP-SAM 치매 진단 모델 트레이너
순수 PyTorch 구현 (SAM 옵티마이저 지원)
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import wandb
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoProcessor
from typing import Dict, List

from config import SigLIPSAMConfig
from model import SigLIPSAMDementiaClassifier
from data_processor import create_dataloaders
from sam_optimizer import SAM

def setup_wandb(config: SigLIPSAMConfig):
    """wandb 설정 - 실험 설정이 포함된 상세한 이름 생성"""
    # 실행 이름 생성 - 설정 정보 포함
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 언어 정보
    if config.cross_lingual_mode:
        train_langs = "_".join(config.train_languages) if config.train_languages else "Unknown"
        test_langs = "_".join(config.test_languages) if config.test_languages else "Unknown"
        lang_info = f"CrossLingual_Train{train_langs}_Test{test_langs}"
    else:
        lang_info = "_".join(config.languages) if len(config.languages) <= 2 else f"{len(config.languages)}langs"
    
    # 모델 및 설정 정보
    model_info = config.model_name.split("/")[-1] if "/" in config.model_name else config.model_name
    loss_info = config.loss_type
    opt_info = config.optimizer_type
    
    run_name = f"siglip-sam_{lang_info}_{model_info}_{loss_info}_{opt_info}_bs{config.batch_size}_lr{config.learning_rate}_{timestamp}"
    
    wandb.init(
        project="dementia-prediction-siglip-sam",
        name=run_name,
        tags=[
            f"loss_{config.loss_type}",
            f"optimizer_{config.optimizer_type}",
            f"batch_size_{config.batch_size}",
            f"languages_{len(config.languages)}",
            "cross_lingual" if config.cross_lingual_mode else "standard",
            "sam_optimizer" if config.optimizer_type == "sam" else "standard_optimizer"
        ],
        config={
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "languages": config.languages,
            "loss_type": config.loss_type,
            "optimizer_type": config.optimizer_type,
            "focal_alpha": config.focal_alpha,
            "focal_gamma": config.focal_gamma,
            "sam_rho": config.sam_rho,
            "sam_adaptive": config.sam_adaptive,
            "sample_rate": config.sample_rate,
            "n_mels": config.n_mels,
            "image_size": config.image_size,
            "max_length": config.max_length,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "mixed_precision": config.mixed_precision,
            "gradient_clip_norm": config.gradient_clip_norm,
            # Cross-lingual 설정
            "cross_lingual_mode": config.cross_lingual_mode,
            "train_languages": config.train_languages,
            "test_languages": config.test_languages,
        }
    )

def compute_metrics(predictions, labels):
    """메트릭 계산"""
    # 예측값과 실제값
    preds = np.argmax(predictions, axis=1)
    
    # 기본 메트릭
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    # AUC 계산 (확률값 사용)
    try:
        probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train_epoch(model, train_loader, optimizer, config, scaler=None):
    """한 에포크 훈련"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        # GPU로 이동
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(config.device)
        
        # SAM 옵티마이저 사용 시
        if config.optimizer_type == "sam":
            # 첫 번째 forward pass
            def closure():
                if scaler and config.mixed_precision:
                    with autocast():
                        logits = model(batch)
                        loss = model.compute_loss(logits, batch['labels'])
                    scaler.scale(loss).backward()
                else:
                    logits = model(batch)
                    loss = model.compute_loss(logits, batch['labels'])
                    loss.backward()
                return loss
            
            # SAM first step
            if scaler and config.mixed_precision:
                with autocast():
                    logits = model(batch)
                    loss = model.compute_loss(logits, batch['labels'])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                optimizer.first_step(zero_grad=True)
                
                # 두 번째 forward pass
                closure()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.second_step(zero_grad=True)
                scaler.update()
            else:
                logits = model(batch)
                loss = model.compute_loss(logits, batch['labels'])
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # 두 번째 forward pass
                closure()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.second_step(zero_grad=True)
        
        else:
            # 일반 옵티마이저
            optimizer.zero_grad()
            
            if scaler and config.mixed_precision:
                with autocast():
                    logits = model(batch)
                    loss = model.compute_loss(logits, batch['labels'])
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch)
                loss = model.compute_loss(logits, batch['labels'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
        
        # 메트릭 수집
        total_loss += loss.item()
        all_predictions.extend(logits.detach().cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
        
        # 로깅
        if batch_idx % config.log_interval == 0:
            print(f'Train Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}')
            wandb.log({
                'train_batch_loss': loss.item(),
                'train_step': batch_idx
            })
    
    # 에포크 메트릭 계산
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    
    return avg_loss, metrics

def evaluate(model, test_loader, config):
    """모델 평가"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # GPU로 이동
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.device)
            
            if config.mixed_precision:
                with autocast():
                    logits = model(batch)
                    loss = model.compute_loss(logits, batch['labels'])
            else:
                logits = model(batch)
                loss = model.compute_loss(logits, batch['labels'])
            
            # 메트릭 수집
            total_loss += loss.item()
            all_predictions.extend(logits.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # 메트릭 계산
    avg_loss = total_loss / len(test_loader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    
    return avg_loss, metrics

def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    filename = f"checkpoint_epoch_{epoch:03d}_auc_{metrics['auc']:.3f}.pt"
    if is_best:
        filename = f"best_model_auc_{metrics['auc']:.3f}_epoch_{epoch:03d}.pt"
    
    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"💾 체크포인트 저장: {filepath}")
    
    return filepath

def train_model(config: SigLIPSAMConfig):
    """메인 훈련 함수"""
    print("=== SigLIP-SAM 치매 진단 모델 훈련 시작 ===")
    
    # 디바이스 설정
    if torch.cuda.is_available():
        config.device = "cuda"
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
    else:
        config.device = "cpu"
        print("CPU 사용")
    
    # 시드 설정
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # SigLIP2 프로세서 로드
    print("SigLIP2 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained(config.model_name)
    
    # 데이터로더 생성
    print("데이터로더 생성 중...")
    train_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        processor=processor,
        config=config,
        cross_lingual_mode=config.cross_lingual_mode,
        train_languages=config.train_languages,
        test_languages=config.test_languages
    )
    
    print(f"훈련 데이터: {len(train_loader.dataset)} 샘플")
    print(f"테스트 데이터: {len(test_loader.dataset)} 샘플")
    
    # 모델 생성
    print("모델 생성 중...")
    model = SigLIPSAMDementiaClassifier(config)
    model.to(config.device)
    
    # 옵티마이저 생성
    optimizer = model.create_optimizer(config)
    
    # 스케줄러 생성
    total_steps = len(train_loader) * config.num_epochs
    scheduler = model.create_scheduler(optimizer, config, total_steps)
    
    # Mixed precision 스케일러
    scaler = GradScaler() if config.mixed_precision else None
    
    # wandb 설정
    setup_wandb(config)
    
    # 훈련 루프
    best_auc = 0.0
    best_model_path = None
    
    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        
        # 훈련
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, config, scaler)
        
        # 평가
        test_loss, test_metrics = evaluate(model, test_loader, config)
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 로깅
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_f1': train_metrics['f1'],
            'train_auc': train_metrics['auc'],
            'test_loss': test_loss,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_auc': test_metrics['auc'],
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # 결과 출력
        print(f"훈련 - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"테스트 - Loss: {test_loss:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
        
        # 베스트 모델 저장
        if test_metrics['auc'] > best_auc:
            best_auc = test_metrics['auc']
            best_model_path = save_checkpoint(model, optimizer, epoch + 1, test_metrics, config, is_best=True)
            print(f"🏆 새로운 베스트 모델! AUC: {best_auc:.4f}")
        
        # 정기 체크포인트 저장
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, test_metrics, config, is_best=False)
    
    print(f"\n=== 훈련 완료 ===")
    print(f"🏆 베스트 AUC: {best_auc:.4f}")
    print(f"💾 베스트 모델: {best_model_path}")
    
    # wandb 종료
    wandb.finish()
    
    return model, best_model_path

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="SigLIP-SAM 치매 진단 모델 훈련")
    
    # 기본 설정
    parser.add_argument("--data_dir", type=str, default="../../training_dset", help="데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default="../modules/outputs/siglip-sam", help="출력 디렉토리")
    parser.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-naflex", help="모델 이름")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="학습률")
    parser.add_argument("--num_epochs", type=int, default=100, help="에포크 수")
    
    # 언어별 파서 선택 옵션
    parser.add_argument("--parser", type=str, default="all", 
                       choices=["all", "English", "Greek", "Spanish", "Mandarin", "cross_lingual"],
                       help="사용할 언어 파서")
    parser.add_argument("--languages", nargs="+", default=None, help="특정 언어 목록")
    
    # Cross-lingual 모드 옵션
    parser.add_argument("--train_languages", nargs="+", default=["English", "Spanish", "Mandarin"],
                       help="Cross-lingual 모드에서 훈련에 사용할 언어들")
    parser.add_argument("--test_languages", nargs="+", default=["Greek"],
                       help="Cross-lingual 모드에서 테스트에 사용할 언어들")
    
    # 손실 함수 선택 옵션
    parser.add_argument("--loss_type", type=str, default="cross_entropy",
                       choices=["cross_entropy", "focal", "bce"],
                       help="손실 함수 타입")
    parser.add_argument("--focal_alpha", type=float, default=1.0, help="Focal Loss alpha 파라미터")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma 파라미터")
    
    # 옵티마이저 선택 옵션
    parser.add_argument("--optimizer_type", type=str, default="sam",
                       choices=["adamw", "lion", "sam"],
                       help="옵티마이저 타입")
    parser.add_argument("--sam_rho", type=float, default=0.05, help="SAM rho 파라미터")
    parser.add_argument("--sam_adaptive", action="store_true", help="Adaptive SAM 사용")
    
    args = parser.parse_args()
    
    # 설정 생성
    config = SigLIPSAMConfig()
    
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
    
    # 옵티마이저 설정
    if args.optimizer_type:
        config.optimizer_type = args.optimizer_type
    if args.sam_rho:
        config.sam_rho = args.sam_rho
    config.sam_adaptive = args.sam_adaptive
    
    # 언어 파서 설정
    if args.parser == "cross_lingual":
        # Cross-lingual 모드
        config.cross_lingual_mode = True
        config.train_languages = args.train_languages
        config.test_languages = args.test_languages
        
        # 출력 디렉토리 이름 업데이트
        train_langs_str = "_".join(args.train_languages)
        test_langs_str = "_".join(args.test_languages)
        config.output_dir = f"{config.output_dir}/CrossLingual_Train_{train_langs_str}_Test_{test_langs_str}"
        config.checkpoint_dir = f"{config.output_dir}/checkpoints"
        
        print("🌍 Cross-Lingual 모드 활성화")
        print(f"  훈련 언어: {args.train_languages}")
        print(f"  테스트 언어: {args.test_languages}")
        print(f"  출력 디렉토리: {config.output_dir}")
        
        # config.languages는 모든 언어 포함 (데이터 확인용)
        config.languages = args.train_languages + args.test_languages
        
    elif args.parser == "all":
        if args.languages:
            config.languages = args.languages
        else:
            config.languages = ["English", "Greek", "Spanish", "Mandarin"]
    else:
        # 단일 언어 선택
        config.languages = [args.parser]
    
    print(f"선택된 언어: {config.languages}")
    print(f"데이터 디렉토리: {config.data_dir}")
    print(f"옵티마이저: {config.optimizer_type}")
    print(f"손실 함수: {config.loss_type}")
    
    # 디렉토리 생성
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 모델 훈련
    model, best_model_path = train_model(config)

if __name__ == "__main__":
    main()
