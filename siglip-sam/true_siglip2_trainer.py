"""
진정한 SigLIP2 치매 진단 모델 트레이너
- EMA Teacher-Student 학습
- Multi-Loss: SILC/TIPS + Sigmoid + LoCa + Classification
- Caption generation 및 dense captioning
- SAM 옵티마이저 지원
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import wandb
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor, AutoTokenizer
from typing import Dict, List

from config import SigLIPSAMConfig
from true_siglip2_model import TrueSigLIP2DementiaClassifier
from data_processor import create_dataloaders
from sam_optimizer import SAM

def setup_wandb(config: SigLIPSAMConfig):
    """wandb 설정 - True SigLIP2 전용"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 언어 정보
    if config.cross_lingual_mode:
        train_langs = "_".join(config.train_languages) if config.train_languages else "Unknown"
        test_langs = "_".join(config.test_languages) if config.test_languages else "Unknown"
        lang_info = f"CrossLingual_Train{train_langs}_Test{test_langs}"
    else:
        lang_info = "_".join(config.languages) if len(config.languages) <= 2 else f"{len(config.languages)}langs"
    
    model_info = "TrueSigLIP2"
    loss_info = f"{config.loss_type}_MultiLoss"
    opt_info = config.optimizer_type
    
    run_name = f"true-siglip2_{lang_info}_{model_info}_{loss_info}_{opt_info}_bs{config.batch_size}_lr{config.learning_rate}_{timestamp}"
    
    wandb.init(
        project="dementia-prediction-true-siglip2",
        name=run_name,
        tags=[
            f"loss_{config.loss_type}",
            f"optimizer_{config.optimizer_type}",
            f"batch_size_{config.batch_size}",
            f"languages_{len(config.languages)}",
            "cross_lingual" if config.cross_lingual_mode else "standard",
            "true_siglip2",
            "ema_teacher_student",
            "multi_loss",
            "caption_generation"
        ],
        config={
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "languages": config.languages,
            "loss_type": config.loss_type,
            "optimizer_type": config.optimizer_type,
            "sam_rho": config.sam_rho,
            "ema_momentum": getattr(config, 'ema_momentum', 0.999),
            "silc_weight": getattr(config, 'silc_weight', 0.2),
            "sigmoid_weight": getattr(config, 'sigmoid_weight', 1.0),
            "loca_weight": getattr(config, 'loca_weight', 1.0),
            "classification_weight": getattr(config, 'classification_weight', 1.0),
            "cross_lingual_mode": config.cross_lingual_mode,
            "train_languages": config.train_languages,
            "test_languages": config.test_languages,
        }
    )

def compute_metrics(predictions, labels, languages=None):
    """메트릭 계산 - 기존 코드 재사용"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    if predictions.shape[1] == 2:
        probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
        
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
        
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(labels, probs)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        except:
            optimal_threshold = 0.5
        
        optimal_preds = (probs >= optimal_threshold).astype(int)
        default_preds = (probs >= 0.5).astype(int)
        argmax_preds = np.argmax(predictions, axis=1)
        
        optimal_accuracy = accuracy_score(labels, optimal_preds)
        optimal_precision, optimal_recall, optimal_f1, _ = precision_recall_fscore_support(
            labels, optimal_preds, average='weighted', zero_division=0
        )
        
        default_accuracy = accuracy_score(labels, default_preds)
        default_precision, default_recall, default_f1, _ = precision_recall_fscore_support(
            labels, default_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': optimal_accuracy,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1': optimal_f1,
            'auc': auc,
            'optimal_threshold': optimal_threshold,
            'accuracy_default': default_accuracy,
            'precision_default': default_precision,
            'recall_default': default_recall,
            'f1_default': default_f1,
        }
        
        if languages is not None:
            from trainer import compute_language_specific_metrics
            language_metrics = compute_language_specific_metrics(probs, labels, languages, optimal_threshold)
            metrics.update(language_metrics)
        
        return metrics
    else:
        preds = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': 0.0,
            'optimal_threshold': 0.5
        }

def train_epoch(model, train_loader, optimizer, config, scaler=None, use_mixed_precision=False):
    """한 에포크 훈련 - True SigLIP2 Multi-Loss"""
    model.train()
    
    total_loss = 0.0
    total_classification_loss = 0.0
    total_silc_loss = 0.0
    total_sigmoid_loss = 0.0
    total_loca_loss = 0.0
    
    all_predictions = []
    all_labels = []
    loss_components_sum = {}
    
    for batch_idx, batch in enumerate(train_loader):
        # GPU로 이동
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(config.device)
        
        # 환자 ID 추출
        if 'patient_id' in batch:
            patient_ids = batch['patient_id'] if isinstance(batch['patient_id'], list) else [batch['patient_id']]
        else:
            if 'language' in batch:
                languages = batch['language'] if isinstance(batch['language'], list) else ['Unknown'] * len(batch['labels'])
                patient_ids = [f"{lang}_{i // 2}" for i, lang in enumerate(languages)]
            else:
                patient_ids = [f"patient_{i // 2}" for i in range(len(batch['labels']))]
        
        # Caption targets (임시로 None - 실제 구현시 텍스트에서 추출)
        caption_targets = None  # TODO: 실제 caption 데이터 추가시 구현
        
        # SAM 옵티마이저 사용 시
        if config.optimizer_type == "sam":
            # 첫 번째 forward pass
            model_outputs = model(batch, return_embeddings=True, training=True)
            loss_dict = model.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
            loss = loss_dict['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.first_step(zero_grad=True)
            
            # 두 번째 forward pass
            model_outputs = model(batch, return_embeddings=True, training=True)
            loss_dict = model.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
            loss = loss_dict['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.second_step(zero_grad=True)
            
            # EMA Teacher 업데이트
            model.update_teacher()
        
        else:
            # 일반 옵티마이저
            optimizer.zero_grad()
            
            if scaler and use_mixed_precision:
                with autocast('cuda'):
                    model_outputs = model(batch, return_embeddings=True, training=True)
                    loss_dict = model.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
                    loss = loss_dict['total_loss']
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                model_outputs = model(batch, return_embeddings=True, training=True)
                loss_dict = model.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            
            # EMA Teacher 업데이트
            model.update_teacher()
        
        # 메트릭 수집
        logits = model_outputs['classification_logits']
        total_loss += loss.item()
        
        # Loss components 수집
        for key, value in loss_dict['loss_components'].items():
            if key not in loss_components_sum:
                loss_components_sum[key] = 0.0
            if isinstance(value, torch.Tensor):
                loss_components_sum[key] += value.item()
            else:
                loss_components_sum[key] += value
        
        all_predictions.extend(logits.detach().cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
        
        # 로깅
        if batch_idx % config.log_interval == 0:
            log_msg = f'Train Batch {batch_idx}/{len(train_loader)}: Total Loss = {loss.item():.4f}'
            
            # 주요 loss components 출력
            if 'classification_loss' in loss_dict['loss_components']:
                log_msg += f', Cls = {loss_dict["loss_components"]["classification_loss"]:.4f}'
            if 'silc_silc_tips_loss' in loss_dict['loss_components']:
                log_msg += f', SILC = {loss_dict["loss_components"]["silc_silc_tips_loss"]:.4f}'
            if 'sigmoid_contrastive_loss' in loss_dict['loss_components']:
                log_msg += f', Sigmoid = {loss_dict["loss_components"]["sigmoid_contrastive_loss"]:.4f}'
            if 'loca_loca_loss' in loss_dict['loss_components']:
                log_msg += f', LoCa = {loss_dict["loss_components"]["loca_loca_loss"]:.4f}'
            
            print(log_msg)
            
            # wandb 로깅
            wandb_log = {
                'train_batch_total_loss': loss.item(),
                'train_step': batch_idx
            }
            
            # Loss components 추가
            for key, value in loss_dict['loss_components'].items():
                if isinstance(value, torch.Tensor):
                    wandb_log[f'train_batch_{key}'] = value.item()
                else:
                    wandb_log[f'train_batch_{key}'] = value
                    
            wandb.log(wandb_log)
    
    # 에포크 메트릭 계산
    num_batches = len(train_loader)
    avg_total_loss = total_loss / num_batches
    
    # Loss components 평균 계산
    avg_loss_components = {}
    for key, value in loss_components_sum.items():
        avg_loss_components[key] = value / num_batches
    
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    metrics.update(avg_loss_components)
    
    return avg_total_loss, metrics

def evaluate(model, test_loader, config, use_mixed_precision=False, title_prefix="Test"):
    """모델 평가 - True SigLIP2"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_languages = []
    loss_components_sum = {}
    
    with torch.no_grad():
        for batch in test_loader:
            # GPU로 이동
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.device)
            
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
            
            if use_mixed_precision:
                with autocast('cuda'):
                    model_outputs = model(batch, return_embeddings=True, training=False)
                    loss_dict = model.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
                    loss = loss_dict['total_loss']
            else:
                model_outputs = model(batch, return_embeddings=True, training=False)
                loss_dict = model.compute_loss(model_outputs, batch['labels'], patient_ids, caption_targets)
                loss = loss_dict['total_loss']
            
            logits = model_outputs['classification_logits']
            
            # 메트릭 수집
            total_loss += loss.item()
            
            # Loss components 수집
            for key, value in loss_dict['loss_components'].items():
                if key not in loss_components_sum:
                    loss_components_sum[key] = 0.0
                if isinstance(value, torch.Tensor):
                    loss_components_sum[key] += value.item()
                else:
                    loss_components_sum[key] += value
            
            all_predictions.extend(logits.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            # 언어 정보 수집
            if 'language' in batch:
                if isinstance(batch['language'], list):
                    all_languages.extend(batch['language'])
                else:
                    all_languages.extend(['Unknown'] * len(batch['labels']))
            else:
                all_languages.extend(['Unknown'] * len(batch['labels']))
    
    # 메트릭 계산
    num_batches = len(test_loader)
    avg_total_loss = total_loss / num_batches
    
    # Loss components 평균 계산
    avg_loss_components = {}
    for key, value in loss_components_sum.items():
        avg_loss_components[key] = value / num_batches
    
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels), all_languages)
    metrics.update(avg_loss_components)
    
    # ROC 곡선 및 Confusion Matrix 생성 (기존 코드 재사용)
    try:
        from trainer import plot_roc_curve, plot_confusion_matrix
        plot_roc_curve(
            predictions=np.array(all_predictions), 
            labels=np.array(all_labels), 
            title=f"{title_prefix} ROC Curve",
            save_path=os.path.join(config.output_dir, f"{title_prefix.lower()}_roc_curve.png")
        )
        
        plot_confusion_matrix(
            predictions=np.array(all_predictions), 
            labels=np.array(all_labels), 
            title=f"{title_prefix} Confusion Matrix",
            save_path=os.path.join(config.output_dir, f"{title_prefix.lower()}_confusion_matrix.png")
        )
    except Exception as e:
        print(f"⚠️ 시각화 생성 실패: {e}")
    
    return avg_total_loss, metrics

def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    filename = f"true_siglip2_checkpoint_epoch_{epoch:03d}_auc_{metrics['auc']:.3f}.pt"
    if is_best:
        filename = f"true_siglip2_best_model_auc_{metrics['auc']:.3f}_epoch_{epoch:03d}.pt"
    
    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"💾 True SigLIP2 체크포인트 저장: {filepath}")
    
    return filepath

def train_model(config: SigLIPSAMConfig):
    """메인 훈련 함수 - True SigLIP2"""
    print("=== 진정한 SigLIP2 치매 진단 모델 훈련 시작 ===")
    
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
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        processor=processor,
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
    model = TrueSigLIP2DementiaClassifier(config)
    model.to(config.device)
    
    # 클래스 가중치 계산 및 손실 함수 설정
    from data_processor import compute_class_weights
    
    if hasattr(train_loader.dataset, 'dataset'):
        original_dataset = train_loader.dataset.dataset
    else:
        original_dataset = train_loader.dataset
    
    class_weights = compute_class_weights(original_dataset, config)
    model.setup_loss_function(class_weights)
    
    # 옵티마이저 생성
    optimizer = model.create_optimizer(config)
    
    # 스케줄러 생성
    total_steps = len(train_loader) * config.num_epochs
    scheduler = model.create_scheduler(optimizer, config, total_steps)
    
    # Mixed precision 스케일러 (SAM 사용 시 비활성화)
    use_mixed_precision = config.mixed_precision and config.optimizer_type != "sam"
    scaler = GradScaler('cuda') if use_mixed_precision else None
    
    if config.optimizer_type == "sam" and config.mixed_precision:
        print("⚠️ SAM 옵티마이저 사용 시 Mixed Precision을 비활성화합니다")
        use_mixed_precision = False
    
    # wandb 설정
    setup_wandb(config)
    
    # 훈련 루프
    best_val_auc = 0.0
    best_model_path = None
    early_stopping_patience = getattr(config, 'early_stopping_patience', 15)
    epochs_without_improvement = 0
    
    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        
        # 훈련
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, config, scaler, use_mixed_precision)
        
        # 검증
        val_loss, val_metrics = evaluate(model, val_loader, config, use_mixed_precision, title_prefix="Val")
        
        # 테스트 (참고용)
        test_loss, test_metrics = evaluate(model, test_loader, config, use_mixed_precision, title_prefix="Test")
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # wandb 로깅
        wandb_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_auc': train_metrics['auc'],
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_auc': val_metrics['auc'],
            'test_loss': test_loss,
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Multi-Loss components 추가
        for prefix, metrics_dict in [('train', train_metrics), ('val', val_metrics), ('test', test_metrics)]:
            for key, value in metrics_dict.items():
                if any(loss_type in key for loss_type in ['classification_loss', 'silc_', 'sigmoid_', 'loca_']):
                    wandb_log[f'{prefix}_{key}'] = value
        
        # 언어별 메트릭 추가 (테스트에서만)
        for key, value in test_metrics.items():
            if any(lang in key for lang in ['English', 'Greek', 'Spanish', 'Mandarin']):
                wandb_log[f'test_{key}'] = value
        
        wandb.log(wandb_log)
        
        # 결과 출력
        print(f"훈련 - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"검증 - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"테스트 - Loss: {test_loss:.4f}, Acc: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")
        
        # 베스트 모델 저장 및 Early Stopping
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_path = save_checkpoint(model, optimizer, epoch + 1, val_metrics, config, is_best=True)
            epochs_without_improvement = 0
            print(f"🏆 새로운 베스트 모델! Validation AUC: {best_val_auc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"⏳ 개선 없음: {epochs_without_improvement}/{early_stopping_patience} epochs")
        
        # Early Stopping 체크
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n🛑 Early Stopping! {early_stopping_patience} epochs 동안 validation AUC 개선 없음")
            print(f"🏆 최종 베스트 Validation AUC: {best_val_auc:.4f}")
            break
        
        # 정기 체크포인트 저장
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, config, is_best=False)
    
    print(f"\n=== 진정한 SigLIP2 훈련 완료 ===")
    print(f"🏆 베스트 Validation AUC: {best_val_auc:.4f}")
    print(f"💾 베스트 모델: {best_model_path}")
    
    # wandb 종료
    wandb.finish()
    
    return model, best_model_path

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="진정한 SigLIP2 치매 진단 모델 훈련")
    
    # 기본 설정 (기존 trainer.py와 동일한 인터페이스)
    parser.add_argument("--data_dir", type=str, default="../../training_dset", help="데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default="../modules/outputs/siglip-sam/true-siglip2", help="출력 디렉토리")
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
    parser.add_argument("--loss_type", type=str, default="focal",
                       choices=["cross_entropy", "focal", "bce"],
                       help="손실 함수 타입")
    parser.add_argument("--focal_alpha", type=float, default=1.0, help="Focal Loss alpha 파라미터")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma 파라미터")
    parser.add_argument("--auto_class_weights", action="store_true", help="클래스 불균형 자동 보정")
    
    # 옵티마이저 선택 옵션
    parser.add_argument("--optimizer_type", type=str, default="sam",
                       choices=["adamw", "sam"],
                       help="옵티마이저 타입")
    parser.add_argument("--sam_rho", type=float, default=0.05, help="SAM rho 파라미터")
    parser.add_argument("--sam_adaptive", action="store_true", help="Adaptive SAM 사용")
    
    # True SigLIP2 전용 옵션
    parser.add_argument("--ema_momentum", type=float, default=0.999, help="EMA Teacher momentum")
    parser.add_argument("--silc_weight", type=float, default=0.2, help="SILC/TIPS Loss 가중치")
    parser.add_argument("--sigmoid_weight", type=float, default=1.0, help="Sigmoid Loss 가중치")
    parser.add_argument("--loca_weight", type=float, default=1.0, help="LoCa Loss 가중치")
    parser.add_argument("--classification_weight", type=float, default=1.0, help="Classification Loss 가중치")
    
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
    config.auto_class_weights = args.auto_class_weights
    
    # 옵티마이저 설정
    if args.optimizer_type:
        config.optimizer_type = args.optimizer_type
    if args.sam_rho:
        config.sam_rho = args.sam_rho
    config.sam_adaptive = args.sam_adaptive
    
    # True SigLIP2 설정
    config.ema_momentum = args.ema_momentum
    config.silc_weight = args.silc_weight
    config.sigmoid_weight = args.sigmoid_weight
    config.loca_weight = args.loca_weight
    config.classification_weight = args.classification_weight
    
    # 언어 파서 설정 (기존 trainer.py와 동일)
    if args.parser == "cross_lingual":
        config.cross_lingual_mode = True
        config.train_languages = args.train_languages
        config.test_languages = args.test_languages
        
        train_langs_str = "_".join(args.train_languages)
        test_langs_str = "_".join(args.test_languages)
        config.output_dir = f"{config.output_dir}/CrossLingual_Train_{train_langs_str}_Test_{test_langs_str}"
        config.checkpoint_dir = f"{config.output_dir}/checkpoints"
        
        print("🌍 Cross-Lingual 모드 활성화")
        print(f"  훈련 언어: {args.train_languages}")
        print(f"  테스트 언어: {args.test_languages}")
        
        config.languages = args.train_languages + args.test_languages
        
    elif args.parser == "all":
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
    model, best_model_path = train_model(config)

if __name__ == "__main__":
    main()
