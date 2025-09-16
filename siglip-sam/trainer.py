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
from torch.amp import GradScaler, autocast
import wandb
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    """메트릭 계산 - 최적 threshold 기반"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    if predictions.shape[1] == 2:
        # 이진 분류: 치매 클래스 확률 사용
        probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
        
        # ROC AUC 계산
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
        
        # 최적 threshold 찾기 (Youden's J statistic)
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(labels, probs)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        except:
            optimal_threshold = 0.5
        
        # 최적 threshold로 예측
        optimal_preds = (probs >= optimal_threshold).astype(int)
        
        # 기본 threshold (0.5)로도 예측
        default_preds = (probs >= 0.5).astype(int)
        
        # argmax 예측 (기존 방식)
        argmax_preds = np.argmax(predictions, axis=1)
        
        # 최적 threshold 기반 메트릭 (메인)
        optimal_accuracy = accuracy_score(labels, optimal_preds)
        optimal_precision, optimal_recall, optimal_f1, _ = precision_recall_fscore_support(
            labels, optimal_preds, average='weighted', zero_division=0
        )
        
        # 비교용 메트릭들
        default_accuracy = accuracy_score(labels, default_preds)
        default_precision, default_recall, default_f1, _ = precision_recall_fscore_support(
            labels, default_preds, average='weighted', zero_division=0
        )
        
        argmax_accuracy = accuracy_score(labels, argmax_preds)
        argmax_precision, argmax_recall, argmax_f1, _ = precision_recall_fscore_support(
            labels, argmax_preds, average='weighted', zero_division=0
        )
        
        return {
            # 메인 지표 (최적 threshold 기반)
            'accuracy': optimal_accuracy,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1': optimal_f1,
            'auc': auc,
            'optimal_threshold': optimal_threshold,
            
            # 비교 지표들
            'accuracy_default': default_accuracy,
            'precision_default': default_precision,
            'recall_default': default_recall,
            'f1_default': default_f1,
            
            'accuracy_argmax': argmax_accuracy,
            'precision_argmax': argmax_precision,
            'recall_argmax': argmax_recall,
            'f1_argmax': argmax_f1,
        }
    else:
        # 다중 분류
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

def plot_roc_curve(predictions, labels, title="ROC Curve", save_path=None):
    """ROC 곡선을 그리고 wandb에 로깅"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        # 확률값 추출 (이진 분류의 positive class 확률)
        if len(predictions.shape) > 1:
            probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
        else:
            probs = predictions
        
        # ROC 곡선 계산
        if len(np.unique(labels)) > 1:
            fpr, tpr, thresholds = roc_curve(labels, probs)
            auc_score = roc_auc_score(labels, probs)
            
            # ROC 곡선 그리기
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'{title} (AUC = {auc_score:.3f})', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 최적 임계값 표시
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
                    label=f'Optimal threshold = {optimal_threshold:.3f}')
            ax.legend(loc="lower right", fontsize=10)
            
            print(f"📊 ROC 곡선 생성 완료: AUC = {auc_score:.3f}")
            
        else:
            ax.text(0.5, 0.5, 'Cannot plot ROC curve\n(only one class present)', 
                    ha='center', va='center', fontsize=14)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            print("⚠️ ROC 곡선 생성 불가: 단일 클래스만 존재")
        
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 ROC 곡선 저장: {save_path}")
        
        # wandb에 로깅
        if wandb.run is not None:
            wandb.log({f"{title.lower().replace(' ', '_')}_plot": wandb.Image(fig)})
            print(f"📊 ROC 곡선 wandb 업로드: {title}")
        
        plt.close(fig)
        return fig
        
    except Exception as e:
        print(f"❌ ROC 곡선 생성 오류: {e}")
        plt.close(fig)
        return None

def plot_confusion_matrix(predictions, labels, title="Confusion Matrix", save_path=None):
    """Confusion Matrix를 그리고 wandb에 로깅"""
    try:
        # 예측값 변환
        if len(predictions.shape) > 1:
            preds = np.argmax(predictions, axis=1)
        else:
            preds = (predictions > 0.5).astype(int)
        
        # Confusion Matrix 계산
        cm = confusion_matrix(labels, preds)
        
        # 그래프 그리기
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Dementia'], 
                   yticklabels=['Normal', 'Dementia'])
        plt.title(f'{title}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion Matrix 저장: {save_path}")
        
        # wandb에 로깅
        if wandb.run is not None:
            wandb.log({f"{title.lower().replace(' ', '_')}_plot": wandb.Image(plt.gcf())})
            print(f"📊 Confusion Matrix wandb 업로드: {title}")
        
        plt.close()
        
        # 분류 리포트 출력
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"📊 {title} 요약:")
        print(f"   정확도: {accuracy:.4f}")
        print(f"   정상 → 정상: {cm[0,0]}, 정상 → 치매: {cm[0,1]}")
        print(f"   치매 → 정상: {cm[1,0]}, 치매 → 치매: {cm[1,1]}")
        
    except Exception as e:
        print(f"❌ Confusion Matrix 생성 오류: {e}")
        plt.close()

def train_epoch(model, train_loader, optimizer, config, scaler=None, use_mixed_precision=False):
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
        
        # SAM 옵티마이저 사용 시 (mixed precision 비활성화로 안정성 확보)
        if config.optimizer_type == "sam":
            # SAM은 mixed precision 없이 사용 (안정성을 위해)
            # 첫 번째 forward pass
            logits = model(batch)
            loss = model.compute_loss(logits, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.first_step(zero_grad=True)
            
            # 두 번째 forward pass
            logits = model(batch)
            loss = model.compute_loss(logits, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.second_step(zero_grad=True)
        
        else:
            # 일반 옵티마이저
            optimizer.zero_grad()
            
            if scaler and use_mixed_precision:
                with autocast('cuda'):
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

def evaluate(model, test_loader, config, use_mixed_precision=False, title_prefix="Test"):
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
            
            if use_mixed_precision:
                with autocast('cuda'):
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
    
    # ROC 곡선 그리기 및 wandb 업로드
    try:
        plot_roc_curve(
            predictions=np.array(all_predictions), 
            labels=np.array(all_labels), 
            title=f"{title_prefix} ROC Curve",
            save_path=os.path.join(config.output_dir, f"{title_prefix.lower()}_roc_curve.png")
        )
    except Exception as e:
        print(f"⚠️ ROC 곡선 생성 실패: {e}")
    
    # Confusion Matrix 그리기 및 wandb 업로드
    try:
        plot_confusion_matrix(
            predictions=np.array(all_predictions), 
            labels=np.array(all_labels), 
            title=f"{title_prefix} Confusion Matrix",
            save_path=os.path.join(config.output_dir, f"{title_prefix.lower()}_confusion_matrix.png")
        )
    except Exception as e:
        print(f"⚠️ Confusion Matrix 생성 실패: {e}")
    
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
    
    # 모델 생성
    print("모델 생성 중...")
    model = SigLIPSAMDementiaClassifier(config)
    model.to(config.device)
    
    # 옵티마이저 생성
    optimizer = model.create_optimizer(config)
    
    # 스케줄러 생성
    total_steps = len(train_loader) * config.num_epochs
    scheduler = model.create_scheduler(optimizer, config, total_steps)
    
    # Mixed precision 스케일러 (SAM 사용 시 비활성화)
    use_mixed_precision = config.mixed_precision and config.optimizer_type != "sam"
    scaler = GradScaler('cuda') if use_mixed_precision else None
    
    if config.optimizer_type == "sam" and config.mixed_precision:
        print("⚠️ SAM 옵티마이저 사용 시 Mixed Precision을 비활성화합니다 (안정성을 위해)")
        use_mixed_precision = False
    
    # wandb 설정
    setup_wandb(config)
    
    # 훈련 루프
    best_val_auc = 0.0
    best_model_path = None
    
    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        
        # 훈련
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, config, scaler, use_mixed_precision)
        
        # 검증
        val_loss, val_metrics = evaluate(model, val_loader, config, use_mixed_precision, title_prefix="Val")
        
        # 테스트 (매 에포크마다 참고용으로만)
        test_loss, test_metrics = evaluate(model, test_loader, config, use_mixed_precision, title_prefix="Test")
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 로깅 (최적 threshold 기반 메트릭 포함)
        wandb_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'train_auc': train_metrics['auc'],
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'],
            'test_loss': test_loss,
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_auc': test_metrics['auc'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # 최적 threshold 정보 추가
        if 'optimal_threshold' in val_metrics:
            wandb_log['val_optimal_threshold'] = val_metrics['optimal_threshold']
        if 'optimal_threshold' in test_metrics:
            wandb_log['test_optimal_threshold'] = test_metrics['optimal_threshold']
        
        # 비교 메트릭도 추가
        if 'accuracy_default' in val_metrics:
            wandb_log['val_accuracy_default_0.5'] = val_metrics['accuracy_default']
            wandb_log['val_accuracy_argmax'] = val_metrics['accuracy_argmax']
        if 'accuracy_default' in test_metrics:
            wandb_log['test_accuracy_default_0.5'] = test_metrics['accuracy_default']
            wandb_log['test_accuracy_argmax'] = test_metrics['accuracy_argmax']
        
        wandb.log(wandb_log)
        
        # 결과 출력 (최적 threshold 기반)
        print(f"훈련 - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"검증 - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"테스트 - Loss: {test_loss:.4f}, Acc: {test_metrics['accuracy']:.4f}, Prec: {test_metrics['precision']:.4f}, Rec: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
        
        # Threshold 정보 출력
        if 'optimal_threshold' in val_metrics:
            print(f"🎯 검증 최적 threshold: {val_metrics['optimal_threshold']:.3f}")
        if 'optimal_threshold' in test_metrics:
            print(f"🎯 테스트 최적 threshold: {test_metrics['optimal_threshold']:.3f}")
        
        # 베스트 모델 저장 (validation AUC 기준)
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_path = save_checkpoint(model, optimizer, epoch + 1, val_metrics, config, is_best=True)
            print(f"🏆 새로운 베스트 모델! Validation AUC: {best_val_auc:.4f}")
        
        # 정기 체크포인트 저장
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, config, is_best=False)
    
    print(f"\n=== 훈련 완료 ===")
    print(f"🏆 베스트 Validation AUC: {best_val_auc:.4f}")
    print(f"💾 베스트 모델: {best_model_path}")
    
    # 최종 테스트 (베스트 모델 로드해서 최종 평가)
    if best_model_path and os.path.exists(best_model_path):
        print("\n🔍 베스트 모델로 최종 평가 수행...")
        try:
            # 베스트 모델 로드
            checkpoint = torch.load(best_model_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 최종 테스트
            final_test_loss, final_test_metrics = evaluate(model, test_loader, config, use_mixed_precision)
            
            # 최종 결과 wandb 로깅
            wandb.log({
                'final_test_loss': final_test_loss,
                'final_test_accuracy': final_test_metrics['accuracy'],
                'final_test_f1': final_test_metrics['f1'],
                'final_test_auc': final_test_metrics['auc'],
            })
            
            print(f"🎯 최종 테스트 결과 (베스트 모델):")
            print(f"   Loss: {final_test_loss:.4f}")
            print(f"   AUC: {final_test_metrics['auc']:.4f}")
            print(f"   Accuracy: {final_test_metrics['accuracy']:.4f}")
            print(f"   Precision: {final_test_metrics['precision']:.4f}")
            print(f"   Recall: {final_test_metrics['recall']:.4f}")
            print(f"   F1: {final_test_metrics['f1']:.4f}")
            
            if 'optimal_threshold' in final_test_metrics:
                print(f"   최적 Threshold: {final_test_metrics['optimal_threshold']:.3f}")
                
                # Threshold 비교 출력
                if 'accuracy_default' in final_test_metrics:
                    print(f"\n📊 Threshold 비교:")
                    print(f"   최적 threshold ({final_test_metrics['optimal_threshold']:.3f}): Acc={final_test_metrics['accuracy']:.4f}")
                    print(f"   기본 threshold (0.500): Acc={final_test_metrics['accuracy_default']:.4f}")
                    print(f"   Argmax 방식: Acc={final_test_metrics['accuracy_argmax']:.4f}")
            
        except Exception as e:
            print(f"⚠️ 최종 평가 실패: {e}")
    
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
    
    # 경로 디버깅
    print(f"\n🔍 경로 디버깅:")
    print(f"  현재 작업 디렉토리: {os.getcwd()}")
    print(f"  config.data_dir: {config.data_dir}")
    print(f"  절대 경로: {os.path.abspath(config.data_dir)}")
    print(f"  경로 존재 여부: {os.path.exists(config.data_dir)}")
    
    # 디렉토리 생성
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 모델 훈련
    model, best_model_path = train_model(config)

if __name__ == "__main__":
    main()
