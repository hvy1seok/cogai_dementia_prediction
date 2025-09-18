"""
Text-only Model Training Script
Text-only (Gemma Encoder, multilingual joint) 훈련 스크립트
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Tuple
import json
from datetime import datetime

from config import TextOnlyConfig
from models import TextOnlyModel, compute_metrics, compute_language_specific_metrics
from data_processor import create_dataloaders, compute_class_weights

def train_epoch(model: nn.Module, 
                train_loader, 
                optimizer, 
                scheduler,
                device: str,
                config: TextOnlyConfig) -> Dict[str, float]:
    """한 에포크 훈련"""
    
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, batch in enumerate(train_loader):
        # 데이터 이동
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = model.compute_loss(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 예측 및 확률 계산
        if config.num_classes == 2:
            probs = torch.sigmoid(logits[:, 1] - logits[:, 0])
            preds = (probs > 0.5).long()
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        # gradient 분리하여 numpy 변환
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        # 확률 처리 - gradient 분리 및 예외 처리
        try:
            if config.num_classes == 2:
                all_probs.extend(probs.detach().cpu().numpy())
            else:
                all_probs.extend(probs[:, 1].detach().cpu().numpy())
        except Exception as e:
            print(f"⚠️ 확률 계산 중 오류: {e}")
            # 폴백: sigmoid 확률 사용
            prob_values = torch.sigmoid(logits).detach().cpu().numpy()
            if len(prob_values.shape) > 1 and prob_values.shape[1] > 1:
                all_probs.extend(prob_values[:, 1])
            else:
                all_probs.extend(prob_values.flatten())
    
    if scheduler:
        scheduler.step()
    
    # 지표 계산
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    metrics['loss'] = total_loss / len(train_loader)
    return metrics

def validate_epoch(model: nn.Module,
                   val_loader,
                   device: str,
                   config: TextOnlyConfig) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """검증"""
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_languages = []
    
    with torch.no_grad():
        for batch in val_loader:
            # 데이터 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            languages = batch['language']
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = model.compute_loss(logits, labels)
            
            total_loss += loss.item()
            
            # 예측 및 확률 계산
            if config.num_classes == 2:
                probs = torch.sigmoid(logits[:, 1] - logits[:, 0])
                preds = (probs > 0.5).long()
            else:
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
            
            # gradient 분리하여 numpy 변환
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # 확률 처리 - gradient 분리 및 예외 처리
            try:
                if config.num_classes == 2:
                    all_probs.extend(probs.detach().cpu().numpy())
                else:
                    all_probs.extend(probs[:, 1].detach().cpu().numpy())
            except Exception as e:
                print(f"⚠️ 검증 확률 계산 중 오류: {e}")
                # 폴백: sigmoid 확률 사용
                prob_values = torch.sigmoid(logits).detach().cpu().numpy()
                if len(prob_values.shape) > 1 and prob_values.shape[1] > 1:
                    all_probs.extend(prob_values[:, 1])
                else:
                    all_probs.extend(prob_values.flatten())
            all_languages.extend(languages)
    
    # 전체 지표 계산
    overall_metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    overall_metrics['loss'] = total_loss / len(val_loader)
    
    # 언어별 지표 계산
    language_metrics = compute_language_specific_metrics(
        all_labels, all_preds, all_probs, all_languages
    )
    
    return overall_metrics, language_metrics

def compute_target_languages_avg_macro_f1(language_metrics: Dict[str, Dict[str, float]], 
                                          target_languages: List[str]) -> float:
    """타겟 언어들의 평균 Macro F1 계산"""
    
    valid_macro_f1s = []
    for lang in target_languages:
        if lang in language_metrics and 'macro_f1' in language_metrics[lang]:
            if language_metrics[lang]['macro_f1'] > 0:
                valid_macro_f1s.append(language_metrics[lang]['macro_f1'])
                print(f"  {lang} Macro F1: {language_metrics[lang]['macro_f1']:.4f}")
    
    if valid_macro_f1s:
        avg_macro_f1 = np.mean(valid_macro_f1s)
        print(f"  평균 Macro F1 ({len(valid_macro_f1s)}개 언어): {avg_macro_f1:.4f}")
        return avg_macro_f1
    else:
        print("  ⚠️ 유효한 언어별 Macro F1가 없어 전체 Macro F1 사용")
        return 0.0

def train_model(config: TextOnlyConfig) -> Tuple[nn.Module, str]:
    """모델 훈련"""
    
    print("🔥 Text-only (Gemma Encoder) 모델 훈련 시작")
    print(f"  언어: {config.languages}")
    print(f"  텍스트 인코더: {config.text_encoder}")
    print(f"  배치 크기: {config.batch_size}")
    print(f"  학습률: {config.learning_rate}")
    print(f"  에포크: {config.num_epochs}")
    print(f"  Early Stopping: {config.early_stopping_patience} epochs")
    
    # 디바이스 설정
    device = config.device
    print(f"  디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 데이터로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        config, mode="text_only", tokenizer=tokenizer
    )
    
    # 모델 생성
    model = TextOnlyModel(config).to(device)
    
    # 클래스 가중치 계산 및 적용
    if config.auto_class_weights:
        class_weights = compute_class_weights(train_loader.dataset, config)
        if hasattr(model.criterion, 'alpha'):
            model.criterion.alpha = torch.tensor(class_weights[1] / class_weights[0]).to(device)
    
    # 옵티마이저 및 스케줄러
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    # Wandb 초기화
    if config.log_wandb:
        wandb.init(
            project="dementia-prediction-control-groups",
            name=f"text-only-{'-'.join(config.languages)}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config.__dict__
        )
    
    # 훈련 루프
    best_metric = 0.0
    patience_counter = 0
    best_model_path = None
    
    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        
        # 훈련
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        print(f"훈련 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"Macro F1: {train_metrics['macro_f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        
        # 검증
        val_metrics, lang_metrics = validate_epoch(model, val_loader, device, config)
        print(f"검증 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"Macro F1: {val_metrics['macro_f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # 언어별 성능
        for lang, metrics in lang_metrics.items():
            print(f"  {lang} - Acc: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}, "
                  f"AUC: {metrics['auc']:.4f}")
        
        # 베스트 모델 선택 기준 계산
        if config.best_model_metric == "avg_lang_macro_f1":
            current_metric = compute_target_languages_avg_macro_f1(lang_metrics, config.target_languages)
        else:
            current_metric = val_metrics['macro_f1']
        
        # Wandb 로깅
        if config.log_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_macro_f1': train_metrics['macro_f1'],
                'train_auc': train_metrics['auc'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_auc': val_metrics['auc'],
                'current_metric': current_metric
            }
            
            # 언어별 지표 추가
            for lang, metrics in lang_metrics.items():
                for metric_name, value in metrics.items():
                    log_dict[f'{lang}_{metric_name}'] = value
            
            wandb.log(log_dict)
        
        # 베스트 모델 저장
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            
            # 모델 저장
            if config.save_checkpoints:
                os.makedirs(config.output_dir, exist_ok=True)
                best_model_path = os.path.join(config.output_dir, 'best_text_only_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.__dict__,
                    'epoch': epoch + 1,
                    'best_metric': best_metric,
                    'val_metrics': val_metrics,
                    'lang_metrics': lang_metrics
                }, best_model_path)
                print(f"✅ 베스트 모델 저장: {best_model_path}")
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"🛑 Early Stopping: {config.early_stopping_patience} epochs 동안 성능 향상 없음")
            break
    
    # 최종 테스트 (있는 경우)
    if test_loader and best_model_path:
        print("\n=== 최종 테스트 ===")
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
        test_metrics, test_lang_metrics = validate_epoch(model, test_loader, device, config)
        
        print(f"테스트 - Acc: {test_metrics['accuracy']:.4f}, Macro F1: {test_metrics['macro_f1']:.4f}, "
              f"AUC: {test_metrics['auc']:.4f}")
        
        for lang, metrics in test_lang_metrics.items():
            print(f"  {lang} - Acc: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}, "
                  f"AUC: {metrics['auc']:.4f}")
        
        if config.log_wandb:
            test_log_dict = {'test_accuracy': test_metrics['accuracy'],
                           'test_macro_f1': test_metrics['macro_f1'],
                           'test_auc': test_metrics['auc']}
            
            for lang, metrics in test_lang_metrics.items():
                for metric_name, value in metrics.items():
                    test_log_dict[f'test_{lang}_{metric_name}'] = value
            
            wandb.log(test_log_dict)
    
    if config.log_wandb:
        wandb.finish()
    
    print(f"\n✅ Text-only 모델 훈련 완료! 베스트 {config.best_model_metric}: {best_metric:.4f}")
    return model, best_model_path

def main():
    parser = argparse.ArgumentParser(description="Text-only Model Training")
    
    # 데이터 관련
    parser.add_argument('--data_dir', type=str, default='../../training_dset', help='데이터 디렉토리')
    parser.add_argument('--languages', nargs='+', default=['English', 'Mandarin'], help='사용할 언어')
    
    # 훈련 관련
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=100, help='에포크 수')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    # 모델 관련
    parser.add_argument('--text_encoder', type=str, default='google/gemma-2b', help='텍스트 인코더 모델명')
    parser.add_argument('--use_cls_token', action='store_true', help='[CLS] 토큰 사용')
    
    # 손실 함수 관련
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'bce'], help='손실 함수')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--auto_class_weights', action='store_true', help='자동 클래스 가중치')
    
    # 평가 관련
    parser.add_argument('--best_model_metric', type=str, default='avg_lang_macro_f1', 
                       choices=['val_macro_f1', 'avg_lang_macro_f1'], help='베스트 모델 선택 기준')
    parser.add_argument('--target_languages', nargs='+', help='타겟 언어 (평균 계산용)')
    parser.add_argument('--split_by_patient', type=str, default='true', choices=['true', 'false'], 
                       help='환자 단위 분할 여부')
    
    # 출력 관련
    parser.add_argument('--output_dir', type=str, default='../modules/outputs/controlgroups', help='출력 디렉토리')
    parser.add_argument('--no_wandb', action='store_true', help='Wandb 비활성화')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = TextOnlyConfig(
        data_dir=args.data_dir,
        languages=args.languages,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        text_encoder=args.text_encoder,
        use_cls_token=args.use_cls_token,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        auto_class_weights=args.auto_class_weights,
        best_model_metric=args.best_model_metric,
        target_languages=args.target_languages or args.languages,
        split_by_patient=(args.split_by_patient.lower() == 'true'),
        output_dir=args.output_dir,
        log_wandb=(not args.no_wandb)
    )
    
    # 훈련 실행
    train_model(config)

if __name__ == "__main__":
    main()
