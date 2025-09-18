#!/usr/bin/env python3
"""
SigLIP-Text-Only 모델 훈련 스크립트
SigLIP의 텍스트 인코더 + Gemma 토크나이저를 사용하는 텍스트 전용 대조군
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TextOnlyConfig
from siglip_control_models import SigLIPTextOnlyModel, compute_metrics, compute_language_specific_metrics
from data_processor import create_dataloaders
from transformers import AutoTokenizer

def compute_class_weights(train_labels: List[int]) -> np.ndarray:
    """클래스 가중치 계산"""
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    return class_weights

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

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                config: TextOnlyConfig, scheduler=None) -> Tuple[float, Dict[str, float]]:
    """한 에포크 훈련"""
    
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_languages = []
    
    for batch in tqdm(train_loader, desc="훈련"):
        optimizer.zero_grad()
        
        # 입력 준비
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['label'].to(config.device)
        languages = batch['language']
        
        # 순전파
        logits = model(input_ids, attention_mask)
        loss = model.compute_loss(logits, labels)
        
        # 역전파
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
        all_languages.extend(languages)
        
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
    
    # 메트릭 계산
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels), np.array(all_probs))
    
    return avg_loss, metrics

def validate_epoch(model: nn.Module, val_loader: DataLoader, config: TextOnlyConfig) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """한 에포크 검증"""
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_languages = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="검증"):
            # 입력 준비
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['label'].to(config.device)
            languages = batch['language']
            
            # 순전파
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
            
            # numpy 변환
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_languages.extend(languages)
            
            # 확률 처리 - 예외 처리
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
    
    # 메트릭 계산
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels), np.array(all_probs))
    language_metrics = compute_language_specific_metrics(
        np.array(all_preds), np.array(all_labels), np.array(all_probs), 
        all_languages, metrics['optimal_threshold']
    )
    
    return avg_loss, metrics, language_metrics

def train_model(config: TextOnlyConfig) -> Tuple[nn.Module, str]:
    """모델 훈련"""
    
    print("🔥 SigLIP-Text-Only 모델 훈련 시작")
    print(f"  언어: {config.languages}")
    print(f"  SigLIP 모델: {config.siglip_model}")
    print(f"  텍스트 토크나이저: {config.text_tokenizer}")
    print(f"  배치 크기: {config.batch_size}")
    print(f"  학습률: {config.learning_rate}")
    print(f"  에포크: {config.num_epochs}")
    print(f"  Early Stopping: {config.early_stopping_patience} epochs")
    
    # 멀티 GPU 디바이스 설정
    device = config.device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"  사용 가능한 GPU: {device_count}개")
        print(f"  주 디바이스: {device}")
        if device_count > 1:
            print(f"  멀티 GPU 모드: GPU 0-{device_count-1} 사용")
    else:
        print(f"  디바이스: {device}")
    
    # Gemma 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 데이터로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        config, mode="text_only", tokenizer=tokenizer
    )
    
    # 모델 생성 및 멀티 GPU 설정
    model = SigLIPTextOnlyModel(config).to(device)
    
    # 멀티 GPU 사용 가능 시 DataParallel 적용
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"  🔥 멀티 GPU 훈련 활성화: {torch.cuda.device_count()}개 GPU 사용")
        model = nn.DataParallel(model)
        use_dataparallel = True
    else:
        use_dataparallel = False
    
    # 클래스 가중치 계산 및 적용
    if config.auto_class_weights:
        # DataParallel 사용 시 train_loader.dataset.data 접근 방식 수정
        if hasattr(train_loader.dataset, 'data'):
            train_labels = [item['label'] for item in train_loader.dataset.data]
        else:
            # Subset인 경우
            train_labels = [train_loader.dataset.dataset.data[i]['label'] for i in train_loader.dataset.indices]
        
        class_weights = compute_class_weights(train_labels)
        print(f"📊 자동 클래스 가중치: {class_weights}")
        
        # 모델에 클래스 가중치 적용
        if use_dataparallel:
            model.module.setup_loss_function(class_weights)
        else:
            model.setup_loss_function(class_weights)
    
    # 옵티마이저 및 스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    # Wandb 초기화
    if config.log_wandb:
        wandb.init(
            project="dementia-controlgroups",
            name=f"siglip-text-only-{'_'.join(config.languages)}",
            config={
                "model_type": "siglip_text_only",
                "siglip_model": config.siglip_model,
                "text_tokenizer": config.text_tokenizer,
                "languages": config.languages,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "early_stopping_patience": config.early_stopping_patience,
                "loss_type": config.loss_type,
                "auto_class_weights": config.auto_class_weights,
                "best_model_metric": config.best_model_metric,
                "target_languages": config.target_languages,
                "split_by_patient": config.split_by_patient
            }
        )
    
    # 훈련 루프
    best_metric = 0.0
    patience_counter = 0
    best_model_path = None
    
    for epoch in range(config.num_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{config.num_epochs}")
        
        # 훈련
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, config, scheduler)
        
        # 검증
        val_loss, val_metrics, val_lang_metrics = validate_epoch(model, val_loader, config)
        
        # 베스트 모델 선택 기준에 따른 현재 메트릭 계산
        if config.best_model_metric == "val_macro_f1":
            current_metric = val_metrics['macro_f1']
        elif config.best_model_metric == "avg_lang_macro_f1":
            current_metric = compute_target_languages_avg_macro_f1(val_lang_metrics, config.target_languages)
        else:  # val_auc (기본값)
            current_metric = val_metrics['auc']
        
        # 결과 출력
        print(f"  훈련 손실: {train_loss:.4f}")
        print(f"  검증 손실: {val_loss:.4f}")
        print(f"  훈련 Acc: {train_metrics['accuracy']:.4f}, Macro F1: {train_metrics['macro_f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  검증 Acc: {val_metrics['accuracy']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  현재 {config.best_model_metric}: {current_metric:.4f}")
        
        # 언어별 결과 출력
        print("  📊 언어별 검증 성능:")
        for lang, metrics in val_lang_metrics.items():
            print(f"    {lang} - Acc: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}, AUC: {metrics['auc']:.4f}")
        
        # Wandb 로깅
        if config.log_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_metrics['accuracy'],
                'train_macro_f1': train_metrics['macro_f1'],
                'train_auc': train_metrics['auc'],
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_auc': val_metrics['auc'],
                'current_metric': current_metric,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            # 언어별 메트릭 추가
            for lang, metrics in val_lang_metrics.items():
                for metric_name, value in metrics.items():
                    log_dict[f'val_{lang}_{metric_name}'] = value
            
            wandb.log(log_dict)
        
        # 베스트 모델 저장
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            
            # 모델 저장
            os.makedirs(config.output_dir, exist_ok=True)
            best_model_path = os.path.join(config.output_dir, f"best_siglip_text_only_{'_'.join(config.languages)}.pt")
            
            model_to_save = model.module if use_dataparallel else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'config': config
            }, best_model_path)
            
            print(f"  ✅ 베스트 모델 저장: {config.best_model_metric} = {best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ Early Stopping: {patience_counter}/{config.early_stopping_patience}")
        
        # Early Stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"  🛑 Early Stopping 발동! 베스트 {config.best_model_metric}: {best_metric:.4f}")
            break
    
    # 베스트 모델 로드
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n📥 베스트 모델 로드: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model_to_load = model.module if use_dataparallel else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    # 최종 테스트
    if test_loader is not None:
        print(f"\n🧪 최종 테스트 평가")
        test_loss, test_metrics, test_lang_metrics = validate_epoch(model, test_loader, config)
        
        print(f"📊 최종 테스트 성능:")
        print(f"  전체 - Acc: {test_metrics['accuracy']:.4f}, Macro F1: {test_metrics['macro_f1']:.4f}, "
              f"AUC: {test_metrics['auc']:.4f}")
        
        print(f"📊 언어별 테스트 성능:")
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
    
    print(f"\n✅ SigLIP-Text-Only 모델 훈련 완료! 베스트 {config.best_model_metric}: {best_metric:.4f}")
    return model, best_model_path

def main():
    parser = argparse.ArgumentParser(description="SigLIP-Text-Only 모델 훈련")
    
    # 데이터 관련
    parser.add_argument('--data_dir', type=str, default='../../training_dset', help='데이터 디렉토리')
    parser.add_argument('--languages', nargs='+', default=['English', 'Mandarin'], help='사용할 언어들')
    
    # 훈련 관련
    parser.add_argument('--batch_size', type=int, default=10, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=100, help='에포크 수')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    
    # 모델 관련
    parser.add_argument('--siglip_model', type=str, default='google/siglip-base-patch16-224', help='SigLIP 모델')
    parser.add_argument('--text_tokenizer', type=str, default='google/gemma-2b', help='텍스트 토크나이저')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'bce'], help='손실 함수')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--auto_class_weights', type=str, default='true', choices=['true', 'false'], help='자동 클래스 가중치')
    parser.add_argument('--best_model_metric', type=str, default='avg_lang_macro_f1', 
                        choices=['val_macro_f1', 'avg_lang_macro_f1', 'val_auc'], help='베스트 모델 선택 기준')
    parser.add_argument('--target_languages', nargs='*', help='타겟 언어들 (기본값: languages와 동일)')
    parser.add_argument('--split_by_patient', type=str, default='true', choices=['true', 'false'], help='환자 단위 분할')
    
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
        siglip_model=args.siglip_model,
        text_tokenizer=args.text_tokenizer,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        auto_class_weights=(args.auto_class_weights.lower() == 'true'),
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
