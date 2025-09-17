import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import os
import argparse
from pathlib import Path

from dataset_multilingual import (
    prepare_multilingual_dataset, 
    collate_fn_multilingual,
    create_stratified_split_multilingual,
    create_cross_lingual_split
)
from models_multilingual import (
    MultilingualMultimodalModel, 
    train_multilingual_model,
    train_cross_lingual_model
)

def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dataloaders(dataset, train_indices, val_indices, test_indices, batch_size, num_workers=0):
    """데이터로더 생성"""
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_multilingual,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_multilingual,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_multilingual,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description="다국어 멀티모달 치매 진단 모델")
    
    # 데이터 관련 설정
    parser.add_argument('--data_dir', type=str, default='../training_dset', 
                       help='데이터 디렉토리 경로')
    parser.add_argument('--max_seq_len', type=int, default=512, 
                       help='최대 시퀀스 길이')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='데이터로더 워커 수')
    
    # 모델 관련 설정
    parser.add_argument('--text_model_type', type=int, default=1, choices=[1, 2],
                       help='텍스트 모델 타입 (1: BERT only, 2: BERT + LSTM)')
    parser.add_argument('--dropout', type=float, default=0.3, 
                       help='드롭아웃 비율')
    
    # 훈련 관련 설정
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='훈련 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                       help='학습률')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                       help='가중치 감쇠')
    parser.add_argument('--warmup_steps', type=int, default=100, 
                       help='워밍업 스텝 수')
    parser.add_argument('--seed', type=int, default=42, 
                       help='랜덤 시드')
    
    # 실험 모드 설정
    parser.add_argument('--mode', type=str, default='all_languages', 
                       choices=['all_languages', 'cross_lingual'],
                       help='실험 모드')
    parser.add_argument('--languages', nargs='+', default=['English', 'Greek', 'Spanish', 'Mandarin'],
                       help='사용할 언어 목록')
    parser.add_argument('--train_languages', nargs='+', 
                       help='Cross-lingual 모드에서 훈련에 사용할 언어')
    parser.add_argument('--test_languages', nargs='+',
                       help='Cross-lingual 모드에서 테스트에 사용할 언어')
    
    # 기타 설정
    parser.add_argument('--device', type=str, default='auto',
                       help='사용할 디바이스 (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='모델 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ 사용 디바이스: {device}")
    
    # 시드 고정
    set_seed(args.seed)
    print(f"🎲 랜덤 시드: {args.seed}")
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # =================== 데이터 준비 ===================
    print(f"\n📂 데이터 준비 중...")
    print(f"  데이터 디렉토리: {args.data_dir}")
    print(f"  언어: {args.languages}")
    print(f"  최대 시퀀스 길이: {args.max_seq_len}")
    
    # 다국어 데이터셋 로드
    dataset = prepare_multilingual_dataset(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        languages=args.languages
    )
    
    print(f"✅ 총 {len(dataset)}개 샘플 로드됨")
    
    # =================== 데이터 분할 ===================
    if args.mode == 'all_languages':
        print(f"\n🌍 전체 언어 환자 단위 Stratified Split 모드")
        
        # 환자 단위 stratified split (7:1:2)
        train_indices, val_indices, test_indices = create_stratified_split_multilingual(
            dataset.data,
            train_split=0.7,
            val_split=0.1,
            test_split=0.2,
            random_seed=args.seed
        )
        
        experiment_name = f"AllLanguages_{'_'.join(args.languages)}"
        train_languages = args.languages
        test_languages = args.languages
        
    elif args.mode == 'cross_lingual':
        print(f"\n🌍 Cross-lingual 모드")
        
        if not args.train_languages or not args.test_languages:
            raise ValueError("Cross-lingual 모드에서는 --train_languages와 --test_languages를 지정해야 합니다.")
        
        # Cross-lingual split (7:1:2)
        train_indices, val_indices, test_indices = create_cross_lingual_split(
            dataset.data,
            train_languages=args.train_languages,
            test_languages=args.test_languages,
            random_seed=args.seed
        )
        
        experiment_name = f"CrossLingual_Train_{'_'.join(args.train_languages)}_Test_{'_'.join(args.test_languages)}"
        train_languages = args.train_languages
        test_languages = args.test_languages
    
    # 데이터로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, train_indices, val_indices, test_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"\n📊 데이터로더 생성 완료:")
    print(f"  훈련: {len(train_loader)} 배치 ({len(train_indices)} 샘플)")
    print(f"  검증: {len(val_loader)} 배치 ({len(val_indices)} 샘플)")
    print(f"  테스트: {len(test_loader)} 배치 ({len(test_indices)} 샘플)")
    
    # =================== 모델 초기화 ===================
    print(f"\n🧠 모델 초기화...")
    
    model = MultilingualMultimodalModel(
        text_model_type=args.text_model_type,
        dropout=args.dropout
    ).to(device)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  텍스트 모델 타입: {args.text_model_type}")
    print(f"  드롭아웃: {args.dropout}")
    print(f"  전체 파라미터: {total_params:,}")
    print(f"  훈련 가능 파라미터: {trainable_params:,}")
    
    # =================== 옵티마이저 및 스케줄러 ===================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # 총 스텝 수 계산
    total_steps = len(train_loader) * args.num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 손실 함수
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\n⚙️ 훈련 설정:")
    print(f"  옵티마이저: AdamW")
    print(f"  학습률: {args.learning_rate}")
    print(f"  가중치 감쇠: {args.weight_decay}")
    print(f"  워밍업 스텝: {args.warmup_steps}")
    print(f"  총 스텝: {total_steps}")
    print(f"  손실 함수: BCEWithLogitsLoss")
    
    # =================== 모델 훈련 ===================
    print(f"\n🚀 모델 훈련 시작...")
    print(f"  실험명: {experiment_name}")
    print(f"  에포크: {args.num_epochs}")
    print(f"  배치 크기: {args.batch_size}")
    
    if args.mode == 'cross_lingual':
        # Cross-lingual 훈련
        model, best_val_auc, final_metrics = train_cross_lingual_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=criterion,
            num_epochs=args.num_epochs,
            device=device,
            train_languages=train_languages,
            test_languages=test_languages
        )
    else:
        # 전체 언어 훈련
        model, best_val_auc, final_metrics = train_multilingual_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=criterion,
            num_epochs=args.num_epochs,
            device=device,
            experiment_name=experiment_name
        )
    
    # =================== 결과 저장 ===================
    results = {
        'experiment_name': experiment_name,
        'mode': args.mode,
        'languages': args.languages,
        'train_languages': train_languages,
        'test_languages': test_languages,
        'best_val_auc': best_val_auc,
        'final_metrics': final_metrics,
        'args': vars(args)
    }
    
    results_path = os.path.join(args.save_dir, f'results_{experiment_name}.json')
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        # numpy 값들을 JSON 직렬화 가능하도록 변환
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, np.floating):
                json_results[key] = float(value)
            elif isinstance(value, np.integer):
                json_results[key] = int(value)
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 결과 저장 완료: {results_path}")
    
    # =================== 최종 요약 ===================
    print(f"\n🎉 실험 완료!")
    print(f"  실험명: {experiment_name}")
    print(f"  최고 검증 AUC: {best_val_auc:.4f}")
    print(f"  모델 저장: best_model_{experiment_name}.pth")
    print(f"  결과 저장: {results_path}")
    
    if args.mode == 'cross_lingual':
        print(f"\n🌍 Cross-lingual 성능:")
        print(f"  훈련 언어: {train_languages}")
        print(f"  테스트 언어: {test_languages}")
        
        # 테스트 언어별 성능 출력
        for lang in test_languages:
            if f'{lang}_accuracy' in final_metrics:
                acc = final_metrics[f'{lang}_accuracy']
                auc = final_metrics.get(f'{lang}_auc', 0.0)
                f1 = final_metrics.get(f'{lang}_f1', 0.0)
                samples = final_metrics.get(f'{lang}_samples', 0)
                print(f"    {lang}: Acc={acc:.3f}, AUC={auc:.3f}, F1={f1:.3f} ({samples} samples)")
    
    print(f"\n✅ 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
