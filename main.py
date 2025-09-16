import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from transformers import get_linear_schedule_with_warmup
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import mlflow
import random
import numpy as np
import os

from dataset import prepare_dataset, collate_fn
from models import MultimodalModel, train_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # MLflow 설정
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # 시드 설정
    set_seed(cfg.training.seed)
    
    # 디바이스 설정
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # 데이터셋 준비
    dataset = prepare_dataset(cfg.dataset.path, cfg.dataset.max_seq_len)
    
    # 학습/검증 데이터 분할
    train_size = int(cfg.dataset.train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # 데이터로더 생성
    collate_fn_ = partial(collate_fn, pad_val=0, device=device, max_seq_len=cfg.dataset.max_seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_,
        num_workers=cfg.dataset.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_,
        num_workers=cfg.dataset.num_workers
    )
    
    # 모델 초기화
    model = MultimodalModel(
        text_model_type=cfg.model.text_model_type,
        dropout=cfg.model.dropout
    ).to(device)
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.learning_rate, 
        eps=cfg.training.eps
    )
    total_steps = len(train_loader) * cfg.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 손실 함수
    criterion = nn.BCEWithLogitsLoss()
    
    # MLflow로 파라미터 기록
    with mlflow.start_run():
        # 설정 로깅
        mlflow.log_params({
            "model_type": cfg.model.name,
            "text_model_type": cfg.model.text_model_type,
            "batch_size": cfg.dataset.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "num_epochs": cfg.training.num_epochs,
            "max_seq_len": cfg.dataset.max_seq_len,
        })
        
        # 모델 학습
        train_model(
            model=model,
            train_iterator=train_loader,
            validation_iterator=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=criterion,
            num_epochs=cfg.training.num_epochs
        )
        
        # 모델 저장
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
