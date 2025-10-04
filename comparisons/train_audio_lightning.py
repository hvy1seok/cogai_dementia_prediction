import os
from pathlib import Path
from typing import Optional, Tuple
import colorlog
import logging

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import torchmetrics
from torchmetrics.functional import auroc, precision, recall
from sklearn.metrics import accuracy_score, f1_score
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from transformers import get_linear_schedule_with_warmup


# 프로젝트 루트 디렉토리 찾기
def get_project_root() -> Path:
    current_path = Path(__file__).resolve()
    while not (current_path / "pyproject.toml").exists():
        if current_path.parent == current_path:
            raise RuntimeError("프로젝트 루트를 찾을 수 없습니다.")
        current_path = current_path.parent
    return current_path


class AudioDataset(Dataset):
    def __init__(self, root_dir: str, language: str, classes: Optional[Tuple[str, str]] = ("HC", "AD")):
        project_root = get_project_root()
        self.root_dir = project_root / root_dir  # training_dset 폴더의 절대 경로
        self.language = language
        self.audio_paths = []
        self.labels = []
        self.classes = classes  # (negative, positive)
        
        logger = logging.getLogger(__name__)
        logger.info(f"프로젝트 루트 경로: {project_root}")
        logger.info(f"데이터셋 기본 경로: {self.root_dir}")
        logger.info(f"선택된 언어: {self.language}")
        
        # 언어별 데이터 폴더
        lang_folders = {
            "Greek": "Greek",
            "Spanish": "Spanish",
            "Mandarin": "Mandarin",
            "English": "English"
        }
        
        if self.language not in lang_folders:
            raise ValueError(f"지원하지 않는 언어입니다: {self.language}")
            
        lang_path = self.root_dir / lang_folders[self.language]
        voice_path = lang_path / "voicedata"
        logger.info(f"음성 데이터 경로: {voice_path}")
        
        # 선택 클래스만 수집 (기본: HC=0, AD=1 이진 분류)
        neg_class, pos_class = self.classes

        neg_path = voice_path / neg_class
        pos_path = voice_path / pos_class

        neg_files = list(neg_path.glob("*.npy"))
        pos_files = list(pos_path.glob("*.npy"))

        logger.info(f"{neg_class} 파일 개수: {len(neg_files)}, 경로: {neg_path}")
        logger.info(f"{pos_class} 파일 개수: {len(pos_files)}, 경로: {pos_path}")

        self.audio_paths.extend(neg_files)
        self.labels.extend([0] * len(neg_files))

        self.audio_paths.extend(pos_files)
        self.labels.extend([1] * len(pos_files))
        
        logger.info(f"전체 데이터셋 크기: {len(self.labels)}")
        if len(self.labels) == 0:
            raise ValueError(
                f"데이터를 찾을 수 없습니다. 경로를 확인해주세요: {voice_path}/[{neg_class}|{pos_class}]"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_specs = np.load(str(self.audio_paths[idx]))
        audio_specs = torch.FloatTensor(audio_specs)
        return audio_specs, self.labels[idx]


class AudioModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = getattr(cfg.model, "num_classes", 2)
        
        # CNN 백본 선택
        if cfg.model.name == "densenet":
            self.cnn = models.densenet201(pretrained=cfg.model.pretrained)
            out_features = self.cnn.classifier.out_features
        elif cfg.model.name == "resnet":
            self.cnn = models.resnet101(pretrained=cfg.model.pretrained)
            out_features = self.cnn.fc.out_features
        elif cfg.model.name == "mobilenet":
            self.cnn = models.mobilenet_v2(pretrained=cfg.model.pretrained)
            out_features = self.cnn.classifier[1].out_features
        else:
            raise ValueError(f"Unknown model: {cfg.model.name}")

        # 분류기 (기본: 2개 클래스 HC/AD)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.model.dropout),
            nn.Linear(out_features, self.num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        if self.num_classes == 2:
            self.train_acc = torchmetrics.Accuracy(task="binary")
            self.val_acc = torchmetrics.Accuracy(task="binary")
            self.val_f1 = torchmetrics.F1Score(task="binary", average="macro")
        else:
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        f1 = self.val_f1(preds, y)
        # 확률 점수 (이진일 때 클래스 1의 확률)
        if self.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
            try:
                auc = auroc(probs, y, task="binary")
            except Exception:
                auc = torch.tensor(0.0, device=logits.device)
            prec = precision(preds, y, task="binary", ignore_index=None)
            rec = recall(preds, y, task="binary", ignore_index=None)
            self.log("val/auc", auc, on_epoch=True, prog_bar=False)
            self.log("val/precision", prec, on_epoch=True, prog_bar=False)
            self.log("val/recall", rec, on_epoch=True, prog_bar=False)
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.training.learning_rate
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.training.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset = None
        self.classes = ("HC", "AD")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset = AudioDataset(
                self.cfg.paths.data_dir,
                self.cfg.language,
                classes=self.classes
            )
            
            # Train/Val 분할
            train_size = int(len(self.dataset) * self.cfg.data.train_ratio)
            val_size = len(self.dataset) - train_size
            
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, 
                [train_size, val_size]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers
        )


@hydra.main(config_path="../conf/train", config_name="audio", version_base="1.1")
def main(cfg: DictConfig):
    # wandb 초기화
    wandb_kwargs = {
        "project": cfg.wandb.project,
        "name": f"{cfg.language}_audio_{cfg.model.name}",  # 언어와 모델명으로 실험 이름 생성
        "tags": ["audio", cfg.language.lower()]  # 언어를 태그에 추가
    }
    if cfg.wandb.entity is not None:
        wandb_kwargs["entity"] = cfg.wandb.entity
    
    wandb_logger = WandbLogger(**wandb_kwargs)
    
    # 체크포인트 콜백
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename=f"{cfg.language}_" + "{epoch:02d}-{val/f1:.3f}",  # 언어 정보 추가
        monitor="val/f1",
        mode="max",
        save_top_k=3
    )
    
    # 데이터 모듈
    datamodule = AudioDataModule(cfg)
    
    # 모델
    model = AudioModel(cfg)
    
    # 트레이너
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )
    
    # 학습 시작
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main() 