import torch
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
import time
import torch.nn as nn
from torchvision import models
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix
import os
import pandas as pd

# MLflow
import mlflow
import mlflow.pytorch

# 하이퍼파라미터 및 설정
seed_val = 0
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True

nEpochs = 100
pittPath = '../../voicedata/'
batchSize = 8
cnn = 'densenet'
modelName = 'audio_' + cnn + str(nEpochs)

def readIdx():
    audioPaths = []
    labels = []
    categories = ['dem', 'control']
    for cat in categories:
        path = pittPath + ('controlCha.txt' if cat == 'control' else 'dementiaCha.txt')
        with open(path, 'r') as index:
            files = index.readlines()
            for file in files:
                labels.append(0 if cat == 'control' else 1)
                audioPaths.append(file.strip()[:-4] + '.npy')
    return audioPaths, labels

class Dataset:
    def __init__(self, audioPaths, labels):
        self.audioPaths = audioPaths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        audioSpecs = np.load(self.audioPaths[item])
        return audioSpecs, self.labels[item]

def collate_fn(batch, device):
    batchSize = len(batch)
    audio = torch.FloatTensor(batchSize, 3, 128, 250).fill_(0).to(device)
    label = torch.FloatTensor(batchSize).fill_(0).to(device)
    for i, (audioSpec, labelD) in enumerate(batch):
        audio[i] = torch.tensor(audioSpec)
        label[i] = labelD
    return audio, label

def getDataloaders(device):
    audioPaths, labels = readIdx()
    dataset = Dataset(audioPaths, labels)
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    trainDataset, validationDataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    collate_fn_ = partial(collate_fn, device=device)
    trainIterator = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    validationIterator = DataLoader(validationDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    return trainIterator, validationIterator

class audioModel(nn.Module):
    def __init__(self, dropout=0.2):
        super(audioModel, self).__init__()
        if cnn == 'resnet':
            self.cnn = models.resnet101(pretrained=True)
            outFeatures = self.cnn.fc.out_features
        elif cnn == 'densenet':
            self.cnn = models.densenet201(pretrained=True)
            outFeatures = self.cnn.classifier.out_features
        elif cnn == 'mobilenet':
            self.cnn = models.mobilenet_v2()
            outFeatures = self.cnn.classifier[1].out_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(outFeatures, 1)
        )

    def forward(self, audio):
        x = self.cnn(audio)
        x = self.classifier(x)
        return x

def train(model, trainIterator, validationIterator, optimizer, scheduler, lossfn):
    with mlflow.start_run():
        # 파라미터 로깅
        mlflow.log_param("nEpochs", nEpochs)
        mlflow.log_param("batchSize", batchSize)
        mlflow.log_param("cnn", cnn)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
        mlflow.log_param("pos_weight", lossfn.pos_weight.item())

        for epoch in range(nEpochs):
            start = time.time()
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_size = 0
            for audio, labels in trainIterator:
                outputs = model(audio)
                optimizer.zero_grad()
                predictions = torch.round(torch.sigmoid(outputs))
                acc = torch.round((predictions == labels.unsqueeze(1)).sum().float())
                loss = lossfn(outputs, labels.unsqueeze(1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                epoch_accuracy += acc.item()
                epoch_loss += loss.item()
                epoch_size += labels.size(0)

            end = time.time()
            train_loss = epoch_loss / len(trainIterator)
            train_acc = epoch_accuracy / epoch_size * 100
            print(f'Train Epoch {epoch+1} | Time: {end-start:.2f}s | Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%')

            # 로그 기록
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            # Validation
            start = time.time()
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_size = 0
            conf_matrix = [[0, 0], [0, 0]]
            for audio, labels in validationIterator:
                with torch.no_grad():
                    outputs = model(audio)
                predictions = torch.round(torch.sigmoid(outputs))
                conf = confusion_matrix(y_true=labels.cpu(), y_pred=predictions.squeeze(1).cpu(), labels=[0, 1])
                for i in range(2):
                    for j in range(2):
                        conf_matrix[i][j] += conf[i][j]
                acc = torch.round((predictions == labels.unsqueeze(1)).sum().float())
                loss = lossfn(outputs, labels.unsqueeze(1))
                epoch_accuracy += acc.item()
                epoch_loss += loss.item()
                epoch_size += labels.size(0)

            val_loss = epoch_loss / len(validationIterator)
            val_acc = epoch_accuracy / epoch_size * 100
            print(f'Validation Epoch {epoch+1} | Time: {time.time()-start:.2f}s | Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%')
            print(f'Confusion Matrix: {conf_matrix}')

            # 로그 기록
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            # Confusion matrix 저장
            cm_df = pd.DataFrame(conf_matrix, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
            os.makedirs("confusion", exist_ok=True)
            cm_path = f"confusion/cm_epoch_{epoch+1}.csv"
            cm_df.to_csv(cm_path)
            mlflow.log_artifact(cm_path)

            # 모델 저장
            os.makedirs("modelos", exist_ok=True)
            model_path = f"modelos/{modelName}_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)

        # 전체 모델 구조 포함 저장
        mlflow.pytorch.log_model(model, artifact_path="models/audio_model")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainIterator, validationIterator = getDataloaders(device)
    model = audioModel()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.78))
    total_steps = len(trainIterator) * nEpochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train(model, trainIterator, validationIterator, optimizer, scheduler, lossfn)
