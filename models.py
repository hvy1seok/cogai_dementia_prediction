import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix
import time
import mlflow

class AudioModel(nn.Module):
    def __init__(self, output_dim=256):
        super(AudioModel, self).__init__()
        self.cnn = models.resnet101(pretrained=True)
        self.cnn.fc = nn.Sequential(nn.Linear(self.cnn.fc.in_features, output_dim))
        
    def forward(self, audio):
        return self.cnn(audio)

class TextModel(nn.Module):
    def __init__(self, text_model_type=1, d_model=1024, dropout=0.2):
        super(TextModel, self).__init__()
        self.textModel = BertModel.from_pretrained('bert-base-uncased')
        self.d_model = d_model
        self.text_model_type = text_model_type
        if text_model_type == 2:
            self.lstm = nn.LSTM(768, d_model, 1, batch_first=True, bidirectional=True)

    def forward(self, text, attention):
        textOut = self.textModel(text, attention_mask=attention)[0]
        if self.text_model_type == 1:
            output = textOut[:,0,:]
        elif self.text_model_type == 2:
            packed_input = pack_padded_sequence(textOut, [512 for _ in range(text.size(0))], batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            output = output[range(len(output)), [512 - 1 for _ in range(text.size(0))] ,:self.d_model]
        return output

class MultimodalModel(nn.Module):
    def __init__(self, text_model_type=1, dropout=0.3):
        super(MultimodalModel, self).__init__()
        self.audioModel = AudioModel()
        self.textModel = TextModel(text_model_type=text_model_type)
        
        classifier_input_size = 256 + (1024 if text_model_type == 2 else 768)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(classifier_input_size, 1)
        )

    def forward(self, text, attention, audio):
        audioOut = self.audioModel(audio)
        textOut = self.textModel(text, attention)
        outputs = torch.cat((audioOut, textOut), dim=1)
        outputs = self.classifier(outputs)
        return outputs

def train_model(model, train_iterator, validation_iterator, optimizer, scheduler, loss_fn, num_epochs=20):
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        start = time.time()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_size = 0
        
        # Training phase
        model.train()
        for text, attention, audio, labels in train_iterator:
            outputs = model(text, attention, audio)
            optimizer.zero_grad()
            predictions = torch.round(torch.sigmoid(outputs))
            acc = torch.round((predictions == labels.unsqueeze(1)).sum().float())
            loss = loss_fn(outputs, labels.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            
            epoch_accuracy += acc.item()
            epoch_loss += loss.item()
            epoch_size += labels.size(0)
        
        train_loss = epoch_loss / len(train_iterator)
        train_accuracy = epoch_accuracy / epoch_size * 100
        
        print(f'Train Epoch: {epoch + 1} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_size = 0
        conf_matrix = [[0,0], [0,0]]
        
        with torch.no_grad():
            for text, attention, audio, labels in validation_iterator:
                outputs = model(text, attention, audio)
                predictions = torch.round(torch.sigmoid(outputs))
                conf = confusion_matrix(labels.cpu(), predictions.squeeze(0).cpu(), labels=[0, 1])
                
                for i in range(2):
                    for j in range(2):
                        conf_matrix[i][j] += conf[i][j]
                
                acc = torch.round((predictions == labels.unsqueeze(1)).sum().float())
                loss = loss_fn(outputs, labels.unsqueeze(1))
                
                val_accuracy += acc.item()
                val_loss += loss.item()
                val_size += labels.size(0)
        
        val_loss = val_loss / len(validation_iterator)
        val_accuracy = val_accuracy / val_size * 100
        
        print(f'Validation Epoch: {epoch + 1} | Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.2f}%')
        print(f'Confusion Matrix:\n{conf_matrix}')
        
        # MLflow 메트릭 로깅
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch + 1
        })
        
        # 최고 성능 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            mlflow.log_metric("best_val_accuracy", best_val_accuracy)
            
        # 에포크 시간 로깅
        epoch_time = time.time() - start
        mlflow.log_metric("epoch_time", epoch_time) 