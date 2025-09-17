import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
import time
import wandb
from collections import defaultdict, Counter
import numpy as np

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
            output = textOut[:,0,:]  # [CLS] 토큰 사용
        elif self.text_model_type == 2:
            # LSTM 추가 처리
            seq_lengths = attention.sum(dim=1).cpu()  # 실제 시퀀스 길이
            packed_input = pack_padded_sequence(textOut, seq_lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            # 마지막 유효한 출력 선택
            batch_size = text.size(0)
            indices = (seq_lengths - 1).long()
            output = output[range(batch_size), indices, :self.d_model]
        return output

class MultilingualMultimodalModel(nn.Module):
    def __init__(self, text_model_type=1, dropout=0.3):
        super(MultilingualMultimodalModel, self).__init__()
        self.audioModel = AudioModel()
        self.textModel = TextModel(text_model_type=text_model_type)
        
        # 분류기 입력 크기 계산
        text_output_dim = 1024 if text_model_type == 2 else 768
        classifier_input_size = 256 + text_output_dim  # audio + text
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(classifier_input_size, 1)
        )

    def forward(self, input_ids, attention_mask, audio):
        audioOut = self.audioModel(audio)  # (batch, 256)
        textOut = self.textModel(input_ids, attention_mask)  # (batch, 768/1024)
        
        # 특징 연결
        outputs = torch.cat((audioOut, textOut), dim=1)  # (batch, 256+768/1024)
        outputs = self.classifier(outputs)  # (batch, 1)
        return outputs

def compute_language_specific_metrics(predictions, labels, languages, optimal_threshold=0.5):
    """언어별 상세 메트릭 계산"""
    language_metrics = {}
    
    # 언어별 데이터 그룹화
    lang_data = defaultdict(lambda: {'preds': [], 'labels': []})
    
    for pred, label, lang in zip(predictions, labels, languages):
        lang_data[lang]['preds'].append(pred)
        lang_data[lang]['labels'].append(label)
    
    print(f"\n📊 언어별 성능 분석 (임계값: {optimal_threshold:.3f}):")
    print("="*60)
    
    overall_metrics = {}
    
    for language in sorted(lang_data.keys()):
        lang_preds = np.array(lang_data[language]['preds'])
        lang_labels = np.array(lang_data[language]['labels'])
        
        if len(lang_preds) == 0:
            continue
            
        # 이진 예측 (임계값 적용)
        lang_binary_preds = (lang_preds > optimal_threshold).astype(int)
        
        # 기본 메트릭
        accuracy = (lang_binary_preds == lang_labels).mean()
        
        # AUC (두 클래스가 모두 있는 경우만)
        auc = 0.0
        if len(np.unique(lang_labels)) > 1:
            try:
                auc = roc_auc_score(lang_labels, lang_preds)
            except:
                auc = 0.0
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            lang_labels, lang_binary_preds, average='binary', zero_division=0
        )
        
        # 언어별 메트릭 저장
        language_metrics[f'{language}_accuracy'] = accuracy
        language_metrics[f'{language}_auc'] = auc
        language_metrics[f'{language}_precision'] = precision
        language_metrics[f'{language}_recall'] = recall
        language_metrics[f'{language}_f1'] = f1
        language_metrics[f'{language}_samples'] = len(lang_preds)
        
        # 라벨 분포
        label_dist = Counter(lang_labels)
        normal_count = label_dist[0]
        dementia_count = label_dist[1]
        
        print(f"🌍 {language}:")
        print(f"  📊 샘플: {len(lang_preds)}개 (정상: {normal_count}, 치매: {dementia_count})")
        print(f"  🎯 정확도: {accuracy:.3f}")
        print(f"  📈 AUC: {auc:.3f}")
        print(f"  🔍 정밀도: {precision:.3f}")
        print(f"  🔍 재현율: {recall:.3f}")
        print(f"  🔍 F1: {f1:.3f}")
        print()
    
    return language_metrics

def compute_optimal_threshold(predictions, labels):
    """ROC 곡선에서 최적 임계값 계산"""
    from sklearn.metrics import roc_curve
    
    if len(np.unique(labels)) < 2:
        return 0.5
    
    try:
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        # Youden's J statistic으로 최적 임계값 찾기
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold
    except:
        return 0.5

def train_multilingual_model(model, train_loader, val_loader, test_loader, optimizer, scheduler, 
                           loss_fn, num_epochs=20, device='cuda', experiment_name="multilingual"):
    """다국어 모델 훈련 (언어별 분석 포함)"""
    
    best_val_accuracy = 0.0
    best_val_auc = 0.0
    
    # wandb 초기화
    wandb.init(
        project="dementia-multilingual",
        name=experiment_name,
        config={
            "epochs": num_epochs,
            "model": "MultilingualMultimodal",
            "optimizer": type(optimizer).__name__,
            "loss_function": type(loss_fn).__name__
        }
    )
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # =================== 훈련 단계 ===================
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        train_languages = []
        
        for batch in train_loader:
            # 배치 데이터 GPU로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, audio).squeeze()
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # 메트릭 수집
            train_loss += loss.item()
            predictions = torch.sigmoid(outputs).detach().cpu().numpy()
            train_predictions.extend(predictions)
            train_labels.extend(labels.detach().cpu().numpy())
            train_languages.extend(batch['languages'])
        
        # 훈련 메트릭 계산
        train_loss /= len(train_loader)
        train_predictions = np.array(train_predictions)
        train_labels = np.array(train_labels)
        
        # 최적 임계값 계산
        train_optimal_threshold = compute_optimal_threshold(train_predictions, train_labels)
        train_binary_preds = (train_predictions > train_optimal_threshold).astype(int)
        train_accuracy = (train_binary_preds == train_labels).mean()
        
        # 훈련 AUC
        train_auc = 0.0
        if len(np.unique(train_labels)) > 1:
            try:
                train_auc = roc_auc_score(train_labels, train_predictions)
            except:
                pass
        
        # =================== 검증 단계 ===================
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        val_languages = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, audio).squeeze()
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                predictions = torch.sigmoid(outputs).cpu().numpy()
                val_predictions.extend(predictions)
                val_labels.extend(labels.cpu().numpy())
                val_languages.extend(batch['languages'])
        
        # 검증 메트릭 계산
        val_loss /= len(val_loader)
        val_predictions = np.array(val_predictions)
        val_labels = np.array(val_labels)
        
        val_optimal_threshold = compute_optimal_threshold(val_predictions, val_labels)
        val_binary_preds = (val_predictions > val_optimal_threshold).astype(int)
        val_accuracy = (val_binary_preds == val_labels).mean()
        
        val_auc = 0.0
        if len(np.unique(val_labels)) > 1:
            try:
                val_auc = roc_auc_score(val_labels, val_predictions)
            except:
                pass
        
        # =================== 테스트 단계 (참고용) ===================
        test_loss = 0.0
        test_predictions = []
        test_labels = []
        test_languages = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, audio).squeeze()
                loss = loss_fn(outputs, labels)
                
                test_loss += loss.item()
                predictions = torch.sigmoid(outputs).cpu().numpy()
                test_predictions.extend(predictions)
                test_labels.extend(labels.cpu().numpy())
                test_languages.extend(batch['languages'])
        
        # 테스트 메트릭 계산
        test_loss /= len(test_loader)
        test_predictions = np.array(test_predictions)
        test_labels = np.array(test_labels)
        
        test_optimal_threshold = compute_optimal_threshold(test_predictions, test_labels)
        test_binary_preds = (test_predictions > test_optimal_threshold).astype(int)
        test_accuracy = (test_binary_preds == test_labels).mean()
        
        test_auc = 0.0
        if len(np.unique(test_labels)) > 1:
            try:
                test_auc = roc_auc_score(test_labels, test_predictions)
            except:
                pass
        
        # =================== 언어별 분석 ===================
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        print(f"훈련 - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, AUC: {train_auc:.4f}")
        print(f"검증 - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
        print(f"테스트 - Loss: {test_loss:.4f}, Acc: {test_accuracy:.4f}, AUC: {test_auc:.4f}")
        
        # 언어별 메트릭 (테스트 세트)
        test_lang_metrics = compute_language_specific_metrics(
            test_predictions, test_labels, test_languages, test_optimal_threshold
        )
        
        # =================== wandb 로깅 ===================
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_auc': train_auc,
            'train_optimal_threshold': train_optimal_threshold,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'val_optimal_threshold': val_optimal_threshold,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'test_optimal_threshold': test_optimal_threshold,
            'learning_rate': optimizer.param_groups[0]['lr'] if optimizer else 0,
            'epoch_time': time.time() - start_time
        }
        
        # 언어별 메트릭 추가
        for key, value in test_lang_metrics.items():
            log_dict[f'test_{key}'] = value
        
        wandb.log(log_dict)
        
        # =================== 베스트 모델 저장 ===================
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_accuracy = val_accuracy
            
            # 모델 저장
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'val_accuracy': val_accuracy,
                'val_auc': val_auc,
                'test_metrics': test_lang_metrics
            }, f'best_model_{experiment_name}.pth')
            
            print(f"🏆 새로운 베스트 모델! Val AUC: {val_auc:.4f}")
            wandb.log({'best_val_auc': best_val_auc, 'best_val_accuracy': best_val_accuracy})
    
    print(f"\n🎉 훈련 완료!")
    print(f"🏆 최고 검증 AUC: {best_val_auc:.4f}")
    print(f"🏆 최고 검증 정확도: {best_val_accuracy:.4f}")
    
    # 최종 언어별 성능 요약
    final_lang_metrics = compute_language_specific_metrics(
        test_predictions, test_labels, test_languages, test_optimal_threshold
    )
    
    # 최종 요약을 wandb에 로깅
    wandb.log({
        'final_best_val_auc': best_val_auc,
        'final_best_val_accuracy': best_val_accuracy,
        **{f'final_{key}': value for key, value in final_lang_metrics.items()}
    })
    
    wandb.finish()
    
    return model, best_val_auc, final_lang_metrics

def train_cross_lingual_model(model, train_loader, val_loader, test_loader, optimizer, scheduler,
                            loss_fn, num_epochs, device, train_languages, test_languages):
    """Cross-lingual 모델 훈련"""
    
    experiment_name = f"CrossLingual_Train_{'_'.join(train_languages)}_Test_{'_'.join(test_languages)}"
    
    print(f"\n🌍 Cross-lingual 실험 시작:")
    print(f"  훈련 언어: {train_languages}")
    print(f"  테스트 언어: {test_languages}")
    print(f"  실험명: {experiment_name}")
    
    return train_multilingual_model(
        model, train_loader, val_loader, test_loader, optimizer, scheduler,
        loss_fn, num_epochs, device, experiment_name
    )
