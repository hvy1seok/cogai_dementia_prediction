import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from functools import partial
import time
import torch.nn as nn
from torchvision import models
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import wandb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
import os


seed_val = 0

torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True

nEpochs = 100
data_root = '../../training_dset/'  # training_dset 기준
batchSize = 8
maxSeqLen = 256  # 텍스트 문맥 보존을 위해 확장
textModel = 1
modelName = 'Multimodal'+str(textModel)+str(nEpochs)

from pathlib import Path

def readFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return " ".join([ln.strip() for ln in lines])

def readIdx():
    utts = []
    audioPaths = []
    labels = []
    langs = []
    stems = []

    # EN & CN, HC/AD 구조만 사용 (대조군)
    pairs = [
        ("English", "HC", 0),
        ("English", "AD", 1),
        ("Mandarin", "HC", 0),
        ("Mandarin", "AD", 1),
    ]

    root_path = Path(data_root)
    for lang, cls, lab in pairs:
        txt_dir = root_path / lang / "textdata" / cls
        wav_dir = root_path / lang / "voicedata" / cls
        if not txt_dir.exists() or not wav_dir.exists():
            print(f"⚠️ Missing lang/class dir: {txt_dir} or {wav_dir}")
            continue

        # 하위폴더 포함 재귀 탐색: 텍스트 상대경로를 기준으로 동일 구조의 .npy 매칭
        for txt_path in txt_dir.rglob("*.txt"):
            try:
                rel_path = txt_path.relative_to(txt_dir)
            except Exception:
                rel_path = txt_path.name
            npy_path = (wav_dir / rel_path).with_suffix('.npy')
            if not npy_path.exists():
                continue
            utt = readFile(txt_path)
            utts.append(utt)
            audioPaths.append(str(npy_path))
            labels.append(lab)
            langs.append(lang)
            stems.append(txt_path.stem)

    return utts, audioPaths, labels, langs, stems

def extract_patient_id(file_stem: str) -> str:
    parts = re.split(r"[_\-]", file_stem)
    if len(parts) == 0:
        return file_stem
    # 우선 숫자가 포함된 첫 파트를 우선 사용
    for p in parts:
        if any(ch.isdigit() for ch in p):
            return p
    # 아니면 앞의 두 파트 결합에 숫자가 있으면 사용
    if len(parts) >= 2:
        combined = f"{parts[0]}_{parts[1]}"
        if any(ch.isdigit() for ch in combined):
            return combined
    return parts[0]


class Dataset:
    def __init__(self, text, attentions, audioPaths, labels, langs, patient_ids):
        self.text = text
        self.attentions = attentions
        self.audioPaths = audioPaths
        self.labels = labels
        self.langs = langs
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        audioSpecs = np.load(self.audioPaths[item])
        return self.text[item], self.attentions[item], audioSpecs, self.labels[item], self.langs[item]
            


def collate_fn(batch, padVal, device):

    batchSize = len(batch)

    text = torch.LongTensor(batchSize, maxSeqLen).fill_(padVal).to(device)
    attentions = torch.IntTensor(batchSize, maxSeqLen).fill_(padVal).to(device)
    audio = torch.FloatTensor(batchSize, 3, 128, 250).fill_(0).to(device)
    label = torch.FloatTensor(batchSize).fill_(0).to(device)

    langs = []
    for i, (transcript, attentionsD, audioSpec, labelD, langD) in enumerate(batch):
        text[i] = transcript.detach().clone()
        attentions[i] = attentionsD.detach().clone()
        audio[i] = torch.tensor(audioSpec)
        label[i] = labelD
        langs.append(langD)

    return text, attentions, audio, label, langs


def getDataloaders(device, tokenizer):
    utterances, audioPaths, labels, langs, stems = readIdx()

    tokenized_inputs = []
    attention_masks = []
    for utt in utterances:
        token = tokenizer.encode_plus(utt,
                            add_special_tokens = True,
                            max_length = maxSeqLen,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',   # Return pytorch tensors.
                            truncation = True
                      )
        tokenized_inputs.append(token['input_ids'])
        attention_masks.append(token['attention_mask'])

    patient_ids = [extract_patient_id(s) for s in stems]
    dataset = Dataset(tokenized_inputs, attention_masks, audioPaths, labels, langs, patient_ids)

    # 언어별 환자 단위 8:2 분할 후 합치기 (SigLIP-SAM 로직 참고)
    train_indices = []
    test_indices = []
    for lang in ["English", "Mandarin"]:
        lang_idxs = [i for i, lg in enumerate(dataset.langs) if lg == lang]
        if not lang_idxs:
            continue
        # 환자별 그룹
        pid_to_indices = {}
        for i in lang_idxs:
            pid = dataset.patient_ids[i]
            pid_to_indices.setdefault(pid, []).append(i)
        pids = list(pid_to_indices.keys())
        # 환자 라벨(대표 라벨) 구성
        pid_labels = []
        for pid in pids:
            lbls = [dataset.labels[i] for i in pid_to_indices[pid]]
            # 다수결 또는 첫 라벨
            lab = 1 if sum(lbls) > (len(lbls)/2) else 0
            pid_labels.append(lab)
        # Stratify 가능 여부 체크
        try:
            p_train, p_test = train_test_split(
                pids,
                test_size=0.2,
                stratify=pid_labels if len(set(pid_labels)) > 1 else None,
                random_state=42
            )
        except Exception:
            # 실패 시 무작위 분할
            p_train, p_test = train_test_split(pids, test_size=0.2, random_state=42)
        # 최소 1 환자 보장
        if len(p_test) == 0 and len(p_train) > 1:
            p_test = [p_train[-1]]
            p_train = p_train[:-1]
        # 인덱스 수집
        for pid in p_train:
            train_indices.extend(pid_to_indices[pid])
        for pid in p_test:
            test_indices.extend(pid_to_indices[pid])

    trainDataset = Subset(dataset, train_indices)
    testDataset = Subset(dataset, test_indices)

    collate_fn_ = partial(collate_fn, device=device, padVal=0)
    trainIterator = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    testIterator = DataLoader(testDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    return trainIterator, testIterator

class audioModel(nn.Module):
    def __init__(self):
        super(audioModel, self).__init__()
        self.cnn = models.resnet101(pretrained=True)
        self.cnn.fc = nn.Sequential(nn.Linear(self.cnn.fc.in_features, 256))
        
    def forward(self, audio):
        x = self.cnn(audio)
        return x

class text(nn.Module):
    def __init__(self, dModel = 1024, dropout = 0.2):
      super(text, self).__init__()
      self.textModel = BertModel.from_pretrained('bert-base-uncased')
      self.dModel = dModel
      if textModel == 2:
        self.lstm = nn.LSTM(768, dModel, 1, batch_first = True, bidirectional = True)

    def forward(self, text, attention):
      textOut = self.textModel(text, attention_mask=attention)[0]
      if textModel == 1:
        output = textOut[:,0,:]
      elif textModel == 2:
        packed_input = pack_padded_sequence(textOut, [maxSeqLen for _ in range(text.size(0))], batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output = output[range(len(output)), [maxSeqLen - 1 for _ in range(text.size(0))] ,:self.dModel]

      return output

class MultimodalModel(nn.Module):
    def __init__(self, dropout = 0.3):
      super(MultimodalModel, self).__init__()
      self.audioModel = audioModel()
      self.textModel = text()

      classifier_input_size = 256 + (1024 if textModel == 2 else 768)
      self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(classifier_input_size, 1)
              )

    def forward(self, text, attention, audio):
      audioOut = self.audioModel(audio)
      textOut = self.textModel(text, attention)
      outputs = torch.cat((audioOut, textOut), dim = 1)
      outputs = self.classifier(outputs)
      return outputs

def train(model, trainIterator, testIterator, optimizer, scheduler, lossfn):

    best_avg_auc = -1.0
    os.makedirs('modelos', exist_ok=True)

    for epoch in range(nEpochs):
        start = time.time()
        epoch_loss = 0
        train_y_true = []
        train_y_pred = []
        epoch_size = 0
        train_y_prob = []
        for i, (text, attention, audio, labels, batch_langs) in enumerate(trainIterator):
            outputs = model(text, attention, audio)
 
            optimizer.zero_grad()

            predictions = torch.round(torch.sigmoid(outputs))
            loss = lossfn(outputs, labels.unsqueeze(1))
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            scheduler.step()
            # Statistics
            epoch_loss += loss.item()
            epoch_size += labels.size(0)
            train_y_true.extend(labels.detach().cpu().numpy().tolist())
            train_y_pred.extend(predictions.detach().cpu().numpy().reshape(-1).tolist())
            train_y_prob.extend(torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1).tolist())
            
        end = time.time()
        train_acc = accuracy_score(train_y_true, train_y_pred) if train_y_true else 0.0
        train_f1 = f1_score(train_y_true, train_y_pred, average='macro') if train_y_true else 0.0
        print('Train Epoch: ', epoch + 1, ' | in ', end - start, ' seconds')
        print('Train Loss: ', epoch_loss / max(1, len(trainIterator)))
        print('Train Acc: ', round(train_acc * 100, 3))
        print('Train Macro-F1: ', round(train_f1, 4))
        try:
            train_auc = roc_auc_score(train_y_true, train_y_prob) if train_y_true else 0.0
        except Exception:
            train_auc = 0.0
        train_prec = precision_score(train_y_true, train_y_pred, zero_division=0) if train_y_true else 0.0
        train_rec = recall_score(train_y_true, train_y_pred, zero_division=0) if train_y_true else 0.0
        print('Train AUC: ', round(train_auc, 4))
        print('Train Precision: ', round(train_prec, 4))
        print('Train Recall: ', round(train_rec, 4))

        start = time.time()
        epoch_loss = 0
        val_y_true = []
        val_y_pred = []
        epoch_size = 0
        val_y_prob = []
        conf_matrix = [[0,0], [0,0]]
        val_langs = []
        for i, (text, attention, audio, labels, batch_langs) in enumerate(testIterator):
           
            with torch.no_grad():
                outputs = model(text, attention, audio)
            
            predictions = torch.round(torch.sigmoid(outputs))
            

            conf = confusion_matrix(y_true = labels.cpu(), y_pred = predictions.squeeze(0).cpu(), labels=[0, 1])
            

            for i in range(2):
                for j in range(2):
                    conf_matrix[i][j] += conf[i][j]

            loss = lossfn(outputs, labels.unsqueeze(1))

            # Statistics
            epoch_loss += loss.item()
            epoch_size += labels.size(0)
            val_y_true.extend(labels.detach().cpu().numpy().tolist())
            val_y_pred.extend(predictions.detach().cpu().numpy().reshape(-1).tolist())
            val_y_prob.extend(torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1).tolist())
            val_langs.extend(batch_langs)
            
        end = time.time()
        val_acc = accuracy_score(val_y_true, val_y_pred) if val_y_true else 0.0
        val_f1 = f1_score(val_y_true, val_y_pred, average='macro') if val_y_true else 0.0
        print('Test Epoch: ', epoch + 1, ' | in ', end - start, ' seconds')
        print('Test Loss: ', epoch_loss / max(1, len(testIterator)))
        print('Test Acc: ', round(val_acc * 100, 3))
        print('Test Macro-F1: ', round(val_f1, 4))
        try:
            val_auc = roc_auc_score(val_y_true, val_y_prob) if val_y_true else 0.0
        except Exception:
            val_auc = 0.0
        val_prec = precision_score(val_y_true, val_y_pred, zero_division=0) if val_y_true else 0.0
        val_rec = recall_score(val_y_true, val_y_pred, zero_division=0) if val_y_true else 0.0
        print('Test AUC: ', round(val_auc, 4))
        print('Test Precision: ', round(val_prec, 4))
        print('Test Recall: ', round(val_rec, 4))

        # 언어별 평가 (English / Mandarin)
        def lang_mask(lang):
            return [j for j, lg in enumerate(val_langs) if lg == lang]

        en_auc = None
        cn_auc = None
        for lg, key in [("English", "en"), ("Mandarin", "cn")]:
            idxs = lang_mask(lg)
            if idxs:
                y_true_l = [val_y_true[j] for j in idxs]
                y_pred_l = [val_y_pred[j] for j in idxs]
                y_prob_l = [val_y_prob[j] for j in idxs]
                try:
                    auc_l = roc_auc_score(y_true_l, y_prob_l)
                except Exception:
                    auc_l = 0.0
                acc_l = accuracy_score(y_true_l, y_pred_l)
                f1_l = f1_score(y_true_l, y_pred_l, average='macro')
                prec_l = precision_score(y_true_l, y_pred_l, zero_division=0)
                rec_l = recall_score(y_true_l, y_pred_l, zero_division=0)
                print(f"[Test {lg}] Acc: {acc_l:.4f}, F1: {f1_l:.4f}, AUC: {auc_l:.4f}, P: {prec_l:.4f}, R: {rec_l:.4f}")
                if key == 'en':
                    en_auc = auc_l
                elif key == 'cn':
                    cn_auc = auc_l
            else:
                print(f"[Test {lg}] 샘플이 없습니다.")

        print('Test confusion_matrix: ', conf_matrix)

        # W&B logging
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': epoch_loss / max(1, len(trainIterator)),
            'train/acc': train_acc,
            'train/macro_f1': train_f1,
            'train/auc': train_auc,
            'train/precision': train_prec,
            'train/recall': train_rec,
            'test/loss': epoch_loss / max(1, len(testIterator)),
            'test/acc': val_acc,
            'test/macro_f1': val_f1,
            'test/auc': val_auc,
            'test/precision': val_prec,
            'test/recall': val_rec,
        })

        # 언어별 메트릭 W&B 로깅
        log_lang = {}
        for lg, key in [("English", "en"), ("Mandarin", "cn")]:
            idxs = [j for j, l in enumerate(val_langs) if l == lg]
            if idxs:
                y_true_l = [val_y_true[j] for j in idxs]
                y_pred_l = [val_y_pred[j] for j in idxs]
                y_prob_l = [val_y_prob[j] for j in idxs]
                try:
                    auc_l = roc_auc_score(y_true_l, y_prob_l)
                except Exception:
                    auc_l = 0.0
                acc_l = accuracy_score(y_true_l, y_pred_l)
                f1_l = f1_score(y_true_l, y_pred_l, average='macro')
                prec_l = precision_score(y_true_l, y_pred_l, zero_division=0)
                rec_l = recall_score(y_true_l, y_pred_l, zero_division=0)
                log_lang[f'test/{key}/acc'] = acc_l
                log_lang[f'test/{key}/macro_f1'] = f1_l
                log_lang[f'test/{key}/auc'] = auc_l
                log_lang[f'test/{key}/precision'] = prec_l
                log_lang[f'test/{key}/recall'] = rec_l
        if log_lang:
            wandb.log(log_lang)

        # 평균 AUC 기반 베스트 모델 저장
        auc_list = [v for v in [en_auc, cn_auc] if v is not None]
        if auc_list:
            avg_auc = sum(auc_list) / len(auc_list)
            print(f"[Test Avg AUC (EN/CN)] {avg_auc:.4f}")
            wandb.log({'test/avg_lang_auc': avg_auc})
            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc
                best_path = f"modelos/{modelName}_best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"✅ Best model updated (avg AUC={best_avg_auc:.4f}) → {best_path}")


if __name__ == "__main__":

    wandb.init(project=os.environ.get('WANDB_PROJECT', 'control_baselines'),
               entity=os.environ.get('WANDB_ENTITY', None),
               name='Multimodal_EN_CN_HC_AD')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainIterator, testIterator = getDataloaders(device, tokenizer)

    model = MultimodalModel()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))

    
    total_steps = len(trainIterator) * nEpochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


    train(model, trainIterator, testIterator, optimizer, scheduler, lossfn)
    torch.save(model, 'modelos/' + modelName)