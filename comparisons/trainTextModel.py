import torch
from torch.utils.data import DataLoader
from functools import partial
import time
import torch.nn as nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import wandb


seed_val = 0

torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True

nEpochs = 20
data_root = '../training_dset/'  # training_dset 기준
batchSize = 16
maxSeqLen = 64  # SigLIP 호환 고려, 빠른 대조군용
textModel = 1
modelName = 'Texto'+str(textModel)+str(nEpochs)

def read_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return " ".join([ln.strip() for ln in lines])

def readIdx():
    utts = []
    labels = []

    pairs = [
        ("English", "HC", 0),
        ("English", "AD", 1),
        ("Mandarin", "HC", 0),
        ("Mandarin", "AD", 1),
    ]

    for lang, cls, lab in pairs:
        txt_dir = os.path.join(data_root, lang, 'textdata', cls)
        if not os.path.isdir(txt_dir):
            print(f"⚠️ Missing dir: {txt_dir}")
            continue
        for fname in os.listdir(txt_dir):
            if not fname.endswith('.txt'):
                continue
            utt = read_txt(os.path.join(txt_dir, fname))
            utts.append(utt)
            labels.append(lab)
    return utts, labels


class Dataset:
    def __init__(self, text, attentions, labels):
        self.text = text
        self.attentions = attentions
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item], self.attentions[item], self.labels[item]
            


def collate_fn(batch, padVal, device):

    batchSize = len(batch)

    text = torch.LongTensor(batchSize, maxSeqLen).fill_(padVal).to(device)
    attentions = torch.IntTensor(batchSize, maxSeqLen).fill_(padVal).to(device)
    label = torch.FloatTensor(batchSize).fill_(0).to(device)

    for i, (transcript, attentionsD, labelD) in enumerate(batch):
        text[i] = transcript.detach().clone()
        attentions[i] = attentionsD.detach().clone()
        label[i] = labelD

    return text, attentions, label


def getDataloaders(device, tokenizer):
    utterances, labels = readIdx()

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

    dataset = Dataset(tokenized_inputs, attention_masks, labels)

    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    trainDataset, validationDataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    collate_fn_ = partial(collate_fn, device=device, padVal=0)
    trainIterator = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    validationIterator = DataLoader(validationDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    return trainIterator, validationIterator


class text(nn.Module):
    def __init__(self, dModel = 1024, dropout = 0.2):
      super(text, self).__init__()
      self.textModel = BertModel.from_pretrained('bert-base-uncased')
      self.dModel = dModel
      if textModel == 2:
        self.lstm = nn.LSTM(768, dModel, 1, batch_first = True, bidirectional = True)
      self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dModel if textModel == 2 else 768, 1),
              )

    def forward(self, text, attention):
      textOut = self.textModel(text, attention_mask=attention)[0]
      if textModel == 1:
        output = textOut[:,0,:]
      elif textModel == 2:
        packed_input = pack_padded_sequence(textOut, [maxSeqLen for _ in range(text.size(0))], batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output = output[range(len(output)), [maxSeqLen - 1 for _ in range(text.size(0))] ,:self.dModel]

      output = self.classifier(output)
      return output

def train(model, trainIterator, valiadtionIterator, optimizer, scheduler, lossfn):

    for epoch in range(nEpochs):
        start = time.time()
        epoch_loss = 0
        train_y_true = []
        train_y_pred = []
        epoch_size = 0
        train_y_prob = []
        for i, (text, attention, labels) in enumerate(trainIterator):
            outputs = model(text, attention)
 
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
        for i, (text, attention, labels) in enumerate(valiadtionIterator):
           
            with torch.no_grad():
                outputs = model(text, attention)
            
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
            
        end = time.time()
        val_acc = accuracy_score(val_y_true, val_y_pred) if val_y_true else 0.0
        val_f1 = f1_score(val_y_true, val_y_pred, average='macro') if val_y_true else 0.0
        print('Validation Epoch: ', epoch + 1, ' | in ', end - start, ' seconds')
        print('Validation Loss: ', epoch_loss / max(1, len(validationIterator)))
        print('Validation Acc: ', round(val_acc * 100, 3))
        print('Validation Macro-F1: ', round(val_f1, 4))
        try:
            val_auc = roc_auc_score(val_y_true, val_y_prob) if val_y_true else 0.0
        except Exception:
            val_auc = 0.0
        val_prec = precision_score(val_y_true, val_y_pred, zero_division=0) if val_y_true else 0.0
        val_rec = recall_score(val_y_true, val_y_pred, zero_division=0) if val_y_true else 0.0
        print('Validation AUC: ', round(val_auc, 4))
        print('Validation Precision: ', round(val_prec, 4))
        print('Validation Recall: ', round(val_rec, 4))

        print('Validation confusion_matrix: ', conf_matrix)

        # W&B logging
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': epoch_loss / max(1, len(trainIterator)),
            'train/acc': train_acc,
            'train/macro_f1': train_f1,
            'train/auc': train_auc,
            'train/precision': train_prec,
            'train/recall': train_rec,
            'val/loss': epoch_loss / max(1, len(validationIterator)),
            'val/acc': val_acc,
            'val/macro_f1': val_f1,
            'val/auc': val_auc,
            'val/precision': val_prec,
            'val/recall': val_rec,
        })

if __name__ == "__main__":

    wandb.init(project=os.environ.get('WANDB_PROJECT', 'control_baselines'),
               entity=os.environ.get('WANDB_ENTITY', None),
               name='Text_EN_CN_HC_AD')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainIterator, validationIterator = getDataloaders(device, tokenizer)

    model = text()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.78))

    
    total_steps = len(trainIterator) * nEpochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


    train(model, trainIterator, validationIterator, optimizer, scheduler, lossfn)
    torch.save(model, 'modelos/' + modelName)
