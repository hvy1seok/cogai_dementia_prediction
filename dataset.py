import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from transformers import BertTokenizer

class PittDataset(Dataset):
    def __init__(self, text, attentions, audio_paths, labels):
        self.text = text
        self.attentions = attentions
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        audio_specs = np.load(self.audio_paths[item])
        return self.text[item], self.attentions[item], audio_specs, self.labels[item]

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readline().strip()

def read_data(pitt_path):
    utts = []
    audio_paths = []
    labels = []

    root_path = Path(pitt_path)
    text_root = root_path / "textdata"
    voice_root = root_path / "voicedata"

    categories = ['dem', 'control']
    for cat in categories:
        meta_file = root_path / ('controlCha.txt' if cat == 'control' else 'dementiaCha.txt')
        with open(meta_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            cha_path = Path(line.strip())
            group = cha_path.parts[0]
            subdir = cha_path.parts[1]
            stem = cha_path.stem

            txt_path = text_root / group / subdir / f"{stem}.txt"
            npy_path = voice_root / group / subdir / f"{stem}.npy"

            if txt_path.exists() and npy_path.exists():
                utt = read_file(txt_path)
                utts.append(utt)
                audio_paths.append(str(npy_path))
                labels.append(0 if cat == 'control' else 1)
            else:
                print(f"⚠️ Missing: {txt_path if not txt_path.exists() else npy_path} → skipped.")

    return utts, audio_paths, labels

def collate_fn(batch, pad_val, device, max_seq_len):
    batch_size = len(batch)

    text = torch.LongTensor(batch_size, max_seq_len).fill_(pad_val).to(device)
    attentions = torch.IntTensor(batch_size, max_seq_len).fill_(pad_val).to(device)
    audio = torch.FloatTensor(batch_size, 3, 128, 250).fill_(0).to(device)
    label = torch.FloatTensor(batch_size).fill_(0).to(device)

    for i, (transcript, attentions_d, audio_spec, label_d) in enumerate(batch):
        text[i] = transcript.detach().clone()
        attentions[i] = attentions_d.detach().clone()
        audio[i] = torch.tensor(audio_spec)
        label[i] = label_d

    return text, attentions, audio, label

def prepare_dataset(pitt_path, max_seq_len):
    # BERT 토크나이저 초기화
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 데이터 로드
    utterances, audio_paths, labels = read_data(pitt_path)
    
    # 텍스트 토큰화
    tokenized_inputs = []
    attention_masks = []
    for utt in utterances:
        token = tokenizer.encode_plus(
            utt,
            add_special_tokens=True,
            max_length=max_seq_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        tokenized_inputs.append(token['input_ids'])
        attention_masks.append(token['attention_mask'])
    
    return PittDataset(tokenized_inputs, attention_masks, audio_paths, labels) 