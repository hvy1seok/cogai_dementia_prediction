import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from omegaconf import DictConfig
import logging
import os

logger = logging.getLogger(__name__)

class MultilingualDataset(Dataset):
    def __init__(self, text, attentions, audio_paths, labels, languages):
        self.text = text
        self.attentions = attentions
        self.audio_paths = audio_paths
        self.labels = labels
        self.languages = languages

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        audio_specs = np.load(self.audio_paths[item])
        return (
            self.text[item], 
            self.attentions[item], 
            audio_specs, 
            self.labels[item],
            self.languages[item]
        )

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readline().strip()

def get_tokenizer(language):
    """언어별 적절한 BERT 토크나이저 반환"""
    tokenizer_map = {
        'English': 'bert-base-uncased',
        'German': 'bert-base-german-cased',
        'Spanish': 'dccuchile/bert-base-spanish-wwm-cased',
        'Chinese': 'bert-base-chinese',  # Mandarin용
        'Greek': 'nlpaueb/bert-base-greek-uncased-v1',
        'Taiwanese': 'bert-base-chinese',  # Taiwanese용 (중국어 토크나이저 사용)
    }
    return AutoTokenizer.from_pretrained(tokenizer_map.get(language, 'bert-base-multilingual-cased'))

def read_pitt_style_data(root_path, group_name):
    """Pitt 스타일의 데이터셋 읽기 (예: Pitt, Hopkins 등)"""
    utts = []
    audio_paths = []
    labels = []
    languages = []

    text_root = root_path / "textdata"
    voice_root = root_path / "voicedata"

    categories = ['dem', 'control']
    for cat in categories:
        meta_file = root_path / ('controlCha.txt' if cat == 'control' else 'dementiaCha.txt')
        if not meta_file.exists():
            continue

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
                languages.append(group_name)
            else:
                logger.warning(f"⚠️ Missing: {txt_path if not txt_path.exists() else npy_path} → skipped.")

    return utts, audio_paths, labels, languages

def read_general_data(root_path, group_name):
    """일반적인 구조의 데이터셋 읽기"""
    utts = []
    audio_paths = []
    labels = []
    languages = []

    # Control/HC 그룹
    control_paths = list((root_path / group_name).glob("**/Control/**/*.npy"))
    control_paths.extend(list((root_path / group_name).glob("**/HC/**/*.npy")))
    
    # Dementia/AD/MCI 그룹
    dementia_paths = list((root_path / group_name).glob("**/Dementia/**/*.npy"))
    dementia_paths.extend(list((root_path / group_name).glob("**/AD/**/*.npy")))
    dementia_paths.extend(list((root_path / group_name).glob("**/MCI/**/*.npy")))

    for npy_path in control_paths + dementia_paths:
        txt_path = npy_path.with_suffix('.txt')
        if txt_path.exists():
            utt = read_file(txt_path)
            utts.append(utt)
            audio_paths.append(str(npy_path))
            labels.append(0 if npy_path in control_paths else 1)
            languages.append(group_name)
        else:
            logger.warning(f"⚠️ Missing text file for {npy_path} → skipped.")

    return utts, audio_paths, labels, languages

def read_data(cfg: DictConfig):
    """데이터 읽기 함수"""
    all_utts = []
    all_audio_paths = []
    all_labels = []
    all_languages = []

    root_path = Path(cfg.dataset.root_path)
    selected_languages = cfg.dataset.languages

    # 언어별 데이터 로딩
    for lang in selected_languages:
        lang_path = root_path / lang
        if not lang_path.exists():
            logger.warning(f"⚠️ Language directory not found: {lang}")
            continue

        # Pitt 스타일 데이터셋 체크
        if (lang_path / "controlCha.txt").exists() or (lang_path / "dementiaCha.txt").exists():
            utts, audio_paths, labels, languages = read_pitt_style_data(lang_path, lang)
        else:
            utts, audio_paths, labels, languages = read_general_data(lang_path, lang)

        all_utts.extend(utts)
        all_audio_paths.extend(audio_paths)
        all_labels.extend(labels)
        all_languages.extend(languages)

    return all_utts, all_audio_paths, all_labels, all_languages

def collate_fn(batch, pad_val, device, max_seq_len):
    batch_size = len(batch)

    text = torch.LongTensor(batch_size, max_seq_len).fill_(pad_val).to(device)
    attentions = torch.IntTensor(batch_size, max_seq_len).fill_(pad_val).to(device)
    audio = torch.FloatTensor(batch_size, 3, 128, 250).fill_(0).to(device)
    label = torch.FloatTensor(batch_size).fill_(0).to(device)
    language = []

    for i, (transcript, attentions_d, audio_spec, label_d, lang) in enumerate(batch):
        text[i] = transcript.detach().clone()
        attentions[i] = attentions_d.detach().clone()
        audio[i] = torch.tensor(audio_spec)
        label[i] = label_d
        language.append(lang)

    return text, attentions, audio, label, language

def prepare_dataset(cfg: DictConfig):
    """데이터셋 준비 함수"""
    # 데이터 로드
    utterances, audio_paths, labels, languages = read_data(cfg)
    
    # 언어별 토크나이저로 텍스트 토큰화
    tokenized_inputs = []
    attention_masks = []
    
    for utt, lang in zip(utterances, languages):
        tokenizer = get_tokenizer(lang)
        token = tokenizer.encode_plus(
            utt,
            add_special_tokens=True,
            max_length=cfg.dataset.max_seq_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        tokenized_inputs.append(token['input_ids'])
        attention_masks.append(token['attention_mask'])
    
    return MultilingualDataset(
        tokenized_inputs, 
        attention_masks, 
        audio_paths, 
        labels,
        languages
    ) 