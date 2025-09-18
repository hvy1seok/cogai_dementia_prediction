"""
Data Processor for Control Groups
대조군 모델을 위한 데이터 처리
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoProcessor
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import librosa
import pandas as pd
from collections import defaultdict, Counter

from config import ControlGroupConfig

class ControlGroupDataset(Dataset):
    """대조군 모델을 위한 데이터셋"""
    
    def __init__(self, 
                 data: List[Dict], 
                 config: ControlGroupConfig,
                 tokenizer: Optional[AutoTokenizer] = None,
                 processor: Optional[AutoProcessor] = None,
                 mode: str = "multimodal"):
        """
        Args:
            data: 데이터 리스트
            config: 설정
            tokenizer: 텍스트 토크나이저 (Text-only, Concat에서 사용)
            processor: 이미지 프로세서 (Audio-only, Concat에서 사용)
            mode: "audio_only", "text_only", "multimodal"
        """
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mode = mode
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        result = {
            'label': torch.tensor(item['label'], dtype=torch.long),
            'language': item['language'],
            'patient_id': item.get('patient_id', item['file_id']),
            'file_id': item['file_id']
        }
        
        # 오디오 처리 (Audio-only, Multimodal)
        if self.mode in ["audio_only", "multimodal"] and self.processor is not None:
            try:
                # 스펙트로그램 이미지 로드
                spectrogram_path = item['spectrogram_path']
                if os.path.exists(spectrogram_path):
                    image = Image.open(spectrogram_path).convert('RGB')
                    pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
                    result['pixel_values'] = pixel_values
                else:
                    # 더미 이미지 (224x224x3)
                    result['pixel_values'] = torch.zeros((3, 224, 224), dtype=torch.float32)
            except Exception as e:
                print(f"오디오 처리 오류: {e}")
                result['pixel_values'] = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        # 텍스트 처리 (Text-only, Multimodal)
        if self.mode in ["text_only", "multimodal"] and self.tokenizer is not None:
            try:
                text = item['text']
                if text and text.strip():
                    # 토큰화
                    encoded = self.tokenizer(
                        text,
                        max_length=self.config.max_seq_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    result['input_ids'] = encoded['input_ids'][0]
                    result['attention_mask'] = encoded['attention_mask'][0]
                else:
                    # 빈 텍스트 처리
                    result['input_ids'] = torch.zeros(self.config.max_seq_length, dtype=torch.long)
                    result['attention_mask'] = torch.zeros(self.config.max_seq_length, dtype=torch.long)
            except Exception as e:
                print(f"텍스트 처리 오류: {e}")
                result['input_ids'] = torch.zeros(self.config.max_seq_length, dtype=torch.long)
                result['attention_mask'] = torch.zeros(self.config.max_seq_length, dtype=torch.long)
        
        return result

def load_multilingual_data(data_dir: str, languages: List[str]) -> List[Dict]:
    """멀티링궐 데이터 로드"""
    
    print(f"📂 다국어 데이터 로드 중: {languages}")
    all_data = []
    
    for language in languages:
        lang_dir = os.path.join(data_dir, language)
        if not os.path.exists(lang_dir):
            print(f"⚠️ 언어 디렉토리 없음: {lang_dir}")
            continue
        
        print(f"  📁 {language} 데이터 로드 중...")
        lang_data = load_language_data(lang_dir, language)
        all_data.extend(lang_data)
        print(f"    ✅ {len(lang_data)}개 샘플 로드됨")
    
    print(f"📊 전체 로드된 데이터: {len(all_data)}개 샘플")
    
    # 언어별 분포
    lang_counts = Counter([item['language'] for item in all_data])
    print("📈 언어별 분포:")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}개")
    
    # 라벨별 분포
    label_counts = Counter([item['label'] for item in all_data])
    print("📈 라벨별 분포:")
    for label, count in label_counts.items():
        label_name = "정상" if label == 0 else "치매"
        print(f"  {label_name}: {count}개")
    
    return all_data

def load_language_data(lang_dir: str, language: str) -> List[Dict]:
    """특정 언어 데이터 로드"""
    
    data = []
    
    # 텍스트 데이터 경로
    text_dir = os.path.join(lang_dir, 'textdata')
    voice_dir = os.path.join(lang_dir, 'voicedata')
    
    if not os.path.exists(text_dir):
        print(f"    ⚠️ 텍스트 디렉토리 없음: {text_dir}")
        return data
    
    # HC (정상) 및 AD (치매) 데이터 로드
    for label, label_name in [(0, 'HC'), (1, 'AD')]:
        text_label_dir = os.path.join(text_dir, label_name)
        voice_label_dir = os.path.join(voice_dir, label_name) if os.path.exists(voice_dir) else None
        
        if not os.path.exists(text_label_dir):
            continue
        
        # 텍스트 파일 처리
        for filename in os.listdir(text_label_dir):
            if not filename.endswith('.txt'):
                continue
            
            file_id = filename.replace('.txt', '')
            text_path = os.path.join(text_label_dir, filename)
            
            # 텍스트 읽기
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except:
                continue
            
            if not text:
                continue
            
            # 오디오 파일 경로 찾기
            spectrogram_path = None
            if voice_label_dir and os.path.exists(voice_label_dir):
                # 여러 확장자 시도
                for ext in ['.wav', '.mp3', '.mp4', '.png', '.jpg']:
                    audio_file = os.path.join(voice_label_dir, f"{file_id}{ext}")
                    if os.path.exists(audio_file):
                        # 스펙트로그램 이미지가 있다면 사용, 없으면 오디오 파일 경로 저장
                        if ext in ['.png', '.jpg']:
                            spectrogram_path = audio_file
                        else:
                            # 오디오 파일을 스펙트로그램으로 변환한 경로 (가정)
                            spectrogram_path = audio_file.replace(ext, '_spectrogram.png')
                        break
            
            # 환자 ID 추출 (파일명에서)
            patient_id = file_id.split('_')[0] if '_' in file_id else file_id
            
            data.append({
                'file_id': file_id,
                'patient_id': patient_id,
                'language': language,
                'label': label,
                'text': text,
                'text_path': text_path,
                'spectrogram_path': spectrogram_path or "",
                'audio_available': spectrogram_path is not None and os.path.exists(spectrogram_path)
            })
    
    return data

def create_stratified_split(
    dataset: List[Dict], 
    train_split: float = 0.7, 
    val_split: float = 0.1, 
    test_split: float = 0.2,
    random_seed: int = 42,
    split_by_patient: bool = True
) -> Tuple[List[int], List[int], List[int]]:
    """계층화된 데이터 분할"""
    
    if split_by_patient:
        print("👥 환자 단위 Stratified Split 수행 중...")
        return create_patient_based_split(dataset, train_split, val_split, test_split, random_seed)
    else:
        print("📄 샘플 단위 Stratified Split 수행 중...")
        return create_sample_based_split(dataset, train_split, val_split, test_split, random_seed)

def create_patient_based_split(
    dataset: List[Dict], 
    train_split: float = 0.7, 
    val_split: float = 0.1, 
    test_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """환자 단위 계층화 분할"""
    
    # 환자별 데이터 그룹핑
    patient_groups = defaultdict(list)
    for idx, item in enumerate(dataset):
        key = f"{item['patient_id']}_{item['language']}_{item['label']}"
        patient_groups[key].append(idx)
    
    # 계층화 키 생성
    stratify_keys = []
    patient_indices = []
    
    for key, indices in patient_groups.items():
        _, language, label = key.rsplit('_', 2)
        stratify_key = f"{language}_{label}"
        stratify_keys.append(stratify_key)
        patient_indices.append(indices)
    
    print(f"  전체 환자 그룹: {len(patient_indices)}개")
    print(f"  Stratify 키 분포: {Counter(stratify_keys)}")
    
    # 환자 그룹 분할
    if test_split > 0:
        # 3-way split
        train_patient_idx, temp_patient_idx = train_test_split(
            range(len(patient_indices)),
            test_size=(val_split + test_split),
            stratify=stratify_keys,
            random_state=random_seed
        )
        
        # val/test 분할을 위한 새로운 stratify 키
        temp_stratify_keys = [stratify_keys[i] for i in temp_patient_idx]
        val_ratio = val_split / (val_split + test_split)
        
        val_patient_idx, test_patient_idx = train_test_split(
            temp_patient_idx,
            test_size=(1 - val_ratio),
            stratify=temp_stratify_keys,
            random_state=random_seed
        )
    else:
        # 2-way split (cross-lingual 모드)
        train_patient_idx, val_patient_idx = train_test_split(
            range(len(patient_indices)),
            test_size=val_split,
            stratify=stratify_keys,
            random_state=random_seed
        )
        test_patient_idx = []
    
    # 샘플 인덱스로 변환
    train_indices = []
    for i in train_patient_idx:
        train_indices.extend(patient_indices[i])
    
    val_indices = []
    for i in val_patient_idx:
        val_indices.extend(patient_indices[i])
    
    test_indices = []
    for i in test_patient_idx:
        test_indices.extend(patient_indices[i])
    
    print(f"📈 분할 결과:")
    print(f"  훈련: {len(train_indices)}개 샘플")
    print(f"  검증: {len(val_indices)}개 샘플")
    print(f"  테스트: {len(test_indices)}개 샘플")
    
    return train_indices, val_indices, test_indices

def create_sample_based_split(
    dataset: List[Dict], 
    train_split: float = 0.7, 
    val_split: float = 0.1, 
    test_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """샘플 단위 계층화 분할"""
    
    # 계층화 키 생성
    stratify_keys = []
    for item in dataset:
        stratify_key = f"{item['language']}_{item['label']}"
        stratify_keys.append(stratify_key)
    
    print(f"  전체 샘플: {len(dataset)}개")
    print(f"  Stratify 키 분포: {Counter(stratify_keys)}")
    
    indices = list(range(len(dataset)))
    
    if test_split > 0:
        # 3-way split
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(val_split + test_split),
            stratify=stratify_keys,
            random_state=random_seed
        )
        
        temp_stratify_keys = [stratify_keys[i] for i in temp_indices]
        val_ratio = val_split / (val_split + test_split)
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_ratio),
            stratify=temp_stratify_keys,
            random_state=random_seed
        )
    else:
        # 2-way split
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split,
            stratify=stratify_keys,
            random_state=random_seed
        )
        test_indices = []
    
    print(f"📈 분할 결과:")
    print(f"  훈련: {len(train_indices)}개 샘플")
    print(f"  검증: {len(val_indices)}개 샘플")
    print(f"  테스트: {len(test_indices)}개 샘플")
    
    return train_indices, val_indices, test_indices

def compute_class_weights(dataset: Union[List[Dict], Subset], config: ControlGroupConfig) -> np.ndarray:
    """클래스 가중치 계산"""
    
    if isinstance(dataset, Subset):
        labels = [dataset.dataset.data[i]['label'] for i in dataset.indices]
    else:
        labels = [item['label'] for item in dataset]
    
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    
    print(f"📊 클래스 가중치 계산:")
    for label, weight in zip(unique_labels, class_weights):
        label_name = "정상(HC)" if label == 0 else "치매(AD)"
        print(f"  {label_name}: {weight:.4f}")
    
    return class_weights

def create_dataloaders(
    config: ControlGroupConfig,
    mode: str = "multimodal",
    tokenizer: Optional[AutoTokenizer] = None,
    processor: Optional[AutoProcessor] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """데이터로더 생성"""
    
    # 데이터 로드
    all_data = load_multilingual_data(config.data_dir, config.languages)
    
    if len(all_data) == 0:
        raise ValueError("로드된 데이터가 없습니다!")
    
    # 데이터 분할
    train_indices, val_indices, test_indices = create_stratified_split(
        all_data,
        split_by_patient=config.split_by_patient,
        random_seed=config.random_seed
    )
    
    # 데이터셋 생성
    full_dataset = ControlGroupDataset(all_data, config, tokenizer, processor, mode)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices) if test_indices else None
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader
