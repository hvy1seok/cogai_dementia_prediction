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
import sys
sys.path.append('../siglip')
from language_parsers import parse_all_languages
from models import AudioToMelSpectrogram

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
        
        # siglip 방식 오디오 프로세서 초기화
        self.audio_processor = AudioToMelSpectrogram(
            sample_rate=16000,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            fmin=0.0,
            fmax=8000.0,
            image_size=224
        )
    
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
        
        # 오디오 처리 (Audio-only, Multimodal) - siglip 방식 사용
        if self.mode in ["audio_only", "multimodal"] and self.processor is not None:
            try:
                audio_path = item.get('audio_path', item.get('spectrogram_path', ''))
                
                if audio_path and os.path.exists(audio_path):
                    # .npy 파일인 경우 (전처리된 스펙트로그램)
                    if audio_path.endswith('.npy'):
                        try:
                            audio_spec = np.load(audio_path)
                            # 이미 멜스펙토그램 형태라면 바로 이미지로 변환
                            if len(audio_spec.shape) == 2:
                                image = self.audio_processor.melspectrogram_to_image(audio_spec)
                            else:
                                # 3차원인 경우 (3, H, W) -> (H, W) 변환
                                if audio_spec.shape[0] == 3:
                                    # RGB 채널 평균 또는 첫 번째 채널 사용
                                    audio_spec_2d = np.mean(audio_spec, axis=0)
                                else:
                                    audio_spec_2d = audio_spec[0] if len(audio_spec.shape) == 3 else audio_spec
                                image = self.audio_processor.melspectrogram_to_image(audio_spec_2d)
                        except Exception as e:
                            print(f"NPY 파일 로드 오류 {audio_path}: {e}")
                            # 빈 멜스펙토그램으로 대체
                            empty_spec = np.zeros((128, 100))
                            image = self.audio_processor.melspectrogram_to_image(empty_spec)
                    else:
                        # 일반 오디오 파일인 경우 siglip 방식으로 처리
                        image = self.audio_processor.process_audio_file(audio_path)
                    
                    # 이미지를 ViT용 텐서로 변환
                    pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
                    result['pixel_values'] = pixel_values
                else:
                    # SigLIP 방식: 이미 필터링되어 이 경우는 발생하지 않아야 함
                    raise ValueError(f"오디오 파일 없음 (필터링 오류): {audio_path}")
                    
            except Exception as e:
                # SigLIP 방식: 오류 시 예외 발생 (필터링되어야 할 샘플)
                raise ValueError(f"오디오 처리 오류 (필터링 오류): {e}")
        
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
                    # SigLIP 방식: 이미 필터링되어 이 경우는 발생하지 않아야 함
                    raise ValueError(f"텍스트 없음 (필터링 오류): {item.get('file_id', 'unknown')}")
            except Exception as e:
                # SigLIP 방식: 오류 시 예외 발생 (필터링되어야 할 샘플)
                raise ValueError(f"텍스트 처리 오류 (필터링 오류): {e}")
        
        return result

def load_multilingual_data(data_dir: str, languages: List[str]) -> List[Dict]:
    """멀티링궐 데이터 로드"""
    
    print(f"📂 siglip 파서를 사용하여 다국어 데이터 로드 중: {languages}")
    
    # siglip/language_parsers.py의 parse_all_languages 함수 사용
    data = parse_all_languages(data_dir, languages)
    
    # 대조군 모델 호환성을 위해 필드 추가
    for item in data:
        if 'audio_path' in item and 'spectrogram_path' not in item:
            item['spectrogram_path'] = item['audio_path']
        if 'spectrogram_path' in item and 'audio_available' not in item:
            item['audio_available'] = os.path.exists(item['spectrogram_path']) if item['spectrogram_path'] else False
        
        # patient_id가 없는 경우 file_id에서 추출
        if 'patient_id' not in item or not item['patient_id']:
            file_id = item.get('file_id', 'unknown')
            # 파일명에서 환자 ID 추출 (기본 로직)
            patient_id = file_id.split('_')[0] if '_' in file_id else file_id
            item['patient_id'] = patient_id
    
    print(f"📊 전체 로드된 데이터: {len(data)}개 샘플")
    
    # SigLIP과 동일한 방식: 누락 데이터가 있는 샘플 필터링
    filtered_data = []
    excluded_count = 0
    
    for item in data:
        # 오디오 파일 존재 여부 확인
        audio_available = False
        audio_path = item.get('audio_path') or item.get('spectrogram_path', '')
        if audio_path and os.path.exists(audio_path):
            audio_available = True
        
        # 텍스트 데이터 존재 여부 확인
        text_available = False
        text = item.get('text', '')
        if text and text.strip():
            text_available = True
        
        # 완전한 샘플만 포함 (SigLIP 방식)
        # 멀티모달 실험을 위해 오디오와 텍스트 모두 필요
        if audio_available and text_available:
            filtered_data.append(item)
        else:
            excluded_count += 1
            missing_parts = []
            if not audio_available:
                missing_parts.append("오디오")
            if not text_available:
                missing_parts.append("텍스트")
            print(f"⚠️ 누락 데이터로 인한 샘플 제외: {item.get('file_id', 'unknown')} ({', '.join(missing_parts)} 누락)")
    
    print(f"🔍 데이터 필터링 결과:")
    print(f"  ✅ 완전한 샘플: {len(filtered_data)}개")
    print(f"  ❌ 제외된 샘플: {excluded_count}개")
    print(f"  📊 사용률: {len(filtered_data)/(len(filtered_data)+excluded_count)*100:.1f}%")
    
    # 필터링된 데이터 사용
    data = filtered_data
    
    # 언어별 분포 (필터링 후)
    lang_counts = Counter([item['language'] for item in data])
    print("📈 언어별 분포 (필터링 후):")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}개")
    
    # 라벨별 분포
    label_counts = Counter([item['label'] for item in data])
    print("📈 라벨별 분포:")
    for label, count in label_counts.items():
        label_name = "정상" if label == 0 else "치매"
        print(f"  {label_name}: {count}개")
    
    return data

def load_language_data(lang_dir: str, language: str) -> List[Dict]:
    """특정 언어 데이터 로드"""
    
    data = []
    
    # 영어는 특별한 구조를 가짐
    if language == "English":
        return load_english_data(lang_dir, language)
    
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
        patient_id = item.get('patient_id', item.get('file_id', f'patient_{idx}'))
        key = f"{patient_id}_{item['language']}_{item['label']}"
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

# 더 이상 필요 없음 - siglip의 language_parsers.py 사용

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
