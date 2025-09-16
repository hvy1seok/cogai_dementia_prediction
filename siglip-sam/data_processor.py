"""
SigLIP-SAM용 데이터 처리기
오디오를 멜스펙토그램으로 변환하고 텍스트와 함께 처리
training_dset 폴더 구조에 맞춰 언어별 파서 사용
"""
import os
import torch
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from transformers import AutoProcessor  # SigLIP2 지원
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
import soundfile as sf
from language_parsers import parse_all_languages, get_language_parser

class AudioToMelSpectrogram:
    """오디오를 멜스펙토그램으로 변환하는 클래스"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 fmin: float = 0.0,
                 fmax: float = 8000.0,
                 image_size: int = 224):
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.image_size = image_size
    
    def audio_to_melspectrogram(self, audio_path: str) -> np.ndarray:
        """오디오 파일을 멜스펙토그램으로 변환"""
        try:
            # .npy 파일인 경우 직접 로드
            if audio_path.endswith('.npy'):
                audio = np.load(audio_path)
                if len(audio.shape) > 1:
                    audio = audio.flatten()
            else:
                # 오디오 파일 로드
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 멜스펙토그램 계산
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            # dB 스케일로 변환
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 정규화 (0-1 범위)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            return mel_spec_norm
            
        except Exception as e:
            print(f"❌ 오디오 처리 오류 {audio_path}: {e}")
            # 오류 시 빈 멜스펙토그램 반환
            return np.zeros((self.n_mels, 100))
    
    def melspectrogram_to_image(self, mel_spec: np.ndarray) -> Image.Image:
        """멜스펙토그램을 PIL 이미지로 변환"""
        try:
            # 멜스펙토그램을 3채널 RGB 이미지로 변환
            mel_rgb = np.stack([mel_spec, mel_spec, mel_spec], axis=-1)
            
            # PIL 이미지로 변환
            mel_image = Image.fromarray((mel_rgb * 255).astype(np.uint8))
            
            # 리사이즈
            mel_image = mel_image.resize((self.image_size, self.image_size))
            
            return mel_image
            
        except Exception as e:
            print(f"❌ 이미지 변환 오류: {e}")
            # 오류 시 빈 이미지 반환
            empty_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            return empty_image
    
    def process_audio(self, audio_path: str) -> Image.Image:
        """오디오 파일을 처리해서 PIL 이미지 반환"""
        mel_spec = self.audio_to_melspectrogram(audio_path)
        image = self.melspectrogram_to_image(mel_spec)
        return image

class DementiaDataset(Dataset):
    """치매 진단용 데이터셋"""
    
    def __init__(self, 
                 data_dir: str,
                 processor: AutoProcessor,  # SigLIP2 지원
                 audio_processor: AudioToMelSpectrogram,
                 split: str = "train",
                 max_length: int = 512,
                 languages: Optional[List[str]] = None):
        
        self.data_dir = data_dir
        self.processor = processor
        self.audio_processor = audio_processor
        self.split = split
        self.max_length = max_length
        self.languages = languages or ["English", "Greek", "Spanish", "Mandarin"]
        
        # 데이터 로드
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """데이터 로드 및 전처리 - 언어별 파서 사용"""
        print("언어별 파서를 사용하여 training_dset 데이터 로드 중...")
        
        # 언어별 파서를 사용하여 데이터 파싱
        data = parse_all_languages(self.data_dir, self.languages)
        
        print(f"총 {len(data)}개 샘플 로드 완료")
        
        # 언어별 통계 출력
        language_stats = {}
        for item in data:
            lang = item['language']
            language_stats[lang] = language_stats.get(lang, 0) + 1
        
        print("언어별 샘플 수:")
        for lang, count in language_stats.items():
            print(f"  {lang}: {count}개")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 오디오 파일 경로 확인 및 처리
        audio_path = item['audio_path']
        
        # .npy 파일인 경우 (전처리된 오디오 특징) 직접 로드
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
                print(f"오디오 로드 오류 {audio_path}: {e}")
                # 빈 멜스펙토그램으로 대체
                empty_spec = np.zeros((self.audio_processor.n_mels, 100))
                image = self.audio_processor.melspectrogram_to_image(empty_spec)
        else:
            # 일반 오디오 파일인 경우 멜스펙토그램으로 변환
            mel_spec = self.audio_processor.audio_to_melspectrogram(audio_path)
            image = self.audio_processor.melspectrogram_to_image(mel_spec)
        
        # 텍스트 전처리 (언어 무관을 위해 소문자 변환)
        text = item['text'].lower().strip()
        
        # SigLIP2 프로세서로 처리
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 배치 차원 제거
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)
        
        # SigLIP2는 attention_mask가 없을 수 있으므로 생성
        if 'attention_mask' not in inputs and 'input_ids' in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        # 라벨 추가
        inputs['labels'] = torch.tensor(item['label'], dtype=torch.long)
        inputs['language'] = item['language']
        
        return inputs

def create_stratified_split(dataset, train_split: float = 0.7, val_split: float = 0.1, test_split: float = 0.2, random_seed: int = 42):
    """
    환자 단위 + 언어별 + 라벨별 비율을 유지하면서 stratified split 수행 (train:val:test = 7:1:2)
    Speaker-independent evaluation을 위해 동일 환자의 모든 샘플이 한 세트에만 존재하도록 보장
    """
    # 데이터에서 환자, 언어, 라벨 정보 추출
    patients = []
    languages = []
    labels = []
    
    for i in range(len(dataset)):
        item = dataset.data[i]
        patients.append(item.get('patient_id', item['file_id']))  # 환자 ID, 없으면 file_id 사용
        languages.append(item['language'])
        labels.append(item['label'])
    
    # 환자별로 그룹화
    from collections import defaultdict
    patient_groups = defaultdict(list)
    for i, patient_id in enumerate(patients):
        patient_groups[patient_id].append(i)
    
    print(f"\n👥 환자 단위 분할:")
    print(f"  전체 환자 수: {len(patient_groups)}")
    print(f"  전체 샘플 수: {len(dataset)}")
    
    # 환자별 메타데이터 생성
    patient_metadata = []
    for patient_id, indices in patient_groups.items():
        first_idx = indices[0]
        patient_lang = languages[first_idx]
        patient_label = labels[first_idx]
        sample_count = len(indices)
        
        patient_metadata.append({
            'patient_id': patient_id,
            'language': patient_lang,
            'label': patient_label,
            'indices': indices,
            'sample_count': sample_count
        })
    
    # 환자 단위로 stratify 키 생성 (언어-라벨 조합)
    patient_stratify_keys = [f"{p['language']}_{p['label']}" for p in patient_metadata]
    patient_indices_list = list(range(len(patient_metadata)))
    
    # 환자 단위로 분할 수행
    from sklearn.model_selection import train_test_split
    
    # 첫 번째 분할: train vs (val + test) - 환자 단위
    train_patient_indices, temp_patient_indices = train_test_split(
        patient_indices_list,
        test_size=val_split + test_split,
        stratify=patient_stratify_keys,
        random_state=random_seed
    )
    
    # temp 환자들의 stratify 키 생성
    temp_patient_stratify_keys = [patient_stratify_keys[i] for i in temp_patient_indices]
    
    # 두 번째 분할: val vs test - 환자 단위
    val_patient_indices, test_patient_indices = train_test_split(
        temp_patient_indices,
        test_size=test_split / (val_split + test_split),  # test 비율 조정
        stratify=temp_patient_stratify_keys,
        random_state=random_seed
    )
    
    # 환자 인덱스를 샘플 인덱스로 변환
    train_indices = []
    val_indices = []
    test_indices = []
    
    for patient_idx in train_patient_indices:
        train_indices.extend(patient_metadata[patient_idx]['indices'])
    
    for patient_idx in val_patient_indices:
        val_indices.extend(patient_metadata[patient_idx]['indices'])
    
    for patient_idx in test_patient_indices:
        test_indices.extend(patient_metadata[patient_idx]['indices'])
    
    # 분할 결과 통계 출력 (환자 단위 포함)
    print(f"\n📊 환자 단위 Stratified Split 결과 (7:1:2):")
    print(f"  전체 데이터: {len(dataset)} 샘플, {len(patient_groups)} 환자")
    print(f"  훈련 데이터: {len(train_indices)} 샘플 ({len(train_indices)/len(dataset)*100:.1f}%), {len(train_patient_indices)} 환자")
    print(f"  검증 데이터: {len(val_indices)} 샘플 ({len(val_indices)/len(dataset)*100:.1f}%), {len(val_patient_indices)} 환자")
    print(f"  테스트 데이터: {len(test_indices)} 샘플 ({len(test_indices)/len(dataset)*100:.1f}%), {len(test_patient_indices)} 환자")
    
    # 환자 분할 검증: 중복 확인
    train_patients = set([patient_metadata[i]['patient_id'] for i in train_patient_indices])
    val_patients = set([patient_metadata[i]['patient_id'] for i in val_patient_indices])
    test_patients = set([patient_metadata[i]['patient_id'] for i in test_patient_indices])
    
    overlap_train_val = train_patients & val_patients
    overlap_train_test = train_patients & test_patients
    overlap_val_test = val_patients & test_patients
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f"⚠️ 환자 중복 발견!")
        if overlap_train_val:
            print(f"   Train-Val 중복: {overlap_train_val}")
        if overlap_train_test:
            print(f"   Train-Test 중복: {overlap_train_test}")
        if overlap_val_test:
            print(f"   Val-Test 중복: {overlap_val_test}")
    else:
        print(f"✅ 환자 단위 분할 성공: 중복 없음")
    
    # 훈련/검증/테스트 세트의 언어별 분포 확인
    train_lang_dist = Counter([languages[i] for i in train_indices])
    val_lang_dist = Counter([languages[i] for i in val_indices])
    test_lang_dist = Counter([languages[i] for i in test_indices])
    train_label_dist = Counter([labels[i] for i in train_indices])
    val_label_dist = Counter([labels[i] for i in val_indices])
    test_label_dist = Counter([labels[i] for i in test_indices])
    
    print(f"\n📊 언어별 분포:")
    for lang in set(languages):
        train_count = train_lang_dist[lang]
        val_count = val_lang_dist[lang]
        test_count = test_lang_dist[lang]
        total_count = train_count + val_count + test_count
        if total_count > 0:
            print(f"  {lang}: 훈련 {train_count}개 ({train_count/total_count*100:.1f}%), "
                  f"검증 {val_count}개 ({val_count/total_count*100:.1f}%), "
                  f"테스트 {test_count}개 ({test_count/total_count*100:.1f}%)")
    
    print(f"\n📊 라벨별 분포:")
    label_names = {0: '정상', 1: '치매'}
    for label in [0, 1]:
        train_count = train_label_dist[label]
        val_count = val_label_dist[label]
        test_count = test_label_dist[label]
        total_count = train_count + val_count + test_count
        if total_count > 0:
            print(f"  {label_names[label]}: 훈련 {train_count}개 ({train_count/total_count*100:.1f}%), "
                  f"검증 {val_count}개 ({val_count/total_count*100:.1f}%), "
                  f"테스트 {test_count}개 ({test_count/total_count*100:.1f}%)")
    
    return train_indices, val_indices, test_indices

def create_dataloaders(data_dir: str,
                      processor: AutoProcessor,
                      config,
                      cross_lingual_mode: bool = False,
                      train_languages: List[str] = None,
                      test_languages: List[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """데이터로더 생성 (일반 모드 또는 Cross-Lingual 모드)"""
    
    audio_processor = AudioToMelSpectrogram(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
        image_size=config.image_size
    )
    
    if cross_lingual_mode:
        print("🌍 Cross-Lingual 모드: 언어별로 분리된 훈련/테스트 데이터셋 생성")
        print(f"  훈련 언어: {train_languages}")
        print(f"  테스트 언어: {test_languages}")
        
        # 훈련용 데이터셋 생성 (train + val)
        train_full_dataset = DementiaDataset(
            data_dir=data_dir,
            processor=processor,
            audio_processor=audio_processor,
            max_length=config.max_length,
            languages=train_languages
        )
        
        # 훈련 데이터를 train:val = 7:1로 분할
        train_indices, val_indices, _ = create_stratified_split(
            train_full_dataset,
            train_split=0.875,  # 7/(7+1) = 0.875
            val_split=0.125,    # 1/(7+1) = 0.125
            test_split=0.0,     # Cross-lingual에서는 test는 다른 언어
            random_seed=config.random_seed
        )
        
        train_dataset = Subset(train_full_dataset, train_indices)
        val_dataset = Subset(train_full_dataset, val_indices)
        
        # 테스트용 데이터셋 생성
        test_dataset = DementiaDataset(
            data_dir=data_dir,
            processor=processor,
            audio_processor=audio_processor,
            max_length=config.max_length,
            languages=test_languages
        )
        
        print(f"📊 Cross-Lingual 데이터 분할:")
        print(f"  훈련 데이터: {len(train_dataset)} 샘플 (언어: {train_languages})")
        print(f"  검증 데이터: {len(val_dataset)} 샘플 (언어: {train_languages})")
        print(f"  테스트 데이터: {len(test_dataset)} 샘플 (언어: {test_languages})")
        
    else:
        print("🎯 일반 모드: Stratified Split 수행 중...")
        
        # 전체 데이터셋 생성
        full_dataset = DementiaDataset(
            data_dir=data_dir,
            processor=processor,
            audio_processor=audio_processor,
            max_length=config.max_length,
            languages=config.languages
        )
        
        # Stratified 데이터 분할 (언어별 + 라벨별 비율 유지)
        train_indices, val_indices, test_indices = create_stratified_split(
            full_dataset, 
            train_split=config.train_split,
            val_split=config.val_split,
            test_split=config.test_split,
            random_seed=config.random_seed
        )
        
        # Subset으로 데이터셋 분할
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        print(f"📊 일반 데이터 분할 완료:")
        print(f"  훈련 데이터: {len(train_dataset)} 샘플 ({config.train_split*100:.0f}%)")
        print(f"  검증 데이터: {len(val_dataset)} 샘플 ({config.val_split*100:.0f}%)")
        print(f"  테스트 데이터: {len(test_dataset)} 샘플 ({config.test_split*100:.0f}%)")
        print(f"  전체 데이터: {len(full_dataset)} 샘플")
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader
