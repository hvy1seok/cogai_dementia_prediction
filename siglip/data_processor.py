"""
SigLIP2용 데이터 처리기
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
from torch.utils.data import Dataset, DataLoader
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
            # 오디오 로드
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
            
            return mel_spec_db
            
        except Exception as e:
            print(f"오디오 처리 중 오류 발생: {e}")
            # 빈 멜스펙토그램 반환
            return np.zeros((self.n_mels, 100))
    
    def melspectrogram_to_image(self, mel_spec: np.ndarray) -> Image.Image:
        """멜스펙토그램을 PIL 이미지로 변환"""
        # 정규화 (0-255 범위로)
        mel_spec_norm = ((mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) * 255).astype(np.uint8)
        
        # PIL 이미지로 변환
        image = Image.fromarray(mel_spec_norm, mode='L')
        
        # RGB로 변환 (SigLIP2는 RGB 이미지를 요구)
        image_rgb = image.convert('RGB')
        
        # 크기 조정
        image_resized = image_rgb.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        return image_resized

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
    
    def __len__(self) -> int:
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
        
        # 디버깅 완료 - 제거됨
        
        # 라벨 추가
        inputs['labels'] = torch.tensor(item['label'], dtype=torch.long)
        inputs['language'] = item['language']
        
        return inputs

def create_dataloaders(data_dir: str,
                      processor: AutoProcessor,  # SigLIP2 지원
                      config,
                      train_split: float = 0.8,
                      test_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """데이터로더 생성"""
    
    audio_processor = AudioToMelSpectrogram(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
        image_size=config.image_size
    )
    
    # 전체 데이터셋 생성
    full_dataset = DementiaDataset(
        data_dir=data_dir,
        processor=processor,
        audio_processor=audio_processor,
        max_length=config.max_length,
        languages=config.languages
    )
    
    # 데이터 분할 (Train:Test = 8:2)
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 데이터 분할 완료:")
    print(f"  훈련 데이터: {len(train_dataset)} 샘플 ({train_split*100:.0f}%)")
    print(f"  테스트 데이터: {len(test_dataset)} 샘플 ({test_split*100:.0f}%)")
    print(f"  전체 데이터: {total_size} 샘플")
    
    return train_loader, test_loader 