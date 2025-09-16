import warnings
import numpy as np
import torch
import torchaudio
import torchvision
from PIL import Image
import os
from pathlib import Path

warnings.filterwarnings('ignore')

def extract_spectrogram(audio_path: Path):
    """오디오 파일에서 스펙트로그램 추출"""
    try:
        # mp3/wav 파일 로드
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # 🎯 스테레오 → 모노 (채널 평균)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        window_sizes = [25, 50, 100]
        hop_sizes = [10, 25, 50]

        specs = []

        for i in range(len(window_sizes)):
            window_length = int(round(window_sizes[i] * sample_rate / 1000))
            hop_length = int(round(hop_sizes[i] * sample_rate / 1000))
            n_fft = 2 ** (window_length - 1).bit_length()

            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=window_length,
                hop_length=hop_length,
                n_mels=128
            )(waveform)

            spec = spec.squeeze(0).numpy()
            spec = np.log(spec + 1e-6)

            # Normalize for PIL
            spec_img = (255 * (spec - np.min(spec)) / (np.max(spec) - np.min(spec))).astype(np.uint8)

            resized = torchvision.transforms.Resize((128, 250))(Image.fromarray(spec_img))
            specs.append(np.array(resized))

        specs = np.array(specs)
        save_path = audio_path.with_suffix('.npy')
        np.save(save_path, specs)
        print(f"✅ Saved: {save_path}")
        return True

    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {str(e)}")
        return False

def process_directory(root_dir: Path):
    """디렉토리 내의 모든 mp3/wav 파일 처리"""
    # 처리 통계
    total_files = 0
    processed_files = 0
    failed_files = 0
    skipped_files = 0

    # 0wav와 0extra 폴더는 건너뛰기
    skip_dirs = {'0wav', '0extra'}

    for audio_path in root_dir.rglob('*.[mw][pa][3v]'):  # mp3 또는 wav 파일 매칭
        total_files += 1
        
        # 0wav나 0extra 폴더 내 파일은 건너뛰기
        if any(skip_dir in str(audio_path.parent) for skip_dir in skip_dirs):
            skipped_files += 1
            continue

        # 이미 npy 파일이 있으면 건너뛰기
        if audio_path.with_suffix('.npy').exists():
            skipped_files += 1
            print(f"⏩ Already exists: {audio_path.with_suffix('.npy')}")
            continue

        print(f"🔄 Processing: {audio_path}")
        if extract_spectrogram(audio_path):
            processed_files += 1
        else:
            failed_files += 1

    # 처리 결과 출력
    print("\n=== 처리 완료 ===")
    print(f"총 파일 수: {total_files}")
    print(f"성공: {processed_files}")
    print(f"실패: {failed_files}")
    print(f"건너뛴 파일: {skipped_files}")

def preprocessAudio():
    """전체 전처리 실행"""
    root_dir = Path("../fulldata_71/dementia_fulldata")
    if not root_dir.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {root_dir}")
        return

    print(f"🎯 처리 시작: {root_dir}")
    process_directory(root_dir)

# 실행
if __name__ == "__main__":
    preprocessAudio()