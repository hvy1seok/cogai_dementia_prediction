import os
import subprocess
from pathlib import Path

# 설정 경로
pittPath = Path("../Pitt/voicedata/")
meta_files = ['controlCha.txt', 'dementiaCha.txt']

def convert_mp3_to_wav(mp3_path: Path):
    wav_path = mp3_path.with_suffix('.wav')
    if wav_path.exists():
        print(f"✔️ WAV already exists: {wav_path}")
        return

    # ffmpeg 명령 실행
    command = [
        "ffmpeg",
        "-y",  # 덮어쓰기 허용
        "-i", str(mp3_path),
        str(wav_path)
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted: {mp3_path} -> {wav_path}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {mp3_path}")

def main():
    for meta_file in meta_files:
        with open(pittPath / meta_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            cha_path = Path(line.strip())  # 예: Control/cookie/001-0.cha
            mp3_path = pittPath / cha_path.with_suffix('.mp3')

            if mp3_path.exists():
                convert_mp3_to_wav(mp3_path)
            else:
                print(f"⚠️ MP3 not found: {mp3_path}")

if __name__ == "__main__":
    main()
