#!/usr/bin/env python3
"""
SigLIP2 치매 진단 모델 훈련 실행 스크립트
간단한 실행을 위한 래퍼 스크립트
"""
import os
import sys
import subprocess
from datetime import datetime

def main():
    """메인 실행 함수"""
    print("=== SigLIP2 치매 진단 모델 훈련 시작 ===")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 기본 설정
    data_dir = "../dementia_fulldata"
    output_dir = "../modules/outputs/siglip"
    model_name = "google/siglip2-base-patch16-224"
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 10
    languages = ["English", "Greek", "Korean"]
    
    # 데이터 디렉토리 확인
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        print("데이터 디렉토리 경로를 확인해주세요.")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # 훈련 명령어 구성
    cmd = [
        "python", "trainer.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--num_epochs", str(num_epochs),
        "--languages"
    ] + languages
    
    print("훈련 설정:")
    print(f"  데이터 디렉토리: {data_dir}")
    print(f"  출력 디렉토리: {output_dir}")
    print(f"  모델: {model_name}")
    print(f"  배치 크기: {batch_size}")
    print(f"  학습률: {learning_rate}")
    print(f"  에포크 수: {num_epochs}")
    print(f"  언어: {', '.join(languages)}")
    print()
    
    # 훈련 실행
    try:
        print("훈련 시작...")
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ 훈련이 성공적으로 완료되었습니다!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 훈련 중 오류가 발생했습니다: {e}")
        return
    except KeyboardInterrupt:
        print("\n⚠️ 훈련이 사용자에 의해 중단되었습니다.")
        return
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
        return
    
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"모델 체크포인트: {os.path.join(output_dir, 'checkpoints')}")

if __name__ == "__main__":
    main() 