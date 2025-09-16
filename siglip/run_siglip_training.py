#!/usr/bin/env python3
"""
SigLIP2 치매 진단 모델 훈련 실행 스크립트
training_dset 폴더 구조에 맞춰 언어별 파서 선택 가능
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
    data_dir = "../training_dset"
    output_dir = "../modules/outputs/siglip"
    model_name = "google/siglip2-base-patch16-224"
    batch_size = 8  # SigLIP2는 메모리를 많이 사용하므로 작게 시작
    learning_rate = 2e-5
    num_epochs = 5  # 테스트용으로 작게 시작
    
    # 사용 가능한 파서 목록
    available_parsers = ["all", "English", "Greek", "Spanish", "Mandarin"]
    
    print("\n사용 가능한 언어 파서:")
    for i, parser in enumerate(available_parsers, 1):
        print(f"  {i}. {parser}")
    
    # 파서 선택
    while True:
        try:
            choice = input(f"\n사용할 파서를 선택하세요 (1-{len(available_parsers)}): ").strip()
            parser_idx = int(choice) - 1
            if 0 <= parser_idx < len(available_parsers):
                selected_parser = available_parsers[parser_idx]
                break
            else:
                print("올바른 번호를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n훈련이 취소되었습니다.")
            return
    
    print(f"\n선택된 파서: {selected_parser}")
    
    # 데이터 디렉토리 확인
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        print("데이터 디렉토리 경로를 확인해주세요.")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # 먼저 파서 테스트 실행
    print("\n=== 데이터 파서 테스트 ===")
    try:
        test_result = subprocess.run([
            "python", "test_parser.py"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if test_result.returncode == 0:
            print("✅ 데이터 파서 테스트 성공")
            print(test_result.stdout)
        else:
            print("⚠️ 데이터 파서 테스트에서 경고가 있습니다:")
            print(test_result.stdout)
            if test_result.stderr:
                print("오류:", test_result.stderr)
    except Exception as e:
        print(f"⚠️ 파서 테스트 실행 중 오류: {e}")
    
    # 사용자 확인
    proceed = input("\n훈련을 계속 진행하시겠습니까? (y/N): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("훈련이 취소되었습니다.")
        return
    
    # 훈련 명령어 구성
    cmd = [
        "python", "trainer.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--num_epochs", str(num_epochs),
        "--parser", selected_parser
    ]
    
    print("\n훈련 설정:")
    print(f"  데이터 디렉토리: {data_dir}")
    print(f"  출력 디렉토리: {output_dir}")
    print(f"  모델: {model_name}")
    print(f"  배치 크기: {batch_size}")
    print(f"  학습률: {learning_rate}")
    print(f"  에포크 수: {num_epochs}")
    print(f"  선택된 파서: {selected_parser}")
    print()
    
    # 훈련 실행
    try:
        print("훈련 시작...")
        print("명령어:", " ".join(cmd))
        print("=" * 60)
        
        result = subprocess.run(cmd, check=False, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✅ 훈련이 성공적으로 완료되었습니다!")
        else:
            print(f"⚠️ 훈련이 오류와 함께 종료되었습니다 (exit code: {result.returncode})")
            print("자세한 내용은 위의 로그를 확인해주세요.")
        
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
