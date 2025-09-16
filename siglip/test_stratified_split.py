#!/usr/bin/env python3
"""
Stratified Split 테스트 스크립트
"""
import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SigLIPConfig
from data_processor import create_dataloaders
from transformers import AutoProcessor

def test_stratified_split():
    """Stratified split 테스트"""
    print("🧪 Stratified Split 테스트 시작")
    print("=" * 60)
    
    # 설정
    config = SigLIPConfig()
    config.data_dir = "../../training_dset"
    config.languages = ["English", "Greek", "Spanish", "Mandarin"]
    config.batch_size = 4  # 작은 배치로 테스트
    
    try:
        # 프로세서 로드
        print("🔄 SigLIP2 프로세서 로드 중...")
        processor = AutoProcessor.from_pretrained(config.model_name)
        
        # 데이터로더 생성 (Stratified split 포함)
        print("\n🎯 Stratified Split 데이터로더 생성 중...")
        train_loader, test_loader = create_dataloaders(
            data_dir=config.data_dir,
            processor=processor,
            config=config,
            train_split=0.8,
            test_split=0.2
        )
        
        print(f"\n✅ 데이터로더 생성 완료!")
        print(f"  훈련 배치 수: {len(train_loader)}")
        print(f"  테스트 배치 수: {len(test_loader)}")
        
        # 첫 번째 배치 확인
        print(f"\n🔍 첫 번째 훈련 배치 확인:")
        train_batch = next(iter(train_loader))
        print(f"  배치 크기: {len(train_batch['language'])}")
        print(f"  언어들: {train_batch['language']}")
        print(f"  라벨들: {train_batch['labels'].tolist()}")
        
        print(f"\n🔍 첫 번째 테스트 배치 확인:")
        test_batch = next(iter(test_loader))
        print(f"  배치 크기: {len(test_batch['language'])}")
        print(f"  언어들: {test_batch['language']}")
        print(f"  라벨들: {test_batch['labels'].tolist()}")
        
        print(f"\n🎉 Stratified Split 테스트 성공!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def compare_splits():
    """기존 random split vs stratified split 비교"""
    print("\n" + "=" * 60)
    print("📊 Random Split vs Stratified Split 비교")
    print("=" * 60)
    
    # 이 함수는 실제 데이터가 있을 때만 의미가 있으므로
    # 현재는 설명만 출력
    print("""
🔄 기존 Random Split:
  - 완전 랜덤하게 8:2 분할
  - 언어별/라벨별 비율 보장 없음
  - 예: 훈련에 영어 90%, 테스트에 영어 10%일 수도 있음
  
🎯 새로운 Stratified Split:
  - 언어별 + 라벨별 비율 유지
  - 각 언어마다 정확히 8:2 분할
  - 각 라벨(정상/치매)도 8:2 분할 유지
  - 예: 모든 언어가 훈련 80%, 테스트 20%로 균등 분할
    """)

if __name__ == "__main__":
    test_stratified_split()
    compare_splits()
