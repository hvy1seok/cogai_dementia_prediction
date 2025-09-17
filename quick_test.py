#!/usr/bin/env python3
"""
간단한 데이터 로딩 테스트
"""

from dataset_multilingual import read_multilingual_data

def test_quick():
    """빠른 테스트"""
    
    print("🧪 빠른 데이터 로딩 테스트...")
    
    try:
        # 영어만 테스트
        data = read_multilingual_data("../training_dset", ['English'])
        print(f"영어 데이터: {len(data)}개")
        
        if len(data) > 0:
            sample = data[0]
            print(f"첫 번째 샘플: {sample['patient_id']}, 라벨: {sample['label']}")
            print(f"텍스트: {sample['text'][:50]}...")
        
        return len(data) > 0
        
    except Exception as e:
        print(f"오류: {e}")
        return False

if __name__ == "__main__":
    success = test_quick()
    if success:
        print("✅ 성공!")
    else:
        print("❌ 실패!")
