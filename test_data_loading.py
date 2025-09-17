#!/usr/bin/env python3
"""
데이터 로딩 테스트 스크립트
"""

from dataset_multilingual import read_multilingual_data, prepare_multilingual_dataset
from pathlib import Path
import sys

def test_data_loading():
    """데이터 로딩 테스트"""
    
    print("🧪 데이터 로딩 테스트 시작...")
    
    data_dir = "../training_dset"
    
    # 각 언어별로 개별 테스트
    languages = ['English', 'Greek', 'Spanish', 'Mandarin']
    
    for language in languages:
        print(f"\n🌍 {language} 데이터 로딩 테스트...")
        
        try:
            # 단일 언어 로드
            raw_data = read_multilingual_data(data_dir, [language])
            
            if len(raw_data) > 0:
                print(f"  ✅ {len(raw_data)}개 샘플 로드됨")
                
                # 첫 번째 샘플 확인
                sample = raw_data[0]
                print(f"  📄 샘플 예시:")
                print(f"    텍스트: {sample['text'][:100]}...")
                print(f"    오디오 경로: {sample['audio_path']}")
                print(f"    라벨: {sample['label']}")
                print(f"    환자 ID: {sample['patient_id']}")
                
                # 라벨 분포 확인
                labels = [item['label'] for item in raw_data]
                normal_count = labels.count(0)
                dementia_count = labels.count(1)
                print(f"  📊 라벨 분포: 정상 {normal_count}개, 치매 {dementia_count}개")
                
                # 오디오 파일 존재 확인
                audio_exists = 0
                for item in raw_data[:10]:  # 처음 10개만 확인
                    if Path(item['audio_path']).exists():
                        audio_exists += 1
                
                print(f"  🎵 오디오 파일 존재율: {audio_exists}/10")
                
            else:
                print(f"  ❌ 데이터 없음")
                
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
    
    print(f"\n🌍 전체 언어 통합 테스트...")
    
    try:
        # 전체 언어 로드
        all_data = read_multilingual_data(data_dir, languages)
        
        if len(all_data) > 0:
            print(f"  ✅ 총 {len(all_data)}개 샘플 로드됨")
            
            # 언어별 분포
            from collections import Counter
            lang_dist = Counter([item['language'] for item in all_data])
            print(f"  📊 언어별 분포:")
            for lang, count in lang_dist.items():
                print(f"    {lang}: {count}개")
            
            # 라벨 분포
            label_dist = Counter([item['label'] for item in all_data])
            print(f"  📊 전체 라벨 분포:")
            print(f"    정상: {label_dist[0]}개")
            print(f"    치매: {label_dist[1]}개")
            
            print(f"\n✅ 데이터 로딩 테스트 완료!")
            return True
            
        else:
            print(f"  ❌ 전체 데이터 없음")
            return False
            
    except Exception as e:
        print(f"  ❌ 전체 데이터 로딩 오류: {e}")
        return False

def test_tokenization():
    """토큰화 테스트"""
    print(f"\n🔤 토큰화 테스트...")
    
    try:
        # 작은 샘플로 토큰화 테스트
        dataset = prepare_multilingual_dataset(
            data_dir="../training_dset",
            max_seq_len=128,
            languages=['English'],  # 영어만으로 테스트
        )
        
        if len(dataset) > 0:
            print(f"  ✅ {len(dataset)}개 샘플 토큰화 완료")
            
            # 첫 번째 샘플 확인
            sample = dataset[0]
            print(f"  📄 토큰화된 샘플:")
            print(f"    input_ids shape: {sample['input_ids'].shape}")
            print(f"    attention_mask shape: {sample['attention_mask'].shape}")
            print(f"    audio shape: {sample['audio'].shape}")
            print(f"    label: {sample['label']}")
            print(f"    language: {sample['language']}")
            
            return True
        else:
            print(f"  ❌ 토큰화된 데이터 없음")
            return False
            
    except Exception as e:
        print(f"  ❌ 토큰화 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("🧪 다국어 멀티모달 데이터셋 테스트")
    print("="*60)
    
    # 데이터 로딩 테스트
    loading_success = test_data_loading()
    
    if loading_success:
        # 토큰화 테스트
        tokenization_success = test_tokenization()
        
        if tokenization_success:
            print(f"\n🎉 모든 테스트 통과!")
            print(f"이제 main_multilingual.py를 실행할 수 있습니다.")
            sys.exit(0)
        else:
            print(f"\n❌ 토큰화 테스트 실패")
            sys.exit(1)
    else:
        print(f"\n❌ 데이터 로딩 테스트 실패")
        sys.exit(1)
