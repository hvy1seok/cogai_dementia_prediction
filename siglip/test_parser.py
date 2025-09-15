#!/usr/bin/env python3
"""
SigLIP2 언어별 파서 테스트 스크립트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_parsers import test_parser, parse_all_languages

def main():
    print("=== SigLIP2 언어별 파서 테스트 ===\n")
    
    data_dir = "../../training_dset"
    
    # 개별 언어 테스트
    languages = ['English', 'Greek', 'Spanish', 'Mandarin']
    
    for lang in languages:
        test_parser(lang, data_dir)
        print("-" * 50)
    
    # 전체 언어 통합 테스트
    print("\n=== 전체 언어 통합 테스트 ===")
    try:
        all_data = parse_all_languages(data_dir, languages)
        if all_data:
            print(f"\n전체 샘플 수: {len(all_data)}")
            
            # 언어별 분포
            lang_counts = {}
            for item in all_data:
                lang = item['language']
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            print("언어별 분포:")
            for lang, count in lang_counts.items():
                print(f"  {lang}: {count}개")
            
            # 라벨 분포
            normal_count = sum(1 for item in all_data if item['label'] == 0)
            dementia_count = sum(1 for item in all_data if item['label'] == 1)
            
            print(f"\n라벨 분포:")
            print(f"  정상: {normal_count}개")
            print(f"  치매: {dementia_count}개")
            print(f"  치매 비율: {dementia_count/(normal_count+dementia_count)*100:.1f}%")
            
    except Exception as e:
        print(f"전체 테스트 오류: {e}")

if __name__ == "__main__":
    main()
