#!/usr/bin/env python3
"""
영어 파서만 단독 테스트
"""
import sys
import os
sys.path.append('.')

from language_parsers import get_language_parser

def test_english_only():
    data_dir = "../../training_dset"
    print(f"데이터 디렉토리: {data_dir}")
    
    # 영어 파서만 테스트
    print("\n=== 영어 파서 단독 테스트 ===")
    try:
        parser = get_language_parser("English", data_dir)
        print(f"파서 타입: {type(parser)}")
        print(f"파서 언어: {parser.language}")
        print(f"파서 data_dir: {parser.data_dir}")
        
        if hasattr(parser, 'pitt_dir'):
            print(f"파서 pitt_dir: {parser.pitt_dir}")
            print(f"pitt_dir 존재: {parser.pitt_dir.exists()}")
        
        print("\n데이터 파싱 시작...")
        data = parser.parse_data()
        print(f"파싱 완료: {len(data)}개 샘플")
        
        if data:
            print(f"첫 번째 샘플: {data[0]}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_english_only()
