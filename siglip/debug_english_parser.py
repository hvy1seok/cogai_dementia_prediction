#!/usr/bin/env python3
"""
영어 파서 디버깅
"""
import os
import sys
sys.path.append('.')

from language_parsers import EnglishParser

def test_english_parser():
    data_dir = "../../training_dset"
    parser = EnglishParser(data_dir)
    
    print(f"Parser language: {parser.language}")
    print(f"Parser data_dir: {parser.data_dir}")
    print(f"Parser pitt_dir: {parser.pitt_dir}")
    print(f"Pitt dir exists: {parser.pitt_dir.exists()}")
    
    textdata_dir = parser.pitt_dir / "textdata"
    voicedata_dir = parser.pitt_dir / "voicedata"
    
    print(f"textdata_dir: {textdata_dir}")
    print(f"textdata_dir exists: {textdata_dir.exists()}")
    print(f"voicedata_dir: {voicedata_dir}")
    print(f"voicedata_dir exists: {voicedata_dir.exists()}")
    
    # 실제 파싱 테스트
    print("\n=== 파싱 테스트 ===")
    data = parser.parse_data()
    print(f"파싱된 데이터 개수: {len(data)}")
    
    if data:
        print(f"첫 번째 샘플: {data[0]}")

if __name__ == "__main__":
    test_english_parser()
