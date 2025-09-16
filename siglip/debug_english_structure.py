#!/usr/bin/env python3
"""
영어 데이터 구조 디버깅
"""
import os
from pathlib import Path

def debug_english_structure():
    print("=== 영어 데이터 구조 디버깅 ===")
    
    data_dir = Path("../../training_dset")
    english_dir = data_dir / "English"
    
    print(f"English 디렉토리: {english_dir}")
    print(f"존재여부: {english_dir.exists()}")
    
    if english_dir.exists():
        print(f"English 디렉토리 내용:")
        for item in english_dir.iterdir():
            print(f"  - {item.name} ({'디렉토리' if item.is_dir() else '파일'})")
        
        textdata_dir = english_dir / "textdata"
        voicedata_dir = english_dir / "voicedata"
        
        print(f"\ntextdata 디렉토리: {textdata_dir}")
        print(f"존재여부: {textdata_dir.exists()}")
        
        if textdata_dir.exists():
            print(f"textdata 내용:")
            for item in textdata_dir.iterdir():
                print(f"  - {item.name} ({'디렉토리' if item.is_dir() else '파일'})")
            
            # HC, AD 폴더 확인
            for category in ['HC', 'AD']:
                cat_dir = textdata_dir / category
                print(f"\n{category} 디렉토리: {cat_dir}")
                print(f"존재여부: {cat_dir.exists()}")
                
                if cat_dir.exists():
                    print(f"{category} 내용 (처음 10개):")
                    items = list(cat_dir.iterdir())
                    for i, item in enumerate(items[:10]):
                        print(f"  - {item.name} ({'디렉토리' if item.is_dir() else '파일'})")
                    
                    # .txt 파일 개수 확인
                    txt_files = list(cat_dir.glob("*.txt"))
                    print(f"{category}에서 .txt 파일 개수: {len(txt_files)}")
                    
                    if txt_files:
                        print(f"첫 번째 .txt 파일: {txt_files[0].name}")
                    
                    # 하위 디렉토리들 확인
                    subdirs = [item for item in cat_dir.iterdir() if item.is_dir()]
                    if subdirs:
                        print(f"{category}의 하위 디렉토리들:")
                        for subdir in subdirs[:5]:  # 처음 5개만
                            print(f"  - {subdir.name}/")
                            subdir_txt_files = list(subdir.glob("*.txt"))
                            print(f"    .txt 파일 개수: {len(subdir_txt_files)}")
        
        print(f"\nvoicedata 디렉토리: {voicedata_dir}")
        print(f"존재여부: {voicedata_dir.exists()}")
        
        if voicedata_dir.exists():
            print(f"voicedata 내용:")
            for item in voicedata_dir.iterdir():
                print(f"  - {item.name} ({'디렉토리' if item.is_dir() else '파일'})")

if __name__ == "__main__":
    debug_english_structure()
