#!/usr/bin/env python3
"""
디버깅용 파서 테스트
"""
import os
from pathlib import Path

def debug_english_parser():
    data_dir = Path("../../training_dset")
    print(f"기본 데이터 디렉토리: {data_dir}")
    print(f"절대 경로: {data_dir.absolute()}")
    print(f"존재 여부: {data_dir.exists()}")
    
    english_dir = data_dir / "English"
    print(f"\n영어 디렉토리: {english_dir}")
    print(f"존재 여부: {english_dir.exists()}")
    
    if english_dir.exists():
        print(f"영어 디렉토리 내용: {list(english_dir.iterdir())}")
        
        pitt_dir = english_dir / "Pitt"
        print(f"\nPitt 디렉토리: {pitt_dir}")
        print(f"존재 여부: {pitt_dir.exists()}")
        
        if pitt_dir.exists():
            print(f"Pitt 디렉토리 내용: {list(pitt_dir.iterdir())}")
            
            textdata_dir = pitt_dir / "textdata"
            voicedata_dir = pitt_dir / "voicedata"
            
            print(f"\ntextdata 디렉토리: {textdata_dir}")
            print(f"존재 여부: {textdata_dir.exists()}")
            
            print(f"\nvoicedata 디렉토리: {voicedata_dir}")
            print(f"존재 여부: {voicedata_dir.exists()}")
            
            if textdata_dir.exists() and voicedata_dir.exists():
                print(f"\ntextdata 내용: {list(textdata_dir.iterdir())}")
                print(f"voicedata 내용: {list(voicedata_dir.iterdir())}")
                
                hc_text = textdata_dir / "HC"
                hc_voice = voicedata_dir / "HC"
                
                print(f"\nHC textdata: {hc_text}, 존재: {hc_text.exists()}")
                print(f"HC voicedata: {hc_voice}, 존재: {hc_voice.exists()}")
                
                if hc_text.exists():
                    print(f"HC textdata 내용: {list(hc_text.iterdir())}")
                    
                    cookie_text = hc_text / "cookie"
                    if cookie_text.exists():
                        print(f"\ncookie textdata 내용 (처음 5개):")
                        txt_files = list(cookie_text.glob("*.txt"))[:5]
                        for txt_file in txt_files:
                            print(f"  {txt_file.name}")

if __name__ == "__main__":
    debug_english_parser()
