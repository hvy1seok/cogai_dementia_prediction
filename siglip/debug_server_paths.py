#!/usr/bin/env python3
"""
서버 환경에서 데이터 경로 디버깅
"""
import os
from pathlib import Path

def debug_server_paths():
    print("=== 서버 환경 데이터 경로 디버깅 ===")
    
    # 현재 작업 디렉토리
    current_dir = Path.cwd()
    print(f"현재 작업 디렉토리: {current_dir}")
    
    # 가능한 데이터 경로들 확인 (서버 우선)
    possible_paths = [
        "../../training_dset",  # 서버 경로 (우선)
        "../training_dset",
        "../../../training_dset",
        "/workspace/training_dset",
        "/workspace/ucla/cogai_dementia_prediction/training_dset",
        "./training_dset",
        "training_dset"
    ]
    
    print("\n=== 가능한 데이터 경로 확인 ===")
    for path_str in possible_paths:
        path = Path(path_str)
        exists = path.exists()
        abs_path = path.absolute()
        print(f"{'✅' if exists else '❌'} {path_str} -> {abs_path}")
        
        if exists:
            # English 폴더 확인
            english_path = path / "English"
            if english_path.exists():
                print(f"  ✅ English 폴더 존재: {english_path}")
                pitt_path = english_path / "Pitt"
                if pitt_path.exists():
                    print(f"  ✅ Pitt 폴더 존재: {pitt_path}")
                    textdata_path = pitt_path / "textdata"
                    voicedata_path = pitt_path / "voicedata"
                    print(f"  {'✅' if textdata_path.exists() else '❌'} textdata: {textdata_path}")
                    print(f"  {'✅' if voicedata_path.exists() else '❌'} voicedata: {voicedata_path}")
                else:
                    print(f"  ❌ Pitt 폴더 없음")
            else:
                print(f"  ❌ English 폴더 없음")
    
    # 상위 디렉토리 탐색
    print("\n=== 상위 디렉토리 구조 ===")
    parent = current_dir.parent
    for i in range(3):  # 3단계 상위까지 확인
        print(f"상위 {i+1}단계: {parent}")
        if parent.exists():
            subdirs = [d.name for d in parent.iterdir() if d.is_dir()]
            print(f"  하위 폴더: {subdirs[:10]}...")  # 처음 10개만
            
            # training_dset 관련 폴더 찾기
            training_dirs = [d for d in subdirs if 'training' in d.lower() or 'dset' in d.lower()]
            if training_dirs:
                print(f"  📁 훈련 관련 폴더: {training_dirs}")
        parent = parent.parent
    
    print("\n=== 권장 해결책 ===")
    print("1. 올바른 데이터 경로를 찾아서 config.py 또는 스크립트 수정")
    print("2. 심볼릭 링크 생성: ln -s /actual/path/to/training_dset ../training_dset")
    print("3. 환경변수 설정: export DATA_DIR=/path/to/training_dset")

if __name__ == "__main__":
    debug_server_paths()
