#!/usr/bin/env python3
"""
간단한 데이터 구조 확인 스크립트
"""

from pathlib import Path
from collections import Counter

def explore_data_structure():
    """데이터 구조 탐색"""
    
    data_dir = Path("../training_dset")
    
    for language in ['English', 'Greek', 'Spanish', 'Mandarin']:
        lang_dir = data_dir / language
        
        print(f"\n🌍 {language} 디렉토리 구조:")
        
        if not lang_dir.exists():
            print(f"  ❌ 디렉토리 없음: {lang_dir}")
            continue
        
        # txt 파일 찾기
        txt_files = list(lang_dir.glob('**/*.txt'))
        npy_files = list(lang_dir.glob('**/*.npy'))
        
        print(f"  📄 텍스트 파일: {len(txt_files)}개")
        print(f"  🎵 오디오 파일: {len(npy_files)}개")
        
        if txt_files:
            print(f"  📁 텍스트 파일 위치:")
            txt_dirs = set([f.parent for f in txt_files])
            for txt_dir in sorted(txt_dirs):
                count = len(list(txt_dir.glob('*.txt')))
                print(f"    {txt_dir.relative_to(lang_dir)}: {count}개")
        
        if npy_files:
            print(f"  📁 오디오 파일 위치:")
            npy_dirs = set([f.parent for f in npy_files])
            for npy_dir in sorted(npy_dirs):
                count = len(list(npy_dir.glob('*.npy')))
                print(f"    {npy_dir.relative_to(lang_dir)}: {count}개")
        
        # 매칭 가능한 파일 쌍 찾기
        txt_stems = set([f.stem for f in txt_files])
        npy_stems = set([f.stem for f in npy_files])
        matching_stems = txt_stems & npy_stems
        
        print(f"  🔗 매칭 가능한 파일 쌍: {len(matching_stems)}개")
        
        # 첫 번째 텍스트 파일 내용 확인
        if txt_files:
            try:
                with open(txt_files[0], 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                print(f"  📖 첫 번째 텍스트 샘플: '{content[:100]}...'")
            except Exception as e:
                print(f"  ⚠️ 텍스트 읽기 실패: {e}")

if __name__ == "__main__":
    explore_data_structure()
