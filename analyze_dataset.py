import os
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    def __init__(self, root_dir: str = "fulldata_71/dementia_fulldata"):
        self.root_dir = Path(root_dir)
        self.languages = self._get_languages()
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def _get_languages(self) -> List[str]:
        """루트 디렉토리의 언어 폴더 목록 반환"""
        return [d.name for d in self.root_dir.iterdir() if d.is_dir()]
    
    def analyze_language_structure(self) -> Dict[str, Dict[str, int]]:
        """각 언어별 데이터 구조 분석"""
        for lang in self.languages:
            lang_path = self.root_dir / lang
            
            # 오디오 파일 수 계산
            self.stats[lang]['mp3_files'] = len(list(lang_path.rglob('*.mp3')))
            self.stats[lang]['wav_files'] = len(list(lang_path.rglob('*.wav')))
            self.stats[lang]['npy_files'] = len(list(lang_path.rglob('*.npy')))
            
            # 텍스트 파일 수 계산
            self.stats[lang]['cha_files'] = len(list(lang_path.rglob('*.cha')))
            self.stats[lang]['txt_files'] = len(list(lang_path.rglob('*.txt')))
            
            # 그룹별 데이터 수 계산
            self.stats[lang]['control_files'] = len(list(lang_path.rglob('**/Control/**/*.mp3'))) + \
                                              len(list(lang_path.rglob('**/HC/**/*.mp3')))
            self.stats[lang]['dementia_files'] = len(list(lang_path.rglob('**/Dementia/**/*.mp3'))) + \
                                               len(list(lang_path.rglob('**/AD/**/*.mp3'))) + \
                                               len(list(lang_path.rglob('**/MCI/**/*.mp3')))
            
            # 하위 디렉토리 구조 분석
            subdirs = [d.name for d in lang_path.iterdir() if d.is_dir()]
            self.stats[lang]['subdirectories'] = ', '.join(subdirs)
            
        return dict(self.stats)
    
    def analyze_matching_files(self) -> Dict[str, Dict[str, int]]:
        """각 언어별 매칭되는 파일 분석"""
        matching_stats = defaultdict(lambda: defaultdict(int))
        
        for lang in self.languages:
            lang_path = self.root_dir / lang
            
            # 오디오-텍스트 매칭 분석
            audio_files = set()
            for ext in ['.mp3', '.wav']:
                audio_files.update([f.stem for f in lang_path.rglob(f'*{ext}')])
            
            text_files = set([f.stem for f in lang_path.rglob('*.cha')])
            
            matching_stats[lang]['total_audio'] = len(audio_files)
            matching_stats[lang]['total_text'] = len(text_files)
            matching_stats[lang]['matched_files'] = len(audio_files.intersection(text_files))
            matching_stats[lang]['unmatched_audio'] = len(audio_files - text_files)
            matching_stats[lang]['unmatched_text'] = len(text_files - audio_files)
            
        return dict(matching_stats)
    
    def print_analysis(self):
        """분석 결과 출력"""
        # 기본 구조 분석
        logger.info("\n=== 데이터셋 기본 구조 분석 ===")
        structure_stats = self.analyze_language_structure()
        
        df_structure = pd.DataFrame(structure_stats).fillna(0)
        print("\n[파일 타입별 통계]")
        print(df_structure.loc[['mp3_files', 'wav_files', 'npy_files', 'cha_files', 'txt_files']])
        
        print("\n[그룹별 통계]")
        print(df_structure.loc[['control_files', 'dementia_files']])
        
        # 매칭 분석
        logger.info("\n=== 파일 매칭 분석 ===")
        matching_stats = self.analyze_matching_files()
        df_matching = pd.DataFrame(matching_stats).fillna(0)
        print(df_matching)
        
        # 하위 디렉토리 구조
        print("\n=== 언어별 하위 디렉토리 구조 ===")
        for lang in self.languages:
            print(f"\n{lang}:")
            print(structure_stats[lang]['subdirectories'])

def count_files_by_extension(root_dir: Path) -> Dict[str, Dict[str, int]]:
    """각 언어 폴더별로 파일 확장자 개수를 세는 함수"""
    results = {}
    
    for lang_dir in root_dir.iterdir():
        if not lang_dir.is_dir():
            continue
            
        extension_counts = Counter()
        for file_path in lang_dir.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext:  # 확장자가 있는 경우만
                    extension_counts[ext] += 1
        
        results[lang_dir.name] = dict(extension_counts)
    
    return results

def print_analysis(analysis: Dict[str, Dict[str, int]]):
    """분석 결과를 보기 좋게 출력"""
    logger.info("📊 파일 확장자별 개수 분석 결과:")
    print("\n" + "="*60)
    
    # 전체 합계를 위한 카운터
    total_counts = Counter()
    
    for lang, counts in analysis.items():
        print(f"\n📁 {lang}:")
        print("-" * 30)
        
        if not counts:
            print("파일이 없습니다.")
            continue
            
        # 확장자별로 정렬하여 출력
        for ext, count in sorted(counts.items()):
            print(f"{ext:8} : {count:5,d} 파일")
            total_counts[ext] += count
            
    # 전체 합계 출력
    print("\n" + "="*30)
    print("📈 전체 합계:")
    print("-" * 30)
    for ext, count in sorted(total_counts.items()):
        print(f"{ext:8} : {count:5,d} 파일")
    print("=" * 60)

def main():
    root_dir = Path("training_dset")
    
    if not root_dir.exists():
        logger.error(f"❌ training_dset 폴더를 찾을 수 없습니다.")
        return
        
    analysis = count_files_by_extension(root_dir)
    print_analysis(analysis)

if __name__ == "__main__":
    main() 