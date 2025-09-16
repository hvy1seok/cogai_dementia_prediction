import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorpusAnalyzer:
    def __init__(self, root_dir: str = "fulldata_71/dementia_fulldata"):
        self.root_dir = Path(root_dir)
        self.corpus_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
    def count_files_in_dir(self, dir_path: Path, corpus_name: str, group: str = None):
        """특정 디렉토리의 파일 수를 계산"""
        stats = self.corpus_stats[corpus_name]
        
        # 오디오 파일
        if group:
            stats['audio'][f'{group}_mp3'] += len(list(dir_path.rglob('*.mp3')))
            stats['audio'][f'{group}_wav'] += len(list(dir_path.rglob('*.wav')))
            stats['text'][f'{group}_cha'] += len(list(dir_path.rglob('*.cha')))
            stats['text'][f'{group}_txt'] += len(list(dir_path.rglob('*.txt')))
        else:
            stats['audio']['total_mp3'] += len(list(dir_path.rglob('*.mp3')))
            stats['audio']['total_wav'] += len(list(dir_path.rglob('*.wav')))
            stats['text']['total_cha'] += len(list(dir_path.rglob('*.cha')))
            stats['text']['total_txt'] += len(list(dir_path.rglob('*.txt')))

    def analyze_english_corpus(self):
        """영어 코퍼스 분석"""
        english_path = self.root_dir / "English"
        
        # Pitt Corpus
        pitt_path = english_path / "Pitt"
        if pitt_path.exists():
            for group in ['Control', 'Dementia']:
                group_path = pitt_path / "voicedata" / group
                if group_path.exists():
                    self.count_files_in_dir(group_path, "Pitt", group.lower())
        
        # DePaul Corpus
        depaul_path = english_path / "DePaul"
        if depaul_path.exists():
            self.count_files_in_dir(depaul_path, "DePaul")
            
        # Hopkins Corpus
        hopkins_path = english_path / "Hopkins"
        if hopkins_path.exists():
            self.count_files_in_dir(hopkins_path, "Hopkins")
            
        # Delaware Corpus
        delaware_path = english_path / "English-Protocol Data" / "Delaware"
        if delaware_path.exists():
            for group in ['Control', 'MCI']:
                group_path = delaware_path / group
                if group_path.exists():
                    self.count_files_in_dir(group_path, "Delaware", group.lower())

    def analyze_mandarin_corpus(self):
        """중국어 코퍼스 분석"""
        mandarin_path = self.root_dir / "Madarin"
        
        # Chou Corpus
        chou_path = mandarin_path / "Chou"
        if chou_path.exists():
            for group in ['HC', 'MCI']:
                group_path = chou_path / group
                if group_path.exists():
                    self.count_files_in_dir(group_path, "Chou", group.lower())
        
        # Lu Corpus
        lu_path = mandarin_path / "Lu"
        if lu_path.exists():
            self.count_files_in_dir(lu_path, "Lu_Mandarin")

    def analyze_spanish_corpus(self):
        """스페인어 코퍼스 분석"""
        spanish_path = self.root_dir / "Spanish" / "Ivanova"
        if spanish_path.exists():
            for group in ['HC', 'MCI', 'AD']:
                group_path = spanish_path / group
                if group_path.exists():
                    self.count_files_in_dir(group_path, "Ivanova", group.lower())

    def analyze_german_corpus(self):
        """독일어 코퍼스 분석"""
        german_path = self.root_dir / "German" / "Jalvingh"
        if german_path.exists():
            self.count_files_in_dir(german_path, "Jalvingh")

    def analyze_greek_corpus(self):
        """그리스어 코퍼스 분석"""
        greek_path = self.root_dir / "GreekDem@Care"
        if greek_path.exists():
            for subset in ['long', 'short']:
                subset_path = greek_path / subset
                if subset_path.exists():
                    for group in ['AD', 'HC', 'MCI']:
                        group_path = subset_path / group
                        if group_path.exists():
                            self.count_files_in_dir(group_path, f"Dem@Care_{subset}", group.lower())

    def analyze_taiwanese_corpus(self):
        """대만어 코퍼스 분석"""
        taiwanese_path = self.root_dir / "Taiwanese" / "Lu"
        if taiwanese_path.exists():
            self.count_files_in_dir(taiwanese_path, "Lu_Taiwanese")

    def print_corpus_stats(self):
        """코퍼스 통계 출력"""
        # 모든 코퍼스 분석 실행
        self.analyze_english_corpus()
        self.analyze_mandarin_corpus()
        self.analyze_spanish_corpus()
        self.analyze_german_corpus()
        self.analyze_greek_corpus()
        self.analyze_taiwanese_corpus()
        
        # 결과 출력
        for corpus_name, stats in self.corpus_stats.items():
            print(f"\n=== {corpus_name} Corpus ===")
            
            # 오디오 파일 통계
            print("\n[Audio Files]")
            audio_df = pd.DataFrame(stats['audio'], index=[0]).T
            print(audio_df)
            
            # 텍스트 파일 통계
            print("\n[Text Files]")
            text_df = pd.DataFrame(stats['text'], index=[0]).T
            print(text_df)
            
            # 총계 계산
            total_audio = sum(stats['audio'].values())
            total_text = sum(stats['text'].values())
            print(f"\nTotal Audio Files: {total_audio}")
            print(f"Total Text Files: {total_text}")
            print(f"Audio/Text Ratio: {total_audio/total_text:.2f}" if total_text > 0 else "No text files")

if __name__ == "__main__":
    analyzer = CorpusAnalyzer()
    analyzer.print_corpus_stats() 