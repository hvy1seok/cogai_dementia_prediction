"""
언어별 데이터 파서 - training_dset 폴더 구조에 맞춤
각 언어의 textdata/voicedata 구조에서 HC, AD, MCI 데이터를 파싱
"""
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

class BaseLanguageParser(ABC):
    """언어별 파서의 기본 클래스"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.language = self.get_language_name()
    
    @abstractmethod
    def get_language_name(self) -> str:
        """언어 이름 반환"""
        pass
    
    @abstractmethod
    def parse_data(self) -> List[Dict]:
        """데이터 파싱"""
        pass

class TrainingDatasetParser(BaseLanguageParser):
    """training_dset 폴더 구조용 공통 파서"""
    
    def __init__(self, lang_data_dir: Path):
        self.data_dir = lang_data_dir
        self.language = self.get_language_name()
    
    def parse_data(self) -> List[Dict]:
        """textdata와 voicedata 폴더에서 매칭되는 파일들을 파싱"""
        data = []
        
        textdata_dir = self.data_dir / "textdata"
        voicedata_dir = self.data_dir / "voicedata"
        
        if not textdata_dir.exists() or not voicedata_dir.exists():
            print(f"⚠️ {self.language}: textdata 또는 voicedata 폴더가 없습니다.")
            return data
        
        # HC, AD 카테고리 처리 (MCI 제외)
        categories = {
            'HC': 0,   # Healthy Control - 정상
            'AD': 1,   # Alzheimer's Disease - 치매
        }
        
        for category, label in categories.items():
            text_cat_dir = textdata_dir / category
            voice_cat_dir = voicedata_dir / category
            
            if not text_cat_dir.exists() or not voice_cat_dir.exists():
                print(f"⚠️ {self.language}: {category} 폴더가 없습니다.")
                continue
            
            # 텍스트 파일들을 기준으로 매칭
            for txt_file in text_cat_dir.glob("*.txt"):
                # tasks 파일은 제외 (메타데이터 파일)
                if "tasks" in txt_file.stem:
                    continue
                
                # 대응하는 .npy 파일 찾기
                npy_file = voice_cat_dir / f"{txt_file.stem}.npy"
                
                if npy_file.exists():
                    try:
                        # 텍스트 읽기
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        if text:  # 텍스트가 비어있지 않은 경우만
                            data.append({
                                'audio_path': str(npy_file),
                                'text': text,
                                'label': label,
                                'language': self.language,
                                'source': f'{self.language}_{category}',
                                'file_id': txt_file.stem
                            })
                    
                    except Exception as e:
                        print(f"파싱 오류 {txt_file}: {e}")
                else:
                    print(f"⚠️ 매칭되는 음성 파일 없음: {npy_file}")
        
        return data

class EnglishParser(BaseLanguageParser):
    """영어 데이터 파서 - Pitt 구조 처리"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.pitt_dir = Path(data_dir) / "English" / "Pitt"
    
    def get_language_name(self) -> str:
        return "English"
    
    def parse_data(self) -> List[Dict]:
        """Pitt 폴더 구조에서 데이터 파싱"""
        data = []
        
        # 디버깅 정보 출력
        print(f"🔍 영어 파서 디버깅:")
        print(f"  self.data_dir: {self.data_dir}")
        print(f"  self.pitt_dir: {self.pitt_dir}")
        print(f"  pitt_dir 절대경로: {self.pitt_dir.absolute()}")
        print(f"  pitt_dir 존재여부: {self.pitt_dir.exists()}")
        
        textdata_dir = self.pitt_dir / "textdata"
        voicedata_dir = self.pitt_dir / "voicedata"
        
        print(f"  textdata_dir: {textdata_dir}")
        print(f"  textdata_dir 절대경로: {textdata_dir.absolute()}")
        print(f"  textdata_dir 존재여부: {textdata_dir.exists()}")
        print(f"  voicedata_dir: {voicedata_dir}")
        print(f"  voicedata_dir 절대경로: {voicedata_dir.absolute()}")
        print(f"  voicedata_dir 존재여부: {voicedata_dir.exists()}")
        
        if not textdata_dir.exists() or not voicedata_dir.exists():
            print(f"⚠️ {self.language}: Pitt/textdata 또는 Pitt/voicedata 폴더가 없습니다.")
            return data
        
        # HC, AD 카테고리 처리 (MCI 제외)
        categories = {
            'HC': 0,   # Healthy Control - 정상
            'AD': 1,   # Alzheimer's Disease - 치매
        }
        
        for category, label in categories.items():
            text_cat_dir = textdata_dir / category
            voice_cat_dir = voicedata_dir / category
            
            if not text_cat_dir.exists() or not voice_cat_dir.exists():
                print(f"⚠️ {self.language}/{category}: textdata 또는 voicedata 폴더가 없습니다.")
                continue
            
            # 하위 폴더들 (cookie, fluency, recall, sentence) 처리
            for subfolder in text_cat_dir.iterdir():
                if not subfolder.is_dir():
                    continue
                    
                voice_subfolder = voice_cat_dir / subfolder.name
                if not voice_subfolder.exists():
                    continue
                
                # 각 하위 폴더에서 .txt와 .npy 파일 매칭
                for txt_file in subfolder.glob("*.txt"):
                    stem = txt_file.stem
                    npy_file = voice_subfolder / f"{stem}.npy"
                    
                    if npy_file.exists():
                        try:
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            
                            if text:
                                data.append({
                                    'audio_path': str(npy_file),
                                    'text': text,
                                    'label': label,
                                    'language': self.language,
                                    'source': f'{self.language}_{category}_{subfolder.name}'
                                })
                        except Exception as e:
                            print(f"파싱 오류 {txt_file}: {e}")
                    else:
                        print(f"⚠️ {self.language}/{category}/{subfolder.name}: 매칭되는 .npy 파일 없음: {npy_file}")
        
        return data

class GreekParser(TrainingDatasetParser):
    """그리스어 데이터 파서"""
    
    def __init__(self, data_dir: str):
        super().__init__(Path(data_dir) / "Greek")
    
    def get_language_name(self) -> str:
        return "Greek"

class SpanishParser(TrainingDatasetParser):
    """스페인어 데이터 파서"""
    
    def __init__(self, data_dir: str):
        super().__init__(Path(data_dir) / "Spanish")
    
    def get_language_name(self) -> str:
        return "Spanish"

class MandarinParser(TrainingDatasetParser):
    """중국어(만다린) 데이터 파서"""
    
    def __init__(self, data_dir: str):
        super().__init__(Path(data_dir) / "Mandarin")
    
    def get_language_name(self) -> str:
        return "Mandarin"

def get_language_parser(language: str, data_dir: str) -> BaseLanguageParser:
    """언어별 파서 팩토리 함수"""
    parsers = {
        'English': EnglishParser,
        'Greek': GreekParser,
        'Spanish': SpanishParser,
        'Mandarin': MandarinParser,
    }
    
    if language not in parsers:
        raise ValueError(f"지원하지 않는 언어: {language}. 지원 언어: {list(parsers.keys())}")
    
    print(f"🔧 파서 생성: {language}, 데이터 경로: {data_dir}")
    
    # English는 특별 처리 (data_dir를 직접 전달)
    if language == 'English':
        print(f"🔧 영어 파서 생성: {parsers[language].__name__}")
        parser = parsers[language](data_dir)
        print(f"🔧 생성된 파서 타입: {type(parser)}")
        return parser
    else:
        # 다른 언어들은 언어별 하위 디렉토리 전달
        lang_data_dir = os.path.join(data_dir, language)
        print(f"🔧 {language} 파서 생성: {parsers[language].__name__}, 경로: {lang_data_dir}")
        parser = parsers[language](lang_data_dir)
        print(f"🔧 생성된 파서 타입: {type(parser)}")
        return parser

def parse_all_languages(data_dir: str, languages: List[str] = None) -> List[Dict]:
    """모든 언어 데이터 파싱"""
    if languages is None:
        languages = ['English', 'Greek', 'Spanish', 'Mandarin']
    
    all_data = []
    
    for language in languages:
        try:
            parser = get_language_parser(language, data_dir)
            data = parser.parse_data()
            print(f"{language}: {len(data)}개 샘플 파싱 완료")
            
            # 언어별 라벨 분포 출력
            if data:
                normal_count = sum(1 for d in data if d['label'] == 0)
                dementia_count = sum(1 for d in data if d['label'] == 1)
                print(f"  - 정상: {normal_count}개, 치매: {dementia_count}개")
            
            all_data.extend(data)
            
        except Exception as e:
            print(f"{language} 파싱 오류: {e}")
    
    print(f"\n총 {len(all_data)}개 샘플 파싱 완료")
    
    # 전체 라벨 분포 출력
    if all_data:
        total_normal = sum(1 for d in all_data if d['label'] == 0)
        total_dementia = sum(1 for d in all_data if d['label'] == 1)
        print(f"전체 - 정상: {total_normal}개, 치매: {total_dementia}개")
        print(f"치매 비율: {total_dementia/(total_normal+total_dementia)*100:.1f}%")
    
    return all_data

# 테스트 함수
def test_parser(language: str, data_dir: str = "training_dset"):
    """특정 언어 파서 테스트"""
    print(f"=== {language} 파서 테스트 ===")
    try:
        parser = get_language_parser(language, data_dir)
        data = parser.parse_data()
        
        if data:
            print(f"샘플 수: {len(data)}")
            print("첫 번째 샘플:")
            sample = data[0]
            print(f"  오디오: {sample['audio_path']}")
            print(f"  텍스트: {sample['text'][:100]}...")
            print(f"  라벨: {sample['label']} ({'치매' if sample['label'] == 1 else '정상'})")
            print(f"  언어: {sample['language']}")
            print(f"  소스: {sample['source']}")
        else:
            print("데이터가 없습니다.")
            
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    # 모든 언어 테스트
    for lang in ['English', 'Greek', 'Spanish', 'Mandarin']:
        test_parser(lang)
        print()
