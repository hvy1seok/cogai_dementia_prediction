import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_group(file_path: str) -> str:
    """파일 경로를 기반으로 그룹(AD/HC/MCI) 결정"""
    path_lower = str(file_path).lower()
    
    # Lu 폴더의 파일은 AD로 분류
    if "/lu/" in path_lower or "\\lu\\" in path_lower:
        return "AD"
    
    # Ye 폴더의 파일은 MCI로 분류
    if "/ye/" in path_lower or "\\ye\\" in path_lower:
        return "MCI"
    
    # 기존 키워드 기반 분류
    if any(kw in path_lower for kw in ["ad/", "dementia", "daddy"]):
        return "AD"
    elif any(kw in path_lower for kw in ["hc/", "control", "market", "park"]):
        return "HC"
    elif "mci" in path_lower:
        return "MCI"
    
    return None

def reorganize_dataset(language_dir: Path):
    """각 언어 데이터셋을 AD/HC/MCI 구조로 재구성"""
    
    # 새로운 디렉토리 구조 생성
    voice_dir = language_dir / "voicedata"
    text_dir = language_dir / "textdata"
    
    voice_dir.mkdir(exist_ok=True, parents=True)
    text_dir.mkdir(exist_ok=True, parents=True)
    
    # AD/HC/MCI 하위 디렉토리 생성
    for subdir in ["AD", "HC", "MCI"]:
        (voice_dir / subdir).mkdir(exist_ok=True)
        (text_dir / subdir).mkdir(exist_ok=True)
    
    # 기존 파일들을 새 구조로 이동
    for file_path in language_dir.rglob("*"):
        if not file_path.is_file():
            continue
            
        # voicedata나 textdata 폴더 내부의 파일은 건너뛰기
        if "voicedata" in str(file_path) or "textdata" in str(file_path):
            continue
        
        # 파일 확장자 확인
        suffix = file_path.suffix.lower()
        
        # npy 파일만 voicedata로, txt 파일만 textdata로 이동
        if suffix not in [".npy", ".txt"]:
            continue
            
        # 그룹 결정
        group = get_group(str(file_path))
        if not group:
            logger.warning(f"그룹을 결정할 수 없는 파일: {file_path}")
            continue
            
        # 목적지 디렉토리 결정
        if suffix == ".npy":
            dest_dir = voice_dir / group
        else:  # .txt
            dest_dir = text_dir / group
            
        # 파일 이동
        dest_path = dest_dir / file_path.name
        try:
            shutil.copy2(file_path, dest_path)
            logger.info(f"파일 복사 완료: {file_path} -> {dest_path}")
        except Exception as e:
            logger.error(f"파일 복사 실패: {file_path} -> {dest_path}, 에러: {e}")

if __name__ == "__main__":
    # training_dset 폴더에서 직접 작업
    base_dir = Path("training_dset")
    
    # Mandarin 데이터셋 처리
    lang_dir = base_dir / "Mandarin"
    if lang_dir.exists():
        logger.info(f"Mandarin 데이터셋 재구성 시작...")
        reorganize_dataset(lang_dir)
        logger.info(f"Mandarin 데이터셋 재구성 완료") 