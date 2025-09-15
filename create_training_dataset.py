import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def copy_dataset(src_root: Path, dst_root: Path):
    """선택된 데이터셋을 새 폴더로 복사"""
    
    # 복사할 데이터셋 정의
    datasets = {
        'English/Pitt': 'English/Pitt',
        'GreekDem@Care': 'Greek',
        'Spanish/Ivanova': 'Spanish',
        'Madarin': 'Mandarin'  # 원본 폴더명이 Madarin으로 되어있음
    }
    
    # 대상 폴더 생성
    dst_root.mkdir(parents=True, exist_ok=True)
    
    for src_path, dst_name in datasets.items():
        src_full_path = src_root / src_path
        dst_full_path = dst_root / dst_name
        
        if not src_full_path.exists():
            logger.warning(f"❌ 소스 경로가 존재하지 않습니다: {src_full_path}")
            continue
            
        logger.info(f"🔄 복사 중: {src_path} -> {dst_name}")
        
        try:
            if dst_full_path.exists():
                shutil.rmtree(dst_full_path)
            shutil.copytree(src_full_path, dst_full_path)
            logger.info(f"✅ 복사 완료: {dst_name}")
        except Exception as e:
            logger.error(f"⚠️ 복사 중 오류 발생 ({src_path}): {str(e)}")

def main():
    src_root = Path("fulldata_71/dementia_fulldata")
    dst_root = Path("training_dset")
    
    if not src_root.exists():
        logger.error(f"❌ 소스 폴더를 찾을 수 없습니다: {src_root}")
        return
        
    logger.info(f"🎯 데이터셋 복사 시작")
    copy_dataset(src_root, dst_root)
    logger.info(f"✨ 데이터셋 복사 완료")

if __name__ == "__main__":
    main() 