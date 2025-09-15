import os
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preProcessFile(file_path):
    """CHAT 파일에서 발화와 특징 추출"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        traditional = False
        utt = ''
        for line in lines:
            if line.startswith('*PAR'):
                if '[+ exc]' not in line:
                    traditional = True
                    utt += line[6:-1]
            elif traditional and line.startswith('%mor'): 
                end = -2
                while utt[end].isnumeric():
                    end -= 1
                end -= 1
                while utt[end].isnumeric():
                    end -= 1
                traditional = False
                utt = utt[:end]
            elif traditional:
                if '[+ exc]' not in line:
                    utt += line[:-1]

        # 특징 추출
        repetitions = utt.count('[/]')
        retraicings = utt.count('[//]')
        spause = utt.count('(.)')
        mpause = utt.count('(..)')
        lpause = utt.count('(...)')
        tpauses = spause + mpause + lpause
        unintelligible = utt.count('xxx')
        gramerros = utt.count('[+ gram]')
        doubts = utt.count('&uh') + utt.count('&um')

        # 텍스트 정제
        utt = utt.replace('\n', '')
        utt = utt.replace('\t', ' ')
        utt = utt.replace('(', '')
        utt = utt.replace(')', '')
        utt = utt.replace('+<', '')
        utt = utt.replace('&', '')
        utt = re.sub(r'[\<\[].*?[\>\]]', '', utt)

        return [utt, repetitions, retraicings, spause, mpause, lpause, tpauses, unintelligible, gramerros, doubts]
    
    except Exception as e:
        logger.error(f"⚠️ 파일 처리 중 오류 발생 {file_path}: {str(e)}")
        return None

def process_directory(root_dir: Path):
    """디렉토리 내의 모든 .cha 파일 처리"""
    # 처리 통계
    total_files = 0
    processed_files = 0
    failed_files = 0
    skipped_files = 0

    # 0wav와 0extra 폴더는 건너뛰기
    skip_dirs = {'0wav', '0extra'}

    # 모든 .cha 파일 찾기
    for cha_path in root_dir.rglob('*.cha'):
        total_files += 1
        
        # 건너뛸 폴더에 있는 파일인지 확인
        if any(skip_dir in str(cha_path.parent) for skip_dir in skip_dirs):
            skipped_files += 1
            continue

        # 출력 파일 경로 (.txt)
        output_path = cha_path.with_suffix('.txt')
        
        # 이미 처리된 파일이면 건너뛰기
        if output_path.exists():
            skipped_files += 1
            logger.info(f"⏩ 이미 존재하는 파일: {output_path}")
            continue

        logger.info(f"🔄 처리 중: {cha_path}")
        
        try:
            # 파일 처리
            properties = preProcessFile(cha_path)
            if properties is None:
                failed_files += 1
                continue

            # 결과 저장
            with open(output_path, 'w', encoding='utf-8') as out:
                for i, val in enumerate(properties):
                    if i not in [8, 9]:  # 문법 오류, 주저발화 제외
                        out.write(str(val) + '\n')

            processed_files += 1
            logger.info(f"✅ 처리 완료: {cha_path}")

        except Exception as e:
            failed_files += 1
            logger.error(f"⚠️ 처리 실패: {cha_path} | 에러: {e}")

    # 처리 결과 출력
    logger.info("\n=== 처리 완료 ===")
    logger.info(f"총 파일 수: {total_files}")
    logger.info(f"성공: {processed_files}")
    logger.info(f"실패: {failed_files}")
    logger.info(f"건너뛴 파일: {skipped_files}")

def preprocessData():
    """전체 전처리 실행"""
    root_dir = Path("../fulldata_71/dementia_fulldata")
    if not root_dir.exists():
        logger.error(f"❌ 경로를 찾을 수 없습니다: {root_dir}")
        return

    logger.info(f"🎯 처리 시작: {root_dir}")
    process_directory(root_dir)

if __name__ == "__main__":
    preprocessData()
