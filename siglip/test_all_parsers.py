#!/usr/bin/env python3
"""
모든 언어 파서 테스트 스크립트
"""
import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_parsers import get_language_parser, parse_all_languages

def test_individual_parsers():
    """개별 언어 파서 테스트"""
    data_dir = "../../training_dset"
    languages = ['English', 'Greek', 'Spanish', 'Mandarin']
    
    print("=" * 60)
    print("🧪 개별 언어 파서 테스트")
    print("=" * 60)
    
    for language in languages:
        print(f"\n🔍 {language} 파서 테스트:")
        print("-" * 40)
        
        try:
            parser = get_language_parser(language, data_dir)
            data = parser.parse_data()
            
            print(f"✅ {language}: {len(data)}개 샘플 파싱 완료")
            
            if data:
                # 라벨 분포
                normal_count = sum(1 for d in data if d['label'] == 0)
                dementia_count = sum(1 for d in data if d['label'] == 1)
                print(f"   📊 정상: {normal_count}개, 치매: {dementia_count}개")
                
                # 첫 번째 샘플 정보
                sample = data[0]
                print(f"   📝 첫 번째 샘플:")
                print(f"      오디오: {sample['audio_path']}")
                print(f"      텍스트: {sample['text'][:50]}...")
                print(f"      라벨: {sample['label']} ({'정상' if sample['label'] == 0 else '치매'})")
            else:
                print(f"⚠️ {language}: 데이터가 없습니다")
                
        except Exception as e:
            print(f"❌ {language} 파서 오류: {e}")

def test_all_parsers():
    """전체 파서 통합 테스트"""
    data_dir = "../../training_dset"
    
    print("\n" + "=" * 60)
    print("🧪 전체 파서 통합 테스트")
    print("=" * 60)
    
    try:
        all_data = parse_all_languages(data_dir)
        
        print(f"\n✅ 전체 파싱 완료: {len(all_data)}개 샘플")
        
        if all_data:
            # 언어별 통계
            language_stats = {}
            label_stats = {'정상': 0, '치매': 0}
            
            for item in all_data:
                lang = item['language']
                language_stats[lang] = language_stats.get(lang, 0) + 1
                
                if item['label'] == 0:
                    label_stats['정상'] += 1
                else:
                    label_stats['치매'] += 1
            
            print("\n📊 언어별 샘플 수:")
            for lang, count in language_stats.items():
                print(f"   {lang}: {count}개")
            
            print(f"\n📊 전체 라벨 분포:")
            print(f"   정상: {label_stats['정상']}개")
            print(f"   치매: {label_stats['치매']}개")
            print(f"   치매 비율: {label_stats['치매']/(label_stats['정상']+label_stats['치매'])*100:.1f}%")
            
        else:
            print("⚠️ 전체 데이터가 비어있습니다")
            
    except Exception as e:
        print(f"❌ 전체 파싱 오류: {e}")

def check_data_structure():
    """데이터 구조 확인"""
    data_dir = Path("../../training_dset")
    languages = ['English', 'Greek', 'Spanish', 'Mandarin']
    
    print("\n" + "=" * 60)
    print("🔍 데이터 구조 확인")
    print("=" * 60)
    
    print(f"\n📁 기본 데이터 디렉토리: {data_dir.absolute()}")
    print(f"   존재여부: {data_dir.exists()}")
    
    if data_dir.exists():
        print(f"   내용: {list(data_dir.iterdir())}")
    
    for language in languages:
        lang_dir = data_dir / language
        print(f"\n📁 {language} 디렉토리: {lang_dir}")
        print(f"   존재여부: {lang_dir.exists()}")
        
        if lang_dir.exists():
            print(f"   내용: {list(lang_dir.iterdir())}")
            
            textdata_dir = lang_dir / "textdata"
            voicedata_dir = lang_dir / "voicedata"
            
            print(f"   📁 textdata: {textdata_dir.exists()}")
            print(f"   📁 voicedata: {voicedata_dir.exists()}")
            
            if textdata_dir.exists():
                print(f"      textdata 내용: {list(textdata_dir.iterdir())}")
            if voicedata_dir.exists():
                print(f"      voicedata 내용: {list(voicedata_dir.iterdir())}")

if __name__ == "__main__":
    print("🚀 모든 언어 파서 테스트 시작")
    
    # 1. 데이터 구조 확인
    check_data_structure()
    
    # 2. 개별 파서 테스트
    test_individual_parsers()
    
    # 3. 통합 파서 테스트
    test_all_parsers()
    
    print("\n🎉 모든 테스트 완료!")
