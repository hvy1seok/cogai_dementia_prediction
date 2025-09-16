#!/usr/bin/env python3
"""
SigLIP-SAM 데이터 파싱 테스트 스크립트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_parsers import parse_all_languages, get_language_parser

def test_data_parsing():
    """데이터 파싱 테스트"""
    print("=== SigLIP-SAM 데이터 파싱 테스트 ===")
    
    data_dir = "../../training_dset"
    languages = ['English', 'Greek', 'Spanish', 'Mandarin']
    
    print(f"데이터 디렉토리: {data_dir}")
    print(f"테스트할 언어: {languages}")
    print()
    
    # 개별 언어 테스트
    for lang in languages:
        print(f"🔍 {lang} 파서 테스트:")
        try:
            parser = get_language_parser(lang, data_dir)
            data = parser.parse_data()
            
            if data:
                print(f"✅ {lang}: {len(data)}개 샘플 파싱 성공")
                # 첫 번째 샘플 정보
                sample = data[0]
                print(f"   📝 첫 번째 샘플:")
                print(f"      오디오: {sample['audio_path']}")
                print(f"      텍스트: {sample['text'][:50]}...")
                print(f"      라벨: {sample['label']} ({'정상' if sample['label'] == 0 else '치매'})")
            else:
                print(f"⚠️ {lang}: 데이터가 없습니다")
                
        except Exception as e:
            print(f"❌ {lang} 파서 오류: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    # 전체 언어 통합 테스트
    print("\n🔍 전체 언어 통합 테스트:")
    try:
        all_data = parse_all_languages(data_dir, languages)
        
        if all_data:
            print(f"✅ 전체: {len(all_data)}개 샘플 파싱 성공")
            
            # 언어별 통계
            lang_counts = {}
            for item in all_data:
                lang = item['language']
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            print("📊 언어별 분포:")
            for lang, count in lang_counts.items():
                print(f"   {lang}: {count}개")
            
            # 라벨 분포
            normal_count = sum(1 for item in all_data if item['label'] == 0)
            dementia_count = sum(1 for item in all_data if item['label'] == 1)
            
            print(f"\n📊 라벨 분포:")
            print(f"   정상: {normal_count}개")
            print(f"   치매: {dementia_count}개")
            print(f"   치매 비율: {dementia_count/(normal_count+dementia_count)*100:.1f}%")
            
        else:
            print("⚠️ 전체 데이터가 비어있습니다")
            
    except Exception as e:
        print(f"❌ 전체 테스트 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_parsing()
