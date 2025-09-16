import os
from pathlib import Path
from collections import defaultdict

def get_base_name(file_path):
    # Remove extension and path
    return Path(file_path).stem

def check_matching_files():
    root_dir = "./fulldata_71/dementia_fulldata"
    results = defaultdict(lambda: {"audio": [], "text": [], "matched": 0, "unmatched_audio": 0, "unmatched_text": 0})
    
    # Walk through all directories
    for root, dirs, files in os.walk(root_dir):
        try:
            # Get language from path - adjust index based on actual path structure
            path_parts = root.split(os.sep)
            if len(path_parts) < 4:  # Need at least [".", "fulldata_71", "dementia_fulldata", "language"]
                continue
            lang = path_parts[3]
            
            # Skip 0wav and 0extra directories
            if "0wav" in root or "0extra" in root:
                continue
                
            for file in files:
                if file.endswith(('.mp3', '.wav')):
                    base_name = get_base_name(file)
                    results[lang]["audio"].append(base_name)
                    
                elif file.endswith('.cha'):
                    base_name = get_base_name(file)
                    results[lang]["text"].append(base_name)
        except Exception as e:
            print(f"Error processing path {root}: {str(e)}")
    
    # Calculate matching statistics
    print("\n=== 언어별 파일 매칭 통계 ===\n")
    for lang, data in results.items():
        audio_set = set(data["audio"])
        text_set = set(data["text"])
        
        matched = len(audio_set.intersection(text_set))
        unmatched_audio = len(audio_set - text_set)
        unmatched_text = len(text_set - audio_set)
        
        print(f"\n{lang} 언어:")
        print(f"총 음성 파일: {len(audio_set)}")
        print(f"총 텍스트 파일: {len(text_set)}")
        print(f"매칭되는 파일: {matched}")
        print(f"매칭되지 않는 음성 파일: {unmatched_audio}")
        print(f"매칭되지 않는 텍스트 파일: {unmatched_text}")
        
        if unmatched_audio > 0:
            print("\n매칭되지 않는 음성 파일 예시 (최대 5개):")
            unmatched = list(audio_set - text_set)[:5]
            for f in unmatched:
                print(f"- {f}")

if __name__ == "__main__":
    check_matching_files() 