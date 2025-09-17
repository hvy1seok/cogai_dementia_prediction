import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from transformers import BertTokenizer
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import random

class MultilingualDementiaDataset(Dataset):
    def __init__(self, data):
        """
        다국어 치매 데이터셋
        data: list of dicts with keys: 'text', 'audio_path', 'label', 'language', 'patient_id'
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 오디오 스펙트로그램 로드
        try:
            audio_specs = np.load(item['audio_path'])
            # 크기 조정: (3, 128, 250)으로 맞추기
            if audio_specs.shape != (3, 128, 250):
                # 패딩 또는 크롭으로 크기 조정
                audio_specs = self._resize_audio(audio_specs)
        except:
            # 오디오 로드 실패 시 빈 스펙트로그램
            audio_specs = np.zeros((3, 128, 250), dtype=np.float32)
        
        return {
            'input_ids': item['input_ids'].squeeze(0),  # (max_seq_len,)
            'attention_mask': item['attention_mask'].squeeze(0),  # (max_seq_len,)
            'audio': torch.tensor(audio_specs, dtype=torch.float32),  # (3, 128, 250)
            'label': item['label'],
            'language': item['language'],
            'patient_id': item['patient_id']
        }
    
    def _resize_audio(self, audio_specs):
        """오디오 스펙트로그램 크기 조정"""
        target_shape = (3, 128, 250)
        
        if len(audio_specs.shape) == 2:
            # (H, W) -> (3, H, W) 변환
            audio_specs = np.stack([audio_specs] * 3, axis=0)
        
        current_shape = audio_specs.shape
        resized = np.zeros(target_shape, dtype=np.float32)
        
        # 각 차원별로 크기 조정
        h_min = min(current_shape[1], target_shape[1])
        w_min = min(current_shape[2], target_shape[2])
        
        resized[:, :h_min, :w_min] = audio_specs[:, :h_min, :w_min]
        
        return resized

def read_multilingual_data(data_dir, languages=None):
    """
    다국어 데이터 로드
    data_dir: training_dset 경로
    languages: 로드할 언어 리스트 (None이면 모든 언어)
    """
    if languages is None:
        languages = ['English', 'Greek', 'Spanish', 'Mandarin']
    
    all_data = []
    data_dir = Path(data_dir)
    
    print(f"📂 다국어 데이터 로드 중: {languages}")
    
    for language in languages:
        lang_dir = data_dir / language
        if not lang_dir.exists():
            print(f"⚠️ 언어 디렉토리 없음: {lang_dir}")
            continue
            
        print(f"  📁 {language} 데이터 로드 중...")
        lang_data = load_language_data(lang_dir, language)
        all_data.extend(lang_data)
        print(f"    ✅ {len(lang_data)}개 샘플 로드됨")
    
    print(f"📊 전체 로드된 데이터: {len(all_data)}개 샘플")
    
    # 언어별 통계
    lang_stats = Counter([item['language'] for item in all_data])
    label_stats = Counter([item['label'] for item in all_data])
    
    print(f"📈 언어별 분포:")
    for lang, count in lang_stats.items():
        print(f"  {lang}: {count}개")
    
    print(f"📈 라벨별 분포:")
    label_names = {0: '정상', 1: '치매'}
    for label, count in label_stats.items():
        print(f"  {label_names[label]}: {count}개")
    
    return all_data

def load_language_data(lang_dir, language):
    """특정 언어의 데이터 로드"""
    data = []
    
    # 언어별 데이터 구조가 다를 수 있으므로 유연하게 처리
    if language == 'English':
        data = load_english_data(lang_dir, language)
    elif language == 'Greek':
        data = load_greek_data(lang_dir, language)
    elif language == 'Spanish':
        data = load_spanish_data(lang_dir, language)
    elif language == 'Mandarin':
        data = load_mandarin_data(lang_dir, language)
    
    return data

def load_english_data(lang_dir, language):
    """영어 데이터 로드"""
    data = []
    
    # textdata와 voicedata 디렉토리에서 직접 로드 (다른 언어와 동일한 구조)
    text_dir = lang_dir / 'textdata'
    voice_dir = lang_dir / 'voicedata'
    
    if text_dir.exists() and voice_dir.exists():
        # AD, HC 카테고리
        categories = [
            ('HC', 0),    # Healthy Control
            ('AD', 1)     # Alzheimer's Disease
        ]
        
        for cat_name, label in categories:
            cat_text_dir = text_dir / cat_name
            cat_voice_dir = voice_dir / cat_name
            
            if cat_text_dir.exists() and cat_voice_dir.exists():
                # 텍스트 파일들 로드
                for text_file in cat_text_dir.glob('*.txt'):
                    audio_file = cat_voice_dir / f"{text_file.stem}.npy"
                    
                    if audio_file.exists():
                        try:
                            with open(text_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            
                            if text and len(text) >= 10:  # 최소 길이 체크
                                patient_id = f"{language}_{cat_name}_{text_file.stem}"
                                data.append({
                                    'text': text,
                                    'audio_path': str(audio_file),
                                    'label': label,
                                    'language': language,
                                    'patient_id': patient_id
                                })
                        except Exception as e:
                            print(f"⚠️ 영어 파일 로드 실패: {text_file} - {e}")
    
    # Pitt 디렉토리도 있다면 추가로 로드
    pitt_dir = lang_dir / 'Pitt'
    if pitt_dir.exists():
        pitt_data = load_pitt_data(pitt_dir, language)
        data.extend(pitt_data)
    
    return data

def load_pitt_data(pitt_dir, language):
    """Pitt 코퍼스 데이터 로드"""
    data = []
    
    text_dir = pitt_dir / 'textdata'
    voice_dir = pitt_dir / 'voicedata'
    
    if not (text_dir.exists() and voice_dir.exists()):
        return data
    
    # AD, HC 카테고리로 로드 (실제 디렉토리 구조에 맞게)
    categories = [
        ('HC', 0),    # Healthy Control
        ('AD', 1)     # Alzheimer's Disease  
    ]
    
    for cat_name, label in categories:
        cat_text_dir = text_dir / cat_name
        cat_voice_dir = voice_dir / cat_name
        
        if not (cat_text_dir.exists() and cat_voice_dir.exists()):
            continue
        
        # 하위 태스크 디렉토리들 (cookie, fluency, recall, sentence)
        for task_dir in cat_text_dir.iterdir():
            if task_dir.is_dir():
                voice_task_dir = cat_voice_dir / task_dir.name
                
                if not voice_task_dir.exists():
                    continue
                
                # 텍스트 파일들 로드
                for text_file in task_dir.glob('*.txt'):
                    # 해당하는 오디오 파일 찾기
                    audio_file = voice_task_dir / f"{text_file.stem}.npy"
                    
                    if not audio_file.exists():
                        # 다른 확장자로도 시도
                        audio_file = voice_task_dir / f"{text_file.stem}.wav"
                        if not audio_file.exists():
                            continue
                    
                    # 텍스트 읽기
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        if not text or len(text) < 10:  # 너무 짧은 텍스트 스킵
                            continue
                        
                        # 환자 ID 추출 (카테고리_태스크_파일명)
                        patient_id = f"{language}_{cat_name}_{task_dir.name}_{text_file.stem}"
                        
                        data.append({
                            'text': text,
                            'audio_path': str(audio_file),
                            'label': label,
                            'language': language,
                            'patient_id': patient_id
                        })
                    except Exception as e:
                        print(f"⚠️ 파일 로드 실패: {text_file} - {e}")
    
    return data

def load_greek_data(lang_dir, language):
    """그리스어 데이터 로드"""
    data = []
    
    # textdata와 voicedata 디렉토리에서 로드
    text_dir = lang_dir / 'textdata'
    voice_dir = lang_dir / 'voicedata'
    
    if text_dir.exists() and voice_dir.exists():
        # AD, HC, MCI 카테고리
        categories = [
            ('HC', 0),    # Healthy Control
            ('AD', 1),    # Alzheimer's Disease
            ('MCI', 1)    # Mild Cognitive Impairment (치매로 분류)
        ]
        
        for cat_name, label in categories:
            cat_text_dir = text_dir / cat_name
            cat_voice_dir = voice_dir / cat_name
            
            if cat_text_dir.exists() and cat_voice_dir.exists():
                # 텍스트 파일들 로드
                for text_file in cat_text_dir.glob('*.txt'):
                    audio_file = cat_voice_dir / f"{text_file.stem}.npy"
                    
                    if audio_file.exists():
                        try:
                            with open(text_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            
                            if text:
                                patient_id = f"{language}_{cat_name}_{text_file.stem}"
                                data.append({
                                    'text': text,
                                    'audio_path': str(audio_file),
                                    'label': label,
                                    'language': language,
                                    'patient_id': patient_id
                                })
                        except Exception as e:
                            print(f"⚠️ 그리스어 파일 로드 실패: {text_file} - {e}")
    
    # long, short, pilot 디렉토리에서도 로드
    for subdir_name in ['long', 'short', 'pilot']:
        subdir = lang_dir / subdir_name
        if subdir.exists():
            subdir_data = load_greek_subdir(subdir, language, subdir_name)
            data.extend(subdir_data)
    
    return data

def load_greek_subdir(subdir, language, subdir_name):
    """그리스어 서브디렉토리 로드"""
    data = []
    
    if subdir_name in ['long', 'short']:
        # AD, HC, MCI 카테고리
        categories = [
            ('HC', 0),
            ('AD', 1),
            ('MCI', 1)
        ]
        
        for cat_name, label in categories:
            cat_dir = subdir / cat_name
            if cat_dir.exists():
                # .npy와 .mp3 파일 쌍 찾기
                npy_files = list(cat_dir.glob('*.npy'))
                for npy_file in npy_files:
                    # 해당하는 텍스트 파일이 있는지 확인 (다른 디렉토리에서)
                    # 간단히 파일명을 텍스트로 사용
                    text = f"Greek audio sample {npy_file.stem}"
                    patient_id = f"{language}_{subdir_name}_{cat_name}_{npy_file.stem}"
                    
                    data.append({
                        'text': text,
                        'audio_path': str(npy_file),
                        'label': label,
                        'language': language,
                        'patient_id': patient_id
                    })
    
    return data

def load_spanish_data(lang_dir, language):
    """스페인어 데이터 로드"""
    data = []
    
    # textdata와 voicedata 디렉토리에서 로드
    text_dir = lang_dir / 'textdata'
    voice_dir = lang_dir / 'voicedata'
    
    if text_dir.exists() and voice_dir.exists():
        # AD, HC, MCI 카테고리
        categories = [
            ('HC', 0),    # Healthy Control
            ('AD', 1),    # Alzheimer's Disease
            ('MCI', 1)    # Mild Cognitive Impairment (치매로 분류)
        ]
        
        for cat_name, label in categories:
            cat_text_dir = text_dir / cat_name
            cat_voice_dir = voice_dir / cat_name
            
            if cat_text_dir.exists() and cat_voice_dir.exists():
                # 텍스트 파일들 로드
                for text_file in cat_text_dir.glob('*.txt'):
                    audio_file = cat_voice_dir / f"{text_file.stem}.npy"
                    
                    if audio_file.exists():
                        try:
                            with open(text_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            
                            if text:
                                patient_id = f"{language}_{cat_name}_{text_file.stem}"
                                data.append({
                                    'text': text,
                                    'audio_path': str(audio_file),
                                    'label': label,
                                    'language': language,
                                    'patient_id': patient_id
                                })
                        except Exception as e:
                            print(f"⚠️ 스페인어 파일 로드 실패: {text_file} - {e}")
    
    return data

def load_mandarin_data(lang_dir, language):
    """만다린 데이터 로드"""
    data = []
    
    # textdata와 voicedata 디렉토리에서 로드
    text_dir = lang_dir / 'textdata'
    voice_dir = lang_dir / 'voicedata'
    
    if text_dir.exists() and voice_dir.exists():
        # AD, HC, MCI 카테고리
        categories = [
            ('HC', 0),    # Healthy Control
            ('AD', 1),    # Alzheimer's Disease
            ('MCI', 1)    # Mild Cognitive Impairment (치매로 분류)
        ]
        
        for cat_name, label in categories:
            cat_text_dir = text_dir / cat_name
            cat_voice_dir = voice_dir / cat_name
            
            if cat_text_dir.exists() and cat_voice_dir.exists():
                # 텍스트 파일들 로드
                for text_file in cat_text_dir.glob('*.txt'):
                    audio_file = cat_voice_dir / f"{text_file.stem}.npy"
                    
                    if audio_file.exists():
                        try:
                            with open(text_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                            
                            if text:
                                patient_id = f"{language}_{cat_name}_{text_file.stem}"
                                data.append({
                                    'text': text,
                                    'audio_path': str(audio_file),
                                    'label': label,
                                    'language': language,
                                    'patient_id': patient_id
                                })
                        except Exception as e:
                            print(f"⚠️ 만다린 파일 로드 실패: {text_file} - {e}")
    
    # Lu, Ye 디렉토리에서 직접 로드
    for subdir_name in ['Lu', 'Ye']:
        subdir = lang_dir / subdir_name
        if subdir.exists():
            # .npy 파일들 직접 로드
            npy_files = list(subdir.glob('*.npy'))
            for npy_file in npy_files:
                # 파일명에서 라벨 추정 (간단한 규칙)
                label = 1 if any(x in npy_file.stem.lower() for x in ['ad', 'pd', 'df']) else 0
                text = f"Mandarin audio sample {npy_file.stem}"
                patient_id = f"{language}_{subdir_name}_{npy_file.stem}"
                
                data.append({
                    'text': text,
                    'audio_path': str(npy_file),
                    'label': label,
                    'language': language,
                    'patient_id': patient_id
                })
    
    return data

def load_generic_data(data_dir, language, default_label=None):
    """일반적인 데이터 구조 로드"""
    data = []
    
    # textdata와 voicedata 디렉토리 찾기
    text_dir = None
    voice_dir = None
    
    if (data_dir / 'textdata').exists():
        text_dir = data_dir / 'textdata'
    if (data_dir / 'voicedata').exists():
        voice_dir = data_dir / 'voicedata'
    
    # textdata와 voicedata가 없으면 직접 탐색
    if text_dir is None or voice_dir is None:
        # 직접 txt와 npy 파일 매칭
        txt_files = list(data_dir.glob('**/*.txt'))
        npy_files = list(data_dir.glob('**/*.npy'))
        
        # 파일명 기반 매칭
        txt_dict = {f.stem: f for f in txt_files}
        npy_dict = {f.stem: f for f in npy_files}
        
        for stem in txt_dict:
            if stem in npy_dict:
                try:
                    with open(txt_dict[stem], 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    # 라벨 추정 (파일 경로나 이름에서)
                    label = estimate_label(txt_dict[stem], default_label)
                    
                    # 환자 ID 생성
                    patient_id = f"{language}_{stem}"
                    
                    data.append({
                        'text': text,
                        'audio_path': str(npy_dict[stem]),
                        'label': label,
                        'language': language,
                        'patient_id': patient_id
                    })
                except Exception as e:
                    print(f"⚠️ 파일 로드 실패: {txt_dict[stem]} - {e}")
    
    else:
        # textdata/voicedata 구조로 로드
        for text_file in text_dir.glob('**/*.txt'):
            rel_path = text_file.relative_to(text_dir)
            audio_file = voice_dir / rel_path.with_suffix('.npy')
            
            if audio_file.exists():
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    label = estimate_label(text_file, default_label)
                    patient_id = f"{language}_{text_file.stem}"
                    
                    data.append({
                        'text': text,
                        'audio_path': str(audio_file),
                        'label': label,
                        'language': language,
                        'patient_id': patient_id
                    })
                except Exception as e:
                    print(f"⚠️ 파일 로드 실패: {text_file} - {e}")
    
    return data

def estimate_label(file_path, default_label=None):
    """파일 경로에서 라벨 추정"""
    if default_label is not None:
        return default_label
    
    path_str = str(file_path).lower()
    
    # 치매 관련 키워드
    dementia_keywords = ['dementia', 'alzheimer', 'ad', 'mci', 'impaired']
    control_keywords = ['control', 'healthy', 'hc', 'normal']
    
    for keyword in dementia_keywords:
        if keyword in path_str:
            return 1
    
    for keyword in control_keywords:
        if keyword in path_str:
            return 0
    
    # 기본값: 0 (정상)
    return 0

def create_stratified_split_multilingual(dataset, train_split=0.7, val_split=0.1, test_split=0.2, random_seed=42):
    """
    환자 단위 stratified split (언어와 라벨을 모두 고려)
    """
    # 환자별 데이터 그룹화
    patient_groups = defaultdict(list)
    for idx, item in enumerate(dataset):
        patient_id = item['patient_id']
        patient_groups[patient_id].append(idx)
    
    # 환자별 메타데이터 생성
    patient_metadata = []
    for patient_id, indices in patient_groups.items():
        # 첫 번째 샘플에서 메타데이터 추출
        first_item = dataset[indices[0]]
        patient_metadata.append({
            'patient_id': patient_id,
            'language': first_item['language'],
            'label': first_item['label'],
            'indices': indices,
            'sample_count': len(indices)
        })
    
    # Stratify 키 생성 (언어-라벨 조합)
    patient_stratify_keys = [f"{p['language']}_{p['label']}" for p in patient_metadata]
    patient_indices_list = list(range(len(patient_metadata)))
    
    print(f"\n📊 환자 단위 Stratified Split:")
    print(f"  전체 환자 수: {len(patient_metadata)}")
    print(f"  전체 샘플 수: {len(dataset)}")
    
    # 환자별 stratify 키 분포 확인
    stratify_dist = Counter(patient_stratify_keys)
    print(f"  Stratify 키 분포:")
    for key, count in stratify_dist.items():
        print(f"    {key}: {count}명")
    
    # 첫 번째 분할: train vs (val + test)
    train_patient_indices, temp_patient_indices = train_test_split(
        patient_indices_list,
        test_size=val_split + test_split,
        stratify=patient_stratify_keys,
        random_state=random_seed
    )
    
    # 두 번째 분할: val vs test
    temp_patient_stratify_keys = [patient_stratify_keys[i] for i in temp_patient_indices]
    val_patient_indices, test_patient_indices = train_test_split(
        temp_patient_indices,
        test_size=test_split / (val_split + test_split),
        stratify=temp_patient_stratify_keys,
        random_state=random_seed
    )
    
    # 환자 인덱스를 샘플 인덱스로 변환
    train_indices = []
    val_indices = []
    test_indices = []
    
    for patient_idx in train_patient_indices:
        train_indices.extend(patient_metadata[patient_idx]['indices'])
    
    for patient_idx in val_patient_indices:
        val_indices.extend(patient_metadata[patient_idx]['indices'])
    
    for patient_idx in test_patient_indices:
        test_indices.extend(patient_metadata[patient_idx]['indices'])
    
    # 결과 통계
    print(f"\n📈 분할 결과:")
    print(f"  훈련: {len(train_indices)}개 샘플, {len(train_patient_indices)}명 환자")
    print(f"  검증: {len(val_indices)}개 샘플, {len(val_patient_indices)}명 환자")
    print(f"  테스트: {len(test_indices)}개 샘플, {len(test_patient_indices)}명 환자")
    
    # 언어별 분포 확인
    train_lang_dist = Counter([dataset[i]['language'] for i in train_indices])
    val_lang_dist = Counter([dataset[i]['language'] for i in val_indices])
    test_lang_dist = Counter([dataset[i]['language'] for i in test_indices])
    
    print(f"\n📊 언어별 분포:")
    all_languages = set(train_lang_dist.keys()) | set(val_lang_dist.keys()) | set(test_lang_dist.keys())
    for lang in all_languages:
        train_count = train_lang_dist.get(lang, 0)
        val_count = val_lang_dist.get(lang, 0)
        test_count = test_lang_dist.get(lang, 0)
        total = train_count + val_count + test_count
        print(f"  {lang}: 훈련 {train_count}({train_count/total*100:.1f}%), "
              f"검증 {val_count}({val_count/total*100:.1f}%), "
              f"테스트 {test_count}({test_count/total*100:.1f}%)")
    
    return train_indices, val_indices, test_indices

def create_cross_lingual_split(dataset, train_languages, test_languages, random_seed=42):
    """
    Cross-lingual split: 훈련은 소스 언어만, 검증/테스트는 타겟 언어 포함
    - 훈련: 소스 언어만 (7:1:2 중 7 부분)
    - 검증: 소스 언어 일부 + 타겟 언어 일부 (1 부분)
    - 테스트: 소스 언어 일부 + 타겟 언어 일부 (2 부분)
    """
    print(f"\n🌍 Cross-lingual Split (7:1:2):")
    print(f"  훈련 언어 (소스): {train_languages}")
    print(f"  타겟 언어: {test_languages}")
    
    # 언어별로 데이터 분리
    source_data_indices = []
    target_data_indices = []
    
    for idx, item in enumerate(dataset):
        if item['language'] in train_languages:
            source_data_indices.append(idx)
        elif item['language'] in test_languages:
            target_data_indices.append(idx)
    
    print(f"  소스 언어 데이터: {len(source_data_indices)}개")
    print(f"  타겟 언어 데이터: {len(target_data_indices)}개")
    
    # 소스 언어 데이터를 7:1:2로 분할
    source_subset = [dataset[i] for i in source_data_indices]
    source_train_indices, source_val_indices, source_test_indices = create_stratified_split_multilingual(
        source_subset, 
        train_split=0.7,   # 70%
        val_split=0.1,     # 10% 
        test_split=0.2,    # 20%
        random_seed=random_seed
    )
    
    # 인덱스 매핑 (subset → original)
    train_indices = [source_data_indices[i] for i in source_train_indices]
    source_val_mapped = [source_data_indices[i] for i in source_val_indices]
    source_test_mapped = [source_data_indices[i] for i in source_test_indices]
    
    # 타겟 언어 데이터를 1:2로 분할 (val:test)
    val_indices = source_val_mapped.copy()  # 소스 언어 val로 시작
    test_indices = source_test_mapped.copy()  # 소스 언어 test로 시작
    
    if len(target_data_indices) > 0:
        target_subset = [dataset[i] for i in target_data_indices]
        
        # 타겟 언어를 1:2 비율로 val:test 분할
        target_val_ratio = 1 / (1 + 2)  # 1/3
        target_test_ratio = 2 / (1 + 2)  # 2/3
        
        _, target_val_indices, target_test_indices = create_stratified_split_multilingual(
            target_subset,
            train_split=0,
            val_split=target_val_ratio,
            test_split=target_test_ratio,
            random_seed=random_seed
        )
        
        # 타겟 언어의 val/test를 전체에 추가
        val_indices.extend([target_data_indices[i] for i in target_val_indices])
        test_indices.extend([target_data_indices[i] for i in target_test_indices])
    
    # 언어별 분포 확인
    train_langs = [dataset[i]['language'] for i in train_indices]
    val_langs = [dataset[i]['language'] for i in val_indices]
    test_langs = [dataset[i]['language'] for i in test_indices]
    
    from collections import Counter
    train_lang_dist = Counter(train_langs)
    val_lang_dist = Counter(val_langs)
    test_lang_dist = Counter(test_langs)
    
    print(f"\n✅ Cross-lingual Split 완료 (7:1:2):")
    print(f"  훈련: {len(train_indices)}개")
    for lang, count in train_lang_dist.items():
        print(f"    {lang}: {count}개")
    
    print(f"  검증: {len(val_indices)}개")
    for lang, count in val_lang_dist.items():
        print(f"    {lang}: {count}개")
    
    print(f"  테스트: {len(test_indices)}개")
    for lang, count in test_lang_dist.items():
        print(f"    {lang}: {count}개")
    
    return train_indices, val_indices, test_indices

def prepare_multilingual_dataset(data_dir, max_seq_len=512, languages=None, 
                                tokenizer_name='bert-base-uncased'):
    """
    다국어 데이터셋 준비
    """
    # 토크나이저 초기화
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # 다국어 데이터 로드
    raw_data = read_multilingual_data(data_dir, languages)
    
    # 텍스트 토큰화
    print(f"\n🔤 텍스트 토큰화 중...")
    processed_data = []
    
    for item in raw_data:
        try:
            # BERT 토큰화
            encoding = tokenizer.encode_plus(
                item['text'],
                add_special_tokens=True,
                max_length=max_seq_len,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            
            processed_item = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'audio_path': item['audio_path'],
                'label': item['label'],
                'language': item['language'],
                'patient_id': item['patient_id']
            }
            processed_data.append(processed_item)
            
        except Exception as e:
            print(f"⚠️ 토큰화 실패: {item['patient_id']} - {e}")
    
    print(f"✅ {len(processed_data)}개 샘플 토큰화 완료")
    
    return MultilingualDementiaDataset(processed_data)

def collate_fn_multilingual(batch):
    """
    다국어 데이터셋용 collate function
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    audio = torch.stack([item['audio'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
    languages = [item['language'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'audio': audio,
        'labels': labels,
        'languages': languages,
        'patient_ids': patient_ids
    }
