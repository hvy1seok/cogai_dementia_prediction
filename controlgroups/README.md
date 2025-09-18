# Control Groups for Dementia Prediction

치매 예측 실험을 위한 **대조군(Control Groups)** 모델들입니다. 
SigLIP2 모델의 성능을 평가하기 위한 **최소·충분 세트(Must)**로 구성되었습니다.

## 📋 대조군 모델 목록

### 1. 🎵 Audio-only (ViT-Spec)
- **설명**: 오디오 스펙트로그램만 사용하는 멀티링궐 모델
- **구조**: 스펙트로그램 → ViT → [CLS] → FC(sigmoid)
- **특징**: 
  - 언어 독립적인 음성 패턴 학습
  - ViT (Vision Transformer)로 스펙트로그램 처리
  - 텍스트 정보 없이 음성만으로 진단

### 2. 📝 Text-only (Gemma Encoder)
- **설명**: 전사 텍스트만 사용하는 멀티링궐 모델
- **구조**: 전사 → Gemma([CLS]) → FC(sigmoid)
- **특징**:
  - Gemma (256K vocab) 다국어 토크나이저
  - [CLS] 토큰 기반 문장 표현 학습
  - 언어학적 패턴과 의미 정보 활용

### 3. 🔗 Concat (ViT + XLM-R)
- **설명**: Late Fusion 방식의 멀티모달 모델
- **구조**: 두 임베딩 late fusion(concat) → 2층 FFN(sigmoid)
- **특징**:
  - 각 모달리티의 독립적 특징 학습
  - 오디오-텍스트 상호보완적 정보 활용
  - 단순하지만 효과적인 융합 방식

## 🚀 실행 방법

### 개별 모델 실행
```bash
# Audio-only 모델
bash controlgroups/run_audio_only_en_cn.sh

# Text-only 모델
bash controlgroups/run_text_only_en_cn.sh

# Concat 모델
bash controlgroups/run_concat_en_cn.sh
```

### 통합 실행
```bash
# 모든 대조군 순차 실행
bash controlgroups/run_all_control_groups.sh all

# 특정 모델만 실행
bash controlgroups/run_all_control_groups.sh 1  # Audio-only
bash controlgroups/run_all_control_groups.sh 2  # Text-only
bash controlgroups/run_all_control_groups.sh 3  # Concat
```

## ⚙️ 공통 설정

### 데이터 설정
- **언어**: English, Mandarin (학습이 잘 되는 언어에 집중)
- **데이터 분할**: 환자 단위 (Speaker-Independent)
- **분할 비율**: Train 70%, Val 10%, Test 20%

### 훈련 설정
- **배치 크기**: 64
- **학습률**: 2e-5
- **에포크**: 100
- **Early Stopping**: 10 epochs patience
- **손실 함수**: Focal Loss (클래스 불균형 해결)
- **평가 기준**: 평균 Macro F1 (언어 편향 방지)

### 모델 설정
- **Audio Encoder**: ViT-base-patch16-224
- **Text Encoder**: 
  - Gemma-2b (Text-only)
  - XLM-RoBERTa-base (Concat)
- **자동 클래스 가중치**: 활성화
- **환자 단위 분할**: 활성화

## 📊 실험 목적

### 1. 성능 비교 베이스라인
- SigLIP2 모델의 성능 우위성 검증
- 각 모달리티의 기여도 분석
- 융합 방식의 효과 비교

### 2. 모달리티별 분석
- **Audio-only**: 음성 신호의 진단 능력
- **Text-only**: 언어학적 정보의 진단 능력
- **Multimodal**: 모달리티 융합의 효과

### 3. 임상 적용성 검증
- Speaker-Independent 평가
- 언어별 성능 분석
- 실제 임상 환경 시뮬레이션

## 📈 평가 지표

### 전체 성능
- **Accuracy**: 전체 정확도
- **Macro F1**: 클래스 균형 고려 F1
- **AUC**: ROC 곡선 하 면적

### 언어별 성능
- **English**: 영어 데이터 성능
- **Mandarin**: 중국어 데이터 성능
- **Average**: 언어별 성능 평균

### 임상 지표
- **Sensitivity**: 치매 검출 민감도
- **Specificity**: 정상 분류 특이도
- **Precision/Recall**: 정밀도/재현율

## 🏗️ 파일 구조

```
controlgroups/
├── __init__.py              # 패키지 초기화
├── config.py                # 설정 클래스들
├── models.py                # 대조군 모델 구현
├── data_processor.py        # 데이터 처리
├── train_audio_only.py      # Audio-only 훈련 스크립트
├── train_text_only.py       # Text-only 훈련 스크립트
├── train_concat.py          # Concat 훈련 스크립트
├── run_audio_only_en_cn.sh  # Audio-only 실행 스크립트
├── run_text_only_en_cn.sh   # Text-only 실행 스크립트
├── run_concat_en_cn.sh      # Concat 실행 스크립트
├── run_all_control_groups.sh # 통합 실행 스크립트
└── README.md                # 이 파일
```

## 🎯 기대 결과

### 가설
1. **SigLIP2 > Concat > Text-only ≈ Audio-only**
2. **멀티모달 융합이 단일 모달리티보다 우수**
3. **True SigLIP2가 Late Fusion보다 우수**

### 검증 항목
- 전체 성능에서 SigLIP2의 우위성
- 언어별 성능에서 일관된 개선
- 임상 적용성에서 안정적인 성능

## 🔧 커스터마이징

### 언어 변경
```bash
# 4개 언어 사용 시
--languages English Greek Spanish Mandarin
```

### 모델 변경
```bash
# 다른 텍스트 인코더 사용
--text_encoder xlm-roberta-large

# 다른 오디오 인코더 사용
--audio_encoder google/vit-large-patch16-224
```

### 설정 변경
```bash
# 배치 크기 변경
--batch_size 32

# Early stopping 변경
--early_stopping_patience 15
```

## 📝 주의사항

1. **데이터 경로**: `../../training_dset`이 올바른지 확인
2. **GPU 메모리**: 배치 크기가 GPU 메모리에 맞는지 확인
3. **Wandb 로깅**: 필요시 `--no_wandb`로 비활성화
4. **환자 단위 분할**: Speaker-Independent 평가를 위해 필수

## 🎉 결론

이 대조군 모델들은 SigLIP2의 성능을 객관적으로 평가하고, 
각 모달리티와 융합 방식의 기여도를 분석하는 데 필수적인 
**최소·충분 베이스라인**을 제공합니다.

모든 모델이 동일한 조건(데이터, 설정, 평가)에서 훈련되므로, 
공정하고 신뢰할 수 있는 비교 분석이 가능합니다.
