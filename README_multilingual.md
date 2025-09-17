# 다국어 멀티모달 치매 진단 모델

기존 멀티모달 모델을 SigLIP 스타일로 확장하여 다국어 지원과 Cross-lingual 실험이 가능한 치매 진단 시스템입니다.

## 🌟 주요 특징

- **다국어 지원**: English, Greek, Spanish, Mandarin 4개 언어
- **환자 단위 Stratified Split**: 환자별 데이터 누수 방지
- **Cross-lingual 실험**: 언어 간 전이 학습 성능 평가
- **언어별 상세 분석**: 각 언어별 정확도, AUC, F1-score 계산
- **wandb 통합**: 실시간 모니터링 및 시각화
- **멀티모달 융합**: 오디오(ResNet-101) + 텍스트(BERT) 결합

## 📁 프로젝트 구조

```
├── main_multilingual.py          # 메인 실행 스크립트
├── dataset_multilingual.py       # 다국어 데이터셋 처리
├── models_multilingual.py        # 멀티모달 모델 및 훈련 로직
├── run_all_languages.sh         # 전체 언어 실험 스크립트
├── run_cross_lingual.sh          # Cross-lingual 실험 스크립트
├── requirements_multilingual.txt # 의존성 패키지
└── README_multilingual.md        # 이 파일
```

## 🚀 설치 및 설정

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements_multilingual.txt
```

### 2. wandb 설정
```bash
wandb login
# 또는 API 키 직접 설정
export WANDB_API_KEY="your_api_key"
```

### 3. 데이터 준비
데이터는 다음과 같은 구조로 준비해야 합니다:
```
training_dset/
├── English/
│   └── Pitt/
│       ├── textdata/
│       └── voicedata/
├── Greek/
│   └── Dem@Care/
│       ├── textdata/
│       └── voicedata/
├── Spanish/
│   ├── textdata/
│   └── voicedata/
└── Mandarin/
    ├── textdata/
    └── voicedata/
```

## 🎯 사용 방법

### 1. 전체 언어 실험 (환자 단위 Stratified Split)
```bash
# 스크립트 실행 권한 부여
chmod +x run_all_languages.sh

# 실험 실행
./run_all_languages.sh
```

**또는 직접 실행:**
```bash
python main_multilingual.py \
    --mode all_languages \
    --data_dir training_dset \
    --languages English Greek Spanish Mandarin \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 2e-5
```

### 2. Cross-lingual 실험 (언어별 전이 학습)
```bash
# 전체 조합 실험
chmod +x run_cross_lingual.sh
./run_cross_lingual.sh
```

**또는 특정 조합만 실행:**
```bash
python main_multilingual.py \
    --mode cross_lingual \
    --train_languages English Greek Spanish \
    --test_languages Mandarin \
    --batch_size 16 \
    --num_epochs 50
```

### 3. 커스텀 실험
```bash
python main_multilingual.py \
    --mode cross_lingual \
    --train_languages English Spanish \
    --test_languages Greek Mandarin \
    --text_model_type 2 \
    --dropout 0.4 \
    --learning_rate 1e-5 \
    --batch_size 32
```

## ⚙️ 주요 파라미터

### 데이터 관련
- `--data_dir`: 데이터 디렉토리 경로 (기본: training_dset)
- `--languages`: 사용할 언어 목록
- `--max_seq_len`: 최대 시퀀스 길이 (기본: 512)
- `--batch_size`: 배치 크기 (기본: 16)

### 모델 관련
- `--text_model_type`: 텍스트 모델 타입 (1: BERT만, 2: BERT+LSTM)
- `--dropout`: 드롭아웃 비율 (기본: 0.3)

### 훈련 관련
- `--num_epochs`: 에포크 수 (기본: 50)
- `--learning_rate`: 학습률 (기본: 2e-5)
- `--weight_decay`: 가중치 감쇠 (기본: 0.01)
- `--seed`: 랜덤 시드 (기본: 42)

### 실험 모드
- `--mode`: 실험 모드 (`all_languages` 또는 `cross_lingual`)
- `--train_languages`: Cross-lingual에서 훈련 언어
- `--test_languages`: Cross-lingual에서 테스트 언어

## 📊 실험 결과

### 1. 전체 언어 실험 결과
- **목적**: 모든 언어에서 환자 단위로 분할하여 언어별 성능 분석
- **분할 방식**: 7:1.5:1.5 (train:val:test)
- **출력**: 언어별 정확도, AUC, F1-score

### 2. Cross-lingual 실험 결과
- **목적**: 언어 간 전이 학습 효과 분석
- **실험 조합**:
  1. English+Greek+Spanish → Mandarin
  2. English+Greek+Mandarin → Spanish  
  3. English+Spanish+Mandarin → Greek
  4. Greek+Spanish+Mandarin → English

### 3. 결과 파일
- **모델**: `best_model_*.pth`
- **결과**: `checkpoints/results_*.json`
- **로그**: wandb 대시보드

## 🌍 언어별 성능 분석

실험 완료 후 다음과 같은 언어별 메트릭이 자동 계산됩니다:

```
📊 언어별 성능 분석 (임계값: 0.523):
============================================================
🌍 English:
  📊 샘플: 1234개 (정상: 567, 치매: 667)
  🎯 정확도: 0.856
  📈 AUC: 0.912
  🔍 정밀도: 0.834
  🔍 재현율: 0.889
  🔍 F1: 0.861

🌍 Greek:
  📊 샘플: 456개 (정상: 234, 치매: 222)
  🎯 정확도: 0.789
  📈 AUC: 0.834
  🔍 정밀도: 0.756
  🔍 재현율: 0.812
  🔍 F1: 0.783
```

## 📈 wandb 시각화

wandb 대시보드에서 다음을 모니터링할 수 있습니다:

1. **훈련 메트릭**: Loss, Accuracy, AUC 추이
2. **언어별 성능**: 각 언어별 상세 메트릭
3. **실험 비교**: Cross-lingual vs All-languages 성능 비교
4. **하이퍼파라미터**: 최적 파라미터 조합 분석

## 🔧 모델 아키텍처

### 1. 텍스트 인코더
- **BERT-base-uncased**: 768차원 특징
- **선택사항**: LSTM 추가 (1024차원)

### 2. 오디오 인코더
- **ResNet-101**: 사전훈련된 모델
- **출력**: 256차원 특징

### 3. 융합 및 분류
- **Concatenation**: 텍스트 + 오디오 특징
- **분류기**: Dropout + ReLU + Linear
- **출력**: 이진 분류 (정상 vs 치매)

## 🎯 Cross-lingual 분석 인사이트

### 1. 언어 간 전이 효과
- 어떤 언어가 다른 언어로 잘 전이되는지 분석
- 언어 유사성과 전이 성능의 상관관계

### 2. Zero-shot 성능
- 미학습 언어에서의 진단 성능
- 언어 독립적 특징 학습 정도

### 3. 언어별 특성
- 언어별 치매 진단 난이도
- 언어 특화 패턴 vs 공통 패턴

## 🚨 주의사항

1. **GPU 메모리**: 배치 크기가 클 경우 GPU 메모리 부족 가능
2. **데이터 경로**: 데이터 구조가 예상과 다를 경우 `load_*_data` 함수 수정 필요
3. **wandb 로그인**: wandb 사용 전 로그인 필요
4. **환자 ID**: 환자 단위 분할을 위해 고유한 patient_id 필요

## 📚 참고자료

- **BERT**: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- **ResNet**: [He et al., 2016](https://arxiv.org/abs/1512.03385)
- **Cross-lingual Transfer**: [Pires et al., 2019](https://arxiv.org/abs/1906.01502)

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
