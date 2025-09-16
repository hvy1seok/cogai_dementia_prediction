# 치매 예측 멀티모달 모델 (Dementia Prediction Multimodal Model)

이 프로젝트는 오디오와 텍스트 데이터를 활용하여 치매를 예측하는 멀티모달 딥러닝 모델을 구현합니다. Hydra를 사용하여 설정을 관리하고, MLflow를 통해 실험을 추적합니다.

## 프로젝트 구조

```
├── config/
│   ├── config.yaml           # 기본 설정
│   ├── model/
│   │   └── multimodal.yaml  # 모델 설정
│   ├── dataset/
│   │   └── pitt.yaml       # 데이터셋 설정
│   └── training/
│       └── default.yaml    # 학습 설정
├── dataset.py              # 데이터셋 관련 코드
├── models.py               # 모델 정의
├── main.py                # 실행 스크립트
└── README.md
```

## 주요 기능

### 데이터셋 (dataset.py)

- `PittDataset`: PyTorch Dataset 클래스 구현
  - `__init__(text, attentions, audio_paths, labels)`: 데이터셋 초기화
  - `__len__()`: 데이터셋 크기 반환
  - `__getitem__(item)`: 데이터 샘플 반환

- `read_file(file_path)`: 텍스트 파일 읽기
- `read_data(pitt_path)`: Pitt 데이터셋 로드
- `collate_fn(batch, pad_val, device, max_seq_len)`: 배치 데이터 처리
- `prepare_dataset(pitt_path, max_seq_len)`: 데이터셋 준비 및 전처리

### 모델 (models.py)

1. `AudioModel`:
   - ResNet101 기반 오디오 특징 추출
   - `__init__(output_dim=256)`: 모델 초기화
   - `forward(audio)`: 오디오 특징 추출

2. `TextModel`:
   - BERT 기반 텍스트 특징 추출
   - `__init__(text_model_type=1, d_model=1024, dropout=0.2)`: 모델 초기화
   - `forward(text, attention)`: 텍스트 특징 추출

3. `MultimodalModel`:
   - 오디오와 텍스트 특징을 결합한 멀티모달 모델
   - `__init__(text_model_type=1, dropout=0.3)`: 모델 초기화
   - `forward(text, attention, audio)`: 멀티모달 예측

4. `train_model()`: 모델 학습 및 검증 함수
   - 학습/검증 단계 구현
   - MLflow를 통한 메트릭 로깅
   - 최적 모델 저장

### 메인 스크립트 (main.py)

- `set_seed(seed)`: 재현성을 위한 시드 설정
- `main(cfg)`: Hydra 설정 기반 실행
  - MLflow 실험 설정
  - 데이터 로딩 및 전처리
  - 모델 학습 실행

## 설정 (config/)

### 1. 기본 설정 (config.yaml)
- Hydra 실행 디렉토리 설정
- MLflow 실험 설정

### 2. 모델 설정 (model/multimodal.yaml)
- 텍스트 모델 타입
- 드롭아웃 비율
- 오디오/텍스트 모델 파라미터

### 3. 데이터셋 설정 (dataset/pitt.yaml)
- 데이터 경로
- 배치 크기
- 시퀀스 길이
- 학습/검증 분할 비율

### 4. 학습 설정 (training/default.yaml)
- 에포크 수
- 학습률
- 옵티마이저 설정
- 시드값

## 사용법

1. 환경 설정:
```bash
pip install torch torchvision transformers scikit-learn hydra-core mlflow omegaconf numpy
```

2. 기본 실행:
```bash
python main.py
```

3. 설정 수정하여 실행:
```bash
python main.py model.text_model_type=2 training.learning_rate=1e-5
```

4. 실험 결과 확인:
```bash
mlflow ui
```
웹 브라우저에서 `http://localhost:5000` 접속

## 데이터셋 구조

Pitt 데이터셋은 다음 구조를 따라야 합니다:
```
Pitt/
├── textdata/
│   ├── Control/
│   └── Dementia/
├── voicedata/
│   ├── Control/
│   └── Dementia/
├── controlCha.txt
└── dementiaCha.txt
```

## MLflow 실험 추적

다음 메트릭들이 자동으로 기록됩니다:
- 학습/검증 손실
- 학습/검증 정확도
- 에포크 시간
- 최고 검증 정확도
- 혼동 행렬

## 참고사항

- CUDA가 사용 가능한 경우 자동으로 GPU를 사용합니다.
- 학습 중간 결과는 Hydra 출력 디렉토리에 저장됩니다.
- 모든 실험 결과는 MLflow를 통해 추적되며, 웹 UI에서 확인 가능합니다.