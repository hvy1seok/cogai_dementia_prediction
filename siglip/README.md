# SigLIP2 기반 언어 무관 치매 진단 시스템

이 프로젝트는 Google의 SigLIP2 모델을 활용하여 언어에 구애받지 않는 치매 진단 시스템을 구현합니다. 오디오 데이터를 멜스펙토그램으로 변환하고 텍스트 전사본과 함께 멀티모달 학습을 통해 치매를 진단합니다.

## 🚀 주요 특징

- **언어 무관 학습**: 다양한 언어(영어, 그리스어, 한국어, 스페인어, 프랑스어 등)의 데이터로 학습
- **멀티모달 접근**: 오디오(멜스펙토그램) + 텍스트 전사본을 동시에 활용
- **PyTorch Lightning**: 체계적인 훈련 관리 및 실험 추적
- **wandb 통합**: 실시간 실험 모니터링 및 로깅
- **효율적인 추론**: 훈련된 모델을 사용한 빠른 예측

## 📁 프로젝트 구조

```
siglip/
├── config.py              # 설정 파일
├── data_processor.py      # 데이터 처리 및 전처리
├── model.py              # SigLIP2 기반 모델 정의
├── trainer.py            # PyTorch Lightning 훈련 스크립트
├── inference.py          # 추론 스크립트
├── requirements.txt      # 의존성 패키지
└── README.md            # 이 파일
```

## 🛠️ 설치

1. 의존성 패키지 설치:
```bash
pip install -r requirements.txt
```

2. wandb 로그인 (선택사항):
```bash
wandb login
```

## 📊 데이터 형식

데이터는 다음과 같은 구조로 준비되어야 합니다:

```
dementia_fulldata/
├── English/
│   ├── metadata.csv
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── Greek/
│   ├── metadata.txt
│   ├── audio1.wav
│   └── ...
└── Korean/
    ├── metadata.csv
    └── ...
```

### 메타데이터 파일 형식

**CSV 형식:**
```csv
audio_file,transcript,dementia
audio1.wav,"Hello, how are you today?",0
audio2.wav,"I don't remember what I was saying...",1
```

**텍스트 형식 (탭으로 구분):**
```
audio1.wav	Hello, how are you today?	0
audio2.wav	I don't remember what I was saying...	1
```

- `audio_file`: 오디오 파일명
- `transcript`: 텍스트 전사본
- `dementia`: 라벨 (0: 정상, 1: 치매)

## 🚀 사용법

### 1. 데이터 파서 테스트 (권장)

먼저 데이터가 올바르게 로드되는지 확인:
```bash
python test_parser.py
```

### 2. 모델 훈련

#### 대화형 훈련 (권장):
```bash
python run_siglip_training.py
```

#### 직접 훈련:
기본 훈련 (모든 언어):
```bash
python trainer.py --data_dir ../training_dset
```

특정 언어만 훈련:
```bash
python trainer.py \
    --data_dir ../training_dset \
    --parser English
```

고급 옵션:
```bash
python trainer.py \
    --data_dir ../training_dset \
    --model_name google/siglip2-base-patch16-224 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --parser all
```

### 3. 추론

단일 예측:
```bash
python inference.py \
    --model_path ../modules/outputs/siglip/checkpoints/best_model.ckpt \
    --audio_path path/to/audio.wav \
    --text "Hello, how are you today?" \
    --language English
```

배치 예측:
```bash
python inference.py \
    --model_path ../modules/outputs/siglip/checkpoints/best_model.ckpt \
    --batch_file batch_data.json \
    --output predictions.json
```

배치 데이터 JSON 형식:
```json
[
    {
        "audio_path": "path/to/audio1.wav",
        "text": "Hello, how are you today?",
        "language": "English"
    },
    {
        "audio_path": "path/to/audio2.wav",
        "text": "안녕하세요, 오늘 기분이 어떠세요?",
        "language": "Korean"
    }
]
```

## ⚙️ 설정

`config.py`에서 다음 설정들을 조정할 수 있습니다:

### 모델 설정
- `model_name`: SigLIP2 모델 버전
- `max_length`: 텍스트 최대 길이
- `image_size`: 멜스펙토그램 이미지 크기

### 오디오 처리 설정
- `sample_rate`: 샘플링 레이트
- `n_mels`: 멜 스케일 빈 수
- `n_fft`: FFT 윈도우 크기
- `hop_length`: 홉 길이

### 훈련 설정
- `batch_size`: 배치 크기
- `learning_rate`: 학습률
- `num_epochs`: 에포크 수
- `weight_decay`: 가중치 감쇠

## 📈 모니터링

훈련 중에는 wandb를 통해 다음 메트릭들을 실시간으로 모니터링할 수 있습니다:

- **훈련 메트릭**: `train_loss`, `train_acc`
- **검증 메트릭**: `val_loss`, `val_acc`, `val_f1`, `val_precision`, `val_recall`, `val_auc`
- **테스트 메트릭**: `test_accuracy`, `test_f1`, `test_auc`
- **학습률**: `lr`

## 🔬 실험 결과

모델은 다음과 같은 특징을 가집니다:

1. **언어 무관성**: 다양한 언어에서 일관된 성능
2. **멀티모달 학습**: 오디오와 텍스트 정보를 동시에 활용
3. **효율적인 추론**: 빠른 예측 속도
4. **확장 가능성**: 새로운 언어 추가 용이

## 🎯 성능 최적화 팁

1. **배치 크기 조정**: GPU 메모리에 맞게 배치 크기 조정
2. **학습률 스케줄링**: CosineAnnealingLR 사용으로 안정적인 학습
3. **조기 종료**: 과적합 방지를 위한 조기 종료 설정
4. **혼합 정밀도**: FP16 사용으로 메모리 효율성 향상

## 🐛 문제 해결

### 일반적인 오류

1. **CUDA 메모리 부족**:
   - 배치 크기 줄이기
   - 혼합 정밀도 훈련 활성화
   - 그래디언트 누적 사용

2. **데이터 로딩 오류**:
   - 오디오 파일 경로 확인
   - 메타데이터 파일 형식 확인
   - 파일 인코딩 확인 (UTF-8)

3. **모델 로딩 오류**:
   - 체크포인트 파일 경로 확인
   - 모델 버전 호환성 확인

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

## 📚 참고 자료

- [SigLIP2 논문](https://arxiv.org/abs/2403.15396)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Weights & Biases](https://wandb.ai/) 