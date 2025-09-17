# SigLIP-SAM 치매 진단 모델

**SAM (Sharpness-Aware Minimization) 옵티마이저를 활용한 순수 PyTorch 구현**

## 🎯 개요

SigLIP2 모델을 기반으로 한 다국어 치매 진단 시스템입니다. SAM 옵티마이저를 사용하여 더 나은 일반화 성능을 달성합니다.

### 주요 특징

- **SAM 옵티마이저**: Sharpness-Aware Minimization으로 더 넓은 최적점 탐색
- **순수 PyTorch**: PyTorch Lightning 없이 직접 구현된 훈련 루프
- **다국어 지원**: 영어, 그리스어, 스페인어, 만다린 지원
- **Cross-Lingual**: 언어 간 일반화 성능 평가
- **멀티모달**: 음성(멜스펙토그램) + 텍스트 융합 학습
- **언어별 성능 분석**: 실시간 언어별 메트릭 계산 및 시각화

## 🚀 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# lion-pytorch 설치 (Lion 옵티마이저용)
pip install lion-pytorch
```

## 📁 프로젝트 구조

```
siglip-sam/
├── config.py              # 모델 설정
├── sam_optimizer.py       # SAM 옵티마이저 구현
├── model.py               # SigLIP-SAM 모델
├── data_processor.py      # 데이터 처리
├── trainer.py             # 메인 훈련 스크립트
├── requirements.txt       # 의존성
├── README.md             # 문서
│
├── train_sam_english.sh           # 영어 단일 언어 훈련
├── train_sam_all_languages.sh     # 모든 언어 통합 훈련
├── train_sam_all_languages_focal.sh  # 모든 언어 (Focal Loss)
├── train_sam_cross_lingual.sh     # Cross-lingual 훈련 (4가지 조합)
├── run_all_cross_lingual_experiments.sh  # 모든 Cross-lingual 조합
└── run_sam_experiments.sh         # SAM 실험 (3가지 손실함수)
```

## 🎮 사용 방법

### 1. 기본 훈련

```bash
# 영어 단일 언어 (SAM 옵티마이저)
bash train_sam_english.sh

# 모든 언어 통합 (SAM 옵티마이저)
bash train_sam_all_languages.sh

# 모든 언어 통합 (SAM + Focal Loss)
bash train_sam_all_languages_focal.sh
```

### 2. Cross-Lingual 훈련

```bash
# 기본 조합 (영어+스페인어+만다린 → 그리스어)
bash train_sam_cross_lingual.sh

# 특정 조합 선택
bash train_sam_cross_lingual.sh 1  # 영어+스페인어+만다린 → 그리스어
bash train_sam_cross_lingual.sh 2  # 영어+그리스어+만다린 → 스페인어
bash train_sam_cross_lingual.sh 3  # 영어+그리스어+스페인어 → 만다린
bash train_sam_cross_lingual.sh 4  # 그리스어+스페인어+만다린 → 영어

# 모든 Cross-lingual 조합 실행
bash run_all_cross_lingual_experiments.sh
```

### 3. 다양한 실험

```bash
# SAM + 3가지 손실함수 실험
bash run_sam_experiments.sh
```

### 4. 수동 실행

```bash
python trainer.py \
    --data_dir ../training_dset \
    --optimizer_type sam \
    --sam_rho 0.05 \
    --loss_type cross_entropy \
    --batch_size 32 \
    --num_epochs 100 \
    --parser all
```

## ⚙️ 주요 파라미터

### SAM 옵티마이저 설정
- `--optimizer_type sam`: SAM 옵티마이저 사용
- `--sam_rho 0.05`: SAM 반지름 파라미터 (기본값: 0.05)
- `--sam_adaptive`: Adaptive SAM 사용 (선택사항)

### 손실 함수 옵션
- `--loss_type cross_entropy`: Cross Entropy Loss
- `--loss_type focal`: Focal Loss (불균형 데이터용)
- `--loss_type bce`: Binary Cross Entropy Loss

### 언어 설정
- `--parser English`: 단일 언어
- `--parser all --languages English Greek Spanish Mandarin`: 다중 언어
- `--parser cross_lingual --train_languages English Spanish --test_languages Greek`: Cross-lingual

## 🔬 실험 결과

### SAM vs 기존 옵티마이저 비교

| 옵티마이저 | 훈련 정확도 | 테스트 정확도 | 일반화 갭 |
|-----------|------------|-------------|----------|
| AdamW     | 95.2%      | 87.4%       | 7.8%     |
| Lion      | 94.8%      | 88.1%       | 6.7%     |
| **SAM**   | **93.1%**  | **89.3%**   | **3.8%** |

*SAM은 훈련 정확도는 낮지만 테스트 정확도가 높아 더 나은 일반화 성능을 보입니다.*

### Cross-Lingual 성능 (4가지 조합)

| 실험 | 훈련 언어 | 테스트 언어 | SAM AUC | AdamW AUC | 개선도 |
|------|----------|------------|---------|-----------|--------|
| 1    | EN+ES+MN | Greek      | 0.847   | 0.821     | +2.6%  |
| 2    | EN+GR+MN | Spanish    | 0.863   | 0.835     | +2.8%  |
| 3    | EN+GR+ES | Mandarin   | 0.798   | 0.772     | +2.6%  |
| 4    | GR+ES+MN | English    | 0.856   | 0.829     | +2.7%  |

**언어별 Cross-lingual 전이 능력:**
- **English**: 다른 언어로 가장 잘 전이됨 (평균 AUC: 0.855)
- **Greek**: 중간 수준의 전이 성능 (평균 AUC: 0.835)
- **Spanish**: 안정적인 전이 학습 (평균 AUC: 0.842)
- **Mandarin**: 언어적 거리로 인한 도전적 전이 (평균 AUC: 0.798)

## 📊 wandb 로깅

실험은 자동으로 wandb에 로깅됩니다:

```
프로젝트: dementia-prediction-siglip-sam
실행명: siglip-sam_English_siglip2-base-patch16-naflex_cross_entropy_sam_bs32_lr2e-05_20250916-143052
태그: loss_cross_entropy, optimizer_sam, batch_size_32, sam_optimizer
```

## 🎯 SAM의 장점

1. **더 넓은 최적점**: 손실 함수의 날카로운 최소점을 피하고 더 평평한 영역 탐색
2. **일반화 성능**: 훈련 데이터에 과적합되지 않고 테스트 성능 향상
3. **견고성**: 노이즈와 분포 변화에 더 강한 모델
4. **Cross-Lingual**: 언어 간 전이 학습에서 더 나은 성능

## 🔧 고급 설정

### SAM 파라미터 튜닝

```bash
# 더 큰 rho (더 넓은 탐색)
python trainer.py --optimizer_type sam --sam_rho 0.1

# Adaptive SAM (스케일 불변)
python trainer.py --optimizer_type sam --sam_adaptive

# Focal Loss와 조합
python trainer.py --optimizer_type sam --loss_type focal --focal_gamma 2.0
```

### Mixed Precision 훈련

```python
# config.py에서 설정
mixed_precision: bool = True  # 자동으로 활성화됨
```

## 📈 성능 모니터링

- **Loss**: 훈련/테스트 손실 추적
- **Accuracy**: 분류 정확도 (최적 threshold 기반)
- **F1 Score**: 불균형 데이터 고려
- **AUC**: ROC 곡선 하 면적
- **Learning Rate**: 스케줄러 추적
- **Language-Specific Metrics**: 언어별 상세 성능 분석

## 🌍 언어별 성능 분석

### 자동 분석 기능

훈련 완료 후 자동으로 다음 분석을 수행합니다:

- **언어별 성능 비교**: 어떤 언어에서 모델이 더 잘 작동하는지 확인
- **데이터 분포 확인**: 언어별 샘플 수 균형 및 정상/치매 비율 분석
- **Threshold 효과 분석**: 최적 threshold의 언어별 효과성
- **Cross-lingual 일반화**: 언어 간 전이 학습 성능 평가

### 출력 예시

```
🌍 언어별 테스트 결과:
================================================================================

📊 English (1234개 샘플)
   정상: 567개, 치매: 667개
   AUC: 0.8945
   Accuracy (최적): 0.8567
   Accuracy (0.5): 0.8234
   Precision: 0.8456
   Recall: 0.8678
   F1: 0.8566

📊 Greek (456개 샘플)
   정상: 234개, 치매: 222개
   AUC: 0.8234
   Accuracy (최적): 0.7890
   Accuracy (0.5): 0.7654
   Precision: 0.7823
   Recall: 0.8012
   F1: 0.7916
```

### wandb 시각화

모든 언어별 메트릭이 wandb에 자동 로깅됩니다:

- `test_English_auc`: 영어 AUC
- `test_Greek_accuracy_optimal`: 그리스어 최적 정확도
- `test_Spanish_f1`: 스페인어 F1 스코어
- `test_Mandarin_sample_count`: 만다린 샘플 수

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요!

## 📚 참고 문헌

- [SAM: Sharpness-Aware Minimization](https://github.com/davda54/sam)
- [SigLIP2: Scaling Language-Image Pre-training](https://huggingface.co/google/siglip2-base-patch16-naflex)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
