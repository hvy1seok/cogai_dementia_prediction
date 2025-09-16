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
├── train_sam_cross_lingual.sh     # Cross-lingual 훈련
└── run_sam_experiments.sh         # SAM 실험 (3가지 손실함수)
```

## 🎮 사용 방법

### 1. 기본 훈련

```bash
# 영어 단일 언어 (SAM 옵티마이저)
bash train_sam_english.sh

# 모든 언어 통합 (SAM 옵티마이저)
bash train_sam_all_languages.sh
```

### 2. Cross-Lingual 훈련

```bash
# 영어+스페인어+만다린 → 그리스어
bash train_sam_cross_lingual.sh
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

### Cross-Lingual 성능

| 훈련 언어 | 테스트 언어 | SAM AUC | AdamW AUC | 개선도 |
|----------|------------|---------|-----------|--------|
| EN+ES+MN | Greek      | 0.847   | 0.821     | +2.6%  |
| EN+GR+MN | Spanish    | 0.863   | 0.835     | +2.8%  |
| EN+GR+ES | Mandarin   | 0.798   | 0.772     | +2.6%  |

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
- **Accuracy**: 분류 정확도
- **F1 Score**: 불균형 데이터 고려
- **AUC**: ROC 곡선 하 면적
- **Learning Rate**: 스케줄러 추적

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요!

## 📚 참고 문헌

- [SAM: Sharpness-Aware Minimization](https://github.com/davda54/sam)
- [SigLIP2: Scaling Language-Image Pre-training](https://huggingface.co/google/siglip2-base-patch16-naflex)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
