#!/bin/bash
# 진정한 SigLIP2 - 2개 언어 통합 훈련 (영어 + 중국어) - ERM(AdamW)

echo "=== 진정한 SigLIP2 (ERM: AdamW) - 2개 언어 통합 훈련 시작 (영어 + 중국어) ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip-sam/True_SigLIP2_2Languages_EN_CN_ERM"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=64
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# 2개 언어 설정 (영어 + 중국어)
LANGUAGES="English Mandarin"

# ERM + Focal Loss 설정
OPTIMIZER_TYPE="adamw"
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
AUTO_CLASS_WEIGHTS="--auto_class_weights"

# True SigLIP2 Multi-Loss 가중치 (훈련 스크립트와 동일)
EMA_MOMENTUM=0.999
SILC_WEIGHT=0.2
SIGMOID_WEIGHT=1.0
LOCA_WEIGHT=1.0
CLASSIFICATION_WEIGHT=1.0

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "Python 확인..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
  PYTHON_CMD="python"
else
  echo "❌ Python을 찾을 수 없습니다."
  exit 1
fi

echo "🚀 ERM(AdamW) 훈련 시작..."
$PYTHON_CMD true_siglip2_trainer.py \
  --data_dir "$DATA_DIR" \
  --output_dir "../modules/outputs/siglip-sam" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --parser all \
  --languages $LANGUAGES \
  --optimizer_type "$OPTIMIZER_TYPE" \
  --loss_type "$LOSS_TYPE" \
  --focal_alpha $FOCAL_ALPHA \
  --focal_gamma $FOCAL_GAMMA \
  $AUTO_CLASS_WEIGHTS \
  --ema_momentum $EMA_MOMENTUM \
  --silc_weight $SILC_WEIGHT \
  --sigmoid_weight $SIGMOID_WEIGHT \
  --loca_weight $LOCA_WEIGHT \
  --classification_weight $CLASSIFICATION_WEIGHT \
  --best_model_metric "avg_lang_macro_f1" \
  --target_languages "English" "Mandarin" \
  --split_by_patient true

if [ $? -eq 0 ]; then
  echo "✅ ERM(AdamW) 훈련 완료"
else
  echo "❌ ERM(AdamW) 훈련 실패"
  exit 1
fi


