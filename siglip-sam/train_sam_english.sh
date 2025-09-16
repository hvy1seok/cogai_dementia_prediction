#!/bin/bash
# SigLIP-SAM 영어 치매 진단 모델 훈련 (SAM 옵티마이저 사용)

echo "=== SigLIP-SAM 영어 치매 진단 모델 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 설정
DATA_DIR="../training_dset"
OUTPUT_DIR="../modules/outputs/siglip-sam/English_SAM"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100
LANGUAGE="English"

# SAM 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="cross_entropy"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🎯 SAM 훈련 설정:"
echo "  언어: $LANGUAGE"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE (rho=$SAM_RHO)"
echo "  손실 함수: $LOSS_TYPE"
echo ""

# Python 명령어 확인
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python을 찾을 수 없습니다. Python 3.8+ 설치가 필요합니다."
    exit 1
fi

echo "Python 명령어: $PYTHON_CMD"
echo ""

echo "SAM 영어 모델 훈련 시작..."
echo "================================"

# 훈련 실행
$PYTHON_CMD trainer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --parser "$LANGUAGE" \
    --optimizer_type "$OPTIMIZER_TYPE" \
    --sam_rho $SAM_RHO \
    --loss_type "$LOSS_TYPE"

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SAM 영어 모델 훈련이 성공적으로 완료되었습니다!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🎯 SAM 옵티마이저로 훈련된 영어 치매 진단 모델"
    echo "   - Sharpness-Aware Minimization으로 더 나은 일반화 성능 기대"
    echo "   - rho=${SAM_RHO}로 설정된 SAM 파라미터"
else
    echo ""
    echo "❌ SAM 영어 모델 훈련 중 오류가 발생했습니다."
    exit 1
fi
