#!/bin/bash
# SigLIP-SAM Cross-Lingual 치매 진단 모델 훈련
# 훈련: 영어, 스페인어, 만다린 / 테스트: 그리스어 (SAM 옵티마이저 사용)

echo "=== SigLIP-SAM Cross-Lingual 치매 진단 모델 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip-sam/CrossLingual_Train_English_Spanish_Mandarin_Test_Greek_SAM"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# Cross-lingual 언어 설정
TRAIN_LANGUAGES=("English" "Spanish" "Mandarin")
TEST_LANGUAGES=("Greek")

# SAM 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="focal"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🌍 SAM Cross-Lingual 훈련 설정:"
echo "  훈련 언어: ${TRAIN_LANGUAGES[*]}"
echo "  테스트 언어: ${TEST_LANGUAGES[*]}"
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

echo "SAM Cross-Lingual 모델 훈련 시작..."
echo "================================"

# 훈련 실행
$PYTHON_CMD trainer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "../modules/outputs/siglip-sam" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --parser cross_lingual \
    --train_languages "${TRAIN_LANGUAGES[@]}" \
    --test_languages "${TEST_LANGUAGES[@]}" \
    --optimizer_type "$OPTIMIZER_TYPE" \
    --sam_rho $SAM_RHO \
    --loss_type "$LOSS_TYPE"

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SAM Cross-Lingual 모델 훈련이 성공적으로 완료되었습니다!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 훈련 언어: ${TRAIN_LANGUAGES[*]}"
    echo "🎯 테스트 언어: ${TEST_LANGUAGES[*]}"
    echo "🎯 SAM 옵티마이저로 훈련된 Cross-Lingual 모델"
    echo ""
    echo "📊 이 모델은 ${TRAIN_LANGUAGES[*]} 데이터로 훈련되어"
    echo "   ${TEST_LANGUAGES[*]} 데이터에서 언어 간 일반화 성능을 평가합니다."
    echo "   SAM의 Sharpness-Aware Minimization으로 더 나은 일반화 기대!"
else
    echo ""
    echo "❌ SAM Cross-Lingual 모델 훈련 중 오류가 발생했습니다."
    exit 1
fi
