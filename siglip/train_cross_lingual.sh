#!/bin/bash
# SigLIP2 Cross-Lingual 치매 진단 모델 훈련
# 훈련: 영어, 스페인어, 만다린 / 테스트: 그리스어

echo "=== SigLIP2 Cross-Lingual 치매 진단 모델 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip/CrossLingual_Train_English_Spanish_Mandarin_Test_Greek"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# Cross-lingual 언어 설정
TRAIN_LANGUAGES=("English" "Spanish" "Mandarin")
TEST_LANGUAGES=("Greek")

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🌍 Cross-Lingual 훈련 설정:"
echo "  훈련 언어: ${TRAIN_LANGUAGES[*]}"
echo "  테스트 언어: ${TEST_LANGUAGES[*]}"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
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

# 데이터 파서 테스트 자동 실행
echo "데이터 파서 테스트 실행 중..."
$PYTHON_CMD test_parser.py
echo ""

echo "Cross-Lingual 모델 훈련 시작..."
echo "================================"

# 훈련 실행
$PYTHON_CMD trainer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "../modules/outputs/siglip" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --parser cross_lingual \
    --train_languages "${TRAIN_LANGUAGES[@]}" \
    --test_languages "${TEST_LANGUAGES[@]}" \
    --loss_type cross_entropy \
    --optimizer_type adamw

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Cross-Lingual 모델 훈련이 성공적으로 완료되었습니다!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 훈련 언어: ${TRAIN_LANGUAGES[*]}"
    echo "🎯 테스트 언어: ${TEST_LANGUAGES[*]}"
    echo ""
    echo "📊 이 모델은 ${TRAIN_LANGUAGES[*]} 데이터로 훈련되어"
    echo "   ${TEST_LANGUAGES[*]} 데이터에서 언어 간 일반화 성능을 평가합니다."
else
    echo ""
    echo "❌ Cross-Lingual 모델 훈련 중 오류가 발생했습니다."
    exit 1
fi
