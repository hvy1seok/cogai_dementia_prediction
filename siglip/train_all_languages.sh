#!/bin/bash
# SigLIP2 모든 언어 치매 진단 모델 훈련

echo "=== SigLIP2 다국어 치매 진단 모델 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip/All_Languages"
MODEL_NAME="google/siglip2-base-patch16-224"
BATCH_SIZE=8
LEARNING_RATE=2e-5
NUM_EPOCHS=10
LANGUAGES="English Greek Spanish Mandarin"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "훈련 설정:"
echo "  언어: $LANGUAGES"
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

echo "다국어 통합 모델 훈련 시작..."
echo "================================"

# 훈련 실행
$PYTHON_CMD trainer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --parser all \
    --languages $LANGUAGES

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 다국어 통합 모델 훈련이 성공적으로 완료되었습니다!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 훈련된 언어: $LANGUAGES"
else
    echo ""
    echo "❌ 다국어 통합 모델 훈련 중 오류가 발생했습니다."
    exit 1
fi
