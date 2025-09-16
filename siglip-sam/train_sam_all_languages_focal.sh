#!/bin/bash
# SigLIP-SAM 모든 언어 치매 진단 모델 훈련 (SAM + Focal Loss)
# 언어별 성능 분석 기능 포함

echo "=== SigLIP-SAM 다국어 치매 진단 모델 훈련 (Focal Loss) 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip-sam/All_Languages_SAM_Focal"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100
LANGUAGES="English Greek Spanish Mandarin"

# SAM + Focal Loss 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🎯 SAM + Focal Loss 다국어 훈련 설정:"
echo "  언어: $LANGUAGES"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE (rho=$SAM_RHO)"
echo "  손실 함수: $LOSS_TYPE (alpha=$FOCAL_ALPHA, gamma=$FOCAL_GAMMA)"
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

echo "SAM + Focal Loss 다국어 통합 모델 훈련 시작..."
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
    --languages $LANGUAGES \
    --optimizer_type "$OPTIMIZER_TYPE" \
    --sam_rho $SAM_RHO \
    --loss_type "$LOSS_TYPE" \
    --focal_alpha $FOCAL_ALPHA \
    --focal_gamma $FOCAL_GAMMA

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SAM + Focal Loss 다국어 통합 모델 훈련이 성공적으로 완료되었습니다!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 훈련된 언어: $LANGUAGES"
    echo "🎯 SAM + Focal Loss로 훈련된 다국어 치매 진단 모델"
    echo "   - Sharpness-Aware Minimization으로 더 나은 일반화 성능"
    echo "   - Focal Loss로 불균형 데이터 처리 최적화"
    echo "   - 언어별 성능 분석 결과 자동 출력"
    echo ""
    echo "🔍 분석 가능한 인사이트:"
    echo "   ✅ 언어별 성능 비교 - 어떤 언어에서 모델이 더 잘 작동하는지 확인"
    echo "   ✅ 데이터 분포 확인 - 언어별 샘플 수 균형 및 정상/치매 비율 분석"
    echo "   ✅ Threshold 효과 분석 - 최적 threshold의 언어별 효과성"
    echo "   ✅ Cross-lingual 일반화 성능 평가"
    echo ""
    echo "📊 결과 확인:"
    echo "   - 콘솔 출력에서 언어별 상세 분석 결과 확인"
    echo "   - wandb 대시보드에서 언어별 메트릭 시각화"
    echo "   - ROC 곡선 및 Confusion Matrix 자동 생성"
else
    echo ""
    echo "❌ SAM + Focal Loss 다국어 통합 모델 훈련 중 오류가 발생했습니다."
    exit 1
fi
