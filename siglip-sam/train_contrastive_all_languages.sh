#!/bin/bash
# SigLIP2 Contrastive Learning - 전체 언어 통합 훈련
# 모든 언어를 함께 학습하여 언어 무관 표현 학습 및 cross-modal alignment 최적화

echo "=== SigLIP2 Contrastive Learning - 전체 언어 통합 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip-sam/Contrastive_All_Languages"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=128
LEARNING_RATE=2e-5
NUM_EPOCHS=100
LANGUAGES="English Greek Spanish Mandarin"

# SAM + Focal Loss + Contrastive Learning 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
AUTO_CLASS_WEIGHTS="--auto_class_weights"

# SigLIP2 Contrastive Learning 설정
USE_CONTRASTIVE="--use_contrastive"
CONTRASTIVE_WEIGHT=0.5  # Classification 50% + Contrastive 50%
CONTRASTIVE_TEMPERATURE=0.07

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🌍 SigLIP2 Contrastive Learning - 전체 언어 통합 훈련 설정:"
echo "  훈련 언어: $LANGUAGES (모든 언어 통합)"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE (rho=$SAM_RHO)"
echo "  손실 함수: $LOSS_TYPE + Contrastive"
echo "  Early Stopping: Validation AUC 기준 15 epochs patience"
echo ""
echo "🔗 SigLIP2 Contrastive Learning 특징:"
echo "  ✨ In-batch contrastive learning으로 cross-modal alignment"
echo "  ✨ Same-patient audio-text pairs → positive (유사도 증가)"
echo "  ✨ Different-patient combinations → negative (유사도 감소)"
echo "  ✨ Sigmoid matching (SigLIP2 스타일, CLIP의 softmax 대신)"
echo "  ✨ 언어 무관 representation 학습 강화"
echo "  가중치: Classification $((100-$(echo "$CONTRASTIVE_WEIGHT * 100" | bc -l | cut -d. -f1)))% + Contrastive $(echo "$CONTRASTIVE_WEIGHT * 100" | bc -l | cut -d. -f1)%"
echo "  온도: $CONTRASTIVE_TEMPERATURE"
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

echo "🚀 SigLIP2 Contrastive Learning 전체 언어 통합 모델 훈련 시작..."
echo "⚡ 4개 언어의 cross-modal representation을 공동 표현 공간으로 정렬"
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
    --focal_gamma $FOCAL_GAMMA \
    $AUTO_CLASS_WEIGHTS \
    $USE_CONTRASTIVE \
    --contrastive_weight $CONTRASTIVE_WEIGHT \
    --contrastive_temperature $CONTRASTIVE_TEMPERATURE

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SigLIP2 Contrastive Learning 전체 언어 통합 모델 훈련 완료!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 훈련 완료된 언어: $LANGUAGES"
    echo "🔗 SigLIP2 Contrastive Learning 효과:"
    echo "   ✅ Cross-modal alignment - 오디오와 텍스트의 의미적 정렬"
    echo "   ✅ Language-agnostic representation - 언어 무관 특징 학습"
    echo "   ✅ Enhanced cross-lingual transfer - 언어 간 전이 능력 향상"
    echo "   ✅ Improved multimodal fusion - 멀티모달 융합 품질 개선"
    echo ""
    echo "📊 결과 분석 포인트:"
    echo "   🔍 Alignment Score: Positive - Negative similarity 차이"
    echo "   🔍 Positive Similarity: 같은 환자 audio-text 평균 유사도"
    echo "   🔍 Negative Similarity: 다른 환자 조합 평균 유사도"
    echo "   🔍 언어별 성능: 각 언어에서의 분류 및 정렬 성능"
    echo ""
    echo "🎯 이 모델의 장점:"
    echo "   ✨ 모든 언어에서 균형잡힌 성능"
    echo "   ✨ Cross-lingual 일반화 능력 극대화"
    echo "   ✨ Zero-shot 성능 기반 마련"
    echo "   ✨ Robust multimodal representation"
else
    echo ""
    echo "❌ SigLIP2 Contrastive Learning 전체 언어 통합 모델 훈련 실패"
    exit 1
fi
