#!/bin/bash
# SigLIP2-SAM Contrastive Learning 치매 진단 모델 훈련
# True SigLIP2 스타일 contrastive learning with sigmoid matching

echo "=== SigLIP2-SAM Contrastive Learning 치매 진단 모델 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip-sam/SigLIP2_Contrastive_SAM"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100
LANGUAGES="English Greek Spanish Mandarin"

# SAM + Contrastive Learning 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
AUTO_CLASS_WEIGHTS="--auto_class_weights"

# SigLIP2 Contrastive Learning 설정
USE_CONTRASTIVE="--use_contrastive"
CONTRASTIVE_WEIGHT=0.5
CONTRASTIVE_TEMPERATURE=0.07

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🔗 SigLIP2 Contrastive Learning + SAM 훈련 설정:"
echo "  언어: $LANGUAGES"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE (rho=$SAM_RHO)"
echo "  손실 함수: $LOSS_TYPE (alpha=$FOCAL_ALPHA, gamma=$FOCAL_GAMMA)"
echo "  클래스 가중치: $AUTO_CLASS_WEIGHTS (자동 불균형 보정)"
echo "  Early Stopping: Validation AUC 기준 15 epochs patience"
echo ""
echo "🎯 SigLIP2 Contrastive Learning:"
echo "  활성화: $USE_CONTRASTIVE"
echo "  가중치: $CONTRASTIVE_WEIGHT (Classification vs Contrastive)"
echo "  온도: $CONTRASTIVE_TEMPERATURE"
echo "  매칭 방식: Sigmoid (SigLIP2 스타일)"
echo "  Positive pairs: Same patient audio-text"
echo "  Negative pairs: Different patient combinations"
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

echo "🔗 SigLIP2 Contrastive Learning + SAM 모델 훈련 시작..."
echo "⚡ In-batch contrastive learning으로 cross-modal alignment 최적화"
echo "⚡ Same-patient audio-text pairs를 positive로, 다른 조합을 negative로 학습"
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
    echo "✅ SigLIP2 Contrastive Learning + SAM 모델 훈련이 성공적으로 완료되었습니다!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 훈련된 언어: $LANGUAGES"
    echo "🔗 SigLIP2 Contrastive Learning + SAM으로 훈련된 다국어 치매 진단 모델"
    echo "   - True SigLIP2 스타일: Sigmoid matching contrastive learning"
    echo "   - Cross-modal alignment: Audio-text representation 공동 학습"
    echo "   - SAM optimizer: Sharpness-Aware Minimization으로 더 나은 일반화"
    echo "   - Patient-aware positive pairs: 같은 환자의 audio-text를 가까이 배치"
    echo ""
    echo "🔍 SigLIP2 Contrastive Learning 분석 인사이트:"
    echo "   ✅ Cross-modal alignment score - 오디오와 텍스트 표현의 정렬 정도"
    echo "   ✅ Positive vs Negative similarity - 같은/다른 환자 간 유사도 비교"
    echo "   ✅ In-batch contrastive learning - 배치 내 모든 조합으로 학습"
    echo "   ✅ Language-agnostic representation - 언어 무관 특징 학습 강화"
    echo ""
    echo "📊 결과 확인:"
    echo "   - 콘솔 출력에서 Contrastive Learning 메트릭 확인"
    echo "     * Alignment Score: positive - negative similarity"
    echo "     * Positive Similarity: 같은 환자 audio-text 유사도"
    echo "     * Negative Similarity: 다른 환자 조합 유사도"
    echo "   - wandb 대시보드에서 실시간 contrastive 메트릭 시각화"
    echo "   - Classification vs Contrastive loss 분리 추적"
    echo "   - 언어별 cross-modal 전이 성능 분석"
    echo ""
    echo "🚀 기대 효과:"
    echo "   ✨ 더 강력한 cross-lingual 전이 능력"
    echo "   ✨ Cross-modal representation의 semantic alignment 향상"
    echo "   ✨ Zero-shot 성능 개선 (미학습 언어에서도 더 나은 성능)"
    echo "   ✨ Multimodal fusion의 질적 향상"
else
    echo ""
    echo "❌ SigLIP2 Contrastive Learning + SAM 모델 훈련 중 오류가 발생했습니다."
    exit 1
fi
