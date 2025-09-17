#!/bin/bash
# 진정한 SigLIP2 - 3개 언어 통합 훈련 (영어, 만다린, 스페인어)
# EMA Teacher-Student + Self-Distillation + Caption Generation으로 최강 성능

echo "=== 진정한 SigLIP2 - 3개 언어 통합 훈련 시작 (영어, 만다린, 스페인어) ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip-sam/True_SigLIP2_3Languages_EN_MN_ES"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# 3개 언어 설정 (그리스어 제외)
LANGUAGES="English Mandarin Spanish"

# SAM + Focal Loss 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
AUTO_CLASS_WEIGHTS="--auto_class_weights"

# True SigLIP2 Multi-Loss 가중치
EMA_MOMENTUM=0.999
SILC_WEIGHT=0.2      # SILC/TIPS Loss (20%)
SIGMOID_WEIGHT=1.0   # Sigmoid Contrastive Loss (100%)
LOCA_WEIGHT=1.0      # LoCa Caption Loss (100%)
CLASSIFICATION_WEIGHT=1.0  # Classification Loss (100%)

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🔥 진정한 SigLIP2 - 3개 언어 통합 훈련 설정:"
echo "  훈련 언어: $LANGUAGES (그리스어 제외)"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE (rho=$SAM_RHO)"
echo "  손실 함수: $LOSS_TYPE + Multi-Loss"
echo "  Early Stopping: Validation AUC 기준 15 epochs patience"
echo ""
echo "🎯 진정한 SigLIP2 Multi-Loss 구조:"
echo "  🧑‍🏫 EMA Teacher-Student: momentum=$EMA_MOMENTUM"
echo "  📚 SILC/TIPS Loss: ${SILC_WEIGHT} (Self-Distillation + Masked Prediction)"
echo "  🔗 Sigmoid Loss: ${SIGMOID_WEIGHT} (Cross-Modal Contrastive)"
echo "  📝 LoCa Loss: ${LOCA_WEIGHT} (Caption + Dense Caption + Referring)"
echo "  🎯 Classification Loss: ${CLASSIFICATION_WEIGHT} (Dementia Diagnosis)"
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

echo "🚀 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 시작..."
echo "⚡ EMA Teacher-Student + Multi-Loss로 최고 성능 달성!"
echo "⚡ 영어, 만다린, 스페인어의 강력한 언어 무관 표현 학습!"
echo "================================"

# 훈련 실행
$PYTHON_CMD true_siglip2_trainer.py \
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
    --ema_momentum $EMA_MOMENTUM \
    --silc_weight $SILC_WEIGHT \
    --sigmoid_weight $SIGMOID_WEIGHT \
    --loca_weight $LOCA_WEIGHT \
    --classification_weight $CLASSIFICATION_WEIGHT

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 완료!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 훈련 완료된 언어: $LANGUAGES"
    echo "🔥 진정한 SigLIP2 아키텍처로 훈련된 3개 언어 치매 진단 모델"
    echo "   - EMA Teacher-Student로 안정적 학습"
    echo "   - SILC/TIPS Loss로 Self-Distillation 강화"
    echo "   - Sigmoid Contrastive로 Cross-Modal Alignment"
    echo "   - LoCa Loss로 Caption Generation 능력"
    echo "   - Multi-Loss 통합으로 최고 성능 달성"
    echo ""
    echo "🔍 3개 언어 모델 분석 인사이트:"
    echo "   ✅ 영어-만다린-스페인어 간 Cross-lingual 성능 비교"
    echo "   ✅ 그리스어 제외로 더 집중된 언어 무관 학습"
    echo "   ✅ EMA Teacher의 안정적 지식 전달 효과"
    echo "   ✅ Multi-Loss 각 구성요소의 기여도 분석"
    echo ""
    echo "📊 결과 확인:"
    echo "   - 언어별 상세 성능 분석 (영어, 만다린, 스페인어)"
    echo "   - Multi-Loss components 실시간 모니터링"
    echo "   - EMA Teacher vs Student 성능 비교"
    echo "   - Cross-modal alignment 품질 평가"
    echo ""
    echo "🎯 이 모델의 장점:"
    echo "   ✨ 3개 주요 언어에서 균형잡힌 최고 성능"
    echo "   ✨ 진정한 SigLIP2 아키텍처의 모든 이점 활용"
    echo "   ✨ 그리스어 Zero-shot 평가를 위한 최적 기반 모델"
    echo "   ✨ Robust multimodal representation 학습"
else
    echo ""
    echo "❌ 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 실패"
    exit 1
fi
