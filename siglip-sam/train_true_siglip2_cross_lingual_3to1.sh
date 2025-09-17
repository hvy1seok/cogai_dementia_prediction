#!/bin/bash
# 진정한 SigLIP2 - Cross-Lingual 훈련: 3개 언어 → 1개 언어 Zero-shot
# 영어+만다린+스페인어로 훈련 → 그리스어 Zero-shot 평가
# EMA Teacher-Student + Multi-Loss로 최강 Zero-shot 성능

echo "=== 진정한 SigLIP2 - Cross-Lingual 훈련 시작 (3→1 Zero-shot) ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# Cross-lingual 언어 설정
TRAIN_LANGUAGES=("English" "Mandarin" "Spanish")
TEST_LANGUAGES=("Greek")
EXPERIMENT_NAME="Train_English_Mandarin_Spanish_Test_Greek"

# 출력 디렉토리 설정
OUTPUT_DIR="../modules/outputs/siglip-sam/True_SigLIP2_CrossLingual_${EXPERIMENT_NAME}"

# SAM + Focal Loss 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
AUTO_CLASS_WEIGHTS="--auto_class_weights"

# True SigLIP2 Multi-Loss 가중치 (Cross-lingual 특화)
EMA_MOMENTUM=0.999
SILC_WEIGHT=0.3      # Self-Distillation 비중 증가 (30%)
SIGMOID_WEIGHT=1.2   # Contrastive 비중 증가 (120%)
LOCA_WEIGHT=0.8      # Caption 비중 조정 (80%)
CLASSIFICATION_WEIGHT=1.0  # Classification 유지 (100%)

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🌍 진정한 SigLIP2 - Zero-shot Cross-Lingual 훈련 설정:"
echo "  실험명: $EXPERIMENT_NAME"
echo "  훈련 언어 (소스): ${TRAIN_LANGUAGES[*]}"
echo "  타겟 언어 (Zero-shot): ${TEST_LANGUAGES[*]}"
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
echo "🎯 진정한 SigLIP2 Multi-Loss (Cross-lingual 특화):"
echo "  🧑‍🏫 EMA Teacher-Student: momentum=$EMA_MOMENTUM"
echo "  📚 SILC/TIPS Loss: ${SILC_WEIGHT} (Self-Distillation 강화)"
echo "  🔗 Sigmoid Loss: ${SIGMOID_WEIGHT} (Cross-Modal Contrastive 강화)"
echo "  📝 LoCa Loss: ${LOCA_WEIGHT} (Caption Generation)"
echo "  🎯 Classification Loss: ${CLASSIFICATION_WEIGHT} (Dementia Diagnosis)"
echo ""
echo "🔥 Zero-shot Cross-lingual 특화 설정:"
echo "  ✨ 3개 언어로 강력한 언어 무관 표현 학습"
echo "  ✨ Self-Distillation 비중 증가로 일반화 능력 극대화"
echo "  ✨ Contrastive Learning 강화로 Cross-modal alignment 향상"
echo "  ✨ 그리스어 완전 Zero-shot 평가 (훈련 시 전혀 미사용)"
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

echo "🚀 진정한 SigLIP2 Zero-shot Cross-Lingual 모델 훈련 시작..."
echo "Zero-shot 실험: ${TRAIN_LANGUAGES[*]} → ${TEST_LANGUAGES[*]}"
echo "⚡ 타겟 언어는 훈련 시 전혀 사용하지 않음 (완전 Zero-shot)"
echo "⚡ EMA Teacher-Student + Multi-Loss로 최강 Zero-shot 성능 달성"
echo "================================"

# 훈련 실행
$PYTHON_CMD true_siglip2_trainer.py \
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
    echo "✅ 진정한 SigLIP2 Zero-shot Cross-Lingual 모델 훈련 완료!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 실험: $EXPERIMENT_NAME"
    echo "🎯 훈련 언어 (소스): ${TRAIN_LANGUAGES[*]}"
    echo "🎯 타겟 언어 (Zero-shot): ${TEST_LANGUAGES[*]}"
    echo ""
    echo "🔥 진정한 SigLIP2 Zero-shot 효과:"
    echo "   ✅ EMA Teacher-Student로 안정적 Zero-shot 전이"
    echo "   ✅ Enhanced Self-Distillation으로 일반화 능력 극대화"
    echo "   ✅ Strengthened Contrastive Learning으로 Cross-modal alignment 향상"
    echo "   ✅ Multi-Loss 통합으로 Robust representation 학습"
    echo ""
    echo "📊 이 모델은 ${TRAIN_LANGUAGES[*]} 데이터로만 훈련되어"
    echo "   ${TEST_LANGUAGES[*]} 데이터에서 완전 Zero-shot 성능을 평가합니다."
    echo "   ⚡ 타겟 언어는 훈련/검증 시 전혀 보지 않아 진정한 Zero-shot!"
    echo "   ⚡ 진정한 SigLIP2의 모든 이점으로 최강 Zero-shot 성능!"
    echo ""
    echo "🔍 Zero-shot Cross-Lingual 분석 인사이트:"
    echo "   ✅ 완전 Zero-shot 성능 - 그리스어 미학습 상태에서의 성능"
    echo "   ✅ EMA Teacher vs Student Zero-shot 성능 비교"
    echo "   ✅ Multi-Loss components의 Zero-shot 기여도"
    echo "   ✅ 3개 언어 → 1개 언어 전이 학습 품질 평가"
    echo ""
    echo "📊 결과 확인:"
    echo "   - 소스 언어 훈련 성능 vs 타겟 언어 Zero-shot 성능 비교"
    echo "   - EMA Teacher-Student 각각의 Zero-shot 성능"
    echo "   - Multi-Loss 실시간 모니터링 및 기여도 분석"
    echo "   - Cross-modal alignment의 Zero-shot 전이 효과"
    echo ""
    echo "🎯 기대 효과:"
    echo "   ✨ 3개 언어의 강력한 언어 무관 표현으로 최고 Zero-shot 성능"
    echo "   ✨ 진정한 SigLIP2 아키텍처의 모든 이점 활용"
    echo "   ✨ 그리스어에서 예상되는 뛰어난 Zero-shot 성능"
    echo "   ✨ 다른 언어로의 확장 가능성 검증"
else
    echo ""
    echo "❌ 진정한 SigLIP2 Cross-Lingual 모델 훈련 실패"
    exit 1
fi
