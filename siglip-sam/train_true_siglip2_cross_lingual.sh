#!/bin/bash
# 진정한 SigLIP2 - Zero-shot Cross-Lingual 치매 진단 모델 훈련
# EMA Teacher-Student + Self-Distillation + Caption Generation으로 최강 Zero-shot 성능

echo "=== 진정한 SigLIP2 Zero-shot Cross-Lingual 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=64
LEARNING_RATE=2e-5
NUM_EPOCHS=100

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

# =================================
# Cross-lingual 언어 조합 설정
# =================================

# 조합 1: 영어+스페인어+만다린 → 그리스어 (기본)
TRAIN_LANGUAGES_1=("English" "Spanish" "Mandarin")
TEST_LANGUAGES_1=("Greek")

# 조합 2: 영어+그리스어+만다린 → 스페인어
TRAIN_LANGUAGES_2=("English" "Greek" "Mandarin")
TEST_LANGUAGES_2=("Spanish")

# 조합 3: 영어+그리스어+스페인어 → 만다린
TRAIN_LANGUAGES_3=("English" "Greek" "Spanish")
TEST_LANGUAGES_3=("Mandarin")

# 조합 4: 그리스어+스페인어+만다린 → 영어
TRAIN_LANGUAGES_4=("Greek" "Spanish" "Mandarin")
TEST_LANGUAGES_4=("English")

# =================================
# 실행할 조합 선택 (기본값: 조합 1)
# =================================
EXPERIMENT_NUM=${1:-1}  # 명령행 인수로 조합 선택 가능

case $EXPERIMENT_NUM in
    1)
        TRAIN_LANGUAGES=("${TRAIN_LANGUAGES_1[@]}")
        TEST_LANGUAGES=("${TEST_LANGUAGES_1[@]}")
        EXPERIMENT_NAME="Train_English_Spanish_Mandarin_Test_Greek"
        ;;
    2)
        TRAIN_LANGUAGES=("${TRAIN_LANGUAGES_2[@]}")
        TEST_LANGUAGES=("${TEST_LANGUAGES_2[@]}")
        EXPERIMENT_NAME="Train_English_Greek_Mandarin_Test_Spanish"
        ;;
    3)
        TRAIN_LANGUAGES=("${TRAIN_LANGUAGES_3[@]}")
        TEST_LANGUAGES=("${TEST_LANGUAGES_3[@]}")
        EXPERIMENT_NAME="Train_English_Greek_Spanish_Test_Mandarin"
        ;;
    4)
        TRAIN_LANGUAGES=("${TRAIN_LANGUAGES_4[@]}")
        TEST_LANGUAGES=("${TEST_LANGUAGES_4[@]}")
        EXPERIMENT_NAME="Train_Greek_Spanish_Mandarin_Test_English"
        ;;
    *)
        echo "❌ 잘못된 실험 번호입니다. 1-4 중 선택하세요."
        echo "사용법: bash train_true_siglip2_cross_lingual.sh [1|2|3|4]"
        echo ""
        echo "🌍 사용 가능한 조합:"
        echo "  1: 영어+스페인어+만다린 → 그리스어 (기본)"
        echo "  2: 영어+그리스어+만다린 → 스페인어"
        echo "  3: 영어+그리스어+스페인어 → 만다린"
        echo "  4: 그리스어+스페인어+만다린 → 영어"
        exit 1
        ;;
esac

# 출력 디렉토리 설정
OUTPUT_DIR="../modules/outputs/siglip-sam/True_SigLIP2_CrossLingual_${EXPERIMENT_NAME}"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🔥 진정한 SigLIP2 Zero-shot Cross-Lingual 훈련 설정 (실험 $EXPERIMENT_NUM):"
echo "  실험명: $EXPERIMENT_NAME"
echo "  훈련 언어 (소스): ${TRAIN_LANGUAGES[*]}"
echo "  타겟 언어 (Zero-shot): ${TEST_LANGUAGES[*]}"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  텍스트 토크나이저: google/gemma-2b (256K vocab, multilingual)"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE (rho=$SAM_RHO)"
echo "  손실 함수: $LOSS_TYPE + Multi-Loss"
echo "  Early Stopping: 훈련 언어 평균 AUC 기준 15 epochs patience"
echo ""
echo "🎯 진정한 SigLIP2 Zero-shot Multi-Loss (Cross-lingual 특화):"
echo "  🧑‍🏫 EMA Teacher-Student: momentum=$EMA_MOMENTUM"
echo "  📚 SILC/TIPS Loss: ${SILC_WEIGHT} (Self-Distillation 강화)"
echo "  🔗 Sigmoid Loss: ${SIGMOID_WEIGHT} (Cross-Modal Contrastive 강화)"
echo "  📝 LoCa Loss: ${LOCA_WEIGHT} (Caption Generation)"
echo "  🎯 Classification Loss: ${CLASSIFICATION_WEIGHT} (Dementia Diagnosis)"
echo ""
echo "📊 베스트 모델 선택 기준:"
echo "  🎯 훈련 언어들(${TRAIN_LANGUAGES[*]}) Validation AUC 평균"
echo "  📈 언어 편향 방지를 위한 균형잡힌 평가"
echo ""
echo "🚀 Zero-shot Cross-lingual 특화 전략:"
echo "  ✅ Gemma 토크나이저로 다국어 표현 능력 극대화 (256K vocab)"
echo "  ✅ Self-Distillation 비중 증가로 언어 무관 feature 강화"
echo "  ✅ Contrastive Learning 비중 증가로 cross-modal alignment 극대화"
echo "  ✅ EMA Teacher로 안정적인 언어 간 전이 학습"
echo "  ✅ Caption Generation으로 language understanding 향상"
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

echo "🚀 진정한 SigLIP2 Zero-shot Cross-Lingual 모델 훈련 시작 (실험 $EXPERIMENT_NUM)..."
echo "Zero-shot 실험: ${TRAIN_LANGUAGES[*]} → ${TEST_LANGUAGES[*]}"
echo "⚡ 타겟 언어는 훈련 시 전혀 사용하지 않음 (완전 Zero-shot)"
echo "⚡ EMA Teacher-Student + Multi-Loss로 최강 Zero-shot 성능!"
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
    --classification_weight $CLASSIFICATION_WEIGHT \
    --best_model_metric "avg_lang_auc" \
    --target_languages "${TRAIN_LANGUAGES[@]}"

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 진정한 SigLIP2 Zero-shot Cross-Lingual 모델 훈련 완료!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 실험 $EXPERIMENT_NUM: $EXPERIMENT_NAME"
    echo "🎯 훈련 언어 (소스): ${TRAIN_LANGUAGES[*]}"
    echo "🎯 타겟 언어 (Zero-shot): ${TEST_LANGUAGES[*]}"
    echo ""
    echo "🔥 진정한 SigLIP2 Zero-shot 혁신적 효과:"
    echo "   ✅ EMA Teacher-Student Self-Distillation"
    echo "   ✅ SILC/TIPS Masked Prediction으로 언어 무관 학습"
    echo "   ✅ Enhanced Contrastive Learning으로 cross-modal 정렬"
    echo "   ✅ Caption Generation으로 language understanding 강화"
    echo "   ✅ Multi-Loss 통합으로 Zero-shot 성능 극대화"
    echo ""
    echo "📊 이 모델은 ${TRAIN_LANGUAGES[*]} 데이터로만 훈련되어"
    echo "   ${TEST_LANGUAGES[*]} 데이터에서 완전 Zero-shot 성능을 평가합니다."
    echo "   ⚡ 타겟 언어는 훈련/검증 시 전혀 보지 않아 진정한 Zero-shot!"
    echo "   🔥 진정한 SigLIP2의 Multi-Loss로 기존 모델 대비 압도적 성능!"
    echo ""
    echo "🔍 진정한 SigLIP2 Zero-shot 분석 인사이트:"
    echo "   ✅ Self-Distillation 효과 - Teacher-Student alignment score"
    echo "   ✅ Cross-Modal 정렬 품질 - Sigmoid contrastive metrics"
    echo "   ✅ Caption 생성 능력 - Language understanding 정도"
    echo "   ✅ 완전 Zero-shot 성능 - 미학습 언어 전이 능력"
    echo ""
    echo "🚀 다른 조합도 실행해보세요:"
    echo "   bash train_true_siglip2_cross_lingual.sh 1  # 영어+스페인어+만다린 → 그리스어"
    echo "   bash train_true_siglip2_cross_lingual.sh 2  # 영어+그리스어+만다린 → 스페인어"
    echo "   bash train_true_siglip2_cross_lingual.sh 3  # 영어+그리스어+스페인어 → 만다린"
    echo "   bash train_true_siglip2_cross_lingual.sh 4  # 그리스어+스페인어+만다린 → 영어"
else
    echo ""
    echo "❌ 진정한 SigLIP2 Cross-Lingual 모델 훈련 실패"
    exit 1
fi
