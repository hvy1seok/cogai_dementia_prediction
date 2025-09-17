#!/bin/bash
# 진정한 SigLIP2 - Cross-Lingual 훈련: 3개 언어 중 2개로 훈련, 1개로 Zero-shot
# 영어, 만다린, 스페인어 중 다양한 조합으로 Zero-shot 성능 평가
# EMA Teacher-Student + Multi-Loss로 최강 Zero-shot 성능

echo "=== 진정한 SigLIP2 - Cross-Lingual 훈련 시작 (2→1 Zero-shot) ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# =================================
# Cross-lingual 언어 조합 설정 (3개 언어 중 2→1)
# =================================
# 사용 가능한 언어: English, Mandarin, Spanish (그리스어 제외)

# 조합 1: 영어+만다린 → 스페인어 (기본)
TRAIN_LANGUAGES_1=("English" "Mandarin")
TEST_LANGUAGES_1=("Spanish")

# 조합 2: 영어+스페인어 → 만다린
TRAIN_LANGUAGES_2=("English" "Spanish")
TEST_LANGUAGES_2=("Mandarin")

# 조합 3: 만다린+스페인어 → 영어
TRAIN_LANGUAGES_3=("Mandarin" "Spanish")
TEST_LANGUAGES_3=("English")

# =================================
# 실행할 조합 선택 (기본값: 조합 1)
# =================================
EXPERIMENT_NUM=${1:-1}  # 명령행 인수로 조합 선택 가능

case $EXPERIMENT_NUM in
    1)
        TRAIN_LANGUAGES=("${TRAIN_LANGUAGES_1[@]}")
        TEST_LANGUAGES=("${TEST_LANGUAGES_1[@]}")
        EXPERIMENT_NAME="Train_English_Mandarin_Test_Spanish"
        ;;
    2)
        TRAIN_LANGUAGES=("${TRAIN_LANGUAGES_2[@]}")
        TEST_LANGUAGES=("${TEST_LANGUAGES_2[@]}")
        EXPERIMENT_NAME="Train_English_Spanish_Test_Mandarin"
        ;;
    3)
        TRAIN_LANGUAGES=("${TRAIN_LANGUAGES_3[@]}")
        TEST_LANGUAGES=("${TEST_LANGUAGES_3[@]}")
        EXPERIMENT_NAME="Train_Mandarin_Spanish_Test_English"
        ;;
    *)
        echo "❌ 잘못된 실험 번호입니다. 1-3 중 선택하세요."
        echo "사용법: bash train_true_siglip2_cross_lingual_3to1.sh [1|2|3]"
        echo ""
        echo "🌍 사용 가능한 조합 (그리스어 제외):"
        echo "  1: 영어+만다린 → 스페인어 (기본)"
        echo "  2: 영어+스페인어 → 만다린"
        echo "  3: 만다린+스페인어 → 영어"
        exit 1
        ;;
esac

# 출력 디렉토리 설정
OUTPUT_DIR="../modules/outputs/siglip/True_SigLIP2_CrossLingual_${EXPERIMENT_NAME}"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🌍 진정한 SigLIP2 - Zero-shot Cross-Lingual 훈련 설정 (실험 $EXPERIMENT_NUM):"
echo "  실험명: $EXPERIMENT_NAME"
echo "  훈련 언어 (소스): ${TRAIN_LANGUAGES[*]}"
echo "  타겟 언어 (Zero-shot): ${TEST_LANGUAGES[*]}"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo ""

# True SigLIP2 Multi-Loss 설정 (Cross-lingual 특화)
EMA_MOMENTUM=0.999
SILC_WEIGHT=0.3      # Self-Distillation 강화 (30%)
SIGMOID_WEIGHT=1.2   # Contrastive 강화 (120%)
LOCA_WEIGHT=0.8      # Caption 조정 (80%)
CLASSIFICATION_WEIGHT=1.0  # 유지 (100%)
MASK_RATIO=0.15
DECODER_HIDDEN_DIM=512
DECODER_NUM_HEADS=8
DECODER_NUM_LAYERS=6
VOCAB_SIZE=30522
MAX_CAPTION_LENGTH=77

echo "🔥 진정한 SigLIP2 Multi-Loss 구조 (Cross-lingual 특화):"
echo "  🧑‍🏫 EMA Teacher Momentum: ${EMA_MOMENTUM}"
echo "  📚 SILC/TIPS Loss: ${SILC_WEIGHT} (Self-Distillation 강화)"
echo "  🔗 Sigmoid Loss: ${SIGMOID_WEIGHT} (Cross-Modal Contrastive 강화)"
echo "  📝 LoCa Loss: ${LOCA_WEIGHT} (Caption Generation)"
echo "  🎯 Classification Loss: ${CLASSIFICATION_WEIGHT} (Dementia Diagnosis)"
echo ""
echo "📊 베스트 모델 선택 기준:"
echo "  🎯 훈련 언어들(${TRAIN_LANGUAGES[*]}) Validation AUC 평균"
echo "  📈 언어 편향 방지를 위한 균형잡힌 평가"
echo ""
echo "🔥 Zero-shot Cross-lingual 특화 설정:"
echo "  ✨ 2개 언어로 강력한 언어 무관 표현 학습"
echo "  ✨ Self-Distillation 비중 증가로 일반화 능력 극대화"
echo "  ✨ Contrastive Learning 강화로 Cross-modal alignment 향상"
echo "  ✨ 타겟 언어 완전 Zero-shot 평가 (훈련 시 전혀 미사용)"
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

# 훈련 실행
echo "🚀 진정한 SigLIP2 - Zero-shot Cross-Lingual 모델 훈련 시작..."
echo "⏳ Early Stopping: 평균 AUC 기준 10 epochs patience"
echo ""

$PYTHON_CMD true_siglip2_trainer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --parser "cross_lingual" \
    --train_languages "${TRAIN_LANGUAGES[@]}" \
    --test_languages "${TEST_LANGUAGES[@]}" \
    --loss_type "cross_entropy" \
    --optimizer_type "adamw" \
    --ema_momentum $EMA_MOMENTUM \
    --silc_weight $SILC_WEIGHT \
    --sigmoid_weight $SIGMOID_WEIGHT \
    --loca_weight $LOCA_WEIGHT \
    --classification_weight $CLASSIFICATION_WEIGHT \
    --mask_ratio $MASK_RATIO \
    --decoder_hidden_dim $DECODER_HIDDEN_DIM \
    --decoder_num_heads $DECODER_NUM_HEADS \
    --decoder_num_layers $DECODER_NUM_LAYERS \
    --vocab_size $VOCAB_SIZE \
    --max_caption_length $MAX_CAPTION_LENGTH \
    --best_model_metric "avg_lang_auc" \
    --target_languages "${TRAIN_LANGUAGES[@]}"

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 진정한 SigLIP2 - Zero-shot Cross-Lingual 모델 훈련 성공!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "📊 이 모델은 ${TRAIN_LANGUAGES[*]} 데이터로만 훈련되어"
    echo "   ${TEST_LANGUAGES[*]} 데이터에서 완전 Zero-shot 성능을 평가합니다."
    echo "   ⚡ 타겟 언어는 훈련/검증 시 전혀 보지 않아 진정한 Zero-shot!"
    echo "   ⚡ 진정한 SigLIP2의 모든 이점으로 최강 Zero-shot 성능!"
    echo ""
    echo "🔍 Zero-shot Cross-Lingual 분석 인사이트:"
    echo "   ✅ 완전 Zero-shot 성능 - 타겟 언어 미학습 상태에서의 성능"
    echo "   ✅ EMA Teacher vs Student Zero-shot 성능 비교"
    echo "   ✅ Multi-Loss components의 Zero-shot 기여도"
    echo "   ✅ 2개 언어 → 1개 언어 전이 학습 품질 평가"
    echo ""
    echo "📊 결과 확인:"
    echo "   - 소스 언어 훈련 성능 vs 타겟 언어 Zero-shot 성능 비교"
    echo "   - EMA Teacher-Student 각각의 Zero-shot 성능"
    echo "   - Multi-Loss 실시간 모니터링 및 기여도 분석"
    echo "   - Cross-modal alignment의 Zero-shot 전이 효과"
    echo ""
    echo "🎯 기대 효과:"
    echo "   ✨ 2개 언어의 강력한 언어 무관 표현으로 최고 Zero-shot 성능"
    echo "   ✨ 진정한 SigLIP2 아키텍처의 모든 이점 활용"
    echo "   ✨ 타겟 언어에서 예상되는 뛰어난 Zero-shot 성능"
    echo "   ✨ 다양한 언어 조합의 전이 능력 비교 분석"
    echo ""
    echo "🚀 다른 조합도 실행해보세요:"
    echo "   bash train_true_siglip2_cross_lingual_3to1.sh 1  # 영어+만다린 → 스페인어"
    echo "   bash train_true_siglip2_cross_lingual_3to1.sh 2  # 영어+스페인어 → 만다린"
    echo "   bash train_true_siglip2_cross_lingual_3to1.sh 3  # 만다린+스페인어 → 영어"
else
    echo ""
    echo "❌ 진정한 SigLIP2 Cross-Lingual 모델 훈련 실패"
    exit 1
fi
