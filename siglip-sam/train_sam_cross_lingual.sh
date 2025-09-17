#!/bin/bash
# SigLIP-SAM Zero-shot Cross-Lingual 치매 진단 모델 훈련 (SAM 옵티마이저 사용)
# 다양한 언어 조합으로 진정한 Zero-shot 성능 평가
# 훈련: 소스 언어만 사용 / 검증&테스트: 타겟 언어만 사용 (완전 미학습)

echo "=== SigLIP-SAM Zero-shot Cross-Lingual 치매 진단 모델 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# SAM 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="focal"

# =================================
# Cross-lingual 언어 조합 설정
# =================================
# 사용 가능한 언어: English, Greek, Spanish, Mandarin

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
# 다른 조합을 사용하려면 아래 숫자를 변경하세요 (1, 2, 3, 4)
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
        echo "사용법: bash train_sam_cross_lingual.sh [1|2|3|4]"
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
OUTPUT_DIR="../modules/outputs/siglip-sam/CrossLingual_${EXPERIMENT_NAME}_SAM"

# SAM 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05
LOSS_TYPE="focal"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🌍 SAM Zero-shot Cross-Lingual 훈련 설정 (실험 $EXPERIMENT_NUM):"
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

echo "SAM Zero-shot Cross-Lingual 모델 훈련 시작 (실험 $EXPERIMENT_NUM)..."
echo "Zero-shot 실험: ${TRAIN_LANGUAGES[*]} → ${TEST_LANGUAGES[*]}"
echo "⚡ 타겟 언어는 훈련 시 전혀 사용하지 않음 (완전 Zero-shot)"
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
    echo "✅ SAM Zero-shot Cross-Lingual 모델 훈련이 성공적으로 완료되었습니다!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 실험 $EXPERIMENT_NUM: $EXPERIMENT_NAME"
    echo "🎯 훈련 언어 (소스): ${TRAIN_LANGUAGES[*]}"
    echo "🎯 타겟 언어 (Zero-shot): ${TEST_LANGUAGES[*]}"
    echo "🎯 SAM 옵티마이저로 훈련된 Zero-shot Cross-Lingual 모델"
    echo ""
    echo "📊 이 모델은 ${TRAIN_LANGUAGES[*]} 데이터로만 훈련되어"
    echo "   ${TEST_LANGUAGES[*]} 데이터에서 완전 Zero-shot 성능을 평가합니다."
    echo "   ⚡ 타겟 언어는 훈련/검증 시 전혀 보지 않아 진정한 Zero-shot!"
    echo "   SAM의 Sharpness-Aware Minimization으로 더 나은 일반화 기대!"
    echo ""
    echo "🔍 Zero-shot Cross-Lingual 분석 인사이트:"
    echo "   ✅ 완전 Zero-shot 성능 - 타겟 언어 미학습 상태에서의 성능"
    echo "   ✅ 언어별 일반화 능력 비교 - 어떤 언어가 다른 언어로 잘 전이되는지"
    echo "   ✅ SAM의 Zero-shot 효과 - 일반적인 옵티마이저 대비 성능"
    echo "   ✅ 언어 무관 특징 학습 정도 평가 (진정한 언어 독립성)"
    echo ""
    echo "📊 Zero-shot 결과 확인:"
    echo "   - 소스 언어 훈련 성능 vs 타겟 언어 Zero-shot 성능 비교"
    echo "   - wandb에서 Zero-shot Cross-lingual 메트릭 시각화"
    echo "   - 언어별 상세 분석 결과 자동 출력"
    echo "   - 검증/테스트 모두 타겟 언어로만 구성된 진정한 Zero-shot 평가"
    echo ""
    echo "🚀 다른 조합도 실행해보세요:"
    echo "   bash train_sam_cross_lingual.sh 1  # 영어+스페인어+만다린 → 그리스어"
    echo "   bash train_sam_cross_lingual.sh 2  # 영어+그리스어+만다린 → 스페인어"
    echo "   bash train_sam_cross_lingual.sh 3  # 영어+그리스어+스페인어 → 만다린"
    echo "   bash train_sam_cross_lingual.sh 4  # 그리스어+스페인어+만다린 → 영어"
else
    echo ""
    echo "❌ SAM Cross-Lingual 모델 훈련 중 오류가 발생했습니다."
    exit 1
fi
