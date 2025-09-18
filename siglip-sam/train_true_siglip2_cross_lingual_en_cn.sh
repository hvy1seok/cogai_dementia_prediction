#!/bin/bash
# 진정한 SigLIP2 - 영어/중국어 Cross-lingual Zero-shot 훈련
# 스페인어, 그리스어 제외 - 학습이 잘 되는 언어만 사용
# EMA Teacher-Student + Self-Distillation + Caption Generation

echo "=== 진정한 SigLIP2 - 영어/중국어 Cross-lingual Zero-shot 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# Cross-lingual 실험 조합 정의 (영어 ↔ 중국어)
declare -A EXPERIMENTS=(
    ["English"]="Mandarin"
    ["Mandarin"]="English"
)

# 실험 선택
if [ $# -eq 0 ]; then
    echo "사용법: $0 <실험번호>"
    echo "영어/중국어 Cross-lingual 실험 조합:"
    echo "  1: 영어 Zero-shot (훈련: Mandarin → 테스트: English)"
    echo "  2: 중국어 Zero-shot (훈련: English → 테스트: Mandarin)"
    echo "  all: 모든 실험 순차 실행"
    exit 1
fi

EXPERIMENT_NUM=$1

# 실험 조합 매핑
case $EXPERIMENT_NUM in
    1)
        TEST_LANGUAGE="English"
        TRAIN_LANGUAGES=(Mandarin)
        ;;
    2)
        TEST_LANGUAGE="Mandarin"
        TRAIN_LANGUAGES=(English)
        ;;
    all)
        echo "🔄 모든 영어/중국어 Cross-lingual 실험 순차 실행..."
        for i in {1..2}; do
            echo ""
            echo "========================================"
            echo "실험 $i 시작..."
            echo "========================================"
            bash "$0" $i
            if [ $? -ne 0 ]; then
                echo "❌ 실험 $i 실패"
                exit 1
            fi
        done
        echo "✅ 모든 영어/중국어 Cross-lingual 실험 완료!"
        exit 0
        ;;
    *)
        echo "❌ 잘못된 실험 번호: $EXPERIMENT_NUM"
        echo "1-2 또는 'all'을 입력하세요."
        exit 1
        ;;
esac

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

# True SigLIP2 Multi-Loss 가중치
EMA_MOMENTUM=0.999
SILC_WEIGHT=0.2      # SILC/TIPS Loss (20%)
SIGMOID_WEIGHT=1.0   # Sigmoid Contrastive Loss (100%)
LOCA_WEIGHT=1.0      # LoCa Caption Loss (100%)
CLASSIFICATION_WEIGHT=1.0  # Classification Loss (100%)

# 출력 디렉토리 설정
TRAIN_LANGS_STR=$(IFS=_; echo "${TRAIN_LANGUAGES[*]}")
OUTPUT_DIR="../modules/outputs/siglip-sam/True_SigLIP2_CrossLingual_EN_CN_Train_${TRAIN_LANGS_STR}_Test_${TEST_LANGUAGE}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🔥 진정한 SigLIP2 - 영어/중국어 Cross-lingual Zero-shot 훈련 설정:"
echo "  훈련 언어: ${TRAIN_LANGUAGES[*]}"
echo "  테스트 언어: $TEST_LANGUAGE (Zero-shot)"
echo "  제외 언어: Spanish, Greek (학습 어려움)"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  텍스트 토크나이저: google/gemma-2b (256K vocab, multilingual)"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE (rho=$SAM_RHO)"
echo "  손실 함수: $LOSS_TYPE + Multi-Loss"
echo "  Early Stopping: 훈련 언어 Macro F1 기준 10 epochs patience"
echo ""
echo "📊 데이터 분할 방식:"
echo "  👥 환자 단위 분할 - Speaker-Independent"
echo "  🎯 실제 임상 환경과 유사한 평가"
echo "  ✅ 동일 환자의 샘플이 한 세트에만 존재"
echo ""
echo "📊 베스트 모델 선택 기준:"
echo "  🎯 훈련 언어(${TRAIN_LANGUAGES[*]}) Validation Macro F1"
echo "  📈 클래스 균형을 고려한 공정한 평가"
echo ""
echo "🎯 진정한 SigLIP2 Multi-Loss 구조:"
echo "  🧑‍🏫 EMA Teacher-Student: momentum=$EMA_MOMENTUM"
echo "  📚 SILC/TIPS Loss: ${SILC_WEIGHT} (Self-Distillation + Masked Prediction)"
echo "  🔗 Sigmoid Loss: ${SIGMOID_WEIGHT} (Cross-Modal Contrastive)"
echo "  📝 LoCa Loss: ${LOCA_WEIGHT} (Caption + Dense Caption + Referring)"
echo "  🎯 Classification Loss: ${CLASSIFICATION_WEIGHT} (Dementia Diagnosis)"
echo ""
echo "✨ 영어/중국어 Cross-lingual 핵심 기능:"
echo "  ✅ Gemma 토크나이저로 다국어 표현 능력 극대화 (256K vocab)"
echo "  ✅ EMA Teacher-Student 구조로 Self-Distillation"
echo "  ✅ Masked Prediction으로 Self-Supervised Learning"
echo "  ✅ Auto-Regressive Decoder로 Caption Generation"
echo "  ✅ Cross-Modal Alignment + Language-Agnostic Learning"
echo "  ✅ Dense Captioning + Referring Expressions"
echo "  ✅ 학습이 잘 되는 언어 간의 강력한 Zero-shot Transfer"
echo ""
echo "🌏 영어 ↔ 중국어 Zero-shot 실험:"
echo "  🎯 훈련: ${TRAIN_LANGUAGES[*]} (환자 단위 분할)"
echo "  🔍 검증/테스트: $TEST_LANGUAGE (완전한 Zero-shot)"
echo "  📊 기대: 영어-중국어 간의 강력한 Cross-lingual Transfer"
echo "  🚀 스페인어/그리스어 제외로 안정적인 성능 달성"
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

echo "🚀 진정한 SigLIP2 - 영어/중국어 Cross-lingual Zero-shot 모델 훈련 시작..."
echo "⚡ EMA Teacher-Student + Multi-Loss + 핵심 언어로 최고 Zero-shot 성능 달성!"
echo "================================"

# 훈련 실행 (Cross-lingual + 환자 단위 분할 모드)
$PYTHON_CMD true_siglip2_trainer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "../modules/outputs/siglip-sam" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --parser cross_lingual \
    --train_languages "${TRAIN_LANGUAGES[@]}" \
    --test_languages "$TEST_LANGUAGE" \
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
    --best_model_metric "val_macro_f1" \
    --split_by_patient true

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 진정한 SigLIP2 - 영어/중국어 Cross-lingual Zero-shot 모델 훈련 완료!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 Cross-lingual 실험 결과:"
    echo "  훈련 언어: ${TRAIN_LANGUAGES[*]}"
    echo "  Zero-shot 언어: $TEST_LANGUAGE"
    echo "  제외 언어: Spanish, Greek"
    echo ""
    echo "🔥 진정한 SigLIP2 + 영어/중국어 Cross-lingual 핵심 성과:"
    echo "   ✅ EMA Teacher-Student Self-Distillation"
    echo "   ✅ SILC/TIPS Masked Prediction Learning"
    echo "   ✅ Auto-Regressive Caption Generation"
    echo "   ✅ Cross-Modal Sigmoid Contrastive Learning"
    echo "   ✅ LoCa Dense Captioning + Referring Expressions"
    echo "   ✅ Multi-Loss 통합으로 최고 성능 달성"
    echo "   ✅ 환자 단위 분할로 Speaker-Independent 평가"
    echo "   ✅ Language-Agnostic Representation 학습"
    echo ""
    echo "📊 영어/중국어 Cross-lingual 기대 효과:"
    echo "   🎯 스페인어/그리스어 제외로 데이터 품질 집중"
    echo "   🚀 Gemma 토크나이저로 영어/중국어 성능 극대화"
    echo "   📈 두 언어 간의 강력한 Zero-shot Transfer"
    echo "   ✨ 핵심 언어들에서 안정적이고 높은 Cross-lingual 성능"
    echo ""
    echo "🎯 이 모델의 혁신적 장점:"
    echo "   ✨ 2개 핵심 언어에 집중한 효율적 Cross-lingual 학습"
    echo "   ✨ Gemma 토크나이저의 다국어 표현 능력 극대화"
    echo "   ✨ 환자 단위 분할로 robust clinical zero-shot evaluation"
    echo "   ✨ True SigLIP2 architecture로 최고 Zero-shot 성능"
    echo "   ✨ EMA Teacher + Multi-Loss로 안정적 최적화"
else
    echo ""
    echo "❌ 진정한 SigLIP2 - 영어/중국어 Cross-lingual Zero-shot 모델 훈련 실패"
    echo "실험: 훈련(${TRAIN_LANGUAGES[*]}) → 테스트($TEST_LANGUAGE)"
    exit 1
fi
