#!/bin/bash
# 진정한 SigLIP2 - Cross-lingual Zero-shot 훈련 (샘플 단위 분할, PyTorch Lightning)
# EMA Teacher-Student + Self-Distillation + Caption Generation
# 환자 단위가 아닌 파일(샘플) 단위로 분할하여 더 많은 학습 데이터 확보

echo "=== 진정한 SigLIP2 - Cross-lingual Zero-shot 훈련 시작 (샘플 단위 분할, PyTorch Lightning) ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# Cross-lingual 실험 조합 정의
declare -A EXPERIMENTS=(
    ["Greek"]="English Spanish Mandarin"
    ["English"]="Greek Spanish Mandarin"  
    ["Spanish"]="English Greek Mandarin"
    ["Mandarin"]="English Greek Spanish"
)

# 실험 선택
if [ $# -eq 0 ]; then
    echo "사용법: $0 <실험번호>"
    echo "실험 조합:"
    echo "  1: 그리스어 Zero-shot (훈련: English, Spanish, Mandarin)"
    echo "  2: 영어 Zero-shot (훈련: Greek, Spanish, Mandarin)"
    echo "  3: 스페인어 Zero-shot (훈련: English, Greek, Mandarin)"
    echo "  4: 만다린 Zero-shot (훈련: English, Greek, Spanish)"
    echo "  all: 모든 실험 순차 실행"
    exit 1
fi

EXPERIMENT_NUM=$1

# 실험 조합 매핑
case $EXPERIMENT_NUM in
    1)
        TEST_LANGUAGE="Greek"
        TRAIN_LANGUAGES=(English Spanish Mandarin)
        ;;
    2)
        TEST_LANGUAGE="English"
        TRAIN_LANGUAGES=(Greek Spanish Mandarin)
        ;;
    3)
        TEST_LANGUAGE="Spanish"
        TRAIN_LANGUAGES=(English Greek Mandarin)
        ;;
    4)
        TEST_LANGUAGE="Mandarin"
        TRAIN_LANGUAGES=(English Greek Spanish)
        ;;
    all)
        echo "🔄 모든 Cross-lingual 실험 순차 실행 (샘플 단위 분할, PyTorch Lightning)..."
        for i in {1..4}; do
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
        echo "✅ 모든 Cross-lingual 실험 완료!"
        exit 0
        ;;
    *)
        echo "❌ 잘못된 실험 번호: $EXPERIMENT_NUM"
        echo "1-4 또는 'all'을 입력하세요."
        exit 1
        ;;
esac

# 기본 설정
DATA_DIR="../../training_dset"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=64
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# Focal Loss 설정
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
AUTO_CLASS_WEIGHTS="--auto_class_weights"

# 옵티마이저 설정 (PyTorch Lightning에서는 AdamW만 지원)
OPTIMIZER_TYPE="adamw"

# True SigLIP2 Multi-Loss 가중치
EMA_MOMENTUM=0.999
SILC_WEIGHT=0.2      # SILC/TIPS Loss (20%)
SIGMOID_WEIGHT=1.0   # Sigmoid Contrastive Loss (100%)
LOCA_WEIGHT=1.0      # LoCa Caption Loss (100%)
CLASSIFICATION_WEIGHT=1.0  # Classification Loss (100%)

# 출력 디렉토리 설정
TRAIN_LANGS_STR=$(IFS=_; echo "${TRAIN_LANGUAGES[*]}")
OUTPUT_DIR="../modules/outputs/siglip/True_SigLIP2_CrossLingual_Sample_Split_Lightning_Train_${TRAIN_LANGS_STR}_Test_${TEST_LANGUAGE}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🔥 진정한 SigLIP2 - Cross-lingual Zero-shot 훈련 설정 (샘플 단위 분할, PyTorch Lightning):"
echo "  훈련 언어: ${TRAIN_LANGUAGES[*]}"
echo "  테스트 언어: $TEST_LANGUAGE (Zero-shot)"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  텍스트 토크나이저: google/gemma-2b (256K vocab, multilingual)"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE"
echo "  손실 함수: $LOSS_TYPE + Multi-Loss"
echo "  Early Stopping: 훈련 언어 평균 AUC 기준 15 epochs patience"
echo ""
echo "📊 데이터 분할 방식:"
echo "  🔄 샘플(파일) 단위 분할 - Speaker-Dependent"
echo "  📈 더 많은 학습 데이터 확보 가능"
echo "  ⚠️  동일 환자의 샘플이 train/val/test에 분산될 수 있음"
echo ""
echo "📊 베스트 모델 선택 기준:"
echo "  🎯 훈련 언어들(${TRAIN_LANGUAGES[*]}) Validation AUC 평균"
echo "  📈 언어 편향 방지를 위한 균형잡힌 평가"
echo ""
echo "🎯 진정한 SigLIP2 Multi-Loss 구조:"
echo "  🧑‍🏫 EMA Teacher-Student: momentum=$EMA_MOMENTUM"
echo "  📚 SILC/TIPS Loss: ${SILC_WEIGHT} (Self-Distillation + Masked Prediction)"
echo "  🔗 Sigmoid Loss: ${SIGMOID_WEIGHT} (Cross-Modal Contrastive)"
echo "  📝 LoCa Loss: ${LOCA_WEIGHT} (Caption + Dense Caption + Referring)"
echo "  🎯 Classification Loss: ${CLASSIFICATION_WEIGHT} (Dementia Diagnosis)"
echo ""
echo "✨ Cross-lingual + 샘플 단위 분할 + PyTorch Lightning 핵심 기능:"
echo "  ✅ Gemma 토크나이저로 다국어 표현 능력 극대화 (256K vocab)"
echo "  ✅ EMA Teacher-Student 구조로 Self-Distillation"
echo "  ✅ Masked Prediction으로 Self-Supervised Learning"
echo "  ✅ Auto-Regressive Decoder로 Caption Generation"
echo "  ✅ Cross-Modal Alignment + Language-Agnostic Learning"
echo "  ✅ Dense Captioning + Referring Expressions"
echo "  ✅ PyTorch Lightning으로 안정적인 훈련"
echo "  ✅ 샘플 단위 분할로 최대 데이터 활용"
echo ""
echo "🌍 Cross-lingual Zero-shot 실험:"
echo "  🎯 훈련: ${TRAIN_LANGUAGES[*]} (샘플 단위 분할로 최대 데이터 활용)"
echo "  🔍 검증/테스트: $TEST_LANGUAGE (완전한 Zero-shot)"
echo "  📊 기대: 샘플 단위 분할 + Lightning으로 더 robust한 language-agnostic representation 학습"
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

echo "🚀 진정한 SigLIP2 - Cross-lingual Zero-shot 모델 훈련 시작 (샘플 단위 분할, PyTorch Lightning)..."
echo "⚡ EMA Teacher-Student + Multi-Loss + Lightning + 샘플 단위 분할로 최고 Zero-shot 성능 달성!"
echo "================================"

# 훈련 실행 (Cross-lingual + 샘플 단위 분할 + PyTorch Lightning 모드)
$PYTHON_CMD true_siglip2_trainer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --parser cross_lingual \
    --train_languages "${TRAIN_LANGUAGES[@]}" \
    --test_languages "$TEST_LANGUAGE" \
    --optimizer_type "$OPTIMIZER_TYPE" \
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
    --target_languages "${TRAIN_LANGUAGES[@]}" \
    --split_by_patient false

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 진정한 SigLIP2 - Cross-lingual Zero-shot 모델 훈련 완료! (샘플 단위 분할, PyTorch Lightning)"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 Cross-lingual 실험 결과:"
    echo "  훈련 언어: ${TRAIN_LANGUAGES[*]}"
    echo "  Zero-shot 언어: $TEST_LANGUAGE"
    echo ""
    echo "🔥 진정한 SigLIP2 + Cross-lingual + 샘플 단위 분할 + PyTorch Lightning 핵심 성과:"
    echo "   ✅ EMA Teacher-Student Self-Distillation"
    echo "   ✅ SILC/TIPS Masked Prediction Learning"
    echo "   ✅ Auto-Regressive Caption Generation"
    echo "   ✅ Cross-Modal Sigmoid Contrastive Learning"
    echo "   ✅ LoCa Dense Captioning + Referring Expressions"
    echo "   ✅ Multi-Loss 통합으로 최고 성능 달성"
    echo "   ✅ PyTorch Lightning으로 안정적이고 확장가능한 훈련"
    echo "   ✅ 샘플 단위 분할로 최대 데이터 활용"
    echo "   ✅ Language-Agnostic Representation 학습"
    echo ""
    echo "📊 샘플 단위 분할 + Lightning Cross-lingual 기대 효과:"
    echo "   🎯 더 많은 학습 데이터로 robust cross-lingual transfer"
    echo "   🚀 Gemma 토크나이저로 다국어 이해 능력 강화"
    echo "   📈 샘플 단위 분할로 더 높은 Zero-shot 성능"
    echo "   ✨ Language-agnostic feature representation 극대화"
    echo "   ⚡ PyTorch Lightning으로 production-ready 코드"
    echo ""
    echo "🎯 이 모델의 혁신적 장점:"
    echo "   ✨ 샘플 단위 분할로 maximum training data utilization"
    echo "   ✨ True SigLIP2 architecture로 최고 성능"
    echo "   ✨ Cross-lingual transfer capability 극대화"
    echo "   ✨ PyTorch Lightning으로 안정적이고 확장가능한 훈련"
    echo "   ✨ Gemma 토크나이저로 다국어 표현 능력 극대화"
else
    echo ""
    echo "❌ 진정한 SigLIP2 - Cross-lingual Zero-shot 모델 훈련 실패"
    echo "실험: 훈련(${TRAIN_LANGUAGES[*]}) -> 테스트($TEST_LANGUAGE)"
    exit 1
fi
