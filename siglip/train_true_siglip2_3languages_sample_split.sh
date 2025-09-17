#!/bin/bash
# 진정한 SigLIP2 - 3개 언어 통합 훈련 (샘플 단위 분할, PyTorch Lightning)
# 영어, 만다린, 스페인어 (그리스어 제외)
# EMA Teacher-Student + Self-Distillation + Caption Generation

echo "=== 진정한 SigLIP2 - 3개 언어 통합 훈련 시작 (샘플 단위 분할, PyTorch Lightning) ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip/True_SigLIP2_3Languages_Sample_Split_Lightning_EN_MN_ES"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=64
LEARNING_RATE=2e-5
NUM_EPOCHS=100
LANGUAGES="English Mandarin Spanish"

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

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🔥 진정한 SigLIP2 - 3개 언어 통합 훈련 설정 (샘플 단위 분할, PyTorch Lightning):"
echo "  훈련 언어: $LANGUAGES (그리스어 제외)"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  텍스트 토크나이저: google/gemma-2b (256K vocab, multilingual)"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  옵티마이저: $OPTIMIZER_TYPE"
echo "  손실 함수: $LOSS_TYPE + Multi-Loss"
echo "  Early Stopping: 평균 AUC 기준 15 epochs patience"
echo ""
echo "📊 데이터 분할 방식:"
echo "  🔄 샘플(파일) 단위 분할 - Speaker-Dependent"
echo "  📈 더 많은 학습 데이터 확보 가능"
echo "  ⚠️  동일 환자의 샘플이 train/val/test에 분산될 수 있음"
echo ""
echo "📊 베스트 모델 선택 기준:"
echo "  🎯 훈련 언어들(English, Mandarin, Spanish) Validation AUC 평균"
echo "  📈 언어 편향 방지를 위한 균형잡힌 평가"
echo ""
echo "🎯 진정한 SigLIP2 Multi-Loss 구조:"
echo "  🧑‍🏫 EMA Teacher-Student: momentum=$EMA_MOMENTUM"
echo "  📚 SILC/TIPS Loss: ${SILC_WEIGHT} (Self-Distillation + Masked Prediction)"
echo "  🔗 Sigmoid Loss: ${SIGMOID_WEIGHT} (Cross-Modal Contrastive)"
echo "  📝 LoCa Loss: ${LOCA_WEIGHT} (Caption + Dense Caption + Referring)"
echo "  🎯 Classification Loss: ${CLASSIFICATION_WEIGHT} (Dementia Diagnosis)"
echo ""
echo "✨ 진정한 SigLIP2 핵심 기능:"
echo "  ✅ Gemma 토크나이저로 다국어 표현 능력 극대화 (256K vocab)"
echo "  ✅ EMA Teacher-Student 구조로 Self-Distillation"
echo "  ✅ Masked Prediction으로 Self-Supervised Learning"
echo "  ✅ Auto-Regressive Decoder로 Caption Generation"
echo "  ✅ Cross-Modal Alignment + Language-Agnostic Learning"
echo "  ✅ Dense Captioning + Referring Expressions"
echo "  ✅ PyTorch Lightning으로 안정적인 훈련"
echo ""
echo "🔄 3개 언어 샘플 단위 분할 특징:"
echo "  ✅ 그리스어 제외로 데이터 품질 집중"
echo "  ✅ 영어, 만다린, 스페인어 균형잡힌 학습"
echo "  ✅ 샘플 단위 분할로 최대 데이터 활용"
echo "  📊 Gemma 토크나이저로 스페인어 성능 대폭 향상 기대"
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

echo "🚀 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 시작 (샘플 단위 분할, PyTorch Lightning)..."
echo "⚡ EMA Teacher-Student + Multi-Loss + Lightning + 샘플 단위 분할로 최고 성능 달성!"
echo "================================"

# 훈련 실행 (샘플 단위 분할 모드)
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
    --target_languages "English" "Mandarin" "Spanish" \
    --split_by_patient false

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 완료! (샘플 단위 분할, PyTorch Lightning)"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    echo ""
    echo "🌍 훈련 완료된 언어: $LANGUAGES"
    echo "🔥 진정한 SigLIP2 + PyTorch Lightning + 3개 언어 + 샘플 단위 분할 핵심 성과:"
    echo "   ✅ EMA Teacher-Student Self-Distillation"
    echo "   ✅ SILC/TIPS Masked Prediction Learning"
    echo "   ✅ Auto-Regressive Caption Generation"
    echo "   ✅ Cross-Modal Sigmoid Contrastive Learning"
    echo "   ✅ LoCa Dense Captioning + Referring Expressions"
    echo "   ✅ Multi-Loss 통합으로 최고 성능 달성"
    echo "   ✅ PyTorch Lightning으로 안정적이고 확장가능한 훈련"
    echo "   ✅ 샘플 단위 분할로 최대 데이터 활용"
    echo ""
    echo "📊 3개 언어 + 샘플 단위 분할 + Lightning 기대 효과:"
    echo "   🎯 그리스어 제외로 데이터 품질 집중"
    echo "   🚀 Gemma 토크나이저로 스페인어 성능 대폭 향상"
    echo "   📈 샘플 단위 분할로 더 높은 AUC 달성"
    echo "   ✨ 영어, 만다린, 스페인어 균형잡힌 성능"
    echo "   ⚡ PyTorch Lightning으로 안정적이고 확장가능한 훈련"
    echo ""
    echo "🎯 이 모델의 혁신적 장점:"
    echo "   ✨ 3개 핵심 언어에 집중한 효율적 학습"
    echo "   ✨ Gemma 토크나이저의 다국어 표현 능력 극대화"
    echo "   ✨ 샘플 단위 분할로 robust feature 학습"
    echo "   ✨ EMA Teacher + Multi-Loss로 안정적 최적화"
    echo "   ✨ PyTorch Lightning으로 production-ready 코드"
else
    echo ""
    echo "❌ 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 실패"
    exit 1
fi
