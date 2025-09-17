#!/bin/bash
# 진정한 SigLIP2 - 3개 언어 통합 훈련 (그리스어 제외)
# 영어, 만다린, 스페인어만을 사용한 다국어 통합 학습
# EMA Teacher-Student + Multi-Loss로 최강 언어 무관 성능

echo "=== 진정한 SigLIP2 - 3개 언어 통합 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# 언어 설정 (그리스어 제외)
LANGUAGES="English Spanish Mandarin"
OUTPUT_DIR="../modules/outputs/siglip/True_SigLIP2_3Languages"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "🌍 진정한 SigLIP2 - 3개 언어 통합 훈련 설정:"
echo "  실험명: True_SigLIP2_3Languages"
echo "  언어: $LANGUAGES (그리스어 제외)"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  모델: $MODEL_NAME"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo ""

# True SigLIP2 Multi-Loss 설정
EMA_MOMENTUM=0.999
SILC_WEIGHT=0.2
SIGMOID_WEIGHT=1.0
LOCA_WEIGHT=1.0
CLASSIFICATION_WEIGHT=1.0
MASK_RATIO=0.15
DECODER_HIDDEN_DIM=512
DECODER_NUM_HEADS=8
DECODER_NUM_LAYERS=6
VOCAB_SIZE=30522
MAX_CAPTION_LENGTH=77

echo "🔥 진정한 SigLIP2 Multi-Loss 구조:"
echo "  🧑‍🏫 EMA Teacher Momentum: ${EMA_MOMENTUM}"
echo "  📚 SILC/TIPS Loss: ${SILC_WEIGHT} (Self-Distillation)"
echo "  🔗 Sigmoid Loss: ${SIGMOID_WEIGHT} (Cross-Modal Contrastive)"
echo "  📝 LoCa Loss: ${LOCA_WEIGHT} (Caption Generation)"
echo "  🎯 Classification Loss: ${CLASSIFICATION_WEIGHT} (Dementia Diagnosis)"
echo ""
echo "🔥 3개 언어 통합 특화 설정:"
echo "  ✨ 영어, 만다린, 스페인어의 강력한 언어 무관 표현 학습"
echo "  ✨ EMA Teacher-Student로 안정적 특징 학습"
echo "  ✨ Multi-Loss 통합으로 최고 성능 달성"
echo "  ✨ Auto-Regressive Decoder로 캡션 생성 능력 향상"
echo ""
echo "📊 베스트 모델 선택 기준:"
echo "  🎯 영어, 스페인어, 만다린 Validation AUC 평균"
echo "  📈 언어 편향 방지를 위한 균형잡힌 평가"
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
echo "🚀 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 시작..."
echo "⏳ Early Stopping: 평균 AUC 기준 10 epochs patience"
echo ""

$PYTHON_CMD true_siglip2_trainer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --parser "all" \
    --languages $LANGUAGES \
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
    --target_languages "English" "Spanish" "Mandarin"

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 성공!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "📊 이 모델은 영어, 만다린, 스페인어 데이터로 훈련되어"
    echo "   3개 언어에서 뛰어난 성능을 보일 것으로 예상됩니다."
    echo "   ⚡ 진정한 SigLIP2의 모든 이점으로 최강 다국어 성능!"
    echo ""
    echo "🔍 3개 언어 통합 분석 인사이트:"
    echo "   ✅ 3개 언어 통합 성능 - 언어 무관 표현 학습"
    echo "   ✅ EMA Teacher vs Student 성능 비교"
    echo "   ✅ Multi-Loss components의 기여도 분석"
    echo "   ✅ 언어별 세부 성능 분석"
    echo ""
    echo "📊 결과 확인:"
    echo "   - 언어별 훈련 성능 분석"
    echo "   - EMA Teacher-Student 각각의 성능"
    echo "   - Multi-Loss 실시간 모니터링 및 기여도 분석"
    echo "   - Cross-modal alignment의 효과"
    echo ""
    echo "🎯 기대 효과:"
    echo "   ✨ 3개 언어의 강력한 언어 무관 표현으로 최고 성능"
    echo "   ✨ 진정한 SigLIP2 아키텍처의 모든 이점 활용"
    echo "   ✨ 영어, 만다린, 스페인어에서 예상되는 뛰어난 성능"
    echo "   ✨ 다른 언어로의 확장 가능성 검증"
else
    echo ""
    echo "❌ 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 실패"
    exit 1
fi
