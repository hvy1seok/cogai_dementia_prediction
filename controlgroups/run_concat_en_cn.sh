#!/bin/bash
# Control Group: Concat (ViT + XLM-R) Late Fusion - 영어 + 중국어
# Concat (vit + XLM-R): 두 임베딩 late fusion(concat) → 2층 FFN(sigmoid)

echo "=== Control Group: Concat (ViT + XLM-R) Late Fusion 모델 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/controlgroups/concat_en_cn"
LANGUAGES="English Mandarin"

# 훈련 설정
BATCH_SIZE=64
LEARNING_RATE=2e-5
NUM_EPOCHS=100
EARLY_STOPPING_PATIENCE=10

# 모델 설정
AUDIO_ENCODER="google/vit-base-patch16-224"
TEXT_ENCODER="xlm-roberta-base"
FUSION_METHOD="concat"

# 손실 함수 설정
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

echo ""
echo "🔗 Concat (ViT + XLM-R) Late Fusion 대조군 모델 설정:"
echo "  모델 유형: Multimodal Late Fusion (오디오 + 텍스트)"
echo "  언어: $LANGUAGES"
echo "  오디오 인코더: $AUDIO_ENCODER"
echo "  텍스트 인코더: $TEXT_ENCODER"
echo "  Fusion 방식: $FUSION_METHOD (Late Fusion)"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크 수: $NUM_EPOCHS"
echo "  Early Stopping: $EARLY_STOPPING_PATIENCE epochs"
echo "  손실 함수: $LOSS_TYPE (alpha=$FOCAL_ALPHA, gamma=$FOCAL_GAMMA)"
echo ""
echo "📊 데이터 분할 방식:"
echo "  👥 환자 단위 분할 - Speaker-Independent"
echo "  🎯 실제 임상 환경과 유사한 평가"
echo "  ✅ 동일 환자의 샘플이 한 세트에만 존재"
echo ""
echo "📊 베스트 모델 선택 기준:"
echo "  🎯 2개 언어들(English, Mandarin) Validation Macro F1 평균"
echo "  📈 언어 편향 방지를 위한 균형잡힌 평가"
echo ""
echo "🔗 Concat Late Fusion 모델의 특징:"
echo "  ✅ ViT로 스펙트로그램 특징 추출"
echo "  ✅ XLM-R로 다국어 텍스트 이해"
echo "  ✅ Late Fusion: 두 모달리티 임베딩 연결"
echo "  ✅ 2층 FFN으로 융합된 특징 분류"
echo "  ✅ Focal Loss로 클래스 불균형 해결"
echo ""
echo "🌍 Late Fusion 아키텍처의 장점:"
echo "  🎯 각 모달리티의 독립적 특징 학습"
echo "  🚀 오디오-텍스트 상호보완적 정보 활용"
echo "  📈 단순하지만 효과적인 융합 방식"
echo "  ✨ 영어-중국어 멀티모달 패턴 학습"
echo ""
echo "🔧 모델 구조:"
echo "  📸 오디오: 스펙트로그램 → ViT → [CLS] 토큰 (768차원)"
echo "  📝 텍스트: 전사 → XLM-R → [CLS] 토큰 (768차원)"
echo "  🔗 Fusion: Concat(오디오, 텍스트) → 1536차원"
echo "  🧠 분류기: 1536 → 1024 → 512 → 2 (2층 FFN)"
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

echo "🚀 Concat (ViT + XLM-R) Late Fusion 대조군 모델 훈련 시작..."
echo "🔗 멀티모달 Late Fusion 치매 진단 모델 학습!"
echo "================================"

# controlgroups 디렉토리로 이동하여 실행 (import 오류 방지)
cd "$(dirname "$0")"

# Python path 설정 (현재 디렉토리를 Python path에 추가)
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 훈련 실행
$PYTHON_CMD train_concat.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --languages $LANGUAGES \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --audio_encoder "$AUDIO_ENCODER" \
    --text_encoder "$TEXT_ENCODER" \
    --fusion_method "$FUSION_METHOD" \
    --use_cls_token \
    --loss_type "$LOSS_TYPE" \
    --focal_alpha $FOCAL_ALPHA \
    --focal_gamma $FOCAL_GAMMA \
    --auto_class_weights \
    --best_model_metric "avg_lang_macro_f1" \
    --target_languages "English" "Mandarin" \
    --split_by_patient true

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Concat (ViT + XLM-R) Late Fusion 대조군 모델 훈련 완료!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR"
    echo ""
    echo "🔗 Concat Late Fusion 모델 핵심 성과:"
    echo "   ✅ ViT로 스펙트로그램 특징 추출"
    echo "   ✅ XLM-R로 다국어 텍스트 이해"
    echo "   ✅ Late Fusion으로 모달리티 융합"
    echo "   ✅ 2층 FFN으로 융합된 특징 분류"
    echo "   ✅ Focal Loss로 클래스 불균형 해결"
    echo "   ✅ 환자 단위 분할로 Speaker-Independent 평가"
    echo ""
    echo "📊 이 모델의 장점:"
    echo "   🎯 오디오-텍스트 상호보완적 정보 활용"
    echo "   🚀 각 모달리티의 독립적 특징 학습"
    echo "   📈 단순하지만 효과적인 Late Fusion"
    echo "   ✨ 영어-중국어 멀티모달 패턴 학습"
    echo ""
    echo "🔧 Late Fusion vs Early Fusion:"
    echo "   ✨ 각 모달리티가 독립적으로 학습"
    echo "   ✨ 모달리티 간 간섭 최소화"
    echo "   ✨ 해석 가능한 융합 방식"
    echo "   ✨ 안정적이고 예측 가능한 성능"
else
    echo ""
    echo "❌ Concat (ViT + XLM-R) Late Fusion 대조군 모델 훈련 실패"
    exit 1
fi
