#!/bin/bash
# Control Group: Audio-only (ViT-Spec) - 영어 + 중국어
# vit-Spec (Audio-only, multilingual joint)

echo "=== Control Group: Audio-only (ViT-Spec) 모델 훈련 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/controlgroups/audio_only_en_cn"
LANGUAGES="English Mandarin"

# 훈련 설정
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100
EARLY_STOPPING_PATIENCE=15

# 모델 설정
MODEL_NAME="google/vit-base-patch16-224"

# 손실 함수 설정
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

echo ""
echo "🎵 Audio-only (ViT-Spec) 대조군 모델 설정:"
echo "  모델 유형: Audio-only (스펙트로그램만 사용)"
echo "  언어: $LANGUAGES"
echo "  모델: $MODEL_NAME"
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
echo "🎵 Audio-only 모델의 특징:"
echo "  ✅ ViT (Vision Transformer)로 스펙트로그램 처리"
echo "  ✅ 오디오 신호만으로 치매 진단"
echo "  ✅ 언어 독립적인 음성 패턴 학습"
echo "  ✅ Focal Loss로 클래스 불균형 해결"
echo "  ✅ 자동 클래스 가중치로 성능 최적화"
echo "  ✅ SigLIP과 동일한 데이터 품질 (완전한 샘플만 사용)"
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

echo "🚀 Audio-only (ViT-Spec) 대조군 모델 훈련 시작..."
echo "🎵 스펙트로그램 기반 치매 진단 모델 학습!"
echo "================================"

# controlgroups 디렉토리로 이동하여 실행 (import 오류 방지)
cd "$(dirname "$0")"

# Python path 설정 (현재 디렉토리를 Python path에 추가)
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 훈련 실행
$PYTHON_CMD train_audio_only.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --languages $LANGUAGES \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --model_name "$MODEL_NAME" \
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
    echo "✅ Audio-only (ViT-Spec) 대조군 모델 훈련 완료!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "모델 저장 위치: $OUTPUT_DIR"
    echo ""
    echo "🎵 Audio-only 모델 핵심 성과:"
    echo "   ✅ ViT로 스펙트로그램 특징 추출"
    echo "   ✅ 오디오 신호만으로 치매 진단"
    echo "   ✅ 언어 독립적인 음성 패턴 학습"
    echo "   ✅ Focal Loss로 클래스 불균형 해결"
    echo "   ✅ 환자 단위 분할로 Speaker-Independent 평가"
    echo ""
    echo "📊 이 모델의 장점:"
    echo "   🎯 텍스트 정보 없이도 음성만으로 진단"
    echo "   🚀 언어 장벽 없는 범용적 적용 가능"
    echo "   📈 스펙트로그램 패턴 기반 안정적 성능"
    echo "   ✨ 영어-중국어 음성 특성 학습"
else
    echo ""
    echo "❌ Audio-only (ViT-Spec) 대조군 모델 훈련 실패"
    exit 1
fi
