#!/bin/bash

# SigLIP-Text-Only 모델 훈련 스크립트
# SigLIP의 텍스트 인코더 + Gemma 토크나이저를 사용하는 텍스트 전용 대조군

set -e

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 실험 설정
LANGUAGES=("English" "Mandarin")
DATA_DIR="../../training_dset"
BATCH_SIZE=10
LEARNING_RATE=2e-5
NUM_EPOCHS=100
EARLY_STOPPING_PATIENCE=15

# 모델 설정
SIGLIP_MODEL="google/siglip-base-patch16-224"
TEXT_TOKENIZER="google/gemma-2b"
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
AUTO_CLASS_WEIGHTS="true"
BEST_MODEL_METRIC="avg_lang_macro_f1"
SPLIT_BY_PATIENT="true"

# 출력 설정
OUTPUT_DIR="../modules/outputs/controlgroups"
WANDB_PROJECT="dementia-controlgroups"

echo "🔥 SigLIP-Text-Only 대조군 모델 훈련 시작"
echo "============================================="
echo ""
echo "📋 실험 설정:"
echo "  모델 타입: SigLIP-Text-Only (텍스트 인코더만)"
echo "  언어: ${LANGUAGES[*]}"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크: $NUM_EPOCHS"
echo ""
echo "🤖 모델 구성:"
echo "  SigLIP 백본: $SIGLIP_MODEL"
echo "  텍스트 토크나이저: $TEXT_TOKENIZER (256K vocab, multilingual)"
echo "  손실 함수: $LOSS_TYPE (alpha=$FOCAL_ALPHA, gamma=$FOCAL_GAMMA)"
echo "  자동 클래스 가중치: $AUTO_CLASS_WEIGHTS"
echo "  환자 단위 분할: $SPLIT_BY_PATIENT"
echo ""
echo "📊 평가 설정:"
echo "  베스트 모델 선택: $BEST_MODEL_METRIC"
echo "  Early Stopping: $EARLY_STOPPING_PATIENCE epochs patience"
echo "  타겟 언어: ${LANGUAGES[*]}"
echo ""
echo "🎯 SigLIP vs 기존 대조군 비교:"
echo "  ✅ SigLIP의 검증된 텍스트 인코더 사용"
echo "  ✅ 동일한 Gemma 토크나이저 (256K vocab)"
echo "  ✅ 공정한 성능 비교 기준"
echo "  ✅ SigLIP과 동일한 데이터 품질 (완전한 샘플만 사용)"
echo ""
echo "⚡ 성능 최적화:"
echo "  🔥 멀티 GPU 훈련 활성화"
echo "  🎯 Focal Loss + 자동 클래스 가중치"
echo "  📈 Macro F1 기준 베스트 모델 선택"
echo "  ⏰ 15 epochs Early Stopping"
echo ""
echo "📝 처리 파이프라인:"
echo "  텍스트 → Gemma 토크나이저 → SigLIP Text Encoder → 분류기"
echo "  특징: 언어 정보만으로 치매 진단 수행"
echo ""

# GPU 정보 출력
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | nl -v0 | sed 's/^/  GPU /'
    echo ""
fi

echo "🚀 훈련 시작..."
echo ""

# Python 훈련 실행
python train_siglip_text_only.py \
    --data_dir "$DATA_DIR" \
    --languages "${LANGUAGES[@]}" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --siglip_model "$SIGLIP_MODEL" \
    --text_tokenizer "$TEXT_TOKENIZER" \
    --loss_type "$LOSS_TYPE" \
    --focal_alpha $FOCAL_ALPHA \
    --focal_gamma $FOCAL_GAMMA \
    --auto_class_weights "$AUTO_CLASS_WEIGHTS" \
    --best_model_metric "$BEST_MODEL_METRIC" \
    --target_languages "${LANGUAGES[@]}" \
    --split_by_patient "$SPLIT_BY_PATIENT" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ SigLIP-Text-Only 모델 훈련 완료!"
echo "📁 모델 저장 위치: $OUTPUT_DIR"
echo ""
echo "🔍 다음 단계:"
echo "  1. SigLIP-Audio-Only 모델과 성능 비교"
echo "  2. SigLIP-Concat 모델 훈련"
echo "  3. Full SigLIP 모델과 성능 비교"
echo ""
echo "📊 성능 비교 예상:"
echo "  텍스트 정보는 오디오 정보보다 치매 진단에 더 유용할 것으로 예상"
echo "  언어별 성능 차이: 영어 vs 중국어 비교 분석"
