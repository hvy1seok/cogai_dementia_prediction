#!/bin/bash

# SigLIP 기반 대조군 모델 전체 실행 스크립트
# Audio-only, Text-only, Concat 모델을 순차적으로 실행

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
TEXT_TOKENIZER="xlm-roberta-base"
LOSS_TYPE="focal"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
AUTO_CLASS_WEIGHTS="true"
BEST_MODEL_METRIC="avg_lang_macro_f1"
SPLIT_BY_PATIENT="true"

# 출력 설정
OUTPUT_DIR="../modules/outputs/controlgroups"
WANDB_PROJECT="dementia-controlgroups"

# 실행할 모델 선택 (기본값: 모든 모델)
MODELS_TO_RUN=${1:-"all"}

echo "🔥 SigLIP 기반 대조군 모델 전체 실험 시작"
echo "=============================================="
echo ""
echo "📋 실험 설정:"
echo "  언어: ${LANGUAGES[*]}"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  배치 크기: $BATCH_SIZE"
echo "  학습률: $LEARNING_RATE"
echo "  에포크: $NUM_EPOCHS"
echo ""
echo "🤖 모델 구성:"
echo "  SigLIP 백본: $SIGLIP_MODEL"
echo "  텍스트 토크나이저: $TEXT_TOKENIZER (다국어 지원, 안정적)"
echo "  손실 함수: $LOSS_TYPE (alpha=$FOCAL_ALPHA, gamma=$FOCAL_GAMMA)"
echo "  자동 클래스 가중치: $AUTO_CLASS_WEIGHTS"
echo "  환자 단위 분할: $SPLIT_BY_PATIENT"
echo ""
echo "📊 평가 설정:"
echo "  베스트 모델 선택: $BEST_MODEL_METRIC"
echo "  Early Stopping: $EARLY_STOPPING_PATIENCE epochs patience"
echo "  타겟 언어: ${LANGUAGES[*]}"
echo ""
echo "🎯 SigLIP 기반 대조군 특징:"
echo "  ✅ 동일한 SigLIP 백본 사용 (공정한 비교)"
echo "  ✅ 동일한 전처리 파이프라인 (.npy 파일)"
echo "  ✅ 동일한 XLM-R 토크나이저 (다국어 지원)"
echo "  ✅ 동일한 데이터 품질 (완전한 샘플만 사용)"
echo "  ✅ 동일한 처리 속도 (SigLIP과 동일)"
echo ""
echo "⚡ 성능 최적화:"
echo "  🔥 멀티 GPU 훈련 활성화"
echo "  🎯 Focal Loss + 자동 클래스 가중치"
echo "  📈 Macro F1 기준 베스트 모델 선택"
echo "  ⏰ 15 epochs Early Stopping"
echo ""
echo "📊 예상 성능 순서:"
echo "  SigLIP-Audio < SigLIP-Text < SigLIP-Concat < Full SigLIP"
echo "     (60-70%)      (70-80%)      (80-85%)      (85-90%)"
echo ""

# GPU 정보 출력
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | nl -v0 | sed 's/^/  GPU /'
    echo ""
fi

echo "📝 실행할 모델: $MODELS_TO_RUN"
echo ""

# 시작 시간 기록
TOTAL_START_TIME=$(date +%s)

# 1. SigLIP-Audio-Only 모델
if [[ "$MODELS_TO_RUN" == "all" ]] || [[ "$MODELS_TO_RUN" == "1" ]] || [[ "$MODELS_TO_RUN" == "audio" ]]; then
    echo "=================================="
    echo "🎵 1/3: SigLIP-Audio-Only 모델 훈련"
    echo "=================================="
    echo ""
    echo "📋 모델 구성:"
    echo "  타입: SigLIP 이미지 인코더만 사용"
    echo "  입력: 멜스펙토그램 (.npy 파일)"
    echo "  처리: SigLIP Vision Encoder → 분류기"
    echo "  특징: 오디오 정보만으로 치매 진단"
    echo ""
    
    AUDIO_START_TIME=$(date +%s)
    
    python train_siglip_audio_only.py \
        --data_dir "$DATA_DIR" \
        --languages "${LANGUAGES[@]}" \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --siglip_model "$SIGLIP_MODEL" \
        --loss_type "$LOSS_TYPE" \
        --focal_alpha $FOCAL_ALPHA \
        --focal_gamma $FOCAL_GAMMA \
        --auto_class_weights "$AUTO_CLASS_WEIGHTS" \
        --best_model_metric "$BEST_MODEL_METRIC" \
        --target_languages "${LANGUAGES[@]}" \
        --split_by_patient "$SPLIT_BY_PATIENT" \
        --output_dir "$OUTPUT_DIR"
    
    AUDIO_END_TIME=$(date +%s)
    AUDIO_DURATION=$((AUDIO_END_TIME - AUDIO_START_TIME))
    
    echo ""
    echo "✅ SigLIP-Audio-Only 완료! (소요시간: ${AUDIO_DURATION}초)"
    echo ""
fi

# 2. SigLIP-Text-Only 모델
if [[ "$MODELS_TO_RUN" == "all" ]] || [[ "$MODELS_TO_RUN" == "2" ]] || [[ "$MODELS_TO_RUN" == "text" ]]; then
    echo "=================================="
    echo "📝 2/3: SigLIP-Text-Only 모델 훈련"
    echo "=================================="
    echo ""
    echo "📋 모델 구성:"
    echo "  타입: SigLIP 텍스트 인코더 + Gemma 토크나이저"
    echo "  입력: 텍스트 전사 (Gemma 토크나이저)"
    echo "  처리: Gemma 토크나이저 → SigLIP Text Encoder → 분류기"
    echo "  특징: 언어 정보만으로 치매 진단"
    echo ""
    
    TEXT_START_TIME=$(date +%s)
    
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
    
    TEXT_END_TIME=$(date +%s)
    TEXT_DURATION=$((TEXT_END_TIME - TEXT_START_TIME))
    
    echo ""
    echo "✅ SigLIP-Text-Only 완료! (소요시간: ${TEXT_DURATION}초)"
    echo ""
fi

# 3. SigLIP-Concat 모델
if [[ "$MODELS_TO_RUN" == "all" ]] || [[ "$MODELS_TO_RUN" == "3" ]] || [[ "$MODELS_TO_RUN" == "concat" ]]; then
    echo "=================================="
    echo "🔗 3/3: SigLIP-Concat 모델 훈련"
    echo "=================================="
    echo ""
    echo "📋 모델 구성:"
    echo "  타입: SigLIP 이미지+텍스트 인코더 Late Fusion"
    echo "  입력: 멜스펙토그램 + 텍스트 전사"
    echo "  처리: SigLIP Vision + SigLIP Text → Concat → 분류기"
    echo "  특징: 오디오+언어 정보 분리 융합으로 치매 진단"
    echo ""
    
    CONCAT_START_TIME=$(date +%s)
    
    python train_siglip_concat.py \
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
    
    CONCAT_END_TIME=$(date +%s)
    CONCAT_DURATION=$((CONCAT_END_TIME - CONCAT_START_TIME))
    
    echo ""
    echo "✅ SigLIP-Concat 완료! (소요시간: ${CONCAT_DURATION}초)"
    echo ""
fi

# 전체 실험 완료
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo "=============================================="
echo "🎉 SigLIP 기반 대조군 모델 전체 실험 완료!"
echo "=============================================="
echo ""

if [[ "$MODELS_TO_RUN" == "all" ]]; then
    echo "⏱️  소요 시간 요약:"
    if [[ -n "$AUDIO_DURATION" ]]; then
        echo "  SigLIP-Audio-Only: ${AUDIO_DURATION}초 ($(($AUDIO_DURATION/60))분)"
    fi
    if [[ -n "$TEXT_DURATION" ]]; then
        echo "  SigLIP-Text-Only:  ${TEXT_DURATION}초 ($(($TEXT_DURATION/60))분)"
    fi
    if [[ -n "$CONCAT_DURATION" ]]; then
        echo "  SigLIP-Concat:     ${CONCAT_DURATION}초 ($(($CONCAT_DURATION/60))분)"
    fi
    echo "  전체 실험:         ${TOTAL_DURATION}초 ($(($TOTAL_DURATION/60))분)"
else
    echo "⏱️  소요 시간: ${TOTAL_DURATION}초 ($(($TOTAL_DURATION/60))분)"
fi

echo ""
echo "📁 모델 저장 위치: $OUTPUT_DIR"
echo "  - best_siglip_audio_only_English_Mandarin.pt"
echo "  - best_siglip_text_only_English_Mandarin.pt"
echo "  - best_siglip_concat_English_Mandarin.pt"
echo ""
echo "📊 Wandb 프로젝트: $WANDB_PROJECT"
echo "  - siglip-audio-only-English_Mandarin"
echo "  - siglip-text-only-English_Mandarin"
echo "  - siglip-concat-English_Mandarin"
echo ""
echo "🔍 다음 단계:"
echo "  1. Wandb에서 성능 비교 분석"
echo "  2. Full SigLIP 모델과 성능 비교"
echo "  3. 각 컴포넌트의 기여도 분석"
echo "  4. 최종 논문용 결과 정리"
echo ""
echo "📈 예상 성능 검증:"
echo "  Audio < Text < Concat < Full SigLIP 순서 확인"
echo "  각 모델의 언어별 성능 차이 분석"
echo "  멀티모달 융합의 효과 정량화"
echo ""
echo "✨ SigLIP 기반 공정한 대조군 실험 성공!"
