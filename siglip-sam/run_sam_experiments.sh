#!/bin/bash
# SigLIP-SAM 실험 - 3가지 손실 함수 × SAM 옵티마이저 = 3가지 조합

echo "=== SigLIP-SAM 실험 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 공통 설정
DATA_DIR="../training_dset"
BASE_OUTPUT_DIR="../modules/outputs/siglip-sam/SAM_Experiments"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100
LANGUAGES="English Greek Spanish Mandarin"

# SAM 설정
OPTIMIZER_TYPE="sam"
SAM_RHO=0.05

# 실험 조합 정의 (SAM + 3가지 손실 함수)
declare -a LOSS_TYPES=("cross_entropy" "focal" "bce")

# 손실 함수 이름 매핑
declare -A LOSS_NAMES
LOSS_NAMES["cross_entropy"]="CrossEntropy"
LOSS_NAMES["focal"]="FocalLoss"
LOSS_NAMES["bce"]="BCE"

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

# 실험 카운터
EXPERIMENT_COUNT=0
TOTAL_EXPERIMENTS=${#LOSS_TYPES[@]}
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# 결과 로그 파일
RESULTS_LOG="$BASE_OUTPUT_DIR/sam_experiment_results.log"
mkdir -p "$BASE_OUTPUT_DIR"

echo "📊 SAM 실험 계획:" > "$RESULTS_LOG"
echo "총 실험 수: $TOTAL_EXPERIMENTS" >> "$RESULTS_LOG"
echo "손실 함수: ${LOSS_TYPES[*]}" >> "$RESULTS_LOG"
echo "옵티마이저: SAM (rho=$SAM_RHO)" >> "$RESULTS_LOG"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_LOG"
echo "================================" >> "$RESULTS_LOG"
echo ""

# SAM 실험 실행
for loss_type in "${LOSS_TYPES[@]}"; do
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
    
    # 실험 이름 생성
    EXPERIMENT_NAME="SAM_${LOSS_NAMES[$loss_type]}"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$EXPERIMENT_NAME"
    
    echo "🎯 실험 $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS: $EXPERIMENT_NAME"
    echo "================================"
    echo "  손실 함수: $loss_type"
    echo "  옵티마이저: SAM (rho=$SAM_RHO)"
    echo "  출력 디렉토리: $OUTPUT_DIR"
    echo "  시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/checkpoints"
    
    # 실험 시작 로그
    echo "실험 $EXPERIMENT_COUNT: $EXPERIMENT_NAME - 시작 $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_LOG"
    
    # 훈련 실행
    START_TIME=$(date +%s)
    
    $PYTHON_CMD trainer.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --parser all \
        --languages $LANGUAGES \
        --loss_type "$loss_type" \
        --optimizer_type "$OPTIMIZER_TYPE" \
        --sam_rho "$SAM_RHO" \
        --focal_alpha 1.0 \
        --focal_gamma 2.0
    
    # 결과 확인
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    
    if [ $? -eq 0 ]; then
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        echo "✅ 실험 $EXPERIMENT_COUNT 성공: $EXPERIMENT_NAME (소요시간: ${DURATION_MIN}분)"
        echo "실험 $EXPERIMENT_COUNT: $EXPERIMENT_NAME - 성공 $(date '+%Y-%m-%d %H:%M:%S') (${DURATION_MIN}분)" >> "$RESULTS_LOG"
    else
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        echo "❌ 실험 $EXPERIMENT_COUNT 실패: $EXPERIMENT_NAME (소요시간: ${DURATION_MIN}분)"
        echo "실험 $EXPERIMENT_COUNT: $EXPERIMENT_NAME - 실패 $(date '+%Y-%m-%d %H:%M:%S') (${DURATION_MIN}분)" >> "$RESULTS_LOG"
    fi
    
    echo ""
    echo "진행 상황: $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS 완료"
    echo "성공: $SUCCESSFUL_EXPERIMENTS, 실패: $FAILED_EXPERIMENTS"
    echo ""
    echo "================================"
    echo ""
done

# 최종 결과 요약
echo "🎉 모든 SAM 실험 완료!"
echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 실험 결과 요약:"
echo "  총 실험 수: $TOTAL_EXPERIMENTS"
echo "  성공한 실험: $SUCCESSFUL_EXPERIMENTS"
echo "  실패한 실험: $FAILED_EXPERIMENTS"
echo "  성공률: $((SUCCESSFUL_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))%"
echo ""
echo "🏆 결과 저장 위치:"
echo "  메인 디렉토리: $BASE_OUTPUT_DIR"
echo "  결과 로그: $RESULTS_LOG"
echo ""

# 최종 결과를 로그에 기록
echo "" >> "$RESULTS_LOG"
echo "================================" >> "$RESULTS_LOG"
echo "최종 결과 요약 - $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_LOG"
echo "총 실험 수: $TOTAL_EXPERIMENTS" >> "$RESULTS_LOG"
echo "성공한 실험: $SUCCESSFUL_EXPERIMENTS" >> "$RESULTS_LOG"
echo "실패한 실험: $FAILED_EXPERIMENTS" >> "$RESULTS_LOG"
echo "성공률: $((SUCCESSFUL_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))%" >> "$RESULTS_LOG"

# 성공한 실험들의 베스트 모델 경로 출력
echo ""
echo "🏆 성공한 실험들의 베스트 모델:"
for loss_type in "${LOSS_TYPES[@]}"; do
    EXPERIMENT_NAME="SAM_${LOSS_NAMES[$loss_type]}"
    CHECKPOINT_DIR="$BASE_OUTPUT_DIR/$EXPERIMENT_NAME/checkpoints"
    
    if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
        BEST_MODEL=$(ls "$CHECKPOINT_DIR"/best_model*.pt 2>/dev/null | head -1)
        if [ -n "$BEST_MODEL" ]; then
            echo "  $EXPERIMENT_NAME: $(basename "$BEST_MODEL")"
        fi
    fi
done

echo ""
echo "🎯 SAM 옵티마이저 실험이 모두 완료되었습니다!"
echo "각 손실 함수와 SAM의 조합 성능을 비교하여 최적의 설정을 찾아보세요! 📈"
