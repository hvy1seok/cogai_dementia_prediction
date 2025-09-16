#!/bin/bash
# SigLIP2 Cross-Lingual 실험 - 다양한 언어 조합 테스트

echo "=== SigLIP2 Cross-Lingual 실험 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 공통 설정
DATA_DIR="../../training_dset"
BASE_OUTPUT_DIR="../modules/outputs/siglip/CrossLingual_Experiments"
MODEL_NAME="google/siglip2-base-patch16-naflex"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=100

# Cross-lingual 실험 조합 정의
declare -a EXPERIMENTS=(
    # 실험 1: 영어+스페인어+만다린 → 그리스어
    "English Spanish Mandarin|Greek|Train_ESM_Test_Greek"
    # 실험 2: 영어+그리스어+만다린 → 스페인어  
    "English Greek Mandarin|Spanish|Train_EGM_Test_Spanish"
    # 실험 3: 영어+그리스어+스페인어 → 만다린
    "English Greek Spanish|Mandarin|Train_EGS_Test_Mandarin"
    # 실험 4: 그리스어+스페인어+만다린 → 영어
    "Greek Spanish Mandarin|English|Train_GSM_Test_English"
)

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
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# 결과 로그 파일
RESULTS_LOG="$BASE_OUTPUT_DIR/cross_lingual_results.log"
mkdir -p "$BASE_OUTPUT_DIR"

echo "📊 Cross-Lingual 실험 계획:" > "$RESULTS_LOG"
echo "총 실험 수: $TOTAL_EXPERIMENTS" >> "$RESULTS_LOG"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_LOG"
echo "================================" >> "$RESULTS_LOG"
echo ""

# Cross-lingual 실험 실행
for experiment in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
    
    # 실험 정보 파싱
    IFS='|' read -r TRAIN_LANGS_STR TEST_LANGS_STR EXP_NAME <<< "$experiment"
    IFS=' ' read -r -a TRAIN_LANGS <<< "$TRAIN_LANGS_STR"
    IFS=' ' read -r -a TEST_LANGS <<< "$TEST_LANGS_STR"
    
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$EXP_NAME"
    
    echo "🌍 실험 $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS: $EXP_NAME"
    echo "================================"
    echo "  훈련 언어: ${TRAIN_LANGS[*]}"
    echo "  테스트 언어: ${TEST_LANGS[*]}"
    echo "  출력 디렉토리: $OUTPUT_DIR"
    echo "  시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/checkpoints"
    
    # 실험 시작 로그
    echo "실험 $EXPERIMENT_COUNT: $EXP_NAME - 시작 $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_LOG"
    
    # 훈련 실행
    START_TIME=$(date +%s)
    
    $PYTHON_CMD trainer.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$BASE_OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --parser cross_lingual \
        --train_languages "${TRAIN_LANGS[@]}" \
        --test_languages "${TEST_LANGS[@]}" \
        --loss_type cross_entropy \
        --optimizer_type adamw
    
    # 결과 확인
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    
    if [ $? -eq 0 ]; then
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        echo "✅ 실험 $EXPERIMENT_COUNT 성공: $EXP_NAME (소요시간: ${DURATION_MIN}분)"
        echo "실험 $EXPERIMENT_COUNT: $EXP_NAME - 성공 $(date '+%Y-%m-%d %H:%M:%S') (${DURATION_MIN}분)" >> "$RESULTS_LOG"
    else
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        echo "❌ 실험 $EXPERIMENT_COUNT 실패: $EXP_NAME (소요시간: ${DURATION_MIN}분)"
        echo "실험 $EXPERIMENT_COUNT: $EXP_NAME - 실패 $(date '+%Y-%m-%d %H:%M:%S') (${DURATION_MIN}분)" >> "$RESULTS_LOG"
    fi
    
    echo ""
    echo "진행 상황: $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS 완료"
    echo "성공: $SUCCESSFUL_EXPERIMENTS, 실패: $FAILED_EXPERIMENTS"
    echo ""
    echo "================================"
    echo ""
done

# 최종 결과 요약
echo "🎉 모든 Cross-Lingual 실험 완료!"
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

echo ""
echo "🌍 Cross-Lingual 일반화 실험이 모두 완료되었습니다!"
echo "각 언어 조합의 성능을 비교하여 언어 간 전이 학습 효과를 분석해보세요! 🎯"
