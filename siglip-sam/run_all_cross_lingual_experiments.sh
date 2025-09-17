#!/bin/bash
# SigLIP-SAM 모든 Zero-shot Cross-Lingual 조합 실험 실행
# 4가지 언어 조합을 순차적으로 실행하여 포괄적인 Zero-shot 분석 수행

echo "=== SigLIP-SAM 전체 Zero-shot Cross-Lingual 실험 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 실험 설정
TOTAL_EXPERIMENTS=4
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
BASE_OUTPUT_DIR="../modules/outputs/siglip-sam/All_CrossLingual_Experiments"
RESULTS_LOG="$BASE_OUTPUT_DIR/cross_lingual_experiment_results.log"

# 기본 디렉토리 생성
mkdir -p "$BASE_OUTPUT_DIR"

# 실험 조합 정의
declare -a EXPERIMENT_NAMES=(
    "영어+스페인어+만다린 → 그리스어"
    "영어+그리스어+만다린 → 스페인어"
    "영어+그리스어+스페인어 → 만다린"
    "그리스어+스페인어+만다린 → 영어"
)

echo "📊 전체 Cross-Lingual 실험 계획:" > "$RESULTS_LOG"
echo "총 실험 수: $TOTAL_EXPERIMENTS" >> "$RESULTS_LOG"
echo "실험 조합:" >> "$RESULTS_LOG"
for i in {0..3}; do
    echo "  실험 $((i+1)): ${EXPERIMENT_NAMES[$i]}" >> "$RESULTS_LOG"
done
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_LOG"
echo "================================" >> "$RESULTS_LOG"
echo ""

echo "🌍 실행할 Cross-Lingual 실험:"
for i in {0..3}; do
    echo "  실험 $((i+1)): ${EXPERIMENT_NAMES[$i]}"
done
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

# 모든 Cross-lingual 실험 실행
for experiment_num in {1..4}; do
    echo "🎯 실험 $experiment_num/$TOTAL_EXPERIMENTS: ${EXPERIMENT_NAMES[$((experiment_num-1))]}"
    echo "================================"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 실험 시작 로그
    echo "실험 $experiment_num: ${EXPERIMENT_NAMES[$((experiment_num-1))]} - 시작 $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_LOG"
    
    # Cross-lingual 스크립트 실행
    START_TIME=$(date +%s)
    
    bash train_sam_cross_lingual.sh $experiment_num
    
    # 결과 확인
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    
    if [ $? -eq 0 ]; then
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        echo "✅ 실험 $experiment_num 성공: ${EXPERIMENT_NAMES[$((experiment_num-1))]} (소요시간: ${DURATION_MIN}분)"
        echo "실험 $experiment_num: ${EXPERIMENT_NAMES[$((experiment_num-1))]} - 성공 $(date '+%Y-%m-%d %H:%M:%S') (${DURATION_MIN}분)" >> "$RESULTS_LOG"
    else
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        echo "❌ 실험 $experiment_num 실패: ${EXPERIMENT_NAMES[$((experiment_num-1))]} (소요시간: ${DURATION_MIN}분)"
        echo "실험 $experiment_num: ${EXPERIMENT_NAMES[$((experiment_num-1))]} - 실패 $(date '+%Y-%m-%d %H:%M:%S') (${DURATION_MIN}분)" >> "$RESULTS_LOG"
    fi
    
    echo ""
    echo "진행 상황: $experiment_num/$TOTAL_EXPERIMENTS 완료"
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

# 최종 결과를 로그에 기록
echo "" >> "$RESULTS_LOG"
echo "================================" >> "$RESULTS_LOG"
echo "최종 결과 요약 - $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_LOG"
echo "총 실험 수: $TOTAL_EXPERIMENTS" >> "$RESULTS_LOG"
echo "성공한 실험: $SUCCESSFUL_EXPERIMENTS" >> "$RESULTS_LOG"
echo "실패한 실험: $FAILED_EXPERIMENTS" >> "$RESULTS_LOG"
echo "성공률: $((SUCCESSFUL_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))%" >> "$RESULTS_LOG"

echo "🏆 결과 저장 위치:"
echo "  메인 디렉토리: $BASE_OUTPUT_DIR"
echo "  결과 로그: $RESULTS_LOG"
echo ""

# 성공한 실험들의 베스트 모델 경로 출력
echo "🏆 성공한 실험들의 베스트 모델:"
for experiment_num in {1..4}; do
    case $experiment_num in
        1) EXPERIMENT_NAME="Train_English_Spanish_Mandarin_Test_Greek" ;;
        2) EXPERIMENT_NAME="Train_English_Greek_Mandarin_Test_Spanish" ;;
        3) EXPERIMENT_NAME="Train_English_Greek_Spanish_Test_Mandarin" ;;
        4) EXPERIMENT_NAME="Train_Greek_Spanish_Mandarin_Test_English" ;;
    esac
    
    CHECKPOINT_DIR="../modules/outputs/siglip-sam/CrossLingual_${EXPERIMENT_NAME}_SAM/checkpoints"
    
    if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
        BEST_MODEL=$(ls "$CHECKPOINT_DIR"/best_model*.pt 2>/dev/null | head -1)
        if [ -n "$BEST_MODEL" ]; then
            echo "  실험 $experiment_num (${EXPERIMENT_NAMES[$((experiment_num-1))]}): $(basename "$BEST_MODEL")"
        fi
    fi
done

echo ""
echo "🔍 Cross-Lingual 분석 인사이트:"
echo "   ✅ 4가지 언어 조합의 Cross-lingual 성능 비교"
echo "   ✅ 어떤 언어가 다른 언어로 가장 잘 전이되는지 확인"
echo "   ✅ SAM 옵티마이저의 언어 간 일반화 효과 분석"
echo "   ✅ 언어별 특성이 Cross-lingual 성능에 미치는 영향"
echo ""
echo "📊 결과 확인 방법:"
echo "   - 각 실험 폴더의 콘솔 출력에서 언어별 분석 확인"
echo "   - wandb 대시보드에서 실험별 Cross-lingual 메트릭 비교"
echo "   - ROC 곡선과 Confusion Matrix로 시각적 비교"
echo "   - 언어 조합별 최적 threshold 차이 분석"
echo ""
echo "🎯 모든 Cross-Lingual 실험이 완료되었습니다!"
echo "다양한 언어 조합의 성능을 비교하여 최적의 Cross-lingual 전략을 찾아보세요! 🌍📈"
