#!/bin/bash
# Cross-lingual 실험 - 한 언어 빼고 나머지로 훈련

echo "=== 다국어 멀티모달 치매 진단 모델 - Cross-lingual 실험 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="training_dset"
MAX_SEQ_LEN=512
BATCH_SIZE=16
NUM_EPOCHS=50
LEARNING_RATE=2e-5
TEXT_MODEL_TYPE=1
DROPOUT=0.3
SEED=42

# Cross-lingual 실험 조합 정의
declare -a EXPERIMENTS=(
    # 실험 1: 영어+그리스어+스페인어 → 만다린
    "English Greek Spanish|Mandarin|Train_EGS_Test_Mandarin"
    # 실험 2: 영어+그리스어+만다린 → 스페인어
    "English Greek Mandarin|Spanish|Train_EGM_Test_Spanish"
    # 실험 3: 영어+스페인어+만다린 → 그리스어
    "English Spanish Mandarin|Greek|Train_ESM_Test_Greek"
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

# 실험 실행
EXPERIMENT_COUNT=0
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

echo "🌍 Cross-lingual 실험 계획:"
echo "  총 실험 수: $TOTAL_EXPERIMENTS"
echo "  각 실험: 3개 언어로 훈련 → 1개 언어로 테스트"
echo ""

for experiment in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
    
    # 실험 정보 파싱
    IFS='|' read -r TRAIN_LANGS_STR TEST_LANGS_STR EXP_NAME <<< "$experiment"
    
    echo "🔬 실험 $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS: $EXP_NAME"
    echo "================================"
    echo "  훈련 언어: $TRAIN_LANGS_STR"
    echo "  테스트 언어: $TEST_LANGS_STR"
    echo "  시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 훈련 실행
    START_TIME=$(date +%s)
    
    $PYTHON_CMD main_multilingual.py \
        --mode cross_lingual \
        --data_dir "$DATA_DIR" \
        --train_languages $TRAIN_LANGS_STR \
        --test_languages $TEST_LANGS_STR \
        --max_seq_len $MAX_SEQ_LEN \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --text_model_type $TEXT_MODEL_TYPE \
        --dropout $DROPOUT \
        --seed $SEED \
        --num_workers 4
    
    # 결과 확인
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    
    if [ $? -eq 0 ]; then
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        echo ""
        echo "✅ 실험 $EXPERIMENT_COUNT 성공: $EXP_NAME (소요시간: ${DURATION_MIN}분)"
        echo ""
    else
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        echo ""
        echo "❌ 실험 $EXPERIMENT_COUNT 실패: $EXP_NAME (소요시간: ${DURATION_MIN}분)"
        echo ""
    fi
    
    echo "진행 상황: $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS 완료"
    echo "성공: $SUCCESSFUL_EXPERIMENTS, 실패: $FAILED_EXPERIMENTS"
    echo ""
    echo "================================"
    echo ""
done

# 최종 결과 요약
echo "🎉 모든 Cross-lingual 실험 완료!"
echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 실험 결과 요약:"
echo "  총 실험 수: $TOTAL_EXPERIMENTS"
echo "  성공한 실험: $SUCCESSFUL_EXPERIMENTS"
echo "  실패한 실험: $FAILED_EXPERIMENTS"
echo "  성공률: $((SUCCESSFUL_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS))%"
echo ""
echo "🏆 결과 저장 위치:"
echo "  모델: best_model_CrossLingual_*.pth"
echo "  결과: checkpoints/results_CrossLingual_*.json"
echo "  wandb: https://wandb.ai/your-username/dementia-multilingual"
echo ""
echo "🌍 Cross-lingual 분석 인사이트:"
echo "  ✅ 언어 간 전이 학습 효과 분석"
echo "  ✅ 각 언어의 일반화 능력 평가"
echo "  ✅ 언어별 치매 진단 특성 비교"
echo "  ✅ Zero-shot 성능 vs 훈련 언어 성능 비교"
echo ""
echo "📊 결과 분석 방법:"
echo "  1. wandb에서 실험별 성능 비교"
echo "  2. JSON 파일에서 상세 언어별 메트릭 확인"
echo "  3. 언어 조합별 최적 성능 분석"
echo ""
echo "🚀 다음 단계:"
echo "  - 성능이 좋은 언어 조합 분석"
echo "  - 특정 언어에 특화된 모델 개발"
echo "  - 더 많은 언어로 확장 실험"
