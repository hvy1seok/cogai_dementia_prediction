#!/bin/bash
# 전체 언어 환자 단위 Stratified Split 실험

echo "=== 다국어 멀티모달 치매 진단 모델 - 전체 언어 실험 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 기본 설정
DATA_DIR="training_dset"
MAX_SEQ_LEN=512
BATCH_SIZE=16
NUM_EPOCHS=50
LEARNING_RATE=2e-5
TEXT_MODEL_TYPE=1  # 1: BERT only, 2: BERT + LSTM
DROPOUT=0.3
SEED=42

# 사용할 언어들
LANGUAGES="English Greek Spanish Mandarin"

echo ""
echo "🌍 전체 언어 환자 단위 Stratified Split 실험 설정:"
echo "  언어: $LANGUAGES"
echo "  데이터 디렉토리: $DATA_DIR"
echo "  배치 크기: $BATCH_SIZE"
echo "  에포크: $NUM_EPOCHS"
echo "  학습률: $LEARNING_RATE"
echo "  텍스트 모델: Type $TEXT_MODEL_TYPE"
echo "  드롭아웃: $DROPOUT"
echo "  랜덤 시드: $SEED"
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

echo "🚀 전체 언어 모델 훈련 시작..."
echo "================================"

# 훈련 실행
$PYTHON_CMD main_multilingual.py \
    --mode all_languages \
    --data_dir "$DATA_DIR" \
    --languages $LANGUAGES \
    --max_seq_len $MAX_SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --text_model_type $TEXT_MODEL_TYPE \
    --dropout $DROPOUT \
    --seed $SEED \
    --num_workers 4

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 전체 언어 모델 훈련이 성공적으로 완료되었습니다!"
    echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "🎯 실험 결과:"
    echo "  - 환자 단위 Stratified Split으로 데이터 분할"
    echo "  - 모든 언어($LANGUAGES)에서 훈련 및 테스트"
    echo "  - 언어별 상세 성능 메트릭 자동 계산"
    echo "  - wandb에서 실시간 모니터링 및 시각화"
    echo ""
    echo "📊 결과 확인:"
    echo "  - 모델: best_model_AllLanguages_*.pth"
    echo "  - 결과: checkpoints/results_AllLanguages_*.json"
    echo "  - wandb: https://wandb.ai/your-username/dementia-multilingual"
    echo ""
    echo "🌍 언어별 성능 분석:"
    echo "  각 언어별로 정확도, AUC, F1-score 등이 자동 계산됩니다."
    echo "  wandb 대시보드에서 언어 간 성능 비교가 가능합니다."
else
    echo ""
    echo "❌ 전체 언어 모델 훈련 중 오류가 발생했습니다."
    exit 1
fi
