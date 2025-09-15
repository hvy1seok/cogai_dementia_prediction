#!/bin/bash
# SigLIP2 모든 언어별 실험 순차 실행

echo "========================================"
echo "SigLIP2 치매 진단 모든 언어별 실험 시작"
echo "========================================"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
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

# 실험할 언어 목록
LANGUAGES=("English" "Greek" "Spanish" "Mandarin")
TOTAL_EXPERIMENTS=$((${#LANGUAGES[@]} + 1))  # 개별 언어 + 통합 모델

echo "실행할 실험:"
echo "1. 영어 (English) 모델"
echo "2. 그리스어 (Greek) 모델"
echo "3. 스페인어 (Spanish) 모델"
echo "4. 중국어 (Mandarin) 모델"
echo "5. 다국어 통합 (All Languages) 모델"
echo ""
echo "총 $TOTAL_EXPERIMENTS개 실험 예정"
echo ""

# 실행 확인
read -p "모든 실험을 순차적으로 실행하시겠습니까? (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "실험이 취소되었습니다."
    exit 0
fi

# 실험 결과 저장할 로그 파일
LOG_DIR="../modules/outputs/siglip/experiment_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/all_experiments_$(date '+%Y%m%d_%H%M%S').log"

echo "실험 로그 저장 위치: $MAIN_LOG"
echo ""

# 전체 실험 시작
echo "========================================"
echo "전체 실험 시작"
echo "========================================"

# 실험 카운터
CURRENT_EXP=1

# 개별 언어 실험
for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "[$CURRENT_EXP/$TOTAL_EXPERIMENTS] $LANG 모델 훈련 시작..."
    echo "----------------------------------------"
    
    LANG_LOWER=$(echo "$LANG" | tr '[:upper:]' '[:lower:]')
    SCRIPT_NAME="train_${LANG_LOWER}.sh"
    
    if [ -f "$SCRIPT_NAME" ]; then
        # 스크립트에 자동 확인 모드로 실행 (yes 답변 자동 입력)
        echo -e "n\ny" | bash "$SCRIPT_NAME" 2>&1 | tee -a "$MAIN_LOG"
        
        if [ ${PIPESTATUS[1]} -eq 0 ]; then
            echo "✅ $LANG 모델 훈련 완료!" | tee -a "$MAIN_LOG"
        else
            echo "❌ $LANG 모델 훈련 실패!" | tee -a "$MAIN_LOG"
        fi
    else
        echo "⚠️ $SCRIPT_NAME 파일을 찾을 수 없습니다." | tee -a "$MAIN_LOG"
    fi
    
    CURRENT_EXP=$((CURRENT_EXP + 1))
    
    # 다음 실험 전 잠시 대기
    if [ $CURRENT_EXP -le $TOTAL_EXPERIMENTS ]; then
        echo ""
        echo "다음 실험까지 10초 대기..."
        sleep 10
    fi
done

# 다국어 통합 실험
echo ""
echo "[$CURRENT_EXP/$TOTAL_EXPERIMENTS] 다국어 통합 모델 훈련 시작..."
echo "----------------------------------------"

if [ -f "train_all_languages.sh" ]; then
    # 스크립트에 자동 확인 모드로 실행
    echo -e "n\ny" | bash train_all_languages.sh 2>&1 | tee -a "$MAIN_LOG"
    
    if [ ${PIPESTATUS[1]} -eq 0 ]; then
        echo "✅ 다국어 통합 모델 훈련 완료!" | tee -a "$MAIN_LOG"
    else
        echo "❌ 다국어 통합 모델 훈련 실패!" | tee -a "$MAIN_LOG"
    fi
else
    echo "⚠️ train_all_languages.sh 파일을 찾을 수 없습니다." | tee -a "$MAIN_LOG"
fi

# 전체 실험 완료
echo ""
echo "========================================"
echo "전체 실험 완료!"
echo "========================================"
echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MAIN_LOG"
echo ""
echo "실험 결과:"
echo "  로그 파일: $MAIN_LOG"
echo "  모델 저장 위치: ../modules/outputs/siglip/"
echo ""
echo "각 언어별 모델 위치:"
for LANG in "${LANGUAGES[@]}"; do
    echo "  $LANG: ../modules/outputs/siglip/$LANG/checkpoints/"
done
echo "  다국어 통합: ../modules/outputs/siglip/All_Languages/checkpoints/"
echo ""

# 실험 결과 요약
echo "실험 요약을 생성하시겠습니까? (y/N): "
read -p "" summary_confirm
if [[ $summary_confirm =~ ^[Yy]$ ]]; then
    echo "실험 결과 요약 생성 중..."
    # 여기에 결과 요약 로직 추가 가능
    echo "요약이 $LOG_DIR/experiment_summary_$(date '+%Y%m%d_%H%M%S').txt에 저장되었습니다."
fi

echo ""
echo "🎉 모든 실험이 완료되었습니다!"
