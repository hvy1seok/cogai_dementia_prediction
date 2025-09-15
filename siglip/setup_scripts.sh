#!/bin/bash
# SigLIP2 훈련 스크립트들에 실행 권한 부여

echo "=== SigLIP2 훈련 스크립트 설정 ==="

# 실행 권한 부여할 스크립트 목록
SCRIPTS=(
    "train_english.sh"
    "train_greek.sh"
    "train_spanish.sh"
    "train_mandarin.sh"
    "train_all_languages.sh"
    "run_all_experiments.sh"
    "setup_scripts.sh"
)

echo "실행 권한을 부여할 스크립트들:"
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        echo "✅ $script - 실행 권한 부여 완료"
    else
        echo "⚠️ $script - 파일을 찾을 수 없음"
    fi
done

echo ""
echo "🎉 모든 스크립트 설정 완료!"
echo ""
echo "사용 방법:"
echo "  개별 언어 훈련:"
echo "    ./train_english.sh    # 영어"
echo "    ./train_greek.sh      # 그리스어"
echo "    ./train_spanish.sh    # 스페인어"
echo "    ./train_mandarin.sh   # 중국어"
echo ""
echo "  다국어 통합 훈련:"
echo "    ./train_all_languages.sh"
echo ""
echo "  모든 실험 순차 실행:"
echo "    ./run_all_experiments.sh"
echo ""
echo "  데이터 파서 테스트:"
echo "    python test_parser.py"
