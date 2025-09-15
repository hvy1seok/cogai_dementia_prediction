#!/bin/bash
# SigLIP2 환경 확인 스크립트

echo "=== SigLIP2 환경 확인 ==="
echo ""

# Python 확인
echo "1. Python 확인:"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Python3 발견: $(which python3)"
    echo "   버전: $($PYTHON_CMD --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✅ Python 발견: $(which python)"
    echo "   버전: $($PYTHON_CMD --version)"
else
    echo "❌ Python을 찾을 수 없습니다."
    echo "   Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip"
    echo "   CentOS/RHEL: sudo yum install python3 python3-pip"
    exit 1
fi

echo ""

# pip 확인
echo "2. pip 확인:"
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
    echo "✅ pip3 발견: $(which pip3)"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
    echo "✅ pip 발견: $(which pip)"
else
    echo "⚠️ pip을 찾을 수 없습니다. 패키지 설치가 어려울 수 있습니다."
    PIP_CMD=""
fi

echo ""

# 필수 패키지 확인
echo "3. 필수 Python 패키지 확인:"
REQUIRED_PACKAGES=("torch" "transformers" "pytorch_lightning" "librosa" "pillow" "numpy" "pandas")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if $PYTHON_CMD -c "import $package" 2>/dev/null; then
        echo "✅ $package - 설치됨"
    else
        echo "❌ $package - 미설치"
    fi
done

echo ""

# 데이터 디렉토리 확인
echo "4. 데이터 디렉토리 확인:"
DATA_DIR="../training_dset"
if [ -d "$DATA_DIR" ]; then
    echo "✅ 데이터 디렉토리 존재: $DATA_DIR"
    
    # 언어별 디렉토리 확인
    LANGUAGES=("English" "Greek" "Spanish" "Mandarin")
    for lang in "${LANGUAGES[@]}"; do
        LANG_DIR="$DATA_DIR/$lang"
        if [ -d "$LANG_DIR" ]; then
            TEXTDATA_DIR="$LANG_DIR/textdata"
            VOICEDATA_DIR="$LANG_DIR/voicedata"
            
            if [ -d "$TEXTDATA_DIR" ] && [ -d "$VOICEDATA_DIR" ]; then
                echo "✅ $lang - textdata 및 voicedata 폴더 존재"
            else
                echo "⚠️ $lang - textdata 또는 voicedata 폴더 누락"
            fi
        else
            echo "❌ $lang - 언어 디렉토리 없음"
        fi
    done
else
    echo "❌ 데이터 디렉토리 없음: $DATA_DIR"
fi

echo ""

# GPU 확인
echo "5. GPU 확인:"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 드라이버 설치됨"
    echo "GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    
    # PyTorch CUDA 확인
    if $PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        echo "✅ PyTorch CUDA 지원 확인됨"
    else
        echo "⚠️ PyTorch CUDA 지원 확인 실패 (PyTorch가 설치되지 않았거나 CPU 버전일 수 있음)"
    fi
else
    echo "⚠️ NVIDIA GPU 드라이버를 찾을 수 없음 (CPU 모드로 실행됩니다)"
fi

echo ""

# 권장사항
echo "6. 권장사항:"
echo "   - Python 3.8 이상 권장"
echo "   - GPU 메모리 8GB 이상 권장"
echo "   - 배치 크기는 GPU 메모리에 따라 조정 (8GB: batch_size=4, 16GB: batch_size=8)"
echo ""

# 환경 설정 완료 확인
echo "=== 환경 확인 완료 ==="
if command -v python3 &> /dev/null && [ -d "$DATA_DIR" ]; then
    echo "✅ 기본 환경 설정이 완료되었습니다."
    echo ""
    echo "다음 단계:"
    echo "1. 필요한 패키지 설치: pip3 install -r requirements.txt"
    echo "2. 데이터 파서 테스트: $PYTHON_CMD test_parser.py"
    echo "3. 모델 훈련 시작: ./train_english.sh (또는 다른 언어)"
else
    echo "❌ 환경 설정이 완료되지 않았습니다. 위의 오류를 해결해주세요."
fi
