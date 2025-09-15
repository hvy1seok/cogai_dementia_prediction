# SigLIP2 환경 확인 스크립트 (PowerShell)

Write-Host "=== SigLIP2 환경 확인 ===" -ForegroundColor Green
Write-Host ""

# Python 확인
Write-Host "1. Python 확인:" -ForegroundColor Cyan
$PYTHON_CMD = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "python"
    Write-Host "✅ Python 발견: $(Get-Command python | Select-Object -ExpandProperty Source)" -ForegroundColor Green
    Write-Host "   버전: $(& python --version 2>&1)"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "python3"
    Write-Host "✅ Python3 발견: $(Get-Command python3 | Select-Object -ExpandProperty Source)" -ForegroundColor Green
    Write-Host "   버전: $(& python3 --version 2>&1)"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "py"
    Write-Host "✅ Python Launcher 발견: $(Get-Command py | Select-Object -ExpandProperty Source)" -ForegroundColor Green
    Write-Host "   버전: $(& py --version 2>&1)"
} else {
    Write-Host "❌ Python을 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "   https://www.python.org/downloads/ 에서 Python 3.8+ 설치 필요"
    exit 1
}

Write-Host ""

# pip 확인
Write-Host "2. pip 확인:" -ForegroundColor Cyan
$PIP_CMD = $null
if (Get-Command pip -ErrorAction SilentlyContinue) {
    $PIP_CMD = "pip"
    Write-Host "✅ pip 발견: $(Get-Command pip | Select-Object -ExpandProperty Source)" -ForegroundColor Green
} elseif (Get-Command pip3 -ErrorAction SilentlyContinue) {
    $PIP_CMD = "pip3"
    Write-Host "✅ pip3 발견: $(Get-Command pip3 | Select-Object -ExpandProperty Source)" -ForegroundColor Green
} else {
    Write-Host "⚠️ pip을 찾을 수 없습니다. 패키지 설치가 어려울 수 있습니다." -ForegroundColor Yellow
    $PIP_CMD = $null
}

Write-Host ""

# 필수 패키지 확인
Write-Host "3. 필수 Python 패키지 확인:" -ForegroundColor Cyan
$REQUIRED_PACKAGES = @("torch", "transformers", "pytorch_lightning", "librosa", "pillow", "numpy", "pandas")

foreach ($package in $REQUIRED_PACKAGES) {
    try {
        & $PYTHON_CMD -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $package - 설치됨" -ForegroundColor Green
        } else {
            Write-Host "❌ $package - 미설치" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ $package - 미설치" -ForegroundColor Red
    }
}

Write-Host ""

# 데이터 디렉토리 확인
Write-Host "4. 데이터 디렉토리 확인:" -ForegroundColor Cyan
$DATA_DIR = "../training_dset"
if (Test-Path $DATA_DIR) {
    Write-Host "✅ 데이터 디렉토리 존재: $DATA_DIR" -ForegroundColor Green
    
    # 언어별 디렉토리 확인
    $LANGUAGES = @("English", "Greek", "Spanish", "Mandarin")
    foreach ($lang in $LANGUAGES) {
        $LANG_DIR = "$DATA_DIR/$lang"
        if (Test-Path $LANG_DIR) {
            $TEXTDATA_DIR = "$LANG_DIR/textdata"
            $VOICEDATA_DIR = "$LANG_DIR/voicedata"
            
            if ((Test-Path $TEXTDATA_DIR) -and (Test-Path $VOICEDATA_DIR)) {
                Write-Host "✅ $lang - textdata 및 voicedata 폴더 존재" -ForegroundColor Green
            } else {
                Write-Host "⚠️ $lang - textdata 또는 voicedata 폴더 누락" -ForegroundColor Yellow
            }
        } else {
            Write-Host "❌ $lang - 언어 디렉토리 없음" -ForegroundColor Red
        }
    }
} else {
    Write-Host "❌ 데이터 디렉토리 없음: $DATA_DIR" -ForegroundColor Red
}

Write-Host ""

# GPU 확인
Write-Host "5. GPU 확인:" -ForegroundColor Cyan
try {
    $nvidia_smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidia_smi) {
        Write-Host "✅ NVIDIA GPU 드라이버 설치됨" -ForegroundColor Green
        Write-Host "GPU 정보:"
        & nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | Select-Object -First 1
        
        # PyTorch CUDA 확인
        try {
            $cuda_check = & $PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ PyTorch CUDA 지원 확인됨" -ForegroundColor Green
                Write-Host "   $cuda_check"
            } else {
                Write-Host "⚠️ PyTorch CUDA 지원 확인 실패 (PyTorch가 설치되지 않았거나 CPU 버전일 수 있음)" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "⚠️ PyTorch CUDA 확인 중 오류 발생" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️ NVIDIA GPU 드라이버를 찾을 수 없음 (CPU 모드로 실행됩니다)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ GPU 확인 중 오류 발생" -ForegroundColor Yellow
}

Write-Host ""

# PowerShell 실행 정책 확인
Write-Host "6. PowerShell 실행 정책 확인:" -ForegroundColor Cyan
$execution_policy = Get-ExecutionPolicy
Write-Host "현재 실행 정책: $execution_policy"
if ($execution_policy -eq "Restricted") {
    Write-Host "⚠️ PowerShell 스크립트 실행이 제한되어 있습니다." -ForegroundColor Yellow
    Write-Host "   다음 명령어로 실행 정책을 변경하세요:"
    Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Cyan
} else {
    Write-Host "✅ PowerShell 스크립트 실행 가능" -ForegroundColor Green
}

Write-Host ""

# 권장사항
Write-Host "7. 권장사항:" -ForegroundColor Cyan
Write-Host "   - Python 3.8 이상 권장"
Write-Host "   - GPU 메모리 8GB 이상 권장"
Write-Host "   - 배치 크기는 GPU 메모리에 따라 조정 (8GB: batch_size=4, 16GB: batch_size=8)"
Write-Host "   - PowerShell 5.0 이상 권장"
Write-Host ""

# 환경 설정 완료 확인
Write-Host "=== 환경 확인 완료 ===" -ForegroundColor Green
if ($PYTHON_CMD -and (Test-Path $DATA_DIR)) {
    Write-Host "✅ 기본 환경 설정이 완료되었습니다." -ForegroundColor Green
    Write-Host ""
    Write-Host "다음 단계:"
    if ($PIP_CMD) {
        Write-Host "1. 필요한 패키지 설치: $PIP_CMD install -r requirements.txt"
    }
    Write-Host "2. 데이터 파서 테스트: $PYTHON_CMD test_parser.py"
    Write-Host "3. 모델 훈련 시작:"
    Write-Host "   .\train_english.ps1    # 영어"
    Write-Host "   .\train_greek.ps1      # 그리스어"
    Write-Host "   .\train_spanish.ps1    # 스페인어"
    Write-Host "   .\train_mandarin.ps1   # 중국어"
    Write-Host "   .\train_all_languages.ps1  # 다국어 통합"
} else {
    Write-Host "❌ 환경 설정이 완료되지 않았습니다. 위의 오류를 해결해주세요." -ForegroundColor Red
}

Write-Host ""
Write-Host "PowerShell 스크립트 실행 방법:"
Write-Host "  .\check_environment.ps1      # 환경 확인"
Write-Host "  .\train_english.ps1          # 영어 모델 훈련"
Write-Host "  .\run_all_experiments.ps1    # 모든 실험 실행"
