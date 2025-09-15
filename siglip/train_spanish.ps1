# SigLIP2 스페인어 치매 진단 모델 훈련 (PowerShell)

Write-Host "=== SigLIP2 스페인어 치매 진단 모델 훈련 시작 ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 설정
$DATA_DIR = "../training_dset"
$OUTPUT_DIR = "../modules/outputs/siglip/Spanish"
$MODEL_NAME = "google/siglip2-base-patch16-224"
$BATCH_SIZE = 8
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 10
$LANGUAGE = "Spanish"

# 출력 디렉토리 생성
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
}
if (-not (Test-Path "$OUTPUT_DIR/checkpoints")) {
    New-Item -ItemType Directory -Path "$OUTPUT_DIR/checkpoints" -Force | Out-Null
}

Write-Host ""
Write-Host "훈련 설정:"
Write-Host "  언어: $LANGUAGE"
Write-Host "  데이터 디렉토리: $DATA_DIR"
Write-Host "  출력 디렉토리: $OUTPUT_DIR"
Write-Host "  모델: $MODEL_NAME"
Write-Host "  배치 크기: $BATCH_SIZE"
Write-Host "  학습률: $LEARNING_RATE"
Write-Host "  에포크 수: $NUM_EPOCHS"
Write-Host ""

# Python 명령어 확인
$PYTHON_CMD = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "python3"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "py"
} else {
    Write-Host "❌ Python을 찾을 수 없습니다. Python 3.8+ 설치가 필요합니다." -ForegroundColor Red
    Write-Host "   https://www.python.org/downloads/ 에서 Python을 설치해주세요."
    exit 1
}

Write-Host "Python 명령어: $PYTHON_CMD" -ForegroundColor Cyan

# 데이터 파서 테스트 자동 실행
Write-Host "데이터 파서 테스트 실행 중..." -ForegroundColor Yellow
& $PYTHON_CMD test_parser.py
Write-Host ""

Write-Host "스페인어 모델 훈련 시작..." -ForegroundColor Green
Write-Host "================================"

# 훈련 실행
try {
    & $PYTHON_CMD trainer.py `
        --data_dir $DATA_DIR `
        --output_dir $OUTPUT_DIR `
        --model_name $MODEL_NAME `
        --batch_size $BATCH_SIZE `
        --learning_rate $LEARNING_RATE `
        --num_epochs $NUM_EPOCHS `
        --parser $LANGUAGE

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ 스페인어 모델 훈련이 성공적으로 완료되었습니다!" -ForegroundColor Green
        Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    } else {
        Write-Host ""
        Write-Host "❌ 스페인어 모델 훈련 중 오류가 발생했습니다." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ 훈련 실행 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
