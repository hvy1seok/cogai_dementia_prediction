# SigLIP2 모든 언어 치매 진단 모델 훈련 - Focal Loss 사용

Write-Host "=== SigLIP2 다국어 치매 진단 모델 훈련 (Focal Loss) 시작 ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 설정
$DATA_DIR = "../../training_dset"
$OUTPUT_DIR = "../modules/outputs/siglip/All_Languages_Focal"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 100
$LANGUAGES = @("English", "Greek", "Spanish", "Mandarin")

# Focal Loss 설정
$LOSS_TYPE = "focal"
$FOCAL_ALPHA = 1.0
$FOCAL_GAMMA = 2.0

# 출력 디렉토리 생성
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTPUT_DIR/checkpoints" | Out-Null

Write-Host ""
Write-Host "🎯 훈련 설정 (Focal Loss):" -ForegroundColor Cyan
Write-Host "  언어: $($LANGUAGES -join ', ')"
Write-Host "  데이터 디렉토리: $DATA_DIR"
Write-Host "  출력 디렉토리: $OUTPUT_DIR"
Write-Host "  모델: $MODEL_NAME"
Write-Host "  배치 크기: $BATCH_SIZE"
Write-Host "  학습률: $LEARNING_RATE"
Write-Host "  에포크 수: $NUM_EPOCHS"
Write-Host "  손실 함수: $LOSS_TYPE"
Write-Host "  Focal Alpha: $FOCAL_ALPHA"
Write-Host "  Focal Gamma: $FOCAL_GAMMA"
Write-Host ""

# Python 명령어 확인
$PYTHON_CMD = $null
if (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "python3"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "python"
} else {
    Write-Host "❌ Python을 찾을 수 없습니다. Python 3.8+ 설치가 필요합니다." -ForegroundColor Red
    exit 1
}

Write-Host "Python 명령어: $PYTHON_CMD"

# 데이터 파서 테스트 자동 실행
Write-Host "데이터 파서 테스트 실행 중..."
& $PYTHON_CMD test_parser.py
Write-Host ""

Write-Host "🎯 다국어 통합 모델 훈련 시작 (Focal Loss)..." -ForegroundColor Green
Write-Host "================================"

# 훈련 실행 (Focal Loss 포함)
try {
    & $PYTHON_CMD trainer.py `
        --data_dir $DATA_DIR `
        --output_dir $OUTPUT_DIR `
        --model_name $MODEL_NAME `
        --batch_size $BATCH_SIZE `
        --learning_rate $LEARNING_RATE `
        --num_epochs $NUM_EPOCHS `
        --parser "all" `
        --languages $LANGUAGES `
        --loss_type $LOSS_TYPE `
        --focal_alpha $FOCAL_ALPHA `
        --focal_gamma $FOCAL_GAMMA

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ 다국어 통합 모델 훈련이 성공적으로 완료되었습니다! (Focal Loss)" -ForegroundColor Green
        Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "모델 저장 위치: $OUTPUT_DIR/checkpoints"
        Write-Host ""
        Write-Host "🎯 훈련 설정:" -ForegroundColor Cyan
        Write-Host "  🌍 훈련된 언어: $($LANGUAGES -join ', ')"
        Write-Host "  📊 손실 함수: $LOSS_TYPE (α=$FOCAL_ALPHA, γ=$FOCAL_GAMMA)"
        Write-Host "  🏆 베스트 모델: AUC 기준 자동 저장"
    } else {
        Write-Host ""
        Write-Host "❌ 다국어 통합 모델 훈련 중 오류가 발생했습니다." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ 훈련 실행 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
