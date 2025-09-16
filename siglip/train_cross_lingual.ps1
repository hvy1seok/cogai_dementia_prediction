# SigLIP2 Cross-Lingual 치매 진단 모델 훈련
# 훈련: 영어, 스페인어, 만다린 / 테스트: 그리스어

Write-Host "=== SigLIP2 Cross-Lingual 치매 진단 모델 훈련 시작 ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 설정
$DATA_DIR = "../../training_dset"
$OUTPUT_DIR = "../modules/outputs/siglip/CrossLingual_Train_English_Spanish_Mandarin_Test_Greek"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 100

# Cross-lingual 언어 설정
$TRAIN_LANGUAGES = @("English", "Spanish", "Mandarin")
$TEST_LANGUAGES = @("Greek")

# 출력 디렉토리 생성
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTPUT_DIR/checkpoints" | Out-Null

Write-Host ""
Write-Host "🌍 Cross-Lingual 훈련 설정:" -ForegroundColor Cyan
Write-Host "  훈련 언어: $($TRAIN_LANGUAGES -join ', ')"
Write-Host "  테스트 언어: $($TEST_LANGUAGES -join ', ')"
Write-Host "  데이터 디렉토리: $DATA_DIR"
Write-Host "  출력 디렉토리: $OUTPUT_DIR"
Write-Host "  모델: $MODEL_NAME"
Write-Host "  배치 크기: $BATCH_SIZE"
Write-Host "  학습률: $LEARNING_RATE"
Write-Host "  에포크 수: $NUM_EPOCHS"
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
Write-Host "데이터 파서 테스트 실행 중..." -ForegroundColor Yellow
& $PYTHON_CMD test_parser.py
Write-Host ""

Write-Host "Cross-Lingual 모델 훈련 시작..." -ForegroundColor Green
Write-Host "================================"

# 훈련 실행
try {
    & $PYTHON_CMD trainer.py `
        --data_dir $DATA_DIR `
        --output_dir "../modules/outputs/siglip" `
        --model_name $MODEL_NAME `
        --batch_size $BATCH_SIZE `
        --learning_rate $LEARNING_RATE `
        --num_epochs $NUM_EPOCHS `
        --parser "cross_lingual" `
        --train_languages $TRAIN_LANGUAGES `
        --test_languages $TEST_LANGUAGES `
        --loss_type "cross_entropy" `
        --optimizer_type "adamw"

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Cross-Lingual 모델 훈련이 성공적으로 완료되었습니다!" -ForegroundColor Green
        Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "모델 저장 위치: $OUTPUT_DIR/checkpoints"
        Write-Host ""
        Write-Host "🌍 훈련 언어: $($TRAIN_LANGUAGES -join ', ')" -ForegroundColor Cyan
        Write-Host "🎯 테스트 언어: $($TEST_LANGUAGES -join ', ')" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "📊 이 모델은 $($TRAIN_LANGUAGES -join ', ') 데이터로 훈련되어" -ForegroundColor Yellow
        Write-Host "   $($TEST_LANGUAGES -join ', ') 데이터에서 언어 간 일반화 성능을 평가합니다." -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "❌ Cross-Lingual 모델 훈련 중 오류가 발생했습니다." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "❌ 훈련 실행 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
