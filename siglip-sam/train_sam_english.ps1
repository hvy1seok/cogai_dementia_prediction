# SigLIP-SAM 영어 치매 진단 모델 훈련 (SAM 옵티마이저 사용)

Write-Host "=== SigLIP-SAM 영어 치매 진단 모델 훈련 시작 ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 설정
$DATA_DIR = "../../training_dset"
$OUTPUT_DIR = "../modules/outputs/siglip-sam/English_SAM"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 100
$LANGUAGE = "English"

# SAM 설정
$OPTIMIZER_TYPE = "sam"
$SAM_RHO = 0.05
$LOSS_TYPE = "focal"

# 출력 디렉토리 생성
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTPUT_DIR/checkpoints" | Out-Null

Write-Host ""
Write-Host "🎯 SAM 훈련 설정:" -ForegroundColor Cyan
Write-Host "  언어: $LANGUAGE"
Write-Host "  데이터 디렉토리: $DATA_DIR"
Write-Host "  출력 디렉토리: $OUTPUT_DIR"
Write-Host "  모델: $MODEL_NAME"
Write-Host "  배치 크기: $BATCH_SIZE"
Write-Host "  학습률: $LEARNING_RATE"
Write-Host "  에포크 수: $NUM_EPOCHS"
Write-Host "  옵티마이저: $OPTIMIZER_TYPE (rho=$SAM_RHO)"
Write-Host "  손실 함수: $LOSS_TYPE"
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
Write-Host ""

Write-Host "SAM 영어 모델 훈련 시작..." -ForegroundColor Green
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
        --parser $LANGUAGE `
        --optimizer_type $OPTIMIZER_TYPE `
        --sam_rho $SAM_RHO `
        --loss_type $LOSS_TYPE

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ SAM 영어 모델 훈련이 성공적으로 완료되었습니다!" -ForegroundColor Green
        Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "모델 저장 위치: $OUTPUT_DIR/checkpoints"
        Write-Host ""
        Write-Host "🎯 SAM 옵티마이저로 훈련된 영어 치매 진단 모델" -ForegroundColor Cyan
        Write-Host "   - Sharpness-Aware Minimization으로 더 나은 일반화 성능 기대"
        Write-Host "   - rho=${SAM_RHO}로 설정된 SAM 파라미터"
        Write-Host "   - 영어 데이터에 특화된 성능 분석"
        Write-Host ""
        Write-Host "🔍 영어 모델 분석 인사이트:" -ForegroundColor Cyan
        Write-Host "   ✅ 영어 데이터에서의 최적 성능 확인"
        Write-Host "   ✅ SAM vs 기존 옵티마이저 성능 비교 기준점"
        Write-Host "   ✅ 영어 특화 threshold 최적화"
        Write-Host "   ✅ 환자 단위 분할의 효과 검증"
        Write-Host ""
        Write-Host "📊 결과 확인:" -ForegroundColor Cyan
        Write-Host "   - 영어 데이터에서의 상세 성능 분석"
        Write-Host "   - 최적 threshold 기반 정확한 메트릭"
        Write-Host "   - ROC 곡선 및 Confusion Matrix"
    } else {
        Write-Host ""
        Write-Host "❌ SAM 영어 모델 훈련 중 오류가 발생했습니다." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ 훈련 실행 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
