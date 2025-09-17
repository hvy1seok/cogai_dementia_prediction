# 진정한 SigLIP2 - 전체 언어 통합 훈련 (PyTorch Lightning) - PowerShell
# EMA Teacher-Student + Self-Distillation + Caption Generation + Multi-Loss

Write-Host "=== 진정한 SigLIP2 전체 언어 통합 훈련 시작 (PyTorch Lightning) ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow

# 설정
$DATA_DIR = "../../training_dset"
$OUTPUT_DIR = "../modules/outputs/siglip/True_SigLIP2_All_Languages_Lightning"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = 2e-5
$NUM_EPOCHS = 100
$LANGUAGES = @("English", "Greek", "Spanish", "Mandarin")

# Focal Loss 설정
$LOSS_TYPE = "focal"
$FOCAL_ALPHA = 1.0
$FOCAL_GAMMA = 2.0
$AUTO_CLASS_WEIGHTS = "--auto_class_weights"

# 옵티마이저 설정 (PyTorch Lightning에서는 AdamW만 지원)
$OPTIMIZER_TYPE = "adamw"

# True SigLIP2 Multi-Loss 가중치
$EMA_MOMENTUM = 0.999
$SILC_WEIGHT = 0.2      # SILC/TIPS Loss (20%)
$SIGMOID_WEIGHT = 1.0   # Sigmoid Contrastive Loss (100%)
$LOCA_WEIGHT = 1.0      # LoCa Caption Loss (100%)
$CLASSIFICATION_WEIGHT = 1.0  # Classification Loss (100%)

# 출력 디렉토리 생성
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
}
if (-not (Test-Path "$OUTPUT_DIR/checkpoints")) {
    New-Item -ItemType Directory -Path "$OUTPUT_DIR/checkpoints" -Force | Out-Null
}

Write-Host ""
Write-Host "🔥 진정한 SigLIP2 전체 언어 통합 훈련 설정 (PyTorch Lightning):" -ForegroundColor Cyan
Write-Host "  훈련 언어: $($LANGUAGES -join ' ') (모든 언어 통합)" -ForegroundColor White
Write-Host "  데이터 디렉토리: $DATA_DIR" -ForegroundColor White
Write-Host "  출력 디렉토리: $OUTPUT_DIR" -ForegroundColor White
Write-Host "  모델: $MODEL_NAME" -ForegroundColor White
Write-Host "  배치 크기: $BATCH_SIZE" -ForegroundColor White
Write-Host "  학습률: $LEARNING_RATE" -ForegroundColor White
Write-Host "  에포크 수: $NUM_EPOCHS" -ForegroundColor White
Write-Host "  옵티마이저: $OPTIMIZER_TYPE" -ForegroundColor White
Write-Host "  손실 함수: $LOSS_TYPE + Multi-Loss" -ForegroundColor White
Write-Host "  Early Stopping: Validation AUC 기준 15 epochs patience" -ForegroundColor White
Write-Host ""
Write-Host "🎯 진정한 SigLIP2 Multi-Loss 구조:" -ForegroundColor Magenta
Write-Host "  🧑‍🏫 EMA Teacher-Student: momentum=$EMA_MOMENTUM" -ForegroundColor White
Write-Host "  📚 SILC/TIPS Loss: $SILC_WEIGHT (Self-Distillation + Masked Prediction)" -ForegroundColor White
Write-Host "  🔗 Sigmoid Loss: $SIGMOID_WEIGHT (Cross-Modal Contrastive)" -ForegroundColor White
Write-Host "  📝 LoCa Loss: $LOCA_WEIGHT (Caption + Dense Caption + Referring)" -ForegroundColor White
Write-Host "  🎯 Classification Loss: $CLASSIFICATION_WEIGHT (Dementia Diagnosis)" -ForegroundColor White
Write-Host ""
Write-Host "✨ 진정한 SigLIP2 핵심 기능:" -ForegroundColor Green
Write-Host "  ✅ EMA Teacher-Student 구조로 Self-Distillation" -ForegroundColor White
Write-Host "  ✅ Masked Prediction으로 Self-Supervised Learning" -ForegroundColor White
Write-Host "  ✅ Auto-Regressive Decoder로 Caption Generation" -ForegroundColor White
Write-Host "  ✅ Cross-Modal Alignment + Language-Agnostic Learning" -ForegroundColor White
Write-Host "  ✅ Dense Captioning + Referring Expressions" -ForegroundColor White
Write-Host "  ✅ PyTorch Lightning으로 안정적인 훈련" -ForegroundColor White
Write-Host ""

# Python 명령어 확인
$PYTHON_CMD = ""
if (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "python3"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "python"
} else {
    Write-Host "❌ Python을 찾을 수 없습니다. Python 3.8+ 설치가 필요합니다." -ForegroundColor Red
    exit 1
}

Write-Host "Python 명령어: $PYTHON_CMD" -ForegroundColor Yellow
Write-Host ""

Write-Host "🚀 진정한 SigLIP2 전체 언어 통합 모델 훈련 시작 (PyTorch Lightning)..." -ForegroundColor Green
Write-Host "⚡ EMA Teacher-Student + Multi-Loss + Lightning으로 최고 성능 달성!" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Cyan

# 훈련 실행
$arguments = @(
    "true_siglip2_trainer.py"
    "--data_dir", $DATA_DIR
    "--output_dir", $OUTPUT_DIR
    "--model_name", $MODEL_NAME
    "--batch_size", $BATCH_SIZE
    "--learning_rate", $LEARNING_RATE
    "--num_epochs", $NUM_EPOCHS
    "--parser", "all"
    "--languages", ($LANGUAGES -join " ")
    "--optimizer_type", $OPTIMIZER_TYPE
    "--loss_type", $LOSS_TYPE
    "--focal_alpha", $FOCAL_ALPHA
    "--focal_gamma", $FOCAL_GAMMA
    $AUTO_CLASS_WEIGHTS
    "--ema_momentum", $EMA_MOMENTUM
    "--silc_weight", $SILC_WEIGHT
    "--sigmoid_weight", $SIGMOID_WEIGHT
    "--loca_weight", $LOCA_WEIGHT
    "--classification_weight", $CLASSIFICATION_WEIGHT
)

& $PYTHON_CMD @arguments

# 결과 확인
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ 진정한 SigLIP2 전체 언어 통합 모델 훈련 완료! (PyTorch Lightning)" -ForegroundColor Green
    Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
    Write-Host "모델 저장 위치: $OUTPUT_DIR/checkpoints" -ForegroundColor White
    Write-Host ""
    Write-Host "🌍 훈련 완료된 언어: $($LANGUAGES -join ', ')" -ForegroundColor Cyan
    Write-Host "🔥 진정한 SigLIP2 + PyTorch Lightning 핵심 성과:" -ForegroundColor Magenta
    Write-Host "   ✅ EMA Teacher-Student Self-Distillation" -ForegroundColor White
    Write-Host "   ✅ SILC/TIPS Masked Prediction Learning" -ForegroundColor White
    Write-Host "   ✅ Auto-Regressive Caption Generation" -ForegroundColor White
    Write-Host "   ✅ Cross-Modal Sigmoid Contrastive Learning" -ForegroundColor White
    Write-Host "   ✅ LoCa Dense Captioning + Referring Expressions" -ForegroundColor White
    Write-Host "   ✅ Multi-Loss 통합으로 최고 성능 달성" -ForegroundColor White
    Write-Host "   ✅ PyTorch Lightning으로 안정적이고 확장가능한 훈련" -ForegroundColor White
    Write-Host ""
    Write-Host "📊 진정한 SigLIP2 vs 기존 모델 비교:" -ForegroundColor Yellow
    Write-Host "   🆚 기존: 단순 embedding averaging + basic contrastive" -ForegroundColor White
    Write-Host "   🔥 SigLIP2: Teacher-Student + Multi-Loss + Caption Generation" -ForegroundColor White
    Write-Host "   ⚡ Lightning: 자동 체크포인팅, Early Stopping, wandb 로깅" -ForegroundColor White
    Write-Host "   📈 기대 효과: 더 강력한 representation + 더 나은 일반화" -ForegroundColor White
    Write-Host ""
    Write-Host "🎯 이 모델의 혁신적 장점:" -ForegroundColor Green
    Write-Host "   ✨ Self-Supervised Learning으로 robust feature 학습" -ForegroundColor White
    Write-Host "   ✨ Caption generation으로 language understanding 강화" -ForegroundColor White
    Write-Host "   ✨ EMA Teacher로 안정적인 학습" -ForegroundColor White
    Write-Host "   ✨ Multi-Loss로 다각도 최적화" -ForegroundColor White
    Write-Host "   ✨ PyTorch Lightning으로 production-ready 코드" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ 진정한 SigLIP2 전체 언어 통합 모델 훈련 실패" -ForegroundColor Red
    exit 1
}
