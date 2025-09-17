# 진정한 SigLIP2 - 3개 언어 통합 훈련 (그리스어 제외)
# 영어, 만다린, 스페인어만을 사용한 다국어 통합 학습
# EMA Teacher-Student + Multi-Loss로 최강 언어 무관 성능

Write-Host "=== 진정한 SigLIP2 - 3개 언어 통합 훈련 시작 ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 기본 설정
$DATA_DIR = "../../training_dset"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 100

# 언어 설정 (그리스어 제외)
$LANGUAGES = @("English", "Spanish", "Mandarin")
$OUTPUT_DIR = "../modules/outputs/siglip/True_SigLIP2_3Languages"

# 출력 디렉토리 생성
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTPUT_DIR/checkpoints" | Out-Null

Write-Host ""
Write-Host "🌍 진정한 SigLIP2 - 3개 언어 통합 훈련 설정:" -ForegroundColor Cyan
Write-Host "  실험명: True_SigLIP2_3Languages"
Write-Host "  언어: $($LANGUAGES -join ', ') (그리스어 제외)"
Write-Host "  데이터 디렉토리: $DATA_DIR"
Write-Host "  출력 디렉토리: $OUTPUT_DIR"
Write-Host "  모델: $MODEL_NAME"
Write-Host "  배치 크기: $BATCH_SIZE"
Write-Host "  학습률: $LEARNING_RATE"
Write-Host "  에포크 수: $NUM_EPOCHS"
Write-Host ""

# True SigLIP2 Multi-Loss 설정
$EMA_MOMENTUM = 0.999
$SILC_WEIGHT = 0.2
$SIGMOID_WEIGHT = 1.0
$LOCA_WEIGHT = 1.0
$CLASSIFICATION_WEIGHT = 1.0
$MASK_RATIO = 0.15
$DECODER_HIDDEN_DIM = 512
$DECODER_NUM_HEADS = 8
$DECODER_NUM_LAYERS = 6
$VOCAB_SIZE = 30522
$MAX_CAPTION_LENGTH = 77

Write-Host "🔥 진정한 SigLIP2 Multi-Loss 구조:" -ForegroundColor Yellow
Write-Host "  🧑‍🏫 EMA Teacher Momentum: $EMA_MOMENTUM"
Write-Host "  📚 SILC/TIPS Loss: $SILC_WEIGHT (Self-Distillation)"
Write-Host "  🔗 Sigmoid Loss: $SIGMOID_WEIGHT (Cross-Modal Contrastive)"
Write-Host "  📝 LoCa Loss: $LOCA_WEIGHT (Caption Generation)"
Write-Host "  🎯 Classification Loss: $CLASSIFICATION_WEIGHT (Dementia Diagnosis)"
Write-Host ""
Write-Host "🔥 3개 언어 통합 특화 설정:" -ForegroundColor Yellow
Write-Host "  ✨ 영어, 만다린, 스페인어의 강력한 언어 무관 표현 학습"
Write-Host "  ✨ EMA Teacher-Student로 안정적 특징 학습"
Write-Host "  ✨ Multi-Loss 통합으로 최고 성능 달성"
Write-Host "  ✨ Auto-Regressive Decoder로 캡션 생성 능력 향상"
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

# 훈련 실행
Write-Host "🚀 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 시작..." -ForegroundColor Green
Write-Host "⏳ Early Stopping: Validation AUC 기준 15 epochs patience"
Write-Host ""

& $PYTHON_CMD true_siglip2_trainer.py `
    --data_dir $DATA_DIR `
    --output_dir $OUTPUT_DIR `
    --model_name $MODEL_NAME `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --num_epochs $NUM_EPOCHS `
    --parser "all" `
    --languages $LANGUAGES `
    --loss_type "cross_entropy" `
    --optimizer_type "adamw" `
    --ema_momentum $EMA_MOMENTUM `
    --silc_weight $SILC_WEIGHT `
    --sigmoid_weight $SIGMOID_WEIGHT `
    --loca_weight $LOCA_WEIGHT `
    --classification_weight $CLASSIFICATION_WEIGHT `
    --mask_ratio $MASK_RATIO `
    --decoder_hidden_dim $DECODER_HIDDEN_DIM `
    --decoder_num_heads $DECODER_NUM_HEADS `
    --decoder_num_layers $DECODER_NUM_LAYERS `
    --vocab_size $VOCAB_SIZE `
    --max_caption_length $MAX_CAPTION_LENGTH

# 결과 확인
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 성공!" -ForegroundColor Green
    Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""
    Write-Host "📊 이 모델은 영어, 만다린, 스페인어 데이터로 훈련되어"
    Write-Host "   3개 언어에서 뛰어난 성능을 보일 것으로 예상됩니다."
    Write-Host "   ⚡ 진정한 SigLIP2의 모든 이점으로 최강 다국어 성능!"
    Write-Host ""
    Write-Host "🔍 3개 언어 통합 분석 인사이트:" -ForegroundColor Cyan
    Write-Host "   ✅ 3개 언어 통합 성능 - 언어 무관 표현 학습"
    Write-Host "   ✅ EMA Teacher vs Student 성능 비교"
    Write-Host "   ✅ Multi-Loss components의 기여도 분석"
    Write-Host "   ✅ 언어별 세부 성능 분석"
    Write-Host ""
    Write-Host "📊 결과 확인:"
    Write-Host "   - 언어별 훈련 성능 분석"
    Write-Host "   - EMA Teacher-Student 각각의 성능"
    Write-Host "   - Multi-Loss 실시간 모니터링 및 기여도 분석"
    Write-Host "   - Cross-modal alignment의 효과"
    Write-Host ""
    Write-Host "🎯 기대 효과:" -ForegroundColor Yellow
    Write-Host "   ✨ 3개 언어의 강력한 언어 무관 표현으로 최고 성능"
    Write-Host "   ✨ 진정한 SigLIP2 아키텍처의 모든 이점 활용"
    Write-Host "   ✨ 영어, 만다린, 스페인어에서 예상되는 뛰어난 성능"
    Write-Host "   ✨ 다른 언어로의 확장 가능성 검증"
} else {
    Write-Host ""
    Write-Host "❌ 진정한 SigLIP2 - 3개 언어 통합 모델 훈련 실패" -ForegroundColor Red
    exit 1
}
