# SigLIP2 Contrastive Learning - 전체 언어 통합 훈련 (PowerShell)
# 모든 언어를 함께 학습하여 언어 무관 표현 학습 및 cross-modal alignment 최적화

Write-Host "=== SigLIP2 Contrastive Learning - 전체 언어 통합 훈련 시작 ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 설정
$DATA_DIR = "../../training_dset"
$OUTPUT_DIR = "../modules/outputs/siglip/Contrastive_All_Languages"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 100
$LANGUAGES = @("English", "Greek", "Spanish", "Mandarin")

# 손실 함수 + Focal Loss + Contrastive Learning 설정
$LOSS_TYPE = "focal"
$FOCAL_ALPHA = 1.0
$FOCAL_GAMMA = 2.0
$AUTO_CLASS_WEIGHTS = "--auto_class_weights"

# SigLIP2 Contrastive Learning 설정
$USE_CONTRASTIVE = "--use_contrastive"
$CONTRASTIVE_WEIGHT = 0.5  # Classification 50% + Contrastive 50%
$CONTRASTIVE_TEMPERATURE = 0.07

# 출력 디렉토리 생성
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTPUT_DIR/checkpoints" | Out-Null

Write-Host ""
Write-Host "🌍 SigLIP2 Contrastive Learning - 전체 언어 통합 훈련 설정:" -ForegroundColor Cyan
Write-Host "  훈련 언어: $($LANGUAGES -join ' ') (모든 언어 통합)"
Write-Host "  데이터 디렉토리: $DATA_DIR"
Write-Host "  출력 디렉토리: $OUTPUT_DIR"
Write-Host "  모델: $MODEL_NAME"
Write-Host "  배치 크기: $BATCH_SIZE"
Write-Host "  학습률: $LEARNING_RATE"
Write-Host "  에포크 수: $NUM_EPOCHS"
Write-Host "  손실 함수: $LOSS_TYPE + Contrastive"
Write-Host "  Early Stopping: Validation AUC 기준 15 epochs patience"
Write-Host ""
Write-Host "🔗 SigLIP2 Contrastive Learning 특징:" -ForegroundColor Yellow
Write-Host "  ✨ In-batch contrastive learning으로 cross-modal alignment"
Write-Host "  ✨ Same-patient audio-text pairs → positive (유사도 증가)"
Write-Host "  ✨ Different-patient combinations → negative (유사도 감소)"
Write-Host "  ✨ Sigmoid matching (SigLIP2 스타일, CLIP의 softmax 대신)"
Write-Host "  ✨ 언어 무관 representation 학습 강화"
Write-Host "  가중치: Classification $((1-$CONTRASTIVE_WEIGHT)*100)% + Contrastive $($CONTRASTIVE_WEIGHT*100)%"
Write-Host "  온도: $CONTRASTIVE_TEMPERATURE"
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

Write-Host "Python 명령어: $PYTHON_CMD" -ForegroundColor Cyan
Write-Host ""

Write-Host "🚀 SigLIP2 Contrastive Learning 전체 언어 통합 모델 훈련 시작..." -ForegroundColor Green
Write-Host "⚡ 4개 언어의 cross-modal representation을 공동 표현 공간으로 정렬"
Write-Host "================================"

# 훈련 실행
& $PYTHON_CMD trainer.py `
    --data_dir $DATA_DIR `
    --output_dir $OUTPUT_DIR `
    --model_name $MODEL_NAME `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --num_epochs $NUM_EPOCHS `
    --parser all `
    --languages $LANGUAGES `
    --loss_type $LOSS_TYPE `
    --focal_alpha $FOCAL_ALPHA `
    --focal_gamma $FOCAL_GAMMA `
    $AUTO_CLASS_WEIGHTS `
    $USE_CONTRASTIVE `
    --contrastive_weight $CONTRASTIVE_WEIGHT `
    --contrastive_temperature $CONTRASTIVE_TEMPERATURE

# 결과 확인
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ SigLIP2 Contrastive Learning 전체 언어 통합 모델 훈련 완료!" -ForegroundColor Green
    Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    Write-Host ""
    Write-Host "🌍 훈련 완료된 언어: $($LANGUAGES -join ', ')" -ForegroundColor Cyan
    Write-Host "🔗 SigLIP2 Contrastive Learning 효과:" -ForegroundColor Yellow
    Write-Host "   ✅ Cross-modal alignment - 오디오와 텍스트의 의미적 정렬"
    Write-Host "   ✅ Language-agnostic representation - 언어 무관 특징 학습"
    Write-Host "   ✅ Enhanced cross-lingual transfer - 언어 간 전이 능력 향상"
    Write-Host "   ✅ Improved multimodal fusion - 멀티모달 융합 품질 개선"
    Write-Host ""
    Write-Host "📊 결과 분석 포인트:" -ForegroundColor Cyan
    Write-Host "   🔍 Alignment Score: Positive - Negative similarity 차이"
    Write-Host "   🔍 Positive Similarity: 같은 환자 audio-text 평균 유사도"
    Write-Host "   🔍 Negative Similarity: 다른 환자 조합 평균 유사도"
    Write-Host "   🔍 언어별 성능: 각 언어에서의 분류 및 정렬 성능"
    Write-Host ""
    Write-Host "🎯 이 모델의 장점:" -ForegroundColor Green
    Write-Host "   ✨ 모든 언어에서 균형잡힌 성능"
    Write-Host "   ✨ Cross-lingual 일반화 능력 극대화"
    Write-Host "   ✨ Zero-shot 성능 기반 마련"
    Write-Host "   ✨ Robust multimodal representation"
} else {
    Write-Host ""
    Write-Host "❌ SigLIP2 Contrastive Learning 전체 언어 통합 모델 훈련 실패" -ForegroundColor Red
    exit 1
}
