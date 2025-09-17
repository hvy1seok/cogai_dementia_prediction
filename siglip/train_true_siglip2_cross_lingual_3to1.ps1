# 진정한 SigLIP2 - Cross-Lingual 훈련: 3개 언어 중 2개로 훈련, 1개로 Zero-shot
# 영어, 만다린, 스페인어 중 다양한 조합으로 Zero-shot 성능 평가
# EMA Teacher-Student + Multi-Loss로 최강 Zero-shot 성능

Write-Host "=== 진정한 SigLIP2 - Cross-Lingual 훈련 시작 (2→1 Zero-shot) ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 기본 설정
$DATA_DIR = "../../training_dset"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 100

# =================================
# Cross-lingual 언어 조합 설정 (3개 언어 중 2→1)
# =================================
# 사용 가능한 언어: English, Mandarin, Spanish (그리스어 제외)

# 조합 1: 영어+만다린 → 스페인어 (기본)
$TRAIN_LANGUAGES_1 = @("English", "Mandarin")
$TEST_LANGUAGES_1 = @("Spanish")

# 조합 2: 영어+스페인어 → 만다린
$TRAIN_LANGUAGES_2 = @("English", "Spanish")
$TEST_LANGUAGES_2 = @("Mandarin")

# 조합 3: 만다린+스페인어 → 영어
$TRAIN_LANGUAGES_3 = @("Mandarin", "Spanish")
$TEST_LANGUAGES_3 = @("English")

# =================================
# 실행할 조합 선택 (기본값: 조합 1)
# =================================
$EXPERIMENT_NUM = if ($args.Count -gt 0) { $args[0] } else { 1 }  # 명령행 인수로 조합 선택 가능

switch ($EXPERIMENT_NUM) {
    1 {
        $TRAIN_LANGUAGES = $TRAIN_LANGUAGES_1
        $TEST_LANGUAGES = $TEST_LANGUAGES_1
        $EXPERIMENT_NAME = "Train_English_Mandarin_Test_Spanish"
    }
    2 {
        $TRAIN_LANGUAGES = $TRAIN_LANGUAGES_2
        $TEST_LANGUAGES = $TEST_LANGUAGES_2
        $EXPERIMENT_NAME = "Train_English_Spanish_Test_Mandarin"
    }
    3 {
        $TRAIN_LANGUAGES = $TRAIN_LANGUAGES_3
        $TEST_LANGUAGES = $TEST_LANGUAGES_3
        $EXPERIMENT_NAME = "Train_Mandarin_Spanish_Test_English"
    }
    default {
        Write-Host "❌ 잘못된 실험 번호입니다. 1-3 중 선택하세요." -ForegroundColor Red
        Write-Host "사용법: .\train_true_siglip2_cross_lingual_3to1.ps1 [1|2|3]"
        Write-Host ""
        Write-Host "🌍 사용 가능한 조합 (그리스어 제외):"
        Write-Host "  1: 영어+만다린 → 스페인어 (기본)"
        Write-Host "  2: 영어+스페인어 → 만다린"
        Write-Host "  3: 만다린+스페인어 → 영어"
        exit 1
    }
}

# 출력 디렉토리 설정
$OUTPUT_DIR = "../modules/outputs/siglip/True_SigLIP2_CrossLingual_$EXPERIMENT_NAME"

# 출력 디렉토리 생성
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTPUT_DIR/checkpoints" | Out-Null

Write-Host ""
Write-Host "🌍 진정한 SigLIP2 - Zero-shot Cross-Lingual 훈련 설정 (실험 $EXPERIMENT_NUM):" -ForegroundColor Cyan
Write-Host "  실험명: $EXPERIMENT_NAME"
Write-Host "  훈련 언어 (소스): $($TRAIN_LANGUAGES -join ', ')"
Write-Host "  타겟 언어 (Zero-shot): $($TEST_LANGUAGES -join ', ')"
Write-Host "  데이터 디렉토리: $DATA_DIR"
Write-Host "  출력 디렉토리: $OUTPUT_DIR"
Write-Host "  모델: $MODEL_NAME"
Write-Host "  배치 크기: $BATCH_SIZE"
Write-Host "  학습률: $LEARNING_RATE"
Write-Host "  에포크 수: $NUM_EPOCHS"
Write-Host ""

# True SigLIP2 Multi-Loss 설정 (Cross-lingual 특화)
$EMA_MOMENTUM = 0.999
$SILC_WEIGHT = 0.3      # Self-Distillation 강화 (30%)
$SIGMOID_WEIGHT = 1.2   # Contrastive 강화 (120%)
$LOCA_WEIGHT = 0.8      # Caption 조정 (80%)
$CLASSIFICATION_WEIGHT = 1.0  # 유지 (100%)
$MASK_RATIO = 0.15
$DECODER_HIDDEN_DIM = 512
$DECODER_NUM_HEADS = 8
$DECODER_NUM_LAYERS = 6
$VOCAB_SIZE = 30522
$MAX_CAPTION_LENGTH = 77

Write-Host "🔥 진정한 SigLIP2 Multi-Loss 구조 (Cross-lingual 특화):" -ForegroundColor Yellow
Write-Host "  🧑‍🏫 EMA Teacher Momentum: $EMA_MOMENTUM"
Write-Host "  📚 SILC/TIPS Loss: $SILC_WEIGHT (Self-Distillation 강화)"
Write-Host "  🔗 Sigmoid Loss: $SIGMOID_WEIGHT (Cross-Modal Contrastive 강화)"
Write-Host "  📝 LoCa Loss: $LOCA_WEIGHT (Caption Generation)"
Write-Host "  🎯 Classification Loss: $CLASSIFICATION_WEIGHT (Dementia Diagnosis)"
Write-Host ""
Write-Host "🔥 Zero-shot Cross-lingual 특화 설정:" -ForegroundColor Yellow
Write-Host "  ✨ 2개 언어로 강력한 언어 무관 표현 학습"
Write-Host "  ✨ Self-Distillation 비중 증가로 일반화 능력 극대화"
Write-Host "  ✨ Contrastive Learning 강화로 Cross-modal alignment 향상"
Write-Host "  ✨ 타겟 언어 완전 Zero-shot 평가 (훈련 시 전혀 미사용)"
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
Write-Host "🚀 진정한 SigLIP2 - Zero-shot Cross-Lingual 모델 훈련 시작..." -ForegroundColor Green
Write-Host "⏳ Early Stopping: Validation AUC 기준 15 epochs patience"
Write-Host ""

& $PYTHON_CMD true_siglip2_trainer.py `
    --data_dir $DATA_DIR `
    --output_dir $OUTPUT_DIR `
    --model_name $MODEL_NAME `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --num_epochs $NUM_EPOCHS `
    --parser "cross_lingual" `
    --train_languages $TRAIN_LANGUAGES `
    --test_languages $TEST_LANGUAGES `
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
    Write-Host "🎉 진정한 SigLIP2 - Zero-shot Cross-Lingual 모델 훈련 성공!" -ForegroundColor Green
    Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""
    Write-Host "📊 이 모델은 $($TRAIN_LANGUAGES -join ', ') 데이터로만 훈련되어"
    Write-Host "   $($TEST_LANGUAGES -join ', ') 데이터에서 완전 Zero-shot 성능을 평가합니다."
    Write-Host "   ⚡ 타겟 언어는 훈련/검증 시 전혀 보지 않아 진정한 Zero-shot!"
    Write-Host "   ⚡ 진정한 SigLIP2의 모든 이점으로 최강 Zero-shot 성능!"
    Write-Host ""
    Write-Host "🔍 Zero-shot Cross-Lingual 분석 인사이트:" -ForegroundColor Cyan
    Write-Host "   ✅ 완전 Zero-shot 성능 - 타겟 언어 미학습 상태에서의 성능"
    Write-Host "   ✅ EMA Teacher vs Student Zero-shot 성능 비교"
    Write-Host "   ✅ Multi-Loss components의 Zero-shot 기여도"
    Write-Host "   ✅ 2개 언어 → 1개 언어 전이 학습 품질 평가"
    Write-Host ""
    Write-Host "📊 결과 확인:"
    Write-Host "   - 소스 언어 훈련 성능 vs 타겟 언어 Zero-shot 성능 비교"
    Write-Host "   - EMA Teacher-Student 각각의 Zero-shot 성능"
    Write-Host "   - Multi-Loss 실시간 모니터링 및 기여도 분석"
    Write-Host "   - Cross-modal alignment의 Zero-shot 전이 효과"
    Write-Host ""
    Write-Host "🎯 기대 효과:" -ForegroundColor Yellow
    Write-Host "   ✨ 2개 언어의 강력한 언어 무관 표현으로 최고 Zero-shot 성능"
    Write-Host "   ✨ 진정한 SigLIP2 아키텍처의 모든 이점 활용"
    Write-Host "   ✨ 타겟 언어에서 예상되는 뛰어난 Zero-shot 성능"
    Write-Host "   ✨ 다양한 언어 조합의 전이 능력 비교 분석"
    Write-Host ""
    Write-Host "🚀 다른 조합도 실행해보세요:" -ForegroundColor Green
    Write-Host "   .\train_true_siglip2_cross_lingual_3to1.ps1 1  # 영어+만다린 → 스페인어"
    Write-Host "   .\train_true_siglip2_cross_lingual_3to1.ps1 2  # 영어+스페인어 → 만다린"
    Write-Host "   .\train_true_siglip2_cross_lingual_3to1.ps1 3  # 만다린+스페인어 → 영어"
} else {
    Write-Host ""
    Write-Host "❌ 진정한 SigLIP2 Cross-Lingual 모델 훈련 실패" -ForegroundColor Red
    exit 1
}
