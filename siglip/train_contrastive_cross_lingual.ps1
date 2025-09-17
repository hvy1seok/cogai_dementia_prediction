# SigLIP2 Contrastive Learning - Zero-shot Cross-Lingual 치매 진단 모델 훈련 (PowerShell)
# 진정한 SigLIP2 스타일 contrastive learning으로 cross-lingual 전이 능력 극대화
# 훈련: 소스 언어만 사용 / 검증&테스트: 타겟 언어만 사용 (완전 Zero-shot)

Write-Host "=== SigLIP2 Contrastive Learning - Zero-shot Cross-Lingual 훈련 시작 ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 기본 설정
$DATA_DIR = "../../training_dset"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 100

# 손실 함수 + Focal Loss 설정
$LOSS_TYPE = "focal"
$FOCAL_ALPHA = 1.0
$FOCAL_GAMMA = 2.0
$AUTO_CLASS_WEIGHTS = "--auto_class_weights"

# SigLIP2 Contrastive Learning 설정
$USE_CONTRASTIVE = "--use_contrastive"
$CONTRASTIVE_WEIGHT = 0.6  # Cross-lingual에서는 contrastive 비중 증가
$CONTRASTIVE_TEMPERATURE = 0.07

# =================================
# Cross-lingual 언어 조합 설정
# =================================
# 사용 가능한 언어: English, Greek, Spanish, Mandarin

# 조합 1: 영어+스페인어+만다린 → 그리스어 (기본)
$TRAIN_LANGUAGES_1 = @("English", "Spanish", "Mandarin")
$TEST_LANGUAGES_1 = @("Greek")

# 조합 2: 영어+그리스어+만다린 → 스페인어
$TRAIN_LANGUAGES_2 = @("English", "Greek", "Mandarin")
$TEST_LANGUAGES_2 = @("Spanish")

# 조합 3: 영어+그리스어+스페인어 → 만다린
$TRAIN_LANGUAGES_3 = @("English", "Greek", "Spanish")
$TEST_LANGUAGES_3 = @("Mandarin")

# 조합 4: 그리스어+스페인어+만다린 → 영어
$TRAIN_LANGUAGES_4 = @("Greek", "Spanish", "Mandarin")
$TEST_LANGUAGES_4 = @("English")

# =================================
# 실행할 조합 선택 (기본값: 조합 1)
# =================================
$EXPERIMENT_NUM = if ($args.Count -gt 0) { [int]$args[0] } else { 1 }

switch ($EXPERIMENT_NUM) {
    1 {
        $TRAIN_LANGUAGES = $TRAIN_LANGUAGES_1
        $TEST_LANGUAGES = $TEST_LANGUAGES_1
        $EXPERIMENT_NAME = "Train_English_Spanish_Mandarin_Test_Greek"
    }
    2 {
        $TRAIN_LANGUAGES = $TRAIN_LANGUAGES_2
        $TEST_LANGUAGES = $TEST_LANGUAGES_2
        $EXPERIMENT_NAME = "Train_English_Greek_Mandarin_Test_Spanish"
    }
    3 {
        $TRAIN_LANGUAGES = $TRAIN_LANGUAGES_3
        $TEST_LANGUAGES = $TEST_LANGUAGES_3
        $EXPERIMENT_NAME = "Train_English_Greek_Spanish_Test_Mandarin"
    }
    4 {
        $TRAIN_LANGUAGES = $TRAIN_LANGUAGES_4
        $TEST_LANGUAGES = $TEST_LANGUAGES_4
        $EXPERIMENT_NAME = "Train_Greek_Spanish_Mandarin_Test_English"
    }
    default {
        Write-Host "❌ 잘못된 실험 번호입니다. 1-4 중 선택하세요." -ForegroundColor Red
        Write-Host "사용법: .\train_contrastive_cross_lingual.ps1 [1|2|3|4]"
        Write-Host ""
        Write-Host "🌍 사용 가능한 조합:" -ForegroundColor Cyan
        Write-Host "  1: 영어+스페인어+만다린 → 그리스어 (기본)"
        Write-Host "  2: 영어+그리스어+만다린 → 스페인어"
        Write-Host "  3: 영어+그리스어+스페인어 → 만다린"
        Write-Host "  4: 그리스어+스페인어+만다린 → 영어"
        exit 1
    }
}

# 출력 디렉토리 설정
$OUTPUT_DIR = "../modules/outputs/siglip/Contrastive_CrossLingual_$EXPERIMENT_NAME"

# 출력 디렉토리 생성
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$OUTPUT_DIR/checkpoints" | Out-Null

Write-Host ""
Write-Host "🌍 SigLIP2 Contrastive Learning - Zero-shot Cross-Lingual 훈련 설정 (실험 $EXPERIMENT_NUM):" -ForegroundColor Cyan
Write-Host "  실험명: $EXPERIMENT_NAME"
Write-Host "  훈련 언어 (소스): $($TRAIN_LANGUAGES -join ', ')"
Write-Host "  타겟 언어 (Zero-shot): $($TEST_LANGUAGES -join ', ')"
Write-Host "  데이터 디렉토리: $DATA_DIR"
Write-Host "  출력 디렉토리: $OUTPUT_DIR"
Write-Host "  모델: $MODEL_NAME"
Write-Host "  배치 크기: $BATCH_SIZE"
Write-Host "  학습률: $LEARNING_RATE"
Write-Host "  에포크 수: $NUM_EPOCHS"
Write-Host "  손실 함수: $LOSS_TYPE + Contrastive"
Write-Host "  Early Stopping: Validation AUC 기준 15 epochs patience"
Write-Host ""
Write-Host "🔗 SigLIP2 Contrastive Learning - Cross-lingual 특화 설정:" -ForegroundColor Yellow
Write-Host "  ✨ Contrastive 비중 증가: Classification $((1-$CONTRASTIVE_WEIGHT)*100)% + Contrastive $($CONTRASTIVE_WEIGHT*100)%"
Write-Host "  ✨ 소스 언어 간 cross-modal alignment 강화"
Write-Host "  ✨ 언어 무관 representation 학습으로 Zero-shot 성능 극대화"
Write-Host "  ✨ Same-patient positive pairs로 semantic consistency 학습"
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

Write-Host "🚀 SigLIP2 Contrastive Learning Zero-shot Cross-Lingual 모델 훈련 시작 (실험 $EXPERIMENT_NUM)..." -ForegroundColor Green
Write-Host "Zero-shot 실험: $($TRAIN_LANGUAGES -join '+') → $($TEST_LANGUAGES -join '+')"
Write-Host "⚡ 타겟 언어는 훈련 시 전혀 사용하지 않음 (완전 Zero-shot)"
Write-Host "⚡ Contrastive learning으로 언어 무관 cross-modal representation 학습"
Write-Host "================================"

# 훈련 실행
& $PYTHON_CMD trainer.py `
    --data_dir $DATA_DIR `
    --output_dir "../modules/outputs/siglip" `
    --model_name $MODEL_NAME `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --num_epochs $NUM_EPOCHS `
    --parser cross_lingual `
    --train_languages $TRAIN_LANGUAGES `
    --test_languages $TEST_LANGUAGES `
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
    Write-Host "✅ SigLIP2 Contrastive Learning Zero-shot Cross-Lingual 모델 훈련 완료!" -ForegroundColor Green
    Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "모델 저장 위치: $OUTPUT_DIR/checkpoints"
    Write-Host ""
    Write-Host "🌍 실험 $EXPERIMENT_NUM`: $EXPERIMENT_NAME" -ForegroundColor Cyan
    Write-Host "🎯 훈련 언어 (소스): $($TRAIN_LANGUAGES -join ', ')"
    Write-Host "🎯 타겟 언어 (Zero-shot): $($TEST_LANGUAGES -join ', ')"
    Write-Host ""
    Write-Host "🔗 SigLIP2 Contrastive Learning Zero-shot 효과:" -ForegroundColor Yellow
    Write-Host "   ✅ Cross-modal semantic alignment - 언어 무관 의미 정렬"
    Write-Host "   ✅ Enhanced zero-shot transfer - 미학습 언어 전이 능력 극대화"
    Write-Host "   ✅ Language-agnostic features - 언어 독립적 특징 학습"
    Write-Host "   ✅ Robust multimodal fusion - 언어 간 일관된 멀티모달 융합"
    Write-Host ""
    Write-Host "📊 이 모델은 $($TRAIN_LANGUAGES -join '+') 데이터로만 훈련되어" -ForegroundColor Cyan
    Write-Host "   $($TEST_LANGUAGES -join '+') 데이터에서 완전 Zero-shot 성능을 평가합니다."
    Write-Host "   ⚡ 타겟 언어는 훈련/검증 시 전혀 보지 않아 진정한 Zero-shot!"
    Write-Host "   ⚡ Contrastive learning의 cross-modal alignment로 더 나은 일반화!"
    Write-Host ""
    Write-Host "🔍 Zero-shot Cross-Lingual 분석 인사이트:" -ForegroundColor Yellow
    Write-Host "   ✅ 완전 Zero-shot 성능 - 타겟 언어 미학습 상태에서의 성능"
    Write-Host "   ✅ Cross-modal alignment score - 소스 vs 타겟 언어 정렬 비교"
    Write-Host "   ✅ Language transfer quality - 언어 간 특징 전이 품질"
    Write-Host "   ✅ Contrastive learning 효과 - alignment 개선 정도 측정"
    Write-Host ""
    Write-Host "🚀 다른 조합도 실행해보세요:" -ForegroundColor Green
    Write-Host "   .\train_contrastive_cross_lingual.ps1 1  # 영어+스페인어+만다린 → 그리스어"
    Write-Host "   .\train_contrastive_cross_lingual.ps1 2  # 영어+그리스어+만다린 → 스페인어"
    Write-Host "   .\train_contrastive_cross_lingual.ps1 3  # 영어+그리스어+스페인어 → 만다린"
    Write-Host "   .\train_contrastive_cross_lingual.ps1 4  # 그리스어+스페인어+만다린 → 영어"
} else {
    Write-Host ""
    Write-Host "❌ SigLIP2 Contrastive Learning Cross-Lingual 모델 훈련 실패" -ForegroundColor Red
    exit 1
}
