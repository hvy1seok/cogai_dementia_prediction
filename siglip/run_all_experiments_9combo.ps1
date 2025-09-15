# SigLIP2 다국어 치매 진단 모델 - 9가지 조합 실험
# 3가지 손실 함수 × 3가지 옵티마이저 = 9가지 조합

Write-Host "=== SigLIP2 다국어 모델 9가지 조합 실험 시작 ===" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 공통 설정
$DATA_DIR = "../../training_dset"
$BASE_OUTPUT_DIR = "../modules/outputs/siglip/9_Combo_Experiments"
$MODEL_NAME = "google/siglip2-base-patch16-naflex"
$BATCH_SIZE = 32
$LEARNING_RATE = "2e-5"
$NUM_EPOCHS = 100
$LANGUAGES = @("English", "Greek", "Spanish", "Mandarin")

# 실험 조합 정의
$LOSS_TYPES = @("cross_entropy", "focal", "bce")
$OPTIMIZERS = @("adamw", "lion", "sam")

# 손실 함수 이름 매핑
$LOSS_NAMES = @{
    "cross_entropy" = "CrossEntropy"
    "focal" = "FocalLoss"
    "bce" = "BCE"
}

# 옵티마이저 이름 매핑
$OPT_NAMES = @{
    "adamw" = "AdamW"
    "lion" = "Lion"
    "sam" = "SAM"
}

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

# 실험 카운터
$EXPERIMENT_COUNT = 0
$TOTAL_EXPERIMENTS = $LOSS_TYPES.Count * $OPTIMIZERS.Count
$SUCCESSFUL_EXPERIMENTS = 0
$FAILED_EXPERIMENTS = 0

# 결과 로그 파일
$RESULTS_LOG = "$BASE_OUTPUT_DIR/experiment_results.log"
New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

# 로그 파일 초기화
@"
📊 실험 계획:
총 실험 수: $TOTAL_EXPERIMENTS
손실 함수: $($LOSS_TYPES -join ', ')
옵티마이저: $($OPTIMIZERS -join ', ')
시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
================================

"@ | Out-File -FilePath $RESULTS_LOG -Encoding UTF8

Write-Host "📊 실험 계획:" -ForegroundColor Cyan
Write-Host "  총 실험 수: $TOTAL_EXPERIMENTS"
Write-Host "  손실 함수: $($LOSS_TYPES -join ', ')"
Write-Host "  옵티마이저: $($OPTIMIZERS -join ', ')"
Write-Host ""

# 9가지 조합 실험 실행
foreach ($loss_type in $LOSS_TYPES) {
    foreach ($optimizer in $OPTIMIZERS) {
        $EXPERIMENT_COUNT++
        
        # 실험 이름 생성
        $EXPERIMENT_NAME = "$($LOSS_NAMES[$loss_type])_$($OPT_NAMES[$optimizer])"
        $OUTPUT_DIR = "$BASE_OUTPUT_DIR/$EXPERIMENT_NAME"
        
        Write-Host "🧪 실험 $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS`: $EXPERIMENT_NAME" -ForegroundColor Yellow
        Write-Host "================================"
        Write-Host "  손실 함수: $loss_type"
        Write-Host "  옵티마이저: $optimizer"
        Write-Host "  출력 디렉토리: $OUTPUT_DIR"
        Write-Host "  시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host ""
        
        # 출력 디렉토리 생성
        New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
        New-Item -ItemType Directory -Force -Path "$OUTPUT_DIR/checkpoints" | Out-Null
        
        # 실험 시작 로그
        "실험 $EXPERIMENT_COUNT`: $EXPERIMENT_NAME - 시작 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $RESULTS_LOG -Append -Encoding UTF8
        
        # 훈련 실행
        $START_TIME = Get-Date
        
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
                --loss_type $loss_type `
                --optimizer_type $optimizer `
                --focal_alpha 1.0 `
                --focal_gamma 2.0 `
                --sam_rho 0.05
            
            # 결과 확인
            $END_TIME = Get-Date
            $DURATION = ($END_TIME - $START_TIME).TotalMinutes
            $DURATION_MIN = [math]::Round($DURATION, 1)
            
            if ($LASTEXITCODE -eq 0) {
                $SUCCESSFUL_EXPERIMENTS++
                Write-Host "✅ 실험 $EXPERIMENT_COUNT 성공: $EXPERIMENT_NAME (소요시간: ${DURATION_MIN}분)" -ForegroundColor Green
                "실험 $EXPERIMENT_COUNT`: $EXPERIMENT_NAME - 성공 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (${DURATION_MIN}분)" | Out-File -FilePath $RESULTS_LOG -Append -Encoding UTF8
            } else {
                $FAILED_EXPERIMENTS++
                Write-Host "❌ 실험 $EXPERIMENT_COUNT 실패: $EXPERIMENT_NAME (소요시간: ${DURATION_MIN}분)" -ForegroundColor Red
                "실험 $EXPERIMENT_COUNT`: $EXPERIMENT_NAME - 실패 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (${DURATION_MIN}분)" | Out-File -FilePath $RESULTS_LOG -Append -Encoding UTF8
            }
        }
        catch {
            $FAILED_EXPERIMENTS++
            $END_TIME = Get-Date
            $DURATION = ($END_TIME - $START_TIME).TotalMinutes
            $DURATION_MIN = [math]::Round($DURATION, 1)
            Write-Host "❌ 실험 $EXPERIMENT_COUNT 오류: $EXPERIMENT_NAME - $($_.Exception.Message)" -ForegroundColor Red
            "실험 $EXPERIMENT_COUNT`: $EXPERIMENT_NAME - 오류 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (${DURATION_MIN}분) - $($_.Exception.Message)" | Out-File -FilePath $RESULTS_LOG -Append -Encoding UTF8
        }
        
        Write-Host ""
        Write-Host "진행 상황: $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS 완료" -ForegroundColor Cyan
        Write-Host "성공: $SUCCESSFUL_EXPERIMENTS, 실패: $FAILED_EXPERIMENTS"
        Write-Host ""
        Write-Host "================================"
        Write-Host ""
    }
}

# 최종 결과 요약
Write-Host "🎉 모든 실험 완료!" -ForegroundColor Green
Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
Write-Host "📊 실험 결과 요약:" -ForegroundColor Cyan
Write-Host "  총 실험 수: $TOTAL_EXPERIMENTS"
Write-Host "  성공한 실험: $SUCCESSFUL_EXPERIMENTS"
Write-Host "  실패한 실험: $FAILED_EXPERIMENTS"
Write-Host "  성공률: $([math]::Round($SUCCESSFUL_EXPERIMENTS * 100 / $TOTAL_EXPERIMENTS, 1))%"
Write-Host ""
Write-Host "🏆 결과 저장 위치:" -ForegroundColor Yellow
Write-Host "  메인 디렉토리: $BASE_OUTPUT_DIR"
Write-Host "  결과 로그: $RESULTS_LOG"
Write-Host ""

# 최종 결과를 로그에 기록
@"

================================
최종 결과 요약 - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
총 실험 수: $TOTAL_EXPERIMENTS
성공한 실험: $SUCCESSFUL_EXPERIMENTS
실패한 실험: $FAILED_EXPERIMENTS
성공률: $([math]::Round($SUCCESSFUL_EXPERIMENTS * 100 / $TOTAL_EXPERIMENTS, 1))%
"@ | Out-File -FilePath $RESULTS_LOG -Append -Encoding UTF8

# 성공한 실험들의 베스트 모델 경로 출력
Write-Host ""
Write-Host "🏆 성공한 실험들의 베스트 모델:" -ForegroundColor Green
foreach ($loss_type in $LOSS_TYPES) {
    foreach ($optimizer in $OPTIMIZERS) {
        $EXPERIMENT_NAME = "$($LOSS_NAMES[$loss_type])_$($OPT_NAMES[$optimizer])"
        $CHECKPOINT_DIR = "$BASE_OUTPUT_DIR/$EXPERIMENT_NAME/checkpoints"
        
        if (Test-Path $CHECKPOINT_DIR) {
            $BEST_MODEL = Get-ChildItem -Path $CHECKPOINT_DIR -Filter "*best-auc*.ckpt" | Select-Object -First 1
            if ($BEST_MODEL) {
                Write-Host "  $EXPERIMENT_NAME`: $($BEST_MODEL.Name)" -ForegroundColor White
            }
        }
    }
}

Write-Host ""
Write-Host "🔬 9가지 조합 실험이 모두 완료되었습니다!" -ForegroundColor Green
Write-Host "각 조합의 성능을 비교하여 최적의 설정을 찾아보세요! 🎯" -ForegroundColor Cyan
