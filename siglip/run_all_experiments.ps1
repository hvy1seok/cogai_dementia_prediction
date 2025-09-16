# SigLIP2 모든 언어별 실험 순차 실행 (PowerShell)

Write-Host "========================================" -ForegroundColor Green
Write-Host "SigLIP2 치매 진단 모든 언어별 실험 시작" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
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
    exit 1
}

Write-Host "Python 명령어: $PYTHON_CMD" -ForegroundColor Cyan
Write-Host ""

# 실험할 언어 목록
$LANGUAGES = @("English", "Greek", "Spanish", "Mandarin")
$TOTAL_EXPERIMENTS = $LANGUAGES.Count + 1  # 개별 언어 + 통합 모델

Write-Host "실행할 실험:"
Write-Host "1. 영어 (English) 모델"
Write-Host "2. 그리스어 (Greek) 모델"
Write-Host "3. 스페인어 (Spanish) 모델"
Write-Host "4. 중국어 (Mandarin) 모델"
Write-Host "5. 다국어 통합 (All Languages) 모델"
Write-Host ""
Write-Host "총 $TOTAL_EXPERIMENTS 개 실험 예정"
Write-Host ""

# 자동 실행 시작
Write-Host "모든 실험을 순차적으로 실행합니다..." -ForegroundColor Green

# 실험 결과 저장할 로그 디렉토리
$LOG_DIR = "../modules/outputs/siglip/experiment_logs"
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
}
$MAIN_LOG = "$LOG_DIR/all_experiments_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host "실험 로그 저장 위치: $MAIN_LOG" -ForegroundColor Cyan
Write-Host ""

# 전체 실험 시작
Write-Host "========================================" -ForegroundColor Green
Write-Host "전체 실험 시작" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 실험 카운터
$CURRENT_EXP = 1

# 개별 언어 실험
foreach ($LANG in $LANGUAGES) {
    Write-Host ""
    Write-Host "[$CURRENT_EXP/$TOTAL_EXPERIMENTS] $LANG 모델 훈련 시작..." -ForegroundColor Yellow
    Write-Host "----------------------------------------"
    
    $LANG_LOWER = $LANG.ToLower()
    $SCRIPT_NAME = "train_$LANG_LOWER.ps1"
    
    if (Test-Path $SCRIPT_NAME) {
        try {
            # PowerShell 스크립트 실행 (자동 확인 모드)
            $process = Start-Process -FilePath "powershell.exe" -ArgumentList "-ExecutionPolicy Bypass -File $SCRIPT_NAME" -Wait -PassThru -RedirectStandardOutput "$LOG_DIR/temp_$LANG.log" -RedirectStandardError "$LOG_DIR/temp_${LANG}_error.log"
            
            # 로그 파일 내용을 메인 로그에 추가
            if (Test-Path "$LOG_DIR/temp_$LANG.log") {
                Get-Content "$LOG_DIR/temp_$LANG.log" | Add-Content $MAIN_LOG
                Remove-Item "$LOG_DIR/temp_$LANG.log"
            }
            if (Test-Path "$LOG_DIR/temp_${LANG}_error.log") {
                Get-Content "$LOG_DIR/temp_${LANG}_error.log" | Add-Content $MAIN_LOG
                Remove-Item "$LOG_DIR/temp_${LANG}_error.log"
            }
            
            if ($process.ExitCode -eq 0) {
                Write-Host "✅ $LANG 모델 훈련 완료!" -ForegroundColor Green
                "✅ $LANG 모델 훈련 완료!" | Add-Content $MAIN_LOG
            } else {
                Write-Host "❌ $LANG 모델 훈련 실패!" -ForegroundColor Red
                "❌ $LANG 모델 훈련 실패!" | Add-Content $MAIN_LOG
            }
        } catch {
            Write-Host "❌ $LANG 모델 실행 중 오류: $($_.Exception.Message)" -ForegroundColor Red
            "❌ $LANG 모델 실행 중 오류: $($_.Exception.Message)" | Add-Content $MAIN_LOG
        }
    } else {
        Write-Host "⚠️ $SCRIPT_NAME 파일을 찾을 수 없습니다." -ForegroundColor Yellow
        "⚠️ $SCRIPT_NAME 파일을 찾을 수 없습니다." | Add-Content $MAIN_LOG
    }
    
    $CURRENT_EXP++
    
    # 다음 실험 전 잠시 대기
    if ($CURRENT_EXP -le $TOTAL_EXPERIMENTS) {
        Write-Host ""
        Write-Host "다음 실험까지 10초 대기..." -ForegroundColor Cyan
        Start-Sleep -Seconds 10
    }
}

# 다국어 통합 실험
Write-Host ""
Write-Host "[$CURRENT_EXP/$TOTAL_EXPERIMENTS] 다국어 통합 모델 훈련 시작..." -ForegroundColor Yellow
Write-Host "----------------------------------------"

if (Test-Path "train_all_languages.ps1") {
    try {
        $process = Start-Process -FilePath "powershell.exe" -ArgumentList "-ExecutionPolicy Bypass -File train_all_languages.ps1" -Wait -PassThru -RedirectStandardOutput "$LOG_DIR/temp_all.log" -RedirectStandardError "$LOG_DIR/temp_all_error.log"
        
        # 로그 파일 내용을 메인 로그에 추가
        if (Test-Path "$LOG_DIR/temp_all.log") {
            Get-Content "$LOG_DIR/temp_all.log" | Add-Content $MAIN_LOG
            Remove-Item "$LOG_DIR/temp_all.log"
        }
        if (Test-Path "$LOG_DIR/temp_all_error.log") {
            Get-Content "$LOG_DIR/temp_all_error.log" | Add-Content $MAIN_LOG
            Remove-Item "$LOG_DIR/temp_all_error.log"
        }
        
        if ($process.ExitCode -eq 0) {
            Write-Host "✅ 다국어 통합 모델 훈련 완료!" -ForegroundColor Green
            "✅ 다국어 통합 모델 훈련 완료!" | Add-Content $MAIN_LOG
        } else {
            Write-Host "❌ 다국어 통합 모델 훈련 실패!" -ForegroundColor Red
            "❌ 다국어 통합 모델 훈련 실패!" | Add-Content $MAIN_LOG
        }
    } catch {
        Write-Host "❌ 다국어 통합 모델 실행 중 오류: $($_.Exception.Message)" -ForegroundColor Red
        "❌ 다국어 통합 모델 실행 중 오류: $($_.Exception.Message)" | Add-Content $MAIN_LOG
    }
} else {
    Write-Host "⚠️ train_all_languages.ps1 파일을 찾을 수 없습니다." -ForegroundColor Yellow
    "⚠️ train_all_languages.ps1 파일을 찾을 수 없습니다." | Add-Content $MAIN_LOG
}

# 전체 실험 완료
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "전체 실험 완료!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
"완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $MAIN_LOG
Write-Host ""
Write-Host "실험 결과:"
Write-Host "  로그 파일: $MAIN_LOG"
Write-Host "  모델 저장 위치: ../modules/outputs/siglip/"
Write-Host ""
Write-Host "각 언어별 모델 위치:"
foreach ($LANG in $LANGUAGES) {
    Write-Host "  $LANG`: ../modules/outputs/siglip/$LANG/checkpoints/"
}
Write-Host "  다국어 통합: ../modules/outputs/siglip/All_Languages/checkpoints/"
Write-Host ""

# 실험 결과 요약 자동 생성
Write-Host "실험 결과 요약 생성 중..." -ForegroundColor Yellow
$summary_file = "$LOG_DIR/experiment_summary_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

# 요약 생성
@"
=== SigLIP2 치매 진단 실험 결과 요약 ===
실험 완료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

실험한 언어:
$($LANGUAGES -join ', ')

모델 저장 위치:
$(foreach ($LANG in $LANGUAGES) { "- $LANG`: ../modules/outputs/siglip/$LANG/checkpoints/" })
- 다국어 통합: ../modules/outputs/siglip/All_Languages/checkpoints/

상세 로그: $MAIN_LOG
"@ | Out-File -FilePath $summary_file -Encoding UTF8

Write-Host "요약이 $summary_file 에 저장되었습니다." -ForegroundColor Green

Write-Host ""
Write-Host "🎉 모든 실험이 완료되었습니다!" -ForegroundColor Green
