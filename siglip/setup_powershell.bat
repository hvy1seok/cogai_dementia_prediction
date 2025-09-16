@echo off
REM SigLIP2 PowerShell 스크립트 설정 및 실행

echo === SigLIP2 PowerShell 환경 설정 ===
echo.

REM PowerShell 실행 정책 확인
echo PowerShell 실행 정책 확인 중...
powershell -Command "Get-ExecutionPolicy"

echo.
echo PowerShell 스크립트 실행을 위해 실행 정책을 설정합니다.
echo 관리자 권한이 필요할 수 있습니다.
echo.

REM 실행 정책 설정
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"

if %ERRORLEVEL% EQU 0 (
    echo ✅ PowerShell 실행 정책 설정 완료
) else (
    echo ⚠️ PowerShell 실행 정책 설정 실패 - 수동으로 설정해주세요
    echo    PowerShell에서 다음 명령어 실행:
    echo    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
)

echo.
echo === 사용 가능한 스크립트 ===
echo 1. 환경 확인:           powershell .\check_environment.ps1
echo 2. 영어 모델 훈련:      powershell .\train_english.ps1
echo 3. 그리스어 모델 훈련:  powershell .\train_greek.ps1
echo 4. 스페인어 모델 훈련:  powershell .\train_spanish.ps1
echo 5. 중국어 모델 훈련:    powershell .\train_mandarin.ps1
echo 6. 다국어 통합 훈련:    powershell .\train_all_languages.ps1
echo 7. 모든 실험 실행:      powershell .\run_all_experiments.ps1
echo.

pause
