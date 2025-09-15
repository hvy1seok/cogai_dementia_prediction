# SigLIP2 PowerShell 스크립트 사용법

Windows 환경에서 PowerShell을 사용하여 SigLIP2 치매 진단 모델을 훈련하는 방법입니다.

## 🚀 빠른 시작

### 1. PowerShell 실행 정책 설정

먼저 PowerShell 스크립트 실행을 위해 실행 정책을 설정해야 합니다:

```batch
# 배치 파일로 자동 설정
setup_powershell.bat

# 또는 PowerShell에서 수동 설정
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. 환경 확인

```powershell
.\check_environment.ps1
```

### 3. 모델 훈련

#### 개별 언어 훈련:
```powershell
.\train_english.ps1    # 영어
.\train_greek.ps1      # 그리스어
.\train_spanish.ps1    # 스페인어
.\train_mandarin.ps1   # 중국어
```

#### 다국어 통합 훈련:
```powershell
.\train_all_languages.ps1
```

#### 모든 실험 순차 실행:
```powershell
.\run_all_experiments.ps1
```

## 📁 PowerShell 스크립트 파일 목록

| 파일명 | 설명 |
|--------|------|
| `check_environment.ps1` | 환경 확인 (Python, 패키지, 데이터) |
| `train_english.ps1` | 영어 모델 훈련 |
| `train_greek.ps1` | 그리스어 모델 훈련 |
| `train_spanish.ps1` | 스페인어 모델 훈련 |
| `train_mandarin.ps1` | 중국어 모델 훈련 |
| `train_all_languages.ps1` | 다국어 통합 모델 훈련 |
| `run_all_experiments.ps1` | 모든 실험 순차 실행 |
| `setup_powershell.bat` | PowerShell 환경 설정 |

## 🔧 주요 특징

### 1. **자동 Python 감지**
- `python`, `python3`, `py` 명령어를 자동으로 감지
- Windows 환경에 최적화된 Python 실행

### 2. **컬러 출력**
- 성공/실패/경고를 색상으로 구분
- 진행 상황을 시각적으로 표시

### 3. **오류 처리**
- 각 단계별 오류 확인 및 처리
- 상세한 오류 메시지 제공

### 4. **로그 관리**
- 실험 결과를 자동으로 로그 파일에 저장
- 각 언어별 개별 로그 생성

## 💡 사용 팁

### PowerShell 실행 방법:
```powershell
# 방법 1: PowerShell에서 직접 실행
.\train_english.ps1

# 방법 2: 명령 프롬프트에서 실행
powershell .\train_english.ps1

# 방법 3: 실행 정책과 함께 실행
powershell -ExecutionPolicy Bypass -File .\train_english.ps1
```

### GPU 메모리 부족 시:
```powershell
# 배치 크기를 줄여서 실행하려면 스크립트 내부의 $BATCH_SIZE 값을 수정
# 기본값: 8 → 4 또는 2로 변경
```

### 중단된 실험 재시작:
```powershell
# 특정 언어만 다시 실행
.\train_english.ps1

# 또는 전체 실험 재실행
.\run_all_experiments.ps1
```

## 🐛 문제 해결

### 1. **실행 정책 오류**
```
실행 정책으로 인해 스크립트를 실행할 수 없습니다.
```
**해결책:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. **Python 찾을 수 없음**
```
❌ Python을 찾을 수 없습니다.
```
**해결책:**
- [Python 공식 사이트](https://www.python.org/downloads/)에서 Python 3.8+ 설치
- 설치 시 "Add Python to PATH" 옵션 체크

### 3. **패키지 설치 오류**
```
❌ torch - 미설치
```
**해결책:**
```powershell
pip install -r requirements.txt
# 또는
pip install torch transformers pytorch-lightning librosa pillow numpy pandas
```

### 4. **GPU 메모리 부족**
```
CUDA out of memory
```
**해결책:**
- 스크립트 파일에서 `$BATCH_SIZE` 값을 4 또는 2로 감소
- 또는 `--batch_size 4` 파라미터로 직접 실행

## 📊 실험 결과

실험 완료 후 다음 위치에서 결과를 확인할 수 있습니다:

```
modules/outputs/siglip/
├── English/checkpoints/          # 영어 모델
├── Greek/checkpoints/            # 그리스어 모델
├── Spanish/checkpoints/          # 스페인어 모델
├── Mandarin/checkpoints/         # 중국어 모델
├── All_Languages/checkpoints/    # 다국어 통합 모델
└── experiment_logs/              # 실험 로그
```

## 🎯 다음 단계

1. **모델 평가**: 훈련된 모델의 성능 평가
2. **추론 테스트**: 새로운 데이터로 예측 테스트
3. **하이퍼파라미터 튜닝**: 더 나은 성능을 위한 파라미터 조정

Windows 환경에서 PowerShell을 활용한 효율적인 치매 진단 모델 개발을 시작해보세요! 🚀
