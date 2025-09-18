#!/bin/bash
# 모든 대조군(Control Groups) 모델 훈련 실행
# Must (본문 표에 포함) - 최소·충분 세트

echo "=== 치매 예측 대조군 모델 전체 실행 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 실행 선택
if [ $# -eq 0 ]; then
    echo "사용법: $0 <모델번호>"
    echo "대조군 모델 목록:"
    echo "  1: Audio-only (ViT-Spec) - 오디오만 사용"
    echo "  2: Text-only (Gemma Encoder) - 텍스트만 사용"
    echo "  3: Concat (ViT + XLM-R) - Late Fusion"
    echo "  all: 모든 대조군 순차 실행"
    exit 1
fi

MODEL_NUM=$1

# 모델별 실행
case $MODEL_NUM in
    1)
        echo "🎵 Audio-only (ViT-Spec) 모델 훈련 시작..."
        echo "========================================"
        bash run_audio_only_en_cn.sh
        ;;
    2)
        echo "📝 Text-only (Gemma Encoder) 모델 훈련 시작..."
        echo "========================================"
        bash run_text_only_en_cn.sh
        ;;
    3)
        echo "🔗 Concat (ViT + XLM-R) Late Fusion 모델 훈련 시작..."
        echo "========================================"
        bash run_concat_en_cn.sh
        ;;
    all)
        echo "🔄 모든 대조군 모델 순차 실행..."
        echo ""
        
        # 1. Audio-only 모델
        echo "========================================"
        echo "🎵 1/3: Audio-only (ViT-Spec) 모델 훈련..."
        echo "========================================"
        bash run_audio_only_en_cn.sh
        if [ $? -ne 0 ]; then
            echo "❌ Audio-only 모델 훈련 실패"
            exit 1
        fi
        echo ""
        
        # 2. Text-only 모델
        echo "========================================"
        echo "📝 2/3: Text-only (Gemma Encoder) 모델 훈련..."
        echo "========================================"
        bash run_text_only_en_cn.sh
        if [ $? -ne 0 ]; then
            echo "❌ Text-only 모델 훈련 실패"
            exit 1
        fi
        echo ""
        
        # 3. Concat 모델
        echo "========================================"
        echo "🔗 3/3: Concat (ViT + XLM-R) Late Fusion 모델 훈련..."
        echo "========================================"
        bash run_concat_en_cn.sh
        if [ $? -ne 0 ]; then
            echo "❌ Concat 모델 훈련 실패"
            exit 1
        fi
        
        echo ""
        echo "✅ 모든 대조군 모델 훈련 완료!"
        echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "📊 훈련 완료된 대조군 모델들:"
        echo "   🎵 Audio-only (ViT-Spec): 스펙트로그램만으로 치매 진단"
        echo "   📝 Text-only (Gemma): 전사 텍스트만으로 치매 진단"
        echo "   🔗 Concat Late Fusion: 오디오+텍스트 Late Fusion"
        echo ""
        echo "🎯 대조군 실험의 목적:"
        echo "   ✅ SigLIP2의 성능 우위성 검증"
        echo "   ✅ 각 모달리티의 기여도 분석"
        echo "   ✅ 융합 방식의 효과 비교"
        echo "   ✅ 최소·충분 베이스라인 제공"
        echo ""
        echo "📈 비교 분석 항목:"
        echo "   🎯 전체 성능 (Accuracy, Macro F1, AUC)"
        echo "   🌍 언어별 성능 (English, Mandarin)"
        echo "   🏥 임상 적용성 (Speaker-Independent)"
        echo "   ⚡ 훈련 효율성 (수렴 속도, 안정성)"
        ;;
    *)
        echo "❌ 잘못된 모델 번호: $MODEL_NUM"
        echo "1-3 또는 'all'을 입력하세요."
        exit 1
        ;;
esac

echo ""
echo "🎉 대조군 모델 실행 완료!"
echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 대조군 모델 특징 요약:"
echo ""
echo "🎵 Audio-only (ViT-Spec):"
echo "   • 스펙트로그램만 사용하는 순수 오디오 모델"
echo "   • ViT (Vision Transformer)로 음성 패턴 학습"
echo "   • 언어 독립적인 음향 특징 기반 진단"
echo ""
echo "📝 Text-only (Gemma Encoder):"
echo "   • 전사 텍스트만 사용하는 순수 언어 모델"
echo "   • Gemma (256K vocab)로 다국어 텍스트 이해"
echo "   • 언어학적 패턴과 의미 정보 기반 진단"
echo ""
echo "🔗 Concat (ViT + XLM-R) Late Fusion:"
echo "   • 오디오와 텍스트를 후반부에서 융합"
echo "   • 각 모달리티가 독립적으로 특징 학습"
echo "   • 단순하지만 효과적인 멀티모달 접근"
echo ""
echo "🎯 이 대조군들은 SigLIP2 모델의 성능을 평가하는"
echo "   최소·충분(Must) 베이스라인으로 사용됩니다."
