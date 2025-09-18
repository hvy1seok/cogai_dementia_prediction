# Control Groups for Dementia Prediction Experiments
# 치매 예측 실험을 위한 대조군 모델들

"""
대조군 모델 구성:
1. ViT-Spec (Audio-only): 오디오만 사용하는 멀티링궐 모델
2. Text-only (Gemma Encoder): 텍스트만 사용하는 멀티링궐 모델  
3. Concat (ViT + XLM-R): Late Fusion 방식의 멀티모달 모델
"""

__version__ = "1.0.0"
__author__ = "Dementia Prediction Research Team"
