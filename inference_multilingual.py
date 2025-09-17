#!/usr/bin/env python3
"""
다국어 멀티모달 치매 진단 모델 추론 스크립트
"""
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path
from transformers import BertTokenizer
import librosa
import cv2

from models_multilingual import MultilingualMultimodalModel
from dataset_multilingual import load_audio_features

def load_model_and_tokenizer(model_path, tokenizer_path, device):
    """
    모델과 토크나이저 로드
    """
    print(f"📦 모델 로드 중: {model_path}")
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 초기화 (체크포인트에서 설정 가져오기)
    model_config = checkpoint.get('model_config', {})
    text_model_type = model_config.get('text_model_type', 1)
    dropout = model_config.get('dropout', 0.3)
    
    model = MultilingualMultimodalModel(
        text_model_type=text_model_type,
        dropout=dropout
    )
    
    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 모델 로드 완료")
    
    # 토크나이저 로드
    print(f"🔤 토크나이저 로드 중: {tokenizer_path}")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print(f"✅ 토크나이저 로드 완료")
    
    return model, tokenizer

def preprocess_text(text, tokenizer, max_seq_len=512):
    """
    텍스트 전처리
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    return encoding['input_ids'], encoding['attention_mask']

def preprocess_audio(audio_path, target_size=(224, 224)):
    """
    오디오 전처리 (멜스펙토그램)
    """
    try:
        # .npy 파일인 경우 직접 로드
        if audio_path.endswith('.npy'):
            audio_features = np.load(audio_path)
            if len(audio_features.shape) == 2:
                # 2D -> 3D (채널 차원 추가)
                audio_features = np.expand_dims(audio_features, axis=0)
            # 3채널로 복사 (RGB)
            if audio_features.shape[0] == 1:
                audio_features = np.repeat(audio_features, 3, axis=0)
        else:
            # 일반 오디오 파일 처리
            audio_features = load_audio_features(audio_path)
        
        # 크기 조정
        if audio_features.shape[1:] != target_size:
            audio_resized = []
            for channel in range(audio_features.shape[0]):
                resized = cv2.resize(audio_features[channel], target_size, 
                                   interpolation=cv2.INTER_LINEAR)
                audio_resized.append(resized)
            audio_features = np.stack(audio_resized, axis=0)
        
        # 정규화
        audio_features = (audio_features - audio_features.mean()) / (audio_features.std() + 1e-8)
        
        return torch.FloatTensor(audio_features).unsqueeze(0)  # 배치 차원 추가
        
    except Exception as e:
        print(f"⚠️ 오디오 처리 실패 {audio_path}: {e}")
        # 빈 특징 반환
        return torch.zeros(1, 3, target_size[0], target_size[1])

def predict_single(model, tokenizer, text, audio_path, device, max_seq_len=512):
    """
    단일 샘플 예측
    """
    # 텍스트 전처리
    input_ids, attention_mask = preprocess_text(text, tokenizer, max_seq_len)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # 오디오 전처리
    audio_features = preprocess_audio(audio_path)
    audio_features = audio_features.to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(audio_features, input_ids, attention_mask)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return {
        'prediction': prediction,
        'prediction_label': '치매' if prediction == 1 else '정상',
        'confidence': confidence,
        'probabilities': {
            '정상': probabilities[0][0].item(),
            '치매': probabilities[0][1].item()
        }
    }

def predict_batch(model, tokenizer, data_list, device, max_seq_len=512):
    """
    배치 예측
    """
    results = []
    
    for i, data in enumerate(data_list):
        text = data['text']
        audio_path = data['audio_path']
        language = data.get('language', 'Unknown')
        
        print(f"예측 중 ({i+1}/{len(data_list)}): {language}")
        
        try:
            result = predict_single(model, tokenizer, text, audio_path, device, max_seq_len)
            result.update({
                'text': text,
                'audio_path': audio_path,
                'language': language
            })
            results.append(result)
            
        except Exception as e:
            print(f"⚠️ 예측 실패: {e}")
            results.append({
                'error': str(e),
                'text': text,
                'audio_path': audio_path,
                'language': language
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="다국어 멀티모달 치매 진단 모델 추론")
    
    # 모델 관련
    parser.add_argument('--model_path', type=str, required=True,
                       help='훈련된 모델 경로 (.pth)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                       help='토크나이저 경로')
    
    # 입력 데이터
    parser.add_argument('--text', type=str, 
                       help='분석할 텍스트 (단일 예측용)')
    parser.add_argument('--audio_path', type=str,
                       help='오디오 파일 경로 (단일 예측용)')
    parser.add_argument('--language', type=str, default='Unknown',
                       help='언어')
    
    # 배치 예측용
    parser.add_argument('--batch_file', type=str,
                       help='배치 예측용 텍스트 파일 (각 줄: text\taudio_path\tlanguage)')
    
    # 기타 설정
    parser.add_argument('--max_seq_len', type=int, default=512,
                       help='최대 시퀀스 길이')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto, cpu, cuda)')
    parser.add_argument('--output_file', type=str,
                       help='결과 저장 파일')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ 사용 디바이스: {device}")
    
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, 
        args.tokenizer_path, 
        device
    )
    
    if args.batch_file:
        # 배치 예측
        print(f"\n📂 배치 파일 로드: {args.batch_file}")
        
        data_list = []
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 2:
                    text = parts[0]
                    audio_path = parts[1]
                    language = parts[2] if len(parts) > 2 else 'Unknown'
                    
                    data_list.append({
                        'text': text,
                        'audio_path': audio_path,
                        'language': language
                    })
                else:
                    print(f"⚠️ 라인 {line_num} 형식 오류: {line}")
        
        print(f"✅ {len(data_list)}개 샘플 로드됨")
        
        # 배치 예측 실행
        results = predict_batch(model, tokenizer, data_list, device, args.max_seq_len)
        
        # 결과 출력
        print(f"\n🎯 배치 예측 결과:")
        print("=" * 80)
        
        correct_predictions = 0
        total_predictions = len(results)
        
        for i, result in enumerate(results, 1):
            if 'error' in result:
                print(f"{i:3d}. 오류: {result['error']}")
            else:
                print(f"{i:3d}. [{result['language']:8s}] {result['prediction_label']} "
                      f"(신뢰도: {result['confidence']:.3f}) - {result['text'][:50]}...")
        
        # 결과 저장
        if args.output_file:
            import json
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과 저장: {args.output_file}")
    
    else:
        # 단일 예측
        if not args.text or not args.audio_path:
            print("❌ 단일 예측을 위해서는 --text와 --audio_path가 필요합니다.")
            return
        
        print(f"\n🎯 단일 예측:")
        print(f"  언어: {args.language}")
        print(f"  텍스트: {args.text}")
        print(f"  오디오: {args.audio_path}")
        
        result = predict_single(
            model, tokenizer, args.text, args.audio_path, device, args.max_seq_len
        )
        
        print(f"\n📊 예측 결과:")
        print("=" * 50)
        print(f"예측: {result['prediction_label']}")
        print(f"신뢰도: {result['confidence']:.3f}")
        print(f"정상 확률: {result['probabilities']['정상']:.3f}")
        print(f"치매 확률: {result['probabilities']['치매']:.3f}")
        
        # 결과 저장
        if args.output_file:
            import json
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과 저장: {args.output_file}")

if __name__ == '__main__':
    main()
