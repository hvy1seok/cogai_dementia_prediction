"""
SigLIP2 치매 진단 모델 추론 스크립트
"""
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from PIL import Image
from transformers import Siglip2Processor
import argparse
import json
from typing import Dict, List, Tuple
import os

from config import SigLIPConfig
from data_processor import AudioToMelSpectrogram
from model import SigLIP2DementiaClassifier

class DementiaPredictor:
    """치매 진단 예측기"""
    
    def __init__(self, 
                 model_path: str,
                 config: SigLIPConfig,
                 device: str = "auto"):
        
        self.config = config
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 로드
        print(f"모델 로드 중: {model_path}")
        self.model = SigLIP2DementiaClassifier.load_from_checkpoint(
            model_path,
            map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()
        
        # 프로세서 로드
        self.processor = Siglip2Processor.from_pretrained(config.model_name)
        
        # 오디오 처리기
        self.audio_processor = AudioToMelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            fmin=config.fmin,
            fmax=config.fmax,
            image_size=config.image_size
        )
        
        print(f"모델이 {self.device}에 로드되었습니다.")
    
    def predict_from_audio_and_text(self, 
                                  audio_path: str, 
                                  text: str, 
                                  language: str = "English") -> Dict:
        """오디오와 텍스트로부터 치매 예측"""
        
        # 오디오를 멜스펙토그램으로 변환
        mel_spec = self.audio_processor.audio_to_melspectrogram(audio_path)
        image = self.audio_processor.melspectrogram_to_image(mel_spec)
        
        # 텍스트 전처리
        text = text.lower().strip()
        
        # SigLIP2 프로세서로 처리
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            max_length=self.config.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 디바이스로 이동
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # 언어 ID 변환
        language_ids = self.model._get_language_ids([language])
        if language_ids is not None:
            language_ids = language_ids.to(self.device)
        
        # 예측
        with torch.no_grad():
            logits = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs['pixel_values'],
                language_ids=language_ids
            )
            
            # 확률 계산
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred_class].item()
        
        # 결과 반환
        result = {
            "prediction": "치매" if pred_class == 1 else "정상",
            "confidence": confidence,
            "probabilities": {
                "정상": probs[0][0].item(),
                "치매": probs[0][1].item()
            },
            "language": language,
            "audio_path": audio_path,
            "text": text
        }
        
        return result
    
    def predict_batch(self, 
                     audio_paths: List[str], 
                     texts: List[str], 
                     languages: List[str] = None) -> List[Dict]:
        """배치 예측"""
        
        if languages is None:
            languages = ["English"] * len(audio_paths)
        
        results = []
        for audio_path, text, language in zip(audio_paths, texts, languages):
            try:
                result = self.predict_from_audio_and_text(audio_path, text, language)
                results.append(result)
            except Exception as e:
                print(f"예측 중 오류 발생: {e}")
                results.append({
                    "error": str(e),
                    "audio_path": audio_path,
                    "text": text,
                    "language": language
                })
        
        return results

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="SigLIP2 치매 진단 모델 추론")
    parser.add_argument("--model_path", type=str, required=True, help="훈련된 모델 경로")
    parser.add_argument("--audio_path", type=str, required=True, help="오디오 파일 경로")
    parser.add_argument("--text", type=str, required=True, help="텍스트 전사본")
    parser.add_argument("--language", type=str, default="English", help="언어")
    parser.add_argument("--output", type=str, default=None, help="결과 저장 경로")
    parser.add_argument("--batch_file", type=str, default=None, help="배치 예측용 JSON 파일")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = SigLIPConfig()
    
    # 예측기 생성
    predictor = DementiaPredictor(args.model_path, config)
    
    if args.batch_file:
        # 배치 예측
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        audio_paths = [item['audio_path'] for item in batch_data]
        texts = [item['text'] for item in batch_data]
        languages = [item.get('language', 'English') for item in batch_data]
        
        results = predictor.predict_batch(audio_paths, texts, languages)
        
        # 결과 저장
        output_path = args.output or "batch_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"배치 예측 완료. 결과가 {output_path}에 저장되었습니다.")
        
        # 요약 통계
        successful_predictions = [r for r in results if 'error' not in r]
        if successful_predictions:
            dementia_count = sum(1 for r in successful_predictions if r['prediction'] == '치매')
            normal_count = len(successful_predictions) - dementia_count
            
            print(f"\n=== 예측 요약 ===")
            print(f"총 샘플: {len(results)}")
            print(f"성공: {len(successful_predictions)}")
            print(f"오류: {len(results) - len(successful_predictions)}")
            print(f"치매 예측: {dementia_count}")
            print(f"정상 예측: {normal_count}")
            print(f"치매 비율: {dementia_count/len(successful_predictions)*100:.1f}%")
    
    else:
        # 단일 예측
        result = predictor.predict_from_audio_and_text(
            args.audio_path, args.text, args.language
        )
        
        print("\n=== 치매 진단 결과 ===")
        print(f"예측: {result['prediction']}")
        print(f"신뢰도: {result['confidence']:.3f}")
        print(f"정상 확률: {result['probabilities']['정상']:.3f}")
        print(f"치매 확률: {result['probabilities']['치매']:.3f}")
        print(f"언어: {result['language']}")
        print(f"오디오: {result['audio_path']}")
        print(f"텍스트: {result['text']}")
        
        # 결과 저장
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n결과가 {args.output}에 저장되었습니다.")

if __name__ == "__main__":
    main() 