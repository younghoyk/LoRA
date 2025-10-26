#!/usr/bin/env python3
"""
통합 이미지 평가 시스템 (integrated_test_final.py)
코드에 직접 지정된 대표 이미지와 폴더 내의 여러 이미지들을 비교 평가합니다.
각 이미지에 대해 BLIP, CLIP, LPIPS 평가를 수행하고 결과를 통합하여 내보냅니다.
"""

import sys
import torch
from PIL import Image
import os
import time
from datetime import datetime
import json
import csv
import gc
# argparse는 더 이상 필요 없으므로 주석 처리하거나 삭제할 수 있습니다.
# import argparse 

# --- ✅ 여기에서 모든 설정을 직접 수정하세요 ---
# LPIPS 비교의 기준이 될 대표 이미지 경로
REPRESENTATIVE_IMAGE_PATH = "/Users/choimungi/Desktop/3-SV/AIM/evaluate/evaluatedPhoto/image.png"
# 평가할 이미지들이 들어있는 폴더 경로
COMPARE_DIR_PATH = "/Users/choimungi/Desktop/3-SV/AIM/evaluate/compare_folder"
# 결과 파일을 내보낼지 여부와 형식 ('json', 'csv', 또는 None)
EXPORT_FORMAT = 'csv' # 'json', 'csv', None 중 선택
# 결과 파일을 저장할 디렉토리 (None으로 두면 COMPARE_DIR_PATH에 저장됩니다)
OUTPUT_DIR = None

# 프롬프트 설정
ORIGINAL_PROMPT = "A wooden cabin with a smoking chimney stands among snow-covered trees and a snowy mountain in the back."
CLIP_PROMPT = "a detailed photo of a wooden log cabin with a tall stone chimney emitting smoke a thick pristine snow covers the roof snowy pine trees in the background a majestic mountain peak in the background under a clear blue sky a clear natural daylight a sharp focus a winter landscape view"
class IntegratedEvaluator:
    """통합 이미지 평가 클래스"""

    def __init__(self, representative_image_path, compare_dir_path, original_prompt, clip_prompt):
        self.representative_image_path = representative_image_path
        self.compare_dir = compare_dir_path
        self.original_prompt = original_prompt
        self.clip_prompt = clip_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = {}
        self.comparison_images = self._get_image_files()

        print("🚀 통합 이미지 평가 시스템 시작")
        print(f"💻 장치: {self.device}")
        print(f"🖼️ 대표 이미지 (LPIPS 기준): {os.path.basename(self.representative_image_path)}")
        print(f"📂 평가 대상 폴더: {self.compare_dir}")
        print(f"📊 평가할 이미지 수: {len(self.comparison_images)}개")
        print("=" * 80)

        if not self.comparison_images:
            print("⚠️ 평가할 이미지가 폴더에 없습니다. 프로그램을 종료합니다.")
            sys.exit(0)

    def _get_image_files(self):
        """지정된 디렉토리에서 이미지 파일 목록을 가져옵니다."""
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
        images = []
        if not os.path.isdir(self.compare_dir):
            print(f"❌ 오류: '{self.compare_dir}'는 유효한 디렉토리가 아닙니다.")
            return []
        for filename in sorted(os.listdir(self.compare_dir)):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                images.append(os.path.join(self.compare_dir, filename))
        return images

    def evaluate_blip(self, image_path):
        """BLIP-2 모델을 사용한 캡셔닝 + 코사인 유사도 평가 (단일 이미지)"""
        start_time = time.time()
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            from sentence_transformers import SentenceTransformer, util

            print(" 🧠 BLIP 모델 로딩 중...")
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
            ).to(self.device)

            print(" ✍️ 캡션 생성 중...")
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(self.device, torch.float16)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=50)

            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            del model, processor, inputs, generated_ids
            gc.collect()
            torch.cuda.empty_cache()

            print(" 🧮 코사인 유사도 계산 중...")
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
            embedding_caption = embedding_model.encode(caption, convert_to_tensor=True)
            embedding_prompt = embedding_model.encode(self.original_prompt, convert_to_tensor=True)
            cosine_score = util.pytorch_cos_sim(embedding_caption, embedding_prompt).item()

            return {
                'score': cosine_score, 'caption': caption, 'time': time.time() - start_time,
                'status': 'success', 'description': 'BLIP-2 캡셔닝 + 코사인 유사도'
            }
        except Exception as e:
            return {
                'score': None, 'caption': None, 'time': time.time() - start_time,
                'status': 'error', 'error': str(e), 'description': 'BLIP-2 캡셔닝 + 코사인 유사도'
            }

    def evaluate_clip(self, image_path):
        """CLIP 모델을 사용한 이미지-텍스트 유사도 평가 (단일 이미지)"""
        start_time = time.time()
        try:
            import clip

            print(" 🎯 CLIP 모델 로딩 중...")
            model, preprocess = clip.load("ViT-B/32", device=self.device)

            image = Image.open(image_path)
            image_input = preprocess(image).unsqueeze(0).to(self.device)
            tokens = clip.tokenize(self.clip_prompt, truncate=False).to(self.device)
            
            text_tokens = tokens[0, 1:torch.where(tokens[0] == 49407)[0][0]]
            chunks = [text_tokens[i:i + 75] for i in range(0, len(text_tokens), 75)]
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features_list = []
            for chunk in chunks:
                chunk_with_sot_eot = torch.cat([
                    torch.tensor([49406], device=self.device), chunk, torch.tensor([49407], device=self.device)
                ])
                padded_chunk = torch.zeros(77, dtype=torch.long, device=self.device)
                padded_chunk[:len(chunk_with_sot_eot)] = chunk_with_sot_eot
                with torch.no_grad():
                    features = model.encode_text(padded_chunk.unsqueeze(0))
                    text_features_list.append(features)
            
            if text_features_list:
                text_features_avg = torch.stack(text_features_list).mean(dim=0)
                text_features_avg /= text_features_avg.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features_avg.T).item()
            else:
                similarity = 0.0

            return {
                'score': similarity, 'time': time.time() - start_time, 'status': 'success',
                'description': 'CLIP 이미지-텍스트 유사도'
            }
        except Exception as e:
            return {
                'score': None, 'time': time.time() - start_time, 'status': 'error',
                'error': str(e), 'description': 'CLIP 이미지-텍스트 유사도'
            }

    def evaluate_lpips(self, image_to_compare_path):
        """LPIPS 모델을 사용한 지각적 유사도 평가 (대표 이미지와 비교)"""
        start_time = time.time()
        try:
            # sys.path.append(os.path.join(os.path.dirname(__file__), "PerceptualSimilarity"))
            import lpips
            import torchvision.transforms as transforms

            print(" 🔍 LPIPS 모델 로딩 중...")
            loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            image1 = Image.open(self.representative_image_path).convert('RGB')
            image2 = Image.open(image_to_compare_path).convert('RGB')
            
            image1_tensor = transform(image1).unsqueeze(0).to(self.device)
            image2_tensor = transform(image2).unsqueeze(0).to(self.device)

            with torch.no_grad():
                lpips_distance = loss_fn.forward(image1_tensor, image2_tensor).item()

            converted_score = max(0, 1.0 - lpips_distance)
            similarity_percentage = converted_score * 100

            return {
                'score': lpips_distance, 'converted_score': converted_score,
                'similarity_percentage': similarity_percentage, 'time': time.time() - start_time,
                'status': 'success',
                'description': f'vs {os.path.basename(self.representative_image_path)}',
                'image1': os.path.basename(self.representative_image_path),
                'image2': os.path.basename(image_to_compare_path)
            }
        except Exception as e:
            return {
                'score': None, 'converted_score': None, 'similarity_percentage': None,
                'time': time.time() - start_time, 'status': 'error', 'error': str(e),
                'description': 'LPIPS 지각적 유사도',
                'image1': os.path.basename(self.representative_image_path),
                'image2': os.path.basename(image_to_compare_path)
            }

    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "=" * 90)
        print("📊 통합 평가 결과 요약")
        print("=" * 90)
        print(f"🖼️ 대표 이미지: {os.path.basename(self.representative_image_path)}")
        print(f"📝 원본 프롬프트: {self.original_prompt}")
        print(f"📝 CLIP 프롬프트: {self.clip_prompt[:60]}...")
        print(f"⏰ 평가 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for img_name, results in self.results.items():
            print("\n" + "-" * 90)
            print(f"📁 평가 대상: {img_name}")
            print("-" * 90)
            print(f"{'모델':<10} {'상태':<10} {'점수':<20} {'설명':<30} {'소요시간':<10}")
            print("-" * 90)

            for model, result in results.items():
                status = "성공" if result['status'] == 'success' else "실패"
                score_str = "N/A"
                if result.get('score') is not None:
                    if model == 'LPIPS':
                        score_str = f"{result['score']:.4f} ({result.get('similarity_percentage', 0):.1f}%)"
                    else:
                        score_str = f"{result['score']:.4f}"
                
                description = result['description'][:30]
                time_taken = f"{result['time']:.2f}s"
                print(f"{model:<10} {status:<10} {score_str:<20} {description:<30} {time_taken:<10}")

            if 'BLIP' in results and results['BLIP'].get('caption'):
                print(f"  🤖 BLIP 캡션: '{results['BLIP']['caption']}'")
        
        print("\n" + "=" * 90)
        print("🎉 모든 평가가 완료되었습니다!")

    def export_results(self, format='json', output_dir=None):
        """결과를 파일로 내보내기"""
        if output_dir is None:
            output_dir = self.compare_dir

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"evaluation_results_{timestamp}"
        
        export_data = {
            'metadata': {
                'representative_image_path': self.representative_image_path,
                'comparison_directory': self.compare_dir,
                'original_prompt': self.original_prompt,
                'clip_prompt': self.clip_prompt,
                'evaluation_time': datetime.now().isoformat(),
                'device': self.device
            },
            'results': self.results
        }

        if format and format.lower() == 'json':
            output_file = os.path.join(output_dir, f"{base_filename}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"📄 JSON 결과 저장: {output_file}")

        elif format and format.lower() == 'csv':
            output_file = os.path.join(output_dir, f"{base_filename}.csv")
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'Model', 'Score', 'Status', 'Time_sec', 'Description', 'Additional_Info'])
                
                for img_name, results in self.results.items():
                    for model, result in results.items():
                        additional_info = ""
                        if model == 'BLIP' and result.get('caption'):
                            additional_info = f"Caption: {result['caption']}"
                        elif model == 'LPIPS' and result.get('converted_score') is not None:
                            additional_info = f"Similarity: {result['converted_score']:.4f} ({result.get('similarity_percentage', 0):.1f}%)"
                        
                        writer.writerow([
                            img_name, model, result.get('score', 'N/A'), result['status'],
                            f"{result['time']:.2f}", result['description'], additional_info
                        ])
            print(f"📊 CSV 결과 저장: {output_file}")
        
        return output_dir

    def run_evaluation(self, export_format=None, output_dir=None):
        """전체 평가 실행"""
        total_images = len(self.comparison_images)
        for i, image_path in enumerate(self.comparison_images):
            img_name = os.path.basename(image_path)
            print(f"\n{'='*20} 이미지 평가 ({i+1}/{total_images}): {img_name} {'='*20}")
            
            image_results = {}
            
            
            # 1. BLIP 평가
            print("\n📱 BLIP 평가 시작...")
            blip_res = self.evaluate_blip(image_path)
            image_results['BLIP'] = blip_res
            print(f"✅ BLIP 완료: {blip_res.get('score', 'N/A'):.4f} (소요시간: {blip_res['time']:.2f}초)")
            gc.collect(); torch.cuda.empty_cache()
            

            # 2. CLIP 평가
            print("\n🎯 CLIP 평가 시작...")
            clip_res = self.evaluate_clip(image_path)
            image_results['CLIP'] = clip_res
            print(f"✅ CLIP 완료: {clip_res.get('score', 'N/A'):.4f} (소요시간: {clip_res['time']:.2f}초)")
            gc.collect(); torch.cuda.empty_cache()
            
            
            # 3. LPIPS 평가
            print("\n🔍 LPIPS 평가 시작...")
            lpips_res = self.evaluate_lpips(image_path)
            image_results['LPIPS'] = lpips_res
            print(f"✅ LPIPS 완료: {lpips_res.get('score', 'N/A'):.4f} (소요시간: {lpips_res['time']:.2f}초)")
            gc.collect(); torch.cuda.empty_cache()

            self.results[img_name] = image_results
        

        self.print_summary()
        
        if export_format:
            return self.export_results(format=export_format, output_dir=output_dir)
        
        return self.results

# --- ▼▼▼ main 함수 수정됨 ▼▼▼ ---
def main():
    """메인 함수"""
    try:
        # argparse를 사용하지 않고, 코드 상단의 변수를 직접 사용합니다.
        evaluator = IntegratedEvaluator(
            representative_image_path=REPRESENTATIVE_IMAGE_PATH,
            compare_dir_path=COMPARE_DIR_PATH,
            original_prompt=ORIGINAL_PROMPT,
            clip_prompt=CLIP_PROMPT
        )
        # 코드 상단에 설정된 EXPORT_FORMAT과 OUTPUT_DIR 변수를 사용해 실행합니다.
        evaluator.run_evaluation(export_format=EXPORT_FORMAT, output_dir=OUTPUT_DIR)

        if EXPORT_FORMAT:
            print(f"\n✅ 평가 완료! 결과가 {EXPORT_FORMAT.upper()} 형식으로 저장되었습니다.")
        else:
            print(f"\n✅ 평가 완료!")

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 평가가 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()