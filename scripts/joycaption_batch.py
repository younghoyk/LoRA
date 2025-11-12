#!/usr/bin/env python3
"""
JoyCaption 배치 캡셔닝 스크립트
이미지 폴더를 입력받아 각 이미지에 대해 캡션을 생성하고 .txt 파일로 저장합니다.
"""

import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse
from tqdm import tqdm

# 설정
MODEL_NAME = "fancyfeast/joy-caption-alpha-two"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 캡셔닝 프롬프트 (스타일 단어 제외 - 잔차 학습 전략)
CAPTION_PROMPT = """Write a detailed description of this image. Focus on:
- The physical objects and their arrangement
- Colors (be specific, e.g., "cerulean blue" instead of just "blue")
- Lighting and shadows
- Textures and materials
- Spatial relationships between elements

Do NOT mention:
- Art style terms (painting, brushwork, impressionist, etc.)
- Camera or photography terms (bokeh, macro, etc.)
- Artistic interpretations or emotions

Describe only what you physically see in concrete, objective terms."""


def load_model():
    """모델 로드"""
    print(f"모델 로딩 중: {MODEL_NAME}")
    print(f"디바이스: {DEVICE}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )

    if DEVICE == "cpu":
        model = model.to(DEVICE)

    return processor, model


def generate_caption(image_path, processor, model):
    """단일 이미지에 대해 캡션 생성"""
    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            text=CAPTION_PROMPT,
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        caption = processor.decode(output[0], skip_special_tokens=True)

        # 프롬프트 부분 제거
        if CAPTION_PROMPT in caption:
            caption = caption.replace(CAPTION_PROMPT, "").strip()

        return caption

    except Exception as e:
        print(f"오류 발생 ({image_path}): {e}")
        return None


def process_folder(input_dir, output_dir=None):
    """폴더 내 모든 이미지 처리"""
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    # 이미지 파일 목록
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    image_files = [
        f for f in os.listdir(input_dir)
        if any(f.lower().endswith(ext) for ext in valid_extensions)
    ]

    if not image_files:
        print(f"이미지 파일이 없습니다: {input_dir}")
        return

    print(f"처리할 이미지 수: {len(image_files)}")

    # 모델 로드
    processor, model = load_model()

    # 배치 처리
    for filename in tqdm(image_files, desc="캡셔닝 진행"):
        image_path = os.path.join(input_dir, filename)

        # 출력 파일명 (.txt)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        # 이미 존재하면 스킵
        if os.path.exists(output_path):
            print(f"스킵 (이미 존재): {output_path}")
            continue

        # 캡션 생성
        caption = generate_caption(image_path, processor, model)

        if caption:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            print(f"저장 완료: {output_path}")

        # 메모리 정리
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("모든 이미지 캡셔닝 완료!")


def main():
    parser = argparse.ArgumentParser(description="JoyCaption 배치 캡셔닝")
    parser.add_argument("--input_dir", type=str, required=True, help="이미지 폴더 경로")
    parser.add_argument("--output_dir", type=str, default=None, help="캡션 저장 경로 (기본: input_dir)")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"오류: 유효한 디렉토리가 아닙니다: {args.input_dir}")
        return

    process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
