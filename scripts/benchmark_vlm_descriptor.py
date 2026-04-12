#!/usr/bin/env python3
"""
Benchmark garment descriptor generation across MiniCPM-V and Gemma 4.

This script is intended for direct model-output comparison using the same
garment image and the same prompt contract.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import requests
import torch
from PIL import Image

from config.prompts import get_minicpm_garment_prompt
from core.minicpm_runner import MiniCPMVRunner


@dataclass
class BenchmarkResult:
    model: str
    prompt_words: int
    image_fetch_s: float
    model_load_s: float
    gen1_s: float
    gen2_s: float
    output1_words: int
    output2_words: int
    output1: str
    output2: str
    extra: dict[str, Any]


def now() -> float:
    return time.perf_counter()


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def word_count(text: str) -> int:
    return len([part for part in str(text or "").split() if part.strip()])


def load_image(image_url: str) -> tuple[Image.Image, float]:
    start = now()
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image, now() - start


def run_minicpm(image: Image.Image, prompt: str, garment_type: str, max_new_tokens: int) -> BenchmarkResult:
    prev_max_tokens = os.getenv("MINICPM_GARMENT_MAX_NEW_TOKENS")
    prev_min_words = os.getenv("MINICPM_GARMENT_MIN_WORDS")
    os.environ["MINICPM_GARMENT_MAX_NEW_TOKENS"] = str(max_new_tokens)
    os.environ.setdefault("MINICPM_GARMENT_MIN_WORDS", "40")
    try:
        runner = MiniCPMVRunner()
        load_t0 = now()
        runner.ensure_ready()
        load_s = now() - load_t0

        gen1_t0 = now()
        out1 = normalize_text(runner.describe_garment(image, garment_type=garment_type, prompt_override=prompt))
        gen1_s = now() - gen1_t0

        gen2_t0 = now()
        out2 = normalize_text(runner.describe_garment(image, garment_type=garment_type, prompt_override=prompt))
        gen2_s = now() - gen2_t0

        return BenchmarkResult(
            model=runner.model_id,
            prompt_words=word_count(prompt),
            image_fetch_s=0.0,
            model_load_s=load_s,
            gen1_s=gen1_s,
            gen2_s=gen2_s,
            output1_words=word_count(out1),
            output2_words=word_count(out2),
            output1=out1,
            output2=out2,
            extra={
                "device": runner.device,
                "attn_implementation": runner.attn_implementation,
                "max_new_tokens": max_new_tokens,
            },
        )
    finally:
        if prev_max_tokens is None:
            os.environ.pop("MINICPM_GARMENT_MAX_NEW_TOKENS", None)
        else:
            os.environ["MINICPM_GARMENT_MAX_NEW_TOKENS"] = prev_max_tokens

        if prev_min_words is None:
            os.environ.pop("MINICPM_GARMENT_MIN_WORDS", None)
        else:
            os.environ["MINICPM_GARMENT_MIN_WORDS"] = prev_min_words


def run_gemma(
    image: Image.Image,
    prompt: str,
    model_id: str,
    image_seq_length: int,
    max_new_tokens: int,
) -> BenchmarkResult:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"

    load_t0 = now()
    processor = AutoProcessor.from_pretrained(model_id, image_seq_length=image_seq_length)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_s = now() - load_t0

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def generate_once() -> tuple[float, str]:
        start = now()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = now() - start
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return elapsed, normalize_text(decoded)

    gen1_s, out1 = generate_once()
    gen2_s, out2 = generate_once()

    return BenchmarkResult(
        model=model_id,
        prompt_words=word_count(prompt),
        image_fetch_s=0.0,
        model_load_s=load_s,
        gen1_s=gen1_s,
        gen2_s=gen2_s,
        output1_words=word_count(out1),
        output2_words=word_count(out2),
        output1=out1,
        output2=out2,
        extra={
            "device": str(model.device),
            "attn_implementation": attn_impl,
            "image_seq_length": image_seq_length,
            "max_new_tokens": max_new_tokens,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VLM garment descriptor output.")
    parser.add_argument("--image-url", required=True, help="Public image URL to benchmark.")
    parser.add_argument("--garment-type", default="top", help="Garment type hint: top, bottom, dress, outer.")
    parser.add_argument("--prompt", default="", help="Prompt override. Defaults to current prompt config.")
    parser.add_argument("--gemma-model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--gemma-image-seq-length", type=int, default=1120)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["minicpm", "gemma"],
        choices=["minicpm", "gemma"],
        help="Models to benchmark.",
    )
    args = parser.parse_args()

    prompt = args.prompt.strip() or get_minicpm_garment_prompt(args.garment_type)
    image, image_fetch_s = load_image(args.image_url)

    results: list[BenchmarkResult] = []
    for model_name in args.models:
        if model_name == "minicpm":
            result = run_minicpm(
                image=image,
                prompt=prompt,
                garment_type=args.garment_type,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            result = run_gemma(
                image=image,
                prompt=prompt,
                model_id=args.gemma_model_id,
                image_seq_length=args.gemma_image_seq_length,
                max_new_tokens=args.max_new_tokens,
            )
        result.image_fetch_s = image_fetch_s
        results.append(result)

    payload = {
        "image_url": args.image_url,
        "garment_type": args.garment_type,
        "prompt": prompt,
        "prompt_words": word_count(prompt),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
