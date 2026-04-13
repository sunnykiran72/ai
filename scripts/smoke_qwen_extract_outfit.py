#!/usr/bin/env python3
"""
Run one smoke test inference with Qwen Image Edit + Extract-Outfit LoRA.
"""

import argparse
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.qwen_image_edit_runner import QwenImageEditRunner

DEFAULT_PROMPT = (
    "Extract the clothing from the image and convert it into a clean, standalone mockup. "
    "Preserve the original fabric texture, stitching, folds, patterns, and color accuracy. "
    "Remove the model and background completely, keeping the garment's natural shape and proportions intact. "
    "Present the clothing as a flat-lay or neutral mockup on a plain background with even lighting, "
    "maintaining photorealistic detail and sharp edges."
)
DEFAULT_INPUT_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)


def load_image(source: str) -> Image.Image:
    raw = str(source or "").strip()
    if raw.startswith(("http://", "https://")):
        response = requests.get(raw, timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image.convert("RGB")

    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    image = Image.open(path)
    image.load()
    return image.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Qwen Extract-Outfit")
    parser.add_argument("--input", default=DEFAULT_INPUT_URL, help="Local image path or URL")
    parser.add_argument("--output", default="/tmp/qwen_extract_outfit_smoke.png", help="Output PNG path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image = load_image(args.input)
    runner = QwenImageEditRunner()

    start = time.time()
    output, metadata = runner.run_edit(
        image=image,
        prompt=str(args.prompt),
        steps=int(args.steps),
        guidance_scale=(float(args.guidance_scale) if args.guidance_scale is not None else None),
        negative_prompt=args.negative_prompt,
        seed=int(args.seed),
    )
    elapsed = time.time() - start

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path, format="PNG")

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] elapsed_s: {elapsed:.3f}")
    print(f"[ok] metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
