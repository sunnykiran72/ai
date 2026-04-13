#!/usr/bin/env python3
"""
Benchmark Qwen Extract-Outfit runtime across dtypes and output sizes.

Example:
  python3 scripts/benchmark_qwen_extract_runtime.py \
    --image /tmp/test1.png \
    --steps 8 \
    --max-input-edge 512 \
    --max-output-edge 512 \
    --dtypes bf16,fp16 \
    --warmups 1 \
    --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from core.qwen_image_edit_runner import QwenImageEditRunner


DEFAULT_PROMPT = "Extract the clothing and create a flat mockup."


def _resize_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    ratio = float(max_edge) / float(longest)
    target = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    return image.resize(target, Image.Resampling.LANCZOS)


def _fit_to_max_edge_without_upscale(width: int, height: int, max_edge: int) -> Tuple[int, int]:
    w = int(width)
    h = int(height)
    longest = max(w, h)
    edge = int(max_edge)
    if longest > edge:
        ratio = float(edge) / float(longest)
        w = max(1, int(round(w * ratio)))
        h = max(1, int(round(h * ratio)))
    return w, h


def _run_case(
    *,
    dtype_name: str,
    source: Image.Image,
    prompt: str,
    steps: int,
    guidance_scale: float,
    warmups: int,
    runs: int,
    output_max_edge: int,
) -> Dict[str, object]:
    os.environ["QWEN_IMAGE_EDIT_DTYPE"] = str(dtype_name).strip().lower()
    runner = QwenImageEditRunner()

    target_w, target_h = _fit_to_max_edge_without_upscale(source.width, source.height, output_max_edge)
    for _ in range(max(0, int(warmups))):
        _ = runner.run_edit(
            source,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=42,
            output_width=target_w,
            output_height=target_h,
        )

    samples: List[float] = []
    metadata = {}
    for run_idx in range(max(1, int(runs))):
        started = time.time()
        _, meta = runner.run_edit(
            source,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=42 + run_idx,
            output_width=target_w,
            output_height=target_h,
        )
        elapsed = float(time.time() - started)
        samples.append(elapsed)
        metadata = dict(meta or {})

    return {
        "dtype": dtype_name,
        "samples_seconds": [round(v, 4) for v in samples],
        "mean_seconds": round(float(statistics.mean(samples)), 4),
        "median_seconds": round(float(statistics.median(samples)), 4),
        "min_seconds": round(float(min(samples)), 4),
        "max_seconds": round(float(max(samples)), 4),
        "input_size": {"width": int(source.width), "height": int(source.height)},
        "requested_output_size": {"width": int(target_w), "height": int(target_h)},
        "runner_metadata": {
            "device": metadata.get("device"),
            "dtype": metadata.get("dtype"),
            "lora_loaded": metadata.get("lora_loaded"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen Extract-Outfit runtime.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--max-input-edge", type=int, default=512, help="Input resize max edge")
    parser.add_argument("--max-output-edge", type=int, default=512, help="Output max edge")
    parser.add_argument("--dtypes", default="bf16,fp16", help="Comma separated dtypes")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per dtype")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per dtype")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    source = Image.open(image_path).convert("RGB")
    source = _resize_to_max_edge(source, int(args.max_input_edge))
    dtypes = [d.strip().lower() for d in str(args.dtypes).split(",") if d.strip()]
    if not dtypes:
        raise SystemExit("No dtypes provided.")

    report = {
        "image": str(image_path),
        "steps": int(args.steps),
        "guidance_scale": float(args.guidance_scale),
        "max_input_edge": int(args.max_input_edge),
        "max_output_edge": int(args.max_output_edge),
        "dtypes": dtypes,
        "results": [],
    }

    for dtype_name in dtypes:
        result = _run_case(
            dtype_name=dtype_name,
            source=source,
            prompt=str(args.prompt),
            steps=int(args.steps),
            guidance_scale=float(args.guidance_scale),
            warmups=int(args.warmups),
            runs=int(args.runs),
            output_max_edge=int(args.max_output_edge),
        )
        report["results"].append(result)

    ranked = sorted(report["results"], key=lambda item: float(item.get("median_seconds") or 1e9))
    report["recommended_dtype"] = ranked[0]["dtype"] if ranked else ""
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
