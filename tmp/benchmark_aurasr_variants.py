#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS],
        key=lambda path: path.name.lower(),
    )


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.copy()
    scale = max_side / float(longest)
    return image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.LANCZOS,
    )


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def find_first_existing(patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(Path("/").glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No match for patterns: {patterns}")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AuraVariantRunner:
    def __init__(self) -> None:
        from aura_sr import AuraSR

        self.model_file = find_first_existing(
            [
                "tmp/hf-home-aura/hub/models--fal--AuraSR-v2/snapshots/*/model.safetensors",
                "tmp/hf-home-bella-extra/hub/models--fal--AuraSR-v2/snapshots/*/model.safetensors",
            ]
        )
        started = time.perf_counter()
        self.model = AuraSR.from_pretrained(str(self.model_file))
        self.load_seconds = round(time.perf_counter() - started, 4)

    def run_variant(self, image: Image.Image, seed: int, weight_type: str, max_batch_size: int) -> tuple[Image.Image, float]:
        set_seed(seed)
        started = time.perf_counter()
        output = self.model.upscale_4x_overlapped(
            image.convert("RGB"),
            max_batch_size=max_batch_size,
            weight_type=weight_type,
        )
        infer_seconds = time.perf_counter() - started
        return output.convert("RGB"), infer_seconds


def parse_variants(values: list[str]) -> list[dict[str, Any]]:
    variants = []
    for raw in values:
        seed_text, weight_type = raw.split(":", 1)
        seed = int(seed_text)
        label = f"AuraSR-v2_seed{seed}_{weight_type}"
        variants.append(
            {
                "name": label,
                "seed": seed,
                "weight_type": weight_type,
                "max_batch_size": 8,
            }
        )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="seed:weight_type, for example 1234:checkboard",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)
    images = iter_images(input_dir)
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")

    variants = parse_variants(args.variant)
    runner = AuraVariantRunner()
    rows = []
    settings = {}

    for variant in variants:
        model_name = variant["name"]
        settings[model_name] = {
            "status": "ok",
            "load_seconds": runner.load_seconds,
            "settings": {
                "model_variant": "AuraSR-v2",
                "native_scale": 4,
                "method": "upscale_4x_overlapped",
                "seed": variant["seed"],
                "weight_type": variant["weight_type"],
                "max_batch_size": variant["max_batch_size"],
                "postprocess_max_side": args.max_side,
                "postprocess_resample": "LANCZOS",
                "weight_path": str(runner.model_file),
            },
        }
        output_variant_dir = output_dir / model_name
        ensure_dir(output_variant_dir)
        for image_path in images:
            total_started = time.perf_counter()
            source = Image.open(image_path).convert("RGB")
            input_width, input_height = source.size
            output_path = output_variant_dir / f"{image_path.stem}.png"
            try:
                restored, infer_seconds = runner.run_variant(
                    source,
                    seed=variant["seed"],
                    weight_type=variant["weight_type"],
                    max_batch_size=variant["max_batch_size"],
                )
                native_width, native_height = restored.size
                final_image = resize_max_side(restored, args.max_side)
                output_width, output_height = final_image.size
                final_image.save(output_path, format="PNG", optimize=True)
                rows.append(
                    {
                        "image_name": image_path.name,
                        "model": model_name,
                        "status": "ok",
                        "input_width": input_width,
                        "input_height": input_height,
                        "native_width": native_width,
                        "native_height": native_height,
                        "output_width": output_width,
                        "output_height": output_height,
                        "inference_seconds": round(infer_seconds, 4),
                        "total_seconds": round(time.perf_counter() - total_started, 4),
                        "output_path": str(output_path),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "image_name": image_path.name,
                        "model": model_name,
                        "status": "error",
                        "input_width": input_width,
                        "input_height": input_height,
                        "native_width": 0,
                        "native_height": 0,
                        "output_width": 0,
                        "output_height": 0,
                        "inference_seconds": 0.0,
                        "total_seconds": round(time.perf_counter() - total_started, 4),
                        "output_path": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    cleanup_cuda()

    summary = {}
    for variant in variants:
        model_name = variant["name"]
        ok_rows = [row for row in rows if row["model"] == model_name and row["status"] == "ok"]
        summary[model_name] = {
            "status": "ok",
            "load_seconds": runner.load_seconds,
            "avg_inference_seconds": round(sum(row["inference_seconds"] for row in ok_rows) / len(ok_rows), 4) if ok_rows else None,
            "completed_images": len(ok_rows),
            "failed_images": len([row for row in rows if row["model"] == model_name and row["status"] != "ok"]),
        }

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(images),
        "images": [path.name for path in images],
        "settings": settings,
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_name",
                "model",
                "status",
                "input_width",
                "input_height",
                "native_width",
                "native_height",
                "output_width",
                "output_height",
                "inference_seconds",
                "total_seconds",
                "output_path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
