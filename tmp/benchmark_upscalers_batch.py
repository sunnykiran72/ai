#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SEED = 1234


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS],
        key=lambda path: path.name.lower(),
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.copy()
    scale = max_side / float(longest)
    resized = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def find_first_existing(patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(Path("/").glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No match for patterns: {patterns}")


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class ModelResult:
    load_seconds: float | None
    settings: dict[str, Any]
    error: str | None = None


class RealESRGANRunner:
    name = "Real-ESRGAN_x4plus"

    def __init__(self) -> None:
        import cv2
        import torch
        import torchvision.transforms._functional_tensor as _functional_tensor

        sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        self.cv2 = cv2
        self.weight_path = find_first_existing(
            [
                "tmp/realesrgan_weights/RealESRGAN_x4plus.pth",
                "tmp/RealESRGAN_x4plus.pth",
            ]
        )
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(self.weight_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            gpu_id=0 if torch.cuda.is_available() else None,
        )
        self.settings = {
            "model_variant": "RealESRGAN_x4plus",
            "native_scale": 4,
            "tile": 0,
            "tile_pad": 10,
            "pre_pad": 0,
            "half": bool(torch.cuda.is_available()),
            "weight_path": str(self.weight_path),
        }

    def run(self, image: Image.Image) -> tuple[Image.Image, float]:
        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        started = time.perf_counter()
        output_bgr, _ = self.upsampler.enhance(bgr, outscale=4)
        infer_seconds = time.perf_counter() - started
        output_rgb = output_bgr[:, :, ::-1]
        return Image.fromarray(output_rgb), infer_seconds


class UltraSharpV2Runner:
    name = "UltraSharpV2"

    def __init__(self) -> None:
        import onnxruntime as ort

        self.model_path = find_first_existing(
            [
                "tmp/ultrasharp/4x-UltraSharpV2_fp32_op17.onnx",
            ]
        )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.settings = {
            "model_variant": "UltraSharpV2_full",
            "native_scale": 4,
            "onnx_path": str(self.model_path),
            "providers": self.session.get_providers(),
        }

    def run(self, image: Image.Image) -> tuple[Image.Image, float]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        started = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        infer_seconds = time.perf_counter() - started
        restored = np.transpose(output[0], (1, 2, 0))
        restored = np.clip(restored, 0.0, 1.0)
        restored = (restored * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(restored), infer_seconds


class AuraSRRunner:
    name = "AuraSR-v2"

    def __init__(self) -> None:
        import torch
        from aura_sr import AuraSR

        self.torch = torch
        self.model_file = find_first_existing(
            [
                "tmp/hf-home-aura/hub/models--fal--AuraSR-v2/snapshots/*/model.safetensors",
                "tmp/hf-home-bella-extra/hub/models--fal--AuraSR-v2/snapshots/*/model.safetensors",
            ]
        )
        self.model = AuraSR.from_pretrained(str(self.model_file))
        self.settings = {
            "model_variant": "AuraSR-v2",
            "native_scale": 4,
            "method": "upscale_4x_overlapped",
            "max_batch_size": 8,
            "weight_type": "checkboard",
            "seed": SEED,
            "weight_path": str(self.model_file),
        }

    def run(self, image: Image.Image) -> tuple[Image.Image, float]:
        self.torch.manual_seed(SEED)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(SEED)
        started = time.perf_counter()
        output = self.model.upscale_4x_overlapped(
            image.convert("RGB"),
            max_batch_size=8,
            weight_type="checkboard",
        )
        infer_seconds = time.perf_counter() - started
        return output.convert("RGB"), infer_seconds


class Swin2SRRunner:
    name = "Swin2SR_realworld_x4"

    def __init__(self) -> None:
        import torch
        import torch.nn.functional as torch_f
        from transformers import Swin2SRForImageSuperResolution

        self.torch = torch
        self.torch_f = torch_f
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = find_first_existing(
            [
                "tmp/hf-home-swin/transformers/models--caidas--swin2SR-realworld-sr-x4-64-bsrgan-psnr/snapshots/*",
                "tmp/hf-home-bella-extra/transformers/models--caidas--swin2SR-realworld-sr-x4-64-bsrgan-psnr/snapshots/*",
            ]
        )
        self.model = Swin2SRForImageSuperResolution.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.window_size = int(getattr(self.model.config, "window_size", 8))
        self.upscale = int(getattr(self.model.config, "upscale", 4))
        self.settings = {
            "model_variant": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
            "native_scale": 4,
            "device": self.device,
            "model_dir": str(self.model_dir),
            "window_size": self.window_size,
        }

    def run(self, image: Image.Image) -> tuple[Image.Image, float]:
        image = image.convert("RGB")
        original_width, original_height = image.size
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        pixel_values = self.torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(self.device)
        pad_height = (self.window_size - original_height % self.window_size) % self.window_size
        pad_width = (self.window_size - original_width % self.window_size) % self.window_size
        if pad_height or pad_width:
            pixel_values = self.torch_f.pad(pixel_values, (0, pad_width, 0, pad_height), mode="reflect")
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(pixel_values)
        infer_seconds = time.perf_counter() - started
        reconstruction = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        reconstruction = np.transpose(reconstruction, (1, 2, 0))
        target_width = original_width * self.upscale
        target_height = original_height * self.upscale
        reconstruction = reconstruction[:target_height, :target_width, :]
        reconstruction = (reconstruction * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(reconstruction), infer_seconds


RUNNER_TYPES = [
    RealESRGANRunner,
    UltraSharpV2Runner,
    AuraSRRunner,
    Swin2SRRunner,
]


def benchmark_model(
    runner_type: type,
    images: list[Path],
    output_dir: Path,
    max_side: int,
    rows: list[dict[str, Any]],
) -> ModelResult:
    started = time.perf_counter()
    try:
        runner = runner_type()
    except Exception as exc:
        return ModelResult(
            load_seconds=None,
            settings={"load_error": str(exc)},
            error=traceback.format_exc(),
        )
    load_seconds = time.perf_counter() - started
    model_output_dir = output_dir / runner.name
    ensure_dir(model_output_dir)
    for image_path in images:
        total_started = time.perf_counter()
        source = Image.open(image_path).convert("RGB")
        input_width, input_height = source.size
        output_path = model_output_dir / f"{image_path.stem}.png"
        try:
            restored, infer_seconds = runner.run(source)
            native_width, native_height = restored.size
            final_image = resize_max_side(restored, max_side)
            output_width, output_height = final_image.size
            final_image.save(output_path, format="PNG", optimize=True)
            rows.append(
                {
                    "image_name": image_path.name,
                    "model": runner.name,
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
                    "model": runner.name,
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
    del runner
    cleanup_cuda()
    return ModelResult(load_seconds=round(load_seconds, 4), settings=runner_type().settings if False else {})  # placeholder


def runner_settings(runner_type: type) -> dict[str, Any]:
    runner = runner_type()
    settings = runner.settings
    del runner
    cleanup_cuda()
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)
    images = iter_images(input_dir)
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")

    rows: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] = {}

    selected_models = set(args.models)
    runner_types = [
        runner_type for runner_type in RUNNER_TYPES
        if not selected_models or runner_type.name in selected_models
    ]
    if not runner_types:
        raise SystemExit(f"No matching models for selection: {sorted(selected_models)}")

    for runner_type in runner_types:
        load_started = time.perf_counter()
        runner = None
        try:
            runner = runner_type()
            load_seconds = round(time.perf_counter() - load_started, 4)
            model_metadata[runner.name] = {
                "status": "ok",
                "load_seconds": load_seconds,
                "settings": {
                    **runner.settings,
                    "postprocess_max_side": args.max_side,
                    "postprocess_resample": "LANCZOS",
                },
            }
            model_output_dir = output_dir / runner.name
            ensure_dir(model_output_dir)
            for image_path in images:
                total_started = time.perf_counter()
                source = Image.open(image_path).convert("RGB")
                input_width, input_height = source.size
                output_path = model_output_dir / f"{image_path.stem}.png"
                try:
                    restored, infer_seconds = runner.run(source)
                    native_width, native_height = restored.size
                    final_image = resize_max_side(restored, args.max_side)
                    output_width, output_height = final_image.size
                    final_image.save(output_path, format="PNG", optimize=True)
                    rows.append(
                        {
                            "image_name": image_path.name,
                            "model": runner.name,
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
                            "model": runner.name,
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
        except Exception as exc:
            load_seconds = round(time.perf_counter() - load_started, 4)
            model_name = getattr(runner_type, "name", runner_type.__name__)
            model_metadata[model_name] = {
                "status": "error",
                "load_seconds": load_seconds,
                "settings": {},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        finally:
            if runner is not None:
                del runner
            cleanup_cuda()

    summary = {}
    for model_name, metadata in model_metadata.items():
        model_rows = [row for row in rows if row["model"] == model_name and row["status"] == "ok"]
        avg_infer = None
        if model_rows:
            avg_infer = round(sum(row["inference_seconds"] for row in model_rows) / len(model_rows), 4)
        summary[model_name] = {
            "status": metadata["status"],
            "load_seconds": metadata["load_seconds"],
            "avg_inference_seconds": avg_infer,
            "completed_images": len(model_rows),
            "failed_images": len([row for row in rows if row["model"] == model_name and row["status"] != "ok"]),
        }

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(images),
        "images": [path.name for path in images],
        "settings": model_metadata,
        "summary": summary,
        "rows": rows,
    }
    write_json(output_dir / "results.json", payload)
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
