#!/usr/bin/env python3
"""Persistent SeedVR2 3B worker for fast repeated upscaling requests."""

import argparse
import importlib.util
import io
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict


def _load_module(cli_path: Path):
    spec = importlib.util.spec_from_file_location("seedvr2_inference_cli_worker", str(cli_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import SeedVR2 CLI from: {cli_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "debug"):
        module.debug.enabled = False
    return module


def _build_args(module, cli_path: Path, payload: Dict[str, Any], model_name: str):
    model_dir = cli_path.parent / "models" / "SEEDVR2"
    offload_device = str(payload.get("offload_device", "cpu")).strip().lower()
    if offload_device not in {"cpu", "none"}:
        offload_device = "cpu"
    argv = [
        str(cli_path),
        str(payload["input_path"]),
        "--output",
        str(payload["output_path"]),
        "--output_format",
        "png",
        "--dit_model",
        model_name,
        "--model_dir",
        str(model_dir.resolve()),
        "--resolution",
        str(int(payload["resolution"])),
        "--max_resolution",
        str(int(payload["max_resolution"])),
        "--batch_size",
        str(int(payload["batch_size"])),
        "--cache_dit",
        "--cache_vae",
        "--dit_offload_device",
        offload_device,
        "--vae_offload_device",
        offload_device,
        "--tensor_offload_device",
        "cpu",
    ]
    original = list(sys.argv)
    try:
        sys.argv = argv
        return module.parse_arguments()
    finally:
        sys.argv = original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-path", required=True)
    parser.add_argument("--model-name", default="seedvr2_ema_3b_fp8_e4m3fn.safetensors")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cli_path = Path(args.cli_path).resolve()
    startup_capture = io.StringIO()
    with redirect_stdout(startup_capture), redirect_stderr(startup_capture):
        module = _load_module(cli_path)
    model_name = str(args.model_name)
    cache: Dict[str, Any] = {}
    backend = str(module.get_gpu_backend())
    device_list = ["0"] if backend == "cuda" else ["cpu"]

    for raw in sys.stdin:
        line = (raw or "").strip()
        if not line:
            continue
        if line == "__quit__":
            break
        try:
            payload = json.loads(line)
            req_args = _build_args(module, cli_path, payload, model_name)
            capture = io.StringIO()
            start = time.perf_counter()
            with redirect_stdout(capture), redirect_stderr(capture):
                frames = module.process_single_file(
                    str(payload["input_path"]),
                    req_args,
                    device_list=device_list,
                    output_path=str(payload["output_path"]),
                    format_auto_detected=False,
                    runner_cache=cache,
                )
            elapsed = time.perf_counter() - start
            response = {
                "ok": True,
                "frames_written": int(frames),
                "elapsed": float(elapsed),
                "backend": backend,
            }
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
