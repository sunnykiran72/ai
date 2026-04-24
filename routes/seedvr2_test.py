"""
Dev SeedVR2 3B quality test routes.

Provides a small upload UI and API endpoint to run SeedVR2 3B inference_cli
for direct quality checks.
"""

import io
import json
import logging
import os
import select
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.responses import FileResponse, HTMLResponse

from routes.models import SuccessResponse, SeedVR2UpscaleUrlRequest
from shared.azure_storage import storage

logger = logging.getLogger("glamify-ai")
router = APIRouter()

_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "seedvr2_api_tests"
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
_SEEDVR2_PERSISTENT_RUNNERS: Dict[str, "SeedVR2PersistentRunner"] = {}
_MODEL_VARIANTS: Dict[str, str] = {
    "3b": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "7b": "seedvr2_ema_7b_fp16.safetensors",
    "7b-fp8": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "7b-fp8-sharp": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
}


def _default_cli_path() -> str:
    return str(
        os.getenv(
            "SEEDVR2_CLI_PATH",
            "/workspace/seedvr2_eval/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py",
        )
    )


def _default_python_bin() -> str:
    explicit = str(os.getenv("SEEDVR2_PYTHON_BIN", "")).strip()
    if explicit and Path(explicit).exists():
        return explicit
    candidates = [
        "/workspace/hybrid_vto_v1_latest_v1/.venv/bin/python",
        "/workspace/hybrid_vto_v1_latest_v1/.venv311/bin/python",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return sys.executable


def _resolve_public_base_url(request: Request) -> str:
    """Build browser-reachable base URL behind reverse proxies."""
    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").strip()
    forwarded_host = (request.headers.get("x-forwarded-host") or "").strip()
    host = (request.headers.get("host") or "").strip()

    proto = forwarded_proto or request.url.scheme or "http"
    netloc = forwarded_host or host or request.url.netloc
    return f"{proto}://{netloc}" if netloc else str(request.base_url).rstrip("/")


def _download_image_url_for_upscale(image_url: str, timeout_seconds: int = 30) -> Image.Image:
    """Download URL image with browser-like headers for broader host compatibility."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(image_url, headers=headers, timeout=max(10, int(timeout_seconds)), stream=False)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


class SeedVR2PersistentRunner:
    """Persistent sidecar worker that keeps SeedVR2 cache warm across requests."""

    def __init__(self, cli_path: Path, python_bin: Path, model_name: str):
        self._cli_path = Path(cli_path).resolve()
        # Keep the venv interpreter path as-is; resolving symlinks would
        # collapse to system python and lose virtualenv site-packages.
        self._python_bin = Path(python_bin)
        self._worker_script = Path(__file__).resolve().parents[1] / "scripts" / "seedvr2_persistent_worker.py"
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._model_name = str(model_name)
        self._warm_calls = 0

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if not self._cli_path.exists():
            raise FileNotFoundError(f"SEEDVR2 CLI not found: {self._cli_path}")
        if not self._python_bin.exists():
            raise FileNotFoundError(f"Python runtime not found: {self._python_bin}")
        if not self._worker_script.exists():
            raise FileNotFoundError(f"Persistent worker script not found: {self._worker_script}")
        self._proc = subprocess.Popen(
            [
                str(self._python_bin),
                str(self._worker_script),
                "--cli-path",
                str(self._cli_path),
                "--model-name",
                str(self._model_name),
            ],
            cwd=str(self._cli_path.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._proc

    def _close_worker(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write("__quit__\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    def _send_request(self, proc: subprocess.Popen, payload: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("SeedVR2 worker IO pipe is unavailable")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], float(timeout_seconds))
        if not ready:
            raise TimeoutError(f"SeedVR2 worker timed out after {timeout_seconds}s")
        line = proc.stdout.readline()
        if not line:
            exit_code = proc.poll()
            stderr_tail = ""
            if proc.stderr is not None:
                try:
                    stderr_tail = (proc.stderr.read() or "")[-800:]
                except Exception:
                    stderr_tail = ""
            raise RuntimeError(
                f"SeedVR2 worker closed stdout unexpectedly (exit={exit_code}). "
                f"python_bin={self._python_bin} "
                f"stderr_tail={stderr_tail}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"SeedVR2 worker returned invalid JSON: {line[:240]}") from exc
        return response

    def run(
        self,
        input_path: Path,
        output_path: Path,
        resolution: int,
        max_resolution: int,
        batch_size: int,
        offload_device: str,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        with self._lock:
            payload = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "resolution": int(resolution),
                "max_resolution": int(max_resolution),
                "batch_size": int(batch_size),
                "offload_device": str(offload_device or "cpu"),
            }
            last_error = None
            for _ in range(2):
                proc = self._ensure_worker()
                try:
                    t0 = time.perf_counter()
                    response = self._send_request(proc, payload, timeout_seconds=timeout_seconds)
                    elapsed = time.perf_counter() - t0
                    if not response.get("ok", False):
                        raise RuntimeError(str(response.get("error", "Unknown worker error")))
                    warm_before = self._warm_calls
                    self._warm_calls += 1
                    return {
                        "elapsed": float(elapsed),
                        "frames_written": int(response.get("frames_written", 0)),
                        "backend": str(response.get("backend", "unknown")),
                        "warm_cache_hit": bool(warm_before > 0),
                    }
                except Exception as exc:
                    last_error = exc
                    self._close_worker()
            raise RuntimeError(f"Persistent SeedVR2 worker failed: {last_error}")

    @property
    def model_name(self) -> str:
        return self._model_name


def _get_persistent_runner(model_name: str) -> SeedVR2PersistentRunner:
    key = str(model_name).strip()
    runner = _SEEDVR2_PERSISTENT_RUNNERS.get(key)
    if runner is None:
        runner = SeedVR2PersistentRunner(
            Path(_default_cli_path()),
            Path(_default_python_bin()),
            key,
        )
        _SEEDVR2_PERSISTENT_RUNNERS[key] = runner
    return runner


@router.get("/v1/dev/seedvr2-3b/output/{job_id}/{filename}")
async def get_seedvr2_output(job_id: str, filename: str):
    """Serve generated SeedVR2 test output images."""
    safe_job = (job_id or "").strip()
    safe_file = Path(filename).name
    target = (_OUTPUT_ROOT / safe_job / safe_file).resolve()
    if not str(target).startswith(str(_OUTPUT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid output path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(str(target), media_type="image/png", filename=safe_file)


@router.post("/v1/dev/seedvr2-3b/run", response_model=SuccessResponse)
async def run_seedvr2_3b(
    request: Request,
    file: UploadFile = File(...),
    resolution: int = Form(1365),
    max_resolution: int = Form(2048),
    batch_size: int = Form(1),
    cache_models: bool = Form(False),
    use_persistent: bool = Form(True),
    model_variant: str = Form("7b-fp8"),
    gpu_resident: bool = Form(True),
    timeout_seconds: int = Form(900),
) -> SuccessResponse:
    """Run SeedVR2 3B on one uploaded image and return output URL/metrics."""
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded image is empty")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Invalid image file") from exc

    job_id = uuid.uuid4().hex
    job_dir = _OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.png"
    output_path = job_dir / "seedvr2_3b.png"
    log_path = job_dir / "seedvr2.log"
    image.save(input_path, format="PNG")

    cli_path = Path(_default_cli_path())
    py_bin = Path(_default_python_bin())
    if not cli_path.exists():
        raise HTTPException(status_code=500, detail=f"SEEDVR2 CLI not found: {cli_path}")
    if not py_bin.exists():
        raise HTTPException(status_code=500, detail=f"Python runtime not found: {py_bin}")

    run_meta: Dict[str, Any] = {}
    return_code = 0
    combined_log = ""
    model_key = str(model_variant or "3b").strip().lower()
    if model_key not in _MODEL_VARIANTS:
        raise HTTPException(status_code=422, detail=f"Unsupported model_variant: {model_key}")
    model_name = _MODEL_VARIANTS[model_key]
    offload_device = "none" if bool(gpu_resident) else "cpu"
    effective_resolution = max(256, int(resolution))
    effective_max_resolution = max(0, int(max_resolution))
    effective_batch_size = max(1, int(batch_size))
    if use_persistent:
        try:
            start = time.perf_counter()
            run_meta = _get_persistent_runner(model_name=model_name).run(
                input_path=input_path,
                output_path=output_path,
                resolution=effective_resolution,
                max_resolution=effective_max_resolution,
                batch_size=effective_batch_size,
                offload_device=offload_device,
                timeout_seconds=max(30, int(timeout_seconds)),
            )
            elapsed = time.perf_counter() - start
            combined_log = (
                f"mode=persistent\n"
                f"backend={run_meta.get('backend')}\n"
                f"warm_cache_hit={run_meta.get('warm_cache_hit')}\n"
                f"frames_written={run_meta.get('frames_written')}\n"
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SeedVR2 persistent runner failed: {exc}") from exc
    else:
        cmd = [
            str(py_bin),
            str(cli_path),
            str(input_path),
            "--dit_model",
            model_name,
            "--resolution",
            str(effective_resolution),
            "--max_resolution",
            str(effective_max_resolution),
            "--batch_size",
            str(effective_batch_size),
            "--output",
            str(output_path),
        ]
        if cache_models:
            cmd.extend(["--cache_dit", "--cache_vae"])
        cmd.extend(["--dit_offload_device", offload_device, "--vae_offload_device", offload_device])

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(cli_path.parent),
            capture_output=True,
            text=True,
            timeout=max(30, int(timeout_seconds)),
        )
        elapsed = time.perf_counter() - start
        return_code = int(proc.returncode)
        combined_log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            tail = combined_log[-3000:] if combined_log else "No logs."
            raise HTTPException(
                status_code=500,
                detail=f"SeedVR2 failed (code={proc.returncode}). Log tail:\n{tail}",
            )

    log_path.write_text((combined_log or "")[-250000:], encoding="utf-8")
    if not output_path.exists():
        raise HTTPException(status_code=500, detail="SeedVR2 finished without output file.")

    output_image = Image.open(output_path)
    output_route = request.app.url_path_for(
        "get_seedvr2_output",
        job_id=job_id,
        filename=output_path.name,
    )
    output_url_public = f"{_resolve_public_base_url(request)}{output_route}"

    return SuccessResponse(
        status="success",
        message="SeedVR2 run completed",
        data={
            "job_id": job_id,
            "input": {
                "filename": str(file.filename or ""),
                "width": int(image.width),
                "height": int(image.height),
            },
            "output": {
                "width": int(output_image.width),
                "height": int(output_image.height),
                "path": str(output_path),
                "url": output_url_public,
                "url_public": output_url_public,
                "url_relative": str(output_route),
            },
            "settings": {
                "mode": "persistent" if use_persistent else "subprocess",
                "model_variant": model_key,
                "model": _get_persistent_runner(model_name=model_name).model_name if use_persistent else model_name,
                "resolution": int(effective_resolution),
                "max_resolution": int(effective_max_resolution),
                "batch_size": int(effective_batch_size),
                "cache_models": bool(cache_models),
                "gpu_resident": bool(gpu_resident),
            },
            "metrics": {
                "wall_seconds": float(round(elapsed, 3)),
                "runner_seconds": float(round(float(run_meta.get("elapsed", elapsed)), 3)),
                "runner_backend": run_meta.get("backend"),
                "warm_cache_hit": bool(run_meta.get("warm_cache_hit", False)),
                "return_code": int(return_code),
                "log_path": str(log_path),
            },
        },
    )


@router.post("/v1/user-image/upscale", response_model=SuccessResponse)
async def upscale_user_image_from_url(
    request: Request,
    payload: SeedVR2UpscaleUrlRequest,
) -> SuccessResponse:
    """
    URL-based upscaling endpoint.

    Fixed model: 7B FP8 mixed variant for best quality/latency balance.
    Supports metric-based sizing via payload.metric ("2k" or "4k").
    """
    try:
        image = _download_image_url_for_upscale(payload.image_url, timeout_seconds=30)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to download image_url: {exc}") from exc

    job_id = uuid.uuid4().hex
    job_dir = _OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.png"
    output_path = job_dir / "seedvr2_7b_fp8.png"
    log_path = job_dir / "seedvr2.log"
    image.convert("RGB").save(input_path, format="PNG")

    cli_path = Path(_default_cli_path())
    py_bin = Path(_default_python_bin())
    if not cli_path.exists():
        raise HTTPException(status_code=500, detail=f"SEEDVR2 CLI not found: {cli_path}")
    if not py_bin.exists():
        raise HTTPException(status_code=500, detail=f"Python runtime not found: {py_bin}")

    model_key = "7b-fp8"
    model_name = _MODEL_VARIANTS[model_key]
    offload_device = "none" if bool(payload.gpu_resident) else "cpu"
    requested_metric = str(payload.metric or "").strip().lower()
    if requested_metric == "4k":
        target_long_edge = 4096
    elif requested_metric == "2k":
        target_long_edge = 2048
    else:
        target_long_edge = int(payload.target_long_edge)
    if target_long_edge == 4096:
        applied_metric = "4k"
    elif target_long_edge == 2048:
        applied_metric = "2k"
    else:
        applied_metric = "custom"
    input_long_edge = max(int(image.width), int(image.height))
    input_short_edge = min(int(image.width), int(image.height))
    if input_long_edge <= 0 or input_short_edge <= 0:
        raise HTTPException(status_code=422, detail="Invalid input image dimensions")
    scale = float(target_long_edge) / float(input_long_edge)
    derived_short_edge = max(1, int(round(float(input_short_edge) * scale)))
    effective_resolution = max(256, derived_short_edge)
    effective_max_resolution = max(0, target_long_edge)
    effective_batch_size = max(1, int(payload.batch_size))

    run_meta: Dict[str, Any] = {}
    return_code = 0
    combined_log = ""

    if bool(payload.use_persistent):
        try:
            start = time.perf_counter()
            run_meta = _get_persistent_runner(model_name=model_name).run(
                input_path=input_path,
                output_path=output_path,
                resolution=effective_resolution,
                max_resolution=effective_max_resolution,
                batch_size=effective_batch_size,
                offload_device=offload_device,
                timeout_seconds=max(30, int(payload.timeout_seconds)),
            )
            elapsed = time.perf_counter() - start
            combined_log = (
                f"mode=persistent\n"
                f"backend={run_meta.get('backend')}\n"
                f"warm_cache_hit={run_meta.get('warm_cache_hit')}\n"
                f"frames_written={run_meta.get('frames_written')}\n"
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SeedVR2 persistent runner failed: {exc}") from exc
    else:
        cmd = [
            str(py_bin),
            str(cli_path),
            str(input_path),
            "--dit_model",
            model_name,
            "--resolution",
            str(effective_resolution),
            "--max_resolution",
            str(effective_max_resolution),
            "--batch_size",
            str(effective_batch_size),
            "--output",
            str(output_path),
        ]
        if bool(payload.cache_models):
            cmd.extend(["--cache_dit", "--cache_vae"])
        cmd.extend(["--dit_offload_device", offload_device, "--vae_offload_device", offload_device])

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(cli_path.parent),
            capture_output=True,
            text=True,
            timeout=max(30, int(payload.timeout_seconds)),
        )
        elapsed = time.perf_counter() - start
        return_code = int(proc.returncode)
        combined_log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            tail = combined_log[-3000:] if combined_log else "No logs."
            raise HTTPException(
                status_code=500,
                detail=f"SeedVR2 failed (code={proc.returncode}). Log tail:\n{tail}",
            )

    log_path.write_text((combined_log or "")[-250000:], encoding="utf-8")
    if not output_path.exists():
        raise HTTPException(status_code=500, detail="SeedVR2 finished without output file.")

    output_image = Image.open(output_path)
    output_route = request.app.url_path_for(
        "get_seedvr2_output",
        job_id=job_id,
        filename=output_path.name,
    )
    output_url_public = f"{_resolve_public_base_url(request)}{output_route}"

    storage_url: Optional[str] = None
    storage_error: Optional[str] = None
    if bool(payload.upload_to_storage):
        try:
            requested_name = str(payload.output_filename or "").strip()
            safe_name = Path(requested_name).name if requested_name else f"seedvr2/{job_id}.png"
            if "." not in Path(safe_name).name:
                safe_name = f"{safe_name}.png"
            storage_url = storage.upload_image(
                output_path.read_bytes(),
                filename=safe_name,
                content_type="image/png",
            )
        except Exception as exc:
            storage_error = str(exc)

    return SuccessResponse(
        status="success",
        message="SeedVR2 URL upscale completed",
        data={
            "job_id": job_id,
            "input": {
                "image_url": payload.image_url,
                "width": int(image.width),
                "height": int(image.height),
            },
            "output": {
                "width": int(output_image.width),
                "height": int(output_image.height),
                "path": str(output_path),
                "url": storage_url or output_url_public,
                "url_public": output_url_public,
                "url_relative": str(output_route),
                "storage_url": storage_url,
            },
            "settings": {
                "mode": "persistent" if bool(payload.use_persistent) else "subprocess",
                "model_variant": model_key,
                "model": model_name,
                "metric_requested": requested_metric or None,
                "metric_applied": applied_metric,
                "target_long_edge": int(target_long_edge),
                "resolution": int(effective_resolution),
                "max_resolution": int(effective_max_resolution),
                "batch_size": int(effective_batch_size),
                "cache_models": bool(payload.cache_models),
                "gpu_resident": bool(payload.gpu_resident),
                "upload_to_storage": bool(payload.upload_to_storage),
            },
            "metrics": {
                "wall_seconds": float(round(elapsed, 3)),
                "runner_seconds": float(round(float(run_meta.get("elapsed", elapsed)), 3)),
                "runner_backend": run_meta.get("backend"),
                "warm_cache_hit": bool(run_meta.get("warm_cache_hit", False)),
                "return_code": int(return_code),
                "input_long_edge": int(input_long_edge),
                "input_short_edge": int(input_short_edge),
                "derived_short_edge": int(derived_short_edge),
                "log_path": str(log_path),
            },
            "warnings": {
                "storage_upload_error": storage_error,
            },
        },
    )


@router.get("/dev/seedvr2-3b-lab", response_class=HTMLResponse)
async def seedvr2_3b_lab_page() -> HTMLResponse:
    """Simple visual lab page to test SeedVR2 3B quality."""
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SeedVR2 3B Lab</title>
  <style>
    :root {
      --bg: #edf5f9; --ink: #123; --muted: #526272; --line: #cddbe6;
      --card: #fff; --ok: #1f7a4f; --err: #a11a3a;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Space Grotesk","Manrope",sans-serif; background: var(--bg); color: var(--ink); }
    .wrap { max-width: 1200px; margin: 14px auto; padding: 0 12px 20px; }
    .layout { display: grid; grid-template-columns: 360px 1fr; gap: 12px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
    label { display:block; font-size:12px; color:var(--muted); margin-bottom:5px; text-transform:uppercase; font-weight:700; }
    input, select { width:100%; border:1px solid var(--line); border-radius:10px; padding:9px; margin-bottom:10px; }
    .row2 { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .btn { border:0; border-radius:10px; padding:10px 14px; font-weight:700; color:#fff; background:#0f766e; cursor:pointer; }
    .status { margin-top:10px; min-height:20px; color:var(--muted); }
    .status.ok { color:var(--ok); } .status.err { color:var(--err); }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .tile img { width:100%; height:420px; object-fit:contain; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .metrics { display:grid; grid-template-columns:repeat(3,minmax(120px,1fr)); gap:8px; margin-bottom:10px; }
    .metric { border:1px solid var(--line); border-radius:10px; padding:8px; background:#f8fbfd; }
    .k { font-size:11px; color:var(--muted); text-transform:uppercase; } .v { font-weight:700; font-size:14px; }
    pre { margin:10px 0 0; border:1px solid var(--line); border-radius:10px; background:#0f172a; color:#e2e8f0; padding:10px; min-height:120px; overflow:auto; }
    @media (max-width: 980px) { .layout { grid-template-columns:1fr; } .grid2 { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>SeedVR2 3B Quality Lab</h2>
    <div class="layout">
      <section class="card">
        <label for="apiBase">API Base</label>
        <input id="apiBase" type="text" placeholder="Leave blank for same origin" />
        <label for="fileInput">Input Image</label>
        <input id="fileInput" type="file" accept="image/*" />
        <div class="row2">
          <div><label for="resolution">Resolution</label><input id="resolution" type="number" min="256" max="3072" value="1365" /></div>
          <div><label for="maxRes">Max Resolution</label><input id="maxRes" type="number" min="0" max="4096" value="2048" /></div>
        </div>
        <div class="row2">
          <div><label for="batchSize">Batch Size</label><input id="batchSize" type="number" min="1" max="33" value="1" /></div>
          <div><label for="cacheModels">Cache Models</label><select id="cacheModels"><option value="false" selected>false</option><option value="true">true</option></select></div>
        </div>
        <div class="row2">
          <div><label for="usePersistent">Use Persistent Runner</label><select id="usePersistent"><option value="true" selected>true</option><option value="false">false</option></select></div>
          <div><label for="modelVariant">Model</label><select id="modelVariant"><option value="3b">3B FP8 (Fast)</option><option value="7b">7B FP16</option><option value="7b-fp8" selected>7B FP8 Mixed (Recommended)</option><option value="7b-fp8-sharp">7B FP8 Sharp</option></select></div>
        </div>
        <div class="row2">
          <div><label for="gpuResident">Keep Model On GPU</label><select id="gpuResident"><option value="true" selected>true</option><option value="false">false</option></select></div>
          <div></div>
        </div>
        <button id="runBtn" class="btn">Run SeedVR2 3B</button>
        <div id="status" class="status">Ready.</div>
      </section>
      <section class="card">
        <div class="metrics">
          <div class="metric"><div class="k">Wall Seconds</div><div id="mWall" class="v">-</div></div>
          <div class="metric"><div class="k">Output Size</div><div id="mSize" class="v">-</div></div>
          <div class="metric"><div class="k">Job ID</div><div id="mJob" class="v">-</div></div>
        </div>
        <div class="grid2">
          <div class="tile"><h4>Input</h4><img id="imgInput" alt="input" /></div>
          <div class="tile"><h4>SeedVR2 3B</h4><img id="imgOut" alt="output" /></div>
        </div>
        <pre id="jsonOut">No response yet.</pre>
      </section>
    </div>
  </div>
  <script>
    const fileInput = document.getElementById("fileInput");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const jsonOut = document.getElementById("jsonOut");
    const imgInput = document.getElementById("imgInput");
    const imgOut = document.getElementById("imgOut");
    const apiBaseEl = document.getElementById("apiBase");
    const resolutionEl = document.getElementById("resolution");
    const maxResEl = document.getElementById("maxRes");
    const batchSizeEl = document.getElementById("batchSize");
    const cacheModelsEl = document.getElementById("cacheModels");
    const usePersistentEl = document.getElementById("usePersistent");
    const modelVariantEl = document.getElementById("modelVariant");
    const gpuResidentEl = document.getElementById("gpuResident");
    function setStatus(txt, kind) { statusEl.textContent = txt; statusEl.className = `status ${kind||""}`.trim(); }
    function safe(v, fb="-") { return (v===undefined||v===null||v==="")?fb:String(v); }
    function parseBool(v) { return String(v).toLowerCase()==="true"; }
    function joinUrl(base, path) {
      const b = (base || "").replace(/\\/+$/, "");
      const p = String(path || "");
      if (!p) return "";
      if (p.startsWith("http://") || p.startsWith("https://")) return p;
      return p.startsWith("/") ? `${b}${p}` : `${b}/${p}`;
    }
    fileInput.addEventListener("change", () => {
      const f = fileInput.files && fileInput.files[0]; if (!f) return; imgInput.src = URL.createObjectURL(f);
    });
    runBtn.addEventListener("click", async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) { setStatus("Choose an image first.", "err"); return; }
      runBtn.disabled = true; setStatus("Running...", "");
      const fd = new FormData();
      fd.append("file", file);
      fd.append("resolution", String(parseInt(resolutionEl.value || "768", 10)));
      fd.append("max_resolution", String(parseInt(maxResEl.value || "1024", 10)));
      fd.append("batch_size", String(parseInt(batchSizeEl.value || "1", 10)));
      fd.append("cache_models", String(parseBool(cacheModelsEl.value || "false")));
      fd.append("use_persistent", String(parseBool(usePersistentEl.value || "true")));
      fd.append("model_variant", String((modelVariantEl.value || "3b")).toLowerCase());
      fd.append("gpu_resident", String(parseBool(gpuResidentEl.value || "true")));
      const base = (apiBaseEl.value || "").trim();
      try {
        const res = await fetch(`${base}/v1/dev/seedvr2-3b/run`, { method: "POST", body: fd });
        const data = await res.json().catch(() => ({}));
        jsonOut.textContent = JSON.stringify(data, null, 2);
        if (!res.ok) { setStatus(`Failed (${res.status})`, "err"); return; }
        const d = data.data || {};
        setStatus("Success", "ok");
        document.getElementById("mWall").textContent = safe(d.metrics && d.metrics.wall_seconds);
        document.getElementById("mSize").textContent = `${safe(d.output && d.output.width, "?")}x${safe(d.output && d.output.height, "?")}`;
        document.getElementById("mJob").textContent = safe(d.job_id);
        const output = d.output || {};
        const resolvedSrc =
          output.url_public ||
          output.url ||
          joinUrl(base || window.location.origin, output.url_relative);
        imgOut.src = resolvedSrc || "";
      } catch (e) {
        setStatus(`Request error: ${e}`, "err");
        jsonOut.textContent = String(e);
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
