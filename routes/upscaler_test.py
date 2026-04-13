"""
Dev upscaler benchmark routes.

Provides a small API for side-by-side upscaler timing/output checks.
"""

import io
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.responses import FileResponse

from core.artisan_upscaler_runner import ArtisanUpscalerRunner
from routes.models import SuccessResponse
from services import AIEngine

logger = logging.getLogger("glamify-ai")
router = APIRouter()

_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "upscaler_api_tests"
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
_ARTISAN_RUNNER: Optional[ArtisanUpscalerRunner] = None


def get_ai_engine() -> AIEngine:
    """Dependency to get AIEngine instance."""
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_ai_engine()


def get_artisan_runner() -> ArtisanUpscalerRunner:
    global _ARTISAN_RUNNER
    if _ARTISAN_RUNNER is None:
        _ARTISAN_RUNNER = ArtisanUpscalerRunner()
    return _ARTISAN_RUNNER


def _json_float_list(values: List[float]) -> List[float]:
    return [float(round(v, 6)) for v in values]


@router.get("/v1/dev/upscaler-test/output/{job_id}/{filename}")
async def get_upscaler_test_output(job_id: str, filename: str):
    """Serve generated benchmark output images."""
    safe_job = (job_id or "").strip()
    safe_file = Path(filename).name
    target = (_OUTPUT_ROOT / safe_job / safe_file).resolve()
    if not str(target).startswith(str(_OUTPUT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid output path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(str(target), media_type="image/png", filename=safe_file)


@router.post("/v1/dev/upscaler-test/compare", response_model=SuccessResponse)
async def run_upscaler_compare(
    request: Request,
    file: UploadFile = File(...),
    run_artisan: bool = Form(True),
    run_realesrgan: bool = Form(True),
    repeats: int = Form(1),
    max_output_edge: int = Form(1024),
    artisan_precision: str = Form("bf16"),
    artisan_compile: bool = Form(False),
    artisan_tile: int = Form(128),
    artisan_overlap: int = Form(32),
    realesrgan_target_max_edge: int = Form(0),
    ai_engine: AIEngine = Depends(get_ai_engine),
) -> SuccessResponse:
    """
    Compare Artisan and Real-ESRGAN on the same uploaded image.

    Returns per-model timings and output image URLs for visual checks.
    """
    if not run_artisan and not run_realesrgan:
        raise HTTPException(status_code=422, detail="Enable at least one model: run_artisan or run_realesrgan")

    repeat_count = max(1, min(int(repeats), 20))
    output_edge = max(0, int(max_output_edge))
    artisan_precision = str(artisan_precision or "bf16").strip().lower()
    if artisan_precision not in {"bf16", "fp32"}:
        raise HTTPException(status_code=422, detail="artisan_precision must be 'bf16' or 'fp32'")

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
    image.save(input_path, format="PNG")

    results: Dict[str, Any] = {}

    if run_realesrgan:
        realesrgan = getattr(ai_engine, "realesrgan", None)
        if realesrgan is None or not realesrgan.is_available:
            results["realesrgan"] = {
                "enabled": True,
                "available": False,
                "error": "RealESRGAN dependencies are unavailable.",
            }
        else:
            load_seconds = 0.0
            if not realesrgan.is_loaded:
                t0 = time.perf_counter()
                realesrgan.ensure_ready()
                load_seconds = time.perf_counter() - t0

            infer_seconds: List[float] = []
            output_image: Optional[Image.Image] = None
            target_edge = int(realesrgan_target_max_edge) if int(realesrgan_target_max_edge) > 0 else None

            for _ in range(repeat_count):
                t0 = time.perf_counter()
                candidate = realesrgan.upscale(image, target_max_edge=target_edge)
                elapsed = time.perf_counter() - t0
                infer_seconds.append(float(elapsed))
                output_image = candidate.convert("RGB")

            assert output_image is not None
            if output_edge > 0 and max(output_image.size) > output_edge:
                scale = float(output_edge) / float(max(output_image.size))
                output_image = output_image.resize(
                    (
                        max(1, int(round(output_image.width * scale))),
                        max(1, int(round(output_image.height * scale))),
                    ),
                    getattr(getattr(Image, "Resampling", Image), "LANCZOS"),
                )

            out_path = job_dir / "realesrgan.png"
            output_image.save(out_path, format="PNG")
            out_url = str(request.url_for("get_upscaler_test_output", job_id=job_id, filename=out_path.name))
            results["realesrgan"] = {
                "enabled": True,
                "available": True,
                "model_name": str(getattr(realesrgan, "model_name", "RealESRGAN_x4plus")),
                "load_seconds": float(round(load_seconds, 6)),
                "repeat_count": repeat_count,
                "inference_seconds": _json_float_list(infer_seconds),
                "avg_inference_seconds": float(round(sum(infer_seconds) / len(infer_seconds), 6)),
                "output_width": int(output_image.width),
                "output_height": int(output_image.height),
                "output_path": str(out_path),
                "output_url": out_url,
            }

    if run_artisan:
        artisan = get_artisan_runner()
        prepare_info = artisan.ensure_ready(
            precision=artisan_precision,
            compile_model=bool(artisan_compile),
        )

        infer_seconds = []
        output_image = None
        for _ in range(repeat_count):
            candidate, elapsed = artisan.upscale(
                image,
                tile_size=max(16, int(artisan_tile)),
                overlap=max(0, int(artisan_overlap)),
            )
            infer_seconds.append(float(elapsed))
            output_image = candidate

        assert output_image is not None
        output_image = artisan.resize_to_max_edge(output_image, output_edge if output_edge > 0 else None)

        out_path = job_dir / "artisan.png"
        output_image.save(out_path, format="PNG")
        out_url = str(request.url_for("get_upscaler_test_output", job_id=job_id, filename=out_path.name))

        results["artisan"] = {
            "enabled": True,
            "available": True,
            "model_name": "ArtisanLabs/artisan-upscaler",
            "precision": artisan_precision,
            "compiled": bool(float(prepare_info.get("compiled", 0.0)) > 0.0),
            "tile": int(max(16, int(artisan_tile))),
            "overlap": int(max(0, int(artisan_overlap))),
            "load_seconds": float(round(float(prepare_info.get("load_seconds", 0.0)), 6)),
            "compile_seconds": float(round(float(prepare_info.get("compile_seconds", 0.0)), 6)),
            "repeat_count": repeat_count,
            "inference_seconds": _json_float_list(infer_seconds),
            "avg_inference_seconds": float(round(sum(infer_seconds) / len(infer_seconds), 6)),
            "output_width": int(output_image.width),
            "output_height": int(output_image.height),
            "output_path": str(out_path),
            "output_url": out_url,
        }

    return SuccessResponse(
        status="success",
        message="Upscaler comparison completed",
        data={
            "job_id": job_id,
            "input": {
                "filename": str(file.filename or ""),
                "width": int(image.width),
                "height": int(image.height),
                "input_path": str(input_path),
            },
            "settings": {
                "run_artisan": bool(run_artisan),
                "run_realesrgan": bool(run_realesrgan),
                "repeats": repeat_count,
                "max_output_edge": output_edge,
            },
            "results": results,
        },
    )
