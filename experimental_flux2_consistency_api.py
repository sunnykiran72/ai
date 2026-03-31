"""
Standalone experimental API for FLUX.2-klein-9B + Consistency LoRA.

This file is intentionally independent from the main app/config/services.
It does not import project runtime config and uses explicit constants only.

Run:
    uvicorn experimental_flux2_consistency_api:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Any

import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from PIL import Image


logger = logging.getLogger("flux2-consistency-test")
logging.basicConfig(level=logging.INFO)


# Explicit test constants (no project config/env coupling).
MODEL_SOURCE_ID = "black-forest-labs/FLUX.2-klein-9B"
MODEL_LOCAL_PATH = "/workspace/models/flux2-klein"
LORA_SOURCE_REPO_ID = "dx8152/Flux2-Klein-9B-Consistency"
LORA_LOCAL_DIR = "/workspace/models/flux2-lora/dx8152-consistency"
LORA_WEIGHT_NAME = "Klein-consistency.safetensors"
LORA_ADAPTER_NAME = "klein_consistency"
DEFAULT_LORA_SCALE = 0.8
DEFAULT_STEPS = 8
DEFAULT_SEED = 42
DEFAULT_GUIDANCE_SCALE = 3.5
HTTP_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class PipelineConfig:
    model_source_id: str = MODEL_SOURCE_ID
    model_local_path: str = MODEL_LOCAL_PATH
    lora_source_repo_id: str = LORA_SOURCE_REPO_ID
    lora_local_dir: str = LORA_LOCAL_DIR
    lora_weight_name: str = LORA_WEIGHT_NAME
    adapter_name: str = LORA_ADAPTER_NAME
    lora_scale: float = DEFAULT_LORA_SCALE


class ConsistencyTryonRequest(BaseModel):
    user_image_url: str = Field(..., description="Person image URL (Image 1)")
    garment_image_url: str = Field(..., description="Garment image URL (Image 2)")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional edit prompt. If omitted, a strict garment-swap prompt is used.",
    )
    steps: int = Field(default=DEFAULT_STEPS, ge=4, le=40)
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2147483647)
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, ge=1.0, le=12.0)
    lora_scale: float = Field(default=DEFAULT_LORA_SCALE, ge=0.0, le=2.0)
    output_width: Optional[int] = Field(default=None, ge=256, le=2048)
    output_height: Optional[int] = Field(default=None, ge=256, le=2048)

    @field_validator("user_image_url", "garment_image_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("image URL must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("image URL cannot be empty")
        return cleaned


class ConsistencyTryonResponse(BaseModel):
    status: str
    message: str
    image_base64_jpeg: str
    metadata: dict


_APP = FastAPI(title="Experimental FLUX2 Consistency API", version="0.1.0")
_PIPELINE_LOCK = threading.Lock()
_PIPELINE = None
_PIPELINE_CONFIG = PipelineConfig()
app = _APP


def _flatten_rgba_to_white_rgb(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    base = Image.new("RGB", rgba.size, (255, 255, 255))
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def _download_image(url: str) -> Image.Image:
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {url} ({exc})")
    try:
        return _flatten_rgba_to_white_rgb(Image.open(io.BytesIO(resp.content)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image content from URL: {url} ({exc})")


def _fit_garment_to_canvas(garment: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    tw, th = target_size
    canvas = Image.new("RGB", (tw, th), (245, 245, 245))
    gw, gh = garment.size
    scale = min(float(tw) / float(gw), float(th) / float(gh))
    nw, nh = max(1, int(gw * scale)), max(1, int(gh * scale))
    resized = garment.resize((nw, nh), Image.Resampling.LANCZOS)
    ox = (tw - nw) // 2
    oy = (th - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def _resolve_pipeline_class() -> Any:
    # Keep this local and explicit for standalone behavior.
    try:
        from diffusers import Flux2KleinPipeline  # type: ignore

        return Flux2KleinPipeline
    except Exception:
        pass
    try:
        from diffusers import Flux2Pipeline  # type: ignore

        return Flux2Pipeline
    except Exception as exc:
        raise RuntimeError(f"Could not import a FLUX2 pipeline class from diffusers: {exc}")


def _ensure_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipeline_cls = _resolve_pipeline_class()
        logger.info(
            "Loading pipeline model_source=%s model_local=%s device=%s",
            _PIPELINE_CONFIG.model_source_id,
            _PIPELINE_CONFIG.model_local_path,
            device,
        )
        pipe = pipeline_cls.from_pretrained(
            _PIPELINE_CONFIG.model_local_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        # Reduce VRAM pressure on shared GPUs.
        if device == "cuda" and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        pipe.load_lora_weights(
            _PIPELINE_CONFIG.lora_local_dir,
            weight_name=_PIPELINE_CONFIG.lora_weight_name,
            adapter_name=_PIPELINE_CONFIG.adapter_name,
        )
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([_PIPELINE_CONFIG.adapter_name], adapter_weights=[_PIPELINE_CONFIG.lora_scale])
        elif hasattr(pipe, "set_adapter"):
            pipe.set_adapter(_PIPELINE_CONFIG.adapter_name)

        _PIPELINE = pipe
        return _PIPELINE


def _default_tryon_prompt() -> str:
    return (
        "Use Image 1 as the identity and body anchor. "
        "Apply only the garment from Image 2 onto the person in Image 1. "
        "Preserve exact face identity, skin tone, hair, body proportions, pose, hands, and background. "
        "Do not change camera framing. Do not alter non-garment regions."
    )


def _encode_jpeg_base64(image: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@_APP.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_source_id": _PIPELINE_CONFIG.model_source_id,
        "model_local_path": _PIPELINE_CONFIG.model_local_path,
        "lora_source_repo": _PIPELINE_CONFIG.lora_source_repo_id,
        "lora_local_dir": _PIPELINE_CONFIG.lora_local_dir,
        "lora_weight": _PIPELINE_CONFIG.lora_weight_name,
    }


@_APP.post("/v1/flux2/consistency-tryon-test", response_model=ConsistencyTryonResponse)
def consistency_tryon_test(request: ConsistencyTryonRequest) -> ConsistencyTryonResponse:
    person = _download_image(request.user_image_url)
    garment = _download_image(request.garment_image_url)

    try:
        pipe = _ensure_pipeline()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}")

    out_w = int(request.output_width or person.width)
    out_h = int(request.output_height or person.height)
    garment_board = _fit_garment_to_canvas(garment, (out_w, out_h))
    person_resized = person.resize((out_w, out_h), Image.Resampling.LANCZOS)

    prompt = str(request.prompt or _default_tryon_prompt()).strip()
    seed = int(request.seed)
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        generator = torch.Generator(device=gen_device).manual_seed(seed)
    except Exception:
        generator = torch.Generator().manual_seed(seed)

    # Per-request scale override for this single LoRA.
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters([_PIPELINE_CONFIG.adapter_name], adapter_weights=[float(request.lora_scale)])

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        output = pipe(
            image=[person_resized, garment_board],
            prompt=prompt,
            num_inference_steps=int(request.steps),
            guidance_scale=float(request.guidance_scale),
            width=out_w,
            height=out_h,
            generator=generator,
        ).images[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return ConsistencyTryonResponse(
        status="success",
        message="Experimental consistency try-on completed",
        image_base64_jpeg=_encode_jpeg_base64(output),
        metadata={
            "model_source_id": _PIPELINE_CONFIG.model_source_id,
            "model_local_path": _PIPELINE_CONFIG.model_local_path,
            "lora_source_repo": _PIPELINE_CONFIG.lora_source_repo_id,
            "lora_local_dir": _PIPELINE_CONFIG.lora_local_dir,
            "lora_weight": _PIPELINE_CONFIG.lora_weight_name,
            "lora_scale": float(request.lora_scale),
            "steps": int(request.steps),
            "seed": seed,
            "guidance_scale": float(request.guidance_scale),
            "resolution": [out_w, out_h],
            "input_resolution": [int(person.width), int(person.height)],
        },
    )


app = _APP
