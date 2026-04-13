"""
Reusable request/build helpers for Qwen Extract-Outfit inference.
"""

from __future__ import annotations

import base64
import io
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from PIL import Image

from core.qwen_image_edit_runner import QwenImageEditRunner

DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT = (
    "Extract the clothing from the image and convert it into a clean, standalone mockup. "
    "Preserve the original fabric texture, stitching, folds, patterns, and color accuracy. "
    "Remove the model and background completely, keeping the garment's natural shape and proportions intact. "
    "Present the clothing as a flat-lay or neutral mockup on a plain background with even lighting, "
    "maintaining photorealistic detail and sharp edges."
)
DEFAULT_QWEN_EXTRACT_OUTFIT_GUIDANCE = None


def _is_truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class QwenExtractOutfitRequest:
    prompt: str
    prompt_default_applied: bool
    steps: int
    seed: int
    guidance_scale: Optional[float]
    negative_prompt: str
    max_input_edge: int
    output_max_edge: int
    output_aspect_ratio: Optional[str]
    upload_output: bool
    include_base64: bool


def build_qwen_extract_outfit_request(
    *,
    prompt: Optional[str],
    steps: int,
    seed: int,
    guidance_scale: Optional[float],
    guidance_scale_alias: Optional[float],
    negative_prompt: Optional[str],
    negative_prompt_alias: Optional[str],
    max_input_edge: int,
    max_input_edge_alias: Optional[int],
    output_max_edge: Optional[int],
    output_max_edge_alias: Optional[int],
    output_aspect_ratio: Optional[str],
    output_aspect_ratio_alias: Optional[str],
    upload_output: bool,
    include_base64: bool,
) -> QwenExtractOutfitRequest:
    """
    Independent function #1:
    Normalize all route params into one reusable request contract.
    """
    normalized_prompt = " ".join(str(prompt or "").split()).strip()
    effective_prompt = normalized_prompt or DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT
    effective_guidance = (
        float(guidance_scale_alias)
        if guidance_scale_alias is not None
        else (float(guidance_scale) if guidance_scale is not None else None)
    )
    effective_negative = (
        " ".join(str(negative_prompt_alias if negative_prompt_alias is not None else negative_prompt or "").split()).strip()
    )
    effective_max_edge = int(max_input_edge_alias) if max_input_edge_alias is not None else int(max_input_edge)
    # By default, output edge follows input edge to keep latency predictable.
    if output_max_edge_alias is not None:
        effective_output_max_edge = int(output_max_edge_alias)
    elif output_max_edge is not None:
        effective_output_max_edge = int(output_max_edge)
    else:
        effective_output_max_edge = int(effective_max_edge)
    effective_aspect_ratio = " ".join(
        str(output_aspect_ratio_alias if output_aspect_ratio_alias is not None else output_aspect_ratio or "").split()
    ).strip()
    return QwenExtractOutfitRequest(
        prompt=effective_prompt,
        prompt_default_applied=not bool(normalized_prompt),
        steps=int(steps),
        seed=int(seed),
        guidance_scale=effective_guidance,
        negative_prompt=effective_negative,
        max_input_edge=effective_max_edge,
        output_max_edge=effective_output_max_edge,
        output_aspect_ratio=effective_aspect_ratio or None,
        upload_output=bool(upload_output),
        include_base64=bool(include_base64),
    )


def execute_qwen_extract_outfit_request(
    *,
    request: QwenExtractOutfitRequest,
    source_image: Image.Image,
    runner: QwenImageEditRunner,
    upload_image_fn: Optional[Callable[..., str]] = None,
    output_dir: str = "/tmp/qwen_extract_outfit_outputs",
) -> Dict[str, object]:
    """
    Independent function #2:
    Execute one normalized request and return a reusable API payload block.
    """
    source = _resize_to_max_edge(source_image.convert("RGB"), request.max_input_edge)
    output_width, output_height = _resolve_output_size(
        source_width=source.width,
        source_height=source.height,
        max_edge=request.output_max_edge,
        output_aspect_ratio=request.output_aspect_ratio,
    )
    started_at = time.time()
    output_image, meta = runner.run_edit(
        source,
        prompt=request.prompt,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        output_width=output_width,
        output_height=output_height,
    )
    elapsed = time.time() - started_at

    save_local_output = _is_truthy(os.getenv("QWEN_EXTRACT_OUTFIT_SAVE_LOCAL", "0"))
    need_output_bytes = bool(request.upload_output or request.include_base64 or save_local_output)

    output_bytes: bytes = b""
    output_name = ""
    if need_output_bytes:
        out_buf = io.BytesIO()
        output_image.save(out_buf, format="PNG")
        output_bytes = out_buf.getvalue()
        output_name = f"qwen-extract-outfit-{uuid.uuid4().hex}.png"

    local_path = ""
    if save_local_output and output_bytes:
        local_dir = Path(output_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        path = local_dir / output_name
        path.write_bytes(output_bytes)
        local_path = str(path)

    output_url = ""
    if request.upload_output and callable(upload_image_fn) and output_bytes:
        output_url = str(upload_image_fn(output_bytes, filename=output_name, content_type="image/png") or "")

    image_base64 = ""
    if request.include_base64 and output_bytes:
        image_base64 = base64.b64encode(output_bytes).decode("ascii")

    response_meta = dict(meta or {})
    response_meta["prompt"] = request.prompt
    response_meta["prompt_default_applied"] = bool(request.prompt_default_applied)
    response_meta["input_size"] = {"width": int(source.width), "height": int(source.height)}
    response_meta["requested_output_size"] = {"width": int(output_width), "height": int(output_height)}
    response_meta["output_size"] = {"width": int(output_image.width), "height": int(output_image.height)}
    response_meta["elapsed_seconds"] = round(float(elapsed), 3)
    response_meta["max_input_edge"] = int(request.max_input_edge)
    response_meta["max_output_edge"] = int(request.output_max_edge)
    response_meta["output_aspect_ratio"] = str(request.output_aspect_ratio or "")

    return {
        "output_url": output_url,
        "local_path": local_path,
        "image_base64": image_base64,
        "metadata": response_meta,
    }


def _resize_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    ratio = float(max_edge) / float(longest)
    target = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    return image.resize(target, Image.Resampling.LANCZOS)


def _fit_to_max_edge_without_upscale(width: int, height: int, max_edge: int) -> tuple[int, int]:
    w = int(width)
    h = int(height)
    edge = int(max_edge)
    longest = max(w, h)
    if longest > edge:
        ratio = float(edge) / float(longest)
        w = max(1, int(round(w * ratio)))
        h = max(1, int(round(h * ratio)))
    return max(1, w), max(1, h)


def _resolve_output_size(
    *,
    source_width: int,
    source_height: int,
    max_edge: int,
    output_aspect_ratio: Optional[str],
) -> Tuple[int, int]:
    parsed_ratio = _parse_aspect_ratio(output_aspect_ratio)
    if not parsed_ratio:
        return _fit_to_max_edge_without_upscale(source_width, source_height, max_edge)

    ratio_w, ratio_h = parsed_ratio
    edge = int(max_edge)
    if ratio_w >= ratio_h:
        out_w = edge
        out_h = max(1, int(round(edge * (float(ratio_h) / float(ratio_w)))))
    else:
        out_h = edge
        out_w = max(1, int(round(edge * (float(ratio_w) / float(ratio_h)))))
    return max(1, int(out_w)), max(1, int(out_h))


def _parse_aspect_ratio(raw_ratio: Optional[str]) -> Optional[Tuple[int, int]]:
    text = str(raw_ratio or "").strip().lower()
    if not text:
        return None
    text = text.replace("x", ":").replace("/", ":")
    parts = [part.strip() for part in text.split(":", maxsplit=1)]
    if len(parts) != 2:
        return None
    try:
        w = int(parts[0])
        h = int(parts[1])
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)
