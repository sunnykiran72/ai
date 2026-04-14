"""
Reusable request/build helpers for Qwen Extract-Outfit inference.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image

from core.qwen_image_edit_runner import QwenImageEditRunner
from shared.category_mapping import infer_style_from_text, wardrobe_category_from_garment_type
from utils.prompt_generation import parse_structured_descriptor
from utils.runtime_compat import _build_garment_metadata
from utils.validation import normalize_garment_type

DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT = (
    "Extract the clothing from the image and convert it into a clean, standalone mockup. "
    "Preserve the original fabric texture, stitching, folds, patterns, and color accuracy. "
    "Remove the model and background completely, keeping the garment's natural shape and proportions intact. "
    "Present the clothing as a flat-lay or neutral mockup on a plain background with even lighting, "
    "maintaining photorealistic detail and sharp edges."
)
DEFAULT_QWEN_EXTRACT_OUTFIT_GUIDANCE = None
PROMPT_GENERATION_FAILED_CODE = "PROMPT_GENERATION_FAILED"
PROMPT_SOURCE_INPUT_PARALLEL = "minicpm_input_parallel"
PROMPT_SOURCE_EXTRACTED_FALLBACK = "minicpm_extracted_fallback"
_INVALID_PROMPT_TOKENS = {"none", "n/a", "unknown", "no garment"}


class PromptGenerationFailedError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.reason_code = PROMPT_GENERATION_FAILED_CODE


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


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _is_prompt_usable(prompt_text: str) -> bool:
    normalized = _normalize_text(prompt_text)
    if not normalized:
        return False
    if len(normalized) < 24:
        return False
    collapsed = normalized.lower().strip(" .,:;!?")
    if collapsed in _INVALID_PROMPT_TOKENS:
        return False
    alphabetic_words = re.findall(r"[A-Za-z]+", normalized)
    if len(alphabetic_words) < 5:
        return False
    return True


def _infer_garment_type(prompt_text: str) -> str:
    normalized_prompt = _normalize_text(prompt_text)
    parsed = parse_structured_descriptor(normalized_prompt)
    candidates = [
        parsed.get("type"),
        parsed.get("category"),
        normalized_prompt,
    ]
    for candidate in candidates:
        normalized = normalize_garment_type(str(candidate or ""))
        if normalized in {"top", "bottom", "dress", "outer"}:
            return normalized
    lowered = normalized_prompt.lower()
    if any(token in lowered for token in {"dress", "gown", "jumpsuit", "romper"}):
        return "dress"
    if any(token in lowered for token in {"jacket", "coat", "hoodie", "blazer", "outerwear"}):
        return "outer"
    if any(token in lowered for token in {"jeans", "pants", "trousers", "shorts", "skirt", "bottom"}):
        return "bottom"
    return "top"


def _build_analyze_style_garment_metadata(*, prompt_description: str, prompt_source: str) -> Dict[str, object]:
    target_type = _infer_garment_type(prompt_description)
    style = infer_style_from_text(prompt_description, garment_type=target_type)
    category = wardrobe_category_from_garment_type(target_type, style=style)
    try:
        return _build_garment_metadata(
            base_garment_prompt=prompt_description,
            extraction_avoid_clause="",
            prompt_sections_raw="",
            descriptor_raw_text=prompt_description,
            prompt_description=prompt_description,
            prompt_source=prompt_source,
            target_type=target_type,
            backend_target_type=target_type,
            style=category.get("style", ""),
            primary_category_key=category.get("primary_category_key", ""),
            category_key=category.get("category_key", ""),
            dominant_hexes=[],
            accent_hexes=[],
            color_hints=[],
            color_profile={},
            color_mask_source="",
            fashion_color_classifier={},
            color_sampling_mask_meta={},
        )
    except Exception:
        return {
            "schema_version": "garment_metadata.v1",
            "prompt": {
                "base_garment_prompt": prompt_description,
                "prompt_description": prompt_description,
                "extraction_avoid_clause": "",
                "prompt_sections_raw": "",
                "descriptor_raw_text": prompt_description,
                "source": prompt_source,
            },
            "classification": {
                "target_type": target_type,
                "backend_target_type": target_type,
                "style": str(category.get("style", "") or ""),
                "primary_category_key": str(category.get("primary_category_key", "") or ""),
                "category_key": str(category.get("category_key", "") or ""),
            },
            "color": {
                "dominant_hexes": [],
                "accent_hexes": [],
                "color_hints": [],
                "profile": {},
                "mask_source": "",
                "resolved_source": "",
                "signal_confidence": 0.0,
                "signal_strength": "",
                "fashion_basecolour": {},
                "rich": {},
            },
            "details": {},
        }


def _run_minicpm_prompt(
    *,
    minicpm_runner: Any,
    image: Image.Image,
    garment_type: Optional[str],
) -> str:
    if minicpm_runner is None:
        raise PromptGenerationFailedError("MiniCPM runner is unavailable for prompt generation.")
    try:
        described = minicpm_runner.describe_garment(
            image=image.convert("RGB"),
            garment_type=garment_type,
            prompt_override=None,
        )
    except Exception as exc:
        raise PromptGenerationFailedError(f"MiniCPM prompt generation failed: {exc}") from exc
    return _normalize_text(described)


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
    minicpm_runner: Optional[Any],
    upload_image_fn: Optional[Callable[..., str]] = None,
    output_dir: str = "/tmp/qwen_extract_outfit_outputs",
) -> Dict[str, object]:
    """
    Independent function #2:
    Execute one normalized request and return a reusable API payload block.
    """
    source = _resize_to_max_edge(source_image.convert("RGB"), request.max_input_edge)
    minicpm_garment_type = _infer_garment_type(request.prompt)
    output_width, output_height = _resolve_output_size(
        source_width=source.width,
        source_height=source.height,
        max_edge=request.output_max_edge,
        output_aspect_ratio=request.output_aspect_ratio,
    )
    prompt_started_at = time.time()
    input_prompt_error = None
    try:
        # Match /analyze behavior: run MiniCPM as its own stage (no Qwen GPU contention).
        input_prompt_description = _run_minicpm_prompt(
            minicpm_runner=minicpm_runner,
            image=source,
            garment_type=minicpm_garment_type,
        )
    except Exception as exc:
        input_prompt_description = ""
        input_prompt_error = exc

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

    prompt_description = ""
    prompt_source = ""
    prompt_fallback_used = False
    if _is_prompt_usable(input_prompt_description):
        prompt_description = input_prompt_description
        prompt_source = PROMPT_SOURCE_INPUT_PARALLEL
    else:
        prompt_fallback_used = True
        fallback_error = None
        try:
            fallback_prompt_description = _run_minicpm_prompt(
                minicpm_runner=minicpm_runner,
                image=output_image,
                garment_type=minicpm_garment_type,
            )
        except Exception as exc:
            fallback_prompt_description = ""
            fallback_error = exc
        if _is_prompt_usable(fallback_prompt_description):
            prompt_description = fallback_prompt_description
            prompt_source = PROMPT_SOURCE_EXTRACTED_FALLBACK
        else:
            reasons = []
            if input_prompt_error is not None:
                reasons.append(f"input_prompt_error={input_prompt_error}")
            else:
                reasons.append("input_prompt_invalid")
            if fallback_error is not None:
                reasons.append(f"fallback_prompt_error={fallback_error}")
            else:
                reasons.append("fallback_prompt_invalid")
            detail = ", ".join(reasons)
            raise PromptGenerationFailedError(
                f"{PROMPT_GENERATION_FAILED_CODE}: failed to generate usable prompt ({detail})"
            )
    prompt_elapsed_seconds = round(float(time.time() - prompt_started_at), 3)
    garment_metadata = _build_analyze_style_garment_metadata(
        prompt_description=prompt_description,
        prompt_source=prompt_source,
    )

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
        "promptDescription": prompt_description,
        "promptDescriptionSource": prompt_source,
        "garmentMetadata": garment_metadata,
        "promptElapsedSeconds": prompt_elapsed_seconds,
        "promptFallbackUsed": bool(prompt_fallback_used),
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
