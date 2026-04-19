"""
Reusable request/build helpers for Qwen Extract-Outfit inference.
"""

from __future__ import annotations

import base64
import io
import json
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
PROMPT_SOURCE_QWEN_FASTPATH = "qwen_fastpath_default"
_INVALID_PROMPT_TOKENS = {"none", "n/a", "unknown", "no garment"}
_DIRECTIONAL_TERMS = {
    "left",
    "right",
    "front",
    "back",
    "side",
    "one side",
    "opposite side",
    "viewer-left",
    "viewer-right",
}
_COLOR_TERMS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "orange",
    "brown",
    "purple",
    "gray",
    "grey",
    "beige",
}


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


def _default_prompt_description_for_type(garment_type: str) -> str:
    normalized = normalize_garment_type(garment_type) or "top"
    if normalized == "bottom":
        return "A bottom garment with visible waistband, leg structure, and hem details."
    if normalized == "dress":
        return "A dress garment with visible bodice, waist, skirt structure, and hem details."
    if normalized == "outer":
        return "An outerwear garment with visible front opening, sleeve structure, and hem details."
    return "A top garment with visible neckline, sleeve structure, and hem details."


def _default_subtype_for_type(garment_type: str) -> str:
    normalized = normalize_garment_type(garment_type) or "top"
    if normalized == "bottom":
        return "bottom"
    if normalized == "dress":
        return "dress"
    if normalized == "outer":
        return "outer"
    return "top"


def _extract_json_object_text(raw_text: str) -> Optional[str]:
    text = str(raw_text or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    return text[start : end + 1]


def _normalize_subtype(raw_subtype: Any, garment_type: str) -> str:
    text = _normalize_text(raw_subtype)
    if not text:
        return _default_subtype_for_type(garment_type)
    text = re.sub(r"[^a-zA-Z0-9 _-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower().replace(" ", "_")
    if not text:
        return _default_subtype_for_type(garment_type)
    return text


def _sanitize_construction_prompt(raw_prompt: Any) -> str:
    text = _normalize_text(raw_prompt)
    if not text:
        return ""
    for directional in _DIRECTIONAL_TERMS:
        text = re.sub(rf"\b{re.escape(directional)}\b", " ", text, flags=re.IGNORECASE)
    for color in _COLOR_TERMS:
        text = re.sub(rf"\b{re.escape(color)}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    return text


def _parse_minicpm_prompt_contract(raw_text: Any, garment_type: str) -> Dict[str, object]:
    raw = str(raw_text or "").strip()
    subtype = _default_subtype_for_type(garment_type)

    json_valid = False
    fallback_used = False
    construction_prompt = ""

    payload: Optional[Dict[str, object]] = None
    json_text = _extract_json_object_text(raw)
    if json_text:
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = None

    if payload:
        subtype = _normalize_subtype(
            payload.get("category_type")
            or payload.get("garment_category_subtype")
            or payload.get("top_category_subtype")
            or payload.get("bottom_category_subtype")
            or payload.get("dress_category_subtype")
            or payload.get("outer_category_subtype"),
            garment_type,
        )
        construction_prompt = _sanitize_construction_prompt(payload.get("garment_construction_prompt"))
        json_valid = bool(_is_prompt_usable(construction_prompt))
        fallback_used = not json_valid

    if not construction_prompt:
        fallback_used = True
        construction_prompt = _sanitize_construction_prompt(raw)

    if not _is_prompt_usable(construction_prompt):
        fallback_used = True
        construction_prompt = ""

    return {
        "category_type": subtype,
        "garment_category_subtype": subtype,
        "garment_construction_prompt": construction_prompt,
        "json_valid": bool(json_valid),
        "fallback_used": bool(fallback_used),
        "raw_text": _normalize_text(raw),
    }


def _render_qwen_prompt_with_subtype(prompt_template: str, garment_category_subtype: str) -> str:
    template = str(prompt_template or "").strip()
    subtype = _normalize_text(garment_category_subtype)
    if "{garment_category_subtype}" not in template and "{category_type}" not in template:
        return template
    replacement = subtype or "garment"
    template = template.replace("{garment_category_subtype}", replacement)
    template = template.replace("{category_type}", replacement)
    return template


def _run_minicpm_prompt(
    *,
    minicpm_runner: Any,
    image: Image.Image,
    garment_type: Optional[str],
) -> Dict[str, object]:
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
    return _parse_minicpm_prompt_contract(
        described,
        garment_type=(normalize_garment_type(garment_type) or "top"),
    )


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
    enable_minicpm_prompt_override: Optional[bool] = None,
) -> Dict[str, object]:
    """
    Independent function #2:
    Execute one normalized request and return a reusable API payload block.
    """
    request_started_at = time.time()
    source_original = source_image.convert("RGB")
    source_original_width, source_original_height = source_original.size
    source = _resize_to_max_edge(source_original, request.max_input_edge)
    minicpm_garment_type = _infer_garment_type(request.prompt)
    if enable_minicpm_prompt_override is None:
        enable_minicpm_prompt = _is_truthy(os.getenv("QWEN_EXTRACT_ENABLE_MINICPM_PROMPT", "1"))
    else:
        enable_minicpm_prompt = bool(enable_minicpm_prompt_override)
    output_width, output_height = _resolve_output_size(
        source_width=source.width,
        source_height=source.height,
        max_edge=request.output_max_edge,
        output_aspect_ratio=request.output_aspect_ratio,
    )
    minicpm_elapsed_seconds = 0.0
    input_prompt_error = None
    input_prompt_contract: Dict[str, object] = {}
    if enable_minicpm_prompt:
        try:
            minicpm_started_at = time.time()
            # Match /analyze behavior: run MiniCPM as its own stage (no Qwen GPU contention).
            input_prompt_contract = _run_minicpm_prompt(
                minicpm_runner=minicpm_runner,
                image=source,
                garment_type=minicpm_garment_type,
            )
            minicpm_elapsed_seconds += max(0.0, time.time() - minicpm_started_at)
        except Exception as exc:
            minicpm_elapsed_seconds += max(0.0, time.time() - locals().get("minicpm_started_at", time.time()))
            input_prompt_contract = {}
            input_prompt_error = exc

    initial_subtype = _normalize_subtype(
        input_prompt_contract.get("category_type") or input_prompt_contract.get("garment_category_subtype"),
        minicpm_garment_type,
    )
    rendered_qwen_prompt = _render_qwen_prompt_with_subtype(request.prompt, initial_subtype)

    qwen_started_at = time.time()
    output_image, meta = runner.run_edit(
        source,
        prompt=rendered_qwen_prompt,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        output_width=output_width,
        output_height=output_height,
    )
    qwen_elapsed_seconds = max(0.0, time.time() - qwen_started_at)

    prompt_description = ""
    prompt_source = ""
    prompt_fallback_used = False
    prompt_contract: Dict[str, object] = {}
    if not enable_minicpm_prompt:
        prompt_description = _default_prompt_description_for_type(minicpm_garment_type)
        prompt_source = PROMPT_SOURCE_QWEN_FASTPATH
        prompt_contract = {
            "category_type": _default_subtype_for_type(minicpm_garment_type),
            "garment_construction_prompt": prompt_description,
            "json_valid": False,
            "fallback_used": True,
            "raw_text": "",
        }
    elif _is_prompt_usable(str(input_prompt_contract.get("garment_construction_prompt") or "")):
        prompt_contract = dict(input_prompt_contract)
        prompt_description = str(prompt_contract.get("garment_construction_prompt") or "").strip()
        prompt_source = PROMPT_SOURCE_INPUT_PARALLEL
    else:
        prompt_fallback_used = True
        fallback_error: Optional[Exception] = None
        fallback_prompt_contract: Dict[str, object] = {}
        try:
            fallback_started_at = time.time()
            fallback_prompt_contract = _run_minicpm_prompt(
                minicpm_runner=minicpm_runner,
                image=output_image,
                garment_type=minicpm_garment_type,
            )
            minicpm_elapsed_seconds += max(0.0, time.time() - fallback_started_at)
        except Exception as exc:
            minicpm_elapsed_seconds += max(0.0, time.time() - locals().get("fallback_started_at", time.time()))
            fallback_prompt_contract = {}
            fallback_error = exc
        if _is_prompt_usable(str(fallback_prompt_contract.get("garment_construction_prompt") or "")):
            prompt_contract = dict(fallback_prompt_contract)
            prompt_description = str(prompt_contract.get("garment_construction_prompt") or "").strip()
            prompt_source = PROMPT_SOURCE_EXTRACTED_FALLBACK
        else:
            prompt_description = _default_prompt_description_for_type(minicpm_garment_type)
            prompt_source = PROMPT_SOURCE_QWEN_FASTPATH
            prompt_contract = {
                "category_type": _default_subtype_for_type(minicpm_garment_type),
                "garment_construction_prompt": prompt_description,
                "json_valid": False,
                "fallback_used": True,
                "raw_text": "",
            }
            if input_prompt_error is not None or fallback_error is not None:
                prompt_fallback_used = True

    if not prompt_contract:
        prompt_contract = {
            "category_type": _default_subtype_for_type(minicpm_garment_type),
            "garment_construction_prompt": prompt_description or _default_prompt_description_for_type(minicpm_garment_type),
            "json_valid": False,
            "fallback_used": True,
            "raw_text": "",
        }
    resolved_subtype = _normalize_subtype(
        prompt_contract.get("category_type") or prompt_contract.get("garment_category_subtype"),
        minicpm_garment_type,
    )
    if not prompt_description:
        prompt_description = str(prompt_contract.get("garment_construction_prompt") or "").strip()
    prompt_elapsed_seconds = round(float(minicpm_elapsed_seconds), 3)
    total_elapsed_seconds = round(float(time.time() - request_started_at), 3)
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
    aligned_output_size = dict(response_meta.get("requested_output_size_aligned") or {})
    if not aligned_output_size:
        aligned_output_size = dict(response_meta.get("requested_output_size") or {})
    response_meta["prompt_template"] = request.prompt
    response_meta["prompt"] = rendered_qwen_prompt
    response_meta["prompt_default_applied"] = bool(request.prompt_default_applied)
    response_meta["input_original_size"] = {
        "width": int(source_original_width),
        "height": int(source_original_height),
    }
    response_meta["input_preprocessed_size"] = {"width": int(source.width), "height": int(source.height)}
    # Backward-compatible alias.
    response_meta["input_size"] = {"width": int(source.width), "height": int(source.height)}
    response_meta["requested_output_size"] = {"width": int(output_width), "height": int(output_height)}
    response_meta["requested_output_size_aligned"] = {
        "width": int(aligned_output_size.get("width") or output_width),
        "height": int(aligned_output_size.get("height") or output_height),
    }
    response_meta["output_size"] = {"width": int(output_image.width), "height": int(output_image.height)}
    response_meta["elapsed_seconds"] = round(float(qwen_elapsed_seconds), 3)
    response_meta["max_input_edge"] = int(request.max_input_edge)
    response_meta["max_output_edge"] = int(request.output_max_edge)
    response_meta["output_aspect_ratio"] = str(request.output_aspect_ratio or "")
    response_meta["minicpm_prompt_enabled"] = bool(enable_minicpm_prompt)
    response_meta["minicpm_elapsed_seconds"] = round(float(minicpm_elapsed_seconds), 3)
    response_meta["minicpm_json_valid"] = bool(prompt_contract.get("json_valid"))
    response_meta["minicpm_json_fallback_used"] = bool(prompt_contract.get("fallback_used"))
    response_meta["normalized_category_type"] = resolved_subtype
    response_meta["normalized_garment_category_subtype"] = resolved_subtype
    response_meta["qwen_elapsed_seconds"] = round(float(qwen_elapsed_seconds), 3)
    response_meta["total_elapsed_seconds"] = total_elapsed_seconds

    return {
        "output_url": output_url,
        "local_path": local_path,
        "image_base64": image_base64,
        "promptDescription": prompt_description,
        "promptDescriptionSource": prompt_source,
        "categoryType": resolved_subtype,
        "garmentCategorySubtype": resolved_subtype,
        "garmentMetadata": garment_metadata,
        "promptElapsedSeconds": prompt_elapsed_seconds,
        "promptFallbackUsed": bool(prompt_fallback_used),
        "minicpmJsonValid": bool(prompt_contract.get("json_valid")),
        "minicpmJsonFallbackUsed": bool(prompt_contract.get("fallback_used")),
        "minicpmElapsedSeconds": round(float(minicpm_elapsed_seconds), 3),
        "qwenElapsedSeconds": round(float(qwen_elapsed_seconds), 3),
        "totalElapsedSeconds": total_elapsed_seconds,
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
