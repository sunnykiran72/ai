"""
Runtime compatibility helpers extracted from main.py.

Centralized shims used by tests and API adapters during the refactor.
"""

import logging
import os
import re
from dataclasses import replace
from typing import Optional, List, Dict, Tuple

import numpy as np
from PIL import Image

from shared.image_ops import binary_open, binary_close, rgb_to_lab

from core.garment_color_context import (
    GarmentColorContextSettings,
    build_single_image_color_context as _shared_build_single_image_color_context,
    build_visual_lock_clauses as _shared_build_visual_lock_clauses,
)

from utils import color_processing as color_processing_mod
from utils.prompt_generation import (
    parse_structured_descriptor as _parse_structured_descriptor,
    parse_garment_prompt_sections as _parse_garment_prompt_sections,
    build_florence_contamination_avoid_clause as _build_florence_contamination_avoid_clause,
    extract_generation_only_avoid_directives as _extract_generation_only_avoid_directives,
    merge_avoid_clause_sentences as _merge_avoid_clause_sentences,
    extract_prompt_fact_segments as _extract_prompt_fact_segments,
    serialize_prompt_fact_segments as _serialize_prompt_fact_segments,
)
from utils.validation import (
    normalize_garment_type as _normalize_garment_type,
    sanitize_prompt_fact_value as _sanitize_prompt_fact_value,
    descriptor_is_weak as _descriptor_is_weak_base,
    descriptor_word_count as _descriptor_word_count_base,
    canonical_coverage_for_type as _canonical_coverage_for_type,
)
from utils.color_processing import (
    canonical_color_token as _canonical_color_token,
    color_family as _color_family,
    is_neutral_color_token as _is_neutral_color_token,
    palette_weighted_hue_deg as _palette_weighted_hue_deg,
    palette_hue_from_hexes as _palette_hue_from_hexes,
    filter_palette_entries_by_area as _filter_palette_entries_by_area,
    bucket_color_brightness as _bucket_color_brightness,
    bucket_color_saturation as _bucket_color_saturation,
    bucket_color_undertone as _bucket_color_undertone,
    hex_to_rgb_triplet as _hex_to_rgb_triplet,
)
from shared.category_mapping import (
    wardrobe_category_from_garment_type as _wardrobe_category_from_garment_type,
    infer_style_from_text as _infer_style_from_text,
)

logger = logging.getLogger("glamify-ai")

# Injected by main.py at runtime for legacy shims
engine = None
estimate_type_focused_color_mask_fn = None
from utils.image_preprocessing import (
    mask_connected_components as _mask_connected_components,
    bbox_from_mask as _bbox_from_mask,
    bbox_x_overlap_ratio as _bbox_x_overlap_ratio,
)

# Runtime-configured constants (synced by main.py wrappers)
_MINICPM_SERVICE_URL_EXPLICIT = False
_ANALYZE_MINICPM_SERVICE_URL_EXPLICIT = False

MINICPM_SERVICE_URL = ""
ANALYZE_MINICPM_SERVICE_URL = ""
MINICPM_SERVICE_GARMENT_MIN_WORDS = 0

FLUX2_DESCRIPTOR_BACKEND = "minicpm"
FLUX2_ALLOW_QWEN_BACKEND = False
FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE = "auto"
FLUX2_TRYON_DISABLE_RUNTIME_NEGATIVE_PROMPT = False

FLUX2_COLOR_LOCK_TOP_K = 3
FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT = 0.02
FLUX2_COLOR_DECONTAMINATION_ENABLED = True
FLUX2_COLOR_DECONTAM_ALPHA_HIGH = 255
FLUX2_COLOR_DECONTAM_ALPHA_LOW = 0
FLUX2_COLOR_DECONTAM_ERODE_ITERS = 1
FLUX2_COLOR_DECONTAM_MIN_PIXELS = 128
FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO = 0.02
FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE = 0.04
FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE = 0.02
FLUX2_DETAIL_LOCK_ENABLED = True

FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE = 1536
FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE = 512
FLUX2_MINICPM_USER_CAPTION_MAX_SIDE = 1536
FLUX2_MINICPM_USER_CAPTION_MIN_SIDE = 512
FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE = 1536
FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE = 512

COLOR_CONTEXT_DISABLE_MASKING = False
GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = True

ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE = 0.90
ANALYZE_COLOR_PARSER_SAMPLING_TRIAL_ENABLED = True

# Uncertain full-body fallback settings (synced by main.py)
ANALYZE_UNCERTAIN_FULLBODY_TO_DRESS = True
ANALYZE_UNCERTAIN_FULLBODY_MIN_HEIGHT_RATIO = 0.78
ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_RATIO = 0.22
ANALYZE_UNCERTAIN_FULLBODY_MAX_TOP_RATIO = 0.26
ANALYZE_UNCERTAIN_FULLBODY_MIN_BOTTOM_RATIO = 0.90
ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_ADVANTAGE = 1.55

def _descriptor_word_count(text: str) -> int:
    return _descriptor_word_count_base(text)


def _descriptor_is_weak(
    text: str,
    *,
    garment_type: Optional[str] = None,
    min_words: int = MINICPM_SERVICE_GARMENT_MIN_WORDS,
) -> bool:
    return _descriptor_is_weak_base(text, garment_type=garment_type, min_words=min_words)


def _default_minicpm_family_backend() -> str:
    if _ANALYZE_MINICPM_SERVICE_URL_EXPLICIT or _MINICPM_SERVICE_URL_EXPLICIT:
        return "minicpm_service"
    return "minicpm"


def _normalize_descriptor_backend(raw: Optional[str]) -> str:
    value = str(raw or FLUX2_DESCRIPTOR_BACKEND).strip().lower()
    # Florence is intentionally disabled in the refactor; fall back to MiniCPM family.
    if value in {"florence", ""}:
        return _default_minicpm_family_backend()
    if value not in {"qwen2_5_vl", "joycaption", "minicpm", "minicpm_service"}:
        return _default_minicpm_family_backend()
    if value == "qwen2_5_vl" and not FLUX2_ALLOW_QWEN_BACKEND:
        return _default_minicpm_family_backend()
    return value


def _normalize_prompt_descriptor_backend(raw: Optional[str]) -> str:
    value = str(raw or FLUX2_DESCRIPTOR_BACKEND).strip().lower()
    if value not in {"minicpm", "minicpm_service"}:
        return _default_minicpm_family_backend()
    return value


def _resize_for_qwen_caption(image: Image.Image, max_side: int, min_side: int) -> Image.Image:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= 0 or h <= 0:
        return rgb

    longest = max(w, h)
    shortest = min(w, h)
    scale = min(1.0, float(max_side) / float(longest))
    if shortest > min_side and (shortest * scale) < min_side:
        scale = min(1.0, float(min_side) / float(shortest))
    if scale >= 0.999:
        return rgb
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return rgb.resize((new_w, new_h), Image.BICUBIC)


def _minicpm_bundle_has_valid_json_contract(bundle: Optional[Dict[str, str]]) -> bool:
    if not isinstance(bundle, dict):
        return False
    base_prompt = " ".join(str(bundle.get("base_garment_prompt") or "").split()).strip()
    return bool(base_prompt) and str(bundle.get("json_contract_valid") or "").strip().lower() == "true"


def _default_extraction_avoid_clause_for_type(garment_type: Optional[str]) -> str:
    gtype = _normalize_garment_type(garment_type) or "garment"
    return {
        "top": "Ignore skin, hair, face, hands, background, accessories, and lower-body garments.",
        "bottom": "Ignore skin, hands, background, accessories, and upper-body garments.",
        "dress": "Ignore skin, hair, face, hands, legs, background, and accessories.",
        "outer": "Ignore skin, hair, face, hands, background, accessories, and inner-layer garments.",
    }.get(gtype, "Ignore skin, hair, face, body parts, background, accessories, and other garments.")


def _build_minicpm_garment_prompt(
    garment_type: str,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    accent_hexes: Optional[List[str]] = None,
    accent_hints: Optional[List[str]] = None,
) -> str:
    gtype = _normalize_garment_type(garment_type) or "garment"
    type_label = {
        "top": "top",
        "bottom": "bottom",
        "dress": "dress",
        "outer": "outerwear",
    }.get(gtype, "garment")
    hexes = [str(v).strip().upper() for v in (dominant_color_hexes or []) if str(v).strip()]
    hints = [str(v).strip().lower() for v in (color_hints or []) if str(v).strip()]
    accent_hex = [str(v).strip().upper() for v in (accent_hexes or []) if str(v).strip()]
    accent_hint = [str(v).strip().lower() for v in (accent_hints or []) if str(v).strip()]

    color_clause = ""
    if hexes:
        color_clause += " Use the source garment colors exactly with palette lock: " + ", ".join(hexes) + "."
    if hints:
        color_clause += " Keep the color family locked to: " + ", ".join(hints[: max(2, FLUX2_COLOR_LOCK_TOP_K)]) + "."
    if accent_hex:
        color_clause += " Accent palette (if visible): " + ", ".join(accent_hex[:2]) + "."
    if accent_hint:
        color_clause += " Accent color words (if visible): " + ", ".join(accent_hint[:2]) + "."

    return (
        "Describe the product garment for high-fidelity virtual try-on. "
        f"The required garment category is {type_label}. "
        f"Describe only that single {type_label} and ignore every other clothing item or body region. "
        "Return exactly one valid JSON object with keys \"base_garment_prompt\" and \"extraction_avoid_clause\"; "
        "these keys must always be present. "
        "The JSON must contain: base_garment_prompt (one detailed sentence describing type, neckline/opening, "
        "sleeve or strap style, asymmetry if present, silhouette/fit, hem/length, fabric/texture, and visible trims/closures) and "
        "extraction_avoid_clause (short sentence listing what to ignore). "
        "Do not include any extra keys. Do not mention a person, pose, or background. "
        "If a detail is not visible, omit it; do not guess. "
        "Use 1-3 plain color words for the garment fabric; never output hex codes. "
        f"{color_clause}"
    ).strip()


def _build_minicpm_garment_retry_prompt(
    *,
    garment_type: str,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    accent_hexes: Optional[List[str]] = None,
    accent_hints: Optional[List[str]] = None,
) -> str:
    return (
        _build_minicpm_garment_prompt(
            garment_type=garment_type,
            dominant_color_hexes=dominant_color_hexes,
            color_hints=color_hints,
            accent_hexes=accent_hexes,
            accent_hints=accent_hints,
        )
        + " Schema reminder: return a JSON object with keys \"base_garment_prompt\" and \"extraction_avoid_clause\". "
        + "Your previous response did not follow the format."
    )


def _build_minicpm_garment_detail_retry_prompt(
    *,
    garment_type: str,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    accent_hexes: Optional[List[str]] = None,
    accent_hints: Optional[List[str]] = None,
) -> str:
    gtype = _normalize_garment_type(garment_type) or "garment"
    detail_clause = {
        "top": (
            "Your previous response was too generic. Include garment type, neckline or front opening, sleeve configuration, "
            "fit/silhouette, hem behavior, fabric/texture, and any trims/closures."
        ),
        "dress": (
            "Your previous response was too generic. Include neckline or collar, sleeve length, "
            "waist treatment, skirt or hem length, closure or placket if visible, fabric/texture, and pattern summary if present."
        ),
        "bottom": (
            "Your previous response was too generic. Include rise, leg shape, closure, pleats or creases, "
            "pockets if visible, hem behavior, and fabric/texture."
        ),
        "outer": (
            "Your previous response was too generic. Include collar or lapel, sleeve length, "
            "closure details, hem or waist behavior, fabric/texture, and front structure."
        ),
    }.get(gtype, "Your previous response was too generic. Include more visible front-facing garment structure.")
    return (
        _build_minicpm_garment_retry_prompt(
            garment_type=garment_type,
            dominant_color_hexes=dominant_color_hexes,
            color_hints=color_hints,
            accent_hexes=accent_hexes,
            accent_hints=accent_hints,
        )
        + " "
        + detail_clause
    )


def _choose_better_garment_prompt_bundle(
    primary_bundle: Dict[str, str],
    candidate_bundle: Dict[str, str],
    *,
    garment_type: Optional[str] = None,
) -> Dict[str, str]:
    primary_prompt = str(primary_bundle.get("base_garment_prompt") or "")
    candidate_prompt = str(candidate_bundle.get("base_garment_prompt") or "")
    primary_weak = _descriptor_is_weak(primary_prompt, garment_type=garment_type)
    candidate_weak = _descriptor_is_weak(candidate_prompt, garment_type=garment_type)
    if primary_weak and not candidate_weak:
        return candidate_bundle
    if candidate_weak and not primary_weak:
        return primary_bundle
    if _descriptor_word_count(candidate_prompt) > _descriptor_word_count(primary_prompt):
        return candidate_bundle
    return primary_bundle


def _repair_non_json_garment_prompt_bundle(
    *bundles: Optional[Dict[str, str]],
    garment_type: Optional[str] = None,
) -> Dict[str, str]:
    candidates: List[Dict[str, str]] = []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        if " ".join(str(bundle.get("base_garment_prompt") or "").split()).strip():
            candidates.append(dict(bundle))
    if not candidates:
        return _parse_garment_prompt_sections("", garment_type=garment_type)

    repaired = candidates[0]
    for candidate in candidates[1:]:
        repaired = _choose_better_garment_prompt_bundle(
            repaired,
            candidate,
            garment_type=garment_type,
        )

    base_prompt = " ".join(str(repaired.get("base_garment_prompt") or "").split()).strip()
    avoid_clause = " ".join(str(repaired.get("extraction_avoid_clause") or "").split()).strip()
    if not avoid_clause:
        avoid_clause = _default_extraction_avoid_clause_for_type(garment_type)
    if avoid_clause and not avoid_clause.endswith("."):
        avoid_clause = f"{avoid_clause}."

    source_format = str(repaired.get("source_format") or "freeform").strip() or "freeform"
    if not source_format.endswith("_repaired"):
        source_format = f"{source_format}_repaired"

    repaired["base_garment_prompt"] = base_prompt or "Garment."
    repaired["extraction_avoid_clause"] = avoid_clause
    repaired["json_contract_valid"] = "false"
    repaired["source_format"] = source_format
    repaired["serialized_sections"] = (
        f"BASE_GARMENT_PROMPT: {repaired['base_garment_prompt']}\n"
        f"EXTRACTION_AVOID_CLAUSE: {avoid_clause}"
    )
    return repaired


def _ensure_garment_prompt_bundle_avoid_clause(
    bundle: Dict[str, str],
    *,
    garment_type: Optional[str] = None,
) -> Dict[str, str]:
    ensured = dict(bundle or {})
    base_prompt = " ".join(str(ensured.get("base_garment_prompt") or "").split()).strip() or "Garment."
    avoid_clause = " ".join(str(ensured.get("extraction_avoid_clause") or "").split()).strip()
    if not avoid_clause:
        avoid_clause = _default_extraction_avoid_clause_for_type(garment_type)
    if avoid_clause and not avoid_clause.endswith("."):
        avoid_clause = f"{avoid_clause}."
    ensured["base_garment_prompt"] = base_prompt
    ensured["extraction_avoid_clause"] = avoid_clause
    ensured["serialized_sections"] = (
        f"BASE_GARMENT_PROMPT: {base_prompt}\n"
        f"EXTRACTION_AVOID_CLAUSE: {avoid_clause}"
    )
    return ensured


def _apply_florence_avoid_clause(
    bundle: Dict[str, str],
    *,
    image: Image.Image,
    garment_type: Optional[str] = None,
) -> Dict[str, str]:
    gtype = _normalize_garment_type(garment_type) or ""
    try:
        joycaption = getattr(engine, "joycaption", None)
        if joycaption is None:
            return bundle
        instruction = (
            "Describe non-garment elements in the image (person/body parts, accessories, background objects, "
            "other garments). One short sentence."
        )
        joycaption_caption = " ".join(
            str(
                joycaption.describe_garment(
                    image,
                    instruction_override=instruction,
                )
                or ""
            ).split()
        ).strip()
    except Exception as err:
        logger.warning("JoyCaption avoid-clause support failed: %s", err)
        return bundle

    florence_clause = _build_florence_contamination_avoid_clause(
        joycaption_caption,
        garment_type=gtype,
    )
    if not florence_clause:
        return bundle

    existing_clause = str(bundle.get("extraction_avoid_clause") or "").strip()
    generation_directives = _extract_generation_only_avoid_directives(existing_clause)
    merged_clause = _merge_avoid_clause_sentences(
        florence_clause,
        " ".join(generation_directives),
    )
    if not merged_clause:
        return bundle

    updated = dict(bundle)
    updated["extraction_avoid_clause"] = merged_clause
    updated["serialized_sections"] = (
        f"BASE_GARMENT_PROMPT: {updated.get('base_garment_prompt') or 'Garment.'}\n"
        f"EXTRACTION_AVOID_CLAUSE: {merged_clause}"
    )
    updated["avoid_clause_source"] = "joycaption_support"
    return updated


def _build_minicpm_garment_color_prompt(garment_type: str) -> str:
    gtype = _normalize_garment_type(garment_type) or "garment"
    type_label = {
        "top": "top",
        "bottom": "bottom",
        "dress": "dress",
        "outer": "outerwear",
    }.get(gtype, "garment")
    return (
        f"Look only at the requested {type_label}. "
        "Ignore skin, body, face, hair, gloves, jewelry, mannequin, room, background, and lighting. "
        "Return only 1 to 3 short garment fabric color words in plain English, comma-separated. "
        "If the garment is solid, return only one color. "
        "Focus on the main fabric color first, then any true trim or accent color. "
        "Never output hex codes. Never mention skin tone or background color."
    ).strip()


def _describe_garment_color_terms_with_backend(
    image: Image.Image,
    backend: str,
    image_url: Optional[str] = None,
    service_url: Optional[str] = None,
    garment_type: Optional[str] = None,
) -> List[str]:
    resolved = _normalize_descriptor_backend(backend)
    prompt = _build_minicpm_garment_color_prompt(str(garment_type or ""))
    if resolved == "minicpm_service":
        if not image_url:
            return []
        logger.warning("minicpm_service color term extraction not wired in refactor.")
        return []
    if resolved == "minicpm":
        try:
            minicpm_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            text = str(engine.minicpm.describe_garment(minicpm_img, prompt_override=prompt)).strip()
            return _extract_text_color_terms(text, max_items=max(2, FLUX2_COLOR_LOCK_TOP_K))
        except Exception as err:
            logger.warning(f"MiniCPM semantic garment color fallback failed: {err}")
            return []
    return []


def _describe_garment_with_backend(
    image: Image.Image,
    backend: str,
    image_url: Optional[str] = None,
    service_url: Optional[str] = None,
    garment_type: Optional[str] = None,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    accent_hexes: Optional[List[str]] = None,
    accent_hints: Optional[List[str]] = None,
) -> str:
    resolved = _normalize_descriptor_backend(backend)
    if resolved == "minicpm_service":
        if not image_url:
            raise RuntimeError("minicpm_service requires a valid image URL for garment description")
        raise RuntimeError("minicpm_service is not wired in refactor")
    if resolved == "minicpm":
        try:
            minicpm_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            raw_text = str(
                engine.minicpm.describe_garment(
                    minicpm_img,
                    prompt_override=_build_minicpm_garment_prompt(
                        garment_type=str(garment_type or ""),
                        dominant_color_hexes=dominant_color_hexes,
                        color_hints=color_hints,
                        accent_hexes=accent_hexes,
                        accent_hints=accent_hints,
                    ),
                )
            ).strip()
            return _parse_garment_prompt_sections(raw_text, garment_type=garment_type).get(
                "base_garment_prompt", ""
            )
        except Exception as err:
            raise RuntimeError(f"MiniCPM garment description failed: {err}") from err
    if resolved == "joycaption":
        try:
            return str(engine.joycaption.describe_garment(image)).strip()
        except Exception as err:
            logger.warning(f"JoyCaption garment description failed; fallback to MiniCPM. error={err}")
    if resolved == "qwen2_5_vl":
        try:
            qwen_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE,
            )
            return str(engine.qwen25vl.describe_garment(qwen_img)).strip()
        except Exception as err:
            logger.warning(f"Qwen2.5-VL garment description failed; fallback to MiniCPM. error={err}")
    # Final fallback: MiniCPM (local only; service path handled above).
    try:
        minicpm_img = _resize_for_qwen_caption(
            image=image,
            max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
            min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
        )
        return str(engine.minicpm.describe_garment(minicpm_img)).strip()
    except Exception as err:
        raise RuntimeError(f"MiniCPM garment description failed: {err}") from err


def _describe_garment_prompt_bundle_with_backend(
    image: Image.Image,
    backend: str,
    image_url: Optional[str] = None,
    service_url: Optional[str] = None,
    garment_type: Optional[str] = None,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    accent_hexes: Optional[List[str]] = None,
    accent_hints: Optional[List[str]] = None,
) -> Dict[str, str]:
    # Prompt bundle extraction must always use MiniCPM family for base garment prompt.
    resolved = _normalize_prompt_descriptor_backend(backend)

    def _parse_minicpm_bundle_or_retry(fetch_raw_text) -> Dict[str, str]:
        primary_prompt = _build_minicpm_garment_prompt(
            garment_type=str(garment_type or ""),
            dominant_color_hexes=dominant_color_hexes,
            color_hints=color_hints,
            accent_hexes=accent_hexes,
            accent_hints=accent_hints,
        )
        raw_text = str(fetch_raw_text(primary_prompt) or "").strip()
        bundle = _parse_garment_prompt_sections(raw_text, garment_type=garment_type)

        if not _minicpm_bundle_has_valid_json_contract(bundle):
            retry_prompt = _build_minicpm_garment_retry_prompt(
                garment_type=str(garment_type or ""),
                dominant_color_hexes=dominant_color_hexes,
                color_hints=color_hints,
                accent_hexes=accent_hexes,
                accent_hints=accent_hints,
            )
            retry_text = str(fetch_raw_text(retry_prompt) or "").strip()
            retry_bundle = _parse_garment_prompt_sections(retry_text, garment_type=garment_type)
            if _minicpm_bundle_has_valid_json_contract(retry_bundle):
                bundle = retry_bundle
            else:
                logger.warning(
                    "MiniCPM garment response missed the expected format; repairing locally "
                    "(first_format=%s, retry_format=%s)",
                    bundle.get("source_format"),
                    retry_bundle.get("source_format"),
                )
                bundle = _repair_non_json_garment_prompt_bundle(
                    bundle,
                    retry_bundle,
                    garment_type=garment_type,
                )

        if _descriptor_is_weak(str(bundle.get("base_garment_prompt") or ""), garment_type=garment_type):
            detail_prompt = _build_minicpm_garment_detail_retry_prompt(
                garment_type=str(garment_type or ""),
                dominant_color_hexes=dominant_color_hexes,
                color_hints=color_hints,
                accent_hexes=accent_hexes,
                accent_hints=accent_hints,
            )
            detail_text = str(fetch_raw_text(detail_prompt) or "").strip()
            detail_bundle = _parse_garment_prompt_sections(detail_text, garment_type=garment_type)
            if _minicpm_bundle_has_valid_json_contract(detail_bundle):
                bundle = _choose_better_garment_prompt_bundle(
                    bundle,
                    detail_bundle,
                    garment_type=garment_type,
                )
            elif str(detail_bundle.get("base_garment_prompt") or "").strip():
                logger.warning(
                    "MiniCPM garment detail retry stayed non-compliant; repairing locally (format=%s)",
                    detail_bundle.get("source_format"),
                )
                bundle = _choose_better_garment_prompt_bundle(
                    bundle,
                    _repair_non_json_garment_prompt_bundle(detail_bundle, garment_type=garment_type),
                    garment_type=garment_type,
                )
        if not str(bundle.get("base_garment_prompt") or "").strip():
            raise RuntimeError("MiniCPM garment response did not contain a usable base_garment_prompt")
        bundle = _apply_florence_avoid_clause(
            bundle,
            image=image,
            garment_type=garment_type,
        )
        if not str(bundle.get("extraction_avoid_clause") or "").strip():
            bundle = _ensure_garment_prompt_bundle_avoid_clause(bundle, garment_type=garment_type)
        return bundle

    if resolved == "minicpm_service":
        if not image_url:
            raise RuntimeError("minicpm_service requires a valid image URL for garment description")
        raise RuntimeError("minicpm_service is not wired in refactor")
    if resolved == "minicpm":
        try:
            minicpm_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            return _parse_minicpm_bundle_or_retry(
                lambda prompt_override: str(
                    engine.minicpm.describe_garment(
                        minicpm_img,
                        prompt_override=prompt_override,
                    )
                ).strip()
            )
        except Exception as err:
            raise RuntimeError(f"MiniCPM garment description failed: {err}") from err

    desc = _describe_garment_with_backend(
        image=image,
        backend=resolved,
        image_url=image_url,
        service_url=service_url,
        garment_type=garment_type,
        dominant_color_hexes=dominant_color_hexes,
        color_hints=color_hints,
        accent_hexes=accent_hexes,
        accent_hints=accent_hints,
    )
    return _parse_garment_prompt_sections(desc, garment_type=garment_type)


def _identity_only_user_context(user_description: str) -> str:
    text = str(user_description or "").strip()
    if not text:
        return ""

    apparel_terms = (
        "wearing", "wears", "outfit", "clothing", "garment", "dress", "gown", "top", "shirt", "blouse", "jacket", "coat",
        "pants", "trousers", "jeans", "skirt", "shorts", "sneaker", "shoe", "sleeve", "bodice", "bag", "accessory",
    )
    identity_terms = (
        "face", "facial", "hair", "skin", "complexion", "body", "build", "shape", "pose", "posture", "standing", "sitting",
        "hand", "arm", "leg", "height", "age", "eyes", "nose", "mouth", "jaw", "lighting",
    )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    kept: List[str] = []
    for sentence in sentences:
        low = sentence.lower()
        has_apparel = any(term in low for term in apparel_terms)
        has_identity = any(term in low for term in identity_terms)
        if has_identity and not has_apparel:
            kept.append(sentence.strip(" ."))

    if kept:
        cleaned = ". ".join(kept).strip(" .")
        return cleaned

    fallback = re.sub(r"\b(?:wearing|wears|dressed in)\b[^.]*\.?", "", text, flags=re.IGNORECASE)
    fallback = re.sub(
        r"\b(?:dress|gown|top|shirt|blouse|jacket|coat|pants|trousers|jeans|skirt|shorts|sneakers?|shoes?)\b[^.]*\.?",
        "",
        fallback,
        flags=re.IGNORECASE,
    )
    fallback = re.sub(r"\s+", " ", fallback).strip(" ,.")
    if re.search(r"\bis\s*\.\s*$", fallback, flags=re.IGNORECASE) or len(fallback) < 20:
        return "person in image 1"
    return fallback


def _build_flux2_targeted_prompt(
    garment_descriptions: List[str],
    user_description: str,
    target_types: List[str],
    board_mode: str,
    color_lock_clause: str = "",
    detail_lock_clause: str = "",
    transparency_lock_clause: str = "",
    collage_item_clause: str = "",
    preserve_background: bool = True,
) -> str:
    target_hint = ", ".join(garment_descriptions)
    types = {t for t in target_types if t}
    is_dress_mode = ("dress" in types) and len(types) == 1 and len(garment_descriptions) == 1
    target_hint_low = target_hint.lower()
    is_saree_mode = any(token in target_hint_low for token in ("saree", "sari"))
    is_multi = board_mode == "collage"
    identity_context = _identity_only_user_context(user_description)
    if not identity_context or re.search(r"\bis\s*\.\s*$", identity_context, flags=re.IGNORECASE):
        identity_context = "person in image 1"

    prompt = "Identity-preserving photorealistic virtual try-on image edit, not a new photoshoot. "
    if preserve_background:
        prompt += "Keep the original camera framing, background, and lighting from image 1. "
    else:
        prompt += (
            "Keep the original camera framing and subject lighting from image 1, but do not recreate or invent any room, wall, floor, "
            "furniture, scenery, or scene background around the person. Treat image 1 as an isolated subject reference only. "
        )
    prompt += (
        "Use image 1 as strict identity source for face, body shape, skin tone, and pose. "
        f"TRANSFER the {target_hint} from image 2 onto the person in image 1. "
        "Match exact garment attributes from image 2: color tone, print/pattern, neckline, sleeve length, hem length, fit, and trims. "
        "Preserve exact source garment color arrangement from image 2: dominant tone, accent tone, border/trim colors, "
        "brightness depth, undertone, and contrast distribution across the fabric. "
        "Do not invent a new garment design, new pattern, or new fabric. "
    )
    if color_lock_clause:
        prompt += color_lock_clause
    if detail_lock_clause:
        prompt += detail_lock_clause
    if transparency_lock_clause:
        prompt += transparency_lock_clause

    if is_dress_mode:
        prompt += (
            "Treat this as full-body outfit replacement with a single dress. "
            "Replace the original top and bottom garments with the target dress silhouette. "
            "Do not keep, overlay, or blend previous tops, pants, skirts, or shorts under/over the dress. "
            "Completely occlude old clothing in the replaced region. "
            "Do not alter face pixels, hairstyle, or body posture while replacing clothing. "
            "For lower-body fidelity, match exact skirt architecture from image 2: flare start point, "
            "ruffle/tier distribution, hem contour (including asymmetry/high-low), slit position, and train length. "
            "No layering artifacts. Ignore existing clothing design details in image 1 while preserving the person identity. "
        )
        if is_saree_mode:
            prompt += (
                "This is a saree transfer. Use image 2 only for textile attributes: border placement, motif layout, pleat flow, "
                "pallu path, drape layering, and hem fall. Ignore any mannequin, shoulder stump, torso contour, arm pose, elbow, "
                "wrist, palm, fingers, or hidden hand silhouette implied by image 2. Use arm and hand geometry only from image 1. "
            )
    else:
        if "top" in types:
            prompt += "Replace only the upper-body garment region with the target top. "
        if "bottom" in types:
            prompt += "Replace only the lower-body garment region with the target bottom. "
        if "outer" in types:
            prompt += "Replace/add the outerwear layer according to the target item. "
        prompt += (
            "Preserve untargeted garments from image 1 unless a target item explicitly replaces that region. "
            "Do not modify face, hair, hands, skin texture, body shape, pose, camera framing, or background. "
        )
        if types == {"top"}:
            prompt += (
                "Top-only lock: keep lower-body garments unchanged (pants/skirt/shorts/shoes), "
                "with original silhouette, color, and texture from image 1. "
                "Do not edit anything below the natural waistline except minor occlusion cleanup. "
                "Preserve lower-body garment pixels and folds as in image 1. "
            )
        if types == {"bottom"}:
            prompt += (
                "Bottom-only lock: keep upper-body garments unchanged (top/outerwear), "
                "with original silhouette, color, and texture from image 1. "
                "Do not edit anything above the natural waistline except minor occlusion cleanup. "
                "Preserve upper-body garment pixels and folds as in image 1. "
            )
        if types == {"outer"}:
            prompt += (
                "Outerwear-only lock: keep the original lower-body garment completely unchanged "
                "(pants/skirt/shorts/shoes, including silhouette, color, texture, folds, and hem). "
                "Keep the original inner top from image 1 unchanged except where the outerwear naturally covers it. "
                "Do not recolor or redesign the pants, and do not replace the full outfit when only outerwear is targeted. "
            )

    if is_multi:
        prompt += (
            "Image 2 is a multi-item outfit board; apply all listed items together with coherent layering and fit. "
            "Keep each item isolated: no cross-item color bleed, no print transfer, and no texture mixing between items. "
        )
        if collage_item_clause:
            prompt += collage_item_clause

    prompt += (
        "Preserve the exact number of visible arms and hands from image 1. "
        "Never generate an extra hand, extra arm, duplicate fingers, mirrored limb, or floating hand. "
        "Hands must stay naturally attached to the original arms and remain separate from the transferred garment, "
        "except for realistic occlusion caused by the original pose. "
    )

    if is_dress_mode:
        prompt += (
            f"Identity reference from image 1: {identity_context}. "
            "Keep exact same person identity: facial features, skin tone, hair, hands, body proportions, "
            "and scene lighting. Keep exact same head angle, facial expression, and hair silhouette. "
            "No face replacement, no age shift, and no body-shape change. "
            "Ensure realistic fabric drape, seams, folds, and shadows."
        )
    else:
        prompt += (
            f"Person identity reference from image 1: {identity_context}. "
            "Keep exact same person identity: facial features, skin tone, hair, hands, body proportions, "
            "and scene lighting. Keep exact same head angle and facial expression. "
            "Ensure realistic fabric drape, seams, folds, and shadows."
        )
    return prompt


def _build_flux2_runtime_negative_prompt(
    *,
    target_types: List[str],
    board_mode: str,
    custom_negative_prompt: str = "",
) -> str:
    mode = FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE
    custom = " ".join(str(custom_negative_prompt or "").split()).strip()
    if mode == "disabled":
        return ""
    if custom and mode in {"request_only", "request_or_auto"}:
        return custom
    if mode == "request_only":
        return ""

    base_terms: List[str] = [
        "wrong garment color",
        "recolored fabric",
        "over-brightened fabric",
        "washed-out garment color",
        "white cast on garment",
        "metallic sheen",
        "glossy silver cast",
        "wrong border color",
        "wrong trim color",
        "pattern drift",
        "print placement shift",
        "texture swap",
        "incorrect neckline",
        "incorrect sleeve length",
        "incorrect hem length",
        "lost garment structure",
        "missing trims",
        "face identity change",
        "skin tone change",
        "body shape change",
        "extra hand",
        "third hand",
        "duplicate arm",
        "floating hand",
        "hand fused to garment",
        "garment fused to skin",
        "duplicate fingers",
    ]
    if board_mode == "collage" and len(target_types) > 1:
        base_terms.extend(
            [
                "cross-item color bleed",
                "pattern mixing between items",
                "top-bottom swap",
                "wrong item placement",
                "merged garments",
                "fused outfit panels",
                "mixed textures between collage items",
            ]
        )

    return " | ".join([p for p in base_terms if p]).strip()


def _resolve_tryon_runtime_negative_prompt(
    *,
    target_types: List[str],
    board_mode: str,
    custom_negative_prompt: str = "",
) -> Tuple[str, str]:
    if FLUX2_TRYON_DISABLE_RUNTIME_NEGATIVE_PROMPT:
        return "", "disabled"

    runtime_negative_prompt = _build_flux2_runtime_negative_prompt(
        target_types=target_types,
        board_mode=board_mode,
        custom_negative_prompt=custom_negative_prompt,
    )
    if custom_negative_prompt and runtime_negative_prompt:
        return runtime_negative_prompt, "request"
    if runtime_negative_prompt:
        return runtime_negative_prompt, "auto"
    return "", "none"


_COLOR_LABEL_RGB: List[Tuple[str, Tuple[int, int, int]]] = [
    ("black", (20, 20, 20)),
    ("charcoal", (60, 60, 60)),
    ("gray", (128, 128, 128)),
    ("silver", (185, 185, 185)),
    ("white", (245, 245, 245)),
    ("off-white", (242, 242, 236)),
    ("cream", (244, 235, 215)),
    ("ivory", (242, 235, 210)),
    ("beige", (214, 192, 155)),
    ("champagne", (233, 214, 170)),
    ("gold", (212, 175, 55)),
    ("brown", (112, 74, 43)),
    ("red", (190, 40, 40)),
    ("maroon", (120, 35, 45)),
    ("orange", (220, 125, 35)),
    ("yellow", (225, 200, 55)),
    ("green", (52, 135, 64)),
    ("teal", (45, 138, 137)),
    ("blue", (55, 96, 185)),
    ("navy", (35, 52, 95)),
    ("purple", (120, 72, 155)),
    ("plum", (98, 56, 102)),
    ("pink", (214, 120, 165)),
]

_COLOR_LABEL_RGB_MAP: Dict[str, Tuple[int, int, int]] = dict(_COLOR_LABEL_RGB)

_TEXT_COLOR_TERMS: List[str] = [
    "cream",
    "rose gold",
    "blush pink",
    "dusty pink",
    "hot pink",
    "off-white",
    "ivory",
    "champagne",
    "nude",
    "beige",
    "tan",
    "khaki",
    "silver",
    "gold",
    "bronze",
    "black",
    "charcoal",
    "gray",
    "grey",
    "white",
    "red",
    "maroon",
    "burgundy",
    "orange",
    "yellow",
    "green",
    "olive",
    "teal",
    "blue",
    "navy",
    "purple",
    "plum",
    "lavender",
    "pink",
    "peach",
]

_DETAIL_LOCK_TERMS: List[Tuple[str, str]] = [
    ("button", "button count and placement"),
    ("buttons", "button count and placement"),
    ("dotted button", "dotted button detailing"),
    ("dot button", "dotted button detailing"),
    ("stud", "stud detailing"),
    ("lapel", "lapel shape"),
    ("pocket flap", "pocket flap placement"),
    ("border", "border placement"),
    ("sequined", "sequined surface"),
    ("sequin", "sequined surface"),
    ("beaded", "beadwork"),
    ("beading", "beadwork"),
    ("embroidery", "embroidery motifs"),
    ("lace", "lace texture"),
    ("feather", "feather trims"),
    ("ruffle", "ruffle placement"),
    ("pleat", "pleated structure"),
    ("satin", "satin finish"),
    ("silk", "silk-like drape"),
    ("velvet", "velvet texture"),
    ("sheer", "sheer panels"),
    ("mesh", "mesh panels"),
    ("corset", "corset bodice"),
    ("strapless", "strapless neckline"),
    ("off-shoulder", "off-shoulder cut"),
    ("v-neck", "V-neckline"),
    ("deep v", "deep V-neckline"),
    ("sleeveless", "sleeveless cut"),
    ("long sleeve", "long sleeves"),
    ("train", "long train"),
    ("slit", "slit placement"),
    ("mermaid", "mermaid flare profile"),
    ("high-low", "high-low hem shape"),
    ("high low", "high-low hem shape"),
    ("asym", "asymmetric hem contour"),
    ("tier", "tiered skirt layers"),
]


def _extract_text_color_terms(description: str, max_items: int = 4) -> List[str]:
    low = str(description or "").lower()
    if not low:
        return []
    hits: List[str] = []
    for term in _TEXT_COLOR_TERMS:
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, low) and term not in hits:
            normalized = _canonical_color_token(term)
            hits.append(normalized)
            if len(hits) >= max_items:
                break
    return hits


def _profile_is_near_white(profile: Dict[str, object]) -> bool:
    if not isinstance(profile, dict):
        return False
    median_l = profile.get("medianL")
    p90_l = profile.get("p90L")
    mean_c = profile.get("meanChroma")
    if not all(isinstance(v, (int, float)) for v in (median_l, p90_l, mean_c)):
        return False
    return bool(
        profile.get("isNeutral")
        and float(p90_l) >= 84.0
        and float(median_l) >= 68.0
        and float(mean_c) <= 5.5
    )


def _palette_hue_from_hexes(hexes: Optional[List[str]]) -> Optional[float]:
    palette: List[Dict[str, object]] = []
    for idx, hx in enumerate(hexes or []):
        token = str(hx or "").strip()
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", token):
            continue
        palette.append({"hex": token, "areaPercent": max(1.0, float(100 - idx * 10))})
    if not palette:
        return None
    return _palette_weighted_hue_deg(palette, max_colors=min(4, len(palette)))


def _palette_supports_color_family(
    family: str,
    dominant_hexes: Optional[List[str]],
    color_profile: Optional[Dict[str, object]],
) -> bool:
    fam = str(family or "").strip().lower()
    if not fam or fam.startswith("neutral") or fam == "brown":
        return True

    hue = _palette_hue_from_hexes(dominant_hexes)
    profile = color_profile if isinstance(color_profile, dict) else {}
    mean_chroma = float(profile.get("meanChroma") or 0.0)
    mean_a = float(profile.get("meanA") or 0.0)
    mean_b = float(profile.get("meanB") or 0.0)
    median_l = float(profile.get("medianL") or 0.0)

    if fam == "blue":
        return bool(
            hue is not None
            and 170.0 <= float(hue) <= 255.0
            and (mean_b <= -2.0 or mean_chroma >= 14.0)
        )
    if fam == "green":
        return bool(
            hue is not None
            and 70.0 <= float(hue) <= 155.0
            and (mean_a <= -1.0 or mean_b >= 1.5)
        )
    if fam == "yellow":
        return bool(
            hue is not None
            and 35.0 <= float(hue) <= 80.0
            and mean_b >= 8.0
        )
    if fam == "pink":
        return bool(
            hue is not None
            and (float(hue) >= 320.0 or float(hue) <= 25.0)
            and mean_a >= 4.0
            and median_l >= 45.0
        )
    if fam == "red":
        return bool(
            hue is not None
            and (float(hue) >= 345.0 or float(hue) <= 20.0)
            and mean_a >= 8.0
            and median_l < 72.0
        )
    if fam == "purple":
        return bool(hue is not None and 255.0 <= float(hue) <= 330.0 and mean_a >= 3.0)
    if fam == "orange":
        return bool(hue is not None and 15.0 <= float(hue) <= 45.0 and mean_b >= 10.0)
    return True


def _augment_pixel_hints_with_muted_hue_family(
    pixel_hints: List[str],
    dominant_hexes: Optional[List[str]],
    color_profile: Optional[Dict[str, object]],
) -> List[str]:
    ordered = list(dict.fromkeys(str(v).strip().lower() for v in (pixel_hints or []) if str(v).strip()))
    if not ordered:
        return ordered
    if any(not _is_neutral_color_token(term) and _color_family(term) != "brown" for term in ordered):
        return ordered

    profile = color_profile if isinstance(color_profile, dict) else {}
    hue = _palette_hue_from_hexes(dominant_hexes)
    if hue is None:
        return ordered

    mean_chroma = float(profile.get("meanChroma") or 0.0)
    mean_a = float(profile.get("meanA") or 0.0)
    mean_b = float(profile.get("meanB") or 0.0)
    median_l = float(profile.get("medianL") or 0.0)
    p90_l = float(profile.get("p90L") or 0.0)
    near_white = _profile_is_near_white(profile)
    soft_warm_neutrals = {"beige", "champagne", "tan", "nude", "ivory", "cream", "off-white", "white"}
    light_neutral_majority = sum(
        1
        for term in ordered[:3]
        if term in soft_warm_neutrals or term in {"silver", "gray"}
    ) >= 2
    if bool(profile.get("isNeutral")) and mean_chroma <= 14.0 and median_l >= 50.0 and p90_l >= 70.0:
        return ordered
    promoted: Optional[str] = None

    if (
        mean_chroma <= 22.0
        and 5.0 <= float(hue) <= 42.0
        and median_l < 24.0
        and mean_a >= 2.0
        and mean_b >= 4.0
    ):
        promoted = "brown"

    if mean_chroma <= 18.0:
        if (
            70.0 <= float(hue) <= 150.0
            and mean_chroma >= 6.5
            and (mean_a <= -4.0 or mean_b >= 5.0)
            and not near_white
            and not (light_neutral_majority and median_l >= 50.0 and mean_chroma <= 14.0)
        ):
            promoted = "olive" if mean_b >= 7.0 and median_l < 66.0 else "green"
        elif (float(hue) >= 320.0 or float(hue) <= 25.0) and mean_a >= 4.0 and median_l >= 52.0:
            promoted = "pink"
        elif 35.0 <= float(hue) <= 80.0 and mean_b >= 8.0:
            if near_white:
                promoted = None
            elif any(term in soft_warm_neutrals for term in ordered):
                if mean_chroma >= 18.0 and mean_b >= 16.0 and median_l < 72.0:
                    promoted = "yellow"
            elif mean_chroma >= 12.0 and mean_b >= 10.0:
                promoted = "yellow"
        elif 170.0 <= float(hue) <= 255.0 and mean_b <= -2.0 and mean_chroma >= 8.0:
            promoted = "blue"

    if promoted and promoted not in ordered:
        return [promoted] + ordered
    return ordered


def _nearest_color_label(rgb_triplet: Tuple[int, int, int]) -> str:
    r_i, g_i, b_i = [int(v) for v in rgb_triplet]
    lab = rgb_to_lab(np.array([[r_i, g_i, b_i]], dtype=np.uint8))[0]
    l_star = float(lab[0]) * (100.0 / 255.0)
    a_star = float(lab[1]) - 128.0
    b_star = float(lab[2]) - 128.0
    chroma = float(np.sqrt((a_star * a_star) + (b_star * b_star)))

    if 52.0 <= l_star <= 84.0 and chroma >= 26.0 and b_star >= 28.0 and a_star >= -2.0:
        if b_star >= 34.0 or (b_star >= 28.0 and chroma >= 32.0):
            return "gold"

    if (
        45.0 <= l_star < 86.0
        and chroma < 35.0
        and a_star >= 6.0
        and b_star >= 16.0
    ):
        if l_star >= 74.0:
            return "champagne" if b_star < 26.0 else "beige"
        if l_star >= 62.0:
            return "beige"
        if l_star >= 48.0:
            return "tan"
        return "brown"

    if chroma < 14.0:
        if l_star < 38.0 and a_star >= 2.5 and b_star >= 6.0:
            return "brown"
        if l_star < 50.0 and a_star >= 4.0 and b_star >= 10.0:
            return "brown"
        if l_star < 28.0 and b_star <= -5.0:
            return "plum" if a_star >= 9.0 else "navy"
        if l_star < 45.0 and a_star >= 10.0 and b_star <= -2.0:
            return "plum" if l_star < 34.0 else "purple"
        if l_star < 62.0 and chroma >= 8.0 and a_star <= -3.5 and b_star >= 6.0:
            return "olive" if b_star >= 10.0 and l_star < 58.0 else "green"
        if l_star < 10.0:
            return "black"
        if l_star < 28.0:
            return "charcoal"
        if l_star < 62.0:
            return "gray" if b_star < 9.0 else "tan"
        if l_star < 78.0:
            return "gray" if b_star < 10.0 else "tan"
        if l_star < 90.0:
            if chroma < 6.0 and b_star < 8.0 and abs(a_star) < 8.0:
                return "white" if l_star >= 76.0 else "silver"
            return "silver" if b_star < 9.0 else "beige"
        if l_star < 96.0:
            return "cream" if b_star > 11.0 else ("ivory" if b_star > 4.0 else "white")
        return "white"

    vec = np.array(rgb_triplet, dtype=np.float32)
    best_label = "unknown"
    best_dist = float("inf")
    for label, ref_rgb in _COLOR_LABEL_RGB:
        ref = np.array(ref_rgb, dtype=np.float32)
        dist = float(np.sum((vec - ref) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def _color_labels_from_hex_palette(hexes: List[str], top_k: int = 3) -> List[str]:
    out: List[str] = []
    for hx in hexes:
        token = str(hx or "").strip()
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", token):
            continue
        r = int(token[1:3], 16)
        g = int(token[3:5], 16)
        b = int(token[5:7], 16)
        label = _nearest_color_label((r, g, b))
        if label == "grey":
            label = "gray"
        label = _canonical_color_token(label)
        if label and label not in out:
            out.append(label)
        if len(out) >= max(1, int(top_k)):
            break
    return out


def _build_garment_color_tone_guidance(
    *,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    hints = [str(v).strip().lower() for v in (color_hints or []) if str(v).strip()]
    if not hints:
        return {"phrase": "", "negative_terms": []}

    primary = hints[0]
    family = _color_family(primary)
    profile = color_profile if isinstance(color_profile, dict) else {}
    median_l = profile.get("medianL")
    mean_c = profile.get("meanChroma")
    mean_b = profile.get("meanB")
    phrase = primary
    negative_terms: List[str] = []

    if family in {"neutral_light", "neutral_mid"}:
        if (
            isinstance(median_l, (int, float))
            and isinstance(mean_c, (int, float))
            and isinstance(mean_b, (int, float))
            and float(median_l) >= 45.0
            and float(mean_c) <= 18.0
            and float(mean_b) >= 1.0
            and isinstance(profile.get("meanA"), (int, float))
            and float(profile.get("meanA")) <= -2.0
        ):
            phrase = "soft mint green"
            negative_terms = ["gray", "silver", "beige", "ivory", "white", "washed-out fabric"]
            return {
                "phrase": phrase,
                "negative_terms": negative_terms,
            }

    if family == "yellow":
        if (
            isinstance(median_l, (int, float))
            and isinstance(mean_c, (int, float))
            and isinstance(mean_b, (int, float))
            and float(median_l) >= 74.0
            and float(mean_c) >= 28.0
            and float(mean_b) >= 28.0
        ):
            phrase = "bright lemon yellow"
            negative_terms = ["gold", "golden", "mustard", "beige", "champagne", "tan", "bronze", "brown", "orange"]
        elif (
            isinstance(median_l, (int, float))
            and isinstance(mean_c, (int, float))
            and float(median_l) >= 66.0
            and float(mean_c) >= 18.0
        ):
            phrase = "clear yellow"
            negative_terms = ["gold", "mustard", "beige", "champagne", "tan", "brown"]
    elif family == "green":
        if (
            isinstance(mean_c, (int, float))
            and isinstance(median_l, (int, float))
            and float(median_l) >= 64.0
            and float(mean_c) <= 24.0
        ):
            phrase = "soft mint green"
            negative_terms = ["gray", "silver", "gold", "beige", "olive brown", "washed-out green"]
        elif (
            isinstance(mean_c, (int, float))
            and isinstance(median_l, (int, float))
            and float(mean_c) < 18.0
            and float(median_l) >= 48.0
        ):
            phrase = "muted sage green"
            negative_terms = ["gray", "silver", "gold", "beige", "olive brown"]
    elif family == "pink":
        if (
            isinstance(median_l, (int, float))
            and isinstance(mean_c, (int, float))
            and float(median_l) >= 72.0
            and float(mean_c) <= 24.0
        ):
            phrase = "soft blush pink"
            negative_terms = ["brown", "burgundy", "maroon", "gray", "silver", "beige", "tan"]
        elif (
            isinstance(median_l, (int, float))
            and isinstance(mean_c, (int, float))
            and float(median_l) >= 60.0
            and float(mean_c) <= 28.0
        ):
            phrase = "blush pink"
            negative_terms = ["brown", "burgundy", "maroon", "gray", "silver", "tan"]
        elif isinstance(mean_c, (int, float)) and float(mean_c) < 24.0:
            phrase = "dusty pink"
            negative_terms = ["brown", "burgundy", "maroon", "gray", "silver"]

    return {
        "phrase": phrase,
        "negative_terms": negative_terms,
    }


def _compose_color_descriptor_phrase(
    primary_label: str,
    brightness: str,
    saturation: str,
    undertone: str,
) -> str:
    label = str(primary_label or "").strip().lower()
    if not label:
        return ""
    brightness_tokens = {
        "very_light": "bright",
        "light": "light",
        "mid": "",
        "deep": "deep",
        "dark": "dark",
        "unknown": "",
    }
    saturation_tokens = {
        "neutral": "",
        "pale": "pale",
        "muted": "muted",
        "soft": "soft",
        "balanced": "",
        "rich": "rich",
        "vivid": "vivid",
        "unknown": "",
    }
    tokens: List[str] = []
    bright_token = brightness_tokens.get(brightness, "")
    saturation_token = saturation_tokens.get(saturation, "")
    if label in {"white", "off-white", "ivory", "cream"} and bright_token == "bright":
        tokens.append(bright_token)
        if undertone == "warm" and label in {"off-white", "ivory", "cream"}:
            tokens.append("warm")
        tokens.append(label)
        return " ".join(tokens).strip()
    if bright_token:
        tokens.append(bright_token)
    if saturation_token:
        tokens.append(saturation_token)
    if label in {"white", "off-white", "ivory", "cream", "beige", "tan", "champagne", "gray", "silver"} and undertone in {"warm", "cool"}:
        tokens.append(undertone)
    tokens.append(label)
    return " ".join(token for token in tokens if token).strip()


def _estimate_color_signal_confidence(
    *,
    color_mask_source: str = "",
    color_profile: Optional[Dict[str, object]] = None,
    color_sampling_mask_meta: Optional[Dict[str, object]] = None,
    resolved_source: str = "",
) -> float:
    mask_source = str(color_mask_source or "").strip().lower()
    resolved = str(resolved_source or "").strip().lower()
    profile = color_profile if isinstance(color_profile, dict) else {}
    sampling_meta = color_sampling_mask_meta if isinstance(color_sampling_mask_meta, dict) else {}

    score = 0.42
    if mask_source.startswith("parser_strict_runtime"):
        score += 0.30
    elif mask_source.startswith("detector_parser_intersection"):
        score += 0.28
    elif mask_source.startswith("parser_relaxed_runtime"):
        score += 0.22
    elif mask_source.startswith("parser_bottom_refined_runtime"):
        score += 0.20
    elif mask_source.startswith("parser_bottom_spatial_runtime"):
        score += 0.14
    elif mask_source.startswith("detector_mask_runtime"):
        score += 0.08
    elif mask_source.startswith("heuristic"):
        score -= 0.16
    elif mask_source in {"disabled", "none", "mask_error", "error", "color_mask_error"}:
        score -= 0.22

    try:
        mask_pixels = int(sampling_meta.get("mask_pixels", 0) or 0)
    except Exception:
        mask_pixels = 0
    try:
        area_ratio = float(sampling_meta.get("area_ratio", 0.0) or 0.0)
    except Exception:
        area_ratio = 0.0
    if mask_pixels >= 20000:
        score += 0.07
    elif mask_pixels >= 5000:
        score += 0.04
    elif 0 < mask_pixels < 512:
        score -= 0.08
    if area_ratio >= 0.08:
        score += 0.05
    elif area_ratio >= 0.03:
        score += 0.02
    elif 0.0 < area_ratio < 0.006:
        score -= 0.08

    if resolved == "pixel":
        score += 0.06
    elif resolved.startswith("fashion_basecolour"):
        score -= 0.02
    elif resolved.startswith("semantic_"):
        score -= 0.08

    median_l = profile.get("medianL")
    mean_chroma = profile.get("meanChroma")
    p90_l = profile.get("p90L")
    if isinstance(median_l, (int, float)) and isinstance(mean_chroma, (int, float)):
        score += 0.02
        if (
            isinstance(p90_l, (int, float))
            and bool(profile.get("isNeutral"))
            and float(p90_l) >= 70.0
            and float(mean_chroma) <= 14.0
        ):
            score += 0.03

    return max(0.05, min(0.99, float(score)))


def _bucket_color_signal_confidence(score: float) -> str:
    value = float(score or 0.0)
    if value >= 0.74:
        return "high"
    if value >= 0.52:
        return "medium"
    return "low"


def _build_rich_color_metadata(
    *,
    dominant_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
    fashion_color_classifier: Optional[Dict[str, object]] = None,
    color_sampling_mask_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    resolved_hints = [
        str(v).strip().lower()
        for v in (color_hints or [])
        if str(v).strip()
    ]
    primary_label = resolved_hints[0] if resolved_hints else ""
    secondary_label = resolved_hints[1] if len(resolved_hints) > 1 else ""

    if not primary_label:
        signal = fashion_color_classifier if isinstance(fashion_color_classifier, dict) else {}
        primary_label = _canonical_color_token(
            str(signal.get("top_label") or "")
        )
    if not primary_label:
        rgb_triplet = _hex_to_rgb_triplet((dominant_hexes or [""])[0] if dominant_hexes else "")
        if rgb_triplet is not None:
            primary_label = _canonical_color_token(_nearest_color_label(rgb_triplet))
    if not secondary_label and len(resolved_hints) > 1:
        secondary_label = resolved_hints[1]

    brightness = _bucket_color_brightness(color_profile)
    saturation = _bucket_color_saturation(color_profile)
    undertone = _bucket_color_undertone(color_profile)
    primary_family = _color_family(primary_label) if primary_label else ""
    secondary_family = _color_family(secondary_label) if secondary_label else ""
    primary_descriptor = _compose_color_descriptor_phrase(
        primary_label=primary_label,
        brightness=brightness,
        saturation=saturation,
        undertone=undertone,
    )
    secondary_descriptor = ""
    if secondary_label:
        secondary_descriptor = f"{secondary_label} accent"

    classifier_signal = fashion_color_classifier if isinstance(fashion_color_classifier, dict) else {}
    classifier_top = ""
    classifier_score = 0.0
    if classifier_signal:
        classifier_top = str(classifier_signal.get("top_label") or "").strip()
        try:
            classifier_score = float(classifier_signal.get("top_score", 0.0) or 0.0)
        except Exception:
            classifier_score = 0.0

    signal_confidence = _estimate_color_signal_confidence(
        color_mask_source=(
            str(color_sampling_mask_meta.get("source") or "")
            if isinstance(color_sampling_mask_meta, dict)
            else ""
        ),
        color_profile=color_profile if isinstance(color_profile, dict) else None,
        color_sampling_mask_meta=color_sampling_mask_meta if isinstance(color_sampling_mask_meta, dict) else None,
        resolved_source="pixel",
    )
    signal_strength = _bucket_color_signal_confidence(signal_confidence)

    return {
        "primary_color_label": primary_label,
        "primary_color_family": primary_family,
        "secondary_color_label": secondary_label,
        "secondary_color_family": secondary_family,
        "primary_color_descriptor": primary_descriptor,
        "secondary_color_descriptor": secondary_descriptor,
        "brightness": brightness,
        "saturation": saturation,
        "undertone": undertone,
        "descriptor_source": "profile_composition",
        "classifier_top_label": classifier_top,
        "classifier_top_score": round(classifier_score, 6) if classifier_signal else 0.0,
        "signal_confidence": round(float(signal_confidence), 4),
        "signal_strength": signal_strength,
        "sampling_mask": color_sampling_mask_meta if isinstance(color_sampling_mask_meta, dict) else {},
    }


def _sanitize_prompt_fact_fields(
    fields: Dict[str, str],
    *,
    target_type: str = "",
) -> Dict[str, str]:
    cleaned_fields = {
        str(key): _sanitize_prompt_fact_value(str(value or ""))
        for key, value in dict(fields or {}).items()
        if str(key).strip()
    }
    normalized_type = _normalize_garment_type(str(target_type or cleaned_fields.get("category") or ""))

    generic_drop_terms = (
        "tattoo",
        "abdomen",
        "navel",
        "belly",
        "skin",
        "torso",
        "face",
        "hair",
        "hands",
        "fingers",
        "legs",
        "feet",
        "room",
        "background",
        "props",
        "bag",
        "jewelry",
        "phone",
        "mirror",
    )
    type_specific_drop_terms = {
        "top": ("legs", "thigh", "knee", "calf", "ankle", "shoe"),
        "bottom": ("neckline", "sleeve", "strap", "shoulder", "arm", "bust", "chest", "collar"),
        "dress": (),
        "outer": ("legs", "thigh", "knee", "calf", "ankle", "shoe"),
    }
    leak_patterns = (
        r"\b(?:revealing|exposing|showing)\b[^,;.]*",
        r"\b(?:visible|showing)\s+(?:skin|tattoo|abdomen|midriff|navel|belly)\b[^,;.]*",
        r"\bpart of the\b[^,;.]*",
    )

    def _clean_field_segments(value: str, drop_terms: Tuple[str, ...]) -> str:
        cleaned = _sanitize_prompt_fact_value(value)
        if not cleaned:
            return ""
        for pattern in leak_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        segments = [seg.strip(" ,.") for seg in re.split(r"\s*,\s*", cleaned) if seg.strip(" ,.")]
        kept = []
        for seg in segments:
            low = seg.lower()
            if any(term in low for term in generic_drop_terms):
                continue
            if any(term in low for term in drop_terms):
                continue
            kept.append(seg)
        collapsed = ", ".join(kept)
        collapsed = re.sub(r"\s{2,}", " ", collapsed).strip(" ,.")
        return collapsed

    for label in ("construction", "details", "silhouette", "preserve"):
        if cleaned_fields.get(label):
            cleaned_fields[label] = _clean_field_segments(
                cleaned_fields[label],
                type_specific_drop_terms.get(normalized_type, ()),
            )

    canonical_coverage = _canonical_coverage_for_type(normalized_type)
    if canonical_coverage:
        cleaned_fields["coverage"] = canonical_coverage
    elif cleaned_fields.get("coverage"):
        cleaned_fields["coverage"] = _clean_field_segments(cleaned_fields["coverage"], ())

    type_value = cleaned_fields.get("type") or ""
    if normalized_type == "bottom":
        type_value = re.sub(
            r"\b(?:dress|gown|top|shirt|blouse|bodysuit|sleeve|sleeveless|one-shoulder)\b",
            "",
            type_value,
            flags=re.IGNORECASE,
        )
    elif normalized_type in {"top", "outer"}:
        type_value = re.sub(
            r"\b(?:skirt|trouser|trousers|pants|shorts)\b",
            "",
            type_value,
            flags=re.IGNORECASE,
        )
    cleaned_type_value = " ".join(type_value.split()).strip(" ,.-")
    if cleaned_type_value:
        cleaned_fields["type"] = cleaned_type_value

    normalized_type_value = str(cleaned_fields.get("type") or "").lower()
    if normalized_type in {"top", "outer"} and any(token in normalized_type_value for token in ("crop top", "cropped top", "bralette", "bra", "bustier", "corset")):
        for label in ("construction", "details", "silhouette"):
            value = str(cleaned_fields.get(label) or "")
            if not value:
                continue
            segments = [seg.strip(" ,.") for seg in re.split(r"\s*,\s*", value) if seg.strip(" ,.")]
            kept = []
            for seg in segments:
                low = seg.lower()
                if any(term in low for term in ("slit", "cutout", "wrap")):
                    continue
                kept.append(seg)
            cleaned = ", ".join(kept).strip(" ,.")
            if cleaned:
                cleaned_fields[label] = cleaned
            else:
                cleaned_fields.pop(label, None)

    return {
        key: value
        for key, value in cleaned_fields.items()
        if str(value or "").strip()
    }


def _extract_garment_descriptor_facts(
    *,
    base_garment_prompt: str,
    descriptor_raw_text: str = "",
) -> Dict[str, str]:
    raw_fields = _parse_structured_descriptor(descriptor_raw_text)
    prompt_fields = _extract_prompt_fact_segments(base_garment_prompt)
    merged: Dict[str, str] = {}
    for key in (
        "category",
        "type",
        "colors",
        "pattern",
        "material",
        "silhouette",
        "asymmetry",
        "construction",
        "details",
        "coverage",
        "preserve",
    ):
        value = str(raw_fields.get(key) or prompt_fields.get(key) or "").strip()
        if value:
            merged[key] = _sanitize_prompt_fact_value(value)
    return merged


def _resolve_garment_color_truth(
    *,
    base_garment_prompt: str,
    descriptor_raw_text: str = "",
    target_type: str = "",
    dominant_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
    color_mask_source: str = "",
    color_sampling_mask_meta: Optional[Dict[str, object]] = None,
    fashion_color_classifier: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    structured_prompt_fields = dict(_parse_structured_descriptor(base_garment_prompt))

    def _extract_color_terms(text: str) -> List[str]:
        prompt_fields_local = dict(_parse_structured_descriptor(text))
        prompt_fields_local.update(_extract_prompt_fact_segments(text))
        prompt_color_text_local = str(prompt_fields_local.get("colors") or "").strip()
        terms: List[str] = []
        if prompt_color_text_local:
            for raw_piece in re.split(r",|/|\band\b|&", prompt_color_text_local, flags=re.IGNORECASE):
                clean_piece = _canonical_color_token(raw_piece)
                if clean_piece and clean_piece in _TEXT_COLOR_TERMS and clean_piece not in terms:
                    terms.append(clean_piece)
                    continue
                for term in _extract_text_color_terms(str(raw_piece or ""), max_items=2):
                    clean_term = _canonical_color_token(term)
                    if clean_term and clean_term not in terms:
                        terms.append(clean_term)
        if not terms:
            terms = [
                term for term in _extract_text_color_terms(str(prompt_color_text_local or text or ""), max_items=4)
                if str(term).strip()
            ]
        return list(dict.fromkeys(terms))

    prompt_fields = dict(structured_prompt_fields)
    prompt_fields.update(_extract_prompt_fact_segments(base_garment_prompt))
    prompt_color_terms = _extract_color_terms(base_garment_prompt)
    descriptor_color_terms = _extract_color_terms(descriptor_raw_text)
    prompt_non_neutral = [
        term for term in prompt_color_terms
        if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}
    ]
    descriptor_non_neutral = [
        term for term in descriptor_color_terms
        if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}
    ]

    pixel_hexes = [
        str(v).strip().upper()
        for v in (dominant_hexes or [])
        if str(v).strip()
    ]
    pixel_hints = [
        str(v).strip().lower()
        for v in (color_hints or [])
        if str(v).strip()
    ]
    pixel_hints = list(dict.fromkeys(pixel_hints))
    mask_source_norm = str(color_mask_source or "").strip().lower()
    signal_confidence = _estimate_color_signal_confidence(
        color_mask_source=color_mask_source,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
        color_sampling_mask_meta=color_sampling_mask_meta if isinstance(color_sampling_mask_meta, dict) else None,
        resolved_source="pixel",
    )
    weak_mask_source = bool(
        mask_source_norm.startswith("heuristic")
        or mask_source_norm in {"color_mask_error", "mask_error", "error", "disabled", "none"}
        or float(signal_confidence) < 0.48
    )
    if weak_mask_source and pixel_hexes:
        pixel_from_hex = _color_labels_from_hex_palette(pixel_hexes, top_k=max(1, FLUX2_COLOR_LOCK_TOP_K))
        if pixel_from_hex:
            should_override = not pixel_hints
            if not should_override:
                for term in pixel_hints:
                    family = _color_family(term)
                    if family and not _palette_supports_color_family(family, pixel_hexes, color_profile):
                        should_override = True
                        break
            if should_override:
                pixel_hints = pixel_from_hex
    pixel_hints = _augment_pixel_hints_with_muted_hue_family(pixel_hints, pixel_hexes, color_profile)
    pixel_non_neutral = [
        term for term in pixel_hints
        if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}
    ]

    resolved_hints = list(pixel_hints)
    resolved_hexes = list(pixel_hexes)
    resolved_source = "pixel"

    prompt_families = {_color_family(term) for term in prompt_non_neutral if term}
    descriptor_families = {_color_family(term) for term in descriptor_non_neutral if term}
    pixel_families = {_color_family(term) for term in pixel_non_neutral if term}
    semantic_terms = descriptor_color_terms or prompt_color_terms
    semantic_non_neutral = descriptor_non_neutral or prompt_non_neutral
    semantic_families = descriptor_families or prompt_families
    light_neutral_semantic_terms = [
        term for term in semantic_terms
        if term in {"white", "off-white", "ivory", "cream"}
    ]
    semantic_supported = bool(
        not semantic_families
        or any(_palette_supports_color_family(family, pixel_hexes, color_profile) for family in semantic_families)
    )
    semantic_override_source = (
        "semantic_prompt_override"
        if (not descriptor_color_terms or descriptor_color_terms == prompt_color_terms)
        else "semantic_descriptor_override"
    )
    strong_semantic_override = bool(
        GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED
        and semantic_non_neutral
        and semantic_supported
        and (
            str(color_mask_source or "").strip().lower().startswith("heuristic")
            or
            not pixel_non_neutral
            or pixel_families.isdisjoint(semantic_families)
            or all(_is_neutral_color_token(term) for term in pixel_hints[: max(1, len(pixel_hints))])
        )
    )
    weak_mask_non_neutral_rescue = bool(
        GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED
        and weak_mask_source
        and semantic_non_neutral
        and not pixel_non_neutral
        and pixel_hints
        and all(_color_family(term) in {"neutral_dark", "neutral_mid", "neutral_light", "brown"} for term in pixel_hints)
        and (
            not isinstance(color_profile, dict)
            or not bool(color_profile.get("isNeutral"))
            or (
                isinstance(color_profile.get("medianL"), (int, float))
                and float(color_profile.get("medianL")) <= 55.0
            )
        )
    )
    strong_light_neutral_semantic_override = bool(
        GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED
        and weak_mask_source
        and light_neutral_semantic_terms
        and semantic_supported
        and not semantic_non_neutral
        and (
            not pixel_hints
            or all(_color_family(term) in {"neutral_dark", "neutral_mid", "neutral_light", "brown"} for term in pixel_hints)
        )
    )
    if strong_light_neutral_semantic_override:
        semantic_hexes: List[str] = []
        for term in light_neutral_semantic_terms[: max(2, FLUX2_COLOR_LOCK_TOP_K)]:
            rgb = _COLOR_LABEL_RGB_MAP.get(term)
            if rgb:
                semantic_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
                if semantic_hex not in semantic_hexes:
                    semantic_hexes.append(semantic_hex)
        resolved_hints = list(dict.fromkeys(light_neutral_semantic_terms[: max(2, FLUX2_COLOR_LOCK_TOP_K)]))
        resolved_source = semantic_override_source
        if semantic_hexes:
            resolved_hexes = list(semantic_hexes)
    elif weak_mask_non_neutral_rescue:
        semantic_hexes = []
        rescued_terms = list(dict.fromkeys(semantic_terms[: max(2, FLUX2_COLOR_LOCK_TOP_K)]))
        for term in rescued_terms:
            rgb = _COLOR_LABEL_RGB_MAP.get(term)
            if rgb:
                semantic_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
                if semantic_hex not in semantic_hexes:
                    semantic_hexes.append(semantic_hex)
        if rescued_terms:
            resolved_hints = rescued_terms
            resolved_source = semantic_override_source + "_weak_mask_rescue"
        if semantic_hexes:
            resolved_hexes = list(semantic_hexes)
    elif strong_semantic_override:
        semantic_hexes: List[str] = []
        for term in semantic_terms[: max(2, FLUX2_COLOR_LOCK_TOP_K)]:
            rgb = _COLOR_LABEL_RGB_MAP.get(term)
            if rgb:
                semantic_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
                if semantic_hex not in semantic_hexes:
                    semantic_hexes.append(semantic_hex)
        if semantic_terms:
            resolved_hints = list(semantic_terms[: max(2, FLUX2_COLOR_LOCK_TOP_K)])
            resolved_source = semantic_override_source
        if semantic_hexes:
            resolved_hexes = list(semantic_hexes)
    elif (
        GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED
        and semantic_terms
        and semantic_families
        and pixel_families
        and semantic_families.intersection(pixel_families)
        and len(pixel_families - semantic_families) >= 1
    ):
        merged_semantic_terms = list(semantic_terms[: max(2, FLUX2_COLOR_LOCK_TOP_K)])
        merged_neutrals = [
            term for term in pixel_hints
            if _is_neutral_color_token(term) and term not in merged_semantic_terms
        ]
        resolved_hints = list(dict.fromkeys(merged_semantic_terms + merged_neutrals))
        resolved_source = "semantic_descriptor_bias" if descriptor_color_terms else "semantic_prompt_bias"

    def _normalize_resolved_color_hints(
        hints: List[str],
        profile: Optional[Dict[str, object]],
    ) -> List[str]:
        ordered = list(dict.fromkeys(str(v).strip().lower() for v in (hints or []) if str(v).strip()))
        if not ordered:
            return []
        mean_b = profile.get("meanB") if isinstance(profile, dict) else None
        mean_chroma = profile.get("meanChroma") if isinstance(profile, dict) else None
        median_l = profile.get("medianL") if isinstance(profile, dict) else None
        if _profile_is_near_white(profile if isinstance(profile, dict) else {}):
            white_label = "ivory" if isinstance(mean_b, (int, float)) and float(mean_b) >= 4.0 else "white"
            remapped: List[str] = []
            for term in ordered:
                candidate = white_label if term in {"silver", "gray", "off-white"} else term
                if candidate not in remapped:
                    remapped.append(candidate)
            if white_label not in remapped:
                remapped.insert(0, white_label)
            return remapped
        if (
            isinstance(profile, dict)
            and bool(profile.get("isNeutral"))
            and all(term in {"silver", "gray", "off-white", "white", "ivory", "cream"} for term in ordered[:3])
            and isinstance(mean_chroma, (int, float))
            and isinstance(median_l, (int, float))
            and isinstance(mean_b, (int, float))
            and float(median_l) >= 70.0
            and float(mean_chroma) <= 4.5
            and float(mean_b) >= 1.5
        ):
            light_label = "ivory" if float(mean_b) >= 2.5 else "off-white"
            remapped: List[str] = []
            for term in ordered:
                candidate = light_label if term in {"silver", "gray"} else term
                if candidate not in remapped:
                    remapped.append(candidate)
            if light_label not in remapped:
                remapped.insert(0, light_label)
            return remapped
        if (
            weak_mask_source
            and all(term in {"silver", "gray", "off-white", "white", "ivory", "cream"} for term in ordered[:3])
            and isinstance(mean_chroma, (int, float))
            and isinstance(median_l, (int, float))
            and float(mean_chroma) <= 6.5
            and float(median_l) >= 66.0
        ):
            light_label = "white" if float(median_l) >= 76.0 else ("ivory" if isinstance(mean_b, (int, float)) and float(mean_b) >= 2.5 else "off-white")
            remapped: List[str] = []
            for term in ordered:
                candidate = light_label if term in {"silver", "gray"} else term
                if candidate not in remapped:
                    remapped.append(candidate)
            if light_label not in remapped:
                remapped.insert(0, light_label)
            return remapped
        p90_l = profile.get("p90L") if isinstance(profile, dict) else None
        cast_corrected = profile.get("castCorrected") if isinstance(profile, dict) and isinstance(profile.get("castCorrected"), dict) else {}
        cast_corrected_hints = list(
            dict.fromkeys(
                str(v).strip().lower()
                for v in (profile.get("castCorrectedHints") or [])
                if isinstance(profile, dict) and str(v).strip()
            )
        ) if isinstance(profile, dict) else []
        cast_median_l = cast_corrected.get("medianL") if isinstance(cast_corrected, dict) else None
        cast_mean_chroma = cast_corrected.get("meanChroma") if isinstance(cast_corrected, dict) else None
        cast_mean_b = cast_corrected.get("meanB") if isinstance(cast_corrected, dict) else None
        cast_is_neutral = bool(cast_corrected.get("isNeutral")) if isinstance(cast_corrected, dict) else False
        shadowed_light_neutral_terms = {
            "black",
            "charcoal",
            "brown",
            "tan",
            "beige",
            "champagne",
            "silver",
            "gray",
            "off-white",
            "white",
            "ivory",
            "cream",
        }
        parser_primary_shadow_rescue = bool(
            mask_source_norm.startswith("parser_strict_runtime")
            or mask_source_norm.startswith("parser_relaxed_runtime")
            or mask_source_norm.startswith("parser_bottom_refined_runtime")
            or mask_source_norm.startswith("parser_bottom_spatial_runtime")
        )
        if (
            parser_primary_shadow_rescue
            and str(target_type or "").strip().lower() in {"top", "outer", "bottom", "dress"}
            and cast_is_neutral
            and isinstance(cast_mean_chroma, (int, float))
            and isinstance(cast_median_l, (int, float))
            and isinstance(cast_mean_b, (int, float))
            and float(cast_median_l) >= 48.0
            and float(cast_mean_chroma) <= 10.5
            and 2.0 <= float(cast_mean_b) <= 9.5
            and all(term in shadowed_light_neutral_terms for term in (cast_corrected_hints or ordered)[:4])
            and (
                any(term in {"silver", "gray", "white", "off-white", "ivory", "cream"} for term in ordered[:3])
                or any(term in {"silver", "gray", "white", "off-white", "ivory", "cream"} for term in cast_corrected_hints[:3])
            )
        ):
            light_label = "ivory" if float(cast_mean_b) >= 4.0 else "off-white"
            remapped: List[str] = [light_label]
            for term in ordered:
                if term in {"black", "charcoal", "brown", "tan", "gray", "silver"}:
                    continue
                candidate = light_label if term in {"off-white", "white", "cream"} else term
                if candidate not in remapped:
                    remapped.append(candidate)
            if len(remapped) == 1:
                remapped.append("cream" if float(cast_mean_b) >= 5.5 else "off-white")
            return remapped
        if (
            parser_primary_shadow_rescue
            and str(target_type or "").strip().lower() in {"top", "outer", "bottom", "dress"}
            and all(term in shadowed_light_neutral_terms for term in ordered[:4])
            and isinstance(mean_chroma, (int, float))
            and isinstance(median_l, (int, float))
            and isinstance(p90_l, (int, float))
            and isinstance(mean_b, (int, float))
            and bool(profile.get("isNeutral"))
            and float(median_l) >= 48.0
            and float(p90_l) >= 70.0
            and float(mean_chroma) <= 13.5
            and 1.0 <= float(mean_b) <= 10.5
            and (
                (
                    any(term in {"silver", "gray", "white", "off-white", "ivory", "cream"} for term in ordered[:3])
                    and float(p90_l) >= 70.0
                )
                or float(p90_l) >= 84.0
            )
        ):
            light_label = "ivory" if float(mean_b) >= 4.0 else "off-white"
            remapped: List[str] = [light_label]
            for term in ordered:
                if term in {"black", "charcoal", "brown", "tan", "gray", "silver"}:
                    continue
                candidate = light_label if term in {"off-white", "white", "cream"} else term
                if candidate not in remapped:
                    remapped.append(candidate)
            if len(remapped) == 1:
                remapped.append("cream" if float(mean_b) >= 5.5 else "off-white")
            return remapped
        non_neutral = [term for term in ordered if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}]
        if not non_neutral:
            return ordered

        dominant_family = _color_family(non_neutral[0])
        soft_warm_neutrals = {"beige", "champagne", "tan", "nude", "khaki"}
        light_neutrals = soft_warm_neutrals | {"ivory", "cream", "off-white", "white"}
        neutral_mid = {"gray", "silver", "charcoal"}

        filtered = list(ordered)
        light_neutral_majority = sum(
            1
            for term in filtered[:3]
            if term in light_neutrals or term in neutral_mid
        ) >= 2
        if (
            dominant_family == "green"
            and light_neutral_majority
            and isinstance(mean_chroma, (int, float))
            and isinstance(median_l, (int, float))
            and isinstance(mean_b, (int, float))
            and float(mean_chroma) <= 14.0
            and float(median_l) >= 46.0
            and float(mean_b) <= 10.0
        ):
            filtered = [term for term in filtered if _color_family(term) != "green"]
            if not any(term in light_neutrals for term in filtered):
                filtered = ["beige", "champagne"] + filtered
        elif (
            dominant_family in {"yellow", "green"}
            and isinstance(mean_chroma, (int, float))
            and isinstance(mean_b, (int, float))
            and float(mean_chroma) >= 18.0
            and float(mean_b) >= 12.0
        ):
            pruned = [term for term in ordered if term not in soft_warm_neutrals]
            if any(_color_family(term) == dominant_family for term in pruned):
                filtered = pruned
        elif dominant_family == "pink":
            if "white" in filtered:
                filtered = [term for term in filtered if term not in {"gold", "beige", "champagne", "tan", "brown"}]
            elif (
                isinstance(mean_chroma, (int, float))
                and isinstance(median_l, (int, float))
                and float(mean_chroma) <= 22.0
                and float(median_l) >= 62.0
            ):
                filtered = [term for term in filtered if term not in {"gold", "brown", "beige", "champagne", "tan"}]
            if any(_color_family(term) == "pink" for term in filtered):
                filtered = [term for term in filtered if term not in neutral_mid]
        elif dominant_family == "blue":
            if any(_color_family(term) == "blue" for term in filtered):
                filtered = [term for term in filtered if term not in {"brown", "tan", "beige", "champagne", "khaki"}]
                if isinstance(mean_b, (int, float)) and float(mean_b) <= 6.0:
                    filtered = [term for term in filtered if term not in {"gold"}]

        non_neutral_filtered = [term for term in filtered if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}]
        neutral_filtered = [term for term in filtered if term not in non_neutral_filtered]
        return non_neutral_filtered + neutral_filtered

    resolved_hints = _normalize_resolved_color_hints(resolved_hints, color_profile)

    def _apply_fashion_basecolour_signal(
        hints: List[str],
        source: str,
    ) -> Tuple[List[str], str]:
        signal = fashion_color_classifier if isinstance(fashion_color_classifier, dict) else {}
        if not bool(signal.get("applied")):
            return hints, source
        predictions = signal.get("predictions") if isinstance(signal.get("predictions"), list) else []
        top_prediction = predictions[0] if predictions and isinstance(predictions[0], dict) else {}
        top_hint = _canonical_color_token(
            str(
                top_prediction.get("canonical_hint")
                or signal.get("top_label")
                or top_prediction.get("label")
                or ""
            )
        )
        if not top_hint or top_hint in {"multi", "unknown"}:
            return hints, source
        try:
            top_score = float(top_prediction.get("score", signal.get("top_score", 0.0)) or 0.0)
        except Exception:
            top_score = 0.0
        if top_score < float(ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE):
            return hints, source

        top_family = _color_family(top_hint)
        neutralish_families = {"neutral_dark", "neutral_mid", "neutral_light", "brown"}
        palette_supported = (
            top_family in neutralish_families
            or _palette_supports_color_family(top_family, pixel_hexes, color_profile)
        )
        if not palette_supported:
            return hints, source

        existing = list(dict.fromkeys(str(v).strip().lower() for v in (hints or []) if str(v).strip()))
        existing_families = {_color_family(term) for term in existing if term}
        all_existing_neutralish = bool(
            not existing
            or all(_color_family(term) in neutralish_families for term in existing[:3])
        )

        should_apply = bool(
            top_hint in existing
            or (
                (weak_mask_source or float(signal_confidence) < 0.62)
                and (
                    not pixel_non_neutral
                    or all_existing_neutralish
                    or source.startswith("semantic_")
                )
            )
            or (top_family in semantic_families and not pixel_families.intersection(semantic_families))
            or (
                top_family in {"neutral_dark", "neutral_light"}
                and all_existing_neutralish
            )
        )
        if not should_apply:
            return existing, source

        merged: List[str] = [top_hint]
        for term in existing:
            if term == top_hint:
                continue
            if _color_family(term) == top_family:
                continue
            merged.append(term)
        new_source = (
            "fashion_basecolour_blend"
            if (not existing or existing[0] != top_hint or source.startswith("semantic_"))
            else source
        )
        return merged, new_source

    resolved_hints, resolved_source = _apply_fashion_basecolour_signal(resolved_hints, resolved_source)
    resolved_hints = resolved_hints[: max(2, FLUX2_COLOR_LOCK_TOP_K)]
    resolved_hexes = resolved_hexes[: max(2, FLUX2_COLOR_LOCK_TOP_K + 1)]
    resolved_color_text = ", ".join(resolved_hints)

    rebuild_signal_labels = {
        "category",
        "type",
        "pattern",
        "material",
        "silhouette",
        "construction",
        "details",
        "preserve",
    }
    can_rebuild_prompt = bool(
        structured_prompt_fields
        or any(str(prompt_fields.get(label) or "").strip() for label in rebuild_signal_labels)
    )
    reconciled_prompt = ""
    if can_rebuild_prompt:
        reconciled_fields = dict(prompt_fields)
        if resolved_color_text:
            reconciled_fields["colors"] = resolved_color_text
        reconciled_fields = _sanitize_prompt_fact_fields(
            reconciled_fields,
            target_type=target_type,
        )
        reconciled_prompt = _serialize_prompt_fact_segments(reconciled_fields)
        if reconciled_prompt:
            reconciled_prompt = f"{reconciled_prompt}."

    return {
        "base_garment_prompt": reconciled_prompt or " ".join(str(base_garment_prompt or "").split()).strip(),
        "dominant_hexes": resolved_hexes,
        "color_hints": resolved_hints,
        "color_source": resolved_source,
        "color_signal_confidence": round(float(signal_confidence), 4),
        "color_signal_strength": _bucket_color_signal_confidence(signal_confidence),
        "fashion_color_classifier": fashion_color_classifier if isinstance(fashion_color_classifier, dict) else {},
    }


def _build_garment_metadata(
    *,
    base_garment_prompt: str,
    extraction_avoid_clause: str = "",
    prompt_sections_raw: str = "",
    descriptor_raw_text: str = "",
    prompt_description: str = "",
    prompt_source: str = "",
    target_type: str = "",
    backend_target_type: str = "",
    style: str = "",
    primary_category_key: str = "",
    category_key: str = "",
    dominant_hexes: Optional[List[str]] = None,
    accent_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
    color_mask_source: str = "",
    fashion_color_classifier: Optional[Dict[str, object]] = None,
    color_sampling_mask_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    base_prompt = " ".join(str(base_garment_prompt or "").split()).strip()
    avoid_clause = " ".join(str(extraction_avoid_clause or "").split()).strip()
    normalized_prompt = " ".join(str(prompt_description or base_prompt).split()).strip()
    classification_target = str(target_type or "").strip()
    classification_backend = str(backend_target_type or classification_target).strip()
    reconciled_color = _resolve_garment_color_truth(
        base_garment_prompt=base_prompt,
        descriptor_raw_text=descriptor_raw_text,
        target_type=classification_backend or classification_target,
        dominant_hexes=dominant_hexes,
        color_hints=color_hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
        color_mask_source=color_mask_source,
        color_sampling_mask_meta=color_sampling_mask_meta if isinstance(color_sampling_mask_meta, dict) else None,
        fashion_color_classifier=fashion_color_classifier if isinstance(fashion_color_classifier, dict) else None,
    )
    resolved_base_prompt = str(reconciled_color.get("base_garment_prompt") or base_prompt).strip()
    resolved_prompt_description = " ".join(str(prompt_description or resolved_base_prompt).split()).strip()
    if resolved_base_prompt and resolved_prompt_description:
        prompt_fields = _extract_prompt_fact_segments(resolved_prompt_description)
        base_fields = _extract_prompt_fact_segments(resolved_base_prompt)
        if base_fields.get("colors"):
            prompt_fields["colors"] = base_fields["colors"]
            rebuilt_prompt = _serialize_prompt_fact_segments(prompt_fields)
            if rebuilt_prompt:
                resolved_prompt_description = f"{rebuilt_prompt}."

    rich_color = _build_rich_color_metadata(
        dominant_hexes=reconciled_color.get("dominant_hexes") if isinstance(reconciled_color, dict) else dominant_hexes,
        color_hints=reconciled_color.get("color_hints") if isinstance(reconciled_color, dict) else color_hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
        fashion_color_classifier=(
            reconciled_color.get("fashion_color_classifier")
            if isinstance(reconciled_color.get("fashion_color_classifier"), dict)
            else fashion_color_classifier
        ),
        color_sampling_mask_meta=color_sampling_mask_meta if isinstance(color_sampling_mask_meta, dict) else None,
    )

    fact_fields = _extract_garment_descriptor_facts(
        base_garment_prompt=resolved_base_prompt,
        descriptor_raw_text=descriptor_raw_text,
    )
    fact_fields = _sanitize_prompt_fact_fields(
        fact_fields,
        target_type=classification_backend or classification_target,
    )
    return {
        "schema_version": "garment_metadata.v1",
        "prompt": {
            "base_garment_prompt": resolved_base_prompt,
            "prompt_description": resolved_prompt_description or normalized_prompt,
            "extraction_avoid_clause": avoid_clause,
            "prompt_sections_raw": str(prompt_sections_raw or "").strip(),
            "descriptor_raw_text": str(descriptor_raw_text or "").strip(),
            "source": str(prompt_source or "").strip(),
        },
        "classification": {
            "target_type": classification_target,
            "backend_target_type": classification_backend,
            "style": str(style or "").strip(),
            "primary_category_key": str(primary_category_key or "").strip(),
            "category_key": str(category_key or "").strip(),
        },
        "color": {
            "dominant_hexes": [str(v).strip().upper() for v in (reconciled_color.get("dominant_hexes") or []) if str(v).strip()],
            "accent_hexes": [str(v).strip().upper() for v in (accent_hexes or []) if str(v).strip()],
            "color_hints": [str(v).strip().lower() for v in (reconciled_color.get("color_hints") or []) if str(v).strip()],
            "profile": color_profile if isinstance(color_profile, dict) else {},
            "mask_source": str(color_mask_source or "").strip(),
            "resolved_source": str(reconciled_color.get("color_source") or "pixel"),
            "signal_confidence": round(float(reconciled_color.get("color_signal_confidence", rich_color.get("signal_confidence", 0.0)) or 0.0), 4),
            "signal_strength": str(reconciled_color.get("color_signal_strength") or rich_color.get("signal_strength") or "").strip(),
            "fashion_basecolour": (
                reconciled_color.get("fashion_color_classifier")
                if isinstance(reconciled_color.get("fashion_color_classifier"), dict)
                else {}
            ),
            "primary_color_label": str(rich_color.get("primary_color_label") or "").strip(),
            "primary_color_family": str(rich_color.get("primary_color_family") or "").strip(),
            "secondary_color_label": str(rich_color.get("secondary_color_label") or "").strip(),
            "secondary_color_family": str(rich_color.get("secondary_color_family") or "").strip(),
            "primary_color_descriptor": str(rich_color.get("primary_color_descriptor") or "").strip(),
            "secondary_color_descriptor": str(rich_color.get("secondary_color_descriptor") or "").strip(),
            "brightness": str(rich_color.get("brightness") or "").strip(),
            "saturation": str(rich_color.get("saturation") or "").strip(),
            "undertone": str(rich_color.get("undertone") or "").strip(),
            "descriptor_source": str(rich_color.get("descriptor_source") or "").strip(),
            "sampling_mask": rich_color.get("sampling_mask") if isinstance(rich_color.get("sampling_mask"), dict) else {},
        },
        "details": fact_fields,
    }


def _extract_garment_metadata_prompt(garment_metadata: object) -> str:
    if not isinstance(garment_metadata, dict):
        return ""
    prompt_block = garment_metadata.get("prompt")
    if not isinstance(prompt_block, dict):
        return ""
    return " ".join(
        str(
            prompt_block.get("base_garment_prompt")
            or prompt_block.get("prompt_description")
            or ""
        ).split()
    ).strip()


def _extract_garment_metadata_target_type(garment_metadata: object) -> Optional[str]:
    if not isinstance(garment_metadata, dict):
        return None
    classification = garment_metadata.get("classification")
    if not isinstance(classification, dict):
        return None
    for key in ("backend_target_type", "target_type"):
        value = _normalize_garment_type(str(classification.get(key) or ""))
        if value:
            return value
    return None


def _extract_garment_metadata_color_payload(garment_metadata: object) -> Tuple[List[str], List[str]]:
    if not isinstance(garment_metadata, dict):
        return [], []
    color_block = garment_metadata.get("color")
    if not isinstance(color_block, dict):
        return [], []
    dominant_hexes = [
        str(v).strip().upper()
        for v in (color_block.get("dominant_hexes") or [])
        if str(v).strip()
    ]
    color_hints = [
        str(v).strip().lower()
        for v in (color_block.get("color_hints") or [])
        if str(v).strip()
    ]
    return dominant_hexes, color_hints


def _extract_garment_metadata_color_block(garment_metadata: object) -> Dict[str, object]:
    if not isinstance(garment_metadata, dict):
        return {}
    color_block = garment_metadata.get("color")
    if not isinstance(color_block, dict):
        return {}
    dominant_hexes = [
        str(v).strip().upper()
        for v in (color_block.get("dominant_hexes") or [])
        if str(v).strip()
    ]
    color_hints = [
        str(v).strip().lower()
        for v in (color_block.get("color_hints") or [])
        if str(v).strip()
    ]
    accent_hexes = [
        str(v).strip().upper()
        for v in (color_block.get("accent_hexes") or [])
        if str(v).strip()
    ]
    profile = color_block.get("profile") if isinstance(color_block.get("profile"), dict) else {}
    mask_source = str(color_block.get("mask_source") or "").strip()
    sampling_mask = color_block.get("sampling_mask") if isinstance(color_block.get("sampling_mask"), dict) else {}
    try:
        signal_confidence = float(color_block.get("signal_confidence", 0.0) or 0.0)
    except Exception:
        signal_confidence = 0.0
    if signal_confidence <= 0.0:
        signal_confidence = _estimate_color_signal_confidence(
            color_mask_source=mask_source,
            color_profile=profile if isinstance(profile, dict) else None,
            color_sampling_mask_meta=sampling_mask if isinstance(sampling_mask, dict) else None,
            resolved_source=str(color_block.get("resolved_source") or "pixel"),
        )
    return {
        "dominant_hexes": dominant_hexes,
        "color_hints": color_hints,
        "accent_hexes": accent_hexes,
        "profile": profile,
        "mask_source": mask_source,
        "resolved_source": str(color_block.get("resolved_source") or "").strip(),
        "signal_confidence": round(signal_confidence, 4),
        "signal_strength": str(color_block.get("signal_strength") or _bucket_color_signal_confidence(signal_confidence)).strip(),
        "primary_color_label": str(color_block.get("primary_color_label") or "").strip(),
        "primary_color_family": str(color_block.get("primary_color_family") or "").strip(),
        "secondary_color_label": str(color_block.get("secondary_color_label") or "").strip(),
        "secondary_color_family": str(color_block.get("secondary_color_family") or "").strip(),
        "primary_color_descriptor": str(color_block.get("primary_color_descriptor") or "").strip(),
        "secondary_color_descriptor": str(color_block.get("secondary_color_descriptor") or "").strip(),
        "brightness": str(color_block.get("brightness") or "").strip(),
        "saturation": str(color_block.get("saturation") or "").strip(),
        "undertone": str(color_block.get("undertone") or "").strip(),
        "sampling_mask": sampling_mask,
    }


def _should_prefer_metadata_color_signal(color_block: object) -> bool:
    if not isinstance(color_block, dict):
        return False
    score_val = color_block.get("signal_confidence")
    try:
        score = float(score_val or 0.0)
    except Exception:
        score = 0.0
    if score >= 0.58:
        return True
    resolved_source = str(color_block.get("resolved_source") or "").strip().lower()
    mask_source = str(color_block.get("mask_source") or "").strip().lower()
    if resolved_source.startswith("semantic_"):
        return False
    if mask_source.startswith("heuristic"):
        return score >= 0.68
    return bool(score >= 0.52)


def _build_metadata_color_descriptor_lock(
    color_block: object,
    *,
    item_index: int,
    total_items: int,
) -> str:
    if not isinstance(color_block, dict):
        return ""
    if not _should_prefer_metadata_color_signal(color_block):
        return ""

    primary_descriptor = " ".join(str(color_block.get("primary_color_descriptor") or "").split()).strip()
    if not primary_descriptor:
        return ""
    secondary_descriptor = " ".join(str(color_block.get("secondary_color_descriptor") or "").split()).strip()
    undertone = str(color_block.get("undertone") or "").strip().lower()
    saturation = str(color_block.get("saturation") or "").strip().lower()
    brightness = str(color_block.get("brightness") or "").strip().lower()

    prefix = f"item {item_index + 1}: " if total_items > 1 else ""
    clause = prefix + primary_descriptor
    if secondary_descriptor and secondary_descriptor not in {primary_descriptor, ""}:
        clause += f" with {secondary_descriptor}"

    modifiers: List[str] = []
    if brightness in {"very_light", "light", "mid", "deep", "dark"}:
        modifiers.append(f"brightness={brightness}")
    if saturation in {"neutral", "pale", "muted", "soft", "balanced", "rich", "vivid"}:
        modifiers.append(f"saturation={saturation}")
    if undertone in {"warm", "cool", "neutral"}:
        modifiers.append(f"undertone={undertone}")
    if modifiers:
        clause += " (" + ", ".join(modifiers) + ")"
    return clause


def _extract_detail_lock_terms(description: str, max_items: int = 6) -> List[str]:
    low = str(description or "").lower()
    if not low:
        return []
    hits: List[str] = []

    def _needle_present(needle: str) -> bool:
        token = str(needle or "").strip().lower()
        if not token:
            return False
        if token == "asym":
            return bool(re.search(r"\basym(?:metric|metry)?\b", low))
        if (" " in token) or ("-" in token):
            pattern = r"\b" + re.escape(token).replace(r"\ ", r"[\s-]+") + r"\b"
            return bool(re.search(pattern, low))
        pattern = r"\b" + re.escape(token) + r"(?:s|ed|ing)?\b"
        return bool(re.search(pattern, low))

    for needle, phrase in _DETAIL_LOCK_TERMS:
        if _needle_present(needle) and phrase not in hits:
            hits.append(phrase)
            if len(hits) >= max_items:
                break
    return hits


def _has_transparency_signal(text: str) -> bool:
    low = str(text or "").lower()
    if not low:
        return False
    signals = (
        "sheer",
        "mesh",
        "transparent",
        "see-through",
        "see through",
        "translucent",
    )
    return any(sig in low for sig in signals)


def _garment_color_context_settings() -> GarmentColorContextSettings:
    return GarmentColorContextSettings(
        top_k=int(FLUX2_COLOR_LOCK_TOP_K),
        palette_top_k=7,
        palette_min_area_percent=float(FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT),
        decontamination_enabled=bool(FLUX2_COLOR_DECONTAMINATION_ENABLED),
        decontam_alpha_high=int(FLUX2_COLOR_DECONTAM_ALPHA_HIGH),
        decontam_alpha_low=int(FLUX2_COLOR_DECONTAM_ALPHA_LOW),
        decontam_erode_iters=int(FLUX2_COLOR_DECONTAM_ERODE_ITERS),
        decontam_min_pixels=int(FLUX2_COLOR_DECONTAM_MIN_PIXELS),
        decontam_min_coverage_ratio=float(FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO),
        profile_trim_dark_percentile=float(FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE),
        profile_trim_bright_percentile=float(FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE),
        disable_masking=bool(COLOR_CONTEXT_DISABLE_MASKING),
    )


def _build_flux2_visual_lock_clauses(
    product_images: List[Image.Image],
    garment_descriptions: List[str],
    garment_metadata_list: Optional[List[Dict[str, object]]] = None,
) -> dict:
    shared = _shared_build_visual_lock_clauses(
        product_images=product_images,
        garment_descriptions=garment_descriptions,
        settings=_garment_color_context_settings(),
        transparency_signal_fn=_has_transparency_signal,
    )
    detail_terms: List[str] = []
    if FLUX2_DETAIL_LOCK_ENABLED:
        for desc in garment_descriptions:
            for term in _extract_detail_lock_terms(desc):
                if term not in detail_terms:
                    detail_terms.append(term)

    detail_clause = ""
    if detail_terms:
        detail_clause = (
            "Detail lock from image 2: preserve "
            + ", ".join(detail_terms[:6])
            + ". Do not simplify or replace these details. "
        )

    out = dict(shared or {})
    metadata_colors = [
        _extract_garment_metadata_color_block(
            garment_metadata_list[idx] if garment_metadata_list and idx < len(garment_metadata_list) else {}
        )
        for idx in range(len(product_images))
    ]
    if any(block.get("dominant_hexes") or block.get("color_hints") for block in metadata_colors):
        color_palettes = [list(v) for v in (out.get("color_palettes") or [])]
        color_hints = [list(v) for v in (out.get("color_hints") or [])]
        color_profiles = list(out.get("color_profiles") or [])
        color_palette_metrics = [list(v) for v in (out.get("color_palette_metrics") or [])]
        accent_hints = [list(v) for v in (out.get("accent_hints") or [])]
        contexts = list(out.get("contexts") or [])
        max_len = len(product_images)
        while len(color_palettes) < max_len:
            color_palettes.append([])
        while len(color_hints) < max_len:
            color_hints.append([])
        while len(color_profiles) < max_len:
            color_profiles.append({})
        while len(color_palette_metrics) < max_len:
            color_palette_metrics.append([])
        while len(accent_hints) < max_len:
            accent_hints.append([])
        while len(contexts) < max_len:
            contexts.append({})

        for idx, block in enumerate(metadata_colors):
            meta_hexes = [str(v).strip().upper() for v in (block.get("dominant_hexes") or []) if str(v).strip()]
            meta_hints = [str(v).strip().lower() for v in (block.get("color_hints") or []) if str(v).strip()]
            meta_accents = [str(v).strip().upper() for v in (block.get("accent_hexes") or []) if str(v).strip()]
            meta_profile = block.get("profile") if isinstance(block.get("profile"), dict) else {}
            prefer_metadata = _should_prefer_metadata_color_signal(block)
            if meta_hexes and prefer_metadata:
                color_palettes[idx] = meta_hexes
                color_palette_metrics[idx] = [{"hex": hx, "areaPercent": 0.0, "pixelCount": 0} for hx in meta_hexes]
            if meta_hints and prefer_metadata:
                color_hints[idx] = meta_hints
            if meta_profile and prefer_metadata:
                color_profiles[idx] = meta_profile
            if meta_accents and prefer_metadata:
                accent_hints[idx] = []
            ctx = contexts[idx] if isinstance(contexts[idx], dict) else {}
            if meta_hexes and prefer_metadata:
                ctx["dominantHexes"] = meta_hexes
                ctx["paletteHexes"] = meta_hexes
            if meta_hints and prefer_metadata:
                ctx["colorHints"] = meta_hints
                ctx["hints"] = meta_hints
            if meta_accents and prefer_metadata:
                ctx["accentHexes"] = meta_accents
            if meta_profile and prefer_metadata:
                ctx["profile"] = meta_profile
            if block.get("mask_source"):
                ctx["maskSource"] = block.get("mask_source")
            if block.get("signal_confidence") is not None:
                ctx["signalConfidence"] = block.get("signal_confidence")
            if block.get("primary_color_descriptor"):
                ctx["primaryColorDescriptor"] = block.get("primary_color_descriptor")
            contexts[idx] = ctx

        out["color_palettes"] = color_palettes
        out["color_hints"] = color_hints
        out["color_profiles"] = color_profiles
        out["color_palette_metrics"] = color_palette_metrics
        out["accent_hints"] = accent_hints
        out["contexts"] = contexts

        non_empty_color_hints = [colors for colors in color_hints if colors]
        color_clause = ""
        if non_empty_color_hints:
            if len(non_empty_color_hints) == 1:
                palette = ", ".join(non_empty_color_hints[0])
                color_clause = (
                    f"Color lock from image 2: keep the garment in {palette} tones only. "
                    "Do not recolor, hue-shift, or replace with a different color family. "
                )
            else:
                per_item = []
                for idx, colors in enumerate(color_hints, start=1):
                    if colors:
                        per_item.append(f"item {idx}: {', '.join(colors)}")
                if per_item:
                    color_clause = (
                        "Color lock from image 2 for each product: "
                        + "; ".join(per_item)
                        + ". Do not recolor any item. "
                    )

        neutral_lock_items: List[str] = []
        max_items = max(len(color_hints), len(color_palette_metrics), len(color_profiles))
        for idx in range(max_items):
            colors = color_hints[idx] if idx < len(color_hints) else []
            profile = color_profiles[idx] if idx < len(color_profiles) else {}
            profile_is_neutral = bool(profile.get("isNeutral"))
            head = colors[:3]
            neutral_count = sum(1 for color in head if _is_neutral_color_token(color))
            colors_look_neutral = bool(head) and (neutral_count >= max(1, (len(head) + 1) // 2))
            if not (colors_look_neutral or profile_is_neutral):
                continue
            palette = color_palette_metrics[idx] if idx < len(color_palette_metrics) else []
            dominant_hexes: List[str] = []
            for entry in palette[:3]:
                hx = str(entry.get("hex", "")).strip().upper()
                if re.fullmatch(r"#[0-9A-F]{6}", hx):
                    dominant_hexes.append(hx)
            target_l = profile.get("medianL")
            target_c = profile.get("meanChroma")
            target_b = profile.get("meanB")
            profile_tag = ""
            if isinstance(target_l, (int, float)) and isinstance(target_c, (int, float)):
                profile_tag = f", L*~{float(target_l):.1f}, C*~{float(target_c):.1f}"
                if isinstance(target_b, (int, float)):
                    profile_tag += f", b*~{float(target_b):.1f}"
            if dominant_hexes:
                neutral_lock_items.append(f"item {idx + 1}: {'/'.join(dominant_hexes)}{profile_tag}")
            else:
                neutral_lock_items.append(f"item {idx + 1}: neutral tones{profile_tag}")
        if neutral_lock_items:
            color_clause += (
                "Neutral tone lock: preserve exact lightness depth and undertone from image 2 ("
                + "; ".join(neutral_lock_items)
                + "). Avoid over-brightening to white/cream, avoid darkening to charcoal, "
                + "and avoid metallic/glossy silver sheen unless explicitly present in source fabric. "
                + "Keep neutral luminance close to source (roughly +/-4 L*). "
            )

        profile_lock_items: List[str] = []
        for idx in range(max_items):
            profile = color_profiles[idx] if idx < len(color_profiles) else {}
            if not isinstance(profile, dict):
                continue
            median_l = profile.get("medianL")
            mean_c = profile.get("meanChroma")
            if not (isinstance(median_l, (int, float)) and isinstance(mean_c, (int, float))):
                continue
            palette = color_palette_metrics[idx] if idx < len(color_palette_metrics) else []
            hue_deg = _palette_weighted_hue_deg(palette, max_colors=4)
            if _profile_is_near_white(profile):
                profile_lock_items.append(
                    f"item {idx + 1}: near-white base (L*~{float(median_l):.1f}, C*~{float(mean_c):.1f})"
                )
                continue
            if float(mean_c) >= 18.0 and isinstance(hue_deg, (int, float)):
                profile_lock_items.append(
                    f"item {idx + 1}: hue~{float(hue_deg):.0f} deg, L*~{float(median_l):.1f}, C*~{float(mean_c):.1f}"
                )
        if profile_lock_items:
            color_clause += (
                "Tone and saturation lock from image 2: preserve hue/lightness/saturation profile ("
                + "; ".join(profile_lock_items)
                + "). Avoid over-saturating, neon amplification, or warm/cool hue drift. "
                + "Do not shift yellow toward orange/red, and do not shift white/off-white toward gray/silver. "
            )
        descriptor_lock_items: List[str] = []
        total_items = len(metadata_colors)
        for idx, block in enumerate(metadata_colors):
            descriptor_clause = _build_metadata_color_descriptor_lock(
                block,
                item_index=idx,
                total_items=total_items,
            )
            if descriptor_clause:
                descriptor_lock_items.append(descriptor_clause)
        if descriptor_lock_items:
            if len(descriptor_lock_items) == 1:
                color_clause += (
                    "Descriptive color lock from analyze metadata: preserve the garment as "
                    + descriptor_lock_items[0]
                    + ". Do not neutralize or flatten this color impression. "
                )
            else:
                color_clause += (
                    "Descriptive color lock from analyze metadata: "
                    + "; ".join(descriptor_lock_items)
                    + ". Keep each item's brightness, undertone, and saturation aligned to source. "
                )
        out["color_clause"] = color_clause

    out["detail_clause"] = detail_clause
    out["detail_terms"] = detail_terms[:6]
    return out


def _build_single_image_color_context(
    image: Image.Image,
    description: str = "",
    mask: Optional[np.ndarray] = None,
    top_k: int = 7,
    force_masking: bool = False,
) -> Dict[str, object]:
    settings = _garment_color_context_settings()
    effective_mask = None if COLOR_CONTEXT_DISABLE_MASKING else mask
    if force_masking:
        settings = replace(settings, disable_masking=False)
        effective_mask = mask
    return _shared_build_single_image_color_context(
        image=image,
        description=description,
        mask=effective_mask,
        settings=settings,
        top_k=top_k,
    )


def _parser_candidate_category_text(garment_type: str, subtype: Optional[str] = None) -> str:
    gt = _normalize_garment_type(garment_type) or "top"
    st = str(subtype or "").strip().lower()
    if gt == "bottom":
        if st in {"pants", "trousers"}:
            return "pants"
        if st == "shorts":
            return "shorts"
        if st == "skirt":
            return "skirt"
        return "bottom"
    if gt in {"top", "dress", "outer"}:
        return gt
    return "garment"


def _build_flux2_single_garment_extract_prompt(
    garment_type: str,
    prompt_description: str,
    extraction_avoid_clause: str = "",
    category_text: Optional[str] = None,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
    accent_hexes: Optional[List[str]] = None,
    accent_hints: Optional[List[str]] = None,
) -> str:
    gtype = _normalize_garment_type(garment_type) or "top"
    clean_desc = " ".join(str(prompt_description or "").split()).strip()
    low_desc = clean_desc.lower()

    inferred_subtype = ""
    subtype_map = (
        ("saree", "saree"),
        ("sari", "saree"),
        ("lehenga", "lehenga"),
        ("anarkali", "anarkali"),
        ("blazer", "blazer"),
        ("jacket", "jacket"),
        ("coat", "coat"),
        ("bralette", "bralette"),
        ("bra", "bra"),
        ("bustier", "bustier"),
        ("corset", "corset"),
        ("blouse", "blouse"),
        ("shirt", "shirt"),
        ("skirt", "skirt"),
        ("trouser", "trousers"),
        ("pants", "pants"),
    )
    for needle, subtype in subtype_map:
        if needle in low_desc:
            inferred_subtype = subtype
            break

    explicit_category = str(category_text or inferred_subtype or _parser_candidate_category_text(gtype)).strip().lower()
    if not clean_desc:
        clean_desc = (
            "category=unknown; type=unknown; colors=unknown; pattern=unknown; material=unknown; "
            "silhouette=unknown; construction=unknown; details=unknown; coverage=unknown; preserve=unknown"
        )
    color_lock_clause = ""
    hexes = [str(v).strip() for v in (dominant_color_hexes or []) if str(v).strip()]
    if hexes:
        color_lock_clause = (
            f" Keep exact source colors with strict palette lock: {', '.join(hexes)}. "
            "Do not recolor and do not shift hue/saturation/value."
        )
    color_hint_clause = ""
    hints = [str(v).strip().lower() for v in (color_hints or []) if str(v).strip()]
    if hints:
        color_hint_clause = (
            " Keep color-family fidelity to source tones: "
            + ", ".join(hints[: max(2, FLUX2_COLOR_LOCK_TOP_K)])
            + ". "
        )
    accent_clause = ""
    accent_hex = [str(v).strip() for v in (accent_hexes or []) if str(v).strip()]
    accent_hint = [str(v).strip().lower() for v in (accent_hints or []) if str(v).strip()]
    if accent_hex:
        accent_clause += " Preserve accent palette if visible: " + ", ".join(accent_hex[:2]) + ". "
    if accent_hint:
        accent_clause += " Preserve accent color words if visible: " + ", ".join(accent_hint[:2]) + ". "
    tone_guidance = _build_garment_color_tone_guidance(
        color_hints=hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
    )
    color_tone_clause = ""
    tone_phrase = str(tone_guidance.get("phrase") or "").strip()
    tone_negative_terms = [str(v).strip() for v in (tone_guidance.get("negative_terms") or []) if str(v).strip()]
    if tone_phrase:
        color_tone_clause += f" Match the source garment as {tone_phrase} exactly. "
    if tone_negative_terms:
        color_tone_clause += "Do not reinterpret this color as " + ", ".join(tone_negative_terms) + ". "
    profile_clause = ""
    profile_obj = color_profile if isinstance(color_profile, dict) else {}
    if profile_obj:
        median_l = profile_obj.get("medianL")
        mean_c = profile_obj.get("meanChroma")
        if isinstance(median_l, (int, float)) and isinstance(mean_c, (int, float)):
            profile_clause = (
                f" Preserve source tone profile (target L*≈{float(median_l):.1f}, C*≈{float(mean_c):.1f}); "
                "avoid washout, over-darkening, or metallic sheen drift. "
            )

    structural_markers = {
        "pockets": ("pocket",),
        "buttons": ("button", "snap", "stud"),
        "zipper": ("zipper", "zip"),
        "placket": ("placket",),
        "belt": ("belt", "belted"),
        "cutout": ("cutout",),
        "slit": ("slit",),
        "ruffles": ("ruffle", "ruffled"),
        "pleats": ("pleat", "pleated"),
        "bows": ("bow",),
    }
    visible_structure = {
        name for name, needles in structural_markers.items()
        if any(needle in low_desc for needle in needles)
    }
    preserve_visible_clause = ""
    if visible_structure:
        preserve_visible_clause = (
            " Preserve only the visible source features: "
            + ", ".join(sorted(visible_structure))
            + ". "
        )
    structural_absent = [
        name for name in ("pockets", "buttons", "zipper", "placket", "belt", "cutout", "slit", "bows")
        if name not in visible_structure
    ]
    no_invention_clause = ""
    if structural_absent:
        no_invention_clause = (
            " Do not invent "
            + ", ".join(structural_absent)
            + ", or any extra bands, panels, layered sections, or closures not clearly visible in the source. "
        )

    type_lock_clause = {
        "top": (
            "Generate only a top garment. Never generate bottoms, dress silhouettes, legs, or shoes. "
            "Keep neckline, sleeve geometry, shoulder width, and hem shape identical to reference. "
            "Keep the color uniform across all top panels and drapes; do not add a faded, aged, or shadow-tinted secondary shade."
        ),
        "bottom": (
            "Generate only a bottom garment. Never generate tops, jackets, dresses, torso, face, or arms. "
            "Keep waistline, rise, leg width, inseam length, and hem opening identical to reference."
        ),
        "dress": (
            "Generate only a single dress garment with full one-piece replacement behavior. "
            "Never generate separate top+bottom combinations or layered two-piece outfits. "
            "Keep bodice-to-skirt continuity, waist seam placement, and hem length identical to reference."
        ),
    }.get(gtype, "Generate only one garment from the requested category.")

    subtype_lock_clause = ""
    if inferred_subtype == "saree":
        subtype_lock_clause = (
            " This garment is a saree. Generate a full saree drape only, never a blouse-only top, kurti, gown, or generic dress. "
            "Preserve pallu drape, border bands, pleated fall, tassels, textile continuity, and full-length vertical silhouette exactly as in the source."
        )
    elif inferred_subtype == "blazer":
        subtype_lock_clause = (
            " This garment is a blazer. Preserve exact lapel shape, front opening depth, button count, button size, button spacing, button placement, pocket flap placement, shoulder line, and sleeve length."
        )
    elif inferred_subtype in {"bra", "bralette", "bustier", "corset"}:
        subtype_lock_clause = (
            f" This garment is a {inferred_subtype}. Preserve exact cup shape, neckline, strap geometry, closure placement, and bust contour from the source."
        )
    if any(token in low_desc for token in ("crop top", "cropped top", "bralette", "bra", "bustier", "corset")):
        subtype_lock_clause += (
            " Preserve the exact cropped hemline from the source. "
            "The garment must end at the original cropped waist hem and must not extend into a full-length top or tunic. "
            "Do not generate any detached waistband, extra lower strip, separate abdominal band, second garment section below the hem, or extended torso panel."
        )
    if any(token in low_desc for token in ("one-shoulder", "one shoulder", "single shoulder", "single-shoulder")):
        subtype_lock_clause += (
            " Preserve exactly one shoulder connection and one sleeve/strap layout only. "
            "Do not generate a second strap, second shoulder panel, mirrored shoulder edge, mirrored sleeve cap, or any extra lower torso band. "
            "Do not close the open side or complete the missing shoulder into a symmetric top."
        )
    descriptor_clause = ""
    if clean_desc:
        descriptor_clause = (
            f" Use this exact garment description as the single source of structure and visible details: {clean_desc} "
            "Do not omit any described elements and do not add elements not described. "
        )
    clean_avoid_clause = " ".join(str(extraction_avoid_clause or "").split()).strip()
    avoid_clause = ""
    if clean_avoid_clause:
        avoid_clause = f" Extraction-only avoid requirements: {clean_avoid_clause}"

    return (
        "Extract and regenerate a standalone ecommerce product image from the provided garment crop. "
        "This is garment extraction only, not person generation. "
        "No person, no mannequin, no body parts, no skin regions, no neck, no shoulders, no torso, no arms, no hands, no legs, no feet, "
        "no hanger, no props, no text, no watermark. "
        "Use pure white background (RGB 255,255,255). Keep one centered garment with full silhouette visible. "
        "Output must be vertically composed in 2:3 ratio and center aligned. "
        f"Generate only one {explicit_category} category garment and nothing from other categories. "
        "Do not generate any artificial fashion variant, redesign, or alternate styling. "
        "Reconstruct only the exact source garment visible in the crop. "
        f"{descriptor_clause}{type_lock_clause}{subtype_lock_clause}{color_lock_clause}{color_hint_clause}{accent_clause}{color_tone_clause}{profile_clause}{preserve_visible_clause}{no_invention_clause}"
        "Preserve exact garment structure, fabric, texture, print placement, and only the seams, trims, and closures that are clearly visible in the source. "
        "Preserve the exact source color and material appearance; do not brighten black garments into gray, silver, or white. "
        "Do not include visible limbs, face, torso, neck, shoulders, hands, fingers, legs, feet, or any human remnants in the output garment region. "
        f"{avoid_clause}"
    )


def _build_flux2_single_garment_extract_negative_prompt(
    *,
    garment_type: str,
    extraction_avoid_clause: str = "",
    custom_negative_prompt: str = "",
    prompt_description: str = "",
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
) -> str:
    gtype = _normalize_garment_type(garment_type) or "top"
    custom = " ".join(str(custom_negative_prompt or "").split()).strip()
    prompt_low = " ".join(str(prompt_description or "").split()).strip().lower()
    parts: List[str] = []
    if custom:
        parts.append(custom)
    if extraction_avoid_clause:
        parts.append(extraction_avoid_clause)

    parts.append(
        "person, human body, face, eyes, hair, skin, neck, shoulders, chest, torso, hands, fingers, arms, legs, feet, toes, mannequin, hanger, "
        "background scene, props, accessories, bag, jewelry, watermark, text, logo, human silhouette, skin patch, limb fragment, body fragment, arm fragment, leg fragment"
    )
    parts.append(
        "wrong garment color, recolored fabric, hue shift, saturation drift, value drift, pattern drift, print swap, "
        "texture swap, material swap, wrong garment category, extra garment, duplicate garment, merged garments, "
        "broken silhouette, distorted seams, wrong neckline, wrong sleeve length, wrong waistline, wrong hem length, artificial variant, redesigned garment, alternate style, "
        "blur, low detail, overexposed, underexposed"
    )

    if gtype == "top":
        parts.append(
            "pants, trousers, skirt, shorts, dress, lower-body garment, shoes, lower-body structure, bottom garment overlay"
        )
    elif gtype == "bottom":
        parts.append(
            "shirt, blouse, t-shirt, jacket, hoodie, dress, upper-body garment, torso garment overlay, sleeves"
        )
    elif gtype == "dress":
        parts.append(
            "two-piece outfit, separate top and bottom, skirt with separate blouse, trouser plus shirt combo, "
            "incomplete dress replacement, split bodice, split hemline"
        )

    tone_guidance = _build_garment_color_tone_guidance(
        color_hints=color_hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
    )
    tone_negative_terms = [str(v).strip() for v in (tone_guidance.get("negative_terms") or []) if str(v).strip()]
    if tone_negative_terms:
        parts.append(", ".join(tone_negative_terms))

    if any(token in f"{custom.lower()} {prompt_low}" for token in ("one-shoulder", "one shoulder", "single shoulder", "single-shoulder")):
        parts.append(
            "second strap, second sleeve, extra shoulder panel, duplicate shoulder, mirrored shoulder, mirrored strap, "
            "symmetrical neckline, closed open side, faded secondary shade, aged panel tint"
        )
    if any(token in prompt_low for token in ("crop top", "cropped top", "bralette", "bra", "bustier", "corset")):
        parts.append("detached waistband, extra lower strip, separate abdominal band, extra lower torso panel, second garment section below hem, full-length top, tunic length, extended torso panel")
    if "pocket" not in prompt_low:
        parts.append("invented pockets, pocket flaps")
    if all(token not in prompt_low for token in ("button", "snap", "stud")):
        parts.append("invented buttons, snaps, studs")
    if "zip" not in prompt_low:
        parts.append("invented zipper")
    if "placket" not in prompt_low:
        parts.append("invented placket")

    return " | ".join([p for p in parts if p]).strip()


def _parser_category_ids(category: str, fallback: Optional[List[int]] = None) -> List[int]:
    ids: List[int] = []
    if engine.parser and hasattr(engine.parser, "category_ids"):
        try:
            raw_ids = engine.parser.category_ids(category)
            if raw_ids:
                ids.extend(int(v) for v in raw_ids)
        except Exception:
            pass
    if fallback:
        ids.extend(int(v) for v in fallback)
    return sorted({int(v) for v in ids})


def _parser_extraction_keep_ids(garment_type: str) -> List[int]:
    g = _normalize_garment_type(garment_type) or "top"
    fallback_map = {
        "top": [4, 3],
        "outer": [4, 3],
        "bottom": [5, 6],
        "dress": [7],
    }
    return _parser_category_ids(g, fallback_map.get(g, [4]))


def _parser_fallback_ids(garment_type: str) -> List[int]:
    g = _normalize_garment_type(garment_type) or "top"
    return {
        "top": [4, 3],
        "outer": [4, 3],
        "bottom": [5, 6],
        "dress": [7],
    }.get(g, [4])


def _parser_strict_mask_fallback(parsing: np.ndarray, garment_type: str) -> np.ndarray:
    ids = _parser_fallback_ids(garment_type)
    if not ids:
        return np.zeros_like(parsing, dtype=bool)
    mask = np.isin(parsing, ids)
    mask = binary_open(mask, 3)
    mask = binary_close(mask, 3)
    return np.asarray(mask).astype(bool)


def _parser_strict_mask(parsing: np.ndarray, garment_type: str) -> np.ndarray:
    ids = _parser_extraction_keep_ids(garment_type)
    if not ids:
        return np.zeros_like(parsing, dtype=bool)
    mask = np.isin(parsing, ids)
    mask = binary_open(mask, 3)
    mask = binary_close(mask, 3)
    return np.asarray(mask).astype(bool)


def _split_outfit_signature_from_parsing(parsing: np.ndarray) -> Dict[str, object]:
    if not isinstance(parsing, np.ndarray) or parsing.ndim != 2 or parsing.size == 0:
        return {"detected": False, "reason": "bad_parsing"}

    image_height, image_width = parsing.shape[:2]
    total_pixels = float(max(1, image_height * image_width))
    top_mask = _parser_strict_mask(parsing, "top")
    bottom_mask = _parser_strict_mask(parsing, "bottom")
    dress_mask = _parser_strict_mask(parsing, "dress")

    try:
        overlap = np.mean(top_mask & bottom_mask)
        top_area = np.mean(top_mask)
        bottom_area = np.mean(bottom_mask)
        if overlap > 0.15 and top_area > 0.01 and bottom_area > 0.01:
            top_mask = _parser_strict_mask_fallback(parsing, "top")
            bottom_mask = _parser_strict_mask_fallback(parsing, "bottom")
            dress_mask = _parser_strict_mask_fallback(parsing, "dress")
    except Exception:
        pass

    top_area_ratio = float(np.sum(top_mask)) / total_pixels
    bottom_area_ratio = float(np.sum(bottom_mask)) / total_pixels
    dress_area_ratio = float(np.sum(dress_mask)) / total_pixels
    if top_area_ratio < 0.01 or bottom_area_ratio < 0.01:
        return {
            "detected": False,
            "reason": "missing_top_or_bottom",
            "top_area_ratio": round(top_area_ratio, 6),
            "bottom_area_ratio": round(bottom_area_ratio, 6),
            "dress_area_ratio": round(dress_area_ratio, 6),
        }

    top_bbox = _bbox_from_mask(top_mask)
    bottom_bbox = _bbox_from_mask(bottom_mask)
    dress_bbox = _bbox_from_mask(dress_mask)
    if top_bbox is None or bottom_bbox is None:
        return {
            "detected": False,
            "reason": "bbox_missing",
            "top_area_ratio": round(top_area_ratio, 6),
            "bottom_area_ratio": round(bottom_area_ratio, 6),
            "dress_area_ratio": round(dress_area_ratio, 6),
        }

    tx0, ty0, tx1, ty1 = [int(v) for v in top_bbox]
    bx0, by0, bx1, by1 = [int(v) for v in bottom_bbox]
    if ty0 > int(image_height * 0.55) or by1 < int(image_height * 0.45):
        return {
            "detected": False,
            "reason": "regions_not_upper_lower",
            "top_bbox": top_bbox,
            "bottom_bbox": bottom_bbox,
            "dress_bbox": dress_bbox,
        }

    gap_px = max(0, by0 - ty1)
    gap_ratio = float(gap_px) / float(max(1, image_height))
    x_overlap = _bbox_x_overlap_ratio(top_bbox, bottom_bbox)
    center_x0 = min(tx0, bx0)
    center_x1 = max(tx1, bx1)
    center_width = max(1, center_x1 - center_x0)
    centered_cover = (
        tx0 <= int(center_x0 + center_width * 0.18)
        and tx1 >= int(center_x1 - center_width * 0.18)
        and bx0 <= int(center_x0 + center_width * 0.18)
        and bx1 >= int(center_x1 - center_width * 0.18)
    )
    if x_overlap < 0.32 or not centered_cover:
        return {
            "detected": False,
            "reason": "alignment_low",
            "top_bbox": top_bbox,
            "bottom_bbox": bottom_bbox,
            "dress_bbox": dress_bbox,
            "gap_px": int(gap_px),
            "x_overlap": round(float(x_overlap), 4),
        }

    gap_fill_ratio = 0.0
    dress_bridge_ratio = 0.0
    if gap_px > 0:
        x0 = max(0, min(tx0, bx0))
        x1 = min(image_width, max(tx1, bx1))
        gap_union = top_mask | bottom_mask | dress_mask
        gap_slice = gap_union[ty1:by0, x0:x1]
        dress_slice = dress_mask[ty1:by0, x0:x1]
        if gap_slice.size > 0:
            gap_fill_ratio = float(np.mean(gap_slice))
        if dress_slice.size > 0:
            dress_bridge_ratio = float(np.mean(dress_slice))

    detected = bool(
        gap_px >= max(8, int(round(image_height * 0.015)))
        and gap_ratio >= 0.015
        and gap_fill_ratio <= 0.08
        and dress_bridge_ratio <= 0.05
        and dress_area_ratio <= max(0.03, (top_area_ratio + bottom_area_ratio) * 0.55)
    )
    return {
        "detected": detected,
        "reason": "split_gap_detected" if detected else "bridge_present_or_gap_small",
        "top_bbox": top_bbox,
        "bottom_bbox": bottom_bbox,
        "dress_bbox": dress_bbox,
        "gap_px": int(gap_px),
        "gap_ratio": round(gap_ratio, 4),
        "gap_fill_ratio": round(gap_fill_ratio, 4),
        "dress_bridge_ratio": round(dress_bridge_ratio, 4),
        "top_area_ratio": round(top_area_ratio, 6),
        "bottom_area_ratio": round(bottom_area_ratio, 6),
        "dress_area_ratio": round(dress_area_ratio, 6),
        "x_overlap": round(float(x_overlap), 4),
    }


def _item_rank_score(item: dict) -> float:
    conf = item.get("confidence", {})
    if isinstance(conf, dict):
        return float(conf.get("hybrid", conf.get("yolo", 0.0)))
    return 0.0


def _requested_type_geometry_score(item: dict, requested_type: str, image_height: int) -> float:
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or image_height <= 0:
        return _item_rank_score(item)
    _, y0, _, y1 = [int(v) for v in bbox]
    box_h = max(1, y1 - y0)
    center_y = y0 + (box_h / 2.0)
    center_ratio = float(center_y) / float(image_height)
    height_ratio = float(box_h) / float(image_height)
    base = _item_rank_score(item)
    req = _normalize_garment_type(requested_type)

    if req in {"top", "outer"}:
        return base + (1.0 - center_ratio) + (0.25 * min(height_ratio, 0.6))
    if req == "bottom":
        return base + center_ratio + (0.15 * min(height_ratio, 0.75))
    if req == "dress":
        return base + (1.5 * height_ratio) - abs(center_ratio - 0.52)
    return base


def _caption_fullbody_dress_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().lower().split())
    if not t:
        return False
    dress_terms = (
        "dress", "gown", "one-piece", "one piece", "maxi", "midi", "mini",
        "anarkali", "saree", "sari", "lehenga", "jumpsuit", "romper", "kurti",
        "traditional drape", "draped garment", "full-length drape",
    )
    return any(term in t for term in dress_terms)


def _maybe_force_uncertain_fullbody_to_dress(
    items: list[dict],
    *,
    image_width: int,
    image_height: int,
    requested_type: Optional[str] = None,
) -> tuple[list[dict], Optional[dict]]:
    if (
        requested_type
        or not ANALYZE_UNCERTAIN_FULLBODY_TO_DRESS
        or len(items) < 2
        or image_width <= 0
        or image_height <= 0
    ):
        return items, None

    normalized_types = [_normalize_garment_type(str(item.get("type") or "")) for item in items]
    unique_types = {t for t in normalized_types if t}

    candidates: list[tuple[int, dict]] = []
    for idx, item in enumerate(items):
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        area_ratio = float(width * height) / max(1.0, float(image_width * image_height))
        height_ratio = float(height) / float(image_height)
        top_ratio = float(y0) / float(image_height)
        bottom_ratio = float(y1) / float(image_height)
        prompt_text = " ".join(
            str(item.get(key) or "")
            for key in ("promptDescription", "description", "style", "category_key")
        ).strip()
        candidates.append(
            (
                idx,
                {
                    "item": item,
                    "bbox": [x0, y0, x1, y1],
                    "area_ratio": area_ratio,
                    "height_ratio": height_ratio,
                    "top_ratio": top_ratio,
                    "bottom_ratio": bottom_ratio,
                    "type": _normalize_garment_type(str(item.get("type") or "")),
                    "rank_score": _item_rank_score(item),
                    "dress_signal": _caption_fullbody_dress_signal(prompt_text),
                },
            )
        )

    if len(candidates) < 2:
        return items, None

    candidates.sort(key=lambda pair: (pair[1]["area_ratio"], pair[1]["rank_score"]), reverse=True)
    dominant_idx, dominant = candidates[0]
    runner_up_area = float(candidates[1][1]["area_ratio"])
    area_advantage = float(dominant["area_ratio"]) / max(1e-6, runner_up_area)

    dominant_is_fullbody = (
        dominant["height_ratio"] >= ANALYZE_UNCERTAIN_FULLBODY_MIN_HEIGHT_RATIO
        and dominant["area_ratio"] >= ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_RATIO
        and dominant["top_ratio"] <= ANALYZE_UNCERTAIN_FULLBODY_MAX_TOP_RATIO
        and dominant["bottom_ratio"] >= ANALYZE_UNCERTAIN_FULLBODY_MIN_BOTTOM_RATIO
    )
    if not dominant_is_fullbody:
        return items, None

    types_allow_fallback = len(unique_types) <= 1 or dominant["dress_signal"]
    if not types_allow_fallback:
        return items, None

    required_area_advantage = ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_ADVANTAGE
    if dominant["dress_signal"]:
        required_area_advantage = min(required_area_advantage, 1.30)
    if area_advantage < required_area_advantage:
        return items, None

    chosen = dict(items[dominant_idx])
    original_type = chosen.get("type")
    chosen["type_original"] = original_type
    chosen["garment_type_original"] = chosen.get("garment_type")
    chosen["type"] = "dress"
    chosen["garment_type"] = "dress"
    chosen["type_source"] = "uncertain_fullbody_dress_fallback"
    chosen["dress_fallback"] = {
        "triggered": True,
        "dominant_index": int(dominant_idx),
        "area_ratio": round(float(dominant["area_ratio"]), 4),
        "height_ratio": round(float(dominant["height_ratio"]), 4),
        "top_ratio": round(float(dominant["top_ratio"]), 4),
        "bottom_ratio": round(float(dominant["bottom_ratio"]), 4),
        "area_advantage": round(area_advantage, 4),
        "required_area_advantage": round(required_area_advantage, 4),
        "unique_types_before": sorted(unique_types),
        "dress_signal": bool(dominant["dress_signal"]),
    }
    inferred_style = _infer_style_from_text(
        str(chosen.get("promptDescription") or chosen.get("description") or ""),
        garment_type="dress",
    )
    category_meta = _wardrobe_category_from_garment_type("dress", style=inferred_style if inferred_style else None)
    chosen["style"] = category_meta["style"]
    chosen["category_key"] = category_meta["category_key"]
    chosen["primary_category_key"] = category_meta["primary_category_key"]
    return [chosen], chosen["dress_fallback"]


def _skin_like_mask(rgb_image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return np.zeros((0, 0), dtype=bool)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    skin_rgb = (
        (r > 95) & (g > 40) & (b > 20)
        & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15)
        & (np.abs(r - g) > 15)
        & (r > g) & (r > b)
    )
    return skin_rgb.astype(bool)


def _estimate_type_focused_color_mask(
    image: Image.Image,
    garment_type: str,
    description: str = "",
    return_meta: bool = False,
) -> Optional[np.ndarray]:
    try:
        mask, meta = engine.garment_color_masker.estimate_mask(
            image=image,
            garment_type=_normalize_garment_type(str(garment_type or "")) or "top",
            description=description,
        )
        if isinstance(meta, dict) and meta.get("used") is False:
            logger.info("Garment color mask provider skipped: %s", meta)
        resolved_mask = np.asarray(mask).astype(bool) if isinstance(mask, np.ndarray) else None
        if return_meta:
            return resolved_mask, meta
        return resolved_mask
    except Exception as mask_err:
        logger.warning(f"Type-focused color mask fallback triggered: {mask_err}")
        if return_meta:
            return None, {"source": "color_mask_error", "used": False, "reason": str(mask_err)}
        return None


def _cleanup_color_sampling_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(mask).astype(bool)
    if cleaned.ndim != 2 or cleaned.size == 0:
        return cleaned
    opened = binary_open(cleaned, 3)
    closed = binary_close(opened, 3)
    if int(np.sum(closed)) <= 0:
        return cleaned
    return np.asarray(closed).astype(bool)


def _normalize_color_sampling_mask(
    mask: Optional[np.ndarray],
    image_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    if not isinstance(mask, np.ndarray):
        return None
    bool_mask = np.asarray(mask).astype(bool)
    width, height = image_size
    if bool_mask.ndim != 2 or bool_mask.shape[:2] != (height, width):
        return None
    return _cleanup_color_sampling_mask(bool_mask)


def _color_sampling_mask_is_usable(
    mask: Optional[np.ndarray],
    image_size: Tuple[int, int],
    min_area_ratio: float = 0.006,
) -> bool:
    if not isinstance(mask, np.ndarray):
        return False
    width, height = image_size
    bool_mask = np.asarray(mask).astype(bool)
    if bool_mask.ndim != 2 or bool_mask.shape[:2] != (height, width):
        return False
    mask_pixels = int(np.sum(bool_mask))
    if mask_pixels < 96:
        return False
    return (float(mask_pixels) / float(max(1, width * height))) >= float(min_area_ratio)


def _resolve_color_sampling_mask(
    *,
    image: Image.Image,
    garment_type: str,
    description: str = "",
    reference_mask: Optional[np.ndarray] = None,
    apply_type_color_mask: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    fallback_min_area_ratio = 0.02
    ref_mask = _normalize_color_sampling_mask(reference_mask, image.size)
    ref_meta: Dict[str, object] = {}
    if isinstance(ref_mask, np.ndarray):
        ref_meta = {
            "source": "detector_mask_runtime",
            "used": True,
            "reason": "provided_reference_mask",
            "mask_pixels": int(np.sum(ref_mask)),
            "area_ratio": round(float(np.mean(ref_mask)), 6),
        }

    parser_mask = None
    parser_meta: Dict[str, object] = {"source": "disabled", "used": False, "reason": "type_mask_not_requested"}
    if apply_type_color_mask:
        mask_fn = estimate_type_focused_color_mask_fn or _estimate_type_focused_color_mask
        result = mask_fn(
            image,
            garment_type,
            description,
            return_meta=True,
        )
        if isinstance(result, tuple) and len(result) == 2:
            parser_mask, parser_meta = result
        else:
            parser_mask = result
            parser_meta = {
                "source": "type_mask_runtime",
                "used": bool(parser_mask is not None),
                "reason": "type_mask_runtime",
            }
        parser_mask = _normalize_color_sampling_mask(parser_mask, image.size)
        if isinstance(parser_mask, np.ndarray):
            parser_meta = dict(parser_meta or {})
            parser_meta["mask_pixels"] = int(np.sum(parser_mask))
            parser_meta["area_ratio"] = round(float(np.mean(parser_mask)), 6)

    fallback_mask = None
    fallback_meta: Dict[str, object] = {}
    if (
        isinstance(parser_mask, np.ndarray)
        and bool(parser_meta.get("used"))
        and float(parser_meta.get("area_ratio", 0.0) or 0.0) < float(fallback_min_area_ratio)
    ):
        candidate_base = ref_mask if isinstance(ref_mask, np.ndarray) else None
        try:
            candidate = color_processing_mod._get_clean_foreground_mask(image, mask=candidate_base)
        except Exception:
            candidate = None
        if isinstance(candidate, np.ndarray):
            try:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                skin = _skin_like_mask(rgb)
                trimmed = candidate & (~skin)
                if _color_sampling_mask_is_usable(trimmed, image.size, min_area_ratio=fallback_min_area_ratio):
                    candidate = trimmed
            except Exception:
                pass
            if _color_sampling_mask_is_usable(candidate, image.size, min_area_ratio=fallback_min_area_ratio):
                fallback_mask = candidate
                fallback_meta = {
                    "source": "clean_foreground_fallback",
                    "used": True,
                    "reason": "parser_mask_too_small",
                    "mask_pixels": int(np.sum(candidate)),
                    "area_ratio": round(float(np.mean(candidate)), 6),
                    "parser_area_ratio": round(float(parser_meta.get("area_ratio", 0.0) or 0.0), 6),
                }

    if (
        ANALYZE_COLOR_PARSER_SAMPLING_TRIAL_ENABLED
        and isinstance(ref_mask, np.ndarray)
        and isinstance(parser_mask, np.ndarray)
    ):
        intersection = _cleanup_color_sampling_mask(ref_mask & parser_mask)
        intersection_pixels = int(np.sum(intersection))
        parser_pixels = int(np.sum(parser_mask))
        ref_pixels = int(np.sum(ref_mask))
        parser_overlap_ratio = float(intersection_pixels) / float(max(1, parser_pixels))
        ref_overlap_ratio = float(intersection_pixels) / float(max(1, ref_pixels))
        if (
            _color_sampling_mask_is_usable(intersection, image.size, min_area_ratio=0.004)
            and parser_overlap_ratio >= 0.34
            and ref_overlap_ratio >= 0.10
        ):
            return intersection, {
                "source": "detector_parser_intersection",
                "used": True,
                "reason": "parser_trimmed_detector_context",
                "mask_pixels": intersection_pixels,
                "area_ratio": round(float(np.mean(intersection)), 6),
                "parser_overlap_ratio": round(parser_overlap_ratio, 4),
                "reference_overlap_ratio": round(ref_overlap_ratio, 4),
                "reference_mask_pixels": ref_pixels,
                "parser_mask_pixels": parser_pixels,
            }
        if bool(parser_meta.get("used")) and _color_sampling_mask_is_usable(parser_mask, image.size):
            preferred_meta = dict(parser_meta or {})
            preferred_meta["source"] = str(preferred_meta.get("source") or "parser").strip() or "parser"
            preferred_meta["reason"] = "parser_preferred_for_color_sampling"
            preferred_meta["reference_mask_pixels"] = ref_pixels
            return parser_mask, preferred_meta

    if isinstance(ref_mask, np.ndarray):
        return ref_mask, ref_meta
    if isinstance(fallback_mask, np.ndarray):
        return fallback_mask, fallback_meta
    if isinstance(parser_mask, np.ndarray):
        return parser_mask, parser_meta
    if apply_type_color_mask:
        return None, parser_meta
    return None, {"source": "disabled", "used": False, "reason": "no_color_mask"}


def _restore_outer_lower_body_from_reference(
    reference_image: Image.Image,
    output_image: Image.Image,
    target_types: List[str],
    product_descriptions: Optional[List[str]] = None,
) -> Tuple[Image.Image, Dict[str, object]]:
    try:
        types = {str(t or "").strip().lower() for t in (target_types or []) if str(t or "").strip()}
        if types != {"outer"}:
            return output_image, {"applied": False, "reason": "not_outer_only"}

        desc_low = " ".join(str(v or "") for v in (product_descriptions or [])).lower()
        if any(token in desc_low for token in ("trench", "overcoat", "long coat", "full-length coat", "duster", "parka")):
            return output_image, {"applied": False, "reason": "long_outerwear"}
        if not any(token in desc_low for token in ("blazer", "jacket", "cardigan", "hoodie", "bomber", "shrug", "bolero", "outerwear")):
            return output_image, {"applied": False, "reason": "unsupported_outer_shape"}

        ref = reference_image.convert("RGB").resize(output_image.size, Image.BICUBIC)
        out = output_image.convert("RGB")
        ref_arr = np.asarray(ref, dtype=np.float32)
        out_arr = np.asarray(out, dtype=np.float32)
        if ref_arr.shape != out_arr.shape or ref_arr.ndim != 3:
            return output_image, {"applied": False, "reason": "shape_mismatch"}

        h, w = ref_arr.shape[:2]
        lower_start = int(round(h * 0.58))
        lower_full = int(round(h * 0.66))
        if lower_full <= lower_start:
            return output_image, {"applied": False, "reason": "invalid_blend_band"}

        lower_delta = np.mean(np.abs(out_arr[lower_start:, :, :] - ref_arr[lower_start:, :, :]), axis=2)
        changed_ratio = float(np.mean(lower_delta > 12.0))
        if changed_ratio < 0.015:
            return output_image, {"applied": False, "reason": "lower_body_unchanged", "changedRatio": round(changed_ratio, 4)}

        weights = np.ones((h,), dtype=np.float32)
        weights[lower_full:] = 0.0
        blend_band = max(1, lower_full - lower_start)
        for yy in range(lower_start, lower_full):
            frac = float(yy - lower_start) / float(blend_band)
            weights[yy] = max(0.0, min(1.0, 1.0 - frac))
        alpha = np.repeat(weights[:, None], w, axis=1)[:, :, None]
        blended = ((out_arr * alpha) + (ref_arr * (1.0 - alpha))).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended), {
            "applied": True,
            "reason": "outer_lower_body_restore",
            "lowerStartY": int(lower_start),
            "lowerFullY": int(lower_full),
            "changedRatio": round(changed_ratio, 4),
        }
    except Exception as restore_err:
        return output_image, {"applied": False, "reason": f"error:{restore_err}"}


_USER_PREP_IDENTITY_TERMS = (
    "identity", "face", "facial", "hair", "skin", "complexion", "body shape", "pose", "posture",
    "height", "age", "eyes", "nose", "mouth", "jaw", "framing", "lighting", "occlusion",
)
_USER_PREP_BACKGROUND_TERMS = (
    "background", "backdrop", "scene", "room", "wall", "floor", "studio", "furniture",
)
_USER_PREP_APPAREL_TERMS = (
    "wearing", "wears", "outfit", "clothing", "garment", "dress", "gown", "top", "shirt",
    "blouse", "jacket", "coat", "pants", "trousers", "jeans", "skirt", "shorts", "shoe",
    "footwear", "sleeve", "bodice",
)
_USER_PREP_DIRECTION_TERMS = (
    "left", "right", "viewer-left", "viewer-right", "toward left", "toward right",
    "facing left", "facing right", "to the left", "to the right",
)
_USER_PREP_BANNED_CONTEXT_TERMS = (
    "background", "backdrop", "scene", "room", "wall", "floor", "studio",
    "lighting", "camera", "framing", "aesthetic", "mood", "style",
    "outfit", "clothing", "garment", "dress", "top", "bottom", "outer", "sleeve",
)


def _format_user_prepare_reference(fields: Dict[str, str], *, include_outfit: bool) -> str:
    def _clean(value: object) -> str:
        return " ".join(str(value or "").split()).strip(" ,.;:/-")

    identity = _clean(fields.get("identity", ""))
    face = _clean(fields.get("face", ""))
    pose = _clean(fields.get("body_pose", "") or fields.get("by_pose", ""))
    lower_body_pose = _clean(fields.get("lower_body_pose", ""))
    outfit = ""
    if include_outfit:
        outfit = _clean(fields.get("current_outfit", "") or fields.get("outfit", "") or fields.get("clothing", ""))
    framing = _clean(fields.get("framing_lighting", ""))
    occlusion = _clean(fields.get("occlusion", ""))
    preserve = _clean(fields.get("preserve", ""))

    parts: List[str] = []
    if identity:
        parts.append(identity)
    if face:
        parts.append(face)
    if pose:
        parts.append(pose)
    if lower_body_pose:
        parts.append(lower_body_pose)
    if include_outfit and outfit:
        parts.append(outfit)
    if occlusion and occlusion.lower() not in {"none", "no", "n/a"}:
        parts.append(occlusion)
    if preserve:
        parts.append(preserve)
    return " ".join(parts).strip(" .>;,:")


def _finalize_user_prepare_brief(text: str, *, max_words: Optional[int] = None) -> str:
    cleaned = " ".join(str(text or "").split()).strip(" ,.;:>")
    if not cleaned:
        return "person with visible hair, balanced build, relaxed standing pose."

    cleaned = re.sub(
        r"\b(?:identity|face|pose|body pose|lower body pose|framing\/lighting|occlusion|preserve)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    for term in _USER_PREP_DIRECTION_TERMS:
        cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.IGNORECASE)
    for term in _USER_PREP_BANNED_CONTEXT_TERMS:
        cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:>")
    if not cleaned:
        return "person with visible hair, balanced build, relaxed standing pose."

    if max_words and max_words > 0:
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).strip(" ,.;:>")

    if not cleaned:
        cleaned = "person with visible hair, balanced build, relaxed standing pose"
    if not cleaned.endswith("."):
        cleaned = f"{cleaned}."
    return cleaned


def _normalize_user_prepare_api_prompt_description(raw_text: str) -> str:
    text = re.sub(r"[<>]+", " ", " ".join(str(raw_text or "").split())).strip()
    if not text:
        return ""

    fields = _parse_structured_descriptor(text)
    if fields:
        cleaned = _format_user_prepare_reference(fields, include_outfit=False)
        if cleaned:
            return _finalize_user_prepare_brief(cleaned)

    text = re.sub(
        r"\b(?:current\s*outfit|outfit|clothing|garments?)\b\s*(?:=|:)\s*[^;|.]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:background|backdrop|scene|room|wall|floor|studio|furniture)\b\s*(?:=|:)\s*[^;|.]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # For plain-language outputs (non key-value), reuse the identity-first cleaner
    # so API promptDescription stays concise and avoids outfit-heavy wording.
    text = _normalize_user_prepare_prompt_description(text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:>")
    return _finalize_user_prepare_brief(text)


def _normalize_user_prepare_prompt_description(raw_text: str) -> str:
    text = re.sub(r"[<>]+", " ", " ".join(str(raw_text or "").split())).strip()
    if not text:
        return ""

    fields = _parse_structured_descriptor(text)
    if fields:
        cleaned = _format_user_prepare_reference(fields, include_outfit=False)
        if cleaned:
            return _finalize_user_prepare_brief(cleaned)

    text = re.sub(
        r"\b(?:current\s*outfit|outfit|clothing|garments?)\b\s*(?:=|:)\s*[^;|.]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:background|scene)\b\s*(?:=|:)\s*[^;|.]+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    fragments = [frag.strip(" ,.") for frag in re.split(r"[.;|]\s*", text) if frag.strip()]
    kept: List[str] = []
    for frag in fragments:
        low = frag.lower()
        has_identity = any(term in low for term in _USER_PREP_IDENTITY_TERMS)
        has_background = any(term in low for term in _USER_PREP_BACKGROUND_TERMS)
        has_apparel = any(term in low for term in _USER_PREP_APPAREL_TERMS)
        if has_background:
            continue
        if has_identity and not has_apparel:
            kept.append(frag)

    cleaned = ". ".join(kept[:4]).strip(" .")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")
    if cleaned:
        return _finalize_user_prepare_brief(cleaned)
    return _finalize_user_prepare_brief(text)


def _user_prep_has_multiple_prominent_people(candidates: List[Dict]) -> bool:
    if len(candidates) < 2:
        return False
    top = candidates[0]
    second = candidates[1]
    top_score = float(top.get("person_score", 0.0))
    second_score = float(second.get("person_score", 0.0))
    top_area = float(top.get("area_ratio", 0.0))
    second_area = float(second.get("area_ratio", 0.0))
    return (
        second_score >= max(0.16, top_score * 0.82)
        and second_area >= max(0.06, top_area * 0.58)
    )


def _score_user_prep_face_candidate(candidate: Dict, image: Image.Image, person_bbox: Optional[List[int]] = None) -> float:
    bbox = [int(v) for v in (candidate.get("bbox") or [0, 0, 0, 0])]
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    area_ratio = float(candidate.get("area_ratio", 0.0))
    aspect_score = min(fw, fh) / float(max(fw, fh))

    if person_bbox:
        px0, py0, px1, py1 = [int(v) for v in person_bbox]
        ph = max(1, py1 - py0)
        rel_cy = (cy - py0) / float(ph)
    else:
        rel_cy = cy / float(max(1, image.height))

    top_prior = max(0.0, 1.0 - rel_cy)
    source_bonus = 0.12 if str(candidate.get("source") or "") == "parser_face" else 0.0
    return (0.55 * area_ratio) + (0.25 * top_prior) + (0.20 * aspect_score) + source_bonus


def _select_best_user_prep_face_candidate(
    candidates: List[Dict],
    image: Image.Image,
    person_bbox: Optional[List[int]] = None,
) -> Optional[Dict]:
    if not candidates:
        return None

    filtered: List[Dict] = []
    for candidate in candidates:
        bbox = [int(v) for v in (candidate.get("bbox") or [0, 0, 0, 0])]
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if person_bbox:
            px0, py0, px1, py1 = [int(v) for v in person_bbox]
            ph = max(1, py1 - py0)
            pw = max(1, px1 - px0)
            rel_cy = (cy - py0) / float(ph)
            rel_cx = (cx - px0) / float(pw)
            if rel_cy > 0.62:
                continue
            if rel_cx < -0.15 or rel_cx > 1.15:
                continue
        filtered.append(candidate)

    ranked = filtered or candidates
    ranked = sorted(
        ranked,
        key=lambda c: _score_user_prep_face_candidate(c, image=image, person_bbox=person_bbox),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _parser_alias_ids(aliases: List[str], fallback: Optional[List[int]] = None) -> List[int]:
    if engine.parser and hasattr(engine.parser, "category_ids"):
        for alias in aliases or []:
            try:
                ids = engine.parser.category_ids(alias)
                if ids:
                    return sorted({int(v) for v in ids})
            except Exception:
                continue
    return [int(v) for v in (fallback or [])]


def _score_visible_limb_preservation(
    reference_image: Image.Image,
    output_image: Image.Image,
) -> float:
    try:
        if engine.parser is None:
            return 0.0

        ref = reference_image.convert("RGB").resize((256, 384), Image.BICUBIC)
        out = output_image.convert("RGB").resize((256, 384), Image.BICUBIC)
        ref_parse = engine.parser.parse(ref)
        out_parse = engine.parser.parse(out)

        arm_ids = _parser_alias_ids(["arms", "arm", "left_arm", "right_arm"], [12, 14, 15])
        if not arm_ids:
            return 0.0

        ref_arm = np.isin(ref_parse, arm_ids)
        out_arm = np.isin(out_parse, arm_ids)

        h, w = ref_arm.shape[:2]
        roi = np.zeros((h, w), dtype=bool)
        roi[int(h * 0.18): int(h * 0.95), :] = True
        ref_arm &= roi
        out_arm &= roi

        total_pixels = max(1.0, float(h * w))
        min_pixels = max(24, int(total_pixels * 0.0012))
        ref_components = _mask_connected_components(ref_arm, min_pixels=min_pixels)
        out_components = _mask_connected_components(out_arm, min_pixels=min_pixels)

        ref_area = float(np.sum(ref_arm)) / total_pixels
        out_area = float(np.sum(out_arm)) / total_pixels
        if ref_area < 0.002 and out_area < 0.002:
            return 1.0

        count_penalty = abs(len(out_components) - len(ref_components)) / float(max(1, len(ref_components) + 1))
        area_penalty = abs(out_area - ref_area) / float(max(ref_area, 0.01))
        score = 1.0 - (0.55 * count_penalty + 0.45 * area_penalty)
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.0


def _to_public_item(item: Dict) -> Dict:
    public = {k: v for k, v in item.items() if not str(k).startswith("_")}

    compat_metadata: Dict[str, object] = {}
    extraction = public.get("extraction")
    if isinstance(extraction, dict) and extraction:
        compat_metadata["garmentExtractionMeta"] = extraction
    garment_metadata = public.get("garmentMetadata")
    if isinstance(garment_metadata, dict) and garment_metadata:
        compat_metadata["garmentMetadata"] = garment_metadata
    progress_sync = public.get("progress_sync")
    if isinstance(progress_sync, dict) and progress_sync:
        compat_metadata["progressSync"] = progress_sync
    if compat_metadata and not isinstance(public.get("metadata"), dict):
        public["metadata"] = compat_metadata

    output_url = str(public.get("output_image_url") or public.get("url") or "").strip()
    if output_url:
        public.setdefault("imageUrl", output_url)
        public.setdefault("outputImage", output_url)

    return public
