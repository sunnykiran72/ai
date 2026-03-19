"""
Metadata extraction utilities for the fashion analysis system.

This module contains functions for building garment metadata, extracting descriptor facts,
building rich color metadata, and estimating color signal confidence.
"""

import re
from typing import Optional, List, Tuple, Dict, Any

from utils.color_processing import (
    canonical_color_token,
    color_family,
    bucket_color_brightness,
    bucket_color_saturation,
    bucket_color_undertone,
    nearest_color_label,
    hex_to_rgb_triplet,
    palette_supports_color_family,
    compose_color_descriptor_phrase,
)
from utils.validation import (
    normalize_garment_type,
    sanitize_prompt_fact_value,
    descriptor_word_count,
)
from utils.prompt_generation import (
    parse_structured_descriptor,
    extract_prompt_fact_segments,
    serialize_prompt_fact_segments,
)


def build_garment_metadata(
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
    color_profile: Optional[Dict[str, Any]] = None,
    color_mask_source: str = "",
    fashion_color_classifier: Optional[Dict[str, Any]] = None,
    color_sampling_mask_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build comprehensive garment metadata from various inputs.
    
    Args:
        base_garment_prompt: Base garment description prompt
        extraction_avoid_clause: Terms to avoid in extraction
        prompt_sections_raw: Raw prompt sections
        descriptor_raw_text: Raw descriptor text
        prompt_description: Processed prompt description
        prompt_source: Source of the prompt
        target_type: Target garment type
        backend_target_type: Backend-specific target type
        style: Garment style
        primary_category_key: Primary category key
        category_key: Category key
        dominant_hexes: List of dominant color hex codes
        accent_hexes: List of accent color hex codes
        color_hints: List of color hint strings
        color_profile: Color profile dictionary
        color_mask_source: Source of color mask
        fashion_color_classifier: Fashion color classifier results
        color_sampling_mask_meta: Color sampling mask metadata
        
    Returns:
        Comprehensive garment metadata dictionary
    """
    base_prompt = " ".join(str(base_garment_prompt or "").split()).strip()
    avoid_clause = " ".join(str(extraction_avoid_clause or "").split()).strip()
    normalized_prompt = " ".join(str(prompt_description or base_prompt).split()).strip()
    classification_target = str(target_type or "").strip()
    classification_backend = str(backend_target_type or classification_target).strip()
    
    # Resolve color truth (simplified version - full implementation would be complex)
    reconciled_color = _resolve_garment_color_truth_simple(
        base_garment_prompt=base_prompt,
        descriptor_raw_text=descriptor_raw_text,
        target_type=classification_backend or classification_target,
        dominant_hexes=dominant_hexes,
        color_hints=color_hints,
        color_profile=color_profile,
        color_mask_source=color_mask_source,
        color_sampling_mask_meta=color_sampling_mask_meta,
        fashion_color_classifier=fashion_color_classifier,
    )
    
    resolved_base_prompt = str(reconciled_color.get("base_garment_prompt") or base_prompt).strip()
    resolved_prompt_description = " ".join(str(prompt_description or resolved_base_prompt).split()).strip()
    
    if resolved_base_prompt and resolved_prompt_description:
        prompt_fields = extract_prompt_fact_segments(resolved_prompt_description)
        base_fields = extract_prompt_fact_segments(resolved_base_prompt)
        if base_fields.get("colors"):
            prompt_fields["colors"] = base_fields["colors"]
            rebuilt_prompt = serialize_prompt_fact_segments(prompt_fields)
            if rebuilt_prompt:
                resolved_prompt_description = f"{rebuilt_prompt}."

    rich_color = build_rich_color_metadata(
        dominant_hexes=reconciled_color.get("dominant_hexes") if isinstance(reconciled_color, dict) else dominant_hexes,
        color_hints=reconciled_color.get("color_hints") if isinstance(reconciled_color, dict) else color_hints,
        color_profile=color_profile,
        fashion_color_classifier=(
            reconciled_color.get("fashion_color_classifier")
            if isinstance(reconciled_color.get("fashion_color_classifier"), dict)
            else fashion_color_classifier
        ),
        color_sampling_mask_meta=color_sampling_mask_meta,
    )

    fact_fields = extract_garment_descriptor_facts(
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


def extract_garment_descriptor_facts(
    *,
    base_garment_prompt: str,
    descriptor_raw_text: str = "",
) -> Dict[str, str]:
    """
    Extract structured facts from garment descriptor text.
    
    Args:
        base_garment_prompt: Base garment prompt
        descriptor_raw_text: Raw descriptor text
        
    Returns:
        Dictionary of extracted facts
    """
    raw_fields = parse_structured_descriptor(descriptor_raw_text)
    prompt_fields = extract_prompt_fact_segments(base_garment_prompt)
    merged: Dict[str, str] = {}
    
    for key in (
        "category",
        "type",
        "colors",
        "pattern",
        "material",
        "silhouette",
        "construction",
        "details",
        "coverage",
        "preserve",
    ):
        value = str(raw_fields.get(key) or prompt_fields.get(key) or "").strip()
        if value:
            merged[key] = sanitize_prompt_fact_value(value)
    
    return merged


def build_rich_color_metadata(
    *,
    dominant_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, Any]] = None,
    fashion_color_classifier: Optional[Dict[str, Any]] = None,
    color_sampling_mask_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build rich color metadata from various color inputs.
    
    Args:
        dominant_hexes: List of dominant color hex codes
        color_hints: List of color hint strings
        color_profile: Color profile dictionary
        fashion_color_classifier: Fashion color classifier results
        color_sampling_mask_meta: Color sampling mask metadata
        
    Returns:
        Rich color metadata dictionary
    """
    resolved_hints = [
        str(v).strip().lower()
        for v in (color_hints or [])
        if str(v).strip()
    ]
    primary_label = resolved_hints[0] if resolved_hints else ""
    secondary_label = resolved_hints[1] if len(resolved_hints) > 1 else ""

    if not primary_label:
        signal = fashion_color_classifier if isinstance(fashion_color_classifier, dict) else {}
        primary_label = canonical_color_token(
            str(signal.get("top_label") or "")
        )
    
    if not primary_label:
        rgb_triplet = hex_to_rgb_triplet((dominant_hexes or [""])[0] if dominant_hexes else "")
        if rgb_triplet is not None:
            primary_label = canonical_color_token(nearest_color_label(rgb_triplet))
    
    if not secondary_label and len(resolved_hints) > 1:
        secondary_label = resolved_hints[1]

    brightness = bucket_color_brightness(color_profile)
    saturation = bucket_color_saturation(color_profile)
    undertone = bucket_color_undertone(color_profile)
    primary_family = color_family(primary_label) if primary_label else ""
    secondary_family = color_family(secondary_label) if secondary_label else ""
    
    primary_descriptor = compose_color_descriptor_phrase(
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

    signal_confidence = estimate_color_signal_confidence(
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


def estimate_color_signal_confidence(
    *,
    color_mask_source: str = "",
    color_profile: Optional[Dict[str, Any]] = None,
    color_sampling_mask_meta: Optional[Dict[str, Any]] = None,
    resolved_source: str = "",
) -> float:
    """
    Estimate confidence score for color signal based on various factors.
    
    Args:
        color_mask_source: Source of the color mask
        color_profile: Color profile dictionary
        color_sampling_mask_meta: Color sampling mask metadata
        resolved_source: Resolved color source
        
    Returns:
        Confidence score between 0.05 and 0.99
    """
    mask_source = str(color_mask_source or "").strip().lower()
    resolved = str(resolved_source or "").strip().lower()
    profile = color_profile if isinstance(color_profile, dict) else {}
    sampling_meta = color_sampling_mask_meta if isinstance(color_sampling_mask_meta, dict) else {}

    score = 0.42
    
    # Adjust score based on mask source quality
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

    # Adjust score based on mask size and coverage
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

    # Adjust score based on resolved source
    if resolved == "pixel":
        score += 0.06
    elif resolved.startswith("fashion_basecolour"):
        score -= 0.02
    elif resolved.startswith("semantic_"):
        score -= 0.08

    # Adjust score based on color profile quality
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


def extract_garment_metadata_prompt(garment_metadata: Any) -> str:
    """
    Extract prompt from garment metadata.
    
    Args:
        garment_metadata: Garment metadata dictionary
        
    Returns:
        Extracted prompt string
    """
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


def extract_garment_metadata_target_type(garment_metadata: Any) -> Optional[str]:
    """
    Extract target type from garment metadata.
    
    Args:
        garment_metadata: Garment metadata dictionary
        
    Returns:
        Normalized target type or None
    """
    if not isinstance(garment_metadata, dict):
        return None
    
    classification = garment_metadata.get("classification")
    if not isinstance(classification, dict):
        return None
    
    for key in ("backend_target_type", "target_type"):
        value = normalize_garment_type(str(classification.get(key) or ""))
        if value:
            return value
    
    return None


def extract_garment_metadata_color_payload(garment_metadata: Any) -> Tuple[List[str], List[str]]:
    """
    Extract color payload from garment metadata.
    
    Args:
        garment_metadata: Garment metadata dictionary
        
    Returns:
        Tuple of (dominant_hexes, color_hints)
    """
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


def extract_garment_metadata_color_block(garment_metadata: Any) -> Dict[str, Any]:
    """
    Extract complete color block from garment metadata.
    
    Args:
        garment_metadata: Garment metadata dictionary
        
    Returns:
        Color block dictionary
    """
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
        signal_confidence = estimate_color_signal_confidence(
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


def strip_descriptor_color_clause(description: str) -> str:
    """
    Remove model-predicted color phrase from schema-like descriptors.
    We keep pixel-derived color lock as the source of truth.
    
    Args:
        description: Input description string
        
    Returns:
        Description with color clause stripped
    """
    text = " ".join(str(description or "").split()).strip()
    if not text:
        return text

    def _strip_freeform_color_terms(value: str) -> str:
        cleaned = str(value or "")
        # Remove standalone color words and the most common color modifiers that
        # tend to travel with them in freeform VLM prose.
        color_modifiers = (
            "light", "dark", "deep", "soft", "muted", "pale", "bright",
            "vivid", "rich", "washed", "neutral", "warm", "cool", "solid",
            "plain",
        )
        for term in color_modifiers:
            cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.IGNORECASE)
        color_terms = (
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
            "black", "white", "gray", "grey", "beige", "cream", "ivory", "maroon",
            "navy", "teal", "cyan", "magenta", "gold", "silver", "bronze", "tan",
            "khaki", "mustard", "lavender", "peach", "coral", "turquoise", "lime",
            "olive", "indigo", "violet", "burgundy", "charcoal", "multicolored",
            "multi-colored", "multicolor", "colorful", "colourful", "monochrome",
            "grayscale", "greyscale",
        )
        for term in color_terms:
            cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bcolors?\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:/-")
        return cleaned

    segments = [seg.strip() for seg in text.split(",") if seg.strip()]
    if not segments:
        return _strip_freeform_color_terms(text)

    key_prefixes = (
        "type ",
        "colors ",
        "pattern ",
        "material ",
        "silhouette ",
        "construction ",
        "details ",
        "coverage ",
        "preserve ",
    )

    out: List[str] = []
    dropping_color_tail = False
    
    for seg in segments:
        low = seg.lower()
        if low.startswith("colors "):
            dropping_color_tail = True
            continue
        if dropping_color_tail:
            is_new_key = any(low.startswith(k) for k in key_prefixes if k != "colors ")
            if not is_new_key:
                continue
            dropping_color_tail = False
        out.append(seg)

    normalized = ", ".join(out).strip(" ,")
    normalized = normalized or text
    normalized = _strip_freeform_color_terms(normalized)
    return normalized or text


# Helper functions

def _bucket_color_signal_confidence(score: float) -> str:
    """
    Convert confidence score to categorical strength.
    
    Args:
        score: Confidence score
        
    Returns:
        Strength category string
    """
    if score >= 0.75:
        return "high"
    elif score >= 0.55:
        return "medium"
    elif score >= 0.35:
        return "low"
    else:
        return "very_low"


def _sanitize_prompt_fact_fields(
    fields: Dict[str, str],
    *,
    target_type: str = "",
) -> Dict[str, str]:
    """
    Sanitize and clean prompt fact fields.
    
    Args:
        fields: Dictionary of fact fields
        target_type: Target garment type
        
    Returns:
        Sanitized fields dictionary
    """
    cleaned_fields = {
        str(key): sanitize_prompt_fact_value(str(value or ""))
        for key, value in dict(fields or {}).items()
        if str(key).strip()
    }
    
    # Additional cleaning logic would go here
    # This is a simplified version
    
    return {k: v for k, v in cleaned_fields.items() if v}


def _resolve_garment_color_truth_simple(
    *,
    base_garment_prompt: str,
    descriptor_raw_text: str = "",
    target_type: str = "",
    dominant_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, Any]] = None,
    color_mask_source: str = "",
    color_sampling_mask_meta: Optional[Dict[str, Any]] = None,
    fashion_color_classifier: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Simplified version of color truth resolution.
    
    This is a simplified implementation. The full version in main.py is very complex
    and handles many edge cases for color resolution.
    
    Args:
        base_garment_prompt: Base garment prompt
        descriptor_raw_text: Raw descriptor text
        target_type: Target garment type
        dominant_hexes: List of dominant color hex codes
        color_hints: List of color hint strings
        color_profile: Color profile dictionary
        color_mask_source: Source of color mask
        color_sampling_mask_meta: Color sampling mask metadata
        fashion_color_classifier: Fashion color classifier results
        
    Returns:
        Resolved color truth dictionary
    """
    # This is a simplified implementation
    # The full implementation in main.py is extremely complex
    
    signal_confidence = estimate_color_signal_confidence(
        color_mask_source=color_mask_source,
        color_profile=color_profile,
        color_sampling_mask_meta=color_sampling_mask_meta,
        resolved_source="pixel",
    )
    
    return {
        "base_garment_prompt": base_garment_prompt,
        "dominant_hexes": dominant_hexes or [],
        "color_hints": color_hints or [],
        "color_source": "pixel",
        "color_signal_confidence": signal_confidence,
        "color_signal_strength": _bucket_color_signal_confidence(signal_confidence),
        "fashion_color_classifier": fashion_color_classifier or {},
    }
