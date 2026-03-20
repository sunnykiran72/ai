from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Tuple

from config.prompts import get_analyze_flux2_positive_prompt, get_analyze_top_subtype_clause
from utils.prompt_generation import infer_top_prompt_subtype


_COLOR_TERMS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "grey", "beige", "cream", "ivory", "maroon", "navy", "teal", "cyan", "magenta",
    "gold", "silver", "bronze", "tan", "khaki", "mustard", "lavender", "peach", "coral",
    "turquoise", "lime", "olive", "indigo", "violet", "burgundy", "charcoal",
    "multicolored", "multi-colored", "multicolor", "colorful", "colourful",
    "color", "colour", "monochrome", "grayscale", "greyscale",
}
_BODY_TERMS = {
    "person", "people", "model", "mannequin", "woman", "man", "girl", "boy",
    "body", "torso", "chest", "waist", "hip", "hips", "leg", "legs", "arm", "arms",
    "hand", "hands", "finger", "fingers", "face", "neck", "shoulder", "shoulders",
}
_SCENE_TERMS = {
    "background", "room", "camera", "phone", "selfie", "mirror", "prop", "props",
    "accessory", "accessories", "watermark", "text", "logo", "caption", "screen",
    "screenshot", "scene", "person", "people", "model", "mannequin", "body",
    "face", "skin", "hair", "hands", "hand", "legs", "leg", "arms", "arm", "wearing",
    "worn", "wears", "pose",
}


def _strip_terms(text: str, terms: set[str]) -> str:
    cleaned = str(text or "")
    for term in terms:
        cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:/-")
    return cleaned


def _parse_minicpm_kv(desc: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for chunk in str(desc or "").split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key:
            continue
        parsed[key] = value
    return parsed


def _sanitize_minicpm_attributes(desc: str) -> Dict[str, str]:
    raw = _parse_minicpm_kv(desc)
    # Normalize key aliases
    if "details" in raw and "special_details" not in raw:
        raw["special_details"] = raw.get("details", "")
    if "fabric" in raw and "fabric_texture" not in raw:
        raw["fabric_texture"] = raw.get("fabric", "")
    if "hem" in raw and "length_hem" not in raw:
        raw["length_hem"] = raw.get("hem", "")
    if "collar_style" in raw and "collar" not in raw:
        raw["collar"] = raw.get("collar_style", "")
    if "lapels" in raw and "lapel" not in raw:
        raw["lapel"] = raw.get("lapels", "")
    if "shoulder" in raw and "shoulder_style" not in raw:
        raw["shoulder_style"] = raw.get("shoulder", "")
    if "waist" in raw and "waistline" not in raw:
        raw["waistline"] = raw.get("waist", "")
    if "length" in raw and "length_hem" not in raw:
        raw["length_hem"] = raw.get("length", "")

    # Keep only garment attributes (no colors/body terms) and drop unknown values.
    allowed = [
        "neckline",
        "collar",
        "lapel",
        "shoulder_style",
        "sleeves",
        "cuffs",
        "bodice_cut",
        "waistline",
        "silhouette",
        "length_hem",
        "rise",
        "leg_shape",
        "skirt_style",
        "fabric_texture",
        "pattern",
        "embellishments",
        "closure",
        "pockets",
        "slits",
        "straps",
        "special_details",
    ]

    sanitized: Dict[str, str] = {}
    for key in allowed:
        value = raw.get(key, "")
        if not value or str(value).strip().lower() in {"unknown", "n/a", "none", "no", "yes"}:
            continue
        # Remove hex colors and color/body terms
        cleaned = re.sub(r"#(?:[0-9a-fA-F]{3}){1,2}\b", " ", value)
        cleaned = _strip_terms(cleaned, _COLOR_TERMS)
        cleaned = _strip_terms(cleaned, _BODY_TERMS)
        cleaned = re.sub(r"\bcolors?\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:/-")
        if cleaned:
            sanitized[key] = cleaned
    return sanitized


def _format_attribute(label: str, value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    if label == "shoulder style":
        return f"{value} style"
    if label == "sleeves":
        return f"{value} sleeves"
    if label == "cuffs":
        return f"{value} cuffs"
    if label == "straps":
        return f"{value} straps"
    if label == "lapel":
        return f"{value} lapel"
    if label == "collar":
        return f"{value} collar"
    return f"{label} {value}"


def _build_attribute_clause(minicpm_desc: str) -> str:
    attributes = _sanitize_minicpm_attributes(minicpm_desc)
    if not attributes:
        return ""
    label_map = {
        "neckline": "neckline",
        "collar": "collar",
        "lapel": "lapel",
        "shoulder_style": "shoulder style",
        "sleeves": "sleeves",
        "cuffs": "cuffs",
        "bodice_cut": "bodice cut",
        "waistline": "waistline",
        "silhouette": "silhouette",
        "length_hem": "hem length",
        "rise": "rise",
        "leg_shape": "leg shape",
        "skirt_style": "skirt style",
        "fabric_texture": "fabric texture",
        "pattern": "pattern",
        "embellishments": "embellishments",
        "closure": "closure",
        "pockets": "pockets",
        "slits": "slits",
        "straps": "straps",
        "special_details": "special details",
    }
    parts = []
    for key in label_map:
        value = attributes.get(key)
        if value:
            formatted = _format_attribute(label_map[key], value)
            if formatted:
                parts.append(formatted)
    if not parts:
        return ""
    return "Garment attributes: " + "; ".join(parts) + "."


def _build_direct_minicpm_prompt(
    minicpm_desc: str,
    *,
    strip_descriptor_color_clause: Callable[[str], str],
) -> str:
    text = " ".join(str(minicpm_desc or "").split()).strip()
    if not text:
        return ""

    text = strip_descriptor_color_clause(text)
    for term in _SCENE_TERMS:
        text = re.sub(rf"\b{re.escape(term)}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:/-")
    return text


def _is_direct_prompt_weak(text: str) -> bool:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return True
    if len(clean.split()) < 4:
        return True
    generic = clean.lower()
    if generic in {"garment", "top", "bottom", "dress", "outerwear", "clothing", "apparel", "item"}:
        return True
    return False


def _build_flux2_prompt(
    selected_type: str,
    minicpm_desc: str,
    strip_descriptor_color_clause: Callable[[str], str],
) -> tuple[str, str]:
    static_prompt = get_analyze_flux2_positive_prompt(selected_type)
    if selected_type == "top":
        subtype = infer_top_prompt_subtype(minicpm_desc)
        subtype_clause = get_analyze_top_subtype_clause(subtype)
        if subtype and subtype != "standard_top" and subtype_clause:
            static_prompt = f"{static_prompt} {subtype_clause}".strip()
    direct_prompt = _build_direct_minicpm_prompt(
        minicpm_desc,
        strip_descriptor_color_clause=strip_descriptor_color_clause,
    )
    prompt_source = "minicpm_direct_prompt"
    flux_prompt = f"{static_prompt} {direct_prompt}".strip() if direct_prompt else static_prompt
    return flux_prompt, direct_prompt, prompt_source


def apply_selected_item_prompting(
    *,
    selected_item: Optional[Dict[str, object]],
    requested_type: Optional[str],
    analyze_prompt_from_extracted: bool,
    normalize_garment_type: Callable[[Optional[str]], Optional[str]],
    infer_style_from_text: Callable[..., Optional[str]],
    wardrobe_category_from_garment_type: Callable[..., Dict[str, str]],
    product_prompt_description: Callable[..., str],
    build_garment_metadata: Callable[..., Dict[str, object]],
    strip_descriptor_color_clause: Callable[[str], str],
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    """
    Prompting stage that builds FLUX-friendly positive prompts only.
    - No negative prompts (FLUX does not support them).
    - No color terms (preserve reference colors implicitly).
    - MiniCPM contributes only structural garment attributes.
    """
    if not selected_item:
        return selected_item, {}

    # Determine the garment type
    forced_type = requested_type if requested_type in {"top", "bottom", "dress", "outer"} else None
    selected_type = forced_type or normalize_garment_type(str(selected_item.get("type") or "")) or "top"
    sync_category = wardrobe_category_from_garment_type(selected_type, style=None)
    
    # Get MiniCPM description (raw)
    minicpm_desc = (
        selected_item.get("minicpm_description")
        or selected_item.get("baseGarmentPrompt")
        or selected_item.get("promptDescription")
        or selected_item.get("description")
        or ""
    )

    # Build natural garment prompt from MiniCPM attributes
    positive_prompt, prompt_desc, prompt_source = _build_flux2_prompt(
        selected_type,
        str(minicpm_desc or ""),
        strip_descriptor_color_clause,
    )
    avoid_prompt = ""
    
    # Set the prompts on the item
    selected_item["baseGarmentPrompt"] = positive_prompt
    selected_item["promptDescription"] = prompt_desc
    selected_item["description"] = prompt_desc
    selected_item["extractionAvoidClause"] = avoid_prompt
    selected_item["promptDescriptionSource"] = prompt_source
    selected_item["type"] = selected_type
    
    selected_item["primary_category_key"] = sync_category["primary_category_key"]
    selected_item["category_key"] = sync_category["category_key"]
    selected_item["style"] = sync_category["style"]
    
    garment_metadata = build_garment_metadata(
        base_garment_prompt=positive_prompt,
        extraction_avoid_clause=avoid_prompt,
        prompt_sections_raw="",
        descriptor_raw_text=minicpm_desc,
        prompt_description=prompt_desc,
        prompt_source=prompt_source,
        target_type=selected_type,
        backend_target_type=selected_type,
        style=sync_category["style"],
        primary_category_key=sync_category["primary_category_key"],
        category_key=sync_category["category_key"],
        dominant_hexes=[],
        accent_hexes=[],
        color_hints=[],
        color_profile={},
        color_mask_source="",
        fashion_color_classifier={},
        color_sampling_mask_meta={},
    )
    
    selected_item["garmentMetadata"] = garment_metadata

    return selected_item, {
        "selected_type": selected_type,
        "prompt_description": prompt_desc,
        "avoid_prompt": avoid_prompt,
        "sync_category": sync_category,
        "garment_metadata": garment_metadata,
        "prompt_source": prompt_source,
        "minicpm_description": minicpm_desc,  # Pass through for extraction stage
    }
