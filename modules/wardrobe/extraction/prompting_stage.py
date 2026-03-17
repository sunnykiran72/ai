from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Tuple


_COLOR_TERMS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "grey", "beige", "cream", "ivory", "maroon", "navy", "teal", "cyan", "magenta",
    "gold", "silver", "bronze", "tan", "khaki", "mustard", "lavender", "peach", "coral",
    "turquoise", "lime", "olive", "indigo", "violet", "burgundy", "charcoal",
}
_BODY_TERMS = {
    "person", "people", "model", "mannequin", "woman", "man", "girl", "boy",
    "body", "torso", "chest", "waist", "hip", "hips", "leg", "legs", "arm", "arms",
    "hand", "hands", "finger", "fingers", "face", "neck", "shoulder", "shoulders",
}


def _strip_terms(text: str, terms: set[str]) -> str:
    cleaned = str(text or "")
    for term in terms:
        cleaned = re.sub(rf"\\b{re.escape(term)}\\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip(" ,.;:/-")
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

    allowed = [
        "neckline",
        "sleeves",
        "bodice_cut",
        "silhouette",
        "length_hem",
        "fabric_texture",
        "embellishments",
        "special_details",
    ]

    sanitized: Dict[str, str] = {}
    for key in allowed:
        value = raw.get(key, "")
        if not value or str(value).strip().lower() in {"unknown", "n/a", "none"}:
            continue
        # Remove hex colors and color/body terms
        cleaned = re.sub(r"#(?:[0-9a-fA-F]{3}){1,2}\\b", " ", value)
        cleaned = _strip_terms(cleaned, _COLOR_TERMS)
        cleaned = _strip_terms(cleaned, _BODY_TERMS)
        cleaned = re.sub(r"\\bcolors?\\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s+", " ", cleaned).strip(" ,.;:/-")
        if cleaned:
            sanitized[key] = cleaned
    return sanitized


def _build_attribute_clause(minicpm_desc: str) -> str:
    attributes = _sanitize_minicpm_attributes(minicpm_desc)
    if not attributes:
        return ""
    label_map = {
        "neckline": "neckline",
        "sleeves": "sleeves",
        "bodice_cut": "bodice cut",
        "silhouette": "silhouette",
        "length_hem": "hem length",
        "fabric_texture": "fabric texture",
        "embellishments": "embellishments",
        "special_details": "special details",
    }
    parts = []
    for key in label_map:
        value = attributes.get(key)
        if value:
            parts.append(f"{label_map[key]} {value}")
    if not parts:
        return ""
    return "Garment attributes: " + "; ".join(parts) + "."


def _build_flux2_prompt(selected_type: str, minicpm_desc: str) -> str:
    garment_label = {
        "top": "top garment",
        "bottom": "bottom garment",
        "dress": "dress garment",
        "outer": "outerwear garment",
    }.get(selected_type, "garment")

    attribute_clause = _build_attribute_clause(minicpm_desc)
    segments = [
        f"A single {garment_label} displayed alone.",
    ]
    if attribute_clause:
        segments.append(attribute_clause)
    segments.extend([
        "Centered product presentation, front view or flat lay.",
        "Professional studio product photography on a seamless backdrop, clean and uncluttered scene.",
        "Soft diffused studio lighting with clear edge definition.",
        "Sharp focus, high detail.",
        "Preserve the exact silhouette, neckline, sleeve length, hem shape, fabric texture, and print placement from the reference image.",
    ])
    return " ".join(segments).strip()


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
    
    # Get MiniCPM description (raw)
    minicpm_desc = selected_item.get("minicpm_description", "")

    # Build FLUX prompt with sanitized MiniCPM attributes
    positive_prompt = _build_flux2_prompt(selected_type, str(minicpm_desc or ""))
    avoid_prompt = ""
    
    # Set the prompts on the item
    selected_item["baseGarmentPrompt"] = positive_prompt
    selected_item["promptDescription"] = positive_prompt
    selected_item["description"] = positive_prompt
    selected_item["extractionAvoidClause"] = avoid_prompt
    selected_item["type"] = selected_type
    
    # Get category information
    sync_category = wardrobe_category_from_garment_type(selected_type, style=None)
    selected_item["primary_category_key"] = sync_category["primary_category_key"]
    selected_item["category_key"] = sync_category["category_key"]
    selected_item["style"] = sync_category["style"]
    
    # Build simplified metadata (no color information)
    garment_metadata = {
        "prompt": {
            "base_garment_prompt": positive_prompt,
            "prompt_description": positive_prompt,
            "avoid_clause": avoid_prompt,
            "minicpm_description": minicpm_desc,  # MiniCPM description
        },
        "type": selected_type,
        "style": sync_category["style"],
        "primary_category_key": sync_category["primary_category_key"],
        "category_key": sync_category["category_key"],
        "prompt_source": "type_based_semantic_minicpm_only",
        # No color information
        "dominant_hexes": [],
        "accent_hexes": [],
        "color_hints": [],
        "color_profile": {},
    }
    
    selected_item["garmentMetadata"] = garment_metadata

    return selected_item, {
        "selected_type": selected_type,
        "prompt_description": positive_prompt,
        "avoid_prompt": avoid_prompt,
        "sync_category": sync_category,
        "garment_metadata": garment_metadata,
        "prompt_source": "type_based_semantic_minicpm_only",
        "minicpm_description": minicpm_desc,  # Pass through for extraction stage
    }
