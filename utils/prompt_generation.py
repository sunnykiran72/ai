"""
Prompt generation utilities for the fashion analysis system.

This module contains functions for parsing structured descriptors, generating prompts,
building avoid clauses, and processing prompt sections.
"""

import json
import re
from typing import Optional, List, Dict, Any

from utils.validation import normalize_garment_type, sanitize_prompt_fact_value


def parse_structured_descriptor(text: str) -> Dict[str, str]:
    """
    Parse lightweight key-value descriptors from VLM output lines.
    Supports: key=value, key: value, key[value], key=<value>
    
    Args:
        text: Input text to parse
        
    Returns:
        Dictionary of parsed key-value pairs
        
    Examples:
        >>> parse_structured_descriptor("type=dress, colors=blue")
        {'type': 'dress', 'colors': 'blue'}
        >>> parse_structured_descriptor("Category: evening gown Type: formal")
        {'category': 'evening gown', 'type': 'formal'}
    """
    src = str(text or "").strip()
    if not src:
        return {}

    # Remove lightweight markdown formatting often returned by VLMs.
    src = src.replace("**", "").replace("`", "")
    src = re.sub(r"\s+", " ", src).strip()

    # Inject separators before known keys when model returns a single stream like:
    # "Category: dress Type: evening gown Colors: beige ..."
    known_labels = [
        "category", "type", "colors", "pattern", "material", "silhouette",
        "construction", "details", "coverage", "preserve",
        "identity", "body pose", "body_pose", "by pose", "by_pose",
        "framing lighting", "framing_lighting", "current outfit", "current_outfit",
        "occlusion",
    ]
    for label in sorted(known_labels, key=len, reverse=True):
        pattern = rf"(?i)\b{re.escape(label)}\b\s*:"
        src = re.sub(pattern, f"; {label}:", src)
    src = src.lstrip("; ").strip()

    parsed: Dict[str, str] = {}
    segments = [seg.strip() for seg in re.split(r"[;|]\s*", src) if seg.strip()]
    for seg in segments:
        m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_\- ]{0,40})\s*(?:=|:)\s*(.+?)\s*$", seg)
        if not m:
            m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_\- ]{0,40})\s*\[(.+?)\]\s*$", seg)
        if not m:
            continue
        raw_key = m.group(1).strip().lower()
        raw_key = raw_key.replace("/", "_").replace("-", "_").replace(" ", "_")
        key_aliases = {
            "bodypose": "body_pose",
            "body_pose": "body_pose",
            "by_pose": "body_pose",
            "bypose": "body_pose",
            "framinglighting": "framing_lighting",
            "framing_lighting": "framing_lighting",
            "currentoutfit": "current_outfit",
            "current_outfit": "current_outfit",
        }
        raw_key = key_aliases.get(raw_key, raw_key)
        raw_val = m.group(2).strip()
        raw_val = raw_val.strip("<>[](){} \t\r\n")
        # Drop accidental markdown/list punctuation wrapping.
        raw_val = raw_val.strip("*- ")
        if raw_key and raw_val:
            parsed[raw_key] = raw_val
    return parsed


def parse_garment_prompt_sections(
    raw_text: str,
    *,
    garment_type: Optional[str] = None,
) -> Dict[str, str]:
    """
    Parse garment prompt sections from raw text.
    
    Args:
        raw_text: Raw text input
        garment_type: Optional garment type for validation
        
    Returns:
        Dictionary with parsed sections including base_garment_prompt and extraction_avoid_clause
    """
    text = str(raw_text or "").strip()
    normalized_text = " ".join(text.split()).strip()
    base_prompt = ""
    avoid_clause = ""
    json_contract_valid = False
    source_format = "freeform"

    parsed_json = extract_json_object_from_text(text)
    if isinstance(parsed_json, dict):
        base_key_present = any(
            key in parsed_json
            for key in ("base_garment_prompt", "BASE_GARMENT_PROMPT")
        )
        avoid_key_present = any(
            key in parsed_json
            for key in ("extraction_avoid_clause", "EXTRACTION_AVOID_CLAUSE")
        )
        base_prompt = clean_prompt_section_text(
            str(
                parsed_json.get("base_garment_prompt")
                or parsed_json.get("BASE_GARMENT_PROMPT")
                or ""
            )
        )
        avoid_clause = clean_prompt_section_text(
            str(
                parsed_json.get("extraction_avoid_clause")
                or parsed_json.get("EXTRACTION_AVOID_CLAUSE")
                or ""
            )
        )
        json_contract_valid = bool(base_key_present)
        source_format = "json"

    if not base_prompt:
        base_match = re.search(
            r"BASE_GARMENT_PROMPT\s*:\s*(.*?)(?=\bEXTRACTION_AVOID_CLAUSE\s*:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if base_match:
            base_prompt = clean_prompt_section_text(base_match.group(1))
            if source_format == "freeform":
                source_format = "tagged_text"
    if not avoid_clause:
        avoid_match = re.search(
            r"EXTRACTION_AVOID_CLAUSE\s*:\s*(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if avoid_match:
            avoid_clause = clean_prompt_section_text(avoid_match.group(1))
            if source_format == "freeform":
                source_format = "tagged_text"

    if not base_prompt:
        base_prompt = normalize_minicpm_descriptor_text(text, kind="garment")
    base_prompt = sanitize_florence_garment_description(base_prompt or "")
    base_prompt = " ".join(str(base_prompt or "").split()).strip(" ,.")
    if base_prompt and not base_prompt.endswith("."):
        base_prompt = f"{base_prompt}."
    base_prompt = ensure_target_type_in_description(base_prompt, str(garment_type or ""))

    avoid_clause = " ".join(str(avoid_clause or "").split()).strip(" ,.")
    if avoid_clause and not avoid_clause.endswith("."):
        avoid_clause = f"{avoid_clause}."

    serialized_sections = f"BASE_GARMENT_PROMPT: {base_prompt or 'Garment.'}"
    serialized_sections += f"\nEXTRACTION_AVOID_CLAUSE: {avoid_clause}"

    return {
        "raw_text": normalized_text,
        "base_garment_prompt": base_prompt or "Garment.",
        "extraction_avoid_clause": avoid_clause,
        "serialized_sections": serialized_sections,
        "json_contract_valid": "true" if json_contract_valid else "false",
        "source_format": source_format,
    }


def normalize_minicpm_descriptor_text(raw_text: str, kind: str) -> str:
    """
    Convert structured MiniCPM descriptor into a concise, Flux-friendly sentence.
    Keeps key fidelity terms (color/pattern/structure/details/identity).
    
    Args:
        raw_text: Raw descriptor text
        kind: Type of descriptor ("person" or "garment")
        
    Returns:
        Normalized descriptor text
    """
    text = " ".join(str(raw_text or "").split()).strip()
    if not text:
        return text

    fields = parse_structured_descriptor(text)
    if not fields:
        return text

    if kind == "person":
        identity = fields.get("identity", "")
        pose = fields.get("body_pose", "") or fields.get("by_pose", "")
        framing = fields.get("framing_lighting", "")
        occlusion = fields.get("occlusion", "")
        preserve = fields.get("preserve", "")
        parts = []
        if identity:
            parts.append(f"identity: {identity}")
        if pose:
            parts.append(f"pose: {pose}")
        if framing:
            parts.append(f"framing/lighting: {framing}")
        if occlusion and occlusion.lower() not in {"none", "no", "n/a"}:
            parts.append(f"occlusion: {occlusion}")
        if preserve:
            parts.append(f"preserve: {preserve}")
        return ". ".join(parts).strip(" .") or text

    category = fields.get("category", "")
    gtype = fields.get("type", "")
    colors = fields.get("colors", "")
    pattern = fields.get("pattern", "")
    material = fields.get("material", "")
    silhouette = fields.get("silhouette", "")
    construction = fields.get("construction", "")
    details = fields.get("details", "")
    coverage = fields.get("coverage", "")
    preserve = fields.get("preserve", "")
    parts = []
    if category:
        parts.append(f"{category} garment")
    if gtype:
        parts.append(f"type {gtype}")
    if colors:
        parts.append(f"colors {colors}")
    if pattern:
        parts.append(f"pattern {pattern}")
    if material:
        parts.append(f"material {material}")
    if silhouette:
        parts.append(f"silhouette {silhouette}")
    if construction:
        parts.append(f"construction {construction}")
    if details and details.lower() not in {"none", "no visible details", "n/a"}:
        parts.append(f"details {details}")
    if coverage:
        parts.append(f"coverage {coverage}")
    if preserve:
        parts.append(f"preserve {preserve}")
    return ", ".join(parts).strip(" ,.") or text


def sanitize_florence_garment_description(description: str) -> str:
    """
    Keep garment attributes and strip scene/mannequin/background chatter.
    
    Args:
        description: Input description text
        
    Returns:
        Sanitized description
    """
    text = str(description or "").strip()
    if not text:
        return ""

    text = re.sub(r"^\s*the image shows\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:a|the)?\s*mannequin\s+(?:is\s+)?wearing\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    fragments = [frag.strip(" .") for frag in re.split(r"[.]+", text) if frag.strip()]
    if not fragments:
        return text

    drop_markers = (
        "background",
        "white wall",
        "set against",
        "mannequin",
        "model is standing",
        "the image",
    )
    keep_markers = (
        "dress",
        "gown",
        "top",
        "shirt",
        "blouse",
        "corset",
        "skirt",
        "pants",
        "trousers",
        "jeans",
        "jacket",
        "coat",
        "fabric",
        "ruffle",
        "sleeve",
        "bodice",
        "silhouette",
        "color",
        "black",
        "white",
        "red",
        "blue",
        "green",
    )

    filtered: List[str] = []
    for frag in fragments:
        low = frag.lower()
        if any(marker in low for marker in drop_markers):
            continue
        filtered.append(frag)

    cleaned = ". ".join(filtered[:3]).strip(" .")
    cleaned = cleaned or text
    # Remove hedging words that weaken transfer constraints.
    cleaned = re.sub(r"\b(?:likely|possibly|probably|maybe)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:appears to be|seems to be|looks like)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthe garment is\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")
    return cleaned or text


def build_tryon_prompt(
    garment_description: str,
    user_description: str,
    target_type: str = "",
    style: str = "",
) -> str:
    """
    Build a complete try-on prompt from garment and user descriptions.
    
    Args:
        garment_description: Description of the garment
        user_description: Description of the user/person
        target_type: Target garment type
        style: Style specification
        
    Returns:
        Complete try-on prompt
    """
    garment_part = str(garment_description or "").strip()
    user_part = str(user_description or "").strip()
    
    if target_type:
        garment_part = ensure_target_type_in_description(garment_part, target_type)
    
    if style:
        style_part = f"Style: {style}. "
    else:
        style_part = ""
    
    if user_part:
        user_part = augment_identity_lock(user_part)
    
    parts = [p for p in [style_part, garment_part, user_part] if p]
    return " ".join(parts).strip()


_COLOR_TERMS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "grey", "beige", "cream", "ivory", "maroon", "navy", "teal", "cyan", "magenta",
    "gold", "silver", "bronze", "tan", "khaki", "mustard", "lavender", "peach", "coral",
    "turquoise", "lime", "olive", "indigo", "violet", "burgundy", "charcoal",
    "multicolored", "multi-colored", "multicolor", "colorful", "colourful",
    "color", "colour", "monochrome", "grayscale", "greyscale",
}


def _clean_descriptor_value(value: str) -> str:
    cleaned = " ".join(str(value or "").split()).strip()
    if not cleaned:
        return ""
    low = cleaned.lower()
    if low in {"unknown", "n/a", "none", "no", "yes"}:
        return ""
    # Remove hex colors
    cleaned = re.sub(r"#(?:[0-9a-fA-F]{3}){1,2}\\b", " ", cleaned)
    # Remove color terms
    for term in _COLOR_TERMS:
        cleaned = re.sub(rf"\\b{re.escape(term)}\\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip(" ,.;:/-")
    return cleaned


def _join_phrases(phrases: List[str]) -> str:
    parts = [p for p in phrases if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _maybe_suffix(value: str, suffix: str, keyword: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if keyword.lower() in raw.lower():
        return raw
    return f"{raw} {suffix}".strip()


def _extract_structured_fields(desc: str) -> Dict[str, str]:
    raw = parse_structured_descriptor(desc)
    aliases = {
        "details": "special_details",
        "fabric": "fabric_texture",
        "hem": "length_hem",
        "length": "length_hem",
        "collar_style": "collar",
        "lapels": "lapel",
        "shoulder": "shoulder_style",
        "waist": "waistline",
        "type": "type",
        "category": "category",
        "material": "material",
        "texture": "texture",
        "fit": "fit",
        "length": "length",
        "pattern": "pattern",
        "embellishments": "embellishments",
        "special_details": "special_details",
        "sleeves": "sleeves",
        "neckline": "neckline",
        "bodice_cut": "bodice_cut",
        "silhouette": "silhouette",
        "fabric_texture": "fabric_texture",
        "length_hem": "length_hem",
        "closure": "closure",
        "pockets": "pockets",
        "slits": "slits",
        "straps": "straps",
        "layering": "layering",
        "opening": "opening",
        "lining": "lining",
        "sheer": "sheer",
        "hardware": "hardware",
        "rise": "rise",
        "leg_shape": "leg_shape",
        "skirt_style": "skirt_style",
        "construction": "construction",
    }
    normalized: Dict[str, str] = {}
    for key, value in raw.items():
        alias = aliases.get(key, key)
        if alias not in aliases.values():
            continue
        cleaned = _clean_descriptor_value(value)
        if cleaned:
            normalized[alias] = cleaned
    return normalized


def build_garment_prompt_natural(
    desc: str,
    garment_type_hint: Optional[str] = None,
) -> str:
    fields = _extract_structured_fields(desc)
    garment_type = (fields.get("type") or "").strip()
    if not garment_type:
        hint = (garment_type_hint or "").strip().lower()
        garment_type = {
            "top": "top garment",
            "bottom": "bottom garment",
            "dress": "dress",
            "outer": "outerwear",
        }.get(hint, "garment")

    construction_bits: List[str] = []
    neckline = fields.get("neckline")
    if neckline:
        construction_bits.append(_maybe_suffix(neckline, "neckline", "neckline"))
    collar = fields.get("collar")
    if collar:
        construction_bits.append(_maybe_suffix(collar, "collar", "collar"))
    lapel = fields.get("lapel")
    if lapel:
        construction_bits.append(_maybe_suffix(lapel, "lapel", "lapel"))
    shoulder = fields.get("shoulder_style")
    if shoulder:
        construction_bits.append(_maybe_suffix(shoulder, "shoulder", "shoulder"))
    sleeves = fields.get("sleeves")
    if sleeves:
        construction_bits.append(_maybe_suffix(sleeves, "sleeves", "sleeve"))
    bodice = fields.get("bodice_cut")
    if bodice:
        construction_bits.append(_maybe_suffix(bodice, "bodice", "bodice"))
    waistline = fields.get("waistline")
    if waistline:
        construction_bits.append(_maybe_suffix(waistline, "waist", "waist"))
    silhouette = fields.get("silhouette")
    if silhouette:
        construction_bits.append(_maybe_suffix(silhouette, "silhouette", "silhouette"))
    length_hem = fields.get("length_hem")
    if length_hem:
        lowered = length_hem.lower()
        if "hem" in lowered:
            hem_phrase = length_hem
        elif lowered in {"crop", "cropped"}:
            hem_phrase = "cropped hem"
        elif lowered in {"mini", "miniskirt"}:
            hem_phrase = "mini hem"
        elif lowered in {"midi"}:
            hem_phrase = "midi hem"
        elif lowered in {"maxi"}:
            hem_phrase = "maxi hem"
        else:
            hem_phrase = f"{length_hem} hem"
        construction_bits.append(hem_phrase)
    length_value = fields.get("length")
    if length_value:
        construction_bits.append(_maybe_suffix(length_value, "length", "length"))
    rise = fields.get("rise")
    if rise:
        construction_bits.append(f"{rise} rise")
    leg_shape = fields.get("leg_shape")
    if leg_shape:
        construction_bits.append(f"{leg_shape} leg shape")
    skirt_style = fields.get("skirt_style")
    if skirt_style:
        construction_bits.append(f"{skirt_style} skirt")

    sentence_one = f"A {garment_type}".strip()
    construction_clause = _join_phrases(construction_bits)
    if construction_clause:
        sentence_one += f" with {construction_clause}."
    else:
        sentence_one += "."

    fabric_texture = fields.get("fabric_texture") or fields.get("texture")
    material = fields.get("material")
    sentence_two = ""
    fit = fields.get("fit")
    if material and fabric_texture:
        sentence_two = f"Made of {material} with a {fabric_texture} texture."
    elif material:
        sentence_two = f"Made of {material}."
    elif fabric_texture:
        sentence_two = f"{fabric_texture.capitalize()} texture."
    if fit:
        sentence_two = (sentence_two + " " if sentence_two else "") + f"{fit.capitalize()} fit."

    pattern = fields.get("pattern")
    if pattern and pattern.lower() != "solid":
        sentence_two = (sentence_two + " " if sentence_two else "") + f"Patterned with {pattern}."

    details_bits: List[str] = []
    for key in ("embellishments", "special_details", "closure", "pockets", "slits", "straps", "layering", "hardware"):
        value = fields.get(key)
        if value:
            details_bits.append(value)
    opening = fields.get("opening")
    if opening:
        details_bits.append(f"{opening} closure")
    lining = fields.get("lining")
    if lining:
        details_bits.append(f"{lining} lining")
    sheer = fields.get("sheer")
    if sheer:
        details_bits.append(f"{sheer} fabric")
    if details_bits:
        details_clause = _join_phrases(details_bits)
        sentence_two = (sentence_two + " " if sentence_two else "") + f"Details include {details_clause}."

    construction = fields.get("construction")
    if construction:
        sentence_two = (sentence_two + " " if sentence_two else "") + f"Construction details: {construction}."

    output = " ".join([sentence_one, sentence_two]).strip()
    return output or "A garment."


def _strip_unknown_tokens(text: str) -> str:
    raw = " ".join(str(text or "").split()).strip()
    if not raw:
        return ""
    fragments = [frag.strip() for frag in raw.replace("\n", " ").split(";") if frag.strip()]
    cleaned: List[str] = []
    for frag in fragments:
        if "unknown" in frag.lower():
            continue
        cleaned.append(frag)
    if not cleaned:
        return raw
    return ", ".join(cleaned).strip(" ,")


def build_tryon_prompt_v2(
    user_description: str,
    garment_descriptions: List[str],
    target_types: List[str],
    board_mode: str,
) -> str:
    """
    Build a Flux2 try-on prompt following FLUX best practices:
    - Natural language
    - Positive preservation constraints
    - Explicit scope for edits
    """
    safe_user = _strip_unknown_tokens(user_description)
    garment_lines: List[str] = []
    for desc in garment_descriptions or []:
        cleaned = _strip_unknown_tokens(desc)
        if cleaned:
            garment_lines.append(cleaned if cleaned.endswith(".") else f"{cleaned}.")

    intro = (
        "A photorealistic virtual try-on image edit of the same person from the reference."
    )
    preserve = (
        "Preserve the person's identity, face, skin tone, hair, body proportions, pose, hands, "
        "and the original background and lighting."
    )
    if board_mode == "collage":
        scope = "Apply all garments from the collage reference as a complete outfit."
    else:
        scope = "Replace only the target garment with the reference garment."

    parts = [intro, preserve, scope]
    if safe_user:
        parts.append(f"User details: {safe_user}.")
    if garment_lines:
        if len(garment_lines) == 1:
            parts.append(f"Garment details: {garment_lines[0]}")
        else:
            parts.append("Outfit pieces include: " + " ".join(garment_lines))
    parts.append("Sharp focus, high detail, clean composition.")
    return " ".join(part.strip() for part in parts if part).strip()


def build_flux2_prompt(
    base_description: str,
    negative_prompt: str = "",
    target_type: str = "",
) -> Dict[str, str]:
    """
    Build Flux2 prompt with positive and negative components.
    
    Args:
        base_description: Base description text
        negative_prompt: Negative prompt text
        target_type: Target garment type
        
    Returns:
        Dictionary with 'positive' and 'negative' prompt components
    """
    positive = str(base_description or "").strip()
    negative = str(negative_prompt or "").strip()
    
    if target_type:
        positive = ensure_target_type_in_description(positive, target_type)
    
    return {
        "positive": positive,
        "negative": negative,
    }


def extract_generation_only_avoid_directives(text: str) -> List[str]:
    """
    Extract avoid directives that start with "do not", "don't", or "never".
    
    Args:
        text: Input text to parse
        
    Returns:
        List of avoid directive strings
    """
    raw = " ".join(str(text or "").split()).strip()
    if not raw:
        return []
    
    directives: List[str] = []
    for segment in re.split(r"(?<=[.!?])\s+", raw):
        clean = segment.strip(" ,.")
        low = clean.lower()
        if not clean:
            continue
        if low.startswith("do not ") or low.startswith("don't ") or low.startswith("never "):
            directives.append(clean if clean.endswith(".") else f"{clean}.")
    return directives


def build_florence_contamination_avoid_clause(
    caption: str,
    *,
    garment_type: Optional[str] = None,
) -> str:
    """
    Build avoid clause to prevent Florence contamination based on caption content.
    
    Args:
        caption: Input caption text
        garment_type: Optional garment type for context
        
    Returns:
        Avoid clause string
    """
    text = " ".join(str(caption or "").split()).strip().lower()
    if not text:
        return ""

    terms: List[str] = []

    def add(term: str) -> None:
        if term and term not in terms:
            terms.append(term)

    person_markers = (
        "woman", "man", "girl", "boy", "person", "model", "wearing", "selfie",
        "standing", "holding", "posing",
    )
    if any(marker in text for marker in person_markers):
        for term in ("skin", "hair", "face", "hands"):
            add(term)

    for needle, label in (
        ("phone", "phone"),
        ("mirror", "mirror"),
        ("bed", "bed"),
        ("pillow", "bed"),
        ("bag", "bag"),
        ("purse", "bag"),
        ("handbag", "bag"),
        ("cup", "drink"),
        ("drink", "drink"),
        ("coffee", "drink"),
        ("sunglasses", "sunglasses"),
        ("glasses", "glasses"),
        ("earring", "earrings"),
        ("necklace", "necklace"),
    ):
        if needle in text:
            add(label)

    gtype = normalize_garment_type(garment_type) or "garment"
    if gtype == "top":
        for marker in ("pants", "trousers", "jeans", "leggings", "skirt", "shorts", "dress", "gown"):
            if marker in text:
                add("lower-body garments")
                break
    elif gtype == "bottom":
        for marker in ("shirt", "top", "blouse", "jacket", "coat", "sweater", "crop top", "bralette"):
            if marker in text:
                add("upper-body garments")
                break

    if terms:
        add("background")

    joined = join_avoid_terms(terms)
    return f"Ignore {joined}." if joined else ""


def extract_prompt_fact_segments(text: str) -> Dict[str, str]:
    """
    Extract structured fact segments from prompt text.
    
    Args:
        text: Input text to parse
        
    Returns:
        Dictionary of extracted fact segments
    """
    src = " ".join(str(text or "").split()).strip().strip(" ,.")
    if not src:
        return {}

    labels = [
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
    ]
    parsed: Dict[str, str] = {}
    structured = parse_structured_descriptor(src)
    for label in labels:
        value = " ".join(str(structured.get(label) or "").split()).strip(" ,.")
        if value:
            parsed[label] = value

    for label in labels:
        if parsed.get(label):
            continue
        match = re.search(
            rf"(?:^|,\s*){re.escape(label)}\s+(.+?)(?=(?:,\s*(?:{'|'.join(labels)})\s+)|$)",
            src,
            flags=re.IGNORECASE,
        )
        if match:
            value = " ".join(str(match.group(1) or "").split()).strip(" ,.")
            if value:
                parsed[label] = value
    return parsed


def serialize_prompt_fact_segments(fields: Dict[str, str]) -> str:
    """
    Serialize fact segments back into structured text.
    
    Args:
        fields: Dictionary of fact fields
        
    Returns:
        Serialized fact segments string
    """
    ordered_labels = (
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
    )
    parts: List[str] = []
    for label in ordered_labels:
        value = sanitize_prompt_fact_value(str(fields.get(label) or ""))
        if value:
            parts.append(f"{label}={value}")
    return "; ".join(parts).strip(" ;.")


def enrich_garment_descriptor(primary: str, fallback: str, garment_type: str) -> str:
    """
    Ensure analyze/extract prompts stay descriptive even when VLM emits a short sentence.
    
    Args:
        primary: Primary description
        fallback: Fallback description
        garment_type: Type of garment
        
    Returns:
        Enriched description
    """
    primary_clean = sanitize_florence_garment_description(primary or "")
    fallback_clean = sanitize_florence_garment_description(fallback or "")
    primary_clean = " ".join(primary_clean.split()).strip(" ,.")
    fallback_clean = " ".join(fallback_clean.split()).strip(" ,.")

    if not fallback_clean:
        return primary_clean or fallback_clean
    if not primary_clean:
        return fallback_clean
    if fallback_clean.lower() in primary_clean.lower():
        return primary_clean
    if primary_clean.lower() in fallback_clean.lower():
        return fallback_clean

    # Keep strongest descriptor first, append fallback context once.
    joined = f"{primary_clean}. {fallback_clean}"
    joined = re.sub(r"\s{2,}", " ", joined).strip(" .")
    if joined and garment_type:
        g = normalize_garment_type(garment_type) or garment_type
        if g.lower() not in joined.lower():
            joined = f"{g} garment. {joined}"
    return joined


def augment_identity_lock(user_description: str) -> str:
    """
    Augment user description with identity preservation instructions.
    
    Args:
        user_description: Base user description
        
    Returns:
        Augmented description with identity lock
    """
    base = str(user_description or "").strip()
    lock = (
        "Keep exact same person identity and pose: preserve facial geometry, expression, skin tone, "
        "hairstyle, hairline, hands, body proportions, camera framing, and scene lighting. "
        "Do not restyle face or change head/body posture."
    )
    if not base:
        return lock
    if "skin tone" in base.lower() and "face" in base.lower():
        return base
    return f"{base} {lock}"


def infer_flux2_target_type(description: str) -> str:
    """
    Infer target garment type from description text.
    
    Args:
        description: Description text to analyze
        
    Returns:
        Inferred target type ("dress", "top", "bottom", "outer")
    """
    text = str(description or "").lower()
    dress_terms = (
        "dress", "gown", "one-piece", "one piece", "maxi", "midi", "mini",
        "anarkali", "saree", "sari", "lehenga", "jumpsuit", "romper", "kurti",
    )
    bottom_terms = (
        "bottom", "pant", "pants", "trouser", "trousers", "jean", "jeans", "skirt", "shorts",
        "palazzo", "chino", "legging", "leggings",
    )
    outer_terms = ("jacket", "coat", "blazer", "hoodie", "cardigan", "outerwear", "outer", "shrug")
    top_terms = ("shirt", "t-shirt", "tee", "top", "blouse", "corset", "sweater", "kurta", "tunic")

    has_dress_term = any(term in text for term in dress_terms)
    has_bottom_term = any(term in text for term in bottom_terms)
    has_outer_term = any(term in text for term in outer_terms)
    has_top_term = any(term in text for term in top_terms)
    
    lower_body_cues = (
        "coverage full legs",
        "coverage legs",
        "coverage lower body",
        "lower body",
        "lower-body",
        "waist to ankle",
        "full legs",
        "leg coverage",
        "pants only",
    )
    upper_body_cues = (
        "coverage torso",
        "coverage upper body",
        "coverage upper-body",
        "torso and arms",
        "upper body",
        "upper-body",
    )
    has_lower_body_cue = any(cue in text for cue in lower_body_cues)
    has_upper_body_cue = any(cue in text for cue in upper_body_cues)

    # Be conservative with dress detection for model-generated captions.
    # If top/bottom/outer cues co-exist, prefer region-specific replacement.
    if has_dress_term and not (has_top_term or has_bottom_term or has_outer_term):
        return "dress"
    if has_lower_body_cue and not has_upper_body_cue:
        return "bottom"
    if has_top_term and has_bottom_term:
        if has_lower_body_cue:
            return "bottom"
        if has_upper_body_cue:
            return "top"
    if has_bottom_term:
        return "bottom"
    if has_outer_term:
        return "outer"
    if has_top_term:
        return "top"
    return "top"


def ensure_target_type_in_description(description: str, target_type: str) -> str:
    """
    Ensure target type is mentioned in description.
    
    Args:
        description: Input description
        target_type: Target garment type
        
    Returns:
        Description with target type ensured
    """
    text = str(description or "").strip()
    kind = str(target_type or "").strip().lower()
    if kind not in {"dress", "top", "bottom", "outer"}:
        return text

    alias_map = {
        "dress": ("dress", "gown", "one-piece", "maxi", "midi", "mini", "anarkali", "saree", "sari", "lehenga"),
        "top": ("top", "shirt", "blouse", "tee", "t-shirt", "kurta", "tunic", "corset", "sweater"),
        "bottom": ("bottom", "pants", "trousers", "jeans", "skirt", "shorts", "palazzo", "leggings"),
        "outer": ("outer", "outerwear", "jacket", "coat", "blazer", "hoodie", "cardigan"),
    }
    low = text.lower()
    if any(alias in low for alias in alias_map[kind]):
        return text

    if not text:
        return f"{kind} garment"
    return f"{kind} garment, {text}"


def merge_avoid_clause_sentences(*clauses: str) -> str:
    """
    Merge multiple avoid clause sentences into one.
    
    Args:
        *clauses: Variable number of clause strings
        
    Returns:
        Merged avoid clause string
    """
    seen: List[str] = []
    for clause in clauses:
        raw = " ".join(str(clause or "").split()).strip()
        if not raw:
            continue
        for segment in re.split(r"(?<=[.!?])\s+", raw):
            clean = segment.strip(" ,.")
            if not clean:
                continue
            sentence = clean if clean.endswith(".") else f"{clean}."
            if sentence not in seen:
                seen.append(sentence)
    return " ".join(seen).strip()


# Helper functions

def clean_prompt_section_text(text: str) -> str:
    """
    Clean prompt section text by removing prefixes and formatting.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    cleaned = " ".join(str(text or "").split()).strip()
    cleaned = re.sub(
        r"^(?:BASE_GARMENT_PROMPT|EXTRACTION_AVOID_CLAUSE)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned.strip(" -")


def join_avoid_terms(terms: List[str]) -> str:
    """
    Join avoid terms with proper grammar.
    
    Args:
        terms: List of terms to join
        
    Returns:
        Grammatically joined terms string
    """
    ordered = [str(term).strip() for term in terms if str(term).strip()]
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} and {ordered[1]}"
    return ", ".join(ordered[:-1]) + f", and {ordered[-1]}"


def extract_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text, handling various formats.
    
    Args:
        text: Input text that may contain JSON
        
    Returns:
        Parsed JSON dictionary or None if not found
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    
    # Try fenced code block first
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    
    # Try raw text
    candidates.append(raw)
    
    # Try brace-delimited content
    brace_match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1))
    
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    
    return None
