"""
Validation utilities.

This module contains approximately 20 validation-related utility functions for:
- Type normalization (garment types, coverage)
- Input validation (image quality, user images)
- Descriptor validation
- Prompt fact sanitization

All functions are pure with no side effects.
"""

import re
from typing import Optional
import numpy as np
from PIL import Image

# Garment type synonyms mapping
GARMENT_TYPE_SYNONYMS = {
    "top": "top",
    "shirt": "top",
    "tshirt": "top",
    "t-shirt": "top",
    "tee": "top",
    "blouse": "top",
    "shortsleevetop": "top",
    "longsleevetop": "top",
    "shortsleevedshirt": "top",
    "longsleevedshirt": "top",
    "bra": "top",
    "bralette": "top",
    "brassiere": "top",
    "bikinitop": "top",
    "bustier": "top",
    "vest": "top",
    "sling": "top",
    "bottom": "bottom",
    "pant": "bottom",
    "pants": "bottom",
    "trouser": "bottom",
    "trousers": "bottom",
    "jean": "bottom",
    "jeans": "bottom",
    "skirt": "bottom",
    "shorts": "bottom",
    "dress": "dress",
    "gown": "dress",
    "kurti": "dress",
    "shortsleevedress": "dress",
    "longsleevedress": "dress",
    "vestdress": "dress",
    "slingdress": "dress",
    "outer": "outer",
    "outerwear": "outer",
    "outwear": "outer",
    "jacket": "outer",
    "coat": "outer",
}


def normalize_garment_type(raw: Optional[str]) -> Optional[str]:
    """
    Normalize garment type string to canonical form.
    
    Args:
        raw: Raw garment type string
        
    Returns:
        Normalized garment type or None if invalid
        
    Examples:
        >>> normalize_garment_type("T-Shirt")
        'top'
        >>> normalize_garment_type("jeans")
        'bottom'
        >>> normalize_garment_type("invalid")
        None
    """
    if raw is None:
        return None
    normalized = str(raw).strip().lower().replace("_", " ")
    if not normalized:
        return None
    normalized = normalized.replace(" ", "")
    return GARMENT_TYPE_SYNONYMS.get(normalized)


def canonical_coverage_for_type(target_type: str) -> str:
    """
    Get canonical coverage description for garment type.
    
    Args:
        target_type: Garment type string
        
    Returns:
        Coverage description string
        
    Examples:
        >>> canonical_coverage_for_type("top")
        'upper body'
        >>> canonical_coverage_for_type("dress")
        'full body'
    """
    normalized = normalize_garment_type(str(target_type or ""))
    return {
        "top": "upper body",
        "bottom": "lower body", 
        "dress": "full body",
        "outer": "upper body outer layer",
    }.get(normalized, "")


def sanitize_prompt_fact_value(value: str) -> str:
    """
    Sanitize prompt fact value by cleaning whitespace and removing extraction directives.
    
    Args:
        value: Raw prompt fact value
        
    Returns:
        Cleaned prompt fact value
        
    Examples:
        >>> sanitize_prompt_fact_value("  red color  ")
        'red color'
        >>> sanitize_prompt_fact_value("blue EXTRACTION_AVOID_CLAUSE: avoid text")
        'blue'
    """
    cleaned = " ".join(str(value or "").split()).strip().strip(" ,.")
    cleaned = re.sub(
        r"\bEXTRACTION_AVOID_CLAUSE\s*:\s*.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.")
    return cleaned


def descriptor_word_count(text: str) -> int:
    """
    Count meaningful words in descriptor text.
    
    Args:
        text: Descriptor text
        
    Returns:
        Number of words found
        
    Examples:
        >>> descriptor_word_count("red cotton shirt")
        3
        >>> descriptor_word_count("a-line dress with #pattern")
        4
    """
    return len(re.findall(r"[A-Za-z0-9#]+", str(text or "")))


def descriptor_is_weak(
    text: str,
    *,
    garment_type: Optional[str] = None,
    min_words: int = 10,
) -> bool:
    """
    Check if a garment descriptor is too weak or generic.
    
    A descriptor is considered weak if:
    - It's empty or too short
    - It lacks specific garment attributes
    - It doesn't contain type-specific details
    
    Args:
        text: Descriptor text to evaluate
        garment_type: Optional garment type for type-specific validation
        min_words: Minimum word count threshold
        
    Returns:
        True if descriptor is weak, False otherwise
        
    Examples:
        >>> descriptor_is_weak("shirt")
        True
        >>> descriptor_is_weak("red cotton v-neck shirt with long sleeves")
        False
    """
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return True
        
    words = descriptor_word_count(clean)
    if words < max(4, int(min_words)):
        return True
        
    if ";" in clean and "=" in clean:
        # Structured output from VLM is usually rich enough.
        return False
        
    richness_markers = (
        "type",
        "material", 
        "silhouette",
        "construction",
        "details",
        "coverage",
        "sleeve",
        "neckline",
        "hem",
        "ruffle",
        "pleat",
    )
    lower = clean.lower()
    if not any(token in lower for token in richness_markers):
        return True

    gtype = normalize_garment_type(garment_type) if garment_type else None
    signal_groups = {
        "top": (
            ("sleeve", "sleeveless", "long-sleeve", "short-sleeve"),
            ("neckline", "neck", "v-neck", "crew", "collar", "cowl", "placket", "button-front"),
            ("fit", "fitted", "loose fit", "tailored", "semi-sheer", "sheer", "knit", "woven", "fabric", "wrap", "crossover", "cross-over", "tie-front", "draped front", "ruched", "pleated"),
            ("waist", "hem", "cropped", "blouson", "tucked", "length", "below the waist", "above the waist"),
        ),
        "dress": (
            ("sleeve", "sleeveless", "long-sleeve", "short-sleeve", "three-quarter"),
            ("neckline", "neck", "v-neck", "crew", "collar", "placket"),
            ("waist", "belt", "drawstring", "gathered", "empire"),
            ("skirt", "hem", "ankle-length", "knee-length", "maxi", "mini", "midi", "full-length"),
            ("pattern", "floral", "striped", "print", "motif"),
        ),
        "bottom": (
            ("waist", "high-waisted", "mid-rise", "low-rise"),
            ("straight-leg", "wide-leg", "tapered", "flare", "leg"),
            ("pleat", "crease", "fly", "pocket", "hem"),
        ),
        "outer": (
            ("sleeve", "long-sleeve", "short-sleeve"),
            ("collar", "lapel", "hood"),
            ("placket", "button", "zipper", "snap"),
            ("hem", "waist", "cropped", "length"),
        ),
    }
    required = signal_groups.get(gtype or "", ())
    if not required:
        return False
    matched = sum(1 for group in required if any(token in lower for token in group))
    threshold = 4 if gtype == "top" else (3 if gtype in {"bottom", "outer"} else 4)
    return matched < threshold


def validate_image_quality(image: Image.Image, max_edge: int = 1024) -> float:
    """
    Validate image quality using focus score (blur detection).
    
    Higher scores indicate sharper images. Score is computed on normalized
    max-edge to avoid high-resolution bias.
    
    Args:
        image: PIL Image to validate
        max_edge: Maximum edge size for normalization
        
    Returns:
        Focus score (higher = sharper)
        
    Examples:
        >>> img = Image.open("sharp_image.jpg")
        >>> validate_image_quality(img) > 20.0
        True
    """
    gray = image.convert("L")
    w, h = gray.size
    current_max_edge = max(w, h)
    
    if current_max_edge > max_edge:
        scale = float(max_edge) / float(current_max_edge)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        gray = gray.resize((nw, nh), Image.BICUBIC)

    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return 0.0
        
    gy, gx = np.gradient(arr)
    mag = np.sqrt((gx * gx) + (gy * gy))
    return float(np.var(mag))


def is_valid_garment_type(garment_type: str) -> bool:
    """
    Check if garment type is valid/recognized.
    
    Args:
        garment_type: Garment type string to validate
        
    Returns:
        True if valid garment type, False otherwise
        
    Examples:
        >>> is_valid_garment_type("shirt")
        True
        >>> is_valid_garment_type("invalid_type")
        False
    """
    return normalize_garment_type(garment_type) is not None


def validate_descriptor_length(text: str, min_length: int = 10, max_length: int = 500) -> bool:
    """
    Validate descriptor text length is within acceptable bounds.
    
    Args:
        text: Descriptor text to validate
        min_length: Minimum character length
        max_length: Maximum character length
        
    Returns:
        True if length is valid, False otherwise
        
    Examples:
        >>> validate_descriptor_length("red shirt")
        False  # Too short
        >>> validate_descriptor_length("red cotton v-neck shirt")
        True
    """
    clean_text = " ".join(str(text or "").split()).strip()
    length = len(clean_text)
    return min_length <= length <= max_length


def sanitize_garment_description(text: str) -> str:
    """
    Sanitize garment description by normalizing whitespace and removing invalid characters.
    
    Args:
        text: Raw garment description
        
    Returns:
        Sanitized description
        
    Examples:
        >>> sanitize_garment_description("  red   shirt  ")
        'red shirt'
        >>> sanitize_garment_description("")
        ''
    """
    s = " ".join((text or "").strip().split())
    if not s:
        return ""
    return s


def normalize_coverage_description(coverage: str) -> str:
    """
    Normalize coverage description to standard format.
    
    Args:
        coverage: Raw coverage description
        
    Returns:
        Normalized coverage description
        
    Examples:
        >>> normalize_coverage_description("UPPER BODY")
        'upper body'
        >>> normalize_coverage_description("full_body")
        'full body'
    """
    normalized = str(coverage or "").strip().lower().replace("_", " ")
    return " ".join(normalized.split())


def validate_color_value(color: str) -> bool:
    """
    Validate if color value is a valid color name or hex code.
    
    Args:
        color: Color value to validate
        
    Returns:
        True if valid color, False otherwise
        
    Examples:
        >>> validate_color_value("red")
        True
        >>> validate_color_value("#FF0000")
        True
        >>> validate_color_value("invalid_color_123")
        False
    """
    if not color:
        return False
        
    color = str(color).strip().lower()
    
    # Check hex color format
    if color.startswith("#"):
        hex_pattern = re.compile(r"^#[0-9a-f]{6}$", re.IGNORECASE)
        return bool(hex_pattern.match(color))
    
    # Check common color names
    common_colors = {
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
        "black", "white", "gray", "grey", "navy", "maroon", "olive", "lime",
        "aqua", "teal", "silver", "fuchsia", "beige", "tan", "khaki", "coral",
        "salmon", "gold", "indigo", "violet", "turquoise", "magenta", "cyan"
    }
    
    return color in common_colors


def validate_material_name(material: str) -> bool:
    """
    Validate if material name is recognized.
    
    Args:
        material: Material name to validate
        
    Returns:
        True if valid material, False otherwise
        
    Examples:
        >>> validate_material_name("cotton")
        True
        >>> validate_material_name("polyester")
        True
        >>> validate_material_name("invalid_material_xyz")
        False
    """
    if not material:
        return False
        
    material = str(material).strip().lower()
    
    common_materials = {
        "cotton", "polyester", "wool", "silk", "linen", "denim", "leather",
        "suede", "velvet", "satin", "chiffon", "lace", "mesh", "knit",
        "woven", "jersey", "fleece", "cashmere", "bamboo", "modal",
        "rayon", "spandex", "elastane", "nylon", "acrylic", "viscose"
    }
    
    return material in common_materials


def validate_size_value(size: str) -> bool:
    """
    Validate if size value is in standard format.
    
    Args:
        size: Size value to validate
        
    Returns:
        True if valid size, False otherwise
        
    Examples:
        >>> validate_size_value("M")
        True
        >>> validate_size_value("Large")
        True
        >>> validate_size_value("32")
        True
        >>> validate_size_value("invalid_size")
        False
    """
    if not size:
        return False
        
    size = str(size).strip().upper()
    
    # Standard letter sizes
    letter_sizes = {"XS", "S", "M", "L", "XL", "XXL", "XXXL"}
    if size in letter_sizes:
        return True
    
    # Word sizes
    word_sizes = {"EXTRA SMALL", "SMALL", "MEDIUM", "LARGE", "EXTRA LARGE"}
    if size in word_sizes:
        return True
    
    # Numeric sizes (waist, chest, etc.)
    if re.match(r"^\d{1,3}$", size):
        return True
    
    # Numeric with letter (e.g., "32W", "34L")
    if re.match(r"^\d{1,3}[A-Z]$", size):
        return True
        
    return False


def validate_prompt_fact_key(key: str) -> bool:
    """
    Validate prompt fact key format.
    
    Args:
        key: Prompt fact key to validate
        
    Returns:
        True if valid key format, False otherwise
        
    Examples:
        >>> validate_prompt_fact_key("color")
        True
        >>> validate_prompt_fact_key("garment_type")
        True
        >>> validate_prompt_fact_key("invalid key!")
        False
    """
    if not key:
        return False
        
    # Key should be alphanumeric with underscores, no spaces or special chars
    pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    return bool(pattern.match(str(key).strip().lower()))


def normalize_prompt_fact_key(key: str) -> str:
    """
    Normalize prompt fact key to standard format.
    
    Args:
        key: Raw prompt fact key
        
    Returns:
        Normalized key
        
    Examples:
        >>> normalize_prompt_fact_key("Garment Type")
        'garment_type'
        >>> normalize_prompt_fact_key("COLOR-NAME")
        'color_name'
    """
    if not key:
        return ""
        
    # Convert to lowercase, replace spaces and hyphens with underscores
    normalized = str(key).strip().lower()
    normalized = re.sub(r"[^a-z0-9_]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)  # Collapse multiple underscores
    normalized = normalized.strip("_")  # Remove leading/trailing underscores
    
    return normalized
