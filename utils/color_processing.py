"""
Color processing utilities.

This module contains functions for color manipulation, conversion, and analysis:
- RGB/LAB/HSV color space conversions
- Hex color parsing and validation
- Color distance calculations (Delta E)
- Color family and undertone classification
- Color palette extraction and analysis
- Color brightness and saturation bucketing
"""

import re
import colorsys
import logging
from typing import Optional, List, Tuple, Dict

import numpy as np
from PIL import Image

# Import shared utilities
from shared.image_ops import rgb_to_lab, delta_e_cie76

logger = logging.getLogger("glamify-ai")


# Color label to RGB mapping (from main.py)
_COLOR_LABEL_RGB = [
    ("red", (220, 20, 60)),
    ("crimson", (220, 20, 60)),
    ("burgundy", (128, 0, 32)),
    ("maroon", (128, 0, 0)),
    ("pink", (255, 192, 203)),
    ("blush pink", (255, 192, 203)),
    ("hot pink", (255, 105, 180)),
    ("dusty pink", (220, 180, 180)),
    ("rose gold", (183, 110, 121)),
    ("orange", (255, 165, 0)),
    ("peach", (255, 218, 185)),
    ("yellow", (255, 255, 0)),
    ("gold", (255, 215, 0)),
    ("green", (0, 128, 0)),
    ("olive", (128, 128, 0)),
    ("blue", (0, 0, 255)),
    ("navy", (0, 0, 128)),
    ("teal", (0, 128, 128)),
    ("purple", (128, 0, 128)),
    ("lavender", (230, 230, 250)),
    ("plum", (221, 160, 221)),
    ("brown", (165, 42, 42)),
    ("beige", (245, 245, 220)),
    ("tan", (210, 180, 140)),
    ("champagne", (247, 231, 206)),
    ("nude", (230, 200, 180)),
    ("ivory", (255, 255, 240)),
    ("off-white", (250, 250, 248)),
    ("white", (255, 255, 255)),
    ("silver", (192, 192, 192)),
    ("gray", (128, 128, 128)),
    ("charcoal", (54, 69, 79)),
    ("black", (0, 0, 0)),
]

_COLOR_LABEL_RGB_MAP: Dict[str, Tuple[int, int, int]] = dict(_COLOR_LABEL_RGB)


def canonical_color_token(color: str) -> str:
    """
    Normalize color token by applying common aliases.
    
    Args:
        color: Color name string
        
    Returns:
        Normalized color token
    """
    token = str(color or "").strip().lower()
    aliases = {
        "grey": "gray",
        "off white": "off-white",
        "offwhite": "off-white",
    }
    return aliases.get(token, token)


def color_family(color: str) -> str:
    """
    Map color name to its family group.
    
    Args:
        color: Color name string
        
    Returns:
        Color family (e.g., "red", "blue", "neutral_dark")
    """
    c = canonical_color_token(color)
    if c in {"black", "charcoal"}:
        return "neutral_dark"
    if c in {"gray", "silver"}:
        return "neutral_mid"
    if c in {"white", "off-white", "ivory", "beige", "champagne", "tan", "nude"}:
        return "neutral_light"
    if c in {"red", "maroon", "burgundy"}:
        return "red"
    if c in {"orange", "peach"}:
        return "orange"
    if c in {"yellow", "gold"}:
        return "yellow"
    if c in {"green", "olive"}:
        return "green"
    if c in {"blue", "navy", "teal"}:
        return "blue"
    if c in {"purple", "lavender", "plum"}:
        return "purple"
    if c in {"pink", "blush pink", "dusty pink", "hot pink", "rose gold"}:
        return "pink"
    if c in {"brown"}:
        return "brown"
    return c


def is_neutral_color_token(color: str) -> bool:
    """
    Check if color is a neutral (black/white/gray).
    
    Args:
        color: Color name string
        
    Returns:
        True if color is neutral
    """
    fam = color_family(color)
    return fam in {"neutral_dark", "neutral_mid", "neutral_light"}


def hex_to_rgb_triplet(token: str) -> Optional[Tuple[int, int, int]]:
    """
    Convert hex color string to RGB triplet.
    
    Args:
        token: Hex color string (with or without #)
        
    Returns:
        RGB triplet (r, g, b) or None if invalid
    """
    t = str(token or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", t):
        return None
    return (int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB triplet to hex color string.
    
    Args:
        rgb: RGB triplet (r, g, b)
        
    Returns:
        Hex color string with # prefix
    """
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_hue_deg(rgb: Tuple[int, int, int]) -> float:
    """
    Calculate hue in degrees from RGB triplet.
    
    Args:
        rgb: RGB triplet (r, g, b)
        
    Returns:
        Hue in degrees (0-360)
    """
    r, g, b = [max(0, min(255, int(v))) / 255.0 for v in rgb]
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return float(h * 360.0)


def hex_to_lab_triplet(hex_color: str) -> Optional[np.ndarray]:
    """
    Convert hex color to LAB color space.
    
    Args:
        hex_color: Hex color string
        
    Returns:
        LAB triplet as numpy array or None if invalid
    """
    token = str(hex_color or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", token):
        return None
    rgb = hex_to_rgb_triplet(token)
    if rgb is None:
        return None
    return rgb_to_lab(np.array([[rgb]], dtype=np.uint8))[0, 0]


def palette_weighted_hue_deg(palette: List[Dict[str, object]], max_colors: int = 4) -> Optional[float]:
    """
    Calculate weighted average hue from color palette.
    
    Args:
        palette: List of color entries with 'hex' and 'areaPercent' keys
        max_colors: Maximum number of colors to consider
        
    Returns:
        Weighted hue in degrees or None if invalid
    """
    vals: List[Tuple[float, float]] = []
    for entry in (palette or [])[: max(1, int(max_colors))]:
        hx = str(entry.get("hex", "")).strip()
        rgb = hex_to_rgb_triplet(hx)
        if rgb is None:
            continue
        w = float(entry.get("areaPercent", 0.0) or 0.0)
        vals.append((rgb_hue_deg(rgb), max(0.1, w)))
    if not vals:
        return None
    # Weighted circular mean
    sin_sum = 0.0
    cos_sum = 0.0
    for deg, w in vals:
        rad = np.deg2rad(deg)
        sin_sum += np.sin(rad) * w
        cos_sum += np.cos(rad) * w
    if abs(sin_sum) < 1e-6 and abs(cos_sum) < 1e-6:
        return None
    ang = float(np.rad2deg(np.arctan2(sin_sum, cos_sum)))
    if ang < 0.0:
        ang += 360.0
    return ang


def palette_hue_from_hexes(hexes: Optional[List[str]]) -> Optional[float]:
    """
    Calculate weighted hue from list of hex colors.
    
    Args:
        hexes: List of hex color strings
        
    Returns:
        Weighted hue in degrees or None if invalid
    """
    palette: List[Dict[str, object]] = []
    for idx, hx in enumerate(hexes or []):
        token = str(hx or "").strip()
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", token):
            continue
        palette.append({"hex": token, "areaPercent": max(1.0, float(100 - idx * 10))})
    if not palette:
        return None
    return palette_weighted_hue_deg(palette, max_colors=min(4, len(palette)))


def palette_supports_color_family(
    family: str,
    dominant_hexes: Optional[List[str]],
    color_profile: Optional[Dict[str, object]],
) -> bool:
    """
    Heuristic check to see whether a palette is compatible with a color family.

    Args:
        family: Color family string
        dominant_hexes: List of dominant hex colors
        color_profile: Color profile dict (LAB stats)

    Returns:
        True if palette supports the requested family, else False
    """
    fam = str(family or "").strip().lower()
    if not fam or fam.startswith("neutral") or fam == "brown":
        return True

    hue = palette_hue_from_hexes(dominant_hexes)
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


def bucket_color_brightness(profile: Optional[Dict[str, object]]) -> str:
    """
    Classify color brightness from LAB profile.
    
    Args:
        profile: Color profile dict with 'medianL' key
        
    Returns:
        Brightness bucket: "very_light", "light", "mid", "deep", "dark", or "unknown"
    """
    median_l = profile.get("medianL") if isinstance(profile, dict) else None
    if not isinstance(median_l, (int, float)):
        return "unknown"
    val = float(median_l)
    if val >= 84.0:
        return "very_light"
    if val >= 68.0:
        return "light"
    if val >= 48.0:
        return "mid"
    if val >= 30.0:
        return "deep"
    return "dark"


def bucket_color_saturation(profile: Optional[Dict[str, object]]) -> str:
    """
    Classify color saturation from LAB profile.
    
    Args:
        profile: Color profile dict with 'meanChroma' and 'medianL' keys
        
    Returns:
        Saturation bucket: "neutral", "pale", "muted", "soft", "balanced", "rich", "vivid", or "unknown"
    """
    mean_chroma = profile.get("meanChroma") if isinstance(profile, dict) else None
    median_l = profile.get("medianL") if isinstance(profile, dict) else None
    if not isinstance(mean_chroma, (int, float)):
        return "unknown"
    
    chroma = float(mean_chroma)
    lightish = isinstance(median_l, (int, float)) and float(median_l) >= 70.0
    
    if chroma < 6.0:
        return "neutral"
    if chroma < 12.0:
        return "pale" if lightish else "muted"
    if chroma < 20.0:
        return "soft"
    if chroma < 34.0:
        return "balanced"
    if chroma < 48.0:
        return "rich"
    return "vivid"


def bucket_color_undertone(profile: Optional[Dict[str, object]]) -> str:
    """
    Classify color undertone from LAB profile.
    
    Args:
        profile: Color profile dict with 'meanA' and 'meanB' keys
        
    Returns:
        Undertone: "warm", "cool", or "neutral"
    """
    if not isinstance(profile, dict):
        return "unknown"
    
    mean_a = profile.get("meanA")
    mean_b = profile.get("meanB")
    
    if not isinstance(mean_a, (int, float)) or not isinstance(mean_b, (int, float)):
        return "unknown"
    
    a_val = float(mean_a)
    b_val = float(mean_b)
    
    if abs(a_val) <= 4.0 and abs(b_val) <= 4.0:
        return "neutral"
    if b_val >= 6.0:
        return "warm"
    if b_val <= -6.0:
        return "cool"
    if a_val >= 7.0:
        return "warm"
    if a_val <= -7.0:
        return "cool"
    return "neutral"


def compose_color_descriptor_phrase(
    primary_label: str,
    brightness: str,
    saturation: str,
    undertone: str,
) -> str:
    """
    Build a descriptive color phrase from label and tone buckets.

    Args:
        primary_label: Base color label
        brightness: Brightness bucket
        saturation: Saturation bucket
        undertone: Undertone bucket

    Returns:
        Human-readable color description
    """
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


def nearest_color_label(rgb_triplet: Tuple[int, int, int]) -> str:
    """
    Find nearest named color label for RGB triplet using Delta E.
    
    Args:
        rgb_triplet: RGB color (r, g, b)
        
    Returns:
        Nearest color label name
    """
    lab_target = rgb_to_lab(np.array([[rgb_triplet]], dtype=np.uint8))[0, 0]
    
    best_label = "unknown"
    best_distance = float("inf")
    
    for label, rgb_ref in _COLOR_LABEL_RGB:
        lab_ref = rgb_to_lab(np.array([[rgb_ref]], dtype=np.uint8))[0, 0]
        distance = delta_e_cie76(lab_target, lab_ref)
        if distance < best_distance:
            best_distance = distance
            best_label = label
    
    return best_label


def color_labels_from_hex_palette(hexes: List[str], top_k: int = 3) -> List[str]:
    """
    Convert hex palette to named color labels.
    
    Args:
        hexes: List of hex color strings
        top_k: Maximum number of labels to return
        
    Returns:
        List of color label names
    """
    out: List[str] = []
    for hx in hexes:
        rgb = hex_to_rgb_triplet(hx)
        if rgb is None:
            continue
        label = nearest_color_label(rgb)
        if label and label not in out:
            out.append(label)
        if len(out) >= top_k:
            break
    return out


def compute_palette_delta_e(
    input_hexes: List[str],
    output_hexes: List[str],
    max_pairs: int = 3,
) -> Dict[str, object]:
    """
    Compute color fidelity metrics between input and output palettes.
    
    Args:
        input_hexes: Input hex colors
        output_hexes: Output hex colors
        max_pairs: Maximum color pairs to compare
        
    Returns:
        Dict with delta_e metrics (min, max, mean, median)
    """
    input_labs: List[np.ndarray] = []
    for hx in (input_hexes or [])[: max(1, int(max_pairs))]:
        lab = hex_to_lab_triplet(hx)
        if lab is not None:
            input_labs.append(lab)
    
    output_labs: List[np.ndarray] = []
    for hx in (output_hexes or [])[: max(1, int(max_pairs))]:
        lab = hex_to_lab_triplet(hx)
        if lab is not None:
            output_labs.append(lab)
    
    if not input_labs or not output_labs:
        return {
            "min_delta_e": None,
            "max_delta_e": None,
            "mean_delta_e": None,
            "median_delta_e": None,
            "pairs_compared": 0,
        }
    
    deltas: List[float] = []
    for in_lab in input_labs:
        for out_lab in output_labs:
            delta = delta_e_cie76(in_lab, out_lab)
            deltas.append(float(delta))
    
    if not deltas:
        return {
            "min_delta_e": None,
            "max_delta_e": None,
            "mean_delta_e": None,
            "median_delta_e": None,
            "pairs_compared": 0,
        }
    
    return {
        "min_delta_e": float(np.min(deltas)),
        "max_delta_e": float(np.max(deltas)),
        "mean_delta_e": float(np.mean(deltas)),
        "median_delta_e": float(np.median(deltas)),
        "pairs_compared": len(deltas),
    }


def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB triplet to HSV color space.
    
    Args:
        rgb: RGB triplet (r, g, b) in range 0-255
        
    Returns:
        HSV triplet (h, s, v) where h is in degrees (0-360), s and v are 0-1
    """
    r, g, b = [max(0, min(255, int(v))) / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360.0, s, v)


def hsv_to_rgb(hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Convert HSV color space to RGB triplet.
    
    Args:
        hsv: HSV triplet (h, s, v) where h is in degrees (0-360), s and v are 0-1
        
    Returns:
        RGB triplet (r, g, b) in range 0-255
    """
    h, s, v = hsv
    h_norm = (h % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h_norm, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert LAB color space to RGB.
    
    Args:
        lab: LAB array (can be single pixel or image)
        
    Returns:
        RGB array in range 0-255
    """
    try:
        import cv2
        # Ensure proper shape
        original_shape = lab.shape
        if lab.ndim == 1:
            lab = lab.reshape(1, 1, 3)
        elif lab.ndim == 2:
            lab = lab.reshape(lab.shape[0], 1, 3)
        
        # Convert LAB to RGB
        lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)
        
        # Restore original shape
        if len(original_shape) == 1:
            return rgb.reshape(3)
        elif len(original_shape) == 2:
            return rgb.reshape(original_shape[0], 3)
        return rgb
    except Exception:
        # Fallback: return as-is
        return lab.astype(np.uint8)


def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """
    Calculate perceptual color distance using Delta E (CIE76).
    
    Args:
        color1: First RGB color (r, g, b)
        color2: Second RGB color (r, g, b)
        
    Returns:
        Delta E distance (lower = more similar)
    """
    lab1 = rgb_to_lab(np.array([[color1]], dtype=np.uint8))[0, 0]
    lab2 = rgb_to_lab(np.array([[color2]], dtype=np.uint8))[0, 0]
    return delta_e_cie76(lab1, lab2)


def extract_color_palette(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    top_k: int = 4,
    decontamination_enabled: bool = True,
) -> List[Dict[str, object]]:
    """
    Extract dominant colors from image using K-Means clustering in LAB space.
    
    Args:
        image: Input image
        mask: Optional binary mask to focus on specific regions
        top_k: Number of dominant colors to extract
        decontamination_enabled: Whether to apply foreground cleaning
        
    Returns:
        List of color entries with 'hex', 'areaPercent', and 'pixelCount' keys
    """
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return []

        pixels = arr.reshape(-1, 3)
        use_mask = mask
        
        # Apply decontamination if enabled
        if decontamination_enabled:
            clean_mask = _get_clean_foreground_mask(image, mask=mask)
            if isinstance(clean_mask, np.ndarray):
                use_mask = clean_mask
        
        if isinstance(use_mask, np.ndarray):
            m = np.asarray(use_mask).astype(bool)
            if m.shape[:2] == arr.shape[:2]:
                keep = m.reshape(-1)
                if int(np.sum(keep)) > 32:
                    pixels = pixels[keep]

        if pixels.size == 0:
            return []

        # Remove near-white pixels
        near_white = np.all(pixels >= 248, axis=1)
        if int(np.sum(~near_white)) > 16:
            pixels = pixels[~near_white]
        if pixels.size < 4:
            return []

        import cv2

        # Convert to LAB for better perceptual clustering
        pixels_u8 = np.ascontiguousarray(pixels.astype(np.uint8))
        pixels_lab = cv2.cvtColor(pixels_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        if pixels_lab.size < 12:
            return []

        # K-Means clustering
        unique_lab_count = int(np.unique(pixels_lab, axis=0).shape[0])
        k = min(max(1, int(top_k) + 1), int(pixels_lab.shape[0]), max(1, unique_lab_count))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels_lab.astype(np.float32),
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        flat_labels = labels.reshape(-1)
        unique, counts = np.unique(flat_labels, return_counts=True)
        order = np.argsort(-counts)

        # Convert cluster centers back to RGB
        centers_u8 = np.clip(centers, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        centers_rgb = cv2.cvtColor(centers_u8, cv2.COLOR_LAB2RGB).reshape(-1, 3)

        total_pixels = max(1, int(np.sum(counts)))
        out: List[Dict[str, object]] = []
        seen_hex = set()
        
        for order_idx in order.tolist():
            center_idx = int(unique[order_idx])
            r, g, b = [int(v) for v in centers_rgb[center_idx].tolist()]
            r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
            hx = f"#{r:02X}{g:02X}{b:02X}"
            if hx in seen_hex:
                continue
            seen_hex.add(hx)
            pixel_count = int(counts[order_idx])
            area_percent = round((float(pixel_count) / float(total_pixels)) * 100.0, 2)
            out.append(
                {
                    "hex": hx,
                    "areaPercent": area_percent,
                    "pixelCount": pixel_count,
                }
            )
            if len(out) >= max(1, int(top_k)):
                break
        return out
    except Exception as e:
        logger.warning(f"K-Means color extraction failed: {e}")
        return []


def filter_palette_entries_by_area(
    palette: List[Dict[str, object]],
    min_area_percent: float,
    min_items: int = 2,
) -> List[Dict[str, object]]:
    """
    Filter palette entries by minimum area coverage.
    
    Args:
        palette: List of color entries with 'areaPercent' key
        min_area_percent: Minimum area percentage threshold
        min_items: Minimum number of items to keep regardless of threshold
        
    Returns:
        Filtered palette entries
    """
    entries = list(palette or [])
    if not entries:
        return []
    threshold = float(max(0.0, min(100.0, min_area_percent)))
    filtered = [
        e for e in entries
        if float(e.get("areaPercent", 0.0) or 0.0) >= threshold
    ]
    if len(filtered) >= int(max(1, min_items)):
        return filtered
    return entries[: max(1, min(int(min_items), len(entries)))]


def extract_lab_color_profile(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    decontamination_enabled: bool = True,
    trim_dark_percentile: float = 10.0,
    trim_bright_percentile: float = 98.0,
) -> Dict[str, object]:
    """
    Extract compact LAB color profile for neutral fidelity analysis.
    
    Args:
        image: Input image
        mask: Optional binary mask
        decontamination_enabled: Whether to apply foreground cleaning
        trim_dark_percentile: Percentile for dark outlier trimming
        trim_bright_percentile: Percentile for bright outlier trimming
        
    Returns:
        Dict with LAB statistics: medianL, p10L, p90L, meanChroma, p90Chroma, meanA, meanB, isNeutral
    """
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return {}

        use_mask = None
        if decontamination_enabled:
            clean_mask = _get_clean_foreground_mask(image, mask=mask)
            if isinstance(clean_mask, np.ndarray):
                use_mask = clean_mask
        if not isinstance(use_mask, np.ndarray):
            use_mask = mask
        if not isinstance(use_mask, np.ndarray):
            use_mask = _extract_alpha_mask(image)
        if not isinstance(use_mask, np.ndarray):
            use_mask = _estimate_foreground_mask_from_border(image)

        pixels = arr.reshape(-1, 3)
        if isinstance(use_mask, np.ndarray) and use_mask.shape[:2] == arr.shape[:2]:
            keep = np.asarray(use_mask).astype(bool).reshape(-1)
            if int(np.sum(keep)) > 64:
                pixels = pixels[keep]

        if pixels.size == 0:
            return {}

        # Remove near-white pixels
        near_white = np.all(pixels >= 248, axis=1)
        if int(np.sum(~near_white)) > 16:
            pixels = pixels[~near_white]
        if pixels.size < 64:
            return {}

        import cv2
        lab = cv2.cvtColor(
            np.ascontiguousarray(pixels.astype(np.uint8)).reshape(-1, 1, 3),
            cv2.COLOR_RGB2LAB,
        ).reshape(-1, 3).astype(np.float32)

        lab_for_stats = lab
        if decontamination_enabled and int(lab.shape[0]) >= 64:
            quick_a = lab[:, 1] - 128.0
            quick_b = lab[:, 2] - 128.0
            quick_chroma = np.sqrt((quick_a * quick_a) + (quick_b * quick_b))
            quick_mean_chroma = float(np.mean(quick_chroma))
            quick_p90_chroma = float(np.percentile(quick_chroma, 90))
            # Trimming is most helpful for neutral/near-neutral garments
            if quick_mean_chroma < 18.0 and quick_p90_chroma < 28.0:
                trimmed = _trim_lab_profile_outliers(
                    lab,
                    dark_percentile=trim_dark_percentile,
                    bright_percentile=trim_bright_percentile,
                    min_pixels=64,
                )
                if isinstance(trimmed, np.ndarray) and int(trimmed.shape[0]) >= 64:
                    lab_for_stats = trimmed

        l_star = lab_for_stats[:, 0] * (100.0 / 255.0)
        a_star = lab_for_stats[:, 1] - 128.0
        b_star = lab_for_stats[:, 2] - 128.0
        chroma = np.sqrt((a_star * a_star) + (b_star * b_star))

        median_l = float(np.median(l_star))
        p10_l = float(np.percentile(l_star, 10))
        p90_l = float(np.percentile(l_star, 90))
        mean_chroma = float(np.mean(chroma))
        p90_chroma = float(np.percentile(chroma, 90))
        mean_a = float(np.mean(a_star))
        mean_b = float(np.mean(b_star))

        neutral = (mean_chroma < 16.0) and (p90_chroma < 26.0)
        return {
            "medianL": round(median_l, 2),
            "p10L": round(p10_l, 2),
            "p90L": round(p90_l, 2),
            "meanChroma": round(mean_chroma, 2),
            "p90Chroma": round(p90_chroma, 2),
            "meanA": round(mean_a, 2),
            "meanB": round(mean_b, 2),
            "isNeutral": bool(neutral),
        }
    except Exception:
        return {}


def _extract_alpha_mask(image: Image.Image, threshold: int = 24) -> Optional[np.ndarray]:
    """
    Returns a foreground mask from alpha channel when available.
    
    Args:
        image: Input image
        threshold: Alpha threshold (0-255)
        
    Returns:
        Binary mask or None
    """
    try:
        rgba = image.convert("RGBA")
        alpha = np.array(rgba, dtype=np.uint8)[:, :, 3]
        mask = alpha >= int(max(1, threshold))
        keep = int(np.sum(mask))
        if keep < 64:
            return None
        coverage = float(keep) / float(mask.shape[0] * mask.shape[1])
        if coverage >= 0.985:
            # Mostly opaque image; alpha likely not meaningful
            return None
        return mask
    except Exception:
        return None


def _estimate_foreground_mask_from_border(
    image: Image.Image,
    quant_step: int = 16,
    bg_tolerance: int = 28,
) -> Optional[np.ndarray]:
    """
    Border-based foreground estimation for images without useful alpha.
    
    Args:
        image: Input image
        quant_step: Quantization step for color binning
        bg_tolerance: Background color tolerance
        
    Returns:
        Binary foreground mask or None
    """
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        h, w = arr.shape[:2]
        if h < 8 or w < 8:
            return None

        b = max(1, min(h, w) // 30)
        border_pixels = np.concatenate(
            [
                arr[:b, :, :].reshape(-1, 3),
                arr[h - b :, :, :].reshape(-1, 3),
                arr[:, :b, :].reshape(-1, 3),
                arr[:, w - b :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        if border_pixels.size == 0:
            return None

        binned = (border_pixels // quant_step) * quant_step
        unique, counts = np.unique(binned, axis=0, return_counts=True)
        bg_rgb = unique[int(np.argmax(counts))]

        diff = arr.astype(np.int16) - bg_rgb.astype(np.int16)
        dist2 = np.sum(diff * diff, axis=2)
        fg_mask = dist2 > int(bg_tolerance * bg_tolerance)
        keep = int(np.sum(fg_mask))
        if keep < 64:
            return None
        coverage = float(keep) / float(h * w)
        if coverage < 0.02 or coverage > 0.98:
            return None
        return fg_mask
    except Exception:
        return None


def _erode_binary_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Erode binary mask using morphological operations.
    
    Args:
        mask: Binary mask
        iterations: Number of erosion iterations
        
    Returns:
        Eroded mask
    """
    arr = np.asarray(mask).astype(bool)
    if int(iterations) <= 0:
        return arr
    try:
        import cv2
        eroded = cv2.erode(
            arr.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=max(1, int(iterations)),
        )
        return eroded > 0
    except Exception:
        return arr


def _get_clean_foreground_mask(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    alpha_high_threshold: int = 240,
    alpha_low_threshold: int = 200,
    erode_iterations: int = 1,
    min_pixels: int = 64,
    min_coverage_ratio: float = 0.006,
) -> Optional[np.ndarray]:
    """
    Builds a high-confidence garment mask for color profiling.
    Combines strong alpha cues with optional external mask and edge erosion.
    
    Args:
        image: Input image
        mask: Optional external mask
        alpha_high_threshold: High alpha threshold
        alpha_low_threshold: Low alpha threshold
        erode_iterations: Number of erosion iterations
        min_pixels: Minimum pixels to keep
        min_coverage_ratio: Minimum coverage ratio
        
    Returns:
        Clean foreground mask or None
    """
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        h, w = arr.shape[:2]
        min_keep = max(
            int(min_pixels),
            int(float(h * w) * float(min_coverage_ratio)),
        )

        base_mask: Optional[np.ndarray] = None
        if isinstance(mask, np.ndarray) and mask.shape[:2] == (h, w):
            base_mask = np.asarray(mask).astype(bool)

        alpha_high = _extract_alpha_mask(image, threshold=int(alpha_high_threshold))
        alpha_low = _extract_alpha_mask(image, threshold=int(alpha_low_threshold))
        alpha_mask = alpha_high if isinstance(alpha_high, np.ndarray) else alpha_low

        combined = None
        if isinstance(alpha_mask, np.ndarray) and alpha_mask.shape[:2] == (h, w):
            combined = np.asarray(alpha_mask).astype(bool)
            if isinstance(base_mask, np.ndarray):
                intersect = combined & base_mask
                if int(np.sum(intersect)) >= int(min_keep):
                    combined = intersect
                elif int(np.sum(base_mask)) >= int(min_keep):
                    combined = base_mask
        elif isinstance(base_mask, np.ndarray):
            combined = base_mask
        else:
            fallback = _extract_alpha_mask(image, threshold=72)
            if not isinstance(fallback, np.ndarray):
                fallback = _extract_alpha_mask(image, threshold=24)
            if not isinstance(fallback, np.ndarray):
                fallback = _estimate_foreground_mask_from_border(image)
            if isinstance(fallback, np.ndarray) and fallback.shape[:2] == (h, w):
                combined = np.asarray(fallback).astype(bool)

        if not isinstance(combined, np.ndarray):
            return None
        if int(np.sum(combined)) < int(min_keep):
            return None

        eroded = _erode_binary_mask(combined, iterations=int(erode_iterations))
        if int(np.sum(eroded)) >= int(min_keep):
            combined = eroded

        return combined.astype(bool)
    except Exception as e:
        logger.warning(f"Clean foreground mask extraction failed: {e}")
        return None


def _trim_lab_profile_outliers(
    lab_pixels: np.ndarray,
    dark_percentile: float,
    bright_percentile: float,
    min_pixels: int = 64,
) -> np.ndarray:
    """
    Trims extreme lightness tails for robust neutral color profiling.
    
    Args:
        lab_pixels: LAB pixel array
        dark_percentile: Dark percentile threshold
        bright_percentile: Bright percentile threshold
        min_pixels: Minimum pixels to keep
        
    Returns:
        Trimmed LAB pixel array
    """
    try:
        if not isinstance(lab_pixels, np.ndarray) or lab_pixels.ndim != 2 or lab_pixels.shape[1] != 3:
            return lab_pixels
        if int(lab_pixels.shape[0]) < max(16, int(min_pixels)):
            return lab_pixels
        lo = float(max(0.0, min(100.0, dark_percentile)))
        hi = float(max(0.0, min(100.0, bright_percentile)))
        if hi <= lo:
            return lab_pixels

        l_star = lab_pixels[:, 0] * (100.0 / 255.0)
        p_low = float(np.percentile(l_star, lo))
        p_high = float(np.percentile(l_star, hi))
        keep = (l_star >= p_low) & (l_star <= p_high)
        if int(np.sum(keep)) < max(16, int(min_pixels)):
            return lab_pixels
        return lab_pixels[keep]
    except Exception:
        return lab_pixels


def resolve_garment_color_truth(
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
    color_lock_top_k: int = 3,
    semantic_override_enabled: bool = True,
    fashion_basecolour_apply_min_score: float = 0.90,
) -> Dict[str, object]:
    """
    Resolve garment color truth by reconciling pixel-based and semantic color signals.
    
    This function implements sophisticated color resolution logic that:
    - Extracts color terms from prompts and descriptors
    - Compares pixel-based colors with semantic descriptions
    - Applies semantic overrides when appropriate
    - Integrates fashion color classifier predictions
    - Normalizes and filters color hints based on LAB profiles
    
    Args:
        base_garment_prompt: Base garment description
        descriptor_raw_text: Raw descriptor text
        target_type: Target garment type
        dominant_hexes: Dominant hex colors from image
        color_hints: Color hint labels
        color_profile: LAB color profile dict
        color_mask_source: Source of color mask
        color_sampling_mask_meta: Mask metadata
        fashion_color_classifier: Fashion color classifier results
        color_lock_top_k: Number of top colors to keep
        semantic_override_enabled: Whether to enable semantic overrides
        fashion_basecolour_apply_min_score: Minimum score for fashion classifier
        
    Returns:
        Dict with resolved color information:
        - base_garment_prompt: Reconciled prompt
        - dominant_hexes: Resolved hex colors
        - color_hints: Resolved color labels
        - color_source: Source of color resolution
        - color_signal_confidence: Confidence score
        - color_signal_strength: Strength bucket
        - fashion_color_classifier: Classifier results
    """
    # NOTE: This is a complex function with many dependencies on other utility functions
    # from main.py. For now, we provide a stub that returns the input data.
    # The full implementation requires extracting many helper functions from main.py.
    
    logger.warning(
        "resolve_garment_color_truth is a stub implementation. "
        "Full implementation requires additional utility functions from main.py."
    )
    
    return {
        "base_garment_prompt": base_garment_prompt,
        "dominant_hexes": dominant_hexes or [],
        "color_hints": color_hints or [],
        "color_source": "pixel",
        "color_signal_confidence": 0.5,
        "color_signal_strength": "moderate",
        "fashion_color_classifier": fashion_color_classifier or {},
    }
