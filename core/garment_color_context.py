import colorsys
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from shared.image_ops import delta_e_cie76, rgb_to_lab


logger = logging.getLogger("glamify-ai")


ColorProfile = Dict[str, object]
PaletteEntry = Dict[str, object]


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


@dataclass(frozen=True)
class GarmentColorContextSettings:
    top_k: int = 3
    palette_top_k: int = 7
    palette_min_area_percent: float = 7.0
    decontamination_enabled: bool = True
    decontam_alpha_high: int = 240
    decontam_alpha_low: int = 200
    decontam_erode_iters: int = 1
    decontam_min_pixels: int = 64
    decontam_min_coverage_ratio: float = 0.006
    profile_trim_dark_percentile: float = 10.0
    profile_trim_bright_percentile: float = 98.0
    accent_min_area_percent: float = 0.35
    accent_top_k: int = 2
    disable_masking: bool = False
    cluster_merge_delta_e: float = 10.0


def _canonical_color_token(color: str) -> str:
    token = str(color or "").strip().lower()
    aliases = {
        "grey": "gray",
        "off white": "off-white",
        "offwhite": "off-white",
    }
    return aliases.get(token, token)


def _color_family(color: str) -> str:
    c = _canonical_color_token(color)
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


def _is_neutral_color_token(color: str) -> bool:
    return _color_family(color) in {"neutral_dark", "neutral_mid", "neutral_light"}


def _extract_text_color_terms(description: str, max_items: int = 4) -> List[str]:
    low = str(description or "").lower()
    if not low:
        return []
    hits: List[str] = []
    for term in _TEXT_COLOR_TERMS:
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, low) and term not in hits:
            hits.append(_canonical_color_token(term))
            if len(hits) >= max_items:
                break
    return hits


def _hex_to_rgb_triplet(token: str) -> Optional[Tuple[int, int, int]]:
    t = str(token or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", t):
        return None
    return (int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16))


def _rgb_hue_deg(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [max(0, min(255, int(v))) / 255.0 for v in rgb]
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return float(h * 360.0)


def _palette_weighted_hue_deg(
    palette: List[PaletteEntry],
    max_colors: int = 4,
) -> Optional[float]:
    vals: List[Tuple[float, float]] = []
    for entry in (palette or [])[: max(1, int(max_colors))]:
        hx = str(entry.get("hex", "")).strip()
        rgb = _hex_to_rgb_triplet(hx)
        if rgb is None:
            continue
        weight = float(entry.get("areaPercent", 0.0) or 0.0)
        vals.append((_rgb_hue_deg(rgb), max(0.1, weight)))
    if not vals:
        return None
    sin_sum = 0.0
    cos_sum = 0.0
    for deg, weight in vals:
        rad = np.deg2rad(deg)
        sin_sum += np.sin(rad) * weight
        cos_sum += np.cos(rad) * weight
    if abs(sin_sum) < 1e-6 and abs(cos_sum) < 1e-6:
        return None
    angle = float(np.rad2deg(np.arctan2(sin_sum, cos_sum)))
    if angle < 0.0:
        angle += 360.0
    return angle


def _profile_is_near_white(profile: ColorProfile) -> bool:
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


def _nearest_color_label(rgb_triplet: Tuple[int, int, int]) -> str:
    r_i, g_i, b_i = [int(v) for v in rgb_triplet]
    lab = rgb_to_lab(np.array([[r_i, g_i, b_i]], dtype=np.uint8))[0]
    l_star = float(lab[0]) * (100.0 / 255.0)
    a_star = float(lab[1]) - 128.0
    b_star = float(lab[2]) - 128.0
    chroma = float(np.sqrt((a_star * a_star) + (b_star * b_star)))

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
        if l_star < 10.0:
            return "black"
        if l_star < 28.0:
            return "charcoal"
        if l_star < 62.0:
            return "gray" if b_star < 8.0 else "tan"
        if l_star < 85.0:
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
        rgb = _hex_to_rgb_triplet(token)
        if rgb is None:
            continue
        label = _nearest_color_label(rgb)
        if label == "grey":
            label = "gray"
        if label not in out:
            out.append(label)
        if len(out) >= max(1, int(top_k)):
            break
    return out


def _erode_binary_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
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


def _extract_alpha_mask(image: Image.Image, threshold: int = 24) -> Optional[np.ndarray]:
    try:
        rgba = image.convert("RGBA")
        alpha = np.array(rgba, dtype=np.uint8)[:, :, 3]
        mask = alpha >= int(max(1, threshold))
        keep = int(np.sum(mask))
        if keep < 64:
            return None
        coverage = float(keep) / float(mask.shape[0] * mask.shape[1])
        if coverage >= 0.985:
            return None
        return mask
    except Exception:
        return None


def _estimate_foreground_mask_from_border(
    image: Image.Image,
    quant_step: int = 16,
    bg_tolerance: int = 28,
) -> Optional[np.ndarray]:
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        height, width = arr.shape[:2]
        if height < 8 or width < 8:
            return None

        border = max(1, min(height, width) // 30)
        border_pixels = np.concatenate(
            [
                arr[:border, :, :].reshape(-1, 3),
                arr[height - border :, :, :].reshape(-1, 3),
                arr[:, :border, :].reshape(-1, 3),
                arr[:, width - border :, :].reshape(-1, 3),
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
        coverage = float(keep) / float(height * width)
        if coverage < 0.02 or coverage > 0.98:
            return None
        return fg_mask
    except Exception:
        return None


def _get_clean_foreground_mask(
    image: Image.Image,
    settings: GarmentColorContextSettings,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        height, width = arr.shape[:2]
        min_keep = max(
            int(settings.decontam_min_pixels),
            int(float(height * width) * float(settings.decontam_min_coverage_ratio)),
        )

        base_mask: Optional[np.ndarray] = None
        if isinstance(mask, np.ndarray) and mask.shape[:2] == (height, width):
            base_mask = np.asarray(mask).astype(bool)

        alpha_high = _extract_alpha_mask(image, threshold=int(settings.decontam_alpha_high))
        alpha_low = _extract_alpha_mask(image, threshold=int(settings.decontam_alpha_low))
        alpha_mask = alpha_high if isinstance(alpha_high, np.ndarray) else alpha_low

        combined = None
        if isinstance(alpha_mask, np.ndarray) and alpha_mask.shape[:2] == (height, width):
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
            if isinstance(fallback, np.ndarray) and fallback.shape[:2] == (height, width):
                combined = np.asarray(fallback).astype(bool)

        if not isinstance(combined, np.ndarray):
            return None
        if int(np.sum(combined)) < int(min_keep):
            return None

        eroded = _erode_binary_mask(combined, iterations=int(settings.decontam_erode_iters))
        if int(np.sum(eroded)) >= int(min_keep):
            combined = eroded
        return combined.astype(bool)
    except Exception as err:
        logger.warning("Clean foreground mask extraction failed: %s", err)
        return None


def _extract_dominant_hex_colors_with_coverage(
    image: Image.Image,
    settings: GarmentColorContextSettings,
    mask: Optional[np.ndarray] = None,
    top_k: int = 4,
    allow_near_white: bool = False,
) -> List[PaletteEntry]:
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return []

        pixels = arr.reshape(-1, 3)
        use_mask = mask
        if settings.decontamination_enabled and not settings.disable_masking:
            clean_mask = _get_clean_foreground_mask(image, settings=settings, mask=mask)
            if isinstance(clean_mask, np.ndarray):
                use_mask = clean_mask
        if isinstance(use_mask, np.ndarray):
            keep_mask = np.asarray(use_mask).astype(bool)
            if keep_mask.shape[:2] == arr.shape[:2]:
                keep = keep_mask.reshape(-1)
                if int(np.sum(keep)) > 32:
                    pixels = pixels[keep]

        if pixels.size == 0:
            return []

        if not bool(allow_near_white):
            near_white = np.all(pixels >= 248, axis=1)
            if int(np.sum(~near_white)) > 16:
                pixels = pixels[~near_white]
        if pixels.size < 4:
            return []

        import cv2

        pixels_u8 = np.ascontiguousarray(pixels.astype(np.uint8))
        pixels_lab = cv2.cvtColor(pixels_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        if pixels_lab.size < 12:
            return []

        if settings.decontamination_enabled and int(pixels_lab.shape[0]) >= int(settings.decontam_min_pixels):
            quick_a = pixels_lab[:, 1] - 128.0
            quick_b = pixels_lab[:, 2] - 128.0
            quick_chroma = np.sqrt((quick_a * quick_a) + (quick_b * quick_b))
            if float(np.mean(quick_chroma)) < 18.0 and float(np.percentile(quick_chroma, 90)) < 28.0:
                l_star = pixels_lab[:, 0] * (100.0 / 255.0)
                low_p = float(np.percentile(l_star, max(0.0, min(45.0, settings.profile_trim_dark_percentile))))
                keep = l_star >= low_p
                if int(np.sum(keep)) >= int(settings.decontam_min_pixels):
                    pixels_lab = pixels_lab[keep]

        unique_lab_count = int(np.unique(pixels_lab, axis=0).shape[0])
        k = min(max(2, int(top_k) + 2), int(pixels_lab.shape[0]), max(1, unique_lab_count))
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
        centers_u8 = np.clip(centers, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        centers_rgb = cv2.cvtColor(centers_u8, cv2.COLOR_LAB2RGB).reshape(-1, 3)

        total_pixels = max(1, int(np.sum(counts)))
        out: List[PaletteEntry] = []
        seen_hex = set()
        for order_idx in order.tolist():
            center_idx = int(unique[order_idx])
            r, g, b = [int(v) for v in centers_rgb[center_idx].tolist()]
            hx = f"#{max(0, min(255, r)):02X}{max(0, min(255, g)):02X}{max(0, min(255, b)):02X}"
            if hx in seen_hex:
                continue
            seen_hex.add(hx)
            pixel_count = int(counts[order_idx])
            area_percent = round((float(pixel_count) / float(total_pixels)) * 100.0, 2)
            out.append({"hex": hx, "areaPercent": area_percent, "pixelCount": pixel_count})
        merged = _merge_similar_palette_entries_by_delta_e(
            out,
            merge_delta_e=float(settings.cluster_merge_delta_e),
        )
        return merged[: max(1, int(top_k))]
    except Exception as err:
        logger.warning("K-Means color extraction failed: %s", err)
        return []


def _merge_similar_palette_entries_by_delta_e(
    palette: List[PaletteEntry],
    merge_delta_e: float,
) -> List[PaletteEntry]:
    entries = list(palette or [])
    if len(entries) <= 1 or float(merge_delta_e) <= 0.0:
        return entries

    total_pixels = max(
        1,
        int(
            sum(
                max(0, int(entry.get("pixelCount", 0) or 0))
                for entry in entries
            )
        ),
    )
    candidates: List[Dict[str, object]] = []
    for entry in entries:
        hx = str(entry.get("hex", "")).strip().upper()
        rgb = _hex_to_rgb_triplet(hx)
        if rgb is None:
            continue
        pixel_count = max(1, int(entry.get("pixelCount", 0) or 0))
        lab = rgb_to_lab(np.array([rgb], dtype=np.uint8))[0].astype(np.float32)
        candidates.append(
            {
                "rgb_sum": np.array(rgb, dtype=np.float32) * float(pixel_count),
                "lab": lab,
                "pixelCount": pixel_count,
            }
        )
    if len(candidates) <= 1:
        return entries

    merged: List[Dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: int(item["pixelCount"]), reverse=True):
        matched_bucket: Optional[Dict[str, object]] = None
        for bucket in merged:
            if delta_e_cie76(candidate["lab"], bucket["lab"]) <= float(merge_delta_e):
                matched_bucket = bucket
                break
        if matched_bucket is None:
            merged.append(dict(candidate))
            continue
        matched_bucket["rgb_sum"] = np.asarray(matched_bucket["rgb_sum"], dtype=np.float32) + np.asarray(
            candidate["rgb_sum"], dtype=np.float32
        )
        matched_bucket["pixelCount"] = int(matched_bucket["pixelCount"]) + int(candidate["pixelCount"])
        mean_rgb = np.clip(
            np.asarray(matched_bucket["rgb_sum"], dtype=np.float32) / float(max(1, int(matched_bucket["pixelCount"]))),
            0,
            255,
        ).astype(np.uint8)
        matched_bucket["lab"] = rgb_to_lab(mean_rgb.reshape(1, 3))[0].astype(np.float32)

    merged_entries: List[PaletteEntry] = []
    for bucket in sorted(merged, key=lambda item: int(item["pixelCount"]), reverse=True):
        mean_rgb = np.clip(
            np.asarray(bucket["rgb_sum"], dtype=np.float32) / float(max(1, int(bucket["pixelCount"]))),
            0,
            255,
        ).astype(np.uint8)
        hx = "#{:02X}{:02X}{:02X}".format(*[int(v) for v in mean_rgb.tolist()])
        pixel_count = int(bucket["pixelCount"])
        merged_entries.append(
            {
                "hex": hx,
                "pixelCount": pixel_count,
                "areaPercent": round((float(pixel_count) / float(total_pixels)) * 100.0, 2),
            }
        )
    return merged_entries


def _filter_palette_entries_by_area(
    palette: List[PaletteEntry],
    min_area_percent: float,
    min_items: int = 2,
) -> List[PaletteEntry]:
    entries = list(palette or [])
    if not entries:
        return []
    threshold = float(max(0.0, min(100.0, min_area_percent)))
    filtered = [
        entry for entry in entries if float(entry.get("areaPercent", 0.0) or 0.0) >= threshold
    ]
    if len(filtered) >= int(max(1, min_items)):
        return filtered
    return entries[: max(1, min(int(min_items), len(entries)))]


def _hex_to_lab_l(hx: str) -> Optional[float]:
    rgb = _hex_to_rgb_triplet(hx)
    if rgb is None:
        return None
    lab = rgb_to_lab(np.array([rgb], dtype=np.uint8))[0].astype(np.float32)
    return float(lab[0]) * (100.0 / 255.0)


def _palette_primary_l(palette: List[PaletteEntry]) -> Optional[float]:
    if not palette:
        return None
    best = max(palette, key=lambda item: float(item.get("areaPercent", 0.0) or 0.0))
    hx = str(best.get("hex", "")).strip().upper()
    return _hex_to_lab_l(hx)


def _filter_palette_for_light_neutrals(
    palette: List[PaletteEntry],
    profile: ColorProfile,
) -> List[PaletteEntry]:
    if not palette or not isinstance(profile, dict):
        return palette
    if not bool(profile.get("isNeutral")):
        return palette
    median_l = profile.get("medianL")
    p90_l = profile.get("p90L")
    mean_chroma = profile.get("meanChroma")
    if not all(isinstance(v, (int, float)) for v in (median_l, p90_l, mean_chroma)):
        return palette
    if float(mean_chroma) > 16.0 or float(p90_l) < 55.0 or float(median_l) < 45.0:
        return palette

    scored: List[Tuple[PaletteEntry, float, float]] = []
    for entry in palette:
        hx = str(entry.get("hex", "")).strip().upper()
        l_star = _hex_to_lab_l(hx)
        if l_star is None:
            continue
        area = float(entry.get("areaPercent", 0.0) or 0.0)
        scored.append((entry, l_star, area))
    if len(scored) < 2:
        return palette

    l_values = [v for _entry, v, _area in scored]
    threshold = max(float(median_l), float(np.percentile(l_values, 40)))
    filtered = [entry for entry, l_star, _area in scored if l_star >= threshold]
    if len(filtered) >= 2:
        return filtered
    return palette


def _trim_lab_profile_outliers(
    lab_pixels: np.ndarray,
    dark_percentile: float,
    bright_percentile: float,
    min_pixels: int = 64,
) -> np.ndarray:
    try:
        if not isinstance(lab_pixels, np.ndarray) or lab_pixels.ndim != 2 or lab_pixels.shape[1] != 3:
            return lab_pixels
        if int(lab_pixels.shape[0]) < max(16, int(min_pixels)):
            return lab_pixels
        low = float(max(0.0, min(100.0, dark_percentile)))
        high = float(max(0.0, min(100.0, bright_percentile)))
        if high <= low:
            return lab_pixels
        l_star = lab_pixels[:, 0] * (100.0 / 255.0)
        p_low = float(np.percentile(l_star, low))
        p_high = float(np.percentile(l_star, high))
        keep = (l_star >= p_low) & (l_star <= p_high)
        if int(np.sum(keep)) < max(16, int(min_pixels)):
            return lab_pixels
        return lab_pixels[keep]
    except Exception:
        return lab_pixels


def _extract_lab_color_profile(
    image: Image.Image,
    settings: GarmentColorContextSettings,
    mask: Optional[np.ndarray] = None,
) -> ColorProfile:
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return {}

        use_mask = None
        if settings.decontamination_enabled and not settings.disable_masking:
            clean_mask = _get_clean_foreground_mask(image, settings=settings, mask=mask)
            if isinstance(clean_mask, np.ndarray):
                use_mask = clean_mask
        if not isinstance(use_mask, np.ndarray):
            use_mask = mask
        if not settings.disable_masking and not isinstance(use_mask, np.ndarray):
            use_mask = _extract_alpha_mask(image)
        if not settings.disable_masking and not isinstance(use_mask, np.ndarray):
            use_mask = _estimate_foreground_mask_from_border(image)

        pixels = arr.reshape(-1, 3)
        if isinstance(use_mask, np.ndarray) and use_mask.shape[:2] == arr.shape[:2]:
            keep = np.asarray(use_mask).astype(bool).reshape(-1)
            if int(np.sum(keep)) > 64:
                pixels = pixels[keep]
        if pixels.size == 0:
            return {}

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
        if settings.decontamination_enabled and int(lab.shape[0]) >= int(settings.decontam_min_pixels):
            quick_a = lab[:, 1] - 128.0
            quick_b = lab[:, 2] - 128.0
            quick_chroma = np.sqrt((quick_a * quick_a) + (quick_b * quick_b))
            if float(np.mean(quick_chroma)) < 18.0 and float(np.percentile(quick_chroma, 90)) < 28.0:
                trimmed = _trim_lab_profile_outliers(
                    lab,
                    dark_percentile=float(settings.profile_trim_dark_percentile),
                    bright_percentile=float(settings.profile_trim_bright_percentile),
                    min_pixels=int(settings.decontam_min_pixels),
                )
                if isinstance(trimmed, np.ndarray) and int(trimmed.shape[0]) >= int(settings.decontam_min_pixels):
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


def _apply_mask_highlight_white_balance(
    image: Image.Image,
    mask: Optional[np.ndarray],
    highlight_percentile: float = 90.0,
    min_scale: float = 0.85,
    max_scale: float = 1.35,
    target_white: Optional[float] = None,
) -> Optional[Image.Image]:
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None

        pixels = arr.reshape(-1, 3)
        if isinstance(mask, np.ndarray) and mask.shape[:2] == arr.shape[:2]:
            keep = np.asarray(mask).astype(bool).reshape(-1)
            if int(np.sum(keep)) > 64:
                pixels = pixels[keep]
        if pixels.size == 0 or int(pixels.shape[0]) < 64:
            return None

        luminance = (0.2126 * pixels[:, 0]) + (0.7152 * pixels[:, 1]) + (0.0722 * pixels[:, 2])
        threshold = float(np.percentile(luminance, max(50.0, min(99.0, highlight_percentile))))
        bright_pixels = pixels[luminance >= threshold]
        if bright_pixels.size == 0:
            bright_pixels = pixels

        mean_rgb = np.mean(bright_pixels, axis=0)
        if target_white is None:
            target = float(np.max(mean_rgb))
        else:
            target = float(max(1.0, min(255.0, target_white)))
        if target <= 1.0:
            return None

        scales = target / np.clip(mean_rgb, 1.0, None)
        scales = np.clip(scales, float(min_scale), float(max_scale))
        if float(np.max(np.abs(scales - 1.0))) < 0.04:
            return None

        balanced = np.clip(arr * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        return Image.fromarray(balanced)
    except Exception:
        return None


def _apply_mask_l_channel_clahe(
    image: Image.Image,
    mask: Optional[np.ndarray],
    clip_limit: float = 1.8,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Optional[Image.Image]:
    try:
        import cv2

        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_chan = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=float(max(1.0, clip_limit)), tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l_chan)
        if isinstance(mask, np.ndarray) and mask.shape[:2] == l_chan.shape:
            keep = np.asarray(mask).astype(bool)
            l_chan = np.where(keep, l_eq, l_chan)
        else:
            l_chan = l_eq
        lab[:, :, 0] = l_chan
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(out.astype(np.uint8))
    except Exception:
        return None


def build_single_image_color_context(
    image: Image.Image,
    description: str = "",
    mask: Optional[np.ndarray] = None,
    settings: Optional[GarmentColorContextSettings] = None,
    top_k: Optional[int] = None,
) -> Dict[str, object]:
    config = settings or GarmentColorContextSettings()
    palette_top_k = max(3, int(top_k if isinstance(top_k, int) and top_k > 0 else config.palette_top_k))

    def _select_dominant_hexes(
        palette_metrics_full: List[PaletteEntry],
        profile_for_hints: ColorProfile,
        profile_is_neutral: bool,
    ) -> Tuple[List[str], List[PaletteEntry]]:
        adjusted_palette_metrics = _filter_palette_for_light_neutrals(
            palette=palette_metrics_full,
            profile=profile_for_hints,
        )
        dominant_palette_metrics = _filter_palette_entries_by_area(
            palette=adjusted_palette_metrics,
            min_area_percent=float(config.palette_min_area_percent),
            min_items=1,
        )

        dominant_hexes: List[str] = []
        primary_area = float(dominant_palette_metrics[0].get("areaPercent", 0.0) or 0.0) if dominant_palette_metrics else 0.0
        dominant_area_floor = max(float(config.palette_min_area_percent), primary_area * 0.20)
        for idx, entry in enumerate(dominant_palette_metrics):
            hx = str(entry.get("hex", "")).strip().upper()
            if not re.fullmatch(r"#[0-9A-F]{6}", hx):
                continue
            area = float(entry.get("areaPercent", 0.0) or 0.0)
            if idx == 0 or area >= dominant_area_floor:
                dominant_hexes.append(hx)

        if profile_is_neutral:
            median_l = profile_for_hints.get("medianL")
            p90_l = profile_for_hints.get("p90L")
            mean_chroma = profile_for_hints.get("meanChroma")
            if (
                isinstance(median_l, (int, float))
                and isinstance(p90_l, (int, float))
                and isinstance(mean_chroma, (int, float))
                and float(mean_chroma) <= 16.0
                and float(p90_l) >= 55.0
                and float(median_l) >= 45.0
            ):
                bright: List[Tuple[float, float, str]] = []
                for entry in palette_metrics_full:
                    hx = str(entry.get("hex", "")).strip().upper()
                    l_star = _hex_to_lab_l(hx)
                    if l_star is None:
                        continue
                    area = float(entry.get("areaPercent", 0.0) or 0.0)
                    bright.append((float(l_star), area, hx))
                bright.sort(key=lambda item: (item[0], item[1]), reverse=True)
                bright_hexes: List[str] = []
                for l_star, area, hx in bright:
                    if area < 2.0 and bright_hexes:
                        continue
                    if hx not in bright_hexes:
                        bright_hexes.append(hx)
                    if len(bright_hexes) >= max(1, int(config.top_k)):
                        break
                if bright_hexes:
                    dominant_hexes = bright_hexes

        return dominant_hexes, dominant_palette_metrics

    use_mask: Optional[np.ndarray] = None
    mask_source = "disabled" if config.disable_masking else "clean_foreground"
    allow_near_white = bool(isinstance(mask, np.ndarray))
    if not config.disable_masking:
        use_mask = _get_clean_foreground_mask(image, settings=config, mask=mask)
        if not isinstance(use_mask, np.ndarray):
            use_mask = _extract_alpha_mask(image, threshold=72)
            mask_source = "alpha72"
        if not isinstance(use_mask, np.ndarray):
            use_mask = _extract_alpha_mask(image, threshold=24)
            mask_source = "alpha24"
        if not isinstance(use_mask, np.ndarray):
            use_mask = _estimate_foreground_mask_from_border(image)
            mask_source = "border_estimate"
        if not isinstance(use_mask, np.ndarray):
            mask_source = "none"
    if allow_near_white and mask_source == "clean_foreground":
        mask_source = "provided_mask"

    raw_palette_metrics = _extract_dominant_hex_colors_with_coverage(
        image=image,
        settings=config,
        mask=use_mask if isinstance(use_mask, np.ndarray) else None,
        top_k=palette_top_k,
        allow_near_white=allow_near_white,
    )
    palette_metrics_full = list(raw_palette_metrics)

    profile = _extract_lab_color_profile(
        image=image,
        settings=config,
        mask=use_mask if isinstance(use_mask, np.ndarray) else None,
    )
    profile_is_neutral = bool(isinstance(profile, dict) and profile.get("isNeutral"))
    balance_max_scale = 1.12
    balance_min_scale = 0.90
    balance_target_white: Optional[float] = None
    apply_preprocess = False
    apply_clahe = False
    if isinstance(profile, dict):
        mean_chroma = profile.get("meanChroma")
        p90_chroma = profile.get("p90Chroma")
        median_l = profile.get("medianL")
        mean_a = profile.get("meanA")
        mean_b = profile.get("meanB")
        cast_strength = None
        if isinstance(mean_a, (int, float)) and isinstance(mean_b, (int, float)):
            cast_strength = float(np.sqrt((float(mean_a) ** 2) + (float(mean_b) ** 2)))
        if profile_is_neutral:
            apply_preprocess = True
            mean_b = profile.get("meanB")
            if isinstance(mean_b, (int, float)) and float(mean_b) >= 6.0:
                balance_max_scale = 1.75
                balance_min_scale = 0.85
                balance_target_white = 235.0
            else:
                balance_max_scale = 1.35
                balance_min_scale = 0.90
        else:
            if (
                isinstance(mean_chroma, (int, float))
                and isinstance(p90_chroma, (int, float))
                and float(mean_chroma) <= 24.0
                and float(p90_chroma) <= 38.0
                and isinstance(cast_strength, (int, float))
                and float(cast_strength) >= 6.0
            ):
                apply_preprocess = True
        if apply_preprocess and isinstance(median_l, (int, float)) and float(median_l) <= 62.0:
            apply_clahe = True

    cast_corrected_profile: Optional[Dict[str, object]] = None
    cast_corrected_hints: List[str] = []
    cast_corrected_palette: Optional[List[PaletteEntry]] = None
    use_corrected_palette = False
    if apply_preprocess:
        corrected_image = _apply_mask_highlight_white_balance(
            image=image,
            mask=use_mask if isinstance(use_mask, np.ndarray) else None,
            min_scale=balance_min_scale,
            max_scale=balance_max_scale,
            target_white=balance_target_white,
        )
        if isinstance(corrected_image, Image.Image) and apply_clahe:
            clahe_image = _apply_mask_l_channel_clahe(
                corrected_image,
                mask=use_mask if isinstance(use_mask, np.ndarray) else None,
                clip_limit=1.6,
                tile_grid_size=(8, 8),
            )
            if isinstance(clahe_image, Image.Image):
                corrected_image = clahe_image
        if isinstance(corrected_image, Image.Image):
            cast_corrected_profile = _extract_lab_color_profile(
                image=corrected_image,
                settings=config,
                mask=use_mask if isinstance(use_mask, np.ndarray) else None,
            )
            corrected_palette = _extract_dominant_hex_colors_with_coverage(
                image=corrected_image,
                settings=config,
                mask=use_mask if isinstance(use_mask, np.ndarray) else None,
                top_k=palette_top_k,
                allow_near_white=allow_near_white,
            )
            cast_corrected_palette = list(corrected_palette or [])
            cast_corrected_hints = _color_labels_from_hex_palette(
                [str(entry.get("hex", "")).strip().upper() for entry in corrected_palette],
                top_k=config.top_k,
            )
            if cast_corrected_profile:
                profile = dict(profile or {})
                profile["castCorrected"] = cast_corrected_profile
                if cast_corrected_hints:
                    profile["castCorrectedHints"] = list(cast_corrected_hints)

    if cast_corrected_palette:
        raw_primary_l = _palette_primary_l(raw_palette_metrics)
        corrected_primary_l = _palette_primary_l(cast_corrected_palette)
        raw_mean_chroma = profile.get("meanChroma") if isinstance(profile, dict) else None
        corrected_mean_chroma = (
            cast_corrected_profile.get("meanChroma") if isinstance(cast_corrected_profile, dict) else None
        )
        chroma_ok = True
        if isinstance(raw_mean_chroma, (int, float)) and isinstance(corrected_mean_chroma, (int, float)):
            if float(raw_mean_chroma) > 0.1:
                ratio = float(corrected_mean_chroma) / float(raw_mean_chroma)
                if ratio > 1.45 or ratio < 0.55:
                    chroma_ok = False
        min_delta_l = 6.0 if profile_is_neutral else 3.0
        if corrected_primary_l is not None and chroma_ok and (
            raw_primary_l is None or (corrected_primary_l - raw_primary_l) >= float(min_delta_l)
        ):
            use_corrected_palette = True

    if use_corrected_palette and cast_corrected_palette:
        palette_metrics_full = list(cast_corrected_palette)
        if isinstance(profile, dict):
            profile["colorCorrectionApplied"] = True
            profile["colorCorrectionSource"] = "mask_preprocess_palette"

    profile_for_hints = (
        cast_corrected_profile
        if (use_corrected_palette and isinstance(cast_corrected_profile, dict))
        else (profile if isinstance(profile, dict) else {})
    )

    dominant_hexes, dominant_palette_metrics = _select_dominant_hexes(
        palette_metrics_full=palette_metrics_full,
        profile_for_hints=profile_for_hints,
        profile_is_neutral=profile_is_neutral,
    )

    if cast_corrected_palette:
        corrected_bright: List[Tuple[float, float, str]] = []
        for entry in cast_corrected_palette:
            hx = str(entry.get("hex", "")).strip().upper()
            l_star = _hex_to_lab_l(hx)
            if l_star is None:
                continue
            area = float(entry.get("areaPercent", 0.0) or 0.0)
            corrected_bright.append((float(l_star), area, hx))
        corrected_bright.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if corrected_bright:
            current_l = _hex_to_lab_l(dominant_hexes[0]) if dominant_hexes else None
            corrected_l = corrected_bright[0][0]
            if current_l is None or (corrected_l - float(current_l)) >= 6.0:
                corrected_hexes: List[str] = []
                for l_star, area, hx in corrected_bright:
                    if area < 1.5 and corrected_hexes:
                        continue
                    if hx not in corrected_hexes:
                        corrected_hexes.append(hx)
                    if len(corrected_hexes) >= max(1, int(config.top_k)):
                        break
                if corrected_hexes:
                    dominant_hexes = corrected_hexes
                    use_corrected_palette = True
                    palette_metrics_full = list(cast_corrected_palette)
                    if isinstance(profile, dict):
                        profile["colorCorrectionApplied"] = True
                        profile["colorCorrectionSource"] = "mask_preprocess_dominant"
                    if isinstance(cast_corrected_profile, dict):
                        profile_for_hints = cast_corrected_profile
                    dominant_hexes, dominant_palette_metrics = _select_dominant_hexes(
                        palette_metrics_full=palette_metrics_full,
                        profile_for_hints=profile_for_hints,
                        profile_is_neutral=profile_is_neutral,
                    )

    palette_hexes = list(dominant_hexes)
    accent_hexes: List[str] = []
    for entry in palette_metrics_full:
        hx = str(entry.get("hex", "")).strip().upper()
        if not re.fullmatch(r"#[0-9A-F]{6}", hx):
            continue
        if hx in dominant_hexes:
            continue
        area = float(entry.get("areaPercent", 0.0) or 0.0)
        if area < float(config.accent_min_area_percent):
            continue
        accent_hexes.append(hx)
        if len(accent_hexes) >= max(1, int(config.accent_top_k)):
            break

    image_colors = _color_labels_from_hex_palette(dominant_hexes, top_k=config.top_k)
    if not image_colors:
        image_colors = _color_labels_from_hex_palette(
            [str(entry.get("hex", "")).strip().upper() for entry in palette_metrics_full],
            top_k=config.top_k,
        )
    palette_label_entries: List[Tuple[str, float]] = []
    for entry in palette_metrics_full:
        hx = str(entry.get("hex", "")).strip().upper()
        rgb = _hex_to_rgb_triplet(hx)
        if rgb is None:
            continue
        palette_label_entries.append(
            (_nearest_color_label(rgb), float(entry.get("areaPercent", 0.0) or 0.0))
        )

    text_colors = [
        _canonical_color_token(c)
        for c in _extract_text_color_terms(description, max_items=config.top_k)
    ]
    hints = list(image_colors)
    high_chroma = bool(
        isinstance(profile_for_hints.get("meanChroma"), (int, float))
        and float(profile_for_hints.get("meanChroma")) >= 18.0
    )
    profile_is_neutral_for_hints = bool(
        (isinstance(profile_for_hints, dict) and profile_for_hints.get("isNeutral"))
        or profile_is_neutral
    )
    if profile_is_neutral_for_hints:
        hints = [token for token in hints if _is_neutral_color_token(token)]
    non_neutral = [color for color in hints if not _is_neutral_color_token(color)]
    if high_chroma or (non_neutral and not profile_is_neutral):
        dedup_non_neutral: List[str] = []
        for token in non_neutral:
            if token not in dedup_non_neutral:
                dedup_non_neutral.append(token)
        if dedup_non_neutral:
            hints = dedup_non_neutral
    for token in text_colors:
        if high_chroma and _is_neutral_color_token(token):
            continue
        if profile_is_neutral and not _is_neutral_color_token(token):
            continue
        if token not in hints:
            hints.append(token)

    profile_mean_b = float(profile_for_hints.get("meanB")) if isinstance(profile_for_hints.get("meanB"), (int, float)) else None
    profile_median_l = float(profile_for_hints.get("medianL")) if isinstance(profile_for_hints.get("medianL"), (int, float)) else None
    profile_mean_chroma = float(profile_for_hints.get("meanChroma")) if isinstance(profile_for_hints.get("meanChroma"), (int, float)) else None
    profile_p90_l = float(profile_for_hints.get("p90L")) if isinstance(profile_for_hints.get("p90L"), (int, float)) else None
    if profile_is_neutral_for_hints and profile_mean_chroma is not None and profile_p90_l is not None:
        brightest = None
        for entry in palette_metrics_full:
            hx = str(entry.get("hex", "")).strip().upper()
            l_star = _hex_to_lab_l(hx)
            if l_star is None:
                continue
            area = float(entry.get("areaPercent", 0.0) or 0.0)
            if brightest is None or float(l_star) > brightest[0]:
                brightest = (float(l_star), area, hx)
        if (
            brightest
            and float(profile_mean_chroma) <= 18.0
            and float(profile_p90_l) >= 70.0
            and float(brightest[0]) >= 78.0
            and float(brightest[1]) >= 10.0
        ):
            white_label = "ivory" if profile_mean_b is not None and profile_mean_b >= 8.0 else "white"
            alt_label = "white" if white_label == "ivory" else "ivory"
            hints = [white_label, alt_label]
    if _profile_is_near_white(profile_for_hints):
        white_label = "ivory" if profile_mean_b is not None and profile_mean_b >= 4.0 else "white"
        hints = [white_label] + [
            token for token in hints
            if token not in {"silver", "gray", "off-white", white_label}
        ]
    if (
        hints
        and (not profile_is_neutral_for_hints)
        and all(_is_neutral_color_token(token) for token in hints)
        and profile_mean_b is not None
        and profile_median_l is not None
    ):
        warm_neutral_hints: List[str] = []
        if profile_mean_b >= 14.0:
            warm_neutral_hints = ["beige", "tan"] if profile_median_l >= 72.0 else ["tan", "beige"]
        elif profile_mean_b >= 8.0 and profile_median_l < 90.0:
            warm_neutral_hints = ["beige", "champagne"]
        if warm_neutral_hints:
            hints = warm_neutral_hints + [token for token in hints if token not in warm_neutral_hints]

    # Preserve high-contrast dark accents for patterned garments such as cream/black knits.
    if hints and any(_color_family(token) == "neutral_light" for token in hints):
        accent_candidate = next(
            (
                label
                for label, area in palette_label_entries
                if label in {"black", "charcoal", "navy"}
                and area >= 6.0
                and label not in hints
            ),
            None,
        )
        if accent_candidate:
            max_hint_items = max(2, int(config.top_k))
            if len(hints) >= max_hint_items:
                hints = hints[: max_hint_items - 1]
            hints.append(accent_candidate)

    hints = hints[: max(2, int(config.top_k))]

    accent_labels = _color_labels_from_hex_palette(accent_hexes, top_k=max(1, int(config.accent_top_k)))

    return {
        "maskSource": mask_source,
        "maskPixels": int(np.sum(use_mask)) if isinstance(use_mask, np.ndarray) else 0,
        "paletteMetrics": dominant_palette_metrics,
        "paletteMetricsFull": palette_metrics_full,
        "paletteHexes": palette_hexes,
        "dominantHexes": palette_hexes,
        "accentHexes": accent_hexes,
        "profile": profile if isinstance(profile, dict) else {},
        "hints": hints,
        "colorHints": hints,
        "accentHints": accent_labels,
    }


def build_visual_lock_clauses(
    product_images: List[Image.Image],
    garment_descriptions: List[str],
    settings: Optional[GarmentColorContextSettings] = None,
    transparency_signal_fn: Optional[Callable[[str], bool]] = None,
) -> Dict[str, object]:
    config = settings or GarmentColorContextSettings()
    contexts = [
        build_single_image_color_context(
            image=img,
            description=garment_descriptions[idx] if idx < len(garment_descriptions) else "",
            settings=config,
            top_k=max(3, int(config.palette_top_k)),
        )
        for idx, img in enumerate(product_images)
    ]
    color_palette_metrics = [list(ctx.get("paletteMetricsFull") or []) for ctx in contexts]
    color_profiles = [
        ctx.get("profile") if isinstance(ctx.get("profile"), dict) else {}
        for ctx in contexts
    ]
    color_palettes = [
        [str(v) for v in (ctx.get("dominantHexes") or []) if str(v).strip()]
        for ctx in contexts
    ]
    color_hints = [
        [str(v) for v in (ctx.get("colorHints") or []) if str(v).strip()]
        for ctx in contexts
    ]
    accent_hints = [
        [str(v) for v in (ctx.get("accentHints") or []) if str(v).strip()]
        for ctx in contexts
    ]

    color_clause = ""
    non_empty_color_hints = [colors for colors in color_hints if colors]
    if non_empty_color_hints:
        if len(non_empty_color_hints) == 1:
            palette = ", ".join(non_empty_color_hints[0])
            color_clause = (
                f"Color lock from image 2: keep the garment in {palette} tones only. "
                "Do not recolor, hue-shift, or replace with a different color family. "
            )
            if accent_hints and accent_hints[0]:
                color_clause += (
                    "Accent-color lock from image 2: preserve trim, border, button, or piping colors as "
                    + ", ".join(accent_hints[0])
                    + ". "
                )
        else:
            per_item = []
            for idx, colors in enumerate(color_hints, start=1):
                if colors:
                    suffix = ""
                    item_accents = accent_hints[idx - 1] if idx - 1 < len(accent_hints) else []
                    if item_accents:
                        suffix = f" (accents: {', '.join(item_accents)})"
                    per_item.append(f"item {idx}: {', '.join(colors)}{suffix}")
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

    transparency_lock = bool(
        transparency_signal_fn
        and any(transparency_signal_fn(desc) for desc in garment_descriptions)
    )
    transparency_clause = ""
    if transparency_lock:
        transparency_clause = (
            "Transparency lock from image 2: preserve sheer/mesh panel transparency and translucency. "
            "Keep subtle skin visibility only where the source garment is transparent. "
            "Do not make sheer panels opaque or overexpose skin beyond those panel regions. "
        )

    return {
        "color_clause": color_clause,
        "detail_clause": "",
        "transparency_clause": transparency_clause,
        "transparency_lock": transparency_lock,
        "neutral_tone_lock": bool(neutral_lock_items),
        "color_profiles": color_profiles,
        "color_hints": color_hints,
        "color_palettes": color_palettes,
        "color_palette_metrics": color_palette_metrics,
        "accent_hints": accent_hints,
        "contexts": contexts,
        "detail_terms": [],
    }
