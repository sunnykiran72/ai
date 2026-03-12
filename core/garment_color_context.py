import colorsys
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ai.shared.image_ops import delta_e_cie76, rgb_to_lab


logger = logging.getLogger("glamify-ai")


ColorProfile = Dict[str, object]
PaletteEntry = Dict[str, object]


_COLOR_LABEL_RGB: List[Tuple[str, Tuple[int, int, int]]] = [
    ("black", (20, 20, 20)),
    ("charcoal", (60, 60, 60)),
    ("gray", (128, 128, 128)),
    ("silver", (185, 185, 185)),
    ("white", (245, 245, 245)),
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


def build_single_image_color_context(
    image: Image.Image,
    description: str = "",
    mask: Optional[np.ndarray] = None,
    settings: Optional[GarmentColorContextSettings] = None,
    top_k: Optional[int] = None,
) -> Dict[str, object]:
    config = settings or GarmentColorContextSettings()
    palette_top_k = max(3, int(top_k if isinstance(top_k, int) and top_k > 0 else config.palette_top_k))

    use_mask: Optional[np.ndarray] = None
    mask_source = "disabled" if config.disable_masking else "clean_foreground"
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

    palette_metrics_full = _extract_dominant_hex_colors_with_coverage(
        image=image,
        settings=config,
        mask=use_mask if isinstance(use_mask, np.ndarray) else None,
        top_k=palette_top_k,
    )
    dominant_palette_metrics = _filter_palette_entries_by_area(
        palette=palette_metrics_full,
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

    profile = _extract_lab_color_profile(
        image=image,
        settings=config,
        mask=use_mask if isinstance(use_mask, np.ndarray) else None,
    )

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
    profile_is_neutral = bool(isinstance(profile, dict) and profile.get("isNeutral"))
    high_chroma = bool(
        isinstance(profile.get("meanChroma"), (int, float))
        and float(profile.get("meanChroma")) >= 18.0
    )
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

    profile_mean_b = float(profile.get("meanB")) if isinstance(profile.get("meanB"), (int, float)) else None
    profile_median_l = float(profile.get("medianL")) if isinstance(profile.get("medianL"), (int, float)) else None
    if _profile_is_near_white(profile):
        white_label = "ivory" if profile_mean_b is not None and profile_mean_b >= 4.0 else "white"
        hints = [white_label] + [
            token for token in hints
            if token not in {"silver", "gray", "off-white", white_label}
        ]
    if (
        hints
        and (not profile_is_neutral)
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
