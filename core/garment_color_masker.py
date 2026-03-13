from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ai.shared.image_ops import binary_close, binary_open


MaskFn = Callable[[Image.Image], Optional[np.ndarray]]
SkinMaskFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class GarmentColorMaskerSettings:
    min_pixels: int = 96
    min_area_ratio: float = 0.02
    parser_min_area_ratio: float = 0.02
    parser_strict_min_area_ratio: float = 0.01
    component_min_area_ratio: float = 0.004
    keep_score_ratio: float = 0.42
    keep_area_ratio: float = 0.12
    bottom_skin_upper_ratio: float = 0.42
    bottom_skin_side_ratio: float = 0.10
    top_skin_upper_ratio: float = 0.24
    top_skin_side_ratio: float = 0.08
    outer_skin_upper_ratio: float = 0.20
    outer_skin_side_ratio: float = 0.06
    dress_skin_upper_ratio: float = 0.16
    dress_skin_side_ratio: float = 0.06
    bottom_target_y: float = 0.73
    top_target_y: float = 0.34
    outer_target_y: float = 0.40
    dress_target_y: float = 0.52
    parser_skin_max_strip_ratio: float = 0.38
    parser_skin_min_remaining_ratio: float = 0.32
    parser_bottom_refine_min_overlap_ratio: float = 0.55
    parser_bottom_refine_min_heuristic_overlap_ratio: float = 0.18


class GarmentColorMasker:
    """
    Type-aware garment color mask estimation.

    Primary path:
    - fashion parser / SegFormer mask if available

    Fallback path:
    - foreground mask + type-aware component scoring

    This component is intentionally standalone so it can be unit-tested and
    swapped without changing analyze / try-on call sites.
    """

    def __init__(
        self,
        parser=None,
        base_mask_fn: Optional[MaskFn] = None,
        skin_mask_fn: Optional[SkinMaskFn] = None,
        settings: Optional[GarmentColorMaskerSettings] = None,
    ):
        self.parser = parser
        self.base_mask_fn = base_mask_fn
        self.skin_mask_fn = skin_mask_fn
        self.settings = settings or GarmentColorMaskerSettings()

    def estimate_mask(
        self,
        image: Image.Image,
        garment_type: str,
        description: str = "",
    ) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
        parser_mask, parser_meta = self._mask_from_parser(image, garment_type)
        if isinstance(parser_mask, np.ndarray):
            parser_mask, parser_meta = self._refine_parser_primary_mask(
                image=image,
                garment_type=garment_type,
                description=description,
                parser_mask=parser_mask,
                parser_meta=parser_meta,
            )
            return parser_mask.astype(bool), parser_meta

        heuristic_mask, heuristic_meta = self._mask_from_heuristic(image, garment_type, description=description)
        return heuristic_mask, heuristic_meta

    def _refine_parser_primary_mask(
        self,
        *,
        image: Image.Image,
        garment_type: str,
        description: str,
        parser_mask: np.ndarray,
        parser_meta: Dict[str, object],
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        normalized_type = str(garment_type or "").strip().lower()
        parser_variant = str((parser_meta or {}).get("parser_variant") or "").strip().lower()
        if normalized_type != "bottom" or parser_variant != "bottom_spatial_rescue":
            return parser_mask, parser_meta

        heuristic_mask, heuristic_meta = self._mask_from_heuristic(image, garment_type, description=description)
        if not isinstance(heuristic_mask, np.ndarray):
            return parser_mask, parser_meta

        parser_arr = np.asarray(parser_mask).astype(bool)
        heuristic_arr = np.asarray(heuristic_mask).astype(bool)
        intersection = self._cleanup_mask(parser_arr & heuristic_arr, keep_largest=True)
        parser_pixels = int(np.sum(parser_arr))
        heuristic_pixels = int(np.sum(heuristic_arr))
        intersection_pixels = int(np.sum(intersection))
        if intersection_pixels <= 0:
            return parser_mask, parser_meta

        parser_overlap = float(intersection_pixels) / float(max(1, parser_pixels))
        heuristic_overlap = float(intersection_pixels) / float(max(1, heuristic_pixels))
        if not self._mask_shape_ok(
            intersection,
            image.size,
            garment_type,
            min_area_ratio=float(self.settings.parser_strict_min_area_ratio),
        ):
            return parser_mask, parser_meta
        if parser_overlap < float(self.settings.parser_bottom_refine_min_overlap_ratio):
            return parser_mask, parser_meta
        if heuristic_overlap < float(self.settings.parser_bottom_refine_min_heuristic_overlap_ratio):
            return parser_mask, parser_meta

        refined_meta = dict(parser_meta or {})
        refined_meta["source"] = "parser_bottom_refined_runtime"
        refined_meta["reason"] = "parser_primary_refined_by_heuristic_overlap"
        refined_meta["heuristic_mask_pixels"] = heuristic_pixels
        refined_meta["parser_overlap_ratio"] = round(parser_overlap, 4)
        refined_meta["heuristic_overlap_ratio"] = round(heuristic_overlap, 4)
        refined_meta["mask_pixels"] = intersection_pixels
        refined_meta["area_ratio"] = round(float(np.mean(intersection)), 6)
        return intersection, refined_meta

    def _mask_from_parser(
        self,
        image: Image.Image,
        garment_type: str,
    ) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
        if self.parser is None:
            return None, {"source": "parser", "used": False, "reason": "parser_unavailable"}
        try:
            parsing = self.parser.parse(image)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            candidates = self._parser_mask_candidates(parsing, garment_type)
            candidate_debug = []
            for mask, mask_source, min_area_ratio, variant in candidates:
                if mask is None:
                    continue
                mask_arr = np.asarray(mask).astype(bool)
                if int(np.sum(mask_arr)) <= 0:
                    continue
                candidate_mask, skin_pixels = self._strip_skin_from_parser_mask(
                    mask=mask_arr,
                    rgb=rgb,
                    garment_type=garment_type,
                    image_size=image.size,
                    min_area_ratio=min_area_ratio,
                )
                cleaned = self._cleanup_mask(candidate_mask, keep_largest=True)
                if self._mask_shape_ok(cleaned, image.size, garment_type, min_area_ratio=min_area_ratio):
                    return cleaned, {
                        "source": mask_source,
                        "used": True,
                        "mask_pixels": int(np.sum(cleaned)),
                        "area_ratio": round(float(np.mean(cleaned)), 6),
                        "skin_pixels_removed": skin_pixels,
                        "parser_variant": variant,
                    }
                candidate_debug.append(
                    {
                        "source": mask_source,
                        "variant": variant,
                        "mask_pixels": int(np.sum(cleaned)),
                        "area_ratio": round(float(np.mean(cleaned)), 6),
                    }
                )
            best = max(candidate_debug, key=lambda item: item.get("mask_pixels", 0)) if candidate_debug else {}
            return None, {
                "source": str(best.get("source") or "parser"),
                "used": False,
                "reason": "mask_quality_low",
                "mask_pixels": int(best.get("mask_pixels", 0) or 0),
                "area_ratio": round(float(best.get("area_ratio", 0.0) or 0.0), 6),
                "parser_candidates": candidate_debug[:3],
            }
        except Exception as err:
            return None, {"source": "parser", "used": False, "reason": f"error:{err}"}

    def _strict_mask_from_parser(
        self,
        parsing: np.ndarray,
        garment_type: str,
    ) -> Tuple[Optional[np.ndarray], str]:
        runtime_labels = self._runtime_labels()
        if not runtime_labels:
            return None, "parser"
        strict_mask, _relaxed_mask = self._build_type_specific_parser_masks(parsing, garment_type, runtime_labels)
        if strict_mask is None:
            return None, "parser"
        return strict_mask, "parser_strict_runtime"

    def _runtime_labels(self) -> Dict[str, int]:
        runtime_labels = {}
        runtime_fn = getattr(self.parser, "_runtime_labels", None)
        if callable(runtime_fn):
            try:
                runtime_labels = dict(runtime_fn() or {})
            except Exception:
                runtime_labels = {}
        return runtime_labels

    @staticmethod
    def _normalize_label_token(name: str) -> str:
        return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _ids_for_aliases(self, runtime_labels: Dict[str, int], aliases: Tuple[str, ...]) -> list[int]:
        ids = []
        for alias in aliases:
            key = self._normalize_label_token(alias)
            if key not in runtime_labels:
                continue
            try:
                ids.append(int(runtime_labels[key]))
            except Exception:
                continue
        return sorted(set(ids))

    def _mask_for_aliases(
        self,
        parsing: np.ndarray,
        runtime_labels: Dict[str, int],
        aliases: Tuple[str, ...],
    ) -> np.ndarray:
        ids = self._ids_for_aliases(runtime_labels, aliases)
        if not ids:
            return np.zeros_like(parsing, dtype=bool)
        return np.isin(parsing, ids)

    def _build_type_specific_parser_masks(
        self,
        parsing: np.ndarray,
        garment_type: str,
        runtime_labels: Dict[str, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        normalized_type = str(garment_type or "").strip().lower()
        body_aliases = (
            "background",
            "hat",
            "hair",
            "sunglasses",
            "glasses",
            "face",
            "torso",
            "arms",
            "hands",
            "legs",
            "feet",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
            "left_shoe",
            "right_shoe",
            "bag",
            "scarf",
            "belt",
        )
        configs = {
            "top": {
                "strict_keep": ("top", "upper", "upper_clothes"),
                "relaxed_keep": ("top", "upper", "upper_clothes", "scarf"),
                "kill": body_aliases + ("bottom", "pants", "trousers", "skirt", "shorts", "dress", "outer", "outerwear", "coat", "jacket", "blazer"),
            },
            "outer": {
                "strict_keep": ("outer", "outerwear", "coat", "jacket", "blazer", "hoodie"),
                "relaxed_keep": ("outer", "outerwear", "coat", "jacket", "blazer", "hoodie", "top", "upper", "upper_clothes"),
                "kill": body_aliases + ("bottom", "pants", "trousers", "skirt", "shorts", "dress"),
            },
            "bottom": {
                "strict_keep": ("bottom", "pants", "trousers", "skirt", "shorts"),
                "relaxed_keep": ("bottom", "pants", "trousers", "skirt", "shorts"),
                "kill": body_aliases + ("top", "upper", "upper_clothes", "outer", "outerwear", "coat", "jacket", "blazer", "dress"),
            },
            "dress": {
                "strict_keep": ("dress",),
                "relaxed_keep": ("dress", "top", "upper", "upper_clothes", "skirt"),
                "kill": body_aliases + ("outer", "outerwear", "coat", "jacket", "blazer", "pants", "trousers", "shorts"),
            },
        }
        cfg = configs.get(normalized_type)
        if not cfg:
            return None, None
        strict_keep = self._mask_for_aliases(parsing, runtime_labels, cfg["strict_keep"])
        relaxed_keep = self._mask_for_aliases(parsing, runtime_labels, cfg["relaxed_keep"])
        kill_mask = self._mask_for_aliases(parsing, runtime_labels, cfg["kill"])
        strict_mask = strict_keep & (~kill_mask)
        relaxed_mask = relaxed_keep & (~kill_mask)
        if int(np.sum(strict_mask)) <= 0:
            strict_mask = None
        if int(np.sum(relaxed_mask)) <= 0:
            relaxed_mask = None
        return strict_mask, relaxed_mask

    def _build_bottom_spatial_rescue_mask(
        self,
        parsing: np.ndarray,
        runtime_labels: Dict[str, int],
    ) -> Optional[np.ndarray]:
        h, w = parsing.shape[:2]
        if h <= 0 or w <= 0:
            return None
        garment_aliases = (
            "top",
            "upper",
            "upper_clothes",
            "outer",
            "outerwear",
            "coat",
            "jacket",
            "blazer",
            "dress",
            "bottom",
            "pants",
            "trousers",
            "skirt",
            "shorts",
        )
        kill_aliases = (
            "background",
            "hat",
            "hair",
            "sunglasses",
            "glasses",
            "face",
            "torso",
            "arms",
            "hands",
            "legs",
            "feet",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
            "left_shoe",
            "right_shoe",
            "bag",
            "scarf",
        )
        garment_mask = self._mask_for_aliases(parsing, runtime_labels, garment_aliases)
        kill_mask = self._mask_for_aliases(parsing, runtime_labels, kill_aliases)
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        lower_region = y >= int(0.38 * h)
        center_region = (x >= int(0.06 * w)) & (x <= int(0.94 * w))
        rescue_mask = garment_mask & (~kill_mask) & lower_region & center_region
        if int(np.sum(rescue_mask)) <= 0:
            return None
        return rescue_mask

    def _parser_mask_candidates(
        self,
        parsing: np.ndarray,
        garment_type: str,
    ) -> list[Tuple[Optional[np.ndarray], str, float, str]]:
        runtime_labels = self._runtime_labels()
        candidates: list[Tuple[Optional[np.ndarray], str, float, str]] = []
        if runtime_labels:
            strict_mask, relaxed_mask = self._build_type_specific_parser_masks(parsing, garment_type, runtime_labels)
            candidates.append(
                (
                    strict_mask,
                    "parser_strict_runtime",
                    float(self.settings.parser_strict_min_area_ratio),
                    "strict_type_specific",
                )
            )
            candidates.append(
                (
                    relaxed_mask,
                    "parser_relaxed_runtime",
                    float(self.settings.parser_min_area_ratio),
                    "relaxed_type_specific",
                )
            )
            if str(garment_type or "").strip().lower() == "bottom":
                candidates.append(
                    (
                        self._build_bottom_spatial_rescue_mask(parsing, runtime_labels),
                        "parser_bottom_spatial_runtime",
                        float(self.settings.parser_strict_min_area_ratio),
                        "bottom_spatial_rescue",
                    )
                )
        try:
            category_mask = self.parser.get_mask_for_category(parsing, garment_type)
        except Exception:
            category_mask = None
        candidates.append(
            (
                np.asarray(category_mask).astype(bool) if isinstance(category_mask, np.ndarray) else None,
                "parser_category",
                float(self.settings.parser_min_area_ratio),
                "category_fallback",
            )
        )
        return candidates

    def _strip_skin_from_parser_mask(
        self,
        mask: np.ndarray,
        rgb: np.ndarray,
        garment_type: str,
        image_size: Tuple[int, int],
        min_area_ratio: float,
    ) -> Tuple[np.ndarray, int]:
        skin_pixels = 0
        candidate_mask = np.asarray(mask).astype(bool)
        original_pixels = int(np.sum(candidate_mask))
        if not callable(self.skin_mask_fn):
            return candidate_mask, skin_pixels
        try:
            skin = np.asarray(self.skin_mask_fn(rgb)).astype(bool)
            skin = self._limit_skin_mask(skin=skin, garment_type=garment_type, shape=candidate_mask.shape[:2])
            masked_skin = skin & candidate_mask
            masked_skin_ratio = float(np.sum(masked_skin)) / float(max(1, np.sum(candidate_mask)))
            if int(np.sum(masked_skin)) > 0:
                stripped = candidate_mask & (~masked_skin)
                stripped_cleaned = self._cleanup_mask(stripped, keep_largest=True)
                retained_ratio = float(np.sum(stripped_cleaned)) / float(max(1, original_pixels))
                if self._mask_shape_ok(
                    stripped_cleaned,
                    image_size,
                    garment_type,
                    min_area_ratio=min_area_ratio,
                ) and retained_ratio >= float(self.settings.parser_skin_min_remaining_ratio):
                    candidate_mask = stripped
                    skin_pixels = int(np.sum(masked_skin))
                elif masked_skin_ratio <= float(self.settings.parser_skin_max_strip_ratio):
                    candidate_mask = stripped
                    skin_pixels = int(np.sum(masked_skin))
        except Exception:
            skin_pixels = 0
        return candidate_mask, skin_pixels

    def _mask_from_heuristic(
        self,
        image: Image.Image,
        garment_type: str,
        description: str = "",
    ) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
        try:
            import cv2
        except Exception as err:
            return None, {"source": "heuristic", "used": False, "reason": f"cv2_unavailable:{err}"}

        if self.base_mask_fn is None:
            return None, {"source": "heuristic", "used": False, "reason": "base_mask_unavailable"}

        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None, {"source": "heuristic", "used": False, "reason": "bad_image"}
        h, w = arr.shape[:2]

        base = self.base_mask_fn(rgb)
        if not isinstance(base, np.ndarray):
            return None, {"source": "heuristic", "used": False, "reason": "base_mask_empty"}
        base = np.asarray(base).astype(bool)
        if int(np.sum(base)) < max(self.settings.min_pixels, int(self.settings.min_area_ratio * h * w)):
            return None, {"source": "heuristic", "used": False, "reason": "base_mask_too_small"}

        skin = np.zeros((h, w), dtype=bool)
        if callable(self.skin_mask_fn):
            try:
                skin = np.asarray(self.skin_mask_fn(arr)).astype(bool)
            except Exception:
                skin = np.zeros((h, w), dtype=bool)

        skin = self._limit_skin_mask(skin=skin, garment_type=garment_type, shape=(h, w))
        working = base & (~skin)
        if int(np.sum(working)) < max(self.settings.min_pixels, int(0.06 * np.sum(base))):
            working = base

        if garment_type == "bottom":
            # Heuristic skin suppression for bottoms when parser is unavailable.
            # This keeps leg/skin tones from dominating skirt/pant color sampling.
            r = arr[:, :, 0].astype(np.int16)
            g = arr[:, :, 1].astype(np.int16)
            b = arr[:, :, 2].astype(np.int16)
            maxc = np.maximum.reduce([r, g, b])
            minc = np.minimum.reduce([r, g, b])
            skin_like = (
                (r > 95)
                & (g > 40)
                & (b > 20)
                & ((maxc - minc) > 15)
                & (np.abs(r - g) > 15)
                & (r > g)
                & (r > b)
            )
            trimmed = working & (~skin_like)
            if int(np.sum(trimmed)) >= max(self.settings.min_pixels, int(0.04 * np.sum(working))):
                working = trimmed
            # If the mask is dominated by very dark pixels (background/shadows),
            # trim them so color sampling doesn't collapse to black/charcoal.
            very_dark = (
                (arr[:, :, 0] <= 28)
                & (arr[:, :, 1] <= 28)
                & (arr[:, :, 2] <= 28)
                & ((arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)) <= 14)
            )
            dark_ratio = float(np.mean(very_dark[working])) if np.any(working) else 0.0
            desc_low = str(description or "").lower()
            dark_intended = bool(re.search(r"\\b(black|charcoal|ebony|midnight|dark|navy)\\b", desc_low))
            if (not dark_intended) and dark_ratio > 0.2:
                dark_trim = working & (~very_dark)
                if int(np.sum(dark_trim)) >= max(self.settings.min_pixels, int(0.06 * np.sum(working))):
                    working = dark_trim

        very_light = (
            (arr[:, :, 0] >= 226)
            & (arr[:, :, 1] >= 226)
            & (arr[:, :, 2] >= 226)
            & ((arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)) <= 20)
        )
        very_dark = (
            (arr[:, :, 0] <= 28)
            & (arr[:, :, 1] <= 28)
            & (arr[:, :, 2] <= 28)
            & ((arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)) <= 14)
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            working.astype(np.uint8), connectivity=8
        )
        if int(num_labels) <= 1:
            cleaned = self._cleanup_mask(working)
            return cleaned, {
                "source": "heuristic",
                "used": True,
                "reason": "single_component",
                "mask_pixels": int(np.sum(cleaned)),
            }

        seed_lab = self._seed_lab(arr=arr, working=working, garment_type=garment_type)
        target_y = self._target_y(garment_type)
        min_component_area = max(self.settings.min_pixels // 2, int(self.settings.component_min_area_ratio * h * w))

        ranked = []
        for label_idx in range(1, int(num_labels)):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue

            comp_mask = labels == label_idx
            cx, cy = centroids[label_idx]
            comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            area_ratio = float(area) / float(max(1, h * w))
            x_center_score = 1.0 - min(1.0, abs((float(cx) / float(w)) - 0.5) / 0.5)
            y_center_score = 1.0 - min(1.0, abs((float(cy) / float(h)) - target_y) / 0.65)
            height_ratio = float(comp_h) / float(max(1, h))
            width_ratio = float(comp_w) / float(max(1, w))
            light_ratio = float(np.mean(very_light[comp_mask])) if np.any(comp_mask) else 0.0
            score = (2.4 * area_ratio) + (0.7 * x_center_score) + (0.9 * y_center_score)
            if garment_type in {"top", "outer"}:
                score += 0.35 * min(1.0, height_ratio / 0.55)
            elif garment_type == "bottom":
                score += 0.75 * min(1.0, width_ratio / 0.55)
                score += 0.25 * min(1.0, height_ratio / 0.70)
                score -= self._bottom_component_penalty(
                    arr=arr,
                    comp_mask=comp_mask,
                    stats=stats,
                    label_idx=label_idx,
                    seed_lab=seed_lab,
                    light_ratio=light_ratio,
                    width_ratio=width_ratio,
                    image_width=w,
                )
                dark_ratio = float(np.mean(very_dark[comp_mask])) if np.any(comp_mask) else 0.0
                if dark_ratio > 0.35:
                    score *= 0.65
            elif garment_type == "dress":
                score += 0.30 * min(1.0, height_ratio / 0.75)
            if garment_type != "bottom" and light_ratio > 0.70:
                score *= 0.25
            ranked.append((score, int(label_idx), area))

        if not ranked:
            cleaned = self._cleanup_mask(working)
            return cleaned, {
                "source": "heuristic",
                "used": True,
                "reason": "rank_empty",
                "mask_pixels": int(np.sum(cleaned)),
            }

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score = float(ranked[0][0])
        best_area = int(ranked[0][2])
        keep_labels = []
        for score, label_idx, area in ranked[:4]:
            if score < (best_score * float(self.settings.keep_score_ratio)):
                continue
            if area < max(min_component_area, int(best_area * float(self.settings.keep_area_ratio))):
                continue
            keep_labels.append(int(label_idx))
        if not keep_labels:
            keep_labels = [int(ranked[0][1])]

        focused = np.isin(labels, keep_labels)
        cleaned = self._cleanup_mask(focused)
        if int(np.sum(cleaned)) < max(self.settings.min_pixels, int(0.04 * np.sum(base))):
            cleaned = self._cleanup_mask(working)

        return cleaned, {
            "source": "heuristic",
            "used": True,
            "reason": "ranked_components",
            "mask_pixels": int(np.sum(cleaned)),
            "area_ratio": round(float(np.mean(cleaned)), 6),
            "keep_labels": keep_labels,
        }

    def _mask_shape_ok(
        self,
        mask: np.ndarray,
        image_size: Tuple[int, int],
        garment_type: str,
        min_area_ratio: float,
    ) -> bool:
        h = int(image_size[1])
        w = int(image_size[0])
        if mask.shape[:2] != (h, w):
            return False
        keep = int(np.sum(mask))
        if keep < int(self.settings.min_pixels):
            return False
        area_ratio = float(keep) / float(max(1, h * w))
        if area_ratio < float(min_area_ratio):
            return False
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return False
        cy = float(np.mean(ys)) / float(max(1, h))
        if garment_type == "bottom" and cy < 0.42:
            return False
        if garment_type in {"top", "outer"} and cy > 0.72:
            return False
        return True

    def _cleanup_mask(self, mask: np.ndarray, keep_largest: bool = False) -> np.ndarray:
        cleaned = np.asarray(mask).astype(bool)
        if cleaned.size == 0:
            return cleaned
        cleaned = binary_open(cleaned, 3)
        cleaned = binary_close(cleaned, 3)
        try:
            import cv2

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned.astype(np.uint8) * 255))
            out = np.zeros_like(cleaned, dtype=bool)
            min_area = max(48, self.settings.min_pixels // 2)
            ranked = []
            for idx in range(1, int(num_labels)):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area >= min_area:
                    ranked.append((area, idx))
            if keep_largest and ranked:
                ranked = [max(ranked, key=lambda item: item[0])]
            for _area, idx in ranked:
                out[labels == idx] = True
            if int(np.sum(out)) > 0:
                cleaned = out
        except Exception:
            pass
        return cleaned.astype(bool)

    def _limit_skin_mask(self, skin: np.ndarray, garment_type: str, shape: Tuple[int, int]) -> np.ndarray:
        if skin.size == 0:
            return skin
        h, w = shape
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        if garment_type == "bottom":
            return skin & (
                (y < int(self.settings.bottom_skin_upper_ratio * h))
                | (x < int(self.settings.bottom_skin_side_ratio * w))
                | (x > int((1.0 - self.settings.bottom_skin_side_ratio) * w))
            )
        if garment_type == "top":
            return skin & (
                (y < int(self.settings.top_skin_upper_ratio * h))
                | (x < int(self.settings.top_skin_side_ratio * w))
                | (x > int((1.0 - self.settings.top_skin_side_ratio) * w))
            )
        if garment_type == "outer":
            return skin & (
                (y < int(self.settings.outer_skin_upper_ratio * h))
                | (x < int(self.settings.outer_skin_side_ratio * w))
                | (x > int((1.0 - self.settings.outer_skin_side_ratio) * w))
            )
        if garment_type == "dress":
            return skin & (
                (y < int(self.settings.dress_skin_upper_ratio * h))
                | (x < int(self.settings.dress_skin_side_ratio * w))
                | (x > int((1.0 - self.settings.dress_skin_side_ratio) * w))
            )
        return skin

    def _target_y(self, garment_type: str) -> float:
        return {
            "top": float(self.settings.top_target_y),
            "outer": float(self.settings.outer_target_y),
            "dress": float(self.settings.dress_target_y),
            "bottom": float(self.settings.bottom_target_y),
        }.get(str(garment_type or "").strip().lower(), 0.45)

    def _seed_lab(self, arr: np.ndarray, working: np.ndarray, garment_type: str) -> Optional[np.ndarray]:
        try:
            import cv2
        except Exception:
            return None
        h, w = arr.shape[:2]
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        if garment_type == "bottom":
            seed = (
                (y >= int(0.38 * h))
                & (y <= int(0.92 * h))
                & (x >= int(0.18 * w))
                & (x <= int(0.82 * w))
                & working
            )
        elif garment_type in {"top", "outer"}:
            seed = (
                (y >= int(0.12 * h))
                & (y <= int(0.62 * h))
                & (x >= int(0.15 * w))
                & (x <= int(0.85 * w))
                & working
            )
        else:
            seed = working
        if int(np.sum(seed)) < self.settings.min_pixels:
            seed = working
        if int(np.sum(seed)) < self.settings.min_pixels:
            return None
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        return np.median(lab[seed], axis=0)

    def _bottom_component_penalty(
        self,
        arr: np.ndarray,
        comp_mask: np.ndarray,
        stats,
        label_idx: int,
        seed_lab: Optional[np.ndarray],
        light_ratio: float,
        width_ratio: float,
        image_width: int,
    ) -> float:
        penalty = 0.0
        try:
            import cv2
        except Exception:
            return penalty
        if seed_lab is not None:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            comp_median = np.median(lab[comp_mask], axis=0)
            color_dist = float(np.linalg.norm(comp_median - seed_lab))
            penalty += 0.02 * color_dist
        if light_ratio > 0.25:
            penalty += 0.45 * light_ratio
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        touch_side = left <= 1 or (left + width >= image_width - 1)
        if touch_side and width_ratio > 0.20:
            penalty += 0.35
        return penalty
