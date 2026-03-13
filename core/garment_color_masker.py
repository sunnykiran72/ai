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
    component_min_area_ratio: float = 0.004
    keep_score_ratio: float = 0.42
    keep_area_ratio: float = 0.12
    bottom_skin_upper_ratio: float = 0.42
    bottom_skin_side_ratio: float = 0.10
    bottom_target_y: float = 0.73
    top_target_y: float = 0.34
    outer_target_y: float = 0.40
    dress_target_y: float = 0.52
    parser_skin_max_strip_ratio: float = 0.38


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
            return parser_mask.astype(bool), parser_meta

        heuristic_mask, heuristic_meta = self._mask_from_heuristic(image, garment_type, description=description)
        return heuristic_mask, heuristic_meta

    def _mask_from_parser(
        self,
        image: Image.Image,
        garment_type: str,
    ) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
        if self.parser is None:
            return None, {"source": "parser", "used": False, "reason": "parser_unavailable"}
        try:
            parsing = self.parser.parse(image)
            mask, mask_source = self._strict_mask_from_parser(parsing, garment_type)
            if mask is None:
                mask = self.parser.get_mask_for_category(parsing, garment_type)
                mask_source = "parser_category"
            mask = np.asarray(mask).astype(bool)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            skin_pixels = 0
            if callable(self.skin_mask_fn):
                try:
                    skin = np.asarray(self.skin_mask_fn(rgb)).astype(bool)
                    skin = self._limit_skin_mask(skin=skin, garment_type=garment_type, shape=mask.shape[:2])
                    masked_skin = skin & mask
                    masked_skin_ratio = float(np.sum(masked_skin)) / float(max(1, np.sum(mask)))
                    if int(np.sum(masked_skin)) > 0:
                        candidate = mask & (~masked_skin)
                        candidate_cleaned = self._cleanup_mask(candidate, keep_largest=True)
                        if self._mask_shape_ok(
                            candidate_cleaned,
                            image.size,
                            garment_type,
                            min_area_ratio=self.settings.parser_min_area_ratio,
                        ):
                            mask = candidate
                            skin_pixels = int(np.sum(masked_skin))
                        elif masked_skin_ratio <= float(self.settings.parser_skin_max_strip_ratio):
                            mask = candidate
                            skin_pixels = int(np.sum(masked_skin))
                except Exception:
                    skin_pixels = 0
            cleaned = self._cleanup_mask(mask, keep_largest=True)
            if not self._mask_shape_ok(cleaned, image.size, garment_type, min_area_ratio=self.settings.parser_min_area_ratio):
                return None, {
                    "source": mask_source,
                    "used": False,
                    "reason": "mask_quality_low",
                    "mask_pixels": int(np.sum(cleaned)),
                    "area_ratio": round(float(np.mean(cleaned)), 6),
                }
            return cleaned, {
                "source": mask_source,
                "used": True,
                "mask_pixels": int(np.sum(cleaned)),
                "area_ratio": round(float(np.mean(cleaned)), 6),
                "skin_pixels_removed": skin_pixels,
            }
        except Exception as err:
            return None, {"source": "parser", "used": False, "reason": f"error:{err}"}

    def _strict_mask_from_parser(
        self,
        parsing: np.ndarray,
        garment_type: str,
    ) -> Tuple[Optional[np.ndarray], str]:
        runtime_labels = {}
        runtime_fn = getattr(self.parser, "_runtime_labels", None)
        if callable(runtime_fn):
            try:
                runtime_labels = dict(runtime_fn() or {})
            except Exception:
                runtime_labels = {}
        if not runtime_labels:
            return None, "parser"

        alias_map = {
            "top": ["top", "upper", "upper_clothes"],
            "outer": ["outer", "outerwear", "coat", "jacket", "blazer", "top", "upper", "upper_clothes"],
            "bottom": ["bottom", "pants", "trousers", "skirt", "shorts"],
            "dress": ["dress"],
        }
        normalized_type = str(garment_type or "").strip().lower()
        aliases = alias_map.get(normalized_type, [])
        ids = []
        for alias in aliases:
            key = str(alias).strip().lower().replace("-", "_").replace(" ", "_")
            if key in runtime_labels:
                try:
                    ids.append(int(runtime_labels[key]))
                except Exception:
                    continue
        ids = sorted(set(ids))
        if not ids:
            return None, "parser"
        return np.isin(parsing, ids), "parser_strict_runtime"

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
