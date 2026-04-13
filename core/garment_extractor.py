from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GarmentExtractionConfig:
    force_bbox_crop: bool = True
    crop_pad_ratio_x: float = 0.30
    crop_pad_ratio_y: float = 0.10
    crop_pad_ratio: float = 0.18
    crop_pad_ratio_dress: float = 0.28
    crop_bottom_extra_ratio_dress: float = 0.32
    crop_bottom_extra_ratio_top_multi: float = 0.04
    crop_top_extra_ratio_bottom: float = 0.12
    crop_top_extra_ratio_bottom_multi: float = 0.12
    dress_top_recovery_ratio: float = 0.14
    top_top_recovery_ratio: float = 0.08
    semantic_refine_enabled: bool = True
    semantic_min_horizontal_overlap_ratio: float = 0.55


@dataclass(frozen=True)
class GarmentExtractionRequest:
    full_image: Image.Image
    garment_type: str
    total_items: int
    detected_bbox: Optional[list[int]] = None
    detector_mask: Optional[np.ndarray] = None
    semantic_bbox: Optional[list[int]] = None


@dataclass(frozen=True)
class GarmentExtractionResult:
    image: Image.Image
    anchor_bbox: list[int]
    extract_bbox: list[int]
    crop_mode: str
    geometry_source: str
    mask_bbox: Optional[list[int]] = None
    semantic_bbox: Optional[list[int]] = None


def _normalize_garment_type(value: str) -> str:
    text = " ".join(str(value or "").strip().lower().replace("_", " ").split())
    if text in {"top", "tops"}:
        return "top"
    if text in {"bottom", "bottoms", "pant", "pants", "trouser", "trousers", "skirt", "shorts"}:
        return "bottom"
    if text in {"dress", "dresses", "gown"}:
        return "dress"
    if text in {"outer", "outerwear", "jacket", "coat", "hoodie", "blazer"}:
        return "outer"
    return text or "top"


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(x0, max(0, width - 1)))
    y0 = max(0, min(y0, max(0, height - 1)))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return [x0, y0, x1, y1]


def _bbox_from_mask(mask: Optional[np.ndarray], width: int, height: int) -> Optional[list[int]]:
    if not isinstance(mask, np.ndarray):
        return None
    if mask.ndim != 2 or mask.shape[0] != height or mask.shape[1] != width:
        return None
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _horizontal_overlap_ratio(a: list[int], b: list[int]) -> float:
    ax0, _, ax1, _ = [int(v) for v in a]
    bx0, _, bx1, _ = [int(v) for v in b]
    inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    amin = max(1, ax1 - ax0)
    bmin = max(1, bx1 - bx0)
    return float(inter) / float(max(1, min(amin, bmin)))


class GarmentExtractor:
    def __init__(self, config: Optional[GarmentExtractionConfig] = None):
        self.config = config or GarmentExtractionConfig()

    def _select_anchor_bbox(
        self,
        width: int,
        height: int,
        detected_bbox: Optional[list[int]],
        detector_mask: Optional[np.ndarray],
        semantic_bbox: Optional[list[int]],
        garment_type: str,
    ) -> tuple[list[int], str, Optional[list[int]]]:
        detected = _clamp_bbox(detected_bbox or [0, 0, width, height], width, height) if detected_bbox else None
        mask_bbox = _bbox_from_mask(detector_mask, width, height)
        base = list(mask_bbox or detected or [0, 0, width, height])
        source = "mask_bbox" if mask_bbox else ("detector_bbox" if detected else "full_image")

        gt = _normalize_garment_type(garment_type)
        if gt == "dress":
            recover = int(round((base[3] - base[1]) * max(0.0, self.config.dress_top_recovery_ratio)))
            if recover > 0:
                base[1] = max(0, base[1] - recover)
                source = f"{source}_dress_top_recovery"
        elif gt in {"top", "outer"}:
            recover = int(round((base[3] - base[1]) * max(0.0, self.config.top_top_recovery_ratio)))
            if recover > 0:
                base[1] = max(0, base[1] - recover)
                source = f"{source}_top_recovery"

        semantic = _clamp_bbox(semantic_bbox, width, height) if semantic_bbox else None
        if (
            self.config.semantic_refine_enabled
            and semantic
            and gt in {"dress", "top", "outer"}
            and _horizontal_overlap_ratio(base, semantic) >= self.config.semantic_min_horizontal_overlap_ratio
            and semantic[1] < base[1]
        ):
            base = [
                min(base[0], semantic[0]),
                min(base[1], semantic[1]),
                max(base[2], semantic[2]),
                max(base[3], semantic[3]),
            ]
            source = f"{source}_semantic_top_refine"

        return _clamp_bbox(base, width, height), source, mask_bbox

    def prepare(self, request: GarmentExtractionRequest) -> GarmentExtractionResult:
        width, height = request.full_image.size
        if not self.config.force_bbox_crop:
            return GarmentExtractionResult(
                image=request.full_image.copy(),
                anchor_bbox=[0, 0, width, height],
                extract_bbox=[0, 0, width, height],
                crop_mode="full_image_no_bbox",
                geometry_source="full_image",
            )

        anchor_bbox, geometry_source, mask_bbox = self._select_anchor_bbox(
            width=width,
            height=height,
            detected_bbox=request.detected_bbox,
            detector_mask=request.detector_mask,
            semantic_bbox=request.semantic_bbox,
            garment_type=request.garment_type,
        )

        x0, y0, x1, y1 = anchor_bbox
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        # Uniform ratio-based padding for every garment type.
        x_pad_ratio = max(0.0, min(0.8, float(self.config.crop_pad_ratio_x)))
        y_pad_ratio = max(0.0, min(0.8, float(self.config.crop_pad_ratio_y)))
        x_pad = int(bw * x_pad_ratio)
        y_pad_top = int(bh * y_pad_ratio)
        y_pad_bottom = int(bh * y_pad_ratio)

        extract_bbox = _clamp_bbox(
            [x0 - x_pad, y0 - y_pad_top, x1 + x_pad, y1 + y_pad_bottom],
            width,
            height,
        )
        ex0, ey0, ex1, ey1 = extract_bbox
        crop_mode = "garment_mask_bbox_expanded" if mask_bbox else "garment_bbox_expanded"
        return GarmentExtractionResult(
            image=request.full_image.crop((ex0, ey0, ex1, ey1)),
            anchor_bbox=anchor_bbox,
            extract_bbox=extract_bbox,
            crop_mode=crop_mode,
            geometry_source=geometry_source,
            mask_bbox=mask_bbox,
            semantic_bbox=_clamp_bbox(request.semantic_bbox, width, height) if request.semantic_bbox else None,
        )
