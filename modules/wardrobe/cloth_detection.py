from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image


_RAW_TO_CANONICAL = {
    "top": "top",
    "upper": "top",
    "upper_clothes": "top",
    "shirt": "top",
    "t-shirt": "top",
    "tee": "top",
    "blouse": "top",
    "shirt": "top",
    "sweatshirt": "top",
    "long_sleeved_shirt": "top",
    "short_sleeved_shirt": "top",
    "long sleeve shirt": "top",
    "short sleeve shirt": "top",
    "long_sleeved_top": "top",
    "short_sleeved_top": "top",
    "short sleeve top": "top",
    "long sleeve top": "top",
    "vest": "top",
    "sling": "top",
    "bottom": "bottom",
    "pants": "bottom",
    "trousers": "bottom",
    "jeans": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "long pants": "bottom",
    "dress": "dress",
    "gown": "dress",
    "jumpsuit": "dress",
    "short_sleeved_dress": "dress",
    "long_sleeved_dress": "dress",
    "short sleeve dress": "dress",
    "long sleeve dress": "dress",
    "vest dress": "dress",
    "sling dress": "dress",
    "outer": "outer",
    "outerwear": "outer",
    "coat": "outer",
    "jacket": "outer",
    "blazer": "outer",
    "long_sleeved_outwear": "outer",
    "short_sleeved_outwear": "outer",
    "long sleeve outwear": "outer",
    "short sleeve outwear": "outer",
}

_NON_GARMENT_LABELS = {
    "bag",
    "bag_wallet",
    "bag, wallet",
    "hat",
    "shoe",
    "shoes",
    "sock",
    "glasses",
    "watch",
    "belt",
    "scarf",
    "umbrella",
    "tie",
    "glove",
    "headband",
    "head_covering",
    "hair_accessory",
}


@dataclass
class ClothDetectionConfig:
    detector_threshold: float = 0.30
    strong_confidence_threshold: float = 0.55
    ambiguity_margin: float = 0.08
    duplicate_iou_threshold: float = 0.72
    min_upper_center_ratio_for_bottom: float = 0.42
    max_lower_center_ratio_for_top: float = 0.72
    min_height_ratio_for_dress: float = 0.44
    min_crop_edge_px: int = 32
    pair_boundary_gap_px: int = 18
    top_bottom_boundary_blend: float = 0.50


class ClothDetector:
    """
    Comparison-friendly detector wrapper.

    It keeps the legacy YOLO path intact while exposing a fashion-specific
    detector path that can be verified and tightened with parser semantics.
    """

    def __init__(
        self,
        *,
        legacy_detector=None,
        fashion_detector=None,
        config: Optional[ClothDetectionConfig] = None,
    ):
        self.legacy_detector = legacy_detector
        self.fashion_detector = fashion_detector
        self.config = config or ClothDetectionConfig()

    @staticmethod
    def _normalize_type(raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if not text:
            return None
        return _RAW_TO_CANONICAL.get(text)

    @staticmethod
    def _is_non_garment_label(raw: Optional[str]) -> bool:
        if raw is None:
            return False
        text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if not text:
            return False
        if text in _NON_GARMENT_LABELS:
            return True
        return any(token in text for token in _NON_GARMENT_LABELS)

    @staticmethod
    def _bbox_iou(a: List[int], b: List[int]) -> float:
        ax0, ay0, ax1, ay1 = [int(v) for v in a]
        bx0, by0, bx1, by1 = [int(v) for v in b]
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter_w = max(0, inter_x1 - inter_x0)
        inter_h = max(0, inter_y1 - inter_y0)
        inter = float(inter_w * inter_h)
        area_a = float(max(1, (ax1 - ax0) * (ay1 - ay0)))
        area_b = float(max(1, (bx1 - bx0) * (by1 - by0)))
        union = max(1.0, area_a + area_b - inter)
        return inter / union

    @staticmethod
    def _normalize_prompt_label(raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        text = str(raw).strip().lower()
        text = text.replace("garment", "").replace("wear", "").replace("clothing", "")
        text = " ".join(text.split())
        for token in (
            "top",
            "bottom",
            "dress",
            "outer",
            "outerwear",
            "coat",
            "jacket",
            "blazer",
            "pants",
            "trousers",
            "jeans",
            "shorts",
            "skirt",
        ):
            if token in text:
                return ClothDetector._normalize_type(token)
        return ClothDetector._normalize_type(text)

    def _resolve_type(self, raw_label: Optional[str], requested_type: Optional[str]) -> Tuple[str, str]:
        if self._is_non_garment_label(raw_label):
            return "non_garment", "detector_non_garment"
        detector_type = self._normalize_type(raw_label) or self._normalize_prompt_label(raw_label)
        requested = self._normalize_type(requested_type)
        if requested and detector_type == requested:
            return requested, "requested_type_match"
        if detector_type in {"top", "bottom", "dress", "outer"}:
            return detector_type, "detector"
        if requested in {"top", "bottom", "dress", "outer"}:
            return requested, "requested_type_fallback"
        return "dress", "fallback_uncertain_garment"

    def _geometry_conflict(self, item: Dict[str, object], image_height: int) -> bool:
        bbox = item.get("bbox") or [0, 0, 0, 0]
        _, y0, _, y1 = [int(v) for v in bbox]
        box_h = max(1, y1 - y0)
        center_ratio = float(y0 + (box_h / 2.0)) / max(1.0, float(image_height))
        height_ratio = float(box_h) / max(1.0, float(image_height))
        item_type = self._normalize_type(item.get("label")) or self._normalize_prompt_label(item.get("label"))

        if item_type == "bottom" and center_ratio < self.config.min_upper_center_ratio_for_bottom:
            return True
        if item_type in {"top", "outer"} and center_ratio > self.config.max_lower_center_ratio_for_top:
            return True
        if item_type == "dress" and height_ratio < self.config.min_height_ratio_for_dress:
            return True
        return False

    @staticmethod
    def _bbox_metrics(bbox: List[int], image: Image.Image) -> Dict[str, float]:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        cx = x0 + (w / 2.0)
        cy = y0 + (h / 2.0)
        image_area = max(1.0, float(image.width * image.height))
        return {
            "width": float(w),
            "height": float(h),
            "aspect_ratio": float(w) / float(h),
            "area_ratio": float(w * h) / image_area,
            "center_x_ratio": float(cx) / max(1.0, float(image.width)),
            "center_y_ratio": float(cy) / max(1.0, float(image.height)),
            "height_ratio": float(h) / max(1.0, float(image.height)),
            "width_ratio": float(w) / max(1.0, float(image.width)),
        }

    def _dedupe_candidates(self, items: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not items:
            return []
        ranked = sorted(items, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        kept: List[Dict[str, object]] = []
        for item in ranked:
            bbox = item.get("bbox") or [0, 0, 0, 0]
            item_type = self._normalize_type(item.get("type"))
            duplicate = False
            for existing in kept:
                existing_bbox = existing.get("bbox") or [0, 0, 0, 0]
                existing_type = self._normalize_type(existing.get("type"))
                if item_type != existing_type:
                    continue
                if self._bbox_iou(bbox, existing_bbox) >= self.config.duplicate_iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(item)
        return kept

    def _is_weak_or_ambiguous(
        self,
        detections: List[Dict[str, object]],
        requested_type: Optional[str],
        image_height: int,
    ) -> bool:
        if not detections:
            return True
        sorted_detections = sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        garment_detections = [
            det for det in sorted_detections
            if not self._is_non_garment_label(det.get("label"))
            and (self._normalize_type(det.get("label")) or self._normalize_prompt_label(det.get("label")))
        ]
        if not garment_detections:
            return True
        sorted_detections = garment_detections
        top_score = float(sorted_detections[0].get("score", 0.0))
        if top_score < self.config.strong_confidence_threshold:
            return True

        if self._geometry_conflict(sorted_detections[0], image_height):
            return True

        if len(sorted_detections) >= 2:
            second_score = float(sorted_detections[1].get("score", 0.0))
            first_type = self._normalize_type(sorted_detections[0].get("label")) or self._normalize_prompt_label(sorted_detections[0].get("label"))
            second_type = self._normalize_type(sorted_detections[1].get("label")) or self._normalize_prompt_label(sorted_detections[1].get("label"))
            first_bbox = sorted_detections[0].get("bbox") or [0, 0, 0, 0]
            second_bbox = sorted_detections[1].get("bbox") or [0, 0, 0, 0]
            pair_types = {first_type, second_type}
            if (
                pair_types == {"top", "bottom"}
                and not self._geometry_conflict(sorted_detections[1], image_height)
                and self._bbox_iou(first_bbox, second_bbox) < self.config.duplicate_iou_threshold
            ):
                pass
            elif first_type != second_type and (top_score - second_score) <= self.config.ambiguity_margin:
                return True
            if self._bbox_iou(
                first_bbox,
                second_bbox,
            ) >= self.config.duplicate_iou_threshold:
                return True

        requested = self._normalize_type(requested_type)
        if requested and all((self._normalize_type(det.get("label")) or self._normalize_prompt_label(det.get("label"))) != requested for det in sorted_detections):
            return True

        return False

    def _build_candidates(
        self,
        detections: List[Dict[str, object]],
        image: Image.Image,
        *,
        requested_type: Optional[str],
        source_name: str,
    ) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        for idx, det in enumerate(detections):
            bbox = [int(v) for v in (det.get("bbox") or [0, 0, image.width, image.height])]
            x0, y0, x1, y1 = bbox
            if (x1 - x0) < self.config.min_crop_edge_px or (y1 - y0) < self.config.min_crop_edge_px:
                continue

            resolved_type, type_source = self._resolve_type(det.get("label"), requested_type)
            if resolved_type == "non_garment":
                continue
            crop_img = image.crop(tuple(bbox))
            metrics = self._bbox_metrics(bbox, image)
            candidates.append({
                "id": idx,
                "label": str(det.get("label") or ""),
                "type": resolved_type,
                "type_source": type_source,
                "confidence": float(det.get("score", 0.0)),
                "bbox": bbox,
                "image": crop_img,
                "mask": None,
                "source": source_name,
                "metrics": metrics,
            })
        return self._dedupe_candidates(candidates)

    def _tighten_pair_boundaries(self, items: List[Dict[str, object]], image: Image.Image) -> List[Dict[str, object]]:
        if len(items) < 2:
            return items

        top_like = [item for item in items if item.get("type") in {"top", "outer"}]
        bottom_like = [item for item in items if item.get("type") == "bottom"]
        if not top_like or not bottom_like:
            return items

        top_item = max(top_like, key=lambda item: float(item.get("confidence", 0.0)))
        bottom_item = max(bottom_like, key=lambda item: float(item.get("confidence", 0.0)))
        tx0, ty0, tx1, ty1 = [int(v) for v in (top_item.get("bbox") or [0, 0, image.width, image.height])]
        bx0, by0, bx1, by1 = [int(v) for v in (bottom_item.get("bbox") or [0, 0, image.width, image.height])]
        if ty1 >= by1:
            return items

        gap = by0 - ty1
        overlap = ty1 - by0
        target_boundary = int(round((ty1 * (1.0 - self.config.top_bottom_boundary_blend)) + (by0 * self.config.top_bottom_boundary_blend)))
        tightened_boundary = target_boundary
        if gap > self.config.pair_boundary_gap_px:
            tightened_boundary = min(by0, target_boundary + (gap // 3))
        elif overlap > 0:
            tightened_boundary = max(ty0 + 1, target_boundary - (overlap // 2))

        new_top_bbox = [tx0, ty0, tx1, max(ty0 + 1, min(ty1, tightened_boundary))]
        new_bottom_bbox = [bx0, max(new_top_bbox[3], by0), bx1, by1]

        updated: List[Dict[str, object]] = []
        for item in items:
            clone = dict(item)
            if item is top_item or item.get("id") == top_item.get("id"):
                clone["bbox"] = new_top_bbox
                clone["image"] = image.crop(tuple(new_top_bbox))
                clone["metrics"] = self._bbox_metrics(new_top_bbox, image)
                clone["tighten_reason"] = "pair_boundary_top"
            elif item is bottom_item or item.get("id") == bottom_item.get("id"):
                clone["bbox"] = new_bottom_bbox
                clone["image"] = image.crop(tuple(new_bottom_bbox))
                clone["metrics"] = self._bbox_metrics(new_bottom_bbox, image)
                clone["tighten_reason"] = "pair_boundary_bottom"
            updated.append(clone)
        return updated

    def _build_legacy_candidates(
        self,
        image: Image.Image,
        *,
        requested_type: Optional[str],
    ) -> List[Dict[str, object]]:
        if self.legacy_detector is None:
            return []
        instances = self.legacy_detector.detect_instances(image)
        crops = self.legacy_detector.get_crops(image, instances)
        raw: List[Dict[str, object]] = []
        for item in crops:
            raw.append({
                "bbox": [int(v) for v in (item.get("bbox") or [0, 0, image.width, image.height])],
                "score": float(item.get("confidence", 0.0) or 0.0),
                "label": str(item.get("label") or ""),
                "source": "legacy_yolo",
            })
        candidates = self._build_candidates(raw, image, requested_type=requested_type, source_name="legacy_yolo")
        return self._tighten_pair_boundaries(candidates, image)

    def detect_fashion_candidates(
        self,
        image: Image.Image,
        *,
        requested_type: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        if self.fashion_detector is None:
            raise RuntimeError("Fashion detector is not configured.")

        raw = self.fashion_detector.predict(image, threshold=threshold or self.config.detector_threshold)
        candidates = self._build_candidates(
            raw,
            image,
            requested_type=requested_type,
            source_name="fashion_object_detection",
        )
        candidates = self._tighten_pair_boundaries(candidates, image)
        use_legacy_fallback = self._is_weak_or_ambiguous(raw, requested_type, image.height) or not candidates
        if use_legacy_fallback:
            legacy_candidates = self._build_legacy_candidates(image, requested_type=requested_type)
            if legacy_candidates:
                candidates = legacy_candidates

        def _sort_key(item: Dict[str, object]) -> Tuple[float, float]:
            bbox = item.get("bbox") or [0, 0, 0, 0]
            x0, y0, x1, y1 = [int(v) for v in bbox]
            area = float(max(1, (x1 - x0) * (y1 - y0)))
            req_bonus = 0.0
            if requested_type and item.get("type") == self._normalize_type(requested_type):
                req_bonus = 0.08
            return (float(item.get("confidence", 0.0)) + req_bonus, area)

        return sorted(candidates, key=_sort_key, reverse=True)

    def compare_backends(
        self,
        image: Image.Image,
        *,
        requested_type: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        comparison: Dict[str, List[Dict[str, object]]] = {"fashion": [], "legacy": []}
        raw_fashion = self.fashion_detector.predict(image, threshold=threshold or self.config.detector_threshold) if self.fashion_detector is not None else []
        comparison["fashion"] = self._tighten_pair_boundaries(
            self._build_candidates(
                raw_fashion,
                image,
                requested_type=requested_type,
                source_name="fashion_object_detection",
            ),
            image,
        )

        if self.legacy_detector is not None:
            comparison["legacy"] = self._build_legacy_candidates(image, requested_type=requested_type)

        return comparison
