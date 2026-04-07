"""
User preparation utilities.

This module contains the modular user-image preparation pipeline used by
``/v1/user-image/prepare``. The functions are intentionally small and mostly
pure so they can be unit tested without running the full API stack.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

try:  # Optional local-only dependency for fast face visibility checks.
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - face detection is optional in tests
    cv2 = None
    np = None

from utils.validation import validate_image_quality


DEFAULT_USER_DESCRIPTION = (
    "person with visible hairstyle, balanced build, relaxed standing pose, identity cues preserved."
)
ALLOWED_WORN_TYPES = ("top", "bottom", "outer", "dress")

DEFAULT_VERIFICATION_PROMPT = (
    "You are a strict image eligibility validator for a user photo upload. "
    "Return JSON only with these keys: "
    '{"single_person": true/false, "face_visible": true/false, "upper_body_visible": true/false, '
    '"lower_body_visible": true/false, "clear_human": true/false, "reason": "short reason"}. '
    "Do not add extra keys. "
    "Rules: "
    "single_person should be true only when one dominant foreground person is present. "
    "face_visible should be true only when the face is clearly visible and usable for try-on identity consistency. "
    "upper_body_visible should be true only when torso/upper-body region is clearly visible. "
    "lower_body_visible should be true only when lower-body region is clearly visible. "
    "clear_human should be true only when the subject is a clear real human photo. "
    "If uncertain, return false values. Do not include markdown or extra text."
)

GROUNDING_DINO_PROMPTS = [
    "person",
    "face",
    "head",
    "upper body",
    "lower body",
    "torso",
    "hip",
    "thigh",
    "leg",
    "foot",
]

GROUNDING_PERSON_ANCHOR_PROMPTS = [
    "single person",
    "full body person",
    "person",
    "human body",
    "human",
]

GROUNDING_PERSON_KEYS = ("person", "human")
GROUNDING_FACE_KEYS = ("face",)
GROUNDING_HEAD_KEYS = ("head",)
GROUNDING_TOP_KEYS = ("upper body", "torso", "shirt", "t-shirt", "blouse", "jacket", "top")
GROUNDING_BOTTOM_KEYS = (
    "lower body",
    "hip",
    "thigh",
    "leg",
    "foot",
    "pants",
    "trousers",
    "jeans",
    "skirt",
    "shorts",
    "bottom",
)

# Low threshold by design to preserve recall while still enforcing explicit section visibility.
GROUNDING_MIN_SCORE = 0.30
GROUNDING_MIN_BODY_EXTENT_RATIO = 0.30
GROUNDING_MULTIPLE_RATIO = 0.85
GROUNDING_MULTIPLE_SECOND_MIN = 0.30
GROUNDING_MULTIPLE_AREA_RATIO = 0.45


def _normalize_worn_types(values: Any) -> List[str]:
    out: List[str] = []
    if values is None:
        return out
    if isinstance(values, str):
        items = re.split(r"[,\s]+", values)
    elif isinstance(values, (list, tuple, set)):
        items = [str(v or "") for v in values]
    else:
        items = [str(values or "")]
    for raw in items:
        token = str(raw or "").strip().lower()
        if token in ALLOWED_WORN_TYPES and token not in out:
            out.append(token)
    return out


def _extract_user_prepare_prompt_bundle(raw_text: str) -> Tuple[str, List[str]]:
    text = str(raw_text or "").strip()
    if not text:
        return DEFAULT_USER_DESCRIPTION, []

    candidate_payloads: List[str] = [text]
    if text.startswith("```"):
        stripped = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE).strip()
        if stripped:
            candidate_payloads.append(stripped)
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate_payloads.append(text[start : end + 1].strip())

    prompt = ""
    worn_types: List[str] = []
    for payload in candidate_payloads:
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        prompt = str(obj.get("prompt") or obj.get("description") or "").strip()
        worn_types = _normalize_worn_types(obj.get("garments") or obj.get("wornTypes"))
        if prompt or worn_types:
            break

    if not prompt:
        prompt = text
    worn_types = _normalize_worn_types(worn_types)

    if not prompt:
        prompt = DEFAULT_USER_DESCRIPTION
    return prompt, worn_types


def _label_matches_any(label: Any, keys: Sequence[str]) -> bool:
    low = str(label or "").strip().lower()
    if not low:
        return False
    return any(key in low for key in keys)


def _bbox_contains_point(bbox: Sequence[int], x: float, y: float) -> bool:
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return x0 <= x <= x1 and y0 <= y <= y1


def _top_label_candidate(
    detections: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
    min_score: float,
    person_bbox: Optional[Sequence[int]],
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, Any]]:
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for det in detections:
        if not _label_matches_any(det.get("label"), keys):
            continue
        score = float(det.get("score") or det.get("confidence") or 0.0)
        if score < min_score:
            continue
        bbox = _normalize_bbox(det.get("bbox") or [0, 0, image_width, image_height], image_width, image_height)
        cx, cy = _bbox_center(bbox)
        inside_bonus = 0.0
        if person_bbox is not None and _bbox_contains_point(person_bbox, cx, cy):
            inside_bonus = 0.05
        ranked.append((score + inside_bonus, {**det, "bbox": bbox, "score": score}))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def detect_grounding_visibility(
    image: Image.Image,
    detector_fn: Optional[Callable[..., Any]] = None,
    *,
    min_score: float = GROUNDING_MIN_SCORE,
    min_body_extent_ratio: float = GROUNDING_MIN_BODY_EXTENT_RATIO,
) -> Dict[str, Any]:
    if not isinstance(image, Image.Image):
        return {"ok": False, "error": "invalid_image", "message": "Expected a valid image."}

    width, height = image.size
    if width <= 0 or height <= 0:
        return {"ok": False, "error": "invalid_image_size", "message": "Invalid image size."}

    if detector_fn is None:
        return {
            "ok": False,
            "error": "detector_unavailable",
            "message": "Grounding detector is unavailable.",
            "meta": {"backend": "grounding_dino", "reason": "detector_fn_missing"},
        }

    def _run_detector(prompts: Sequence[str]) -> Any:
        try:
            return detector_fn(image, prompts=prompts)
        except TypeError:
            return detector_fn(image, prompts)

    try:
        outputs = _run_detector(GROUNDING_DINO_PROMPTS)
    except Exception as exc:
        return {
            "ok": False,
            "error": "detector_failed",
            "message": f"Grounding detection failed: {exc}",
            "meta": {"backend": "grounding_dino", "reason": "detector_call_failed", "error": str(exc)},
        }

    if not isinstance(outputs, list):
        outputs = []
    detections: List[Dict[str, Any]] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        bbox = _normalize_bbox(item.get("bbox") or [0, 0, width, height], width, height)
        score = float(item.get("score") or item.get("confidence") or 0.0)
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        detections.append(
            {
                "bbox": bbox,
                "score": score,
                "label": label,
                "source": str(item.get("source") or "grounding_dino"),
            }
        )

    person_pool = [
        det for det in detections
        if _label_matches_any(det.get("label"), GROUNDING_PERSON_KEYS) and float(det.get("score") or 0.0) >= min_score
    ]
    person_anchor_used = False
    anchor_counts: Dict[str, Any] = {"all": 0, "person": 0}
    if not person_pool:
        anchor_outputs: Any = []
        try:
            anchor_outputs = _run_detector(GROUNDING_PERSON_ANCHOR_PROMPTS)
        except Exception:
            anchor_outputs = []

        if not isinstance(anchor_outputs, list):
            anchor_outputs = []

        anchor_detections: List[Dict[str, Any]] = []
        for item in anchor_outputs:
            if not isinstance(item, dict):
                continue
            bbox = _normalize_bbox(item.get("bbox") or [0, 0, width, height], width, height)
            score = float(item.get("score") or item.get("confidence") or 0.0)
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            anchor_detections.append(
                {
                    "bbox": bbox,
                    "score": score,
                    "label": label,
                    "source": str(item.get("source") or "grounding_dino_anchor"),
                }
            )

        anchor_person_pool = [
            det
            for det in anchor_detections
            if _label_matches_any(det.get("label"), GROUNDING_PERSON_KEYS) and float(det.get("score") or 0.0) >= min_score
        ]
        anchor_counts = {"all": len(anchor_detections), "person": len(anchor_person_pool)}
        if anchor_person_pool:
            person_anchor_used = True
            person_pool = anchor_person_pool
            detections.extend(anchor_detections)

    if not person_pool:
        return {
            "ok": False,
            "error": "no_person",
            "message": "No person detected in the image.",
            "meta": {
                "backend": "grounding_dino",
                "prompts": list(GROUNDING_DINO_PROMPTS),
                "person_anchor_prompts": list(GROUNDING_PERSON_ANCHOR_PROMPTS),
                "thresholds": {"score_min": float(min_score), "body_extent_min": float(min_body_extent_ratio)},
                "counts": {"all": len(detections), "person": 0},
                "anchor_counts": dict(anchor_counts),
            },
        }

    def _person_rank_key(det: Dict[str, Any]) -> Tuple[float, float]:
        bbox = det.get("bbox") or [0, 0, width, height]
        area_ratio = float(_bbox_area(bbox)) / float(max(1, width * height))
        return (
            float(det.get("score") or 0.0) + (0.08 * area_ratio),
            area_ratio,
        )

    person_ranked = sorted(person_pool, key=_person_rank_key, reverse=True)
    primary_person = person_ranked[0]
    primary_bbox = _normalize_bbox(primary_person.get("bbox") or [0, 0, width, height], width, height)
    primary_score = float(primary_person.get("score") or 0.0)
    primary_area_ratio = float(_bbox_area(primary_bbox)) / float(max(1, width * height))

    if len(person_ranked) > 1:
        second = person_ranked[1]
        second_bbox = _normalize_bbox(second.get("bbox") or [0, 0, width, height], width, height)
        second_score = float(second.get("score") or 0.0)
        second_area_ratio = float(_bbox_area(second_bbox)) / float(max(1, width * height))
        if (
            second_score >= max(GROUNDING_MULTIPLE_SECOND_MIN, primary_score * GROUNDING_MULTIPLE_RATIO)
            and second_area_ratio >= (primary_area_ratio * GROUNDING_MULTIPLE_AREA_RATIO)
        ):
            return {
                "ok": False,
                "error": "multiple_people",
                "message": "Multiple prominent people detected. Please upload a photo with one clearly dominant person.",
                "meta": {
                    "backend": "grounding_dino",
                    "counts": {"all": len(detections), "person": len(person_pool)},
                    "selected_person": {
                        "bbox": primary_bbox,
                        "score": round(primary_score, 4),
                        "area_ratio": round(primary_area_ratio, 4),
                    },
                    "second_person": {
                        "bbox": second_bbox,
                        "score": round(second_score, 4),
                        "area_ratio": round(second_area_ratio, 4),
                    },
                    "thresholds": {
                        "multiple_ratio": GROUNDING_MULTIPLE_RATIO,
                        "multiple_second_min": GROUNDING_MULTIPLE_SECOND_MIN,
                        "multiple_area_ratio": GROUNDING_MULTIPLE_AREA_RATIO,
                    },
                },
            }

    face_det = _top_label_candidate(
        detections,
        keys=GROUNDING_FACE_KEYS,
        min_score=min_score,
        person_bbox=primary_bbox,
        image_width=width,
        image_height=height,
    )
    if face_det is None:
        return {
            "ok": False,
            "error": "face_hidden",
            "message": "Face not clearly visible.",
            "meta": {
                "backend": "grounding_dino",
                "counts": {"all": len(detections), "person": len(person_pool), "face": 0},
                "selected_person": {"bbox": primary_bbox, "score": round(primary_score, 4)},
                "head_detected": bool(
                    _top_label_candidate(
                        detections,
                        keys=GROUNDING_HEAD_KEYS,
                        min_score=min_score,
                        person_bbox=primary_bbox,
                        image_width=width,
                        image_height=height,
                    )
                ),
            },
        }

    face_bbox = _normalize_bbox(face_det.get("bbox") or [0, 0, width, height], width, height)
    fx0, fy0, fx1, fy1 = face_bbox
    face_center_x = (fx0 + fx1) / 2.0
    face_center_y = (fy0 + fy1) / 2.0
    x0, y0, x1, y1 = primary_bbox
    bbox_h = max(1, y1 - y0)
    bbox_w = max(1, x1 - x0)
    face_rel_y = (face_center_y - y0) / float(bbox_h)

    top_det = _top_label_candidate(
        detections,
        keys=GROUNDING_TOP_KEYS,
        min_score=min_score,
        person_bbox=primary_bbox,
        image_width=width,
        image_height=height,
    )
    if top_det is None:
        # Fallback: if the face is clearly in upper half of person bbox, infer upper-body visibility.
        if 0.0 <= face_rel_y <= 0.55:
            top_det = {
                "bbox": [x0, y0, x1, _clamp(y0 + int(round(0.58 * bbox_h)), y0 + 1, y1)],
                "score": float(face_det.get("score") or 0.0),
                "label": "upper_body_proxy_from_face",
                "source": "grounding_dino_proxy",
            }
        else:
            return {
                "ok": False,
                "error": "top_section_not_visible",
                "message": "Upper body is not clearly visible.",
                "meta": {
                    "backend": "grounding_dino",
                    "selected_person": {"bbox": primary_bbox, "score": round(primary_score, 4)},
                    "selected_face": {"bbox": face_bbox, "score": round(float(face_det.get("score") or 0.0), 4)},
                },
            }

    bottom_det = _top_label_candidate(
        detections,
        keys=GROUNDING_BOTTOM_KEYS,
        min_score=min_score,
        person_bbox=primary_bbox,
        image_width=width,
        image_height=height,
    )
    if bottom_det is None:
        # Fallback: infer lower-body section from person bbox when face is in upper region.
        if 0.0 <= face_rel_y <= 0.55:
            bottom_det = {
                "bbox": [_clamp(x0, 0, width), _clamp(y0 + int(round(0.42 * bbox_h)), 0, height), _clamp(x1, 0, width), _clamp(y1, 0, height)],
                "score": float(primary_score),
                "label": "lower_body_proxy_from_person",
                "source": "grounding_dino_proxy",
            }
        else:
            return {
                "ok": False,
                "error": "bottom_section_not_visible",
                "message": "Lower body is not clearly visible.",
                "meta": {
                    "backend": "grounding_dino",
                    "selected_person": {"bbox": primary_bbox, "score": round(primary_score, 4)},
                    "selected_face": {"bbox": face_bbox, "score": round(float(face_det.get("score") or 0.0), 4)},
                    "selected_top": {"bbox": top_det.get("bbox"), "score": round(float(top_det.get("score") or 0.0), 4)},
                },
            }

    major_axis = "vertical" if bbox_h >= bbox_w else "horizontal"
    body_extent_ratio = (bbox_h / float(max(1, height))) if major_axis == "vertical" else (bbox_w / float(max(1, width)))
    if body_extent_ratio < float(min_body_extent_ratio):
        return {
            "ok": False,
            "error": "body_not_clear",
            "message": "Body visibility is too low.",
            "meta": {
                "backend": "grounding_dino",
                "selected_person": {
                    "bbox": primary_bbox,
                    "score": round(primary_score, 4),
                    "area_ratio": round(primary_area_ratio, 4),
                },
                "metrics": {
                    "major_axis": major_axis,
                    "body_extent_ratio": round(float(body_extent_ratio), 4),
                },
                "thresholds": {"body_extent_min": float(min_body_extent_ratio)},
            },
        }

    face_area_ratio_in_person = float(_bbox_area(face_bbox)) / float(max(1, _bbox_area(primary_bbox)))
    result_meta = {
        "backend": "grounding_dino",
        "prompts": list(GROUNDING_DINO_PROMPTS),
        "person_anchor_prompts": list(GROUNDING_PERSON_ANCHOR_PROMPTS),
        "person_anchor_used": bool(person_anchor_used),
        "thresholds": {"score_min": float(min_score), "body_extent_min": float(min_body_extent_ratio)},
        "counts": {
            "all": len(detections),
            "person": len(person_pool),
            "face": len([d for d in detections if _label_matches_any(d.get("label"), GROUNDING_FACE_KEYS)]),
            "head": len([d for d in detections if _label_matches_any(d.get("label"), GROUNDING_HEAD_KEYS)]),
            "top": len([d for d in detections if _label_matches_any(d.get("label"), GROUNDING_TOP_KEYS)]),
            "bottom": len([d for d in detections if _label_matches_any(d.get("label"), GROUNDING_BOTTOM_KEYS)]),
        },
        "selected": {
            "person": {"bbox": primary_bbox, "score": round(primary_score, 4), "label": str(primary_person.get("label") or "person")},
            "face": {"bbox": face_bbox, "score": round(float(face_det.get("score") or 0.0), 4), "label": str(face_det.get("label") or "face")},
            "top": {"bbox": _normalize_bbox(top_det.get("bbox") or [0, 0, width, height], width, height), "score": round(float(top_det.get("score") or 0.0), 4), "label": str(top_det.get("label") or "upper body")},
            "bottom": {"bbox": _normalize_bbox(bottom_det.get("bbox") or [0, 0, width, height], width, height), "score": round(float(bottom_det.get("score") or 0.0), 4), "label": str(bottom_det.get("label") or "lower body")},
        },
        "metrics": {
            "major_axis": major_axis,
            "body_extent_ratio": round(float(body_extent_ratio), 4),
            "bbox_area_ratio": round(float(primary_area_ratio), 4),
            "face_area_ratio_in_person": round(float(face_area_ratio_in_person), 4),
            "face_rel_y": round(float(face_rel_y), 4),
        },
    }
    return {
        "ok": True,
        "primary_person": {
            "bbox": primary_bbox,
            "confidence": float(primary_score),
            "area_ratio": float(primary_area_ratio),
            "person_score": float(primary_score),
            "source": "grounding_dino",
        },
        "meta": result_meta,
    }


@dataclass(frozen=True)
class PreparedCandidate:
    bbox: List[int]
    confidence: float
    area_ratio: float
    person_score: float
    source: str = "detector"


@dataclass(frozen=True)
class FaceCandidate:
    bbox: List[int]
    confidence: float
    area_ratio: float
    source: str = "face_detector"


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _bbox_area(bbox: Sequence[int]) -> int:
    if len(bbox) != 4:
        return 0
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return max(0, x1 - x0) * max(0, y1 - y0)


def _bbox_center(bbox: Sequence[int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _normalize_bbox(bbox: Sequence[Any], width: int, height: int) -> List[int]:
    if len(bbox) != 4:
        return [0, 0, width, height]
    x0, y0, x1, y1 = [_safe_int(v) for v in bbox]
    x0 = _clamp(x0, 0, max(0, width))
    x1 = _clamp(x1, 0, max(0, width))
    y0 = _clamp(y0, 0, max(0, height))
    y1 = _clamp(y1, 0, max(0, height))
    if x1 <= x0 or y1 <= y0:
        return [0, 0, width, height]
    return [x0, y0, x1, y1]


def _sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(candidate.get("person_score", 0.0)),
        float(candidate.get("area_ratio", 0.0)),
        float(candidate.get("confidence", 0.0)),
    )


def score_person_candidate(candidate: Dict[str, Any], image_width: int, image_height: int) -> float:
    bbox = _normalize_bbox(candidate.get("bbox") or [0, 0, image_width, image_height], image_width, image_height)
    x0, y0, x1, y1 = bbox
    area_ratio = float(candidate.get("area_ratio") or 0.0)
    confidence = float(candidate.get("confidence") or 0.0)
    center_x, center_y = _bbox_center(bbox)

    width_ratio = max(1.0, float(image_width))
    height_ratio = max(1.0, float(image_height))
    center_x_norm = 1.0 - min(abs((center_x / width_ratio) - 0.5) / 0.5, 1.0)
    center_y_norm = 1.0 - min(abs((center_y / height_ratio) - 0.55) / 0.55, 1.0)
    aspect_ratio = max(1.0, float(x1 - x0)) / max(1.0, float(y1 - y0))
    shape_bonus = 1.0 - min(abs(aspect_ratio - 0.45), 0.8)

    score = (
        (0.58 * area_ratio)
        + (0.18 * confidence)
        + (0.14 * center_x_norm)
        + (0.10 * center_y_norm)
        + (0.04 * shape_bonus)
    )
    return float(max(0.0, min(1.0, score)))


def _coerce_prepared_candidate(raw_candidate: Dict[str, Any], image_width: int, image_height: int) -> PreparedCandidate:
    bbox = _normalize_bbox(raw_candidate.get("bbox") or [0, 0, image_width, image_height], image_width, image_height)
    confidence = float(raw_candidate.get("confidence") or 0.0)
    area_ratio = float(raw_candidate.get("area_ratio") or 0.0)
    if area_ratio <= 0.0:
        area_ratio = float(_bbox_area(bbox)) / float(max(1, image_width * image_height))
    person_score = float(raw_candidate.get("person_score") or 0.0)
    if person_score <= 0.0:
        person_score = score_person_candidate(raw_candidate, image_width, image_height)
    source = str(raw_candidate.get("source") or "detector")
    return PreparedCandidate(
        bbox=bbox,
        confidence=confidence,
        area_ratio=area_ratio,
        person_score=person_score,
        source=source,
    )


def _coerce_face_candidate(raw_candidate: Dict[str, Any], image_width: int, image_height: int) -> FaceCandidate:
    bbox = _normalize_bbox(raw_candidate.get("bbox") or [0, 0, image_width, image_height], image_width, image_height)
    confidence = float(raw_candidate.get("confidence") or raw_candidate.get("score") or 0.0)
    area_ratio = float(raw_candidate.get("area_ratio") or 0.0)
    if area_ratio <= 0.0:
        area_ratio = float(_bbox_area(bbox)) / float(max(1, image_width * image_height))
    source = str(raw_candidate.get("source") or "face_detector")
    return FaceCandidate(
        bbox=bbox,
        confidence=confidence,
        area_ratio=area_ratio,
        source=source,
    )


def _extract_candidates_from_yolo_output(outputs: Any, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    if outputs is None:
        return []

    if isinstance(outputs, dict):
        if "candidates" in outputs and isinstance(outputs["candidates"], list):
            return [
                _coerce_prepared_candidate(candidate, image_width, image_height).__dict__
                for candidate in outputs["candidates"]
                if isinstance(candidate, dict)
            ]
        if "bbox" in outputs:
            return [_coerce_prepared_candidate(outputs, image_width, image_height).__dict__]

    if isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], dict):
        return [
            _coerce_prepared_candidate(candidate, image_width, image_height).__dict__
            for candidate in outputs
            if isinstance(candidate, dict)
        ]

    results = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    candidates: List[Dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        cls = getattr(boxes, "cls", None)
        if xyxy is None:
            continue
        try:
            xyxy_arr = xyxy.detach().cpu().numpy()
        except Exception:
            try:
                xyxy_arr = xyxy.cpu().numpy()
            except Exception:
                xyxy_arr = xyxy
        try:
            conf_arr = conf.detach().cpu().numpy() if conf is not None else None
        except Exception:
            try:
                conf_arr = conf.cpu().numpy() if conf is not None else None
            except Exception:
                conf_arr = conf
        try:
            cls_arr = cls.detach().cpu().numpy() if cls is not None else None
        except Exception:
            try:
                cls_arr = cls.cpu().numpy() if cls is not None else None
            except Exception:
                cls_arr = cls

        for idx, row in enumerate(xyxy_arr):
            bbox = _normalize_bbox(row[:4], image_width, image_height)
            confidence = float(conf_arr[idx]) if conf_arr is not None and len(conf_arr) > idx else 0.0
            class_id = int(cls_arr[idx]) if cls_arr is not None and len(cls_arr) > idx else 0
            if class_id != 0:
                continue
            candidate = {
                "bbox": bbox,
                "confidence": confidence,
                "area_ratio": float(_bbox_area(bbox)) / float(max(1, image_width * image_height)),
                "source": "detector",
            }
            candidate["person_score"] = score_person_candidate(candidate, image_width, image_height)
            candidates.append(candidate)

    return candidates


def detect_person_candidates(
    image: Image.Image,
    detector_fn: Optional[Callable[..., Any]] = None,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(image, Image.Image):
        return [], {"count": 0, "reason": "invalid_image"}

    width, height = image.size
    if width <= 0 or height <= 0:
        return [], {"count": 0, "reason": "invalid_image_size"}

    if detector_fn is None:
        return [], {"count": 0, "reason": "person_detector_unavailable"}

    try:
        try:
            outputs = detector_fn(image, conf=conf, iou=iou)
        except TypeError:
            outputs = detector_fn(image)
    except Exception as exc:
        return [], {"count": 0, "reason": "person_detector_failed", "error": str(exc)}

    candidates = _extract_candidates_from_yolo_output(outputs, width, height)
    enriched = []
    for candidate in candidates:
        bbox = _normalize_bbox(candidate.get("bbox") or [0, 0, width, height], width, height)
        area_ratio = float(_bbox_area(bbox)) / float(max(1, width * height))
        enriched.append(
            {
                "bbox": bbox,
                "confidence": float(candidate.get("confidence") or 0.0),
                "area_ratio": area_ratio,
                "person_score": float(candidate.get("person_score") or score_person_candidate(candidate, width, height)),
                "source": str(candidate.get("source") or "detector"),
            }
        )

    enriched.sort(key=_sort_key, reverse=True)
    meta = {
        "count": len(enriched),
        "image_size": {"width": width, "height": height},
        "top_scores": [
            {
                "bbox": item["bbox"],
                "confidence": round(float(item["confidence"]), 4),
                "area_ratio": round(float(item["area_ratio"]), 4),
                "person_score": round(float(item["person_score"]), 4),
                "source": item["source"],
            }
            for item in enriched[:3]
        ],
    }
    return enriched, meta


def _bbox_iou_simple(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter_w = max(0, ix1 - ix0)
    inter_h = max(0, iy1 - iy0)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _opencv_face_detector(image: Image.Image) -> List[Dict[str, Any]]:
    if cv2 is None or np is None or not isinstance(image, Image.Image):
        return []

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    candidates: List[Dict[str, Any]] = []
    cascade_names = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    )
    for cascade_name in cascade_names:
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
            if cascade.empty():
                continue
            rects = cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(20, 20),
            )
        except Exception:
            continue

        for rect in rects:
            try:
                x, y, w, h = [int(v) for v in rect]
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            candidates.append(
                {
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.9,
                    "source": "opencv",
                }
            )

    if not candidates:
        return []

    deduped: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda c: float(_bbox_area(c.get("bbox") or [0, 0, 0, 0])), reverse=True):
        bbox = candidate.get("bbox") or [0, 0, 0, 0]
        if any(_bbox_iou_simple(bbox, chosen.get("bbox") or [0, 0, 0, 0]) >= 0.35 for chosen in deduped):
            continue
        deduped.append(candidate)
    return deduped


def detect_face_candidates(
    image: Image.Image,
    detector_fn: Optional[Callable[..., Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(image, Image.Image):
        return [], {"count": 0, "reason": "invalid_image"}

    width, height = image.size
    if width <= 0 or height <= 0:
        return [], {"count": 0, "reason": "invalid_image_size"}

    if detector_fn is None:
        return [], {"count": 0, "reason": "face_detector_unavailable"}

    try:
        try:
            outputs = detector_fn(image)
        except TypeError:
            outputs = detector_fn(image, conf=0.25)
    except Exception as exc:
        return [], {"count": 0, "reason": "face_detector_failed", "error": str(exc)}

    if outputs is None:
        return [], {"count": 0, "reason": "no_face_detected"}

    if isinstance(outputs, dict):
        if "candidates" in outputs and isinstance(outputs["candidates"], list):
            raw_candidates = [item for item in outputs["candidates"] if isinstance(item, dict)]
        else:
            raw_candidates = [outputs]
    elif isinstance(outputs, (list, tuple)):
        raw_candidates = [item for item in outputs if isinstance(item, dict)]
        if not raw_candidates and outputs and isinstance(outputs[0], (list, tuple)) and len(outputs[0]) >= 4:
            raw_candidates = [{"bbox": list(item)} for item in outputs if isinstance(item, (list, tuple))]
    else:
        raw_candidates = []

    candidates = [
        _coerce_face_candidate(candidate, width, height).__dict__
        for candidate in raw_candidates
        if isinstance(candidate, dict)
    ]
    candidates.sort(key=lambda c: float(c.get("area_ratio", 0.0)), reverse=True)
    meta = {
        "count": len(candidates),
        "image_size": {"width": width, "height": height},
        "top_scores": [
            {
                "bbox": item["bbox"],
                "confidence": round(float(item["confidence"]), 4),
                "area_ratio": round(float(item["area_ratio"]), 4),
                "source": item["source"],
            }
            for item in candidates[:3]
        ],
    }
    return candidates, meta


def select_main_face_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: float(c.get("area_ratio", 0.0)), reverse=True)[0]


def assess_full_body_visibility(
    image: Image.Image,
    person_candidate: Dict[str, Any],
    face_candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(image, Image.Image):
        return {"visible": False, "reason": "invalid_image"}

    width, height = image.size
    bbox = _normalize_bbox(person_candidate.get("bbox") or [0, 0, width, height], width, height)
    x0, y0, x1, y1 = bbox
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    bbox_area = float(_bbox_area(bbox))
    image_area = float(max(1, width * height))
    bbox_area_ratio = bbox_area / image_area

    if face_candidate is None:
        return {
            "visible": False,
            "reason": "face_required_for_full_body_check",
            "bbox_area_ratio": bbox_area_ratio,
        }

    face_bbox = _normalize_bbox(face_candidate.get("bbox") or [0, 0, width, height], width, height)
    fx0, fy0, fx1, fy1 = face_bbox
    face_area_ratio = float(_bbox_area(face_bbox)) / float(max(1, _bbox_area(bbox)))
    face_center_x = (fx0 + fx1) / 2.0
    face_center_y = (fy0 + fy1) / 2.0
    face_rel_x = (face_center_x - x0) / float(bbox_w)
    face_rel_y = (face_center_y - y0) / float(bbox_h)
    major_axis = "vertical" if bbox_h >= bbox_w else "horizontal"
    if major_axis == "vertical":
        face_end_distance = min(face_rel_y, 1.0 - face_rel_y)
    else:
        face_end_distance = min(face_rel_x, 1.0 - face_rel_x)

    reasons: List[str] = []
    if bbox_area_ratio < 0.10:
        reasons.append("person_too_small")
    if major_axis == "vertical":
        body_extent_ratio = bbox_h / float(max(1, height))
    else:
        body_extent_ratio = bbox_w / float(max(1, width))
    if body_extent_ratio < 0.55:
        reasons.append("body_too_small")
    if not (0.0 <= face_rel_x <= 1.0 and 0.0 <= face_rel_y <= 1.0):
        reasons.append("face_not_inside_body")
    if face_area_ratio < 0.02:
        reasons.append("face_too_small")
    if face_area_ratio > 0.12:
        reasons.append("face_too_large_for_full_body")
    if major_axis == "vertical" and face_rel_y > 0.28:
        reasons.append("face_too_low_in_body")
    if face_end_distance > 0.60:
        reasons.append("face_not_at_body_end")

    visible = not reasons
    return {
        "visible": visible,
        "reason": "ok" if visible else ";".join(reasons),
        "bbox": bbox,
        "bbox_area_ratio": round(bbox_area_ratio, 4),
        "face_area_ratio": round(face_area_ratio, 4),
        "major_axis": major_axis,
        "body_extent_ratio": round(body_extent_ratio, 4),
        "face_end_distance": round(face_end_distance, 4),
        "face_rel_x": round(face_rel_x, 4),
        "face_rel_y": round(face_rel_y, 4),
    }


def has_multiple_prominent_people(candidates: List[Dict[str, Any]]) -> bool:
    if len(candidates) < 2:
        return False

    ranked = sorted(candidates, key=_sort_key, reverse=True)
    top = ranked[0]
    second = ranked[1]
    top_score = float(top.get("person_score", 0.0))
    second_score = float(second.get("person_score", 0.0))
    top_area = float(top.get("area_ratio", 0.0))
    second_area = float(second.get("area_ratio", 0.0))

    return (
        second_score >= max(0.16, top_score * 0.82)
        and second_area >= max(0.06, top_area * 0.58)
    )


def select_main_person_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return sorted(candidates, key=_sort_key, reverse=True)[0]


def crop_main_person(image: Image.Image, candidate: Optional[Dict[str, Any]] = None) -> Tuple[Image.Image, List[int]]:
    if not isinstance(image, Image.Image):
        raise ValueError("Expected a PIL image")
    width, height = image.size
    bbox = _normalize_bbox((candidate or {}).get("bbox") or [0, 0, width, height], width, height)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return image, [0, 0, width, height]
    return image.crop((x0, y0, x1, y1)), bbox


def focus_score(image: Image.Image, max_edge: int = 1024) -> float:
    if not isinstance(image, Image.Image):
        return 0.0
    return float(validate_image_quality(image, max_edge=max_edge))


def _clean_jsonish_text(raw_text: str) -> str:
    text = " ".join(str(raw_text or "").split()).strip()
    if not text:
        return ""
    if "```" in text:
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def parse_verifier_response(raw_text: Any) -> Dict[str, Any]:
    def _coerce_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value or "").strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
        return None

    def _normalize_verdict(verdict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(verdict, dict):
            return {}
        out = dict(verdict)
        for key in (
            "single_person",
            "face_visible",
            "upper_body_visible",
            "lower_body_visible",
            "clear_human",
        ):
            parsed = _coerce_bool(out.get(key))
            if parsed is not None:
                out[key] = parsed
        # Backward compatibility with older verifier schema.
        full_body_visible = _coerce_bool(out.get("full_body_visible"))
        legs_visible = _coerce_bool(out.get("legs_70_visible"))
        if _coerce_bool(out.get("upper_body_visible")) is None and full_body_visible is not None:
            out["upper_body_visible"] = bool(full_body_visible)
        if _coerce_bool(out.get("lower_body_visible")) is None:
            if full_body_visible is not None:
                out["lower_body_visible"] = bool(full_body_visible)
            elif legs_visible is not None:
                out["lower_body_visible"] = bool(legs_visible)
        return out

    if isinstance(raw_text, dict):
        return _normalize_verdict(raw_text)

    text = _clean_jsonish_text(str(raw_text or ""))
    if not text:
        return {}

    json_blob = None
    match = re.search(r"\{.*\}", text)
    if match:
        json_blob = match.group(0)
        try:
            parsed = json.loads(json_blob)
            if isinstance(parsed, dict):
                return _normalize_verdict(parsed)
        except Exception:
            pass

    # Fallback for terse yes/no style answers.
    low = text.lower()
    parsed = {
        "single_person": "single" in low and ("true" in low or "yes" in low),
        "face_visible": "face" in low and ("true" in low or "yes" in low),
        "upper_body_visible": ("upper body" in low or "torso" in low) and ("true" in low or "yes" in low),
        "lower_body_visible": ("lower body" in low or "legs" in low or "hip" in low or "thigh" in low) and ("true" in low or "yes" in low),
        "clear_human": "human" in low and ("true" in low or "yes" in low),
        "reason": text[:160],
        "_raw": text,
    }
    return _normalize_verdict(parsed)


def build_verification_prompt(
    *,
    candidates: List[Dict[str, Any]],
    primary_candidate: Optional[Dict[str, Any]],
) -> str:
    top = primary_candidate or (candidates[0] if candidates else {})
    bbox = top.get("bbox") or []
    return (
        DEFAULT_VERIFICATION_PROMPT
        + f" The main subject bbox is {bbox}. "
        + "Pay attention to whether the face is clearly visible, whether upper and lower body sections are visible, "
        + "and whether there is only one dominant foreground person."
    )


def _call_optional_image_text_fn(
    fn: Optional[Callable[..., Any]],
    image: Image.Image,
    prompt: Optional[str] = None,
) -> Any:
    if fn is None:
        return ""
    try:
        if prompt is None:
            result = fn(image)
        else:
            result = fn(image, prompt)
    except TypeError:
        if prompt is None:
            result = fn(image)
        else:
            result = fn(image)
    except Exception:
        return ""
    if isinstance(result, str):
        return " ".join(result.split()).strip()
    return result


def _prepare_image_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _error_payload(error: str, message: str, *, meta: Optional[Dict[str, Any]] = None, status_code: int = 422) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status_code": int(status_code),
        "error": error,
        "message": message,
    }
    if meta is not None:
        payload["meta"] = meta
    return payload


def prepare_user_image_core(
    image: Image.Image,
    *,
    grounding_detector_fn: Optional[Callable[..., Any]] = None,
    verifier_fn: Optional[Callable[..., Any]] = None,
    description_fn: Optional[Callable[..., Any]] = None,
    fallback_description_fn: Optional[Callable[..., Any]] = None,
    upload_fn: Optional[Callable[[bytes], str]] = None,
    blur_check_enabled: bool = False,
    blur_min_focus_score: float = 22.0,
    blur_focus_max_edge: int = 1024,
    verification_required: bool = True,
) -> Dict[str, Any]:
    if not isinstance(image, Image.Image):
        return _error_payload("invalid_image", "Expected a valid image.", status_code=422)

    width, height = image.size
    candidates: List[Dict[str, Any]] = []
    detect_meta: Dict[str, Any] = {"backend": "none", "count": 0}
    primary_candidate: Optional[Dict[str, Any]] = None

    if grounding_detector_fn is None:
        return _error_payload(
            "detector_unavailable",
            "GroundingDINO detector is unavailable.",
            meta={"detect": {"backend": "grounding_dino", "reason": "detector_fn_missing"}},
            status_code=503,
        )

    grounding_gate = detect_grounding_visibility(image, detector_fn=grounding_detector_fn)
    detect_meta = dict(grounding_gate.get("meta") or {})
    detect_meta.setdefault("backend", "grounding_dino")
    if not bool(grounding_gate.get("ok", False)):
        return _error_payload(
            str(grounding_gate.get("error") or "detection_failed"),
            str(grounding_gate.get("message") or "Image detection failed."),
            meta={"detect": detect_meta},
        )
    primary_candidate = dict(grounding_gate.get("primary_person") or {})
    primary_candidate["bbox"] = _normalize_bbox(primary_candidate.get("bbox") or [0, 0, width, height], width, height)
    candidates = [primary_candidate]
    detect_meta["count"] = 1

    if primary_candidate is None:
        return _error_payload(
            "no_person",
            "No person detected in the image.",
            meta={"detect": detect_meta},
        )
    crop, bbox = crop_main_person(image, primary_candidate)
    focus = focus_score(crop, max_edge=blur_focus_max_edge)
    if blur_check_enabled and focus < float(blur_min_focus_score):
        return _error_payload(
            "low_quality",
            "The image is too blurry. Please upload a sharper photo.",
            meta={
                "detect": detect_meta,
                "focusScore": round(float(focus), 4),
                "threshold": float(blur_min_focus_score),
            },
        )

    face_meta: Dict[str, Any] = {"enabled": True, "backend": "grounding_dino"}

    verification_meta: Dict[str, Any] = {"enabled": False, "backend": "none"}
    if verifier_fn is None and verification_required:
        return _error_payload(
            "verifier_unavailable",
            "Image eligibility verification is unavailable. Please try again later.",
            meta={
                "detect": detect_meta,
                "face": face_meta,
                "focusScore": round(float(focus), 4),
            },
            status_code=503,
        )

    if verifier_fn is not None:
        verification_meta["enabled"] = True
        prompt = build_verification_prompt(candidates=candidates, primary_candidate=primary_candidate)
        raw_verdict = _call_optional_image_text_fn(verifier_fn, image, prompt)
        verdict = parse_verifier_response(raw_verdict)
        verification_meta.update(
            {
                "backend": str(verdict.get("backend") or "vlm"),
                "raw": raw_verdict,
                "parsed": verdict,
            }
        )
        if not verdict:
            return _error_payload(
                "verification_failed",
                "Could not verify the image. Please upload a clearer photo.",
                meta={"detect": detect_meta, "verification": verification_meta},
            )

        if not bool(verdict.get("single_person", True)):
            return _error_payload(
                "multiple_people",
                str(verdict.get("reason") or "Multiple people detected."),
                meta={"detect": detect_meta, "verification": verification_meta},
            )
        if not bool(verdict.get("face_visible", True)):
            return _error_payload(
                "face_hidden",
                str(verdict.get("reason") or "Face not clearly visible."),
                meta={"detect": detect_meta, "verification": verification_meta},
            )
        if not bool(verdict.get("upper_body_visible", True)):
            return _error_payload(
                "top_section_not_visible",
                str(verdict.get("reason") or "Upper body is not clearly visible."),
                meta={"detect": detect_meta, "verification": verification_meta},
            )
        if not bool(verdict.get("lower_body_visible", True)):
            return _error_payload(
                "bottom_section_not_visible",
                str(verdict.get("reason") or "Lower body is not clearly visible."),
                meta={"detect": detect_meta, "verification": verification_meta},
            )
        if not bool(verdict.get("clear_human", True)):
            return _error_payload(
                "not_a_clear_human",
                str(verdict.get("reason") or "Human subject is not clear."),
                meta={"detect": detect_meta, "verification": verification_meta},
            )

    if upload_fn is None:
        return _error_payload(
            "upload_unavailable",
            "Upload function is unavailable.",
            meta={"detect": detect_meta, "face": face_meta, "verification": verification_meta},
            status_code=503,
        )

    prepared_bytes = _prepare_image_bytes(image)
    try:
        url = str(upload_fn(prepared_bytes))
    except Exception as exc:
        return _error_payload(
            "upload_failed",
            f"Failed to upload prepared image: {exc}",
            meta={"detect": detect_meta, "face": face_meta, "verification": verification_meta},
            status_code=500,
        )

    description_raw = str(_call_optional_image_text_fn(description_fn, image) or "").strip()
    if not description_raw:
        description_raw = str(_call_optional_image_text_fn(description_fn, crop) or "").strip()
    if not description_raw and fallback_description_fn is not None:
        try:
            description_raw = str(fallback_description_fn(image)).strip()
        except Exception:
            description_raw = ""
    if not description_raw:
        description_raw = DEFAULT_USER_DESCRIPTION

    prompt_description, worn_types = _extract_user_prepare_prompt_bundle(description_raw)

    return {
        "url": url,
        "promptDescription": prompt_description,
        "wornTypes": worn_types,
        "minicpmOutput": {
            "garments": worn_types,
            "prompt": prompt_description,
        },
        "focusScore": float(focus),
        "meta": {
            "person_bbox": bbox,
            "analysis_crop_size": {"width": int(crop.width), "height": int(crop.height)},
            "prepared_image_size": {"width": int(width), "height": int(height)},
            "prepared_image_mode": "original_full_frame",
            "detect": detect_meta,
            "face": face_meta,
            "verification": verification_meta,
            "background": {"enabled": False, "backend": "none"},
        },
    }


async def prepare_user_image_pipeline(
    upload: Any,
    *,
    grounding_detector_fn: Optional[Callable[..., Any]] = None,
    verifier_fn: Optional[Callable[..., Any]] = None,
    description_fn: Optional[Callable[..., Any]] = None,
    fallback_description_fn: Optional[Callable[..., Any]] = None,
    upload_fn: Optional[Callable[[bytes], str]] = None,
    blur_check_enabled: bool = False,
    blur_min_focus_score: float = 22.0,
    blur_focus_max_edge: int = 1024,
    verification_required: bool = True,
) -> Dict[str, Any]:
    if upload is None or not hasattr(upload, "read"):
        return _error_payload("invalid_upload", "UserImageService expects an uploaded file.", status_code=422)

    try:
        payload = await upload.read()
    except Exception as exc:
        return _error_payload("upload_read_failed", f"Failed to read upload: {exc}", status_code=422)

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError:
        return _error_payload("invalid_image", "The uploaded file is not a valid image.", status_code=422)
    except Exception as exc:
        return _error_payload("invalid_image", f"Failed to decode image: {exc}", status_code=422)

    return prepare_user_image_core(
        image,
        grounding_detector_fn=grounding_detector_fn,
        verifier_fn=verifier_fn,
        description_fn=description_fn,
        fallback_description_fn=fallback_description_fn,
        upload_fn=upload_fn,
        blur_check_enabled=blur_check_enabled,
        blur_min_focus_score=blur_min_focus_score,
        blur_focus_max_edge=blur_focus_max_edge,
        verification_required=verification_required,
    )
