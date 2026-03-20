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

try:  # The runtime compat helper keeps the prompt cleanup consistent with the app.
    from utils.runtime_compat import _normalize_user_prepare_api_prompt_description
except Exception:  # pragma: no cover - fallback for isolated unit tests
    def _normalize_user_prepare_api_prompt_description(raw_text: str) -> str:
        return " ".join(str(raw_text or "").split()).strip()


DEFAULT_USER_DESCRIPTION = (
    "identity: face-preservation reference for a person with unknown facial details, hairline, and age band. "
    "face: neutral expression with unchanged facial geometry and head shape. "
    "body pose: standing with upper-body orientation, arm placement, and shoulder angle preserved. "
    "lower body pose: legs, knees, feet, and stance remain in the same position. "
    "framing/lighting: centered full-body crop with even lighting. "
    "occlusion: any phone or held object remains in the same hand, same angle, and same overlap. "
    "preserve: face identity, facial geometry, body proportions, pose, hand placement, object placement, leg position, framing, lighting, and background unchanged."
)

DEFAULT_VERIFICATION_PROMPT = (
    "You are a strict image eligibility validator for a user photo upload. "
    "Return JSON only with these keys: "
    '{"single_person": true/false, "face_visible": true/false, "full_body_visible": true/false, '
    '"clear_human": true/false, "reason": "short reason"}. '
    "Rules: "
    "single_person should be true only when one dominant foreground person is present. "
    "Small background people are allowed only if they are clearly not competing with the main subject. "
    "face_visible should be false if the face is hidden, covered, turned away too much, or too small to verify. "
    "A partial face, side face, or softly occluded face can still count as visible if it is clear enough for try-on. "
    "full_body_visible should be true only if the complete person is visible in frame, regardless of pose. "
    "Standing, sitting, and lying down are allowed only when the body is not cropped and the full outline is visible. "
    "If the result is ambiguous, return false values. Do not include markdown or extra text."
)


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
    if isinstance(raw_text, dict):
        return raw_text

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
                return parsed
        except Exception:
            pass

    # Fallback for terse yes/no style answers.
    low = text.lower()
    return {
        "single_person": "single" in low and ("true" in low or "yes" in low),
        "face_visible": "face" in low and ("true" in low or "yes" in low),
        "full_body_visible": "full body" in low and ("true" in low or "yes" in low),
        "clear_human": "human" in low and ("true" in low or "yes" in low),
        "reason": text[:160],
        "_raw": text,
    }


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
        + "Pay attention to whether the face is clearly visible, the full body is visible in any pose, and whether there is only one dominant foreground person."
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
    person_detector_fn: Optional[Callable[..., Any]] = None,
    face_detector_fn: Optional[Callable[..., Any]] = None,
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
    candidates, detect_meta = detect_person_candidates(image, detector_fn=person_detector_fn)
    if not candidates:
        return _error_payload(
            "no_person",
            "No person detected in the image.",
            meta={"detect": detect_meta},
        )

    if has_multiple_prominent_people(candidates):
        return _error_payload(
            "multiple_people",
            "Multiple prominent people detected. Please upload a photo with one clearly dominant person.",
            meta={"detect": detect_meta},
        )

    primary_candidate = select_main_person_candidate(candidates)
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

    face_meta: Dict[str, Any] = {"enabled": False, "backend": "none"}
    if face_detector_fn is not None:
        face_meta["enabled"] = True
        face_candidates, face_detect_meta = detect_face_candidates(image, detector_fn=face_detector_fn)
        face_meta["backend"] = str((face_candidates[0].get("source") if face_candidates else "none") or "none")
        face_meta["detect"] = face_detect_meta
        primary_face = select_main_face_candidate(face_candidates)
        if not face_candidates or primary_face is None:
            return _error_payload(
                "face_hidden",
                "Face not clearly visible.",
                meta={
                    "detect": detect_meta,
                    "face": face_meta,
                    "focusScore": round(float(focus), 4),
                },
            )

        body_visibility = assess_full_body_visibility(image, primary_candidate, primary_face)
        face_meta["body_visibility"] = body_visibility
        if not bool(body_visibility.get("visible", False)):
            return _error_payload(
                "full_body_not_visible",
                "Full body is not visible in the image.",
                meta={
                    "detect": detect_meta,
                    "face": face_meta,
                    "focusScore": round(float(focus), 4),
                },
            )

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
        if not bool(verdict.get("full_body_visible", True)):
            return _error_payload(
                "full_body_not_visible",
                str(verdict.get("reason") or "Full body is not visible."),
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

    prompt_description = _normalize_user_prepare_api_prompt_description(description_raw)

    return {
        "url": url,
        "promptDescription": prompt_description,
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
    person_detector_fn: Optional[Callable[..., Any]] = None,
    face_detector_fn: Optional[Callable[..., Any]] = None,
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
        person_detector_fn=person_detector_fn,
        face_detector_fn=face_detector_fn,
        verifier_fn=verifier_fn,
        description_fn=description_fn,
        fallback_description_fn=fallback_description_fn,
        upload_fn=upload_fn,
        blur_check_enabled=blur_check_enabled,
        blur_min_focus_score=blur_min_focus_score,
        blur_focus_max_edge=blur_focus_max_edge,
        verification_required=verification_required,
    )
