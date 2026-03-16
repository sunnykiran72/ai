"""
Scoring utilities for the fashion analysis system.

This module contains functions for calculating various scores used in ranking,
filtering, and evaluating detection results and user inputs.
"""

import os
from typing import List, Dict, Any, Optional, Tuple

from utils.validation import normalize_garment_type

# Configuration constants
HYBRID_WEIGHT_YOLO = float(os.getenv("HYBRID_WEIGHT_YOLO", "0.45"))
HYBRID_WEIGHT_FLORENCE = float(os.getenv("HYBRID_WEIGHT_FLORENCE", "0.45"))
HYBRID_WEIGHT_BBOX = float(os.getenv("HYBRID_WEIGHT_BBOX", "0.10"))
USER_PREP_MIN_PROMPT_WORDS = max(4, int(os.getenv("USER_PREP_MIN_PROMPT_WORDS", "10")))

# Term lists for user preparation scoring
USER_PREP_IDENTITY_TERMS = (
    "identity", "face", "facial", "hair", "skin", "complexion", "body", "build", "shape",
    "pose", "posture", "standing", "sitting", "hand", "arm", "leg", "eyes", "nose", "mouth",
    "jaw", "lighting", "framing", "crop", "camera", "angle", "occlusion", "silhouette",
)

USER_PREP_BACKGROUND_TERMS = (
    "background", "backdrop", "wall", "door", "floor", "room", "studio", "field", "flowers",
    "street", "outdoor", "indoor", "furniture", "chair", "sofa", "window",
)

USER_PREP_APPAREL_TERMS = (
    "wearing", "wears", "outfit", "clothing", "garment", "dress", "gown", "top", "shirt",
    "blouse", "jacket", "coat", "pants", "trousers", "jeans", "skirt", "shorts", "shoe",
    "shoes", "boot", "boots", "sandal", "sandals", "sneaker", "sneakers", "heel", "heels",
)


def hybrid_score(yolo_conf: float, florence_conf: float, bbox_prior: float) -> float:
    """
    Calculate hybrid confidence score from multiple detection sources.
    
    Args:
        yolo_conf: YOLO detection confidence (0.0 to 1.0)
        florence_conf: Florence detection confidence (0.0 to 1.0)
        bbox_prior: Bounding box prior score (0.0 to 1.0)
        
    Returns:
        Weighted hybrid score (0.0 to 1.0)
        
    Examples:
        >>> hybrid_score(0.8, 0.7, 0.6)
        0.735
        >>> hybrid_score(0.9, 0.5, 0.8)
        0.785
    """
    score = (
        (HYBRID_WEIGHT_YOLO * yolo_conf)
        + (HYBRID_WEIGHT_FLORENCE * florence_conf)
        + (HYBRID_WEIGHT_BBOX * bbox_prior)
    )
    return max(0.0, min(1.0, score))


def rank_items(items: List[Dict[str, Any]], reverse: bool = True) -> List[Dict[str, Any]]:
    """
    Rank items by their confidence scores.
    
    Args:
        items: List of item dictionaries with confidence information
        reverse: If True, rank highest scores first
        
    Returns:
        Sorted list of items
    """
    return sorted(items, key=item_rank_by_score, reverse=reverse)


def item_rank_by_score(item: Dict[str, Any]) -> float:
    """
    Extract ranking score from item dictionary.
    
    Args:
        item: Item dictionary with confidence information
        
    Returns:
        Ranking score for sorting
    """
    conf = item.get("confidence", {})
    if isinstance(conf, dict):
        return float(conf.get("hybrid", conf.get("yolo", 0.0)))
    return 0.0


def filter_by_min_score(items: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
    """
    Filter items by minimum confidence score.
    
    Args:
        items: List of item dictionaries
        min_score: Minimum score threshold
        
    Returns:
        Filtered list of items above threshold
    """
    return [item for item in items if item_rank_by_score(item) >= min_score]


def calculate_detection_confidence(
    yolo_confidence: float,
    florence_confidence: float,
    bbox: List[int],
    image_width: int,
    image_height: int,
) -> Dict[str, float]:
    """
    Calculate comprehensive detection confidence metrics.
    
    Args:
        yolo_confidence: YOLO detection confidence
        florence_confidence: Florence detection confidence
        bbox: Bounding box as [x0, y0, x1, y1]
        image_width: Image width
        image_height: Image height
        
    Returns:
        Dictionary with confidence metrics
    """
    bbox_prior = calculate_bbox_prior(bbox, image_width, image_height)
    hybrid = hybrid_score(yolo_confidence, florence_confidence, bbox_prior)
    
    return {
        "yolo": float(yolo_confidence),
        "florence": float(florence_confidence),
        "bbox_prior": float(bbox_prior),
        "hybrid": float(hybrid),
    }


def calculate_bbox_prior(bbox: List[int], width: int, height: int) -> float:
    """
    Calculate bounding box prior score based on size and position.
    
    Args:
        bbox: Bounding box as [x0, y0, x1, y1]
        width: Image width
        height: Image height
        
    Returns:
        Prior score (0.0 to 1.0)
        
    Examples:
        >>> calculate_bbox_prior([100, 100, 200, 200], 400, 400)
        0.625
    """
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    area_ratio = min(1.0, (bw * bh) / max(1.0, float(width * height)))

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    nx = abs(cx - (width / 2.0)) / max(1.0, (width / 2.0))
    ny = abs(cy - (height / 2.0)) / max(1.0, (height / 2.0))
    center_score = max(0.0, 1.0 - ((nx + ny) / 2.0))
    
    return (0.7 * area_ratio) + (0.3 * center_score)


def score_user_prep_face_candidate(
    candidate: Dict[str, Any],
    image_width: int,
    image_height: int,
    person_bbox: Optional[List[int]] = None
) -> float:
    """
    Score face candidate for user preparation.
    
    Args:
        candidate: Face candidate dictionary with bbox and metadata
        image_width: Image width
        image_height: Image height
        person_bbox: Optional person bounding box for context
        
    Returns:
        Face candidate score
    """
    bbox = [int(v) for v in (candidate.get("bbox") or [0, 0, 0, 0])]
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    area_ratio = float(candidate.get("area_ratio", 0.0))
    aspect_score = min(fw, fh) / float(max(fw, fh))

    if person_bbox:
        px0, py0, px1, py1 = [int(v) for v in person_bbox]
        ph = max(1, py1 - py0)
        rel_cy = (cy - py0) / float(ph)
    else:
        rel_cy = cy / float(max(1, image_height))

    top_prior = max(0.0, 1.0 - rel_cy)
    source_bonus = 0.12 if str(candidate.get("source") or "") == "parser_face" else 0.0
    
    return (0.55 * area_ratio) + (0.25 * top_prior) + (0.20 * aspect_score) + source_bonus


def score_user_prepare_prompt_description(text: str) -> float:
    """
    Score user preparation prompt description quality.
    
    Args:
        text: Prompt description text
        
    Returns:
        Quality score (higher is better)
    """
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return -1e6

    low = normalized.lower()
    score = float(len(normalized.split()))
    
    # Bonus for identity terms
    score += 4.0 * sum(1 for term in USER_PREP_IDENTITY_TERMS if term in low)
    
    # Penalty for background terms
    score -= 6.0 * sum(1 for term in USER_PREP_BACKGROUND_TERMS if term in low)
    
    # Penalty for apparel terms
    score -= 3.0 * sum(1 for term in USER_PREP_APPAREL_TERMS if term in low)
    
    # Penalty for too short descriptions
    if len(normalized.split()) < USER_PREP_MIN_PROMPT_WORDS:
        score -= 20.0
    
    return score


def score_requested_type_geometry(
    item: Dict[str, Any],
    requested_type: str,
    image_height: int
) -> float:
    """
    Score item based on geometry match with requested garment type.
    
    Args:
        item: Item dictionary with bbox and confidence
        requested_type: Requested garment type
        image_height: Image height for normalization
        
    Returns:
        Geometry-adjusted score
    """
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or image_height <= 0:
        return item_rank_by_score(item)
    
    _, y0, _, y1 = [int(v) for v in bbox]
    box_h = max(1, y1 - y0)
    center_y = y0 + (box_h / 2.0)
    center_ratio = float(center_y) / float(image_height)
    height_ratio = float(box_h) / float(image_height)
    base = item_rank_by_score(item)
    req = normalize_garment_type(requested_type)

    if req in {"top", "outer"}:
        # Prefer upper-body candidates with smaller center_y and meaningful height.
        return base + (1.0 - center_ratio) + (0.25 * min(height_ratio, 0.6))
    elif req == "bottom":
        # Prefer lower-body candidates with larger center_y.
        return base + center_ratio + (0.15 * min(height_ratio, 0.75))
    elif req == "dress":
        # Prefer tall full-body candidates spanning most of the frame.
        return base + (1.5 * height_ratio) - abs(center_ratio - 0.52)
    
    return base


def calculate_yolo_support_score(candidate_bbox: List[int], yolo_boxes: List[Dict[str, Any]]) -> float:
    """
    Calculate support score from YOLO detections for a candidate bbox.
    
    Args:
        candidate_bbox: Candidate bounding box as [x0, y0, x1, y1]
        yolo_boxes: List of YOLO detection boxes with confidence
        
    Returns:
        Support score (0.0 to 1.0)
    """
    if not yolo_boxes:
        return 0.0
    
    cb = tuple(int(v) for v in candidate_bbox)
    best = 0.0
    
    for y in yolo_boxes:
        yb = tuple(int(v) for v in (y.get("bbox") or [0, 0, 0, 0]))
        iou = calculate_bbox_iou(cb, yb)
        yc = float(y.get("confidence", 0.0))
        support = max(0.0, min(1.0, (0.65 * iou) + (0.35 * yc)))
        if support > best:
            best = support
    
    return float(max(0.0, min(1.0, best)))


def dedupe_items_by_iou(items: List[Dict[str, Any]], iou_threshold: float = 0.60) -> List[Dict[str, Any]]:
    """
    Remove duplicate items based on IoU threshold.
    
    Args:
        items: List of item dictionaries with bbox and type
        iou_threshold: IoU threshold for considering items duplicates
        
    Returns:
        Deduplicated list of items
    """
    if not items:
        return []
    
    ranked = sorted(items, key=item_rank_by_score, reverse=True)
    kept: List[Dict[str, Any]] = []
    
    for candidate in ranked:
        cb = candidate.get("bbox")
        ctype = candidate.get("type")
        duplicate = False
        
        for existing in kept:
            eb = existing.get("bbox")
            if ctype != existing.get("type"):
                continue
            if not cb or not eb:
                continue
            if calculate_bbox_iou(tuple(cb), tuple(eb)) >= iou_threshold:
                duplicate = True
                break
        
        if not duplicate:
            kept.append(candidate)
    
    return kept


def find_largest_instance(instances: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the largest instance by area.
    
    Args:
        instances: List of instance dictionaries with bbox
        
    Returns:
        Largest instance or None if list is empty
    """
    if not instances:
        return None
    return max(instances, key=calculate_instance_area)


def calculate_instance_area(instance: Dict[str, Any]) -> int:
    """
    Calculate area of an instance from its bounding box.
    
    Args:
        instance: Instance dictionary with bbox
        
    Returns:
        Area in pixels
    """
    bbox = instance.get("bbox") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0
    
    try:
        x0, y0, x1, y1 = [int(v) for v in bbox]
    except Exception:
        return 0
    
    return max(1, x1 - x0) * max(1, y1 - y0)


def calculate_bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        bbox1: First bounding box as (x0, y0, x1, y1)
        bbox2: Second bounding box as (x0, y0, x1, y1)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return float(inter_area) / float(union_area)


def score_adaptive_crop_caption(caption: str, target_type: str) -> Tuple[float, Dict[str, Any]]:
    """
    Score adaptive crop caption quality for target type.
    
    Args:
        caption: Caption text to score
        target_type: Target garment type
        
    Returns:
        Tuple of (score, metadata_dict)
    """
    text = " ".join(str(caption or "").split()).strip()
    if not text:
        return 0.0, {"reason": "empty_caption"}
    
    # This is a simplified implementation
    # The full version would have complex type inference logic
    
    base_score = len(text.split()) * 0.1  # Basic word count score
    
    # Bonus for target type mentions
    if target_type.lower() in text.lower():
        base_score += 0.5
    
    # Penalty for very short captions
    if len(text.split()) < 3:
        base_score -= 0.3
    
    metadata = {
        "word_count": len(text.split()),
        "has_target_type": target_type.lower() in text.lower(),
        "text_length": len(text),
    }
    
    return max(0.0, min(1.0, base_score)), metadata