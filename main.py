import io
import time
import logging
import asyncio
import os
import uuid
import tempfile
import re
import hashlib
from typing import Optional

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageEnhance, ImageFilter

# Load .env before importing modules that read env at import time (for example storage singleton).
load_dotenv()

# Shared Utilities
from ai.shared.azure_storage import storage
from ai.shared.image_ops import download_image, bbox_iou, binary_open, binary_close, build_soft_alpha
from ai.shared.category_mapping import wardrobe_category_from_garment_type as _wardrobe_category_from_garment_type, infer_style_from_text as _infer_style_from_text
from ai.shared.response_payloads import (
    build_error_payload as _build_error_payload,
    build_success_payload as _build_success_payload,
    build_multipart_parts as _build_multipart_parts,
    json_response as _json_response,
    multipart_form_response as _multipart_form_response,
)
from ai.shared.security import verify_bearer_token as _verify_bearer_token_impl

try:
    import jwt as jwt_module
except ImportError:
    jwt_module = None

# Modular Components
from ai.modules.vto.prompt_factory import prompt_factory
from ai.modules.vto.board_builder import BoardBuilder
from ai.modules.wardrobe.yolo_cropper import YoloCropper
from ai.modules.wardrobe.human_parser import HumanParser

# Core Models (Heavy Runners)
from ai.core.flux2_cvton_runner import Flux2CVTONRunner
from ai.core.florence_runner import FlorenceRunner
from ai.core.yolo_runner import YoloRunner
from ai.core.human_parser_runner import HumanParserRunner

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("glamify-ai")

JWT_ACCESS_SECRET = os.getenv("JWT_ACCESS_SECRET", "")

def _verify_bearer_token(authorization: Optional[str]) -> dict:
    return _verify_bearer_token_impl(
        authorization,
        jwt_access_secret=JWT_ACCESS_SECRET,
        jwt_module=jwt_module,
        logger=logger,
    )

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid integer for {name}={raw!r}. Using {default}.")
        return default

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid float for {name}={raw!r}. Using {default}.")
        return default

GPU_CONCURRENCY = max(1, _env_int("GPU_CONCURRENCY", 1))
REQUIRE_AZURE_UPLOAD = os.getenv("REQUIRE_AZURE_UPLOAD", "0") == "1"
VTO_OUTPUT_CONTAINER = (
    os.getenv("AZURE_STORAGE_VTO_OUTPUT_CONTAINER")
    or os.getenv("AZURE_STORAGE_OUTPUT_CONTAINER", "wardrobe-outputs")
)
USE_FLORENCE_HYBRID_VERIFY = os.getenv("USE_FLORENCE_HYBRID_VERIFY", "0") == "1"
USE_FLORENCE_DETAILED_PROMPT = os.getenv("USE_FLORENCE_DETAILED_PROMPT", "0") == "1"
HYBRID_TOP_K = max(1, _env_int("HYBRID_TOP_K", 3))
ANALYZE_MAX_ITEMS = max(1, _env_int("ANALYZE_MAX_ITEMS", 3))
HYBRID_MIN_SCORE = _env_float("HYBRID_MIN_SCORE", 0.0)
HYBRID_WEIGHT_YOLO = _env_float("HYBRID_WEIGHT_YOLO", 0.45)
HYBRID_WEIGHT_FLORENCE = _env_float("HYBRID_WEIGHT_FLORENCE", 0.45)
HYBRID_WEIGHT_BBOX = _env_float("HYBRID_WEIGHT_BBOX", 0.10)
ANALYZE_REQUIRE_SELECTION = os.getenv("ANALYZE_REQUIRE_SELECTION", "1") == "1"
ANALYZE_CAPTION_MODE = os.getenv("ANALYZE_CAPTION_MODE", "short").strip().lower()
ANALYZE_ENABLE_PARSER_SPLIT = os.getenv("ANALYZE_ENABLE_PARSER_SPLIT", "0") == "1"
ANALYZE_PARSER_MIN_AREA_RATIO = _env_float("ANALYZE_PARSER_MIN_AREA_RATIO", 0.015)
ANALYZE_PARSER_PAD = max(0, _env_int("ANALYZE_PARSER_PAD", 12))
ANALYZE_ENABLE_HUMAN_PARSER = os.getenv("ANALYZE_ENABLE_HUMAN_PARSER", "1") == "1"
ANALYZE_ENABLE_HEURISTIC_SPLIT = os.getenv("ANALYZE_ENABLE_HEURISTIC_SPLIT", "1") == "1"
ANALYZE_HEURISTIC_MIN_HEIGHT_RATIO = _env_float("ANALYZE_HEURISTIC_MIN_HEIGHT_RATIO", 0.78)
ANALYZE_HEURISTIC_TOP_PORTION = _env_float("ANALYZE_HEURISTIC_TOP_PORTION", 0.52)
ANALYZE_HEURISTIC_TOP_TRIM_PX = max(0, _env_int("ANALYZE_HEURISTIC_TOP_TRIM_PX", 0))
ANALYZE_HEURISTIC_BOTTOM_OVERLAP_PX = max(0, _env_int("ANALYZE_HEURISTIC_BOTTOM_OVERLAP_PX", 32))
ANALYZE_HEURISTIC_BOTTOM_OVERLAP_RATIO = _env_float("ANALYZE_HEURISTIC_BOTTOM_OVERLAP_RATIO", 0.11)
ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO = _env_float("ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO", 0.44)
ANALYZE_EXTRACT_CLOTH = os.getenv("ANALYZE_EXTRACT_CLOTH", "1") == "1"
ANALYZE_PROMPT_FROM_EXTRACTED = os.getenv("ANALYZE_PROMPT_FROM_EXTRACTED", "1") == "1"
ANALYZE_EXTRACT_MIN_MASK_RATIO = _env_float("ANALYZE_EXTRACT_MIN_MASK_RATIO", 0.01)
ANALYZE_VTON_FALLBACK_ENABLED = os.getenv("ANALYZE_VTON_FALLBACK_ENABLED", "1") == "1"
ANALYZE_VTON_CLOTH_ONLY_ENDPOINT = os.getenv("ANALYZE_VTON_CLOTH_ONLY_ENDPOINT", "").strip()
ANALYZE_VTON_TIMEOUT_S = max(10, _env_int("ANALYZE_VTON_TIMEOUT_S", 120))
ANALYZE_VTON_QUALITY_PRESET = os.getenv("ANALYZE_VTON_QUALITY_PRESET", "balanced").strip().lower()
if ANALYZE_VTON_QUALITY_PRESET not in {"balanced", "high", "max"}:
    ANALYZE_VTON_QUALITY_PRESET = "balanced"
ANALYZE_VTON_NUM_TIMESTEPS = max(10, _env_int("ANALYZE_VTON_NUM_TIMESTEPS", 28))
ANALYZE_VTON_GUIDANCE_SCALE = _env_float("ANALYZE_VTON_GUIDANCE_SCALE", 2.2)
ANALYZE_VTON_SEGMENTATION_FREE = os.getenv("ANALYZE_VTON_SEGMENTATION_FREE", "1") == "1"
ANALYZE_VTON_CUTOUT_FEATHER_PX = max(0, _env_int("ANALYZE_VTON_CUTOUT_FEATHER_PX", 2))
ANALYZE_VTON_ZOOM_PADDING_RATIO = _env_float("ANALYZE_VTON_ZOOM_PADDING_RATIO", 0.12)
ANALYZE_VTON_ALLOW_SOURCE_FALLBACK = os.getenv("ANALYZE_VTON_ALLOW_SOURCE_FALLBACK", "0") == "1"
ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH = os.getenv("ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH", "1") == "1"
ANALYZE_VTON_MAX_SOURCE_MAE = _env_float("ANALYZE_VTON_MAX_SOURCE_MAE", 2.0)
ANALYZE_VTON_USE_SHOWROOM_PERSON = os.getenv("ANALYZE_VTON_USE_SHOWROOM_PERSON", "0") == "1"
ANALYZE_VTON_SHOWROOM_PERSON_IMAGE_URL = os.getenv("ANALYZE_VTON_SHOWROOM_PERSON_IMAGE_URL", "").strip()
ANALYZE_VTON_DRESS_USE_FULL_IMAGE = os.getenv("ANALYZE_VTON_DRESS_USE_FULL_IMAGE", "1") == "1"
ANALYZE_VTON_SINGLE_ITEM_USE_FULL_IMAGE = os.getenv("ANALYZE_VTON_SINGLE_ITEM_USE_FULL_IMAGE", "1") == "1"
ANALYZE_VTON_CROP_PAD_RATIO = _env_float("ANALYZE_VTON_CROP_PAD_RATIO", 0.18)
ANALYZE_VTON_CROP_PAD_RATIO_DRESS = _env_float("ANALYZE_VTON_CROP_PAD_RATIO_DRESS", 0.28)
ANALYZE_VTON_CROP_BOTTOM_EXTRA_RATIO_DRESS = _env_float("ANALYZE_VTON_CROP_BOTTOM_EXTRA_RATIO_DRESS", 0.32)
ANALYZE_VTON_TOP_ZOOM_ENABLED = os.getenv("ANALYZE_VTON_TOP_ZOOM_ENABLED", "0") == "1"
ANALYZE_VTON_BOTTOM_UPSCALE_ENABLED = os.getenv("ANALYZE_VTON_BOTTOM_UPSCALE_ENABLED", "0") == "1"
ANALYZE_REQUIRE_EXTRACTED_PROMPT = os.getenv("ANALYZE_REQUIRE_EXTRACTED_PROMPT", "1") == "1"
ANALYZE_REQUIRE_MIRRORED_VTON_URL = os.getenv("ANALYZE_REQUIRE_MIRRORED_VTON_URL", "1") == "1"
ANALYZE_VTON_MIN_ALPHA_RATIO = _env_float("ANALYZE_VTON_MIN_ALPHA_RATIO", 0.004)
ANALYZE_VTON_MIN_RGB_STD = _env_float("ANALYZE_VTON_MIN_RGB_STD", 1.2)
ANALYZE_GARMENT_POSTPROCESS_ENABLED = os.getenv("ANALYZE_GARMENT_POSTPROCESS_ENABLED", "1") == "1"
ANALYZE_GARMENT_TARGET_ASPECT_W = max(1, _env_int("ANALYZE_GARMENT_TARGET_ASPECT_W", 2))
ANALYZE_GARMENT_TARGET_ASPECT_H = max(1, _env_int("ANALYZE_GARMENT_TARGET_ASPECT_H", 3))
ANALYZE_GARMENT_ALPHA_THRESHOLD = max(0, min(255, _env_int("ANALYZE_GARMENT_ALPHA_THRESHOLD", 12)))
ANALYZE_GARMENT_WHITE_THRESHOLD = max(200, min(255, _env_int("ANALYZE_GARMENT_WHITE_THRESHOLD", 246)))
ANALYZE_GARMENT_ENHANCE_ENABLED = os.getenv("ANALYZE_GARMENT_ENHANCE_ENABLED", "1") == "1"
ANALYZE_GARMENT_ENHANCE_SHARPNESS = _env_float("ANALYZE_GARMENT_ENHANCE_SHARPNESS", 1.22)
ANALYZE_GARMENT_ENHANCE_CONTRAST = _env_float("ANALYZE_GARMENT_ENHANCE_CONTRAST", 1.08)
ANALYZE_GARMENT_ENHANCE_COLOR = _env_float("ANALYZE_GARMENT_ENHANCE_COLOR", 1.04)
ANALYZE_GARMENT_ENHANCE_BRIGHTNESS = _env_float("ANALYZE_GARMENT_ENHANCE_BRIGHTNESS", 1.02)
ANALYZE_GARMENT_ENHANCE_LIGHTING_AUTO = os.getenv("ANALYZE_GARMENT_ENHANCE_LIGHTING_AUTO", "0") == "1"
ENABLE_WARDROBE_PROGRESS_SYNC = os.getenv("ENABLE_WARDROBE_PROGRESS_SYNC", "0") == "1"
WARDROBE_PROGRESS_API_BASE_URL = os.getenv("WARDROBE_PROGRESS_API_BASE_URL", "").strip()
WARDROBE_PROGRESS_SYNC_TIMEOUT_S = max(5, _env_int("WARDROBE_PROGRESS_SYNC_TIMEOUT_S", 20))
WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE = os.getenv("WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE", "0") == "1"
ANALYZE_AUX_MIN_REL_AREA = _env_float("ANALYZE_AUX_MIN_REL_AREA", 0.22)
gpu_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)

GARMENT_TYPE_SYNONYMS = {
    "top": "top",
    "shirt": "top",
    "tshirt": "top",
    "t-shirt": "top",
    "tee": "top",
    "blouse": "top",
    "bottom": "bottom",
    "pant": "bottom",
    "pants": "bottom",
    "trouser": "bottom",
    "trousers": "bottom",
    "jean": "bottom",
    "jeans": "bottom",
    "skirt": "bottom",
    "shorts": "bottom",
    "dress": "dress",
    "gown": "dress",
    "kurti": "dress",
    "outer": "outer",
    "outerwear": "outer",
    "jacket": "outer",
    "coat": "outer",
    "blazer": "outer",
    "hoodie": "outer",
}

app = FastAPI(
    title="Glamify AI Engine",
    description="Unified API for Wardrobe Digitization and Virtual Try-On",
    version="2.0.0"
)

# --- Engine Orchestrator ---
class AIEngine:
    def __init__(self):
        self.yolo_runner = YoloRunner()
        self.parser_runner = HumanParserRunner() if ANALYZE_ENABLE_HUMAN_PARSER else None
        self.yolo = YoloCropper(predictor=self.yolo_runner.predict)
        self.parser = HumanParser(parser_fn=self.parser_runner.parse) if self.parser_runner is not None else None
        self.florence = FlorenceRunner()
        self.flux2 = Flux2CVTONRunner()
        self.board_builder = BoardBuilder()
        
    def ensure_vto_ready(self):
        self.flux2.ensure_ready()
        self.florence._ensure_loaded()

    def ensure_analyze_ready(self):
        self.yolo_runner.ensure_ready()
        if self.parser_runner:
            self.parser_runner.ensure_ready()

    def model_status(self):
        return {
            "flux2_loaded": self.flux2._pipeline is not None,
            "florence_loaded": self.florence._model is not None,
            "yolo_loaded": self.yolo_runner.is_loaded,
            "human_parser_loaded": bool(self.parser_runner and self.parser_runner.is_loaded),
        }

engine = AIEngine()

def _upload_or_raise(image_bytes: bytes, container: Optional[str] = None) -> Optional[str]:
    try:
        return storage.upload_image(image_bytes, container=container)
    except Exception:
        if REQUIRE_AZURE_UPLOAD:
            raise
        return None

def _normalize_garment_type(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    normalized = str(raw).strip().lower().replace("_", " ")
    if not normalized:
        return None
    normalized = normalized.replace(" ", "")
    return GARMENT_TYPE_SYNONYMS.get(normalized)

def _bbox_prior(bbox: list[int], width: int, height: int) -> float:
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

def _hybrid_score(yolo_conf: float, florence_conf: float, bbox_prior: float) -> float:
    score = (
        (HYBRID_WEIGHT_YOLO * yolo_conf)
        + (HYBRID_WEIGHT_FLORENCE * florence_conf)
        + (HYBRID_WEIGHT_BBOX * bbox_prior)
    )
    return max(0.0, min(1.0, score))

def _expand_bbox(bbox: list[int], width: int, height: int, pad: int) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    ]

def _bbox_from_mask(mask: np.ndarray) -> Optional[list[int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def _parser_split_candidates(image: Image.Image) -> list[dict]:
    if engine.parser is None:
        return []
    parsing = engine.parser.parse(image)
    category_masks = engine.parser.build_category_masks(parsing)
    total_pixels = float(max(1, image.width * image.height))

    candidates = []
    for garment_type in ("outer", "top", "bottom", "dress"):
        mask = category_masks.get(garment_type)
        if mask is None:
            continue
        area_ratio = float(mask.sum()) / total_pixels
        if area_ratio < ANALYZE_PARSER_MIN_AREA_RATIO:
            continue

        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        bbox = _expand_bbox(bbox, image.width, image.height, ANALYZE_PARSER_PAD)
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < 24 or (y1 - y0) < 24:
            continue

        crop = image.crop((x0, y0, x1, y1))
        confidence = min(0.95, 0.35 + (area_ratio * 2.5))
        candidates.append(
            {
                "label": garment_type,
                "confidence": confidence,
                "image": crop,
                "bbox": bbox,
                "mask": mask[y0:y1, x0:x1],
                "source": "human_parser",
                "parser_area_ratio": area_ratio,
            }
        )
    return candidates

def _parser_split_is_plausible(candidates: list[dict], image_height: int) -> tuple[bool, str]:
    if len(candidates) < 2:
        return (False, "insufficient_candidates")

    candidate_tops = []
    for c in candidates:
        b = c.get("bbox") or [0, 0, 0, 0]
        candidate_tops.append(int(b[1]))
    if candidate_tops and min(candidate_tops) > int(image_height * 0.40):
        return (False, "all_candidates_too_low")

    by_type: dict[str, dict] = {}
    for c in candidates:
        label = _normalize_garment_type(str(c.get("label", ""))) or str(c.get("label", ""))
        score = float(c.get("confidence", 0.0))
        if label not in by_type or score > float(by_type[label].get("confidence", 0.0)):
            by_type[label] = c

    top = by_type.get("top")
    bottom = by_type.get("bottom")
    if top and bottom:
        tb = top.get("bbox") or [0, 0, 0, 0]
        bb = bottom.get("bbox") or [0, 0, 0, 0]
        top_y0, top_y1 = int(tb[1]), int(tb[3])
        bottom_y0 = int(bb[1])

        if top_y0 > int(image_height * 0.45):
            return (False, "top_starts_too_low")
        if top_y1 > int(image_height * 0.92):
            return (False, "top_extends_too_far_down")
        if bottom_y0 < int(image_height * 0.20):
            return (False, "bottom_starts_too_high")
        if top_y0 >= bottom_y0:
            return (False, "top_not_above_bottom")

    return (True, "ok")

def _heuristic_split_candidates(image: Image.Image, base: dict) -> list[dict]:
    bbox = base.get("bbox") or [0, 0, image.width, image.height]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    height_ratio = float(bh) / max(1.0, float(image.height))
    width_ratio = float(bw) / max(1.0, float(image.width))
    if height_ratio < ANALYZE_HEURISTIC_MIN_HEIGHT_RATIO:
        return []
    if width_ratio < 0.15 or width_ratio > 0.90:
        return []

    split_portion = min(0.70, max(0.40, ANALYZE_HEURISTIC_TOP_PORTION))
    split_y = y0 + int(bh * split_portion)
    split_y = min(y1 - 20, max(y0 + 20, split_y))

    top_end = min(y1 - 20, max(y0 + 20, split_y))
    dynamic_overlap_px = max(ANALYZE_HEURISTIC_BOTTOM_OVERLAP_PX, int(bh * ANALYZE_HEURISTIC_BOTTOM_OVERLAP_RATIO))
    bottom_start = max(y0, split_y - dynamic_overlap_px)
    if bottom_start >= top_end:
        bottom_start = max(y0, top_end - 12)

    top_bbox = [x0, y0, x1, top_end]
    bottom_bbox = [x0, bottom_start, x1, y1]
    top_crop = image.crop(tuple(top_bbox))
    bottom_crop = image.crop(tuple(bottom_bbox))

    return [
        {
            "label": "top",
            "confidence": max(0.40, float(base.get("confidence", 0.0)) * 0.55),
            "image": top_crop,
            "bbox": top_bbox,
            "mask": None,
            "source": "heuristic_split",
        },
        {
            "label": "bottom",
            "confidence": max(0.40, float(base.get("confidence", 0.0)) * 0.55),
            "image": bottom_crop,
            "bbox": bottom_bbox,
            "mask": None,
            "source": "heuristic_split",
        },
    ]

def _item_rank_score(item: dict) -> float:
    conf = item.get("confidence", {})
    if isinstance(conf, dict):
        return float(conf.get("hybrid", conf.get("yolo", 0.0)))
    return 0.0

def _dedupe_items(items: list[dict], iou_threshold: float = 0.60) -> list[dict]:
    if not items:
        return []
    ranked = sorted(items, key=_item_rank_score, reverse=True)
    kept: list[dict] = []
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
            if bbox_iou(tuple(cb), tuple(eb)) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept

def _is_shorts_like_caption(text: str) -> bool:
    t = (text or "").lower()
    keys = ("shorts", "cargo shorts", "bermuda", "skirt", "mini skirt", "midi skirt", "maxi skirt")
    return any(k in t for k in keys)

def _infer_type_from_caption(text: str) -> Optional[str]:
    t = (text or "").lower()
    if not t:
        return None
    if any(k in t for k in ("dress", "gown", "kurti", "one-piece", "one piece", "jumpsuit")):
        return "dress"
    if any(k in t for k in ("jacket", "coat", "blazer", "hoodie", "cardigan", "outerwear")):
        return "outer"
    if any(k in t for k in ("pant", "pants", "trouser", "trousers", "jean", "jeans", "shorts", "skirt", "bottom")):
        return "bottom"
    if any(k in t for k in ("shirt", "t-shirt", "tshirt", "tee", "blouse", "top")):
        return "top"
    return None

def _sanitize_garment_description(text: str) -> str:
    s = " ".join((text or "").strip().split())
    if not s:
        return s
    patterns = [
        r"^(?:a|an|the)\s+(?:woman|man|person|model|girl|boy)\s+wearing\s+(.+)$",
        r"^(?:a|an|the)\s+(?:woman|man|person|model|girl|boy)\s+in\s+(.+)$",
        r"^(?:woman|man|person|model|girl|boy)\s+wearing\s+(.+)$",
        r"^(?:woman|man|person|model|girl|boy)\s+in\s+(.+)$",
    ]
    lowered = s.lower().strip().rstrip(".")
    for pattern in patterns:
        m = re.match(pattern, lowered)
        if m:
            cleaned = m.group(1).strip()
            if cleaned:
                return cleaned[0].upper() + cleaned[1:] + "."
    return s

def _suppress_auxiliary_instances(instances: list[dict], image_width: int, image_height: int) -> list[dict]:
    if len(instances) <= 1:
        return instances
    scored = []
    for inst in instances:
        bbox = inst.get("bbox") or [0, 0, image_width, image_height]
        x0, y0, x1, y1 = [int(v) for v in bbox]
        area = max(1, x1 - x0) * max(1, y1 - y0)
        scored.append((area, inst))
    max_area = max(a for a, _ in scored)
    cutoff = max(1.0, float(max_area) * max(0.0, min(1.0, ANALYZE_AUX_MIN_REL_AREA)))
    kept = [inst for area, inst in scored if float(area) >= cutoff]
    return kept if kept else [max(scored, key=lambda x: x[0])[1]]

def _to_public_item(item: dict) -> dict:
    return {k: v for k, v in item.items() if not str(k).startswith("_")}

def _write_temp_png(image: Image.Image) -> str:
    fd, path = tempfile.mkstemp(prefix="analyze_crop_", suffix=".png")
    os.close(fd)
    image.save(path, format="PNG")
    return f"file://{path}"

def _prepare_vton_source_image(
    full_image: Image.Image,
    bbox: Optional[list[int]],
    garment_type: str,
    total_items: int,
) -> tuple[Image.Image, list[int], str]:
    width, height = full_image.size
    if ANALYZE_VTON_SINGLE_ITEM_USE_FULL_IMAGE and total_items <= 1:
        return full_image.copy(), [0, 0, width, height], "full_image_single_item"

    if not bbox or len(bbox) != 4:
        return full_image.copy(), [0, 0, width, height], "full_image_no_bbox"

    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))

    if garment_type == "dress" and ANALYZE_VTON_DRESS_USE_FULL_IMAGE and total_items <= 1:
        return full_image.copy(), [0, 0, width, height], "full_image_dress_single"

    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad_ratio = ANALYZE_VTON_CROP_PAD_RATIO_DRESS if garment_type == "dress" else ANALYZE_VTON_CROP_PAD_RATIO
    pad_ratio = max(0.0, min(0.8, pad_ratio))
    x_pad = int(bw * pad_ratio)
    y_pad_top = int(bh * pad_ratio)
    y_pad_bottom = int(bh * pad_ratio)
    if garment_type == "dress":
        y_pad_bottom = int(bh * max(pad_ratio, ANALYZE_VTON_CROP_BOTTOM_EXTRA_RATIO_DRESS))

    ex0 = max(0, x0 - x_pad)
    ey0 = max(0, y0 - y_pad_top)
    ex1 = min(width, x1 + x_pad)
    ey1 = min(height, y1 + y_pad_bottom)
    return full_image.crop((ex0, ey0, ex1, ey1)), [ex0, ey0, ex1, ey1], "expanded_bbox"

def _flatten_rgba_on_white(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGB", image.size, (255, 255, 255))
    bg.paste(image, mask=image.split()[-1])
    return bg

def _content_bbox_from_image(image: Image.Image) -> tuple[int, int, int, int]:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba)
    alpha = arr[:, :, 3]
    content = alpha > ANALYZE_GARMENT_ALPHA_THRESHOLD
    if not np.any(content):
        rgb = arr[:, :, :3]
        content = np.any(rgb < ANALYZE_GARMENT_WHITE_THRESHOLD, axis=2)

    ys, xs = np.where(content)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, rgba.width, rgba.height)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)

def _fit_image_to_aspect(image: Image.Image, aspect_w: int, aspect_h: int) -> Image.Image:
    iw, ih = image.size
    if iw < 1 or ih < 1:
        return image
    target_ratio = float(aspect_w) / float(aspect_h)
    cur_ratio = float(iw) / float(ih)
    if abs(cur_ratio - target_ratio) < 1e-4:
        return image

    if cur_ratio > target_ratio:
        out_w = iw
        out_h = int(round(iw / target_ratio))
    else:
        out_h = ih
        out_w = int(round(ih * target_ratio))

    out_w = max(out_w, iw)
    out_h = max(out_h, ih)
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 0))
    else:
        canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    paste_x = (out_w - iw) // 2
    paste_y = (out_h - ih) // 2
    canvas.paste(image, (paste_x, paste_y), image if image.mode == "RGBA" else None)
    return canvas

def _enhance_garment_image(image: Image.Image) -> Image.Image:
    if not ANALYZE_GARMENT_ENHANCE_ENABLED:
        return image

    rgba = image.convert("RGBA")
    alpha = rgba.split()[-1]
    
    # Standard enhancements
    # 1. Lighting Balance (Fixes the 'half section shadow' issue)
    if ANALYZE_GARMENT_ENHANCE_LIGHTING_AUTO:
        try:
            # Convert to HSV to ensure hue-safe brightness adjustments
            hsv = np.asarray(rgba.convert("HSV")).astype(np.float32)
            v_chan = hsv[:, :, 2]
            mask = np.asarray(alpha) > 10
            
            if np.any(mask):
                h, w = v_chan.shape[0], v_chan.shape[1]
                
                # --- HORIZONTAL CORRECTION ---
                h_brightness = np.zeros(w)
                for x in range(w):
                    col_mask = mask[:, x]
                    if np.any(col_mask):
                        h_brightness[x] = np.mean(v_chan[col_mask, x])
                
                win = max(5, w // 10)
                h_brightness = np.convolve(h_brightness, np.ones(win)/win, mode="same")
                left_avg = np.mean(h_brightness[:w//2][h_brightness[:w//2] > 0])
                right_avg = np.mean(h_brightness[w//2:][h_brightness[w//2:] > 0])
                
                if not np.isnan(left_avg) and not np.isnan(right_avg) and left_avg > 2 and right_avg > 2:
                    h_ratio = right_avg / left_avg
                    if abs(1.0 - h_ratio) > 0.10:
                        logger.info(f"Applying horizontal lighting correction: ratio={h_ratio:.2f}")
                        h_gradient = np.linspace(h_ratio, 1.0, w)
                        v_chan *= h_gradient

                # --- VERTICAL CORRECTION (Gamma-based) ---
                v_brightness = np.zeros(h)
                for y in range(h):
                    row_mask = mask[y, :]
                    if np.any(row_mask):
                        v_brightness[y] = np.mean(v_chan[y, row_mask])
                
                v_win = max(5, h // 10)
                v_brightness = np.convolve(v_brightness, np.ones(v_win)/v_win, mode="same")
                
                top_zone = np.mean(v_brightness[:h//6][v_brightness[:h//6] > 0])
                bot_zone = np.mean(v_brightness[h//2:][v_brightness[h//2:] > 0])
                
                if not np.isnan(top_zone) and not np.isnan(bot_zone) and top_zone > 2 and bot_zone > 2:
                    v_ratio = bot_zone / top_zone
                    if v_ratio > 1.15: # Only lift if top is 15%+ darker
                        logger.info(f"Applying vertical gamma-lift (HSV): ratio={v_ratio:.2f}")
                        # Damping the ratio to prevent over-lifting and 'washing out' colors
                        gamma = 1.0 / (v_ratio ** 0.4) 
                        v_gamma_grad = np.linspace(gamma, 1.0, h).reshape(-1, 1)
                        v_chan = 255.0 * (v_chan / 255.0) ** v_gamma_grad

                # --- GLOBAL LIFT FOR DARK FABRICS ---
                avg_v = np.mean(v_chan[mask])
                if avg_v < 55:
                    logger.info(f"Applying deep shadow boost (HSV avg={avg_v:.1f})")
                    # Subtle 10% lift for very dark pixels only
                    v_chan = np.where(v_chan < 50, v_chan * 1.10, v_chan)

                hsv[:, :, 2] = np.clip(v_chan, 0, 255)
                # Convert back to RGB via HSV
                rgba = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB").convert("RGBA")
                rgba.putalpha(alpha)
        except Exception as e:
            logger.error(f"Adaptive lighting correction failed: {e}", exc_info=True)
            # Fallback to original image if correction fails to avoid bad artifacts
            rgba = image.convert("RGBA")

    # 2. PIL Image Enhancements
    res = rgba.convert("RGB")
    if ANALYZE_GARMENT_ENHANCE_BRIGHTNESS != 1.0:
        res = ImageEnhance.Brightness(res).enhance(ANALYZE_GARMENT_ENHANCE_BRIGHTNESS)
    if ANALYZE_GARMENT_ENHANCE_CONTRAST != 1.0:
        res = ImageEnhance.Contrast(res).enhance(ANALYZE_GARMENT_ENHANCE_CONTRAST)
    if ANALYZE_GARMENT_ENHANCE_COLOR != 1.0:
        res = ImageEnhance.Color(res).enhance(ANALYZE_GARMENT_ENHANCE_COLOR)
    if ANALYZE_GARMENT_ENHANCE_SHARPNESS != 1.0:
        # Sharpness can cause halos if too high; use UnsharpMask for better results
        res = ImageEnhance.Sharpness(res).enhance(ANALYZE_GARMENT_ENHANCE_SHARPNESS)
        res = res.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3))

    res.putalpha(alpha)
    return res

def _postprocess_extracted_garment_bytes(image_bytes: bytes) -> tuple[bytes, dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    
    # 0. Safe Hole Filling (Fixes 'missing cloth patches' on patterned garments)
    try:
        from scipy.ndimage import binary_fill_holes
        alpha = image.split()[-1]
        alpha_arr = np.asarray(alpha)
        # Use a high threshold to find 'true' gaps within the garment
        mask = (alpha_arr > 50).astype(np.uint8)
        filled = binary_fill_holes(mask)
        # Only fill pixels that were previously 'holes' (mask=0, filled=1)
        new_alpha = np.where(filled & (alpha_arr < 150), 255, alpha_arr)
        image.putalpha(Image.fromarray(new_alpha.astype(np.uint8), mode="L"))
    except Exception as hole_e:
        logger.warning(f"Simple hole filling failed: {hole_e}")

    # Crop to content
    original_size = (int(image.width), int(image.height))
    bbox = _content_bbox_from_image(image)
    cropped = image.crop(bbox)

    # Upscale small crops to ensure sharpness (min 512px on shortest side)
    min_dim = min(cropped.width, cropped.height)
    if min_dim < 512 and min_dim > 0:
        scale = 512.0 / min_dim
        new_w = int(cropped.width * scale)
        new_h = int(cropped.height * scale)
        cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"Upscaled small garment crop from {min_dim}px to {min(new_w, new_h)}px (scale={scale:.2f})")

    processed = cropped

    if ANALYZE_GARMENT_POSTPROCESS_ENABLED:
        processed = _fit_image_to_aspect(
            processed,
            aspect_w=ANALYZE_GARMENT_TARGET_ASPECT_W,
            aspect_h=ANALYZE_GARMENT_TARGET_ASPECT_H,
        )
        processed = _enhance_garment_image(processed)

    out = io.BytesIO()
    processed.save(out, format="PNG")
    return out.getvalue(), {
        "enabled": ANALYZE_GARMENT_POSTPROCESS_ENABLED,
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "content_bbox": [int(v) for v in bbox],
        "content_size": {"width": int(cropped.width), "height": int(cropped.height)},
        "output_size": {"width": int(processed.width), "height": int(processed.height)},
        "target_aspect": f"{ANALYZE_GARMENT_TARGET_ASPECT_W}:{ANALYZE_GARMENT_TARGET_ASPECT_H}",
        "enhance_enabled": ANALYZE_GARMENT_ENHANCE_ENABLED,
    }

def _fetch_image_bytes(url: str, timeout: int = 45) -> bytes:
    if str(url).startswith("file://"):
        with open(str(url)[7:], "rb") as f:
            return f.read()
    resp = requests.get(url, timeout=timeout)
    if not (200 <= resp.status_code < 300) or not resp.content:
        raise RuntimeError(f"Could not download image bytes: status={resp.status_code} url={url}")
    return resp.content

def _compare_image_similarity(source_bytes: bytes, output_bytes: bytes) -> dict:
    source_md5 = hashlib.md5(source_bytes).hexdigest()
    output_md5 = hashlib.md5(output_bytes).hexdigest()
    exact_bytes = source_md5 == output_md5

    src_img = _flatten_rgba_on_white(Image.open(io.BytesIO(source_bytes)))
    out_img = _flatten_rgba_on_white(Image.open(io.BytesIO(output_bytes)))
    target = (256, 256)
    src_arr = np.asarray(src_img.resize(target, Image.BICUBIC), dtype=np.float32)
    out_arr = np.asarray(out_img.resize(target, Image.BICUBIC), dtype=np.float32)
    mae = float(np.mean(np.abs(src_arr - out_arr)))

    return {
        "exact_bytes": exact_bytes,
        "source_md5": source_md5,
        "output_md5": output_md5,
        "resized_mae": round(mae, 4),
    }

def _type_to_vton_category(garment_type: str) -> str:
    g = (_normalize_garment_type(garment_type) or garment_type or "dress").strip().lower()
    if g in {"top", "outer"}:
        return "tops"
    if g == "bottom":
        return "bottoms"
    return "one-pieces"

def _validate_vton_output_bytes(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise RuntimeError(f"Invalid VTON output image bytes: {exc}") from exc

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise RuntimeError(f"Unsupported VTON output image shape: {arr.shape}")

    has_alpha = arr.shape[2] == 4
    if has_alpha:
        alpha = arr[:, :, 3].astype(np.uint8)
        alpha_ratio = float(np.mean(alpha > 8))
        visible_mask = alpha > 8
        visible_rgb = arr[:, :, :3][visible_mask]
    else:
        alpha_ratio = 1.0
        visible_rgb = arr[:, :, :3].reshape(-1, 3)

    if visible_rgb.size == 0:
        rgb_std = 0.0
        non_white_ratio = 0.0
    else:
        visible_f = visible_rgb.astype(np.float32)
        rgb_std = float(np.std(visible_f))
        non_white_ratio = float(np.mean(np.any(visible_rgb < 245, axis=1)))

    if has_alpha and alpha_ratio < ANALYZE_VTON_MIN_ALPHA_RATIO:
        raise RuntimeError(
            f"VTON output appears mostly transparent (alpha_ratio={alpha_ratio:.6f})."
        )
    if has_alpha and alpha_ratio > 0.95 and rgb_std < ANALYZE_VTON_MIN_RGB_STD and non_white_ratio < 0.01:
        raise RuntimeError("VTON output appears as a near-solid blank frame.")
    if (not has_alpha) and rgb_std < ANALYZE_VTON_MIN_RGB_STD and non_white_ratio < 0.01:
        raise RuntimeError("VTON output appears blank/near-uniform white.")

    return {
        "has_alpha": has_alpha,
        "alpha_ratio": round(alpha_ratio, 6),
        "rgb_std": round(rgb_std, 4),
        "non_white_ratio": round(non_white_ratio, 6),
        "width": int(img.width),
        "height": int(img.height),
    }

def _extract_cloth_from_crop(crop: Image.Image, garment_type: str) -> tuple[Image.Image, dict]:
    """
    High-quality garment extraction from a person/mannequin image.
    Uses rembg neural model for pixel-accurate background removal,
    followed by HSV skin-tone masking and parser masking to remove the mannequin body.
    """
    selected_type = _normalize_garment_type(garment_type) or "top"
    total_pixels = float(max(1, crop.width * crop.height))
    img_np = np.asarray(crop.convert("RGB"))
    h, w = img_np.shape[:2]

    try:
        import cv2
        from rembg import remove, new_session

        # STEP 1: Neural Background Removal (ISNet)
        session = new_session("isnet-general-use")
        rembg_result = remove(crop.convert("RGB"), session=session, post_process_mask=True)
        rembg_alpha = np.array(rembg_result)[:, :, 3].astype(np.uint8)
        logger.info(f"rembg alpha coverage: {rembg_alpha.mean():.1f}")

        # STEP 2: Body Part & Cross-Category Kill (Surgical Isolation)
        # We use Segformer B2 Clothes labels: 4:Upper, 6:Pants, 5:Skirt, 7:Dress, 8:Belt, 9/10:Shoes, 11:Face, 12/13:Legs, 14/15:Arms
        parsing = engine.parser.parse(crop)
        
        # Always kill body parts, hair, and shoes to isolate the floating garment
        kill_ids = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16]
        
        if selected_type == "top":
            # Keep: Upper(4), Scarf(17)
            # Kill: Skirt(5), Pants(6), Dress(7), Belt(8)
            kill_ids.extend([5, 6, 7, 8])
        elif selected_type == "bottom":
            # Keep: Skirt(5), Pants(6), Belt(8)
            # Kill: Upper(4), Dress(7), Scarf(17)
            kill_ids.extend([4, 7, 17])
        elif selected_type == "dress":
            # Keep: Dress(7), Upper(4), Skirt(5), Pants(6), Belt(8), Scarf(17)
            # (Keeping most for dress as it often blends classes)
            pass

        parser_kill = np.isin(parsing, kill_ids).astype(np.uint8) * 255
        # 2x2 dilation is enough to clean skin shadows without eating the cloth
        parser_kill = cv2.dilate(parser_kill, np.ones((2, 2), np.uint8), iterations=1)

        # STEP 3: Fill holes in the rembg mask BEFORE combining
        # This prevents crocheted/mesh tops from being dissolved into fragments
        inv_rembg = cv2.bitwise_not(rembg_alpha)
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(inv_rembg)
        rembg_filled = rembg_alpha.copy()
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 5000:
                rembg_filled[labels_im == i] = 255

        # STEP 4: Apply precise sharp cut
        final_alpha = rembg_filled.copy()
        final_alpha[parser_kill > 0] = 0

        # STEP 5: Dust removal (eliminate isolated background floating specks)
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(final_alpha)
        clean_alpha = np.zeros_like(final_alpha)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 500:
                clean_alpha[labels_im == i] = final_alpha[labels_im == i]

        # STEP 6: Micro-feather for anti-aliasing edge
        alpha = cv2.GaussianBlur(clean_alpha, (3, 3), 0)
        final_mask = alpha > 10
        path_name = "rembg_isnet_premium"

    except Exception as e:
        logger.warning(f"rembg extraction failed, falling back to parser: {e}")
        try:
            parsing = engine.parser.parse(crop)
            garment_mask = np.isin(parsing, [5, 6, 7, 8, 9, 10, 11, 12])
            kill_mask = np.isin(parsing, [1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19])
            final_mask = garment_mask & (~kill_mask) & (parsing != 0)
            alpha = build_soft_alpha(final_mask, feather_px=2)
            path_name = "parser_fallback"
        except Exception as e2:
            logger.error(f"Parser fallback also failed: {e2}")
            raise RuntimeError(f"All extraction methods failed: {e} / {e2}")

    final_area_ratio = float(final_mask.sum()) / total_pixels
    if final_area_ratio <= 0.0:
        raise RuntimeError(f"Extraction produced empty result for: {selected_type}")

    alpha_plane = alpha[:, :, None]
    cloth_rgba = np.concatenate([img_np, alpha_plane], axis=2)
    cloth_image = Image.fromarray(cloth_rgba, mode="RGBA")

    meta = {"path": path_name, "mask_area_ratio": round(final_area_ratio, 6)}
    return cloth_image, meta

def _run_vton_cloth_only_fallback(image_url: str, garment_type: str, vto_mode: bool = False) -> dict:
    if not ANALYZE_VTON_CLOTH_ONLY_ENDPOINT:
        raise RuntimeError("ANALYZE_VTON_CLOTH_ONLY_ENDPOINT is empty.")
    category = _type_to_vton_category(garment_type)
    person_image_url = image_url
    if ANALYZE_VTON_USE_SHOWROOM_PERSON:
        if not ANALYZE_VTON_SHOWROOM_PERSON_IMAGE_URL:
            raise RuntimeError(
                "ANALYZE_VTON_SHOWROOM_PERSON_IMAGE_URL is empty while ANALYZE_VTON_USE_SHOWROOM_PERSON=1."
            )
        person_image_url = ANALYZE_VTON_SHOWROOM_PERSON_IMAGE_URL
        logger.info(f"VTON fallback using showroom person reference: {person_image_url}")
    else:
        logger.info(f"VTON fallback using original image as person reference: {person_image_url}")
    
    # --- FASHN PAYLOAD CONFIG ---
    # To understand the root cause of the 'holes', we now request the FULL tryon result
    # (mannequin wearing the dress) instead of the cutout.
    # This reveals if the generator itself is failing or if the cutout process is the culprit.
    should_cutout = False 

    payload = {
        "image_url": person_image_url,
        "garment_image_url": image_url,
        "category": category,
        "garment_photo_type": "model",
        "quality_preset": ANALYZE_VTON_QUALITY_PRESET,
        "num_timesteps": ANALYZE_VTON_NUM_TIMESTEPS,
        "guidance_scale": ANALYZE_VTON_GUIDANCE_SCALE,
        "skip_cfg_last_n_steps": 1,
        "num_samples": 1,
        "segmentation_free": bool(ANALYZE_VTON_SEGMENTATION_FREE),
        "output_format": "png",
        "cutout_enabled": should_cutout,
        "cutout_feather_px": 2,
        "cutout_background": "original",
        "drop_shadow": False,
        "zoom_enabled": bool(ANALYZE_VTON_TOP_ZOOM_ENABLED and category == "tops"),
        "zoom_padding_ratio": max(0.0, min(0.5, ANALYZE_VTON_ZOOM_PADDING_RATIO)),
        "upscale_enabled": bool(ANALYZE_VTON_BOTTOM_UPSCALE_ENABLED and category == "bottoms"),
        "upscale_factor": 2,
        "upscale_allow_bicubic_fallback": True,
        "allow_source_fallback": ANALYZE_VTON_ALLOW_SOURCE_FALLBACK,
    }
    response = requests.post(ANALYZE_VTON_CLOTH_ONLY_ENDPOINT, json=payload, timeout=ANALYZE_VTON_TIMEOUT_S)
    if not (200 <= response.status_code < 300):
        if response.status_code == 400 and "multiple garments" in response.text.lower() and "selected_index" not in str(payload):
             # Auto-retry with index 0 if multiple garments detected (often happens with patterns)
             payload["selected_index"] = 0
             logger.info("Retrying with selected_index=0 due to multiple garment detection.")
             response = requests.post(ANALYZE_VTON_CLOTH_ONLY_ENDPOINT, json=payload, timeout=ANALYZE_VTON_TIMEOUT_S)
             if not (200 <= response.status_code < 300):
                 raise RuntimeError(f"VTON retry failed: {response.status_code} {response.text[:200]}")
        else:
            raise RuntimeError(f"VTON cloth-only fallback failed: {response.status_code} {response.text[:200]}")
            
    body = response.json()
    url = body.get("url") or body.get("azure_url")
    if not url:
        raise RuntimeError("VTON fallback response missing url.")
    public_url = str(url)
    mirrored_ok = False
    image_quality = {}
    source_similarity = {}
    source_bytes = b""
    raw_vton_bytes = b""
    processed_vton_bytes = b""
    postprocess_meta = {}

    source_bytes = _fetch_image_bytes(image_url, timeout=45)
    raw_vton_bytes = _fetch_image_bytes(public_url, timeout=45)

    # --- LOCAL EXTRACTION FROM MANNEQUIN ---
    # We now have the mannequin wearing the garment (raw_vton_bytes).
    # We perform local extraction to isolate the garment and ensure no holes are introduced.
    try:
        vton_img_mannequin = Image.open(io.BytesIO(raw_vton_bytes)).convert("RGB")
        logger.info(f"Loaded VTON mannequin image for local extraction: {vton_img_mannequin.size} {vton_img_mannequin.mode}")
        
        extracted_cloth, extraction_meta = _extract_cloth_from_crop(vton_img_mannequin, garment_type)
        extraction_meta["vto_path"] = "vton_fallback_mannequin_local_parser"
        
        # Save as bytes for postprocessing
        buf = io.BytesIO()
        extracted_cloth.save(buf, format="PNG")
        final_vton_bytes = buf.getvalue()
    except Exception as extract_err:
        logger.error(f"Local extraction of VTON mannequin failed: {extract_err}", exc_info=True)
        # Fallback to the raw mannequin result (captured as a solid asset)
        final_vton_bytes = raw_vton_bytes
        extraction_meta = {"path": "vton_fallback_mannequin_bypass", "error": str(extract_err)}

    source_similarity = _compare_image_similarity(source_bytes, final_vton_bytes)
    if ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH:
        if source_similarity.get("exact_bytes"):
            raise RuntimeError("VTON output is byte-identical to source crop (passthrough).")
        if float(source_similarity.get("resized_mae", 999.0)) <= ANALYZE_VTON_MAX_SOURCE_MAE:
            raise RuntimeError(
                f"VTON output too similar to source crop (resized_mae={source_similarity.get('resized_mae')})."
            )
    
    processed_vton_bytes, postprocess_meta = _postprocess_extracted_garment_bytes(final_vton_bytes)
    image_quality = _validate_vton_output_bytes(processed_vton_bytes)

    # Normalize to publicly consumable URLs
    mirrored_raw_url = ""
    try:
        # Save RAW VTON for explicit user inspection
        raw_mirrored = _upload_or_raise(raw_vton_bytes)
        if raw_mirrored:
            mirrored_raw_url = str(raw_mirrored)
    except Exception as raw_e:
        logger.warning(f"Failed to upload raw VTON for debug: {raw_e}")

    try:
        mirrored = _upload_or_raise(processed_vton_bytes)
        if mirrored:
            public_url = str(mirrored)
            mirrored_ok = True
        else:
            mirrored_ok = False
    except Exception as mirror_err:
        if ANALYZE_REQUIRE_MIRRORED_VTON_URL:
            raise RuntimeError(f"Could not mirror VTON output URL to public Azure blob: {mirror_err}") from mirror_err
        logger.warning(f"Could not mirror VTON output URL to Azure: {mirror_err}")
        mirrored_ok = False

    return {
        "url": public_url,
        "raw_url": mirrored_raw_url or public_url,
        "meta": {
            "path": extraction_meta.get("path", "vton_fallback"),
            "endpoint": ANALYZE_VTON_CLOTH_ONLY_ENDPOINT,
            "category": category,
            "vto_mode": vto_mode,
            "showroom_person_enabled": ANALYZE_VTON_USE_SHOWROOM_PERSON,
            "person_image_url": person_image_url,
            "fashn_raw_url": str(url),
            "raw_output_url": mirrored_raw_url,
            "mirrored": mirrored_ok,
            "quality": image_quality,
            "source_similarity": source_similarity,
            "postprocess": postprocess_meta,
            "request_payload": payload,
            "hybrid_meta": extraction_meta
        },
    }

def _sync_wardrobe_progress(
    *,
    authorization: Optional[str],
    progress_id: str,
    output_url: Optional[str],
    prompt_description: str,
    metadata: dict,
) -> dict:
    if not ENABLE_WARDROBE_PROGRESS_SYNC:
        return {"enabled": False, "synced": False, "reason": "disabled_by_flag", "id": progress_id}
    if not WARDROBE_PROGRESS_API_BASE_URL:
        return {"enabled": True, "synced": False, "reason": "missing_base_url", "id": progress_id}
    if not output_url:
        return {"enabled": True, "synced": False, "reason": "missing_output_url", "id": progress_id}

    base = WARDROBE_PROGRESS_API_BASE_URL.strip().strip("\"'")
    if base.endswith("/wardrobe/progress"):
        endpoint = base
    else:
        endpoint = f"{base.rstrip('/')}/wardrobe/progress"

    payload = {
        "id": progress_id,
        "outputImage": output_url,
        "isAcceptable": True,
        "isProcessed": False,
        "promptDescription": str(prompt_description or ""),
        "metadata": metadata,
    }
    headers = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=WARDROBE_PROGRESS_SYNC_TIMEOUT_S)
        retried_with_input_url = False

        if not (200 <= resp.status_code < 300):
            return {
                "enabled": True,
                "synced": False,
                "reason": "http_error",
                "status_code": resp.status_code,
                "error": resp.text[:200],
                "id": progress_id,
                "endpoint": endpoint,
                "request_payload": payload,
                "retried_with_input_url": retried_with_input_url,
            }
        body = resp.json() if resp.content else {}
        data = body.get("data", {}) if isinstance(body, dict) else {}
        synced_id = str(data.get("id") or progress_id)
        return {
            "enabled": True,
            "synced": True,
            "id": synced_id,
            "url": data.get("url") or data.get("progressUrl") or data.get("progress_url"),
            "endpoint": endpoint,
            "request_payload": payload,
            "retried_with_input_url": retried_with_input_url,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "synced": False,
            "reason": "request_exception",
            "error": str(exc),
            "id": progress_id,
            "endpoint": endpoint,
            "request_payload": payload,
        }

# --- Schemas ---

class VTORequest(BaseModel):
    user_image_url: str
    garment_image_url: str
    # Optional context for better prompt generation
    user_top_description: Optional[str] = None 
    steps: int = Field(default=6, ge=4, le=30)
    seed: int = Field(default=23, ge=0, le=2147483647)

# --- Endpoints ---
@app.on_event("startup")
async def startup_load():
    t0 = time.time()
    if os.getenv("PRELOAD", "1") == "1":
        await asyncio.to_thread(engine.ensure_vto_ready)
    
    if os.getenv("PRELOAD_ANALYZE", "1") == "1":
        await asyncio.to_thread(engine.ensure_analyze_ready)
    
    logger.info(f"Startup preloading completed in {time.time() - t0:.2f}s")

@app.get("/")
def read_root():
    return {"status": "ok", "engine": "Glamify-AI-Unified", "ready": True}

@app.post("/analyze")
@app.post("/analzye")
async def analyze_garment(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form(None, alias="type"),
    garmentType: Optional[str] = Form(None),
    selected_index: Optional[int] = Form(None),
    debug: bool = Form(False),
    _authorization: Optional[str] = Header(None, alias="Authorization")
):
    """
    Digitizes a garment from an image (YOLO + Florence-2).
    Returns Flutter-contract-compliant responses:
    - 200 JSON for success (ACCEPTED)
    - 400 multipart/form-data for multi-item selection
    - 400 JSON for rejections
    - 401 JSON for auth failure
    - 500 JSON for server errors
    """
    t0 = time.time()

    # ── AUTH ──
    try:
        auth_payload = _verify_bearer_token(_authorization)
    except PermissionError:
        payload = _build_error_payload(
            title="Session Expired",
            description="Please log in again and try uploading your item.",
            reason_codes=["UNAUTHORIZED"],
            status_code=401,
            result="ERROR",
        )
        return _multipart_form_response(payload)

    # ── INPUT VALIDATION ──
    upload = file or image
    if not upload:
        payload = _build_error_payload(
            title="No Image Provided",
            description="Please upload an image to analyze.",
            reason_codes=["INVALID_IMAGE"],
            status_code=400,
        )
        return _multipart_form_response(payload)

    effective_type = garment_type or garmentType
    requested_type = _normalize_garment_type(effective_type)

    try:
        # 1. Load Image
        image_bytes = await upload.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        async with gpu_semaphore:
            # 2. Detect & Crop (YOLO)
            parser_split_used = False
            parser_candidates_count = 0
            parser_split_reject_reason = ""
            heuristic_split_used = False
            heuristic_candidates_count = 0
            instances = engine.yolo.detect_instances(img)
            if not instances:
                # Fallback to full image if no garment detected
                instances = [{"mask": None, "bbox": (0, 0, img.width, img.height), "label": "garment", "image": img}]
            else:
                # Finalize crops
                instances = engine.yolo.get_crops(img, instances)

            # 2a. If YOLO returns only one person-like crop, use parser split candidates.
            raw_detected_count = len(instances)
            if ANALYZE_ENABLE_PARSER_SPLIT and engine.parser is not None and len(instances) <= 1:
                try:
                    parser_candidates = _parser_split_candidates(img)
                    parser_candidates_count = len(parser_candidates)
                    if len(parser_candidates) >= 2:
                        plausible, reason = _parser_split_is_plausible(parser_candidates, img.height)
                        if plausible:
                            parser_split_used = True
                            instances = parser_candidates
                        else:
                            parser_split_reject_reason = reason
                except Exception as parser_err:
                    logger.warning(f"Human parser split fallback failed: {parser_err}")

            # 2b. Heuristic split for full-body single candidate when parser could not split.
            if ANALYZE_ENABLE_HEURISTIC_SPLIT and len(instances) == 1 and not parser_split_used:
                try:
                    base_inst = instances[0]
                    classify = engine.florence.classify_garment_type(base_inst["image"], hint_type=requested_type)
                    base_type = str(classify.get("type") or "").strip().lower()
                    if base_type != "dress":
                        heuristics = _heuristic_split_candidates(img, base_inst)
                        heuristic_candidates_count = len(heuristics)
                        if len(heuristics) >= 2:
                            heuristic_split_used = True
                            instances = heuristics
                except Exception as heur_err:
                    logger.warning(f"Heuristic split fallback failed: {heur_err}")

            instances = _suppress_auxiliary_instances(instances, img.width, img.height)

            # 2c. Optional hybrid semantic rerank with Florence-2 (feature-flagged).
            if USE_FLORENCE_HYBRID_VERIFY and instances:
                ranked = sorted(instances, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                for idx, inst in enumerate(ranked):
                    yolo_conf = float(inst.get("confidence", 0.0))
                    bbox = inst.get("bbox") or [0, 0, img.width, img.height]
                    prior = _bbox_prior(bbox, img.width, img.height)

                    florence_type = _normalize_garment_type(str(inst.get("label", ""))) or "top"
                    florence_conf = 0.50
                    florence_reason = "fallback_no_florence"
                    florence_caption = ""

                    if idx < HYBRID_TOP_K:
                        try:
                            source_hint = _normalize_garment_type(str(inst.get("label", "")))
                            hint = requested_type or source_hint
                            verdict = engine.florence.classify_garment_type(inst["image"], hint_type=hint)
                            florence_type = verdict.get("type") or florence_type
                            florence_conf = float(verdict.get("score", florence_conf))
                            florence_reason = str(verdict.get("reason", "ok"))
                            florence_caption = str(verdict.get("caption", ""))
                            # Keep heuristic split labels stable unless Florence is very confident.
                            if (
                                str(inst.get("source", "")) == "heuristic_split"
                                and source_hint
                                and florence_type != source_hint
                                and florence_conf < 0.90
                            ):
                                florence_type = source_hint
                                florence_reason = f"{florence_reason}|locked_to_heuristic_label"

                            # For shorts/skirt-like bottoms, trim lower leg-heavy region from heuristic split.
                            if (
                                str(inst.get("source", "")) == "heuristic_split"
                                and source_hint == "bottom"
                                and _is_shorts_like_caption(florence_caption)
                            ):
                                bx = inst.get("bbox") or [0, 0, img.width, img.height]
                                x0, y0, x1, y1 = [int(v) for v in bx]
                                h = max(1, y1 - y0)
                                keep_h = max(int(img.height * 0.16), int(h * ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO))
                                new_y1 = min(y1, y0 + keep_h)
                                if new_y1 - y0 >= 48:
                                    inst["bbox"] = [x0, y0, x1, new_y1]
                                    inst["image"] = img.crop((x0, y0, x1, new_y1))
                                    bbox = inst["bbox"]
                                    prior = _bbox_prior(bbox, img.width, img.height)
                        except Exception as classify_err:
                            logger.warning(f"Florence hybrid classify failed, falling back to YOLO candidate ordering: {classify_err}")

                    score = _hybrid_score(yolo_conf=yolo_conf, florence_conf=florence_conf, bbox_prior=prior)
                    if requested_type and florence_type == requested_type:
                        score = min(1.0, score + 0.06)

                    inst["_hybrid"] = {
                        "predicted_type": florence_type,
                        "yolo_confidence": yolo_conf,
                        "florence_confidence": florence_conf,
                        "bbox_prior": prior,
                        "score": score,
                        "florence_reason": florence_reason,
                        "florence_caption": florence_caption,
                    }

                ranked = sorted(ranked, key=lambda inst: float(inst.get("_hybrid", {}).get("score", 0.0)), reverse=True)
                if HYBRID_MIN_SCORE > 0.0:
                    thresholded = [inst for inst in ranked if float(inst.get("_hybrid", {}).get("score", 0.0)) >= HYBRID_MIN_SCORE]
                    if thresholded:
                        ranked = thresholded
                instances = ranked[:ANALYZE_MAX_ITEMS]

            # 3. Analyze with Florence-2 (Auto-Captioning)
            items = []
            for idx, inst in enumerate(instances):
                if ANALYZE_CAPTION_MODE == "detailed":
                    prompt_desc = engine.florence.describe_garment(inst["image"])
                else:
                    prompt_desc = engine.florence.describe_garment_short(inst["image"])

                hybrid_meta = inst.get("_hybrid", {})
                autotype_meta = {}
                if USE_FLORENCE_HYBRID_VERIFY:
                    resolved_type = hybrid_meta.get("predicted_type")
                    type_source = "florence_hybrid"
                else:
                    resolved_type = _normalize_garment_type(str(inst.get("label")))
                    type_source = "yolo"
                    if not resolved_type:
                        caption_guess = _infer_type_from_caption(prompt_desc)
                        if caption_guess:
                            resolved_type = caption_guess
                            type_source = "caption_autotype"
                            autotype_meta = {"reason": "caption_guess"}
                        else:
                            try:
                                verdict = engine.florence.classify_garment_type(inst["image"], hint_type=requested_type)
                                guess = _normalize_garment_type(str(verdict.get("type")))
                                if guess:
                                    resolved_type = guess
                                    type_source = "florence_autotype"
                                    autotype_meta = {
                                        "score": float(verdict.get("score", 0.0)),
                                        "reason": str(verdict.get("reason", "")),
                                    }
                            except Exception as auto_type_err:
                                logger.warning(f"Per-instance automatic type resolution failed: {auto_type_err}")
                    if not resolved_type:
                        resolved_type = "top"
                detection_source = str(inst.get("source", "yolo"))
                bbox = inst["bbox"]
                x0, y0, x1, y1 = bbox
                crop_w = max(1, int(x1) - int(x0))
                crop_h = max(1, int(y1) - int(y0))
                crop_area_ratio = min(1.0, float(crop_w * crop_h) / max(1.0, float(img.width * img.height)))

                # Category mapping from prompt/auto-type
                style_infer = autotype_meta.get("specific_clothing_style") if autotype_meta else None
                if not style_infer and prompt_desc:
                    style_infer = _infer_style_from_text(prompt_desc)
                
                category_meta = _wardrobe_category_from_garment_type(resolved_type, style=style_infer)

                items.append({
                    "garment_id": idx,
                    "type": resolved_type,
                    "garment_type": resolved_type,
                    "type_source": type_source,
                    "detection_source": detection_source,
                    "promptDescription": prompt_desc,
                    "description": prompt_desc,
                    "style": category_meta["style"],
                    "category_key": category_meta["category_key"],
                    "primary_category_key": category_meta["primary_category_key"],
                    "url": None,
                    "bbox": bbox,
                    "crop": {
                        "width": crop_w,
                        "height": crop_h,
                        "area_ratio": crop_area_ratio,
                    },
                    "confidence": {
                        "yolo": float(hybrid_meta.get("yolo_confidence", inst.get("confidence", 0.0))),
                        "florence_type": float(hybrid_meta.get("florence_confidence", 0.0)),
                        "bbox_prior": float(hybrid_meta.get("bbox_prior", 0.0)),
                        "hybrid": float(hybrid_meta.get("score", 0.0)),
                    } if USE_FLORENCE_HYBRID_VERIFY else {
                        "yolo": float(inst.get("confidence", 0.0)),
                    },
                    "_image_obj": inst["image"],
                })
                if autotype_meta:
                    items[-1]["autotype"] = autotype_meta
                if detection_source == "human_parser":
                    items[-1]["parser_area_ratio"] = float(inst.get("parser_area_ratio", 0.0))

            deduped_items = _dedupe_items(items, iou_threshold=0.60)
            dedup_removed = len(items) - len(deduped_items)
            items = deduped_items

            # Collapse to single candidate when all remaining detections map to same type.
            unique_types = sorted({str(it.get("type")) for it in items if it.get("type")})
            collapsed_same_type = False
            if len(items) > 1 and len(unique_types) <= 1:
                best = max(items, key=_item_rank_score)
                items = [best]
                collapsed_same_type = True

            # Reindex ids after dedupe/collapse to keep selected_index stable for client.
            for new_idx, item in enumerate(items):
                item["garment_id"] = new_idx

            auto_selected_index = None
            if selected_index is None and requested_type and len(items) > 1:
                matched = [
                    idx for idx, item in enumerate(items)
                    if _normalize_garment_type(str(item.get("type"))) == requested_type
                ]
                if len(matched) == 1:
                    auto_selected_index = matched[0]

            # ── MULTI-ITEM SELECTION REQUIRED (400 with multipart) ──
            if ANALYZE_REQUIRE_SELECTION and len(items) > 1 and selected_index is None and auto_selected_index is None:
                total_s = round(time.time() - t0, 4)
                # Build item_breakdown for Flutter
                item_breakdown = []
                binary_parts = []
                for item in items:
                    pub = _to_public_item(item)
                    pub["rank"] = pub.get("garment_id", 0)
                    mapped_type = item.get("garment_type") or item.get("type", "unknown")
                    pub["garment_type"] = mapped_type
                    item_breakdown.append(pub)

                    img = item.get("_image_obj")
                    if img is not None:
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        part_name = f"item_{int(pub['rank'])}"
                        binary_parts.append({
                            "name": part_name,
                            "filename": f"{part_name}.png",
                            "content_type": "image/png",
                            "bytes": buf.getvalue(),
                        })

                data = {
                    "result": "WARNING",
                    "title": "Multiple Items Found",
                    "description": "Multiple garments were detected. Please select a type (top, bottom, dress) and re-upload to continue.",
                    "reason_codes": ["MULTI_ITEM_SELECTION_REQUIRED"],
                    "selection_required": True,
                    "total_garments_found": len(items),
                    "selection_hint": {
                        "expected_field": "garmentType",
                        "allowed_types": ["top", "bottom", "dress"],
                    },
                    "item_breakdown": item_breakdown,
                    "multipart_data": _build_multipart_parts(items=item_breakdown),
                    "latencies": {"total": total_s},
                    "processing_time_ms": int(total_s * 1000),
                }
                payload = _build_success_payload(data=data, status_code=400, message="")
                return _multipart_form_response(payload, binary_parts=binary_parts)

            selected_item = None
            if selected_index is not None:
                if selected_index < 0 or selected_index >= len(items):
                    payload = _build_error_payload(
                        title="Invalid Selection",
                        description=f"selected_index must be between 0 and {max(0, len(items) - 1)}.",
                        reason_codes=["INVALID_SELECTION_TYPE"],
                        status_code=400,
                    )
                    return _multipart_form_response(payload)
                selected_item = items[selected_index]
            elif auto_selected_index is not None:
                selected_item = items[auto_selected_index]
                selected_index = auto_selected_index
            elif items:
                selected_item = items[0]

            if selected_item and ANALYZE_EXTRACT_CLOTH:
                forced_type = requested_type if requested_type in {"top", "bottom", "dress", "outer"} else None
                selected_type = forced_type or _normalize_garment_type(str(selected_item.get("type")))
                if not selected_type:
                    caption_guess = _infer_type_from_caption(
                        str(selected_item.get("promptDescription") or selected_item.get("description") or "")
                    )
                    if caption_guess:
                        selected_item["type_original"] = selected_item.get("type")
                        selected_item["type"] = caption_guess
                        selected_item["type_source"] = "caption_autotype"
                        selected_type = caption_guess
                if not selected_type:
                    try:
                        auto_verdict = engine.florence.classify_garment_type(selected_item["_image_obj"], hint_type=requested_type)
                        auto_type = _normalize_garment_type(str(auto_verdict.get("type")))
                        if auto_type:
                            selected_item["type_original"] = selected_item.get("type")
                            selected_item["type"] = auto_type
                            selected_item["type_source"] = "florence_autotype"
                            selected_item["autotype_score"] = float(auto_verdict.get("score", 0.0))
                            selected_item["autotype_reason"] = str(auto_verdict.get("reason", ""))
                            selected_type = auto_type
                    except Exception as auto_type_err:
                        logger.warning(f"Automatic type resolution failed: {auto_type_err}")
                if not selected_type:
                    selected_type = "top"
                if forced_type and str(selected_item.get("type")) != forced_type:
                    selected_item["type_original"] = selected_item.get("type")
                    selected_item["type"] = forced_type
                    selected_item["type_source"] = "requested_type"

                if not ANALYZE_VTON_FALLBACK_ENABLED or not ANALYZE_VTON_CLOTH_ONLY_ENDPOINT:
                    payload = _build_error_payload(
                        title="Extraction Not Configured",
                        description="Garment extraction is not available. Please try again later.",
                        reason_codes=["EXTRACTION_FAILED"],
                        status_code=500,
                        result="ERROR",
                    )
                    return _multipart_form_response(payload)

                try:
                    vton_source_image, vton_bbox, vton_crop_mode = _prepare_vton_source_image(
                        full_image=img,
                        bbox=selected_item.get("bbox"),
                        garment_type=selected_type,
                        total_items=len(items),
                    )
                    selected_item["vton_crop_bbox"] = vton_bbox
                    selected_item["vton_crop_mode"] = vton_crop_mode
                    selected_crop_url = _write_temp_png(vton_source_image)
                except Exception as crop_save_err:
                    payload = _build_error_payload(
                        title="Invalid Image",
                        description="Could not process the image. Please upload a clearer photo.",
                        reason_codes=["INVALID_IMAGE"],
                        status_code=400,
                    )
                    return _multipart_form_response(payload)

                if ANALYZE_PROMPT_FROM_EXTRACTED:
                    selected_item["promptDescription"] = ""
                    selected_item["description"] = ""

                extracted_url = ""
                extraction_meta = {}
                try:
                    # Deep Analysis fix: If single item detected, use vto_mode=True to
                    # preserve person and prevent truncation during extraction.
                    is_single_item = len(items) <= 1
                    fallback = _run_vton_cloth_only_fallback(
                        selected_crop_url,
                        selected_type,
                        vto_mode=is_single_item
                    )
                    extracted_url = str(fallback.get("url") or "")
                    extraction_meta = dict(fallback.get("meta") or {})
                except Exception as fallback_err:
                    payload = _build_error_payload(
                        title="Extraction Failed",
                        description="Garment extraction failed. Please retry with a clearer image.",
                        reason_codes=["EXTRACTION_FAILED"],
                        status_code=400,
                    )
                    return _multipart_form_response(payload)

                if not extracted_url:
                    payload = _build_error_payload(
                        title="Extraction Failed",
                        description="No garment could be extracted. Please upload a clearer image.",
                        reason_codes=["EXTRACTION_FAILED"],
                        status_code=400,
                    )
                    return _multipart_form_response(payload)

                selected_item["url"] = extracted_url
                selected_item["output_image_url"] = extracted_url
                selected_item["output_image_source"] = str(extraction_meta.get("path", "vton_fallback"))
                selected_item["raw_image_url"] = str(fallback.get("raw_url") or extracted_url)
                selected_item["raw_image_source"] = "vton_raw"
                selected_item["extraction"] = extraction_meta
                selected_item["cloth_verified"] = selected_item["output_image_source"] == "vton_fallback"
                selected_item["cloth_verification_source"] = "vton_extraction"

                if ANALYZE_PROMPT_FROM_EXTRACTED:
                    try:
                        cloth_image = download_image(extracted_url)
                        caption_image = _flatten_rgba_on_white(cloth_image)
                        if ANALYZE_CAPTION_MODE == "detailed":
                            prompt_desc = engine.florence.describe_garment(caption_image)
                        else:
                            prompt_desc = engine.florence.describe_garment_short(caption_image)
                        prompt_desc = _sanitize_garment_description(prompt_desc)
                        selected_item["promptDescription"] = prompt_desc
                        selected_item["description"] = prompt_desc
                    except Exception as caption_err:
                        if ANALYZE_REQUIRE_EXTRACTED_PROMPT:
                            payload = _build_error_payload(
                                title="Extraction Failed",
                                description="Could not generate garment description. Please retry.",
                                reason_codes=["EXTRACTION_FAILED"],
                                status_code=400,
                            )
                            return _multipart_form_response(payload)
                        logger.warning(f"Prompt generation from extracted cloth failed: {caption_err}")

                if forced_type:
                    selected_item["extract_type_forced"] = forced_type

            progress_id = None
            progress_sync = None
            if selected_item:
                progress_id = str(uuid.uuid4())
                forced_type = requested_type if requested_type in {"top", "bottom", "dress", "outer"} else None
                selected_type = forced_type or _normalize_garment_type(str(selected_item.get("type"))) or "top"
                output_url = str(selected_item.get("url") or "")
                prompt_description = str(selected_item.get("promptDescription") or "")
                output_source = str(selected_item.get("output_image_source") or "")
                progress_meta = {
                    "selected_type": selected_type,
                    "requested_type": requested_type,
                    "outputImageSource": output_source or "crop",
                    "extraction": selected_item.get("extraction"),
                    "cloth_verified": bool(selected_item.get("cloth_verified")),
                    "cloth_verification_source": selected_item.get("cloth_verification_source"),
                    "detection_source": selected_item.get("detection_source"),
                    "type_source": selected_item.get("type_source"),
                    "bbox": selected_item.get("bbox"),
                    "crop": selected_item.get("crop"),
                }
                if output_source == "vton_fallback":
                    progress_sync = _sync_wardrobe_progress(
                        authorization=_authorization,
                        progress_id=progress_id,
                        output_url=output_url,
                        prompt_description=prompt_description,
                        metadata=progress_meta,
                    )
                else:
                    progress_sync = {
                        "enabled": True,
                        "synced": False,
                        "reason": "skipped_non_vton_output",
                        "id": progress_id,
                    }
                progress_id = str(progress_sync.get("id") or progress_id)
                selected_item["wardrobe_progress_id"] = progress_id
                selected_item["progress_sync"] = progress_sync

        # ── SUCCESS RESPONSE (200 with JSON) ──
        public_item = _to_public_item(selected_item) if selected_item else None
        total_s = round(time.time() - t0, 4)

        # Resolve category mapping for the selected item
        final_type = str(public_item.get("type", "top")) if public_item else "top"
        final_style = str(public_item.get("style", "")) if public_item else ""
        category_meta = _wardrobe_category_from_garment_type(final_type, style=final_style if final_style else None)

        reason_codes = ["SINGLE_ITEM"]
        extraction_path = str(public_item.get("output_image_source", "")) if public_item else ""
        if extraction_path == "vton_fallback":
            reason_codes.append("HIGH_OCCLUSION_FALLBACK")

        data = {
            "result": "ACCEPTED",
            "title": "Added To Wardrobe",
            "description": "Garment extracted successfully and ready for wardrobe save.",
            "reason_codes": reason_codes,
            "selection_required": False,
            "clothing_type": category_meta["style"],
            "category_key": category_meta["category_key"],
            "primary_category_key": category_meta["primary_category_key"],
            "style": category_meta["style"],
            "selected_type": final_type,
            "selected_item": public_item,
            "cloth_url": public_item.get("url") if public_item else None,
            "output_image_url": public_item.get("output_image_url", public_item.get("url")) if public_item else None,
            "extraction_path": extraction_path,
            "promptDescription": public_item.get("promptDescription") if public_item else None,
            "wardrobe_progress_id": public_item.get("wardrobe_progress_id") if public_item else None,
            "total_garments_found": len(items),
            "latencies": {"total": total_s},
            "processing_time_ms": int(total_s * 1000),
        }

        # Legacy fields for backward compatibility
        data["imageUrl"] = data["cloth_url"]
        data["progressId"] = data["wardrobe_progress_id"]

        payload = _build_success_payload(data=data, status_code=200, message="")
        return _multipart_form_response(payload)

    except HTTPException as he:
        # Convert old-style HTTPException to Flutter contract envelope
        detail = he.detail if isinstance(he.detail, dict) else {"message": str(he.detail)}
        status_msg = str(detail.get("status", detail.get("message", "Request failed")))
        payload = _build_error_payload(
            title="Request Failed",
            description=str(detail.get("message", status_msg)),
            reason_codes=[str(detail.get("reason", "REQUEST_FAILED")).upper()],
            status_code=he.status_code,
            result="ERROR" if he.status_code >= 500 else "REJECTED",
        )
        return _multipart_form_response(payload)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        payload = _build_error_payload(
            title="Server Error",
            description="An unexpected error occurred. Please try again.",
            reason_codes=["SERVER_ERROR"],
            status_code=500,
            result="ERROR",
            message=str(e),
        )
        return _multipart_form_response(payload)

@app.post("/v1/flux/tryon")
@app.post("/v1/flux2/tryon")
async def vto_tryon(request: VTORequest):
    """
    Performs Virtual Try-On using Vision-Guided Prompting (Flux 2.0).
    """
    t0 = time.time()
    try:
        # 1. Download Images
        user_img = download_image(request.user_image_url)
        garment_img = download_image(request.garment_image_url)

        async with gpu_semaphore:
            # 2. Vision Integration (Florence-2)
            # We "look" at the garment to get a precise description
            if USE_FLORENCE_DETAILED_PROMPT:
                garment_desc = engine.florence.describe_garment(garment_img)
            else:
                garment_desc = engine.florence.describe_garment_short(garment_img)

            # 3. Build Board (Collage)
            board = engine.board_builder.build_board([garment_img])

            # 4. Defensive Prompting (The Fix for Color Bleed)
            prompt = prompt_factory.build_dynamic_prompt(
                garment_descriptions=[garment_desc],
                user_description=request.user_top_description
            )

            # 5. Inference (Flux 2.0)
            result = engine.flux2.run_tryon(
                person_image=user_img,
                board_image=board,
                prompt=prompt,
                steps=request.steps,
                seed=request.seed
            )

            # 6. Upload Result
            res_buf = io.BytesIO()
            result["image"].save(res_buf, format="PNG")
            result_url = _upload_or_raise(res_buf.getvalue(), container=VTO_OUTPUT_CONTAINER)

        return {
            "status": "success",
            "result_url": result_url,
            "promptDescription": garment_desc,
            "prompt": prompt,
            "latency": result["latency"],
            "total_latency": time.time() - t0
        }
    except Exception as e:
        logger.error(f"Try-on failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gpu_concurrency": GPU_CONCURRENCY,
        "gpu_available": torch.cuda.is_available(),
        "vram_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "feature_flags": {
            "use_florence_hybrid_verify": USE_FLORENCE_HYBRID_VERIFY,
            "use_florence_detailed_prompt": USE_FLORENCE_DETAILED_PROMPT,
            "analyze_extract_cloth": ANALYZE_EXTRACT_CLOTH,
            "analyze_extract_mode": "forced_vton",
            "analyze_require_extracted_prompt": ANALYZE_REQUIRE_EXTRACTED_PROMPT,
            "analyze_require_mirrored_vton_url": ANALYZE_REQUIRE_MIRRORED_VTON_URL,
            "analyze_vton_fallback_enabled": ANALYZE_VTON_FALLBACK_ENABLED,
            "analyze_vton_segmentation_free": ANALYZE_VTON_SEGMENTATION_FREE,
            "analyze_vton_reject_source_passthrough": ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH,
            "analyze_vton_use_showroom_person": ANALYZE_VTON_USE_SHOWROOM_PERSON,
            "analyze_vton_dress_use_full_image": ANALYZE_VTON_DRESS_USE_FULL_IMAGE,
            "analyze_vton_single_item_use_full_image": ANALYZE_VTON_SINGLE_ITEM_USE_FULL_IMAGE,
            "analyze_vton_top_zoom_enabled": ANALYZE_VTON_TOP_ZOOM_ENABLED,
            "analyze_vton_bottom_upscale_enabled": ANALYZE_VTON_BOTTOM_UPSCALE_ENABLED,
            "analyze_garment_postprocess_enabled": ANALYZE_GARMENT_POSTPROCESS_ENABLED,
            "analyze_garment_target_aspect": f"{ANALYZE_GARMENT_TARGET_ASPECT_W}:{ANALYZE_GARMENT_TARGET_ASPECT_H}",
            "analyze_garment_enhance_enabled": ANALYZE_GARMENT_ENHANCE_ENABLED,
            "wardrobe_progress_sync_enabled": ENABLE_WARDROBE_PROGRESS_SYNC,
            "wardrobe_progress_include_input_image": WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE,
        },
        "models": engine.model_status(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
