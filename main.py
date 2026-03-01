import io
import time
import logging
import asyncio
import os
import uuid
import tempfile
import re
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
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

_REMBG_SESSION = None
_REMBG_SESSION_LOCK = threading.Lock()

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
ANALYZE_MAX_FILE_BYTES = max(1, _env_int("ANALYZE_MAX_FILE_BYTES", 3 * 1024 * 1024))
ANALYZE_BLUR_CHECK_ENABLED = os.getenv("ANALYZE_BLUR_CHECK_ENABLED", "1") == "1"
ANALYZE_BLUR_MIN_FOCUS_SCORE = _env_float("ANALYZE_BLUR_MIN_FOCUS_SCORE", 22.0)
ANALYZE_BLUR_FOCUS_MAX_EDGE = max(256, _env_int("ANALYZE_BLUR_FOCUS_MAX_EDGE", 1024))
ANALYZE_MIN_ACCEPT_CONFIDENCE = _env_float("ANALYZE_MIN_ACCEPT_CONFIDENCE", 0.25)
ANALYZE_ENABLE_PARSER_SPLIT = os.getenv("ANALYZE_ENABLE_PARSER_SPLIT", "0") == "1"
ANALYZE_PARSER_MIN_AREA_RATIO = _env_float("ANALYZE_PARSER_MIN_AREA_RATIO", 0.015)
ANALYZE_PARSER_PAD = max(0, _env_int("ANALYZE_PARSER_PAD", 12))
ANALYZE_ENABLE_HUMAN_PARSER = os.getenv("ANALYZE_ENABLE_HUMAN_PARSER", "1") == "1"
ANALYZE_ENABLE_HEURISTIC_SPLIT = os.getenv("ANALYZE_ENABLE_HEURISTIC_SPLIT", "1") == "1"
ANALYZE_TIGHTEN_SPLIT_CROPS = os.getenv("ANALYZE_TIGHTEN_SPLIT_CROPS", "1") == "1"
ANALYZE_TIGHTEN_SPLIT_PAD = max(0, _env_int("ANALYZE_TIGHTEN_SPLIT_PAD", 8))
ANALYZE_TIGHTEN_SPLIT_MIN_PIXELS = max(32, _env_int("ANALYZE_TIGHTEN_SPLIT_MIN_PIXELS", 48))
ANALYZE_TIGHTEN_BOTTOM_TOP_EXTRA_RATIO = _env_float("ANALYZE_TIGHTEN_BOTTOM_TOP_EXTRA_RATIO", 0.12)
ANALYZE_TIGHTEN_BOTTOM_BOTTOM_EXTRA_RATIO = _env_float("ANALYZE_TIGHTEN_BOTTOM_BOTTOM_EXTRA_RATIO", 0.05)
ANALYZE_TIGHTEN_BOTTOM_TOP_MAX_OVERLAP_PX = max(0, _env_int("ANALYZE_TIGHTEN_BOTTOM_TOP_MAX_OVERLAP_PX", 72))
ANALYZE_TIGHTEN_BOTTOM_MAX_DOWN_SHIFT_RATIO = max(0.0, _env_float("ANALYZE_TIGHTEN_BOTTOM_MAX_DOWN_SHIFT_RATIO", 0.08))
ANALYZE_TIGHTEN_BOTTOM_MAX_GAP_FROM_TOP_PX = max(0, _env_int("ANALYZE_TIGHTEN_BOTTOM_MAX_GAP_FROM_TOP_PX", 96))
ANALYZE_HEURISTIC_MIN_HEIGHT_RATIO = _env_float("ANALYZE_HEURISTIC_MIN_HEIGHT_RATIO", 0.78)
ANALYZE_HEURISTIC_TOP_PORTION = _env_float("ANALYZE_HEURISTIC_TOP_PORTION", 0.52)
ANALYZE_HEURISTIC_TOP_TRIM_PX = max(0, _env_int("ANALYZE_HEURISTIC_TOP_TRIM_PX", 0))
ANALYZE_HEURISTIC_BOTTOM_OVERLAP_PX = max(0, _env_int("ANALYZE_HEURISTIC_BOTTOM_OVERLAP_PX", 32))
ANALYZE_HEURISTIC_BOTTOM_OVERLAP_RATIO = _env_float("ANALYZE_HEURISTIC_BOTTOM_OVERLAP_RATIO", 0.11)
ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO = _env_float("ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO", 0.44)
ANALYZE_HEURISTIC_MAX_WIDTH_RATIO = _env_float("ANALYZE_HEURISTIC_MAX_WIDTH_RATIO", 0.98)
ANALYZE_FORCE_FULLBODY_SPLIT_ON_SAME_TYPE = os.getenv("ANALYZE_FORCE_FULLBODY_SPLIT_ON_SAME_TYPE", "1") == "1"
ANALYZE_FORCE_FULLBODY_SPLIT_MIN_HEIGHT_RATIO = _env_float("ANALYZE_FORCE_FULLBODY_SPLIT_MIN_HEIGHT_RATIO", 0.72)
ANALYZE_COLLAPSE_SAME_TYPE = os.getenv("ANALYZE_COLLAPSE_SAME_TYPE", "0") == "1"
ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU = _env_float("ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU", 0.85)
ANALYZE_EXTRACT_CLOTH = os.getenv("ANALYZE_EXTRACT_CLOTH", "1") == "1"
ANALYZE_PROMPT_FROM_EXTRACTED = os.getenv("ANALYZE_PROMPT_FROM_EXTRACTED", "1") == "1"
ANALYZE_EXTRACT_MIN_MASK_RATIO = _env_float("ANALYZE_EXTRACT_MIN_MASK_RATIO", 0.01)
ANALYZE_EXTRACT_RELAXED_RESCUE = os.getenv("ANALYZE_EXTRACT_RELAXED_RESCUE", "1") == "1"
ANALYZE_EXTRACT_EDGE_FEATHER_PX = max(0, _env_int("ANALYZE_EXTRACT_EDGE_FEATHER_PX", 1))
ANALYZE_EXTRACT_COMPONENT_MIN_RATIO = _env_float("ANALYZE_EXTRACT_COMPONENT_MIN_RATIO", 0.0007)
ANALYZE_EXTRACT_KEEP_DILATE = max(0, _env_int("ANALYZE_EXTRACT_KEEP_DILATE", 3))
ANALYZE_EXTRACT_PARSER_KILL_DILATE = max(1, _env_int("ANALYZE_EXTRACT_PARSER_KILL_DILATE", 2))
ANALYZE_GARMENT_HOLE_FILL_MAX_PIXELS = max(0, _env_int("ANALYZE_GARMENT_HOLE_FILL_MAX_PIXELS", 7000))
ANALYZE_EXTRACT_MAX_BODY_RATIO = max(0.0, _env_float("ANALYZE_EXTRACT_MAX_BODY_RATIO", 0.008))
ANALYZE_EXTRACT_BODY_STRIP_DILATE = max(0, _env_int("ANALYZE_EXTRACT_BODY_STRIP_DILATE", 3))
ANALYZE_VTON_MAX_SHOWROOM_MAE = max(0.0, _env_float("ANALYZE_VTON_MAX_SHOWROOM_MAE", 3.5))
ANALYZE_VTON_MIN_SHOWROOM_CHANGE_RATIO = max(0.0, _env_float("ANALYZE_VTON_MIN_SHOWROOM_CHANGE_RATIO", 0.02))
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
ANALYZE_VTON_MIRROR_RAW_OUTPUT = os.getenv("ANALYZE_VTON_MIRROR_RAW_OUTPUT", "0") == "1"
ANALYZE_VTON_MAX_SOURCE_MAE = _env_float("ANALYZE_VTON_MAX_SOURCE_MAE", 2.0)
ANALYZE_VTON_USE_SHOWROOM_PERSON = os.getenv("ANALYZE_VTON_USE_SHOWROOM_PERSON", "0") == "1"
ANALYZE_VTON_SHOWROOM_PERSON_IMAGE_URL = os.getenv("ANALYZE_VTON_SHOWROOM_PERSON_IMAGE_URL", "").strip()
ANALYZE_VTON_DRESS_USE_FULL_IMAGE = os.getenv("ANALYZE_VTON_DRESS_USE_FULL_IMAGE", "1") == "1"
ANALYZE_VTON_SINGLE_ITEM_USE_FULL_IMAGE = os.getenv("ANALYZE_VTON_SINGLE_ITEM_USE_FULL_IMAGE", "1") == "1"
ANALYZE_VTON_CROP_PAD_RATIO = _env_float("ANALYZE_VTON_CROP_PAD_RATIO", 0.18)
ANALYZE_VTON_CROP_PAD_RATIO_DRESS = _env_float("ANALYZE_VTON_CROP_PAD_RATIO_DRESS", 0.28)
ANALYZE_VTON_CROP_BOTTOM_EXTRA_RATIO_DRESS = _env_float("ANALYZE_VTON_CROP_BOTTOM_EXTRA_RATIO_DRESS", 0.32)
ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM = _env_float("ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM", 0.12)
ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI = _env_float("ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI", 0.12)
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

def _focus_score(image: Image.Image) -> float:
    """
    Lightweight blur metric: higher score means sharper image.
    Score is computed on a normalized max-edge to avoid high-resolution bias.
    """
    gray = image.convert("L")
    w, h = gray.size
    max_edge = max(w, h)
    if max_edge > ANALYZE_BLUR_FOCUS_MAX_EDGE:
        scale = float(ANALYZE_BLUR_FOCUS_MAX_EDGE) / float(max_edge)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        gray = gray.resize((nw, nh), Image.BICUBIC)

    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    gy, gx = np.gradient(arr)
    mag = np.sqrt((gx * gx) + (gy * gy))
    return float(np.var(mag))

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
    max_width_ratio = max(0.60, min(0.995, ANALYZE_HEURISTIC_MAX_WIDTH_RATIO))
    if width_ratio < 0.15 or width_ratio > max_width_ratio:
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


def _instance_area(inst: dict) -> int:
    bbox = inst.get("bbox") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0
    try:
        x0, y0, x1, y1 = [int(v) for v in bbox]
    except Exception:
        return 0
    return max(1, x1 - x0) * max(1, y1 - y0)


def _instance_height_ratio(inst: dict, image_height: int) -> float:
    if image_height <= 0:
        return 0.0
    bbox = inst.get("bbox") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        _, y0, _, y1 = [int(v) for v in bbox]
    except Exception:
        return 0.0
    return float(max(1, y1 - y0)) / float(max(1, image_height))


def _largest_instance(instances: list[dict]) -> Optional[dict]:
    if not instances:
        return None
    return max(instances, key=_instance_area)


def _should_force_fullbody_split(instances: list[dict], image_height: int) -> bool:
    if not ANALYZE_FORCE_FULLBODY_SPLIT_ON_SAME_TYPE or len(instances) < 2:
        return False

    # Keep parser/heuristic outputs stable; this guard is only for raw detector outputs.
    if any(str(inst.get("source", "")) in {"human_parser", "heuristic_split"} for inst in instances):
        return False

    normalized_types = {
        _normalize_garment_type(str(inst.get("label") or inst.get("type") or ""))
        for inst in instances
    }
    normalized_types.discard(None)
    # Force split only when detector effectively returned one semantic class.
    if len(normalized_types) > 1:
        return False

    base_inst = _largest_instance(instances)
    if base_inst is None:
        return False
    return _instance_height_ratio(base_inst, image_height) >= ANALYZE_FORCE_FULLBODY_SPLIT_MIN_HEIGHT_RATIO

def _tighten_instances_with_parser_masks(image: Image.Image, instances: list[dict]) -> list[dict]:
    """
    Tighten coarse split boxes using parser category masks.
    This primarily improves heuristic top/bottom splits that are often too loose.
    """
    if not ANALYZE_TIGHTEN_SPLIT_CROPS or engine.parser is None or not instances:
        return instances

    try:
        parsing = engine.parser.parse(image)
        category_masks = engine.parser.build_category_masks(parsing)
    except Exception as err:
        logger.warning(f"Parser-based crop tightening skipped due parser error: {err}")
        return instances

    top_boxes: list[tuple[int, int, int, int]] = []
    for inst in instances:
        inst_type = _normalize_garment_type(str(inst.get("label") or inst.get("type") or ""))
        if inst_type not in {"top", "outer"}:
            continue
        bbox = inst.get("bbox") or [0, 0, image.width, image.height]
        try:
            tx0, ty0, tx1, ty1 = [int(v) for v in bbox]
        except Exception:
            continue
        tx0 = max(0, min(tx0, image.width - 1))
        tx1 = max(tx0 + 1, min(tx1, image.width))
        ty0 = max(0, min(ty0, image.height - 1))
        ty1 = max(ty0 + 1, min(ty1, image.height))
        ref_y0, ref_y1 = ty0, ty1
        tmask = category_masks.get(inst_type or "")
        if tmask is not None and tmask.shape[:2] == (image.height, image.width):
            t_region = tmask[ty0:ty1, tx0:tx1]
            t_local = _bbox_from_mask(t_region)
            if t_local:
                _, ly0, _, ly1 = [int(v) for v in t_local]
                ref_y0 = ty0 + ly0
                ref_y1 = ty0 + ly1
        top_boxes.append((tx0, ref_y0, tx1, ref_y1))

    tightened: list[dict] = []
    for inst in instances:
        inst_type = _normalize_garment_type(str(inst.get("label") or inst.get("type") or ""))
        source = str(inst.get("source", ""))
        mask = category_masks.get(inst_type or "")
        bbox = inst.get("bbox") or [0, 0, image.width, image.height]
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0 = max(0, min(x0, image.width - 1))
        y0 = max(0, min(y0, image.height - 1))
        x1 = max(x0 + 1, min(x1, image.width))
        y1 = max(y0 + 1, min(y1, image.height))

        updated = dict(inst)
        if mask is None or mask.shape[:2] != (image.height, image.width):
            updated["bbox"] = [x0, y0, x1, y1]
            updated["image"] = image.crop((x0, y0, x1, y1))
            tightened.append(updated)
            continue

        region = mask[y0:y1, x0:x1]
        if region.size == 0 or int(region.sum()) <= 0:
            updated["bbox"] = [x0, y0, x1, y1]
            updated["image"] = image.crop((x0, y0, x1, y1))
            tightened.append(updated)
            continue

        local_bbox = _bbox_from_mask(region)
        if not local_bbox:
            updated["bbox"] = [x0, y0, x1, y1]
            updated["image"] = image.crop((x0, y0, x1, y1))
            tightened.append(updated)
            continue

        lx0, ly0, lx1, ly1 = [int(v) for v in local_bbox]
        gx0 = x0 + lx0
        gy0 = y0 + ly0
        gx1 = x0 + lx1
        gy1 = y0 + ly1
        tight_bbox = _expand_bbox([gx0, gy0, gx1, gy1], image.width, image.height, ANALYZE_TIGHTEN_SPLIT_PAD)
        tx0, ty0, tx1, ty1 = [int(v) for v in tight_bbox]

        bottom_adjustments: list[str] = []
        # Bottom crops need upward context for waistband, but must avoid leaking top garment context.
        if inst_type == "bottom":
            th = max(1, ty1 - ty0)
            top_extra = max(ANALYZE_TIGHTEN_SPLIT_PAD, int(th * max(0.0, ANALYZE_TIGHTEN_BOTTOM_TOP_EXTRA_RATIO)))
            bottom_extra = max(0, int(th * max(0.0, ANALYZE_TIGHTEN_BOTTOM_BOTTOM_EXTRA_RATIO)))
            ty0 = max(0, ty0 - top_extra)
            ty1 = min(image.height, ty1 + bottom_extra)
            bottom_adjustments.append("bottom_context_expanded")

            # Guard against parser under-segmentation: keep lower-body crop from shifting too far down.
            base_h = max(1, y1 - y0)
            max_down_shift = int(base_h * max(0.0, ANALYZE_TIGHTEN_BOTTOM_MAX_DOWN_SHIFT_RATIO))
            max_ty0 = y0 + max_down_shift
            if ty0 > max_ty0:
                ty0 = max_ty0
                bottom_adjustments.append("bottom_upper_recovered")

            best_top_y1 = None
            for px0, py0, px1, py1 in top_boxes:
                inter_w = max(0, min(tx1, px1) - max(tx0, px0))
                min_w = max(1, min(tx1 - tx0, px1 - px0))
                overlap = float(inter_w) / float(min_w)
                if overlap < 0.20:
                    continue
                if best_top_y1 is None or py1 > best_top_y1:
                    best_top_y1 = py1
            if best_top_y1 is not None:
                max_gap_ty0 = min(image.height - 1, int(best_top_y1) + ANALYZE_TIGHTEN_BOTTOM_MAX_GAP_FROM_TOP_PX)
                if ty0 > max_gap_ty0:
                    ty0 = max_gap_ty0
                    bottom_adjustments.append("top_gap_recovered")

                min_ty0 = max(0, int(best_top_y1) - ANALYZE_TIGHTEN_BOTTOM_TOP_MAX_OVERLAP_PX)
                if ty0 < min_ty0:
                    ty0 = min_ty0
                    bottom_adjustments.append("top_overlap_clamped")

        if (tx1 - tx0) < ANALYZE_TIGHTEN_SPLIT_MIN_PIXELS or (ty1 - ty0) < ANALYZE_TIGHTEN_SPLIT_MIN_PIXELS:
            updated["bbox"] = [x0, y0, x1, y1]
            updated["image"] = image.crop((x0, y0, x1, y1))
            tightened.append(updated)
            continue

        old_area = max(1, (x1 - x0) * (y1 - y0))
        new_area = max(1, (tx1 - tx0) * (ty1 - ty0))
        # Only accept refinement if it is meaningfully tighter (or source is heuristic split).
        if source == "heuristic_split" or new_area <= int(old_area * 0.94):
            updated["bbox"] = [tx0, ty0, tx1, ty1]
            updated["image"] = image.crop((tx0, ty0, tx1, ty1))
            updated["tighten_source"] = "parser_mask"
            if inst_type == "bottom":
                updated["tighten_adjustment"] = ",".join(bottom_adjustments) if bottom_adjustments else "bottom_context_expanded"
        else:
            updated["bbox"] = [x0, y0, x1, y1]
            updated["image"] = image.crop((x0, y0, x1, y1))
        tightened.append(updated)

    return tightened

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


def _should_collapse_same_type(items: list[dict]) -> bool:
    if len(items) < 2:
        return False
    if not ANALYZE_COLLAPSE_SAME_TYPE:
        return False
    # Never collapse intentional split candidates.
    if any(str(it.get("detection_source", "")) in {"heuristic_split", "human_parser"} for it in items):
        return False

    bboxes = []
    for item in items:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        bboxes.append(tuple(int(v) for v in bbox))

    min_iou = 1.0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            min_iou = min(min_iou, bbox_iou(bboxes[i], bboxes[j]))
    return min_iou >= ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU

def _is_shorts_like_caption(text: str) -> bool:
    t = (text or "").lower()
    keys = ("shorts", "cargo shorts", "bermuda", "skirt", "mini skirt", "midi skirt", "maxi skirt")
    return any(k in t for k in keys)

def _infer_type_from_caption(text: str) -> Optional[str]:
    t = (text or "").lower()
    if not t:
        return None
    # Ignore generic positional phrase to avoid false garment type "top".
    t = t.replace("on top of", " ")
    if any(k in t for k in ("dress", "gown", "kurti", "one-piece", "one piece", "jumpsuit")):
        return "dress"
    if any(k in t for k in ("jacket", "coat", "blazer", "hoodie", "cardigan", "outerwear")):
        return "outer"
    if any(k in t for k in ("pant", "pants", "trouser", "trousers", "jean", "jeans", "shorts", "skirt", "bottom")):
        return "bottom"
    if any(k in t for k in ("shirt", "t-shirt", "tshirt", "tee", "blouse", "crop top", "tank top", "sports top", "camisole")):
        return "top"
    return None

def _caption_garment_signal(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    garment_terms = (
        "dress", "gown", "kurti", "jumpsuit",
        "shirt", "t-shirt", "tee", "blouse", "hoodie", "sweater", "cardigan", "jacket", "coat", "blazer",
        "pant", "pants", "trouser", "trousers", "jean", "jeans", "shorts", "skirt", "sweatpants", "leggings",
        "camisole", "tank top", "crop top", "bodysuit",
    )
    return any(term in t for term in garment_terms)

def _caption_non_garment_signal(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    non_garment_terms = (
        "table", "chair", "sofa", "couch", "bench", "desk", "lamp", "cabinet",
        "plant", "tree", "flower", "grass", "garden", "field",
        "road", "street", "building", "house", "wall", "floor", "sky", "cloud",
        "car", "bicycle", "motorcycle", "bus", "truck",
    )
    return any(term in t for term in non_garment_terms)

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

def _product_prompt_description(
    text: str,
    *,
    garment_type: Optional[str] = None,
    style: Optional[str] = None,
    category_key: Optional[str] = None,
) -> str:
    desc = _sanitize_garment_description(text or "")
    lowered = desc.lower()
    human_terms = (
        " woman ",
        " man ",
        " person ",
        " model ",
        " mannequin ",
        " showroom ",
        " wearing ",
        " posing ",
        " standing ",
    )
    padded = f" {lowered} "
    if any(term in padded for term in human_terms):
        preferred = (style or "").strip()
        if not preferred or preferred.lower() in {"top", "bottom", "dress", "outerwear", "unknown"}:
            ck = (category_key or "").strip().replace("_", " ")
            if ck and ck.lower() not in {"top", "bottom", "dress", "outerwear", "unknown"}:
                preferred = ck
            else:
                gt = _normalize_garment_type(garment_type) or "top"
                preferred = {
                    "top": "Top",
                    "bottom": "Bottom",
                    "dress": "Dress",
                    "outer": "Outerwear",
                }.get(gt, "Garment")
        preferred = " ".join(preferred.split())
        if preferred:
            return f"{preferred[0].upper() + preferred[1:]}."
        return "Garment."
    if desc and not desc.endswith("."):
        desc = f"{desc}."
    return desc or "Garment."

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

def _get_rembg_isnet_session():
    """
    Reuse a single rembg ISNet session across requests to avoid per-request
    model/session initialization overhead.
    """
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    with _REMBG_SESSION_LOCK:
        if _REMBG_SESSION is None:
            from rembg import new_session
            _REMBG_SESSION = new_session("isnet-general-use")
    return _REMBG_SESSION

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
    elif garment_type == "bottom":
        top_extra_ratio = max(0.0, ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM)
        if total_items > 1:
            top_extra_ratio = min(top_extra_ratio, max(0.0, ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI))
            y_pad_top = min(y_pad_top, int(bh * top_extra_ratio))
        else:
            y_pad_top = max(y_pad_top, int(bh * top_extra_ratio))

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
        from scipy.ndimage import binary_fill_holes, label
        alpha = image.split()[-1]
        alpha_arr = np.asarray(alpha)
        # Use a high threshold to find true interior holes and only fill smaller ones.
        mask = (alpha_arr > 50)
        filled = binary_fill_holes(mask)
        holes = filled & (~mask)
        new_alpha = alpha_arr.copy()
        if np.any(holes):
            labeled, num_labels = label(holes)
            for idx in range(1, int(num_labels) + 1):
                hole_area = int(np.sum(labeled == idx))
                if hole_area <= ANALYZE_GARMENT_HOLE_FILL_MAX_PIXELS:
                    new_alpha[labeled == idx] = 255
        image.putalpha(Image.fromarray(new_alpha.astype(np.uint8), mode="L"))
    except Exception as hole_e:
        logger.warning(f"Simple hole filling failed: {hole_e}")

    # 0.5 Edge cleanup: reduce fuzzy borders and retain cleaner garment silhouette.
    try:
        alpha_arr = np.asarray(image.split()[-1]).astype(np.uint8)
        edge_mask = alpha_arr > max(16, ANALYZE_GARMENT_ALPHA_THRESHOLD)
        edge_mask = binary_open(edge_mask, 3)
        edge_mask = binary_close(edge_mask, 3)
        alpha_clean = build_soft_alpha(edge_mask, feather_px=ANALYZE_EXTRACT_EDGE_FEATHER_PX)
        image.putalpha(Image.fromarray(alpha_clean.astype(np.uint8), mode="L"))
    except Exception as edge_e:
        logger.warning(f"Alpha edge cleanup failed: {edge_e}")

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

def _pixel_change_ratio(source_bytes: bytes, output_bytes: bytes, size: int = 256, delta: float = 12.0) -> float:
    src_img = _flatten_rgba_on_white(Image.open(io.BytesIO(source_bytes)))
    out_img = _flatten_rgba_on_white(Image.open(io.BytesIO(output_bytes)))
    target = (int(size), int(size))
    src_arr = np.asarray(src_img.resize(target, Image.BICUBIC), dtype=np.float32)
    out_arr = np.asarray(out_img.resize(target, Image.BICUBIC), dtype=np.float32)
    if src_arr.ndim != 3 or out_arr.ndim != 3:
        return 0.0
    diff = np.mean(np.abs(src_arr - out_arr), axis=2)
    return float(np.mean(diff > float(delta)))

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

def _body_leakage_stats(rgba_image: Image.Image) -> dict:
    if engine.parser is None:
        return {"ratio": 0.0, "visible_pixels": 0, "body_pixels": 0, "body_mask": None}

    rgba = rgba_image.convert("RGBA")
    arr = np.asarray(rgba)
    alpha = arr[:, :, 3]
    visible_mask = alpha > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD)
    visible_pixels = int(np.sum(visible_mask))
    if visible_pixels <= 0:
        return {"ratio": 0.0, "visible_pixels": 0, "body_pixels": 0, "body_mask": None}

    parse_input = _flatten_rgba_on_white(rgba)
    parsing = engine.parser.parse(parse_input)
    # Segformer labels: exclude all body/face/hair/limbs/shoes pixels.
    body_ids = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16]
    body_mask = np.isin(parsing, body_ids) & visible_mask
    body_pixels = int(np.sum(body_mask))
    ratio = float(body_pixels) / float(max(1, visible_pixels))
    return {
        "ratio": ratio,
        "visible_pixels": visible_pixels,
        "body_pixels": body_pixels,
        "body_mask": body_mask,
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
        from rembg import remove

        # STEP 1: Neural Background Removal (ISNet)
        session = _get_rembg_isnet_session()
        rembg_result = remove(crop.convert("RGB"), session=session, post_process_mask=True)
        rembg_alpha = np.array(rembg_result)[:, :, 3].astype(np.uint8)
        logger.info(f"rembg alpha coverage: {rembg_alpha.mean():.1f}")

        # STEP 2: Body Part & Cross-Category Kill (Surgical Isolation)
        # We use Segformer B2 Clothes labels: 4:Upper, 6:Pants, 5:Skirt, 7:Dress, 8:Belt, 9/10:Shoes, 11:Face, 12/13:Legs, 14/15:Arms
        parsing = engine.parser.parse(crop)

        # Always kill body parts, hair, and shoes to isolate the floating garment.
        base_body_kill_ids = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16]
        kill_ids = list(base_body_kill_ids)
        rescue_mode = "strict"
        type_keep_map = {
            "top": [4, 17],
            "outer": [4, 17, 8],
            "bottom": [5, 6, 8],
            "dress": [7, 4, 5, 6, 8, 17],
        }
        strict_keep_ids = type_keep_map.get(selected_type, [4, 5, 6, 7, 8, 17])

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

        # STEP 3: Fill holes in the rembg mask BEFORE combining
        # This prevents crocheted/mesh tops from being dissolved into fragments
        inv_rembg = cv2.bitwise_not(rembg_alpha)
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(inv_rembg)
        rembg_filled = rembg_alpha.copy()
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 5000:
                rembg_filled[labels_im == i] = 255

        def _compose_alpha_from_seed(seed_alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            seed_mask = seed_alpha > 0
            seed_mask = binary_open(seed_mask, 3)
            seed_mask = binary_close(seed_mask, 3)

            # Dust removal (eliminate isolated floating specks).
            num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats((seed_mask.astype(np.uint8) * 255))
            clean_alpha = np.zeros_like(seed_alpha, dtype=np.uint8)
            area_min = max(160, int(total_pixels * ANALYZE_EXTRACT_COMPONENT_MIN_RATIO))
            for j in range(1, num_l):
                if int(stats_l[j, cv2.CC_STAT_AREA]) >= area_min:
                    clean_alpha[labels_l == j] = 255

            alpha_local = build_soft_alpha(clean_alpha > 0, feather_px=ANALYZE_EXTRACT_EDGE_FEATHER_PX)
            final_mask_local = alpha_local > max(8, ANALYZE_EXTRACT_EDGE_FEATHER_PX * 6)
            area_ratio_local = float(final_mask_local.sum()) / total_pixels
            return alpha_local, final_mask_local, area_ratio_local

        def _compose_alpha_with_kill(active_kill_ids: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
            parser_kill = np.isin(parsing, active_kill_ids).astype(np.uint8) * 255
            kill_k = max(1, int(ANALYZE_EXTRACT_PARSER_KILL_DILATE))
            parser_kill = cv2.dilate(parser_kill, np.ones((kill_k, kill_k), np.uint8), iterations=1)

            final_alpha = rembg_filled.copy()
            final_alpha[parser_kill > 0] = 0

            # Type-aware keep-region gating removes residual body at waist/hands/legs.
            keep_mask = np.isin(parsing, strict_keep_ids).astype(np.uint8) * 255
            keep_dilate = max(0, int(ANALYZE_EXTRACT_KEEP_DILATE))
            if keep_dilate > 0:
                keep_mask = cv2.dilate(keep_mask, np.ones((keep_dilate, keep_dilate), np.uint8), iterations=1)
            final_alpha[keep_mask == 0] = 0
            return _compose_alpha_from_seed(final_alpha)

        def _compose_alpha_with_keep(keep_ids: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
            keep_mask = np.isin(parsing, keep_ids).astype(np.uint8) * 255
            keep_mask = cv2.dilate(keep_mask, np.ones((2, 2), np.uint8), iterations=1)
            final_alpha = rembg_filled.copy()
            final_alpha[keep_mask == 0] = 0
            return _compose_alpha_from_seed(final_alpha)

        # STEP 4/5/6: strict parser-aware extraction.
        alpha, final_mask, final_area_ratio = _compose_alpha_with_kill(kill_ids)

        # Guard against intermittent parser over-kill producing empty masks.
        if ANALYZE_EXTRACT_RELAXED_RESCUE and final_area_ratio < ANALYZE_EXTRACT_MIN_MASK_RATIO:
            keep_ids = strict_keep_ids
            alpha_keep, mask_keep, area_keep = _compose_alpha_with_keep(keep_ids)
            if area_keep > final_area_ratio:
                logger.warning(
                    "Extraction strict mask too small (type=%s strict=%.6f). Using parser-keep rescue mask %.6f.",
                    selected_type,
                    final_area_ratio,
                    area_keep,
                )
                alpha, final_mask, final_area_ratio = alpha_keep, mask_keep, area_keep
                rescue_mode = "parser_keep"

            # If parser keep still fails, relax to body-only kill as a broader rescue.
            if final_area_ratio < ANALYZE_EXTRACT_MIN_MASK_RATIO:
                alpha_relaxed, mask_relaxed, area_relaxed = _compose_alpha_with_kill(base_body_kill_ids)
                if area_relaxed > final_area_ratio:
                    logger.warning(
                        "Extraction parser-keep rescue too small (type=%s current=%.6f). Using body-only rescue mask %.6f.",
                        selected_type,
                        final_area_ratio,
                        area_relaxed,
                    )
                    alpha, final_mask, final_area_ratio = alpha_relaxed, mask_relaxed, area_relaxed
                    rescue_mode = "body_only"

            # Spatial guard for broad rescue masks on two-piece categories.
            if rescue_mode in {"body_only", "rembg_only"} and final_area_ratio > 0.0:
                ys, xs = np.where(final_mask)
                if len(xs) and len(ys):
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    h_mask = max(1, y1 - y0)
                    if selected_type in {"top", "outer"}:
                        cut_y = y0 + int(h_mask * 0.72)
                        final_mask[cut_y:, :] = False
                        alpha = np.where(final_mask, alpha, 0).astype(np.uint8)
                        final_area_ratio = float(final_mask.sum()) / total_pixels
                        rescue_mode = f"{rescue_mode}_upper_guard"
                    elif selected_type == "bottom":
                        cut_y = y0 + int(h_mask * 0.28)
                        final_mask[:cut_y, :] = False
                        alpha = np.where(final_mask, alpha, 0).astype(np.uint8)
                        final_area_ratio = float(final_mask.sum()) / total_pixels
                        rescue_mode = f"{rescue_mode}_lower_guard"

            if final_area_ratio <= 0.0:
                rembg_only_alpha = cv2.GaussianBlur(rembg_filled, (3, 3), 0)
                rembg_only_mask = rembg_only_alpha > 10
                rembg_only_area = float(rembg_only_mask.sum()) / total_pixels
                if rembg_only_area > final_area_ratio:
                    logger.warning(
                        "Extraction mask still empty after parser/body rescue (type=%s). Using rembg-only rescue %.6f.",
                        selected_type,
                        rembg_only_area,
                    )
                    alpha, final_mask, final_area_ratio = rembg_only_alpha, rembg_only_mask, rembg_only_area
                    rescue_mode = "rembg_only"

            if rescue_mode == "rembg_only" and final_area_ratio > 0.0:
                ys, xs = np.where(final_mask)
                if len(xs) and len(ys):
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    h_mask = max(1, y1 - y0)
                    if selected_type in {"top", "outer"}:
                        cut_y = y0 + int(h_mask * 0.72)
                        final_mask[cut_y:, :] = False
                        alpha = np.where(final_mask, alpha, 0).astype(np.uint8)
                        final_area_ratio = float(final_mask.sum()) / total_pixels
                        rescue_mode = "rembg_only_upper_guard"
                    elif selected_type == "bottom":
                        cut_y = y0 + int(h_mask * 0.28)
                        final_mask[:cut_y, :] = False
                        alpha = np.where(final_mask, alpha, 0).astype(np.uint8)
                        final_area_ratio = float(final_mask.sum()) / total_pixels
                        rescue_mode = "rembg_only_lower_guard"

        path_name = "rembg_isnet_premium"

    except Exception as e:
        logger.warning(f"rembg extraction failed, falling back to parser: {e}")
        try:
            parsing = engine.parser.parse(crop)
            garment_mask = np.isin(parsing, [5, 6, 7, 8, 9, 10, 11, 12])
            kill_mask = np.isin(parsing, [1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19])
            final_mask = garment_mask & (~kill_mask) & (parsing != 0)
            alpha = build_soft_alpha(final_mask, feather_px=max(1, ANALYZE_EXTRACT_EDGE_FEATHER_PX))
            path_name = "parser_fallback"
            rescue_mode = "parser_fallback"
        except Exception as e2:
            logger.error(f"Parser fallback also failed: {e2}")
            raise RuntimeError(f"All extraction methods failed: {e} / {e2}")

    final_area_ratio = float(final_mask.sum()) / total_pixels
    if final_area_ratio <= 0.0:
        raise RuntimeError(f"Extraction produced empty result for: {selected_type}")

    alpha_plane = alpha[:, :, None]
    cloth_rgba = np.concatenate([img_np, alpha_plane], axis=2)
    cloth_image = Image.fromarray(cloth_rgba, mode="RGBA")

    body_guard = {
        "enabled": bool(engine.parser is not None),
        "max_body_ratio": float(ANALYZE_EXTRACT_MAX_BODY_RATIO),
    }
    if engine.parser is not None:
        try:
            import cv2

            leak_before = _body_leakage_stats(cloth_image)
            body_guard["ratio_before"] = round(float(leak_before.get("ratio", 0.0)), 6)
            body_guard["visible_pixels"] = int(leak_before.get("visible_pixels", 0))
            body_guard["body_pixels_before"] = int(leak_before.get("body_pixels", 0))

            if float(leak_before.get("ratio", 0.0)) > ANALYZE_EXTRACT_MAX_BODY_RATIO:
                body_mask = leak_before.get("body_mask")
                if body_mask is not None:
                    body_mask_u8 = (body_mask.astype(np.uint8) * 255)
                    d = max(0, int(ANALYZE_EXTRACT_BODY_STRIP_DILATE))
                    if d > 0:
                        body_mask_u8 = cv2.dilate(body_mask_u8, np.ones((d, d), np.uint8), iterations=1)

                    rgba_arr = np.asarray(cloth_image.convert("RGBA")).copy()
                    alpha_arr = rgba_arr[:, :, 3].astype(np.uint8)
                    alpha_arr[body_mask_u8 > 0] = 0
                    cleaned_mask = alpha_arr > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD)
                    alpha_arr = build_soft_alpha(cleaned_mask, feather_px=ANALYZE_EXTRACT_EDGE_FEATHER_PX).astype(np.uint8)
                    rgba_arr[:, :, 3] = alpha_arr
                    cloth_image = Image.fromarray(rgba_arr, mode="RGBA")

                    leak_after = _body_leakage_stats(cloth_image)
                    body_guard["ratio_after"] = round(float(leak_after.get("ratio", 0.0)), 6)
                    body_guard["body_pixels_after"] = int(leak_after.get("body_pixels", 0))

                    if float(leak_after.get("ratio", 0.0)) > ANALYZE_EXTRACT_MAX_BODY_RATIO:
                        raise RuntimeError(
                            f"Extraction contains mannequin/body remnants (ratio={leak_after.get('ratio'):.6f})."
                        )
        except Exception as guard_err:
            if isinstance(guard_err, RuntimeError):
                raise
            logger.warning(f"Body leakage guard skipped due error: {guard_err}")

    alpha_after_guard = np.asarray(cloth_image.split()[-1]) > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD)
    final_area_ratio = float(np.sum(alpha_after_guard)) / total_pixels
    if final_area_ratio <= 0.0:
        raise RuntimeError(f"Extraction produced empty result after body cleanup for: {selected_type}")

    meta = {
        "path": path_name,
        "mask_area_ratio": round(final_area_ratio, 6),
        "rescue_mode": rescue_mode if "rescue_mode" in locals() else "none",
        "body_guard": body_guard,
    }
    return cloth_image, meta

def _run_vton_cloth_only_fallback(image_url: str, garment_type: str, vto_mode: bool = False) -> dict:
    t_all_start = time.time()
    t_vton_start = time.time()
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
        raise RuntimeError(f"VTON cloth-only fallback failed: {response.status_code} {response.text[:200]}")
    t_vton_s = time.time() - t_vton_start
            
    body = response.json()
    url = body.get("url") or body.get("azure_url")
    if not url:
        raise RuntimeError("VTON fallback response missing url.")
    public_url = str(url)
    mirrored_ok = False
    image_quality = {}
    source_similarity = {}
    showroom_similarity = {}
    source_bytes = b""
    showroom_bytes = b""
    raw_vton_bytes = b""
    processed_vton_bytes = b""
    postprocess_meta = {}

    t_fetch_start = time.time()
    # Parallelize source/raw fetches to reduce wall-clock IO time.
    with ThreadPoolExecutor(max_workers=3) as pool:
        source_fut = pool.submit(_fetch_image_bytes, image_url, 45)
        raw_fut = pool.submit(_fetch_image_bytes, public_url, 45)
        showroom_fut = None
        if ANALYZE_VTON_USE_SHOWROOM_PERSON:
            showroom_fut = pool.submit(_fetch_image_bytes, person_image_url, 45)
        source_bytes = source_fut.result()
        raw_vton_bytes = raw_fut.result()
        if showroom_fut is not None:
            showroom_bytes = showroom_fut.result()
    t_fetch_s = time.time() - t_fetch_start

    if ANALYZE_VTON_USE_SHOWROOM_PERSON and showroom_bytes:
        showroom_similarity = _compare_image_similarity(showroom_bytes, raw_vton_bytes)
        showroom_similarity["change_ratio"] = round(
            _pixel_change_ratio(showroom_bytes, raw_vton_bytes),
            6,
        )
        if showroom_similarity.get("exact_bytes"):
            raise RuntimeError("VTON returned showroom model image unchanged (exact match).")
        if (
            float(showroom_similarity.get("resized_mae", 999.0)) <= ANALYZE_VTON_MAX_SHOWROOM_MAE
            and float(showroom_similarity.get("change_ratio", 0.0)) < ANALYZE_VTON_MIN_SHOWROOM_CHANGE_RATIO
        ):
            raise RuntimeError(
                "VTON returned showroom-like output "
                f"(resized_mae={showroom_similarity.get('resized_mae')}, "
                f"change_ratio={showroom_similarity.get('change_ratio')})."
            )

    # --- LOCAL EXTRACTION FROM MANNEQUIN ---
    # We now have the mannequin wearing the garment (raw_vton_bytes).
    # We perform local extraction to isolate the garment and ensure no holes are introduced.
    t_extract_start = time.time()
    vton_img_mannequin = Image.open(io.BytesIO(raw_vton_bytes)).convert("RGB")
    logger.info(f"Loaded VTON mannequin image for local extraction: {vton_img_mannequin.size} {vton_img_mannequin.mode}")
    extracted_cloth, extraction_meta = _extract_cloth_from_crop(vton_img_mannequin, garment_type)
    extraction_meta["vto_path"] = "vton_fallback_mannequin_local_parser"

    # Save as bytes for postprocessing
    buf = io.BytesIO()
    extracted_cloth.save(buf, format="PNG")
    final_vton_bytes = buf.getvalue()
    t_extract_s = time.time() - t_extract_start

    source_similarity = _compare_image_similarity(source_bytes, final_vton_bytes)
    if ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH:
        if source_similarity.get("exact_bytes"):
            raise RuntimeError("VTON output is byte-identical to source crop (passthrough).")
        if float(source_similarity.get("resized_mae", 999.0)) <= ANALYZE_VTON_MAX_SOURCE_MAE:
            raise RuntimeError(
                f"VTON output too similar to source crop (resized_mae={source_similarity.get('resized_mae')})."
            )
    
    t_post_start = time.time()
    processed_vton_bytes, postprocess_meta = _postprocess_extracted_garment_bytes(final_vton_bytes)
    image_quality = _validate_vton_output_bytes(processed_vton_bytes)
    t_post_s = time.time() - t_post_start

    # Normalize to publicly consumable URLs
    mirrored_raw_url = str(url)
    t_upload_start = time.time()
    if ANALYZE_VTON_MIRROR_RAW_OUTPUT:
        try:
            # Optional debug mirror for raw VTON payload.
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
    if not mirrored_ok:
        raise RuntimeError(
            "Processed garment output could not be mirrored; refusing to return raw try-on/showroom output."
        )
    t_upload_s = time.time() - t_upload_start
    t_all_s = time.time() - t_all_start

    # Avoid exposing mannequin/showroom raw images to client response payloads.
    public_raw_url = ""
    if ANALYZE_VTON_MIRROR_RAW_OUTPUT and mirrored_raw_url:
        public_raw_url = str(mirrored_raw_url)

    return {
        "url": public_url,
        "raw_url": public_raw_url,
        "_processed_image_bytes": processed_vton_bytes,
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
            "showroom_similarity": showroom_similarity,
            "postprocess": postprocess_meta,
            "timings": {
                "total": round(t_all_s, 4),
                "vton_request": round(t_vton_s, 4),
                "fetch_io": round(t_fetch_s, 4),
                "local_extract": round(t_extract_s, 4),
                "postprocess": round(t_post_s, 4),
                "upload": round(t_upload_s, 4),
            },
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
    Digitizes a garment from an image and always returns multipart/form-data:
    - metadata (application/json)
    - optional binary parts (extracted_cloth, item_1, item_2, ...)
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
            result="REJECTED",
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
        if len(image_bytes) > ANALYZE_MAX_FILE_BYTES:
            payload = _build_error_payload(
                title="File Too Large",
                description="Please upload an image smaller than 3MB.",
                reason_codes=["FILE_TOO_LARGE"],
                status_code=400,
            )
            return _multipart_form_response(payload)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if ANALYZE_BLUR_CHECK_ENABLED:
            focus_score = _focus_score(img)
            if focus_score < ANALYZE_BLUR_MIN_FOCUS_SCORE:
                payload = _build_error_payload(
                    title="Image Too Blurry",
                    description="The garment is not clear enough. Try a brighter, sharper photo.",
                    reason_codes=["IMAGE_TOO_BLURRY"],
                    status_code=400,
                )
                return _multipart_form_response(payload)

        async with gpu_semaphore:
            # 2. Detect & Crop (YOLO)
            parser_split_used = False
            parser_candidates_count = 0
            parser_split_reject_reason = ""
            heuristic_split_used = False
            heuristic_candidates_count = 0
            instances = engine.yolo.detect_instances(img)
            if not instances:
                payload = _build_error_payload(
                    title="No Clothing Found",
                    description="No clothing item was detected in the uploaded image.",
                    reason_codes=["NO_CLOTHING"],
                    status_code=400,
                )
                return _multipart_form_response(payload)
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

            # 2b. Heuristic split when parser could not split:
            # - single full-body candidate
            # - multi-candidate same-type full-body detections (prevents top-only collapse)
            if ANALYZE_ENABLE_HEURISTIC_SPLIT and not parser_split_used and instances:
                try:
                    trigger_split = False
                    base_inst = instances[0]
                    if len(instances) == 1:
                        trigger_split = True
                    elif _should_force_fullbody_split(instances, img.height):
                        largest = _largest_instance(instances)
                        if largest is not None:
                            base_inst = largest
                            trigger_split = True

                    if trigger_split:
                        classify = engine.florence.classify_garment_type(base_inst["image"], hint_type=requested_type)
                        base_type = str(classify.get("type") or "").strip().lower()
                        allow_split = base_type != "dress" or requested_type in {"top", "bottom"}
                        if allow_split:
                            heuristics = _heuristic_split_candidates(img, base_inst)
                            heuristic_candidates_count = max(heuristic_candidates_count, len(heuristics))
                            if len(heuristics) >= 2:
                                heuristic_split_used = True
                                instances = heuristics
                except Exception as heur_err:
                    logger.warning(f"Heuristic split fallback failed: {heur_err}")

            # 2b.1 Tighten split boxes with parser masks for cleaner top/bottom crops.
            if len(instances) >= 1 and engine.parser is not None:
                try:
                    instances = _tighten_instances_with_parser_masks(img, instances)
                except Exception as tighten_err:
                    logger.warning(f"Parser-based crop tightening failed: {tighten_err}")

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

                # Drop obvious non-garment detections (for example furniture/background objects),
                # but keep plausible garment crops even when caption text is weak.
                non_garment_caption = _caption_non_garment_signal(prompt_desc)
                has_garment_caption = _caption_garment_signal(prompt_desc)
                if non_garment_caption and not has_garment_caption:
                    det_conf = float(inst.get("confidence", 0.0))
                    label_hint = (
                        _normalize_garment_type(str(inst.get("label") or ""))
                        or _normalize_garment_type(str(resolved_type or ""))
                    )
                    plausible_garment = (
                        label_hint in {"top", "bottom", "dress", "outer"}
                        and crop_area_ratio >= 0.045
                        and det_conf >= 0.35
                    )
                    if not plausible_garment:
                        logger.info(
                            "Suppressing non-garment candidate from caption: %s",
                            (prompt_desc or "")[:140],
                        )
                        continue

                # Category mapping from prompt/auto-type
                style_infer = autotype_meta.get("specific_clothing_style") if autotype_meta else None
                if not style_infer and prompt_desc:
                    style_infer = _infer_style_from_text(prompt_desc, garment_type=resolved_type)
                
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
                if inst.get("tighten_source"):
                    items[-1]["tighten_source"] = str(inst.get("tighten_source"))
                if inst.get("tighten_adjustment"):
                    items[-1]["tighten_adjustment"] = str(inst.get("tighten_adjustment"))

            if not items:
                payload = _build_error_payload(
                    title="No Clothing Found",
                    description="No clothing item was detected in the uploaded image.",
                    reason_codes=["NO_CLOTHING"],
                    status_code=400,
                )
                return _multipart_form_response(payload)

            deduped_items = _dedupe_items(items, iou_threshold=0.60)
            dedup_removed = len(items) - len(deduped_items)
            items = deduped_items

            # Collapse to single candidate when all remaining detections map to same type.
            unique_types = sorted({str(it.get("type")) for it in items if it.get("type")})
            collapsed_same_type = False
            if len(items) > 1 and len(unique_types) <= 1 and _should_collapse_same_type(items):
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
                # If type is explicitly requested, proceed with the top-ranked matching candidate.
                if len(matched) >= 1:
                    auto_selected_index = matched[0]

            # ── MULTI-ITEM SELECTION REQUIRED (400 with multipart) ──
            if ANALYZE_REQUIRE_SELECTION and len(items) > 1 and selected_index is None and auto_selected_index is None:
                total_s = round(time.time() - t0, 4)
                # Build item_breakdown for Flutter
                item_breakdown = []
                binary_parts = []
                split_selection_needed = raw_detected_count <= 1 and (parser_split_used or heuristic_split_used)
                for item in items:
                    pub = _to_public_item(item)
                    pub["selection_index"] = int(pub.get("garment_id", 0))
                    pub["rank"] = int(pub["selection_index"]) + 1
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
                    "result": "REJECTED",
                    "title": "Selection Required",
                    "description": "We found multiple garments. Please select one item to continue.",
                    "reason_codes": (
                        ["MULTI_ITEM_SELECTION_REQUIRED", "LOW_CONFIDENCE"]
                        if split_selection_needed
                        else ["MULTI_ITEM_SELECTION_REQUIRED"]
                    ),
                    "selection_required": True,
                    "total_garments_found": len(items),
                    "selection_hint": {
                        "expected_field": "type",
                        "alternate_field": "selected_index",
                        "allowed_types": ["top", "bottom", "dress", "outer"],
                    },
                    "item_breakdown": item_breakdown,
                    "multipart_data": _build_multipart_parts(items=item_breakdown),
                    "latencies": {"total": total_s},
                    "processing_time_ms": int(total_s * 1000),
                }
                payload = _build_success_payload(data=data, status_code=400, message="")
                return _multipart_form_response(payload, binary_parts=binary_parts)

            selected_item = None
            selected_index_internal = None
            if selected_index is not None:
                # Backward compatibility: accept 0-based index and 1-based rank.
                if 1 <= selected_index <= len(items):
                    selected_index_internal = selected_index - 1
                else:
                    selected_index_internal = selected_index
                if selected_index_internal < 0 or selected_index_internal >= len(items):
                    payload = _build_error_payload(
                        title="Invalid Selection",
                        description=f"selected_index must be 0..{max(0, len(items) - 1)} (legacy) or 1..{len(items)} (rank).",
                        reason_codes=["INVALID_SELECTION_TYPE"],
                        status_code=400,
                    )
                    return _multipart_form_response(payload)
                selected_item = items[selected_index_internal]
            elif auto_selected_index is not None:
                selected_item = items[auto_selected_index]
                selected_index = auto_selected_index
            elif items:
                selected_item = items[0]

            if selected_item and not requested_type:
                conf_obj = selected_item.get("confidence") if isinstance(selected_item.get("confidence"), dict) else {}
                best_conf = float(conf_obj.get("hybrid", conf_obj.get("yolo", 0.0)))
                if best_conf < ANALYZE_MIN_ACCEPT_CONFIDENCE:
                    payload = _build_error_payload(
                        title="Low Confidence Detection",
                        description="The garment is not clear enough. Try a centered and brighter photo.",
                        reason_codes=["LOW_CONFIDENCE"],
                        status_code=400,
                    )
                    return _multipart_form_response(payload)

            if selected_item and ANALYZE_EXTRACT_CLOTH:
                forced_type = requested_type if requested_type in {"top", "bottom", "dress", "outer"} else None
                selected_type = forced_type or _normalize_garment_type(str(selected_item.get("type")))
                if not selected_type:
                    payload = _build_error_payload(
                        title="Type Required",
                        description="Could not determine garment type. Please provide type as top, bottom, dress, or outer.",
                        reason_codes=["INVALID_SELECTION_TYPE"],
                        status_code=400,
                    )
                    return _multipart_form_response(payload)
                if forced_type and str(selected_item.get("type")) != forced_type:
                    selected_item["type_original"] = selected_item.get("type")
                    selected_item["type"] = forced_type
                    selected_item["type_source"] = "requested_type"

                if not ANALYZE_VTON_FALLBACK_ENABLED or not ANALYZE_VTON_CLOTH_ONLY_ENDPOINT:
                    payload = _build_error_payload(
                        title="Extraction Not Configured",
                        description="Garment extraction is not available. Please try again later.",
                        reason_codes=["EXTRACTION_FAILED"],
                        status_code=502,
                        result="REJECTED",
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
                extracted_image_bytes = b""
                try:
                    # Deep Analysis fix: If single item detected, use vto_mode=True to
                    # preserve person and prevent truncation during extraction.
                    is_single_item = len(items) <= 1
                    fallback = None
                    last_fallback_err = None
                    for attempt in range(2):
                        try:
                            fallback = _run_vton_cloth_only_fallback(
                                selected_crop_url,
                                selected_type,
                                vto_mode=is_single_item
                            )
                            break
                        except Exception as run_err:
                            last_fallback_err = run_err
                            logger.warning(
                                "VTON extract attempt %d failed (type=%s, bbox=%s): %s",
                                attempt + 1,
                                selected_type,
                                selected_item.get("bbox"),
                                run_err,
                            )
                            if attempt == 0:
                                time.sleep(0.35)
                                continue
                    if fallback is None and last_fallback_err is not None:
                        raise last_fallback_err
                    extracted_url = str(fallback.get("url") or "")
                    extraction_meta = dict(fallback.get("meta") or {})
                    extracted_image_bytes = bytes(fallback.get("_processed_image_bytes") or b"")
                except Exception as fallback_err:
                    logger.error(
                        "Extraction fallback failed after retries (type=%s, bbox=%s): %s",
                        selected_type,
                        selected_item.get("bbox"),
                        fallback_err,
                    )
                    payload = _build_error_payload(
                        title="Extraction Failed",
                        description="Garment extraction failed. Please retry with a clearer image.",
                        reason_codes=["EXTRACTION_FAILED"],
                        status_code=502,
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
                selected_item["_extracted_image_bytes"] = extracted_image_bytes
                selected_item["cloth_verified"] = selected_item["output_image_source"] == "vton_fallback"
                selected_item["cloth_verification_source"] = "vton_extraction"

                if ANALYZE_PROMPT_FROM_EXTRACTED:
                    try:
                        if extracted_image_bytes:
                            cloth_image = Image.open(io.BytesIO(extracted_image_bytes)).convert("RGBA")
                        else:
                            cloth_image = download_image(extracted_url)
                        caption_image = _flatten_rgba_on_white(cloth_image)
                        if ANALYZE_CAPTION_MODE == "detailed":
                            prompt_desc = engine.florence.describe_garment(caption_image)
                        else:
                            prompt_desc = engine.florence.describe_garment_short(caption_image)
                        prompt_desc = _sanitize_garment_description(prompt_desc)
                        selected_item["promptDescription"] = prompt_desc
                        selected_item["description"] = prompt_desc
                        extracted_style = _infer_style_from_text(prompt_desc, garment_type=selected_type)
                        extracted_category = _wardrobe_category_from_garment_type(selected_type, style=extracted_style)
                        selected_item["style"] = extracted_category["style"]
                        selected_item["category_key"] = extracted_category["category_key"]
                        selected_item["primary_category_key"] = extracted_category["primary_category_key"]
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
                prompt_for_product = _product_prompt_description(
                    str(selected_item.get("promptDescription") or selected_item.get("description") or ""),
                    garment_type=str(selected_item.get("type") or ""),
                    style=str(selected_item.get("style") or ""),
                    category_key=str(selected_item.get("category_key") or ""),
                )
                selected_item["promptDescription"] = prompt_for_product
                selected_item["description"] = prompt_for_product

                progress_id = str(uuid.uuid4())
                forced_type = requested_type if requested_type in {"top", "bottom", "dress", "outer"} else None
                selected_type = forced_type or _normalize_garment_type(str(selected_item.get("type"))) or "top"
                output_url = str(selected_item.get("url") or "")
                prompt_description = str(selected_item.get("promptDescription") or "")
                output_source = str(selected_item.get("output_image_source") or "")
                extraction_obj = selected_item.get("extraction") if isinstance(selected_item.get("extraction"), dict) else {}
                has_vton_pipeline = bool(extraction_obj.get("endpoint"))
                sync_style_guess = str(selected_item.get("style") or "")
                if not sync_style_guess:
                    sync_style_guess = str(
                        _infer_style_from_text(prompt_description, garment_type=selected_type) or ""
                    )
                sync_category = _wardrobe_category_from_garment_type(
                    selected_type,
                    style=sync_style_guess if sync_style_guess else None,
                )
                selected_item["primary_category_key"] = sync_category["primary_category_key"]
                selected_item["category_key"] = sync_category["category_key"]
                selected_item["style"] = sync_category["style"]
                progress_meta = {
                    "selected_type": selected_type,
                    "requested_type": requested_type,
                    "primary_category_key": sync_category["primary_category_key"],
                    "category_key": sync_category["category_key"],
                    "style": sync_category["style"],
                    "outputImageSource": output_source or "crop",
                    "prompt_source": "extracted_output" if ANALYZE_PROMPT_FROM_EXTRACTED else "detected_crop",
                    "extraction": selected_item.get("extraction"),
                    "cloth_verified": bool(selected_item.get("cloth_verified")),
                    "cloth_verification_source": selected_item.get("cloth_verification_source"),
                    "detection_source": selected_item.get("detection_source"),
                    "type_source": selected_item.get("type_source"),
                    "bbox": selected_item.get("bbox"),
                    "crop": selected_item.get("crop"),
                }
                if has_vton_pipeline and output_url:
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

        # ── SUCCESS RESPONSE (200 with multipart: metadata + extracted_cloth) ──
        public_item = _to_public_item(selected_item) if selected_item else None
        total_s = round(time.time() - t0, 4)

        # Resolve category mapping for the selected item
        final_type = str(public_item.get("type", "top")) if public_item else "top"
        final_style = str(public_item.get("style", "")) if public_item else ""
        category_meta = _wardrobe_category_from_garment_type(final_type, style=final_style if final_style else None)

        reason_codes = ["SINGLE_ITEM"]
        extraction_path = str(public_item.get("output_image_source", "")) if public_item else ""
        extraction_meta_public = (public_item or {}).get("extraction") or {}
        if extraction_meta_public.get("endpoint"):
            reason_codes.append("VTON_ONLY_PIPELINE")
            reason_codes.append("VTON_USED")
        if requested_type:
            reason_codes.append("TYPE_FORCED_VTON")

        output_image_url = public_item.get("output_image_url", public_item.get("url")) if public_item else None
        multipart_data = _build_multipart_parts(items=[], cloth_url=output_image_url)
        data = {
            "result": "ACCEPTED",
            "title": "Added To Wardrobe",
            "description": "Garment extracted successfully.",
            "reason_codes": reason_codes,
            "selection_required": False,
            "clothing_type": category_meta["style"],
            "category_key": category_meta["category_key"],
            "primary_category_key": category_meta["primary_category_key"],
            "style": category_meta["style"],
            "selected_type": final_type,
            "selected_item": public_item,
            "cloth_url": public_item.get("url") if public_item else None,
            "output_image_url": output_image_url,
            "extraction_path": extraction_path,
            "promptDescription": public_item.get("promptDescription") if public_item else None,
            "wardrobe_progress_id": public_item.get("wardrobe_progress_id") if public_item else None,
            "multipart_data": multipart_data,
            "total_garments_found": len(items),
            "latencies": {"total": total_s},
            "processing_time_ms": int(total_s * 1000),
        }

        # Legacy fields for backward compatibility
        data["imageUrl"] = data["cloth_url"]
        data["progressId"] = data["wardrobe_progress_id"]

        success_binary_parts = []
        extracted_bytes = bytes((selected_item or {}).get("_extracted_image_bytes") or b"")
        if not extracted_bytes and output_image_url:
            try:
                extracted_bytes = _fetch_image_bytes(str(output_image_url), timeout=45)
            except Exception as dl_err:
                logger.warning(f"Could not fetch output_image_url for multipart extracted_cloth part: {dl_err}")
        if extracted_bytes:
            success_binary_parts.append({
                "name": "extracted_cloth",
                "filename": "extracted_cloth.png",
                "content_type": "image/png",
                "bytes": extracted_bytes,
            })

        payload = _build_success_payload(data=data, status_code=200, message="")
        return _multipart_form_response(payload, binary_parts=success_binary_parts)

    except HTTPException as he:
        # Convert old-style HTTPException to Flutter contract envelope
        detail = he.detail if isinstance(he.detail, dict) else {"message": str(he.detail)}
        status_msg = str(detail.get("status", detail.get("message", "Request failed")))
        payload = _build_error_payload(
            title="Request Failed",
            description=str(detail.get("message", status_msg)),
            reason_codes=[str(detail.get("reason", "REQUEST_FAILED")).upper()],
            status_code=he.status_code,
            result="REJECTED",
        )
        return _multipart_form_response(payload)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        payload = _build_error_payload(
            title="Server Error",
            description="An unexpected error occurred. Please try again.",
            reason_codes=["SERVER_ERROR"],
            status_code=500,
            result="REJECTED",
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
            "analyze_vton_mirror_raw_output": ANALYZE_VTON_MIRROR_RAW_OUTPUT,
            "analyze_vton_use_showroom_person": ANALYZE_VTON_USE_SHOWROOM_PERSON,
            "analyze_vton_dress_use_full_image": ANALYZE_VTON_DRESS_USE_FULL_IMAGE,
            "analyze_vton_single_item_use_full_image": ANALYZE_VTON_SINGLE_ITEM_USE_FULL_IMAGE,
            "analyze_vton_crop_top_extra_ratio_bottom": ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM,
            "analyze_vton_crop_top_extra_ratio_bottom_multi": ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI,
            "analyze_tighten_bottom_top_max_overlap_px": ANALYZE_TIGHTEN_BOTTOM_TOP_MAX_OVERLAP_PX,
            "analyze_tighten_bottom_max_down_shift_ratio": ANALYZE_TIGHTEN_BOTTOM_MAX_DOWN_SHIFT_RATIO,
            "analyze_tighten_bottom_max_gap_from_top_px": ANALYZE_TIGHTEN_BOTTOM_MAX_GAP_FROM_TOP_PX,
            "analyze_force_fullbody_split_on_same_type": ANALYZE_FORCE_FULLBODY_SPLIT_ON_SAME_TYPE,
            "analyze_force_fullbody_split_min_height_ratio": ANALYZE_FORCE_FULLBODY_SPLIT_MIN_HEIGHT_RATIO,
            "analyze_collapse_same_type": ANALYZE_COLLAPSE_SAME_TYPE,
            "analyze_collapse_same_type_min_iou": ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU,
            "analyze_extract_max_body_ratio": ANALYZE_EXTRACT_MAX_BODY_RATIO,
            "analyze_extract_body_strip_dilate": ANALYZE_EXTRACT_BODY_STRIP_DILATE,
            "analyze_vton_max_showroom_mae": ANALYZE_VTON_MAX_SHOWROOM_MAE,
            "analyze_vton_min_showroom_change_ratio": ANALYZE_VTON_MIN_SHOWROOM_CHANGE_RATIO,
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
