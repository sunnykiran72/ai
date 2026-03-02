import io
import time
import logging
import asyncio
import os
import uuid
import tempfile
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple

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
from ai.core.qwen25vl_runner import Qwen25VLRunner
from ai.core.joycaption_runner import JoyCaptionRunner
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
FLUX2_DESCRIPTOR_BACKEND = os.getenv("FLUX2_DESCRIPTOR_BACKEND", "florence").strip().lower()
if FLUX2_DESCRIPTOR_BACKEND not in {"florence", "qwen2_5_vl", "joycaption"}:
    FLUX2_DESCRIPTOR_BACKEND = "florence"
FLUX2_FIDELITY_BACKEND = os.getenv("FLUX2_FIDELITY_BACKEND", "florence").strip().lower()
if FLUX2_FIDELITY_BACKEND not in {"florence", "qwen2_5_vl"}:
    FLUX2_FIDELITY_BACKEND = "florence"
FLUX2_DESCRIPTOR_COMPARE = os.getenv("FLUX2_DESCRIPTOR_COMPARE", "0") == "1"
FLUX2_DRESS_SECOND_PASS_ENABLED = os.getenv("FLUX2_DRESS_SECOND_PASS_ENABLED", "1") == "1"
FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS = max(1, _env_int("FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS", 4))
FLUX2_DRESS_SECOND_PASS_MAX_STEPS = max(6, _env_int("FLUX2_DRESS_SECOND_PASS_MAX_STEPS", 18))
FLUX2_QWEN_SECOND_PASS_ENABLED = os.getenv("FLUX2_QWEN_SECOND_PASS_ENABLED", "1") == "1"
FLUX2_QWEN_SECOND_PASS_EXTRA_STEPS = max(1, _env_int("FLUX2_QWEN_SECOND_PASS_EXTRA_STEPS", 6))
FLUX2_QWEN_SECOND_PASS_MAX_STEPS = max(8, _env_int("FLUX2_QWEN_SECOND_PASS_MAX_STEPS", 24))
FLUX2_QWEN_MIN_STEPS = max(4, _env_int("FLUX2_QWEN_MIN_STEPS", 10))
FLUX2_PRELOAD_QWEN_WITH_FLUX2 = os.getenv("FLUX2_PRELOAD_QWEN_WITH_FLUX2", "1") == "1"
FLUX2_QWEN_EXTRA_CLASSIFY_PASS = os.getenv("FLUX2_QWEN_EXTRA_CLASSIFY_PASS", "0") == "1"
FLUX2_UNLOAD_QWEN_BEFORE_FLUX2 = os.getenv("FLUX2_UNLOAD_QWEN_BEFORE_FLUX2", "0") == "1"
FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE = max(384, _env_int("FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE", 768))
FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE = max(256, _env_int("FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE", 512))
FLUX2_QWEN_USER_CAPTION_MAX_SIDE = max(384, _env_int("FLUX2_QWEN_USER_CAPTION_MAX_SIDE", 768))
FLUX2_QWEN_USER_CAPTION_MIN_SIDE = max(256, _env_int("FLUX2_QWEN_USER_CAPTION_MIN_SIDE", 512))
FLUX2_PRELOAD_JOYCAPTION_WITH_FLUX2 = os.getenv("FLUX2_PRELOAD_JOYCAPTION_WITH_FLUX2", "1") == "1"
FLUX2_COLOR_LOCK_ENABLED = os.getenv("FLUX2_COLOR_LOCK_ENABLED", "1") == "1"
FLUX2_COLOR_LOCK_TOP_K = min(5, max(1, _env_int("FLUX2_COLOR_LOCK_TOP_K", 3)))
FLUX2_DETAIL_LOCK_ENABLED = os.getenv("FLUX2_DETAIL_LOCK_ENABLED", "1") == "1"
FLUX2_SINGLE_CANDIDATE_MODE = os.getenv("FLUX2_SINGLE_CANDIDATE_MODE", "auto").strip().lower()
if FLUX2_SINGLE_CANDIDATE_MODE not in {"auto", "base", "dress_strict", "qwen_strict"}:
    FLUX2_SINGLE_CANDIDATE_MODE = "auto"
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
ANALYZE_USE_PARSER_FOR_PREROUTING = os.getenv("ANALYZE_USE_PARSER_FOR_PREROUTING", "0") == "1"
ANALYZE_ENABLE_HEURISTIC_SPLIT = os.getenv("ANALYZE_ENABLE_HEURISTIC_SPLIT", "1") == "1"
ANALYZE_TIGHTEN_SPLIT_CROPS = os.getenv("ANALYZE_TIGHTEN_SPLIT_CROPS", "1") == "1"
ANALYZE_TIGHTEN_SPLIT_PAD = max(0, _env_int("ANALYZE_TIGHTEN_SPLIT_PAD", 8))
ANALYZE_TIGHTEN_SPLIT_MIN_PIXELS = max(32, _env_int("ANALYZE_TIGHTEN_SPLIT_MIN_PIXELS", 48))
ANALYZE_TIGHTEN_BOTTOM_TOP_EXTRA_RATIO = _env_float("ANALYZE_TIGHTEN_BOTTOM_TOP_EXTRA_RATIO", 0.12)
ANALYZE_TIGHTEN_BOTTOM_BOTTOM_EXTRA_RATIO = _env_float("ANALYZE_TIGHTEN_BOTTOM_BOTTOM_EXTRA_RATIO", 0.05)
ANALYZE_TIGHTEN_BOTTOM_TOP_MAX_OVERLAP_PX = max(0, _env_int("ANALYZE_TIGHTEN_BOTTOM_TOP_MAX_OVERLAP_PX", 72))
ANALYZE_TIGHTEN_BOTTOM_MAX_DOWN_SHIFT_RATIO = max(0.0, _env_float("ANALYZE_TIGHTEN_BOTTOM_MAX_DOWN_SHIFT_RATIO", 0.08))
ANALYZE_TIGHTEN_BOTTOM_MAX_GAP_FROM_TOP_PX = max(0, _env_int("ANALYZE_TIGHTEN_BOTTOM_MAX_GAP_FROM_TOP_PX", 96))
ANALYZE_TIGHTEN_BOTTOM_MIN_WIDTH_RATIO = _env_float("ANALYZE_TIGHTEN_BOTTOM_MIN_WIDTH_RATIO", 0.92)
ANALYZE_HEURISTIC_MIN_HEIGHT_RATIO = _env_float("ANALYZE_HEURISTIC_MIN_HEIGHT_RATIO", 0.78)
ANALYZE_HEURISTIC_TOP_PORTION = _env_float("ANALYZE_HEURISTIC_TOP_PORTION", 0.52)
ANALYZE_HEURISTIC_TOP_TRIM_PX = max(0, _env_int("ANALYZE_HEURISTIC_TOP_TRIM_PX", 0))
ANALYZE_HEURISTIC_BOTTOM_OVERLAP_PX = max(0, _env_int("ANALYZE_HEURISTIC_BOTTOM_OVERLAP_PX", 32))
ANALYZE_HEURISTIC_BOTTOM_OVERLAP_RATIO = _env_float("ANALYZE_HEURISTIC_BOTTOM_OVERLAP_RATIO", 0.11)
ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO = _env_float("ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO", 0.44)
ANALYZE_HEURISTIC_MAX_WIDTH_RATIO = _env_float("ANALYZE_HEURISTIC_MAX_WIDTH_RATIO", 0.98)
ANALYZE_PARSER_TOP_DRESS_BACKFILL = os.getenv("ANALYZE_PARSER_TOP_DRESS_BACKFILL", "1") == "1"
ANALYZE_PARSER_TOP_MIN_RATIO = _env_float("ANALYZE_PARSER_TOP_MIN_RATIO", 0.008)
ANALYZE_PARSER_DRESS_BACKFILL_MIN_RATIO = _env_float("ANALYZE_PARSER_DRESS_BACKFILL_MIN_RATIO", 0.015)
ANALYZE_FORCE_FULLBODY_SPLIT_ON_SAME_TYPE = os.getenv("ANALYZE_FORCE_FULLBODY_SPLIT_ON_SAME_TYPE", "1") == "1"
ANALYZE_FORCE_FULLBODY_SPLIT_MIN_HEIGHT_RATIO = _env_float("ANALYZE_FORCE_FULLBODY_SPLIT_MIN_HEIGHT_RATIO", 0.72)
ANALYZE_FLORENCE_DRESS_LOCK_MIN_SCORE = _env_float("ANALYZE_FLORENCE_DRESS_LOCK_MIN_SCORE", 0.74)
ANALYZE_COLLAPSE_SAME_TYPE = os.getenv("ANALYZE_COLLAPSE_SAME_TYPE", "0") == "1"
ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU = _env_float("ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU", 0.85)
ANALYZE_AUTO_SELECT_MULTI_DRESS = os.getenv("ANALYZE_AUTO_SELECT_MULTI_DRESS", "1") == "1"
ANALYZE_EXTRACT_CLOTH = os.getenv("ANALYZE_EXTRACT_CLOTH", "1") == "1"
ANALYZE_PROMPT_FROM_EXTRACTED = os.getenv("ANALYZE_PROMPT_FROM_EXTRACTED", "1") == "1"
ANALYZE_EXTRACT_PARSER_ONLY = os.getenv("ANALYZE_EXTRACT_PARSER_ONLY", "1") == "1"
ANALYZE_EXTRACT_MIN_MASK_RATIO = _env_float("ANALYZE_EXTRACT_MIN_MASK_RATIO", 0.01)
ANALYZE_EXTRACT_RELAXED_RESCUE = os.getenv("ANALYZE_EXTRACT_RELAXED_RESCUE", "1") == "1"
ANALYZE_EXTRACT_EDGE_FEATHER_PX = max(0, _env_int("ANALYZE_EXTRACT_EDGE_FEATHER_PX", 1))
ANALYZE_EXTRACT_COMPONENT_MIN_RATIO = _env_float("ANALYZE_EXTRACT_COMPONENT_MIN_RATIO", 0.0007)
ANALYZE_EXTRACT_KEEP_DILATE = max(0, _env_int("ANALYZE_EXTRACT_KEEP_DILATE", 3))
ANALYZE_EXTRACT_PARSER_KILL_DILATE = max(1, _env_int("ANALYZE_EXTRACT_PARSER_KILL_DILATE", 2))
ANALYZE_GARMENT_HOLE_FILL_MAX_PIXELS = max(0, _env_int("ANALYZE_GARMENT_HOLE_FILL_MAX_PIXELS", 7000))
ANALYZE_EXTRACT_MAX_BODY_RATIO = max(0.0, _env_float("ANALYZE_EXTRACT_MAX_BODY_RATIO", 0.008))
ANALYZE_EXTRACT_BODY_STRIP_DILATE = max(0, _env_int("ANALYZE_EXTRACT_BODY_STRIP_DILATE", 3))
ANALYZE_EXTRACT_BODY_STRIP_MAX_RATIO = min(
    0.95,
    max(0.01, _env_float("ANALYZE_EXTRACT_BODY_STRIP_MAX_RATIO", 0.28)),
)
ANALYZE_SHOWROOM_DELTA_REFINEMENT_ENABLED = os.getenv("ANALYZE_SHOWROOM_DELTA_REFINEMENT_ENABLED", "1") == "1"
ANALYZE_SHOWROOM_DELTA_THRESHOLD = max(1.0, _env_float("ANALYZE_SHOWROOM_DELTA_THRESHOLD", 12.0))
ANALYZE_SHOWROOM_DELTA_DILATE = max(0, _env_int("ANALYZE_SHOWROOM_DELTA_DILATE", 2))
ANALYZE_SHOWROOM_DELTA_MIN_RETAIN_RATIO = min(
    0.99,
    max(0.10, _env_float("ANALYZE_SHOWROOM_DELTA_MIN_RETAIN_RATIO", 0.58)),
)
ANALYZE_SHOWROOM_DELTA_SKIN_EXTRA = max(0.0, _env_float("ANALYZE_SHOWROOM_DELTA_SKIN_EXTRA", 8.0))
ANALYZE_TOP_RECOVER_MAX_SKIN_RATIO = min(0.95, max(0.0, _env_float("ANALYZE_TOP_RECOVER_MAX_SKIN_RATIO", 0.30)))
ANALYZE_TOP_RECOVER_NEAR_DILATE = max(7, _env_int("ANALYZE_TOP_RECOVER_NEAR_DILATE", 31))
ANALYZE_TOP_SKIN_RIM_CLEANUP = os.getenv("ANALYZE_TOP_SKIN_RIM_CLEANUP", "1") == "1"
ANALYZE_TOP_SKIN_RIM_MAX_RATIO = min(0.30, max(0.0, _env_float("ANALYZE_TOP_SKIN_RIM_MAX_RATIO", 0.08)))
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
ANALYZE_VTON_SEED = max(0, _env_int("ANALYZE_VTON_SEED", 23))
ANALYZE_VTON_SEGMENTATION_FREE = os.getenv("ANALYZE_VTON_SEGMENTATION_FREE", "1") == "1"
ANALYZE_VTON_CUTOUT_FEATHER_PX = max(0, _env_int("ANALYZE_VTON_CUTOUT_FEATHER_PX", 2))
ANALYZE_VTON_ZOOM_PADDING_RATIO = _env_float("ANALYZE_VTON_ZOOM_PADDING_RATIO", 0.12)
ANALYZE_VTON_ALLOW_SOURCE_FALLBACK = os.getenv("ANALYZE_VTON_ALLOW_SOURCE_FALLBACK", "0") == "1"
ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH = os.getenv("ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH", "1") == "1"
ANALYZE_VTON_MIRROR_RAW_OUTPUT = os.getenv("ANALYZE_VTON_MIRROR_RAW_OUTPUT", "0") == "1"
ANALYZE_VTON_STRICT_SAFETY_CHECKS = os.getenv("ANALYZE_VTON_STRICT_SAFETY_CHECKS", "0") == "1"
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
ANALYZE_GARMENT_OUTPUT_BACKGROUND = os.getenv("ANALYZE_GARMENT_OUTPUT_BACKGROUND", "white").strip().lower()
if ANALYZE_GARMENT_OUTPUT_BACKGROUND not in {"transparent", "white"}:
    ANALYZE_GARMENT_OUTPUT_BACKGROUND = "white"
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
        self.qwen25vl = Qwen25VLRunner()
        self.joycaption = JoyCaptionRunner()
        self.flux2 = Flux2CVTONRunner()
        self.board_builder = BoardBuilder()
        
    def ensure_vto_ready(self):
        self.flux2.ensure_ready()
        if FLUX2_DESCRIPTOR_COMPARE:
            self.florence._ensure_loaded()
            if FLUX2_PRELOAD_QWEN_WITH_FLUX2:
                self.qwen25vl.ensure_ready()
            if FLUX2_PRELOAD_JOYCAPTION_WITH_FLUX2:
                self.joycaption.ensure_ready()
        elif FLUX2_DESCRIPTOR_BACKEND == "qwen2_5_vl":
            if FLUX2_PRELOAD_QWEN_WITH_FLUX2:
                self.qwen25vl.ensure_ready()
        elif FLUX2_DESCRIPTOR_BACKEND == "joycaption":
            if FLUX2_PRELOAD_JOYCAPTION_WITH_FLUX2:
                self.joycaption.ensure_ready()
        else:
            self.florence._ensure_loaded()

    def ensure_analyze_ready(self):
        self.yolo_runner.ensure_ready()
        if self.parser_runner:
            self.parser_runner.ensure_ready()

    def model_status(self):
        return {
            "flux2_loaded": self.flux2._pipeline is not None,
            "florence_loaded": self.florence._model is not None,
            "qwen25vl_loaded": self.qwen25vl.is_loaded,
            "joycaption_loaded": self.joycaption.is_loaded,
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

def _sanitize_florence_garment_description(description: str) -> str:
    """
    Keep garment attributes and strip scene/mannequin/background chatter.
    """
    text = str(description or "").strip()
    if not text:
        return ""

    text = re.sub(r"^\s*the image shows\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:a|the)?\s*mannequin\s+(?:is\s+)?wearing\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    fragments = [frag.strip(" .") for frag in re.split(r"[.]+", text) if frag.strip()]
    if not fragments:
        return text

    drop_markers = (
        "background",
        "white wall",
        "set against",
        "mannequin",
        "model is standing",
        "the image",
    )
    keep_markers = (
        "dress",
        "gown",
        "top",
        "shirt",
        "blouse",
        "corset",
        "skirt",
        "pants",
        "trousers",
        "jeans",
        "jacket",
        "coat",
        "fabric",
        "ruffle",
        "sleeve",
        "bodice",
        "silhouette",
        "color",
        "black",
        "white",
        "red",
        "blue",
        "green",
    )

    filtered: List[str] = []
    for frag in fragments:
        low = frag.lower()
        if any(marker in low for marker in drop_markers):
            continue
        filtered.append(frag)

    cleaned = ". ".join(filtered[:3]).strip(" .")
    cleaned = cleaned or text
    # Remove hedging words that weaken transfer constraints.
    cleaned = re.sub(r"\b(?:likely|possibly|probably|maybe)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:appears to be|seems to be|looks like)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthe garment is\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")
    return cleaned or text

def _augment_identity_lock(user_description: str) -> str:
    base = str(user_description or "").strip()
    lock = (
        "Keep exact same person identity: face structure, skin tone, hair, hands, "
        "body shape, pose, and scene lighting."
    )
    if not base:
        return lock
    if "skin tone" in base.lower() and "face" in base.lower():
        return base
    return f"{base} {lock}"

def _infer_flux2_target_type(description: str) -> str:
    text = str(description or "").lower()
    dress_terms = (
        "dress", "gown", "one-piece", "one piece", "maxi", "midi", "mini",
        "anarkali", "saree", "sari", "lehenga", "jumpsuit", "romper", "kurti",
    )
    bottom_terms = (
        "pant", "pants", "trouser", "trousers", "jean", "jeans", "skirt", "shorts",
        "palazzo", "chino", "legging", "leggings",
    )
    outer_terms = ("jacket", "coat", "blazer", "hoodie", "cardigan", "outerwear", "outer", "shrug")
    top_terms = ("shirt", "t-shirt", "tee", "top", "blouse", "corset", "sweater", "kurta", "tunic")

    if any(term in text for term in dress_terms):
        return "dress"
    if any(term in text for term in bottom_terms):
        return "bottom"
    if any(term in text for term in outer_terms):
        return "outer"
    if any(term in text for term in top_terms):
        return "top"
    return "top"

def _ensure_target_type_in_description(description: str, target_type: str) -> str:
    text = str(description or "").strip()
    kind = str(target_type or "").strip().lower()
    if kind not in {"dress", "top", "bottom", "outer"}:
        return text

    alias_map = {
        "dress": ("dress", "gown", "one-piece", "maxi", "midi", "mini", "anarkali", "saree", "sari", "lehenga"),
        "top": ("top", "shirt", "blouse", "tee", "t-shirt", "kurta", "tunic", "corset", "sweater"),
        "bottom": ("bottom", "pants", "trousers", "jeans", "skirt", "shorts", "palazzo", "leggings"),
        "outer": ("outer", "outerwear", "jacket", "coat", "blazer", "hoodie", "cardigan"),
    }
    low = text.lower()
    if any(alias in low for alias in alias_map[kind]):
        return text

    if not text:
        return f"{kind} garment"
    return f"{kind} garment, {text}"

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
    ("pink", (214, 120, 165)),
]

_DETAIL_LOCK_TERMS: List[Tuple[str, str]] = [
    ("sequined", "sequined surface"),
    ("sequin", "sequined surface"),
    ("beaded", "beadwork"),
    ("beading", "beadwork"),
    ("embroidery", "embroidery motifs"),
    ("lace", "lace texture"),
    ("feather", "feather trims"),
    ("ruffle", "ruffle placement"),
    ("pleat", "pleated structure"),
    ("satin", "satin finish"),
    ("silk", "silk-like drape"),
    ("velvet", "velvet texture"),
    ("sheer", "sheer panels"),
    ("mesh", "mesh panels"),
    ("corset", "corset bodice"),
    ("strapless", "strapless neckline"),
    ("off-shoulder", "off-shoulder cut"),
    ("v-neck", "V-neckline"),
    ("deep v", "deep V-neckline"),
    ("sleeveless", "sleeveless cut"),
    ("long sleeve", "long sleeves"),
    ("train", "long train"),
    ("slit", "slit placement"),
    ("mermaid", "mermaid flare profile"),
    ("high-low", "high-low hem shape"),
    ("high low", "high-low hem shape"),
    ("asym", "asymmetric hem contour"),
    ("tier", "tiered skirt layers"),
    ("ruffled hem", "ruffled hemline"),
]

_TEXT_COLOR_TERMS: List[str] = [
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
    "lavender",
    "pink",
    "peach",
]

def _nearest_color_label(rgb_triplet: Tuple[int, int, int]) -> str:
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

def _extract_dominant_color_labels(image: Image.Image, top_k: int = 3) -> List[str]:
    """
    Fast palette hinting for stronger color fidelity in Flux2 prompting.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w < 4 or h < 4:
        return []

    # Focus center region to reduce white/neutral studio background dominance.
    x_pad = int(w * 0.1)
    y_pad = int(h * 0.1)
    crop = rgb.crop((x_pad, y_pad, max(x_pad + 1, w - x_pad), max(y_pad + 1, h - y_pad)))
    crop.thumbnail((192, 192), Image.Resampling.BICUBIC)
    arr = np.array(crop, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return []

    pixels = arr.reshape(-1, 3)
    if pixels.size == 0:
        return []

    near_white = np.all(pixels >= 245, axis=1)
    non_white_ratio = float(np.mean(~near_white)) if pixels.shape[0] else 0.0
    if non_white_ratio > 0.20:
        pixels = pixels[~near_white]
    if pixels.size == 0:
        return []

    # Bin colors to make dominant counts stable and cheap.
    binned = (pixels // 16) * 16
    unique, counts = np.unique(binned, axis=0, return_counts=True)
    order = np.argsort(-counts)
    label_counts = {}
    for idx in order.tolist():
        rgb_triplet = tuple(int(v) for v in unique[idx].tolist())
        label = _nearest_color_label(rgb_triplet)
        label_counts[label] = label_counts.get(label, 0) + int(counts[idx])

    ordered_labels = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ordered_labels[: max(1, top_k)]]

def _extract_detail_lock_terms(description: str, max_items: int = 6) -> List[str]:
    low = str(description or "").lower()
    if not low:
        return []
    hits: List[str] = []
    for needle, phrase in _DETAIL_LOCK_TERMS:
        if needle in low and phrase not in hits:
            hits.append(phrase)
            if len(hits) >= max_items:
                break
    return hits

def _has_transparency_signal(text: str) -> bool:
    low = str(text or "").lower()
    if not low:
        return False
    signals = (
        "sheer",
        "mesh",
        "transparent",
        "see-through",
        "see through",
        "translucent",
    )
    return any(sig in low for sig in signals)

def _extract_text_color_terms(description: str, max_items: int = 4) -> List[str]:
    low = str(description or "").lower()
    if not low:
        return []
    hits: List[str] = []
    for term in _TEXT_COLOR_TERMS:
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, low) and term not in hits:
            normalized = "gray" if term == "grey" else term
            hits.append(normalized)
            if len(hits) >= max_items:
                break
    return hits

def _build_flux2_visual_lock_clauses(
    product_images: List[Image.Image],
    garment_descriptions: List[str],
) -> dict:
    color_hints: List[List[str]] = []
    if FLUX2_COLOR_LOCK_ENABLED:
        for idx, img in enumerate(product_images):
            desc = garment_descriptions[idx] if idx < len(garment_descriptions) else ""
            text_colors = _extract_text_color_terms(desc, max_items=FLUX2_COLOR_LOCK_TOP_K)
            if text_colors:
                color_hints.append(text_colors)
                continue
            color_hints.append(_extract_dominant_color_labels(img, top_k=FLUX2_COLOR_LOCK_TOP_K))

    detail_terms: List[str] = []
    if FLUX2_DETAIL_LOCK_ENABLED:
        for desc in garment_descriptions:
            for term in _extract_detail_lock_terms(desc):
                if term not in detail_terms:
                    detail_terms.append(term)

    color_clause = ""
    non_empty_color_hints = [h for h in color_hints if h]
    if non_empty_color_hints:
        if len(non_empty_color_hints) == 1:
            palette = ", ".join(non_empty_color_hints[0])
            color_clause = (
                f"Color lock from image 2: keep the garment in {palette} tones only. "
                "Do not recolor, hue-shift, or replace with a different color family. "
            )
        else:
            per_item = []
            for idx, colors in enumerate(color_hints, start=1):
                if colors:
                    per_item.append(f"item {idx}: {', '.join(colors)}")
            if per_item:
                color_clause = (
                    "Color lock from image 2 for each product: "
                    + "; ".join(per_item)
                    + ". Do not recolor any item. "
                )

    detail_clause = ""
    if detail_terms:
        detail_clause = (
            "Detail lock from image 2: preserve "
            + ", ".join(detail_terms[:6])
            + ". Do not simplify or replace these details. "
        )
    transparency_lock = any(_has_transparency_signal(desc) for desc in garment_descriptions)
    transparency_clause = ""
    if transparency_lock:
        transparency_clause = (
            "Transparency lock from image 2: preserve sheer/mesh panel transparency and translucency. "
            "Keep subtle skin visibility only where the source garment is transparent. "
            "Do not make sheer panels opaque or overexpose skin beyond those panel regions. "
        )

    return {
        "color_clause": color_clause,
        "detail_clause": detail_clause,
        "transparency_clause": transparency_clause,
        "transparency_lock": transparency_lock,
        "color_hints": color_hints,
        "detail_terms": detail_terms[:6],
    }

def _build_flux2_targeted_prompt(
    garment_descriptions: List[str],
    user_description: str,
    target_types: List[str],
    board_mode: str,
    color_lock_clause: str = "",
    detail_lock_clause: str = "",
    transparency_lock_clause: str = "",
) -> str:
    target_hint = ", ".join(garment_descriptions)
    types = {t for t in target_types if t}
    is_dress_mode = ("dress" in types) and len(types) == 1 and len(garment_descriptions) == 1
    is_multi = board_mode == "collage"
    identity_context = _identity_only_user_context(user_description)
    if not identity_context or re.search(r"\bis\s*\.\s*$", identity_context, flags=re.IGNORECASE):
        identity_context = "person in image 1"

    prompt = (
        "Photorealistic high-fidelity virtual try-on, premium fashion photography, "
        "sharp details, clean background, 8k resolution. "
        "Use image 1 as strict identity source for face, body shape, skin tone, and pose. "
        f"TRANSFER the {target_hint} from image 2 onto the person in image 1. "
        "Match exact garment attributes from image 2: color tone, print/pattern, neckline, sleeve length, hem length, fit, and trims. "
        "Do not invent a new garment design, new pattern, or new fabric. "
    )
    if color_lock_clause:
        prompt += color_lock_clause
    if detail_lock_clause:
        prompt += detail_lock_clause
    if transparency_lock_clause:
        prompt += transparency_lock_clause

    if is_dress_mode:
        prompt += (
            "Treat this as full-body outfit replacement with a single dress. "
            "Replace the original top and bottom garments with the target dress silhouette. "
            "Do not keep, overlay, or blend previous tops, pants, skirts, or shorts under/over the dress. "
            "Completely occlude old clothing in the replaced region. "
            "For lower-body fidelity, match exact skirt architecture from image 2: flare start point, "
            "ruffle/tier distribution, hem contour (including asymmetry/high-low), slit position, and train length. "
            "No layering artifacts. Ignore existing clothing design details in image 1 while preserving the person identity. "
        )
    else:
        if "top" in types:
            prompt += "Replace only the upper-body garment region with the target top. "
        if "bottom" in types:
            prompt += "Replace only the lower-body garment region with the target bottom. "
        if "outer" in types:
            prompt += "Replace/add the outerwear layer according to the target item. "
        prompt += (
            "Preserve untargeted garments from image 1 unless a target item explicitly replaces that region. "
        )

    if is_multi:
        prompt += (
            "Image 2 is a multi-item outfit board; apply all listed items together with coherent layering and fit. "
        )

    if is_dress_mode:
        prompt += (
            f"Identity reference from image 1: {identity_context}. "
            "Keep exact same person identity: facial features, skin tone, hair, hands, body proportions, "
            "and scene lighting. "
            "Ensure realistic fabric drape, seams, folds, and shadows."
        )
    else:
        prompt += (
            f"Person and current outfit reference from image 1: {user_description}. "
            "Keep exact same person identity: facial features, skin tone, hair, hands, body proportions, "
            "and scene lighting. "
            "Ensure realistic fabric drape, seams, folds, and shadows."
        )
    return prompt

def _normalize_descriptor_backend(raw: Optional[str]) -> str:
    value = str(raw or FLUX2_DESCRIPTOR_BACKEND).strip().lower()
    return value if value in {"florence", "qwen2_5_vl", "joycaption"} else "florence"

def _resize_for_qwen_caption(image: Image.Image, max_side: int, min_side: int) -> Image.Image:
    """
    Reduce visual tokens for Qwen while preserving garment/person detail.
    Never upscales small images.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= 0 or h <= 0:
        return rgb

    longest = max(w, h)
    shortest = min(w, h)
    scale = min(1.0, float(max_side) / float(longest))

    # Prevent over-downscaling thin details.
    if shortest > min_side and (shortest * scale) < min_side:
        scale = min(1.0, float(min_side) / float(shortest))

    if scale >= 0.999:
        return rgb

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return rgb.resize((new_w, new_h), Image.BICUBIC)

def _describe_garment_with_backend(image: Image.Image, backend: str) -> str:
    resolved = _normalize_descriptor_backend(backend)
    if resolved == "joycaption":
        try:
            return str(engine.joycaption.describe_garment(image)).strip()
        except Exception as err:
            logger.warning(f"JoyCaption garment description failed; fallback to Florence. error={err}")
    if resolved == "qwen2_5_vl":
        try:
            qwen_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE,
            )
            return str(engine.qwen25vl.describe_garment(qwen_img)).strip()
        except Exception as err:
            logger.warning(f"Qwen2.5-VL garment description failed; fallback to Florence. error={err}")
    return str(engine.florence.describe_garment(image)).strip()

def _describe_user_image_for_flux2(user_img: Image.Image, backend: str = "florence") -> str:
    """
    Ask selected descriptor model for a full-person detailed description.
    """
    resolved = _normalize_descriptor_backend(backend)
    if resolved == "joycaption":
        # JoyCaption in this pipeline is garment-focused; keep user lock generic
        # unless user promptDescription is explicitly provided in request.
        return ""
    if resolved == "qwen2_5_vl":
        try:
            qwen_img = _resize_for_qwen_caption(
                image=user_img,
                max_side=FLUX2_QWEN_USER_CAPTION_MAX_SIDE,
                min_side=FLUX2_QWEN_USER_CAPTION_MIN_SIDE,
            )
            return str(engine.qwen25vl.describe_person_and_outfit(qwen_img)).strip()
        except Exception as err:
            logger.warning(f"Qwen2.5-VL user description failed; fallback to Florence. error={err}")

    try:
        return str(
            engine.florence.run_task(
                image=user_img,
                task_prompt="<DETAILED_CAPTION>",
                text_input=" Describe the person's identity cues and all worn garments with colors and fit.",
                max_new_tokens=engine.florence.detailed_max_tokens,
                num_beams=engine.florence.detailed_num_beams,
                use_cache_generate=False,
            )
        ).strip()
    except Exception as err:
        logger.warning(f"Detailed user-image description fallback to Florence describe_garment due error: {err}")
        return str(engine.florence.describe_garment(user_img)).strip()

def _identity_only_user_context(user_description: str) -> str:
    """
    Remove clothing-specific mentions from user description to avoid conflicting replacement cues.
    """
    text = str(user_description or "").strip()
    if not text:
        return ""

    apparel_terms = (
        "wearing", "outfit", "dress", "gown", "top", "shirt", "blouse", "jacket", "coat",
        "pants", "trousers", "jeans", "skirt", "shorts", "sneaker", "shoe", "sleeve", "bodice",
    )
    identity_terms = (
        "face", "facial", "hair", "skin", "complexion", "body", "build", "shape", "pose",
        "hand", "arm", "leg", "height", "age", "eyes", "nose", "mouth", "jaw", "lighting",
        "background", "scene",
    )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    kept: List[str] = []
    for sentence in sentences:
        low = sentence.lower()
        has_apparel = any(term in low for term in apparel_terms)
        has_identity = any(term in low for term in identity_terms)
        if has_identity and not has_apparel:
            kept.append(sentence.strip(" ."))

    if kept:
        cleaned = ". ".join(kept).strip(" .")
        return cleaned

    # Fallback: aggressively strip outfit clauses.
    fallback = re.sub(r"\b(?:wearing|wears|dressed in)\b[^.]*\.?", "", text, flags=re.IGNORECASE)
    fallback = re.sub(
        r"\b(?:dress|gown|top|shirt|blouse|jacket|coat|pants|trousers|jeans|skirt|shorts|sneakers?|shoes?)\b[^.]*\.?",
        "",
        fallback,
        flags=re.IGNORECASE,
    )
    fallback = re.sub(r"\s+", " ", fallback).strip(" ,.")
    if re.search(r"\bis\s*\.\s*$", fallback, flags=re.IGNORECASE) or len(fallback) < 20:
        return "person in image 1"
    return fallback

_FIDELITY_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "of", "in", "on", "to", "for", "is", "are", "from", "image", "person",
    "wearing", "shows", "showing", "standing", "front", "background", "white", "black", "color",
}

def _extract_fidelity_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\\-]+", str(text or "").lower())
    return {w for w in words if len(w) >= 4 and w not in _FIDELITY_STOPWORDS}

def _score_tryon_garment_fidelity(
    output_image: Image.Image,
    target_description: str,
    descriptor_backend: str,
) -> tuple[float, str]:
    """
    Lightweight text-overlap score between target garment description and output garment description.
    Higher is better.
    """
    try:
        output_desc = _sanitize_florence_garment_description(
            _describe_garment_with_backend(output_image, descriptor_backend)
        )
    except Exception:
        output_desc = ""

    target_terms = _extract_fidelity_terms(target_description)
    output_terms = _extract_fidelity_terms(output_desc)
    if not target_terms:
        return (0.0, output_desc)

    overlap = len(target_terms.intersection(output_terms)) / float(len(target_terms))

    # Penalize obvious layering mismatches for dress targets.
    output_low = output_desc.lower()
    if "dress" in target_description.lower() or "gown" in target_description.lower():
        if any(token in output_low for token in ("pants", "trousers", "jeans", "shorts", "skirt and top")):
            overlap *= 0.75

    return (float(overlap), output_desc)

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

    def _mask_shape_ok(mask_obj: Optional[np.ndarray]) -> bool:
        return bool(
            mask_obj is not None
            and isinstance(mask_obj, np.ndarray)
            and mask_obj.shape[:2] == (image.height, image.width)
        )

    def _fallback_bottom_mask_from_dress(x0: int, y0: int, x1: int, y1: int) -> Optional[np.ndarray]:
        """
        If parser misses the explicit bottom class (common on one-piece/dress-like silhouettes),
        derive a bottom-support mask from dress mask while removing top region.
        """
        dress_mask = category_masks.get("dress")
        if not _mask_shape_ok(dress_mask):
            return None

        fallback = dress_mask.astype(bool).copy()
        if fallback.size == 0 or int(fallback.sum()) <= 0:
            return None

        top_like = np.zeros_like(fallback, dtype=bool)
        for key in ("top", "outer"):
            mk = category_masks.get(key)
            if _mask_shape_ok(mk):
                top_like |= mk.astype(bool)

        # Remove parser top/outer region plus a small buffer so bottom starts closer to waist.
        local_top = top_like[y0:y1, x0:x1]
        if local_top.size > 0 and int(local_top.sum()) > 0:
            ys, _ = np.where(local_top)
            if len(ys):
                top_end = y0 + int(ys.max())
                cut_y = min(image.height - 1, top_end + max(4, ANALYZE_TIGHTEN_SPLIT_PAD // 2))
                fallback[:cut_y, :] = False

        # Keep only the lower section relative to current split box to avoid pulling torso remnants.
        local_fallback = fallback[y0:y1, x0:x1]
        if local_fallback.size == 0 or int(local_fallback.sum()) <= 0:
            return None
        return fallback

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
        used_bottom_fallback_mask = False

        if inst_type == "bottom":
            primary_ok = _mask_shape_ok(mask)
            primary_region_has_pixels = False
            if primary_ok:
                primary_region = mask[y0:y1, x0:x1]
                primary_region_has_pixels = bool(primary_region.size > 0 and int(primary_region.sum()) > 0)
            if (not primary_ok) or (not primary_region_has_pixels):
                fb = _fallback_bottom_mask_from_dress(x0, y0, x1, y1)
                if fb is not None:
                    mask = fb
                    used_bottom_fallback_mask = True

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
            if used_bottom_fallback_mask:
                bottom_adjustments.append("bottom_mask_fallback_dress")

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

            # Preserve horizontal coverage for bottoms. Parser fallback can under-segment
            # draped/ruffled side fabric and clip one side too aggressively.
            orig_w = max(1, x1 - x0)
            min_width_ratio = max(0.60, min(1.0, ANALYZE_TIGHTEN_BOTTOM_MIN_WIDTH_RATIO))
            min_w = max(1, int(orig_w * min_width_ratio))
            cur_w = max(1, tx1 - tx0)
            if used_bottom_fallback_mask:
                tx0, tx1 = x0, x1
                bottom_adjustments.append("bottom_width_preserved")
            elif cur_w < min_w:
                cx = (tx0 + tx1) // 2
                half = max(1, min_w // 2)
                nx0 = max(x0, cx - half)
                nx1 = min(x1, nx0 + min_w)
                if (nx1 - nx0) < min_w:
                    nx1 = x1
                    nx0 = max(x0, nx1 - min_w)
                tx0, tx1 = nx0, nx1
                bottom_adjustments.append("bottom_width_recovered")

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

def _parser_runtime_id2label() -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        if engine.parser_runner and isinstance(engine.parser_runner.id2label, dict):
            for k, v in engine.parser_runner.id2label.items():
                try:
                    out[int(k)] = str(v).strip().lower().replace(" ", "_")
                except Exception:
                    continue
    except Exception:
        pass
    try:
        runtime_fn = getattr(engine.parser, "_runtime_labels", None)
        runtime = runtime_fn() if callable(runtime_fn) else {}
        if isinstance(runtime, dict):
            for name, idx in runtime.items():
                try:
                    out[int(idx)] = str(name).strip().lower().replace(" ", "_")
                except Exception:
                    continue
    except Exception:
        pass
    return out

def _square_bbox_from_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    safe_pad = max(0.0, min(0.8, float(padding_ratio)))
    side = int(round(max(bw, bh) * (1.0 + (2.0 * safe_pad))))
    side = max(2, side)

    cx = float(x0 + x1) / 2.0
    cy = float(y0 + y1) / 2.0
    sx0 = int(round(cx - (side / 2.0)))
    sy0 = int(round(cy - (side / 2.0)))
    return [sx0, sy0, sx0 + side, sy0 + side]

def _crop_square_with_padding(
    image: Image.Image,
    square_bbox: list[int],
    fill_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    x0, y0, x1, y1 = [int(v) for v in square_bbox]
    side = max(2, int(x1 - x0), int(y1 - y0))
    canvas = Image.new("RGB", (side, side), fill_rgb)

    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(image.width, x1)
    iy1 = min(image.height, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return canvas

    src_crop = image.crop((ix0, iy0, ix1, iy1)).convert("RGB")
    paste_x = int(ix0 - x0)
    paste_y = int(iy0 - y0)
    canvas.paste(src_crop, (paste_x, paste_y))
    return canvas

def _expand_section_bbox_by_type(
    bbox: list[int],
    garment_type: str,
    image_width: int,
    image_height: int,
) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    gt = _normalize_garment_type(garment_type) or "top"
    if gt == "top":
        x_pad = int(bw * 0.03)
        y_pad_top = int(bh * 0.02)
        y_pad_bottom = int(bh * 0.06)
    elif gt == "bottom":
        x_pad = int(bw * 0.03)
        y_pad_top = int(bh * 0.06)
        y_pad_bottom = int(bh * 0.04)
    elif gt == "dress":
        x_pad = int(bw * 0.03)
        y_pad_top = int(bh * 0.02)
        y_pad_bottom = int(bh * 0.04)
    else:
        x_pad = int(bw * 0.03)
        y_pad_top = int(bh * 0.02)
        y_pad_bottom = int(bh * 0.04)

    ex0 = max(0, x0 - x_pad)
    ey0 = max(0, y0 - y_pad_top)
    ex1 = min(image_width, x1 + x_pad)
    ey1 = min(image_height, y1 + y_pad_bottom)
    return [ex0, ey0, ex1, ey1]

def _mask_connected_components(mask: np.ndarray, min_pixels: int) -> list[dict]:
    clean = np.asarray(mask).astype(bool)
    if clean.size == 0:
        return []
    if int(clean.sum()) < max(1, int(min_pixels)):
        return []

    try:
        import cv2
    except Exception:
        cv2 = None

    out: list[dict] = []
    if cv2 is None:
        bbox = _bbox_from_mask(clean)
        if bbox is None:
            return []
        out.append({"bbox": bbox, "area": int(clean.sum()), "mask": clean})
        return out

    num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats((clean.astype(np.uint8) * 255), connectivity=8)
    for i in range(1, int(num_l)):
        area = int(stats_l[i, cv2.CC_STAT_AREA])
        if area < max(1, int(min_pixels)):
            continue
        x = int(stats_l[i, cv2.CC_STAT_LEFT])
        y = int(stats_l[i, cv2.CC_STAT_TOP])
        w = int(stats_l[i, cv2.CC_STAT_WIDTH])
        h = int(stats_l[i, cv2.CC_STAT_HEIGHT])
        if w < 8 or h < 8:
            continue
        out.append(
            {
                "bbox": [x, y, x + w, y + h],
                "area": area,
                "mask": labels_l == i,
            }
        )
    out.sort(key=lambda item: int(item.get("area", 0)), reverse=True)
    return out

def _bbox_y_overlap_ratio(a: list[int], b: list[int]) -> float:
    ay0, ay1 = int(a[1]), int(a[3])
    by0, by1 = int(b[1]), int(b[3])
    inter = max(0, min(ay1, by1) - max(ay0, by0))
    ha = max(1, ay1 - ay0)
    hb = max(1, by1 - by0)
    return float(inter) / float(max(1, min(ha, hb)))

def _bbox_x_gap(a: list[int], b: list[int]) -> int:
    ax0, ax1 = int(a[0]), int(a[2])
    bx0, bx1 = int(b[0]), int(b[2])
    if ax1 >= bx0 and bx1 >= ax0:
        return 0
    if ax1 < bx0:
        return int(bx0 - ax1)
    return int(ax0 - bx1)

def _bbox_x_overlap_ratio(a: list[int], b: list[int]) -> float:
    ax0, ax1 = int(a[0]), int(a[2])
    bx0, bx1 = int(b[0]), int(b[2])
    inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    wa = max(1, ax1 - ax0)
    wb = max(1, bx1 - bx0)
    return float(inter) / float(max(1, min(wa, wb)))

def _filter_parser_candidates_by_consistency(
    candidates: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Remove parser false-positives using garment-structure priors.
    This is intentionally applied only to parser-joycaption candidate discovery.
    """
    if not candidates:
        return candidates

    def _area_ratio(item: dict) -> float:
        return float(item.get("parser_component_area_ratio", 0.0) or 0.0)

    def _bbox(item: dict) -> list[int]:
        b = item.get("bbox") or [0, 0, image_width, image_height]
        return [int(v) for v in b]

    all_x0 = [int(_bbox(c)[0]) for c in candidates]
    all_x1 = [int(_bbox(c)[2]) for c in candidates]
    person_x0 = min(all_x0) if all_x0 else 0
    person_x1 = max(all_x1) if all_x1 else image_width
    person_w = max(1, person_x1 - person_x0)
    person_cx = int((person_x0 + person_x1) * 0.5)

    top_items = [c for c in candidates if str(c.get("type")) == "top"]
    bottom_items = [c for c in candidates if str(c.get("type")) == "bottom"]
    best_top = max(top_items, key=_area_ratio) if top_items else None
    best_bottom = max(bottom_items, key=_area_ratio) if bottom_items else None

    filtered: list[dict] = []
    for item in candidates:
        t = str(item.get("type") or "")
        b = _bbox(item)
        x0, y0, x1, y1 = [int(v) for v in b]
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        cy = 0.5 * (y0 + y1)
        area_ratio = _area_ratio(item)

        keep = True
        if t == "bottom":
            # Prevent upper-body fragments being returned as bottoms.
            if cy < (image_height * 0.44) and area_ratio < 0.10:
                keep = False

        elif t == "dress":
            # Generic dress plausibility.
            if bh < int(image_height * 0.36):
                keep = False
            if bw < int(max(32, person_w * 0.28)):
                keep = False

            # If strong top+bottom exist, require dress to actually bridge them and
            # cover center torso width; this suppresses sleeve-only dress false positives.
            if keep and best_top is not None and best_bottom is not None:
                tb = _bbox(best_top)
                bb = _bbox(best_bottom)
                max_tb_area = max(_area_ratio(best_top), _area_ratio(best_bottom))
                waist_mid_y = int((int(tb[3]) + int(bb[1])) * 0.5)
                spans_waist = y0 <= waist_mid_y <= y1
                centered_cover = (
                    x0 <= int(person_cx - person_w * 0.12)
                    and x1 >= int(person_cx + person_w * 0.12)
                )
                x_overlap_top = _bbox_x_overlap_ratio(b, tb)
                x_overlap_bottom = _bbox_x_overlap_ratio(b, bb)
                if not (
                    spans_waist
                    and centered_cover
                    and x_overlap_top >= 0.22
                    and x_overlap_bottom >= 0.22
                    and area_ratio >= max(0.045, max_tb_area * 0.55)
                ):
                    keep = False

        if keep:
            filtered.append(item)

    if not filtered:
        return candidates

    # Final guard: if top+bottom are both strong, keep dress only when truly dominant/central.
    top_kept = [c for c in filtered if str(c.get("type")) == "top"]
    bottom_kept = [c for c in filtered if str(c.get("type")) == "bottom"]
    if top_kept and bottom_kept:
        best_top_k = max(top_kept, key=_area_ratio)
        best_bottom_k = max(bottom_kept, key=_area_ratio)
        top_area = _area_ratio(best_top_k)
        bottom_area = _area_ratio(best_bottom_k)
        if top_area >= 0.04 and bottom_area >= 0.04:
            max_tb_area = max(top_area, bottom_area)
            tmp: list[dict] = []
            for item in filtered:
                if str(item.get("type")) != "dress":
                    tmp.append(item)
                    continue
                b = _bbox(item)
                x0, _, x1, _ = [int(v) for v in b]
                centered_cover = (
                    x0 <= int(person_cx - person_w * 0.12)
                    and x1 >= int(person_cx + person_w * 0.12)
                )
                if _area_ratio(item) >= max(0.08, max_tb_area * 0.75) and centered_cover:
                    tmp.append(item)
            if tmp:
                filtered = tmp

    return filtered

def _merge_bottom_components_if_same_item(
    components: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Merge fragmented bottom components (for example left/right pant legs)
    when parser splits a single garment into adjacent blobs.
    """
    if len(components) <= 1:
        return components

    n = len(components)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    x_gap_limit = max(28, int(image_width * 0.05))
    # Secondary thresholds for wide-leg / stride poses where parser splits one pants
    # into two blobs with limited vertical overlap but nearly aligned lower hems.
    x_gap_limit_relaxed = max(x_gap_limit, int(image_width * 0.16))
    bottom_edge_align_limit = max(18, int(image_height * 0.08))
    y_center_align_ratio = 0.24
    for i in range(n):
        bi = components[i].get("bbox") or [0, 0, image_width, image_height]
        i_x0, i_y0, i_x1, i_y1 = [int(v) for v in bi]
        hi = max(1, int(bi[3]) - int(bi[1]))
        i_cy = 0.5 * (i_y0 + i_y1)
        for j in range(i + 1, n):
            bj = components[j].get("bbox") or [0, 0, image_width, image_height]
            j_x0, j_y0, j_x1, j_y1 = [int(v) for v in bj]
            hj = max(1, int(bj[3]) - int(bj[1]))
            j_cy = 0.5 * (j_y0 + j_y1)
            y_overlap = _bbox_y_overlap_ratio(bi, bj)
            x_gap = _bbox_x_gap(bi, bj)
            height_ratio = float(max(hi, hj)) / float(max(1, min(hi, hj)))
            bottom_edge_delta = abs(i_y1 - j_y1)
            y_center_delta = abs(i_cy - j_cy)
            aligned_bottom_halves = (
                x_gap <= x_gap_limit_relaxed
                and bottom_edge_delta <= bottom_edge_align_limit
                and y_center_delta <= (max(hi, hj) * y_center_align_ratio)
                and height_ratio <= 2.8
            )
            if (y_overlap >= 0.60 and x_gap <= x_gap_limit and height_ratio <= 2.4) or aligned_bottom_halves:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    merged: list[dict] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            comp = dict(components[idxs[0]])
            comp["merged_component_count"] = int(comp.get("merged_component_count", 1))
            merged.append(comp)
            continue

        merged_mask = np.zeros((image_height, image_width), dtype=bool)
        bbox_union = [image_width, image_height, 0, 0]
        merged_area_fallback = 0
        for idx in idxs:
            comp = components[idx]
            b = comp.get("bbox") or [0, 0, image_width, image_height]
            bbox_union[0] = min(bbox_union[0], int(b[0]))
            bbox_union[1] = min(bbox_union[1], int(b[1]))
            bbox_union[2] = max(bbox_union[2], int(b[2]))
            bbox_union[3] = max(bbox_union[3], int(b[3]))
            merged_area_fallback += int(comp.get("area", 0))
            m = comp.get("mask")
            if isinstance(m, np.ndarray) and m.shape[:2] == (image_height, image_width):
                merged_mask |= m.astype(bool)

        merged_bbox = _bbox_from_mask(merged_mask)
        if merged_bbox is None:
            merged_bbox = [int(v) for v in bbox_union]
            merged_area = int(merged_area_fallback)
            merged_mask_out = merged_mask if int(merged_mask.sum()) > 0 else None
        else:
            merged_area = int(merged_mask.sum())
            merged_mask_out = merged_mask

        merged.append(
            {
                "bbox": [int(v) for v in merged_bbox],
                "area": merged_area,
                "mask": merged_mask_out,
                "merged_component_count": len(idxs),
            }
        )

    merged.sort(key=lambda item: int(item.get("area", 0)), reverse=True)
    return merged

def _robust_bbox_from_component_mask(component_mask: np.ndarray, fallback_bbox: list[int]) -> list[int]:
    """
    Tighten component bbox by ignoring tiny sparse outlier pixels on edges.
    This still uses parser mask only; it does not use detector boxes.
    """
    try:
        mask = np.asarray(component_mask).astype(bool)
        if mask.ndim != 2 or mask.size == 0 or int(mask.sum()) <= 0:
            return [int(v) for v in fallback_bbox]

        h, w = mask.shape
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)

        # Keep rows/cols that have meaningful support, not isolated specks.
        row_min = max(1, int(w * 0.006))
        col_min = max(1, int(h * 0.006))

        valid_rows = np.where(row_counts >= row_min)[0]
        valid_cols = np.where(col_counts >= col_min)[0]
        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return [int(v) for v in fallback_bbox]

        x0 = int(valid_cols.min())
        y0 = int(valid_rows.min())
        x1 = int(valid_cols.max()) + 1
        y1 = int(valid_rows.max()) + 1
        if x1 <= x0 or y1 <= y0:
            return [int(v) for v in fallback_bbox]
        return [x0, y0, x1, y1]
    except Exception:
        return [int(v) for v in fallback_bbox]

def _component_label_summary(
    parsing: np.ndarray,
    component_mask: np.ndarray,
    id2label: dict[int, str],
    top_k: int = 3,
) -> list[dict]:
    if parsing.size == 0 or component_mask.size == 0:
        return []
    selected = parsing[component_mask]
    if selected.size == 0:
        return []

    ids, counts = np.unique(selected, return_counts=True)
    total = float(max(1, int(counts.sum())))
    pairs = sorted(zip(ids.tolist(), counts.tolist()), key=lambda x: int(x[1]), reverse=True)
    out: list[dict] = []
    for idx, count in pairs[: max(1, int(top_k))]:
        label = id2label.get(int(idx), f"class_{int(idx)}")
        out.append(
            {
                "id": int(idx),
                "label": str(label),
                "ratio": round(float(count) / total, 4),
            }
        )
    return out

def _build_parser_square_candidates(
    image: Image.Image,
    requested_type: Optional[str],
    min_component_area_ratio: float,
    square_padding_ratio: float,
) -> list[dict]:
    if engine.parser is None:
        raise RuntimeError("Human parser is not enabled.")

    parsing = engine.parser.parse(image)
    h, w = parsing.shape[:2]
    total_pixels = float(max(1, w * h))
    min_ratio = max(0.0005, min(0.25, float(min_component_area_ratio)))
    min_pixels = max(96, int(total_pixels * min_ratio))
    id2label = _parser_runtime_id2label()

    normalized_type = _normalize_garment_type(requested_type)
    if normalized_type in {"top", "bottom", "dress", "outer"}:
        target_types = [normalized_type]
    else:
        target_types = ["top", "bottom", "dress"]

    candidates: list[dict] = []
    for garment_type in target_types:
        keep_ids = _parser_extraction_keep_ids(garment_type)
        if not keep_ids:
            continue

        type_mask = np.isin(parsing, keep_ids)
        if int(type_mask.sum()) < min_pixels:
            continue
        type_mask = binary_open(type_mask, 3)
        type_mask = binary_close(type_mask, 3)

        components = _mask_connected_components(type_mask, min_pixels=min_pixels)
        if garment_type == "bottom" and len(components) > 1:
            components = _merge_bottom_components_if_same_item(
                components,
                image_width=w,
                image_height=h,
            )
        for component in components:
            bbox = component.get("bbox") or [0, 0, w, h]
            area = int(component.get("area", 0))
            component_mask = component.get("mask")
            if area <= 0 or component_mask is None:
                continue
            bbox = _robust_bbox_from_component_mask(component_mask, bbox)

            section_bbox = _expand_section_bbox_by_type(
                bbox=bbox,
                garment_type=garment_type,
                image_width=w,
                image_height=h,
            )
            square_bbox = _square_bbox_from_bbox(
                bbox=section_bbox,
                image_width=w,
                image_height=h,
                padding_ratio=square_padding_ratio,
            )
            sx0, sy0, sx1, sy1 = [int(v) for v in square_bbox]
            side = max(2, int(sx1 - sx0), int(sy1 - sy0))
            square_area = float(max(1, side * side))
            label_summary = _component_label_summary(parsing, component_mask, id2label=id2label, top_k=3)
            ex0, ey0, ex1, ey1 = [int(v) for v in section_bbox]
            rect_crop = image.crop((ex0, ey0, ex1, ey1)).convert("RGB")
            raw_square_crop = _crop_square_with_padding(image, square_bbox)

            subtype = ""
            if garment_type == "bottom":
                for entry in label_summary:
                    lbl = str(entry.get("label", "")).strip().lower()
                    if lbl in {"skirt", "pants", "trousers", "shorts"}:
                        subtype = lbl
                        break
            category_text = _parser_candidate_category_text(garment_type, subtype)

            candidates.append(
                {
                    "type": garment_type,
                    "subtype": subtype,
                    "category_text": category_text,
                    "bbox": [int(v) for v in bbox],
                    "parser_bbox": [int(v) for v in bbox],
                    "section_bbox": [int(v) for v in section_bbox],
                    "square_bbox": square_bbox,
                    "crop_mode": "tight_rect_parser_context",
                    "parser_component_area_ratio": round(float(area) / total_pixels, 6),
                    "component_coverage_in_square": round(float(area) / square_area, 6),
                    "merged_component_count": int(component.get("merged_component_count", 1)),
                    "parser_labels": label_summary,
                    "_crop_image": rect_crop,
                    "_square_crop_image": raw_square_crop,
                }
            )

    candidates = _filter_parser_candidates_by_consistency(
        candidates,
        image_width=w,
        image_height=h,
    )

    candidates.sort(
        key=lambda item: (
            float(item.get("parser_component_area_ratio", 0.0)),
            float(item.get("component_coverage_in_square", 0.0)),
        ),
        reverse=True,
    )
    for idx, candidate in enumerate(candidates):
        candidate["garment_id"] = idx
    return candidates

def _parser_candidate_category_text(garment_type: str, subtype: Optional[str] = None) -> str:
    gt = _normalize_garment_type(garment_type) or "top"
    st = str(subtype or "").strip().lower()
    if gt == "bottom":
        if st in {"pants", "trousers"}:
            return "pants"
        if st == "shorts":
            return "shorts"
        if st == "skirt":
            return "skirt"
        return "bottom"
    if gt in {"top", "dress", "outer"}:
        return gt
    return "garment"

def _build_flux2_garment_only_prompt(
    garment_type: str,
    prompt_description: str,
    category_text: Optional[str] = None,
) -> str:
    clean_desc = " ".join(str(prompt_description or "").split()).strip()
    item_phrase = {
        "top": "top garment",
        "bottom": "bottom garment",
        "dress": "dress",
        "outer": "outerwear piece",
    }.get(_normalize_garment_type(garment_type) or "top", "garment")
    explicit_category = str(category_text or _parser_candidate_category_text(garment_type)).strip().lower()
    return (
        "Generate a standalone product shot of the garment only (no person, no mannequin, no body parts). "
        "Use a clean pure white studio background only (RGB 255,255,255), with no props and no scene context. "
        "Center a single garment in frame with full silhouette visible and edge-to-edge clarity. "
        f"Generate only a single {explicit_category} category garment and no other categories. "
        f"Reconstruct the exact {item_phrase} from the reference crop with strict fidelity to silhouette, "
        "neckline/waist/hem geometry, fit, fabric texture, transparency level, print/embellishment placement, "
        f"and color palette. Garment details: {clean_desc}"
    )

def _to_public_parser_candidate(item: dict) -> dict:
    return {k: v for k, v in item.items() if not str(k).startswith("_")}

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

def _postprocess_extracted_garment_bytes(image_bytes: bytes, garment_type: str = "") -> tuple[bytes, dict]:
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

    # Top-only rim cleanup: trims skin-like fringe on neckline/lower edge without touching sleeves.
    if ANALYZE_TOP_SKIN_RIM_CLEANUP and _normalize_garment_type(garment_type) in {"top", "outer"}:
        try:
            import cv2

            rgba = processed.convert("RGBA")
            arr = np.asarray(rgba).copy()
            alpha = arr[:, :, 3].astype(np.uint8)
            visible = alpha > max(12, ANALYZE_GARMENT_ALPHA_THRESHOLD)
            if np.any(visible):
                skin = _skin_like_mask(arr[:, :, :3]) & visible
                vis_u8 = (visible.astype(np.uint8) * 255)
                eroded = cv2.erode(vis_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
                edge = visible & (~eroded)
                edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0

                ys, _ = np.where(visible)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                h = max(1, y1 - y0)
                yy = np.indices(alpha.shape)[0]
                band = (yy <= (y0 + int(h * 0.32))) | (yy >= (y0 + int(h * 0.68)))

                rim = skin & edge & band
                rim_ratio = float(np.sum(rim)) / float(max(1, np.sum(visible)))
                if rim_ratio <= ANALYZE_TOP_SKIN_RIM_MAX_RATIO:
                    alpha[rim] = 0
                    alpha = build_soft_alpha(alpha > max(12, ANALYZE_GARMENT_ALPHA_THRESHOLD), feather_px=1).astype(np.uint8)
                    arr[:, :, 3] = alpha
                    processed = Image.fromarray(arr, mode="RGBA")
        except Exception as rim_e:
            logger.warning(f"Top skin-rim cleanup skipped: {rim_e}")

    # Optional final render mode for frontend-facing outputs.
    # White background is more forgiving for minor alpha-edge artifacts.
    if ANALYZE_GARMENT_OUTPUT_BACKGROUND == "white":
        try:
            import cv2

            rgba = processed.convert("RGBA")
            arr = np.asarray(rgba).copy()
            alpha = arr[:, :, 3].astype(np.uint8)
            mask = alpha > max(18, ANALYZE_GARMENT_ALPHA_THRESHOLD + 6)
            mask = binary_open(mask, 3)
            mask = binary_close(mask, 3)

            # Drop tiny disconnected islands that often correspond to residual body fragments.
            num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats((mask.astype(np.uint8) * 255))
            clean = np.zeros_like(mask, dtype=bool)
            min_area = max(96, int(mask.size * max(0.0, ANALYZE_EXTRACT_COMPONENT_MIN_RATIO) * 0.35))
            for i in range(1, num_l):
                if int(stats_l[i, cv2.CC_STAT_AREA]) >= min_area:
                    clean[labels_l == i] = True
            alpha_clean = build_soft_alpha(clean, feather_px=1).astype(np.uint8)
            arr[:, :, 3] = alpha_clean
            rgba_clean = Image.fromarray(arr, mode="RGBA")

            white = Image.new("RGB", rgba_clean.size, (255, 255, 255))
            white.paste(rgba_clean, mask=rgba_clean.split()[-1])
            processed = white
        except Exception as white_e:
            logger.warning(f"White background render cleanup failed: {white_e}")
            base = Image.new("RGB", processed.size, (255, 255, 255))
            base.paste(processed.convert("RGBA"), mask=processed.convert("RGBA").split()[-1])
            processed = base

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
        "output_background": ANALYZE_GARMENT_OUTPUT_BACKGROUND,
        "top_skin_rim_cleanup": bool(
            ANALYZE_TOP_SKIN_RIM_CLEANUP and _normalize_garment_type(garment_type) in {"top", "outer"}
        ),
    }

def _crop_and_fit_garment_bytes(image_bytes: bytes) -> tuple[bytes, dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    original_size = (int(image.width), int(image.height))
    bbox = _content_bbox_from_image(image)
    cropped = image.crop(bbox)
    processed = cropped
    if ANALYZE_GARMENT_POSTPROCESS_ENABLED:
        processed = _fit_image_to_aspect(
            processed,
            aspect_w=ANALYZE_GARMENT_TARGET_ASPECT_W,
            aspect_h=ANALYZE_GARMENT_TARGET_ASPECT_H,
        )
    out = io.BytesIO()
    processed.save(out, format="PNG")
    return out.getvalue(), {
        "enabled": ANALYZE_GARMENT_POSTPROCESS_ENABLED,
        "mode": "crop_then_aspect_only",
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "content_bbox": [int(v) for v in bbox],
        "content_size": {"width": int(cropped.width), "height": int(cropped.height)},
        "output_size": {"width": int(processed.width), "height": int(processed.height)},
        "target_aspect": f"{ANALYZE_GARMENT_TARGET_ASPECT_W}:{ANALYZE_GARMENT_TARGET_ASPECT_H}",
        "output_background": "transparent",
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

def _refine_with_showroom_delta(
    extracted_rgba: Image.Image,
    raw_vton_bytes: bytes,
    showroom_bytes: bytes,
    garment_type: str = "",
) -> tuple[Image.Image, dict]:
    try:
        import cv2

        ex = extracted_rgba.convert("RGBA")
        ex_arr = np.asarray(ex).copy()
        alpha = ex_arr[:, :, 3]
        visible = alpha > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD)
        visible_px = int(np.sum(visible))
        if visible_px <= 0:
            return ex, {"enabled": True, "applied": False, "reason": "empty_alpha"}

        raw_img = _flatten_rgba_on_white(Image.open(io.BytesIO(raw_vton_bytes)))
        show_img = _flatten_rgba_on_white(Image.open(io.BytesIO(showroom_bytes)))
        if show_img.size != raw_img.size:
            show_img = show_img.resize(raw_img.size, Image.BICUBIC)
        if ex.size != raw_img.size:
            ex = ex.resize(raw_img.size, Image.LANCZOS).convert("RGBA")
            ex_arr = np.asarray(ex).copy()
            alpha = ex_arr[:, :, 3]
            visible = alpha > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD)
            visible_px = int(np.sum(visible))
            if visible_px <= 0:
                return ex, {"enabled": True, "applied": False, "reason": "empty_after_resize"}

        raw_arr = np.asarray(raw_img, dtype=np.float32)
        show_arr = np.asarray(show_img, dtype=np.float32)
        diff = np.mean(np.abs(raw_arr - show_arr), axis=2)
        delta_threshold = np.full(diff.shape, float(ANALYZE_SHOWROOM_DELTA_THRESHOLD), dtype=np.float32)
        # For tops/outerwear, be stricter on skin-like regions to suppress residual body strips.
        if _normalize_garment_type(garment_type) in {"top", "outer"}:
            skin_like = _skin_like_mask(raw_arr.astype(np.uint8))
            delta_threshold[skin_like] = float(ANALYZE_SHOWROOM_DELTA_THRESHOLD + ANALYZE_SHOWROOM_DELTA_SKIN_EXTRA)
        delta_mask = diff >= delta_threshold
        if ANALYZE_SHOWROOM_DELTA_DILATE > 0:
            k = int(ANALYZE_SHOWROOM_DELTA_DILATE)
            delta_mask = cv2.dilate(delta_mask.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1) > 0

        refined_mask = visible & delta_mask
        refined_px = int(np.sum(refined_mask))
        retain_ratio = float(refined_px) / float(max(1, visible_px))

        meta = {
            "enabled": True,
            "applied": False,
            "visible_before": visible_px,
            "visible_after": refined_px,
            "retain_ratio": round(retain_ratio, 6),
            "threshold": float(ANALYZE_SHOWROOM_DELTA_THRESHOLD),
            "skin_extra_threshold": float(ANALYZE_SHOWROOM_DELTA_SKIN_EXTRA),
            "dilate": int(ANALYZE_SHOWROOM_DELTA_DILATE),
            "min_retain_ratio": float(ANALYZE_SHOWROOM_DELTA_MIN_RETAIN_RATIO),
        }

        if retain_ratio < ANALYZE_SHOWROOM_DELTA_MIN_RETAIN_RATIO:
            meta["reason"] = "retain_ratio_too_low"
            return ex, meta

        alpha_new = np.where(refined_mask, alpha, 0).astype(np.uint8)
        alpha_new = build_soft_alpha(alpha_new > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD), feather_px=ANALYZE_EXTRACT_EDGE_FEATHER_PX).astype(np.uint8)
        ex_arr[:, :, 3] = alpha_new
        meta["applied"] = True
        return Image.fromarray(ex_arr, mode="RGBA"), meta
    except Exception as exc:
        return extracted_rgba, {"enabled": True, "applied": False, "reason": f"error:{exc}"}

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

def _parser_category_ids(category: str, fallback: Optional[list[int]] = None) -> list[int]:
    if engine.parser is not None and hasattr(engine.parser, "category_ids"):
        try:
            ids = list(getattr(engine.parser, "category_ids")(category) or [])
            if ids:
                return [int(v) for v in ids]
        except Exception:
            pass
    return [int(v) for v in (fallback or [])]


def _parser_runtime_label2id() -> dict[str, int]:
    if engine.parser is None:
        return {}
    out: dict[str, int] = {}
    try:
        runner = getattr(getattr(engine.parser, "_parser_fn", None), "__self__", None)
        runtime = getattr(runner, "label2id", None)
        if isinstance(runtime, dict):
            for k, v in runtime.items():
                try:
                    key = str(k).strip().lower().replace("-", "_").replace(" ", "_")
                    out[key] = int(v)
                except Exception:
                    continue
    except Exception:
        pass
    if out:
        return out
    try:
        runtime_fn = getattr(engine.parser, "_runtime_labels", None)
        runtime = runtime_fn() if callable(runtime_fn) else {}
        if isinstance(runtime, dict):
            for k, v in runtime.items():
                try:
                    key = str(k).strip().lower().replace("-", "_").replace(" ", "_")
                    out[key] = int(v)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _parser_extraction_keep_ids(garment_type: str) -> list[int]:
    """
    Extraction-only mapping for the 18-class FASHN parser.
    Keeps category domains explicit:
      - top: top/upper
      - bottom: pants/skirt/shorts (+belt)
      - dress: dress
    """
    g = _normalize_garment_type(garment_type) or "top"
    label2id = _parser_runtime_label2id()

    alias_map = {
        "top": ["top", "upper_clothes", "upper"],
        "outer": ["outerwear", "outer", "coat", "jacket", "blazer", "top", "upper_clothes", "upper"],
        "bottom": ["pants", "skirt", "shorts", "trousers", "belt"],
        "dress": ["dress"],
    }
    aliases = alias_map.get(g, alias_map["top"])
    ids = []
    for alias in aliases:
        key = str(alias).strip().lower().replace("-", "_").replace(" ", "_")
        if key in label2id:
            ids.append(int(label2id[key]))
    ids = sorted(set(ids))
    if ids:
        return ids

    # Fallback to existing parser category logic only if runtime labels are unavailable.
    fallback_map = {
        "top": [4, 17],
        "outer": [4, 17, 8],
        "bottom": [5, 6, 8],
        "dress": [7],
    }
    return _parser_category_ids(g, fallback_map.get(g, [4, 17]))

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
    # Use parser semantic labels (works across model variants).
    body_ids = _parser_category_ids("body", [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16])
    body_mask = np.isin(parsing, body_ids) & visible_mask
    body_pixels = int(np.sum(body_mask))
    ratio = float(body_pixels) / float(max(1, visible_pixels))
    return {
        "ratio": ratio,
        "visible_pixels": visible_pixels,
        "body_pixels": body_pixels,
        "body_mask": body_mask,
    }

def _skin_like_mask(rgb_image: np.ndarray) -> np.ndarray:
    try:
        import cv2

        rgb = np.asarray(rgb_image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return np.zeros((0, 0), dtype=bool)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        skin_hsv = (((h <= 25) | (h >= 160)) & (s >= 20) & (s <= 200) & (v >= 35))

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        skin_rgb = (
            (r > 95) & (g > 40) & (b > 20)
            & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15)
            & (np.abs(r - g) > 15)
            & (r > g) & (r > b)
        )

        # Keep mask conservative to avoid stripping light/skin-like garments.
        mask = (skin_hsv & skin_ycrcb) | (skin_ycrcb & skin_rgb)
        mask = binary_open(mask, 3)
        mask = binary_close(mask, 3)
        return mask.astype(bool)
    except Exception:
        h, w = rgb_image.shape[:2]
        return np.zeros((h, w), dtype=bool)

def _extract_cloth_from_crop(
    crop: Image.Image,
    garment_type: str,
    enforce_safety_guards: bool = True,
) -> tuple[Image.Image, dict]:
    """
    High-quality garment extraction from a person/mannequin image.
    Primary path uses type-aware human parser masks for garment isolation.
    """
    selected_type = _normalize_garment_type(garment_type) or "top"
    total_pixels = float(max(1, crop.width * crop.height))
    img_np = np.asarray(crop.convert("RGB"))
    h, w = img_np.shape[:2]

    try:
        import cv2

        # STEP 1: Type-aware parser setup.
        # We use Segformer B2 Clothes labels: 4:Upper, 6:Pants, 5:Skirt, 7:Dress, 8:Belt, 9/10:Shoes, 11:Face, 12/13:Legs, 14/15:Arms
        parsing = engine.parser.parse(crop)

        strict_keep_ids = _parser_extraction_keep_ids(selected_type)
        strict_keep_ids = sorted(set(int(v) for v in strict_keep_ids))
        keep_adjustment = {}

        # Some FASHN parser outputs place upper garments under dress-like class IDs.
        # For top extraction only, backfill with dress ID when top mask is too small.
        if ANALYZE_PARSER_TOP_DRESS_BACKFILL and selected_type in {"top", "outer"}:
            label2id = _parser_runtime_label2id()
            top_id = label2id.get("top")
            dress_id = label2id.get("dress")
            if top_id is not None and dress_id is not None and int(dress_id) not in strict_keep_ids:
                top_ratio = float(np.mean(parsing == int(top_id)))
                dress_ratio = float(np.mean(parsing == int(dress_id)))
                if top_ratio < ANALYZE_PARSER_TOP_MIN_RATIO and dress_ratio >= ANALYZE_PARSER_DRESS_BACKFILL_MIN_RATIO:
                    strict_keep_ids = sorted(set(strict_keep_ids + [int(dress_id)]))
                    keep_adjustment = {
                        "top_id": int(top_id),
                        "dress_id": int(dress_id),
                        "top_ratio": round(top_ratio, 6),
                        "dress_ratio": round(dress_ratio, 6),
                        "applied": True,
                    }

        if ANALYZE_EXTRACT_PARSER_ONLY:
            # Strict parser-only extraction: keep only the requested garment region
            # from the try-on frame, then pass that output to enhancement.
            seed_mask = np.isin(parsing, strict_keep_ids)
            keep_dilate = max(0, int(ANALYZE_EXTRACT_KEEP_DILATE))
            if keep_dilate > 0:
                seed_mask = cv2.dilate(
                    (seed_mask.astype(np.uint8) * 255),
                    np.ones((keep_dilate, keep_dilate), np.uint8),
                    iterations=1,
                ) > 0
            seed_mask = binary_open(seed_mask, 3)
            seed_mask = binary_close(seed_mask, 3)

            # Remove tiny islands while preserving the parser-selected garment body.
            num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats((seed_mask.astype(np.uint8) * 255))
            clean_mask = np.zeros_like(seed_mask, dtype=bool)
            area_min = max(160, int(total_pixels * ANALYZE_EXTRACT_COMPONENT_MIN_RATIO))
            for j in range(1, num_l):
                if int(stats_l[j, cv2.CC_STAT_AREA]) >= area_min:
                    clean_mask[labels_l == j] = True

            alpha = build_soft_alpha(clean_mask, feather_px=ANALYZE_EXTRACT_EDGE_FEATHER_PX).astype(np.uint8)
            final_mask = alpha > max(8, ANALYZE_EXTRACT_EDGE_FEATHER_PX * 6)
            final_area_ratio = float(final_mask.sum()) / total_pixels
            if final_area_ratio <= 0.0:
                raise RuntimeError(f"Parser-only extraction produced empty result for: {selected_type}")

            alpha_plane = alpha[:, :, None]
            cloth_rgba = np.concatenate([img_np, alpha_plane], axis=2)
            cloth_image = Image.fromarray(cloth_rgba, mode="RGBA")
            return cloth_image, {
                "path": "parser_type_only",
                "mask_area_ratio": round(final_area_ratio, 6),
                "rescue_mode": "none",
                "parser_only": True,
                "keep_ids": strict_keep_ids,
                "keep_adjustment": keep_adjustment,
            }

        # Non-parser-only mode still uses parser masks only.
        base_body_kill_ids = _parser_category_ids("body", [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16])
        kill_ids = list(base_body_kill_ids)
        rescue_mode = "strict"

        if selected_type == "top":
            # Kill opposite garment classes for stronger separation.
            kill_ids.extend(_parser_category_ids("bottom", [5, 6, 8]))
            kill_ids.extend(_parser_category_ids("dress", [7]))
        elif selected_type == "bottom":
            kill_ids.extend(_parser_category_ids("top", [4, 17]))
            kill_ids.extend(_parser_category_ids("dress", [7]))
        elif selected_type == "dress":
            # Dress already keeps broad clothing regions.
            pass

        kill_ids = sorted(set(int(v) for v in kill_ids))

        path_name = "parser_type_guided"

        def _compose_alpha_from_seed(seed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            seed_mask = binary_open(seed_mask, 3)
            seed_mask = binary_close(seed_mask, 3)

            # Dust removal (eliminate isolated floating specks).
            num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats((seed_mask.astype(np.uint8) * 255))
            clean_alpha = np.zeros_like(seed_mask, dtype=np.uint8)
            area_min = max(160, int(total_pixels * ANALYZE_EXTRACT_COMPONENT_MIN_RATIO))
            for j in range(1, num_l):
                if int(stats_l[j, cv2.CC_STAT_AREA]) >= area_min:
                    clean_alpha[labels_l == j] = 255

            alpha_local = build_soft_alpha(clean_alpha > 0, feather_px=ANALYZE_EXTRACT_EDGE_FEATHER_PX)
            final_mask_local = alpha_local > max(8, ANALYZE_EXTRACT_EDGE_FEATHER_PX * 6)
            area_ratio_local = float(final_mask_local.sum()) / total_pixels
            return alpha_local, final_mask_local, area_ratio_local

        def _compose_alpha_with_kill(active_kill_ids: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
            parser_kill = np.isin(parsing, active_kill_ids)
            kill_k = max(1, int(ANALYZE_EXTRACT_PARSER_KILL_DILATE))
            parser_kill = cv2.dilate(
                (parser_kill.astype(np.uint8) * 255),
                np.ones((kill_k, kill_k), np.uint8),
                iterations=1,
            ) > 0

            # Type-aware keep-region gating removes residual body at waist/hands/legs.
            keep_mask = np.isin(parsing, strict_keep_ids)
            keep_dilate = max(0, int(ANALYZE_EXTRACT_KEEP_DILATE))
            if keep_dilate > 0:
                keep_mask = cv2.dilate(
                    (keep_mask.astype(np.uint8) * 255),
                    np.ones((keep_dilate, keep_dilate), np.uint8),
                    iterations=1,
                ) > 0
            seed_mask = keep_mask & (~parser_kill)
            return _compose_alpha_from_seed(seed_mask)

        def _compose_alpha_with_keep(keep_ids: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
            keep_mask = np.isin(parsing, keep_ids)
            keep_mask = cv2.dilate((keep_mask.astype(np.uint8) * 255), np.ones((2, 2), np.uint8), iterations=1) > 0
            return _compose_alpha_from_seed(keep_mask)

        def _compose_alpha_body_only() -> tuple[np.ndarray, np.ndarray, float]:
            garment_ids = _parser_category_ids("garment_fallback", [4, 5, 6, 7, 8, 17])
            garment_mask = np.isin(parsing, garment_ids)
            body_mask = np.isin(parsing, base_body_kill_ids)
            kill_k = max(1, int(ANALYZE_EXTRACT_PARSER_KILL_DILATE))
            body_mask = cv2.dilate(
                (body_mask.astype(np.uint8) * 255),
                np.ones((kill_k, kill_k), np.uint8),
                iterations=1,
            ) > 0
            seed_mask = garment_mask & (~body_mask)
            return _compose_alpha_from_seed(seed_mask)

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
            if rescue_mode == "body_only" and final_area_ratio > 0.0:
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
                alpha_body, mask_body, area_body = _compose_alpha_body_only()
                if area_body > final_area_ratio:
                    logger.warning(
                        "Extraction mask empty after strict/keep rescue (type=%s). Using parser body-only rescue %.6f.",
                        selected_type,
                        area_body,
                    )
                    alpha, final_mask, final_area_ratio = alpha_body, mask_body, area_body
                    rescue_mode = "body_only"

    except Exception as e:
        if ANALYZE_EXTRACT_PARSER_ONLY:
            raise RuntimeError(f"Parser-only extraction failed: {e}")
        logger.warning(f"Primary extraction path failed, falling back to parser: {e}")
        try:
            parsing = engine.parser.parse(crop)
            garment_ids = _parser_category_ids("garment_fallback", [4, 5, 6, 7, 8, 17])
            kill_ids_fallback = _parser_category_ids("kill_fallback", [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16])
            garment_mask = np.isin(parsing, garment_ids)
            kill_mask = np.isin(parsing, kill_ids_fallback)
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
        "enforce": bool(enforce_safety_guards),
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
                    visible_before = int(np.sum(alpha_arr > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD)))
                    skin_mask = _skin_like_mask(rgba_arr[:, :, :3])
                    strip_mask = (body_mask_u8 > 0) & skin_mask & (alpha_arr > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD))
                    strip_pixels = int(np.sum(strip_mask))
                    strip_ratio = float(strip_pixels) / float(max(1, visible_before))
                    body_guard["strip_mode"] = "skin_gated"
                    body_guard["strip_pixels_candidate"] = int(np.sum(body_mask_u8 > 0))
                    body_guard["strip_pixels_applied"] = strip_pixels
                    body_guard["strip_ratio_applied"] = round(strip_ratio, 6)

                    skip_reason = None
                    if strip_pixels <= 0:
                        skip_reason = "no_skin_overlap"
                    elif strip_ratio > ANALYZE_EXTRACT_BODY_STRIP_MAX_RATIO:
                        skip_reason = f"overbroad:{strip_ratio:.6f}"

                    if skip_reason:
                        body_guard["warning"] = f"body_strip_skipped:{skip_reason}"
                        if enforce_safety_guards:
                            raise RuntimeError(
                                f"Extraction has body leakage but cleanup is unsafe ({skip_reason})."
                            )
                        leak_after = leak_before
                    else:
                        alpha_arr[strip_mask] = 0
                        cleaned_mask = alpha_arr > max(8, ANALYZE_GARMENT_ALPHA_THRESHOLD)
                        alpha_arr = build_soft_alpha(cleaned_mask, feather_px=ANALYZE_EXTRACT_EDGE_FEATHER_PX).astype(np.uint8)
                        rgba_arr[:, :, 3] = alpha_arr
                        cloth_image = Image.fromarray(rgba_arr, mode="RGBA")
                        leak_after = _body_leakage_stats(cloth_image)

                    body_guard["ratio_after"] = round(float(leak_after.get("ratio", 0.0)), 6)
                    body_guard["body_pixels_after"] = int(leak_after.get("body_pixels", 0))

                    if float(leak_after.get("ratio", 0.0)) > ANALYZE_EXTRACT_MAX_BODY_RATIO:
                        if enforce_safety_guards:
                            raise RuntimeError(
                                f"Extraction contains mannequin/body remnants (ratio={leak_after.get('ratio'):.6f})."
                            )
                        body_guard["warning"] = (
                            "body_ratio_above_threshold:"
                            f"{float(leak_after.get('ratio', 0.0)):.6f}"
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
    strict_safety = bool(ANALYZE_VTON_STRICT_SAFETY_CHECKS)
    guard_warnings: list[str] = []
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
        "seed": ANALYZE_VTON_SEED,
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
            msg = "VTON returned showroom model image unchanged (exact match)."
            if strict_safety:
                raise RuntimeError(msg)
            guard_warnings.append("showroom_exact_match")
            logger.warning(msg)
        if (
            float(showroom_similarity.get("resized_mae", 999.0)) <= ANALYZE_VTON_MAX_SHOWROOM_MAE
            and float(showroom_similarity.get("change_ratio", 0.0)) < ANALYZE_VTON_MIN_SHOWROOM_CHANGE_RATIO
        ):
            msg = (
                "VTON returned showroom-like output "
                f"(resized_mae={showroom_similarity.get('resized_mae')}, "
                f"change_ratio={showroom_similarity.get('change_ratio')})."
            )
            if strict_safety:
                raise RuntimeError(msg)
            guard_warnings.append("showroom_similarity_high")
            logger.warning(msg)

    # --- LOCAL EXTRACTION FROM MANNEQUIN ---
    # We now have the mannequin wearing the garment (raw_vton_bytes).
    # We perform local extraction to isolate the garment and ensure no holes are introduced.
    t_extract_start = time.time()
    vton_img_mannequin = Image.open(io.BytesIO(raw_vton_bytes)).convert("RGB")
    logger.info(f"Loaded VTON mannequin image for local extraction: {vton_img_mannequin.size} {vton_img_mannequin.mode}")
    extraction_meta = {}
    try:
        extracted_cloth, extraction_meta = _extract_cloth_from_crop(
            vton_img_mannequin,
            garment_type,
            enforce_safety_guards=strict_safety,
        )
        extraction_meta["parser_input_source"] = "vton_raw_tryon_frame"
        extraction_meta["parser_input_size"] = {
            "width": int(vton_img_mannequin.width),
            "height": int(vton_img_mannequin.height),
        }
        if (
            ANALYZE_SHOWROOM_DELTA_REFINEMENT_ENABLED
            and ANALYZE_VTON_USE_SHOWROOM_PERSON
            and bool(showroom_bytes)
            and bool(raw_vton_bytes)
        ):
            extracted_cloth, showroom_delta_meta = _refine_with_showroom_delta(
                extracted_cloth,
                raw_vton_bytes=raw_vton_bytes,
                showroom_bytes=showroom_bytes,
                garment_type=garment_type,
            )
            extraction_meta["showroom_delta_refine"] = showroom_delta_meta
        extraction_meta["vto_path"] = "vton_fallback_mannequin_local_parser"
    except Exception as extract_err:
        if strict_safety or ANALYZE_EXTRACT_PARSER_ONLY:
            raise
        logger.warning(f"Local extraction failed in relaxed mode, using raw VTON frame: {extract_err}")
        guard_warnings.append(f"extract_fallback_raw:{str(extract_err)[:120]}")
        extracted_cloth = vton_img_mannequin.convert("RGBA")
        extraction_meta = {
            "path": "vton_raw_passthrough_relaxed",
            "rescue_mode": "raw_frame",
            "warning": str(extract_err),
            "vto_path": "vton_raw_passthrough_relaxed",
        }

    # Save as bytes for postprocessing
    buf = io.BytesIO()
    extracted_cloth.save(buf, format="PNG")
    final_vton_bytes = buf.getvalue()
    t_extract_s = time.time() - t_extract_start

    source_similarity = _compare_image_similarity(source_bytes, final_vton_bytes)
    if ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH:
        if source_similarity.get("exact_bytes"):
            msg = "VTON output is byte-identical to source crop (passthrough)."
            if strict_safety:
                raise RuntimeError(msg)
            guard_warnings.append("source_exact_match")
            logger.warning(msg)
        if float(source_similarity.get("resized_mae", 999.0)) <= ANALYZE_VTON_MAX_SOURCE_MAE:
            msg = f"VTON output too similar to source crop (resized_mae={source_similarity.get('resized_mae')})."
            if strict_safety:
                raise RuntimeError(msg)
            guard_warnings.append("source_similarity_high")
            logger.warning(msg)
    
    t_post_start = time.time()
    if bool(extraction_meta.get("parser_only")):
        # Keep parser-only extraction, but still apply final content-crop + 2:3 framing.
        try:
            processed_vton_bytes, postprocess_meta = _crop_and_fit_garment_bytes(final_vton_bytes)
            postprocess_meta["mode"] = "parser_only_crop_then_aspect"
        except Exception as parser_pp_err:
            logger.warning(f"Parser-only crop/aspect step failed, returning raw parser extraction: {parser_pp_err}")
            processed_vton_bytes = final_vton_bytes
            postprocess_meta = {
                "enabled": False,
                "bypassed": "parser_only_direct",
                "output_background": "transparent",
                "warning": str(parser_pp_err),
            }
    else:
        processed_vton_bytes, postprocess_meta = _postprocess_extracted_garment_bytes(
            final_vton_bytes,
            garment_type=garment_type,
        )
    try:
        image_quality = _validate_vton_output_bytes(processed_vton_bytes)
    except Exception as quality_err:
        if strict_safety:
            raise
        logger.warning(f"VTON output quality validation skipped in relaxed mode: {quality_err}")
        guard_warnings.append(f"quality_validation_skipped:{str(quality_err)[:120]}")
        image_quality = {
            "validated": False,
            "warning": str(quality_err),
        }
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
            "strict_safety": strict_safety,
            "guard_warnings": guard_warnings,
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

class Flux2TryonProduct(BaseModel):
    image: str
    promptDescription: Optional[str] = None

class Flux2TryonUserImage(BaseModel):
    tryonImage: str
    promptDescription: Optional[str] = None

class Flux2TryonRequest(BaseModel):
    products: List[Flux2TryonProduct]
    user_image: Flux2TryonUserImage
    description_backend: Optional[str] = None
    description_compare: bool = False
    steps: int = Field(default=6, ge=4, le=30)
    seed: int = Field(default=23, ge=0, le=2147483647)

class ParserJoyCaptionAnalyzeRequest(BaseModel):
    image_url: str
    garment_type: Optional[str] = None
    selected_index: Optional[int] = Field(default=None, ge=0)
    square_padding_ratio: float = Field(default=0.12, ge=0.0, le=0.60)
    min_component_area_ratio: float = Field(default=0.01, ge=0.0005, le=0.25)
    upload_candidate_previews: bool = True
    run_flux_garment_only: bool = False
    flux_steps: int = Field(default=8, ge=4, le=30)
    flux_seed: int = Field(default=23, ge=0, le=2147483647)
    flux_extract_only: bool = False
    flux_extract_strict_safety: bool = True

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

@app.post("/v1/analyze/parser-joycaption")
async def analyze_parser_joycaption(request: ParserJoyCaptionAnalyzeRequest):
    """
    New parser-first garment analysis API.
    This route is isolated and does not modify /analyze or /v1/flux/tryon behavior.
    """
    t0 = time.time()
    stage_timings = {
        "download_image_s": 0.0,
        "parser_detect_s": 0.0,
        "preview_upload_s": 0.0,
        "joycaption_s": 0.0,
        "selected_upload_s": 0.0,
        "flux_garment_gen_s": 0.0,
        "flux_extract_s": 0.0,
        "flux_upload_s": 0.0,
    }
    try:
        image_url = str(request.image_url or "").strip()
        if not image_url:
            raise HTTPException(status_code=422, detail="image_url is required")

        requested_type_raw = str(request.garment_type or "").strip()
        requested_type = _normalize_garment_type(requested_type_raw) if requested_type_raw else None
        if requested_type_raw and requested_type is None:
            raise HTTPException(status_code=422, detail="garment_type must be one of top, bottom, dress, outer")
        if engine.parser is None:
            raise HTTPException(status_code=503, detail="human parser is not available")

        t_stage = time.time()
        source_img = download_image(image_url).convert("RGB")
        stage_timings["download_image_s"] = round(time.time() - t_stage, 4)

        async with gpu_semaphore:
            t_stage = time.time()
            candidates = _build_parser_square_candidates(
                image=source_img,
                requested_type=requested_type,
                min_component_area_ratio=request.min_component_area_ratio,
                square_padding_ratio=request.square_padding_ratio,
            )
            stage_timings["parser_detect_s"] = round(time.time() - t_stage, 4)

            if not candidates:
                raise HTTPException(status_code=400, detail="No garment component found for the requested type")

            preview_upload_start = time.time()
            if request.upload_candidate_previews:
                for candidate in candidates:
                    crop_image = candidate.get("_crop_image")
                    if crop_image is None:
                        continue
                    crop_buf = io.BytesIO()
                    crop_image.save(crop_buf, format="PNG")
                    candidate["preview_url"] = _upload_or_raise(crop_buf.getvalue(), container=VTO_OUTPUT_CONTAINER)
            stage_timings["preview_upload_s"] = round(time.time() - preview_upload_start, 4)

            selection_required = len(candidates) > 1 and request.selected_index is None
            if selection_required:
                return {
                    "status": "selection_required",
                    "selection_required": True,
                    "requested_garment_type": requested_type,
                    "total_candidates": len(candidates),
                    "candidates": [_to_public_parser_candidate(c) for c in candidates],
                    "stageTimings": stage_timings,
                    "total_latency": round(time.time() - t0, 4),
                }

            selected_index = int(request.selected_index) if request.selected_index is not None else 0
            if selected_index < 0 or selected_index >= len(candidates):
                raise HTTPException(status_code=422, detail=f"selected_index must be between 0 and {len(candidates) - 1}")

            selected = candidates[selected_index]
            selected_crop = selected.get("_crop_image")
            if selected_crop is None:
                sx0, sy0, sx1, sy1 = [int(v) for v in selected.get("square_bbox", [0, 0, source_img.width, source_img.height])]
                selected_crop = _crop_square_with_padding(source_img, [sx0, sy0, sx1, sy1])

            selected_type = str(selected.get("type") or requested_type or "top")
            selected_category_text = str(
                selected.get("category_text") or _parser_candidate_category_text(selected_type, str(selected.get("subtype") or ""))
            ).strip()
            joy_instruction = (
                f"The image is a section crop with person context. Describe only the selected {selected_category_text} garment "
                "for virtual try-on, not the person/background. "
                "Return a single detailed line with exact attributes: garment type, silhouette, fit, neckline, sleeves, "
                "waistline, hem shape/length, fabric, texture, transparency, pattern, embellishments, and dominant colors. "
                "Do not repeat words or phrases."
            )
            t_stage = time.time()
            prompt_description = engine.joycaption.describe_garment(
                selected_crop,
                instruction_override=joy_instruction,
            )
            stage_timings["joycaption_s"] = round(time.time() - t_stage, 4)
            prompt_description = " ".join(str(prompt_description or "").split()).strip()
            if not prompt_description:
                # Fallback only for this endpoint to avoid hard failures on occasional JoyCaption empty output.
                try:
                    prompt_description = " ".join(
                        str(engine.florence.describe_garment_short(selected_crop) or "").split()
                    ).strip()
                except Exception as fallback_err:
                    logger.warning(f"Florence fallback failed in parser-joycaption endpoint: {fallback_err}")
                    color_terms = _extract_dominant_color_labels(selected_crop, top_k=3)
                    color_text = ", ".join(color_terms) if color_terms else "unknown colors"
                    prompt_description = (
                        f"{selected_category_text} garment with full silhouette visible, "
                        f"dominant colors: {color_text}, preserve exact shape, fabric texture, "
                        "embellishments, and hem/neckline geometry from the reference crop."
                    )
            if not prompt_description:
                raise HTTPException(status_code=502, detail="Caption models did not return a garment description")

            selected_upload_start = time.time()
            selected_buf = io.BytesIO()
            selected_crop.save(selected_buf, format="PNG")
            selected_crop_url = _upload_or_raise(selected_buf.getvalue(), container=VTO_OUTPUT_CONTAINER)
            stage_timings["selected_upload_s"] = round(time.time() - selected_upload_start, 4)

            flux_prompt = _build_flux2_garment_only_prompt(
                selected_type,
                prompt_description,
                category_text=selected_category_text,
            )
            flux_result_payload = None
            if bool(request.run_flux_garment_only):
                t_flux = time.time()
                flux_result = engine.flux2.run_tryon(
                    person_image=selected_crop,
                    board_image=selected_crop,
                    prompt=flux_prompt,
                    steps=int(request.flux_steps),
                    seed=int(request.flux_seed),
                )
                stage_timings["flux_garment_gen_s"] = round(time.time() - t_flux, 4)

                flux_raw_img = flux_result["image"].convert("RGB")
                flux_raw_buf = io.BytesIO()
                flux_raw_img.save(flux_raw_buf, format="PNG")
                flux_raw_bytes = flux_raw_buf.getvalue()

                extraction_meta = {}
                postprocess_meta = {}
                extraction_warning = ""
                final_flux_bytes = flux_raw_bytes
                if bool(request.flux_extract_only):
                    t_extract = time.time()
                    try:
                        extracted_rgba, extraction_meta = _extract_cloth_from_crop(
                            flux_raw_img,
                            selected_type,
                            enforce_safety_guards=bool(request.flux_extract_strict_safety),
                        )
                        ext_buf = io.BytesIO()
                        extracted_rgba.save(ext_buf, format="PNG")
                        extracted_bytes = ext_buf.getvalue()
                        final_flux_bytes, postprocess_meta = _postprocess_extracted_garment_bytes(
                            extracted_bytes,
                            garment_type=selected_type,
                        )
                    except Exception as flux_extract_err:
                        if bool(request.flux_extract_strict_safety):
                            raise HTTPException(
                                status_code=502,
                                detail=f"Flux garment extraction failed: {flux_extract_err}",
                            )
                        extraction_warning = str(flux_extract_err)
                    stage_timings["flux_extract_s"] = round(time.time() - t_extract, 4)

                t_upload = time.time()
                flux_raw_url = _upload_or_raise(flux_raw_bytes, container=VTO_OUTPUT_CONTAINER)
                flux_final_url = _upload_or_raise(final_flux_bytes, container=VTO_OUTPUT_CONTAINER)
                stage_timings["flux_upload_s"] = round(time.time() - t_upload, 4)

                flux_result_payload = {
                    "status": "success",
                    "steps": int(request.flux_steps),
                    "seed": int(request.flux_seed),
                    "latency": round(float(flux_result.get("latency", 0.0)), 4),
                    "prompt": flux_prompt,
                    "raw_output_url": flux_raw_url,
                    "output_url": flux_final_url or flux_raw_url,
                    "extract_only": bool(request.flux_extract_only),
                    "extract_strict_safety": bool(request.flux_extract_strict_safety),
                    "extraction_warning": extraction_warning or None,
                    "extraction_meta": extraction_meta or {},
                    "postprocess_meta": postprocess_meta or {},
                }

            return {
                "status": "success",
                "selection_required": False,
                "requested_garment_type": requested_type,
                "selected_index": selected_index,
                "total_candidates": len(candidates),
                "selected_item": {
                    **_to_public_parser_candidate(selected),
                    "preview_url": selected.get("preview_url") or selected_crop_url,
                    "selected_category_text": selected_category_text,
                    "promptDescription": prompt_description,
                    "fluxGarmentOnlyPrompt": flux_prompt,
                    "fluxGarmentGeneration": flux_result_payload,
                },
                "candidates": [_to_public_parser_candidate(c) for c in candidates],
                "stageTimings": stage_timings,
                "total_latency": round(time.time() - t0, 4),
            }
    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Parser JoyCaption analyze failed: {err}")
        raise HTTPException(status_code=500, detail=str(err))

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

            # 2a. Optional parser split fallback (disabled by default for routing stability).
            raw_detected_count = len(instances)
            if (
                ANALYZE_USE_PARSER_FOR_PREROUTING
                and ANALYZE_ENABLE_PARSER_SPLIT
                and engine.parser is not None
                and len(instances) <= 1
            ):
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
                        base_score = float(classify.get("score", 0.0) or 0.0)
                        dress_locked = (
                            base_type == "dress"
                            and base_score >= ANALYZE_FLORENCE_DRESS_LOCK_MIN_SCORE
                            and requested_type not in {"top", "bottom"}
                        )
                        allow_split = not dress_locked
                        if allow_split:
                            heuristics = _heuristic_split_candidates(img, base_inst)
                            heuristic_candidates_count = max(heuristic_candidates_count, len(heuristics))
                            if len(heuristics) >= 2:
                                heuristic_split_used = True
                                instances = heuristics
                except Exception as heur_err:
                    logger.warning(f"Heuristic split fallback failed: {heur_err}")

            # 2b.1 Optional parser tightening. Keep disabled for first-step routing by default.
            if ANALYZE_USE_PARSER_FOR_PREROUTING and len(instances) >= 1 and engine.parser is not None:
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
            elif (
                selected_index is None
                and not requested_type
                and len(items) > 1
                and ANALYZE_AUTO_SELECT_MULTI_DRESS
            ):
                normalized_types = [
                    _normalize_garment_type(str(item.get("type") or "")) for item in items
                ]
                unique_norm_types = sorted({t for t in normalized_types if t})
                # If detector+Florence agree this is effectively a dress-only scene,
                # select the highest-ranked item instead of forcing a manual selection.
                if len(unique_norm_types) == 1 and unique_norm_types[0] == "dress":
                    auto_selected_index = max(
                        range(len(items)),
                        key=lambda idx: _item_rank_score(items[idx]),
                    )

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
                    requested_type_forced = bool(forced_type)
                    candidate_types: list[str] = [selected_type]
                    if not requested_type_forced:
                        # Auto mode guard: recover from parser-empty masks by trying
                        # nearby garment families before returning 502.
                        if selected_type == "dress":
                            candidate_types.extend(["top", "bottom"])
                        else:
                            candidate_types.append("dress")
                    # Keep order stable while removing duplicates.
                    candidate_types = list(dict.fromkeys(candidate_types))

                    active_type = selected_type
                    for candidate_idx, candidate_type in enumerate(candidate_types):
                        active_type = candidate_type
                        max_attempts = 2 if candidate_idx == 0 else 1
                        for attempt in range(max_attempts):
                            try:
                                fallback = _run_vton_cloth_only_fallback(
                                    selected_crop_url,
                                    candidate_type,
                                    vto_mode=is_single_item
                                )
                                if candidate_type != selected_type:
                                    logger.info(
                                        "Auto extraction type fallback applied: %s -> %s",
                                        selected_type,
                                        candidate_type,
                                    )
                                break
                            except Exception as run_err:
                                last_fallback_err = run_err
                                logger.warning(
                                    "VTON extract attempt %d failed (type=%s, bbox=%s): %s",
                                    attempt + 1,
                                    candidate_type,
                                    selected_item.get("bbox"),
                                    run_err,
                                )
                                if attempt < (max_attempts - 1):
                                    time.sleep(0.35)
                                    continue
                                # Only auto-fallback on parser-empty errors.
                                if (
                                    not requested_type_forced
                                    and candidate_idx < (len(candidate_types) - 1)
                                    and "Parser-only extraction produced empty result" in str(run_err)
                                ):
                                    break
                        if fallback is not None:
                            break

                    if fallback is None and last_fallback_err is not None:
                        raise last_fallback_err
                    extracted_url = str(fallback.get("url") or "")
                    extraction_meta = dict(fallback.get("meta") or {})
                    if (
                        fallback is not None
                        and active_type != selected_type
                        and not requested_type_forced
                    ):
                        extraction_meta["auto_type_fallback_from"] = selected_type
                        extraction_meta["auto_type_fallback_to"] = active_type
                        selected_type = active_type
                        selected_item["type_original"] = selected_item.get("type")
                        selected_item["type"] = active_type
                        selected_item["type_source"] = "auto_extract_fallback"
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

@app.post("/v1/flux2/tryon")
async def vto_tryon_flux2(request: Flux2TryonRequest):
    """
    Performs Virtual Try-On using Vision-Guided Prompting (Flux 2.0).
    """
    t0 = time.time()
    stage_timings = {
        "download_user_image_s": 0.0,
        "download_products_s": 0.0,
        "product_prompt_generation_total_s": 0.0,
        "product_prompt_generation_per_item_s": [],
        "user_prompt_generation_s": 0.0,
        "board_build_s": 0.0,
        "prompt_build_s": 0.0,
        "flux_generation_sum_s": 0.0,
        "upload_s": 0.0,
    }
    try:
        if not request.products:
            raise HTTPException(status_code=422, detail="products must contain at least one item")

        user_image_url = str(request.user_image.tryonImage or "").strip()
        if not user_image_url:
            raise HTTPException(status_code=422, detail="user_image.tryonImage is required")

        product_urls: List[str] = []
        for idx, product in enumerate(request.products):
            product_url = str(product.image or "").strip()
            if not product_url:
                raise HTTPException(status_code=422, detail=f"products[{idx}].image is required")
            product_urls.append(product_url)

        # 1. Download Images
        t_stage = time.time()
        user_img = download_image(user_image_url)
        stage_timings["download_user_image_s"] = round(time.time() - t_stage, 4)

        t_stage = time.time()
        product_imgs = [download_image(url) for url in product_urls]
        stage_timings["download_products_s"] = round(time.time() - t_stage, 4)
        descriptor_backend = _normalize_descriptor_backend(request.description_backend)
        descriptor_compare_enabled = bool(request.description_compare or FLUX2_DESCRIPTOR_COMPARE)
        descriptor_comparisons = {"products": [], "user_image": {}} if descriptor_compare_enabled else None

        async with gpu_semaphore:
            # 2. Resolve product prompt descriptions (request-provided or selected model backend)
            product_descriptions: List[str] = []
            product_target_types: List[str] = []
            generated_product_prompt_indices: List[int] = []
            for idx, (product, product_img) in enumerate(zip(request.products, product_imgs)):
                provided_prompt = str(product.promptDescription or "").strip()
                resolved_target_type = "top"
                item_prompt_gen_s = 0.0
                if provided_prompt:
                    cleaned_desc = _sanitize_florence_garment_description(provided_prompt)
                else:
                    generated_product_prompt_indices.append(idx)
                    t_desc = time.time()
                    generated_desc = _describe_garment_with_backend(product_img, descriptor_backend)
                    item_prompt_gen_s += time.time() - t_desc
                    cleaned_desc = _sanitize_florence_garment_description(generated_desc)
                resolved_target_type = _infer_flux2_target_type(cleaned_desc)
                if (
                    not provided_prompt
                    and descriptor_backend == "qwen2_5_vl"
                    and FLUX2_QWEN_EXTRA_CLASSIFY_PASS
                ):
                    try:
                        t_desc = time.time()
                        qwen_type = engine.qwen25vl.classify_garment_type(product_img)
                        item_prompt_gen_s += time.time() - t_desc
                        if qwen_type in {"dress", "top", "bottom", "outer"}:
                            resolved_target_type = qwen_type
                    except Exception as type_err:
                        logger.warning(f"Qwen garment type classification failed for products[{idx}]: {type_err}")
                stage_timings["product_prompt_generation_per_item_s"].append(round(item_prompt_gen_s, 4))
                stage_timings["product_prompt_generation_total_s"] += item_prompt_gen_s
                cleaned_desc = _ensure_target_type_in_description(cleaned_desc, resolved_target_type)
                product_descriptions.append(cleaned_desc)
                product_target_types.append(resolved_target_type)
                if descriptor_compare_enabled and not provided_prompt:
                    compare_item = {}
                    for backend_name in ("florence", "qwen2_5_vl"):
                        try:
                            if backend_name == descriptor_backend:
                                compare_item[backend_name] = cleaned_desc
                            else:
                                alt_desc = _describe_garment_with_backend(product_img, backend_name)
                                compare_item[backend_name] = _sanitize_florence_garment_description(alt_desc)
                        except Exception as cmp_err:
                            compare_item[backend_name] = f"[error] {cmp_err}"
                    descriptor_comparisons["products"].append(compare_item)
                elif descriptor_compare_enabled:
                    descriptor_comparisons["products"].append({
                        "provided": cleaned_desc,
                    })

            # 3. Resolve user prompt description for stronger preservation constraints
            user_prompt_raw = str(request.user_image.promptDescription or "").strip()
            user_prompt_generated = False
            if not user_prompt_raw:
                t_desc = time.time()
                user_prompt_raw = _describe_user_image_for_flux2(user_img, backend=descriptor_backend)
                stage_timings["user_prompt_generation_s"] = round(time.time() - t_desc, 4)
                user_prompt_generated = True
            if descriptor_compare_enabled and user_prompt_generated:
                compare_user = {}
                for backend_name in ("florence", "qwen2_5_vl"):
                    try:
                        if backend_name == descriptor_backend:
                            compare_user[backend_name] = user_prompt_raw
                        else:
                            compare_user[backend_name] = _describe_user_image_for_flux2(user_img, backend=backend_name)
                    except Exception as cmp_err:
                        compare_user[backend_name] = f"[error] {cmp_err}"
                descriptor_comparisons["user_image"] = compare_user
            elif descriptor_compare_enabled:
                descriptor_comparisons["user_image"] = {"provided": user_prompt_raw}

            user_prompt_description = user_prompt_raw
            user_prompt_description = _augment_identity_lock(user_prompt_description)

            # Keep Qwen warm by default to avoid per-request cold starts.
            # Enable FLUX2_UNLOAD_QWEN_BEFORE_FLUX2=1 only when VRAM pressure requires it.
            if (
                FLUX2_UNLOAD_QWEN_BEFORE_FLUX2
                and descriptor_backend == "qwen2_5_vl"
                and getattr(engine.qwen25vl, "device", "cpu") == "cuda"
                and engine.qwen25vl.is_loaded
            ):
                engine.qwen25vl.unload()

            # 4. Build board only for multi-product try-on
            t_stage = time.time()
            if len(product_imgs) > 1:
                board = engine.board_builder.build_board(product_imgs)
                board_mode = "collage"
            else:
                board = product_imgs[0]
                board_mode = "single"
            stage_timings["board_build_s"] = round(time.time() - t_stage, 4)
            visual_locks = _build_flux2_visual_lock_clauses(
                product_images=product_imgs,
                garment_descriptions=product_descriptions,
            )

            # 5. Build flux2-only target-aware prompt (dress = replacement, not layering)
            t_stage = time.time()
            prompt = _build_flux2_targeted_prompt(
                garment_descriptions=product_descriptions,
                user_description=user_prompt_description,
                target_types=product_target_types,
                board_mode=board_mode,
                color_lock_clause=str(visual_locks.get("color_clause") or ""),
                detail_lock_clause=str(visual_locks.get("detail_clause") or ""),
                transparency_lock_clause=str(visual_locks.get("transparency_clause") or ""),
            )
            stage_timings["prompt_build_s"] = round(time.time() - t_stage, 4)
            target_desc = " | ".join(product_descriptions)
            candidate_runs: List[dict] = []

            def _run_candidate(candidate_prompt: str, candidate_steps: int, candidate_seed: int, label: str):
                candidate_result = engine.flux2.run_tryon(
                    person_image=user_img,
                    board_image=board,
                    prompt=candidate_prompt,
                    steps=candidate_steps,
                    seed=candidate_seed,
                )
                fidelity_score, output_desc = _score_tryon_garment_fidelity(
                    output_image=candidate_result["image"],
                    target_description=target_desc,
                    descriptor_backend=FLUX2_FIDELITY_BACKEND,
                )
                candidate_runs.append({
                    "label": label,
                    "steps": candidate_steps,
                    "seed": candidate_seed,
                    "latency": float(candidate_result["latency"]),
                    "fidelity_score": float(fidelity_score),
                    "output_description": output_desc,
                })
                return candidate_result

            is_single_dress = (
                board_mode == "single"
                and len(product_target_types) == 1
                and product_target_types[0] == "dress"
            )
            is_single_item = board_mode == "single" and len(product_target_types) == 1
            dress_strict_prompt = (
                prompt
                + " Source product image may include mannequin/body; transfer only the garment piece, never mannequin skin/body parts."
                + " Match exact neckline, bodice shape, and ruffle distribution from the source garment."
                + " Lock lower section fidelity: preserve exact hemline shape, flare profile, skirt volume, "
                + "and bottom-edge asymmetry/high-low geometry from image 2."
            )
            dress_strict_steps = min(
                FLUX2_DRESS_SECOND_PASS_MAX_STEPS,
                max(request.steps, request.steps + FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS),
            )
            dress_strict_seed = min(2147483647, request.seed + 1)

            qwen_strict_prompt = (
                prompt
                + " Enforce strict garment fidelity from image 2 only."
                + " Keep exact print placement, texture, seams, neckline, sleeve shape, and hemline."
                + " Avoid any mixed-layer artifacts from original clothes in image 1."
            )
            if product_target_types and product_target_types[0] == "dress":
                qwen_strict_prompt += (
                    " This is a dress replacement: remove prior top and bottom garments completely, no overlap."
                )
            qwen_strict_steps = min(
                FLUX2_QWEN_SECOND_PASS_MAX_STEPS,
                max(request.steps, request.steps + FLUX2_QWEN_SECOND_PASS_EXTRA_STEPS, FLUX2_QWEN_MIN_STEPS),
            )
            qwen_strict_seed = min(2147483647, request.seed + 11)

            selected_prompt = prompt
            selected_candidate_index = 0
            mode = FLUX2_SINGLE_CANDIDATE_MODE

            # 6. Inference (Flux 2.0): either single selected candidate or auto multi-candidate
            if mode == "base":
                result = _run_candidate(
                    candidate_prompt=prompt,
                    candidate_steps=request.steps,
                    candidate_seed=request.seed,
                    label="base",
                )
            elif mode == "dress_strict" and is_single_dress:
                result = _run_candidate(
                    candidate_prompt=dress_strict_prompt,
                    candidate_steps=dress_strict_steps,
                    candidate_seed=dress_strict_seed,
                    label="dress_strict",
                )
                selected_prompt = dress_strict_prompt
            elif (
                mode == "qwen_strict"
                and is_single_item
                and descriptor_backend == "qwen2_5_vl"
            ):
                result = _run_candidate(
                    candidate_prompt=qwen_strict_prompt,
                    candidate_steps=qwen_strict_steps,
                    candidate_seed=qwen_strict_seed,
                    label="qwen_strict",
                )
                selected_prompt = qwen_strict_prompt
            else:
                # Fallback to existing auto-selection behavior.
                if mode != "auto":
                    logger.warning(
                        "Requested FLUX2_SINGLE_CANDIDATE_MODE=%s is not applicable for this request; using auto mode.",
                        mode,
                    )

                result = _run_candidate(
                    candidate_prompt=prompt,
                    candidate_steps=request.steps,
                    candidate_seed=request.seed,
                    label="base",
                )

                if is_single_dress and FLUX2_DRESS_SECOND_PASS_ENABLED:
                    strict_result = _run_candidate(
                        candidate_prompt=dress_strict_prompt,
                        candidate_steps=dress_strict_steps,
                        candidate_seed=dress_strict_seed,
                        label="dress_strict",
                    )

                    base_score = float(candidate_runs[0]["fidelity_score"])
                    strict_score = float(candidate_runs[1]["fidelity_score"])
                    if strict_score >= base_score:
                        result = strict_result
                        selected_candidate_index = 1
                        selected_prompt = dress_strict_prompt

                if (
                    is_single_item
                    and descriptor_backend == "qwen2_5_vl"
                    and FLUX2_QWEN_SECOND_PASS_ENABLED
                ):
                    qwen_strict_result = _run_candidate(
                        candidate_prompt=qwen_strict_prompt,
                        candidate_steps=qwen_strict_steps,
                        candidate_seed=qwen_strict_seed,
                        label="qwen_strict",
                    )
                    qwen_index = len(candidate_runs) - 1
                    current_best_score = float(candidate_runs[selected_candidate_index]["fidelity_score"])
                    qwen_score = float(candidate_runs[qwen_index]["fidelity_score"])
                    if qwen_score >= current_best_score:
                        result = qwen_strict_result
                        selected_candidate_index = qwen_index
                        selected_prompt = qwen_strict_prompt

            # 7. Upload selected result
            stage_timings["flux_generation_sum_s"] = round(
                sum(float(run.get("latency", 0.0)) for run in candidate_runs), 4
            )
            t_stage = time.time()
            res_buf = io.BytesIO()
            result["image"].save(res_buf, format="PNG")
            result_url = _upload_or_raise(res_buf.getvalue(), container=VTO_OUTPUT_CONTAINER)
            stage_timings["upload_s"] = round(time.time() - t_stage, 4)

        total_latency = time.time() - t0
        stage_timings["product_prompt_generation_total_s"] = round(
            float(stage_timings["product_prompt_generation_total_s"]), 4
        )
        stage_timings["api_total_s"] = round(total_latency, 4)
        logger.info(
            "flux2_tryon_timing "
            f"total={stage_timings['api_total_s']}s "
            f"qwen_products={stage_timings['product_prompt_generation_total_s']}s "
            f"qwen_user={stage_timings['user_prompt_generation_s']}s "
            f"flux_sum={stage_timings['flux_generation_sum_s']}s "
            f"upload={stage_timings['upload_s']}s "
            f"backend={descriptor_backend} "
            f"product_count={len(product_descriptions)} "
            f"user_prompt_generated={user_prompt_generated}"
        )

        return {
            "status": "success",
            "result_url": result_url,
            "promptDescription": " | ".join(product_descriptions),
            "productPromptDescriptions": product_descriptions,
            "userPromptDescription": user_prompt_description,
            "productColorHints": visual_locks.get("color_hints", []),
            "productDetailHints": visual_locks.get("detail_terms", []),
            "transparencyLockApplied": bool(visual_locks.get("transparency_lock")),
            "prompt": selected_prompt,
            "latency": result["latency"],
            "total_latency": total_latency,
            "boardMode": board_mode,
            "productTargetTypes": product_target_types,
            "descriptionBackend": descriptor_backend,
            "fidelityBackend": FLUX2_FIDELITY_BACKEND,
            "descriptionCompareEnabled": descriptor_compare_enabled,
            "descriptionComparisons": descriptor_comparisons,
            "singleCandidateMode": FLUX2_SINGLE_CANDIDATE_MODE,
            "singleCandidateApplied": len(candidate_runs) == 1,
            "stageTimings": stage_timings,
            "candidateRuns": candidate_runs,
            "selectedCandidateIndex": selected_candidate_index,
            "generatedProductPromptIndices": generated_product_prompt_indices,
            "userPromptGenerated": user_prompt_generated,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Try-on failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/flux/tryon")
async def vto_tryon_flux(request: VTORequest):
    """
    Legacy Flux try-on route kept separate from /v1/flux2/tryon so
    flux2-specific changes do not affect other APIs.
    """
    t0 = time.time()
    try:
        # 1. Download Images
        user_img = download_image(request.user_image_url)
        garment_img = download_image(request.garment_image_url)

        async with gpu_semaphore:
            # 2. Vision Integration (Florence-2)
            if USE_FLORENCE_DETAILED_PROMPT:
                garment_desc = engine.florence.describe_garment(garment_img)
            else:
                garment_desc = engine.florence.describe_garment_short(garment_img)

            # 3. Build Board (Collage)
            board = engine.board_builder.build_board([garment_img])

            # 4. Defensive Prompting
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
                "flux2_descriptor_backend": FLUX2_DESCRIPTOR_BACKEND,
                "flux2_fidelity_backend": FLUX2_FIDELITY_BACKEND,
                "flux2_descriptor_compare": FLUX2_DESCRIPTOR_COMPARE,
                "flux2_dress_second_pass_enabled": FLUX2_DRESS_SECOND_PASS_ENABLED,
                "flux2_dress_second_pass_extra_steps": FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS,
                "flux2_dress_second_pass_max_steps": FLUX2_DRESS_SECOND_PASS_MAX_STEPS,
                "flux2_qwen_second_pass_enabled": FLUX2_QWEN_SECOND_PASS_ENABLED,
                "flux2_qwen_second_pass_extra_steps": FLUX2_QWEN_SECOND_PASS_EXTRA_STEPS,
                "flux2_qwen_second_pass_max_steps": FLUX2_QWEN_SECOND_PASS_MAX_STEPS,
                "flux2_qwen_min_steps": FLUX2_QWEN_MIN_STEPS,
                "flux2_preload_qwen_with_flux2": FLUX2_PRELOAD_QWEN_WITH_FLUX2,
                "flux2_qwen_extra_classify_pass": FLUX2_QWEN_EXTRA_CLASSIFY_PASS,
                "flux2_unload_qwen_before_flux2": FLUX2_UNLOAD_QWEN_BEFORE_FLUX2,
                "flux2_qwen_product_caption_max_side": FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE,
                "flux2_qwen_product_caption_min_side": FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE,
                "flux2_qwen_user_caption_max_side": FLUX2_QWEN_USER_CAPTION_MAX_SIDE,
                "flux2_qwen_user_caption_min_side": FLUX2_QWEN_USER_CAPTION_MIN_SIDE,
                "flux2_preload_joycaption_with_flux2": FLUX2_PRELOAD_JOYCAPTION_WITH_FLUX2,
                "flux2_single_candidate_mode": FLUX2_SINGLE_CANDIDATE_MODE,
                "flux2_color_lock_enabled": FLUX2_COLOR_LOCK_ENABLED,
                "flux2_color_lock_top_k": FLUX2_COLOR_LOCK_TOP_K,
                "flux2_detail_lock_enabled": FLUX2_DETAIL_LOCK_ENABLED,
                "analyze_extract_cloth": ANALYZE_EXTRACT_CLOTH,
            "analyze_extract_mode": "forced_vton",
            "analyze_extract_parser_only": ANALYZE_EXTRACT_PARSER_ONLY,
            "analyze_require_extracted_prompt": ANALYZE_REQUIRE_EXTRACTED_PROMPT,
            "analyze_require_mirrored_vton_url": ANALYZE_REQUIRE_MIRRORED_VTON_URL,
            "analyze_vton_fallback_enabled": ANALYZE_VTON_FALLBACK_ENABLED,
            "analyze_vton_segmentation_free": ANALYZE_VTON_SEGMENTATION_FREE,
            "analyze_vton_reject_source_passthrough": ANALYZE_VTON_REJECT_SOURCE_PASSTHROUGH,
            "analyze_vton_strict_safety_checks": ANALYZE_VTON_STRICT_SAFETY_CHECKS,
            "analyze_vton_mirror_raw_output": ANALYZE_VTON_MIRROR_RAW_OUTPUT,
            "analyze_vton_use_showroom_person": ANALYZE_VTON_USE_SHOWROOM_PERSON,
            "analyze_vton_dress_use_full_image": ANALYZE_VTON_DRESS_USE_FULL_IMAGE,
            "analyze_vton_single_item_use_full_image": ANALYZE_VTON_SINGLE_ITEM_USE_FULL_IMAGE,
            "analyze_vton_num_timesteps": ANALYZE_VTON_NUM_TIMESTEPS,
            "analyze_vton_guidance_scale": ANALYZE_VTON_GUIDANCE_SCALE,
            "analyze_vton_seed": ANALYZE_VTON_SEED,
            "analyze_vton_crop_top_extra_ratio_bottom": ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM,
            "analyze_vton_crop_top_extra_ratio_bottom_multi": ANALYZE_VTON_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI,
            "analyze_use_parser_for_prerouting": ANALYZE_USE_PARSER_FOR_PREROUTING,
            "analyze_florence_dress_lock_min_score": ANALYZE_FLORENCE_DRESS_LOCK_MIN_SCORE,
            "analyze_tighten_bottom_top_max_overlap_px": ANALYZE_TIGHTEN_BOTTOM_TOP_MAX_OVERLAP_PX,
            "analyze_tighten_bottom_max_down_shift_ratio": ANALYZE_TIGHTEN_BOTTOM_MAX_DOWN_SHIFT_RATIO,
            "analyze_tighten_bottom_max_gap_from_top_px": ANALYZE_TIGHTEN_BOTTOM_MAX_GAP_FROM_TOP_PX,
            "analyze_tighten_bottom_min_width_ratio": ANALYZE_TIGHTEN_BOTTOM_MIN_WIDTH_RATIO,
            "analyze_force_fullbody_split_on_same_type": ANALYZE_FORCE_FULLBODY_SPLIT_ON_SAME_TYPE,
            "analyze_force_fullbody_split_min_height_ratio": ANALYZE_FORCE_FULLBODY_SPLIT_MIN_HEIGHT_RATIO,
            "analyze_collapse_same_type": ANALYZE_COLLAPSE_SAME_TYPE,
            "analyze_collapse_same_type_min_iou": ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU,
            "analyze_auto_select_multi_dress": ANALYZE_AUTO_SELECT_MULTI_DRESS,
            "analyze_extract_max_body_ratio": ANALYZE_EXTRACT_MAX_BODY_RATIO,
            "analyze_extract_body_strip_dilate": ANALYZE_EXTRACT_BODY_STRIP_DILATE,
            "analyze_extract_body_strip_max_ratio": ANALYZE_EXTRACT_BODY_STRIP_MAX_RATIO,
            "analyze_showroom_delta_refinement_enabled": ANALYZE_SHOWROOM_DELTA_REFINEMENT_ENABLED,
            "analyze_showroom_delta_threshold": ANALYZE_SHOWROOM_DELTA_THRESHOLD,
            "analyze_showroom_delta_skin_extra": ANALYZE_SHOWROOM_DELTA_SKIN_EXTRA,
            "analyze_showroom_delta_dilate": ANALYZE_SHOWROOM_DELTA_DILATE,
            "analyze_showroom_delta_min_retain_ratio": ANALYZE_SHOWROOM_DELTA_MIN_RETAIN_RATIO,
            "analyze_top_recover_max_skin_ratio": ANALYZE_TOP_RECOVER_MAX_SKIN_RATIO,
            "analyze_top_recover_near_dilate": ANALYZE_TOP_RECOVER_NEAR_DILATE,
            "analyze_top_skin_rim_cleanup": ANALYZE_TOP_SKIN_RIM_CLEANUP,
            "analyze_top_skin_rim_max_ratio": ANALYZE_TOP_SKIN_RIM_MAX_RATIO,
            "analyze_vton_max_showroom_mae": ANALYZE_VTON_MAX_SHOWROOM_MAE,
            "analyze_vton_min_showroom_change_ratio": ANALYZE_VTON_MIN_SHOWROOM_CHANGE_RATIO,
            "analyze_vton_top_zoom_enabled": ANALYZE_VTON_TOP_ZOOM_ENABLED,
            "analyze_vton_bottom_upscale_enabled": ANALYZE_VTON_BOTTOM_UPSCALE_ENABLED,
            "analyze_garment_postprocess_enabled": ANALYZE_GARMENT_POSTPROCESS_ENABLED,
            "analyze_garment_target_aspect": f"{ANALYZE_GARMENT_TARGET_ASPECT_W}:{ANALYZE_GARMENT_TARGET_ASPECT_H}",
            "analyze_garment_enhance_enabled": ANALYZE_GARMENT_ENHANCE_ENABLED,
            "analyze_garment_output_background": ANALYZE_GARMENT_OUTPUT_BACKGROUND,
            "wardrobe_progress_sync_enabled": ENABLE_WARDROBE_PROGRESS_SYNC,
            "wardrobe_progress_include_input_image": WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE,
        },
        "models": engine.model_status(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
