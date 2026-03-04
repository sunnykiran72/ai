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
from typing import Optional, List, Tuple, Dict

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
from ai.core.minicpm_runner import MiniCPMVRunner
from ai.core.yolo_runner import YoloRunner
from ai.core.human_parser_runner import HumanParserRunner
from ai.core.openclip_runner import OpenCLIPRunner

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
FLUX2_ALLOW_QWEN_BACKEND = os.getenv("FLUX2_ALLOW_QWEN_BACKEND", "0") == "1"
FLUX2_DESCRIPTOR_BACKEND = os.getenv("FLUX2_DESCRIPTOR_BACKEND", "minicpm_service").strip().lower()
if FLUX2_DESCRIPTOR_BACKEND not in {"florence", "qwen2_5_vl", "joycaption", "minicpm", "minicpm_service"}:
    FLUX2_DESCRIPTOR_BACKEND = "minicpm_service"
if FLUX2_DESCRIPTOR_BACKEND == "qwen2_5_vl" and not FLUX2_ALLOW_QWEN_BACKEND:
    FLUX2_DESCRIPTOR_BACKEND = "minicpm_service"
FLUX2_FIDELITY_BACKEND = os.getenv("FLUX2_FIDELITY_BACKEND", "florence").strip().lower()
if FLUX2_FIDELITY_BACKEND not in {"florence", "qwen2_5_vl"}:
    FLUX2_FIDELITY_BACKEND = "florence"
if FLUX2_FIDELITY_BACKEND == "qwen2_5_vl" and not FLUX2_ALLOW_QWEN_BACKEND:
    FLUX2_FIDELITY_BACKEND = "florence"
FLUX2_DESCRIPTOR_COMPARE = os.getenv("FLUX2_DESCRIPTOR_COMPARE", "0") == "1"
FLUX2_NEGATIVE_PROMPT_ENABLE = os.getenv("FLUX2_NEGATIVE_PROMPT_ENABLE", "1") == "1"
FLUX2_NEGATIVE_PROMPT_DEFAULT = os.getenv(
    "FLUX2_NEGATIVE_PROMPT_DEFAULT",
    (
        "low quality, blurry, deformed body, extra limbs, extra fingers, wrong hands, "
        "identity change, different face, wrong skin tone, recolored garment, hue shift, "
        "color drift, pattern drift, texture swap, logo/text watermark, duplicate garment, "
        "layering artifacts, garment merge, ghost garment, incorrect neckline, incorrect hemline"
    ),
).strip()
FLUX2_DRESS_SECOND_PASS_ENABLED = os.getenv("FLUX2_DRESS_SECOND_PASS_ENABLED", "1") == "1"
FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS = max(1, _env_int("FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS", 4))
FLUX2_DRESS_SECOND_PASS_MAX_STEPS = max(6, _env_int("FLUX2_DRESS_SECOND_PASS_MAX_STEPS", 18))
FLUX2_REGION_LOCK_SECOND_PASS_ENABLED = os.getenv("FLUX2_REGION_LOCK_SECOND_PASS_ENABLED", "1") == "1"
FLUX2_REGION_LOCK_SECOND_PASS_EXTRA_STEPS = max(1, _env_int("FLUX2_REGION_LOCK_SECOND_PASS_EXTRA_STEPS", 2))
FLUX2_REGION_LOCK_SECOND_PASS_MAX_STEPS = max(6, _env_int("FLUX2_REGION_LOCK_SECOND_PASS_MAX_STEPS", 16))
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
FLUX2_PRELOAD_MINICPM_WITH_FLUX2 = os.getenv("FLUX2_PRELOAD_MINICPM_WITH_FLUX2", "0") == "1"
FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE = max(512, _env_int("FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE", 1024))
FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE = max(256, _env_int("FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE", 512))
FLUX2_MINICPM_USER_CAPTION_MAX_SIDE = max(512, _env_int("FLUX2_MINICPM_USER_CAPTION_MAX_SIDE", 1024))
FLUX2_MINICPM_USER_CAPTION_MIN_SIDE = max(256, _env_int("FLUX2_MINICPM_USER_CAPTION_MIN_SIDE", 512))
MINICPM_SERVICE_URL = os.getenv("MINICPM_SERVICE_URL", "http://127.0.0.1:8010").strip().rstrip("/")
MINICPM_SERVICE_TIMEOUT_S = max(5, _env_int("MINICPM_SERVICE_TIMEOUT_S", 120))
MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS = max(32, _env_int("MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS", 260))
MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS = max(32, _env_int("MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS", 260))
MINICPM_SERVICE_GARMENT_PROMPT = os.getenv(
    "MINICPM_SERVICE_GARMENT_PROMPT",
    (
        "Describe only the product garment for high-fidelity virtual try-on. "
        "Return one detailed line with schema: "
        "category=<dress|top|bottom|outerwear|set|unknown>; "
        "type=<specific garment type>; "
        "colors=<primary, secondary>; "
        "pattern=<solid|striped|floral|graphic|etc>; "
        "material=<fabric/material>; "
        "silhouette=<fit and shape>; "
        "construction=<neckline, sleeve style/length, waist shaping, hem/length>; "
        "details=<buttons, zipper, pleats, ruffles, lace, embroidery, pockets, slit, logo>; "
        "coverage=<body area to replace>; "
        "preserve=<color, print placement, and garment structure must remain unchanged>. "
        "Use unknown when not visible."
    ),
).strip()
MINICPM_SERVICE_PERSON_PROMPT = os.getenv(
    "MINICPM_SERVICE_PERSON_PROMPT",
    (
        "Describe person context for identity/scene preservation in virtual try-on. "
        "Return one detailed line with schema: "
        "identity=<face traits, skin tone, hair style/color, age band>; "
        "body_pose=<pose, camera angle, visible limbs>; "
        "framing_lighting=<framing/crop, light direction/intensity>; "
        "current_outfit=<top, bottom, footwear, accessories>; "
        "occlusion=<hair/hands/bags/objects overlapping garment region>; "
        "preserve=<face identity, skin tone, hair, body proportions, pose, background unchanged>. "
        "Use unknown when not visible."
    ),
).strip()
FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE = os.getenv(
    "FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE",
    "request_or_auto",
).strip().lower()
if FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE not in {"disabled", "request_only", "request_or_auto", "auto_only"}:
    FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE = "request_or_auto"
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
PARSER_FUSION_V1_ENABLED = os.getenv("PARSER_FUSION_V1_ENABLED", "1") == "1"
PARSER_FUSION_USE_YOLO_SUPPORT = os.getenv("PARSER_FUSION_USE_YOLO_SUPPORT", "1") == "1"
PARSER_FUSION_USE_OPENCLIP = os.getenv("PARSER_FUSION_USE_OPENCLIP", "1") == "1"
PARSER_FUSION_WEIGHT_PARSER = _env_float("PARSER_FUSION_WEIGHT_PARSER", 0.40)
PARSER_FUSION_WEIGHT_YOLO = _env_float("PARSER_FUSION_WEIGHT_YOLO", 0.20)
PARSER_FUSION_WEIGHT_OPENCLIP = _env_float("PARSER_FUSION_WEIGHT_OPENCLIP", 0.40)
PARSER_FUSION_MIN_SCORE_KEEP = _env_float("PARSER_FUSION_MIN_SCORE_KEEP", 0.24)
PARSER_FUSION_AMBIGUITY_GAP = _env_float("PARSER_FUSION_AMBIGUITY_GAP", 0.08)
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
        self.minicpm = MiniCPMVRunner()
        self.openclip = OpenCLIPRunner()
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
            if FLUX2_PRELOAD_MINICPM_WITH_FLUX2:
                self.minicpm.ensure_ready()
        elif FLUX2_DESCRIPTOR_BACKEND == "qwen2_5_vl":
            if FLUX2_PRELOAD_QWEN_WITH_FLUX2:
                self.qwen25vl.ensure_ready()
        elif FLUX2_DESCRIPTOR_BACKEND == "joycaption":
            if FLUX2_PRELOAD_JOYCAPTION_WITH_FLUX2:
                self.joycaption.ensure_ready()
        elif FLUX2_DESCRIPTOR_BACKEND == "minicpm":
            if FLUX2_PRELOAD_MINICPM_WITH_FLUX2:
                self.minicpm.ensure_ready()
        elif FLUX2_DESCRIPTOR_BACKEND == "minicpm_service":
            # External MiniCPM service handles descriptor generation.
            # Keep local descriptor runners unloaded for better VRAM headroom.
            pass
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
            "minicpm_loaded": self.minicpm.is_loaded,
            "openclip_loaded": self.openclip.is_loaded,
            "openclip_available": self.openclip.is_available,
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


def _parse_structured_descriptor(text: str) -> Dict[str, str]:
    """
    Parse lightweight key-value descriptors from VLM output lines.
    Supports: key=value, key: value, key[value], key=<value>
    """
    src = str(text or "").strip()
    if not src:
        return {}

    # Remove lightweight markdown formatting often returned by VLMs.
    src = src.replace("**", "").replace("`", "")
    src = re.sub(r"\s+", " ", src).strip()

    # Inject separators before known keys when model returns a single stream like:
    # "Category: dress Type: evening gown Colors: beige ..."
    known_labels = [
        "category", "type", "colors", "pattern", "material", "silhouette",
        "construction", "details", "coverage", "preserve",
        "identity", "body pose", "body_pose", "by pose", "by_pose",
        "framing lighting", "framing_lighting", "current outfit", "current_outfit",
        "occlusion",
    ]
    for label in sorted(known_labels, key=len, reverse=True):
        pattern = rf"(?i)\b{re.escape(label)}\b\s*:"
        src = re.sub(pattern, f"; {label}:", src)
    src = src.lstrip("; ").strip()

    parsed: Dict[str, str] = {}
    segments = [seg.strip() for seg in re.split(r"[;|]\s*", src) if seg.strip()]
    for seg in segments:
        m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_\- ]{0,40})\s*(?:=|:)\s*(.+?)\s*$", seg)
        if not m:
            m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_\- ]{0,40})\s*\[(.+?)\]\s*$", seg)
        if not m:
            continue
        raw_key = m.group(1).strip().lower()
        raw_key = raw_key.replace("/", "_").replace("-", "_").replace(" ", "_")
        key_aliases = {
            "bodypose": "body_pose",
            "body_pose": "body_pose",
            "by_pose": "body_pose",
            "bypose": "body_pose",
            "framinglighting": "framing_lighting",
            "framing_lighting": "framing_lighting",
            "currentoutfit": "current_outfit",
            "current_outfit": "current_outfit",
        }
        raw_key = key_aliases.get(raw_key, raw_key)
        raw_val = m.group(2).strip()
        raw_val = raw_val.strip("<>[](){} \t\r\n")
        # Drop accidental markdown/list punctuation wrapping.
        raw_val = raw_val.strip("*- ")
        if raw_key and raw_val:
            parsed[raw_key] = raw_val
    return parsed


def _normalize_minicpm_descriptor_text(raw_text: str, kind: str) -> str:
    """
    Convert structured MiniCPM descriptor into a concise, Flux-friendly sentence.
    Keeps key fidelity terms (color/pattern/structure/details/identity).
    """
    text = " ".join(str(raw_text or "").split()).strip()
    if not text:
        return text

    fields = _parse_structured_descriptor(text)
    if not fields:
        return text

    if kind == "person":
        identity = fields.get("identity", "")
        pose = fields.get("body_pose", "") or fields.get("by_pose", "")
        framing = fields.get("framing_lighting", "")
        outfit = fields.get("current_outfit", "")
        occlusion = fields.get("occlusion", "")
        preserve = fields.get("preserve", "")
        parts = []
        if identity:
            parts.append(f"identity: {identity}")
        if pose:
            parts.append(f"pose: {pose}")
        if framing:
            parts.append(f"framing/lighting: {framing}")
        if outfit:
            parts.append(f"current outfit: {outfit}")
        if occlusion and occlusion.lower() not in {"none", "no", "n/a"}:
            parts.append(f"occlusion: {occlusion}")
        if preserve:
            parts.append(f"preserve: {preserve}")
        return ". ".join(parts).strip(" .") or text

    category = fields.get("category", "")
    gtype = fields.get("type", "")
    colors = fields.get("colors", "")
    pattern = fields.get("pattern", "")
    material = fields.get("material", "")
    silhouette = fields.get("silhouette", "")
    construction = fields.get("construction", "")
    details = fields.get("details", "")
    coverage = fields.get("coverage", "")
    preserve = fields.get("preserve", "")
    parts = []
    if category:
        parts.append(f"{category} garment")
    if gtype:
        parts.append(f"type {gtype}")
    if colors:
        parts.append(f"colors {colors}")
    if pattern:
        parts.append(f"pattern {pattern}")
    if material:
        parts.append(f"material {material}")
    if silhouette:
        parts.append(f"silhouette {silhouette}")
    if construction:
        parts.append(f"construction {construction}")
    if details and details.lower() not in {"none", "no visible details", "n/a"}:
        parts.append(f"details {details}")
    if coverage:
        parts.append(f"coverage {coverage}")
    if preserve:
        parts.append(f"preserve {preserve}")
    return ", ".join(parts).strip(" ,.") or text

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

    has_dress_term = any(term in text for term in dress_terms)
    has_bottom_term = any(term in text for term in bottom_terms)
    has_outer_term = any(term in text for term in outer_terms)
    has_top_term = any(term in text for term in top_terms)

    # Be conservative with dress detection for model-generated captions.
    # If top/bottom/outer cues co-exist, prefer region-specific replacement.
    if has_dress_term and not (has_top_term or has_bottom_term or has_outer_term):
        return "dress"
    if has_bottom_term:
        return "bottom"
    if has_outer_term:
        return "outer"
    if has_top_term:
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

def _extract_dominant_hex_colors(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    top_k: int = 4,
) -> List[str]:
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return []

    pixels = arr.reshape(-1, 3)
    if isinstance(mask, np.ndarray):
        m = np.asarray(mask).astype(bool)
        if m.shape[:2] == arr.shape[:2]:
            keep = m.reshape(-1)
            if int(np.sum(keep)) > 16:
                pixels = pixels[keep]

    if pixels.size == 0:
        return []
    near_white = np.all(pixels >= 245, axis=1)
    if int(np.sum(~near_white)) > 0:
        pixels = pixels[~near_white]
    if pixels.size == 0:
        return []

    binned = (pixels // 16) * 16
    unique, counts = np.unique(binned, axis=0, return_counts=True)
    order = np.argsort(-counts)
    out: List[str] = []
    for idx in order.tolist():
        r, g, b = [int(v) for v in unique[idx].tolist()]
        hx = f"#{r:02X}{g:02X}{b:02X}"
        if hx not in out:
            out.append(hx)
        if len(out) >= max(1, int(top_k)):
            break
    return out

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
    collage_item_clause: str = "",
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
            "Do not modify face, hair, hands, skin texture, body shape, pose, camera framing, or background. "
        )
        if types == {"top"}:
            prompt += (
                "Top-only lock: keep lower-body garments unchanged (pants/skirt/shorts/shoes), "
                "with original silhouette, color, and texture from image 1. "
                "Do not edit anything below the natural waistline except minor occlusion cleanup. "
                "Preserve lower-body garment pixels and folds as in image 1. "
            )
        if types == {"bottom"}:
            prompt += (
                "Bottom-only lock: keep upper-body garments unchanged (top/outerwear), "
                "with original silhouette, color, and texture from image 1. "
                "Do not edit anything above the natural waistline except minor occlusion cleanup. "
                "Preserve upper-body garment pixels and folds as in image 1. "
            )

    if is_multi:
        prompt += (
            "Image 2 is a multi-item outfit board; apply all listed items together with coherent layering and fit. "
            "Keep each item isolated: no cross-item color bleed, no print transfer, and no texture mixing between items. "
        )
        if collage_item_clause:
            prompt += collage_item_clause

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


def _board_panel_label(idx: int, total: int) -> str:
    if total <= 1:
        return "full panel"
    if total == 2:
        return "top panel" if idx == 0 else "bottom panel"
    if idx == 0:
        return "left-top panel"
    if idx == 1:
        return "left-bottom panel"
    return f"right-column panel {idx - 1}"


def _trim_prompt_fragment(text: str, max_chars: int = 120) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _build_flux2_collage_item_clause(
    garment_descriptions: List[str],
    target_types: List[str],
) -> str:
    total = len(garment_descriptions)
    if total <= 1:
        return ""
    segments: List[str] = []
    for idx in range(total):
        slot = _board_panel_label(idx, total)
        g_type = _normalize_garment_type(target_types[idx] if idx < len(target_types) else "") or "item"
        desc = _trim_prompt_fragment(garment_descriptions[idx], max_chars=110)
        segments.append(f"item {idx + 1} ({g_type}) from {slot}: {desc}")
    return (
        "Exact board mapping in image 2: "
        + "; ".join(segments)
        + ". Apply each mapped item only to its intended body region. "
    )


def _build_flux2_negative_prompt(
    *,
    target_types: List[str],
    board_mode: str,
    custom_negative_prompt: str = "",
) -> str:
    parts: List[str] = []
    custom = " ".join(str(custom_negative_prompt or "").split()).strip()
    if custom:
        parts.append(custom)
    elif FLUX2_NEGATIVE_PROMPT_ENABLE and FLUX2_NEGATIVE_PROMPT_DEFAULT:
        parts.append(FLUX2_NEGATIVE_PROMPT_DEFAULT)

    types = {str(t or "").strip().lower() for t in target_types if str(t or "").strip()}
    auto_terms = (
        "wrong garment color, recolored fabric, hue shift, saturation drift, color family change, "
        "pattern drift, print swap, texture swap, wrong material, missing trims, wrong seams, "
        "incorrect neckline, incorrect sleeve length, incorrect hem length, extra garment, duplicate garment, "
        "floating cloth, ghost cloth, broken folds, low detail fabric, blurry edges, watermark, text, logo, "
        "face swap, identity drift, changed facial features, altered skin tone, altered hair style, changed body shape"
    )
    parts.append(auto_terms)

    if board_mode == "collage" and len(target_types) > 1:
        parts.append(
            "cross-item color bleeding, pattern mixing between items, top-bottom swap, wrong item placement, "
            "merged garments, fused outfit panels, mixed textures between collage items"
        )

    if types == {"dress"}:
        parts.append(
            "top or bottom layering under dress, incomplete dress replacement, split two-piece look, "
            "wrong skirt silhouette, wrong hem contour"
        )
    elif types == {"top"}:
        parts.append(
            "modified pants, modified skirt, modified shorts, modified shoes, altered lower-body garment color, "
            "lower-body garment structure change"
        )
    elif types == {"bottom"}:
        parts.append(
            "modified shirt, modified top, modified jacket, altered upper-body garment color, "
            "upper-body garment structure change"
        )

    return " | ".join([p for p in parts if p]).strip()


def _build_flux2_runtime_negative_prompt(
    *,
    target_types: List[str],
    board_mode: str,
    custom_negative_prompt: str = "",
) -> str:
    """
    Runtime negative prompt optimized for Flux2 distilled behavior.
    Keep this shorter and highly targeted so prompt-fallback stays effective.
    """
    mode = FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE
    custom = " ".join(str(custom_negative_prompt or "").split()).strip()
    if mode == "disabled":
        return ""
    if custom and mode in {"request_only", "request_or_auto"}:
        return custom
    if mode == "request_only":
        return ""

    base_terms: List[str] = [
        "wrong garment color",
        "recolored fabric",
        "pattern drift",
        "print placement shift",
        "texture swap",
        "incorrect neckline",
        "incorrect sleeve length",
        "incorrect hem length",
        "lost garment structure",
        "missing trims",
        "face identity change",
        "skin tone change",
        "body shape change",
    ]
    if board_mode == "collage" and len(target_types) > 1:
        base_terms.extend(
            [
                "cross-item color bleed",
                "top-bottom swap",
                "mixed item textures",
            ]
        )

    types = {str(t or "").strip().lower() for t in target_types if str(t or "").strip()}
    if types == {"dress"}:
        base_terms.extend(
            [
                "incomplete dress replacement",
                "top/bottom layering under dress",
                "split two-piece look",
                "wrong skirt silhouette",
            ]
        )
    elif types == {"top"}:
        base_terms.extend(
            [
                "changed pants",
                "changed skirt",
                "changed shorts",
                "changed shoes",
                "modified lower-body garment color",
                "modified lower-body garment structure",
                "edits below waistline",
            ]
        )
    elif types == {"bottom"}:
        base_terms.extend(
            [
                "changed top",
                "changed shirt",
                "changed jacket",
                "modified upper-body garment color",
                "modified upper-body garment structure",
                "edits above waistline",
            ]
        )

    return ", ".join(dict.fromkeys(base_terms))

def _normalize_descriptor_backend(raw: Optional[str]) -> str:
    value = str(raw or FLUX2_DESCRIPTOR_BACKEND).strip().lower()
    if value not in {"florence", "qwen2_5_vl", "joycaption", "minicpm", "minicpm_service"}:
        return "minicpm_service"
    if value == "qwen2_5_vl" and not FLUX2_ALLOW_QWEN_BACKEND:
        return "minicpm_service"
    return value


def _describe_with_minicpm_service(image_url: Optional[str], kind: str) -> str:
    """
    Fetch descriptor text from standalone MiniCPM service using source image URL.
    """
    clean_url = str(image_url or "").strip()
    if not clean_url:
        raise RuntimeError("minicpm_service requires a valid image URL")
    if not MINICPM_SERVICE_URL:
        raise RuntimeError("MINICPM_SERVICE_URL is not configured")

    if kind == "person":
        endpoint = f"{MINICPM_SERVICE_URL}/describe/person"
        max_new_tokens = MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS
        prompt = MINICPM_SERVICE_PERSON_PROMPT
    else:
        endpoint = f"{MINICPM_SERVICE_URL}/describe/garment"
        max_new_tokens = MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS
        prompt = MINICPM_SERVICE_GARMENT_PROMPT

    payload = {
        "image_url": clean_url,
        "max_new_tokens": int(max_new_tokens),
        "prompt": prompt,
    }
    resp = requests.post(endpoint, json=payload, timeout=MINICPM_SERVICE_TIMEOUT_S)
    if resp.status_code != 200:
        detail = resp.text[:500]
        raise RuntimeError(
            f"minicpm_service {kind} failed: status={resp.status_code} detail={detail}"
        )
    data = resp.json() if resp.content else {}
    text = str(data.get("text", "")).strip()
    if not text:
        raise RuntimeError(f"minicpm_service {kind} returned empty text")
    return _normalize_minicpm_descriptor_text(text, kind=kind)

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

def _describe_garment_with_backend(
    image: Image.Image,
    backend: str,
    image_url: Optional[str] = None,
) -> str:
    resolved = _normalize_descriptor_backend(backend)
    if resolved == "minicpm_service":
        try:
            if image_url:
                return _describe_with_minicpm_service(image_url=image_url, kind="garment")
        except Exception as err:
            logger.warning(f"MiniCPM service garment description failed; fallback to local/Floorence. error={err}")
    if resolved == "minicpm":
        try:
            minicpm_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            return str(engine.minicpm.describe_garment(minicpm_img)).strip()
        except Exception as err:
            logger.warning(f"MiniCPM garment description failed; fallback to Florence. error={err}")
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

def _describe_user_image_for_flux2(
    user_img: Image.Image,
    backend: str = "florence",
    image_url: Optional[str] = None,
) -> str:
    """
    Ask selected descriptor model for a full-person detailed description.
    """
    resolved = _normalize_descriptor_backend(backend)
    if resolved == "minicpm_service":
        try:
            if image_url:
                return _describe_with_minicpm_service(image_url=image_url, kind="person")
        except Exception as err:
            logger.warning(f"MiniCPM service user description failed; fallback to local/Floorence. error={err}")
    if resolved == "minicpm":
        try:
            minicpm_img = _resize_for_qwen_caption(
                image=user_img,
                max_side=FLUX2_MINICPM_USER_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_USER_CAPTION_MIN_SIDE,
            )
            return str(engine.minicpm.describe_person_and_outfit(minicpm_img)).strip()
        except Exception as err:
            logger.warning(f"MiniCPM user description failed; fallback to Florence. error={err}")
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


def _score_untargeted_region_preservation(
    reference_image: Image.Image,
    output_image: Image.Image,
    target_types: List[str],
) -> float:
    """
    Score how well the untargeted body region is preserved.
    Used for top/bottom single-item selection.
    """
    try:
        types = {str(t or "").strip().lower() for t in target_types if str(t or "").strip()}
        if types == {"top"}:
            mode = "lower"
        elif types == {"bottom"}:
            mode = "upper"
        else:
            return 0.0

        ref = reference_image.convert("RGB").resize((256, 384), Image.BICUBIC)
        out = output_image.convert("RGB").resize((256, 384), Image.BICUBIC)
        arr_ref = np.asarray(ref, dtype=np.float32) / 255.0
        arr_out = np.asarray(out, dtype=np.float32) / 255.0
        h = arr_ref.shape[0]
        if mode == "lower":
            y0 = int(h * 0.58)
            ref_roi = arr_ref[y0:, :, :]
            out_roi = arr_out[y0:, :, :]
        else:
            y1 = int(h * 0.52)
            ref_roi = arr_ref[:y1, :, :]
            out_roi = arr_out[:y1, :, :]

        mae = float(np.mean(np.abs(ref_roi - out_roi)))
        # Map mae to [0,1], where 1 means strong preservation.
        score = max(0.0, min(1.0, 1.0 - (mae / 0.28)))
        return score
    except Exception:
        return 0.0

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
        # Parser top masks frequently truncate one-sleeve/long-sleeve regions.
        # Keep richer context for captioning + flux garment reconstruction.
        x_pad = int(bw * 0.12)
        y_pad_top = int(bh * 0.05)
        y_pad_bottom = int(bh * 0.36)
        if y0 < int(image_height * 0.38):
            y_pad_bottom = max(y_pad_bottom, int(image_height * 0.20))
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

def _expand_bbox_with_ratios(
    bbox: list[int],
    image_width: int,
    image_height: int,
    x_pad_ratio: float,
    y_pad_top_ratio: float,
    y_pad_bottom_ratio: float,
) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    x_pad = int(bw * max(0.0, x_pad_ratio))
    y_pad_top = int(bh * max(0.0, y_pad_top_ratio))
    y_pad_bottom = int(bh * max(0.0, y_pad_bottom_ratio))
    return [
        max(0, x0 - x_pad),
        max(0, y0 - y_pad_top),
        min(image_width, x1 + x_pad),
        min(image_height, y1 + y_pad_bottom),
    ]

def _adaptive_rect_profiles_for_type(garment_type: str) -> list[tuple[str, tuple[float, float, float]]]:
    gt = _normalize_garment_type(garment_type) or "top"
    if gt == "top":
        return [
            # Keep top crops object-aligned; allow only small context growth per variant.
            ("tight", (0.06, 0.03, 0.06)),
            ("balanced", (0.10, 0.05, 0.10)),
            ("wide", (0.14, 0.07, 0.14)),
        ]
    if gt == "bottom":
        return [
            ("tight", (0.06, 0.20, 0.08)),
            ("balanced", (0.10, 0.30, 0.12)),
            ("wide", (0.14, 0.40, 0.18)),
        ]
    if gt == "dress":
        return [
            ("tight", (0.06, 0.08, 0.12)),
            ("balanced", (0.10, 0.12, 0.20)),
            ("wide", (0.14, 0.16, 0.28)),
        ]
    return [
        ("tight", (0.06, 0.05, 0.10)),
        ("balanced", (0.10, 0.08, 0.18)),
        ("wide", (0.14, 0.12, 0.24)),
    ]

def _top_crop_bottom_cutoff_y(
    selected_candidate: dict,
    all_candidates: Optional[list[dict]],
    image_height: int,
) -> Optional[int]:
    parser_bbox = selected_candidate.get("parser_bbox") or selected_candidate.get("bbox")
    if not isinstance(parser_bbox, list) or len(parser_bbox) < 4:
        return None
    try:
        top_y1 = int(parser_bbox[3])
        top_h = max(1, int(parser_bbox[3]) - int(parser_bbox[1]))
    except Exception:
        return None

    bottoms: list[int] = []
    for item in all_candidates or []:
        if _normalize_garment_type(item.get("type")) != "bottom":
            continue
        b = item.get("bbox") or item.get("parser_bbox")
        if not isinstance(b, list) or len(b) < 2:
            continue
        try:
            bottoms.append(int(b[1]))
        except Exception:
            continue

    if not bottoms:
        return None
    first_bottom_y = min(bottoms)
    # Keep a small overlap allowance so sleeve/tie details near waist are not clipped.
    slack = max(int(top_h * 0.22), int(image_height * 0.02))
    return max(top_y1, min(int(image_height), int(first_bottom_y + slack)))

def _build_adaptive_rect_crop_variants(
    full_image: Image.Image,
    selected_candidate: dict,
    garment_type: str,
    all_candidates: Optional[list[dict]] = None,
) -> list[dict]:
    gt = _normalize_garment_type(garment_type) or "top"
    if gt == "top":
        preview_img = selected_candidate.get("_preview_image")
        top_obj_bbox = selected_candidate.get("top_object_bbox")
        top_support_bbox = selected_candidate.get("top_support_bbox")
        parser_bbox = selected_candidate.get("parser_bbox") or selected_candidate.get("bbox")
        base_bbox = top_obj_bbox or top_support_bbox or parser_bbox or [0, 0, full_image.width, full_image.height]
        variants: list[dict] = []
        if isinstance(preview_img, Image.Image):
            # Primary top variant uses mask-aligned upper-body context (top + torso + hands).
            variants.append(
                {
                    "name": "masked_context",
                    "bbox": [int(v) for v in base_bbox[:4]],
                    "image": _flatten_rgba_on_white(preview_img),
                }
            )
        sq_img = selected_candidate.get("_square_crop_image")
        sq_bbox = selected_candidate.get("square_bbox")
        if isinstance(sq_img, Image.Image) and isinstance(sq_bbox, list) and len(sq_bbox) >= 4:
            variants.append(
                {
                    "name": "square_context",
                    "bbox": [int(v) for v in sq_bbox[:4]],
                    "image": sq_img.convert("RGB"),
                }
            )
        if variants:
            return variants

    if gt == "top":
        anchor = (
            selected_candidate.get("top_support_bbox")
            or selected_candidate.get("top_object_bbox")
            or selected_candidate.get("parser_bbox")
            or selected_candidate.get("section_bbox")
            or selected_candidate.get("bbox")
            or [0, 0, full_image.width, full_image.height]
        )
    else:
        anchor = selected_candidate.get("section_bbox") or selected_candidate.get("bbox") or [0, 0, full_image.width, full_image.height]
    ax0, ay0, ax1, ay1 = [int(v) for v in anchor]
    top_cutoff_y = None
    if gt == "top" and selected_candidate.get("top_object_bbox") is None:
        top_cutoff_y = _top_crop_bottom_cutoff_y(selected_candidate, all_candidates, full_image.height)

    variants: list[dict] = []
    for name, ratios in _adaptive_rect_profiles_for_type(garment_type):
        ex = _expand_bbox_with_ratios(
            [ax0, ay0, ax1, ay1],
            image_width=full_image.width,
            image_height=full_image.height,
            x_pad_ratio=float(ratios[0]),
            y_pad_top_ratio=float(ratios[1]),
            y_pad_bottom_ratio=float(ratios[2]),
        )
        x0, y0, x1, y1 = [int(v) for v in ex]
        if top_cutoff_y is not None:
            y1 = min(y1, int(top_cutoff_y))
            if y1 <= y0 + 24:
                y1 = min(full_image.height, y0 + max(64, int((ay1 - ay0) * 0.80)))
        if x1 <= x0 or y1 <= y0:
            continue
        variants.append(
            {
                "name": name,
                "bbox": [x0, y0, x1, y1],
                "image": full_image.crop((x0, y0, x1, y1)).convert("RGB"),
            }
        )
    if gt == "top":
        sq_img = selected_candidate.get("_square_crop_image")
        sq_bbox = selected_candidate.get("square_bbox")
        if isinstance(sq_img, Image.Image) and isinstance(sq_bbox, list) and len(sq_bbox) >= 4:
            variants.append(
                {
                    "name": "square_context",
                    "bbox": [int(v) for v in sq_bbox[:4]],
                    "image": sq_img.convert("RGB"),
                }
            )
    return variants

def _joycaption_describe_with_retry(image: Image.Image, instruction_primary: str, garment_type: str) -> tuple[str, str]:
    caption = " ".join(
        str(
            engine.joycaption.describe_garment(
                image,
                instruction_override=instruction_primary,
            )
            or ""
        ).split()
    ).strip()
    if caption:
        return caption, "primary"

    normalized_type = _normalize_garment_type(garment_type) or "garment"
    relaxed_instruction = (
        f"Describe only the {normalized_type} garment in this crop. "
        "One detailed sentence only. Mention silhouette, fit, neckline/waist/hem, sleeves, fabric, transparency, "
        "pattern, embellishments, and exact dominant colors. Do not describe person or background."
    )
    caption_retry = " ".join(
        str(
            engine.joycaption.describe_garment(
                image,
                instruction_override=relaxed_instruction,
            )
            or ""
        ).split()
    ).strip()
    if caption_retry:
        return caption_retry, "relaxed_retry"
    return "", "empty"

def _score_adaptive_crop_caption(caption: str, target_type: str) -> tuple[float, dict]:
    text = " ".join(str(caption or "").split()).strip()
    inferred = _infer_type_from_caption(text)
    has_garment = _caption_garment_signal(text)
    has_non_garment = _caption_non_garment_signal(text)
    words = len(text.split())
    score = 0.0
    if inferred == target_type:
        score += 1.20
    elif inferred is None:
        score += 0.20
    else:
        score -= 0.35
    if has_garment:
        score += 0.40
    if has_non_garment:
        score -= 0.40
    score += min(0.60, (words / 28.0) * 0.60)
    return score, {
        "inferred_type": inferred,
        "has_garment_signal": bool(has_garment),
        "has_non_garment_signal": bool(has_non_garment),
        "word_count": int(words),
    }

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

def _clip_prompt_labels() -> dict[str, str]:
    return {
        "top": "a fashion photo of an upper-body top garment like blouse shirt crop-top or jacket",
        "bottom": "a fashion photo of a lower-body garment like skirt pants shorts or trousers",
        "dress": "a fashion photo of a one-piece dress or gown",
    }

def _parser_candidate_geom_scores(
    item: dict,
    image_width: int,
    image_height: int,
    person_center_x: int,
    person_width: int,
) -> dict[str, float]:
    b = item.get("bbox") or [0, 0, image_width, image_height]
    x0, y0, x1, y1 = [int(v) for v in b]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cyr = (0.5 * (y0 + y1)) / float(max(1, image_height))
    y0r = y0 / float(max(1, image_height))
    y1r = y1 / float(max(1, image_height))
    hr = bh / float(max(1, image_height))
    centered_cover = (
        x0 <= int(person_center_x - person_width * 0.12)
        and x1 >= int(person_center_x + person_width * 0.12)
    )
    centered_factor = 1.0 if centered_cover else 0.72

    top_geo = max(0.0, min(1.0, (0.72 - cyr) / 0.38))
    bottom_geo = max(0.0, min(1.0, (cyr - 0.30) / 0.42))
    dress_span = max(0.0, min(1.0, (hr - 0.28) / 0.46))
    dress_vertical = 1.0 if (y0r < 0.62 and y1r > 0.56) else 0.28
    dress_geo = dress_span * dress_vertical * centered_factor

    # Dress-like regions should not be tiny narrow side patches.
    min_dress_w = int(max(28, person_width * 0.24))
    if bw < min_dress_w:
        dress_geo *= 0.35

    return {
        "top": float(max(0.0, min(1.0, top_geo * centered_factor))),
        "bottom": float(max(0.0, min(1.0, bottom_geo * centered_factor))),
        "dress": float(max(0.0, min(1.0, dress_geo))),
    }

def _collect_yolo_support_boxes(image: Image.Image) -> list[dict]:
    if not PARSER_FUSION_USE_YOLO_SUPPORT:
        return []
    try:
        yolo_instances = engine.yolo.detect_instances(image)
    except Exception as yolo_err:
        logger.warning(f"Parser fusion YOLO support failed: {yolo_err}")
        return []

    out: list[dict] = []
    total = float(max(1, image.width * image.height))
    for inst in yolo_instances:
        bbox = inst.get("bbox") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        if x1 <= x0 or y1 <= y0:
            continue
        area_ratio = ((x1 - x0) * (y1 - y0)) / total
        if area_ratio < 0.01:
            continue
        out.append(
            {
                "bbox": [x0, y0, x1, y1],
                "confidence": float(inst.get("confidence", 0.0)),
                "area_ratio": float(area_ratio),
            }
        )
    return out

def _yolo_support_score(candidate_bbox: list[int], yolo_boxes: list[dict]) -> float:
    if not yolo_boxes:
        return 0.0
    cb = tuple(int(v) for v in candidate_bbox)
    best = 0.0
    for y in yolo_boxes:
        yb = tuple(int(v) for v in (y.get("bbox") or [0, 0, 0, 0]))
        iou = bbox_iou(cb, yb)
        yc = float(y.get("confidence", 0.0))
        support = max(0.0, min(1.0, (0.65 * iou) + (0.35 * yc)))
        if support > best:
            best = support
    return float(max(0.0, min(1.0, best)))

def _resolve_parser_candidates_with_fusion(
    image: Image.Image,
    candidates: list[dict],
) -> list[dict]:
    if not candidates:
        return candidates
    if not PARSER_FUSION_V1_ENABLED:
        return candidates

    w, h = image.size
    all_x0 = [int((c.get("bbox") or [0, 0, w, h])[0]) for c in candidates]
    all_x1 = [int((c.get("bbox") or [0, 0, w, h])[2]) for c in candidates]
    person_x0 = min(all_x0) if all_x0 else 0
    person_x1 = max(all_x1) if all_x1 else w
    person_w = max(1, person_x1 - person_x0)
    person_cx = int((person_x0 + person_x1) * 0.5)

    yolo_boxes = _collect_yolo_support_boxes(image)
    clip_labels = _clip_prompt_labels()

    wp = max(0.0, float(PARSER_FUSION_WEIGHT_PARSER))
    wy = max(0.0, float(PARSER_FUSION_WEIGHT_YOLO))
    wc = max(0.0, float(PARSER_FUSION_WEIGHT_OPENCLIP))
    wsum = max(1e-6, wp + wy + wc)
    wp, wy, wc = (wp / wsum), (wy / wsum), (wc / wsum)

    for item in candidates:
        parser_type = _normalize_garment_type(str(item.get("type") or "")) or "top"
        item["parser_type"] = parser_type
        subtype = str(item.get("subtype") or "").strip().lower()

        bbox = item.get("bbox") or [0, 0, w, h]
        geom_scores = _parser_candidate_geom_scores(item, w, h, person_cx, person_w)
        yolo_support = _yolo_support_score([int(v) for v in bbox], yolo_boxes)

        clip_scores = {"top": 1.0 / 3.0, "bottom": 1.0 / 3.0, "dress": 1.0 / 3.0}
        if PARSER_FUSION_USE_OPENCLIP:
            try:
                crop = item.get("_crop_image")
                if isinstance(crop, Image.Image):
                    label_order = [clip_labels["top"], clip_labels["bottom"], clip_labels["dress"]]
                    raw_clip = engine.openclip.score_labels(crop, label_order)
                    clip_scores = {
                        "top": float(raw_clip.get(clip_labels["top"], 0.0)),
                        "bottom": float(raw_clip.get(clip_labels["bottom"], 0.0)),
                        "dress": float(raw_clip.get(clip_labels["dress"], 0.0)),
                    }
            except Exception as clip_err:
                logger.warning(f"Parser fusion OpenCLIP scoring failed: {clip_err}")

        parser_prior = {"top": 0.05, "bottom": 0.05, "dress": 0.05}
        parser_prior[parser_type] = 0.90

        type_scores: dict[str, float] = {}
        for t in ("top", "bottom", "dress"):
            geom_yolo = float(geom_scores.get(t, 0.0))
            if PARSER_FUSION_USE_YOLO_SUPPORT:
                geom_yolo = geom_yolo * max(0.35, yolo_support)
            type_scores[t] = (wp * parser_prior[t]) + (wy * geom_yolo) + (wc * float(clip_scores.get(t, 0.0)))

        ranked = sorted(type_scores.items(), key=lambda kv: float(kv[1]), reverse=True)
        resolved_type = str(ranked[0][0])
        resolved_score = float(ranked[0][1])
        second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        gap = resolved_score - second_score

        reason_codes: list[str] = []
        if resolved_type != parser_type:
            reason_codes.append("fusion_type_override")
        if gap < PARSER_FUSION_AMBIGUITY_GAP:
            reason_codes.append("fusion_ambiguous")
        if yolo_support < 0.12:
            reason_codes.append("yolo_support_weak")

        item["type"] = resolved_type
        item["category_text"] = _parser_candidate_category_text(
            resolved_type,
            subtype if resolved_type == "bottom" else "",
        )
        item["fusion_score"] = round(resolved_score, 4)
        item["fusion_gap"] = round(gap, 4)
        item["fusion_reason_codes"] = reason_codes
        item["fusion_support"] = {
            "parser_prior": round(float(parser_prior.get(resolved_type, 0.0)), 4),
            "geometry": round(float(geom_scores.get(resolved_type, 0.0)), 4),
            "yolo": round(float(yolo_support), 4),
            "openclip": round(float(clip_scores.get(resolved_type, 0.0)), 4),
        }
        item["fusion_type_scores"] = {k: round(float(v), 4) for k, v in type_scores.items()}

    # Global conflict cleanup: suppress weak dress fragments when top+bottom are strong.
    top_best = None
    bottom_best = None
    dress_items: list[dict] = []
    for item in candidates:
        t = str(item.get("type") or "")
        if t == "top":
            if top_best is None or float(item.get("fusion_score", 0.0)) > float(top_best.get("fusion_score", 0.0)):
                top_best = item
        elif t == "bottom":
            if bottom_best is None or float(item.get("fusion_score", 0.0)) > float(bottom_best.get("fusion_score", 0.0)):
                bottom_best = item
        elif t == "dress":
            dress_items.append(item)

    filtered = list(candidates)
    if top_best is not None and bottom_best is not None and dress_items:
        tb = top_best.get("bbox") or [0, 0, w, h]
        bb = bottom_best.get("bbox") or [0, 0, w, h]
        top_score = float(top_best.get("fusion_score", 0.0))
        bottom_score = float(bottom_best.get("fusion_score", 0.0))
        max_tb = max(top_score, bottom_score)
        waist_mid_y = int((int(tb[3]) + int(bb[1])) * 0.5)

        keep_set = set(id(x) for x in filtered)
        for dress_item in dress_items:
            db = dress_item.get("bbox") or [0, 0, w, h]
            x0, y0, x1, y1 = [int(v) for v in db]
            spans_waist = y0 <= waist_mid_y <= y1
            centered_cover = (
                x0 <= int(person_cx - person_w * 0.12)
                and x1 >= int(person_cx + person_w * 0.12)
            )
            dress_score = float(dress_item.get("fusion_score", 0.0))
            if max_tb >= 0.30 and (not spans_waist or not centered_cover or dress_score < (max_tb * 0.90)):
                dress_item.setdefault("fusion_reason_codes", [])
                dress_item["fusion_reason_codes"].append("dress_conflicts_with_top_bottom")
                if id(dress_item) in keep_set:
                    keep_set.remove(id(dress_item))
        filtered = [x for x in filtered if id(x) in keep_set]

    strong = [x for x in filtered if float(x.get("fusion_score", 0.0)) >= PARSER_FUSION_MIN_SCORE_KEEP]
    if strong:
        filtered = strong
    if not filtered:
        return candidates
    return filtered

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
    # Build all garment regions first so fusion can use inter-type context;
    # requested_type filtering is applied after fusion.
    if normalized_type == "outer":
        target_types = ["outer"]
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

            top_support_bbox = None
            top_object_bbox = None
            top_object_mask = None
            top_object_crop_rgba = None
            if garment_type == "top":
                top_support_bbox = _top_context_bbox_from_parsing(
                    parsing=parsing,
                    top_component_mask=component_mask,
                    top_bbox=bbox,
                    image_width=w,
                    image_height=h,
                )
                top_object_mask = _top_object_mask_from_parsing(
                    parsing=parsing,
                    top_component_mask=component_mask,
                    top_bbox=top_support_bbox or bbox,
                    image_width=w,
                    image_height=h,
                )
                object_bbox = _bbox_from_mask(top_object_mask) if top_object_mask is not None else None
                if object_bbox is not None:
                    top_object_crop_rgba, top_object_bbox = _crop_rgba_with_mask(
                        image=image,
                        object_mask=top_object_mask,
                        bbox=object_bbox,
                        x_pad_ratio=0.03,
                        y_pad_top_ratio=0.02,
                        y_pad_bottom_ratio=0.04,
                    )

            if garment_type == "top" and top_object_bbox is not None:
                # Dedicated top crop: tightly follow garment object while retaining sleeve continuity.
                section_bbox = _expand_bbox_with_ratios(
                    bbox=top_object_bbox,
                    image_width=w,
                    image_height=h,
                    x_pad_ratio=0.05,
                    y_pad_top_ratio=0.03,
                    y_pad_bottom_ratio=0.08,
                )
            else:
                section_bbox = _expand_section_bbox_by_type(
                    bbox=top_support_bbox or bbox,
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
            if garment_type == "top" and top_object_crop_rgba is not None:
                rect_crop = _flatten_rgba_on_white(top_object_crop_rgba)
            else:
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
            dominant_hexes = _extract_dominant_hex_colors(
                image=image,
                mask=component_mask,
                top_k=4,
            )

            candidates.append(
                {
                    "type": garment_type,
                    "subtype": subtype,
                    "category_text": category_text,
                    "bbox": [int(v) for v in bbox],
                    "parser_bbox": [int(v) for v in bbox],
                    "top_support_bbox": [int(v) for v in top_support_bbox] if top_support_bbox else None,
                    "top_object_bbox": [int(v) for v in top_object_bbox] if top_object_bbox else None,
                    "section_bbox": [int(v) for v in section_bbox],
                    "square_bbox": square_bbox,
                    "crop_mode": "object_mask_aligned_top" if top_object_crop_rgba is not None else "tight_rect_parser_context",
                    "parser_component_area_ratio": round(float(area) / total_pixels, 6),
                    "component_coverage_in_square": round(float(area) / square_area, 6),
                    "merged_component_count": int(component.get("merged_component_count", 1)),
                    "parser_labels": label_summary,
                    "dominant_color_hexes": dominant_hexes,
                    "_crop_image": rect_crop,
                    "_square_crop_image": raw_square_crop,
                    "_preview_image": top_object_crop_rgba if top_object_crop_rgba is not None else rect_crop,
                    "_object_cutout_image": top_object_crop_rgba if top_object_crop_rgba is not None else None,
                }
            )

    candidates = _filter_parser_candidates_by_consistency(
        candidates,
        image_width=w,
        image_height=h,
    )
    if normalized_type != "outer":
        candidates = _resolve_parser_candidates_with_fusion(image=image, candidates=candidates)
    candidates.sort(
        key=lambda item: (
            float(item.get("fusion_score", 0.0)),
            float(item.get("parser_component_area_ratio", 0.0)),
            float(item.get("component_coverage_in_square", 0.0)),
        ),
        reverse=True,
    )
    for idx, candidate in enumerate(candidates):
        candidate["garment_id"] = idx
    return candidates

def _build_parser_unified_square_split_candidates(
    image: Image.Image,
    requested_type: Optional[str],
    min_component_area_ratio: float,
    square_padding_ratio: float,
) -> list[dict]:
    """
    Unified-first strategy:
    1) Build one square crop covering all garment regions.
    2) Re-parse that square.
    3) Split top/bottom/dress from square parsing masks.
    """
    if engine.parser is None:
        raise RuntimeError("Human parser is not enabled.")

    parsing_full = engine.parser.parse(image)
    h, w = parsing_full.shape[:2]
    total_pixels = float(max(1, w * h))
    min_ratio = max(0.0005, min(0.25, float(min_component_area_ratio)))
    min_pixels = max(96, int(total_pixels * min_ratio))

    normalized_type = _normalize_garment_type(requested_type)
    if normalized_type == "outer":
        target_types = ["outer"]
    else:
        target_types = ["top", "bottom", "dress"]

    union_ids: list[int] = []
    for t in target_types:
        union_ids.extend([int(v) for v in (_parser_extraction_keep_ids(t) or [])])
    union_ids = sorted(set(union_ids))
    if not union_ids:
        return []

    union_mask = np.isin(parsing_full, union_ids)
    if int(union_mask.sum()) < max(64, int(min_pixels * 0.6)):
        return []
    union_mask = binary_open(union_mask, 3)
    union_mask = binary_close(union_mask, 3)
    union_bbox = _bbox_from_mask(union_mask)
    if union_bbox is None:
        return []

    unified_square_bbox = _square_bbox_from_bbox(
        bbox=[int(v) for v in union_bbox],
        image_width=w,
        image_height=h,
        padding_ratio=max(0.10, float(square_padding_ratio)),
    )
    unified_square = _crop_square_with_padding(image, unified_square_bbox)
    parsing_square = engine.parser.parse(unified_square)
    sh, sw = parsing_square.shape[:2]
    square_total_pixels = float(max(1, sw * sh))
    square_min_pixels = max(48, int(square_total_pixels * max(0.0003, min_ratio * 0.45)))
    id2label_square = _parser_runtime_id2label()

    candidates: list[dict] = []
    sqx0, sqy0, _, _ = [int(v) for v in unified_square_bbox]

    for garment_type in target_types:
        keep_ids = _parser_extraction_keep_ids(garment_type)
        if not keep_ids:
            continue
        type_mask_sq = np.isin(parsing_square, [int(v) for v in keep_ids])
        if int(type_mask_sq.sum()) < square_min_pixels:
            continue
        type_mask_sq = binary_open(type_mask_sq, 3)
        type_mask_sq = binary_close(type_mask_sq, 3)

        components = _mask_connected_components(type_mask_sq, min_pixels=square_min_pixels)
        if garment_type == "bottom" and len(components) > 1:
            components = _merge_bottom_components_if_same_item(
                components,
                image_width=sw,
                image_height=sh,
            )

        for component in components:
            comp_bbox_local = component.get("bbox") or [0, 0, sw, sh]
            comp_area = int(component.get("area", 0))
            comp_mask_local = component.get("mask")
            if comp_area <= 0 or comp_mask_local is None:
                continue
            comp_bbox_local = _robust_bbox_from_component_mask(comp_mask_local, comp_bbox_local)

            top_support_bbox_local = None
            top_object_bbox_local = None
            top_object_crop_rgba = None
            preview_image: Image.Image
            crop_mode = "unified_square_parser_split"

            if garment_type == "top":
                top_support_bbox_local = _top_context_bbox_from_parsing(
                    parsing=parsing_square,
                    top_component_mask=comp_mask_local,
                    top_bbox=comp_bbox_local,
                    image_width=sw,
                    image_height=sh,
                )
                top_object_mask_local = _top_object_mask_from_parsing(
                    parsing=parsing_square,
                    top_component_mask=comp_mask_local,
                    top_bbox=top_support_bbox_local or comp_bbox_local,
                    image_width=sw,
                    image_height=sh,
                )
                obj_bbox = _bbox_from_mask(top_object_mask_local)
                if obj_bbox is not None:
                    top_object_crop_rgba, top_object_bbox_local = _crop_rgba_with_mask(
                        image=unified_square,
                        object_mask=top_object_mask_local,
                        bbox=obj_bbox,
                        x_pad_ratio=0.03,
                        y_pad_top_ratio=0.02,
                        y_pad_bottom_ratio=0.04,
                    )
                preview_image = top_object_crop_rgba if top_object_crop_rgba is not None else unified_square.copy()
                crop_mode = "unified_square_top_mask_context"
            else:
                rgba_crop, obj_bbox_local = _crop_rgba_with_mask(
                    image=unified_square,
                    object_mask=comp_mask_local,
                    bbox=comp_bbox_local,
                    x_pad_ratio=0.03,
                    y_pad_top_ratio=0.02,
                    y_pad_bottom_ratio=0.04,
                )
                preview_image = rgba_crop
                top_object_bbox_local = [int(v) for v in obj_bbox_local]

            label_summary = _component_label_summary(parsing_square, comp_mask_local, id2label=id2label_square, top_k=3)
            subtype = ""
            if garment_type == "bottom":
                for entry in label_summary:
                    lbl = str(entry.get("label", "")).strip().lower()
                    if lbl in {"skirt", "pants", "trousers", "shorts"}:
                        subtype = lbl
                        break

            parser_bbox_global = [sqx0 + int(comp_bbox_local[0]), sqy0 + int(comp_bbox_local[1]), sqx0 + int(comp_bbox_local[2]), sqy0 + int(comp_bbox_local[3])]
            section_bbox_global = [sqx0, sqy0, sqx0 + sw, sqy0 + sh]
            top_support_bbox_global = None
            if top_support_bbox_local is not None:
                top_support_bbox_global = [
                    sqx0 + int(top_support_bbox_local[0]),
                    sqy0 + int(top_support_bbox_local[1]),
                    sqx0 + int(top_support_bbox_local[2]),
                    sqy0 + int(top_support_bbox_local[3]),
                ]
            top_object_bbox_global = None
            if top_object_bbox_local is not None:
                top_object_bbox_global = [
                    sqx0 + int(top_object_bbox_local[0]),
                    sqy0 + int(top_object_bbox_local[1]),
                    sqx0 + int(top_object_bbox_local[2]),
                    sqy0 + int(top_object_bbox_local[3]),
                ]

            dominant_hexes = _extract_dominant_hex_colors(
                image=unified_square,
                mask=comp_mask_local,
                top_k=4,
            )
            area_ratio = float(comp_area) / float(max(1, sw * sh))

            candidates.append(
                {
                    "type": garment_type,
                    "subtype": subtype,
                    "category_text": _parser_candidate_category_text(garment_type, subtype),
                    "bbox": [int(v) for v in parser_bbox_global],
                    "parser_bbox": [int(v) for v in parser_bbox_global],
                    "top_support_bbox": [int(v) for v in top_support_bbox_global] if top_support_bbox_global else None,
                    "top_object_bbox": [int(v) for v in top_object_bbox_global] if top_object_bbox_global else None,
                    "section_bbox": [int(v) for v in section_bbox_global],
                    "square_bbox": [int(v) for v in unified_square_bbox],
                    "crop_mode": crop_mode,
                    "parser_component_area_ratio": round(area_ratio, 6),
                    "component_coverage_in_square": round(area_ratio, 6),
                    "merged_component_count": int(component.get("merged_component_count", 1)),
                    "parser_labels": label_summary,
                    "dominant_color_hexes": dominant_hexes,
                    "_crop_image": _flatten_rgba_on_white(preview_image),
                    "_preview_image": preview_image,
                    "_object_cutout_image": preview_image if preview_image.mode == "RGBA" else None,
                    "_context_crop_image": unified_square.copy(),
                }
            )

    candidates = _filter_parser_candidates_by_consistency(
        candidates,
        image_width=w,
        image_height=h,
    )
    if normalized_type != "outer":
        candidates = _resolve_parser_candidates_with_fusion(image=image, candidates=candidates)
    candidates.sort(
        key=lambda item: (
            float(item.get("fusion_score", 0.0)),
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
    dominant_color_hexes: Optional[List[str]] = None,
) -> str:
    clean_desc = " ".join(str(prompt_description or "").split()).strip()
    item_phrase = {
        "top": "top garment",
        "bottom": "bottom garment",
        "dress": "dress",
        "outer": "outerwear piece",
    }.get(_normalize_garment_type(garment_type) or "top", "garment")
    explicit_category = str(category_text or _parser_candidate_category_text(garment_type)).strip().lower()
    hexes = [str(v).strip() for v in (dominant_color_hexes or []) if str(v).strip()]
    color_lock_clause = ""
    if hexes:
        color_lock_clause = (
            f" Keep exact source colors with strict palette lock: {', '.join(hexes)}. "
            "Do not recolor and do not shift hue/saturation/value."
        )
    return (
        "Generate a standalone product shot of the garment only (no person, no mannequin, no body parts). "
        "Use a clean pure white studio background only (RGB 255,255,255), with no props and no scene context. "
        "Center a single garment in frame with full silhouette visible and edge-to-edge clarity. "
        f"Generate only a single {explicit_category} category garment and no other categories. "
        f"Reconstruct the exact {item_phrase} from the reference crop with strict fidelity to silhouette, "
        "neckline/waist/hem geometry, fit, fabric texture, transparency level, print/embellishment placement, "
        f"and color palette.{color_lock_clause} Garment details: {clean_desc}"
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

def _parser_alias_ids(aliases: list[str], fallback: Optional[list[int]] = None) -> list[int]:
    out: list[int] = []
    runtime = _parser_runtime_label2id()
    for alias in aliases or []:
        key = str(alias or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            continue
        if key in runtime:
            try:
                out.append(int(runtime[key]))
            except Exception:
                continue
    if out:
        return sorted(set(out))
    return _parser_category_ids((aliases or [""])[0] if aliases else "", fallback or [])

def _connected_support_mask(base_mask: np.ndarray, support_mask: np.ndarray, dilate_iters: int = 3) -> np.ndarray:
    base = np.asarray(base_mask).astype(bool)
    sup = np.asarray(support_mask).astype(bool)
    if base.shape != sup.shape:
        return np.zeros_like(base, dtype=bool)
    if not np.any(base) or not np.any(sup):
        return np.zeros_like(base, dtype=bool)
    try:
        from scipy.ndimage import binary_dilation, label

        seed = binary_dilation(base, iterations=max(1, int(dilate_iters)))
        labeled, num_labels = label(sup)
        connected = np.zeros_like(sup, dtype=bool)
        for comp_id in range(1, int(num_labels) + 1):
            comp = labeled == comp_id
            if np.any(comp & seed):
                connected |= comp
        return connected
    except Exception:
        # Conservative fallback: keep only direct overlap.
        return sup & base

def _crop_rgba_with_mask(
    image: Image.Image,
    object_mask: np.ndarray,
    bbox: list[int],
    x_pad_ratio: float = 0.03,
    y_pad_top_ratio: float = 0.02,
    y_pad_bottom_ratio: float = 0.04,
) -> tuple[Image.Image, list[int]]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    ex0 = max(0, x0 - int(bw * max(0.0, x_pad_ratio)))
    ey0 = max(0, y0 - int(bh * max(0.0, y_pad_top_ratio)))
    ex1 = min(image.width, x1 + int(bw * max(0.0, x_pad_ratio)))
    ey1 = min(image.height, y1 + int(bh * max(0.0, y_pad_bottom_ratio)))
    if ex1 <= ex0 or ey1 <= ey0:
        ex0, ey0, ex1, ey1 = x0, y0, x1, y1

    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).copy()
    alpha = arr[:, :, 3]
    keep = np.asarray(object_mask).astype(bool)
    if keep.shape != alpha.shape:
        keep = np.zeros_like(alpha, dtype=bool)
    alpha[~keep] = 0
    arr[:, :, 3] = alpha
    masked = Image.fromarray(arr, mode="RGBA")
    crop = masked.crop((ex0, ey0, ex1, ey1))
    return crop, [int(ex0), int(ey0), int(ex1), int(ey1)]

def _top_object_mask_from_parsing(
    parsing: np.ndarray,
    top_component_mask: np.ndarray,
    top_bbox: list[int],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    top_mask = np.asarray(top_component_mask).astype(bool)
    if not np.any(top_mask):
        return top_mask
    try:
        x0, y0, x1, y1 = [int(v) for v in top_bbox]
    except Exception:
        x0, y0, x1, y1 = [0, 0, int(image_width), int(image_height)]

    tw = max(1, x1 - x0)
    th = max(1, y1 - y0)
    # Large support window to keep long sleeves and hands connected to upper-body context.
    wx0 = max(0, x0 - int(tw * 0.55))
    wy0 = max(0, y0 - int(th * 0.20))
    wx1 = min(image_width, x1 + int(tw * 0.55))
    wy1 = min(image_height, y1 + int(th * 2.20))

    top_ids = _parser_extraction_keep_ids("top")
    arm_ids = _parser_alias_ids(["arms", "arm", "left_arm", "right_arm"], [14, 15])
    hand_ids = _parser_alias_ids(["hands", "hand", "left_hand", "right_hand"], [16])
    torso_ids = _parser_alias_ids(["torso", "upper_body", "body"], [1, 2, 3, 9, 10, 11, 12, 13])
    support_ids = sorted(set([int(v) for v in (top_ids + arm_ids + hand_ids + torso_ids)]))
    support = np.isin(parsing, support_ids)
    window = np.zeros_like(support, dtype=bool)
    window[wy0:wy1, wx0:wx1] = True
    support = support & window
    connected = _connected_support_mask(top_mask, support, dilate_iters=3)

    out = top_mask | connected
    # Hard-remove competing lower categories from top object mask.
    lower_ids = _parser_extraction_keep_ids("bottom") + _parser_extraction_keep_ids("dress")
    if lower_ids:
        lower = np.isin(parsing, [int(v) for v in lower_ids])
        out = out & (~lower)
    out = binary_open(out, 3)
    out = binary_close(out, 3)
    return out

def _top_context_bbox_from_parsing(
    parsing: np.ndarray,
    top_component_mask: np.ndarray,
    top_bbox: list[int],
    image_width: int,
    image_height: int,
) -> Optional[list[int]]:
    try:
        x0, y0, x1, y1 = [int(v) for v in top_bbox]
    except Exception:
        return None
    tw = max(1, x1 - x0)
    th = max(1, y1 - y0)
    pad_x = int(tw * 0.38)
    pad_y_top = int(th * 0.12)
    pad_y_bottom = int(th * 0.30)
    wx0 = max(0, x0 - pad_x)
    wy0 = max(0, y0 - pad_y_top)
    wx1 = min(image_width, x1 + pad_x)
    wy1 = min(image_height, y1 + pad_y_bottom)
    if wx1 <= wx0 or wy1 <= wy0:
        return None

    top_ids = _parser_extraction_keep_ids("top")
    arm_ids = _parser_alias_ids(["arms", "arm", "left_arm", "right_arm"], [14, 15])
    hand_ids = _parser_alias_ids(["hands", "hand", "left_hand", "right_hand"], [16])
    keep_ids = sorted(set([int(v) for v in (top_ids + arm_ids + hand_ids)]))
    if not keep_ids:
        return None

    support = np.isin(parsing, keep_ids)
    if not np.any(support):
        return None

    window = np.zeros_like(support, dtype=bool)
    window[wy0:wy1, wx0:wx1] = True
    support = support & window
    if not np.any(support):
        return None

    top_mask = np.asarray(top_component_mask).astype(bool)
    connected = _connected_support_mask(top_mask, support, dilate_iters=3)
    if np.any(connected):
        support = connected

    merged = support | top_mask
    bbox = _bbox_from_mask(merged)
    if bbox is None:
        return None
    bx0, by0, bx1, by1 = [int(v) for v in bbox]
    # Ensure upper garment anchor always keeps original top component.
    bx0 = min(bx0, x0)
    by0 = min(by0, y0)
    bx1 = max(bx1, x1)
    by1 = max(by1, y1)
    return [bx0, by0, bx1, by1]

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
    targetType: Optional[str] = None

class Flux2TryonUserImage(BaseModel):
    tryonImage: str
    promptDescription: Optional[str] = None

class Flux2TryonRequest(BaseModel):
    products: List[Flux2TryonProduct]
    user_image: Flux2TryonUserImage
    description_backend: Optional[str] = None
    description_compare: bool = False
    negative_prompt: Optional[str] = None
    steps: int = Field(default=6, ge=4, le=30)
    seed: int = Field(default=23, ge=0, le=2147483647)

class ParserJoyCaptionAnalyzeRequest(BaseModel):
    image_url: str
    garment_type: Optional[str] = None
    selected_index: Optional[int] = Field(default=None, ge=0)
    use_unified_square_split: bool = True
    square_padding_ratio: float = Field(default=0.12, ge=0.0, le=0.60)
    min_component_area_ratio: float = Field(default=0.01, ge=0.0005, le=0.25)
    upload_candidate_previews: bool = True
    run_flux_garment_only: bool = False
    flux_steps: int = Field(default=8, ge=4, le=30)
    flux_seed: int = Field(default=23, ge=0, le=2147483647)
    flux_extract_only: bool = False
    flux_extract_strict_safety: bool = True
    adaptive_rect_crop: bool = False

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
        "adaptive_crop_s": 0.0,
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
            if bool(request.use_unified_square_split):
                candidates = _build_parser_unified_square_split_candidates(
                    image=source_img,
                    requested_type=requested_type,
                    min_component_area_ratio=request.min_component_area_ratio,
                    square_padding_ratio=request.square_padding_ratio,
                )
            else:
                candidates = _build_parser_square_candidates(
                    image=source_img,
                    requested_type=requested_type,
                    min_component_area_ratio=request.min_component_area_ratio,
                    square_padding_ratio=request.square_padding_ratio,
                )
            stage_timings["parser_detect_s"] = round(time.time() - t_stage, 4)

            if not candidates:
                raise HTTPException(status_code=400, detail="No garment component found for the requested type")

            context_candidates = list(candidates)
            if requested_type in {"top", "bottom", "dress", "outer"}:
                requested_filtered = [
                    c for c in candidates if _normalize_garment_type(c.get("type")) == requested_type
                ]
                if requested_filtered:
                    candidates = requested_filtered
                else:
                    raise HTTPException(status_code=400, detail="No garment component found for the requested type")

            preview_upload_start = time.time()
            if request.upload_candidate_previews:
                for candidate in candidates:
                    preview_image = candidate.get("_preview_image") or candidate.get("_crop_image")
                    if preview_image is None:
                        continue
                    crop_buf = io.BytesIO()
                    preview_image.save(crop_buf, format="PNG")
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
            selected_type = str(selected.get("type") or requested_type or "top")
            selected_category_text = str(
                selected.get("category_text") or _parser_candidate_category_text(selected_type, str(selected.get("subtype") or ""))
            ).strip()
            selected_crop = selected.get("_crop_image")
            if selected_crop is None:
                sx0, sy0, sx1, sy1 = [int(v) for v in selected.get("square_bbox", [0, 0, source_img.width, source_img.height])]
                selected_crop = _crop_square_with_padding(source_img, [sx0, sy0, sx1, sy1])

            adaptive_variants = [
                {
                    "name": "default",
                    "bbox": [int(v) for v in (selected.get("section_bbox") or selected.get("bbox") or [0, 0, source_img.width, source_img.height])],
                    "image": selected_crop,
                }
            ]
            unified_context_image = selected.get("_context_crop_image")
            if bool(request.use_unified_square_split) and isinstance(unified_context_image, Image.Image):
                adaptive_variants = [
                    {
                        "name": "split_mask",
                        "bbox": [int(v) for v in (selected.get("top_object_bbox") or selected.get("bbox") or selected.get("section_bbox") or [0, 0, source_img.width, source_img.height])],
                        "image": selected_crop,
                    },
                    {
                        "name": "square_context",
                        "bbox": [int(v) for v in (selected.get("square_bbox") or selected.get("section_bbox") or [0, 0, source_img.width, source_img.height])],
                        "image": unified_context_image.convert("RGB"),
                    },
                ]
            elif bool(request.adaptive_rect_crop):
                t_adaptive = time.time()
                built = _build_adaptive_rect_crop_variants(
                    full_image=source_img,
                    selected_candidate=selected,
                    garment_type=selected_type,
                    all_candidates=context_candidates,
                )
                stage_timings["adaptive_crop_s"] = round(time.time() - t_adaptive, 4)
                if built:
                    adaptive_variants = built

            joy_instruction = (
                f"The image is a section crop with person context. Describe only the selected {selected_category_text} garment "
                "for virtual try-on, not the person/background. "
                "Return a single detailed line with exact attributes: garment type, silhouette, fit, neckline, sleeves, "
                "waistline, hem shape/length, fabric, texture, transparency, pattern, embellishments, and dominant colors. "
                "Do not repeat words or phrases."
            )
            t_stage = time.time()
            prompt_description = ""
            best_variant_name = adaptive_variants[0]["name"]
            best_variant_score = float("-inf")
            best_variant_explain: dict = {}
            adaptive_variant_details: list[dict] = []
            for variant in adaptive_variants:
                v_img = variant.get("image")
                if not isinstance(v_img, Image.Image):
                    continue
                caption, caption_mode = _joycaption_describe_with_retry(
                    v_img,
                    instruction_primary=joy_instruction,
                    garment_type=selected_type,
                )
                if not caption:
                    adaptive_variant_details.append(
                        {
                            "name": str(variant.get("name")),
                            "bbox": [int(v) for v in (variant.get("bbox") or [])],
                            "score": None,
                            "caption": "",
                            "meta": {"empty_caption": True, "caption_mode": caption_mode},
                        }
                    )
                    continue
                score, score_meta = _score_adaptive_crop_caption(caption, selected_type)
                score_meta["caption_mode"] = caption_mode
                adaptive_variant_details.append(
                    {
                        "name": str(variant.get("name")),
                        "bbox": [int(v) for v in (variant.get("bbox") or [])],
                        "score": round(float(score), 4),
                        "caption": caption,
                        "meta": score_meta,
                    }
                )
                if score > best_variant_score:
                    best_variant_score = float(score)
                    best_variant_name = str(variant.get("name"))
                    best_variant_explain = score_meta
                    prompt_description = caption
                    selected_crop = v_img
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
            selected_preview_image = selected.get("_preview_image") or selected_crop
            selected_buf = io.BytesIO()
            selected_preview_image.save(selected_buf, format="PNG")
            selected_crop_url = _upload_or_raise(selected_buf.getvalue(), container=VTO_OUTPUT_CONTAINER)
            selected_object_cutout_url = None
            object_cutout_image = selected.get("_object_cutout_image")
            if isinstance(object_cutout_image, Image.Image):
                cutout_buf = io.BytesIO()
                object_cutout_image.save(cutout_buf, format="PNG")
                selected_object_cutout_url = _upload_or_raise(cutout_buf.getvalue(), container=VTO_OUTPUT_CONTAINER)
            selected_context_crop_url = None
            context_crop_image = selected.get("_context_crop_image")
            if isinstance(context_crop_image, Image.Image):
                ctx_buf = io.BytesIO()
                context_crop_image.save(ctx_buf, format="PNG")
                selected_context_crop_url = _upload_or_raise(ctx_buf.getvalue(), container=VTO_OUTPUT_CONTAINER)
            adaptive_variant_previews: list[dict] = []
            if bool(request.adaptive_rect_crop) and bool(request.upload_candidate_previews):
                for variant in adaptive_variants:
                    v_img = variant.get("image")
                    if not isinstance(v_img, Image.Image):
                        continue
                    v_buf = io.BytesIO()
                    v_img.save(v_buf, format="PNG")
                    adaptive_variant_previews.append(
                        {
                            "name": str(variant.get("name")),
                            "bbox": [int(v) for v in (variant.get("bbox") or [])],
                            "preview_url": _upload_or_raise(v_buf.getvalue(), container=VTO_OUTPUT_CONTAINER),
                        }
                    )
            stage_timings["selected_upload_s"] = round(time.time() - selected_upload_start, 4)

            flux_prompt = _build_flux2_garment_only_prompt(
                selected_type,
                prompt_description,
                category_text=selected_category_text,
                dominant_color_hexes=[str(v) for v in (selected.get("dominant_color_hexes") or [])],
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
                    "adaptive_crop_enabled": bool(request.adaptive_rect_crop),
                    "adaptive_crop_selected": best_variant_name,
                    "adaptive_crop_score": None if best_variant_score == float("-inf") else round(float(best_variant_score), 4),
                    "adaptive_crop_meta": best_variant_explain or {},
                    "adaptive_crop_variants": adaptive_variant_details,
                    "adaptive_crop_variant_previews": adaptive_variant_previews,
                    "object_cutout_url": selected_object_cutout_url,
                    "context_crop_url": selected_context_crop_url,
                    "strategy": "unified_square_split" if bool(request.use_unified_square_split) else "component_first",
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
    request_id = str(uuid.uuid4())
    descriptor_backend = _normalize_descriptor_backend(request.description_backend)
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
    input_summary = {
        "userImage": "",
        "products": [],
        "productCount": 0,
        "steps": int(request.steps),
        "seed": int(request.seed),
        "descriptionBackendRequested": str(request.description_backend or ""),
        "descriptionBackendResolved": descriptor_backend,
        "descriptionCompareRequested": bool(request.description_compare),
        "hasCustomNegativePrompt": bool(str(request.negative_prompt or "").strip()),
    }

    def _build_tryon_error_payload(
        *,
        code: str,
        message: str,
        status_code: int,
        extra: Optional[dict] = None,
    ) -> dict:
        stage_timings["api_total_s"] = round(time.time() - t0, 4)
        payload = {
            "status": "error",
            "requestId": request_id,
            "error": {
                "code": str(code or "TRYON_ERROR"),
                "message": str(message or "Try-on request failed"),
                "statusCode": int(status_code),
            },
            "meta": {
                "descriptionBackend": descriptor_backend,
                "stageTimings": stage_timings,
                "input": input_summary,
            },
        }
        if isinstance(extra, dict) and extra:
            payload["error"]["details"] = extra
        return payload

    try:
        if not request.products:
            raise HTTPException(status_code=422, detail="products must contain at least one item")

        user_image_url = str(request.user_image.tryonImage or "").strip()
        if not user_image_url:
            raise HTTPException(status_code=422, detail="user_image.tryonImage is required")
        input_summary["userImage"] = user_image_url

        product_urls: List[str] = []
        for idx, product in enumerate(request.products):
            product_url = str(product.image or "").strip()
            if not product_url:
                raise HTTPException(status_code=422, detail=f"products[{idx}].image is required")
            product_urls.append(product_url)
            input_summary["products"].append(
                {
                    "index": idx,
                    "image": product_url,
                    "promptProvided": bool(str(product.promptDescription or "").strip()),
                    "targetTypeProvided": bool(str(product.targetType or "").strip()),
                    "targetTypeValue": (str(product.targetType or "").strip() or None),
                }
            )
        input_summary["productCount"] = len(product_urls)

        # 1. Download Images
        t_stage = time.time()
        user_img = download_image(user_image_url)
        stage_timings["download_user_image_s"] = round(time.time() - t_stage, 4)

        t_stage = time.time()
        product_imgs = [download_image(url) for url in product_urls]
        stage_timings["download_products_s"] = round(time.time() - t_stage, 4)
        descriptor_compare_enabled = bool(request.description_compare or FLUX2_DESCRIPTOR_COMPARE)
        compare_candidates = ["florence", "joycaption", "minicpm", "minicpm_service"]
        if FLUX2_ALLOW_QWEN_BACKEND:
            compare_candidates.append("qwen2_5_vl")
        compare_candidates.append(descriptor_backend)
        compare_backends: tuple[str, ...] = tuple(dict.fromkeys(compare_candidates))
        descriptor_comparisons = {"products": [], "user_image": {}} if descriptor_compare_enabled else None

        async with gpu_semaphore:
            # 2. Resolve product prompt descriptions (request-provided or selected model backend)
            product_descriptions: List[str] = []
            product_target_types: List[str] = []
            product_target_type_sources: List[str] = []
            generated_product_prompt_indices: List[int] = []
            for idx, (product, product_img) in enumerate(zip(request.products, product_imgs)):
                provided_prompt = str(product.promptDescription or "").strip()
                requested_type_raw = str(product.targetType or "").strip()
                requested_target_type = _normalize_garment_type(requested_type_raw) if requested_type_raw else None
                if requested_type_raw and requested_target_type is None:
                    raise HTTPException(
                        status_code=422,
                        detail=f"products[{idx}].targetType must be one of top, bottom, dress, outer",
                    )
                resolved_target_type = requested_target_type or "top"
                item_prompt_gen_s = 0.0
                if provided_prompt:
                    cleaned_desc = _sanitize_florence_garment_description(provided_prompt)
                else:
                    generated_product_prompt_indices.append(idx)
                    t_desc = time.time()
                    generated_desc = _describe_garment_with_backend(
                        product_img,
                        descriptor_backend,
                        image_url=product_urls[idx],
                    )
                    item_prompt_gen_s += time.time() - t_desc
                    cleaned_desc = _sanitize_florence_garment_description(generated_desc)
                if requested_target_type is None:
                    resolved_target_type = _infer_flux2_target_type(cleaned_desc)
                if (
                    requested_target_type is None
                    and not provided_prompt
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
                product_target_type_sources.append("request" if requested_target_type else "inferred")
                if descriptor_compare_enabled and not provided_prompt:
                    compare_item = {}
                    for backend_name in compare_backends:
                        try:
                            if backend_name == descriptor_backend:
                                compare_item[backend_name] = cleaned_desc
                            else:
                                alt_desc = _describe_garment_with_backend(
                                    product_img,
                                    backend_name,
                                    image_url=product_urls[idx],
                                )
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
                user_prompt_raw = _describe_user_image_for_flux2(
                    user_img,
                    backend=descriptor_backend,
                    image_url=user_image_url,
                )
                stage_timings["user_prompt_generation_s"] = round(time.time() - t_desc, 4)
                user_prompt_generated = True
            if descriptor_compare_enabled and user_prompt_generated:
                compare_user = {}
                for backend_name in compare_backends:
                    try:
                        if backend_name == descriptor_backend:
                            compare_user[backend_name] = user_prompt_raw
                        else:
                            compare_user[backend_name] = _describe_user_image_for_flux2(
                                user_img,
                                backend=backend_name,
                                image_url=user_image_url,
                            )
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
            collage_item_clause = _build_flux2_collage_item_clause(
                garment_descriptions=product_descriptions,
                target_types=product_target_types,
            )
            custom_negative_prompt = str(request.negative_prompt or "").strip()
            negative_prompt_source = (
                "request"
                if custom_negative_prompt
                else ("default" if FLUX2_NEGATIVE_PROMPT_ENABLE and FLUX2_NEGATIVE_PROMPT_DEFAULT else "auto")
            )
            negative_prompt = _build_flux2_negative_prompt(
                target_types=product_target_types,
                board_mode=board_mode,
                custom_negative_prompt=custom_negative_prompt,
            )
            runtime_negative_prompt = _build_flux2_runtime_negative_prompt(
                target_types=product_target_types,
                board_mode=board_mode,
                custom_negative_prompt=custom_negative_prompt,
            )
            if custom_negative_prompt and runtime_negative_prompt:
                runtime_negative_prompt_source = "request"
            elif runtime_negative_prompt:
                runtime_negative_prompt_source = "auto"
            else:
                runtime_negative_prompt_source = "none"

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
                collage_item_clause=collage_item_clause,
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
                    negative_prompt=runtime_negative_prompt,
                )
                fidelity_score, output_desc = _score_tryon_garment_fidelity(
                    output_image=candidate_result["image"],
                    target_description=target_desc,
                    descriptor_backend=FLUX2_FIDELITY_BACKEND,
                )
                preservation_score = _score_untargeted_region_preservation(
                    reference_image=user_img,
                    output_image=candidate_result["image"],
                    target_types=product_target_types,
                )
                candidate_runs.append({
                    "label": label,
                    "steps": candidate_steps,
                    "seed": candidate_seed,
                    "latency": float(candidate_result["latency"]),
                    "fidelity_score": float(fidelity_score),
                    "preservation_score": float(preservation_score),
                    "output_description": output_desc,
                    "negativePromptMode": str(candidate_result.get("metadata", {}).get("negative_prompt_mode", "unknown")),
                    "negativePromptSupported": bool(candidate_result.get("metadata", {}).get("negative_prompt_supported", False)),
                })
                return candidate_result

            is_single_dress = (
                board_mode == "single"
                and len(product_target_types) == 1
                and product_target_types[0] == "dress"
            )
            is_single_item = board_mode == "single" and len(product_target_types) == 1
            is_single_top_or_bottom = (
                is_single_item and product_target_types[0] in {"top", "bottom"}
            )
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

            region_lock_prompt = prompt
            if is_single_top_or_bottom:
                if product_target_types[0] == "top":
                    region_lock_prompt += (
                        " Strict untargeted-region lock: keep pants/skirt/shorts/shoes exactly unchanged from image 1, "
                        "including color, folds, silhouette, hem shape, and shading. "
                        "Do not alter any pixels below waistline except unavoidable edge blending."
                    )
                else:
                    region_lock_prompt += (
                        " Strict untargeted-region lock: keep top/outerwear exactly unchanged from image 1, "
                        "including color, folds, neckline, sleeve shape, and shading. "
                        "Do not alter any pixels above waistline except unavoidable edge blending."
                    )
            region_lock_steps = min(
                FLUX2_REGION_LOCK_SECOND_PASS_MAX_STEPS,
                max(request.steps, request.steps + FLUX2_REGION_LOCK_SECOND_PASS_EXTRA_STEPS),
            )
            region_lock_seed = min(2147483647, request.seed + 5)

            def _candidate_rank(idx: int) -> float:
                run = candidate_runs[idx]
                fid = float(run.get("fidelity_score", 0.0))
                preserve = float(run.get("preservation_score", 0.0))
                if is_single_top_or_bottom:
                    return (0.45 * fid) + (0.55 * preserve)
                return fid

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

                    base_score = _candidate_rank(0)
                    strict_score = _candidate_rank(1)
                    if strict_score >= base_score:
                        result = strict_result
                        selected_candidate_index = 1
                        selected_prompt = dress_strict_prompt

                if is_single_top_or_bottom and FLUX2_REGION_LOCK_SECOND_PASS_ENABLED:
                    region_result = _run_candidate(
                        candidate_prompt=region_lock_prompt,
                        candidate_steps=region_lock_steps,
                        candidate_seed=region_lock_seed,
                        label="region_lock_strict",
                    )
                    region_index = len(candidate_runs) - 1
                    current_best = _candidate_rank(selected_candidate_index)
                    region_score = _candidate_rank(region_index)
                    if region_score >= current_best:
                        result = region_result
                        selected_candidate_index = region_index
                        selected_prompt = region_lock_prompt

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
                    current_best_score = _candidate_rank(selected_candidate_index)
                    qwen_score = _candidate_rank(qwen_index)
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
            f"desc_products={stage_timings['product_prompt_generation_total_s']}s "
            f"desc_user={stage_timings['user_prompt_generation_s']}s "
            f"flux_sum={stage_timings['flux_generation_sum_s']}s "
            f"upload={stage_timings['upload_s']}s "
            f"backend={descriptor_backend} "
            f"product_count={len(product_descriptions)} "
            f"user_prompt_generated={user_prompt_generated} "
            f"negative_source={negative_prompt_source}"
        )

        return {
            "status": "success",
            "requestId": request_id,
            "result_url": result_url,
            "promptDescription": " | ".join(product_descriptions),
            "productPromptDescriptions": product_descriptions,
            "userPromptDescription": user_prompt_description,
            "productColorHints": visual_locks.get("color_hints", []),
            "productDetailHints": visual_locks.get("detail_terms", []),
            "transparencyLockApplied": bool(visual_locks.get("transparency_lock")),
            "prompt": selected_prompt,
            "negativePrompt": negative_prompt,
            "negativePromptSource": negative_prompt_source,
            "negativePromptRuntimeSource": runtime_negative_prompt_source,
            "negativePromptRuntimeInput": runtime_negative_prompt,
            "negativePromptRuntimeMode": FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE,
            "negativePromptAppliedMode": str(result.get("metadata", {}).get("negative_prompt_mode", "unknown")),
            "negativePromptPipelineSupported": bool(result.get("metadata", {}).get("negative_prompt_supported", False)),
            "collageItemMapping": collage_item_clause,
            "latency": result["latency"],
            "total_latency": total_latency,
            "boardMode": board_mode,
            "productTargetTypes": product_target_types,
            "productTargetTypeSources": product_target_type_sources,
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
            # Staging-friendly structured trace for DB persistence.
            "input": input_summary,
            "processing": {
                "descriptorBackend": descriptor_backend,
                "fidelityBackend": FLUX2_FIDELITY_BACKEND,
                "boardMode": board_mode,
                "productTargetTypes": product_target_types,
                "productTargetTypeSources": product_target_type_sources,
                "generatedProductPromptIndices": generated_product_prompt_indices,
                "userPromptGenerated": user_prompt_generated,
                "singleCandidateMode": FLUX2_SINGLE_CANDIDATE_MODE,
                "singleCandidateApplied": len(candidate_runs) == 1,
                "selectedCandidateIndex": selected_candidate_index,
                "descriptionCompareEnabled": descriptor_compare_enabled,
                "descriptionComparisons": descriptor_comparisons,
                "collageItemMapping": collage_item_clause,
                "transparencyLockApplied": bool(visual_locks.get("transparency_lock")),
                "negativePrompt": {
                    "source": negative_prompt_source,
                    "resolved": negative_prompt,
                    "runtimeMode": FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE,
                    "runtimeSource": runtime_negative_prompt_source,
                    "runtimeInput": runtime_negative_prompt,
                    "appliedMode": str(result.get("metadata", {}).get("negative_prompt_mode", "unknown")),
                    "pipelineSupported": bool(result.get("metadata", {}).get("negative_prompt_supported", False)),
                },
            },
            "timings": stage_timings,
        }
    except HTTPException as he:
        detail = he.detail
        if isinstance(detail, dict) and detail.get("status") == "error":
            # Already structured.
            raise
        message = detail.get("message") if isinstance(detail, dict) else str(detail)
        code = detail.get("code") if isinstance(detail, dict) else "TRYON_REQUEST_INVALID"
        raise HTTPException(
            status_code=he.status_code,
            detail=_build_tryon_error_payload(
                code=str(code or "TRYON_REQUEST_INVALID"),
                message=message,
                status_code=he.status_code,
                extra=(detail if isinstance(detail, dict) else None),
            ),
        )
    except Exception as e:
        logger.error(f"Try-on failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=_build_tryon_error_payload(
                code="TRYON_INTERNAL_ERROR",
                message=str(e),
                status_code=500,
            ),
        )

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
                "flux2_region_lock_second_pass_enabled": FLUX2_REGION_LOCK_SECOND_PASS_ENABLED,
                "flux2_region_lock_second_pass_extra_steps": FLUX2_REGION_LOCK_SECOND_PASS_EXTRA_STEPS,
                "flux2_region_lock_second_pass_max_steps": FLUX2_REGION_LOCK_SECOND_PASS_MAX_STEPS,
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
                "flux2_preload_minicpm_with_flux2": FLUX2_PRELOAD_MINICPM_WITH_FLUX2,
                "flux2_minicpm_product_caption_max_side": FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                "flux2_minicpm_product_caption_min_side": FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
                "flux2_minicpm_user_caption_max_side": FLUX2_MINICPM_USER_CAPTION_MAX_SIDE,
                "flux2_minicpm_user_caption_min_side": FLUX2_MINICPM_USER_CAPTION_MIN_SIDE,
                "flux2_single_candidate_mode": FLUX2_SINGLE_CANDIDATE_MODE,
                "flux2_color_lock_enabled": FLUX2_COLOR_LOCK_ENABLED,
                "flux2_color_lock_top_k": FLUX2_COLOR_LOCK_TOP_K,
                "flux2_detail_lock_enabled": FLUX2_DETAIL_LOCK_ENABLED,
                "flux2_allow_qwen_backend": FLUX2_ALLOW_QWEN_BACKEND,
                "flux2_negative_prompt_enable": FLUX2_NEGATIVE_PROMPT_ENABLE,
                "flux2_negative_prompt_default_len": len(FLUX2_NEGATIVE_PROMPT_DEFAULT),
                "flux2_negative_prompt_runtime_mode": FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE,
                "minicpm_service_url": MINICPM_SERVICE_URL,
                "minicpm_service_timeout_s": MINICPM_SERVICE_TIMEOUT_S,
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
