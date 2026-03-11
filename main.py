import io
import time
import logging
import asyncio
import os
import tempfile
import uuid
import re
import hashlib
import colorsys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
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
from ai.shared.image_ops import (
    download_image,
    bbox_iou,
    resize_mask_to_image,
    binary_open,
    binary_close,
    build_soft_alpha,
    rgb_to_lab,
    delta_e_cie76,
)
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
from ai.modules.wardrobe.cloth_detection import ClothDetector
from ai.modules.wardrobe.yolo_cropper import YoloCropper
from ai.modules.wardrobe.human_parser import HumanParser
from ai.modules.wardrobe.extraction.detection_stage import (
    build_selection_required_response as _build_extraction_selection_required_response,
    resolve_selected_item_or_response,
    run_detection_stage_or_response,
)
from ai.modules.wardrobe.extraction.generation_stage import run_selected_item_extraction_or_response
from ai.modules.wardrobe.extraction.pipeline import default_extraction_stage_timings
from ai.modules.wardrobe.extraction.postprocess_stage import sync_selected_item_progress
from ai.modules.wardrobe.extraction.prompting_stage import apply_selected_item_prompting
from ai.modules.wardrobe.extraction.utils import (
    build_success_response as _build_extraction_success_response,
    read_input_image_or_response,
    resolve_upload_or_response,
    verify_authorization_or_response,
)

# Core Models (Heavy Runners)
from ai.core.flux2_cvton_runner import Flux2CVTONRunner
from ai.core.florence_runner import FlorenceRunner
from ai.core.qwen25vl_runner import Qwen25VLRunner
from ai.core.joycaption_runner import JoyCaptionRunner
from ai.core.minicpm_runner import MiniCPMVRunner
from ai.core.fashion_detection_runner import FashionDetectionRunner
from ai.core.yolo_runner import YoloRunner
from ai.core.human_parser_runner import HumanParserRunner
from ai.core.openclip_runner import OpenCLIPRunner
from ai.core.garment_extractor import (
    GarmentExtractionConfig,
    GarmentExtractionRequest,
    GarmentExtractor,
)
from ai.core.garment_color_masker import GarmentColorMasker
from ai.core.garment_color_context import (
    GarmentColorContextSettings,
    build_single_image_color_context as _shared_build_single_image_color_context,
    build_visual_lock_clauses as _shared_build_visual_lock_clauses,
)

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
FLUX2_DESCRIPTOR_BACKEND = os.getenv("FLUX2_DESCRIPTOR_BACKEND", "minicpm").strip().lower()
if FLUX2_DESCRIPTOR_BACKEND not in {"florence", "qwen2_5_vl", "joycaption", "minicpm", "minicpm_service"}:
    FLUX2_DESCRIPTOR_BACKEND = "minicpm"
if FLUX2_DESCRIPTOR_BACKEND == "qwen2_5_vl" and not FLUX2_ALLOW_QWEN_BACKEND:
    FLUX2_DESCRIPTOR_BACKEND = "minicpm"
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
        "extra hand, third hand, duplicate arms, hand fused to garment, garment fused to skin, "
        "identity change, different face, wrong skin tone, recolored garment, hue shift, "
        "color drift, pattern drift, texture swap, logo/text watermark, duplicate garment, "
        "layering artifacts, garment merge, ghost garment, incorrect neckline, incorrect hemline"
    ),
).strip()
FLUX2_LOW_LATENCY_MODE = os.getenv("FLUX2_LOW_LATENCY_MODE", "0") == "1"
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
ANALYZE_MINICPM_SERVICE_URL = os.getenv("ANALYZE_MINICPM_SERVICE_URL", "").strip().rstrip("/")
MINICPM_SERVICE_TIMEOUT_S = max(5, _env_int("MINICPM_SERVICE_TIMEOUT_S", 60 if FLUX2_LOW_LATENCY_MODE else 120))
MINICPM_SERVICE_CONNECT_TIMEOUT_S = max(
    1.0, _env_float("MINICPM_SERVICE_CONNECT_TIMEOUT_S", 6.0 if FLUX2_LOW_LATENCY_MODE else 10.0)
)
MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS = max(
    32, _env_int("MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS", 180 if FLUX2_LOW_LATENCY_MODE else 180)
)
MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS = max(
    32, _env_int("MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS", 96 if FLUX2_LOW_LATENCY_MODE else 140)
)
MINICPM_SERVICE_CACHE_ENABLED = os.getenv("MINICPM_SERVICE_CACHE_ENABLED", "1") == "1"
MINICPM_SERVICE_CACHE_TTL_SECONDS = max(
    0, _env_int("MINICPM_SERVICE_CACHE_TTL_SECONDS", 1800 if FLUX2_LOW_LATENCY_MODE else 900)
)
MINICPM_SERVICE_CACHE_MAX_ENTRIES = max(16, _env_int("MINICPM_SERVICE_CACHE_MAX_ENTRIES", 1024))
MINICPM_SERVICE_POOL_MAXSIZE = max(4, _env_int("MINICPM_SERVICE_POOL_MAXSIZE", 16))
MINICPM_SERVICE_LOCAL_FILE_FIRST = os.getenv("MINICPM_SERVICE_LOCAL_FILE_FIRST", "1") == "1"
MINICPM_SERVICE_GARMENT_MIN_WORDS = max(4, _env_int("MINICPM_SERVICE_GARMENT_MIN_WORDS", 10))
MINICPM_SERVICE_GARMENT_PROMPT = os.getenv(
    "MINICPM_SERVICE_GARMENT_PROMPT",
    (
        "Describe only the product garment for high-fidelity virtual try-on. "
        "Describe exactly one garment only and never describe any person, skin, hair, hands, legs, pose, room, props, or background objects. "
        "Ignore studio/background pixels, alpha-matte edges, and lighting shadows outside the garment fabric. "
        "Report garment fabric color only (dominant first, then secondary undertone if visible). "
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
        "If the requested garment type is known, keep category/type locked to that garment only and ignore any other clothing visible in the crop. "
        "Use unknown when not visible."
    ),
).strip()
MINICPM_SERVICE_PERSON_PROMPT = os.getenv(
    "MINICPM_SERVICE_PERSON_PROMPT",
    (
        "Describe only identity and scene context for identity-preserving virtual try-on. "
        "Return one detailed line with schema: "
        "identity=<face traits, skin tone, hair style/color, age band>; "
        "body_pose=<pose, camera angle, visible limbs>; "
        "framing_lighting=<framing/crop, light direction/intensity, background>; "
        "occlusion=<hair/hands/objects overlapping garment region>; "
        "preserve=<face identity, skin tone, hair, body proportions, pose, background unchanged>. "
        "Do not describe outfit details unless they directly occlude the target garment region. "
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
FLUX2_COLOR_DECONTAMINATION_ENABLED = os.getenv("FLUX2_COLOR_DECONTAMINATION_ENABLED", "1") == "1"
FLUX2_COLOR_DECONTAM_ALPHA_HIGH = max(1, min(255, _env_int("FLUX2_COLOR_DECONTAM_ALPHA_HIGH", 240)))
FLUX2_COLOR_DECONTAM_ALPHA_LOW = max(1, min(255, _env_int("FLUX2_COLOR_DECONTAM_ALPHA_LOW", 200)))
FLUX2_COLOR_DECONTAM_ERODE_ITERS = max(0, _env_int("FLUX2_COLOR_DECONTAM_ERODE_ITERS", 1))
FLUX2_COLOR_DECONTAM_MIN_PIXELS = max(16, _env_int("FLUX2_COLOR_DECONTAM_MIN_PIXELS", 64))
FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO = min(
    0.2,
    max(0.0, _env_float("FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO", 0.006)),
)
FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT = min(
    40.0,
    max(0.0, _env_float("FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT", 7.0)),
)
FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE = min(
    40.0,
    max(0.0, _env_float("FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE", 10.0)),
)
FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE = min(
    100.0,
    max(60.0, _env_float("FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE", 98.0)),
)
FLUX2_COLOR_GUARD_RERUN_ENABLED = os.getenv("FLUX2_COLOR_GUARD_RERUN_ENABLED", "1") == "1"
FLUX2_COLOR_GUARD_DRIFT_THRESHOLD = _env_float("FLUX2_COLOR_GUARD_DRIFT_THRESHOLD", 12.0)
FLUX2_COLOR_GUARD_RERUN_EXTRA_STEPS = max(1, _env_int("FLUX2_COLOR_GUARD_RERUN_EXTRA_STEPS", 2))
FLUX2_COLOR_GUARD_RERUN_MAX_STEPS = max(6, _env_int("FLUX2_COLOR_GUARD_RERUN_MAX_STEPS", 18))
FLUX2_COLOR_FIRST_GATE_ENABLED = os.getenv("FLUX2_COLOR_FIRST_GATE_ENABLED", "1") == "1"
FLUX2_COLOR_FIRST_GATE_MAX_DRIFT_DELTA = max(
    0.0,
    _env_float("FLUX2_COLOR_FIRST_GATE_MAX_DRIFT_DELTA", 6.0),
)
FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED = (
    os.getenv("FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED", "1") == "1"
)
FLUX2_NEUTRAL_CALIBRATION_MAX_DELTA_L = max(1.0, _env_float("FLUX2_NEUTRAL_CALIBRATION_MAX_DELTA_L", 10.0))
FLUX2_NEUTRAL_CALIBRATION_LIGHT_OUTPUT_MIN_L = _env_float("FLUX2_NEUTRAL_CALIBRATION_LIGHT_OUTPUT_MIN_L", 65.0)
FLUX2_NEUTRAL_CALIBRATION_MAX_DARKEN_LIGHT_OUTPUT = max(
    0.5, _env_float("FLUX2_NEUTRAL_CALIBRATION_MAX_DARKEN_LIGHT_OUTPUT", 5.0)
)
FLUX2_NEUTRAL_CALIBRATION_DARK_OUTPUT_MAX_L = _env_float("FLUX2_NEUTRAL_CALIBRATION_DARK_OUTPUT_MAX_L", 40.0)
FLUX2_NEUTRAL_CALIBRATION_MAX_BRIGHTEN_DARK_OUTPUT = max(
    0.5, _env_float("FLUX2_NEUTRAL_CALIBRATION_MAX_BRIGHTEN_DARK_OUTPUT", 15.0)
)
FLUX2_NEUTRAL_CALIBRATION_BRIGHTNESS_MARGIN_L = max(
    0.0, _env_float("FLUX2_NEUTRAL_CALIBRATION_BRIGHTNESS_MARGIN_L", 15.0)
)
FLUX2_NEUTRAL_CALIBRATION_LIGHT_SOURCE_MIN_L = _env_float("FLUX2_NEUTRAL_CALIBRATION_LIGHT_SOURCE_MIN_L", 60.0)
FLUX2_NEUTRAL_CALIBRATION_LIGHT_SOURCE_MAX_DARKEN = max(
    0.5, _env_float("FLUX2_NEUTRAL_CALIBRATION_LIGHT_SOURCE_MAX_DARKEN", 15.0)
)
FLUX2_NEUTRAL_CALIBRATION_DARK_SOURCE_MAX_L = _env_float("FLUX2_NEUTRAL_CALIBRATION_DARK_SOURCE_MAX_L", 40.0)
FLUX2_NEUTRAL_CALIBRATION_DARK_SOURCE_MAX_BRIGHTEN = max(
    0.5, _env_float("FLUX2_NEUTRAL_CALIBRATION_DARK_SOURCE_MAX_BRIGHTEN", 20.0)
)
FLUX2_DETAIL_LOCK_ENABLED = os.getenv("FLUX2_DETAIL_LOCK_ENABLED", "1") == "1"
FLUX2_SINGLE_CANDIDATE_MODE = os.getenv("FLUX2_SINGLE_CANDIDATE_MODE", "auto").strip().lower()
if FLUX2_SINGLE_CANDIDATE_MODE not in {"auto", "base", "dress_strict", "qwen_strict"}:
    FLUX2_SINGLE_CANDIDATE_MODE = "auto"
if FLUX2_LOW_LATENCY_MODE and "FLUX2_SINGLE_CANDIDATE_MODE" not in os.environ:
    FLUX2_SINGLE_CANDIDATE_MODE = "base"
if FLUX2_LOW_LATENCY_MODE and "FLUX2_DRESS_SECOND_PASS_ENABLED" not in os.environ:
    FLUX2_DRESS_SECOND_PASS_ENABLED = False
if FLUX2_LOW_LATENCY_MODE and "FLUX2_REGION_LOCK_SECOND_PASS_ENABLED" not in os.environ:
    FLUX2_REGION_LOCK_SECOND_PASS_ENABLED = False
if FLUX2_LOW_LATENCY_MODE and "FLUX2_QWEN_SECOND_PASS_ENABLED" not in os.environ:
    FLUX2_QWEN_SECOND_PASS_ENABLED = False
if FLUX2_LOW_LATENCY_MODE and "FLUX2_COLOR_GUARD_RERUN_ENABLED" not in os.environ:
    FLUX2_COLOR_GUARD_RERUN_ENABLED = False
FLUX2_RUNTIME_SCORING_ENABLED = os.getenv(
    "FLUX2_RUNTIME_SCORING_ENABLED",
    "0" if FLUX2_LOW_LATENCY_MODE else "1",
) == "1"
FLUX2_FORCE_RUNTIME_SCORING_FOR_SINGLE_CANDIDATE = os.getenv(
    "FLUX2_FORCE_RUNTIME_SCORING_FOR_SINGLE_CANDIDATE",
    "0",
) == "1"
FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND = os.getenv(
    "FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND",
    "minicpm",
).strip().lower()
if FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND not in {"florence", "joycaption", "minicpm", "minicpm_service"}:
    FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND = "minicpm"
FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS = max(4, _env_int("FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS", 10))
FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED = max(0, _env_int("FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED", 23))
FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_DEBUG = os.getenv("FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_DEBUG", "0") == "1"
FLUX2_SINGLE_GARMENT_EXTRACT_DISABLE_PARSER = os.getenv(
    "FLUX2_SINGLE_GARMENT_EXTRACT_DISABLE_PARSER",
    "1",
) == "1"
FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_RAW_DEBUG = os.getenv(
    "FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_RAW_DEBUG",
    "0",
) == "1"
FLUX2_SINGLE_GARMENT_EXTRACT_EXPOSE_INTERMEDIATE_URLS = os.getenv(
    "FLUX2_SINGLE_GARMENT_EXTRACT_EXPOSE_INTERMEDIATE_URLS",
    "0",
) == "1"
HYBRID_TOP_K = max(1, _env_int("HYBRID_TOP_K", 3))
ANALYZE_MAX_ITEMS = max(1, _env_int("ANALYZE_MAX_ITEMS", 3))
HYBRID_MIN_SCORE = _env_float("HYBRID_MIN_SCORE", 0.0)
HYBRID_WEIGHT_YOLO = _env_float("HYBRID_WEIGHT_YOLO", 0.45)
HYBRID_WEIGHT_FLORENCE = _env_float("HYBRID_WEIGHT_FLORENCE", 0.45)
HYBRID_WEIGHT_BBOX = _env_float("HYBRID_WEIGHT_BBOX", 0.10)
ANALYZE_REQUIRE_SELECTION = os.getenv("ANALYZE_REQUIRE_SELECTION", "1") == "1"
ANALYZE_CAPTION_MODE = os.getenv("ANALYZE_CAPTION_MODE", "short").strip().lower()
ANALYZE_SELECTION_PREVIEW_FORMAT = os.getenv("ANALYZE_SELECTION_PREVIEW_FORMAT", "jpeg").strip().lower()
if ANALYZE_SELECTION_PREVIEW_FORMAT not in {"jpeg", "png"}:
    ANALYZE_SELECTION_PREVIEW_FORMAT = "jpeg"
ANALYZE_SELECTION_PREVIEW_MAX_SIDE = max(128, _env_int("ANALYZE_SELECTION_PREVIEW_MAX_SIDE", 640))
ANALYZE_SELECTION_PREVIEW_JPEG_QUALITY = max(40, min(95, _env_int("ANALYZE_SELECTION_PREVIEW_JPEG_QUALITY", 80)))
ANALYZE_GPU_QUEUE_TIMEOUT_S = max(5.0, _env_float("ANALYZE_GPU_QUEUE_TIMEOUT_S", 70.0))
ANALYZE_MAX_FILE_BYTES = max(1, _env_int("ANALYZE_MAX_FILE_BYTES", 3 * 1024 * 1024))
ANALYZE_BLUR_CHECK_ENABLED = os.getenv("ANALYZE_BLUR_CHECK_ENABLED", "0") == "1"
ANALYZE_BLUR_MIN_FOCUS_SCORE = _env_float("ANALYZE_BLUR_MIN_FOCUS_SCORE", 22.0)
ANALYZE_BLUR_FOCUS_MAX_EDGE = max(256, _env_int("ANALYZE_BLUR_FOCUS_MAX_EDGE", 1024))
ANALYZE_MIN_ACCEPT_CONFIDENCE = _env_float("ANALYZE_MIN_ACCEPT_CONFIDENCE", 0.25)
ANALYZE_ENABLE_PARSER_SPLIT = os.getenv("ANALYZE_ENABLE_PARSER_SPLIT", "0") == "1"
ANALYZE_PARSER_MIN_AREA_RATIO = _env_float("ANALYZE_PARSER_MIN_AREA_RATIO", 0.015)
ANALYZE_PARSER_PAD = max(0, _env_int("ANALYZE_PARSER_PAD", 12))
ANALYZE_ENABLE_HUMAN_PARSER = os.getenv("ANALYZE_ENABLE_HUMAN_PARSER", "1") == "1"
ANALYZE_USE_PARSER_FOR_PREROUTING = os.getenv("ANALYZE_USE_PARSER_FOR_PREROUTING", "0") == "1"
ANALYZE_ENABLE_HEURISTIC_SPLIT = os.getenv("ANALYZE_ENABLE_HEURISTIC_SPLIT", "0") == "1"
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
ANALYZE_UNCERTAIN_FULLBODY_TO_DRESS = os.getenv("ANALYZE_UNCERTAIN_FULLBODY_TO_DRESS", "1") == "1"
ANALYZE_UNCERTAIN_FULLBODY_MIN_HEIGHT_RATIO = _env_float("ANALYZE_UNCERTAIN_FULLBODY_MIN_HEIGHT_RATIO", 0.78)
ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_RATIO = _env_float("ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_RATIO", 0.22)
ANALYZE_UNCERTAIN_FULLBODY_MAX_TOP_RATIO = _env_float("ANALYZE_UNCERTAIN_FULLBODY_MAX_TOP_RATIO", 0.26)
ANALYZE_UNCERTAIN_FULLBODY_MIN_BOTTOM_RATIO = _env_float("ANALYZE_UNCERTAIN_FULLBODY_MIN_BOTTOM_RATIO", 0.90)
ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_ADVANTAGE = _env_float("ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_ADVANTAGE", 1.55)
ANALYZE_PRIMARY_TYPE_WITH_FLORENCE = os.getenv("ANALYZE_PRIMARY_TYPE_WITH_FLORENCE", "1") == "1"
ANALYZE_PRELOAD_FLORENCE = os.getenv("ANALYZE_PRELOAD_FLORENCE", "1") == "1"
ANALYZE_EXTRACT_CLOTH = os.getenv("ANALYZE_EXTRACT_CLOTH", "1") == "1"
ANALYZE_FLUX_DISABLE_LORA = os.getenv("ANALYZE_FLUX_DISABLE_LORA", "1") == "1"
ANALYZE_PRELOAD_FLUX_RUNNER = os.getenv("ANALYZE_PRELOAD_FLUX_RUNNER", "0") == "1"
FLUX2_SHARE_BASE_RUNNER = os.getenv("FLUX2_SHARE_BASE_RUNNER", "1") == "1"
ANALYZE_USE_PARSER_POST_EXTRACT = os.getenv("ANALYZE_USE_PARSER_POST_EXTRACT", "0") == "1"
ANALYZE_PASS_DETECTION_PROMPT_TO_EXTRACT = os.getenv("ANALYZE_PASS_DETECTION_PROMPT_TO_EXTRACT", "1") == "1"
ANALYZE_PROMPT_FROM_EXTRACTED = os.getenv("ANALYZE_PROMPT_FROM_EXTRACTED", "1") == "1"
ANALYZE_EXTRACT_PARSER_ONLY = os.getenv("ANALYZE_EXTRACT_PARSER_ONLY", "1") == "1"
ANALYZE_EXTRACT_FORCE_BBOX_CROP = os.getenv("ANALYZE_EXTRACT_FORCE_BBOX_CROP", "1") == "1"
ANALYZE_EXTRACT_CROP_PAD_RATIO = _env_float("ANALYZE_EXTRACT_CROP_PAD_RATIO", 0.18)
ANALYZE_EXTRACT_CROP_PAD_RATIO_DRESS = _env_float("ANALYZE_EXTRACT_CROP_PAD_RATIO_DRESS", 0.28)
ANALYZE_EXTRACT_CROP_BOTTOM_EXTRA_RATIO_DRESS = _env_float("ANALYZE_EXTRACT_CROP_BOTTOM_EXTRA_RATIO_DRESS", 0.32)
ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM = _env_float("ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM", 0.12)
ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI = _env_float("ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI", 0.12)
ANALYZE_EXTRACT_DRESS_TOP_RECOVERY_RATIO = _env_float("ANALYZE_EXTRACT_DRESS_TOP_RECOVERY_RATIO", 0.14)
ANALYZE_EXTRACT_TOP_TOP_RECOVERY_RATIO = _env_float("ANALYZE_EXTRACT_TOP_TOP_RECOVERY_RATIO", 0.08)
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
ANALYZE_TOP_SKIN_RIM_CLEANUP = os.getenv("ANALYZE_TOP_SKIN_RIM_CLEANUP", "1") == "1"
ANALYZE_TOP_SKIN_RIM_MAX_RATIO = min(0.30, max(0.0, _env_float("ANALYZE_TOP_SKIN_RIM_MAX_RATIO", 0.08)))
ANALYZE_REQUIRE_EXTRACTED_PROMPT = os.getenv("ANALYZE_REQUIRE_EXTRACTED_PROMPT", "1") == "1"
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
ANALYZE_BG_REMOVAL_BACKEND = os.getenv("ANALYZE_BG_REMOVAL_BACKEND", "raw").strip().lower()
if ANALYZE_BG_REMOVAL_BACKEND not in {"raw", "white", "rembg", "birefnet"}:
    ANALYZE_BG_REMOVAL_BACKEND = "raw"
ANALYZE_BIREFNET_MODEL_ID = os.getenv("ANALYZE_BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet").strip()
ANALYZE_BIREFNET_INPUT_SIZE = max(512, _env_int("ANALYZE_BIREFNET_INPUT_SIZE", 1024))
ANALYZE_PROGRESS_SYNC_ASYNC = os.getenv("ANALYZE_PROGRESS_SYNC_ASYNC", "1") == "1"
ANALYZE_PROGRESS_SYNC_MAX_WORKERS = max(1, _env_int("ANALYZE_PROGRESS_SYNC_MAX_WORKERS", 2))
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
COLOR_CONTEXT_DISABLE_MASKING = os.getenv("COLOR_CONTEXT_DISABLE_MASKING", "0") == "1"
GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = os.getenv("GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED", "0") == "1"
USER_PREP_MAX_FILE_BYTES = max(1, _env_int("USER_PREP_MAX_FILE_BYTES", 8 * 1024 * 1024))
USER_PREP_BLUR_CHECK_ENABLED = os.getenv("USER_PREP_BLUR_CHECK_ENABLED", "1") == "1"
USER_PREP_MIN_FOCUS_SCORE = _env_float("USER_PREP_MIN_FOCUS_SCORE", 18.0)
USER_PREP_COMPONENT_MIN_AREA_RATIO = min(
    0.8,
    max(0.0005, _env_float("USER_PREP_COMPONENT_MIN_AREA_RATIO", 0.008)),
)
USER_PREP_MAIN_PERSON_MIN_AREA_RATIO = min(
    0.95,
    max(0.001, _env_float("USER_PREP_MAIN_PERSON_MIN_AREA_RATIO", 0.08)),
)
USER_PREP_MAIN_PERSON_PAD_RATIO = min(0.5, max(0.0, _env_float("USER_PREP_MAIN_PERSON_PAD_RATIO", 0.08)))
USER_PREP_MIN_CROP_SIDE_PX = max(64, _env_int("USER_PREP_MIN_CROP_SIDE_PX", 192))
USER_PREP_REQUIRE_FACE = os.getenv("USER_PREP_REQUIRE_FACE", "1") == "1"
USER_PREP_FACE_MIN_PIXELS = max(64, _env_int("USER_PREP_FACE_MIN_PIXELS", 520))
USER_PREP_FACE_MIN_AREA_RATIO = min(
    0.2,
    max(0.0002, _env_float("USER_PREP_FACE_MIN_AREA_RATIO", 0.0016)),
)
USER_PREP_FACE_MIN_SIDE_PX = max(16, _env_int("USER_PREP_FACE_MIN_SIDE_PX", 28))
USER_PREP_FACE_MIN_FOCUS_SCORE = _env_float("USER_PREP_FACE_MIN_FOCUS_SCORE", 16.0)
USER_PREP_REQUIRE_BG_REMOVAL = os.getenv("USER_PREP_REQUIRE_BG_REMOVAL", "1") == "1"
USER_PREP_BG_BACKEND = os.getenv("USER_PREP_BG_BACKEND", "birefnet").strip().lower()
if USER_PREP_BG_BACKEND not in {"birefnet", "rembg", "auto"}:
    USER_PREP_BG_BACKEND = "birefnet"
USER_PREP_ALPHA_MIN_FOREGROUND_RATIO = min(
    0.99,
    max(0.01, _env_float("USER_PREP_ALPHA_MIN_FOREGROUND_RATIO", 0.05)),
)
USER_PREP_ALPHA_MIN_TRANSPARENT_RATIO = min(
    0.99,
    max(0.0, _env_float("USER_PREP_ALPHA_MIN_TRANSPARENT_RATIO", 0.02)),
)
USER_PREP_DESCRIPTION_BACKEND = os.getenv("USER_PREP_DESCRIPTION_BACKEND", "minicpm").strip().lower()
if USER_PREP_DESCRIPTION_BACKEND not in {"florence", "qwen2_5_vl", "joycaption", "minicpm", "minicpm_service"}:
    USER_PREP_DESCRIPTION_BACKEND = "minicpm"
USER_PREP_MIN_PROMPT_WORDS = max(4, _env_int("USER_PREP_MIN_PROMPT_WORDS", 10))
USER_PREP_UPLOAD_CONTAINER = os.getenv("USER_PREP_UPLOAD_CONTAINER", VTO_OUTPUT_CONTAINER).strip() or VTO_OUTPUT_CONTAINER
gpu_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)
_WARDROBE_PROGRESS_EXECUTOR = ThreadPoolExecutor(max_workers=ANALYZE_PROGRESS_SYNC_MAX_WORKERS)
_REMBG_SESSION = None
_REMBG_SESSION_LOCK = threading.Lock()
_BIREFNET_MODEL = None
_BIREFNET_LOCK = threading.Lock()
STARTUP_BACKGROUND_PRELOAD = os.getenv("STARTUP_BACKGROUND_PRELOAD", "1") == "1"
_STARTUP_PRELOAD_STATE = {
    "mode": "background" if STARTUP_BACKGROUND_PRELOAD else "blocking",
    "started": False,
    "running": False,
    "completed": False,
    "started_at": None,
    "completed_at": None,
    "duration_s": None,
    "errors": [],
}
_STARTUP_PRELOAD_TASK = None

GARMENT_TYPE_SYNONYMS = {
    "top": "top",
    "shirt": "top",
    "tshirt": "top",
    "t-shirt": "top",
    "tee": "top",
    "blouse": "top",
    "shortsleevetop": "top",
    "longsleevetop": "top",
    "shortsleevedshirt": "top",
    "longsleevedshirt": "top",
    "bra": "top",
    "bralette": "top",
    "brassiere": "top",
    "bikinitop": "top",
    "bustier": "top",
    "vest": "top",
    "sling": "top",
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
    "shortsleevedress": "dress",
    "longsleevedress": "dress",
    "vestdress": "dress",
    "slingdress": "dress",
    "outer": "outer",
    "outerwear": "outer",
    "outwear": "outer",
    "jacket": "outer",
    "coat": "outer",
    "blazer": "outer",
    "hoodie": "outer",
    "shortsleeveoutwear": "outer",
    "longsleeveoutwear": "outer",
    "shortsleevedoutwear": "outer",
    "longsleevedoutwear": "outer",
    "shortsleeveouterwear": "outer",
    "longsleeveouterwear": "outer",
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
        self.fashion_detection_runner = FashionDetectionRunner()
        self.parser_runner = HumanParserRunner() if ANALYZE_ENABLE_HUMAN_PARSER else None
        self.yolo = YoloCropper(predictor=self.yolo_runner.predict)
        self.parser = HumanParser(parser_fn=self.parser_runner.parse) if self.parser_runner is not None else None
        self.cloth_detector = ClothDetector(
            legacy_detector=self.yolo,
            fashion_detector=self.fashion_detection_runner,
        )
        self.garment_color_masker = GarmentColorMasker(
            parser=self.parser,
            base_mask_fn=lambda image: _get_clean_foreground_mask(image),
            skin_mask_fn=lambda rgb: _skin_like_mask(rgb),
        )
        self.florence = FlorenceRunner()
        self.qwen25vl = Qwen25VLRunner()
        self.joycaption = JoyCaptionRunner()
        self.minicpm = MiniCPMVRunner()
        self.openclip = OpenCLIPRunner()
        shared_flux2_config: Dict[str, object] = {}
        self._share_flux2_base_runner = bool(FLUX2_SHARE_BASE_RUNNER and ANALYZE_FLUX_DISABLE_LORA)
        if self._share_flux2_base_runner:
            shared_flux2_config["runtime_lora_toggle"] = True
            shared_flux2_config["fuse_lora"] = False
        self.flux2 = Flux2CVTONRunner(config=shared_flux2_config)
        self._analyze_flux2: Optional[Flux2CVTONRunner] = None
        self._analyze_flux2_lock = threading.Lock()
        self.board_builder = BoardBuilder()

    def get_flux2_for_analyze(self) -> Flux2CVTONRunner:
        # Shared-base mode keeps one FLUX pipeline in memory and toggles LoRA per request.
        if self._share_flux2_base_runner:
            return self.flux2
        # Strict isolation: /analyze can run on a separate no-LoRA runner so try-on remains unchanged.
        if not ANALYZE_FLUX_DISABLE_LORA:
            return self.flux2
        if self._analyze_flux2 is None:
            with self._analyze_flux2_lock:
                if self._analyze_flux2 is None:
                    self._analyze_flux2 = Flux2CVTONRunner(
                        config={
                            "enable_lora": False,
                            "require_lora": False,
                            "fuse_lora": False,
                        }
                    )
        return self._analyze_flux2
        
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
        if ANALYZE_PRELOAD_FLORENCE:
            self.florence._ensure_loaded()
        if ANALYZE_PRELOAD_FLUX_RUNNER:
            self.get_flux2_for_analyze().ensure_ready()

    def model_status(self):
        analyze_flux2 = self.flux2 if self._share_flux2_base_runner else self._analyze_flux2
        analyze_startup = dict(getattr(analyze_flux2, "_startup_metrics", {}) or {}) if analyze_flux2 else {}
        return {
            "flux2_loaded": self.flux2._pipeline is not None,
            "analyze_flux2_loaded": bool(analyze_flux2 and analyze_flux2._pipeline is not None),
            "analyze_flux2_isolated": bool(ANALYZE_FLUX_DISABLE_LORA and not self._share_flux2_base_runner),
            "flux2_shared_base_runner": bool(self._share_flux2_base_runner),
            "flux2_runtime_lora_toggle": bool(getattr(self.flux2, "runtime_lora_toggle", False)),
            "analyze_flux2_lora_enabled": bool(
                analyze_startup.get("lora_enabled", False)
            ) if analyze_flux2 else False,
            "analyze_flux2_lora_loaded": bool(
                analyze_startup.get("lora_loaded", False)
            ) if analyze_flux2 else False,
            "florence_loaded": self.florence._model is not None,
            "qwen25vl_loaded": self.qwen25vl.is_loaded,
            "joycaption_loaded": self.joycaption.is_loaded,
            "minicpm_loaded": self.minicpm.is_loaded,
            "minicpm_model_id": str(getattr(self.minicpm, "model_id", "")),
            "openclip_loaded": self.openclip.is_loaded,
            "openclip_available": self.openclip.is_available,
            "yolo_loaded": self.yolo_runner.is_loaded,
            "yolo_model_path": self.yolo_runner.model_path,
            "yolo_expected_label_family": self.yolo_runner.expected_label_family,
            "yolo_label_family": self.yolo_runner.label_family,
            "yolo_class_count": self.yolo_runner.class_count,
            "fashion_detection_loaded": self.fashion_detection_runner.is_loaded,
            "fashion_detection_model_path": self.fashion_detection_runner.model_path,
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


def _clean_prompt_section_text(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    cleaned = re.sub(
        r"^(?:BASE_GARMENT_PROMPT|EXTRACTION_AVOID_CLAUSE)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned.strip(" -")


def _parse_garment_prompt_sections(
    raw_text: str,
    *,
    garment_type: Optional[str] = None,
) -> Dict[str, str]:
    text = str(raw_text or "").strip()
    normalized_text = " ".join(text.split()).strip()
    base_prompt = ""
    avoid_clause = ""

    base_match = re.search(
        r"BASE_GARMENT_PROMPT\s*:\s*(.*?)(?=\bEXTRACTION_AVOID_CLAUSE\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    avoid_match = re.search(
        r"EXTRACTION_AVOID_CLAUSE\s*:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if base_match:
        base_prompt = _clean_prompt_section_text(base_match.group(1))
    if avoid_match:
        avoid_clause = _clean_prompt_section_text(avoid_match.group(1))

    if not base_prompt:
        base_prompt = _normalize_minicpm_descriptor_text(text, kind="garment")
    base_prompt = _sanitize_florence_garment_description(base_prompt or "")
    base_prompt = " ".join(str(base_prompt or "").split()).strip(" ,.")
    if base_prompt and not base_prompt.endswith("."):
        base_prompt = f"{base_prompt}."
    base_prompt = _ensure_target_type_in_description(base_prompt, str(garment_type or ""))

    avoid_clause = " ".join(str(avoid_clause or "").split()).strip(" ,.")
    if avoid_clause and not avoid_clause.endswith("."):
        avoid_clause = f"{avoid_clause}."

    serialized_sections = f"BASE_GARMENT_PROMPT: {base_prompt or 'Garment.'}"
    if avoid_clause:
        serialized_sections += f"\nEXTRACTION_AVOID_CLAUSE: {avoid_clause}"

    return {
        "raw_text": normalized_text,
        "base_garment_prompt": base_prompt or "Garment.",
        "extraction_avoid_clause": avoid_clause,
        "serialized_sections": serialized_sections,
    }


def _extract_prompt_fact_segments(text: str) -> Dict[str, str]:
    src = " ".join(str(text or "").split()).strip().strip(" ,.")
    if not src:
        return {}

    labels = [
        "category",
        "type",
        "colors",
        "pattern",
        "material",
        "silhouette",
        "construction",
        "details",
        "coverage",
        "preserve",
    ]
    parsed: Dict[str, str] = {}
    structured = _parse_structured_descriptor(src)
    for label in labels:
        value = " ".join(str(structured.get(label) or "").split()).strip(" ,.")
        if value:
            parsed[label] = value

    for label in labels:
        if parsed.get(label):
            continue
        match = re.search(
            rf"(?:^|,\s*){re.escape(label)}\s+(.+?)(?=(?:,\s*(?:{'|'.join(labels)})\s+)|$)",
            src,
            flags=re.IGNORECASE,
        )
        if match:
            value = " ".join(str(match.group(1) or "").split()).strip(" ,.")
            if value:
                parsed[label] = value
    return parsed


def _sanitize_prompt_fact_value(value: str) -> str:
    cleaned = " ".join(str(value or "").split()).strip().strip(" ,.")
    cleaned = re.sub(
        r"\bEXTRACTION_AVOID_CLAUSE\s*:\s*.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.")
    return cleaned


def _serialize_prompt_fact_segments(fields: Dict[str, str]) -> str:
    ordered_labels = (
        "category",
        "type",
        "colors",
        "pattern",
        "material",
        "silhouette",
        "construction",
        "details",
        "coverage",
        "preserve",
    )
    parts: List[str] = []
    for label in ordered_labels:
        value = _sanitize_prompt_fact_value(str(fields.get(label) or ""))
        if value:
            parts.append(f"{label}={value}")
    return "; ".join(parts).strip(" ;.")


def _resolve_garment_color_truth(
    *,
    base_garment_prompt: str,
    dominant_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    prompt_fields = dict(_parse_structured_descriptor(base_garment_prompt))
    prompt_fields.update(_extract_prompt_fact_segments(base_garment_prompt))
    prompt_color_text = str(prompt_fields.get("colors") or "").strip()
    prompt_color_terms: List[str] = []
    if prompt_color_text:
        for raw_piece in re.split(r",|/|\band\b|&", prompt_color_text, flags=re.IGNORECASE):
            clean_piece = _canonical_color_token(raw_piece)
            if clean_piece and clean_piece in _TEXT_COLOR_TERMS and clean_piece not in prompt_color_terms:
                prompt_color_terms.append(clean_piece)
    if not prompt_color_terms:
        prompt_color_terms = [
            term for term in _extract_text_color_terms(str(prompt_color_text or base_garment_prompt or ""), max_items=4)
            if str(term).strip()
        ]
    prompt_color_terms = list(dict.fromkeys(prompt_color_terms))
    prompt_non_neutral = [
        term for term in prompt_color_terms
        if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}
    ]

    pixel_hexes = [
        str(v).strip().upper()
        for v in (dominant_hexes or [])
        if str(v).strip()
    ]
    pixel_hints = [
        str(v).strip().lower()
        for v in (color_hints or [])
        if str(v).strip()
    ]
    pixel_hints = list(dict.fromkeys(pixel_hints))
    pixel_non_neutral = [
        term for term in pixel_hints
        if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}
    ]

    resolved_hints = list(pixel_hints)
    resolved_hexes = list(pixel_hexes)
    resolved_source = "pixel"

    prompt_families = {_color_family(term) for term in prompt_non_neutral if term}
    pixel_families = {_color_family(term) for term in pixel_non_neutral if term}
    strong_semantic_override = bool(
        GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED
        and prompt_non_neutral
        and (
            not pixel_non_neutral
            or pixel_families.isdisjoint(prompt_families)
            or all(_is_neutral_color_token(term) for term in pixel_hints[: max(1, len(pixel_hints))])
        )
    )
    if strong_semantic_override:
        semantic_terms = prompt_color_terms[: max(2, FLUX2_COLOR_LOCK_TOP_K)]
        semantic_hexes: List[str] = []
        for term in semantic_terms:
            rgb = _COLOR_LABEL_RGB_MAP.get(term)
            if rgb:
                semantic_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
                if semantic_hex not in semantic_hexes:
                    semantic_hexes.append(semantic_hex)
        if semantic_terms:
            resolved_hints = list(semantic_terms)
            resolved_source = "semantic_prompt_override"
        if semantic_hexes:
            resolved_hexes = list(semantic_hexes)

    def _normalize_resolved_color_hints(
        hints: List[str],
        profile: Optional[Dict[str, object]],
    ) -> List[str]:
        ordered = list(dict.fromkeys(str(v).strip().lower() for v in (hints or []) if str(v).strip()))
        if not ordered:
            return []
        non_neutral = [term for term in ordered if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}]
        if not non_neutral:
            return ordered

        dominant_family = _color_family(non_neutral[0])
        mean_chroma = profile.get("meanChroma") if isinstance(profile, dict) else None
        mean_b = profile.get("meanB") if isinstance(profile, dict) else None
        soft_warm_neutrals = {"beige", "champagne", "tan", "nude", "khaki"}

        filtered = list(ordered)
        if (
            dominant_family in {"yellow", "green"}
            and isinstance(mean_chroma, (int, float))
            and isinstance(mean_b, (int, float))
            and float(mean_chroma) >= 18.0
            and float(mean_b) >= 12.0
        ):
            pruned = [term for term in ordered if term not in soft_warm_neutrals]
            if any(_color_family(term) == dominant_family for term in pruned):
                filtered = pruned

        non_neutral_filtered = [term for term in filtered if _color_family(term) not in {"neutral_dark", "neutral_mid", "neutral_light", "brown"}]
        neutral_filtered = [term for term in filtered if term not in non_neutral_filtered]
        return non_neutral_filtered + neutral_filtered

    resolved_hints = _normalize_resolved_color_hints(resolved_hints, color_profile)
    resolved_hints = resolved_hints[: max(2, FLUX2_COLOR_LOCK_TOP_K)]
    resolved_hexes = resolved_hexes[: max(2, FLUX2_COLOR_LOCK_TOP_K + 1)]
    resolved_color_text = ", ".join(resolved_hints)

    reconciled_fields = dict(prompt_fields)
    if resolved_color_text:
        reconciled_fields["colors"] = resolved_color_text
    for key in ("details", "preserve", "construction", "coverage", "material", "silhouette", "pattern", "type", "category"):
        if key in reconciled_fields:
            reconciled_fields[key] = _sanitize_prompt_fact_value(reconciled_fields.get(key, ""))
    reconciled_prompt = _serialize_prompt_fact_segments(reconciled_fields)
    if reconciled_prompt:
        reconciled_prompt = f"{reconciled_prompt}."

    return {
        "base_garment_prompt": reconciled_prompt or " ".join(str(base_garment_prompt or "").split()).strip(),
        "dominant_hexes": resolved_hexes,
        "color_hints": resolved_hints,
        "color_source": resolved_source,
    }


def _extract_garment_descriptor_facts(
    *,
    base_garment_prompt: str,
    descriptor_raw_text: str = "",
) -> Dict[str, str]:
    raw_fields = _parse_structured_descriptor(descriptor_raw_text)
    prompt_fields = _extract_prompt_fact_segments(base_garment_prompt)
    merged: Dict[str, str] = {}
    for key in (
        "category",
        "type",
        "colors",
        "pattern",
        "material",
        "silhouette",
        "construction",
        "details",
        "coverage",
        "preserve",
    ):
        value = str(raw_fields.get(key) or prompt_fields.get(key) or "").strip()
        if value:
            merged[key] = _sanitize_prompt_fact_value(value)
    return merged


def _build_garment_metadata(
    *,
    base_garment_prompt: str,
    extraction_avoid_clause: str = "",
    prompt_sections_raw: str = "",
    descriptor_raw_text: str = "",
    prompt_description: str = "",
    prompt_source: str = "",
    target_type: str = "",
    backend_target_type: str = "",
    style: str = "",
    primary_category_key: str = "",
    category_key: str = "",
    dominant_hexes: Optional[List[str]] = None,
    accent_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
    color_mask_source: str = "",
) -> Dict[str, object]:
    base_prompt = " ".join(str(base_garment_prompt or "").split()).strip()
    avoid_clause = " ".join(str(extraction_avoid_clause or "").split()).strip()
    normalized_prompt = " ".join(str(prompt_description or base_prompt).split()).strip()
    classification_target = str(target_type or "").strip()
    classification_backend = str(backend_target_type or classification_target).strip()
    reconciled_color = _resolve_garment_color_truth(
        base_garment_prompt=base_prompt,
        dominant_hexes=dominant_hexes,
        color_hints=color_hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
    )
    resolved_base_prompt = str(reconciled_color.get("base_garment_prompt") or base_prompt).strip()
    resolved_prompt_description = " ".join(str(prompt_description or resolved_base_prompt).split()).strip()
    if resolved_base_prompt and resolved_prompt_description:
        prompt_fields = _extract_prompt_fact_segments(resolved_prompt_description)
        base_fields = _extract_prompt_fact_segments(resolved_base_prompt)
        if base_fields.get("colors"):
            prompt_fields["colors"] = base_fields["colors"]
            rebuilt_prompt = _serialize_prompt_fact_segments(prompt_fields)
            if rebuilt_prompt:
                resolved_prompt_description = f"{rebuilt_prompt}."

    fact_fields = _extract_garment_descriptor_facts(
        base_garment_prompt=resolved_base_prompt,
        descriptor_raw_text=descriptor_raw_text,
    )
    return {
        "schema_version": "garment_metadata.v1",
        "prompt": {
            "base_garment_prompt": resolved_base_prompt,
            "prompt_description": resolved_prompt_description or normalized_prompt,
            "extraction_avoid_clause": avoid_clause,
            "prompt_sections_raw": str(prompt_sections_raw or "").strip(),
            "descriptor_raw_text": str(descriptor_raw_text or "").strip(),
            "source": str(prompt_source or "").strip(),
        },
        "classification": {
            "target_type": classification_target,
            "backend_target_type": classification_backend,
            "style": str(style or "").strip(),
            "primary_category_key": str(primary_category_key or "").strip(),
            "category_key": str(category_key or "").strip(),
        },
        "color": {
            "dominant_hexes": [str(v).strip().upper() for v in (reconciled_color.get("dominant_hexes") or []) if str(v).strip()],
            "accent_hexes": [str(v).strip().upper() for v in (accent_hexes or []) if str(v).strip()],
            "color_hints": [str(v).strip().lower() for v in (reconciled_color.get("color_hints") or []) if str(v).strip()],
            "profile": color_profile if isinstance(color_profile, dict) else {},
            "mask_source": str(color_mask_source or "").strip(),
            "resolved_source": str(reconciled_color.get("color_source") or "pixel"),
        },
        "details": fact_fields,
    }


def _extract_garment_metadata_prompt(garment_metadata: object) -> str:
    if not isinstance(garment_metadata, dict):
        return ""
    prompt_block = garment_metadata.get("prompt")
    if not isinstance(prompt_block, dict):
        return ""
    return " ".join(
        str(
            prompt_block.get("base_garment_prompt")
            or prompt_block.get("prompt_description")
            or ""
        ).split()
    ).strip()


def _extract_garment_metadata_target_type(garment_metadata: object) -> Optional[str]:
    if not isinstance(garment_metadata, dict):
        return None
    classification = garment_metadata.get("classification")
    if not isinstance(classification, dict):
        return None
    for key in ("backend_target_type", "target_type"):
        value = _normalize_garment_type(str(classification.get(key) or ""))
        if value:
            return value
    return None


def _extract_garment_metadata_color_payload(garment_metadata: object) -> Tuple[List[str], List[str]]:
    if not isinstance(garment_metadata, dict):
        return [], []
    color_block = garment_metadata.get("color")
    if not isinstance(color_block, dict):
        return [], []
    dominant_hexes = [
        str(v).strip().upper()
        for v in (color_block.get("dominant_hexes") or [])
        if str(v).strip()
    ]
    color_hints = [
        str(v).strip().lower()
        for v in (color_block.get("color_hints") or [])
        if str(v).strip()
    ]
    return dominant_hexes, color_hints


def _extract_garment_metadata_color_block(garment_metadata: object) -> Dict[str, object]:
    if not isinstance(garment_metadata, dict):
        return {}
    color_block = garment_metadata.get("color")
    if not isinstance(color_block, dict):
        return {}
    dominant_hexes = [
        str(v).strip().upper()
        for v in (color_block.get("dominant_hexes") or [])
        if str(v).strip()
    ]
    color_hints = [
        str(v).strip().lower()
        for v in (color_block.get("color_hints") or [])
        if str(v).strip()
    ]
    accent_hexes = [
        str(v).strip().upper()
        for v in (color_block.get("accent_hexes") or [])
        if str(v).strip()
    ]
    profile = color_block.get("profile") if isinstance(color_block.get("profile"), dict) else {}
    mask_source = str(color_block.get("mask_source") or "").strip()
    return {
        "dominant_hexes": dominant_hexes,
        "color_hints": color_hints,
        "accent_hexes": accent_hexes,
        "profile": profile,
        "mask_source": mask_source,
    }

def _strip_descriptor_color_clause(description: str) -> str:
    """
    Remove model-predicted color phrase from schema-like descriptors.
    We keep pixel-derived color lock as the source of truth.
    """
    text = " ".join(str(description or "").split()).strip()
    if not text:
        return text
    segments = [seg.strip() for seg in text.split(",") if seg.strip()]
    if not segments:
        return text

    key_prefixes = (
        "type ",
        "colors ",
        "pattern ",
        "material ",
        "silhouette ",
        "construction ",
        "details ",
        "coverage ",
        "preserve ",
    )

    out: List[str] = []
    dropping_color_tail = False
    for seg in segments:
        low = seg.lower()
        if low.startswith("colors "):
            dropping_color_tail = True
            continue
        if dropping_color_tail:
            is_new_key = any(low.startswith(k) for k in key_prefixes if k != "colors ")
            if not is_new_key:
                continue
            dropping_color_tail = False
        out.append(seg)

    normalized = ", ".join(out).strip(" ,")
    return normalized or text


def _descriptor_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9#]+", str(text or "")))


def _descriptor_is_weak(text: str, *, min_words: int = MINICPM_SERVICE_GARMENT_MIN_WORDS) -> bool:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return True
    words = _descriptor_word_count(clean)
    if words < max(4, int(min_words)):
        return True
    if ";" in clean and "=" in clean:
        # Structured output from VLM is usually rich enough.
        return False
    richness_markers = (
        "type",
        "material",
        "silhouette",
        "construction",
        "details",
        "coverage",
        "sleeve",
        "neckline",
        "hem",
        "ruffle",
        "pleat",
    )
    lower = clean.lower()
    return not any(token in lower for token in richness_markers)


def _enrich_garment_descriptor(primary: str, fallback: str, garment_type: str) -> str:
    """
    Ensure analyze/extract prompts stay descriptive even when VLM emits a short sentence.
    """
    primary_clean = _sanitize_florence_garment_description(primary or "")
    fallback_clean = _sanitize_florence_garment_description(fallback or "")
    primary_clean = " ".join(primary_clean.split()).strip(" ,.")
    fallback_clean = " ".join(fallback_clean.split()).strip(" ,.")

    if not fallback_clean:
        return primary_clean or fallback_clean
    if not primary_clean:
        return fallback_clean
    if fallback_clean.lower() in primary_clean.lower():
        return primary_clean
    if primary_clean.lower() in fallback_clean.lower():
        return fallback_clean

    # Keep strongest descriptor first, append fallback context once.
    joined = f"{primary_clean}. {fallback_clean}"
    joined = re.sub(r"\s{2,}", " ", joined).strip(" .")
    if joined and garment_type:
        g = _normalize_garment_type(garment_type) or garment_type
        if g.lower() not in joined.lower():
            joined = f"{g} garment. {joined}"
    return joined

def _augment_identity_lock(user_description: str) -> str:
    base = str(user_description or "").strip()
    lock = (
        "Keep exact same person identity and pose: preserve facial geometry, expression, skin tone, "
        "hairstyle, hairline, hands, body proportions, camera framing, and scene lighting. "
        "Do not restyle face or change head/body posture."
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
        "bottom", "pant", "pants", "trouser", "trousers", "jean", "jeans", "skirt", "shorts",
        "palazzo", "chino", "legging", "leggings",
    )
    outer_terms = ("jacket", "coat", "blazer", "hoodie", "cardigan", "outerwear", "outer", "shrug")
    top_terms = ("shirt", "t-shirt", "tee", "top", "blouse", "corset", "sweater", "kurta", "tunic")

    has_dress_term = any(term in text for term in dress_terms)
    has_bottom_term = any(term in text for term in bottom_terms)
    has_outer_term = any(term in text for term in outer_terms)
    has_top_term = any(term in text for term in top_terms)
    lower_body_cues = (
        "coverage full legs",
        "coverage legs",
        "coverage lower body",
        "lower body",
        "lower-body",
        "waist to ankle",
        "full legs",
        "leg coverage",
        "pants only",
    )
    upper_body_cues = (
        "coverage torso",
        "coverage upper body",
        "coverage upper-body",
        "torso and arms",
        "upper body",
        "upper-body",
    )
    has_lower_body_cue = any(cue in text for cue in lower_body_cues)
    has_upper_body_cue = any(cue in text for cue in upper_body_cues)

    # Be conservative with dress detection for model-generated captions.
    # If top/bottom/outer cues co-exist, prefer region-specific replacement.
    if has_dress_term and not (has_top_term or has_bottom_term or has_outer_term):
        return "dress"
    if has_lower_body_cue and not has_upper_body_cue:
        return "bottom"
    if has_top_term and has_bottom_term:
        if has_lower_body_cue:
            return "bottom"
        if has_upper_body_cue:
            return "top"
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
    ("plum", (98, 56, 102)),
    ("pink", (214, 120, 165)),
]

_DETAIL_LOCK_TERMS: List[Tuple[str, str]] = [
    ("button", "button count and placement"),
    ("buttons", "button count and placement"),
    ("dotted button", "dotted button detailing"),
    ("dot button", "dotted button detailing"),
    ("stud", "stud detailing"),
    ("lapel", "lapel shape"),
    ("pocket flap", "pocket flap placement"),
    ("border", "border placement"),
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
    "cream",
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
    "plum",
    "lavender",
    "pink",
    "peach",
]

_COLOR_LABEL_RGB_MAP: Dict[str, Tuple[int, int, int]] = dict(_COLOR_LABEL_RGB)

def _canonical_color_token(color: str) -> str:
    token = str(color or "").strip().lower()
    aliases = {
        "grey": "gray",
        "off white": "off-white",
        "offwhite": "off-white",
    }
    return aliases.get(token, token)

def _color_family(color: str) -> str:
    c = _canonical_color_token(color)
    if c in {"black", "charcoal"}:
        return "neutral_dark"
    if c in {"gray", "silver"}:
        return "neutral_mid"
    if c in {"white", "off-white", "ivory", "beige", "champagne", "tan", "nude"}:
        return "neutral_light"
    if c in {"red", "maroon", "burgundy"}:
        return "red"
    if c in {"orange", "peach"}:
        return "orange"
    if c in {"yellow", "gold"}:
        return "yellow"
    if c in {"green", "olive"}:
        return "green"
    if c in {"blue", "navy", "teal"}:
        return "blue"
    if c in {"purple", "lavender", "plum"}:
        return "purple"
    if c in {"pink", "blush pink", "dusty pink", "hot pink", "rose gold"}:
        return "pink"
    if c in {"brown"}:
        return "brown"
    return c

def _is_neutral_color_token(color: str) -> bool:
    fam = _color_family(color)
    return fam in {"neutral_dark", "neutral_mid", "neutral_light"}

def _hex_to_rgb_triplet(token: str) -> Optional[Tuple[int, int, int]]:
    t = str(token or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", t):
        return None
    return (int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16))

def _rgb_hue_deg(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [max(0, min(255, int(v))) / 255.0 for v in rgb]
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return float(h * 360.0)

def _palette_weighted_hue_deg(palette: List[Dict[str, object]], max_colors: int = 4) -> Optional[float]:
    vals: List[Tuple[float, float]] = []
    for entry in (palette or [])[: max(1, int(max_colors))]:
        hx = str(entry.get("hex", "")).strip()
        rgb = _hex_to_rgb_triplet(hx)
        if rgb is None:
            continue
        w = float(entry.get("areaPercent", 0.0) or 0.0)
        vals.append((_rgb_hue_deg(rgb), max(0.1, w)))
    if not vals:
        return None
    # Weighted circular mean
    sin_sum = 0.0
    cos_sum = 0.0
    for deg, w in vals:
        rad = np.deg2rad(deg)
        sin_sum += np.sin(rad) * w
        cos_sum += np.cos(rad) * w
    if abs(sin_sum) < 1e-6 and abs(cos_sum) < 1e-6:
        return None
    ang = float(np.rad2deg(np.arctan2(sin_sum, cos_sum)))
    if ang < 0.0:
        ang += 360.0
    return ang

def _profile_is_near_white(profile: Dict[str, object]) -> bool:
    if not isinstance(profile, dict):
        return False
    median_l = profile.get("medianL")
    p90_l = profile.get("p90L")
    mean_c = profile.get("meanChroma")
    if not all(isinstance(v, (int, float)) for v in (median_l, p90_l, mean_c)):
        return False
    return bool(
        profile.get("isNeutral")
        and float(p90_l) >= 84.0
        and float(median_l) >= 68.0
        and float(mean_c) <= 5.5
    )

def _nearest_color_label(rgb_triplet: Tuple[int, int, int]) -> str:
    """
    Standardizes color terms for consistent Flux2 conditioning.
    Uses CIELAB for neutrals to solve the cream/ivory vs silver/gray drift.
    """
    r_i, g_i, b_i = [int(v) for v in rgb_triplet]

    # 1. Check for neutrals using Lab coordinates for better stability than RGB warmness.
    # OpenCV returns L in [0,255] and a/b centered at 128 (not 0).
    lab = rgb_to_lab(np.array([[r_i, g_i, b_i]], dtype=np.uint8))[0]
    l_star = float(lab[0]) * (100.0 / 255.0)
    a_star = float(lab[1]) - 128.0
    b_star = float(lab[2]) - 128.0

    # Perceptual saturation in CIELAB (distance from neutral axis).
    chroma = float(np.sqrt((a_star * a_star) + (b_star * b_star)))

    if chroma < 14.0:
        if l_star < 45.0 and a_star >= 10.0 and b_star <= -2.0:
            return "plum" if l_star < 34.0 else "purple"
        if l_star < 72.0 and a_star <= -2.5 and b_star >= 3.0:
            return "olive" if b_star >= 6.0 else "green"
        # This is a neutral/near-neutral color. Use L* to bucket it.
        if l_star < 10.0:
            return "black"
        if l_star < 28.0:
            return "charcoal"
        if l_star < 62.0:
            # Gray vs Tan/Stone
            return "gray" if b_star < 8.0 else "tan"
        if l_star < 85.0:
            # Silver vs Beige/Khaki
            return "silver" if b_star < 9.0 else "beige"
        if l_star < 96.0:
            # Ivory/Cream vs White
            # Cream/Ivory typically has b* around 10-20.
            return "cream" if b_star > 11.0 else ("ivory" if b_star > 4.0 else "white")
        return "white"

    # 2. Saturated colors: use Euclidean distance to reference palette
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
    hexes = _extract_dominant_hex_colors(crop, top_k=max(2, int(top_k) + 2))
    out: List[str] = []
    for hx in hexes:
        token = str(hx or "").strip()
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", token):
            continue
        r = int(token[1:3], 16)
        g = int(token[3:5], 16)
        b = int(token[5:7], 16)
        label = _nearest_color_label((r, g, b))
        if label == "grey":
            label = "gray"
        if label not in out:
            out.append(label)
        if len(out) >= max(1, int(top_k)):
            break
    return out


def _build_garment_color_tone_guidance(
    *,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    hints = [str(v).strip().lower() for v in (color_hints or []) if str(v).strip()]
    if not hints:
        return {"phrase": "", "negative_terms": []}

    primary = hints[0]
    family = _color_family(primary)
    profile = color_profile if isinstance(color_profile, dict) else {}
    median_l = profile.get("medianL")
    mean_c = profile.get("meanChroma")
    mean_b = profile.get("meanB")
    phrase = primary
    negative_terms: List[str] = []

    if family == "yellow":
        if (
            isinstance(median_l, (int, float))
            and isinstance(mean_c, (int, float))
            and isinstance(mean_b, (int, float))
            and float(median_l) >= 74.0
            and float(mean_c) >= 28.0
            and float(mean_b) >= 28.0
        ):
            phrase = "bright lemon yellow"
            negative_terms = ["gold", "golden", "mustard", "beige", "champagne", "tan", "bronze", "brown", "orange"]
        elif (
            isinstance(median_l, (int, float))
            and isinstance(mean_c, (int, float))
            and float(median_l) >= 66.0
            and float(mean_c) >= 18.0
        ):
            phrase = "clear yellow"
            negative_terms = ["gold", "mustard", "beige", "champagne", "tan", "brown"]
    elif family == "green":
        if (
            isinstance(mean_c, (int, float))
            and isinstance(median_l, (int, float))
            and float(mean_c) < 18.0
            and float(median_l) >= 48.0
        ):
            phrase = "muted sage green"
            negative_terms = ["gray", "silver", "gold", "beige", "olive brown"]

    return {
        "phrase": phrase,
        "negative_terms": negative_terms,
    }

def _extract_dominant_hex_colors_with_coverage(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    top_k: int = 4,
) -> List[Dict[str, object]]:
    """
    LAB K-Means based color extraction with area coverage metadata.
    """
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return []

        pixels = arr.reshape(-1, 3)
        use_mask = mask
        if FLUX2_COLOR_DECONTAMINATION_ENABLED:
            clean_mask = _get_clean_foreground_mask(image, mask=mask)
            if isinstance(clean_mask, np.ndarray):
                use_mask = clean_mask
        if isinstance(use_mask, np.ndarray):
            m = np.asarray(use_mask).astype(bool)
            if m.shape[:2] == arr.shape[:2]:
                keep = m.reshape(-1)
                if int(np.sum(keep)) > 32:
                    pixels = pixels[keep]

        if pixels.size == 0:
            return []

        near_white = np.all(pixels >= 248, axis=1)
        if int(np.sum(~near_white)) > 16:
            pixels = pixels[~near_white]
        if pixels.size < 4:
            return []

        import cv2

        pixels_u8 = np.ascontiguousarray(pixels.astype(np.uint8))
        pixels_lab = cv2.cvtColor(pixels_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        if pixels_lab.size < 12:
            return []

        unique_lab_count = int(np.unique(pixels_lab, axis=0).shape[0])
        k = min(max(1, int(top_k) + 1), int(pixels_lab.shape[0]), max(1, unique_lab_count))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels_lab.astype(np.float32),
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        flat_labels = labels.reshape(-1)
        unique, counts = np.unique(flat_labels, return_counts=True)
        order = np.argsort(-counts)

        centers_u8 = np.clip(centers, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        centers_rgb = cv2.cvtColor(centers_u8, cv2.COLOR_LAB2RGB).reshape(-1, 3)

        total_pixels = max(1, int(np.sum(counts)))
        out: List[Dict[str, object]] = []
        seen_hex = set()
        for order_idx in order.tolist():
            center_idx = int(unique[order_idx])
            r, g, b = [int(v) for v in centers_rgb[center_idx].tolist()]
            r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
            hx = f"#{r:02X}{g:02X}{b:02X}"
            if hx in seen_hex:
                continue
            seen_hex.add(hx)
            pixel_count = int(counts[order_idx])
            area_percent = round((float(pixel_count) / float(total_pixels)) * 100.0, 2)
            out.append(
                {
                    "hex": hx,
                    "areaPercent": area_percent,
                    "pixelCount": pixel_count,
                }
            )
            if len(out) >= max(1, int(top_k)):
                break
        return out
    except Exception as e:
        logger.warning(f"K-Means color extraction failed: {e}")
        return []

def _extract_dominant_hex_colors(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    top_k: int = 4,
) -> List[str]:
    entries = _extract_dominant_hex_colors_with_coverage(image=image, mask=mask, top_k=top_k)
    out: List[str] = []
    for entry in entries:
        token = str(entry.get("hex", "")).strip()
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", token):
            out.append(token.upper())
    return out

def _filter_palette_entries_by_area(
    palette: List[Dict[str, object]],
    min_area_percent: float,
    min_items: int = 2,
) -> List[Dict[str, object]]:
    entries = list(palette or [])
    if not entries:
        return []
    threshold = float(max(0.0, min(100.0, min_area_percent)))
    filtered = [
        e for e in entries
        if float(e.get("areaPercent", 0.0) or 0.0) >= threshold
    ]
    if len(filtered) >= int(max(1, min_items)):
        return filtered
    return entries[: max(1, min(int(min_items), len(entries)))]

def _extract_lab_color_profile(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Extract compact Lab profile for neutral fidelity locking/scoring.
    """
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return {}

        use_mask = None
        if FLUX2_COLOR_DECONTAMINATION_ENABLED:
            clean_mask = _get_clean_foreground_mask(image, mask=mask)
            if isinstance(clean_mask, np.ndarray):
                use_mask = clean_mask
        if not isinstance(use_mask, np.ndarray):
            use_mask = mask
        if not isinstance(use_mask, np.ndarray):
            use_mask = _extract_alpha_mask(image)
        if not isinstance(use_mask, np.ndarray):
            use_mask = _estimate_foreground_mask_from_border(image)

        pixels = arr.reshape(-1, 3)
        if isinstance(use_mask, np.ndarray) and use_mask.shape[:2] == arr.shape[:2]:
            keep = np.asarray(use_mask).astype(bool).reshape(-1)
            if int(np.sum(keep)) > 64:
                pixels = pixels[keep]

        if pixels.size == 0:
            return {}

        near_white = np.all(pixels >= 248, axis=1)
        if int(np.sum(~near_white)) > 16:
            pixels = pixels[~near_white]
        if pixels.size < 64:
            return {}

        import cv2
        lab = cv2.cvtColor(
            np.ascontiguousarray(pixels.astype(np.uint8)).reshape(-1, 1, 3),
            cv2.COLOR_RGB2LAB,
        ).reshape(-1, 3).astype(np.float32)

        lab_for_stats = lab
        if FLUX2_COLOR_DECONTAMINATION_ENABLED and int(lab.shape[0]) >= int(FLUX2_COLOR_DECONTAM_MIN_PIXELS):
            quick_a = lab[:, 1] - 128.0
            quick_b = lab[:, 2] - 128.0
            quick_chroma = np.sqrt((quick_a * quick_a) + (quick_b * quick_b))
            quick_mean_chroma = float(np.mean(quick_chroma))
            quick_p90_chroma = float(np.percentile(quick_chroma, 90))
            # Trimming is most helpful for neutral/near-neutral garments that are sensitive to dark-fringe contamination.
            if quick_mean_chroma < 18.0 and quick_p90_chroma < 28.0:
                trimmed = _trim_lab_profile_outliers(
                    lab,
                    dark_percentile=float(FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE),
                    bright_percentile=float(FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE),
                    min_pixels=int(FLUX2_COLOR_DECONTAM_MIN_PIXELS),
                )
                if isinstance(trimmed, np.ndarray) and int(trimmed.shape[0]) >= int(FLUX2_COLOR_DECONTAM_MIN_PIXELS):
                    lab_for_stats = trimmed

        l_star = lab_for_stats[:, 0] * (100.0 / 255.0)
        a_star = lab_for_stats[:, 1] - 128.0
        b_star = lab_for_stats[:, 2] - 128.0
        chroma = np.sqrt((a_star * a_star) + (b_star * b_star))

        median_l = float(np.median(l_star))
        p10_l = float(np.percentile(l_star, 10))
        p90_l = float(np.percentile(l_star, 90))
        mean_chroma = float(np.mean(chroma))
        p90_chroma = float(np.percentile(chroma, 90))
        mean_a = float(np.mean(a_star))
        mean_b = float(np.mean(b_star))

        neutral = (mean_chroma < 16.0) and (p90_chroma < 26.0)
        return {
            "medianL": round(median_l, 2),
            "p10L": round(p10_l, 2),
            "p90L": round(p90_l, 2),
            "meanChroma": round(mean_chroma, 2),
            "p90Chroma": round(p90_chroma, 2),
            "meanA": round(mean_a, 2),
            "meanB": round(mean_b, 2),
            "isNeutral": bool(neutral),
        }
    except Exception:
        return {}

def _extract_alpha_mask(image: Image.Image, threshold: int = 24) -> Optional[np.ndarray]:
    """
    Returns a foreground mask from alpha when available.
    Useful for extracted cloth PNGs where large white/gray background exists.
    """
    try:
        rgba = image.convert("RGBA")
        alpha = np.array(rgba, dtype=np.uint8)[:, :, 3]
        mask = alpha >= int(max(1, threshold))
        keep = int(np.sum(mask))
        if keep < 64:
            return None
        coverage = float(keep) / float(mask.shape[0] * mask.shape[1])
        if coverage >= 0.985:
            # Mostly opaque image; alpha likely not meaningful for foreground isolation.
            return None
        return mask
    except Exception:
        return None

def _estimate_foreground_mask_from_border(
    image: Image.Image,
    quant_step: int = 16,
    bg_tolerance: int = 28,
) -> Optional[np.ndarray]:
    """
    Border-based foreground estimation for images without useful alpha.
    Helps isolate garment pixels when background is flat/near-flat.
    """
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        h, w = arr.shape[:2]
        if h < 8 or w < 8:
            return None

        b = max(1, min(h, w) // 30)
        border_pixels = np.concatenate(
            [
                arr[:b, :, :].reshape(-1, 3),
                arr[h - b :, :, :].reshape(-1, 3),
                arr[:, :b, :].reshape(-1, 3),
                arr[:, w - b :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        if border_pixels.size == 0:
            return None

        binned = (border_pixels // quant_step) * quant_step
        unique, counts = np.unique(binned, axis=0, return_counts=True)
        bg_rgb = unique[int(np.argmax(counts))]

        diff = arr.astype(np.int16) - bg_rgb.astype(np.int16)
        dist2 = np.sum(diff * diff, axis=2)
        fg_mask = dist2 > int(bg_tolerance * bg_tolerance)
        keep = int(np.sum(fg_mask))
        if keep < 64:
            return None
        coverage = float(keep) / float(h * w)
        if coverage < 0.02 or coverage > 0.98:
            return None
        return fg_mask
    except Exception:
        return None

def _erode_binary_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    arr = np.asarray(mask).astype(bool)
    if int(iterations) <= 0:
        return arr
    try:
        import cv2

        eroded = cv2.erode(
            arr.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=max(1, int(iterations)),
        )
        return eroded > 0
    except Exception:
        return arr

def _get_clean_foreground_mask(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Builds a high-confidence garment mask for color profiling.
    Combines strong alpha cues with optional external mask and edge erosion.
    """
    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        h, w = arr.shape[:2]
        min_keep = max(
            int(FLUX2_COLOR_DECONTAM_MIN_PIXELS),
            int(float(h * w) * float(FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO)),
        )

        base_mask: Optional[np.ndarray] = None
        if isinstance(mask, np.ndarray) and mask.shape[:2] == (h, w):
            base_mask = np.asarray(mask).astype(bool)

        alpha_high = _extract_alpha_mask(image, threshold=int(FLUX2_COLOR_DECONTAM_ALPHA_HIGH))
        alpha_low = _extract_alpha_mask(image, threshold=int(FLUX2_COLOR_DECONTAM_ALPHA_LOW))
        alpha_mask = alpha_high if isinstance(alpha_high, np.ndarray) else alpha_low

        combined = None
        if isinstance(alpha_mask, np.ndarray) and alpha_mask.shape[:2] == (h, w):
            combined = np.asarray(alpha_mask).astype(bool)
            if isinstance(base_mask, np.ndarray):
                intersect = combined & base_mask
                if int(np.sum(intersect)) >= int(min_keep):
                    combined = intersect
                elif int(np.sum(base_mask)) >= int(min_keep):
                    combined = base_mask
        elif isinstance(base_mask, np.ndarray):
            combined = base_mask
        else:
            fallback = _extract_alpha_mask(image, threshold=72)
            if not isinstance(fallback, np.ndarray):
                fallback = _extract_alpha_mask(image, threshold=24)
            if not isinstance(fallback, np.ndarray):
                fallback = _estimate_foreground_mask_from_border(image)
            if isinstance(fallback, np.ndarray) and fallback.shape[:2] == (h, w):
                combined = np.asarray(fallback).astype(bool)

        if not isinstance(combined, np.ndarray):
            return None
        if int(np.sum(combined)) < int(min_keep):
            return None

        eroded = _erode_binary_mask(combined, iterations=int(FLUX2_COLOR_DECONTAM_ERODE_ITERS))
        if int(np.sum(eroded)) >= int(min_keep):
            combined = eroded

        return combined.astype(bool)
    except Exception as e:
        logger.warning(f"Clean foreground mask extraction failed: {e}")
        return None

def _trim_lab_profile_outliers(
    lab_pixels: np.ndarray,
    dark_percentile: float,
    bright_percentile: float,
    min_pixels: int = 64,
) -> np.ndarray:
    """
    Trims extreme lightness tails for robust neutral color profiling.
    """
    try:
        if not isinstance(lab_pixels, np.ndarray) or lab_pixels.ndim != 2 or lab_pixels.shape[1] != 3:
            return lab_pixels
        if int(lab_pixels.shape[0]) < max(16, int(min_pixels)):
            return lab_pixels
        lo = float(max(0.0, min(100.0, dark_percentile)))
        hi = float(max(0.0, min(100.0, bright_percentile)))
        if hi <= lo:
            return lab_pixels

        l_star = lab_pixels[:, 0] * (100.0 / 255.0)
        p_low = float(np.percentile(l_star, lo))
        p_high = float(np.percentile(l_star, hi))
        keep = (l_star >= p_low) & (l_star <= p_high)
        if int(np.sum(keep)) < max(16, int(min_pixels)):
            return lab_pixels
        return lab_pixels[keep]
    except Exception:
        return lab_pixels

def _extract_masked_dominant_color_labels(image: Image.Image, top_k: int = 3) -> List[str]:
    """
    Foreground-aware color extraction for cloth images.
    Uses alpha when available; otherwise estimates foreground from border.
    """
    mask = _get_clean_foreground_mask(image)
    if not isinstance(mask, np.ndarray):
        mask = _extract_alpha_mask(image, threshold=72)
    if not isinstance(mask, np.ndarray):
        mask = _extract_alpha_mask(image, threshold=24)
    if not isinstance(mask, np.ndarray):
        mask = _estimate_foreground_mask_from_border(image)
    if not isinstance(mask, np.ndarray):
        return []

    hexes = _extract_dominant_hex_colors(image, mask=mask, top_k=max(2, int(top_k) + 2))
    out: List[str] = []
    for hx in hexes:
        token = str(hx or "").strip()
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", token):
            continue
        r = int(token[1:3], 16)
        g = int(token[3:5], 16)
        b = int(token[5:7], 16)
        label = _nearest_color_label((r, g, b))
        if label == "grey":
            label = "gray"
        if label not in out:
            out.append(label)
        if len(out) >= max(1, int(top_k)):
            break
    return out


def _color_labels_from_hex_palette(hexes: List[str], top_k: int = 3) -> List[str]:
    out: List[str] = []
    for hx in hexes:
        token = str(hx or "").strip()
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", token):
            continue
        r = int(token[1:3], 16)
        g = int(token[3:5], 16)
        b = int(token[5:7], 16)
        label = _nearest_color_label((r, g, b))
        if label == "grey":
            label = "gray"
        label = _canonical_color_token(label)
        if label and label not in out:
            out.append(label)
        if len(out) >= max(1, int(top_k)):
            break
    return out

def _extract_detail_lock_terms(description: str, max_items: int = 6) -> List[str]:
    low = str(description or "").lower()
    if not low:
        return []
    hits: List[str] = []

    def _needle_present(needle: str) -> bool:
        token = str(needle or "").strip().lower()
        if not token:
            return False
        if token == "asym":
            return bool(re.search(r"\basym(?:metric|metry)?\b", low))
        if (" " in token) or ("-" in token):
            pattern = r"\b" + re.escape(token).replace(r"\ ", r"[\s-]+") + r"\b"
            return bool(re.search(pattern, low))
        pattern = r"\b" + re.escape(token) + r"(?:s|ed|ing)?\b"
        return bool(re.search(pattern, low))

    for needle, phrase in _DETAIL_LOCK_TERMS:
        if _needle_present(needle) and phrase not in hits:
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
            normalized = _canonical_color_token(term)
            hits.append(normalized)
            if len(hits) >= max_items:
                break
    return hits

def _garment_color_context_settings() -> GarmentColorContextSettings:
    return GarmentColorContextSettings(
        top_k=int(FLUX2_COLOR_LOCK_TOP_K),
        palette_top_k=7,
        palette_min_area_percent=float(FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT),
        decontamination_enabled=bool(FLUX2_COLOR_DECONTAMINATION_ENABLED),
        decontam_alpha_high=int(FLUX2_COLOR_DECONTAM_ALPHA_HIGH),
        decontam_alpha_low=int(FLUX2_COLOR_DECONTAM_ALPHA_LOW),
        decontam_erode_iters=int(FLUX2_COLOR_DECONTAM_ERODE_ITERS),
        decontam_min_pixels=int(FLUX2_COLOR_DECONTAM_MIN_PIXELS),
        decontam_min_coverage_ratio=float(FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO),
        profile_trim_dark_percentile=float(FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE),
        profile_trim_bright_percentile=float(FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE),
        disable_masking=bool(COLOR_CONTEXT_DISABLE_MASKING),
    )

def _build_flux2_visual_lock_clauses(
    product_images: List[Image.Image],
    garment_descriptions: List[str],
    garment_metadata_list: Optional[List[Dict[str, object]]] = None,
) -> dict:
    shared = _shared_build_visual_lock_clauses(
        product_images=product_images,
        garment_descriptions=garment_descriptions,
        settings=_garment_color_context_settings(),
        transparency_signal_fn=_has_transparency_signal,
    )
    detail_terms: List[str] = []
    if FLUX2_DETAIL_LOCK_ENABLED:
        for desc in garment_descriptions:
            for term in _extract_detail_lock_terms(desc):
                if term not in detail_terms:
                    detail_terms.append(term)

    detail_clause = ""
    if detail_terms:
        detail_clause = (
            "Detail lock from image 2: preserve "
            + ", ".join(detail_terms[:6])
            + ". Do not simplify or replace these details. "
        )

    out = dict(shared or {})
    metadata_colors = [
        _extract_garment_metadata_color_block(
            garment_metadata_list[idx] if garment_metadata_list and idx < len(garment_metadata_list) else {}
        )
        for idx in range(len(product_images))
    ]
    if any(block.get("dominant_hexes") or block.get("color_hints") for block in metadata_colors):
        color_palettes = [list(v) for v in (out.get("color_palettes") or [])]
        color_hints = [list(v) for v in (out.get("color_hints") or [])]
        color_profiles = list(out.get("color_profiles") or [])
        color_palette_metrics = [list(v) for v in (out.get("color_palette_metrics") or [])]
        accent_hints = [list(v) for v in (out.get("accent_hints") or [])]
        contexts = list(out.get("contexts") or [])
        max_len = len(product_images)
        while len(color_palettes) < max_len:
            color_palettes.append([])
        while len(color_hints) < max_len:
            color_hints.append([])
        while len(color_profiles) < max_len:
            color_profiles.append({})
        while len(color_palette_metrics) < max_len:
            color_palette_metrics.append([])
        while len(accent_hints) < max_len:
            accent_hints.append([])
        while len(contexts) < max_len:
            contexts.append({})

        for idx, block in enumerate(metadata_colors):
            meta_hexes = [str(v).strip().upper() for v in (block.get("dominant_hexes") or []) if str(v).strip()]
            meta_hints = [str(v).strip().lower() for v in (block.get("color_hints") or []) if str(v).strip()]
            meta_accents = [str(v).strip().upper() for v in (block.get("accent_hexes") or []) if str(v).strip()]
            meta_profile = block.get("profile") if isinstance(block.get("profile"), dict) else {}
            if meta_hexes:
                color_palettes[idx] = meta_hexes
                color_palette_metrics[idx] = [{"hex": hx, "areaPercent": 0.0, "pixelCount": 0} for hx in meta_hexes]
            if meta_hints:
                color_hints[idx] = meta_hints
            if meta_profile:
                color_profiles[idx] = meta_profile
            if meta_accents:
                accent_hints[idx] = []
            ctx = contexts[idx] if isinstance(contexts[idx], dict) else {}
            if meta_hexes:
                ctx["dominantHexes"] = meta_hexes
                ctx["paletteHexes"] = meta_hexes
            if meta_hints:
                ctx["colorHints"] = meta_hints
                ctx["hints"] = meta_hints
            if meta_accents:
                ctx["accentHexes"] = meta_accents
            if meta_profile:
                ctx["profile"] = meta_profile
            if block.get("mask_source"):
                ctx["maskSource"] = block.get("mask_source")
            contexts[idx] = ctx

        out["color_palettes"] = color_palettes
        out["color_hints"] = color_hints
        out["color_profiles"] = color_profiles
        out["color_palette_metrics"] = color_palette_metrics
        out["accent_hints"] = accent_hints
        out["contexts"] = contexts

        non_empty_color_hints = [colors for colors in color_hints if colors]
        color_clause = ""
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

        neutral_lock_items: List[str] = []
        max_items = max(len(color_hints), len(color_palette_metrics), len(color_profiles))
        for idx in range(max_items):
            colors = color_hints[idx] if idx < len(color_hints) else []
            profile = color_profiles[idx] if idx < len(color_profiles) else {}
            profile_is_neutral = bool(profile.get("isNeutral"))
            head = colors[:3]
            neutral_count = sum(1 for color in head if _is_neutral_color_token(color))
            colors_look_neutral = bool(head) and (neutral_count >= max(1, (len(head) + 1) // 2))
            if not (colors_look_neutral or profile_is_neutral):
                continue
            palette = color_palette_metrics[idx] if idx < len(color_palette_metrics) else []
            dominant_hexes: List[str] = []
            for entry in palette[:3]:
                hx = str(entry.get("hex", "")).strip().upper()
                if re.fullmatch(r"#[0-9A-F]{6}", hx):
                    dominant_hexes.append(hx)
            target_l = profile.get("medianL")
            target_c = profile.get("meanChroma")
            target_b = profile.get("meanB")
            profile_tag = ""
            if isinstance(target_l, (int, float)) and isinstance(target_c, (int, float)):
                profile_tag = f", L*~{float(target_l):.1f}, C*~{float(target_c):.1f}"
                if isinstance(target_b, (int, float)):
                    profile_tag += f", b*~{float(target_b):.1f}"
            if dominant_hexes:
                neutral_lock_items.append(f"item {idx + 1}: {'/'.join(dominant_hexes)}{profile_tag}")
            else:
                neutral_lock_items.append(f"item {idx + 1}: neutral tones{profile_tag}")
        if neutral_lock_items:
            color_clause += (
                "Neutral tone lock: preserve exact lightness depth and undertone from image 2 ("
                + "; ".join(neutral_lock_items)
                + "). Avoid over-brightening to white/cream, avoid darkening to charcoal, "
                + "and avoid metallic/glossy silver sheen unless explicitly present in source fabric. "
                + "Keep neutral luminance close to source (roughly +/-4 L*). "
            )

        profile_lock_items: List[str] = []
        for idx in range(max_items):
            profile = color_profiles[idx] if idx < len(color_profiles) else {}
            if not isinstance(profile, dict):
                continue
            median_l = profile.get("medianL")
            mean_c = profile.get("meanChroma")
            if not (isinstance(median_l, (int, float)) and isinstance(mean_c, (int, float))):
                continue
            palette = color_palette_metrics[idx] if idx < len(color_palette_metrics) else []
            hue_deg = _palette_weighted_hue_deg(palette, max_colors=4)
            if _profile_is_near_white(profile):
                profile_lock_items.append(
                    f"item {idx + 1}: near-white base (L*~{float(median_l):.1f}, C*~{float(mean_c):.1f})"
                )
                continue
            if float(mean_c) >= 18.0 and isinstance(hue_deg, (int, float)):
                profile_lock_items.append(
                    f"item {idx + 1}: hue~{float(hue_deg):.0f} deg, L*~{float(median_l):.1f}, C*~{float(mean_c):.1f}"
                )
        if profile_lock_items:
            color_clause += (
                "Tone and saturation lock from image 2: preserve hue/lightness/saturation profile ("
                + "; ".join(profile_lock_items)
                + "). Avoid over-saturating, neon amplification, or warm/cool hue drift. "
                + "Do not shift yellow toward orange/red, and do not shift white/off-white toward gray/silver. "
            )
        out["color_clause"] = color_clause

    out["detail_clause"] = detail_clause
    out["detail_terms"] = detail_terms[:6]
    return out

def _build_flux2_hex_color_guard_clause(
    color_palette_metrics: List[List[Dict[str, object]]],
    max_items: int = 3,
    max_colors_per_item: int = 4,
) -> str:
    """
    Stronger prompt suffix for one-shot color correction re-run.
    """
    if not color_palette_metrics:
        return ""

    item_lines: List[str] = []
    for idx, palette in enumerate(color_palette_metrics[: max(1, int(max_items))], start=1):
        if not palette:
            continue
        filtered_palette = _filter_palette_entries_by_area(
            palette=palette,
            min_area_percent=float(FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT),
            min_items=2,
        )
        tokens: List[str] = []
        for entry in filtered_palette[: max(1, int(max_colors_per_item))]:
            hx = str(entry.get("hex", "")).upper().strip()
            if not re.fullmatch(r"#[0-9A-F]{6}", hx):
                continue
            area = float(entry.get("areaPercent", 0.0) or 0.0)
            if area > 0.0:
                tokens.append(f"{hx} (~{area:.1f}%)")
            else:
                tokens.append(hx)
        if tokens:
            item_lines.append(f"item {idx}: " + ", ".join(tokens))

    if not item_lines:
        return ""
    return (
        " Color-guard correction: preserve exact source palette from image 2 by hex. "
        + "; ".join(item_lines)
        + ". Do not shift hue, saturation, or brightness family. "
    )

def _build_single_image_color_context(
    image: Image.Image,
    description: str = "",
    mask: Optional[np.ndarray] = None,
    top_k: int = 7,
    force_masking: bool = False,
) -> Dict[str, object]:
    settings = _garment_color_context_settings()
    effective_mask = None if COLOR_CONTEXT_DISABLE_MASKING else mask
    if force_masking:
        settings = replace(settings, disable_masking=False)
        effective_mask = mask
    return _shared_build_single_image_color_context(
        image=image,
        description=description,
        mask=effective_mask,
        settings=settings,
        top_k=top_k,
    )

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
        "Identity-preserving photorealistic virtual try-on image edit, not a new photoshoot. "
        "Keep the original camera framing, background, and lighting from image 1. "
        "Use image 1 as strict identity source for face, body shape, skin tone, and pose. "
        f"TRANSFER the {target_hint} from image 2 onto the person in image 1. "
        "Match exact garment attributes from image 2: color tone, print/pattern, neckline, sleeve length, hem length, fit, and trims. "
        "Preserve exact source garment color arrangement from image 2: dominant tone, accent tone, border/trim colors, "
        "brightness depth, undertone, and contrast distribution across the fabric. "
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
            "Do not alter face pixels, hairstyle, or body posture while replacing clothing. "
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
        if types == {"outer"}:
            prompt += (
                "Outerwear-only lock: keep the original lower-body garment completely unchanged "
                "(pants/skirt/shorts/shoes, including silhouette, color, texture, folds, and hem). "
                "Keep the original inner top from image 1 unchanged except where the outerwear naturally covers it. "
                "Do not recolor or redesign the pants, and do not replace the full outfit when only outerwear is targeted. "
            )

    if is_multi:
        prompt += (
            "Image 2 is a multi-item outfit board; apply all listed items together with coherent layering and fit. "
            "Keep each item isolated: no cross-item color bleed, no print transfer, and no texture mixing between items. "
        )
        if collage_item_clause:
            prompt += collage_item_clause

    prompt += (
        "Preserve the exact number of visible arms and hands from image 1. "
        "Never generate an extra hand, extra arm, duplicate fingers, mirrored limb, or floating hand. "
        "Hands must stay naturally attached to the original arms and remain separate from the transferred garment, "
        "except for realistic occlusion caused by the original pose. "
    )

    if is_dress_mode:
        prompt += (
            f"Identity reference from image 1: {identity_context}. "
            "Keep exact same person identity: facial features, skin tone, hair, hands, body proportions, "
            "and scene lighting. Keep exact same head angle, facial expression, and hair silhouette. "
            "No face replacement, no age shift, and no body-shape change. "
            "Ensure realistic fabric drape, seams, folds, and shadows."
        )
    else:
        prompt += (
            f"Person and current outfit reference from image 1: {user_description}. "
            "Keep exact same person identity: facial features, skin tone, hair, hands, body proportions, "
            "and scene lighting. Keep exact same head angle and facial expression. "
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
    elif types == {"outer"}:
        parts.append(
            "modified pants, modified skirt, modified shorts, modified shoes, altered lower-body garment color, "
            "altered lower-body garment structure, recolored inner top, full outfit replacement instead of outerwear-only edit"
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
        "over-brightened fabric",
        "washed-out garment color",
        "white cast on garment",
        "metallic sheen",
        "glossy silver cast",
        "wrong border color",
        "wrong trim color",
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
        "extra hand",
        "third hand",
        "duplicate arm",
        "floating hand",
        "hand fused to garment",
        "garment fused to skin",
        "duplicate fingers",
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
    elif types == {"outer"}:
        base_terms.extend(
            [
                "changed pants",
                "changed skirt",
                "changed shorts",
                "changed shoes",
                "modified lower-body garment color",
                "modified lower-body garment structure",
                "recolored inner top",
                "full outfit replacement",
            ]
        )

    return ", ".join(dict.fromkeys(base_terms))

def _normalize_descriptor_backend(raw: Optional[str]) -> str:
    value = str(raw or FLUX2_DESCRIPTOR_BACKEND).strip().lower()
    if value not in {"florence", "qwen2_5_vl", "joycaption", "minicpm", "minicpm_service"}:
        return "minicpm"
    if value == "qwen2_5_vl" and not FLUX2_ALLOW_QWEN_BACKEND:
        return "minicpm"
    return value


_MINICPM_SERVICE_SESSION = requests.Session()
_MINICPM_SERVICE_ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=MINICPM_SERVICE_POOL_MAXSIZE,
    pool_maxsize=MINICPM_SERVICE_POOL_MAXSIZE,
    max_retries=0,
)
_MINICPM_SERVICE_SESSION.mount("http://", _MINICPM_SERVICE_ADAPTER)
_MINICPM_SERVICE_SESSION.mount("https://", _MINICPM_SERVICE_ADAPTER)
_MINICPM_DESCRIPTOR_CACHE: Dict[str, Tuple[float, str]] = {}
_MINICPM_DESCRIPTOR_CACHE_LOCK = threading.Lock()


def _image_signature_for_cache(image: Optional[Image.Image]) -> str:
    if not isinstance(image, Image.Image):
        return ""
    try:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.size <= 0:
            return ""
        h = hashlib.sha256()
        h.update(arr.tobytes())
        h.update(f"{rgb.width}x{rgb.height}".encode("utf-8"))
        return h.hexdigest()
    except Exception:
        return ""


def _minicpm_service_cache_key(
    kind: str,
    image_url: str,
    max_new_tokens: int,
    prompt: str,
    image_signature: str = "",
    service_url: str = "",
) -> str:
    cache_source = str(image_signature or image_url or "").strip()
    base = str(service_url or "").strip().lower()
    raw = f"{kind}|{base}|{cache_source}|{int(max_new_tokens)}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _minicpm_service_cache_get(cache_key: str) -> Optional[str]:
    if not MINICPM_SERVICE_CACHE_ENABLED or MINICPM_SERVICE_CACHE_TTL_SECONDS <= 0:
        return None
    now = time.time()
    with _MINICPM_DESCRIPTOR_CACHE_LOCK:
        cached = _MINICPM_DESCRIPTOR_CACHE.get(cache_key)
        if not cached:
            return None
        ts, value = cached
        if (now - float(ts)) > float(MINICPM_SERVICE_CACHE_TTL_SECONDS):
            _MINICPM_DESCRIPTOR_CACHE.pop(cache_key, None)
            return None
        return value


def _minicpm_service_cache_put(cache_key: str, value: str) -> None:
    if not MINICPM_SERVICE_CACHE_ENABLED or MINICPM_SERVICE_CACHE_TTL_SECONDS <= 0:
        return
    now = time.time()
    text = str(value or "").strip()
    if not text:
        return
    with _MINICPM_DESCRIPTOR_CACHE_LOCK:
        _MINICPM_DESCRIPTOR_CACHE[cache_key] = (now, text)
        if len(_MINICPM_DESCRIPTOR_CACHE) <= MINICPM_SERVICE_CACHE_MAX_ENTRIES:
            return
        oldest_key = min(_MINICPM_DESCRIPTOR_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _MINICPM_DESCRIPTOR_CACHE.pop(oldest_key, None)


def _describe_with_minicpm_service(
    image_url: Optional[str],
    kind: str,
    image_signature: str = "",
    service_url: Optional[str] = None,
    prompt_override: Optional[str] = None,
    return_raw: bool = False,
) -> str:
    """
    Fetch descriptor text from standalone MiniCPM service using source image URL.
    """
    clean_url = str(image_url or "").strip()
    if not clean_url:
        raise RuntimeError("minicpm_service requires a valid image URL")
    base_url = str(service_url or MINICPM_SERVICE_URL or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("MINICPM_SERVICE_URL is not configured")

    if kind == "person":
        endpoint = f"{base_url}/describe/person"
        max_new_tokens = MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS
        prompt = MINICPM_SERVICE_PERSON_PROMPT
    else:
        endpoint = f"{base_url}/describe/garment"
        max_new_tokens = MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS
        prompt = MINICPM_SERVICE_GARMENT_PROMPT
    if prompt_override:
        prompt = str(prompt_override).strip() or prompt

    payload = {
        "image_url": clean_url,
        "max_new_tokens": int(max_new_tokens),
        "prompt": prompt,
    }
    cache_key = _minicpm_service_cache_key(
        kind=str(kind or "").strip().lower(),
        image_url=clean_url,
        max_new_tokens=int(max_new_tokens),
        prompt=prompt,
        image_signature=str(image_signature or "").strip(),
        service_url=base_url,
    )
    if return_raw:
        cache_key = f"{cache_key}::raw"
    cached = _minicpm_service_cache_get(cache_key)
    if cached:
        return cached

    resp = _MINICPM_SERVICE_SESSION.post(
        endpoint,
        json=payload,
        timeout=(MINICPM_SERVICE_CONNECT_TIMEOUT_S, MINICPM_SERVICE_TIMEOUT_S),
    )
    if resp.status_code != 200:
        detail = resp.text[:500]
        raise RuntimeError(
            f"minicpm_service {kind} failed: status={resp.status_code} detail={detail}"
        )
    data = resp.json() if resp.content else {}
    text = str(data.get("text", "")).strip()
    if not text:
        raise RuntimeError(f"minicpm_service {kind} returned empty text")
    result_text = text if return_raw else _normalize_minicpm_descriptor_text(text, kind=kind)
    _minicpm_service_cache_put(cache_key, result_text)
    return result_text

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
    service_url: Optional[str] = None,
    garment_type: Optional[str] = None,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
) -> str:
    resolved = _normalize_descriptor_backend(backend)
    if resolved == "minicpm_service":
        try:
            if not image_url:
                raise RuntimeError("minicpm_service requires a valid image URL for garment description")
            cache_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            image_signature = _image_signature_for_cache(cache_img)
            raw_text = _describe_with_minicpm_service(
                image_url=image_url,
                kind="garment",
                image_signature=image_signature,
                service_url=service_url,
                prompt_override=_build_minicpm_garment_prompt(
                    garment_type=str(garment_type or ""),
                    dominant_color_hexes=dominant_color_hexes,
                    color_hints=color_hints,
                ),
                return_raw=True,
            )
            return _parse_garment_prompt_sections(raw_text, garment_type=garment_type).get(
                "base_garment_prompt", ""
            )
        except Exception as err:
            raise RuntimeError(f"MiniCPM service garment description failed: {err}") from err
    if resolved == "minicpm":
        try:
            minicpm_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            return str(engine.minicpm.describe_garment(minicpm_img)).strip()
        except Exception as err:
            raise RuntimeError(f"MiniCPM garment description failed: {err}") from err
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


def _describe_garment_prompt_bundle_with_backend(
    image: Image.Image,
    backend: str,
    image_url: Optional[str] = None,
    service_url: Optional[str] = None,
    garment_type: Optional[str] = None,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
) -> Dict[str, str]:
    resolved = _normalize_descriptor_backend(backend)
    if resolved == "minicpm_service":
        try:
            if not image_url:
                raise RuntimeError("minicpm_service requires a valid image URL for garment description")
            cache_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            image_signature = _image_signature_for_cache(cache_img)
            raw_text = _describe_with_minicpm_service(
                image_url=image_url,
                kind="garment",
                image_signature=image_signature,
                service_url=service_url,
                prompt_override=_build_minicpm_garment_prompt(
                    garment_type=str(garment_type or ""),
                    dominant_color_hexes=dominant_color_hexes,
                    color_hints=color_hints,
                ),
                return_raw=True,
            )
            return _parse_garment_prompt_sections(raw_text, garment_type=garment_type)
        except Exception as err:
            raise RuntimeError(f"MiniCPM service garment description failed: {err}") from err

    desc = _describe_garment_with_backend(
        image=image,
        backend=resolved,
        image_url=image_url,
        service_url=service_url,
        garment_type=garment_type,
        dominant_color_hexes=dominant_color_hexes,
        color_hints=color_hints,
    )
    return _parse_garment_prompt_sections(desc, garment_type=garment_type)

def _describe_user_image_for_flux2(
    user_img: Image.Image,
    backend: str = "florence",
    image_url: Optional[str] = None,
    service_url: Optional[str] = None,
) -> str:
    """
    Ask selected descriptor model for a full-person detailed description.
    """
    resolved = _normalize_descriptor_backend(backend)
    if resolved == "minicpm_service":
        try:
            if not image_url:
                raise RuntimeError("minicpm_service requires a valid image URL for person description")
            cache_img = _resize_for_qwen_caption(
                image=user_img,
                max_side=FLUX2_MINICPM_USER_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_USER_CAPTION_MIN_SIDE,
            )
            image_signature = _image_signature_for_cache(cache_img)
            return _describe_with_minicpm_service(
                image_url=image_url,
                kind="person",
                image_signature=image_signature,
                service_url=service_url,
            )
        except Exception as err:
            raise RuntimeError(f"MiniCPM service user description failed: {err}") from err
    if resolved == "minicpm":
        try:
            minicpm_img = _resize_for_qwen_caption(
                image=user_img,
                max_side=FLUX2_MINICPM_USER_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_USER_CAPTION_MIN_SIDE,
            )
            return str(engine.minicpm.describe_person_and_outfit(minicpm_img)).strip()
        except Exception as err:
            raise RuntimeError(f"MiniCPM user description failed: {err}") from err
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


def _hex_to_lab_triplet(hex_color: str) -> Optional[np.ndarray]:
    token = str(hex_color or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", token):
        return None
    r = int(token[0:2], 16)
    g = int(token[2:4], 16)
    b = int(token[4:6], 16)
    return rgb_to_lab(np.array([[r, g, b]], dtype=np.uint8))[0]


def _compute_palette_delta_e(
    input_hexes: List[str],
    output_hexes: List[str],
    max_colors: int = 5,
) -> Optional[Dict[str, object]]:
    input_labs: List[Tuple[str, np.ndarray]] = []
    output_labs: List[Tuple[str, np.ndarray]] = []

    for hx in input_hexes[: max(1, int(max_colors))]:
        lab = _hex_to_lab_triplet(hx)
        if isinstance(lab, np.ndarray):
            input_labs.append((str(hx).upper(), lab))
    for hx in output_hexes[: max(1, int(max_colors) + 2)]:
        lab = _hex_to_lab_triplet(hx)
        if isinstance(lab, np.ndarray):
            output_labs.append((str(hx).upper(), lab))

    if not input_labs or not output_labs:
        return None

    drifts: List[float] = []
    lightness_drifts: List[float] = []
    per_color: List[Dict[str, object]] = []
    for in_hx, in_lab in input_labs:
        distances = [
            (out_hx, out_lab, delta_e_cie76(in_lab, out_lab))
            for out_hx, out_lab in output_labs
        ]
        if not distances:
            continue
        best_out_hx, best_out_lab, best_drift = min(distances, key=lambda it: it[2])
        in_l = float(in_lab[0]) * (100.0 / 255.0)
        out_l = float(best_out_lab[0]) * (100.0 / 255.0)
        l_drift = abs(in_l - out_l)
        drifts.append(float(best_drift))
        lightness_drifts.append(float(l_drift))
        per_color.append(
            {
                "input": in_hx,
                "matchedOutput": best_out_hx,
                "drift": round(float(best_drift), 2),
                "lightnessDrift": round(float(l_drift), 2),
            }
        )

    if not drifts:
        return None

    return {
        "avgDrift": round(float(np.mean(drifts)), 2),
        "maxDrift": round(float(np.max(drifts)), 2),
        "avgLightnessDrift": round(float(np.mean(lightness_drifts)), 2) if lightness_drifts else 0.0,
        "comparedColors": int(len(drifts)),
        "perColor": per_color,
    }

def _score_color_fidelity(
    output_image: Image.Image,
    input_palettes: List[List[str]],
    target_types: List[str],
    input_profiles: Optional[List[Dict[str, object]]] = None,
) -> dict:
    """
    Computes perceptual color drift (CIELAB DeltaE) between input garments and generated output.
    Uses HumanParser to segment the output garment regions.
    """
    if not engine.parser:
        return {
            "status": "invalid",
            "error": "parser_unavailable",
            "total_drift": None,
            "total_lightness_drift": None,
            "total_profile_drift": None,
            "validComparisons": 0,
            "per_item": [],
        }

    try:
        parsing = engine.parser.parse(output_image)
        per_item = []
        drifts = []
        lightness_drifts = []
        valid_comparisons = 0
        profile_drifts: List[float] = []

        for idx, (target_type, input_hexes) in enumerate(zip(target_types, input_palettes)):
            if not input_hexes:
                per_item.append({"index": idx, "type": target_type, "reason": "empty_input_palette"})
                continue
            
            # Segment the specific category in the output
            mask = engine.parser.get_mask_for_category(parsing, target_type)
            if not isinstance(mask, np.ndarray) or int(np.sum(mask)) < 16:
                per_item.append({"index": idx, "type": target_type, "reason": "not_segmented"})
                continue

            # Extract output palette from the mask
            output_hexes = _extract_dominant_hex_colors(output_image, mask=mask, top_k=7)
            if not output_hexes:
                per_item.append({"index": idx, "type": target_type, "reason": "empty_output_palette"})
                continue

            drift_metrics = _compute_palette_delta_e(input_hexes, output_hexes, max_colors=5)
            if not drift_metrics:
                per_item.append(
                    {
                        "index": idx,
                        "type": target_type,
                        "input_palette": input_hexes,
                        "output_palette": output_hexes,
                        "reason": "invalid_palette_compare",
                    }
                )
                continue

            drift = float(drift_metrics["avgDrift"])
            drifts.append(drift)
            lightness_drifts.append(float(drift_metrics.get("avgLightnessDrift", 0.0)))
            valid_comparisons += 1

            profile_drift_value: Optional[float] = None
            if isinstance(input_profiles, list) and idx < len(input_profiles):
                in_profile = input_profiles[idx] if isinstance(input_profiles[idx], dict) else {}
                out_profile = _extract_lab_color_profile(output_image, mask=mask)
                in_med_l = in_profile.get("medianL")
                in_mean_c = in_profile.get("meanChroma")
                out_med_l = out_profile.get("medianL")
                out_mean_c = out_profile.get("meanChroma")
                if (
                    isinstance(in_med_l, (int, float))
                    and isinstance(in_mean_c, (int, float))
                    and isinstance(out_med_l, (int, float))
                    and isinstance(out_mean_c, (int, float))
                ):
                    l_term = abs(float(out_med_l) - float(in_med_l))
                    # Penalize chroma inflation more strongly for neutral garments.
                    c_term = max(0.0, float(out_mean_c) - float(in_mean_c))
                    profile_drift_value = float(l_term + (0.8 * c_term))
                    profile_drifts.append(profile_drift_value)

            per_item.append({
                "index": idx,
                "type": target_type,
                "input_palette": [str(h).upper() for h in input_hexes],
                "output_palette": [str(h).upper() for h in output_hexes],
                "drift": round(drift, 2),
                "maxDrift": drift_metrics["maxDrift"],
                "lightnessDrift": drift_metrics.get("avgLightnessDrift", 0.0),
                "profileDrift": round(profile_drift_value, 2) if isinstance(profile_drift_value, (int, float)) else None,
                "comparedColors": drift_metrics["comparedColors"],
                "perColor": drift_metrics["perColor"],
            })

        if not drifts:
            return {
                "total_drift": None,
                "total_lightness_drift": None,
                "total_profile_drift": None,
                "validComparisons": int(valid_comparisons),
                "per_item": per_item,
                "status": "invalid",
                "reason": "no_valid_color_comparisons",
            }

        total_drift = round(float(np.mean(drifts)), 2)
        total_lightness_drift = round(float(np.mean(lightness_drifts)), 2) if lightness_drifts else 0.0
        total_profile_drift = round(float(np.mean(profile_drifts)), 2) if profile_drifts else None
        return {
            "total_drift": total_drift,
            "total_lightness_drift": total_lightness_drift,
            "total_profile_drift": total_profile_drift,
            "validComparisons": int(valid_comparisons),
            "per_item": per_item,
            "status": "success"
        }
    except Exception as e:
        logger.warning(f"Color fidelity scoring failed: {e}")
        return {
            "status": "invalid",
            "error": str(e),
            "total_drift": None,
            "total_lightness_drift": None,
            "total_profile_drift": None,
            "validComparisons": 0,
            "per_item": [],
        }

def _apply_neutral_post_color_calibration(
    output_image: Image.Image,
    source_profile: Dict[str, object],
    target_type: str,
) -> Tuple[Image.Image, Dict[str, object]]:
    """
    Deterministic neutral-tone correction pass on selected result.
    Keeps structure, adjusts only garment chroma/lightness drift.
    """
    try:
        if not engine.parser:
            return output_image, {"applied": False, "reason": "parser_unavailable"}
        if not isinstance(source_profile, dict) or not source_profile.get("isNeutral"):
            return output_image, {"applied": False, "reason": "source_not_neutral"}

        parsing = engine.parser.parse(output_image)
        mask = engine.parser.get_mask_for_category(parsing, target_type)
        if not isinstance(mask, np.ndarray) or int(np.sum(mask)) < 64:
            return output_image, {"applied": False, "reason": "mask_unavailable"}

        out_profile = _extract_lab_color_profile(output_image, mask=mask)
        if not out_profile:
            return output_image, {"applied": False, "reason": "output_profile_unavailable"}

        in_l = source_profile.get("medianL")
        in_a = source_profile.get("meanA")
        in_b = source_profile.get("meanB")
        out_l = out_profile.get("medianL")
        out_a = out_profile.get("meanA")
        out_b = out_profile.get("meanB")
        if not all(isinstance(v, (int, float)) for v in (in_l, in_a, in_b, out_l, out_a, out_b)):
            return output_image, {"applied": False, "reason": "profile_values_invalid"}

        delta_l = float(in_l) - float(out_l)
        delta_a = float(in_a) - float(out_a)
        delta_b = float(in_b) - float(out_b)

        if abs(delta_l) > float(FLUX2_NEUTRAL_CALIBRATION_MAX_DELTA_L):
            return output_image, {
                "applied": False,
                "reason": "deltaL_exceeds_safety_limit",
                "deltaL": round(delta_l, 2),
                "deltaA": round(delta_a, 2),
                "deltaB": round(delta_b, 2),
                "limit": float(FLUX2_NEUTRAL_CALIBRATION_MAX_DELTA_L),
                "sourceMedianL": round(float(in_l), 2),
                "outputMedianL": round(float(out_l), 2),
            }

        if (
            delta_l < -float(FLUX2_NEUTRAL_CALIBRATION_MAX_DARKEN_LIGHT_OUTPUT)
            and float(out_l) > float(FLUX2_NEUTRAL_CALIBRATION_LIGHT_OUTPUT_MIN_L)
        ):
            return output_image, {
                "applied": False,
                "reason": "refusing_to_darken_light_output",
                "deltaL": round(delta_l, 2),
                "deltaA": round(delta_a, 2),
                "deltaB": round(delta_b, 2),
                "sourceMedianL": round(float(in_l), 2),
                "outputMedianL": round(float(out_l), 2),
            }

        if (
            delta_l > float(FLUX2_NEUTRAL_CALIBRATION_MAX_BRIGHTEN_DARK_OUTPUT)
            and float(out_l) < float(FLUX2_NEUTRAL_CALIBRATION_DARK_OUTPUT_MAX_L)
        ):
            return output_image, {
                "applied": False,
                "reason": "refusing_to_excessively_brighten_dark_output",
                "deltaL": round(delta_l, 2),
                "deltaA": round(delta_a, 2),
                "deltaB": round(delta_b, 2),
                "sourceMedianL": round(float(in_l), 2),
                "outputMedianL": round(float(out_l), 2),
            }

        if abs(delta_l) < 0.8 and abs(delta_a) < 0.8 and abs(delta_b) < 0.8:
            return output_image, {
                "applied": False,
                "reason": "drift_too_small",
                "deltaL": round(delta_l, 2),
                "deltaA": round(delta_a, 2),
                "deltaB": round(delta_b, 2),
                "sourceMedianL": round(float(in_l), 2),
                "outputMedianL": round(float(out_l), 2),
            }

        import cv2
        arr = np.array(output_image.convert("RGB"), dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        m = np.asarray(mask).astype(np.float32)
        if m.shape[:2] != arr.shape[:2]:
            m = resize_mask_to_image(m, arr.shape[1], arr.shape[0])
        m = np.clip(m, 0.0, 1.0)
        if cv2 is not None:
            m = cv2.GaussianBlur(m, (5, 5), 0)
        m3 = np.repeat(m[:, :, None], 3, axis=2)

        # Conservative correction factors to avoid texture flattening.
        lab[:, :, 0] = np.clip(lab[:, :, 0] + ((delta_l * (255.0 / 100.0) * 0.65) * m), 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + ((delta_a * 0.70) * m), 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + ((delta_b * 0.85) * m), 0, 255)

        corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        blended = ((corrected.astype(np.float32) * m3) + (arr.astype(np.float32) * (1.0 - m3))).clip(0, 255).astype(np.uint8)
        out_image = Image.fromarray(blended, mode="RGB")
        post_profile = _extract_lab_color_profile(out_image, mask=mask)
        post_l = post_profile.get("medianL") if isinstance(post_profile, dict) else None
        return out_image, {
            "applied": True,
            "deltaL": round(delta_l, 2),
            "deltaA": round(delta_a, 2),
            "deltaB": round(delta_b, 2),
            "sourceMedianL": round(float(in_l), 2),
            "outputMedianL": round(float(out_l), 2),
            "postMedianL": round(float(post_l), 2) if isinstance(post_l, (int, float)) else None,
        }
    except Exception as e:
        return output_image, {"applied": False, "reason": f"error:{e}"}

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

def _score_identity_face_preservation(
    reference_image: Image.Image,
    output_image: Image.Image,
) -> float:
    """
    Score identity preservation from upper-body/face region.
    Higher is better and focuses on face+skin consistency for try-on outputs.
    """
    try:
        ref = reference_image.convert("RGB").resize((256, 384), Image.BICUBIC)
        out = output_image.convert("RGB").resize((256, 384), Image.BICUBIC)
        ref_arr = np.asarray(ref, dtype=np.float32) / 255.0
        out_arr = np.asarray(out, dtype=np.float32) / 255.0

        h, w = ref_arr.shape[:2]
        y1 = max(8, int(h * 0.45))
        x0 = int(w * 0.18)
        x1 = max(x0 + 8, int(w * 0.82))
        ref_roi = ref_arr[:y1, x0:x1, :]
        out_roi = out_arr[:y1, x0:x1, :]
        if ref_roi.size == 0 or out_roi.size == 0:
            return 0.0

        mae = float(np.mean(np.abs(ref_roi - out_roi)))
        base_score = max(0.0, min(1.0, 1.0 - (mae / 0.30)))

        # Optional skin-tone consistency bonus using coarse YCrCb skin mask.
        ref_u8 = (ref_roi * 255.0).clip(0, 255).astype(np.uint8)
        out_u8 = (out_roi * 255.0).clip(0, 255).astype(np.uint8)
        ref_ycc = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2YCrCb)
        out_ycc = cv2.cvtColor(out_u8, cv2.COLOR_RGB2YCrCb)

        ref_skin = (
            (ref_ycc[:, :, 1] >= 133) & (ref_ycc[:, :, 1] <= 173)
            & (ref_ycc[:, :, 2] >= 77) & (ref_ycc[:, :, 2] <= 127)
        )
        out_skin = (
            (out_ycc[:, :, 1] >= 133) & (out_ycc[:, :, 1] <= 173)
            & (out_ycc[:, :, 2] >= 77) & (out_ycc[:, :, 2] <= 127)
        )

        if int(ref_skin.sum()) < 128 or int(out_skin.sum()) < 128:
            return float(base_score)

        ref_lab = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        out_lab = cv2.cvtColor(out_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_mean = np.mean(ref_lab[ref_skin], axis=0)
        out_mean = np.mean(out_lab[out_skin], axis=0)
        skin_delta = float(np.linalg.norm(ref_mean - out_mean))
        skin_score = max(0.0, min(1.0, 1.0 - (skin_delta / 28.0)))
        return float((0.70 * base_score) + (0.30 * skin_score))
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

def _build_user_prepare_payload(
    *,
    status_code: int,
    message: str,
    url: Optional[str] = None,
    prompt_description: Optional[str] = None,
) -> dict:
    payload: dict = {
        "status": int(status_code),
        "message": str(message or ""),
    }
    if int(status_code) == 200:
        payload["data"] = {
            "url": str(url or ""),
            "promptDescription": str(prompt_description or ""),
        }
    return payload

def _user_prep_person_components(image: Image.Image) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    if engine.parser is None:
        raise RuntimeError("Human parser is not available")

    parsing = engine.parser.parse(image)
    h, w = parsing.shape[:2]
    total_pixels = max(1, int(h * w))
    min_pixels = max(64, int(total_pixels * USER_PREP_COMPONENT_MIN_AREA_RATIO))

    body_ids = _parser_category_ids("body", fallback=[2, 11, 12, 13, 14, 15, 16])
    garment_ids = _parser_category_ids("garment_fallback", fallback=[4, 5, 6, 7, 8, 17])
    person_ids = sorted(set(int(v) for v in (body_ids + garment_ids)))
    if not person_ids:
        person_ids = [2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17]

    person_mask = np.isin(parsing, person_ids)
    components = _mask_connected_components(person_mask, min_pixels=min_pixels)
    meta = {
        "person_ids": person_ids,
        "components": int(len(components)),
        "min_pixels": int(min_pixels),
    }
    return parsing, person_mask, components, meta

def _user_prep_select_main_component(components: list[dict], width: int, height: int) -> Optional[dict]:
    if not components:
        return None

    total_pixels = max(1.0, float(width * height))
    ranked: List[Tuple[float, dict]] = []
    for comp in components:
        bbox = comp.get("bbox") or [0, 0, 0, 0]
        area = int(comp.get("area", 0))
        area_ratio = float(area) / total_pixels
        center_prior = _bbox_prior([int(v) for v in bbox], width, height)
        score = (0.78 * area_ratio) + (0.22 * center_prior)
        rank_item = dict(comp)
        rank_item["area_ratio"] = float(area_ratio)
        rank_item["center_prior"] = float(center_prior)
        rank_item["main_person_score"] = float(score)
        ranked.append((score, rank_item))

    ranked.sort(key=lambda x: float(x[0]), reverse=True)
    return ranked[0][1] if ranked else None

def _parser_ids_for_aliases(aliases: List[str], fallback: Optional[List[int]] = None) -> list[int]:
    label2id = _parser_runtime_label2id()
    ids: list[int] = []
    for alias in aliases:
        key = str(alias).strip().lower().replace("-", "_").replace(" ", "_")
        if key in label2id:
            try:
                ids.append(int(label2id[key]))
            except Exception:
                continue
    if ids:
        return sorted(set(ids))
    return [int(v) for v in (fallback or [])]

def _user_prep_face_component_from_parsing(
    parsing: np.ndarray,
    person_component_mask: Optional[np.ndarray] = None,
) -> Optional[dict]:
    face_ids = _parser_ids_for_aliases(
        ["face", "human_face", "head", "heads", "facial"],
        fallback=[11],
    )
    if not face_ids:
        return None
    face_mask = np.isin(parsing, face_ids)
    if isinstance(person_component_mask, np.ndarray) and person_component_mask.shape == face_mask.shape:
        face_mask = face_mask & person_component_mask.astype(bool)

    min_pixels = max(16, int(USER_PREP_FACE_MIN_PIXELS))
    comps = _mask_connected_components(face_mask, min_pixels=min_pixels)
    if not comps:
        return None
    best = comps[0]
    bbox = [int(v) for v in (best.get("bbox") or [0, 0, 0, 0])]
    area = int(best.get("area", 0))
    total = max(1.0, float(face_mask.size))
    return {
        "source": "parser_face",
        "bbox": bbox,
        "area": area,
        "area_ratio": float(area / total),
    }

def _user_prep_face_component_from_haar(image: Image.Image) -> Optional[dict]:
    try:
        import cv2
    except Exception:
        return None

    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return None

    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ]
    boxes: list[list[int]] = []
    for name in cascade_names:
        try:
            path = os.path.join(cv2.data.haarcascades, name)
            if not os.path.exists(path):
                continue
            detector = cv2.CascadeClassifier(path)
            if detector.empty():
                continue
            found = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(USER_PREP_FACE_MIN_SIDE_PX, USER_PREP_FACE_MIN_SIDE_PX),
            )
            for (x, y, w, h) in found:
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
        except Exception:
            continue
    if not boxes:
        return None

    def _area(b: list[int]) -> int:
        return max(1, int((b[2] - b[0]) * (b[3] - b[1])))

    best = sorted(boxes, key=_area, reverse=True)[0]
    area = _area(best)
    total = max(1.0, float(image.width * image.height))
    return {
        "source": "haar_face",
        "bbox": [int(v) for v in best],
        "area": int(area),
        "area_ratio": float(area / total),
    }

def _user_prep_validate_face(
    image: Image.Image,
    parsing: Optional[np.ndarray] = None,
    person_component_mask: Optional[np.ndarray] = None,
) -> tuple[bool, dict]:
    candidates: list[dict] = []
    if isinstance(parsing, np.ndarray):
        parser_face = _user_prep_face_component_from_parsing(parsing, person_component_mask=person_component_mask)
        if parser_face:
            candidates.append(parser_face)
    haar_face = _user_prep_face_component_from_haar(image)
    if haar_face:
        candidates.append(haar_face)

    if not candidates:
        return False, {"reason": "face_not_detected"}

    best = sorted(candidates, key=lambda c: float(c.get("area", 0)), reverse=True)[0]
    x0, y0, x1, y1 = [int(v) for v in (best.get("bbox") or [0, 0, 0, 0])]
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    area = int(best.get("area", fw * fh))
    area_ratio = float(best.get("area_ratio", 0.0))

    if fw < USER_PREP_FACE_MIN_SIDE_PX or fh < USER_PREP_FACE_MIN_SIDE_PX:
        return False, {
            "reason": "face_too_small",
            "source": best.get("source"),
            "bbox": [x0, y0, x1, y1],
            "face_w": fw,
            "face_h": fh,
            "face_area_ratio": area_ratio,
        }
    if area < USER_PREP_FACE_MIN_PIXELS or area_ratio < USER_PREP_FACE_MIN_AREA_RATIO:
        return False, {
            "reason": "face_area_below_threshold",
            "source": best.get("source"),
            "bbox": [x0, y0, x1, y1],
            "face_area": int(area),
            "face_area_ratio": area_ratio,
        }

    face_patch = image.crop((x0, y0, x1, y1)).convert("RGB")
    focus = _focus_score(face_patch)
    if focus < USER_PREP_FACE_MIN_FOCUS_SCORE:
        return False, {
            "reason": "face_blurry",
            "source": best.get("source"),
            "bbox": [x0, y0, x1, y1],
            "face_focus": float(focus),
        }

    return True, {
        "source": best.get("source"),
        "bbox": [x0, y0, x1, y1],
        "face_area": int(area),
        "face_area_ratio": float(area_ratio),
        "face_focus": float(focus),
    }

def _user_prep_crop_main_person(image: Image.Image, bbox: list[int]) -> tuple[Image.Image, list[int]]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad = int(round(max(bw, bh) * USER_PREP_MAIN_PERSON_PAD_RATIO))
    expanded = _expand_bbox([x0, y0, x1, y1], image.width, image.height, pad)
    ex0, ey0, ex1, ey1 = [int(v) for v in expanded]
    crop = image.crop((ex0, ey0, ex1, ey1)).convert("RGB")
    return crop, [ex0, ey0, ex1, ey1]

def _user_prep_alpha_stats(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 4:
        total = max(1, int(img.width * img.height))
        return {
            "has_alpha": False,
            "foreground_ratio": 1.0,
            "transparent_ratio": 0.0,
            "total_pixels": int(total),
        }
    alpha = arr[:, :, 3]
    total = max(1, int(alpha.size))
    fg_ratio = float(np.mean(alpha >= 12))
    transparent_ratio = float(np.mean(alpha <= 4))
    return {
        "has_alpha": True,
        "foreground_ratio": float(fg_ratio),
        "transparent_ratio": float(transparent_ratio),
        "total_pixels": int(total),
    }

def _user_prep_is_valid_alpha(alpha_stats: dict) -> bool:
    fg_ratio = float(alpha_stats.get("foreground_ratio", 0.0))
    transparent_ratio = float(alpha_stats.get("transparent_ratio", 0.0))
    return (
        fg_ratio >= USER_PREP_ALPHA_MIN_FOREGROUND_RATIO
        and transparent_ratio >= USER_PREP_ALPHA_MIN_TRANSPARENT_RATIO
    )

def _remove_user_background(image_bytes: bytes) -> tuple[Optional[bytes], dict]:
    mode = USER_PREP_BG_BACKEND
    ordered = ["birefnet", "rembg"] if mode == "auto" else [mode]
    if mode == "birefnet":
        ordered.append("rembg")
    elif mode == "rembg":
        ordered.append("birefnet")

    errors: List[dict] = []
    for backend in ordered:
        if backend == "birefnet":
            out_bytes, meta = _remove_background_with_birefnet(image_bytes)
        else:
            out_bytes, meta = _remove_background_with_rembg(image_bytes)
        if not out_bytes:
            errors.append({"backend": backend, "meta": meta})
            continue

        alpha_stats = _user_prep_alpha_stats(out_bytes)
        if _user_prep_is_valid_alpha(alpha_stats):
            return out_bytes, {
                "backend": backend,
                "removal": meta,
                "alpha_stats": alpha_stats,
            }
        errors.append(
            {
                "backend": backend,
                "meta": meta,
                "alpha_stats": alpha_stats,
                "reason": "invalid_alpha_coverage",
            }
        )
    return None, {"errors": errors}

def _describe_user_image_for_prepare(user_img: Image.Image, image_url: str) -> str:
    candidates: List[str] = []
    requested = _normalize_descriptor_backend(USER_PREP_DESCRIPTION_BACKEND)
    for item in [requested, "minicpm", "minicpm_service", "florence"]:
        if item and item not in candidates:
            candidates.append(item)

    service_url = ANALYZE_MINICPM_SERVICE_URL or MINICPM_SERVICE_URL
    for backend in candidates:
        try:
            desc = _describe_user_image_for_flux2(
                user_img=user_img,
                backend=backend,
                image_url=image_url,
                service_url=service_url,
            )
            desc = " ".join(str(desc or "").split()).strip()
            if len(desc.split()) >= USER_PREP_MIN_PROMPT_WORDS:
                return desc
        except Exception as desc_err:
            logger.warning(f"User prep description failed backend={backend}: {desc_err}")
    return ""

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
    total_pixels = float(max(1, image.width * image.height))

    candidates = []
    for garment_type in ("outer", "top", "bottom", "dress"):
        keep_ids = _parser_extraction_keep_ids(garment_type)
        if not keep_ids:
            continue
        mask = np.isin(parsing, keep_ids)
        area_ratio = float(mask.sum()) / total_pixels
        if area_ratio < ANALYZE_PARSER_MIN_AREA_RATIO:
            continue

        mask = binary_open(mask, 3)
        mask = binary_close(mask, 3)
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


def _parser_preroute_instances(
    image: Image.Image,
    requested_type: Optional[str],
    square_padding_ratio: float = 0.12,
) -> list[dict]:
    try:
        parser_candidates = _build_parser_unified_square_split_candidates(
            image=image,
            requested_type=requested_type,
            min_component_area_ratio=ANALYZE_PARSER_MIN_AREA_RATIO,
            square_padding_ratio=float(square_padding_ratio),
        )
    except Exception:
        parser_candidates = []
    if not parser_candidates:
        parser_candidates = _parser_split_candidates(image)
        return parser_candidates

    instances: list[dict] = []
    for idx, candidate in enumerate(parser_candidates):
        bbox = (
            candidate.get("section_bbox")
            or candidate.get("bbox")
            or [0, 0, image.width, image.height]
        )
        crop = candidate.get("_crop_image")
        if not isinstance(crop, Image.Image):
            x0, y0, x1, y1 = [int(v) for v in bbox]
            crop = image.crop((x0, y0, x1, y1)).convert("RGB")
        instances.append(
            {
                "id": idx,
                "label": str(candidate.get("type") or "top"),
                "confidence": float(
                    candidate.get("fusion_score")
                    or candidate.get("parser_component_area_ratio")
                    or 0.35
                ),
                "image": crop,
                "bbox": [int(v) for v in bbox],
                "mask": None,
                "source": "human_parser",
                "parser_area_ratio": float(candidate.get("parser_component_area_ratio", 0.0) or 0.0),
            }
        )
    return instances

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


def _requested_type_geometry_score(item: dict, requested_type: str, image_height: int) -> float:
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or image_height <= 0:
        return _item_rank_score(item)
    _, y0, _, y1 = [int(v) for v in bbox]
    box_h = max(1, y1 - y0)
    center_y = y0 + (box_h / 2.0)
    center_ratio = float(center_y) / float(image_height)
    height_ratio = float(box_h) / float(image_height)
    base = _item_rank_score(item)
    req = _normalize_garment_type(requested_type)

    if req in {"top", "outer"}:
        # Prefer upper-body candidates with smaller center_y and meaningful height.
        return base + (1.0 - center_ratio) + (0.25 * min(height_ratio, 0.6))
    if req == "bottom":
        # Prefer lower-body candidates with larger center_y.
        return base + center_ratio + (0.15 * min(height_ratio, 0.75))
    if req == "dress":
        # Prefer tall full-body candidates spanning most of the frame.
        return base + (1.5 * height_ratio) - abs(center_ratio - 0.52)
    return base

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


def _caption_fullbody_dress_signal(text: str) -> bool:
    t = " ".join(str(text or "").strip().lower().split())
    if not t:
        return False
    dress_terms = (
        "dress", "gown", "one-piece", "one piece", "maxi", "midi", "mini",
        "anarkali", "saree", "sari", "lehenga", "jumpsuit", "romper", "kurti",
        "traditional drape", "draped garment", "full-length drape",
    )
    return any(term in t for term in dress_terms)


def _maybe_force_uncertain_fullbody_to_dress(
    items: list[dict],
    *,
    image_width: int,
    image_height: int,
    requested_type: Optional[str] = None,
) -> tuple[list[dict], Optional[dict]]:
    if (
        requested_type
        or not ANALYZE_UNCERTAIN_FULLBODY_TO_DRESS
        or len(items) < 2
        or image_width <= 0
        or image_height <= 0
    ):
        return items, None

    normalized_types = [_normalize_garment_type(str(item.get("type") or "")) for item in items]
    unique_types = {t for t in normalized_types if t}

    candidates: list[tuple[int, dict]] = []
    for idx, item in enumerate(items):
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        area_ratio = float(width * height) / max(1.0, float(image_width * image_height))
        height_ratio = float(height) / float(image_height)
        top_ratio = float(y0) / float(image_height)
        bottom_ratio = float(y1) / float(image_height)
        prompt_text = " ".join(
            str(item.get(key) or "")
            for key in ("promptDescription", "description", "style", "category_key")
        ).strip()
        candidates.append(
            (
                idx,
                {
                    "item": item,
                    "bbox": [x0, y0, x1, y1],
                    "area_ratio": area_ratio,
                    "height_ratio": height_ratio,
                    "top_ratio": top_ratio,
                    "bottom_ratio": bottom_ratio,
                    "type": _normalize_garment_type(str(item.get("type") or "")),
                    "rank_score": _item_rank_score(item),
                    "dress_signal": _caption_fullbody_dress_signal(prompt_text),
                },
            )
        )

    if len(candidates) < 2:
        return items, None

    candidates.sort(key=lambda pair: (pair[1]["area_ratio"], pair[1]["rank_score"]), reverse=True)
    dominant_idx, dominant = candidates[0]
    runner_up_area = float(candidates[1][1]["area_ratio"])
    area_advantage = float(dominant["area_ratio"]) / max(1e-6, runner_up_area)

    dominant_is_fullbody = (
        dominant["height_ratio"] >= ANALYZE_UNCERTAIN_FULLBODY_MIN_HEIGHT_RATIO
        and dominant["area_ratio"] >= ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_RATIO
        and dominant["top_ratio"] <= ANALYZE_UNCERTAIN_FULLBODY_MAX_TOP_RATIO
        and dominant["bottom_ratio"] >= ANALYZE_UNCERTAIN_FULLBODY_MIN_BOTTOM_RATIO
    )
    if not dominant_is_fullbody:
        return items, None

    types_allow_fallback = len(unique_types) <= 1 or dominant["dress_signal"]
    if not types_allow_fallback:
        return items, None

    required_area_advantage = ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_ADVANTAGE
    if dominant["dress_signal"]:
        required_area_advantage = min(required_area_advantage, 1.30)
    if area_advantage < required_area_advantage:
        return items, None

    chosen = dict(items[dominant_idx])
    original_type = chosen.get("type")
    chosen["type_original"] = original_type
    chosen["garment_type_original"] = chosen.get("garment_type")
    chosen["type"] = "dress"
    chosen["garment_type"] = "dress"
    chosen["type_source"] = "uncertain_fullbody_dress_fallback"
    chosen["dress_fallback"] = {
        "triggered": True,
        "dominant_index": int(dominant_idx),
        "area_ratio": round(float(dominant["area_ratio"]), 4),
        "height_ratio": round(float(dominant["height_ratio"]), 4),
        "top_ratio": round(float(dominant["top_ratio"]), 4),
        "bottom_ratio": round(float(dominant["bottom_ratio"]), 4),
        "area_advantage": round(area_advantage, 4),
        "required_area_advantage": round(required_area_advantage, 4),
        "unique_types_before": sorted(unique_types),
        "dress_signal": bool(dominant["dress_signal"]),
    }
    inferred_style = _infer_style_from_text(
        str(chosen.get("promptDescription") or chosen.get("description") or ""),
        garment_type="dress",
    )
    category_meta = _wardrobe_category_from_garment_type("dress", style=inferred_style if inferred_style else None)
    chosen["style"] = category_meta["style"]
    chosen["category_key"] = category_meta["category_key"]
    chosen["primary_category_key"] = category_meta["primary_category_key"]
    return [chosen], chosen["dress_fallback"]

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
    if any(k in t for k in ("shirt", "t-shirt", "tshirt", "tee", "blouse", "crop top", "tank top", "sports top", "camisole", "bra", "bralette", "brassiere", "bikini top", "bustier")):
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
        "camisole", "tank top", "crop top", "bodysuit", "bra", "bralette", "brassiere", "bikini top", "bustier",
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

def _prepare_extract_source_image(
    full_image: Image.Image,
    bbox: Optional[list[int]],
    garment_type: str,
    total_items: int,
    detector_mask: Optional[np.ndarray] = None,
    semantic_bbox: Optional[list[int]] = None,
):
    extractor = GarmentExtractor(
        config=GarmentExtractionConfig(
            force_bbox_crop=ANALYZE_EXTRACT_FORCE_BBOX_CROP,
            crop_pad_ratio=ANALYZE_EXTRACT_CROP_PAD_RATIO,
            crop_pad_ratio_dress=ANALYZE_EXTRACT_CROP_PAD_RATIO_DRESS,
            crop_bottom_extra_ratio_dress=ANALYZE_EXTRACT_CROP_BOTTOM_EXTRA_RATIO_DRESS,
            crop_top_extra_ratio_bottom=ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM,
            crop_top_extra_ratio_bottom_multi=ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI,
            dress_top_recovery_ratio=ANALYZE_EXTRACT_DRESS_TOP_RECOVERY_RATIO,
            top_top_recovery_ratio=ANALYZE_EXTRACT_TOP_TOP_RECOVERY_RATIO,
        )
    )
    return extractor.prepare(
        GarmentExtractionRequest(
            full_image=full_image,
            garment_type=garment_type,
            total_items=total_items,
            detected_bbox=bbox,
            detector_mask=detector_mask,
            semantic_bbox=semantic_bbox,
        )
    )

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
            "Do not recolor and do not shift hue/saturation/value. Preserve true fabric color under neutral lighting."
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


def _build_minicpm_garment_prompt(
    garment_type: str,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
) -> str:
    gtype = _normalize_garment_type(garment_type) or "garment"
    type_label = {
        "top": "top",
        "bottom": "bottom",
        "dress": "dress",
        "outer": "outerwear",
    }.get(gtype, "garment")
    hexes = [str(v).strip().upper() for v in (dominant_color_hexes or []) if str(v).strip()]
    hints = [str(v).strip().lower() for v in (color_hints or []) if str(v).strip()]
    color_clause = ""
    if hexes:
        color_clause += " Use the source garment colors exactly with palette lock: " + ", ".join(hexes) + "."
    if hints:
        color_clause += " Keep the color family locked to: " + ", ".join(hints[: max(2, FLUX2_COLOR_LOCK_TOP_K)]) + "."
    type_clause = (
        f"The required garment category is {type_label}. "
        f"Describe only that single {type_label} and ignore every other clothing item or body region."
    )
    return (
        f"{MINICPM_SERVICE_GARMENT_PROMPT} {type_clause} "
        "If body parts, face, hair, hands, legs, room, bed, mirror, phone, bag, or props are visible, ignore them completely. "
        "Do not mention a person wearing the garment. "
        "Do not describe pose, scene, background, or accessories. "
        "Return exactly these two labeled sections and no extra text: "
        "BASE_GARMENT_PROMPT: one concise garment-only prompt containing only the requested garment's characteristics. "
        "EXTRACTION_AVOID_CLAUSE: only extraction-specific exclusions or contamination to avoid. "
        "If multiple garments are visible, mention the non-target garments only inside EXTRACTION_AVOID_CLAUSE as exclusions, never inside BASE_GARMENT_PROMPT. "
        "Do not put avoid words, negatives, or exclusion phrases inside BASE_GARMENT_PROMPT. "
        "Report garment colors using plain color words only; never output hex codes. "
        "Do not use skin tone, gloves, jewelry, mannequin color, or background color as garment color. "
        "Do not invent metallic, gold, silver, or hardware colors unless they are clearly visible on the garment itself. "
        "If visible, explicitly preserve and report front placket shape, button or snap count, button spacing, button size, and button placement. "
        "Return category and type for the requested garment only."
        f"{color_clause}"
    ).strip()


def _build_minicpm_garment_color_prompt(garment_type: str) -> str:
    gtype = _normalize_garment_type(garment_type) or "garment"
    type_label = {
        "top": "top",
        "bottom": "bottom",
        "dress": "dress",
        "outer": "outerwear",
    }.get(gtype, "garment")
    return (
        f"Look only at the requested {type_label}. "
        "Ignore skin, body, face, hair, gloves, jewelry, mannequin, room, background, and lighting. "
        "Return only 1 to 3 short garment fabric color words in plain English, comma-separated. "
        "Focus on the main fabric color first, then any true trim or accent color. "
        "Never output hex codes. Never mention skin tone or background color."
    ).strip()


def _describe_garment_color_terms_with_backend(
    image: Image.Image,
    backend: str,
    image_url: Optional[str] = None,
    service_url: Optional[str] = None,
    garment_type: Optional[str] = None,
) -> List[str]:
    resolved = _normalize_descriptor_backend(backend)
    prompt = _build_minicpm_garment_color_prompt(str(garment_type or ""))
    if resolved == "minicpm_service":
        try:
            if not image_url:
                return []
            cache_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            image_signature = _image_signature_for_cache(cache_img)
            text = _describe_with_minicpm_service(
                image_url=image_url,
                kind="garment",
                image_signature=image_signature,
                service_url=service_url,
                prompt_override=prompt,
            )
            return _extract_text_color_terms(text, max_items=max(2, FLUX2_COLOR_LOCK_TOP_K))
        except Exception as err:
            logger.warning(f"MiniCPM semantic garment color description failed: {err}")
            return []
    if resolved == "minicpm":
        try:
            minicpm_img = _resize_for_qwen_caption(
                image=image,
                max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
                min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
            )
            text = str(engine.minicpm.describe_garment(minicpm_img)).strip()
            return _extract_text_color_terms(text, max_items=max(2, FLUX2_COLOR_LOCK_TOP_K))
        except Exception as err:
            logger.warning(f"MiniCPM semantic garment color fallback failed: {err}")
            return []
    return []

def _build_flux2_single_garment_extract_prompt(
    garment_type: str,
    prompt_description: str,
    extraction_avoid_clause: str = "",
    category_text: Optional[str] = None,
    dominant_color_hexes: Optional[List[str]] = None,
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
) -> str:
    gtype = _normalize_garment_type(garment_type) or "top"
    clean_desc = " ".join(str(prompt_description or "").split()).strip()
    low_desc = clean_desc.lower()

    inferred_subtype = ""
    subtype_map = (
        ("saree", "saree"),
        ("sari", "saree"),
        ("lehenga", "lehenga"),
        ("anarkali", "anarkali"),
        ("blazer", "blazer"),
        ("jacket", "jacket"),
        ("coat", "coat"),
        ("bralette", "bralette"),
        ("bra", "bra"),
        ("bustier", "bustier"),
        ("corset", "corset"),
        ("blouse", "blouse"),
        ("shirt", "shirt"),
        ("skirt", "skirt"),
        ("trouser", "trousers"),
        ("pants", "pants"),
    )
    for needle, subtype in subtype_map:
        if needle in low_desc:
            inferred_subtype = subtype
            break

    explicit_category = str(category_text or inferred_subtype or _parser_candidate_category_text(gtype)).strip().lower()
    if not clean_desc:
        clean_desc = (
            "category=unknown; type=unknown; colors=unknown; pattern=unknown; material=unknown; "
            "silhouette=unknown; construction=unknown; details=unknown; coverage=unknown; preserve=unknown"
        )
    color_lock_clause = ""
    hexes = [str(v).strip() for v in (dominant_color_hexes or []) if str(v).strip()]
    if hexes:
        color_lock_clause = (
            f" Keep exact source colors with strict palette lock: {', '.join(hexes)}. "
            "Do not recolor and do not shift hue/saturation/value."
        )
    color_hint_clause = ""
    hints = [str(v).strip().lower() for v in (color_hints or []) if str(v).strip()]
    if hints:
        color_hint_clause = (
            " Keep color-family fidelity to source tones: "
            + ", ".join(hints[: max(2, FLUX2_COLOR_LOCK_TOP_K)])
            + ". "
        )
    tone_guidance = _build_garment_color_tone_guidance(
        color_hints=hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
    )
    color_tone_clause = ""
    tone_phrase = str(tone_guidance.get("phrase") or "").strip()
    tone_negative_terms = [str(v).strip() for v in (tone_guidance.get("negative_terms") or []) if str(v).strip()]
    if tone_phrase:
        color_tone_clause += f" Match the source garment as {tone_phrase} exactly. "
    if tone_negative_terms:
        color_tone_clause += "Do not reinterpret this color as " + ", ".join(tone_negative_terms) + ". "
    profile_clause = ""
    profile_obj = color_profile if isinstance(color_profile, dict) else {}
    if profile_obj:
        median_l = profile_obj.get("medianL")
        mean_c = profile_obj.get("meanChroma")
        if isinstance(median_l, (int, float)) and isinstance(mean_c, (int, float)):
            profile_clause = (
                f" Preserve source tone profile (target L*≈{float(median_l):.1f}, C*≈{float(mean_c):.1f}); "
                "avoid washout, over-darkening, or metallic sheen drift. "
            )

    type_lock_clause = {
        "top": (
            "Generate only a top garment. Never generate bottoms, dress silhouettes, legs, or shoes. "
            "Keep neckline, sleeve geometry, shoulder width, and hem shape identical to reference."
        ),
        "bottom": (
            "Generate only a bottom garment. Never generate tops, jackets, dresses, torso, face, or arms. "
            "Keep waistline, rise, leg width, inseam length, and hem opening identical to reference."
        ),
        "dress": (
            "Generate only a single dress garment with full one-piece replacement behavior. "
            "Never generate separate top+bottom combinations or layered two-piece outfits. "
            "Keep bodice-to-skirt continuity, waist seam placement, and hem length identical to reference."
        ),
    }.get(gtype, "Generate only one garment from the requested category.")

    subtype_lock_clause = ""
    if inferred_subtype == "saree":
        subtype_lock_clause = (
            " This garment is a saree. Generate a full saree drape only, never a blouse-only top, kurti, gown, or generic dress. "
            "Preserve pallu drape, border bands, pleated fall, tassels, textile continuity, and full-length vertical silhouette exactly as in the source."
        )
    elif inferred_subtype == "blazer":
        subtype_lock_clause = (
            " This garment is a blazer. Preserve exact lapel shape, front opening depth, button count, button size, button spacing, button placement, pocket flap placement, shoulder line, and sleeve length."
        )
    elif inferred_subtype in {"bra", "bralette", "bustier", "corset"}:
        subtype_lock_clause = (
            f" This garment is a {inferred_subtype}. Preserve exact cup shape, neckline, strap geometry, closure placement, and bust contour from the source."
        )
    clean_avoid_clause = " ".join(str(extraction_avoid_clause or "").split()).strip()
    avoid_clause = ""
    if clean_avoid_clause:
        avoid_clause = f" Extraction-only avoid requirements: {clean_avoid_clause}"

    return (
        "Extract and regenerate a standalone ecommerce product image from the provided garment crop. "
        "This is garment extraction only, not person generation. "
        "No person, no mannequin, no body parts, no skin regions, no neck, no shoulders, no torso, no arms, no hands, no legs, no feet, "
        "no hanger, no props, no text, no watermark. "
        "Use pure white background (RGB 255,255,255). Keep one centered garment with full silhouette visible. "
        "Output must be vertically composed in 2:3 ratio and center aligned. "
        f"Generate only one {explicit_category} category garment and nothing from other categories. "
        "Do not generate any artificial fashion variant, redesign, or alternate styling. "
        "Reconstruct only the exact source garment visible in the crop. "
        f"{type_lock_clause}{subtype_lock_clause}{color_lock_clause}{color_hint_clause}{color_tone_clause}{profile_clause}"
        "Preserve exact garment structure, fabric, texture, print placement, seams, pleats, closures, trims, closures, button count, button size, button spacing, button placement, front placket geometry, border placement, lapel geometry, and pocket placement. "
        "Keep every visible button, snap, stud, or dot-button exactly where it appears in the source, with the same count and vertical spacing. "
        "Preserve the exact source color and material appearance; do not brighten black garments into gray, silver, or white. "
        "Do not include visible limbs, face, torso, neck, shoulders, hands, fingers, legs, feet, or any human remnants in the output garment region. "
        f"Garment descriptor: {clean_desc}{avoid_clause}"
    )

def _build_flux2_single_garment_extract_negative_prompt(
    *,
    garment_type: str,
    custom_negative_prompt: str = "",
    color_hints: Optional[List[str]] = None,
    color_profile: Optional[Dict[str, object]] = None,
) -> str:
    gtype = _normalize_garment_type(garment_type) or "top"
    custom = " ".join(str(custom_negative_prompt or "").split()).strip()
    parts: List[str] = []
    if custom:
        parts.append(custom)

    parts.append(
        "person, human body, face, eyes, hair, skin, neck, shoulders, chest, torso, hands, fingers, arms, legs, feet, toes, mannequin, hanger, "
        "background scene, props, accessories, bag, jewelry, watermark, text, logo, human silhouette, skin patch, limb fragment, body fragment, arm fragment, leg fragment"
    )
    parts.append(
        "wrong garment color, recolored fabric, hue shift, saturation drift, value drift, pattern drift, print swap, "
        "texture swap, material swap, wrong garment category, extra garment, duplicate garment, merged garments, "
        "broken silhouette, distorted seams, wrong neckline, wrong sleeve length, wrong waistline, wrong hem length, artificial variant, redesigned garment, alternate style, "
        "blur, low detail, overexposed, underexposed"
    )

    if gtype == "top":
        parts.append(
            "pants, trousers, skirt, shorts, dress, lower-body garment, shoes, lower-body structure, bottom garment overlay"
        )
    elif gtype == "bottom":
        parts.append(
            "shirt, blouse, t-shirt, jacket, hoodie, dress, upper-body garment, torso garment overlay, sleeves"
        )
    elif gtype == "dress":
        parts.append(
            "two-piece outfit, separate top and bottom, skirt with separate blouse, trouser plus shirt combo, "
            "incomplete dress replacement, split bodice, split hemline"
        )

    tone_guidance = _build_garment_color_tone_guidance(
        color_hints=color_hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
    )
    tone_negative_terms = [str(v).strip() for v in (tone_guidance.get("negative_terms") or []) if str(v).strip()]
    if tone_negative_terms:
        parts.append(", ".join(tone_negative_terms))

    return " | ".join([p for p in parts if p]).strip()

def _select_single_garment_candidate(
    image: Image.Image,
    garment_type: str,
) -> tuple[Image.Image, dict]:
    normalized_type = _normalize_garment_type(garment_type) or "top"
    attempts: List[dict] = []

    builders = [
        ("unified_square_split", _build_parser_unified_square_split_candidates),
        ("component_first", _build_parser_square_candidates),
    ]
    for strategy_name, builder in builders:
        try:
            candidates = builder(
                image=image,
                requested_type=normalized_type,
                min_component_area_ratio=max(0.001, float(ANALYZE_PARSER_MIN_AREA_RATIO)),
                square_padding_ratio=0.12,
            )
            filtered = [
                c for c in candidates
                if _normalize_garment_type(str(c.get("type") or "")) == normalized_type
            ]
            attempts.append(
                {
                    "strategy": strategy_name,
                    "candidate_count": len(filtered),
                }
            )
            if not filtered:
                continue

            selected = filtered[0]
            selected_img = selected.get("_object_cutout_image") or selected.get("_preview_image") or selected.get("_crop_image")
            if not isinstance(selected_img, Image.Image):
                sx0, sy0, sx1, sy1 = [int(v) for v in (selected.get("section_bbox") or selected.get("bbox") or [0, 0, image.width, image.height])]
                selected_img = image.crop((sx0, sy0, sx1, sy1)).convert("RGB")

            if selected_img.mode == "RGBA":
                selected_img = _flatten_rgba_on_white(selected_img)
            else:
                selected_img = selected_img.convert("RGB")

            meta = {
                "strategy": strategy_name,
                "selectedIndex": 0,
                "selectedType": normalized_type,
                "candidateCount": len(filtered),
                "selectedCandidate": _to_public_parser_candidate(selected),
                "attempts": attempts,
            }
            return selected_img, meta
        except Exception as err:
            attempts.append(
                {
                    "strategy": strategy_name,
                    "candidate_count": 0,
                    "error": str(err),
                }
            )

    raise RuntimeError(
        f"No garment component found for requested type={normalized_type}. attempts={attempts}"
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

def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    with _REMBG_SESSION_LOCK:
        if _REMBG_SESSION is not None:
            return _REMBG_SESSION
        try:
            from rembg import new_session
            _REMBG_SESSION = new_session("isnet-general-use")
        except Exception as rembg_err:
            logger.warning(f"rembg session init failed: {rembg_err}")
            _REMBG_SESSION = False
        return _REMBG_SESSION


def _get_birefnet_model():
    global _BIREFNET_MODEL
    if _BIREFNET_MODEL is not None:
        return _BIREFNET_MODEL
    with _BIREFNET_LOCK:
        if _BIREFNET_MODEL is not None:
            return _BIREFNET_MODEL
        try:
            from transformers import AutoModelForImageSegmentation

            model = AutoModelForImageSegmentation.from_pretrained(
                ANALYZE_BIREFNET_MODEL_ID,
                trust_remote_code=True,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            _BIREFNET_MODEL = {
                "model": model,
                "device": device,
            }
        except Exception as birefnet_err:
            logger.warning(f"BiRefNet init failed: {birefnet_err}")
            _BIREFNET_MODEL = False
        return _BIREFNET_MODEL


def _remove_background_with_birefnet(image_bytes: bytes) -> tuple[Optional[bytes], dict]:
    bundle = _get_birefnet_model()
    if not bundle:
        return None, {"mode": "birefnet_unavailable"}
    try:
        from torchvision import transforms

        model = bundle["model"]
        device = bundle["device"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size
        transform = transforms.Compose([
            transforms.Resize((ANALYZE_BIREFNET_INPUT_SIZE, ANALYZE_BIREFNET_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(tensor)
        if isinstance(preds, (list, tuple)):
            pred = preds[-1]
        elif isinstance(preds, dict):
            pred = preds.get("pred") or preds.get("out") or next(iter(preds.values()))
        else:
            pred = preds
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = torch.sigmoid(pred)
        if pred.ndim == 4:
            pred = pred[0, 0]
        elif pred.ndim == 3:
            pred = pred[0]
        mask = pred.detach().float().cpu().numpy()
        mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        alpha = Image.fromarray(mask_u8, mode="L").resize((orig_w, orig_h), Image.LANCZOS)

        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)
        out = io.BytesIO()
        rgba.save(out, format="PNG")
        alpha_arr = np.asarray(alpha)
        return out.getvalue(), {
            "mode": "birefnet",
            "model_id": ANALYZE_BIREFNET_MODEL_ID,
            "input_size": ANALYZE_BIREFNET_INPUT_SIZE,
            "original_size": {"width": int(orig_w), "height": int(orig_h)},
            "output_size": {"width": int(orig_w), "height": int(orig_h)},
            "opaque_pixels": int(np.sum(alpha_arr >= 250)),
            "transparent_pixels": int(np.sum(alpha_arr <= 4)),
        }
    except Exception as birefnet_err:
        logger.warning(f"BiRefNet background removal failed: {birefnet_err}")
        return None, {"mode": "birefnet_failed", "error": str(birefnet_err)}

def _remove_background_with_rembg(image_bytes: bytes) -> tuple[Optional[bytes], dict]:
    session = _get_rembg_session()
    if not session:
        return None, {"mode": "rembg_unavailable"}
    try:
        from rembg import remove

        output = remove(
            image_bytes,
            session=session,
            alpha_matting=False,
            only_mask=False,
            post_process_mask=False,
            force_return_bytes=True,
        )
        out_img = Image.open(io.BytesIO(output)).convert("RGBA")
        out = io.BytesIO()
        out_img.save(out, format="PNG")
        alpha = np.asarray(out_img)[:, :, 3]
        return out.getvalue(), {
            "mode": "rembg_isnet_general_use",
            "original_size": {"width": int(out_img.width), "height": int(out_img.height)},
            "output_size": {"width": int(out_img.width), "height": int(out_img.height)},
            "opaque_pixels": int(np.sum(alpha >= 250)),
            "transparent_pixels": int(np.sum(alpha <= 4)),
        }
    except Exception as rembg_err:
        logger.warning(f"rembg background removal failed: {rembg_err}")
        return None, {"mode": "rembg_failed", "error": str(rembg_err)}

def _white_background_to_transparent_bytes(image_bytes: bytes) -> tuple[bytes, dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    arr = np.asarray(image).copy()
    rgb = arr[:, :, :3].astype(np.uint8)

    # Only remove near-white background that is connected to the frame border.
    near_white = (
        (rgb[:, :, 0] >= ANALYZE_GARMENT_WHITE_THRESHOLD)
        & (rgb[:, :, 1] >= ANALYZE_GARMENT_WHITE_THRESHOLD)
        & (rgb[:, :, 2] >= ANALYZE_GARMENT_WHITE_THRESHOLD)
    )
    low_chroma = (
        (np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2).astype(np.int16))
        <= 12
    )
    background_seed = near_white & low_chroma

    h, w = background_seed.shape
    background = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def _enqueue(y: int, x: int) -> None:
        if 0 <= y < h and 0 <= x < w and background_seed[y, x] and not background[y, x]:
            background[y, x] = True
            q.append((y, x))

    for x in range(w):
        _enqueue(0, x)
        _enqueue(h - 1, x)
    for y in range(h):
        _enqueue(y, 0)
        _enqueue(y, w - 1)

    while q:
        y, x = q.popleft()
        _enqueue(y - 1, x)
        _enqueue(y + 1, x)
        _enqueue(y, x - 1)
        _enqueue(y, x + 1)

    alpha = np.full((h, w), 255, dtype=np.uint8)
    alpha[background] = 0
    arr[:, :, 3] = alpha
    out_img = Image.fromarray(arr, mode="RGBA")

    out = io.BytesIO()
    out_img.save(out, format="PNG")
    return out.getvalue(), {
        "mode": "white_to_transparent_border_connected",
        "original_size": {"width": int(image.width), "height": int(image.height)},
        "output_size": {"width": int(out_img.width), "height": int(out_img.height)},
        "white_threshold": int(ANALYZE_GARMENT_WHITE_THRESHOLD),
        "background_pixels": int(background.sum()),
    }

def _postprocess_extracted_garment_bytes(
    image_bytes: bytes,
    garment_type: str = "",
    force_transparent: bool = False,
    transparent_only: bool = False,
) -> tuple[bytes, dict]:
    if transparent_only:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        original_meta = {
            "enabled": False,
            "original_size": {"width": int(image.width), "height": int(image.height)},
            "content_bbox": [0, 0, int(image.width), int(image.height)],
            "content_size": {"width": int(image.width), "height": int(image.height)},
            "output_size": {"width": int(image.width), "height": int(image.height)},
            "target_aspect": None,
            "enhance_enabled": False,
            "transparent_only": True,
            "alpha_seed_mode": "disabled_return_raw_flux",
            "output_background": "transparent",
            "top_skin_rim_cleanup": False,
            "white_bg_alpha": {"mode": "disabled"},
            "background_removal_backend": ANALYZE_BG_REMOVAL_BACKEND,
        }
        bg_removed = None
        bg_meta: Dict[str, object] = {}
        if ANALYZE_BG_REMOVAL_BACKEND == "birefnet":
            bg_removed, bg_meta = _remove_background_with_birefnet(image_bytes)
        elif ANALYZE_BG_REMOVAL_BACKEND == "rembg":
            bg_removed, bg_meta = _remove_background_with_rembg(image_bytes)
        elif ANALYZE_BG_REMOVAL_BACKEND == "white":
            bg_removed, bg_meta = _white_background_to_transparent_bytes(image_bytes)

        if bg_removed:
            out_img = Image.open(io.BytesIO(bg_removed)).convert("RGBA")
            out = io.BytesIO()
            out_img.save(out, format="PNG")
            original_meta["output_size"] = {"width": int(out_img.width), "height": int(out_img.height)}
            original_meta["background_removal"] = bg_meta
            return out.getvalue(), original_meta

        if bg_meta:
            original_meta["background_removal"] = bg_meta
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue(), original_meta

    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha_seed_mode = "existing_alpha"
    try:
        alpha_mask = _extract_alpha_mask(image, threshold=24)
        if not isinstance(alpha_mask, np.ndarray):
            alpha_mask = _estimate_foreground_mask_from_border(image)
            if isinstance(alpha_mask, np.ndarray):
                alpha_seed_mode = "border_estimate"
            else:
                alpha_seed_mode = "none"
        if isinstance(alpha_mask, np.ndarray) and alpha_mask.shape[:2] == (image.height, image.width):
            alpha_u8 = build_soft_alpha(alpha_mask, feather_px=max(1, ANALYZE_EXTRACT_EDGE_FEATHER_PX)).astype(np.uint8)
            rgba_arr = np.asarray(image).copy()
            rgba_arr[:, :, 3] = alpha_u8
            image = Image.fromarray(rgba_arr, mode="RGBA")
    except Exception as alpha_seed_err:
        alpha_seed_mode = f"failed:{alpha_seed_err}"
        logger.warning(f"Initial transparent alpha synthesis skipped: {alpha_seed_err}")
    
    # 0. Safe Hole Filling (Fixes 'missing cloth patches' on patterned garments)
    if not transparent_only and alpha_seed_mode != "border_estimate":
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
    if not transparent_only:
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
        if not transparent_only:
            processed = _enhance_garment_image(processed)

    # Top-only rim cleanup: trims skin-like fringe on neckline/lower edge without touching sleeves.
    if (not transparent_only) and ANALYZE_TOP_SKIN_RIM_CLEANUP and _normalize_garment_type(garment_type) in {"top", "outer"}:
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
    if ANALYZE_GARMENT_OUTPUT_BACKGROUND == "white" and not force_transparent:
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
        "transparent_only": bool(transparent_only),
        "alpha_seed_mode": alpha_seed_mode,
        "output_background": "transparent" if force_transparent else ANALYZE_GARMENT_OUTPUT_BACKGROUND,
        "top_skin_rim_cleanup": bool(
            ANALYZE_TOP_SKIN_RIM_CLEANUP and _normalize_garment_type(garment_type) in {"top", "outer"}
        ),
    }

def _crop_and_fit_garment_bytes(image_bytes: bytes) -> tuple[bytes, dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha_seed_mode = "existing_alpha"
    try:
        alpha_mask = _extract_alpha_mask(image, threshold=24)
        if not isinstance(alpha_mask, np.ndarray):
            alpha_mask = _estimate_foreground_mask_from_border(image)
            if isinstance(alpha_mask, np.ndarray):
                alpha_seed_mode = "border_estimate"
            else:
                alpha_seed_mode = "none"
        if isinstance(alpha_mask, np.ndarray) and alpha_mask.shape[:2] == (image.height, image.width):
            alpha_u8 = build_soft_alpha(alpha_mask, feather_px=max(1, ANALYZE_EXTRACT_EDGE_FEATHER_PX)).astype(np.uint8)
            rgba_arr = np.asarray(image).copy()
            rgba_arr[:, :, 3] = alpha_u8
            image = Image.fromarray(rgba_arr, mode="RGBA")
    except Exception as alpha_seed_err:
        alpha_seed_mode = f"failed:{alpha_seed_err}"
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
        "alpha_seed_mode": alpha_seed_mode,
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


def _parser_strict_mask(parsing: np.ndarray, garment_type: str) -> np.ndarray:
    ids = _parser_extraction_keep_ids(garment_type)
    if not ids:
        return np.zeros_like(parsing, dtype=bool)
    mask = np.isin(parsing, ids)
    mask = binary_open(mask, 3)
    mask = binary_close(mask, 3)
    return np.asarray(mask).astype(bool)


def _split_outfit_signature_from_parsing(parsing: np.ndarray) -> dict[str, object]:
    if not isinstance(parsing, np.ndarray) or parsing.ndim != 2 or parsing.size == 0:
        return {"detected": False, "reason": "bad_parsing"}

    image_height, image_width = parsing.shape[:2]
    total_pixels = float(max(1, image_height * image_width))
    top_mask = _parser_strict_mask(parsing, "top")
    bottom_mask = _parser_strict_mask(parsing, "bottom")
    dress_mask = _parser_strict_mask(parsing, "dress")

    top_area_ratio = float(np.sum(top_mask)) / total_pixels
    bottom_area_ratio = float(np.sum(bottom_mask)) / total_pixels
    dress_area_ratio = float(np.sum(dress_mask)) / total_pixels
    if top_area_ratio < 0.01 or bottom_area_ratio < 0.01:
        return {
            "detected": False,
            "reason": "missing_top_or_bottom",
            "top_area_ratio": round(top_area_ratio, 6),
            "bottom_area_ratio": round(bottom_area_ratio, 6),
            "dress_area_ratio": round(dress_area_ratio, 6),
        }

    top_bbox = _bbox_from_mask(top_mask)
    bottom_bbox = _bbox_from_mask(bottom_mask)
    dress_bbox = _bbox_from_mask(dress_mask)
    if top_bbox is None or bottom_bbox is None:
        return {
            "detected": False,
            "reason": "bbox_missing",
            "top_area_ratio": round(top_area_ratio, 6),
            "bottom_area_ratio": round(bottom_area_ratio, 6),
            "dress_area_ratio": round(dress_area_ratio, 6),
        }

    tx0, ty0, tx1, ty1 = [int(v) for v in top_bbox]
    bx0, by0, bx1, by1 = [int(v) for v in bottom_bbox]
    if ty0 > int(image_height * 0.55) or by1 < int(image_height * 0.45):
        return {
            "detected": False,
            "reason": "regions_not_upper_lower",
            "top_bbox": top_bbox,
            "bottom_bbox": bottom_bbox,
            "dress_bbox": dress_bbox,
        }

    gap_px = max(0, by0 - ty1)
    gap_ratio = float(gap_px) / float(max(1, image_height))
    x_overlap = _bbox_x_overlap_ratio(top_bbox, bottom_bbox)
    center_x0 = min(tx0, bx0)
    center_x1 = max(tx1, bx1)
    center_width = max(1, center_x1 - center_x0)
    centered_cover = (
        tx0 <= int(center_x0 + center_width * 0.18)
        and tx1 >= int(center_x1 - center_width * 0.18)
        and bx0 <= int(center_x0 + center_width * 0.18)
        and bx1 >= int(center_x1 - center_width * 0.18)
    )
    if x_overlap < 0.32 or not centered_cover:
        return {
            "detected": False,
            "reason": "alignment_low",
            "top_bbox": top_bbox,
            "bottom_bbox": bottom_bbox,
            "dress_bbox": dress_bbox,
            "gap_px": int(gap_px),
            "x_overlap": round(float(x_overlap), 4),
        }

    gap_fill_ratio = 0.0
    dress_bridge_ratio = 0.0
    if gap_px > 0:
        x0 = max(0, min(tx0, bx0))
        x1 = min(image_width, max(tx1, bx1))
        gap_union = top_mask | bottom_mask | dress_mask
        gap_slice = gap_union[ty1:by0, x0:x1]
        dress_slice = dress_mask[ty1:by0, x0:x1]
        if gap_slice.size > 0:
            gap_fill_ratio = float(np.mean(gap_slice))
        if dress_slice.size > 0:
            dress_bridge_ratio = float(np.mean(dress_slice))

    detected = bool(
        gap_px >= max(8, int(round(image_height * 0.015)))
        and gap_ratio >= 0.015
        and gap_fill_ratio <= 0.08
        and dress_bridge_ratio <= 0.05
        and dress_area_ratio <= max(0.03, (top_area_ratio + bottom_area_ratio) * 0.55)
    )
    return {
        "detected": detected,
        "reason": "split_gap_detected" if detected else "bridge_present_or_gap_small",
        "top_bbox": top_bbox,
        "bottom_bbox": bottom_bbox,
        "dress_bbox": dress_bbox,
        "gap_px": int(gap_px),
        "gap_ratio": round(gap_ratio, 4),
        "gap_fill_ratio": round(gap_fill_ratio, 4),
        "dress_bridge_ratio": round(dress_bridge_ratio, 4),
        "top_area_ratio": round(top_area_ratio, 6),
        "bottom_area_ratio": round(bottom_area_ratio, 6),
        "dress_area_ratio": round(dress_area_ratio, 6),
        "x_overlap": round(float(x_overlap), 4),
    }


def _dress_item_split_outfit_signature(
    item: dict,
    full_image: Optional[Image.Image] = None,
    total_items: Optional[int] = None,
) -> dict[str, object]:
    if engine.parser is None:
        return {"detected": False, "reason": "parser_unavailable"}
    try:
        crop = None
        if isinstance(full_image, Image.Image):
            try:
                extract_plan = _prepare_extract_source_image(
                    full_image=full_image,
                    bbox=item.get("bbox"),
                    garment_type="dress",
                    total_items=int(total_items or 1),
                    detector_mask=item.get("_mask_obj"),
                )
                crop = extract_plan.image
            except Exception:
                crop = None
        if not isinstance(crop, Image.Image):
            crop = item.get("_image_obj")
        if not isinstance(crop, Image.Image):
            crop = item.get("_preview_image") or item.get("_crop_image")
        if not isinstance(crop, Image.Image):
            return {"detected": False, "reason": "crop_unavailable"}
        parsing = engine.parser.parse(crop.convert("RGB"))
        return _split_outfit_signature_from_parsing(parsing)
    except Exception as split_err:
        return {"detected": False, "reason": f"error:{split_err}"}


def _suppress_split_outfit_dress_items(
    items: list[dict],
    full_image: Optional[Image.Image] = None,
) -> list[dict]:
    if len(items) <= 1 or engine.parser is None:
        return items

    top_exists = any(_normalize_garment_type(str(item.get("type") or "")) == "top" for item in items)
    bottom_exists = any(_normalize_garment_type(str(item.get("type") or "")) == "bottom" for item in items)
    if not (top_exists or bottom_exists):
        return items

    filtered: list[dict] = []
    removed = 0
    for item in items:
        item_type = _normalize_garment_type(str(item.get("type") or ""))
        if item_type != "dress":
            filtered.append(item)
            continue
        signature = _dress_item_split_outfit_signature(
            item,
            full_image=full_image,
            total_items=len(items),
        )
        item["split_outfit_signature"] = signature
        if bool(signature.get("detected")) and (top_exists or bottom_exists):
            item.setdefault("reason_codes", [])
            item["reason_codes"].append("dress_split_outfit_suppressed")
            removed += 1
            continue
        filtered.append(item)

    if filtered and removed > 0:
        for idx, item in enumerate(filtered):
            item["garment_id"] = idx
        return filtered
    return items

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


def _estimate_type_focused_color_mask(
    image: Image.Image,
    garment_type: str,
    description: str = "",
    return_meta: bool = False,
) -> Optional[np.ndarray]:
    try:
        mask, meta = engine.garment_color_masker.estimate_mask(
            image=image,
            garment_type=_normalize_garment_type(str(garment_type or "")) or "top",
            description=description,
        )
        if isinstance(meta, dict) and meta.get("used") is False:
            logger.info("Garment color mask provider skipped: %s", meta)
        resolved_mask = np.asarray(mask).astype(bool) if isinstance(mask, np.ndarray) else None
        if return_meta:
            return resolved_mask, meta
        return resolved_mask
    except Exception as mask_err:
        logger.warning(f"Type-focused color mask fallback triggered: {mask_err}")
        if return_meta:
            return None, {"source": "color_mask_error", "used": False, "reason": str(mask_err)}
        return None


def _restore_outer_lower_body_from_reference(
    reference_image: Image.Image,
    output_image: Image.Image,
    target_types: List[str],
    product_descriptions: Optional[List[str]] = None,
) -> Tuple[Image.Image, Dict[str, object]]:
    """
    Restore the lower-body region from the reference image for short outerwear cases.
    This guards against pants/shoes changing when only outerwear should change.
    """
    try:
        types = {str(t or "").strip().lower() for t in (target_types or []) if str(t or "").strip()}
        if types != {"outer"}:
            return output_image, {"applied": False, "reason": "not_outer_only"}

        desc_low = " ".join(str(v or "") for v in (product_descriptions or [])).lower()
        if any(token in desc_low for token in ("trench", "overcoat", "long coat", "full-length coat", "duster", "parka")):
            return output_image, {"applied": False, "reason": "long_outerwear"}
        if not any(token in desc_low for token in ("blazer", "jacket", "cardigan", "hoodie", "bomber", "shrug", "bolero", "outerwear")):
            return output_image, {"applied": False, "reason": "unsupported_outer_shape"}

        ref = reference_image.convert("RGB").resize(output_image.size, Image.BICUBIC)
        out = output_image.convert("RGB")
        ref_arr = np.asarray(ref, dtype=np.float32)
        out_arr = np.asarray(out, dtype=np.float32)
        if ref_arr.shape != out_arr.shape or ref_arr.ndim != 3:
            return output_image, {"applied": False, "reason": "shape_mismatch"}

        h, w = ref_arr.shape[:2]
        lower_start = int(round(h * 0.58))
        lower_full = int(round(h * 0.66))
        if lower_full <= lower_start:
            return output_image, {"applied": False, "reason": "invalid_blend_band"}

        lower_delta = np.mean(np.abs(out_arr[lower_start:, :, :] - ref_arr[lower_start:, :, :]), axis=2)
        changed_ratio = float(np.mean(lower_delta > 12.0))
        if changed_ratio < 0.015:
            return output_image, {"applied": False, "reason": "lower_body_unchanged", "changedRatio": round(changed_ratio, 4)}

        weights = np.ones((h,), dtype=np.float32)
        weights[lower_full:] = 0.0
        blend_band = max(1, lower_full - lower_start)
        for yy in range(lower_start, lower_full):
            frac = float(yy - lower_start) / float(blend_band)
            weights[yy] = max(0.0, min(1.0, 1.0 - frac))
        alpha = np.repeat(weights[:, None], w, axis=1)[:, :, None]
        blended = ((out_arr * alpha) + (ref_arr * (1.0 - alpha))).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended), {
            "applied": True,
            "reason": "outer_lower_body_restore",
            "lowerStartY": int(lower_start),
            "lowerFullY": int(lower_full),
            "changedRatio": round(changed_ratio, 4),
        }
    except Exception as restore_err:
        return output_image, {"applied": False, "reason": f"error:{restore_err}"}

def _extract_cloth_from_crop(
    crop: Image.Image,
    garment_type: str,
    enforce_safety_guards: bool = True,
    allow_top_dress_backfill: bool = True,
    parser_only_override: Optional[bool] = None,
) -> tuple[Image.Image, dict]:
    """
    High-quality garment extraction from a person/mannequin image.
    Primary path uses type-aware human parser masks for garment isolation.
    """
    selected_type = _normalize_garment_type(garment_type) or "top"
    total_pixels = float(max(1, crop.width * crop.height))
    img_np = np.asarray(crop.convert("RGB"))
    h, w = img_np.shape[:2]

    effective_parser_only = (
        ANALYZE_EXTRACT_PARSER_ONLY
        if parser_only_override is None
        else bool(parser_only_override)
    )

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
        if allow_top_dress_backfill and ANALYZE_PARSER_TOP_DRESS_BACKFILL and selected_type in {"top", "outer"}:
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

        if effective_parser_only:
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
        if effective_parser_only:
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

def _run_flux2_cloth_only_extract(
    source_image: Image.Image,
    garment_type: str,
    prompt_description: str = "",
    fallback_prompt_description: str = "",
    negative_prompt: str = "",
    description_backend: Optional[str] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    color_reference_image: Optional[Image.Image] = None,
    apply_type_color_mask: bool = False,
) -> dict:
    """
    Internal analyze-path garment extraction using Flux2 single-garment prompt logic.
    Analyze-path garment extraction:
    - optional type-guided color masking on the reference image
    - raw Flux output postprocessed only with background removal / crop-fit
    """
    t_all_start = time.time()
    resolved_type = _normalize_garment_type(garment_type)
    mapped_from_outer = False
    if resolved_type == "outer":
        resolved_type = "top"
        mapped_from_outer = True
    if resolved_type not in {"top", "bottom", "dress"}:
        raise RuntimeError(f"Unsupported garment_type for flux2 extract: {garment_type}")

    if source_image is None:
        raise RuntimeError("source image is required for flux2 extract")

    requested_backend = _normalize_descriptor_backend(
        str(description_backend or FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND)
    )
    allowed_extract_backends = {"florence", "joycaption", "minicpm", "minicpm_service"}
    resolved_backend = requested_backend if requested_backend in allowed_extract_backends else "minicpm"
    analyze_service_url = str(ANALYZE_MINICPM_SERVICE_URL or MINICPM_SERVICE_URL or "").strip().rstrip("/")
    run_steps = int(steps if isinstance(steps, int) and steps >= 4 else FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS)
    run_seed = int(seed if isinstance(seed, int) and seed >= 0 else FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED)
    helper_warnings = []

    src = source_image.convert("RGB")
    descriptor_image = src
    selected_crop_url = None
    selected_crop_url_source = ""
    selected_crop_local_path = ""
    descriptor_color_ctx_image = (
        color_reference_image.convert("RGB")
        if isinstance(color_reference_image, Image.Image)
        else descriptor_image
    )
    color_mask_description = " ".join(
        str(v).strip()
        for v in (prompt_description, fallback_prompt_description)
        if str(v).strip()
    ).strip()
    descriptor_color_mask = None
    descriptor_color_mask_meta = {"source": "disabled", "used": False, "reason": "type_mask_not_requested"}
    if apply_type_color_mask:
        descriptor_color_mask, descriptor_color_mask_meta = _estimate_type_focused_color_mask(
            descriptor_color_ctx_image,
            resolved_type,
            color_mask_description,
            return_meta=True,
        )
    descriptor_color_ctx = _build_single_image_color_context(
        image=descriptor_color_ctx_image,
        description="",
        mask=descriptor_color_mask,
        top_k=7,
        force_masking=bool(apply_type_color_mask and isinstance(descriptor_color_mask, np.ndarray)),
    )
    descriptor_dominant_hexes = [
        str(v)
        for v in (descriptor_color_ctx.get("dominantHexes") or descriptor_color_ctx.get("paletteHexes") or [])
        if str(v).strip()
    ]
    descriptor_color_hints = [
        str(v) for v in (descriptor_color_ctx.get("colorHints") or []) if str(v).strip()
    ]
    stage = {
        "descriptor_input_s": 0.0,
        "descriptor_s": 0.0,
        "prompt_build_s": 0.0,
        "flux_generation_s": 0.0,
        "flux_extract_s": 0.0,
        "upload_s": 0.0,
        "total_s": 0.0,
    }
    prompt_bundle: Dict[str, str] = {
        "raw_text": "",
        "base_garment_prompt": "",
        "extraction_avoid_clause": "",
        "serialized_sections": "",
    }

    t_stage = time.time()
    selected_crop_buf = io.BytesIO()
    descriptor_upload_image = descriptor_image
    if resolved_backend == "minicpm_service":
        descriptor_upload_image = _resize_for_qwen_caption(
            image=descriptor_image,
            max_side=FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE,
            min_side=FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE,
        )
    descriptor_upload_image.save(selected_crop_buf, format="PNG")
    selected_crop_bytes = selected_crop_buf.getvalue()
    if resolved_backend == "minicpm_service":
        if MINICPM_SERVICE_LOCAL_FILE_FIRST:
            try:
                fd, tmp_path = tempfile.mkstemp(prefix="minicpm_desc_", suffix=".png")
                os.close(fd)
                with open(tmp_path, "wb") as fp:
                    fp.write(selected_crop_bytes)
                selected_crop_local_path = tmp_path
                selected_crop_url = f"file://{tmp_path}"
                selected_crop_url_source = "local_file_url"
            except Exception as file_err:
                helper_warnings.append(f"selected_crop_file_prep_failed:{file_err}")
        if not selected_crop_url:
            try:
                selected_crop_url = _upload_or_raise(selected_crop_bytes, container=VTO_OUTPUT_CONTAINER)
                if selected_crop_url:
                    selected_crop_url_source = "azure_upload_url"
            except Exception as upload_err:
                helper_warnings.append(f"selected_crop_upload_failed:{upload_err}")
        if not str(selected_crop_url or "").startswith("http"):
            if not str(selected_crop_url or "").startswith("file://"):
                resolved_backend = "minicpm"
    stage["descriptor_input_s"] = round(time.time() - t_stage, 4)

    t_stage = time.time()
    clean_prompt = " ".join(str(prompt_description or "").split()).strip()
    fallback_prompt = " ".join(str(fallback_prompt_description or "").split()).strip()
    if clean_prompt:
        garment_desc = clean_prompt
        prompt_bundle = _parse_garment_prompt_sections(clean_prompt, garment_type=resolved_type)
        prompt_source = "request"
    else:
        garment_desc = ""
        prompt_source = resolved_backend
        desc_err: Optional[Exception] = None
        try:
            prompt_bundle = _describe_garment_prompt_bundle_with_backend(
                image=descriptor_image,
                backend=resolved_backend,
                image_url=selected_crop_url,
                service_url=analyze_service_url if resolved_backend == "minicpm_service" else None,
                garment_type=resolved_type,
                dominant_color_hexes=descriptor_dominant_hexes,
                color_hints=descriptor_color_hints,
            )
            garment_desc = str(prompt_bundle.get("base_garment_prompt") or "").strip()
        except Exception as err:
            desc_err = err

        if desc_err and resolved_backend == "minicpm_service" and selected_crop_url_source == "local_file_url":
            # Fallback transport path when service process cannot read file:// URLs.
            try:
                selected_crop_url = _upload_or_raise(selected_crop_bytes, container=VTO_OUTPUT_CONTAINER)
                if selected_crop_url:
                    selected_crop_url_source = "azure_upload_url_retry"
                    prompt_bundle = _describe_garment_prompt_bundle_with_backend(
                        image=descriptor_image,
                        backend=resolved_backend,
                        image_url=selected_crop_url,
                        service_url=analyze_service_url if resolved_backend == "minicpm_service" else None,
                        garment_type=resolved_type,
                        dominant_color_hexes=descriptor_dominant_hexes,
                        color_hints=descriptor_color_hints,
                    )
                    garment_desc = str(prompt_bundle.get("base_garment_prompt") or "").strip()
                    desc_err = None
                else:
                    helper_warnings.append("selected_crop_retry_upload_empty_url")
            except Exception as retry_err:
                helper_warnings.append(f"selected_crop_retry_upload_failed:{retry_err}")

        garment_desc = " ".join(str(garment_desc or "").split()).strip()
        if desc_err and not garment_desc and fallback_prompt:
            garment_desc = fallback_prompt
            prompt_bundle = _parse_garment_prompt_sections(fallback_prompt, garment_type=resolved_type)
            prompt_source = "selection_fallback_on_descriptor_error"
            helper_warnings.append(f"descriptor_failed_fallback_used:{desc_err}")
        elif desc_err and not garment_desc:
            raise desc_err

        if fallback_prompt and _descriptor_is_weak(garment_desc):
            enriched = _enrich_garment_descriptor(
                primary=garment_desc,
                fallback=fallback_prompt,
                garment_type=resolved_type,
            )
            enriched = " ".join(str(enriched or "").split()).strip()
            if enriched and enriched != garment_desc:
                garment_desc = enriched
                prompt_bundle["base_garment_prompt"] = garment_desc
                prompt_bundle["serialized_sections"] = (
                    f"BASE_GARMENT_PROMPT: {garment_desc}"
                    + (
                        f"\nEXTRACTION_AVOID_CLAUSE: {prompt_bundle.get('extraction_avoid_clause')}"
                        if str(prompt_bundle.get("extraction_avoid_clause") or "").strip()
                        else ""
                    )
                )
                prompt_source = f"{prompt_source}+selection_fallback_enrich"
                helper_warnings.append("descriptor_enriched_from_selection_fallback")
    stage["descriptor_s"] = round(time.time() - t_stage, 4)
    if selected_crop_local_path:
        try:
            os.remove(selected_crop_local_path)
        except Exception:
            pass
    if not garment_desc:
        raise RuntimeError("Could not generate garment description for flux2 extract")

    color_ctx_image = descriptor_color_ctx_image
    input_color_ctx = _build_single_image_color_context(
        image=color_ctx_image,
        description=garment_desc,
        mask=descriptor_color_mask,
        top_k=7,
        force_masking=bool(apply_type_color_mask and isinstance(descriptor_color_mask, np.ndarray)),
    )
    dominant_hexes = [
        str(v)
        for v in (input_color_ctx.get("dominantHexes") or input_color_ctx.get("paletteHexes") or [])
        if str(v).strip()
    ]
    if not dominant_hexes:
        dominant_hexes = _extract_dominant_hex_colors(
            image=color_ctx_image,
            top_k=4,
        )
    color_hints = [str(v) for v in (input_color_ctx.get("hints") or []) if str(v).strip()]
    color_profile = input_color_ctx.get("profile") if isinstance(input_color_ctx.get("profile"), dict) else {}
    color_mask_source = str(
        (descriptor_color_mask_meta or {}).get("source")
        or input_color_ctx.get("maskSource")
        or descriptor_color_ctx.get("maskSource")
        or "none"
    ).strip()
    reconciled_color = _resolve_garment_color_truth(
        base_garment_prompt=garment_desc,
        dominant_hexes=dominant_hexes,
        color_hints=color_hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
    )
    resolved_garment_desc = " ".join(
        str(reconciled_color.get("base_garment_prompt") or garment_desc).split()
    ).strip()
    reconciled_hexes = [
        str(v).strip().upper()
        for v in (reconciled_color.get("dominant_hexes") or [])
        if str(v).strip()
    ]
    reconciled_hints = [
        str(v).strip().lower()
        for v in (reconciled_color.get("color_hints") or [])
        if str(v).strip()
    ]
    if reconciled_hexes:
        dominant_hexes = reconciled_hexes
    if reconciled_hints:
        color_hints = reconciled_hints
    garment_desc = resolved_garment_desc or garment_desc
    prompt_bundle["base_garment_prompt"] = garment_desc
    prompt_bundle["serialized_sections"] = (
        f"BASE_GARMENT_PROMPT: {garment_desc}"
        + (
            f"\nEXTRACTION_AVOID_CLAUSE: {prompt_bundle.get('extraction_avoid_clause')}"
            if str(prompt_bundle.get("extraction_avoid_clause") or "").strip()
            else ""
        )
    )

    t_stage = time.time()
    flux_prompt = _build_flux2_single_garment_extract_prompt(
        garment_type=resolved_type,
        prompt_description=garment_desc,
        extraction_avoid_clause=str(prompt_bundle.get("extraction_avoid_clause") or ""),
        category_text=resolved_type,
        dominant_color_hexes=dominant_hexes,
        color_hints=color_hints,
        color_profile=color_profile,
    )
    built_negative_prompt = _build_flux2_single_garment_extract_negative_prompt(
        garment_type=resolved_type,
        custom_negative_prompt=" ".join(str(negative_prompt or "").split()).strip(),
        color_hints=color_hints,
        color_profile=color_profile,
    )
    stage["prompt_build_s"] = round(time.time() - t_stage, 4)

    t_stage = time.time()
    analyze_flux_runner = engine.get_flux2_for_analyze()
    flux_result = analyze_flux_runner.run_tryon(
        person_image=src,
        board_image=src,
        prompt=flux_prompt,
        steps=run_steps,
        seed=run_seed,
        negative_prompt=built_negative_prompt,
        use_lora=not ANALYZE_FLUX_DISABLE_LORA,
    )
    stage["flux_generation_s"] = round(time.time() - t_stage, 4)

    raw_img = flux_result["image"].convert("RGBA")
    raw_buf = io.BytesIO()
    raw_img.save(raw_buf, format="PNG")
    raw_bytes = raw_buf.getvalue()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    t_stage = time.time()
    final_bytes = raw_bytes
    extraction_meta: Dict[str, object] = {
        "path": "background_removal_only",
        "parserSkipped": True,
    }
    postprocess_meta: Dict[str, object] = {}
    extraction_path = "background_removal_only"
    try:
        final_bytes, postprocess_meta = _postprocess_extracted_garment_bytes(
            raw_bytes,
            garment_type=resolved_type,
            force_transparent=True,
            transparent_only=True,
        )
    except Exception as extract_err:
        helper_warnings.append(f"extract_postprocess_failed:{extract_err}")
        final_bytes, postprocess_meta = _crop_and_fit_garment_bytes(raw_bytes)
        extraction_meta = {"warning": str(extract_err)}
        extraction_path = "crop_then_aspect_only_fallback"

    extraction_meta = dict(extraction_meta or {})
    extraction_meta["raw_flux_sha256"] = raw_sha256
    extraction_meta["parser_input_source"] = "none"
    extraction_meta["parser_input_sha256"] = ""
    extraction_meta["use_parser_post_extract"] = False
    extraction_meta["parser_skipped"] = True
    extraction_meta["parser_used_flux_raw"] = False
    extraction_meta["parser_input_matches_flux_raw"] = False

    color_guard_meta: Dict[str, object] = {"attempted": False, "applied": False, "reason": "not_run"}
    if FLUX2_COLOR_GUARD_RERUN_ENABLED and dominant_hexes:
        try:
            final_image_for_score = Image.open(io.BytesIO(final_bytes)).convert("RGB")
            base_color_score = _score_color_fidelity(
                output_image=final_image_for_score,
                input_palettes=[dominant_hexes],
                target_types=[resolved_type],
                input_profiles=[color_profile] if isinstance(color_profile, dict) and color_profile else None,
            )
            color_guard_meta["attempted"] = True
            color_guard_meta["initial"] = base_color_score
            base_drift = base_color_score.get("total_drift")
            drift_is_valid = isinstance(base_drift, (int, float)) and np.isfinite(float(base_drift))
            if drift_is_valid and float(base_drift) > float(FLUX2_COLOR_GUARD_DRIFT_THRESHOLD):
                palette_metrics = [[{"hex": hx, "areaPercent": 0.0} for hx in dominant_hexes if str(hx).strip()]]
                color_guard_clause = _build_flux2_hex_color_guard_clause(palette_metrics)
                if color_guard_clause:
                    guard_prompt = f"{flux_prompt} {color_guard_clause}".strip()
                    guard_steps = min(
                        FLUX2_COLOR_GUARD_RERUN_MAX_STEPS,
                        max(run_steps, run_steps + FLUX2_COLOR_GUARD_RERUN_EXTRA_STEPS),
                    )
                    guard_seed = min(2147483647, run_seed + 31)
                    retry_result = analyze_flux_runner.run_tryon(
                        person_image=src,
                        board_image=src,
                        prompt=guard_prompt,
                        steps=guard_steps,
                        seed=guard_seed,
                        negative_prompt=built_negative_prompt,
                        use_lora=not ANALYZE_FLUX_DISABLE_LORA,
                    )
                    retry_raw_img = retry_result["image"].convert("RGBA")
                    retry_raw_buf = io.BytesIO()
                    retry_raw_img.save(retry_raw_buf, format="PNG")
                    retry_raw_bytes = retry_raw_buf.getvalue()
                    retry_final_bytes = retry_raw_bytes
                    retry_postprocess_meta: Dict[str, object] = {}
                    retry_extraction_path = "background_removal_only"
                    try:
                        retry_final_bytes, retry_postprocess_meta = _postprocess_extracted_garment_bytes(
                            retry_raw_bytes,
                            garment_type=resolved_type,
                            force_transparent=True,
                            transparent_only=True,
                        )
                    except Exception as retry_extract_err:
                        helper_warnings.append(f"extract_postprocess_retry_failed:{retry_extract_err}")
                        retry_final_bytes, retry_postprocess_meta = _crop_and_fit_garment_bytes(retry_raw_bytes)
                        retry_extraction_path = "crop_then_aspect_only_fallback"

                    retry_score = _score_color_fidelity(
                        output_image=Image.open(io.BytesIO(retry_final_bytes)).convert("RGB"),
                        input_palettes=[dominant_hexes],
                        target_types=[resolved_type],
                        input_profiles=[color_profile] if isinstance(color_profile, dict) and color_profile else None,
                    )
                    color_guard_meta["retry"] = retry_score
                    retry_drift = retry_score.get("total_drift")
                    retry_is_valid = isinstance(retry_drift, (int, float)) and np.isfinite(float(retry_drift))
                    if retry_is_valid and (not drift_is_valid or float(retry_drift) <= float(base_drift)):
                        final_bytes = retry_final_bytes
                        postprocess_meta = retry_postprocess_meta
                        extraction_path = retry_extraction_path
                        raw_bytes = retry_raw_bytes
                        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
                        extraction_meta["raw_flux_sha256"] = raw_sha256
                        flux_result = retry_result
                        flux_prompt = guard_prompt
                        run_steps = guard_steps
                        run_seed = guard_seed
                        color_guard_meta["applied"] = True
                        color_guard_meta["reason"] = "retry_improved_or_matched_drift"
                    else:
                        color_guard_meta["reason"] = "retry_not_better"
                else:
                    color_guard_meta["reason"] = "no_guard_clause"
            else:
                color_guard_meta["reason"] = "drift_below_threshold" if drift_is_valid else "drift_unavailable"
        except Exception as color_guard_err:
            helper_warnings.append(f"extract_color_guard_failed:{color_guard_err}")
            color_guard_meta = {"attempted": True, "applied": False, "reason": f"error:{color_guard_err}"}

    stage["flux_extract_s"] = round(time.time() - t_stage, 4)

    t_stage = time.time()
    output_url = _upload_or_raise(final_bytes, container=VTO_OUTPUT_CONTAINER)
    stage["upload_s"] = round(time.time() - t_stage, 4)
    stage["total_s"] = round(time.time() - t_all_start, 4)

    return {
        "url": output_url,
        "raw_url": output_url,
        "_processed_image_bytes": final_bytes,
        "meta": {
            "pipeline": "flux2_extract",
            "endpoint": "/v1/flux2/extract-garment",
            "path": extraction_path,
            "postprocess": postprocess_meta or {"enabled": False, "reason": "postprocess_not_applied"},
            "extraction_meta": extraction_meta or {},
            "descriptor_backend_requested": requested_backend,
            "descriptor_backend_resolved": resolved_backend,
            "descriptor_service_url": analyze_service_url if resolved_backend == "minicpm_service" else "",
            "descriptor_service_url_source": "analyze_override" if ANALYZE_MINICPM_SERVICE_URL else "default",
            "descriptor_transport_source": selected_crop_url_source,
            "prompt_source": prompt_source,
            "prompt_description": garment_desc,
            "base_garment_prompt": garment_desc,
            "extraction_avoid_clause": str(prompt_bundle.get("extraction_avoid_clause") or ""),
            "prompt_sections_raw": str(prompt_bundle.get("serialized_sections") or ""),
            "descriptor_raw_text": str(prompt_bundle.get("raw_text") or ""),
            "dominant_hexes": dominant_hexes,
            "accent_hexes": [str(v) for v in (input_color_ctx.get("accentHexes") or []) if str(v).strip()],
            "color_hints": color_hints,
            "color_profile": color_profile,
            "color_mask_source": color_mask_source or str(input_color_ctx.get("maskSource") or "none"),
            "color_resolved_source": str(reconciled_color.get("color_source") or "pixel"),
            "flux_prompt": flux_prompt,
            "negative_prompt": built_negative_prompt,
            "warnings": helper_warnings,
            "color_guard": color_guard_meta,
            "steps": run_steps,
            "seed": run_seed,
            "resolved_type": resolved_type,
            "mapped_from_outer": mapped_from_outer,
            "timings": stage,
            "flux": {
                "latency_s": round(float(flux_result.get("latency", 0.0)), 4),
                "metadata": flux_result.get("metadata", {}),
            },
            "runner": {
                "analyze_flux_isolated": bool(ANALYZE_FLUX_DISABLE_LORA and not engine._share_flux2_base_runner),
                "analyze_flux_shared_base_runner": bool(engine._share_flux2_base_runner),
                "analyze_flux_lora_disabled": bool(ANALYZE_FLUX_DISABLE_LORA),
                "analyze_flux_runtime_lora_toggle": bool(getattr(analyze_flux_runner, "runtime_lora_toggle", False)),
            },
        },
    }


def _sync_wardrobe_progress(
    *,
    authorization: Optional[str],
    progress_id: str,
    output_url: Optional[str],
    prompt_description: str,
    garment_metadata: Optional[dict],
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
        "garmentMetadata": garment_metadata if isinstance(garment_metadata, dict) else {},
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


def _sync_wardrobe_progress_dispatch(
    *,
    authorization: Optional[str],
    progress_id: str,
    output_url: Optional[str],
    prompt_description: str,
    garment_metadata: Optional[dict],
    metadata: dict,
) -> dict:
    if not ANALYZE_PROGRESS_SYNC_ASYNC:
        return _sync_wardrobe_progress(
            authorization=authorization,
            progress_id=progress_id,
            output_url=output_url,
            prompt_description=prompt_description,
            garment_metadata=garment_metadata,
            metadata=metadata,
        )
    try:
        _WARDROBE_PROGRESS_EXECUTOR.submit(
            _sync_wardrobe_progress,
            authorization=authorization,
            progress_id=progress_id,
            output_url=output_url,
            prompt_description=prompt_description,
            garment_metadata=garment_metadata,
            metadata=metadata,
        )
        return {
            "enabled": bool(ENABLE_WARDROBE_PROGRESS_SYNC),
            "synced": False,
            "queued": True,
            "mode": "async_background",
            "id": progress_id,
        }
    except Exception as queue_err:
        logger.warning(f"Async wardrobe progress queue failed. Falling back to sync call. error={queue_err}")
        return _sync_wardrobe_progress(
            authorization=authorization,
            progress_id=progress_id,
            output_url=output_url,
            prompt_description=prompt_description,
            garment_metadata=garment_metadata,
            metadata=metadata,
        )

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
    garmentMetadata: Optional[Dict[str, object]] = None

class Flux2TryonUserImage(BaseModel):
    tryonImage: str
    promptDescription: Optional[str] = None

class Flux2TryonRequest(BaseModel):
    products: List[Flux2TryonProduct]
    user_image: Flux2TryonUserImage
    description_backend: Optional[str] = None
    description_compare: bool = False
    negative_prompt: Optional[str] = None
    disable_neutral_calibration: bool = False
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
    async def _run_preload():
        t0 = time.time()
        _STARTUP_PRELOAD_STATE["started"] = True
        _STARTUP_PRELOAD_STATE["running"] = True
        _STARTUP_PRELOAD_STATE["completed"] = False
        _STARTUP_PRELOAD_STATE["started_at"] = t0
        _STARTUP_PRELOAD_STATE["completed_at"] = None
        _STARTUP_PRELOAD_STATE["duration_s"] = None
        _STARTUP_PRELOAD_STATE["errors"] = []
        try:
            if os.getenv("PRELOAD", "1") == "1":
                await asyncio.to_thread(engine.ensure_vto_ready)
            if os.getenv("PRELOAD_ANALYZE", "1") == "1":
                await asyncio.to_thread(engine.ensure_analyze_ready)
            _STARTUP_PRELOAD_STATE["completed"] = True
        except Exception as preload_err:
            logger.exception(f"Startup preloading failed: {preload_err}")
            _STARTUP_PRELOAD_STATE["errors"] = [str(preload_err)]
        finally:
            _STARTUP_PRELOAD_STATE["running"] = False
            _STARTUP_PRELOAD_STATE["completed_at"] = time.time()
            _STARTUP_PRELOAD_STATE["duration_s"] = round(
                _STARTUP_PRELOAD_STATE["completed_at"] - t0, 4
            )
            logger.info(
                "Startup preloading finished mode=%s completed=%s duration=%.2fs",
                _STARTUP_PRELOAD_STATE["mode"],
                _STARTUP_PRELOAD_STATE["completed"],
                _STARTUP_PRELOAD_STATE["duration_s"] or 0.0,
            )

    global _STARTUP_PRELOAD_TASK
    if STARTUP_BACKGROUND_PRELOAD:
        _STARTUP_PRELOAD_TASK = asyncio.create_task(_run_preload())
        logger.info("Startup preloading scheduled in background")
        return
    await _run_preload()

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
                    use_lora=False,
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

@app.post("/v1/flux2/extract-garment")
async def flux2_extract_single_garment(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form(None, alias="type"),
    garmentType: Optional[str] = Form(None),
    prompt_description: Optional[str] = Form(None),
    promptDescription: Optional[str] = Form(None),
    description_backend: Optional[str] = Form(None),
    descriptionBackend: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    negativePrompt: Optional[str] = Form(None),
    steps: int = Form(FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS, ge=4, le=30),
    seed: int = Form(FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED, ge=0, le=2147483647),
    upload_debug_images: bool = Form(FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_DEBUG),
    use_full_image_context: bool = Form(True),
    strict_section_enforcement: bool = Form(True),
    use_parser_board_reference: bool = Form(False),
    use_parser_post_extract: bool = Form(False),
):
    """
    Single-garment extraction API:
    1) Accept one uploaded garment image.
    2) Build prompt from MiniCPM (or provided promptDescription).
    3) Run Flux2 garment-only generation with strict negative prompt.
    4) Return 2:3 center-aligned extracted garment output + full metadata.
    """
    t0 = time.time()
    request_id = str(uuid.uuid4())
    stage_timings = {
        "read_input_s": 0.0,
        "parser_select_s": 0.0,
        "descriptor_s": 0.0,
        "prompt_build_s": 0.0,
        "flux_generation_s": 0.0,
        "flux_extract_s": 0.0,
        "upload_input_debug_s": 0.0,
        "upload_selected_crop_s": 0.0,
        "upload_raw_output_s": 0.0,
        "upload_final_output_s": 0.0,
        "upload_s": 0.0,
        "api_total_s": 0.0,
    }
    warnings: List[str] = []
    input_summary = {
        "requestId": request_id,
        "garmentTypeRaw": "",
        "garmentTypeResolved": "",
        "filename": "",
        "contentType": "",
        "bytes": 0,
        "steps": int(steps),
        "seed": int(seed),
        "descriptionBackendRequested": "",
        "descriptionBackendResolved": "",
        "hasPromptDescription": False,
        "hasCustomNegativePrompt": False,
        "useFullImageContextRequested": bool(use_full_image_context),
        "useFullImageContext": bool(use_full_image_context),
        "strictSectionEnforcement": bool(strict_section_enforcement),
        "useParserBoardReferenceRequested": bool(use_parser_board_reference),
        "useParserPostExtractRequested": bool(use_parser_post_extract),
        "useParserBoardReference": bool(use_parser_board_reference),
        "useParserPostExtract": bool(use_parser_post_extract),
        "parserDisabledByConfig": bool(FLUX2_SINGLE_GARMENT_EXTRACT_DISABLE_PARSER),
    }

    def _build_error_response(
        *,
        code: str,
        message: str,
        status_code: int,
        details: Optional[dict] = None,
    ) -> dict:
        stage_timings["api_total_s"] = round(time.time() - t0, 4)
        payload = {
            "status": int(status_code),
            "data": {
                "result": "REJECTED",
                "requestId": request_id,
                "error": {
                    "code": str(code or "EXTRACT_ERROR"),
                    "message": str(message or "Garment extraction failed"),
                    "statusCode": int(status_code),
                },
                "meta": {
                    "stageTimings": stage_timings,
                    "input": input_summary,
                    "warnings": warnings,
                },
            },
            "message": "",
        }
        if isinstance(details, dict) and details:
            payload["data"]["error"]["details"] = details
        return payload

    try:
        upload = file or image
        if upload is None:
            raise HTTPException(
                status_code=422,
                detail={"code": "IMAGE_REQUIRED", "message": "Provide one image file using `file` or `image`."},
            )

        raw_type = str(garment_type or garmentType or "").strip()
        resolved_type = _normalize_garment_type(raw_type)
        if resolved_type not in {"top", "bottom", "dress"}:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INVALID_GARMENT_TYPE",
                    "message": "type/garmentType must be one of: top, bottom, dress.",
                    "details": {"received": raw_type},
                },
            )

        t_stage = time.time()
        image_bytes = await upload.read()
        stage_timings["read_input_s"] = round(time.time() - t_stage, 4)
        if not image_bytes:
            raise HTTPException(
                status_code=422,
                detail={"code": "EMPTY_IMAGE", "message": "Uploaded file is empty."},
            )
        if len(image_bytes) > ANALYZE_MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "FILE_TOO_LARGE",
                    "message": f"Image must be <= {ANALYZE_MAX_FILE_BYTES} bytes.",
                    "details": {"maxBytes": ANALYZE_MAX_FILE_BYTES, "receivedBytes": len(image_bytes)},
                },
            )

        try:
            source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as decode_err:
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_IMAGE", "message": f"Could not decode image: {decode_err}"},
            ) from decode_err

        desc_backend_raw = str(description_backend or descriptionBackend or FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND).strip()
        requested_backend = _normalize_descriptor_backend(desc_backend_raw)
        # This endpoint is MiniCPM-first by design.
        resolved_backend = requested_backend if requested_backend in {"minicpm", "minicpm_service"} else "minicpm"

        provided_prompt = " ".join(str(prompt_description or promptDescription or "").split()).strip()
        provided_negative = " ".join(str(negative_prompt or negativePrompt or "").split()).strip()
        requested_use_full_image_context = bool(use_full_image_context)
        requested_use_parser_board_reference = bool(use_parser_board_reference)
        requested_use_parser_post_extract = False

        effective_use_full_image_context = requested_use_full_image_context
        effective_use_parser_board_reference = requested_use_parser_board_reference
        effective_use_parser_post_extract = False
        if FLUX2_SINGLE_GARMENT_EXTRACT_DISABLE_PARSER:
            effective_use_full_image_context = True
            effective_use_parser_board_reference = False
            effective_use_parser_post_extract = False
            if (
                requested_use_parser_board_reference
                or (not requested_use_full_image_context)
            ):
                warnings.append("parser_disabled_for_extract_endpoint")

        expose_intermediate_urls = bool(
            upload_debug_images or FLUX2_SINGLE_GARMENT_EXTRACT_EXPOSE_INTERMEDIATE_URLS
        )

        input_summary["garmentTypeRaw"] = raw_type
        input_summary["garmentTypeResolved"] = resolved_type
        input_summary["filename"] = str(upload.filename or "")
        input_summary["contentType"] = str(upload.content_type or "")
        input_summary["bytes"] = len(image_bytes)
        input_summary["descriptionBackendRequested"] = requested_backend
        input_summary["descriptionBackendResolved"] = resolved_backend
        input_summary["hasPromptDescription"] = bool(provided_prompt)
        input_summary["hasCustomNegativePrompt"] = bool(provided_negative)
        input_summary["useFullImageContext"] = bool(effective_use_full_image_context)
        input_summary["useParserBoardReference"] = bool(effective_use_parser_board_reference)
        input_summary["useParserPostExtract"] = bool(effective_use_parser_post_extract)

        input_image_url = None
        if expose_intermediate_urls:
            try:
                t_upload_part = time.time()
                input_image_url = _upload_or_raise(image_bytes, container=VTO_OUTPUT_CONTAINER)
                stage_timings["upload_input_debug_s"] = round(time.time() - t_upload_part, 4)
            except Exception as upload_in_err:
                warnings.append(f"input_upload_failed:{upload_in_err}")

        async with gpu_semaphore:
            person_image_for_flux = source_image
            board_image_for_flux = source_image
            descriptor_image = source_image
            selected_candidate_obj: dict = {}

            if effective_use_full_image_context:
                parser_meta = {
                    "strategy": "full_image_context",
                    "selectedIndex": 0,
                    "selectedType": resolved_type,
                    "candidateCount": 1,
                    "selectedCandidate": None,
                    "attempts": [],
                }
                stage_timings["parser_select_s"] = 0.0
                # Optional parser assist:
                # keep full image for person/context, but use focused garment crop as board reference.
                if effective_use_parser_board_reference and engine.parser is not None:
                    t_stage = time.time()
                    try:
                        focused_board, focused_meta = _select_single_garment_candidate(
                            image=source_image,
                            garment_type=resolved_type,
                        )
                        board_image_for_flux = focused_board
                        descriptor_image = focused_board
                        selected_candidate_obj = dict(focused_meta.get("selectedCandidate") or {})
                        parser_meta["boardReferenceSource"] = "parser_candidate"
                        parser_meta["boardReferenceStrategy"] = str(focused_meta.get("strategy") or "")
                        parser_meta["boardReferenceCandidateCount"] = int(focused_meta.get("candidateCount") or 0)
                        parser_meta["selectedCandidate"] = selected_candidate_obj or None
                        stage_timings["parser_select_s"] = round(time.time() - t_stage, 4)
                    except Exception as parser_assist_err:
                        warnings.append(f"parser_board_reference_unavailable:{parser_assist_err}")
            else:
                if engine.parser is None:
                    raise HTTPException(
                        status_code=503,
                        detail={"code": "PARSER_UNAVAILABLE", "message": "Human parser is required when use_full_image_context=false."},
                    )
                t_stage = time.time()
                selected_crop, parser_meta = _select_single_garment_candidate(
                    image=source_image,
                    garment_type=resolved_type,
                )
                person_image_for_flux = selected_crop
                board_image_for_flux = selected_crop
                descriptor_image = selected_crop
                selected_candidate_obj = dict(parser_meta.get("selectedCandidate") or {})
                stage_timings["parser_select_s"] = round(time.time() - t_stage, 4)

            selected_crop_bytes_io = io.BytesIO()
            descriptor_image.save(selected_crop_bytes_io, format="PNG")
            selected_crop_bytes = selected_crop_bytes_io.getvalue()

            selected_crop_url = None
            need_selected_crop_upload = (resolved_backend == "minicpm_service")
            if expose_intermediate_urls or need_selected_crop_upload:
                try:
                    t_upload_part = time.time()
                    selected_crop_url = _upload_or_raise(selected_crop_bytes, container=VTO_OUTPUT_CONTAINER)
                    stage_timings["upload_selected_crop_s"] = round(time.time() - t_upload_part, 4)
                except Exception as selected_upload_err:
                    warnings.append(f"selected_crop_upload_failed:{selected_upload_err}")

            if resolved_backend == "minicpm_service":
                if not str(selected_crop_url or "").startswith("http"):
                    warnings.append("minicpm_service_requires_http_url_fallback_to_local_minicpm")
                    resolved_backend = "minicpm"
                    input_summary["descriptionBackendResolved"] = resolved_backend

            t_stage = time.time()
            if provided_prompt:
                garment_desc = provided_prompt
                prompt_source = "request"
            else:
                garment_desc = _describe_garment_with_backend(
                    image=descriptor_image,
                    backend=resolved_backend,
                    image_url=selected_crop_url,
                )
                garment_desc = " ".join(str(garment_desc or "").split()).strip()
                prompt_source = resolved_backend
            stage_timings["descriptor_s"] = round(time.time() - t_stage, 4)
            if not garment_desc:
                raise HTTPException(
                    status_code=502,
                    detail={"code": "PROMPT_GENERATION_FAILED", "message": "Could not generate garment prompt description."},
                )

            selected_candidate = selected_candidate_obj if isinstance(selected_candidate_obj, dict) else {}
            parser_hexes = (
                [str(v) for v in (selected_candidate.get("dominant_color_hexes") or []) if str(v).strip()]
                if isinstance(selected_candidate, dict)
                else []
            )
            descriptor_color_mask = _estimate_type_focused_color_mask(
                descriptor_image,
                resolved_type,
                garment_desc,
            )
            input_color_ctx = _build_single_image_color_context(
                image=descriptor_image,
                description=garment_desc,
                mask=descriptor_color_mask,
                top_k=7,
                force_masking=bool(isinstance(descriptor_color_mask, np.ndarray)),
            )
            dominant_hexes = [
                str(v)
                for v in (input_color_ctx.get("dominantHexes") or input_color_ctx.get("paletteHexes") or [])
                if str(v).strip()
            ]
            if not dominant_hexes:
                dominant_hexes = [str(v) for v in parser_hexes if str(v).strip()]
            if not dominant_hexes:
                dominant_hexes = _extract_dominant_hex_colors(
                    image=descriptor_image,
                    top_k=4,
                )
            color_hints = [str(v) for v in (input_color_ctx.get("hints") or []) if str(v).strip()]
            color_profile = input_color_ctx.get("profile") if isinstance(input_color_ctx.get("profile"), dict) else {}
            category_text = str((selected_candidate or {}).get("category_text") or resolved_type).strip()

            t_stage = time.time()
            flux_prompt = _build_flux2_single_garment_extract_prompt(
                garment_type=resolved_type,
                prompt_description=garment_desc,
                category_text=category_text,
                dominant_color_hexes=dominant_hexes,
                color_hints=color_hints,
                color_profile=color_profile,
            )
            built_negative_prompt = _build_flux2_single_garment_extract_negative_prompt(
                garment_type=resolved_type,
                custom_negative_prompt=provided_negative,
            )
            stage_timings["prompt_build_s"] = round(time.time() - t_stage, 4)

            t_stage = time.time()
            flux_result = engine.flux2.run_tryon(
                person_image=person_image_for_flux,
                board_image=board_image_for_flux,
                prompt=flux_prompt,
                steps=int(steps),
                seed=int(seed),
                negative_prompt=built_negative_prompt,
                use_lora=False,
            )
            stage_timings["flux_generation_s"] = round(time.time() - t_stage, 4)

            flux_raw_img = flux_result["image"].convert("RGB")
            raw_buf = io.BytesIO()
            flux_raw_img.save(raw_buf, format="PNG")
            raw_bytes = raw_buf.getvalue()
            raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            parser_input_source = "flux_raw"
            parser_input_sha256 = raw_sha256

            t_stage = time.time()
            extraction_mode = "background_removal_only"
            extraction_meta = {
                "path": extraction_mode,
                "parserSkipped": True,
            }
            try:
                final_bytes, postprocess_meta = _postprocess_extracted_garment_bytes(
                    raw_bytes,
                    garment_type=resolved_type,
                    force_transparent=True,
                    transparent_only=True,
                )
            except Exception as extract_err:
                extraction_mode = "crop_then_aspect_only_fallback"
                extraction_meta = {
                    "path": extraction_mode,
                    "warning": str(extract_err),
                }
                final_bytes, postprocess_meta = _crop_and_fit_garment_bytes(raw_bytes)

            extraction_meta = dict(extraction_meta or {})
            extraction_meta["raw_flux_sha256"] = raw_sha256
            extraction_meta["parser_input_source"] = "none"
            extraction_meta["parser_input_sha256"] = ""
            extraction_meta["parser_used_flux_raw"] = False
            extraction_meta["parser_input_matches_flux_raw"] = False

            try:
                final_img = Image.open(io.BytesIO(final_bytes))
                if final_img.mode not in {"RGB", "RGBA"}:
                    final_img = final_img.convert("RGBA")
                final_img = _fit_image_to_aspect(final_img, aspect_w=2, aspect_h=3)
                final_buf = io.BytesIO()
                final_img.save(final_buf, format="PNG")
                final_bytes = final_buf.getvalue()
            except Exception as force_ratio_err:
                warnings.append(f"force_2x3_failed:{force_ratio_err}")
                final_img = Image.open(io.BytesIO(final_bytes)).convert("RGBA")

            output_color_ctx = _build_single_image_color_context(
                image=final_img.convert("RGB"),
                description="",
                mask=_extract_alpha_mask(final_img, threshold=24),
                top_k=7,
            )
            color_compare = _compute_palette_delta_e(
                input_hexes=[
                    str(v)
                    for v in (input_color_ctx.get("dominantHexes") or input_color_ctx.get("paletteHexes") or [])
                    if str(v).strip()
                ],
                output_hexes=[
                    str(v)
                    for v in (output_color_ctx.get("dominantHexes") or output_color_ctx.get("paletteHexes") or [])
                    if str(v).strip()
                ],
                max_colors=5,
            )
            stage_timings["flux_extract_s"] = round(time.time() - t_stage, 4)

            t_stage = time.time()
            raw_output_url = None
            if expose_intermediate_urls and FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_RAW_DEBUG:
                t_upload_part = time.time()
                raw_output_url = _upload_or_raise(raw_bytes, container=VTO_OUTPUT_CONTAINER)
                stage_timings["upload_raw_output_s"] = round(time.time() - t_upload_part, 4)
            t_upload_part = time.time()
            final_output_url = _upload_or_raise(final_bytes, container=VTO_OUTPUT_CONTAINER)
            stage_timings["upload_final_output_s"] = round(time.time() - t_upload_part, 4)
            stage_timings["upload_s"] = round(time.time() - t_stage, 4)

        stage_timings["api_total_s"] = round(time.time() - t0, 4)
        process_trace = [
            {"stage": "read_input", "seconds": stage_timings["read_input_s"]},
            {"stage": "parser_select", "seconds": stage_timings["parser_select_s"]},
            {"stage": "descriptor", "seconds": stage_timings["descriptor_s"]},
            {"stage": "prompt_build", "seconds": stage_timings["prompt_build_s"]},
            {"stage": "flux_generation", "seconds": stage_timings["flux_generation_s"]},
            {"stage": "flux_extract", "seconds": stage_timings["flux_extract_s"]},
            {"stage": "upload", "seconds": stage_timings["upload_s"]},
        ]

        payload = _build_success_payload(
            data={
                "result": "ACCEPTED",
                "requestId": request_id,
                "api": "/v1/flux2/extract-garment",
                "input": input_summary,
                "parser": parser_meta,
                "descriptor": {
                    "source": prompt_source,
                    "backendRequested": requested_backend,
                    "backendResolved": resolved_backend,
                    "promptDescription": garment_desc,
                },
                "prompts": {
                    "fluxPrompt": flux_prompt,
                    "negativePrompt": built_negative_prompt,
                    "negativePromptSource": "request" if provided_negative else "auto",
                    "dominantHexesUsed": dominant_hexes,
                },
                "flux": {
                    "steps": int(steps),
                    "seed": int(seed),
                    "latency_s": round(float(flux_result.get("latency", 0.0)), 4),
                    "metadata": flux_result.get("metadata", {}),
                },
                "color": {
                    "input": input_color_ctx,
                    "output": output_color_ctx,
                    "compare": color_compare,
                },
                "extraction": {
                    "mode": extraction_mode,
                    "warning": extraction_warning or None,
                    "meta": extraction_meta or {},
                    "postprocess": postprocess_meta or {},
                    "outputAspect": "2:3",
                    "centerAligned": True,
                },
                "outputs": {
                    "inputImageUrl": input_image_url if expose_intermediate_urls else None,
                    "selectedCropUrl": selected_crop_url if expose_intermediate_urls else None,
                    "rawOutputUrl": raw_output_url if expose_intermediate_urls else None,
                    "rawOutputPurpose": "debug_before_parser_extraction" if raw_output_url else None,
                    "outputUrl": final_output_url or raw_output_url,
                },
                "stageTimings": stage_timings,
                "processTrace": process_trace,
                "warnings": warnings,
            },
            status_code=200,
            message="",
        )
        return _json_response(payload)
    except HTTPException as he:
        detail = he.detail if isinstance(he.detail, dict) else {"message": str(he.detail)}
        code = str(detail.get("code") or "REQUEST_INVALID")
        message = str(detail.get("message") or "Request failed")
        details = detail.get("details") if isinstance(detail.get("details"), dict) else None
        return _json_response(
            _build_error_response(
                code=code,
                message=message,
                status_code=he.status_code,
                details=details,
            )
        )
    except Exception as err:
        logger.error(f"Flux2 single garment extraction failed: {err}")
        return _json_response(
            _build_error_response(
                code="EXTRACT_INTERNAL_ERROR",
                message=str(err),
                status_code=500,
            )
        )

@app.post("/v1/user-image/prepare")
@app.post("/v1/flux2/prepare-user-image")
async def prepare_user_image_for_tryon(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    t0 = time.time()
    stage = {
        "read_input_s": 0.0,
        "blur_check_s": 0.0,
        "parser_s": 0.0,
        "crop_s": 0.0,
        "bg_remove_s": 0.0,
        "upload_s": 0.0,
        "describe_s": 0.0,
        "api_total_s": 0.0,
    }

    upload = file or image
    if upload is None:
        return _json_response(
            _build_user_prepare_payload(
                status_code=400,
                message="Provide one image file using `file` or `image`.",
            )
        )

    gpu_slot_acquired = False
    try:
        t_stage = time.time()
        raw_bytes = await upload.read()
        stage["read_input_s"] = round(time.time() - t_stage, 4)
        if not raw_bytes:
            return _json_response(
                _build_user_prepare_payload(status_code=400, message="Uploaded image is empty.")
            )
        if len(raw_bytes) > USER_PREP_MAX_FILE_BYTES:
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message=f"File too large. Max allowed is {USER_PREP_MAX_FILE_BYTES // (1024 * 1024)}MB.",
                )
            )

        try:
            src_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return _json_response(
                _build_user_prepare_payload(status_code=400, message="Invalid image file.")
            )

        if USER_PREP_BLUR_CHECK_ENABLED:
            t_stage = time.time()
            focus_score = _focus_score(src_img)
            stage["blur_check_s"] = round(time.time() - t_stage, 4)
            if focus_score < USER_PREP_MIN_FOCUS_SCORE:
                return _json_response(
                    _build_user_prepare_payload(
                        status_code=400,
                        message="Image is too blurry. Upload a sharper full-person photo.",
                    )
                )

        try:
            await asyncio.wait_for(gpu_semaphore.acquire(), timeout=ANALYZE_GPU_QUEUE_TIMEOUT_S)
            gpu_slot_acquired = True
        except asyncio.TimeoutError:
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message="Server is busy. Please retry in a few seconds.",
                )
            )

        t_stage = time.time()
        parsing, _, components, parser_meta = _user_prep_person_components(src_img)
        stage["parser_s"] = round(time.time() - t_stage, 4)
        if not components:
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message="No clear person detected in the image.",
                )
            )

        selected = _user_prep_select_main_component(components, src_img.width, src_img.height)
        if not selected:
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message="Could not identify the main person in the image.",
                )
            )
        if float(selected.get("area_ratio", 0.0)) < USER_PREP_MAIN_PERSON_MIN_AREA_RATIO:
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message="Person is too small in frame. Upload a closer full-person image.",
                )
            )
        if USER_PREP_REQUIRE_FACE:
            face_ok, face_meta = _user_prep_validate_face(
                src_img,
                parsing=parsing,
                person_component_mask=selected.get("mask") if isinstance(selected.get("mask"), np.ndarray) else None,
            )
            if not face_ok:
                logger.info(f"user_image_prepare rejected: face validation failed meta={face_meta}")
                return _json_response(
                    _build_user_prepare_payload(
                        status_code=400,
                        message="No clear face detected. Upload a front-facing image with visible face.",
                    )
                )

        t_stage = time.time()
        person_crop, crop_bbox = _user_prep_crop_main_person(src_img, [int(v) for v in selected.get("bbox", [])])
        stage["crop_s"] = round(time.time() - t_stage, 4)
        if min(person_crop.width, person_crop.height) < USER_PREP_MIN_CROP_SIDE_PX:
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message="Detected person crop is too small for try-on.",
                )
            )

        if USER_PREP_BLUR_CHECK_ENABLED:
            crop_focus = _focus_score(person_crop)
            if crop_focus < USER_PREP_MIN_FOCUS_SCORE:
                return _json_response(
                    _build_user_prepare_payload(
                        status_code=400,
                        message="Main person region is blurry. Upload a clearer image.",
                    )
                )

        crop_buf = io.BytesIO()
        person_crop.save(crop_buf, format="PNG")
        crop_bytes = crop_buf.getvalue()

        t_stage = time.time()
        processed_bytes = crop_bytes
        bg_meta: dict = {"applied": False, "required": bool(USER_PREP_REQUIRE_BG_REMOVAL)}
        removed_bytes, remove_meta = _remove_user_background(crop_bytes)
        if removed_bytes:
            processed_bytes = removed_bytes
            bg_meta = {"applied": True, **(remove_meta or {})}
        elif USER_PREP_REQUIRE_BG_REMOVAL:
            logger.warning(f"User prep background removal failed meta={remove_meta}")
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message="Background removal failed for this image. Try another image.",
                )
            )
        stage["bg_remove_s"] = round(time.time() - t_stage, 4)

        t_stage = time.time()
        output_url = _upload_or_raise(processed_bytes, container=USER_PREP_UPLOAD_CONTAINER)
        stage["upload_s"] = round(time.time() - t_stage, 4)
        if not output_url:
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message="Failed to upload processed image.",
                )
            )

        processed_img = Image.open(io.BytesIO(processed_bytes)).convert("RGB")
        t_stage = time.time()
        prompt_description = _describe_user_image_for_prepare(processed_img, output_url)
        stage["describe_s"] = round(time.time() - t_stage, 4)
        if not prompt_description:
            return _json_response(
                _build_user_prepare_payload(
                    status_code=400,
                    message="Failed to generate user image description.",
                )
            )

        stage["api_total_s"] = round(time.time() - t0, 4)
        logger.info(
            "user_image_prepare success total=%.2fs components=%s selected_area=%.4f crop_bbox=%s bg=%s",
            stage["api_total_s"],
            parser_meta.get("components", 0),
            float(selected.get("area_ratio", 0.0)),
            crop_bbox,
            bg_meta,
        )
        return _json_response(
            _build_user_prepare_payload(
                status_code=200,
                message="User image prepared successfully.",
                url=output_url,
                prompt_description=prompt_description,
            )
        )
    except Exception as err:
        logger.exception(f"User image prepare failed: {err}")
        return _json_response(
            _build_user_prepare_payload(
                status_code=400,
                message=f"User image prepare failed: {err}",
            )
        )
    finally:
        stage["api_total_s"] = round(time.time() - t0, 4)
        if gpu_slot_acquired:
            gpu_semaphore.release()

@app.post("/analyze")
@app.post("/analzye")
async def analyze_garment(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form(None, alias="type"),
    garmentType: Optional[str] = Form(None),
    use_parser_post_extract: Optional[bool] = Form(None),
    useParserPostExtract: Optional[bool] = Form(None),
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
    analyze_stage_timings = default_extraction_stage_timings()

    # ── AUTH ──
    auth_result = verify_authorization_or_response(
        _authorization,
        verify_bearer_token=_verify_bearer_token,
        build_error_payload=_build_error_payload,
        multipart_form_response=_multipart_form_response,
    )
    if not isinstance(auth_result, dict):
        return auth_result

    # ── INPUT VALIDATION ──
    upload = resolve_upload_or_response(
        file_upload=file,
        image_upload=image,
        build_error_payload=_build_error_payload,
        multipart_form_response=_multipart_form_response,
    )
    if not hasattr(upload, "read"):
        return upload

    effective_type = garment_type or garmentType
    requested_type = _normalize_garment_type(effective_type)
    # /analyze parser extraction fallback is disabled by design for color consistency and latency stability.
    requested_use_parser_post_extract = False

    try:
        # 1. Load Image
        image_input = await read_input_image_or_response(
            upload,
            max_file_bytes=ANALYZE_MAX_FILE_BYTES,
            blur_check_enabled=ANALYZE_BLUR_CHECK_ENABLED,
            blur_min_focus_score=ANALYZE_BLUR_MIN_FOCUS_SCORE,
            focus_score_fn=_focus_score,
            build_error_payload=_build_error_payload,
            multipart_form_response=_multipart_form_response,
            stage_timings=analyze_stage_timings,
        )
        if not hasattr(image_input, "image"):
            return image_input
        image_bytes = image_input.image_bytes
        img = image_input.image

        gpu_queue_wait_s = 0.0
        gpu_slot_acquired = False
        t_gpu_wait = time.time()
        try:
            await asyncio.wait_for(gpu_semaphore.acquire(), timeout=ANALYZE_GPU_QUEUE_TIMEOUT_S)
            gpu_slot_acquired = True
            gpu_queue_wait_s = round(time.time() - t_gpu_wait, 4)

            # 2. Detect & Crop (YOLO) or explicit-type fast path.
            detection_result, detection_response = run_detection_stage_or_response(
                image=img,
                requested_type=requested_type,
                selected_index=selected_index,
                engine=engine,
                stage_timings=analyze_stage_timings,
                logger=logger,
                build_error_payload=_build_error_payload,
                multipart_form_response=_multipart_form_response,
                wardrobe_category_from_garment_type=_wardrobe_category_from_garment_type,
                normalize_garment_type=_normalize_garment_type,
                parser_preroute_instances=_parser_preroute_instances,
                parser_split_is_plausible=_parser_split_is_plausible,
                heuristic_split_candidates=_heuristic_split_candidates,
                should_force_fullbody_split=_should_force_fullbody_split,
                largest_instance=_largest_instance,
                suppress_auxiliary_instances=_suppress_auxiliary_instances,
                bbox_prior=_bbox_prior,
                hybrid_score=_hybrid_score,
                is_shorts_like_caption=_is_shorts_like_caption,
                infer_type_from_caption=_infer_type_from_caption,
                caption_non_garment_signal=_caption_non_garment_signal,
                caption_garment_signal=_caption_garment_signal,
                infer_style_from_text=_infer_style_from_text,
                dedupe_items=_dedupe_items,
                should_collapse_same_type=_should_collapse_same_type,
                item_rank_score=_item_rank_score,
                maybe_force_uncertain_fullbody_to_dress=_maybe_force_uncertain_fullbody_to_dress,
                requested_type_geometry_score=_requested_type_geometry_score,
                analyze_use_parser_for_prerouting=ANALYZE_USE_PARSER_FOR_PREROUTING,
                analyze_enable_parser_split=ANALYZE_ENABLE_PARSER_SPLIT,
                analyze_enable_heuristic_split=ANALYZE_ENABLE_HEURISTIC_SPLIT,
                analyze_florence_dress_lock_min_score=ANALYZE_FLORENCE_DRESS_LOCK_MIN_SCORE,
                analyze_heuristic_bottom_trim_shorts_ratio=ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO,
                use_florence_hybrid_verify=USE_FLORENCE_HYBRID_VERIFY,
                hybrid_top_k=HYBRID_TOP_K,
                hybrid_min_score=HYBRID_MIN_SCORE,
                analyze_max_items=ANALYZE_MAX_ITEMS,
                analyze_caption_mode=ANALYZE_CAPTION_MODE,
                analyze_primary_type_with_florence=ANALYZE_PRIMARY_TYPE_WITH_FLORENCE,
                analyze_auto_select_multi_dress=ANALYZE_AUTO_SELECT_MULTI_DRESS,
            )
            if detection_response is not None:
                return detection_response
            items = list(detection_result.get("items") or [])
            direct_requested_type_mode = bool(detection_result.get("direct_requested_type_mode"))
            raw_detected_count = int(detection_result.get("raw_detected_count") or 0)
            parser_split_used = bool(detection_result.get("parser_split_used"))
            heuristic_split_used = bool(detection_result.get("heuristic_split_used"))
            auto_selected_index = detection_result.get("auto_selected_index")

            # ── MULTI-ITEM SELECTION REQUIRED (400 with multipart) ──
            if ANALYZE_REQUIRE_SELECTION and len(items) > 1 and selected_index is None and auto_selected_index is None:
                return _build_extraction_selection_required_response(
                    items=items,
                    full_image=img,
                    started_at=t0,
                    gpu_queue_wait_s=gpu_queue_wait_s,
                    raw_detected_count=raw_detected_count,
                    parser_split_used=parser_split_used,
                    heuristic_split_used=heuristic_split_used,
                    selection_preview_format=ANALYZE_SELECTION_PREVIEW_FORMAT,
                    selection_preview_max_side=ANALYZE_SELECTION_PREVIEW_MAX_SIDE,
                    selection_preview_jpeg_quality=ANALYZE_SELECTION_PREVIEW_JPEG_QUALITY,
                    to_public_item=_to_public_item,
                    build_adaptive_rect_crop_variants=_build_adaptive_rect_crop_variants,
                    prepare_extract_source_image=_prepare_extract_source_image,
                    build_multipart_parts=_build_multipart_parts,
                    build_success_payload=_build_success_payload,
                    multipart_form_response=_multipart_form_response,
                )

            selected_item, _selected_index_internal, selected_item_response = resolve_selected_item_or_response(
                items=items,
                requested_type=requested_type,
                selected_index=selected_index,
                auto_selected_index=auto_selected_index,
                min_accept_confidence=ANALYZE_MIN_ACCEPT_CONFIDENCE,
                build_error_payload=_build_error_payload,
                multipart_form_response=_multipart_form_response,
            )
            if selected_item_response is not None:
                return selected_item_response

            selected_item, selected_item_response = run_selected_item_extraction_or_response(
                selected_item=selected_item,
                requested_type=requested_type,
                direct_requested_type_mode=direct_requested_type_mode,
                full_image=img,
                all_items_count=len(items),
                stage_timings=analyze_stage_timings,
                engine=engine,
                logger=logger,
                analyze_extract_cloth=ANALYZE_EXTRACT_CLOTH,
                analyze_prompt_from_extracted=ANALYZE_PROMPT_FROM_EXTRACTED,
                analyze_require_extracted_prompt=ANALYZE_REQUIRE_EXTRACTED_PROMPT,
                analyze_caption_mode=ANALYZE_CAPTION_MODE,
                flux2_single_garment_extract_default_steps=FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS,
                flux2_single_garment_extract_default_seed=FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED,
                normalize_garment_type=_normalize_garment_type,
                prepare_extract_source_image=_prepare_extract_source_image,
                build_error_payload=_build_error_payload,
                multipart_form_response=_multipart_form_response,
                run_flux2_cloth_only_extract=_run_flux2_cloth_only_extract,
                descriptor_is_weak=_descriptor_is_weak,
                caption_non_garment_signal=_caption_non_garment_signal,
                download_image=download_image,
                flatten_rgba_on_white=_flatten_rgba_on_white,
                sanitize_garment_description=_sanitize_garment_description,
                infer_style_from_text=_infer_style_from_text,
                wardrobe_category_from_garment_type=_wardrobe_category_from_garment_type,
            )
            if selected_item_response is not None:
                return selected_item_response

            prompting_context = {}
            if selected_item:
                selected_item, prompting_context = apply_selected_item_prompting(
                    selected_item=selected_item,
                    requested_type=requested_type,
                    analyze_prompt_from_extracted=ANALYZE_PROMPT_FROM_EXTRACTED,
                    normalize_garment_type=_normalize_garment_type,
                    infer_style_from_text=_infer_style_from_text,
                    wardrobe_category_from_garment_type=_wardrobe_category_from_garment_type,
                    product_prompt_description=_product_prompt_description,
                    build_garment_metadata=_build_garment_metadata,
                    strip_descriptor_color_clause=_strip_descriptor_color_clause,
                )
                selected_item, _sync_context = sync_selected_item_progress(
                    selected_item=selected_item,
                    requested_type=requested_type,
                    stage_timings=analyze_stage_timings,
                    enable_progress_sync=ENABLE_WARDROBE_PROGRESS_SYNC,
                    authorization=_authorization,
                    sync_wardrobe_progress_dispatch=_sync_wardrobe_progress_dispatch,
                    prompting_context=prompting_context,
                )
        except asyncio.TimeoutError:
            wait_s = round(time.time() - t_gpu_wait, 4)
            payload = _build_error_payload(
                title="Server Busy",
                description="The server is processing other requests. Please retry in a few seconds.",
                reason_codes=["GPU_QUEUE_TIMEOUT"],
                status_code=503,
                result="REJECTED",
            )
            payload_data = payload.setdefault("data", {})
            payload_data["latencies"] = {"gpu_queue_wait": wait_s}
            payload_data["processing_time_ms"] = int(wait_s * 1000)
            return _multipart_form_response(payload)
        finally:
            if gpu_slot_acquired:
                gpu_semaphore.release()

        # ── SUCCESS RESPONSE (200 with multipart: metadata + extracted_cloth) ──
        public_item = _to_public_item(selected_item) if selected_item else None
        total_s = round(time.time() - t0, 4)
        analyze_stage_timings["caption_total_s"] = round(float(analyze_stage_timings["caption_total_s"]), 4)
        analyze_stage_timings["primary_type_total_s"] = round(float(analyze_stage_timings["primary_type_total_s"]), 4)
        return _build_extraction_success_response(
            public_item=public_item,
            selected_item=selected_item,
            items=items,
            requested_type=requested_type,
            total_s=total_s,
            gpu_queue_wait_s=gpu_queue_wait_s,
            analyze_stage_timings=analyze_stage_timings.to_dict(),
            wardrobe_category_from_garment_type=_wardrobe_category_from_garment_type,
            build_multipart_parts=_build_multipart_parts,
            build_success_payload=_build_success_payload,
            multipart_form_response=_multipart_form_response,
            fetch_image_bytes=_fetch_image_bytes,
        )

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
    descriptor_service_url = str(ANALYZE_MINICPM_SERVICE_URL or MINICPM_SERVICE_URL or "").strip().rstrip("/")
    request_disable_neutral_calibration = bool(request.disable_neutral_calibration)
    stage_timings = {
        "gpu_queue_wait_s": 0.0,
        "download_user_image_s": 0.0,
        "download_products_s": 0.0,
        "descriptor_wall_s": 0.0,
        "product_prompt_generation_total_s": 0.0,
        "product_prompt_generation_per_item_s": [],
        "user_prompt_generation_s": 0.0,
        "board_build_s": 0.0,
        "prompt_build_s": 0.0,
        "flux_generation_sum_s": 0.0,
        "candidate_scoring_sum_s": 0.0,
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
        "descriptionServiceUrl": descriptor_service_url if descriptor_backend == "minicpm_service" else "",
        "descriptionCompareRequested": bool(request.description_compare),
        "hasCustomNegativePrompt": bool(str(request.negative_prompt or "").strip()),
        "disableNeutralCalibrationRequested": request_disable_neutral_calibration,
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
        product_garment_metadata: List[Dict[str, object]] = []
        for idx, product in enumerate(request.products):
            product_url = str(product.image or "").strip()
            if not product_url:
                raise HTTPException(status_code=422, detail=f"products[{idx}].image is required")
            product_urls.append(product_url)
            garment_metadata_obj = product.garmentMetadata if isinstance(product.garmentMetadata, dict) else {}
            product_garment_metadata.append(dict(garment_metadata_obj or {}))
            metadata_prompt = _extract_garment_metadata_prompt(garment_metadata_obj)
            metadata_target_type = _extract_garment_metadata_target_type(garment_metadata_obj)
            input_summary["products"].append(
                {
                    "index": idx,
                    "image": product_url,
                    "promptProvided": bool(
                        " ".join(str(product.promptDescription or "").split()).strip() or metadata_prompt
                    ),
                    "garmentMetadataProvided": bool(garment_metadata_obj),
                    "targetTypeProvided": bool(str(product.targetType or "").strip()),
                    "targetTypeValue": (
                        str(product.targetType or "").strip()
                        or metadata_target_type
                        or None
                    ),
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
        product_descriptor_color_hexes: List[List[str]] = []
        product_descriptor_color_hints: List[List[str]] = []
        for idx, product_img in enumerate(product_imgs):
            metadata_hexes, metadata_hints = _extract_garment_metadata_color_payload(
                product_garment_metadata[idx] if idx < len(product_garment_metadata) else {}
            )
            product_color_ctx = None
            if not metadata_hexes and not metadata_hints:
                product_color_ctx = _build_single_image_color_context(
                    image=product_img,
                    description="",
                    mask=None,
                    top_k=7,
                )
            product_descriptor_color_hexes.append(
                metadata_hexes
                or [
                    str(v)
                    for v in (product_color_ctx.get("dominantHexes") or product_color_ctx.get("paletteHexes") or [])
                    if str(v).strip()
                ]
            )
            product_descriptor_color_hints.append(
                metadata_hints
                or [
                    str(v)
                    for v in (product_color_ctx.get("colorHints") or product_color_ctx.get("hints") or [])
                    if str(v).strip()
                ]
            )
        descriptor_compare_enabled = bool(request.description_compare or FLUX2_DESCRIPTOR_COMPARE)
        compare_candidates = ["florence", "joycaption", "minicpm", "minicpm_service"]
        if FLUX2_ALLOW_QWEN_BACKEND:
            compare_candidates.append("qwen2_5_vl")
        compare_candidates.append(descriptor_backend)
        compare_backends: tuple[str, ...] = tuple(dict.fromkeys(compare_candidates))
        descriptor_comparisons = {"products": [], "user_image": {}} if descriptor_compare_enabled else None

        t_gpu_wait = time.time()
        async with gpu_semaphore:
            stage_timings["gpu_queue_wait_s"] = round(time.time() - t_gpu_wait, 4)
            # 2. Resolve product prompt descriptions (request-provided or selected model backend)
            product_descriptions: List[str] = []
            product_target_types: List[str] = []
            product_target_type_sources: List[str] = []
            generated_product_prompt_indices: List[int] = []
            provided_product_prompts: List[str] = []
            requested_target_types: List[Optional[str]] = []
            for idx, product in enumerate(request.products):
                garment_metadata_obj = product_garment_metadata[idx] if idx < len(product_garment_metadata) else {}
                metadata_prompt = _extract_garment_metadata_prompt(garment_metadata_obj)
                metadata_target_type = _extract_garment_metadata_target_type(garment_metadata_obj)
                provided_prompt = " ".join(str(product.promptDescription or metadata_prompt or "").split()).strip()
                requested_type_raw = str(product.targetType or metadata_target_type or "").strip()
                requested_target_type = _normalize_garment_type(requested_type_raw) if requested_type_raw else None
                if requested_type_raw and requested_target_type is None:
                    raise HTTPException(
                        status_code=422,
                        detail=f"products[{idx}].targetType must be one of top, bottom, dress, outer",
                    )
                provided_product_prompts.append(provided_prompt)
                requested_target_types.append(requested_target_type)
                if not provided_prompt:
                    generated_product_prompt_indices.append(idx)

            user_prompt_raw = str(request.user_image.promptDescription or "").strip()
            user_prompt_generated = False
            generated_product_desc_results: Dict[int, Tuple[Dict[str, str], float]] = {}
            generated_user_desc_result: Optional[Tuple[str, float]] = None

            def _timed_call(fn, *args, **kwargs):
                t_call = time.time()
                out = fn(*args, **kwargs)
                return out, (time.time() - t_call)

            if generated_product_prompt_indices or (not user_prompt_raw):
                t_desc_wall = time.time()
                max_workers = max(1, len(generated_product_prompt_indices) + (0 if user_prompt_raw else 1))
                with ThreadPoolExecutor(max_workers=max_workers) as desc_pool:
                    future_map = {}
                    for idx in generated_product_prompt_indices:
                        future = desc_pool.submit(
                            _timed_call,
                            _describe_garment_prompt_bundle_with_backend,
                            product_imgs[idx],
                            descriptor_backend,
                            image_url=product_urls[idx],
                            service_url=descriptor_service_url if descriptor_backend == "minicpm_service" else None,
                            garment_type=str(requested_target_types[idx] or ""),
                            dominant_color_hexes=product_descriptor_color_hexes[idx],
                            color_hints=product_descriptor_color_hints[idx],
                        )
                        future_map[future] = ("product", idx)
                    if not user_prompt_raw:
                        future = desc_pool.submit(
                            _timed_call,
                            _describe_user_image_for_flux2,
                            user_img,
                            backend=descriptor_backend,
                            image_url=user_image_url,
                            service_url=descriptor_service_url if descriptor_backend == "minicpm_service" else None,
                        )
                        future_map[future] = ("user", -1)

                    for future in as_completed(future_map):
                        kind, idx = future_map[future]
                        text, elapsed = future.result()
                        if kind == "product":
                            generated_product_desc_results[int(idx)] = (dict(text or {}), float(elapsed))
                        else:
                            generated_user_desc_result = (str(text or ""), float(elapsed))
                stage_timings["descriptor_wall_s"] = round(time.time() - t_desc_wall, 4)

            for idx, (product, product_img) in enumerate(zip(request.products, product_imgs)):
                provided_prompt = provided_product_prompts[idx]
                requested_target_type = requested_target_types[idx]
                resolved_target_type = requested_target_type or "top"
                item_prompt_gen_s = 0.0
                if provided_prompt:
                    cleaned_desc = _sanitize_florence_garment_description(provided_prompt)
                else:
                    generated_bundle, elapsed = generated_product_desc_results[idx]
                    item_prompt_gen_s += float(elapsed)
                    generated_desc = str(generated_bundle.get("base_garment_prompt") or "")
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
                                    service_url=descriptor_service_url if backend_name == "minicpm_service" else None,
                                    garment_type=str(requested_target_type or ""),
                                    dominant_color_hexes=product_descriptor_color_hexes[idx],
                                    color_hints=product_descriptor_color_hints[idx],
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
            if not user_prompt_raw:
                if generated_user_desc_result is None:
                    raise RuntimeError("user prompt generation failed: descriptor result missing")
                user_prompt_raw, user_desc_elapsed = generated_user_desc_result
                stage_timings["user_prompt_generation_s"] = round(float(user_desc_elapsed), 4)
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
                                service_url=descriptor_service_url if backend_name == "minicpm_service" else None,
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
                garment_metadata_list=product_garment_metadata,
            )
            prompt_product_descriptions = list(product_descriptions)
            if descriptor_backend in {"minicpm", "minicpm_service"}:
                for idx in generated_product_prompt_indices:
                    if idx < 0 or idx >= len(prompt_product_descriptions):
                        continue
                    prompt_product_descriptions[idx] = _strip_descriptor_color_clause(
                        prompt_product_descriptions[idx]
                    )
            collage_item_clause = _build_flux2_collage_item_clause(
                garment_descriptions=prompt_product_descriptions,
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
                garment_descriptions=prompt_product_descriptions,
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

            def _run_candidate(
                candidate_prompt: str,
                candidate_steps: int,
                candidate_seed: int,
                label: str,
                compute_runtime_scores: bool = True,
            ):
                candidate_result = engine.flux2.run_tryon(
                    person_image=user_img,
                    board_image=board,
                    prompt=candidate_prompt,
                    steps=candidate_steps,
                    seed=candidate_seed,
                    negative_prompt=runtime_negative_prompt,
                )
                score_t0 = time.time()
                if compute_runtime_scores:
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
                    identity_score = _score_identity_face_preservation(
                        reference_image=user_img,
                        output_image=candidate_result["image"],
                    )
                    color_fidelity = _score_color_fidelity(
                        output_image=candidate_result["image"],
                        input_palettes=visual_locks.get("color_palettes", []),
                        target_types=product_target_types,
                        input_profiles=visual_locks.get("color_profiles", []),
                    )
                else:
                    fidelity_score, output_desc = 0.0, ""
                    preservation_score = 0.0
                    identity_score = 0.0
                    color_fidelity = {"status": "skipped_runtime_scoring"}
                stage_timings["candidate_scoring_sum_s"] += (time.time() - score_t0)
                
                # Normalize color drift into a 0.0 - 1.0 "fidelity" score
                # DeltaE < 6 is excellent (0.95+), > 18 is poor (< 0.5)
                drift_value = color_fidelity.get("total_drift", None)
                lightness_value = color_fidelity.get("total_lightness_drift", None)
                profile_value = color_fidelity.get("total_profile_drift", None)
                raw_drift: Optional[float]
                if isinstance(drift_value, (int, float)) and np.isfinite(float(drift_value)):
                    raw_drift = float(drift_value)
                    drift_score = max(0.0, min(1.0, 1.0 - (raw_drift / 40.0)))
                    lightness_score = None
                    if isinstance(lightness_value, (int, float)) and np.isfinite(float(lightness_value)):
                        # L* drift in [0,100]; <6 is very good, >20 is poor.
                        lightness_score = max(0.0, min(1.0, 1.0 - (float(lightness_value) / 25.0)))
                    profile_score = None
                    if isinstance(profile_value, (int, float)) and np.isfinite(float(profile_value)):
                        # Profile drift mixes L* + chroma inflation; <4 good, >16 poor.
                        profile_score = max(0.0, min(1.0, 1.0 - (float(profile_value) / 20.0)))

                    if lightness_score is not None and profile_score is not None:
                        color_score = (0.60 * drift_score) + (0.25 * lightness_score) + (0.15 * profile_score)
                    elif lightness_score is not None:
                        color_score = (0.75 * drift_score) + (0.25 * lightness_score)
                    else:
                        color_score = drift_score
                else:
                    raw_drift = None
                    color_score = 0.0

                candidate_runs.append({
                    "label": label,
                    "steps": candidate_steps,
                    "seed": candidate_seed,
                    "latency": float(candidate_result["latency"]),
                    "fidelity_score": float(fidelity_score),
                    "preservation_score": float(preservation_score),
                    "identity_score": float(identity_score),
                    "color_fidelity_score": float(color_score),
                    "color_drift": raw_drift,
                    "color_details": color_fidelity,
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
                + " Keep exact same face identity and skin-tone continuity across face, neck, arms, and hands from image 1."
                + " Never alter facial proportions, expression, or hairstyle."
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
                + " Keep exact same face identity and skin-tone continuity across visible skin regions."
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
                identity = float(run.get("identity_score", 0.0))
                cfid = float(run.get("color_fidelity_score", 0.0))
                neutral_tone_lock = bool(visual_locks.get("neutral_tone_lock"))
                
                # Composite Score: Fidelity (Structure) + Preserve (Background) + Color
                if is_single_top_or_bottom:
                    if neutral_tone_lock:
                        base_score = (0.22 * fid) + (0.22 * preserve) + (0.38 * cfid) + (0.18 * identity)
                    else:
                        base_score = (0.33 * fid) + (0.33 * preserve) + (0.14 * cfid) + (0.20 * identity)
                elif neutral_tone_lock:
                    base_score = (0.40 * fid) + (0.35 * cfid) + (0.25 * identity)
                else:
                    base_score = (0.50 * fid) + (0.25 * cfid) + (0.25 * identity)

                if not FLUX2_COLOR_FIRST_GATE_ENABLED:
                    return base_score

                valid_drifts = [
                    float(c.get("color_drift"))
                    for c in candidate_runs
                    if isinstance(c.get("color_drift"), (int, float))
                    and np.isfinite(float(c.get("color_drift")))
                ]
                if not valid_drifts:
                    return base_score

                run_drift = run.get("color_drift")
                if not isinstance(run_drift, (int, float)) or not np.isfinite(float(run_drift)):
                    return -1e6

                best_drift = min(valid_drifts)
                if float(run_drift) > (best_drift + float(FLUX2_COLOR_FIRST_GATE_MAX_DRIFT_DELTA)):
                    return -1e6
                return base_score

            selected_prompt = prompt
            selected_candidate_index = 0
            mode = FLUX2_SINGLE_CANDIDATE_MODE
            runtime_scoring_enabled = bool(FLUX2_RUNTIME_SCORING_ENABLED)
            if mode == "auto":
                # Auto multi-candidate flows rely on these scores to rank candidates.
                runtime_scoring_enabled = True
            elif not FLUX2_FORCE_RUNTIME_SCORING_FOR_SINGLE_CANDIDATE:
                # For explicit single-candidate modes, scoring does not affect output selection.
                runtime_scoring_enabled = False
            if request_disable_neutral_calibration:
                neutral_color_calibration: Dict[str, object] = {"applied": False, "reason": "disabled_by_request"}
            elif not FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED:
                neutral_color_calibration = {"applied": False, "reason": "disabled_by_env"}
            else:
                neutral_color_calibration = {"applied": False, "reason": "not_run"}
            outer_region_restore: Dict[str, object] = {"applied": False, "reason": "not_run"}

            # 6. Inference (Flux 2.0): either single selected candidate or auto multi-candidate
            if mode == "base":
                result = _run_candidate(
                    candidate_prompt=prompt,
                    candidate_steps=request.steps,
                    candidate_seed=request.seed,
                    label="base",
                    compute_runtime_scores=runtime_scoring_enabled,
                )
            elif mode == "dress_strict" and is_single_dress:
                result = _run_candidate(
                    candidate_prompt=dress_strict_prompt,
                    candidate_steps=dress_strict_steps,
                    candidate_seed=dress_strict_seed,
                    label="dress_strict",
                    compute_runtime_scores=runtime_scoring_enabled,
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
                    compute_runtime_scores=runtime_scoring_enabled,
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

                if FLUX2_COLOR_GUARD_RERUN_ENABLED and candidate_runs:
                    best_run = candidate_runs[selected_candidate_index]
                    best_drift = best_run.get("color_drift")
                    drift_is_valid = isinstance(best_drift, (int, float)) and np.isfinite(float(best_drift))
                    if drift_is_valid and float(best_drift) > float(FLUX2_COLOR_GUARD_DRIFT_THRESHOLD):
                        color_guard_clause = _build_flux2_hex_color_guard_clause(
                            visual_locks.get("color_palette_metrics", []),
                        )
                        if color_guard_clause:
                            color_guard_prompt = f"{selected_prompt} {color_guard_clause}".strip()
                            best_steps = int(best_run.get("steps", request.steps))
                            color_guard_steps = min(
                                FLUX2_COLOR_GUARD_RERUN_MAX_STEPS,
                                max(request.steps, best_steps + FLUX2_COLOR_GUARD_RERUN_EXTRA_STEPS),
                            )
                            color_guard_seed = min(2147483647, request.seed + 31)
                            color_guard_result = _run_candidate(
                                candidate_prompt=color_guard_prompt,
                                candidate_steps=color_guard_steps,
                                candidate_seed=color_guard_seed,
                                label="color_guard_rerun",
                            )
                            color_guard_index = len(candidate_runs) - 1
                            current_best_score = _candidate_rank(selected_candidate_index)
                            color_guard_score = _candidate_rank(color_guard_index)
                            if color_guard_score >= current_best_score:
                                result = color_guard_result
                                selected_candidate_index = color_guard_index
                                selected_prompt = color_guard_prompt

                if (
                    FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED
                    and (not request_disable_neutral_calibration)
                    and board_mode == "single"
                    and len(product_target_types) == 1
                    and bool(visual_locks.get("neutral_tone_lock"))
                ):
                    src_profile = (
                        visual_locks.get("color_profiles", [{}])[0]
                        if isinstance(visual_locks.get("color_profiles"), list) and visual_locks.get("color_profiles")
                        else {}
                    )
                    calibrated_img, calib_meta = _apply_neutral_post_color_calibration(
                        output_image=result["image"],
                        source_profile=(src_profile if isinstance(src_profile, dict) else {}),
                        target_type=product_target_types[0],
                    )
                    neutral_color_calibration = dict(calib_meta or {})
                    if bool(calib_meta.get("applied")):
                        post_cf = _score_color_fidelity(
                            output_image=calibrated_img,
                            input_palettes=visual_locks.get("color_palettes", []),
                            target_types=product_target_types,
                            input_profiles=visual_locks.get("color_profiles", []),
                        )
                        pre_run = candidate_runs[selected_candidate_index] if selected_candidate_index < len(candidate_runs) else {}
                        pre_drift_val = pre_run.get("color_drift")
                        post_drift_val = post_cf.get("total_drift")
                        pre_ok = isinstance(pre_drift_val, (int, float)) and np.isfinite(float(pre_drift_val))
                        post_ok = isinstance(post_drift_val, (int, float)) and np.isfinite(float(post_drift_val))

                        brightness_acceptable = True
                        brightness_details: Dict[str, object] = {}
                        src_median_l = src_profile.get("medianL") if isinstance(src_profile, dict) else None
                        src_p10_l = src_profile.get("p10L") if isinstance(src_profile, dict) else None
                        src_p90_l = src_profile.get("p90L") if isinstance(src_profile, dict) else None
                        post_output_l = neutral_color_calibration.get("postMedianL")
                        if (
                            isinstance(src_median_l, (int, float))
                            and isinstance(src_p10_l, (int, float))
                            and isinstance(src_p90_l, (int, float))
                            and isinstance(post_output_l, (int, float))
                        ):
                            acceptable_min = max(
                                0.0,
                                float(src_p10_l) - float(FLUX2_NEUTRAL_CALIBRATION_BRIGHTNESS_MARGIN_L),
                            )
                            acceptable_max = min(
                                100.0,
                                float(src_p90_l) + float(FLUX2_NEUTRAL_CALIBRATION_BRIGHTNESS_MARGIN_L),
                            )
                            if not (acceptable_min <= float(post_output_l) <= acceptable_max):
                                brightness_acceptable = False
                                brightness_details["reason"] = "outside_acceptable_range"
                            if (
                                float(src_median_l) > float(FLUX2_NEUTRAL_CALIBRATION_LIGHT_SOURCE_MIN_L)
                                and float(post_output_l)
                                < (float(src_median_l) - float(FLUX2_NEUTRAL_CALIBRATION_LIGHT_SOURCE_MAX_DARKEN))
                            ):
                                brightness_acceptable = False
                                brightness_details["reason"] = "light_garment_darkened_too_much"
                            if (
                                float(src_median_l) < float(FLUX2_NEUTRAL_CALIBRATION_DARK_SOURCE_MAX_L)
                                and float(post_output_l)
                                > (float(src_median_l) + float(FLUX2_NEUTRAL_CALIBRATION_DARK_SOURCE_MAX_BRIGHTEN))
                            ):
                                brightness_acceptable = False
                                brightness_details["reason"] = "dark_garment_brightened_too_much"
                            brightness_details.update(
                                {
                                    "sourceMedianL": round(float(src_median_l), 2),
                                    "sourceP10L": round(float(src_p10_l), 2),
                                    "sourceP90L": round(float(src_p90_l), 2),
                                    "postOutputL": round(float(post_output_l), 2),
                                    "acceptableRange": [round(acceptable_min, 2), round(acceptable_max, 2)],
                                }
                            )
                        else:
                            brightness_acceptable = False
                            brightness_details["reason"] = "cannot_verify_brightness"
                            if isinstance(post_output_l, (int, float)):
                                brightness_details["postOutputL"] = round(float(post_output_l), 2)

                        drift_improved = post_ok and ((not pre_ok) or (float(post_drift_val) <= float(pre_drift_val)))
                        if drift_improved and brightness_acceptable:
                            result["image"] = calibrated_img
                            if selected_candidate_index < len(candidate_runs):
                                drift_score = max(0.0, min(1.0, 1.0 - (float(post_drift_val) / 40.0)))
                                post_lightness = post_cf.get("total_lightness_drift")
                                post_profile = post_cf.get("total_profile_drift")
                                lightness_score = (
                                    max(0.0, min(1.0, 1.0 - (float(post_lightness) / 25.0)))
                                    if isinstance(post_lightness, (int, float)) and np.isfinite(float(post_lightness))
                                    else None
                                )
                                profile_score = (
                                    max(0.0, min(1.0, 1.0 - (float(post_profile) / 20.0)))
                                    if isinstance(post_profile, (int, float)) and np.isfinite(float(post_profile))
                                    else None
                                )
                                if lightness_score is not None and profile_score is not None:
                                    new_color_score = (0.60 * drift_score) + (0.25 * lightness_score) + (0.15 * profile_score)
                                elif lightness_score is not None:
                                    new_color_score = (0.75 * drift_score) + (0.25 * lightness_score)
                                else:
                                    new_color_score = drift_score
                                candidate_runs[selected_candidate_index]["color_details"] = post_cf
                                candidate_runs[selected_candidate_index]["color_drift"] = float(post_drift_val)
                                candidate_runs[selected_candidate_index]["color_fidelity_score"] = float(new_color_score)
                                candidate_runs[selected_candidate_index]["label"] = (
                                    str(candidate_runs[selected_candidate_index].get("label", "selected"))
                                    + "+neutral_color_calibrated"
                                )
                            neutral_color_calibration["accepted"] = True
                            neutral_color_calibration["postDrift"] = float(post_drift_val)
                            neutral_color_calibration["brightnessCheck"] = "passed"
                            neutral_color_calibration["brightnessDetails"] = brightness_details
                        else:
                            neutral_color_calibration["accepted"] = False
                            neutral_color_calibration["brightnessCheck"] = (
                                "failed" if not brightness_acceptable else "passed"
                            )
                            neutral_color_calibration["brightnessDetails"] = brightness_details
                            neutral_color_calibration["reason"] = (
                                str(brightness_details.get("reason") or "brightness_guard_failed")
                                if not brightness_acceptable
                                else "no_drift_improvement"
                            )

                t_outer_restore = time.time()
                restored_image, outer_region_restore = _restore_outer_lower_body_from_reference(
                    reference_image=user_img,
                    output_image=result["image"],
                    target_types=product_target_types,
                    product_descriptions=product_descriptions,
                )
                result["image"] = restored_image
                stage_timings["outer_restore_s"] = round(time.time() - t_outer_restore, 4)

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
            f"desc_wall={stage_timings['descriptor_wall_s']}s "
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
            "productGarmentMetadata": product_garment_metadata,
            "userPromptDescription": user_prompt_description,
            "productColorHints": visual_locks.get("color_hints", []),
            "productColorPalettes": visual_locks.get("color_palettes", []),
            "productColorPaletteMetrics": visual_locks.get("color_palette_metrics", []),
            "productColorProfiles": visual_locks.get("color_profiles", []),
            "productDetailHints": visual_locks.get("detail_terms", []),
            "neutralColorCalibration": neutral_color_calibration,
            "outerLowerBodyRestore": outer_region_restore,
            "neutralColorCalibrationEnabled": bool(
                FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED and (not request_disable_neutral_calibration)
            ),
            "colorFirstGateEnabled": bool(FLUX2_COLOR_FIRST_GATE_ENABLED),
            "colorFirstGateMaxDriftDelta": float(FLUX2_COLOR_FIRST_GATE_MAX_DRIFT_DELTA),
            "colorDecontaminationEnabled": bool(FLUX2_COLOR_DECONTAMINATION_ENABLED),
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
                "productGarmentMetadata": product_garment_metadata,
                "generatedProductPromptIndices": generated_product_prompt_indices,
                "userPromptGenerated": user_prompt_generated,
                "singleCandidateMode": FLUX2_SINGLE_CANDIDATE_MODE,
                "singleCandidateApplied": len(candidate_runs) == 1,
                "selectedCandidateIndex": selected_candidate_index,
                "descriptionCompareEnabled": descriptor_compare_enabled,
                "descriptionComparisons": descriptor_comparisons,
                "collageItemMapping": collage_item_clause,
                "transparencyLockApplied": bool(visual_locks.get("transparency_lock")),
                "productColorPalettes": visual_locks.get("color_palettes", []),
                "productColorPaletteMetrics": visual_locks.get("color_palette_metrics", []),
                "productColorProfiles": visual_locks.get("color_profiles", []),
                "neutralColorCalibration": neutral_color_calibration,
                "outerLowerBodyRestore": outer_region_restore,
                "neutralColorCalibrationEnabled": bool(
                    FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED and (not request_disable_neutral_calibration)
                ),
                "colorFirstGateEnabled": bool(FLUX2_COLOR_FIRST_GATE_ENABLED),
                "colorFirstGateMaxDriftDelta": float(FLUX2_COLOR_FIRST_GATE_MAX_DRIFT_DELTA),
                "colorDecontaminationEnabled": bool(FLUX2_COLOR_DECONTAMINATION_ENABLED),
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
        "startup_preload": {
            "mode": _STARTUP_PRELOAD_STATE["mode"],
            "started": bool(_STARTUP_PRELOAD_STATE["started"]),
            "running": bool(_STARTUP_PRELOAD_STATE["running"]),
            "completed": bool(_STARTUP_PRELOAD_STATE["completed"]),
            "duration_s": _STARTUP_PRELOAD_STATE["duration_s"],
            "errors": list(_STARTUP_PRELOAD_STATE["errors"]),
        },
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
                "flux2_single_garment_extract_default_backend": FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND,
                "flux2_single_garment_extract_default_steps": FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS,
                "flux2_single_garment_extract_default_seed": FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED,
                "flux2_single_garment_extract_upload_debug": FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_DEBUG,
                "flux2_single_garment_extract_disable_parser": FLUX2_SINGLE_GARMENT_EXTRACT_DISABLE_PARSER,
                "flux2_single_garment_extract_upload_raw_debug": FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_RAW_DEBUG,
                "flux2_single_garment_extract_expose_intermediate_urls": FLUX2_SINGLE_GARMENT_EXTRACT_EXPOSE_INTERMEDIATE_URLS,
                "flux2_color_lock_enabled": FLUX2_COLOR_LOCK_ENABLED,
                "flux2_color_lock_top_k": FLUX2_COLOR_LOCK_TOP_K,
                "flux2_color_decontamination_enabled": FLUX2_COLOR_DECONTAMINATION_ENABLED,
                "flux2_color_decontam_alpha_high": FLUX2_COLOR_DECONTAM_ALPHA_HIGH,
                "flux2_color_decontam_alpha_low": FLUX2_COLOR_DECONTAM_ALPHA_LOW,
                "flux2_color_decontam_erode_iters": FLUX2_COLOR_DECONTAM_ERODE_ITERS,
                "flux2_color_decontam_min_pixels": FLUX2_COLOR_DECONTAM_MIN_PIXELS,
                "flux2_color_decontam_min_coverage_ratio": FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO,
                "flux2_color_palette_min_area_percent": FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT,
                "flux2_color_profile_trim_dark_percentile": FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE,
                "flux2_color_profile_trim_bright_percentile": FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE,
                "flux2_color_first_gate_enabled": FLUX2_COLOR_FIRST_GATE_ENABLED,
                "flux2_color_first_gate_max_drift_delta": FLUX2_COLOR_FIRST_GATE_MAX_DRIFT_DELTA,
                "flux2_neutral_post_color_calibration_enabled": FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED,
                "flux2_neutral_calibration_max_delta_l": FLUX2_NEUTRAL_CALIBRATION_MAX_DELTA_L,
                "flux2_neutral_calibration_light_output_min_l": FLUX2_NEUTRAL_CALIBRATION_LIGHT_OUTPUT_MIN_L,
                "flux2_neutral_calibration_max_darken_light_output": FLUX2_NEUTRAL_CALIBRATION_MAX_DARKEN_LIGHT_OUTPUT,
                "flux2_neutral_calibration_dark_output_max_l": FLUX2_NEUTRAL_CALIBRATION_DARK_OUTPUT_MAX_L,
                "flux2_neutral_calibration_max_brighten_dark_output": FLUX2_NEUTRAL_CALIBRATION_MAX_BRIGHTEN_DARK_OUTPUT,
                "flux2_neutral_calibration_brightness_margin_l": FLUX2_NEUTRAL_CALIBRATION_BRIGHTNESS_MARGIN_L,
                "flux2_detail_lock_enabled": FLUX2_DETAIL_LOCK_ENABLED,
                "flux2_allow_qwen_backend": FLUX2_ALLOW_QWEN_BACKEND,
                "flux2_negative_prompt_enable": FLUX2_NEGATIVE_PROMPT_ENABLE,
                "flux2_negative_prompt_default_len": len(FLUX2_NEGATIVE_PROMPT_DEFAULT),
                "flux2_negative_prompt_runtime_mode": FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE,
                "flux2_low_latency_mode": FLUX2_LOW_LATENCY_MODE,
                "flux2_runtime_scoring_enabled": FLUX2_RUNTIME_SCORING_ENABLED,
                "flux2_force_runtime_scoring_for_single_candidate": FLUX2_FORCE_RUNTIME_SCORING_FOR_SINGLE_CANDIDATE,
                "minicpm_service_url": MINICPM_SERVICE_URL,
                "analyze_minicpm_service_url": ANALYZE_MINICPM_SERVICE_URL or MINICPM_SERVICE_URL,
                "analyze_minicpm_service_url_override_enabled": bool(ANALYZE_MINICPM_SERVICE_URL),
                "minicpm_service_timeout_s": MINICPM_SERVICE_TIMEOUT_S,
                "minicpm_service_connect_timeout_s": MINICPM_SERVICE_CONNECT_TIMEOUT_S,
                "minicpm_service_cache_enabled": MINICPM_SERVICE_CACHE_ENABLED,
                "minicpm_service_cache_ttl_s": MINICPM_SERVICE_CACHE_TTL_SECONDS,
                "minicpm_service_cache_max_entries": MINICPM_SERVICE_CACHE_MAX_ENTRIES,
                "minicpm_service_cache_current_entries": len(_MINICPM_DESCRIPTOR_CACHE),
                "minicpm_service_pool_maxsize": MINICPM_SERVICE_POOL_MAXSIZE,
                "minicpm_service_local_file_first": MINICPM_SERVICE_LOCAL_FILE_FIRST,
                "minicpm_service_garment_min_words": MINICPM_SERVICE_GARMENT_MIN_WORDS,
                "analyze_extract_cloth": ANALYZE_EXTRACT_CLOTH,
                "analyze_use_parser_post_extract": ANALYZE_USE_PARSER_POST_EXTRACT,
                "analyze_flux_disable_lora": ANALYZE_FLUX_DISABLE_LORA,
                "flux2_share_base_runner": FLUX2_SHARE_BASE_RUNNER,
                "analyze_preload_flux_runner": ANALYZE_PRELOAD_FLUX_RUNNER,
                "analyze_extract_mode": "flux2_single_garment_extract",
                "analyze_primary_type_with_florence": ANALYZE_PRIMARY_TYPE_WITH_FLORENCE,
                "analyze_preload_florence": ANALYZE_PRELOAD_FLORENCE,
                "analyze_selection_preview_format": ANALYZE_SELECTION_PREVIEW_FORMAT,
                "analyze_selection_preview_max_side": ANALYZE_SELECTION_PREVIEW_MAX_SIDE,
                "analyze_selection_preview_jpeg_quality": ANALYZE_SELECTION_PREVIEW_JPEG_QUALITY,
                "analyze_gpu_queue_timeout_s": ANALYZE_GPU_QUEUE_TIMEOUT_S,
                "analyze_extract_parser_only": ANALYZE_EXTRACT_PARSER_ONLY,
                "analyze_require_extracted_prompt": ANALYZE_REQUIRE_EXTRACTED_PROMPT,
                "analyze_extract_force_bbox_crop": ANALYZE_EXTRACT_FORCE_BBOX_CROP,
                "analyze_extract_crop_pad_ratio": ANALYZE_EXTRACT_CROP_PAD_RATIO,
                "analyze_extract_crop_pad_ratio_dress": ANALYZE_EXTRACT_CROP_PAD_RATIO_DRESS,
                "analyze_extract_crop_bottom_extra_ratio_dress": ANALYZE_EXTRACT_CROP_BOTTOM_EXTRA_RATIO_DRESS,
                "analyze_extract_crop_top_extra_ratio_bottom": ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM,
                "analyze_extract_crop_top_extra_ratio_bottom_multi": ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI,
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
                "analyze_top_skin_rim_cleanup": ANALYZE_TOP_SKIN_RIM_CLEANUP,
                "analyze_top_skin_rim_max_ratio": ANALYZE_TOP_SKIN_RIM_MAX_RATIO,
                "analyze_garment_postprocess_enabled": ANALYZE_GARMENT_POSTPROCESS_ENABLED,
                "analyze_garment_target_aspect": f"{ANALYZE_GARMENT_TARGET_ASPECT_W}:{ANALYZE_GARMENT_TARGET_ASPECT_H}",
                "analyze_garment_enhance_enabled": ANALYZE_GARMENT_ENHANCE_ENABLED,
                "analyze_garment_output_background": ANALYZE_GARMENT_OUTPUT_BACKGROUND,
                "analyze_bg_removal_backend": ANALYZE_BG_REMOVAL_BACKEND,
                "analyze_birefnet_model_id": ANALYZE_BIREFNET_MODEL_ID,
                "wardrobe_progress_sync_enabled": ENABLE_WARDROBE_PROGRESS_SYNC,
                "wardrobe_progress_include_input_image": WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE,
        },
        "models": engine.model_status(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
