"""
Glamify AI Engine - Main Application Entry Point

This is the refactored main.py file that serves as the application entry point.
All business logic has been extracted to services, utilities to utils modules,
and configuration to config modules.

The original 14,257-line monolithic file has been reduced to ~300 lines.
"""

import asyncio
import logging
import os
import re
from dataclasses import replace
from typing import Optional, List, Dict, Tuple

import numpy as np
from PIL import Image

from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables before importing modules
load_dotenv()

# Configuration and Services
from config import get_config
from services import AIEngine, AnalyzeService, TryonService, GarmentExtractionService, UserImageService

# Route handlers
from routes import (
    tryon_router, 
    analyze_router, 
    extract_router, 
    user_prep_router, 
    health_router
)

# Shared utilities
from shared.azure_storage import storage
from shared.image_ops import binary_open, binary_close, rgb_to_lab

from core.garment_color_context import (
    GarmentColorContextSettings,
    build_single_image_color_context as _shared_build_single_image_color_context,
    build_visual_lock_clauses as _shared_build_visual_lock_clauses,
)

from utils import color_processing as color_processing_mod
from utils.prompt_generation import (
    parse_structured_descriptor as _parse_structured_descriptor,
    parse_garment_prompt_sections as _parse_garment_prompt_sections,
    build_florence_contamination_avoid_clause as _build_florence_contamination_avoid_clause,
    extract_generation_only_avoid_directives as _extract_generation_only_avoid_directives,
    merge_avoid_clause_sentences as _merge_avoid_clause_sentences,
    extract_prompt_fact_segments as _extract_prompt_fact_segments,
    serialize_prompt_fact_segments as _serialize_prompt_fact_segments,
)
from utils.validation import (
    normalize_garment_type as _normalize_garment_type,
    sanitize_prompt_fact_value as _sanitize_prompt_fact_value,
    descriptor_is_weak as _descriptor_is_weak_base,
    descriptor_word_count as _descriptor_word_count_base,
    canonical_coverage_for_type as _canonical_coverage_for_type,
)
from utils.color_processing import (
    canonical_color_token as _canonical_color_token,
    color_family as _color_family,
    is_neutral_color_token as _is_neutral_color_token,
    palette_weighted_hue_deg as _palette_weighted_hue_deg,
    palette_hue_from_hexes as _palette_hue_from_hexes,
    filter_palette_entries_by_area as _filter_palette_entries_by_area,
    bucket_color_brightness as _bucket_color_brightness,
    bucket_color_saturation as _bucket_color_saturation,
    bucket_color_undertone as _bucket_color_undertone,
    hex_to_rgb_triplet as _hex_to_rgb_triplet,
)
from utils.image_preprocessing import (
    mask_connected_components as _mask_connected_components,
    bbox_from_mask as _bbox_from_mask,
    bbox_x_overlap_ratio as _bbox_x_overlap_ratio,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("glamify-ai")

# Initialize FastAPI application
app = FastAPI(
    title="Glamify AI Engine",
    description="Unified API for Wardrobe Digitization and Virtual Try-On",
    version="2.0.0"
)

# Initialize configuration and services
config = get_config()
engine = AIEngine(config)

# Initialize service layer with dependency injection
analyze_service = AnalyzeService(engine, config.analyze)
tryon_service = TryonService(engine, config.flux2)
garment_extraction_service = GarmentExtractionService(engine, config.flux2)
user_image_service = UserImageService(engine, config)

# GPU concurrency semaphore (preserved from original)
gpu_semaphore = asyncio.Semaphore(config.app.gpu_concurrency)

# Register route handlers
app.include_router(health_router, tags=["health"])
app.include_router(tryon_router, tags=["tryon"])
app.include_router(analyze_router, tags=["analyze"])
app.include_router(extract_router, tags=["extract"])
app.include_router(user_prep_router, tags=["user-prep"])

# Dependency injection helpers
def get_tryon_service() -> TryonService:
    """Get TryonService instance for dependency injection."""
    return tryon_service

def get_analyze_service() -> AnalyzeService:
    """Get AnalyzeService instance for dependency injection."""
    return analyze_service

def get_extract_service() -> GarmentExtractionService:
    """Get GarmentExtractionService instance for dependency injection."""
    return garment_extraction_service

def get_user_prep_service() -> UserImageService:
    """Get UserImageService instance for dependency injection."""
    return user_image_service

def get_ai_engine() -> AIEngine:
    """Get AIEngine instance for dependency injection."""
    return engine

def get_config_instance():
    """Get configuration instance for dependency injection."""
    return config

# Application startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Glamify AI Engine...")
    
    # Preload models if configured
    if config.app.startup_background_preload:
        logger.info("Starting background model preloading...")
        # Preload analyze models
        if hasattr(engine, 'ensure_analyze_ready'):
            engine.ensure_analyze_ready()
        # Preload VTO models  
        if hasattr(engine, 'ensure_vto_ready'):
            engine.ensure_vto_ready()
    
    logger.info("Glamify AI Engine started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Glamify AI Engine...")
    # Add any cleanup logic here if needed
    logger.info("Glamify AI Engine shut down successfully")

# Health check endpoint (basic)
@app.get("/")
async def root():
    """Root endpoint for basic health check."""
    return {
        "message": "Glamify AI Engine",
        "version": "2.0.0",
        "status": "running"
    }

# Store services in app state for access in routes
app.state.tryon_service = tryon_service
app.state.analyze_service = analyze_service
app.state.garment_extraction_service = garment_extraction_service
app.state.user_image_service = user_image_service
app.state.ai_engine = engine
app.state.config = config
app.state.gpu_semaphore = gpu_semaphore

# ---------------------------------------------------------------------------
# Legacy compatibility helpers (tests and refactor shims)
# ---------------------------------------------------------------------------

_RAW_MINICPM_SERVICE_URL = os.getenv("MINICPM_SERVICE_URL", "").strip().rstrip("/")
_RAW_ANALYZE_MINICPM_SERVICE_URL = os.getenv("ANALYZE_MINICPM_SERVICE_URL", "").strip().rstrip("/")
_MINICPM_SERVICE_URL_EXPLICIT = bool(_RAW_MINICPM_SERVICE_URL)
_ANALYZE_MINICPM_SERVICE_URL_EXPLICIT = bool(_RAW_ANALYZE_MINICPM_SERVICE_URL)

MINICPM_SERVICE_URL = config.minicpm.service_url
ANALYZE_MINICPM_SERVICE_URL = config.minicpm.analyze_service_url or MINICPM_SERVICE_URL
MINICPM_SERVICE_GARMENT_MIN_WORDS = int(config.minicpm.garment_min_words)

FLUX2_DESCRIPTOR_BACKEND = str(config.flux2.descriptor_backend)
FLUX2_ALLOW_QWEN_BACKEND = bool(config.flux2.allow_qwen_backend)
FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE = str(config.flux2.negative_prompt_runtime_mode)
FLUX2_TRYON_DISABLE_RUNTIME_NEGATIVE_PROMPT = bool(config.flux2.tryon_disable_runtime_negative_prompt)

FLUX2_COLOR_LOCK_TOP_K = int(config.flux2.color_lock_top_k)
FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT = float(config.flux2.color_palette_min_area_percent)
FLUX2_COLOR_DECONTAMINATION_ENABLED = bool(config.flux2.color_decontamination_enabled)
FLUX2_COLOR_DECONTAM_ALPHA_HIGH = int(config.flux2.color_decontam_alpha_high)
FLUX2_COLOR_DECONTAM_ALPHA_LOW = int(config.flux2.color_decontam_alpha_low)
FLUX2_COLOR_DECONTAM_ERODE_ITERS = int(config.flux2.color_decontam_erode_iters)
FLUX2_COLOR_DECONTAM_MIN_PIXELS = int(config.flux2.color_decontam_min_pixels)
FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO = float(config.flux2.color_decontam_min_coverage_ratio)
FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE = float(config.flux2.color_profile_trim_dark_percentile)
FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE = float(config.flux2.color_profile_trim_bright_percentile)
FLUX2_DETAIL_LOCK_ENABLED = bool(config.flux2.detail_lock_enabled)

FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE = int(config.flux2.minicpm_product_caption_max_side)
FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE = int(config.flux2.minicpm_product_caption_min_side)
FLUX2_MINICPM_USER_CAPTION_MAX_SIDE = int(config.flux2.minicpm_user_caption_max_side)
FLUX2_MINICPM_USER_CAPTION_MIN_SIDE = int(config.flux2.minicpm_user_caption_min_side)
FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE = int(config.flux2.qwen_product_caption_max_side)
FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE = int(config.flux2.qwen_product_caption_min_side)

COLOR_CONTEXT_DISABLE_MASKING = bool(config.color.context_disable_masking)
GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = bool(config.color.garment_semantic_override_enabled)

ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE = float(config.analyze.fashion_basecolour_apply_min_score)
ANALYZE_COLOR_PARSER_SAMPLING_TRIAL_ENABLED = bool(config.analyze.color_parser_sampling_trial_enabled)

from utils import legacy_compat as _legacy_compat


def _sync_legacy_compat() -> None:
    _legacy_compat._MINICPM_SERVICE_URL_EXPLICIT = _MINICPM_SERVICE_URL_EXPLICIT
    _legacy_compat._ANALYZE_MINICPM_SERVICE_URL_EXPLICIT = _ANALYZE_MINICPM_SERVICE_URL_EXPLICIT
    _legacy_compat.MINICPM_SERVICE_URL = MINICPM_SERVICE_URL
    _legacy_compat.ANALYZE_MINICPM_SERVICE_URL = ANALYZE_MINICPM_SERVICE_URL
    _legacy_compat.MINICPM_SERVICE_GARMENT_MIN_WORDS = MINICPM_SERVICE_GARMENT_MIN_WORDS

    _legacy_compat.FLUX2_DESCRIPTOR_BACKEND = FLUX2_DESCRIPTOR_BACKEND
    _legacy_compat.FLUX2_ALLOW_QWEN_BACKEND = FLUX2_ALLOW_QWEN_BACKEND
    _legacy_compat.FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE = FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE
    _legacy_compat.FLUX2_TRYON_DISABLE_RUNTIME_NEGATIVE_PROMPT = FLUX2_TRYON_DISABLE_RUNTIME_NEGATIVE_PROMPT

    _legacy_compat.FLUX2_COLOR_LOCK_TOP_K = FLUX2_COLOR_LOCK_TOP_K
    _legacy_compat.FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT = FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT
    _legacy_compat.FLUX2_COLOR_DECONTAMINATION_ENABLED = FLUX2_COLOR_DECONTAMINATION_ENABLED
    _legacy_compat.FLUX2_COLOR_DECONTAM_ALPHA_HIGH = FLUX2_COLOR_DECONTAM_ALPHA_HIGH
    _legacy_compat.FLUX2_COLOR_DECONTAM_ALPHA_LOW = FLUX2_COLOR_DECONTAM_ALPHA_LOW
    _legacy_compat.FLUX2_COLOR_DECONTAM_ERODE_ITERS = FLUX2_COLOR_DECONTAM_ERODE_ITERS
    _legacy_compat.FLUX2_COLOR_DECONTAM_MIN_PIXELS = FLUX2_COLOR_DECONTAM_MIN_PIXELS
    _legacy_compat.FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO = FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO
    _legacy_compat.FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE = FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE
    _legacy_compat.FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE = FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE
    _legacy_compat.FLUX2_DETAIL_LOCK_ENABLED = FLUX2_DETAIL_LOCK_ENABLED

    _legacy_compat.FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE = FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE
    _legacy_compat.FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE = FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE
    _legacy_compat.FLUX2_MINICPM_USER_CAPTION_MAX_SIDE = FLUX2_MINICPM_USER_CAPTION_MAX_SIDE
    _legacy_compat.FLUX2_MINICPM_USER_CAPTION_MIN_SIDE = FLUX2_MINICPM_USER_CAPTION_MIN_SIDE
    _legacy_compat.FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE = FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE
    _legacy_compat.FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE = FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE

    _legacy_compat.COLOR_CONTEXT_DISABLE_MASKING = COLOR_CONTEXT_DISABLE_MASKING
    _legacy_compat.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED

    _legacy_compat.ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE = ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE
    _legacy_compat.ANALYZE_COLOR_PARSER_SAMPLING_TRIAL_ENABLED = ANALYZE_COLOR_PARSER_SAMPLING_TRIAL_ENABLED


def _with_legacy_sync(fn):
    def wrapper(*args, **kwargs):
        _sync_legacy_compat()
        return fn(*args, **kwargs)
    return wrapper

_descriptor_word_count = _with_legacy_sync(_legacy_compat._descriptor_word_count)
_descriptor_is_weak = _with_legacy_sync(_legacy_compat._descriptor_is_weak)
_default_minicpm_family_backend = _with_legacy_sync(_legacy_compat._default_minicpm_family_backend)
_normalize_descriptor_backend = _with_legacy_sync(_legacy_compat._normalize_descriptor_backend)
_normalize_prompt_descriptor_backend = _with_legacy_sync(_legacy_compat._normalize_prompt_descriptor_backend)
_resize_for_qwen_caption = _with_legacy_sync(_legacy_compat._resize_for_qwen_caption)
_minicpm_bundle_has_valid_json_contract = _with_legacy_sync(_legacy_compat._minicpm_bundle_has_valid_json_contract)
_default_extraction_avoid_clause_for_type = _with_legacy_sync(_legacy_compat._default_extraction_avoid_clause_for_type)
_build_minicpm_garment_prompt = _with_legacy_sync(_legacy_compat._build_minicpm_garment_prompt)
_build_minicpm_garment_retry_prompt = _with_legacy_sync(_legacy_compat._build_minicpm_garment_retry_prompt)
_build_minicpm_garment_detail_retry_prompt = _with_legacy_sync(_legacy_compat._build_minicpm_garment_detail_retry_prompt)
_choose_better_garment_prompt_bundle = _with_legacy_sync(_legacy_compat._choose_better_garment_prompt_bundle)
_repair_non_json_garment_prompt_bundle = _with_legacy_sync(_legacy_compat._repair_non_json_garment_prompt_bundle)
_ensure_garment_prompt_bundle_avoid_clause = _with_legacy_sync(_legacy_compat._ensure_garment_prompt_bundle_avoid_clause)
_apply_florence_avoid_clause = _with_legacy_sync(_legacy_compat._apply_florence_avoid_clause)
_build_minicpm_garment_color_prompt = _with_legacy_sync(_legacy_compat._build_minicpm_garment_color_prompt)
_describe_garment_color_terms_with_backend = _with_legacy_sync(_legacy_compat._describe_garment_color_terms_with_backend)
_describe_garment_with_backend = _with_legacy_sync(_legacy_compat._describe_garment_with_backend)
_describe_garment_prompt_bundle_with_backend = _with_legacy_sync(_legacy_compat._describe_garment_prompt_bundle_with_backend)
_identity_only_user_context = _with_legacy_sync(_legacy_compat._identity_only_user_context)
_build_flux2_targeted_prompt = _with_legacy_sync(_legacy_compat._build_flux2_targeted_prompt)
_build_flux2_runtime_negative_prompt = _with_legacy_sync(_legacy_compat._build_flux2_runtime_negative_prompt)
_resolve_tryon_runtime_negative_prompt = _with_legacy_sync(_legacy_compat._resolve_tryon_runtime_negative_prompt)
_extract_text_color_terms = _with_legacy_sync(_legacy_compat._extract_text_color_terms)
_profile_is_near_white = _with_legacy_sync(_legacy_compat._profile_is_near_white)
_palette_hue_from_hexes = _with_legacy_sync(_legacy_compat._palette_hue_from_hexes)
_palette_supports_color_family = _with_legacy_sync(_legacy_compat._palette_supports_color_family)
_augment_pixel_hints_with_muted_hue_family = _with_legacy_sync(_legacy_compat._augment_pixel_hints_with_muted_hue_family)
_nearest_color_label = _with_legacy_sync(_legacy_compat._nearest_color_label)
_color_labels_from_hex_palette = _with_legacy_sync(_legacy_compat._color_labels_from_hex_palette)
_build_garment_color_tone_guidance = _with_legacy_sync(_legacy_compat._build_garment_color_tone_guidance)
_compose_color_descriptor_phrase = _with_legacy_sync(_legacy_compat._compose_color_descriptor_phrase)
_estimate_color_signal_confidence = _with_legacy_sync(_legacy_compat._estimate_color_signal_confidence)
_bucket_color_signal_confidence = _with_legacy_sync(_legacy_compat._bucket_color_signal_confidence)
_build_rich_color_metadata = _with_legacy_sync(_legacy_compat._build_rich_color_metadata)
_sanitize_prompt_fact_fields = _with_legacy_sync(_legacy_compat._sanitize_prompt_fact_fields)
_extract_garment_descriptor_facts = _with_legacy_sync(_legacy_compat._extract_garment_descriptor_facts)
_resolve_garment_color_truth = _with_legacy_sync(_legacy_compat._resolve_garment_color_truth)
_build_garment_metadata = _with_legacy_sync(_legacy_compat._build_garment_metadata)
_extract_garment_metadata_prompt = _with_legacy_sync(_legacy_compat._extract_garment_metadata_prompt)
_extract_garment_metadata_target_type = _with_legacy_sync(_legacy_compat._extract_garment_metadata_target_type)
_extract_garment_metadata_color_payload = _with_legacy_sync(_legacy_compat._extract_garment_metadata_color_payload)
_extract_garment_metadata_color_block = _with_legacy_sync(_legacy_compat._extract_garment_metadata_color_block)
_should_prefer_metadata_color_signal = _with_legacy_sync(_legacy_compat._should_prefer_metadata_color_signal)
_build_metadata_color_descriptor_lock = _with_legacy_sync(_legacy_compat._build_metadata_color_descriptor_lock)
_extract_detail_lock_terms = _with_legacy_sync(_legacy_compat._extract_detail_lock_terms)
_garment_color_context_settings = _with_legacy_sync(_legacy_compat._garment_color_context_settings)
_build_flux2_visual_lock_clauses = _with_legacy_sync(_legacy_compat._build_flux2_visual_lock_clauses)
_build_single_image_color_context = _with_legacy_sync(_legacy_compat._build_single_image_color_context)
_parser_candidate_category_text = _with_legacy_sync(_legacy_compat._parser_candidate_category_text)
_build_flux2_single_garment_extract_prompt = _with_legacy_sync(_legacy_compat._build_flux2_single_garment_extract_prompt)
_build_flux2_single_garment_extract_negative_prompt = _with_legacy_sync(_legacy_compat._build_flux2_single_garment_extract_negative_prompt)
_parser_category_ids = _with_legacy_sync(_legacy_compat._parser_category_ids)
_parser_extraction_keep_ids = _with_legacy_sync(_legacy_compat._parser_extraction_keep_ids)
_parser_strict_mask = _with_legacy_sync(_legacy_compat._parser_strict_mask)
_split_outfit_signature_from_parsing = _with_legacy_sync(_legacy_compat._split_outfit_signature_from_parsing)
_skin_like_mask = _with_legacy_sync(_legacy_compat._skin_like_mask)
_estimate_type_focused_color_mask = _with_legacy_sync(_legacy_compat._estimate_type_focused_color_mask)
_cleanup_color_sampling_mask = _with_legacy_sync(_legacy_compat._cleanup_color_sampling_mask)
_normalize_color_sampling_mask = _with_legacy_sync(_legacy_compat._normalize_color_sampling_mask)
_color_sampling_mask_is_usable = _with_legacy_sync(_legacy_compat._color_sampling_mask_is_usable)
_resolve_color_sampling_mask = _with_legacy_sync(_legacy_compat._resolve_color_sampling_mask)
_restore_outer_lower_body_from_reference = _with_legacy_sync(_legacy_compat._restore_outer_lower_body_from_reference)
_normalize_user_prepare_api_prompt_description = _with_legacy_sync(_legacy_compat._normalize_user_prepare_api_prompt_description)
_normalize_user_prepare_prompt_description = _with_legacy_sync(_legacy_compat._normalize_user_prepare_prompt_description)
_user_prep_has_multiple_prominent_people = _with_legacy_sync(_legacy_compat._user_prep_has_multiple_prominent_people)
_score_user_prep_face_candidate = _with_legacy_sync(_legacy_compat._score_user_prep_face_candidate)
_select_best_user_prep_face_candidate = _with_legacy_sync(_legacy_compat._select_best_user_prep_face_candidate)
_parser_alias_ids = _with_legacy_sync(_legacy_compat._parser_alias_ids)
_score_visible_limb_preservation = _with_legacy_sync(_legacy_compat._score_visible_limb_preservation)
_to_public_item = _with_legacy_sync(_legacy_compat._to_public_item)

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )
