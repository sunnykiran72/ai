"""
Flux2 model configuration.

This module defines configuration for the Flux2 CVTON model including:
- Backend selection (descriptor, fidelity)
- Prompting and negative prompts
- Performance settings
- Second pass refinement
- Model preloading
- Color processing and locking
- Color guard rerun
- Neutral color calibration
- Detail preservation
- Single garment extraction
"""

import os
from typing import Literal
from pydantic import BaseModel, Field
from .base import env_int, env_float, env_bool


def _default_minicpm_family_backend() -> str:
    """
    Determine default MiniCPM backend based on service URL configuration.
    
    Prefer the standalone MiniCPM service when the deployment explicitly configures it.
    This avoids loading the same vision-language model inside the main API process.
    """
    minicpm_service_url = os.getenv("MINICPM_SERVICE_URL", "").strip().rstrip("/")
    analyze_minicpm_service_url = os.getenv("ANALYZE_MINICPM_SERVICE_URL", "").strip().rstrip("/")
    
    if minicpm_service_url or analyze_minicpm_service_url:
        return "minicpm_service"
    return "minicpm"


class Flux2Config(BaseModel):
    """
    Flux2 model configuration.
    
    This configuration manages all settings for the Flux2 CVTON (Controllable Virtual Try-On)
    model including backend selection, prompting strategies, performance optimizations,
    second pass refinement, color processing, and single garment extraction.
    """
    
    # Backend selection
    descriptor_backend: Literal["minicpm", "minicpm_service", "florence", "joycaption"] = Field(
        default="minicpm_service",
        description="Backend for generating garment descriptors"
    )
    fidelity_backend: Literal["florence", "qwen2_5_vl"] = Field(
        default="florence",
        description="Backend for fidelity verification"
    )
    allow_qwen_backend: bool = Field(
        default=False,
        description="Allow Qwen2.5-VL as fidelity backend"
    )
    
    # Prompting
    descriptor_compare: bool = Field(
        default=False,
        description="Enable descriptor comparison across multiple backends"
    )
    negative_prompt_enable: bool = Field(
        default=True,
        description="Enable negative prompts for generation"
    )
    negative_prompt_default: str = Field(
        default=(
            "low quality, blurry, deformed body, extra limbs, extra fingers, wrong hands, "
            "extra hand, third hand, duplicate arms, hand fused to garment, garment fused to skin, "
            "identity change, different face, wrong skin tone, recolored garment, hue shift, "
            "color drift, pattern drift, texture swap, logo/text watermark, duplicate garment, "
            "layering artifacts, garment merge, ghost garment, incorrect neckline, incorrect hemline"
        ),
        description="Default negative prompt text"
    )
    negative_prompt_runtime_mode: Literal["disabled", "request_only", "request_or_auto", "auto_only"] = Field(
        default="request_or_auto",
        description="Runtime negative prompt mode"
    )
    tryon_disable_runtime_negative_prompt: bool = Field(
        default=True,
        description="Disable runtime negative prompt for try-on endpoint"
    )
    
    # Performance
    low_latency_mode: bool = Field(
        default=False,
        description="Enable low latency mode with reduced quality"
    )
    single_candidate_mode: Literal["auto", "base", "dress_strict", "qwen_strict"] = Field(
        default="auto",
        description="Single candidate generation mode"
    )
    runtime_scoring_enabled: bool = Field(
        default=True,
        description="Enable runtime scoring of generated candidates"
    )
    force_runtime_scoring_for_single_candidate: bool = Field(
        default=False,
        description="Force runtime scoring even for single candidate"
    )
    
    # Second pass refinement - Dress
    dress_second_pass_enabled: bool = Field(
        default=True,
        description="Enable second pass refinement for dresses"
    )
    dress_second_pass_extra_steps: int = Field(
        default=4,
        ge=1,
        description="Extra steps for dress second pass"
    )
    dress_second_pass_max_steps: int = Field(
        default=18,
        ge=6,
        description="Maximum steps for dress second pass"
    )
    
    # Second pass refinement - Region lock
    region_lock_second_pass_enabled: bool = Field(
        default=True,
        description="Enable second pass refinement with region locking"
    )
    region_lock_second_pass_extra_steps: int = Field(
        default=2,
        ge=1,
        description="Extra steps for region lock second pass"
    )
    region_lock_second_pass_max_steps: int = Field(
        default=16,
        ge=6,
        description="Maximum steps for region lock second pass"
    )
    
    # Second pass refinement - Qwen
    qwen_second_pass_enabled: bool = Field(
        default=True,
        description="Enable second pass refinement for Qwen backend"
    )
    qwen_second_pass_extra_steps: int = Field(
        default=6,
        ge=1,
        description="Extra steps for Qwen second pass"
    )
    qwen_second_pass_max_steps: int = Field(
        default=24,
        ge=8,
        description="Maximum steps for Qwen second pass"
    )
    qwen_min_steps: int = Field(
        default=10,
        ge=4,
        description="Minimum steps for Qwen generation"
    )
    
    # Model preloading
    preload_qwen_with_flux2: bool = Field(
        default=True,
        description="Preload Qwen2.5-VL model with Flux2"
    )
    preload_joycaption_with_flux2: bool = Field(
        default=True,
        description="Preload JoyCaption model with Flux2"
    )
    preload_minicpm_with_flux2: bool = Field(
        default=False,
        description="Preload MiniCPM model with Flux2"
    )
    unload_qwen_before_flux2: bool = Field(
        default=False,
        description="Unload Qwen before running Flux2 to save memory"
    )
    qwen_extra_classify_pass: bool = Field(
        default=False,
        description="Enable extra classification pass with Qwen"
    )
    
    # Qwen image sizing
    qwen_product_caption_max_side: int = Field(
        default=768,
        ge=384,
        description="Maximum side length for Qwen product captioning"
    )
    qwen_product_caption_min_side: int = Field(
        default=512,
        ge=256,
        description="Minimum side length for Qwen product captioning"
    )
    qwen_user_caption_max_side: int = Field(
        default=768,
        ge=384,
        description="Maximum side length for Qwen user captioning"
    )
    qwen_user_caption_min_side: int = Field(
        default=512,
        ge=256,
        description="Minimum side length for Qwen user captioning"
    )
    
    # MiniCPM image sizing
    minicpm_product_caption_max_side: int = Field(
        default=1024,
        ge=512,
        description="Maximum side length for MiniCPM product captioning"
    )
    minicpm_product_caption_min_side: int = Field(
        default=512,
        ge=256,
        description="Minimum side length for MiniCPM product captioning"
    )
    minicpm_user_caption_max_side: int = Field(
        default=1024,
        ge=512,
        description="Maximum side length for MiniCPM user captioning"
    )
    minicpm_user_caption_min_side: int = Field(
        default=512,
        ge=256,
        description="Minimum side length for MiniCPM user captioning"
    )
    
    # Color processing
    color_lock_enabled: bool = Field(
        default=True,
        description="Enable color locking to preserve garment colors"
    )
    color_lock_top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of top colors to lock"
    )
    color_decontamination_enabled: bool = Field(
        default=True,
        description="Enable color decontamination to remove background colors"
    )
    color_decontam_alpha_high: int = Field(
        default=240,
        ge=1,
        le=255,
        description="High alpha threshold for color decontamination"
    )
    color_decontam_alpha_low: int = Field(
        default=200,
        ge=1,
        le=255,
        description="Low alpha threshold for color decontamination"
    )
    color_decontam_erode_iters: int = Field(
        default=1,
        ge=0,
        description="Erosion iterations for color decontamination"
    )
    color_decontam_min_pixels: int = Field(
        default=64,
        ge=16,
        description="Minimum pixels for color decontamination"
    )
    color_decontam_min_coverage_ratio: float = Field(
        default=0.006,
        ge=0.0,
        le=0.2,
        description="Minimum coverage ratio for color decontamination"
    )
    color_palette_min_area_percent: float = Field(
        default=7.0,
        ge=0.0,
        le=40.0,
        description="Minimum area percentage for color palette extraction"
    )
    color_profile_trim_dark_percentile: float = Field(
        default=10.0,
        ge=0.0,
        le=40.0,
        description="Dark percentile for color profile trimming"
    )
    color_profile_trim_bright_percentile: float = Field(
        default=98.0,
        ge=60.0,
        le=100.0,
        description="Bright percentile for color profile trimming"
    )
    
    # Color guard rerun
    color_guard_rerun_enabled: bool = Field(
        default=True,
        description="Enable color guard rerun on drift detection"
    )
    color_guard_drift_threshold: float = Field(
        default=12.0,
        ge=0.0,
        description="Color drift threshold (Delta E) for triggering rerun"
    )
    color_guard_rerun_extra_steps: int = Field(
        default=2,
        ge=1,
        description="Extra steps for color guard rerun"
    )
    color_guard_rerun_max_steps: int = Field(
        default=18,
        ge=6,
        description="Maximum steps for color guard rerun"
    )
    color_first_gate_enabled: bool = Field(
        default=True,
        description="Enable first gate color check"
    )
    color_first_gate_max_drift_delta: float = Field(
        default=6.0,
        ge=0.0,
        description="Maximum drift delta for first gate"
    )
    
    # Neutral color calibration
    neutral_post_color_calibration_enabled: bool = Field(
        default=True,
        description="Enable post-generation neutral color calibration"
    )
    neutral_calibration_max_delta_l: float = Field(
        default=10.0,
        ge=1.0,
        description="Maximum Delta L for neutral calibration"
    )
    neutral_calibration_light_output_min_l: float = Field(
        default=65.0,
        description="Minimum L value for light output calibration"
    )
    neutral_calibration_max_darken_light_output: float = Field(
        default=5.0,
        ge=0.5,
        description="Maximum darkening for light output"
    )
    neutral_calibration_dark_output_max_l: float = Field(
        default=40.0,
        description="Maximum L value for dark output calibration"
    )
    neutral_calibration_max_brighten_dark_output: float = Field(
        default=15.0,
        ge=0.5,
        description="Maximum brightening for dark output"
    )
    neutral_calibration_brightness_margin_l: float = Field(
        default=15.0,
        ge=0.0,
        description="Brightness margin L for calibration"
    )
    neutral_calibration_light_source_min_l: float = Field(
        default=60.0,
        description="Minimum L value for light source calibration"
    )
    neutral_calibration_light_source_max_darken: float = Field(
        default=15.0,
        ge=0.5,
        description="Maximum darkening for light source"
    )
    neutral_calibration_dark_source_max_l: float = Field(
        default=40.0,
        description="Maximum L value for dark source calibration"
    )
    neutral_calibration_dark_source_max_brighten: float = Field(
        default=20.0,
        ge=0.5,
        description="Maximum brightening for dark source"
    )
    
    # Detail lock
    detail_lock_enabled: bool = Field(
        default=True,
        description="Enable detail locking to preserve garment details"
    )
    
    # Single garment extraction
    single_garment_extract_default_backend: str = Field(
        default="minicpm_service",
        description="Default backend for single garment extraction"
    )
    single_garment_extract_default_steps: int = Field(
        default=10,
        ge=4,
        description="Default steps for single garment extraction"
    )
    single_garment_extract_default_seed: int = Field(
        default=23,
        ge=0,
        description="Default seed for single garment extraction"
    )
    single_garment_extract_upload_debug: bool = Field(
        default=False,
        description="Upload debug images for single garment extraction"
    )
    single_garment_extract_disable_parser: bool = Field(
        default=True,
        description="Disable parser for single garment extraction"
    )
    single_garment_extract_upload_raw_debug: bool = Field(
        default=False,
        description="Upload raw debug images for single garment extraction"
    )
    single_garment_extract_expose_intermediate_urls: bool = Field(
        default=False,
        description="Expose intermediate URLs in extraction response"
    )
    
    # Shared runner
    share_base_runner: bool = Field(
        default=True,
        description="Share base Flux2 runner between endpoints"
    )
    
    @classmethod
    def from_env(cls) -> "Flux2Config":
        """
        Load configuration from environment variables.
        
        Returns:
            Flux2Config instance populated from environment variables
        """
        # Backend selection
        descriptor_backend = os.getenv("FLUX2_DESCRIPTOR_BACKEND", "auto").strip().lower()
        if descriptor_backend == "auto":
            descriptor_backend = _default_minicpm_family_backend()
        elif descriptor_backend not in {"minicpm", "minicpm_service", "florence", "joycaption"}:
            descriptor_backend = _default_minicpm_family_backend()
        
        allow_qwen_backend = env_bool("FLUX2_ALLOW_QWEN_BACKEND", "0")
        
        fidelity_backend = os.getenv("FLUX2_FIDELITY_BACKEND", "florence").strip().lower()
        if fidelity_backend not in {"florence", "qwen2_5_vl"}:
            fidelity_backend = "florence"
        if fidelity_backend == "qwen2_5_vl" and not allow_qwen_backend:
            fidelity_backend = "florence"
        
        # Performance - low latency mode affects defaults
        low_latency_mode = env_bool("FLUX2_LOW_LATENCY_MODE", "0")
        
        # Single candidate mode
        single_candidate_mode = os.getenv("FLUX2_SINGLE_CANDIDATE_MODE", "auto").strip().lower()
        if single_candidate_mode not in {"auto", "base", "dress_strict", "qwen_strict"}:
            single_candidate_mode = "auto"
        if low_latency_mode and "FLUX2_SINGLE_CANDIDATE_MODE" not in os.environ:
            single_candidate_mode = "base"
        
        # Second pass settings - affected by low latency mode
        dress_second_pass_enabled = env_bool(
            "FLUX2_DRESS_SECOND_PASS_ENABLED",
            "0" if (low_latency_mode and "FLUX2_DRESS_SECOND_PASS_ENABLED" not in os.environ) else "1"
        )
        region_lock_second_pass_enabled = env_bool(
            "FLUX2_REGION_LOCK_SECOND_PASS_ENABLED",
            "0" if (low_latency_mode and "FLUX2_REGION_LOCK_SECOND_PASS_ENABLED" not in os.environ) else "1"
        )
        qwen_second_pass_enabled = env_bool(
            "FLUX2_QWEN_SECOND_PASS_ENABLED",
            "0" if (low_latency_mode and "FLUX2_QWEN_SECOND_PASS_ENABLED" not in os.environ) else "1"
        )
        
        # Color guard - affected by low latency mode
        color_guard_rerun_enabled = env_bool(
            "FLUX2_COLOR_GUARD_RERUN_ENABLED",
            "0" if (low_latency_mode and "FLUX2_COLOR_GUARD_RERUN_ENABLED" not in os.environ) else "1"
        )
        
        # Runtime scoring - affected by low latency mode
        runtime_scoring_enabled = env_bool(
            "FLUX2_RUNTIME_SCORING_ENABLED",
            "0" if low_latency_mode else "1"
        )
        
        # Negative prompt runtime mode
        negative_prompt_runtime_mode = os.getenv(
            "FLUX2_NEGATIVE_PROMPT_RUNTIME_MODE",
            "request_or_auto"
        ).strip().lower()
        if negative_prompt_runtime_mode not in {"disabled", "request_only", "request_or_auto", "auto_only"}:
            negative_prompt_runtime_mode = "request_or_auto"
        
        # Single garment extraction backend
        single_garment_extract_default_backend = os.getenv(
            "FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND",
            "auto"
        ).strip().lower()
        if single_garment_extract_default_backend == "auto":
            single_garment_extract_default_backend = _default_minicpm_family_backend()
        elif single_garment_extract_default_backend not in {"florence", "joycaption", "minicpm", "minicpm_service"}:
            single_garment_extract_default_backend = _default_minicpm_family_backend()
        
        return cls(
            # Backend selection
            descriptor_backend=descriptor_backend,
            fidelity_backend=fidelity_backend,
            allow_qwen_backend=allow_qwen_backend,
            
            # Prompting
            descriptor_compare=env_bool("FLUX2_DESCRIPTOR_COMPARE", "0"),
            negative_prompt_enable=env_bool("FLUX2_NEGATIVE_PROMPT_ENABLE", "1"),
            negative_prompt_default=os.getenv(
                "FLUX2_NEGATIVE_PROMPT_DEFAULT",
                (
                    "low quality, blurry, deformed body, extra limbs, extra fingers, wrong hands, "
                    "extra hand, third hand, duplicate arms, hand fused to garment, garment fused to skin, "
                    "identity change, different face, wrong skin tone, recolored garment, hue shift, "
                    "color drift, pattern drift, texture swap, logo/text watermark, duplicate garment, "
                    "layering artifacts, garment merge, ghost garment, incorrect neckline, incorrect hemline"
                )
            ).strip(),
            negative_prompt_runtime_mode=negative_prompt_runtime_mode,
            tryon_disable_runtime_negative_prompt=env_bool("FLUX2_TRYON_DISABLE_RUNTIME_NEGATIVE_PROMPT", "1"),
            
            # Performance
            low_latency_mode=low_latency_mode,
            single_candidate_mode=single_candidate_mode,
            runtime_scoring_enabled=runtime_scoring_enabled,
            force_runtime_scoring_for_single_candidate=env_bool("FLUX2_FORCE_RUNTIME_SCORING_FOR_SINGLE_CANDIDATE", "0"),
            
            # Second pass refinement - Dress
            dress_second_pass_enabled=dress_second_pass_enabled,
            dress_second_pass_extra_steps=max(1, env_int("FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS", 4)),
            dress_second_pass_max_steps=max(6, env_int("FLUX2_DRESS_SECOND_PASS_MAX_STEPS", 18)),
            
            # Second pass refinement - Region lock
            region_lock_second_pass_enabled=region_lock_second_pass_enabled,
            region_lock_second_pass_extra_steps=max(1, env_int("FLUX2_REGION_LOCK_SECOND_PASS_EXTRA_STEPS", 2)),
            region_lock_second_pass_max_steps=max(6, env_int("FLUX2_REGION_LOCK_SECOND_PASS_MAX_STEPS", 16)),
            
            # Second pass refinement - Qwen
            qwen_second_pass_enabled=qwen_second_pass_enabled,
            qwen_second_pass_extra_steps=max(1, env_int("FLUX2_QWEN_SECOND_PASS_EXTRA_STEPS", 6)),
            qwen_second_pass_max_steps=max(8, env_int("FLUX2_QWEN_SECOND_PASS_MAX_STEPS", 24)),
            qwen_min_steps=max(4, env_int("FLUX2_QWEN_MIN_STEPS", 10)),
            
            # Model preloading
            preload_qwen_with_flux2=env_bool("FLUX2_PRELOAD_QWEN_WITH_FLUX2", "1"),
            preload_joycaption_with_flux2=env_bool("FLUX2_PRELOAD_JOYCAPTION_WITH_FLUX2", "1"),
            preload_minicpm_with_flux2=env_bool("FLUX2_PRELOAD_MINICPM_WITH_FLUX2", "0"),
            unload_qwen_before_flux2=env_bool("FLUX2_UNLOAD_QWEN_BEFORE_FLUX2", "0"),
            qwen_extra_classify_pass=env_bool("FLUX2_QWEN_EXTRA_CLASSIFY_PASS", "0"),
            
            # Qwen image sizing
            qwen_product_caption_max_side=max(384, env_int("FLUX2_QWEN_PRODUCT_CAPTION_MAX_SIDE", 768)),
            qwen_product_caption_min_side=max(256, env_int("FLUX2_QWEN_PRODUCT_CAPTION_MIN_SIDE", 512)),
            qwen_user_caption_max_side=max(384, env_int("FLUX2_QWEN_USER_CAPTION_MAX_SIDE", 768)),
            qwen_user_caption_min_side=max(256, env_int("FLUX2_QWEN_USER_CAPTION_MIN_SIDE", 512)),
            
            # MiniCPM image sizing
            minicpm_product_caption_max_side=max(512, env_int("FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE", 1024)),
            minicpm_product_caption_min_side=max(256, env_int("FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE", 512)),
            minicpm_user_caption_max_side=max(512, env_int("FLUX2_MINICPM_USER_CAPTION_MAX_SIDE", 1024)),
            minicpm_user_caption_min_side=max(256, env_int("FLUX2_MINICPM_USER_CAPTION_MIN_SIDE", 512)),
            
            # Color processing
            color_lock_enabled=env_bool("FLUX2_COLOR_LOCK_ENABLED", "1"),
            color_lock_top_k=min(5, max(1, env_int("FLUX2_COLOR_LOCK_TOP_K", 3))),
            color_decontamination_enabled=env_bool("FLUX2_COLOR_DECONTAMINATION_ENABLED", "1"),
            color_decontam_alpha_high=max(1, min(255, env_int("FLUX2_COLOR_DECONTAM_ALPHA_HIGH", 240))),
            color_decontam_alpha_low=max(1, min(255, env_int("FLUX2_COLOR_DECONTAM_ALPHA_LOW", 200))),
            color_decontam_erode_iters=max(0, env_int("FLUX2_COLOR_DECONTAM_ERODE_ITERS", 1)),
            color_decontam_min_pixels=max(16, env_int("FLUX2_COLOR_DECONTAM_MIN_PIXELS", 64)),
            color_decontam_min_coverage_ratio=min(0.2, max(0.0, env_float("FLUX2_COLOR_DECONTAM_MIN_COVERAGE_RATIO", 0.006))),
            color_palette_min_area_percent=min(40.0, max(0.0, env_float("FLUX2_COLOR_PALETTE_MIN_AREA_PERCENT", 7.0))),
            color_profile_trim_dark_percentile=min(40.0, max(0.0, env_float("FLUX2_COLOR_PROFILE_TRIM_DARK_PERCENTILE", 10.0))),
            color_profile_trim_bright_percentile=min(100.0, max(60.0, env_float("FLUX2_COLOR_PROFILE_TRIM_BRIGHT_PERCENTILE", 98.0))),
            
            # Color guard rerun
            color_guard_rerun_enabled=color_guard_rerun_enabled,
            color_guard_drift_threshold=env_float("FLUX2_COLOR_GUARD_DRIFT_THRESHOLD", 12.0),
            color_guard_rerun_extra_steps=max(1, env_int("FLUX2_COLOR_GUARD_RERUN_EXTRA_STEPS", 2)),
            color_guard_rerun_max_steps=max(6, env_int("FLUX2_COLOR_GUARD_RERUN_MAX_STEPS", 18)),
            color_first_gate_enabled=env_bool("FLUX2_COLOR_FIRST_GATE_ENABLED", "1"),
            color_first_gate_max_drift_delta=max(0.0, env_float("FLUX2_COLOR_FIRST_GATE_MAX_DRIFT_DELTA", 6.0)),
            
            # Neutral color calibration
            neutral_post_color_calibration_enabled=env_bool("FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED", "1"),
            neutral_calibration_max_delta_l=max(1.0, env_float("FLUX2_NEUTRAL_CALIBRATION_MAX_DELTA_L", 10.0)),
            neutral_calibration_light_output_min_l=env_float("FLUX2_NEUTRAL_CALIBRATION_LIGHT_OUTPUT_MIN_L", 65.0),
            neutral_calibration_max_darken_light_output=max(0.5, env_float("FLUX2_NEUTRAL_CALIBRATION_MAX_DARKEN_LIGHT_OUTPUT", 5.0)),
            neutral_calibration_dark_output_max_l=env_float("FLUX2_NEUTRAL_CALIBRATION_DARK_OUTPUT_MAX_L", 40.0),
            neutral_calibration_max_brighten_dark_output=max(0.5, env_float("FLUX2_NEUTRAL_CALIBRATION_MAX_BRIGHTEN_DARK_OUTPUT", 15.0)),
            neutral_calibration_brightness_margin_l=max(0.0, env_float("FLUX2_NEUTRAL_CALIBRATION_BRIGHTNESS_MARGIN_L", 15.0)),
            neutral_calibration_light_source_min_l=env_float("FLUX2_NEUTRAL_CALIBRATION_LIGHT_SOURCE_MIN_L", 60.0),
            neutral_calibration_light_source_max_darken=max(0.5, env_float("FLUX2_NEUTRAL_CALIBRATION_LIGHT_SOURCE_MAX_DARKEN", 15.0)),
            neutral_calibration_dark_source_max_l=env_float("FLUX2_NEUTRAL_CALIBRATION_DARK_SOURCE_MAX_L", 40.0),
            neutral_calibration_dark_source_max_brighten=max(0.5, env_float("FLUX2_NEUTRAL_CALIBRATION_DARK_SOURCE_MAX_BRIGHTEN", 20.0)),
            
            # Detail lock
            detail_lock_enabled=env_bool("FLUX2_DETAIL_LOCK_ENABLED", "1"),
            
            # Single garment extraction
            single_garment_extract_default_backend=single_garment_extract_default_backend,
            single_garment_extract_default_steps=max(4, env_int("FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS", 10)),
            single_garment_extract_default_seed=max(0, env_int("FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED", 23)),
            single_garment_extract_upload_debug=env_bool("FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_DEBUG", "0"),
            single_garment_extract_disable_parser=env_bool("FLUX2_SINGLE_GARMENT_EXTRACT_DISABLE_PARSER", "1"),
            single_garment_extract_upload_raw_debug=env_bool("FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_RAW_DEBUG", "0"),
            single_garment_extract_expose_intermediate_urls=env_bool("FLUX2_SINGLE_GARMENT_EXTRACT_EXPOSE_INTERMEDIATE_URLS", "0"),
            
            # Shared runner
            share_base_runner=env_bool("FLUX2_SHARE_BASE_RUNNER", "1"),
        )
