"""
Analysis pipeline configuration.

This module defines configuration for the garment analysis pipeline including:
- Detection settings (max items, confidence thresholds)
- Hybrid scoring weights
- Caption and preview generation
- Resource limits
- Quality checks
- Parser integration
- Heuristic splitting
- Florence integration
- Extraction settings
- Garment postprocessing
- Background removal
- Progress sync
- Color processing
- Type inference
"""

import os
from typing import Literal
from pydantic import BaseModel, Field
from .base import env_int, env_float, env_bool


class AnalyzeConfig(BaseModel):
    """
    Analysis pipeline configuration.
    
    This configuration covers all aspects of the garment analysis workflow:
    - Detection and scoring of garments in images
    - Parser-based and heuristic splitting for multi-garment images
    - Garment extraction and background removal
    - Postprocessing and enhancement
    - Color analysis and type inference
    """
    
    # Detection
    max_items: int = Field(default=3, description="Maximum number of garments to detect")
    require_selection: bool = Field(default=True, description="Require user selection for multiple items")
    min_accept_confidence: float = Field(default=0.25, description="Minimum confidence to accept detection")
    
    # Hybrid scoring
    hybrid_top_k: int = Field(default=3, description="Top K items for hybrid scoring")
    hybrid_min_score: float = Field(default=0.0, description="Minimum hybrid score threshold")
    hybrid_weight_yolo: float = Field(default=0.45, description="Weight for YOLO score in hybrid")
    hybrid_weight_florence: float = Field(default=0.45, description="Weight for Florence score in hybrid")
    hybrid_weight_bbox: float = Field(default=0.10, description="Weight for bbox score in hybrid")
    
    # Caption and preview
    caption_mode: Literal["short", "detailed"] = Field(default="short", description="Caption generation mode")
    selection_preview_format: Literal["jpeg", "png"] = Field(default="jpeg", description="Preview image format")
    selection_preview_max_side: int = Field(default=640, description="Maximum preview dimension")
    selection_preview_jpeg_quality: int = Field(default=80, description="JPEG quality for previews")
    
    # Resource limits
    gpu_queue_timeout_s: float = Field(default=70.0, description="GPU queue timeout in seconds")
    max_file_bytes: int = Field(default=3 * 1024 * 1024, description="Maximum file size in bytes")
    
    # Quality checks
    blur_check_enabled: bool = Field(default=False, description="Enable blur detection")
    blur_min_focus_score: float = Field(default=22.0, description="Minimum focus score to pass blur check")
    blur_focus_max_edge: int = Field(default=1024, description="Maximum edge for blur check downsampling")
    
    # Parser integration
    enable_human_parser: bool = Field(default=True, description="Enable human parser for segmentation")
    enable_parser_split: bool = Field(default=False, description="Enable parser-based outfit splitting")
    parser_min_area_ratio: float = Field(default=0.015, description="Minimum area ratio for parser segments")
    parser_pad: int = Field(default=12, description="Padding for parser crops")
    use_parser_for_prerouting: bool = Field(default=False, description="Use parser for pre-routing decisions")
    parser_top_dress_backfill: bool = Field(default=True, description="Backfill top/dress from parser")
    parser_top_min_ratio: float = Field(default=0.008, description="Minimum ratio for parser top detection")
    parser_dress_backfill_min_ratio: float = Field(default=0.015, description="Minimum ratio for dress backfill")
    
    # Heuristic splitting
    enable_heuristic_split: bool = Field(default=False, description="Enable heuristic outfit splitting")
    tighten_split_crops: bool = Field(default=True, description="Tighten split crops to content")
    tighten_split_pad: int = Field(default=8, description="Padding for tightened splits")
    tighten_split_min_pixels: int = Field(default=48, description="Minimum pixels for split crops")
    heuristic_min_height_ratio: float = Field(default=0.78, description="Minimum height ratio for heuristic split")
    heuristic_top_portion: float = Field(default=0.52, description="Top portion ratio for heuristic split")
    heuristic_top_trim_px: int = Field(default=0, description="Top trim in pixels for heuristic split")
    heuristic_bottom_overlap_px: int = Field(default=32, description="Bottom overlap in pixels")
    heuristic_bottom_overlap_ratio: float = Field(default=0.11, description="Bottom overlap ratio")
    heuristic_bottom_trim_shorts_ratio: float = Field(default=0.44, description="Bottom trim ratio for shorts")
    heuristic_max_width_ratio: float = Field(default=0.98, description="Maximum width ratio for heuristic split")
    
    # Tighten bottom adjustments
    tighten_bottom_top_extra_ratio: float = Field(default=0.12, description="Top extra ratio for bottom tightening")
    tighten_bottom_bottom_extra_ratio: float = Field(default=0.05, description="Bottom extra ratio for tightening")
    tighten_bottom_top_max_overlap_px: int = Field(default=72, description="Max top overlap in pixels")
    tighten_bottom_max_down_shift_ratio: float = Field(default=0.08, description="Max downward shift ratio")
    tighten_bottom_max_gap_from_top_px: int = Field(default=96, description="Max gap from top in pixels")
    tighten_bottom_min_width_ratio: float = Field(default=0.92, description="Minimum width ratio for bottom")
    
    # Florence integration
    primary_type_with_florence: bool = Field(default=True, description="Use Florence for primary type detection")
    preload_florence: bool = Field(default=True, description="Preload Florence model at startup")
    florence_dress_lock_min_score: float = Field(default=0.74, description="Minimum score to lock dress type")
    
    # Extraction
    extract_cloth: bool = Field(default=True, description="Extract cloth from detected garments")
    flux_disable_lora: bool = Field(default=True, description="Disable LoRA for Flux2 in analyze")
    preload_flux_runner: bool = Field(default=False, description="Preload Flux2 runner at startup")
    flux2_single_garment_extract_default_steps: int = Field(default=10, description="Default steps for Flux2 single garment extraction")
    flux2_single_garment_extract_default_seed: int = Field(default=23, description="Default seed for Flux2 single garment extraction")
    use_parser_post_extract: bool = Field(default=False, description="Use parser after extraction")
    pass_detection_prompt_to_extract: bool = Field(default=True, description="Pass detection prompt to extraction")
    prompt_from_extracted: bool = Field(default=True, description="Generate prompt from extracted image")
    extract_parser_only: bool = Field(default=True, description="Use parser-only extraction mode")
    extract_force_bbox_crop: bool = Field(default=True, description="Force bbox crop in extraction")
    extract_crop_pad_ratio: float = Field(default=0.18, description="Padding ratio for extraction crops")
    extract_crop_pad_ratio_dress: float = Field(default=0.28, description="Padding ratio for dress extraction")
    extract_crop_bottom_extra_ratio_dress: float = Field(default=0.32, description="Bottom extra ratio for dress crops")
    extract_crop_top_extra_ratio_bottom: float = Field(default=0.12, description="Top extra ratio for bottom crops")
    extract_crop_top_extra_ratio_bottom_multi: float = Field(default=0.12, description="Top extra ratio for multi-bottom")
    extract_dress_top_recovery_ratio: float = Field(default=0.14, description="Top recovery ratio for dress")
    extract_top_top_recovery_ratio: float = Field(default=0.08, description="Top recovery ratio for top garments")
    extract_min_mask_ratio: float = Field(default=0.01, description="Minimum mask ratio for extraction")
    extract_relaxed_rescue: bool = Field(default=True, description="Enable relaxed rescue for failed extractions")
    extract_edge_feather_px: int = Field(default=1, description="Edge feathering in pixels")
    extract_component_min_ratio: float = Field(default=0.0007, description="Minimum component ratio")
    extract_keep_dilate: int = Field(default=3, description="Dilation iterations for keep mask")
    extract_parser_kill_dilate: int = Field(default=2, description="Dilation iterations for parser kill mask")
    extract_max_body_ratio: float = Field(default=0.008, description="Maximum body ratio in extraction")
    extract_body_strip_dilate: int = Field(default=3, description="Dilation for body strip removal")
    extract_body_strip_max_ratio: float = Field(default=0.28, description="Maximum body strip ratio")
    require_extracted_prompt: bool = Field(default=True, description="Require prompt from extracted garment")
    
    # Garment postprocessing
    garment_postprocess_enabled: bool = Field(default=True, description="Enable garment postprocessing")
    garment_target_aspect_w: int = Field(default=2, description="Target aspect ratio width")
    garment_target_aspect_h: int = Field(default=3, description="Target aspect ratio height")
    garment_alpha_threshold: int = Field(default=12, description="Alpha threshold for transparency")
    garment_white_threshold: int = Field(default=246, description="White threshold for background")
    garment_hole_fill_max_pixels: int = Field(default=7000, description="Maximum pixels for hole filling")
    garment_enhance_enabled: bool = Field(default=True, description="Enable garment enhancement")
    garment_enhance_sharpness: float = Field(default=1.22, description="Sharpness enhancement factor")
    garment_enhance_contrast: float = Field(default=1.08, description="Contrast enhancement factor")
    garment_enhance_color: float = Field(default=1.04, description="Color enhancement factor")
    garment_enhance_brightness: float = Field(default=1.02, description="Brightness enhancement factor")
    garment_enhance_lighting_auto: bool = Field(default=False, description="Enable automatic lighting enhancement")
    garment_output_background: Literal["transparent", "white"] = Field(
        default="white",
        description="Output background type"
    )
    
    # Top skin rim cleanup
    top_skin_rim_cleanup: bool = Field(default=True, description="Enable top skin rim cleanup")
    top_skin_rim_max_ratio: float = Field(default=0.08, description="Maximum skin rim ratio for cleanup")
    
    # Background removal
    bg_removal_backend: Literal["raw", "white", "rembg", "birefnet"] = Field(
        default="raw",
        description="Background removal backend"
    )
    birefnet_model_id: str = Field(default="ZhengPeng7/BiRefNet", description="BiRefNet model ID")
    birefnet_input_size: int = Field(default=1024, description="BiRefNet input size")
    
    # Progress sync
    progress_sync_async: bool = Field(default=True, description="Enable async progress sync")
    progress_sync_max_workers: int = Field(default=2, description="Max workers for progress sync")
    
    # Color processing
    fashion_basecolour_trial_enabled: bool = Field(default=True, description="Enable fashion base color trial")
    color_parser_sampling_trial_enabled: bool = Field(default=True, description="Enable color parser sampling")
    fashion_basecolour_apply_min_score: float = Field(
        default=0.90,
        description="Minimum score to apply fashion base color"
    )
    
    # Type inference
    collapse_same_type: bool = Field(default=False, description="Collapse detections of same type")
    collapse_same_type_min_iou: float = Field(default=0.85, description="Minimum IoU for same-type collapse")
    auto_select_multi_dress: bool = Field(default=True, description="Auto-select when multiple dresses detected")
    uncertain_fullbody_to_dress: bool = Field(default=True, description="Convert uncertain fullbody to dress")
    uncertain_fullbody_min_height_ratio: float = Field(
        default=0.78,
        description="Minimum height ratio for uncertain fullbody"
    )
    uncertain_fullbody_min_area_ratio: float = Field(
        default=0.22,
        description="Minimum area ratio for uncertain fullbody"
    )
    uncertain_fullbody_max_top_ratio: float = Field(
        default=0.26,
        description="Maximum top ratio for uncertain fullbody"
    )
    uncertain_fullbody_min_bottom_ratio: float = Field(
        default=0.90,
        description="Minimum bottom ratio for uncertain fullbody"
    )
    uncertain_fullbody_min_area_advantage: float = Field(
        default=1.55,
        description="Minimum area advantage for uncertain fullbody"
    )
    force_fullbody_split_on_same_type: bool = Field(
        default=True,
        description="Force fullbody split when same type detected"
    )
    force_fullbody_split_min_height_ratio: float = Field(
        default=0.72,
        description="Minimum height ratio for forced fullbody split"
    )
    
    # Auxiliary detection
    aux_min_rel_area: float = Field(default=0.22, description="Minimum relative area for auxiliary detections")
    
    @classmethod
    def from_env(cls) -> "AnalyzeConfig":
        """
        Load configuration from environment variables.
        
        All ANALYZE_* environment variables are parsed and validated.
        Invalid values fall back to defaults with warnings logged.
        
        Returns:
            AnalyzeConfig instance with values from environment
        """
        # Parse caption mode
        caption_mode = os.getenv("ANALYZE_CAPTION_MODE", "short").strip().lower()
        if caption_mode not in {"short", "detailed"}:
            caption_mode = "short"
        
        # Parse selection preview format
        selection_preview_format = os.getenv("ANALYZE_SELECTION_PREVIEW_FORMAT", "jpeg").strip().lower()
        if selection_preview_format not in {"jpeg", "png"}:
            selection_preview_format = "jpeg"
        
        # Parse garment output background
        garment_output_background = os.getenv("ANALYZE_GARMENT_OUTPUT_BACKGROUND", "white").strip().lower()
        if garment_output_background not in {"transparent", "white"}:
            garment_output_background = "white"
        
        # Parse background removal backend
        bg_removal_backend = os.getenv("ANALYZE_BG_REMOVAL_BACKEND", "raw").strip().lower()
        if bg_removal_backend not in {"raw", "white", "rembg", "birefnet"}:
            bg_removal_backend = "raw"
        
        return cls(
            # Detection
            max_items=max(1, env_int("ANALYZE_MAX_ITEMS", 3)),
            require_selection=env_bool("ANALYZE_REQUIRE_SELECTION", "1"),
            min_accept_confidence=env_float("ANALYZE_MIN_ACCEPT_CONFIDENCE", 0.25),
            
            # Hybrid scoring
            hybrid_top_k=max(1, env_int("HYBRID_TOP_K", 3)),
            hybrid_min_score=env_float("HYBRID_MIN_SCORE", 0.0),
            hybrid_weight_yolo=env_float("HYBRID_WEIGHT_YOLO", 0.45),
            hybrid_weight_florence=env_float("HYBRID_WEIGHT_FLORENCE", 0.45),
            hybrid_weight_bbox=env_float("HYBRID_WEIGHT_BBOX", 0.10),
            
            # Caption and preview
            caption_mode=caption_mode,
            selection_preview_format=selection_preview_format,
            selection_preview_max_side=max(128, env_int("ANALYZE_SELECTION_PREVIEW_MAX_SIDE", 640)),
            selection_preview_jpeg_quality=max(40, min(95, env_int("ANALYZE_SELECTION_PREVIEW_JPEG_QUALITY", 80))),
            
            # Resource limits
            gpu_queue_timeout_s=max(5.0, env_float("ANALYZE_GPU_QUEUE_TIMEOUT_S", 70.0)),
            max_file_bytes=max(1, env_int("ANALYZE_MAX_FILE_BYTES", 3 * 1024 * 1024)),
            
            # Quality checks
            blur_check_enabled=env_bool("ANALYZE_BLUR_CHECK_ENABLED", "0"),
            blur_min_focus_score=env_float("ANALYZE_BLUR_MIN_FOCUS_SCORE", 22.0),
            blur_focus_max_edge=max(256, env_int("ANALYZE_BLUR_FOCUS_MAX_EDGE", 1024)),
            
            # Parser integration
            enable_human_parser=env_bool("ANALYZE_ENABLE_HUMAN_PARSER", "1"),
            enable_parser_split=env_bool("ANALYZE_ENABLE_PARSER_SPLIT", "0"),
            parser_min_area_ratio=env_float("ANALYZE_PARSER_MIN_AREA_RATIO", 0.015),
            parser_pad=max(0, env_int("ANALYZE_PARSER_PAD", 12)),
            use_parser_for_prerouting=env_bool("ANALYZE_USE_PARSER_FOR_PREROUTING", "0"),
            parser_top_dress_backfill=env_bool("ANALYZE_PARSER_TOP_DRESS_BACKFILL", "1"),
            parser_top_min_ratio=env_float("ANALYZE_PARSER_TOP_MIN_RATIO", 0.008),
            parser_dress_backfill_min_ratio=env_float("ANALYZE_PARSER_DRESS_BACKFILL_MIN_RATIO", 0.015),
            
            # Heuristic splitting
            enable_heuristic_split=env_bool("ANALYZE_ENABLE_HEURISTIC_SPLIT", "0"),
            tighten_split_crops=env_bool("ANALYZE_TIGHTEN_SPLIT_CROPS", "1"),
            tighten_split_pad=max(0, env_int("ANALYZE_TIGHTEN_SPLIT_PAD", 8)),
            tighten_split_min_pixels=max(32, env_int("ANALYZE_TIGHTEN_SPLIT_MIN_PIXELS", 48)),
            heuristic_min_height_ratio=env_float("ANALYZE_HEURISTIC_MIN_HEIGHT_RATIO", 0.78),
            heuristic_top_portion=env_float("ANALYZE_HEURISTIC_TOP_PORTION", 0.52),
            heuristic_top_trim_px=max(0, env_int("ANALYZE_HEURISTIC_TOP_TRIM_PX", 0)),
            heuristic_bottom_overlap_px=max(0, env_int("ANALYZE_HEURISTIC_BOTTOM_OVERLAP_PX", 32)),
            heuristic_bottom_overlap_ratio=env_float("ANALYZE_HEURISTIC_BOTTOM_OVERLAP_RATIO", 0.11),
            heuristic_bottom_trim_shorts_ratio=env_float("ANALYZE_HEURISTIC_BOTTOM_TRIM_SHORTS_RATIO", 0.44),
            heuristic_max_width_ratio=env_float("ANALYZE_HEURISTIC_MAX_WIDTH_RATIO", 0.98),
            
            # Tighten bottom adjustments
            tighten_bottom_top_extra_ratio=env_float("ANALYZE_TIGHTEN_BOTTOM_TOP_EXTRA_RATIO", 0.12),
            tighten_bottom_bottom_extra_ratio=env_float("ANALYZE_TIGHTEN_BOTTOM_BOTTOM_EXTRA_RATIO", 0.05),
            tighten_bottom_top_max_overlap_px=max(0, env_int("ANALYZE_TIGHTEN_BOTTOM_TOP_MAX_OVERLAP_PX", 72)),
            tighten_bottom_max_down_shift_ratio=max(0.0, env_float("ANALYZE_TIGHTEN_BOTTOM_MAX_DOWN_SHIFT_RATIO", 0.08)),
            tighten_bottom_max_gap_from_top_px=max(0, env_int("ANALYZE_TIGHTEN_BOTTOM_MAX_GAP_FROM_TOP_PX", 96)),
            tighten_bottom_min_width_ratio=env_float("ANALYZE_TIGHTEN_BOTTOM_MIN_WIDTH_RATIO", 0.92),
            
            # Florence integration
            primary_type_with_florence=env_bool("ANALYZE_PRIMARY_TYPE_WITH_FLORENCE", "1"),
            preload_florence=env_bool("ANALYZE_PRELOAD_FLORENCE", "1"),
            florence_dress_lock_min_score=env_float("ANALYZE_FLORENCE_DRESS_LOCK_MIN_SCORE", 0.74),
            
            # Extraction
            extract_cloth=env_bool("ANALYZE_EXTRACT_CLOTH", "1"),
            flux_disable_lora=env_bool("ANALYZE_FLUX_DISABLE_LORA", "1"),
            preload_flux_runner=env_bool("ANALYZE_PRELOAD_FLUX_RUNNER", "0"),
            flux2_single_garment_extract_default_steps=max(4, env_int("FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS", 10)),
            flux2_single_garment_extract_default_seed=max(0, env_int("FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED", 23)),
            use_parser_post_extract=env_bool("ANALYZE_USE_PARSER_POST_EXTRACT", "0"),
            pass_detection_prompt_to_extract=env_bool("ANALYZE_PASS_DETECTION_PROMPT_TO_EXTRACT", "1"),
            prompt_from_extracted=env_bool("ANALYZE_PROMPT_FROM_EXTRACTED", "1"),
            extract_parser_only=env_bool("ANALYZE_EXTRACT_PARSER_ONLY", "1"),
            extract_force_bbox_crop=env_bool("ANALYZE_EXTRACT_FORCE_BBOX_CROP", "1"),
            extract_crop_pad_ratio=env_float("ANALYZE_EXTRACT_CROP_PAD_RATIO", 0.18),
            extract_crop_pad_ratio_dress=env_float("ANALYZE_EXTRACT_CROP_PAD_RATIO_DRESS", 0.28),
            extract_crop_bottom_extra_ratio_dress=env_float("ANALYZE_EXTRACT_CROP_BOTTOM_EXTRA_RATIO_DRESS", 0.32),
            extract_crop_top_extra_ratio_bottom=env_float("ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM", 0.12),
            extract_crop_top_extra_ratio_bottom_multi=env_float("ANALYZE_EXTRACT_CROP_TOP_EXTRA_RATIO_BOTTOM_MULTI", 0.12),
            extract_dress_top_recovery_ratio=env_float("ANALYZE_EXTRACT_DRESS_TOP_RECOVERY_RATIO", 0.14),
            extract_top_top_recovery_ratio=env_float("ANALYZE_EXTRACT_TOP_TOP_RECOVERY_RATIO", 0.08),
            extract_min_mask_ratio=env_float("ANALYZE_EXTRACT_MIN_MASK_RATIO", 0.01),
            extract_relaxed_rescue=env_bool("ANALYZE_EXTRACT_RELAXED_RESCUE", "1"),
            extract_edge_feather_px=max(0, env_int("ANALYZE_EXTRACT_EDGE_FEATHER_PX", 1)),
            extract_component_min_ratio=env_float("ANALYZE_EXTRACT_COMPONENT_MIN_RATIO", 0.0007),
            extract_keep_dilate=max(0, env_int("ANALYZE_EXTRACT_KEEP_DILATE", 3)),
            extract_parser_kill_dilate=max(1, env_int("ANALYZE_EXTRACT_PARSER_KILL_DILATE", 2)),
            extract_max_body_ratio=max(0.0, env_float("ANALYZE_EXTRACT_MAX_BODY_RATIO", 0.008)),
            extract_body_strip_dilate=max(0, env_int("ANALYZE_EXTRACT_BODY_STRIP_DILATE", 3)),
            extract_body_strip_max_ratio=min(0.95, max(0.01, env_float("ANALYZE_EXTRACT_BODY_STRIP_MAX_RATIO", 0.28))),
            require_extracted_prompt=env_bool("ANALYZE_REQUIRE_EXTRACTED_PROMPT", "1"),
            
            # Garment postprocessing
            garment_postprocess_enabled=env_bool("ANALYZE_GARMENT_POSTPROCESS_ENABLED", "1"),
            garment_target_aspect_w=max(1, env_int("ANALYZE_GARMENT_TARGET_ASPECT_W", 2)),
            garment_target_aspect_h=max(1, env_int("ANALYZE_GARMENT_TARGET_ASPECT_H", 3)),
            garment_alpha_threshold=max(0, min(255, env_int("ANALYZE_GARMENT_ALPHA_THRESHOLD", 12))),
            garment_white_threshold=max(200, min(255, env_int("ANALYZE_GARMENT_WHITE_THRESHOLD", 246))),
            garment_hole_fill_max_pixels=max(0, env_int("ANALYZE_GARMENT_HOLE_FILL_MAX_PIXELS", 7000)),
            garment_enhance_enabled=env_bool("ANALYZE_GARMENT_ENHANCE_ENABLED", "1"),
            garment_enhance_sharpness=env_float("ANALYZE_GARMENT_ENHANCE_SHARPNESS", 1.22),
            garment_enhance_contrast=env_float("ANALYZE_GARMENT_ENHANCE_CONTRAST", 1.08),
            garment_enhance_color=env_float("ANALYZE_GARMENT_ENHANCE_COLOR", 1.04),
            garment_enhance_brightness=env_float("ANALYZE_GARMENT_ENHANCE_BRIGHTNESS", 1.02),
            garment_enhance_lighting_auto=env_bool("ANALYZE_GARMENT_ENHANCE_LIGHTING_AUTO", "0"),
            garment_output_background=garment_output_background,
            
            # Top skin rim cleanup
            top_skin_rim_cleanup=env_bool("ANALYZE_TOP_SKIN_RIM_CLEANUP", "1"),
            top_skin_rim_max_ratio=min(0.30, max(0.0, env_float("ANALYZE_TOP_SKIN_RIM_MAX_RATIO", 0.08))),
            
            # Background removal
            bg_removal_backend=bg_removal_backend,
            birefnet_model_id=os.getenv("ANALYZE_BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet").strip(),
            birefnet_input_size=max(512, env_int("ANALYZE_BIREFNET_INPUT_SIZE", 1024)),
            
            # Progress sync
            progress_sync_async=env_bool("ANALYZE_PROGRESS_SYNC_ASYNC", "1"),
            progress_sync_max_workers=max(1, env_int("ANALYZE_PROGRESS_SYNC_MAX_WORKERS", 2)),
            
            # Color processing
            fashion_basecolour_trial_enabled=env_bool("ANALYZE_FASHION_BASECOLOUR_TRIAL_ENABLED", "1"),
            color_parser_sampling_trial_enabled=env_bool("ANALYZE_COLOR_PARSER_SAMPLING_TRIAL_ENABLED", "1"),
            fashion_basecolour_apply_min_score=min(
                0.99,
                max(0.50, env_float("ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE", 0.90)),
            ),
            
            # Type inference
            collapse_same_type=env_bool("ANALYZE_COLLAPSE_SAME_TYPE", "0"),
            collapse_same_type_min_iou=env_float("ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU", 0.85),
            auto_select_multi_dress=env_bool("ANALYZE_AUTO_SELECT_MULTI_DRESS", "1"),
            uncertain_fullbody_to_dress=env_bool("ANALYZE_UNCERTAIN_FULLBODY_TO_DRESS", "1"),
            uncertain_fullbody_min_height_ratio=env_float("ANALYZE_UNCERTAIN_FULLBODY_MIN_HEIGHT_RATIO", 0.78),
            uncertain_fullbody_min_area_ratio=env_float("ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_RATIO", 0.22),
            uncertain_fullbody_max_top_ratio=env_float("ANALYZE_UNCERTAIN_FULLBODY_MAX_TOP_RATIO", 0.26),
            uncertain_fullbody_min_bottom_ratio=env_float("ANALYZE_UNCERTAIN_FULLBODY_MIN_BOTTOM_RATIO", 0.90),
            uncertain_fullbody_min_area_advantage=env_float("ANALYZE_UNCERTAIN_FULLBODY_MIN_AREA_ADVANTAGE", 1.55),
            force_fullbody_split_on_same_type=env_bool("ANALYZE_FORCE_FULLBODY_SPLIT_ON_SAME_TYPE", "1"),
            force_fullbody_split_min_height_ratio=env_float("ANALYZE_FORCE_FULLBODY_SPLIT_MIN_HEIGHT_RATIO", 0.72),
            
            # Auxiliary detection
            aux_min_rel_area=env_float("ANALYZE_AUX_MIN_REL_AREA", 0.22),
        )
