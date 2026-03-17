"""
Utilities package for the fashion analysis system.

This package provides utility functions organized by domain:

- color_processing: Color analysis, conversion, and palette operations
- validation: Input validation, garment type normalization, and quality checks
- image_preprocessing: Image manipulation, cropping, resizing, and enhancement
- metadata_extraction: Garment metadata building and extraction
- prompt_generation: Prompt parsing, generation, and processing
- scoring: Ranking, filtering, and confidence scoring
- user_preparation: User image validation and preparation (placeholder)
- parser_operations: Human parser integration and operations (placeholder)
- descriptor_processing: Descriptor normalization and processing (placeholder)

Usage Examples:
    >>> from utils.color_processing import hex_to_rgb_triplet, color_family
    >>> from utils.validation import normalize_garment_type
    >>> from utils.image_preprocessing import crop_image, resize_image
    >>> from utils.scoring import hybrid_score, rank_items
    
    # Color processing
    >>> rgb = hex_to_rgb_triplet("#FF0000")
    >>> family = color_family("red")
    
    # Validation
    >>> garment_type = normalize_garment_type("T-Shirt")
    
    # Image preprocessing
    >>> cropped = crop_image(image, [0, 0, 100, 100])
    >>> resized = resize_image(image, 512, 256)
    
    # Scoring
    >>> score = hybrid_score(0.8, 0.7, 0.6)
    >>> ranked = rank_items(detection_results)

Module Organization:
    The utilities are organized into domain-specific modules to maintain
    clear separation of concerns and make the codebase more maintainable.
    Each module contains pure functions with no side effects.
"""

# Color Processing Utilities
from .color_processing import (
    # Color conversion functions
    hex_to_rgb_triplet,
    rgb_to_hsv,
    hsv_to_rgb,
    rgb_to_hex,
    hex_to_lab_triplet,
    lab_to_rgb,
    
    # Color analysis functions
    color_distance,
    color_family,
    is_neutral_color_token,
    canonical_color_token,
    nearest_color_label,
    rgb_hue_deg,
    
    # Color bucketing functions
    bucket_color_brightness,
    bucket_color_saturation,
    bucket_color_undertone,
    
    # Palette operations
    extract_color_palette,
    palette_weighted_hue_deg,
    palette_hue_from_hexes,
    filter_palette_entries_by_area,
    extract_lab_color_profile,
    color_labels_from_hex_palette,
    compute_palette_delta_e,
    
    # Color descriptor functions
    resolve_garment_color_truth,
)

# Validation Utilities
from .validation import (
    normalize_garment_type,
    canonical_coverage_for_type,
    sanitize_prompt_fact_value,
    descriptor_word_count,
    descriptor_is_weak,
    validate_image_quality,
    is_valid_garment_type,
    validate_descriptor_length,
    sanitize_garment_description,
    normalize_coverage_description,
    validate_color_value,
    validate_material_name,
    validate_size_value,
    validate_prompt_fact_key,
    normalize_prompt_fact_key,
    GARMENT_TYPE_SYNONYMS,
)

# Image Preprocessing Utilities
from .image_preprocessing import (
    # Basic image operations
    crop_image,
    resize_image,
    pad_image,
    
    # Bounding box operations
    bbox_iou,
    expand_bbox,
    merge_bboxes,
    bbox_from_mask,
    bbox_y_overlap_ratio,
    bbox_x_overlap_ratio,
    bbox_x_gap,
    square_bbox_from_bbox,
    crop_square_with_padding,
    expand_bbox_by_type,
    expand_bbox_with_ratios,
    robust_bbox_from_component_mask,
    content_bbox_from_image,
    
    # Mask operations
    dilate_mask,
    erode_mask,
    feather_mask_edges,
    apply_mask,
    extract_alpha_mask,
    connected_support_mask,
    crop_rgba_with_mask,
    mask_connected_components,
    
    # Background removal
    remove_background_rembg,
    remove_background_birefnet,
    
    # Image enhancement
    enhance_image,
    adjust_brightness,
    adjust_contrast,
)

# Prompt Generation Utilities
from .prompt_generation import (
    # Parsing functions
    parse_structured_descriptor,
    parse_garment_prompt_sections,
    extract_json_object_from_text,
    
    # Normalization functions
    normalize_minicpm_descriptor_text,
    sanitize_florence_garment_description,
    
    # Prompt building functions
    build_tryon_prompt,
    build_tryon_prompt_v2,
    build_garment_prompt_natural,
    build_flux2_prompt,
    
    # Avoid clause functions
    extract_generation_only_avoid_directives,
    build_florence_contamination_avoid_clause,
    merge_avoid_clause_sentences,
    
    # Fact processing functions
    extract_prompt_fact_segments,
    serialize_prompt_fact_segments,
    
    # Descriptor enhancement
    enrich_garment_descriptor,
    augment_identity_lock,
    
    # Type inference
    infer_flux2_target_type,
    ensure_target_type_in_description,
)

# Scoring Utilities
from .scoring import (
    # Core scoring functions
    hybrid_score,
    calculate_bbox_prior,
    calculate_detection_confidence,
    
    # Ranking and filtering
    rank_items,
    item_rank_by_score,
    filter_by_min_score,
    dedupe_items_by_iou,
    
    # Specialized scoring
    score_user_prep_face_candidate,
    score_user_prepare_prompt_description,
    score_requested_type_geometry,
    calculate_yolo_support_score,
    score_adaptive_crop_caption,
    
    # Utility functions
    find_largest_instance,
    calculate_instance_area,
    calculate_bbox_iou,
)

# Note: The following modules are placeholders and will be implemented in future tasks
# - user_preparation: User image validation and preparation utilities
# - parser_operations: Human parser integration and mask operations
# - descriptor_processing: Descriptor normalization and processing utilities

__all__ = [
    # Color processing
    "hex_to_rgb_triplet", "rgb_to_hsv", "hsv_to_rgb", "rgb_to_hex", "hex_to_lab_triplet", "lab_to_rgb",
    "color_distance", "color_family", "is_neutral_color_token", "canonical_color_token",
    "nearest_color_label", "rgb_hue_deg", "bucket_color_brightness", "bucket_color_saturation", "bucket_color_undertone",
    "extract_color_palette", "palette_weighted_hue_deg", "palette_hue_from_hexes", "filter_palette_entries_by_area",
    "extract_lab_color_profile", "color_labels_from_hex_palette", "compute_palette_delta_e", "resolve_garment_color_truth",
    
    # Validation
    "normalize_garment_type", "canonical_coverage_for_type", "sanitize_prompt_fact_value", "descriptor_word_count",
    "descriptor_is_weak", "validate_image_quality", "is_valid_garment_type", "validate_descriptor_length",
    "sanitize_garment_description", "normalize_coverage_description", "validate_color_value", "validate_material_name",
    "validate_size_value", "validate_prompt_fact_key", "normalize_prompt_fact_key", "GARMENT_TYPE_SYNONYMS",
    
    # Image preprocessing
    "crop_image", "resize_image", "pad_image", "bbox_iou", "expand_bbox", "merge_bboxes",
    "bbox_from_mask", "bbox_y_overlap_ratio", "bbox_x_overlap_ratio", "bbox_x_gap",
    "square_bbox_from_bbox", "crop_square_with_padding", "expand_bbox_by_type", "expand_bbox_with_ratios",
    "robust_bbox_from_component_mask", "content_bbox_from_image", "dilate_mask", "erode_mask",
    "feather_mask_edges", "apply_mask", "extract_alpha_mask", "connected_support_mask",
    "crop_rgba_with_mask", "mask_connected_components", "remove_background_rembg", "remove_background_birefnet",
    "enhance_image", "adjust_brightness", "adjust_contrast",
    
    # Prompt generation
    "parse_structured_descriptor", "parse_garment_prompt_sections", "extract_json_object_from_text",
    "normalize_minicpm_descriptor_text", "sanitize_florence_garment_description", "build_tryon_prompt",
    "build_flux2_prompt", "extract_generation_only_avoid_directives", "build_florence_contamination_avoid_clause",
    "merge_avoid_clause_sentences", "extract_prompt_fact_segments", "serialize_prompt_fact_segments",
    "enrich_garment_descriptor", "augment_identity_lock", "infer_flux2_target_type", "ensure_target_type_in_description",
    
    # Scoring
    "hybrid_score", "calculate_bbox_prior", "calculate_detection_confidence", "rank_items",
    "item_rank_by_score", "filter_by_min_score", "dedupe_items_by_iou", "score_user_prep_face_candidate",
    "score_user_prepare_prompt_description", "score_requested_type_geometry", "calculate_yolo_support_score",
    "score_adaptive_crop_caption", "find_largest_instance", "calculate_instance_area", "calculate_bbox_iou",
]
