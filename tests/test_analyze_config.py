"""
Tests for AnalyzeConfig configuration module.

Verifies:
- Default values match specification
- Environment variable parsing
- Field validation and constraints
- from_env() class method
"""

import os
import pytest
from config.analyze_config import AnalyzeConfig


class TestAnalyzeConfigDefaults:
    """Test default configuration values."""
    
    def test_detection_defaults(self):
        """Verify detection field defaults."""
        config = AnalyzeConfig()
        assert config.max_items == 3
        assert config.require_selection is True
        assert config.min_accept_confidence == 0.25
    
    def test_hybrid_scoring_defaults(self):
        """Verify hybrid scoring defaults."""
        config = AnalyzeConfig()
        assert config.hybrid_top_k == 3
        assert config.hybrid_min_score == 0.0
        assert config.hybrid_weight_yolo == 0.45
        assert config.hybrid_weight_florence == 0.45
        assert config.hybrid_weight_bbox == 0.10
    
    def test_caption_preview_defaults(self):
        """Verify caption and preview defaults."""
        config = AnalyzeConfig()
        assert config.caption_mode == "short"
        assert config.selection_preview_format == "jpeg"
        assert config.selection_preview_max_side == 640
        assert config.selection_preview_jpeg_quality == 80
    
    def test_resource_limits_defaults(self):
        """Verify resource limit defaults."""
        config = AnalyzeConfig()
        assert config.gpu_queue_timeout_s == 70.0
        assert config.max_file_bytes == 3 * 1024 * 1024
    
    def test_quality_checks_defaults(self):
        """Verify quality check defaults."""
        config = AnalyzeConfig()
        assert config.blur_check_enabled is False
        assert config.blur_min_focus_score == 22.0
        assert config.blur_focus_max_edge == 1024
    
    def test_parser_integration_defaults(self):
        """Verify parser integration defaults."""
        config = AnalyzeConfig()
        assert config.enable_human_parser is True
        assert config.enable_parser_split is False
        assert config.parser_min_area_ratio == 0.015
        assert config.parser_pad == 12
        assert config.use_parser_for_prerouting is False
        assert config.parser_top_dress_backfill is True
        assert config.parser_top_min_ratio == 0.008
        assert config.parser_dress_backfill_min_ratio == 0.015
    
    def test_heuristic_splitting_defaults(self):
        """Verify heuristic splitting defaults."""
        config = AnalyzeConfig()
        assert config.enable_heuristic_split is False
        assert config.tighten_split_crops is True
        assert config.tighten_split_pad == 8
        assert config.tighten_split_min_pixels == 48
        assert config.heuristic_min_height_ratio == 0.78
        assert config.heuristic_top_portion == 0.52
    
    def test_florence_integration_defaults(self):
        """Verify Florence integration defaults."""
        config = AnalyzeConfig()
        assert config.primary_type_with_florence is True
        assert config.preload_florence is True
        assert config.florence_dress_lock_min_score == 0.74
        assert config.preload_minicpm is True
    
    def test_extraction_defaults(self):
        """Verify extraction defaults."""
        config = AnalyzeConfig()
        assert config.extract_cloth is True
        assert config.flux_disable_lora is True
        assert config.preload_flux_runner is False
        assert config.use_parser_post_extract is False
        assert config.pass_detection_prompt_to_extract is True
        assert config.prompt_from_extracted is True
        assert config.extract_parser_only is True
        assert config.extract_force_bbox_crop is True
        assert config.extract_crop_pad_ratio == 0.08
        assert config.extract_crop_pad_ratio_dress == 0.15
        assert config.extract_min_mask_ratio == 0.01
        assert config.extract_relaxed_rescue is True
        assert config.extract_edge_feather_px == 1
        assert config.require_extracted_prompt is True
    
    def test_garment_postprocessing_defaults(self):
        """Verify garment postprocessing defaults."""
        config = AnalyzeConfig()
        assert config.garment_postprocess_enabled is True
        assert config.garment_target_aspect_w == 2
        assert config.garment_target_aspect_h == 3
        assert config.garment_alpha_threshold == 12
        assert config.garment_white_threshold == 246
        assert config.garment_enhance_enabled is True
        assert config.garment_enhance_sharpness == 1.22
        assert config.garment_enhance_contrast == 1.08
        assert config.garment_enhance_color == 1.04
        assert config.garment_enhance_brightness == 1.02
        assert config.garment_output_background == "white"
    
    def test_background_removal_defaults(self):
        """Verify background removal defaults."""
        config = AnalyzeConfig()
        assert config.bg_removal_backend == "raw"
        assert config.birefnet_model_id == "ZhengPeng7/BiRefNet"
        assert config.birefnet_input_size == 1024
    
    def test_progress_sync_defaults(self):
        """Verify progress sync defaults."""
        config = AnalyzeConfig()
        assert config.progress_sync_async is True
        assert config.progress_sync_max_workers == 2
    
    def test_color_processing_defaults(self):
        """Verify color processing defaults."""
        config = AnalyzeConfig()
        assert config.fashion_basecolour_trial_enabled is True
        assert config.color_parser_sampling_trial_enabled is True
        assert config.fashion_basecolour_apply_min_score == 0.90
    
    def test_type_inference_defaults(self):
        """Verify type inference defaults."""
        config = AnalyzeConfig()
        assert config.collapse_same_type is False
        assert config.collapse_same_type_min_iou == 0.85
        assert config.auto_select_multi_dress is True
        assert config.uncertain_fullbody_to_dress is True
        assert config.uncertain_fullbody_min_height_ratio == 0.78
        assert config.force_fullbody_split_on_same_type is True
        assert config.force_fullbody_split_min_height_ratio == 0.72


class TestAnalyzeConfigFromEnv:
    """Test configuration loading from environment variables."""
    
    def test_from_env_with_defaults(self, monkeypatch):
        """Test from_env() with no environment variables set."""
        # Clear all ANALYZE_* env vars
        for key in list(os.environ.keys()):
            if key.startswith("ANALYZE_") or key.startswith("HYBRID_"):
                monkeypatch.delenv(key, raising=False)
        
        config = AnalyzeConfig.from_env()
        
        # Verify defaults are used
        assert config.max_items == 3
        assert config.require_selection is True
        assert config.blur_check_enabled is False
    
    def test_from_env_with_custom_values(self, monkeypatch):
        """Test from_env() with custom environment variables."""
        monkeypatch.setenv("ANALYZE_MAX_ITEMS", "5")
        monkeypatch.setenv("ANALYZE_REQUIRE_SELECTION", "0")
        monkeypatch.setenv("ANALYZE_MIN_ACCEPT_CONFIDENCE", "0.5")
        monkeypatch.setenv("ANALYZE_BLUR_CHECK_ENABLED", "1")
        monkeypatch.setenv("ANALYZE_BLUR_MIN_FOCUS_SCORE", "30.0")
        monkeypatch.setenv("ANALYZE_PRELOAD_MINICPM", "0")
        
        config = AnalyzeConfig.from_env()
        
        assert config.max_items == 5
        assert config.require_selection is False
        assert config.min_accept_confidence == 0.5
        assert config.blur_check_enabled is True
        assert config.blur_min_focus_score == 30.0
        assert config.preload_minicpm is False
    
    def test_from_env_with_invalid_values(self, monkeypatch):
        """Test from_env() handles invalid values gracefully."""
        monkeypatch.setenv("ANALYZE_MAX_ITEMS", "invalid")
        monkeypatch.setenv("ANALYZE_MIN_ACCEPT_CONFIDENCE", "not_a_float")
        
        config = AnalyzeConfig.from_env()
        
        # Should fall back to defaults
        assert config.max_items == 3
        assert config.min_accept_confidence == 0.25
    
    def test_from_env_caption_mode_validation(self, monkeypatch):
        """Test caption_mode validates allowed values."""
        monkeypatch.setenv("ANALYZE_CAPTION_MODE", "invalid")
        config = AnalyzeConfig.from_env()
        assert config.caption_mode == "short"
        
        monkeypatch.setenv("ANALYZE_CAPTION_MODE", "detailed")
        config = AnalyzeConfig.from_env()
        assert config.caption_mode == "detailed"
    
    def test_from_env_preview_format_validation(self, monkeypatch):
        """Test selection_preview_format validates allowed values."""
        monkeypatch.setenv("ANALYZE_SELECTION_PREVIEW_FORMAT", "invalid")
        config = AnalyzeConfig.from_env()
        assert config.selection_preview_format == "jpeg"
        
        monkeypatch.setenv("ANALYZE_SELECTION_PREVIEW_FORMAT", "png")
        config = AnalyzeConfig.from_env()
        assert config.selection_preview_format == "png"
    
    def test_from_env_background_validation(self, monkeypatch):
        """Test garment_output_background validates allowed values."""
        monkeypatch.setenv("ANALYZE_GARMENT_OUTPUT_BACKGROUND", "invalid")
        config = AnalyzeConfig.from_env()
        assert config.garment_output_background == "white"
        
        monkeypatch.setenv("ANALYZE_GARMENT_OUTPUT_BACKGROUND", "transparent")
        config = AnalyzeConfig.from_env()
        assert config.garment_output_background == "transparent"
    
    def test_from_env_bg_removal_backend_validation(self, monkeypatch):
        """Test bg_removal_backend validates allowed values."""
        monkeypatch.setenv("ANALYZE_BG_REMOVAL_BACKEND", "invalid")
        config = AnalyzeConfig.from_env()
        assert config.bg_removal_backend == "raw"
        
        monkeypatch.setenv("ANALYZE_BG_REMOVAL_BACKEND", "birefnet")
        config = AnalyzeConfig.from_env()
        assert config.bg_removal_backend == "birefnet"
    
    def test_from_env_min_max_constraints(self, monkeypatch):
        """Test from_env() applies min/max constraints."""
        # Test max_items minimum
        monkeypatch.setenv("ANALYZE_MAX_ITEMS", "0")
        config = AnalyzeConfig.from_env()
        assert config.max_items == 1  # Should be clamped to min 1
        
        # Test jpeg quality bounds
        monkeypatch.setenv("ANALYZE_SELECTION_PREVIEW_JPEG_QUALITY", "10")
        config = AnalyzeConfig.from_env()
        assert config.selection_preview_jpeg_quality == 40  # Should be clamped to min 40
        
        monkeypatch.setenv("ANALYZE_SELECTION_PREVIEW_JPEG_QUALITY", "100")
        config = AnalyzeConfig.from_env()
        assert config.selection_preview_jpeg_quality == 95  # Should be clamped to max 95
        
        # Test alpha threshold bounds
        monkeypatch.setenv("ANALYZE_GARMENT_ALPHA_THRESHOLD", "-10")
        config = AnalyzeConfig.from_env()
        assert config.garment_alpha_threshold == 0  # Should be clamped to min 0
        
        monkeypatch.setenv("ANALYZE_GARMENT_ALPHA_THRESHOLD", "300")
        config = AnalyzeConfig.from_env()
        assert config.garment_alpha_threshold == 255  # Should be clamped to max 255


class TestAnalyzeConfigValidation:
    """Test Pydantic field validation."""
    
    def test_literal_field_validation(self):
        """Test Literal fields reject invalid values."""
        with pytest.raises(ValueError):
            AnalyzeConfig(caption_mode="invalid")
        
        with pytest.raises(ValueError):
            AnalyzeConfig(selection_preview_format="bmp")
        
        with pytest.raises(ValueError):
            AnalyzeConfig(garment_output_background="gray")
        
        with pytest.raises(ValueError):
            AnalyzeConfig(bg_removal_backend="unknown")
    
    def test_type_validation(self):
        """Test field type validation."""
        with pytest.raises(ValueError):
            AnalyzeConfig(max_items="not_an_int")
        
        with pytest.raises(ValueError):
            AnalyzeConfig(require_selection="not_a_bool")
        
        with pytest.raises(ValueError):
            AnalyzeConfig(min_accept_confidence="not_a_float")
    
    def test_valid_literal_values(self):
        """Test Literal fields accept valid values."""
        config = AnalyzeConfig(
            caption_mode="detailed",
            selection_preview_format="png",
            garment_output_background="transparent",
            bg_removal_backend="birefnet",
        )
        assert config.caption_mode == "detailed"
        assert config.selection_preview_format == "png"
        assert config.garment_output_background == "transparent"
        assert config.bg_removal_backend == "birefnet"


class TestAnalyzeConfigDocumentation:
    """Test configuration documentation and field descriptions."""
    
    def test_all_fields_have_descriptions(self):
        """Verify all fields have Field descriptions."""
        schema = AnalyzeConfig.model_json_schema()
        properties = schema.get("properties", {})
        
        # Check that key fields have descriptions
        assert "description" in properties["max_items"]
        assert "description" in properties["require_selection"]
        assert "description" in properties["extract_cloth"]
        assert "description" in properties["garment_postprocess_enabled"]
    
    def test_docstring_present(self):
        """Verify class has docstring."""
        assert AnalyzeConfig.__doc__ is not None
        assert "Analysis pipeline configuration" in AnalyzeConfig.__doc__
    
    def test_from_env_docstring(self):
        """Verify from_env method has docstring."""
        assert AnalyzeConfig.from_env.__doc__ is not None
        assert "environment variables" in AnalyzeConfig.from_env.__doc__.lower()
