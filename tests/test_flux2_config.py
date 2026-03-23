"""
Unit tests for config/flux2_config.py Flux2 configuration.

Tests the Flux2Config Pydantic model and from_env() class method
for loading Flux2-related configuration from environment variables.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.flux2_config import Flux2Config, _default_minicpm_family_backend


class TestFlux2ConfigDefaults(unittest.TestCase):
    """Tests for Flux2Config default values."""
    
    def test_default_backend_selection(self):
        """Test default backend selection logic."""
        # Without service URLs, should default to minicpm
        with patch.dict(os.environ, {}, clear=True):
            backend = _default_minicpm_family_backend()
            self.assertEqual(backend, "minicpm")
        
        # With MINICPM_SERVICE_URL, should default to minicpm_service
        with patch.dict(os.environ, {"MINICPM_SERVICE_URL": "http://localhost:8010"}):
            backend = _default_minicpm_family_backend()
            self.assertEqual(backend, "minicpm_service")
    
    def test_from_env_defaults(self):
        """Test loading config with all default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = Flux2Config.from_env()
            
            # Backend selection
            self.assertEqual(config.descriptor_backend, "minicpm")
            self.assertEqual(config.fidelity_backend, "florence")
            self.assertFalse(config.allow_qwen_backend)
            
            # Prompting
            self.assertFalse(config.descriptor_compare)
            self.assertTrue(config.negative_prompt_enable)
            self.assertIn("low quality", config.negative_prompt_default)
            self.assertEqual(config.negative_prompt_runtime_mode, "request_or_auto")
            self.assertTrue(config.tryon_disable_runtime_negative_prompt)
            
            # Performance
            self.assertFalse(config.low_latency_mode)
            self.assertEqual(config.single_candidate_mode, "auto")
            self.assertTrue(config.runtime_scoring_enabled)
            self.assertFalse(config.force_runtime_scoring_for_single_candidate)
            
            # Second pass refinement
            self.assertTrue(config.dress_second_pass_enabled)
            self.assertEqual(config.dress_second_pass_extra_steps, 4)
            self.assertEqual(config.dress_second_pass_max_steps, 18)
            
            # Color processing
            self.assertTrue(config.color_lock_enabled)
            self.assertEqual(config.color_lock_top_k, 3)
            self.assertTrue(config.color_decontamination_enabled)
            
            # Detail lock
            self.assertTrue(config.detail_lock_enabled)
            
            # Shared runner
            self.assertTrue(config.share_base_runner)
            self.assertEqual(config.lora_mode, "tryon")
            self.assertIn("BFS-Best-Face-Swap", config.bfs_lora_path)
            self.assertIn("bfs_head_v1", config.bfs_lora_weight_name)
            self.assertAlmostEqual(config.bfs_lora_scale, 0.65)
            self.assertIn("BFS-Best-Face-Swap", config.bfs_lora_fallback_repo)
            self.assertIn("bfs-best-face-swap", config.bfs_lora_local_cache_dir)


class TestFlux2ConfigEnvironmentVariables(unittest.TestCase):
    """Tests for Flux2Config loading from environment variables."""
    
    def test_backend_selection_from_env(self):
        """Test backend selection from environment variables."""
        with patch.dict(os.environ, {
            "FLUX2_DESCRIPTOR_BACKEND": "florence",
            "FLUX2_FIDELITY_BACKEND": "qwen2_5_vl",
            "FLUX2_ALLOW_QWEN_BACKEND": "1",
            "FLUX2_LORA_MODE": "stacked",
            "FLUX2_BFS_LORA_SCALE": "0.8",
            "FLUX2_BFS_LORA_FALLBACK_REPO": "custom/bfs",
        }):
            config = Flux2Config.from_env()
            self.assertEqual(config.descriptor_backend, "florence")
            self.assertEqual(config.fidelity_backend, "qwen2_5_vl")
            self.assertTrue(config.allow_qwen_backend)
            self.assertEqual(config.lora_mode, "stacked")
            self.assertAlmostEqual(config.bfs_lora_scale, 0.8)
            self.assertEqual(config.bfs_lora_fallback_repo, "custom/bfs")
    
    def test_low_latency_mode_affects_defaults(self):
        """Test that low latency mode affects other default values."""
        with patch.dict(os.environ, {"FLUX2_LOW_LATENCY_MODE": "1"}):
            config = Flux2Config.from_env()
            self.assertTrue(config.low_latency_mode)
            self.assertEqual(config.single_candidate_mode, "base")
            self.assertFalse(config.runtime_scoring_enabled)
            self.assertFalse(config.dress_second_pass_enabled)
            self.assertFalse(config.region_lock_second_pass_enabled)
            self.assertFalse(config.qwen_second_pass_enabled)
            self.assertFalse(config.color_guard_rerun_enabled)
    
    def test_color_processing_from_env(self):
        """Test color processing configuration from environment."""
        with patch.dict(os.environ, {
            "FLUX2_COLOR_LOCK_ENABLED": "0",
            "FLUX2_COLOR_LOCK_TOP_K": "5",
            "FLUX2_COLOR_DECONTAMINATION_ENABLED": "0",
            "FLUX2_COLOR_GUARD_DRIFT_THRESHOLD": "15.0",
        }):
            config = Flux2Config.from_env()
            self.assertFalse(config.color_lock_enabled)
            self.assertEqual(config.color_lock_top_k, 5)
            self.assertFalse(config.color_decontamination_enabled)
            self.assertEqual(config.color_guard_drift_threshold, 15.0)
    
    def test_second_pass_refinement_from_env(self):
        """Test second pass refinement configuration from environment."""
        with patch.dict(os.environ, {
            "FLUX2_DRESS_SECOND_PASS_ENABLED": "0",
            "FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS": "6",
            "FLUX2_QWEN_SECOND_PASS_MAX_STEPS": "30",
        }):
            config = Flux2Config.from_env()
            self.assertFalse(config.dress_second_pass_enabled)
            self.assertEqual(config.dress_second_pass_extra_steps, 6)
            self.assertEqual(config.qwen_second_pass_max_steps, 30)
    
    def test_model_preloading_from_env(self):
        """Test model preloading configuration from environment."""
        with patch.dict(os.environ, {
            "FLUX2_PRELOAD_QWEN_WITH_FLUX2": "0",
            "FLUX2_PRELOAD_JOYCAPTION_WITH_FLUX2": "0",
            "FLUX2_PRELOAD_MINICPM_WITH_FLUX2": "1",
            "FLUX2_UNLOAD_QWEN_BEFORE_FLUX2": "1",
        }):
            config = Flux2Config.from_env()
            self.assertFalse(config.preload_qwen_with_flux2)
            self.assertFalse(config.preload_joycaption_with_flux2)
            self.assertTrue(config.preload_minicpm_with_flux2)
            self.assertTrue(config.unload_qwen_before_flux2)
    
    def test_neutral_calibration_from_env(self):
        """Test neutral color calibration configuration from environment."""
        with patch.dict(os.environ, {
            "FLUX2_NEUTRAL_POST_COLOR_CALIBRATION_ENABLED": "0",
            "FLUX2_NEUTRAL_CALIBRATION_MAX_DELTA_L": "15.0",
            "FLUX2_NEUTRAL_CALIBRATION_LIGHT_OUTPUT_MIN_L": "70.0",
        }):
            config = Flux2Config.from_env()
            self.assertFalse(config.neutral_post_color_calibration_enabled)
            self.assertEqual(config.neutral_calibration_max_delta_l, 15.0)
            self.assertEqual(config.neutral_calibration_light_output_min_l, 70.0)
    
    def test_single_garment_extraction_from_env(self):
        """Test single garment extraction configuration from environment."""
        with patch.dict(os.environ, {
            "FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_BACKEND": "florence",
            "FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_STEPS": "15",
            "FLUX2_SINGLE_GARMENT_EXTRACT_DEFAULT_SEED": "42",
            "FLUX2_SINGLE_GARMENT_EXTRACT_UPLOAD_DEBUG": "1",
        }):
            config = Flux2Config.from_env()
            self.assertEqual(config.single_garment_extract_default_backend, "florence")
            self.assertEqual(config.single_garment_extract_default_steps, 15)
            self.assertEqual(config.single_garment_extract_default_seed, 42)
            self.assertTrue(config.single_garment_extract_upload_debug)


class TestFlux2ConfigValidation(unittest.TestCase):
    """Tests for Flux2Config field validation."""
    
    def test_color_lock_top_k_bounds(self):
        """Test that color_lock_top_k is bounded between 1 and 5."""
        with patch.dict(os.environ, {"FLUX2_COLOR_LOCK_TOP_K": "10"}):
            config = Flux2Config.from_env()
            self.assertEqual(config.color_lock_top_k, 5)  # Clamped to max
        
        with patch.dict(os.environ, {"FLUX2_COLOR_LOCK_TOP_K": "0"}):
            config = Flux2Config.from_env()
            self.assertEqual(config.color_lock_top_k, 1)  # Clamped to min
    
    def test_step_counts_minimum_values(self):
        """Test that step counts have minimum values enforced."""
        with patch.dict(os.environ, {
            "FLUX2_DRESS_SECOND_PASS_EXTRA_STEPS": "0",
            "FLUX2_DRESS_SECOND_PASS_MAX_STEPS": "2",
            "FLUX2_QWEN_MIN_STEPS": "1",
        }):
            config = Flux2Config.from_env()
            self.assertEqual(config.dress_second_pass_extra_steps, 1)  # Min 1
            self.assertEqual(config.dress_second_pass_max_steps, 6)    # Min 6
            self.assertEqual(config.qwen_min_steps, 4)                 # Min 4
    
    def test_alpha_threshold_bounds(self):
        """Test that alpha thresholds are bounded between 1 and 255."""
        with patch.dict(os.environ, {
            "FLUX2_COLOR_DECONTAM_ALPHA_HIGH": "300",
            "FLUX2_COLOR_DECONTAM_ALPHA_LOW": "0",
        }):
            config = Flux2Config.from_env()
            self.assertEqual(config.color_decontam_alpha_high, 255)  # Clamped to max
            self.assertEqual(config.color_decontam_alpha_low, 1)     # Clamped to min
    
    def test_invalid_backend_uses_default(self):
        """Test that invalid backend values fall back to defaults."""
        with patch.dict(os.environ, {
            "FLUX2_DESCRIPTOR_BACKEND": "invalid_backend",
            "FLUX2_FIDELITY_BACKEND": "invalid_backend",
        }):
            config = Flux2Config.from_env()
            self.assertEqual(config.descriptor_backend, "minicpm")  # Falls back to default
            self.assertEqual(config.fidelity_backend, "florence")  # Falls back to default


class TestFlux2ConfigFieldCount(unittest.TestCase):
    """Tests for Flux2Config field coverage."""
    
    def test_field_count_matches_spec(self):
        """Test that Flux2Config has approximately the expected number of fields."""
        config = Flux2Config.from_env()
        field_count = len(Flux2Config.model_fields)
        
        # Design spec mentions ~60 fields, we have 72 which is reasonable
        # (includes all the detailed color calibration and sizing fields)
        self.assertGreaterEqual(field_count, 60, "Should have at least 60 configuration fields")
        self.assertLessEqual(field_count, 80, "Should have at most 80 configuration fields")


if __name__ == '__main__':
    unittest.main()
