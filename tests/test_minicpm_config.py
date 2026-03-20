"""
Unit tests for config/minicpm_config.py MiniCPM configuration.

Tests the MiniCPMConfig Pydantic model and from_env() class method
for loading MiniCPM service-related configuration from environment variables.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.minicpm_config import MiniCPMConfig
from config.prompts import get_minicpm_garment_prompt


class TestMiniCPMConfigDefaults(unittest.TestCase):
    """Tests for MiniCPMConfig default values."""
    
    def test_from_env_defaults(self):
        """Test loading config with all default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = MiniCPMConfig.from_env()
            
            # Service URLs
            self.assertEqual(config.service_url, "http://127.0.0.1:8010")
            self.assertEqual(config.analyze_service_url, "")
            
            # Timeouts
            self.assertEqual(config.timeout_s, 120)
            self.assertEqual(config.connect_timeout_s, 10.0)
            
            # Token limits
            self.assertEqual(config.garment_max_new_tokens, 640)
            self.assertEqual(config.person_max_new_tokens, 140)
            
            # Caching
            self.assertTrue(config.cache_enabled)
            self.assertEqual(config.cache_ttl_seconds, 900)
            self.assertEqual(config.cache_max_entries, 1024)
            
            # Connection pooling
            self.assertEqual(config.pool_maxsize, 16)
            self.assertTrue(config.local_file_first)
            
            # Prompts
            self.assertEqual(config.garment_min_words, 200)
            self.assertEqual(config.garment_prompt, get_minicpm_garment_prompt())
            garment_prompt = config.garment_prompt.lower()
            self.assertIn("requested garment type", garment_prompt)
            self.assertIn("never infer hidden length", garment_prompt)
            self.assertIn("not fully visible", garment_prompt)
            self.assertIn("structured, feature-rich way", garment_prompt)
            self.assertIn("Describe only the human subject", config.person_prompt)
            self.assertIn("face=<facial expression", config.person_prompt)
            self.assertIn("lower_body_pose=<lower-body stance", config.person_prompt)
            self.assertIn("held_object=<objects held or used in hand", config.person_prompt)
            
            # Image sizing
            self.assertEqual(config.product_caption_max_side, 1024)
            self.assertEqual(config.product_caption_min_side, 512)
            self.assertEqual(config.user_caption_max_side, 1024)
            self.assertEqual(config.user_caption_min_side, 512)


class TestMiniCPMConfigEnvironmentVariables(unittest.TestCase):
    """Tests for MiniCPMConfig loading from environment variables."""
    
    def test_service_urls_from_env(self):
        """Test service URL configuration from environment."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_URL": "http://minicpm.example.com:8080",
            "ANALYZE_MINICPM_SERVICE_URL": "http://analyze.example.com:8081",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.service_url, "http://minicpm.example.com:8080")
            self.assertEqual(config.analyze_service_url, "http://analyze.example.com:8081")
    
    def test_service_url_trailing_slash_stripped(self):
        """Test that trailing slashes are stripped from service URLs."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_URL": "http://localhost:8010/",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.service_url, "http://localhost:8010")
    
    def test_timeouts_from_env(self):
        """Test timeout configuration from environment."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_TIMEOUT_S": "180",
            "MINICPM_SERVICE_CONNECT_TIMEOUT_S": "15.5",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.timeout_s, 180)
            self.assertEqual(config.connect_timeout_s, 15.5)
    
    def test_low_latency_mode_affects_timeouts(self):
        """Test that low latency mode affects timeout defaults."""
        with patch.dict(os.environ, {"FLUX2_LOW_LATENCY_MODE": "1"}):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.timeout_s, 60)
            self.assertEqual(config.connect_timeout_s, 6.0)
    
    def test_token_limits_from_env(self):
        """Test token limit configuration from environment."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS": "512",
            "MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS": "200",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.garment_max_new_tokens, 512)
            self.assertEqual(config.person_max_new_tokens, 200)
    
    def test_low_latency_mode_affects_token_limits(self):
        """Test that low latency mode affects token limit defaults."""
        with patch.dict(os.environ, {"FLUX2_LOW_LATENCY_MODE": "1"}):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.garment_max_new_tokens, 512)
            self.assertEqual(config.person_max_new_tokens, 96)
    
    def test_caching_from_env(self):
        """Test caching configuration from environment."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_CACHE_ENABLED": "0",
            "MINICPM_SERVICE_CACHE_TTL_SECONDS": "3600",
            "MINICPM_SERVICE_CACHE_MAX_ENTRIES": "2048",
        }):
            config = MiniCPMConfig.from_env()
            self.assertFalse(config.cache_enabled)
            self.assertEqual(config.cache_ttl_seconds, 3600)
            self.assertEqual(config.cache_max_entries, 2048)
    
    def test_low_latency_mode_affects_cache_ttl(self):
        """Test that low latency mode affects cache TTL default."""
        with patch.dict(os.environ, {"FLUX2_LOW_LATENCY_MODE": "1"}):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.cache_ttl_seconds, 1800)
    
    def test_connection_pooling_from_env(self):
        """Test connection pooling configuration from environment."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_POOL_MAXSIZE": "32",
            "MINICPM_SERVICE_LOCAL_FILE_FIRST": "0",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.pool_maxsize, 32)
            self.assertFalse(config.local_file_first)
    
    def test_prompts_from_env(self):
        """Test prompt configuration from environment."""
        custom_garment_prompt = "Custom garment description prompt"
        custom_person_prompt = "Custom person description prompt"
        
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_GARMENT_MIN_WORDS": "15",
            "MINICPM_SERVICE_GARMENT_PROMPT": custom_garment_prompt,
            "MINICPM_SERVICE_PERSON_PROMPT": custom_person_prompt,
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.garment_min_words, 15)
            self.assertEqual(config.garment_prompt, custom_garment_prompt)
            self.assertEqual(config.person_prompt, custom_person_prompt)

    def test_category_prompt_helpers_are_rich_and_type_specific(self):
        top_prompt = get_minicpm_garment_prompt("top").lower()
        bottom_prompt = get_minicpm_garment_prompt("bottom").lower()
        dress_prompt = get_minicpm_garment_prompt("dress").lower()
        outer_prompt = get_minicpm_garment_prompt("outer").lower()

        self.assertIn("top-only guidance", top_prompt)
        self.assertIn("shoulder layout", top_prompt)
        self.assertIn("lower-body features", top_prompt)
        self.assertIn("not fully visible", top_prompt)

        self.assertIn("bottom-only guidance", bottom_prompt)
        self.assertIn("waistband", bottom_prompt)
        self.assertIn("upper-body features", bottom_prompt)
        self.assertIn("not fully visible", bottom_prompt)

        self.assertIn("dress-only guidance", dress_prompt)
        self.assertIn("bodice", dress_prompt)
        self.assertIn("two pieces", dress_prompt)
        self.assertIn("not fully visible", dress_prompt)

        self.assertIn("outerwear-only guidance", outer_prompt)
        self.assertIn("collar", outer_prompt)
        self.assertIn("inner garments", outer_prompt)
        self.assertIn("not fully visible", outer_prompt)
    
    def test_image_sizing_from_env(self):
        """Test image sizing configuration from environment."""
        with patch.dict(os.environ, {
            "FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE": "2048",
            "FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE": "768",
            "FLUX2_MINICPM_USER_CAPTION_MAX_SIDE": "1536",
            "FLUX2_MINICPM_USER_CAPTION_MIN_SIDE": "640",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.product_caption_max_side, 2048)
            self.assertEqual(config.product_caption_min_side, 768)
            self.assertEqual(config.user_caption_max_side, 1536)
            self.assertEqual(config.user_caption_min_side, 640)


class TestMiniCPMConfigValidation(unittest.TestCase):
    """Tests for MiniCPMConfig field validation."""
    
    def test_timeout_minimum_values(self):
        """Test that timeouts have minimum values enforced."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_TIMEOUT_S": "2",
            "MINICPM_SERVICE_CONNECT_TIMEOUT_S": "0.5",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.timeout_s, 5)    # Min 5
            self.assertEqual(config.connect_timeout_s, 1.0)  # Min 1.0
    
    def test_token_limit_minimum_values(self):
        """Test that token limits have minimum values enforced."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS": "32",
            "MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS": "16",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.garment_max_new_tokens, 64)  # Min 64
            self.assertEqual(config.person_max_new_tokens, 32)   # Min 32
    
    def test_cache_max_entries_minimum_value(self):
        """Test that cache max entries has minimum value enforced."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_CACHE_MAX_ENTRIES": "8",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.cache_max_entries, 16)  # Min 16
    
    def test_pool_maxsize_minimum_value(self):
        """Test that pool maxsize has minimum value enforced."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_POOL_MAXSIZE": "2",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.pool_maxsize, 4)  # Min 4
    
    def test_garment_min_words_minimum_value(self):
        """Test that garment min words has minimum value enforced."""
        with patch.dict(os.environ, {
            "MINICPM_SERVICE_GARMENT_MIN_WORDS": "2",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.garment_min_words, 4)  # Min 4
    
    def test_image_sizing_minimum_values(self):
        """Test that image sizing parameters have minimum values enforced."""
        with patch.dict(os.environ, {
            "FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE": "256",
            "FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE": "128",
            "FLUX2_MINICPM_USER_CAPTION_MAX_SIDE": "256",
            "FLUX2_MINICPM_USER_CAPTION_MIN_SIDE": "128",
        }):
            config = MiniCPMConfig.from_env()
            self.assertEqual(config.product_caption_max_side, 512)  # Min 512
            self.assertEqual(config.product_caption_min_side, 256)  # Min 256
            self.assertEqual(config.user_caption_max_side, 512)     # Min 512
            self.assertEqual(config.user_caption_min_side, 256)     # Min 256


class TestMiniCPMConfigFieldCount(unittest.TestCase):
    """Tests for MiniCPMConfig field coverage."""
    
    def test_field_count_matches_spec(self):
        """Test that MiniCPMConfig has the expected number of fields."""
        config = MiniCPMConfig.from_env()
        field_count = len(MiniCPMConfig.model_fields)
        
        # Design spec shows 18 fields for MiniCPMConfig
        self.assertGreaterEqual(field_count, 16, "Should have at least 16 configuration fields")
        self.assertLessEqual(field_count, 20, "Should have at most 20 configuration fields")
    
    def test_all_required_fields_present(self):
        """Test that all required fields from design spec are present."""
        config = MiniCPMConfig.from_env()
        
        # Service URLs and timeouts
        self.assertTrue(hasattr(config, 'service_url'))
        self.assertTrue(hasattr(config, 'analyze_service_url'))
        self.assertTrue(hasattr(config, 'timeout_s'))
        self.assertTrue(hasattr(config, 'connect_timeout_s'))
        
        # Token limits
        self.assertTrue(hasattr(config, 'garment_max_new_tokens'))
        self.assertTrue(hasattr(config, 'person_max_new_tokens'))
        
        # Caching
        self.assertTrue(hasattr(config, 'cache_enabled'))
        self.assertTrue(hasattr(config, 'cache_ttl_seconds'))
        self.assertTrue(hasattr(config, 'cache_max_entries'))
        
        # Connection pooling
        self.assertTrue(hasattr(config, 'pool_maxsize'))
        self.assertTrue(hasattr(config, 'local_file_first'))
        
        # Prompts
        self.assertTrue(hasattr(config, 'garment_min_words'))
        self.assertTrue(hasattr(config, 'garment_prompt'))
        self.assertTrue(hasattr(config, 'person_prompt'))
        
        # Image sizing
        self.assertTrue(hasattr(config, 'product_caption_max_side'))
        self.assertTrue(hasattr(config, 'product_caption_min_side'))
        self.assertTrue(hasattr(config, 'user_caption_max_side'))
        self.assertTrue(hasattr(config, 'user_caption_min_side'))


if __name__ == '__main__':
    unittest.main()
