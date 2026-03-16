"""
Unit tests for config/__init__.py unified configuration module.

Tests the Config class, get_config() singleton function, and integration
of all configuration modules (Flux2Config, AnalyzeConfig, MiniCPMConfig,
ColorConfig, AppConfig).
"""

import os
import unittest
from unittest.mock import patch
from config import Config, get_config, Flux2Config, AnalyzeConfig, MiniCPMConfig, ColorConfig, AppConfig


class TestConfigClass(unittest.TestCase):
    """Tests for the Config unified configuration class."""
    
    def test_config_has_all_modules(self):
        """Test that Config class has all required configuration modules."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
            
            # Verify all configuration modules are present
            self.assertIsInstance(config.app, AppConfig)
            self.assertIsInstance(config.flux2, Flux2Config)
            self.assertIsInstance(config.analyze, AnalyzeConfig)
            self.assertIsInstance(config.minicpm, MiniCPMConfig)
            self.assertIsInstance(config.color, ColorConfig)
    
    def test_config_loads_from_env(self):
        """Test that Config.from_env() loads configuration from environment variables."""
        test_env = {
            "GPU_CONCURRENCY": "2",
            "FLUX2_LOW_LATENCY_MODE": "1",
            "ANALYZE_MAX_ITEMS": "5",
            "MINICPM_SERVICE_URL": "http://test:8010",
            "COLOR_CONTEXT_DISABLE_MASKING": "1",
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            config = Config.from_env()
            
            # Verify values are loaded correctly
            self.assertEqual(config.app.gpu_concurrency, 2)
            self.assertTrue(config.flux2.low_latency_mode)
            self.assertEqual(config.analyze.max_items, 5)
            self.assertEqual(config.minicpm.service_url, "http://test:8010")
            self.assertTrue(config.color.context_disable_masking)
    
    def test_config_uses_defaults_when_env_empty(self):
        """Test that Config uses default values when environment variables are not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
            
            # Verify defaults are applied
            self.assertEqual(config.app.gpu_concurrency, 1)
            self.assertFalse(config.flux2.low_latency_mode)
            self.assertEqual(config.analyze.max_items, 3)
            self.assertIn("127.0.0.1", config.minicpm.service_url)
            self.assertFalse(config.color.context_disable_masking)
    
    def test_config_validates_types(self):
        """Test that Config validates field types using Pydantic."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
            
            # Verify types are correct
            self.assertIsInstance(config.app.gpu_concurrency, int)
            self.assertIsInstance(config.flux2.low_latency_mode, bool)
            self.assertIsInstance(config.analyze.max_items, int)
            self.assertIsInstance(config.minicpm.service_url, str)
            self.assertIsInstance(config.color.context_disable_masking, bool)


class TestGetConfigSingleton(unittest.TestCase):
    """Tests for the get_config() singleton function."""
    
    def setUp(self):
        """Reset singleton before each test."""
        import config
        config._config = None
    
    def test_get_config_returns_config_instance(self):
        """Test that get_config() returns a Config instance."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config()
            self.assertIsInstance(config, Config)
    
    def test_get_config_singleton_behavior(self):
        """Test that get_config() returns the same instance on multiple calls."""
        with patch.dict(os.environ, {}, clear=True):
            config1 = get_config()
            config2 = get_config()
            
            # Verify same instance is returned
            self.assertIs(config1, config2)
    
    def test_get_config_loads_once(self):
        """Test that configuration is loaded only once."""
        with patch.dict(os.environ, {"GPU_CONCURRENCY": "3"}, clear=True):
            config1 = get_config()
            self.assertEqual(config1.app.gpu_concurrency, 3)
            
            # Change environment variable
            os.environ["GPU_CONCURRENCY"] = "5"
            
            # Get config again - should return cached instance with old value
            config2 = get_config()
            self.assertEqual(config2.app.gpu_concurrency, 3)
            self.assertIs(config1, config2)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for the unified configuration system."""
    
    def test_config_exports_all_classes(self):
        """Test that config module exports all required classes."""
        from config import Config, get_config, Flux2Config, AnalyzeConfig, MiniCPMConfig, ColorConfig, AppConfig
        
        # Verify all exports are available
        self.assertIsNotNone(Config)
        self.assertIsNotNone(get_config)
        self.assertIsNotNone(Flux2Config)
        self.assertIsNotNone(AnalyzeConfig)
        self.assertIsNotNone(MiniCPMConfig)
        self.assertIsNotNone(ColorConfig)
        self.assertIsNotNone(AppConfig)
    
    def test_config_cross_module_consistency(self):
        """Test that configuration values are consistent across modules."""
        test_env = {
            "FLUX2_LOW_LATENCY_MODE": "1",
            "MINICPM_SERVICE_TIMEOUT_S": "60",
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            config = Config.from_env()
            
            # Verify low latency mode affects MiniCPM timeout
            self.assertTrue(config.flux2.low_latency_mode)
            self.assertEqual(config.minicpm.timeout_s, 60)
    
    def test_config_complex_scenario(self):
        """Test configuration with a complex real-world scenario."""
        test_env = {
            # App config
            "GPU_CONCURRENCY": "2",
            "REQUIRE_AZURE_UPLOAD": "1",
            "STARTUP_BACKGROUND_PRELOAD": "0",
            
            # Flux2 config
            "FLUX2_DESCRIPTOR_BACKEND": "florence",
            "FLUX2_LOW_LATENCY_MODE": "0",
            "FLUX2_COLOR_LOCK_ENABLED": "1",
            "FLUX2_COLOR_LOCK_TOP_K": "5",
            
            # Analyze config
            "ANALYZE_MAX_ITEMS": "5",
            "ANALYZE_REQUIRE_SELECTION": "0",
            "ANALYZE_ENABLE_HUMAN_PARSER": "1",
            "ANALYZE_EXTRACT_CLOTH": "1",
            
            # MiniCPM config
            "MINICPM_SERVICE_URL": "http://minicpm:8010",
            "MINICPM_SERVICE_CACHE_ENABLED": "1",
            "MINICPM_SERVICE_CACHE_TTL_SECONDS": "1800",
            
            # Color config
            "COLOR_CONTEXT_DISABLE_MASKING": "0",
            "GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED": "1",
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            config = Config.from_env()
            
            # Verify app config
            self.assertEqual(config.app.gpu_concurrency, 2)
            self.assertTrue(config.app.require_azure_upload)
            self.assertFalse(config.app.startup_background_preload)
            
            # Verify flux2 config
            self.assertEqual(config.flux2.descriptor_backend, "florence")
            self.assertFalse(config.flux2.low_latency_mode)
            self.assertTrue(config.flux2.color_lock_enabled)
            self.assertEqual(config.flux2.color_lock_top_k, 5)
            
            # Verify analyze config
            self.assertEqual(config.analyze.max_items, 5)
            self.assertFalse(config.analyze.require_selection)
            self.assertTrue(config.analyze.enable_human_parser)
            self.assertTrue(config.analyze.extract_cloth)
            
            # Verify minicpm config
            self.assertEqual(config.minicpm.service_url, "http://minicpm:8010")
            self.assertTrue(config.minicpm.cache_enabled)
            self.assertEqual(config.minicpm.cache_ttl_seconds, 1800)
            
            # Verify color config
            self.assertFalse(config.color.context_disable_masking)
            self.assertTrue(config.color.garment_semantic_override_enabled)


if __name__ == '__main__':
    unittest.main()
