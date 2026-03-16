"""
Unit tests for config/app_config.py Application configuration.

Tests the AppConfig Pydantic model and from_env() class method
for loading general application configuration from environment variables.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import AppConfig


class TestAppConfigDefaults(unittest.TestCase):
    """Tests for AppConfig default values."""
    
    def test_from_env_defaults(self):
        """Test loading config with all default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env()
            
            # Default values
            self.assertEqual(config.gpu_concurrency, 1)
            self.assertFalse(config.require_azure_upload)
            self.assertEqual(config.vto_output_container, "wardrobe-outputs")
            self.assertEqual(config.jwt_access_secret, "")
            self.assertTrue(config.startup_background_preload)
            self.assertTrue(config.wardrobe_results_enabled)
            self.assertTrue(config.wardrobe_results_save_input)
            self.assertTrue(config.wardrobe_results_save_output)
            self.assertTrue(config.wardrobe_results_save_masks)
            self.assertFalse(config.enable_wardrobe_progress_sync)
            self.assertEqual(config.wardrobe_progress_api_base_url, "")
            self.assertEqual(config.wardrobe_progress_sync_timeout_s, 20)
            self.assertFalse(config.wardrobe_progress_include_input_image)


class TestAppConfigEnvironmentVariables(unittest.TestCase):
    """Tests for AppConfig loading from environment variables."""
    
    def test_gpu_concurrency_from_env(self):
        """Test gpu_concurrency configuration from environment."""
        # Test custom value
        with patch.dict(os.environ, {"GPU_CONCURRENCY": "4"}):
            config = AppConfig.from_env()
            self.assertEqual(config.gpu_concurrency, 4)
        
        # Test minimum value enforcement (should be at least 1)
        with patch.dict(os.environ, {"GPU_CONCURRENCY": "0"}):
            config = AppConfig.from_env()
            self.assertEqual(config.gpu_concurrency, 1)
        
        with patch.dict(os.environ, {"GPU_CONCURRENCY": "-5"}):
            config = AppConfig.from_env()
            self.assertEqual(config.gpu_concurrency, 1)
    
    def test_require_azure_upload_from_env(self):
        """Test require_azure_upload configuration from environment."""
        # Test enabling Azure upload requirement
        with patch.dict(os.environ, {"REQUIRE_AZURE_UPLOAD": "1"}):
            config = AppConfig.from_env()
            self.assertTrue(config.require_azure_upload)
        
        # Test default (not required)
        with patch.dict(os.environ, {"REQUIRE_AZURE_UPLOAD": "0"}):
            config = AppConfig.from_env()
            self.assertFalse(config.require_azure_upload)
    
    def test_vto_output_container_from_env(self):
        """Test vto_output_container configuration from environment."""
        # Test primary env var
        with patch.dict(os.environ, {"AZURE_STORAGE_VTO_OUTPUT_CONTAINER": "custom-vto"}):
            config = AppConfig.from_env()
            self.assertEqual(config.vto_output_container, "custom-vto")
        
        # Test fallback env var
        with patch.dict(os.environ, {"AZURE_STORAGE_OUTPUT_CONTAINER": "fallback-container"}):
            config = AppConfig.from_env()
            self.assertEqual(config.vto_output_container, "fallback-container")
        
        # Test primary takes precedence over fallback
        with patch.dict(os.environ, {
            "AZURE_STORAGE_VTO_OUTPUT_CONTAINER": "primary",
            "AZURE_STORAGE_OUTPUT_CONTAINER": "fallback"
        }):
            config = AppConfig.from_env()
            self.assertEqual(config.vto_output_container, "primary")
    
    def test_jwt_access_secret_from_env(self):
        """Test jwt_access_secret configuration from environment."""
        with patch.dict(os.environ, {"JWT_ACCESS_SECRET": "my-secret-key"}):
            config = AppConfig.from_env()
            self.assertEqual(config.jwt_access_secret, "my-secret-key")
    
    def test_startup_background_preload_from_env(self):
        """Test startup_background_preload configuration from environment."""
        # Test disabling background preload
        with patch.dict(os.environ, {"STARTUP_BACKGROUND_PRELOAD": "0"}):
            config = AppConfig.from_env()
            self.assertFalse(config.startup_background_preload)
        
        # Test default (enabled)
        with patch.dict(os.environ, {"STARTUP_BACKGROUND_PRELOAD": "1"}):
            config = AppConfig.from_env()
            self.assertTrue(config.startup_background_preload)
    
    def test_wardrobe_results_from_env(self):
        """Test wardrobe results configuration from environment."""
        with patch.dict(os.environ, {
            "WARDROBE_RESULTS_ENABLED": "0",
            "WARDROBE_RESULTS_DIR": "/custom/path",
            "WARDROBE_RESULTS_SAVE_INPUT": "0",
            "WARDROBE_RESULTS_SAVE_OUTPUT": "0",
            "WARDROBE_RESULTS_SAVE_MASKS": "0",
        }):
            config = AppConfig.from_env()
            self.assertFalse(config.wardrobe_results_enabled)
            self.assertEqual(config.wardrobe_results_dir, "/custom/path")
            self.assertFalse(config.wardrobe_results_save_input)
            self.assertFalse(config.wardrobe_results_save_output)
            self.assertFalse(config.wardrobe_results_save_masks)
    
    def test_progress_sync_from_env(self):
        """Test progress sync configuration from environment."""
        with patch.dict(os.environ, {
            "ENABLE_WARDROBE_PROGRESS_SYNC": "1",
            "WARDROBE_PROGRESS_API_BASE_URL": "https://api.example.com",
            "WARDROBE_PROGRESS_SYNC_TIMEOUT_S": "30",
            "WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE": "1",
        }):
            config = AppConfig.from_env()
            self.assertTrue(config.enable_wardrobe_progress_sync)
            self.assertEqual(config.wardrobe_progress_api_base_url, "https://api.example.com")
            self.assertEqual(config.wardrobe_progress_sync_timeout_s, 30)
            self.assertTrue(config.wardrobe_progress_include_input_image)
    
    def test_progress_sync_timeout_minimum(self):
        """Test progress sync timeout minimum value enforcement."""
        # Test minimum value enforcement (should be at least 5)
        with patch.dict(os.environ, {"WARDROBE_PROGRESS_SYNC_TIMEOUT_S": "2"}):
            config = AppConfig.from_env()
            self.assertEqual(config.wardrobe_progress_sync_timeout_s, 5)
    
    def test_all_fields_from_env(self):
        """Test loading all fields from environment variables."""
        with patch.dict(os.environ, {
            "GPU_CONCURRENCY": "2",
            "REQUIRE_AZURE_UPLOAD": "1",
            "AZURE_STORAGE_VTO_OUTPUT_CONTAINER": "test-container",
            "JWT_ACCESS_SECRET": "test-secret",
            "STARTUP_BACKGROUND_PRELOAD": "0",
            "WARDROBE_RESULTS_ENABLED": "1",
            "WARDROBE_RESULTS_DIR": "/test/results",
            "WARDROBE_RESULTS_SAVE_INPUT": "1",
            "WARDROBE_RESULTS_SAVE_OUTPUT": "1",
            "WARDROBE_RESULTS_SAVE_MASKS": "0",
            "ENABLE_WARDROBE_PROGRESS_SYNC": "1",
            "WARDROBE_PROGRESS_API_BASE_URL": "https://test.api",
            "WARDROBE_PROGRESS_SYNC_TIMEOUT_S": "25",
            "WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE": "1",
        }):
            config = AppConfig.from_env()
            self.assertEqual(config.gpu_concurrency, 2)
            self.assertTrue(config.require_azure_upload)
            self.assertEqual(config.vto_output_container, "test-container")
            self.assertEqual(config.jwt_access_secret, "test-secret")
            self.assertFalse(config.startup_background_preload)
            self.assertTrue(config.wardrobe_results_enabled)
            self.assertEqual(config.wardrobe_results_dir, "/test/results")
            self.assertTrue(config.wardrobe_results_save_input)
            self.assertTrue(config.wardrobe_results_save_output)
            self.assertFalse(config.wardrobe_results_save_masks)
            self.assertTrue(config.enable_wardrobe_progress_sync)
            self.assertEqual(config.wardrobe_progress_api_base_url, "https://test.api")
            self.assertEqual(config.wardrobe_progress_sync_timeout_s, 25)
            self.assertTrue(config.wardrobe_progress_include_input_image)


class TestAppConfigFieldCount(unittest.TestCase):
    """Tests for AppConfig field coverage."""
    
    def test_field_count_matches_spec(self):
        """Test that AppConfig has the expected number of fields."""
        config = AppConfig.from_env()
        field_count = len(AppConfig.model_fields)
        
        # Design spec shows 14 fields for AppConfig
        self.assertEqual(field_count, 14, "Should have exactly 14 configuration fields")
    
    def test_all_fields_have_descriptions(self):
        """Test that all fields have description metadata."""
        for field_name, field_info in AppConfig.model_fields.items():
            self.assertIsNotNone(
                field_info.description,
                f"Field {field_name} should have a description"
            )


class TestAppConfigPydanticModel(unittest.TestCase):
    """Tests for AppConfig Pydantic model behavior."""
    
    def test_direct_instantiation(self):
        """Test creating AppConfig directly with values."""
        config = AppConfig(
            gpu_concurrency=3,
            require_azure_upload=True,
            vto_output_container="custom",
            jwt_access_secret="secret",
            startup_background_preload=False,
            wardrobe_results_enabled=True,
            wardrobe_results_dir="/path",
            wardrobe_results_save_input=True,
            wardrobe_results_save_output=True,
            wardrobe_results_save_masks=False,
            enable_wardrobe_progress_sync=True,
            wardrobe_progress_api_base_url="https://api.test",
            wardrobe_progress_sync_timeout_s=15,
            wardrobe_progress_include_input_image=True
        )
        self.assertEqual(config.gpu_concurrency, 3)
        self.assertTrue(config.require_azure_upload)
        self.assertEqual(config.vto_output_container, "custom")
        self.assertEqual(config.jwt_access_secret, "secret")
        self.assertFalse(config.startup_background_preload)
    
    def test_model_dump(self):
        """Test serializing AppConfig to dictionary."""
        config = AppConfig(
            gpu_concurrency=2,
            require_azure_upload=True
        )
        data = config.model_dump()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data["gpu_concurrency"], 2)
        self.assertEqual(data["require_azure_upload"], True)
    
    def test_model_validation(self):
        """Test that Pydantic validates field types."""
        # Valid values
        config = AppConfig(
            gpu_concurrency=1,
            require_azure_upload=False,
            vto_output_container="test",
            jwt_access_secret="key",
            startup_background_preload=True,
            wardrobe_results_enabled=True,
            wardrobe_results_dir="/path",
            wardrobe_results_save_input=True,
            wardrobe_results_save_output=True,
            wardrobe_results_save_masks=True,
            enable_wardrobe_progress_sync=False,
            wardrobe_progress_api_base_url="",
            wardrobe_progress_sync_timeout_s=20,
            wardrobe_progress_include_input_image=False
        )
        self.assertIsInstance(config.gpu_concurrency, int)
        self.assertIsInstance(config.require_azure_upload, bool)
        self.assertIsInstance(config.vto_output_container, str)
        self.assertIsInstance(config.jwt_access_secret, str)
        self.assertIsInstance(config.startup_background_preload, bool)


if __name__ == '__main__':
    unittest.main()
