"""
Unit tests for config/color_config.py Color configuration.

Tests the ColorConfig Pydantic model and from_env() class method
for loading color processing configuration from environment variables.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.color_config import ColorConfig


class TestColorConfigDefaults(unittest.TestCase):
    """Tests for ColorConfig default values."""
    
    def test_from_env_defaults(self):
        """Test loading config with all default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = ColorConfig.from_env()
            
            # Default values
            self.assertFalse(config.context_disable_masking)
            self.assertTrue(config.garment_semantic_override_enabled)


class TestColorConfigEnvironmentVariables(unittest.TestCase):
    """Tests for ColorConfig loading from environment variables."""
    
    def test_context_disable_masking_from_env(self):
        """Test context_disable_masking configuration from environment."""
        # Test enabling masking disable
        with patch.dict(os.environ, {"COLOR_CONTEXT_DISABLE_MASKING": "1"}):
            config = ColorConfig.from_env()
            self.assertTrue(config.context_disable_masking)
        
        # Test default (masking enabled)
        with patch.dict(os.environ, {"COLOR_CONTEXT_DISABLE_MASKING": "0"}):
            config = ColorConfig.from_env()
            self.assertFalse(config.context_disable_masking)
    
    def test_garment_semantic_override_from_env(self):
        """Test garment_semantic_override_enabled configuration from environment."""
        # Test disabling semantic override
        with patch.dict(os.environ, {"GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED": "0"}):
            config = ColorConfig.from_env()
            self.assertFalse(config.garment_semantic_override_enabled)
        
        # Test default (semantic override enabled)
        with patch.dict(os.environ, {"GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED": "1"}):
            config = ColorConfig.from_env()
            self.assertTrue(config.garment_semantic_override_enabled)
    
    def test_all_fields_from_env(self):
        """Test loading all fields from environment variables."""
        with patch.dict(os.environ, {
            "COLOR_CONTEXT_DISABLE_MASKING": "1",
            "GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED": "0",
        }):
            config = ColorConfig.from_env()
            self.assertTrue(config.context_disable_masking)
            self.assertFalse(config.garment_semantic_override_enabled)


class TestColorConfigFieldCount(unittest.TestCase):
    """Tests for ColorConfig field coverage."""
    
    def test_field_count_matches_spec(self):
        """Test that ColorConfig has the expected number of fields."""
        config = ColorConfig.from_env()
        field_count = len(ColorConfig.model_fields)
        
        # Design spec shows 2 fields for ColorConfig
        self.assertEqual(field_count, 2, "Should have exactly 2 configuration fields")
    
    def test_all_fields_have_descriptions(self):
        """Test that all fields have description metadata."""
        for field_name, field_info in ColorConfig.model_fields.items():
            self.assertIsNotNone(
                field_info.description,
                f"Field {field_name} should have a description"
            )


class TestColorConfigPydanticModel(unittest.TestCase):
    """Tests for ColorConfig Pydantic model behavior."""
    
    def test_direct_instantiation(self):
        """Test creating ColorConfig directly with values."""
        config = ColorConfig(
            context_disable_masking=True,
            garment_semantic_override_enabled=False
        )
        self.assertTrue(config.context_disable_masking)
        self.assertFalse(config.garment_semantic_override_enabled)
    
    def test_model_dump(self):
        """Test serializing ColorConfig to dictionary."""
        config = ColorConfig(
            context_disable_masking=True,
            garment_semantic_override_enabled=False
        )
        data = config.model_dump()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data["context_disable_masking"], True)
        self.assertEqual(data["garment_semantic_override_enabled"], False)
    
    def test_model_validation(self):
        """Test that Pydantic validates field types."""
        # Valid boolean values
        config = ColorConfig(
            context_disable_masking=True,
            garment_semantic_override_enabled=False
        )
        self.assertIsInstance(config.context_disable_masking, bool)
        self.assertIsInstance(config.garment_semantic_override_enabled, bool)


if __name__ == '__main__':
    unittest.main()
