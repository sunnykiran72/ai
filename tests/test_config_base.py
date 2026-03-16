"""
Unit tests for config/base.py configuration utilities.

Tests the env_int, env_float, and env_bool helper functions
for parsing environment variables with type safety and fallback defaults.
"""

import os
import unittest
import logging
from unittest.mock import patch
import importlib.util


# Import base module directly to avoid config/__init__.py import issues
spec = importlib.util.spec_from_file_location("base", "config/base.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class TestEnvInt(unittest.TestCase):
    """Tests for env_int() helper function."""
    
    def test_valid_integer(self):
        """Test parsing valid integer from environment variable."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = base.env_int("TEST_INT", 0)
            self.assertEqual(result, 42)
    
    def test_invalid_integer_uses_default(self):
        """Test that invalid integer value uses default."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            result = base.env_int("TEST_INT", 99)
            self.assertEqual(result, 99)
    
    def test_missing_variable_uses_default(self):
        """Test that missing environment variable uses default."""
        result = base.env_int("NONEXISTENT_VAR_12345", 100)
        self.assertEqual(result, 100)
    
    def test_negative_integer(self):
        """Test parsing negative integer."""
        with patch.dict(os.environ, {"TEST_INT": "-42"}):
            result = base.env_int("TEST_INT", 0)
            self.assertEqual(result, -42)
    
    def test_zero(self):
        """Test parsing zero."""
        with patch.dict(os.environ, {"TEST_INT": "0"}):
            result = base.env_int("TEST_INT", 99)
            self.assertEqual(result, 0)


class TestEnvFloat(unittest.TestCase):
    """Tests for env_float() helper function."""
    
    def test_valid_float(self):
        """Test parsing valid float from environment variable."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = base.env_float("TEST_FLOAT", 0.0)
            self.assertEqual(result, 3.14)
    
    def test_invalid_float_uses_default(self):
        """Test that invalid float value uses default."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            result = base.env_float("TEST_FLOAT", 2.71)
            self.assertEqual(result, 2.71)
    
    def test_missing_variable_uses_default(self):
        """Test that missing environment variable uses default."""
        result = base.env_float("NONEXISTENT_VAR_12345", 1.0)
        self.assertEqual(result, 1.0)
    
    def test_negative_float(self):
        """Test parsing negative float."""
        with patch.dict(os.environ, {"TEST_FLOAT": "-3.14"}):
            result = base.env_float("TEST_FLOAT", 0.0)
            self.assertEqual(result, -3.14)
    
    def test_integer_as_float(self):
        """Test parsing integer value as float."""
        with patch.dict(os.environ, {"TEST_FLOAT": "42"}):
            result = base.env_float("TEST_FLOAT", 0.0)
            self.assertEqual(result, 42.0)


class TestEnvBool(unittest.TestCase):
    """Tests for env_bool() helper function."""
    
    def test_true_value(self):
        """Test parsing '1' as True."""
        with patch.dict(os.environ, {"TEST_BOOL": "1"}):
            result = base.env_bool("TEST_BOOL", "0")
            self.assertTrue(result)
    
    def test_false_value(self):
        """Test parsing '0' as False."""
        with patch.dict(os.environ, {"TEST_BOOL": "0"}):
            result = base.env_bool("TEST_BOOL", "1")
            self.assertFalse(result)
    
    def test_missing_variable_uses_default_false(self):
        """Test that missing variable with default '0' returns False."""
        result = base.env_bool("NONEXISTENT_VAR_12345", "0")
        self.assertFalse(result)
    
    def test_missing_variable_uses_default_true(self):
        """Test that missing variable with default '1' returns True."""
        result = base.env_bool("NONEXISTENT_VAR_12345", "1")
        self.assertTrue(result)
    
    def test_any_other_value_is_false(self):
        """Test that any value other than '1' is treated as False."""
        with patch.dict(os.environ, {"TEST_BOOL": "true"}):
            result = base.env_bool("TEST_BOOL", "0")
            self.assertFalse(result)
        
        with patch.dict(os.environ, {"TEST_BOOL": "yes"}):
            result = base.env_bool("TEST_BOOL", "0")
            self.assertFalse(result)
        
        with patch.dict(os.environ, {"TEST_BOOL": "2"}):
            result = base.env_bool("TEST_BOOL", "0")
            self.assertFalse(result)


class TestLogging(unittest.TestCase):
    """Tests for logging behavior."""
    
    def test_env_int_logs_warning_on_invalid_value(self):
        """Test that env_int logs warning for invalid values."""
        with self.assertLogs(base.logger, level='WARNING') as cm:
            with patch.dict(os.environ, {"TEST_INT": "invalid"}):
                result = base.env_int("TEST_INT", 42)
                self.assertEqual(result, 42)
                self.assertTrue(any("Invalid integer for TEST_INT='invalid'" in msg for msg in cm.output))
                self.assertTrue(any("Using 42" in msg for msg in cm.output))
    
    def test_env_float_logs_warning_on_invalid_value(self):
        """Test that env_float logs warning for invalid values."""
        with self.assertLogs(base.logger, level='WARNING') as cm:
            with patch.dict(os.environ, {"TEST_FLOAT": "invalid"}):
                result = base.env_float("TEST_FLOAT", 3.14)
                self.assertEqual(result, 3.14)
                self.assertTrue(any("Invalid float for TEST_FLOAT='invalid'" in msg for msg in cm.output))
                self.assertTrue(any("Using 3.14" in msg for msg in cm.output))


if __name__ == '__main__':
    unittest.main()
