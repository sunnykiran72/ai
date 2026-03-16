"""
Base configuration utilities.

This module provides helper functions for parsing environment variables
with type safety and fallback defaults.

Functions:
    env_int: Parse integer from environment variable
    env_float: Parse float from environment variable
    env_bool: Parse boolean from environment variable
"""

import os
import logging

logger = logging.getLogger("glamify-ai")


def env_int(name: str, default: int) -> int:
    """
    Parse integer from environment variable with fallback.
    
    Args:
        name: Environment variable name
        default: Default value if parsing fails
    
    Returns:
        Parsed integer value or default
    """
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid integer for {name}={raw!r}. Using {default}.")
        return default


def env_float(name: str, default: float) -> float:
    """
    Parse float from environment variable with fallback.
    
    Args:
        name: Environment variable name
        default: Default value if parsing fails
    
    Returns:
        Parsed float value or default
    """
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid float for {name}={raw!r}. Using {default}.")
        return default


def env_bool(name: str, default: str = "0") -> bool:
    """
    Parse boolean from environment variable.
    
    Args:
        name: Environment variable name
        default: Default value ("0" or "1")
    
    Returns:
        True if value is "1", False otherwise
    """
    return os.getenv(name, default) == "1"
