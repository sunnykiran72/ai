"""
Color processing configuration.

This module defines configuration for color processing including:
- Color context masking
- Semantic override settings

These settings control how color information is extracted and processed
from garment images, particularly for color locking and semantic analysis.
"""

import os
from pydantic import BaseModel, Field
from .base import env_bool


class ColorConfig(BaseModel):
    """
    Color processing configuration.
    
    This configuration manages settings for color extraction and processing
    in garment analysis and virtual try-on workflows.
    """
    
    context_disable_masking: bool = Field(
        default=False,
        description="Disable masking when building color context (use full image)"
    )
    garment_semantic_override_enabled: bool = Field(
        default=True,
        description="Enable semantic override for garment color classification"
    )
    
    @classmethod
    def from_env(cls) -> "ColorConfig":
        """
        Load configuration from environment variables.
        
        Environment variables:
        - COLOR_CONTEXT_DISABLE_MASKING: Set to "1" to disable masking
        - GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED: Set to "0" to disable semantic override
        
        Returns:
            ColorConfig instance populated from environment variables
        """
        return cls(
            context_disable_masking=env_bool("COLOR_CONTEXT_DISABLE_MASKING", "0"),
            garment_semantic_override_enabled=env_bool("GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED", "1"),
        )
