"""
Configuration management module.

This module provides type-safe configuration management using Pydantic models.
All environment variables are parsed and validated through configuration classes.

Exports:
    Config: Unified configuration object
    get_config: Singleton configuration accessor
    Flux2Config: Flux2 model configuration
    AnalyzeConfig: Analysis pipeline configuration
    MiniCPMConfig: MiniCPM service configuration
    ColorConfig: Color processing configuration
    AppConfig: General application configuration
"""

from typing import Optional
from pydantic import BaseModel
from .flux2_config import Flux2Config
from .analyze_config import AnalyzeConfig
from .minicpm_config import MiniCPMConfig
from .color_config import ColorConfig
from .app_config import AppConfig


class Config(BaseModel):
    """
    Unified application configuration.
    
    This class combines all configuration modules into a single configuration
    object that can be accessed throughout the application. Configuration is
    loaded from environment variables and validated using Pydantic models.
    
    Attributes:
        app: General application settings (GPU, Azure, JWT, startup, results)
        flux2: Flux2 model settings (backends, prompting, performance, color)
        analyze: Analysis pipeline settings (detection, extraction, processing)
        minicpm: MiniCPM service settings (URLs, timeouts, caching, prompts)
        color: Color processing settings (masking, semantic override)
    """
    
    app: AppConfig
    flux2: Flux2Config
    analyze: AnalyzeConfig
    minicpm: MiniCPMConfig
    color: ColorConfig
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Load all configuration from environment variables.
        
        This method creates a complete configuration object by loading each
        configuration module from its respective environment variables. All
        values are validated and defaults are applied where appropriate.
        
        Returns:
            Config instance with all settings loaded from environment
        """
        return cls(
            app=AppConfig.from_env(),
            flux2=Flux2Config.from_env(),
            analyze=AnalyzeConfig.from_env(),
            minicpm=MiniCPMConfig.from_env(),
            color=ColorConfig.from_env(),
        )


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get or create the singleton configuration instance.
    
    This function implements the singleton pattern to ensure configuration
    is loaded only once during application lifetime. Subsequent calls return
    the cached configuration instance.
    
    Returns:
        Config instance (created on first call, cached thereafter)
    
    Example:
        >>> from config import get_config
        >>> config = get_config()
        >>> print(config.app.gpu_concurrency)
        1
        >>> print(config.flux2.descriptor_backend)
        'minicpm_service'
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


__all__ = [
    "Config",
    "get_config",
    "Flux2Config",
    "AnalyzeConfig",
    "MiniCPMConfig",
    "ColorConfig",
    "AppConfig",
]
