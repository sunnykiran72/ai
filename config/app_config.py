"""
General application configuration.

This module defines general application settings including:
- GPU concurrency
- Azure storage settings
- JWT authentication
- Startup preloading
- Wardrobe results storage
- Progress sync

These settings control core application behavior, resource management,
and integration with external services.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from .base import env_int, env_bool


class AppConfig(BaseModel):
    """
    General application configuration.
    
    This configuration manages core application settings including GPU
    resource management, Azure storage integration, authentication,
    startup behavior, and result persistence.
    """
    
    gpu_concurrency: int = Field(
        default=1,
        description="Maximum number of concurrent GPU operations (minimum 1)"
    )
    require_azure_upload: bool = Field(
        default=False,
        description="Require Azure storage upload for outputs"
    )
    vto_output_container: str = Field(
        default="wardrobe-outputs",
        description="Azure storage container name for virtual try-on outputs"
    )
    jwt_access_secret: str = Field(
        default="",
        description="Secret key for JWT token verification"
    )
    
    # Startup
    startup_background_preload: bool = Field(
        default=True,
        description="Preload AI models in background during startup"
    )
    
    # Wardrobe results
    wardrobe_results_enabled: bool = Field(
        default=True,
        description="Enable local storage of wardrobe processing results"
    )
    wardrobe_results_dir: str = Field(
        default="./wardrobe_results",
        description="Directory path for storing wardrobe results"
    )
    wardrobe_results_save_input: bool = Field(
        default=True,
        description="Save input images to wardrobe results directory"
    )
    wardrobe_results_save_output: bool = Field(
        default=True,
        description="Save output images to wardrobe results directory"
    )
    wardrobe_results_save_masks: bool = Field(
        default=True,
        description="Save mask images to wardrobe results directory"
    )
    
    # Progress sync
    enable_wardrobe_progress_sync: bool = Field(
        default=False,
        description="Enable progress synchronization with external API"
    )
    wardrobe_progress_api_base_url: str = Field(
        default="",
        description="Base URL for wardrobe progress API"
    )
    wardrobe_progress_sync_timeout_s: int = Field(
        default=20,
        description="Timeout in seconds for progress sync API calls"
    )
    wardrobe_progress_include_input_image: bool = Field(
        default=False,
        description="Include input image in progress sync payloads"
    )
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Load configuration from environment variables.
        
        Environment variables:
        - GPU_CONCURRENCY: Maximum concurrent GPU operations (default: 1)
        - REQUIRE_AZURE_UPLOAD: Set to "1" to require Azure upload (default: "0")
        - AZURE_STORAGE_VTO_OUTPUT_CONTAINER: Azure container for VTO outputs
        - AZURE_STORAGE_OUTPUT_CONTAINER: Fallback Azure container name
        - JWT_ACCESS_SECRET: JWT secret key for authentication
        - STARTUP_BACKGROUND_PRELOAD: Set to "0" to disable background preload (default: "1")
        - WARDROBE_RESULTS_ENABLED: Set to "0" to disable result storage (default: "1")
        - WARDROBE_RESULTS_DIR: Directory for storing results (default: "./wardrobe_results")
        - WARDROBE_RESULTS_SAVE_INPUT: Set to "0" to skip input storage (default: "1")
        - WARDROBE_RESULTS_SAVE_OUTPUT: Set to "0" to skip output storage (default: "1")
        - WARDROBE_RESULTS_SAVE_MASKS: Set to "0" to skip mask storage (default: "1")
        - ENABLE_WARDROBE_PROGRESS_SYNC: Set to "1" to enable progress sync (default: "0")
        - WARDROBE_PROGRESS_API_BASE_URL: Base URL for progress API
        - WARDROBE_PROGRESS_SYNC_TIMEOUT_S: Progress sync timeout (default: 20)
        - WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE: Set to "1" to include input (default: "0")
        
        Returns:
            AppConfig instance populated from environment variables
        """
        # Determine VTO output container with fallback logic
        vto_output_container = (
            os.getenv("AZURE_STORAGE_VTO_OUTPUT_CONTAINER")
            or os.getenv("AZURE_STORAGE_OUTPUT_CONTAINER", "wardrobe-outputs")
        )
        
        # Determine wardrobe results directory with fallback to parent directory
        default_results_dir = str(Path(__file__).resolve().parent.parent / "wardrobe_results")
        wardrobe_results_dir = os.getenv("WARDROBE_RESULTS_DIR", default_results_dir).strip()
        
        return cls(
            gpu_concurrency=max(1, env_int("GPU_CONCURRENCY", 1)),
            require_azure_upload=env_bool("REQUIRE_AZURE_UPLOAD", "0"),
            vto_output_container=vto_output_container,
            jwt_access_secret=os.getenv("JWT_ACCESS_SECRET", ""),
            startup_background_preload=env_bool("STARTUP_BACKGROUND_PRELOAD", "1"),
            wardrobe_results_enabled=env_bool("WARDROBE_RESULTS_ENABLED", "1"),
            wardrobe_results_dir=wardrobe_results_dir,
            wardrobe_results_save_input=env_bool("WARDROBE_RESULTS_SAVE_INPUT", "1"),
            wardrobe_results_save_output=env_bool("WARDROBE_RESULTS_SAVE_OUTPUT", "1"),
            wardrobe_results_save_masks=env_bool("WARDROBE_RESULTS_SAVE_MASKS", "1"),
            enable_wardrobe_progress_sync=env_bool("ENABLE_WARDROBE_PROGRESS_SYNC", "0"),
            wardrobe_progress_api_base_url=os.getenv("WARDROBE_PROGRESS_API_BASE_URL", "").strip(),
            wardrobe_progress_sync_timeout_s=max(5, env_int("WARDROBE_PROGRESS_SYNC_TIMEOUT_S", 20)),
            wardrobe_progress_include_input_image=env_bool("WARDROBE_PROGRESS_INCLUDE_INPUT_IMAGE", "0"),
        )
