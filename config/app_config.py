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
    
    # User image prepare upscaling
    user_prep_upscale_enabled: bool = Field(
        default=False,
        description="Enable RealESRGAN-based upscaling for undersized user-image/prepare uploads"
    )
    user_prep_upscale_target_max_edge: int = Field(
        default=1024,
        description="Target max edge for upscaled prepare images"
    )
    user_prep_upscale_model_name: str = Field(
        default="RealESRGAN_x2plus",
        description="RealESRGAN model name used for user-image/prepare upscaling"
    )
    user_prep_upscale_model_path: str = Field(
        default="",
        description="Optional local model path override for prepare-image upscaling"
    )
    user_prep_upscale_tile: int = Field(
        default=0,
        description="Tile size for RealESRGAN user-image/prepare upscaling"
    )
    user_prep_upscale_tile_pad: int = Field(
        default=10,
        description="Tile padding for RealESRGAN user-image/prepare upscaling"
    )
    user_prep_upscale_pre_pad: int = Field(
        default=0,
        description="Pre-padding for RealESRGAN user-image/prepare upscaling"
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
        - USER_PREP_UPSCALE_ENABLED: Set to "1" to enable RealESRGAN upscaling for undersized prepare uploads
        - USER_PREP_UPSCALE_TARGET_MAX_EDGE: Target max edge for prepared uploads (default: 1024)
        - USER_PREP_UPSCALE_MODEL_NAME: RealESRGAN model name (default: RealESRGAN_x2plus)
        - USER_PREP_UPSCALE_MODEL_PATH: Optional local model path override
        - USER_PREP_UPSCALE_TILE: Tile size for RealESRGAN inference (default: 0)
        - USER_PREP_UPSCALE_TILE_PAD: Tile padding for RealESRGAN inference (default: 10)
        - USER_PREP_UPSCALE_PRE_PAD: Pre-padding for RealESRGAN inference (default: 0)
        
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
            user_prep_upscale_enabled=env_bool("USER_PREP_UPSCALE_ENABLED", "0"),
            user_prep_upscale_target_max_edge=max(512, env_int("USER_PREP_UPSCALE_TARGET_MAX_EDGE", 1024)),
            user_prep_upscale_model_name=os.getenv("USER_PREP_UPSCALE_MODEL_NAME", "RealESRGAN_x2plus").strip() or "RealESRGAN_x2plus",
            user_prep_upscale_model_path=os.getenv("USER_PREP_UPSCALE_MODEL_PATH", "").strip(),
            user_prep_upscale_tile=max(0, env_int("USER_PREP_UPSCALE_TILE", 0)),
            user_prep_upscale_tile_pad=max(0, env_int("USER_PREP_UPSCALE_TILE_PAD", 10)),
            user_prep_upscale_pre_pad=max(0, env_int("USER_PREP_UPSCALE_PRE_PAD", 0)),
        )
