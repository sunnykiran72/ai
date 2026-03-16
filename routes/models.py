"""
Pydantic request and response models for API endpoints.

This module defines all request and response models used by the route handlers.
Models provide:
- Request validation with type checking
- Response structure consistency
- API documentation through schemas
- Field validation and defaults
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


# ── Base Models ──

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    status: str = Field(..., description="Response status")
    message: str = Field(default="", description="Response message")


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: str = Field(default="error", description="Error status")
    error: Dict[str, Any] = Field(..., description="Error details")


class SuccessResponse(BaseResponse):
    """Success response model."""
    status: str = Field(default="success", description="Success status")
    data: Dict[str, Any] = Field(..., description="Response data")


# ── Try-on Models ──

class TryonRequest(BaseModel):
    """Request model for virtual try-on endpoint."""
    user_image_url: str = Field(..., description="URL of user image")
    garment_image_url: str = Field(..., description="URL of garment image")
    garment_type: Optional[str] = Field(None, description="Type of garment (top, bottom, dress, outer)")
    prompt_description: Optional[str] = Field(None, description="Custom garment description")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt for generation")
    steps: int = Field(default=20, ge=4, le=50, description="Number of generation steps")
    seed: int = Field(default=42, ge=0, le=2147483647, description="Random seed for generation")
    use_second_pass: Optional[bool] = Field(None, description="Enable second pass refinement")
    color_lock_enabled: Optional[bool] = Field(None, description="Enable color preservation")

    @field_validator("user_image_url", "garment_image_url")
    @classmethod
    def _validate_required_url(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("URL must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("URL must not be empty")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return cleaned

    @field_validator("garment_type")
    @classmethod
    def _validate_garment_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("garment_type must be a string")
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if cleaned not in {"top", "bottom", "dress", "outer"}:
            raise ValueError("garment_type must be one of: top, bottom, dress, outer")
        return cleaned


class TryonResponse(SuccessResponse):
    """Response model for virtual try-on endpoint."""
    data: Dict[str, Any] = Field(
        ..., 
        description="Try-on result data including output URL, metadata, and timings"
    )


# ── Analyze Models ──

class AnalyzeRequest(BaseModel):
    """Request model for garment analysis endpoint."""
    garment_type: Optional[str] = Field(None, description="Expected garment type")
    selected_index: Optional[int] = Field(None, ge=0, description="Index of selected garment if multiple detected")
    debug: bool = Field(default=False, description="Enable debug mode")


class AnalyzeSelectionRequest(BaseModel):
    """Request model for analyze with selection."""
    selected_index: int = Field(..., ge=0, description="Index of selected garment")
    garment_type: Optional[str] = Field(None, description="Expected garment type")


class AnalyzeResponse(SuccessResponse):
    """Response model for garment analysis."""
    data: Dict[str, Any] = Field(
        ...,
        description="Analysis result including garment metadata, extracted image, and selection info"
    )


class SelectionRequiredResponse(BaseResponse):
    """Response when multiple garments detected and selection required."""
    status: str = Field(default="selection_required", description="Selection required status")
    selection_required: bool = Field(default=True, description="Indicates selection is required")
    candidates: List[Dict[str, Any]] = Field(..., description="List of detected garment candidates")
    total_candidates: int = Field(..., description="Total number of candidates")


# ── Extract Models ──

class ExtractRequest(BaseModel):
    """Request model for single garment extraction."""
    garment_type: str = Field(..., description="Type of garment to extract (top, bottom, dress)")
    prompt_description: Optional[str] = Field(None, description="Custom garment description")
    description_backend: Optional[str] = Field(None, description="Backend for description generation")
    negative_prompt: Optional[str] = Field(None, description="Custom negative prompt")
    steps: int = Field(default=10, ge=4, le=30, description="Number of generation steps")
    seed: int = Field(default=23, ge=0, le=2147483647, description="Random seed")
    upload_debug_images: bool = Field(default=False, description="Upload intermediate debug images")
    use_full_image_context: bool = Field(default=True, description="Use full image as context")
    strict_section_enforcement: bool = Field(default=True, description="Enforce strict section detection")
    use_parser_board_reference: bool = Field(default=False, description="Use parser for board reference")


class ExtractResponse(SuccessResponse):
    """Response model for garment extraction."""
    data: Dict[str, Any] = Field(
        ...,
        description="Extraction result including output URL, metadata, and processing details"
    )


# ── User Prep Models ──

class UserPrepRequest(BaseModel):
    """Request model for user image preparation."""
    # File upload handled by FastAPI, no additional fields needed
    pass


class UserPrepResponse(SuccessResponse):
    """Response model for user image preparation."""
    data: Dict[str, Any] = Field(
        ...,
        description="Prepared user image data including URL and description"
    )


# ── Health Models ──

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(default="ok", description="Health status")
    engine: str = Field(default="Glamify-AI-Unified", description="Engine name")
    ready: bool = Field(default=True, description="Service readiness")


class StatusResponse(BaseModel):
    """Response model for detailed status check."""
    status: str = Field(default="ok", description="Overall status")
    models: Dict[str, Any] = Field(..., description="Model loading status")
    config: Dict[str, Any] = Field(..., description="Configuration summary")
    startup: Dict[str, Any] = Field(..., description="Startup information")


# ── Parser JoyCaption Models (specialized analyze endpoint) ──

class ParserJoyCaptionAnalyzeRequest(BaseModel):
    """Request model for parser-joycaption analyze endpoint."""
    image_url: str = Field(..., description="URL of image to analyze")
    garment_type: Optional[str] = None
    selected_index: Optional[int] = Field(default=None, ge=0)
    use_unified_square_split: bool = True
    square_padding_ratio: float = Field(default=0.12, ge=0.0, le=0.60)
    min_component_area_ratio: float = Field(default=0.01, ge=0.0005, le=0.25)
    upload_candidate_previews: bool = True
    run_flux_garment_only: bool = False
    flux_steps: int = Field(default=8, ge=4, le=30)
    flux_seed: int = Field(default=23, ge=0, le=2147483647)
    flux_extract_only: bool = False
    flux_extract_strict_safety: bool = True
    adaptive_rect_crop: bool = False


# ── Flux2 and Legacy VTO Models ──

class Flux2TryonRequest(TryonRequest):
    """Request model for Flux2 try-on endpoint."""
    pass


class VTORequest(BaseModel):
    """Legacy VTO request model."""
    user_image_url: str = Field(..., description="User image URL")
    garment_image_url: str = Field(..., description="Garment image URL")
    garment_type: Optional[str] = Field(None, description="Type of garment (top, bottom, dress, outer)")
    prompt: Optional[str] = Field(None, description="Legacy prompt field")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt for generation")
    steps: int = Field(default=20, ge=4, le=50, description="Generation steps")
    seed: int = Field(default=42, ge=0, le=2147483647, description="Random seed")
