"""
Single garment extraction route handlers.

This module provides endpoints for extracting individual garments:
- /v1/flux2/extract-garment: Main extraction endpoint

Delegates business logic to GarmentExtractionService.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from starlette.responses import Response
from pydantic import ValidationError

from .models import ExtractRequest, ExtractResponse
from services import GarmentExtractionService
from shared.response_payloads import json_response

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def get_extract_service() -> GarmentExtractionService:
    """Dependency to get GarmentExtractionService instance."""
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_extract_service()


@router.post("/v1/flux2/extract-garment", response_model=ExtractResponse)
async def extract_garment_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form(None, alias="type"),
    garmentType: Optional[str] = Form(None),
    prompt_description: Optional[str] = Form(None),
    promptDescription: Optional[str] = Form(None),
    description_backend: Optional[str] = Form(None),
    descriptionBackend: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    negativePrompt: Optional[str] = Form(None),
    steps: int = Form(10, ge=4, le=30),
    seed: int = Form(23, ge=0, le=2147483647),
    upload_debug_images: bool = Form(False),
    use_full_image_context: bool = Form(True),
    strict_section_enforcement: bool = Form(True),
    use_parser_board_reference: bool = Form(False),
    use_parser_post_extract: bool = Form(False),
    extract_service: GarmentExtractionService = Depends(get_extract_service)
) -> ExtractResponse:
    """
    Single garment extraction endpoint.
    
    Extracts and processes a single garment from an uploaded image:
    1. Accepts uploaded garment image
    2. Builds prompt from MiniCPM or provided description
    3. Runs Flux2 garment-only generation
    4. Returns 2:3 aspect ratio extracted garment
    
    Args:
        file: Uploaded image file (primary)
        image: Uploaded image file (alternative)
        garment_type: Type of garment to extract (top, bottom, dress)
        garmentType: Alternative garment type parameter
        prompt_description: Custom garment description
        promptDescription: Alternative description parameter
        description_backend: Backend for description generation
        descriptionBackend: Alternative backend parameter
        negative_prompt: Custom negative prompt
        negativePrompt: Alternative negative prompt parameter
        steps: Number of generation steps (4-30)
        seed: Random seed for generation
        upload_debug_images: Upload intermediate debug images
        use_full_image_context: Use full image as context
        strict_section_enforcement: Enforce strict section detection
        use_parser_board_reference: Use parser for board reference
        use_parser_post_extract: Use parser post-extraction
        extract_service: Injected GarmentExtractionService instance
        
    Returns:
        ExtractResponse with extracted garment URL and metadata
        
    Raises:
        HTTPException: For validation errors or service failures
    """
    try:
        # Validate upload
        upload = file or image
        if upload is None:
            raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")
        
        # Normalize parameters
        effective_type = garment_type or garmentType
        effective_description = prompt_description or promptDescription
        effective_backend = description_backend or descriptionBackend
        effective_negative = negative_prompt or negativePrompt
        
        if not effective_type:
            raise HTTPException(status_code=422, detail="garment_type is required")
        
        logger.info(f"Extract request: type={effective_type}, steps={steps}, seed={seed}")
        
        # Create request model for validation
        request = ExtractRequest(
            garment_type=effective_type,
            prompt_description=effective_description,
            description_backend=effective_backend,
            negative_prompt=effective_negative,
            steps=steps,
            seed=seed,
            upload_debug_images=upload_debug_images,
            use_full_image_context=use_full_image_context,
            strict_section_enforcement=strict_section_enforcement,
            use_parser_board_reference=use_parser_board_reference,
        )
        
        # Delegate to service layer
        result = await extract_service.extract_garment(
            upload=upload,
            garment_type=request.garment_type,
            steps=request.steps,
            seed=request.seed,
            backend=request.description_backend,
            prompt_description=request.prompt_description,
            negative_prompt=request.negative_prompt,
            upload_debug_images=request.upload_debug_images,
            use_full_image_context=request.use_full_image_context,
            strict_section_enforcement=request.strict_section_enforcement,
            use_parser_board_reference=request.use_parser_board_reference,
            use_parser_post_extract=use_parser_post_extract,
        )
        
        # Pass through legacy response objects
        if isinstance(result, Response):
            return result

        # Legacy extract payloads return a top-level status code and should bypass Pydantic wrapping
        if isinstance(result, dict) and isinstance(result.get("status"), int):
            return json_response(result)

        return ExtractResponse(
            status="success",
            message="Garment extraction completed successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"Extract validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Extract failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
