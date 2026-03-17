"""
Virtual try-on route handlers.

This module provides endpoints for virtual try-on functionality:
- /tryon: Main try-on endpoint
- /v1/flux2/tryon: Flux2-specific try-on endpoint  
- /v1/flux/tryon: Legacy Flux try-on endpoint

All endpoints delegate business logic to TryonService.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import ValidationError

from .models import TryonRequest, TryonResponse, Flux2TryonRequest, VTORequest
from services import TryonService

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def get_tryon_service() -> TryonService:
    """Dependency to get TryonService instance."""
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_tryon_service()


@router.post("/tryon", response_model=TryonResponse)
async def tryon_endpoint(
    request: TryonRequest,
    tryon_service: TryonService = Depends(get_tryon_service)
) -> TryonResponse:
    """
    Main virtual try-on endpoint.
    
    Performs virtual try-on by combining user image with garment image.
    Uses Flux2 model with LoRA for high-quality generation.
    
    Args:
        request: Try-on request with user/garment images and parameters
        tryon_service: Injected TryonService instance
        
    Returns:
        TryonResponse with generated image URL and metadata
        
    Raises:
        HTTPException: For validation errors or service failures
    """
    try:
        logger.info(f"Try-on request: user={request.user_image_url}, garment={request.garment_image_url}")
        
        # Delegate to service layer
        result = await tryon_service.try_on(
            user_image_url=request.user_image_url,
            garment_image_url=request.garment_image_url,
            garment_type=request.garment_type,
            prompt_description=request.prompt_description,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            seed=request.seed,
            use_second_pass=request.use_second_pass,
            color_lock_enabled=request.color_lock_enabled,
        )
        
        return TryonResponse(
            status="success",
            message="Try-on completed successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"Try-on validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Try-on failed: {e}")
        raise HTTPException(status_code=500, detail=f"Try-on failed: {str(e)}")


@router.post("/v1/flux2/tryon", response_model=TryonResponse)
async def flux2_tryon_endpoint(
    request: Flux2TryonRequest,
    tryon_service: TryonService = Depends(get_tryon_service)
) -> TryonResponse:
    """
    Flux2-specific virtual try-on endpoint.
    
    Enhanced try-on endpoint with Flux2-specific parameters and features.
    Provides more control over generation parameters and quality settings.
    
    Args:
        request: Flux2 try-on request with enhanced parameters
        tryon_service: Injected TryonService instance
        
    Returns:
        TryonResponse with generated image and detailed metadata
    """
    try:
        logger.info("Flux2 try-on request: products=%s", len(request.products))

        result = await tryon_service.try_on(
            user_image_url=request.user_image.tryonImage,
            user_prompt_description=request.user_image.promptDescription,
            products=request.products,
            steps=request.steps,
            seed=request.seed,
        )
        
        return TryonResponse(
            status="success",
            message="Flux2 try-on completed successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"Flux2 try-on validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Flux2 try-on failed: {e}")
        raise HTTPException(status_code=500, detail=f"Flux2 try-on failed: {str(e)}")


@router.post("/v1/flux/tryon", response_model=TryonResponse)
async def legacy_flux_tryon_endpoint(
    request: VTORequest,
    tryon_service: TryonService = Depends(get_tryon_service)
) -> TryonResponse:
    """
    Legacy Flux virtual try-on endpoint.
    
    Maintains backward compatibility with older VTO request format.
    Maps legacy parameters to current try-on service interface.
    
    Args:
        request: Legacy VTO request
        tryon_service: Injected TryonService instance
        
    Returns:
        TryonResponse with generated image and metadata
    """
    try:
        logger.info(f"Legacy VTO request: user={request.user_image_url}, garment={request.garment_image_url}")
        
        # Delegate to try-on service using legacy request fields
        result = await tryon_service.try_on(
            user_image_url=request.user_image_url,
            garment_image_url=request.garment_image_url,
            garment_type=request.garment_type,
            prompt_description=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            seed=request.seed,
            use_second_pass=None,
            color_lock_enabled=None,
        )
        
        return TryonResponse(
            status="success", 
            message="Legacy try-on completed successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"Legacy try-on validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legacy try-on failed: {e}")
        raise HTTPException(status_code=500, detail=f"Legacy try-on failed: {str(e)}")
