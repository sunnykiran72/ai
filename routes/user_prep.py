"""
User image preparation route handlers.

This module provides endpoints for preparing user images for try-on:
- /v1/user-image/prepare: Main user preparation endpoint
- /v1/flux2/prepare-user-image: Alternative endpoint

Delegates business logic to UserImageService.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from starlette.responses import Response
from pydantic import ValidationError

from .models import UserPrepRequest, UserPrepResponse
from services import UserImageService

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def get_user_prep_service() -> UserImageService:
    """Dependency to get UserImageService instance."""
    from main import get_user_prep_service as _get_user_prep_service
    return _get_user_prep_service()


@router.post("/v1/user-image/prepare", response_model=UserPrepResponse)
@router.post("/v1/flux2/prepare-user-image", response_model=UserPrepResponse)
async def prepare_user_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    user_prep_service: UserImageService = Depends(get_user_prep_service)
) -> UserPrepResponse:
    """
    User image preparation endpoint.
    
    Prepares user images for virtual try-on by:
    1. Validating image quality and focus
    2. Detecting and cropping main person
    3. Removing background if required
    4. Generating user description
    5. Uploading processed image
    
    Args:
        file: Uploaded image file (primary)
        image: Uploaded image file (alternative)
        user_prep_service: Injected UserImageService instance
        
    Returns:
        UserPrepResponse with processed image URL and description
        
    Raises:
        HTTPException: For validation errors or service failures
    """
    try:
        # Validate upload
        upload = file or image
        if upload is None:
            raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")
        
        logger.info("User image preparation request")
        
        # Delegate to service layer
        result = await user_prep_service.prepare_user_image(upload=upload)

        # Pass through legacy response objects
        if isinstance(result, Response):
            return result
        
        return UserPrepResponse(
            status="success",
            message="User image prepared successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"User prep validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"User prep failed: {e}")
        raise HTTPException(status_code=500, detail=f"User preparation failed: {str(e)}")
