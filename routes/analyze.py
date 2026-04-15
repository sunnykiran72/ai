"""
Garment analysis route handlers.

This module provides the main garment analysis endpoint:
- /analyze: Main garment analysis endpoint
- /analzye: Typo endpoint for backward compatibility

All endpoints delegate business logic to AnalyzeService.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, Depends
from starlette.responses import Response
from pydantic import ValidationError

from .models import AnalyzeResponse
from services import AnalyzeService
from shared.response_payloads import json_response

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def get_analyze_service() -> AnalyzeService:
    """Dependency to get AnalyzeService instance."""
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_analyze_service()


@router.post("/analyze", response_model=AnalyzeResponse)
@router.post("/analzye", response_model=AnalyzeResponse)  # Typo compatibility
async def analyze_garment_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form(None, alias="type"),
    debug: bool = Form(False),
    _authorization: Optional[str] = Header(None, alias="Authorization"),
    analyze_service: AnalyzeService = Depends(get_analyze_service)
) -> AnalyzeResponse:
    """
    Main garment analysis endpoint.
    
    Analyzes uploaded garment images to detect, extract, and digitize garments.
    
    Args:
        file: Uploaded image file (required)
        garment_type: Expected garment type (top, bottom, dress, outer) - optional
        debug: Enable debug mode
        _authorization: Authorization header
        analyze_service: Injected AnalyzeService instance
        
    Returns:
        AnalyzeResponse with garment analysis data
        
    Raises:
        HTTPException: For validation errors or service failures
    """
    try:
        upload = file or image
        if upload is None:
            raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")

        logger.info(f"Analyze request: type={garment_type}")
        
        # Delegate to service layer
        result = await analyze_service.analyze_image(
            upload=upload,
            garment_type=garment_type,
            debug=debug,
            authorization=_authorization,
        )
        
        # Pass through legacy response objects (multipart/form-data)
        if isinstance(result, Response):
            return result

        # Legacy analyze payloads return a top-level status code and should bypass Pydantic wrapping
        if isinstance(result, dict) and isinstance(result.get("status"), int):
            return json_response(result)

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Analyze service returned an invalid response type")
        
        return AnalyzeResponse(
            status="success",
            message="Garment analysis completed successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"Analyze validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Analyze failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
