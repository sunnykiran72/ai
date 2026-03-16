"""
Garment analysis route handlers.

This module provides endpoints for garment analysis and detection:
- /analyze: Main garment analysis endpoint
- /analzye: Typo endpoint for backward compatibility
- /v1/analyze/parser-joycaption: Parser-first analysis with JoyCaption

All endpoints delegate business logic to AnalyzeService.
"""

import logging
from typing import Optional, Union
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, Depends
from starlette.responses import Response
from pydantic import ValidationError

from .models import (
    AnalyzeRequest, AnalyzeResponse, AnalyzeSelectionRequest, 
    SelectionRequiredResponse, ParserJoyCaptionAnalyzeRequest
)
from services import AnalyzeService

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def get_analyze_service() -> AnalyzeService:
    """Dependency to get AnalyzeService instance."""
    from main import get_analyze_service as _get_analyze_service
    return _get_analyze_service()


@router.post("/analyze", response_model=Union[AnalyzeResponse, SelectionRequiredResponse])
@router.post("/analzye", response_model=Union[AnalyzeResponse, SelectionRequiredResponse])  # Typo compatibility
async def analyze_garment_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form(None, alias="type"),
    garmentType: Optional[str] = Form(None),
    use_parser_post_extract: Optional[bool] = Form(None),
    useParserPostExtract: Optional[bool] = Form(None),
    selected_index: Optional[int] = Form(None),
    debug: bool = Form(False),
    _authorization: Optional[str] = Header(None, alias="Authorization"),
    analyze_service: AnalyzeService = Depends(get_analyze_service)
) -> Union[AnalyzeResponse, SelectionRequiredResponse]:
    """
    Main garment analysis endpoint.
    
    Analyzes uploaded garment images to detect, extract, and digitize garments.
    Returns either complete analysis or selection_required response for multiple garments.
    
    Args:
        file: Uploaded image file (primary)
        image: Uploaded image file (alternative)
        garment_type: Expected garment type (top, bottom, dress, outer)
        garmentType: Alternative garment type parameter
        use_parser_post_extract: Enable parser post-extraction (deprecated)
        useParserPostExtract: Alternative parser parameter
        selected_index: Index of selected garment if multiple detected
        debug: Enable debug mode
        _authorization: Authorization header
        analyze_service: Injected AnalyzeService instance
        
    Returns:
        AnalyzeResponse with garment data or SelectionRequiredResponse for multi-garment images
        
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
        
        logger.info(f"Analyze request: type={effective_type}, selected_index={selected_index}")
        
        # Delegate to service layer
        result = await analyze_service.analyze_image(
            upload=upload,
            garment_type=effective_type,
            selected_index=selected_index,
            debug=debug,
            authorization=_authorization,
        )
        
        # Pass through legacy response objects (multipart/form-data)
        if isinstance(result, Response):
            return result

        # Handle selection required response for dict payloads
        if result.get("selection_required"):
            return SelectionRequiredResponse(
                status="selection_required",
                selection_required=True,
                candidates=result.get("candidates", []),
                total_candidates=result.get("total_candidates", 0),
                message="Multiple garments detected. Please select one."
            )
        
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


@router.post("/analyze-selection", response_model=AnalyzeResponse)
async def analyze_with_selection_endpoint(
    request: AnalyzeSelectionRequest,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    _authorization: Optional[str] = Header(None, alias="Authorization"),
    analyze_service: AnalyzeService = Depends(get_analyze_service)
) -> AnalyzeResponse:
    """
    Complete analysis after user selects a garment.
    
    Used when initial analysis returned selection_required response.
    Processes the selected garment and returns complete analysis.
    
    Args:
        request: Selection request with selected index
        file: Uploaded image file (primary)
        image: Uploaded image file (alternative)
        _authorization: Authorization header
        analyze_service: Injected AnalyzeService instance
        
    Returns:
        AnalyzeResponse with selected garment analysis
    """
    try:
        # Validate upload
        upload = file or image
        if upload is None:
            raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")
        
        logger.info(f"Analyze selection request: selected_index={request.selected_index}")
        
        # Delegate to service layer
        result = await analyze_service.analyze_with_selection(
            upload=upload,
            selected_index=request.selected_index,
            garment_type=request.garment_type,
            authorization=_authorization,
        )

        # Pass through legacy response objects (multipart/form-data)
        if isinstance(result, Response):
            return result
        
        return AnalyzeResponse(
            status="success",
            message="Selected garment analysis completed successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"Analyze selection validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Analyze selection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Selection analysis failed: {str(e)}")


@router.post("/v1/analyze/parser-joycaption")
async def parser_joycaption_analyze_endpoint(
    request: ParserJoyCaptionAnalyzeRequest,
    analyze_service: AnalyzeService = Depends(get_analyze_service)
):
    """
    Parser-first garment analysis with JoyCaption.
    
    Specialized analysis endpoint that uses human parser for garment detection
    followed by JoyCaption for detailed description generation.
    
    Args:
        request: Parser JoyCaption analysis request
        analyze_service: Injected AnalyzeService instance
        
    Returns:
        Analysis result with parser-based detection and JoyCaption descriptions
    """
    try:
        logger.info(f"Parser JoyCaption analyze request: image={request.image_url}")
        
        # Delegate to service layer
        result = await analyze_service.parser_joycaption_analyze(
            image_url=request.image_url,
            garment_type=request.garment_type,
            selected_index=request.selected_index,
            use_unified_square_split=request.use_unified_square_split,
            min_component_area_ratio=request.min_component_area_ratio,
            square_padding_ratio=request.square_padding_ratio,
            upload_candidate_previews=request.upload_candidate_previews,
            adaptive_rect_crop=request.adaptive_rect_crop,
            run_flux_garment_only=request.run_flux_garment_only,
            flux_steps=request.flux_steps,
            flux_seed=request.flux_seed,
            flux_extract_only=request.flux_extract_only,
            flux_extract_strict_safety=request.flux_extract_strict_safety,
        )

        # Legacy endpoint already returns JSON-compatible dict
        return result
        
    except ValidationError as e:
        logger.error(f"Parser JoyCaption validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Parser JoyCaption analyze failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parser analysis failed: {str(e)}")
