"""
Integration tests for analyze endpoints.

This test suite verifies the analyze endpoint integration with service layer:
- /analyze endpoint functionality
- /analyze-selection endpoint functionality
- /v1/analyze/parser-joycaption endpoint functionality
- File upload handling
- Selection required scenarios
- Service integration
- Error scenarios
"""

import pytest
import asyncio
import io
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from fastapi import UploadFile, HTTPException

from routes.models import (
    AnalyzeRequest, AnalyzeResponse, AnalyzeSelectionRequest, 
    SelectionRequiredResponse, ParserJoyCaptionAnalyzeRequest
)
from routes.analyze import (
    analyze_garment_endpoint, analyze_with_selection_endpoint, 
    parser_joycaption_analyze_endpoint
)


class TestAnalyzeIntegration:
    """Integration tests for analyze endpoints."""
    
    @pytest.fixture
    def mock_analyze_service(self):
        """Mock AnalyzeService for testing."""
        service = Mock()
        service.analyze_image = AsyncMock()
        service.analyze_with_selection = AsyncMock()
        service.parser_joycaption_analyze = AsyncMock()
        return service
    
    @pytest.fixture
    def sample_upload_file(self):
        """Sample upload file for testing."""
        file_content = b"fake_image_data"
        file_obj = io.BytesIO(file_content)
        return UploadFile(
            file=file_obj,
            filename="test_garment.jpg",
        )
    
    @pytest.fixture
    def sample_analyze_response(self):
        """Sample successful analyze response."""
        return {
            "status": "success",
            "selected_item": {
                "type": "top",
                "confidence": 0.95,
                "bbox": [100, 100, 300, 400],
                "extracted_url": "https://example.com/extracted.png",
                "metadata": {
                    "color": "blue",
                    "pattern": "solid",
                    "material": "cotton"
                }
            },
            "timings": {
                "detection": 0.5,
                "extraction": 2.0,
                "total": 2.5
            }
        }
    
    @pytest.fixture
    def sample_selection_required_response(self):
        """Sample selection required response."""
        return {
            "selection_required": True,
            "candidates": [
                {
                    "index": 0,
                    "type": "top",
                    "confidence": 0.9,
                    "bbox": [50, 50, 200, 250],
                    "preview_url": "https://example.com/preview0.png"
                },
                {
                    "index": 1,
                    "type": "bottom",
                    "confidence": 0.8,
                    "bbox": [60, 250, 220, 450],
                    "preview_url": "https://example.com/preview1.png"
                }
            ],
            "total_candidates": 2
        }
    
    @pytest.mark.asyncio
    async def test_analyze_endpoint_success(self, mock_analyze_service, sample_upload_file, sample_analyze_response):
        """Test successful analyze endpoint execution."""
        # Configure mock service
        mock_analyze_service.analyze_image.return_value = sample_analyze_response
        
        # Call endpoint
        response = await analyze_garment_endpoint(
            file=sample_upload_file,
            image=None,
            garment_type="top",
            garmentType=None,
            use_parser_post_extract=None,
            useParserPostExtract=None,
            selected_index=None,
            debug=False,
            _authorization="Bearer test_token",
            analyze_service=mock_analyze_service
        )
        
        # Verify service was called correctly
        mock_analyze_service.analyze_image.assert_called_once_with(
            upload=sample_upload_file,
            garment_type="top",
            selected_index=None,
            debug=False,
            authorization="Bearer test_token"
        )
        
        # Verify response structure
        assert isinstance(response, AnalyzeResponse)
        assert response.status == "success"
        assert response.message == "Garment analysis completed successfully"
        assert response.data == sample_analyze_response
    
    @pytest.mark.asyncio
    async def test_analyze_endpoint_selection_required(self, mock_analyze_service, sample_upload_file, sample_selection_required_response):
        """Test analyze endpoint returns selection required response."""
        # Configure mock service to return selection required
        mock_analyze_service.analyze_image.return_value = sample_selection_required_response
        
        # Call endpoint
        response = await analyze_garment_endpoint(
            file=sample_upload_file,
            image=None,
            garment_type=None,
            garmentType=None,
            use_parser_post_extract=None,
            useParserPostExtract=None,
            selected_index=None,
            debug=False,
            _authorization=None,
            analyze_service=mock_analyze_service
        )
        
        # Verify response is selection required
        assert isinstance(response, SelectionRequiredResponse)
        assert response.status == "selection_required"
        assert response.selection_required is True
        assert len(response.candidates) == 2
        assert response.total_candidates == 2
        assert response.message == "Multiple garments detected. Please select one."
    
    @pytest.mark.asyncio
    async def test_analyze_endpoint_no_file_error(self, mock_analyze_service):
        """Test analyze endpoint handles missing file correctly."""
        # Call endpoint without file
        with pytest.raises(HTTPException) as exc_info:
            await analyze_garment_endpoint(
                file=None,
                image=None,
                garment_type=None,
                garmentType=None,
                use_parser_post_extract=None,
                useParserPostExtract=None,
                selected_index=None,
                debug=False,
                _authorization=None,
                analyze_service=mock_analyze_service
            )
        
        # Verify error details
        assert exc_info.value.status_code == 422
        assert "image file" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_analyze_endpoint_service_error(self, mock_analyze_service, sample_upload_file):
        """Test analyze endpoint handles service errors correctly."""
        # Configure mock service to raise error
        mock_analyze_service.analyze_image.side_effect = ValueError("Invalid image format")
        
        # Call endpoint and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await analyze_garment_endpoint(
                file=sample_upload_file,
                image=None,
                garment_type=None,
                garmentType=None,
                use_parser_post_extract=None,
                useParserPostExtract=None,
                selected_index=None,
                debug=False,
                _authorization=None,
                analyze_service=mock_analyze_service
            )
        
        # Verify error details
        assert exc_info.value.status_code == 500
        assert "Invalid image format" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_analyze_with_selection_endpoint_success(self, mock_analyze_service, sample_upload_file, sample_analyze_response):
        """Test successful analyze with selection endpoint execution."""
        # Create selection request
        selection_request = AnalyzeSelectionRequest(
            selected_index=1,
            garment_type="bottom"
        )
        
        # Configure mock service
        mock_analyze_service.analyze_with_selection.return_value = sample_analyze_response
        
        # Call endpoint
        response = await analyze_with_selection_endpoint(
            request=selection_request,
            file=sample_upload_file,
            image=None,
            _authorization="Bearer test_token",
            analyze_service=mock_analyze_service
        )
        
        # Verify service was called correctly
        mock_analyze_service.analyze_with_selection.assert_called_once_with(
            upload=sample_upload_file,
            selected_index=1,
            garment_type="bottom",
            authorization="Bearer test_token"
        )
        
        # Verify response
        assert isinstance(response, AnalyzeResponse)
        assert response.status == "success"
        assert response.message == "Selected garment analysis completed successfully"
        assert response.data == sample_analyze_response
    
    @pytest.mark.asyncio
    async def test_parser_joycaption_analyze_endpoint_success(self, mock_analyze_service):
        """Test successful parser JoyCaption analyze endpoint execution."""
        # Create parser JoyCaption request
        parser_request = ParserJoyCaptionAnalyzeRequest(
            image_url="https://example.com/garment.jpg",
            garment_type="dress",
            selected_index=None,
            use_unified_square_split=True,
            min_component_area_ratio=0.02,
            square_padding_ratio=0.15,
            upload_candidate_previews=True,
            adaptive_rect_crop=False,
            run_flux_garment_only=True,
            flux_steps=12,
            flux_seed=789,
            flux_extract_only=True,
            flux_extract_strict_safety=False
        )
        
        # Sample parser response
        parser_response = {
            "status": "success",
            "selection_required": False,
            "selected_item": {
                "type": "dress",
                "promptDescription": "elegant black evening dress with lace details",
                "preview_url": "https://example.com/preview.png",
                "fluxGarmentGeneration": {
                    "output_url": "https://example.com/generated.png",
                    "steps": 12,
                    "seed": 789
                }
            },
            "total_latency": 3.2
        }
        
        # Configure mock service
        mock_analyze_service.parser_joycaption_analyze.return_value = parser_response
        
        # Call endpoint
        response = await parser_joycaption_analyze_endpoint(
            request=parser_request,
            analyze_service=mock_analyze_service
        )
        
        # Verify service was called correctly
        mock_analyze_service.parser_joycaption_analyze.assert_called_once_with(
            image_url=parser_request.image_url,
            garment_type=parser_request.garment_type,
            selected_index=parser_request.selected_index,
            use_unified_square_split=parser_request.use_unified_square_split,
            min_component_area_ratio=parser_request.min_component_area_ratio,
            square_padding_ratio=parser_request.square_padding_ratio,
            upload_candidate_previews=parser_request.upload_candidate_previews,
            adaptive_rect_crop=parser_request.adaptive_rect_crop,
            run_flux_garment_only=parser_request.run_flux_garment_only,
            flux_steps=parser_request.flux_steps,
            flux_seed=parser_request.flux_seed,
            flux_extract_only=parser_request.flux_extract_only,
            flux_extract_strict_safety=parser_request.flux_extract_strict_safety
        )
        
        # Verify response
        assert response == parser_response
        assert response["status"] == "success"
        assert "selected_item" in response
    
    @pytest.mark.asyncio
    async def test_analyze_endpoint_parameter_normalization(self, mock_analyze_service, sample_upload_file, sample_analyze_response):
        """Test that endpoint normalizes parameters correctly."""
        # Configure mock service
        mock_analyze_service.analyze_image.return_value = sample_analyze_response
        
        # Call endpoint with alternative parameter names
        response = await analyze_garment_endpoint(
            file=None,
            image=sample_upload_file,  # Use 'image' instead of 'file'
            garment_type=None,
            garmentType="top",  # Use 'garmentType' instead of 'garment_type'
            use_parser_post_extract=None,
            useParserPostExtract=True,  # Use camelCase version
            selected_index=2,
            debug=True,
            _authorization="Bearer alt_token",
            analyze_service=mock_analyze_service
        )
        
        # Verify service was called with normalized parameters
        mock_analyze_service.analyze_image.assert_called_once_with(
            upload=sample_upload_file,  # Should use 'image' when 'file' is None
            garment_type="top",  # Should use 'garmentType' when 'garment_type' is None
            selected_index=2,
            debug=True,
            authorization="Bearer alt_token"
        )
        
        # Verify response
        assert response.status == "success"
    
    @pytest.mark.asyncio
    async def test_analyze_endpoint_with_selected_index(self, mock_analyze_service, sample_upload_file, sample_analyze_response):
        """Test analyze endpoint with pre-selected index."""
        # Configure mock service
        mock_analyze_service.analyze_image.return_value = sample_analyze_response
        
        # Call endpoint with selected index
        response = await analyze_garment_endpoint(
            file=sample_upload_file,
            image=None,
            garment_type="dress",
            garmentType=None,
            use_parser_post_extract=None,
            useParserPostExtract=None,
            selected_index=1,  # Pre-select garment
            debug=False,
            _authorization=None,
            analyze_service=mock_analyze_service
        )
        
        # Verify service was called with selected index
        mock_analyze_service.analyze_image.assert_called_once_with(
            upload=sample_upload_file,
            garment_type="dress",
            selected_index=1,
            debug=False,
            authorization=None
        )
        
        # Should return direct analysis (not selection required)
        assert isinstance(response, AnalyzeResponse)
        assert response.status == "success"
    
    @pytest.mark.asyncio
    async def test_analyze_endpoint_debug_mode(self, mock_analyze_service, sample_upload_file, sample_analyze_response):
        """Test analyze endpoint in debug mode."""
        # Add debug information to response
        debug_response = {
            **sample_analyze_response,
            "debug_info": {
                "detection_details": {"yolo_confidence": 0.95},
                "extraction_details": {"mask_quality": 0.88},
                "timing_breakdown": {"yolo": 0.3, "florence": 0.2, "extraction": 2.0}
            }
        }
        
        # Configure mock service
        mock_analyze_service.analyze_image.return_value = debug_response
        
        # Call endpoint with debug enabled
        response = await analyze_garment_endpoint(
            file=sample_upload_file,
            image=None,
            garment_type=None,
            garmentType=None,
            use_parser_post_extract=None,
            useParserPostExtract=None,
            selected_index=None,
            debug=True,  # Enable debug mode
            _authorization=None,
            analyze_service=mock_analyze_service
        )
        
        # Verify debug flag was passed to service
        mock_analyze_service.analyze_image.assert_called_once_with(
            upload=sample_upload_file,
            garment_type=None,
            selected_index=None,
            debug=True,
            authorization=None
        )
        
        # Verify debug information is included
        assert response.data["debug_info"] is not None
        assert "detection_details" in response.data["debug_info"]
    
    @pytest.mark.asyncio
    async def test_concurrent_analyze_requests(self, mock_analyze_service, sample_analyze_response):
        """Test that multiple concurrent analyze requests are handled correctly."""
        # Configure mock service
        mock_analyze_service.analyze_image.return_value = sample_analyze_response
        
        # Create multiple upload files
        uploads = [
            UploadFile(
                file=io.BytesIO(f"fake_data_{i}".encode()),
                filename=f"test{i}.jpg",
            )
            for i in range(3)
        ]
        
        # Execute requests concurrently
        tasks = [
            analyze_garment_endpoint(
                file=upload,
                image=None,
                garment_type=f"type_{i}",
                garmentType=None,
                use_parser_post_extract=None,
                useParserPostExtract=None,
                selected_index=None,
                debug=False,
                _authorization=None,
                analyze_service=mock_analyze_service
            )
            for i, upload in enumerate(uploads)
        ]
        responses = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded
        assert len(responses) == 3
        for response in responses:
            assert response.status == "success"
        
        # Verify service was called for each request
        assert mock_analyze_service.analyze_image.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
