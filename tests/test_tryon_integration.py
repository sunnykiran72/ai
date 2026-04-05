"""
Integration tests for tryon endpoint.

This test suite verifies the tryon endpoint integration with service layer:
- /tryon endpoint functionality
- /v1/flux2/tryon endpoint functionality  
- /v1/flux/tryon legacy endpoint functionality
- Request/response handling
- Service integration
- Error scenarios
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from routes.models import TryonRequest, TryonResponse, Flux2TryonRequest, VTORequest
from routes.tryon import tryon_endpoint, flux2_tryon_endpoint, legacy_flux_tryon_endpoint


class TestTryonIntegration:
    """Integration tests for tryon endpoints."""
    
    @pytest.fixture
    def mock_tryon_service(self):
        """Mock TryonService for testing."""
        service = Mock()
        service.try_on = AsyncMock()
        return service
    
    @pytest.fixture
    def sample_tryon_request(self):
        """Sample valid tryon request."""
        return TryonRequest(
            user_image_url="https://example.com/user.jpg",
            garment_image_url="https://example.com/garment.jpg",
            garment_type="top",
            steps=20,
            seed=42
        )
    
    @pytest.fixture
    def sample_service_response(self):
        """Sample service response."""
        return {
            "output_url": "https://example.com/result.png",
            "latency": 2.5,
            "steps": 20,
            "seed": 42,
            "metadata": {
                "model": "flux2",
                "lora_enabled": True
            },
            "timings": {
                "total": 2.5,
                "generation": 2.0,
                "postprocess": 0.5
            }
        }
    
    @pytest.mark.asyncio
    async def test_tryon_endpoint_success(self, mock_tryon_service, sample_tryon_request, sample_service_response):
        """Test successful tryon endpoint execution."""
        # Configure mock service
        mock_tryon_service.try_on.return_value = sample_service_response
        
        # Call endpoint
        response = await tryon_endpoint(sample_tryon_request, mock_tryon_service)
        
        # Verify service was called correctly
        mock_tryon_service.try_on.assert_called_once_with(
            user_image_url=sample_tryon_request.user_image_url,
            garment_image_url=sample_tryon_request.garment_image_url,
            garment_type=sample_tryon_request.garment_type,
            prompt_description=sample_tryon_request.prompt_description,
            negative_prompt=sample_tryon_request.negative_prompt,
            steps=sample_tryon_request.steps,
            seed=sample_tryon_request.seed,
            use_second_pass=sample_tryon_request.use_second_pass,
            color_lock_enabled=sample_tryon_request.color_lock_enabled,
        )
        
        # Verify response structure
        assert isinstance(response, TryonResponse)
        assert response.status == "success"
        assert response.message == "Try-on completed successfully"
        assert response.data == sample_service_response
    
    @pytest.mark.asyncio
    async def test_tryon_endpoint_service_error(self, mock_tryon_service, sample_tryon_request):
        """Test tryon endpoint handles service errors correctly."""
        # Configure mock service to raise error
        mock_tryon_service.try_on.side_effect = ValueError("Invalid garment type")
        
        # Call endpoint and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await tryon_endpoint(sample_tryon_request, mock_tryon_service)
        
        # Verify error details
        assert exc_info.value.status_code == 500
        assert "Invalid garment type" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_flux2_tryon_endpoint_success(self, mock_tryon_service, sample_service_response):
        """Test successful Flux2 tryon endpoint execution."""
        # Create Flux2 request
        flux2_request = Flux2TryonRequest(
            user_image={"tryonImage": "https://example.com/user.jpg", "promptDescription": "portrait, soft studio"},
            products=[
                {
                    "image": "https://example.com/garment.jpg",
                    "promptDescription": "elegant evening dress",
                    "targetType": "dress",
                }
            ],
            steps=25,
            seed=123
        )
        
        # Configure mock service
        mock_tryon_service.try_on.return_value = sample_service_response
        
        # Call endpoint
        response = await flux2_tryon_endpoint(flux2_request, mock_tryon_service)
        
        # Verify service was called with Flux2 parameters
        mock_tryon_service.try_on.assert_called_once_with(
            user_image_url=flux2_request.user_image.tryonImage,
            user_prompt_description=flux2_request.user_image.promptDescription,
            source_worn_types=flux2_request.user_image.wornTypes,
            products=flux2_request.products,
            mode=flux2_request.mode,
            steps=flux2_request.steps,
            seed=flux2_request.seed,
        )
        
        # Verify response
        assert response.status == "success"
        assert response.message == "Flux2 try-on completed successfully"
        assert response.data == sample_service_response

    @pytest.mark.asyncio
    async def test_flux2_tryon_endpoint_forwards_guidance_and_lora_scale(self, mock_tryon_service, sample_service_response):
        """Test Flux2 try-on forwards supported generation overrides."""
        flux2_request = Flux2TryonRequest(
            user_image={"tryonImage": "https://example.com/user.jpg", "promptDescription": "portrait, soft studio"},
            products=[
                {
                    "image": "https://example.com/garment.jpg",
                    "promptDescription": "elegant evening dress",
                    "targetType": "dress",
                }
            ],
            steps=12,
            seed=99,
            guidanceScale=2.5,
            loraScale=0.9,
        )

        mock_tryon_service.try_on.return_value = sample_service_response

        response = await flux2_tryon_endpoint(flux2_request, mock_tryon_service)

        mock_tryon_service.try_on.assert_called_once_with(
            user_image_url=flux2_request.user_image.tryonImage,
            user_prompt_description=flux2_request.user_image.promptDescription,
            source_worn_types=flux2_request.user_image.wornTypes,
            products=flux2_request.products,
            mode=flux2_request.mode,
            steps=flux2_request.steps,
            seed=flux2_request.seed,
            guidance_scale=2.5,
            lora_scale=0.9,
        )
        assert response.status == "success"

    def test_flux2_tryon_request_default_mode(self):
        request = Flux2TryonRequest(
            user_image={"tryonImage": "https://example.com/user.jpg"},
            products=[
                {
                    "image": "https://example.com/garment.jpg",
                    "promptDescription": "elegant evening dress",
                }
            ],
        )
        assert request.mode == "tryon-lora"

    def test_flux2_tryon_request_invalid_mode(self):
        with pytest.raises(Exception):
            Flux2TryonRequest(
                user_image={"tryonImage": "https://example.com/user.jpg"},
                products=[
                    {
                        "image": "https://example.com/garment.jpg",
                        "promptDescription": "elegant evening dress",
                    }
                ],
                mode="invalid-mode",
            )

    def test_flux2_tryon_request_accepts_user_worn_types(self):
        request = Flux2TryonRequest(
            user_image={
                "tryonImage": "https://example.com/user.jpg",
                "wornTypes": ["dress", "outer"],
            },
            products=[
                {
                    "image": "https://example.com/garment.jpg",
                    "promptDescription": "structured cropped top",
                    "targetType": "top",
                }
            ],
        )
        assert request.user_image.wornTypes == ["dress", "outer"]

    def test_flux2_tryon_request_invalid_user_worn_types(self):
        with pytest.raises(Exception):
            Flux2TryonRequest(
                user_image={
                    "tryonImage": "https://example.com/user.jpg",
                    "wornTypes": ["invalid-type"],
                },
                products=[
                    {
                        "image": "https://example.com/garment.jpg",
                        "promptDescription": "structured cropped top",
                    }
                ],
            )
    
    @pytest.mark.asyncio
    async def test_legacy_flux_tryon_endpoint_success(self, mock_tryon_service, sample_service_response):
        """Test successful legacy VTO endpoint execution."""
        # Create legacy VTO request
        vto_request = VTORequest(
            user_image_url="https://example.com/user.jpg",
            garment_image_url="https://example.com/garment.jpg",
            garment_type="bottom",
            prompt="blue jeans with distressed details",
            negative_prompt="low quality, blurry",
            steps=15,
            seed=456
        )
        
        # Configure mock service
        mock_tryon_service.try_on.return_value = sample_service_response
        
        # Call endpoint
        response = await legacy_flux_tryon_endpoint(vto_request, mock_tryon_service)
        
        # Verify service was called with mapped parameters
        mock_tryon_service.try_on.assert_called_once_with(
            user_image_url=vto_request.user_image_url,
            garment_image_url=vto_request.garment_image_url,
            garment_type=vto_request.garment_type,
            prompt_description=vto_request.prompt,  # Legacy uses 'prompt' field
            negative_prompt=vto_request.negative_prompt,
            steps=vto_request.steps,
            seed=vto_request.seed,
            use_second_pass=None,  # Use service defaults
            color_lock_enabled=None,  # Use service defaults
        )
        
        # Verify response
        assert response.status == "success"
        assert response.message == "Legacy try-on completed successfully"
        assert response.data == sample_service_response
    
    @pytest.mark.asyncio
    async def test_tryon_endpoint_parameter_validation(self, mock_tryon_service):
        """Test that endpoint validates parameters correctly."""
        # Test with invalid steps
        with pytest.raises(Exception):  # Pydantic ValidationError
            TryonRequest(
                user_image_url="https://example.com/user.jpg",
                garment_image_url="https://example.com/garment.jpg",
                steps=-1  # Invalid: below minimum
            )
        
        # Test with invalid seed
        with pytest.raises(Exception):  # Pydantic ValidationError
            TryonRequest(
                user_image_url="https://example.com/user.jpg",
                garment_image_url="https://example.com/garment.jpg",
                seed=-1  # Invalid: below minimum
            )
        
        # Test with missing required fields
        with pytest.raises(Exception):  # Pydantic ValidationError
            TryonRequest(
                user_image_url="https://example.com/user.jpg"
                # Missing garment_image_url
            )
    
    @pytest.mark.asyncio
    async def test_tryon_endpoint_optional_parameters(self, mock_tryon_service, sample_service_response):
        """Test that optional parameters are handled correctly."""
        # Create minimal request with only required fields
        minimal_request = TryonRequest(
            user_image_url="https://example.com/user.jpg",
            garment_image_url="https://example.com/garment.jpg"
        )
        
        # Configure mock service
        mock_tryon_service.try_on.return_value = sample_service_response
        
        # Call endpoint
        response = await tryon_endpoint(minimal_request, mock_tryon_service)
        
        # Verify service was called with default values
        mock_tryon_service.try_on.assert_called_once_with(
            user_image_url=minimal_request.user_image_url,
            garment_image_url=minimal_request.garment_image_url,
            garment_type=None,  # Optional field
            prompt_description=None,  # Optional field
            negative_prompt=None,  # Optional field
            steps=20,  # Default value
            seed=42,   # Default value
            use_second_pass=None,  # Optional field
            color_lock_enabled=None,  # Optional field
        )
        
        # Verify response
        assert response.status == "success"
    
    @pytest.mark.asyncio
    async def test_tryon_endpoint_timeout_handling(self, mock_tryon_service, sample_tryon_request):
        """Test that endpoint handles service timeouts correctly."""
        # Configure mock service to raise timeout
        mock_tryon_service.try_on.side_effect = asyncio.TimeoutError("Service timeout")
        
        # Call endpoint and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await tryon_endpoint(sample_tryon_request, mock_tryon_service)
        
        # Verify error details
        assert exc_info.value.status_code == 500
        assert "Service timeout" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_tryon_endpoint_connection_error(self, mock_tryon_service, sample_tryon_request):
        """Test that endpoint handles connection errors correctly."""
        # Configure mock service to raise connection error
        mock_tryon_service.try_on.side_effect = ConnectionError("External service unavailable")
        
        # Call endpoint and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await tryon_endpoint(sample_tryon_request, mock_tryon_service)
        
        # Verify error details
        assert exc_info.value.status_code == 500
        assert "External service unavailable" in str(exc_info.value.detail)
    
    def test_tryon_request_model_validation(self):
        """Test TryonRequest model validation."""
        # Test valid request
        valid_request = TryonRequest(
            user_image_url="https://example.com/user.jpg",
            garment_image_url="https://example.com/garment.jpg",
            garment_type="top",
            steps=20,
            seed=42
        )
        
        assert valid_request.user_image_url == "https://example.com/user.jpg"
        assert valid_request.garment_image_url == "https://example.com/garment.jpg"
        assert valid_request.garment_type == "top"
        assert valid_request.steps == 20
        assert valid_request.seed == 42
    
    def test_tryon_response_model_structure(self, sample_service_response):
        """Test TryonResponse model structure."""
        response = TryonResponse(
            status="success",
            message="Try-on completed successfully",
            data=sample_service_response
        )
        
        assert response.status == "success"
        assert response.message == "Try-on completed successfully"
        assert response.data == sample_service_response
        assert "output_url" in response.data
        assert "latency" in response.data
        assert "metadata" in response.data
    
    @pytest.mark.asyncio
    async def test_service_dependency_injection(self, sample_tryon_request, sample_service_response):
        """Test that service dependency injection works correctly."""
        # This test would verify that the dependency injection system
        # properly provides the service instance to the endpoint
        
        # Mock the dependency function
        with patch('routes.tryon.get_tryon_service') as mock_get_service:
            mock_service = Mock()
            mock_service.try_on = AsyncMock(return_value=sample_service_response)
            mock_get_service.return_value = mock_service
            
            # Call endpoint with dependency
            response = await tryon_endpoint(sample_tryon_request, mock_service)
            
            # Verify service was used
            assert response.status == "success"
            mock_service.try_on.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_tryon_requests(self, mock_tryon_service, sample_service_response):
        """Test that multiple concurrent tryon requests are handled correctly."""
        # Configure mock service
        mock_tryon_service.try_on.return_value = sample_service_response
        
        # Create multiple requests
        requests = [
            TryonRequest(
                user_image_url=f"https://example.com/user{i}.jpg",
                garment_image_url=f"https://example.com/garment{i}.jpg",
                seed=i
            )
            for i in range(3)
        ]
        
        # Execute requests concurrently
        tasks = [
            tryon_endpoint(request, mock_tryon_service)
            for request in requests
        ]
        responses = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded
        assert len(responses) == 3
        for response in responses:
            assert response.status == "success"
        
        # Verify service was called for each request
        assert mock_tryon_service.try_on.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
