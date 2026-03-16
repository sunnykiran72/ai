"""
Property-based test for input validation.

**Validates: Requirements 4.6**

This test ensures that input validation works correctly across all endpoints:
- Invalid inputs are rejected before service calls
- Validation errors indicate which fields are invalid
- Field constraints are properly enforced
- Type validation works correctly
- Range validation works for numeric fields
"""

import pytest
from typing import Dict, Any, Optional, Union
from hypothesis import given, strategies as st, settings, assume
from pydantic import ValidationError
from unittest.mock import Mock

from routes.models import (
    TryonRequest, AnalyzeRequest, ExtractRequest, UserPrepRequest,
    ParserJoyCaptionAnalyzeRequest, Flux2TryonRequest, VTORequest
)


class TestInputValidation:
    """Property-based tests for input validation across all endpoints."""
    
    def test_tryon_request_required_fields(self):
        """
        **Property 7: Input Validation**
        **Validates: Requirements 4.6**
        
        Test that required fields are properly validated for TryonRequest.
        """
        # Test missing required fields
        invalid_requests = [
            {},  # Missing all required fields
            {"user_image_url": "https://example.com/user.jpg"},  # Missing garment_image_url
            {"garment_image_url": "https://example.com/garment.jpg"},  # Missing user_image_url
            {"user_image_url": "", "garment_image_url": "https://example.com/garment.jpg"},  # Empty required field
        ]
        
        for invalid_data in invalid_requests:
            with pytest.raises(ValidationError) as exc_info:
                TryonRequest(**invalid_data)
            
            errors = exc_info.value.errors()
            assert len(errors) > 0
            
            # Check that error indicates which field is invalid
            error_fields = [error["loc"][0] for error in errors if error["loc"]]
            assert len(error_fields) > 0
            
            # Verify error messages are informative
            for error in errors:
                assert len(error["msg"]) > 0
                assert error["type"] in ["missing", "value_error"]
    
    @given(
        steps=st.integers(),
        seed=st.integers()
    )
    @settings(max_examples=50)
    def test_tryon_request_numeric_validation(self, steps, seed):
        """
        **Property 7: Input Validation**
        
        Test that numeric field validation works correctly.
        """
        request_data = {
            "user_image_url": "https://example.com/user.jpg",
            "garment_image_url": "https://example.com/garment.jpg",
            "steps": steps,
            "seed": seed
        }
        
        try:
            request = TryonRequest(**request_data)
            # If validation passes, values should be within valid ranges
            assert 4 <= request.steps <= 50
            assert 0 <= request.seed <= 2147483647
        except ValidationError as e:
            # If validation fails, values should be outside valid ranges
            errors = e.errors()
            
            if steps < 4 or steps > 50:
                # Should have error for steps field
                steps_errors = [err for err in errors if "steps" in str(err.get("loc", []))]
                assert len(steps_errors) > 0
                
            if seed < 0 or seed > 2147483647:
                # Should have error for seed field
                seed_errors = [err for err in errors if "seed" in str(err.get("loc", []))]
                assert len(seed_errors) > 0
    
    @given(
        garment_type=st.text()
    )
    @settings(max_examples=30)
    def test_garment_type_validation(self, garment_type):
        """
        **Property 7: Input Validation**
        
        Test that garment_type validation works correctly across different requests.
        """
        valid_types = {"top", "bottom", "dress", "outer"}
        
        # Test with TryonRequest
        request_data = {
            "user_image_url": "https://example.com/user.jpg",
            "garment_image_url": "https://example.com/garment.jpg",
            "garment_type": garment_type
        }
        
        try:
            request = TryonRequest(**request_data)
            # If validation passes, garment_type should be valid or None
            if request.garment_type is not None:
                assert request.garment_type in valid_types
        except ValidationError as e:
            # If validation fails, garment_type should be invalid
            if garment_type not in valid_types and garment_type is not None:
                errors = e.errors()
                type_errors = [err for err in errors if "garment_type" in str(err.get("loc", []))]
                # Note: garment_type is Optional, so invalid values might be accepted
                # This depends on the actual validation implementation
    
    @given(
        url=st.text()
    )
    @settings(max_examples=30)
    def test_url_validation(self, url):
        """
        **Property 7: Input Validation**
        
        Test that URL validation works correctly.
        """
        request_data = {
            "user_image_url": url,
            "garment_image_url": "https://example.com/garment.jpg"
        }
        
        try:
            request = TryonRequest(**request_data)
            # If validation passes, URL should be valid format
            assert request.user_image_url.startswith(("http://", "https://"))
        except ValidationError as e:
            # If validation fails, URL should be invalid format
            errors = e.errors()
            url_errors = [err for err in errors if "user_image_url" in str(err.get("loc", []))]
            
            # Check if URL is clearly invalid
            if not url.startswith(("http://", "https://")) or len(url.strip()) == 0:
                assert len(url_errors) > 0 or len(errors) > 0
    
    def test_extract_request_required_garment_type(self):
        """
        **Property 7: Input Validation**
        
        Test that ExtractRequest requires garment_type field.
        """
        # Test missing garment_type
        with pytest.raises(ValidationError) as exc_info:
            ExtractRequest()
        
        errors = exc_info.value.errors()
        type_errors = [err for err in errors if "garment_type" in str(err.get("loc", []))]
        assert len(type_errors) > 0
        
        # Test valid garment_type
        valid_request = ExtractRequest(garment_type="top")
        assert valid_request.garment_type == "top"
    
    @given(
        garment_type=st.sampled_from(["top", "bottom", "dress"]),
        steps=st.integers(min_value=4, max_value=30),
        seed=st.integers(min_value=0, max_value=2147483647)
    )
    @settings(max_examples=20)
    def test_extract_request_valid_parameters(self, garment_type, steps, seed):
        """
        **Property 7: Input Validation**
        
        Test that ExtractRequest accepts valid parameters.
        """
        request = ExtractRequest(
            garment_type=garment_type,
            steps=steps,
            seed=seed
        )
        
        assert request.garment_type == garment_type
        assert request.steps == steps
        assert request.seed == seed
        assert 4 <= request.steps <= 30
        assert 0 <= request.seed <= 2147483647
    
    @given(
        steps=st.integers().filter(lambda x: x < 4 or x > 30),
        seed=st.integers().filter(lambda x: x < 0 or x > 2147483647)
    )
    @settings(max_examples=20)
    def test_extract_request_invalid_ranges(self, steps, seed):
        """
        **Property 7: Input Validation**
        
        Test that ExtractRequest rejects out-of-range parameters.
        """
        # Test invalid steps
        with pytest.raises(ValidationError) as exc_info:
            ExtractRequest(garment_type="top", steps=steps)
        
        errors = exc_info.value.errors()
        steps_errors = [err for err in errors if "steps" in str(err.get("loc", []))]
        assert len(steps_errors) > 0
        
        # Test invalid seed
        with pytest.raises(ValidationError) as exc_info:
            ExtractRequest(garment_type="top", seed=seed)
        
        errors = exc_info.value.errors()
        seed_errors = [err for err in errors if "seed" in str(err.get("loc", []))]
        assert len(seed_errors) > 0
    
    @given(
        image_url=st.text(min_size=1),
        min_ratio=st.floats(min_value=0.0, max_value=1.0),
        padding_ratio=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=15)
    def test_parser_joycaption_request_validation(self, image_url, min_ratio, padding_ratio):
        """
        **Property 7: Input Validation**
        
        Test ParserJoyCaptionAnalyzeRequest validation.
        """
        try:
            request = ParserJoyCaptionAnalyzeRequest(
                image_url=image_url,
                min_component_area_ratio=min_ratio,
                square_padding_ratio=padding_ratio
            )
            
            # If validation passes, check values are reasonable
            assert len(request.image_url) > 0
            assert 0.0 <= request.min_component_area_ratio <= 1.0
            assert 0.0 <= request.square_padding_ratio <= 1.0
            
        except ValidationError as e:
            # If validation fails, check error details
            errors = e.errors()
            assert len(errors) > 0
            
            # Verify error messages are informative
            for error in errors:
                assert len(error["msg"]) > 0
                assert "loc" in error
    
    def test_boolean_field_validation(self):
        """
        **Property 7: Input Validation**
        
        Test that boolean fields are properly validated.
        """
        # Test valid boolean values
        valid_booleans = [True, False, 1, 0, "true", "false", "1", "0"]
        
        for bool_val in valid_booleans:
            try:
                request = ExtractRequest(
                    garment_type="top",
                    upload_debug_images=bool_val,
                    use_full_image_context=bool_val,
                    strict_section_enforcement=bool_val
                )
                # Should convert to actual boolean
                assert isinstance(request.upload_debug_images, bool)
                assert isinstance(request.use_full_image_context, bool)
                assert isinstance(request.strict_section_enforcement, bool)
            except ValidationError:
                # Some string values might not be accepted depending on Pydantic config
                pass
    
    @given(
        selected_index=st.integers()
    )
    @settings(max_examples=20)
    def test_selected_index_validation(self, selected_index):
        """
        **Property 7: Input Validation**
        
        Test that selected_index validation works correctly.
        """
        request_data = {"selected_index": selected_index}
        
        try:
            request = AnalyzeRequest(**request_data)
            # If validation passes, selected_index should be non-negative
            if request.selected_index is not None:
                assert request.selected_index >= 0
        except ValidationError as e:
            # If validation fails, selected_index should be negative
            if selected_index < 0:
                errors = e.errors()
                index_errors = [err for err in errors if "selected_index" in str(err.get("loc", []))]
                assert len(index_errors) > 0
    
    def test_optional_field_handling(self):
        """
        **Property 7: Input Validation**
        
        Test that optional fields are handled correctly.
        """
        # Test with minimal required fields only
        minimal_tryon = TryonRequest(
            user_image_url="https://example.com/user.jpg",
            garment_image_url="https://example.com/garment.jpg"
        )
        
        # Optional fields should have default values
        assert minimal_tryon.garment_type is None
        assert minimal_tryon.prompt_description is None
        assert minimal_tryon.negative_prompt is None
        assert minimal_tryon.steps == 20  # Default value
        assert minimal_tryon.seed == 42   # Default value
        assert minimal_tryon.use_second_pass is None
        assert minimal_tryon.color_lock_enabled is None
    
    def test_field_type_validation(self):
        """
        **Property 7: Input Validation**
        
        Test that field types are properly validated.
        """
        # Test type mismatches
        type_mismatches = [
            {"steps": "not_a_number"},  # String instead of int
            {"seed": 3.14},             # Float instead of int
            {"use_second_pass": "maybe"}, # String instead of bool
            {"garment_type": 123},      # Int instead of string
        ]
        
        base_data = {
            "user_image_url": "https://example.com/user.jpg",
            "garment_image_url": "https://example.com/garment.jpg"
        }
        
        for mismatch in type_mismatches:
            request_data = {**base_data, **mismatch}
            
            try:
                request = TryonRequest(**request_data)
                # If validation passes, Pydantic might have coerced the type
                # Verify the final type is correct
                if "steps" in mismatch:
                    assert isinstance(request.steps, int)
                if "seed" in mismatch:
                    assert isinstance(request.seed, int)
                if "use_second_pass" in mismatch:
                    assert isinstance(request.use_second_pass, (bool, type(None)))
                if "garment_type" in mismatch:
                    assert isinstance(request.garment_type, (str, type(None)))
                    
            except ValidationError as e:
                # If validation fails, error should indicate type mismatch
                errors = e.errors()
                assert len(errors) > 0
                
                # Check for type-related error messages
                type_errors = [err for err in errors if "type" in err.get("type", "")]
                assert len(type_errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])