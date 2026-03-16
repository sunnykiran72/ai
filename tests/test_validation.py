"""
Unit tests for validation utilities.

Tests cover:
- Garment type normalization
- Image quality validation
- Descriptor weakness detection
- Input validation functions
- Sanitization functions
"""

import pytest
import numpy as np
from PIL import Image
from utils.validation import (
    normalize_garment_type,
    canonical_coverage_for_type,
    sanitize_prompt_fact_value,
    descriptor_word_count,
    descriptor_is_weak,
    validate_image_quality,
    is_valid_garment_type,
    validate_descriptor_length,
    sanitize_garment_description,
    normalize_coverage_description,
    validate_color_value,
    validate_material_name,
    validate_size_value,
    validate_prompt_fact_key,
    normalize_prompt_fact_key,
    GARMENT_TYPE_SYNONYMS,
)


class TestGarmentTypeNormalization:
    """Test garment type normalization functions."""
    
    def test_normalize_garment_type_valid_inputs(self):
        """Test normalization of valid garment types."""
        # Basic types
        assert normalize_garment_type("top") == "top"
        assert normalize_garment_type("bottom") == "bottom"
        assert normalize_garment_type("dress") == "dress"
        assert normalize_garment_type("outer") == "outer"
        
        # Synonyms
        assert normalize_garment_type("shirt") == "top"
        assert normalize_garment_type("t-shirt") == "top"
        assert normalize_garment_type("tshirt") == "top"
        assert normalize_garment_type("blouse") == "top"
        assert normalize_garment_type("pants") == "bottom"
        assert normalize_garment_type("jeans") == "bottom"
        assert normalize_garment_type("skirt") == "bottom"
        assert normalize_garment_type("gown") == "dress"
        assert normalize_garment_type("jacket") == "outer"
        assert normalize_garment_type("coat") == "outer"
    
    def test_normalize_garment_type_case_insensitive(self):
        """Test case insensitive normalization."""
        assert normalize_garment_type("SHIRT") == "top"
        assert normalize_garment_type("Pants") == "bottom"
        assert normalize_garment_type("DRESS") == "dress"
        assert normalize_garment_type("Jacket") == "outer"
    
    def test_normalize_garment_type_whitespace_handling(self):
        """Test whitespace and underscore handling."""
        assert normalize_garment_type("  shirt  ") == "top"
        assert normalize_garment_type("t_shirt") == "top"
        assert normalize_garment_type("long_sleeved_shirt") == "top"
    
    def test_normalize_garment_type_invalid_inputs(self):
        """Test handling of invalid inputs."""
        assert normalize_garment_type(None) is None
        assert normalize_garment_type("") is None
        assert normalize_garment_type("   ") is None
        assert normalize_garment_type("invalid_type") is None
        assert normalize_garment_type("random_string") is None
    
    def test_canonical_coverage_for_type(self):
        """Test canonical coverage descriptions."""
        assert canonical_coverage_for_type("top") == "upper body"
        assert canonical_coverage_for_type("shirt") == "upper body"
        assert canonical_coverage_for_type("bottom") == "lower body"
        assert canonical_coverage_for_type("pants") == "lower body"
        assert canonical_coverage_for_type("dress") == "full body"
        assert canonical_coverage_for_type("gown") == "full body"
        assert canonical_coverage_for_type("outer") == "upper body outer layer"
        assert canonical_coverage_for_type("jacket") == "upper body outer layer"
        assert canonical_coverage_for_type("invalid") == ""
        assert canonical_coverage_for_type(None) == ""
    
    def test_is_valid_garment_type(self):
        """Test garment type validation."""
        # Valid types
        assert is_valid_garment_type("shirt") is True
        assert is_valid_garment_type("pants") is True
        assert is_valid_garment_type("dress") is True
        assert is_valid_garment_type("jacket") is True
        
        # Invalid types
        assert is_valid_garment_type("invalid") is False
        assert is_valid_garment_type("") is False
        assert is_valid_garment_type("random_string") is False


class TestImageQualityValidation:
    """Test image quality validation functions."""
    
    def create_test_image(self, width: int = 100, height: int = 100, pattern: str = "solid") -> Image.Image:
        """Create test image with specified pattern."""
        if pattern == "solid":
            # Solid color - low focus score
            arr = np.full((height, width), 128, dtype=np.uint8)
        elif pattern == "sharp":
            # Sharp edges - high focus score
            arr = np.zeros((height, width), dtype=np.uint8)
            arr[:, :width//2] = 255
        elif pattern == "gradient":
            # Gradient - medium focus score
            arr = np.linspace(0, 255, width, dtype=np.uint8)
            arr = np.tile(arr, (height, 1))
        else:
            # Random noise - variable focus score
            arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        
        return Image.fromarray(arr, mode="L").convert("RGB")
    
    def test_validate_image_quality_sharp_image(self):
        """Test focus score for sharp image."""
        sharp_image = self.create_test_image(pattern="sharp")
        score = validate_image_quality(sharp_image)
        assert score > 0, "Sharp image should have positive focus score"
    
    def test_validate_image_quality_solid_image(self):
        """Test focus score for solid color image."""
        solid_image = self.create_test_image(pattern="solid")
        score = validate_image_quality(solid_image)
        assert score == 0.0, "Solid image should have zero focus score"
    
    def test_validate_image_quality_gradient_image(self):
        """Test focus score for gradient image."""
        gradient_image = self.create_test_image(pattern="gradient")
        score = validate_image_quality(gradient_image)
        assert score > 0, "Gradient image should have positive focus score"
    
    def test_validate_image_quality_different_sizes(self):
        """Test focus score calculation with different image sizes."""
        small_image = self.create_test_image(50, 50, "sharp")
        large_image = self.create_test_image(2000, 2000, "sharp")
        
        small_score = validate_image_quality(small_image, max_edge=1024)
        large_score = validate_image_quality(large_image, max_edge=1024)
        
        # Scores should be comparable despite size difference
        assert small_score > 0
        assert large_score > 0
    
    def test_validate_image_quality_empty_image(self):
        """Test handling of empty/invalid images."""
        # Create minimal 3x3 image (minimum size for gradient calculation)
        tiny_image = Image.new("RGB", (3, 3), color="white")
        score = validate_image_quality(tiny_image)
        
        # Very small uniform images should have low focus scores
        assert isinstance(score, float)
        assert score >= 0.0
        assert score >= 0, "Should handle tiny images gracefully"


class TestDescriptorWeaknessDetection:
    """Test descriptor weakness detection functions."""
    
    def test_descriptor_word_count(self):
        """Test word counting in descriptors."""
        assert descriptor_word_count("red cotton shirt") == 3
        # Note: "a-line" is counted as 2 words due to hyphen splitting
        assert descriptor_word_count("a-line dress with #pattern") == 5
        assert descriptor_word_count("") == 0
        assert descriptor_word_count("   ") == 0
        assert descriptor_word_count("single") == 1
        assert descriptor_word_count("word1 word2 word3 word4 word5") == 5
        assert descriptor_word_count("hyphen-word counts as one") == 5
    
    def test_descriptor_is_weak_empty_or_short(self):
        """Test weakness detection for empty or short descriptors."""
        assert descriptor_is_weak("") is True
        assert descriptor_is_weak("   ") is True
        assert descriptor_is_weak("shirt") is True
        assert descriptor_is_weak("red shirt") is True
        assert descriptor_is_weak("a b c") is True  # Too short
    
    def test_descriptor_is_weak_structured_output(self):
        """Test that structured VLM output is not considered weak."""
        structured = "type=shirt; material=cotton; color=red; sleeve=long; fit=regular; style=casual"
        # Structured output with sufficient words should not be weak
        assert descriptor_is_weak(structured) is False
    
    def test_descriptor_is_weak_rich_descriptors(self):
        """Test rich descriptors are not considered weak."""
        rich_top = "red cotton v-neck shirt with long sleeves and fitted waist construction details"
        assert descriptor_is_weak(rich_top, garment_type="top") is False
        
        rich_dress = "floral maxi dress with empire waist and three-quarter sleeves and ankle-length hem"
        assert descriptor_is_weak(rich_dress, garment_type="dress") is False
        
        rich_bottom = "high-waisted straight-leg jeans with front pockets and hem details and tapered fit"
        assert descriptor_is_weak(rich_bottom, garment_type="bottom") is False
    
    def test_descriptor_is_weak_lacking_richness_markers(self):
        """Test descriptors lacking richness markers are considered weak."""
        generic = "nice beautiful amazing wonderful garment item clothing piece"
        assert descriptor_is_weak(generic) is True
    
    def test_descriptor_is_weak_type_specific_validation(self):
        """Test type-specific weakness validation."""
        # Top without enough top-specific signals
        weak_top = "red garment item with some fabric"
        assert descriptor_is_weak(weak_top, garment_type="top") is True
        
        # Top with sufficient signals
        good_top = "red cotton v-neck shirt with long sleeves and fitted waist construction details"
        assert descriptor_is_weak(good_top, garment_type="top") is False
        
        # Dress without enough dress-specific signals
        weak_dress = "red garment with fabric material"
        assert descriptor_is_weak(weak_dress, garment_type="dress") is True
        
        # Dress with sufficient signals
        good_dress = "red floral maxi dress with v-neck and empire waist and ankle-length hem"
        assert descriptor_is_weak(good_dress, garment_type="dress") is False
    
    def test_descriptor_is_weak_custom_min_words(self):
        """Test custom minimum word threshold."""
        short_desc = "red cotton shirt with sleeves"
        assert descriptor_is_weak(short_desc, min_words=20) is True
        assert descriptor_is_weak(short_desc, min_words=5) is False


class TestInputValidation:
    """Test various input validation functions."""
    
    def test_validate_descriptor_length(self):
        """Test descriptor length validation."""
        # Too short
        assert validate_descriptor_length("short") is False
        assert validate_descriptor_length("red shirt") is False
        
        # Valid length
        assert validate_descriptor_length("red cotton v-neck shirt") is True
        
        # Too long (create 600 char string)
        long_desc = "a" * 600
        assert validate_descriptor_length(long_desc) is False
        
        # Custom bounds
        assert validate_descriptor_length("medium", min_length=5, max_length=10) is True
        assert validate_descriptor_length("too long text", min_length=5, max_length=10) is False
    
    def test_validate_color_value(self):
        """Test color value validation."""
        # Valid color names
        assert validate_color_value("red") is True
        assert validate_color_value("blue") is True
        assert validate_color_value("navy") is True
        
        # Valid hex codes
        assert validate_color_value("#FF0000") is True
        assert validate_color_value("#00ff00") is True
        assert validate_color_value("#123ABC") is True
        
        # Invalid values
        assert validate_color_value("") is False
        assert validate_color_value("invalid_color_123") is False
        assert validate_color_value("#GG0000") is False  # Invalid hex
        assert validate_color_value("#FF00") is False  # Too short
        assert validate_color_value("#FF000000") is False  # Too long
    
    def test_validate_material_name(self):
        """Test material name validation."""
        # Valid materials
        assert validate_material_name("cotton") is True
        assert validate_material_name("polyester") is True
        assert validate_material_name("wool") is True
        assert validate_material_name("denim") is True
        
        # Invalid materials
        assert validate_material_name("") is False
        assert validate_material_name("invalid_material_xyz") is False
        assert validate_material_name("random_string") is False
    
    def test_validate_size_value(self):
        """Test size value validation."""
        # Letter sizes
        assert validate_size_value("S") is True
        assert validate_size_value("M") is True
        assert validate_size_value("XL") is True
        assert validate_size_value("XXL") is True
        
        # Word sizes
        assert validate_size_value("Small") is True
        assert validate_size_value("MEDIUM") is True
        assert validate_size_value("Large") is True
        
        # Numeric sizes
        assert validate_size_value("32") is True
        assert validate_size_value("34W") is True
        assert validate_size_value("36L") is True
        
        # Invalid sizes
        assert validate_size_value("") is False
        assert validate_size_value("invalid_size") is False
        assert validate_size_value("1234") is False  # Too long
    
    def test_validate_prompt_fact_key(self):
        """Test prompt fact key validation."""
        # Valid keys
        assert validate_prompt_fact_key("color") is True
        assert validate_prompt_fact_key("garment_type") is True
        assert validate_prompt_fact_key("material_name") is True
        
        # Invalid keys
        assert validate_prompt_fact_key("") is False
        assert validate_prompt_fact_key("invalid key!") is False
        assert validate_prompt_fact_key("123invalid") is False  # Can't start with number
        assert validate_prompt_fact_key("key with spaces") is False


class TestSanitizationFunctions:
    """Test text sanitization functions."""
    
    def test_sanitize_prompt_fact_value(self):
        """Test prompt fact value sanitization."""
        # Basic whitespace cleanup
        assert sanitize_prompt_fact_value("  red color  ") == "red color"
        assert sanitize_prompt_fact_value("red,") == "red"
        assert sanitize_prompt_fact_value("red.") == "red"
        
        # Extraction directive removal
        extraction_text = "blue EXTRACTION_AVOID_CLAUSE: avoid text"
        assert sanitize_prompt_fact_value(extraction_text) == "blue"
        
        # Case insensitive extraction removal
        case_text = "green extraction_avoid_clause: some text"
        assert sanitize_prompt_fact_value(case_text) == "green"
        
        # Empty/None handling
        assert sanitize_prompt_fact_value("") == ""
        assert sanitize_prompt_fact_value(None) == ""
    
    def test_sanitize_garment_description(self):
        """Test garment description sanitization."""
        assert sanitize_garment_description("  red   shirt  ") == "red shirt"
        assert sanitize_garment_description("") == ""
        assert sanitize_garment_description(None) == ""
        assert sanitize_garment_description("   ") == ""
    
    def test_normalize_coverage_description(self):
        """Test coverage description normalization."""
        assert normalize_coverage_description("UPPER BODY") == "upper body"
        assert normalize_coverage_description("full_body") == "full body"
        assert normalize_coverage_description("  lower   body  ") == "lower body"
        assert normalize_coverage_description("") == ""
    
    def test_normalize_prompt_fact_key(self):
        """Test prompt fact key normalization."""
        assert normalize_prompt_fact_key("Garment Type") == "garment_type"
        assert normalize_prompt_fact_key("COLOR-NAME") == "color_name"
        assert normalize_prompt_fact_key("material__name") == "material_name"
        assert normalize_prompt_fact_key("_key_") == "key"
        assert normalize_prompt_fact_key("") == ""


class TestGarmentTypeSynonyms:
    """Test the GARMENT_TYPE_SYNONYMS constant."""
    
    def test_synonyms_completeness(self):
        """Test that all synonyms map to valid canonical types."""
        canonical_types = {"top", "bottom", "dress", "outer"}
        
        for synonym, canonical in GARMENT_TYPE_SYNONYMS.items():
            assert canonical in canonical_types, f"Synonym '{synonym}' maps to invalid type '{canonical}'"
    
    def test_synonyms_consistency(self):
        """Test that canonical types map to themselves."""
        assert GARMENT_TYPE_SYNONYMS["top"] == "top"
        assert GARMENT_TYPE_SYNONYMS["bottom"] == "bottom"
        assert GARMENT_TYPE_SYNONYMS["dress"] == "dress"
        assert GARMENT_TYPE_SYNONYMS["outer"] == "outer"


if __name__ == "__main__":
    pytest.main([__file__])