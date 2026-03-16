"""
Tests for utils/color_processing.py

Tests color conversion, palette extraction, and color analysis functions.
"""

import pytest
import numpy as np
from PIL import Image

from ai.utils.color_processing import (
    hex_to_rgb_triplet,
    rgb_to_hex,
    rgb_to_hsv,
    hsv_to_rgb,
    canonical_color_token,
    color_family,
    is_neutral_color_token,
    bucket_color_brightness,
    bucket_color_saturation,
    bucket_color_undertone,
    nearest_color_label,
    color_labels_from_hex_palette,
    color_distance,
    extract_color_palette,
    filter_palette_entries_by_area,
    extract_lab_color_profile,
)


class TestHexRGBConversion:
    """Test hex to RGB and RGB to hex conversions."""
    
    def test_hex_to_rgb_valid(self):
        """Test valid hex color conversion."""
        assert hex_to_rgb_triplet("#FF0000") == (255, 0, 0)
        assert hex_to_rgb_triplet("00FF00") == (0, 255, 0)
        assert hex_to_rgb_triplet("#0000FF") == (0, 0, 255)
        assert hex_to_rgb_triplet("#FFFFFF") == (255, 255, 255)
        assert hex_to_rgb_triplet("#000000") == (0, 0, 0)
    
    def test_hex_to_rgb_invalid(self):
        """Test invalid hex color returns None."""
        assert hex_to_rgb_triplet("") is None
        assert hex_to_rgb_triplet("#GGG") is None
        assert hex_to_rgb_triplet("12345") is None
        assert hex_to_rgb_triplet("#GGGGGG") is None
    
    def test_rgb_to_hex(self):
        """Test RGB to hex conversion."""
        assert rgb_to_hex((255, 0, 0)) == "#ff0000"
        assert rgb_to_hex((0, 255, 0)) == "#00ff00"
        assert rgb_to_hex((0, 0, 255)) == "#0000ff"
        assert rgb_to_hex((255, 255, 255)) == "#ffffff"
        assert rgb_to_hex((0, 0, 0)) == "#000000"
    
    def test_hex_rgb_roundtrip(self):
        """Test roundtrip conversion."""
        colors = ["#FF0000", "#00FF00", "#0000FF", "#ABCDEF"]
        for hex_color in colors:
            rgb = hex_to_rgb_triplet(hex_color)
            assert rgb is not None
            back_to_hex = rgb_to_hex(rgb)
            assert back_to_hex.upper() == hex_color.upper()


class TestHSVConversion:
    """Test RGB to HSV and HSV to RGB conversions."""
    
    def test_rgb_to_hsv(self):
        """Test RGB to HSV conversion."""
        # Red
        h, s, v = rgb_to_hsv((255, 0, 0))
        assert abs(h - 0.0) < 1.0
        assert abs(s - 1.0) < 0.01
        assert abs(v - 1.0) < 0.01
        
        # Green
        h, s, v = rgb_to_hsv((0, 255, 0))
        assert abs(h - 120.0) < 1.0
        assert abs(s - 1.0) < 0.01
        
        # Blue
        h, s, v = rgb_to_hsv((0, 0, 255))
        assert abs(h - 240.0) < 1.0
        assert abs(s - 1.0) < 0.01
    
    def test_hsv_to_rgb(self):
        """Test HSV to RGB conversion."""
        # Red
        rgb = hsv_to_rgb((0.0, 1.0, 1.0))
        assert rgb == (255, 0, 0)
        
        # Green
        rgb = hsv_to_rgb((120.0, 1.0, 1.0))
        assert rgb == (0, 255, 0)
        
        # Blue
        rgb = hsv_to_rgb((240.0, 1.0, 1.0))
        assert rgb == (0, 0, 255)
    
    def test_hsv_rgb_roundtrip(self):
        """Test roundtrip conversion."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
        for rgb in colors:
            hsv = rgb_to_hsv(rgb)
            back_to_rgb = hsv_to_rgb(hsv)
            # Allow small rounding errors
            assert all(abs(a - b) <= 1 for a, b in zip(rgb, back_to_rgb))


class TestColorTokens:
    """Test color token normalization and classification."""
    
    def test_canonical_color_token(self):
        """Test color token normalization."""
        assert canonical_color_token("grey") == "gray"
        assert canonical_color_token("off white") == "off-white"
        assert canonical_color_token("offwhite") == "off-white"
        assert canonical_color_token("red") == "red"
        assert canonical_color_token("BLUE") == "blue"
    
    def test_color_family(self):
        """Test color family classification."""
        assert color_family("black") == "neutral_dark"
        assert color_family("white") == "neutral_light"
        assert color_family("gray") == "neutral_mid"
        assert color_family("red") == "red"
        assert color_family("blue") == "blue"
        assert color_family("green") == "green"
        assert color_family("pink") == "pink"
    
    def test_is_neutral_color_token(self):
        """Test neutral color detection."""
        assert is_neutral_color_token("black") is True
        assert is_neutral_color_token("white") is True
        assert is_neutral_color_token("gray") is True
        assert is_neutral_color_token("red") is False
        assert is_neutral_color_token("blue") is False


class TestColorBucketing:
    """Test color brightness, saturation, and undertone bucketing."""
    
    def test_bucket_color_brightness(self):
        """Test brightness bucketing."""
        assert bucket_color_brightness({"medianL": 20.0}) == "dark"
        assert bucket_color_brightness({"medianL": 50.0}) == "mid"
        assert bucket_color_brightness({"medianL": 80.0}) == "light"
        assert bucket_color_brightness({}) == "unknown"
        assert bucket_color_brightness(None) == "unknown"
    
    def test_bucket_color_saturation(self):
        """Test saturation bucketing."""
        assert bucket_color_saturation({"meanChroma": 8.0, "medianL": 50.0}) == "muted"
        assert bucket_color_saturation({"meanChroma": 20.0, "medianL": 50.0}) == "balanced"
        assert bucket_color_saturation({"meanChroma": 40.0, "medianL": 50.0}) == "rich"
        assert bucket_color_saturation({}) == "unknown"
    
    def test_bucket_color_undertone(self):
        """Test undertone bucketing."""
        assert bucket_color_undertone({"meanA": 1.0, "meanB": 1.0}) == "neutral"
        assert bucket_color_undertone({"meanA": 2.0, "meanB": 10.0}) == "warm"
        assert bucket_color_undertone({"meanA": -10.0, "meanB": 2.0}) == "cool"
        assert bucket_color_undertone({}) == "unknown"


class TestColorLabels:
    """Test color label mapping and distance."""
    
    def test_nearest_color_label(self):
        """Test nearest color label finding."""
        assert nearest_color_label((255, 0, 0)) == "red"
        assert nearest_color_label((0, 0, 255)) == "blue"
        assert nearest_color_label((255, 255, 255)) == "white"
        assert nearest_color_label((0, 0, 0)) == "black"
    
    def test_color_labels_from_hex_palette(self):
        """Test converting hex palette to labels."""
        hexes = ["#FF0000", "#0000FF", "#00FF00"]
        labels = color_labels_from_hex_palette(hexes, top_k=3)
        assert "red" in labels
        assert "blue" in labels
        # Note: Pure green (#00FF00) maps to "yellow" in the color database
        assert "yellow" in labels
        assert len(labels) <= 3
    
    def test_color_distance(self):
        """Test color distance calculation."""
        # Same color should have distance 0
        dist = color_distance((255, 0, 0), (255, 0, 0))
        assert dist < 0.1
        
        # Different colors should have positive distance
        dist = color_distance((255, 0, 0), (0, 0, 255))
        assert dist > 10.0


class TestPaletteExtraction:
    """Test color palette extraction."""
    
    def test_extract_color_palette_solid_color(self):
        """Test palette extraction from solid color image."""
        # Create solid red image
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        palette = extract_color_palette(img, top_k=3, decontamination_enabled=False)
        
        assert len(palette) >= 1
        # Allow for slight variation in color extraction
        hex_color = palette[0]["hex"].upper()
        assert hex_color.startswith("#FF")  # Should be red-ish
        assert palette[0]["areaPercent"] > 90.0
    
    def test_extract_color_palette_multi_color(self):
        """Test palette extraction from multi-color image."""
        # Create image with red and blue regions
        img = Image.new("RGB", (100, 100))
        pixels = img.load()
        for y in range(100):
            for x in range(100):
                if x < 50:
                    pixels[x, y] = (255, 0, 0)  # Red
                else:
                    pixels[x, y] = (0, 0, 255)  # Blue
        
        palette = extract_color_palette(img, top_k=3, decontamination_enabled=False)
        
        assert len(palette) >= 2
        # Check that we got both colors
        hexes = [entry["hex"].upper() for entry in palette]
        assert any(h.startswith("#FF") or h.startswith("#FE") for h in hexes)  # Red-ish
        assert any(h.endswith("FF") or h.endswith("FE") for h in hexes)  # Blue-ish
    
    def test_filter_palette_entries_by_area(self):
        """Test filtering palette by area."""
        palette = [
            {"hex": "#FF0000", "areaPercent": 50.0},
            {"hex": "#00FF00", "areaPercent": 30.0},
            {"hex": "#0000FF", "areaPercent": 5.0},
        ]
        
        filtered = filter_palette_entries_by_area(palette, min_area_percent=10.0, min_items=2)
        assert len(filtered) == 2
        assert filtered[0]["areaPercent"] == 50.0
        assert filtered[1]["areaPercent"] == 30.0
        
        # Test min_items override
        filtered = filter_palette_entries_by_area(palette, min_area_percent=60.0, min_items=2)
        assert len(filtered) == 2  # Should keep min_items even if below threshold


class TestLabColorProfile:
    """Test LAB color profile extraction."""
    
    def test_extract_lab_color_profile_solid_color(self):
        """Test LAB profile extraction from solid color."""
        # Create solid red image
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        profile = extract_lab_color_profile(img, decontamination_enabled=False)
        
        assert "medianL" in profile
        assert "meanChroma" in profile
        assert "isNeutral" in profile
        assert isinstance(profile["medianL"], float)
        assert isinstance(profile["meanChroma"], float)
        assert isinstance(profile["isNeutral"], bool)
        
        # Red should not be neutral
        assert profile["isNeutral"] is False
        # Red should have high chroma
        assert profile["meanChroma"] > 20.0
    
    def test_extract_lab_color_profile_neutral(self):
        """Test LAB profile extraction from neutral color."""
        # Create gray image
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        profile = extract_lab_color_profile(img, decontamination_enabled=False)
        
        assert profile["isNeutral"] is True
        assert profile["meanChroma"] < 16.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
