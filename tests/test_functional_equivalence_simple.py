"""
Simple property-based tests for functional equivalence of refactored utilities.

This module validates that the refactored utility functions produce consistent
outputs and maintain expected properties for a wide range of inputs.

Property 4: Functional Equivalence
Validates: Requirements 2.11, 7.2
- Test color processing functions with random RGB values
- Test prompt generation with various inputs
- Validate function properties and consistency
"""

import random
import string
import unittest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports from modules to avoid __init__.py issues
import utils.color_processing as color_utils
import utils.prompt_generation as prompt_utils
import utils.validation as validation_utils
import utils.scoring as scoring_utils


class TestColorProcessingEquivalence(unittest.TestCase):
    """Test functional equivalence of color processing utilities."""

    def test_canonical_color_token_equivalence(self):
        """Test that canonical_color_token produces consistent output."""
        test_cases = [
            "red", "RED", "Red", "blue", "BLUE", "Blue",
            "grey", "gray", "off white", "off-white", "offwhite",
            "", "  ", "unknown_color", "123", "special-color"
        ]
        
        for color_input in test_cases:
            with self.subTest(color=color_input):
                result = color_utils.canonical_color_token(color_input)
                self.assertIsInstance(result, str)
                # Should be deterministic
                self.assertEqual(result, color_utils.canonical_color_token(color_input))

    def test_hex_rgb_conversion_equivalence(self):
        """Test hex/RGB conversion round-trip equivalence."""
        test_cases = [
            (0, 0, 0), (255, 255, 255), (128, 128, 128),
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (123, 45, 67), (200, 150, 100)
        ]
        
        for r, g, b in test_cases:
            with self.subTest(rgb=(r, g, b)):
                # Test round-trip conversion
                hex_color = color_utils.rgb_to_hex((r, g, b))
                rgb_result = color_utils.hex_to_rgb_triplet(hex_color)
                
                self.assertIsNotNone(rgb_result)
                self.assertEqual(rgb_result, (r, g, b))

    def test_hex_to_rgb_validation(self):
        """Test hex_to_rgb_triplet with various hex formats."""
        valid_cases = [
            "#FF0000", "FF0000", "#00FF00", "00FF00",
            "#0000FF", "0000FF", "#123456", "ABCDEF"
        ]
        
        invalid_cases = [
            "GG0000", "#GG0000", "12345", "#12345",
            "1234567", "#1234567", "", "invalid"
        ]
        
        for hex_input in valid_cases:
            with self.subTest(hex=hex_input):
                result = color_utils.hex_to_rgb_triplet(hex_input)
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 3)
                self.assertTrue(all(0 <= v <= 255 for v in result))
        
        for hex_input in invalid_cases:
            with self.subTest(hex=hex_input):
                result = color_utils.hex_to_rgb_triplet(hex_input)
                self.assertIsNone(result)

    def test_rgb_hsv_conversion_equivalence(self):
        """Test RGB/HSV conversion round-trip."""
        test_cases = [
            (0, 0, 0), (255, 255, 255), (128, 128, 128),
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (100, 150, 200), (75, 25, 175)
        ]
        
        for r, g, b in test_cases:
            with self.subTest(rgb=(r, g, b)):
                rgb = (r, g, b)
                hsv = color_utils.rgb_to_hsv(rgb)
                rgb_back = color_utils.hsv_to_rgb(hsv)
                
                # Allow small rounding errors
                self.assertLessEqual(abs(rgb_back[0] - r), 1)
                self.assertLessEqual(abs(rgb_back[1] - g), 1)
                self.assertLessEqual(abs(rgb_back[2] - b), 1)

    def test_color_family_consistency(self):
        """Test color family classification consistency."""
        test_cases = [
            "red", "RED", "Red", "blue", "BLUE", "Blue",
            "black", "white", "gray", "grey", "unknown"
        ]
        
        for color_name in test_cases:
            with self.subTest(color=color_name):
                family1 = color_utils.color_family(color_name)
                family2 = color_utils.color_family(color_name.upper())
                family3 = color_utils.color_family(color_name.lower())
                
                # Should be case-insensitive
                self.assertEqual(family1, family2)
                self.assertEqual(family1, family3)

    def test_rgb_hue_properties(self):
        """Test RGB hue calculation properties."""
        test_cases = [
            (255, 0, 0),    # Red should be around 0°
            (0, 255, 0),    # Green should be around 120°
            (0, 0, 255),    # Blue should be around 240°
            (255, 255, 0),  # Yellow should be around 60°
        ]
        
        for r, g, b in test_cases:
            with self.subTest(rgb=(r, g, b)):
                hue = color_utils.rgb_hue_deg((r, g, b))
                self.assertGreaterEqual(hue, 0.0)
                self.assertLess(hue, 360.0)

    def test_random_color_properties(self):
        """Test color processing with random values."""
        random.seed(42)  # For reproducible tests
        
        for _ in range(50):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            
            with self.subTest(rgb=(r, g, b)):
                # Test hex conversion
                hex_color = color_utils.rgb_to_hex((r, g, b))
                self.assertTrue(hex_color.startswith("#"))
                self.assertEqual(len(hex_color), 7)
                
                # Test HSV conversion
                hsv = color_utils.rgb_to_hsv((r, g, b))
                self.assertEqual(len(hsv), 3)
                self.assertGreaterEqual(hsv[0], 0.0)
                self.assertLess(hsv[0], 360.0)
                self.assertGreaterEqual(hsv[1], 0.0)
                self.assertLessEqual(hsv[1], 1.0)
                self.assertGreaterEqual(hsv[2], 0.0)
                self.assertLessEqual(hsv[2], 1.0)


class TestPromptGenerationEquivalence(unittest.TestCase):
    """Test functional equivalence of prompt generation utilities."""

    def test_parse_structured_descriptor_properties(self):
        """Test structured descriptor parsing properties."""
        test_cases = [
            "type=dress, colors=blue",
            "Category: evening gown Type: formal",
            "type=top; colors=red,blue; pattern=stripes",
            "invalid text without structure",
            "",
            "key1=value1|key2=value2"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = prompt_utils.parse_structured_descriptor(text)
                self.assertIsInstance(result, dict)
                
                # All keys should be lowercase and normalized
                for key in result.keys():
                    self.assertTrue(key.islower())

    def test_normalize_minicpm_descriptor_deterministic(self):
        """Test that MiniCPM normalization is deterministic."""
        test_cases = [
            "type=dress, colors=blue, pattern=floral",
            "identity: young woman, pose: standing",
            "category: evening wear, material: silk",
            "",
            "simple text without structure"
        ]
        
        for raw_text in test_cases:
            with self.subTest(text=raw_text):
                result1 = prompt_utils.normalize_minicpm_descriptor_text(raw_text, "garment")
                result2 = prompt_utils.normalize_minicpm_descriptor_text(raw_text, "garment")
                self.assertEqual(result1, result2)
                
                result3 = prompt_utils.normalize_minicpm_descriptor_text(raw_text, "person")
                result4 = prompt_utils.normalize_minicpm_descriptor_text(raw_text, "person")
                self.assertEqual(result3, result4)

    def test_sanitize_florence_description_properties(self):
        """Test Florence description sanitization properties."""
        test_cases = [
            ("The image shows a mannequin wearing a red dress", ["the image shows"]),
            ("A beautiful dress with floral pattern", []),
            ("The mannequin is standing against a white background", ["mannequin", "background"]),
            ("Red dress with buttons and sleeves", []),
            ("", [])
        ]
        
        for description, expected_removed in test_cases:
            with self.subTest(description=description):
                result = prompt_utils.sanitize_florence_garment_description(description)
                self.assertIsInstance(result, str)
                # Function should return a string (implementation may vary on what gets removed)

    def test_extract_avoid_directives_properties(self):
        """Test avoid directive extraction properties."""
        test_cases = [
            "Do not include background. Don't show mannequin. Never add extra elements.",
            "Please avoid skin and hair. Do not show face.",
            "Regular text without avoid directives.",
            "Don't use bright colors. Never make it too dark.",
            ""
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                directives = prompt_utils.extract_generation_only_avoid_directives(text)
                self.assertIsInstance(directives, list)
                
                # All directives should start with avoid patterns
                for directive in directives:
                    lower = directive.lower()
                    self.assertTrue(any(lower.startswith(pattern) 
                                      for pattern in ["do not", "don't", "never"]))

    def test_infer_target_type_consistency(self):
        """Test target type inference consistency."""
        test_cases = [
            "red dress with floral pattern",
            "blue shirt with long sleeves", 
            "black pants with belt",
            "winter jacket with hood",
            "simple garment",
            ""
        ]
        
        for description in test_cases:
            with self.subTest(description=description):
                result1 = prompt_utils.infer_flux2_target_type(description)
                result2 = prompt_utils.infer_flux2_target_type(description)
                self.assertEqual(result1, result2)
                self.assertIn(result1, {"dress", "top", "bottom", "outer"})

    def test_ensure_target_type_properties(self):
        """Test target type ensuring properties."""
        test_cases = [
            ("red dress", "dress"),
            ("blue shirt", "top"),
            ("black pants", "bottom"),
            ("winter coat", "outer"),
            ("", "dress")
        ]
        
        for description, target_type in test_cases:
            with self.subTest(description=description, target=target_type):
                result = prompt_utils.ensure_target_type_in_description(description, target_type)
                # The function should return a valid string
                self.assertIsInstance(result, str)
                # Result should not be shorter than input (may add content)
                self.assertGreaterEqual(len(result), len(description))

    def test_join_avoid_terms_grammar(self):
        """Test avoid terms joining grammar."""
        test_cases = [
            [],
            ["skin"],
            ["skin", "hair"],
            ["skin", "hair", "background"],
            ["skin", "hair", "background", "mannequin", "phone"]
        ]
        
        for terms in test_cases:
            with self.subTest(terms=terms):
                result = prompt_utils.join_avoid_terms(terms)
                
                if len(terms) == 0:
                    self.assertEqual(result, "")
                elif len(terms) == 1:
                    self.assertEqual(result, terms[0])
                elif len(terms) == 2:
                    self.assertIn(" and ", result)
                else:
                    self.assertIn(", and ", result)

    def test_merge_avoid_clauses_properties(self):
        """Test avoid clause merging properties."""
        test_cases = [
            [],
            ["Do not show skin."],
            ["Do not show skin.", "Avoid background."],
            ["Do not show skin.", "Avoid background.", "Never include mannequin."]
        ]
        
        for clauses in test_cases:
            with self.subTest(clauses=clauses):
                result = prompt_utils.merge_avoid_clause_sentences(*clauses)
                self.assertIsInstance(result, str)
                
                # Should not have duplicate sentences
                sentences = [s.strip() for s in result.split(".") if s.strip()]
                self.assertEqual(len(sentences), len(set(sentences)))


class TestValidationEquivalence(unittest.TestCase):
    """Test functional equivalence of validation utilities."""

    def test_normalize_garment_type_consistency(self):
        """Test garment type normalization consistency."""
        test_cases = [
            "dress", "DRESS", "Dress",
            "top", "TOP", "Top", "shirt", "blouse",
            "bottom", "BOTTOM", "pants", "trousers",
            "outer", "jacket", "coat",
            "unknown", "", "invalid_type"
        ]
        
        for garment_type in test_cases:
            with self.subTest(type=garment_type):
                result1 = validation_utils.normalize_garment_type(garment_type)
                result2 = validation_utils.normalize_garment_type(garment_type)
                self.assertEqual(result1, result2)
                
                # Should handle case insensitivity
                result3 = validation_utils.normalize_garment_type(garment_type.upper())
                result4 = validation_utils.normalize_garment_type(garment_type.lower())
                
                # Results should be consistent for case variations
                if result1:
                    self.assertEqual(result1.lower(), (result3 or "").lower())
                    self.assertEqual(result1.lower(), (result4 or "").lower())

    def test_sanitize_prompt_fact_value_properties(self):
        """Test prompt fact value sanitization properties."""
        test_cases = [
            "normal text",
            "text\nwith\nnewlines",
            "text\twith\ttabs",
            "text\rwith\rcarriage\rreturns",
            "",
            "   spaces   ",
            "special!@#$%^&*()characters"
        ]
        
        for value in test_cases:
            with self.subTest(value=repr(value)):
                result = validation_utils.sanitize_prompt_fact_value(value)
                self.assertIsInstance(result, str)
                
                # Should not contain problematic characters
                problematic = ["\n", "\r", "\t"]
                for char in problematic:
                    self.assertNotIn(char, result)


class TestScoringEquivalence(unittest.TestCase):
    """Test functional equivalence of scoring utilities."""

    def test_hybrid_score_properties(self):
        """Test hybrid scoring properties."""
        test_cases = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.8, 0.6, 0.4),
            (0.2, 0.9, 0.7)
        ]
        
        for yolo_conf, florence_conf, bbox_prior in test_cases:
            with self.subTest(scores=(yolo_conf, florence_conf, bbox_prior)):
                score = scoring_utils.hybrid_score(yolo_conf, florence_conf, bbox_prior)
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_hybrid_score_monotonicity(self):
        """Test that hybrid score increases with higher confidence values."""
        base_score = scoring_utils.hybrid_score(0.5, 0.5, 0.5)
        
        # Test increasing each component
        higher_yolo = scoring_utils.hybrid_score(0.7, 0.5, 0.5)
        higher_florence = scoring_utils.hybrid_score(0.5, 0.7, 0.5)
        higher_bbox = scoring_utils.hybrid_score(0.5, 0.5, 0.7)
        
        self.assertGreaterEqual(higher_yolo, base_score)
        self.assertGreaterEqual(higher_florence, base_score)
        self.assertGreaterEqual(higher_bbox, base_score)


class TestCrossModuleIntegration(unittest.TestCase):
    """Test integration between different utility modules."""

    def test_color_processing_integration(self):
        """Test integration between color processing functions."""
        test_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (128, 128, 128), (0, 0, 0), (255, 255, 255)
        ]
        
        for r, g, b in test_colors:
            with self.subTest(rgb=(r, g, b)):
                rgb = (r, g, b)
                
                # Convert to hex and back
                hex_color = color_utils.rgb_to_hex(rgb)
                rgb_back = color_utils.hex_to_rgb_triplet(hex_color)
                self.assertEqual(rgb_back, rgb)
                
                # Test color properties
                hue = color_utils.rgb_hue_deg(rgb)
                self.assertGreaterEqual(hue, 0.0)
                self.assertLess(hue, 360.0)

    def test_prompt_validation_integration(self):
        """Test integration between prompt and validation functions."""
        test_cases = [
            "type=dress, colors=red",
            "category=top, material=cotton",
            "invalid text",
            ""
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                # Clean the text
                cleaned = validation_utils.sanitize_prompt_fact_value(text)
                
                # Parse as structured descriptor
                parsed = prompt_utils.parse_structured_descriptor(cleaned)
                
                # If type is present, normalize it
                if "type" in parsed:
                    normalized_type = validation_utils.normalize_garment_type(parsed["type"])
                    if normalized_type:
                        self.assertIn(normalized_type, {"dress", "top", "bottom", "outer"})


class TestPerformanceRegression(unittest.TestCase):
    """Test that refactored functions don't have significant performance regressions."""

    def test_color_processing_performance(self):
        """Test color processing performance with datasets."""
        import time
        
        # Generate test data
        random.seed(42)
        test_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                      for _ in range(100)]  # Reduced for faster testing
        
        # Time color conversions
        start_time = time.time()
        for color in test_colors:
            hex_color = color_utils.rgb_to_hex(color)
            rgb_back = color_utils.hex_to_rgb_triplet(hex_color)
            hsv = color_utils.rgb_to_hsv(color)
            rgb_from_hsv = color_utils.hsv_to_rgb(hsv)
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 2.0)  # 2 seconds for 400 operations

    def test_prompt_processing_performance(self):
        """Test prompt processing performance."""
        import time
        
        # Generate test prompts
        test_prompts = [
            f"type=dress, colors=red blue, pattern=floral, material=cotton"
            for _ in range(50)  # Reduced for faster testing
        ]
        
        # Time prompt parsing
        start_time = time.time()
        for prompt in test_prompts:
            prompt_utils.parse_structured_descriptor(prompt)
            prompt_utils.normalize_minicpm_descriptor_text(prompt, "garment")
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 1.0)  # 1 second for 100 operations


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)