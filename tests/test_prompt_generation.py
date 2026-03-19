"""
Tests for utils/prompt_generation.py

Tests structured descriptor parsing, prompt fact extraction, and avoid clause generation.
"""

import pytest
from typing import Dict, List

from utils.prompt_generation import (
    parse_structured_descriptor,
    extract_prompt_fact_segments,
    extract_generation_only_avoid_directives,
    merge_avoid_clause_sentences,
    serialize_prompt_fact_segments,
    enrich_garment_descriptor,
    infer_flux2_target_type,
    ensure_target_type_in_description,
    clean_prompt_section_text,
    join_avoid_terms,
    build_garment_prompt_natural,
)
from config.prompts import get_analyze_flux2_positive_prompt


class TestStructuredDescriptorParsing:
    """Test structured descriptor parsing functionality."""
    
    def test_parse_empty_input(self):
        """Test parsing empty or None input."""
        assert parse_structured_descriptor("") == {}
        assert parse_structured_descriptor(None) == {}
        assert parse_structured_descriptor("   ") == {}
    
    def test_parse_key_value_equals(self):
        """Test parsing key=value format."""
        result = parse_structured_descriptor("type=dress; colors=blue")
        assert result == {"type": "dress", "colors": "blue"}
        
        result = parse_structured_descriptor("category=evening; material=silk")
        assert result == {"category": "evening", "material": "silk"}
    
    def test_parse_key_value_colon(self):
        """Test parsing key: value format."""
        result = parse_structured_descriptor("Category: evening gown Type: formal")
        assert result == {"category": "evening gown", "type": "formal"}
        
        result = parse_structured_descriptor("colors: red and blue; pattern: floral")
        assert result == {"colors": "red and blue", "pattern": "floral"}
    
    def test_parse_key_value_brackets(self):
        """Test parsing key[value] format."""
        result = parse_structured_descriptor("type[dress]; colors[blue and white]")
        assert result == {"type": "dress", "colors": "blue and white"}
    
    def test_parse_mixed_formats(self):
        """Test parsing mixed key-value formats."""
        result = parse_structured_descriptor("type=dress; colors: blue; pattern[floral]")
        assert result == {"type": "dress", "colors": "blue", "pattern": "floral"}
    
    def test_parse_with_markdown_cleanup(self):
        """Test parsing with markdown formatting removal."""
        result = parse_structured_descriptor("**type**=`dress`; **colors**=`blue`")
        assert result == {"type": "dress", "colors": "blue"}
    
    def test_parse_with_whitespace_normalization(self):
        """Test parsing with whitespace normalization."""
        result = parse_structured_descriptor("type   =   dress  ;   colors   =   blue  ")
        assert result == {"type": "dress", "colors": "blue"}
    
    def test_parse_with_known_label_injection(self):
        """Test parsing with automatic separator injection for known labels."""
        result = parse_structured_descriptor("Category: dress Type: evening gown Colors: beige")
        assert result == {"category": "dress", "type": "evening gown", "colors": "beige"}
    
    def test_parse_key_aliases(self):
        """Test parsing with key aliases."""
        result = parse_structured_descriptor("bodypose: standing; framinglighting: soft")
        assert result == {"body_pose": "standing", "framing_lighting": "soft"}
        
        result = parse_structured_descriptor("currentoutfit: casual")
        assert result == {"current_outfit": "casual"}
    
    def test_parse_value_cleanup(self):
        """Test parsing with value cleanup (brackets, markdown, etc.)."""
        result = parse_structured_descriptor("type=<dress>; colors=[blue]; pattern=(floral)")
        assert result == {"type": "dress", "colors": "blue", "pattern": "floral"}
        
        result = parse_structured_descriptor("type=*dress*; colors=- blue")
        assert result == {"type": "dress", "colors": "blue"}
    
    def test_parse_invalid_segments(self):
        """Test parsing ignores invalid segments."""
        result = parse_structured_descriptor("valid=value; invalid_segment; another=good")
        assert result == {"valid": "value", "another": "good"}
    
    def test_parse_key_length_limits(self):
        """Test parsing respects key length limits."""
        long_key = "a" * 50  # Too long
        result = parse_structured_descriptor(f"{long_key}=value")
        assert result == {}
        
        valid_key = "a" * 30  # Within limit
        result = parse_structured_descriptor(f"{valid_key}=value")
        assert result == {valid_key: "value"}


class TestPromptFactExtraction:
    """Test prompt fact extraction functionality."""
    
    def test_extract_empty_input(self):
        """Test extracting from empty input."""
        assert extract_prompt_fact_segments("") == {}
        assert extract_prompt_fact_segments(None) == {}
        assert extract_prompt_fact_segments("   ") == {}
    
    def test_extract_structured_facts(self):
        """Test extracting facts from structured input."""
        text = "category=dress; type=evening; colors=blue and gold"
        result = extract_prompt_fact_segments(text)
        
        assert result["category"] == "dress"
        assert result["type"] == "evening"
        assert result["colors"] == "blue and gold"
    
    def test_extract_unstructured_facts(self):
        """Test extracting facts from unstructured text."""
        text = "category formal dress, type evening gown, colors deep blue"
        result = extract_prompt_fact_segments(text)
        
        assert result["category"] == "formal dress"
        assert result["type"] == "evening gown"
        assert result["colors"] == "deep blue"
    
    def test_extract_mixed_structured_unstructured(self):
        """Test extracting from mixed structured and unstructured text."""
        text = "category=formal; type=evening wear; colors: navy blue; pattern=floral"
        result = extract_prompt_fact_segments(text)
        
        assert result["category"] == "formal"
        assert result["type"] == "evening wear"
        assert result["colors"] == "navy blue"
        assert result["pattern"] == "floral"
    
    def test_extract_with_whitespace_cleanup(self):
        """Test extraction with whitespace and punctuation cleanup."""
        text = "  category  =  dress  ;  type  =  formal  .  "
        result = extract_prompt_fact_segments(text)
        
        assert result["category"] == "dress"
        assert result["type"] == "formal"
    
    def test_extract_known_labels_only(self):
        """Test extraction only returns known labels."""
        text = "category=dress; unknown_field=value; type=formal; random=data"
        result = extract_prompt_fact_segments(text)
        
        assert "category" in result
        assert "type" in result
        assert "unknown_field" not in result
        assert "random" not in result
    
    def test_extract_precedence_structured_over_unstructured(self):
        """Test structured format takes precedence over unstructured."""
        text = "category formal wear; category=dress; type=evening"
        result = extract_prompt_fact_segments(text)
        
        # Structured "category=dress" should take precedence
        assert result["category"] == "dress"
        assert result["type"] == "evening"
    
    def test_extract_all_supported_labels(self):
        """Test extraction supports all expected labels."""
        text = (
            "category=formal; type=dress; colors=blue; pattern=floral; "
            "material=silk; silhouette=a-line; construction=tailored; "
            "details=beaded; coverage=full; preserve=original"
        )
        result = extract_prompt_fact_segments(text)
        
        expected_labels = [
            "category", "type", "colors", "pattern", "material",
            "silhouette", "construction", "details", "coverage", "preserve"
        ]
        
        for label in expected_labels:
            assert label in result
            assert result[label]  # Non-empty value


class TestAvoidClauseGeneration:
    """Test avoid clause generation functionality."""
    
    def test_extract_avoid_directives_empty(self):
        """Test extracting avoid directives from empty input."""
        assert extract_generation_only_avoid_directives("") == []
        assert extract_generation_only_avoid_directives(None) == []
        assert extract_generation_only_avoid_directives("   ") == []
    
    def test_extract_avoid_directives_do_not(self):
        """Test extracting 'do not' directives."""
        text = "Do not add accessories. Keep the original style."
        result = extract_generation_only_avoid_directives(text)
        
        assert len(result) == 1
        assert result[0] == "Do not add accessories."
    
    def test_extract_avoid_directives_dont(self):
        """Test extracting 'don't' directives."""
        text = "Don't change the color. Maintain the fit."
        result = extract_generation_only_avoid_directives(text)
        
        assert len(result) == 1
        assert result[0] == "Don't change the color."
    
    def test_extract_avoid_directives_never(self):
        """Test extracting 'never' directives."""
        text = "Never alter the pattern. Keep it simple."
        result = extract_generation_only_avoid_directives(text)
        
        assert len(result) == 1
        assert result[0] == "Never alter the pattern."
    
    def test_extract_multiple_avoid_directives(self):
        """Test extracting multiple avoid directives."""
        text = "Do not add jewelry. Don't change colors. Never remove details."
        result = extract_generation_only_avoid_directives(text)
        
        assert len(result) == 3
        assert "Do not add jewelry." in result
        assert "Don't change colors." in result
        assert "Never remove details." in result
    
    def test_extract_avoid_directives_with_periods(self):
        """Test extracting directives that already have periods."""
        text = "Do not modify. Don't alter the style."
        result = extract_generation_only_avoid_directives(text)
        
        assert len(result) == 2
        assert result[0] == "Do not modify."
        assert result[1] == "Don't alter the style."
    
    def test_extract_avoid_directives_case_insensitive(self):
        """Test extraction is case insensitive."""
        text = "DO NOT CHANGE. don't modify. NEVER alter."
        result = extract_generation_only_avoid_directives(text)
        
        assert len(result) == 3
        assert "DO NOT CHANGE." in result
        assert "don't modify." in result
        assert "NEVER alter." in result
    
    def test_extract_ignores_non_avoid_sentences(self):
        """Test extraction ignores sentences that don't start with avoid words."""
        text = "This is a dress. Do not add accessories. Keep it simple."
        result = extract_generation_only_avoid_directives(text)
        
        assert len(result) == 1
        assert result[0] == "Do not add accessories."
    
    def test_merge_avoid_clauses_empty(self):
        """Test merging empty avoid clauses."""
        assert merge_avoid_clause_sentences() == ""
        assert merge_avoid_clause_sentences("", None, "   ") == ""
    
    def test_merge_avoid_clauses_single(self):
        """Test merging single avoid clause."""
        result = merge_avoid_clause_sentences("Do not add accessories.")
        assert result == "Do not add accessories."
    
    def test_merge_avoid_clauses_multiple(self):
        """Test merging multiple avoid clauses."""
        result = merge_avoid_clause_sentences(
            "Do not add jewelry.",
            "Don't change colors.",
            "Never remove details."
        )
        
        expected_sentences = [
            "Do not add jewelry.",
            "Don't change colors.",
            "Never remove details."
        ]
        
        for sentence in expected_sentences:
            assert sentence in result
    
    def test_merge_avoid_clauses_deduplication(self):
        """Test merging removes duplicate sentences."""
        result = merge_avoid_clause_sentences(
            "Do not add jewelry.",
            "Don't change colors.",
            "Do not add jewelry."  # Duplicate
        )
        
        sentences = result.split(" ")
        # Should only appear once
        jewelry_count = sum(1 for s in sentences if "jewelry" in s)
        assert jewelry_count == 1
    
    def test_merge_avoid_clauses_adds_periods(self):
        """Test merging adds periods to sentences without them."""
        result = merge_avoid_clause_sentences(
            "Do not add jewelry",  # No period
            "Don't change colors."  # Has period
        )
        
        assert "Do not add jewelry." in result
        assert "Don't change colors." in result
    
    def test_merge_avoid_clauses_splits_sentences(self):
        """Test merging splits multi-sentence input."""
        result = merge_avoid_clause_sentences(
            "Do not add jewelry. Don't change colors.",
            "Never remove details."
        )
        
        expected_sentences = [
            "Do not add jewelry.",
            "Don't change colors.",
            "Never remove details."
        ]
        
        for sentence in expected_sentences:
            assert sentence in result


class TestPromptGenerationHelpers:
    """Test additional prompt generation helper functions."""
    
    def test_serialize_prompt_fact_segments(self):
        """Test serializing fact segments back to text."""
        fields = {
            "category": "formal",
            "type": "dress",
            "colors": "navy blue",
            "pattern": "floral"
        }
        
        result = serialize_prompt_fact_segments(fields)
        
        # Should contain all fields in key=value format
        assert "category=formal" in result
        assert "type=dress" in result
        assert "colors=navy blue" in result
        assert "pattern=floral" in result
    
    def test_enrich_garment_descriptor(self):
        """Test enriching garment descriptor with fallback."""
        primary = "elegant evening dress"
        fallback = "formal gown with beading"
        garment_type = "dress"
        
        result = enrich_garment_descriptor(primary, fallback, garment_type)
        
        # Should contain primary descriptor
        assert "elegant evening dress" in result
        # Should be enriched with additional details
        assert len(result) > len(primary)
    
    def test_infer_flux2_target_type(self):
        """Test inferring target type from description."""
        # Test dress detection
        assert infer_flux2_target_type("elegant evening dress") == "dress"
        assert infer_flux2_target_type("formal gown") == "dress"
        
        # Test top detection
        assert infer_flux2_target_type("casual t-shirt") == "top"
        assert infer_flux2_target_type("button-up shirt") == "top"
        
        # Test bottom detection
        assert infer_flux2_target_type("denim jeans") == "bottom"
        assert infer_flux2_target_type("formal trousers") == "bottom"
    
    def test_ensure_target_type_in_description(self):
        """Test ensuring target type appears in description."""
        description = "elegant evening wear"
        target_type = "dress"
        
        result = ensure_target_type_in_description(description, target_type)
        
        # Should contain the target type
        assert "dress" in result.lower()
        # Should preserve original description
        assert "elegant evening wear" in result
    
    def test_clean_prompt_section_text(self):
        """Test cleaning prompt section text."""
        dirty_text = "  BASE_GARMENT_PROMPT:  red cotton shirt  with   extra   spaces  "
        clean_text = clean_prompt_section_text(dirty_text)
        
        # Should remove prefixes and normalize whitespace
        assert clean_text == "red cotton shirt with extra spaces"
        assert "BASE_GARMENT_PROMPT:" not in clean_text
        # Should normalize whitespace
        assert "   " not in clean_text
        # Should trim
        assert not clean_text.startswith(" ")
        assert not clean_text.endswith(" ")
    
    def test_join_avoid_terms(self):
        """Test joining avoid terms into readable text."""
        terms = ["jewelry", "accessories", "bright colors"]
        result = join_avoid_terms(terms)
        
        # Should be comma-separated with Oxford comma
        assert result == "jewelry, accessories, and bright colors"


class TestGarmentPromptNatural:
    def test_freeform_description_is_preserved_instead_of_collapsing_to_generic_prompt(self):
        prompt = build_garment_prompt_natural(
            "Beige long-sleeve wrap blouse with a crossover V-neckline, draped front, and blouson waist.",
            garment_type_hint="top",
        )

        assert "wrap blouse" in prompt.lower()
        assert "crossover v-neckline" in prompt.lower()
        assert "beige" not in prompt.lower()
        assert "a top garment" not in prompt.lower()

    def test_analyze_top_prompt_contains_flat_front_guard(self):
        prompt = get_analyze_flux2_positive_prompt("top").lower()

        assert "flat and cloth-like" in prompt
        assert "torso volume" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
