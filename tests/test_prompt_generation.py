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
    build_tryon_prompt_v2,
    infer_top_prompt_subtype,
)
from config.prompts import get_analyze_flux2_positive_prompt, get_analyze_top_subtype_clause, get_minicpm_garment_prompt


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

        result = parse_structured_descriptor(
            "lower body pose: straight legs; held object: phone; object placement: left hand"
        )
        assert result == {
            "body_pose": "straight legs",
            "held_object": "left hand",
        }

    def test_parse_asymmetry_label_injection(self):
        """Test parsing one-shoulder and single-sleeve asymmetry labels."""
        result = parse_structured_descriptor("Type: top Asymmetry: one-shoulder Sleeves: single sleeve")
        assert result == {"type": "top", "asymmetry": "one-shoulder", "sleeves": "single sleeve"}
    
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
            "material=silk; silhouette=a-line; asymmetry=one-shoulder; construction=tailored; "
            "details=beaded; coverage=full; preserve=original"
        )
        result = extract_prompt_fact_segments(text)
        
        expected_labels = [
            "category", "type", "colors", "pattern", "material",
            "silhouette", "asymmetry", "construction", "details", "coverage", "preserve"
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


class TestTryonPromptBuilding:
    """Test try-on prompt building for FLUX-friendly structure."""

    def test_build_tryon_prompt_v2_keeps_identity_pose_and_background(self):
        prompt = build_tryon_prompt_v2(
            user_description=(
                "identity: oval face, fair skin, straight black hair. "
                "face: calm neutral expression looking forward. "
                "pose: front-facing standing pose with arms relaxed. "
                "lower body pose: straight legs with feet planted shoulder-width apart. "
                "framing/lighting: full-body crop with soft indoor lighting. "
                "occlusion: phone partially covering the face from the left hand. "
                "preserve: face identity, pose, body proportions, and background unchanged."
            ),
            garment_descriptions=[
                "category=top; type=top; asymmetry=one-shoulder; sleeves=single long sleeve; "
                "silhouette=slim fit; length_hem=cropped; fabric_texture=smooth; "
                "special_details=ruched bodice.",
            ],
            target_types=["top"],
            board_mode="single",
        )

        assert "Identity-preserving virtual try-on edit of the same person from image 1." in prompt
        assert "Treat the face in image 1 as the identity anchor" in prompt
        assert "Do not beautify, restyle, or replace the face." in prompt
        assert "Keep the exact body proportions, pose, hands, lower-body stance, leg spacing, knee angle, foot placement, hips, ankles, and camera framing from image 1." in prompt
        assert "Preserve the original background and lighting from image 1 exactly." in prompt
        assert "same grip, finger arrangement, wrist angle" in prompt
        assert "Prepared user reference: identity: oval face, fair skin, straight black hair. face: calm neutral expression looking forward. pose: front-facing standing pose with arms relaxed." in prompt
        assert "lower body pose: straight legs with feet planted shoulder-width apart" in prompt
        assert "If a phone or other held object is present" in prompt
        assert "Garment reference: A top with" in prompt
        assert "single long sleeve" in prompt
        assert "cropped hem" in prompt
        assert "ruched bodice" in prompt
        assert "Replace only the upper garment region from shoulders to hem." in prompt
        assert "Keep the lower body, legs, shoes, and lower-body pose exactly as in image 1." in prompt
        assert "Photorealistic fabric drape, realistic occlusion at the garment boundary, accurate seams, crisp detail, and clean composition." in prompt

    def test_build_tryon_prompt_v2_scope_changes_by_target_type(self):
        prompt = build_tryon_prompt_v2(
            user_description=(
                "identity: medium skin tone, long dark hair. "
                "face: neutral expression. "
                "pose: standing upright. "
                "preserve: face identity, pose, and background unchanged."
            ),
            garment_descriptions=[
                "category=bottom; type=wide leg trousers; silhouette=flowy; rise=high; length_hem=ankle.",
            ],
            target_types=["bottom"],
            board_mode="single",
        )

        assert "Replace only the lower garment region from waistband to hem." in prompt
        assert "Keep the top garment, face, hair, arms, and upper-body pose exactly as in image 1." in prompt

        dress_prompt = build_tryon_prompt_v2(
            user_description=(
                "identity: medium skin tone, long dark hair. "
                "face: neutral expression. "
                "pose: standing upright. "
                "preserve: face identity, pose, and background unchanged."
            ),
            garment_descriptions=[
                "category=dress; type=maxi dress; silhouette=flowy; length_hem=ankle.",
            ],
            target_types=["dress"],
            board_mode="single",
        )

        assert "Replace the full outfit with the dress reference." in dress_prompt
        assert "Keep the same face, body proportions, pose, hands, hair, background, and lighting from image 1." in dress_prompt
    
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

    def test_asymmetry_is_preserved_from_freeform_description(self):
        prompt = build_garment_prompt_natural(
            "Mint one-shoulder crop top with a single long sleeve and asymmetric drape.",
            garment_type_hint="top",
        )

        lowered = prompt.lower()
        assert "one-shoulder" in lowered or "single-shoulder" in lowered
        assert "single long sleeve" in lowered or "single sleeve" in lowered

    def test_top_prompt_drops_contradictory_strapless_shoulder_and_midriff_phrase(self):
        prompt = build_garment_prompt_natural(
            "category=top; type=crop top; neckline=bishop; shoulder_style=strapless; sleeves=long sleeves; bodice_cut=ruched bust; waistline=high; silhouette=figure-hugging; length_hem=midriff; fabric_texture=smooth; special_details=ruched bust detail",
            garment_type_hint="top",
        )

        lowered = prompt.lower()
        assert "strapless shoulder" not in lowered
        assert "midriff hem" not in lowered
        assert "cropped hem" in lowered
        assert "high waist" not in lowered
        assert "long sleeves" in lowered or "long sleeve" in lowered

    def test_top_prompt_rewrites_bishop_neckline_when_other_fields_indicate_low_open_bust_top(self):
        prompt = build_garment_prompt_natural(
            "category=top; type=crop top; neckline=bishop; shoulder_style=strapless; sleeves=long sleeves; "
            "bodice_cut=ruched bust; waistline=high; silhouette=figure-hugging; length_hem=midriff; "
            "fabric_texture=smooth; special_details=ruched bust detail",
            garment_type_hint="top",
        )

        lowered = prompt.lower()
        assert "low scoop neckline" in lowered
        assert "bishop neckline" not in lowered

    def test_top_prompt_uses_targeted_upper_edge_fields_to_override_high_coverage_neckline(self):
        prompt = build_garment_prompt_natural(
            "category=top; type=crop top; neckline=crew; shoulder_style=strapless; sleeves=long sleeves; "
            "upper_edge_shape=straight; torso_panel_continuity=short_panel; "
            "upper_edge_depth=low; upper_chest_exposed=yes; underbust_visible=yes; abdomen_visible=yes; "
            "bodice_cut=ruched bust; silhouette=figure-hugging; fabric_texture=smooth; special_details=ruched bodice",
            garment_type_hint="top",
        )

        lowered = prompt.lower()
        assert "low scoop neckline" in lowered
        assert "crew neckline" not in lowered
        assert "cropped hem" in lowered or "underbust hem" in lowered
        assert "short front panel" in lowered or "narrow underbust band" in lowered

    def test_top_prompt_preserves_bralette_context_from_structured_descriptor(self):
        prompt = build_garment_prompt_natural(
            "category=top; type=push-up bralette; neckline=low scoop; shoulder_style=strapless; asymmetry=unknown; "
            "sleeves=long sleeves; torso_panel_continuity=band_only; upper_chest_exposed=yes; "
            "sleeves=long sleeves; bodice_cut=push-up; waistline=high; silhouette=figure-hugging; "
            "fabric_texture=smooth; embellishments=ruching; special_details=ruched bodice",
            garment_type_hint="top",
        )

        lowered = prompt.lower()
        assert "push-up bralette" in lowered
        assert "strapless shoulder" in lowered
        assert "push-up bodice" in lowered
        assert "asymmetric shoulder structure" not in lowered
        assert "high waist" not in lowered
        assert "underbust hem" in lowered or "cropped hem" in lowered
        assert "band-only front" in lowered or "narrow underbust band" in lowered
        assert "open upper chest" in lowered

    def test_unknown_structured_asymmetry_does_not_infer_asymmetric_from_key_name(self):
        prompt = build_garment_prompt_natural(
            "category=top; type=brassiere; neckline=v-neck; shoulder_style=thin_straps; asymmetry=unknown; "
            "sleeves=unknown; silhouette=close_fit; fabric_texture=satin; special_details=bow_center_front",
            garment_type_hint="top",
        )

        lowered = prompt.lower()
        assert "asymmetric shoulder structure" not in lowered
        assert "brassiere" in lowered
        assert "cropped hem" in lowered

    def test_analyze_top_prompt_contains_flat_front_guard(self):
        prompt = get_analyze_flux2_positive_prompt("top").lower()

        assert "flat and cloth-like" in prompt
        assert "torso volume" in prompt
        assert "do not extend the garment into the abdomen" in prompt
        assert "do not warm it, cool it, bleach it, brighten it, darken it" in prompt
        assert "do not raise it into a higher-coverage chest panel" in prompt

    def test_minicpm_garment_prompt_requests_detailed_garment_paragraph(self):
        prompt = get_minicpm_garment_prompt().lower()

        assert "requested garment type" in prompt
        assert "rich garment paragraph" in prompt
        assert "visible garment facts" in prompt
        assert "do not mention colors" in prompt

    def test_minicpm_top_prompt_includes_top_only_structure_fields(self):
        prompt = get_minicpm_garment_prompt("top").lower()

        assert "top-only guidance" in prompt
        assert "shoulder layout" in prompt
        assert "torso panel continuity" in prompt
        assert "lower-body features" in prompt

    def test_minicpm_outer_prompt_excludes_inner_layers(self):
        prompt = get_minicpm_garment_prompt("outer").lower()

        assert "outerwear-only guidance" in prompt
        assert "inner garments" in prompt
        assert "collar" in prompt

    def test_top_subtype_router_identifies_bust_band_top(self):
        subtype = infer_top_prompt_subtype(
            "category=top; type=push-up bralette; upper_edge_shape=straight; upper_edge_depth=low; "
            "upper_chest_exposed=yes; shoulder_panel_present=no; underbust_visible=yes; abdomen_visible=no; "
            "torso_panel_continuity=band_only; lower_front_coverage=narrow_underbust_band; "
            "sleeve_attachment_mode=side_bust; sleeves=long sleeves; shoulder_style=strapless; bodice_cut=ruched bust"
        )

        assert subtype == "bust_band_top"

    def test_top_subtype_router_identifies_cropped_panel_top(self):
        subtype = infer_top_prompt_subtype(
            "category=top; type=crop top; upper_edge_shape=scoop; upper_edge_depth=low; "
            "upper_chest_exposed=yes; shoulder_panel_present=yes; underbust_visible=no; abdomen_visible=yes; "
            "torso_panel_continuity=short_panel; lower_front_coverage=short_panel; "
            "sleeve_attachment_mode=shoulder_seam; sleeves=long sleeves; shoulder_style=strapless; bodice_cut=ruched bust"
        )

        assert subtype == "cropped_panel_top"

    def test_top_subtype_router_identifies_asymmetric_top(self):
        subtype = infer_top_prompt_subtype(
            "category=top; type=top; neckline=asymmetric; upper_edge_shape=asymmetric; "
            "asymmetry=one-shoulder; sleeves=single long sleeve"
        )

        assert subtype == "asymmetric_top"

    def test_bust_band_top_clause_blocks_extra_panels(self):
        clause = get_analyze_top_subtype_clause("bust_band_top").lower()

        assert "do not add a continuous chest panel above the visible bust band" in clause
        assert "do not add a shoulder yoke" in clause
        assert "do not extend the garment into a longer torso panel" in clause

    def test_structured_corset_top_clause_blocks_sleeves_and_shoulder_coverage(self):
        clause = get_analyze_top_subtype_clause("structured_corset_top").lower()

        assert "do not add sleeves" in clause
        assert "do not add" in clause and "shoulder coverage" in clause
        assert "structured boning" in clause

    def test_march12_style_top_routes_to_bust_band_and_keeps_underbust_band_language(self):
        prompt = build_garment_prompt_natural(
            "category=top; type=crop top; neckline=sweetheart; upper_edge_shape=sweetheart; "
            "shoulder_style=strapless; sleeves=long sleeves; upper_edge_depth=low; upper_chest_exposed=yes; "
            "shoulder_panel_present=no; underbust_visible=no; abdomen_visible=yes; torso_panel_continuity=short_panel; "
            "lower_front_coverage=narrow_underbust_band; sleeve_attachment_mode=side_bust; "
            "bodice_cut=shaping; silhouette=cropped; length_hem=midriff; fabric_texture=smooth; "
            "embellishments=ribbed texture; special_details=ribbed bust band",
            garment_type_hint="top",
        )

        lowered = prompt.lower()
        assert infer_top_prompt_subtype(
            "category=top; type=crop top; neckline=sweetheart; upper_edge_shape=sweetheart; "
            "shoulder_style=strapless; sleeves=long sleeves; upper_edge_depth=low; upper_chest_exposed=yes; "
            "shoulder_panel_present=no; underbust_visible=no; abdomen_visible=yes; torso_panel_continuity=short_panel; "
            "lower_front_coverage=narrow_underbust_band; sleeve_attachment_mode=side_bust; "
            "bodice_cut=shaping; silhouette=cropped; length_hem=midriff"
        ) == "bust_band_top"
        assert "strapless shoulder" in lowered
        assert "underbust hem" in lowered
        assert "narrow underbust band" in lowered
        assert "short front panel" not in lowered

    def test_corset_top_preserves_none_as_absence_and_routes_to_structured_corset_top(self):
        desc = (
            "category=top; type=corset top; neckline=sweetheart; upper_edge_shape=sweetheart; "
            "shoulder_style=strapless; asymmetry=symmetric; sleeves=none; upper_edge_depth=very low; "
            "upper_chest_exposed=yes; shoulder_panel_present=no; underbust_visible=no; abdomen_visible=no; "
            "torso_panel_continuity=full panel; lower_front_coverage=full panel; bodice_cut=structured; "
            "silhouette=close-fitting; length_hem=underbust; fabric_texture=smooth; special_details=structured boning"
        )
        prompt = build_garment_prompt_natural(desc, garment_type_hint="top")
        lowered = prompt.lower()

        assert infer_top_prompt_subtype(desc) == "structured_corset_top"
        assert "strapless shoulder" in lowered
        assert "no shoulder panel" in lowered
        assert "sleeveless" in lowered
        assert "extended structured torso panel" in lowered
        assert "underbust hem" not in lowered

    def test_analyze_bottom_prompt_contains_single_piece_guard(self):
        prompt = get_analyze_flux2_positive_prompt("bottom").lower()

        assert "one continuous bottom garment" in prompt
        assert "upper-body panel" in prompt
        assert "split the garment" in prompt

    def test_analyze_dress_prompt_contains_single_piece_guard(self):
        prompt = get_analyze_flux2_positive_prompt("dress").lower()

        assert "one continuous garment from bodice to hem" in prompt
        assert "do not split it into a separate top and skirt" in prompt
        assert "two-piece outfit" in prompt

    def test_analyze_outer_prompt_contains_single_layer_guard(self):
        prompt = get_analyze_flux2_positive_prompt("outer").lower()

        assert "single layer" in prompt
        assert "no inner base garment" in prompt
        assert "layered outfit" in prompt

    def test_dress_prompt_drops_split_language(self):
        prompt = build_garment_prompt_natural(
            "category=dress; type=evening dress; bodice_cut=fitted; skirt_style=a-line; special_details=split bodice and separate skirt; layering=two-piece outfit",
            garment_type_hint="dress",
            ignore_layering=True,
        )

        lowered = prompt.lower()
        assert "split bodice" not in lowered
        assert "separate skirt" not in lowered
        assert "two-piece" not in lowered
        assert "dress" in lowered

    def test_outer_prompt_drops_layering_language(self):
        prompt = build_garment_prompt_natural(
            "category=outerwear; type=coat; collar=notch; lapel=wide; sleeves=long; special_details=inner layer and secondary garment",
            garment_type_hint="outer",
            ignore_layering=True,
        )

        lowered = prompt.lower()
        assert "inner layer" not in lowered
        assert "secondary garment" not in lowered
        assert "coat" in lowered or "outerwear" in lowered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
