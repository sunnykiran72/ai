import unittest

import ai.main as main_mod

from ai.main import (
    _build_flux2_single_garment_extract_negative_prompt,
    _build_flux2_single_garment_extract_prompt,
    _build_garment_metadata,
    _extract_garment_metadata_color_payload,
    _extract_garment_metadata_prompt,
    _extract_garment_metadata_target_type,
)


class GarmentMetadataContractTests(unittest.TestCase):
    def test_build_garment_metadata_keeps_prompt_classification_and_color(self):
        metadata = _build_garment_metadata(
            base_garment_prompt=(
                "bottom garment, type high-waisted trousers, colors beige, tan, pattern solid, "
                "material cotton, silhouette relaxed fit, construction high waist, straight leg, "
                "coverage lower body."
            ),
            extraction_avoid_clause="Ignore top garment and exposed skin.",
            prompt_sections_raw=(
                "BASE_GARMENT_PROMPT: bottom garment, type high-waisted trousers, colors beige, tan.\n"
                "EXTRACTION_AVOID_CLAUSE: Ignore top garment and exposed skin."
            ),
            descriptor_raw_text=(
                "category=bottom; type=high-waisted trousers; colors=beige, tan; pattern=solid; "
                "material=cotton; silhouette=relaxed fit; construction=high waist, straight leg; "
                "coverage=lower body"
            ),
            prompt_description="bottom garment, type high-waisted trousers, colors beige, tan.",
            prompt_source="flux2_extract_descriptor",
            target_type="bottom",
            backend_target_type="bottom",
            style="trousers",
            primary_category_key="bottoms",
            category_key="trousers",
            dominant_hexes=["#D7CAB9", "#BB8C68"],
            accent_hexes=["#8D6341"],
            color_hints=["beige", "tan"],
            color_profile={"medianL": 79.6, "meanChroma": 17.5},
            color_mask_source="segformer_b2_clothes",
        )

        self.assertEqual(metadata["schema_version"], "garment_metadata.v1")
        self.assertEqual(metadata["classification"]["target_type"], "bottom")
        self.assertEqual(metadata["classification"]["category_key"], "trousers")
        self.assertEqual(metadata["color"]["dominant_hexes"], ["#D7CAB9", "#BB8C68"])
        self.assertEqual(metadata["color"]["mask_source"], "segformer_b2_clothes")
        self.assertEqual(metadata["details"]["material"], "cotton")
        self.assertEqual(metadata["details"]["coverage"], "lower body")

    def test_extractors_prefer_structured_metadata(self):
        metadata = {
            "prompt": {
                "base_garment_prompt": "outer garment, type blazer, colors plum.",
            },
            "classification": {
                "backend_target_type": "outer",
            },
            "color": {
                "dominant_hexes": ["#4E3047"],
                "color_hints": ["plum"],
            },
        }

        self.assertEqual(
            _extract_garment_metadata_prompt(metadata),
            "outer garment, type blazer, colors plum.",
        )
        self.assertEqual(_extract_garment_metadata_target_type(metadata), "outer")
        self.assertEqual(
            _extract_garment_metadata_color_payload(metadata),
            (["#4E3047"], ["plum"]),
        )

    def test_build_garment_metadata_reconciles_semantic_color_over_neutral_pixel_drift(self):
        original = main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED
        main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = True
        try:
            metadata = _build_garment_metadata(
                base_garment_prompt=(
                    "category=dress; type=maxi dress; colors=blue, white; pattern=floral; material=cotton; "
                    "silhouette=fitted bodice, flared skirt; construction=collared neckline; details=drawstring at waist; "
                    "coverage=full body; preserve=color, print placement, and garment structure must remain unchanged."
                ),
                descriptor_raw_text=(
                    "category=dress; type=maxi dress; colors=blue, white; pattern=floral; material=cotton; "
                    "silhouette=fitted bodice, flared skirt; construction=collared neckline; details=drawstring at waist; "
                    "coverage=full body; preserve=color, print placement, and garment structure must remain unchanged."
                ),
                prompt_description=(
                    "category=dress; type=maxi dress; colors=blue, white; pattern=floral; material=cotton; "
                    "silhouette=fitted bodice, flared skirt; construction=collared neckline; details=drawstring at waist; "
                    "coverage=full body; preserve=color, print placement, and garment structure must remain unchanged."
                ),
                prompt_source="flux2_extract_descriptor",
                target_type="dress",
                backend_target_type="dress",
                style="maxi dress",
                primary_category_key="dresses",
                category_key="maxi_dresses",
                dominant_hexes=["#94A1AD", "#606B75", "#DDE4EA"],
                color_hints=["silver", "gray", "white"],
                color_profile={"medianL": 59.22, "meanChroma": 9.57, "isNeutral": True},
                color_mask_source="reference_mask",
            )
        finally:
            main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = original

        self.assertEqual(metadata["prompt"]["base_garment_prompt"].split("colors=")[1].split(";")[0], "blue, white")
        self.assertEqual(metadata["prompt"]["prompt_description"].split("colors=")[1].split(";")[0], "blue, white")
        self.assertEqual(metadata["color"]["color_hints"][:2], ["blue", "white"])
        self.assertEqual(metadata["color"]["resolved_source"], "semantic_prompt_override")
        self.assertEqual(metadata["details"]["colors"], "blue, white")

    def test_build_garment_metadata_prunes_soft_neutrals_for_saturated_yellow(self):
        metadata = _build_garment_metadata(
            base_garment_prompt=(
                "category=dress; type=gown; colors=beige, champagne, yellow; pattern=solid; material=unknown; "
                "silhouette=figure-hugging; construction=turtleneck, long sleeves; details=ruched; "
                "coverage=torso, legs; preserve=color, print placement, and structure must remain unchanged."
            ),
            descriptor_raw_text=(
                "category=dress; type=gown; colors=yellow, unknown; pattern=solid; material=unknown; "
                "silhouette=figure-hugging; construction=turtleneck, long sleeves; details=ruched; "
                "coverage=torso, legs; preserve=color, print placement, and structure must remain unchanged."
            ),
            prompt_description=(
                "category=dress; type=gown; colors=beige, champagne, yellow; pattern=solid; material=unknown; "
                "silhouette=figure-hugging; construction=turtleneck, long sleeves; details=ruched; "
                "coverage=torso, legs; preserve=color, print placement, and structure must remain unchanged."
            ),
            prompt_source="flux2_extract_descriptor",
            target_type="dress",
            backend_target_type="dress",
            style="gown",
            primary_category_key="dresses",
            category_key="maxi_dresses",
            dominant_hexes=["#E3D478", "#D9CA77", "#EEDF7C", "#CABC6A"],
            accent_hexes=["#AEA35D", "#877A43"],
            color_hints=["beige", "champagne", "yellow"],
            color_profile={"medianL": 81.96, "meanChroma": 45.0, "meanA": -6.08, "meanB": 44.57, "isNeutral": False},
            color_mask_source="parser_strict_runtime",
        )

        self.assertEqual(metadata["color"]["color_hints"], ["yellow"])
        self.assertEqual(metadata["prompt"]["base_garment_prompt"].split("colors=")[1].split(";")[0], "yellow")
        self.assertEqual(metadata["prompt"]["prompt_description"].split("colors=")[1].split(";")[0], "yellow")

    def test_extract_prompt_adds_bright_yellow_tone_guidance(self):
        profile = {"medianL": 81.96, "meanChroma": 45.0, "meanB": 44.57, "isNeutral": False}
        prompt = _build_flux2_single_garment_extract_prompt(
            garment_type="dress",
            prompt_description="type=bodycon gown; colors=yellow; pattern=solid.",
            dominant_color_hexes=["#DACB78", "#E3D478"],
            color_hints=["yellow"],
            color_profile=profile,
        )
        negative = _build_flux2_single_garment_extract_negative_prompt(
            garment_type="dress",
            color_hints=["yellow"],
            color_profile=profile,
        )

        self.assertIn("bright lemon yellow", prompt)
        self.assertIn("Do not reinterpret this color as gold, golden, mustard, beige, champagne, tan, bronze, brown, orange.", prompt)
        self.assertIn("gold, golden, mustard, beige, champagne, tan, bronze, brown, orange", negative)


if __name__ == "__main__":
    unittest.main()
