import unittest

from ai.main import (
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

        self.assertEqual(metadata["prompt"]["base_garment_prompt"].split("colors=")[1].split(";")[0], "blue, white")
        self.assertEqual(metadata["prompt"]["prompt_description"].split("colors=")[1].split(";")[0], "blue, white")
        self.assertEqual(metadata["color"]["color_hints"][:2], ["blue", "white"])
        self.assertEqual(metadata["color"]["resolved_source"], "semantic_prompt_override")
        self.assertEqual(metadata["details"]["colors"], "blue, white")


if __name__ == "__main__":
    unittest.main()
