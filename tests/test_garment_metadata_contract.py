import unittest

from PIL import Image

import ai.main as main_mod

from ai.main import (
    _build_flux2_visual_lock_clauses,
    _build_flux2_single_garment_extract_negative_prompt,
    _build_flux2_single_garment_extract_prompt,
    _build_garment_metadata,
    _extract_garment_metadata_color_payload,
    _extract_garment_metadata_color_block,
    _extract_garment_metadata_prompt,
    _extract_garment_metadata_target_type,
    _to_public_item,
)


class GarmentMetadataContractTests(unittest.TestCase):
    def test_to_public_item_restores_nested_metadata_compatibility(self):
        public = _to_public_item(
            {
                "type": "top",
                "url": "https://example.com/out.png",
                "output_image_url": "https://example.com/out.png",
                "extraction": {"path": "flux2_extract", "colorHints": ["ivory"]},
                "garmentMetadata": {"prompt": {"base_garment_prompt": "white crop top"}},
                "progress_sync": {"id": "abc123"},
                "_extracted_image_bytes": b"ignore",
            }
        )

        self.assertEqual(public["imageUrl"], "https://example.com/out.png")
        self.assertEqual(public["outputImage"], "https://example.com/out.png")
        self.assertEqual(public["metadata"]["garmentExtractionMeta"]["path"], "flux2_extract")
        self.assertEqual(
            public["metadata"]["garmentMetadata"]["prompt"]["base_garment_prompt"],
            "white crop top",
        )
        self.assertEqual(public["metadata"]["progressSync"]["id"], "abc123")
        self.assertNotIn("_extracted_image_bytes", public)

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

        self.assertEqual(metadata["prompt"]["base_garment_prompt"].split("colors=")[1].split(";")[0], "silver, gray, white")
        self.assertEqual(metadata["prompt"]["prompt_description"].split("colors=")[1].split(";")[0], "silver, gray, white")
        self.assertEqual(metadata["color"]["color_hints"][:2], ["silver", "gray"])
        self.assertEqual(metadata["color"]["resolved_source"], "pixel")
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

    def test_build_garment_metadata_prefers_descriptor_blue_when_pixel_path_is_neutral(self):
        original = main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED
        main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = True
        try:
            metadata = _build_garment_metadata(
                base_garment_prompt=(
                    "category=top; type=one-shoulder crop top; colors=gray, silver; pattern=solid; material=stretchy knit; "
                    "silhouette=snug fit; construction=one-shoulder neckline; details=side cutout; coverage=upper body."
                ),
                descriptor_raw_text=(
                    "category=top; type=one-shoulder crop top; colors=light blue, unknown; pattern=solid; material=stretchy knit; "
                    "silhouette=snug fit; construction=one-shoulder neckline; details=side cutout; coverage=upper body."
                ),
                prompt_description=(
                    "category=top; type=one-shoulder crop top; colors=gray, silver; pattern=solid; material=stretchy knit; "
                    "silhouette=snug fit; construction=one-shoulder neckline; details=side cutout; coverage=upper body."
                ),
                prompt_source="flux2_extract_descriptor",
                target_type="top",
                backend_target_type="top",
                style="crop top",
                primary_category_key="tops",
                category_key="crop_tops",
                dominant_hexes=["#878F84", "#80897F", "#90958B", "#999E91"],
                color_hints=["gray", "silver"],
                color_profile={"medianL": 58.82, "meanChroma": 7.42, "isNeutral": True},
                color_mask_source="parser_strict_runtime",
            )
        finally:
            main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = original

        self.assertEqual(metadata["color"]["color_hints"][0], "gray")
        self.assertEqual(metadata["color"]["resolved_source"], "pixel")
        self.assertEqual(metadata["details"]["colors"], "light blue, unknown")

    def test_build_garment_metadata_does_not_promote_muted_beige_to_yellow(self):
        metadata = _build_garment_metadata(
            base_garment_prompt="Beige long-sleeve top with a deep V-neckline and loose fit. The fabric appears to have a subtle sheen.",
            descriptor_raw_text="Beige long-sleeve top with a deep V-neckline and loose fit. The fabric appears to have a subtle sheen.",
            prompt_description="Beige long-sleeve top with a deep V-neckline and loose fit. The fabric appears to have a subtle sheen.",
            prompt_source="flux2_extract_descriptor",
            target_type="top",
            backend_target_type="top",
            style="blouse",
            primary_category_key="tops",
            category_key="blouses",
            dominant_hexes=["#A99E8C", "#A09381", "#B6AA98", "#968674"],
            color_hints=["beige", "tan"],
            color_profile={"medianL": 65.88, "meanChroma": 11.71, "meanA": 1.63, "meanB": 11.55, "isNeutral": True},
            color_mask_source="parser_strict_runtime",
        )

        self.assertEqual(metadata["color"]["color_hints"][:2], ["beige", "tan"])
        self.assertNotIn("yellow", metadata["color"]["color_hints"])
        self.assertIn("deep V-neckline", metadata["prompt"]["base_garment_prompt"])

    def test_build_garment_metadata_preserves_fashion_basecolour_payload(self):
        metadata = _build_garment_metadata(
            base_garment_prompt="White shirt with pointed collar and button placket.",
            descriptor_raw_text="White shirt with pointed collar and button placket.",
            prompt_description="White shirt with pointed collar and button placket.",
            prompt_source="flux2_extract_descriptor",
            target_type="top",
            backend_target_type="top",
            style="shirt",
            primary_category_key="tops",
            category_key="shirts",
            dominant_hexes=["#F1F1EF", "#E7E6E2"],
            accent_hexes=[],
            color_hints=["white"],
            color_profile={"medianL": 88.0, "meanChroma": 1.8, "meanB": 1.1, "isNeutral": True},
            color_mask_source="reference_mask",
            fashion_color_classifier={
                "applied": True,
                "top_label": "White",
                "top_score": 0.97,
                "predictions": [
                    {"label": "White", "score": 0.97, "canonical_hint": "white"},
                ],
            },
        )

        self.assertEqual(metadata["color"]["fashion_basecolour"]["top_label"], "White")
        self.assertTrue(metadata["color"]["fashion_basecolour"]["applied"])
        self.assertEqual(metadata["color"]["primary_color_label"], "white")
        self.assertEqual(metadata["color"]["primary_color_family"], "neutral_light")
        self.assertEqual(metadata["color"]["primary_color_descriptor"], "bright white")

    def test_build_garment_metadata_includes_color_signal_confidence(self):
        metadata = _build_garment_metadata(
            base_garment_prompt="Ivory long sleeve top.",
            descriptor_raw_text="Ivory long sleeve top.",
            prompt_description="Ivory long sleeve top.",
            prompt_source="flux2_extract_descriptor",
            target_type="top",
            backend_target_type="top",
            style="top",
            primary_category_key="tops",
            category_key="tops",
            dominant_hexes=["#F0E9D8", "#E2D9C7"],
            color_hints=["ivory", "cream"],
            color_profile={"medianL": 76.0, "meanChroma": 4.0, "meanB": 4.2, "isNeutral": True},
            color_mask_source="parser_strict_runtime",
            color_sampling_mask_meta={"source": "parser_strict_runtime", "mask_pixels": 12000, "area_ratio": 0.14},
        )

        color_block = _extract_garment_metadata_color_block(metadata)
        self.assertGreater(float(color_block["signal_confidence"]), 0.7)
        self.assertEqual(color_block["signal_strength"], "high")
        self.assertEqual(color_block["primary_color_descriptor"], "light ivory")

    def test_visual_lock_uses_high_confidence_metadata_descriptor(self):
        image = Image.new("RGB", (32, 32), (240, 238, 232))
        locks = _build_flux2_visual_lock_clauses(
            product_images=[image],
            garment_descriptions=["simple top"],
            garment_metadata_list=[
                {
                    "color": {
                        "dominant_hexes": ["#F0E9D8", "#E2D9C7"],
                        "color_hints": ["ivory", "cream"],
                        "profile": {"medianL": 76.0, "meanChroma": 4.0, "meanB": 4.2, "isNeutral": True},
                        "mask_source": "parser_strict_runtime",
                        "resolved_source": "pixel",
                        "signal_confidence": 0.84,
                        "signal_strength": "high",
                        "primary_color_descriptor": "light warm ivory",
                        "brightness": "light",
                        "saturation": "neutral",
                        "undertone": "warm",
                    }
                }
            ],
        )

        self.assertIn("light warm ivory", str(locks.get("color_clause") or ""))
        self.assertIn("Descriptive color lock", str(locks.get("color_clause") or ""))

    def test_build_garment_metadata_adds_rich_color_descriptors_and_sampling_meta(self):
        metadata = _build_garment_metadata(
            base_garment_prompt=(
                "category=dress; type=gown; colors=yellow; pattern=solid; material=satin; "
                "silhouette=column; construction=sleeveless; details=ruched; coverage=full body."
            ),
            descriptor_raw_text=(
                "category=dress; type=gown; colors=yellow; pattern=solid; material=satin; "
                "silhouette=column; construction=sleeveless; details=ruched; coverage=full body."
            ),
            prompt_description=(
                "category=dress; type=gown; colors=yellow; pattern=solid; material=satin; "
                "silhouette=column; construction=sleeveless; details=ruched; coverage=full body."
            ),
            prompt_source="flux2_extract_descriptor",
            target_type="dress",
            backend_target_type="dress",
            style="gown",
            primary_category_key="dresses",
            category_key="gowns",
            dominant_hexes=["#EAD97A", "#E5D16F"],
            accent_hexes=[],
            color_hints=["yellow"],
            color_profile={"medianL": 82.3, "meanChroma": 10.4, "meanA": -1.2, "meanB": 24.7, "isNeutral": False},
            color_mask_source="detector_parser_intersection",
            color_sampling_mask_meta={
                "source": "detector_parser_intersection",
                "used": True,
                "reason": "parser_trimmed_detector_context",
                "mask_pixels": 5240,
            },
        )

        self.assertEqual(metadata["color"]["primary_color_label"], "yellow")
        self.assertEqual(metadata["color"]["primary_color_family"], "yellow")
        self.assertEqual(metadata["color"]["brightness"], "light")
        self.assertEqual(metadata["color"]["saturation"], "pale")
        self.assertEqual(metadata["color"]["undertone"], "warm")
        self.assertEqual(metadata["color"]["primary_color_descriptor"], "light pale yellow")
        self.assertEqual(metadata["color"]["sampling_mask"]["source"], "detector_parser_intersection")
        self.assertTrue(metadata["color"]["sampling_mask"]["used"])

    def test_build_garment_metadata_preserves_freeform_structure_when_reconciling_color(self):
        metadata = _build_garment_metadata(
            base_garment_prompt="Long-sleeve top in beige, featuring a deep V-neckline and loose fit. The fabric appears lightweight and slightly sheer, featuring subtle draping at the front.",
            descriptor_raw_text="Long-sleeve top in beige, featuring a deep V-neckline and loose fit. The fabric appears lightweight and slightly sheer, featuring subtle draping at the front.",
            prompt_description="Long-sleeve top in beige, featuring a deep V-neckline and loose fit. The fabric appears lightweight and slightly sheer, featuring subtle draping at the front.",
            prompt_source="flux2_extract_descriptor",
            target_type="top",
            backend_target_type="top",
            style="blouse",
            primary_category_key="tops",
            category_key="blouses",
            dominant_hexes=["#A99E8C", "#A09381", "#B6AA98", "#968674"],
            color_hints=["beige", "tan"],
            color_profile={"medianL": 65.88, "meanChroma": 11.71, "meanA": 1.63, "meanB": 11.55, "isNeutral": True},
            color_mask_source="parser_strict_runtime",
        )

        self.assertIn("deep V-neckline", metadata["prompt"]["base_garment_prompt"])
        self.assertNotEqual(metadata["prompt"]["base_garment_prompt"], "colors=beige, tan; coverage=upper body.")

    def test_build_garment_metadata_near_white_neutral_prefers_white_over_silver(self):
        metadata = _build_garment_metadata(
            base_garment_prompt="White pants with a high waist and belt loops, featuring front pockets. The fabric appears smooth and fitted through the legs.",
            descriptor_raw_text="White pants with a high waist and belt loops, featuring front pockets. The fabric appears smooth and fitted through the legs.",
            prompt_description="White pants with a high waist and belt loops, featuring front pockets. The fabric appears smooth and fitted through the legs.",
            prompt_source="flux2_extract_descriptor",
            target_type="bottom",
            backend_target_type="bottom",
            style="trousers",
            primary_category_key="bottoms",
            category_key="trousers",
            dominant_hexes=["#B9B9B9", "#C4C5C7"],
            color_hints=["silver", "white"],
            color_profile={"medianL": 77.65, "p90L": 84.31, "meanChroma": 2.17, "meanA": 0.44, "meanB": 0.16, "isNeutral": True},
            color_mask_source="parser_strict_runtime",
        )

        self.assertEqual(metadata["color"]["color_hints"][0], "white")
        self.assertNotIn("silver", metadata["color"]["color_hints"][:1])

    def test_build_garment_metadata_prunes_warm_pixel_drift_when_descriptor_is_pink(self):
        original = main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED
        main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = True
        try:
            metadata = _build_garment_metadata(
                base_garment_prompt=(
                    "category=top; type=brassiere; colors=gold, pink; pattern=solid; material=satin; "
                    "silhouette=firm control fit; construction=low neckline; details=bow embellishment; coverage=upper torso."
                ),
                descriptor_raw_text=(
                    "category=top; type=brassiere; colors=pink and white; pattern=solid; material=satin; "
                    "silhouette=firm control fit; construction=low neckline; details=bow embellishment; coverage=upper torso."
                ),
                prompt_description=(
                    "category=top; type=brassiere; colors=gold, pink; pattern=solid; material=satin; "
                    "silhouette=firm control fit; construction=low neckline; details=bow embellishment; coverage=upper torso."
                ),
                prompt_source="flux2_extract_descriptor",
                target_type="top",
                backend_target_type="top",
                style="bralette",
                primary_category_key="tops",
                category_key="tanks_and_camis",
                dominant_hexes=["#C4906B", "#D8B5B0", "#F2E0DE", "#D2A27D"],
                color_hints=["gold", "pink"],
                color_profile={"medianL": 72.16, "meanChroma": 18.99, "isNeutral": False},
                color_mask_source="parser_strict_runtime",
            )
        finally:
            main_mod.GARMENT_COLOR_SEMANTIC_OVERRIDE_ENABLED = original

        self.assertEqual(metadata["color"]["color_hints"][:2], ["pink", "white"])
        self.assertIn("colors=pink, white", metadata["prompt"]["base_garment_prompt"])
        self.assertEqual(metadata["color"]["resolved_source"], "semantic_descriptor_bias")

    def test_build_garment_metadata_sanitizes_top_body_leakage(self):
        metadata = _build_garment_metadata(
            base_garment_prompt=(
                "category=top; type=one-shoulder crop top; colors=olive, gray, silver; pattern=solid; "
                "material=stretchy knit; silhouette=tight fit with bust emphasis; "
                "construction=one-shoulder neckline, long sleeve on one side, asymmetrical hem at waistline, "
                "midriff cutout revealing lower back tattoo; details=slit at front center of torso exposing part of the abdomen; "
                "coverage=upper body including chest, shoulders, arms (one), and upper hips; "
                "preserve=garment colors, print placement, and structure must remain unchanged."
            ),
            descriptor_raw_text=(
                "category=top; type=one-shoulder crop top; colors=light gray, unknown; pattern=solid; "
                "material=stretchy knit; silhouette=tight fit with bust emphasis; "
                "construction=one-shoulder neckline, long sleeve on one side, asymmetrical hem at waistline, "
                "midriff cutout revealing lower back tattoo; details=slit at front center of torso exposing part of the abdomen; "
                "coverage=upper body including chest, shoulders, arms (one), and upper hips."
            ),
            prompt_description=(
                "category=top; type=one-shoulder crop top; colors=olive, gray, silver; pattern=solid; "
                "material=stretchy knit; silhouette=tight fit with bust emphasis; "
                "construction=one-shoulder neckline, long sleeve on one side, asymmetrical hem at waistline, "
                "midriff cutout revealing lower back tattoo; details=slit at front center of torso exposing part of the abdomen; "
                "coverage=upper body including chest, shoulders, arms (one), and upper hips."
            ),
            prompt_source="flux2_extract_descriptor",
            target_type="top",
            backend_target_type="top",
            style="crop top",
            primary_category_key="tops",
            category_key="crop_tops",
            dominant_hexes=["#828B81", "#899186", "#92968A"],
            color_hints=["olive", "gray", "silver"],
            color_profile={"medianL": 59.22, "meanChroma": 6.99, "isNeutral": True},
            color_mask_source="heuristic",
        )

        prompt_text = metadata["prompt"]["base_garment_prompt"]
        self.assertIn("coverage=upper body", prompt_text)
        self.assertNotIn("tattoo", prompt_text.lower())
        self.assertNotIn("abdomen", prompt_text.lower())
        self.assertNotIn("upper hips", prompt_text.lower())
        self.assertNotIn("slit", prompt_text.lower())

    def test_build_garment_metadata_sanitizes_bottom_top_contamination(self):
        metadata = _build_garment_metadata(
            base_garment_prompt=(
                "category=bottom; type=mini skirt; colors=gray, beige, brown; pattern=solid; material=suede; "
                "silhouette=skinny fit; construction=unknown neckline, long sleeves, ruched waist/hip shaping, mini hem length; "
                "details=ruching on front and sides; coverage=lower torso to mid-thigh."
            ),
            descriptor_raw_text=(
                "category=bottom; type=mini skirt; colors=light blue, unknown; pattern=solid; material=suede; "
                "silhouette=skinny fit; construction=unknown neckline, long sleeves, ruched waist/hip shaping, mini hem length; "
                "details=ruching on front and sides; coverage=lower torso to mid-thigh."
            ),
            prompt_description=(
                "category=bottom; type=mini skirt; colors=gray, beige, brown; pattern=solid; material=suede; "
                "silhouette=skinny fit; construction=unknown neckline, long sleeves, ruched waist/hip shaping, mini hem length; "
                "details=ruching on front and sides; coverage=lower torso to mid-thigh."
            ),
            prompt_source="flux2_extract_descriptor",
            target_type="bottom",
            backend_target_type="bottom",
            style="mini skirt",
            primary_category_key="skirts",
            category_key="mini_skirts",
            dominant_hexes=["#936751", "#B37E60", "#C89A71"],
            color_hints=["gray", "beige", "brown"],
            color_profile={"medianL": 48.63, "meanChroma": 19.71, "isNeutral": False},
            color_mask_source="heuristic",
        )

        prompt_text = metadata["prompt"]["base_garment_prompt"]
        self.assertIn("coverage=lower body", prompt_text)
        self.assertNotIn("neckline", prompt_text.lower())
        self.assertNotIn("sleeve", prompt_text.lower())

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

    def test_extract_prompt_and_negative_lock_one_shoulder_crop_top_shape(self):
        prompt = _build_flux2_single_garment_extract_prompt(
            garment_type="top",
            prompt_description=(
                "type=one-shoulder crop top; colors=olive, gray; pattern=solid; material=stretchy knit; "
                "construction=one-shoulder neckline, long sleeve on one side, asymmetrical hem at waistline."
            ),
            dominant_color_hexes=["#828B81", "#899186"],
            color_hints=["olive", "gray"],
            color_profile={"medianL": 59.22, "meanChroma": 6.99, "isNeutral": True},
        )
        negative = _build_flux2_single_garment_extract_negative_prompt(
            garment_type="top",
            prompt_description="type=one-shoulder crop top; construction=one-shoulder neckline.",
        )

        self.assertIn("detached waistband", prompt)
        self.assertIn("must not extend into a full-length top", prompt)
        self.assertIn("exactly one shoulder", prompt)
        self.assertIn("second strap", negative)
        self.assertIn("detached waistband", negative)
        self.assertIn("full-length top", negative)

    def test_extract_prompt_for_plain_skirt_forbids_invented_pockets_and_buttons(self):
        prompt = _build_flux2_single_garment_extract_prompt(
            garment_type="bottom",
            prompt_description=(
                "type=mini skirt; colors=olive, gray; pattern=solid; material=suede; "
                "construction=high waist, ruched body, mini hem length."
            ),
            dominant_color_hexes=["#878F84", "#80897F"],
            color_hints=["olive", "gray"],
            color_profile={"medianL": 59.22, "meanChroma": 6.99, "isNeutral": True},
        )
        negative = _build_flux2_single_garment_extract_negative_prompt(
            garment_type="bottom",
            prompt_description="type=mini skirt; construction=high waist, ruched body, mini hem length.",
        )

        self.assertIn("Do not invent pockets", prompt)
        self.assertIn("invented pockets", negative)
        self.assertIn("invented buttons", negative)


if __name__ == "__main__":
    unittest.main()
