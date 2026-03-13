import unittest

import numpy as np
import ai.main as main_mod

from ai.main import (
    _augment_pixel_hints_with_muted_hue_family,
    _nearest_color_label,
    _resolve_garment_color_truth,
    _split_outfit_signature_from_parsing,
)


class SplitOutfitDetectionTests(unittest.TestCase):
    def test_detects_exposed_waist_two_piece_set(self):
        parsing = np.zeros((220, 140), dtype=np.uint8)
        parsing[24:92, 28:112] = 4   # upper
        parsing[112:212, 24:116] = 5  # skirt

        signature = _split_outfit_signature_from_parsing(parsing)

        self.assertTrue(bool(signature.get("detected")))
        self.assertEqual(signature.get("reason"), "split_gap_detected")
        self.assertGreater(int(signature.get("gap_px", 0)), 8)

    def test_does_not_flag_one_piece_dress_as_split_set(self):
        parsing = np.zeros((220, 140), dtype=np.uint8)
        parsing[24:212, 24:116] = 7  # dress

        signature = _split_outfit_signature_from_parsing(parsing)

        self.assertFalse(bool(signature.get("detected")))
        self.assertIn(signature.get("reason"), {"missing_top_or_bottom", "bridge_present_or_gap_small"})

    def test_low_chroma_green_gray_palette_is_promoted_to_green_family(self):
        hints = _augment_pixel_hints_with_muted_hue_family(
            ["gray"],
            ["#858D82"],
            {
                "medianL": 57.65,
                "meanChroma": 7.07,
                "meanA": -5.0,
                "meanB": 5.0,
                "isNeutral": True,
            },
        )

        self.assertIn(hints[0], {"green", "olive"})

    def test_dark_cool_low_chroma_resolves_to_navy_or_plum_not_black(self):
        # Approximate inky navy-black blazer tones from the top color failure case.
        self.assertIn(_nearest_color_label((18, 15, 27)), {"navy", "plum"})

    def test_dark_warm_palette_promotes_brown_over_black(self):
        hints = _augment_pixel_hints_with_muted_hue_family(
            ["black"],
            ["#130C07", "#25170F", "#82614E"],
            {
                "medianL": 5.49,
                "meanChroma": 8.51,
                "meanA": 4.94,
                "meanB": 6.65,
                "isNeutral": True,
            },
        )

        self.assertTrue(hints)
        self.assertEqual(hints[0], "brown")

    def test_light_neutral_shadow_does_not_stay_green_dominant(self):
        hints = _augment_pixel_hints_with_muted_hue_family(
            ["silver", "gray"],
            ["#A2A69D", "#898E89", "#5F6752", "#767C73"],
            {
                "medianL": 57.65,
                "meanChroma": 11.65,
                "meanA": -4.72,
                "meanB": 8.8,
                "isNeutral": False,
            },
        )

        self.assertFalse(any(term in {"green", "olive"} for term in hints[:1]))

    def test_weak_mask_bright_neutral_prefers_off_white_over_silver(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="leggings",
            target_type="bottom",
            dominant_hexes=["#A2A7A0", "#C2CAC8", "#B1B7B1", "#8E9DAB"],
            color_hints=["silver", "gray"],
            color_profile={
                "medianL": 69.41,
                "meanChroma": 5.27,
                "meanB": 0.83,
                "isNeutral": True,
            },
            color_mask_source="heuristic",
        )

        self.assertEqual((resolved.get("color_hints") or [None])[0], "off-white")

    def test_weak_mask_white_prompt_overrides_brown_contamination(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="short white skirt",
            target_type="bottom",
            dominant_hexes=["#563E29", "#CBC8C3", "#BDB9B0", "#9B9386"],
            color_hints=["brown", "silver", "tan"],
            color_profile={
                "medianL": 63.53,
                "meanChroma": 8.75,
                "meanB": 8.28,
                "isNeutral": True,
            },
            color_mask_source="heuristic",
        )

        self.assertEqual((resolved.get("color_hints") or [None])[0], "white")

    def test_relaxed_near_white_neutral_prefers_ivory_over_silver(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="leggings",
            target_type="bottom",
            dominant_hexes=["#ABB0A8", "#BFC7C2", "#9CA29C", "#929693"],
            color_hints=["silver", "gray"],
            color_profile={
                "medianL": 71.76,
                "meanChroma": 3.72,
                "meanB": 2.57,
                "isNeutral": True,
            },
            color_mask_source="parser_strict_runtime",
        )

        self.assertEqual((resolved.get("color_hints") or [None])[0], "ivory")

    def test_parser_top_shadowed_light_neutral_avoids_brown_drift(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="crop top with scoop neckline and long sleeves",
            target_type="top",
            dominant_hexes=["#999D95", "#7E847F", "#AEB2AB", "#D4D8BF"],
            color_hints=["silver", "brown", "tan"],
            color_profile={
                "medianL": 65.10,
                "p90L": 83.92,
                "meanChroma": 8.85,
                "meanB": 5.91,
                "isNeutral": True,
            },
            color_mask_source="parser_strict_runtime",
        )

        hints = resolved.get("color_hints") or []
        self.assertTrue(hints)
        self.assertEqual(hints[0], "ivory")
        self.assertNotIn("brown", hints[:2])

    def test_parser_top_shadowed_light_neutral_with_dark_shadow_prefers_ivory(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="halter top",
            target_type="top",
            dominant_hexes=["#786B56", "#867966", "#675B47", "#908573"],
            color_hints=["gray", "tan", "black"],
            color_profile={
                "medianL": 52.8,
                "p90L": 84.6,
                "meanChroma": 10.4,
                "meanB": 6.1,
                "isNeutral": True,
            },
            color_mask_source="parser_strict_runtime",
        )

        hints = resolved.get("color_hints") or []
        self.assertTrue(hints)
        self.assertEqual(hints[0], "ivory")
        self.assertNotIn("black", hints[:2])
        self.assertNotIn("tan", hints[:2])

    def test_parser_bottom_shadowed_light_neutral_prefers_ivory(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="mini skirt",
            target_type="bottom",
            dominant_hexes=["#9B9387", "#8F877C", "#A49F93", "#C8C7BF"],
            color_hints=["gray", "silver"],
            color_profile={
                "medianL": 54.4,
                "p90L": 82.1,
                "meanChroma": 6.8,
                "meanB": 4.6,
                "isNeutral": True,
            },
            color_mask_source="parser_bottom_refined_runtime",
        )

        hints = resolved.get("color_hints") or []
        self.assertTrue(hints)
        self.assertEqual(hints[0], "ivory")

    def test_parser_top_cast_corrected_light_neutral_prefers_ivory(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="long sleeve top",
            target_type="top",
            dominant_hexes=["#9EA199", "#B2B4AD", "#7D7C75", "#8A694B"],
            color_hints=["silver", "gray", "brown"],
            color_profile={
                "medianL": 60.0,
                "p90L": 71.76,
                "meanChroma": 9.64,
                "meanB": 8.13,
                "isNeutral": True,
                "castCorrected": {
                    "medianL": 60.0,
                    "meanChroma": 9.69,
                    "meanB": 8.31,
                    "isNeutral": True,
                },
                "castCorrectedHints": ["silver", "gray", "brown"],
            },
            color_mask_source="parser_strict_runtime",
        )

        hints = resolved.get("color_hints") or []
        self.assertTrue(hints)
        self.assertEqual(hints[0], "ivory")
        self.assertNotIn("brown", hints[:2])

    def test_parser_top_silver_gray_brown_shadow_prefers_ivory(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="long sleeve top",
            target_type="top",
            dominant_hexes=["#9EA199", "#B2B4AD", "#7D7C75", "#8D8D82", "#69513C"],
            color_hints=["silver", "gray", "brown"],
            color_profile={
                "medianL": 60.0,
                "p90L": 71.76,
                "meanChroma": 9.64,
                "meanB": 8.13,
                "isNeutral": True,
            },
            color_mask_source="parser_strict_runtime",
        )

        hints = resolved.get("color_hints") or []
        self.assertTrue(hints)
        self.assertEqual(hints[0], "ivory")
        self.assertNotIn("brown", hints[:2])

    def test_weak_mask_yellow_prompt_rescues_dark_neutral_pixel_drift(self):
        resolved = _resolve_garment_color_truth(
            base_garment_prompt="A long yellow dress with a high turtleneck and ruched detailing.",
            descriptor_raw_text="A long yellow dress with a high turtleneck and ruched detailing.",
            target_type="dress",
            dominant_hexes=["#787B78", "#4B4E4A", "#686C69", "#3C403D"],
            color_hints=["gray", "charcoal"],
            color_profile={
                "medianL": 41.57,
                "meanChroma": 2.61,
                "meanB": 2.09,
                "isNeutral": True,
            },
            color_mask_source="heuristic",
        )

        hints = resolved.get("color_hints") or []
        self.assertTrue(hints)
        self.assertEqual(hints[0], "yellow")
        self.assertEqual(resolved.get("color_source"), "semantic_prompt_override_weak_mask_rescue")

    def test_high_confidence_fashion_basecolour_blends_when_mask_is_weak(self):
        original_threshold = main_mod.ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE
        main_mod.ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE = 0.90
        try:
            resolved = _resolve_garment_color_truth(
                base_garment_prompt="shirt",
                target_type="top",
                dominant_hexes=["#F1F0EC", "#E7E4DD", "#D4D0C6"],
                color_hints=["gray", "silver"],
                color_profile={
                    "medianL": 60.0,
                    "meanChroma": 3.2,
                    "meanB": 1.2,
                    "isNeutral": True,
                },
                color_mask_source="heuristic",
                fashion_color_classifier={
                    "applied": True,
                    "top_label": "White",
                    "top_score": 0.97,
                    "predictions": [
                        {"label": "White", "score": 0.97, "canonical_hint": "white"},
                    ],
                },
            )
        finally:
            main_mod.ANALYZE_FASHION_BASECOLOUR_APPLY_MIN_SCORE = original_threshold

        hints = resolved.get("color_hints") or []
        self.assertTrue(hints)
        self.assertEqual(hints[0], "white")
        self.assertEqual(resolved.get("color_source"), "fashion_basecolour_blend")


if __name__ == "__main__":
    unittest.main()
