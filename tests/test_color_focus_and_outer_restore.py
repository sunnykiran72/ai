import unittest
from unittest.mock import patch

import ai.main as main_mod
import numpy as np
from PIL import Image, ImageDraw

from ai.main import (
    _build_single_image_color_context,
    _estimate_type_focused_color_mask,
    _resolve_color_sampling_mask,
    _restore_outer_lower_body_from_reference,
)


class ColorFocusAndOuterRestoreTests(unittest.TestCase):
    def test_type_focused_color_mask_prefers_top_garment_over_skin_and_gloves(self):
        img = Image.new("RGB", (240, 360), color=(245, 242, 238))
        draw = ImageDraw.Draw(img)

        # Skin torso region.
        draw.ellipse((70, 40, 170, 250), fill=(211, 160, 132))
        draw.ellipse((120, 45, 220, 250), fill=(211, 160, 132))
        # White gloves on sides.
        draw.rectangle((10, 120, 45, 300), fill=(245, 245, 245))
        draw.rectangle((195, 120, 230, 300), fill=(245, 245, 245))
        # Pink bra region in upper center.
        draw.polygon([(70, 90), (120, 70), (120, 165), (75, 170)], fill=(235, 188, 206))
        draw.polygon([(170, 70), (220, 90), (215, 170), (170, 165)], fill=(235, 188, 206))
        draw.rectangle((112, 120, 178, 140), fill=(225, 175, 192))

        expected_mask = np.zeros((360, 240), dtype=bool)
        expected_mask[78:170, 72:120] = True
        expected_mask[78:170, 170:218] = True
        expected_mask[118:142, 112:178] = True

        class _StubMasker:
            def estimate_mask(self, image, garment_type, description=""):
                return expected_mask, {"source": "stub", "used": True, "reason": "unit_test"}

        with patch.object(main_mod.engine, "garment_color_masker", _StubMasker()):
            mask = _estimate_type_focused_color_mask(img, "top", "pink satin bra with bow")

        self.assertIsNotNone(mask)
        mask = np.asarray(mask).astype(bool)
        # Bra center kept.
        self.assertTrue(bool(mask[130, 120]))
        self.assertTrue(bool(mask[130, 170]))
        # Glove and torso skin suppressed.
        self.assertFalse(bool(mask[200, 25]))
        self.assertFalse(bool(mask[180, 120]))

    def test_outer_restore_replaces_changed_lower_body_with_reference(self):
        ref = Image.new("RGB", (120, 200), color=(230, 230, 230))
        ref_arr = np.asarray(ref).copy()
        ref_arr[120:, :, :] = np.array([80, 80, 95], dtype=np.uint8)
        ref = Image.fromarray(ref_arr)

        out_arr = np.asarray(ref).copy()
        out_arr[:120, 20:100, :] = np.array([55, 35, 55], dtype=np.uint8)
        out_arr[130:, :, :] = np.array([145, 145, 160], dtype=np.uint8)
        out = Image.fromarray(out_arr)

        restored, meta = _restore_outer_lower_body_from_reference(
            reference_image=ref,
            output_image=out,
            target_types=["outer"],
            product_descriptions=["tailored velvet blazer"],
        )

        self.assertTrue(bool(meta.get("applied")))
        restored_arr = np.asarray(restored)
        # Lower body restored close to reference.
        self.assertTrue(np.allclose(restored_arr[180, 40], np.asarray(ref)[180, 40], atol=2))
        # Upper body/jacket remains from output.
        self.assertTrue(np.allclose(restored_arr[40, 40], np.asarray(out)[40, 40], atol=2))

    def test_force_masking_overrides_global_disable_for_color_context(self):
        img = Image.new("RGB", (120, 120), color=(194, 143, 108))
        draw = ImageDraw.Draw(img)
        draw.rectangle((30, 20, 90, 100), fill=(124, 131, 124))
        mask = np.zeros((120, 120), dtype=bool)
        mask[20:101, 30:91] = True

        with patch("ai.main.COLOR_CONTEXT_DISABLE_MASKING", True):
            unmasked = _build_single_image_color_context(
                image=img,
                description="",
                mask=mask,
                top_k=5,
                force_masking=False,
            )
            masked = _build_single_image_color_context(
                image=img,
                description="",
                mask=mask,
                top_k=5,
                force_masking=True,
            )

        self.assertEqual(unmasked.get("maskSource"), "disabled")
        self.assertNotEqual(masked.get("maskSource"), "disabled")
        self.assertIn("gray", [str(v) for v in (masked.get("colorHints") or [])])

    def test_color_sampling_mask_prefers_parser_trimmed_region_over_detector_mask(self):
        img = Image.new("RGB", (120, 160), color=(245, 242, 238))
        detector_mask = np.zeros((160, 120), dtype=bool)
        detector_mask[20:140, 15:105] = True
        parser_mask = np.zeros((160, 120), dtype=bool)
        parser_mask[32:118, 28:96] = True

        with patch("ai.main._estimate_type_focused_color_mask", return_value=(parser_mask, {"source": "parser_strict_runtime", "used": True})):
            mask, meta = _resolve_color_sampling_mask(
                image=img,
                garment_type="top",
                description="light pale yellow camisole",
                reference_mask=detector_mask,
                apply_type_color_mask=True,
            )

        self.assertIsNotNone(mask)
        mask = np.asarray(mask).astype(bool)
        self.assertEqual(meta.get("source"), "detector_parser_intersection")
        self.assertTrue(bool(mask[60, 60]))
        self.assertFalse(bool(mask[24, 20]))
        self.assertLess(int(meta.get("mask_pixels", 0)), int(np.sum(detector_mask)))


if __name__ == "__main__":
    unittest.main()
