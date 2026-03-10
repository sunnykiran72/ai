import unittest

import numpy as np
from PIL import Image, ImageDraw

from ai.main import _estimate_type_focused_color_mask, _restore_outer_lower_body_from_reference


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


if __name__ == "__main__":
    unittest.main()
