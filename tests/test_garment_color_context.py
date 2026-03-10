import unittest

import numpy as np
from PIL import Image

from ai.core.garment_color_context import (
    GarmentColorContextSettings,
    build_single_image_color_context,
    build_visual_lock_clauses,
)


def _rgba_canvas() -> Image.Image:
    arr = np.zeros((160, 120, 4), dtype=np.uint8)
    arr[20:140, 20:100, 0] = 36
    arr[20:140, 20:100, 1] = 24
    arr[20:140, 20:100, 2] = 40
    arr[20:140, 20:100, 3] = 255
    arr[35:105, 50:58, 0] = 212
    arr[35:105, 50:58, 1] = 175
    arr[35:105, 50:58, 2] = 55
    arr[35:105, 50:58, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


class TestGarmentColorContext(unittest.TestCase):
    def test_single_image_context_separates_dominant_and_accent_colors(self):
        image = _rgba_canvas()
        ctx = build_single_image_color_context(
            image=image,
            description="velvet blazer with gold buttons",
            settings=GarmentColorContextSettings(
                top_k=3,
                palette_top_k=6,
                palette_min_area_percent=5.0,
                accent_min_area_percent=0.3,
                accent_top_k=2,
            ),
        )

        dominant_hexes = [str(v) for v in (ctx.get("dominantHexes") or [])]
        accent_hexes = [str(v) for v in (ctx.get("accentHexes") or [])]
        color_hints = [str(v) for v in (ctx.get("colorHints") or [])]
        accent_hints = [str(v) for v in (ctx.get("accentHints") or [])]

        self.assertTrue(dominant_hexes)
        self.assertTrue(any(hx.startswith("#25") or hx.startswith("#24") for hx in dominant_hexes))
        self.assertIn("#D4AF37", accent_hexes)
        self.assertIn("plum", color_hints)
        self.assertIn("gold", accent_hints)

    def test_visual_locks_include_per_item_accent_hints(self):
        image = _rgba_canvas()
        locks = build_visual_lock_clauses(
            product_images=[image, image],
            garment_descriptions=[
                "velvet blazer with gold buttons",
                "velvet blazer with gold buttons",
            ],
            settings=GarmentColorContextSettings(
                top_k=3,
                palette_top_k=6,
                palette_min_area_percent=5.0,
                accent_min_area_percent=0.3,
                accent_top_k=2,
            ),
        )

        self.assertEqual(len(locks.get("color_hints", [])), 2)
        self.assertEqual(len(locks.get("accent_hints", [])), 2)
        self.assertIn("gold", locks["accent_hints"][0])
        self.assertIn("accents:", str(locks.get("color_clause", "")))


if __name__ == "__main__":
    unittest.main()
