import unittest

import numpy as np
from PIL import Image

from ai.core.garment_color_context import (
    GarmentColorContextSettings,
    _nearest_color_label,
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
    def test_patterned_light_garment_keeps_black_accent_hint(self):
        arr = np.zeros((160, 120, 4), dtype=np.uint8)
        arr[20:140, 20:100, 3] = 255
        arr[20:140, 20:100, 0:3] = (235, 220, 205)
        for x0 in range(24, 100, 16):
            arr[20:140, x0 : min(x0 + 4, 100), 0:3] = (16, 14, 12)
        image = Image.fromarray(arr, mode="RGBA")

        ctx = build_single_image_color_context(
            image=image,
            description="crochet striped knit top",
            settings=GarmentColorContextSettings(
                top_k=4,
                palette_top_k=7,
                palette_min_area_percent=5.0,
                accent_min_area_percent=0.3,
                accent_top_k=2,
            ),
        )

        color_hints = [str(v) for v in (ctx.get("colorHints") or [])]
        self.assertTrue(any(token in color_hints for token in {"black", "charcoal"}))
        self.assertTrue(any(token in color_hints for token in {"beige", "champagne", "ivory"}))

    def test_warm_neutral_trouser_tones_do_not_collapse_to_gray(self):
        self.assertIn(_nearest_color_label((171, 130, 94)), {"tan", "brown", "beige"})
        self.assertIn(_nearest_color_label((184, 144, 108)), {"tan", "beige"})
        self.assertIn(_nearest_color_label((158, 116, 80)), {"tan", "brown"})

    def test_single_image_context_merges_shadow_variants_of_same_color(self):
        arr = np.zeros((120, 120, 4), dtype=np.uint8)
        arr[10:110, 15:105, 3] = 255
        arr[10:110, 15:45, 0:3] = (132, 143, 124)
        arr[10:110, 45:75, 0:3] = (126, 137, 118)
        arr[10:110, 75:105, 0:3] = (120, 130, 112)
        image = Image.fromarray(arr, mode="RGBA")

        ctx = build_single_image_color_context(
            image=image,
            description="solid knit top",
            settings=GarmentColorContextSettings(
                top_k=3,
                palette_top_k=6,
                palette_min_area_percent=5.0,
                accent_min_area_percent=0.3,
                accent_top_k=2,
                cluster_merge_delta_e=10.0,
            ),
        )

        dominant_hexes = [str(v) for v in (ctx.get("dominantHexes") or [])]
        color_hints = [str(v) for v in (ctx.get("colorHints") or [])]

        self.assertEqual(len(dominant_hexes), 1)
        self.assertTrue(color_hints)
        self.assertIn(color_hints[0], {"green", "olive", "gray", "tan"})

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
