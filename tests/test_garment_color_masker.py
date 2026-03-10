import unittest

import numpy as np
from PIL import Image, ImageDraw

from ai.core.garment_color_masker import GarmentColorMasker


class _FakeParser:
    def __init__(self, parsing: np.ndarray):
        self._parsing = parsing

    def parse(self, image: Image.Image) -> np.ndarray:
        return self._parsing

    def get_mask_for_category(self, parsing: np.ndarray, category: str) -> np.ndarray:
        if category == "bottom":
            return parsing == 6
        if category == "top":
            return parsing == 4
        if category == "dress":
            return parsing == 7
        if category == "outer":
            return parsing == 8
        return np.zeros_like(parsing, dtype=bool)


class GarmentColorMaskerTests(unittest.TestCase):
    def test_parser_mask_is_preferred_for_bottom(self):
        w, h = 180, 240
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[90:230, 35:145] = 6  # pants
        parsing[15:95, 25:155] = 4   # top
        parser = _FakeParser(parsing)

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=lambda image: np.ones((h, w), dtype=bool),
            skin_mask_fn=lambda rgb: np.zeros((h, w), dtype=bool),
        )
        mask, meta = masker.estimate_mask(img, "bottom", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser")
        mask = np.asarray(mask).astype(bool)
        self.assertTrue(bool(mask[180, 90]))
        self.assertFalse(bool(mask[40, 90]))

    def test_heuristic_bottom_mask_keeps_light_neutral_trousers(self):
        w, h = 220, 300
        img = Image.new("RGB", (w, h), color=(246, 243, 239))
        draw = ImageDraw.Draw(img)

        # Crochet top and skin.
        draw.rectangle((45, 20, 175, 135), fill=(232, 220, 202))
        draw.rectangle((70, 70, 150, 150), fill=(211, 160, 132))
        # Warm neutral trousers.
        draw.polygon([(48, 110), (105, 110), (95, 285), (25, 285)], fill=(201, 161, 125))
        draw.polygon([(115, 110), (172, 110), (195, 285), (125, 285)], fill=(214, 177, 140))

        def _base_mask(_image: Image.Image) -> np.ndarray:
            arr = np.asarray(_image.convert("RGB"))
            return np.any(arr < 238, axis=2)

        # Deliberately over-broad skin detector to simulate the current failure mode.
        def _skin_mask(rgb: np.ndarray) -> np.ndarray:
            r = rgb[:, :, 0]
            g = rgb[:, :, 1]
            b = rgb[:, :, 2]
            return (r > 95) & (g > 40) & (b > 20) & (r > g) & (g > b)

        masker = GarmentColorMasker(
            parser=None,
            base_mask_fn=_base_mask,
            skin_mask_fn=_skin_mask,
        )
        mask, meta = masker.estimate_mask(img, "bottom", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "heuristic")
        mask = np.asarray(mask).astype(bool)
        # Trouser center should remain selected.
        self.assertTrue(bool(mask[210, 80]))
        self.assertTrue(bool(mask[210, 145]))
        # Upper torso should be excluded.
        self.assertFalse(bool(mask[55, 110]))


if __name__ == "__main__":
    unittest.main()
