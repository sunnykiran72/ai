import unittest

import numpy as np
from PIL import Image, ImageDraw

from ai.core.garment_color_masker import GarmentColorMasker, GarmentColorMaskerSettings


class _FakeParser:
    def __init__(self, parsing: np.ndarray):
        self._parsing = parsing
        self._labels = {
            "upper": 4,
            "skirt": 5,
            "pants": 6,
            "dress": 7,
            "belt": 8,
            "face": 11,
            "left_arm": 14,
            "right_arm": 15,
            "bag": 16,
            "scarf": 17,
            "torso": 18,
            "coat": 19,
        }

    def parse(self, image: Image.Image) -> np.ndarray:
        return self._parsing

    def _runtime_labels(self) -> dict:
        return dict(self._labels)

    def get_mask_for_category(self, parsing: np.ndarray, category: str) -> np.ndarray:
        if category == "bottom":
            return np.isin(parsing, [5, 6])
        if category == "top":
            return parsing == 4
        if category == "dress":
            return np.isin(parsing, [7, 18])
        if category == "outer":
            return np.isin(parsing, [19, 8])
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
            base_mask_fn=None,
            skin_mask_fn=lambda rgb: np.zeros((h, w), dtype=bool),
        )
        mask, meta = masker.estimate_mask(img, "bottom", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser_strict_runtime")
        mask = np.asarray(mask).astype(bool)
        self.assertTrue(bool(mask[180, 90]))
        self.assertFalse(bool(mask[40, 90]))

    def test_parser_mask_removes_skin_overlap_for_top_color_sampling(self):
        w, h = 160, 220
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[40:130, 25:135] = 4
        parser = _FakeParser(parsing)

        def _skin_mask(rgb: np.ndarray) -> np.ndarray:
            mask = np.zeros((h, w), dtype=bool)
            mask[24:78, 82:126] = True
            return mask

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=lambda image: np.ones((h, w), dtype=bool),
            skin_mask_fn=_skin_mask,
        )
        mask, meta = masker.estimate_mask(img, "top", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser_strict_runtime")
        self.assertGreater(int(meta.get("skin_pixels_removed", 0)), 0)
        mask = np.asarray(mask).astype(bool)
        self.assertTrue(bool(mask[80, 60]))
        self.assertFalse(bool(mask[45, 100]))

    def test_type_specific_parser_top_mask_excludes_body_and_accessories(self):
        w, h = 180, 220
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[48:150, 42:138] = 4   # top
        parsing[20:58, 70:112] = 11   # face
        parsing[58:150, 18:42] = 14   # arm
        parsing[72:150, 138:168] = 16  # bag
        parser = _FakeParser(parsing)

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=None,
            skin_mask_fn=lambda rgb: np.zeros((h, w), dtype=bool),
        )
        mask, meta = masker.estimate_mask(img, "top", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser_strict_runtime")
        self.assertEqual(meta.get("parser_variant"), "strict_type_specific")
        mask = np.asarray(mask).astype(bool)
        self.assertTrue(bool(mask[100, 90]))
        self.assertFalse(bool(mask[35, 85]))
        self.assertFalse(bool(mask[100, 30]))
        self.assertFalse(bool(mask[100, 150]))

    def test_parser_mask_still_removes_skin_when_overlap_is_large_but_garment_remains(self):
        w, h = 180, 220
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[28:170, 25:155] = 4
        parser = _FakeParser(parsing)

        def _skin_mask(rgb: np.ndarray) -> np.ndarray:
            mask = np.zeros((h, w), dtype=bool)
            mask[20:130, 50:150] = True
            return mask

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=lambda image: np.ones((h, w), dtype=bool),
            skin_mask_fn=_skin_mask,
        )
        mask, meta = masker.estimate_mask(img, "top", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser_strict_runtime")
        self.assertGreater(int(meta.get("skin_pixels_removed", 0)), 0)
        mask = np.asarray(mask).astype(bool)
        self.assertFalse(bool(mask[40, 90]))
        self.assertTrue(bool(mask[150, 40]))
        self.assertTrue(bool(mask[150, 90]))

    def test_parser_skin_overstrip_guard_keeps_original_mask_when_stripping_would_collapse_top(self):
        w, h = 180, 220
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[28:170, 25:155] = 4
        parser = _FakeParser(parsing)

        def _skin_mask(rgb: np.ndarray) -> np.ndarray:
            mask = np.zeros((h, w), dtype=bool)
            mask[28:170, 25:155] = True
            return mask

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=lambda image: np.ones((h, w), dtype=bool),
            skin_mask_fn=_skin_mask,
            settings=GarmentColorMaskerSettings(
                top_skin_upper_ratio=1.0,
                top_skin_side_ratio=1.0,
                parser_skin_min_remaining_ratio=0.40,
            ),
        )
        mask, meta = masker.estimate_mask(img, "top", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(int(meta.get("skin_pixels_removed", 0)), 0)
        mask = np.asarray(mask).astype(bool)
        self.assertTrue(bool(mask[100, 90]))

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

    def test_parser_strict_runtime_mask_excludes_torso_from_dress_color_mask(self):
        w, h = 180, 240
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[40:95, 60:120] = 18
        parsing[90:220, 35:145] = 7
        parser = _FakeParser(parsing)

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=None,
            skin_mask_fn=lambda rgb: np.zeros((h, w), dtype=bool),
        )
        mask, meta = masker.estimate_mask(img, "dress", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser_strict_runtime")
        mask = np.asarray(mask).astype(bool)
        self.assertTrue(bool(mask[180, 90]))
        self.assertFalse(bool(mask[60, 90]))

    def test_type_specific_outer_mask_prefers_outerwear_without_bottom_bleed(self):
        w, h = 200, 260
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[34:155, 38:162] = 19  # coat
        parsing[150:248, 46:154] = 6  # pants
        parsing[74:170, 160:195] = 16  # bag
        parser = _FakeParser(parsing)

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=None,
            skin_mask_fn=lambda rgb: np.zeros((h, w), dtype=bool),
        )
        mask, meta = masker.estimate_mask(img, "outer", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser_strict_runtime")
        mask = np.asarray(mask).astype(bool)
        self.assertTrue(bool(mask[90, 100]))
        self.assertFalse(bool(mask[200, 100]))
        self.assertFalse(bool(mask[120, 175]))

    def test_parser_bottom_mask_keeps_largest_component_only(self):
        w, h = 200, 260
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[115:250, 40:135] = 6
        parsing[95:125, 145:190] = 6
        parser = _FakeParser(parsing)

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=lambda image: np.ones((h, w), dtype=bool),
            skin_mask_fn=lambda rgb: np.zeros((h, w), dtype=bool),
        )
        mask, meta = masker.estimate_mask(img, "bottom", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        mask = np.asarray(mask).astype(bool)
        self.assertTrue(bool(mask[180, 80]))
        self.assertFalse(bool(mask[105, 165]))

    def test_bottom_parser_spatial_rescue_uses_lower_garment_when_pants_label_is_missing(self):
        w, h = 220, 320
        img = Image.new("RGB", (w, h), color=(245, 242, 238))
        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[20:135, 45:175] = 4   # upper clothes
        parsing[125:305, 40:180] = 4  # lower garment mislabeled as upper clothes
        parsing[138:305, 12:40] = 12  # left leg
        parsing[138:305, 180:208] = 13  # right leg
        parser = _FakeParser(parsing)

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=None,
            skin_mask_fn=lambda rgb: np.zeros((h, w), dtype=bool),
        )
        mask, meta = masker.estimate_mask(img, "bottom", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser_bottom_spatial_runtime")
        self.assertEqual(meta.get("parser_variant"), "bottom_spatial_rescue")
        mask = np.asarray(mask).astype(bool)
        self.assertFalse(bool(mask[80, 100]))
        self.assertTrue(bool(mask[230, 100]))
        self.assertFalse(bool(mask[230, 20]))

    def test_bottom_parser_spatial_rescue_can_be_trimmed_by_heuristic_overlap(self):
        w, h = 220, 320
        img = Image.new("RGB", (w, h), color=(250, 248, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle((70, 150, 155, 305), fill=(62, 54, 48))

        parsing = np.zeros((h, w), dtype=np.uint8)
        parsing[125:305, 40:180] = 4  # lower garment mislabeled as upper clothes
        parser = _FakeParser(parsing)

        def _base_mask(image: Image.Image) -> np.ndarray:
            arr = np.asarray(image.convert("RGB"))
            return np.any(arr < 240, axis=2)

        masker = GarmentColorMasker(
            parser=parser,
            base_mask_fn=_base_mask,
            skin_mask_fn=lambda rgb: np.zeros((h, w), dtype=bool),
            settings=GarmentColorMaskerSettings(
                parser_bottom_refine_min_overlap_ratio=0.30,
                parser_bottom_refine_min_heuristic_overlap_ratio=0.10,
            ),
        )
        mask, meta = masker.estimate_mask(img, "bottom", "")

        self.assertIsNotNone(mask)
        self.assertTrue(bool(meta.get("used")))
        self.assertEqual(meta.get("source"), "parser_bottom_refined_runtime")
        self.assertEqual(meta.get("parser_variant"), "bottom_spatial_rescue")
        mask = np.asarray(mask).astype(bool)
        self.assertFalse(bool(mask[170, 50]))
        self.assertTrue(bool(mask[220, 100]))
        self.assertFalse(bool(mask[120, 100]))


if __name__ == "__main__":
    unittest.main()
