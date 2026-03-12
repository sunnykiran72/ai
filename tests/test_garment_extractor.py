import unittest

import numpy as np
from PIL import Image

from ai.core.garment_extractor import (
    GarmentExtractionConfig,
    GarmentExtractionRequest,
    GarmentExtractor,
)


class TestGarmentExtractor(unittest.TestCase):
    def test_prefers_mask_bbox_and_returns_expanded_crop(self):
        image = Image.new("RGB", (200, 300), (255, 255, 255))
        mask = np.zeros((300, 200), dtype=bool)
        mask[40:240, 60:140] = True
        extractor = GarmentExtractor(
            GarmentExtractionConfig(
                crop_pad_ratio=0.10,
                crop_pad_ratio_dress=0.20,
                crop_bottom_extra_ratio_dress=0.25,
                dress_top_recovery_ratio=0.0,
                top_top_recovery_ratio=0.0,
            )
        )

        result = extractor.prepare(
            GarmentExtractionRequest(
                full_image=image,
                garment_type="top",
                total_items=1,
                detected_bbox=[70, 80, 130, 220],
                detector_mask=mask,
            )
        )

        self.assertEqual(result.anchor_bbox, [60, 40, 140, 240])
        self.assertEqual(result.geometry_source, "mask_bbox")
        self.assertEqual(result.extract_bbox, [52, 20, 148, 260])
        self.assertEqual(result.image.size, (96, 240))

    def test_dress_top_recovery_moves_anchor_upward(self):
        image = Image.new("RGB", (200, 300), (255, 255, 255))
        mask = np.zeros((300, 200), dtype=bool)
        mask[90:260, 50:150] = True
        extractor = GarmentExtractor(
            GarmentExtractionConfig(
                crop_pad_ratio_dress=0.20,
                crop_bottom_extra_ratio_dress=0.30,
                dress_top_recovery_ratio=0.20,
                top_top_recovery_ratio=0.0,
            )
        )

        result = extractor.prepare(
            GarmentExtractionRequest(
                full_image=image,
                garment_type="dress",
                total_items=1,
                detected_bbox=[52, 92, 148, 258],
                detector_mask=mask,
            )
        )

        self.assertEqual(result.anchor_bbox, [50, 56, 150, 260])
        self.assertIn("dress_top_recovery", result.geometry_source)
        self.assertLess(result.anchor_bbox[1], 90)
        self.assertEqual(result.extract_bbox, [30, 16, 170, 300])

    def test_bottom_multi_item_limits_top_padding(self):
        image = Image.new("RGB", (200, 300), (255, 255, 255))
        extractor = GarmentExtractor(
            GarmentExtractionConfig(
                crop_pad_ratio=0.20,
                crop_top_extra_ratio_bottom=0.20,
                crop_top_extra_ratio_bottom_multi=0.08,
                dress_top_recovery_ratio=0.0,
                top_top_recovery_ratio=0.0,
            )
        )

        result = extractor.prepare(
            GarmentExtractionRequest(
                full_image=image,
                garment_type="bottom",
                total_items=2,
                detected_bbox=[40, 120, 140, 260],
            )
        )

        self.assertEqual(result.anchor_bbox, [40, 120, 140, 260])
        self.assertEqual(result.extract_bbox, [20, 109, 160, 288])
        self.assertEqual(result.crop_mode, "garment_bbox_expanded")

    def test_top_multi_item_limits_bottom_padding(self):
        image = Image.new("RGB", (200, 300), (255, 255, 255))
        extractor = GarmentExtractor(
            GarmentExtractionConfig(
                crop_pad_ratio=0.20,
                crop_bottom_extra_ratio_top_multi=0.04,
                dress_top_recovery_ratio=0.0,
                top_top_recovery_ratio=0.0,
            )
        )

        result = extractor.prepare(
            GarmentExtractionRequest(
                full_image=image,
                garment_type="top",
                total_items=2,
                detected_bbox=[40, 60, 140, 180],
            )
        )

        self.assertEqual(result.anchor_bbox, [40, 60, 140, 180])
        self.assertEqual(result.extract_bbox, [20, 36, 160, 184])
        self.assertEqual(result.crop_mode, "garment_bbox_expanded")

    def test_semantic_box_can_refine_top_when_overlap_is_strong(self):
        image = Image.new("RGB", (220, 320), (255, 255, 255))
        extractor = GarmentExtractor(
            GarmentExtractionConfig(
                dress_top_recovery_ratio=0.0,
                top_top_recovery_ratio=0.0,
                semantic_refine_enabled=True,
                semantic_min_horizontal_overlap_ratio=0.50,
            )
        )

        result = extractor.prepare(
            GarmentExtractionRequest(
                full_image=image,
                garment_type="dress",
                total_items=1,
                detected_bbox=[60, 120, 160, 300],
                semantic_bbox=[50, 80, 170, 305],
            )
        )

        self.assertEqual(result.anchor_bbox, [50, 80, 170, 305])
        self.assertIn("semantic_top_refine", result.geometry_source)


if __name__ == "__main__":
    unittest.main()
