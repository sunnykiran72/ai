import unittest

from ai import main as api_main


class TestUncertainFullbodyDressFallback(unittest.TestCase):
    def test_forces_single_dress_for_dominant_fullbody_traditional_garment(self):
        items = [
            {
                "type": "top",
                "garment_type": "top",
                "bbox": [40, 20, 360, 950],
                "promptDescription": "A woman wearing a red sari with gold border.",
                "description": "A woman wearing a red sari with gold border.",
                "style": "top",
                "category_key": "tops",
                "confidence": {"yolo": 0.81},
            },
            {
                "type": "top",
                "garment_type": "top",
                "bbox": [78, 420, 248, 820],
                "promptDescription": "Draped fabric detail.",
                "description": "Draped fabric detail.",
                "style": "top",
                "category_key": "tops",
                "confidence": {"yolo": 0.52},
            },
        ]

        updated, meta = api_main._maybe_force_uncertain_fullbody_to_dress(
            items,
            image_width=400,
            image_height=1000,
            requested_type=None,
        )

        self.assertIsNotNone(meta)
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["type"], "dress")
        self.assertEqual(updated[0]["garment_type"], "dress")
        self.assertEqual(updated[0]["type_source"], "uncertain_fullbody_dress_fallback")
        self.assertEqual(updated[0]["type_original"], "top")

    def test_keeps_real_top_bottom_pair_as_multi_item(self):
        items = [
            {
                "type": "top",
                "garment_type": "top",
                "bbox": [90, 40, 320, 360],
                "promptDescription": "Pink bra top.",
                "description": "Pink bra top.",
                "style": "bralette",
                "category_key": "tops",
                "confidence": {"yolo": 0.88},
            },
            {
                "type": "bottom",
                "garment_type": "bottom",
                "bbox": [84, 360, 330, 890],
                "promptDescription": "Pink satin skirt.",
                "description": "Pink satin skirt.",
                "style": "skirt",
                "category_key": "skirts",
                "confidence": {"yolo": 0.86},
            },
        ]

        updated, meta = api_main._maybe_force_uncertain_fullbody_to_dress(
            items,
            image_width=400,
            image_height=1000,
            requested_type=None,
        )

        self.assertIsNone(meta)
        self.assertEqual(len(updated), 2)
        self.assertEqual([item["type"] for item in updated], ["top", "bottom"])

    def test_requested_type_disables_fallback(self):
        items = [
            {
                "type": "top",
                "garment_type": "top",
                "bbox": [40, 20, 360, 950],
                "promptDescription": "A woman wearing a red sari with gold border.",
                "description": "A woman wearing a red sari with gold border.",
                "style": "top",
                "category_key": "tops",
                "confidence": {"yolo": 0.81},
            },
            {
                "type": "top",
                "garment_type": "top",
                "bbox": [78, 420, 248, 820],
                "promptDescription": "Draped fabric detail.",
                "description": "Draped fabric detail.",
                "style": "top",
                "category_key": "tops",
                "confidence": {"yolo": 0.52},
            },
        ]

        updated, meta = api_main._maybe_force_uncertain_fullbody_to_dress(
            items,
            image_width=400,
            image_height=1000,
            requested_type="top",
        )

        self.assertIsNone(meta)
        self.assertEqual(len(updated), 2)


if __name__ == "__main__":
    unittest.main()
