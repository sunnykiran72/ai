import unittest
from unittest.mock import Mock, patch

from PIL import Image

from services.tryon_service import TryonService


class TryonServiceCanvasTests(unittest.TestCase):
    def test_match_canvas_preserves_source_dimensions(self):
        service = TryonService(engine=Mock(), config=Mock())
        generated = Image.new("RGB", (512, 768), "white")

        matched = service._match_canvas(generated, (642, 926))

        self.assertEqual(matched.size, (642, 926))

    @patch("services.tryon_service.download_image")
    def test_resolve_products_uses_nested_metadata_target_type(self, mock_download_image):
        service = TryonService(engine=Mock(), config=Mock())
        mock_download_image.return_value = Image.new("RGB", (16, 16), "white")

        products = [
            {
                "image": "https://example.com/garment.png",
                "promptDescription": "category=dress; type=maxi dress; silhouette=flowy",
                "garmentMetadata": {
                    "classification": {
                        "target_type": "dress",
                    }
                },
            }
        ]

        garments, descriptions, target_types = service._resolve_products(
            products=products,
            garment_images=None,
            garment_image_url=None,
            garment_type=None,
            prompt_description=None,
        )

        self.assertEqual(len(garments), 1)
        self.assertEqual(descriptions[0], "category=dress; type=maxi dress; silhouette=flowy")
        self.assertEqual(target_types[0], "dress")

    @patch("services.tryon_service.download_image")
    def test_resolve_products_infers_target_type_from_prompt_description(self, mock_download_image):
        service = TryonService(engine=Mock(), config=Mock())
        mock_download_image.return_value = Image.new("RGB", (16, 16), "white")

        products = [
            {
                "image": "https://example.com/garment.png",
                "promptDescription": "blue jeans with wide leg fit",
            }
        ]

        _, _, target_types = service._resolve_products(
            products=products,
            garment_images=None,
            garment_image_url=None,
            garment_type=None,
            prompt_description=None,
        )

        self.assertEqual(target_types[0], "bottom")


if __name__ == "__main__":
    unittest.main()
