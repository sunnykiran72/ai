import unittest
from unittest.mock import Mock, patch

from PIL import Image

from services.tryon_service import TryonService


class TryonServiceModeRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = Mock()
        self.engine.board_builder = None
        self.engine.flux2 = Mock()
        self.engine.flux2_consistency = Mock()
        self.config = Mock()
        self.config.lora_mode = "tryon"

        self.service = TryonService(engine=self.engine, config=self.config)
        self.person = Image.new("RGB", (400, 600), color="white")
        self.garment = Image.new("RGB", (300, 450), color="black")

    @patch("services.tryon_service.storage.upload_image", return_value="https://example.com/output.png")
    @patch("services.tryon_service.download_image")
    async def test_default_routes_to_tryon_runner(
        self,
        mock_download_image,
        _mock_upload,
    ):
        mock_download_image.side_effect = [self.person, self.garment]
        self.engine.flux2.run_tryon.return_value = {
            "image": Image.new("RGB", (576, 768), color="gray"),
            "latency": 1.5,
            "metadata": {
                "prompt": "TRYON full body test prompt",
                "lora_scale": 0.6,
            },
        }

        result = await self.service.try_on(
            user_image_url="https://example.com/user.png",
            products=[
                {
                    "image": "https://example.com/garment.png",
                    "promptDescription": "black top with long sleeves",
                    "targetType": "top",
                }
            ],
            lora_scale=0.6,
            steps=10,
            seed=42,
        )

        self.engine.flux2.run_tryon.assert_called_once()
        self.engine.flux2_consistency.run_tryon.assert_not_called()
        self.assertEqual(result["metadata"]["mode"], "tryon-lora")
        self.assertEqual(result["metadata"]["engine_variant"], "flux2_tryon_lora")
        self.assertEqual(result["metadata"]["effective_lora_scale"], 0.6)

    @patch("services.tryon_service.storage.upload_image", return_value="https://example.com/output.png")
    @patch("services.tryon_service.download_image")
    async def test_tryon_mode_defaults_lora_scale_to_one(
        self,
        mock_download_image,
        _mock_upload,
    ):
        mock_download_image.side_effect = [self.person, self.garment]
        self.engine.flux2.run_tryon.return_value = {
            "image": Image.new("RGB", (576, 768), color="gray"),
            "latency": 1.5,
            "metadata": {
                "prompt": "TRYON full body test prompt",
            },
        }

        result = await self.service.try_on(
            user_image_url="https://example.com/user.png",
            products=[
                {
                    "image": "https://example.com/garment.png",
                    "promptDescription": "black top with long sleeves",
                    "targetType": "top",
                }
            ],
            steps=10,
            seed=42,
        )

        self.engine.flux2.run_tryon.assert_called_once()
        called_kwargs = self.engine.flux2.run_tryon.call_args.kwargs
        self.assertEqual(called_kwargs["lora_scale"], 1.0)
        self.assertEqual(result["metadata"]["effective_lora_scale"], 1.0)

    @patch("services.tryon_service.storage.upload_image", return_value="https://example.com/output.png")
    @patch("services.tryon_service.download_image")
    async def test_consistency_mode_routes_to_consistency_runner(
        self,
        mock_download_image,
        _mock_upload,
    ):
        mock_download_image.side_effect = [self.person, self.garment]
        self.engine.flux2_consistency.run_tryon.return_value = {
            "image": Image.new("RGB", (576, 768), color="gray"),
            "latency": 1.1,
            "metadata": {},
        }

        result = await self.service.try_on(
            user_image_url="https://example.com/user.png",
            products=[
                {
                    "image": "https://example.com/garment.png",
                    "promptDescription": "black top with long sleeves",
                    "targetType": "top",
                }
            ],
            mode="consistency-lora",
            lora_scale=0.7,
            steps=12,
            seed=7,
        )

        self.engine.flux2_consistency.run_tryon.assert_called_once()
        self.engine.flux2.run_tryon.assert_not_called()
        self.assertEqual(result["metadata"]["mode"], "consistency-lora")
        self.assertEqual(result["metadata"]["engine_variant"], "flux2_consistency")


if __name__ == "__main__":
    unittest.main()
