import importlib.util
import asyncio
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image


def _load_tryon_service_module():
    services_pkg = types.ModuleType("services")
    services_pkg.__path__ = []
    sys.modules["services"] = services_pkg

    ai_engine_module = types.ModuleType("services.ai_engine")

    class AIEngine:
        pass

    ai_engine_module.AIEngine = AIEngine
    sys.modules["services.ai_engine"] = ai_engine_module
    services_pkg.ai_engine = ai_engine_module

    config_module = types.ModuleType("config")

    class Flux2Config:
        pass

    config_module.Flux2Config = Flux2Config
    sys.modules["config"] = config_module

    shared_pkg = types.ModuleType("shared")
    shared_pkg.__path__ = []
    sys.modules["shared"] = shared_pkg

    image_ops_module = types.ModuleType("shared.image_ops")

    def download_image(*_args, **_kwargs):
        raise NotImplementedError

    image_ops_module.download_image = download_image
    sys.modules["shared.image_ops"] = image_ops_module
    shared_pkg.image_ops = image_ops_module

    azure_storage_module = types.ModuleType("shared.azure_storage")

    class _Storage:
        def upload_image(self, *_args, **_kwargs):
            return "https://example.com/output.png"

    azure_storage_module.storage = _Storage()
    sys.modules["shared.azure_storage"] = azure_storage_module
    shared_pkg.azure_storage = azure_storage_module

    utils_module = types.ModuleType("utils")

    def normalize_garment_type(value):
        text = str(value or "").strip().lower()
        mapping = {
            "top": "top",
            "upper": "top",
            "shirt": "top",
            "bottom": "bottom",
            "jeans": "bottom",
            "pants": "bottom",
            "trousers": "bottom",
            "outer": "outer",
            "outerwear": "outer",
            "jacket": "outer",
            "blazer": "outer",
            "dress": "dress",
        }
        return mapping.get(text, "")

    def infer_flux2_target_type(description):
        text = str(description or "").strip().lower()
        if "dress" in text:
            return "dress"
        if any(token in text for token in ("jeans", "pants", "trousers", "skirt", "shorts")):
            return "bottom"
        if any(token in text for token in ("jacket", "blazer", "coat", "outer")):
            return "outer"
        return "top"

    utils_module.normalize_garment_type = normalize_garment_type
    utils_module.infer_flux2_target_type = infer_flux2_target_type
    sys.modules["utils"] = utils_module

    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    sys.modules["modules"] = modules_pkg

    vto_pkg = types.ModuleType("modules.vto")
    vto_pkg.__path__ = []
    sys.modules["modules.vto"] = vto_pkg
    modules_pkg.vto = vto_pkg

    board_builder_module = types.ModuleType("modules.vto.board_builder")

    class BoardBuilder:
        def build_board(self, images):
            return images[0]

    board_builder_module.BoardBuilder = BoardBuilder
    sys.modules["modules.vto.board_builder"] = board_builder_module
    vto_pkg.board_builder = board_builder_module

    module_path = Path(__file__).resolve().parents[1] / "services" / "tryon_service.py"
    spec = importlib.util.spec_from_file_location("services.tryon_service", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["services.tryon_service"] = module
    services_pkg.tryon_service = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_TRYON_SERVICE_MODULE = _load_tryon_service_module()
SINGLE_GARMENT_PRESERVE_TAIL = _TRYON_SERVICE_MODULE.SINGLE_GARMENT_PRESERVE_TAIL
TryonService = _TRYON_SERVICE_MODULE.TryonService


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

    @patch("services.tryon_service.download_image")
    def test_try_on_defaults_output_max_edge_to_1280(self, mock_download_image):
        person_image = Image.new("RGB", (64, 96), "white")
        garment_image = Image.new("RGB", (32, 32), "black")
        mock_download_image.side_effect = [person_image, garment_image]

        engine = Mock()
        engine.flux2 = Mock()
        engine.flux2.run_tryon.return_value = {
            "image": Image.new("RGB", (64, 96), "white"),
            "latency": 1.23,
            "metadata": {},
        }

        service = TryonService(engine=engine, config=Mock())

        result = asyncio.run(
            service.try_on(
                user_image_url="https://example.com/user.jpg",
                garment_image_url="https://example.com/garment.png",
                garment_type="top",
                prompt_description="structured cropped top",
                steps=12,
                seed=123,
                guidance_scale=2.5,
                lora_scale=1.0,
            )
        )

        self.assertEqual(result["metadata"]["output_size"], [64, 96])
        engine.flux2.run_tryon.assert_called_once()
        self.assertEqual(engine.flux2.run_tryon.call_args.kwargs["output_max_edge"], 1280)


class TryonServicePromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.service = TryonService(engine=Mock(), config=Mock())
        self.user_desc = "same person description"

    def _prompt(self, garment_type, garment_desc, source_worn_types=None):
        return self.service._build_tryon_lora_prompt(
            user_description=self.user_desc,
            garment_descriptions=[garment_desc],
            target_types=[garment_type],
            board_mode="single",
            source_worn_types=source_worn_types,
        )

    def test_single_top_without_worn_types_uses_fallback_prompt(self):
        prompt = self._prompt("top", "structured cropped top.")

        self.assertIn("TRYON same person description. Replace entire upper garment with structured cropped top as shown in the reference images.", prompt)
        self.assertIn("Keep the bottom garment unchanged.", prompt)
        self.assertIn("Preserve the exact garment structure from the reference, including the front opening shape, closure placement, hem length, exposed torso areas, sleeve construction, edge finish and fabric pattern layout.", prompt)
        self.assertIn("Preserve any intentionally open or cutout areas exactly as part of the garment design", prompt)
        self.assertIn("Strictly Keep the same face, body measurements, hair color, eye directions, exact footwear, same accessories and preserve the body pose.", prompt)
        self.assertNotIn("top.", prompt)

    def test_single_top_over_dress_uses_overlay_prompt(self):
        prompt = self._prompt("top", "structured cropped top", ["dress"])

        self.assertIn("Replace entire upper garment with structured cropped top as shown in the reference images.", prompt)
        self.assertIn("Keep the bottom garment unchanged.", prompt)

    def test_worn_types_normalization_dedupes_and_drops_invalid_values(self):
        prompt = self._prompt("top", "structured cropped top", ["Dress", "dress", "invalid-value"])

        self.assertIn("Replace entire upper garment with structured cropped top as shown in the reference images.", prompt)
        self.assertNotIn("invalid-value", prompt)

    def test_single_outer_over_dress_uses_outermost_layer_prompt(self):
        prompt = self._prompt("outer", "tailored blazer", ["dress"])

        self.assertIn("Treat the selected outer garment as the outermost layer over the existing dress.", prompt)
        self.assertIn("Keep the dress unchanged underneath.", prompt)

    def test_single_bottom_under_dress_uses_naturally_visible_prompt(self):
        prompt = self._prompt("bottom", "straight-leg trousers", ["dress"])

        self.assertIn("Place the bottom garment straight-leg trousers underneath the existing dress", prompt)
        self.assertIn("Show the bottom only where it would naturally be visible", prompt)
        self.assertIn("Do not replace the dress with the bottom garment.", prompt)
        self.assertIn("The final image is a full body shot.", prompt)

    def test_single_top_over_top_bottom_replaces_only_top(self):
        prompt = self._prompt("top", "structured cropped top", ["top", "bottom"])

        self.assertIn("Replace entire upper garment with structured cropped top as shown in the reference images.", prompt)
        self.assertIn("Keep the bottom garment unchanged.", prompt)
        self.assertIn("Preserve the exact garment structure from the reference, including the front opening shape, closure placement, hem length, exposed torso areas, sleeve construction, edge finish and fabric pattern layout.", prompt)

    def test_single_bottom_over_top_bottom_replaces_only_bottom(self):
        prompt = self._prompt("bottom", "straight-leg trousers", ["top", "bottom"])

        self.assertIn("Replace entire lower garment with straight-leg trousers as shown in the reference images.", prompt)
        self.assertIn("Keep the top garment unchanged.", prompt)
        self.assertIn("Strictly Keep the same face, body measurements, hair color, eye directions, exact footwear, same accessories and preserve the body pose.", prompt)

    def test_single_top_over_top_bottom_outer_preserves_outer(self):
        prompt = self._prompt("top", "structured cropped top", ["top", "bottom", "outer"])

        self.assertIn("Replace entire upper garment with structured cropped top as shown in the reference images.", prompt)
        self.assertIn("Keep the bottom garment unchanged.", prompt)
        self.assertNotIn("Keep the existing outer layer unchanged on top where visible.", prompt)

    def test_single_bottom_over_top_bottom_outer_preserves_outer(self):
        prompt = self._prompt("bottom", "straight-leg trousers", ["top", "bottom", "outer"])

        self.assertIn("Replace entire lower garment with straight-leg trousers as shown in the reference images.", prompt)
        self.assertIn("Keep the top garment unchanged.", prompt)
        self.assertNotIn("Keep the existing outer layer unchanged on top where visible.", prompt)

    def test_single_outer_over_top_bottom_outer_replaces_only_outer(self):
        prompt = self._prompt("outer", "tailored blazer", ["top", "bottom", "outer"])

        self.assertIn("Replace only the outer layer with tailored blazer", prompt)
        self.assertIn("Keep the top and bottom garments unchanged underneath.", prompt)

    def test_single_top_with_outer_only_preserves_visible_non_target_garments(self):
        prompt = self._prompt("top", "structured cropped top", ["outer"])

        self.assertIn("Replace entire upper garment with structured cropped top as shown in the reference images.", prompt)

    def test_single_bottom_with_outer_only_preserves_visible_non_target_garments(self):
        prompt = self._prompt("bottom", "straight-leg trousers", ["outer"])

        self.assertIn("Replace entire lower garment with straight-leg trousers as shown in the reference images.", prompt)
        self.assertIn("Keep the top garment unchanged.", prompt)

    def test_single_outer_with_outer_only_replaces_only_outer(self):
        prompt = self._prompt("outer", "tailored blazer", ["outer"])

        self.assertIn("Replace only the outer layer with tailored blazer", prompt)
        self.assertIn("Keep the visible underlying non-target garments unchanged.", prompt)

    def test_ambiguous_worn_types_fall_back_to_existing_single_prompt(self):
        prompt = self._prompt("top", "structured cropped top", ["dress", "top"])

        self.assertIn("Replace entire upper garment with structured cropped top as shown in the reference images.", prompt)
        self.assertIn("Keep the bottom garment unchanged.", prompt)

    def test_selected_dress_prompt_remains_unchanged_by_worn_types(self):
        prompt = self._prompt("dress", "structured mini dress", ["top", "bottom", "outer"])

        self.assertIn("Replace the entire outfit completely with structured mini dress", prompt)
        self.assertIn("Strictly remove other worn garments.", prompt)
        self.assertIn(SINGLE_GARMENT_PRESERVE_TAIL, prompt)

    def test_multi_top_bottom_uses_approved_prompt(self):
        prompt = self.service._build_tryon_lora_prompt(
            user_description=self.user_desc,
            garment_descriptions=["structured cropped top", "straight-leg trousers"],
            target_types=["top", "bottom"],
            board_mode="multi",
            source_worn_types=["dress"],
        )

        self.assertIn("TRYON same person description. Replace the entire outfit with structured cropped top and straight-leg trousers as shown in the reference images.", prompt)
        self.assertIn("Preserve the exact garment structure from the reference, including the front opening shape, closure placement, hem length, exposed torso areas, sleeve construction, edge finish and fabric pattern layout.", prompt)
        self.assertIn("Preserve any intentionally open or cutout areas exactly as part of the garment design", prompt)
        self.assertIn("Strictly Keep the same face, body measurements, hair color, eye directions, exact footwear, same accessories and preserve the body pose.", prompt)
        self.assertTrue(prompt.endswith("The final image is a full body shot."))

    def test_multi_dress_top_uses_approved_prompt(self):
        prompt = self.service._build_tryon_lora_prompt(
            user_description=self.user_desc,
            garment_descriptions=["structured mini dress", "structured cropped top"],
            target_types=["dress", "top"],
            board_mode="multi",
            source_worn_types=["top", "bottom"],
        )

        self.assertIn("Replace the entire outfit with structured mini dress and structured cropped top as shown in the reference images.", prompt)
        self.assertIn("Use the dress as the base garment and apply the top garment as the upper-region overlay layer.", prompt)
        self.assertIn("Keep the lower dress structure visible where it is not covered by the top layer.", prompt)
        self.assertIn("Render both garments with accurate construction, silhouette, fit, seam and edge placement, drape, and length based on their references.", prompt)

    def test_multi_top_dress_order_is_irrelevant(self):
        prompt = self.service._build_tryon_lora_prompt(
            user_description=self.user_desc + ".",
            garment_descriptions=["structured cropped top.", "structured mini dress."],
            target_types=["top", "dress"],
            board_mode="multi",
            source_worn_types=["outer"],
        )

        self.assertIn("TRYON same person description. Replace the entire outfit with structured mini dress and structured cropped top as shown in the reference images.", prompt)
        self.assertNotIn("description..", prompt)
        self.assertNotIn("top..", prompt)
        self.assertNotIn("dress..", prompt)

    def test_multi_dress_bottom_uses_approved_prompt(self):
        prompt = self.service._build_tryon_lora_prompt(
            user_description=self.user_desc,
            garment_descriptions=["structured mini dress", "straight-leg trousers"],
            target_types=["dress", "bottom"],
            board_mode="multi",
            source_worn_types=["dress"],
        )

        self.assertIn("Replace the entire outfit with structured mini dress and straight-leg trousers as shown in the reference images.", prompt)
        self.assertIn("Use the dress as the base garment for the upper-body and dress structure, and assign the lower-body clothing region to the bottom garment.", prompt)
        self.assertIn("Preserve the exact garment color, tone, shading, and visible pattern placement from their references.", prompt)

    def test_multi_dress_top_bottom_uses_approved_prompt(self):
        prompt = self.service._build_tryon_lora_prompt(
            user_description=self.user_desc,
            garment_descriptions=["structured mini dress", "structured cropped top", "straight-leg trousers"],
            target_types=["dress", "top", "bottom"],
            board_mode="multi",
            source_worn_types=["top", "bottom"],
        )

        self.assertIn("Replace the outfit with structured mini dress, structured cropped top, and straight-leg trousers as shown in the reference images.", prompt)
        self.assertIn("Use the dress as the base garment, apply the top garment as the upper-region overlay layer, and assign the lower-body clothing region to the bottom garment.", prompt)
        self.assertIn("Keep edits confined to their owned regions and layers.", prompt)

    def test_multi_dress_outer_now_falls_back_to_generic_prompt(self):
        prompt = self.service._build_tryon_lora_prompt(
            user_description=self.user_desc,
            garment_descriptions=["structured mini dress", "tailored blazer"],
            target_types=["dress", "outer"],
            board_mode="multi",
            source_worn_types=["dress"],
        )

        self.assertIn("Replace the entire outfit with structured mini dress, tailored blazer as shown in the reference images.", prompt)
        self.assertIn("Render all garments with accurate construction, silhouette, fit, seam and edge placement, drape, and length based on the reference garments.", prompt)
        self.assertIn("Keep face identity, hair, body proportions, pose, hands, camera framing, background, and lighting unchanged.", prompt)


if __name__ == "__main__":
    unittest.main()
