import importlib.util
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
        prompt = self._prompt("top", "structured cropped top")

        self.assertIn("Replace the upper garment with structured cropped top", prompt)
        self.assertIn("Keep lower-body clothing unchanged.", prompt)
        self.assertIn(SINGLE_GARMENT_PRESERVE_TAIL, prompt)

    def test_single_top_over_dress_uses_overlay_prompt(self):
        prompt = self._prompt("top", "structured cropped top", ["dress"])

        self.assertIn("Layer the top garment structured cropped top over the existing dress", prompt)
        self.assertIn("Treat the selected top as an overlay worn on top of the dress.", prompt)
        self.assertIn("Keep the dress unchanged underneath", prompt)

    def test_worn_types_normalization_dedupes_and_drops_invalid_values(self):
        prompt = self._prompt("top", "structured cropped top", ["Dress", "dress", "invalid-value"])

        self.assertIn("Layer the top garment structured cropped top over the existing dress", prompt)
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

    def test_single_top_over_top_bottom_replaces_only_top(self):
        prompt = self._prompt("top", "structured cropped top", ["top", "bottom"])

        self.assertIn("Replace only the upper garment with structured cropped top", prompt)
        self.assertIn("Keep the bottom garment unchanged.", prompt)

    def test_single_bottom_over_top_bottom_replaces_only_bottom(self):
        prompt = self._prompt("bottom", "straight-leg trousers", ["top", "bottom"])

        self.assertIn("Replace only the lower garment with straight-leg trousers", prompt)
        self.assertIn("Keep the top garment unchanged.", prompt)

    def test_single_top_over_top_bottom_outer_preserves_outer(self):
        prompt = self._prompt("top", "structured cropped top", ["top", "bottom", "outer"])

        self.assertIn("Replace only the upper garment with structured cropped top", prompt)
        self.assertIn("Keep the bottom garment unchanged.", prompt)
        self.assertIn("Keep the existing outer layer unchanged on top where visible.", prompt)

    def test_single_bottom_over_top_bottom_outer_preserves_outer(self):
        prompt = self._prompt("bottom", "straight-leg trousers", ["top", "bottom", "outer"])

        self.assertIn("Replace only the lower garment with straight-leg trousers", prompt)
        self.assertIn("Keep the top garment unchanged.", prompt)
        self.assertIn("Keep the existing outer layer unchanged on top where visible.", prompt)

    def test_single_outer_over_top_bottom_outer_replaces_only_outer(self):
        prompt = self._prompt("outer", "tailored blazer", ["top", "bottom", "outer"])

        self.assertIn("Replace only the outer layer with tailored blazer", prompt)
        self.assertIn("Keep the top and bottom garments unchanged underneath.", prompt)

    def test_single_top_with_outer_only_preserves_visible_non_target_garments(self):
        prompt = self._prompt("top", "structured cropped top", ["outer"])

        self.assertIn("Preserve the existing visible non-target garments, including any outer-layer coverage", prompt)

    def test_single_bottom_with_outer_only_preserves_visible_non_target_garments(self):
        prompt = self._prompt("bottom", "straight-leg trousers", ["outer"])

        self.assertIn("Preserve the existing visible non-target garments, including any outer-layer coverage", prompt)

    def test_single_outer_with_outer_only_replaces_only_outer(self):
        prompt = self._prompt("outer", "tailored blazer", ["outer"])

        self.assertIn("Replace only the outer layer with tailored blazer", prompt)
        self.assertIn("Keep the visible underlying non-target garments unchanged.", prompt)

    def test_ambiguous_worn_types_fall_back_to_existing_single_prompt(self):
        prompt = self._prompt("top", "structured cropped top", ["dress", "top"])

        self.assertIn("Replace the upper garment with structured cropped top", prompt)
        self.assertIn("Keep lower-body clothing unchanged.", prompt)

    def test_selected_dress_prompt_remains_unchanged_by_worn_types(self):
        prompt = self._prompt("dress", "structured mini dress", ["top", "bottom", "outer"])

        self.assertIn("Replace the entire outfit completely with structured mini dress", prompt)
        self.assertIn("Strictly remove other worn garments.", prompt)
        self.assertIn(SINGLE_GARMENT_PRESERVE_TAIL, prompt)

    def test_multi_garment_prompt_branch_remains_unchanged(self):
        prompt = self.service._build_tryon_lora_prompt(
            user_description=self.user_desc,
            garment_descriptions=["structured cropped top", "straight-leg trousers"],
            target_types=["top", "bottom"],
            board_mode="multi",
            source_worn_types=["dress"],
        )

        self.assertIn("Replace the upper garment with structured cropped top and replace the lower garment with straight-leg trousers", prompt)
        self.assertIn("Assign region ownership explicitly: upper-body clothing region is owned by the top garment and lower-body clothing region is owned by the bottom garment.", prompt)
        self.assertIn("Keep face identity, hair, body proportions, pose, hands, camera framing, background, and lighting unchanged.", prompt)


if __name__ == "__main__":
    unittest.main()
