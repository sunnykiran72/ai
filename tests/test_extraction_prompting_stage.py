import types
import unittest

from PIL import Image

from ai.modules.wardrobe.extraction.generation_stage import run_selected_item_extraction_or_response
from ai.modules.wardrobe.extraction.prompting_stage import apply_selected_item_prompting


class _Plan:
    def __init__(self, image):
        self.image = image
        self.anchor_bbox = [1, 2, 50, 80]
        self.geometry_source = "detector"
        self.mask_bbox = None
        self.extract_bbox = [1, 2, 50, 80]
        self.crop_mode = "detector_bbox"


class PromptingStageTests(unittest.TestCase):
    def test_apply_prompting_strips_descriptor_color_before_assembly(self):
        calls = {}

        def product_prompt_description(text, **_kwargs):
            calls["product_text"] = text
            return text

        def build_garment_metadata(**kwargs):
            calls["metadata_base_prompt"] = kwargs["base_garment_prompt"]
            return {
                "prompt": {
                    "base_garment_prompt": "category=dress; type=mini dress; colors=sage green; details=ruched.",
                    "prompt_description": "category=dress; type=mini dress; colors=sage green; details=ruched.",
                },
                "color": {
                    "dominant_hexes": ["#88937B"],
                    "color_hints": ["sage green"],
                },
            }

        item = {
            "type": "dress",
            "style": "",
            "category_key": "",
            "promptDescriptionSource": "flux2_extract_descriptor",
            "baseGarmentPrompt": "category=dress; type=mini dress; colors=gold; details=ruched.",
            "promptDescription": "category=dress; type=mini dress; colors=gold; details=ruched.",
            "extraction": {
                "descriptor_raw_text": "category=dress; type=mini dress; colors=gold; details=ruched.",
                "dominant_hexes": ["#88937B"],
                "color_hints": ["sage green"],
                "color_profile": {"isNeutral": False},
                "color_mask_source": "parser_strict_runtime",
            },
        }

        selected_item, context = apply_selected_item_prompting(
            selected_item=item,
            requested_type=None,
            analyze_prompt_from_extracted=True,
            normalize_garment_type=lambda raw: raw,
            infer_style_from_text=lambda *_args, **_kwargs: "fitted",
            wardrobe_category_from_garment_type=lambda *_args, **_kwargs: {
                "style": "fitted",
                "primary_category_key": "dresses",
                "category_key": "mini_dress",
            },
            product_prompt_description=product_prompt_description,
            build_garment_metadata=build_garment_metadata,
            strip_descriptor_color_clause=lambda text: text.replace("colors=gold; ", ""),
        )

        self.assertNotIn("gold", calls["metadata_base_prompt"])
        self.assertNotIn("gold", selected_item["promptDescription"])
        self.assertEqual(selected_item["promptDescription"], "category=dress; type=mini dress; colors=sage green; details=ruched.")
        self.assertEqual(selected_item["garmentMetadata"]["color"]["color_hints"], ["sage green"])
        self.assertEqual(context["selected_type"], "dress")

    def test_generation_uses_selected_type_for_color_mask_without_forced_request(self):
        captured = {}

        def run_flux2_cloth_only_extract(**kwargs):
            captured["garment_type"] = kwargs["garment_type"]
            captured["apply_type_color_mask"] = kwargs["apply_type_color_mask"]
            return {
                "url": "https://example.com/out.png",
                "_processed_image_bytes": b"png",
                "meta": {
                    "prompt_description": "",
                    "base_garment_prompt": "category=dress; type=maxi dress; details=pleated.",
                    "extraction_avoid_clause": "",
                    "prompt_sections_raw": "",
                },
            }

        class _Florence:
            def describe_garment(self, *_args, **_kwargs):
                raise AssertionError("Florence fallback should not run in extracted prompting path")

            def describe_garment_short(self, *_args, **_kwargs):
                raise AssertionError("Florence fallback should not run in extracted prompting path")

        engine = types.SimpleNamespace(florence=_Florence())
        image = Image.new("RGB", (100, 120), "white")
        selected_item = {
            "type": "dress",
            "bbox": [1, 2, 50, 80],
            "_image_obj": image,
            "promptDescription": "",
            "description": "",
        }

        updated_item, response = run_selected_item_extraction_or_response(
            selected_item=selected_item,
            requested_type=None,
            direct_requested_type_mode=False,
            full_image=image,
            all_items_count=1,
            stage_timings={},
            engine=engine,
            logger=types.SimpleNamespace(error=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
            analyze_extract_cloth=True,
            analyze_prompt_from_extracted=True,
            analyze_require_extracted_prompt=False,
            analyze_caption_mode="short",
            flux2_single_garment_extract_default_steps=12,
            flux2_single_garment_extract_default_seed=7,
            normalize_garment_type=lambda raw: raw,
            prepare_extract_source_image=lambda **kwargs: _Plan(kwargs["full_image"]),
            build_error_payload=lambda **kwargs: kwargs,
            multipart_form_response=lambda payload: payload,
            run_flux2_cloth_only_extract=run_flux2_cloth_only_extract,
            descriptor_is_weak=lambda text: not bool(text.strip()),
            caption_non_garment_signal=lambda _text: False,
            download_image=lambda _url: image,
            flatten_rgba_on_white=lambda img: img,
            sanitize_garment_description=lambda text: text,
            infer_style_from_text=lambda *_args, **_kwargs: "evening",
            wardrobe_category_from_garment_type=lambda *_args, **_kwargs: {
                "style": "evening",
                "primary_category_key": "dresses",
                "category_key": "maxi_dress",
            },
        )

        self.assertIsNone(response)
        self.assertEqual(captured["garment_type"], "dress")
        self.assertTrue(captured["apply_type_color_mask"])
        self.assertEqual(updated_item["url"], "https://example.com/out.png")


if __name__ == "__main__":
    unittest.main()
