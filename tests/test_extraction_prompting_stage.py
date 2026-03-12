import types
import unittest

import numpy as np
from PIL import Image

import ai.main as main_mod
from ai.main import _parse_garment_prompt_sections
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
    def test_parse_garment_prompt_sections_accepts_json_object(self):
        bundle = _parse_garment_prompt_sections(
            '{"base_garment_prompt":"type=one-shoulder crop top; construction=one long sleeve only.",'
            '"extraction_avoid_clause":"ignore skin, tattoo, matching skirt."}',
            garment_type="top",
        )

        self.assertEqual(
            bundle["base_garment_prompt"],
            "type=one-shoulder crop top; construction=one long sleeve only.",
        )
        self.assertEqual(
            bundle["extraction_avoid_clause"],
            "ignore skin, tattoo, matching skirt.",
        )
        self.assertIn("BASE_GARMENT_PROMPT", bundle["serialized_sections"])
        self.assertIn("EXTRACTION_AVOID_CLAUSE", bundle["serialized_sections"])
        self.assertEqual(bundle["json_contract_valid"], "true")
        self.assertEqual(bundle["source_format"], "json")

    def test_parse_garment_prompt_sections_marks_freeform_as_non_json_contract(self):
        bundle = _parse_garment_prompt_sections(
            "Long-sleeve top in beige with draped V-neckline.",
            garment_type="top",
        )

        self.assertEqual(bundle["base_garment_prompt"], "Long-sleeve top in beige with draped V-neckline.")
        self.assertEqual(bundle["extraction_avoid_clause"], "")
        self.assertEqual(bundle["json_contract_valid"], "false")
        self.assertEqual(bundle["source_format"], "freeform")
        self.assertIn("EXTRACTION_AVOID_CLAUSE:", bundle["serialized_sections"])

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
            captured["reference_mask_shape"] = None if kwargs.get("reference_mask") is None else tuple(kwargs["reference_mask"].shape)
            color_ref = kwargs.get("color_reference_image")
            captured["color_reference_image_size"] = None if color_ref is None else tuple(color_ref.size)
            descriptor_ref = kwargs.get("descriptor_source_image")
            captured["descriptor_source_image_size"] = None if descriptor_ref is None else tuple(descriptor_ref.size)
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
        detector_crop = Image.new("RGB", (49, 78), "white")
        selected_item = {
            "type": "dress",
            "bbox": [1, 2, 50, 80],
            "_image_obj": detector_crop,
            "_mask_obj": np.ones((120, 100), dtype=bool),
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
        self.assertEqual(captured["reference_mask_shape"], (78, 49))
        self.assertEqual(captured["color_reference_image_size"], (100, 120))
        self.assertEqual(captured["descriptor_source_image_size"], (49, 78))
        self.assertEqual(updated_item["url"], "https://example.com/out.png")

    def test_local_minicpm_uses_structured_prompt_override_for_garment_description(self):
        captured = {}

        class _MiniCPM:
            def describe_garment(self, image, prompt_override=None):
                captured["image_size"] = tuple(image.size)
                captured["prompt_override"] = prompt_override
                return (
                    '{"base_garment_prompt":"type=one-shoulder crop top; colors=sage green; '
                    'construction=one long sleeve only, opposite side sleeveless.",'
                    '"extraction_avoid_clause":"ignore skin, tattoo, hair, and matching skirt."}'
                )

        original_engine = main_mod.engine
        main_mod.engine = types.SimpleNamespace(minicpm=_MiniCPM(), florence=None)
        try:
            image = Image.new("RGB", (320, 480), "white")
            desc = main_mod._describe_garment_with_backend(
                image=image,
                backend="minicpm",
                garment_type="top",
                dominant_color_hexes=["#88937B"],
                color_hints=["sage green"],
            )
        finally:
            main_mod.engine = original_engine

        self.assertEqual(
            desc,
            "type=one-shoulder crop top; colors=sage green; construction=one long sleeve only, opposite side sleeveless.",
        )
        self.assertEqual(captured["image_size"], (320, 480))
        self.assertIn("Return exactly one valid JSON object", captured["prompt_override"])
        self.assertIn("\"base_garment_prompt\"", captured["prompt_override"])
        self.assertIn("\"extraction_avoid_clause\"", captured["prompt_override"])
        self.assertIn("The required garment category is top.", captured["prompt_override"])
        self.assertIn("#88937B", captured["prompt_override"])

    def test_local_minicpm_prompt_bundle_preserves_json_avoid_clause(self):
        captured = {"calls": 0}

        class _MiniCPM:
            def describe_garment(self, image, prompt_override=None):
                captured["calls"] += 1
                captured["prompt_override"] = prompt_override
                captured["image_size"] = tuple(image.size)
                return (
                    '{"base_garment_prompt":"Long-sleeve wrap blouse in beige with a crossover V-neckline, draped front, and blouson waist.",'
                    '"extraction_avoid_clause":"ignore face, hair, skin, background, and white pants; do not simplify the wrap front into a plain cowl-neck top."}'
                )

        original_engine = main_mod.engine
        main_mod.engine = types.SimpleNamespace(minicpm=_MiniCPM(), florence=None)
        try:
            image = Image.new("RGB", (320, 480), "white")
            bundle = main_mod._describe_garment_prompt_bundle_with_backend(
                image=image,
                backend="minicpm",
                garment_type="top",
                dominant_color_hexes=["#CDB9A6"],
                color_hints=["beige"],
            )
        finally:
            main_mod.engine = original_engine

        self.assertEqual(captured["calls"], 1)
        self.assertEqual(captured["image_size"], (320, 480))
        self.assertEqual(
            bundle["base_garment_prompt"],
            "Long-sleeve wrap blouse in beige with a crossover V-neckline, draped front, and blouson waist.",
        )
        self.assertIn("white pants", bundle["extraction_avoid_clause"])
        self.assertEqual(bundle["json_contract_valid"], "true")

    def test_local_minicpm_prompt_bundle_retries_until_json_contract_valid(self):
        captured = {"calls": 0, "prompts": []}

        class _MiniCPM:
            def describe_garment(self, image, prompt_override=None):
                captured["calls"] += 1
                captured["prompts"].append(prompt_override)
                if captured["calls"] == 1:
                    return "Long-sleeve top in beige with a draped V-neckline."
                if captured["calls"] == 2:
                    return (
                        '{"base_garment_prompt":"Long-sleeve wrap blouse in beige with a crossover draped front and blouson waist.",'
                        '"extraction_avoid_clause":"ignore face, hair, skin, background, and white pants."}'
                    )
                return (
                    '{"base_garment_prompt":"Long-sleeve wrap blouse in beige with a crossover V-neckline, draped front, and blouson waist.",'
                    '"extraction_avoid_clause":"ignore face, hair, skin, background, and white pants."}'
                )

        original_engine = main_mod.engine
        main_mod.engine = types.SimpleNamespace(minicpm=_MiniCPM(), florence=None)
        try:
            image = Image.new("RGB", (320, 480), "white")
            bundle = main_mod._describe_garment_prompt_bundle_with_backend(
                image=image,
                backend="minicpm",
                garment_type="top",
                dominant_color_hexes=["#CDB9A6"],
                color_hints=["beige"],
            )
        finally:
            main_mod.engine = original_engine

        self.assertEqual(captured["calls"], 3)
        self.assertIn("must always be present", captured["prompts"][0])
        self.assertIn("Schema reminder", captured["prompts"][1])
        self.assertIn("too generic", captured["prompts"][2])
        self.assertEqual(bundle["json_contract_valid"], "true")
        self.assertIn("white pants", bundle["extraction_avoid_clause"])

    def test_local_minicpm_prompt_bundle_retries_when_json_is_valid_but_too_generic(self):
        captured = {"calls": 0, "prompts": []}

        class _MiniCPM:
            def describe_garment(self, image, prompt_override=None):
                captured["calls"] += 1
                captured["prompts"].append(prompt_override)
                if captured["calls"] == 1:
                    return (
                        '{"base_garment_prompt":"Beige long-sleeve top with a draped V-neckline and loose fit.",'
                        '"extraction_avoid_clause":"ignore background and hair."}'
                    )
                return (
                    '{"base_garment_prompt":"Beige long-sleeve wrap blouse with a crossover front, draped neckline, gathered blouson waist, and soft knit fabric.",'
                    '"extraction_avoid_clause":"ignore background, hair, skin, and white pants; do not simplify the wrap front."}'
                )

        original_engine = main_mod.engine
        main_mod.engine = types.SimpleNamespace(minicpm=_MiniCPM(), florence=None)
        try:
            image = Image.new("RGB", (320, 480), "white")
            bundle = main_mod._describe_garment_prompt_bundle_with_backend(
                image=image,
                backend="minicpm",
                garment_type="top",
                dominant_color_hexes=["#CDB9A6"],
                color_hints=["beige"],
            )
        finally:
            main_mod.engine = original_engine

        self.assertEqual(captured["calls"], 2)
        self.assertIn("too generic", captured["prompts"][1])
        self.assertIn("crossover front", bundle["base_garment_prompt"])
        self.assertIn("white pants", bundle["extraction_avoid_clause"])

    def test_local_minicpm_prompt_bundle_repairs_non_json_response_without_raising(self):
        captured = {"calls": 0, "prompts": []}

        class _MiniCPM:
            def describe_garment(self, image, prompt_override=None):
                captured["calls"] += 1
                captured["prompts"].append(prompt_override)
                if captured["calls"] == 1:
                    return "Beige long-sleeve top with a draped V-neckline and loose fit."
                if captured["calls"] == 2:
                    return "Beige long-sleeve blouse with crossover drape and soft fabric."
                return (
                    "Beige long-sleeve wrap blouse with a crossover front, softly draped neckline, "
                    "and blouson waist sitting below the waist."
                )

        original_engine = main_mod.engine
        main_mod.engine = types.SimpleNamespace(minicpm=_MiniCPM(), florence=None)
        try:
            image = Image.new("RGB", (320, 480), "white")
            bundle = main_mod._describe_garment_prompt_bundle_with_backend(
                image=image,
                backend="minicpm",
                garment_type="top",
                dominant_color_hexes=["#CDB9A6"],
                color_hints=["beige"],
            )
        finally:
            main_mod.engine = original_engine

        self.assertEqual(captured["calls"], 3)
        self.assertEqual(bundle["json_contract_valid"], "false")
        self.assertEqual(bundle["source_format"], "freeform_repaired")
        self.assertIn("crossover front", bundle["base_garment_prompt"])
        self.assertIn("Ignore skin", bundle["extraction_avoid_clause"])

    def test_local_minicpm_uses_color_prompt_override_for_color_terms(self):
        captured = {}

        class _MiniCPM:
            def describe_garment(self, image, prompt_override=None):
                captured["image_size"] = tuple(image.size)
                captured["prompt_override"] = prompt_override
                return "sage green, olive"

        original_engine = main_mod.engine
        main_mod.engine = types.SimpleNamespace(minicpm=_MiniCPM(), florence=None)
        try:
            image = Image.new("RGB", (320, 480), "white")
            terms = main_mod._describe_garment_color_terms_with_backend(
                image=image,
                backend="minicpm",
                garment_type="bottom",
            )
        finally:
            main_mod.engine = original_engine

        self.assertEqual(terms, ["green", "olive"])
        self.assertEqual(captured["image_size"], (320, 480))
        self.assertIn("Look only at the requested bottom.", captured["prompt_override"])
        self.assertIn("Return only 1 to 3 short garment fabric color words", captured["prompt_override"])

    def test_normalize_user_prepare_prompt_description_keeps_identity_pose_only(self):
        raw = (
            "identity=oval face, fair skin, straight black hair; "
            "body_pose=front-facing standing pose with arms relaxed; "
            "current_outfit=pink satin bra top and matching skirt; "
            "framing_lighting=mid-length crop with soft studio lighting; "
            "occlusion=large tinted visor covering the eyes; "
            "preserve=face identity, pose, body proportions, and lighting"
        )

        cleaned = main_mod._normalize_user_prepare_prompt_description(raw)

        self.assertIn("identity: oval face, fair skin, straight black hair", cleaned)
        self.assertIn("pose: front-facing standing pose with arms relaxed", cleaned)
        self.assertIn("framing/lighting: mid-length crop with soft studio lighting", cleaned)
        self.assertIn("occlusion: large tinted visor covering the eyes", cleaned)
        self.assertIn("preserve: face identity, pose, body proportions, and lighting", cleaned)
        self.assertNotIn("current outfit", cleaned.lower())
        self.assertNotIn("matching skirt", cleaned.lower())

    def test_build_flux2_targeted_prompt_uses_identity_reference_not_outfit_reference(self):
        prompt = main_mod._build_flux2_targeted_prompt(
            garment_descriptions=["category=top garment, type crop top."],
            user_description=(
                "identity: oval face, fair skin, straight black hair. "
                "pose: front-facing standing pose. "
                "framing/lighting: mid-length crop with soft studio lighting. "
                "current outfit: pink satin bra top and matching skirt."
            ),
            target_types=["top"],
            board_mode="single",
        )

        self.assertIn("Person identity reference from image 1:", prompt)
        self.assertNotIn("Person and current outfit reference", prompt)
        self.assertNotIn("pink satin bra top", prompt.lower())

    def test_normalize_prompt_descriptor_backend_forces_minicpm_family(self):
        self.assertEqual(main_mod._normalize_prompt_descriptor_backend("qwen2_5_vl"), "minicpm")
        self.assertEqual(main_mod._normalize_prompt_descriptor_backend("florence"), "minicpm")
        self.assertEqual(main_mod._normalize_prompt_descriptor_backend("minicpm"), "minicpm")
        self.assertEqual(main_mod._normalize_prompt_descriptor_backend("minicpm_service"), "minicpm_service")

    def test_build_flux2_targeted_prompt_adds_saree_limb_guard(self):
        prompt = main_mod._build_flux2_targeted_prompt(
            garment_descriptions=["category=dress garment, type=saree, details=black draped saree with red border."],
            user_description="identity: woman. pose: standing.",
            target_types=["dress"],
            board_mode="single",
        )

        self.assertIn("This is a saree transfer.", prompt)
        self.assertIn("Use arm and hand geometry only from image 1.", prompt)

    def test_visible_limb_preservation_penalizes_extra_arm_region(self):
        ref_parse = np.zeros((384, 256), dtype=np.uint8)
        out_parse = np.zeros((384, 256), dtype=np.uint8)
        ref_parse[120:264, 28:68] = 14
        ref_parse[120:264, 188:228] = 15
        out_parse[:, :] = ref_parse
        out_parse[180:288, 108:140] = 14

        ref_img = Image.new("RGB", (64, 64), "white")
        out_img = Image.new("RGB", (64, 64), "white")
        skin_rgb = np.array([220, 170, 145], dtype=np.uint8)
        for img, extra in ((ref_img, False), (out_img, True)):
            arr = np.asarray(img.resize((256, 384), Image.BICUBIC)).copy()
            arr[120:264, 28:68] = skin_rgb
            arr[120:264, 188:228] = skin_rgb
            if extra:
                arr[180:288, 108:140] = skin_rgb
            resized = Image.fromarray(arr).resize((64, 64), Image.BICUBIC)
            img.paste(resized)

        class _Parser:
            def __init__(self, parses):
                self._parses = list(parses)

            def parse(self, _image):
                return self._parses.pop(0)

            def category_ids(self, category):
                if category in {"arms", "arm", "left_arm", "right_arm"}:
                    return [14, 15]
                return []

        original_parser = main_mod.engine.parser
        try:
            main_mod.engine.parser = _Parser([ref_parse.copy(), ref_parse.copy()])
            same_score = main_mod._score_visible_limb_preservation(ref_img, ref_img)
            main_mod.engine.parser = _Parser([ref_parse.copy(), out_parse.copy()])
            extra_score = main_mod._score_visible_limb_preservation(ref_img, out_img)
        finally:
            main_mod.engine.parser = original_parser

        self.assertGreater(same_score, extra_score)

    def test_select_best_user_prep_face_candidate_prefers_upper_face_region(self):
        image = Image.new("RGB", (638, 938), "white")
        person_bbox = [220, 0, 562, 938]
        top_face = {
            "source": "haar_face",
            "bbox": [286, 32, 466, 212],
            "area": 32400,
            "area_ratio": 32400 / float(638 * 938),
        }
        torso_false_positive = {
            "source": "haar_face",
            "bbox": [235, 508, 550, 823],
            "area": 99225,
            "area_ratio": 99225 / float(638 * 938),
        }

        best = main_mod._select_best_user_prep_face_candidate(
            [top_face, torso_false_positive],
            image=image,
            person_bbox=person_bbox,
        )

        self.assertEqual(best["bbox"], top_face["bbox"])


if __name__ == "__main__":
    unittest.main()
