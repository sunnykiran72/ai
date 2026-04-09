import unittest

from PIL import Image

from utils.user_preparation import (
    prepare_user_image_core,
    _build_user_prepare_prompt,
    _extract_user_prepare_prompt_bundle,
)


class TestUserPreparationPipeline(unittest.TestCase):
    def test_extract_prompt_bundle_from_json(self):
        raw = (
            '{"garments":["top","bottom","top"],'
            '"prompt":"young woman with curly hair, medium build, smiling while seated with hands grounded and legs spread"}'
        )
        prompt, worn_types = _extract_user_prepare_prompt_bundle(raw)
        self.assertTrue(prompt.endswith("."))
        self.assertEqual(worn_types, ["top", "bottom"])

    def test_extract_prompt_bundle_from_plain_text_without_keyword_inference(self):
        raw = "young woman wearing a jacket over a dress, seated casually with one hand on the floor"
        prompt, worn_types = _extract_user_prepare_prompt_bundle(raw)
        self.assertTrue(prompt.endswith("."))
        self.assertEqual(worn_types, [])

    def test_build_user_prompt_normalizes_loose_hair(self):
        prompt = _build_user_prepare_prompt(
            {
                "person_type": "girl",
                "age_band": "young",
                "hair_color": "brown",
                "hair_length": "long",
                "hair_style": "loose",
                "body_build": "slim",
            }
        )
        self.assertEqual(prompt, "A young girl with brown long loose hair and slim body build.")

    def test_build_user_prompt_maps_legacy_hair_labels(self):
        prompt = _build_user_prepare_prompt(
            {
                "person_type": "woman",
                "age_band": "adult",
                "hair_color": "blonde",
                "hair_length": "shoulder-length",
                "hair_style": "straight",
                "body_build": "average",
            }
        )
        self.assertEqual(prompt, "An adult woman with blonde medium hair and average body build.")

    def test_extract_prompt_bundle_accepts_loose_and_empty_style(self):
        raw = (
            '{"garments":["dress"],'
            '"person_type":"girl","age_band":"young","hair_color":"brown",'
            '"hair_length":"long","hair_style":"loose","body_build":"slim"}'
        )
        prompt, worn_types = _extract_user_prepare_prompt_bundle(raw)
        self.assertEqual(prompt, "A young girl with brown long loose hair and slim body build.")
        self.assertEqual(worn_types, ["dress"])

    def test_build_user_prompt_supports_half_up_style(self):
        prompt = _build_user_prepare_prompt(
            {
                "person_type": "woman",
                "age_band": "young",
                "hair_color": "brown",
                "hair_length": "long",
                "hair_style": "half-up",
                "body_build": "slim",
            }
        )
        self.assertEqual(prompt, "A young woman with brown long half-up hair and slim body build.")

    def test_build_user_prompt_dedupes_repeated_hair_tokens(self):
        prompt = _build_user_prepare_prompt(
            {
                "person_type": "woman",
                "age_band": "young",
                "hair_color": "multi-tone",
                "hair_length": "medium",
                "hair_style": "medium",
                "body_build": "average",
            }
        )
        self.assertEqual(prompt, "A young woman with multi-tone medium hair and average body build.")

    def test_accepts_with_grounding_dino_and_verifier(self):
        image = Image.new("RGB", (400, 600), "white")

        def grounding_detector_fn(_image, prompts):
            self.assertIn("person", prompts)
            return [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [150, 40, 240, 140], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ]

        def verifier_fn(_image, _prompt):
            return {
                "single_person": True,
                "face_visible": True,
                "upper_body_visible": True,
                "lower_body_visible": True,
                "clear_human": True,
                "reason": "ok",
            }

        result = prepare_user_image_core(
            image,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=lambda _img: "identity: test subject. face: clear.",
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["url"], "https://example.com/prepared.png")
        self.assertEqual(result["meta"]["detect"].get("backend"), "grounding_dino")

    def test_rejects_when_grounding_detector_is_missing(self):
        image = Image.new("RGB", (320, 480), "white")
        result = prepare_user_image_core(
            image,
            grounding_detector_fn=None,
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=False,
        )
        self.assertEqual(result["error"], "detector_unavailable")
        self.assertEqual(int(result["status_code"]), 503)

    def test_rejects_multiple_people_from_grounding_gate(self):
        image = Image.new("RGB", (400, 600), "white")
        called = {"verifier": 0}

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [20, 20, 190, 590], "score": 0.82, "source": "grounding_dino"},
                {"label": "person", "bbox": [210, 20, 380, 590], "score": 0.78, "source": "grounding_dino"},
                {"label": "face", "bbox": [50, 40, 120, 120], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [30, 120, 170, 310], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [30, 300, 170, 580], "score": 0.68, "source": "grounding_dino"},
            ]

        def verifier_fn(_image, _prompt):
            called["verifier"] += 1
            return {}

        result = prepare_user_image_core(
            image,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertEqual(result["error"], "multiple_people")
        self.assertEqual(called["verifier"], 0)

    def test_rejects_when_face_is_missing_in_detection_gate(self):
        image = Image.new("RGB", (400, 600), "white")

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ]

        result = prepare_user_image_core(
            image,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=False,
        )

        self.assertEqual(result["error"], "face_hidden")

    def test_rejects_when_bottom_is_missing_in_detection_gate(self):
        image = Image.new("RGB", (400, 600), "white")
        called = {"verifier": 0}

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [150, 420, 240, 520], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
            ]

        def verifier_fn(_image, _prompt):
            called["verifier"] += 1
            return {}

        result = prepare_user_image_core(
            image,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertEqual(result["error"], "bottom_section_not_visible")
        self.assertEqual(called["verifier"], 0)

    def test_rejects_when_verifier_reports_lower_body_not_visible(self):
        image = Image.new("RGB", (400, 600), "white")

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [150, 40, 240, 140], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ]

        def verifier_fn(_image, _prompt):
            return {
                "single_person": True,
                "face_visible": True,
                "upper_body_visible": True,
                "lower_body_visible": False,
                "clear_human": True,
                "reason": "lower body is cropped",
            }

        result = prepare_user_image_core(
            image,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertEqual(result["error"], "bottom_section_not_visible")

    def test_rejects_when_verifier_is_required_but_missing(self):
        image = Image.new("RGB", (400, 600), "white")

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [150, 40, 240, 140], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ]

        result = prepare_user_image_core(
            image,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertEqual(result["error"], "verifier_unavailable")
        self.assertEqual(int(result["status_code"]), 503)

    def test_accepts_with_grounding_person_anchor_fallback_when_primary_has_no_person_label(self):
        image = Image.new("RGB", (400, 600), "white")
        called = {"anchor": 0}

        def grounding_detector_fn(_image, prompts):
            prompt_set = {str(p).strip().lower() for p in prompts}
            if "single person" in prompt_set:
                called["anchor"] += 1
                return [
                    {"label": "single person", "bbox": [55, 20, 345, 585], "score": 0.81, "source": "grounding_dino"},
                ]
            return [
                {"label": "face", "bbox": [155, 45, 240, 140], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ]

        def verifier_fn(_image, _prompt):
            return {
                "single_person": True,
                "face_visible": True,
                "upper_body_visible": True,
                "lower_body_visible": True,
                "clear_human": True,
                "reason": "ok",
            }

        result = prepare_user_image_core(
            image,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(called["anchor"], 1)
        self.assertTrue(result["meta"]["detect"].get("person_anchor_used"))


if __name__ == "__main__":
    unittest.main()
