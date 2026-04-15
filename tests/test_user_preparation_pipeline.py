import io
import shutil
import unittest

from PIL import Image

from utils.user_preparation import (
    prepare_user_image_core,
    _build_user_prepare_prompt,
    _extract_user_prepare_prompt_bundle,
)

TEST_RESIZE_METHOD = "pillow_lanczos"


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
        image = Image.new("RGB", (800, 1200), "white")

        def grounding_detector_fn(_image, prompts):
            self.assertIn("person", prompts)
            return [
                {"label": "person", "bbox": [200, 100, 1400, 2300], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [650, 180, 950, 520], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [420, 520, 1180, 1160], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [420, 1160, 1180, 2220], "score": 0.68, "source": "grounding_dino"},
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
            resize_method=TEST_RESIZE_METHOD,
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
        self.assertEqual(result["meta"]["prepared_source_image_size"], {"width": 800, "height": 1200})
        self.assertEqual(result["meta"]["prepared_image_size"], {"width": 800, "height": 1200})
        self.assertEqual(result["meta"]["prepared_image_mode"], "original_full_frame")
        self.assertEqual(result["meta"]["prepare_policy"]["resize_action"], "none")
        self.assertTrue(result["meta"]["prepare_policy"]["jpeg"]["within_budget"])
        self.assertEqual(result["meta"]["prepare_policy"]["target_long_edge"], 1024)

    def test_prepare_can_force_fixed_long_edge_downscale(self):
        image = Image.new("RGB", (1600, 2400), "white")

        def grounding_detector_fn(_image, prompts):
            self.assertIn("person", prompts)
            return [
                {"label": "person", "bbox": [200, 100, 1400, 2300], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [650, 180, 950, 520], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [420, 520, 1180, 1160], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [420, 1160, 1180, 2220], "score": 0.68, "source": "grounding_dino"},
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
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=lambda _img: "identity: test subject. face: clear.",
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            target_height=1024,
            keep_long_edge_min=1024,
            output_max_long_edge=1024,
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(max(result["meta"]["prepared_image_size"].values()), 1024)
        self.assertEqual(result["meta"]["prepare_policy"]["target_long_edge"], 1024)
        self.assertEqual(result["meta"]["prepare_policy"]["max_long_edge"], 1024)

    def test_prepare_can_force_fixed_long_edge_upscale(self):
        image = Image.new("RGB", (600, 900), "white")

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
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=lambda _img: "identity: test subject. face: clear.",
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            target_height=1024,
            keep_long_edge_min=1024,
            output_max_long_edge=1024,
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(max(result["meta"]["prepared_image_size"].values()), 1024)
        self.assertEqual(result["meta"]["prepare_policy"]["resize_action"], "upscale")

    def test_rejects_when_grounding_detector_is_missing(self):
        image = Image.new("RGB", (800, 1200), "white")
        result = prepare_user_image_core(
            image,
            resize_method=TEST_RESIZE_METHOD,
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
        image = Image.new("RGB", (800, 1200), "white")
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
            resize_method=TEST_RESIZE_METHOD,
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
        image = Image.new("RGB", (800, 1200), "white")

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ]

        result = prepare_user_image_core(
            image,
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=False,
        )

        self.assertEqual(result["error"], "face_hidden")

    def test_rejects_when_bottom_is_missing_in_detection_gate(self):
        image = Image.new("RGB", (800, 1200), "white")
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
            resize_method=TEST_RESIZE_METHOD,
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
        image = Image.new("RGB", (800, 1200), "white")

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
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertEqual(result["error"], "bottom_section_not_visible")

    def test_rejects_when_verifier_is_required_but_missing(self):
        image = Image.new("RGB", (800, 1200), "white")

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [150, 40, 240, 140], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ]

        result = prepare_user_image_core(
            image,
            resize_method=TEST_RESIZE_METHOD,
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
        image = Image.new("RGB", (800, 1200), "white")
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
            resize_method=TEST_RESIZE_METHOD,
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

    def test_prepare_core_uses_prepared_image_fn_for_upload(self):
        image = Image.new("RGB", (538, 800), "white")
        uploaded = {}

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [40, 20, 500, 780], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [180, 40, 300, 180], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [120, 160, 420, 420], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [120, 420, 430, 780], "score": 0.68, "source": "grounding_dino"},
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

        def prepared_image_fn(source_image):
            self.assertEqual(source_image.size, (538, 800))
            return source_image.resize((689, 1024))

        def upload_fn(payload):
            uploaded["bytes"] = payload
            prepared = Image.open(io.BytesIO(payload))
            uploaded["size"] = prepared.size
            return "https://example.com/prepared-upscaled.jpg"

        result = prepare_user_image_core(
            image,
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=lambda _img: "A young woman with brown long loose hair and slim body build.",
            prepared_image_fn=prepared_image_fn,
            upload_fn=upload_fn,
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(uploaded["size"], (689, 1024))
        self.assertEqual(result["meta"]["prepared_source_image_size"], {"width": 538, "height": 800})
        self.assertEqual(result["meta"]["prepared_image_size"], {"width": 689, "height": 1024})
        self.assertEqual(result["meta"]["prepared_image_mode"], "ai_upscaled_normalized_full_frame")
        self.assertTrue(result["meta"]["prepare_policy"]["upscale"]["used"])
        self.assertEqual(result["meta"]["prepare_policy"]["jpeg"]["format"], "jpg")

    def test_rejects_when_longest_side_is_below_minimum(self):
        image = Image.new("RGB", (412, 634), "white")

        result = prepare_user_image_core(
            image,
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=lambda *_args, **_kwargs: [],
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=False,
        )

        self.assertEqual(result["error"], "image_too_small")
        self.assertEqual(result["meta"]["source_long_edge"], 634)
        self.assertEqual(result["meta"]["min_required_long_edge"], 768)

    def test_accepts_when_short_side_is_below_768_but_long_side_meets_minimum(self):
        image = Image.new("RGB", (1248, 720), "white")

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [140, 30, 1080, 700], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [460, 50, 760, 220], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [350, 210, 860, 430], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [360, 420, 860, 700], "score": 0.68, "source": "grounding_dino"},
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
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=lambda _img: "A young woman with brown long loose hair and slim body build.",
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["meta"]["prepared_source_image_size"], {"width": 1248, "height": 720})
        self.assertEqual(result["meta"]["prepared_image_size"], {"width": 1248, "height": 720})
        self.assertEqual(result["meta"]["prepare_policy"]["source_long_edge"], 1248)
        self.assertEqual(result["meta"]["prepared_image_mode"], "original_full_frame")
        self.assertEqual(result["meta"]["prepare_policy"]["resize_action"], "none")

    def test_prepare_core_adapts_jpeg_quality_to_size_budget(self):
        image = Image.effect_noise((1024, 1536), 100).convert("RGB")
        uploaded = {}

        def grounding_detector_fn(_image, _prompts):
            return [
                {"label": "person", "bbox": [120, 40, 900, 1490], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [320, 80, 650, 360], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [220, 300, 760, 760], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [220, 760, 800, 1480], "score": 0.68, "source": "grounding_dino"},
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

        def upload_fn(payload):
            uploaded["bytes"] = len(payload)
            uploaded["magic"] = payload[:3]
            return "https://example.com/prepared-budget.jpg"

        result = prepare_user_image_core(
            image,
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=grounding_detector_fn,
            verifier_fn=verifier_fn,
            description_fn=lambda _img: "A young woman with brown long loose hair and slim body build.",
            upload_fn=upload_fn,
            output_max_bytes=250 * 1024,
            jpeg_quality=92,
            jpeg_min_quality=52,
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(uploaded["magic"], b"\xff\xd8\xff")
        self.assertEqual(result["meta"]["prepare_policy"]["jpeg"]["bytes"], uploaded["bytes"])
        self.assertLess(result["meta"]["prepare_policy"]["jpeg"]["quality"], 92)
        self.assertGreaterEqual(result["meta"]["prepare_policy"]["jpeg"]["quality"], 52)
        self.assertEqual(
            result["meta"]["prepare_policy"]["jpeg"]["within_budget"],
            uploaded["bytes"] <= 250 * 1024,
        )

    def test_prepare_core_uses_jpeg_passthrough_when_stable_and_no_resize_needed(self):
        image = Image.new("RGB", (764, 1142), "white")
        src_buf = io.BytesIO()
        image.save(src_buf, format="JPEG", quality=87)
        source_payload = src_buf.getvalue()
        uploaded = {}

        def upload_fn(payload):
            uploaded["bytes"] = payload
            return "https://example.com/prepared-passthrough.jpg"

        result = prepare_user_image_core(
            image,
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=lambda *_args, **_kwargs: [
                {"label": "person", "bbox": [80, 20, 680, 1100], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [280, 40, 460, 220], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [220, 220, 560, 560], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [220, 560, 560, 1100], "score": 0.68, "source": "grounding_dino"},
            ],
            verifier_fn=lambda *_args, **_kwargs: {
                "single_person": True,
                "face_visible": True,
                "upper_body_visible": True,
                "lower_body_visible": True,
                "clear_human": True,
            },
            description_fn=lambda _img: "identity: test subject. face: clear.",
            upload_fn=upload_fn,
            source_upload_payload=source_payload,
            source_upload_format="JPEG",
            source_exif_orientation=1,
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(uploaded["bytes"], source_payload)
        self.assertEqual(result["meta"]["prepared_image_size"], {"width": 764, "height": 1142})
        self.assertTrue(result["meta"]["prepare_policy"]["passthrough_used"])
        self.assertFalse(result["meta"]["prepare_policy"]["orientation_fixed"])
        self.assertEqual(result["meta"]["prepare_policy"]["resize_action"], "none")
        self.assertFalse(result["meta"]["prepare_policy"]["jpeg_reencoded"])
        self.assertIn("jpeg_passthrough", result["meta"]["prepare_policy"]["actions"])
        self.assertEqual(result["meta"]["prepare_policy"]["jpeg"]["backend"], "passthrough")

    def test_prepare_core_reencodes_jpeg_when_orientation_fix_is_needed(self):
        image = Image.new("RGB", (764, 1142), "white")
        src_buf = io.BytesIO()
        image.save(src_buf, format="JPEG", quality=87)
        source_payload = src_buf.getvalue()

        result = prepare_user_image_core(
            image,
            resize_method=TEST_RESIZE_METHOD,
            grounding_detector_fn=lambda *_args, **_kwargs: [
                {"label": "person", "bbox": [80, 20, 680, 1100], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [280, 40, 460, 220], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [220, 220, 560, 560], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [220, 560, 560, 1100], "score": 0.68, "source": "grounding_dino"},
            ],
            verifier_fn=lambda *_args, **_kwargs: {
                "single_person": True,
                "face_visible": True,
                "upper_body_visible": True,
                "lower_body_visible": True,
                "clear_human": True,
            },
            description_fn=lambda _img: "identity: test subject. face: clear.",
            upload_fn=lambda _bytes: "https://example.com/prepared-reencoded.jpg",
            source_upload_payload=source_payload,
            source_upload_format="JPEG",
            source_exif_orientation=6,
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertFalse(result["meta"]["prepare_policy"]["passthrough_used"])
        self.assertTrue(result["meta"]["prepare_policy"]["orientation_fixed"])
        self.assertFalse("jpeg_passthrough" in result["meta"]["prepare_policy"]["actions"])
        self.assertIn("jpeg_encoded", result["meta"]["prepare_policy"]["actions"])
        self.assertTrue(result["meta"]["prepare_policy"]["jpeg_reencoded"])

    def test_rejects_invalid_resize_method(self):
        image = Image.new("RGB", (800, 1200), "white")

        result = prepare_user_image_core(
            image,
            resize_method="imagemagick",
            grounding_detector_fn=lambda *_args, **_kwargs: [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [150, 40, 240, 140], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ],
            verifier_fn=lambda *_args, **_kwargs: {
                "single_person": True,
                "face_visible": True,
                "upper_body_visible": True,
                "lower_body_visible": True,
                "clear_human": True,
            },
            description_fn=lambda _img: "A young woman with brown long loose hair and slim body build.",
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertEqual(result["error"], "invalid_resize_method")

    @unittest.skipUnless(shutil.which("vipsthumbnail"), "libvips CLI is required for this test")
    def test_accepts_with_explicit_libvips_resize_backend(self):
        image = Image.new("RGB", (800, 1200), "white")

        result = prepare_user_image_core(
            image,
            resize_method="libvips",
            grounding_detector_fn=lambda *_args, **_kwargs: [
                {"label": "person", "bbox": [60, 20, 340, 580], "score": 0.82, "source": "grounding_dino"},
                {"label": "face", "bbox": [150, 40, 240, 140], "score": 0.74, "source": "grounding_dino"},
                {"label": "upper body", "bbox": [100, 120, 310, 320], "score": 0.65, "source": "grounding_dino"},
                {"label": "lower body", "bbox": [100, 300, 320, 570], "score": 0.68, "source": "grounding_dino"},
            ],
            verifier_fn=lambda *_args, **_kwargs: {
                "single_person": True,
                "face_visible": True,
                "upper_body_visible": True,
                "lower_body_visible": True,
                "clear_human": True,
            },
            description_fn=lambda _img: "identity: test subject. face: clear.",
            upload_fn=lambda _bytes: "https://example.com/prepared.jpg",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["meta"]["prepare_policy"]["resize"]["used"], "libvips")
        self.assertEqual(result["meta"]["prepare_policy"]["jpeg"]["backend"], "libvips")


if __name__ == "__main__":
    unittest.main()
