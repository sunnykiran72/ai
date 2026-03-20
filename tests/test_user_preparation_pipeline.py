import unittest

from PIL import Image

from utils.user_preparation import prepare_user_image_core


class TestUserPreparationPipeline(unittest.TestCase):
    def test_accepts_single_person_with_visible_face(self):
        image = Image.new("RGB", (320, 480), "white")

        def detector_fn(_image, conf=0.25, iou=0.45):
            return [{"bbox": [24, 12, 272, 444], "confidence": 0.96, "area_ratio": 0.69}]

        def verifier_fn(_image, prompt):
            self.assertIn("dominant foreground person", prompt)
            return {
                "single_person": True,
                "face_visible": True,
                "full_body_visible": True,
                "clear_human": True,
                "reason": "ok",
            }

        def upload_fn(image_bytes):
            self.assertGreater(len(image_bytes), 0)
            return "https://example.com/prepared.png"

        def description_fn(_image):
            return "identity: test subject. face: clear. pose: standing. preserve: pose"

        result = prepare_user_image_core(
            image,
            person_detector_fn=detector_fn,
            verifier_fn=verifier_fn,
            description_fn=description_fn,
            upload_fn=upload_fn,
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["url"], "https://example.com/prepared.png")
        self.assertIn("test subject", result["promptDescription"])
        self.assertEqual(result["meta"]["detect"]["count"], 1)
        self.assertTrue(result["meta"]["verification"]["parsed"]["face_visible"])

    def test_accepts_pose_agnostic_full_body_when_face_detector_is_present(self):
        image = Image.new("RGB", (320, 480), "white")

        def detector_fn(_image, conf=0.25, iou=0.45):
            return [{"bbox": [18, 12, 302, 448], "confidence": 0.96, "area_ratio": 0.78}]

        def face_detector_fn(_image):
            return [{"bbox": [120, 26, 170, 96], "confidence": 0.93, "source": "stub"}]

        result = prepare_user_image_core(
            image,
            person_detector_fn=detector_fn,
            face_detector_fn=face_detector_fn,
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=False,
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["meta"]["face"]["body_visibility"]["major_axis"], "vertical")
        self.assertTrue(result["meta"]["face"]["body_visibility"]["visible"])

    def test_accepts_horizontal_full_body_pose_when_face_is_at_body_end(self):
        image = Image.new("RGB", (500, 400), "white")

        def detector_fn(_image, conf=0.25, iou=0.45):
            return [{"bbox": [20, 110, 470, 260], "confidence": 0.95, "area_ratio": 0.62}]

        def face_detector_fn(_image):
            return [{"bbox": [32, 126, 88, 196], "confidence": 0.92, "source": "stub"}]

        result = prepare_user_image_core(
            image,
            person_detector_fn=detector_fn,
            face_detector_fn=face_detector_fn,
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=False,
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["meta"]["face"]["body_visibility"]["major_axis"], "horizontal")
        self.assertTrue(result["meta"]["face"]["body_visibility"]["visible"])

    def test_rejects_cropped_body_even_when_face_is_visible(self):
        image = Image.new("RGB", (320, 480), "white")

        def detector_fn(_image, conf=0.25, iou=0.45):
            return [{"bbox": [18, 12, 150, 220], "confidence": 0.95, "area_ratio": 0.20}]

        def face_detector_fn(_image):
            return [{"bbox": [120, 24, 170, 96], "confidence": 0.92, "source": "stub"}]

        result = prepare_user_image_core(
            image,
            person_detector_fn=detector_fn,
            face_detector_fn=face_detector_fn,
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=False,
        )

        self.assertEqual(result["error"], "full_body_not_visible")
        self.assertIn("full body is not visible", result["message"].lower())

    def test_rejects_multiple_prominent_people(self):
        image = Image.new("RGB", (320, 480), "white")

        def detector_fn(_image, conf=0.25, iou=0.45):
            return [
                {"bbox": [18, 16, 156, 438], "confidence": 0.91, "area_ratio": 0.47},
                {"bbox": [168, 20, 304, 440], "confidence": 0.89, "area_ratio": 0.44},
            ]

        result = prepare_user_image_core(
            image,
            person_detector_fn=detector_fn,
            verifier_fn=None,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=False,
        )

        self.assertEqual(result["error"], "multiple_people")
        self.assertIn("dominant person", result["message"].lower())

    def test_rejects_hidden_face_from_verifier(self):
        image = Image.new("RGB", (320, 480), "white")

        def detector_fn(_image, conf=0.25, iou=0.45):
            return [{"bbox": [24, 12, 272, 444], "confidence": 0.96, "area_ratio": 0.69}]

        def verifier_fn(_image, prompt):
            self.assertIn("full_body_visible", prompt)
            return {
                "single_person": True,
                "face_visible": False,
                "full_body_visible": True,
                "clear_human": True,
                "reason": "face hidden by hand",
            }

        result = prepare_user_image_core(
            image,
            person_detector_fn=detector_fn,
            verifier_fn=verifier_fn,
            description_fn=None,
            upload_fn=lambda _bytes: "https://example.com/prepared.png",
            blur_check_enabled=False,
            verification_required=True,
        )

        self.assertEqual(result["error"], "face_hidden")
        self.assertIn("face hidden", result["message"].lower())


if __name__ == "__main__":
    unittest.main()
