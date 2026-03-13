import unittest

import numpy as np
from PIL import Image

from ai.modules.wardrobe.cloth_detection import ClothDetector


class _FakeFashionDetector:
    def __init__(self, detections):
        self._detections = detections

    def predict(self, _image, threshold=None):
        return list(self._detections)


class _FakeParser:
    def __init__(self, parsing, labels):
        self._parsing = np.asarray(parsing)
        self._labels = labels

    def parse(self, _image):
        return self._parsing

    def _runtime_labels(self):
        return dict(self._labels)

    def get_mask_for_category(self, parsing, category):
        label_id = self._labels.get(category)
        if label_id is None:
            return np.zeros_like(parsing, dtype=bool)
        return parsing == int(label_id)


class _FakeLegacyDetector:
    def detect_instances(self, _image):
        return [{"bbox": [0, 0, 60, 60], "mask": np.ones((60, 60), dtype=bool), "confidence": 0.9, "label": "top"}]

    def get_crops(self, image, instances):
        return [{
            "label": instances[0]["label"],
            "bbox": instances[0]["bbox"],
            "image": image.crop((0, 0, 60, 60)),
            "confidence": instances[0]["confidence"],
        }]


class TestClothDetector(unittest.TestCase):
    def test_legacy_fallback_used_for_weak_detector(self):
        image = Image.new("RGB", (100, 100), "white")
        detector = ClothDetector(
            fashion_detector=_FakeFashionDetector([
                {"bbox": [10, 10, 90, 95], "score": 0.31, "label": "dress", "source": "fashion_object_detection"}
            ]),
            legacy_detector=_FakeLegacyDetector(),
        )

        candidates = detector.detect_fashion_candidates(image)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["type"], "top")
        self.assertEqual(candidates[0]["source"], "legacy_yolo")

    def test_strong_fashion_detection_skips_legacy(self):
        image = Image.new("RGB", (100, 100), "white")
        detector = ClothDetector(
            fashion_detector=_FakeFashionDetector([
                {"bbox": [10, 10, 90, 90], "score": 0.88, "label": "top", "source": "fashion_object_detection"}
            ]),
            legacy_detector=_FakeLegacyDetector(),
        )

        candidates = detector.detect_fashion_candidates(image)
        self.assertEqual(len(candidates), 1)
        bbox = candidates[0]["bbox"]
        self.assertEqual(candidates[0]["type"], "top")
        self.assertEqual(candidates[0]["source"], "fashion_object_detection")
        self.assertGreaterEqual(bbox[0], 10)
        self.assertGreaterEqual(bbox[1], 10)
        self.assertLessEqual(bbox[2], 90)
        self.assertLessEqual(bbox[3], 90)

    def test_compare_backends_includes_both_paths(self):
        image = Image.new("RGB", (100, 100), "white")
        detector = ClothDetector(
            legacy_detector=_FakeLegacyDetector(),
            fashion_detector=_FakeFashionDetector([
                {"bbox": [10, 10, 90, 90], "score": 0.88, "label": "top", "source": "fashion_object_detection"}
            ]),
        )

        result = detector.compare_backends(image)
        self.assertEqual(len(result["fashion"]), 1)
        self.assertEqual(len(result["legacy"]), 1)

    def test_non_garment_labels_are_dropped_from_fashion_candidates(self):
        image = Image.new("RGB", (100, 100), "white")
        detector = ClothDetector(
            fashion_detector=_FakeFashionDetector([
                {"bbox": [5, 5, 30, 30], "score": 0.95, "label": "shoes", "source": "fashion_object_detection"},
                {"bbox": [35, 10, 80, 55], "score": 0.82, "label": "top", "source": "fashion_object_detection"},
            ]),
            legacy_detector=None,
        )

        result = detector.compare_backends(image)
        self.assertEqual(len(result["fashion"]), 1)
        self.assertEqual(result["fashion"][0]["type"], "top")
        self.assertEqual(result["fashion"][0]["label"], "top")

    def test_strong_top_bottom_pair_does_not_trigger_legacy_fallback(self):
        image = Image.new("RGB", (100, 100), "white")
        detector = ClothDetector(
            fashion_detector=_FakeFashionDetector([
                {"bbox": [20, 10, 70, 46], "score": 0.84, "label": "top", "source": "fashion_object_detection"},
                {"bbox": [18, 40, 72, 88], "score": 0.78, "label": "bottom", "source": "fashion_object_detection"},
            ]),
            legacy_detector=_FakeLegacyDetector(),
        )

        candidates = detector.detect_fashion_candidates(image)
        self.assertEqual(len(candidates), 2)
        self.assertEqual({item["type"] for item in candidates}, {"top", "bottom"})
        self.assertTrue(all(item["source"] == "fashion_object_detection" for item in candidates))

    def test_composite_bottom_label_is_not_retyped_to_requested_top(self):
        image = Image.new("RGB", (100, 100), "white")
        detector = ClothDetector(
            fashion_detector=_FakeFashionDetector([
                {"bbox": [18, 40, 72, 96], "score": 0.91, "label": "wide leg trousers", "source": "fashion_object_detection"},
            ]),
            legacy_detector=None,
        )

        candidates = detector.detect_fashion_candidates(image, requested_type="top")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["label"], "wide leg trousers")
        self.assertEqual(candidates[0]["type"], "bottom")
        self.assertEqual(candidates[0]["type_source"], "detector")


if __name__ == "__main__":
    unittest.main()
