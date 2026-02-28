import io
import unittest
from typing import Any, List

from fastapi.testclient import TestClient
from PIL import Image

from ai import main as api_main


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (128, 192)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _AnalyzePatchContext:
    def __init__(self, crop_count: int, labels=None):
        self.crop_count = crop_count
        self.labels = labels or ["top" if i == 0 else "bottom" for i in range(crop_count)]
        self.orig_detect = api_main.engine.yolo.detect_instances
        self.orig_get_crops = api_main.engine.yolo.get_crops
        self.orig_desc_short = api_main.engine.florence.describe_garment_short
        self.orig_desc_detailed = api_main.engine.florence.describe_garment
        self.orig_upload = api_main._upload_or_raise
        self.orig_hybrid = api_main.USE_FLORENCE_HYBRID_VERIFY
        self.orig_require_selection = api_main.ANALYZE_REQUIRE_SELECTION
        self.orig_caption_mode = api_main.ANALYZE_CAPTION_MODE
        self.orig_parser_split = api_main.ANALYZE_ENABLE_PARSER_SPLIT
        self.orig_heuristic_split = api_main.ANALYZE_ENABLE_HEURISTIC_SPLIT
        self.orig_extract_cloth = api_main.ANALYZE_EXTRACT_CLOTH
        self.orig_prompt_from_extracted = api_main.ANALYZE_PROMPT_FROM_EXTRACTED
        self.orig_vton_fallback = api_main.ANALYZE_VTON_FALLBACK_ENABLED

    def __enter__(self):
        def fake_detect_instances(_img: Image.Image) -> List[Any]:
            return [{"label": "top", "confidence": 0.9, "bbox": (0, 0, 64, 96), "mask": None}] * self.crop_count

        def fake_get_crops(_img: Image.Image, _instances: List[Any]) -> List[Any]:
            out = []
            for idx in range(self.crop_count):
                crop = Image.new("RGB", (96, 120), (80 + idx, 70, 60))
                label = self.labels[idx] if idx < len(self.labels) else self.labels[-1]
                out.append(
                    {
                        "id": idx,
                        "label": label,
                        "confidence": 0.91 - (0.1 * idx),
                        "image": crop,
                        "bbox": [10 + (idx * 3), 12, 88 + (idx * 3), 118],
                        "mask": None,
                    }
                )
            return out

        api_main.engine.yolo.detect_instances = fake_detect_instances
        api_main.engine.yolo.get_crops = fake_get_crops
        api_main.engine.florence.describe_garment_short = lambda _img: "short floral cotton top"
        api_main.engine.florence.describe_garment = lambda _img: "detailed floral cotton top with long sleeves"
        api_main._upload_or_raise = lambda _bytes, container=None: f"https://example.local/{container or 'wardrobe'}.png"
        api_main.USE_FLORENCE_HYBRID_VERIFY = False
        api_main.ANALYZE_REQUIRE_SELECTION = True
        api_main.ANALYZE_CAPTION_MODE = "short"
        api_main.ANALYZE_ENABLE_PARSER_SPLIT = False
        api_main.ANALYZE_ENABLE_HEURISTIC_SPLIT = False
        api_main.ANALYZE_EXTRACT_CLOTH = False
        api_main.ANALYZE_PROMPT_FROM_EXTRACTED = False
        api_main.ANALYZE_VTON_FALLBACK_ENABLED = False
        return self

    def __exit__(self, exc_type, exc, tb):
        api_main.engine.yolo.detect_instances = self.orig_detect
        api_main.engine.yolo.get_crops = self.orig_get_crops
        api_main.engine.florence.describe_garment_short = self.orig_desc_short
        api_main.engine.florence.describe_garment = self.orig_desc_detailed
        api_main._upload_or_raise = self.orig_upload
        api_main.USE_FLORENCE_HYBRID_VERIFY = self.orig_hybrid
        api_main.ANALYZE_REQUIRE_SELECTION = self.orig_require_selection
        api_main.ANALYZE_CAPTION_MODE = self.orig_caption_mode
        api_main.ANALYZE_ENABLE_PARSER_SPLIT = self.orig_parser_split
        api_main.ANALYZE_ENABLE_HEURISTIC_SPLIT = self.orig_heuristic_split
        api_main.ANALYZE_EXTRACT_CLOTH = self.orig_extract_cloth
        api_main.ANALYZE_PROMPT_FROM_EXTRACTED = self.orig_prompt_from_extracted
        api_main.ANALYZE_VTON_FALLBACK_ENABLED = self.orig_vton_fallback


class TestAnalyzeSelectionFlow(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_main.app)
        self.file_bytes = _png_bytes((120, 90, 80))

    def test_multi_garment_requires_selection(self):
        with _AnalyzePatchContext(crop_count=2):
            resp = self.client.post(
                "/analyze",
                files={"file": ("multi.png", self.file_bytes, "image/png")},
                data={"debug": "true"},
            )
            self.assertEqual(resp.status_code, 400)
            body = resp.json()
            detail = body.get("detail", {})
            self.assertEqual(detail.get("status"), "selection_required")
            self.assertEqual(detail.get("selection_field"), "selected_index")
            self.assertEqual(detail.get("detected_items"), 2)
            self.assertEqual(len(detail.get("items", [])), 2)
            self.assertIn("promptDescription", detail["items"][0])

    def test_multi_garment_selected_index_returns_single_item(self):
        with _AnalyzePatchContext(crop_count=2):
            resp = self.client.post(
                "/analyze",
                files={"file": ("multi.png", self.file_bytes, "image/png")},
                data={"selected_index": "1", "debug": "true"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertFalse(body.get("selection_required"))
            self.assertEqual(body.get("detected_items"), 2)
            self.assertEqual(body.get("selected_index"), 1)
            item = body.get("item")
            self.assertIsNotNone(item)
            self.assertEqual(item.get("garment_id"), 1)
            self.assertIn("promptDescription", item)
            self.assertEqual(len(body.get("items", [])), 1)

    def test_single_garment_returns_success_directly(self):
        with _AnalyzePatchContext(crop_count=1):
            resp = self.client.post(
                "/analyze",
                files={"file": ("single.png", self.file_bytes, "image/png")},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body.get("detected_items"), 1)
            self.assertEqual(body.get("selected_index"), 0)
            item = body.get("item")
            self.assertIsNotNone(item)
            self.assertIn("promptDescription", item)

    def test_same_type_duplicates_auto_collapse(self):
        with _AnalyzePatchContext(crop_count=2, labels=["dress", "dress"]):
            resp = self.client.post(
                "/analyze",
                files={"file": ("single_dress.png", self.file_bytes, "image/png")},
                data={"debug": "true"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body.get("detected_items"), 1)
            dbg = body.get("debug", {})
            self.assertTrue(bool(dbg.get("collapsed_same_type")) or int(dbg.get("dedup_removed", 0)) >= 1)
            self.assertGreaterEqual(int(dbg.get("dedup_removed", 0)), 1)


if __name__ == "__main__":
    unittest.main()
