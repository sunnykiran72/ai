import io
import json
import re
import unittest
from typing import Any, Dict, List

from fastapi.testclient import TestClient
from PIL import Image

from ai import main as api_main


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (128, 192)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _parse_multipart_response(resp) -> Dict[str, Dict[str, Any]]:
    content_type = resp.headers.get("content-type", "")
    boundary_match = re.search(r"boundary=([^;]+)", content_type)
    if not boundary_match:
        raise AssertionError(f"Missing multipart boundary in content-type: {content_type}")
    boundary = boundary_match.group(1).strip().strip('"')
    boundary_bytes = f"--{boundary}".encode("utf-8")

    parts: Dict[str, Dict[str, Any]] = {}
    for raw in resp.content.split(boundary_bytes):
        part = raw.strip()
        if not part or part == b"--":
            continue
        if part.startswith(b"--"):
            part = part[2:]
        if part.startswith(b"\r\n"):
            part = part[2:]

        headers_blob, sep, body = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]

        header_lines = headers_blob.decode("utf-8", errors="ignore").split("\r\n")
        disposition = ""
        for line in header_lines:
            if line.lower().startswith("content-disposition:"):
                disposition = line
                break
        name_match = re.search(r'name="([^"]+)"', disposition)
        filename_match = re.search(r'filename="([^"]+)"', disposition)
        if not name_match:
            continue

        name = name_match.group(1)
        parts[name] = {
            "filename": filename_match.group(1) if filename_match else None,
            "bytes": body,
            "headers": header_lines,
        }
    return parts


class _AnalyzePatchContext:
    def __init__(self, crop_count: int, labels=None, bboxes=None):
        self.crop_count = crop_count
        self.labels = labels or ["top" if i == 0 else "bottom" for i in range(crop_count)]
        self.bboxes = bboxes
        self.orig_detect = api_main.engine.yolo.detect_instances
        self.orig_get_crops = api_main.engine.yolo.get_crops
        self.orig_desc_short = api_main.engine.florence.describe_garment_short
        self.orig_desc_detailed = api_main.engine.florence.describe_garment
        self.orig_run_vton = getattr(api_main, "_run_vton_cloth_only_fallback", None)
        self.orig_hybrid = api_main.USE_FLORENCE_HYBRID_VERIFY
        self.orig_require_selection = api_main.ANALYZE_REQUIRE_SELECTION
        self.orig_caption_mode = api_main.ANALYZE_CAPTION_MODE
        self.orig_parser_split = api_main.ANALYZE_ENABLE_PARSER_SPLIT
        self.orig_parser_prerouting = api_main.ANALYZE_USE_PARSER_FOR_PREROUTING
        self.orig_heuristic_split = api_main.ANALYZE_ENABLE_HEURISTIC_SPLIT
        self.orig_extract_cloth = api_main.ANALYZE_EXTRACT_CLOTH
        self.orig_parser_post_extract = api_main.ANALYZE_USE_PARSER_POST_EXTRACT
        self.orig_prompt_from_extracted = api_main.ANALYZE_PROMPT_FROM_EXTRACTED
        self.orig_vton_fallback = getattr(api_main, "ANALYZE_VTON_FALLBACK_ENABLED", None)
        self.orig_vton_endpoint = getattr(api_main, "ANALYZE_VTON_CLOTH_ONLY_ENDPOINT", None)
        self.orig_blur_enabled = api_main.ANALYZE_BLUR_CHECK_ENABLED
        self.orig_min_accept_conf = api_main.ANALYZE_MIN_ACCEPT_CONFIDENCE
        self.orig_max_file = api_main.ANALYZE_MAX_FILE_BYTES
        self.orig_collapse_same_type = api_main.ANALYZE_COLLAPSE_SAME_TYPE
        self.orig_collapse_same_type_iou = api_main.ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU

    def __enter__(self):
        def fake_detect_instances(_img: Image.Image) -> List[Any]:
            return [{"label": "top", "confidence": 0.9, "bbox": (0, 0, 64, 96), "mask": None}] * self.crop_count

        def fake_get_crops(_img: Image.Image, _instances: List[Any]) -> List[Any]:
            out = []
            for idx in range(self.crop_count):
                crop = Image.new("RGB", (96, 120), (80 + idx, 70, 60))
                label = self.labels[idx] if idx < len(self.labels) else self.labels[-1]
                bbox = [10 + (idx * 3), 12, 88 + (idx * 3), 118]
                if self.bboxes and idx < len(self.bboxes):
                    bbox = self.bboxes[idx]
                out.append(
                    {
                        "id": idx,
                        "label": label,
                        "confidence": 0.91 - (0.1 * idx),
                        "image": crop,
                        "bbox": bbox,
                        "mask": None,
                    }
                )
            return out

        def fake_vton(_image_url: str, garment_type: str, vto_mode: bool = False) -> Dict[str, Any]:
            out = Image.new("RGBA", (128, 192), (200, 40, 70, 255))
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            return {
                "url": "https://example.local/extracted.png",
                "raw_url": "https://example.local/raw.png",
                "_processed_image_bytes": buf.getvalue(),
                "meta": {
                    "path": "vton_fallback",
                    "endpoint": "http://fake-vton.local/v1/cloth-only",
                    "category": garment_type,
                    "vto_mode": bool(vto_mode),
                },
            }

        api_main.engine.yolo.detect_instances = fake_detect_instances
        api_main.engine.yolo.get_crops = fake_get_crops
        api_main.engine.florence.describe_garment_short = lambda _img: "short floral cotton top"
        api_main.engine.florence.describe_garment = lambda _img: "detailed floral cotton top with long sleeves"
        if hasattr(api_main, "_run_vton_cloth_only_fallback"):
            api_main._run_vton_cloth_only_fallback = fake_vton
        api_main.USE_FLORENCE_HYBRID_VERIFY = False
        api_main.ANALYZE_REQUIRE_SELECTION = True
        api_main.ANALYZE_CAPTION_MODE = "short"
        api_main.ANALYZE_ENABLE_PARSER_SPLIT = False
        api_main.ANALYZE_USE_PARSER_FOR_PREROUTING = False
        api_main.ANALYZE_ENABLE_HEURISTIC_SPLIT = False
        api_main.ANALYZE_EXTRACT_CLOTH = True
        api_main.ANALYZE_USE_PARSER_POST_EXTRACT = False
        api_main.ANALYZE_PROMPT_FROM_EXTRACTED = False
        if hasattr(api_main, "ANALYZE_VTON_FALLBACK_ENABLED"):
            api_main.ANALYZE_VTON_FALLBACK_ENABLED = True
        if hasattr(api_main, "ANALYZE_VTON_CLOTH_ONLY_ENDPOINT"):
            api_main.ANALYZE_VTON_CLOTH_ONLY_ENDPOINT = "http://fake-vton.local/v1/cloth-only"
        api_main.ANALYZE_BLUR_CHECK_ENABLED = False
        api_main.ANALYZE_MIN_ACCEPT_CONFIDENCE = 0.0
        api_main.ANALYZE_MAX_FILE_BYTES = 3 * 1024 * 1024
        api_main.ANALYZE_COLLAPSE_SAME_TYPE = False
        api_main.ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU = 0.85
        return self

    def __exit__(self, exc_type, exc, tb):
        api_main.engine.yolo.detect_instances = self.orig_detect
        api_main.engine.yolo.get_crops = self.orig_get_crops
        api_main.engine.florence.describe_garment_short = self.orig_desc_short
        api_main.engine.florence.describe_garment = self.orig_desc_detailed
        if self.orig_run_vton is not None and hasattr(api_main, "_run_vton_cloth_only_fallback"):
            api_main._run_vton_cloth_only_fallback = self.orig_run_vton
        api_main.USE_FLORENCE_HYBRID_VERIFY = self.orig_hybrid
        api_main.ANALYZE_REQUIRE_SELECTION = self.orig_require_selection
        api_main.ANALYZE_CAPTION_MODE = self.orig_caption_mode
        api_main.ANALYZE_ENABLE_PARSER_SPLIT = self.orig_parser_split
        api_main.ANALYZE_USE_PARSER_FOR_PREROUTING = self.orig_parser_prerouting
        api_main.ANALYZE_ENABLE_HEURISTIC_SPLIT = self.orig_heuristic_split
        api_main.ANALYZE_EXTRACT_CLOTH = self.orig_extract_cloth
        api_main.ANALYZE_USE_PARSER_POST_EXTRACT = self.orig_parser_post_extract
        api_main.ANALYZE_PROMPT_FROM_EXTRACTED = self.orig_prompt_from_extracted
        if self.orig_vton_fallback is not None and hasattr(api_main, "ANALYZE_VTON_FALLBACK_ENABLED"):
            api_main.ANALYZE_VTON_FALLBACK_ENABLED = self.orig_vton_fallback
        if self.orig_vton_endpoint is not None and hasattr(api_main, "ANALYZE_VTON_CLOTH_ONLY_ENDPOINT"):
            api_main.ANALYZE_VTON_CLOTH_ONLY_ENDPOINT = self.orig_vton_endpoint
        api_main.ANALYZE_BLUR_CHECK_ENABLED = self.orig_blur_enabled
        api_main.ANALYZE_MIN_ACCEPT_CONFIDENCE = self.orig_min_accept_conf
        api_main.ANALYZE_MAX_FILE_BYTES = self.orig_max_file
        api_main.ANALYZE_COLLAPSE_SAME_TYPE = self.orig_collapse_same_type
        api_main.ANALYZE_COLLAPSE_SAME_TYPE_MIN_IOU = self.orig_collapse_same_type_iou


class TestAnalyzeSelectionFlow(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_main.app)
        self.file_bytes = _png_bytes((120, 90, 80))
        self.headers = {"Authorization": "Bearer local-test-token"}

    def test_multi_garment_requires_selection_multipart(self):
        with _AnalyzePatchContext(crop_count=2):
            resp = self.client.post(
                "/analyze",
                files={"image": ("multi.png", self.file_bytes, "image/png")},
                headers=self.headers,
            )
            self.assertEqual(resp.status_code, 400)

            parts = _parse_multipart_response(resp)
            self.assertIn("metadata", parts)
            metadata = json.loads(parts["metadata"]["bytes"].decode("utf-8"))

            self.assertEqual(metadata.get("status"), 400)
            data = metadata.get("data", {})
            self.assertEqual(data.get("result"), "REJECTED")
            self.assertIn("MULTI_ITEM_SELECTION_REQUIRED", data.get("reason_codes", []))
            self.assertEqual(len(data.get("item_breakdown", [])), 2)

            part_names = [p.get("part_name") for p in data.get("multipart_data", {}).get("parts", [])]
            self.assertIn("item_1", part_names)
            self.assertIn("item_2", part_names)
            self.assertIn("item_1", parts)
            self.assertIn("item_2", parts)

    def test_multi_garment_selected_index_returns_success_with_extracted_cloth(self):
        with _AnalyzePatchContext(crop_count=2):
            resp = self.client.post(
                "/analyze",
                files={"image": ("multi.png", self.file_bytes, "image/png")},
                data={"selected_index": "2"},
                headers=self.headers,
            )
            self.assertEqual(resp.status_code, 200)

            parts = _parse_multipart_response(resp)
            metadata = json.loads(parts["metadata"]["bytes"].decode("utf-8"))
            data = metadata.get("data", {})

            self.assertEqual(data.get("result"), "ACCEPTED")
            self.assertIn("SINGLE_ITEM", data.get("reason_codes", []))
            self.assertIn("VTON_ONLY_PIPELINE", data.get("reason_codes", []))
            self.assertIn("extracted_cloth", parts)
            self.assertGreater(len(parts["extracted_cloth"]["bytes"]), 0)

    def test_multi_garment_type_hint_auto_selects_and_flags_type_forced(self):
        with _AnalyzePatchContext(crop_count=2, labels=["top", "bottom"]):
            resp = self.client.post(
                "/analyze",
                files={"image": ("multi.png", self.file_bytes, "image/png")},
                data={"type": "bottom"},
                headers=self.headers,
            )
            self.assertEqual(resp.status_code, 200)
            parts = _parse_multipart_response(resp)
            metadata = json.loads(parts["metadata"]["bytes"].decode("utf-8"))
            reason_codes = metadata.get("data", {}).get("reason_codes", [])
            self.assertIn("TYPE_FORCED_VTON", reason_codes)

    def test_file_too_large_rejected(self):
        with _AnalyzePatchContext(crop_count=1):
            api_main.ANALYZE_MAX_FILE_BYTES = 32
            resp = self.client.post(
                "/analyze",
                files={"image": ("large.png", b"X" * 64, "image/png")},
                headers=self.headers,
            )
            self.assertEqual(resp.status_code, 400)
            parts = _parse_multipart_response(resp)
            metadata = json.loads(parts["metadata"]["bytes"].decode("utf-8"))
            self.assertIn("FILE_TOO_LARGE", metadata.get("data", {}).get("reason_codes", []))

    def test_same_type_distinct_items_do_not_collapse(self):
        # Regression: same-type detections should still return selection when boxes are distinct.
        bboxes = [
            [8, 8, 86, 58],
            [8, 70, 86, 118],
        ]
        with _AnalyzePatchContext(crop_count=2, labels=["top", "top"], bboxes=bboxes):
            resp = self.client.post(
                "/analyze",
                files={"image": ("multi.png", self.file_bytes, "image/png")},
                headers=self.headers,
            )
            self.assertEqual(resp.status_code, 400)
            parts = _parse_multipart_response(resp)
            metadata = json.loads(parts["metadata"]["bytes"].decode("utf-8"))
            data = metadata.get("data", {})
            self.assertIn("MULTI_ITEM_SELECTION_REQUIRED", data.get("reason_codes", []))
            self.assertEqual(data.get("total_garments_found"), 2)
            self.assertEqual(len(data.get("item_breakdown", [])), 2)

    def test_requested_type_auto_select_prefers_best_geometry_match_not_first_match(self):
        bboxes = [
            [8, 82, 86, 150],
            [8, 12, 86, 88],
        ]
        with _AnalyzePatchContext(crop_count=2, labels=["top", "top"], bboxes=bboxes):
            resp = self.client.post(
                "/analyze",
                files={"image": ("multi.png", self.file_bytes, "image/png")},
                data={"type": "top"},
                headers=self.headers,
            )
            self.assertEqual(resp.status_code, 200)
            parts = _parse_multipart_response(resp)
            metadata = json.loads(parts["metadata"]["bytes"].decode("utf-8"))
            selected_item = metadata.get("data", {}).get("selected_item", {})
            self.assertEqual(selected_item.get("bbox"), bboxes[1])


if __name__ == "__main__":
    unittest.main()
