from __future__ import annotations

from pathlib import Path

from PIL import Image

from config.analyze_config import AnalyzeConfig
from modules.wardrobe.extraction.debug_artifacts import (
    maybe_save_debug_artifacts,
    parse_capture_types,
    should_capture_debug_artifacts,
)


def test_parse_capture_types_normalizes_values():
    assert parse_capture_types(" top, dress ,TOP ,, ") == {"top", "dress"}


def test_should_capture_debug_artifacts_filters_to_top_and_dress():
    assert should_capture_debug_artifacts(enabled=True, selected_type="top", allowed_types="top,dress") is True
    assert should_capture_debug_artifacts(enabled=True, selected_type="dress", allowed_types="top,dress") is True
    assert should_capture_debug_artifacts(enabled=True, selected_type="bottom", allowed_types="top,dress") is False
    assert should_capture_debug_artifacts(enabled=False, selected_type="top", allowed_types="top,dress") is False


def test_maybe_save_debug_artifacts_writes_expected_files(tmp_path: Path):
    selected_item = {
        "type": "top",
        "type_source": "requested_type",
        "detection_source": "human_parser",
        "bbox": [1, 2, 30, 40],
        "extract_crop_bbox": [0, 0, 32, 48],
        "extract_crop_mode": "full_image_direct",
        "promptDescription": "asymmetric one-shoulder top.",
        "baseGarmentPrompt": "A single top garment displayed alone with one shoulder exposed.",
        "promptDescriptionSource": "minicpm_direct_prompt",
        "_extract_source_image": Image.new("RGB", (16, 24), "white"),
        "_extracted_image_bytes": b"\x89PNG\r\n\x1a\n",
    }

    saved_dir = maybe_save_debug_artifacts(
        enabled=True,
        capture_dir=tmp_path,
        allowed_types="top,dress",
        upload_name="sample-image.jpg",
        selected_item=selected_item,
        requested_type="top",
    )

    assert saved_dir is not None
    assert (saved_dir / "selected_crop.png").exists()
    assert (saved_dir / "flux_output.png").exists()
    assert (saved_dir / "details.md").exists()
    assert (saved_dir / "metadata.json").exists()
    assert "Base Garment Prompt" in (saved_dir / "details.md").read_text(encoding="utf-8")


def test_maybe_save_debug_artifacts_skips_bottom(tmp_path: Path):
    saved_dir = maybe_save_debug_artifacts(
        enabled=True,
        capture_dir=tmp_path,
        allowed_types="top,dress",
        upload_name="bottom.jpg",
        selected_item={"type": "bottom"},
        requested_type="bottom",
    )
    assert saved_dir is None


def test_analyze_config_reads_debug_artifact_settings(monkeypatch):
    monkeypatch.setenv("ANALYZE_DEBUG_ARTIFACT_CAPTURE_ENABLED", "1")
    monkeypatch.setenv("ANALYZE_DEBUG_ARTIFACT_CAPTURE_DIR", "/tmp/analyze-debug")
    monkeypatch.setenv("ANALYZE_DEBUG_ARTIFACT_CAPTURE_TYPES", "top,dress")

    config = AnalyzeConfig.from_env()

    assert config.debug_artifact_capture_enabled is True
    assert config.debug_artifact_capture_dir == "/tmp/analyze-debug"
    assert config.debug_artifact_capture_types == "top,dress"
