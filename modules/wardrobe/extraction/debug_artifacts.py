from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set

from PIL import Image

logger = logging.getLogger("glamify-ai")


def parse_capture_types(raw_value: object) -> Set[str]:
    return {
        part.strip().lower()
        for part in str(raw_value or "").split(",")
        if part.strip()
    }


def should_capture_debug_artifacts(
    *,
    enabled: bool,
    selected_type: object,
    allowed_types: object,
) -> bool:
    if not enabled:
        return False
    normalized_type = str(selected_type or "").strip().lower()
    if not normalized_type:
        return False
    return normalized_type in parse_capture_types(allowed_types)


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "sample"


def _write_image(path: Path, image: Optional[Image.Image]) -> Optional[Path]:
    if not isinstance(image, Image.Image):
        return None
    output = image.convert("RGB") if image.mode not in {"RGB", "L"} else image
    output.save(path, format="PNG")
    return path


def _write_bytes(path: Path, payload: bytes) -> Optional[Path]:
    if not payload:
        return None
    path.write_bytes(payload)
    return path


def save_debug_artifacts(
    *,
    capture_dir: object,
    upload_name: object,
    selected_item: Optional[Dict[str, object]],
    requested_type: object,
) -> Optional[Path]:
    item = selected_item or {}
    selected_type = str(item.get("type") or "").strip().lower()
    base_dir = Path(str(capture_dir or "").strip() or "asymmetric_dataset/debug_capture").expanduser()
    upload_slug = _safe_slug(Path(str(upload_name or "")).stem or "sample")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sample_dir = base_dir / selected_type / f"{upload_slug}__{timestamp}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    crop_path = _write_image(sample_dir / "selected_crop.png", item.get("_extract_source_image"))  # type: ignore[arg-type]
    if crop_path is None:
        crop_path = _write_image(sample_dir / "selected_crop.png", item.get("_image_obj"))  # type: ignore[arg-type]
    flux_path = _write_bytes(sample_dir / "flux_output.png", bytes(item.get("_extracted_image_bytes") or b""))

    prompt_description = " ".join(str(item.get("promptDescription") or "").split()).strip()
    base_prompt = " ".join(str(item.get("baseGarmentPrompt") or "").split()).strip()
    prompt_source = str(item.get("promptDescriptionSource") or "").strip()
    extracted_url = str(item.get("output_image_url") or item.get("url") or "").strip()

    metadata = {
        "upload_name": str(upload_name or ""),
        "requested_type": str(requested_type or ""),
        "selected_type": selected_type,
        "type_source": str(item.get("type_source") or ""),
        "detection_source": str(item.get("detection_source") or ""),
        "bbox": list(item.get("bbox") or []),
        "extract_crop_bbox": list(item.get("extract_crop_bbox") or []),
        "extract_crop_mode": str(item.get("extract_crop_mode") or ""),
        "output_image_url": extracted_url,
        "prompt_description": prompt_description,
        "base_garment_prompt": base_prompt,
        "prompt_description_source": prompt_source,
        "saved_files": {
            "selected_crop": crop_path.name if crop_path else "",
            "flux_output": flux_path.name if flux_path else "",
            "details": "details.md",
        },
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    details_md = "\n".join(
        [
            "# Analyze Debug Sample",
            "",
            f"- Upload: `{metadata['upload_name']}`",
            f"- Requested type: `{metadata['requested_type'] or 'auto'}`",
            f"- Selected type: `{metadata['selected_type']}`",
            f"- Type source: `{metadata['type_source'] or 'unknown'}`",
            f"- Detection source: `{metadata['detection_source'] or 'unknown'}`",
            f"- BBox: `{metadata['bbox']}`",
            f"- Extract crop bbox: `{metadata['extract_crop_bbox']}`",
            f"- Extract crop mode: `{metadata['extract_crop_mode'] or 'unknown'}`",
            f"- Output image URL: `{metadata['output_image_url'] or 'n/a'}`",
            "",
            "## Prompt Description",
            "",
            prompt_description or "_empty_",
            "",
            "## Base Garment Prompt",
            "",
            base_prompt or "_empty_",
            "",
            "## Prompt Source",
            "",
            prompt_source or "_empty_",
            "",
            "## Saved Files",
            "",
            f"- Crop: `{metadata['saved_files']['selected_crop'] or 'not_saved'}`",
            f"- Flux output: `{metadata['saved_files']['flux_output'] or 'not_saved'}`",
            "",
        ]
    )
    (sample_dir / "details.md").write_text(details_md, encoding="utf-8")
    return sample_dir


def maybe_save_debug_artifacts(
    *,
    enabled: bool,
    capture_dir: object,
    allowed_types: object,
    upload_name: object,
    selected_item: Optional[Dict[str, object]],
    requested_type: object,
) -> Optional[Path]:
    if not should_capture_debug_artifacts(
        enabled=bool(enabled),
        selected_type=(selected_item or {}).get("type"),
        allowed_types=allowed_types,
    ):
        return None
    try:
        return save_debug_artifacts(
            capture_dir=capture_dir,
            upload_name=upload_name,
            selected_item=selected_item,
            requested_type=requested_type,
        )
    except Exception as exc:  # pragma: no cover - non-fatal debug path
        logger.warning("Failed to save analyze debug artifacts: %s", exc)
        return None
