from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from PIL import Image


@dataclass
class AnalyzeStageTimings:
    read_input_s: float = 0.0
    blur_check_s: float = 0.0
    yolo_detect_s: float = 0.0
    yolo_crop_s: float = 0.0
    caption_total_s: float = 0.0
    primary_type_total_s: float = 0.0
    extract_total_s: float = 0.0
    progress_sync_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
            return
        self.extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "read_input_s": round(float(self.read_input_s), 4),
            "blur_check_s": round(float(self.blur_check_s), 4),
            "yolo_detect_s": round(float(self.yolo_detect_s), 4),
            "yolo_crop_s": round(float(self.yolo_crop_s), 4),
            "caption_total_s": round(float(self.caption_total_s), 4),
            "primary_type_total_s": round(float(self.primary_type_total_s), 4),
            "extract_total_s": round(float(self.extract_total_s), 4),
            "progress_sync_s": round(float(self.progress_sync_s), 4),
        }
        payload.update(self.extra)
        return payload


@dataclass
class AnalyzeInputImage:
    upload_name: str
    image_bytes: bytes
    image: Image.Image
