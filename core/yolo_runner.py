import logging
import os
import threading
from typing import Any, Dict, Iterable, Optional

import torch
from ultralytics import YOLO

logger = logging.getLogger("glamify-ai")

_DEEPFASHION2_LABELS = {
    "short sleeve top",
    "long sleeve top",
    "short sleeve outwear",
    "long sleeve outwear",
    "short sleeved shirt",
    "long sleeved shirt",
    "short sleeved outwear",
    "long sleeved outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short sleeve dress",
    "long sleeve dress",
    "short sleeved dress",
    "long sleeved dress",
    "vest dress",
    "sling dress",
}


def _normalize_label_name(label: str) -> str:
    text = str(label or "").strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    return text


def _detect_label_family(names: Iterable[str]) -> str:
    normalized = {_normalize_label_name(name) for name in names if str(name or "").strip()}
    if not normalized:
        return "unknown"

    deepfashion_hits = len(normalized & _DEEPFASHION2_LABELS)
    if deepfashion_hits >= 6:
        return "deepfashion2"

    coco_markers = {"person", "car", "dog", "traffic light", "toothbrush", "airplane"}
    coco_hits = len(normalized & coco_markers)
    if coco_hits >= 2 and "person" in normalized:
        return "coco"

    if deepfashion_hits >= 2:
        return "deepfashion2_like"
    return "unknown"


class YoloRunner:
    """
    Stateful YOLO-Seg runner that owns model lifecycle and GPU memory.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or os.getenv("YOLO_MODEL_PATH", "yolov8x-seg.pt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.expected_label_family = os.getenv("YOLO_EXPECTED_LABEL_FAMILY", "deepfashion2").strip().lower() or "deepfashion2"
        self.strict_label_family = os.getenv("YOLO_STRICT_LABEL_FAMILY", "1") == "1"
        self._model = None
        self._class_names: Dict[int, str] = {}
        self._class_count: int = 0
        self._label_family: str = "unknown"
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_count(self) -> int:
        return int(self._class_count)

    @property
    def label_family(self) -> str:
        return str(self._label_family)

    def ensure_ready(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is None:
                logger.info(f"Loading YOLO-Seg model from {self.model_path} on {self.device}...")
                model = YOLO(self.model_path, task="segment")
                model.to(self.device)
                self._model = model
                names_obj = getattr(getattr(model, "model", None), "names", None)
                if isinstance(names_obj, dict):
                    self._class_names = {int(k): str(v) for k, v in names_obj.items()}
                elif isinstance(names_obj, (list, tuple)):
                    self._class_names = {idx: str(v) for idx, v in enumerate(names_obj)}
                else:
                    self._class_names = {}
                self._class_count = len(self._class_names)
                self._label_family = _detect_label_family(self._class_names.values())
                logger.info(
                    "YOLO metadata: classes=%s, label_family=%s, expected=%s",
                    self._class_count,
                    self._label_family,
                    self.expected_label_family,
                )
                if (
                    self.expected_label_family not in {"", "any", "auto"}
                    and self._label_family != self.expected_label_family
                ):
                    msg = (
                        "YOLO label family mismatch: "
                        f"expected={self.expected_label_family}, "
                        f"actual={self._label_family}, model_path={self.model_path}"
                    )
                    if self.strict_label_family:
                        raise RuntimeError(msg)
                    logger.warning(msg)

    def predict(self, image_rgb: Any, conf: float, iou: float) -> Any:
        self.ensure_ready()
        return self._model(image_rgb, verbose=False, conf=conf, iou=iou)


class YoloPersonDetectorRunner:
    """
    Dedicated YOLO detect runner for person detection on user-uploaded photos.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or os.getenv("YOLO_PERSON_DETECT_MODEL_PATH", "yolov8m.pt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._class_names: Dict[int, str] = {}
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_ready(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is None:
                logger.info(f"Loading YOLO detect model from {self.model_path} on {self.device}...")
                model = YOLO(self.model_path, task="detect")
                model.to(self.device)
                self._model = model
                names_obj = getattr(getattr(model, "model", None), "names", None)
                if isinstance(names_obj, dict):
                    self._class_names = {int(k): str(v) for k, v in names_obj.items()}
                elif isinstance(names_obj, (list, tuple)):
                    self._class_names = {idx: str(v) for idx, v in enumerate(names_obj)}
                else:
                    self._class_names = {}

    def predict(self, image_rgb: Any, conf: float, iou: float) -> Any:
        self.ensure_ready()
        return self._model(image_rgb, verbose=False, conf=conf, iou=iou)
