import logging
import os
import threading
from typing import Any, Optional

import torch
from ultralytics import YOLO

logger = logging.getLogger("glamify-ai")


class YoloRunner:
    """
    Stateful YOLO-Seg runner that owns model lifecycle and GPU memory.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or os.getenv("YOLO_MODEL_PATH", "yolov8x-seg.pt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_ready(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is None:
                logger.info(f"Loading YOLO-Seg model from {self.model_path} on {self.device}...")
                model = YOLO(self.model_path, task="segment")
                model.to(self.device)
                self._model = model

    def predict(self, image_rgb: Any, conf: float, iou: float) -> Any:
        self.ensure_ready()
        return self._model(image_rgb, verbose=False, conf=conf, iou=iou)

