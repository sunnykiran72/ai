import logging
import os
import threading
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

logger = logging.getLogger("glamify-ai")


class FashionDetectionRunner:
    """
    Lazy fashion-specific detector wrapper.

    The model is used as an alternative to the legacy YOLO garment proposer so
    both backends can be compared on the same inputs.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or os.getenv("FASHION_DETECTION_MODEL_PATH", "yainage90/fashion-object-detection")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = self._env_float("FASHION_DETECTION_SCORE_THRESHOLD", 0.30)
        self._processor = None
        self._model = None
        self._id2label: Dict[int, str] = {}
        self._load_lock = threading.Lock()

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name, str(default))
        try:
            return float(raw)
        except ValueError:
            return default

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(self._id2label)

    def ensure_ready(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return

            logger.info("Loading fashion detector from %s on %s...", self.model_path, self.device)
            self._processor = AutoImageProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForObjectDetection.from_pretrained(self.model_path).to(self.device)
            self._model.eval()
            raw_id2label = getattr(self._model.config, "id2label", {}) or {}
            self._id2label = {int(k): str(v) for k, v in raw_id2label.items()}

    def predict(self, image: Image.Image, threshold: Optional[float] = None) -> List[Dict[str, object]]:
        self.ensure_ready()
        assert self._processor is not None
        assert self._model is not None

        resolved_threshold = self.score_threshold if threshold is None else float(threshold)
        rgb = image.convert("RGB")
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        processed = self._processor.post_process_object_detection(
            outputs,
            threshold=resolved_threshold,
            target_sizes=[(rgb.height, rgb.width)],
        )
        if not processed:
            return []

        result = processed[0]
        detections: List[Dict[str, object]] = []
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        for box, score, label_idx in zip(boxes, scores, labels):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box.tolist()]
            if x1 <= x0 or y1 <= y0:
                continue
            class_id = int(label_idx.item() if hasattr(label_idx, "item") else label_idx)
            label = self._id2label.get(class_id, str(class_id))
            detections.append({
                "bbox": [x0, y0, x1, y1],
                "score": float(score.item() if hasattr(score, "item") else score),
                "label": str(label),
                "class_id": class_id,
                "source": "fashion_object_detection",
            })

        return detections
