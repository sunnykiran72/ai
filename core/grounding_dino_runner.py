import logging
import os
import threading
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger("glamify-ai")


class GroundingDinoRunner:
    """
    Text-guided fallback detector used when the primary fashion detector is weak.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or os.getenv("GROUNDING_DINO_MODEL_PATH", "IDEA-Research/grounding-dino-base")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = self._env_float("GROUNDING_DINO_BOX_THRESHOLD", 0.35)
        self.text_threshold = self._env_float("GROUNDING_DINO_TEXT_THRESHOLD", 0.25)
        self._processor = None
        self._model = None
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
        return self._model is not None and self._processor is not None

    def ensure_ready(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            logger.info("Loading Grounding DINO from %s on %s...", self.model_path, self.device)
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path).to(self.device)
            self._model.eval()

    def detect(
        self,
        image: Image.Image,
        *,
        prompts: List[str],
        threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        self.ensure_ready()
        assert self._processor is not None
        assert self._model is not None

        clean_prompts = [str(p).strip() for p in prompts if str(p).strip()]
        if not clean_prompts:
            return []

        rgb = image.convert("RGB")
        inputs = self._processor(images=rgb, text=[clean_prompts], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.threshold if threshold is None else float(threshold),
            text_threshold=self.text_threshold if text_threshold is None else float(text_threshold),
            target_sizes=[rgb.size[::-1]],
        )
        if not results:
            return []

        processed = results[0]
        detections: List[Dict[str, object]] = []
        for box, score, labels in zip(processed.get("boxes", []), processed.get("scores", []), processed.get("labels", [])):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box.tolist()]
            if x1 <= x0 or y1 <= y0:
                continue
            label_text = labels[0] if isinstance(labels, list) and labels else labels
            detections.append({
                "bbox": [x0, y0, x1, y1],
                "score": float(score.item() if hasattr(score, "item") else score),
                "label": str(label_text),
                "source": "grounding_dino",
            })

        return detections
