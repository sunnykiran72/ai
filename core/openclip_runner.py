import logging
import os
import threading
from typing import Dict, List

import torch
from PIL import Image

logger = logging.getLogger("glamify-ai")


class OpenCLIPRunner:
    """
    Lightweight CLIP scorer used for parser-candidate type validation.
    It is lazy-loaded and optional; callers should handle unavailable state.
    """

    def __init__(self):
        self.model_id = os.getenv("OPENCLIP_MODEL_ID", "openai/clip-vit-base-patch32").strip()
        preferred = os.getenv("OPENCLIP_DEVICE", "cpu").strip().lower()
        if preferred == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._available = True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def is_available(self) -> bool:
        return bool(self._available)

    def ensure_ready(self) -> bool:
        if not self._available:
            return False
        if self._model is not None and self._processor is not None:
            return True

        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return True
            if not self._available:
                return False
            try:
                from transformers import CLIPModel, CLIPProcessor

                logger.info(f"Loading OpenCLIP model {self.model_id} on {self.device}...")
                self._processor = CLIPProcessor.from_pretrained(self.model_id)
                self._model = CLIPModel.from_pretrained(self.model_id, torch_dtype=self.dtype)
                self._model.to(self.device)
                self._model.eval()
            except Exception as err:
                self._available = False
                self._model = None
                self._processor = None
                logger.warning(f"OpenCLIP disabled (load failure): {err}")
                return False
        return True

    def score_labels(self, image: Image.Image, labels: List[str]) -> Dict[str, float]:
        if not labels:
            return {}
        if not self.ensure_ready():
            # Graceful fallback: uniform scores when model is unavailable.
            n = float(max(1, len(labels)))
            return {str(label): (1.0 / n) for label in labels}

        img = image.convert("RGB")
        inputs = self._processor(
            text=[str(l) for l in labels],
            images=img,
            return_tensors="pt",
            padding=True,
        )
        for key, val in list(inputs.items()):
            if isinstance(val, torch.Tensor):
                inputs[key] = val.to(self.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=-1).detach().float().cpu().numpy()[0].tolist()

        return {str(label): float(prob) for label, prob in zip(labels, probs)}
