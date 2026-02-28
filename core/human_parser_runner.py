import logging
import os
import threading
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

logger = logging.getLogger("glamify-ai")


class HumanParserRunner:
    """
    Stateful SegFormer runner that owns model lifecycle and GPU memory.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or os.getenv("PARSING_MODEL_PATH", "mattmdjaga/segformer_b2_clothes")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
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
                logger.info(f"Loading Human Parser from {self.model_path} on {self.device}...")
                self._processor = SegformerImageProcessor.from_pretrained(self.model_path)
                self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_path)
                self._model.to(self.device)

    def parse(self, image: Image.Image) -> np.ndarray:
        self.ensure_ready()

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        return upsampled_logits.argmax(dim=1)[0].numpy()

