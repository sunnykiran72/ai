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
        self.model_path = model_path or os.getenv("PARSING_MODEL_PATH", "fashn-ai/fashn-human-parser")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self._load_lock = threading.Lock()
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_ready(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is None:
                logger.info(f"Loading Human Parser from {self.model_path} on {self.device}...")
                try:
                    self._processor = SegformerImageProcessor.from_pretrained(self.model_path, token=False)
                except TypeError:
                    self._processor = SegformerImageProcessor.from_pretrained(self.model_path, use_auth_token=False)
                try:
                    self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_path, token=False)
                except TypeError:
                    self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_path, use_auth_token=False)
                self._model.to(self.device)
                cfg = getattr(self._model, "config", None)
                id2label = getattr(cfg, "id2label", {}) if cfg is not None else {}
                label2id = getattr(cfg, "label2id", {}) if cfg is not None else {}
                self.id2label = {}
                self.label2id = {}
                if isinstance(id2label, dict):
                    for k, v in id2label.items():
                        try:
                            idx = int(k)
                        except Exception:
                            continue
                        self.id2label[idx] = str(v)
                if isinstance(label2id, dict):
                    for k, v in label2id.items():
                        try:
                            self.label2id[str(k).strip().lower().replace("-", "_").replace(" ", "_")] = int(v)
                        except Exception:
                            continue
                if not self.label2id and self.id2label:
                    for idx, name in self.id2label.items():
                        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
                        self.label2id[key] = int(idx)
                if self.label2id:
                    logger.info(
                        "Human parser labels loaded (%d classes): %s",
                        len(self.label2id),
                        ",".join(sorted(self.label2id.keys())[:12]),
                    )

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
