import logging
import os
import threading
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoProcessor

logger = logging.getLogger("glamify-ai")


class FashionColorClassifierRunner:
    """
    Lazy image classifier wrapper for garment base-colour prediction.

    This is intentionally additive. It should be used as a trial signal
    alongside the current masked palette/color-profile pipeline.
    """

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model_id or os.getenv(
            "FASHION_BASECOLOUR_MODEL_ID",
            "prithivMLmods/Fashion-Product-baseColour",
        ).strip()
        preferred = (device or os.getenv("FASHION_BASECOLOUR_DEVICE", "cpu")).strip().lower()
        if preferred == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.default_top_k = self._env_int("FASHION_BASECOLOUR_TOP_K", 5)

        self._processor = None
        self._model = None
        self._id2label: Dict[int, str] = {}
        self._load_lock = threading.Lock()
        self._available = True

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, str(default))
        try:
            return int(raw)
        except ValueError:
            return default

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    @property
    def is_available(self) -> bool:
        return bool(self._available)

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(self._id2label)

    def ensure_ready(self) -> bool:
        if not self._available:
            return False
        if self.is_loaded:
            return True

        with self._load_lock:
            if self.is_loaded:
                return True
            if not self._available:
                return False
            try:
                logger.info("Loading fashion base-colour model %s on %s...", self.model_id, self.device)
                try:
                    self._processor = AutoProcessor.from_pretrained(self.model_id)
                except Exception:
                    self._processor = AutoImageProcessor.from_pretrained(self.model_id)
                load_kwargs = {}
                if self.device == "cuda":
                    load_kwargs["torch_dtype"] = self.dtype
                try:
                    self._model = AutoModelForImageClassification.from_pretrained(
                        self.model_id,
                        **load_kwargs,
                    )
                except TypeError:
                    self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
                    if self.device == "cuda":
                        self._model = self._model.to(dtype=self.dtype)
                self._model = self._model.to(self.device)
                self._model.eval()
                raw_id2label = getattr(self._model.config, "id2label", {}) or {}
                self._id2label = {int(k): str(v) for k, v in raw_id2label.items()}
            except Exception as err:
                self._available = False
                self._processor = None
                self._model = None
                logger.warning("Fashion base-colour classifier disabled (load failure): %s", err)
                return False
        return True

    def predict_topk(self, image: Image.Image, top_k: Optional[int] = None) -> Dict[str, object]:
        if not self.ensure_ready():
            return {
                "applied": False,
                "reason": "unavailable",
                "model_id": self.model_id,
                "predictions": [],
            }

        assert self._processor is not None
        assert self._model is not None

        resolved_top_k = max(1, int(top_k or self.default_top_k))
        rgb = image.convert("RGB")
        inputs = self._processor(images=rgb, return_tensors="pt")
        for key, val in list(inputs.items()):
            if isinstance(val, torch.Tensor):
                inputs[key] = val.to(self.device)

        with torch.inference_mode():
            logits = self._model(**inputs).logits
            probs = logits.softmax(dim=-1)[0]

        k = min(resolved_top_k, int(probs.shape[-1]))
        scores, indices = torch.topk(probs, k=k, dim=-1)

        predictions: List[Dict[str, object]] = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            label = self._id2label.get(int(index), str(index))
            predictions.append(
                {
                    "label": str(label),
                    "score": round(float(score), 6),
                    "canonical_hint": self._canonical_hint_for_label(str(label)),
                }
            )

        canonical_hints: List[str] = []
        for prediction in predictions:
            candidate = str(prediction.get("canonical_hint") or "").strip().lower()
            if candidate and candidate not in canonical_hints:
                canonical_hints.append(candidate)

        return {
            "applied": True,
            "reason": "ok",
            "model_id": self.model_id,
            "device": self.device,
            "predictions": predictions,
            "top_label": str(predictions[0]["label"]) if predictions else "",
            "top_score": float(predictions[0]["score"]) if predictions else 0.0,
            "canonical_hints": canonical_hints,
        }

    @staticmethod
    def _canonical_hint_for_label(label: str) -> str:
        normalized = " ".join(str(label or "").strip().lower().split())
        mapping = {
            "beige": "beige",
            "black": "black",
            "blue": "blue",
            "bronze": "brown",
            "brown": "brown",
            "burgundy": "maroon",
            "charcoal": "charcoal",
            "coffee brown": "brown",
            "copper": "brown",
            "cream": "cream",
            "fluorescent green": "green",
            "gold": "gold",
            "green": "green",
            "grey": "gray",
            "grey melange": "gray",
            "khaki": "khaki",
            "lavender": "lavender",
            "lime green": "green",
            "magenta": "pink",
            "maroon": "maroon",
            "mauve": "pink",
            "metallic": "silver",
            "multi": "multi",
            "mushroom brown": "brown",
            "mustard": "yellow",
            "navy blue": "navy",
            "nude": "nude",
            "off white": "off-white",
            "olive": "olive",
            "orange": "orange",
            "peach": "peach",
            "pink": "pink",
            "purple": "purple",
            "red": "red",
            "rose": "pink",
            "rust": "orange",
            "sea green": "green",
            "silver": "silver",
            "skin": "nude",
            "steel": "gray",
            "tan": "tan",
            "taupe": "beige",
            "teal": "teal",
            "turquoise blue": "blue",
            "white": "white",
            "yellow": "yellow",
        }
        return mapping.get(normalized, normalized)
