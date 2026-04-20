import logging
import os
import threading
from typing import Dict, List, Sequence

import torch
from PIL import Image

logger = logging.getLogger("glamify-ai")


class MarqoFashionSiglipRunner:
    """
    Lightweight zero-shot fashion classifier.
    Uses label strings for matching and leaves key mapping to caller.
    """

    def __init__(self, *, model_id: str | None = None, device: str | None = None):
        self.model_id = str(
            model_id
            or os.getenv("ANALYZE_MARQO_MODEL_ID", "Marqo/marqo-fashionSigLIP")
        ).strip()
        preferred = str(device or os.getenv("ANALYZE_MARQO_DEVICE", "auto")).strip().lower()
        if preferred == "auto":
            preferred = "cuda" if torch.cuda.is_available() else "cpu"
        if preferred == "cuda" and not torch.cuda.is_available():
            preferred = "cpu"
        if preferred not in {"cpu", "cuda"}:
            preferred = "cpu"
        self.device = preferred
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._processor = None
        self._image_preprocess = None
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._available = True

    @property
    def is_loaded(self) -> bool:
        return (
            self._model is not None
            and self._image_preprocess is not None
            and self._tokenizer is not None
        )

    @property
    def is_available(self) -> bool:
        return bool(self._available)

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
                from open_clip import create_model_and_transforms, get_tokenizer

                logger.info("Loading Marqo fashionSigLIP from %s on %s", self.model_id, self.device)
                model_id = self.model_id
                if not model_id.startswith("hf-hub:"):
                    model_id = f"hf-hub:{model_id}"
                model, _, image_preprocess = create_model_and_transforms(
                    model_id,
                    precision="fp16" if self.device == "cuda" else "fp32",
                )
                model = model.to(self.device)
                model.eval()
                self._model = model
                self._image_preprocess = image_preprocess
                self._tokenizer = get_tokenizer(model_id)
            except Exception as err:
                self._available = False
                self._image_preprocess = None
                self._tokenizer = None
                self._model = None
                logger.warning("Marqo fashionSigLIP disabled (load failure): %s", err)
                return False
        return True

    def score_labels(self, image: Image.Image, labels: Sequence[str]) -> Dict[str, float]:
        normalized = [str(label or "").strip() for label in labels if str(label or "").strip()]
        if not normalized:
            return {}
        if not self.ensure_ready():
            uniform = 1.0 / float(len(normalized))
            return {label: uniform for label in normalized}

        pil_image = image.convert("RGB")
        with self._infer_lock, torch.inference_mode():
            image_tensor = self._image_preprocess(pil_image).unsqueeze(0).to(
                device=self.device,
                dtype=self.dtype,
            )
            text_tensor = self._tokenizer(normalized).to(self.device)
            image_features = self._model.encode_image(image_tensor, normalize=True)
            text_features = self._model.encode_text(text_tensor, normalize=True)

            logits = 100.0 * (image_features @ text_features.T)
            probs = logits.softmax(dim=-1).detach().float().cpu().squeeze(0).tolist()

        return {label: float(score) for label, score in zip(normalized, probs)}

    def rank_labels(self, image: Image.Image, labels: Sequence[str], top_k: int = 5) -> List[Dict[str, float]]:
        scored = self.score_labels(image=image, labels=labels)
        ranked = sorted(
            (
                {"label": label, "score": float(score)}
                for label, score in scored.items()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        if top_k > 0:
            return ranked[: int(top_k)]
        return ranked
