import logging
import sys
import threading
import types
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("glamify-ai")


def _ensure_torchvision_functional_tensor_compat() -> None:
    """BasicSR 1.4.x imports a removed torchvision module on newer versions."""
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return
    try:
        from torchvision.transforms import functional as tv_functional
    except Exception:
        return
    compat = types.ModuleType(module_name)
    compat.rgb_to_grayscale = tv_functional.rgb_to_grayscale
    sys.modules[module_name] = compat


class GFPGANRunner:
    """Optional GFPGAN runner for face enhancement."""

    MODEL_URLS = {
        "GFPGANv1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    }

    def __init__(
        self,
        *,
        model_name: str = "GFPGANv1.4",
        model_path: Optional[str] = None,
        face_weight: float = 0.7,
        device: Optional[str] = None,
    ):
        self.model_name = str(model_name or "GFPGANv1.4").strip() or "GFPGANv1.4"
        self.model_path = str(model_path or "").strip()
        self.face_weight = float(face_weight)
        if device is None or str(device).strip().lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            desired = str(device).strip().lower()
            self.device = "cuda" if desired == "cuda" and torch.cuda.is_available() else "cpu"
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._restorer = None

    @property
    def is_loaded(self) -> bool:
        return self._restorer is not None

    @property
    def is_available(self) -> bool:
        try:
            from gfpgan.utils import GFPGANer  # noqa: F401
        except Exception:
            return False
        return True

    def _build_restorer(self):
        _ensure_torchvision_functional_tensor_compat()
        try:
            from gfpgan.utils import GFPGANer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "GFPGAN dependencies are not installed. Install `gfpgan` and `basicsr` to enable face enhancement."
            ) from exc

        if self.model_name not in self.MODEL_URLS:
            raise ValueError(f"Unsupported GFPGAN model: {self.model_name}")

        model_path = self.model_path or self.MODEL_URLS[self.model_name]
        return GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device,
        )

    def ensure_ready(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            logger.info("Loading %s for face enhancement on %s", self.model_name, self.device)
            self._restorer = self._build_restorer()

    def enhance(self, image: Image.Image, *, weight: Optional[float] = None) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width < 1 or height < 1:
            return rgb

        self.ensure_ready()
        assert self._restorer is not None

        face_weight = self.face_weight if weight is None else float(weight)
        face_weight = max(0.0, min(1.0, face_weight))

        bgr = np.array(rgb, dtype=np.uint8)[:, :, ::-1]
        with self._infer_lock, torch.inference_mode():
            _, _, restored = self._restorer.enhance(
                bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=face_weight,
            )
        if restored is None:
            return rgb
        return Image.fromarray(restored[:, :, ::-1], mode="RGB")
