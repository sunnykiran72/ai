import logging
import sys
import types
import threading
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("glamify-ai")
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


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


class RealESRGANRunner:
    """Optional RealESRGAN runner for user-image/prepare upscaling."""

    MODEL_URLS = {
        "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }

    def __init__(
        self,
        *,
        model_name: str = "RealESRGAN_x2plus",
        model_path: Optional[str] = None,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        device: Optional[str] = None,
    ):
        self.model_name = str(model_name or "RealESRGAN_x2plus").strip() or "RealESRGAN_x2plus"
        self.model_path = str(model_path or "").strip()
        self.tile = max(0, int(tile))
        self.tile_pad = max(0, int(tile_pad))
        self.pre_pad = max(0, int(pre_pad))
        if device is None or str(device).strip().lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            desired = str(device).strip().lower()
            self.device = "cuda" if desired == "cuda" and torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._upsampler = None

    @property
    def is_loaded(self) -> bool:
        return self._upsampler is not None

    @property
    def is_available(self) -> bool:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
            from realesrgan import RealESRGANer  # noqa: F401
        except Exception:
            return False
        return True

    def _build_upsampler(self):
        _ensure_torchvision_functional_tensor_compat()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except Exception as exc:  # pragma: no cover - exercised on environments without deps
            raise RuntimeError(
                "RealESRGAN dependencies are not installed. Install `realesrgan` and `basicsr` to enable user-image upscaling."
            ) from exc

        if self.model_name not in self.MODEL_URLS:
            raise ValueError(f"Unsupported RealESRGAN model: {self.model_name}")

        scale = 2 if self.model_name == "RealESRGAN_x2plus" else 4
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        model_path = self.model_path or self.MODEL_URLS[self.model_name]
        return RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=self.half,
            device=torch.device(self.device),
        )

    def ensure_ready(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            logger.info("Loading %s for user-image/prepare on %s", self.model_name, self.device)
            self._upsampler = self._build_upsampler()

    def upscale(self, image: Image.Image, *, target_max_edge: Optional[int] = None) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width < 1 or height < 1:
            return rgb

        self.ensure_ready()
        assert self._upsampler is not None

        bgr = np.array(rgb, dtype=np.uint8)[:, :, ::-1]
        with self._infer_lock, torch.inference_mode():
            output, _ = self._upsampler.enhance(bgr, outscale=2)
        upscaled = Image.fromarray(output[:, :, ::-1], mode="RGB")

        if target_max_edge is None or max(upscaled.size) <= int(target_max_edge):
            return upscaled

        scale = float(target_max_edge) / float(max(upscaled.size))
        resized = (
            max(1, int(round(upscaled.width * scale))),
            max(1, int(round(upscaled.height * scale))),
        )
        return upscaled.resize(resized, _LANCZOS)
