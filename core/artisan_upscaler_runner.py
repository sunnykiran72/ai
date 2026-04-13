import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("glamify-ai")
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


class ArtisanUpscalerRunner:
    """Lazy Artisan upscaler runner for benchmark/dev API usage."""

    def __init__(
        self,
        *,
        repo_id: str = "ArtisanLabs/artisan-upscaler",
        weights_filename: str = "weights/artisan_upscaler_bf16.safetensors",
        weights_path: str = "",
        device: Optional[str] = None,
    ):
        self.repo_id = str(repo_id or "ArtisanLabs/artisan-upscaler").strip()
        self.weights_filename = str(weights_filename or "weights/artisan_upscaler_bf16.safetensors").strip()
        self.weights_path_override = str(weights_path or "").strip()
        if device is None or str(device).strip().lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            desired = str(device).strip().lower()
            self.device = "cuda" if desired == "cuda" and torch.cuda.is_available() else "cpu"

        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._repo_dir: Optional[Path] = None
        self._weights_path: Optional[Path] = None
        self._DAT = None
        self._DATConfig = None
        self._model = None
        self._precision = ""
        self._dtype = torch.float32
        self._compiled = False
        self._last_load_seconds = 0.0
        self._last_compile_seconds = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_load_seconds(self) -> float:
        return float(self._last_load_seconds)

    @property
    def last_compile_seconds(self) -> float:
        return float(self._last_compile_seconds)

    def _load_module_from_path(self, module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _ensure_repo_assets(self) -> Tuple[Path, Path]:
        from huggingface_hub import hf_hub_download, snapshot_download

        if self._repo_dir is None:
            repo_path = snapshot_download(
                repo_id=self.repo_id,
                allow_patterns=["DAT.py", "config.py", "vectorized_ops.py"],
            )
            self._repo_dir = Path(repo_path)

        if self._weights_path is None:
            if self.weights_path_override:
                candidate = Path(self.weights_path_override).expanduser().resolve()
                if not candidate.exists():
                    raise FileNotFoundError(f"Artisan weights not found: {candidate}")
                self._weights_path = candidate
            else:
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.weights_filename,
                )
                self._weights_path = Path(path)

        assert self._repo_dir is not None
        assert self._weights_path is not None
        return self._repo_dir, self._weights_path

    def _ensure_artisan_modules(self) -> None:
        if self._DAT is not None and self._DATConfig is not None:
            return

        repo_dir, _ = self._ensure_repo_assets()
        config_path = repo_dir / "config.py"
        dat_path = repo_dir / "DAT.py"
        vec_ops_path = repo_dir / "vectorized_ops.py"
        if not config_path.exists() or not dat_path.exists() or not vec_ops_path.exists():
            raise FileNotFoundError(f"Missing required Artisan files in {repo_dir}")

        token = f"{int(time.time() * 1000)}_{id(self)}"
        vectorized_mod = self._load_module_from_path(
            f"_artisan_vectorized_ops_{token}",
            vec_ops_path,
        )
        previous_vectorized = sys.modules.get("vectorized_ops")
        sys.modules["vectorized_ops"] = vectorized_mod
        try:
            config_mod = self._load_module_from_path(
                f"_artisan_config_{token}",
                config_path,
            )
            dat_mod = self._load_module_from_path(
                f"_artisan_dat_{token}",
                dat_path,
            )
        finally:
            if previous_vectorized is None:
                sys.modules.pop("vectorized_ops", None)
            else:
                sys.modules["vectorized_ops"] = previous_vectorized

        self._DATConfig = getattr(config_mod, "DATConfig", None)
        self._DAT = getattr(dat_mod, "DAT", None)
        if self._DATConfig is None or self._DAT is None:
            raise RuntimeError("Failed to load DATConfig/DAT from Artisan repo assets.")

    @staticmethod
    def _build_model(dat_cls, dat_cfg_cls):
        cfg = dat_cfg_cls(
            attn_type="natten",
            multiscale=True,
            hr_refine_blocks=2,
            use_chk=False,
        )
        return dat_cls(
            img_size=cfg.img_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            split_size=cfg.split_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            expansion_factor=cfg.expansion_factor,
            qkv_bias=cfg.qkv_bias,
            drop_path_rate=cfg.drop_path_rate,
            upscale=cfg.upscale,
            img_range=cfg.img_range,
            upsampler=cfg.upsampler,
            resi_connection=cfg.resi_connection,
            attn_type=cfg.attn_type,
            natten_kernel=cfg.natten_kernel,
            natten_dilation=cfg.natten_dilation,
            use_chk=cfg.use_chk,
            multiscale=cfg.multiscale,
            ms_enc_groups=cfg.ms_enc_groups,
            ms_dec_groups=cfg.ms_dec_groups,
            hr_refine_blocks=cfg.hr_refine_blocks,
        )

    def ensure_ready(self, *, precision: str = "bf16", compile_model: bool = False) -> Dict[str, float]:
        precision_value = str(precision or "bf16").strip().lower()
        if precision_value not in {"bf16", "fp32"}:
            raise ValueError("precision must be one of: bf16, fp32")

        with self._load_lock:
            self._ensure_artisan_modules()
            _, weights_path = self._ensure_repo_assets()
            assert self._DAT is not None
            assert self._DATConfig is not None

            needs_reload = (self._model is None) or (self._precision != precision_value)
            if needs_reload:
                from safetensors.torch import load_file

                model = self._build_model(self._DAT, self._DATConfig)
                state_dict = load_file(str(weights_path), device="cpu")
                model.load_state_dict(state_dict, strict=False)

                dtype = torch.float32 if precision_value == "fp32" else torch.bfloat16
                t0 = time.perf_counter()
                model = model.to(
                    device=torch.device(self.device),
                    dtype=dtype,
                    memory_format=torch.channels_last,
                ).eval()
                self._last_load_seconds = time.perf_counter() - t0
                self._model = model
                self._precision = precision_value
                self._dtype = dtype
                self._compiled = False
                self._last_compile_seconds = 0.0
                logger.info(
                    "Loaded Artisan upscaler (%s) on %s in %.3fs",
                    precision_value,
                    self.device,
                    self._last_load_seconds,
                )

            if compile_model and not self._compiled:
                t0 = time.perf_counter()
                self._model = torch.compile(self._model, mode="max-autotune-no-cudagraphs")
                self._compiled = True
                self._last_compile_seconds = time.perf_counter() - t0
                logger.info("Compiled Artisan upscaler in %.3fs", self._last_compile_seconds)

            return {
                "load_seconds": float(self._last_load_seconds),
                "compile_seconds": float(self._last_compile_seconds),
                "compiled": 1.0 if self._compiled else 0.0,
            }

    @staticmethod
    def _tile_forward(model: torch.nn.Module, lr: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
        scale = int(getattr(model, "upscale", 4))
        _, _, h, w = lr.shape

        if h <= tile_size and w <= tile_size:
            return model(lr).clamp(0, 1)

        lr_padded = F.pad(lr, [overlap] * 4, mode="replicate")
        ph, pw = int(lr_padded.shape[2]), int(lr_padded.shape[3])

        if ph <= tile_size and pw <= tile_size:
            sr = model(lr_padded)
            c = overlap * scale
            return sr[:, :, c:c + h * scale, c:c + w * scale].clamp(0, 1)

        stride = max(tile_size - overlap, 1)
        sr_tile = tile_size * scale
        out_h, out_w = ph * scale, pw * scale

        output = lr.new_zeros(1, 3, out_h, out_w)
        weight = lr.new_zeros(1, 1, out_h, out_w)

        hann_1d = torch.hann_window(sr_tile, device=lr.device, dtype=lr.dtype)
        hann_2d = (hann_1d[None, :] * hann_1d[:, None]).unsqueeze(0).unsqueeze(0)

        h_starts = list(range(0, max(1, ph - tile_size + 1), stride))
        if h_starts[-1] + tile_size < ph:
            h_starts.append(ph - tile_size)
        w_starts = list(range(0, max(1, pw - tile_size + 1), stride))
        if w_starts[-1] + tile_size < pw:
            w_starts.append(pw - tile_size)

        for top in h_starts:
            for left in w_starts:
                tile = lr_padded[:, :, top:top + tile_size, left:left + tile_size]
                sr = model(tile)
                ot, ol = top * scale, left * scale
                output[:, :, ot:ot + sr_tile, ol:ol + sr_tile] += sr * hann_2d
                weight[:, :, ot:ot + sr_tile, ol:ol + sr_tile] += hann_2d

        output = output / weight.clamp(min=1e-8)
        c = overlap * scale
        return output[:, :, c:c + h * scale, c:c + w * scale].clamp(0, 1)

    def upscale(self, image: Image.Image, *, tile_size: int = 128, overlap: int = 32) -> Tuple[Image.Image, float]:
        if self._model is None:
            raise RuntimeError("Artisan model is not loaded. Call ensure_ready() first.")

        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected RGB image tensor.")
        lr = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        lr = lr.to(
            device=torch.device(self.device),
            dtype=self._dtype,
            memory_format=torch.channels_last,
        )

        use_amp = (self._dtype != torch.float32) and (self.device == "cuda")
        with self._infer_lock, torch.inference_mode():
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=self._dtype, enabled=use_amp):
                sr = self._tile_forward(
                    self._model,
                    lr,
                    tile_size=max(16, int(tile_size)),
                    overlap=max(0, int(overlap)),
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        out = (
            sr[0]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return Image.fromarray(out, mode="RGB"), float(elapsed)

    @staticmethod
    def resize_to_max_edge(image: Image.Image, max_edge: Optional[int]) -> Image.Image:
        if max_edge is None:
            return image
        edge = int(max_edge)
        if edge <= 0 or max(image.size) <= edge:
            return image
        scale = float(edge) / float(max(image.size))
        resized = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        return image.resize(resized, _LANCZOS)
