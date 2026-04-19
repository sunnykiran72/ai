import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from diffusers import DiffusionPipeline

logger = logging.getLogger("glamify-ai")


def _looks_like_repo_id(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    if text.startswith(".") or text.startswith("/"):
        return False
    return "/" in text


def _align_to_model_grid(value: int, *, base: int = 8) -> int:
    raw = max(int(value), int(base))
    return max(int(base), (raw // int(base)) * int(base))


class QwenImageEditRunner:
    """
    Minimal Qwen Image Edit runner with optional LoRA adapter support.
    This intentionally keeps only the stable path (no rapid/FA3/compile branches).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        lora_repo: Optional[str] = None,
        lora_weight_name: Optional[str] = None,
    ) -> None:
        self.model_id = str(model_id or os.getenv("QWEN_IMAGE_EDIT_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")).strip()
        self.model_path = str(os.getenv("QWEN_IMAGE_EDIT_MODEL_PATH", "")).strip()
        self.lora_repo = str(
            lora_repo or os.getenv("QWEN_IMAGE_EDIT_LORA_REPO", "prithivMLmods/QIE-2511-Extract-Outfit")
        ).strip()
        self.lora_path = str(os.getenv("QWEN_IMAGE_EDIT_LORA_PATH", "")).strip()
        self.lora_weight_name = str(
            lora_weight_name or os.getenv("QWEN_IMAGE_EDIT_LORA_WEIGHT_NAME", "QIE-2511-Extract-Outfit-4200.safetensors")
        ).strip()
        self.lora_scale = float(os.getenv("QWEN_IMAGE_EDIT_LORA_SCALE", "1.0"))
        self.enable_lora = str(os.getenv("QWEN_IMAGE_EDIT_ENABLE_LORA", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.adapter_name = str(os.getenv("QWEN_IMAGE_EDIT_LORA_ADAPTER_NAME", "extract_outfit")).strip() or "extract_outfit"
        self.warmup_prompt = str(
            os.getenv("QWEN_IMAGE_EDIT_WARMUP_PROMPT", "Extract the clothing and create a flat mockup.")
        ).strip() or "Extract the clothing and create a flat mockup."
        self.warmup_steps = max(1, int(os.getenv("QWEN_IMAGE_EDIT_WARMUP_STEPS", "4")))
        self.warmup_edge = max(256, int(os.getenv("QWEN_IMAGE_EDIT_WARMUP_EDGE", "512")))
        self._warmed_up = False
        grid_raw = str(os.getenv("QWEN_IMAGE_EDIT_GRID_BASE", "8")).strip()
        self.grid_base = 16 if grid_raw == "16" else 8

        requested_device = str(os.getenv("QWEN_IMAGE_EDIT_DEVICE", "auto")).strip().lower()
        if requested_device not in {"auto", "cuda", "cpu"}:
            requested_device = "auto"
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device
        requested_dtype = str(os.getenv("QWEN_IMAGE_EDIT_DTYPE", "bf16")).strip().lower()
        self.torch_dtype = self._resolve_torch_dtype(requested_dtype)
        self.dtype_name = self._dtype_name(self.torch_dtype)

        self._pipeline: Optional[Any] = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._lora_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _resolve_model_source(self) -> str:
        local = self.model_path
        if local:
            p = Path(local).expanduser()
            if p.exists():
                return str(p)
        return self.model_id

    def _resolve_lora_source(self) -> str:
        local = self.lora_path
        if local:
            p = Path(local).expanduser()
            if p.exists():
                return str(p)
        return self.lora_repo

    def _resolve_torch_dtype(self, requested_dtype: str):
        if self.device != "cuda":
            return torch.float32
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if requested_dtype in aliases:
            return aliases[requested_dtype]
        return torch.bfloat16

    @staticmethod
    def _dtype_name(dtype_obj) -> str:
        return str(dtype_obj).replace("torch.", "")

    def _load_pipeline(self) -> None:
        source = self._resolve_model_source()
        kwargs: Dict[str, Any] = {"torch_dtype": self.torch_dtype}
        if self.device == "cuda":
            kwargs["device_map"] = "cuda"

        logger.info("Loading Qwen image-edit pipeline from %s", source)
        try:
            pipe = DiffusionPipeline.from_pretrained(source, **kwargs)
        except TypeError:
            # Backward-compatible fallback for pipelines that use `dtype`.
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("torch_dtype", None)
            fallback_kwargs["dtype"] = self.torch_dtype
            pipe = DiffusionPipeline.from_pretrained(source, **fallback_kwargs)

        if self.device != "cuda":
            pipe = pipe.to(self.device)

        lora_loaded = False
        if self.enable_lora:
            lora_source = self._resolve_lora_source()
            try:
                if self.lora_weight_name:
                    pipe.load_lora_weights(
                        lora_source,
                        weight_name=self.lora_weight_name,
                        adapter_name=self.adapter_name,
                    )
                else:
                    pipe.load_lora_weights(lora_source, adapter_name=self.adapter_name)
                lora_loaded = True

                if hasattr(pipe, "set_adapters"):
                    try:
                        pipe.set_adapters([self.adapter_name], adapter_weights=[float(self.lora_scale)])
                    except Exception:
                        pipe.set_adapters(self.adapter_name, adapter_weights=[float(self.lora_scale)])
            except Exception as exc:
                raise RuntimeError(f"Failed to load Qwen Extract-Outfit LoRA from '{lora_source}': {exc}") from exc

        if self.device == "cuda":
            try:
                pipe = pipe.to(dtype=self.torch_dtype)
            except Exception as exc:
                logger.warning("Failed to cast Qwen pipeline to dtype=%s: %s", self.dtype_name, exc)

        self._pipeline = pipe
        self._lora_loaded = bool(lora_loaded)
        logger.info(
            "Qwen image-edit pipeline ready (device=%s, dtype=%s, lora_loaded=%s)",
            self.device,
            self.dtype_name,
            self._lora_loaded,
        )

    def ensure_ready(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if not self.is_loaded:
                self._load_pipeline()

    def unload(self) -> None:
        with self._load_lock:
            self._pipeline = None
            self._lora_loaded = False
            self._warmed_up = False
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def warmup(self) -> None:
        self.ensure_ready()
        if self._warmed_up:
            return
        image = Image.new("RGB", (self.warmup_edge, self.warmup_edge), color="white")
        try:
            self.run_edit(
                image,
                prompt=self.warmup_prompt,
                steps=self.warmup_steps,
                guidance_scale=1.0,
                negative_prompt=None,
                seed=1,
                output_width=self.warmup_edge,
                output_height=self.warmup_edge,
            )
            self._warmed_up = True
            logger.info(
                "Qwen image-edit warmup completed (steps=%s, edge=%s)",
                int(self.warmup_steps),
                int(self.warmup_edge),
            )
        except Exception as exc:
            logger.warning("Qwen image-edit warmup failed: %s", exc)

    def run_edit(
        self,
        image: Image.Image,
        *,
        prompt: str,
        steps: int = 28,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = 42,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        self.ensure_ready()
        if self._pipeline is None:
            raise RuntimeError("Qwen image-edit pipeline is not loaded.")

        prepared = image.convert("RGB")
        run_kwargs: Dict[str, Any] = {
            "image": prepared,
            "prompt": str(prompt or "").strip(),
            "num_inference_steps": int(steps),
        }
        if guidance_scale is not None:
            run_kwargs["guidance_scale"] = float(guidance_scale)
        cleaned_negative = " ".join(str(negative_prompt or "").split()).strip()
        if cleaned_negative:
            run_kwargs["negative_prompt"] = cleaned_negative

        if seed is not None:
            generator = torch.Generator(device=self.device if self.device == "cuda" else "cpu").manual_seed(int(seed))
            run_kwargs["generator"] = generator

        requested_output_size = None
        requested_output_size_aligned = None
        if output_width is not None and output_height is not None:
            requested_output_size = {"width": int(output_width), "height": int(output_height)}
            aligned_width = _align_to_model_grid(int(output_width), base=int(self.grid_base))
            aligned_height = _align_to_model_grid(int(output_height), base=int(self.grid_base))
            run_kwargs["width"] = int(aligned_width)
            run_kwargs["height"] = int(aligned_height)
            requested_output_size_aligned = {"width": int(aligned_width), "height": int(aligned_height)}

        with self._infer_lock, torch.inference_mode():
            try:
                result = self._pipeline(**run_kwargs)
            except TypeError as exc:
                # Some pipeline builds may not expose width/height kwargs.
                if ("width" in run_kwargs or "height" in run_kwargs) and (
                    "width" in str(exc).lower() or "height" in str(exc).lower()
                ):
                    run_kwargs.pop("width", None)
                    run_kwargs.pop("height", None)
                    requested_output_size = None
                    requested_output_size_aligned = None
                    result = self._pipeline(**run_kwargs)
                else:
                    raise
            except Exception as exc:
                # Some Qwen stacks fail when generator device mismatches internal tensor device.
                if "generator" in run_kwargs and (
                    "generator" in str(exc).lower() or "same device" in str(exc).lower() or "cpu tensor" in str(exc).lower()
                ):
                    run_kwargs.pop("generator", None)
                    result = self._pipeline(**run_kwargs)
                else:
                    raise

        images = getattr(result, "images", None) or []
        if not images:
            raise RuntimeError("Qwen image-edit pipeline returned no images.")

        metadata = {
            "model_id": self.model_id,
            "model_source": self._resolve_model_source(),
            "lora_repo": self.lora_repo,
            "lora_source": self._resolve_lora_source() if self.enable_lora else "",
            "lora_weight_name": self.lora_weight_name,
            "lora_scale": float(self.lora_scale),
            "lora_enabled": bool(self.enable_lora),
            "lora_loaded": bool(self._lora_loaded),
            "device": self.device,
            "dtype": self.dtype_name,
            "steps": int(steps),
            "guidance_scale": (float(guidance_scale) if guidance_scale is not None else None),
            "negative_prompt_supplied": bool(cleaned_negative),
            "seed": int(seed) if seed is not None else None,
            "lora_source_is_repo": _looks_like_repo_id(self.lora_repo),
            "requested_output_size": requested_output_size,
            "requested_output_size_aligned": requested_output_size_aligned,
            "grid_base": int(self.grid_base),
        }
        return images[0], metadata
