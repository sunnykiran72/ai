import importlib
import inspect
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger("glamify-ai")


def _torch_version_at_least(version: str) -> bool:
    def _parts(raw: str) -> List[int]:
        out: List[int] = []
        for token in raw.split("+", 1)[0].split("."):
            num = ""
            for ch in token:
                if ch.isdigit():
                    num += ch
                else:
                    break
            out.append(int(num) if num else 0)
        while len(out) < 3:
            out.append(0)
        return out[:3]

    return _parts(torch.__version__) >= _parts(version)


def _apply_hybridcache_shim() -> None:
    try:
        import transformers
        import transformers.cache_utils as cache_utils
    except Exception:
        return

    class HybridCache:  # pragma: no cover - runtime compatibility shim
        pass

    if not hasattr(transformers, "HybridCache"):
        setattr(transformers, "HybridCache", HybridCache)
    if not hasattr(cache_utils, "HybridCache"):
        setattr(cache_utils, "HybridCache", HybridCache)


def _patch_attention_dispatch_for_torch_compat() -> bool:
    if _torch_version_at_least("2.6.0"):
        return False

    try:
        import diffusers
    except Exception:
        return False

    dispatch_path = Path(diffusers.__file__).resolve().parent / "models" / "attention_dispatch.py"
    if not dispatch_path.exists():
        return False

    source = dispatch_path.read_text(encoding="utf-8")
    updated = source

    old_guard = (
        '# Version guard for PyTorch compatibility - custom_op was added in PyTorch 2.4\n'
        'if torch.__version__ >= "2.4.0":\n'
        "    _custom_op = torch.library.custom_op\n"
        "    _register_fake = torch.library.register_fake\n"
        "else:\n"
    )
    new_guard = (
        "# Version guard for PyTorch compatibility - custom_op registration here\n"
        "# requires newer torch schema handling.\n"
        'if torch.__version__ >= "2.6.0":\n'
        "    _custom_op = torch.library.custom_op\n"
        "    _register_fake = torch.library.register_fake\n"
        "else:\n"
    )
    if old_guard in updated:
        updated = updated.replace(old_guard, new_guard, 1)

    old_sdpa = (
        "    if _parallel_config is None:\n"
        "        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))\n"
        "        out = torch.nn.functional.scaled_dot_product_attention(\n"
        "            query=query,\n"
        "            key=key,\n"
        "            value=value,\n"
        "            attn_mask=attn_mask,\n"
        "            dropout_p=dropout_p,\n"
        "            is_causal=is_causal,\n"
        "            scale=scale,\n"
        "            enable_gqa=enable_gqa,\n"
        "        )\n"
        "        out = out.permute(0, 2, 1, 3)\n"
    )
    new_sdpa = (
        "    if _parallel_config is None:\n"
        "        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))\n"
        "        sdpa_kwargs = dict(\n"
        "            query=query,\n"
        "            key=key,\n"
        "            value=value,\n"
        "            attn_mask=attn_mask,\n"
        "            dropout_p=dropout_p,\n"
        "            is_causal=is_causal,\n"
        "            scale=scale,\n"
        "        )\n"
        '        if is_torch_version(">=", "2.5.0"):\n'
        '            sdpa_kwargs["enable_gqa"] = enable_gqa\n'
        "        out = torch.nn.functional.scaled_dot_product_attention(**sdpa_kwargs)\n"
        "        out = out.permute(0, 2, 1, 3)\n"
    )
    if old_sdpa in updated:
        updated = updated.replace(old_sdpa, new_sdpa, 1)

    if updated == source:
        return False

    dispatch_path.write_text(updated, encoding="utf-8")
    importlib.invalidate_caches()
    return True


def _import_flux2_klein_pipeline() -> Any:
    try:
        patched = _patch_attention_dispatch_for_torch_compat()
        if patched:
            logger.info("[compat] Patched diffusers attention_dispatch for torch compatibility")
    except Exception as err:
        logger.warning(f"[compat] attention patch failed: {err}")

    import_errors: List[str] = []

    for label, loader in (
        ("Flux2KleinPipeline", lambda: __import__("diffusers", fromlist=["Flux2KleinPipeline"]).Flux2KleinPipeline),
        (
            "Flux2Pipeline",
            lambda: __import__(
                "diffusers.pipelines.flux2.pipeline_flux2",
                fromlist=["Flux2Pipeline"],
            ).Flux2Pipeline,
        ),
    ):
        try:
            pipeline_cls = loader()
            if label != "Flux2KleinPipeline":
                logger.warning(f"[compat] Falling back to {label}; Flux2KleinPipeline is unavailable in this diffusers build.")
            return pipeline_cls
        except Exception as err:
            if "HybridCache" in str(err):
                _apply_hybridcache_shim()
                try:
                    pipeline_cls = loader()
                    if label != "Flux2KleinPipeline":
                        logger.warning(
                            f"[compat] Falling back to {label} after HybridCache shim; Flux2KleinPipeline is unavailable."
                        )
                    return pipeline_cls
                except Exception as shim_err:
                    import_errors.append(f"{label} (after HybridCache shim): {shim_err}")
                    continue
            import_errors.append(f"{label}: {err}")

    details = " | ".join(import_errors)
    raise RuntimeError(
        "Unable to import a FLUX.2 pipeline class from diffusers. "
        "Install compatible diffusers/transformers versions for FLUX.2-klein-9B support. "
        f"Details: {details}"
    )


class Flux2CVTONRunner:
    """
    Stateful runner for FLUX.2-klein-9B + VTO LoRA.
    Encapsulates model loading, runtime optimization, and inference.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.model_path = cfg.get("model_path") or os.getenv("FLUX2_MODEL_PATH", "./flux2-klein")
        self.lora_path = cfg.get("lora_path") or os.getenv("FLUX2_LORA_PATH", "fal/flux-klein-9b-virtual-tryon-lora")
        self.lora_weight_name = cfg.get("lora_weight_name") or os.getenv("FLUX2_LORA_WEIGHT_NAME", "flux-klein-tryon.safetensors")
        self.adapter_name = cfg.get("adapter_name") or os.getenv("FLUX2_ADAPTER_NAME", "fal_tryon")
        self.device = cfg.get("device") or os.getenv("FLUX2_DEVICE", "cuda")
        self.dtype = self._resolve_dtype(cfg.get("dtype") or os.getenv("FLUX2_DTYPE", "fp16"))

        self.width = int(os.getenv("FLUX2_WIDTH", "512"))
        self.height = int(os.getenv("FLUX2_HEIGHT", "768"))
        self.guidance_scale = float(os.getenv("FLUX2_GUIDANCE_SCALE", "3.5"))
        self.seed = int(os.getenv("FLUX2_SEED", "23"))
        self.lora_scale = float(os.getenv("FLUX2_LORA_SCALE", "1.0"))
        self.num_warmups = int(os.getenv("FLUX2_WARMUPS", "0"))

        self.enable_channels_last = os.getenv("FLUX2_ENABLE_CHANNELS_LAST", "1") == "1"
        self.fuse_lora = os.getenv("FLUX2_FUSE_LORA", "1") == "1"
        self.compile_mode = os.getenv("FLUX2_COMPILE_MODE", "none").strip().lower()
        self.compile_vae_decode = os.getenv("FLUX2_COMPILE_VAE_DECODE", "0") == "1"
        self.allow_compile_fallback = os.getenv("FLUX2_ALLOW_COMPILE_FALLBACK", "1") == "1"
        self.enable_tf32 = os.getenv("FLUX2_ENABLE_TF32", "1") == "1"

        self._pipeline = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._startup_metrics: Dict[str, Any] = {}
        self._did_warmup = False
        self._supports_negative_prompt: Optional[bool] = None
        self._warned_negative_prompt_unsupported = False
        self.negative_prompt_fallback_mode = (
            str(os.getenv("FLUX2_NEGATIVE_PROMPT_FALLBACK_MODE", "append_prompt")).strip().lower()
        )
        if self.negative_prompt_fallback_mode not in {"append_prompt", "ignore"}:
            self.negative_prompt_fallback_mode = "append_prompt"
        self.negative_prompt_fallback_max_chars = max(
            240,
            int(os.getenv("FLUX2_NEGATIVE_PROMPT_FALLBACK_MAX_CHARS", "900")),
        )

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> torch.dtype:
        value = str(dtype_name).lower()
        if "bf16" in value:
            return torch.bfloat16
        if "fp32" in value:
            return torch.float32
        return torch.float16

    def _maybe_channels_last(self, pipe: Any) -> None:
        if not self.enable_channels_last:
            return
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(memory_format=torch.channels_last)
        if hasattr(pipe, "vae"):
            pipe.vae.to(memory_format=torch.channels_last)

    def _maybe_compile(self, pipe: Any) -> str:
        if self.compile_mode == "none":
            return "disabled"

        try:
            pipe.transformer = torch.compile(pipe.transformer, mode=self.compile_mode, fullgraph=False, dynamic=True)
            if self.compile_vae_decode and hasattr(pipe, "vae") and hasattr(pipe.vae, "decode"):
                pipe.vae.decode = torch.compile(pipe.vae.decode, mode=self.compile_mode, fullgraph=False, dynamic=True)
            return "enabled"
        except Exception as err:
            if not self.allow_compile_fallback:
                raise
            logger.warning(f"torch.compile failed, falling back to eager mode: {err}")
            return "failed_fallback_eager"

    def _load_pipeline(self):
        load_t0 = time.time()
        if self.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        Flux2PipelineClass = _import_flux2_klein_pipeline()
        logger.info(f"Loading Flux2 CVTON from {self.model_path}...")
        model_load_t0 = time.time()
        pipe = Flux2PipelineClass.from_pretrained(self.model_path, torch_dtype=self.dtype)
        pipe.to(self.device)
        model_load_seconds = time.time() - model_load_t0

        logger.info(f"Loading LoRA {self.lora_path}...")
        lora_t0 = time.time()
        pipe.load_lora_weights(
            self.lora_path,
            weight_name=self.lora_weight_name,
            adapter_name=self.adapter_name,
        )
        pipe.set_adapters([self.adapter_name], adapter_weights=[self.lora_scale])

        if self.fuse_lora and hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora()
        lora_load_seconds = time.time() - lora_t0

        optimize_t0 = time.time()
        self._maybe_channels_last(pipe)
        compile_status = self._maybe_compile(pipe)
        optimize_seconds = time.time() - optimize_t0

        self._startup_metrics = {
            "pipeline_init_total_seconds": time.time() - load_t0,
            "pipeline_model_load_seconds": model_load_seconds,
            "pipeline_lora_load_seconds": lora_load_seconds,
            "pipeline_optimize_seconds": optimize_seconds,
            "compile_mode": self.compile_mode,
            "compile_status": compile_status,
            "compile_vae_decode": self.compile_vae_decode,
            "channels_last": self.enable_channels_last,
            "fuse_lora": self.fuse_lora,
            "tf32_enabled": self.enable_tf32,
            "torch_version": torch.__version__,
            "device": self.device,
            "dtype": str(self.dtype),
            "pipeline_class": getattr(Flux2PipelineClass, "__name__", str(Flux2PipelineClass)),
        }
        return pipe

    def ensure_ready(self):
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is None:
                self._pipeline = self._load_pipeline()

    def _pipeline_accepts_negative_prompt(self) -> bool:
        if self._supports_negative_prompt is not None:
            return bool(self._supports_negative_prompt)

        supported = False
        try:
            if self._pipeline is not None:
                call_sig = inspect.signature(self._pipeline.__call__)
                supported = "negative_prompt" in call_sig.parameters
        except Exception:
            supported = False
        self._supports_negative_prompt = supported
        return supported

    def _negative_prompt_to_prompt_suffix(self, negative_prompt: str) -> str:
        raw = " ".join(str(negative_prompt or "").split()).strip()
        if not raw:
            return ""

        chunks = [c.strip(" ,.;") for c in raw.split("|") if c.strip(" ,.;")]
        tokens: List[str] = []
        for chunk in chunks:
            parts = [p.strip(" ,.;") for p in chunk.split(",") if p.strip(" ,.;")]
            tokens.extend(parts if parts else [chunk])

        dedup: List[str] = []
        seen = set()
        for tok in tokens:
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(tok)

        if not dedup:
            return ""

        joined = "; ".join(dedup)
        if len(joined) > self.negative_prompt_fallback_max_chars:
            joined = joined[: self.negative_prompt_fallback_max_chars].rstrip(" ,;.")
        return (
            "Hard constraints: keep the exact same person identity, pose, body proportions, and background. "
            "Do not introduce these artifacts: "
            + joined
            + "."
        )

    def run_tryon(
        self,
        person_image: Image.Image,
        board_image: Image.Image,
        prompt: str,
        steps: int = 6,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ensure_ready()

        run_t0 = time.time()
        gen_seed = seed if seed is not None else self.seed
        generator = torch.Generator(device=self.device).manual_seed(gen_seed)
        warmup_seconds = 0.0

        person = person_image.convert("RGB")
        board = board_image.convert("RGB")
        resolved_negative_prompt = str(negative_prompt or "").strip()
        supports_negative_prompt = self._pipeline_accepts_negative_prompt()
        effective_prompt = prompt
        negative_prompt_mode = "none"
        if resolved_negative_prompt:
            if supports_negative_prompt:
                negative_prompt_mode = "native"
            elif self.negative_prompt_fallback_mode == "append_prompt":
                suffix = self._negative_prompt_to_prompt_suffix(resolved_negative_prompt)
                if suffix:
                    effective_prompt = f"{prompt} {suffix}"
                    negative_prompt_mode = "prompt_fallback"
                else:
                    negative_prompt_mode = "ignored"
            else:
                negative_prompt_mode = "ignored"

        def _invoke_once() -> Any:
            call_kwargs: Dict[str, Any] = {
                "image": [person, board],
                "prompt": effective_prompt,
                "num_inference_steps": steps,
                "guidance_scale": self.guidance_scale,
                "width": self.width,
                "height": self.height,
                "generator": generator,
            }
            if resolved_negative_prompt:
                if supports_negative_prompt:
                    call_kwargs["negative_prompt"] = resolved_negative_prompt
                elif negative_prompt_mode == "ignored" and not self._warned_negative_prompt_unsupported:
                    logger.warning("Flux2 pipeline does not expose `negative_prompt`; ignoring provided negative prompt.")
                    self._warned_negative_prompt_unsupported = True
            return self._pipeline(**call_kwargs).images[0]

        with self._infer_lock, torch.inference_mode():
            if (not self._did_warmup) and self.num_warmups > 0:
                warm_t0 = time.time()
                for _ in range(max(0, self.num_warmups)):
                    _ = _invoke_once()
                warmup_seconds = time.time() - warm_t0
                self._did_warmup = True

            infer_t0 = time.time()
            result = _invoke_once()
            latency = time.time() - infer_t0

        return {
            "image": result,
            "latency": latency,
            "metadata": {
                "steps": steps,
                "seed": gen_seed,
                "resolution": (self.width, self.height),
                "warmup_seconds": warmup_seconds,
                "request_total_seconds": time.time() - run_t0,
                "negative_prompt_used": bool(resolved_negative_prompt and negative_prompt_mode in {"native", "prompt_fallback"}),
                "negative_prompt_supported": bool(supports_negative_prompt),
                "negative_prompt_mode": negative_prompt_mode,
                "negative_prompt_requested": bool(resolved_negative_prompt),
                "negative_prompt_fallback_mode": self.negative_prompt_fallback_mode,
                "startup_metrics": dict(self._startup_metrics),
            },
        }


if __name__ == "__main__":
    # Test block for isolated validation
    logging.basicConfig(level=logging.INFO)
    print("Runner initialized (Isolated Mode)")
    # Note: Full test requires GPU and model weights
