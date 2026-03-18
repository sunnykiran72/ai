import importlib
import inspect
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger("glamify-ai")


HF_TOKEN_ENV_KEYS: Tuple[str, ...] = ("HUGGING_FACE_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def _flatten_rgba_to_white_rgb(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    base = Image.new("RGB", rgba.size, (255, 255, 255))
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def _as_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_hf_repo_id(raw: str) -> bool:
    value = str(raw or "").strip()
    if not value:
        return False
    if value.startswith(("/", ".", "~")):
        return False
    if Path(value).expanduser().exists():
        return False
    return "/" in value


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
        self._adapter_name_effective = str(self.adapter_name or "")
        self.device = cfg.get("device") or os.getenv("FLUX2_DEVICE", "cuda")
        self.dtype = self._resolve_dtype(cfg.get("dtype") or os.getenv("FLUX2_DTYPE", "fp16"))

        self.width = int(cfg.get("width") or os.getenv("FLUX2_WIDTH", "512"))
        self.height = int(cfg.get("height") or os.getenv("FLUX2_HEIGHT", "768"))
        self.guidance_scale = float(cfg.get("guidance_scale") or os.getenv("FLUX2_GUIDANCE_SCALE", "3.5"))
        self.true_cfg_scale = float(
            cfg["true_cfg_scale"] if ("true_cfg_scale" in cfg and cfg.get("true_cfg_scale") is not None)
            else os.getenv("FLUX2_TRUE_CFG_SCALE", "1.0")
        )
        self.seed = int(cfg.get("seed") or os.getenv("FLUX2_SEED", "23"))
        self.lora_scale = float(
            cfg["lora_scale"] if ("lora_scale" in cfg and cfg.get("lora_scale") is not None) else os.getenv("FLUX2_LORA_SCALE", "1.0")
        )
        self.enable_lora = _as_bool(
            str(cfg.get("enable_lora")) if "enable_lora" in cfg else os.getenv("FLUX2_ENABLE_LORA", "1"),
            default=True,
        )
        self.require_lora = _as_bool(
            str(cfg.get("require_lora")) if "require_lora" in cfg else os.getenv("FLUX2_REQUIRE_LORA", "1"),
            default=True,
        )
        self.lora_auto_download = _as_bool(
            str(cfg.get("lora_auto_download")) if "lora_auto_download" in cfg else os.getenv("FLUX2_LORA_AUTO_DOWNLOAD", "1"),
            default=True,
        )
        self.lora_fallback_repo = str(
            cfg.get("lora_fallback_repo") or os.getenv("FLUX2_LORA_FALLBACK_REPO", "fal/flux-klein-9b-virtual-tryon-lora")
        ).strip()
        self.lora_local_cache_dir = str(
            cfg.get("lora_local_cache_dir") or os.getenv("FLUX2_LORA_LOCAL_CACHE_DIR", "/tmp/flux2-lora/fal-virtual-tryon")
        ).strip()
        self.num_warmups = int(cfg.get("num_warmups") or os.getenv("FLUX2_WARMUPS", "0"))

        self.enable_channels_last = _as_bool(
            str(cfg.get("enable_channels_last")) if "enable_channels_last" in cfg else os.getenv("FLUX2_ENABLE_CHANNELS_LAST", "1"),
            default=True,
        )
        self.fuse_lora = _as_bool(
            str(cfg.get("fuse_lora")) if "fuse_lora" in cfg else os.getenv("FLUX2_FUSE_LORA", "1"),
            default=True,
        )
        self.runtime_lora_toggle = _as_bool(
            str(cfg.get("runtime_lora_toggle")) if "runtime_lora_toggle" in cfg else os.getenv("FLUX2_RUNTIME_LORA_TOGGLE", "0"),
            default=False,
        )
        if self.runtime_lora_toggle and self.fuse_lora:
            logger.info("Disabling LoRA fusion because FLUX2_RUNTIME_LORA_TOGGLE=1.")
            self.fuse_lora = False
        self.compile_mode = str(cfg.get("compile_mode") or os.getenv("FLUX2_COMPILE_MODE", "none")).strip().lower()
        self.compile_vae_decode = _as_bool(
            str(cfg.get("compile_vae_decode")) if "compile_vae_decode" in cfg else os.getenv("FLUX2_COMPILE_VAE_DECODE", "0"),
            default=False,
        )
        self.allow_compile_fallback = _as_bool(
            str(cfg.get("allow_compile_fallback")) if "allow_compile_fallback" in cfg else os.getenv("FLUX2_ALLOW_COMPILE_FALLBACK", "1"),
            default=True,
        )
        self.enable_tf32 = _as_bool(
            str(cfg.get("enable_tf32")) if "enable_tf32" in cfg else os.getenv("FLUX2_ENABLE_TF32", "1"),
            default=True,
        )

        self._pipeline = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._startup_metrics: Dict[str, Any] = {}
        self._did_warmup = False
        self._supports_negative_prompt: Optional[bool] = None
        self._supports_true_cfg_scale: Optional[bool] = None
        self._warned_negative_prompt_unsupported = False
        self._lora_loaded = False
        self._lora_fused = False
        self._lora_runtime_enabled = False
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
    def _find_local_lora_files(path: Path) -> List[Path]:
        if not path.exists():
            return []
        if path.is_file() and path.suffix.lower() == ".safetensors":
            return [path]
        if not path.is_dir():
            return []

        files = sorted(path.glob("*.safetensors"))
        if files:
            return files
        # Fallback for nested weight layouts.
        return sorted(path.rglob("*.safetensors"))

    def _resolve_hf_token(self) -> Optional[str]:
        for key in HF_TOKEN_ENV_KEYS:
            value = str(os.getenv(key, "")).strip()
            if value:
                return value
        return None

    def _candidate_lora_weight_names(self) -> List[str]:
        names = [
            str(self.lora_weight_name or "").strip(),
            "flux-klein-tryon.safetensors",
            "flux-klein-tryon-comfy.safetensors",
            "pytorch_lora_weights.safetensors",
        ]
        dedup: List[str] = []
        seen = set()
        for name in names:
            key = name.strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            dedup.append(key)
        return dedup

    @staticmethod
    def _list_loaded_adapters(pipe: Any) -> List[str]:
        adapters: List[str] = []
        try:
            if hasattr(pipe, "get_active_adapters"):
                active = pipe.get_active_adapters()
                if isinstance(active, (list, tuple)):
                    adapters.extend([str(a) for a in active if a])
            if hasattr(pipe, "get_list_adapters"):
                listed = pipe.get_list_adapters()
                if isinstance(listed, (list, tuple)):
                    adapters.extend([str(a) for a in listed if a])
        except Exception:
            pass
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for name in adapters:
            if name in seen:
                continue
            seen.add(name)
            uniq.append(name)
        return uniq

    def _download_lora_snapshot(self, repo_id: str, local_dir: Path) -> None:
        try:
            from huggingface_hub import snapshot_download
        except Exception as err:
            raise RuntimeError(
                "huggingface_hub is required for FLUX2_LORA_AUTO_DOWNLOAD=1. "
                "Install it or disable auto-download."
            ) from err

        local_dir.mkdir(parents=True, exist_ok=True)
        token = self._resolve_hf_token()
        logger.info(f"[lora] Downloading missing LoRA weights: repo={repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            resume_download=True,
            allow_patterns=["*.safetensors", "*.json", "README.md", "*.txt"],
        )

    def _resolve_lora_source(self) -> Dict[str, Any]:
        raw = str(self.lora_path or "").strip()
        if not raw:
            return {"resolved_source": "", "is_local": False, "downloaded": False}

        local_path = Path(raw).expanduser()
        if local_path.exists():
            files = self._find_local_lora_files(local_path)
            if files:
                return {"resolved_source": str(local_path), "is_local": True, "downloaded": False}
            if not self.lora_auto_download:
                raise RuntimeError(f"Configured FLUX2_LORA_PATH exists but has no .safetensors files: {local_path}")
            repo_to_download = self.lora_fallback_repo or ""
            if not repo_to_download:
                raise RuntimeError(
                    f"Configured FLUX2_LORA_PATH has no .safetensors files and FLUX2_LORA_FALLBACK_REPO is empty: {local_path}"
                )
            self._download_lora_snapshot(repo_to_download, local_path)
            return {"resolved_source": str(local_path), "is_local": True, "downloaded": True}

        if _looks_like_hf_repo_id(raw):
            if not self.lora_auto_download:
                return {"resolved_source": raw, "is_local": False, "downloaded": False}
            cache_dir = Path(self.lora_local_cache_dir).expanduser()
            if not self._find_local_lora_files(cache_dir):
                self._download_lora_snapshot(raw, cache_dir)
                return {"resolved_source": str(cache_dir), "is_local": True, "downloaded": True}
            return {"resolved_source": str(cache_dir), "is_local": True, "downloaded": False}

        if not self.lora_auto_download:
            raise RuntimeError(
                f"Configured FLUX2_LORA_PATH not found: {raw}. "
                "Enable FLUX2_LORA_AUTO_DOWNLOAD=1 or provide a valid local path/repo id."
            )

        repo_to_download = self.lora_fallback_repo or ""
        if not repo_to_download:
            raise RuntimeError(
                f"Configured FLUX2_LORA_PATH not found: {raw}, and FLUX2_LORA_FALLBACK_REPO is empty."
            )
        cache_dir = Path(self.lora_local_cache_dir).expanduser()
        self._download_lora_snapshot(repo_to_download, cache_dir)
        return {"resolved_source": str(cache_dir), "is_local": True, "downloaded": True}

    def _load_lora_with_fallback(self, pipe: Any) -> Dict[str, Any]:
        source_info = self._resolve_lora_source()
        resolved_source = str(source_info.get("resolved_source") or "").strip()
        if not resolved_source:
            raise RuntimeError("LoRA source is empty after resolution.")

        weight_candidates = self._candidate_lora_weight_names()
        attempt_weights = list(weight_candidates)

        if bool(source_info.get("is_local")):
            local_path = Path(resolved_source).expanduser()
            local_files = self._find_local_lora_files(local_path)
            local_names = [p.name for p in local_files]
            if local_names:
                preferred = [w for w in weight_candidates if w in local_names]
                if preferred:
                    attempt_weights = preferred + [w for w in weight_candidates if w not in preferred]
                else:
                    # No explicit match: fallback to the first available local file.
                    attempt_weights = [local_names[0]] + weight_candidates

        load_errors: List[str] = []
        for weight_name in attempt_weights:
            try:
                pipe.load_lora_weights(
                    resolved_source,
                    weight_name=weight_name,
                    adapter_name=self.adapter_name,
                )
                return {
                    "source": resolved_source,
                    "weight_name": weight_name,
                    "downloaded": bool(source_info.get("downloaded")),
                }
            except Exception as err:
                load_errors.append(f"{weight_name}: {err}")

        # Final fallback: let diffusers resolve weight file automatically.
        try:
            pipe.load_lora_weights(resolved_source, adapter_name=self.adapter_name)
            return {
                "source": resolved_source,
                "weight_name": "",
                "downloaded": bool(source_info.get("downloaded")),
            }
        except Exception as err:
            load_errors.append(f"<auto>: {err}")

        details = " | ".join(load_errors[-4:])
        raise RuntimeError(f"Unable to load LoRA weights from {resolved_source}. Attempts: {details}")

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

        lora_load_seconds = 0.0
        lora_loaded = False
        lora_source = ""
        lora_weight_loaded = ""
        lora_downloaded = False
        if not self.enable_lora:
            if self.require_lora:
                raise RuntimeError("LoRA is mandatory but FLUX2_ENABLE_LORA is disabled.")
            logger.info("Skipping LoRA load for Flux2 CVTON (FLUX2_ENABLE_LORA=0).")
        else:
            logger.info(f"Loading LoRA (configured source={self.lora_path})...")
            lora_t0 = time.time()
            try:
                load_meta = self._load_lora_with_fallback(pipe)
                lora_source = str(load_meta.get("source") or "")
                lora_weight_loaded = str(load_meta.get("weight_name") or "")
                lora_downloaded = bool(load_meta.get("downloaded"))
                adapter_name = self.adapter_name
                available_adapters = self._list_loaded_adapters(pipe)
                try:
                    if available_adapters:
                        if adapter_name not in available_adapters:
                            adapter_name = available_adapters[0]
                        if hasattr(pipe, "set_adapters"):
                            pipe.set_adapters([adapter_name], adapter_weights=[self.lora_scale])
                        elif hasattr(pipe, "set_adapter"):
                            pipe.set_adapter(adapter_name)
                    else:
                        logger.warning("LoRA adapters not reported; skipping adapter activation.")
                        adapter_name = ""
                except Exception as err:
                    logger.warning(f"LoRA adapter activation failed; continuing without explicit activation: {err}")
                    adapter_name = ""
                self._adapter_name_effective = str(adapter_name or "")
                if self.fuse_lora and hasattr(pipe, "fuse_lora"):
                    pipe.fuse_lora()
                    self._lora_fused = True
                lora_loaded = True
                self._lora_loaded = True
                self._lora_runtime_enabled = True
                lora_load_seconds = time.time() - lora_t0
            except Exception as err:
                if self.require_lora:
                    raise RuntimeError(f"Failed to load mandatory LoRA: {err}") from err
                logger.warning(f"LoRA load failed; continuing without LoRA: {err}")

        optimize_t0 = time.time()
        self._maybe_channels_last(pipe)
        compile_status = self._maybe_compile(pipe)
        optimize_seconds = time.time() - optimize_t0

        self._startup_metrics = {
            "pipeline_init_total_seconds": time.time() - load_t0,
            "pipeline_model_load_seconds": model_load_seconds,
            "pipeline_lora_load_seconds": lora_load_seconds,
            "lora_enabled": bool(self.enable_lora),
            "lora_required": bool(self.require_lora),
            "lora_loaded": bool(lora_loaded),
            "lora_path": str(self.lora_path or ""),
            "lora_source_resolved": lora_source,
            "lora_weight_loaded": lora_weight_loaded,
            "lora_auto_download": bool(self.lora_auto_download),
            "lora_downloaded": bool(lora_downloaded),
            "pipeline_optimize_seconds": optimize_seconds,
            "compile_mode": self.compile_mode,
            "compile_status": compile_status,
            "compile_vae_decode": self.compile_vae_decode,
            "channels_last": self.enable_channels_last,
            "fuse_lora": self.fuse_lora,
            "runtime_lora_toggle": self.runtime_lora_toggle,
            "tf32_enabled": self.enable_tf32,
            "torch_version": torch.__version__,
            "device": self.device,
            "dtype": str(self.dtype),
            "pipeline_class": getattr(Flux2PipelineClass, "__name__", str(Flux2PipelineClass)),
        }
        return pipe

    def _set_runtime_lora_state(self, enabled: bool) -> bool:
        if self._pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")
        if not self._lora_loaded:
            if enabled and self.require_lora:
                raise RuntimeError("LoRA is required for this request but is not loaded.")
            self._lora_runtime_enabled = False
            return False
        if self._lora_fused:
            if not enabled:
                raise RuntimeError("Cannot disable LoRA at request-time because it was fused into the pipeline.")
            self._lora_runtime_enabled = True
            return True

        pipe = self._pipeline
        if enabled:
            if hasattr(pipe, "enable_lora"):
                pipe.enable_lora()
            if hasattr(pipe, "enable_adapters"):
                pipe.enable_adapters()
            adapter_name = str(self._adapter_name_effective or "")
            if adapter_name:
                try:
                    if hasattr(pipe, "set_adapters"):
                        pipe.set_adapters([adapter_name], adapter_weights=[self.lora_scale])
                    elif hasattr(pipe, "set_adapter"):
                        pipe.set_adapter(adapter_name)
                except Exception as err:
                    logger.warning(f"Runtime LoRA adapter activation failed; continuing without explicit activation: {err}")
            self._lora_runtime_enabled = True
            return True

        if hasattr(pipe, "disable_lora"):
            pipe.disable_lora()
        elif hasattr(pipe, "disable_adapters"):
            pipe.disable_adapters()
        elif hasattr(pipe, "set_adapters"):
            adapter_name = str(self._adapter_name_effective or "")
            if adapter_name:
                try:
                    pipe.set_adapters([adapter_name], adapter_weights=[0.0])
                except Exception as err:
                    logger.warning(f"Runtime LoRA disable failed; continuing without explicit deactivation: {err}")
        else:
            raise RuntimeError("Pipeline does not expose a runtime LoRA disable API.")
        self._lora_runtime_enabled = False
        return False

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

    def _pipeline_accepts_true_cfg_scale(self) -> bool:
        if self._supports_true_cfg_scale is not None:
            return bool(self._supports_true_cfg_scale)

        supported = False
        try:
            if self._pipeline is not None:
                call_sig = inspect.signature(self._pipeline.__call__)
                supported = "true_cfg_scale" in call_sig.parameters
        except Exception:
            supported = False
        self._supports_true_cfg_scale = supported
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
        use_lora: Optional[bool] = None,
        true_cfg_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.ensure_ready()

        run_t0 = time.time()
        gen_seed = seed if seed is not None else self.seed
        generator = torch.Generator(device=self.device).manual_seed(gen_seed)
        warmup_seconds = 0.0

        person = _flatten_rgba_to_white_rgb(person_image)
        board = _flatten_rgba_to_white_rgb(board_image)
        resolved_negative_prompt = str(negative_prompt or "").strip()
        requested_lora = self.enable_lora if use_lora is None else bool(use_lora)
        
        # Determine the CFG scale to use for this run
        effective_true_cfg_scale = float(true_cfg_scale if true_cfg_scale is not None else self.true_cfg_scale)
        
        supports_negative_prompt = self._pipeline_accepts_negative_prompt()
        supports_true_cfg_scale = self._pipeline_accepts_true_cfg_scale()
        supports_native_negative_prompt = bool(supports_negative_prompt and supports_true_cfg_scale)
        effective_prompt = prompt
        negative_prompt_mode = "none"
        if resolved_negative_prompt:
            if supports_native_negative_prompt and effective_true_cfg_scale > 1.0:
                negative_prompt_mode = "native_true_cfg"
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
                if negative_prompt_mode == "native_true_cfg":
                    call_kwargs["negative_prompt"] = resolved_negative_prompt
                    call_kwargs["true_cfg_scale"] = effective_true_cfg_scale
                elif negative_prompt_mode == "ignored" and not self._warned_negative_prompt_unsupported:
                    logger.warning("Flux2 pipeline cannot apply native negative prompts; ignoring provided negative prompt.")
                    self._warned_negative_prompt_unsupported = True
                elif (
                    supports_native_negative_prompt
                    and effective_true_cfg_scale <= 1.0
                    and not self._warned_negative_prompt_unsupported
                ):
                    logger.warning(
                        "Flux2 pipeline exposes native negative prompt inputs, but effective_true_cfg_scale<=1 keeps them inactive; "
                        "using prompt fallback instead."
                    )
                    self._warned_negative_prompt_unsupported = True
            return self._pipeline(**call_kwargs).images[0]

        with self._infer_lock, torch.inference_mode():
            effective_lora = self._set_runtime_lora_state(requested_lora)
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
                "negative_prompt_used": bool(
                    resolved_negative_prompt and negative_prompt_mode in {"native_true_cfg", "prompt_fallback"}
                ),
                "negative_prompt_supported": bool(supports_native_negative_prompt),
                "negative_prompt_argument_supported": bool(supports_negative_prompt),
                "negative_prompt_true_cfg_supported": bool(supports_true_cfg_scale),
                "negative_prompt_mode": negative_prompt_mode,
                "negative_prompt_requested": bool(resolved_negative_prompt),
                "negative_prompt_fallback_mode": self.negative_prompt_fallback_mode,
                "negative_prompt_true_cfg_scale": float(effective_true_cfg_scale),
                "lora_requested": bool(requested_lora),
                "lora_effective": bool(effective_lora),
                "runtime_lora_toggle": bool(self.runtime_lora_toggle),
                "startup_metrics": dict(self._startup_metrics),
            },
        }


    def run_extraction(
        self,
        garment_image: Image.Image,
        prompt: str,
        steps: int = 30,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Extract clean garment from image (background removal + enhancement).
        
        This method is optimized for garment extraction in the /analyze API:
        - Takes a single garment image (no person image)
        - Removes background and enhances garment details
        - Returns clean garment on transparent/white background
        
        Args:
            garment_image: Input garment image (can have background)
            prompt: Text description of the garment (e.g., "Top with crew neck")
            steps: Number of inference steps (default: 30)
            seed: Random seed for reproducibility
            guidance_scale: Guidance scale override (default: uses config)
            
        Returns:
            Dict with:
                - image: PIL Image of extracted garment
                - latency: Generation time in seconds
                - metadata: Generation parameters and metrics
        """
        self.ensure_ready()

        run_t0 = time.time()
        gen_seed = seed if seed is not None else self.seed
        generator = torch.Generator(device=self.device).manual_seed(gen_seed)
        effective_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Flatten RGBA to white background for input
        garment = _flatten_rgba_to_white_rgb(garment_image)

        effective_lora = False
        with self._infer_lock, torch.inference_mode():
            # Disable LoRA for extraction to avoid try-on artifacts.
            prev_lora_state = bool(self._lora_runtime_enabled)
            self._set_runtime_lora_state(False)
            effective_lora = bool(self._lora_runtime_enabled)

            call_kwargs: Dict[str, Any] = {
                "image": garment,
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": effective_guidance,
                "width": self.width,
                "height": self.height,
                "generator": generator,
            }

            infer_t0 = time.time()
            try:
                result = self._pipeline(**call_kwargs).images[0]
                latency = time.time() - infer_t0
            finally:
                self._set_runtime_lora_state(prev_lora_state)

        return {
            "image": result,
            "latency": latency,
            "metadata": {
                "pipeline": "flux2_extraction",
                "steps": steps,
                "seed": gen_seed,
                "guidance_scale": effective_guidance,
                "resolution": (self.width, self.height),
                "request_total_seconds": time.time() - run_t0,
                "base_garment_prompt": prompt,
                "prompt_description": prompt,
                "lora_enabled": bool(effective_lora),
                "startup_metrics": dict(self._startup_metrics),
            },
        }


if __name__ == "__main__":
    # Test block for isolated validation
    logging.basicConfig(level=logging.INFO)
    print("Runner initialized (Isolated Mode)")
    # Note: Full test requires GPU and model weights
