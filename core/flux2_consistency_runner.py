import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageChops

from utils.consistency_prompt_policy import build_consistency_prompt_plan

logger = logging.getLogger("glamify-ai")


MODEL_SOURCE_ID = "black-forest-labs/FLUX.2-klein-9B"
MODEL_LOCAL_PATH = "/workspace/models/flux2-klein"
LORA_SOURCE_REPO_ID = "dx8152/Flux2-Klein-9B-Consistency"
LORA_LOCAL_DIR = "/workspace/models/flux2-lora/dx8152-consistency"
LORA_WEIGHT_NAME = "Klein-consistency.safetensors"
LORA_ADAPTER_NAME = "klein_consistency"
DEFAULT_LORA_SCALE = 0.8
DEFAULT_STEPS = 10
DEFAULT_SEED = 42
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_MAX_EDGE = 768


def _flatten_rgba_to_white_rgb(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    base = Image.new("RGB", rgba.size, (255, 255, 255))
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def _trim_reference_margins(image: Image.Image, *, tolerance: int = 10) -> Image.Image:
    rgb = _flatten_rgba_to_white_rgb(image)
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg).convert("L")
    mask = diff.point(lambda p: 255 if p > tolerance else 0)
    bbox = mask.getbbox()
    if not bbox:
        return rgb
    cropped = rgb.crop(bbox)
    if cropped.width < 8 or cropped.height < 8:
        return rgb
    return cropped


def _resolve_flux2_pipeline_class() -> Any:
    try:
        from diffusers import Flux2KleinPipeline  # type: ignore

        return Flux2KleinPipeline
    except Exception:
        pass

    try:
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline  # type: ignore

        return Flux2Pipeline
    except Exception as exc:
        raise RuntimeError(f"Could not import a FLUX2 pipeline class from diffusers: {exc}")


def _list_active_adapters(pipe: Any) -> List[str]:
    names: List[str] = []
    if pipe is None:
        return names
    if hasattr(pipe, "get_active_adapters"):
        try:
            raw = pipe.get_active_adapters()  # type: ignore[attr-defined]
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    name = str(item or "").strip()
                    if name and name not in names:
                        names.append(name)
        except Exception:
            pass
    if not names and hasattr(pipe, "active_adapters"):
        try:
            raw = getattr(pipe, "active_adapters")
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    name = str(item or "").strip()
                    if name and name not in names:
                        names.append(name)
        except Exception:
            pass
    return names


class Flux2ConsistencyRunner:
    """FLUX.2 single-LoRA consistency try-on runner."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        self.model_source_id = str(cfg.get("model_source_id") or MODEL_SOURCE_ID)
        self.model_local_path = str(cfg.get("model_local_path") or MODEL_LOCAL_PATH)
        self.lora_source_repo = str(cfg.get("lora_source_repo") or LORA_SOURCE_REPO_ID)
        self.lora_local_dir = str(cfg.get("lora_local_dir") or LORA_LOCAL_DIR)
        self.lora_weight_name = str(cfg.get("lora_weight_name") or LORA_WEIGHT_NAME)
        self.adapter_name = str(cfg.get("adapter_name") or LORA_ADAPTER_NAME)

        self.default_lora_scale = float(cfg.get("lora_scale") or DEFAULT_LORA_SCALE)
        self.default_steps = int(cfg.get("steps") or DEFAULT_STEPS)
        self.default_seed = int(cfg.get("seed") or DEFAULT_SEED)
        self.default_guidance_scale = float(cfg.get("guidance_scale") or DEFAULT_GUIDANCE_SCALE)
        self.max_edge = int(cfg.get("max_edge") or DEFAULT_MAX_EDGE)

        self._pipeline = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._startup_metrics: Dict[str, Any] = {}

    @staticmethod
    def _resolve_generator(seed: int) -> torch.Generator:
        if torch.cuda.is_available():
            return torch.Generator(device="cuda").manual_seed(seed)
        return torch.Generator().manual_seed(seed)

    def _resolve_dimensions(self, person_image: Image.Image) -> Tuple[int, int]:
        src_w, src_h = [int(v) for v in person_image.size]
        if src_w <= 0 or src_h <= 0:
            return 576, 768
        scale = min(1.0, float(self.max_edge) / float(max(src_w, src_h)))
        out_w = max(64, int(round((src_w * scale) / 8.0)) * 8)
        out_h = max(64, int(round((src_h * scale) / 8.0)) * 8)
        return out_w, out_h

    @staticmethod
    def _fit_reference_to_canvas(reference: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        tw, th = target_size
        canvas = Image.new("RGB", (tw, th), (255, 255, 255))
        trimmed = _trim_reference_margins(reference)
        rw, rh = trimmed.size
        usable_w = max(1, int(tw * 0.92))
        usable_h = max(1, int(th * 0.92))
        scale = min(float(usable_w) / float(rw), float(usable_h) / float(rh))
        nw, nh = max(1, int(rw * scale)), max(1, int(rh * scale))
        resized = trimmed.resize((nw, nh), Image.Resampling.LANCZOS)
        ox = (tw - nw) // 2
        oy = (th - nh) // 2
        canvas.paste(resized, (ox, oy))
        return canvas

    def _load_pipeline(self):
        load_t0 = time.time()
        pipeline_cls = _resolve_flux2_pipeline_class()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = pipeline_cls.from_pretrained(
            self.model_local_path,
            torch_dtype=dtype,
            local_files_only=True,
        )

        if device == "cuda" and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        pipe.load_lora_weights(
            self.lora_local_dir,
            weight_name=self.lora_weight_name,
            adapter_name=self.adapter_name,
        )
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([self.adapter_name], adapter_weights=[self.default_lora_scale])
        elif hasattr(pipe, "set_adapter"):
            pipe.set_adapter(self.adapter_name)

        self._startup_metrics = {
            "pipeline_class": getattr(pipeline_cls, "__name__", str(pipeline_cls)),
            "device": device,
            "dtype": str(dtype),
            "model_source_id": self.model_source_id,
            "model_local_path": self.model_local_path,
            "lora_source_repo": self.lora_source_repo,
            "lora_local_dir": self.lora_local_dir,
            "lora_weight_name": self.lora_weight_name,
            "load_seconds": time.time() - load_t0,
        }
        return pipe

    def ensure_ready(self):
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is None:
                self._pipeline = self._load_pipeline()

    def run_tryon(
        self,
        person_image: Image.Image,
        board_image: Image.Image,
        *,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        lora_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        target_types: Optional[List[str]] = None,
        source_worn_types: Optional[List[str]] = None,
        garment_descriptions: Optional[List[str]] = None,
        user_description: Optional[str] = None,
        board_mode: str = "single",
        prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ensure_ready()
        if self._pipeline is None:
            raise RuntimeError("Consistency pipeline unavailable")

        run_t0 = time.time()
        person = _flatten_rgba_to_white_rgb(person_image)
        board = _flatten_rgba_to_white_rgb(board_image)
        width, height = self._resolve_dimensions(person)
        person_resized = person.resize((width, height), Image.Resampling.LANCZOS)
        board_canvas = self._fit_reference_to_canvas(board, (width, height))

        gen_steps = int(steps if steps is not None else self.default_steps)
        gen_seed = int(seed if seed is not None else self.default_seed)
        gen_guidance = float(guidance_scale if guidance_scale is not None else self.default_guidance_scale)
        effective_lora_scale = float(lora_scale if lora_scale is not None else self.default_lora_scale)
        generator = self._resolve_generator(gen_seed)
        prompt_plan = build_consistency_prompt_plan(
            target_types=target_types or [],
            source_worn_types=source_worn_types or [],
            source_text=user_description,
            garment_descriptions=garment_descriptions or [],
            board_mode=board_mode,
        )
        override_text = " ".join(str(prompt_override or "").split()).strip()
        prompt_text = override_text or prompt_plan.prompt
        prompt_source = "override" if override_text else "policy"

        with self._infer_lock, torch.inference_mode():
            if hasattr(self._pipeline, "set_adapters"):
                self._pipeline.set_adapters([self.adapter_name], adapter_weights=[effective_lora_scale])
            infer_t0 = time.time()
            result = self._pipeline(
                image=[person_resized, board_canvas],
                prompt=prompt_text,
                num_inference_steps=gen_steps,
                guidance_scale=gen_guidance,
                width=width,
                height=height,
                generator=generator,
            ).images[0]
            latency = time.time() - infer_t0

        return {
            "image": result,
            "latency": latency,
            "metadata": {
                "tryon_impl": "consistency_lora",
                "steps": gen_steps,
                "seed": gen_seed,
                "resolution": (width, height),
                "input_resolution": (int(person.width), int(person.height)),
                "request_total_seconds": time.time() - run_t0,
                "lora_scale": effective_lora_scale,
                "guidance_scale": gen_guidance,
                "prompt": prompt_text,
                "prompt_policy_id": "manual_override_v1" if override_text else prompt_plan.policy_id,
                "prompt_template_version": prompt_plan.template_version,
                "prompt_source": prompt_source,
                "prompt_override_applied": bool(override_text),
                "policy_target_types": list(prompt_plan.target_types),
                "policy_source_worn_types": list(prompt_plan.source_worn_types),
                "policy_board_mode": prompt_plan.board_mode,
                "model_source_id": self.model_source_id,
                "model_local_path": self.model_local_path,
                "lora_source_repo": self.lora_source_repo,
                "lora_local_dir": self.lora_local_dir,
                "lora_weight_name": self.lora_weight_name,
                "adapter_name": self.adapter_name,
                "active_adapters": _list_active_adapters(self._pipeline),
                "startup_metrics": dict(self._startup_metrics),
            },
        }
