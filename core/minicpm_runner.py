import logging
import os
import threading
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("glamify-ai")


class MiniCPMVRunner:
    """
    MiniCPM-V descriptor runner for garment and person prompt descriptions.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("MINICPM_MODEL_ID", "openbmb/MiniCPM-V-4_5")
        desired_device = os.getenv("MINICPM_DEVICE", "auto").strip().lower()
        if desired_device not in {"cpu", "cuda", "auto"}:
            desired_device = "auto"
        if desired_device == "auto":
            desired_device = "cuda" if torch.cuda.is_available() else "cpu"
        if desired_device == "cuda" and not torch.cuda.is_available():
            desired_device = "cpu"
        self.device = desired_device

        dtype_name = os.getenv("MINICPM_DTYPE", "bf16").strip().lower()
        if self.device == "cpu":
            self.torch_dtype = torch.float32
        elif dtype_name in {"fp16", "float16"}:
            self.torch_dtype = torch.float16
        elif dtype_name in {"fp32", "float32"}:
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.bfloat16

        self.max_new_tokens = max(64, int(os.getenv("MINICPM_MAX_NEW_TOKENS", "256")))
        self.garment_max_new_tokens = max(
            48,
            int(os.getenv("MINICPM_GARMENT_MAX_NEW_TOKENS", str(self.max_new_tokens))),
        )
        self.user_max_new_tokens = max(
            64,
            int(os.getenv("MINICPM_USER_MAX_NEW_TOKENS", str(self.max_new_tokens))),
        )
        # MiniCPM-V 4.5 officially uses sdpa in recent transformers runtimes.
        self.attn_implementation = os.getenv("MINICPM_ATTN_IMPLEMENTATION", "sdpa").strip().lower()
        if self.attn_implementation not in {"sdpa", "eager", "flash_attention_2"}:
            self.attn_implementation = "sdpa"

        self._model: Optional[Any] = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _load_model(self) -> None:
        logger.info(f"Loading MiniCPM-V from {self.model_id}...")
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": self.attn_implementation,
        }
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = self.torch_dtype

        model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        if self.device == "cuda":
            model = model.to(self.device)
        model = model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = model
        self._tokenizer = tokenizer

    def ensure_ready(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if not self.is_loaded:
                self._load_model()

    def unload(self) -> None:
        with self._load_lock:
            if self._model is not None:
                try:
                    del self._model
                except Exception:
                    pass
            self._model = None
            self._tokenizer = None
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return " ".join(str(text or "").split()).strip()

    def _chat(self, rgb: Image.Image, instruction: str, max_new_tokens: int) -> str:
        msgs = [{"role": "user", "content": [rgb, instruction]}]

        # Remote-code chat signatures vary slightly across MiniCPM releases.
        attempts = [
            ({"msgs": msgs, "tokenizer": self._tokenizer, "max_new_tokens": int(max_new_tokens), "enable_thinking": False}, False),
            ({"msgs": msgs, "tokenizer": self._tokenizer, "sampling": False, "temperature": 0.0, "max_new_tokens": int(max_new_tokens), "enable_thinking": False}, False),
            ({"msgs": msgs, "tokenizer": self._tokenizer, "sampling": False, "max_new_tokens": int(max_new_tokens)}, False),
            ({"msgs": msgs, "tokenizer": self._tokenizer, "max_new_tokens": int(max_new_tokens)}, False),
            ({"image": None, "msgs": msgs, "tokenizer": self._tokenizer, "sampling": False, "temperature": 0.0, "max_new_tokens": int(max_new_tokens)}, True),
            ({"image": None, "msgs": msgs, "tokenizer": self._tokenizer, "sampling": False, "max_new_tokens": int(max_new_tokens)}, True),
            ({"image": None, "msgs": msgs, "tokenizer": self._tokenizer, "max_new_tokens": int(max_new_tokens)}, True),
            ({"image": None, "msgs": msgs, "tokenizer": self._tokenizer}, True),
        ]
        last_err: Optional[Exception] = None

        for kwargs, _legacy_image_arg in attempts:
            try:
                response = self._model.chat(**kwargs)
                if isinstance(response, (list, tuple)) and response:
                    return self._normalize_text(response[0])
                return self._normalize_text(response)
            except TypeError as err:
                last_err = err
                continue
            except Exception as err:
                last_err = err
                break

        raise RuntimeError(f"MiniCPM chat failed: {last_err}")

    def _run_prompt(self, image: Image.Image, instruction: str, max_new_tokens: int) -> str:
        self.ensure_ready()
        rgb = image.convert("RGB")
        with self._infer_lock, torch.inference_mode():
            return self._chat(
                rgb=rgb,
                instruction=str(instruction).strip(),
                max_new_tokens=int(max_new_tokens),
            )

    def describe_garment(self, image: Image.Image, prompt_override: Optional[str] = None) -> str:
        from config.prompts import get_minicpm_garment_prompt
        
        return self._run_prompt(
            image=image,
            instruction=(
                str(prompt_override).strip()
                if str(prompt_override or "").strip()
                else get_minicpm_garment_prompt()
            ),
            max_new_tokens=self.garment_max_new_tokens,
        )

    def describe_person_and_outfit(self, image: Image.Image, prompt_override: Optional[str] = None) -> str:
        from config.prompts import get_minicpm_person_outfit_prompt
        
        return self._run_prompt(
            image=image,
            instruction=(
                str(prompt_override).strip()
                if str(prompt_override or "").strip()
                else get_minicpm_person_outfit_prompt()
            ),
            max_new_tokens=self.user_max_new_tokens,
        )
