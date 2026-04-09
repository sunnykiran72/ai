import logging
import os
import threading
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("glamify-ai")


def _env_value(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


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

        self.max_new_tokens = max(
            64,
            int(_env_value("MINICPM_MAX_NEW_TOKENS", "MINICPM_SERVICE_MAX_NEW_TOKENS", default="192")),
        )
        self.garment_max_new_tokens = max(
            96,
            int(_env_value("MINICPM_GARMENT_MAX_NEW_TOKENS", "MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS", default="192")),
        )
        self.garment_min_words = max(
            12,
            int(_env_value("MINICPM_GARMENT_MIN_WORDS", "MINICPM_SERVICE_GARMENT_MIN_WORDS", default="40")),
        )
        self.user_max_new_tokens = max(
            64,
            int(_env_value("MINICPM_USER_MAX_NEW_TOKENS", "MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS", default="200")),
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
        try:
            from transformers.modeling_utils import PreTrainedModel
            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                PreTrainedModel.all_tied_weights_keys = {}
        except Exception:
            pass
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": self.attn_implementation,
        }
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = self.torch_dtype

        model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        if self.device == "cuda":
            model = model.to(self.device)
        # PreTrainedModel is patched to expose all_tied_weights_keys via property.
        model = model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = model
        self._tokenizer = tokenizer

    @staticmethod
    def _ensure_tied_keys(_root: Any) -> None:
        return

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

    @staticmethod
    def _word_count(text: str) -> int:
        return len([part for part in str(text or "").split() if part.strip()])

    def _chat(self, rgb: Image.Image, instruction: str, max_new_tokens: int) -> str:
        msgs = [{"role": "user", "content": [rgb, instruction]}]

        # Remote-code chat signatures vary slightly across MiniCPM releases.
        # Prefer deterministic decoding (sampling=False) to reduce prompt volatility.
        attempts = [
            ({"msgs": msgs, "tokenizer": self._tokenizer, "sampling": False, "temperature": 0.0, "max_new_tokens": int(max_new_tokens), "enable_thinking": False}, False),
            ({"msgs": msgs, "tokenizer": self._tokenizer, "sampling": False, "max_new_tokens": int(max_new_tokens), "enable_thinking": False}, False),
            ({"msgs": msgs, "tokenizer": self._tokenizer, "sampling": False, "temperature": 0.0, "max_new_tokens": int(max_new_tokens)}, False),
            ({"msgs": msgs, "tokenizer": self._tokenizer, "sampling": False, "max_new_tokens": int(max_new_tokens)}, False),
            ({"msgs": msgs, "tokenizer": self._tokenizer, "max_new_tokens": int(max_new_tokens), "enable_thinking": False}, False),
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
            except AttributeError as err:
                last_err = err
                continue
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

    def describe_garment(
        self,
        image: Image.Image,
        garment_type: Optional[str] = None,
        prompt_override: Optional[str] = None,
    ) -> str:
        from config.prompts import get_minicpm_garment_prompt

        instruction = (
            str(prompt_override).strip()
            if str(prompt_override or "").strip()
            else get_minicpm_garment_prompt(garment_type)
        )
        response = self._run_prompt(
            image=image,
            instruction=instruction,
            max_new_tokens=self.garment_max_new_tokens,
        )
        if self._word_count(response) >= self.garment_min_words:
            return response

        retry_instruction = (
            f"{instruction} "
            f"Your previous answer was too short. Rewrite it as one compact garment-construction paragraph of {self.garment_min_words} to 90 words and 3 to 5 complete sentences. "
            "Keep only visible garment facts. Prioritize the garment category, key edges, strap or sleeve layout, panel structure, closure, and hem or visible length. "
            "Do not add filler, repeated phrases, fit opinions, styling language, or inferred details."
        ).strip()
        retry_response = self._run_prompt(
            image=image,
            instruction=retry_instruction,
            max_new_tokens=self.garment_max_new_tokens,
        )
        if self._word_count(retry_response) > self._word_count(response):
            return retry_response

        return response

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
