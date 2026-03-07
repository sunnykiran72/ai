import os
import io
import time
import hashlib
import threading
import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger("glamify-ai")

class FlorenceRunner:
    """
    Vision-Language model runner (Florence-2).
    Used for auto-captioning, detailed garment description, and object grounded detection.
    """
    
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.type_max_tokens = self._env_int("FLORENCE_TYPE_MAX_TOKENS", 96)
        self.type_num_beams = self._env_int("FLORENCE_TYPE_NUM_BEAMS", 1)
        self.desc_max_tokens = self._env_int("FLORENCE_DESC_MAX_TOKENS", 192)
        self.desc_num_beams = self._env_int("FLORENCE_DESC_NUM_BEAMS", 2)
        self.detailed_max_tokens = self._env_int("FLORENCE_DETAILED_MAX_TOKENS", 512)
        self.detailed_num_beams = self._env_int("FLORENCE_DETAILED_NUM_BEAMS", 3)

        self.cache_enabled = os.getenv("FLORENCE_CACHE_ENABLED", "1") == "1"
        self.cache_ttl_seconds = self._env_float("FLORENCE_CACHE_TTL_SECONDS", 900.0)
        self.cache_max_entries = self._env_int("FLORENCE_CACHE_MAX_ENTRIES", 512)
        self._cache: Dict[str, Tuple[float, str]] = {}
        self._cache_lock = threading.Lock()

        self._model = None
        self._processor = None

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, str(default))
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name, str(default))
        try:
            return float(raw)
        except ValueError:
            return default

    def _build_cache_key(
        self,
        image: Image.Image,
        task_prompt: str,
        text_input: str,
        max_new_tokens: int,
        num_beams: int,
        use_cache_generate: bool,
    ) -> str:
        rgb_image = image.convert("RGB")
        buff = io.BytesIO()
        rgb_image.save(buff, format="PNG")
        digest = hashlib.sha256()
        digest.update(self.model_id.encode("utf-8"))
        digest.update(task_prompt.encode("utf-8"))
        digest.update(text_input.encode("utf-8"))
        digest.update(str(max_new_tokens).encode("utf-8"))
        digest.update(str(num_beams).encode("utf-8"))
        digest.update(str(use_cache_generate).encode("utf-8"))
        digest.update(buff.getvalue())
        return digest.hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.cache_enabled:
            return None
        with self._cache_lock:
            payload = self._cache.get(key)
            if payload is None:
                return None
            expires_at, value = payload
            if time.time() > expires_at:
                self._cache.pop(key, None)
                return None
            return value

    def _cache_set(self, key: str, value: str) -> None:
        if not self.cache_enabled:
            return
        with self._cache_lock:
            if len(self._cache) >= self.cache_max_entries:
                oldest = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
                self._cache.pop(oldest, None)
            self._cache[key] = (time.time() + self.cache_ttl_seconds, value)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    @classmethod
    def _infer_type_from_text(cls, text: str) -> Tuple[str, float, str]:
        normalized = cls._normalize_text(text)
        if not normalized:
            return ("top", 0.40, "empty_caption_fallback")

        keyword_weights = {
            "dress": {
                "dress": 0.45, "gown": 0.45, "one-piece": 0.40, "one piece": 0.40,
                "maxi": 0.35, "midi": 0.35, "kurti": 0.35,
            },
            "bottom": {
                "pant": 0.35, "pants": 0.35, "trouser": 0.35, "jean": 0.35,
                "jeans": 0.35, "skirt": 0.35, "shorts": 0.30, "legging": 0.30,
            },
            "outer": {
                "jacket": 0.40, "coat": 0.40, "blazer": 0.35, "hoodie": 0.35,
                "cardigan": 0.35, "sweater": 0.30, "outerwear": 0.35,
            },
            "top": {
                "shirt": 0.30, "t-shirt": 0.30, "tee": 0.25, "top": 0.25,
                "blouse": 0.30, "sleeve": 0.20, "tank": 0.25, "camisole": 0.25,
                "bra": 0.40, "bralette": 0.40, "brassiere": 0.40, "bikini top": 0.35, "bustier": 0.35,
            },
        }

        scores = {"top": 0.0, "bottom": 0.0, "dress": 0.0, "outer": 0.0}
        for garment_type, terms in keyword_weights.items():
            for term, weight in terms.items():
                if term in normalized:
                    scores[garment_type] += weight

        # Favor dress/outer when their explicit tokens exist because they are often routed incorrectly.
        if scores["dress"] > 0:
            scores["dress"] += 0.10
        if scores["outer"] > 0:
            scores["outer"] += 0.05

        predicted = max(scores.items(), key=lambda kv: kv[1])[0]
        raw_score = scores[predicted]
        if raw_score <= 0:
            return ("top", 0.50, "no_keyword_fallback")

        confidence = min(0.98, 0.55 + raw_score)
        return (predicted, confidence, f"keyword_match:{predicted}")

    def _apply_generation_compat(self) -> None:
        """
        Some transformers versions expect forced_bos/forced_eos fields to exist.
        Florence remote code may not define them on every nested config object.
        """
        if self._model is None:
            return

        maybe_configs = [
            getattr(self._model, "config", None),
            getattr(self._model, "generation_config", None),
        ]
        language_model = getattr(self._model, "language_model", None)
        if language_model is not None:
            maybe_configs.append(getattr(language_model, "config", None))
            maybe_configs.append(getattr(language_model, "generation_config", None))

        for cfg in maybe_configs:
            if cfg is None:
                continue
            if not hasattr(cfg, "forced_bos_token_id"):
                setattr(cfg, "forced_bos_token_id", None)
            if not hasattr(cfg, "forced_eos_token_id"):
                setattr(cfg, "forced_eos_token_id", None)

    def _ensure_loaded(self):
        if self._model is None:
            logger.info(f"Loading Florence-2 from {self.model_id}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                torch_dtype=self.torch_dtype,
                attn_implementation="eager",
            ).to(self.device)
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._apply_generation_compat()

    def run_task(
        self,
        image: Image.Image,
        task_prompt: str,
        text_input: str = "",
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        use_cache_generate: bool = False,
    ) -> str:
        """
        Executes a Florence-2 task (e.g., '<CAPTION>', '<DETAILED_CAPTION>', '<OD>')
        """
        self._ensure_loaded()
        self._apply_generation_compat()

        prompt = task_prompt + text_input
        resolved_max_tokens = max_new_tokens if max_new_tokens is not None else self.detailed_max_tokens
        resolved_num_beams = num_beams if num_beams is not None else self.detailed_num_beams

        cache_key = self._build_cache_key(
            image=image,
            task_prompt=task_prompt,
            text_input=text_input,
            max_new_tokens=resolved_max_tokens,
            num_beams=resolved_num_beams,
            use_cache_generate=use_cache_generate,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=resolved_max_tokens,
            num_beams=resolved_num_beams,
            use_cache=use_cache_generate,
        )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self._processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        value: str
        if isinstance(parsed_answer, dict):
            parsed_value = parsed_answer.get(task_prompt)
            value = str(parsed_value) if parsed_value is not None else str(parsed_answer)
        else:
            value = str(parsed_answer)

        self._cache_set(cache_key, value)
        return value

    def classify_garment_type(self, image: Image.Image, hint_type: Optional[str] = None) -> Dict[str, Any]:
        caption = self.run_task(
            image=image,
            task_prompt="<CAPTION>",
            max_new_tokens=self.type_max_tokens,
            num_beams=self.type_num_beams,
            use_cache_generate=False,
        )
        predicted_type, score, reason = self._infer_type_from_text(caption)
        if hint_type and predicted_type == hint_type:
            score = min(0.99, score + 0.06)

        return {
            "type": predicted_type,
            "score": float(score),
            "reason": reason,
            "caption": caption,
        }

    def describe_garment_short(self, image: Image.Image) -> str:
        return self.run_task(
            image=image,
            task_prompt="<CAPTION>",
            max_new_tokens=self.desc_max_tokens,
            num_beams=self.desc_num_beams,
            use_cache_generate=False,
        )

    def describe_garment(self, image: Image.Image) -> str:
        """
        Returns a rich description of the garment for prompting.
        """
        return self.run_task(
            image=image,
            task_prompt="<DETAILED_CAPTION>",
            max_new_tokens=self.detailed_max_tokens,
            num_beams=self.detailed_num_beams,
            use_cache_generate=False,
        )

    def detect_garments(self, image: Image.Image) -> Dict[str, Any]:
        """
        Grounded object detection for clothes.
        """
        return self.run_task(
            image=image,
            task_prompt="<OD>",
            max_new_tokens=self.type_max_tokens,
            num_beams=self.type_num_beams,
            use_cache_generate=False,
        )

if __name__ == "__main__":
    # Isolated Test
    logging.basicConfig(level=logging.INFO)
    florence = FlorenceRunner()
    print("Florence-2 Runner Initialized.")
