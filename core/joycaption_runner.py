import logging
import os
import threading
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

logger = logging.getLogger("glamify-ai")


class JoyCaptionRunner:
    """
    JoyCaption (LLaVA-based) runner for garment-only caption generation.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv(
            "JOYCAPTION_MODEL_ID",
            "fancyfeast/llama-joycaption-alpha-two-hf-llava",
        )
        desired_device = os.getenv("JOYCAPTION_DEVICE", "auto").strip().lower()
        if desired_device not in {"cpu", "cuda", "auto"}:
            desired_device = "auto"
        if desired_device == "auto":
            desired_device = "cuda" if torch.cuda.is_available() else "cpu"
        if desired_device == "cuda" and not torch.cuda.is_available():
            desired_device = "cpu"
        self.device = desired_device
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.max_new_tokens = max(64, int(os.getenv("JOYCAPTION_MAX_NEW_TOKENS", "220")))
        self.instruction = os.getenv(
            "JOYCAPTION_GARMENT_INSTRUCTION",
            (
                "Return ONLY one line, no intro text. "
                "Format exactly: TYPE: ...; COLOR_PRIMARY: ...; COLOR_SECONDARY: ...; NECKLINE: ...; "
                "SLEEVES: ...; BODICE_CUT: ...; SILHOUETTE: ...; LENGTH_HEM: ...; FABRIC_TEXTURE: ...; "
                "EMBELLISHMENTS: ...; SPECIAL_DETAILS: ... . "
                "Focus only garment in image. Never mention person/mannequin/background. "
                "Do not include jewelry or external accessories (for example choker, necklace, earrings, bag, shoes). "
                "Use concrete fashion terms."
            ),
        ).strip()

        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _load_model(self) -> None:
        logger.info(f"Loading JoyCaption from {self.model_id}...")
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self._model = model.to(self.device)

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
            self._processor = None
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _clean_generated_text(self, text: str) -> str:
        t = " ".join(str(text or "").split()).strip()
        if not t:
            return ""
        low = t.lower()
        idx = low.rfind("assistant")
        if idx != -1:
            t = t[idx + len("assistant") :].strip(" :")
        parts = [p.strip() for p in t.split(";") if p.strip()]
        if not parts:
            return t
        accessory_terms = (
            "choker",
            "necklace",
            "earring",
            "earrings",
            "bracelet",
            "ring",
            "bag",
            "handbag",
            "purse",
            "shoe",
            "shoes",
            "sandal",
            "heels",
        )
        filtered = []
        for part in parts:
            low_part = part.lower()
            if any(term in low_part for term in accessory_terms):
                continue
            filtered.append(part)
        return "; ".join(filtered).strip()

    def describe_garment(self, image: Image.Image, instruction_override: Optional[str] = None) -> str:
        self.ensure_ready()
        rgb = image.convert("RGB")
        instruction = str(instruction_override or self.instruction).strip() or self.instruction

        messages = [{"role": "user", "content": "<image>\n" + instruction}]
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(text=prompt, images=rgb, return_tensors="pt")
        moved_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved_inputs[key] = value.to(self.device)
            else:
                moved_inputs[key] = value

        with self._infer_lock, torch.inference_mode():
            output_ids = self._model.generate(
                **moved_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        decoded = self._processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return self._clean_generated_text(decoded)
