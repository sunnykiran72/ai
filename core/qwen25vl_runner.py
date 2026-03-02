import os
import threading
import logging
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger("glamify-ai")

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen25VLClass
except Exception:
    _Qwen25VLClass = None


class Qwen25VLRunner:
    """
    Qwen2.5-VL descriptor runner used for prompt description generation.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("QWEN25VL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
        desired_device = os.getenv("QWEN25VL_DEVICE", "cpu").strip().lower()
        if desired_device not in {"cpu", "cuda", "auto"}:
            desired_device = "cpu"
        if desired_device == "auto":
            desired_device = "cuda" if torch.cuda.is_available() else "cpu"
        if desired_device == "cuda" and not torch.cuda.is_available():
            desired_device = "cpu"
        self.device = desired_device
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.max_new_tokens = int(os.getenv("QWEN25VL_MAX_NEW_TOKENS", "96"))
        self.garment_max_new_tokens = int(
            os.getenv("QWEN25VL_GARMENT_MAX_NEW_TOKENS", str(self.max_new_tokens))
        )
        self.user_max_new_tokens = int(
            os.getenv("QWEN25VL_USER_MAX_NEW_TOKENS", str(self.max_new_tokens))
        )
        self.num_beams = int(os.getenv("QWEN25VL_NUM_BEAMS", "1"))

        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _load_model(self) -> None:
        logger.info(f"Loading Qwen2.5-VL from {self.model_id}...")
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        if _Qwen25VLClass is not None:
            model = _Qwen25VLClass.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            )
        else:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            )

        model = model.to(self.device)
        self._model = model

    def ensure_ready(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if not self.is_loaded:
                self._load_model()

    def unload(self) -> None:
        """
        Free model weights from memory (used to avoid GPU OOM before Flux2 inference).
        """
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

    def _run_prompt(self, image: Image.Image, instruction: str, max_new_tokens: Optional[int] = None) -> str:
        self.ensure_ready()
        rgb = image.convert("RGB")
        token_limit = int(max_new_tokens or self.max_new_tokens)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": rgb},
                {"type": "text", "text": instruction},
            ],
        }]
        chat_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(text=[chat_text], images=[rgb], return_tensors="pt")
        moved_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved_inputs[key] = value.to(self.device)
            else:
                moved_inputs[key] = value

        with self._infer_lock, torch.inference_mode():
            output_ids = self._model.generate(
                **moved_inputs,
                max_new_tokens=token_limit,
                do_sample=False,
                num_beams=self.num_beams,
            )

        input_len = int(moved_inputs["input_ids"].shape[1]) if "input_ids" in moved_inputs else 0
        if getattr(output_ids, "ndim", 0) == 2 and input_len > 0 and output_ids.shape[1] > input_len:
            decode_ids = output_ids[:, input_len:]
        else:
            decode_ids = output_ids
        text = self._processor.batch_decode(
            decode_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return " ".join(str(text).split()).strip()

    def describe_garment(self, image: Image.Image) -> str:
        return self._run_prompt(
            image=image,
            instruction=(
                "Describe only the target garment for virtual try-on in one concise sentence. "
                "Start with the exact category (dress/top/bottom/outerwear). "
                "Include dominant color, print/pattern, neckline, sleeve type, hem/length, fit/silhouette, and notable trims/details. "
                "Do not mention mannequin, person, background, or uncertainty words."
            ),
            max_new_tokens=self.garment_max_new_tokens,
        )

    def describe_person_and_outfit(self, image: Image.Image) -> str:
        return self._run_prompt(
            image=image,
            instruction=(
                "Describe person identity and current outfit for virtual try-on preservation in two short sentences. "
                "Sentence 1 must focus on identity cues (face, hair, skin tone, body shape, pose, lighting). "
                "Sentence 2 can summarize currently worn outfit."
            ),
            max_new_tokens=self.user_max_new_tokens,
        )

    def classify_garment_type(self, image: Image.Image) -> Optional[str]:
        raw = self._run_prompt(
            image=image,
            instruction=(
                "Classify the main garment into exactly one label from this set: "
                "dress, top, bottom, outer. Return only the one label."
            ),
        )
        token = str(raw or "").strip().lower()
        token = token.split()[0] if token else ""
        if token in {"dress", "top", "bottom", "outer"}:
            return token
        if "outer" in token or "jacket" in token or "coat" in token:
            return "outer"
        if "pant" in token or "trouser" in token or "jean" in token or "skirt" in token:
            return "bottom"
        if "dress" in token or "gown" in token:
            return "dress"
        if "top" in token or "shirt" in token or "blouse" in token:
            return "top"
        return None
