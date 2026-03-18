"""
Virtual try-on service.

This module provides the TryonService class that orchestrates virtual
try-on workflows.

Responsibilities:
- Prepare user and garment images for Flux2
- Generate descriptive prompts for garments and users
- Execute Flux2 generation with color/detail locking
- Handle multi-pass refinement (dress, region lock, Qwen)
- Apply color guard and quality scoring
"""

from typing import Dict, List, Optional, Tuple
import io
import time
import logging
from PIL import Image

from config import Flux2Config
from services.ai_engine import AIEngine
from shared.image_ops import download_image
from shared.azure_storage import storage
from utils import build_tryon_prompt_v2
from modules.vto.board_builder import BoardBuilder

logger = logging.getLogger("glamify-ai")


class TryonService:
    """
    Orchestrates virtual try-on workflows.
    
    Responsibilities:
    - Prepare user and garment images for Flux2
    - Generate descriptive prompts for garments and users
    - Execute Flux2 generation with color/detail locking
    - Handle multi-pass refinement (dress, region lock, Qwen)
    - Apply color guard and quality scoring
    """
    
    def __init__(self, engine: AIEngine, config: Flux2Config):
        self.engine = engine
        self.config = config
    
    async def try_on(
        self,
        user_image=None,
        garment_images=None,
        products=None,
        user_image_url: Optional[str] = None,
        user_prompt_description: Optional[str] = None,
        garment_image_url: Optional[str] = None,
        garment_type: Optional[str] = None,
        prompt_description: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        **_kwargs,
    ):
        """
        Perform virtual try-on with Flux2 + LoRA.
        """
        t0 = time.time()
        steps = int(steps) if steps is not None else 20
        seed = int(seed) if seed is not None else None

        person = self._resolve_person_image(user_image, user_image_url)
        garments, garment_descriptions, garment_types = self._resolve_products(
            products=products,
            garment_images=garment_images,
            garment_image_url=garment_image_url,
            garment_type=garment_type,
            prompt_description=prompt_description,
        )
        board, board_mode = self._build_board(garments)

        prompt_text = build_tryon_prompt_v2(
            user_description=str(user_prompt_description or ""),
            garment_descriptions=garment_descriptions,
            target_types=garment_types,
            board_mode=board_mode,
        )

        flux_runner = getattr(self.engine, "flux2", None)
        if flux_runner is None:
            raise RuntimeError("Flux2 runner is unavailable.")

        flux_result = flux_runner.run_tryon(
            person_image=person,
            board_image=board,
            prompt=prompt_text,
            steps=steps,
            seed=seed,
            use_lora=True,
        )
        latency = float(flux_result.get("latency") or 0.0)

        image = flux_result.get("image")
        if not isinstance(image, Image.Image):
            raise RuntimeError("Flux2 did not return a valid image.")

        out_buf = io.BytesIO()
        image.save(out_buf, format="PNG")
        output_url = storage.upload_image(out_buf.getvalue(), content_type="image/png")

        total = time.time() - t0
        postprocess = max(0.0, total - latency)
        meta = dict(flux_result.get("metadata") or {})
        meta.update(
            {
                "board_mode": board_mode,
                "garment_count": len(garments),
                "target_types": garment_types,
                "prompt": prompt_text,
            }
        )

        return {
            "output_url": output_url,
            "latency": latency,
            "steps": steps,
            "seed": seed if seed is not None else meta.get("seed"),
            "metadata": meta,
            "timings": {
                "total": total,
                "generation": latency,
                "postprocess": postprocess,
            },
        }

    async def try_on_legacy_flux(self, request):
        """
        Perform legacy Flux try-on (placeholder).
        """
        return await self.try_on(
            user_image_url=getattr(request, "user_image_url", None),
            garment_image_url=getattr(request, "garment_image_url", None),
            garment_type=getattr(request, "garment_type", None),
            prompt_description=getattr(request, "prompt", None),
            negative_prompt=getattr(request, "negative_prompt", None),
            steps=getattr(request, "steps", None),
            seed=getattr(request, "seed", None),
        )
    
    async def _generate_garment_descriptor(
        self,
        garment_image: Image.Image,
        garment_metadata: Optional[Dict] = None,
        backend: Optional[str] = None,
    ) -> str:
        """Generate descriptive prompt for garment."""
        # TODO: Implement garment descriptor generation
        # This would use the configured backend (florence, minicpm, etc.)
        return "garment descriptor placeholder"
    
    async def _generate_user_descriptor(
        self,
        user_image: Image.Image,
        backend: Optional[str] = None,
    ) -> str:
        """Generate descriptive prompt for user."""
        # TODO: Implement user descriptor generation
        return "user descriptor placeholder"
    
    def _should_apply_second_pass(
        self,
        garment_type: str,
        generation_result: Dict,
    ) -> bool:
        """Determine if second pass refinement should be applied."""
        # TODO: Implement second pass logic based on garment type and quality
        return False
    
    async def _apply_second_pass(
        self,
        base_result: Dict,
        garment_type: str,
        refinement_type: str,
    ) -> Dict:
        """Apply second pass refinement (dress, region lock, or Qwen)."""
        # TODO: Implement second pass refinement logic
        return base_result
    
    async def _apply_color_guard(
        self,
        generation_result: Dict,
        target_colors: List[str],
    ) -> Dict:
        """Apply color guard to ensure color fidelity."""
        # TODO: Implement color guard logic
        return generation_result

    def _resolve_person_image(self, image, image_url: Optional[str]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if image_url:
            return download_image(str(image_url), preserve_alpha=True)
        raise ValueError("Missing user image.")

    def _resolve_garment_images(
        self,
        images,
        image_url: Optional[str],
    ) -> List[Image.Image]:
        resolved: List[Image.Image] = []
        if images:
            if isinstance(images, list):
                resolved = [img for img in images if isinstance(img, Image.Image)]
            elif isinstance(images, Image.Image):
                resolved = [images]
        if not resolved and image_url:
            resolved = [download_image(str(image_url), preserve_alpha=True)]
        if not resolved:
            raise ValueError("Missing garment image.")
        return resolved

    def _build_board(self, garments: List[Image.Image]) -> Tuple[Image.Image, str]:
        builder = getattr(self.engine, "board_builder", None)
        if builder is None:
            builder = BoardBuilder()
        if len(garments) == 1:
            return garments[0], "single"
        board = builder.build_board(garments)
        return board, "collage"

    def _resolve_products(
        self,
        *,
        products,
        garment_images,
        garment_image_url: Optional[str],
        garment_type: Optional[str],
        prompt_description: Optional[str],
    ) -> Tuple[List[Image.Image], List[str], List[str]]:
        if products:
            images: List[Image.Image] = []
            descriptions: List[str] = []
            target_types: List[str] = []

            def _safe_get(obj, key: str):
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)

            for idx, product in enumerate(products):
                image_url = _safe_get(product, "image")
                prompt_desc = _safe_get(product, "promptDescription")
                target_type = _safe_get(product, "targetType") or ""
                if not image_url:
                    raise ValueError(f"products[{idx}].image is required")
                if not prompt_desc or not str(prompt_desc).strip():
                    raise ValueError(f"products[{idx}].promptDescription is required")
                images.append(download_image(str(image_url), preserve_alpha=True))
                descriptions.append(" ".join(str(prompt_desc).split()).strip())
                target_types.append(str(target_type or "").strip().lower())
            return images, descriptions, target_types

        garments = self._resolve_garment_images(garment_images, garment_image_url)
        base_desc = str(prompt_description or "").strip()
        if not base_desc:
            base_desc = f"{garment_type} garment" if garment_type else "garment"
        return garments, [base_desc], [str(garment_type or "").strip().lower()]
