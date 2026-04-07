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
from PIL import Image, ImageOps

from config import Flux2Config
from services.ai_engine import AIEngine
from shared.image_ops import download_image
from shared.azure_storage import storage
from utils import infer_flux2_target_type, normalize_garment_type
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
        guidance_scale: Optional[float] = None,
        mode: Optional[str] = None,
        lora_scale: Optional[float] = None,
        output_max_edge: Optional[int] = None,
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
        source_worn_types = _kwargs.get("source_worn_types")
        resolved_mode = str(mode or "tryon-lora").strip().lower()
        if resolved_mode != "tryon-lora":
            raise ValueError("mode must be: tryon-lora")
        effective_lora_scale = lora_scale
        if effective_lora_scale is None:
            # fal/flux-klein-9b-virtual-tryon-lora recommends scale=1.0.
            effective_lora_scale = 1.0
        prompt_text_for_metadata = ""

        prompt_text = self._build_tryon_lora_prompt(
            user_description=user_prompt_description,
            garment_descriptions=garment_descriptions,
            target_types=garment_types,
            board_mode=board_mode,
            source_worn_types=source_worn_types,
        )
        prompt_text_for_metadata = prompt_text

        flux_runner = getattr(self.engine, "flux2", None)
        if flux_runner is None:
            raise RuntimeError("Flux2 runner is unavailable.")

        flux_result = flux_runner.run_tryon(
            person_image=person,
            board_image=board,
            prompt=prompt_text,
            steps=steps,
            seed=seed,
            guidance_scale=guidance_scale,
            use_lora=True,
            lora_mode="tryon",
            lora_scale=effective_lora_scale,
            output_max_edge=output_max_edge,
        )
        latency = float(flux_result.get("latency") or 0.0)

        image = flux_result.get("image")
        if not isinstance(image, Image.Image):
            raise RuntimeError("Flux2 did not return a valid image.")
        image = self._match_canvas(image, person.size)

        out_buf = io.BytesIO()
        image.save(out_buf, format="PNG")
        output_url = storage.upload_image(out_buf.getvalue(), content_type="image/png")

        total = time.time() - t0
        postprocess = max(0.0, total - latency)
        meta = dict(flux_result.get("metadata") or {})
        # Surface the actual effective try-on LoRA scale in metadata.
        meta["lora_scale"] = float(effective_lora_scale) if effective_lora_scale is not None else 1.0
        if not prompt_text_for_metadata:
            prompt_text_for_metadata = str(meta.get("prompt") or "")
        meta.update(
            {
                "mode": resolved_mode,
                "engine_variant": "flux2_tryon_lora",
                "effective_lora_scale": (
                    float(meta.get("lora_scale")) if meta.get("lora_scale") is not None else None
                ),
                "board_mode": board_mode,
                "garment_count": len(garments),
                "target_types": garment_types,
                "prompt": prompt_text_for_metadata,
                "output_size": [int(image.width), int(image.height)],
                "source_size": [int(person.width), int(person.height)],
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

    def _build_tryon_lora_prompt(
        self,
        *,
        user_description: Optional[str],
        garment_descriptions: List[str],
        target_types: List[str],
        board_mode: str,
        source_worn_types: Optional[List[str]],
    ) -> str:
        del board_mode, source_worn_types

        clean_user = " ".join(str(user_description or "").split()).strip()
        # Avoid duplicate punctuation like ".." when user description already ends with a period.
        person_desc = clean_user.rstrip(" .!?").strip() or "same person"

        item_count = max(len(target_types or []), len(garment_descriptions or []))
        if item_count <= 0:
            item_count = 1

        ordered_kind_priority = {"top": 0, "dress": 1, "bottom": 2, "outer": 3}
        garment_entries: List[Tuple[int, int, str, str]] = []
        for idx in range(item_count):
            raw_type = target_types[idx] if idx < len(target_types or []) else ""
            raw_desc = garment_descriptions[idx] if idx < len(garment_descriptions or []) else ""
            kind = str(normalize_garment_type(raw_type) or "").strip().lower()
            if not kind:
                inferred = infer_flux2_target_type(str(raw_desc or ""))
                kind = str(normalize_garment_type(inferred) or "").strip().lower()
            if not kind:
                kind = "top"
            desc = " ".join(str(raw_desc or "").split()).strip()
            garment_entries.append(
                (
                    int(ordered_kind_priority.get(kind, 99)),
                    idx,
                    kind,
                    desc or kind,
                )
            )

        garment_entries.sort(key=lambda item: (item[0], item[1]))
        garment_items = [item[3] for item in garment_entries]
        kinds_ordered = [item[2] for item in garment_entries]
        kind_counts: Dict[str, int] = {}
        kind_first_desc: Dict[str, str] = {}
        for _prio, _idx, kind, desc in garment_entries:
            kind_counts[kind] = int(kind_counts.get(kind, 0)) + 1
            if kind not in kind_first_desc:
                kind_first_desc[kind] = desc

        # Single-garment prompts use independent templates for top/bottom/outer/dress.
        if len(garment_entries) == 1:
            _kind_priority, _idx, kind, garment_text = garment_entries[0]

            common_tail = (
                "Keep face identity, hair, body proportions, pose, hands, camera framing, background, "
                "and lighting unchanged. The final image is a full body shot."
            )

            if kind == "top":
                return (
                    f"TRYON {person_desc}. "
                    f"Replace the upper garment with {garment_text} as shown in the reference images. "
                    "Render the garment with accurate construction, silhouette, fit, seam and edge placement, drape, "
                    "and length based on the reference garment. "
                    "Preserve the exact garment color, tone, and shading from the reference. "
                    "Keep lower-body clothing unchanged. "
                    + common_tail
                ).strip()

            if kind == "bottom":
                return (
                    f"TRYON {person_desc}. "
                    f"Replace the lower garment with {garment_text} as shown in the reference images. "
                    "Render the garment with accurate construction, silhouette, fit, seam and edge placement, drape, "
                    "and length based on the reference garment. "
                    "Preserve the exact garment color, tone, and shading from the reference. "
                    "Keep upper-body clothing unchanged. "
                    + common_tail
                ).strip()

            if kind == "outer":
                return (
                    f"TRYON {person_desc}. "
                    f"Replace the outer layer with {garment_text} as shown in the reference images. "
                    "Render the garment as the outermost layer with accurate construction, silhouette, fit, seam and edge placement, "
                    "drape, and length based on the reference garment. "
                    "Preserve the exact garment color, tone, and shading from the reference. "
                    "Keep the underlying outfit unchanged where visible. "
                    + common_tail
                ).strip()

            if kind == "dress":
                return (
                    f"TRYON {person_desc}. "
                    f"Replace the entire outfit completely with {garment_text} as shown in the reference images. "
                    "Render one continuous dress across upper and lower clothing regions from top edge to hem with accurate construction, silhouette, fit, seam and edge placement, "
                    "drape, and length based on the reference garment. "
                    "Preserve the exact garment color, tone, and shading from the reference. "
                    "Present a full one-piece dress appearance without separate upper and lower garment splits. "
                    + common_tail
                ).strip()

        # Multi-garment prompts (v1): one active garment per category.
        # If duplicates exist in the same category, fall back to generic composition.
        has_duplicates = any(count > 1 for count in kind_counts.values())
        unique_kinds = sorted(set(kinds_ordered), key=lambda k: int(ordered_kind_priority.get(k, 99)))

        if not has_duplicates:
            kind_set = set(unique_kinds)
            common_multi_tail = (
                "Keep face identity, hair, body proportions, pose, hands, camera framing, background, "
                "and lighting unchanged. The final image is a full body shot."
            )

            # 1) top + bottom
            if kind_set == {"top", "bottom"}:
                top_desc = kind_first_desc.get("top", "top garment")
                bottom_desc = kind_first_desc.get("bottom", "bottom garment")
                return (
                    f"TRYON {person_desc}. "
                    f"Replace the upper garment with {top_desc} and replace the lower garment with {bottom_desc} as shown in the reference images. "
                    "Assign region ownership explicitly: upper-body clothing region is owned by the top garment and lower-body clothing region is owned by the bottom garment. "
                    "Render both garments with accurate construction, silhouette, fit, seam and edge placement, drape, and length based on their references. "
                    "Preserve the exact garment color, tone, and shading for both garments from their references. "
                    + common_multi_tail
                ).strip()

            # 2) dress + (top | outer)
            if kind_set in ({"dress", "top"}, {"dress", "outer"}):
                dress_desc = kind_first_desc.get("dress", "dress garment")
                overlay_kind = "top" if "top" in kind_set else "outer"
                overlay_desc = kind_first_desc.get(overlay_kind, f"{overlay_kind} garment")
                return (
                    f"TRYON {person_desc}. "
                    f"Replace the outfit using {dress_desc} and {overlay_desc} as shown in the reference images. "
                    f"Assign layer ownership explicitly: use the dress as the base garment and apply the {overlay_kind} garment as the upper-region overlay layer. "
                    "Keep the lower dress structure visible where not covered by the overlay layer. "
                    "Render both garments with accurate construction, silhouette, fit, seam and edge placement, drape, and length based on their references. "
                    "Preserve the exact garment color, tone, and shading for both garments from their references. "
                    + common_multi_tail
                ).strip()

            # 3) dress + bottom
            if kind_set == {"dress", "bottom"}:
                dress_desc = kind_first_desc.get("dress", "dress garment")
                bottom_desc = kind_first_desc.get("bottom", "bottom garment")
                return (
                    f"TRYON {person_desc}. "
                    f"Replace the outfit using {dress_desc} and {bottom_desc} as shown in the reference images. "
                    "Assign region ownership explicitly: upper-body clothing region is owned by the dress and lower-body clothing region is owned by the bottom garment. "
                    "Render both garments with accurate construction, silhouette, fit, seam and edge placement, drape, and length based on their references. "
                    "Preserve the exact garment color, tone, and shading for both garments from their references. "
                    + common_multi_tail
                ).strip()

            # 4) dress + (top | outer) + bottom
            if kind_set in ({"dress", "top", "bottom"}, {"dress", "outer", "bottom"}):
                dress_desc = kind_first_desc.get("dress", "dress garment")
                bottom_desc = kind_first_desc.get("bottom", "bottom garment")
                overlay_kind = "top" if "top" in kind_set else "outer"
                overlay_desc = kind_first_desc.get(overlay_kind, f"{overlay_kind} garment")
                return (
                    f"TRYON {person_desc}. "
                    f"Replace the outfit using {dress_desc}, {overlay_desc}, and {bottom_desc} as shown in the reference images. "
                    f"Assign layer and region ownership explicitly: use the dress as base, apply the {overlay_kind} garment as an upper-region overlay layer, and assign the lower-body clothing region to the bottom garment. "
                    "Keep edits confined to their owned regions and layers. "
                    "Render all garments with accurate construction, silhouette, fit, seam and edge placement, drape, and length based on their references. "
                    "Preserve the exact garment color, tone, and shading for all garments from their references. "
                    + common_multi_tail
                ).strip()

        # Multi-garment fallback for unsupported/ambiguous combinations.
        garments_text = ", ".join(garment_items)
        return (
            f"TRYON {person_desc}. "
            f"Replace the entire outfit with {garments_text} as shown in the reference images. "
            "Render all garments with accurate construction, silhouette, fit, seam and edge placement, drape, and length "
            "based on the reference garments. Preserve the exact garment color, tone, and shading from the references. "
            "Keep face identity, hair, body proportions, pose, hands, camera framing, background, and lighting unchanged. "
            "The final image is a full body shot."
        ).strip()

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

    @staticmethod
    def _match_canvas(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        if not isinstance(image, Image.Image):
            return image
        tw, th = [int(v) for v in target_size]
        if tw <= 0 or th <= 0 or image.size == (tw, th):
            return image
        fitted = ImageOps.contain(image.convert("RGBA"), (tw, th), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (tw, th), (255, 255, 255, 255))
        ox = max(0, (tw - fitted.width) // 2)
        oy = max(0, (th - fitted.height) // 2)
        canvas.paste(fitted, (ox, oy), fitted.getchannel("A"))
        return canvas.convert("RGB")

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

            def _nested_get(obj, *path):
                current = obj
                for key in path:
                    current = _safe_get(current, key)
                    if current is None:
                        return None
                return current

            def _infer_target_type(product, prompt_desc: str) -> str:
                candidates = [
                    _safe_get(product, "targetType"),
                    _nested_get(product, "garmentMetadata", "classification", "target_type"),
                    _nested_get(product, "garmentMetadata", "classification", "backend_target_type"),
                    _nested_get(product, "garmentMetadata", "details", "category"),
                    _nested_get(product, "garmentMetadata", "details", "type"),
                    _nested_get(product, "garmentMetadata", "prompt", "prompt_description"),
                    _nested_get(product, "garmentMetadata", "prompt", "base_garment_prompt"),
                ]
                for value in candidates:
                    kind = normalize_garment_type(str(value or ""))
                    if kind:
                        return kind
                inferred = infer_flux2_target_type(prompt_desc)
                return normalize_garment_type(inferred) or ""

            for idx, product in enumerate(products):
                image_url = _safe_get(product, "image")
                prompt_desc = _safe_get(product, "promptDescription")
                target_type = _infer_target_type(product, str(prompt_desc or ""))
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
