"""
User image preparation service.

This module provides the UserImageService class that orchestrates user
image preparation workflows.

Responsibilities:
- Run GroundingDINO one-shot eligibility gate
- Run MiniCPM verification when detection passes
- Upload prepared image and return prompt description
"""

from typing import Dict, Optional, Tuple

from PIL import Image

from config import Config
from services.ai_engine import AIEngine
from utils.user_preparation import prepare_user_image_pipeline


class UserImageService:
    """
    Orchestrates user image preparation workflows.
    
    Responsibilities:
    - Run GroundingDINO one-shot eligibility gate
    - Run MiniCPM verification after detection passes
    - Upload prepared image and return prompt description
    """
    
    def __init__(self, engine: AIEngine, config: Config):
        self.engine = engine
        self.config = config
    
    async def prepare_user_image(
        self,
        upload,
        authorization: Optional[str] = None,
        resize_method: Optional[str] = None,
        output_max_edge: Optional[int] = None,
    ):
        """
        Prepare user image.

        Prepare user image via the modular validation pipeline.
        """
        try:
            from ai import main as main_mod
        except ModuleNotFoundError:
            import main as main_mod

        minicpm_runner = getattr(self.engine, "minicpm", None)
        grounding_dino = getattr(self.engine, "grounding_dino", None)
        realesrgan = getattr(self.engine, "realesrgan", None)
        gfpgan = getattr(self.engine, "gfpgan", None)
        resolved_resize_method = resize_method or self.config.app.user_prep_resize_method
        resolved_output_max_edge = max(
            512,
            int(output_max_edge) if output_max_edge is not None else int(self.config.app.user_prep_target_height),
        )

        def _grounding_detector_fn(image, prompts):
            if grounding_dino is None:
                return None
            return grounding_dino.detect(image, prompts=list(prompts or []))

        def _description_fn(image):
            if minicpm_runner is None:
                return ""
            try:
                return str(minicpm_runner.describe_person_and_outfit(image)).strip()
            except Exception:
                return ""

        def _verification_fn(image, prompt):
            if minicpm_runner is None:
                return ""
            try:
                return str(minicpm_runner.describe_person_and_outfit(image, prompt_override=prompt)).strip()
            except Exception:
                return ""

        prepared_image_fn = None
        if realesrgan is not None and bool(self.config.app.user_prep_upscale_enabled):
            def _prepared_image_fn(image):
                keep_long_edge_min = resolved_output_max_edge
                target_long_edge = resolved_output_max_edge
                if max(image.size) >= max(keep_long_edge_min, target_long_edge):
                    return image
                result = image
                passes = 0
                while max(result.size) < target_long_edge and passes < 2:
                    candidate = realesrgan.upscale(result, target_max_edge=None)
                    if not isinstance(candidate, Image.Image) or candidate.size == result.size:
                        break
                    result = candidate.convert("RGB")
                    passes += 1
                if passes > 0 and bool(self.config.app.user_prep_face_enhance_enabled):
                    if gfpgan is None or not gfpgan.is_available:
                        raise RuntimeError("GFPGAN face enhancement requested but unavailable.")
                    face_weight = float(self.config.app.user_prep_face_enhance_weight)
                    enhanced = gfpgan.enhance(result, weight=face_weight)
                    if not isinstance(enhanced, Image.Image):
                        raise RuntimeError("GFPGAN did not return a valid image.")
                    result = enhanced.convert("RGB")
                return result
            prepared_image_fn = _prepared_image_fn

        async def _run_prepare(method: str):
            return await prepare_user_image_pipeline(
                upload,
                grounding_detector_fn=_grounding_detector_fn,
                verifier_fn=_verification_fn,
                description_fn=_description_fn,
                fallback_description_fn=lambda image: main_mod._describe_user_image_for_prepare(image, description_backend=None),
                prepared_image_fn=prepared_image_fn,
                upload_fn=main_mod._upload_or_raise,
                min_input_height=int(self.config.app.user_prep_min_input_height),
                target_height=resolved_output_max_edge,
                keep_long_edge_min=resolved_output_max_edge,
                output_max_long_edge=resolved_output_max_edge,
                output_max_bytes=int(self.config.app.user_prep_output_max_bytes),
                jpeg_quality=int(self.config.app.user_prep_jpeg_quality),
                jpeg_min_quality=int(self.config.app.user_prep_jpeg_min_quality),
                resize_method=str(method),
                blur_check_enabled=bool(self.config.analyze.blur_check_enabled),
                blur_min_focus_score=float(self.config.analyze.blur_min_focus_score),
                blur_focus_max_edge=int(self.config.analyze.blur_focus_max_edge),
                verification_required=True,
            )

        result = await _run_prepare(str(resolved_resize_method))
        if isinstance(result, dict) and result.get("error"):
            method_name = str(resolved_resize_method or "").strip().lower().replace("-", "_")
            message = str(result.get("message") or "").lower()
            fallback_needed = (
                method_name in {"pyvips", "libvips", "vips"}
                and any(token in message for token in {"pyvips", "libvips", "vipsthumbnail"})
            )
            if fallback_needed:
                result = await _run_prepare("pillow_lanczos")
        return result
    
    async def _validate_image_quality(self, image: Image.Image) -> Dict[str, object]:
        """Validate image quality (blur, resolution, etc.)."""
        # TODO: Implement image quality validation
        # This would include blur detection, resolution checks, etc.
        return {
            "passed": True,
            "blur_score": 25.0,  # Placeholder
            "resolution_ok": True,
            "message": "Image quality validation passed"
        }
    
    async def _detect_and_validate_person(self, image: Image.Image) -> Dict[str, object]:
        """Detect and validate person in image."""
        # TODO: Implement person detection using engine.person_detector
        # This would validate that exactly one person is present
        return {
            "valid": True,
            "person_count": 1,
            "main_person_bbox": [0, 0, image.width, image.height],  # Placeholder
            "message": "Person detection passed"
        }
    
    async def _remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from user image."""
        # TODO: Implement background removal using BiRefNet or other methods
        # For now, return the original image
        return image
    
    async def _generate_user_descriptor(self, image: Image.Image) -> str:
        """Generate user descriptor for identity preservation."""
        # TODO: Implement user descriptor generation using configured backend
        return "user descriptor placeholder"
    
    async def _upload_image(self, image_bytes: bytes, container: Optional[str] = None) -> Optional[str]:
        """Upload processed image to storage."""
        # TODO: Implement image upload to Azure storage
        # This would use the storage utility from shared module
        return None
