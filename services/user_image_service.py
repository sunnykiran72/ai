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

        return await prepare_user_image_pipeline(
            upload,
            grounding_detector_fn=_grounding_detector_fn,
            verifier_fn=_verification_fn,
            description_fn=_description_fn,
            fallback_description_fn=lambda image: main_mod._describe_user_image_for_prepare(image, description_backend=None),
            upload_fn=main_mod._upload_or_raise,
            blur_check_enabled=bool(self.config.analyze.blur_check_enabled),
            blur_min_focus_score=float(self.config.analyze.blur_min_focus_score),
            blur_focus_max_edge=int(self.config.analyze.blur_focus_max_edge),
            verification_required=True,
        )
    
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
