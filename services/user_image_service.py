"""
User image preparation service.

This module provides the UserImageService class that orchestrates user
image preparation workflows.

Responsibilities:
- Validate user images (blur, person detection, face detection)
- Remove background using BiRefNet
- Detect and crop to main person
- Generate user descriptors for identity preservation
"""

from typing import Dict, Optional, Tuple
import io
from PIL import Image

from config import Config
from services.ai_engine import AIEngine


class UserImageService:
    """
    Orchestrates user image preparation workflows.
    
    Responsibilities:
    - Validate user images (blur, person detection, face detection)
    - Remove background using BiRefNet
    - Detect and crop to main person
    - Generate user descriptors for identity preservation
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

        Placeholder implementation while the user-prep pipeline is refactored.
        """
        if upload is None or not hasattr(upload, "read"):
            return {
                "status": "not_implemented",
                "message": "UserImageService expects an uploaded file.",
            }

        try:
            from ai import main as main_mod
        except ModuleNotFoundError:
            import main as main_mod

        payload = await upload.read()
        image = Image.open(io.BytesIO(payload)).convert("RGB")

        candidates, detect_meta = main_mod._user_prep_detect_person_candidates(image)
        if not candidates:
            return {
                "error": "no_person",
                "message": "No person detected in the image.",
                "meta": detect_meta,
            }
        if main_mod._user_prep_has_multiple_prominent_people(candidates):
            return {
                "error": "multiple_people",
                "message": "Multiple prominent people detected.",
                "meta": detect_meta,
            }

        crop, bbox = main_mod._user_prep_crop_main_person(image, candidates)
        focus_score = main_mod._focus_score(crop)

        # Background removal disabled for now (use raw crop)
        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="PNG")
        prepared_bytes = buf.getvalue()
        bg_meta = {"enabled": False, "backend": "none"}
        url = main_mod._upload_or_raise(prepared_bytes)

        # Use MiniCPM for user description when available
        description_raw = ""
        minicpm_runner = getattr(self.engine, "minicpm", None)
        if minicpm_runner is not None:
            try:
                description_raw = str(minicpm_runner.describe_person_and_outfit(crop)).strip()
            except Exception:
                description_raw = ""
        if not description_raw:
            description_raw = main_mod._describe_user_image_for_prepare(crop, description_backend=None)
        prompt_description = main_mod._normalize_user_prepare_api_prompt_description(description_raw)

        return {
            "url": url,
            "promptDescription": prompt_description,
            "focusScore": float(focus_score),
            "meta": {
                "person_bbox": bbox,
                "detect": detect_meta,
                "background": bg_meta,
            },
        }
    
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
