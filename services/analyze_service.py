"""
Garment analysis service.

This module provides the AnalyzeService class that orchestrates garment
analysis workflows.

Responsibilities:
- Detect garments in images using YOLO and Florence
- Split multi-garment images (top/bottom separation)
- Extract garment metadata (color, type, descriptors)
- Generate selection previews for user confirmation
- Coordinate extraction pipeline stages
"""

from typing import Dict, List, Optional, Tuple
import io
import inspect
from PIL import Image

from config import AnalyzeConfig
from services.ai_engine import AIEngine
from shared.response_payloads import (
    build_error_payload,
    build_success_payload,
    build_multipart_parts,
    multipart_form_response,
)


class AnalyzeService:
    """
    Orchestrates garment analysis workflows.
    
    Responsibilities:
    - Detect garments in images using YOLO and Florence
    - Split multi-garment images (top/bottom separation)
    - Extract garment metadata (color, type, descriptors)
    - Generate selection previews for user confirmation
    - Coordinate extraction pipeline stages
    """
    
    def __init__(self, engine: AIEngine, config: AnalyzeConfig):
        self.engine = engine
        self.config = config
    
    async def analyze_image(
        self,
        upload,
        garment_type: Optional[str] = None,
        selected_index: Optional[int] = None,
        require_selection: Optional[bool] = None,
        debug: bool = False,
        authorization: Optional[str] = None,
        use_parser_post_extract: Optional[bool] = None,
        useParserPostExtract: Optional[bool] = None,
    ):
        """
        Analyze image and detect garments - delegates to legacy implementation.
        
        This is a temporary bridge to maintain functionality while the refactoring
        is completed. The actual implementation logic remains in main_legacy.py.
        """
        if upload is None or not hasattr(upload, "read"):
            return {
                "status": "not_implemented",
                "message": "AnalyzeService expects an uploaded file.",
            }

        try:
            from ai import main as main_mod
        except ModuleNotFoundError:
            import main as main_mod

        if not inspect.iscoroutinefunction(upload.read):
            return {
                "status": "not_implemented",
                "message": "AnalyzeService expects an async UploadFile.",
            }

        raw_bytes = await upload.read()
        max_file_bytes = int(getattr(main_mod, "ANALYZE_MAX_FILE_BYTES", self.config.max_file_bytes))
        if max_file_bytes > 0 and len(raw_bytes) > max_file_bytes:
            payload = build_error_payload(
                title="File Too Large",
                description="Uploaded image exceeds size limits.",
                reason_codes=["FILE_TOO_LARGE"],
                status_code=400,
            )
            return multipart_form_response(payload)

        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        instances = []
        try:
            instances = self.engine.yolo.detect_instances(image)
        except Exception:
            instances = []

        try:
            crops = self.engine.yolo.get_crops(image, instances)
        except Exception:
            crops = []

        items: List[Dict[str, object]] = []
        for idx, crop in enumerate(crops or []):
            label = crop.get("label") or crop.get("type") or "all"
            bbox = crop.get("bbox")
            item = {
                "rank": idx + 1,
                "garment_id": idx + 1,
                "type": label,
                "garment_type": label,
                "confidence": crop.get("confidence", 0.0),
                "bbox": bbox,
                "image": crop.get("image"),
            }
            items.append(item)

        if not items:
            payload = build_error_payload(
                title="No Garments Detected",
                description="No garments could be detected in the image.",
                reason_codes=["NO_GARMENTS_DETECTED"],
                status_code=400,
            )
            return multipart_form_response(payload)

        selected_item: Optional[Dict[str, object]] = None
        if selected_index is not None:
            try:
                idx = int(selected_index)
                if idx > 0:
                    idx -= 1
                if 0 <= idx < len(items):
                    selected_item = items[idx]
            except Exception:
                selected_item = None

        reason_codes: List[str] = []
        if selected_item is None and garment_type:
            try:
                best = max(
                    items,
                    key=lambda it: main_mod._requested_type_geometry_score(
                        it,
                        garment_type,
                        image.height,
                    ),
                )
                selected_item = best
                reason_codes.append("TYPE_FORCED_VTON")
            except Exception:
                selected_item = None

        require_selection_flag = (
            bool(require_selection)
            if require_selection is not None
            else bool(getattr(main_mod, "ANALYZE_REQUIRE_SELECTION", True))
        )
        if selected_item is None and len(items) > 1 and require_selection_flag:
            item_breakdown = [
                {
                    "rank": int(item.get("rank", 0)),
                    "type": item.get("type"),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                    "bbox": item.get("bbox"),
                }
                for item in items
            ]
            payload = build_error_payload(
                title="Selection Required",
                description="Multiple garments detected. Please select one.",
                reason_codes=["MULTI_ITEM_SELECTION_REQUIRED"],
                status_code=400,
            )
            payload["data"]["item_breakdown"] = item_breakdown
            payload["data"]["total_garments_found"] = len(items)
            payload["data"]["multipart_data"] = build_multipart_parts(items=item_breakdown)

            binary_parts: List[Dict[str, object]] = []
            for item in items:
                img = item.get("image")
                if isinstance(img, Image.Image):
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    binary_parts.append({
                        "name": f"item_{int(item.get('rank', 0))}",
                        "filename": f"item_{int(item.get('rank', 0))}.png",
                        "content_type": "image/png",
                        "bytes": buf.getvalue(),
                    })
            return multipart_form_response(payload, binary_parts=binary_parts)

        if selected_item is None:
            selected_item = items[0]

        reason_codes.extend(["SINGLE_ITEM", "VTON_ONLY_PIPELINE"])

        extracted = None
        cloth_bytes = None
        cloth_url = None
        try:
            extracted = main_mod._run_vton_cloth_only_fallback(
                getattr(upload, "filename", "upload"),
                str(selected_item.get("garment_type") or garment_type or "top"),
                vto_mode=False,
            )
            if isinstance(extracted, dict):
                cloth_url = extracted.get("url") or extracted.get("raw_url")
                cloth_bytes = extracted.get("_processed_image_bytes")
        except Exception:
            extracted = None

        payload = build_success_payload(
            {
                "reason_codes": reason_codes,
                "selected_item": {
                    "rank": int(selected_item.get("rank", 0)),
                    "type": selected_item.get("type"),
                    "confidence": float(selected_item.get("confidence", 0.0) or 0.0),
                    "bbox": selected_item.get("bbox"),
                },
                "item_breakdown": [
                    {
                        "rank": int(selected_item.get("rank", 0)),
                        "type": selected_item.get("type"),
                        "confidence": float(selected_item.get("confidence", 0.0) or 0.0),
                        "bbox": selected_item.get("bbox"),
                    }
                ],
                "multipart_data": build_multipart_parts(
                    items=[
                        {
                            "rank": int(selected_item.get("rank", 0)),
                            "type": selected_item.get("type"),
                            "garment_type": selected_item.get("garment_type"),
                        }
                    ],
                    cloth_url=cloth_url,
                ),
            },
            status_code=200,
        )

        binary_parts: List[Dict[str, object]] = []
        if isinstance(cloth_bytes, (bytes, bytearray)) and len(cloth_bytes) > 0:
            binary_parts.append({
                "name": "extracted_cloth",
                "filename": "extracted_cloth.png",
                "content_type": "image/png",
                "bytes": bytes(cloth_bytes),
            })
        return multipart_form_response(payload, binary_parts=binary_parts)
    
    async def analyze_with_selection(
        self,
        upload,
        selected_index: int,
        garment_type: Optional[str] = None,
        authorization: Optional[str] = None,
    ):
        """
        Complete analysis after user selects a garment - delegates to legacy implementation.
        """
        return await self.analyze_image(
            upload=upload,
            garment_type=garment_type,
            selected_index=selected_index,
            authorization=authorization,
        )
    
    async def _run_detection(self, image: Image.Image, garment_type: Optional[str] = None) -> List[Dict]:
        """Run YOLO and Florence detection."""
        # TODO: Implement detection logic
        # This would delegate to engine.cloth_detector and related detection functions
        return []
    
    async def _extract_garment(
        self,
        image: Image.Image,
        detection: Dict,
        garment_type: Optional[str] = None,
    ) -> Dict[str, object]:
        """Extract and process selected garment."""
        # TODO: Implement extraction logic
        # This would use the extraction pipeline from modules/wardrobe/extraction
        return {}
    
    def _build_selection_response(
        self,
        image: Image.Image,
        items: List[Dict],
    ) -> Dict[str, object]:
        """Build selection_required response with previews."""
        # TODO: Implement selection response building
        # This would generate preview images and metadata for user selection
        return {
            "selection_required": True,
            "items": items,
            "message": "Multiple garments detected. Please select one."
        }

    async def parser_joycaption_analyze(
        self,
        image_url: str,
        garment_type: Optional[str] = None,
        selected_index: Optional[int] = None,
        use_unified_square_split: Optional[bool] = None,
        min_component_area_ratio: Optional[float] = None,
        square_padding_ratio: Optional[float] = None,
        upload_candidate_previews: Optional[bool] = None,
        adaptive_rect_crop: Optional[bool] = None,
        run_flux_garment_only: Optional[bool] = None,
        flux_steps: Optional[int] = None,
        flux_seed: Optional[int] = None,
        flux_extract_only: Optional[bool] = None,
        flux_extract_strict_safety: Optional[bool] = None,
    ):
        """
        Parser-first analysis with JoyCaption - delegates to legacy implementation.
        """
        return {
            "status": "not_implemented",
            "message": "AnalyzeService parser JoyCaption not implemented yet",
        }
