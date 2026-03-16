from __future__ import annotations

import io
import time
from typing import Callable, Dict, List, Optional

from PIL import Image

from modules.wardrobe.extraction.contracts import AnalyzeInputImage, AnalyzeStageTimings


def verify_authorization_or_response(
    authorization: Optional[str],
    *,
    verify_bearer_token: Callable[[Optional[str]], dict],
    build_error_payload: Callable[..., dict],
    multipart_form_response: Callable[..., object],
):
    try:
        return verify_bearer_token(authorization)
    except PermissionError:
        payload = build_error_payload(
            title="Session Expired",
            description="Please log in again and try uploading your item.",
            reason_codes=["UNAUTHORIZED"],
            status_code=401,
            result="REJECTED",
        )
        return multipart_form_response(payload)


def resolve_upload_or_response(
    *,
    file_upload,
    image_upload,
    build_error_payload: Callable[..., dict],
    multipart_form_response: Callable[..., object],
):
    upload = file_upload or image_upload
    if upload is not None:
        return upload

    payload = build_error_payload(
        title="No Image Provided",
        description="Please upload an image to analyze.",
        reason_codes=["INVALID_IMAGE"],
        status_code=400,
    )
    return multipart_form_response(payload)


async def read_input_image_or_response(
    upload,
    *,
    max_file_bytes: int,
    blur_check_enabled: bool,
    blur_min_focus_score: float,
    focus_score_fn: Callable[[Image.Image], float],
    build_error_payload: Callable[..., dict],
    multipart_form_response: Callable[..., object],
    stage_timings: AnalyzeStageTimings,
):
    t_stage = time.time()
    image_bytes = await upload.read()
    stage_timings.read_input_s = time.time() - t_stage

    if len(image_bytes) > max_file_bytes:
        payload = build_error_payload(
            title="File Too Large",
            description="Please upload an image smaller than 3MB.",
            reason_codes=["FILE_TOO_LARGE"],
            status_code=400,
        )
        return multipart_form_response(payload)

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        payload = build_error_payload(
            title="Invalid Image",
            description="Could not process the image. Please upload a clearer photo.",
            reason_codes=["INVALID_IMAGE"],
            status_code=400,
        )
        return multipart_form_response(payload)

    if blur_check_enabled:
        t_stage = time.time()
        focus_score = focus_score_fn(image)
        stage_timings.blur_check_s = time.time() - t_stage
        if focus_score < blur_min_focus_score:
            payload = build_error_payload(
                title="Image Too Blurry",
                description="The garment is not clear enough. Try a brighter, sharper photo.",
                reason_codes=["IMAGE_TOO_BLURRY"],
                status_code=400,
            )
            return multipart_form_response(payload)

    return AnalyzeInputImage(
        upload_name=str(getattr(upload, "filename", "") or ""),
        image_bytes=image_bytes,
        image=image,
    )


def build_success_response(
    *,
    public_item: Optional[Dict[str, object]],
    selected_item: Optional[Dict[str, object]],
    items: List[Dict[str, object]],
    requested_type: Optional[str],
    total_s: float,
    gpu_queue_wait_s: float,
    analyze_stage_timings: Dict[str, object],
    wardrobe_category_from_garment_type: Callable[..., Dict[str, str]],
    build_multipart_parts: Callable[..., Dict[str, object]],
    build_success_payload: Callable[..., Dict[str, object]],
    multipart_form_response: Callable[..., object],
    fetch_image_bytes: Callable[..., bytes],
):
    final_type = str(public_item.get("type", "top")) if public_item else "top"
    final_style = str(public_item.get("style", "")) if public_item else ""
    category_meta = wardrobe_category_from_garment_type(
        final_type,
        style=final_style if final_style else None,
    )

    reason_codes = ["SINGLE_ITEM"]
    extraction_path = str(public_item.get("output_image_source", "")) if public_item else ""
    extraction_meta_public = (public_item or {}).get("extraction") or {}
    if isinstance(extraction_meta_public, dict):
        extraction_timings = extraction_meta_public.get("timings")
        if isinstance(extraction_timings, dict):
            analyze_stage_timings["extraction_pipeline"] = extraction_timings
    if str(extraction_meta_public.get("pipeline") or "") == "flux2_extract":
        reason_codes.append("FLUX2_EXTRACT_PIPELINE")
    elif extraction_meta_public.get("endpoint"):
        reason_codes.append("VTON_ONLY_PIPELINE")
        reason_codes.append("VTON_USED")
    if public_item and public_item.get("type_source") == "uncertain_fullbody_dress_fallback":
        reason_codes.append("UNCERTAIN_FULLBODY_DRESS_FALLBACK")
    if requested_type:
        reason_codes.append("TYPE_FORCED_EXTRACT")

    output_image_url = public_item.get("output_image_url", public_item.get("url")) if public_item else None
    multipart_data = build_multipart_parts(items=[], cloth_url=output_image_url)
    data = {
        "result": "ACCEPTED",
        "title": "Added To Wardrobe",
        "description": "Garment extracted successfully.",
        "reason_codes": reason_codes,
        "selection_required": False,
        "clothing_type": category_meta["style"],
        "category_key": category_meta["category_key"],
        "primary_category_key": category_meta["primary_category_key"],
        "style": category_meta["style"],
        "selected_type": final_type,
        "selected_item": public_item,
        "garmentMetadata": public_item.get("garmentMetadata") if public_item else None,
        "cloth_url": public_item.get("url") if public_item else None,
        "output_image_url": output_image_url,
        "extraction_path": extraction_path,
        "promptDescription": public_item.get("promptDescription") if public_item else None,
        "wardrobe_progress_id": public_item.get("wardrobe_progress_id") if public_item else None,
        "multipart_data": multipart_data,
        "total_garments_found": len(items),
        "latencies": {
            "total": round(float(total_s), 4),
            "gpu_queue_wait": round(float(gpu_queue_wait_s), 4),
            "stages": analyze_stage_timings,
        },
        "processing_time_ms": int(float(total_s) * 1000),
    }
    data["imageUrl"] = data["cloth_url"]
    data["progressId"] = data["wardrobe_progress_id"]

    success_binary_parts = []
    extracted_bytes = bytes((selected_item or {}).get("_extracted_image_bytes") or b"")
    if not extracted_bytes and output_image_url:
        try:
            extracted_bytes = fetch_image_bytes(str(output_image_url), timeout=45)
        except Exception:
            extracted_bytes = b""
    if extracted_bytes:
        success_binary_parts.append({
            "name": "extracted_cloth",
            "filename": "extracted_cloth.png",
            "content_type": "image/png",
            "bytes": extracted_bytes,
        })

    payload = build_success_payload(data=data, status_code=200, message="")
    return multipart_form_response(payload, binary_parts=success_binary_parts)
