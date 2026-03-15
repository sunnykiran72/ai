from __future__ import annotations

import io
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image


def run_selected_item_extraction_or_response(
    *,
    selected_item: Optional[Dict[str, object]],
    requested_type: Optional[str],
    direct_requested_type_mode: bool,
    full_image: Image.Image,
    all_items_count: int,
    stage_timings,
    engine,
    logger,
    analyze_extract_cloth: bool,
    analyze_prompt_from_extracted: bool,
    analyze_require_extracted_prompt: bool,
    analyze_caption_mode: str,
    flux2_single_garment_extract_default_steps: int,
    flux2_single_garment_extract_default_seed: int,
    normalize_garment_type: Callable[[Optional[str]], Optional[str]],
    prepare_extract_source_image: Callable[..., object],
    build_error_payload: Callable[..., Dict[str, object]],
    multipart_form_response: Callable[..., object],
    run_flux2_cloth_only_extract: Callable[..., Dict[str, object]],
    descriptor_is_weak: Callable[[str], bool],
    caption_non_garment_signal: Callable[[str], bool],
    download_image: Callable[[str], Image.Image],
    flatten_rgba_on_white: Callable[[Image.Image], Image.Image],
    sanitize_garment_description: Callable[[str], str],
    infer_style_from_text: Callable[..., Optional[str]],
    wardrobe_category_from_garment_type: Callable[..., Dict[str, str]],
) -> Tuple[Optional[Dict[str, object]], Optional[object]]:
    if not selected_item or not analyze_extract_cloth:
        return selected_item, None

    forced_type = requested_type if requested_type in {"top", "bottom", "dress", "outer"} else None
    selected_type = forced_type or normalize_garment_type(str(selected_item.get("type")))
    if not selected_type:
        payload = build_error_payload(
            title="Type Required",
            description="Could not determine garment type. Please provide type as top, bottom, dress, or outer.",
            reason_codes=["INVALID_SELECTION_TYPE"],
            status_code=400,
        )
        return None, multipart_form_response(payload)

    original_selected_type = normalize_garment_type(str(selected_item.get("type")))
    if (
        forced_type
        and original_selected_type not in {None, "", forced_type}
        and original_selected_type in {"top", "bottom", "dress", "outer"}
    ):
        selected_item["requested_type_mismatch"] = {
            "requested_type": forced_type,
            "detected_type": original_selected_type,
            "detector_label": str(selected_item.get("detector_label") or ""),
        }
        payload = build_error_payload(
            title="Requested Garment Not Found",
            description=f"Could not find a clear {forced_type} in this image. The detected garment looks like {original_selected_type}.",
            reason_codes=["REQUESTED_TYPE_NOT_FOUND"],
            status_code=400,
        )
        payload.setdefault("data", {})["selected_item"] = {
            "type": original_selected_type,
            "detector_label": str(selected_item.get("detector_label") or ""),
            "bbox": list(selected_item.get("bbox") or []),
        }
        return None, multipart_form_response(payload)

    if forced_type and str(selected_item.get("type")) != forced_type:
        selected_item["type_original"] = selected_item.get("type")
        selected_item["type"] = forced_type
        selected_item["garment_type"] = forced_type
        selected_item["type_source"] = "requested_type"

    try:
        if direct_requested_type_mode:
            selected_item["bbox_geometry_source"] = "requested_type_direct_full_image"
            selected_item["extract_crop_bbox"] = [0, 0, full_image.width, full_image.height]
            selected_item["extract_crop_mode"] = "full_image_direct"
            extract_source_image = full_image.copy()
        else:
            extract_plan = prepare_extract_source_image(
                full_image=full_image,
                bbox=selected_item.get("bbox"),
                garment_type=selected_type,
                total_items=all_items_count,
                detector_mask=selected_item.get("_mask_obj"),
            )
            if selected_item.get("bbox") != extract_plan.anchor_bbox:
                selected_item["detector_bbox"] = list(selected_item.get("bbox") or [])
                selected_item["bbox"] = list(extract_plan.anchor_bbox)
            selected_item["bbox_geometry_source"] = str(extract_plan.geometry_source)
            if extract_plan.mask_bbox:
                selected_item["mask_bbox"] = list(extract_plan.mask_bbox)
            selected_item["extract_crop_bbox"] = list(extract_plan.extract_bbox)
            selected_item["extract_crop_mode"] = str(extract_plan.crop_mode)
            extract_source_image = extract_plan.image
    except Exception:
        payload = build_error_payload(
            title="Invalid Image",
            description="Could not process the image. Please upload a clearer photo.",
            reason_codes=["INVALID_IMAGE"],
            status_code=400,
        )
        return None, multipart_form_response(payload)

    selected_prompt_hint = " ".join(str(selected_item.get("promptDescription") or "").split()).strip()
    if (
        not selected_prompt_hint
        or descriptor_is_weak(selected_prompt_hint)
        or caption_non_garment_signal(selected_prompt_hint)
    ):
        selected_prompt_hint = ""

    extracted_url = ""
    extraction_meta = {}
    extracted_image_bytes = b""
    fallback: Dict[str, object] = {}
    reference_mask = None
    color_reference_image = selected_item.get("_image_obj") or extract_source_image
    detector_mask_obj = selected_item.get("_mask_obj")
    if isinstance(detector_mask_obj, np.ndarray):
        try:
            detector_mask = np.asarray(detector_mask_obj).astype(bool)
            if detector_mask.ndim == 2:
                if direct_requested_type_mode and detector_mask.shape[:2] == (full_image.height, full_image.width):
                    reference_mask = detector_mask
                else:
                    extract_bbox = selected_item.get("extract_crop_bbox") or []
                    if (
                        len(extract_bbox) == 4
                        and detector_mask.shape[:2] == (full_image.height, full_image.width)
                    ):
                        ex0, ey0, ex1, ey1 = [int(v) for v in extract_bbox]
                        if ex1 > ex0 and ey1 > ey0:
                            cropped_mask = detector_mask[max(0, ey0):max(0, ey1), max(0, ex0):max(0, ex1)]
                            if cropped_mask.size > 0:
                                reference_mask = cropped_mask
        except Exception:
            reference_mask = None
    if isinstance(reference_mask, np.ndarray):
        color_ref_candidate = extract_source_image
        if reference_mask.shape[:2] != (extract_source_image.height, extract_source_image.width):
            detector_crop_candidate = selected_item.get("_image_obj")
            if (
                isinstance(detector_crop_candidate, Image.Image)
                and reference_mask.shape[:2] == (detector_crop_candidate.height, detector_crop_candidate.width)
            ):
                color_ref_candidate = detector_crop_candidate
        color_reference_image = color_ref_candidate
    t_extract = time.time()
    try:
        flux_fallback = run_flux2_cloth_only_extract(
            source_image=extract_source_image,
            garment_type=selected_type,
            prompt_description="",
            fallback_prompt_description=selected_prompt_hint,
            description_backend="minicpm_service",
            steps=flux2_single_garment_extract_default_steps,
            seed=flux2_single_garment_extract_default_seed,
            descriptor_source_image=selected_item.get("_image_obj") or extract_source_image,
            color_reference_image=color_reference_image,
            reference_mask=reference_mask,
            apply_type_color_mask=bool(selected_type),
        )
        fallback = flux_fallback
        extracted_url = str(flux_fallback.get("url") or "")
        extraction_meta = dict(flux_fallback.get("meta") or {})
        extracted_image_bytes = bytes(flux_fallback.get("_processed_image_bytes") or b"")
    except Exception as extract_err:
        logger.error(
            "Flux2 extraction failed (type=%s, bbox=%s): %s",
            selected_type,
            selected_item.get("bbox"),
            extract_err,
        )
        payload = build_error_payload(
            title="Extraction Failed",
            description="Garment extraction failed. Please retry with a clearer image.",
            reason_codes=["EXTRACTION_FAILED"],
            status_code=502,
        )
        return None, multipart_form_response(payload)

    stage_timings["extract_total_s"] = round(time.time() - t_extract, 4)

    if not extracted_url:
        payload = build_error_payload(
            title="Extraction Failed",
            description="No garment could be extracted. Please upload a clearer image.",
            reason_codes=["EXTRACTION_FAILED"],
            status_code=400,
        )
        return None, multipart_form_response(payload)

    selected_item["url"] = extracted_url
    selected_item["output_image_url"] = extracted_url
    selected_item["output_image_source"] = str(extraction_meta.get("path", "flux2_extract"))
    selected_item["raw_image_url"] = str(fallback.get("raw_url") or extracted_url)
    selected_item["raw_image_source"] = str(extraction_meta.get("pipeline") or "extract")
    selected_item["extraction"] = extraction_meta
    selected_item["_extracted_image_bytes"] = extracted_image_bytes
    selected_item["cloth_verified"] = bool(extracted_url)
    selected_item["cloth_verification_source"] = "flux2_extraction"
    selected_item["baseGarmentPrompt"] = str(extraction_meta.get("base_garment_prompt") or "").strip()
    selected_item["extractionAvoidClause"] = str(extraction_meta.get("extraction_avoid_clause") or "").strip()
    selected_item["promptSectionsRaw"] = str(extraction_meta.get("prompt_sections_raw") or "").strip()
    extracted_prompt_desc = " ".join(str(extraction_meta.get("prompt_description") or "").split()).strip()
    if extracted_prompt_desc:
        selected_item["promptDescriptionRaw"] = extracted_prompt_desc

    if analyze_prompt_from_extracted:
        prompt_desc = ""
        if extracted_prompt_desc:
            prompt_desc = extracted_prompt_desc
            selected_item["promptDescriptionSource"] = "flux2_extract_descriptor"
        elif analyze_require_extracted_prompt:
            payload = build_error_payload(
                title="Extraction Failed",
                description="Could not generate garment description. Please retry.",
                reason_codes=["EXTRACTION_FAILED"],
                status_code=400,
            )
            return None, multipart_form_response(payload)

        if prompt_desc:
            selected_item["promptDescription"] = prompt_desc
            selected_item["description"] = prompt_desc
            extracted_style = infer_style_from_text(prompt_desc, garment_type=selected_type)
            extracted_category = wardrobe_category_from_garment_type(selected_type, style=extracted_style)
            selected_item["style"] = extracted_category["style"]
            selected_item["category_key"] = extracted_category["category_key"]
            selected_item["primary_category_key"] = extracted_category["primary_category_key"]

    if forced_type:
        selected_item["extract_type_forced"] = forced_type

    return selected_item, None
