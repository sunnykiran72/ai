from __future__ import annotations

import io
import time
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image


def _build_detector_prompt_description(raw_label: str, garment_type: str) -> str:
    label = " ".join(str(raw_label or garment_type or "").replace("_", " ").replace("-", " ").split()).strip()
    if not label:
        label = str(garment_type or "garment").strip()
    return f"Detected {label} garment."


def _candidate_to_item(
    *,
    inst: Dict[str, object],
    idx: int,
    image: Image.Image,
    normalize_garment_type: Callable[[Optional[str]], Optional[str]],
    infer_style_from_text: Callable[..., Optional[str]],
    wardrobe_category_from_garment_type: Callable[..., Dict[str, str]],
) -> Optional[Dict[str, object]]:
    resolved_type = normalize_garment_type(str(inst.get("type") or inst.get("label") or ""))
    if resolved_type not in {"top", "bottom", "dress", "outer"}:
        return None

    bbox = [int(v) for v in (inst.get("bbox") or [0, 0, image.width, image.height])]
    x0, y0, x1, y1 = bbox
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    crop_area_ratio = min(1.0, float(crop_w * crop_h) / max(1.0, float(image.width * image.height)))
    width_ratio = float(crop_w) / max(1.0, float(image.width))
    height_ratio = float(crop_h) / max(1.0, float(image.height))
    detector_label = str(inst.get("label") or resolved_type or "").strip()
    prompt_desc = _build_detector_prompt_description(detector_label, resolved_type)
    style_infer = infer_style_from_text(detector_label or prompt_desc, garment_type=resolved_type)
    category_meta = wardrobe_category_from_garment_type(resolved_type, style=style_infer)
    confidence = float(inst.get("confidence", 0.0) or 0.0)
    detection_source = str(inst.get("source") or "cloth_detector")

    # Parser assistance is helpful for typed requests, but occasionally it emits
    # a near full-frame top/bottom box on person photos. Those candidates are not
    # useful for selection and they poison downstream color/prompt extraction.
    if (
        detection_source == "human_parser"
        and resolved_type in {"top", "bottom"}
        and (
            crop_area_ratio >= 0.82
            or (width_ratio >= 0.92 and height_ratio >= 0.90)
        )
    ):
        return None

    item = {
        "garment_id": idx,
        "type": resolved_type,
        "garment_type": resolved_type,
        "type_source": str(inst.get("type_source") or "detector"),
        "detection_source": detection_source,
        "promptDescription": prompt_desc,
        "description": prompt_desc,
        "style": category_meta["style"],
        "category_key": category_meta["category_key"],
        "primary_category_key": category_meta["primary_category_key"],
        "url": None,
        "bbox": bbox,
        "crop": {
            "width": crop_w,
            "height": crop_h,
            "area_ratio": crop_area_ratio,
        },
        "confidence": {
            "yolo": confidence,
            "detector": confidence,
        },
        "_image_obj": inst.get("image") or image.crop(tuple(bbox)),
        "_mask_obj": inst.get("mask"),
        "detector_label": detector_label,
        "detector_metrics": dict(inst.get("metrics") or {}),
    }
    if inst.get("tighten_reason"):
        item["tighten_reason"] = str(inst.get("tighten_reason"))
    if inst.get("parser_area_ratio") is not None:
        item["parser_area_ratio"] = float(inst.get("parser_area_ratio") or 0.0)
    return item


def run_detection_stage_or_response(
    *,
    image: Image.Image,
    requested_type: Optional[str],
    selected_index: Optional[int],
    engine,
    stage_timings,
    logger,
    build_error_payload: Callable[..., Dict[str, object]],
    multipart_form_response: Callable[..., object],
    wardrobe_category_from_garment_type: Callable[..., Dict[str, str]],
    normalize_garment_type: Callable[[Optional[str]], Optional[str]],
    parser_preroute_instances: Callable[..., List[Dict[str, object]]],
    parser_split_is_plausible: Callable[..., Tuple[bool, str]],
    heuristic_split_candidates: Callable[..., List[Dict[str, object]]],
    should_force_fullbody_split: Callable[..., bool],
    largest_instance: Callable[..., Optional[Dict[str, object]]],
    suppress_auxiliary_instances: Callable[..., List[Dict[str, object]]],
    bbox_prior: Callable[..., float],
    hybrid_score: Callable[..., float],
    is_shorts_like_caption: Callable[[str], bool],
    infer_type_from_caption: Callable[[str], Optional[str]],
    caption_non_garment_signal: Callable[[str], bool],
    caption_garment_signal: Callable[[str], bool],
    infer_style_from_text: Callable[..., Optional[str]],
    dedupe_items: Callable[..., List[Dict[str, object]]],
    should_collapse_same_type: Callable[..., bool],
    item_rank_score: Callable[[Dict[str, object]], float],
    maybe_force_uncertain_fullbody_to_dress: Callable[..., Tuple[List[Dict[str, object]], Optional[Dict[str, object]]]],
    requested_type_geometry_score: Callable[..., float],
    analyze_use_parser_for_prerouting: bool,
    analyze_enable_parser_split: bool,
    analyze_enable_heuristic_split: bool,
    analyze_florence_dress_lock_min_score: float,
    analyze_heuristic_bottom_trim_shorts_ratio: float,
    use_florence_hybrid_verify: bool,
    hybrid_top_k: int,
    hybrid_min_score: float,
    analyze_max_items: int,
    analyze_caption_mode: str,
    analyze_primary_type_with_florence: bool,
    analyze_auto_select_multi_dress: bool,
) -> Tuple[Optional[Dict[str, object]], Optional[object]]:
    direct_requested_type_mode = False
    parser_split_used = False
    heuristic_split_used = False
    raw_detected_count = 0
    items: List[Dict[str, object]] = []
    normalized_requested = normalize_garment_type(requested_type)

    def _build_direct_requested_item() -> Dict[str, object]:
        fallback_type = normalized_requested or "top"
        category_meta = wardrobe_category_from_garment_type(fallback_type, style=None)
        prompt_desc = _build_detector_prompt_description(
            str(requested_type or fallback_type or "garment"),
            fallback_type,
        )
        return {
            "garment_id": 0,
            "type": fallback_type,
            "garment_type": fallback_type,
            "type_source": "requested_type_direct",
            "detection_source": "requested_type_direct_full_image",
            "promptDescription": prompt_desc,
            "description": prompt_desc,
            "style": category_meta["style"],
            "category_key": category_meta["category_key"],
            "primary_category_key": category_meta["primary_category_key"],
            "url": None,
            "bbox": [0, 0, image.width, image.height],
            "crop": {
                "width": image.width,
                "height": image.height,
                "area_ratio": 1.0,
            },
            "confidence": {
                "yolo": 0.01,
                "detector": 0.01,
                "hybrid": 0.01,
            },
            "_image_obj": image.copy(),
            "_mask_obj": None,
            "detector_label": str(requested_type or fallback_type or "garment"),
            "detector_metrics": {
                "fallback": "direct_requested_type_full_image",
            },
        }

    t_stage = time.time()
    detector = getattr(engine, "cloth_detector", None)
    detector_candidates: List[Dict[str, object]] = []
    if detector is not None:
        try:
            detector_candidates = detector.detect_fashion_candidates(
                image,
                requested_type=requested_type,
            )
        except Exception as detect_err:
            logger.warning(f"Cloth detector failed, falling back to legacy detection stage: {detect_err}")
    stage_timings["yolo_detect_s"] = round(time.time() - t_stage, 4)
    stage_timings["yolo_crop_s"] = 0.0

    if not detector_candidates:
        if normalized_requested in {"top", "bottom", "dress", "outer"}:
            direct_requested_type_mode = True
            items = [_build_direct_requested_item()]
            raw_detected_count = 0
        else:
            payload = build_error_payload(
                title="No Clothing Found",
                description="No clothing item was detected in the uploaded image.",
                reason_codes=["NO_CLOTHING"],
                status_code=400,
            )
            return None, multipart_form_response(payload)

    if not detector_candidates and not items:
        payload = build_error_payload(
            title="No Clothing Found",
            description="No clothing item was detected in the uploaded image.",
            reason_codes=["NO_CLOTHING"],
            status_code=400,
        )
        return None, multipart_form_response(payload)

    raw_detected_count = len(detector_candidates)
    items = []
    for idx, inst in enumerate(detector_candidates[:analyze_max_items]):
        item = _candidate_to_item(
            inst=inst,
            idx=idx,
            image=image,
            normalize_garment_type=normalize_garment_type,
            infer_style_from_text=infer_style_from_text,
            wardrobe_category_from_garment_type=wardrobe_category_from_garment_type,
        )
        if item is not None:
            items.append(item)

    if not items:
        if normalized_requested in {"top", "bottom", "dress", "outer"}:
            direct_requested_type_mode = True
            items = [_build_direct_requested_item()]
        else:
            payload = build_error_payload(
                title="No Clothing Found",
                description="No clothing item was detected in the uploaded image.",
                reason_codes=["NO_CLOTHING"],
                status_code=400,
            )
            return None, multipart_form_response(payload)

    # For explicit typed requests, always let parser prerouting challenge a bad detector pick.
    # This stays narrow because we only keep parser candidates when they score materially better.
    if normalized_requested in {"top", "bottom", "dress", "outer"}:
        try:
            parser_instances = parser_preroute_instances(image, requested_type=requested_type)
        except Exception as parser_err:
            logger.warning(f"Parser preroute failed during typed selection assist: {parser_err}")
            parser_instances = []

        parser_items: List[Dict[str, object]] = []
        for parser_idx, inst in enumerate(parser_instances):
            item = _candidate_to_item(
                inst=inst,
                idx=len(items) + parser_idx,
                image=image,
                normalize_garment_type=normalize_garment_type,
                infer_style_from_text=infer_style_from_text,
                wardrobe_category_from_garment_type=wardrobe_category_from_garment_type,
            )
            if item is not None:
                parser_items.append(item)

        if parser_items:
            detector_same = [it for it in items if normalize_garment_type(str(it.get("type"))) == normalized_requested]
            parser_same = [it for it in parser_items if normalize_garment_type(str(it.get("type"))) == normalized_requested]
            if parser_same:
                best_parser = max(
                    parser_same,
                    key=lambda it: requested_type_geometry_score(it, normalized_requested or "", image.height),
                )
                best_detector = max(
                    detector_same,
                    key=lambda it: requested_type_geometry_score(it, normalized_requested or "", image.height),
                ) if detector_same else None
                parser_score = requested_type_geometry_score(best_parser, normalized_requested or "", image.height)
                detector_score = requested_type_geometry_score(best_detector, normalized_requested or "", image.height) if best_detector else float("-inf")
                if (best_detector is None) or (parser_score > detector_score + 0.12):
                    parser_split_used = True
                    items.extend(parser_same)

    if normalized_requested in {"top", "bottom"}:
        same_type_items = [
            it for it in items if normalize_garment_type(str(it.get("type"))) == normalized_requested
        ]
        if not same_type_items:
            base_item = largest_instance(items) if items else None
            if base_item is not None:
                heuristic_base = {
                    "bbox": [0, 0, image.width, image.height],
                    "confidence": float(
                        (base_item.get("confidence") or {}).get("detector")
                        if isinstance(base_item.get("confidence"), dict)
                        else 0.0
                    ),
                }
                try:
                    heuristic_instances = heuristic_split_candidates(image, heuristic_base)
                except Exception as heuristic_err:
                    logger.warning(f"Heuristic split failed during typed selection assist: {heuristic_err}")
                    heuristic_instances = []

                heuristic_items: List[Dict[str, object]] = []
                for heuristic_idx, inst in enumerate(heuristic_instances):
                    item = _candidate_to_item(
                        inst=inst,
                        idx=len(items) + heuristic_idx,
                        image=image,
                        normalize_garment_type=normalize_garment_type,
                        infer_style_from_text=infer_style_from_text,
                        wardrobe_category_from_garment_type=wardrobe_category_from_garment_type,
                    )
                    if item is not None:
                        heuristic_items.append(item)

                heuristic_same = [
                    it for it in heuristic_items
                    if normalize_garment_type(str(it.get("type"))) == normalized_requested
                ]
                if heuristic_same:
                    heuristic_split_used = True
                    items.extend(heuristic_same)

    items = dedupe_items(items, iou_threshold=0.60)
    unique_types = sorted({str(it.get("type")) for it in items if it.get("type")})
    if len(items) > 1 and len(unique_types) <= 1 and should_collapse_same_type(items):
        best = max(items, key=item_rank_score)
        items = [best]

    for new_idx, item in enumerate(items):
        item["garment_id"] = new_idx

    auto_selected_index = None
    if selected_index is None and requested_type and len(items) > 1:
        matched = [
            idx for idx, item in enumerate(items)
            if normalize_garment_type(str(item.get("type"))) == requested_type
        ]
        if len(matched) >= 1:
            auto_selected_index = max(
                matched,
                key=lambda idx: requested_type_geometry_score(items[idx], requested_type, image.height),
            )
        else:
            auto_selected_index = max(
                range(len(items)),
                key=lambda idx: requested_type_geometry_score(items[idx], requested_type, image.height),
            )
    elif (
        selected_index is None
        and not requested_type
        and len(items) > 1
        and analyze_auto_select_multi_dress
    ):
        normalized_types = [normalize_garment_type(str(item.get("type") or "")) for item in items]
        unique_norm_types = sorted({t for t in normalized_types if t})
        if len(unique_norm_types) == 1 and unique_norm_types[0] == "dress":
            auto_selected_index = max(range(len(items)), key=lambda idx: item_rank_score(items[idx]))

    return {
        "items": items,
        "direct_requested_type_mode": direct_requested_type_mode,
        "raw_detected_count": raw_detected_count,
        "parser_split_used": parser_split_used,
        "heuristic_split_used": heuristic_split_used,
        "auto_selected_index": auto_selected_index,
    }, None


def build_selection_required_response(
    *,
    items: List[Dict[str, object]],
    full_image: Image.Image,
    started_at: float,
    gpu_queue_wait_s: float,
    raw_detected_count: int,
    parser_split_used: bool,
    heuristic_split_used: bool,
    selection_preview_format: str,
    selection_preview_max_side: int,
    selection_preview_jpeg_quality: int,
    to_public_item: Callable[[Dict[str, object]], Dict[str, object]],
    build_adaptive_rect_crop_variants: Callable[..., List[Dict[str, object]]],
    prepare_extract_source_image: Callable[..., object],
    build_multipart_parts: Callable[..., Dict[str, object]],
    build_success_payload: Callable[..., Dict[str, object]],
    multipart_form_response: Callable[..., object],
):
    total_s = round(time.time() - started_at, 4)
    item_breakdown = []
    binary_parts = []
    split_selection_needed = raw_detected_count <= 1 and (parser_split_used or heuristic_split_used)

    for item in items:
        pub = to_public_item(item)
        pub["selection_index"] = int(pub.get("garment_id", 0))
        pub["rank"] = int(pub["selection_index"]) + 1
        mapped_type = item.get("garment_type") or item.get("type", "unknown")
        pub["garment_type"] = mapped_type
        item_breakdown.append(pub)

        preview = None
        try:
            preview_variants = build_adaptive_rect_crop_variants(
                full_image=full_image,
                selected_candidate=item,
                garment_type=str(mapped_type or ""),
                all_candidates=items,
            )
            if preview_variants:
                chosen_preview = preview_variants[0]
                preview = chosen_preview.get("image")
                if isinstance(preview, Image.Image):
                    preview = preview.convert("RGB")
                pub["preview_crop_bbox"] = [
                    int(v)
                    for v in (
                        chosen_preview.get("bbox")
                        or item.get("bbox")
                        or [0, 0, full_image.width, full_image.height]
                    )
                ]
                pub["preview_geometry_source"] = str(chosen_preview.get("name") or "adaptive_rect")
            if preview is None:
                preview_plan = prepare_extract_source_image(
                    full_image=full_image,
                    bbox=item.get("bbox"),
                    garment_type=str(mapped_type or ""),
                    total_items=len(items),
                    detector_mask=item.get("_mask_obj"),
                )
                preview = preview_plan.image.convert("RGB")
                pub["preview_crop_bbox"] = [int(v) for v in preview_plan.extract_bbox]
                pub["preview_geometry_source"] = str(preview_plan.geometry_source)
        except Exception:
            item_img = item.get("_image_obj")
            if item_img is not None:
                preview = item_img.convert("RGB")

        if preview is None:
            continue

        if max(preview.size) > selection_preview_max_side:
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            preview.thumbnail((selection_preview_max_side, selection_preview_max_side), resampling)
        buf = io.BytesIO()
        if selection_preview_format == "png":
            preview.save(buf, format="PNG", optimize=True)
            file_ext = "png"
            content_type = "image/png"
        else:
            preview.save(
                buf,
                format="JPEG",
                quality=selection_preview_jpeg_quality,
                optimize=True,
            )
            file_ext = "jpg"
            content_type = "image/jpeg"
        part_name = f"item_{int(pub['rank'])}"
        binary_parts.append({
            "name": part_name,
            "filename": f"{part_name}.{file_ext}",
            "content_type": content_type,
            "bytes": buf.getvalue(),
        })

    data = {
        "result": "REJECTED",
        "title": "Selection Required",
        "description": "We found multiple garments. Please select one item to continue.",
        "reason_codes": (
            ["MULTI_ITEM_SELECTION_REQUIRED", "LOW_CONFIDENCE"]
            if split_selection_needed
            else ["MULTI_ITEM_SELECTION_REQUIRED"]
        ),
        "selection_required": True,
        "total_garments_found": len(items),
        "selection_hint": {
            "expected_field": "type",
            "allowed_types": ["top", "bottom", "dress", "outer"],
        },
        "item_breakdown": item_breakdown,
        "multipart_data": build_multipart_parts(items=item_breakdown),
        "latencies": {"total": total_s, "gpu_queue_wait": gpu_queue_wait_s},
        "processing_time_ms": int(total_s * 1000),
    }
    payload = build_success_payload(data=data, status_code=400, message="")
    return multipart_form_response(payload, binary_parts=binary_parts)


def resolve_selected_item_or_response(
    *,
    items: List[Dict[str, object]],
    requested_type: Optional[str],
    selected_index: Optional[int],
    auto_selected_index: Optional[int],
    min_accept_confidence: float,
    build_error_payload: Callable[..., Dict[str, object]],
    multipart_form_response: Callable[..., object],
) -> Tuple[Optional[Dict[str, object]], Optional[int], Optional[object]]:
    selected_item = None
    selected_index_internal = None

    if selected_index is not None:
        if 1 <= selected_index <= len(items):
            selected_index_internal = selected_index - 1
        else:
            selected_index_internal = selected_index
        if selected_index_internal < 0 or selected_index_internal >= len(items):
            payload = build_error_payload(
                title="Invalid Selection",
                description=f"selected_index must be 0..{max(0, len(items) - 1)} (legacy) or 1..{len(items)} (rank).",
                reason_codes=["INVALID_SELECTION_TYPE"],
                status_code=400,
            )
            return None, selected_index_internal, multipart_form_response(payload)
        selected_item = items[selected_index_internal]
    elif auto_selected_index is not None:
        selected_item = items[auto_selected_index]
        selected_index = auto_selected_index
    elif items:
        selected_item = items[0]

    if selected_item and not requested_type:
        conf_obj = selected_item.get("confidence") if isinstance(selected_item.get("confidence"), dict) else {}
        best_conf = float(conf_obj.get("hybrid", conf_obj.get("yolo", 0.0)))
        if best_conf < min_accept_confidence:
            payload = build_error_payload(
                title="Low Confidence Detection",
                description="The garment is not clear enough. Try a centered and brighter photo.",
                reason_codes=["LOW_CONFIDENCE"],
                status_code=400,
            )
            return None, selected_index_internal, multipart_form_response(payload)

    return selected_item, selected_index_internal, None
