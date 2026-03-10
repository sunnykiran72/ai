from __future__ import annotations

import io
import time
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image


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
    direct_requested_type_mode = requested_type in {"top", "bottom", "dress", "outer"} and selected_index is None
    parser_split_used = False
    heuristic_split_used = False
    raw_detected_count = 0
    items: List[Dict[str, object]] = []

    if direct_requested_type_mode:
        direct_category_meta = wardrobe_category_from_garment_type(requested_type, style=None)
        items = [{
            "garment_id": 0,
            "type": requested_type,
            "garment_type": requested_type,
            "type_source": "requested_type_direct",
            "detection_source": "type_direct_image",
            "promptDescription": "",
            "description": "",
            "style": direct_category_meta["style"],
            "category_key": direct_category_meta["category_key"],
            "primary_category_key": direct_category_meta["primary_category_key"],
            "url": None,
            "bbox": [0, 0, image.width, image.height],
            "crop": {
                "width": int(image.width),
                "height": int(image.height),
                "area_ratio": 1.0,
            },
            "confidence": {
                "yolo": 1.0,
            },
            "_image_obj": image.copy(),
            "_mask_obj": None,
        }]
    else:
        t_stage = time.time()
        instances = engine.yolo.detect_instances(image)
        stage_timings["yolo_detect_s"] = round(time.time() - t_stage, 4)
        if not instances:
            payload = build_error_payload(
                title="No Clothing Found",
                description="No clothing item was detected in the uploaded image.",
                reason_codes=["NO_CLOTHING"],
                status_code=400,
            )
            return None, multipart_form_response(payload)

        t_stage = time.time()
        instances = engine.yolo.get_crops(image, instances)
        stage_timings["yolo_crop_s"] = round(time.time() - t_stage, 4)

        raw_detected_count = len(instances)
        if (
            False
            and analyze_use_parser_for_prerouting
            and analyze_enable_parser_split
            and engine.parser is not None
        ):
            try:
                parser_candidates = parser_preroute_instances(
                    image=image,
                    requested_type=requested_type,
                    square_padding_ratio=0.12,
                )
                if len(parser_candidates) >= 2:
                    plausible, _reason = parser_split_is_plausible(parser_candidates, image.height)
                    if plausible:
                        parser_split_used = True
                        instances = parser_candidates
            except Exception as parser_err:
                logger.warning(f"Human parser split fallback failed: {parser_err}")

        if analyze_enable_heuristic_split and not parser_split_used and instances:
            try:
                trigger_split = False
                base_inst = instances[0]
                if len(instances) == 1:
                    trigger_split = True
                elif should_force_fullbody_split(instances, image.height):
                    largest = largest_instance(instances)
                    if largest is not None:
                        base_inst = largest
                        trigger_split = True

                if trigger_split:
                    classify = engine.florence.classify_garment_type(base_inst["image"], hint_type=requested_type)
                    base_type = str(classify.get("type") or "").strip().lower()
                    base_score = float(classify.get("score", 0.0) or 0.0)
                    dress_locked = (
                        base_type == "dress"
                        and base_score >= analyze_florence_dress_lock_min_score
                        and requested_type not in {"top", "bottom"}
                    )
                    if not dress_locked:
                        heuristics = heuristic_split_candidates(image, base_inst)
                        if len(heuristics) >= 2:
                            heuristic_split_used = True
                            instances = heuristics
            except Exception as heur_err:
                logger.warning(f"Heuristic split fallback failed: {heur_err}")

        if False and analyze_use_parser_for_prerouting and len(instances) >= 1 and engine.parser is not None:
            try:
                instances = instances
            except Exception as tighten_err:
                logger.warning(f"Parser-based crop tightening failed: {tighten_err}")

        instances = suppress_auxiliary_instances(instances, image.width, image.height)

        if use_florence_hybrid_verify and instances:
            ranked = sorted(instances, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
            for idx, inst in enumerate(ranked):
                yolo_conf = float(inst.get("confidence", 0.0))
                bbox = inst.get("bbox") or [0, 0, image.width, image.height]
                prior = bbox_prior(bbox, image.width, image.height)

                florence_type = normalize_garment_type(str(inst.get("label", ""))) or "top"
                florence_conf = 0.50
                florence_reason = "fallback_no_florence"
                florence_caption = ""

                if idx < hybrid_top_k:
                    try:
                        source_hint = normalize_garment_type(str(inst.get("label", "")))
                        hint = requested_type or source_hint
                        verdict = engine.florence.classify_garment_type(inst["image"], hint_type=hint)
                        florence_type = verdict.get("type") or florence_type
                        florence_conf = float(verdict.get("score", florence_conf))
                        florence_reason = str(verdict.get("reason", "ok"))
                        florence_caption = str(verdict.get("caption", ""))
                        if (
                            str(inst.get("source", "")) == "heuristic_split"
                            and source_hint
                            and florence_type != source_hint
                            and florence_conf < 0.90
                        ):
                            florence_type = source_hint
                            florence_reason = f"{florence_reason}|locked_to_heuristic_label"

                        if (
                            str(inst.get("source", "")) == "heuristic_split"
                            and source_hint == "bottom"
                            and is_shorts_like_caption(florence_caption)
                        ):
                            bx = inst.get("bbox") or [0, 0, image.width, image.height]
                            x0, y0, x1, y1 = [int(v) for v in bx]
                            h = max(1, y1 - y0)
                            keep_h = max(int(image.height * 0.16), int(h * analyze_heuristic_bottom_trim_shorts_ratio))
                            new_y1 = min(y1, y0 + keep_h)
                            if new_y1 - y0 >= 48:
                                inst["bbox"] = [x0, y0, x1, new_y1]
                                inst["image"] = image.crop((x0, y0, x1, new_y1))
                                bbox = inst["bbox"]
                                prior = bbox_prior(bbox, image.width, image.height)
                    except Exception as classify_err:
                        logger.warning(f"Florence hybrid classify failed, falling back to YOLO candidate ordering: {classify_err}")

                score = hybrid_score(yolo_conf=yolo_conf, florence_conf=florence_conf, bbox_prior=prior)
                if requested_type and florence_type == requested_type:
                    score = min(1.0, score + 0.06)

                inst["_hybrid"] = {
                    "predicted_type": florence_type,
                    "yolo_confidence": yolo_conf,
                    "florence_confidence": florence_conf,
                    "bbox_prior": prior,
                    "score": score,
                    "florence_reason": florence_reason,
                    "florence_caption": florence_caption,
                }

            ranked = sorted(ranked, key=lambda inst: float(inst.get("_hybrid", {}).get("score", 0.0)), reverse=True)
            if hybrid_min_score > 0.0:
                thresholded = [inst for inst in ranked if float(inst.get("_hybrid", {}).get("score", 0.0)) >= hybrid_min_score]
                if thresholded:
                    ranked = thresholded
            instances = ranked[:analyze_max_items]

        items = []
        for idx, inst in enumerate(instances):
            t_caption = time.time()
            if analyze_caption_mode == "detailed":
                prompt_desc = engine.florence.describe_garment(inst["image"])
            else:
                prompt_desc = engine.florence.describe_garment_short(inst["image"])
            stage_timings["caption_total_s"] += (time.time() - t_caption)

            hybrid_meta = inst.get("_hybrid", {})
            autotype_meta = {}
            if use_florence_hybrid_verify:
                resolved_type = hybrid_meta.get("predicted_type")
                type_source = "florence_hybrid"
            else:
                resolved_type = normalize_garment_type(str(inst.get("label")))
                type_source = "yolo"
                if not resolved_type:
                    caption_guess = infer_type_from_caption(prompt_desc)
                    if caption_guess:
                        resolved_type = caption_guess
                        type_source = "caption_autotype"
                        autotype_meta = {"reason": "caption_guess"}
                    else:
                        try:
                            verdict = engine.florence.classify_garment_type(inst["image"], hint_type=requested_type)
                            guess = normalize_garment_type(str(verdict.get("type")))
                            if guess:
                                resolved_type = guess
                                type_source = "florence_autotype"
                                autotype_meta = {
                                    "score": float(verdict.get("score", 0.0)),
                                    "reason": str(verdict.get("reason", "")),
                                }
                        except Exception as auto_type_err:
                            logger.warning(f"Per-instance automatic type resolution failed: {auto_type_err}")
                if not resolved_type:
                    resolved_type = "top"
            detection_source = str(inst.get("source", "yolo"))

            bbox = inst["bbox"]
            x0, y0, x1, y1 = bbox
            crop_w = max(1, int(x1) - int(x0))
            crop_h = max(1, int(y1) - int(y0))
            crop_area_ratio = min(1.0, float(crop_w * crop_h) / max(1.0, float(image.width * image.height)))

            non_garment_caption = caption_non_garment_signal(prompt_desc)
            has_garment_caption = caption_garment_signal(prompt_desc)
            if non_garment_caption and not has_garment_caption:
                det_conf = float(inst.get("confidence", 0.0))
                label_hint = (
                    normalize_garment_type(str(inst.get("label") or ""))
                    or normalize_garment_type(str(resolved_type or ""))
                )
                plausible_garment = (
                    label_hint in {"top", "bottom", "dress", "outer"}
                    and crop_area_ratio >= 0.045
                    and det_conf >= 0.35
                )
                if not plausible_garment:
                    logger.info("Suppressing non-garment candidate from caption: %s", (prompt_desc or "")[:140])
                    continue

            style_infer = autotype_meta.get("specific_clothing_style") if autotype_meta else None
            if not style_infer and prompt_desc:
                style_infer = infer_style_from_text(prompt_desc, garment_type=resolved_type)

            category_meta = wardrobe_category_from_garment_type(resolved_type, style=style_infer)
            item = {
                "garment_id": idx,
                "type": resolved_type,
                "garment_type": resolved_type,
                "type_source": type_source,
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
                    "yolo": float(hybrid_meta.get("yolo_confidence", inst.get("confidence", 0.0))),
                    "florence_type": float(hybrid_meta.get("florence_confidence", 0.0)),
                    "bbox_prior": float(hybrid_meta.get("bbox_prior", 0.0)),
                    "hybrid": float(hybrid_meta.get("score", 0.0)),
                } if use_florence_hybrid_verify else {
                    "yolo": float(inst.get("confidence", 0.0)),
                },
                "_image_obj": inst["image"],
                "_mask_obj": inst.get("mask"),
            }
            if autotype_meta:
                item["autotype"] = autotype_meta
            if detection_source == "human_parser":
                item["parser_area_ratio"] = float(inst.get("parser_area_ratio", 0.0))
            if inst.get("tighten_source"):
                item["tighten_source"] = str(inst.get("tighten_source"))
            if inst.get("tighten_adjustment"):
                item["tighten_adjustment"] = str(inst.get("tighten_adjustment"))
            items.append(item)

        if requested_type in {"top", "bottom", "dress", "outer"} and not items:
            direct_category_meta = wardrobe_category_from_garment_type(requested_type, style=None)
            items = [{
                "garment_id": 0,
                "type": requested_type,
                "garment_type": requested_type,
                "type_source": "requested_type_direct",
                "detection_source": "type_forced_full_image",
                "promptDescription": "",
                "description": "",
                "style": direct_category_meta["style"],
                "category_key": direct_category_meta["category_key"],
                "primary_category_key": direct_category_meta["primary_category_key"],
                "url": None,
                "bbox": [0, 0, image.width, image.height],
                "crop": {
                    "width": int(image.width),
                    "height": int(image.height),
                    "area_ratio": 1.0,
                },
                "confidence": {
                    "yolo": 1.0,
                },
                "_image_obj": image.copy(),
                "_mask_obj": None,
            }]

        if not items:
            payload = build_error_payload(
                title="No Clothing Found",
                description="No clothing item was detected in the uploaded image.",
                reason_codes=["NO_CLOTHING"],
                status_code=400,
            )
            return None, multipart_form_response(payload)

    if not direct_requested_type_mode:
        deduped_items = dedupe_items(items, iou_threshold=0.60)
        items = deduped_items
        unique_types = sorted({str(it.get("type")) for it in items if it.get("type")})
        if len(items) > 1 and len(unique_types) <= 1 and should_collapse_same_type(items):
            best = max(items, key=item_rank_score)
            items = [best]

        for new_idx, item in enumerate(items):
            item["garment_id"] = new_idx

    if (not direct_requested_type_mode) and analyze_primary_type_with_florence and items:
        for item in items:
            try:
                t_type = time.time()
                crop_img = item.get("_image_obj")
                if crop_img is None:
                    continue
                hint = requested_type or normalize_garment_type(str(item.get("type") or ""))
                verdict = engine.florence.classify_garment_type(crop_img, hint_type=hint)
                predicted_type = normalize_garment_type(str(verdict.get("type") or ""))
                predicted_score = float(verdict.get("score", 0.0) or 0.0)
                if predicted_type in {"top", "bottom", "dress", "outer"}:
                    item["type"] = predicted_type
                    item["garment_type"] = predicted_type
                    item["type_source"] = "florence_primary"
                    item["type_model_score"] = round(predicted_score, 4)
                    inferred_style = infer_style_from_text(
                        str(item.get("promptDescription") or item.get("description") or ""),
                        garment_type=predicted_type,
                    )
                    category_meta = wardrobe_category_from_garment_type(
                        predicted_type,
                        style=inferred_style if inferred_style else None,
                    )
                    item["style"] = category_meta["style"]
                    item["category_key"] = category_meta["category_key"]
                    item["primary_category_key"] = category_meta["primary_category_key"]
                stage_timings["primary_type_total_s"] += (time.time() - t_type)
            except Exception as type_err:
                logger.warning(f"Primary type identify failed for analyze item {item.get('garment_id')}: {type_err}")

    if (not direct_requested_type_mode) and len(items) > 1:
        typed_deduped_items = dedupe_items(items, iou_threshold=0.75)
        if typed_deduped_items:
            items = typed_deduped_items
            for new_idx, item in enumerate(items):
                item["garment_id"] = new_idx

    if not direct_requested_type_mode:
        items, _forced_dress_fallback = maybe_force_uncertain_fullbody_to_dress(
            items,
            image_width=image.width,
            image_height=image.height,
            requested_type=requested_type,
        )
        for new_idx, item in enumerate(items):
            item["garment_id"] = new_idx

    auto_selected_index = None
    if selected_index is None and requested_type and len(items) > 1:
        matched = [
            idx for idx, item in enumerate(items)
            if normalize_garment_type(str(item.get("type"))) == requested_type
        ]
        if len(matched) >= 1:
            auto_selected_index = matched[0]
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
