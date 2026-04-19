"""
Garment analysis service.

This module provides the AnalyzeService class that orchestrates garment
analysis workflows using the modular extraction pipeline.
"""

from typing import Dict, List, Optional
import asyncio
import base64
import inspect
import logging
import os
import time
import io
import requests

from PIL import Image

from config import AnalyzeConfig
from services.ai_engine import AIEngine
from shared.response_payloads import (
    build_error_payload,
    build_success_payload,
    build_multipart_parts,
    multipart_form_response,
)
from shared.security import verify_bearer_token
from shared.image_ops import download_image
from shared.category_mapping import wardrobe_category_from_garment_type, infer_style_from_text
from utils.validation import normalize_garment_type, sanitize_garment_description
from utils.scoring import (
    dedupe_items_by_iou,
    calculate_bbox_prior,
    hybrid_score,
    item_rank_by_score,
    find_largest_instance,
)
from core.garment_extractor import GarmentExtractor, GarmentExtractionConfig, GarmentExtractionRequest
from modules.wardrobe.extraction.pipeline import default_extraction_stage_timings
from modules.wardrobe.extraction.utils import (
    verify_authorization_or_response,
    resolve_upload_or_response,
    read_input_image_or_response,
    build_success_response,
)
from modules.wardrobe.extraction.detection_stage import (
    run_detection_stage_or_response,
    resolve_selected_item_or_response,
    build_selection_required_response,
)
from modules.wardrobe.extraction.generation_stage import run_selected_item_extraction_or_response
from modules.wardrobe.extraction.postprocess_stage import sync_selected_item_progress
from modules.wardrobe.extraction.debug_artifacts import maybe_save_debug_artifacts
from core.qwen_extract_shared_runner import get_shared_qwen_extract_runner

logger = logging.getLogger("glamify-ai")

ANALYZE_QWEN_TYPE_PROMPTS = {
    "top": (
        "Extract the clothing and create a flat mockup for a {category_type} garment only as shown into a clean, standalone white background mockup. "
        "Strictly preserve every detail of target garment and size, original fabric texture, stitching, folds, patterns and color accuracy."
    ),
    "bottom": (
        "Extract the clothing and create a flat mockup for a {category_type} garment only as shown into a clean, standalone white background mockup. "
        "Strictly preserve every detail of target garment and size, original fabric texture, stitching, folds, patterns and color accuracy."
    ),
    "dress": (
        "Extract the clothing and create a flat mockup for a {category_type} garment only as shown into a clean, standalone white background mockup. "
        "Strictly preserve every detail of target garment and size, original fabric texture, stitching, folds, patterns and color accuracy."
    ),
    "outer": (
        "Extract the clothing and create a flat mockup for a {category_type} garment only as shown into a clean, standalone white background mockup. "
        "Strictly preserve every detail of target garment and size, original fabric texture, stitching, folds, patterns and color accuracy."
    ),
}
ANALYZE_QWEN_DEFAULT_STEPS = 12
ANALYZE_QWEN_DEFAULT_SEED = 576
ANALYZE_QWEN_DEFAULT_GUIDANCE = 0.0
ANALYZE_QWEN_MAX_INPUT_EDGE = 768
ANALYZE_QWEN_MIN_INPUT_LONGEST_EDGE = 768
ANALYZE_QWEN_MAX_OUTPUT_EDGE = 768
ANALYZE_QWEN_OUTPUT_ASPECT_RATIO = ""
ANALYZE_QWEN_FORCE_MINICPM_PROMPT = str(os.getenv("ANALYZE_QWEN_FORCE_MINICPM_PROMPT", "1")).strip().lower() in {
    "1", "true", "yes", "on"
}
ANALYZE_QWEN_MATCH_LAB_OUTPUT = str(os.getenv("ANALYZE_QWEN_MATCH_LAB_OUTPUT", "1")).strip().lower() in {
    "1", "true", "yes", "on"
}

try:
    import jwt as _jwt
except Exception:  # pragma: no cover - optional dependency
    _jwt = None


class AnalyzeService:
    """
    Orchestrates garment analysis workflows.

    Responsibilities:
    - Detect garments in images using fashion-object-detection with YOLO fallback
    - Split multi-garment images (top/bottom separation)
    - Extract garment metadata (color, type, descriptors)
    - Generate selection previews for user confirmation
    - Coordinate extraction pipeline stages
    """

    def __init__(self, engine: AIEngine, config: AnalyzeConfig):
        self.engine = engine
        self.config = config

    def _get_qwen_extract_runner(self):
        return get_shared_qwen_extract_runner()

    async def analyze_image(
        self,
        upload,
        garment_type: Optional[str] = None,
        debug: bool = False,
        authorization: Optional[str] = None,
    ):
        if upload is None or not hasattr(upload, "read"):
            return {
                "status": "not_implemented",
                "message": "AnalyzeService expects an uploaded file.",
            }

        try:
            from ai import main as main_mod
        except (ModuleNotFoundError, ImportError):
            import main as main_mod

        if not inspect.iscoroutinefunction(upload.read):
            return {
                "status": "not_implemented",
                "message": "AnalyzeService expects an async UploadFile.",
            }

        analyze_stage_timings = default_extraction_stage_timings()
        started_at = time.time()

        auth_result = verify_authorization_or_response(
            authorization,
            verify_bearer_token=lambda auth: verify_bearer_token(
                auth,
                jwt_access_secret=str(main_mod.config.app.jwt_access_secret or ""),
                jwt_module=_jwt,
                logger=logger,
            ),
            build_error_payload=build_error_payload,
            multipart_form_response=multipart_form_response,
        )
        if not isinstance(auth_result, dict):
            return auth_result

        upload = resolve_upload_or_response(
            file_upload=upload,
            image_upload=None,
            build_error_payload=build_error_payload,
            multipart_form_response=multipart_form_response,
        )
        if not hasattr(upload, "read"):
            return upload

        image_input = await read_input_image_or_response(
            upload,
            max_file_bytes=int(main_mod.ANALYZE_MAX_FILE_BYTES),
            blur_check_enabled=bool(main_mod.ANALYZE_BLUR_CHECK_ENABLED),
            blur_min_focus_score=float(main_mod.ANALYZE_BLUR_MIN_FOCUS_SCORE),
            focus_score_fn=main_mod._focus_score,
            build_error_payload=build_error_payload,
            multipart_form_response=multipart_form_response,
            stage_timings=analyze_stage_timings,
        )
        if not hasattr(image_input, "image"):
            return image_input

        image = image_input.image
        requested_type = normalize_garment_type(garment_type)

        gpu_queue_wait_s = 0.0
        gpu_slot_acquired = False
        gpu_sem = getattr(main_mod, "gpu_semaphore", None)
        t_gpu_wait = time.time()
        if gpu_sem is not None:
            timeout_s = float(getattr(self.config, "gpu_queue_timeout_s", 70.0))
            try:
                acquire_call = gpu_sem.acquire()
                if inspect.isawaitable(acquire_call):
                    await asyncio.wait_for(acquire_call, timeout=timeout_s)
                    gpu_slot_acquired = True
                else:
                    try:
                        acquired = gpu_sem.acquire(timeout=timeout_s)
                    except TypeError:
                        acquired = bool(acquire_call)
                    gpu_slot_acquired = bool(acquired)
                gpu_queue_wait_s = round(time.time() - t_gpu_wait, 4)
            except TypeError:
                # Non-async semaphore or unexpected acquire signature: skip queueing.
                gpu_slot_acquired = False
            except asyncio.TimeoutError:
                payload = build_error_payload(
                    title="Server Busy",
                    description="Please retry in a moment.",
                    reason_codes=["GPU_QUEUE_TIMEOUT"],
                    status_code=503,
                )
                return multipart_form_response(payload)
            except Exception as exc:
                logger.warning("GPU queue unavailable, proceeding without semaphore: %s", exc)
                gpu_slot_acquired = False

        try:
            def _simple_build_adaptive_rect_crop_variants(
                *,
                full_image: Image.Image,
                selected_candidate: dict,
                garment_type: str,
                all_candidates: Optional[list[dict]] = None,
            ) -> list[dict]:
                # Force selection previews to reuse the extraction crop plan
                # (same padded geometry) via prepare_extract_source_image fallback.
                _ = (full_image, selected_candidate, garment_type, all_candidates)
                return []

            # Helper functions for extraction
            def _prepare_extract_source_image(
                *,
                full_image: Image.Image,
                bbox: Optional[list[int]],
                garment_type: str,
                total_items: int,
                detector_mask: Optional[object] = None,
                semantic_bbox: Optional[list[int]] = None,
            ):
                extractor = GarmentExtractor(
                    config=GarmentExtractionConfig(
                        force_bbox_crop=bool(self.config.extract_force_bbox_crop),
                        crop_pad_ratio_x=float(self.config.extract_crop_pad_ratio_x),
                        crop_pad_ratio_y=float(self.config.extract_crop_pad_ratio_y),
                        crop_pad_ratio=float(self.config.extract_crop_pad_ratio),
                        crop_pad_ratio_dress=float(self.config.extract_crop_pad_ratio_dress),
                        crop_bottom_extra_ratio_dress=float(self.config.extract_crop_bottom_extra_ratio_dress),
                        crop_top_extra_ratio_bottom=float(self.config.extract_crop_top_extra_ratio_bottom),
                        crop_top_extra_ratio_bottom_multi=float(self.config.extract_crop_top_extra_ratio_bottom_multi),
                        dress_top_recovery_ratio=float(self.config.extract_dress_top_recovery_ratio),
                        top_top_recovery_ratio=float(self.config.extract_top_top_recovery_ratio),
                    )
                )
                return extractor.prepare(
                    GarmentExtractionRequest(
                        full_image=full_image,
                        garment_type=garment_type,
                        total_items=total_items,
                        detected_bbox=bbox,
                        detector_mask=detector_mask,
                        semantic_bbox=semantic_bbox,
                    )
                )

            def _flatten_rgba_on_white(image: Image.Image) -> Image.Image:
                if image.mode != "RGBA":
                    return image.convert("RGB")
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[-1])
                return bg

            def _run_qwen_extraction(**kwargs) -> Dict[str, object]:
                """
                Run Qwen extract-outfit directly (no HTTP calls).
                Keeps the same return contract used by extraction stage.
                """
                from core.qwen_extract_outfit_service import (
                    build_qwen_extract_outfit_request,
                    execute_qwen_extract_outfit_request,
                )

                source_image = kwargs.get("source_image")
                if source_image is None:
                    raise RuntimeError("source_image is required for extraction")
                source_image = source_image.convert("RGB")
                original_source_width, original_source_height = source_image.size

                # Keep Qwen input deterministic for analyze:
                # resize longest edge to exactly 768px (upscale or downscale).
                longest_edge = max(source_image.width, source_image.height)
                target_edge = int(ANALYZE_QWEN_MIN_INPUT_LONGEST_EDGE)
                if longest_edge != target_edge:
                    ratio = float(target_edge) / float(max(1, longest_edge))
                    target_size = (
                        max(1, int(round(source_image.width * ratio))),
                        max(1, int(round(source_image.height * ratio))),
                    )
                    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                    source_image = source_image.resize(target_size, resampling)
                qwen_input_edge = int(max(source_image.width, source_image.height))

                selected_type = normalize_garment_type(str(kwargs.get("garment_type") or "")) or "top"
                qwen_prompt = ANALYZE_QWEN_TYPE_PROMPTS.get(selected_type, ANALYZE_QWEN_TYPE_PROMPTS["top"])
                qwen_output_max_edge_alias = None
                qwen_output_aspect_ratio_alias = None
                top_min_output_width_rule_applied = False
                forced_output_width = None
                forced_output_height = None
                if selected_type == "top" and int(source_image.width) < 512:
                    top_min_output_width_rule_applied = True
                    forced_output_width = 512
                    safe_width = max(1, int(source_image.width))
                    forced_output_height = max(
                        1,
                        int(round(float(source_image.height) * (float(forced_output_width) / float(safe_width)))),
                    )
                    qwen_output_max_edge_alias = int(max(forced_output_width, forced_output_height))
                    qwen_output_aspect_ratio_alias = f"{int(forced_output_width)}:{int(forced_output_height)}"
                qwen_request = build_qwen_extract_outfit_request(
                    prompt=qwen_prompt,
                    steps=ANALYZE_QWEN_DEFAULT_STEPS,
                    seed=ANALYZE_QWEN_DEFAULT_SEED,
                    guidance_scale=ANALYZE_QWEN_DEFAULT_GUIDANCE,
                    guidance_scale_alias=None,
                    negative_prompt="",
                    negative_prompt_alias=None,
                    max_input_edge=ANALYZE_QWEN_MAX_INPUT_EDGE,
                    max_input_edge_alias=qwen_input_edge,
                    output_max_edge=None,
                    output_max_edge_alias=qwen_output_max_edge_alias,
                    output_aspect_ratio=None,
                    output_aspect_ratio_alias=qwen_output_aspect_ratio_alias,
                    upload_output=False,
                    include_base64=True,
                )
                qwen_data = execute_qwen_extract_outfit_request(
                    request=qwen_request,
                    source_image=source_image,
                    runner=self._get_qwen_extract_runner(),
                    minicpm_runner=getattr(self.engine, "minicpm", None),
                    upload_image_fn=None,
                    enable_minicpm_prompt_override=ANALYZE_QWEN_FORCE_MINICPM_PROMPT,
                )
                image_base64 = str(qwen_data.get("image_base64") or "").strip()
                if not image_base64:
                    raise RuntimeError("Qwen extraction did not return image_base64.")
                extracted_image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")

                # Analyze parity mode with qwen-lab:
                # keep raw Qwen output unless explicitly disabled.
                output_image = extracted_image
                analyze_postprocess_applied = False
                if not ANALYZE_QWEN_MATCH_LAB_OUTPUT:
                    if bool(self.config.garment_postprocess_enabled):
                        output_image = main_mod._enhance_image(output_image)
                    output_image = main_mod._pad_image(
                        output_image,
                        int(self.config.garment_target_aspect_w),
                        int(self.config.garment_target_aspect_h),
                    )
                    if str(self.config.garment_output_background).lower() == "white":
                        if output_image.mode != "RGBA":
                            output_image = output_image.convert("RGBA")
                        white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
                        white_bg.paste(output_image, mask=output_image.split()[-1])
                        output_image = white_bg.convert("RGB")
                    analyze_postprocess_applied = True
                
                # Save to bytes
                out_buf = io.BytesIO()
                output_image.save(out_buf, format="PNG")
                processed_bytes = out_buf.getvalue()
                
                # Upload to storage
                url = ""
                try:
                    url = main_mod._upload_or_raise(processed_bytes)
                except Exception as exc:
                    logger.error(f"Upload failed: {exc}")
                    url = ""
                
                # Build metadata
                metadata = dict(qwen_data.get("metadata") or {})
                qwen_output_size_raw = dict(metadata.get("output_size") or {})
                metadata["input_original_size"] = {
                    "width": int(original_source_width),
                    "height": int(original_source_height),
                }
                metadata["input_preprocessed_size"] = {
                    "width": int(source_image.width),
                    "height": int(source_image.height),
                }
                metadata["input_size"] = {
                    "width": int(source_image.width),
                    "height": int(source_image.height),
                }
                metadata["top_min_output_width_rule_applied"] = bool(top_min_output_width_rule_applied)
                if top_min_output_width_rule_applied:
                    metadata["top_min_output_width_rule"] = {
                        "requested_width": int(forced_output_width or 512),
                        "derived_height": int(forced_output_height or source_image.height),
                    }
                metadata["qwen_output_size_raw"] = {
                    "width": int(qwen_output_size_raw.get("width") or extracted_image.width),
                    "height": int(qwen_output_size_raw.get("height") or extracted_image.height),
                }
                metadata["final_output_size"] = {
                    "width": int(output_image.width),
                    "height": int(output_image.height),
                }
                metadata["analyze_postprocess_applied"] = bool(analyze_postprocess_applied)
                metadata["analyze_qwen_match_lab_output"] = bool(ANALYZE_QWEN_MATCH_LAB_OUTPUT)
                metadata["analyze_force_minicpm_prompt"] = bool(ANALYZE_QWEN_FORCE_MINICPM_PROMPT)
                garment_metadata_obj = qwen_data.get("garmentMetadata")
                qwen_garment_metadata = garment_metadata_obj if isinstance(garment_metadata_obj, dict) else {}
                prompt_description = str(qwen_data.get("promptDescription") or "").strip()
                if not prompt_description:
                    prompt_description = str(kwargs.get("prompt_description") or "").strip()
                if not prompt_description:
                    prompt_description = qwen_prompt

                metadata.update({
                    "path": "qwen_extract_outfit",
                    "pipeline": "qwen_extract_outfit",
                    "base_garment_prompt": qwen_prompt,
                    "extraction_avoid_clause": "",
                    "prompt_description": prompt_description,
                    "prompt_sections_raw": "",
                    "descriptor_raw_text": str(qwen_data.get("promptDescription") or kwargs.get("minicpm_description") or "").strip(),
                    "negative_prompt_mode": "none",
                    "negative_prompt_supplied": False,
                    "prompt_source": str(qwen_data.get("promptDescriptionSource") or ""),
                    "prompt_fallback_used": bool(qwen_data.get("promptFallbackUsed")),
                    "prompt_elapsed_seconds": qwen_data.get("promptElapsedSeconds"),
                    "qwen_elapsed_seconds": qwen_data.get("qwenElapsedSeconds"),
                    "total_elapsed_seconds": qwen_data.get("totalElapsedSeconds"),
                    "qwen_steps": ANALYZE_QWEN_DEFAULT_STEPS,
                    "qwen_seed": ANALYZE_QWEN_DEFAULT_SEED,
                    "prefer_extracted_prompt": True,
                    "garment_metadata": qwen_garment_metadata,
                })

                return {
                    "url": url,
                    "raw_url": url,
                    "_processed_image_bytes": processed_bytes,
                    "meta": metadata,
                    "garment_metadata": qwen_garment_metadata,
                }

            def _suppress_auxiliary_instances(
                instances: list[dict],
                image_width: int,
                image_height: int,
            ) -> list[dict]:
                if len(instances) <= 1:
                    return instances
                scored = []
                for inst in instances:
                    bbox = inst.get("bbox") or [0, 0, image_width, image_height]
                    x0, y0, x1, y1 = [int(v) for v in bbox]
                    area = max(1, x1 - x0) * max(1, y1 - y0)
                    scored.append((area, inst))
                max_area = max(a for a, _ in scored)
                cutoff = max(1.0, float(max_area) * max(0.0, min(1.0, float(self.config.aux_min_rel_area))))
                kept = [inst for area, inst in scored if float(area) >= cutoff]
                return kept if kept else [max(scored, key=lambda x: x[0])[1]]

            detection_result, detection_response = run_detection_stage_or_response(
                image=image,
                requested_type=requested_type,
                selected_index=None,
                engine=self.engine,
                stage_timings=analyze_stage_timings,
                logger=logger,
                build_error_payload=build_error_payload,
                multipart_form_response=multipart_form_response,
                wardrobe_category_from_garment_type=wardrobe_category_from_garment_type,
                normalize_garment_type=normalize_garment_type,
                parser_preroute_instances=lambda *_args, **_kwargs: [],
                parser_split_is_plausible=lambda *_args, **_kwargs: (False, ""),
                heuristic_split_candidates=lambda *_args, **_kwargs: [],
                should_force_fullbody_split=lambda *_args, **_kwargs: False,
                largest_instance=lambda items: find_largest_instance(items) if items else None,
                suppress_auxiliary_instances=_suppress_auxiliary_instances,
                bbox_prior=lambda bbox, width, height: calculate_bbox_prior(bbox, width, height),
                hybrid_score=lambda yolo_conf, florence_conf, bbox_prior: hybrid_score(yolo_conf, florence_conf, bbox_prior),
                is_shorts_like_caption=lambda _text: False,
                infer_type_from_caption=lambda _text: None,
                caption_non_garment_signal=lambda _text: False,
                caption_garment_signal=lambda _text: False,
                infer_style_from_text=infer_style_from_text,
                dedupe_items=dedupe_items_by_iou,
                should_collapse_same_type=lambda items: bool(main_mod.ANALYZE_COLLAPSE_SAME_TYPE),
                item_rank_score=item_rank_by_score,
                maybe_force_uncertain_fullbody_to_dress=main_mod._maybe_force_uncertain_fullbody_to_dress,
                requested_type_geometry_score=main_mod._requested_type_geometry_score,
                analyze_use_parser_for_prerouting=bool(main_mod.ANALYZE_USE_PARSER_FOR_PREROUTING),
                analyze_enable_parser_split=bool(main_mod.ANALYZE_ENABLE_PARSER_SPLIT),
                analyze_enable_heuristic_split=bool(main_mod.ANALYZE_ENABLE_HEURISTIC_SPLIT),
                analyze_florence_dress_lock_min_score=float(self.config.florence_dress_lock_min_score),
                analyze_heuristic_bottom_trim_shorts_ratio=float(self.config.heuristic_bottom_trim_shorts_ratio),
                use_florence_hybrid_verify=bool(main_mod.USE_FLORENCE_HYBRID_VERIFY),
                hybrid_top_k=int(getattr(main_mod, "HYBRID_TOP_K", self.config.hybrid_top_k)),
                hybrid_min_score=float(getattr(main_mod, "HYBRID_MIN_SCORE", self.config.hybrid_min_score)),
                analyze_max_items=int(getattr(main_mod, "ANALYZE_MAX_ITEMS", self.config.max_items)),
                analyze_caption_mode=str(getattr(main_mod, "ANALYZE_CAPTION_MODE", self.config.caption_mode)),
                analyze_primary_type_with_florence=bool(getattr(main_mod, "ANALYZE_PRIMARY_TYPE_WITH_FLORENCE", self.config.primary_type_with_florence)),
                analyze_auto_select_multi_dress=bool(getattr(main_mod, "ANALYZE_AUTO_SELECT_MULTI_DRESS", self.config.auto_select_multi_dress)),
                prepare_extract_source_image=_prepare_extract_source_image,
            )
            if detection_response is not None:
                return detection_response

            items = list(detection_result.get("items") or [])
            direct_requested_type_mode = bool(detection_result.get("direct_requested_type_mode"))
            raw_detected_count = int(detection_result.get("raw_detected_count") or 0)
            parser_split_used = bool(detection_result.get("parser_split_used"))
            heuristic_split_used = bool(detection_result.get("heuristic_split_used"))
            auto_selected_index = detection_result.get("auto_selected_index")

            if not requested_type and len(items) > 1:
                return build_selection_required_response(
                    items=items,
                    full_image=image,
                    started_at=started_at,
                    gpu_queue_wait_s=gpu_queue_wait_s,
                    raw_detected_count=raw_detected_count,
                    parser_split_used=parser_split_used,
                    heuristic_split_used=heuristic_split_used,
                    selection_preview_format=str(self.config.selection_preview_format),
                    selection_preview_max_side=int(self.config.selection_preview_max_side),
                    selection_preview_jpeg_quality=int(self.config.selection_preview_jpeg_quality),
                    to_public_item=main_mod._to_public_item,
                    build_adaptive_rect_crop_variants=_simple_build_adaptive_rect_crop_variants,
                    prepare_extract_source_image=_prepare_extract_source_image,
                    build_multipart_parts=build_multipart_parts,
                    build_success_payload=build_success_payload,
                    multipart_form_response=multipart_form_response,
                )

            selected_item, _selected_index_internal, selected_item_response = resolve_selected_item_or_response(
                items=items,
                requested_type=requested_type,
                selected_index=None,
                auto_selected_index=auto_selected_index,
                min_accept_confidence=float(getattr(main_mod, "ANALYZE_MIN_ACCEPT_CONFIDENCE", self.config.min_accept_confidence)),
                build_error_payload=build_error_payload,
                multipart_form_response=multipart_form_response,
            )
            if selected_item_response is not None:
                return selected_item_response

            # Qwen extraction now generates prompt/metadata; skip pre-extraction MiniCPM/prompting in analyze.
            selected_item["minicpm_description"] = str(selected_item.get("minicpm_description") or "").strip()
            selected_item["joycaption_description"] = ""
            analyze_stage_timings["minicpm_s"] = 0.0
            analyze_stage_timings["joycaption_s"] = 0.0
            prompting_context: Dict[str, object] = {}

            # Stage: Extraction (Run Flux2 extraction with prompts)
            selected_item, extraction_response = run_selected_item_extraction_or_response(
                selected_item=selected_item,
                requested_type=requested_type,
                direct_requested_type_mode=direct_requested_type_mode,
                full_image=image,
                all_items_count=len(items),
                stage_timings=analyze_stage_timings,
                engine=self.engine,
                logger=logger,
                analyze_extract_cloth=bool(getattr(main_mod, "ANALYZE_EXTRACT_CLOTH", self.config.extract_cloth)),
                analyze_prompt_from_extracted=bool(getattr(main_mod, "ANALYZE_PROMPT_FROM_EXTRACTED", self.config.prompt_from_extracted)),
                analyze_require_extracted_prompt=bool(self.config.require_extracted_prompt),
                analyze_caption_mode=str(getattr(main_mod, "ANALYZE_CAPTION_MODE", self.config.caption_mode)),
                flux2_single_garment_extract_default_steps=int(self.config.flux2_single_garment_extract_default_steps),
                flux2_single_garment_extract_default_seed=int(self.config.flux2_single_garment_extract_default_seed),
                normalize_garment_type=normalize_garment_type,
                prepare_extract_source_image=_prepare_extract_source_image,
                build_error_payload=build_error_payload,
                multipart_form_response=multipart_form_response,
                run_flux2_cloth_only_extract=_run_qwen_extraction,
                descriptor_is_weak=main_mod._descriptor_is_weak,
                caption_non_garment_signal=lambda _text: False,
                download_image=download_image,
                flatten_rgba_on_white=_flatten_rgba_on_white,
                sanitize_garment_description=sanitize_garment_description,
                infer_style_from_text=infer_style_from_text,
                wardrobe_category_from_garment_type=wardrobe_category_from_garment_type,
            )
            if extraction_response is not None:
                return extraction_response

            # Build sync context from final extraction output fields.
            final_selected_type = normalize_garment_type(
                str(selected_item.get("type") or requested_type or "")
            ) or "top"
            extraction_meta_obj = selected_item.get("extraction")
            extraction_meta = extraction_meta_obj if isinstance(extraction_meta_obj, dict) else {}
            final_prompt_description = " ".join(
                str(
                    selected_item.get("promptDescription")
                    or extraction_meta.get("prompt_description")
                    or selected_item.get("description")
                    or ""
                ).split()
            ).strip()
            final_prompt_source = str(
                selected_item.get("promptDescriptionSource")
                or extraction_meta.get("prompt_source")
                or ""
            ).strip()
            final_style = str(selected_item.get("style") or "").strip()
            if not final_style and final_prompt_description:
                inferred_style = infer_style_from_text(final_prompt_description, garment_type=final_selected_type)
                final_style = str(inferred_style or "").strip()
            final_sync_category = wardrobe_category_from_garment_type(
                final_selected_type, style=final_style or None
            )
            selected_item["style"] = final_sync_category["style"]
            selected_item["category_key"] = final_sync_category["category_key"]
            selected_item["primary_category_key"] = final_sync_category["primary_category_key"]

            garment_meta_obj = selected_item.get("garmentMetadata")
            final_garment_metadata = garment_meta_obj if isinstance(garment_meta_obj, dict) else {}
            if not final_garment_metadata:
                extraction_garment_meta = extraction_meta.get("garment_metadata")
                if isinstance(extraction_garment_meta, dict):
                    final_garment_metadata = extraction_garment_meta
                    selected_item["garmentMetadata"] = extraction_garment_meta

            prompting_context = {
                "selected_type": final_selected_type,
                "prompt_description": final_prompt_description,
                "avoid_prompt": "",
                "sync_category": final_sync_category,
                "garment_metadata": final_garment_metadata,
                "prompt_source": final_prompt_source,
                "minicpm_description": str(selected_item.get("minicpm_description") or ""),
            }

            # Stage: Sync Progress
            def _sync_wardrobe_progress_dispatch(
                *,
                authorization: Optional[str],
                progress_id: str,
                output_url: str,
                prompt_description: str,
                garment_metadata: Dict[str, object],
                metadata: Dict[str, object],
            ) -> Dict[str, object]:
                base_url = str(main_mod.config.app.wardrobe_progress_api_base_url or "").strip()
                if not base_url:
                    return {
                        "enabled": bool(main_mod.config.app.enable_wardrobe_progress_sync),
                        "synced": False,
                        "reason": "missing_base_url",
                        "id": progress_id,
                    }

                payload = {
                    "id": progress_id,
                    "progressId": progress_id,
                    "progress_id": progress_id,
                    "outputImage": output_url,
                    "output_image": output_url,
                    "outputUrl": output_url,
                    "output_url": output_url,
                    "promptDescription": prompt_description,
                    "prompt_description": prompt_description,
                    "garmentMetadata": garment_metadata,
                    "garment_metadata": garment_metadata,
                    "metadata": metadata,
                }

                if bool(main_mod.config.app.wardrobe_progress_include_input_image):
                    input_url = str(selected_item.get("raw_image_url") or "")
                    if input_url:
                        payload["inputImage"] = input_url
                        payload["input_image"] = input_url
                        payload["inputImageUrl"] = input_url
                        payload["input_image_url"] = input_url

                headers = {"Content-Type": "application/json"}
                if authorization:
                    headers["Authorization"] = authorization

                try:
                    resp = requests.post(
                        base_url,
                        json=payload,
                        headers=headers,
                        timeout=float(main_mod.config.app.wardrobe_progress_sync_timeout_s),
                    )
                    try:
                        body = resp.json()
                    except Exception:
                        body = resp.text
                    return {
                        "enabled": True,
                        "synced": bool(resp.ok),
                        "status_code": resp.status_code,
                        "id": progress_id,
                        "response": body,
                    }
                except Exception as exc:
                    return {
                        "enabled": True,
                        "synced": False,
                        "reason": f"request_failed: {exc}",
                        "id": progress_id,
                    }

            selected_item, _postprocess_context = sync_selected_item_progress(
                selected_item=selected_item,
                requested_type=requested_type,
                stage_timings=analyze_stage_timings,
                enable_progress_sync=bool(main_mod.config.app.enable_wardrobe_progress_sync),
                authorization=authorization,
                sync_wardrobe_progress_dispatch=_sync_wardrobe_progress_dispatch,
                prompting_context=prompting_context,
            )

            capture_path = maybe_save_debug_artifacts(
                enabled=bool(getattr(self.config, "debug_artifact_capture_enabled", False)),
                capture_dir=str(getattr(self.config, "debug_artifact_capture_dir", "")),
                allowed_types=str(getattr(self.config, "debug_artifact_capture_types", "top,dress")),
                upload_name=getattr(image_input, "upload_name", ""),
                selected_item=selected_item,
                requested_type=requested_type,
            )
            if capture_path is not None and selected_item is not None:
                selected_item["debugArtifactCapturePath"] = str(capture_path)

            # Build final response
            public_item = main_mod._to_public_item(selected_item) if selected_item else None
            return build_success_response(
                public_item=public_item,
                selected_item=selected_item,
                items=items,
                requested_type=requested_type,
                total_s=time.time() - started_at,
                gpu_queue_wait_s=gpu_queue_wait_s,
                analyze_stage_timings=analyze_stage_timings.to_dict(),
                wardrobe_category_from_garment_type=wardrobe_category_from_garment_type,
                build_multipart_parts=build_multipart_parts,
                build_success_payload=build_success_payload,
                multipart_form_response=multipart_form_response,
                fetch_image_bytes=lambda _url, timeout=45: b"",
            )
        finally:
            if gpu_slot_acquired and gpu_sem is not None:
                gpu_sem.release()
