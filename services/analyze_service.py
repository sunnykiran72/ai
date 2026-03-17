"""
Garment analysis service.

This module provides the AnalyzeService class that orchestrates garment
analysis workflows using the modular extraction pipeline.
"""

from typing import Dict, List, Optional
import asyncio
import inspect
import logging
import time

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
from utils.metadata_extraction import strip_descriptor_color_clause
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
)
from modules.wardrobe.extraction.generation_stage import run_selected_item_extraction_or_response
from modules.wardrobe.extraction.prompting_stage import apply_selected_item_prompting
from modules.wardrobe.extraction.postprocess_stage import sync_selected_item_progress

logger = logging.getLogger("glamify-ai")

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
        except ModuleNotFoundError:
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
                bbox = selected_candidate.get("bbox") or [0, 0, full_image.width, full_image.height]
                x0, y0, x1, y1 = [int(v) for v in bbox]
                x0 = max(0, min(x0, full_image.width - 1))
                y0 = max(0, min(y0, full_image.height - 1))
                x1 = max(x0 + 1, min(x1, full_image.width))
                y1 = max(y0 + 1, min(y1, full_image.height))
                return [{
                    "name": "detector_bbox",
                    "bbox": [x0, y0, x1, y1],
                    "image": full_image.crop((x0, y0, x1, y1)).convert("RGB"),
                }]

            # MiniCPM: Generates detailed garment description for metadata
            # JoyCaption: Generates additional context for negative prompts
            minicpm_runner = getattr(self.engine, "minicpm", None)  # Fixed: was minicpm_runner
            joycaption_runner = getattr(self.engine, "joycaption", None)
            florence_runner = getattr(self.engine, "florence", None)

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

            def _product_prompt_description(
                text: str,
                *,
                garment_type: Optional[str] = None,
                style: Optional[str] = None,
                category_key: Optional[str] = None,
            ) -> str:
                desc = sanitize_garment_description(text or "")
                lowered = desc.lower()
                human_terms = (
                    " woman ",
                    " man ",
                    " person ",
                    " model ",
                    " mannequin ",
                    " showroom ",
                    " wearing ",
                    " posing ",
                    " standing ",
                )
                padded = f" {lowered} "
                if any(term in padded for term in human_terms):
                    preferred = (style or "").strip()
                    if not preferred or preferred.lower() in {"top", "bottom", "dress", "outerwear", "unknown"}:
                        ck = (category_key or "").strip().replace("_", " ")
                        if ck and ck.lower() not in {"top", "bottom", "dress", "outerwear", "unknown"}:
                            preferred = ck
                        else:
                            gt = normalize_garment_type(garment_type) or "top"
                            preferred = {
                                "top": "Top",
                                "bottom": "Bottom",
                                "dress": "Dress",
                                "outer": "Outerwear",
                            }.get(gt, "Garment")
                    preferred = " ".join(preferred.split())
                    if preferred:
                        return f"{preferred[0].upper() + preferred[1:]} .".replace(" .", ".")
                    return "Garment."
                if desc and not desc.endswith("."):
                    desc = f"{desc}."
                return desc or "Garment."

            def _run_flux2_cloth_only_extract(**kwargs) -> Dict[str, object]:
                fallback_fn = getattr(main_mod, "_run_vton_cloth_only_fallback", None)
                if fallback_fn is None:
                    raise RuntimeError("Flux2 extract pipeline not available in refactor.")

                payload = {
                    "image_url": str(getattr(upload, "filename", "") or "upload"),
                    "garment_type": str(kwargs.get("garment_type") or ""),
                    "vto_mode": False,
                    "base_prompt": kwargs.get("base_prompt", ""),
                    "negative_prompt": kwargs.get("negative_prompt", ""),
                    "minicpm_description": kwargs.get("minicpm_description", ""),
                    "prompt_description": kwargs.get("prompt_description", ""),
                    "fallback_prompt_description": kwargs.get("fallback_prompt_description", ""),
                    "source_image": kwargs.get("source_image"),
                    "steps": kwargs.get("steps"),
                    "seed": kwargs.get("seed"),
                }
                try:
                    sig = inspect.signature(fallback_fn)
                    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
                        return fallback_fn(**payload)
                    filtered = {k: v for k, v in payload.items() if k in sig.parameters}
                    return fallback_fn(**filtered)
                except Exception:
                    return fallback_fn(**payload)

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
            )
            if detection_response is not None:
                return detection_response

            items = list(detection_result.get("items") or [])
            direct_requested_type_mode = bool(detection_result.get("direct_requested_type_mode"))
            raw_detected_count = int(detection_result.get("raw_detected_count") or 0)
            parser_split_used = bool(detection_result.get("parser_split_used"))
            heuristic_split_used = bool(detection_result.get("heuristic_split_used"))
            auto_selected_index = detection_result.get("auto_selected_index")

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

            # Stage: MiniCPM + Florence (Run in Parallel)
            # Determine which image to use for description
            if isinstance(selected_item.get("_image_obj"), Image.Image):
                desc_image = selected_item.get("_image_obj")
            elif "bbox" in selected_item and selected_item["bbox"]:
                bbox = selected_item["bbox"]
                x1, y1, x2, y2 = bbox
                desc_image = image.crop((x1, y1, x2, y2))
            else:
                desc_image = image

            # Run MiniCPM and JoyCaption in parallel for efficiency
            async def run_minicpm():
                if not minicpm_runner:
                    return ""
                try:
                    loop = asyncio.get_event_loop()
                    description = await loop.run_in_executor(
                        None,
                        lambda: minicpm_runner.describe_garment(
                            image=desc_image,
                            prompt_override=None
                        )
                    )
                    return description
                except Exception as e:
                    logger.warning(f"MiniCPM description failed: {e}")
                    return ""

            async def run_joycaption():
                runner = joycaption_runner
                if not runner:
                    return ""
                try:
                    loop = asyncio.get_event_loop()
                    description = await loop.run_in_executor(
                        None,
                        lambda: runner.describe_garment(image=desc_image)
                    )
                    return description
                except Exception as e:
                    logger.warning(f"JoyCaption description failed: {e}")
                    return ""

            t_description = time.time()
            minicpm_desc, joycaption_desc = await asyncio.gather(
                run_minicpm(),
                run_joycaption()
            )
            description_time = round(time.time() - t_description, 4)

            selected_item["minicpm_description"] = minicpm_desc
            # Keep the existing key to avoid downstream changes in prompting_stage.
            selected_item["joycaption_description"] = joycaption_desc
            analyze_stage_timings["minicpm_s"] = description_time
            analyze_stage_timings["joycaption_s"] = description_time

            # Stage: Prompting (Generate Flux2 prompts based on type and MiniCPM description)
            selected_item, prompting_context = apply_selected_item_prompting(
                selected_item=selected_item,
                requested_type=requested_type,
                analyze_prompt_from_extracted=bool(self.config.prompt_from_extracted),
                normalize_garment_type=normalize_garment_type,
                infer_style_from_text=infer_style_from_text,
                wardrobe_category_from_garment_type=wardrobe_category_from_garment_type,
                product_prompt_description=_product_prompt_description,
                build_garment_metadata=main_mod._build_garment_metadata,
                strip_descriptor_color_clause=strip_descriptor_color_clause,
            )

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
                run_flux2_cloth_only_extract=_run_flux2_cloth_only_extract,
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

            # Stage: Sync Progress
            selected_item, _postprocess_context = sync_selected_item_progress(
                selected_item=selected_item,
                requested_type=requested_type,
                stage_timings=analyze_stage_timings,
                enable_progress_sync=bool(main_mod.config.app.enable_wardrobe_progress_sync),
                authorization=authorization,
                sync_wardrobe_progress_dispatch=lambda **_kwargs: {
                    "enabled": bool(main_mod.config.app.enable_wardrobe_progress_sync),
                    "synced": False,
                    "reason": "disabled",
                    "id": "",
                },
                prompting_context=prompting_context,
            )

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
