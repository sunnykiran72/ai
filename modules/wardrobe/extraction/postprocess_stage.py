from __future__ import annotations

import time
import uuid
from typing import Callable, Dict, Optional, Tuple


def sync_selected_item_progress(
    *,
    selected_item: Optional[Dict[str, object]],
    requested_type: Optional[str],
    stage_timings,
    enable_progress_sync: bool,
    authorization: Optional[str],
    sync_wardrobe_progress_dispatch: Callable[..., Dict[str, object]],
    prompting_context: Dict[str, object],
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    if not selected_item:
        return selected_item, {}

    progress_id = str(uuid.uuid4())
    output_url = str(selected_item.get("url") or "")
    has_extraction_output = bool(output_url)
    output_source = str(selected_item.get("output_image_source") or "")
    selected_type = str(prompting_context.get("selected_type") or selected_item.get("type") or "top")
    prompt_description = str(prompting_context.get("prompt_description") or selected_item.get("promptDescription") or "")
    sync_category = prompting_context.get("sync_category") if isinstance(prompting_context.get("sync_category"), dict) else {}
    garment_metadata = prompting_context.get("garment_metadata") if isinstance(prompting_context.get("garment_metadata"), dict) else {}

    progress_meta = {
        "selected_type": selected_type,
        "requested_type": requested_type,
        "primary_category_key": sync_category.get("primary_category_key"),
        "category_key": sync_category.get("category_key"),
        "style": sync_category.get("style"),
        "outputImageSource": output_source or "crop",
        "prompt_source": str(prompting_context.get("prompt_source") or ""),
        "extraction": selected_item.get("extraction"),
        "cloth_verified": bool(selected_item.get("cloth_verified")),
        "cloth_verification_source": selected_item.get("cloth_verification_source"),
        "detection_source": selected_item.get("detection_source"),
        "type_source": selected_item.get("type_source"),
        "bbox": selected_item.get("bbox"),
        "crop": selected_item.get("crop"),
    }

    if enable_progress_sync and has_extraction_output:
        t_sync = time.time()
        progress_sync = sync_wardrobe_progress_dispatch(
            authorization=authorization,
            progress_id=progress_id,
            output_url=output_url,
            prompt_description=prompt_description,
            garment_metadata=garment_metadata,
            metadata=progress_meta,
        )
        stage_timings["progress_sync_s"] = round(time.time() - t_sync, 4)
    else:
        progress_sync = {
            "enabled": bool(enable_progress_sync),
            "synced": False,
            "reason": "skipped_missing_output" if not has_extraction_output else "disabled",
            "id": progress_id,
        }

    progress_id = str(progress_sync.get("id") or progress_id)
    selected_item["wardrobe_progress_id"] = progress_id
    selected_item["progress_sync"] = progress_sync
    return selected_item, {"progress_id": progress_id, "progress_sync": progress_sync}
