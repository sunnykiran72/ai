from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple


def apply_selected_item_prompting(
    *,
    selected_item: Optional[Dict[str, object]],
    requested_type: Optional[str],
    analyze_prompt_from_extracted: bool,
    normalize_garment_type: Callable[[Optional[str]], Optional[str]],
    infer_style_from_text: Callable[..., Optional[str]],
    wardrobe_category_from_garment_type: Callable[..., Dict[str, str]],
    product_prompt_description: Callable[..., str],
    build_garment_metadata: Callable[..., Dict[str, object]],
    strip_descriptor_color_clause: Callable[[str], str],
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    if not selected_item:
        return selected_item, {}

    extraction_obj = selected_item.get("extraction") if isinstance(selected_item.get("extraction"), dict) else {}
    prompt_text_raw = str(selected_item.get("promptDescription") or selected_item.get("description") or "")
    prompt_source = str(selected_item.get("promptDescriptionSource") or "").strip().lower()
    structure_seed = str(
        selected_item.get("baseGarmentPrompt")
        or extraction_obj.get("base_garment_prompt")
        or prompt_text_raw
        or ""
    )
    structure_prompt = " ".join(strip_descriptor_color_clause(structure_seed).split()).strip()
    if prompt_source == "flux2_extract_descriptor":
        prompt_for_product = structure_prompt
        if not prompt_for_product:
            prompt_for_product = product_prompt_description(
                prompt_text_raw,
                garment_type=str(selected_item.get("type") or ""),
                style=str(selected_item.get("style") or ""),
                category_key=str(selected_item.get("category_key") or ""),
            )
    else:
        prompt_for_product = product_prompt_description(
            structure_prompt or prompt_text_raw,
            garment_type=str(selected_item.get("type") or ""),
            style=str(selected_item.get("style") or ""),
            category_key=str(selected_item.get("category_key") or ""),
        )
    selected_item["promptDescription"] = prompt_for_product
    selected_item["description"] = prompt_for_product
    if structure_prompt:
        selected_item["baseGarmentPrompt"] = structure_prompt

    forced_type = requested_type if requested_type in {"top", "bottom", "dress", "outer"} else None
    selected_type = forced_type or normalize_garment_type(str(selected_item.get("type"))) or "top"
    prompt_description = str(selected_item.get("promptDescription") or "")

    sync_style_guess = str(selected_item.get("style") or "")
    if not sync_style_guess:
        sync_style_guess = str(infer_style_from_text(prompt_description, garment_type=selected_type) or "")
    sync_category = wardrobe_category_from_garment_type(
        selected_type,
        style=sync_style_guess if sync_style_guess else None,
    )
    selected_item["primary_category_key"] = sync_category["primary_category_key"]
    selected_item["category_key"] = sync_category["category_key"]
    selected_item["style"] = sync_category["style"]

    garment_metadata = build_garment_metadata(
        base_garment_prompt=str(selected_item.get("baseGarmentPrompt") or structure_prompt or prompt_description),
        extraction_avoid_clause=str(selected_item.get("extractionAvoidClause") or ""),
        prompt_sections_raw=str(selected_item.get("promptSectionsRaw") or ""),
        descriptor_raw_text=str(extraction_obj.get("descriptor_raw_text") or ""),
        prompt_description=prompt_description,
        prompt_source=str(selected_item.get("promptDescriptionSource") or ""),
        target_type=selected_type,
        backend_target_type=selected_type,
        style=sync_category["style"],
        primary_category_key=sync_category["primary_category_key"],
        category_key=sync_category["category_key"],
        dominant_hexes=[
            str(v)
            for v in (
                extraction_obj.get("dominant_hexes")
                or selected_item.get("dominant_color_hexes")
                or []
            )
            if str(v).strip()
        ],
        accent_hexes=[
            str(v)
            for v in (extraction_obj.get("accent_hexes") or [])
            if str(v).strip()
        ],
        color_hints=[
            str(v)
            for v in (extraction_obj.get("color_hints") or [])
            if str(v).strip()
        ],
        color_profile=(
            extraction_obj.get("color_profile")
            if isinstance(extraction_obj.get("color_profile"), dict)
            else {}
        ),
        color_mask_source=str(extraction_obj.get("color_mask_source") or ""),
    )
    selected_item["garmentMetadata"] = garment_metadata

    metadata_prompt = garment_metadata.get("prompt") if isinstance(garment_metadata.get("prompt"), dict) else {}
    resolved_base_prompt = " ".join(str(metadata_prompt.get("base_garment_prompt") or "").split()).strip()
    resolved_prompt_desc = " ".join(str(metadata_prompt.get("prompt_description") or "").split()).strip()
    if resolved_base_prompt:
        selected_item["baseGarmentPrompt"] = resolved_base_prompt
    if resolved_prompt_desc:
        selected_item["promptDescription"] = resolved_prompt_desc
        selected_item["description"] = resolved_prompt_desc
        prompt_description = resolved_prompt_desc

    return selected_item, {
        "selected_type": selected_type,
        "prompt_description": prompt_description,
        "sync_category": sync_category,
        "garment_metadata": garment_metadata,
        "prompt_source": "extracted_output" if analyze_prompt_from_extracted else "detected_crop",
    }
