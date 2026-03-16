from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from config.prompts import get_flux2_positive_prompt, get_flux2_negative_prompt


def _extract_unwanted_terms_from_joycaption(joycaption_desc: str, garment_type: str) -> str:
    """
    Extract unwanted terms from JoyCaption description to enhance negative prompts.
    
    JoyCaption often mentions person, accessories, or background elements that should be avoided.
    This function identifies those terms and adds them to the negative prompt.
    """
    if not joycaption_desc:
        return ""
    
    desc_lower = joycaption_desc.lower()
    unwanted = []
    
    # Common unwanted terms that JoyCaption might mention
    person_terms = ["woman", "man", "person", "model", "girl", "lady", "taking", "holding", "wearing", "posing"]
    accessory_terms = ["phone", "mirror", "selfie", "jewelry", "necklace", "earring", "bracelet", "bag", "purse"]
    background_terms = ["room", "wall", "background", "indoor", "outdoor", "scene"]
    
    # Check for person-related terms
    for term in person_terms:
        if term in desc_lower:
            unwanted.append(term)
    
    # Check for accessories
    for term in accessory_terms:
        if term in desc_lower:
            unwanted.append(term)
    
    # Check for background
    for term in background_terms:
        if term in desc_lower:
            unwanted.append(term)
    
    # Remove duplicates and return
    return ", ".join(sorted(set(unwanted))) if unwanted else ""


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
    """
    Simplified prompting stage that generates detailed semantic prompts based on garment type only.
    Uses centralized prompts from config.prompts for consistency and easy debugging.
    No color extraction - just semantic type-based prompts with proper structure and details.
    
    JoyCaption description is used to enhance negative prompts by identifying unwanted elements.
    """
    if not selected_item:
        return selected_item, {}

    # Determine the garment type
    forced_type = requested_type if requested_type in {"top", "bottom", "dress", "outer"} else None
    selected_type = forced_type or normalize_garment_type(str(selected_item.get("type") or "")) or "top"
    
    # Get prompts from centralized configuration
    base_garment_prompt = get_flux2_positive_prompt(selected_type)
    avoid_prompt = get_flux2_negative_prompt(selected_type)
    
    # Get MiniCPM and JoyCaption descriptions
    minicpm_desc = selected_item.get("minicpm_description", "")
    joycaption_desc = selected_item.get("joycaption_description", "")
    
    # Enhance negative prompt with JoyCaption insights
    # JoyCaption helps identify unwanted elements (person, accessories, background)
    if joycaption_desc:
        # Extract unwanted terms from JoyCaption description
        unwanted_terms = _extract_unwanted_terms_from_joycaption(joycaption_desc, selected_type)
        if unwanted_terms:
            # Append to negative prompt
            avoid_prompt = f"{avoid_prompt}, {unwanted_terms}"
    
    # Set the prompts on the item
    selected_item["baseGarmentPrompt"] = base_garment_prompt
    selected_item["promptDescription"] = base_garment_prompt
    selected_item["description"] = base_garment_prompt
    selected_item["extractionAvoidClause"] = avoid_prompt
    selected_item["type"] = selected_type
    
    # Get category information
    sync_category = wardrobe_category_from_garment_type(selected_type, style=None)
    selected_item["primary_category_key"] = sync_category["primary_category_key"]
    selected_item["category_key"] = sync_category["category_key"]
    selected_item["style"] = sync_category["style"]
    
    # Build simplified metadata (no color information, includes MiniCPM + JoyCaption descriptions)
    garment_metadata = {
        "prompt": {
            "base_garment_prompt": base_garment_prompt,
            "prompt_description": base_garment_prompt,
            "avoid_clause": avoid_prompt,
            "minicpm_description": minicpm_desc,  # MiniCPM description
            "joycaption_description": joycaption_desc,  # JoyCaption description
        },
        "type": selected_type,
        "style": sync_category["style"],
        "primary_category_key": sync_category["primary_category_key"],
        "category_key": sync_category["category_key"],
        "prompt_source": "type_based_semantic_with_joycaption",
        # No color information
        "dominant_hexes": [],
        "accent_hexes": [],
        "color_hints": [],
        "color_profile": {},
    }
    
    selected_item["garmentMetadata"] = garment_metadata

    return selected_item, {
        "selected_type": selected_type,
        "prompt_description": base_garment_prompt,
        "avoid_prompt": avoid_prompt,
        "sync_category": sync_category,
        "garment_metadata": garment_metadata,
        "prompt_source": "type_based_semantic_with_joycaption",
        "minicpm_description": minicpm_desc,  # Pass through for extraction stage
        "joycaption_description": joycaption_desc,  # Pass through for extraction stage
    }
