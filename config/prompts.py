"""
Centralized prompt configuration for all AI models.

This module contains all prompts used in the analyze and tryon pipelines:
- MiniCPM garment description prompts
- MiniCPM person/outfit description prompts  
- Flux2 positive prompts (generation)
- Flux2 negative prompts (avoidance)
- JoyCaption negative prompts

All prompts are in one location for easy debugging and modification.
"""

from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# MINICPM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

MINICPM_GARMENT_DESCRIPTION_COMMON = """Describe only the requested garment type in the image and ignore every other garment. If multiple garments are visible, focus only on the requested type.
Strictly return one single precise and accurate garment paragraph of 60 to 120 words only, plain text only, not JSON, not schema, not label list, and not caption.
Use only visible garment construction facts. Strictly do not mention colors, person, mannequin, body, skin, hair, hands, pose, background, camera, room, props, or recommendations.
Do not add fluff, styling language, trend language, mood words, scene description, or aesthetic wording. If a detail is uncertain or not visible, omit it.
Do not infer hidden length, hidden panels, or unseen edges from garment priors.
Write concise, reconstruction-ready garment construction details that can be reused for extraction and virtual try-on."""


MINICPM_GARMENT_DESCRIPTION_BY_TYPE = {
    "top": """Top-only guidance: describe the upper edge or neckline, collar or lapel if visible, shoulder layout, sleeve or strap geometry, shoulder coverage, open chest areas, torso panel continuity, seam placement, panel shapes, bodice and waist structure, bust shaping, hem endpoint, fit, silhouette, closures, trims, ruching, pleats, boning, padding, lining, drape, stretch, stitching, and any other visible construction details. Describe asymmetry through construction terms only, such as one-shoulder, diagonal strap, single sleeve, uneven neckline, asymmetric drape, gathered shoulder detail, side cutout, or asymmetric hem etc.., without assigning a side. When an asymmetric feature is visible, name the feature itself rather than its direction. Do not describe lower-body features, skirts, pants, legs, or full-outfit structure.""",
    "bottom": """Bottom-only guidance: describe the waistband, rise, hip shaping, leg shape, skirt shape if present, inseam or split details, hem endpoint, closures, pockets, seams, panel structure, fit, silhouette, drape, stretch, and any other visible construction details. If the lower hem is cut off or obscured, say that the full length is not fully visible instead of guessing knee, ankle, or full-length coverage. Do not describe upper-body features, sleeves, collars, shoulders, or neckline details.""",
    "dress": """Dress-only guidance: describe the bodice, neckline or upper edge, shoulder layout, sleeve or strap geometry, waist transition, skirt construction, continuous body flow, hem endpoint, fit, silhouette, closures, ruching, pleats, boning, lining, drape, stretch, and any other visible construction details. Describe asymmetry through construction terms only, such as one-shoulder, diagonal strap, single sleeve, uneven neckline, asymmetric drape, gathered shoulder detail, side cutout, or asymmetric hem, without assigning a side. When an asymmetric feature is visible, name the feature itself rather than its direction. Do not split the dress into separate top and bottom garments or describe it as two pieces.""",
    "outer": """Outerwear-only guidance: describe the collar, lapel, opening, shoulder structure, sleeve length, cuff shape, layering behavior, front overlap, waistline, hem endpoint, fit, silhouette, closures, pockets, epaulets, lining, drape, structure, and any other visible construction details. If the lower hem is not fully visible, say that the outerwear length is partially visible or not fully visible instead of guessing a hidden full coat length. Do not describe inner garments as part of the outerwear.""",
}

MINICPM_GARMENT_DESCRIPTION_PROMPT = MINICPM_GARMENT_DESCRIPTION_COMMON

MINICPM_PERSON_OUTFIT_DESCRIPTION_PROMPT = """Describe only the human subject for identity-preserving virtual try-on.
Return exactly one clean sentence with at most 20 words.
Include only person type, hair style, body build, and pose style.
Do not use key-value sections, labels, JSON, bullets, or semicolon fields.
Do not mention left, right, side, viewer-left, viewer-right, or directional wording.
Do not describe clothing, outfit, background, lighting, camera, mood, or aesthetics.
If uncertain, omit the detail instead of guessing.
Output only the sentence."""

# ═══════════════════════════════════════════════════════════════════════════════
# FLUX2 POSITIVE PROMPTS (What TO Generate)
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_PRESERVATION_CLAUSE = (
    "Preserve the garment's original colors, print placement, and material appearance exactly as in the reference. "
    "Match the reference colors and print placement precisely. "
    "Do not recolor the garment or shift its hue, undertone, brightness, saturation, contrast, or shading."
)

FLUX2_POSITIVE_PROMPTS = {
    "top": (
        "A single top garment displayed alone as a standalone product, centered and front-view studio product photography "
        "on a seamless pure white backdrop. Clean, uncluttered composition with only the garment visible. "
        "Soft diffused studio lighting with clear edge definition and crisp detail. "
        "The neckline, shoulder line, sleeve geometry, waistline, and hem shape match the reference exactly. "
        "Keep the garment front panel flat and cloth-like with no torso volume, chest projection, or mannequin-shaped curvature beneath the fabric. "
        f"{COLOR_PRESERVATION_CLAUSE} "
        "The garment structure and overall silhouette match the reference exactly."
    ),
    "bottom": (
        "A single bottom garment displayed alone as a standalone product, centered and front-view studio product photography "
        "on a seamless pure white backdrop. Clean, uncluttered composition with only the garment visible. "
        "Soft diffused studio lighting with clear edge definition and crisp detail. "
        "The waistline, rise, hip shaping, leg shape, and hem length match the reference exactly. "
        f"{COLOR_PRESERVATION_CLAUSE} "
        "The garment structure and overall silhouette match the reference exactly."
    ),
    "dress": (
        "A single dress garment displayed alone as a standalone product, centered and front-view studio product photography "
        "on a seamless pure white backdrop. Clean, uncluttered composition with only the garment visible. "
        "Soft diffused studio lighting with clear edge definition and crisp detail. "
        "The bodice structure, neckline, waistline, skirt silhouette, and full length match the reference exactly. "
        f"{COLOR_PRESERVATION_CLAUSE} "
        "The garment structure and overall silhouette match the reference exactly."
    ),
    "outer": (
        "A single outerwear garment displayed alone as a standalone product, centered and front-view studio product photography "
        "on a seamless pure white backdrop. Clean, uncluttered composition with only the garment visible. "
        "Soft diffused studio lighting with clear edge definition and crisp detail. "
        "The collar, lapel, shoulder structure, sleeve length, waistline, and hem match the reference exactly. "
        f"{COLOR_PRESERVATION_CLAUSE} "
        "The garment structure and overall silhouette match the reference exactly."
    ),
}

# Analyze-only positive clause to suppress human/body leakage in extraction outputs.
ANALYZE_GARMENT_ONLY_CLAUSE = (
    "Only the garment is visible; no person, no mannequin body, no limbs, no head, no skin, no body silhouette. "
    "Ghost mannequin style is acceptable only if the mannequin is completely invisible. "
    "Flat garment presentation; hollow interior: no chest volume, no torso form, no body contours beneath the fabric. "
    "No hangers, no props, no accessories. Pure white background, clean studio product photo."
)

# Top-only refinement to prevent extra fabric above neckline or below hem.
# ANALYZE_TOP_ONLY_CLAUSE = (
#     "Top-only: no fabric above the garment's upper edge, no fabric below the hem, no lower-body garments or legs visible. "
#     "Crop to the garment bounds only; no extra fabric beyond the original top silhouette. "
#     "Keep the front panel flat and cloth-like; do not introduce torso curvature, chest projection, or mannequin form beneath the fabric. "
#     "No added panels, yokes, underlayers, or secondary garment sections above the neckline or below the hem. "
#     "If the source is one-shoulder, single-sleeve, or otherwise asymmetrical, preserve exactly one bare shoulder and one sleeve or strap layout only. Do not normalize it into a standard two-sleeve top. "
#     "Do not invent a second shoulder, mirrored strap, extra sleeve cap, or seam bridge across the open side. "
#     "Do not symmetrize the neckline or complete the missing side of the garment. "
#     "Keep the garment color uniform across the asymmetrical panels; do not add a faded, worn, or washed-out secondary shade on the exposed shoulder or sleeve. "
#     "Do not add padding, cups, or extra bust volume; keep bust shaping, seams, and underbust placement exactly as the reference. "
#     "Do not create a detached waistband, extra lower strip, separate abdominal band, or second garment section below the hem. "
#     "Keep the garment as one continuous top with the original cropped hem and asymmetrical shoulder structure."
# )

ANALYZE_TOP_ONLY_CLAUSE = (
    "Top-only: no fabric above the garment's upper edge, no fabric below the hem, no lower-body garments or legs visible. "
    "Crop to the garment bounds only; no extra fabric beyond the original top silhouette. "
    "Keep the front panel flat and cloth-like; do not introduce torso curvature, chest projection, or mannequin form beneath the fabric. "
    "No added panels, yokes, underlayers, or secondary garment sections above the neckline or below the hem. "
    "Do not change the shoulder layout, sleeve count, strap count, or neckline asymmetry from the reference. "
    "Do not invent a second shoulder, mirrored strap, mirrored sleeve, seam bridge, or new asymmetrical cut. "
    "Preserve the exact upper edge of the garment; if the source neckline is low, wide, or exposes the upper chest, keep that same neckline depth and opening. Do not raise it into a higher-coverage chest panel or a more modest scoop top. "
    "If the source top ends below the bust or has a cropped hem, preserve that exact short hem position and do not extend the garment into the abdomen, waist, or full torso length. "
    "Preserve the exact garment color tone and shading from the reference crop; do not warm it, cool it, bleach it, brighten it, darken it, or make it more saturated. "
    "Do not add padding, cups, or extra bust volume; keep bust shaping, seams, and underbust placement exactly as the reference."
)

ANALYZE_TOP_SUBTYPE_ONLY_CLAUSES = {
    "bust_band_top": (
        "Bust-band top only: keep the exposed upper chest and exposed lower torso exactly as in the source. "
        "Do not add a continuous chest panel above the visible bust band. "
        "Do not add a shoulder yoke, clavicle panel, or fabric bridge across the upper chest. "
        "Keep long sleeves attached from the side-bust or underarm structure when that is how they appear in the source. "
        "Do not extend the garment into a longer torso panel below the underbust or cropped hem."
    ),
    "cropped_panel_top": (
        "Cropped-panel top only: keep the short front torso panel exactly as in the reference. "
        "Do not shorten it into a narrow bra band and do not extend it into full torso length."
    ),
    "structured_corset_top": (
        "Structured corset top only: keep the strapless sleeveless upper edge exactly as in the source. "
        "Do not add sleeves, straps, shoulder coverage, or a shoulder yoke when they are not present. "
        "Keep the structured boning and extended torso panel intact; do not collapse it into an underbust band or a standard tee-like top."
    ),
    "asymmetric_top": (
        "Asymmetric top only: preserve the exact open side, shoulder exposure, and unmatched sleeve or strap arrangement. "
        "Do not mirror the missing side and do not regularize the top into a symmetric chest panel."
    ),
    "standard_top": (
        "Standard top only: keep the garment as a continuous symmetric top with the original neckline, torso panel, and hem coverage. "
        "Do not convert it into a bra band, bandeau, or asymmetrical cut."
    ),
}

ANALYZE_BOTTOM_ONLY_CLAUSE = (
    "Bottom-only: no shirt, no blouse, no top, no torso section, and no upper-body garment visible above the waistband. "
    "Crop to the bottom bounds only; keep the garment as one continuous bottom garment with the original rise, waistband, hips, and hem. "
    "Do not add a detached upper panel, upper-body panel, extra waistband, or torso-shaped section above the rise. "
    "Do not split the garment into separate pieces or introduce skirt-like and pant-like sections in the same item. "
    "Keep the waist and hip area flat and cloth-like rather than body-shaped."
)

ANALYZE_DRESS_ONLY_CLAUSE = (
    "Dress-only: no separate top, no separate skirt, no two-piece outfit look, and no added waistband between bodice and skirt. "
    "Keep the dress as one continuous garment from bodice to hem with the original waist transition preserved in a single piece. "
    "Do not split it into a separate top and skirt or split the dress into bodice and skirt as if they were separate garments. "
    "Keep the bodice and skirt flow smooth, flat, and cloth-like rather than body-shaped."
)

ANALYZE_OUTER_ONLY_CLAUSE = (
    "Outerwear-only: no inner base garment, no inner base layer, no shirt underneath, no separate underlayer, and no extra garment visible below the coat or jacket. "
    "Keep the outerwear as one single layer with the original collar, lapel, opening, sleeves, and hem preserved. "
    "Do not add a second garment inside the silhouette or convert the outerwear into a layered outfit. "
    "Keep the front and body shape flat and cloth-like rather than torso-shaped."
)

ANALYZE_TYPE_ONLY_CLAUSES = {
    "top": ANALYZE_TOP_ONLY_CLAUSE,
    "bottom": ANALYZE_BOTTOM_ONLY_CLAUSE,
    "dress": ANALYZE_DRESS_ONLY_CLAUSE,
    "outer": ANALYZE_OUTER_ONLY_CLAUSE,
}

# ═══════════════════════════════════════════════════════════════════════════════
# FLUX2 NEGATIVE PROMPTS (What NOT to Generate)
# ═══════════════════════════════════════════════════════════════════════════════

# Common negative elements for all garment types
FLUX2_NEGATIVE_COMMON = (
    "person, human body, face, eyes, hair, skin, neck, shoulders, chest, torso, hands, fingers, arms, legs, feet, toes, "
    "mannequin, hanger, background scene, props, accessories, bag, jewelry, watermark, text, logo, "
    "human silhouette, skin patch, limb fragment, body fragment, arm fragment, leg fragment, "
    "wrong garment color, recolored fabric, color palette change, hue shift, saturation drift, value drift, "
    "desaturated, oversaturated, washed out, color swap, "
    "pattern drift, print swap, texture swap, material swap, material change, "
    "wrong garment category, extra garment, duplicate garment, merged garments, "
    "broken silhouette, distorted seams, wrong neckline, wrong sleeve length, wrong waistline, wrong hem length, "
    "artificial variant, redesigned garment, alternate style, "
    "blur, low detail, overexposed, underexposed, multiple garments, background patterns, cropped edges, partial view"
)

# Type-specific negative elements
FLUX2_NEGATIVE_TYPE_SPECIFIC = {
    "top": "pants, trousers, skirt, shorts, dress, lower-body garment, shoes, lower-body structure, bottom garment overlay, second shoulder, mirrored shoulder, second strap, mirrored strap, second sleeve, symmetric neckline",
    "bottom": "shirt, blouse, t-shirt, jacket, hoodie, dress, upper-body garment, torso garment overlay, sleeves, detached upper panel, extra waistband, split garment, body-shaped upper section",
    "dress": "two-piece outfit, separate top and bottom, skirt with separate blouse, trouser plus shirt combo, incomplete dress replacement, split bodice, split hemline, detached bodice, detached skirt, extra waistband",
    "outer": "inner garments, underwear, base layers, bottom wear, pants, inner shirt, base layer, detached underlayer, second garment underneath, padded torso",
}

# Combined negative prompts (common + type-specific)
FLUX2_NEGATIVE_PROMPTS = {
    garment_type: f"{FLUX2_NEGATIVE_COMMON}, {type_specific}"
    for garment_type, type_specific in FLUX2_NEGATIVE_TYPE_SPECIFIC.items()
}

# ═══════════════════════════════════════════════════════════════════════════════
# FLUX2 TRYON NEGATIVE PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

# Default negative prompt for try-on generation
# NOTE: Negative prompts require true_cfg_scale > 1.0 to work in Flux2
# If true_cfg_scale <= 1.0, negative prompts are appended to positive prompt as fallback
FLUX2_TRYON_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed body, extra limbs, extra fingers, wrong hands, "
    "extra hand, third hand, duplicate arms, hand fused to garment, garment fused to skin, "
    "identity change, different face, wrong skin tone, recolored garment, hue shift, "
    "color drift, pattern drift, texture swap, logo/text watermark, duplicate garment, "
    "layering artifacts, garment merge, ghost garment, incorrect neckline, incorrect hemline"
)

# ═══════════════════════════════════════════════════════════════════════════════
# JOYCAPTION NEGATIVE PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

# Negative prompt for JoyCaption model (used in descriptor generation)
JOYCAPTION_NEGATIVE_PROMPT = (
    "person, human, model, mannequin, face, body, skin, hands, arms, legs, "
    "background scene, room, wall, floor, furniture, props, "
    "watermark, text, logo, brand name, "
    "multiple garments, garment overlay, layered clothing"
)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_minicpm_garment_type(garment_type: Optional[str]) -> str:
    raw = " ".join(str(garment_type or "").split()).strip().lower()
    if raw in {"outer", "outerwear", "coat", "jacket", "blazer", "trench", "trench coat"}:
        return "outer"
    if raw in {"bottom", "pants", "trousers", "jeans", "skirt", "shorts", "lower"}:
        return "bottom"
    if raw in {"dress", "gown", "one-piece"}:
        return "dress"
    if raw in {"top", "shirt", "blouse", "tee", "t-shirt", "tshirt", "bra", "bralette", "corset", "corset top"}:
        return "top"
    return ""


def get_minicpm_garment_prompt(garment_type: Optional[str] = None) -> str:
    """Get the MiniCPM garment description prompt."""
    prompt_type = _normalize_minicpm_garment_type(garment_type)
    type_clause = MINICPM_GARMENT_DESCRIPTION_BY_TYPE.get(prompt_type, "")
    if type_clause:
        return f"{MINICPM_GARMENT_DESCRIPTION_COMMON} {type_clause}".strip()
    return MINICPM_GARMENT_DESCRIPTION_COMMON


def get_minicpm_person_outfit_prompt() -> str:
    """Get the MiniCPM person/outfit description prompt for try-on."""
    return MINICPM_PERSON_OUTFIT_DESCRIPTION_PROMPT


def get_flux2_positive_prompt(garment_type: str) -> str:
    """
    Get the Flux2 positive prompt for a specific garment type.
    
    Args:
        garment_type: One of "top", "bottom", "dress", "outer"
        
    Returns:
        Flux2 positive prompt string
    """
    return FLUX2_POSITIVE_PROMPTS.get(
        garment_type,
        f"Generate only a garment on pure white background, professional product photography. {COLOR_PRESERVATION_CLAUSE}"
    )


def get_analyze_flux2_positive_prompt(garment_type: str) -> str:
    """
    Get the Flux2 positive prompt for analyze extraction only.
    Appends the analyze-only garment clause to reduce human artifacts.
    """
    base = get_flux2_positive_prompt(garment_type)
    extra = ANALYZE_TYPE_ONLY_CLAUSES.get(garment_type, "")
    return f"{base} {ANALYZE_GARMENT_ONLY_CLAUSE} {extra}".strip()


def get_analyze_top_subtype_clause(top_subtype: str) -> str:
    """Return an additional analyze guard for a routed top subtype."""
    return ANALYZE_TOP_SUBTYPE_ONLY_CLAUSES.get(str(top_subtype or "").strip().lower(), "")


def get_flux2_negative_prompt(garment_type: str) -> str:
    """
    Get the Flux2 negative prompt for a specific garment type.
    
    Args:
        garment_type: One of "top", "bottom", "dress", "outer"
        
    Returns:
        Flux2 negative prompt string
    """
    return FLUX2_NEGATIVE_PROMPTS.get(
        garment_type,
        f"{FLUX2_NEGATIVE_COMMON}, person, human, model, mannequin, accessories, multiple garments"
    )


def get_flux2_tryon_negative_prompt() -> str:
    """
    Get the default Flux2 negative prompt for try-on generation.
    
    NOTE: Negative prompts require true_cfg_scale > 1.0 to work in Flux2.
    If true_cfg_scale <= 1.0, the negative prompt is appended to the positive prompt as fallback.
    
    Returns:
        Flux2 try-on negative prompt string
    """
    return FLUX2_TRYON_NEGATIVE_PROMPT


def get_joycaption_negative_prompt() -> str:
    """Get the JoyCaption negative prompt for descriptor generation."""
    return JOYCAPTION_NEGATIVE_PROMPT


def get_all_prompts_for_type(garment_type: str) -> dict:
    """
    Get all prompts for a specific garment type.
    
    Args:
        garment_type: One of "top", "bottom", "dress", "outer"
        
    Returns:
        Dictionary with all prompts:
        {
            "minicpm_garment": str,
            "minicpm_person": str,
            "flux2_positive": str,
            "flux2_negative": str,
            "flux2_tryon_negative": str,
            "joycaption_negative": str,
        }
    """
    return {
        "minicpm_garment": get_minicpm_garment_prompt(garment_type),
        "minicpm_person": get_minicpm_person_outfit_prompt(),
        "flux2_positive": get_flux2_positive_prompt(garment_type),
        "flux2_negative": get_flux2_negative_prompt(garment_type),
        "flux2_tryon_negative": get_flux2_tryon_negative_prompt(),
        "joycaption_negative": get_joycaption_negative_prompt(),
    }
