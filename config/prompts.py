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

# ═══════════════════════════════════════════════════════════════════════════════
# MINICPM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

MINICPM_GARMENT_DESCRIPTION_PROMPT = """Describe only the product garment for high-fidelity virtual try-on. 
Return exactly one single line with this schema: 
category=<dress|top|bottom|outerwear|set|unknown>; 
type=<specific garment type>; 
neckline=<neckline style>; 
collar=<collar style>; 
lapel=<lapel style>; 
shoulder_style=<shoulder/strap style>; 
asymmetry=<symmetric|asymmetric|one-shoulder|off-shoulder|single-sleeve|unknown>; 
sleeves=<sleeve style or length>; 
cuffs=<cuff style>; 
bodice_cut=<bodice or torso shaping>; 
waistline=<waist placement or shaping>; 
silhouette=<fit and overall shape>; 
length_hem=<hem length>; 
rise=<rise or waistband height for bottoms>; 
leg_shape=<leg opening shape if applicable>; 
skirt_style=<skirt shape if applicable>; 
fabric_texture=<fabric/material texture>; 
pattern=<print or pattern type>; 
embellishments=<buttons, zipper, pleats, ruffles, lace, embroidery, pockets, slit, logo>; 
closure=<closure type/placement>; 
pockets=<pocket type/placement>; 
slits=<slit type/placement>; 
straps=<strap details if applicable>; 
layering=<layers or overlays if present>; 
special_details=<unique construction or design details>; 
preserve=<state that garment structure and details must remain unchanged>. 
Use 'unknown' only when truly not visible. Prefer a concrete value if it can be inferred from visible pixels.
Do not mention colors, person, mannequin, background, camera, or recommendations."""

MINICPM_PERSON_OUTFIT_DESCRIPTION_PROMPT = """Describe only the human subject for virtual try-on identity preservation. 
Return exactly one single line with this schema: 
identity=<face shape/features, skin tone, hair style/color, age band>; 
body_pose=<pose, camera angle, visible limbs>; 
framing_lighting=<framing, crop, light direction/intensity>; 
occlusion=<hair/hands/accessories/objects overlapping body regions>; 
preserve=<face identity, skin tone, hair, body proportions, pose, framing, and lighting should remain unchanged>. 
Do not describe background or current clothing unless it creates an occlusion. 
Be factual from visible pixels only; use 'unknown' for hidden details."""

# ═══════════════════════════════════════════════════════════════════════════════
# FLUX2 POSITIVE PROMPTS (What TO Generate)
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_PRESERVATION_CLAUSE = (
    "Preserve the garment's original colors, print placement, and material appearance exactly as in the reference. "
    "Match the reference colors and print placement precisely."
)

FLUX2_POSITIVE_PROMPTS = {
    "top": (
        "A single top garment displayed alone as a standalone product, centered and front-view studio product photography "
        "on a seamless pure white backdrop. Clean, uncluttered composition with only the garment visible. "
        "Soft diffused studio lighting with clear edge definition and crisp detail. "
        "The neckline, shoulder line, sleeve geometry, waistline, and hem shape match the reference exactly. "
        "If the reference is asymmetrical, preserve the asymmetry exactly; if it is one-shoulder or single-sleeve, keep one bare shoulder and one sleeve only. "
        "Do not add a second shoulder, second strap, mirrored sleeve, or symmetric neckline on the open side. "
        "Keep the open shoulder truly open and preserve the one-sided silhouette exactly as captured. "
        "Keep one consistent source color family across all panels and drapes; do not introduce a second faded, aged, or shadow-tinted color on the exposed shoulder, sleeve, or hem. "
        "Keep the garment front panel flat and cloth-like with no torso volume, chest projection, or mannequin-shaped curvature beneath the fabric. "
        f"{COLOR_PRESERVATION_CLAUSE} "
        "The garment structure and overall silhouette match the reference exactly."
    ),
    "bottom": (
        "A single bottom garment displayed alone as a standalone product, centered and front-view studio product photography "
        "on a seamless pure white backdrop. Clean, uncluttered composition with only the garment visible. "
        "Soft diffused studio lighting with clear edge definition and crisp detail. "
        "The waistline, rise, hip shaping, leg shape, and hem length match the reference exactly. "
        "Keep the bottom as one continuous bottom garment with no detached upper panel, no upper-body panel, no torso-shaped top section, and no extra waistband layered above the original rise. "
        "Do not split the garment into separate pants and skirt-like sections or add an extra fabric band above the waistband. "
        "Keep the hip and leg geometry clean and cloth-like; do not add body-shaped curvature above the waistband. "
        f"{COLOR_PRESERVATION_CLAUSE} "
        "The garment structure and overall silhouette match the reference exactly."
    ),
    "dress": (
        "A single dress garment displayed alone as a standalone product, centered and front-view studio product photography "
        "on a seamless pure white backdrop. Clean, uncluttered composition with only the garment visible. "
        "Soft diffused studio lighting with clear edge definition and crisp detail. "
        "The bodice structure, neckline, waistline, skirt silhouette, and full length match the reference exactly. "
        "Keep the dress as one continuous garment from bodice to hem; do not split it into a separate top and skirt. "
        "Preserve the waist transition as part of the same garment and do not add a detached waistband, second skirt layer, or two-piece outfit look. "
        "Keep the bodice-to-skirt flow smooth and cloth-like with no body-shaped volume under the waist. "
        f"{COLOR_PRESERVATION_CLAUSE} "
        "The garment structure and overall silhouette match the reference exactly."
    ),
    "outer": (
        "A single outerwear garment displayed alone as a standalone product, centered and front-view studio product photography "
        "on a seamless pure white backdrop. Clean, uncluttered composition with only the garment visible. "
        "Soft diffused studio lighting with clear edge definition and crisp detail. "
        "The collar, lapel, shoulder structure, sleeve length, waistline, and hem match the reference exactly. "
        "Keep the outerwear as one single layer with no inner base garment, no detached underlayer, and no extra body-shaped section beneath the fabric. "
        "Preserve the open front, closure, or layering behavior exactly as shown, but do not invent a second garment underneath or outside the original silhouette. "
        "Keep the coat or jacket structure clean, flat, and cloth-like rather than padded or torso-shaped. "
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
ANALYZE_TOP_ONLY_CLAUSE = (
    "Top-only: no fabric above the garment's upper edge, no fabric below the hem, no lower-body garments or legs visible. "
    "Crop to the garment bounds only; no extra fabric beyond the original top silhouette. "
    "Keep the front panel flat and cloth-like; do not introduce torso curvature, chest projection, or mannequin form beneath the fabric. "
    "No added panels, yokes, underlayers, or secondary garment sections above the neckline or below the hem. "
    "If the source is one-shoulder, single-sleeve, or otherwise asymmetrical, preserve exactly one bare shoulder and one sleeve or strap layout only. Do not normalize it into a standard two-sleeve top. "
    "Do not invent a second shoulder, mirrored strap, extra sleeve cap, or seam bridge across the open side. "
    "Do not symmetrize the neckline or complete the missing side of the garment. "
    "Keep the garment color uniform across the asymmetrical panels; do not add a faded, worn, or washed-out secondary shade on the exposed shoulder or sleeve. "
    "Do not add padding, cups, or extra bust volume; keep bust shaping, seams, and underbust placement exactly as the reference. "
    "Do not create a detached waistband, extra lower strip, separate abdominal band, or second garment section below the hem. "
    "Keep the garment as one continuous top with the original cropped hem and asymmetrical shoulder structure."
)

ANALYZE_BOTTOM_ONLY_CLAUSE = (
    "Bottom-only: no shirt, no blouse, no top, no torso section, and no upper-body garment visible above the waistband. "
    "Crop to the bottom bounds only; keep the garment as one continuous bottom with the original rise, waistband, hips, and hem. "
    "Do not add a detached upper panel, extra waistband, or torso-shaped section above the rise. "
    "Do not split the garment into separate pieces or introduce skirt-like and pant-like sections in the same item. "
    "Keep the waist and hip area flat and cloth-like rather than body-shaped."
)

ANALYZE_DRESS_ONLY_CLAUSE = (
    "Dress-only: no separate top, no separate skirt, no two-piece outfit look, and no added waistband between bodice and skirt. "
    "Keep the dress as one continuous garment from neckline through hem with the original waist transition preserved in a single piece. "
    "Do not split the dress into bodice and skirt as if they were separate garments. "
    "Keep the bodice and skirt flow smooth, flat, and cloth-like rather than body-shaped."
)

ANALYZE_OUTER_ONLY_CLAUSE = (
    "Outerwear-only: no inner base layer, no shirt underneath, no separate underlayer, and no extra garment visible below the coat or jacket. "
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

def get_minicpm_garment_prompt() -> str:
    """Get the MiniCPM garment description prompt."""
    return MINICPM_GARMENT_DESCRIPTION_PROMPT


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
        "minicpm_garment": get_minicpm_garment_prompt(),
        "minicpm_person": get_minicpm_person_outfit_prompt(),
        "flux2_positive": get_flux2_positive_prompt(garment_type),
        "flux2_negative": get_flux2_negative_prompt(garment_type),
        "flux2_tryon_negative": get_flux2_tryon_negative_prompt(),
        "joycaption_negative": get_joycaption_negative_prompt(),
    }
