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

MINICPM_GARMENT_DESCRIPTION_PROMPTS = {
    "top": """Task: Describe only the TOP garment with evidence-based construction detail.

Output format:
- Return exactly one valid JSON object with exactly 2 keys:
  1) "category_type": string
  2) "garment_construction_prompt": string
- Output only JSON. No extra text.

Rules for "category_type":
- Use an accurate type from visible evidence only for top garment.

Rules for "garment_construction_prompt":
- Exactly one line, plain text, 40 to 75 words.
- Natural sentence flow.
- Focus mainly on top category/subtype, upper-edge/neckline/opening shape, sleeve/strap/panel configuration, shoulder coverage, torso silhouette/construction, hem shape/endpoint, and at least one standout visible detail; optional texture/motif with garment-zone placement.
- Use only high-confidence evidence; skip weak details.

Hard constraints:
- Only directly visible top-garment evidence.
- No guessing or inferred hidden structure.
- No directional/side terms.
- Forbidden terms: left, right, side, one side, opposite side, front, back, viewer-left, viewer-right.
- No color terms.
- No body/background/camera/styling terms.
- No non-top garment or hardware details.
- No non-visible region details.
- No absence phrasing: no, not visible, without, absent, unknown.
- No repeated facts.

Disambiguation:
- Use "strap" only for a clearly separate thin support band.
- If upper edge is broad/pleated/draped, use panel/band wording.
- If asymmetry is visible, describe it without side words.

Final safety rewrite:
- Before final answer, replace any directional wording with garment-only phrasing.""",
    "bottom": """Task: Describe only the BOTTOM garment with evidence-based construction detail.

Output format:
- Return exactly one valid JSON object with exactly 2 keys:
  1) "category_type": string
  2) "garment_construction_prompt": string
- Output only JSON. No extra text.

Rules for "category_type":
- Use a conservative bottom subtype from visible evidence only.

Rules for "garment_construction_prompt":
- Exactly one line, plain text, 40 to 75 words.
- Natural sentence flow.
- Focus mainly on bottom category/subtype, waistband/rise/opening shape, leg or skirt panel configuration, torso-to-hip transition, silhouette/construction, hem shape/endpoint, and at least one standout visible detail; optional texture/motif with garment-zone placement.
- Use only high-confidence evidence; skip weak details.

Hard constraints:
- Only directly visible bottom-garment evidence.
- No guessing or inferred hidden structure.
- No directional/side terms.
- Forbidden terms: left, right, side, one side, opposite side, front, back, viewer-left, viewer-right.
- No color terms.
- No body/background/camera/styling terms.
- No non-bottom garment or hardware details.
- No non-visible region details.
- No absence phrasing: no, not visible, without, absent, unknown.
- No repeated facts.

Disambiguation:
- Use "strap" only for a clearly separate thin support band.
- If asymmetry is visible, describe it without side words.

Final safety rewrite:
- Before final answer, replace any directional wording with garment-only phrasing.""",
    "dress": """Task: Describe only the DRESS garment with evidence-based construction detail.

Output format:
- Return exactly one valid JSON object with exactly 2 keys:
  1) "category_type": string
  2) "garment_construction_prompt": string
- Output only JSON. No extra text.

Rules for "category_type":
- Use a conservative dress subtype from visible evidence only.

Rules for "garment_construction_prompt":
- Exactly one line, plain text, 40 to 75 words.
- Natural sentence flow.
- Focus mainly on dress category/subtype, upper-edge/neckline/opening shape, sleeve/strap/panel configuration, shoulder coverage, bodice-to-skirt continuity as one garment, silhouette/construction, hem shape/endpoint, and at least one standout visible detail; optional texture/motif with garment-zone placement.
- Use only high-confidence evidence; skip weak details.

Hard constraints:
- Only directly visible dress-garment evidence.
- No guessing or inferred hidden structure.
- No directional/side terms.
- Forbidden terms: left, right, side, one side, opposite side, front, back, viewer-left, viewer-right.
- No color terms.
- No body/background/camera/styling terms.
- No non-dress garment or hardware details.
- No non-visible region details.
- No absence phrasing: no, not visible, without, absent, unknown.
- No repeated facts.

Disambiguation:
- Use "strap" only for a clearly separate thin support band.
- If upper edge is broad/pleated/draped, use panel/band wording.
- If asymmetry is visible, describe it without side words.

Final safety rewrite:
- Before final answer, replace any directional wording with garment-only phrasing.""",
    "outer": """Task: Describe only the OUTER garment with evidence-based construction detail.

Output format:
- Return exactly one valid JSON object with exactly 2 keys:
  1) "category_type": string
  2) "garment_construction_prompt": string
- Output only JSON. No extra text.

Rules for "category_type":
- Use a conservative outer subtype from visible evidence only.

Rules for "garment_construction_prompt":
- Exactly one line, plain text, 40 to 75 words.
- Natural sentence flow.
- Focus mainly on outer category/subtype, upper-edge/neckline/opening shape, sleeve/strap/panel configuration, shoulder coverage, torso silhouette/construction, hem shape/endpoint, and at least one standout visible detail; optional texture/motif with garment-zone placement.
- Use only high-confidence evidence; skip weak details.

Hard constraints:
- Only directly visible outer-garment evidence.
- No guessing or inferred hidden structure.
- No directional/side terms.
- Forbidden terms: left, right, side, one side, opposite side, front, back, viewer-left, viewer-right.
- No color terms.
- No body/background/camera/styling terms.
- No non-outer garment or hardware details.
- No non-visible region details.
- No absence phrasing: no, not visible, without, absent, unknown.
- No repeated facts.

Disambiguation:
- Use "strap" only for a clearly separate thin support band.
- If upper edge is broad/pleated/draped, use panel/band wording.
- If asymmetry is visible, describe it without side words.

Final safety rewrite:
- Before final answer, replace any directional wording with garment-only phrasing.""",
}

MINICPM_GARMENT_DESCRIPTION_PROMPT = MINICPM_GARMENT_DESCRIPTION_PROMPTS["top"]

MINICPM_PERSON_OUTFIT_DESCRIPTION_PROMPT = """Analyze the image for user-image preparation and return JSON only with this exact schema:
{"garments":["top","bottom","outer","dress"],"person_type":"","age_band":"","hair_style":"","hair_length":"","hair_color":"","head_covering":"","body_build":""}
Rules:
- garments: include only garment categories worn on the main body of the dominant person.
- Allowed garment values: top, bottom, outer, dress.
- garments may include multiple values; do not include duplicates.
- Include garments only when they are physically worn on the person's body (torso, lower body, or outer layer).
- Do not include garments that are held in hands, carried, draped over arms/shoulders without being worn, placed nearby, or present in the background.
- Category definitions:
  - top: upper-body garment piece only.
  - bottom: lower-body garment piece only.
  - dress: one continuous one-piece garment spanning upper and lower body.
  - outer: outermost layer worn over base garment(s).
- Base-outfit exclusivity:
  - Use either top+bottom or dress as base outfit.
  - Do not output dress together with top or bottom for the same worn outfit.
  - outer may appear with either base-outfit form.
- Do not include side objects, hand-held items, nearby garments, or background apparel.
- person_type: person label only (for example woman, man, person, girl, boy, etc.). These are examples for idea, not an exhaustive list. No clothing words.
- age_band: short age hint only (for example young, adult, middle-aged, older, etc.). These are examples for idea, not an exhaustive list. Empty string if unclear.
- hair_style: choose exactly one value from this closed list only: loose, wavy, curly, coily, braided, twists, locs, bun, ponytail, pigtails, half-up, updo, afro. Do not generate any other value. Do not use smooth or straight. Use ponytail only when a clearly tied-back ponytail structure is visible. Use pigtails only when two clearly separated tied sections are visible. Use half-up only when part of the hair is tied up and the rest is left down. Use loose when the hair is visibly open/untied and no other closed-list style fits better. If none of these labels clearly fits, return empty string.
- hair_length: choose exactly one value from this closed list only: bald/shaved, very short, short, medium, long, very long. Do not generate any other value. Do not use shoulder-length. If the visible length is around the shoulder area, use medium. If unclear, return empty string.
- hair_color: short hair-color phrase only (for example black, brown, blonde, auburn, gray, white, dyed, multi-tone, gradient, etc.). These are examples for idea, not an exhaustive list. If gradient/highlights/ombre are visible, return a short phrase such as multi-tone or gradient (optionally with one dominant color). Empty string if unclear.
- head_covering: short visibility state only (for example uncovered, partially-covered, fully-covered, unknown). Empty string if unclear.
- If hair is covered by hijab, scarf, headscarf, veil, cap, hat, hoodie, or any head covering and hair is not clearly visible, set hair_style="", hair_length="", and hair_color="". Do not infer hidden hair.
- If hair is clearly visible, do not leave hair_length empty. hair_style may be empty when no closed-list style is clearly identifiable.
- body_build: short body-build phrase only (for example slim, athletic, curvy, average, petite, plus-size, etc.). These are examples for idea, not an exhaustive list. Empty string if unclear.
- Do not include pose/posture terms (standing, sitting, kneeling, etc.).
- Do not include clothing details, colors, background, lighting, camera, mood, or aesthetics in person fields.
- Do not mention left, right, side, viewer-left, viewer-right, or directional wording in person fields.
- Never include these tokens in person fields: wearing, sunglasses, glasses, eyewear, none, smooth, straight, shoulder-length.
- You may use any other visible cues in the image internally (etc.) to infer these fields more accurately, but do not output those extra cues.
- Output only the schema fields above. Do not add prompt, description, notes, explanations, or any extra keys.
- If uncertain, return empty string for that field.
- Return only valid JSON, no markdown, no extra keys."""

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
    if prompt_type in MINICPM_GARMENT_DESCRIPTION_PROMPTS:
        return MINICPM_GARMENT_DESCRIPTION_PROMPTS[prompt_type]
    return MINICPM_GARMENT_DESCRIPTION_PROMPTS["top"]


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
