from typing import Dict, Optional, Tuple

CLOTHING_STYLES = [
    "tee", "long sleeve t-shirt", "sleeveless t-shirt", "polo shirt",
    "tank top", "camisole", "crop top", "blouse", "shirt",
    "sweatshirt", "hoodie", "sweater", "sweater vest", "sports top",
    "bodysuit", "knitwear", "cardigan", "jeans", "trousers", "dress pants",
    "track pants", "leggings", "sweatpants", "shorts", "skirt", "mini skirt",
    "midi skirt", "maxi skirt", "coat", "trench coat", "fur coat", "shearling coat",
    "blazer", "suit jacket", "jacket", "blouson", "varsity jacket", "trucker jacket",
    "biker jacket", "field jacket", "sports jacket", "fleece jacket",
    "parka", "down jacket", "puffer jacket", "vest", "mini dress", "day dress",
    "t-shirt dress", "shirt dress", "sweatshirt dress", "sweater dress",
    "jacket dress", "suspender dress", "party dress", "maxi dress", "jumpsuit"
]

STYLE_TO_CATEGORY_KEYS = {
    "tee": ("tops", "t_shirts"),
    "long sleeve t-shirt": ("tops", "long_sleeve_t_shirts"),
    "sleeveless t-shirt": ("tops", "sleeveless_t_shirts"),
    "polo shirt": ("tops", "polo_shirts"),
    "tank top": ("tops", "tanks_and_camis"),
    "camisole": ("tops", "tanks_and_camis"),
    "crop top": ("tops", "crop_tops"),
    "blouse": ("tops", "blouses"),
    "shirt": ("tops", "shirts"),
    "sweatshirt": ("tops", "sweatshirts"),
    "hoodie": ("outerwear", "zip_up_hoodies"),
    "sweater": ("tops", "sweaters"),
    "sweater vest": ("tops", "sweater_vests"),
    "sports top": ("tops", "sports_tops"),
    "bodysuit": ("tops", "bodysuits"),
    "knitwear": ("tops", "knitwear"),
    "cardigan": ("outerwear", "cardigans"),
    "jeans": ("bottoms", "jeans"),
    "trousers": ("bottoms", "trousers"),
    "dress pants": ("bottoms", "dress_pants"),
    "track pants": ("bottoms", "track_pants"),
    "leggings": ("bottoms", "leggings"),
    "sweatpants": ("bottoms", "sweatpants"),
    "shorts": ("bottoms", "shorts"),
    "skirt": ("skirts", "midi_skirts"),
    "mini skirt": ("skirts", "mini_skirts"),
    "midi skirt": ("skirts", "midi_skirts"),
    "maxi skirt": ("skirts", "maxi_skirts"),
    "day dress": ("dresses", "day_dresses"),
    "t-shirt dress": ("dresses", "t_shirt_dresses"),
    "shirt dress": ("dresses", "shirt_dresses"),
    "sweatshirt dress": ("dresses", "sweatshirt_dresses"),
    "sweater dress": ("dresses", "sweater_dresses"),
    "jacket dress": ("dresses", "jacket_dresses"),
    "suspender dress": ("dresses", "suspender_dresses"),
    "party dress": ("dresses", "party_dresses"),
    "mini dress": ("dresses", "mini_dresses"),
    "maxi dress": ("dresses", "maxi_dresses"),
    "jumpsuit": ("dresses", "jumpsuits"),
    "coat": ("outerwear", "coats"),
    "trench coat": ("outerwear", "trench_coats"),
    "fur coat": ("outerwear", "fur_coats"),
    "shearling coat": ("outerwear", "shearling_coats"),
    "blazer": ("outerwear", "blazers"),
    "suit jacket": ("outerwear", "blazers"),
    "jacket": ("outerwear", "jackets"),
    "blouson": ("outerwear", "blousons"),
    "varsity jacket": ("outerwear", "varsity_jackets"),
    "trucker jacket": ("outerwear", "trucker_jackets"),
    "biker jacket": ("outerwear", "biker_jackets"),
    "field jacket": ("outerwear", "field_jackets"),
    "sports jacket": ("outerwear", "sports_jackets"),
    "fleece jacket": ("outerwear", "fleece_jackets"),
    "parka": ("outerwear", "parkas"),
    "down jacket": ("outerwear", "down_jackets"),
    "puffer jacket": ("outerwear", "puffer_jackets"),
    "vest": ("outerwear", "vests"),
}

BANNED_CLOTHING = {
    "panties", "bra", "underwear", "undergarment", "lingerie",
    "bikini", "swimwear", "swimsuit", "brassiere", "briefs",
    "thong", "g-string", "nude", "naked", "topless"
}

WARDROBE_CATEGORY_MAP = {
    "top": {"primary_category_key": "tops", "category_key": "top", "style": "Top"},
    "bottom": {"primary_category_key": "bottoms", "category_key": "bottom", "style": "Bottom"},
    "dress": {"primary_category_key": "dresses", "category_key": "dress", "style": "Dress"},
    "outer": {"primary_category_key": "outerwear", "category_key": "outerwear", "style": "Outerwear"},
}

GARMENT_ALLOWED_PRIMARY_KEYS = {
    "top": {"tops", "outerwear"},
    "outer": {"outerwear", "tops"},
    "bottom": {"bottoms", "skirts"},
    "dress": {"dresses"},
}

def normalize_item_garment_type(garment_type: str) -> str:
    g = (garment_type or "all").lower().strip()
    if g in {"top", "bottom", "dress", "outer"}:
        return g
    return "all"

def allowed_primary_keys_for_garment_type(garment_type: str) -> set:
    g = normalize_item_garment_type(garment_type)
    return set(GARMENT_ALLOWED_PRIMARY_KEYS.get(g, set()))

def style_category_keys(style: Optional[str]) -> Optional[Tuple[str, str]]:
    if not style:
        return None
    style_key = str(style).lower().strip()
    keys = STYLE_TO_CATEGORY_KEYS.get(style_key)
    if not keys:
        return None
    return keys

def is_style_compatible_with_garment_type(style: Optional[str], garment_type: str) -> bool:
    keys = style_category_keys(style)
    if not keys:
        return False
    allowed_primary = allowed_primary_keys_for_garment_type(garment_type)
    if not allowed_primary:
        return True
    primary_key, _ = keys
    return primary_key in allowed_primary

def wardrobe_category_from_garment_type(garment_type: str, style: Optional[str] = None) -> Dict[str, str]:
    if style:
        keys = style_category_keys(style)
        if keys and is_style_compatible_with_garment_type(style, garment_type):
            pk, ck = keys
            return {
                "primary_category_key": pk,
                "category_key": ck,
                "style": style.title() if style else "Unknown",
            }

    g = normalize_item_garment_type(garment_type)
    if g in WARDROBE_CATEGORY_MAP:
        return dict(WARDROBE_CATEGORY_MAP[g])

    return {
        "primary_category_key": "unknown",
        "category_key": "unknown",
        "style": style or "Unknown",
    }

def infer_style_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    matches = []
    # Reverse sort by length so longer phrases ("long sleeve t-shirt") matched first
    for style in sorted(STYLE_TO_CATEGORY_KEYS.keys(), key=len, reverse=True):
        if style in t:
            return style
    return None
