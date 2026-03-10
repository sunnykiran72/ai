import difflib
import re
from typing import Dict, Optional, Tuple

CLOTHING_STYLES = [
    "t-shirt", "long sleeve t-shirt", "sleeveless t-shirt", "polo shirt",
    "tank top", "camisole", "crop top", "blouse", "casual shirt",
    "bra", "bralette", "bustier", "corset", "tunic", "kurta", "kurti",
    "sweatshirt", "hoodie", "sweater", "sweater vest", "sports top",
    "bodysuit", "knitwear", "cardigan", "jeans", "trousers", "dress pants",
    "track pants", "leggings", "sweatpants", "shorts", "palazzo", "salwar",
    "dhoti pants", "skirt", "mini skirt", "midi skirt", "maxi skirt", "lehenga skirt",
    "coat", "trench coat", "fur coat", "shearling coat",
    "blazer", "suit jacket", "jacket", "blouson", "varsity jacket", "trucker jacket",
    "biker jacket", "field jacket", "sports jacket", "fleece jacket", "bolero",
    "shrug", "cape", "poncho", "shawl", "dupatta", "kimono",
    "parka", "down jacket", "puffer jacket", "vest", "mini dress", "day dress",
    "t-shirt dress", "shirt dress", "sweatshirt dress", "sweater dress",
    "jacket dress", "suspender dress", "party dress", "maxi dress", "gown",
    "slip dress", "bodycon dress", "jumpsuit", "romper", "kaftan",
    "anarkali", "saree", "lehenga set", "salwar kameez set", "abaya", "qipao"
]

STYLE_TO_CATEGORY_KEYS = {
    "t-shirt": ("tops", "t_shirts"),
    "long sleeve t-shirt": ("tops", "long_sleeve_t_shirts"),
    "sleeveless t-shirt": ("tops", "sleeveless_t_shirts"),
    "polo shirt": ("tops", "polo_shirts"),
    "tank top": ("tops", "tanks_and_camis"),
    "camisole": ("tops", "tanks_and_camis"),
    "crop top": ("tops", "crop_tops"),
    "blouse": ("tops", "blouses"),
    "bra": ("tops", "tanks_and_camis"),
    "bralette": ("tops", "tanks_and_camis"),
    "bustier": ("tops", "crop_tops"),
    "corset": ("tops", "crop_tops"),
    "tunic": ("tops", "blouses"),
    "kurta": ("tops", "blouses"),
    "kurti": ("dresses", "day_dresses"),
    "casual shirt": ("tops", "shirts"),
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
    "palazzo": ("bottoms", "trousers"),
    "salwar": ("bottoms", "pants"),
    "dhoti pants": ("bottoms", "pants"),
    "skirt": ("skirts", "midi_skirts"),
    "mini skirt": ("skirts", "mini_skirts"),
    "midi skirt": ("skirts", "midi_skirts"),
    "maxi skirt": ("skirts", "maxi_skirts"),
    "lehenga skirt": ("skirts", "maxi_skirts"),
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
    "gown": ("dresses", "maxi_dresses"),
    "slip dress": ("dresses", "day_dresses"),
    "bodycon dress": ("dresses", "party_dresses"),
    "jumpsuit": ("dresses", "jumpsuits"),
    "romper": ("dresses", "jumpsuits"),
    "kaftan": ("dresses", "maxi_dresses"),
    "anarkali": ("dresses", "maxi_dresses"),
    "saree": ("dresses", "maxi_dresses"),
    "lehenga set": ("dresses", "maxi_dresses"),
    "salwar kameez set": ("dresses", "day_dresses"),
    "abaya": ("dresses", "maxi_dresses"),
    "qipao": ("dresses", "party_dresses"),
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
    "bolero": ("outerwear", "cardigans"),
    "shrug": ("outerwear", "cardigans"),
    "cape": ("outerwear", "coats"),
    "poncho": ("outerwear", "coats"),
    "shawl": ("outerwear", "cardigans"),
    "dupatta": ("outerwear", "cardigans"),
    "kimono": ("outerwear", "cardigans"),
    "parka": ("outerwear", "parkas"),
    "down jacket": ("outerwear", "down_jackets"),
    "puffer jacket": ("outerwear", "puffer_jackets"),
    "vest": ("outerwear", "vests"),
}

STYLE_ALIASES = {
    "tee": "t-shirt",
    "tee shirt": "t-shirt",
    "t shirt": "t-shirt",
    "long sleeve tee": "long sleeve t-shirt",
    "long sleeve tshirt": "long sleeve t-shirt",
    "sleeveless tee": "sleeveless t-shirt",
    "sleeveless tshirt": "sleeveless t-shirt",
    "tank": "tank top",
    "cami": "camisole",
    "brassiere": "bra",
    "bikini top": "bra",
    "lingerie top": "bralette",
    "button down shirt": "casual shirt",
    "button-up shirt": "casual shirt",
    "joggers": "sweatpants",
    "jogger": "sweatpants",
    "palazzos": "palazzo",
    "salwar pants": "salwar",
    "dhoti pant": "dhoti pants",
    "lehenga": "lehenga set",
    "lehenga choli": "lehenga set",
    "salwar suit": "salwar kameez set",
    "salwar kameez": "salwar kameez set",
    "kurtha": "kurta",
    "sari": "saree",
    "track pant": "track pants",
    "slacks": "dress pants",
    "pant": "trousers",
    "pants": "trousers",
    "denim": "jeans",
    "mini skirt": "mini skirt",
    "midi skirt": "midi skirt",
    "maxi skirt": "maxi skirt",
    "blazer jacket": "blazer",
    "suit blazer": "blazer",
    "hooded sweatshirt": "hoodie",
    "puffer": "puffer jacket",
    "windbreaker": "sports jacket",
}

KEYWORD_STYLE_HINTS = {
    "long sleeve": "long sleeve t-shirt",
    "sleeveless": "sleeveless t-shirt",
    "polo": "polo shirt",
    "tank": "tank top",
    "camisole": "camisole",
    "bra": "bra",
    "bralette": "bralette",
    "bustier": "bustier",
    "corset": "corset",
    "tunic": "tunic",
    "kurta": "kurta",
    "kurti": "kurti",
    "crop": "crop top",
    "blouse": "blouse",
    "shirt dress": "shirt dress",
    "t shirt dress": "t-shirt dress",
    "sweatshirt dress": "sweatshirt dress",
    "sweater dress": "sweater dress",
    "party dress": "party dress",
    "mini dress": "mini dress",
    "maxi dress": "maxi dress",
    "jumpsuit": "jumpsuit",
    "romper": "romper",
    "jeans": "jeans",
    "trouser": "trousers",
    "dress pant": "dress pants",
    "track pant": "track pants",
    "legging": "leggings",
    "sweatpant": "sweatpants",
    "palazzo": "palazzo",
    "salwar": "salwar",
    "dhoti": "dhoti pants",
    "shorts": "shorts",
    "lehenga skirt": "lehenga skirt",
    "lehenga": "lehenga set",
    "mini skirt": "mini skirt",
    "midi skirt": "midi skirt",
    "maxi skirt": "maxi skirt",
    "skirt": "skirt",
    "gown": "gown",
    "slip dress": "slip dress",
    "bodycon": "bodycon dress",
    "kaftan": "kaftan",
    "anarkali": "anarkali",
    "saree": "saree",
    "sari": "saree",
    "salwar kameez": "salwar kameez set",
    "abaya": "abaya",
    "qipao": "qipao",
    "trench": "trench coat",
    "coat": "coat",
    "blazer": "blazer",
    "varsity": "varsity jacket",
    "trucker": "trucker jacket",
    "biker": "biker jacket",
    "field jacket": "field jacket",
    "fleece": "fleece jacket",
    "bolero": "bolero",
    "shrug": "shrug",
    "cape": "cape",
    "poncho": "poncho",
    "shawl": "shawl",
    "dupatta": "dupatta",
    "kimono": "kimono",
    "parka": "parka",
    "down jacket": "down jacket",
    "puffer": "puffer jacket",
    "hoodie": "hoodie",
    "cardigan": "cardigan",
    "sweater vest": "sweater vest",
}

BANNED_CLOTHING = {
    "panties", "underwear", "undergarment",
    "bikini", "swimwear", "swimsuit", "briefs",
    "thong", "g-string", "nude", "naked", "topless"
}

EXISTING_CATEGORY_KEYS_BY_PRIMARY = {
    "tops": (
        "blouses",
        "bodysuits",
        "crop_tops",
        "hoodies",
        "knitwear",
        "long_sleeve_t_shirts",
        "polo_shirts",
        "shirts",
        "sleeveless_t_shirts",
        "sports_tops",
        "sweater_vests",
        "sweaters",
        "sweatshirts",
        "t_shirts",
        "tanks_and_camis",
    ),
    "bottoms": (
        "dress_pants",
        "jeans",
        "leggings",
        "pants",
        "shorts",
        "sweatpants",
        "track_pants",
        "trousers",
    ),
    "skirts": (
        "maxi_skirts",
        "midi_skirts",
        "mini_skirts",
    ),
    "dresses": (
        "day_dresses",
        "jacket_dresses",
        "jumpsuits",
        "maxi_dresses",
        "mini_dresses",
        "party_dresses",
        "shirt_dresses",
        "suspender_dresses",
        "sweater_dresses",
        "sweatshirt_dresses",
        "t_shirt_dresses",
    ),
    "outerwear": (
        "biker_jackets",
        "blazers",
        "blousons",
        "cardigans",
        "coats",
        "down_jackets",
        "field_jackets",
        "fleece_jackets",
        "fur_coats",
        "jackets",
        "parkas",
        "puffer_jackets",
        "shearling_coats",
        "sports_jackets",
        "trench_coats",
        "trucker_jackets",
        "varsity_jackets",
        "vests",
        "zip_up_hoodies",
    ),
}

PRIMARY_CATEGORY_FALLBACK = {
    "tops": ("t_shirts", "t-shirt"),
    "bottoms": ("pants", "pants"),
    "skirts": ("midi_skirts", "midi skirt"),
    "dresses": ("day_dresses", "day dress"),
    "outerwear": ("jackets", "jacket"),
}

WARDROBE_CATEGORY_MAP = {
    "top": {"primary_category_key": "tops", "category_key": "t_shirts", "style": "t-shirt"},
    "bottom": {"primary_category_key": "bottoms", "category_key": "pants", "style": "pants"},
    "dress": {"primary_category_key": "dresses", "category_key": "day_dresses", "style": "day dress"},
    "outer": {"primary_category_key": "outerwear", "category_key": "jackets", "style": "jacket"},
}

GARMENT_ALLOWED_PRIMARY_KEYS = {
    "top": {"tops", "outerwear"},
    "outer": {"outerwear", "tops"},
    "bottom": {"bottoms", "skirts"},
    "dress": {"dresses"},
}

def _normalize_style_text(text: Optional[str]) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = " ".join(s.split())
    return s

_NORMALIZED_STYLE_INDEX = {
    _normalize_style_text(style): style
    for style in STYLE_TO_CATEGORY_KEYS.keys()
}

def _resolve_canonical_style(style: Optional[str], garment_type: Optional[str] = None) -> Optional[str]:
    normalized = _normalize_style_text(style)
    if not normalized:
        return None

    candidates = []
    seen = set()

    def _push(canonical: Optional[str]):
        if not canonical:
            return
        if canonical not in STYLE_TO_CATEGORY_KEYS:
            return
        if canonical in seen:
            return
        seen.add(canonical)
        candidates.append(canonical)

    # 1) Direct alias/exact matches.
    _push(STYLE_ALIASES.get(normalized))
    _push(_NORMALIZED_STYLE_INDEX.get(normalized))

    # 2) Substring/contains matches, preferring longer (more specific) styles.
    for norm_style, canonical in sorted(_NORMALIZED_STYLE_INDEX.items(), key=lambda kv: len(kv[0]), reverse=True):
        if norm_style and (norm_style in normalized or normalized in norm_style):
            _push(canonical)

    # 3) Keyword hints from caption-like descriptions.
    for key, canonical in sorted(KEYWORD_STYLE_HINTS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in normalized:
            _push(canonical)

    # 4) Fuzzy fallback.
    universe = list(_NORMALIZED_STYLE_INDEX.keys()) + list(STYLE_ALIASES.keys())
    for near in difflib.get_close_matches(normalized, universe, n=4, cutoff=0.72):
        _push(STYLE_ALIASES.get(near, _NORMALIZED_STYLE_INDEX.get(near)))

    if not candidates:
        return None

    if garment_type:
        allowed_primary = allowed_primary_keys_for_garment_type(garment_type)
        if allowed_primary:
            for c in candidates:
                pk, _ = STYLE_TO_CATEGORY_KEYS[c]
                if pk in allowed_primary:
                    return c
    return candidates[0]

def normalize_item_garment_type(garment_type: str) -> str:
    g = (garment_type or "all").lower().strip()
    if g in {"top", "bottom", "dress", "outer"}:
        return g
    return "all"

def allowed_primary_keys_for_garment_type(garment_type: str) -> set:
    g = normalize_item_garment_type(garment_type)
    return set(GARMENT_ALLOWED_PRIMARY_KEYS.get(g, set()))

def style_category_keys(style: Optional[str], garment_type: Optional[str] = None) -> Optional[Tuple[str, str]]:
    canonical = _resolve_canonical_style(style, garment_type=garment_type)
    if not canonical:
        return None
    keys = STYLE_TO_CATEGORY_KEYS.get(canonical)
    if not keys:
        return None
    return keys

def is_style_compatible_with_garment_type(style: Optional[str], garment_type: str) -> bool:
    keys = style_category_keys(style, garment_type=garment_type)
    if not keys:
        return False
    allowed_primary = allowed_primary_keys_for_garment_type(garment_type)
    if not allowed_primary:
        return True
    primary_key, _ = keys
    return primary_key in allowed_primary

def wardrobe_category_from_garment_type(garment_type: str, style: Optional[str] = None) -> Dict[str, str]:
    if style:
        canonical = _resolve_canonical_style(style, garment_type=garment_type)
        keys = style_category_keys(canonical, garment_type=garment_type)
        if keys and is_style_compatible_with_garment_type(canonical, garment_type):
            pk, ck = keys
            if ck in set(EXISTING_CATEGORY_KEYS_BY_PRIMARY.get(pk, ())):
                return {
                    "primary_category_key": pk,
                    "category_key": ck,
                    "style": canonical,
                }
            fallback = PRIMARY_CATEGORY_FALLBACK.get(pk)
            if fallback:
                fallback_ck, fallback_style = fallback
                return {
                    "primary_category_key": pk,
                    "category_key": fallback_ck,
                    "style": canonical or fallback_style,
                }
            return {
                "primary_category_key": pk,
                "category_key": ck,
                "style": canonical,
            }

    g = normalize_item_garment_type(garment_type)
    if g in WARDROBE_CATEGORY_MAP:
        return dict(WARDROBE_CATEGORY_MAP[g])

    return {
        "primary_category_key": "unknown",
        "category_key": "unknown",
        "style": style or "Unknown",
    }

def infer_style_from_text(text: str, garment_type: Optional[str] = None) -> Optional[str]:
    return _resolve_canonical_style(text, garment_type=garment_type)
