#!/usr/bin/env python3
"""
Compare the current color pipeline against the Fashion-Product-baseColour model.

Examples:
    python3 scripts/trial_fashion_basecolour.py
    python3 scripts/trial_fashion_basecolour.py /path/to/image1.png /path/to/image2.png
"""

import json
import os
import sys
from pathlib import Path
from typing import List

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from ai import main as main_mod  # noqa: E402


DEFAULT_IMAGES: List[Path] = [
    ROOT / "top_extract_crop.png",
    ROOT / "top_expected_ref.png",
    ROOT / "u2net_top_1.png",
    ROOT / "test_white_shirt_final.png",
]


def _candidate_images_from_argv() -> List[Path]:
    if len(sys.argv) > 1:
        return [Path(arg).expanduser().resolve() for arg in sys.argv[1:]]
    return [path for path in DEFAULT_IMAGES if path.exists()]


def _run_single(path: Path) -> dict:
    image = Image.open(path)
    image_rgb = image.convert("RGBA" if image.mode == "RGBA" else "RGB")
    ctx = main_mod._build_single_image_color_context(
        image=image_rgb,
        description="",
        mask=None,
        top_k=7,
        force_masking=False,
    )
    dominant_hexes = [
        str(v)
        for v in (ctx.get("dominantHexes") or ctx.get("paletteHexes") or [])
        if str(v).strip()
    ]
    color_hints = [str(v) for v in (ctx.get("hints") or ctx.get("colorHints") or []) if str(v).strip()]
    color_profile = ctx.get("profile") if isinstance(ctx.get("profile"), dict) else {}
    color_mask_source = str(ctx.get("maskSource") or "none")
    reconciled = main_mod._resolve_garment_color_truth(
        base_garment_prompt="",
        descriptor_raw_text="",
        target_type="",
        dominant_hexes=dominant_hexes,
        color_hints=color_hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
        color_mask_source=color_mask_source,
    )

    trial_image = main_mod._prepare_fashion_basecolour_trial_image(image_rgb)
    hf_output = main_mod.engine.fashion_basecolour.predict_topk(
        trial_image,
        top_k=max(3, int(main_mod.FLUX2_COLOR_LOCK_TOP_K)),
    )
    rich_color = main_mod._build_rich_color_metadata(
        dominant_hexes=reconciled.get("dominant_hexes") or dominant_hexes,
        color_hints=reconciled.get("color_hints") or color_hints,
        color_profile=color_profile if isinstance(color_profile, dict) else None,
        fashion_color_classifier=hf_output if isinstance(hf_output, dict) else None,
    )
    return {
        "image": str(path),
        "size": {"width": int(image.width), "height": int(image.height)},
        "current_pipeline": {
            "dominant_hexes": dominant_hexes,
            "raw_color_hints": color_hints,
            "profile": color_profile,
            "mask_source": color_mask_source,
            "resolved_color_hints": reconciled.get("color_hints") or [],
            "resolved_source": reconciled.get("color_source") or "",
        },
        "rich_color_metadata": rich_color,
        "fashion_basecolour": hf_output,
    }


def main() -> int:
    image_paths = _candidate_images_from_argv()
    if not image_paths:
        print("No input images found.")
        return 1

    os.environ.setdefault("ANALYZE_FASHION_BASECOLOUR_TRIAL_ENABLED", "1")

    results = []
    for path in image_paths:
        if not path.exists():
            results.append({"image": str(path), "error": "missing"})
            continue
        try:
            results.append(_run_single(path))
        except Exception as err:
            results.append({"image": str(path), "error": str(err)})

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
