#!/usr/bin/env python3
"""
Debug color-extraction steps for a single image.

Usage:
  python3 scripts/debug_analyze_color_steps.py "/path/to/image.png"

Outputs a folder with:
  - input image
  - detection overlay
  - per-item crops, masks, overlays, palettes, and JSON summaries
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from ai import main as main_mod  # noqa: E402


BOX_COLORS: List[Tuple[int, int, int]] = [
    (235, 59, 90),
    (56, 103, 214),
    (32, 191, 107),
    (15, 185, 177),
    (245, 159, 26),
    (136, 84, 208),
]


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
    if hasattr(draw, "textsize"):
        size = draw.textsize(text, font=font)
        return (int(size[0]), int(size[1]))
    if hasattr(font, "getsize"):
        size = font.getsize(text)
        return (int(size[0]), int(size[1]))
    return (max(10, len(text) * 6), 12)


def _draw_detection_overlay(image: Image.Image, items: List[Dict[str, object]]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, item in enumerate(items):
        bbox = item.get("bbox") or [0, 0, canvas.width, canvas.height]
        x0, y0, x1, y1 = [int(v) for v in bbox]
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = str(item.get("label") or item.get("type") or "garment")
        typ = str(item.get("type") or "").strip()
        score = float(item.get("confidence", 0.0) or 0.0)
        text = f"{idx}: {typ} | {label} | {score:.2f}"
        text_w, text_h = _measure_text(draw, text, font)
        text_bg = [x0, max(0, y0 - text_h - 4), x0 + text_w + 6, y0]
        draw.rectangle(text_bg, fill=color)
        draw.text((text_bg[0] + 3, text_bg[1] + 1), text, fill=(255, 255, 255), font=font)
    return canvas


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color + (0,))
    alpha_mask = (mask.astype(np.uint8) * int(max(1, min(255, int(255 * alpha)))))
    overlay.putalpha(Image.fromarray(alpha_mask, mode="L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _masked_rgb(image: Image.Image, mask: Optional[np.ndarray]) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not isinstance(mask, np.ndarray) or mask.shape[:2] != arr.shape[:2]:
        return Image.fromarray(arr)
    out = arr.copy()
    out[~mask.astype(bool)] = 255
    return Image.fromarray(out)


def _palette_strip(palette: List[Dict[str, object]], width: int = 640, height: int = 120) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if not palette:
        draw.text((10, 10), "No palette extracted", fill=(40, 40, 40), font=font)
        return canvas

    swatch_w = max(60, int(width / max(1, len(palette))))
    for idx, entry in enumerate(palette):
        hx = str(entry.get("hex", "#000000")).strip().upper()
        area = float(entry.get("areaPercent", 0.0) or 0.0)
        try:
            r = int(hx[1:3], 16)
            g = int(hx[3:5], 16)
            b = int(hx[5:7], 16)
            color = (r, g, b)
        except Exception:
            color = (0, 0, 0)
        x0 = idx * swatch_w
        x1 = min(width, x0 + swatch_w)
        draw.rectangle([x0, 0, x1, height - 28], fill=color)
        label = f"{hx} ({area:.1f}%)"
        draw.text((x0 + 6, height - 24), label, fill=(20, 20, 20), font=font)
    return canvas


def _color_for_label_id(label_id: int) -> Tuple[int, int, int]:
    seed = int(label_id) & 0xFFFF
    r = (seed * 53 + 97) % 256
    g = (seed * 97 + 13) % 256
    b = (seed * 193 + 53) % 256
    return int(r), int(g), int(b)


def _colorize_parsing(parsing: np.ndarray) -> Image.Image:
    if parsing.size == 0:
        return Image.new("RGB", (1, 1), (0, 0, 0))
    max_id = int(np.max(parsing))
    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for idx in range(max_id + 1):
        palette[idx] = np.array(_color_for_label_id(idx), dtype=np.uint8)
    colored = palette[parsing]
    return Image.fromarray(colored.astype(np.uint8), mode="RGB")


def _overlay_colorized(image: Image.Image, colorized: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    overlay = colorized.convert("RGBA")
    overlay.putalpha(int(max(1, min(255, int(255 * alpha)))))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _rle_encode_2d(arr: np.ndarray) -> List[List[int]]:
    if arr.size == 0:
        return []
    flat = arr.astype(np.int32, copy=False).ravel()
    runs: List[List[int]] = []
    prev = int(flat[0])
    count = 1
    for val in flat[1:]:
        v = int(val)
        if v == prev:
            count += 1
        else:
            runs.append([prev, count])
            prev = v
            count = 1
    runs.append([prev, count])
    return runs


def _write_parser_raw_json(
    parsing: np.ndarray,
    parser_labels: Dict[str, int],
    item_dir: Path,
    alpha: float = 0.45,
) -> None:
    if parsing.size == 0:
        return
    h, w = parsing.shape[:2]
    max_id = int(np.max(parsing)) if parsing.size else 0
    palette: Dict[str, List[int]] = {}
    for idx in range(max_id + 1):
        palette[str(idx)] = list(_color_for_label_id(idx))
    payload = {
        "width": int(w),
        "height": int(h),
        "rle": _rle_encode_2d(parsing),
        "labels": {str(k): int(v) for k, v in parser_labels.items()},
        "overlay_palette": palette,
        "overlay_alpha": float(alpha),
        "note": "RLE is row-major over label ids; overlay uses overlay_palette + overlay_alpha.",
    }
    (item_dir / "01_parser_raw_ids.json").write_text(json.dumps(payload, separators=(",", ":")))


def _norm_label(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _serialize_candidate(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "label": str(item.get("label") or ""),
        "type": str(item.get("type") or ""),
        "confidence": float(item.get("confidence", 0.0) or 0.0),
        "bbox": [int(v) for v in (item.get("bbox") or [])],
        "source": str(item.get("source") or ""),
        "metrics": dict(item.get("metrics") or {}),
    }


def _run(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(str(path))

    output_root = ROOT / f"debug_color_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _safe_mkdir(output_root)

    image = Image.open(path)
    image_rgb = image.convert("RGB")
    image_rgb.save(output_root / "01_input.png")

    # Avoid loading optional captioning models for debug runs.
    main_mod.engine.yolo_runner.ensure_ready()
    if main_mod.engine.parser_runner:
        main_mod.engine.parser_runner.ensure_ready()
    candidates = main_mod.engine.cloth_detector.detect_fashion_candidates(image_rgb)

    overlay = _draw_detection_overlay(image_rgb, candidates)
    overlay.save(output_root / "02_detections.png")

    summary: Dict[str, object] = {
        "input": str(path),
        "output_dir": str(output_root),
        "candidates": [_serialize_candidate(c) for c in candidates],
        "items": [],
    }

    for idx, cand in enumerate(candidates):
        item_dir = output_root / f"item_{idx}_{cand.get('type') or 'unknown'}"
        _safe_mkdir(item_dir)
        bbox = cand.get("bbox") or [0, 0, image_rgb.width, image_rgb.height]
        crop = image_rgb.crop(tuple(int(v) for v in bbox))
        crop.save(item_dir / "01_crop.png")

        parser_labels: Dict[str, int] = {}
        raw_parser_labels: Dict[str, int] = {}
        parsing = None
        if main_mod.engine.parser is not None:
            try:
                parsing = main_mod.engine.parser.parse(crop)
                runtime_fn = getattr(main_mod.engine.parser, "_runtime_labels", None)
                raw_parser_labels = dict(runtime_fn() or {}) if callable(runtime_fn) else {}
                parser_labels = {
                    _norm_label(k): int(v)
                    for k, v in raw_parser_labels.items()
                    if str(k).strip()
                }
            except Exception:
                parsing = None
        if isinstance(parsing, np.ndarray):
            max_id = int(np.max(parsing)) if parsing.size else 0
            if max_id > 0:
                raw_ids = (parsing.astype(np.float32) / float(max_id) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                raw_ids = parsing.astype(np.uint8)
            Image.fromarray(raw_ids, mode="L").save(item_dir / "01_parser_raw_ids.png")
            colorized = _colorize_parsing(parsing)
            colorized.save(item_dir / "01_parser_raw_color.png")
            overlay_alpha = 0.45
            _overlay_colorized(crop, colorized, alpha=overlay_alpha).save(item_dir / "01_parser_raw_overlay.png")
            _write_parser_raw_json(parsing, parser_labels, item_dir, alpha=overlay_alpha)

            focus_labels = [
                "top",
                "upper",
                "upper_clothes",
                "upper-clothes",
                "dress",
                "skirt",
                "pants",
                "trousers",
                "bottom",
                "belt",
                "legs",
                "arms",
                "hands",
                "torso",
            ]
            for name in focus_labels:
                key = _norm_label(name)
                if key not in parser_labels:
                    continue
                mask = parsing == int(parser_labels[key])
                if int(np.sum(mask)) <= 0:
                    continue
                _mask_to_image(mask).save(item_dir / f"01_parser_mask_{key}.png")

        mask, mask_meta = main_mod._resolve_color_sampling_mask(
            image=crop,
            garment_type=str(cand.get("type") or ""),
            description="",
            reference_mask=None,
            apply_type_color_mask=True,
        )
        if isinstance(mask, np.ndarray):
            _mask_to_image(mask).save(item_dir / "02_mask_final.png")
            _overlay_mask(crop, mask, BOX_COLORS[idx % len(BOX_COLORS)]).save(item_dir / "03_mask_overlay.png")
            _masked_rgb(crop, mask).save(item_dir / "04_masked_rgb.png")
        else:
            (item_dir / "02_mask_final.txt").write_text("mask unavailable\n")

        ctx = main_mod._build_single_image_color_context(
            image=crop,
            description="",
            mask=mask if isinstance(mask, np.ndarray) else None,
            top_k=7,
            force_masking=bool(isinstance(mask, np.ndarray)),
        )
        palette = list(ctx.get("paletteMetricsFull") or [])
        _palette_strip(palette).save(item_dir / "05_palette.png")

        item_summary = {
            "index": idx,
            "bbox": [int(v) for v in bbox],
            "type": str(cand.get("type") or ""),
            "label": str(cand.get("label") or ""),
            "confidence": float(cand.get("confidence", 0.0) or 0.0),
            "mask_meta": dict(mask_meta or {}),
            "parser": {
                "labels": parser_labels,
                "raw_label_count": int(len(raw_parser_labels)),
            },
            "color_context": {
                "dominantHexes": list(ctx.get("dominantHexes") or []),
                "accentHexes": list(ctx.get("accentHexes") or []),
                "colorHints": list(ctx.get("colorHints") or []),
                "maskSource": str(ctx.get("maskSource") or ""),
                "profile": dict(ctx.get("profile") or {}),
                "paletteMetricsFull": palette,
            },
        }
        summary["items"].append(item_summary)

        with (item_dir / "summary.json").open("w", encoding="utf-8") as fp:
            json.dump(item_summary, fp, indent=2)

    with (output_root / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return output_root


def main() -> int:
    if len(sys.argv) < 2:
        print("Provide an image path.")
        return 2

    img_path = Path(sys.argv[1]).expanduser().resolve()
    out_dir = _run(img_path)
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
