#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


PREVIEW_DIR = Path("/workspace/fashiontryon/review/train_preview")
OUT_DIR = Path("/workspace/fashiontryon/review/train_top_preview")
PAGES_DIR = OUT_DIR / "pages"


def load_rows() -> list[dict]:
    rows = []
    with (PREVIEW_DIR / "manifest.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def resolve_target(row: dict) -> str:
    targets = [p for p in row["target_paths"].split("|") if p]
    for target in targets:
        if target.endswith("/target.jpg"):
            return target
    return targets[0] if targets else ""


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()[:120]
    classifier = pipeline(
        "zero-shot-image-classification",
        model="google/siglip-base-patch16-224",
        device=0 if __import__("torch").cuda.is_available() else -1,
    )

    selected = []
    all_rows = []
    labels = ["top", "outerwear", "dress", "bottom"]
    for row in rows:
        target_rel = resolve_target(row)
        if not target_rel:
            continue
        path = (PREVIEW_DIR / target_rel).resolve()
        result = classifier(str(path), candidate_labels=labels)
        scores = {item["label"]: float(item["score"]) for item in result}
        best = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        item = {
            "sample_id": row["sample_id"],
            "target_rel": target_rel,
            "scores": scores,
            "label": best,
            "margin": margin,
        }
        all_rows.append(item)
        if best == "top":
            selected.append(item)

    with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "target_rel",
            "label",
            "top_score",
            "outerwear_score",
            "dress_score",
            "bottom_score",
            "margin",
        ])
        for item in selected:
            writer.writerow([
                item["sample_id"],
                item["target_rel"],
                item["label"],
                f"{item['scores']['top']:.4f}",
                f"{item['scores']['outerwear']:.4f}",
                f"{item['scores']['dress']:.4f}",
                f"{item['scores']['bottom']:.4f}",
                f"{item['margin']:.4f}",
            ])

    page_size = 30
    cols = 5
    cell_w, cell_h = 320, 240
    page_count = max(1, math.ceil(len(selected) / page_size))
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=22)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=12)
    font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=14)

    for page_idx in range(page_count):
        batch = selected[page_idx * page_size : (page_idx + 1) * page_size]
        page_rows = math.ceil(page_size / cols)
        page_w = cols * cell_w + 20
        page_h = page_rows * cell_h + 90
        canvas = Image.new("RGB", (page_w, page_h), (235, 235, 235))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f"FashionTryOn top-only preview {page_idx + 1}/{page_count}", font=font_title, fill=(20, 20, 20))
        draw.text((16, 40), f"{len(selected)} top garments selected from 120 garment previews", font=font_small, fill=(70, 70, 70))

        for i, item in enumerate(batch):
            r = i // cols
            c = i % cols
            x0 = 10 + c * cell_w
            y0 = 70 + r * cell_h
            draw.rounded_rectangle([x0, y0, x0 + cell_w - 10, y0 + cell_h - 10], radius=12, outline=(210, 210, 210), width=2, fill=(255, 255, 255))
            draw.text((x0 + 10, y0 + 8), item["sample_id"], font=font_bold, fill=(25, 25, 25))
            draw.text((x0 + 10, y0 + 26), f"top {item['scores']['top']:.3f} | outer {item['scores']['outerwear']:.3f}", font=font_small, fill=(80, 80, 80))

            path = PREVIEW_DIR / item["target_rel"]
            thumb = fit(path, (260, 160))
            canvas.paste(thumb, (x0 + 10, y0 + 46))
            draw.rectangle([x0 + 10, y0 + 46, x0 + 270, y0 + 206], outline=(180, 180, 180), width=1)
            draw.text((x0 + 14, y0 + 50), "target.jpg", font=font_small, fill=(0, 0, 0))

        canvas.save(PAGES_DIR / f"page_{page_idx:04d}.png")

    links = "\n".join(f'<li><a href="pages/page_{idx:04d}.png">page_{idx + 1:04d}.png</a></li>' for idx in range(page_count))
    (OUT_DIR / "index.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FashionTryOn top-only preview</title>
<style>body{{font-family:system-ui;margin:24px;background:#f2f2f2;color:#111}}a{{color:#111}}ul{{line-height:1.9}}code{{background:#fff;padding:2px 6px;border:1px solid #ddd;border-radius:6px}}</style>
</head><body>
<h1>FashionTryOn top-only preview</h1>
<p>Selected by zero-shot category filtering from the garment-only preview.</p>
<p>Count: {len(selected)}</p>
<ul>{links}</ul>
</body></html>""", encoding="utf-8")
    print(f"selected={len(selected)}")
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
