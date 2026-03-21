#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


DATA_ROOT = Path("/workspace/fashiontryon/data/train/train")
OUT_DIR = Path("/workspace/fashiontryon/review/train_top_full")
PAGES_DIR = OUT_DIR / "pages"
LABELS = ["top", "outerwear", "dress", "bottom"]


def discover_targets(data_root: Path) -> list[dict]:
    records = []
    for dirpath, _, filenames in os.walk(data_root):
        if "target.jpg" in filenames:
            sample_dir = Path(dirpath)
            target = sample_dir / "target.jpg"
            if target.is_file():
                records.append(
                    {
                        "sample_id": sample_dir.name,
                        "target_path": str(target),
                    }
                )
    return records


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    records = discover_targets(DATA_ROOT)
    print(f"discovered={len(records)}")

    clf = pipeline(
        "zero-shot-image-classification",
        model="google/siglip-base-patch16-224",
        device=0 if torch.cuda.is_available() else -1,
    )

    selected = []
    batch_size = 64
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = [r["target_path"] for r in batch]
        preds = clf(paths, candidate_labels=LABELS)
        for row, pred in zip(batch, preds):
            scores = {item["label"]: float(item["score"]) for item in pred}
            best = max(scores, key=scores.get)
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
            row["scores"] = scores
            row["label"] = best
            row["margin"] = margin
            if best == "top":
                selected.append(row)
        if (start // batch_size) % 10 == 0:
            print(f"processed={min(start + batch_size, len(records))}/{len(records)} selected={len(selected)}")

    with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "target_path", "label", "top_score", "outerwear_score", "dress_score", "bottom_score", "margin"])
        for row in selected:
            writer.writerow(
                [
                    row["sample_id"],
                    row["target_path"],
                    row["label"],
                    f"{row['scores']['top']:.4f}",
                    f"{row['scores']['outerwear']:.4f}",
                    f"{row['scores']['dress']:.4f}",
                    f"{row['scores']['bottom']:.4f}",
                    f"{row['margin']:.4f}",
                ]
            )

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
        draw.text((16, 12), f"FashionTryOn top-only full {page_idx + 1}/{page_count}", font=font_title, fill=(20, 20, 20))
        draw.text((16, 40), f"{len(selected)} top garments selected from {len(records)} garment targets", font=font_small, fill=(70, 70, 70))

        for i, row in enumerate(batch):
            r = i // cols
            c = i % cols
            x0 = 10 + c * cell_w
            y0 = 70 + r * cell_h
            draw.rounded_rectangle([x0, y0, x0 + cell_w - 10, y0 + cell_h - 10], radius=12, outline=(210, 210, 210), width=2, fill=(255, 255, 255))
            draw.text((x0 + 10, y0 + 8), row["sample_id"], font=font_bold, fill=(25, 25, 25))
            draw.text((x0 + 10, y0 + 26), f"top {row['scores']['top']:.3f} | outer {row['scores']['outerwear']:.3f}", font=font_small, fill=(80, 80, 80))
            thumb = fit(Path(row["target_path"]), (260, 160))
            canvas.paste(thumb, (x0 + 10, y0 + 46))
            draw.rectangle([x0 + 10, y0 + 46, x0 + 270, y0 + 206], outline=(180, 180, 180), width=1)
            draw.text((x0 + 14, y0 + 50), "target.jpg", font=font_small, fill=(0, 0, 0))

        canvas.save(PAGES_DIR / f"page_{page_idx:04d}.png")

    links = "\n".join(f'<li><a href="pages/page_{idx:04d}.png">page_{idx + 1:04d}.png</a></li>' for idx in range(page_count))
    (OUT_DIR / "index.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FashionTryOn top-only full</title>
<style>body{{font-family:system-ui;margin:24px;background:#f2f2f2;color:#111}}a{{color:#111}}ul{{line-height:1.9}}code{{background:#fff;padding:2px 6px;border:1px solid #ddd;border-radius:6px}}</style>
</head><body>
<h1>FashionTryOn top-only full</h1>
<p>Selected by zero-shot category filtering from all garment-only target images.</p>
<p>Count: {len(selected)} of {len(records)}</p>
<ul>{links}</ul>
</body></html>""", encoding="utf-8")
    print(f"selected={len(selected)}")
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
