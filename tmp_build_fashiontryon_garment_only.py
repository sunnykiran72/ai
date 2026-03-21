#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PREVIEW_DIR = Path("/workspace/fashiontryon/review/train_preview")
OUT_DIR = Path("/workspace/fashiontryon/review/train_garment_only_preview")
SHEETS_DIR = OUT_DIR / "pages"


def load_font(size: int, bold: bool = False):
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(path, size=size)


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def read_rows() -> list[dict]:
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


def main() -> None:
    rows = read_rows()[:120]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    page_size = 30
    cols = 5
    cell_w = 320
    cell_h = 240
    page_count = math.ceil(len(rows) / page_size)

    for page_idx in range(page_count):
        batch = rows[page_idx * page_size : (page_idx + 1) * page_size]
        page_rows = math.ceil(page_size / cols)
        page_w = cols * cell_w + 20
        page_h = page_rows * cell_h + 90
        canvas = Image.new("RGB", (page_w, page_h), (235, 235, 235))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (16, 12),
            f"FashionTryOn garment-only preview {page_idx + 1}/{page_count}",
            font=load_font(22, True),
            fill=(20, 20, 20),
        )
        draw.text(
            (16, 40),
            "30 target.jpg garment images per page",
            font=load_font(12),
            fill=(70, 70, 70),
        )

        for i, row in enumerate(batch):
            r = i // cols
            c = i % cols
            x0 = 10 + c * cell_w
            y0 = 70 + r * cell_h
            draw.rounded_rectangle(
                [x0, y0, x0 + cell_w - 10, y0 + cell_h - 10],
                radius=12,
                outline=(210, 210, 210),
                width=2,
                fill=(255, 255, 255),
            )
            draw.text((x0 + 10, y0 + 8), row["sample_id"], font=load_font(14, True), fill=(25, 25, 25))
            draw.text((x0 + 10, y0 + 26), "garment only", font=load_font(11), fill=(80, 80, 80))

            target_rel = resolve_target(row)
            if target_rel:
                target_path = PREVIEW_DIR / target_rel
                canvas.paste(fit(target_path, (260, 160)), (x0 + 10, y0 + 46))
                draw.rectangle([x0 + 10, y0 + 46, x0 + 270, y0 + 206], outline=(180, 180, 180), width=1)
                draw.text((x0 + 14, y0 + 50), "target.jpg", font=load_font(10), fill=(0, 0, 0))

        canvas.save(SHEETS_DIR / f"page_{page_idx:04d}.png")

    links = "\n".join(
        f'<li><a href="pages/page_{idx:04d}.png">page_{idx + 1:04d}.png</a></li>'
        for idx in range(page_count)
    )
    index = SHEETS_DIR / "index.html"
    index.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FashionTryOn garment-only preview</title>
<style>
body{{font-family:system-ui;margin:24px;background:#f2f2f2;color:#111}}
a{{color:#111}}
ul{{line-height:1.9}}
code{{background:#fff;padding:2px 6px;border:1px solid #ddd;border-radius:6px}}
</style>
</head><body>
<h1>FashionTryOn garment-only preview</h1>
<p>These sheets use <code>target.jpg</code> only, which is the clean garment item.</p>
<ul>{links}</ul>
</body></html>
""",
        encoding="utf-8",
    )
    print(f"pages={page_count}")
    print(f"out={SHEETS_DIR}")


if __name__ == "__main__":
    main()
