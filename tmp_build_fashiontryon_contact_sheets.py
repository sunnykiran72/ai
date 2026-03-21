#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PREVIEW_DIR = Path("/workspace/fashiontryon/review/train_preview")
OUT_DIR = PREVIEW_DIR / "contact_sheets"


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


def resolve_preview_rel(rel: str) -> Path:
    return (PREVIEW_DIR / rel).resolve()


def read_rows() -> list[dict]:
    rows = []
    with (PREVIEW_DIR / "manifest.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    rows = read_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    page_size = 30
    cols = 5
    cell_w = 320
    cell_h = 280
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
            f"FashionTryOn preview contact sheet {page_idx + 1}/{page_count}",
            font=load_font(22, True),
            fill=(20, 20, 20),
        )
        draw.text(
            (16, 40),
            "30 samples per page from the 120-sample preview",
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
            draw.text(
                (x0 + 10, y0 + 26),
                f"src {row['source_count']} | tgt {row['target_count']} | mask {row['mask_count']}",
                font=load_font(11),
                fill=(80, 80, 80),
            )

            target_rel = row["target_paths"].split("|")[0] if row["target_paths"] else ""
            srcs = [p for p in row["source_paths"].split("|") if p]
            mask_rel = row["mask_paths"].split("|")[0] if row["mask_paths"] else ""

            if target_rel:
                target_path = resolve_preview_rel(target_rel)
                canvas.paste(fit(target_path, (180, 170)), (x0 + 10, y0 + 50))
                draw.rectangle([x0 + 10, y0 + 50, x0 + 190, y0 + 220], outline=(180, 180, 180), width=1)
                draw.text((x0 + 14, y0 + 54), "target", font=load_font(10), fill=(0, 0, 0))

            side_x = x0 + 200
            side_w = 100
            side_h = 52
            side_items = [srcs[0] if len(srcs) > 0 else "", srcs[1] if len(srcs) > 1 else "", mask_rel]
            for j, rel in enumerate(side_items):
                yy = y0 + 50 + j * 56
                rect = [side_x, yy, side_x + side_w, yy + side_h]
                if rel:
                    path = resolve_preview_rel(rel)
                    canvas.paste(fit(path, (side_w, side_h)), (side_x, yy))
                    draw.rectangle(rect, outline=(180, 180, 180), width=1)
                    draw.text((side_x + 3, yy + 3), Path(rel).name, font=load_font(9), fill=(0, 0, 0))
                else:
                    draw.rectangle(rect, outline=(220, 220, 220), width=1, fill=(248, 248, 248))

        canvas.save(OUT_DIR / f"page_{page_idx:04d}.png")

    index = OUT_DIR / "index.html"
    links = "\n".join(
        f'<li><a href="page_{i:04d}.png">page_{i + 1:04d}.png</a></li>'
        for i in range(page_count)
    )
    index.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FashionTryOn preview contact sheets</title>
<style>
body{{font-family:system-ui;margin:24px;background:#f2f2f2;color:#111}}
a{{color:#111}}
ul{{line-height:1.9}}
code{{background:#fff;padding:2px 6px;border:1px solid #ddd;border-radius:6px}}
</style>
</head><body>
<h1>FashionTryOn preview contact sheets</h1>
<p>Big-image sheets with 30 samples per page. Open the PNG files below.</p>
<ul>{links}</ul>
<p>Manifest: <code>../manifest.csv</code></p>
</body></html>
""",
        encoding="utf-8",
    )
    print(f"pages={page_count}")
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
