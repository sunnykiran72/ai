#!/usr/bin/env python3
"""Build a browsable FashionTryOn review folder.

This keeps the FashionTryOn review separate from the main API tree and writes:
* manifest.csv
* index.html with paginated browsing

It does not copy the dataset; it references the extracted files in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def discover_sample_dirs(data_root: Path, limit: int | None = None) -> list[Path]:
    candidates: set[Path] = set()
    for dirpath, _, filenames in os.walk(data_root):
        if any(name == "target.jpg" or name.endswith("_target.jpg") for name in filenames):
            candidates.add(Path(dirpath))
            if limit is not None and len(candidates) >= limit:
                break
    return sorted(candidates)


def relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def collect_records(data_root: Path, out_dir: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    for sample_dir in discover_sample_dirs(data_root, limit=limit):
        root_images = [p for p in sorted(sample_dir.iterdir()) if is_image(p)]
        targets = [p for p in root_images if p.name == "target.jpg" or p.name.endswith("_target.jpg")]
        masks = [p for p in root_images if p.name == "mask.jpg" or "mask" in p.name.lower()]
        sources = [p for p in root_images if p not in targets and p not in masks]
        records.append(
            {
                "sample_id": sample_dir.name,
                "sample_rel": relpath(sample_dir, out_dir),
                "target_paths": [relpath(p, out_dir) for p in targets],
                "source_paths": [relpath(p, out_dir) for p in sources],
                "mask_paths": [relpath(p, out_dir) for p in masks],
                "target_count": len(targets),
                "source_count": len(sources),
                "mask_count": len(masks),
            }
        )
    return records


def write_manifest(records: list[dict], out_dir: Path) -> None:
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "sample_rel",
                "target_paths",
                "source_paths",
                "mask_paths",
                "target_count",
                "source_count",
                "mask_count",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["sample_id"],
                    record["sample_rel"],
                    "|".join(record["target_paths"]),
                    "|".join(record["source_paths"]),
                    "|".join(record["mask_paths"]),
                    record["target_count"],
                    record["source_count"],
                    record["mask_count"],
                ]
            )


def write_index(records: list[dict], out_dir: Path) -> None:
    js_data = json.dumps(records, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FashionTryOn review</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #efefef; color: #111; }}
    header {{ position: sticky; top: 0; background: rgba(255,255,255,0.97); backdrop-filter: blur(8px); border-bottom: 1px solid #ddd; padding: 14px 18px; z-index: 10; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .sub {{ color: #666; font-size: 13px; margin-top: 4px; }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 10px; }}
    input, select, button {{ padding: 8px 10px; border: 1px solid #bbb; border-radius: 8px; font-size: 14px; }}
    button {{ background: #111; color: #fff; border-color: #111; cursor: pointer; }}
    .pill {{ display: inline-block; padding: 2px 8px; border: 1px solid #ddd; background: #f3f3f3; border-radius: 999px; font-size: 12px; color: #333; }}
    main {{ padding: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 14px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.05); }}
    .card h3 {{ margin: 0; padding: 10px 12px 4px; font-size: 14px; }}
    .card .meta {{ padding: 0 12px 10px; color: #666; font-size: 12px; }}
    .imgs {{ display: grid; grid-template-columns: 2fr 1fr; gap: 8px; padding: 0 12px 12px; }}
    .stack {{ display: grid; gap: 8px; }}
    .box {{ position: relative; background: #f8f8f8; border: 1px solid #e2e2e2; border-radius: 10px; overflow: hidden; min-height: 118px; }}
    .box img {{ width: 100%; height: 100%; object-fit: contain; display: block; background: #fff; }}
    .tag {{ position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,.72); color: #fff; border-radius: 999px; padding: 2px 6px; font-size: 11px; }}
    .footer {{ padding: 8px 18px 20px; color: #666; font-size: 13px; }}
    .pagebox {{ margin-left: auto; display: flex; gap: 8px; align-items: center; }}
    a {{ color: inherit; text-decoration: none; }}
  </style>
</head>
<body>
<header>
  <div style="display:flex; gap:14px; flex-wrap:wrap; align-items:center;">
    <div>
      <div class="title">FashionTryOn review</div>
      <div class="sub">Separate review folder for the extracted FashionTryOn train split. Images are referenced from /workspace/fashiontryon/data/train/train.</div>
    </div>
    <div class="pagebox">
      <button id="prev">Prev</button>
      <span id="pageInfo" class="pill"></span>
      <button id="next">Next</button>
    </div>
  </div>
  <div class="controls">
    <input id="q" type="search" placeholder="Filter by sample id" style="min-width:240px;">
    <select id="pageSize"><option>20</option><option selected>30</option><option>40</option><option>60</option></select>
    <button id="reset">Reset</button>
    <span id="count" class="pill"></span>
  </div>
</header>
<main><div id="grid" class="grid"></div></main>
<div class="footer">Open <code>manifest.csv</code> for the full inventory. This HTML page is paginated and references the dataset in place.</div>
<script>
const DATA = {js_data};
let filtered = DATA.slice();
let page = 0;
let pageSize = 30;
const grid = document.getElementById('grid');
const pageInfo = document.getElementById('pageInfo');
const count = document.getElementById('count');
const q = document.getElementById('q');
const pageSizeSel = document.getElementById('pageSize');
function rel(path) {{ return '../data/train/train/' + path; }}
function box(path, label) {{
  if (!path) return `<div class="box"><span class="tag">${{label}}</span></div>`;
  return `<a class="box" href="${{rel(path)}}" target="_blank"><span class="tag">${{label}}</span><img loading="lazy" src="${{rel(path)}}" alt="${{label}}"></a>`;
}}
function render() {{
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  if (page >= totalPages) page = totalPages - 1;
  const start = page * pageSize;
  const end = Math.min(filtered.length, start + pageSize);
  pageInfo.textContent = `${{filtered.length}} samples | page ${{page + 1}} / ${{totalPages}} | showing ${{start + 1}}-${{end}}`;
  count.textContent = `${{filtered.length}} visible`;
  grid.innerHTML = '';
  filtered.slice(start, end).forEach(s => {{
    const target = s.target_paths[0] || '';
    const srcs = s.source_paths.slice(0, 2);
    const mask = s.mask_paths[0] || '';
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h3>${{s.sample_id}}</h3>
      <div class="meta">targets: ${{s.target_count}} | sources: ${{s.source_count}} | masks: ${{s.mask_count}}<br>${{s.sample_rel}}</div>
      <div class="imgs">
        ${{box(target, 'target')}}
        <div class="stack">
          ${{box(srcs[0], 'src 1')}}
          ${{box(srcs[1], 'src 2')}}
          ${{box(mask, 'mask')}}
        </div>
      </div>`;
    grid.appendChild(card);
  }});
}}
function applyFilter() {{
  const s = q.value.trim().toLowerCase();
  filtered = DATA.filter(r => !s || r.sample_id.toLowerCase().includes(s) || r.sample_rel.toLowerCase().includes(s));
  page = 0;
  render();
}}
document.getElementById('prev').onclick = () => {{ if (page > 0) {{ page--; render(); }} }};
document.getElementById('next').onclick = () => {{ if ((page + 1) * pageSize < filtered.length) {{ page++; render(); }} }};
document.getElementById('reset').onclick = () => {{ q.value = ''; pageSizeSel.value = '30'; pageSize = 30; applyFilter(); }};
q.addEventListener('input', applyFilter);
pageSizeSel.addEventListener('change', () => {{ pageSize = parseInt(pageSizeSel.value, 10); page = 0; render(); }});
render();
</script>
</body>
</html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(data_root, out_dir, args.limit)
    write_manifest(records, out_dir)
    write_index(records, out_dir)
    print(f"samples={len(records)}")
    print(f"out={out_dir}")


if __name__ == "__main__":
    main()
