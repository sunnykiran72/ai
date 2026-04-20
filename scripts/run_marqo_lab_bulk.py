#!/usr/bin/env python3
"""
Bulk Marqo-only category trial runner.

Runs Marqo fashionSigLIP classification over an image folder and writes:
- per-case JSON summaries
- aggregate JSON
- one HTML report with previews and top matches
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.marqo_fashion_siglip_runner import MarqoFashionSiglipRunner
from shared.category_mapping import wardrobe_category_from_garment_type
from shared.marqo_category_taxonomy import MarqoCandidate, load_marqo_taxonomy
from utils.validation import normalize_garment_type


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic", ".heif"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "case"


def iter_images(input_dir: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        candidates = sorted(p for p in input_dir.rglob("*") if p.is_file())
    else:
        candidates = sorted(p for p in input_dir.iterdir() if p.is_file())
    for path in candidates:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _align_dim_to_grid(value: int, base: int) -> int:
    raw = max(int(value), int(base))
    return max(int(base), (raw // int(base)) * int(base))


def preprocess_for_analyze(image: Image.Image, *, max_edge: int = 768, grid_base: int = 16) -> Image.Image:
    rgb = image.convert("RGB")
    w, h = rgb.size
    longest = max(w, h)
    if longest != int(max_edge):
        ratio = float(max_edge) / float(max(1, longest))
        target_w = max(1, int(round(w * ratio)))
        target_h = max(1, int(round(h * ratio)))
        rgb = rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
    aligned_w = _align_dim_to_grid(rgb.width, grid_base)
    aligned_h = _align_dim_to_grid(rgb.height, grid_base)
    if (aligned_w, aligned_h) != rgb.size:
        rgb = rgb.resize((aligned_w, aligned_h), Image.Resampling.LANCZOS)
    return rgb


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def run_case(
    *,
    case_index: int,
    total: int,
    image_path: Path,
    output_dir: Path,
    runner: MarqoFashionSiglipRunner,
    candidates: List[MarqoCandidate],
    top_k: int,
    min_confidence: float,
    preprocess_max_edge: int,
    grid_base: int,
) -> Dict[str, Any]:
    case_id = f"{case_index:04d}_{slugify(image_path.stem)}"
    case_dir = ensure_dir(output_dir / case_id)
    source_copy = case_dir / image_path.name
    shutil.copy2(image_path, source_copy)

    with Image.open(image_path) as source:
        src_rgb = source.convert("RGB")
    preprocessed = preprocess_for_analyze(
        src_rgb,
        max_edge=preprocess_max_edge,
        grid_base=grid_base,
    )
    preprocessed_path = case_dir / "preprocessed_input.png"
    preprocessed.save(preprocessed_path, format="PNG")

    labels = [candidate.label for candidate in candidates]
    started = time.perf_counter()
    ranked_rows = runner.rank_labels(preprocessed, labels=labels, top_k=top_k)
    elapsed_s = round(time.perf_counter() - started, 4)

    by_label = {candidate.label: candidate for candidate in candidates}
    top_matches: List[Dict[str, Any]] = []
    for row in ranked_rows:
        label = str(row.get("label") or "")
        candidate = by_label.get(label)
        if candidate is None:
            continue
        top_matches.append(
            {
                "category_key": candidate.key,
                "category_label": candidate.label,
                "parent_key": candidate.parent_key,
                "parent_label": candidate.parent_label,
                "score": float(row.get("score") or 0.0),
            }
        )
    best = top_matches[0] if top_matches else {}
    best_score = float(best.get("score") or 0.0)
    applied = bool(best) and best_score >= float(min_confidence)

    summary = {
        "case_id": case_id,
        "source_image": str(image_path),
        "progress": f"{case_index}/{total}",
        "model_loaded": bool(runner.is_loaded),
        "model_available": bool(runner.is_available),
        "model_id": str(runner.model_id),
        "device": str(runner.device),
        "elapsed_s": elapsed_s,
        "candidate_count": len(candidates),
        "preprocess": {
            "grid_base": int(grid_base),
            "max_edge": int(preprocess_max_edge),
            "original_size": {"width": int(src_rgb.width), "height": int(src_rgb.height)},
            "preprocessed_size": {"width": int(preprocessed.width), "height": int(preprocessed.height)},
        },
        "threshold": float(min_confidence),
        "applied": bool(applied),
        "best": best,
        "top_matches": top_matches,
        "paths": {
            "source_image_copy": str(source_copy),
            "preprocessed_input_image": str(preprocessed_path),
        },
    }
    write_json(case_dir / "summary.json", summary)
    return summary


def build_html_report(rows: List[Dict[str, Any]], output_dir: Path, title: str) -> str:
    cards: List[str] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        source_image = Path(str(row.get("paths", {}).get("source_image_copy") or ""))
        preprocessed = Path(str(row.get("paths", {}).get("preprocessed_input_image") or ""))
        best = row.get("best") or {}
        top_matches = row.get("top_matches") or []

        def _img_block(label: str, path: Path) -> str:
            if not path.exists():
                return ""
            return (
                f'<div class="image-block"><div class="label">{html.escape(label)}</div>'
                f'<img src="{html.escape(_rel(path, output_dir))}" alt="{html.escape(label)}"></div>'
            )

        top_rows = []
        for item in top_matches[:5]:
            top_rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('category_key') or ''))}</td>"
                f"<td>{html.escape(str(item.get('category_label') or ''))}</td>"
                f"<td>{html.escape(str(item.get('parent_key') or ''))}</td>"
                f"<td>{float(item.get('score') or 0.0):.4f}</td>"
                "</tr>"
            )
        top_table = (
            "<table><thead><tr><th>key</th><th>label</th><th>parent</th><th>score</th></tr></thead>"
            f"<tbody>{''.join(top_rows)}</tbody></table>"
        )

        cards.append(
            f"""
            <section class="card">
              <div class="card-header">
                <h2>{html.escape(case_id)}</h2>
                <div class="meta">
                  <span>applied: {html.escape(str(row.get("applied") or False))}</span>
                  <span>best_key: {html.escape(str(best.get("category_key") or ""))}</span>
                  <span>best_score: {float(best.get("score") or 0.0):.4f}</span>
                  <span>latency: {html.escape(str(row.get("elapsed_s") or ""))}s</span>
                </div>
              </div>
              <div class="grid images">
                {_img_block("Source", source_image)}
                {_img_block("Preprocessed", preprocessed)}
              </div>
              <div class="grid metrics">
                <div class="metric"><strong>original_size</strong><br>{html.escape(str(row.get("preprocess", {}).get("original_size") or ""))}</div>
                <div class="metric"><strong>preprocessed_size</strong><br>{html.escape(str(row.get("preprocess", {}).get("preprocessed_size") or ""))}</div>
                <div class="metric"><strong>candidate_count</strong><br>{html.escape(str(row.get("candidate_count") or ""))}</div>
                <div class="metric"><strong>threshold</strong><br>{html.escape(str(row.get("threshold") or ""))}</div>
              </div>
              <div class="text-block">
                <div class="label">Top Matches</div>
                {top_table}
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #dbe3ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: #eef3fb;
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .meta {{ display: flex; gap: 10px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }}
    .grid {{ display: grid; gap: 10px; margin-top: 10px; }}
    .images {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .metrics {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .image-block {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #fbfdff;
    }}
    .image-block img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1px solid #e5e9f1;
    }}
    .label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 600;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fbfdff;
      font-size: 12px;
      color: var(--muted);
    }}
    .text-block {{
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fbfdff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid #e8edf5;
      padding: 6px 4px;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1 style="margin:0; font-size:20px;">{html.escape(title)}</h1>
      <div style="color:var(--muted); margin-top:4px;">Rows: {len(rows)} | Generated: {html.escape(datetime.now().isoformat(timespec="seconds"))}</div>
    </div>
  </header>
  <main class="wrap">
    {''.join(cards)}
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marqo-only bulk category trial")
    parser.add_argument("--input-dir", required=True, help="Folder containing input images")
    parser.add_argument("--output-dir", default="", help="Output folder (default tmp/marqo-lab-bulk/<timestamp>)")
    parser.add_argument("--garment-type", default="top", help="Garment lane: top|bottom|dress|outer")
    parser.add_argument("--lookup-csv", default="", help="Lookup CSV path")
    parser.add_argument("--model-id", default="Marqo/marqo-fashionSigLIP", help="Marqo model id")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K matches to keep")
    parser.add_argument("--min-confidence", type=float, default=0.22, help="Apply threshold")
    parser.add_argument("--max-edge", type=int, default=768, help="Preprocess max edge")
    parser.add_argument("--grid-base", type=int, default=16, help="Grid base alignment (8|16)")
    parser.add_argument("--recursive", action="store_true", help="Scan images recursively")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of images")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"input directory not found: {input_dir}")

    garment_type = normalize_garment_type(args.garment_type) or "top"
    lookup_csv = str(args.lookup_csv or "").strip() or "/Users/kiran/Downloads/drizzle-data-2026-04-20T09_27_42.748Z.csv"
    taxonomy = load_marqo_taxonomy(csv_path=lookup_csv)
    if not taxonomy.has_rows:
        raise SystemExit(f"no active taxonomy rows found in csv: {lookup_csv}")

    primary_hint = wardrobe_category_from_garment_type(garment_type)
    preferred_primary_key = str(primary_hint.get("primary_category_key") or "").strip()
    candidates = taxonomy.candidates_for_lane(
        garment_type,
        preferred_primary_key=preferred_primary_key,
    )
    if not candidates:
        raise SystemExit(f"no candidates found for lane={garment_type}, preferred_primary={preferred_primary_key}")

    images = list(iter_images(input_dir, recursive=bool(args.recursive)))
    if int(args.limit) > 0:
        images = images[: int(args.limit)]
    if not images:
        raise SystemExit("no images found")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else (Path("tmp") / "marqo-lab-bulk" / f"{ts}_{garment_type}").resolve()
    )
    ensure_dir(output_dir)

    runner = MarqoFashionSiglipRunner(model_id=args.model_id, device=args.device)
    runner.ensure_ready()

    rows: List[Dict[str, Any]] = []
    total = len(images)
    for idx, image_path in enumerate(images, start=1):
        row = run_case(
            case_index=idx,
            total=total,
            image_path=image_path,
            output_dir=output_dir,
            runner=runner,
            candidates=candidates,
            top_k=max(1, int(args.top_k)),
            min_confidence=float(args.min_confidence),
            preprocess_max_edge=max(64, int(args.max_edge)),
            grid_base=16 if int(args.grid_base) == 16 else 8,
        )
        rows.append(row)
        print(
            f"[{idx:04d}/{total:04d}] {image_path.name} | "
            f"best={row.get('best', {}).get('category_key', '')} "
            f"score={float((row.get('best') or {}).get('score') or 0.0):.4f} "
            f"applied={row.get('applied')}"
        )

    applied_count = sum(1 for row in rows if bool(row.get("applied")))
    aggregate = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "lookup_csv": lookup_csv,
        "garment_type": garment_type,
        "preferred_primary_key": preferred_primary_key,
        "candidate_count": len(candidates),
        "candidate_keys": [candidate.key for candidate in candidates],
        "total_images": len(rows),
        "applied_count": applied_count,
        "applied_rate": (float(applied_count) / float(len(rows))) if rows else 0.0,
        "model_loaded": bool(runner.is_loaded),
        "model_available": bool(runner.is_available),
        "model_id": str(runner.model_id),
        "device": str(runner.device),
        "rows": rows,
    }
    write_json(output_dir / "summary.json", aggregate)

    report_title = f"Marqo Bulk Trial | type={garment_type} | n={len(rows)}"
    (output_dir / "report.html").write_text(
        build_html_report(rows, output_dir, report_title),
        encoding="utf-8",
    )

    print(f"output_dir={output_dir}")
    print(f"summary_json={output_dir / 'summary.json'}")
    print(f"report_html={output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
