#!/usr/bin/env python3
"""
Generate a side-by-side HTML comparison report for two or three completed bulk try-on runs.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _success_map(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    mapped: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = str(row.get("user_file") or "").strip()
        if not key:
            continue
        if str(row.get("status") or "").strip().lower() != "success":
            continue
        if not str(row.get("user_input_url") or "").strip():
            continue
        if not str(row.get("output_url") or "").strip():
            continue
        mapped[key] = row
    return mapped


def _summary_chip(label: str, value: str) -> str:
    return (
        '<div class="chip">'
        f'<span class="k">{html.escape(label)}</span>'
        f'<span class="v">{html.escape(value)}</span>'
        "</div>"
    )


def _format_source_worn_types(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return "Unavailable in this run"
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        cleaned = [str(item).strip() for item in parsed if str(item).strip()]
        return ", ".join(cleaned) if cleaned else "Unavailable in this run"
    return text


def _build_html(
    *,
    title: str,
    labels: List[str],
    rows: List[Tuple[int, str, Dict[str, str], List[Dict[str, str]]]],
    summaries: List[Dict[str, object]],
    include_prompts: bool,
) -> str:
    image_grid_class = "image-grid-4" if len(labels) >= 3 else "image-grid-3"
    prompt_css = ""
    prompt_mobile_css = ""
    if include_prompts:
        prompt_css = """
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
      margin-bottom: 12px;
    }}
    .meta-block {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fcfbf8;
      padding: 12px;
      min-width: 0;
    }}
    .meta-block.wide {{
      grid-column: 1 / -1;
    }}
    .meta-value {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
      color: #2a2824;
      font-family: "SFMono-Regular", "Menlo", monospace;
    }}
    .prompt-details {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--surface-soft);
      padding: 12px 14px;
    }}
    .prompt-details summary {{
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
      color: var(--ink);
    }}
    .prompt-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .prompt-block {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #f6f4ef;
      padding: 12px;
      min-width: 0;
    }}
    .prompt-label {{
      margin-bottom: 8px;
      font-size: 12px;
      line-height: 1.2;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .prompt-block pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
      color: #2a2824;
      max-height: 260px;
      overflow: auto;
      font-family: "SFMono-Regular", "Menlo", monospace;
    }}
"""
        prompt_mobile_css = """
      .meta-grid {{
        grid-template-columns: 1fr;
      }}
      .prompt-grid {{
        grid-template-columns: 1fr;
      }}
"""
    garment_url = ""
    garment_type = ""
    for summary in summaries:
        settings = summary.get("settings") if isinstance(summary.get("settings"), dict) else {}
        if not garment_url:
            garment_url = str(settings.get("garment_url") or "").strip()
        if not garment_type:
            garment_type = str(settings.get("garment_type") or "").strip()
    if not garment_url:
        for _idx, _user_file, input_row, _output_rows in rows:
            garment_url = str(input_row.get("garment_url") or "").strip()
            garment_type = garment_type or str(input_row.get("garment_type") or "").strip()
            if garment_url:
                break

    cards: List[str] = []
    for idx, user_file, input_row, output_rows in rows:
        input_url = str(input_row.get("user_input_url") or "").strip()
        source_worn_types = _format_source_worn_types(
            str(input_row.get("worn_types") or output_rows[0].get("worn_types") or "").strip()
        )
        user_prompt = str(input_row.get("user_prompt") or "").strip() or "n/a"
        garment_prompt = str(input_row.get("garment_prompt") or "").strip() or "n/a"
        garment_kind = str(input_row.get("garment_type") or "").strip() or "n/a"
        figures = [
            f"""
    <figure>
      <figcaption>Input</figcaption>
      <a href="{html.escape(input_url)}" target="_blank" rel="noopener">
        <img src="{html.escape(input_url)}" alt="input user" loading="lazy" />
      </a>
    </figure>
"""
        ]
        prompt_blocks: List[str] = []
        for label, row in zip(labels, output_rows):
            output_url = str(row.get("output_url") or "").strip()
            figures.append(
                f"""
    <figure>
      <figcaption>{html.escape(label)}</figcaption>
      <a href="{html.escape(output_url)}" target="_blank" rel="noopener">
        <img src="{html.escape(output_url)}" alt="{html.escape(label)} output" loading="lazy" />
      </a>
    </figure>
"""
            )
            if include_prompts:
                prompt_used = str(row.get("prompt_used") or "").strip()
                prompt_blocks.append(
                    f"""
      <div class="prompt-block">
        <div class="prompt-label">{html.escape(label)} Final Prompt</div>
        <pre>{html.escape(prompt_used or "n/a")}</pre>
      </div>
"""
                )
        prompt_section = ""
        if include_prompts:
            prompt_section = f"""
  <details class="prompt-details">
    <summary>View Local Details</summary>
    <div class="meta-grid">
      <div class="meta-block">
        <div class="prompt-label">User Garment Type</div>
        <pre class="meta-value">{html.escape(source_worn_types)}</pre>
      </div>
      <div class="meta-block">
        <div class="prompt-label">Selected Garment Type</div>
        <pre class="meta-value">{html.escape(garment_kind)}</pre>
      </div>
      <div class="meta-block">
        <div class="prompt-label">Prepared User Prompt</div>
        <pre class="meta-value">{html.escape(user_prompt)}</pre>
      </div>
      <div class="meta-block wide">
        <div class="prompt-label">Garment Prompt</div>
        <pre class="meta-value">{html.escape(garment_prompt)}</pre>
      </div>
    </div>
    <div class="prompt-grid">
      {''.join(prompt_blocks)}
    </div>
  </details>
"""
        cards.append(
            f"""
<section class="case-card">
  <div class="head">
    <div>
      <div class="eyebrow">Case {idx:03d}</div>
      <h3>{html.escape(user_file)}</h3>
    </div>
    <span class="badge">Compared</span>
  </div>
  <div class="image-grid {image_grid_class}">
    {''.join(figures)}
  </div>
  {prompt_section}
</section>
"""
        )

    summary_chips = [_summary_chip("Cases Compared", str(len(rows)))]
    for idx, label in enumerate(labels):
        summary = summaries[idx] if idx < len(summaries) else {}
        cases_completed = summary.get("total_images") or summary.get("count_total") or len(rows)
        summary_chips.append(_summary_chip(f"Variation {idx + 1}", label))
        summary_chips.append(_summary_chip(f"{label} Images", str(cases_completed)))
    summary_grid = "\n".join(summary_chips)
    garment_panel = ""
    if garment_url:
        garment_label = garment_type.title() if garment_type else "Garment"
        garment_panel = f"""
        <div class="garment-panel">
          <div class="garment-eyebrow">Garment</div>
          <h2>{html.escape(garment_label)}</h2>
          <a class="garment-link" href="{html.escape(garment_url)}" target="_blank" rel="noopener">
            <img src="{html.escape(garment_url)}" alt="{html.escape(garment_label)} reference" loading="lazy" />
          </a>
        </div>
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f7f5;
      --surface: #ffffff;
      --surface-soft: #fbfbfa;
      --line: #e7e4df;
      --ink: #181713;
      --muted: #6b655c;
      --accent: #0f766e;
      --badge-bg: #eef7f5;
      --shadow: 0 18px 40px rgba(24, 23, 19, 0.06);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Instrument Sans", "Manrope", "Avenir Next", sans-serif;
      background:
        radial-gradient(circle at top left, #fffdfa 0%, transparent 30%),
        linear-gradient(180deg, #faf9f6 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1880px;
      margin: 0 auto;
      padding: 24px 18px 48px;
    }}
    .hero {{
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(180deg, #fff 0%, #fcfcfb 100%);
      padding: 22px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 38px;
      line-height: 1;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
      max-width: 980px;
    }}
    .hero-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
    }}
    .hero-copy {{
      min-width: 0;
      flex: 1 1 auto;
    }}
    .garment-panel {{
      width: 240px;
      flex: 0 0 240px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: var(--surface-soft);
      padding: 14px;
    }}
    .garment-eyebrow {{
      margin-bottom: 6px;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .garment-panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      line-height: 1.15;
    }}
    .garment-link {{
      display: block;
    }}
    .garment-link img {{
      width: 100%;
      height: 280px;
      object-fit: contain;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 16px;
      display: block;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(14, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--surface);
      padding: 12px 14px;
    }}
    .chip .k {{
      display: block;
      margin-bottom: 4px;
      font-size: 11px;
      line-height: 1.2;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .chip .v {{
      display: block;
      font-size: 18px;
      line-height: 1.1;
      font-weight: 700;
      color: var(--ink);
    }}
    .gallery {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    .case-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--surface);
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      margin-bottom: 6px;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .case-card h3 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
      word-break: break-word;
    }}
    .badge {{
      font-size: 11px;
      background: var(--badge-bg);
      color: var(--accent);
      border: 1px solid rgba(15, 118, 110, 0.18);
      border-radius: 999px;
      padding: 6px 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }}
{prompt_css}
    .image-grid {{
      display: grid;
      gap: 12px;
    }}
    .image-grid-3 {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .image-grid-4 {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 10px;
      background: var(--surface-soft);
    }}
    figcaption {{
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    img {{
      width: 100%;
      height: 520px;
      object-fit: contain;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      display: block;
    }}
    a {{
      text-decoration: none;
      color: inherit;
    }}
    @media (max-width: 1480px) {{
      .summary-grid {{
        grid-template-columns: repeat(4, minmax(120px, 1fr));
      }}
      img {{
        height: 420px;
      }}
    }}
    @media (max-width: 900px) {{
      .hero-top {{
        flex-direction: column;
      }}
      .garment-panel {{
        width: 100%;
        flex-basis: auto;
      }}
      .image-grid-3,
      .image-grid-4 {{
        grid-template-columns: 1fr;
      }}
{prompt_mobile_css}
      .summary-grid {{
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }}
      .wrap {{
        padding: 16px 12px 32px;
      }}
      .hero {{
        padding: 16px;
      }}
      .hero h1 {{
        font-size: 30px;
      }}
      img {{
        height: 340px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-top">
        <div class="hero-copy">
          <h1>{html.escape(title)}</h1>
          <p>Side-by-side review of the same input set with seed variations only. Each card shows the original input image and the corresponding outputs for direct visual comparison.</p>
        </div>
        {garment_panel}
      </div>
      <div class="summary-grid">
        {summary_grid}
      </div>
    </section>
    <div class="gallery">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate side-by-side comparison HTML for two or three bulk try-on runs.")
    parser.add_argument("--left-run-dir", required=True, help="Directory under debug_outputs for the left run")
    parser.add_argument("--right-run-dir", required=True, help="Directory under debug_outputs for the right run")
    parser.add_argument("--third-run-dir", default="", help="Optional third directory under debug_outputs")
    parser.add_argument("--left-label", default="Seed 44")
    parser.add_argument("--right-label", default="Seed 77")
    parser.add_argument("--third-label", default="Seed 123")
    parser.add_argument("--title", default="Glamify Bulk Tryon Comparison")
    parser.add_argument("--output-run-dir", required=True, help="Destination directory under debug_outputs")
    parser.add_argument("--include-prompts", action="store_true", help="Include per-case prompt details in the HTML")
    args = parser.parse_args()

    debug_outputs = Path(__file__).resolve().parent.parent / "debug_outputs"
    left_dir = (debug_outputs / str(args.left_run_dir).strip()).resolve()
    right_dir = (debug_outputs / str(args.right_run_dir).strip()).resolve()
    output_dir = (debug_outputs / str(args.output_run_dir).strip()).resolve()
    third_run_dir = str(args.third_run_dir or "").strip()
    third_dir: Optional[Path] = (debug_outputs / third_run_dir).resolve() if third_run_dir else None

    candidates = [left_dir, right_dir, output_dir]
    if third_dir is not None:
        candidates.append(third_dir)
    for candidate in candidates:
        try:
            candidate.relative_to(debug_outputs.resolve())
        except ValueError as exc:
            raise SystemExit(f"Directory escapes debug_outputs: {candidate}") from exc

    run_maps = [
        _success_map(_read_rows(left_dir / "results.csv")),
        _success_map(_read_rows(right_dir / "results.csv")),
    ]
    labels = [str(args.left_label), str(args.right_label)]
    summaries = [
        _read_summary(left_dir / "summary.json"),
        _read_summary(right_dir / "summary.json"),
    ]

    if third_dir is not None:
        run_maps.append(_success_map(_read_rows(third_dir / "results.csv")))
        labels.append(str(args.third_label))
        summaries.append(_read_summary(third_dir / "summary.json"))

    shared_files = sorted(set.intersection(*(set(run_map) for run_map in run_maps)))
    merged_rows: List[Tuple[int, str, Dict[str, str], List[Dict[str, str]]]] = []
    for idx, user_file in enumerate(shared_files, start=1):
        merged_rows.append((idx, user_file, run_maps[0][user_file], [run_map[user_file] for run_map in run_maps]))

    output_dir.mkdir(parents=True, exist_ok=True)
    html_text = _build_html(
        title=str(args.title),
        labels=labels,
        rows=merged_rows,
        summaries=summaries,
        include_prompts=bool(args.include_prompts),
    )
    (output_dir / "report.html").write_text(html_text, encoding="utf-8")

    manifest = {
        "title": args.title,
        "left_run_dir": args.left_run_dir,
        "right_run_dir": args.right_run_dir,
        "third_run_dir": third_run_dir,
        "left_label": args.left_label,
        "right_label": args.right_label,
        "third_label": args.third_label if third_run_dir else "",
        "cases_compared": len(merged_rows),
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote_report={output_dir / 'report.html'}")
    print(f"cases_compared={len(merged_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
