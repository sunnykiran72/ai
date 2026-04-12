#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path


def esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    results_path = Path(args.results_json).resolve()
    output_html = Path(args.output_html).resolve()
    base_dir = output_html.parent
    payload = json.loads(results_path.read_text())

    rows_by_image = defaultdict(dict)
    for row in payload["rows"]:
        rows_by_image[row["image_name"]][row["model"]] = row

    preferred_order = [
        "Real-ESRGAN_x4plus",
        "UltraSharpV2",
        "AuraSR-v2",
        "Swin2SR_realworld_x4",
    ]
    discovered_models = list(payload["summary"].keys())
    preferred_present = [name for name in preferred_order if name in discovered_models]
    extra_models = sorted([name for name in discovered_models if name not in preferred_present], key=str.lower)
    model_order = preferred_present + extra_models

    summary_rows = []
    for model_name in model_order:
        summary = payload["summary"].get(model_name, {})
        summary_rows.append(
            "<tr>"
            f"<td>{esc(model_name)}</td>"
            f"<td>{esc(str(summary.get('load_seconds', '')))}</td>"
            f"<td>{esc(str(summary.get('avg_inference_seconds', '')))}</td>"
            f"<td>{esc(str(summary.get('completed_images', '')))}</td>"
            f"<td>{esc(str(summary.get('failed_images', '')))}</td>"
            "</tr>"
        )

    settings_cards = []
    for model_name in model_order:
        metadata = payload["settings"].get(model_name, {})
        settings_json = json.dumps(metadata.get("settings", {}), indent=2, ensure_ascii=False)
        settings_cards.append(
            "<section class='settings-card'>"
            f"<h3>{esc(model_name)}</h3>"
            f"<p>Status: {esc(metadata.get('status', 'unknown'))}</p>"
            f"<pre>{esc(settings_json)}</pre>"
            "</section>"
        )

    sections = []
    for image_name in payload["images"]:
        entries = rows_by_image[image_name]
        original_path = results_path.parent.parent / "inputs" / image_name
        cards = [
            "<article class='card'>"
            "<h4>Original</h4>"
            f"<img src='{esc(rel(original_path, base_dir))}' alt='{esc(image_name)}' loading='lazy'>"
            f"<p>{esc(image_name)}</p>"
            "</article>"
        ]
        for model_name in model_order:
            row = entries.get(model_name)
            if not row:
                cards.append(
                    "<article class='card'>"
                    f"<h4>{esc(model_name)}</h4>"
                    "<p>Missing result</p>"
                    "</article>"
                )
                continue
            if row["status"] != "ok":
                cards.append(
                    "<article class='card error'>"
                    f"<h4>{esc(model_name)}</h4>"
                    f"<p>{esc(row['error'])}</p>"
                    "</article>"
                )
                continue
            output_path = Path(row["output_path"])
            cards.append(
                "<article class='card'>"
                f"<h4>{esc(model_name)}</h4>"
                f"<img src='{esc(rel(output_path, base_dir))}' alt='{esc(model_name)}' loading='lazy'>"
                f"<p>{row['inference_seconds']}s inference</p>"
                f"<p>{row['output_width']}x{row['output_height']}</p>"
                "</article>"
            )
        sections.append(
            "<section class='image-section'>"
            f"<h2>{esc(image_name)}</h2>"
            "<div class='grid'>"
            + "".join(cards)
            + "</div></section>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Low-Res Upscaler Benchmark</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --line: #d9ccb8;
      --text: #201913;
      --muted: #6a5a4f;
      --accent: #9e5b2a;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #efe4d0 0%, var(--bg) 35%, #f8f3ec 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 12px; }}
    p {{ margin: 0 0 10px; color: var(--muted); }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      margin-bottom: 28px;
    }}
    .summary-table th, .summary-table td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }}
    .settings {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .settings-card, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 8px 30px rgba(66, 43, 18, 0.06);
    }}
    .settings-card pre {{
      overflow-x: auto;
      white-space: pre-wrap;
      font-size: 12px;
      color: var(--muted);
    }}
    .image-section {{
      margin-bottom: 26px;
      padding-top: 20px;
      border-top: 1px solid rgba(158, 91, 42, 0.2);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
    }}
    .card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 12px;
      background: #ebe3d4;
      margin-bottom: 10px;
    }}
    .card.error {{
      border-color: #c06a55;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Low-Res Upscaler Benchmark</h1>
    <p>Final outputs are capped to a max side of 1024px after each model's native 4x pass.</p>
    <table class="summary-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Load Seconds</th>
          <th>Avg Inference Seconds</th>
          <th>Completed</th>
          <th>Failed</th>
        </tr>
      </thead>
      <tbody>
        {"".join(summary_rows)}
      </tbody>
    </table>
    <section class="settings">
      {"".join(settings_cards)}
    </section>
    {"".join(sections)}
  </main>
</body>
</html>
"""
    output_html.write_text(html)


if __name__ == "__main__":
    main()
