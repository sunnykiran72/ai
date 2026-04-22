from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse


router = APIRouter()

_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "tmp" / "pod_pull"
_APP_ROOT = Path(__file__).resolve().parent.parent


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _detect_variation(run_dir: str, title: str, garment_mode: str) -> str:
    token = f"{run_dir} {title}".lower()
    if "swap1" in token or "swap_1" in token:
        return "swap_1"
    if "swap2" in token or "swap_2" in token:
        return "swap_2"
    if "set2" in token:
        return "set_2"
    if "set1" in token:
        return "set_1"
    if garment_mode == "dress":
        return "dress"
    if "prod" in token:
        return "prod"
    return "other"


def _pretty_label(run: Dict[str, object]) -> str:
    variation = str(run.get("variation") or "other")
    garment_mode = str(run.get("garment_mode") or "")
    if variation == "set_1":
        return "Set 1"
    if variation == "set_2":
        return "Set 2"
    if variation == "swap_1":
        return "Swap 1"
    if variation == "swap_2":
        return "Swap 2"
    if variation == "dress":
        seed = str(run.get("seed") or "")
        title = str(run.get("title") or "")
        if "ring" in title.lower():
            return "Dress 2"
        if "generic" in title.lower():
            return "Dress 3"
        return "Dress 1"
    if variation == "prod":
        return "Prod Run"
    return "Other"


def _collect_result_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    dirs: List[Path] = []
    for results_csv in root.rglob("results.csv"):
        parent = results_csv.parent
        if (parent / "summary.json").exists():
            dirs.append(parent)
    dirs.sort(key=lambda p: str(p))
    return dirs


def _load_summary(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_results(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _resolve_local_report(report: str) -> Path:
    raw = str(report or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="missing report")
    relative = raw.lstrip("/")
    candidate = (_APP_ROOT / relative).resolve()
    try:
        candidate.relative_to(_APP_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid report path") from exc
    if candidate.name != "report.html" or not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return candidate


def _build_dataset() -> Dict[str, object]:
    runs: List[Dict[str, object]] = []
    entries: List[Dict[str, object]] = []
    root = _RESULTS_ROOT
    for run_path in _collect_result_dirs(root):
        summary = _load_summary(run_path / "summary.json")
        rows = _load_results(run_path / "results.csv")
        run_name = run_path.name
        title = str(summary.get("title") or run_name)
        garment_mode = str(summary.get("garment_mode") or ("dress" if str(summary.get("dress_image_url") or "").strip() else "top_bottom"))
        variation = _detect_variation(run_name, title, garment_mode)
        run_record = {
            "run_name": run_name,
            "relative_dir": str(run_path.relative_to(root.parent.parent)),
            "title": title,
            "generated_at": str(summary.get("generated_at") or ""),
            "seed": _safe_int(summary.get("seed")),
            "steps": _safe_int(summary.get("steps")),
            "guidance_scale": _safe_float(summary.get("guidance_scale")),
            "lora_scale": _safe_float(summary.get("lora_scale")),
            "garment_mode": garment_mode,
            "variation": variation,
            "total": _safe_int(summary.get("total")),
            "success": _safe_int(summary.get("success")),
            "error": _safe_int(summary.get("error")),
            "report_path": f"/{run_path.relative_to(root.parent.parent).as_posix()}/report.html",
            "report_view_path": "",
            "top_image_url": str(summary.get("top_image_url") or ""),
            "bottom_image_url": str(summary.get("bottom_image_url") or ""),
            "dress_image_url": str(summary.get("dress_image_url") or ""),
            "top_prompt": str(summary.get("top_prompt") or ""),
            "bottom_prompt": str(summary.get("bottom_prompt") or ""),
            "dress_prompt": str(summary.get("dress_prompt") or ""),
            "sample_input_url": str(rows[0].get("prepared_input_url") or "") if rows else "",
            "sample_output_url": str(rows[0].get("upscale_output_url") or rows[0].get("tryon_output_url") or "") if rows else "",
        }
        run_record["report_view_path"] = f"/v1/tryon-report-view?report={quote(str(run_record['report_path']))}"
        run_record["display_label"] = _pretty_label(run_record)
        runs.append(run_record)
        for row in rows:
            entries.append(
                {
                    "run_name": run_name,
                    "title": title,
                    "variation": variation,
                    "garment_mode": garment_mode,
                    "generated_at": str(summary.get("generated_at") or ""),
                    "seed": _safe_int(summary.get("seed")),
                    "steps": _safe_int(summary.get("steps")),
                    "guidance_scale": _safe_float(summary.get("guidance_scale")),
                    "lora_scale": _safe_float(summary.get("lora_scale")),
                    "index": _safe_int(row.get("index")),
                    "user_file": str(row.get("user_file") or ""),
                    "prepared_input_url": str(row.get("prepared_input_url") or ""),
                    "user_prompt": str(row.get("user_prompt") or ""),
                    "tryon_output_url": str(row.get("tryon_output_url") or ""),
                    "upscale_output_url": str(row.get("upscale_output_url") or ""),
                    "status": str(row.get("status") or ""),
                    "error": str(row.get("error") or ""),
                    "prompt_used": str(row.get("prompt_used") or ""),
                    "tryon_latency_s": _safe_float(row.get("tryon_latency_s")),
                    "upscale_latency_s": _safe_float(row.get("upscale_latency_s")),
                    "total_latency_s": _safe_float(row.get("total_latency_s")),
                    "top_image_url": str(row.get("top_image_url") or summary.get("top_image_url") or ""),
                    "bottom_image_url": str(row.get("bottom_image_url") or summary.get("bottom_image_url") or ""),
                    "dress_image_url": str(row.get("dress_image_url") or summary.get("dress_image_url") or ""),
                    "top_prompt": str(row.get("top_prompt") or summary.get("top_prompt") or ""),
                    "bottom_prompt": str(row.get("bottom_prompt") or summary.get("bottom_prompt") or ""),
                    "dress_prompt": str(row.get("dress_prompt") or summary.get("dress_prompt") or ""),
                    "report_path": f"/{run_path.relative_to(root.parent.parent).as_posix()}/report.html",
                    "report_view_path": f"/v1/tryon-report-view?report={quote(f'/{run_path.relative_to(root.parent.parent).as_posix()}/report.html')}",
                }
            )
    runs.sort(key=lambda item: (str(item["generated_at"]), str(item["run_name"])))
    entries.sort(key=lambda item: (item["index"], item["run_name"]))
    return {
        "root": str(root),
        "run_count": len(runs),
        "entry_count": len(entries),
        "runs": runs,
        "entries": entries,
    }


@router.get("/v1/tryon-results/data")
async def tryon_results_data() -> JSONResponse:
    return JSONResponse(content=_build_dataset())


@router.get("/v1/tryon-report-view", response_class=HTMLResponse)
async def tryon_report_view(
    report: str = Query(...),
    showPrompts: bool = Query(False),
) -> HTMLResponse:
    report_path = _resolve_local_report(report)
    raw_html = report_path.read_text(encoding="utf-8")
    title = html.escape(report_path.parent.name)
    report_href = "/" + report_path.relative_to(_APP_ROOT).as_posix()
    toolbar = f"""
<div class="tryon-report-toolbar">
  <div class="toolbar-title">
    <strong>{title}</strong>
    <span>{"Prompt debug mode" if showPrompts else "Prompts hidden"}</span>
  </div>
  <div class="toolbar-actions">
    <a href="/v1/tryon-reports">Back to reports</a>
  </div>
</div>
"""
    inject_css = """
<style>
  body { padding-top: 78px; }
  .tryon-report-toolbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    display: flex;
    gap: 12px;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    background: rgba(22, 20, 18, 0.94);
    color: #f6f1e8;
    border-bottom: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(8px);
  }
  .tryon-report-toolbar .toolbar-title {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    font: 14px/1.4 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .tryon-report-toolbar .toolbar-title span { color: rgba(246,241,232,0.78); }
  .tryon-report-toolbar .toolbar-actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
  }
  .tryon-report-toolbar .toolbar-actions a {
    color: #f6f1e8;
    text-decoration: none;
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 8px 12px;
    font: 13px/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .tryon-report-toolbar .toolbar-actions a:hover { background: rgba(255,255,255,0.08); }
  .tryon-report-toolbar .toolbar-actions a:last-child {
    background: #f6f1e8;
    color: #161412;
    border-color: #f6f1e8;
  }
  @media (max-width: 900px) {
    body { padding-top: 124px; }
    .tryon-report-toolbar { align-items: flex-start; flex-direction: column; }
    .tryon-report-toolbar .toolbar-actions a { flex: 1 1 auto; }
  }
</style>
"""
    hidden_prompts_css = """
<style>
  .product-card pre,
  .prompt-details {
    display: none !important;
  }
</style>
""" if not showPrompts else ""
    output = raw_html
    if "</head>" in output:
        output = output.replace("</head>", f"{inject_css}{hidden_prompts_css}</head>", 1)
    if "<body>" in output:
        output = output.replace("<body>", f"<body>{toolbar}", 1)
    else:
        output = toolbar + output
    return HTMLResponse(content=output)


@router.get("/v1/tryon-reports", response_class=HTMLResponse)
async def tryon_reports_page() -> HTMLResponse:
    dataset = _build_dataset()
    runs = list(dataset.get("runs") or [])
    payload = json.dumps(runs, ensure_ascii=False)
    run_count = len(runs)
    html_doc = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Try-on Reports</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffdf9;
      --line: #d8cec0;
      --ink: #1e1a16;
      --muted: #746a5c;
      --accent: #a33d2d;
      --accent-soft: #f3ddd7;
      --chip: #eee5d8;
      --shadow: 0 8px 24px rgba(30,26,22,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(163,61,45,0.12), transparent 28%),
        linear-gradient(180deg, #f8f3ec 0%, var(--bg) 100%);
    }}
    .topbar {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.05;
    }}
    .lede {{
      color: var(--muted);
      font-size: 14px;
      max-width: 760px;
      line-height: 1.5;
    }}
    .count {{
      color: var(--muted);
      font-size: 14px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
      background: rgba(255,255,255,0.5);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    label {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    input, select {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .card-link {{
      color: inherit;
      text-decoration: none;
      display: block;
    }}
    .card-link:hover .card {{
      transform: translateY(-1px);
      box-shadow: 0 12px 30px rgba(30,26,22,0.12);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .chip {{
      background: var(--chip);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .chip.accent {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 19px;
      line-height: 1.2;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .stats {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .preview-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-top: 14px;
    }}
    .preview-grid.dress {{
      grid-template-columns: repeat(3, 1fr);
    }}
    .preview-grid img {{
      width: 100%;
      aspect-ratio: 0.75;
      object-fit: cover;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #f2ede4;
    }}
    .preview-grid figure {{
      margin: 0;
    }}
    .preview-grid figcaption {{
      font-size: 11px;
      color: var(--muted);
      margin-top: 5px;
    }}
    .actions {{
      margin-top: 14px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    .report-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 180px;
      padding: 11px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    .report-link:hover {{
      background: var(--accent-soft);
      border-color: #d9b6ae;
    }}
    .muted-note {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 16px;
    }}
    @media (max-width: 1024px) {{
      body {{
        padding: 20px;
      }}
      .grid {{
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      }}
      .preview-grid {{
        grid-template-columns: repeat(2, 1fr);
      }}
      .preview-grid.dress {{
        grid-template-columns: repeat(2, 1fr);
      }}
    }}
    @media (max-width: 640px) {{
      body {{
        padding: 14px;
      }}
      .topbar {{
        margin-bottom: 16px;
      }}
      h1 {{
        font-size: 28px;
      }}
      .toolbar {{
        grid-template-columns: 1fr;
        padding: 12px;
      }}
      .grid {{
        grid-template-columns: 1fr;
        gap: 14px;
      }}
      .card {{
        padding: 14px;
      }}
      .stats {{
        gap: 8px;
      }}
      .preview-grid,
      .preview-grid.dress {{
        grid-template-columns: 1fr;
      }}
      .preview-grid img {{
        aspect-ratio: 0.9;
      }}
      .actions a {{
        display: inline-block;
        width: 100%;
        text-align: center;
        padding: 10px 12px;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: #fff;
      }}
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1>Try-on Reports</h1>
      <div class="lede">A clean index of generated batch reports. This page only links to the batch <code>report.html</code> files so you can quickly open a run without prompt or per-image clutter.</div>
    </div>
    <div class="count" id="count">__RUN_COUNT__ runs</div>
  </div>
  <div class="toolbar">
    <div>
      <label for="search">Search</label>
      <input id="search" placeholder="run name, label..." />
    </div>
    <div>
      <label for="type-filter">Type</label>
      <select id="type-filter"><option value="">All types</option></select>
    </div>
    <div>
      <label for="seed-filter">Seed</label>
      <select id="seed-filter"><option value="">All seeds</option></select>
    </div>
    <div>
      <label for="steps-filter">Steps</label>
      <select id="steps-filter"><option value="">All steps</option></select>
    </div>
    <div>
      <label for="min-total-filter">Min Total Results</label>
      <input id="min-total-filter" type="number" min="0" step="1" placeholder="e.g. 100" />
    </div>
  </div>
  <div class="muted-note">Each card shows the sample input image, the garment reference image(s), one sample output, and a direct link to the full batch report.</div>
  <div class="grid" id="grid"></div>
  <script>
    const RUNS = __RUNS__;
    const grid = document.getElementById('grid');
    const count = document.getElementById('count');
    const search = document.getElementById('search');
    const typeFilter = document.getElementById('type-filter');
    const seedFilter = document.getElementById('seed-filter');
    const stepsFilter = document.getElementById('steps-filter');
    const minTotalFilter = document.getElementById('min-total-filter');

    function uniq(values) {
      return Array.from(new Set(values.filter(v => v !== '' && v !== null && v !== undefined)))
        .sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
    }

    function fillSelect(select, values) {
      values.forEach(v => {
        const opt = document.createElement('option');
        opt.value = String(v);
        opt.textContent = String(v);
        select.appendChild(opt);
      });
    }

    fillSelect(typeFilter, uniq(RUNS.map(r => r.display_label)));
    fillSelect(seedFilter, uniq(RUNS.map(r => r.seed)));
    fillSelect(stepsFilter, uniq(RUNS.map(r => r.steps)));

    function fileNameFromUrl(url) {
      if (!url) return '';
      try {
        return String(url).split('/').pop() || '';
      } catch {
        return '';
      }
    }

    function matches(run) {
      const q = search.value.trim().toLowerCase();
      const type = typeFilter.value;
      const seed = seedFilter.value;
      const steps = stepsFilter.value;
      const minTotal = Number(minTotalFilter.value || 0);
      if (q) {
        const hay = [run.run_name, run.display_label, run.generated_at].join(' ').toLowerCase();
        if (!hay.includes(q)) return false;
      }
      if (type && run.display_label !== type) return false;
      if (seed && String(run.seed) !== seed) return false;
      if (steps && String(run.steps) !== steps) return false;
      if (Number(run.total || 0) < minTotal) return false;
      return true;
    }

    function render() {
      const filtered = RUNS.filter(matches);
      count.textContent = `${filtered.length} runs`;
      if (!filtered.length) {
        grid.innerHTML = '<div class="card">No matching runs.</div>';
        return;
      }
      grid.innerHTML = filtered.map(run => {
        const garmentLabel = run.garment_mode === 'dress' ? 'Dress' : 'Top + Bottom';
        const referenceBlock = run.garment_mode === 'dress'
          ? `
            <div class="preview-grid dress">
              <figure><img src="${run.sample_input_url || ''}" alt="sample input"><figcaption>Input<br>${fileNameFromUrl(run.sample_input_url)}</figcaption></figure>
              <figure><img src="${run.dress_image_url || ''}" alt="dress reference"><figcaption>Dress ref<br>${fileNameFromUrl(run.dress_image_url)}</figcaption></figure>
              <figure><img src="${run.sample_output_url || ''}" alt="sample output"><figcaption>Sample output</figcaption></figure>
            </div>`
          : `
            <div class="preview-grid">
              <figure><img src="${run.sample_input_url || ''}" alt="sample input"><figcaption>Input<br>${fileNameFromUrl(run.sample_input_url)}</figcaption></figure>
              <figure><img src="${run.top_image_url || ''}" alt="top reference"><figcaption>Top ref<br>${fileNameFromUrl(run.top_image_url)}</figcaption></figure>
              <figure><img src="${run.bottom_image_url || ''}" alt="bottom reference"><figcaption>Bottom ref<br>${fileNameFromUrl(run.bottom_image_url)}</figcaption></figure>
              <figure><img src="${run.sample_output_url || ''}" alt="sample output"><figcaption>Sample output</figcaption></figure>
            </div>`;
        const href = run.report_view_path || run.report_path || '#';
        return `
          <article class="card">
            <a class="card-link" href="${href}">
              <div class="chips">
                <span class="chip accent">${run.display_label || 'Other'}</span>
                <span class="chip">${garmentLabel}</span>
                <span class="chip">seed ${run.seed ?? ''}</span>
                <span class="chip">steps ${run.steps ?? ''}</span>
              </div>
              <h2>${run.display_label || run.run_name || ''}</h2>
              <div class="meta">${run.generated_at || ''}</div>
              <div class="meta">${run.run_name || ''}</div>
              <div class="stats">
                <span>total ${run.total || 0}</span>
                <span>success ${run.success || 0}</span>
                <span>error ${run.error || 0}</span>
              </div>
              ${referenceBlock}
            </a>
            <div class="actions">
              <a class="report-link" href="${href}">Open report.html</a>
            </div>
          </article>`;
      }).join('');
    }

    [search, typeFilter, seedFilter, stepsFilter, minTotalFilter].forEach(el => {
      el.addEventListener('input', render);
      el.addEventListener('change', render);
    });

    render();
  </script>
</body>
</html>
"""
    html_doc = (
        html_doc.replace("{{", "{")
        .replace("}}", "}")
        .replace("__RUN_COUNT__", str(run_count))
        .replace("__RUNS__", payload)
    )
    return HTMLResponse(content=html_doc)


@router.get("/v1/tryon-results", response_class=HTMLResponse)
async def tryon_results_page() -> HTMLResponse:
    dataset = _build_dataset()
    payload = json.dumps(dataset, ensure_ascii=False)
    run_count = int(dataset.get("run_count") or 0)
    entry_count = int(dataset.get("entry_count") or 0)
    html_doc = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Try-on Results Dashboard</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --line: #d8ceb9;
      --ink: #1f1c17;
      --muted: #6e6558;
      --accent: #ac3b2f;
      --accent-soft: #f3d9d2;
      --chip: #efe7d8;
      --shadow: 0 10px 30px rgba(31, 28, 23, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(172,59,47,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(85,101,78,0.14), transparent 26%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }}
    .page {{
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      padding: 24px 20px;
      background: rgba(255,253,248,0.82);
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      backdrop-filter: blur(8px);
    }}
    .content {{
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.05;
    }}
    .lede {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
      margin-bottom: 20px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 16px 0 24px;
    }}
    .stat {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 14px;
      padding: 12px;
      box-shadow: var(--shadow);
    }}
    .stat b {{
      display: block;
      font-size: 22px;
      margin-bottom: 4px;
    }}
    .filters {{
      display: grid;
      gap: 12px;
    }}
    label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      display: block;
      margin-bottom: 6px;
    }}
    input, select {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }}
    .toolbar .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    .section {{
      margin-bottom: 28px;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .run-grid, .entry-grid {{
      display: grid;
      gap: 14px;
    }}
    .run-grid {{
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .entry-grid {{
      grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
    }}
    .user-card {{
      padding: 16px;
    }}
    .user-header {{
      margin: 12px 0 14px;
    }}
    .hero img {{
      width: 180px;
      max-width: 100%;
      aspect-ratio: 0.75;
      object-fit: cover;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #f2ede4;
    }}
    .hero figcaption {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 6px;
    }}
    .variant-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .variant-tile {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: #fff;
    }}
    .variant-visuals {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin: 8px 0;
    }}
    .variant-visuals.dress-only {{
      grid-template-columns: repeat(2, 1fr);
    }}
    .variant-visuals img {{
      width: 100%;
      aspect-ratio: 0.75;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #f2ede4;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .chip {{
      background: var(--chip);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .chip.accent {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .run-card h3, .entry-card h3 {{
      margin: 0 0 8px;
      font-size: 18px;
      line-height: 1.2;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    .thumbs {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin: 12px 0;
    }}
    .thumbs.one {{
      grid-template-columns: 1fr;
    }}
    .thumbs img {{
      width: 100%;
      aspect-ratio: 0.75;
      object-fit: cover;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #f2ede4;
    }}
    .thumbs figcaption {{
      font-size: 11px;
      color: var(--muted);
      margin-top: 4px;
    }}
    figure {{
      margin: 0;
    }}
    .prompt {{
      font-size: 12px;
      line-height: 1.45;
      color: var(--muted);
      max-height: 96px;
      overflow: auto;
      border-top: 1px dashed var(--line);
      padding-top: 10px;
      margin-top: 10px;
      white-space: pre-wrap;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .empty {{
      padding: 24px;
      border: 1px dashed var(--line);
      border-radius: 18px;
      color: var(--muted);
      background: rgba(255,255,255,0.55);
    }}
    @media (max-width: 980px) {{
      .page {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <h1>Try-on Results</h1>
      <div class="lede">Unified dashboard for all generated batches under <code>__ROOT__</code>, including downloaded pod runs.</div>
      <div class="stats">
        <div class="stat"><b>__RUN_COUNT__</b><span>Runs</span></div>
        <div class="stat"><b>__ENTRY_COUNT__</b><span>Images</span></div>
      </div>
      <div class="filters">
        <div>
          <label for="q">Search</label>
          <input id="q" placeholder="user file, prompt, run..." />
        </div>
        <div>
          <label for="user-index">User Index</label>
          <input id="user-index" placeholder="e.g. 17 or 153" />
        </div>
        <div>
          <label for="run-filter">Run</label>
          <select id="run-filter"><option value="">All runs</option></select>
        </div>
        <div>
          <label for="variation-filter">Variation</label>
          <select id="variation-filter"><option value="">All variations</option></select>
        </div>
        <div>
          <label for="mode-filter">Garment Mode</label>
          <select id="mode-filter"><option value="">All modes</option></select>
        </div>
        <div>
          <label for="seed-filter">Seed</label>
          <select id="seed-filter"><option value="">All seeds</option></select>
        </div>
        <div>
          <label for="status-filter">Status</label>
          <select id="status-filter">
            <option value="">All statuses</option>
            <option value="success">success</option>
            <option value="error">error</option>
          </select>
        </div>
      </div>
    </aside>
    <main class="content">
      <div class="toolbar">
        <div class="meta" id="summary-line">Loading…</div>
      </div>
      <section class="section">
        <h2>Runs</h2>
        <div id="run-grid" class="run-grid"></div>
      </section>
      <section class="section">
        <h2>User View</h2>
        <div id="user-grid" class="entry-grid"></div>
      </section>
      <section class="section">
        <h2>Generation View</h2>
        <div id="entry-grid" class="entry-grid"></div>
      </section>
    </main>
  </div>
  <script>
    const DATA = __PAYLOAD__;
    const runs = DATA.runs || [];
    const entries = DATA.entries || [];

    const els = {{
      q: document.getElementById('q'),
      userIndex: document.getElementById('user-index'),
      runFilter: document.getElementById('run-filter'),
      variationFilter: document.getElementById('variation-filter'),
      modeFilter: document.getElementById('mode-filter'),
      seedFilter: document.getElementById('seed-filter'),
      statusFilter: document.getElementById('status-filter'),
      runGrid: document.getElementById('run-grid'),
      userGrid: document.getElementById('user-grid'),
      entryGrid: document.getElementById('entry-grid'),
      summaryLine: document.getElementById('summary-line'),
    }};

    function uniq(values) {{
      return Array.from(new Set(values.filter(Boolean))).sort((a, b) => String(a).localeCompare(String(b), undefined, {{ numeric: true }}));
    }}

    function fillSelect(select, values) {{
      values.forEach(v => {{
        const opt = document.createElement('option');
        opt.value = String(v);
        opt.textContent = String(v);
        select.appendChild(opt);
      }});
    }}

    fillSelect(els.runFilter, uniq(runs.map(r => r.run_name)));
    fillSelect(els.variationFilter, uniq(runs.map(r => r.variation)));
    fillSelect(els.modeFilter, uniq(runs.map(r => r.garment_mode)));
    fillSelect(els.seedFilter, uniq(runs.map(r => r.seed)));

    function matches(entry) {{
      const q = els.q.value.trim().toLowerCase();
      const userIndex = els.userIndex.value.trim();
      const run = els.runFilter.value;
      const variation = els.variationFilter.value;
      const mode = els.modeFilter.value;
      const seed = els.seedFilter.value;
      const status = els.statusFilter.value;

      if (q) {{
        const hay = [
          entry.run_name, entry.title, entry.user_file, entry.user_prompt,
          entry.top_prompt, entry.bottom_prompt, entry.dress_prompt, entry.variation
        ].join(' ').toLowerCase();
        if (!hay.includes(q)) return false;
      }}
      if (userIndex && String(entry.index) !== userIndex) return false;
      if (run && entry.run_name !== run) return false;
      if (variation && entry.variation !== variation) return false;
      if (mode && entry.garment_mode !== mode) return false;
      if (seed && String(entry.seed) !== seed) return false;
      if (status && entry.status !== status) return false;
      return true;
    }}

    function renderRuns(filtered) {{
      const runMap = new Map();
      filtered.forEach(entry => {{
        if (!runMap.has(entry.run_name)) {{
          runMap.set(entry.run_name, runs.find(r => r.run_name === entry.run_name));
        }}
      }});
      const cards = Array.from(runMap.values()).filter(Boolean);
      if (!cards.length) {{
        els.runGrid.innerHTML = '<div class="empty">No matching runs.</div>';
        return;
      }}
      els.runGrid.innerHTML = cards.map(run => {{
        const refs = run.garment_mode === 'dress'
          ? `<div class="thumbs one"><figure><img src="${run.dress_image_url || ''}" alt="dress ref"><figcaption>Dress reference</figcaption></figure></div>`
          : `<div class="thumbs">
              <figure><img src="${run.top_image_url || ''}" alt="top ref"><figcaption>Top reference</figcaption></figure>
              <figure><img src="${run.bottom_image_url || ''}" alt="bottom ref"><figcaption>Bottom reference</figcaption></figure>
            </div>`;
        return `
          <article class="card run-card">
            <div class="chips">
              <span class="chip accent">${run.variation}</span>
              <span class="chip">${run.garment_mode}</span>
              <span class="chip">seed ${run.seed}</span>
            </div>
            <h3>${run.title}</h3>
            <div class="muted">${run.run_name}</div>
            <div class="muted">${run.generated_at} · success ${run.success}/${run.total}</div>
            ${refs}
            <div class="prompt">${run.garment_mode === 'dress' ? (run.dress_prompt || '') : `${run.top_prompt || ''}\n\n${run.bottom_prompt || ''}`}</div>
            <div style="margin-top:10px"><a href="${run.report_view_path || run.report_path}">Open report.html</a></div>
          </article>`;
      }}).join('');
    }}

    function renderEntries(filtered) {{
      if (!filtered.length) {{
        els.entryGrid.innerHTML = '<div class="empty">No matching generations.</div>';
        return;
      }}
      els.entryGrid.innerHTML = filtered.map(entry => {{
        const garmentRefs = entry.garment_mode === 'dress'
          ? `<figure><img src="${entry.dress_image_url || ''}" alt="dress ref"><figcaption>Dress reference</figcaption></figure>`
          : `<figure><img src="${entry.top_image_url || ''}" alt="top ref"><figcaption>Top reference</figcaption></figure>
             <figure><img src="${entry.bottom_image_url || ''}" alt="bottom ref"><figcaption>Bottom reference</figcaption></figure>`;
        return `
          <article class="card entry-card">
            <div class="chips">
              <span class="chip accent">${entry.variation}</span>
              <span class="chip">#${entry.index}</span>
              <span class="chip">${entry.status}</span>
              <span class="chip">seed ${entry.seed}</span>
            </div>
            <h3>${entry.user_file || 'unknown user'}</h3>
            <div class="muted">${entry.run_name} · ${entry.generated_at}</div>
            <div class="muted">${entry.user_prompt || ''}</div>
            <div class="thumbs">
              <figure><img src="${entry.prepared_input_url || ''}" alt="input image"><figcaption>User input</figcaption></figure>
              ${garmentRefs}
              <figure><img src="${entry.upscale_output_url || entry.tryon_output_url || ''}" alt="output image"><figcaption>Output</figcaption></figure>
            </div>
            <div class="muted">tryon ${entry.tryon_latency_s.toFixed(2)}s · upscale ${entry.upscale_latency_s.toFixed(2)}s · total ${entry.total_latency_s.toFixed(2)}s</div>
            <div class="prompt">${entry.prompt_used || entry.error || ''}</div>
            <div style="margin-top:10px"><a href="${entry.report_view_path || entry.report_path}">Open parent report</a></div>
          </article>`;
      }}).join('');
    }}

    function renderUsers(filtered) {{
      const grouped = new Map();
      filtered.forEach(entry => {{
        const key = String(entry.index);
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(entry);
      }});
      const users = Array.from(grouped.entries())
        .map(([index, rows]) => {{
          rows.sort((a, b) => String(a.run_name).localeCompare(String(b.run_name)));
          return {{
            index,
            rows,
            first: rows[0],
          }};
        }})
        .sort((a, b) => Number(a.index) - Number(b.index));

      if (!users.length) {{
        els.userGrid.innerHTML = '<div class="empty">No matching users.</div>';
        return;
      }}

      els.userGrid.innerHTML = users.map(user => {{
        const tiles = user.rows.map(entry => {{
          const garmentRef = entry.garment_mode === 'dress'
            ? `<img src="${entry.dress_image_url || ''}" alt="dress ref">`
            : `<img src="${entry.top_image_url || ''}" alt="top ref"><img src="${entry.bottom_image_url || ''}" alt="bottom ref">`;
          return `
            <div class="variant-tile">
              <div class="chips">
                <span class="chip accent">${entry.variation}</span>
                <span class="chip">${entry.run_name}</span>
                <span class="chip">seed ${entry.seed}</span>
              </div>
              <div class="variant-visuals ${entry.garment_mode === 'dress' ? 'dress-only' : ''}">
                ${garmentRef}
                <img src="${entry.upscale_output_url || entry.tryon_output_url || ''}" alt="output image">
              </div>
              <div class="muted">${entry.status} · ${entry.total_latency_s.toFixed(2)}s total</div>
            </div>`;
        }}).join('');

        return `
          <article class="card user-card">
            <div class="chips">
              <span class="chip accent">user #${user.index}</span>
              <span class="chip">${user.rows.length} generations</span>
              <span class="chip">${user.first.user_file || 'unknown file'}</span>
            </div>
            <h3>${user.first.user_file || 'unknown user'}</h3>
            <div class="muted">${user.first.user_prompt || ''}</div>
            <div class="user-header">
              <figure class="hero">
                <img src="${user.first.prepared_input_url || ''}" alt="user input image">
                <figcaption>User input</figcaption>
              </figure>
            </div>
            <div class="variant-grid">${tiles}</div>
          </article>`;
      }}).join('');
    }}

    function render() {{
      const filtered = entries.filter(matches);
      const userCount = new Set(filtered.map(x => x.index)).size;
      els.summaryLine.textContent = `${filtered.length} generations across ${userCount} users from ${new Set(filtered.map(x => x.run_name)).size} runs`;
      renderRuns(filtered);
      renderUsers(filtered);
      renderEntries(filtered);
    }}

    [els.q, els.userIndex, els.runFilter, els.variationFilter, els.modeFilter, els.seedFilter, els.statusFilter]
      .forEach(el => el.addEventListener('input', render));
    [els.runFilter, els.variationFilter, els.modeFilter, els.seedFilter, els.statusFilter]
      .forEach(el => el.addEventListener('change', render));

    render();
  </script>
</body>
</html>
"""
    html_doc = (
        html_doc.replace("{{", "{")
        .replace("}}", "}")
        .replace("__ROOT__", html.escape(str(dataset.get("root") or "")))
        .replace("__RUN_COUNT__", str(run_count))
        .replace("__ENTRY_COUNT__", str(entry_count))
        .replace("__PAYLOAD__", payload)
    )
    return HTMLResponse(content=html_doc)
