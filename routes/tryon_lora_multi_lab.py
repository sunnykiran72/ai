"""
Developer-facing FLUX2 Tryon LoRA multi-garment lab UI.

This module adds a separate, isolated test page for /v1/flux2/tryon-style
experimentation with 4 fixed garment sections:
- top
- bottom
- outer
- dress
"""

from __future__ import annotations

import csv
import html
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from services import TryonService
from shared.azure_storage import storage

logger = logging.getLogger("glamify-ai")
router = APIRouter()

_VALID_GARMENT_TYPES = {"top", "bottom", "outer", "dress"}
_SECTION_ORDER: Tuple[str, ...] = ("top", "bottom", "outer", "dress")
_DEFAULT_SEED44_CSV = Path("debug_outputs/bulk_tryon_seed44_steps12_cache220656_20260407/results.csv")
_DEFAULT_SEED77_CSV = Path("debug_outputs/bulk_tryon_seed77_steps12_cache220656_20260407/results.csv")


def get_tryon_service() -> TryonService:
    """Dependency to get TryonService instance."""
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_tryon_service()


def _load_uploaded_image(file: UploadFile, *, field_name: str) -> Image.Image:
    try:
        raw = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read {field_name}: {exc}") from exc
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} upload is empty")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not a valid image: {exc}") from exc


def _normalize_prompt(value: str, fallback: str) -> str:
    cleaned = " ".join(str(value or "").split()).strip()
    return cleaned if cleaned else fallback


def _parse_form_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_source_worn_types(raw: str) -> List[str]:
    values: List[str] = []
    for part in str(raw or "").split(","):
        kind = part.strip().lower()
        if not kind:
            continue
        if kind not in _VALID_GARMENT_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid source worn type '{kind}'. Allowed: top, bottom, outer, dress.",
            )
        if kind not in values:
            values.append(kind)
    return values


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


@router.get("/dev/flux2/seed-compare-view", response_class=HTMLResponse)
async def tryon_seed_compare_view(
    per_page: int = 10,
) -> HTMLResponse:
    """
    Client-shareable seed comparison page (seed 44 vs seed 77).
    Shows only file name + images, no prompt text.
    """
    per_page = max(5, min(int(per_page or 10), 30))
    csv_44 = _DEFAULT_SEED44_CSV.expanduser().resolve()
    csv_77 = _DEFAULT_SEED77_CSV.expanduser().resolve()

    rows_44 = _read_csv_rows(csv_44)
    rows_77 = _read_csv_rows(csv_77)
    by_77 = {str(row.get("user_file") or "").strip(): row for row in rows_77}

    # Keep row order anchored to seed 44 run progression.
    keys = [str(row.get("user_file") or "").strip() for row in rows_44 if str(row.get("user_file") or "").strip()]
    total_cases = len(keys)
    total_pages = max(1, (total_cases + per_page - 1) // per_page)

    cards: List[str] = []
    matched_success = 0

    def _img(url: str, label: str) -> str:
        cleaned = str(url or "").strip()
        if not cleaned:
            return "<div class='missing'>N/A</div>"
        escaped = html.escape(cleaned)
        return (
            f"<img src='{escaped}' loading='lazy' alt='img'/>"
            f"<div><a href='{escaped}' target='_blank' rel='noopener'>open {label}</a></div>"
        )

    def _status_class(status: str) -> str:
        if status == "success":
            return "ok"
        if status == "error":
            return "err"
        return "na"

    for index, key in enumerate(keys, start=1):
        row_44 = rows_44[index - 1]
        row_77 = by_77.get(key, {})
        user_input = str(row_44.get("user_input_url") or row_77.get("user_input_url") or "").strip()
        garment = str(row_44.get("garment_url") or row_77.get("garment_url") or "").strip()
        out_44 = str(row_44.get("output_url") or "").strip()
        out_77 = str(row_77.get("output_url") or "").strip()
        status_44 = str(row_44.get("status") or "missing").strip() or "missing"
        status_77 = str(row_77.get("status") or "missing").strip() or "missing"
        error_44 = str(row_44.get("error") or "").strip()
        error_77 = str(row_77.get("error") or "").strip()
        page = ((index - 1) // per_page) + 1

        if status_44 == "success" and status_77 == "success" and out_44 and out_77:
            matched_success += 1

        cards.append(
            f"""
<section class='card' data-page='{page}' data-index='{index}'>
  <div class='head'>
    <div class='case-tag'>Case {index:03d}</div>
    <div class='chips'>
      <span class='chip {_status_class(status_44)}'>Seed44: {html.escape(status_44)}</span>
      <span class='chip {_status_class(status_77)}'>Seed77: {html.escape(status_77)}</span>
    </div>
  </div>
  <div class='file-name'>{html.escape(key)}</div>
  <div class='grid'>
    <div class='cell'><h4>Input Person</h4>{_img(user_input, 'input')}</div>
    <div class='cell'><h4>Try-on (seed44)</h4>{_img(out_44, 'seed44 output')}</div>
    <div class='cell'><h4>Try-on (seed77)</h4>{_img(out_77, 'seed77 output')}</div>
    <div class='cell'><h4>Garment</h4>{_img(garment, 'garment')}</div>
  </div>
  <div class='errs'>
    <div><b>seed44 error:</b> {html.escape(error_44)}</div>
    <div><b>seed77 error:</b> {html.escape(error_77)}</div>
  </div>
</section>
"""
        )

    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Try-on Results</title>
  <style>
    :root {{
      --bg: #f4f6fb;
      --ink: #102135;
      --muted: #5d6e83;
      --line: #d5dfeb;
      --card: #ffffff;
      --accent: #0f766e;
      --accent-2: #164e63;
      --ok-bg: #e8f7ef;
      --ok-ink: #166534;
      --err-bg: #fdebec;
      --err-ink: #9f1239;
      --na-bg: #eef2f7;
      --na-ink: #4b5563;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 16px;
      font-family: "Space Grotesk", "Manrope", "Avenir Next", Arial, sans-serif;
      background:
        radial-gradient(960px 520px at 0% -20%, #dfe9ff 0%, transparent 65%),
        radial-gradient(920px 500px at 100% -20%, #d9f4ee 0%, transparent 62%),
        linear-gradient(180deg, #f8fbff, var(--bg));
      color: var(--ink);
    }}
    .top {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 12px 30px rgba(15, 25, 45, 0.08);
      padding: 12px;
      margin-bottom: 14px;
    }}
    .title {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0.2px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-top: 6px;
    }}
    .meta b {{ color: var(--ink); }}
    .pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: #f8fbff;
      font-size: 12px;
      color: var(--ink);
    }}
    .controls {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }}
    .pager-btn {{
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      padding: 7px 11px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
    }}
    .pager-btn:hover {{
      border-color: #b8c6d8;
      background: #f9fbff;
    }}
    .pager-btn:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    .page-btn {{
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      min-width: 34px;
      height: 34px;
      padding: 0 8px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }}
    .page-btn.active {{
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
    }}
    .summary-line {{
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }}
    .loading-strip {{
      height: 3px;
      border-radius: 999px;
      margin-top: 8px;
      opacity: 0;
      background: linear-gradient(90deg, #dbeafe, #14b8a6, #dbeafe);
      background-size: 220% 100%;
      transition: opacity 120ms ease-in-out;
      pointer-events: none;
    }}
    .loading-strip.show {{
      opacity: 1;
      animation: shimmer 1.05s linear infinite;
    }}
    #cards {{
      position: relative;
    }}
    #cards.is-loading .card {{
      opacity: 0.55;
      transform: scale(0.998);
      transition: opacity 120ms ease-in-out, transform 120ms ease-in-out;
    }}
    @keyframes shimmer {{
      0% {{ background-position: 200% 0; }}
      100% {{ background-position: -20% 0; }}
    }}
    .cards {{
      display: grid;
      gap: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
      padding: 10px;
      box-shadow: 0 6px 18px rgba(12, 24, 42, 0.05);
    }}
    .head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .case-tag {{
      font-size: 11px;
      font-weight: 700;
      color: #0f3d5f;
      background: #e7f2fb;
      border: 1px solid #c8ddf0;
      border-radius: 999px;
      padding: 4px 8px;
      letter-spacing: 0.35px;
      text-transform: uppercase;
    }}
    .chips {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .chip {{
      font-size: 11px;
      font-weight: 700;
      border-radius: 999px;
      padding: 4px 8px;
      border: 1px solid transparent;
      letter-spacing: 0.2px;
    }}
    .chip.ok {{
      background: var(--ok-bg);
      color: var(--ok-ink);
      border-color: #b8e4c7;
    }}
    .chip.err {{
      background: var(--err-bg);
      color: var(--err-ink);
      border-color: #f5c7cc;
    }}
    .chip.na {{
      background: var(--na-bg);
      color: var(--na-ink);
      border-color: #d7dee7;
    }}
    .file-name {{
      font-size: 14px;
      font-weight: 600;
      color: var(--ink);
      margin-bottom: 8px;
      word-break: break-word;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(220px, 1fr));
      gap: 10px;
    }}
    .cell {{
      background: #ffffff;
      border: 1px solid #dbe3ee;
      border-radius: 8px;
      padding: 8px;
    }}
    .cell h4 {{
      margin: 0 0 6px;
      font-size: 13px;
    }}
    img {{
      width: 100%;
      border-radius: 6px;
    }}
    a {{
      color: var(--accent-2);
      font-size: 12px;
      text-decoration: none;
      font-weight: 600;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .missing {{
      text-align: center;
      color: #94a3b8;
      padding: 40px 0;
    }}
    .errs {{
      margin-top: 8px;
      font-size: 12px;
      color: #64748b;
      display: grid;
      gap: 4px;
    }}
    @media (max-width: 1200px) {{
      .grid {{
        grid-template-columns: repeat(2, minmax(220px, 1fr));
      }}
    }}
    @media (max-width: 760px) {{
      body {{
        padding: 10px;
      }}
      .title {{
        font-size: 20px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        gap: 6px;
      }}
    }}
  </style>
</head>
<body>
  <div class="top">
    <h2 class="title">Try-on Results</h2>
    <div class="meta">Client view comparison with seed44 and seed77 outputs.</div>
    <div class="pills">
      <span class="pill">Total Cases: {total_cases}</span>
      <span class="pill">Matched Success Pairs: {matched_success}</span>
      <span class="pill">Per Page: {per_page}</span>
    </div>
    <div class="controls">
      <button class="pager-btn" id="firstBtn" onclick="goFirst()">First</button>
      <button class="pager-btn" id="prevBtn" onclick="prevPage()">Prev</button>
      <div id="pageButtons"></div>
      <button class="pager-btn" id="nextBtn" onclick="nextPage()">Next</button>
      <button class="pager-btn" id="lastBtn" onclick="goLast()">Last</button>
      <span id="pageLabel"></span>
    </div>
    <div class="summary-line" id="rangeLabel"></div>
    <div class="loading-strip" id="loadingStrip"></div>
  </div>
  <div id="cards" class="cards">
    {''.join(cards)}
  </div>
  <script>
    let page = 1;
    const perPage = {per_page};
    const totalPages = {total_pages};
    const totalCards = {total_cases};
    const pageWindow = 7;
    let loadingTimer = null;

    function renderPageButtons() {{
      const wrap = document.getElementById('pageButtons');
      wrap.innerHTML = '';
      const half = Math.floor(pageWindow / 2);
      let start = Math.max(1, page - half);
      let end = Math.min(totalPages, start + pageWindow - 1);
      if ((end - start + 1) < pageWindow) {{
        start = Math.max(1, end - pageWindow + 1);
      }}
      for (let p = start; p <= end; p += 1) {{
        const btn = document.createElement('button');
        btn.className = 'page-btn' + (p === page ? ' active' : '');
        btn.textContent = String(p);
        btn.onclick = () => {{
          switchPage(p);
        }};
        wrap.appendChild(btn);
      }}
    }}

    function renderSummary() {{
      const from = totalCards === 0 ? 0 : ((page - 1) * perPage + 1);
      const to = totalCards === 0 ? 0 : Math.min(page * perPage, totalCards);
      document.getElementById('pageLabel').textContent = `Page ${{page}} / ${{totalPages}}`;
      document.getElementById('rangeLabel').textContent = `Showing ${{from}}-${{to}} of ${{totalCards}} cases`;
      document.getElementById('firstBtn').disabled = page <= 1;
      document.getElementById('prevBtn').disabled = page <= 1;
      document.getElementById('nextBtn').disabled = page >= totalPages;
      document.getElementById('lastBtn').disabled = page >= totalPages;
    }}

    function renderCards() {{
      document.querySelectorAll('.card').forEach((card) => {{
        card.style.display = (Number(card.dataset.page) === page) ? 'block' : 'none';
      }});
    }}

    function render() {{
      renderCards();
      renderPageButtons();
      renderSummary();
    }}

    function setLoading(loading) {{
      const strip = document.getElementById('loadingStrip');
      const cards = document.getElementById('cards');
      if (loading) {{
        strip.classList.add('show');
        cards.classList.add('is-loading');
      }} else {{
        strip.classList.remove('show');
        cards.classList.remove('is-loading');
      }}
    }}

    function scrollToPageTop() {{
      try {{
        window.scrollTo({{ top: 0, left: 0, behavior: 'smooth' }});
      }} catch (err) {{
        // Fallback for older browsers
        window.scrollTo(0, 0);
      }}
      document.documentElement.scrollTop = 0;
      if (document.body) {{
        document.body.scrollTop = 0;
      }}
    }}

    function switchPage(nextPage) {{
      const clamped = Math.max(1, Math.min(totalPages, Number(nextPage)));
      if (clamped === page) {{
        scrollToPageTop();
        return;
      }}
      page = clamped;
      scrollToPageTop();
      if (loadingTimer) {{
        clearTimeout(loadingTimer);
      }}
      setLoading(true);
      loadingTimer = setTimeout(() => {{
        render();
        setLoading(false);
      }}, 160);
    }}

    function prevPage() {{
      if (page > 1) {{
        switchPage(page - 1);
      }}
    }}
    function nextPage() {{
      if (page < totalPages) {{
        switchPage(page + 1);
      }}
    }}
    function goFirst() {{
      if (page !== 1) {{
        switchPage(1);
      }}
    }}
    function goLast() {{
      if (page !== totalPages) {{
        switchPage(totalPages);
      }}
    }}
    render();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=page_html)


@router.get("/dev/flux2/tryon-lora-multi-lab", response_class=HTMLResponse)
async def tryon_lora_multi_lab_page() -> HTMLResponse:
    """Serve standalone multi-garment Tryon LoRA lab page."""
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FLUX2 Tryon LoRA Multi Lab</title>
  <style>
    :root {
      --bg0: #f6f8fc;
      --bg1: #eaf0f8;
      --ink: #0f2435;
      --muted: #556879;
      --card: #ffffff;
      --line: #d0d9e5;
      --accent: #0f766e;
      --accent2: #164e63;
      --danger: #9f1239;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Space Grotesk", "Manrope", "Avenir Next", sans-serif;
      background:
        radial-gradient(1000px 520px at 8% -18%, #d5e6ff 0%, transparent 65%),
        radial-gradient(860px 440px at 95% -20%, #ccf3eb 0%, transparent 62%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
      min-height: 100vh;
    }
    .wrap { max-width: 1480px; margin: 18px auto; padding: 0 12px 24px; }
    .title { margin: 6px 0 14px; font-size: 28px; }
    .layout {
      display: grid;
      grid-template-columns: 520px 1fr;
      gap: 12px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(15, 36, 53, 0.06);
      padding: 12px;
    }
    label {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.35px;
      text-transform: uppercase;
      font-weight: 700;
    }
    input[type="text"], input[type="number"], input[type="file"], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
      padding: 8px 10px;
    }
    textarea { min-height: 92px; resize: vertical; line-height: 1.35; }
    .row2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }
    .row4 {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 10px;
    }
    .section-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }
    .section-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f9fbff;
      padding: 10px;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
      gap: 8px;
    }
    .section-title {
      margin: 0;
      font-size: 14px;
      letter-spacing: 0.2px;
      text-transform: uppercase;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      text-transform: none;
      letter-spacing: 0;
    }
    .actions { display: flex; gap: 8px; align-items: center; margin-top: 6px; }
    .btn {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
    }
    .btn.secondary { background: #334155; }
    .btn:disabled { opacity: 0.65; cursor: wait; }
    .status { min-height: 20px; font-size: 13px; color: var(--muted); margin-top: 8px; }
    .status.err { color: var(--danger); font-weight: 700; }
    .preview-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }
    .tile {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #f7f9fc;
    }
    .tile h4 {
      margin: 0 0 6px;
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.4px;
    }
    .tile img {
      width: 100%;
      height: 360px;
      object-fit: contain;
      border: 1px solid #e5ebf3;
      border-radius: 8px;
      background: #fff;
      display: block;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }
    .mini {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 6px;
      background: #f7f9fc;
    }
    .mini b {
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .mini img {
      width: 100%;
      height: 170px;
      object-fit: contain;
      border: 1px solid #e5ebf3;
      border-radius: 8px;
      background: #fff;
      display: block;
    }
    .prompt-box {
      background: #f9fafb;
      border: 1px dashed #c8d2df;
      border-radius: 10px;
      padding: 10px;
      font-size: 13px;
      line-height: 1.35;
      min-height: 64px;
      margin-bottom: 10px;
      white-space: pre-wrap;
    }
    pre {
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 10px;
      padding: 10px;
      font-size: 12px;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    @media (max-width: 1280px) {
      .layout { grid-template-columns: 1fr; }
      .row4 { grid-template-columns: 1fr 1fr; }
      .section-grid { grid-template-columns: 1fr; }
      .tile img { height: 300px; }
      .mini-grid { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">FLUX2 Tryon LoRA Multi Lab (Separate Dev Page)</h1>
    <div class="layout">
      <section class="card">
        <form id="lab-form">
          <div class="row2">
            <div>
              <label>User Image</label>
              <input id="user_image" name="user_image" type="file" accept="image/*" required />
            </div>
            <div>
              <label>Source Worn Types (comma-separated)</label>
              <input id="source_worn_types" name="source_worn_types" type="text" value="top,bottom" />
            </div>
          </div>

          <div class="row4">
            <div>
              <label>Steps</label>
              <input id="steps" name="steps" type="number" min="4" max="50" step="1" value="10" />
            </div>
            <div>
              <label>Seed</label>
              <input id="seed" name="seed" type="number" min="0" max="2147483647" step="1" value="42" />
            </div>
            <div>
              <label>Guidance Scale</label>
              <input id="guidance_scale" name="guidance_scale" type="number" min="0" max="20" step="0.1" value="2.5" />
            </div>
            <div>
              <label>LoRA Scale (Tryon)</label>
              <input id="lora_scale" name="lora_scale" type="number" min="0" max="2" step="0.01" value="1.0" />
            </div>
          </div>

          <div class="row2">
            <div>
              <label>Output Max Edge (px)</label>
              <input id="output_max_edge" name="output_max_edge" type="number" min="512" max="2048" step="8" value="1024" />
            </div>
            <div></div>
          </div>

          <div class="row2">
            <div>
              <label>User Prompt Description (optional)</label>
              <input id="user_prompt_description" name="user_prompt_description" type="text" value="" />
            </div>
            <div>
              <label>Prompt Override (optional)</label>
              <input id="prompt_override" name="prompt_override" type="text" value="" />
            </div>
          </div>

          <div class="section-grid">
            <div class="section-card">
              <div class="section-head">
                <h3 class="section-title">Top</h3>
                <label class="toggle"><input id="top_enabled" type="checkbox" checked /> Include</label>
              </div>
              <label>Top Image</label>
              <input id="top_image" type="file" accept="image/*" />
              <label style="margin-top:8px;">Top Prompt</label>
              <input id="top_prompt" type="text" value="top garment from image" />
            </div>

            <div class="section-card">
              <div class="section-head">
                <h3 class="section-title">Bottom</h3>
                <label class="toggle"><input id="bottom_enabled" type="checkbox" /> Include</label>
              </div>
              <label>Bottom Image</label>
              <input id="bottom_image" type="file" accept="image/*" />
              <label style="margin-top:8px;">Bottom Prompt</label>
              <input id="bottom_prompt" type="text" value="bottom garment from image" />
            </div>

            <div class="section-card">
              <div class="section-head">
                <h3 class="section-title">Outer</h3>
                <label class="toggle"><input id="outer_enabled" type="checkbox" /> Include</label>
              </div>
              <label>Outer Image</label>
              <input id="outer_image" type="file" accept="image/*" />
              <label style="margin-top:8px;">Outer Prompt</label>
              <input id="outer_prompt" type="text" value="outer garment from image" />
            </div>

            <div class="section-card">
              <div class="section-head">
                <h3 class="section-title">Dress</h3>
                <label class="toggle"><input id="dress_enabled" type="checkbox" /> Include</label>
              </div>
              <label>Dress Image</label>
              <input id="dress_image" type="file" accept="image/*" />
              <label style="margin-top:8px;">Dress Prompt</label>
              <input id="dress_prompt" type="text" value="dress garment from image" />
            </div>
          </div>

          <div class="actions">
            <button id="run-btn" class="btn" type="submit">Run Try-on</button>
            <button id="reset-btn" class="btn secondary" type="button">Reset Defaults</button>
          </div>
          <div id="status" class="status">Ready.</div>
        </form>
      </section>

      <section class="card">
        <div class="preview-grid">
          <div class="tile">
            <h4>User</h4>
            <img id="img-user" alt="user preview" />
          </div>
          <div class="tile">
            <h4>Try-on Output</h4>
            <img id="img-output" alt="output preview" />
            <div style="margin-top:6px; font-size:12px; color:#556879; word-break:break-all;">
              <a id="output-url" href="" target="_blank" rel="noopener noreferrer"></a>
            </div>
          </div>
          <div class="tile">
            <h4>Primary Garment Preview</h4>
            <img id="img-primary-garment" alt="primary garment preview" />
          </div>
        </div>

        <div class="mini-grid">
          <div class="mini"><b>Top</b><img id="img-top" alt="top preview" /></div>
          <div class="mini"><b>Bottom</b><img id="img-bottom" alt="bottom preview" /></div>
          <div class="mini"><b>Outer</b><img id="img-outer" alt="outer preview" /></div>
          <div class="mini"><b>Dress</b><img id="img-dress" alt="dress preview" /></div>
        </div>

        <label>Prompt Used By FLUX</label>
        <div id="prompt-used" class="prompt-box"></div>

        <label>All Metrics / Metadata</label>
        <pre id="raw-json"></pre>
      </section>
    </div>
  </div>

  <script>
    const storageKey = "tryon_lora_multi_lab_v1";
    const form = document.getElementById("lab-form");
    const statusEl = document.getElementById("status");
    const rawEl = document.getElementById("raw-json");
    const promptUsedEl = document.getElementById("prompt-used");
    const runBtn = document.getElementById("run-btn");
    const resetBtn = document.getElementById("reset-btn");

    const userImageInput = document.getElementById("user_image");
    const imgUser = document.getElementById("img-user");
    const imgOutput = document.getElementById("img-output");
    const imgPrimaryGarment = document.getElementById("img-primary-garment");
    const outputUrlEl = document.getElementById("output-url");

    const sections = ["top", "bottom", "outer", "dress"];
    const defaults = {
      steps: 10,
      seed: 42,
      guidance_scale: 2.5,
      lora_scale: 1.0,
      output_max_edge: 1024,
      source_worn_types: "top,bottom",
      user_prompt_description: "",
      prompt_override: "",
      top_enabled: true,
      bottom_enabled: false,
      outer_enabled: false,
      dress_enabled: false,
      top_prompt: "top garment from image",
      bottom_prompt: "bottom garment from image",
      outer_prompt: "outer garment from image",
      dress_prompt: "dress garment from image"
    };

    function setStatus(message, isError=false) {
      statusEl.textContent = message;
      statusEl.className = isError ? "status err" : "status";
    }

    function previewFile(inputEl, imgEl) {
      if (!inputEl.files || !inputEl.files[0]) {
        imgEl.removeAttribute("src");
        return;
      }
      imgEl.src = URL.createObjectURL(inputEl.files[0]);
    }

    function getFieldValue(id) {
      const node = document.getElementById(id);
      if (!node) return "";
      if (node.type === "checkbox") return node.checked;
      return node.value;
    }

    function setFieldValue(id, value) {
      const node = document.getElementById(id);
      if (!node) return;
      if (node.type === "checkbox") {
        node.checked = !!value;
      } else {
        node.value = value ?? "";
      }
    }

    function saveState() {
      const data = {};
      for (const key of Object.keys(defaults)) data[key] = getFieldValue(key);
      localStorage.setItem(storageKey, JSON.stringify(data));
    }

    function loadState() {
      try {
        const raw = localStorage.getItem(storageKey);
        if (!raw) return;
        const data = JSON.parse(raw);
        for (const key of Object.keys(defaults)) {
          if (key in data) setFieldValue(key, data[key]);
        }
      } catch (_err) {}
    }

    function applyDefaults() {
      for (const [key, value] of Object.entries(defaults)) setFieldValue(key, value);
      saveState();
    }

    userImageInput.addEventListener("change", () => previewFile(userImageInput, imgUser));
    for (const section of sections) {
      const input = document.getElementById(`${section}_image`);
      const img = document.getElementById(`img-${section}`);
      input.addEventListener("change", () => {
        previewFile(input, img);
        const first = sections.find((name) => document.getElementById(`${name}_image`).files?.length);
        if (first) {
          const firstInput = document.getElementById(`${first}_image`);
          previewFile(firstInput, imgPrimaryGarment);
        }
      });
      const promptInput = document.getElementById(`${section}_prompt`);
      const enabledInput = document.getElementById(`${section}_enabled`);
      promptInput.addEventListener("change", saveState);
      enabledInput.addEventListener("change", saveState);
    }

    for (const key of ["steps", "seed", "guidance_scale", "lora_scale", "source_worn_types", "user_prompt_description", "prompt_override"]) {
      document.getElementById(key).addEventListener("change", saveState);
    }

    document.getElementById("output_max_edge").addEventListener("change", saveState);

    resetBtn.addEventListener("click", () => {
      applyDefaults();
      setStatus("Defaults restored.");
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!userImageInput.files?.length) {
        setStatus("Please upload a user image.", true);
        return;
      }

      const formData = new FormData();
      formData.append("user_image", userImageInput.files[0]);
      formData.append("source_worn_types", getFieldValue("source_worn_types"));
      formData.append("steps", getFieldValue("steps"));
      formData.append("seed", getFieldValue("seed"));
      formData.append("guidance_scale", getFieldValue("guidance_scale"));
      formData.append("lora_scale", getFieldValue("lora_scale"));
      formData.append("output_max_edge", getFieldValue("output_max_edge"));
      formData.append("user_prompt_description", getFieldValue("user_prompt_description"));
      formData.append("prompt_override", getFieldValue("prompt_override"));

      let selected = 0;
      let firstSelectedFile = null;
      for (const section of sections) {
        const enabled = !!getFieldValue(`${section}_enabled`);
        const imageInput = document.getElementById(`${section}_image`);
        const promptInput = document.getElementById(`${section}_prompt`);
        formData.append(`${section}_enabled`, enabled ? "true" : "false");
        formData.append(`${section}_prompt`, promptInput.value || "");
        if (enabled) {
          if (!imageInput.files?.length) {
            setStatus(`Section '${section}' is enabled but no image uploaded.`, true);
            return;
          }
          selected += 1;
          if (!firstSelectedFile) firstSelectedFile = imageInput.files[0];
          formData.append(`${section}_image`, imageInput.files[0]);
        }
      }

      if (selected === 0) {
        setStatus("Enable at least one garment section and upload image(s).", true);
        return;
      }

      if (firstSelectedFile) {
        imgPrimaryGarment.src = URL.createObjectURL(firstSelectedFile);
      }

      saveState();
      runBtn.disabled = true;
      setStatus("Running Tryon LoRA...");
      rawEl.textContent = "";
      promptUsedEl.textContent = "";
      imgOutput.removeAttribute("src");
      outputUrlEl.removeAttribute("href");
      outputUrlEl.textContent = "";

      try {
        const response = await fetch("/dev/flux2/tryon-lora-multi-lab/run", {
          method: "POST",
          body: formData
        });
        const payload = await response.json();
        if (!response.ok) {
          const detail = payload?.detail || payload?.message || `HTTP ${response.status}`;
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }
        const outputUrl = payload?.output_url || payload?.metadata?.output_url || payload?.data?.output_url || "";
        if (outputUrl) {
          const urlWithCacheBust = `${outputUrl}${outputUrl.includes("?") ? "&" : "?"}t=${Date.now()}`;
          imgOutput.onerror = () => {
            setStatus(`Success, but output preview failed to load. Open URL link below.`, true);
          };
          imgOutput.src = urlWithCacheBust;
          outputUrlEl.href = outputUrl;
          outputUrlEl.textContent = outputUrl;
        }
        promptUsedEl.textContent = payload?.metadata?.prompt || "(empty)";
        rawEl.textContent = JSON.stringify(payload, null, 2);
        setStatus(`Success. Latency: ${payload.latency_seconds ?? "n/a"} sec`);
      } catch (err) {
        setStatus(`Run failed: ${err.message}`, true);
      } finally {
        runBtn.disabled = false;
      }
    });

    loadState();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@router.post("/dev/flux2/tryon-lora-multi-lab/run")
async def tryon_lora_multi_lab_run(
    user_image: UploadFile = File(...),
    source_worn_types: str = Form("top,bottom"),
    user_prompt_description: str = Form(""),
    prompt_override: str = Form(""),
    steps: int = Form(10, ge=4, le=50),
    seed: int = Form(42, ge=0, le=2147483647),
    guidance_scale: float = Form(2.5, ge=0.0, le=20.0),
    lora_scale: float = Form(1.0, ge=0.0, le=2.0),
    output_max_edge: int = Form(1024, ge=512, le=2048),
    top_enabled: str = Form("true"),
    top_prompt: str = Form(""),
    top_image: Optional[UploadFile] = File(None),
    bottom_enabled: str = Form("false"),
    bottom_prompt: str = Form(""),
    bottom_image: Optional[UploadFile] = File(None),
    outer_enabled: str = Form("false"),
    outer_prompt: str = Form(""),
    outer_image: Optional[UploadFile] = File(None),
    dress_enabled: str = Form("false"),
    dress_prompt: str = Form(""),
    dress_image: Optional[UploadFile] = File(None),
    tryon_service: TryonService = Depends(get_tryon_service),
):
    """
    Run multi-garment Tryon LoRA test with per-section upload + prompts.
    """
    source_types = _parse_source_worn_types(source_worn_types)
    person_image = _load_uploaded_image(user_image, field_name="user_image")

    section_inputs: Dict[str, Dict[str, object]] = {
        "top": {"enabled": _parse_form_bool(top_enabled, True), "prompt": top_prompt, "file": top_image},
        "bottom": {"enabled": _parse_form_bool(bottom_enabled, False), "prompt": bottom_prompt, "file": bottom_image},
        "outer": {"enabled": _parse_form_bool(outer_enabled, False), "prompt": outer_prompt, "file": outer_image},
        "dress": {"enabled": _parse_form_bool(dress_enabled, False), "prompt": dress_prompt, "file": dress_image},
    }

    garment_images: List[Image.Image] = []
    garment_descriptions: List[str] = []
    target_types: List[str] = []
    selected_sections: List[str] = []

    for section in _SECTION_ORDER:
        section_data = section_inputs.get(section) or {}
        enabled = bool(section_data.get("enabled"))
        upload = section_data.get("file")
        prompt_value = _normalize_prompt(str(section_data.get("prompt") or ""), f"{section} garment from image")
        if not enabled:
            continue
        if upload is None or not str(getattr(upload, "filename", "") or "").strip():
            raise HTTPException(status_code=422, detail=f"Section '{section}' is enabled but image is missing.")
        image = _load_uploaded_image(upload, field_name=f"{section}_image")
        garment_images.append(image)
        garment_descriptions.append(prompt_value)
        target_types.append(section)
        selected_sections.append(section)

    if not garment_images:
        raise HTTPException(status_code=422, detail="Enable at least one garment section with image upload.")

    board_image, board_mode = tryon_service._build_board(garment_images)
    normalized_override = " ".join(str(prompt_override or "").split()).strip()
    prompt_text = normalized_override or tryon_service._build_tryon_lora_prompt(
        user_description=str(user_prompt_description or "").strip(),
        garment_descriptions=garment_descriptions,
        target_types=target_types,
        board_mode=board_mode,
        source_worn_types=source_types,
    )

    flux_runner = getattr(tryon_service.engine, "flux2", None)
    if flux_runner is None:
        raise HTTPException(status_code=500, detail="Tryon runner is unavailable.")

    try:
        run_result = flux_runner.run_tryon(
            person_image=person_image,
            board_image=board_image,
            prompt=prompt_text,
            steps=steps,
            seed=seed,
            guidance_scale=guidance_scale,
            use_lora=True,
            lora_mode="tryon",
            lora_scale=float(lora_scale),
            output_max_edge=int(output_max_edge),
        )
    except Exception as exc:
        logger.exception("Tryon LoRA multi lab run failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    image = run_result.get("image")
    if not isinstance(image, Image.Image):
        raise HTTPException(status_code=500, detail="Runner did not return a valid image.")

    out = io.BytesIO()
    image.save(out, format="PNG")
    png_bytes = out.getvalue()
    try:
        output_url = storage.upload_image(png_bytes, content_type="image/png")
    except Exception as exc:
        logger.exception("Tryon LoRA multi lab Azure upload failed")
        raise HTTPException(status_code=500, detail=f"Azure upload failed: {exc}") from exc

    metadata = dict(run_result.get("metadata") or {})
    metadata["prompt"] = prompt_text
    metadata["lab_mode"] = "tryon-lora-multi"
    metadata["lab_board_mode"] = board_mode
    metadata["lab_selected_sections"] = selected_sections
    metadata["lab_target_types"] = target_types
    metadata["lab_source_worn_types"] = source_types
    metadata["lora_scale"] = float(lora_scale)
    metadata["effective_lora_scale"] = float(metadata.get("effective_lora_scale") or lora_scale)
    metadata["output_max_edge"] = int(output_max_edge)
    metadata["output_url"] = output_url

    return {
        "status": "success",
        "message": "Tryon LoRA multi lab run completed",
        "output_url": output_url,
        "latency_seconds": float(run_result.get("latency") or 0.0),
        "metadata": metadata,
        "request": {
            "steps": steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "lora_scale": float(lora_scale),
            "output_max_edge": int(output_max_edge),
            "source_worn_types": source_types,
            "user_prompt_description": str(user_prompt_description or "").strip(),
            "prompt_override": normalized_override,
            "board_mode": board_mode,
            "selected_sections": selected_sections,
            "products": [
                {"targetType": t, "promptDescription": d}
                for t, d in zip(target_types, garment_descriptions)
            ],
        },
    }
