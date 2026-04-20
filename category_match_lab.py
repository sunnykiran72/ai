"""
Standalone local lab for closed-set fashion category matching.

This app is intentionally isolated from the existing API surface. It loads the
category lookup CSV, scores a provided image against the subcategories for a
selected top-level group, and returns the best matching category key/label.

Run locally:
    uvicorn category_match_lab:app --host 127.0.0.1 --port 8011 --reload
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image


logger = logging.getLogger("glamify-category-lab")
logging.basicConfig(level=logging.INFO)


DEFAULT_LOOKUP_CSV = "/Users/kiran/Downloads/lookups (1).csv"
DEFAULT_SIGLIP_MODEL = "Marqo/marqo-fashionSigLIP"
DEFAULT_CLIP_MODEL = "Marqo/marqo-fashionCLIP"


@dataclass(frozen=True)
class CategoryRow:
    id: str
    key: str
    label: str
    parent_id: str


class LookupCatalog:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.rows: List[CategoryRow] = []
        self._by_key: Dict[str, CategoryRow] = {}
        self._top_level: List[CategoryRow] = []
        self._children_by_parent: Dict[str, List[CategoryRow]] = {}
        self._load()

    def _load(self) -> None:
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Lookup CSV not found: {self.csv_path}")

        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("is_active", "")).strip().lower() not in {"1", "true", "yes", "on"}:
                    continue
                item = CategoryRow(
                    id=str(row.get("id", "")).strip(),
                    key=str(row.get("key", "")).strip(),
                    label=str(row.get("label", "")).strip(),
                    parent_id=str(row.get("parent_id", "")).strip(),
                )
                self.rows.append(item)
                self._by_key[item.key] = item
                if not item.parent_id:
                    self._top_level.append(item)
                else:
                    self._children_by_parent.setdefault(item.parent_id, []).append(item)

        self._top_level.sort(key=lambda r: r.label.lower())
        for children in self._children_by_parent.values():
            children.sort(key=lambda r: r.label.lower())

    def top_level_options(self) -> List[Dict[str, str]]:
        return [
            {
                "key": row.key,
                "label": row.label,
                "id": row.id,
                "subcategories": str(len(self.children_for(row.key))),
            }
            for row in self._top_level
        ]

    def children_for(self, top_level_key: str) -> List[CategoryRow]:
        top = self._by_key.get(top_level_key)
        if top is None or top.parent_id:
            return []
        return list(self._children_by_parent.get(top.id, []))

    def top_level_row(self, top_level_key: str) -> CategoryRow:
        row = self._by_key.get(top_level_key)
        if row is None:
            raise KeyError(top_level_key)
        if row.parent_id:
            raise ValueError(f"{top_level_key} is not a top-level category")
        return row


class FashionMatcher:
    def __init__(self, model_id: str):
        self.model_id = model_id
        preferred = os.getenv("CATEGORY_MATCH_DEVICE", "cpu").strip().lower()
        if preferred == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif preferred == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        self._processor = None
        self._model = None
        self._lock = threading.Lock()
        self._available = True

    @property
    def is_ready(self) -> bool:
        return self._processor is not None and self._model is not None

    def ensure_ready(self) -> bool:
        if not self._available:
            return False
        if self.is_ready:
            return True
        with self._lock:
            if self.is_ready:
                return True
            if not self._available:
                return False
            try:
                from transformers import AutoModel, AutoProcessor

                logger.info("Loading %s on %s", self.model_id, self.device)
                self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                self._model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=self.dtype,
                )
                self._model.to(self.device)
                self._model.eval()
            except Exception as exc:
                self._available = False
                self._processor = None
                self._model = None
                logger.exception("Failed to load %s: %s", self.model_id, exc)
                return False
        return True

    def score(self, image: Image.Image, labels: Sequence[str]) -> List[float]:
        if not labels:
            return []
        if not self.ensure_ready():
            uniform = 1.0 / float(len(labels))
            return [uniform for _ in labels]

        pil_image = image.convert("RGB")
        processed = self._processor(
            text=[str(x) for x in labels],
            images=[pil_image],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        processed = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}

        with torch.inference_mode():
            try:
                image_features = self._model.get_image_features(processed["pixel_values"], normalize=True)
                text_features = self._model.get_text_features(processed["input_ids"], normalize=True)
            except TypeError:
                image_features = self._model.get_image_features(processed["pixel_values"])
                text_features = self._model.get_text_features(processed["input_ids"])
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1).detach().float().cpu().squeeze(0).tolist()
        return [float(x) for x in probs]


def _resolve_csv_path() -> str:
    return os.getenv("LOOKUP_CSV_PATH", DEFAULT_LOOKUP_CSV)


def _resolve_model(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized in {"siglip", "fashionsiglip", "marqo-fashionsiglip", "marqo/marqo-fashionsiglip"}:
        return DEFAULT_SIGLIP_MODEL
    if normalized in {"clip", "fashionclip", "marqo-fashionclip", "marqo/marqo-fashionclip"}:
        return DEFAULT_CLIP_MODEL
    return model_name.strip()


@lru_cache(maxsize=4)
def get_catalog(csv_path: str) -> LookupCatalog:
    return LookupCatalog(csv_path)


@lru_cache(maxsize=8)
def get_matcher(model_id: str) -> FashionMatcher:
    return FashionMatcher(model_id)


def _load_image_from_upload(upload: UploadFile) -> Image.Image:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Empty image upload")
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}") from exc


def _rank_categories(image: Image.Image, categories: Sequence[CategoryRow], model_name: str) -> Dict[str, object]:
    labels = [row.label for row in categories]
    matcher = get_matcher(_resolve_model(model_name))
    scores = matcher.score(image, labels)
    ranked = sorted(
        (
            {
                "key": row.key,
                "label": row.label,
                "score": float(score),
            }
            for row, score in zip(categories, scores)
        ),
        key=lambda x: x["score"],
        reverse=True,
    )
    top = ranked[0] if ranked else None
    return {
        "model": matcher.model_id,
        "top": top,
        "ranked": ranked,
    }


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fashion Category Match Lab</title>
  <style>
    :root {
      --bg: #f7f1e8;
      --card: #fffdf9;
      --ink: #17181c;
      --muted: #646772;
      --line: #ddd5c7;
      --accent: #0f766e;
      --accent2: #92400e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 0% 0%, rgba(15,118,110,0.10), transparent 24%),
        radial-gradient(circle at 100% 0%, rgba(146,64,14,0.09), transparent 22%),
        linear-gradient(180deg, #fcf8f1 0%, #f3ede4 100%);
      min-height: 100vh;
      padding: 18px;
    }
    .wrap { max-width: 1280px; margin: 0 auto; display: grid; gap: 14px; }
    .hero, .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(20,20,20,0.04);
    }
    .hero { padding: 18px; }
    .hero h1 { margin: 0 0 8px 0; font-size: 28px; }
    .hero p { margin: 4px 0; color: var(--muted); }
    .card { padding: 14px; }
    .controls {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr 1fr auto;
      gap: 10px;
      align-items: end;
    }
    .field { display: grid; gap: 6px; }
    label {
      font-size: 11px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    input, select, button {
      min-height: 40px;
      border: 1px solid #d5ccbd;
      border-radius: 12px;
      padding: 8px 10px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    button {
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      border-color: var(--accent);
    }
    button.secondary {
      background: #fff;
      color: var(--accent2);
      border-color: #d9b892;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .grid2 { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 14px; }
    .viewer {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      overflow: auto;
      min-height: 280px;
      padding: 10px;
    }
    .viewer img {
      max-width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); text-align: left; padding: 8px 6px; vertical-align: top; }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
    pre {
      margin: 0;
      padding: 12px;
      background: #fbfaf7;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: auto;
      max-height: 360px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
    }
    .hint { color: var(--muted); font-size: 13px; }
    .error { color: #b42318; white-space: pre-wrap; }
    .compare {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    @media (max-width: 1080px) {
      .controls, .grid2, .compare { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Fashion Category Match Lab</h1>
      <p>Upload one garment image, pick a top-level family, and the app will choose the best matching subcategory from your CSV.</p>
      <p class="hint">This is a standalone lab app. It does not touch the existing analyze/tryon APIs.</p>
    </div>

    <div class="card">
      <div class="controls">
        <div class="field" style="grid-column: span 2;">
          <label for="fileInput">Image</label>
          <input id="fileInput" type="file" accept="image/*" />
        </div>
        <div class="field">
          <label for="familyInput">Top-level family</label>
          <select id="familyInput"></select>
        </div>
        <div class="field">
          <label for="modelInput">Model</label>
          <select id="modelInput">
            <option value="siglip">SigLIP (recommended)</option>
            <option value="clip">CLIP</option>
          </select>
        </div>
        <div class="field">
          <label for="apiBase">API base</label>
          <input id="apiBase" type="text" placeholder="defaults to current origin" />
        </div>
        <div class="field">
          <button id="runBtn">Run match</button>
        </div>
      </div>
      <p id="meta" class="hint" style="margin:10px 0 0 0;"></p>
      <p id="status" class="hint" style="margin:6px 0 0 0;"></p>
      <p id="error" class="error" style="margin:6px 0 0 0;"></p>
    </div>

    <div class="grid2">
      <div class="card">
        <h3 style="margin:0 0 10px 0;">Preview</h3>
        <div class="viewer">
          <img id="preview" alt="preview" />
        </div>
      </div>
      <div class="card">
        <h3 style="margin:0 0 10px 0;">Top Results</h3>
        <table>
          <thead><tr><th>#</th><th>key</th><th>label</th><th>score</th></tr></thead>
          <tbody id="resultRows"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="compare">
        <div>
          <h3 style="margin:0 0 10px 0;">SigLIP vs CLIP</h3>
          <pre id="compareOut">Run a match to compare both models.</pre>
        </div>
        <div>
          <h3 style="margin:0 0 10px 0;">Raw JSON</h3>
          <pre id="jsonOut">Run a match to inspect the payload.</pre>
        </div>
      </div>
    </div>
  </div>

  <script>
    const els = {
      file: document.getElementById("fileInput"),
      family: document.getElementById("familyInput"),
      model: document.getElementById("modelInput"),
      apiBase: document.getElementById("apiBase"),
      run: document.getElementById("runBtn"),
      preview: document.getElementById("preview"),
      meta: document.getElementById("meta"),
      status: document.getElementById("status"),
      error: document.getElementById("error"),
      rows: document.getElementById("resultRows"),
      compareOut: document.getElementById("compareOut"),
      jsonOut: document.getElementById("jsonOut"),
    };

    let currentObjectUrl = null;

    function baseUrl() {
      return els.apiBase.value.trim() || window.location.origin;
    }

    function setBusy(flag) {
      els.run.disabled = flag;
      els.status.textContent = flag ? "Running model inference..." : "";
    }

    function renderResults(items) {
      els.rows.innerHTML = items.map((item, idx) => `
        <tr>
          <td>${idx + 1}</td>
          <td><code>${item.key}</code></td>
          <td>${item.label}</td>
          <td>${(item.score * 100).toFixed(2)}%</td>
        </tr>
      `).join("");
    }

    function setPreview(file) {
      if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
      }
      currentObjectUrl = URL.createObjectURL(file);
      els.preview.src = currentObjectUrl;
    }

    async function loadMeta() {
      const res = await fetch(`${baseUrl()}/api/meta`);
      const data = await res.json();
      els.family.innerHTML = data.top_level_options.map(opt =>
        `<option value="${opt.key}">${opt.label} (${opt.subcategories})</option>`
      ).join("");
      els.meta.textContent = `Loaded ${data.total_categories} categories across ${data.top_level_options.length} top-level families.`;
    }

    async function runMatch() {
      els.error.textContent = "";
      if (!els.file.files.length) {
        els.error.textContent = "Choose an image first.";
        return;
      }
      const file = els.file.files[0];
      const form = new FormData();
      form.append("file", file);
      form.append("top_level_key", els.family.value);
      form.append("model", els.model.value);

      setBusy(true);
      try {
        const [singleRes, compareRes] = await Promise.all([
          fetch(`${baseUrl()}/api/predict`, { method: "POST", body: form }),
          fetch(`${baseUrl()}/api/compare`, { method: "POST", body: form }),
        ]);
        const single = await singleRes.json();
        const compare = await compareRes.json();
        els.jsonOut.textContent = JSON.stringify(single, null, 2);
        els.compareOut.textContent = JSON.stringify(compare, null, 2);
        renderResults(single.ranked);
        els.status.textContent = `Winner: ${single.top.label} (${single.top.key})`;
      } catch (err) {
        els.error.textContent = String(err);
      } finally {
        setBusy(false);
      }
    }

    els.file.addEventListener("change", () => {
      const file = els.file.files[0];
      if (file) setPreview(file);
    });
    els.run.addEventListener("click", runMatch);
    loadMeta().catch(err => {
      els.error.textContent = `Failed to load metadata: ${err}`;
    });
  </script>
</body>
</html>
"""


app = FastAPI(title="Fashion Category Match Lab", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/api/meta")
async def meta() -> JSONResponse:
    catalog = get_catalog(_resolve_csv_path())
    return JSONResponse(
        {
            "csv_path": catalog.csv_path,
            "total_categories": len(catalog.rows),
            "top_level_options": catalog.top_level_options(),
        }
    )


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    top_level_key: str = Form(...),
    model: str = Form(DEFAULT_SIGLIP_MODEL),
) -> JSONResponse:
    catalog = get_catalog(_resolve_csv_path())
    try:
        categories = catalog.children_for(top_level_key)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not categories:
        raise HTTPException(status_code=422, detail=f"No subcategories found for top-level key: {top_level_key}")

    image = _load_image_from_upload(file)
    result = _rank_categories(image, categories, model)
    result.update(
        {
            "top_level_key": top_level_key,
            "top_level_label": catalog.top_level_row(top_level_key).label,
            "candidate_count": len(categories),
        }
    )
    return JSONResponse(result)


@app.post("/api/compare")
async def compare(
    file: UploadFile = File(...),
    top_level_key: str = Form(...),
) -> JSONResponse:
    catalog = get_catalog(_resolve_csv_path())
    categories = catalog.children_for(top_level_key)
    if not categories:
        raise HTTPException(status_code=422, detail=f"No subcategories found for top-level key: {top_level_key}")

    image = _load_image_from_upload(file)
    siglip = _rank_categories(image, categories, DEFAULT_SIGLIP_MODEL)
    clip = _rank_categories(image, categories, DEFAULT_CLIP_MODEL)
    return JSONResponse(
        {
            "top_level_key": top_level_key,
            "top_level_label": catalog.top_level_row(top_level_key).label,
            "siglip": siglip,
            "clip": clip,
        }
    )


def batch_predict(paths: Sequence[str], top_level_key: str, model: str = DEFAULT_SIGLIP_MODEL) -> List[Dict[str, object]]:
    catalog = get_catalog(_resolve_csv_path())
    categories = catalog.children_for(top_level_key)
    if not categories:
        raise ValueError(f"No subcategories found for {top_level_key}")

    outputs: List[Dict[str, object]] = []
    for path in paths:
        image = Image.open(path)
        result = _rank_categories(image, categories, model)
        outputs.append(
            {
                "path": path,
                "top_level_key": top_level_key,
                "top_level_label": catalog.top_level_row(top_level_key).label,
                **result,
            }
        )
    return outputs


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("category_match_lab:app", host="127.0.0.1", port=8011, reload=False)
