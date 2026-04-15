"""
Standalone local UI for probing /analyze detection output.

This route does not modify analyze API behavior. It only serves an HTML page
that calls the existing /analyze endpoint and visualizes detections.
"""

import base64
import io
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from services import AIEngine

router = APIRouter()
_LAB_PAD_RATIO_X = 0.35
_LAB_PAD_RATIO_Y = 0.10
_LAB_BOTTOM_TOP_PAD_RATIO_Y = 0.05
_LAB_MIN_DIM_PX = 100

_TYPE_MAP = {
    "top": "top",
    "shirt": "top",
    "blouse": "top",
    "tee": "top",
    "vest": "top",
    "bottom": "bottom",
    "pants": "bottom",
    "trousers": "bottom",
    "jeans": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "dress": "dress",
    "gown": "dress",
    "outer": "outer",
    "outerwear": "outer",
    "jacket": "outer",
    "coat": "outer",
    "blazer": "outer",
}


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analyze Detection Lab</title>
  <style>
    :root {
      --bg: #f7f4ee;
      --card: #fffdf9;
      --ink: #1e1f24;
      --muted: #5e616a;
      --line: #e4ded1;
      --accent: #0f766e;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 90% 10%, #efe8db 0 12%, transparent 13%),
        radial-gradient(circle at 15% 80%, #ece6da 0 14%, transparent 15%),
        linear-gradient(180deg, #f8f5ee 0%, #f2eee5 100%);
      min-height: 100vh;
      padding: 20px;
    }
    .layout {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      gap: 14px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(7, minmax(120px, 1fr));
      gap: 10px;
      align-items: end;
    }
    .field { display: grid; gap: 6px; }
    label {
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.03em;
      text-transform: uppercase;
      font-weight: 600;
    }
    input, select, button {
      width: 100%;
      min-height: 38px;
      border: 1px solid #d6d0c5;
      border-radius: 10px;
      padding: 8px 10px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    button {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .hint { color: var(--muted); font-size: 13px; }
    .error { color: var(--danger); white-space: pre-wrap; }
    .viewer-wrap {
      position: relative;
      width: 100%;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      min-height: 260px;
    }
    .stage {
      position: relative;
      display: inline-block;
      line-height: 0;
    }
    #preview {
      max-width: min(100%, 980px);
      height: auto;
      display: block;
    }
    .bbox {
      position: absolute;
      border: 2px solid #dc2626;
      color: #fff;
      background: rgba(220, 38, 38, 0.84);
      padding: 1px 4px;
      font-size: 11px;
      font-weight: 700;
      border-radius: 4px;
      transform: translateY(-100%);
      white-space: nowrap;
      pointer-events: none;
    }
    .list {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .list th, .list td {
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px 6px;
      vertical-align: top;
    }
    .list th { color: var(--muted); font-weight: 700; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 420px;
      overflow: auto;
      background: #fbfaf7;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      font-size: 12px;
    }
    @media (max-width: 980px) {
      .controls { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="card">
      <h2 style="margin:0 0 10px 0;">Analyze Detection Lab (Local)</h2>
      <div class="controls">
        <div class="field" style="grid-column: span 2;">
          <label for="fileInput">Garment Images</label>
          <input id="fileInput" type="file" accept="image/*" multiple />
        </div>
        <div class="field">
          <label for="typeInput">Type (Optional)</label>
          <select id="typeInput">
            <option value="">none</option>
            <option value="top">top</option>
            <option value="bottom">bottom</option>
            <option value="dress">dress</option>
            <option value="outer">outer</option>
          </select>
        </div>
        <div class="field">
          <label for="minConfidence">Min Confidence</label>
          <input id="minConfidence" type="number" min="0" max="1" step="0.01" value="0.00" />
        </div>
        <div class="field">
          <label for="cropMode">Crop Mode</label>
          <select id="cropMode">
            <option value="padded">analyze-style padded</option>
            <option value="raw">raw detector bbox</option>
          </select>
        </div>
        <div class="field">
          <label for="apiBase">API Base</label>
          <input id="apiBase" type="text" value="" placeholder="defaults to current origin" />
        </div>
        <div class="field">
          <button id="runBtn">Run Detection</button>
        </div>
      </div>
      <p class="hint" style="margin:10px 0 0 0;">
        Uses core cloth detector directly for fast local testing (no full analyze pipeline).
      </p>
      <p class="hint" style="margin:6px 0 0 0;">
        Min size gate: <code>100x100</code> px. Padded mode uses <code>35% LR</code> and <code>10% TB</code>.
      </p>
      <p id="status" class="hint" style="margin:6px 0 0 0;"></p>
      <p id="error" class="error" style="margin:6px 0 0 0;"></p>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px 0;">Detected Boxes (Filtered)</h3>
      <div class="viewer-wrap">
        <div class="stage" id="stage">
          <img id="preview" alt="preview" />
        </div>
      </div>
      <div style="margin-top:10px; overflow:auto;">
        <table class="list" id="detTable">
          <thead>
            <tr>
              <th>#</th><th>type</th><th>label</th><th>confidence</th><th>bbox</th><th>source</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <h3 style="margin:14px 0 10px 0;">Identified Item Crops</h3>
      <div id="cropGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;"></div>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px 0;">Raw Response Payload</h3>
      <pre id="jsonOut">Run detection to see output.</pre>
    </div>
    <div class="card">
      <h3 style="margin:0 0 10px 0;">Upscale Preview (Pre-Qwen)</h3>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        <button id="extractBtn" type="button" style="max-width:220px;">Upscale Selected Crop</button>
        <span id="extractStatus" class="hint"></span>
      </div>
      <div style="margin-top:10px;display:grid;grid-template-columns:160px 1fr;gap:10px;align-items:start;">
        <img id="extractPreview" alt="extracted" style="width:120px;height:140px;object-fit:contain;background:#f8f7f4;border:1px solid var(--line);border-radius:8px;" />
        <div>
          <div class="hint" id="extractUrlText"></div>
          <button id="downloadExtractBtn" type="button" style="max-width:220px;margin-top:8px;" disabled>Download Upscaled</button>
        </div>
      </div>
      <pre id="extractJson" style="margin-top:10px;">Run upscale to see output.</pre>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById("fileInput");
    const typeInput = document.getElementById("typeInput");
    const minConfidenceInput = document.getElementById("minConfidence");
    const apiBaseInput = document.getElementById("apiBase");
    const cropModeInput = document.getElementById("cropMode");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const errorEl = document.getElementById("error");
    const jsonOut = document.getElementById("jsonOut");
    const extractBtn = document.getElementById("extractBtn");
    const extractStatus = document.getElementById("extractStatus");
    const extractPreview = document.getElementById("extractPreview");
    const extractJson = document.getElementById("extractJson");
    const extractUrlText = document.getElementById("extractUrlText");
    const downloadExtractBtn = document.getElementById("downloadExtractBtn");
    const preview = document.getElementById("preview");
    const stage = document.getElementById("stage");
    const detBody = document.querySelector("#detTable tbody");
    const cropGrid = document.getElementById("cropGrid");

    let allItems = [];
    let lastShownItems = [];
    let selectedItem = null;
    let selectedItemIndex = -1;
    let lastUpscaledDataUrl = "";
    let currentNaturalWidth = 1;
    let currentNaturalHeight = 1;

    function readMinConfidence() {
      const raw = Number(minConfidenceInput.value || 0);
      if (!Number.isFinite(raw)) return 0;
      return Math.max(0, Math.min(1, raw));
    }

    function pickDataRoot(payload) {
      if (payload && typeof payload === "object" && payload.data && typeof payload.data === "object") {
        return payload.data;
      }
      return payload || {};
    }

    function extractItems(payload) {
      const root = pickDataRoot(payload);
      const direct = Array.isArray(root.items) ? root.items : [];
      if (direct.length > 0) return direct;
      const raw = Array.isArray(root.raw_items) ? root.raw_items : [];
      if (raw.length > 0) return raw;
      const selected = root.selected_item;
      if (selected && typeof selected === "object") return [selected];
      return [];
    }

    function itemConfidence(item) {
      const conf = item && typeof item.confidence === "object" ? item.confidence : {};
      const candidates = [conf.hybrid, conf.detector, conf.yolo, conf.florence_type, item.score, item.confidence];
      for (const val of candidates) {
        const num = Number(val);
        if (Number.isFinite(num)) return num;
      }
      return 0;
    }

    function bboxForMode(item) {
      const mode = String(cropModeInput.value || "padded");
      const source = mode === "raw" ? item.bbox : (item.padded_bbox || item.bbox);
      return Array.isArray(source) && source.length === 4 ? source : null;
    }

    function clearOverlay() {
      stage.querySelectorAll(".bbox").forEach((n) => n.remove());
      detBody.innerHTML = "";
      cropGrid.innerHTML = "";
      lastShownItems = [];
      selectedItem = null;
      selectedItemIndex = -1;
    }

    function setSelectedItemByIndex(idx) {
      const n = Number(idx);
      if (!Number.isInteger(n) || n < 0 || n >= lastShownItems.length) return;
      selectedItemIndex = n;
      selectedItem = lastShownItems[n] || null;
      cropGrid.querySelectorAll("[data-crop-idx]").forEach((el) => {
        const active = Number(el.getAttribute("data-crop-idx")) === selectedItemIndex;
        el.style.outline = active ? "2px solid var(--accent)" : "none";
      });
    }

    function bboxToBlob(item) {
      return new Promise((resolve) => {
        const bbox = bboxForMode(item);
        if (!bbox || !preview.src) return resolve(null);
        const [x0, y0, x1, y1] = bbox.map((v) => Number(v) || 0);
        const w = Math.max(1, x1 - x0);
        const h = Math.max(1, y1 - y0);
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return resolve(null);
        ctx.drawImage(preview, x0, y0, w, h, 0, 0, w, h);
        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
      });
    }

    function cropDataUrlFromBbox(bbox) {
      if (!Array.isArray(bbox) || bbox.length !== 4 || !preview.src) return "";
      const [x0, y0, x1, y1] = bbox.map((v) => Number(v) || 0);
      const w = Math.max(1, x1 - x0);
      const h = Math.max(1, y1 - y0);
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return "";
      ctx.drawImage(preview, x0, y0, w, h, 0, 0, w, h);
      return canvas.toDataURL("image/jpeg", 0.92);
    }

    async function downloadBlob(blob, filename) {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function renderCrops(items) {
      cropGrid.innerHTML = "";
      if (!preview.src || !Array.isArray(items) || items.length === 0) return;
      items.forEach((item, idx) => {
        const rawBbox = Array.isArray(item.bbox) && item.bbox.length === 4 ? item.bbox : null;
        const paddedBbox = Array.isArray(item.padded_bbox) && item.padded_bbox.length === 4 ? item.padded_bbox : rawBbox;
        const activeBbox = bboxForMode(item);
        if (!activeBbox || activeBbox.length !== 4) return;
        const [x0, y0, x1, y1] = activeBbox.map((v) => Number(v) || 0);
        const rawUrl = cropDataUrlFromBbox(rawBbox);
        const paddedUrl = cropDataUrlFromBbox(paddedBbox);
        const conf = itemConfidence(item).toFixed(4);
        const type = String(item.type || item.garment_type || "item");
        const label = String(item.detector_label || item.label || "-");
        const card = document.createElement("div");
        card.setAttribute("data-crop-idx", String(idx));
        card.style.border = "1px solid var(--line)";
        card.style.borderRadius = "10px";
        card.style.padding = "8px";
        card.style.background = "#fff";
        card.innerHTML = `
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
            <div>
              <div style="font-size:11px;color:var(--muted);margin-bottom:4px;">raw</div>
              <img src="${rawUrl}" alt="${type}-${idx}-raw" style="width:100%;height:110px;display:block;object-fit:contain;background:#f8f7f4;border-radius:6px;" />
            </div>
            <div>
              <div style="font-size:11px;color:var(--muted);margin-bottom:4px;">padded</div>
              <img src="${paddedUrl}" alt="${type}-${idx}-padded" style="width:100%;height:110px;display:block;object-fit:contain;background:#f8f7f4;border-radius:6px;" />
            </div>
          </div>
          <div style="font-size:12px;margin-top:6px;"><strong>${type}</strong> (${conf})</div>
          <div style="font-size:12px;color:var(--muted);">${label}</div>
          <div style="font-size:11px;color:var(--muted);">[${[x0,y0,x1,y1].join(", ")}]</div>
          <div style="display:grid;gap:6px;margin-top:8px;">
            <button type="button" data-action="select">Use For Extraction</button>
            <button type="button" data-action="extract">Extract</button>
            <button type="button" data-action="download">Download Crop</button>
          </div>`;
        card.querySelectorAll("button").forEach((btn) => {
          btn.style.minHeight = "32px";
          btn.style.fontSize = "12px";
          btn.style.padding = "6px 8px";
          btn.style.maxWidth = "100%";
          btn.style.width = "100%";
        });
        card.addEventListener("click", async (ev) => {
          const target = ev.target;
          if (!(target instanceof HTMLElement)) return;
          const action = target.getAttribute("data-action");
          if (!action) return;
          if (action === "select") {
            setSelectedItemByIndex(idx);
            return;
          }
          if (action === "download") {
            const blob = await bboxToBlob(item);
            await downloadBlob(blob, `detected_crop_${idx + 1}.jpg`);
            return;
          }
          if (action === "extract") {
            setSelectedItemByIndex(idx);
            await runExtractionSelected();
          }
        });
        cropGrid.appendChild(card);
      });
      if (items.length > 0 && selectedItemIndex < 0) {
        setSelectedItemByIndex(0);
      }
    }

    function drawFiltered() {
      clearOverlay();
      if (!preview.src || !Array.isArray(allItems) || allItems.length === 0) return;
      const minConf = readMinConfidence();
      const scaleX = preview.clientWidth / Math.max(1, currentNaturalWidth);
      const scaleY = preview.clientHeight / Math.max(1, currentNaturalHeight);
      let shown = 0;

      const shownItems = [];
      allItems.forEach((item, idx) => {
        const conf = itemConfidence(item);
        if (conf < minConf) return;
        const bbox = bboxForMode(item);
        if (!bbox || bbox.length !== 4) return;
        const [x0, y0, x1, y1] = bbox.map((v) => Number(v) || 0);
        const w = Math.max(1, x1 - x0);
        const h = Math.max(1, y1 - y0);
        const box = document.createElement("div");
        box.className = "bbox";
        box.style.left = (x0 * scaleX) + "px";
        box.style.top = (y0 * scaleY) + "px";
        box.style.width = (w * scaleX) + "px";
        box.style.height = (h * scaleY) + "px";
        const type = String(item.type || item.garment_type || "");
        box.textContent = `${type || "item"} ${conf.toFixed(3)}`;
        stage.appendChild(box);
        shown += 1;
        shownItems.push(item);

        const tr = document.createElement("tr");
        const pad = item.padding_pixels || {};
        tr.innerHTML = `<td>${shown}</td>
          <td>${type || "-"}</td>
          <td>${String(item.detector_label || item.label || "-")}</td>
          <td>${conf.toFixed(4)}</td>
          <td>[${[x0,y0,x1,y1].join(", ")}]<br/><span style="font-size:11px;color:var(--muted);">pad L/R/T/B: ${pad.left ?? "?"}/${pad.right ?? "?"}/${pad.top ?? "?"}/${pad.bottom ?? "?"}</span></td>
          <td>${String(item.detection_source || item.source || "-")}</td>`;
        detBody.appendChild(tr);
      });
      lastShownItems = shownItems;
      renderCrops(shownItems);

      statusEl.textContent = `Showing ${shown}/${allItems.length} detections at min_confidence >= ${minConf.toFixed(2)}`;
    }

    async function parseAnalyzeResponse(resp) {
      const ctype = String(resp.headers.get("content-type") || "").toLowerCase();
      if (ctype.includes("multipart/form-data")) {
        const form = await resp.formData();
        const meta = form.get("metadata");
        if (meta instanceof File) {
          const text = await meta.text();
          try { return JSON.parse(text); } catch { return { raw_metadata_text: text }; }
        }
        if (typeof meta === "string") {
          try { return JSON.parse(meta); } catch { return { raw_metadata_text: meta }; }
        }
        const obj = {};
        for (const [key, value] of form.entries()) {
          obj[key] = value instanceof File ? { filename: value.name, type: value.type, size: value.size } : String(value);
        }
        return obj;
      }
      const text = await resp.text();
      try { return JSON.parse(text); } catch { return { raw_text: text }; }
    }

    runBtn.addEventListener("click", async () => {
      errorEl.textContent = "";
      statusEl.textContent = "";
      clearOverlay();
      const files = Array.from(fileInput.files || []);
      if (files.length === 0) {
        errorEl.textContent = "Choose one or more image files first.";
        return;
      }

      const previewUrl = URL.createObjectURL(files[0]);
      preview.src = previewUrl;
      await new Promise((resolve) => {
        preview.onload = resolve;
        preview.onerror = resolve;
      });
      currentNaturalWidth = preview.naturalWidth || 1;
      currentNaturalHeight = preview.naturalHeight || 1;

      runBtn.disabled = true;
      statusEl.textContent = `Running detector on ${files.length} file(s) ...`;
      try {
        const base = (apiBaseInput.value || "").trim() || window.location.origin;
        const url = base.replace(/\\/$/, "") + "/analyze-detection-lab/run";
        const reqType = (typeInput.value || "").trim();
        const batch = [];
        let firstSuccessPayload = null;
        for (const file of files) {
          const form = new FormData();
          form.append("file", file, file.name);
          if (reqType) form.append("type", reqType);
          form.append("min_confidence", String(readMinConfidence()));
          const resp = await fetch(url, { method: "POST", body: form });
          const payload = await parseAnalyzeResponse(resp);
          batch.push({
            file: file.name,
            status_code: resp.status,
            ok: resp.ok,
            payload: payload,
          });
          if (resp.ok && firstSuccessPayload === null) {
            firstSuccessPayload = payload;
          }
        }
        jsonOut.textContent = JSON.stringify({ results: batch }, null, 2);
        if (firstSuccessPayload) {
          allItems = extractItems(firstSuccessPayload);
          drawFiltered();
        }
        const okCount = batch.filter((x) => x.ok).length;
        statusEl.textContent = `Completed ${okCount}/${batch.length} successfully`;
      } catch (err) {
        errorEl.textContent = String(err && err.message ? err.message : err);
      } finally {
        runBtn.disabled = false;
      }
    });

    preview.addEventListener("load", drawFiltered);
    minConfidenceInput.addEventListener("input", drawFiltered);
    cropModeInput.addEventListener("change", drawFiltered);

    async function runExtractionSelected() {
      if (!selectedItem) {
        extractStatus.textContent = "Select an item crop first.";
        return;
      }
      extractBtn.disabled = true;
      extractStatus.textContent = "Running local upscale policy...";
      try {
        const blob = await bboxToBlob(selectedItem);
        if (!blob) throw new Error("Failed to build selected crop image.");
        const base = (apiBaseInput.value || "").trim() || window.location.origin;
        const url = base.replace(/\\/$/, "") + "/analyze-detection-lab/upscale";
        const form = new FormData();
        form.append("file", blob, `selected_crop_${selectedItemIndex + 1}.jpg`);
        const resp = await fetch(url, { method: "POST", body: form });
        const payload = await parseAnalyzeResponse(resp);
        extractJson.textContent = JSON.stringify(payload, null, 2);
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}: ${JSON.stringify(payload)}`);
        }
        const root = pickDataRoot(payload);
        const data = root && typeof root === "object" ? (root.data && typeof root.data === "object" ? root.data : root) : {};
        const dataUrl = String(data.image_data_url || "").trim();
        const inSize = data.input_size || {};
        const outSize = data.output_size || {};
        const applied = Boolean(data.upscale_applied);
        lastUpscaledDataUrl = dataUrl;
        extractUrlText.textContent = `input ${inSize.width || "?"}x${inSize.height || "?"} -> output ${outSize.width || "?"}x${outSize.height || "?"} | upscale_applied=${applied} | method=lanczos`;
        if (dataUrl) {
          extractPreview.src = dataUrl;
          downloadExtractBtn.disabled = false;
        } else {
          downloadExtractBtn.disabled = true;
        }
        extractStatus.textContent = "Upscale completed.";
      } catch (err) {
        extractStatus.textContent = String(err && err.message ? err.message : err);
      } finally {
        extractBtn.disabled = false;
      }
    }

    extractBtn.addEventListener("click", runExtractionSelected);
    downloadExtractBtn.addEventListener("click", async () => {
      try {
        if (!lastUpscaledDataUrl) return;
        const resp = await fetch(lastUpscaledDataUrl);
        if (!resp.ok) throw new Error(`Download failed: HTTP ${resp.status}`);
        const blob = await resp.blob();
        await downloadBlob(blob, "upscaled_result.png");
      } catch (err) {
        extractStatus.textContent = String(err && err.message ? err.message : err);
      }
    });
  </script>
</body>
</html>
"""


@router.get("/analyze-detection-lab", response_class=HTMLResponse)
async def analyze_detection_lab_page() -> HTMLResponse:
    return HTMLResponse(content=_HTML)


def get_ai_engine() -> AIEngine:
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_ai_engine()


def _normalize_type(label: str) -> str:
    txt = " ".join(str(label or "").lower().replace("_", " ").replace("-", " ").split())
    if txt in _TYPE_MAP:
        return _TYPE_MAP[txt]
    if "dress" in txt or "gown" in txt:
        return "dress"
    if any(t in txt for t in ("pant", "trouser", "jean", "short", "skirt", "bottom")):
        return "bottom"
    if any(t in txt for t in ("coat", "jacket", "blazer", "outer")):
        return "outer"
    return ""


def _padded_bbox(
    bbox: list[int],
    *,
    width: int,
    height: int,
    garment_type: str = "",
) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in (bbox or [0, 0, width, height])]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    px = int(round(bw * _LAB_PAD_RATIO_X))
    kind = str(garment_type or "").strip().lower()
    top_ratio = _LAB_BOTTOM_TOP_PAD_RATIO_Y if kind == "bottom" else _LAB_PAD_RATIO_Y
    py_top = int(round(bh * top_ratio))
    py_bottom = int(round(bh * _LAB_PAD_RATIO_Y))
    nx0 = max(0, x0 - px)
    ny0 = max(0, y0 - py_top)
    nx1 = min(width, x1 + px)
    ny1 = min(height, y1 + py_bottom)
    if nx1 <= nx0:
        nx1 = min(width, nx0 + 1)
    if ny1 <= ny0:
        ny1 = min(height, ny0 + 1)
    return [int(nx0), int(ny0), int(nx1), int(ny1)]


@router.post("/analyze-detection-lab/upscale")
async def analyze_detection_lab_upscale(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")
    try:
        raw = await upload.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    in_w, in_h = pil.size
    longest = max(in_w, in_h)
    target_longest = 768
    upscale_applied = bool(longest < target_longest)
    if upscale_applied:
        ratio = float(target_longest) / float(max(1, longest))
        out_size = (
            max(1, int(round(in_w * ratio))),
            max(1, int(round(in_h * ratio))),
        )
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        out = pil.resize(out_size, resampling)
    else:
        out = pil

    out_buf = io.BytesIO()
    out.save(out_buf, format="PNG")
    image_base64 = base64.b64encode(out_buf.getvalue()).decode("ascii")

    return {
        "status": "ok",
        "data": {
            "policy": {
                "min_longest_edge_px": target_longest,
                "only_if_below_threshold": True,
                "resample": "lanczos",
            },
            "input_size": {"width": int(in_w), "height": int(in_h)},
            "output_size": {"width": int(out.width), "height": int(out.height)},
            "upscale_applied": bool(upscale_applied),
            "image_base64": image_base64,
            "image_data_url": f"data:image/png;base64,{image_base64}",
        },
    }


@router.post("/analyze-detection-lab/run")
async def analyze_detection_lab_run(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form(None, alias="type"),
    min_confidence: float = Form(0.0),
    engine: AIEngine = Depends(get_ai_engine),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")
    try:
        raw = await upload.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    detector = getattr(engine, "cloth_detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="cloth_detector is not available on this runtime")

    t0 = time.perf_counter()
    backend = "fashion_object_detection"
    try:
        candidates = detector.detect_fashion_candidates(pil, requested_type=garment_type)
    except Exception:
        # Offline/local fallback: use local YOLO cropper candidates.
        yolo_cropper = getattr(engine, "yolo", None)
        if yolo_cropper is None:
            raise HTTPException(status_code=500, detail="Detection failed and local YOLO fallback is unavailable")
        instances = yolo_cropper.detect_instances(pil)
        crops = yolo_cropper.get_crops(pil, instances)
        candidates = []
        for c in crops:
            label = str(c.get("label") or "")
            resolved_type = _normalize_type(label)
            if resolved_type not in {"top", "bottom", "dress", "outer"}:
                continue
            candidates.append(
                {
                    "type": resolved_type,
                    "label": label,
                    "confidence": float(c.get("confidence", 0.0) or 0.0),
                    "bbox": [int(v) for v in (c.get("bbox") or [0, 0, pil.width, pil.height])],
                    "source": "legacy_yolo",
                    "type_source": "legacy_yolo",
                    "metrics": {},
                }
            )
        backend = "legacy_yolo"
    elapsed = round(time.perf_counter() - t0, 4)

    threshold = max(0.0, min(1.0, float(min_confidence)))
    raw_items = []
    for idx, item in enumerate(candidates or []):
        bbox = [int(v) for v in (item.get("bbox") or [0, 0, 0, 0])]
        x0, y0, x1, y1 = bbox
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        item_type = str(item.get("type") or "")
        padded = _padded_bbox(
            bbox,
            width=pil.width,
            height=pil.height,
            garment_type=item_type,
        )
        px0, py0, px1, py1 = padded
        pbw = max(1, px1 - px0)
        pbh = max(1, py1 - py0)
        size_ok = bw >= _LAB_MIN_DIM_PX and bh >= _LAB_MIN_DIM_PX
        raw_items.append(
            {
                "index": idx,
                "type": item_type,
                "label": str(item.get("label") or ""),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "bbox": bbox,
                "bbox_size": {"width": bw, "height": bh},
                "padded_bbox": padded,
                "padded_bbox_size": {"width": pbw, "height": pbh},
                "padding_pixels": {
                    "left": int(max(0, x0 - px0)),
                    "right": int(max(0, px1 - x1)),
                    "top": int(max(0, y0 - py0)),
                    "bottom": int(max(0, py1 - y1)),
                },
                "padding_ratio_requested": {
                    "x_left_right": float(_LAB_PAD_RATIO_X),
                    "y_top": float(_LAB_BOTTOM_TOP_PAD_RATIO_Y if item_type == "bottom" else _LAB_PAD_RATIO_Y),
                    "y_bottom": float(_LAB_PAD_RATIO_Y),
                },
                "min_size_pass": bool(size_ok),
                "source": str(item.get("source") or "cloth_detector"),
                "type_source": str(item.get("type_source") or ""),
                "metrics": dict(item.get("metrics") or {}),
            }
        )

    filtered = []
    dropped_items = []
    for item in raw_items:
        conf = float(item.get("confidence", 0.0) or 0.0)
        if not bool(item.get("min_size_pass")):
            dropped_items.append({"index": int(item.get("index", -1)), "reason": "min_dimension"})
            continue
        if conf < threshold:
            dropped_items.append({"index": int(item.get("index", -1)), "reason": "low_confidence"})
            continue
        filtered.append(item)

    return {
        "status": "ok",
        "data": {
            "requested_type": garment_type,
            "min_confidence": threshold,
            "min_dimension_px": int(_LAB_MIN_DIM_PX),
            "crop_pad_ratio_x": float(_LAB_PAD_RATIO_X),
            "crop_pad_ratio_y": float(_LAB_PAD_RATIO_Y),
            "backend": backend,
            "elapsed_seconds": elapsed,
            "image_size": {"width": pil.width, "height": pil.height},
            "raw_count": len(candidates or []),
            "filtered_count": len(filtered),
            "raw_items": raw_items,
            "dropped_items": dropped_items,
            "items": filtered,
        },
    }
