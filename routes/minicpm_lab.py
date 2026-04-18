"""
Developer lab routes for MiniCPM garment/prompt tuning.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from config.prompts import get_minicpm_garment_prompt, get_minicpm_person_outfit_prompt

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _word_count(text: str) -> int:
    return len([part for part in str(text or "").split() if part.strip()])


def _load_uploaded_image(upload: UploadFile, field_name: str) -> Image.Image:
    try:
        raw = upload.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read {field_name}: {exc}") from exc
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not a valid image: {exc}") from exc


def _resolve_minicpm_runner(request: Request):
    engine = getattr(getattr(request, "app", object()), "state", object())
    ai_engine = getattr(engine, "ai_engine", None)
    runner = getattr(ai_engine, "minicpm", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="MiniCPM runner is unavailable on this pod.")
    return runner


def _quality_signals(text: str) -> Dict[str, object]:
    lowered = str(text or "").lower()
    human_terms = [
        "person",
        "model",
        "woman",
        "man",
        "skin",
        "hand",
        "arm",
        "leg",
        "face",
        "hair",
        "body",
        "mannequin",
    ]
    directional_terms = ["left", "right", "front", "back", "side", "viewer-left", "viewer-right"]
    color_terms = [
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "pink",
        "orange",
        "brown",
        "purple",
        "grey",
        "gray",
        "beige",
    ]

    human_hits = [token for token in human_terms if token in lowered]
    directional_hits = [token for token in directional_terms if token in lowered]
    color_hits = [token for token in color_terms if token in lowered]
    return {
        "human_term_hits": human_hits,
        "directional_term_hits": directional_hits,
        "color_term_hits": color_hits,
        "looks_like_json": lowered.startswith("{") and lowered.endswith("}"),
    }


def _build_default_instruction(*, mode: str, garment_type: Optional[str]) -> str:
    if mode == "person_outfit":
        return str(get_minicpm_person_outfit_prompt() or "").strip()
    return str(get_minicpm_garment_prompt(garment_type) or "").strip()


@router.get("/dev/minicpm/garment-lab", response_class=HTMLResponse)
async def minicpm_garment_lab_page() -> HTMLResponse:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MiniCPM Garment Prompt Lab</title>
  <style>
    :root {
      --bg-0: #f8fafc;
      --bg-1: #e2e8f0;
      --ink: #111827;
      --muted: #475569;
      --line: #cbd5e1;
      --card: #ffffff;
      --accent: #0f766e;
      --accent-2: #1e3a8a;
      --danger: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Space Grotesk", "Manrope", "Avenir Next", sans-serif;
      background:
        radial-gradient(1200px 700px at 15% -20%, #dbeafe 0%, transparent 65%),
        radial-gradient(900px 560px at 90% -20%, #d1fae5 0%, transparent 58%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1));
      min-height: 100vh;
    }
    .wrap { max-width: 1340px; margin: 18px auto 26px; padding: 0 14px; }
    h1 { margin: 0 0 14px; font-size: 30px; }
    .sub { margin: 0 0 14px; color: var(--muted); font-size: 14px; }
    .layout { display: grid; grid-template-columns: 420px 1fr; gap: 14px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 8px 20px rgba(17, 24, 39, 0.06);
      padding: 14px;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.35px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 5px;
    }
    input[type="file"], input[type="number"], input[type="text"], select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
      padding: 9px 10px;
    }
    textarea { min-height: 172px; line-height: 1.35; resize: vertical; }
    .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
    .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .btn {
      border: 0;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #fff;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .btn:disabled { opacity: 0.65; cursor: wait; }
    .btn.gray { background: #334155; }
    .status { font-size: 13px; color: var(--muted); min-height: 20px; margin-top: 8px; }
    .status.err { color: var(--danger); font-weight: 700; }
    .tile {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f8fafc;
      padding: 8px;
      margin-bottom: 10px;
    }
    .tile h4 {
      margin: 0 0 6px;
      font-size: 12px;
      letter-spacing: 0.35px;
      text-transform: uppercase;
      color: var(--muted);
    }
    .preview {
      width: 100%;
      height: 320px;
      object-fit: contain;
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      display: block;
    }
    .mini {
      background: #f8fafc;
      border: 1px dashed #cbd5e1;
      border-radius: 10px;
      padding: 10px;
      font-size: 13px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .results { display: grid; gap: 10px; }
    .result-card {
      border: 1px solid #cbd5e1;
      background: #fff;
      border-radius: 10px;
      padding: 10px;
    }
    .result-meta {
      font-size: 12px;
      color: #334155;
      margin-bottom: 8px;
      white-space: pre-wrap;
    }
    pre {
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 10px;
      padding: 10px;
      font-size: 12px;
      max-height: 320px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    @media (max-width: 1140px) {
      .layout { grid-template-columns: 1fr; }
      .row3 { grid-template-columns: 1fr; }
      .preview { height: 260px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MiniCPM Garment Prompt Lab</h1>
    <p class="sub">Tune only MiniCPM prompt behavior. This lab shows exact instruction, raw output, and leakage checks.</p>
    <div class="layout">
      <section class="card">
        <form id="lab-form">
          <div>
            <label>Input Image</label>
            <input id="image" name="image" type="file" accept="image/*" required />
          </div>
          <div class="row3">
            <div>
              <label>Mode</label>
              <select id="mode" name="mode">
                <option value="garment">garment</option>
                <option value="person_outfit">person_outfit</option>
              </select>
            </div>
            <div>
              <label>Garment Type</label>
              <select id="garment_type" name="garment_type">
                <option value="">auto</option>
                <option value="top">top</option>
                <option value="bottom">bottom</option>
                <option value="dress">dress</option>
                <option value="outer">outer</option>
              </select>
            </div>
            <div>
              <label>Runs</label>
              <input id="runs" name="runs" type="number" min="1" max="5" step="1" value="1" />
            </div>
          </div>
          <div>
            <label>Prompt Override (optional)</label>
            <textarea id="prompt_override" name="prompt_override" placeholder="Leave empty to use default MiniCPM prompt."></textarea>
          </div>
          <div class="actions">
            <button class="btn" id="run-btn" type="submit">Run MiniCPM</button>
            <button class="btn gray" id="fill-default-btn" type="button">Fill Default Prompt</button>
            <button class="btn gray" id="clear-btn" type="button">Clear Prompt</button>
          </div>
          <div class="status" id="status">Ready.</div>
        </form>
      </section>

      <section class="card">
        <div class="tile">
          <h4>Input Preview</h4>
          <img id="img-input" class="preview" alt="input preview" />
        </div>
        <div class="tile">
          <h4>Instruction Sent To MiniCPM</h4>
          <div id="instruction-used" class="mini"></div>
        </div>
        <div class="tile">
          <h4>Summary</h4>
          <div id="summary" class="mini"></div>
        </div>
        <div class="tile">
          <h4>Runs Output</h4>
          <div id="runs-output" class="results"></div>
        </div>
        <div class="tile">
          <h4>Raw JSON</h4>
          <pre id="raw-json"></pre>
        </div>
      </section>
    </div>
  </div>

  <script>
    const form = document.getElementById("lab-form");
    const imageInput = document.getElementById("image");
    const modeInput = document.getElementById("mode");
    const garmentTypeInput = document.getElementById("garment_type");
    const runsInput = document.getElementById("runs");
    const promptOverrideInput = document.getElementById("prompt_override");
    const fillDefaultBtn = document.getElementById("fill-default-btn");
    const clearBtn = document.getElementById("clear-btn");
    const runBtn = document.getElementById("run-btn");
    const statusEl = document.getElementById("status");
    const imgInput = document.getElementById("img-input");
    const instructionUsedEl = document.getElementById("instruction-used");
    const summaryEl = document.getElementById("summary");
    const runsOutputEl = document.getElementById("runs-output");
    const rawEl = document.getElementById("raw-json");

    function setStatus(message, isError=false) {
      statusEl.textContent = message;
      statusEl.className = isError ? "status err" : "status";
    }

    function previewInput() {
      if (!imageInput.files || !imageInput.files[0]) return;
      const url = URL.createObjectURL(imageInput.files[0]);
      imgInput.src = url;
    }

    async function fetchDefaultPrompt() {
      try {
        const mode = modeInput.value;
        const garmentType = garmentTypeInput.value;
        const response = await fetch(`/dev/minicpm/garment-lab/default-prompt?mode=${encodeURIComponent(mode)}&garment_type=${encodeURIComponent(garmentType)}`);
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload?.detail || `HTTP ${response.status}`);
        }
        promptOverrideInput.value = payload?.data?.default_instruction || "";
      } catch (error) {
        setStatus(`Failed to load default prompt: ${error.message}`, true);
      }
    }

    imageInput.addEventListener("change", previewInput);
    fillDefaultBtn.addEventListener("click", fetchDefaultPrompt);
    clearBtn.addEventListener("click", () => { promptOverrideInput.value = ""; });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!imageInput.files?.length) {
        setStatus("Please upload an image.", true);
        return;
      }

      const data = new FormData();
      data.append("image", imageInput.files[0]);
      data.append("mode", modeInput.value);
      data.append("garment_type", garmentTypeInput.value || "");
      data.append("runs", runsInput.value || "1");
      if (promptOverrideInput.value && promptOverrideInput.value.trim()) {
        data.append("prompt_override", promptOverrideInput.value.trim());
      }

      runBtn.disabled = true;
      setStatus("Running MiniCPM...");
      instructionUsedEl.textContent = "";
      summaryEl.textContent = "";
      runsOutputEl.innerHTML = "";
      rawEl.textContent = "";

      try {
        const response = await fetch("/dev/minicpm/garment-lab/run", {
          method: "POST",
          body: data
        });
        const payload = await response.json();
        if (!response.ok) {
          const detail = payload?.detail || payload?.message || `HTTP ${response.status}`;
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }

        const d = payload?.data || {};
        instructionUsedEl.textContent = d?.payload_sent_to_minicpm?.instruction || "";
        const summary = d?.summary || {};
        summaryEl.textContent =
          `Runs: ${summary?.runs ?? "?"}\n` +
          `Unique outputs: ${summary?.unique_outputs ?? "?"}\n` +
          `Avg elapsed: ${summary?.avg_elapsed_seconds ?? "?"} sec\n` +
          `Total elapsed: ${summary?.total_elapsed_seconds ?? "?"} sec\n` +
          `Prompt override used: ${d?.prompt_override_applied ? "yes" : "no"}\n` +
          `Model: ${d?.runner?.model_id || "n/a"}\n` +
          `Device: ${d?.runner?.device || "n/a"} | Dtype: ${d?.runner?.dtype || "n/a"}`;

        const runs = d?.runs || [];
        for (const run of runs) {
          const card = document.createElement("div");
          card.className = "result-card";

          const leakage = run?.quality_signals || {};
          const metaText =
            `Run #${run?.index ?? "?"} | elapsed: ${run?.elapsed_seconds ?? "?"} sec | words: ${run?.word_count ?? "?"}\n` +
            `Human hits: ${(leakage?.human_term_hits || []).join(", ") || "-"}\n` +
            `Directional hits: ${(leakage?.directional_term_hits || []).join(", ") || "-"}\n` +
            `Color hits: ${(leakage?.color_term_hits || []).join(", ") || "-"}\n` +
            `Looks like JSON: ${leakage?.looks_like_json ? "yes" : "no"}`;

          const meta = document.createElement("div");
          meta.className = "result-meta";
          meta.textContent = metaText;

          const text = document.createElement("div");
          text.className = "mini";
          text.textContent = run?.text || "";

          card.appendChild(meta);
          card.appendChild(text);
          runsOutputEl.appendChild(card);
        }

        rawEl.textContent = JSON.stringify(payload, null, 2);
        setStatus("MiniCPM run completed.");
      } catch (error) {
        setStatus(`Run failed: ${error.message}`, true);
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/dev/minicpm/garment-lab/default-prompt")
async def minicpm_garment_lab_default_prompt(
    mode: str = "garment",
    garment_type: str = "",
) -> Dict[str, object]:
    normalized_mode = _normalize_text(mode).lower() or "garment"
    if normalized_mode not in {"garment", "person_outfit"}:
        raise HTTPException(status_code=422, detail="mode must be one of: garment, person_outfit")
    default_instruction = _build_default_instruction(mode=normalized_mode, garment_type=garment_type)
    return {
        "status": "success",
        "message": "Default prompt resolved",
        "data": {
            "mode": normalized_mode,
            "garment_type": _normalize_text(garment_type).lower(),
            "default_instruction": default_instruction,
        },
    }


@router.post("/dev/minicpm/garment-lab/run")
async def minicpm_garment_lab_run(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    mode: str = Form("garment"),
    garment_type: Optional[str] = Form(None),
    prompt_override: Optional[str] = Form(None),
    runs: int = Form(1, ge=1, le=5),
) -> Dict[str, object]:
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide one image using 'file' or 'image'.")

    normalized_mode = _normalize_text(mode).lower() or "garment"
    if normalized_mode not in {"garment", "person_outfit"}:
        raise HTTPException(status_code=422, detail="mode must be one of: garment, person_outfit")

    normalized_garment_type = _normalize_text(garment_type).lower()
    if normalized_garment_type not in {"", "top", "bottom", "dress", "outer"}:
        raise HTTPException(status_code=422, detail="garment_type must be one of: top, bottom, dress, outer")

    override_text = _normalize_text(prompt_override)
    default_instruction = _build_default_instruction(
        mode=normalized_mode,
        garment_type=(normalized_garment_type or None),
    )
    instruction_to_send = override_text or default_instruction
    prompt_override_applied = bool(override_text)

    source = _load_uploaded_image(upload, field_name="image").convert("RGB")
    input_size = {"width": int(source.width), "height": int(source.height)}
    runner = _resolve_minicpm_runner(request)

    def _run_blocking() -> Dict[str, object]:
        all_runs: List[Dict[str, object]] = []
        t0 = time.perf_counter()
        for idx in range(1, int(runs) + 1):
            run_started = time.perf_counter()
            if normalized_mode == "person_outfit":
                output_text = str(
                    runner.describe_person_and_outfit(
                        source,
                        prompt_override=(instruction_to_send if prompt_override_applied else None),
                    )
                ).strip()
            else:
                output_text = str(
                    runner.describe_garment(
                        source,
                        garment_type=(normalized_garment_type or None),
                        prompt_override=(instruction_to_send if prompt_override_applied else None),
                    )
                ).strip()
            elapsed = round(float(time.perf_counter() - run_started), 3)
            all_runs.append(
                {
                    "index": int(idx),
                    "text": output_text,
                    "word_count": int(_word_count(output_text)),
                    "elapsed_seconds": elapsed,
                    "quality_signals": _quality_signals(output_text),
                }
            )

        total_elapsed = round(float(time.perf_counter() - t0), 3)
        avg_elapsed = round(
            float(sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in all_runs) / max(1, len(all_runs))),
            3,
        )
        unique_outputs = len({str(item.get("text") or "").strip() for item in all_runs})

        runner_dtype = str(getattr(runner, "torch_dtype", "")).replace("torch.", "")
        model_max_new_tokens = (
            int(getattr(runner, "garment_max_new_tokens", 0))
            if normalized_mode == "garment"
            else int(getattr(runner, "user_max_new_tokens", 0))
        )
        return {
            "mode": normalized_mode,
            "garment_type": normalized_garment_type,
            "prompt_override_applied": prompt_override_applied,
            "payload_sent_to_minicpm": {
                "instruction": instruction_to_send,
                "max_new_tokens": model_max_new_tokens,
                "sampling": False,
                "temperature": 0.0,
                "input_size": input_size,
            },
            "runs": all_runs,
            "summary": {
                "runs": int(len(all_runs)),
                "unique_outputs": int(unique_outputs),
                "avg_elapsed_seconds": avg_elapsed,
                "total_elapsed_seconds": total_elapsed,
            },
            "runner": {
                "model_id": str(getattr(runner, "model_id", "") or ""),
                "device": str(getattr(runner, "device", "") or ""),
                "dtype": runner_dtype,
                "is_loaded": bool(getattr(runner, "is_loaded", False)),
                "garment_max_new_tokens": int(getattr(runner, "garment_max_new_tokens", 0) or 0),
                "user_max_new_tokens": int(getattr(runner, "user_max_new_tokens", 0) or 0),
                "garment_min_words": int(getattr(runner, "garment_min_words", 0) or 0),
            },
        }

    try:
        data = await asyncio.to_thread(_run_blocking)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("MiniCPM garment lab run failed")
        raise HTTPException(status_code=500, detail=f"MiniCPM run failed: {exc}") from exc

    data["input_image_base64"] = ""
    try:
        preview_buf = io.BytesIO()
        source.save(preview_buf, format="PNG")
        data["input_image_base64"] = base64.b64encode(preview_buf.getvalue()).decode("ascii")
    except Exception:
        pass

    return {
        "status": "success",
        "message": "MiniCPM garment lab run completed",
        "data": data,
    }

