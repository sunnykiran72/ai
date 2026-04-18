"""
Dedicated API + dev lab routes for Qwen Image Edit + Extract-Outfit LoRA.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from core.qwen_extract_shared_runner import get_shared_qwen_extract_runner
from core.qwen_extract_outfit_service import (
    DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT,
    PROMPT_GENERATION_FAILED_CODE,
    PromptGenerationFailedError,
    build_qwen_extract_outfit_request,
    execute_qwen_extract_outfit_request,
)
from shared.azure_storage import storage

logger = logging.getLogger("glamify-ai")
router = APIRouter()

_MINICPM_RUNNER = None


def _get_runner():
    return get_shared_qwen_extract_runner()


def _get_minicpm_runner(request: Optional[Request] = None):
    global _MINICPM_RUNNER
    if _MINICPM_RUNNER is not None:
        return _MINICPM_RUNNER
    if request is not None:
        engine = getattr(getattr(request.app, "state", object()), "ai_engine", None)
        runner = getattr(engine, "minicpm", None)
        if runner is not None:
            _MINICPM_RUNNER = runner
            return _MINICPM_RUNNER
    return None


def preload_qwen_extract_outfit_runner(*, run_warmup: bool = True) -> None:
    """
    Optional startup preload hook for lower first-request latency.
    """
    runner = _get_runner()
    runner.ensure_ready()
    if run_warmup:
        runner.warmup()


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


def _resize_longest_edge_to(image: Image.Image, target_longest_edge: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    target = max(1, int(target_longest_edge))
    if longest == target:
        return rgb
    ratio = float(target) / float(max(1, longest))
    target_size = (
        max(1, int(round(width * ratio))),
        max(1, int(round(height * ratio))),
    )
    return rgb.resize(target_size, Image.Resampling.LANCZOS)


def _parse_aspect_ratio(raw_ratio: str) -> Tuple[int, int]:
    text = str(raw_ratio or "").strip().lower()
    normalized = text.replace("x", ":").replace("/", ":")
    parts = [part.strip() for part in normalized.split(":", maxsplit=1)]
    if len(parts) != 2:
        raise HTTPException(
            status_code=422,
            detail="aspect_ratio must be in '<w>:<h>' format (examples: 2:3, 1:1, 3:4).",
        )
    try:
        ratio_w = int(parts[0])
        ratio_h = int(parts[1])
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="aspect_ratio must contain integer values (example: 2:3).",
        ) from exc
    if ratio_w <= 0 or ratio_h <= 0:
        raise HTTPException(status_code=422, detail="aspect_ratio values must be > 0.")
    return ratio_w, ratio_h


def _resolve_output_size_from_ratio(*, output_max_edge: int, ratio_w: int, ratio_h: int) -> Tuple[int, int]:
    edge = int(output_max_edge)
    if ratio_w >= ratio_h:
        out_w = edge
        out_h = max(1, int(round(edge * (float(ratio_h) / float(ratio_w)))))
    else:
        out_h = edge
        out_w = max(1, int(round(edge * (float(ratio_w) / float(ratio_h)))))
    return int(out_w), int(out_h)


def _align_to_model_grid(value: int, *, base: int = 8) -> int:
    raw = max(int(value), int(base))
    return max(int(base), (raw // int(base)) * int(base))


def _run_qwen_extract_outfit(
    *,
    app_request: Optional[Request],
    upload: UploadFile,
    prompt: Optional[str],
    steps: int,
    seed: int,
    guidance_scale: Optional[float],
    guidance_scale_alias: Optional[float],
    negative_prompt: Optional[str],
    negative_prompt_alias: Optional[str],
    max_input_edge: int,
    max_input_edge_alias: Optional[int],
    output_max_edge: Optional[int],
    output_max_edge_alias: Optional[int],
    output_aspect_ratio: Optional[str],
    output_aspect_ratio_alias: Optional[str],
    upload_output: bool,
    include_base64: bool,
) -> dict:
    qwen_request = build_qwen_extract_outfit_request(
        prompt=prompt,
        steps=steps,
        seed=seed,
        guidance_scale=guidance_scale,
        guidance_scale_alias=guidance_scale_alias,
        negative_prompt=negative_prompt,
        negative_prompt_alias=negative_prompt_alias,
        max_input_edge=max_input_edge,
        max_input_edge_alias=max_input_edge_alias,
        output_max_edge=output_max_edge,
        output_max_edge_alias=output_max_edge_alias,
        output_aspect_ratio=output_aspect_ratio,
        output_aspect_ratio_alias=output_aspect_ratio_alias,
        upload_output=upload_output,
        include_base64=include_base64,
    )
    source = _load_uploaded_image(upload, field_name="image")
    return execute_qwen_extract_outfit_request(
        request=qwen_request,
        source_image=source,
        runner=_get_runner(),
        minicpm_runner=_get_minicpm_runner(app_request),
        upload_image_fn=storage.upload_image,
    )


@router.post("/v1/qwen/extract-outfit")
async def qwen_extract_outfit_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    prompt: str = Form(DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT),
    steps: int = Form(28, ge=4, le=80),
    seed: int = Form(42, ge=0, le=2147483647),
    guidance_scale: Optional[float] = Form(None, ge=0.0, le=20.0),
    guidanceScale: Optional[float] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    negativePrompt: Optional[str] = Form(None),
    max_input_edge: int = Form(1536, ge=512, le=4096),
    maxInputEdge: Optional[int] = Form(None),
    output_max_edge: Optional[int] = Form(None, ge=256, le=4096),
    outputMaxEdge: Optional[int] = Form(None),
    output_aspect_ratio: Optional[str] = Form(None),
    outputAspectRatio: Optional[str] = Form(None),
    upload_output: bool = Form(False),
    include_base64: bool = Form(False),
) -> dict:
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide one image using 'file' or 'image'.")

    try:
        def _run_qwen_extract_outfit_blocking() -> dict:
            return _run_qwen_extract_outfit(
                app_request=request,
                upload=upload,
                prompt=prompt,
                steps=steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_alias=guidanceScale,
                negative_prompt=negative_prompt,
                negative_prompt_alias=negativePrompt,
                max_input_edge=max_input_edge,
                max_input_edge_alias=maxInputEdge,
                output_max_edge=output_max_edge,
                output_max_edge_alias=outputMaxEdge,
                output_aspect_ratio=output_aspect_ratio,
                output_aspect_ratio_alias=outputAspectRatio,
                upload_output=upload_output,
                include_base64=include_base64,
            )

        data = await asyncio.to_thread(_run_qwen_extract_outfit_blocking)
    except HTTPException:
        raise
    except PromptGenerationFailedError as exc:
        logger.warning("Qwen extract-outfit prompt generation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "message": str(exc),
                "reason_codes": [PROMPT_GENERATION_FAILED_CODE],
            },
        ) from exc
    except Exception as exc:
        logger.exception("Qwen extract-outfit run failed")
        raise HTTPException(status_code=500, detail=f"Qwen extract-outfit failed: {exc}") from exc

    return {
        "status": "success",
        "message": "Qwen extract-outfit run completed",
        "data": data,
    }


@router.get("/dev/qwen/extract-outfit-lab", response_class=HTMLResponse)
async def qwen_extract_outfit_lab_page() -> HTMLResponse:
    default_prompt = DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT
    short_prompt = "Extract the clothing and create a flat mockup."
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qwen Extract-Outfit Lab</title>
  <style>
    :root {{
      --bg-0: #f3f4f6;
      --bg-1: #e5e7eb;
      --ink: #111827;
      --muted: #4b5563;
      --card: #ffffff;
      --line: #d1d5db;
      --accent: #0f766e;
      --accent-2: #155e75;
      --danger: #9f1239;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Space Grotesk", "Manrope", "Avenir Next", sans-serif;
      background:
        radial-gradient(1100px 600px at 15% -20%, #dbeafe 0%, transparent 65%),
        radial-gradient(900px 500px at 90% -10%, #d1fae5 0%, transparent 60%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1));
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1320px; margin: 20px auto; padding: 0 14px 24px; }}
    .title {{ font-size: 28px; margin: 8px 0 16px; }}
    .layout {{ display: grid; grid-template-columns: 440px 1fr; gap: 14px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
      padding: 14px;
    }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
    .row3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
    label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--muted);
      margin-bottom: 4px;
      font-weight: 700;
    }}
    input[type="number"], input[type="text"], textarea, input[type="file"] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      font-size: 14px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
    }}
    textarea {{ min-height: 140px; resize: vertical; line-height: 1.35; }}
    .check-row {{ display: flex; gap: 14px; margin-bottom: 10px; }}
    .check-row label {{ text-transform: none; letter-spacing: 0; display: flex; gap: 8px; align-items: center; margin: 0; }}
    .actions {{ display: flex; gap: 10px; align-items: center; margin-top: 8px; flex-wrap: wrap; }}
    .btn {{
      border: 0;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #fff;
      font-size: 14px;
      font-weight: 700;
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
    }}
    .btn:disabled {{ opacity: 0.65; cursor: wait; }}
    .btn.gray {{ background: #334155; }}
    .status {{ font-size: 13px; color: var(--muted); min-height: 20px; margin-top: 8px; }}
    .status.err {{ color: var(--danger); font-weight: 700; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }}
    .tile {{ border: 1px solid var(--line); border-radius: 10px; background: #f9fafb; padding: 8px; }}
    .tile h4 {{
      margin: 0 0 6px;
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.4px;
    }}
    .tile img {{
      width: 100%;
      height: 430px;
      object-fit: contain;
      background: #fff;
      border-radius: 8px;
      border: 1px solid #e5e7eb;
      display: block;
    }}
    .meta {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .prompt-box {{
      background: #f9fafb;
      border: 1px dashed #cbd5e1;
      border-radius: 10px;
      padding: 10px;
      font-size: 13px;
      line-height: 1.35;
      min-height: 70px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    pre {{
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
    }}
    @media (max-width: 1140px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .tile img {{ height: 300px; }}
      .row3 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Qwen Extract-Outfit Lab</h1>
    <div class="layout">
      <section class="card">
        <form id="lab-form">
          <div>
            <label>Input Image</label>
            <input id="image" name="image" type="file" accept="image/*" required />
          </div>
          <div class="row3">
            <div>
              <label>Steps</label>
              <input name="steps" type="number" min="4" max="80" step="1" value="8" required />
            </div>
            <div>
              <label>Seed</label>
              <input name="seed" type="number" min="0" max="2147483647" step="1" value="42" required />
            </div>
            <div>
              <label>Input Max Edge (required)</label>
              <input name="input_max_edge" type="number" min="256" max="4096" step="1" value="768" required />
            </div>
          </div>
          <div class="row">
            <div>
              <label>Output Max Edge (fallback)</label>
              <input name="output_max_edge" type="number" min="256" max="4096" step="1" value="768" required />
            </div>
            <div>
              <label>Aspect Ratio (fallback)</label>
              <input name="aspect_ratio" type="text" value="2:3" placeholder="2:3" required />
            </div>
          </div>
          <div class="row">
            <div>
              <label>Output Width (optional override)</label>
              <input name="output_width" type="number" min="256" max="4096" step="1" placeholder="e.g. 768" />
            </div>
            <div>
              <label>Output Height (optional override)</label>
              <input name="output_height" type="number" min="256" max="4096" step="1" placeholder="e.g. 1024" />
            </div>
          </div>
          <div class="row">
            <div>
              <label>Guidance Scale (required)</label>
              <input name="guidance_scale" type="number" min="0" max="20" step="0.1" value="1.0" required />
            </div>
            <div>
              <label>Negative Prompt (optional)</label>
              <input name="negative_prompt" type="text" value="" placeholder="" />
            </div>
          </div>
          <div>
            <label>Prompt</label>
            <textarea name="prompt" id="prompt" required>{default_prompt}</textarea>
          </div>
          <div class="check-row">
            <label><input type="checkbox" name="upload_output" /> Upload output to Azure</label>
            <label><input type="checkbox" name="include_base64" checked /> Include image in JSON</label>
          </div>
          <div class="actions">
            <button class="btn" id="run-btn" type="submit">Run Extraction</button>
            <button class="btn gray" id="long-prompt-btn" type="button">Use Long Prompt</button>
            <button class="btn gray" id="short-prompt-btn" type="button">Use HF Short Prompt</button>
          </div>
          <div class="status" id="status">Ready.</div>
        </form>
      </section>
      <section class="card">
        <div class="grid2">
          <div class="tile">
            <h4>Input</h4>
            <img id="img-input" alt="input preview" />
          </div>
          <div class="tile">
            <h4>Output</h4>
            <img id="img-output" alt="output preview" />
          </div>
        </div>
        <div class="meta">
          <div>
            <label>Prompt Sent To Qwen</label>
            <div id="prompt-used" class="prompt-box"></div>
          </div>
          <div>
            <label>Resolution & Latency Metrics</label>
            <div id="latency-breakdown" class="prompt-box"></div>
          </div>
          <div>
            <label>Metadata / JSON</label>
            <pre id="raw-json"></pre>
          </div>
        </div>
      </section>
    </div>
  </div>
  <script>
    const form = document.getElementById("lab-form");
    const statusEl = document.getElementById("status");
    const rawEl = document.getElementById("raw-json");
    const promptUsedEl = document.getElementById("prompt-used");
    const latencyEl = document.getElementById("latency-breakdown");
    const runBtn = document.getElementById("run-btn");
    const longPromptBtn = document.getElementById("long-prompt-btn");
    const shortPromptBtn = document.getElementById("short-prompt-btn");
    const imageInput = document.getElementById("image");
    const promptInput = document.getElementById("prompt");
    const imgInput = document.getElementById("img-input");
    const imgOutput = document.getElementById("img-output");
    const defaultPrompt = {default_prompt!r};
    const shortPrompt = {short_prompt!r};

    function setStatus(message, isError=false) {{
      statusEl.textContent = message;
      statusEl.className = isError ? "status err" : "status";
    }}

    function previewInput() {{
      if (!imageInput.files || !imageInput.files[0]) return;
      const url = URL.createObjectURL(imageInput.files[0]);
      imgInput.src = url;
    }}

    imageInput.addEventListener("change", previewInput);
    longPromptBtn.addEventListener("click", () => {{ promptInput.value = defaultPrompt; }});
    shortPromptBtn.addEventListener("click", () => {{ promptInput.value = shortPrompt; }});

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      if (!imageInput.files?.length) {{
        setStatus("Please upload an image.", true);
        return;
      }}

      const data = new FormData();
      data.append("image", imageInput.files[0]);
      data.append("prompt", form.querySelector('[name="prompt"]').value);
      data.append("steps", form.querySelector('[name="steps"]').value);
      data.append("seed", form.querySelector('[name="seed"]').value);
      data.append("input_max_edge", form.querySelector('[name="input_max_edge"]').value);
      data.append("output_max_edge", form.querySelector('[name="output_max_edge"]').value);
      data.append("aspect_ratio", form.querySelector('[name="aspect_ratio"]').value);
      const outputWidthValue = form.querySelector('[name="output_width"]').value;
      const outputHeightValue = form.querySelector('[name="output_height"]').value;
      if (outputWidthValue && outputWidthValue.trim()) {{
        data.append("output_width", outputWidthValue);
      }}
      if (outputHeightValue && outputHeightValue.trim()) {{
        data.append("output_height", outputHeightValue);
      }}
      data.append("guidance_scale", form.querySelector('[name="guidance_scale"]').value);
      const negativePromptValue = form.querySelector('[name="negative_prompt"]').value;
      if (negativePromptValue && negativePromptValue.trim()) {{
        data.append("negative_prompt", negativePromptValue);
      }}

      const uploadOutput = form.querySelector('[name="upload_output"]').checked;
      const includeBase64 = form.querySelector('[name="include_base64"]').checked;
      data.append("upload_output", uploadOutput ? "true" : "false");
      data.append("include_base64", includeBase64 ? "true" : "false");

      runBtn.disabled = true;
      setStatus("Running extraction...");
      rawEl.textContent = "";
      promptUsedEl.textContent = "";
      latencyEl.textContent = "";
      imgOutput.src = "";

      try {{
        const response = await fetch("/dev/qwen/extract-outfit-lab/run", {{
          method: "POST",
          body: data
        }});
        const payload = await response.json();
        if (!response.ok) {{
          const detail = payload?.detail || payload?.message || `HTTP ${{response.status}}`;
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }}

        const base64Image = payload?.data?.image_base64 || "";
        if (base64Image) {{
          imgOutput.src = `data:image/png;base64,${{base64Image}}`;
        }} else if (payload?.data?.output_url) {{
          imgOutput.src = payload.data.output_url;
        }}
        promptUsedEl.textContent = payload?.data?.metadata?.prompt || "(empty)";
        rawEl.textContent = JSON.stringify(payload, null, 2);
        const meta = payload?.data?.metadata || {{}};
        const qwenElapsed = meta?.timings?.qwen_elapsed_seconds ?? "n/a";
        const totalElapsed = meta?.timings?.total_elapsed_seconds ?? "n/a";
        const inputOriginal = meta?.input_original_size || {{}};
        const inputPreprocessed = meta?.input_preprocessed_size || {{}};
        const requestedOutput = meta?.requested_output_size || {{}};
        const alignedOutput = meta?.requested_output_size_aligned || {{}};
        const actualOutput = meta?.output_size || {{}};
        const resolutionMode = meta?.resolution_mode || "n/a";
        latencyEl.textContent =
          `Qwen: ${{qwenElapsed ?? "n/a"}} sec\\n` +
          `Total: ${{totalElapsed ?? "n/a"}} sec\\n` +
          `Resolution mode: ${{resolutionMode}}\\n` +
          `Input original: ${{inputOriginal.width ?? "?"}}x${{inputOriginal.height ?? "?"}}\\n` +
          `Input preprocessed: ${{inputPreprocessed.width ?? "?"}}x${{inputPreprocessed.height ?? "?"}}\\n` +
          `Output requested: ${{requestedOutput.width ?? "?"}}x${{requestedOutput.height ?? "?"}}\\n` +
          `Output aligned(/8): ${{alignedOutput.width ?? "?"}}x${{alignedOutput.height ?? "?"}}\\n` +
          `Output actual: ${{actualOutput.width ?? "?"}}x${{actualOutput.height ?? "?"}}`;
        setStatus(`Success. Total: ${{totalElapsed ?? "n/a"}} sec | Qwen: ${{qwenElapsed ?? "n/a"}} sec`);
      }} catch (error) {{
        setStatus(`Run failed: ${{error.message}}`, true);
      }} finally {{
        runBtn.disabled = false;
      }}
    }});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@router.post("/dev/qwen/extract-outfit-lab/run")
async def qwen_extract_outfit_lab_run(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    prompt: str = Form(...),
    steps: int = Form(..., ge=4, le=80),
    seed: int = Form(..., ge=0, le=2147483647),
    guidance_scale: float = Form(..., ge=0.0, le=20.0),
    negative_prompt: Optional[str] = Form(None),
    input_max_edge: int = Form(..., ge=256, le=4096),
    output_max_edge: Optional[int] = Form(None, ge=256, le=4096),
    aspect_ratio: Optional[str] = Form(None),
    output_width: Optional[int] = Form(None, ge=256, le=4096),
    output_height: Optional[int] = Form(None, ge=256, le=4096),
    upload_output: bool = Form(False),
    include_base64: bool = Form(True),
) -> dict:
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide one image using 'file' or 'image'.")
    try:
        normalized_prompt = " ".join(str(prompt or "").split()).strip()
        if not normalized_prompt:
            raise HTTPException(status_code=422, detail="prompt must not be empty.")

        source_image = _load_uploaded_image(upload, field_name="image").convert("RGB")
        original_width, original_height = source_image.size
        preprocessed = _resize_longest_edge_to(source_image, target_longest_edge=int(input_max_edge))
        preprocessed_width, preprocessed_height = preprocessed.size

        explicit_output_requested = output_width is not None or output_height is not None
        if explicit_output_requested:
            if output_width is None or output_height is None:
                raise HTTPException(
                    status_code=422,
                    detail="When using direct output size, provide both output_width and output_height.",
                )
            requested_width = int(output_width)
            requested_height = int(output_height)
            ratio_w, ratio_h = int(requested_width), int(requested_height)
            resolved_output_max_edge = max(int(requested_width), int(requested_height))
            resolution_mode = "direct_dimensions"
        else:
            if output_max_edge is None:
                raise HTTPException(
                    status_code=422,
                    detail="output_max_edge is required when output_width/output_height are not provided.",
                )
            if not str(aspect_ratio or "").strip():
                raise HTTPException(
                    status_code=422,
                    detail="aspect_ratio is required when output_width/output_height are not provided.",
                )
            ratio_w, ratio_h = _parse_aspect_ratio(str(aspect_ratio))
            requested_width, requested_height = _resolve_output_size_from_ratio(
                output_max_edge=int(output_max_edge),
                ratio_w=ratio_w,
                ratio_h=ratio_h,
            )
            resolved_output_max_edge = int(output_max_edge)
            resolution_mode = "max_edge_ratio"

        aligned_requested_width = _align_to_model_grid(requested_width)
        aligned_requested_height = _align_to_model_grid(requested_height)

        def _run_qwen_edit_blocking():
            return _get_runner().run_edit(
                preprocessed,
                prompt=normalized_prompt,
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                negative_prompt=str(negative_prompt or ""),
                seed=int(seed),
                output_width=int(requested_width),
                output_height=int(requested_height),
            )

        t0 = time.perf_counter()
        output_image, run_meta = await asyncio.to_thread(_run_qwen_edit_blocking)
        qwen_elapsed_seconds = round(float(time.perf_counter() - t0), 3)

        output_bytes = b""
        if bool(upload_output) or bool(include_base64):
            out_buf = io.BytesIO()
            output_image.save(out_buf, format="PNG")
            output_bytes = out_buf.getvalue()

        output_url = ""
        if bool(upload_output):
            if not output_bytes:
                out_buf = io.BytesIO()
                output_image.save(out_buf, format="PNG")
                output_bytes = out_buf.getvalue()
            output_url = str(
                storage.upload_image(
                    output_bytes,
                    filename=f"qwen-lab-{int(time.time() * 1000)}.png",
                    content_type="image/png",
                )
                or ""
            )

        image_base64 = ""
        if bool(include_base64):
            if not output_bytes:
                out_buf = io.BytesIO()
                output_image.save(out_buf, format="PNG")
                output_bytes = out_buf.getvalue()
            image_base64 = base64.b64encode(output_bytes).decode("ascii")

        metadata = dict(run_meta or {})
        metadata.update(
            {
                "prompt": normalized_prompt,
                "input_max_edge": int(input_max_edge),
                "output_max_edge": int(resolved_output_max_edge),
                "aspect_ratio": f"{int(ratio_w)}:{int(ratio_h)}",
                "resolution_mode": str(resolution_mode),
                "direct_output_size_input": {
                    "width": int(output_width) if output_width is not None else None,
                    "height": int(output_height) if output_height is not None else None,
                },
                "input_original_size": {"width": int(original_width), "height": int(original_height)},
                "input_preprocessed_size": {
                    "width": int(preprocessed_width),
                    "height": int(preprocessed_height),
                },
                "requested_output_size": {
                    "width": int(requested_width),
                    "height": int(requested_height),
                },
                "requested_output_size_aligned": {
                    "width": int(aligned_requested_width),
                    "height": int(aligned_requested_height),
                },
                "output_size": {
                    "width": int(output_image.width),
                    "height": int(output_image.height),
                },
                "timings": {
                    "qwen_elapsed_seconds": qwen_elapsed_seconds,
                    "total_elapsed_seconds": qwen_elapsed_seconds,
                },
                "lab_mode": "strict_direct_qwen",
            }
        )

        data = {
            "output_url": output_url,
            "image_base64": image_base64,
            "metadata": metadata,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Qwen extract-outfit lab run failed")
        raise HTTPException(status_code=500, detail=f"Qwen extract-outfit failed: {exc}") from exc

    return {
        "status": "success",
        "message": "Qwen extract-outfit lab run completed",
        "data": data,
    }
