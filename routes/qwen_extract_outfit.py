"""
Dedicated API + dev lab routes for Qwen Image Edit + Extract-Outfit LoRA.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from core.qwen_image_edit_runner import QwenImageEditRunner
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

_RUNNER: Optional[QwenImageEditRunner] = None
_MINICPM_RUNNER = None


def _get_runner() -> QwenImageEditRunner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = QwenImageEditRunner()
    return _RUNNER


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
        data = _run_qwen_extract_outfit(
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
              <input name="steps" type="number" min="4" max="80" step="1" value="28" />
            </div>
            <div>
              <label>Seed</label>
              <input name="seed" type="number" min="0" max="2147483647" step="1" value="42" />
            </div>
            <div>
              <label>Max Input Edge</label>
              <input name="max_input_edge" type="number" min="512" max="4096" step="1" value="1536" />
            </div>
          </div>
          <div class="row">
            <div>
              <label>Max Output Edge (optional; blank = use Max Input Edge)</label>
              <input name="output_max_edge" type="text" value="" placeholder="512" />
            </div>
            <div>
              <label>Output Aspect Ratio (optional)</label>
              <input name="output_aspect_ratio" type="text" value="2:3" placeholder="2:3" />
            </div>
          </div>
          <div class="row">
            <div>
              <label>Guidance Scale (optional)</label>
              <input name="guidance_scale" type="text" value="" placeholder="leave blank for default" />
            </div>
            <div>
              <label>Negative Prompt (optional)</label>
              <input name="negative_prompt" type="text" value="" placeholder="blurry, low quality" />
            </div>
          </div>
          <div>
            <label>Prompt</label>
            <textarea name="prompt" id="prompt">{default_prompt}</textarea>
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
            <label>Generated Prompt (MiniCPM)</label>
            <div id="prompt-generated" class="prompt-box"></div>
          </div>
          <div>
            <label>Input Prompt Sent To Qwen</label>
            <div id="prompt-used" class="prompt-box"></div>
          </div>
          <div>
            <label>Latency Breakdown</label>
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
    const promptGeneratedEl = document.getElementById("prompt-generated");
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
      data.append("max_input_edge", form.querySelector('[name="max_input_edge"]').value);
      const outputEdgeValue = form.querySelector('[name="output_max_edge"]').value.trim();
      if (outputEdgeValue) data.append("output_max_edge", outputEdgeValue);
      const outputAspectRatioValue = form.querySelector('[name="output_aspect_ratio"]').value.trim();
      if (outputAspectRatioValue) data.append("output_aspect_ratio", outputAspectRatioValue);

      const guidanceValue = form.querySelector('[name="guidance_scale"]').value.trim();
      const negativeValue = form.querySelector('[name="negative_prompt"]').value.trim();
      if (guidanceValue) data.append("guidance_scale", guidanceValue);
      if (negativeValue) data.append("negative_prompt", negativeValue);

      const uploadOutput = form.querySelector('[name="upload_output"]').checked;
      const includeBase64 = form.querySelector('[name="include_base64"]').checked;
      data.append("upload_output", uploadOutput ? "true" : "false");
      data.append("include_base64", includeBase64 ? "true" : "false");

      runBtn.disabled = true;
      setStatus("Running extraction...");
      rawEl.textContent = "";
      promptUsedEl.textContent = "";
      promptGeneratedEl.textContent = "";
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
        }}
        promptGeneratedEl.textContent = payload?.data?.promptDescription || "(empty)";
        promptUsedEl.textContent = payload?.data?.metadata?.prompt || "(empty)";
        rawEl.textContent = JSON.stringify(payload, null, 2);
        const minicpmElapsed = payload?.data?.minicpmElapsedSeconds
          ?? payload?.data?.promptElapsedSeconds
          ?? payload?.data?.metadata?.minicpm_elapsed_seconds;
        const qwenElapsed = payload?.data?.qwenElapsedSeconds
          ?? payload?.data?.metadata?.qwen_elapsed_seconds
          ?? payload?.data?.metadata?.elapsed_seconds;
        const totalElapsed = payload?.data?.totalElapsedSeconds
          ?? payload?.data?.metadata?.total_elapsed_seconds;
        const promptSource = payload?.data?.promptDescriptionSource || "n/a";
        const fallbackUsed = payload?.data?.promptFallbackUsed ? "yes" : "no";
        latencyEl.textContent =
          `MiniCPM: ${{minicpmElapsed ?? "n/a"}} sec\\n` +
          `Qwen: ${{qwenElapsed ?? "n/a"}} sec\\n` +
          `Total: ${{totalElapsed ?? "n/a"}} sec\\n` +
          `Prompt source: ${{promptSource}}\\n` +
          `Fallback used: ${{fallbackUsed}}`;
        setStatus(`Success. Total: ${{totalElapsed ?? "n/a"}} sec | MiniCPM: ${{minicpmElapsed ?? "n/a"}} sec | Qwen: ${{qwenElapsed ?? "n/a"}} sec`);
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
    include_base64: bool = Form(True),
) -> dict:
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide one image using 'file' or 'image'.")
    try:
        data = _run_qwen_extract_outfit(
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
    except HTTPException:
        raise
    except PromptGenerationFailedError as exc:
        logger.warning("Qwen extract-outfit lab prompt generation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "message": str(exc),
                "reason_codes": [PROMPT_GENERATION_FAILED_CODE],
            },
        ) from exc
    except Exception as exc:
        logger.exception("Qwen extract-outfit lab run failed")
        raise HTTPException(status_code=500, detail=f"Qwen extract-outfit failed: {exc}") from exc

    return {
        "status": "success",
        "message": "Qwen extract-outfit lab run completed",
        "data": data,
    }
