"""
User image preparation route handlers.

This module provides endpoints for preparing user images for try-on:
- /v1/user-image/prepare: Main user preparation endpoint
- /v1/flux2/prepare-user-image: Alternative endpoint

Delegates business logic to UserImageService.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from starlette.responses import Response
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from .models import UserPrepRequest, UserPrepResponse
from services import UserImageService
from shared.response_payloads import json_response
from utils.user_preparation import prepare_user_image_pipeline

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def get_user_prep_service() -> UserImageService:
    """Dependency to get UserImageService instance."""
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_user_prep_service()


@router.get("/dev/user-image/prepare-lab", response_class=HTMLResponse)
async def user_prep_lab_page() -> HTMLResponse:
    """Simple local UI to inspect /user-image/prepare behavior and metadata."""
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>User Prepare Lab</title>
  <style>
    :root {
      --bg: #eef4f8;
      --ink: #102a43;
      --muted: #486581;
      --card: #ffffff;
      --line: #cdd9e5;
      --accent: #0b7285;
      --accent-2: #1f7a8c;
      --err: #9b2226;
      --ok: #2d6a4f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Manrope", "Avenir Next", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1000px 500px at 20% -10%, #cfe8ff 0%, transparent 60%),
        radial-gradient(900px 420px at 85% 0%, #d7f7e7 0%, transparent 58%),
        linear-gradient(180deg, #f4f8fb 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1180px;
      margin: 18px auto;
      padding: 0 14px 24px;
    }
    .title {
      margin: 8px 0 14px;
      font-size: 29px;
      letter-spacing: .2px;
    }
    .sub {
      color: var(--muted);
      margin: 0 0 14px;
      font-size: 14px;
    }
    .layout {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 12px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(16, 42, 67, 0.07);
      padding: 12px;
    }
    label {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: .45px;
      text-transform: uppercase;
      font-weight: 700;
    }
    input[type="file"], select, input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 14px;
      color: var(--ink);
      background: #fff;
      margin-bottom: 10px;
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      color: var(--ink);
      margin: 8px 0 10px;
    }
    .btn {
      border: 0;
      border-radius: 10px;
      color: #fff;
      font-weight: 700;
      padding: 10px 14px;
      cursor: pointer;
      font-size: 14px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }
    .btn:disabled { opacity: .62; cursor: wait; }
    .status {
      margin-top: 10px;
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
    }
    .status.ok { color: var(--ok); }
    .status.err { color: var(--err); font-weight: 700; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #f9fbfd;
    }
    .metric .k {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .4px;
      margin-bottom: 2px;
    }
    .metric .v {
      font-weight: 700;
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }
    .tile {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f7fafc;
      padding: 8px;
    }
    .tile h4 {
      margin: 0 0 6px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .45px;
    }
    .tile img {
      width: 100%;
      height: 360px;
      object-fit: contain;
      border-radius: 8px;
      border: 1px solid #e5edf5;
      background: #fff;
      display: block;
    }
    pre {
      margin: 0;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #0f172a;
      color: #e2e8f0;
      padding: 10px;
      font-size: 12px;
      line-height: 1.38;
      overflow: auto;
      min-height: 220px;
      max-height: 420px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .grid2 { grid-template-columns: 1fr; }
      .tile img { height: 300px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">User Image Prepare Lab</h1>
    <p class="sub">Local tester for flow + stability metadata. Use probe mode to skip MiniCPM verifier dependency.</p>
    <div class="layout">
      <section class="card">
        <label for="apiBase">API Base</label>
        <input id="apiBase" type="text" value="" placeholder="Leave blank for same origin" />
        <label for="fileInput">Image</label>
        <input id="fileInput" type="file" accept="image/*" />
        <label for="resizeMethod">Resize Method</label>
        <select id="resizeMethod">
          <option value="">default</option>
          <option value="pyvips">pyvips</option>
          <option value="libvips">libvips</option>
          <option value="pillow_lanczos">pillow_lanczos</option>
        </select>
        <label class="toggle"><input id="probeMode" type="checkbox" checked /> Probe Mode (skip MiniCPM verification)</label>
        <button id="runBtn" class="btn">Run Prepare</button>
        <div id="status" class="status">Ready.</div>
      </section>

      <section class="card">
        <div class="metrics">
          <div class="metric"><div class="k">Prepared Size</div><div id="mSize" class="v">-</div></div>
          <div class="metric"><div class="k">Passthrough</div><div id="mPass" class="v">-</div></div>
          <div class="metric"><div class="k">Resize Action</div><div id="mResize" class="v">-</div></div>
          <div class="metric"><div class="k">JPEG Reencoded</div><div id="mReenc" class="v">-</div></div>
        </div>
        <div class="grid2">
          <div class="tile">
            <h4>Input Preview</h4>
            <img id="inputPreview" alt="input preview" />
          </div>
          <div class="tile">
            <h4>Prepared URL Preview</h4>
            <img id="outputPreview" alt="output preview" />
          </div>
        </div>
        <h4 style="margin:0 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.45px;font-size:12px;">Raw Response</h4>
        <pre id="jsonOut">No response yet.</pre>
      </section>
    </div>
  </div>
  <script>
    const fileInput = document.getElementById("fileInput");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const jsonOut = document.getElementById("jsonOut");
    const inputPreview = document.getElementById("inputPreview");
    const outputPreview = document.getElementById("outputPreview");
    const apiBaseEl = document.getElementById("apiBase");
    const resizeMethodEl = document.getElementById("resizeMethod");
    const probeModeEl = document.getElementById("probeMode");

    function setStatus(text, kind) {
      statusEl.textContent = text;
      statusEl.className = `status ${kind || ""}`.trim();
    }
    function setMetric(id, value) {
      const node = document.getElementById(id);
      if (node) node.textContent = value;
    }
    function renderMeta(payload) {
      const data = payload && payload.data ? payload.data : payload;
      const meta = (data && data.meta) || {};
      const policy = meta.prepare_policy || {};
      const size = meta.prepared_image_size || {};
      setMetric("mSize", (size.width && size.height) ? `${size.width}x${size.height}` : "-");
      setMetric("mPass", String(policy.passthrough_used));
      setMetric("mResize", String(policy.resize_action || "-"));
      setMetric("mReenc", String(policy.jpeg_reencoded));
      const outUrl = data && data.url ? data.url : "";
      outputPreview.src = outUrl || "";
    }

    fileInput.addEventListener("change", () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;
      inputPreview.src = URL.createObjectURL(f);
    });

    runBtn.addEventListener("click", async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        setStatus("Choose an image first.", "err");
        return;
      }
      runBtn.disabled = true;
      setStatus("Uploading...", "");
      const form = new FormData();
      form.append("file", file);
      const method = (resizeMethodEl.value || "").trim();
      if (method) form.append("resize_method", method);
      const base = (apiBaseEl.value || "").trim();
      const useProbe = !!probeModeEl.checked;
      const endpoint = useProbe ? "/dev/user-image/prepare-probe" : "/v1/user-image/prepare";
      const url = `${base}${endpoint}`;
      try {
        const res = await fetch(url, { method: "POST", body: form });
        const data = await res.json().catch(() => ({}));
        jsonOut.textContent = JSON.stringify(data, null, 2);
        if (!res.ok) {
          setStatus(`Failed (${res.status})`, "err");
          renderMeta(data);
          return;
        }
        setStatus(`Success (${res.status})`, "ok");
        renderMeta(data);
      } catch (err) {
        setStatus(`Request error: ${err}`, "err");
        jsonOut.textContent = String(err);
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.post("/v1/user-image/prepare", response_model=UserPrepResponse)
@router.post("/v1/flux2/prepare-user-image", response_model=UserPrepResponse)
async def prepare_user_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    resize_method: Optional[str] = Form(None),
    resizeMethod: Optional[str] = Form(None),
    output_max_edge: Optional[int] = Form(None),
    outputMaxEdge: Optional[int] = Form(None),
    user_prep_service: UserImageService = Depends(get_user_prep_service)
) -> UserPrepResponse:
    """
    User image preparation endpoint.
    
    Prepares user images for virtual try-on by:
    1. Validating image quality and focus
    2. Detecting and cropping main person
    3. Removing background if required
    4. Generating user description
    5. Uploading processed image
    
    Args:
        file: Uploaded image file (primary)
        image: Uploaded image file (alternative)
        user_prep_service: Injected UserImageService instance
        
    Returns:
        UserPrepResponse with processed image URL and description
        
    Raises:
        HTTPException: For validation errors or service failures
    """
    try:
        # Validate upload
        upload = file or image
        if upload is None:
            raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")
        
        resize_method_value = resizeMethod if resizeMethod is not None else resize_method
        output_max_edge_value = outputMaxEdge if outputMaxEdge is not None else output_max_edge
        prep_request = UserPrepRequest(
            **(
                {
                    key: value
                    for key, value in {
                        "resizeMethod": resize_method_value,
                        "outputMaxEdge": output_max_edge_value,
                    }.items()
                    if value is not None
                }
            )
        )

        logger.info("User image preparation request")
        
        # Delegate to service layer
        result = await user_prep_service.prepare_user_image(
            upload=upload,
            resize_method=prep_request.resize_method,
            output_max_edge=prep_request.output_max_edge,
        )

        # Pass through legacy response objects
        if isinstance(result, Response):
            return result

        if isinstance(result, dict) and result.get("error"):
            status_code = int(result.get("status_code") or 422)
            detail = {
                "error": result.get("error"),
                "message": result.get("message") or result.get("error"),
                "meta": result.get("meta") or {},
            }
            raise HTTPException(status_code=status_code, detail=detail)

        # Legacy user-prep payloads return a top-level status code and should bypass Pydantic wrapping
        if isinstance(result, dict) and isinstance(result.get("status"), int):
            return json_response(result)
        
        return UserPrepResponse(
            status="success",
            message="User image prepared successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"User prep validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"User prep failed: {e}")
        raise HTTPException(status_code=500, detail=f"User preparation failed: {str(e)}")


@router.post("/dev/user-image/prepare-probe")
async def prepare_user_image_probe_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    resize_method: Optional[str] = Form(None),
    resizeMethod: Optional[str] = Form(None),
    user_prep_service: UserImageService = Depends(get_user_prep_service),
):
    """
    Dev probe endpoint for local validation of prepare flow without MiniCPM verifier.
    Keeps the same core image normalization and metadata outputs.
    """
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")

    resize_method_value = resizeMethod if resizeMethod is not None else resize_method
    prep_request = UserPrepRequest(
        **({"resizeMethod": resize_method_value} if resize_method_value is not None else {})
    )
    resolved_resize_method = prep_request.resize_method

    try:
        from ai import main as main_mod
    except ModuleNotFoundError:
        import main as main_mod

    grounding_dino = getattr(user_prep_service.engine, "grounding_dino", None)

    def _grounding_detector_fn(image_obj, prompts):
        if grounding_dino is None:
            return None
        return grounding_dino.detect(image_obj, prompts=list(prompts or []))

    result = await prepare_user_image_pipeline(
        upload,
        grounding_detector_fn=_grounding_detector_fn,
        verifier_fn=None,
        description_fn=lambda _image_obj: "",
        fallback_description_fn=lambda _image_obj: "A person with balanced body build.",
        prepared_image_fn=None,
        upload_fn=main_mod._upload_or_raise,
        min_input_height=int(user_prep_service.config.app.user_prep_min_input_height),
        target_height=int(user_prep_service.config.app.user_prep_target_height),
        keep_long_edge_min=int(user_prep_service.config.app.user_prep_keep_long_edge_min),
        output_max_long_edge=int(user_prep_service.config.app.user_prep_output_max_long_edge),
        output_max_bytes=int(user_prep_service.config.app.user_prep_output_max_bytes),
        jpeg_quality=int(user_prep_service.config.app.user_prep_jpeg_quality),
        jpeg_min_quality=int(user_prep_service.config.app.user_prep_jpeg_min_quality),
        resize_method=str(resolved_resize_method),
        blur_check_enabled=bool(user_prep_service.config.analyze.blur_check_enabled),
        blur_min_focus_score=float(user_prep_service.config.analyze.blur_min_focus_score),
        blur_focus_max_edge=int(user_prep_service.config.analyze.blur_focus_max_edge),
        verification_required=False,
    )

    if isinstance(result, dict) and result.get("error"):
        status_code = int(result.get("status_code") or 422)
        detail = {
            "error": result.get("error"),
            "message": result.get("message") or result.get("error"),
            "meta": result.get("meta") or {},
        }
        raise HTTPException(status_code=status_code, detail=detail)
    return {"status": "success", "message": "Probe prepare completed", "data": result}
