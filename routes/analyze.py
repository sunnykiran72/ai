"""
Garment analysis route handlers.

This module provides the main garment analysis endpoint:
- /analyze: Main garment analysis endpoint
- /analzye: Typo endpoint for backward compatibility

All endpoints delegate business logic to AnalyzeService.
"""

import base64
import json
import logging
from email.parser import BytesParser
from email.policy import default as email_policy_default
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import Response
from pydantic import ValidationError

from .models import AnalyzeResponse
from services import AnalyzeService
from shared.response_payloads import json_response

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def _parse_multipart_form_response(content_type: str, body: bytes) -> Dict[str, Any]:
    """
    Parse our analyze multipart response and extract:
    - metadata JSON part
    - binary parts list (including extracted_cloth image)
    """
    if not content_type or "multipart/form-data" not in content_type.lower():
        return {"metadata": None, "parts": []}
    try:
        mime = BytesParser(policy=email_policy_default).parsebytes(
            (
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
                + bytes(body or b"")
            )
        )
    except Exception:
        return {"metadata": None, "parts": []}

    if not mime.is_multipart():
        return {"metadata": None, "parts": []}

    parsed: Dict[str, Any] = {"metadata": None, "parts": []}
    for part in mime.iter_parts():
        name = part.get_param("name", header="Content-Disposition") or ""
        filename = part.get_filename()
        part_type = str(part.get_content_type() or "application/octet-stream")
        data = part.get_payload(decode=True) or b""
        if name == "metadata":
            try:
                parsed["metadata"] = json.loads(data.decode("utf-8", errors="ignore"))
            except Exception:
                parsed["metadata"] = {"raw": data.decode("utf-8", errors="ignore")}
            continue
        parsed["parts"].append(
            {
                "name": str(name),
                "filename": str(filename or ""),
                "content_type": part_type,
                "size": int(len(data)),
                "bytes": data,
            }
        )
    return parsed


def get_analyze_service() -> AnalyzeService:
    """Dependency to get AnalyzeService instance."""
    try:
        from ai import main as _main
    except (ModuleNotFoundError, ImportError):
        import main as _main
    return _main.get_analyze_service()


@router.post("/analyze", response_model=AnalyzeResponse)
@router.post("/analzye", response_model=AnalyzeResponse)  # Typo compatibility
async def analyze_garment_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form(None, alias="type"),
    debug: bool = Form(False),
    _authorization: Optional[str] = Header(None, alias="Authorization"),
    analyze_service: AnalyzeService = Depends(get_analyze_service)
) -> AnalyzeResponse:
    """
    Main garment analysis endpoint.
    
    Analyzes uploaded garment images to detect, extract, and digitize garments.
    
    Args:
        file: Uploaded image file (required)
        garment_type: Expected garment type (top, bottom, dress, outer) - optional
        debug: Enable debug mode
        _authorization: Authorization header
        analyze_service: Injected AnalyzeService instance
        
    Returns:
        AnalyzeResponse with garment analysis data
        
    Raises:
        HTTPException: For validation errors or service failures
    """
    try:
        upload = file or image
        if upload is None:
            raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")

        logger.info(f"Analyze request: type={garment_type}")
        
        # Delegate to service layer
        result = await analyze_service.analyze_image(
            upload=upload,
            garment_type=garment_type,
            debug=debug,
            authorization=_authorization,
        )
        
        # Pass through legacy response objects (multipart/form-data)
        if isinstance(result, Response):
            return result

        # Legacy analyze payloads return a top-level status code and should bypass Pydantic wrapping
        if isinstance(result, dict) and isinstance(result.get("status"), int):
            return json_response(result)

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Analyze service returned an invalid response type")
        
        return AnalyzeResponse(
            status="success",
            message="Garment analysis completed successfully",
            data=result
        )
        
    except ValidationError as e:
        logger.error(f"Analyze validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Analyze failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/dev/analyze-api-lab", response_class=HTMLResponse)
async def analyze_api_lab_page() -> HTMLResponse:
    html = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Analyze API Lab</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; color: #0f172a; }
      .grid { display: grid; grid-template-columns: 360px 1fr 1fr; gap: 12px; align-items: start; }
      .card { border: 1px solid #d0d7de; border-radius: 10px; padding: 12px; background: #fff; }
      .title { font-size: 13px; font-weight: 700; margin-bottom: 8px; }
      label { display: block; font-size: 12px; font-weight: 600; margin: 10px 0 6px; }
      input[type=file], input[type=text], select, button, textarea { width: 100%; box-sizing: border-box; font-size: 13px; padding: 8px; border-radius: 8px; border: 1px solid #c9d1d9; }
      button { cursor: pointer; font-weight: 700; background: #0f172a; color: #fff; border: none; margin-top: 12px; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      img { width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; min-height: 240px; object-fit: contain; }
      pre { background: #0b1020; color: #dbeafe; padding: 10px; border-radius: 8px; overflow: auto; max-height: 360px; font-size: 12px; }
      .metrics { font-size: 12px; line-height: 1.5; }
      .row { margin-top: 10px; }
      .status { font-size: 12px; color: #334155; margin-top: 8px; min-height: 18px; }
    </style>
  </head>
  <body>
    <h2 style="margin:0 0 10px 0;">Analyze API Lab</h2>
    <div class="grid">
      <div class="card">
        <div class="title">INPUT</div>
        <label>Base URL</label>
        <input id="baseUrl" type="text" />
        <label>Authorization (optional)</label>
        <input id="auth" type="text" placeholder="Bearer ..." />
        <label>Garment Type</label>
        <select id="garmentType">
          <option value="top" selected>top</option>
          <option value="bottom">bottom</option>
          <option value="dress">dress</option>
          <option value="outer">outer</option>
        </select>
        <label>Image</label>
        <input id="fileInput" type="file" accept="image/*" />
        <button id="runBtn" type="button">Run Analyze</button>
        <div id="status" class="status">Ready.</div>
      </div>

      <div class="card">
        <div class="title">INPUT PREVIEW</div>
        <img id="inputPreview" alt="input preview" />
      </div>

      <div class="card">
        <div class="title">OUTPUT PREVIEW</div>
        <img id="outputPreview" alt="output preview" />
      </div>
    </div>

    <div class="row card">
      <div class="title">METRICS</div>
      <div id="metrics" class="metrics">No run yet.</div>
    </div>
    <div class="row card">
      <div class="title">RAW RESPONSE (JSON)</div>
      <pre id="rawJson">{}</pre>
    </div>

    <script>
      const baseInput = document.getElementById("baseUrl");
      const authInput = document.getElementById("auth");
      const fileInput = document.getElementById("fileInput");
      const garmentType = document.getElementById("garmentType");
      const runBtn = document.getElementById("runBtn");
      const statusEl = document.getElementById("status");
      const inputPreview = document.getElementById("inputPreview");
      const outputPreview = document.getElementById("outputPreview");
      const metricsEl = document.getElementById("metrics");
      const rawJsonEl = document.getElementById("rawJson");

      baseInput.value = window.location.origin;

      fileInput.addEventListener("change", () => {
        const f = fileInput.files && fileInput.files[0];
        if (!f) return;
        inputPreview.src = URL.createObjectURL(f);
      });

      runBtn.addEventListener("click", async () => {
        const f = fileInput.files && fileInput.files[0];
        if (!f) {
          statusEl.textContent = "Select an image first.";
          return;
        }
        runBtn.disabled = true;
        statusEl.textContent = "Running analyze...";
        outputPreview.removeAttribute("src");
        metricsEl.textContent = "Running...";
        rawJsonEl.textContent = "{}";
        try {
          const fd = new FormData();
          fd.append("file", f);
          fd.append("type", garmentType.value || "top");
          const headers = {};
          const auth = (authInput.value || "").trim();
          if (auth) headers["Authorization"] = auth;
          const t0 = performance.now();
          const resp = await fetch(baseInput.value.replace(/\\/$/, "") + "/dev/analyze-api-lab/run", {
            method: "POST",
            body: fd,
            headers,
          });
          const data = await resp.json();
          const t1 = performance.now();
          rawJsonEl.textContent = JSON.stringify(data, null, 2);
          if (data && data.extracted_image_data_url) {
            outputPreview.src = data.extracted_image_data_url;
          } else if (data && data.analyze_response && data.analyze_response.data && data.analyze_response.data.cloth_url) {
            outputPreview.src = data.analyze_response.data.cloth_url;
          }
          const payload = (data && data.analyze_response && data.analyze_response.data) ? data.analyze_response.data : {};
          const lat = payload.latencies || {};
          const stage = lat.stages || {};
          metricsEl.innerHTML = [
            `HTTP: ${resp.status}`,
            `Result: ${payload.result || "-"}`,
            `Selected Type: ${payload.selected_type || "-"}`,
            `Total Latency: ${lat.total ?? "-"} sec`,
            `GPU Queue Wait: ${lat.gpu_queue_wait ?? "-"} sec`,
            `Prompt: ${(payload.promptDescription || "").slice(0, 180) || "-"}`,
            `Client Round Trip: ${((t1 - t0) / 1000).toFixed(3)} sec`,
            `Stages: ${JSON.stringify(stage)}`,
          ].join("<br/>");
          statusEl.textContent = "Done.";
        } catch (e) {
          statusEl.textContent = "Failed: " + (e && e.message ? e.message : String(e));
          metricsEl.textContent = "Failed.";
        } finally {
          runBtn.disabled = false;
        }
      });
    </script>
  </body>
</html>"""
    return HTMLResponse(content=html)


@router.post("/dev/analyze-api-lab/run")
async def analyze_api_lab_run(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    garment_type: Optional[str] = Form("top", alias="type"),
    debug: bool = Form(False),
    _authorization: Optional[str] = Header(None, alias="Authorization"),
    analyze_service: AnalyzeService = Depends(get_analyze_service),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Provide one image file using 'file' or 'image'")

    # Dev lab fallback token for easier local/pod testing.
    authorization = _authorization or "Bearer local-test-token"
    result = await analyze_service.analyze_image(
        upload=upload,
        garment_type=garment_type,
        debug=debug,
        authorization=authorization,
    )

    if isinstance(result, Response):
        content_type = str(result.headers.get("content-type", ""))
        body = bytes(getattr(result, "body", b"") or b"")
        if "multipart/form-data" in content_type.lower():
            parsed = _parse_multipart_form_response(content_type, body)
            metadata = parsed.get("metadata") if isinstance(parsed, dict) else None
            parts: List[Dict[str, Any]] = list(parsed.get("parts") or []) if isinstance(parsed, dict) else []
            extracted_data_url = ""
            part_summaries: List[Dict[str, Any]] = []
            for part in parts:
                part_summaries.append(
                    {
                        "name": part.get("name"),
                        "filename": part.get("filename"),
                        "content_type": part.get("content_type"),
                        "size": part.get("size"),
                    }
                )
                if part.get("name") == "extracted_cloth" and isinstance(part.get("bytes"), (bytes, bytearray)):
                    encoded = base64.b64encode(bytes(part["bytes"])).decode("ascii")
                    extracted_data_url = f"data:{part.get('content_type') or 'image/png'};base64,{encoded}"
            return JSONResponse(
                {
                    "status": "success",
                    "content_type": content_type,
                    "analyze_response": metadata,
                    "binary_parts": part_summaries,
                    "extracted_image_data_url": extracted_data_url,
                }
            )
        # Non-multipart fallback response
        try:
            parsed_json = json.loads(body.decode("utf-8", errors="ignore"))
        except Exception:
            parsed_json = {"raw_text": body.decode("utf-8", errors="ignore")}
        return JSONResponse({"status": "success", "content_type": content_type, "analyze_response": parsed_json})

    if isinstance(result, dict):
        return JSONResponse({"status": "success", "analyze_response": result})

    return JSONResponse({"status": "error", "message": "Unexpected analyze response type"}, status_code=500)
