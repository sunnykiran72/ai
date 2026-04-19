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
    <title>Analyze API Full Flow Lab</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 14px; color: #0f172a; background: #f8fafc; }
      .grid { display: grid; grid-template-columns: 340px 1fr 1fr; gap: 10px; align-items: start; }
      .card { border: 1px solid #d0d7de; border-radius: 10px; padding: 10px; background: #fff; }
      .title { font-size: 12px; font-weight: 800; margin-bottom: 8px; color: #0f172a; text-transform: uppercase; letter-spacing: .04em; }
      label { display: block; font-size: 11px; font-weight: 700; margin: 8px 0 4px; color: #334155; }
      input[type=file], input[type=text], select, button {
        width: 100%; box-sizing: border-box; font-size: 13px; padding: 8px; border-radius: 8px; border: 1px solid #c9d1d9; background: #fff;
      }
      button { cursor: pointer; font-weight: 700; background: #0f172a; color: #fff; border: none; margin-top: 10px; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      img { width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; min-height: 240px; object-fit: contain; }
      .status { font-size: 12px; color: #334155; margin-top: 8px; min-height: 18px; }
      .flow-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
      pre {
        margin: 0;
        background: #0b1020;
        color: #dbeafe;
        padding: 10px;
        border-radius: 8px;
        overflow: auto;
        max-height: 280px;
        font-size: 11px;
        line-height: 1.4;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .row { margin-top: 10px; }
      .muted { font-size: 11px; color: #64748b; }
    </style>
  </head>
  <body>
    <h2 style="margin:0 0 8px 0;">Analyze API Full Flow Lab</h2>
    <div class="muted">Upload image + type, then inspect full flow trace: selection, prompt sent, MiniCPM state, sizes, timings, output.</div>

    <div class="grid row">
      <div class="card">
        <div class="title">Input</div>
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
        <div class="title">Input Preview</div>
        <img id="inputPreview" alt="input preview" />
      </div>

      <div class="card">
        <div class="title">Output Preview</div>
        <img id="outputPreview" alt="output preview" />
      </div>
    </div>

    <div class="flow-grid">
      <div class="card">
        <div class="title">Flow Summary</div>
        <pre id="summaryPre">No run yet.</pre>
      </div>
      <div class="card">
        <div class="title">Request + Selection</div>
        <pre id="selectionPre">No run yet.</pre>
      </div>
      <div class="card">
        <div class="title">Prompt Trace</div>
        <pre id="promptPre">No run yet.</pre>
      </div>
      <div class="card">
        <div class="title">Resolution Trace</div>
        <pre id="resolutionPre">No run yet.</pre>
      </div>
      <div class="card">
        <div class="title">Timing Trace</div>
        <pre id="timingPre">No run yet.</pre>
      </div>
      <div class="card">
        <div class="title">MiniCPM + Qwen Flags</div>
        <pre id="flagsPre">No run yet.</pre>
      </div>
    </div>

    <div class="row card">
      <div class="title">Binary Parts</div>
      <pre id="partsPre">No run yet.</pre>
    </div>
    <div class="row card">
      <div class="title">Raw Response JSON</div>
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
      const rawJsonEl = document.getElementById("rawJson");
      const summaryPre = document.getElementById("summaryPre");
      const selectionPre = document.getElementById("selectionPre");
      const promptPre = document.getElementById("promptPre");
      const resolutionPre = document.getElementById("resolutionPre");
      const timingPre = document.getElementById("timingPre");
      const flagsPre = document.getElementById("flagsPre");
      const partsPre = document.getElementById("partsPre");

      baseInput.value = window.location.origin;

      function pp(obj) {
        try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
      }

      function resetPanels() {
        summaryPre.textContent = "Running...";
        selectionPre.textContent = "Running...";
        promptPre.textContent = "Running...";
        resolutionPre.textContent = "Running...";
        timingPre.textContent = "Running...";
        flagsPre.textContent = "Running...";
        partsPre.textContent = "Running...";
        rawJsonEl.textContent = "{}";
      }

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
        resetPanels();

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
          const payload = await resp.json();
          const t1 = performance.now();

          rawJsonEl.textContent = pp(payload);

          if (payload && payload.extracted_image_data_url) {
            outputPreview.src = payload.extracted_image_data_url;
          } else if (payload && payload.analyze_response && payload.analyze_response.data && payload.analyze_response.data.cloth_url) {
            outputPreview.src = payload.analyze_response.data.cloth_url;
          }

          const analyzeResp = payload && payload.analyze_response ? payload.analyze_response : {};
          const data = analyzeResp && analyzeResp.data ? analyzeResp.data : {};
          const selected = data.selected_item || {};
          const extraction = selected.extraction || {};
          const qwenDebug = data.qwen_debug || {};
          const lat = data.latencies || {};
          const stages = lat.stages || {};
          const flow = payload.flow || {};

          summaryPre.textContent = pp({
            http_status: resp.status,
            payload_status: analyzeResp.status,
            result: data.result,
            selected_type: data.selected_type,
            cloth_url: data.cloth_url,
            output_image_url: data.output_image_url,
            promptDescription: data.promptDescription,
            promptDescriptionSource: selected.promptDescriptionSource,
            round_trip_seconds_client: Number(((t1 - t0) / 1000).toFixed(3)),
            reason_codes: data.reason_codes || [],
          });

          selectionPre.textContent = pp({
            input: {
              filename: f.name,
              garment_type_sent: garmentType.value || "top",
              base_url: baseInput.value,
            },
            selected_item: {
              garment_id: selected.garment_id,
              type: selected.type,
              style: selected.style,
              confidence: selected.confidence,
              detector_label: selected.detector_label,
              bbox: selected.bbox,
              extract_crop_mode: selected.extract_crop_mode,
              extract_crop_bbox: selected.extract_crop_bbox,
              output_image_source: selected.output_image_source,
            },
            flow_request: flow.request || {},
          });

          promptPre.textContent = pp({
            prompt_sent_qwen: qwenDebug.prompt_sent || extraction.prompt,
            prompt_template: qwenDebug.prompt_template || extraction.prompt_template,
            prompt_source: qwenDebug.prompt_source || extraction.prompt_source,
            prompt_description: qwenDebug.prompt_description || data.promptDescription,
            prompt_description_source: qwenDebug.prompt_description_source || selected.promptDescriptionSource,
            descriptor_raw_text: extraction.descriptor_raw_text,
            base_garment_prompt: selected.baseGarmentPrompt || extraction.base_garment_prompt,
          });

          resolutionPre.textContent = pp({
            input_original_size: qwenDebug.input_original_size || extraction.input_original_size,
            input_preprocessed_size: qwenDebug.input_preprocessed_size || extraction.input_preprocessed_size,
            requested_output_size: qwenDebug.requested_output_size || extraction.requested_output_size,
            requested_output_size_aligned: qwenDebug.requested_output_size_aligned || extraction.requested_output_size_aligned,
            qwen_output_size_raw: qwenDebug.qwen_output_size_raw || extraction.output_size,
            final_output_size: qwenDebug.final_output_size || extraction.final_output_size || extraction.output_size,
            top_min_output_width_rule_applied: qwenDebug.top_min_output_width_rule_applied || extraction.top_min_output_width_rule_applied,
            top_min_output_width_rule: qwenDebug.top_min_output_width_rule || extraction.top_min_output_width_rule,
            analyze_postprocess_applied: qwenDebug.analyze_postprocess_applied,
            analyze_qwen_match_lab_output: qwenDebug.analyze_qwen_match_lab_output,
          });

          timingPre.textContent = pp({
            latencies_total: lat.total,
            gpu_queue_wait: lat.gpu_queue_wait,
            stage_timings: stages,
            qwen_elapsed_seconds: extraction.qwen_elapsed_seconds,
            prompt_elapsed_seconds: extraction.prompt_elapsed_seconds,
            total_elapsed_seconds_qwen_path: extraction.total_elapsed_seconds,
          });

          flagsPre.textContent = pp({
            minicpm_prompt_enabled: qwenDebug.minicpm_prompt_enabled || extraction.minicpm_prompt_enabled,
            minicpm_json_valid: qwenDebug.minicpm_json_valid || extraction.minicpm_json_valid,
            minicpm_json_fallback_used: qwenDebug.minicpm_json_fallback_used || extraction.minicpm_json_fallback_used,
            normalized_category_type: qwenDebug.normalized_category_type || extraction.normalized_category_type,
            guidance_scale: qwenDebug.guidance_scale || extraction.guidance_scale,
            steps: qwenDebug.steps || extraction.steps,
            seed: qwenDebug.seed || extraction.seed,
            device: extraction.device,
            dtype: extraction.dtype,
            lora_loaded: extraction.lora_loaded,
            grid_base: extraction.grid_base,
          });

          partsPre.textContent = pp({
            binary_parts: payload.binary_parts || [],
            flow_binary_parts: flow.binary_parts || [],
          });

          statusEl.textContent = "Done.";
        } catch (e) {
          statusEl.textContent = "Failed: " + (e && e.message ? e.message : String(e));
          summaryPre.textContent = "Failed.";
          selectionPre.textContent = "Failed.";
          promptPre.textContent = "Failed.";
          resolutionPre.textContent = "Failed.";
          timingPre.textContent = "Failed.";
          flagsPre.textContent = "Failed.";
          partsPre.textContent = "Failed.";
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
            analyze_data = (metadata.get("data") if isinstance(metadata, dict) else {}) or {}
            selected_item = analyze_data.get("selected_item") if isinstance(analyze_data, dict) else {}
            if not isinstance(selected_item, dict):
                selected_item = {}
            extraction = selected_item.get("extraction") if isinstance(selected_item, dict) else {}
            if not isinstance(extraction, dict):
                extraction = {}
            qwen_debug = analyze_data.get("qwen_debug") if isinstance(analyze_data, dict) else {}
            if not isinstance(qwen_debug, dict):
                qwen_debug = {}
            flow = {
                "request": {
                    "garment_type": str(garment_type or ""),
                    "debug": bool(debug),
                    "filename": str(getattr(upload, "filename", "") or ""),
                },
                "selection": {
                    "selected_type": analyze_data.get("selected_type"),
                    "selected_item_type": selected_item.get("type"),
                    "style": selected_item.get("style"),
                    "confidence": selected_item.get("confidence"),
                    "bbox": selected_item.get("bbox"),
                    "extract_crop_mode": selected_item.get("extract_crop_mode"),
                    "extract_crop_bbox": selected_item.get("extract_crop_bbox"),
                    "output_image_source": selected_item.get("output_image_source"),
                },
                "prompt": {
                    "prompt_sent": qwen_debug.get("prompt_sent") or extraction.get("prompt"),
                    "prompt_template": qwen_debug.get("prompt_template") or extraction.get("prompt_template"),
                    "prompt_source": qwen_debug.get("prompt_source") or extraction.get("prompt_source"),
                    "prompt_description": qwen_debug.get("prompt_description") or analyze_data.get("promptDescription"),
                    "prompt_description_source": qwen_debug.get("prompt_description_source")
                    or selected_item.get("promptDescriptionSource"),
                },
                "resolution": {
                    "input_original_size": qwen_debug.get("input_original_size")
                    or extraction.get("input_original_size"),
                    "input_preprocessed_size": qwen_debug.get("input_preprocessed_size")
                    or extraction.get("input_preprocessed_size"),
                    "requested_output_size": qwen_debug.get("requested_output_size")
                    or extraction.get("requested_output_size"),
                    "requested_output_size_aligned": qwen_debug.get("requested_output_size_aligned")
                    or extraction.get("requested_output_size_aligned"),
                    "qwen_output_size_raw": qwen_debug.get("qwen_output_size_raw") or extraction.get("output_size"),
                    "final_output_size": qwen_debug.get("final_output_size")
                    or extraction.get("final_output_size")
                    or extraction.get("output_size"),
                },
                "flags": {
                    "minicpm_prompt_enabled": qwen_debug.get("minicpm_prompt_enabled")
                    or extraction.get("minicpm_prompt_enabled"),
                    "minicpm_json_valid": qwen_debug.get("minicpm_json_valid") or extraction.get("minicpm_json_valid"),
                    "minicpm_json_fallback_used": qwen_debug.get("minicpm_json_fallback_used")
                    or extraction.get("minicpm_json_fallback_used"),
                    "normalized_category_type": qwen_debug.get("normalized_category_type")
                    or extraction.get("normalized_category_type"),
                    "analyze_postprocess_applied": qwen_debug.get("analyze_postprocess_applied"),
                    "analyze_qwen_match_lab_output": qwen_debug.get("analyze_qwen_match_lab_output"),
                },
                "timings": {
                    "latencies": analyze_data.get("latencies"),
                    "qwen_elapsed_seconds": extraction.get("qwen_elapsed_seconds"),
                    "prompt_elapsed_seconds": extraction.get("prompt_elapsed_seconds"),
                    "total_elapsed_seconds": extraction.get("total_elapsed_seconds"),
                },
                "binary_parts": part_summaries,
            }
            return JSONResponse(
                {
                    "status": "success",
                    "content_type": content_type,
                    "analyze_response": metadata,
                    "binary_parts": part_summaries,
                    "extracted_image_data_url": extracted_data_url,
                    "flow": flow,
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
