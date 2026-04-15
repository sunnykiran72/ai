"""
Dev upscaler benchmark routes.

Provides a small API for side-by-side upscaler timing/output checks.
"""

import io
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.responses import FileResponse, HTMLResponse

from core.artisan_upscaler_runner import ArtisanUpscalerRunner
from routes.models import SuccessResponse
from services import AIEngine

logger = logging.getLogger("glamify-ai")
router = APIRouter()

_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "upscaler_api_tests"
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
_ARTISAN_RUNNER: Optional[ArtisanUpscalerRunner] = None


def get_ai_engine() -> AIEngine:
    """Dependency to get AIEngine instance."""
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_ai_engine()


def get_artisan_runner() -> ArtisanUpscalerRunner:
    global _ARTISAN_RUNNER
    if _ARTISAN_RUNNER is None:
        _ARTISAN_RUNNER = ArtisanUpscalerRunner()
    return _ARTISAN_RUNNER


def _json_float_list(values: List[float]) -> List[float]:
    return [float(round(v, 6)) for v in values]


@router.get("/v1/dev/upscaler-test/output/{job_id}/{filename}")
async def get_upscaler_test_output(job_id: str, filename: str):
    """Serve generated benchmark output images."""
    safe_job = (job_id or "").strip()
    safe_file = Path(filename).name
    target = (_OUTPUT_ROOT / safe_job / safe_file).resolve()
    if not str(target).startswith(str(_OUTPUT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid output path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(str(target), media_type="image/png", filename=safe_file)


@router.post("/v1/dev/upscaler-test/compare", response_model=SuccessResponse)
async def run_upscaler_compare(
    request: Request,
    file: UploadFile = File(...),
    run_artisan: bool = Form(True),
    run_realesrgan: bool = Form(False),
    repeats: int = Form(1),
    max_output_edge: int = Form(1024),
    artisan_precision: str = Form("fp32"),
    artisan_compile: bool = Form(False),
    artisan_tile: int = Form(128),
    artisan_overlap: int = Form(32),
    realesrgan_target_max_edge: int = Form(0),
    ai_engine: AIEngine = Depends(get_ai_engine),
) -> SuccessResponse:
    """Run upscaler quality checks and return timings/output URLs."""
    if not run_artisan and not run_realesrgan:
        raise HTTPException(status_code=422, detail="Enable at least one model: run_artisan or run_realesrgan")

    repeat_count = max(1, min(int(repeats), 20))
    output_edge = max(0, int(max_output_edge))
    artisan_precision = str(artisan_precision or "bf16").strip().lower()
    if artisan_precision not in {"bf16", "fp32"}:
        raise HTTPException(status_code=422, detail="artisan_precision must be 'bf16' or 'fp32'")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded image is empty")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Invalid image file") from exc

    job_id = uuid.uuid4().hex
    job_dir = _OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.png"
    image.save(input_path, format="PNG")

    results: Dict[str, Any] = {}

    if run_realesrgan:
        realesrgan = getattr(ai_engine, "realesrgan", None)
        if realesrgan is None or not realesrgan.is_available:
            results["realesrgan"] = {
                "enabled": True,
                "available": False,
                "error": "RealESRGAN dependencies are unavailable.",
            }
        else:
            load_seconds = 0.0
            if not realesrgan.is_loaded:
                t0 = time.perf_counter()
                realesrgan.ensure_ready()
                load_seconds = time.perf_counter() - t0

            infer_seconds: List[float] = []
            output_image: Optional[Image.Image] = None
            target_edge = int(realesrgan_target_max_edge) if int(realesrgan_target_max_edge) > 0 else None

            for _ in range(repeat_count):
                t0 = time.perf_counter()
                candidate = realesrgan.upscale(image, target_max_edge=target_edge)
                elapsed = time.perf_counter() - t0
                infer_seconds.append(float(elapsed))
                output_image = candidate.convert("RGB")

            assert output_image is not None
            if output_edge > 0 and max(output_image.size) > output_edge:
                scale = float(output_edge) / float(max(output_image.size))
                output_image = output_image.resize(
                    (
                        max(1, int(round(output_image.width * scale))),
                        max(1, int(round(output_image.height * scale))),
                    ),
                    getattr(getattr(Image, "Resampling", Image), "LANCZOS"),
                )

            out_path = job_dir / "realesrgan.png"
            output_image.save(out_path, format="PNG")
            out_url = str(request.url_for("get_upscaler_test_output", job_id=job_id, filename=out_path.name))
            results["realesrgan"] = {
                "enabled": True,
                "available": True,
                "model_name": str(getattr(realesrgan, "model_name", "RealESRGAN_x4plus")),
                "load_seconds": float(round(load_seconds, 6)),
                "repeat_count": repeat_count,
                "inference_seconds": _json_float_list(infer_seconds),
                "avg_inference_seconds": float(round(sum(infer_seconds) / len(infer_seconds), 6)),
                "output_width": int(output_image.width),
                "output_height": int(output_image.height),
                "output_path": str(out_path),
                "output_url": out_url,
            }

    if run_artisan:
        artisan = get_artisan_runner()
        prepare_info = artisan.ensure_ready(
            precision=artisan_precision,
            compile_model=bool(artisan_compile),
        )

        infer_seconds = []
        output_image = None
        for _ in range(repeat_count):
            candidate, elapsed = artisan.upscale(
                image,
                tile_size=max(16, int(artisan_tile)),
                overlap=max(0, int(artisan_overlap)),
            )
            infer_seconds.append(float(elapsed))
            output_image = candidate

        assert output_image is not None
        output_image = artisan.resize_to_max_edge(output_image, output_edge if output_edge > 0 else None)

        out_path = job_dir / "artisan.png"
        output_image.save(out_path, format="PNG")
        out_url = str(request.url_for("get_upscaler_test_output", job_id=job_id, filename=out_path.name))

        results["artisan"] = {
            "enabled": True,
            "available": True,
            "model_name": "ArtisanLabs/artisan-upscaler",
            "precision": artisan_precision,
            "compiled": bool(float(prepare_info.get("compiled", 0.0)) > 0.0),
            "tile": int(max(16, int(artisan_tile))),
            "overlap": int(max(0, int(artisan_overlap))),
            "load_seconds": float(round(float(prepare_info.get("load_seconds", 0.0)), 6)),
            "compile_seconds": float(round(float(prepare_info.get("compile_seconds", 0.0)), 6)),
            "repeat_count": repeat_count,
            "inference_seconds": _json_float_list(infer_seconds),
            "avg_inference_seconds": float(round(sum(infer_seconds) / len(infer_seconds), 6)),
            "output_width": int(output_image.width),
            "output_height": int(output_image.height),
            "output_path": str(out_path),
            "output_url": out_url,
        }

    return SuccessResponse(
        status="success",
        message="Upscaler comparison completed",
        data={
            "job_id": job_id,
            "input": {
                "filename": str(file.filename or ""),
                "width": int(image.width),
                "height": int(image.height),
                "input_path": str(input_path),
            },
            "settings": {
                "run_artisan": bool(run_artisan),
                "run_realesrgan": bool(run_realesrgan),
                "repeats": repeat_count,
                "max_output_edge": output_edge,
            },
            "results": results,
        },
    )


@router.get("/dev/upscaler-test-lab", response_class=HTMLResponse)
async def upscaler_test_lab_page() -> HTMLResponse:
    """Simple visual lab focused on Artisan upscaler quality checks."""
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Upscaler Compare Lab</title>
  <style>
    :root {
      --bg: #eef3f8;
      --ink: #112433;
      --muted: #506273;
      --card: #fff;
      --line: #d0dbe7;
      --accent: #0f766e;
      --accent2: #155e75;
      --err: #a11a3a;
      --ok: #1f7a4f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk","Manrope","Avenir Next",sans-serif;
      color: var(--ink);
      background:
        radial-gradient(900px 450px at 20% -15%, #d5e9ff 0, transparent 60%),
        radial-gradient(900px 450px at 85% -20%, #d5f7ea 0, transparent 60%),
        linear-gradient(180deg, #f4f8fb, var(--bg));
      min-height: 100vh;
    }
    .wrap { max-width: 1260px; margin: 16px auto; padding: 0 12px 24px; }
    .title { margin: 8px 0 10px; font-size: 30px; }
    .sub { margin: 0 0 14px; color: var(--muted); font-size: 14px; }
    .layout { display: grid; grid-template-columns: 360px 1fr; gap: 12px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(17, 36, 51, .07);
      padding: 12px;
    }
    label {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .4px;
      text-transform: uppercase;
    }
    input[type="file"], input[type="text"], input[type="number"], select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 14px;
      margin-bottom: 10px;
      color: var(--ink);
      background: #fff;
    }
    .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .btn {
      border: 0;
      border-radius: 10px;
      color: #fff;
      font-weight: 700;
      padding: 10px 14px;
      cursor: pointer;
      font-size: 14px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
    }
    .btn:disabled { opacity: .62; cursor: wait; }
    .status { margin-top: 10px; min-height: 20px; font-size: 13px; color: var(--muted); }
    .status.ok { color: var(--ok); }
    .status.err { color: var(--err); font-weight: 700; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
    .tile {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f7fafd;
      padding: 8px;
    }
    .tile h4 {
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: .4px;
      text-transform: uppercase;
    }
    .tile img {
      width: 100%;
      height: 360px;
      object-fit: contain;
      border: 1px solid #e5edf5;
      border-radius: 8px;
      background: #fff;
      display: block;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4,minmax(120px,1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #f9fbfd;
    }
    .k {
      font-size: 11px;
      color: var(--muted);
      letter-spacing: .4px;
      text-transform: uppercase;
      margin-bottom: 2px;
    }
    .v { font-weight: 700; font-size: 14px; overflow-wrap: anywhere; }
    pre {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #0f172a;
      color: #e2e8f0;
      padding: 10px;
      font-size: 12px;
      line-height: 1.35;
      overflow: auto;
      min-height: 180px;
      max-height: 420px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .grid2 { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2,minmax(120px,1fr)); }
      .tile img { height: 300px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Upscaler Quality Lab</h1>
    <p class="sub">Quick visual test focused on the new Artisan upscaler output.</p>
    <div class="layout">
      <section class="card">
        <label for="apiBase">API Base</label>
        <input id="apiBase" type="text" value="" placeholder="Leave blank for same origin" />

        <label for="fileInput">Input Image</label>
        <input id="fileInput" type="file" accept="image/*" />

        <div class="row2">
          <div>
            <label for="maxEdge">Max Output Edge</label>
            <input id="maxEdge" type="number" min="512" max="4096" step="1" value="2048" />
          </div>
          <div>
            <label for="repeats">Repeats</label>
            <input id="repeats" type="number" min="1" max="5" step="1" value="1" />
          </div>
        </div>

        <div class="row2">
          <div>
            <label for="artisanPrecision">Artisan Precision</label>
            <select id="artisanPrecision">
              <option value="fp32" selected>fp32 (default)</option>
              <option value="bf16">bf16</option>
            </select>
          </div>
          <div>
            <label for="artisanCompile">Artisan Compile</label>
            <select id="artisanCompile">
              <option value="false" selected>false</option>
              <option value="true">true</option>
            </select>
          </div>
        </div>

        <button id="runBtn" class="btn">Run Artisan</button>
        <div id="status" class="status">Ready.</div>
      </section>

      <section class="card">
        <div class="metrics">
          <div class="metric"><div class="k">Artisan Avg (s)</div><div id="mArtisan" class="v">-</div></div>
          <div class="metric"><div class="k">Artisan Size</div><div id="mArtisanSize" class="v">-</div></div>
          <div class="metric"><div class="k">Job ID</div><div id="mJobId" class="v">-</div></div>
          <div class="metric"><div class="k">Repeats</div><div id="mRepeats" class="v">-</div></div>
        </div>
        <div class="grid2">
          <div class="tile">
            <h4>Input</h4>
            <img id="imgInput" alt="input" />
          </div>
          <div class="tile">
            <h4>Artisan</h4>
            <img id="imgArtisan" alt="artisan" />
          </div>
        </div>
        <h4 style="margin:0 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;font-size:12px;">Raw JSON</h4>
        <pre id="jsonOut">No response yet.</pre>
      </section>
    </div>
  </div>
  <script>
    const fileInput = document.getElementById("fileInput");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const jsonOut = document.getElementById("jsonOut");
    const imgInput = document.getElementById("imgInput");
    const imgArtisan = document.getElementById("imgArtisan");
    const apiBaseEl = document.getElementById("apiBase");
    const maxEdgeEl = document.getElementById("maxEdge");
    const repeatsEl = document.getElementById("repeats");
    const artisanPrecisionEl = document.getElementById("artisanPrecision");
    const artisanCompileEl = document.getElementById("artisanCompile");

    function setStatus(text, kind) {
      statusEl.textContent = text;
      statusEl.className = `status ${kind || ""}`.trim();
    }
    function setMetric(id, val) {
      const n = document.getElementById(id);
      if (n) n.textContent = val;
    }
    function safe(v, fb="-") {
      return (v === undefined || v === null || v === "") ? fb : String(v);
    }
    function parseBool(s) {
      return String(s).toLowerCase() === "true";
    }

    fileInput.addEventListener("change", () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;
      imgInput.src = URL.createObjectURL(f);
    });

    runBtn.addEventListener("click", async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        setStatus("Choose an image first.", "err");
        return;
      }
      runBtn.disabled = true;
      setStatus("Running Artisan...", "");
      const fd = new FormData();
      fd.append("file", file);
      fd.append("run_artisan", "true");
      fd.append("run_realesrgan", "false");
      fd.append("repeats", String(parseInt(repeatsEl.value || "1", 10)));
      fd.append("max_output_edge", String(parseInt(maxEdgeEl.value || "2048", 10)));
      fd.append("artisan_precision", artisanPrecisionEl.value || "bf16");
      fd.append("artisan_compile", String(parseBool(artisanCompileEl.value || "false")));
      fd.append("artisan_tile", "128");
      fd.append("artisan_overlap", "32");

      const base = (apiBaseEl.value || "").trim();
      const url = `${base}/v1/dev/upscaler-test/compare`;
      try {
        const res = await fetch(url, { method: "POST", body: fd });
        const data = await res.json().catch(() => ({}));
        jsonOut.textContent = JSON.stringify(data, null, 2);
        if (!res.ok) {
          setStatus(`Failed (${res.status})`, "err");
          return;
        }
        setStatus("Success", "ok");
        const payload = (data && data.data) || {};
        const results = payload.results || {};
        const artisan = results.artisan || {};
        setMetric("mArtisan", safe(artisan.avg_inference_seconds));
        setMetric("mArtisanSize", `${safe(artisan.output_width,"?")}x${safe(artisan.output_height,"?")}`);
        setMetric("mJobId", safe(payload.job_id));
        setMetric("mRepeats", safe(payload.settings && payload.settings.repeats));
        imgArtisan.src = safe(artisan.output_url, "");
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
