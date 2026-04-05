"""
Developer-facing FLUX2 consistency try-on lab UI.

This module adds a lightweight, file-upload based test page for rapid
prompt/parameter iteration without changing the public production contract.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from services import AIEngine
from utils import build_tryon_prompt_v2

logger = logging.getLogger("glamify-ai")
router = APIRouter()

_VALID_GARMENT_TYPES = {"top", "bottom", "dress", "outer"}
_VALID_LORA_PROFILES = {"consistency", "tryon"}
_DEFAULT_CONSISTENCY_PROMPT = (
    "Image 1 is the person and scene anchor. Image 2 is the garment anchor. "
    "Keep the same face identity, skin tone, hair, body proportions, pose, hand placement, "
    "camera framing, background, and lighting from Image 1. "
    "Single-garment try-on target type: top. "
    "Apply the top garment from Image 2 on the upper clothing region from shoulders to waist. "
    "Keep waist-down clothing, legs, and footwear styling from Image 1 consistent. "
    "Render realistic fabric drape, natural fold direction, clean seam lines, and crisp garment edges. "
    "Maintain stable garment boundaries inside the intended edit region with natural transitions."
)
_PROFILE_DEFAULTS = {
    "consistency": {"steps": 8, "guidance_scale": 3.5, "lora_scale": 0.4},
    "tryon": {"steps": 28, "guidance_scale": 2.5, "lora_scale": 1.0},
}


def get_ai_engine() -> AIEngine:
    """Dependency to get AIEngine instance."""
    try:
        from ai import main as _main
    except ModuleNotFoundError:
        import main as _main
    return _main.get_ai_engine()


def _load_uploaded_image(file: UploadFile, *, field_name: str) -> Image.Image:
    try:
        raw = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read {field_name} upload: {exc}") from exc
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} upload is empty")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not a valid image: {exc}") from exc


def _parse_source_worn_types(raw: str) -> List[str]:
    values: List[str] = []
    for part in str(raw or "").split(","):
        kind = part.strip().lower()
        if not kind:
            continue
        if kind not in _VALID_GARMENT_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid source worn type '{kind}'. Allowed: top, bottom, dress, outer.",
            )
        if kind not in values:
            values.append(kind)
    return values


def _build_tryon_prompt(
    *,
    target_type: str,
    garment_description: str,
    user_description: str,
) -> str:
    base_prompt = build_tryon_prompt_v2(
        user_description=str(user_description or "").strip(),
        garment_descriptions=[str(garment_description or "").strip()],
        target_types=[str(target_type or "").strip().lower()],
        board_mode="single",
        lora_mode="tryon",
    )
    return (
        "TRYON full body fashion photo edit of the same person from image 1. "
        "Image 1 is the person reference. Image 2 is the single garment reference. "
        f"{base_prompt} "
        "Use the garment construction, neckline, shoulder line, sleeve or strap layout, hemline, and silhouette from image 2. "
        "Do not preserve the original garment geometry from image 1 inside the target clothing region."
    ).strip()


def _default_prompt_for_profile(
    *,
    profile: str,
    target_type: str,
    garment_description: str,
    user_description: str,
) -> str:
    normalized_profile = str(profile or "consistency").strip().lower()
    if normalized_profile == "tryon":
        return _build_tryon_prompt(
            target_type=target_type,
            garment_description=garment_description,
            user_description=user_description,
        )
    return _DEFAULT_CONSISTENCY_PROMPT


@router.get("/dev/flux2/consistency-lab", response_class=HTMLResponse)
async def consistency_lab_page() -> HTMLResponse:
    """Serve a standalone test page for rapid consistency-LoRA iteration."""
    default_tryon_prompt = _default_prompt_for_profile(
        profile="tryon",
        target_type="top",
        garment_description="asymmetrical top with sleeve detail",
        user_description="",
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Flux2 LoRA Try-on Lab</title>
  <style>
    :root {{
      --bg-0: #f4f7fb;
      --bg-1: #e7edf7;
      --ink: #102231;
      --muted: #526578;
      --card: #ffffff;
      --line: #d0d9e4;
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
        radial-gradient(1100px 600px at 15% -20%, #cde1ff 0%, transparent 65%),
        radial-gradient(900px 500px at 90% -10%, #c6f2e7 0%, transparent 60%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1));
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1320px;
      margin: 20px auto;
      padding: 0 14px 24px;
    }}
    .title {{
      font-size: 28px;
      letter-spacing: 0.2px;
      margin: 8px 0 16px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 430px 1fr;
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(16, 34, 49, 0.06);
      padding: 14px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .row3 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }}
    label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--muted);
      margin-bottom: 4px;
      font-weight: 700;
    }}
    input[type="number"], input[type="text"], select, textarea, input[type="file"] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      font-size: 14px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
    }}
    textarea {{
      min-height: 124px;
      resize: vertical;
      line-height: 1.35;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 8px;
    }}
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
    .status {{
      font-size: 13px;
      color: var(--muted);
      min-height: 20px;
    }}
    .status.err {{ color: var(--danger); font-weight: 700; }}
    .grid3 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .tile {{
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f7f9fc;
      padding: 8px;
    }}
    .tile h4 {{
      margin: 0 0 6px;
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.4px;
    }}
    .tile img {{
      width: 100%;
      height: 360px;
      object-fit: contain;
      background: #fff;
      border-radius: 8px;
      border: 1px solid #e6ecf3;
      display: block;
    }}
    .meta {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    pre {{
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 10px;
      padding: 10px;
      font-size: 12px;
      max-height: 300px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .prompt-box {{
      background: #f9fafb;
      border: 1px dashed #c8d2df;
      border-radius: 10px;
      padding: 10px;
      font-size: 13px;
      line-height: 1.35;
      min-height: 68px;
    }}
    @media (max-width: 1140px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .tile img {{ height: 280px; }}
      .row3 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Flux2 LoRA Try-on Lab</h1>
    <div class="layout">
      <section class="card">
        <form id="lab-form">
          <div class="row">
            <div>
              <label>User Image</label>
              <input id="user_image" name="user_image" type="file" accept="image/*" required />
            </div>
            <div>
              <label>Garment Image</label>
              <input id="garment_image" name="garment_image" type="file" accept="image/*" required />
            </div>
          </div>
          <div class="row3">
            <div>
              <label>LoRA Profile</label>
              <select name="lora_profile" id="lora_profile">
                <option value="consistency">consistency</option>
                <option value="tryon" selected>tryon</option>
              </select>
            </div>
            <div>
              <label>Steps</label>
              <input name="steps" type="number" min="4" max="50" step="1" value="8" />
            </div>
            <div>
              <label>Seed</label>
              <input name="seed" type="number" min="0" max="2147483647" step="1" value="42" />
            </div>
          </div>
          <div class="row">
            <div>
              <label>LoRA Scale</label>
              <input name="lora_scale" type="number" min="0" max="2" step="0.01" value="0.4" />
            </div>
            <div>
              <label>Guidance Scale</label>
              <input name="guidance_scale" type="number" min="0" max="20" step="0.1" value="3.5" />
            </div>
          </div>
          <div class="row">
            <div>
              <label>Target Type</label>
              <select name="target_type">
                <option value="top" selected>top</option>
                <option value="bottom">bottom</option>
                <option value="dress">dress</option>
                <option value="outer">outer</option>
              </select>
            </div>
            <div>
              <label>Source Worn Types (comma-separated)</label>
              <input name="source_worn_types" type="text" value="top,bottom" />
            </div>
          </div>
          <div class="row">
            <div>
              <label>Garment Description (optional)</label>
              <input name="garment_prompt_description" type="text" value="asymmetrical top with sleeve detail" />
            </div>
            <div>
              <label>User Description (optional)</label>
              <input name="user_prompt_description" type="text" value="" />
            </div>
          </div>
          <div>
            <label>Prompt Override (optional; if non-empty, this exact prompt is sent to FLUX)</label>
            <textarea name="prompt">{default_tryon_prompt}</textarea>
          </div>
          <div class="actions">
            <button class="btn" id="run-btn" type="submit">Run Try-on</button>
            <button class="btn" id="clear-btn" type="button" style="background:#334155">Reset Prompt</button>
          </div>
          <div class="status" id="status">Ready.</div>
        </form>
      </section>

      <section class="card">
        <div class="grid3">
          <div class="tile">
            <h4>Garment</h4>
            <img id="img-garment" alt="garment preview" />
          </div>
          <div class="tile">
            <h4>Try-on Output</h4>
            <img id="img-output" alt="output preview" />
          </div>
          <div class="tile">
            <h4>User</h4>
            <img id="img-user" alt="user preview" />
          </div>
        </div>
        <div class="meta">
          <div>
            <label>Prompt Used By FLUX</label>
            <div id="prompt-used" class="prompt-box"></div>
          </div>
          <div>
            <label>All Metrics / Metadata</label>
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
    const runBtn = document.getElementById("run-btn");
    const clearBtn = document.getElementById("clear-btn");
    const userInput = document.getElementById("user_image");
    const garmentInput = document.getElementById("garment_image");
    const imgUser = document.getElementById("img-user");
    const imgGarment = document.getElementById("img-garment");
    const imgOutput = document.getElementById("img-output");
    const promptInput = form.querySelector('textarea[name="prompt"]');
    const loraProfileInput = document.getElementById("lora_profile");
    const stepsInput = form.querySelector('[name="steps"]');
    const guidanceInput = form.querySelector('[name="guidance_scale"]');
    const loraScaleInput = form.querySelector('[name="lora_scale"]');
    const profileDefaults = {{
      consistency: {{ steps: 8, guidance_scale: 3.5, lora_scale: 0.4 }},
      tryon: {{ steps: 28, guidance_scale: 2.5, lora_scale: 1.0 }},
    }};
    const defaultConsistencyPrompt = { _DEFAULT_CONSISTENCY_PROMPT!r };
    const defaultTryonPrompt = { default_tryon_prompt!r };

    function buildDefaultPrompt(profile) {{
      if (profile === "tryon") {{
        return defaultTryonPrompt;
      }}
      return defaultConsistencyPrompt;
    }}

    function applyProfileDefaults(profile, forcePrompt=false) {{
      const defaults = profileDefaults[profile] || profileDefaults.consistency;
      stepsInput.value = defaults.steps;
      guidanceInput.value = defaults.guidance_scale;
      loraScaleInput.value = defaults.lora_scale;
      if (forcePrompt) {{
        promptInput.value = buildDefaultPrompt(profile);
      }}
    }}

    function setStatus(message, isError=false) {{
      statusEl.textContent = message;
      statusEl.className = isError ? "status err" : "status";
    }}

    function filePreview(input, imgTag) {{
      if (!input.files || !input.files[0]) return;
      const url = URL.createObjectURL(input.files[0]);
      imgTag.src = url;
    }}

    userInput.addEventListener("change", () => filePreview(userInput, imgUser));
    garmentInput.addEventListener("change", () => filePreview(garmentInput, imgGarment));

    clearBtn.addEventListener("click", () => {{
      promptInput.value = buildDefaultPrompt(loraProfileInput.value);
    }});

    loraProfileInput.addEventListener("change", () => {{
      applyProfileDefaults(loraProfileInput.value, true);
    }});

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      if (!userInput.files?.length || !garmentInput.files?.length) {{
        setStatus("Please upload both user and garment images.", true);
        return;
      }}

      const formData = new FormData();
      formData.append("user_image", userInput.files[0]);
      formData.append("garment_image", garmentInput.files[0]);

      const fields = [
        "lora_profile", "steps", "seed", "lora_scale", "guidance_scale", "target_type",
        "source_worn_types", "garment_prompt_description", "user_prompt_description", "prompt"
      ];
      for (const key of fields) {{
        const node = form.querySelector(`[name="${{key}}"]`);
        if (node) formData.append(key, node.value);
      }}

      runBtn.disabled = true;
      setStatus("Running try-on...");
      rawEl.textContent = "";
      promptUsedEl.textContent = "";
      imgOutput.src = "";

      try {{
        const response = await fetch("/dev/flux2/consistency-lab/run", {{
          method: "POST",
          body: formData
        }});
        const payload = await response.json();

        if (!response.ok) {{
          const detail = payload?.detail || payload?.message || `HTTP ${{response.status}}`;
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }}

        const b64 = payload.image_base64_jpeg || "";
        if (b64) {{
          imgOutput.src = `data:image/jpeg;base64,${{b64}}`;
        }}
        const usedPrompt = payload?.metadata?.prompt || "";
        promptUsedEl.textContent = usedPrompt || "(empty)";
        rawEl.textContent = JSON.stringify(payload, null, 2);
        setStatus(`Success. Latency: ${{payload.latency_seconds ?? "n/a"}} sec`);
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


@router.post("/dev/flux2/consistency-lab/run")
async def consistency_lab_run(
    user_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    lora_profile: str = Form("tryon"),
    target_type: str = Form("top"),
    source_worn_types: str = Form("top,bottom"),
    garment_prompt_description: str = Form("garment"),
    user_prompt_description: str = Form(""),
    prompt: str = Form(""),
    steps: int = Form(8, ge=4, le=50),
    seed: int = Form(42, ge=0, le=2147483647),
    lora_scale: float = Form(0.4, ge=0.0, le=2.0),
    guidance_scale: float = Form(3.5, ge=0.0, le=20.0),
    engine: AIEngine = Depends(get_ai_engine),
):
    """
    Run consistency-LoRA try-on with uploaded files and return image + full metrics.
    """
    normalized_profile = str(lora_profile or "").strip().lower()
    if normalized_profile not in _VALID_LORA_PROFILES:
        raise HTTPException(status_code=422, detail="lora_profile must be one of: consistency, tryon")
    normalized_target_type = str(target_type or "").strip().lower()
    if normalized_target_type not in _VALID_GARMENT_TYPES:
        raise HTTPException(status_code=422, detail="target_type must be one of: top, bottom, dress, outer")

    source_types = _parse_source_worn_types(source_worn_types)
    person_image = _load_uploaded_image(user_image, field_name="user_image")
    garment_ref = _load_uploaded_image(garment_image, field_name="garment_image")
    garment_prompt_text = str(garment_prompt_description or "").strip()
    user_prompt_text = str(user_prompt_description or "").strip()
    prompt_override = " ".join(str(prompt or "").split()).strip() or None
    prompt_text = prompt_override or _default_prompt_for_profile(
        profile=normalized_profile,
        target_type=normalized_target_type,
        garment_description=garment_prompt_text,
        user_description=user_prompt_text,
    )

    try:
        if normalized_profile == "tryon":
            flux_runner = getattr(engine, "flux2", None)
            if flux_runner is None:
                raise HTTPException(status_code=500, detail="Try-on runner is unavailable.")
            run_result = flux_runner.run_tryon(
                person_image=person_image,
                board_image=garment_ref,
                prompt=prompt_text,
                steps=steps,
                seed=seed,
                guidance_scale=guidance_scale,
                use_lora=True,
                lora_mode="tryon",
                lora_scale=lora_scale,
            )
        else:
            flux_runner = getattr(engine, "flux2_consistency", None)
            if flux_runner is None:
                raise HTTPException(status_code=500, detail="Consistency runner is unavailable.")
            run_result = flux_runner.run_tryon(
                person_image=person_image,
                board_image=garment_ref,
                steps=steps,
                seed=seed,
                lora_scale=lora_scale,
                guidance_scale=guidance_scale,
                target_types=[normalized_target_type],
                source_worn_types=source_types,
                garment_descriptions=[garment_prompt_text],
                user_description=user_prompt_text,
                board_mode="single",
                prompt_override=prompt_override,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Consistency lab run failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    image = run_result.get("image")
    if not isinstance(image, Image.Image):
        raise HTTPException(status_code=500, detail="Runner did not return a valid image.")

    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=95)
    image_base64 = base64.b64encode(out.getvalue()).decode("ascii")
    metadata = dict(run_result.get("metadata") or {})
    metadata.setdefault("prompt", prompt_text)
    metadata["lab_lora_profile"] = normalized_profile
    latency = float(run_result.get("latency") or 0.0)

    return {
        "status": "success",
        "message": "Consistency lab run completed",
        "image_base64_jpeg": image_base64,
        "latency_seconds": latency,
        "metadata": metadata,
        "request": {
            "lora_profile": normalized_profile,
            "target_type": normalized_target_type,
            "source_worn_types": source_types,
            "steps": steps,
            "seed": seed,
            "lora_scale": lora_scale,
            "guidance_scale": guidance_scale,
            "garment_prompt_description": garment_prompt_text,
            "user_prompt_description": user_prompt_text,
            "prompt_override": prompt_override or "",
        },
    }
