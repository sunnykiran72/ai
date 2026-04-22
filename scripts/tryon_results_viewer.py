from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent.parent
ROUTE_FILE = APP_DIR / "routes" / "tryon_results.py"
TMP_DIR = APP_DIR / "tmp"


def _load_tryon_results_router():
    spec = importlib.util.spec_from_file_location("tryon_results_route", ROUTE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load route file: {ROUTE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.router


app = FastAPI(title="Try-on Results Viewer", version="1.0.0")
app.include_router(_load_tryon_results_router(), tags=["tryon-results"])
app.mount("/tmp", StaticFiles(directory=str(TMP_DIR)), name="tmp")


@app.get("/", include_in_schema=False)
async def _root() -> RedirectResponse:
    return RedirectResponse(url="/v1/tryon-results", status_code=307)
