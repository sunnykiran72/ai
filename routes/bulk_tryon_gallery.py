from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse


router = APIRouter()

_DEFAULT_RUN_DIR = "bulk_tryon_compare_seed44_vs_seed123_1280_20260408_closedvocab_cached110"
_DEBUG_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "debug_outputs"


def _resolve_report_path(run_dir: str) -> Path:
    candidate = Path(str(run_dir or "").strip())
    if not candidate.name or candidate.name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid run_dir.")
    if candidate.name != str(candidate):
        raise HTTPException(status_code=400, detail="run_dir must be a single directory name.")
    report_path = (_DEBUG_OUTPUTS_DIR / candidate.name / "report.html").resolve()
    try:
        report_path.relative_to(_DEBUG_OUTPUTS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="run_dir escapes debug_outputs.") from exc
    return report_path


@router.get("/dev/glamify-bulk-tryon-testing")
@router.get("/dev/flux2/glamify-bulk-tryon-testing")
async def bulk_tryon_gallery(
    run_dir: str = Query(
        default=_DEFAULT_RUN_DIR,
        description="Directory name under debug_outputs containing report.html",
    ),
):
    report_path = _resolve_report_path(run_dir)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found for run_dir={run_dir}")
    return FileResponse(report_path, media_type="text/html")
