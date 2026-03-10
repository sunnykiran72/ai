from __future__ import annotations

from ai.modules.wardrobe.extraction.contracts import AnalyzeStageTimings


EXTRACTION_STAGE_ORDER = (
    "input",
    "detection",
    "prompting",
    "generation",
    "postprocess",
    "response",
)


def default_extraction_stage_timings() -> AnalyzeStageTimings:
    return AnalyzeStageTimings()
