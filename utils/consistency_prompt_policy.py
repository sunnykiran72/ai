"""
Consistency LoRA prompt policy for FLUX2 try-on.

This module builds dynamic prompts for the single-LoRA consistency path.
It focuses on deterministic, case-based prompt templates for single-garment
requests while keeping a safe fallback for multi-garment inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional, Sequence

from .validation import normalize_garment_type


PROMPT_TEMPLATE_VERSION = "consistency_prompt_policy_v1"
_TYPE_ORDER = ("top", "bottom", "dress", "outer")
_KNOWN_TYPES = set(_TYPE_ORDER)


@dataclass(frozen=True)
class ConsistencyPromptPlan:
    prompt: str
    policy_id: str
    template_version: str
    target_types: List[str]
    source_worn_types: List[str]
    board_mode: str


def _dedupe_types(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for raw in values:
        kind = normalize_garment_type(str(raw or "").strip().lower()) or ""
        if kind and kind in _KNOWN_TYPES and kind not in seen:
            seen.add(kind)
            ordered.append(kind)
    return ordered


def infer_source_worn_types(source_text: Optional[str]) -> List[str]:
    """
    Infer worn garment types from free-form source/person text.

    Supports direct labels ("top", "dress") and common clothing terms
    normalized through `normalize_garment_type`.
    """
    text = " ".join(str(source_text or "").split()).strip().lower()
    if not text:
        return []

    candidates: List[str] = []

    # Direct known labels in text.
    for label in _TYPE_ORDER:
        if re.search(rf"\b{re.escape(label)}\b", text):
            candidates.append(label)

    # Token-level normalization for common synonyms.
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)?", text)
    for token in tokens:
        kind = normalize_garment_type(token)
        if kind:
            candidates.append(kind)

    # Bi-gram normalization catches phrases like "crop top" or "outer wear".
    for idx in range(len(tokens) - 1):
        pair = f"{tokens[idx]} {tokens[idx + 1]}"
        kind = normalize_garment_type(pair)
        if kind:
            candidates.append(kind)

    return _dedupe_types(candidates)


def _base_identity_clause() -> str:
    return (
        "Image 1 is the person and scene anchor. "
        "Image 2 is the garment anchor. "
        "Keep the same face identity, skin tone, hair, body proportions, pose, hand placement, "
        "camera framing, background, and lighting from Image 1. "
        "Do not preserve the original garment shape inside the target clothing region from Image 1. "
        "When person cues from Image 1 conflict with garment geometry from Image 2, keep Image 1 for identity and scene, "
        "but follow Image 2 for garment geometry, neckline, shoulder construction, sleeve shape, hemline, and asymmetry."
    )


def _quality_clause() -> str:
    return (
        "Render realistic fabric drape, natural fold direction, clean seam lines, and crisp garment edges. "
        "Maintain stable garment boundaries inside the intended edit region with natural transitions. "
        "Match the construction of the garment from Image 2, not the previous clothing silhouette from Image 1."
    )


def _single_target_clause(target_type: str, source_types: Sequence[str]) -> tuple[str, str]:
    source = set(source_types or [])
    if target_type == "top":
        if "dress" in source:
            return (
                "single_top_over_dress_v1",
                "Replace the entire upper-garment structure from shoulders to waist with the top from Image 2. "
                "Treat this as an upper-layer replacement over a dress base. "
                "Follow Image 2 for neckline, collar, shoulder line, sleeve length and width, cuff shape, torso fit, hemline, and asymmetry. "
                "Do not keep the original upper-dress neckline, sleeves, shoulder shape, or upper-body hem from Image 1. "
                "Keep only the lower dress panel, hem length, and lower-body garment silhouette from Image 1 consistent.",
            )
        return (
            "single_top_v1",
            "Replace the entire upper-garment structure from shoulders to waist with the top from Image 2. "
            "Follow Image 2 for neckline, collar, shoulder line, sleeve length and width, cuff shape, torso fit, hemline, and asymmetry. "
            "Do not keep the original shirt or top neckline, sleeves, shoulder shape, or upper-body hem from Image 1. "
            "Keep waist-down clothing, legs, and footwear styling from Image 1 consistent.",
        )
    if target_type == "bottom":
        if "dress" in source:
            return (
                "single_bottom_over_dress_v1",
                "Apply the bottom garment from Image 2 in the lower clothing region from waist to hemline. "
                "Keep the upper bodice and upper-body clothing structure from Image 1 consistent.",
            )
        return (
            "single_bottom_v1",
            "Apply the bottom garment from Image 2 from waist to ankle region. "
            "Keep upper-body clothing, torso silhouette, and sleeve area from Image 1 consistent.",
        )
    if target_type == "dress":
        return (
            "single_dress_v1",
            "Replace the full outfit with the dress from Image 2 as one continuous garment across upper and lower clothing regions. "
            "Follow Image 2 for neckline, shoulder construction, sleeve shape, waistline, skirt volume, hemline, and asymmetry. "
            "Do not keep the previous separate top or bottom silhouette from Image 1. "
            "Keep person geometry and scene cues from Image 1 consistent.",
        )
    if target_type == "outer":
        return (
            "single_outer_v1",
            "Replace the visible outer-layer structure around torso and arms with the outerwear from Image 2. "
            "Follow Image 2 for lapel, collar, shoulder line, sleeve shape, opening, length, and silhouette. "
            "Keep the base outfit structure underneath and lower-body clothing from Image 1 consistent.",
        )
    return (
        "single_generic_v1",
        "Apply the garment from Image 2 only within its intended clothing region on Image 1.",
    )


def _build_single_garment_prompt(
    target_type: str,
    source_types: Sequence[str],
) -> ConsistencyPromptPlan:
    policy_id, scope_clause = _single_target_clause(target_type, source_types)
    prompt = " ".join(
        part.strip()
        for part in (
            _base_identity_clause(),
            f"Single-garment try-on target type: {target_type}.",
            scope_clause,
            _quality_clause(),
        )
        if part
    )
    return ConsistencyPromptPlan(
        prompt=prompt,
        policy_id=policy_id,
        template_version=PROMPT_TEMPLATE_VERSION,
        target_types=[target_type],
        source_worn_types=list(_dedupe_types(source_types)),
        board_mode="single",
    )


def _build_multi_target_prompt(
    target_types: Sequence[str],
    source_types: Sequence[str],
    garment_descriptions: Sequence[str],
    board_mode: str,
) -> ConsistencyPromptPlan:
    types_text = ", ".join(target_types) if target_types else "outfit"
    refs = [f"Garment {idx + 1}: {desc}" for idx, desc in enumerate(garment_descriptions or []) if str(desc or "").strip()]
    refs_text = " ".join(refs) if refs else "Use Image 2 collage as the outfit reference."
    prompt = " ".join(
        part.strip()
        for part in (
            _base_identity_clause(),
            f"Multi-garment try-on with target types: {types_text}.",
            "Apply all garment items from Image 2 as one coherent outfit while preserving non-target identity and scene cues from Image 1.",
            refs_text,
            _quality_clause(),
        )
        if part
    )
    return ConsistencyPromptPlan(
        prompt=prompt,
        policy_id="multi_target_collage_v1" if board_mode == "collage" else "multi_target_single_v1",
        template_version=PROMPT_TEMPLATE_VERSION,
        target_types=list(target_types),
        source_worn_types=list(_dedupe_types(source_types)),
        board_mode=board_mode,
    )


def build_consistency_prompt_plan(
    *,
    target_types: Optional[Sequence[str]] = None,
    source_worn_types: Optional[Sequence[str]] = None,
    source_text: Optional[str] = None,
    garment_descriptions: Optional[Sequence[str]] = None,
    board_mode: str = "single",
) -> ConsistencyPromptPlan:
    """
    Build a deterministic prompt plan for consistency-loRA try-on.

    Priority:
    1) Explicit `source_worn_types`
    2) Types inferred from `source_text`
    """
    normalized_targets = _dedupe_types(target_types or [])
    if not normalized_targets:
        normalized_targets = ["top"]

    normalized_source = _dedupe_types(source_worn_types or [])
    if not normalized_source and source_text:
        normalized_source = infer_source_worn_types(source_text)

    clean_descriptions = [
        " ".join(str(desc or "").split()).strip()
        for desc in (garment_descriptions or [])
        if str(desc or "").strip()
    ]

    mode = str(board_mode or "single").strip().lower()
    if mode not in {"single", "collage"}:
        mode = "single"

    if len(normalized_targets) == 1 and mode == "single":
        return _build_single_garment_prompt(normalized_targets[0], normalized_source)

    return _build_multi_target_prompt(
        target_types=normalized_targets,
        source_types=normalized_source,
        garment_descriptions=clean_descriptions,
        board_mode=mode,
    )
