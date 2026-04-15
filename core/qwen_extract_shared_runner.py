"""
Shared singleton for Qwen extract-outfit runner within a process.

This lets /analyze and /dev/qwen/extract-outfit-lab reuse one loaded pipeline
instead of each endpoint constructing its own runner instance.
"""

from __future__ import annotations

from typing import Optional

from core.qwen_image_edit_runner import QwenImageEditRunner

_SHARED_QWEN_EXTRACT_RUNNER: Optional[QwenImageEditRunner] = None


def get_shared_qwen_extract_runner() -> QwenImageEditRunner:
    global _SHARED_QWEN_EXTRACT_RUNNER
    if _SHARED_QWEN_EXTRACT_RUNNER is None:
        _SHARED_QWEN_EXTRACT_RUNNER = QwenImageEditRunner()
    return _SHARED_QWEN_EXTRACT_RUNNER

