# SPDX-License-Identifier: Apache-2.0
"""Compatibility checks for the SGLang Qwen3-VL HF-parity patch."""

from __future__ import annotations


def test_qwen3_vl_patch_accepts_installed_sglang_release() -> None:
    from sglang_omni.model_runner._sglang_qwen3_vl_patches import (
        _check_sglang_version,
    )

    _check_sglang_version()


def test_qwen3_vl_patch_surface_matches_installed_sglang() -> None:
    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

    from sglang_omni.model_runner._sglang_qwen3_vl_patches import (
        _check_target_class_surface,
    )

    _check_target_class_surface(Qwen3VLMoeVisionModel)
