# SPDX-License-Identifier: Apache-2.0
"""Narrow SGLang runtime compatibility hooks for Qwen3-Omni."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def configure_single_stream_fa3_graph_metadata(
    model_runner: Any,
    *,
    stage: str,
) -> bool:
    """Keep FA3 scheduler planning inside Qwen's single-stream CUDA graph.

    SGLang 0.5.15's FA3 backend can precompute scheduler metadata immediately
    before every graph replay. That is useful for its multi-stream scheduler,
    but Qwen3-Omni uses a single-stream execution bridge and otherwise pays an
    extra out-of-graph planning launch on every decode step.

    This hook intentionally feature-detects the 0.5.15 private backend contract
    instead of importing its private class. It must run after attention backend
    initialization and before CUDA graph capture so FA3 captures its ordinary
    in-graph scheduling path. Other SGLang versions and non-FA3 backends remain
    unchanged.
    """

    server_args = getattr(model_runner, "server_args", None)
    if bool(getattr(server_args, "enable_pdmux", False)) or bool(
        getattr(server_args, "enable_two_batch_overlap", False)
    ):
        logger.info(
            "Qwen %s FA3 graph compatibility not applied: SGLang multi-stream "
            "execution is enabled",
            stage,
        )
        return False

    backend = getattr(model_runner, "attn_backend", None)
    backend_name = type(backend).__name__ if backend is not None else "None"
    scheduler_metadata_fn = getattr(backend, "_get_scheduler_metadata", None)
    disable_attr = "_disable_scheduler_metadata_precompute"
    if (
        backend is None
        or not hasattr(backend, disable_attr)
        or not callable(scheduler_metadata_fn)
    ):
        logger.info(
            "Qwen %s FA3 graph compatibility not applied: backend=%s does not "
            "expose SGLang's scheduler-metadata precompute contract",
            stage,
            backend_name,
        )
        return False

    was_disabled = bool(getattr(backend, disable_attr))
    setattr(backend, disable_attr, True)
    logger.info(
        "Configured Qwen %s single-stream FA3 CUDA graph metadata "
        "(backend=%s, already_disabled=%s)",
        stage,
        backend_name,
        was_disabled,
    )
    return True
