# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni's narrow SGLang execution compatibility hooks."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from sglang_omni.models.qwen3_omni.sglang_compat import (
    configure_single_stream_fa3_graph_metadata,
)


def _runner(backend):
    return SimpleNamespace(
        attn_backend=backend,
        server_args=SimpleNamespace(
            enable_pdmux=False,
            enable_two_batch_overlap=False,
        ),
    )


def test_single_stream_fa3_disables_scheduler_metadata_precompute(caplog) -> None:
    class FlashAttentionBackend:
        _get_scheduler_metadata = staticmethod(lambda: None)
        _disable_scheduler_metadata_precompute = False

    backend = FlashAttentionBackend()

    with caplog.at_level(logging.INFO):
        applied = configure_single_stream_fa3_graph_metadata(
            _runner(backend),
            stage="thinker",
        )

    assert applied is True
    assert backend._disable_scheduler_metadata_precompute is True
    assert "Configured Qwen thinker single-stream FA3 CUDA graph metadata" in caplog.text


def test_non_fa3_backend_is_unchanged(caplog) -> None:
    backend = SimpleNamespace(
        _get_scheduler_metadata=None,
        _disable_scheduler_metadata_precompute=False,
    )

    with caplog.at_level(logging.INFO):
        applied = configure_single_stream_fa3_graph_metadata(
            _runner(backend),
            stage="talker",
        )

    assert applied is False
    assert backend._disable_scheduler_metadata_precompute is False
    assert "compatibility not applied" in caplog.text
