"""Unit tests for the fixed-seed Qwen3-TTS KDA benchmark harness."""

from __future__ import annotations

import argparse
import base64
import json
import runpy
import sys
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "eval"
    / "benchmark_qwen3_tts_kda.py"
)
_MODULE = runpy.run_path(str(_SCRIPT))
_parse_args = _MODULE["_parse_args"]
_request_payload = _MODULE["_request_payload"]
_batch_result = _MODULE["_batch_result"]


def _args(repetition_penalty: float | None) -> argparse.Namespace:
    return argparse.Namespace(
        model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        reference_audio="file:///reference.wav",
        reference_text="reference transcript",
        seed=20260819,
        repetition_penalty=repetition_penalty,
    )


def test_request_payload_preserves_default_when_penalty_is_omitted() -> None:
    assert "repetition_penalty" not in _request_payload(_args(None))
    assert _request_payload(_args(1.0))["repetition_penalty"] == 1.0


def test_parse_args_rejects_nonpositive_repetition_penalty(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_qwen3_tts_kda.py",
            "--reference-audio",
            "file:///reference.wav",
            "--reference-text",
            "reference transcript",
            "--input",
            "benchmark text",
            "--seed",
            "1",
            "--output",
            "report.json",
            "--repetition-penalty",
            "0",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        _parse_args()


def test_batch_result_rejects_missing_items() -> None:
    body = json.dumps(
        {
            "results": [
                {
                    "status": "success",
                    "index": 0,
                    "audio_data": base64.b64encode(b"audio").decode(),
                }
            ]
        }
    ).encode()

    with pytest.raises(RuntimeError, match="unexpected number of items"):
        _batch_result(body, wall_s=1.0, expected_item_count=4)
