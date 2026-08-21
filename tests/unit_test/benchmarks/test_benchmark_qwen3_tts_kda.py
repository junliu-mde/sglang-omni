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
_audio_result = _MODULE["_audio_result"]
_batch_result = _MODULE["_batch_result"]
_controlled_correctness = _MODULE["_controlled_correctness"]
_controlled_signatures = _MODULE["_controlled_signatures"]
_load_correctness_references = _MODULE["_load_correctness_references"]


def _args(
    repetition_penalty: float | None, max_new_tokens: int | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        reference_audio="file:///reference.wav",
        reference_text="reference transcript",
        seed=20260819,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )


def test_request_payload_preserves_default_when_penalty_is_omitted() -> None:
    assert "repetition_penalty" not in _request_payload(_args(None))
    assert _request_payload(_args(1.0))["repetition_penalty"] == 1.0


def test_request_payload_includes_fixed_generation_length() -> None:
    assert "max_new_tokens" not in _request_payload(_args(None))
    assert _request_payload(_args(None, max_new_tokens=64))["max_new_tokens"] == 64


def test_audio_result_preserves_finish_reason_header() -> None:
    result = _audio_result(
        b"audio",
        wall_s=1.0,
        headers={"x-finish-reason": "length", "x-completion-tokens": "64"},
    )

    assert result["finish_reason"] == "length"
    assert result["completion_tokens"] == 64


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


def test_parse_args_rejects_nonpositive_concurrency(monkeypatch) -> None:
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
            "--concurrency",
            "0",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        _parse_args()


def test_parse_args_rejects_nonpositive_max_new_tokens(monkeypatch) -> None:
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
            "--max-new-tokens",
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


def test_batch_result_preserves_finish_reason() -> None:
    body = json.dumps(
        {
            "results": [
                {
                    "status": "success",
                    "index": 0,
                    "audio_data": base64.b64encode(b"audio").decode(),
                    "finish_reason": "length",
                }
            ]
        }
    ).encode()

    result = _batch_result(body, wall_s=1.0, expected_item_count=1)

    assert result["items"][0]["finish_reason"] == "length"


def _controlled_run(*hashes: str) -> dict:
    return {
        "items": [
            {"index": index, "sha256": sha256} for index, sha256 in enumerate(hashes)
        ]
    }


def test_controlled_signatures_ignore_response_item_order() -> None:
    assert _controlled_signatures([_controlled_run("hash-b", "hash-a")]) == [
        ("hash-a", "hash-b")
    ]


def test_controlled_correctness_accepts_new_layout_of_exact_main_hashes() -> None:
    runs = [
        _controlled_run("hash-b", "hash-a"),
        _controlled_run("hash-a", "hash-a"),
    ]
    accepted = {"hash-a", "hash-b"}

    result = _controlled_correctness(
        runs,
        accepted,
        [{"report_sha256": "main", "hash_count": 2}],
    )

    assert result["status"] == "pass"
    assert result["matched_items"] == 4
    assert result["total_items"] == 4
    assert result["unmatched_hashes"] == []


def test_controlled_correctness_rejects_unseen_exact_wav_hash() -> None:
    result = _controlled_correctness(
        [_controlled_run("changed", "hash-a")],
        {"hash-a", "hash-b"},
        [{"report_sha256": "main", "hash_count": 2}],
    )

    assert result["status"] == "fail"
    assert result["matched_items"] == 1
    assert result["total_items"] == 2
    assert result["unmatched_hashes"] == ["changed"]


def test_controlled_correctness_is_explicit_without_references() -> None:
    result = _controlled_correctness([_controlled_run("hash-a", "hash-b")], set(), [])

    assert result["status"] == "not_evaluated"


def _reference_report(config: dict, *hashes: str) -> dict:
    return {
        "schema_version": 3,
        "config": config,
        "controlled_concurrency": {"runs": [_controlled_run(*hashes)]},
    }


def test_correctness_references_combine_exact_wav_hashes(tmp_path: Path) -> None:
    config = {
        "model": "model",
        "input": "input",
        "reference_audio": "file:///reference.wav",
        "reference_text": "reference transcript",
        "seed": 20260819,
        "repetition_penalty": 1.05,
        "max_new_tokens": 64,
        "concurrency": 4,
    }
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_reference_report(config, "hash-a", "hash-b")))
    after.write_text(json.dumps(_reference_report(config, "hash-c", "hash-c")))

    accepted, metadata = _load_correctness_references(
        [before, after], expected_config=config
    )

    assert accepted == {"hash-a", "hash-b", "hash-c"}
    assert [entry["hash_count"] for entry in metadata] == [2, 1]
    assert all("report_sha256" in entry for entry in metadata)


def test_correctness_reference_rejects_config_mismatch(tmp_path: Path) -> None:
    config = {
        "model": "model",
        "input": "input",
        "reference_audio": "file:///reference.wav",
        "reference_text": "reference transcript",
        "seed": 20260819,
        "repetition_penalty": 1.05,
        "max_new_tokens": 64,
        "concurrency": 4,
    }
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(_reference_report(dict(config, concurrency=8), "hash"))
    )

    with pytest.raises(RuntimeError, match="config does not match"):
        _load_correctness_references([reference], expected_config=config)


def test_correctness_reference_rejects_generation_length_mismatch(
    tmp_path: Path,
) -> None:
    config = {
        "model": "model",
        "input": "input",
        "reference_audio": "file:///reference.wav",
        "reference_text": "reference transcript",
        "seed": 20260819,
        "repetition_penalty": 1.05,
        "max_new_tokens": 64,
        "concurrency": 4,
    }
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(_reference_report(dict(config, max_new_tokens=96), "hash"))
    )

    with pytest.raises(RuntimeError, match="config does not match"):
        _load_correctness_references([reference], expected_config=config)


def test_correctness_reference_accepts_legacy_missing_generation_length(
    tmp_path: Path,
) -> None:
    config = {
        "model": "model",
        "input": "input",
        "reference_audio": "file:///reference.wav",
        "reference_text": "reference transcript",
        "seed": 20260819,
        "repetition_penalty": 1.05,
        "max_new_tokens": None,
        "concurrency": 4,
    }
    reference = tmp_path / "reference.json"
    legacy_config = dict(config)
    legacy_config.pop("max_new_tokens")
    reference.write_text(json.dumps(_reference_report(legacy_config, "hash")))

    accepted, _ = _load_correctness_references(
        [reference], expected_config=config
    )

    assert accepted == {"hash"}
