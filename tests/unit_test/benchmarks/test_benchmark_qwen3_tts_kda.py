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


def test_parse_args_requires_fixed_layout_for_correctness_reference(
    monkeypatch,
) -> None:
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
            "--correctness-reference",
            "main.json",
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


def test_controlled_correctness_accepts_stable_exact_signature() -> None:
    result = _controlled_correctness(
        [
            _controlled_run("hash-b", "hash-a"),
            _controlled_run("hash-a", "hash-b"),
        ],
        ("hash-a", "hash-b"),
        [{"report_sha256": "main", "signature": ["hash-a", "hash-b"]}],
        fixed_vocoder_layout=True,
    )

    assert result["status"] == "pass"
    assert result["stable_across_runs"] is True
    assert result["observed_signatures"] == [["hash-a", "hash-b"]]


def test_controlled_correctness_rejects_new_layout_of_known_hashes() -> None:
    result = _controlled_correctness(
        [_controlled_run("hash-a", "hash-a")],
        ("hash-a", "hash-b"),
        [{"report_sha256": "main", "signature": ["hash-a", "hash-b"]}],
        fixed_vocoder_layout=True,
    )

    assert result["status"] == "fail"
    assert result["stable_across_runs"] is True
    assert result["observed_signatures"] == [["hash-a", "hash-a"]]


def test_controlled_correctness_rejects_same_arm_signature_drift() -> None:
    result = _controlled_correctness(
        [
            _controlled_run("hash-a", "hash-b"),
            _controlled_run("hash-a", "hash-a"),
        ],
        ("hash-a", "hash-b"),
        [{"report_sha256": "main", "signature": ["hash-a", "hash-b"]}],
        fixed_vocoder_layout=True,
    )

    assert result["status"] == "fail"
    assert result["stable_across_runs"] is False


def test_controlled_correctness_is_explicit_without_references() -> None:
    result = _controlled_correctness(
        [_controlled_run("hash-a", "hash-b")],
        None,
        [],
        fixed_vocoder_layout=False,
    )

    assert result["status"] == "not_evaluated"


def test_fixed_layout_baseline_requires_stable_signature() -> None:
    result = _controlled_correctness(
        [
            _controlled_run("hash-a", "hash-b"),
            _controlled_run("hash-b", "hash-a"),
        ],
        None,
        [],
        fixed_vocoder_layout=True,
    )

    assert result["status"] == "pass"
    assert result["stable_across_runs"] is True


def _reference_report(config: dict, *hashes: str) -> dict:
    return {
        "schema_version": 4,
        "config": config,
        "controlled_concurrency": {"runs": [_controlled_run(*hashes)]},
    }


def _fixed_reference_config() -> dict:
    return {
        "model": "model",
        "input": "input",
        "reference_audio": "file:///reference.wav",
        "reference_text": "reference transcript",
        "seed": 20260819,
        "repetition_penalty": 1.05,
        "max_new_tokens": 64,
        "concurrency": 4,
        "fixed_vocoder_layout": True,
    }


def test_correctness_references_require_one_exact_signature(tmp_path: Path) -> None:
    config = _fixed_reference_config()
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_reference_report(config, "hash-a", "hash-b")))
    after.write_text(json.dumps(_reference_report(config, "hash-b", "hash-a")))

    signature, metadata = _load_correctness_references(
        [before, after], expected_config=config
    )

    assert signature == ("hash-a", "hash-b")
    assert [entry["signature"] for entry in metadata] == [
        ["hash-a", "hash-b"],
        ["hash-a", "hash-b"],
    ]
    assert all("report_sha256" in entry for entry in metadata)


def test_correctness_references_reject_conflicting_signatures(tmp_path: Path) -> None:
    config = _fixed_reference_config()
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_reference_report(config, "hash-a", "hash-b")))
    after.write_text(json.dumps(_reference_report(config, "hash-a", "hash-a")))

    with pytest.raises(RuntimeError, match="disagrees with earlier references"):
        _load_correctness_references([before, after], expected_config=config)


def test_correctness_reference_rejects_config_mismatch(tmp_path: Path) -> None:
    config = _fixed_reference_config()
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(_reference_report(dict(config, concurrency=8), "hash"))
    )

    with pytest.raises(RuntimeError, match="config does not match"):
        _load_correctness_references([reference], expected_config=config)


def test_correctness_reference_rejects_generation_length_mismatch(
    tmp_path: Path,
) -> None:
    config = _fixed_reference_config()
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(_reference_report(dict(config, max_new_tokens=96), "hash"))
    )

    with pytest.raises(RuntimeError, match="config does not match"):
        _load_correctness_references([reference], expected_config=config)


def test_correctness_reference_rejects_legacy_schema(tmp_path: Path) -> None:
    config = _fixed_reference_config()
    reference = tmp_path / "reference.json"
    report = _reference_report(config, "hash")
    report["schema_version"] = 3
    reference.write_text(json.dumps(report))

    with pytest.raises(RuntimeError, match="must use schema version 4"):
        _load_correctness_references([reference], expected_config=config)
