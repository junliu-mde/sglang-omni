"""Measure seeded Qwen3-TTS behavior for a fixed C1 and controlled C4 workload.

This script is deliberately small.  It sends a fixed voice-cloning request to
the public HTTP API, records end-to-end wall time and response metadata, and
checks C1 audio byte identity.  C4 uses ``/v1/audio/speech/batch`` so all four
items enter the server in one arrival group.  The endpoint does not promise a
fixed scheduler batch layout, so the script records each observed C4 layout but
does not incorrectly fail a valid run when those layouts differ.

Example::

    python -m benchmarks.eval.benchmark_qwen3_tts_kda \
        --base-url http://127.0.0.1:18000 \
        --reference-audio file:///path/to/reference.wav \
        --reference-text "Reference transcript." \
        --input "Text to synthesize." \
        --seed 20260819 \
        --output /tmp/qwen3-tts.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fixed-seed Qwen3-TTS C1 and controlled C4 HTTP behavior."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--input", required=True, dest="input_text")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--repetitions", default=5, type=int)
    parser.add_argument("--warmup", default=1, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--timeout-s", default=180.0, type=float)
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if args.repetition_penalty is not None and args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be positive")
    return args


def _request_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "voice": "default",
        "response_format": "wav",
        "references": [
            {
                "audio_path": args.reference_audio,
                "text": args.reference_text,
            }
        ],
        "seed": args.seed,
    }
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = args.repetition_penalty
    return payload


def _post(
    url: str, payload: dict[str, Any], timeout_s: float
) -> tuple[bytes, dict[str, str], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
            headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST {url} failed with HTTP {exc.code}: {error_body}"
        ) from exc
    return body, headers, time.perf_counter() - started


def _audio_result(audio: bytes, wall_s: float, headers: dict[str, str]) -> dict[str, Any]:
    def numeric_header(name: str) -> float | int | None:
        value = headers.get(name)
        if value is None:
            return None
        try:
            return float(value) if name == "x-engine-time" else int(value)
        except ValueError:
            return None

    return {
        "wall_s": wall_s,
        "bytes": len(audio),
        "sha256": hashlib.sha256(audio).hexdigest(),
        "engine_time_s": numeric_header("x-engine-time"),
        "completion_tokens": numeric_header("x-completion-tokens"),
        "prompt_tokens": numeric_header("x-prompt-tokens"),
        "headers": headers,
    }


def _batch_result(
    body: bytes, wall_s: float, expected_item_count: int
) -> dict[str, Any]:
    try:
        response = json.loads(body)
        raw_results = response["results"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("speech batch response has an invalid schema") from exc

    items: list[dict[str, Any]] = []
    for item in raw_results:
        if item.get("status") != "success":
            raise RuntimeError(f"speech batch item failed: {item}")
        try:
            audio = base64.b64decode(item["audio_data"], validate=True)
            index = int(item["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("successful speech batch item has an invalid schema") from exc
        items.append(
            {
                "index": index,
                "bytes": len(audio),
                "sha256": hashlib.sha256(audio).hexdigest(),
                "audio": audio,
            }
        )
    items.sort(key=lambda item: item["index"])
    if len(items) != expected_item_count:
        raise RuntimeError(
            "speech batch response returned an unexpected number of items: "
            f"expected={expected_item_count}, got={len(items)}"
        )
    if [item["index"] for item in items] != list(range(expected_item_count)):
        raise RuntimeError("speech batch response did not return one item per input")
    return {
        "wall_s": wall_s,
        "item_count": len(items),
        "items": [{key: value for key, value in item.items() if key != "audio"} for item in items],
        "audio": [item["audio"] for item in items],
    }


def _summary(values: list[float | int | None]) -> dict[str, float] | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return {
        "min": min(present),
        "median": statistics.median(present),
        "max": max(present),
        "mean": statistics.fmean(present),
    }


def _write_audio(path: Path, name: str, audio: bytes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_bytes(audio)


def _run_c1(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[bytes]]:
    endpoint = f"{args.base_url.rstrip('/')}/v1/audio/speech"
    results: list[dict[str, Any]] = []
    audio_outputs: list[bytes] = []
    for _ in range(args.warmup):
        _post(endpoint, dict(payload, input=args.input_text), args.timeout_s)
    for iteration in range(args.repetitions):
        audio, headers, wall_s = _post(
            endpoint, dict(payload, input=args.input_text), args.timeout_s
        )
        results.append({"iteration": iteration, **_audio_result(audio, wall_s, headers)})
        audio_outputs.append(audio)
    return results, audio_outputs


def _run_c4(args: argparse.Namespace, payload: dict[str, Any]) -> list[dict[str, Any]]:
    endpoint = f"{args.base_url.rstrip('/')}/v1/audio/speech/batch"
    batch_payload = dict(
        payload,
        items=[{"input": args.input_text} for _ in range(args.batch_size)],
    )
    results: list[dict[str, Any]] = []
    for _ in range(args.warmup):
        _post(endpoint, batch_payload, args.timeout_s)
    for iteration in range(args.repetitions):
        body, _headers, wall_s = _post(endpoint, batch_payload, args.timeout_s)
        results.append(
            {
                "iteration": iteration,
                **_batch_result(body, wall_s, args.batch_size),
            }
        )
    return results


def _c4_layouts(c4: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], int] = {}
    for result in c4:
        layout = tuple(item["sha256"] for item in result["items"])
        counts[layout] = counts.get(layout, 0) + 1
    return [
        {"hashes": list(layout), "runs": count}
        for layout, count in sorted(
            counts.items(), key=lambda entry: (-entry[1], entry[0])
        )
    ]


def _determinism(c1: list[dict[str, Any]]) -> dict[str, bool]:
    c1_hashes = {result["sha256"] for result in c1}
    return {
        "c1_byte_identical": len(c1_hashes) == 1,
    }


def main() -> None:
    args = _parse_args()
    payload = _request_payload(args)
    c1, c1_audio = _run_c1(args, payload)
    c4 = _run_c4(args, payload)
    determinism = _determinism(c1)

    if args.audio_dir is not None:
        for result, audio in zip(c1, c1_audio, strict=True):
            _write_audio(args.audio_dir, f"c1-{result['iteration']}.wav", audio)
        for result in c4:
            for item, audio in zip(result["items"], result["audio"], strict=True):
                _write_audio(
                    args.audio_dir,
                    f"c4-{result['iteration']}-item-{item['index']}.wav",
                    audio,
                )

    report = {
        "schema_version": 2,
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "input": args.input_text,
            "reference_audio": args.reference_audio,
            "reference_text": args.reference_text,
            "seed": args.seed,
            "repetition_penalty": args.repetition_penalty,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "batch_size": args.batch_size,
        },
        "c1": {
            "runs": c1,
            "wall_s": _summary([result["wall_s"] for result in c1]),
            "engine_time_s": _summary([result["engine_time_s"] for result in c1]),
        },
        "controlled_c4": {
            "layout_note": (
                "The batch endpoint controls arrival grouping, not the internal "
                "scheduler layout. Layouts are recorded and are not a seeded "
                "correctness assertion."
            ),
            "layouts": _c4_layouts(c4),
            "runs": [
                {key: value for key, value in result.items() if key != "audio"}
                for result in c4
            ],
            "wall_s": _summary([result["wall_s"] for result in c4]),
        },
        "determinism": determinism,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if not determinism["c1_byte_identical"]:
        raise SystemExit(f"seeded audio identity check failed; see {args.output}")


if __name__ == "__main__":
    main()
