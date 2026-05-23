# SPDX-License-Identifier: Apache-2.0
"""SeedTTS dataset loader.

Loads samples from HuggingFace Arrow/Parquet repos (e.g.
``zhaochenyang20/seed-tts-eval-50-arrow``).  Audio bytes are staged to a
process-scoped temporary directory so that downstream consumers (which
expect filesystem paths) work unchanged.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"sample_id", "ref_text", "ref_audio_path", "target_text", "ref_audio"}


@dataclass
class SampleInput:
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str


_STAGED_CACHE: dict[tuple[str, str], list[SampleInput]] = {}


def load_seedtts_samples(
    repo_id: str,
    max_samples: int | None = None,
    *,
    split: str = "en",
) -> list[SampleInput]:
    """Load SeedTTS evaluation samples from a HuggingFace Arrow dataset."""
    cache_key = (repo_id, split)
    if cache_key in _STAGED_CACHE:
        samples = _STAGED_CACHE[cache_key]
        return samples[:max_samples] if max_samples else list(samples)

    from datasets import Audio, load_dataset

    logger.info("Loading %s split=%s from HuggingFace ...", repo_id, split)
    ds = load_dataset(repo_id, split=split)

    missing = _REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset {repo_id} split={split} is missing columns: {missing}"
        )

    ds = ds.cast_column("ref_audio", Audio(decode=False))

    tmpdir = Path(tempfile.mkdtemp(prefix=f"seedtts_{split}_"))
    atexit.register(shutil.rmtree, str(tmpdir), True)
    logger.info("Staging audio to %s", tmpdir)

    samples: list[SampleInput] = []
    written: set[str] = set()

    for row in ds:
        rel = row["ref_audio_path"]
        audio = row["ref_audio"] or {}
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            raise ValueError(
                f"Empty audio bytes for {repo_id}/{split}/{row['sample_id']}"
            )

        wav_path = tmpdir / rel
        if rel not in written:
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.write_bytes(audio_bytes)
            written.add(rel)

        samples.append(
            SampleInput(
                sample_id=row["sample_id"],
                ref_text=row["ref_text"],
                ref_audio=str(wav_path),
                target_text=row["target_text"],
            )
        )

    _STAGED_CACHE[cache_key] = samples
    logger.info(
        "Loaded %d samples (%d unique audio files) from %s/%s",
        len(samples), len(written), repo_id, split,
    )
    return samples[:max_samples] if max_samples else list(samples)
