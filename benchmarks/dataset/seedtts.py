# SPDX-License-Identifier: Apache-2.0
"""SeedTTS dataset loader.

Supports two source formats:
- Local ``meta.lst`` files (pipe-delimited: ``id|ref_text|ref_audio_path|target_text``)
- HuggingFace Arrow/Parquet repos (e.g. ``zhaochenyang20/seed-tts-eval-50-arrow``)

Arrow datasets stage WAV bytes to a process-scoped temporary directory so that
downstream consumers (which expect filesystem paths) work unchanged.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {
    "sample_id",
    "ref_text",
    "ref_audio_path",
    "target_text",
    "ref_audio",
}


@dataclass
class SampleInput:
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str


_STAGED_CACHE: dict[tuple[str, str, int | None], list[SampleInput]] = {}


def load_seedtts_samples(
    source: str,
    max_samples: int | None = None,
    *,
    split: str = "en",
) -> list[SampleInput]:
    """Load SeedTTS evaluation samples.

    *source* is either a local ``meta.lst`` file path or a HuggingFace dataset
    repo id (e.g. ``zhaochenyang20/seed-tts-eval-50-arrow``).  When a repo id
    is given, the dataset is fetched via ``datasets.load_dataset`` and audio
    bytes are staged to a temporary directory.
    """
    if os.path.isfile(source) or source.endswith(".lst"):
        return _load_from_meta_lst(source, max_samples)
    return _load_from_arrow(source, split, max_samples)


def _load_from_meta_lst(path: str, max_samples: int | None) -> list[SampleInput]:
    """Legacy loader: parse a pipe-delimited meta.lst file."""
    base_dir = os.path.dirname(path)
    samples: list[SampleInput] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            samples.append(
                SampleInput(
                    sample_id=parts[0],
                    ref_text=parts[1],
                    ref_audio=os.path.join(base_dir, parts[2]),
                    target_text=parts[3],
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def _load_from_arrow(
    repo_id: str, split: str, max_samples: int | None
) -> list[SampleInput]:
    """Load from a HuggingFace Arrow/Parquet dataset repo."""
    full_cache_key = (repo_id, split, None)
    if full_cache_key in _STAGED_CACHE:
        samples = _STAGED_CACHE[full_cache_key]
        return samples[:max_samples] if max_samples is not None else list(samples)

    cache_key = (repo_id, split, max_samples)
    if cache_key in _STAGED_CACHE:
        return list(_STAGED_CACHE[cache_key])

    from datasets import Audio, load_dataset

    logger.info("Loading %s split=%s from HuggingFace ...", repo_id, split)
    ds = load_dataset(repo_id, split=split)

    missing = _REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset {repo_id} split={split} is missing columns: {missing}"
        )

    ds = ds.cast_column("ref_audio", Audio(decode=False))
    if max_samples is not None:
        ds = ds.select(list(range(min(max_samples, len(ds)))))

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
        len(samples),
        len(written),
        repo_id,
        split,
    )
    return list(samples)
