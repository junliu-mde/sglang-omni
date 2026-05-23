from __future__ import annotations

import sys
import types
from pathlib import Path

from benchmarks.dataset import prepare, seedtts


class _FakeDataset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []
        self.selected_indices: list[int] | None = None

    def cast_column(self, _name: str, _audio_spec) -> "_FakeDataset":
        return self

    def select(self, indices: list[int]) -> "_FakeDataset":
        self.selected_indices = list(indices)
        return _FakeDataset([self._rows[i] for i in indices])

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


def test_download_dataset_prewarms_all_mmmu_configs(monkeypatch) -> None:
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_get_dataset_config_names(repo_id: str) -> list[str]:
        assert repo_id == "MMMU/MMMU"
        return ["Accounting", "Math"]

    def fake_load_dataset(
        repo_id: str, config_name: str | None = None, split: str | None = None
    ):
        calls.append((repo_id, config_name, split))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=fake_get_dataset_config_names,
            load_dataset=fake_load_dataset,
        ),
    )

    prepare.download_dataset("MMMU/MMMU", quiet=True)

    assert calls == [
        ("MMMU/MMMU", "Accounting", "validation"),
        ("MMMU/MMMU", "Math", "validation"),
    ]


def test_load_seedtts_samples_accepts_local_meta_lst(tmp_path: Path) -> None:
    meta_dir = tmp_path / "en"
    meta_dir.mkdir()
    ref_audio = meta_dir / "ref.wav"
    ref_audio.write_bytes(b"wav")
    meta_path = meta_dir / "meta.lst"
    meta_path.write_text(
        "sample-1|hello|ref.wav|target one\nsample-2|world|ref.wav|target two\n"
    )

    samples = seedtts.load_seedtts_samples(str(meta_path), max_samples=1)

    assert len(samples) == 1
    assert samples[0] == seedtts.SampleInput(
        sample_id="sample-1",
        ref_text="hello",
        ref_audio=str(ref_audio),
        target_text="target one",
    )


def test_load_seedtts_samples_stages_only_selected_rows(
    monkeypatch, tmp_path: Path
) -> None:
    seedtts._STAGED_CACHE.clear()

    rows = [
        {
            "sample_id": f"sample-{idx}",
            "ref_text": f"ref-{idx}",
            "ref_audio_path": f"audio/{idx}.wav",
            "target_text": f"target-{idx}",
            "ref_audio": {"bytes": f"audio-{idx}".encode()},
        }
        for idx in range(5)
    ]
    dataset = _FakeDataset(rows)
    stage_dir = tmp_path / "seedtts_stage"
    stage_dir.mkdir()

    def fake_load_dataset(repo_id: str, split: str):
        assert repo_id == "zhaochenyang20/seed-tts-eval-arrow"
        assert split == "en"
        return dataset

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setattr(seedtts.tempfile, "mkdtemp", lambda prefix: str(stage_dir))
    monkeypatch.setattr(seedtts.atexit, "register", lambda *args, **kwargs: None)

    samples = seedtts.load_seedtts_samples(
        "zhaochenyang20/seed-tts-eval-arrow",
        max_samples=2,
        split="en",
    )

    assert dataset.selected_indices == [0, 1]
    assert [sample.sample_id for sample in samples] == ["sample-0", "sample-1"]
    assert sorted(
        path.relative_to(stage_dir).as_posix() for path in stage_dir.rglob("*.wav")
    ) == [
        "audio/0.wav",
        "audio/1.wav",
    ]

    seedtts._STAGED_CACHE.clear()
