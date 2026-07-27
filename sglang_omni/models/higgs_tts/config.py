# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs TTS (V1)."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)
from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.utils.cpu import bounded_intraop_threads

_PKG = "sglang_omni.models.higgs_tts"
_PREPROCESS_MAX_WORKERS = 2
_MAX_PIPELINE_INTRAOP_THREADS = 8


class HiggsTtsPipelineConfig(PipelineConfig):
    """4-stage TTS pipeline: preprocessing → audio_encoder → tts_engine → vocoder.

    Mirrors the V0 layout: preprocessing tokenises text + delay-pattern-encodes
    the reference audio codes; audio_encoder runs the fused multi-codebook
    embedding once on the delayed ref codes (CPU- or GPU-side); tts_engine
    drives the AR loop on the sglang backbone with the precomputed embed
    pasted at ``-100`` placeholder positions; vocoder reverses the delay
    pattern and decodes to waveform via the higgs-audio-v2-tokenizer codec.

    Preprocessing and audio_encoder share a dedicated process so reference
    waveforms stay on the local handoff while their host work is isolated from
    the autoregressive engine.
    """

    architecture: ClassVar[str] = "HiggsMultimodalQwen3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="tts_frontend",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"max_concurrency": _PREPROCESS_MAX_WORKERS},
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="tts_frontend",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.03)
            ),
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={
                "device": "cuda",
                "max_new_tokens": 2048,
                "enable_async_decode": True,
            },
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.85)
            ),
            next="vocoder",
            stream_to=["vocoder"],
            project_payload={
                "vocoder": (f"{_PKG}.request_builders.project_tts_engine_to_vocoder")
            },
        ),
        StageConfig(
            name="vocoder",
            process="vocoder",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={
                "device": "cuda",
                "compile_decode": False,
                # Before the steady cursor is established, a decode window is
                # bounded by the default 75-row stride plus its 75-row
                # follow-up. Capture that complete finite domain so terminal
                # flushes cannot silently fall back to eager execution.
                "decode_cuda_graph_frame_counts": tuple(range(1, 151)),
            },
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.10)
            ),
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if "OMP_NUM_THREADS" not in self.env_defaults:
            preprocessing = next(
                stage for stage in self.stages if stage.name == "preprocessing"
            )
            factory_args = resolve_stage_static_factory_args(preprocessing, self)
            preprocess_workers = max(
                int(factory_args.get("max_concurrency", _PREPROCESS_MAX_WORKERS)),
                1,
            )
            # The frontend uses a bounded preprocessing pool, but its CPU torch
            # work can still spawn a machine-sized OMP team under a cgroup CPU
            # quota and contend with the separately spawned AR and vocoder
            # processes. The env must be in place before stage processes import
            # Torch.
            self.env_defaults["OMP_NUM_THREADS"] = str(
                bounded_intraop_threads(
                    worker_count=preprocess_workers,
                    max_threads=_MAX_PIPELINE_INTRAOP_THREADS,
                )
            )
        stages = {stage.name: stage for stage in self.stages}
        vocoder = stages["vocoder"]
        tts_engine = stages["tts_engine"]
        vocoder_overrides = self.runtime_overrides.get("vocoder", {})
        tts_engine_overrides = self.runtime_overrides.get("tts_engine", {})
        missing = object()
        for key in ("stream_stride", "stream_followup_stride"):
            value = vocoder_overrides.get(key, vocoder.factory_args.get(key, missing))
            if value is missing:
                if key in tts_engine.factory_args or key in tts_engine_overrides:
                    raise ValueError(
                        f"Higgs TTS {key!r} must be configured on the vocoder stage"
                    )
                continue
            if key in tts_engine_overrides and tts_engine_overrides[key] != value:
                raise ValueError(
                    f"Higgs TTS {key!r} runtime overrides must match between "
                    "the tts_engine and vocoder stages"
                )
            tts_engine.factory_args[key] = value

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return True

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = HiggsTtsPipelineConfig
