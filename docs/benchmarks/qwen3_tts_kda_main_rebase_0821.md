# Qwen3-TTS main vs rebase standard benchmark (2026-08-21)

## Conclusion

On the fixed Qwen3-TTS workload below, the rebased retained stack reduces the
sync C1 wall-time median by 24.0%, the sync C1 engine-time median by 25.1%, and
the controlled C4 wall-time median by 18.5% relative to the bracketed original
main baseline.

All measured C1 outputs are byte-identical to original main. The controlled C4
endpoint fixes request arrival grouping but does not fix the internal scheduler
layout, so C4 hashes are recorded and are not used as a seeded correctness
assertion.

## Revisions

- Original main: `c6bbcc80163d912fcc9d1bff90003b5b21a66157`
- Rebased branch: `57ab62adf96fcc9342a79dab21f4a87eec38add2`
- Merge base: `c6bbcc80163d912fcc9d1bff90003b5b21a66157`

## Standard protocol

- GPU: one NVIDIA H100 80 GB
- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- PyTorch: `2.11.0+cu130`
- Transformers: `5.12.1`
- SGLang source: official `v0.5.16`, commit
  `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`
- `qwen-tts`: `0.1.1`
- `accelerate`: `1.12.0`
- Talker `torch.compile`: off
- Seed: `20260819`
- Repetition penalty: `1.05`
- One warmup followed by ten measured requests per arm
- C1: one request at a time
- Controlled C4: four requests submitted in one batch-endpoint arrival group

The main arm ran before and after the rebase arms. The baseline below is the
midpoint of those two main medians. This brackets execution-order drift without
hiding either observed main result.

## Sync comparison

| Metric | Main before | Main after | Main midpoint | Rebase sync | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 wall median | 1.126313 s | 1.113294 s | 1.119803 s | 0.851592 s | **-24.0%** |
| C1 engine median | 1.048811 s | 1.034956 s | 1.041884 s | 0.780665 s | **-25.1%** |
| Controlled C4 wall median | 1.318330 s | 1.347524 s | 1.332927 s | 1.086554 s | **-18.5%** |

## Async observations

These results are separate from the sync comparison.

- Forced C1 lookahead with `--async-lookahead-min-batch-size 1`: wall median
  `0.939375 s`; engine median `0.863127 s`.
- Default-threshold C4 async: wall median `1.130281 s`, or 15.2% below the
  bracketed main midpoint.
- In this one ten-sample run, the async C4 median is 4.0% above rebase sync,
  while the means differ by 0.7%. This does not establish an async regression.
  A crossed multi-round short/long workload is required for that verdict.

## Correctness

Every C1 arm produced this WAV SHA-256:

```text
c4d7988da4eac5065dd3cc90c05b27f6f73c5a85e18b52794450636e0cb961e2
```

All ten C1 repetitions inside every arm were byte-identical. This includes the
forced-lookahead C1 arm, so it exercises the asynchronous path with the default
`repetition_penalty=1.05` device-side handoff.

The benchmark client is `benchmarks/eval/benchmark_qwen3_tts_kda.py`. It records
each observed controlled-C4 hash layout, but it deliberately fails only on C1
byte divergence until a scheduler-layout-independent C4/C8 parity rule exists.

## Scope

This report replaces earlier end-to-end Qwen3-TTS performance conclusions for
the integrated branch. Earlier reports remain useful only for isolated kernel
or transfer-path attribution. The numbers here apply to this fixed workload and
do not establish a universal sync-versus-async policy or streaming TTFB result.
