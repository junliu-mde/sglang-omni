# Qwen3-TTS main vs rebase standard benchmark (2026-08-21)

## Conclusion

On the fixed Qwen3-TTS workload below, the rebased sync stack reduces the C1
wall-time median by 20.9%, the C1 engine-time median by 21.9%, the controlled C4
wall-time median by 18.2%, and the controlled C8 wall-time median by 9.3%
relative to the bracketed original-main baseline.

The standard matrix observes byte-identical C1 output and main-reference
membership for all 40 C4 and 80 C8 rebase WAVs. An expanded C8 audit shows that
finite hash-set membership is not a complete parity gate: a main-equivalent
repeat also produces a hash absent from the earlier main set. A separate
correctness-only run fixes both the C8 prefill layout and vocoder execution. It
passes 160/160 exact WAV comparisons between original main and the rebase.

Crossed multi-round tests do not support enabling async lookahead by default.
Async is 4.0% slower on short C4 and 8.1% slower on short C8. The long C4 and C8
changes are +1.0% and +0.4%. A separate threshold-9 isolation shows that the
async server configuration without lookahead changes C4 by only -0.6%. This
rules out the server configuration as the cause of the C4 regression. The C8
isolation changes latency by +1.1%. Its five hashes outside the earlier main set
also occur on the synchronous Predictor branch, while a main-equivalent repeat
independently falls outside that set. The isolation therefore cannot attribute
the C8 change to async.

## Revisions and protocol

- Original main: `c6bbcc80163d912fcc9d1bff90003b5b21a66157`
- Measured rebase implementation: `57ab62adf96fcc9342a79dab21f4a87eec38add2`
- Benchmark correctness client: `fda1aaf21d13aef2116b88317a1d7cda10a173f6`
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
- One warm-up followed by ten measured requests per arm
- C1: one request at a time
- Controlled Cn: C requests submitted through one batch-endpoint arrival group

The standard matrix ran main before the rebase arm and main again after it. The
baseline is the midpoint of the two main medians. This brackets execution-order
drift while retaining both observed main results.

## Main vs rebase

| Metric | Main before | Main after | Main midpoint | Rebase sync | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 wall median | 1.136859 s | 1.131399 s | 1.134129 s | 0.897624 s | **-20.9%** |
| C1 engine median | 1.063675 s | 1.051345 s | 1.057510 s | 0.826418 s | **-21.9%** |
| Controlled C4 wall median | 1.382967 s | 1.371367 s | 1.377167 s | 1.126911 s | **-18.2%** |
| Controlled C8 wall median | 1.522798 s | 1.460790 s | 1.491794 s | 1.352928 s | **-9.3%** |

## Output checks

Every C1 arm produced this WAV SHA-256:

```text
c4d7988da4eac5065dd3cc90c05b27f6f73c5a85e18b52794450636e0cb961e2
```

All ten C1 repetitions in every arm are byte-identical.

For C4 and C8, scheduler timing can change the run-level hash multiset even
under one batch-endpoint arrival group. The benchmark client's finite-reference
check is:

1. Build a reference set from every exact WAV SHA-256 observed in the
   corresponding original-main reports.
2. Require every candidate WAV SHA-256 to occur in that set.
3. Report any unseen hash. Do not pair runs or scheduler layouts.
4. Retain each run-level hash multiset only as a layout observation.
5. Do not use numeric or audio tolerance.

The standard rebase result is 40/40 C4 hashes and 80/80 C8 hashes matched to the
available original-main sets. This is an observed result, not proof that the
sets contain every output reachable under later scheduler and vocoder layouts.

### Expanded C8 audit

The expanded reference is the union of the standard main-before and main-after
runs plus 100 additional original-main C8 runs. It contains 12 hashes.

| Arm | Serving revision | WAVs matching the finite set | Unseen WAVs |
| --- | --- | ---: | ---: |
| Main-equivalent repeat | `91d4359fa06d1b60834250624116318ffa448e18` | 799/800 | 1 |
| Rebase sync | `57ab62adf96fcc9342a79dab21f4a87eec38add2` | 794/800 | 6 |
| Threshold-9, lookahead bypassed | `57ab62adf96fcc9342a79dab21f4a87eec38add2` | 75/80 | 5 |

Revision `91d4359` differs from original main `c6bbcc8` only in release metadata
and documentation. It has no serving-code change. Its unseen hash is therefore
a same-code false negative for finite membership. The rebase rows above cannot
establish either parity or a regression without fixing all output-affecting
layouts.

### Fixed-layout C8 parity

The correctness-only protocol fixes those layouts:

- Hold prefill until all eight requests are waiting, with a 500 ms escape
  deadline. Server logs confirm one 8-request prefill for every measured run.
- Decode each vocoder item independently with deterministic vocoder inference
  and disable the initial vocoder CUDA Graph.
- Keep Talker `torch.compile` off and its C8 decode CUDA Graph enabled.
- Run one warm-up and 20 measured C8 batches per arm.

Original main and final rebase each produce the same single WAV SHA-256 for all
160 measured WAVs:

```text
da3fc6523d8ffe2dd3b0cfd303de1841238582c129520609eed9d13b396ab98d
```

The rebase result is 160/160 byte-identical WAVs against original main. The
prefill hold was injected only into the two experiment worktrees because the
public CLI does not expose that scheduler option for Qwen3-TTS. It is not part
of the measured implementation or any pushed branch. This protocol is for
correctness only; its added wait and deterministic vocoder are excluded from
the performance table.

The benchmark client is `benchmarks/eval/benchmark_qwen3_tts_kda.py`.

## Crossed sync and async

Each arm has five rounds. Round one is discarded as warm-up. The table reports
the median of the round medians from rounds two through five.

| Workload | Sync | Async | Async change |
| --- | ---: | ---: | ---: |
| Short C4 | 1.128452 s | 1.173869 s | **+4.0%** |
| Short C8 | 1.402605 s | 1.516409 s | **+8.1%** |
| Long C4 | 2.029622 s | 2.050231 s | +1.0% |
| Long C8 | 2.373556 s | 2.382903 s | +0.4% |

The stable short C8 regression rejects async as the default for this workload.

### Threshold-9 isolation

The server uses `--decode-mode async --async-lookahead-min-batch-size 9`.
C4 and C8 therefore use synchronous decode while retaining the async server
configuration.

| Workload | Rebase sync | Async configuration, lookahead bypassed | Change |
| --- | ---: | ---: | ---: |
| Short C4 | 1.126911 s | 1.119735 s | -0.6% |
| Short C8 | 1.352928 s | 1.368371 s | +1.1% |

C4 matches the available main set at 40/40. C8 matches 75/80 WAV hashes; five
WAVs contain one of two hashes not present in that finite set. The synchronous
Predictor PR reproduces the same five-WAV pattern, and the main-equivalent audit
shows that the finite set can reject unchanged serving code. This result does
not implicate async, and it cannot provide definitive C8 performance
attribution. The fixed-layout check above supplies the separate parity result.

## Latest sync vs async profile

The final profile uses short C4, repetition penalty 1.05, and `torch.compile`
off for both arms.

| Metric | Sync | Async |
| --- | ---: | ---: |
| Profile request wall time | 1.534892 s | 1.669495 s |
| GPU kernel time | 677.956 ms | 679.668 ms |
| Kernel count | 189,404 | 190,119 |
| `cudaStreamSynchronize` | 207 calls / 405.525 ms | 508 calls / 451.511 ms |
| `cudaGraphLaunch` | 216 calls / 296.303 ms | 216 calls / 239.432 ms |
| Total CUDA runtime API time | 826.505 ms | 796.863 ms |

Scheduler event spans, averaged per request:

| Span | Sync | Async |
| --- | ---: | ---: |
| Request build | 4.211 ms | 2.928 ms |
| Build end to queue | 12.041 ms | 8.217 ms |
| Queue to prefill | 4.479 ms | 5.121 ms |
| Prefill | 54.347 ms | 38.536 ms |
| Prefill end to completion | 1,096.154 ms | 1,028.746 ms |

These spans are elapsed event intervals. They are not pure Python CPU time. The
Torch traces contain no `cpu_op` or `python_function` events. A separate py-spy
capture sampled GIL-holding Python stacks at 20 Hz, or one sample every 50 ms.
It locates Python paths only. It cannot support a reliable per-request Python
CPU-time value, so no sampled duration is reported as CPU time.

## Predictor attribution

Both C4 traces contain 109 Predictor graphs with 1,271 kernels per graph and
107 Talker graphs.

| Predictor metric | Sync median | Async median |
| --- | ---: | ---: |
| Kernel sum per graph | 3.731537 ms | 3.734813 ms |
| Graph-launch CPU API time | 1.857554 ms | 1.398711 ms |

Predictor GEMM time per graph:

| Site | Logical calls | GPU launches | Sync | Async | Checkpoint N x K |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV projection | 80 | 160 | 0.549374 ms | 0.548377 ms | 4096 x 1024 |
| Gate/up projection | 80 | 80 | 0.532072 ms | 0.532540 ms | 6144 x 1024 |
| Attention output projection | 80 | 80 | 0.431034 ms | 0.431491 ms | 1024 x 2048 |
| MLP down projection | 80 | 80 | 0.417048 ms | 0.417180 ms | 1024 x 3072 |
| Input projection | 17 | 17 | 0.098944 ms | 0.099008 ms | 1024 x 2048 |
| Per-group LM head | 15 | 15 | 0.065858 ms | 0.066014 ms | 2048 x 1024 |

The checkpoint dimensions are BF16. The C4 trace does not contain CPU shape
events, so it does not establish GEMM M. The earlier C1 calibration established
M=1 only for that separate C1 run.

## PCM streaming TTFB

The final rebase sync server was measured with `stream=true` and
`response_format=pcm`.

| Metric | Result |
| --- | ---: |
| TTFB median | 0.128572 s |
| TTFB mean | 0.128810 s |
| TTFB min | 0.112043 s |
| TTFB max | 0.172498 s |
| Total latency median | 1.144283 s |
| Total latency mean | 1.126537 s |
| Bytes per run | 410,880 |

## Scope

This report replaces earlier end-to-end and profile conclusions for the
integrated Qwen3-TTS branch. Earlier reports remain useful only for isolated
kernel or transfer-path history. These results apply to the fixed workloads and
protocols above. They do not establish a universal performance policy.
