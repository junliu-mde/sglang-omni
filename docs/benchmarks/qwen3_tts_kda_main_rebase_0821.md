# Qwen3-TTS main vs rebase standard benchmark (2026-08-21)

## Conclusion

On the fixed Qwen3-TTS workload below, the rebased sync stack reduces the C1
wall-time median by 20.9%, the C1 engine-time median by 21.9%, the controlled C4
wall-time median by 18.2%, and the controlled C8 wall-time median by 9.3%
relative to the bracketed original-main baseline.

The exact-output gate passes on the standard matrix: C1 is byte-identical, and
every one of the 40 C4 and 80 C8 candidate WAV hashes occurs in the original-main
reference set. The Cn gate compares each WAV independently. It does not pair
runs or scheduler layouts and does not use an audio tolerance.

Crossed multi-round tests do not support enabling async lookahead by default.
Async is 4.0% slower on short C4 and 8.1% slower on short C8. The long C4 and C8
changes are +1.0% and +0.4%. A separate threshold-9 isolation shows that the
async server configuration without lookahead changes C4 by only -0.6%. This
rules out the server configuration as the cause of the C4 regression. The C8
isolation changes latency by +1.1%, but its exact-output gate fails, so it is
directional evidence only and does not establish C8 attribution.

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

## Exact-output gate

Every C1 arm produced this WAV SHA-256:

```text
c4d7988da4eac5065dd3cc90c05b27f6f73c5a85e18b52794450636e0cb961e2
```

All ten C1 repetitions in every arm are byte-identical.

For C4 and C8, scheduler timing can change the run-level hash multiset even
under controlled endpoint arrival. The correctness rule is therefore:

1. Build a reference set from every exact WAV SHA-256 observed in the
   corresponding original-main reports.
2. Require every candidate WAV SHA-256 to occur in that set.
3. Fail on any unseen hash. Do not pair runs or scheduler layouts.
4. Retain each run-level hash multiset only as a layout observation.
5. Do not use numeric or audio tolerance.

The rebase result is 40/40 exact C4 hashes and 80/80 exact C8 hashes matched to
the original-main reference sets.

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

C4 passes the exact-output gate at 40/40. C8 matches 75/80 WAV hashes; five WAVs
contain one of two hashes not present in the available main reference set. This
is a correctness-gate failure for the isolation arm. It does not change the
standard sync result above, but it prevents a C8 parity claim for this isolation.
It also prevents using the C8 isolation as definitive performance attribution.

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
