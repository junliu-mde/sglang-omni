# Qwen3-TTS KDA GEMM round 0821c

## Verdict

Keep the async C4 correctness fix. The async lookahead path applied the
repetition penalty twice to the pending semantic token. The fix skips the
lookahead handoff when the incremental mask update already contains that token.

The residual cross-round WAV variation was not AR state. The non-streaming
vocoder formed timing-dependent terminal batches. Its default batched decoder
produced different floating-point WAV bytes for the same codec input at batch
sizes `1`, `3`, and `4`. A vocoder-only per-plan deterministic override removes
that unrelated variation without changing sync versus async AR decode.

The fixed tree passes controlled C4 `4/4`, forced-lookahead C1 `4/4`, modified
tests `6/6`, and the focused suite `335/335`. The unfixed revision fails the
same controlled C4 protocol `0/2`, so the gate discriminates the fix.

## Async root cause

An experiment-only output hook aligned the host-resolved semantic ID, all 16
Predictor codec IDs, and repetition-penalty state. At step `17`, raw logits are
bit-exact. Token `1537` starts at `31.625` in both arms. The normal mask penalty
changes it to `30.125` in both arms. Async then applies the lookahead handoff to
the same token and changes it to `28.75`. Sync samples `1537`; async samples
`1478`.

All async rows are correctly indexed. Each row records the expected pending row
`0..3`, host snapshot `[1537,1537,1537,1537]`,
`rep_mask_has_pending=true`, and `last_sampled_equals_pending=true`. Step `17`
is only the first seeded decision changed by applying penalty `1.05` twice. It
is not a configured depth or repetition window.

Commit `b06406f4` added the lookahead handoff to an earlier duplicate method
that Python replaced with the later mask implementation. Rebase repair
`9eaca843` extracted the helper and called it from the active implementation.
The incremental path already scatters `_mask_last_sampled` into the repetition
mask, so the helper penalized the same pending token again.

The fix names the incremental-mask condition and calls the lookahead helper only
after a full mask rebuild. Full rebuilds still need the handoff because their
host-side output history can precede the launched lookahead result.

Evidence:

- `runs/p2b_first_divergent_step_0821c.json`
- `runs/rootcause_state_diff_0821c.json`
- `runs/p2b_sync_step_trace_0821e.jsonl`
- `runs/p2b_async_step_trace_0821e.jsonl`

## Cross-round mechanism

The final boundary is after token generation. Two measured sync C4 groups have
identical semantic and all 16 codec IDs for all 102 generated steps, but their
WAV hashes differ.

The non-streaming vocoder waits up to 2 ms for terminal payloads, then decodes
the collected payloads together. A traced default run formed `3+1` groups:
the three-row call produced `ba3f376e...` for each item and the one-row call
produced `cdf456b7...`. All codec inputs had shape `[150,16]` and SHA-256
`b7eb8a924c2ba23d6c0190c8e294bbb027c72e22923f343c4d05a4e13cf3e298`.
A temporary 20 ms wait forced batches of four and produced `de8fe578... x4`.
The codec hash did not change.

With the production vocoder's per-plan deterministic option, the outer
scheduler still formed `3+1` and `2+2` groups, but every item produced
`cdf456b7...` in all five rounds. This proves that vocoder batch composition,
not cache or sampling state, caused the cross-round WAV variation.

The rejected hypotheses have direct discriminators:

- Prefix cache: all six C4 groups report `#new-token: 232` and
  `#cached-token: 0`; all 24 request lifetimes use distinct `extra_key` values.
- Seed phase: every round starts with semantic seed `2103323860`, subtalker seed
  `1031793848`, position `57`, and the same seed-state hash. Seeded sampling is
  positional and has no mutable draw counter.
- Reused request state: all 24 request IDs are distinct. Every row starts with
  zero output IDs, zero codec chunks, zero pending feedback, and no pending
  lookahead token.

Full evidence is `runs/crossround_mechanism_0821c.md`.

## Final gate protocol

- Base revision: `57ab62adf96fcc9342a79dab21f4a87eec38add2`.
- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
- Seed: `20260819`.
- Repetition penalty: `1.05`.
- Talker `torch.compile`: off.
- Sync and async use the same vocoder-only override:
  `enable_deterministic_inference=true` and `initial_cuda_graph=false`.
- Controlled admission holds AR prefill until four requests are ready. Each arm
  logs six single prefill batches of four, including warmup, with 232 new tokens
  and zero cached tokens.
- Each fixed arm runs five measured rounds. Iteration `0` is discarded.
  Iterations `1-4` are compared as sorted four-WAV SHA-256 multisets.
- The unfixed revision runs three measured rounds per arm under the same
  protocol. Iteration `0` is discarded; iterations `1-2` test discrimination.

## Final gate result

Each multiset entry uses `xN` for multiplicity.

| Retained iteration | Fixed sync | Fixed async | Exact |
| ---: | --- | --- | --- |
| 1 | `cdf456b7... x4` | `cdf456b7... x4` | Yes |
| 2 | `cdf456b7... x4` | `cdf456b7... x4` | Yes |
| 3 | `cdf456b7... x4` | `cdf456b7... x4` | Yes |
| 4 | `cdf456b7... x4` | `cdf456b7... x4` | Yes |

| Unfixed retained iteration | Sync | Async | Exact |
| ---: | --- | --- | --- |
| 1 | `cdf456b7... x4` | `3e964a7a... x4` | No |
| 2 | `cdf456b7... x4` | `3e964a7a... x4` | No |

| Gate | Result |
| --- | --- |
| Modified mask tests | `6/6` passed |
| Focused suite | `335/335` passed |
| Forced-lookahead C1 | `4/4` exact |
| Fixed controlled C4 | `4/4` exact |
| Unfixed controlled C4 | `0/2` exact |

Final raw evidence:

- `runs/final_gate_fixed_sync_raw_0821e.json`
- `runs/final_gate_fixed_async_c4_raw_0821e.json`
- `runs/final_gate_fixed_async_c1_raw_0821e.json`
- `runs/final_gate_unfixed_sync_raw_0821e.json`
- `runs/final_gate_unfixed_async_c4_raw_0821e.json`
- `runs/final_mask_tests_0821e.log`
- `runs/final_focused_335_0821e.log`

## P3 profile retained for later work

The compile-off retained stack remains host-limited:

| Metric | Median per step |
| --- | ---: |
| Step wall time | 10.293642 ms |
| GPU busy time | 5.545372 ms |
| GPU busy fraction | 53.941850% |
| Kernel launches | 1672 |
| Predictor GEMM total | 2.087069 ms |
| QKV split-K | 0.543158 ms, 160 launches |
| Gate/up plus separate SiLU-mul | 0.529632 ms |
| Attention output projection | 0.431568 ms |
| MLP down projection | 0.417819 ms |

The profile remains valid evidence for a later P4 round after service parity is
repaired. GEMM kernel or split-K changes remain unsuitable for a bit-identity
candidate because they change BF16 reduction order. The next performance lever
after parity repair is host-side graph-node or submit-cost reduction that does
not change FLOP order.

Full profile evidence is `runs/profile_summary_0821c.md`.
