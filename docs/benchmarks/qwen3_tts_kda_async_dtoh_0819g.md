# Qwen3-TTS A1/B 异步 token DtoH 验证

日期：2026-08-19。

## 结论

保留 Qwen3-TTS 的 semantic token（语义 token）暂存路径。

Qwen runner 在 Talker forward 返回 CUDA semantic token 后，且在 Predictor
开始计算前，将 token 复制到 pinned host buffer（锁页主机缓冲区）。复制在专用
CUDA stream 上执行。Predictor 继续在 decode stream 上运行。输出阶段只等待
copy-complete event，然后让 output processor 读取 CPU tensor。

H100 trace 显示，该改动把目标的 4 B token 读回从 pageable DtoH（可分页主机
内存的 device-to-host 复制）变为 pinned DtoH。未暂存路径的相关
`cudaMemcpyAsync` CPU 调用 p95 是 `4.438 ms`。暂存路径的 p95 是 `16.922 us`。
暂存路径的 `cudaEventSynchronize` p95 是 `19.882 us`，没有重新引入毫秒级等待。

同一组预热后的相邻 S→U C1 对照中，暂存路径的 engine-time 中位数是
`909.836 ms`，未暂存路径是 `972.996 ms`。前者低 `6.49%`。这个数字只覆盖本报告
的固定 workload，不能外推为所有并发和所有请求的服务 SLO。

## 实现

`Qwen3TTSModelRunner` 在三个边界调用 `_stage_token_ids()`：

- `_prepare_and_forward()`：在 Talker forward 后立即发起复制。这是同步 decode
  的关键调用点，因此复制可与 Predictor 重叠。
- `_ensure_next_token_ids()`：覆盖未来在 post hook 后才物化 token 的路径。
- `post_decode_launch()`：覆盖 async decode 路径和 post hook 内采样。

`ModelRunner._stage_token_ids()` 使用两个 ping-pong pinned buffer。它记录 producer
event，让专用 copy stream 等待该 event，再执行非阻塞复制。下一次 decode 只在 GPU
上等待 copy-complete event，因此 CUDA Graph 的可复用 sampler 输出不会与复制读操作
竞争。`_resolve_host_token_ids()` 在 output processor 读取 tensor 前等待同一个 event。

本次 DtoH 代码不调用 `torch.compile` 或 Inductor。

CUDA Graph capture 期间不能建立这个外部 stream 依赖。`_stage_token_ids()` 在
capture 中保留正常 GPU reporting 路径；实际 replay 在 capture 外执行暂存。

## 正确性

工作负载是固定 seed `20260819` 的英文 ICL 声音克隆。模型为
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`。每个 C1 请求生成 110 个 completion token。

暂存和未暂存路径的所有 C1 测量结果都生成相同的 WAV：

| 项目 | SHA-256 |
| --- | --- |
| C1 WAV | `e94c65acd1c6a080de243dae9db697ab2aaa400ef3e8f62d0eed7075ca206b91` |

现有 Predictor 测试覆盖 codec ID 和 summed embedding（各 codec embedding 的累加值）
的行归属和 snapshot。此次 H100 定向测试还覆盖了：

- copy stream 与 decode stream 分离；
- CUDA token 暂存后与参考 CPU token 相等；
- 下一个 decode 只插入 GPU event 依赖；
- Qwen runner 在 forward 返回 CUDA semantic token 后调用暂存函数。

命令：

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 python -m pytest -q \
  tests/unit_test/model_runner/test_base_hooks.py \
  tests/unit_test/qwen3_omni/test_talker_token_readback.py \
  tests/unit_test/qwen3_tts/test_pipeline.py \
  -k 'test_prepare_and_forward_orders_staged_copy_before_decode_forward or test_stage_token_ids_uses_pinned_buffer_and_dedicated_copy_stream or test_stage_token_ids_cuda_matches_reference or test_qwen3_tts_stages_semantic_ids_at_forward_boundary'
```

结果：`4 passed, 96 deselected`。

C4 的 batch endpoint 不能保证相同的内部 scheduler layout。部分 C4 波次的 WAV
哈希随 layout 改变。因此 C4 只记录吞吐行为，不作为跨进程 bit-identity 判定。

## Trace 证据

两个 trace 都只向 `tts_engine` 的 profiler control socket 发消息。每个 trace 使用
1 次 C1 warmup、2 次 C1 测量、1 次 C4 warmup 和 2 次 C4 测量。trace 的开始和停止
位置不同，所以复制次数不能直接相减。下表只比较相同类型复制的每次 CPU API 时长。

| 路径 | 4 B DtoH 类型 | 调用数 | CPU API p50 | CPU API p95 |
| --- | --- | ---: | ---: | ---: |
| 未暂存 | Device → Pageable | 674 | 1.110 ms | 4.438 ms |
| 暂存 | Device → Pinned | 326 | 13.183 us | 16.922 us |

暂存 trace 中有 632 次 `cudaEventSynchronize`。其 p50 为 `7.708 us`，p95 为
`19.882 us`，最大值为 `65.445 us`。这表明 output resolve 等待的是已完成或接近
完成的 copy event，而不是原来的 decode stream 长等待。

未暂存 trace 的所有 DtoH 中，pageable 复制为 1,331 次，pinned 复制为 33 次。
暂存 trace 中，pageable 复制为 47 次，pinned 复制为 697 次。剩余的大型 pageable
复制属于其它运行时工作，不是这个 4 B reporting-token 路径。

## 延迟复测

完整 S→U→S 序列显示，重启后的 graph 和服务状态会显著影响绝对延迟。因此早期的
五次测量不用于性能结论。随后使用十次 C1、一次 warmup 的相邻 S→U 对照：

| 路径 | C1 engine-time 中位数 | C1 wall-time 中位数 | C1 WAV |
| --- | ---: | ---: | --- |
| 暂存 | 909.836 ms | 980.040 ms | 相同 |
| 未暂存 | 972.996 ms | 1,042.246 ms | 相同 |

未暂存服务因上一个 Uvicorn listener 的延迟释放而选择了随机 loopback port。表中的
`engine-time` 来自服务响应头，不包括客户端到该端口的连接成本。为避免把 scheduler
layout 差异解释为性能，C4 数字不用于此结论。

## 环境和证据

- Pod：`<internal-pod>`。
- Kubernetes context：`<internal-cluster>`。节点：`<internal-node>`。
- GPU：NVIDIA H100 80 GB，driver `590.48.01`。
- 源提交：`22faf0a7fb8a25e32d99006aff0c3ed3065aabe4`。
- 运行时：SGLang `0.5.17`、Torch `2.11.0+cu130`、Transformers `5.12.1`、
  qwen-tts `0.1.1`。
- 原始 JSON、服务日志、event JSONL 和 trace 位于：
  `<internal-evidence-path>`。
- 关键 trace：`trace/unstaged_tts_trace/tts_engine/trace_pid10006_rank0.trace.json.gz`
  和 `trace/staged_tts_trace/tts_engine/trace_pid10986_rank0.trace.json.gz`。

CUDA Docs MCP 已对 `cudaMemcpyAsync`、pinned host memory、event 和 stream wait
语义发起查询。该服务返回 HTTP 500，因此本报告不把 MCP 响应作为证据。结论只基于
当前代码、H100 trace、固定 seed 输出和 H100 测试。
