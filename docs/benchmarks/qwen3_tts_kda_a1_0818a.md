# Qwen3-TTS A1 KDA 结果

日期：2026-08-18。

## 结论

不提升 A1。A1 将 Qwen3-TTS 的语义 token 报告路径从 pageable DtoH
复制改为 pinned DtoH 复制。它保持了输出正确性，也消除了每步的
pageable 报告复制。但是 C1 的端到端收益不超过测量噪声。因此此分支
不包含 A1 的 `_stage_token_ids()` 调用。

本次保留两个独立的兼容性修复：

- Qwen-TTS 0.1.1 使用旧的 `input_embeds` 参数名。Transformers 5.12
  使用 `inputs_embeds`。启动时的兼容层转换该参数，并删除当前 helper
  不接受的 `cache_position` 参数。
- 当前 `flashinfer==0.6.14` 会解析到
  `flashinfer-cubin==0.6.15.post1`。cookbook 要求在启动前设置
  `FLASHINFER_DISABLE_VERSION_CHECK=1`。

## 测量设计

- 模型：`Qwen/Qwen3-TTS-12Hz-1.7B-Base`。
- 负载：固定 seed 的英文 ICL 声音克隆。输出为 24 kHz WAV。
- 硬件：HPC3 的 1x H100 80 GB，GPU UUID
  `<gpu-uuid>`，driver `590.48.01`。
- 镜像：`lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155`。
- 运行时：SGLang 0.5.16、Torch 2.11.0+cu130、Transformers 5.12.1、
  qwen-tts 0.1.1。
- 基线：`6a797b8a30f1f9c914a8ab8573d72597350f39a7`，加上两个兼容性修复。
- 候选：仅在 Qwen3-TTS 的 `post_prefill` 和 `post_decode` 中，在
  `_collect_codes()` 后调用已有的 `_stage_token_ids()`。

每个 C1 请求在同一服务进程中交替使用候选和基线。每个方向有 20 对。
正值表示 `baseline latency - candidate latency`，即候选更快。C4 和 C8
以整个并发波次的平均请求延迟配对。

## 正确性

同一进程中先预热一次，再交替运行 8 个 C1 请求。所有测量请求成功。
候选和基线的语义 token、codec token 和 WAV 哈希完全相同。

| 项目 | SHA-256 |
| --- | --- |
| 语义 token | `5c5eb77e2a1c1079635f944825a5e1faf2efc6b6198ce01fa67e3937be4c957d` |
| codec token | `276256410d874e07de6e05ac8c195a215a68e6cfe0ebb0322a5b09546c73e660` |

跨进程固定 seed 的输出不稳定。因此跨进程结果没有用于正确性判定。

正式源文件的 H100 冒烟请求也成功。它返回非空 24 kHz WAV，时长
19.6 s，端到端延迟 2.692 s。

## 传输轨迹

每个轨迹采集一个预热后的 C1 请求。剩余的三个 pageable DtoH 事件是
大的辅助复制，不是逐步的 token 报告复制。

| 路径 | Pageable DtoH | Pinned DtoH |
| --- | ---: | ---: |
| 基线 | 501 次，2,562,632 B | 3 次，10 B |
| A1 候选 | 3 次，2,560,640 B | 252 次，1,006 B |

这证明 A1 改变了目标传输路径。

## 延迟结果

| 并发 | 配对数 | 平均收益 | 配对标准差 | 两倍噪声带 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| C1 | 40 | +14.87 ms | 73.75 ms | 147.50 ms | 不通过提升门槛 |
| C4 | 15 波次 | +60.04 ms | 346.27 ms | 692.53 ms | 无可检出的回归 |
| C8 | 12 波次 | -4.81 ms | 266.21 ms | 532.42 ms | 无可检出的回归 |

C1 的候选先运行和基线先运行结果分别为 +18.38 ms 和 +11.37 ms。
两种顺序都远小于 147.50 ms 的门槛。

## 原因和下一步

`_stage_token_ids()` 发起了 pinned 异步复制，但同一解码步中的
`_resolve_host_token_ids()` 立即等待 CUDA event。A1 因此移除了
pageable 复制，却没有创建主机和 GPU 的步间重叠。它不能单独产生可推广
的端到端收益。

下一轮应单独评估异步 decode。该设计必须保留 GPU token 通路用于下一次
forward，并让已完成的 pinned host snapshot 在下一步 GPU 工作期间完成。
它需要新的正确性和调度测试，不能与 A1 的兼容性修复混合。

开发计划如下：

1. 为 Qwen3-TTS 定义 launch 和 resolve 两个 decode 阶段。launch 只发布
   GPU token 和 pinned host copy。resolve 只读取已经完成的 host copy。
2. 在单并发交替测试中比较语义 token、codec token、EOS 和 WAV 哈希。
   必须覆盖最后一个语义 token 和空 batch。
3. 重新采集 C1 的传输轨迹。确认 host copy 与下一次 forward 重叠，而不是
   仅把同步从 pageable 复制改到 pinned 复制。
4. 使用与本次相同的 C1、C4、C8 交替实验。只有 C1 收益超过两倍噪声带，
   并且 C4/C8 不回归时才提升。

## 验证

定向 H100 测试命令如下。结果为 `92 passed`：

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_pipeline.py \
  tests/unit_test/qwen3_omni/test_talker_row_ownership.py \
  tests/unit_test/qwen3_omni/test_talker_token_readback.py
```
