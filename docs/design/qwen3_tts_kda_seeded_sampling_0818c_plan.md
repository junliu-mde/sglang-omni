# Qwen3-TTS Predictor Seeded Sampling KDA 计划

## Goal Description

在 Qwen3-TTS Predictor 中实现一个单次 Triton 启动的有界 seeded
top-k、top-p 和 Gumbel sample 路径。该路径读取原始 logits 并直接返回
codec ID。它不使用 `torch.compile` 或 Inductor。

该工作只优化所有行都执行采样、词表大小为 2048 的 CUDA Graph 路径。图签名的
`max_top_k` 必须属于现有 Predictor ladder：`4, 8, 16, 32, 50, 64, 128, 256,
512, 1024`。每行真实 `top_k` 可以是 `1..max_top_k`。full-sort、混合采样和
argmax 继续使用当前参考路径。

## KDA 合同

- **K（优化单元）**：对每个 Predictor codec group 的 `[batch, 2048]`
  logits 执行温度缩放、有界 top-k、可选 top-p、按 `(seed, position)`
  生成 Gumbel 值，并写出一个 `int64` codec ID。
- **R（参考实现）**：
  `Qwen3TTSTalker._sample_subtalker_token_seeded()` 中的
  `float → temperature → torch.topk → top-k mask → softmax → top-p mask →
  log → sample_from_sorted_logprobs_with_seed_small_k()`。
- **W（工作负载）**：`Qwen/Qwen3-TTS-12Hz-1.7B-Base` Predictor 的采样几何：
  词表 2048、每个语义 token 15 个 codec group。HPC3 单张 H100 80 GB，BF16 logits，
  batch 1、4、8，top-k 30、50，top-p 1.0、0.8、0.95。该 isolated 测量不加载
  checkpoint，也不报告端到端延迟。
- **不变量**：相同输入、seed、语义位置和 group 时，候选必须产生与 R
  完全相同的 codec ID。它不能读取主机数据，不能改变随机数定义，不能改变
  参考路径的结果，也不能使用 `torch.compile` 或 Inductor。

## Acceptance Criteria

- AC-1：候选只接受 CUDA 的二维连续 BF16 logits、连续的每行参数、正的有界
  top-k、词表为 2048，且 `max_top_k` 为当前 Predictor ladder 中不大于 1024 的值。
  - Positive Tests：batch 1、4、8；top-k 1、2、30、50、1024；top-p
    1.0、0.5、0.8、0.95；不同温度、seed 和位置。
  - Negative Tests：CPU、错误维度、非连续 tensor、非 ladder 的图签名、full-sort、
    超过 1024 的 top-k、混合 sampled/argmax 的调用都必须返回 `None`，使调用方运行 R。
- AC-2：所有 AC-1 正例中，候选 token tensor 的 shape、dtype、device 和每个
  codec ID 都必须与 R 用 `torch.equal` 相等。
  - Positive Tests：随机 BF16 logits、相同值 logits、top-k bucket 宽度大于
    每行真实 top-k、signed-zero、top-p 阈值刚好跨过一个 rank、重复调用和 batch
    切分重组，以及 Murmur3 `hash == 0` 和 `hash == UINT32_MAX` 两个可达端点。
  - Negative Tests：任何一个 token 不相等即拒绝调度该候选。
- AC-3：CUDA Graph 捕获和 replay 不产生主机同步，且参数在每次 replay 时从
  持久 device buffer 生效。
  - Positive Tests：现有 Predictor graph 的 bit-identity 测试以及多步更新
    temperature、top-k、top-p、seed 的测试。
  - Negative Tests：候选不可用或参数条件不满足时，不可抛错，不可改变 eager
    和 graph fallback 的行为。
- AC-4：H100 中位 GPU 时间和 kernel 数都相对 R 减少，且反复交替测量没有
  回归。
  - Positive Tests：batch 1、4、8，15 groups，top-k 30/50 和 top-p
    1.0/0.8 的 isolated CUDA Graph replay；记录 CUDA event、torch trace
    和环境。
  - Negative Tests：任一代表性点更慢、codec ID 不一致或图捕获失败时，不合并
    调度；保留 R 并记录拒绝原因。

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

候选替换受支持的 CUDA Graph 采样链。它将 raw logits 到 codec ID 的计算放在
一次 Triton 启动中。它通过完整 shape 矩阵、同一进程 Predictor 对照和 H100
性能测量。

### Lower Bound (Minimum Acceptable Scope)

候选有严格的运行时资格检查。只对通过精确 ID 测试和 H100 测量的 shape
调度。所有其它形状保留 R。

### Allowed Choices

- Can use：Triton，当前的 Murmur3/Gumbel 定义，CUDA Graph，现有
  `sample_from_sorted_logprobs_with_seed_small_k()` 作为 R 的一部分。
- Cannot use：`torch.compile`，Inductor，修改 seed 或 position 定义，改变
  权重或精度，CPU readback，静默近似或概率容差代替 ID 相等。

## Feasibility Hints and Suggestions

候选先读取一行 2048 个 logits。它在 FP32 中进行温度缩放和 rank 选择。它以
当前 sampler 的 Murmur3 和 float64 Gumbel 算法计算每个候选 rank。它只保留
真实 top-k 和 top-p 允许的 rank。它输出 Gumbel-score 最大的原始 token ID。

`torch.topk` 的 tie 顺序是 R 的一部分。候选必须使用 H100 的相等 logits 和
重复值测试确认该顺序。不能证明时，候选不能调度该输入区域。

## Dependencies and Sequence

1. 锁定参考和测试矩阵。
   - 为 raw-logit 到 token 的候选增加独立测试。
   - 在 H100 记录 R 的 GPU 时间和 kernel 数。
2. 实现并验证候选。
   - 实现 Triton kernel 和 Python 资格检查。
   - 接入仅有界 top-k 的 all-sampled CUDA Graph 路径。
3. 评估和提升。
   - 执行 H100 的正确性、trace 和性能矩阵。
   - 运行 Humanize 独立代码审查。
   - 只在全部 acceptance criteria 通过时提交到 fork branch。

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
| --- | --- | --- | --- | --- |
| task1 | 建立 R 的直接对照和参数矩阵 | AC-1, AC-2 | coding | - |
| task2 | 调查 rank 排序和 Gumbel 的精确语义 | AC-2 | analyze | task1 |
| task3 | 实现 candidate 和严格 fallback | AC-1, AC-3 | coding | task2 |
| task4 | H100 正确性、trace 和性能评估 | AC-2, AC-3, AC-4 | analyze | task3 |
| task5 | 代码审查、结果文档和 fork 推送 | AC-1--AC-4 | analyze | task4 |

## Implementation Notes

代码和注释不使用本计划中的 AC、阶段或任务编号。`kda_runs/` 只保存到 PVC，
不得加入 Git。
