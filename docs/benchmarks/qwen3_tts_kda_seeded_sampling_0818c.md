# Qwen3-TTS Predictor Seeded Sampling KDA 结果

日期：2026-08-18。

## 结论

保留该候选。

它在 Predictor 的 CUDA Graph 捕获中，把每个 codec group 的 bounded
seeded sampling 链合成一个 Triton kernel。该 kernel 从 BF16 logits 直接写出一个
`int64` codec ID。它不使用 `torch.compile` 或 Inductor。

普通 eager 路径和不满足条件的 CUDA Graph 路径继续运行原参考实现。

## KDA 合同

- **K**：`[batch, 2048]` logits 的温度缩放、有界 top-k、可选 top-p，以及按
  `(seed, position)` 生成的 Gumbel sample。
- **R**：`_sample_subtalker_token_seeded()` 中的 `float → temperature → topk →
  top-k mask → softmax → top-p mask → log → seeded sample` 链。
- **W**：1x H100 80 GB。BF16 logits。batch 为 1、4、8。每次 decode 有
  15 个 Predictor codec group。
- **不变量**：在同一进程中，相同 logits、参数、seed 和 position 必须产生完全相同
  的 codec ID。

候选只接收连续 CUDA BF16 logits、连续的每行参数和现有 Predictor top-k ladder：
`4, 8, 16, 32, 50, 64, 128, 256, 512, 1024`。每行真实 `top_k` 可以小于其
ladder 宽度。缺少 Triton `gather` 支持或其它输入不满足条件时，它返回 `None`，
调用方运行 R。

`torch.topk` 的相同分数顺序会影响固定 seed 的 codec ID。对 `k <= 32`，kernel
复现 H100 上 PyTorch 的 threshold gather 和 32-entry bitonic sort，包括 signed-zero
的选择顺序。对更大的 ladder 宽度，测试确认了 H100 上的 source-index 顺序。

## 正确性

- H100 上 `test_sampling_kernels.py`：57 passed。
- H100 上 `test_predictor_cuda_graph.py`：46 passed。
- 独立矩阵：1,500 组随机 BF16 输入和 480 组 threshold-tie 输入。覆盖 batch 1、4、8；
  全部 10 个 ladder 宽度；不同温度、每行 top-k、top-p、seed 和 position。所有
  codec ID 与 R 完全相等。另有 128 组 signed-zero 输入，覆盖 `k=32,50,1024` 的
  selection 顺序。
- 独立审查后的端点回归覆盖 `hash == 0` 和 `hash == UINT32_MAX`。项目固定的
  SGLang 0.5.16，以及 H100 容器中的 0.5.17，都只对 `log(hash / UINT32_MAX)` 的
  float64 下端做截断。候选现在逐项复现该定义：前者是有限值，后者保留 `+inf`。
- CUDA Graph 测试覆盖 capture、replay、参数更新、padded bucket、sampled 和
  argmax fallback。融合路径不调用 `torch.topk`。强制禁用 Triton `gather` 时，
  调用方在图捕获中回退到 R，并保持 bit identity。

## H100 性能

下表为独立审查修复后的 15 个 codec group CUDA Graph replay。每个点交替测量 R 和
候选 3 次，每轮 200 次 replay。数字是 GPU 时间中位数。每个点的 codec ID 都与 R
完全相等。

| Batch | top-k | top-p | R | 融合 kernel | 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 1.0 | 687.45 µs | 415.59 µs | 1.65x |
| 1 | 30 | 0.8 | 887.51 µs | 444.68 µs | 2.00x |
| 1 | 50 | 1.0 | 735.03 µs | 243.09 µs | 3.02x |
| 1 | 50 | 0.8 | 922.34 µs | 247.49 µs | 3.73x |
| 4 | 30 | 1.0 | 797.40 µs | 435.87 µs | 1.83x |
| 4 | 30 | 0.8 | 966.81 µs | 455.70 µs | 2.12x |
| 4 | 50 | 1.0 | 828.10 µs | 243.74 µs | 3.40x |
| 4 | 50 | 0.8 | 1,002.53 µs | 248.73 µs | 4.03x |
| 8 | 30 | 1.0 | 822.12 µs | 436.42 µs | 1.88x |
| 8 | 30 | 0.8 | 988.33 µs | 455.39 µs | 2.17x |
| 8 | 50 | 1.0 | 853.97 µs | 244.00 µs | 3.50x |
| 8 | 50 | 0.8 | 1,039.96 µs | 248.69 µs | 4.18x |

Nsight Systems 在 B=4、k=30、top-p=0.8 的 10 次 graph replay 中记录：

| 路径 | 每 15 个 group 的 kernel 数 | 每 group 的 kernel 数 |
| --- | ---: | ---: |
| R | 390 | 26 |
| 融合 kernel | 60 | 4 |

kernel 数减少 330，或 84.6%。其中每 group 的 1 个融合 kernel 替代采样链；其余
3 个 kernel 是两条路径共有的 position 算术。

这是 isolated sampling 段的 GPU 结果。它不等同于端到端延迟结论。run 0817a 的
主要瓶颈仍是逐步主机读回和调度重叠。

## 可复查步骤

本报告不包含内部服务器标识或原始 profile 文件。仓库内的测试和下方命令
覆盖固定 seed 的正确性以及候选的选择条件。

H100 验证命令：

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONWARNINGS=ignore \
PYTHONPATH=. \
python -m pytest -q \
  tests/unit_test/qwen3_tts/test_sampling_kernels.py \
  tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py
```

修复后的实测结果：`103 passed in 13.62s`。
