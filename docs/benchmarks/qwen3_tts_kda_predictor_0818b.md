# Qwen3-TTS Predictor KDA 结果

日期：2026-08-18。

## 结论

本次保留两个不依赖 `torch.compile` 的 Predictor 改动。

- P2 在 CUDA graph 捕获时，将每个 residual codec group 的
  `nn.Embedding(next_code)` 和 `pos_summed.add_()` 合成一个 Triton kernel。
  1.7B 模型每个语义 token 有 15 个这类 group。
- P3 删除一个没有读取者的 Predictor 暂存 buffer。旧代码在每个语义 token
  清零该 buffer，并写入 talker 输入、layer-0 输入和 15 个 residual 输入。
  后续计算只读取局部张量，不读取该 buffer。

P1 没有保留。P1 尝试使用 SGLang 的 residual-carry RMSNorm。H100 上它和
当前 BF16 顺序的最大绝对差为 `1.562e-02`。这种差异可以改变后续采样的
codec ID。因此 KDA 在性能测量前拒绝了 P1。

本次没有端到端提速结论。run 0817a 已证明 C1 主要受主机侧阻塞读回限制。
此外，跨进程的固定 seed 输出不稳定。P2 和 P3 的正确性和性能结论只覆盖
下面列出的同一进程验证和 isolated GPU 测量。

## KDA 合同

- **K（优化单元）**：Predictor residual codec group 中的 embedding gather
  和 BF16 accumulator add。
- **R（参考实现）**：提交
  `9da7d19123ac5272dab005082d28f11c89452d40` 中的
  `nn.Embedding(next_code)`，随后执行 `pos_summed.add_()`。
- **W（工作负载）**：`Qwen/Qwen3-TTS-12Hz-1.7B-Base`，HPC3 单张 H100
  80 GB，BF16，batch 为 1、4、8。生产验证使用同进程的固定 seed ICL
  声音克隆请求。
- **不变量**：不得使用 `torch.compile` 或 Inductor；不得改变权重、精度、
  sampler、调度或 CUDA graph 的输入输出；codec ID 和 summed embedding
  必须与参考实现完全相等。

P2 的 kernel 只接受 CUDA、连续、非重叠的 BF16 tensor。它将选中的
embedding row 写入一次性分配的 `[batch, talker_hidden]` buffer，并用同一
次启动更新 accumulator。普通 eager 路径不调用 P2。CUDA graph 不可用时，
以及 tensor 条件不满足时，调用方执行原始 `nn.Embedding` 和 `add_` 路径。

P3 不改变计算顺序。它将 `talker_predictor_embed`、
`layer0_predictor_embed` 和 `new_predictor_embed` 直接传入
`_predictor_forward_one_token()`。它删除了未读取的中间写入。

## 正确性

H100 上的定向测试通过。P2 kernel 测试覆盖重复 codec ID、非零 BF16
accumulator、batch 1/4/8、hidden width 8/2048，以及 dtype、layout 和
输出重叠的回退路径。CUDA graph 测试覆盖 sampled、argmax、padded bucket、
多步 replay 和无 host readback。

同一服务进程还执行了 70 次 Predictor 调用。该实验显式运行 P2/P3 路径和
原始 eager embedding/staging 参考路径。比较的 codec codes 与 summed
embeddings 全部相等。P2 kernel 一共执行了 1,050 次。它验证了 P2 的算术和
P3 的删除操作。最终代码不在普通 eager 路径调用 P2。该验证请求生成的 WAV
SHA-256 为：

`1c77a27916bf51a1cb56f4a57c47a7aea912350f5945edb31a98c47a48d967c8`

验证 hook 会额外运行一次 Predictor，所以该请求的 `8.812 s` 延迟不能用于
性能比较。

## GPU 测量

P2 的 isolated CUDA-graph replay 测量使用 15 个 residual codec group 和
hidden width 2048。数值是中位数。

| Batch | 参考实现 | P2 | 收益 |
| ---: | ---: | ---: | ---: |
| 1 | 0.047161 ms | 0.021725 ms | 0.025436 ms |
| 4 | 0.063313 ms | 0.021547 ms | 0.041765 ms |
| 8 | 0.086036 ms | 0.021828 ms | 0.064208 ms |

在 B=1、hidden=2048、15 groups 的 profiler 中，参考实现为 30 个 kernel、
`42.302 us`。P2 为 15 个 kernel、`18.175 us`。这验证 P2 将每组的 gather
和 add 从两次启动变为一次启动。

同一 isolated 操作在普通 eager 路径没有收益。B=1/4/8 的参考中位数为
`0.140232/0.138920/0.138770 ms`，P2 为
`0.229957/0.227934/0.228506 ms`。因此最终实现只在 CUDA graph 捕获时使用
P2。图不可用时保持参考 eager 路径。

P3 使用 B=4、hidden=1024、16 groups 和小词表 fixture 进行结构性测量。
它的 graph replay 从 `1.420595 ms` 变为 `1.378303 ms`，中位收益为
`0.042285 ms`。profiler 从 781 个 kernel 变为 763 个 kernel，正好少 18 个
未读取 buffer 的写入节点。该 fixture 不是完整 1.7B Predictor，不能将该
数值外推为完整模型收益。

## 可复查证据

Pod：`<internal-pod>`。节点：`<internal-node>`。GPU：
H100 80 GB。driver：`590.48.01`。原始证据保存在 PVC：

`<internal-evidence-path>`

主要文件：

- `candidates.jsonl`：P1 的拒绝记录。
- `p2_gather_add_v2.json`：P2 CUDA-event 测量。
- `p2_profile_v2/summary.json`：P2 kernel 数和 GPU 时间。
- `p3_staging_v4.json`：P3 结构性测量。
- `p2_p3_same_process_parity.json`：同一进程的生产路径对照。

H100 定向验证命令如下：

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 python -m pytest -q \
  tests/unit_test/qwen3_tts/test_predictor_kernels.py \
  tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py
```

结果：`53 passed, 37 warnings`。

## 后续顺序

1. 先完成 A1/B 的异步 decode 工作。它必须消除逐步 host 同步，才能让
   Predictor 的 GPU 节省反映到请求延迟。
2. 再单独评估 seeded sampling 融合。它的输入是 raw logits、top-k、top-p
   和 seed，必须以每个 codec ID 的 bit identity 为门槛。
3. 量化和 GEMM 更换必须单独评估，并以 Seed-TTS 质量结果作为上线门槛。
