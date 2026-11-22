# 04. GPU 显存层次、算子融合与量化

推理引擎的宏观调度优化终究受限于底层算子的微观硬件利用率。本节整理面试中最常追问的 GPU 内存层次、FlashAttention 原理、Small-M GEMM 与量化权衡。

## 1. GPU 存储层次与优化铁律

| 存储层次 | 典型容量 | 相对带宽 | 核心工程优化目标 |
| :--- | :--- | :--- | :--- |
| **HBM（Global Memory）** | 8 GB – 192 GB | 1×（基准） | 合并访存（Coalesced Access）、算子垂直融合、杜绝中间大矩阵写回 |
| **L2 Cache** | 数 MB – 数十 MB | ~3–5× | Thread Block Swizzling / Persistent Kernel 提升权重块 L2 复用率 |
| **SRAM / Shared Memory** | 每 SM 64–228 KB | ~10–20× | 分块驻留（Tiling）、Padding/Swizzle 消除 32-Bank Conflict |
| **Registers** | 每 SM 256 KB | 无延迟 | 承载 Tensor Core MMA 累加器，警惕 Register Spilling 溢出到本地内存 |

## 2. FlashAttention 与 Online Softmax 推导

标准 Attention 计算 $O = \text{softmax}(Q K^\top / \sqrt{d_h}) V$ 需要将 $[S, S]$ 的中间矩阵 $S, P$ 写回 HBM，带来 $O(S^2)$ 显存占用与带宽开销。

FlashAttention 的核心是 **Tiling + Online Softmax**：将 $Q$ 按行分块、$K, V$ 按列分块载入片上 SRAM；每读入一块 $K_j, V_j$，在寄存器内更新当前局部最大值 $m_i$、归一化分母 $l_i$ 以及未归一化输出 $O_i$：

$$
m_i^{\text{new}} = \max(m_i^{\text{old}}, \tilde{m}_{ij}), \qquad
l_i^{\text{new}} = e^{m_i^{\text{old}} - m_i^{\text{new}}} l_i^{\text{old}} + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{l}_{ij}
$$

$$
O_i^{\text{new}} = \text{diag}(l_i^{\text{new}})^{-1} \left( \text{diag}(l_i^{\text{old}}) e^{m_i^{\text{old}} - m_i^{\text{new}}} O_i^{\text{old}} + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{P}_{ij} V_j \right)
$$

```python
# FlashAttention 前向核心循环（Triton 思维骨架）
m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

for start_n in range(0, seq_len, BLOCK_N):
    k = tl.load(K_block_ptr)                  # [HEAD_DIM, BLOCK_N]
    qk = tl.dot(q, k) * sm_scale              # [BLOCK_M, BLOCK_N]
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.exp(qk - m_ij[:, None])            # 数值稳定
    alpha = tl.exp(m_i - m_ij)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_ij

out = acc / l_i[:, None]                      # 仅最终结果写回 HBM -> O(S * d_h)
```

## 3. Decode Small-M GEMM 与垂直算子融合

- **Small-M GEMM 痛点**：Decode 时 $M = B \in [1, 32]$，若复用训练/Prefill 的 `M128 × N128` 块，90% 的 M 维计算都在算 Padding，且网格块数少导致大量 SM 闲置（Wave Quantization）。
  - **解法**：针对小 $M$ 定制细粒度 Tiling（如 `M16 / N64 / K128`）并启用 **Split-K**（沿 $K$ 维切分并行归约），拉满 SM 占用率与 HBM 带宽。
- **垂直融合（Vertical Fusion）**：将相邻的访存受限算子合并为一个 Kernel（如 `RMSNorm + RoPE + Paged KV-Cache Write`、`Gate/Up + SwiGLU`），把多次 HBM 读写压成单次。
- **工程必避坑——`int32` 索引溢出**：在长上下文或大批量 Attention / Reduction Kernel 中，元素偏移量 `off_z * stride_z` 极易超过 $2^{31}-1$（约 21.4 亿），必须显式转 `tl.int64` / `int64_t`。

## 4. 推理量化速查（什么时候省带宽，什么时候省算力？）

| 量化方案 | 代表算法 | 核心收益 | 适用阶段与硬件要求 |
| :--- | :--- | :--- | :--- |
| **W4A16（仅权重量化）** | AWQ / GPTQ + Marlin Kernel | 权重体积缩至 **1/4**，大幅提升 Decode HBM 带宽上限 | **最适合低 Batch Decode**（带宽受限）；Prefill 需反量化为 FP16 计算，算力不翻倍 |
| **W8A8（INT8 / FP8）** | SmoothQuant / FP8 Block-wise | 显存减半 **且** 调用 INT8/FP8 Tensor Core 算力翻倍 | **Prefill 与大 Batch Decode 双赢**；FP8 无需复杂离群值平滑，动态范围远优于 INT8 |
| **KV Cache 量化（FP8 / INT8）** | vLLM / SGLang `kv_cache_dtype=fp8` | KV 显存减半，直接把最大并发 $B_{\max}$ 翻倍，压低长上下文 Attention 带宽 | 超长上下文 / 高并发 Serving 必备，几乎无明显精度损失 |
