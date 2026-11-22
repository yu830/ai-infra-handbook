# 06. 高频系统设计、业务题与反问清单

本轮面试邀约明确提出考察 **通用型业务问题 + 过往项目经历 + 价值观与软性技能**。以下整理最可能被问到的 6 道压轴题与高段位反问。

## 1. 高频系统设计与业务场景题

### Q1. 客户反馈线上推理服务「平均延迟正常，但 P99 尾延迟经常飙升」，怎么排查？

**标准四层排查框架**：
1. **看批构成与干扰**：是否混进了超长 Prompt Prefill 阻塞了在途 Decode？→ 开启并调优 **Chunked Prefill**（调整 `max_num_batched_tokens`），集群规模大则上 **P/D 分离**。
2. **看 KV Cache 抢占（Preemption）**：高并发下是否因 KV 显存打满触发了 Recomputation 或 CPU Swap？→ 监控 KV Cache 使用率与抢占次数，开启 **KV FP8 量化**、**前缀缓存（APC / Radix）** 或收紧准入并发。
3. **看控制面与 IPC 气泡**：GPU 利用率掉坑时是否出现 Python GC 停顿、Tokenizer 阻塞或 ZMQ 排队？→ 检查 Overlap Scheduling 是否开启、跨进程交付延迟打点。
4. **看硬件与通信抖动**：多卡 TP 下是否存在单卡降频（Thermal Throttling）或 NCCL/RDMA 拥塞拖慢整组 AllReduce？

### Q2. 如果让你把一个新开源大模型（如 Qwen3 / DeepSeek 新变体）在曦望自研卡上从 0 到 1 跑通并优化到上线标准，你会怎么推进？

**四步工程闭环**：
1. **Day-0 跑通与精度对齐**：优先通过标准算子路径（FlagGems / Generic Math）在单卡跑通前向，用 CPU FP32 做 Golden Reference，逐层比对 Hidden States 余弦相似度与最终 Logprobs，确保零静默精度 Bug。
2. **Roofline 性能建档**：分别测 Prefill（不同输入长度）与 Decode（不同 Batch Size）的算子耗时占比，识别 Top-5 热点算子与 CPU Launch 占比。
3. **算子与图级优化**：热点访存算子用 FlagTree/Triton 做垂直融合（RMSNorm+RoPE+KV Write、SwiGLU）；Attention 切换至硬件特化后端；Decode 打通 CUDA Graph 捕获并标定最优捕获点预算。
4. **集群与业务压测**：基于真实业务流量分布（输入/输出长度分布、前缀命中率）压测 TTFT / TPOT P99，确定最优 TP/DP 并行度与是否启用 Chunked Prefill / P/D 分离。

### Q3. 同样是追求「推理成本降低 90%」（曦望 S3 的目标），推理引擎层能做哪些协同贡献？

- **显存换并发**：MLA 矩阵吸收 + KV Cache FP8/INT8 量化 + Radix/APC 前缀复用，直接把单卡有效并发提升数倍。
- **算力与带宽解耦**：P/D 分离消除混跑损耗，用更低成本的卡配比满足同等 Goodput SLO。
- **算法-系统协同**：投机解码（Speculative Decoding / MTP）用一次权重读取验证多个 Draft Token，直接把 Decode 算术强度乘以接受长度 $\alpha$，打破单 Token 带宽墙。

### Q4. 为什么很多时候 W4A16 量化在 Batch=1 时提速明显，但大 Batch 时反而变慢？

- Batch=1 时是纯 **Memory-Bound**，W4A16 把权重读取量降到 1/4，带宽收益远大于反量化开销。
- 大 Batch（$B > B^*$）进入 **Compute-Bound** 后，W4A16 仍需在片上反量化回 FP16 再送 Tensor Core，反量化指令本身引入额外 ALU 开销，此时应切换为 **W8A8（FP8/INT8）** 直接吃双倍 Tensor Core 算力。

## 2. 价值观与软性技能应答要点

- **大三本科生如何化劣势为优势**：不要回避经验年限，直接用**公开可验证的工程产出力**说话（20+ 上游 PR、直接修过曦望相关的 FlagGems GQA 问题、自驱做完 GraphBudget 并拿出 72.4% 消融数据），证明你具备「**给一个模糊的底层 Bug 就能独立查源码、建复现、提 PR、过 Review 的闭环能力**」。
- **遇到与团队/Reviewer 意见分歧怎么办**：结合你在 `llama.cpp`（采纳 ORippler 的 `output_buffer_view` 重构建议）或 `FlagGems`（根据 Review 补充 `int64` 溢出防护）的真实经历——**以数据、可证伪的 Benchmark 和边界用例为准绳，对事不对人**。

## 3. 高段位反问面试官清单（选 2 个即可）

1. **技术演进向**：「曦望目前在 S2/S3 上推进大模型推理生态时，更侧重通过 `vllm-plugin-FL` / `FlagGems` 这种开源统一插件层做适配，还是会有自研的推理运行时？两者在性能压榨和维护成本上团队目前是怎么权衡的？」
2. **业务痛点向**：「如果我有机会加入推理引擎团队，前 3 个月最核心、最痛的 1–2 个技术瓶颈会偏向**算子融合/后端移植**、**CUDA Graph/调度层优化**，还是**多机 P/D 分离与通信优化**？」
3. **团队文化向**：「团队在面对新模型 Day-0 适配和长期底层性能重构之间，通常怎么安排研发节奏？」
