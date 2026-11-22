# 05. 个人核心项目与 PR 深挖防御

本页针对你简历与 GitHub（`github.com/yu830`）上最具含金量的 **4 个 AI Infra 硬核实战项目**，逐一拆解「30 秒讲法 → 关键技术细节 → 面试官最狠追问与高分答法」。

## 项目 1：`FlagGems` PR #9124 —— 曦望 Sunrise / SGLang Qwen3-4B GQA 修复（最高优先级）

!!! success "为什么必须第一个讲"
    这个 PR 直接解决的就是 **Sunrise（曦望）/ SGLang 运行 Qwen3-4B 时的 GQA SDPA math attention 失败问题**（Issue #6172），与明天面试的公司、芯片和软件栈 100% 精准咬合。

- **30 秒陈述**：在 `flagos-ai/FlagGems` 中定位并修复了通用 SDPA math attention 路径不支持 GQA（`H_q != H_kv`）导致 Sunrise/SGLang 跑 Qwen3-4B 报错的问题，并获社区维护者 Review 认可。
- **核心技术细节**：
  - 根因：generic SDPA math 路径默认假设 `Q` 与 `K/V` 头数一致，面对 Qwen3-4B 的 GQA 布局时维度广播/索引错位。
  - 修复：在不触发不必要大张量拷贝的前提下正确映射 `group_size = H_q // H_kv`，补齐 GQA dispatch 单测。
  - **主动展示工程严谨度（加分项）**：主动提及你在 Review 中识别出的 4 个防御性加固点——① 大张量下 `off_z * stride_wz` 需 `int64` 提升防止 >2B 元素溢出；② `reshape` 前显式校验 4D `key/value` 形状；③ `not isinstance(dropout_mask, bool)` 类型守卫；④ 将测试条件从 `device != "cuda"` 收紧为 `device == "cpu"` 避免在异构国产卡上误跳过。
- **面试官追问预判**：
  - *问：为什么要修 math fallback 路径，线上不是都跑 FlashAttention 吗？*
  - *答*：国产新卡适配或新模型 Day-0 上线时，手写汇编 FlashAttention 往往还没覆盖所有 Head Dim、Mask 变体或 GQA 比例；**Generic Math / Triton 路径既是 Day-0 可用性兜底，也是验证定制融合算子正确性的精度标尺**。

## 项目 2：`GraphBudget` —— SGLang v0.5.9 CUDA Graph 预算规划与 Small-M GEMM

- **30 秒陈述**：针对中小显存卡（8 GB RTX 5070 Laptop）部署 SGLang 时 CUDA Graph 静态池与 KV Cache 显存互相挤压的问题，设计了预算感知的 CUDA Graph 捕获规划与调度框架，并集成 `M16/N64/K128` Small-M GEMM Tiling，在并发 16 下实现 **72.4%** 中位有效输出吞吐提升。
- **面试官追问预判**：
  - *问：72.4% 的提升里，CUDA Graph 预算规划和 Small-M GEMM 各自贡献了多少？你是怎么归因的？*
  - *答*：强调**可控变量消融（Ablation）**——先固定算子，测不同捕获点数下 KV 块数变化导致的抢占（Preemption）率与 Padding 损耗；再固定图配置，用 Roofline/Profiler 对比小 $M$ 下的 SM 占用率与 GEMM 耗时，拆清显存收益与算子收益。
  - *问：这套思路迁移到 80 GB 的启望 S2 推理卡上还有用吗？*
  - *答*：完全通用。在 80 GB 卡跑 70B 模型或多 LoRA / 超长上下文时，剩余 KV 显存同样紧张（如 70B TP=4 单卡只剩 ~35 GB 给 KV），CUDA Graph 捕获预算、冷启动时间与 Padding 浪费的帕累托权衡是所有推理卡共性问题。

## 项目 3：`SGLang` PR #46596 —— 跨进程 ZMQ 调度交付延迟观测

- **30 秒陈述**：针对 SGLang 多进程架构下控制面尾延迟不可见的问题（Issue #43304），新增 `sglang:detokenizer_router_delivery_seconds` 指标，采用 33-byte Linux boot/time-namespace 时钟域多段 ZMQ framing 精确拆分 Scheduler → Detokenizer → Router 的 IPC 交付耗时。
- **核心亮点（展示底层 Linux / 系统功底）**：
  - 为什么不用 `time.time()`？墙上时钟受 NTP 校时跳变影响，跨容器/命名空间还需对齐时钟域；必须用单调时钟并对极端情况下的 TSC 负差值做 Clamping 保护。
  - 发现并定位了 `_worker` 中 5s sleep 瓶颈，以及 `SGLANG_RUST_SERVER` gating 与 `recv_multipart` 前向兼容性边界。
- **面试官追问预判**：
  - *问：生产线上如果客户反馈 TTFT / TPOT 突然变慢，你怎么用这类指标做分层定位？*
  - *答*：按「**网关排队 → Tokenizer → Scheduler 排队 → GPU Forward → ZMQ/Detokenizer 交付**」五段打点拆解，先定位是控制面 CPU/IPC 瓶颈还是计算面 GPU 瓶颈，再分别用 py-spy / Nsight Systems（或芯片自研 Profiler）下钻。

## 项目 4：`llama.cpp` PR #29576 —— 大词表 Pinned Memory 减半与异步同步修复

- **30 秒陈述**：针对 Qwen 等大词表模型（$V = 248{,}320$）在 `llama.cpp` 中 Host Pinned Memory 输出缓冲高达 242.5 MiB 的问题，重构为 `output_buffer_view` 方案将内存占用减半至 **121.3 MiB**，并修复了异步 D2H 拷贝缺失同步导致的竞态隐患。
- **面试官追问预判**：
  - *问：Pinned Memory（锁页内存）为什么不能随便多开？异步拷贝竞态是怎么发生的？*
  - *答*：Pinned Memory 不可被 OS 换出到 Swap，过量分配会挤压主机可用物理内存并拖慢 `cudaHostAlloc`；异步 `cudaMemcpyAsync` 返回时数据尚未真正到达 Host 内存，若未在对应 Stream 上做 Event/Stream 同步就直接读取 logits，会读到上一轮脏数据。
