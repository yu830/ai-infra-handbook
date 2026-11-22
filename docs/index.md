# 曦望 Sunrise · 大模型推理引擎研发工程师面试作战手册

> 面向 **2026 年 11 月 23 日 15:00（北京时间）** 曦望（Sunrise / 上海阵量智能）视频面试定制，覆盖底层推理物理模型、源码级机制、国产 GPGPU 软件栈适配，以及你个人核心项目与开源 PR 的深度防御策略。

## 1. 目标公司与岗位速览

| 维度 | 关键信息（公开资料与面试邀约） | 面试切入点 |
| :--- | :--- | :--- |
| **公司与团队** | **曦望（Sunrise / 上海阵量智能）**，拆分自商汤芯片与架构部，核心骨干来自 AMD、百度昆仑芯、华为海思、商汤（[财联社](https://www.cls.cn/detail/2248341)） | 强调你对「芯片架构约束 → 编译器/算子 → 推理引擎调度」跨层协同的理解 |
| **核心硬件路线** | **启望 S2**（7nm 大模型推理 GPGPU，兼容 CUDA 生态，对标 A100 约 80% 性能，已量产）；**启望 S3**（下一代多模态推理芯片，目标成本降 90%）（[钛媒体](https://www.tmtpost.com/7630475.html)） | 紧扣「单位 Token 推理成本」「显存带宽利用率」「多模态/长上下文 Goodput」 |
| **开源软件栈** | 深度参与 [FlagOS 2.0](https://github.com/flagos-ai/FlagRelease/issues/5) 生态（`FlagTree` 统一 Triton 编译器、`FlagGems` 算子库、`vllm-plugin-FL` 注意力后端移植、`FlagGems-SGLang`） | **王牌命中**：你本人的 `FlagGems` PR #9124 直接修复了 Sunrise / SGLang Qwen3-4B 的 GQA 故障 |
| **本轮考察重点** | 邀约明确说明：**通用型业务问题 + 过往项目经历 + 价值观与软性技能** | 不是纯八股背诵，重点考**工程判断力、排障闭环思维、真实贡献深度与团队协作** |

## 2. 考前冲刺阅读顺序（按优先级）

1. **必背项目深挖（35% 精力）**：先过 [05. 个人核心项目与 PR 深挖防御](interview/project-deep-dive.md) —— 确保 `FlagGems #9124`、`GraphBudget`、`SGLang #46596`、`llama.cpp #29576` 的每个数字、根因和追问都脱口而出。
2. **业务与系统设计（35% 精力）**：精读 [06. 高频系统设计、业务题与反问](interview/qa-and-business.md) 与 [03. P/D 分离与国产芯片适配](inference/pd-disaggregation.md) —— 练熟如何从业务 SLO（TTFT / TPOT / 单位 Token 成本）反推引擎架构选型。
3. **底层硬核原理与手算（30% 精力）**：速查 [01. 物理边界、手算算例与调度循环](inference/roofline-and-batching.md)、[02. KV 缓存、MLA 与 CUDA Graph](inference/kv-cache-and-cudagraph.md) 和 [04. GPU 显存层次、算子融合与量化](kernels/gpu-memory-and-tiling.md) —— 保证白板手算 70B 显存、Online Softmax 公式、Paged/Radix 细节零卡壳。

!!! tip "本轮面试的核心气场定位"
    不要把自己定位成「背过几篇论文的在校生」，而是定位成「**真正在 SGLang / vLLM / llama.cpp / FlashInfer / FlagGems 上游提过代码、踩过硬件与 IPC 真实坑点的系统工程师**」。回答任何业务问题，都按「**约束量化（Roofline / VRAM）→ 机制选型 → 实测验证与兜底**」三段式展开。
