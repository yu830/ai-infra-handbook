# 后训练与强化学习系统

本章节预留用于沉淀 Post-Training 与 RL Rollout 系统架构笔记，计划涵盖：

1. **VeRL + vLLM / SGLang 协同架构**：Rollout 生成调度、训练引擎（FSDP2 / Megatron）与推理引擎之间的显存时分复用与异步权重同步（Weight Sync）。
2. **GRPO / Outcome-RL 系统瓶颈**：长尾轨迹（Straggler Rollout）导致的 GPU 空泡、Logprob 重算与稀疏 Teacher Scoring 显存边界。
3. **可复现与失败回放（Replay Infrastructure）**：训练数值漂移定位与确定性采样校验。
