# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前用户边界：Milestone 1–3及强制post-milestone residual sweep已完成，由`Fix post-milestone residuals`单独落盘；Milestone 4尚未开始。用户已授权Milestone 4，下一步先把Checkpoint改为`IN PROGRESS`，从exact data+fit/PanelState transaction与已记录的Selector Off、FacetGrid overview行为开始。不得把M4实现混进residual fix commit。
