# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前用户边界：Milestone 1–4及强制post-milestone residual sweep均已完成；M4由`Make data, fit, overlay, and Qt lifecycle exact and atomic`落盘。全树1498 tests通过，未访问hardware。下一步只与用户继续讨论正式Camera Area-fit→ROI Image→Rolling链P50/P95约150/167 ms的性能取舍；Milestone 5尚未开始，未经用户确认不得进入。
