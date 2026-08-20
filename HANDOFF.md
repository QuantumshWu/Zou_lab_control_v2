# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前用户边界：Milestone 1–4已落盘；全树1498 tests通过，未访问hardware。用户保留两项尚未授权实施的M4 cleanup：production安全净删约180–240行（首要为Raster单消费者adapter），tests安全合并/删除约500–800行（增量集中在9个纵向文件）。当前只讨论正式Camera Area-fit→ROI Image→Rolling链P50/P95约150/167 ms的性能取舍；Milestone 5尚未开始，未经用户确认不得进入。
