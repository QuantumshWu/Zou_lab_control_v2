# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前用户边界：Milestone 3已由backend commit `ca66c7d`和follow-up `Fix canonical full signal presentation`完成。Finite exact的event chunk只供exact Processor；Logic shape与所有UI/display统一使用对应publication的canonical full view，Monitor仍latest event。Milestone 4尚未开始；用户新报告的Selector Off与FacetGrid overview交互已记录在Architecture/Goal/Plan，等待用户明确确认后再实施。
