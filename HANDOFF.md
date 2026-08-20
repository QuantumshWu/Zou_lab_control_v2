# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前用户边界：Milestone 1–6、M4 cleanup与通用indexed/capacity-one performance follow-up均已完成。M5于2026-08-20 14:31 PDT完成验证并满足15:30门；M6随后完成USB-only SLM、Science Context、robust Feedback与immutable Simulation，全树`1504 passed, 4 skipped`。未访问真实SLM/camera/FPGA；官方USB ABI、光学验收与M7 single-distribution/fresh-install仍待后续。用户本轮四项要求已完成，下一步不自动进入M7。
