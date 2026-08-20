# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前用户边界：Milestone 1–5、M4 cleanup与通用indexed/capacity-one performance follow-up均已完成。M5全树`1514 passed, 4 skipped`，Vivado Simulator四条RTL bench通过，未program/flash或访问真实hardware；M5于2026-08-20 14:31 PDT完成验证，满足15:30门。下一步是M6 USB-only SLM、Science Context与robust Feedback；未授权真实SLM/camera/FPGA实验。
