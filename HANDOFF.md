# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前用户边界：Milestone 1–6、M4 cleanup与通用indexed/capacity-one performance follow-up均已完成。M5于2026-08-20 14:31 PDT完成验证并满足15:30门；M6随后完成USB-only SLM、Science Context、robust Feedback与immutable Simulation。当前HEAD为`55d6ee7 Optimize dense plot projection and regular image fits`。

先前`e8e5517`的Plot performance closure删除Raster旧多frame/bytes队列，保留active+latest并由现有worker独立执行1秒solve timeout；live FitEvent在exact solve后、owner raster前发布，manual fit仍在accepted overlay后通知，主图继续只呈现atomic `data@N + fit@N`。正式100 ms三轮链的candidate joint P50/P95为74.85/93.10 ms、Rolling为87.83/96.15 ms，174/180 valid、6 solver invalid、0 busy miss/queued middle/error；五kind DPR1/2保持pixel exact。

`55d6ee7`已提交获用户手工验收的Plot projection/fit baseline：large dense singleton/facet、native integer Histogram、ungrouped Rolling pool、regular-image radius lower/exact retry和最后两处first-frame copy均已闭合。完整10-case/memory矩阵见Plot performance文档；关键complete P50/P95为2048² Image 35.42/38.05→21.60/22.39 ms、Facet×8 366.48/377.69→45.24/46.98 ms、Histogram 175.79/178.02→33.13/34.80 ms、Rolling 111.80/112.77→15.14/17.30 ms。该baseline的最终证据为Plot`418 passed`、跨层`5 passed`与全树`1535 passed, 4 skipped, 6 warnings`。

当前冻结的Axis/camera/fit-style follow-up在本次文档同步前为16 files `+349/-94`：production净增61、tests净增194；无新增file/class/lane。它为steady SmartOffset layout增加完整input memo；让Camera/Calibration默认photoelectron request在无conversion时明确effective False并发布raw `count`，同时贯通preview/archive/replay/Occupancy unit；让Image fit ellipse复用occupied thin ring并把center从50降至2.25 pt²。精确Axis A/B与call reductions见Plot performance文档。

当前follow-up证据为合并focused `482 passed`、camera定向`64 passed`、Axis parity `67 passed`、Image fit style/golden/Facet `40 passed`及最终style+compose `4 passed`；最终全树`1538 passed, 4 skipped, 6 warnings`，warnings仍仅既有vendor SWIG deprecation。Gate 17已闭合consumer graph、direct API unit defaults、重复camera refusal/Axis private call-count tests与active旧文案。Post-doc冻结树为26 files `+428/-125`：production净增68、tests净增170、docs净增65；无新增file/class/lane，test functions净增2且均KEEP。

未访问真实SLM/camera/FPGA；官方USB ABI、光学验收与M7 single-distribution/fresh-install仍待后续。当前唯一下一步是提交已冻结的Axis/camera/style follow-up；不自动进入M7。
