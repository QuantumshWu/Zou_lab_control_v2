# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前用户边界：Milestone 1–6、M4 cleanup与通用indexed/capacity-one performance follow-up均已完成。M5于2026-08-20 14:31 PDT完成验证并满足15:30门；M6随后完成USB-only SLM、Science Context、robust Feedback与immutable Simulation。当前HEAD为`e8e5517 Complete plot performance and bounded live-fit flow`。

该HEAD的Plot performance closure删除Raster旧多frame/bytes队列，保留active+latest并由现有worker独立执行1秒solve timeout；live FitEvent在exact solve后、owner raster前发布，manual fit仍在accepted overlay后通知，主图继续只呈现atomic `data@N + fit@N`。正式100 ms三轮链的candidate joint P50/P95为74.85/93.10 ms、Rolling为87.83/96.15 ms，174/180 valid、6 solver invalid、0 busy miss/queued middle/error；五kind DPR1/2保持pixel exact。

在该HEAD之上的Plot projection/fit baseline已完成用户手工验收并获准独立commit。文档同步前的冻结code/test candidate为7 files `+735/-78`：production净增215、tests净增442；无新增file/class/kind/model lane。它关闭large dense singleton/facet、native integer Histogram、ungrouped Rolling pool、regular-image radius lower与exact retry false-status，并清掉R=1 initial history和Histogram facet最后两处first-frame copy。完整10-case/memory矩阵见Plot performance文档；关键complete P50/P95为2048² Image 35.42/38.05→21.60/22.39 ms、Facet×8 366.48/377.69→45.24/46.98 ms、Histogram 175.79/178.02→33.13/34.80 ms、Rolling 111.80/112.77→15.14/17.30 ms。

当前证据为Plot全套`418 passed`、跨层定向`5 passed`、最终fit focused `50 passed`与data/facet/history focused `96 passed`、全树`1535 passed, 4 skipped, 6 warnings`；随机矩阵194 projection、954 Histogram及21 Fit cases全部保持numeric parity。Gate 17已完成：新增9个test functions与1个shared helper全部KEEP，active旧名与待删dead fallback为0，active docs已同步。

未访问真实SLM/camera/FPGA；官方USB ABI、光学验收与M7 single-distribution/fresh-install仍待后续。唯一下一步是提交已获批准的Plot baseline；不自动进入M7。
