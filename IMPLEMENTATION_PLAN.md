# ZLC v2 Architecture Convergence — Implementation Plan

本文只保存当前实施状态、阶段证据和下一步，不保存旧Goal历史。目标架构见`ARCHITECTURE_DESIGN.md`，完整范围见`ZLC_V2_IMPLEMENTATION_GOAL.md`。

## 1. Persistent Checkpoint

更新时间：2026-08-17
启动HEAD：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
Branch：`master`
用户执行边界：完成Milestone 1并commit后必须停下，等待用户确认；不得进入Milestone 2。

### 当前状态

- 审计：完成；逐文件证据在`AUDIT/`。
- 用户裁决：完成；记录在`AUDIT/USER-DECISIONS-2026-08-17.md`与根Goal。
- Production代码：尚未修改。
- Hardware：未访问；本Goal不授权program/flash或实验机device操作。
- Milestone 0：`IN PROGRESS` — 正在重建唯一Architecture、Plan、Handoff并纳入审计证据。
- Milestone 1：`PENDING` — pure deletion与历史清扫。
- Milestone 2–7：`PENDING`；本次不得开始。

### 当前工作树基线

开工时仅有未跟踪的`AUDIT/`与`ZLC_V2_IMPLEMENTATION_GOAL.md`，没有production/test修改。Milestone 0将把它们作为authority/evidence纳入版本控制。

### 当前下一步

1. 完成Milestone 0文档替换与link/whitespace检查。
2. Commit：`Establish approved architecture and implementation checkpoint`。
3. 立即把本Checkpoint更新为Milestone 0完成、Milestone 1进行中。
4. 按当前consumer graph执行Milestone 1删除、focused smoke与残余搜索。
5. Commit Milestone 1，更新Checkpoint并停下等用户。

## 2. Milestone状态

| Milestone | Scope | Status | Evidence / Commit |
|---|---|---|---|
| 0 | Approved Architecture、current Plan/Checkpoint、Handoff、Audit evidence | IN PROGRESS | 启动HEAD `92089f5` |
| 1 | Dead framework、parallel pipeline、test-only surface、duplicate launcher/docs/tests删除 | PENDING | — |
| 2 | Data/Durable/Installation truth | PENDING | — |
| 3 | Canonical Runtime live与Logic Node contract | PENDING | — |
| 4 | Exact Data/Fit/Overlay与Qt lifecycle | PENDING | — |
| 5 | Pulse/Camera/Remote/FPGA | PENDING | — |
| 6 | USB-only SLM与robust Feedback | PENDING | — |
| 7 | One distribution、evidence lanes、final docs | PENDING | — |

## 3. Audit finding映射

| Evidence | Milestone owner |
|---|---|
| `AUDIT/06a-data-durable.md` Data dead clusters | 1 |
| `AUDIT/03c-runtime-contract-prune.md`, `06d-runtime-plot-remaining.md` dead Runtime/Plot paths | 1 |
| `AUDIT/06b-ui-workbench.md` dead UI/Workbench surfaces与duplicate launchers | 1 |
| `AUDIT/06c-atom-remaining.md` zero-consumer descriptor/fakes/oracles | 1 |
| `AUDIT/06f-fpga-nonpython.md`, `06h-pulse-remaining-python.md` test-only/dead Pulse surfaces | 1 |
| `AUDIT/06g-test-evidence-architecture.md` fake-only、doc-SHA、API-cap、wrong-checkout tests | 1 |
| Figure archive、validity、selection、unique path、duplicate DeviceSpec | 2 |
| Runtime future data、partial terminal、stamp/slot/processor issues | 3 |
| Plot fit/overlay/selector、Form/Viewer/Qt ownership | 4 |
| Pulse count/STOP/SAFE/DONE/remote/FPGA/camera cadence | 5 |
| SLM context/USB/profile/correction/feedback/dense solve | 6 |
| Packaging、CI/notebooks、current docs | 7 |

## 4. Milestone 0验收

- `ARCHITECTURE_DESIGN.md`只描述批准Target并明确`NOT YET IMPLEMENTED`。
- 本文只描述当前status/evidence/next action，不含多代历史Checkpoint。
- `HANDOFF.md`只指向Architecture、Plan和Goal。
- `AUDIT/`与根Goal进入版本控制。
- 所有Markdown relative links可解析，`git diff --check`无错误。
- Commit完成后进入Milestone 1，不做production改动。

## 5. Milestone 1范围与验收

### 删除范围

- Runtime旧exact/reservation/builder/monitor/live-port/preview与dead failure/cleanup/RunHandle支线及self-tests；
- Plot无consumer LivePlotController、`_live_channel`、FitNumericTable、test-only facade conveniences与self-tests；
- Data零consumer numeric/AxisSourceRef/ResolvedPointRows/ValuePayloadContract clusters；
- Pulse test-only engine model迁入tests所需最小oracle、fake Interprocess lease、dead transport/public surface；
- Atom zero-consumer dynamic resolver/fakes/oracle surface；
- UI/Workbench dead Graph/FormGrid/gallery-only production surface；
- duplicate package launchers、duplicate fit contracts/SHA tests、duplicate surveys、历史acceptance/reacceptance/goal archives；
- every-export notebook guards、arbitrary API-size caps、fake-only与wrong-checkout tests；
- 同步收窄exports、imports、examples、notebooks和docs，不留空package、alias或compatibility shim。

### 证据

- 用当前HEAD consumer graph证明删除项无production consumer；旧tests/docs不算consumer。
- 每个保留package import smoke先import`zou_lab_control_v2`并打印root与tested package路径。
- 运行直接受影响的focused tests与真实consumer smoke，不 reflexively跑full tree。
- Repo搜索证明旧入口、exports、module和duplicate docs不再命中。
- 不新增production file/class/framework。
- `git diff --check`无错误。

### Commit与停止门

Milestone 1建议commit：`Prune dead frameworks and historical product surfaces`。

Commit后把Checkpoint更新为Milestone 1完成、Milestone 2待用户确认；随后停止本Goal turn，不做Milestone 2设计或编辑。

## 6. Checkpoint更新规则

每个milestone开始/完成、长测试前或新用户裁决后立即更新：

- status；
- current HEAD/dirty；
- exact decision；
- focused evidence与路径provenance；
- commit subject（commit hash通过`git log --grep`解析，文档不自引用自己的hash）；
- next unfinished action。

不得恢复append-only历史日志，不把对话摘要当状态，不把passing test数量冒充未覆盖行为完成。
