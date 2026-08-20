# ZLC v2 Architecture Convergence — Implementation Plan

本文只保存当前实施状态、阶段证据和下一步，不保存旧Goal历史。目标架构见`ARCHITECTURE_DESIGN.md`，完整范围见`ZLC_V2_IMPLEMENTATION_GOAL.md`。

## 1. Persistent Checkpoint

更新时间：2026-08-19
启动HEAD：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
Branch：`master`
用户执行边界：Milestone 3已完成并随本次commit落盘；必须停下等待用户确认，不得进入Milestone 4。

### 当前状态

- 审计：完成；逐文件证据在`AUDIT/`。
- 用户裁决：完成；记录在`AUDIT/USER-DECISIONS-2026-08-17.md`与根Goal。
- Production代码：Milestone 1、2、3已实现；Milestone 4尚未开始。
- Hardware：未访问；本Goal不授权program/flash或实验机device操作。
- Milestone 0：`COMPLETE` — commit `e854ddf`（`Establish approved architecture and implementation checkpoint`）；唯一Architecture、Plan、Handoff、Goal与Audit证据已纳入版本控制，未改production。
- Milestone 1：`COMPLETE` — 本milestone commit `Prune dead frameworks and historical product surfaces`；删除dead Runtime/Data/Plot/UI/Pulse/Atom surfaces、self-tests、重复launchers与历史docs，保留所有核实的真实consumer路径。
- Milestone 2：`COMPLETE` — strict Dataset/Figure/JSON truth、并发原子artifact命名、duplicate-device preflight与Calibration dependency/corruption修复；随本milestone commit落盘。
- Milestone 3：`COMPLETE` — Canonical Runtime chunk/current/seal、exact/latest Processor、typed preview/Task contract与真实plugin迁移已收口；commit subject见下方。
- Milestone 4–7：`PENDING`；本次不得开始。

### Milestone 3基线与完成边界

Milestone 3从clean HEAD `23e820d`开始；没有继承未归属的production/test修改，也未访问hardware。完成commit：`Unify Runtime live data and task previews`。

### 当前下一步

1. 停止实施，不开始Milestone 4设计或代码。
2. 等待用户明确确认进入Milestone 4。
3. 用户确认后从本commit的clean HEAD恢复，先完整读取Goal、Architecture与本Checkpoint。

## 2. Milestone状态

| Milestone | Scope | Status | Evidence / Commit |
|---|---|---|---|
| 0 | Approved Architecture、current Plan/Checkpoint、Handoff、Audit evidence | COMPLETE | `e854ddf`；docs/link/diff check，未跑测试 |
| 1 | Dead framework、parallel pipeline、test-only surface、duplicate launcher/docs/tests删除 | COMPLETE | 本milestone commit；见下方证据 |
| 2 | Data/Durable/Installation truth | COMPLETE | `Make dataset, archive, path, and installation truth strict`；见§6 |
| 3 | Canonical Runtime live与Logic Node contract | COMPLETE | `Unify Runtime live data and task previews`；见§7 |
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
| Figure archive、validity、selection、unique path、strict JSON、duplicate DeviceSpec、Calibration corruption/reverse dependency | 2 |
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

本Checkpoint随Milestone 1 commit记录完成状态；commit后立即停止本Goal turn，不做Milestone 2设计或编辑。

### 完成证据

- Current consumer audit由三条独立package scope完成；所有删除名在production全树零命中，唯一保留同名是AXI内部私有`_record_diagnostic`与UI自身合法`set_panel_kinds`。
- Runtime/Data：约11,632行旧builder/exact/monitor/live-port/value clusters删除；保留Coverage、ordered future FollowTap、NodeHost/Plane真实路径。Focused 51 passed in 0.72s；Atom/Workbench真实consumer imports通过。
- Plot/UI：dead live controller/channel、FitNumericTable、Graph/FormGrid、test-only methods与self-tests删除；保留`LiveDefaults`、`_session_live.py`、`QtOwnerWake`。Plot 10 passed，UI 15 passed，Workbench selector 2 passed。
- Pulse/Atom：test-only engine model/lease/dead transport与descriptor动态面/fakes/oracles删除。Pulse focused 26 passed；Pulse remote/Atom focused 42 passed；SLM descriptor passed。
- Workbench integration：六个直接文件运行得到193 passed/1个旧Host convenience断言失败；该测试改为真实terminal+draft source后精确节点1 passed；另外8个DeviceClaim/make_host/preview集成节点passed，device-control正式节点1 passed。
- 全部438个剩余Python文件AST解析通过；八包从当前checkout import成功；7个Logic descriptors与LogicCatalog projection成功。
- 全树`pytest --collect-only -q`成功，1395 tests collected，无删除API import/collection错误。
- 当前Markdown file links、notebook JSON、nested pyproject readme targets均有效。
- `git diff --check`无whitespace error；只存在既有line-ending conversion warnings。
- 当前Milestone diff共206 files、739 insertions、22,750 deletions；未新增production file/class/framework。
- 未运行full test suite、real-screen、Vivado/RTL simulator或任何camera/FPGA/SLM硬件。一个旧Calibration纵向在原10秒deadline仍处于保存report阶段，属于既有慢路径，不作为本删除milestone通过证据或新回归声明。

## 6. Milestone 2范围与完成证据

### 唯一truth与corruption修复

- Figure archive只有`zlc_data.figure_archive`一个format owner；v2先规划全部member，再严格验证schema/version、member集合、dtype、shape、duplicate ZIP/JSON、non-finite metadata与嵌入Dataset。Workbench重复常量已删除。
- Validity所有数组入口只接受真实bool；snapshot restriction对values、validity、coordinates、coordinate labels、frame与units执行同一projection。
- Selection按`AxisId`或唯一人类轴名解析，支持text/None exact coordinate；重名拒绝、implicit fractional coordinate拒绝、numeric range跳过missing coordinate。零consumer的第二selection codec已删除。
- `unique_path`不再返回一个可能被其他process同时选中的空名字。文件writer先完成同目录临时文件，再用no-replace publication取得最终编号；目录用exclusive mkdir。全部六个file consumer已迁移，Calibration run folder保留directory模式。
- Readable JSON拒绝non-string key、tuple/unknown、NaN/Infinity与scalar coercion；Camera/Scan/Calibration等真实producer在自己的owner明确输出dict/list/scalar。
- duplicate `DeviceSpec.key`在world contribution、broker与factory side effect前拒绝。
- Calibration不再依赖或手抄Workbench PanelState/save truth。Sample archive只保存typed Dataset+run chain，三帧PNG仍由zlc_plot显式facet grid生成；writer只保留paths，replay拒绝index gap、mixed generation/schema/run record和错误revision。Calibration JSON拒绝unknown/missing fields、duplicate keys、non-finite和类型coercion，外部workflow、主要artifact、default raw policy与三帧report未重设计。

### Evidence

- Old-red根因：32个并发file callers旧实现全部得到同一路径；duplicate DeviceSpec旧实现执行两个factory且只close第二个；figure member可覆盖snapshot validity；numeric validity、duplicate axis name、text coordinate与Calibration corrupt JSON均被旧路径接受或误解。
- 新durable直接证据：32 threads得到32个完整payload；16个独立Python process得到16个唯一文件/不同payload；32-thread directory分配得到32个目录；writer failure无final/temp残余。
- 当前checkout production-path probes通过：Data projection/selection/strict Figure roundtrip与corruption、Workbench archive sequential naming、strict JSON、duplicate-install preflight、Calibration save/load/import boundary/saved-sample replay integrity、panel archive metadata与camera run metadata。
- 当前checkout 420个Python source/test文件`compileall`通过；八个package顶层均从本repo import成功；删除名、旧Figure常量、第二selection codec、Calibration→Workbench import及未迁移file `unique_path`调用的repo搜索均为零。
- `git diff --check`无whitespace error；只有既有line-ending conversion warning。没有新增production file/class/framework/hash、安全层、GPU路径或兼容实现；未访问hardware。
- 当前desktop bundled Python没有pytest、SciPy或Matplotlib，且机器没有另一可运行product interpreter；因此新增focused pytest nodes和三帧PNG真实render未正式执行。它们已完整写入既有test files；本milestone不把compile/direct probes冒充pytest、GUI或hardware evidence。

### Commit与停止门

Milestone 2 commit：`Make dataset, archive, path, and installation truth strict`。

本Checkpoint随该commit落盘；commit后立即停止，不进入Milestone 3。

## 7. Milestone 3范围与完成证据

### 唯一Runtime truth

- Logic Node只提交本次新增的immutable event chunk；`SignalDataPlane`统一分配generation/revision、冻结declaration/run record，并按固定canonical schema与`(repeat, point)` origin累计run Dataset。
- `current_dataset()`按指定publication materialize当前prefix并缓存；未写位置保持invalid。传入publication必须属于当前owner stream与generation，旧run同sequence不能串入新run。
- terminal只seal同一commit history，不再发布第二份Final replacement。正常Stop保留partial；partial materialization失败明确进入`failed`，不伪装成cancelled。
- Exact Processor replay既有event后逐event follow；latest Processor只保留pending latest。不同Processor可并行，同一Processor严格串行。
- UI `freeze()`只投影已提交publication，不调用plugin materializer。旧slot/listener/dirty/pull、`FinalDatasetOutput`、`publish_final`与`SignalValue.transient`路径全部删除，无compatibility alias或双写。

### Plugin与Node contract迁移

- Camera finite/monitor每次只提交一个complete cycle；finite由Runtime形成固定run geometry，Stop partial与Panel/freeze/Processor订阅无关。Repeat=100不再每次stack全部历史。
- Stepped/Seamless Scan只保留schema/placement planner，逐point commit；Runtime拥有future validity与最终Dataset。Temperature的raw scan与per-shot/per-site `survival`同EventRef，删除第二份`survival_rate` Dataset和私有history；preview/artifact从同一survival truth投影。
- Calibration long/readout/long三帧每cycle直接commit为Monitor preview；删除私有preview slot，terminal按Runtime存在性retire；artifact save前进入不可取消terminal边界。
- Occupancy只处理source event chunk，严格传播source validity；counts/occupied/valid/rate继承canonical placement，`frame_judged`复用source bytes。Descriptor与processor复用同一`OCCUPANCY_OUTPUTS` typed vocabulary。
- Descriptor outputs直接使用`DatasetOutputDeclaration`；Processor delivery显式为exact/latest；Task必须显式声明preview或`()`. 所有production Measurement都声明live Dataset和auto-preview，Host在terminal强制完整declaration union、Task progress与artifact存在。
- SLM Feedback每个coarse/validation使用稳定`<task>/camera`的canonical Camera Measurement `repeat=N` generation。Estimator与preview共用readout frame coordinate 1；preview对同一raw Dataset执行repeat mean。删除私有camera average/readout signal。Task own outputs只保留latest candidate phase与完整history projection；Stop接受并保存本run最好有效测量phase，真实失败恢复incoming。

### Workbench与presentation

- Panel currency由primary与全部companion EventRef共同决定；companion-only变化会更新。Generation replacement先在detached host完成完整配置，再由same-shot cohort统一accept；失败、panel remove、resolver丢失或board close都会关闭/retire staged host，不提前替换当前画面。
- Typed `NodePreviewSpec`的output/producer/semantic由Workbench完整消费；首个真实publication前signal显示waiting，explicit incompatible kind报错且不伪装成功。Terminal progress清除，preview按Runtime signal存在性retain/retire。
- Task运行期间禁止Add/Start/Remove Logic与改变Logic draft/device/source/preview policy；允许新增与操作其它Panel。Task own/companion/overlay Panel冻结signal、overlay、cell kind、semantic、selector与threshold，保留title/size/interval/display/fit/viewport/save等纯显示操作。Calibration仍显示三帧facet。
- 默认Camera/event Panel显示latest event；显式`fate:repeat`/`scope:repeat`才读取Runtime current Dataset进行全history reduction/scope，避免普通preview自动走O(N²)路径。

### Evidence

- Runtime全套：`85 passed`；含跨generation publication拒绝、exact replay/EOS、mixed exact+Monitor、nonblocking materialization、Stop seal failure、无订阅/五次freeze+current读取/exact Processor三种Stop partial完全一致、跨Processor并行与节点内串行/latest coalesce。
- Repeat=100 deterministic证据：commit/freeze期间prefix materialization为0；100个float event只发布800 bytes，Runtime只保留100个one-cell event及100-cell occupancy mask（values+validity共1000 array bytes）；第一次`current_dataset()`只materialize一次，seal不重复。Camera 96×128 uint16的100个cycle event总payload为2.34375 MiB，而旧prefix-stack路径累计约234.375 MiB。
- Atom主纵向在最终修复前后累计验证：Camera/Scan/Calibration/Occupancy/Temperature组73项通过；SLM及直接受影响Camera/Scan/Temperature最终组54项通过，其中SLM完整26项通过。真实virtual Calibration→Temperature、SLM canonical Camera/plot semantic、Stop-before-first/accepted/failure与artifact/device一致均覆盖。
- Workbench/Runtime presentation、same-shot、selection、Logic/Task/preview与Guard-A组最终`159 passed`；`zlc_ui` console views `30 passed`。Staged replacement lifecycle old-red 6项在修复前失败，修复后remove/close/cancel/reaper/cohort回归10项通过。
- 全仓`pytest --collect-only -q`成功，`1465 tests collected`；250个production Python文件AST解析通过；八包current-checkout import与active README links通过。
- Production legacy API搜索为0；仅一个测试保留“Camera execute源码不得重新引入旧API”的负断言。`git diff --check`无whitespace error，只有既有line-ending warning。
- 未运行full 1465-test suite、real-screen、real camera/SLM/FPGA、Vivado/RTL或任何hardware side effect；本milestone不把focused virtual evidence冒充实验机验收。

### Commit与停止门

Milestone 3 commit：`Unify Runtime live data and task previews`。

本Checkpoint随该commit落盘；commit后立即停止，不进入Milestone 4，等待用户明确确认。

## 8. Checkpoint更新规则

每个milestone开始/完成、长测试前或新用户裁决后立即更新：

- status；
- current HEAD/dirty；
- exact decision；
- focused evidence与路径provenance；
- commit subject（commit hash通过`git log --grep`解析，文档不自引用自己的hash）；
- next unfinished action。

不得恢复append-only历史日志，不把对话摘要当状态，不把passing test数量冒充未覆盖行为完成。
