# ZLC v2 Architecture Convergence — Implementation Plan

本文只保存当前实施状态、阶段证据和下一步，不保存旧Goal历史。目标架构见`ARCHITECTURE_DESIGN.md`，完整范围见`ZLC_V2_IMPLEMENTATION_GOAL.md`。

## 1. Persistent Checkpoint

更新时间：2026-08-21
启动HEAD：`af54e24787de67270c54eb154f2b23f43508fc3e`
Branch：`master`
当前HEAD：`9571cd6 Cache smart ticks and make camera units capability-aware`
用户执行边界：Milestone 1–6、Plot projection/fit baseline及Axis/camera/style follow-up均已提交。当前获批Goal只处理Histogram/tick配置、SimulationWorld ownership、registered SLM Feedback与SLM local/remote owner；不进入M7，不访问真实SLM或其它hardware。

### 当前状态

- 审计：完成；逐文件证据在`AUDIT/`。
- 用户裁决：完成；记录在`AUDIT/USER-DECISIONS-2026-08-17.md`与根Goal。
- Production代码：Milestone 1–6及其residual closure均已完成。
- Hardware：未访问；本Goal不授权program/flash或实验机device操作。
- Milestone 0：`COMPLETE` — commit `e854ddf`（`Establish approved architecture and implementation checkpoint`）；唯一Architecture、Plan、Handoff、Goal与Audit证据已纳入版本控制，未改production。
- Milestone 1：`COMPLETE / SWEEP COMPLETE` — test-owned Pulse engine、dead Qt wake、compat aliases、false-green API/notebook/dependency tests与重复contract docs已删除。
- Milestone 2：`COMPLETE / SWEEP COMPLETE` — nested immutable truth、strict embedded manifest grammar、archive-first Panel Save与唯一Figure入口已闭合。
- Milestone 3：`COMPLETE / SWEEP COMPLETE` — replay lineage、selection lock域、canonical presentation/overlay alignment、Refresh/layout lifecycle及测试残余已闭合。
- Milestone 4：`COMPLETE / SWEEP COMPLETE` — exact pair/resync、原子PanelState、canonical selector/threshold、streaming archive以及Qt worker/close均已闭合；全树1498 tests通过。
- Milestone 5：`COMPLETE / SWEEP COMPLETE` — Pulse执行词汇、camera cadence、Remote takeover、Stop/SAFE、RTL/build与strict transport均闭合。
- Milestone 6：`COMPLETE / SWEEP COMPLETE` — USB-only SLM、command truth、Science Context、robust Feedback与immutable Simulation已闭合。
- Plot performance closure：`COMPLETE / SWEEP COMPLETE` — active+latest solve timeout、live FitEvent/raster解耦、renderer/fit直接热路径、重复测试及冻结树残余均已闭合。
- Plot projection/fit baseline：`COMPLETE / SWEEP COMPLETE` — commit `55d6ee7`；large projection、Histogram/Rolling pool、regular-image bounds/exact retry与最终两处first-frame copy均闭合。
- Axis/camera/style follow-up：`COMPLETE / SWEEP COMPLETE` — SmartOffset steady memo、capability-dependent camera unit、Calibration/Occupancy unit lineage与Image fit occupied-ring style均已闭合；全树验证green。
- SLM local/remote owner：`IMPLEMENTED / SWEEP COMPLETE` — `slm.remote`只在installation握手及phase command联网；bounded length-prefix/strict-JSON/raw-float32 server独占本机USB adapter，profile/correction固定在server端，expected command/mapping revision拒绝陈旧写入；`bin/slm_server.bat`为本地/LAN同一入口。healthy/stale/failure/malformed/oversize/timeout/concurrent/close定向17 passed，未访问hardware。
- Histogram/tick配置：`IMPLEMENTED / SWEEP COMPLETE` — bins继续exact一次必要reprojection；density/cumulative只重画bins且不再扫描full payload；qCMOS P50从73.049/71.301 ms降至7.448/7.709 ms，bins 71.539→71.957 ms无material change，10组DPR1/2 RGBA exact。旧settled tick lattice在枚举前限界，百万级range切换不再卡死。Plot全套423、formal/focused72 passed；production净增17，无新class/kind lane。
- SimulationWorld ownership：`IMPLEMENTED / SWEEP COMPLETE` — apparatus root `simulation`唯一拥有image/grid/seed/profile；qCMOS只消费world geometry，MOT geometry独立；workspace-local profile containment、strict legacy拒绝、DeviceManager preservation和双owner拒绝均闭合。最终定向28 passed，无新production file/class。
- Registered SLM Feedback：`IMPLEMENTED / SWEEP COMPLETE` — Context v2是run的唯一frozen Target truth；Target JSON只作Editor authoring，Feedback inputs只有Calibration+Context，Calibration optional registration也只选Context。唯一topology validator在创建/load时重核physics、BOX overlap/bounds和runner-up uniqueness，alternate mapping要求asymmetric fiducial。ReadoutModel持久化dark n/variance；Feedback使用raw BOX-dark、raw/electron provenance与converted saturation、sites×looks correction、最多3个coarse batches/3次power-preserving boost，Stop-before-solve为0 solver calls，无valid result保留incoming candidate 0。相关12 files：production `+1163/-282`（净+881）、tests `+924/-91`（净+833）；无新file/class/lane。6个新production defs仅为`validate_target_registration`、`_register_target_sites`、`_censored_sites`、`_boost_target`、`_incoming_candidate`和`_coarse_measure`，均有直接owner/consumer。Owner Feedback 34/43.68 s、physics/truth 24、descriptor 3以及独立Feedback+truth 41/45.53 s passed，12 files AST与diff-check green；未访问hardware/optical。
- 本Goal combined Gate 17/18：`COMPLETE / SWEEP COMPLETE / READY TO COMMIT` — 冻结candidate 38 files `+3702/-638`（净3064）；production 18 files净+1453、tests 14 files净+1476、active docs 5 files净+130、launcher 1 file +5。Test functions净增16，均覆盖独立业务边界；无剩余production/test blocker、dead owner、旧API、double truth或已知safe deletion。首轮全树是`1555 passed, 1 failed, 4 skipped, 6 warnings in 475.07 s`，唯一failure为既有gallery offscreen Qt teardown subprocess exit `0xC0000005`，不冒充green；随后该exact gallery独立三次全pass（`1.58/1.52/1.52 s`），第二次完整全树`1556 passed, 4 skipped, 6 warnings in 466.48 s`。Warnings仅既有vendor SWIG deprecation，skips仅本机无Icarus。Milestone 7仍`PENDING`，single distribution、wheel/fresh install、evidence lanes和final docs等待用户后续指令。

### Milestone 3完成边界

Milestone 3主体从clean HEAD `23e820d`开始并由`ca66c7d Unify Runtime live data and task previews`落盘。验收follow-up从clean HEAD `ca66c7d`开始，由`Fix canonical full signal presentation`落盘；未访问hardware，未进入Milestone 4。

### 当前停止门

Milestone 6、Plot performance closure、commit `55d6ee7`及`9571cd6`均已完成。当前只执行用户批准的四项follow-up并统一commit；不自动进入M7，不得运行真实SLM、camera或FPGA side effect。100 ms仍是Plot profile警戒线，不是硬验收门。

## 2. Milestone状态

| Milestone | Scope | Status | Evidence / Commit |
|---|---|---|---|
| 0 | Approved Architecture、current Plan/Checkpoint、Handoff、Audit evidence | COMPLETE | `e854ddf`；docs/link/diff check，未跑测试 |
| 1 | Dead framework、parallel pipeline、test-only surface、duplicate launcher/docs/tests删除 | COMPLETE / SWEEP COMPLETE | 原commit + residual fix；见§8 |
| 2 | Data/Durable/Installation truth | COMPLETE / SWEEP COMPLETE | 原commit + residual fix；见§8 |
| 3 | Canonical Runtime live与Logic Node contract | COMPLETE / SWEEP COMPLETE | 两个M3 commits + residual fix；见§8 |
| 4 | Exact Data/Fit/Overlay与Qt lifecycle | COMPLETE / SWEEP COMPLETE | `Make data, fit, overlay, and Qt lifecycle exact and atomic`；见§9 |
| 5 | Pulse/Camera/Remote/FPGA | COMPLETE / SWEEP COMPLETE | `Close pulse, camera, remote, FPGA stop, timing, and ownership semantics`；见§10 |
| 6 | USB-only SLM与robust Feedback | COMPLETE / SWEEP COMPLETE | `Converge USB SLM context, feedback, and simulation truth`；见§11 |
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
- Plot/UI：dead live controller/channel、FitNumericTable、Graph/FormGrid、test-only methods与self-tests删除；保留`LiveDefaults`与`_session_live.py`；回溯又删除零consumer `QtOwnerWake`及其self-test/aliases。Plot/UI/Workbench closure focused 66 passed。
- Pulse/Atom：test-only engine model/lease/dead transport与descriptor动态面/fakes/oracles删除。Pulse focused 26 passed；Pulse remote/Atom focused 42 passed；SLM descriptor passed。
- Workbench integration：六个直接文件运行得到193 passed/1个旧Host convenience断言失败；该测试改为真实terminal+draft source后精确节点1 passed；另外8个DeviceClaim/make_host/preview集成节点passed，device-control正式节点1 passed。
- 全部438个剩余Python文件AST解析通过；八包从当前checkout import成功；7个Logic descriptors与LogicCatalog projection成功。
- 最终全树`pytest --collect-only -q`成功；精确current数量见§8，不以旧collection数字作当前证据。
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
- 当前checkout production-path tests通过：Data projection/selection/strict Figure roundtrip与corruption、formal explicit-path Panel archive、strict JSON、duplicate-install preflight、Calibration save/load/import boundary/saved-sample replay integrity、panel archive metadata与camera run metadata。
- 八个package顶层均从本repo import成功；删除名、旧Figure常量、第二selection codec、Calibration→Workbench import及未迁移file `unique_path`调用的repo搜索均为零。最终AST/collection数字见§8。
- `git diff --check`无whitespace error；只有既有line-ending conversion warning。没有新增production file/class/framework/hash、安全层、GPU路径或兼容实现；未访问hardware。
- 回溯环境已具备当前checkout pytest/SciPy/Matplotlib；strict Data/Durable/Calibration/Installation与formal Panel Save行为均已实际执行，精确结果见§8。仍未访问hardware。

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
- Occupancy只处理source event chunk，严格传播source validity；counts/occupied/frame_judged继承canonical placement。`occupied` bool Dataset自身component validity是site readability的唯一truth；不再发布第二个valid或可由它直接求得的rate signal。Generic image-point contract、geometry validator与adapter归`zlc_plot`；Workbench只按contract和causal lineage路由，不import Occupancy。
- Descriptor outputs直接使用`DatasetOutputDeclaration`；Processor delivery显式为exact/latest；Task必须显式声明preview或`()`. 所有production Measurement都声明live Dataset和auto-preview，Host在terminal强制完整declaration union、Task progress与artifact存在。
- SLM Feedback每个coarse/validation使用稳定`<task>/camera`的canonical Camera Measurement `repeat=N` generation。Estimator与preview共用readout frame coordinate 1；preview对同一raw Dataset执行repeat mean。删除私有camera average/readout signal。Task own outputs只保留latest candidate phase与完整history projection；Stop接受并保存本run最好有效测量phase，真实失败恢复incoming。

### Workbench与presentation

- Panel currency由primary与全部companion EventRef共同决定；companion-only变化会更新。Generation replacement先在detached host完成完整配置，再由same-shot cohort统一accept；失败、panel remove、resolver丢失或board close都会关闭/retire staged host，不提前替换当前画面。
- Typed `NodePreviewSpec`的output/producer/semantic由Workbench完整消费；首个真实publication前signal显示waiting，explicit incompatible kind报错且不伪装成功。Terminal progress清除，preview按Runtime signal存在性retain/retire。
- Task运行期间禁止Add/Start/Remove Logic与改变Logic draft/device/source/preview policy；允许新增与操作其它Panel。Task own/companion/overlay Panel冻结signal、overlay、cell kind、semantic、selector与threshold，保留title/size/interval/display/fit/viewport/save等纯显示操作。Calibration仍显示三帧facet。
- `ca66c7d`错误地让默认finite Panel显示latest event，并用`fate:repeat`/`scope:repeat`决定是否读取current Dataset；真实UI验收证明Logic shape、默认有效semantic、运行中fate切换与host schema彼此矛盾。Follow-up删除该政策：finite exact所有显示统一canonical full，Monitor才latest，semantic不选择数据truth。
- Runtime仍只保存并向exact Processor交付event chunks；Logic shape直接报告canonical schema，Panel presentation按自己的cadence在board worker materialize/cache对应publication的full prefix。首次mount、live update、semantic切换、Edit/Refresh/Save、selector、fit与overlay共享同一个accepted snapshot，Qt owner与producer都不执行full-prefix copy。
- Repeat与point rows使用同一个canonical placement原则；多维scan依靠PointTable/GridTopology逐point填充，future cells invalid。Cell payload保持原子完整，不增加cell内部tile/slice streaming层。
- Generic overlay只保留一个numeric/bool status Dataset；values表达EMPTY/OCCUPIED，Dataset validity表达INVALID。Geometry绑定image axes与完整status data axis。只有scope/facet唯一选中一个cell时才画离散状态，pool或future cell为UNKNOWN，不做跨cells共识。

### Evidence

- Runtime全套：`96 passed`；含canonical repeat与point/grid geometry、publication-prefix race、跨generation拒绝、late exact replay causal roots、mixed exact+Monitor GC、nonblocking materialization/Selection close、Stop seal failure、跨Processor并行与节点内串行/latest coalesce。
- Repeat=100 deterministic证据：commit/freeze期间prefix materialization为0；100个float event只发布800 bytes，Runtime只保留100个one-cell event及100-cell occupancy mask（values+validity共1000 array bytes）；第一次`current_dataset()`只materialize一次，seal不重复。Camera 96×128 uint16的100个cycle event总payload为2.34375 MiB，而旧prefix-stack路径累计约234.375 MiB。
- Atom直接受影响组`46 passed, 1 deselected`；Camera/Scan/SLM/Occupancy/Temperature和generic geometry的定向组通过。唯一deselect是既有Calibration完整报告纵向仍超过原10秒deadline；它未作为通过证据，也未用增加timeout或改产品逻辑掩盖。
- Workbench Console/Task/preview/presentation/same-shot/Guard-A/Guard-C/Viewer相关整组`144 passed`；`zlc_ui` console views `30 passed`。Repeat=30真实virtual Camera证明event保持单cycle、Logic/full Panel为30 repeats，axis fate在同host/同publication即时生效；gated materializer下owner update为2.489 ms、beat submission为0.666 ms，copy发生在另一线程。
- `zlc_plot`generic overlay定向3项及public API/surface 8项通过；repeat/point scope/facet、pooled/future UNKNOWN、status-axis重排拒绝与single-status archive round-trip均有窄测。
- 全仓`pytest --collect-only -q`成功，`1485 tests collected`；37个changed Python文件AST解析通过；八包从current checkout import成功；active documentation relative links通过。
- Production对`growing`、旧plugin overlay/valid signal与Workbench concrete Logic Node import的搜索为0。`git diff --check`无whitespace error，只有既有line-ending warning。
- 未运行full 1485-test suite、real-screen、real camera/SLM/FPGA、Vivado/RTL或任何hardware side effect；本milestone不把focused virtual evidence冒充实验机验收。

### Commit与停止门

Milestone 3 commits：`Unify Runtime live data and task previews`；验收follow-up `Fix canonical full signal presentation`。

该follow-up当时随commit落盘并按门停止；之后仅在用户明确确认后才开始Milestone 4。

## 8. M1–M3 post-milestone residual closure

### M1 closure

- 删除从production `engine_model.py`搬入tests的218行Python engine oracle及其self-simulating wrap/refill/underflow tests；这些行为只能由后续RTL/hardware evidence证明。
- 删除零consumer `zlc_ui.concurrency/QtOwnerWake`、Handle `show()` aliases、Pulse dataset aliases、第二TaskConsole window入口、arbitrary API cap、source-token/doc-`__all__`同步、saved-output notebook与手抄dependency tests。
- Durable内部functions撤出root facade；两份重复contract name lists删除。Data headless subprocess显式绑定并打印current checkout。

### M2 closure

- Calibration topology/report/summary/pulse/run record与Installation config全部deep-own并递归immutable；plain codec输出独立tree；`TrapCalibration.save()`返回durable resolved path。
- `snapshot_from_manifest()`不再用字段缺失猜grammar：standalone manifest严格要求schema fingerprint，embedded figure manifest显式声明并禁止重复fingerprint。
- 删除`ExperimentSession.save_figure`与Workbench legacy `write_figure`第二保存入口。Formal Panel Save先原子写不可重算`.npz`，再render image；image失败明确报告archive已保存，archive失败不启动render。

### M3 closure

- Exact replay保留slim causal parent chain，late Processor仍追溯原producer且未消费Monitor sibling可GC；run record始终deep-freeze；commit value/origin表合并。
- SelectionBridge numeric materialization移出RLock并以epoch CAS拒绝迟到结果；close不等待大Dataset/ROI。
- Generic overlay严格核repeat/PointTable/GridTopology/status axis与same-shot lineage；删除零consumer `PointMarker`与Occupancy rate。单pixel axis不再猜affine step。
- Panel layout首帧回到cohort atomic present；三份Port wiring收成一个owner；Refresh在新surface接受前保留旧Edit/Save snapshot，同revision重复Refresh为no-op；tests-only host/shown Port路径及event-snapshot fallback删除。
- 原用户场景现由真实running Camera gate证明：首个shot后Logic/Panel已经是完整repeat=30 geometry、future invalid，运行中改fate不Restart，后续shots继续填同一个host；point/grid亦在mounted状态继续增量。

### Closure evidence与净变化

- Fix candidate相对`0cbc514`：73 files，`+1840/-2910`，净删1070行；其中production净删67、tests净删825、docs/notebook净删173、其它净删5。相对M3起点`23e820d`，production净增556、tests净增1993，但top-level test函数只净增18项；新增主要是Runtime/Qt/lineage真实纵向，不再含tests-only Port/engine/private-helper保活。
- Runtime `96 passed`；Data `43 passed`；Plot `17 passed`；Pulse+Durable `33 passed`；M1 UI focused `12 passed`；Workbench changed-path组`164 passed`后两条正式Qt subprocess回归修正并`2 passed`；Atom M1–M3 focused组`57 passed`，最新Calibration immutable/result定向`3 passed`。
- 全仓collection：`1462 tests collected`。八包current-checkout imports、changed Python AST、active relative links、notebook JSON和`git diff --check`在commit前最终复核。

### 已明确归属后续milestone，不算未记录残余

- M4：Plot-owned原子PanelState operation、首mount多front、atomic data+fit pair与active+latest admission、selector Off/FacetGrid overview、Viewer唯一PanelState parser、Qt/executor bounded close与whole-archive bytes materialization的性能/owner收口。
- M7：单distribution package-data（含scan JSON templates）、fresh notebook lane、`save_npz(path)` persistence/public-API policy、Calibration/figure format version migration与最终安装文档。

Fix commit：`Fix post-milestone residuals`。该commit之后经用户明确确认才开始Milestone 4；目前M4也已完成。

## 9. Milestone 4 completion与residual closure

### Exact pair、selector与presentation

- `display_interval`只控制Surface deadline；Runtime普通indexed-derived Dataset保存每个source primary index及validity。Surface同一same-shot group只保持一个active完整front，忙时只保留Plane latest和admission debt，中间indices invalid且不排完整frame。
- PlotSession只保留一个serial analysis executor。prepare、manual fit与live fit不再由第二executor或package-global stripe/facet pool并行；fit request、warm state、accepted fit与render仍由PlotSession唯一拥有。
- PanelState作为一个原子Plot target应用：no-op为0 solve/0 render/0 front；title/interval不触发Plot；display/semantic/fit各最多一个完整front；最终render失败会完整rollback，旧fit cancellation/Future settlement只在commit后发生。Startup initial front先于adaptive task，same-front Qt handoff幂等，不再依赖ghost front。
- Fit范围唯一优先级为committed Area ROI（或X-range）→viewport→full range。Plot-native selector在Workbench ack前保持authority；FacetGrid selector保留focused-cell identity。真实Area后立即点击Fit的old-red证明26×26 ROI、676 samples，不会被空PanelState重放成full frame。
- SelectionEvent携带generation/revision、canonical/display bounds、scope/facet/repeat meaning；Workbench不再读取Plot private projection。Restored selector先在Panel已接受publication N上产生derived truth，再于同generation追到Plane latest N+1并继续N+2；processor completion直接唤醒现有Board owner turn，不等待下一periodic beat。
- Per-facet classifier threshold以axis id+canonical coordinate为唯一plural truth；删除最后一个index/scalar truth与`PanelPlotAnnotations` archive。Selector Off禁全部plot pointer gesture并把wheel交给board；FacetGrid overview在On时也只允许double-click focus，focused cell才接受area/pan/zoom。

### Archive、Qt与owned lifecycle

- Figure archive唯一writer改为`write_figure_archive(BinaryIO)` streaming encoder；Workbench与Calibration把same-directory atomic temp stream直接交给它，不再构造archive-sized bytes。Formal Panel Save冻结同一PanelState/frozen snapshot/exact viewport，后台archive-first再render；失败不会留下缺science data的图片。
- FigureViewer的open/dataset/configure/resize/save/close全部completion-driven：candidate先成功mount再retire old，失败恢复last accepted figure；saved fit、coordinate thresholds与exact Figure viewport通过同一次atomic configure恢复。
- TaskConsole/LiveBoard、Device Manager/Experiment flow、Pulse SAFE/Preview与Panel Save各由明确existing owner关闭。Qt slots不执行I/O、fit、device tune或等待Future；window在owned worker/claim/host真实退出前保持可见并诚实拒绝close。ExperimentSession即使installation close失败也会关闭Plane。
- Generic image overlay仍由`zlc_plot` contract/geometry/adapter拥有；Workbench没有concrete Logic Node import，Data/Fit/overlay都从同一accepted canonical snapshot与same-shot lineage投影。

### Profile与验证

- 正式低扰动offscreen TaskConsole链使用真实100 ms timer、96×128 Camera、主图26×26 Area radial fit、并行ROI image与fit amplitude Rolling。三次fresh 60-revision run的clean HEAD→candidate same-shot joint P50/P95为76.46/99.95→74.85/93.10 ms，三轮合并Rolling为93.11/113.03→87.83/96.15 ms；174/180 valid，6个真实solver invalid，busy miss、完整frame FIFO和Panel error均为0。100 ms是profile警戒线，不是硬验收门。
- 300/300 timeline样本的每个stage恰好一次：admission P50约3.3 ms、sampled P95约54 ms；FitEvent→Rolling 0.52/约38.7 ms；last-promote→owner-accept 0.34/1.33 ms。Generation-2 GC每轮一次、耗时78–81 ms，只影响maxima，不改变上述percentile。
- First mount的1/4/8 panels从5/20/40 fronts降为2/8/16（initial+真实DPR）；title为0 front，display为1，fit arm为1 solve+1 front，fit-armed Edit为2。invalid target与final-render failure均保持旧state/front。
- 全仓current checkout八包一次完整运行：`1498 passed, 6 warnings in 427.82s`；warnings仅vendor SWIG deprecation。Runtime+Plot单组484 passed，UI+Workbench单组488 passed，Atom全套310 passed。
- 81个changed Python文件AST解析通过；八包import均解析到本checkout；59个active Markdown relative links全部存在；旧API/alias/test-only wake与pending probes搜索为0；无新增production file/class，删除3个历史文件；`git diff --check`无error，仅line-ending warnings。未访问hardware、real camera/SLM/FPGA或real-screen。

### Gate 17 consumer/test/docs sweep

- Candidate相对启动HEAD `af54e24`：98 files，`+7965/-3349`，净增4616行；其中production 46 files `+4070/-2199`（净增1871）、tests 38 files `+3657/-1035`（净增2622）、active docs 14 files `+238/-115`（净增123）。
- Production增量主要是Plot exact-pair/atomic transaction/coordinate selector（zlc_plot净增1046）与Workbench completion-driven Console/Viewer/Save/device/Pulse ownership（zlc_workbench净增736）；Runtime净增74用于publication catch-up与owed presentation，UI与Data净增0，Atom净增15。没有新增file/class或plugin-specific Workbench framework。
- Test分类：KEEP真实Camera Area→main fit→invalid gap→Rolling断线、无successor的worker active timeout、direct Host capacity-one、atomic rollback/no-op、restored N→latest catch-up、formal Qt Save/Viewer/close与Selector interaction；KEEP tight-colorbar一次更新及Image/Curve/Histogram/Rolling/FacetGrid DPR1/2逐像素parity；MERGE clear-after-solve进Gaussian/Lorentzian/Decay callback-before-render race；DELETE旧wait/bytes多frame队列和private setter-count白盒。原M4阶段全树test函数从1246增至1270；本次closure最终函数/LOC由root冻结树重算。
- Consumer/owner sweep删除package-global fit pool、second prepare/fit executor、`apply_panel_fit`/`compose_panel_spec`、`PanelPlotAnnotations`、OwnerChannels wrapper、SelectionBridge test-only Event/introspection、arbiter/port pending probes及同步Figure/Pulse preview路径；active旧名均为0。

### 明确defer与停止门

- 用户随后授权的M4 cleanup已完成：删除单消费者`_WorkerSessionAdapter`并让Host直接调用现有PlotSession owner，production净删260行；九个高增长测试文件及一条gesture白盒共净删538行。后续performance closure又删除旧wait/bytes多frame队列测试和setter-count白盒，保留ROI→Fit→invalid gap→Rolling、无successor active timeout、capacity-one、atomic rollback、Viewer/Pulse/Qt close、五kind像素parity与coordinate threshold核心证据。
- 100 ms只保留为正式链profile警戒线；早期不同source harness的数字已删除，不作为A/B基线。可信同harness结果记录在下方Performance follow-up。
- 不可中断的vendor discovery保持window可见并拒绝close，直到真实future结束；hardware transport cancellation/priority属于M5，不把`shutdown(wait=False)`冒充安全退出。
- 普通Pulse Stop/FIRE wire priority、Camera/Remote/FPGA归M5；SLM USB/context/feedback归M6；single distribution/fresh install/notebook与final docs归M7。M4未访问hardware，也不把offscreen/virtual证据冒充实验机验收。

Milestone 4 commit：`Make data, fit, overlay, and Qt lifecycle exact and atomic`。提交后立即停止，不进入M5，等待用户的Plot/Fit性能讨论。

M4 cleanup follow-up commit：`Remove residual M4 adapters and duplicate tests`。复盘根因不是缺少规则文字，而是错误地把各agent分片收口当成合并树收口，并用full green、无新增class、test函数净增数替代新增definition/state/consumer与测试正文LOC审计。`AGENTS.md`现强制所有cut合并并冻结candidate后重新独立审计；已知safe deletion不得在标记sweep-complete时延期。

### Performance follow-up

- Runtime新增中立`primary-index` indexed-derived Dataset：每个Measurement source index有value或invalid，普通Monitor仍latest；64 MiB/100k retention按display请求lazy materialize，10,000 publications的window=100为0.393 ms/900 bytes。所有Plot读取同一OwnedSnapshot；普通Plot默认latest，声明window的history projection读取同一axis；同publication扩大window或改fate立即原子rematerialize，Save冻结同一Dataset。
- Surface admission改为capacity-one same-shot group：任一member仍有重绘在途就不排第二张完整frame，只留Plane latest与admission debt；atomic publication wake在deadline已到时立即stage，Pause/closing不admit新Surface。Raster Host同步删除旧多frame/bytes队列，只保留active+latest；现有worker Condition在active超过1秒时不依赖successor到达即取消、loud发布invalid并继续latest。TaskConsole删除第二个`--interval-ms`时钟真相。
- Renderer只跳过未变化的artist/chrome写入并缩小colorbar dynamic set，不改变图形：isolated P50 Image+fit 34.37→32.49、Curve 2.74→2.40、Histogram 3.18→2.82、Rolling 16.55→15.18、FacetGrid(4) 11.84→9.59 ms；五kind DPR1/2、Area/Fit与tight colorbar逐像素一致。Series/Histogram在全局不可超越warm seed上P50下降约62–91%，cold与Image fit不回退。
- 正式100 ms链的可信A/B与stage timeline见上方Profile：joint和Rolling P95均改善，174/180 valid+6 solver invalid，0 busy miss/FIFO/error；live FitEvent在exact solve后、owner raster前发布以允许Rolling并行，manual fit仍在accepted overlay后通知，main visual保持atomic `data@N + fit@N`。
- 已评估并拒绝三个方向：third foreground layer全矩阵理论仅约1.5–4 ms，增加像素顺序风险且prototype留下0 residual；独立wall-deadline admission scheduler在0 admission miss/owner bottleneck下只增加state；full backend ingress需约300–500行而当前busy miss为0。均不实施。
- 已提交的performance commit `69d5514..5be8bb7`真实为36 files `+1585/-214`（净增1371）：production 20 files净增824、tests 7 files净增525、docs 9 files净增22；旧Checkpoint少记22行，现已更正。本次final closure冻结树为19 files `+840/-503`（净增337）：production 7 files净删35、tests 4 files净增333、docs 8 files净增39；无新增production file/net class/kind/model lane，tests删除多frame队列政策及重复白盒，新增函数净数只覆盖pixel parity、fit order/race与active+latest生命周期。Plot全套407、Runtime/Workbench跨层140、全树1524 passed/4 skipped/6个既有vendor warnings。

Performance commit：`Make derived history continuous and presentation capacity-one`。完成后进入M5，不把indexed Dataset、gap或window放进任何Plot-kind/Logic plugin/Workbench专用lane。

### Projection/Fit baseline follow-up（2026-08-21）

- 用户已完成手工验收并授权把当前baseline作为独立commit落盘；本节记录的是文档reconciliation前冻结的code/test candidate：7 files `+735/-78`（净增657），其中production 3 files `+290/-75`（净增215），tests 4 files `+445/-3`（净增442）。无新增file/class；production只新增2个private helper和1个DataView cache state，均有唯一owner/consumer；无plot-kind、fit-model或Workbench专用lane。
- Dense singleton Image保留native values与boolean validity；连续Facet cells保留views，sparse domain只gather请求位置；integer Histogram用native min/max与严格aligned `bincount`，所有float/nonuniform/sparse-span/超`int64`情况保留`np.histogram` fallback；ungrouped Rolling复用一个valid pool，repeat/primary-index history仍按source index语义投影。R=1初始history不再先建全尺寸positions，dense Histogram facet不再boolean-copy contiguous cell。
- Regular-image默认radius lower改为native sampling resolution/2，moment bounds仍拥有upper与其它参数，显式bounds最终覆盖。Full-image cold candidate已经在exact objective成功时，若随后retry只以相同cost返回failure，不再把成功FitResult/FitEvent误写成invalid；只有无实质cost改善时保留原成功，规则由separable kernel metadata共享，不按model id特判。
- 同一fresh-process 10-case A/B中，2048² Image complete P50/P95为35.42/38.05→21.60/22.39 ms，Facet×4为183.48/191.71→24.05/27.13 ms，Facet×8为366.48/377.69→45.24/46.98 ms，Histogram为175.79/178.02→33.13/34.80 ms，Rolling为111.80/112.77→15.14/17.30 ms；完整矩阵与allocation peak见`packages/zlc_plot/docs/performance.md`。R=1 2048² initial pool为11.02/12.79→0.117/0.130 ms、32.01→0.005 MiB；4×1200×1920 Histogram facet为50.43/51.13→40.15/41.26 ms、19.79→17.59 MiB，numeric parity exact。
- 当前验证：Plot全套`418 passed`；跨层定向`5 passed`；最终fit gate focused `50 passed`，data/facet/history focused `96 passed`；全树`1535 passed, 4 skipped, 6 warnings`，warnings仍仅既有vendor SWIG deprecation。独立随机矩阵为194个dense/sparse/grid/validity/group/reduction projection checks、954个Histogram fast/fallback cases、21个Fit parity/success cases，另有repeat 2/3/7与primary-index exact parity；`uint64 > int64.max` fallback blocker已回修。`git diff --check`、AST与active old-name scan均green。
- Gate 17最终分类：新增9个test functions与1个shared fixture helper全部KEEP；唯一重复dense-facet proof已先并入existing route test并净删11行，之后MERGE/DELETE为0。`_all_positions`、`_flat_cache`、`_masked_leading_reduce`、`_histogram_from_positions`、generic grouped/repeat/primary paths和moment bounds均有可达fallback consumer，不存在待删dead branch。该冻结baseline已由commit `55d6ee7`落盘。

### Axis/Camera/Fit-style follow-up（2026-08-21）

- 本次active docs同步前的冻结candidate为16 files `+349/-94`（净增255）：production 8 files `+90/-29`（净增61），tests 8 files `+259/-65`（净增194）。无新增file/class/plot-kind/model/device lane。Production只增加Camera Measurement的2个derived properties、SmartOffset locator的1个cache key与SampleWriter的1个unit state；Image fit同时删除2个重复ring style字段。
- `SmartOffsetLocator`只对完全相同的directed range、axis/figure、以point表示的drawn extent与orientation、locator policy、label size、measure与font输入复用完成的tick layout；range direction、resize、font或policy变化仍重算。三进程A/B的Image、ROI、Curve、Histogram、Rolling与FacetGrid完整P50均改善；精确P50/P95、`_unit`/`_lay_out` call reductions和唯一Rolling P95 tail变化见`packages/zlc_plot/docs/performance.md`。
- Camera Measurement与camera-backed Calibration默认请求photoelectrons；有完整device conversion时发布dimensionless `float32`，无conversion时control unavailable、effective False并发布native raw `count`，authored draft保持不变。Snapshot、run record、Calibration preview/archive/replay与Occupancy numeric counts继承同一effective unit。三条Dataset translator和SampleWriter都要求调用方显式传unit/effective mode，不保留能生成错误unit truth的默认值。
- Image fit ellipse在standalone与Facet中复用`point_occupied`的`#D07850`、alpha 0.58、0.85 pt hollow ring；fit-coloured center从50降至2.25 pt²，ellipse geometry、center coordinate和annotation不变。DPR1/2、`2x2`/`8x8`与four-cell Facet视觉矩阵确认center仍清楚且不遮峰；persisted style/Facet assertions与DPR1/2 full-draw compose分别守住style truth和pixel path。
- 当前证据：合并focused回归`482 passed`，camera effective-unit定向`64 passed`；Axis parity matrix `67 passed`；Image fit style/golden/Facet focused `40 passed`及最终style+compose `4 passed`；最终全树`1538 passed, 4 skipped, 6 warnings`，warnings仍仅既有vendor SWIG deprecation。
- Gate 17：changed tests净增3个functions（Axis unit/integration 2、Image fit standalone/Facet style 1），其余为existing tests强化；全部KEEP。旧camera refusal test已合并进counts/electrons/fallback三模式live-monitor纵向证明，重复test为0。Axis cache consumer、camera effective-unit owner、Workbench unavailable-bool投影与Calibration/Occupancy unit consumer graph均已闭合；active旧行为文字与baseline pending checkpoint在本次docs cut删除。
- 最终post-doc冻结树为26 files `+428/-125`（净增303）：production 11 files `+110/-42`（净增68）、tests 10 files `+237/-67`（净增170）、active docs 5 files `+81/-16`（净增65）。无新增file/class/lane；最终test functions净增2（Axis locator与Image fit style），重复Axis private call-count test及旧camera refusal test均已删除/合并。全树`1538 passed, 4 skipped, 6 warnings`，warnings仍仅既有vendor SWIG deprecation。

## 10. Milestone 5完成证据

- `RepeatRegion`只编译timeline内部loop；唯一应用入口为原子`load(program, source, rows)`，唯一outer执行入口为`fire(cycles=int|None)`。旧`write_slots`、`write_scan_table`、`fire(forever)`、CompiledProgram重复scan/forever字段及compat为0。
- Load在任何program write前核target ABI、clock、geometry、uint32 count与TTL/DAC delay FIFO capacity。Camera每个实际program/table/cycles在LOAD和camera arm前核window count、相邻及wrap cadence；Virtual按compiled wall cadence逐cycle，camera busy保留physical ordinal gap，Temperature 20 ms exposure有15.1 ms recapture gap。
- PulseGUI正式Qt Fire只冻结request并提交device worker；Preview、ordinary device command、SAFE各有独立existing worker owner。Stop立即显示Stopping，SAFE可取消active transport并在普通command结束后再次确认，迟到Fire token不能覆盖Stopped；window等待三owner真实退休。
- Remote使用短owner epoch锁与唯一command lane：takeover先撤旧owner/drop socket，以SAFE取消旧transport，等lane退出后在lane内二次稳定SAFE才授予新owner；SAFE失败保持无owner/fault。正常client无idle timeout，explicit UART只探一个给定port；strict JSON拒duplicate/NaN/unknown/type，UART/AXI bounds与close lifecycle fail closed。
- RTL physical active覆盖running+draining，public DONE等待TTL/DAC FIFO、final latch和SAFE；point0 resident、underflow/overflow/protocol error sticky/loud；50 MHz XDC clock存在。显式`streamer_config.board.lanes`生成host target并把XDC/top当unordered checked projection。Build默认build-only，delete需要contained marked child，program/flash exactly-one target/device/AXI/cfgmem。
- RTL四条existing bench均用`$fatal`，Vivado Simulator 2019.1实跑通过finite tail、sticky overflow、UART watchdog/bounds及top SAFE/status；Python Icarus runner在本机因无`iverilog/vvp`明确4 skip。未program/flash真板、未连接camera/SLM，active FPGA README记录独立hardware acceptance runbook与receipt字段。
- Frozen-tree residual sweep：75 files；production/config `+2428/-1212`（净+1216），RTL/build `+493/-236`（净+257），tests `+1527/-735`（净+792），docs/notebook净删175。正增长KEEP：Remote ownership/strict protocol、Pulse atomic application/capacity/SAFE cancellation、manifest、camera cadence、Qt command/safety owners及其纵向证据；无新增production file/class。DELETE/MERGE已闭合旧执行API、全COM枚举、idle-timeout文案、ResolvedPulse dead metadata、unreachable Remote takeover test、Camera `_capture_spec`、BoardState dead forever、flaky wall-clock gesture gate与三个不完整Qt teardown。
- 验证均先打印current-checkout路径：Pulse `133 passed, 4 skipped`；Atom `311 passed`；Workbench Pulse/Console `134 passed`；最终全树`1514 passed, 4 skipped, 6 warnings in 416.74s`，warnings仅vendor SWIG deprecation。AST/JSON/notebook、active旧名、CRLF batch、launcher help与`git diff --check`均green。

Milestone 5 commit：`Close pulse, camera, remote, FPGA stop, timing, and ownership semantics`。下一步M6，不把SLM物理truth塞进Workbench或simulation旁路。

## 11. Milestone 6完成证据

- 正式X15213 transport只剩USB；DVI production/discovery/UI/tests/docs为0。真实adapter初始command为unknown；只有write、display slot、exact frame-memory readback和profile settle全部完成才是known-new。失败能区分known-old/unknown；command与mapping revision、profile/model/serial/wavelength/orientation/correction/settle和outcome进入immutable receipt。Strict profile记录phase-curve与settle provenance，同波长correction才接受。
- Target strict v2保存`intensity + objective`；bare phase artifact与objective-less v1保活路径删除。Science Context保存Pattern/base、numeric pupil amplitude/support、operator wavefront、typed system correction（kind/reference/wavelength/pupil/coordinate/valid-region/method）和command receipt；所有arrays immutable且strict roundtrip。
- Editor在unknown硬件上只建立deterministic authoring draft，不虚构device zero；Load Context是显式takeover但仍显示unknown/unsent，Send后才known。外部Task或mapping revision变化使draft可见diverge并拒绝旧Send，Adopt/Load snapshot在session claim内；correction load/enable复用同一DeviceUse seam。Target/Context import/load/save在existing worker执行，Qt只冻结input/交付结果；context在任何widget mutation前核shape/operator/pupil/receipt，close有2秒deadline且completion-driven retry。Atom不再反向import Workbench claim类型。
- Sparse WGS-Kim、iteration-12 fixed far phase、selected DFT与caller-owned optimizer state保留；initial/hot均`iterations=None`走canonical numerical gate。Dense Gaussian有finite signal/noise region与vortex-free initial phase，Flat Top用quality-preservingFOM stagnation stop；1024×1272实测Gaussian 26.37s/300轮/3.10% RMS，Flat Top 13.97s/210轮/0.488% RMS。不得宣称dense CPU明显改善，不引GPU。
- Feedback只接受含frozen spots Target的Science Context v2与注册到该Context的Calibration，不另选Target artifact；只更新Pattern并保留pupil/wavefront/system correction。每个candidate复用canonical Camera Measurement sealed Dataset和相同readout projection；missing/saturation同phase重试两次，统计上censored site则按最多3个batches/3次bounded boost处理，不伪造finite value。Controller以simultaneous uncertainty、step clip、confidence trust rollback和gain reduction更新。Final validation按sites×maximum looks做family correction，默认最多1000 shots/60s，terminal明确`accepted`或`inconclusive`并保存CI。Stop应用confidence-best并写formal Context；没有valid best时以incoming candidate 0落盘，异常只在incoming known时restore。
- SimulationWorld保持单一class/state owner；一个frozen`SimulationWorldConfig`拥有全部physics inputs，public projections read-only。唯一构造入口接config；optional strict workspace `world_profile`在任何device factory前解析，duplicate/NaN/unknown/type拒绝；tests用config replace而非运行时mutation。
- Frozen-tree residual：functional candidate 21 files `+3396/-1958`（净+1438）；production净+1084、tests净+329、active product docs净+22、profile净+3。加入三份Checkpoint状态文档后最终24 files净+1453。无新增production file/class，test functions净删8。正增长KEEP为strict Context codec+dense metrics、Feedback estimator/controller、Editor async ownership、immutable Simulation；DELETE/MERGE已闭合DVI、bare phase codec、fixed12/8、point-only/repeated validation、ScanLiveSlot/private average、O(K²) history、公开mutable physics、错误outcome tests及重复Context tests。
- 验证均打印current-checkout路径：M6合并纵向`122 passed`；Atom整包`300 passed`；Gate回修定向`104 passed`；最终全树`1504 passed, 4 skipped, 6 warnings in 424.33s`，warnings仍仅vendor SWIG deprecation，4 skips仍是无Icarus的M5 RTL runner。AST/strict JSON/old-name/import-boundary/diff-check均green。
- 明确未验收：本地无官方Hamamatsu header/DLL，不绑定猜测ctypes ABI；真实USB controller、profile曲线来源、correction编码/方向、active slot与optical settle必须按Atom README实验机runbook记录。Root wheel可能遗漏profile package-data，归M7 single-distribution/fresh-install gate，不称已安装可用。

Milestone 6 commit：`Converge USB SLM context, feedback, and simulation truth`。本轮按用户要求停止；M7等待后续指令。

## 12. Checkpoint更新规则

每个milestone开始/完成、长测试前或新用户裁决后立即更新：

- status；
- current HEAD/dirty；
- exact decision；
- focused evidence与路径provenance；
- commit subject（commit hash通过`git log --grep`解析，文档不自引用自己的hash）；
- next unfinished action。

不得恢复append-only历史日志，不把对话摘要当状态，不把passing test数量冒充未覆盖行为完成。
