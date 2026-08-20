# ZLC v2 Architecture Convergence — Implementation Plan

本文只保存当前实施状态、阶段证据和下一步，不保存旧Goal历史。目标架构见`ARCHITECTURE_DESIGN.md`，完整范围见`ZLC_V2_IMPLEMENTATION_GOAL.md`。

## 1. Persistent Checkpoint

更新时间：2026-08-20
启动HEAD：`af54e24787de67270c54eb154f2b23f43508fc3e`
Branch：`master`
用户执行边界：Milestone 1–4及各自post-milestone residual sweep均已完成；Milestone 4由单一commit `Make data, fit, overlay, and Qt lifecycle exact and atomic`落盘后立即停止。下一步只与用户讨论已记录的Plot/Fit profile，不进入M5。

### 当前状态

- 审计：完成；逐文件证据在`AUDIT/`。
- 用户裁决：完成；记录在`AUDIT/USER-DECISIONS-2026-08-17.md`与根Goal。
- Production代码：Milestone 1–4及其residual closure均已完成。
- Hardware：未访问；本Goal不授权program/flash或实验机device操作。
- Milestone 0：`COMPLETE` — commit `e854ddf`（`Establish approved architecture and implementation checkpoint`）；唯一Architecture、Plan、Handoff、Goal与Audit证据已纳入版本控制，未改production。
- Milestone 1：`COMPLETE / SWEEP COMPLETE` — test-owned Pulse engine、dead Qt wake、compat aliases、false-green API/notebook/dependency tests与重复contract docs已删除。
- Milestone 2：`COMPLETE / SWEEP COMPLETE` — nested immutable truth、strict embedded manifest grammar、archive-first Panel Save与唯一Figure入口已闭合。
- Milestone 3：`COMPLETE / SWEEP COMPLETE` — replay lineage、selection lock域、canonical presentation/overlay alignment、Refresh/layout lifecycle及测试残余已闭合。
- Milestone 4：`COMPLETE / SWEEP COMPLETE` — exact pair/resync、原子PanelState、canonical selector/threshold、streaming archive以及Qt worker/close均已闭合；全树1498 tests通过。
- Milestone 5–7：`PENDING`；本次不得开始。

### Milestone 3完成边界

Milestone 3主体从clean HEAD `23e820d`开始并由`ca66c7d Unify Runtime live data and task previews`落盘。验收follow-up从clean HEAD `ca66c7d`开始，由`Fix canonical full signal presentation`落盘；未访问hardware，未进入Milestone 4。

### 当前停止门

Milestone 4、cleanup与performance follow-up均已完成；下一步进入M5。100 ms仍是profile警戒线，不是硬验收门。

## 2. Milestone状态

| Milestone | Scope | Status | Evidence / Commit |
|---|---|---|---|
| 0 | Approved Architecture、current Plan/Checkpoint、Handoff、Audit evidence | COMPLETE | `e854ddf`；docs/link/diff check，未跑测试 |
| 1 | Dead framework、parallel pipeline、test-only surface、duplicate launcher/docs/tests删除 | COMPLETE / SWEEP COMPLETE | 原commit + residual fix；见§8 |
| 2 | Data/Durable/Installation truth | COMPLETE / SWEEP COMPLETE | 原commit + residual fix；见§8 |
| 3 | Canonical Runtime live与Logic Node contract | COMPLETE / SWEEP COMPLETE | 两个M3 commits + residual fix；见§8 |
| 4 | Exact Data/Fit/Overlay与Qt lifecycle | COMPLETE / SWEEP COMPLETE | `Make data, fit, overlay, and Qt lifecycle exact and atomic`；见§9 |
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

- M4：Plot-owned原子PanelState operation、首mount多front、exact data+fit queue、selector Off/FacetGrid overview、Viewer唯一PanelState parser、Qt/executor bounded close与whole-archive bytes materialization的性能/owner收口。
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

- 正式offscreen TaskConsole链：repeat=0 Camera 96×128、主图26×26 Area radial fit、并行ROI image、fit amplitude Rolling，真实100 ms timer。去掉两个额外cadence wait后，24个稳态revision总延迟P50/P95为149.97/166.83 ms；source→coherent prepare 46.99/62.91，fit 23.61/30.51，paired render/publish 54.31/58.25，fit publication→Rolling prepare 0.33/1.29，Rolling render→accept 18.88/21.11 ms。剩余是authored beat、必要fit与两个实际raster/Qt surface，不为边际收益增加新调度框架。
- First mount的1/4/8 panels从5/20/40 fronts降为2/8/16（initial+真实DPR）；title为0 front，display为1，fit arm为1 solve+1 front，fit-armed Edit为2。invalid target与final-render failure均保持旧state/front。
- 全仓current checkout八包一次完整运行：`1498 passed, 6 warnings in 427.82s`；warnings仅vendor SWIG deprecation。Runtime+Plot单组484 passed，UI+Workbench单组488 passed，Atom全套310 passed。
- 81个changed Python文件AST解析通过；八包import均解析到本checkout；59个active Markdown relative links全部存在；旧API/alias/test-only wake与pending probes搜索为0；无新增production file/class，删除3个历史文件；`git diff --check`无error，仅line-ending warnings。未访问hardware、real camera/SLM/FPGA或real-screen。

### Gate 17 consumer/test/docs sweep

- Candidate相对启动HEAD `af54e24`：98 files，`+7965/-3349`，净增4616行；其中production 46 files `+4070/-2199`（净增1871）、tests 38 files `+3657/-1035`（净增2622）、active docs 14 files `+238/-115`（净增123）。
- Production增量主要是Plot exact-pair/atomic transaction/coordinate selector（zlc_plot净增1046）与Workbench completion-driven Console/Viewer/Save/device/Pulse ownership（zlc_workbench净增736）；Runtime净增74用于publication catch-up与owed presentation，UI与Data净增0，Atom净增15。没有新增file/class或plugin-specific Workbench framework。
- Test分类：KEEP真实Camera Area→main fit→invalid gap→Rolling断线、active backlog即时resync、atomic rollback/no-op、restored N→latest catch-up、formal Qt Save/Viewer/close与Selector interaction；MERGE wait/bytes resync为参数化gate并共享blocked-fit fixture、合并重复semantic/rollback/Qt window setup；DELETE capacity-one/latest-drop旧政策、private helper/Port/container窥视、OwnerChannels/SelectionBridge self-tests、旧event-snapshot直喂Plot、package-local launcher保活与陈旧绝对随机阈值。全树test函数只从1246增至1270（净增24）；测试LOC增长来自真实跨层Qt/lifecycle/lineage纵向，而非数千个新测试。
- Consumer/owner sweep删除package-global fit pool、second prepare/fit executor、`apply_panel_fit`/`compose_panel_spec`、`PanelPlotAnnotations`、OwnerChannels wrapper、SelectionBridge test-only Event/introspection、arbiter/port pending probes及同步Figure/Pulse preview路径；active旧名均为0。

### 明确defer与停止门

- 用户随后授权的M4 cleanup已完成：删除单消费者`_WorkerSessionAdapter`并让Host直接调用现有PlotSession owner，production净删260行；九个高增长测试文件及一条gesture白盒共净删538行，保留exact backlog、ROI→Fit→invalid gap→Rolling、atomic rollback、Viewer/Pulse/Qt close与coordinate threshold核心证据。Plot全套383 passed，四个Plot/Viewer/Pulse文件149 passed，Console/Selection/TaskConsole三文件100 passed。
- 100 ms只保留为正式链profile警戒线；原150/167数字来自不同source harness，不能作为A/B基线。当前同harness结果与未达到的目标均记录在下方Performance follow-up。
- 不可中断的vendor discovery保持window可见并拒绝close，直到真实future结束；hardware transport cancellation/priority属于M5，不把`shutdown(wait=False)`冒充安全退出。
- 普通Pulse Stop/FIRE wire priority、Camera/Remote/FPGA归M5；SLM USB/context/feedback归M6；single distribution/fresh install/notebook与final docs归M7。M4未访问hardware，也不把offscreen/virtual证据冒充实验机验收。

Milestone 4 commit：`Make data, fit, overlay, and Qt lifecycle exact and atomic`。提交后立即停止，不进入M5，等待用户的Plot/Fit性能讨论。

M4 cleanup follow-up commit：`Remove residual M4 adapters and duplicate tests`。复盘根因不是缺少规则文字，而是错误地把各agent分片收口当成合并树收口，并用full green、无新增class、test函数净增数替代新增definition/state/consumer与测试正文LOC审计。`AGENTS.md`现强制所有cut合并并冻结candidate后重新独立审计；已知safe deletion不得在标记sweep-complete时延期。

### Performance follow-up

- Runtime新增中立`primary-index` indexed-derived Dataset：每个Measurement source index有value或invalid，普通Monitor仍latest；64 MiB/100k retention按display请求lazy materialize，10,000 publications的window=100为0.393 ms/900 bytes。所有Plot读取同一OwnedSnapshot；普通Plot默认latest，声明window的history projection读取同一axis；同publication扩大window或改fate立即原子rematerialize，Save冻结同一Dataset。
- Surface admission改为capacity-one same-shot group：任一member仍有重绘在途就不排第二张完整frame，只留Plane latest与admission debt；atomic publication wake在deadline已到时立即stage，Pause/closing不admit新Surface。TaskConsole删除第二个`--interval-ms`时钟真相。
- Tight colorbar保留原actual vmin/vmax norm、ticks与逐像素表现，但把其Axes纳入现有dynamic composition，避免整张Figure native redraw；renderer同条件cProfile P50/P95 80.5/89.4→63.6/69.0 ms，DPR1/2、Area/Fit与golden逐像素一致。
- 同一formal harness（独立66 ms virtual source、100 ms Board、96×128 Camera、26×26 Area fit、并行ROI、Rolling）对比`69d5514`：publication→main 338/435→131/163 ms，publication→Rolling 339/435→191/224 ms；full frames与Host/Port pending最大值3→1，primary index连续、无error。未达到80–105 ms估计，Rolling tail继续作为未来profile对象，不能冒充已完成收益。
- Candidate相对`69d5514`为36 files `+1563/-214`（净增1349）：production 20 files净增821、tests 7 files净增506、docs 9 files净增22；无新增production file/class/lane。完整current-checkout全树`1497 passed, 6 warnings in 412.35s`，warnings仍仅vendor SWIG deprecation；Data+Runtime+Plot 557、UI+Workbench 483均独立全绿。10k NoScanDeque gate证明commit摊还O(1)、window=W最多O(W) lookup；renderer逐像素parity属于KEEP，clock override/完整frame FIFO旧政策测试已删除或改写。

Performance commit：`Make derived history continuous and presentation capacity-one`。完成后进入M5，不把indexed Dataset、gap或window放进任何Plot-kind/Logic plugin/Workbench专用lane。

## 10. Checkpoint更新规则

每个milestone开始/完成、长测试前或新用户裁决后立即更新：

- status；
- current HEAD/dirty；
- exact decision；
- focused evidence与路径provenance；
- commit subject（commit hash通过`git log --grep`解析，文档不自引用自己的hash）；
- next unfinished action。

不得恢复append-only历史日志，不把对话摘要当状态，不把passing test数量冒充未覆盖行为完成。
