# ZLC v2 Architecture Convergence — Implementation Goal

状态：ACTIVE；Milestone 1–4及residual closure完成。当前按用户要求停止在M4 commit，下一步讨论已记录的Plot/Fit profile；Milestone 5–7尚未开始。准确状态看`IMPLEMENTATION_PLAN.md`。

## 1. 唯一目标

在当前`Zou_lab_control_v2`工作树上完成一次root-cause架构收敛：

- 先删除审计确认的dead/parallel/history实现；
- 保留已经正确的八层骨架与science/numeric核心；
- 让Data、Runtime live、Plot/Fit/Overlay、Logic Node preview、Pulse/Camera/FPGA、SLM Feedback各自只有一个truth owner；
- 修复全部明确的数据损坏、hardware lifecycle、Stop、thread ownership与artifact问题；
- 按用户批准的产品行为重写tests和docs；
- 最终交付一个可安装、dependency-locked、可验证的单一ZLC产品。

不得把本Goal缩减成局部补丁或只修用户最近看到的症状。不得在未完成全部milestones时宣称Goal完成。

## 2. Authority与开工规则

执行时authority顺序：

1. 用户在执行期间的新明确指令；
2. 本Goal；
3. `AUDIT/USER-DECISIONS-2026-08-17.md`（若仍存在）；
4. 当前代码与实验事实；
5. 旧`ARCHITECTURE_DESIGN.md`、`IMPLEMENTATION_PLAN.md`、package docs/tests仅作历史证据，不可覆盖本Goal。

开工必须：

- 先完整阅读本Goal；
- 打印并记录repo root、HEAD、branch、dirty state和关键module实际路径；
- 保留用户已有无关修改，不reset、不checkout覆盖；
- 重建当前文件/consumer inventory，不信旧行号；
- 若`AUDIT/`被删，直接从当前代码和本Goal恢复，不要求用户重新提供；
- 把旧`IMPLEMENTATION_PLAN.md`重写为简洁的当前milestone/status/evidence表，不继续在顶部追加历史Checkpoint；
- `AUDIT/`是实施证据与逐文件删除/修复清册，不是可丢弃的临时上下文：每个milestone先读对应报告并把finding映射到新Checkpoint；只有全部finding关闭并进入最终文档后才可archive/delete。若用户已删除则从代码重建必要证据；
- 建立实施plan后持续执行，不在普通实现选择上停下来问用户。

只有以下情况允许停下：需要真实硬件program/flash、不可恢复地操作实验数据、或当前事实与本Goal存在无法合理消解的物理矛盾。普通重构、删除、测试迁移和实现细节均自主裁决。

## 3. 用户已批准的产品行为

### 3.1 Deployment

- 一个可安装ZLC distribution；
- 内部保留八层依赖边界；
- 不再维护八个standalone wheels；
- 一份product manifest、一份dependency lock、一组entrypoints；
- 删除旧source-bootstrap/standalone metadata/duplicate launchers，不保兼容入口。

### 3.2 Runtime live与完整数据访问

- Logic Node只提交新增chunk/event，不在plugin维护另一份history；
- Runtime是唯一run dataset累计owner；
- finite exact的event chunk只供commit/exact Processor；所有UI/display consumer从第一次publication起使用该publication的canonical full geometry，future cells invalid；Monitor才显示latest event；
- Panel可在任意时刻访问截至当前的全部数据并应用任意axis的scope/reduction/point/fate；semantic不得决定event/current数据来源；
- repeat与point rows是统一incremental placement轴；多维scan/grid通过point table/grid topology逐点填充，cell payload保持原子完整；
- Live Panel、Edit/Refresh/Save、selector、fit和overlay必须共享同一accepted canonical presentation snapshot；
- SITE保持cell data axis且每个occupancy event整组原子传递；overlay不累计第二份site状态。Occupancy发布通用numeric/bool status signal并用Dataset validity表达可读性；`zlc_plot`拥有geometry/renderer contract，Workbench只通用路由。跨多个cells的离散status reduction没有明确产品定义时必须UNKNOWN/隐藏；
- Camera、Scan、Calibration只用同一commit/materialization/seal机制；
- full raw、live、Stop partial和Final来自同一truth；
- 未采数据invalid；UI freeze不改变结果；
- Camera使用chunked append，避免producer每次复制全部历史；canonical display按实际Panel cadence后台materialize/cache，不阻塞Qt owner；Scan按固定geometry增长；
- exact Processor逐event处理，display derivation可latest。

### 3.3 Data/Fit/Overlay

- `display_interval`定义Panel pair admission；fit armed时在1秒显示延迟预算内按顺序fit每个已admit Data revision。区间内更高频的raw Monitor publication由Panel明确latest-sample，不冒充已经显示/fit的revision；
- Panel只原子显示`data@N + fit@N`；不得出现data无fit或fit与data不对应；
- Fit数据范围严格按committed Area ROI（或显式X-range）→viewport→full range；FacetGrid重放selector必须携带focused cell identity，不得静默降级；
- FitResult保留source parent/revision，正常负载下Rolling trace逐revision连续；
- 最老pending等待超过1秒或队列超过64 MiB时明确报一次resync错误，丢弃尚未fit的中间revision并从latest继续；断点不得冒充成功fit，Panel/Qt/Stop/close不得被永久锁住，raw data仍完整保存并可离线重算；
- worker可后台计算，但Qt呈现必须atomic且不阻塞owner thread；
- Overlay与Data/Fit共享同一scope/axis/fate projection；无法唯一对齐即拒绝；
- Selector Off时plot完全不接管area、zoom/pan、滚轮或双击focus，普通滚轮滚外层board；Selector On时FacetGrid overview只允许focus cell，不允许开始area selector。

### 3.4 Logic Node contract与Task UI

- live/progress/preview/terminal成为Descriptor/NodeHost强制contract，不依赖实例作者记忆；
- Workbench不得import`zlc_atom.nodes.<concrete_leaf>`或按具体node/output名称分支；新增/删除普通Logic Node只改该leaf目录、资源与测试。跨节点UI能力必须由Data/Runtime/Plot等中立层contract声明后通用路由；
- Measurement bounded live；Task有progress与声明preview；first data前不假报live；terminal正确seal/retire；
- Task运行中禁止Add Logic Node；冻结当前source/preview signal、overlay binding以及scope/reduction/fate等数据identity；
- 允许其它Panel和当前Panel的纯显示参数；
- Calibration继续Grid/Figure显示long/readout/long三帧。

SLM Feedback必须复用canonical Camera Measurement `repeat=N`及其同一Runtime dataset：

- 中间数据由Camera Measurement统一live发布；
- Preview与Feedback estimator使用同一projection request；
- 不允许Feedback另写camera average/accumulator。

### 3.5 Pulse/Camera/Remote

- same-shot采用continuous best-effort；不增加hardware marker或逐cycle arm/fire；
- 仍核compiled windows、frames-per-cycle和received ordinal，错误立即失败；
- Temperature保留约20ms exposure，增加足够trigger/recapture gap；
- Pulse Stop任意时刻立即响应，Qt不得阻塞；高优先级Stop/SAFE可取消普通wait/worker/transport；
- UI立即进入Stopping，hardware ack后台完成；timeout显示错误但不冻结UI；未确认不能显示Safe；
- Remote不加密码、认证、TLS或权限UI；
- second client默认last-client-wins，旧client立即失效；takeover前旧active command必须成功Stop/SAFE；
- 正常连接没有idle timeout；控制进程/socket/连接真正断开时自动SAFE；后台检测不需用户操作；
- 正式UART port显式配置，不向所有COM probe；
- 默认只有server process持hardware transport，不保留假的进程内Interprocess lease。

### 3.6 SLM

- 实验机正式transport是USB；删除DVI production/discovery/UI/tests/docs路径，不留Experimental或兼容入口；
- real command state支持known/unknown，未send/read不能虚构zero；
- correction mutation取得同一DeviceUse claim，并冻结进command receipt；
- 当前Feedback优化all-shot fluorescence，不称trap depth；
- 100 shots为coarse，controller使用最佳可实现的uncertainty/trust/rollback算法；
- Stop表示接受并apply本run最好phase、写正式artifact；用户可主动load旧phase；异常失败只在incoming known时恢复；
- Feedback只更新Pattern/base并compose明确Science Context；
- Target、pupil、operator wavefront、vendor correction、device mapping各有唯一owner。

为未来校准建立最小、明确的`SystemCorrectionArtifact` seam，区分：

- vendor/device correction；
- 可跨geometry复用的optical/system correction（pupil phase map或target response map，必须明确类型）；
- 当前geometry的weights/Pattern。

当前SlmFeedbackTask不得把site brightness weights冒充通用wavefront correction。未来独立SLM Calibration流程可以生成`my_correction`，新Target统一compose使用。

### 3.7 Calibration、Scan、Simulation

- 不重设计Calibration对外骨架、主要artifact、默认raw policy或三帧report；
- 允许不改变外部行为的依赖解耦、明确corruption修复和内存优化；
- Scan正常完成/Stop/失败默认restore pre-run device values；
- SimulationWorld保持一个类和一个state owner，不拆层；
- 参数在init前通过单一API/immutable config确定；
- 可用workspace-local simulation JSON人工编辑/只读显示，Device Manager Init不运行时改写；
- tests用config override，不改public mutable world attributes；hidden truth不泄漏给production算法。

## 4. 实施纪律

- Root cause优先，不做症状补丁；
- DRY、解耦、唯一真相源；
- 先读完整链路，再改；性能改动前后必须profiling并交叉验证；
- 不因旧tests、迁移风险或兼容性保留历史实现；无deprecated alias、compatibility shim、双写或fallback旧路径；
- 不新增密码、安全框架、TLS、权限系统或新的SHA256/content-hash体系；
- 不加推测性“防御框架”；但domain validation、hardware ack、owner identity、format strictness属于核心正确性，必须实现；
- 不使用GPU、不降采样、不降低质量、不放宽科学验收来换速度；
- 不增加manager/registry/base-class等新抽象，除非替换多个现有重复owner且显著减少总复杂度；优先扩展现有Data/Plane/Host/Session骨架；
- 不在Qt owner thread做blocking I/O、fit、device tune或等待Future；
- 不运行真实FPGA program/flash、SLM/camera实验或其它hardware side effect，除非用户另行明确授权；
- 不删除workspace/data或用户实验artifact。

## 5. Commit与工作流

- 先写能杀死旧错误的行为测试，再实现；明确obsolete tests直接删除，不迁就旧结构；
- 每个milestone完成代码、tests、docs和证据后一次commit；不为小修改碎片commit；
- milestone内允许较大diff，但结束时只有一条production路径、worktree清楚、focused tests通过；
- 每个milestone提交前必须做start→candidate residual sweep：重建production consumer graph，删除旧API/alias/重复owner，按production/tests/docs解释净LOC，逐项删除或合并duplicate、white-box、self-test与obsolete-behaviour tests，并把所有有意延期项写入Checkpoint；测试通过不能代替这道删除gate；
- 不用一个巨大最终commit掩盖阶段边界；建议8个milestone commits（含authority/checkpoint阶段）；
- 每个commit message说明删除了哪个旧truth、建立了哪个唯一owner、运行了哪些证据；
- 不自动push或创建PR，除非用户以后明确要求。

## 6. Milestone 0 — 重建唯一Authority与持久Checkpoint

在任何production删除/修改前，完整替换而非追加：

- 重写根`ARCHITECTURE_DESIGN.md`：只保存用户已批准的目标不变量，并对尚未实现项显式标`TARGET / NOT YET IMPLEMENTED`；不得把目标写成当前代码已完成；
- 重写根`IMPLEMENTATION_PLAN.md`：只保存本Goal的milestones、每项状态、当前HEAD/dirty、已运行证据和下一步；
- 初始化持久Checkpoint为：审计与用户裁决完成、production code尚未修改、Milestone 1待开始；
- `HANDOFF.md`只指向新的Architecture、Plan和本Goal；
- package GOAL/README在未处理前明确只是stale evidence，不得覆盖新authority；
- 建audit finding→milestone→status的精简traceability表，可放在新Plan内，不另造巨大第三份权威。

以后每个milestone开始/完成、长测试前、发生新裁决后立即更新Checkpoint；只改当前状态表，不恢复append-only历史日志。

建议commit：`Establish approved architecture and implementation checkpoint`

## 7. Milestone 1 — Pure deletion与历史清扫

先删除无需替代、production consumer为零的内容。执行前用当前tree重新确认consumer，但“旧tests/README提到”不是consumer。

候选至少包括：

- Runtime旧exact/reservation/builder/monitor/live-port/preview框架、dead failure/cleanup/RunHandle支线及self-tests；
- Plot无consumer `LivePlotController/_live_channel`、FitNumericTable、test-only facade conveniences及self-tests；
- Data `numeric.py`、AxisSourceRef/ResolvedPointRows、ValuePayloadContract等零consumer clusters；
- Pulse test-only engine model移入tests并只留实际oracle，假的Interprocess lease、dead transport methods/public exports；
- Atom zero-consumer dynamic resolver/fakes/oracle surfaces；
- UI/Workbench dead Graph/FormGrid/gallery-only production surface；
- duplicate root/package launchers；
- duplicate fit-contract docs/SHA tests、duplicate surveys、历史acceptance/reacceptance/goal archives；
- every-export notebook guards、arbitrary API-size caps、fake-only self-tests和wrong-checkout tests。

同步收窄`__all__`、imports、pyproject exposure、examples/notebooks和docs，不留空package/alias。

Milestone验收：

- 所有保留production import仍可解析；
- 真实consumer smoke通过；
- repo搜索不再命中被删除框架入口；
- 没有新增替代抽象。

建议commit：`Prune dead frameworks and historical product surfaces`

## 8. Milestone 2 — Data、Durability与Installation truth

修复：

- Figure archive预规划唯一member namespace；Reader严格format/version/shape/duplicates/nonfinite；
- validity入口只接受真实bool contract；
- snapshot restriction同步投影coordinates/labels/frame/unit/validity；
- selection按AxisId/typed coordinate唯一解析，支持text，重名明确拒绝；
- `unique_path`与commit完成并发原子分配，禁止多process覆盖；
- duplicate `DeviceSpec.key`在任何factory side effect前拒绝；
- JSON codec unknown/duplicate/nonfinite/coercion全部严格；
- 删除多份figure schema/selection codec；
- 在不改Calibration外部骨架的前提下修明确数据corruption与Atom→Workbench保存依赖。

必须补并发、损坏archive、labeled-axis、duplicate-device old-red。

建议commit：`Make dataset, archive, path, and installation truth strict`

## 9. Milestone 3 — 一个Runtime live与Logic Node contract

以现有`SignalDataPlane`、`NodeHost`、`OwnedSnapshot`为骨架，建立唯一append/commit/seal路径：

- Node提交新增chunk/event与同revision sibling bundle；
- Runtime按run/signal累计canonical data，使用chunked/internal storage避免历史全复制；
- immutable live view、partial seal、final seal来自同一owner；
- generation内schema/content identity稳定，revision严格递增；
- future invalid、coverage准确；
- exact/latest Processor policy显式声明；不同Processor可并发、同Processor串行；
- Plane freeze只读已提交结果，不调用plugin；
- primary/companion变化都调度，generation replacement进入same-shot grouping；
- generic Logic Node contract强制live/progress/preview/terminal/artifact；
- Task运行中冻结Add Logic、source/preview/overlay/data identity，允许纯Panel显示操作。

迁移Camera/Scan/Calibration/SLM Feedback后，在同一milestone删除所有plugin-specific slot/listener/dirty/pull和旧builder路径。不留双写过渡。

SLM Feedback camera acquisition改为复用canonical Camera Measurement `repeat=N`、live dataset和projection；删除Feedback私有average truth。

Profiling：

- Camera repeat=100的累计copy bytes、peak memory、publication time按shots近线性；
- UI freeze不执行science/materialization；
- Stop在是否有Panel/Processor订阅时产生相同partial/final truth。

建议commit：`Unify runtime publication, accumulated data, and node preview contracts`

## 10. Milestone 4 — Exact Data/Fit/Overlay与Qt lifecycle

实现用户批准的exact paired pipeline：

- fit armed后，每个由authored `display_interval` admit的source revision在1秒支持预算内进入有序exact queue；
- 每revision有一个FitResult，带source parent/revision；
- Panel只present matching data+fit；正常负载下Rolling trace无revision gap；
- 最老pending等待超过1秒或队列超过64 MiB时loud报告一次resync，丢弃未fit中间revision并只保留latest继续；不得永久latch，raw data不丢；
- Fit计算后台执行，Qt不阻塞；
- 一次PanelState transaction幂等，no-op=0 solve/0 render；title/layout不re-fit；
- 删除重复configure/clear/replay和多front handoff；
- 正式96×128 Camera、小Area ROI、主图fit、并行ROI image和一个fit-parameter Rolling链路以100 ms为profile警戒线；删除明显额外cadence/HOL/错误串行/重复render，但不为百分之几或十几的边际收益增加不相称的调度复杂度；
- Overlay作为typed companion，Data/Fit/Overlay共同解析scope/axis/fate；ROI/binning使用同一坐标truth；
- Selector Off时plot完全不接管area、zoom/pan、滚轮或双击focus，普通滚轮滚board；
- Selector On时FacetGrid overview只允许双击进入cell，不允许开始area selector；area只在focused cell或非grid surface工作；
- 修Form reconcile dependency graph、唯一PanelState parser、唯一owner wake；
- 所有Qt window、worker、executor、claim有界close；Pulse window不得在safe/release前消失。

Profiling必须覆盖：1/4/8 panels、fit+overlay、真实Qt staged board、Setting/Edit变更、rolling fit parameter完整性、owner-thread latency。不得只跑isolated PlotSession microbench。

建议commit：`Make data, fit, overlay, and Qt lifecycle exact and atomic`

## 11. Milestone 5 — Pulse、Camera、Remote与FPGA边界

Host/Python：

- RepeatRegion、cycles/shots、sweeps、dataset repeats分开；一个finite cycle执行入口；
- load前核target ABI、clock、geometry、counts、actual windows/cadence、delay FIFO capacity；
- count严格int/range，不silent clamp/wrap；
- Camera continuous best-effort分组核window count/ordinal，错误loud fail；
- Temperature 20ms exposure配足够gap并用相容Calibration；
- Virtual按真实cadence逐cycle、支持Stop、模拟camera busy；
- Pulse Stop Qt入口快速返回，优先取消/SAFE，hardware ack后台且有bounded timeout；
- Remote无密码；last-client-wins；旧handler失效；takeover必须先Stop/SAFE；
- 正常连接无idle timeout，真实disconnect/crash由后台检测后自动SAFE；
- explicit UART port；single server process hardware owner；
- strict remote/sequence codec、AXI address、UART frame/lifecycle。

RTL/build：

- hardware SAFE独立gate physical TTL/DAC data/clock；LOAD/FIRE前pin保持safe；
- public DONE等待delay FIFOs与final DAC latch完成并进入安全态；
- underflow/overflow/protocol error sticky且loud；point0 resident；
- compiler与RTL capacity model一致；
- 50MHz engine有正确clock/STA constraints；
- explicit board manifest统一生成host lanes、top mapping、XDC；不靠XDC行序；
- destructive build root做真实path containment；
- program/flash exactly-one target fail closed，默认不自动flash；
- UART truncated frame watchdog和bounds；
- 建自动RTL runner，testbench用`$fatal`/非零失败；generated host image与RTL逐tick对拍。

不得在本Goal中program/flash真板；产生实验机runbook与待验收receipt schema。

建议commit：`Close pulse, camera, remote, FPGA stop, timing, and ownership semantics`

## 12. Milestone 6 — USB-only SLM、Science Context与Robust Feedback

Device：

- 删除全部DVI schema/discovery/presenter/mode/transport/tests/docs；
- USB SDK ABI按官方header收口，command outcome区分known-old/known-new/unknown；
- initial unknown，成功write/display/readback/settle后才known；
- correction load/enable取得DeviceUse claim并冻结mapping revision；
- profile补model/serial/wavelength/curve/provenance/settle语义；不新增hash；
- Editor显示draft/device command divergence，external Task command后旧Send不得静默覆盖。

Artifacts/context：

- Target保存intensity+objective；
- Science Context保存numeric pupil、Pattern/base、operator wavefront与system correction引用；
- command receipt保存USB/profile/wavelength/orientation/correction/outcome；
- 建最小`SystemCorrectionArtifact`类型区分pupil phase map与target response map；不实现未经测量支持的反演算法。

Solver/Feedback：

- 保留sparse WGS-Kim、fixed far-field phase、selected DFT和caller-owned state；
- initial/hot inner solve走到canonical numerical gate，不为省几十毫秒增加physical candidate；
- current Feedback observable明确all-shot fluorescence；
- 复用Camera Measurement统一projection；
- estimator使用uncertainty，controller有step clip/trust/rollback/invalid stop；
- 100 shots coarse；final validation根据实测variance自适应并有最大时间/shots，输出estimate+uncertainty/inconclusive；
- Stop选择confidence-best、apply并写正式artifact；异常failure按known incoming恢复；
- sparse-only contract明确；
- dense Gaussian/Flat Top先修signal/noise region、FOM、initial phase和early stop，再profile CPU；不引GPU。

SimulationWorld保持单类owner；只收口immutable init config/local profile和test diagnostics，不拆physics state层。

不得运行真SLM；输出USB实验机验收runbook：serial/SDK、profile来源、correction、orientation、gray readback、optical settle、system correction。

建议commit：`Make SLM USB command context and fluorescence feedback reproducible`

## 13. Milestone 7 — 单一产品、测试证据与文档替换

Packaging：

- 根distribution真正包含产品bootstrap与八层packages/resources；
- 删除nested standalone distribution truth；
- 一份dependency lock/constraints；
- 一份product manifest生成layers、entrypoints、testpaths与environment check；
- wheel/fresh install验证templates/profiles/FPGA assets；
- root正式launchers唯一，argv完整保真。

Evidence lanes：

- `software` fresh install；
- `gui_offscreen`；
- `virtual_vertical`；
- `notebook_offline` fresh kernel；
- `real_screen` runbook；
- `hardware` runbooks（不在普通CI执行）。

删除wrong-checkout、doc-SHA、prose/API-cap、fake-self tests。保留并加强numeric/property、real Host/Plane、public-button、synthetic independent oracle。

Docs：

- 完整重写根`ARCHITECTURE_DESIGN.md`为最终实际不变量；
- `IMPLEMENTATION_PLAN.md`只保留current status、完成milestone和当前验证，不保留多代相反Checkpoint；
- package README/docs只描述当前public product；
- 删除/移除所有过时survey/acceptance/goal archive，不在活文档尾部追加补丁；
- 教程只保留正式产品路径并fresh execute；
- 明确software/offscreen/virtual与真实hardware证据边界。

最终运行：

- 所有focused old-red；
- 八包package suites；
- fresh installed full tree；
- static dependency/import/asset checks；
- product profiling；
- 不运行真硬件，列出全部pending experiment-machine acceptance。

建议commit：`Ship one ZLC product and replace stale tests and documentation`

## 14. Definition of Done

只有同时满足以下条件才能标Goal complete：

1. 当前tree不存在审计确认的dead parallel frameworks/pipelines/launchers/docs；
2. 每个核心事实只有一个owner：dataset accumulation、live/final seal、PanelState decode、owner wake、figure codec、pulse execution、SLM context；
3. Node只提交新数据，但Panel随时可从Runtime访问完整截至当前数据；
4. Data/Fit在1秒预算内逐revision exact paired；超预算loud resync到latest且不断Qt/原始数据，Rolling断点不冒充成功fit；Overlay投影与Data一致；
5. 新Logic Node由框架强制live/preview/terminal contract；
6. Pulse Stop UI不阻塞，disconnect自动SAFE，second client takeover正确；
7. FPGA source具有clock/SAFE/DONE/error/board/build闭合contract和自动RTL evidence；
8. SLM只保留USB正式路径，known/unknown/context/receipt正确，Feedback复用Camera Measurement并稳健处理100-shot噪声；
9. Calibration外部骨架未被重设计，Scan默认restore，SimulationWorld未拆层；
10. 一个installable locked product在fresh environment通过全部software gates；
11. 旧docs/tests已删除或重写，没有compatibility/history残余；
12. 每个milestone已有阶段性commit、测试/profiling evidence和清晰交接；
13. 工作树最终只剩用户事先存在的无关修改；
14. 最终报告明确列出未执行的real-screen/FPGA/camera/SLM experiment acceptance，不冒充通过。

若仍有一个required milestone或P0未完成，不得因为时间、token、测试数量或“其余以后再说”而标Goal完成。
