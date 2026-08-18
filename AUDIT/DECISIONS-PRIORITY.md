# 用户裁决包：8个顶层Gate

状态：用户已完成裁决；当前实施解释见[USER-DECISIONS-2026-08-17.md](USER-DECISIONS-2026-08-17.md)，本文仅保留技术索引。
用途：把`DECISIONS.md`的细项压成可操作顺序；旧ID继续保留作traceability，不要求逐条回答。

这是一份给实现者快速索引的压缩版。若不熟悉当前代码，请不要从本文件直接裁决，改读[DECISIONS-USER-GUIDE.md](DECISIONS-USER-GUIDE.md)：它解释每个名词、当前行为、错误原因、方案代价和真正需要回答的问题。

## 0. 不需要产品裁决、应直接修的明确错误

以下行为没有合理的“保留现状”产品选项；实施前仍需old-red，但不应把bug包装成用户偏好：

- figure archive成员覆盖、reader不验自己的format；
- validity把非bool静默转bool；snapshot crop丢coordinate labels；重名轴静默选错；
- duplicate `DeviceSpec.key`创建两台只保留/关闭后一台；
- `FluentParameterForm.reconcile()`沿用旧dependency graph；
- Viewer第二套残缺PanelState parser；
- standalone Pulse窗口先消失后safe、Qt atexit绕close guard；
- GUI线程直接hardware tune/同步等Future；worker/executor无owner shutdown；
- Pulse load不核ABI/clock/geometry、count silent coercion/wrap、delay FIFO无capacity validator；
- Pulse remote无认证且默认LAN bind、stale handler仍可command、SAFE失败仍转owner、所谓interprocess lease只在进程内；
- auto transport向所有COM发送probe，可能触碰非Pulse实验仪器；
- Pulse codec接受未知/duplicate/nonfinite/coercion、canonical mapping key碰撞、manifest猜latch clock、AXI负地址wrap与UART超contract reply；
- Camera future cells伪有效、UI freeze改变Stop terminal truth；
- Atom反向import Workbench与漏打包scan templates；
- real SLM未读/未发却把zero当last phase，correction绕过claim；
- tests读取旧checkout、fake只测fake、notebook保存error/stale output仍全绿。

这些进入Phase 1，不等待下面的架构gate。

## G1 — 产品如何部署、哪些public surface真实承诺

关联：D-011/012/019/020/063/064/068/070/076/078。

### 推荐

长期收敛为**一个可安装、可lock、可生成receipt的产品distribution**，代码仍保留八层依赖边界；不再承诺八个standalone wheels。短期若无法迁移，明确标source-checkout-only并删除半安装式root scripts/metadata，不两套并行。

同时：

- `zlc_plot`只作为ZLC产品内部plot package，保留Workbench、Qt widget、offline report/FigureViewer；删除无consumer Notebook/LivePlotController/standalone library surface；
- runtime generic exact/builder/live旧框架删除，只留真实Host/Plane/front/follow；
- facade按真实跨包imports收窄；无仓外兼容承诺时不做deprecation ceremony；
- 历史docs移`docs/history`或删除，tests不再解析其API名单/SHA。
- 明确保留的notebooks必须fresh-kernel执行；建立至少一个fresh Windows software CI或等价本地release evidence owner。

### 若不同意

若要正式支持standalone wheels/plot library/notebook，必须提供真实消费者名单，并接受独立build/install/version/compatibility/thread lifecycle/CI成本。

### 请裁决

`接受G1推荐`，或列出必须保留的standalone package/library/notebook API。

## G2 — Runtime live dataset、identity与partial terminal

关联：D-006/007/018/035–038/066。

### 推荐

- 大Camera默认live发布latest complete cycle与running reductions，raw stack由采集owner增量积累并在partial/terminal seal；不每次复制全部历史；
- Scan等有明确point geometry的数据逐步填充固定shape；已写位置有效，future cells invalid；
- producer向Host提交一个带token/revision的immutable sibling bundle；Plane/UI不pull plugin；
- 大camera若profile证明需要lazy copy，可用**Host-owned** materialization lane，但删除plugin自己的listener/dirty/terminal truth；
- EventRef管理causal run/shot，snapshot ref管理content；generation内schema/ref generation固定、revision严格递增；
- stopped partial正式retained、可Save、可one-shot Processor消费；不受UI freeze影响；
- 科学上不能漏event的Processor增量exact处理，纯显示derivation可latest；策略由input contract声明，不从coverage猜；不同processor可并发、同processor串行。

### 替代

若用户确实需要Camera panel在运行中随时访问全部raw stack，可选择growing full stack，但必须接受更高内存/复制成本并实现增量materialization。

### 请裁决

`接受G2推荐`，或明确Camera live必须提供完整growing raw stack。

## G3 — Plot、fit、overlay、selector与layout

关联：D-002/003/005/008/009/021–027/071/072/075/077。

### 推荐

- live **data-first**：data@N立即显示并清掉旧fit，fit@N完成后只在仍current时二次overlay；Save Fig/frozen输出等待完整fit；
- live fit按显示节奏只处理latest，不为每个不会显示的中间revision重复求解；
- Occupancy发布typed、self-contained、same-lineage overlay sibling；Workbench不重建plugin science；
- 推荐普通wheel滚外层board，`Ctrl+wheel`缩放plot；Selector状态只决定selection tool，不让普通滚轮误缩放；
- PanelState transaction idempotent，一次配置最多一张基础front；title/layout变化不重算fit；
- layout同format新增default自动补齐，结构变化显式version migration；
- UI接收窄structural plot-host protocol，不import具体plot package；
- renderer先不大拆；pulse特例合回或保持现状，等正确性修复后再决定。

### 替代

若坚持atomic data+fit，必须接受慢fit阻塞本panel和same-shot cohort，并把它明确写成产品语义/性能预算。

### 请裁决

`接受G3推荐`，或明确选择atomic fit、每revision fit或其它滚轮规则。

## G4 — Task active、preview与Monitor体验

关联：D-010/029–034。

### 推荐

- Task只锁bench hardware、当前Task draft和冲突操作；Monitor仍可查看、滚动、改布局/fit，并可准备不涉及hardware的stopped draft；
- Measurement必须在bounded cadence内live；Task必须progress + monitor preview，除非descriptor明确声明无preview原因；
- 只有首个真实publication后signal/panel才标live；terminal清除stale progress；
- preview lifecycle显式`transient/retained/final`，不用coverage冒充；
- Calibration preview显示完整long/readout/long facets；
- SLM phase应用后立即publish，camera running mean按时间节流，curve每candidate完成加一点；
- Stop与成功artifact、preview、device terminal必须同一语义。

### 请裁决

`接受G4推荐`，或选择Task期间完全冻结整个窗口；并确认Calibration三帧preview。

## G5 — Pulse、Camera与physical same-shot

关联：D-004/013/014/039–054。

### 推荐

- RepeatRegion只表示timeline内部loop；execution cycles/shots、scan sweeps、dataset repeats分开；
- Start preflight统一核target ABI、clock、counts、actual exposure/window count/cadence、FIFO capacity、dynamic device claims；
- Camera/Sequencer wiring由apparatus config声明；
- remote默认loopback或受认证/隔离的control network；每条command校验当前owner token；第二client默认busy reject，显式takeover只有SAFE成功才生效；
- UART端口必须显式配置/allowlist，禁止默认向所有COM发送probe；
- 正式hardware transport默认只由server进程持有；若允许local绕过，必须是真OS/device lock；
- active forever在连接/lease失联后按用户给定timeout自动SAFE，quiet editor不因无command被踢；
- hardware Notebook与offline教程分开，bring-up默认有限pulse并`finally SAFE`；
- DONE表示所有delayed physical outputs完成并到安全态；
- scan underflow/overflow sticky error，不能被后续refill抹掉；
- off-grid authored值保存authored+actual，Dataset坐标用actual；
- Calibration使用finite N cycles；Temperature先选物理可行的exposure/gap并配套Calibration；
- 普通Monitor可continuous best-effort；正式finite Task至少按cycle/小chunk arm/fire并核数量；只有要求绝对provenance时增加hardware marker；
- Virtual按真实cadence逐cycle并模拟camera busy/Stop。

### 需实验选择

- Temperature是保留20ms并拉长gap，还是约5ms exposure并单独/重新Calibration；
- per-cycle/chunk边界是否足够，还是absolute hardware marker近期必需；
- delayed tail若硬件难改，是否临时暴露`LOGICAL_DONE/PHYSICAL_DONE`两态。

### 请裁决

`接受G5推荐`，并选择Temperature exposure/gap策略与same-shot保证档位。
还需给出active-forever失联timeout；若不同意single server owner，请说明允许哪些local入口。

## G6 — SLM phase authority、反馈observable与真机验收

关联：D-015–017/028/055–062，加correction/settle/DVI close。

### 推荐

- Feedback只更新Pattern/base，保留frozen numeric pupil与operator wavefront；context来自显式science artifact，不读open Editor私有内存；
- Target artifact=`intensity + objective`；独立project/science context保存pupil/wavefront；
- real initial phase unknown时拒绝Feedback，直到known Send/artifact takeover；不虚构zero；
- 100 shots只做coarse；all-shot dark-subtracted fluorescence是当前Task诚实observable，不称trap depth；实际final shots由实验机variance与用户时间上限决定；
- inner WGS走到已有canonical numerical gate；outer controller用uncertainty、step clip、trust/rollback；final independent adaptive CI validation；
- UI拆`Accept best`和`Cancel/restore`；
- correction mutation必须claim/freeze并进command receipt；
- DVI在mode/GPU/gray/orientation/settle/close验收前标Experimental；USB只是优先验收路径，开发机mock bytes不算真机accepted；
- serial profile移installation calibration artifact并保存provenance；同波长vendor correction优先，跨波长需真正2D unwrap验收；
- 当前Feedback明确sparse-only；dense MRAF单独修定义/性能。

### 请裁决

`接受G6推荐`，并给final validation最大shots/时间、profile来源、correction权限、settle/DVI close事实与dense target优先级；或指出必须采用其它observable/full-phase takeover。

## G7 — Scientific artifact、命名与device final state

关联：D-045/067/069/079–083。

### 推荐

- durable unique name allocation与commit原子化，多process不覆盖；
- metadata strict JSON tree，未知类型拒绝，不自动`str()`；
- Calibration保存小deploy artifact + 完整run archive/manifest；raw frames默认策略需按容量决定，但不得只保不可重算summary；
- 旧Calibration缺unit必须显式migration；
- Temperature保存完整survival/validity/played coordinate，curve JSON只summary；
- Scan成功/Stop默认恢复pre-run device values；若选择leave-at-last，UI和run record明确显示；
- artifact保存足以复核pulse cycle、device mapping、command outcome的receipt。
- golden/oracle更新必须附独立变化依据和generator/acquisition provenance。

### 请裁决

`接受G7推荐`，并选择Calibration raw frames默认：`保存`、`只存manifest指向外部raw`、或`operator opt-in`。

## G8 — Simulation配置与hidden diagnostics

关联：D-084/085。

### 推荐

- 一个composition-owned `SimulationWorldConfig`保存稳定apparatus physics；test扰动显式scenario override；
- Device Manager只显示操作者真正需要的simulation knobs；
- hidden truth只通过test-only `SimulationDiagnostics`；若现场确需，升级为正式typed diagnostics signal，不散落production properties；
- world callback在lock外执行，有register/unregister lifecycle；Virtual与real公开contract一致。

### 请裁决

`接受G8推荐`，并列出你希望Device Manager可调的simulation参数；其余默认test-only。

## 最简回复方式

如果整体同意，可以回复：

> 接受G1–G8推荐；Temperature采用___；absolute camera marker要/不要；Calibration raw默认___；Simulation UI保留___参数。

若只不同意某几项，写`G3改为atomic fit`之类即可，其余按推荐。
