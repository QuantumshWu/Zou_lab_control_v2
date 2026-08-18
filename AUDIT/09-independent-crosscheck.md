# 09 — 全审计独立交叉复核

状态：完成（只读“审计的审计”）
复核基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
输入：`AUDIT/01`至`08`的全部报告、三个阶段summary、`DECISIONS.md`、当前repo与持久Checkpoint。
约束：没有重做全量逐符号审查；只对报告互相冲突、证据强度可疑、遗漏范围和priority缺口回到当前树核验。只新增本文，未修改代码、tests、旧文档或其他AUDIT文件。

## 1. 结论先行

本轮审计已经找到大量真实且重要的问题，不能因若干内部矛盾就整体否定。最可靠的部分是那些同时满足“当前HEAD源码位置 + 可复现反例/数值 + 明确证据边界”的结论，例如：

- Camera future-validity、finite→Follow terminal失败、UI freeze改变partial结果；
- Plane接受重复/倒退content stamp；
- delay FIFO overflow在host不拒绝；
- pulse load不核ABI/clock/geometry、count coercion/wrap；
- Camera pulse-window/cardinality无人核对，Temperature默认时序不相容；
- real SLM initial phase虚构zero、post-side-effect outcome不明、correction绕过claim；
- figure archive member collision、reader不执行format、`unique_path`并发同名；
- duplicate `DeviceSpec.key`创建后泄漏第一个leaf；
- `FluentParameterForm.reconcile()`不更新dependency graph；
- Viewer手写第二套`PanelState` parser并丢状态；
- Qt/worker/close边界存在可达的资源和硬件lifecycle缺口。

但“全项目已经逐个文件过关、可以直接按08实施”的结论目前不成立，原因有五类：

1. **08把OPEN决策提前写成了唯一目标链。** data-first、push live、typed overlay、Pulse Repeat语义、SLM phase authority等仍在`DECISIONS.md`明确OPEN，08却用确定语气串成目标架构。
2. **分报告存在直接相反裁决。** `_CameraLiveSlot`、pulse camera-window helpers、`OutputSpec`、SLM pupil artifact归属等尚未收敛。
3. **证据等级没有统一。** “当前repo零consumer”“mock bytes通过”“模型Monte Carlo”“真机物理证明”有时都被同一个P0/PASS词表示，priority难以比较。
4. **全量scope仍缺一块明显拼图。** `zlc_pulse`没有像其余七包一样完成remaining-source审计；remote/transport/UART/codec/manifest/endpoint/canonical、FPGA Python facade和全部RTL/Tcl/XDC并未逐文件裁决。
5. **priority漏掉composition shutdown P0。** Standalone Pulse先关窗后safe、Qt atexit绕过guard、Device Manager无owner executor、GUI线程直接`tune()`没有进入08 Phase 1；这些比后期renderer拆文件更接近真实硬件/进程安全。

总体判定：

| 产物 | 当前可信程度 | 判定 |
|---|---|---|
| 01 inventory | 高，scope标签需校准 | 数字可用；`source`有时含examples，不能与纯`src`数字直接比。 |
| 02–06具体反例 | 高到中 | 直接probe强；静态dead-code与物理推断需保留条件。 |
| 03/04/05 summaries | 中高 | 适合导航；不能覆盖分报告中未解决的相反裁决。 |
| 07 conflict matrix | 中 | 内容有价值，但文件自称“进行中”，尚未吸收06b–06e全部细节。 |
| 08 roadmap | 方案草案 | 不能视为user-approved architecture；应按decision gate改成条件分支。 |
| DECISIONS | 信息完整但不可操作 | 85项中84项OPEN，存在重复、工程细节与产品选择混排。 |

## 2. 复核方法与证据等级

所有审计报告与当前source均在同一HEAD，当前仅有`AUDIT/`新文档未跟踪，没有source漂移。因此报告间差异不是“审了不同版本”，而是真正的分析/表述冲突。

本文按四级使用证据：

| 等级 | 含义 | 可支持的结论 |
|---|---|---|
| E1 | 当前source + 确定性隔离反例/严格数值对拍 | 可直接列明确bug/P0；实施时仍需old-red。 |
| E2 | 当前完整调用图/consumer graph/static invariant | 可判当前仓内dead/层级/重复；不能自动证明无仓外消费者。 |
| E3 | fake/virtual/profile/Monte Carlo/官方规格推断 | 可否定过度保证、指出release blocker；不能冒充真机结果。 |
| E4 | 文档、test名、历史commit、设计建议 | 只能说明意图/冲突，不能单独判当前行为。 |

本轮没有新增Python probe或测试；非Python inventory和consumer复核使用只读repo搜索。

## 3. 报告之间的直接矛盾

### XCHK-001 — 08在用户裁决前冻结了至少八个OPEN选择

`08-target-architecture-roadmap.md`用确定语气指定：

- data-first fit；
- `context.publish_live()` push；
- 删除全部plugin slots；
- Occupancy typed overlay sibling；
- RepeatRegion只作timeline内部循环；
- SLM Feedback只改Pattern并保留wavefront；
- `unique_path` allocation+commit原子化；
- Console/Monitor采用bench-only admission。

但这些分别仍是D-002、D-007/D-038、D-008、D-013、D-015/D-055、D-067、D-010的OPEN选项。08首屏虽标“草案”，正文和phase排序会让后续执行者误以为已裁决。

裁决：08只能作为**推荐分支A**。在用户选择前，每个相关段落必须引用decision gate并列出替代路径；不得把草案用于写old-red或改production。

### XCHK-002 — Live fit还与持久Checkpoint正面冲突

02/07/08推荐data-first，Architecture/Plan正文也有data-first文字；但持久Checkpoint Milestone 3明确写：

> 不得把 Camera auto plot 删除或把同步 data+fit 改成 data-first。

当前code/tests实现atomic pair。D-002保持OPEN是正确的；任何summary若说“根权威已经决定data-first”或“Checkpoint已经决定atomic”都只读到其中一半。

需要用户一次性裁决，并同时更新Architecture、Plan/Checkpoint、plot README和atomic-pair tests。此前不应把两边任一方称为唯一authority。

### XCHK-003 — `_CameraLiveSlot`被同时判DELETE与KEEP

- 03a §5.1：`_CameraLiveSlot = DELETE after replacement`；
- 03b §5：`_CameraLiveSlot = KEEP + FIX finite validity`；
- 03c/08：建议改为Host-owned push并删除plugin slots；
- D-007/D-038：push还是统一materialization lane仍OPEN。

冲突根因是03b聚焦“plugin-local lazy materialization有价值”，03a/03c聚焦“parallel lifecycle与pull/freeze有害”。正确收敛不是现在二选名字：

- 必须删除其**独立listener/dirty/pull/terminal truth**；
- 大camera是否还需要Host-owned tokenized materialization lane，取决于profile；
- 现class可以被替换，不应在修几个field后长期保留原lifecycle。

因此统一裁决应为`REPLACE CONTRACT; implementation form pending D-038`，不是简单KEEP或DELETE。

### XCHK-004 — Pulse camera-window helper被一份报告删、一份报告保留

- 04a §4.3：`CompiledProgram.camera_window_count/exposures = DELETE`，理由是零production consumer、应直接使用schedule；
- 04b §6.3：同两方法=`KEEP AND USE`，理由是Task preflight已有能力；
- 04 summary又偏向“删除便利方法，但必须preflight”。

真正不变量是“Task必须从compiled schedule验证每row camera windows”；是否保留两个convenience methods只是API组织。按默认最简，推荐04a的形式：用`zlc_pulse.schedule.trigger_windows()`作为唯一算法，Task直接调用或让helper仅作薄投影；不得同时维护第二个window-count实现。04b的“KEEP”应解释为**保留能力，不一定保留方法**。

### XCHK-005 — `OutputSpec`的类型owner未收敛

- 03b/03c：descriptor `OutputSpec`与runtime `DatasetOutputDeclaration`重复，应引用/合并同一declaration；
- 06c：`OutputSpec = KEEP`，认为static operator contract有必要；
- 08：说保留一个declaration truth，但未指定落在哪层。

这里的事实不是两种语义都需要，而是Atom descriptor与Runtime host都需要同一`name+contract_id`。按依赖方向，Atom已经依赖Runtime，可以直接使用Runtime declaration，或把最小中性declaration放更底层；不能继续保留两个同形DTO并互转。06c的`KEEP`应读作“静态output概念保留”，不是“当前类必须保留”。

### XCHK-006 — Pupil到底属于Target还是Science Context，05a与05c表述漂移

- 05a明确：target artifact只含`intensity + objective_kind`；pupil/Zernike属于独立Science Context；
- 05c多处写“target/project artifact保存objective+pupil”以及“target save/load保留objective与pupil context”；
- 05 summary重新采用`TargetArtifact(intensity, objective)` + `ScienceContext(pupil,wavefront)`。

应以05 summary的分层为综合结论：target数值truth不应和beam/pupil context合成一个巨大target codec。UI可以提供一个project/save action同时写两个相邻sections，但不能让`save_target()`的strict target JSON暗中变成完整Editor session。

### XCHK-007 — `wait_done/safe`的PASS只表示调用原语存在，不表示语义正确

- 04c §5.7把`wait_done/safe`标`PASS`；
- 04a把`wait_done`标`REDESIGN`，因为DONE不含delayed tail且`tail_elapsed`误名；
- D-049仍OPEN。

统一措辞应是：`safe()` primitive与cancellable wait调用模式要保留；`wait_done()`的物理completion contract未过关。在D-049前不得引用04c的PASS声称pulse execution terminal可靠。

### XCHK-008 — 03b末尾“Camera完整cycle组装正确”超过其自己的证据

03b最终摘要把“Camera完整cycle组装”列为做得正确；03a/04b却已实证finite只按count切片、不查ordinal，adapter ordinal也未必是physical trigger identity。

可保留的准确说法只有：Monitor对**reported ordinal**做连续/对齐检查；Camera Dataset保留一个完整cycle的frame tuple shape。不能概括为“Camera完整cycle组装正确”。

### XCHK-009 — 07、08与AUDIT README状态没有收口

- 07首屏仍为“进行中，剩余package完成后补”；
- 08仍为“草案，待剩余package报告”；
- AUDIT README仍写“剩余逐包进行中、总结待开始”；
- 实际06a–06e均已完成，08和85项账本已存在。

这不会改变source结论，但会误导恢复/交接。因本任务禁止修改旧AUDIT，本文只记录：09完成后仍不能把README的“进度”当真实状态。

## 4. 证据过度声明与需要降级的结论

### XCHK-010 — “零production consumer”只证明当前repo，不自动授权删除public library API

03c、06a、06d的consumer graph质量高，但结论必须带条件：

- 对当前ZLC product可判dead；
- 对外部Notebook/script/library compatibility，需D-011/D-012/D-070/D-078；
- tests/examples不是production consumer，但也可能是明确的独立library产品，只是当前没有release acceptance。

因此`LivePlotController`、Notebook、data helpers、generic runtime island的删除建议有效，但实施必须先完成产品边界gate；不能把E2提升成“世界上无人使用”。

### XCHK-011 — SLM Monte Carlo的精确概率是模型结果，不是硬件测量

05b非常有价值地证明：在其独立/正态或Bernoulli假设及观测CV下，100-shot极值ratio无法支撑1%。应保留的强结论是“当前artifact uncertainty与35-site extreme gate不相容，不能宣称已证明1%”。

`1.5e-5`等精确通过概率依赖site independence、noise model、common-mode correlation和真实distribution；它属于E3，不应被摘要写成装置无条件物理定理。最终validation仍需真实shot vectors/covariance和experiment-machine evidence。

### XCHK-012 — “P0”混合了三种不同release意义

当前P0同时表示：

1. 当前正式source-checkout路径可触发的数据/硬件错误；
2. 选择standalone wheel后才触发的package-data/metadata错误；
3. 真机acceptance缺失导致的“不得宣称完成”。

例如：delay FIFO、archive collision、duplicate DeviceSpec是1；Atom wheel漏resource是2；X15213 profile无provenance/DVI optics未验是3。它们都重要，但修复/验收动作不同。

建议以后统一标签：

- `P0-CODE`：当前路径可导致错数据、错波形、泄漏或无法safe；
- `P0-PACKAGE`：选定distribution后阻断install；
- `P0-ACCEPTANCE`：软件可保留，但禁止real-hardware release声明。

### XCHK-013 — 测试通过数字只能作为局部回归证据

报告总体已经多次声明这一点，但summary仍容易让读者把`50 passed`、`178 passed`、`85 passed`当覆盖率。具体例子：

- 04b的50 camera tests不含busy trigger/physical ordinal；
- 06a的85中两项解析旧standalone install；
- 06b的178不含form dependency mutation和formal Panel Save→Viewer full roundtrip；
- SLM fake SDK只证明项目自己假定的ABI在fake两端一致。

09不汇总一个“总passed数”，因为不同agent的窄集有重叠、环境和证据目标不同。

### XCHK-014 — 01规模数字与06报告数字scope不同，不是source漂移

01把package examples也计入“Source Python”；例如`zlc_ui`的56 files/18,188 lines = 49个`src` production files + 7个examples。06b的49/16,765只算`src`。Runtime的17 vs 16同理。

因此数字本身不矛盾，但01表头应理解为“非test Python”，不能拿它断言production文件漏审。真正的scope遗漏见下一节。

## 5. 尚未完成的package/file/non-Python范围

### XCHK-015（最高scope缺口）— `zlc_pulse`没有完成remaining-source全量审计

04a深入审了model/binding/compile/schedule/scan/device/wire的pulse语义，04c审了node调用，测试表也概括了remote/transport。但以下production files没有获得与06a–06d同等级的逐文件/类/函数必要性裁决：

- `canonical.py`；
- `codec.py`；
- `endpoint.py`；
- `manifest.py`；
- `remote.py`；
- `transport/{base,lease,memory,axi,uart,uart_frame}.py`；
- root facade与metadata/notebook的完整consumer/public-surface审查；
- `fpga.py`除“wire.py重导出”以外的最终owner边界。

尤其remote/transport是sequencer真实硬件路径，不应只用“tests很全”代替source audit。当前不能宣布八包全量完成。

### XCHK-016 — 28个RTL/build源文件没有逐文件审查

当前一方非Python硬件源包括：

- 19个`.v`；
- 5个`.tcl`；
- 3个`.vh`；
- 1个`.xdc`。

04a通过Python mirror与资产tests发现delay FIFO P0，并对`wire.py`/build owner提出建议；但没有逐RTL module、CDC/reset/FIFO/width/overflow/UART/scan-bank/timing constraint审查，也未运行Vivado/iverilog formal simulation。本用户要求“整个项目一个文件一个过关”，这部分仍未完成。

因此PC-001证明“host允许一个已知会令当前mirror/RTL设计丢edge的program”，不等于“其余RTL已审过且只有这一处问题”。硬件代码应另开只读专项，真Vivado/timing仍属实验/工具链验收。

### XCHK-017 — Examples/notebooks/package launchers覆盖不完整

已深入审计Atom/Data/Plot/Runtime notebook的若干问题，但仍缺统一逐文件裁决：

- `zlc_pulse/notebooks/usage.ipynb`只被token/coverage test概括，未执行或逐cell核当前API；
- `zlc_ui/notebooks/usage.ipynb`与`zlc_workbench/notebooks/usage.ipynb`未在06b逐cell审查；
- 7个`zlc_ui/examples`、6个`zlc_plot/examples`、runtime demo未全部逐文件判保留/历史/错误路径；
- `packages/zlc_workbench/bin`的五个batch wrapper与README没有和root `bin`逐字/行为关系审计。

它们多数不是production runtime，但属于用户要求的项目文件、也是文档/验收入口；不能在最终coverage表里默认为PASS。

### XCHK-018 — Root distribution的non-Python assets没有一条真实wheel proof

06e正确发现root find-packages漏`zou_lab_control_v2` bootstrap；06c发现Atom standalone package-data漏两个scan模板。进一步对账显示：root`pyproject.toml`不会继承八个nested `pyproject.toml`的package-data配置，root自身也未声明JSON/TTF/`py.typed`/FPGA assets。

所以若D-063选择“单一installable product”，至少以下内容没有配置级保证进入wheel：Calibration/Temperature/Scan templates、SLM profiles、plot font、typing markers，以及位于`packages/zlc_pulse/fpga`而不在任何`src` package下的全部FPGA assets。未实际build wheel前，本文不宣称每项必然缺失；但这是明确的`P0-PACKAGE` gate，现有source-checkout tests不能证明。

### XCHK-019 — 有意未覆盖的真机/外部范围仍必须保持OPEN

报告正确排除了vendor binary全面审计与真硬件操作。因此以下不是audit“已通过”：

- qCMOS/Pylon真实trigger stamps/busy/overrun；
- FPGA timing/bitstream与physical delayed tail；
- X15213 SDK ABI/controller mode/orientation/LUT/correction/settle/optics；
- DVI GPU/scanout；
- dependency lock在实验机的可复现安装。

这些应进入acceptance checklist，不应因软件报告“阶段完成”自动变PASS。

### XCHK-020 — 07承诺的逐technical-doc终表尚未完成

07 §8明确说待补“每份保留technical doc的逐项KEEP/REWRITE/MOVE/DELETE表、package README/API差异、examples/notebooks正式路径与最终迁移顺序”。当前文件仍停在family级inventory和重点冲突矩阵；`zlc_pulse`/FPGA操作文档、各package technical guides、全部历史acceptance/survey并没有逐文件终态。

01开工时的81份Markdown是pre-AUDIT基线；当前104份包含后来新增的AUDIT文件，不能把新增报告数量误当原文档已逐个审完。07的高层结论可信，但其“后续补全”仍是真scope debt。

## 6. `DECISIONS.md`复核：85项需要压成少数gate

当前统计：85个decision headings，1项已裁决，84项OPEN。内容覆盖全面，但直接逐项让用户选择会把互相依赖的实现细节拆成几十次重复裁决。

### 6.1 明确重复/应互为alias

| IDs | 关系 | 建议 |
|---|---|---|
| D-007 与 D-038 | 都在选push live vs Host materialization/slot | 合并为一个Runtime live commit gate；D-038作为详细选项。 |
| D-019 与 D-063 | 都在选单一产品distribution vs standalone/source-checkout | 合并；D-063是更完整版本。 |
| D-015 与 D-055 | 都在选Feedback只改Pattern/保留context还是full takeover | 合并；D-055补context来源。 |
| D-018 与 D-066 | EventRef vs DatasetRevisionRef责任及同ref同内容 | 合并为identity gate，injectivity是其不变量。 |
| D-068 与 D-078 | facade最小化与仓外compatibility | 合并；先问是否有真实外部consumer，再决定facade。 |
| D-012、D-070、D-076 | Plot/Data/UI的test-only library surface | 归入一个“各package是否是独立public library”gate，保留包别子项。 |
| D-008、D-021、D-025 | overlay owner、绑定拒绝点、repeat scope | 一个overlay contract下的三个子问题，不应分三轮独立选择。 |
| D-006、D-035、D-036 | finite live view、stopped partial、processor subscription | 同一extent/lifecycle gate，先定terminal matrix再定view。 |
| D-014、D-039、D-045 | same-shot evidence、trigger mapping、capture provenance | 同一Pulse↔Camera proof gate的强度/配置/持久化子项。 |
| D-022、D-023、D-024、D-027 | panel interaction/output/restart persistence | 同一PanelState durability与cohort policy，可作为一组裁决。 |

### 6.2 不应继续当“用户产品选择”的明确错误

以下有最小正确答案，不需要用OPEN decision拖延：

- archive member key collision、reader忽略format/duplicate/nonfinite；
- unknown provenance自动`str()`；
- non-bool validity truthiness；
- duplicate DeviceSpec key先创建后泄漏；
- form reconcile旧dependency graph；
- Viewer第二个残缺PanelState parser；
- standalone Pulse在safe前关窗；
- Qt owner同步等Future/直接跑hardware tune；
- worker/executor无shutdown owner；
- pulse load不核ABI/clock/geometry、count silent wrap、delay FIFO无validator；
- correction mapping绕过现有DeviceUse claim；
- Atom顶层反向importWorkbench。

用户仍可裁最终大方向，但这些不应各自生成“也许保留错误行为”的选项。

`D-066`的content-digest选项还与当前明确“不新增产品hash”的约束冲突，应删除该选项，除非用户主动重新开放hash。

### 6.3 建议给用户的八个顶层gate

保留旧ID作traceability，但实际裁决按下列顺序：

1. **G1 部署与public product surface**：D-011/012/019/020/063/064/068/070/076/078。
2. **G2 Runtime data identity/lifecycle**：D-006/007/018/035/036/037/038/066。
3. **G3 Plot/Panel/interaction**：D-002/003/005/008/009/021–027/071/072/075/077。
4. **G4 Task/preview policy**：D-010/029–034。
5. **G5 Pulse↔Camera execution proof**：D-004/013/014/039–054。
6. **G6 SLM authority与验收**：D-015–017/028/055–062，再补runtime correction、settle与DVI close子项。
7. **G7 Scientific artifact与device final state**：D-045/067/069/079–083。
8. **G8 Simulation与diagnostics**：D-084/085。

这样用户先决定会改变架构的8件事，工程子项才有上下文；不必连续回答84个彼此依赖的问题。

### 6.4 决策账本遗漏的三个真实选项

1. **Runtime correction policy**：Editor可在stopped状态用claim临时load/toggle，还是correction只能由Device Manager安装配置改变。05c明确要求用户选择，账本只有receipt/profile，没有该权限策略。
2. **SLM settle policy**：profile最坏固定值、用户authoring值、还是command-dependent/adaptive；无论哪项都需该head实验测量。05c列出但账本未单列。
3. **DVI session close**：接受销毁window即撤销输入，还是要求持久presenter/service。D-059只问Experimental级别，没有决定close语义。

次要遗漏：是否允许Unicode artifact filename（06a DUR-02）可并入G7命名策略，无需单独顶层decision。

## 7. Priority复核：最严重但尚未进入08 Phase 1的P0

### P0-MISSING-1 — Composition shutdown / hardware command retirement

这一组是当前正式source-checkout产品可达的`P0-CODE`，08没有列入Phase 1：

- standalone Pulse Editor在window closed后才调用safe/release；
- `zlc_ui.qt` atexit直接`sip.delete`全部顶层窗口，绕过所有close guards；
- Device Manager outer/inner executors没有lifecycle owner，hung vendor call可留住进程；
- generic device`tune()`在Qt GUI线程同步执行，冻结Stop/close；
- FigureViewer bound controls在Qt slot同步等待worker最多10秒。

建议把它们提升到Phase 1，与device leak/SLM unknown command同级，先建立“窗口不能在owned hardware/worker释放前消失”的纵向guard。Qt display editor blocking可随后同一lifecycle切片修。

### P0-MISSING-2 — Dynamic form reconcile

`FluentParameterForm.reconcile()`实测使用旧`enabled_when` graph，直接命中用户“新增logic node总显示错”的点名症状。08只笼统把UI放Phase 3，未列这条。

它应在任何新增node或schema migration前先修，且只需现有form owner一个old-red，不依赖D-002/overlay/renderer大裁决。

### P0-MISSING-3 — Root/wheel asset gate（条件P0）

若用户选installable product，root bootstrap缺包、nested package-data不继承、FPGA assets在`src`外必须在Phase 0/1解决；不能等Phase 6“metadata收敛”才发现wheel根本没有正式resource。若选择source-checkout-only，则把它降为删除误导metadata的工作。

### P0-MISSING-4 — `zlc_pulse`与RTL audit coverage

在remote/transport/RTL未审前，Phase 1不能声称pulse hardware blockers清册完整。先补只读scope，不代表要延迟已知的delay FIFO/ABI/count修复；两者可并行。

## 8. 重新排序后的执行前置条件

本文不修改08，只给出复核后的门：

### Gate A — 不等用户的明确P0

1. 数据编码拒绝静默覆盖/类型丢失；
2. device/install duplicate owner与close ordering；
3. pulse ABI/clock/count/FIFO preflight；
4. form reconcile与Viewer唯一decoder；
5. close guard/worker ownership/Qt blocking hardware call；
6. Atom reverse dependency与source/wheel resource truth（按部署选择）。

### Gate B — 用户八项顶层裁决

完成G1–G8，尤其G2 live contract、G3 data/fit/overlay、G5 pulse-camera、G6 SLM authority。

### Gate C — 再定08实现路线

只有Gate A/B完成，才能把08从“推荐方案A”升级为target architecture。若用户选择atomic fit、slot materialization、full SLM takeover或source-checkout，08相应章节必须改写，不能只改DECISIONS状态。

### Gate D — 补全审计scope

完成zlc_pulse remaining source、RTL/build assets、未审notebook/examples/package launchers；形成明确coverage appendix。

## 9. 最终可信结论

### 可以直接进入问题清单的事实

- 各具体E1反例是真实当前HEAD缺陷；
- Runtime/plot的平行live框架、Viewer parser、Workbench wake、save路径和package metadata确实存在多份truth；
- Pulse/Camera当前不能证明绝对physical same-shot；
- 真实X15213链尚未完成hardware/optical acceptance；
- 当前tests/docs存在多处false-green与historical authority回流。

### 仍不能宣称

- “全项目所有文件、class/function/test已经审完”；
- “08已经是用户批准的目标架构”；
- “84个OPEN decision可以逐项独立回答”；
- “所有P0都已进入Phase 1”；
- “virtual/mock/50或178个passing tests证明真机链”；
- “当前只有已报告的一个RTL问题”。

### 本轮独立复核结论

现有审计足以支撑一轮高质量的用户架构裁决与若干明确P0修复，但还不足以关账。最优下一步不是继续扩大08，而是：

1. 先把84个OPEN压成8个gate交用户；
2. 同时把composition shutdown、form reconcile和明确数据/hardware P0提到最前；
3. 补完`zlc_pulse` remaining source与RTL scope；
4. 用户裁决后再生成一份真正final、无条件分支的target architecture。

本文没有改变旧报告；发生冲突时，以本文的“冲突未收敛”状态为准，直到用户明确裁决或新增当前HEAD证据关闭它。
