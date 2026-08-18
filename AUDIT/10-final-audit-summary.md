# 10 — 全项目只读架构审计最终总结

状态：完成；等待用户裁决，不含任何代码修复。
基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`

## 1. 最终判定

这个项目不是“应该全部重写”的废墟。八层划分、immutable dataset、SignalPlane/Host、Plot semantic projection、Pulse model/compiler、Calibration science、SLM canonical target/phase等主骨架都有真实价值。

但它目前也不能被判定为“只是需要清理代码风格”。审计确认了四类系统性问题：

1. **局部正确、跨层断链**：每层各有一份revision、preview、phase、pulse count、panel state或artifact truth，连接处没有冻结同一个context。
2. **旧框架与真实路径并存**：Runtime约4k行无人使用的generic live/builder体系旁边，Camera/Scan/Calibration又各自手写slot；Plot也有无consumer LivePlotController与真实RasterHost两套pipeline。
3. **测试很多，但证据层错位**：1,346个tests里相当一部分保护private shape、fake自洽、旧docs/API表或offscreen/virtual行为；它们不能证明physical same-shot、FPGA SAFE、SDK ABI、真机光学和高精度统计。
4. **文档把历史、目标和现状混成一份authority**：同一主题保留互相相反的段落与Checkpoint，tests又让旧Markdown重新变成机器truth。

所以最终结论是：**保留骨架，先关数据/硬件/lifecycle P0，再由用户裁8个产品gate，然后沿唯一owner删平行实现；不要再加新framework。**

## 2. 审计scope

- 459个tracked Python全部归档：271 source/root/FPGA shims、174 tests/support、14 examples；
- 166 test files、1,346 test definitions、55,304行test code逐文件分类；
- 7 canonical notebooks；
- 28个Verilog/VH/Tcl/XDC文件及全部testbench/proc/module；
- 15个Windows batch入口；
- package JSON/profile/templates、fixtures/goldens/fonts/metadata；
- 81份原Markdown；
- 四条端到端重点链。

完整映射见`06-symbol-coverage-index.md`。未执行真硬件、Vivado、烧写、vendor SDK或光学实验；这些作为证据边界而非漏审。

## 3. 立即修复、不等待产品选择的P0

### 3.1 数据与artifact完整性

- Figure archive member名可互相覆盖，writer能产出自身reader打不开的文件。
- Reader不严格执行自己的format/version/shape/duplicate/nonfinite contract。
- numeric validity接受整数/float/NaN后转bool；错误mask可变成正式truth。
- snapshot restriction丢coordinate labels，部分合法labeled axis直接崩溃。
- selection按重名label静默选错axis，text coordinate不支持。
- `unique_path()` 32个并发调用得到同一路径；后续atomic replace可互相覆盖。
- Calibration artifact无严格version/unit，saved samples不验连续/同run；Temperature artifact只留summary而丢逐site survival。

### 3.2 Runtime live与partial truth

- finite Camera把未采future cells填零却标valid；Occupancy分类全容量并标complete，制造未来empty事实。
- finite live到Final会让Follow Processor失败。
- Stop一帧后，是否被UI freeze过会改变terminal dataset shape、coverage和transient语义。
- Plane接受同generation snapshot generation漂移、revision重复/倒退；Plot随后静默丢更新。
- companion-only overlay advance不调度；generation replacement绕过same-shot cohort。
- slot mutate/dirty/pull race可让同一content revision发布两个EventRef。

### 3.3 UI、线程与shutdown

- `FluentParameterForm.reconcile()`保留旧`enabled_when` dependency graph；新增logic/schema后字段启用错误。
- duplicate `DeviceSpec.key`会创建两台leaf、dict只留后一台、首台永不close。
- Standalone Pulse Editor先关window再safe/release；失败时用户已失去控制面。
- Qt atexit直接`sip.delete`顶层窗口，绕过device/task close guards。
- Device Manager多层executors无owner/shutdown；hung vendor call可阻止退出。
- generic device`tune()`在Qt线程同步执行；FigureViewer control可在Qt slot等Future 10秒。
- Workbench写了无锁OwnerWake，而UI已有带锁/replay/fault的正确primitive却无人使用。
- Figure Viewer第二套PanelState parser实测丢selector/classifier/focus/published outputs。

### 3.4 Pulse host/remote/transport

- `load()`不核program target ABI、clock、connected geometry。
- sweeps/count接受bool/float/0/negative并silent coercion，超32位可wrap。
- compiler/pack没有delay event FIFO capacity validator，RTL可静默丢波形。
- remote默认LAN bind且无认证；旧handler在owner replacement后仍能command。
- takeover SAFE失败仍把新client设owner。
- `InterprocessDeviceLease`只是进程内dict且没有transport使用，名字/测试形成false-green。
- auto resolver会依次打开并向所有COM发送probe，可能触碰其他实验仪器。
- codec接受unknown/duplicate/nonfinite/coercion；canonical mapping key可碰撞。
- AXI负地址wrap成大unsigned；timeout/close不能证明物理命令与reader/process已终止。
- UART reply decoder接受超contract frame；RTL截断帧可长期占control path。
- hardware notebook最后cell forever fire全TTL/DAC且无finally SAFE，文档宣称的idle auto-safe并不存在。

### 3.5 FPGA/build physical boundary

- 50MHz engine主逻辑没有`create_clock`；Tcl timing gate可能只检查JTAG TCK并误报通过。
- `CMD_SAFE`不独立gate physical pins；已enable的DAC clock在SAFE后仍可跳，LOAD时可在FIRE前打开。
- DONE早于delayed FIFO drain；`wait_done()`后SAFE/LOAD可能截断物理尾部。
- XDC行序与top手写index是两份lane/pin truth，word63 fingerprint不检测pin mapping改变。
- scan underflow可被清掉、point0不验bank、短frame无read-latency contract。
- FIFO overflow和若干protocol error不sticky、不loud fail。
- Tcl所谓safe project dir只查路径长度，误设环境变量可递归删除不属于build的目录。
- program/flash默认第一个target/device，可能写错板或永久烧错flash。
- 普通suite不compile/run任何RTL testbench；多数bench打印BAD仍exit 0。

在这些边界修复/硬件复核前，不应把新bitstream称为timing-qualified，也不应依赖当前恢复脚本自动program/flash或把remote server当安全LAN设备。

### 3.6 SLM command/context

- real X15213从未send/read当前phase却把zero报告为last-commanded；failure restore可覆盖真实未知状态。
- write/display/readback某阶段失败后，physical可能new而software仍old，contract没有unknown outcome。
- correction load/toggle绕过DeviceUse，可在Task/Send中改变mapping。
- Feedback丢objective、Gaussian pupil和operator wavefront，使用default hard pupil并把full science phase当Pattern warm start。
- Target JSON丢objective；candidate artifact缺actual target/science context/device mapping/receipt。
- external Feedback改变device后，open Editor仍显示旧draft并可覆盖。

## 4. 需要用户裁决后才能定的核心架构

完整选项与默认建议已压缩到`DECISIONS-PRIORITY.md`的8个gate：

1. 产品deployment与public surface；
2. Runtime live dataset/identity/partial；
3. Plot/fit/overlay/selector/layout；
4. Task active与preview；
5. Pulse↔Camera/FPGA/remote execution proof；
6. SLM authority/observable/validation/真机路径；
7. Scientific artifacts、naming和device final state；
8. Simulation config与hidden diagnostics。

`DECISIONS.md`保留109个细ID作traceability，但不要求用户逐条回答。`08-target-architecture-roadmap.md`是按推荐选项串成的**方案A草案**，不是已批准架构。

## 5. Plot、fit、overlay、selector最终审计结论

### 保留

- PlotSession semantic projection、fit models/Jacobians、persistent renderer artists；
- RasterFront/identity、Qt front handoff；
- Board same-shot cohort概念；
- selector/gesture核心；
- PanelState作为authored state owner。

### 重设计

- live data-first还是atomic由G3裁决；审计推荐data-first，Save Fig等待fit。
- `_match_host_to_panel`改成一次idempotent state transaction；noop不render、不solve。
- typed Occupancy overlay由plugin发布，Workbench不组science；ROI/binning统一坐标owner。
- composite input stamp包含primary+companions，overlay-only变化可更新。
- generation host replacement也进cohort。
- Qt controls异步，不等Future。

### 删除/合并候选

- 无product caller的`LivePlotController/_live_channel`第二pipeline；
- `FitNumericTable`与test-only public surface；
- duplicate Viewer parser；
- duplicate owner wake；
- renderer只保留一种组织规则，文件拆分不先于correctness。

## 6. Measurement/Task/signal/preview最终结论

推荐一个Host-owned atomic commit contract：output siblings、revision、coverage、run record一次提交；Plane freeze纯读取。大camera可有统一Host materialization lane，但当前plugin slot的listener/dirty/pull/terminal lifecycle应替换。

Descriptor必须让新Logic Node默认满足：

- Measurement bounded live；
- Task progress+preview；
- resolved preview属于resolved output；
- terminal清除progress；
- artifact存在且可读；
- first real publication前不假报live。

当前新增node反复漏preview不是个人粗心，而是框架不强制且通用test只检查`execute/evaluate`。

## 7. SLM solver与feedback最终结论

### Sparse路径

保留WGS-Kim、fixed far-field phase、caller-owned optimizer state与selected DFT。真实`1024×1272` 5×7 default约0.53秒；hot8停在ratio约1.027，继续到已有1%数值gate只多约0.06秒。为了省inner几十毫秒而多拍一个100-shot candidate是反优化。

### Feedback

当前100-shot single-site SEM约4.4–5.1%，旧run可10–14%；35-site extrema噪声地板约1.2。`ratio<=1.01 + SEM<=0.5%` validation几乎不可达。Controller又无uncertainty、step clip、rollback、invalid recovery。

推荐：100 shots只coarse；uncertainty-weighted trust update；inner solve到gate；confidence best；independent adaptive CI validation。当前observable诚实名称是all-shot dark-subtracted fluorescence，不是trap depth。

### Dense路径

Gaussian没有MRAF noise region、random initial、固定300轮且FOM无意义；真实shape约27秒，Flat Top约15秒。先修MRAF问题定义/early stop，再决定CPU/GPU；GPU不会修正错误目标。

## 8. Pulse/Camera科学调用最终结论

统一词汇：timeline RepeatRegion、execution cycle/shot、scan sweep、dataset repeat分开。Task在任何side effect前用actual camera working point与compiled trigger windows核cardinality/cadence；`trigger_windows()`是唯一算法owner，camera-named helper只可薄投影。

Temperature当前20ms exposure面对5.02ms edges不成立；Calibration 31ms等自由参数也可超过固定gap。Virtual同步爆发全部trigger并只延迟DONE，不能证明真camera busy或arrival cadence。

absolute same-shot若是产品要求，需要hardware marker/trigger counter；连续retrieved/copied index只能证明可见frames没有gap。

## 9. 大类与层级裁决

### 应保留但拆责

- `ConsolePresenter`：保留composition owner，拆panel projection、logic execution、artifact/layout、device controls协作者。
- `PlotSession/MatplotlibRenderer`：保留session/renderer owner，先删重复pipeline/noop fronts，再决定文件组织。
- `Calibration calibration.py/task.py`：拆domain+codec、site detection、readout training、orchestrator、sample/replay、report adapter；不复制science truth。
- `SimulationWorld`：保留一个state owner，pure physics helpers可拆；callback移锁外、加unregister、hidden diagnostics隔离。
- `PulseEditorPresenter`：纯Pulse authoring mutation回`zlc_pulse` owner，Workbench只接线/device/preview。

### 明确层级错误

- Atom Calibration/Sequencer/SLM直接import Workbench composition root；
- Workbench临时构造Occupancy science overlay；
- UI/Viewer维护domain parser/save truth；
- root与package launchers给出不同bootstrap；
- FPGA `wire.py`混runtime packing与build/resource/CLI；
- test-support通过root path injection变成隐式跨包API。

## 10. 死代码与历史残余

高置信删除候选：

- Runtime exact/reservation/builder/monitor/live-port/preview旧岛；
- Runtime `_failure.py`, `_public.py`, `cleanup.py`等只连dead path/test的部分；
- Data `numeric.py`, `AxisSourceRef/ResolvedPointRows`, `ValuePayloadContract`等零product consumer clusters；
- Plot LivePlotController、FitNumericTable与无consumer conveniences；
- Pulse `engine_model.py`移tests并只留必要oracle；假的Interprocess lease/dead transport helpers；
- Atom dynamic output resolver、fake-only FakeNodeHost/FakePulseStreamer、test-only hidden oracle properties；
- UI Graph/FormGrid/gallery-only surface（由G1最终确认）；
- duplicate package launchers、duplicate fit contract/SHA tests、duplicate surveys；
- historical acceptance/reacceptance/survey/goal archives从active docs移走。

“零repo consumer”是E2证据：若用户确认有仓外脚本，相关surface转为compatibility decision，不直接删。

## 11. 测试体系结论

静态分类：L0 shape/text 116、L1 direct 642、L2 seam/private 213、L3 process/UI 295、L4 virtual vertical 80、L5 real hardware 0。非互斥markers包括private 220、monkey 109、fake/mock下限172、offscreen 304、source-token 50、sleep 45。

需要建立lanes：software、gui_offscreen、virtual_vertical、notebook_offline、real_screen、hardware。每个result绑定HEAD/dirty/env/artifact。删除wrong-checkout、docs SHA、fake self-tests和dead seam self-tests；关键场景从private presenter提升到public handle/real Host。

现有green不能证明：

- RTL compile/STA/bitstream；
- camera accepted physical edges；
- X15213 optical phase；
- real-screen不卡顿；
- 100-shot 1%；
- wheel/install completeness。

## 12. 推荐实施顺序

### Phase 0

- 用户回答8个gate；
- 暂停新增Logic Node/public framework；
- 对build/program/flash与remote LAN路径采取保守操作边界。

### Phase 1：明确P0

- archive/validity/snapshot/unique path；
- duplicate device key、form reconcile、unique decoder；
- close guard/executor/Qt blocking hardware call；
- Pulse remote ownership/security/COM、load/count/FIFO；
- FPGA clock/SAFE/DONE/board identity/build containment/target；
- camera cadence/windows；
- SLM unknown state/correction lease；
- Atom reverse dependency/resources。

### Phase 2：Runtime transaction

- fixed schema/invalid future/partial terminal；
- atomic commit/pure freeze/stamp invariants；
- processor policy、companion/generation cohort、preview lifecycle。

### Phase 3：Plot/UI

- data-first或用户选的atomic语义；
- idempotent PanelState；typed overlay；selector/layout；
- remove blocking Future、duplicate parser/wake。

### Phase 4：Measurement science

- finite cycle API、Camera preflight、Calibration/Temperature；
- scan claims/actual coordinate/final device state；
- bounded buffer、complete artifacts。

### Phase 5：SLM

- science/device context、known command receipt；
- numerical gate、robust controller、CI validation；
- real device acceptance；dense MRAF另行。

### Phase 6：删除、部署、docs

- 删dead frameworks/pipelines/seams；
- 单一product manifest/dependency lock/CI；
- 历史docs隔离，重写current architecture/contracts/status。

## 13. 当前操作建议

在修复前，审计建议：

- 不用当前`build_and_program.bat`默认路径自动program/flash；显式确认build root与唯一target/device。
- 不把Pulse remote无认证暴露到非隔离LAN；优先loopback/防火墙并显式UART port。
- 不依赖`CMD_SAFE`/DONE文字承诺作为物理安全证明；由硬件负责人核现有部署bitstream/pin状态。
- 不用当前Temperature默认20ms/5.02ms protocol做真机科学结论。
- real X15213在known Send/receipt前不运行需要restore语义的Feedback。
- 不把100-shot curve plateau/ratio当1% trap-depth验收。

这些是风险控制建议，不等于已对实验机现有部署做了硬件测量。

## 14. 交付物

- `README.md`：审计索引/进度；
- `01–09`：inventory、四条重点链、逐包/FPGA/tests、docs与独立复核；
- `06-symbol-coverage-index.md`：全scope映射；
- `DECISIONS-PRIORITY.md`：用户8个gate；
- `DECISIONS.md`：细项traceability；
- `08-target-architecture-roadmap.md`：推荐方案A草案；
- 本文：最终总结。

没有修改production、tests、旧文档或硬件；没有commit。下一步只能是用户裁决或另行授权实施明确P0。
