# 06b — `zlc_ui` / `zlc_workbench` 全量边界、状态与生命周期深审

状态：本子阶段完成
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：`zlc_ui` 与 `zlc_workbench` 的全部 production source、直接 tests、package docs、metadata、应用入口与共享 Windows launcher；`ConsolePresenter`、`PlotPanelPort`、selection/fit/same-shot 的内部算法只引用 02/03/04 系列报告，不在本文重复大段。
约束：只读源码、权威设计、既有测试与无硬件隔离探针；只新增本文，未修改 production、tests、旧文档或硬件。

关联报告：

- [02-plot-fit-overlay-selector.md](02-plot-fit-overlay-selector.md)：PanelState、fit/overlay/selector与front-set；
- [03-runtime-measurement-task-summary.md](03-runtime-measurement-task-summary.md)、[03b-task-preview-contract.md](03b-task-preview-contract.md)：Logic/preview/TaskConsole运行链；
- [04a-pulse-api-semantics.md](04a-pulse-api-semantics.md)：pulse wire/scan/repeat及Editor同步；
- [04b-camera-same-shot-contract.md](04b-camera-same-shot-contract.md)：camera/trigger/same-shot；
- [05c-slm-device-editor.md](05c-slm-device-editor.md)：plugin-local device editor与composition close/lease边界。

## 1. 结论先行

这两个包不是“全部推倒”。当前最值得保留的骨架是：

- `zlc_ui` 没有 import `zlc_atom`/`zlc_plot`/`zlc_runtime`，绝大多数 view 只发intent、吃plain projection；
- `FormSpec`/`FormFieldProps` 与 `authoring_form.project_*` 是一条正确且唯一的schema→widget边界；
- board几何算法只在`zlc_ui.board`，Workbench没有复制pixel layout；
- `DeviceUseCoordinator`是共享session设备仲裁的唯一owner；
- `PanelState`本体是TaskConsole panel authored state的合理唯一记录；
- `PulseEditorState`是`zlc_pulse` document之外必要的Editor扩展状态，而不是完整第二套pulse model；
- Workbench没有持有`QWidget`，四个正式窗口都从`zlc_ui` facade打开；
- archive底层使用JSON metadata + `allow_pickle=False` arrays，单个NPZ原子写入。

但当前不能把UI/composition判为健康，原因不是风格，而是已经存在可复现的state/lifecycle断裂：

1. **动态表单增量更新没有更新依赖图。** Logic/Panel表单一旦`reconcile()`后换了`enabled_when` controller，UI仍按旧controller启用/禁用字段；这直接命中“新增logic node总显示不规范”的用户症状。
2. **生产线程桥不是包内已经写好的线程安全实现。** `zlc_ui.QtOwnerWake`有锁、dispatch/replay与fault保存，却没有生产消费者；Workbench另写了无锁`OwnerWake`，又另写一个无法shutdown的`ThreadPoolExecutor`桥。
3. **Pulse Editor存在两个close等级。** session-bound窗口先安全退役再允许关闭；standalone窗口却先完成window close，再由`closed` signal调用presenter cleanup。safe失败、lease未释放或Qt signal异常时，操作面已经消失。
4. **Figure Viewer有第二套且不完整的`PanelState` parser。** 实测会丢`published_outputs`、`selector`、`classifier_threshold`与`focused_cell`；它的Flow tab又只理解旧式multi-panel archive，而正式Panel Save写的是另一种结构。
5. **Device Manager的异步并非有界生命周期。** 每个window建一个不可关闭worker pool；scan的“20秒”是逐future等待，N个挂死family最坏N×20秒，并把挂死线程留在进程里。
6. **Qt全局退出直接`sip.delete`所有顶层窗口。** 这能减少wrapper销毁崩溃，却会绕过Device Manager/TaskConsole/Pulse/SLM的close guard和硬件退役顺序；它不是可接受的设备安全兜底。
7. **保存/重开仍有平行真相。** formal Panel Save、`ExperimentSession.save_figure`、Viewer legacy archive解释、`PanelPlotAnnotations`与`PanelState.classifier_threshold`同时存在；每条单测各自通过仍不能组成一个round-trip contract。
8. **大而混杂的文件已经开始掩盖owner错误。** `fluent/fluent.py` 3,931行尚属一个UI package内的机械可拆文件；`pulse_editor.py`全文件3,320行，其中`PulseEditorPresenter`约2,390行/约90个方法，同时拥有authoring mutation、连接、硬件drive、scan、IO、preview和sync，已经是God object。

总体裁决：

| 区域 | 裁决 |
|---|---|
| `zlc_ui`纯view、Form model、board geometry、pulse VM/views | `KEEP` |
| `FluentParameterForm.reconcile` | `P0 FIX IN EXISTING OWNER` |
| `zlc_ui.QtOwnerWake` + Workbench重复wake | `USE EXISTING UI PRIMITIVE; DELETE DUPLICATE` |
| Workbench worker/executor ownership | `P0 LIFECYCLE REDESIGN, NO NEW FRAMEWORK` |
| Device Manager presenter | `KEEP + FIX TRUTH/THREADING` |
| Pulse Editor presenter | `SPLIT RESPONSIBILITY AT EXISTING DOMAIN BOUNDARY` |
| Figure Viewer/parser/archive | `ONE CODEC + ONE FORMAL SAVE CONTRACT` |
| `zlc_ui.graph` | `DELETE/MOVE TO DEMO` |
| acceptance/gallery-only public surface | `MOVE TO TOOLS OR MAKE PRIVATE` |
| docs/metadata | `STALE/CONTRADICTORY; DO NOT USE AS PROOF` |

## 2. 规模与边界图

当前规模：

| Package | production Python files | physical lines | direct test files | test lines |
|---|---:|---:|---:|---:|
| `zlc_ui` | 49 | 16,765 | 15 | 4,716 |
| `zlc_workbench` | 30 | 15,786 | 27 | 14,271 |

当前意图链是：

~~~text
zlc_atom declarations / zlc_runtime state / zlc_plot hosts
                         |
                         v
                 zlc_workbench composition
        authoring projection / state / lease / lifecycle
                         |
                         v
                  zlc_ui handles + views
                 intent out / projection in
~~~

import方向基本守住了，但object方向并未完全守住：

~~~text
plot host(domain object)
  -> TaskConsoleHandle.show_panel / PulseEditorHandle.show_preview
     / FigureViewerHandle.show_figure
  -> host.qt_widget(), host.wheel_target(), host.logical_size
~~~

因此“zlc_ui不import plotting package”为真；README所说“domain objects do not cross the boundary”为假。当前其实采用的是**structural host port**，只是没有诚实写入contract。

## 3. 已确认缺陷与架构问题

### UIWB-001（P0 state）— `FluentParameterForm.reconcile()`保留旧依赖图

位置：`packages/zlc_ui/src/zlc_ui/form/qt_form.py:852-905,1068-1255`。

构造器只在第一次建立表单时计算`self._dependents`并连接controller变化：

~~~text
old spec: a enables b
reconcile
new spec: c enables b
~~~

隔离探针结果：

- reconcile前：`_dependents == {'a': ['b']}`，`b`按`a`禁用；
- reconcile后：`_dependents`仍是`{'a': ['b']}`；
- 新controller `c`的false值没有禁用`b`，改`c`也没有正确驱动它。

`reconcile()`更新了`_spec`、`_fields`、`_handlers`和widgets，却没有重建依赖图，也没有对新controllers重新应用condition。新增logic node、设备schema升级、field availability变化都会走这条稳定form owner，故这不是理论边角。

最小修复owner就是`FluentParameterForm`自身：在全部incoming validation成功后，先构造新的dependency map并校验controller存在；完成widget reconcile后原子替换map并重算所有controller。不要在Workbench为每种node重建窗口，也不要加logic-specific条件框架。

同模块还有两项次级风险：

- `_controller_changed()`用普通`current == value`，而form choice codec特意区分`bool/int/float`；`True`与`1`可能被当成同一个enabling value；
- `_automatic_toggled()`在无minimum时填`0`，没有考虑`maximum < 0`；可产生widget/field bounds冲突。

这两项需各自补最小probe后修，不能用“表单测试很多”代替。

### UIWB-002（P0 lifecycle）— Workbench worker没有owner、shutdown和统一deadline

位置：

- `zlc_workbench/board.py:177-217` `attach_qt_worker()`；
- `zlc_workbench/device_manager.py:329-354` `_scan_families()`；
- `zlc_workbench/apps/device_manager.py:54-81` composition。

`attach_qt_worker()`创建`ThreadPoolExecutor(max_workers=1)`，只返回`run(work, deliver, failed)`。pool本身无返回值、无`close()`、无cancel、无drain；每次Device Manager build都有一个executor活到解释器退出。

Device discovery又在这个worker内部创建第二个pool。全部discover同时submit，但按dict顺序逐个执行`future.result(timeout=20)`：若多个family都挂住，总等待可线性累计到`family_count × 20s`。`shutdown(wait=False)`不终止已经运行的vendor call，非daemon executor thread还可能使窗口关了、Python进程仍不退出。

同一问题也覆盖Init：outer worker没有任务deadline；device factory永不返回时，presenter永远busy，close guard也永远无法完成。

最小方向不是再建“worker manager”：

1. 让现有composition明确持有一个可`submit/close`的worker owner；
2. presenter close先拒绝新任务，再有限等待/报告仍未释放任务，最后shutdown；
3. discovery使用一个总deadline，而不是每future一份deadline；
4. 无法取消的vendor discover必须由具体plugin给出可终止/有界调用，Workbench不能谎称`wait=False`已经停止。

### UIWB-003（P0 hardware close）— Standalone Pulse Editor在安全退役前已经关闭窗口

位置：

- `zlc_workbench/apps/pulse_editor.py:217-252` `create_window()`；
- 同文件`:254-308` `create_bound_window()`；
- `zlc_workbench/pulse_editor.py:1659-1708,3266-3284`。

session-bound入口正确安装close guard：guard调用`presenter.close()`，失败时显示warning并返回false，窗口继续存在。

standalone入口仅执行：

~~~python
window.closed.connect(window.presenter.close)
~~~

即window已提交关闭后才做`_retire_drive()`、`sequencer.safe()`、lease release、owned connection close与preview host close。若safe/lease release失败，窗口已经消失；Qt signal callback抛异常也没有可靠的操作者恢复面。

这违背权威“任何window、claim或device ownership尚未释放都不得伪装成成功退出”。standalone和bound应走同一个既有close-guard语义，差异只在是否关闭注入的sequencer。

另一个真相问题在`PulseEditorPresenter._release()`：owned sequencer的`close()`异常被静默吞掉，随后仍清空引用并允许重新dial。软件报告offline/新连接时，旧connection可能仍活着。应保留失败事实并拒绝伪成功替换；不需要新connection registry。

详细pulse compile/AppliedState/scan/repeat问题见04a，本文不重列。

### UIWB-004（P1 concurrency）— 正确的`QtOwnerWake`无人使用，生产使用无锁复制品

位置：

- `zlc_ui/concurrency/owner_wake.py:26-105` `QtOwnerWake`；
- `zlc_workbench/board.py:44-81` `OwnerWake`；
- 同文件`:220-248` `attach_qt_owner_turn()`。

`QtOwnerWake`已有：

- `threading.Lock`保护scheduled/dispatching/replay；
- dispatch期间到达的新completion精确保留一次replay；
- owner-thread bind/detach检查；
- callback fault保存。

生产`LiveBoard`却使用只有一个普通bool的`OwnerWake`，worker thread写、GUI thread读均无同步；它与另一个Qt relay分开接线。前者只在UI tests/gallery使用，后者在Workbench tests/生产使用，形成两套各自通过测试的并发语义。

裁决：保留`QtOwnerWake`并把现有生产consumer接过去，删除Workbench复制品；若Headless test需要普通trigger，应测试同一state machine的非Qt核心或直接驱动callback，不能保留第二份并发算法。

### UIWB-005（P0 shutdown）— Qt atexit teardown绕过所有composition close guard

位置：`zlc_ui/qt.py:36-68`。

`_destroy_windows_before_python_lets_go()`枚举全部top-level widgets并直接`sip.delete(widget)`。这能在解释器析构期减少Qt wrapper crash，但不会触发正常close handshake，也没有：

- TaskConsole node/plot worker退役；
- Pulse sequencer safe/lease release；
- plugin device editor command retirement；
- session installation/device close；
- Device Manager retained control window顺序。

测试`test_qt_app_single_entry.py`把“注册并直接销毁窗口”本身当作成功，却没有带一个持有hardware/device lease的正式flow。此处只能是**最后的GUI内存兜底**，不能承担产品shutdown。正式app必须在`application.exec_()`外层`finally`完成composition close；Notebook/kernel与异常退出仍需用户裁决其安全策略。至少不能把`sip.delete`后无窗口误报为设备已安全关闭。

### UIWB-006（P1 truth）— Device Manager显示、draft、template三种事实会分裂

位置：`zlc_workbench/device_manager.py:711-837`。

三个具体问题：

1. 老apparatus缺少新schema field时，`_show()`用`stored.get(field.key, field.default)`把default画到表单上，却不把default写回`DeviceInstanceConfig.parameters`。因此屏幕显示一份完整值，presenter draft仍缺字段；不编辑直接Save/Init的结果取决于下游补默认策略，而不是屏幕所见。
2. `_template_name()`只比较`(instance_id, type_id)`，完全忽略parameters。一个参数已修改的apparatus仍会被标为“Virtual template”等，属于false source projection。
3. `set_type()`按同名field搬值；若两个device type恰好有同名但不同语义/单位的字段，旧值会被带入新type。现在catalog未证明已发生，但这条policy没有schema compatibility依据。

已有`test_an_apparatus_saved_before_a_type_gained_a_field_still_opens`只证明“窗口能显示”，没有证明draft/save/init与显示一致。最小修复应在加载/选择type的唯一normalization点把schema defaults materialize到draft；template identity比较完整canonical config。

### UIWB-007（P0 UI responsiveness）— Generic device `tune()`直接跑在Qt owner thread

位置：`zlc_workbench/apps/task_console.py:322-385`。

generic control的commit slot在GUI线程：读取表单→取得exclusive lease→直接调用`tune()`→重新读取`tunable_fields()`。真实camera SDK setter/readback稍慢或挂住，整个GUI、Stop、close与状态paint同时冻结。这直接违背权威“阻塞camera/sequencer调用在worker/session侧执行”。

当前实现的float coercion与目前两个camera consumer吻合，但generic surface宣称接任意`tunable_fields()`，contract本身并未保证以后只有float。若保留generic control，应由device schema的typed projection传原值，并复用一个composition-owned有界worker；不要在每个device plugin再造Qt worker。

### UIWB-008（P1 state/archive）— Figure Viewer的第二parser已实测丢状态

位置：

- `zlc_workbench/viewer.py:100-138` `_panel_record()`/`_panel_state()`；
- `zlc_workbench/console_layout.py:342-399` `_panel_from_tree()`；
- `zlc_workbench/panel_state.py:207-311` `PanelState.document()`。

`PanelState.document()`写14项。layout parser读完14项；Viewer手工只读前10项。隔离round-trip probe以非默认值保存：

~~~text
published_outputs = {'roi_mean': True}
selector = {'kind': 'area'}
classifier_threshold = {'value': 2.5}
focused_cell = 3
~~~

Viewer恢复结果分别为`{}`, `{}`, `{}`, `None`。

这是一个确定的第二parser缺陷。`PanelState`应有唯一strict `from_document()`（或现有module-level唯一parser）；layout与Viewer都调用它。不要新增“viewer-compatible state”。

同一Viewer还有四条具体不一致：

- `_flow_rows()`遍历`sections['panel']`并假设它是`{panel_name: entry}`；formal Panel Save写`{'dataset':'data','state':...}`，所以Flow tab读不懂正式writer；
- `FigureViewerView`宣称可打开PNG/JPG/NPZ，presenter却一律`read_archive()`；普通图像必失败；
- `rename_figure()`只写`self.figure_title`，不影响window/card title、save路径或archive metadata，是无效果public feature；
- `open/show/configure/save/fit`在Qt signal路径同步`.result()`，大archive、fit或save会冻结窗口。

失败open还会更新path输入而撤下旧host，用户可能看到“新失败路径 + 无图”，没有明确保留上一份成功document。需要先在后台完整parse/prepare candidate，再在owner thread原子换入；仍用现有host/handle，不加viewer transaction类。

### UIWB-009（P1 persistence）— Layout升级策略把新增field变成旧文件全量失效

位置：`zlc_workbench/console_layout.py:84-132,143-280`。

`LayoutDocument.from_tree()`只接受exact `zlc.console-board/v7`与exact fields；`resolve_layout()`又要求saved `entry.values`包含当前descriptor的每个authoring field。于是一个node新增有default的field，所有旧layout都会报`missing authoring fields`，而不是用schema的canonical default补齐stopped draft。

这与Device Manager“新field仍能打开”的意图相反，也正是“每加logic node/field UI又坏一次”的持久化版本。需要用户裁决兼容策略：

1. 同format内允许只为**新增且有明确default**的field补值，同时严格拒绝未知/删除字段；或
2. 每次schema变化提升layout format并写一条显式migration；或
3. 明确宣布旧layout不兼容并在UI提供可操作说明。

当前行为实际选择3，却没有产品说明。

另有一处silent failure：构建output contract表时，`descriptor.outputs_for()`的`TypeError/ValueError`被吞掉并当`outputs=()`。这可能让source validation缺少本应存在的contract，直到Start才失败。应把保存draft不可解析明确记为layout issue，而不是假装node无outputs。

### UIWB-010（P1 SSOT）— Panel save、legacy session save与annotation形成平行保存链

位置：

- `zlc_workbench/panel_save.py:37-316`；
- `zlc_workbench/session.py:449-472`；
- `zlc_workbench/archive.py:39-103`；
- `zlc_workbench/plot_annotations.py:20-55`。

已确认：

- `archive.py`从`zlc_data.figure_archive` import `FIGURE_SCHEMA`后，立即在本模块重定义同一字符串；这是最直接的重复真相源，应只re-export imported owner constant。
- formal `save_panel_figure()`写PanelState、run chain、overlay和annotations；`ExperimentSession.save_figure()`只写可选pulse/panel mapping，没有run chain或actual device snapshots，主要由notebook/tests/legacy Viewer fixture使用。两者都叫“save figure”，承诺却不同。
- `save_panel_figure()`先写derived image，再写不可重算data archive。image成功、archive失败会留下看似完成但没有数据的孤儿；image失败则连data也不落盘。应先原子保存authoritative data，再生成derived image，并对pair的partial状态给出明确报告。
- `capture_run_chain._plain()`对未知object直接`str(value)`，把本应由owner提供portable record的问题静默降级成不稳定字符串。portable archive应拒绝未知类型或要求run_record owner先提供plain tree。
- `PanelPlotAnnotations.classifier_thresholds`与`PanelState.classifier_threshold`重叠，02已进一步证明classifier/fit state分叉。必须确定authoring state与materialized result各自唯一字段，不能两边都能驱动replay。

`zlc_durable.unique_path()`的check-then-use竞争已记为`INV-002`，本文不重复展开。

### UIWB-011（P1 architecture）— `PulseEditorPresenter`是真God object，且纯domain mutation放错层

位置：`zlc_workbench/pulse_editor.py:878-3318`。

该class约2,390行、约90个methods，同时负责：

- document/file baseline与dirty；
- PulseSequence结构编辑、period/target/binding/repeat mutation；
- board target adoption与lane mapping；
- dial/owned/injected sequencer lifecycle；
- compile/load/fire/safe/wait/poll；
- scan source exec/table/repeat/hold/step；
- preview host与save；
- view model projection和状态文本。

这不是“文件大”本身，而是authoring rule与device/composition lifecycle互相遮蔽。`replace_sequence()`、target/binding mutation等纯规则属于`zlc_pulse`现有authoring/model owner；Workbench应保留：把view intent投到domain API、借session sequencer/lease、安排preview和window lifecycle。

拆分时禁止新建Editor controller家族。最小顺序应是：

1. 先把已有纯mutation变成`zlc_pulse`中现有model附近的普通函数；
2. presenter继续直线调用这些函数；
3. 把standalone/bound close统一；
4. 再依据真实profile决定是否仅机械拆文件。

04a已经证明AppliedState sync忽略slot values/forever、device ABI与board clock未完整验证、delay/scan/FIFO等语义缺口；这些不能被95个Editor fake tests判绿覆盖。

### UIWB-012（P1 global state）— import Workbench会安装可被任意替换的全局panel-size callback

位置：

- `zlc_workbench/__init__.py:15-22`；
- `zlc_workbench/panel_sizes.py:22-44`；
- `zlc_ui/board/panel_geometry.py:38-83`。

任何`import zlc_workbench`都会执行`panel_sizes.install()`，把process-global `_measure`写入`zlc_ui`。`use_panel_display_sizes()`没有one-time guard、owner token或restore，后来的任意caller可静默替换全局card尺寸。import顺序因此能改变UI几何。

这个跨包注入需求本身合理，但应在正式app composition时显式安装一次；至少重复安装不同callable时拒绝。不能让“import composition package”暗含process-wide UI mutation。

### UIWB-013（P2/dead surface）— 测试和gallery正在保护无产品消费者的public seam

静态consumer清册确认：

- `zlc_ui.graph`整包（`FlowGraph*`、parser、view、shape helpers，约750行）没有product consumer；只有tests/gallery。Figure Viewer的Flow只是text rows，并未使用它。
- `QtOwnerWake`没有product consumer，只有tests/gallery；与此同时生产另写了一份较弱实现。
- `FluentFormGrid`只有gallery/test；没有产品consumer。
- `zlc_ui.acceptance`是manual acceptance/tool support，却从顶层stable facade公开`capture_window`。
- Fluent submodule至少20个export没有跨模块production consumer；其中若干在`fluent.py`内部使用，说明应private而非删除，public surface test不应冻结它们。
- `LogicCatalog`是单一production consumer的三方法wrapper（discover→dict→rows/get），大量测试直接实例化它；按“默认删/不默认抽象”，可直接内联到ConsolePresenter的catalog snapshot，而不是作为第二registry。
- `create_console_window()`、`ExperimentSession.save_figure()`、`session.camera/sequencer` convenience主要由tests/notebook使用，正式flow走更具体入口。

裁决：

- `zlc_ui.graph`删除或移到examples；不要因为已有public-surface test继续保留；
- acceptance放`tools`或明确developer-only，不属于stable runtime facade；
- Fluent exports逐个按真实consumer收窄；
- notebook convenience是否保留由用户裁决，但必须命名为convenience/legacy，不得与正式Save/设备选择同名承诺。

### UIWB-014（P1 contract/docs）— UI boundary charter与真实API互相矛盾

`zlc_ui/README.md`同时声称：

- public vocabulary不能有`Device`、`Pulse`、`Plot`等概念；
- public payload只允许plain/`QWidget`/自有VM，domain object不跨边界；
- 模块图又正式公开`device_manager`、`pulse`、console panel。

真实top-level facade含`open_device_manager`、`open_pulse_editor`；handles接收plot host并调用`qt_widget()`。这不是import leak，但显然是domain-named pure view和structural domain host跨边界。

权威架构表允许`zlc_ui`拥有window/tab/form/widget/operator intent，所以合理裁决不是把所有Pulse/Device view移回Workbench，而是把README改成真实规则：

> UI可拥有domain-named纯projection，但不拥有domain state/rules、不得import domain package；跨包surface只允许明确的plain VM和一个被书面定义的widget-host protocol。

是否接受host protocol，还是改成composition先取得`QWidget`并由自身保留host，需要用户裁决。当前两条理念——“outside不得碰widget”与“domain object不得进UI”——不能同时满足；必须公开选一条。

## 4. `zlc_ui`逐文件/符号裁决

以下`KEEP`只代表职责和层级成立，不代表每个视觉细节已做真机验收。

| 文件/符号 | 裁决 | 理由/动作 |
|---|---|---|
| `__init__.py::__getattr__/__dir__` | `KEEP` | lazy小facade，import不创建QApplication；但收窄tool/demo exports。 |
| `acceptance.py::AcceptanceCapture,capture_window,*pixmap helpers` | `MOVE TO TOOLS` | 有真实manual acceptance价值，无runtime consumer；不应是stable UI facade。 |
| `board/board_layout.py::BoardMetrics,GeomProxy,first_free_slot,min_board_width,board_width,pack,nearest_anchor` | `KEEP` | 纯几何、唯一owner，Workbench未复制。私有AABB/overlap helper层级正确。 |
| `board/panel_geometry.py::panel_size_cells,panel_display_size,use_panel_display_sizes` | `KEEP + EXPLICIT INSTALL` | seam合理；移除import side effect并拒绝不同callback覆盖。 |
| `concurrency/owner_wake.py::QtOwnerWake,error_summary` | `KEEP AND USE` | 当前正确实现未进入生产；替代Workbench复制品。 |
| `console/_panel_projection.py::*` | `KEEP WITH 02 CROSS-REF` | 是纯plain plot-form projection但包含plot vocabulary；与README“无Plot概念”冲突。semantic/display SSOT见02。 |
| `console/board_view.py::ConsoleBoardView` | `KEEP` | UI独占瞬时几何、drag/drop/order；职责正确。 |
| `console/handle.py::TaskConsoleHandle` | `KEEP + DEFINE HOST PORT` | seals widget tree，但对象payload/plot host seam必须写实；不继续扩展成第二presenter。 |
| `console/logic_editor_view.py::LogicEditorView` | `KEEP` | 纯projection；动态form正确性依赖UIWB-001。 |
| `console/logic_row_view.py::LogicRowView` | `KEEP` | 一行projection/intent，层级正确。 |
| `console/panel_card_view.py::PanelCardView` | `KEEP` | 交互/卡片widget owner；selector/scroll细节见02。 |
| `console/panel_editor_view.py::PanelEditorView` | `KEEP` | shared state projection；不得存第二PanelState。 |
| `console/signal_chooser.py::SignalChooser,choose_signal` | `KEEP` | 通用grouped choice modal；无runtime读取。 |
| `console/status_strip.py::StatusStrip` | `KEEP` | 小而单责。 |
| `console/task_console_view.py::TaskConsoleView` | `KEEP` | 窗口body组合属于UI；不应加入runtime/persistence。 |
| `device_manager/handle.py::DeviceControlHandle,DeviceManagerHandle` | `KEEP` | sealed window port；`cancel_requested`若产品已无Cancel需与权威UI核对后删除。 |
| `device_manager/view.py::_DeviceCard,_LiveDeviceCard,DeviceControlView,DeviceManagerView` | `KEEP` | widget ownership正确；依赖form reconcile/default truth修复。 |
| `figure_viewer/handle.py::FigureViewerHandle` | `KEEP + FIX FALSE FEATURES` | 删除/接通rename，定义host seam。 |
| `figure_viewer/view.py::FigureViewerView` | `KEEP + FIX FILTER` | 纯view；file filter必须只承诺NPZ或实现真正image reader。 |
| `fluent/choice_picker.py::*` | `KEEP` | typed choice encode/read是必要共享控件；typed equality应复用于enabled_when。 |
| `fluent/info_pane.py::InfoPane` | `KEEP` | 已参数化，无archive IO。 |
| `fluent/published_items.py::PublishedItemsLegend` | `KEEP/PRIVATE` | 当前consumer有限；若只在console内部使用，不需submodule public API。 |
| `fluent/style.py` | `KEEP` | 单一style token owner。 |
| `fluent/fluent.py` | `KEEP CONCEPTS / SPLIT MECHANICALLY` | 3,931行混scale、window registry、style、popup/dialog、控件、tabs、scroll、window launcher、metrics。概念大多有消费者，但文件已是maintenance God module；只做机械内聚拆分，不新增framework。`FluentFormGrid`删除。 |
| `form/form.py::parse_number_text,choice codecs,FormChoice,FormFieldProps,FormSpec` | `KEEP` | Qt-free严格form contract；职责正确。 |
| `form/qt_form.py::handlers,FluentParameterForm` | `KEEP + P0 FIX` | 唯一Qt projection；重建dependency graph、typed comparison、automatic bounds。 |
| `graph/flow_graph.py,flow_graph_view.py,shape_text.py` | `DELETE/MOVE DEMO` | 无产品consumer；public tests/gallery不是存在理由。若未来Flow tab真需要图，再由实际consumer证明。 |
| `pulse/_layout.py::*` | `KEEP PRIVATE` | view geometry helper。 |
| `pulse/models.py::*VM` | `KEEP` | frozen/plain，阻止PulseSequence进UI。 |
| `pulse/editor_view.py::PulseEditorView` | `KEEP` | shell只组合tabs/intents。 |
| `pulse/handle.py::PulseEditorHandle` | `KEEP + HOST CONTRACT` | 不应再增长domain逻辑；`show_preview(host)`边界要书面裁决。 |
| `pulse/preview_view.py::PulsePreviewView` | `KEEP` | mount point，不自绘第二renderer。 |
| `pulse/scan_line_edit.py::*` | `KEEP` | scan/API badge是projection；transition仍归zlc_pulse。 |
| `pulse/scan_view.py::PulseScanView` | `KEEP` | draft仲裁是UI输入保护，不是第二scan state。 |
| `pulse/schedule_view.py::PeriodCard,ChannelNamesPanel,ChannelPanel,RepeatBracket,PulseDragContainer,PulseScheduleView` | `KEEP` | 大但按同一schedule UI协作；不承载compile。可按subview机械拆文件，不改变VM contract。 |
| `pulse/target_view.py::*` | `KEEP` | pure target authoring projection；domain validation仍外置。 |
| `qt.py::ensure_qt_app,*bootstrap` | `KEEP + REMOVE SAFETY PRETENCE` | single QApplication/HiDPI/ipykernel seam成立；atexit raw delete不得替代composition shutdown。 |
| `windows.py::open_*` | `KEEP` | one-window entry正确；所有app必须通过handle close guard。 |

## 5. `zlc_workbench`逐文件/类/函数裁决

| 文件/符号 | 裁决 | 理由/动作 |
|---|---|---|
| `__init__.py` | `REMOVE IMPORT SIDE EFFECT` | version/path guard可留；panel size install移到app composition。 |
| `apps/device_manager.py::_parser,apparatus_path,build,create_window,main` | `KEEP + OWN WORKER` | composition正确；`--check`创建可见window handle且未完整close worker，和help“without opening”冲突。 |
| `apps/figure_viewer.py::*` | `KEEP + ASYNC CANDIDATE` | host construction在composition正确；`--check`同样实际open window。 |
| `apps/pulse_editor.py::resolve/build/create_window/create_bound_window/main` | `KEEP + UNIFY CLOSE` | standalone/bound安全等级必须一致；dial/default endpoint见04a。 |
| `apps/task_console.py::build_panel_host,build_console,ExperimentGuiFlow,create_*` | `KEEP + FIX GENERIC CONTROL` | 正式session/window owner；generic tune移出GUI线程，control close顺序保留。`create_console_window`若仅兼容测试/notebook应私有化。 |
| `archive.py::write_figure_file,write_figure` | `KEEP ONE WRITER` | 删除local `FIGURE_SCHEMA`复制；与formal Panel Save明确分工或删除legacy writer入口。 |
| `authoring_form.py::project_schema,project_logic_schema,project_artifact_inputs,_project_*,display_value` | `KEEP` | 正确composition adapter；schema→widget唯一翻译。 |
| `board.py::LiveBoard` | `KEEP` | runtime scheduler/arbiter接线合理。 |
| `board.py::OwnerWake` | `DELETE` | 重复且弱于`zlc_ui.QtOwnerWake`。 |
| `board.py::attach_qt` | `KEEP` | narrow QTimer shim。 |
| `board.py::attach_qt_worker` | `REPLACE RETURN CONTRACT` | 当前泄漏executor；现有owner应可close，不新增manager。 |
| `board.py::attach_qt_owner_turn` | `DELETE/MERGE` | 接入QtOwnerWake后不再保留第二relay语义。 |
| `console.py::ConsolePresenter,PanelBinding,_LayoutCandidate` | `KEEP WITH 02/03 FIXES` | composition核心，详细panel/task问题已在前报告；4,656行仍过大，应只按已证明owner切分，不做抽象式重写。 |
| `console_layout.py::LayoutDocument,LogicLayoutEntry,ResolvedLayout,resolve_layout,parsers` | `KEEP + ONE STATE CODEC + MIGRATION POLICY` | layout owner正确；PanelState parser重复、schema新增field不兼容、outputs error被吞。 |
| `device_manager.py::DeviceManagerPresenter` | `KEEP + FIX` | 单一apparatus/session lifecycle虽然851行仍内聚；修worker、default materialization、template identity。 |
| `device_use.py::DeviceClaim,DeviceLease,LogicReservation,DeviceUseCoordinator` | `KEEP` | 权威明确的session device ownership；不要另造editor-specific lease manager。 |
| `logic.py::LogicDraft,LogicDraftFinalization,LogicCandidate,LogicBinding` | `KEEP/PRUNE FIELDS` | 代表同一row的draft/finalized/pending/running状态，是真实lifecycle；长期应审查缓存字段是否都被消费，但不建立新DTO。 |
| `logic.py::stable_signal_key,dataset_inputs,artifact_input_specs,device_key_options,finalize_logic_draft,build_arguments,make_host` | `KEEP` | 基本composition与Start admission，层级正确。 |
| `logic.py::LogicCatalog` | `INLINE/DELETE` | 单consumer registry wrapper；直接持有discovered descriptor mapping即可。 |
| `panel_catalog.py::*` | `KEEP` | TaskConsole本地plot offering/labels；具体cell kind矛盾见02与权威文档。 |
| `panel_save.py::capture_run_chain,overlay_payload,save/restore*` | `KEEP + ONE SAVE CONTRACT` | formal Panel Save owner；strict plain serialization、data-first、pair partial报告。 |
| `panel_sizes.py::install,_display_size` | `KEEP FUNCTION / EXPLICIT CALL` | 跨包size projection合理；不要import时安装。 |
| `panel_spec.py::_dense_series_x,fitting_panel_spec` | `KEEP OR MOVE TO zlc_plot` | 目前是Workbench generic inference glue；若zlc_plot已有相同schema→spec入口则删除这里，02再裁决。 |
| `panel_state.py::*projection,PanelState,PanelFrozenData` | `KEEP + ADD UNIQUE DECODER` | PanelState是正确SSOT；不要让Viewer/Layout各自解析。fit/annotation重叠见02。 |
| `plot_annotations.py::PanelPlotAnnotations` | `MERGE/DELETE DUPLICATE STATE` | materialized threshold若确实不是authored threshold，应改名并只存result；否则并回PanelState。 |
| `presentation.py::PlotPanelPort` | `KEEP WITH 02` | 不重复审。 |
| `pulse_editor.py::projection/timeline helpers` | `KEEP, MOVE PURE RULES TO DOMAIN` | projection可留Workbench；sequence mutation/target/binding规则归zlc_pulse。 |
| `pulse_editor.py::BoardState,PulseEditorPresenter,replace_sequence` | `GOD OBJECT / CUT BY OWNER` | 先修close/truth，再把纯authoring mutation下沉；不要新增controller层。 |
| `pulse_state.py::PulseEditorState,state_from_tree/state_to_tree/read/write` | `KEEP` | Workbench只拥有Editor扩展段，复用zlc_pulse sequence codec；不是完整第二parser。需核对core codec对unknown top-level字段的policy。 |
| `selection.py` | `KEEP WITH 02` | 不重复审。 |
| `session.py::Workspace,seed_packaged_pulses` | `KEEP + FIX INSTALLED DEFAULT` | workspace owner合理；`Path(__file__).parents[4]/workspace`只在checkout成立，wheel/site-packages下可能落到错误/不可写位置。默认必须是显式env/user data或launcher注入。 |
| `session.py::ExperimentSession` | `KEEP FACADE + PRUNE LEGACY` | 同一Notebook/GUI session正确；`camera/sequencer`硬编码default conveniences、legacy save需标明。`fire(shots<=0)`应拒绝；`close()`在installation close抛错时不会关闭signal plane，需汇总清理失败。 |
| `tools/capture_acceptance.py` | `KEEP AS DEV TOOL` | 真实屏幕验收入口，不是runtime API。 |
| `tools/check_environment.py` | `KEEP/RENAME DEV CHECK` | 能发现shadow editable install；硬编码repo layout，不能被描述为installed package健康检查。 |
| `topology.py::SignalRow,project_signals,format_signal_shape` | `KEEP` | plane→plain UI projection，小而单责；docstring不应把generic tensor固定称R×P。 |
| `viewer.py::describe_archive,_*rows,FigureViewerPresenter` | `KEEP + DELETE PARALLEL PARSER` | archive projection owner成立；用唯一PanelState codec、formal section shape和异步candidate。 |

## 6. Tests逐文件裁决与证据边界

### 6.1 本阶段实际执行

所有Python验证进程先import并打印了本树`zou_lab_control_v2`与被测package的`__file__`。窄测试结果：

| 组 | 结果 |
|---|---:|
| Workbench Device Manager / Viewer / Archive / Environment / Windows | 56 passed |
| `zlc_ui` controls / settings / facade / import / figure / device view | 27 passed |
| Workbench Pulse Editor | 95 passed |
| 合计 | **178 passed** |

这些结果说明现有测试对当前实现是稳定的；下列隔离probe仍失败，说明测试没有证明跨路径contract。

### 6.2 `zlc_ui` tests

| 测试文件 | 裁决 |
|---|---|
| `conftest.py` | `KEEP`；Qt/offscreen setup。 |
| `test_console_extension_cost.py` | `KEEP`；只证明synthetic widget可挂载，不证明host boundary正确。 |
| `test_console_views.py` | `KEEP HIGH VALUE`；大量真实QTest/幂等/drag/close；需增加form dependency repro而不是继续扩大控件矩阵。 |
| `test_controls_smoke.py` | `PRUNE DEMO-ONLY ASSERTIONS`；当前保护FlowGraph和未消费QtOwnerWake，反而掩盖production duplicate。 |
| `test_device_manager_view.py` | `KEEP`；view intent有效，但不覆盖presenter default SSOT/worker shutdown。 |
| `test_figure_viewer.py` | `KEEP`；只审view mount，不证明PNG filter或archive round-trip。 |
| `test_gallery.py` | `KEEP AS DEV SMOKE`；不能作为production consumer证明。 |
| `test_import_purity.py` | `KEEP`；import边界真实有效，但只检查module import，不检查domain object穿过handle。 |
| `test_modal_repaints_what_is_there.py` | `KEEP`；具体Qt退休回归。 |
| `test_overloaded_signals.py` | `KEEP`；机械guard，有真实PyQt崩溃价值。 |
| `test_panel_card_plot_interaction.py` | `KEEP WITH 02`；真实交互价值高。 |
| `test_public_surface.py` | `REDUCE`；顶层facade allow-list有价值，但显式保护`zlc_ui.graph`与tool-only API不合理。 |
| `test_pulse_views.py` | `KEEP HIGH VALUE`；VM stale/reuse/drag意图，不证明presenter/device semantics。 |
| `test_qt_app_single_entry.py` | `KEEP + ADD COMPOSITION SAFETY`；当前把raw atexit deletion当成功，需区分GUI teardown与hardware shutdown。 |
| `test_settings_layout.py` | `KEEP`；真实form reflow，但漏掉`enabled_when` graph变化。 |

### 6.3 `zlc_workbench` tests

| 测试文件/组 | 裁决 |
|---|---|
| `pulse_fixtures.py` | `KEEP TEST SUPPORT`；不要演变成第二serializer。 |
| `test_archive.py` | `KEEP`；证明单NPZ原子与portable arrays，不证明image+archive pair。 |
| `test_auto_panel_kind.py`,`test_panel_spec.py` | `KEEP WITH 02`；plot policy待权威矛盾裁决。 |
| `test_console_logic.py` | `KEEP HIGH VALUE`；Start/claims/drafts强；大量直接`LogicCatalog`使用正在测试wrapper而非必要产品seam。 |
| `test_console_presenter.py` | `KEEP WITH 02/03`；覆盖面广但主要fake view/host，不替代human GUI/性能。 |
| `test_device_manager.py` | `KEEP + ADD TRUTH ASSERTIONS`；新增field test只看open，不看draft/save/init；没有hung worker shutdown。 |
| `test_editor_named_behaviours.py` | `KEEP`；补充命名行为，不是pulse wire proof。 |
| `test_end_to_end.py` | `KEEP`；多处使用legacy`session.save_figure`，因此会继续保护第二save contract。 |
| `test_environment.py` | `KEEP DEV`；证明本checkout editable paths，不证明wheel install。 |
| `test_guard_a_virtual_chain.py` | `KEEP`；纵向headless价值高；virtual≠hardware见04b。 |
| `test_guard_b_task_console_interaction.py` | `KEEP`；真实主链价值高。 |
| `test_guard_c_save_semantics.py` | `KEEP + EXTEND ROUND-TRIP`；要通过formal Viewer读回全部PanelState，而非只检查文件区分。 |
| `test_gui_seam.py` | `KEEP BUT RENAME CLAIM`；证明Workbench不import子module/QtWidget；没有证明host object不穿边界。 |
| `test_launcher_imports.py`,`test_launchers.py` | `KEEP`；只守module routing/错误输出/CRLF；不守argv fidelity、`--check`无窗口、cleanup。 |
| `test_notebook.py` | `KEEP DOC SMOKE`；目前保护legacy conveniences，需在用户裁决后同步。 |
| `test_panel_front_coherence.py`,`test_presentation.py`,`test_same_shot_presentation.py`,`test_selection.py` | `KEEP WITH 02/04b`；分别只证明publication/display cohort，不证明physical shot。 |
| `test_pulse_editor.py` | `KEEP BUT DECOUPLE`；95项细行为很强，却广泛fake私有presenter状态；没测standalone real close guard、owned close failure、04a wire缺口。 |
| `test_task_console_app.py` | `KEEP`；正式shared session/control reopen价值高；仍有内部presenter调用替代全部human buttons。 |
| `test_topology.py` | `KEEP`；小而正确。 |
| `test_view_contracts.py` | `KEEP`；fake/real surface同步有用；不能冻结错误contract。 |
| `test_viewer.py` | `KEEP + REPLACE LEGACY FIXTURE`；当前主要fixture由`ExperimentSession.save_figure`生成，故漏掉formal Panel Save的Flow/parser结构。 |
| `test_windows.py` | `KEEP + ADD STANDALONE PULSE CLOSE FAILURE`；目前只对console guard与部分sealed handles做证明。 |

### 6.4 必须新增的最少纵向证明（未来实施阶段）

不建议新增字段级GUI矩阵，只需要以下几条：

1. `FluentParameterForm` reconcile时controller从A换成C，立刻按C重算且A不再影响；
2. formal Panel Save → Viewer →完整`PanelState.document()`等价；
3. Device Manager旧config缺default field → UI/draft/save/init四者同值；
4. 一个永不返回的discover/init fake → window close有界、无残留executor；
5. standalone Pulse drive safe失败 → window仍在、lease/错误可见；
6. formal app close → control windows→nodes/plots→session/devices→owner顺序，最后无线程/claim；
7. launcher`--check`结束后top-level windows/executors为零，argv含空格/`!`的路径不被改写。

## 7. Docs、metadata、launcher审查

### 7.1 明确错误/过期

| 文件 | 事实 |
|---|---|
| `packages/zlc_ui/pyproject.toml` | 声明Python `>=3.9`，source使用`@dataclass(slots=True)`（Python 3.10引入）；metadata不成立。应至少`>=3.10`，若monorepo统一则`>=3.11`。 |
| `zlc_ui/docs/loc-report.md` | 记录39 files/8,659 lines；当前49/16,765，且仍把历史迁移LOC当现状。 |
| `zlc_ui/docs/goal-archive.md` | 首屏说“活的计划在GOAL.md”；当前GOAL是明确inactive tombstone。 |
| `zlc_ui/docs/console-views.md` | TaskConsole/DeviceManager signals与setters已明显少于真实handle，payload/signature漂移。 |
| `zlc_ui/docs/pulse-views.md` | 仍是旧view survey contract，部分方法/signals和当前handle/presenter不同；只能作为历史。 |
| `zlc_ui/README.md` | “无Device/Pulse/Plot public vocabulary”“domain object不跨边界”与公开handles/host seam冲突。 |
| `zlc_ui/docs/survey-workbench-ui-2026-08-02.md` 与 `zlc_workbench/docs/survey-workbench-2026-08-02.md` | SHA256完全相同：`73871D...8148`，是跨package重复文档。保留一份历史索引或全部移入AUDIT archive，不应双份维护。 |
| 四个app `--check` help/comments | 声称“不打开/不显示window”，实际调用`open_*`创建并show正式window handle；部分路径只close presenter，不明确close handle/worker。 |

七份GOAL中的两个本包GOAL当前tombstone是正确的，不应再扩写；错误在旧archive doc仍把它指为live。

### 7.2 Launcher

`bin/_launch.bat`正确做到：

- 从root bootstrap本checkout；
- 统一解释器resolver；
- 保留错误输出；
- repo用`.gitattributes`守CRLF。

但它把argv收集成一个delayed-expansion字符串`FORWARD`再二次解析。含`!`的合法路径会被delayed expansion改写，empty/复杂quoted argument也不能保证逐argv保真。现有tests只检查CRLF、module routing和不吞stderr，没有行为probe。应以真实batch probe证明后再改，不能只看字符串猜Windows quoting。

`tools/check_environment.py`把`WORKSPACE = Path(__file__).parents[4]`作为expected source root，适合开发checkout防shadow import；它不适合installed wheel。文档应称“checkout integrity check”，而不是一般package health。

## 8. 建议的最小收敛顺序

这不是实施授权，只是避免未来修复继续长出新层的顺序：

1. **先修确定性state bug：** Form dependency reconcile；PanelState唯一decoder；Device Manager default materialization/template identity。
2. **再修硬件/线程close：** 用现有QtOwnerWake替代复制品；composition持有worker shutdown；standalone Pulse close guard；generic tune异步；正式app有界shutdown。
3. **收敛保存真相：** formal Panel Save为唯一产品Figure contract；Viewer只读同一section；data first；legacy session save重命名/删除；annotation state归一。
4. **再移动纯domain规则：** Pulse sequence mutations回`zlc_pulse`现有owner；Workbench presenter只留接线/device/preview。
5. **最后做机械清理：** 删除Graph/demo-only surface、内联LogicCatalog、收窄Fluent exports、拆Fluent/Pulse文件、同步metadata/docs。

不要反序：若先拆大文件或建新controller/codec/worker manager，只会把当前重复真相固化成更多文件。

## 9. 需要用户裁决

以下不是用“设计问题”掩盖明确bug；前述P0应修。这里是真实产品选择：

### D06B-001 — UI/plot host边界到底选哪条铁律

选项：

1. `zlc_ui`接受一个明确、窄、structural host protocol（`qt_widget/logical_size/wheel_target`），承认domain host会进入UI但UI不import其package；
2. Workbench先取`QWidget`传给UI，同时WorkBench保留host生命周期，放弃“outside不碰widget”的绝对规则。

当前代码选1，README同时宣称1和2都禁止。审计倾向1，因为它已避免Workbench持有widget且消费者有三个；但必须书面化并禁止继续扩张host API。

### D06B-002 — Saved layout向前兼容策略

选项：补有default的新字段、显式version migration、或明确不兼容。审计倾向“同format只补新增default；结构变化升format并写直接migration”，但用户需决定是否需要打开历史layout。

### D06B-003 — Notebook convenience是否是产品public API

`ExperimentSession.camera/sequencer/save_figure`和`create_console_window`当前主要支撑tests/notebook。选项：

1. 保留并明确“default-device/legacy notebook convenience”，不承诺多device与formal Panel Save；
2. 删除，Notebook也使用named installation capability和formal save API。

审计倾向2，避免两套产品路径；若实验现场频繁用Notebook，选1但必须改名。

### D06B-004 — Interpreter退出时硬件策略

正常窗口close顺序可以确定；Notebook kernel exit/atexit时是否：

1. 尝试composition close/safe并有界等待，失败留下强诊断；
2. 只销毁UI，不对硬件做任何隐式command，并明确设备状态未知；
3. 由每个实验入口注册实验室特定emergency policy。

当前`sip.delete all`等于只做UI销毁却没有诚实标注。SLM“close保持phase”和sequencer“close前safe”物理语义不同，不能用一个通用zero/safe猜测。

### D06B-005 — Figure Viewer是否支持普通image

若Viewer目标是archive provenance浏览器，file picker应只给NPZ；若也要看PNG/JPG，需定义普通image没有dataset/flow/device/fit时的明确降级UI。当前文案承诺、实现拒绝。

### D06B-006 — Developer UI组件是否继续随runtime发布

Graph、gallery、acceptance capture、FluentFormGrid是否作为独立UI toolkit产品保留，还是只服务ZLC实验产品。按当前用户“默认删”，审计倾向：Graph/FormGrid删除，acceptance移tools，gallery只留最小manual smoke。

## 10. 最终判定

`zlc_ui`的package import纯度和view ownership是这轮更新里少数真正守住的边界，应该保护；它的问题主要是contract文字不诚实、一个关键reconcile bug、未消费的正确并发primitive和demo历史表面。

`zlc_workbench`仍然履行composition root，但已经在四处越界：

- 用自己的并发/lifecycle替代UI已有primitive；
- 在Pulse Editor拥有纯pulse authoring规则；
- 在Viewer/Save拥有第二parser与第二保存语义；
- 用全局import side effect和无owner executors隐藏composition资源。

因此本阶段不建议“重写UI”。建议先以六条纵向失败证明锁住state、save、close，然后沿已有owner删掉重复实现。做到以下条件前，不能宣布UI/workbench层过关：

1. 任意logic schema reconcile后condition/default/view/draft一致；
2. formal Panel Save经Viewer完整round-trip不丢一项PanelState；
3. Device Manager/Pulse/TaskConsole所有worker、lease、device在close前有界退役；
4. production只剩一套owner wake、一套PanelState decoder、一套产品Figure save语义；
5. Pulse纯mutation不再由Workbench自创规则；
6. docs/metadata只描述当前代码，历史survey不再作为活contract。
