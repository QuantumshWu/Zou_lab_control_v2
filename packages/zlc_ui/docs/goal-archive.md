# GOAL 归档 — zlc_ui 已完成轮次

> 已完成并验收的轮次原文(R / B / PE / FV),留作证据与追溯。**活的计划在 `GOAL.md`**。

## R 阶段:验收修复(先于 PE,顺序执行)

### R1 提交现有未提交工作(立即,按主题拆 3 个 commit)

- [x] R1.1 `fix: flow graph arrow heads crash (staticmethod self)` = `graph/flow_graph_view.py` + `tests/test_controls_smoke.py` 新 spy 测试(该测试在 HEAD 上必红,自证非空洞)。
- [x] R1.2 `fix: panel card settings popup uses shared FluentPopup anchor` = `console/panel_card_view.py` 弹窗部分。
- [x] R1.3 `fix: bound ipykernel qt-loop stalls` = `qt.py` 的 `_install_ipykernel_wake_timer` + `notebooks/usage.ipynb`。**提交前 strip notebook 全部 outputs**(工作区版 cell 8 内嵌 181KB base64 PNG,违反 HEAD 的 0-output 惯例)。
- [x] R1.4 `docs: pulse editor view surveys` = `docs/survey-pulse-editor-*.md` 两个文件单独一个 commit(它们是 PE 阶段的参考资料,有意入库)。

### R2 违规修复(必须改)

- [x] R2.1 **`TaskConsoleView.set_logic_rows` 幂等化**(`console/task_console_view.py:101-115`):现实现对布局内现有行 `deleteLater()` 后把同一批行加回——同批重设时排队的 DeferredDelete 会销毁在用行("wrapped C/C++ object deleted"),且 `addStretch(1)` 逐次累积。改成 `DeviceManagerView.set_devices`(`device_manager/view.py:139-159`)同款保留/调和写法:不在 incoming 的才卸载(`setParent(None)`,绝不 delete presenter 拥有的行),stretch 只加一次。补一条"同批行重调 set_logic_rows 两次 + processEvents 后行仍可用"的回归测试。
- [x] R2.2 **`PanelCardView.dropped` 载荷去 QPoint**(`console/panel_card_view.py:42`):`pyqtSignal(QtCore.QPoint)` 超出允许类型表;改为 `pyqtSignal(tuple)` 发 `(x:int, y:int)`。同步 `console/board_view.py:61`、`docs/console-views.md:17`、demo 与测试。
- [x] R2.3 **`FigureInfoPane` 去领域化**(`fluent/figure_info_pane.py:67-91,118-121`):"Plot/Measurement/Device" 三页签与 "saved Figure (.npz)" 文案是宪章禁词。改造成通用 `InfoPane`:页签标题与行内容全部构造注入(`tabs: tuple[tuple[str, rows], ...]`),包内零领域字面量;或整件移出包(去向记 README)。选改造(保住现成消费者)。
- [x] R2.4 **signal_picker 族词汇中性化**:`fluent/signal_picker.py`(producer/formats/ready-waiting-unbound 是 SignalHub 领域模型词汇)、`FormFieldKind` 的 `"signal"` 种类、`FormRuntimeContext.signal_*` 四回调名、`PublishedSignalsLegend`、`FluentTreeComboBox.signalPicked`。统一改中性词(建议 `source`/`keyed choice`;状态词参数化注入),docstring 里 "PulseScan y / hub / plot panel" 全清。这是机械改名+参数化,不改行为;consumers(console 视图、examples、tests、docs)同一切片跟改。
- [x] R2.5 **移植 v1 `75059ca` 的 `drop_index(raw_position=...)` 修复**到 `board/board_layout.py:144-156`(防 Qt 拖拽瞬态像素位置泄入持久语义布局记录);移植时按铁律去掉 v1 版的 isinstance 仪式块;补对应测试。(唯一 owner=zlc_ui 已由用户确认;v1 侧不归本仓管。)

### R3 API 门面与命名收敛(小修)

- [x] R3.1 门面完整性:公开 `zlc_ui.fluent.style`(或在 `fluent/__init__` 导出 ACCENT/GREEN/GREY/ORANGE/RED 等 token);console 视图与 examples 全部改走门面 import,消灭对 `zlc_ui.fluent.fluent` 的深 import;`concurrency/__init__` 补导出 `wait_for_owner_retirement`/`error_summary`(或从模块 `__all__` 删掉);停止转出 `FramelessWindow`/`StandardTitleBar`;`FORM_WIDGET_HANDLERS` 注册表要么公开成真实扩展点(附注册测试),要么停止导出 `FormWidgetHandler`。
- [x] R3.2 信号命名:`DeviceManagerView.parameter_changed`→`parameter_committed`,且 text 字段从 `textChanged`(每击键一发)改接 `editingFinished`(`form/qt_form.py:166`);`pathCommitted`/`signalPicked` 改 snake_case;README 宪章补记四缀。(`ConsoleBoardView.order_changed` 不在此改名——B1 会把它整体替换为 `order_committed(tuple)`,别做两遍。)
- [x] R3.3 杂项:删未用 import(`console/panel_card_view.py:16,17,24`);`set_cards` 注解改 `tuple[PanelCardView,...]`;`add_panel_requested`/`add_logic_requested` 的写死常量载荷改无参信号;`FlowGraphNode.has_devices` 与 `flow_graph_view.py:48-56` 写死的 measurement/device/processor/plot 调色板改为注入式 role→style 映射;`.gitattributes` 定死 LF(现 CRLF/LF 混杂,对照 v1 diff 永远带噪);控件层 docstring 领域词清扫(`fluent.py:78,157,185,223,819,1159,2361,2826,3162`、`form/qt_form.py:268`、`graph/flow_graph.py:43`);README:39-40 "每个 demo 都有窗内 log" 与 gallery 实况不符,二选一修齐。

### R4 验收工件与测试补强

- [x] R4.1 **三档 DPR 工件改真渲染**:`gallery.py:346-358` 现在是 1.0 DPR grab 后插值放大(高 DPR 下的字体渲染与 scaled_px 布局完全没被执行,违背"三档 DPR 必截真窗"铁律)。改为三次子进程各设 `QT_SCALE_FACTOR=1/1.5/2` 真渲染截图;notebook 同法;`test_gallery.py` 断言不变。
- [x] R4.2 demo_console 冒烟测试:子进程 offscreen 跑 `examples/demo_console.py --once`,断言 exit 0 且 stdout 含信号回显(现在 demo import 崩了 extension-cost 测试照样绿)。
- [x] R4.3 extension-cost 证明升级:demo 里至少一张 synthetic 卡走 `PanelCardView.set_surface(SyntheticCardView(...))` 进 console 卡片体系(现在挂在独立 tab 的裸 QHBoxLayout,证明的只是"能加任意 QWidget")。
- [x] R4.4 (可选)`_install_ipykernel_wake_timer` 机械测试:SimpleNamespace 假 kernel,断言 fake loop 的 quit 被周期调用。

---

## B 阶段:board 拖拽排版自包含(用户点名重要;R 全绿后、PE 前)

> 现状:重力 packer 纯几何(`zlc_ui.board`:pack/drop_index/BoardMetrics,"左上角重力、顺序即真理")已在包内,但 `ConsoleBoardView` 只发 `dropped` 意图、`arrange()` 等外部喂几何——"拖卡→实时重排→松手定序"的完整体验要靠使用方自己接线,不开箱即用。裁决:排版是纯几何零领域知识,**整条拖拽重排链下沉进视图**,布局几何的唯一权威=视图内 packer。参照重建源(只读):v1 `zlc_workbench/task_console/window.py:2001-2063`(arrange/drop 处理)与 `panel_board.py`(语义尺寸→像素换算);注意 v1 记忆里的坑:新增卡用 first-free-slot、拖放落点用 gravity 槽位,两职责别混。
>
> 契约:**顺序即真理**——presenter 只持久化/恢复 `panel_id` 顺序(和每卡尺寸),几何永远由视图内 packer 现算;`arrange()` 公共方法删除(保留即双权威)。

- [x] B1 `ConsoleBoardView` 自排版:注入 `BoardMetrics`;`set_cards` 后自动 pack+摆放;宿主 resize 触发重排(board 宽→列数,含 `min_board_width`);拖拽中 ghost + 落点实时重排预览;松手 `drop_index`(带 raw_position 语义)→ 新顺序 → re-pack → 出向 `order_committed(tuple[str, ...])`(替代 `reorder_requested` 的空意图);卡尺寸变化(size_picked 被 presenter 应用后)自动重排。
- [x] B2 测试 + demo:QTest 真拖拽(press-move-release)断言重排后几何等于 pack 纯函数结果、`order_committed` payload 正确;同批 `set_cards` 幂等(不重建卡);resize 重排测试;`demo_console.py` 里卡片可亲手拖动重排(人审验收点);**gallery 也挂一节可拖拽 board**(假卡若干,信号回显,随三档 DPR 截图一起进 artifacts/);README/`docs/console-views.md` 同步。

## PE 阶段:pulse_editor 纯视图(B 全绿后开工)

> 先读两份随仓 survey(`docs/survey-pulse-editor-schedule-window-2026-08-03.md`、`docs/survey-pulse-editor-scan-target-preview-2026-08-03.md`)——里面有逐文件纯度判定、全部操作者意图清单、视图模型 record 草案、与 controller 调用面的对账表。本阶段**只做纯 UI**:接口定型 + 假数据 demo 供用户验收;presenter/controller/PulseDocument 接线是用户验收接口之后的另一个 cut,不在本 goal。
> 参照重建源(只读):`..\Zou_lab_control_v1_claude\Zou_lab_control_v1\zlc_workbench\pulse_editor`。铁律:视图零 `zlc_pulse`/`zlc_plot` import;领域常量(TIME_UNIT_CHOICES、DEFAULT/MIN_REPEAT_COUNT、宽度规则、size 名单)一律经 VM/注入进来;`PulseFieldRef` 之类领域对象一律换 plain 载荷。

- [x] PE1 **接口先行**:新建 `docs/pulse-views.md`,把全部视图类签名写全再实现——视图模型 record(`FieldVM/PortRowVM/PeriodVM/RepeatVM/DelayRowVM/ScheduleVM/ScanPageRecord/TargetPortRecord/TargetWidthRule`,字段按 survey §3 草案)+ 每个视图的出向信号(survey §2 意图全清单:编辑类 `document_name/port_label/period_name/duration/digital/analog/delay_committed`、`binding_cycle_requested(field_kind:str, period_id:str|None, port:str|None)`、`repeat_committed`、`visible_ports_committed`;结构类 `insert/move/remove_period_requested`、`clear_port/clear_all_requested`;运行/文件/连接类)+ 入向窄 `set_*`(替代现 `apply_*` 家族:`set_schedule(vm)`、`accept_local_commit(generation,revision)`、`set_period/set_delay_row/set_port_label/set_visible_ports/set_summary/set_scan_source/set_scan_busy/set_connection/set_control_state`、`set_capabilities(can_sync,can_hold,can_step)`)。
- [x] PE2 建 `zlc_ui.pulse` 子包 + 原样搬纯件:`scan_line_edit.py`(192 行,dot 三态徽章,已零领域)、`_layout.py` 几何 token(rebase 到 zlc_ui.fluent 的 scaled_px);VM record 全部落为 frozen dataclass。
- [x] PE3 **schedule 视图重建**(最大件,参照 schedule_view.py 2,892 行):`PeriodCard`(duration/单位/名称 + digital 复选行 + DAC 模式/数值行,FieldVM 驱动含 sN/aN 徽章与只读锁)、`ChannelNamesPanel`(端口目录+可编辑显示名)、`ChannelPanel`(delay 行+单位+清除,DelayRowVM 的 unit_quantums 本地重配 validator)、`RepeatBracket`、`PulseDragContainer`(拖拽 ghost/插入指示/自动滚动/提案式 emit,已是纯 Qt 直接平移)、`PulseScheduleView`(set_schedule 吃 ScheduleVM;保留 (generation,revision) 陈旧拒收协议、选中/gap 高亮、左栏折叠、隐藏端口 combo 幂等重建、双 scroll 联动)。survey §1.1-A 那层投影函数**一行不进包**(它们的产物已烘焙进 VM)。
- [x] PE4 scan 视图:`PulseScanView` 近乎平移(信号已全 plain);剥掉模块级 `format_scan_progress/format_held_scan_point`(presenter 层,不进包);`DEFAULT/MIN_SCAN_SWEEP_COUNT` 改 `set_repeats_range(minimum,default)`;**三路草稿仲裁接口原样保留**(`set_scan_code`/`replace_scan_draft`/`acknowledge_scan_draft` + `code_dirty`/`source_revision`——这是"回投喂不踩正在输入的编辑器"的机制,接口一动手感就没了)。
- [x] PE5 target 视图:改 record 注入(`set_ports(records, editable, status_text)` + `set_width_rules`),`applyRequested` 载荷从 `PulseTargetManifest` 改 `tuple[TargetPortRecord,...]`(域构造/校验留 presenter,拒绝经 `set_feedback(str)` 回注);保留 view 内合法逻辑(key 自增分配、endpoint 数≠width 文本预检、宽度变化增删 endpoint 行);endpoint 占位命名模板作为 str 参数注入。`lane_order` 作为不透明 tuple 随行携带原样回传。
- [x] PE6 preview 视图:平移 `preview_view.py`(控制条+滚动挂载区),**保持 QWidget 挂载点、不自绘**(时间轴渲染主权在 zlc_plot,自绘=第二渲染链);唯一改动:`zlc_plot DEFAULTS` import 改 `set_size_names(tuple[str,...])` 注入。
- [x] PE7 编辑器壳 `PulseEditorView`:header(状态点+标题+summary+Clear All,状态灯只做"语义态字符串→颜色"映射,态由 presenter 算)+ 四页签组装 + 底部 Control/Connection/Ports 三组按钮条;对话框以命令式方法暴露(`ask_open_path/ask_save_path(caption,start_dir,filter)->str`、`confirm(title,text,ok,cancel)->bool`、`show_warning(text)`,plain 进 plain 出,供 Qt-free presenter 调用);close-guard 走 `close_requested` 信号 + `finish_close()`。窗口壳不知道 controller 存在。
- [x] PE8 **人审验收包** `examples/demo_pulse_editor.py`:手写假 `ScheduleVM`(≥3 period、一个 repeat bracket、scan/API 徽章各一、delay 行、隐藏端口)+ 假 ScanPageRecord + 假 TargetPortRecord + preview 占位 QWidget;全部出向信号窗内 log + stdout 回显;拖拽重排、dot 点击、草稿仲裁都能亲手玩;offscreen `--once` 退出码 0;三档 `QT_SCALE_FACTOR` 真渲染截图进 artifacts/。
- [x] PE9 测试:每视图"构造+set_* 对象级断言"+"QTest 驱动→信号 payload 断言";专项:ScheduleVM 陈旧拒收(旧 (generation,revision) 必拒)、同 VM 重复 set_schedule 幂等(不重建 widget,复用 `_reconcile_widget_order` 语义)、拖放 emit `move_period_requested` 且不本地应用、草稿仲裁三入口协议、demo_pulse_editor 子进程冒烟。
- [x] PE10 文档收尾:README 模块地图补 `zlc_ui.pulse`;`docs/pulse-views.md` 与实现零漂移;LOC 报告(survey 预估纯视图 ~4.6-4.9k 行,超出要逐项说明)。

## FV 阶段:figure_viewer 纯壳视图(小而独立;默认排在 PE 后,用户要求可提前)

> 首轮 GOAL 漏排了它,2026-08-03 用户指出后补入。v1 构成与归属裁决(参照源只读:`..\Zou_lab_control_v1_claude\Zou_lab_control_v1\zlc_workbench\figure_viewer` 357+120+49 行、`data_figure` 505+258+97 行):
> - **进 zlc_ui**:FigureViewer 的**纯壳**——文件打开/浏览意图、info 展示(用 R2.3 通用化后的 InfoPane)、图面挂载点、pane 退休交接。
> - **不进 zlc_ui**:`archive_io.py`(npz 读写=zlc_data/storage 职责)、`info_projection.py`(archive→tuple 的纯投影,presenter 侧)、**`DataFigureWindow` 整个**(它是 RasterPlotHost/Qt5PlotWidget/PlotSpec/FitResult 的组合件,吃 zlc_plot 类型,归 zlc_plot 侧/组合层;其 `embedded: bool` 旗标是宪章禁的模式旗标,留给那边重建时消灭)。README "不搬清单" 补记这两条去向。

- [x] FV1 `zlc_ui.figure_viewer.FigureViewerView`(参照 figure_viewer/window.py 重建,~250-350 行):出向 `open_path_requested()`(触发 `ask_open_path`)、`path_committed(str)`、`close_requested()`;入向 `set_info(tabs)`(通用 InfoPane 数据)、`set_figure_surface(QWidget|None)`(挂载 DataFigure 面,保留现 window.py 的 pane 退休集合语义:旧面 `setParent(None)` 待新面就绪原子换,不闪白)、`set_status(str, error: bool)`、`set_title(str)`。加载/投影/future 编排全留 presenter(现 `_load_and_project`+`submit_compute`+`_loadFinished` 链是 presenter 骨架,不进包)。
- [x] FV2 demo + 测试:`examples/demo_figure_viewer.py` 假 info tabs + 彩色占位 QWidget 当图面,信号回显,offscreen `--once` 退出码 0;测试=构造 + set_* 幂等 + QTest 打开按钮→`open_path_requested` 断言;gallery 挂载;README 模块地图更新。

## 机械终态判据(全绿才 GOAL COMPLETE)

1. `pytest -q` 全绿(含 R/PE/FV 全部新测试);干净 venv 判据复跑通过。
2. grep 为零(src/):`QtCore.QPoint` 作为信号载荷、`zlc_pulse`、`zlc_plot`、`PulseFieldRef`、`FramelessWindow`(门面导出处)、`textChanged` 直连出向信号;examples/console 无 `zlc_ui.fluent.fluent` 深 import。
3. `demo_console.py --once`、`demo_pulse_editor.py --once`、`demo_figure_viewer.py --once` offscreen 退出码 0 且 stdout 有信号回显;artifacts/ 有三档 **真渲染** DPR 截图(gallery + pulse editor)。
4. `set_logic_rows` 同批重设回归测试存在且绿;`drop_index(raw_position)` 测试存在且绿;board 真拖拽重排测试(几何==pack 结果 + `order_committed` payload)存在且绿,demo_console 可亲手拖卡。
5. README/docs(含 console-views.md、pulse-views.md)与实现零漂移;工作树干净(全部按主题提交)。
