# 任务5 报告:zlc_workbench / zlc_frontend 深读 —— "纯 UI 模块"拆分依据

**范围**:迁移分支 `Zou_lab_control_v1/zlc_workbench`(19,497 行)与 `zlc_frontend`(8,227 行),参考 `task_console.py` 启动脚本与 `zlc_plot`/`zlc_neutral_atom` 的边界。全程只读。

**先说总判断**:这两个包已经不是"vibe-coding 泥球",迁移分支已经做了大量刻意分层(headless 值/codec 与 Qt 分离、Qt-free controller、惰性导出、纯几何 packer)。真正的问题集中在**一个文件**——`task_console/window.py`(3,449 行)——它同时是 UI 壳、信号拓扑投影器、数据平面 derived-signal 生产者、run 生命周期协调器和呈现节拍器。拆"纯 UI 包"的主要工作不是清理垃圾,而是把这个窗口类按已经存在的内部缝合线切成 View / Presenter / Runtime 三份。

---

## 1) workbench(task console)现在的架构与逐文件归类

### 组织方式

- **面板(Panel)**:`PanelConfig`(console_records.py:193)= 一个 plot kind + 尺寸预设 + 像素位置 + **恰好一个** typed dataset key(:246-249 明确注释:合并多 producer 是 Processor 的事,GUI 不许发明 independent-latest 表达式)。`PanelCard`(panel_card.py:96)是 Fluent chrome 包一个 `zlc_plot.RasterPlotHost` 工作面;投影/selector/fit/rasterize 全在 host 里,卡片"只拥有路由和 widget 生命周期"(panel_card.py:97-102,基本属实)。
- **卡片摆放**:`PanelBoard`(panel_board.py:85)只做"把已存在的卡放到 packer 算出的位置";真正的重力 packer 是 `zlc_frontend/board_layout.py` 的纯函数。
- **Producer**:Logic 标签页每行一个 `LogicNodeConfig`(console_records.py:96),由注入的 `host_factory` 冻结成 `LogicNodeHost`(window.py:2559-2594),start/stop 在 window.py:2414-2510/2678。窗口不 import 任何设备后端——设备目录、descriptor、data plane 全部构造注入(window.py:178-194,app.py:14-56)。
- **处理链**:没有 workbench 内的处理链。数据流是 `LogicNodeHost → SignalDataPlane(zlc_neutral_atom/processing/signal_plane.py:574)→ freeze() → SignalFront → PanelCard`。面板上的 selector/fit 结果再由窗口发布回数据平面成为 derived signal(见 §3 耦合一)。

### 逐文件归类(行数 / 定性)

| 文件 | 行数 | 归类 |
|---|---|---|
| `task_console/window.py` | 3,449 | **混合体**。约 1/3 纯 UI(_build_ui :344-548、arrange :2001-2063、add/remove panel、tab/editor 管理、文件对话框 :3133-);约 1/3 数据逻辑(信号拓扑 :1023-1444、selection/fit 发布路由 :683-940、resolve_node_inputs :1629、run 生命周期 :2344-2790);约 1/3 呈现运行时(owner wake 三通道 :2867-2933、surface 批次 :2596-2696、tick :3062、关闭协议 :3107-3375) |
| `task_console/panel_card.py` | 1,198 | **混合体**。~400 行 chrome/settings 弹窗 UI(:862-1125);~800 行呈现运行时:host 代际替换与 schema fingerprint 比对(:334-405)、future 应答(:407-449)、worker 事件泵(:504-538)、host 退休(:1126-1200) |
| `task_console/panel_editor.py` | 384 | 混合:Edit 标签页 UI + 自己的 host/surface 生命周期副本 |
| `task_console/logic_node_parameter_panel.py` | 412 | UI + descriptor→FormSpec 投影(依赖 zlc_neutral_atom 的 InputSpec/Descriptor) |
| `task_console/console_records.py` / `console_state.py` | 332+99 | **纯值+codec,无 Qt**(文件头明说 renderer-free),仅依赖 zlc_plot 的 PlotKind/PlotSpec 词汇与 zlc_storage.canonical |
| `task_console/layout_repository.py` | 98 | 纯存储 IO(原子写 + durable_makedirs) |
| `task_console/panel_board.py` | 99 | 纯 UI 几何适配(唯一的语义尺寸→像素换算,card_size :26-49) |
| `logic_node_row.py` / `published_signal_row.py` / `logic_node_editor.py` | 137+81+83 | 纯 UI |
| `task_console/app.py` | 56 | 纯组合层(参数验证+转发) |
| `window_runtime.py` | 87 | **运行时**:双 ThreadPoolExecutor + 原子导出发布,零 Qt |
| `pulse_editor/` | 9,959 | 已是 MVP 分层样板:`controller.py`(2,563,文件头 "Qt-free current Pulse GUI controller")+ `session.py`(255,文档状态)+ `window.py`/`schedule_view.py`/`target_view.py` 等 view(~6,000)。但 view 直接读 `PulseDocument` 做投影(schedule_view.py:121-315 的一堆 `_schedule_facts` 类函数——headless 但住在 view 文件里) |
| `device_manager/` | 1,622 | 同样三分:`controller.py`(QObject 但薄)+ `editor_session.py`(纯草稿状态机)+ `window.py`(UI) |
| `data_figure/` + `figure_viewer/` | 855+530 | **最干净的样板**:`archive_io.py`(IO)/`info_projection.py`(纯投影,吃 archive 返回 tuple)/`window.py`(UI) |

**比例感**:全 workbench 里真正"纯 UI"约 55-60%;数据/领域投影约 25%;呈现运行时与线程护栏约 15-20%。杂质高度集中在 `task_console/window.py` 与 `panel_card.py` 两个文件。

---

## 2) zlc_frontend 离"纯 UI 控件库"还差多远

**已经很近**。逐模块:

- ✅ `qt_widgets/fluent.py`(3,462 行):纯 Fluent 控件 + scale/window 管理,零领域依赖。
- ✅ `qt_widgets/style.py`、`owner_wake.py`(343 行,QtOwnerWake/SerialWorkerWindow——通用 Qt 并发原语)、`published_signals.py`、`figure_info_pane.py`、`signal_picker.py`(文件头明说"只依赖 frontend 控件和 plain data",输入全是 dict/list)。
- ✅ `board_layout.py`(166 行):纯矩形几何,`BoardMetrics.card_size` 保持 callable 防 stale scale(:19-35),测试友好。
- ✅ `form.py`(546 行 headless FormSpec)+ `qt_widgets/form.py`(1,068 行 Qt 投影):动态数据通过 `FormRuntimeContext` 四个回调注入(qt_widgets/form.py:63-70),不 import 领域包。这是**正确的模式**。
- ✅ `qt_widgets/__init__.py` 惰性导出(:141-148):import `ensure_qt_app` 不拉起 Matplotlib。

**还差的三处杂质**(全部有 import 证据):

1. **`plot_parameters.py` / `plot_spec.py` / `plot_fit.py`(合计 1,129 行)依赖 `zlc_plot`**(plot_parameters.py:18-24 import RasterPlotHost 等)。它们不是通用控件,而是 *zlc_plot 的 Qt 编辑面板*。zlc_plot 自己已经内含 Qt 适配(backends.py 动态构造 Qt5PlotWidget),这三件应该跟着 zlc_plot 走(独立 `zlc_plot_qt` 或 zlc_plot 的 extras),不该留在通用控件库。
2. **`shape_text.py` 依赖 `zlc_data.schema.DatasetSchema`**(:12)并做 shape 校验(:61-68)。校验是数据层职责;控件库版本应只接受一个已算好的字符串,或把此模块移入 zlc_data 侧。
3. **`flow_graph.py` 依赖 `zlc_storage.canonical_text`**(:8)——只用一个字符串验证函数,内联即可断掉整个 zlc_storage 依赖。

结论:剥掉这三处后,`zlc_frontend` 就是一个只依赖 PyQt5 的独立控件包,可以直接按 zlc_plot 的模式拆出去。

---

## 3) 达到"UI 只提供界面+信号槽"目标,最难剥离的 3 处耦合

**耦合一(最深):窗口是数据平面上的 derived-signal 生产者。**
`_accept_card_selection`(window.py:725-855)/`_accept_card_fit`(:857-931):面板上的框选/十字/fit 结果,由窗口 `materialize_value_dataset` 后经 `SignalDataPlane.bind_continuous_derived / publish_continuous_derived / bind_event_derived`(signal_plane.py:1577/1734/1834)发布成新信号,并维护 `_PanelSelectionRoute`(:158-171)里的 generation/contract/schema 路由事实,还要在正确时机 `withdraw_derived` + 触发拓扑刷新。这不是"UI 显示数据",而是"UI 手势生产数据"。剥离难在:路由正确性依赖 publication identity、代际比较和拓扑投影三者的原子协同(:788-855 的 same_route 判定)。重写时这~330 行应整体成为 runtime 侧的 `SelectionRouter`,UI 只发 `selection_committed(panel_id, SelectionData, publications)` 信号。

**耦合二:信号拓扑投影住在窗口里。**
`_signal_topology`/`_artifact_topology`/`_SignalTopologyProjection`(:131-156, 1023-1211)+ `_promote_data_front`(:1373-1407):窗口遍历所有 logic row、卡片、输出声明,合成"谁在生产什么"的全局投影,喂给 picker/legend/status,并用 `_signal_info_dirty` 手工管理缓存一致性(:277-279 注释)。它消费 `DatasetOutputSpec/ArtifactOutputSpec/LogicNodeDescriptor/DeviceCatalogView` 等纯领域类型(:92-102)。难点:每个 UI 表面(卡片 picker、logic legend、form runtime context :2368-2412)都从它取数,拆开要先定义一个稳定的 ViewModel 数据形状(现在直接传领域对象)。

**耦合三:PanelCard↔RasterPlotHost 的代际/批次呈现协议。**
`prepare_surface_update`(panel_card.py:334-405,含 host 替换判定与 fingerprint 比较)+ `can_accept/accept_surface_update`(:407-449)+ 窗口侧 `_enqueue_surface_batch/_drain_surface_batches`(window.py:2596-2696,保证"整组卡片同一 SignalFront 原子上屏,不呈现半个 board")。这套 board-coherent 批次仲裁是精心设计的核心不变量(对应记忆中 W1b 渲染线程弧的最终形态),**值得保留**,但它是呈现运行时,不是控件。剥离难在 PanelCard 同时持有 chrome UI 和这套状态机——需要把卡片拆成 `PanelCardView`(纯 chrome,接受任意 QWidget 面)+ `PanelSurfaceCoordinator`(非 QWidget,可脱 GUI 测试)。

(第四名:`_start_logic_node` 的 run 生命周期 :2414-2510——但 host_factory 已注入,窗口只做编排,剥离相对容易。)

---

## 4) tick / render / 信号消费的实际路径,及 runtime/UI 归属

**先纠正一个历史事实**:记忆里的 `render_loop.py`、"GUI 触 figure 先 barrier"在迁移分支已**不存在**(全仓 grep 无 render_loop;"barrier" 无命中)。渲染 worker 已整体内化进 `zlc_plot.RasterPlotHost`(zlc_plot/session.py,自带 worker 线程 + 独立 fit analysis executor :1038)。workbench 只剩**节拍与一致性仲裁**:

1. **显示节拍**:单 `QTimer`,base = 所有面板 `update_ms` 的最小值(谐波集合 `UPDATE_INTERVALS` 来自 `zlc_plot.DEFAULTS.live`,console_records.py:176),每面板按 `elapsed % update_ms` 分拍(window.py:3062-3088)。无卡片/暂停时 timer 停(:2960-2975)。
2. **每 tick**:`self._data.freeze()` 取一个不可变 `SignalFront` → `_panel_render_groups` 按 continuous group 分组(:3047-3060)→ 组内所有卡从**同一个 front** `prepare_surface_update` 得到 worker future,打包成一个 batch(:2596-2638)。
3. **worker 完成** → `_surface_future_done` 置 surface 标志 → `QtOwnerWake`(owner_wake.py:27-105,合并唤醒 + 竞态 replay)→ Qt owner 线程 `_drain_surface_batches` 整组验收上屏(:2647-2696)。
4. **三通道唤醒合一**:`_owner_cycle`(:2900-2933)按 lifecycle(节点轮询)/ data(数据平面 `bind_owner_wake` 借出的 wake,:2885-2898)/ surface 三个 pending 位分发;另有窄 terminal poll timer 只在有 run 且无完成事件时跑(:2977-2989)。
5. **卡片级事件泵**:selection/fit/configuration 等 worker 回调进 `_worker_events` deque,由卡片自己的 `QtOwnerWake` 在 GUI 线程消费(panel_card.py:504-538)。

**归属判定**:
- 属于未来 **runtime 包**:`SignalDataPlane`(freeze/publication/derived 绑定,现在在 zlc_neutral_atom.processing)、`LogicNodeHost`/host_factory、`window_runtime.py` 的 compute executors、`RasterPlotHost`(留在 zlc_plot)。
- 属于**呈现运行时**(单独一层,非纯控件,但可无头测试):surface batch 仲裁、continuous-group 分拍、代际替换判定、`QtOwnerWake`(原语本身放控件包没问题,它零领域依赖)。
- 属于**纯 UI**:QTimer 起停、pause/selectors 开关、状态条优先级梯(:3101-3130)、board 摆放。

---

## 5) 重写提案:纯 UI 包的 API 骨架

三层切法(下两层依赖上层,反向禁止):

**A. `zlc_widgets` —— 通用控件库(只依赖 PyQt5)**
= 现 zlc_frontend 去掉 plot_*/shape_text 的 zlc_data 依赖/flow_graph 的 canonical_text。API 即现状:Fluent 家族、`scaled_px/set_fluent_scale/window_pad`、`QtOwnerWake`、`board_layout.pack/drop_index/BoardMetrics`、`FormSpec + FluentParameterForm + FormRuntimeContext`、`fill_grouped_signal_combo(names, sources, formats, labels)`、`FlowGraphView.set_graph`。这层**现在就基本达标**,拆包成本低。

**B. `zlc_console_ui` —— task console 纯视图(依赖 zlc_widgets;不依赖 zlc_neutral_atom/zlc_data)**

```python
class PanelCardView(FluentGroupBox):
    # 出向:全部是"操作者意图",无返回值
    signal_picked      = pyqtSignal(str)          # 信号 key(不透明字符串)
    size_picked        = pyqtSignal(str)
    update_ms_picked   = pyqtSignal(int)
    title_committed    = pyqtSignal(str)
    remove_requested   = pyqtSignal()
    edit_requested     = pyqtSignal()
    dropped            = pyqtSignal(QPoint)        # 拖放落点(像素)
    # 入向:全部是"给我看什么"
    def set_surface(self, widget: QWidget | None) -> None: ...   # 任意面板面;None=占位
    def set_signal_choices(self, groups: Sequence[PickerGroup]) -> None: ...
    def set_status(self, text: str, *, error: bool) -> None: ...
    def set_selectors_enabled(self, on: bool) -> None: ...

class ConsoleBoardView(QWidget):
    order_changed = pyqtSignal(tuple)              # panel_id 新顺序(拖放后)
    def set_cards(self, cards: Sequence[PanelCardView]) -> None: ...
    def arrange(self, geometry: Mapping[str, QRect]) -> None: ...  # 位置由外部 packer 算
    def grab_board(self) -> QPixmap: ...

class LogicRowView(FluentFrame):
    start_requested/stop_requested/edit_requested/remove_requested = pyqtSignal()
    def set_state(self, state: Literal["idle","running","error"], status: str) -> None: ...
    def set_publishes(self, rows: Sequence[tuple[str, str, str]]) -> None: ...  # name/shape/desc

class TaskConsoleView(QWidget):
    add_panel_requested = pyqtSignal(str)          # kind key
    add_logic_requested = pyqtSignal(str)          # descriptor key
    pause_toggled/selectors_toggled = pyqtSignal(bool)
    save_requested/load_requested/save_image_requested = pyqtSignal()
    def show_status(self, text: str, severity: Literal["info","warning","error"]) -> None: ...
    def set_summary(self, text: str) -> None: ...
```

关键约束:视图接口里**只有 str key、plain tuple、QWidget**——`SignalPublication/DatasetSchema/LogicNodeDescriptor` 一律不过界。picker 的展示形状沿用现 `signal_picker.py` 的 `(display, key)` 约定(它已经是纯的)。

**C. Presenter/Runtime(留在 workbench 应用包,不进 UI 包)**
`TaskConsolePresenter` 持有 SignalDataPlane、host_factory、拓扑投影(§3 耦合二)、SelectionRouter(§3 耦合一)、SurfaceCoordinator(§3 耦合三,产出 QWidget 交 `set_surface`);连接 view 信号。`console_records/console_state/layout_repository` 原样保留在这层(它们已是无 Qt 纯值,是现成资产)。

**值得保留的概念**(重写别丢):north-west 重力 packer 的"顺序即真理"(board_layout.py:117-141)、`QtOwnerWake` 的合并唤醒+replay、板级原子批次呈现、`PanelConfig` 单信号铁律、layout_record 的精确键集 codec、`FormRuntimeContext` 回调注入模式、pulse_editor 的 Qt-free controller 分层。

**该删的偶然复杂度**:window.py 里 `embedded` 双模式分支(:243-248,大量 `None` 按钮判断——应由组合而非 flag 实现);panel_editor.py 与 panel_card.py 各养一套 host 退休状态机(应共享 SurfaceCoordinator);`_task_takeover_row` 这类"仅 UI 命令门"却存在窗口字段上的应用级状态(:292-294);figure_viewer 内嵌整个 TaskConsole 来显示静态图(拆出 View 后自然消解)。