# pulse_editor 纯视图重建审查报告(负责范围:schedule_view.py / window.py / _layout.py / repeat_presentation.py / app.py)

所有路径相对 `C:/Users/eadri/Dropbox/WorkCode/Github/Zou_lab_control_v1_claude/Zou_lab_control_v1/zlc_workbench/pulse_editor/`。

---

## 1. 各文件职责与内部结构

### 1.1 schedule_view.py(2892 行)—— Edit 页(schedule 网格)

自评为"pure-Qt view"(docstring :1-7),但**入口全部吃 `PulseDocument`/`PulseTargetManifest`,并在模块级藏了一整层投影函数**。结构:

**A. 模块级投影函数(全部是"从领域类型算显示事实"的代码,拆包时整体迁到 presenter 侧)**

| 位置 | 函数 | 领域触点 |
|---|---|---|
| :108-113 | `_summary_time_text` | `TIME_UNIT_CHOICES`/`TIME_UNIT_TO_NS`(zlc_pulse 常量) |
| :121-124 | `_expanded_pulse_count` | `count_authored_digital_pulses(document)` |
| :127-142 | `_repeat_summary_text` | `document.periods`/`document.repeat` → 调 `pulse_repeat_presentation` |
| :145-148 | `_restart_high_ports` | `document.repeat_restart_high_ports()` |
| :151-203 | **`_schedule_facts`** | 核心汇总投影:active 端口集、`document.authored_duration_ns()`、repeat 三元组、`document.scan_table.rows` 点数、header summary 文本 |
| :240-265 | `_port_rows` | `document.target.by_key`、`manifest.ports`、**abi_fingerprint 一致性硬校验(:247)** |
| :268-279 | `_field_bindings` | `document.scan_parameters`/`api_parameters` → `PulseFieldRef` 键的 dict |
| :282-287 | `_delay_values` | `document.delays` |
| :290-297 | `_digital_values` | `document.digital_output_cells()` |
| :300-314 | `_analog_values` | `document.effective_dac_cells()`、`document.target.ports` |

纯 UI 的模块级助手::83-95(`_bus_mode_title/_value` 字符串映射)、:98-105(时钟文本)、:116-118(`_number`)、:225-237(`_apply_field_state`)、:317-327(`_set_duration_units`,但引用领域常量 `TIME_UNIT_CHOICES`)、:330-335(`_set_widget_text` 幂等写)、:338-365(`_reconcile_widget_order` 稳定键 reconcile)。

**B. 两个 headless 值类型(已是 zlc_ui 形态的雏形)**:`_PortRow`(:206-215,key/kind/label/width/signed_range/endpoints,全 plain)、`_Binding`(:218-222,kind/position/parameter_id)。

**C. 类**

- **`PeriodCard`**(:368-854)—— 一列 period 卡。纯 Qt 绘制/交互为主,但:构造与 `reconcile()`(:788-850)直接吃领域 `period` 对象;`_configure_duration_edit`(:634-638)调 `time_value_per_tick`(领域换算);`_projection_state` 缓存(:806-824)把 `period` 领域对象放进相等性键。出向信号 :371-375。
- **`ChannelNamesPanel`**(:857-1042)—— 左侧 Port Catalog 列:文档名编辑 + Total/Periods/Visible 只读行 + 每端口"硬件 endpoint 标签 + 可编辑显示名"。除 `_PortRow` 外已纯。信号 :860-861。
- **`ChannelPanel`**(:1045-1356)—— 左侧 Delay/Scan 列:Clock 只读、scan 摘要、Load Array 按钮 + 文件名、"Use loaded file" 开关、每端口"标签 + delay 编辑(带 dot)+ 单位 combo + X 清除钮"。领域触点:构造 `PulseFieldRef`(:1172、:1224、:1271)、`time_value_per_tick`(:1278)。信号 :1048-1052。
- **`RepeatBracket`**(:1359-1421)—— bracket start/end 窄卡;end 卡带 repeat 次数 spin。纯 Qt(仅 `DEFAULT_REPEAT_COUNT`/`MIN_REPEAT_COUNT` 常量)。
- **`PulseDragContainer`**(:1431-1769)—— 拖拽面。**完全纯 Qt**:mime `"application/x-zlc-pulse-card"`、半透明 ghost(:1565-1587)、插入指示条(:1712-1731)、拖拽自动滚动(:1597-1608)、bracket 合法性 `_bracket_ok`(end≥start+3,:1757-1769)、选中描边/gap 高亮 `show_selection`(:1733-1746)。**提案式:拖放从不本地应用,只 emit**(:1624-1651)。
- **`PulseScheduleView`**(:1772-2883)—— 页面本体。24 个出向信号(:1775-1800);`set_document`(:2236-2445)吃 PulseDocument+Manifest,带 (generation, revision) 陈旧拒收 + 同 revision 双文档身份硬校验(:2277-2295);`accept_local_commit`(:2447);十余个窄 `apply_*`(**全部再吃 PulseDocument**,:2457-2606);`refresh_summary`(:2608)重跑 `_schedule_facts`;纯视图状态:选中 period/gap(:2720-2734)、左栏折叠(:2863-2883)、隐藏端口 combo 幂等重建(:2814-2851);意图组装:`_request_add_period/_remove/_toggle_bracket/_repeat_count/_add_port/_hide_off/_show_all`(:2736-2812)只用留存的 id/label 字符串,已是纯视图逻辑。布局:左双栏 scroll 与 timeline scroll 垂直滚动条双向绑定(:1935-1938),timeline 垂直滚条永久占位防列跳(:1924-1926)。

### 1.2 window.py(1738 行)—— 壳 + (实质上的)presenter

- 模块助手 `_pulse_files_dir/_pulse_figure_dir/_safe_file_stem`(:76-99)—— 路径策略,归 presenter。
- **`PulseEditorWindowBody`**(:102-1704):
  - 构造(:105-158):校验 controller 类型、`QtOwnerWake` 绑 `_owner_cycle`、`controller.set_notify`、40ms `QTimer` runtime tick。
  - `_build_ui`(:164-226):header(status dot + 名称 label + summary 行 + Clear All 钮)+ 4 tab(Edit/Preview/Scan/Target)。组装本身纯 UI,但各页构造实参是领域投影。
  - `_wire_ui`(:228-291):把每个 view 信号接到 `_edit_*`/`_invoke_*` 私有槽 —— **这一段就是未来 presenter 的接线表**。
  - 命令-呈现骨架(:297-620):`_invoke_worker/_invoke_scan_worker/_invoke_connection/_invoke_editor_boundary/_commit_local_edit`(:356-415,同步命令 + 窄呈现 + 本地投影账本前滚)+ 每个 `_edit_*` 的"调 controller → 调对应 `schedule_view.apply_*`"配对(:417-573)。全是 presenter 代码。
  - 对话框::622-628 `_message`(fluent_message;offscreen 时写 summary)、`fluent_confirm`(:644-651 target 破坏性确认、:1634-1641 关窗弃存确认)、五处 `QFileDialog`(:725、:735、:758、:773、:783、:801)。
  - **领域逻辑混入点(逐处)**:`_run_from_edit`(:665-679)按 `document.scan_parameters`/`scan_sweep_count` 选 `PulseExecutionForm` —— 应下沉 controller;`_apply_scan_progress`(:1308-1338)读 `applied.source_document.scan_table` + `scan_column_specs` 拼进度值 —— 领域投影;状态灯颜色(:1448-1462)按 `RunState` 枚举 + `runtime.is_document_applied(document)` 推色;标题字符串(:1429-1444)与连接措辞(:1497-1539)是纯字符串策略但输入是领域快照。
  - Preview 管线(:792-939、:1541-1614):`RasterPlotHost`/`Qt5PlotWidget`(zlc_plot 类型)、Future 追踪、挂载 —— 属 plot/presenter 侧;唯一过界物应只剩 `preview_view.mount_content(widget)` 的 QWidget。
  - Owner 循环(:945-1000):`controller.pump()` → 按 `PulseOwnerUpdate` 分发 5 类窄更新;`_runtime_tick`(:988)轮询;`_sync_runtime_watchers`(:1281-1292)按需启停 timer。presenter 主循环。
  - 生命周期(:1601-1704):关窗确认、借用方 retire、永久关闭提交。
- `launch_pulse_editor_window`(:1707-1735):Fluent 窗口包装 + close guard 接线。

### 1.3 _layout.py(159 行)—— 冻结几何 token

纯几何:ROW_HEIGHT/PERIOD_CARD_WIDTH 等常量(:19-25)+ DPI 缩放包装(:28-77)+ 三个布局助手(:80-126)+ `channel_row_height` 密度铁律(:129-137)。仅依赖 `zlc_frontend.qt_widgets` 的 `scaled_px/measure_text_width/FluentLabel`。**整文件可原样进 zlc_ui**(前提:Fluent 部件库本身的归属已裁决)。

### 1.4 repeat_presentation.py(47 行)—— 纯呈现策略

`pulse_repeat_presentation(period_count, repeat_spec)`(:12-44):int 进、(措辞, bracket span) 出,零 Qt 零领域 import。**已是 zlc_ui 级 headless 值逻辑,原样搬**。注意当前被两侧共用:schedule_view :141 只取措辞,bracket span 由 preview 侧用(不在本范围)。

### 1.5 app.py(144 行)—— 组合根

`_editor_session`(:32-48)构 session;`open_pulse_editor`(:51-144)校验 facade/descriptor 成对、bound 分支做 manifest 收窄 + 时钟栅格校验(:92-99)、offline 分支 `load_deployed_pulse_target/geometry`(:119-126)、建 controller、`launch_pulse_editor_window`、可选 `controller.connect("remote", …)`(:142-143)。**全部留在领域/编排侧,zlc_ui 零内容**。

---

## 2. schedule 网格画什么、手势与"操作者意图"全清单

### 画面(period×channel 网格 = 左侧两冻结列 + 横向滚动的 period 卡序列)

- **每张 PeriodCard**:标题 "Period i/N";duration 编辑框(`FluentScanLineEdit`,右侧内嵌 scan/API dot)、单位 combo(scan 绑定时锁成 `str (us)` 伪单位,:317-327)、period 名编辑;下方按端口顺序:digital 行 = 带标签复选框(:510-519),DAC 行 = 模式 combo(Edge/Ramp/Hold)+ 带 dot 的带符号整数编辑框(:575-632)。
- **绑定徽章**:dot 三态 —— 空心=未绑、橙实心+数字=scan sN、紫实心+数字=API aN(scan_line_edit.py :53-78);scan 绑定行文本显示 `sN` 且只读、combo 禁用(:617-622)。
- **RepeatBracket**:start 窄卡 + end 窄卡(带 ×count spin),插在卡序列中(:2201-2225)。
- **高亮**:选中 period 的 ACCENT 描边、选中 gap 的竖插入条(:1733-1746);拖拽时 ghost 图 + 插入指示条 + 边缘自动滚动。
- **左列**:Port Catalog(endpoint 文本如 `d0` / `a0…a15` + 可编辑显示名)与 Delay/Scan(Clock 只读、scan 摘要、Load Array、Use loaded file 开关、每端口 delay+单位+X)。
- **底栏三组**:Control(On Pulse*/Stop/Sync/Add Period/Remove/Add Bracket/Save*/Load/Collapse,:1958-2007)、Connection(virtual/remote/offline combo + host:port + Connect + 状态行,:2010-2048)、Ports(隐藏端口 combo + Add/Hide Off/Show All + Visible 统计,:2050-2088)。

### 手势 → 操作者意图全清单(未来的出向信号)

**编辑类(*_committed 语义)**
| 意图 | 载荷(建议纯类型) | 现触发 |
|---|---|---|
| `document_name_committed` | (name: str) | :900 |
| `port_label_committed` | (port: str, label: str) | :947 |
| `period_name_committed` | (period_id: str, name: str) | :657-661 |
| `duration_committed` | (period_id: str, value: float, unit: str) | :645-655 |
| `digital_committed` | (period_id: str, port: str, high: bool) | :514-518 |
| `analog_committed` | (period_id: str, port: str, mode: str, value: int\|None) | :663-674(hold→None) |
| `delay_committed` | (port: str, value: float\|None, unit: str)(0→None,:1292) | :1286-1293 |
| `binding_cycle_requested` | **(field_kind: str, period_id: str\|None, port: str\|None)** —— 现在 emit `PulseFieldRef` 对象(:475、:614、:1173),**必须改 plain tuple** | dot 点击 |
| `repeat_committed` | (start_id: str\|None, end_id: str\|None, count: int) | 三源:Add/Del Bracket(:2759-2770)、spin 编辑(:2772-2783)、bracket 拖拽(:1649-1651) |
| `visible_ports_committed` | (ports: tuple[str, ...]) | Add/Hide Off/Show All(:2785-2812) |

**结构类**
| `insert_period_requested` | (before_period_id: str\|None,由选中 gap/卡换算,:2736-2745) |
| `move_period_requested` | (period_id: str, before_id: str\|None)(拖放,:1644-1648) |
| `remove_period_requested` | (period_id: str,选中优先、否则末位,:2747-2757) |
| `clear_port_requested` | (port: str)(X 钮,:1189-1191) |
| `clear_all_requested` | ()(header 钮,window :245) |

**运行/文件/连接类**
| `run_requested` / `stop_requested` / `sync_requested` / `save_requested` / `load_requested` | () | :1791-1795 |
| `connection_requested` | (mode: str, endpoint: str) | :2858-2861 |
| `scan_array_load_requested` | ()(文件对话框归属见 §4) | :1125 |
| `scan_source_committed` | (use_loaded: bool) | :1141 |

**视图状态通报(可留可发)**
| `left_panels_collapsed` | (bool)(:2875/:2883) | `feedback_requested` | (text: str)(:2764,视图本地校验措辞) |

**纯视图内部手势(不出界)**:卡点击选中切换、gap 点击选中(:2720-2734)、拖拽过程指示、滚动同步。

---

## 3. 纯视图输入视图模型草案(plain data)

原则:presenter 把 §1.1-A 全部投影函数的输出预烘焙成下面的 record;视图不再 import zlc_pulse。record 全部 frozen dataclass(str/int/float/bool/tuple),归 zlc_ui 自己的 headless 值类型。

```python
# ---- 通用字段(duration / DAC 值 / delay 共用) ----
FieldVM:
    text: str                    # 已含 "sN" 替换后的显示文本
    editable: bool               # False → muted 样式 + 只读
    binding_kind: str            # "" | "scan" | "api"
    binding_number: int          # 1 基;binding_kind=="" 时为 0
    binding_tooltip: str         # "Parameter: <id>" 或 ""
    validator_kind: str          # "int" | "float" | "none"(scan 绑定时 none)
    validator_lo: float
    validator_hi: float          # int 用;float 只用 lo
    resolution: float            # 当前单位下的 tick 量子
    allow_any: bool

PortRowVM:                       # 三列共享的行目录(替代 _PortRow)
    key: str
    kind: str                    # "digital" | "dac"
    label: str
    endpoint_text: str           # 预拼 "d0" 或 "a0…a15"
    endpoint_tooltip: str
    width: int                   # DAC 位宽(combo 显示 "(N pins)" 用)
    lo: int; hi: int             # signed_range,digital 为 (0,0)
    visible: bool
    active: bool                 # 任一 period 高/非 hold —— Hide Off 依据(替代 _active_ports)

PeriodVM:
    period_id: str
    name: str
    duration: FieldVM
    unit: str
    unit_choices: tuple[str, ...]        # 含伪单位 "str (us)" 时由 presenter 追加
    unit_locked: bool                    # scan 绑定 → combo 禁用
    digital: tuple[tuple[str, bool], ...]           # (port_key, checked)
    analog: tuple[tuple[str, str, FieldVM], ...]    # (port_key, mode_title, value)

RepeatVM:
    start_period_id: str; end_period_id: str; count: int    # 无 bracket → ScheduleVM.repeat=None

DelayRowVM:
    port_key: str
    value: FieldVM
    unit: str
    unit_quantums: tuple[tuple[str, float], ...]   # (unit, tick量子)——单位切换时视图本地重配 validator,
                                                   # 免去视图调 time_value_per_tick(:638/:1278 的替代)

ScheduleVM:                      # set_schedule(vm) 的唯一入参
    document_generation: int
    revision: int                # 陈旧拒收判据照抄 :2279-2295
    document_name: str
    clock_text: str              # "50 MHz · 20 ns"
    total_text: str; total_tooltip: str
    period_count: int
    visible_text: str            # "4/12"
    summary_text: str            # header summary 整句(_schedule_facts 第 5 元)
    ports: tuple[PortRowVM, ...]           # 全量目录;visible 标志控制显隐
    periods: tuple[PeriodVM, ...]
    repeat: RepeatVM | None
    delay_rows: tuple[DelayRowVM, ...]
    scan_summary_text: str                 # "2 slots · 100 pts"
    scan_source_loaded: bool; scan_file_path: str
    min_repeat_count: int; default_repeat_count: int   # 替代 zlc_pulse 常量 import
```

配套窄 set_*(替代现 apply_* 家族,均幂等):`set_schedule(vm)`、`accept_local_commit(generation, revision)`、`set_period(PeriodVM)`、`set_delay_row(DelayRowVM)`、`set_port_label(key, label)`、`set_visible_ports(tuple[str,...])`、`set_summary(total_text, total_tooltip, period_count, visible_text, summary_text, scan_summary_text)`、`set_scan_source(use_loaded, path)`、`set_scan_busy(bool)`、`set_connection(mode, endpoint, status)`、`set_control_state(running, synchronized, file_dirty)`。

要点:现 `apply_*` 每个都重新吃整个 document 再局部投影(:2457-2606)——纯视图化后 presenter 投影出**该窄 record** 传入即可,视图侧 reconcile 骨架(`_reconcile_widget_order`、`_projection_state` 缓存、`signals_blocked` 幂等写)原样保留。

---

## 4. window.py 壳的成分裁决

| 区块 | 现状 | 裁决 |
|---|---|---|
| header(status dot/名称/summary/Clear All,:178-201) | 纯 UI 组装 | 进 zlc_ui;输入改 `set_title_line(text)`、`set_status_color(token: str)`、`set_summary(text)` |
| 标题/星号策略(:1429-1444) | 纯字符串策略,输入是领域投影 | 措辞留视图或 presenter 皆可,输入必须变 (name, file_label, status, dirty) plain |
| 状态灯配色(:1448-1462) | **混**:`RunState` 枚举 + `is_document_applied` | presenter 算出语义态(如 "running-synced"/"running-stale"/"cancelling"/"failed"/"dirty-ready"/"idle"),视图只做态→色映射 |
| tabs 组装(:203-226) | 纯 UI | 进 zlc_ui;但四页构造参数要先 VM 化 |
| `_wire_ui` + `_edit_*`/`_invoke_*`/`_commit_local_edit`(:228-620) | **presenter 本体**(命令→controller→窄 apply 配对) | 全部迁出为 Qt-free presenter;`_commit_local_edit` 的"前滚本地投影账本"模式(:356-415)是 presenter 核心资产 |
| `_run_from_edit`(:665-679) | **领域决策**(选 PulseExecutionForm) | 下沉 controller,视图只发 `run_requested()` |
| 对话框:fluent_message/confirm、5×QFileDialog(:622-821、:1634) | Qt 纯,但上下文(建议路径、过滤器、确认文案)由领域算 | 对话框机制留 zlc_ui;由于 presenter 必须 Qt-free,建议视图暴露命令式方法 `ask_open_path(caption, start_dir: str, filter: str) -> str`、`ask_save_path(...)`、`confirm(title, text, ok, cancel) -> bool`、`show_warning(text)`,presenter 传 plain 参数并接返回值 |
| `_apply_scan_progress`(:1308-1338) | **领域投影**(applied.source_document.scan_table + scan_column_specs) | 迁 presenter;视图只收 `set_progress_text(str)` |
| 连接措辞(:1497-1539)与"已连接"弹窗 | 字符串策略 + 状态沿检测 | 沿检测归 presenter;措辞可留视图(输入 plain state 串) |
| preview 管线(:792-939、:1541-1614) | zlc_plot 类型(RasterPlotHost/Qt5PlotWidget)+ Future | 归 presenter/plot 适配层;过界只允许 `mount_content(widget: QWidget, logical_size, wheel_target)`(QWidget 在允许清单内) |
| owner 循环 + 40ms tick(:945-1000、:1281-1292) | presenter 主循环(pump/poll/去重键) | 迁 presenter;视图不知道 controller 存在 |
| 生命周期(:1601-1704)+ `launch_pulse_editor_window`(:1707-1735) | 混:close 确认是 UI,`request_close`/`retire_borrowed_authority` 是 controller 协议 | 窗口包装、close-guard 机制进 zlc_ui;关闭协议由 presenter 经 `close_requested` 信号 + `finish_close()` set 方法接线 |

---

## 5. 与 controller.py / session.py 的现有调用面

### view(window)→controller —— 出向信号清单的现实依据

**同步编辑命令**(经 `_commit_local_edit`,controller.py 行号):`rename_document`(:694)、`rename_port`(:741)、`rename_period`(:716)、`set_period_duration`(:762)、`set_digital`(:779)、`set_analog(cascade=True)`(:793)、`set_delay(cascade=True)`(:824)、`cycle_binding`(:1026)、`add_period`(:903)、`move_period`(:930)、`remove_period`(:950)、`set_repeat`(:968)、`set_visible_ports`(:846)、`clear_port`(:878)、`clear_all`(:893)、`set_scan_sweep_count`(:705)、`select_scan_source`(:1106)。
**worker/边界命令**:`save`(:1195)、`open_path`(:1182)、`load_scan_array`(:1087)、`load_scan_program`(:1069)、`save_scan_array`(:1144)、`generate_scan_source`(:1045)、`scan_template_source`(:1039)、`connect(mode, endpoint)`(:1265)、`start(form, scan_sweep_count)`(:1384)、`cancel`(:1629)、`sync_applied`(:1646)、`hold_scan_point`(:1612)、`step_scan_point`(:1616)、`apply_target_manifest(…, cascade)`(:669)、`request_preview`(:1214)、`set_preview_include_off`(:1206)、`request_scan_progress`(:1516)、`request_close`(:1663)、`retire_borrowed_authority`(:1670)。
**拉取面(presenter 专属,视图化后消失)**:`pump`(:1683)、`poll_runtime_change`(:1726)、`editor_projection/runtime_update/preview_update`(:539/:588/:580)及属性 `current_document/current_document_generation/current_editor_revision/dirty/current_path/current_scan_workspace/current_display_visible_ports/worker_idle/runtime_poll_required/scan_progress_poll_required`(:486-527、:1758-1781)。

对账:出向信号清单(§2)与同步命令表一一对应,仅 `run_requested` 需 controller 侧补"选 execution form"的收口;`sync/hold/step` 三处 `getattr(...) is None` 兜底(window :682-700)在拆包时应换成 capability 标志下发(`set_capabilities(can_sync: bool, can_hold: bool, …)`)。

### controller→view 更新调用 —— set_* 清单的现实依据

载体是五个 frozen 更新类型(controller.py):`PulseEditorProjection`(:291-308,**含 PulseDocument/PulseTargetManifest/ScanWorkspaceSnapshot,是过界主犯**)、`PulseRuntimeUpdate`(:236-268,含 RunSnapshot/AppliedPulseSnapshot/描述符)、`PulseFileUpdate`(:271-280,已近乎 plain,仅 Path)、`PulsePreviewUpdate`+`PulsePreviewPlot`(:283-288、:149-182,zlc_plot 类型)、`PulseScanProgressUpdate`(:223-233)、聚合 `PulseOwnerUpdate`(:311-320)。

由 window 落到 schedule_view 的调用(= 未来 `set_*` 面):`set_document`(:2236)、`accept_local_commit`(:2447)、`apply_document_name/period_name/port_label/duration/digital/analog_port/delay/all_bindings/period_structure/visible_ports/port_clear/clear_all`(:2457-2606)、`refresh_summary`(:2608)+`summary_text()`(:2652)、`set_scan_source`(:2655)、`set_scan_workspace_busy`(:2663)、`set_connection_state`(:2670)、`set_control_state`(:2694);window 自有部件:`label_name.setText`/`setWindowTitle`(:1438-1446)、`status_dot.set_color`(:1462)、`summary.setText`(:385、:1069)。

### session.py(255 行)定位

`PulseEditorSession`(:31-220)= 文档唯一可变 owner(revision 计数 :152-161、文件指纹冲突检测 :202-211)+ `project_pulse_preview`(:223-252,编译投影)。**全留领域侧**;对拆包唯一相关事实:generation/revision 版本纪律(view :2279-2295 的拒收协议)源头在此,ScheduleVM 必须携带这对整数。

---

## 附:拆包裁决摘要(供 goal 清单直接取用)

1. **原样进 zlc_ui**:`_layout.py` 全文件;`repeat_presentation.py` 全文件;`scan_line_edit.py`(dot 三态绘制,零领域);`PulseDragContainer`+`RepeatBracket` 整类;`PeriodCard/ChannelNamesPanel/ChannelPanel/PulseScheduleView` 的 Qt 骨架、reconcile 机制、选中/折叠/隐藏 combo 逻辑。
2. **换血后进 zlc_ui**:`set_document`/`apply_*` 全家改吃 §3 的 VM;`bindingCycleRequested` 载荷 `PulseFieldRef` → plain tuple;删除 `zlc_pulse` 全部 import(:39-57),常量(TIME_UNIT_CHOICES、DEFAULT/MIN_REPEAT_COUNT、FIELD_*/PORT_*)改由 VM 携带或 zlc_ui 自定义 kind 串。
3. **留领域/presenter 侧**:schedule_view :108-314 全部投影函数、window 的 `_wire_ui`+命令-呈现骨架+owner 循环+preview 管线+`_run_from_edit`+`_apply_scan_progress`+状态灯语义推导、`app.py` 与 `session.py` 全部。
4. **前置依赖裁决**:两文件深度依赖 `zlc_frontend.qt_widgets`(约 25 个 Fluent 部件/色 token/scaled_px)——zlc_ui 的部件基座归属必须先裁,否则 schedule_view 无法独立。
5. **presenter Qt-free 约束下的对话框**:QFileDialog/fluent_confirm/fluent_message 留视图,以命令式方法(plain 参数、plain 返回)暴露给 presenter(§4)。