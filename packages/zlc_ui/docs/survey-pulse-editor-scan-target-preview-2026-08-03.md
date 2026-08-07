# pulse_editor 纯视图重建审查报告

范围:`Zou_lab_control_v1/zlc_workbench/pulse_editor/`(总 9,989 行)。本人负责 target_view / scan_workspace / scan_view / preview_projection / preview_view / scan_line_edit / session / controller(公共面),window.py 与 schedule_view.py 仅作接线佐证。

---

## 1) 各文件职责与纯度判定

### scan_line_edit.py(192 行)— ✅ 完全纯 UI,原样平移
带内嵌 Scan/API 绑定圆点的数值输入框。只依赖 PyQt5 + `zlc_frontend.qt_widgets` 样式 token(scan_line_edit.py:5-24),零域 import。出向仅 `scanClicked`(:109);入向 `set_field_state(editable: bool, binding: 'scan'|'api'|None, number: int|None)`(:153-192)已幂等(状态相等即 return,:166-168)。`_FluentScanDot.nextCheckState` 置空(:50-51)= 绑定态归 model 所有,正是 zlc_ui 哲学的现成示范。唯一前提:zlc_ui 自带 Fluent token/控件底座。

### scan_view.py(399 行)— ✅ 类主体近乎达标,模块级函数放错了包
`PulseScanView`(:115-397)出向信号全部意图化、全 plain 类型(:118-125);入向 `set_*` 全 str/int/bool(:317-396),含成熟的草稿仲裁协议(`set_scan_code`/`replace_scan_draft`/`acknowledge_scan_draft` + `code_dirty`/`source_revision`,:345-376)。三处杂质:
- :29 `from zlc_pulse import DEFAULT_SCAN_SWEEP_COUNT, MIN_SCAN_SWEEP_COUNT` → 用于 spin 初值/下限(:153-156)。改为构造参数或 `set_repeats_range(minimum:int, default:int)`。
- **模块级 `format_scan_progress`(:42-93)/`format_held_scan_point`(:96-112)**:纯函数但 duck-typed 吃 `PulseScanProgress` 域对象,且 window.py:70-71 从这里 import —— 属 presenter 层,整体移出 view 模块。
- 200ms 进度轮询 QTimer(:305-315)只在可见时 emit `progressRefreshRequested` —— 纯 UI 节拍,合法保留。

### preview_view.py(231 行)— ✅ 已是"QWidget 挂载点",一处杂质
控制条(off 行开关/selectors 开关/size combo/保存钮/状态框)+ 滚动挂载区。出向 `includeOffToggled(bool)`/`selectorsToggled(bool)`/`sizeActivated(str)`/`saveFigureRequested`(:26-29);入向 `mount_content(QWidget, logical_size: tuple[int,int], wheel_target: QWidget)`(:166-196)、`show_placeholder(str)`、`set_status(str)`、`set_preview_size(str, pinned)`。QWidget 过界在 zlc_ui 白名单内。唯一杂质::18 `from zlc_plot import DEFAULTS` → :74 combo 选项 `DEFAULTS.layout.size_names`,改为 `set_size_names(tuple[str,...])` 注入。wheel 事件转发 eventFilter(:202-228)纯 Qt,平移。

### target_view.py(580 行)— ⚠️ UI 骨架纯,域逻辑深度内嵌(逐处点名见 §3)
:22-30 import 了 7 个 zlc_pulse 符号。混域处:
- `set_manifest` 吃 `PulseTargetManifest` + isinstance 硬检查(:67, :154),读 `manifest.fingerprint`(:158),view 内调域投影 `pulse_target_port_drafts(manifest)`(:161);
- `_resize_dac_endpoints` 调 `pulse_target_port_width_spec(PORT_DAC).normalize`(:468)—— 域校验进了 view;
- `_current_drafts` 在 view 内构造域对象 `PulseTargetPortDraft`(:488-516);
- `_add_dac` 用 `width_spec.default` 和 endpoint 占位命名约定(:546-554);
- `draft_manifest`/`_emit_apply` 调域构造器 `build_pulse_target_manifest` 并 **emit `applyRequested(manifest)` —— 域对象出境**(:567-577),直接违反 zlc_ui 契约。
可平移部分(~70%):`_TargetRowWidgets` 字段全 plain(:40-51)、key 锚定的行 reconcile 机制(:175-218)、`_set_line_text` 光标保真(:245-264)、`_place_rows` 网格搬迁(:292-339)、`_set_editable`(:418-441)、`_queue_reveal_row`(:478-486)。

### scan_workspace.py(488 行)— 🚫 纯域/编排层,整体留域侧
文件头自称 "independent of Qt"(:1)属实(零 Qt),但它是域操作集:zlc_pulse import(:20-36)、执行受信 Python(`execute_scan_program`,:108-119)、numpy/csv/文件 IO(`load_scan_array` :122-176、`save_scan_array` :192-237)、document commit(`commit_scan_candidate`,:267-288)。一行不进 zlc_ui。注意其中 `format_scan_table`(:304-336)/`format_scan_slots`(:339-413)是纯字符串 presenter,产物就是 scan_view 的 `set_scan_table_text`/`set_slots_text` 输入 —— 留 presenter 侧。`ScanWorkspaceSnapshot`(:72-90)携带 `FrozenScanTable`/`Path`,presenter 消费后压平为 str/bool/int 再喂 view。

### preview_projection.py(260 行)— 🚫 纯域投影,整体留域侧
零 Qt。**输入**:`PulseTimelineDocument`(zlc_pulse)+ `include_off_rows: bool`;**输出**:`(PulseTimelineData, PulseTimelinePlot)`(zlc_plot 类型,:28-127)、`recommended_size: str`(:130-161)、状态字符串(:171-190)。两端都是 zlc_ui 禁运类型,它正是 zlc_pulse→zlc_plot 的授权 seam(文件头 :1-7 明言)。依赖的 repeat_presentation.py(47 行,零 import 纯函数)随它留域侧。

### session.py(255 行)— 🚫 纯域,零改动
`PulseEditorSession` = 单一可变 document 拥有者:线程锁 + revision + dirty + save 指纹冲突协议(:184-220);`project_pulse_preview`(:223-252)= 编译投影(调 `compile_pulse_document`)。不与任何 view 见面,是 controller 底座。

### controller.py(2596 行)— 见 §5。

---

## 2) scan 工作区:交互与数据形状

**操作者意图清单**(现有信号已是意图,直接沿用):

| 信号 | payload | 触发 |
|---|---|---|
| `repeatsChanged` | int | editingFinished 且值变化(scan_view.py:333-337) |
| `holdRequested` | — | Hold 钮 |
| `stepRequested` | int(±1) | step 钮(:185-193) |
| `loadProgramRequested` | — | 文件对话留 presenter |
| `templateRequested` | str('column_stack'\|'grid') | 模板钮 |
| `runRequested` | str(编辑器全文) | Run 钮(:268-270) |
| `saveArrayRequested` | — | |
| `progressRefreshRequested` | — | 可见时 200ms 定时(:313-315) |

**纯视图模型 record 草案**(zlc_ui headless 值类型;由 presenter 从 `ScanWorkspaceSnapshot` + 域 format 函数压平):

```python
@dataclass(frozen=True)
class ScanPageRecord:
    slots_text: str          # format_scan_slots 产物
    table_text: str          # format_scan_table 产物(含 STALE 前缀)
    source_text: str
    source_revision: int     # 草稿仲裁基数
    source_dirty: bool
    repeats: int
    busy: bool               # busy_operation is not None
    progress_text: str       # held 优先,否则 format_scan_progress
    progress_polling: bool
```

**关键不可简化点**:三路草稿仲裁协议必须原样保留在 zlc_ui 接口(`set_scan_code` 全量重建 / `replace_scan_draft` UI 侧替换 / `acknowledge_scan_draft` 只推元数据)。window.py:1340-1391 的四分支仲裁(replace_source / loaded_program_completed / code_dirty 相等确认 / revision 落后重建)上移 presenter,这是"回投喂不踩正在输入的编辑器"的机制。

---

## 3) target_view 展示内容与纯视图模型

展示 = `PulseTargetManifest` 的可编辑行投影:两张卡(Digital / DAC),每行一个逻辑端口:signal 名、endpoints(逗号分隔文本)、DAC 加 width spin + latch clock endpoint。**lane 映射不直接展示**:`lane_order: tuple[int,...]` 只是随行携带的不透明数据(:50, :243, :510-513),Apply 时供域侧保持 lane 稳定;`key`/`clock_key` 同为不透明行锚。

**纯视图模型草案**:

```python
@dataclass(frozen=True)
class TargetPortRecord:          # zlc_ui 自己的 headless 值类型
    key: str                     # reconcile 锚
    kind: str                    # 'digital' | 'dac'
    signal: str
    endpoints: tuple[str, ...]
    clock_key: str | None
    clock_endpoint: str | None
    lane_order: tuple[int, ...]  # 不透明,原样回传

@dataclass(frozen=True)
class TargetWidthRule:           # 替换 pulse_target_port_width_spec
    minimum: int
    default: int
    maximum: int | None
```

**接口改写**:
- 入向:`set_ports(records: tuple[TargetPortRecord,...], editable: bool, status_text: str)` + `set_width_rules(digital: TargetWidthRule, dac: TargetWidthRule)`;fingerprint 短路(:158-162)改用 record 元组 `==`(plain 可直接比较)。
- 出向:`applyRequested(tuple[TargetPortRecord,...])`;`build_pulse_target_manifest` + 域校验移 presenter,拒绝经 `set_feedback(str)` 回注。
- 保留在 view 的合法逻辑:key 分配(:451-456,纯文本自增)、"endpoint 数 ≠ width"的文本级预检(:492-496)、DAC 宽度变化时的 endpoint 增删(:462-476,但阈值来自注入的 `TargetWidthRule`)。
- Add Digital/DAC 的 endpoint 占位命名 `endpoint:{key}[{bit}]`(:529, :551-553)是域命名约定 —— 二选一:作为模板 str 参数注入,或改 emit `addDigitalRequested`/`addDacRequested` 由 presenter 回注新行(推荐前者,保住"加行即时可见"的同步手感)。

---

## 4) preview 渲染裁决:QWidget 挂载点

**现状不自绘,全走 zlc_plot**。链路:controller worker 调 `pulse_timeline_plot` → 发布 `PulsePreviewPlot`(含 zlc_plot 的 data/spec,controller.py:149-182)→ window 建 `RasterPlotHost` + `Qt5PlotWidget`(window.py:47, :919)→ `host.replace_spec/update_data`(window.py:1589-1594)→ `preview_view.mount_content(widget, ...)`(window.py:894)。preview_view 自身只有控制条 + 滚动区 + 占位标签。

**裁决:纯 UI 包里 preview 保持"QWidget 挂载点",不自绘。** 理由:
1. `mount_content` 接口(preview_view.py:166-196)本来就符合 zlc_ui 白名单(QWidget 可过界),现状零改动;
2. 时间轴渲染主权(raster/blit/selector/导出)全在 zlc_plot(preview_projection.py:1-7 明文),zlc_ui 里自绘 = 第二渲染链,重蹈 QtRasterBoard/QtImageBoard 双链覆辙;
3. selectors 开关/size/save 都是对 host/widget 的操作,由 presenter 持有 host,view 只需转发意图信号;
4. 唯一补充:`set_size_names(tuple[str,...])` 消除 `DEFAULTS` import;`wheel_target` 参数保持 QWidget。
代价可接受:preview 页离开 zlc_plot 无内容 → placeholder 即空态。

---

## 5) controller.py 公共面盘点

**Qt-free 属实**:grep `PyQt|QtCore|QtWidgets|QWidget` 零命中;文件头声称(:1)成立。**但不是领域-free**:import zlc_plot(:16)、zlc_neutral_atom(:18-25)、zlc_pulse(:26-66)、session/preview_projection/scan_workspace(:67-90)。它是应用拥有者(自带 ThreadPoolExecutor worker、run facade `PulseRunFacade` Protocol :93-121、连接生命周期),**从不直接调 view**——window.py 是唯一接线者,方向是 view 信号→window→controller 方法,controller `pump()`→window→view `set_*`。

**公共面三组**:
- 同步编辑命令(返回 revision int|None):`replace_document`、`apply_target_manifest(PulseTargetManifest)`、`rename_document/rename_period/rename_port`、`set_period_duration/set_digital/set_analog/set_delay/set_visible_ports/set_repeat`、`add/move/remove_period`、`clear_port/clear_all`、`replace_binding/cycle_binding(PulseFieldRef)`、`set_scan_sweep_count(int)`(:627-1038)。
- 异步意图:`generate_scan_source/load_scan_program/load_scan_array/select_scan_source/save_scan_array`(:1045-1163)、`new_document/open_path/save`(:1164-1205)、`set_preview_include_off/request_preview`(:1206-1264)、`connect/start/cancel/sync_applied/hold_scan_point/step_scan_point/request_scan_progress/request_close`(:1265-1682)。
- 发布/拉取:初始组合四件套 `editor_projection/file_update/preview_update/runtime_update`(:539-606);`pump() -> PulseOwnerUpdate`(六路 coalesced,:311-321, :1683)、`poll_runtime_change`、`set_notify(callable)` 唤醒钩(:481)、`runtime_poll_required/scan_progress_poll_required`(:1759-1780)。

**作为 presenter 对接纯视图,缺什么**:
- 域→plain 压平层。`PulseOwnerUpdate` 各分支携带 PulseDocument/PulseTargetManifest/FrozenScanTable/RunSnapshot/PulseScanProgress,zlc_ui view 一律不能吃。压平逻辑现散在 window.py 的 `_apply_*` 方法群(~1738 行的近半)+ scan_view 模块级 format 函数 + scan_workspace format 函数 —— 需正式收拢为 presenter 模块(吃 pump,产 plain record,调 view set_*)。
- 意图对向翻译:field 定位要从域 `PulseFieldRef` 变为 plain 三元组(如 `(period_id: str, port: str, field_kind: str)`),presenter 重建域引用。
**多什么**:几乎没有。controller 把 zlc_plot 类型放进发布物对 zlc_ui 无碍(presenter 消费后喂 RasterPlotHost,不过界)。真正放错位置的只有 scan_view.py 里两个 format 函数。

**session.py 角色**:controller 的纯域底座(document 单一可变 owner + save 冲突协议 + preview 编译),不见 view,拆分中零改动。

---

## 6) 综合:拆分账本

**进 zlc_ui(纯视图包)估算 ~4,600-4,900 行**:
- scan_line_edit.py 192(原样)
- preview_view.py ~230(去 DEFAULTS)
- scan_view.py ~320(剥离两个 format 函数约 80 行)
- target_view.py ~500(域调用换 record/rule 注入)
- _layout.py 159(几何 token,随视图走)
- schedule_view.py 2,892(非本人范围,按同类纯度假设大部平移)
- window.py 的纯壳部分(tab 组装/布局/对话框骨架)~300-500
- 前置依赖:zlc_frontend.qt_widgets 的 Fluent 控件底座须先进 zlc_ui(另计)。

**留域侧估算 ~4,400-4,700 行**:
- controller.py 2,596 + session.py 255 + scan_workspace.py 488 + preview_projection.py 260 + repeat_presentation.py 47
- window.py 的 presenter 半壁(压平/仲裁/文件对话执行/preview host 持有/format 函数收编)~800-1,100
- 新写:headless record 定义 + 双向翻译 ~500-800 行(净增)。

**最难三处解耦**:
1. **target_view 的域内嵌闭环**(target_view.py:462-577):Add/Remove/Apply 是"读回全部行→域校验→重 reconcile"的同步闭环,校验上移 presenter 会把一次闭环变成 view↔presenter 往返;须裁决哪些校验(width=endpoint 数、key 分配、占位模板)以注入参数形式留 view 保手感,哪些(manifest build、lane 一致性)必须上移。
2. **scan 草稿三路仲裁迁移**(window.py:1340-1391 ↔ scan_view.py:345-391):replace_source/loaded_program_completed/code_dirty/revision 四分支语义极细,目的就是不踩正在输入的编辑器;迁移必须逐分支保真,且验收要真 Qt 编辑器输入时序,不能只靠合成测试。
3. **preview host 生命周期跨界**(window.py:900-940, :1580-1595):`wait_for_front` 超时重试(10ms singleShot)、`subscribe_front`→owner 唤醒、`_track_preview_operation`/`_drain_preview_operations` 的 future→状态文本回写,横跨 presenter(持 host)与 view(挂载点/状态框);线程唤醒与挂载时序要在 presenter 重建,是唯一涉及跨线程时序的拆分点。

**接口过界违例现存清单**(重建时逐条消除):`PulseTargetView.applyRequested(PulseTargetManifest)` 与 `set_manifest(PulseTargetManifest)`(target_view.py:56, :147);scan_view :29 与 preview_view :18 的域常量 import;scan_view 模块级 format 函数吃 `PulseScanProgress`。其余接口(scan_view/preview_view/scan_line_edit 的信号与 set_*)已符合 zlc_ui 契约。