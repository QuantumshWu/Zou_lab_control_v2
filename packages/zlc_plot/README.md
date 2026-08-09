# zlc-plot

`zlc-plot` 提供 Zou Lab Control 的 Matplotlib 可视化接口。安装单元只发行
`zlc_plot`；顶层 `zlc_data` 名称保留给角色轴数据仓 `zlc-data`，本仓不会再安装
同名顶层包。

科学数据对象由独立的角色轴 `zlc-data` 包提供；`zlc_plot` 不再捆绑第二份数据模型。
`zlc_plot` 本身提供 plot specification、Matplotlib renderer、selector、fit、
固定尺寸、static/live session，以及 Notebook canvas / PyQt5 QImage adapter。

外部代码只通过公开 API 提交数据、修改显示参数、读取 selector/fit 结果或嵌入 canvas。Notebook 和 GUI 使用同一个 `PlotSession` 语义。
顶层 `zlc_plot` facade 只放常规使用路径；模型注册、参数 schema、底层 raster
mapping 等扩展接口分别位于 `zlc_plot.fit`、`zlc_plot.parameters`、
`zlc_plot.raster`、`zlc_plot.ui` 和 `zlc_plot.layout`，避免把扩展作者 API
与普通调用混成一个平面命名空间。

## 安装

Headless 与静态绘图：

```bash
python -m pip install -e .
```

Notebook 交互：

```bash
python -m pip install -e ".[notebook]"
```

从仓库根目录启动 JupyterLab，示例 Notebook 会优先加载当前 `src`：

```bash
cd zlc_plot
python -m jupyter lab
```

PyQt5 嵌入：

```bash
python -m pip install -e ".[qt]"
```

全部可选运行依赖：

```bash
python -m pip install -e ".[all]"
```

本仓库钉住 Matplotlib 3.10.8；修改依赖或 Notebook 前端代码后，必须在当前
Jupyter kernel 所用的解释器中重新执行 editable 安装并重启 kernel/Lab view，避免
旧的 widget registry 或旧的 Python 模块留在进程里：

```bash
python -m pip install --upgrade --force-reinstall -e ".[notebook]"
```

Notebook 和 Qt 依赖均惰性加载。普通 `import zlc_plot` 不依赖 Jupyter 或 Qt；Qt adapter 只支持 PyQt5。

本仓可以与角色轴数据包同时安装：

```bash
python -m pip install -e ../zlc_data
python -m pip install -e .
```

此时 `import zlc_data` 和 `import zlc_plot` 使用不同命名空间；旧契约若仅用于当前
绘图层直接接收 `zlc_data.OwnedSnapshot`。

## 创建数据与静态图

所有常规科学数据使用 `(R, P, *data_dim)` shape。下面的例子是两个 repeat、41 个 PointTable rows、没有额外 data dimension：

```python
import numpy as np

from zlc_data import (
    AxisId, AxisSpec, DatasetSchema, PointColumn, PointTable,
    REPEAT, SCAN_POINT, COMPONENT, ValidityContract, ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_plot import AxisRef, PlotLabels, curve

x = np.linspace(-2.0e-3, 2.0e-3, 41)
points = PointTable(41, (
    PointColumn(AxisId("scan"), "scan", SCAN_POINT, PointColumn.NUMERIC, tuple(x), "V"),
))
repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
value_axis = AxisSpec(AxisId("value"), "value", COMPONENT, 1, (0,))
schema = DatasetSchema(
    repeat, points, None,
    ValueSchema((value_axis,), ValidityContract.value(), np.dtype("float64"), "V"),
)
values = np.stack([
    0.3e-3 + 2.0e-3 * np.exp(-0.5 * (x / 0.6e-3) ** 2),
    0.32e-3 + 2.0e-3 * np.exp(-0.5 * (x / 0.6e-3) ** 2),
])
snapshot = owned_snapshot_from_arrays(schema=schema, values=values[..., None], revision=0)

session = curve(
    snapshot,
    AxisRef.point("scan"),
    labels=PlotLabels(title="Signal", x="Scan", y="Signal"),
    size="2x2",
)
session.save("curve.png")
```

## Notebook 交互

Notebook 使用 `zlc_plot.raster.RasterFront` 与一个标准 `anywidget` DOM canvas，不生成额外的按钮、下拉框或参数面板。安装 `zlc-plot[notebook]` 后，导入并构造 `zlc_plot.notebook.NotebookView` 时会自动加载这个 adapter，不需要运行 `%matplotlib widget` 或任何 IPython Matplotlib magic：

```python
from zlc_plot import SelectorKind, show

view = show(session)
```

`display()` 是唯一的 Notebook 输出入口，同一个 view 重复调用不会再创建输出。固定尺寸与 DPR 变化只替换完整的物理像素 front，仍复用同一个 widget；每个 front 以单个 `frame_packet` buffer 原子携带 RGBA、尺寸、DPR 与 interaction map，fit/live comm 到达时不会把 base 暂时清空。Area 拖拽候选和提交态都由 kernel 的同一套 Matplotlib renderer 烘焙进完整 front；浏览器只回传 press/release 与节流 move，不绘制几何。其它浏览器输入只回传归一化指针事件，和 Qt 使用同一 `RasterPlotHost` 协议。图上的 selector、包提供的滚轮缩放/拖动平移与代码调用共享同一个 session state：

- 任意主图左键拖动都会直接创建一个完整矩形 area selector，不需要先调用 enable API；每种 selector 在一个 session 中最多存在一个。
- area 的中心区域用于整体移动，四边与四角的八个 handle 用于改变范围。Curve、Rolling、Histogram、PulseTimeline 和 1D facet 按 x 范围选数据；Image 和 image facet 按 x/y 范围选数据。
- 已有 area 时在主轴空白处单击会删除它；`Esc` 或离开画布只取消本次未完成手势并恢复 committed area。
- 右键设置 crosshair、双击右键删除；滚轮缩放；中键拖动平移；双击中键始终恢复完整 home view；`Esc` 或 `session.cancel_interaction()` 取消未完成手势。
- FacetGrid overview 中双击 cell 进入 focus view；在 focus 中再次双击或按 `Esc` 返回网格，也可调用 `session.show_facet_overview()`。
- Image 的 side distribution 上可直接拖动 low/high color-limit handle。

```python
session.set_x_selector(-1.2, 1.2, display=True)
display_range = session.selector_state(SelectorKind.X_RANGE, display=True)
selected = session.selector_data(SelectorKind.X_RANGE)
fit_result = session.fit("gaussian_offset")

session.set_axis_unit(AxisRef.point("scan"), "V")
session.set_value_unit("V")
session.set_size("2x4")
```

每种 selector 最多存在一个；再次设置同一种 kind 会原子更新它。Fit 的唯一默认范围优先级是 `AREA > X_RANGE > viewport > all`；`selector_kind=` 也可显式绑定 Area、X-range 或 Threshold。Histogram bimodal fit 的 crossover 作为唯一有效的 Threshold selector 显示并参与命中，用户拖动后转为 authored Threshold；它默认只负责显示/分类，不反馈裁剪生成它的 fit。所有具有数值 fit 语义的 plot kind 共用一条 `FitSelection -> FitEngine -> overlay/presentation` 生命周期：Curve/Rolling 使用当前 DataView 的第一条 painted series，Histogram 使用当前 painted bin centers/counts，Image 使用当前 painted scalar field，FacetGrid 只把输入投影委托给当前 focused cell 的 Curve/Histogram/Image 语义。group、reduction 与 valid mask 只在 DataView 中评估一次，selector/viewport 随后筛选实际显示的 projection；Fit 不会另建 raw tensor mask/reduction 路径。Rolling 还会限制在当前可见 window。selector/viewport 只决定参数估计使用的样本，成功的拟合曲线仍覆盖当前完整显示域，并以当前显示单位写出公式、参数值和 `±` 不确定度。`FitResult` 保存 canonical 参数、covariance 和 data revision；`subscribe_fit()` 收到的 `FitEvent.formula` 与 `display_parameters` 则是图上正在使用的显示公式、单位和不确定度。Area/X 拖动中的 draft 默认每 30 ms 只更新视觉场景；selector、viewport、unit 和 resize 不会启动拟合，旧 overlay 保持稳定并在上下文变化时标为 lagging。只有显式调用 `fit()`，或 live fit 已 armed 且出现新的 data revision，才会求解并原子替换完整结果。Image 的 color-limit handles 属于显示色阶控制，不是 data selector；拖动时 handles 与色阶实时更新，较贵的 raster preview 最多每 100 ms 运行一次，释放时提交精确终值。删除由 `selector_kind=` 显式绑定的 selector 会取消并关闭该 live fit；自动选择的 live fit 则按相同优先级继续。Histogram 只对 count projection 执行 fit；启用 `density` 或 `cumulative` 时必须先切回 count。普通数据的 Selector API 提供 canonical/display range；只有显式调用 `selector_data()` 时才从调用当时的最新 snapshot 计算 mask、indices、坐标、数据值和 revision；crosshair 只返回显示坐标中距离光标最近的一个有效样本。PulseTimeline 返回 `PulseTimelineSelectionData`（与所选时间范围相交的 blocks、analog traces、scan regions、DAC segments 与 repeat markers）。selection event 本身不切片或缓存数据。

`session.fit_models` 与 `plot_host.fit_models()` 只返回当前 plot 语义和坐标单位都兼容的模型，并把该语义的默认模型排在第一位。Curve/Rolling 提供 Lorentzian、Gaussian with offset、symmetric Lorentzian doublet、damped sine 和 exponential decay；Histogram 提供 bimodal 与 single Gaussian；Image 仅在 x/y 坐标量纲兼容时提供 radial Gaussian center；PulseTimeline 不伪造可用的数值 fit。

Live fit 的唯一自动触发源是新的 data revision：`update_data()` 先发布新 data front并撤下旧 revision 的 fit overlay，再取消旧 solver、后台拟合当前最新 revision；fit 晚到时仍须通过 revision/request-generation 校验。selector、viewport、unit 和 resize 只更新交互/显示层；需要立即按新选择重算时显式调用 `fit()`。

Rolling 的静态快照也保留 R 轴的逐 shot 历史种子，因此 static 与 live 共用同一
projection 语义；live 端仍以严格递增 revision 和 capacity-one latest-only lane
提交新快照。Notebook 拖拽候选由 kernel 烘焙进单一 raster front，详见
[`docs/acceptance-decisions-2026-08-04.md`](docs/acceptance-decisions-2026-08-04.md)。

唯一的可执行教程 [`notebooks/usage.ipynb`](notebooks/usage.ipynb)
包含六种 plot kind、交互 selector、fit、单位/limits/labels 热更新、Image point
overlay、保存、真正持续更新且 non-block 的 live plot，以及从 Notebook 启动
PyQt5 窗口。

## 快速显示参数更新

Notebook 直接修改已有 plot surface；Qt 控件把同一调用提交给 raster host：

```python
session.set_parameter("show_grid", True)
session.set_labels(title="Signal", x="Delay", y="Response")
session.set_relim_mode("fixed")
session.set_x_limits(-1.0, 1.0)  # current display unit
widget.host.set_parameter("bin_count", 80)
```

Histogram 的 `bin_count`、Rolling 的 `window`、Image 的 colormap、
color limits 与 colorbar visibility 等都由对应 session 的唯一
`parameter_schema` 声明。`describe_display()` 同时给出当前状态、固定尺寸选项、
当前 limits 以及单位/colormap 等动态选择域；Image colormap 的封闭选择集是
`inferno`、`viridis`、`magma`、`plasma`、`gray`，前端无需复制合法值：

```python
from zlc_plot import parameter_controls

description = session.describe_display()
controls = parameter_controls(
    description.parameter_schema,
    description.display_state.values,
    choice_overrides=description.parameter_choices,
)
```

核心 schema 持有名称、类型、默认值、范围、合法选项和 render impact；
`zlc_plot.ui` 只把它投影成 toolkit-neutral control description，PyQt5 子模块再映射成
实际 widgets。具体应用只负责页面布局和业务流程。`describe_semantics(schema, spec)`
及 `session.describe_semantics()` 从 kind registry 机械生成 kind、AxisRef、group、
reduction、samples、facet_rows/facet_cols 的编辑域；`zlc_plot.ui.semantic_controls()` 复用同一
control 管线。拥有完整表单状态的宿主一次调用
`session.configure(...)` / `RasterPlotHost.configure(...)`，同时提交 semantic mapping、
display mapping、size、Image overlay 和 fit choice；宿主不判断原位更新还是重排。
`zlc_plot` 比较当前状态、合并 `RenderEffect`，并保留同一个 Figure。
`replace_spec()` 仍供已经拥有完整 typed `PlotSpec` 的代码直接使用。

参数 schema 用 `RenderEffect` flags 声明每个修改真正失效的投影、geometry、style、
axis transform、text/chrome、overlay 或 layout。一次 transaction 合并所有 effects，
复用原有 Figure 与 artists，并且只做一次最终 paint；普通交互使用完整 Agg draw，
Image payload 更新可以只把 image artist 作为单独 axis layer paint。文字更新只修改
现有 title/axis-label/colorbar-label artists，不重新投影数据或替换 data artists；grid
和 colorbar visibility 属于独立 chrome lane。
Curve、Rolling 与 Histogram 提供 `set_y_limits()` / `reset_y_limits()`；Image
提供 `set_color_limits()` / `reset_color_limits()` / `resolved_color_limits()`，这些图都支持
normal/tight/fixed relim。`set_view_limits()` 可原子修改当前显示单位下的 x/y viewport。
Histogram 的 `normal` / `fixed` bin domain 在 live revision 间只向越界一侧扩展，
不会缩回或重新居中；`tight` 才逐帧重新贴合当前数据。

## Live plot

Static plot 是只接收一个 immutable payload 的 session；live plot 则通过同一个 session/render API 持续提交新 revision，因此 selector、viewport、单位、显示参数和 fit 的行为一致。常规 `(R, P, *data_dim)` live plot 直接发布固定 schema/generation、严格递增 revision 的 snapshot：

```python
import asyncio
from zlc_data import owned_snapshot_from_arrays

from zlc_plot import DEFAULTS

live = view.live_controller(
    schema,
    refresh_interval_ms=DEFAULTS.live.default_refresh_interval_ms,
)
live.start()
live_stop = asyncio.Event()

async def publish_live():
    revision = session.data_revision
    while not live_stop.is_set():
        revision += 1
        live.publish(owned_snapshot_from_arrays(
            schema=schema,
            values=values * (1.0 + 0.08 * np.sin(0.2 * revision)),
            revision=revision,
        ))
        await asyncio.sleep(DEFAULTS.live.refresh_intervals_ms[0] / 1000.0)

live_task = asyncio.create_task(publish_live())
```

这段启动格立即返回，canvas 随后持续可见地更新。停止时先终止 producer，
再以非阻塞请求停止 consumer；Notebook event loop 继续运行到 worker 退出：

```python
live_stop.set()
await live_task
live.stop(wait=False)
while live.worker_alive:
    await asyncio.sleep(0.01)
live.close(timeout=0)
```

每个 cadence tick 只接纳 capacity-one ingress 中的 latest revision。新 data projection 完成后立即提交；若 live fit 已启用，同一 session 随后取消旧 solver并只拟合当前 revision，完成后以单独 overlay front 发布。新 data 绝不等待慢 fit，也不会带着旧 revision overlay；中间 producer revision 继续由 capacity-one ingress 合并。Area/X 拖动只更新交互场景，不进入 fit worker；只有显式 fit 或新的 data revision才会求解。只有应用显式 stop/pause 时，尚未呈现的 active revision 才会作为单个 latest 值保留给 resume/pump。可直接运行的
PyQt5 持续 live 窗口是：

```bash
python examples/live_simulation.py
```

`PulseTimelineData` 本身不携带 producer revision；live transport 使用独立 immutable envelope，避免把可变采集状态塞进可复用 payload：

```python
from zlc_plot import LiveDataRevision, LivePlotController

initial = LiveDataRevision(revision=0, payload=pulse_data)
pulse_live = LivePlotController(pulse_session, initial)
pulse_live.publish(LiveDataRevision(revision=1, payload=pulse_data))
pulse_live.pump_once()  # 或 start() 后按配置 cadence 消费
assert pulse_session.data_revision == 1
pulse_live.close()
```

Pending capacity 固定为 1。Producer 不等待 render；当生产速度高于显示刷新时，只保留最新 revision。Envelope revision 会原样成为 session、selection event 与 selected data 的 data revision；PulseTimeline 直接调用 `update_data(pulse)` 时从当前 revision 自动加一。有无 automatic live fit 都使用同一个 `update_data()`；`LivePlotController.publish()` 只增加 capacity-one ingress 和 100/200/400/800 ms cadence（默认 400 ms，最高 10 Hz），不增加另一套 fit 状态机。

`stop()` 停止 consumer 但不关闭 controller，可用 `start()` 恢复；`close()` 是终止并释放 ingress 的操作。Notebook 的持续 producer 与完整 cleanup 写法见唯一的 `notebooks/usage.ipynb`。

鼠标 motion cadence、selector 命中半径和滚轮缩放倍率集中在
`DEFAULTS.interaction`，与 live cadence 一样可以通过完整 defaults 对象配置。

## PyQt5 嵌入

`Qt5PlotWidget` 是只显示 immutable QImage front 的 raster adapter。外部应用拥有按钮、参数控件和状态显示，并将信号连接到异步 `RasterPlotHost` API；Matplotlib session 在专用 worker 创建。`Qt5ParameterPanel` 可直接消费 `describe_display()`，不要求应用重复维护参数名、类型或选项：

```python
from zlc_plot import (
    AxisRef,
    CurvePlot,
    Qt5ParameterPanel,
    Qt5PlotWidget,
    RasterPlotHost,
    ensure_qt5_application,
)

app = ensure_qt5_application()
plot_host = RasterPlotHost.from_plot(
    snapshot,
    CurvePlot(AxisRef.point("scan")),
    size="2x2",
)
widget = Qt5PlotWidget(plot_host)
description = plot_host.describe_display().result().value
parameters = Qt5ParameterPanel(description)
parameters.parameterEdited.connect(
    lambda name, value: plot_host.set_parameter(name, value)
)
parameters.semanticEdited.connect(
    lambda name, value: plot_host.replace_spec(next_spec(name, value))
)
widget.show()
try:
    app.exec_()
finally:
    widget.close_adapter()
    plot_host.close()
```

只有确实需要自定义 `PlotSession` 子类时才使用 `RasterPlotHost(factory)`；普通 GUI
集成统一走 `from_plot()`，不在应用里重复 session factory 样板。

GUI 可从 `plot_host.fit_models()` 读取 session 的公开 fit catalogue，把
`FitModelSpec.display_name/model_id` 填入下拉框，再将选中的 model id 传给
`plot_host.fit(...)`；返回值已经按 Series、Histogram 或 Image 语义以及单位兼容性过滤，
默认项在第一位，UI 不再维护 arity 或 preferred-model 映射。

`plot_host.selectors()` 与 `plot_host.selector_state(SelectorKind.AREA, display=True)` 只读取 selector
geometry；`plot_host.selector_data(SelectorKind.AREA)` 才在显式调用时从最新 snapshot 物化 mask、
indices、coordinates 和 values。GUI 因而不需要维护第二份 selector range，也不会为了
显示范围提前切片数据。

Pulse preview 或嵌套 scroll area 可调用 `widget.set_interaction_enabled(False)`；这只关闭 adapter input transport，不重建 Figure/host，也不停止 live、resize、DPR 或参数更新。需要 same-shot group display 时，各 widget 使用 `auto_present=False`，应用按自己的 causal shot identity 等齐各 host 返回的 `RasterOperation.front`，再在 Qt owner thread 调用每个 widget 的 `present_front(front)`。`zlc_plot` 不用恰好相等的 revision 猜测 same-shot；join 仍属于应用。`front.identity` 同时携带 dataset generation/revision 和 Image overlay revision 供应用核对。

`widget.presented_front.buffer.save(path)` 保存当前屏幕已经显示的准确物理 RGBA 像素，不触发重绘；`plot_host.save(path, dpi=...)` 则按指定 DPI 正式重绘导出。

必须在创建首个 `QApplication` 前调用 `ensure_qt5_application()`。它统一配置 PyQt5 High-DPI 属性并注册包内 Helvetica Light；固定 preset 的逻辑尺寸不变，而高 DPR 屏幕使用更多物理像素绘制同一 surface。首帧、live 更新和 preset 切换都通过 worker 内 session 的同一重绘路径进入 Qt QImage front。Live producer 使用 `plot_host.live_controller(initial)`，不会把 session 暴露给 UI thread。

完整的外部控件、selector range/data、fit、单位、固定尺寸与 live 连接示例：

```bash
python examples/pyqt5_embed.py
```

常规热更新与 1024²/2048² monitor-camera live profiling 结果及复现命令见
[`docs/performance.md`](docs/performance.md)。

## Plot kinds 与固定尺寸

公开的六种 plot kind 是 Curve、Image、Histogram、Rolling、FacetGrid 和 PulseTimeline。Rolling 通过 `side_distribution` 参数选择是否显示 side distribution。FacetGrid 可沿 repeat、PointTable row/coordinate、GridTopology dimension 或 `data_dim` 展开；同一个 grid 的 cells 使用同一种 Curve、Image 或 Histogram kind。
一维 FacetGrid 使用 `facet_display_unit`；二维 rows×columns grid 分别使用
`facet_row_display_unit` 和 `facet_col_display_unit`，因此两个 facet 轴不会被错误地
强制共用单位。

坐标标记不是单独的 plot kind。普通 Image 可叠加独立、可动态更新的 `ImagePointOverlay`；`coordinates` 是 canonical x/y 的 `N×2` 数组，ID、label 与 `PointStatus` 都可选。仅修改点层时可独立推进 overlay revision，不重投影 background：

```python
overlay = ImagePointOverlay(
    revision=0,
    coordinates=np.array([[-2.0e-3, 1.0e-3], [0.0, -1.0e-3]]),
    point_ids=("a", "b"),
    statuses=(PointStatus.EMPTY, PointStatus.OCCUPIED),
)
image_session.update_image_overlay(overlay)
image_session.set_parameter("show_point_labels", True)
```

点环尺寸和状态颜色由 package style 统一解析，不属于可变显示参数。

同一 shot 的 image 与动态坐标/状态必须作为一个 `ImageFrame` 发布；frame revision 只来自 snapshot，不建立第二个 frame 时钟。data 与 overlay 作为同一 frame 提交；live fit随后只针对该当前 revision运行。overlay 自身仍保持单调 revision；清空点层使用 `ImagePointOverlay.empty(revision)`。如果 frame 在后台准备期间点层被独立更新，旧 frame 的 CAS 会失败，而不是覆盖较新的点层：

```python
from zlc_data import owned_snapshot_from_arrays
from zlc_plot import ImageFrame

frame = ImageFrame(snapshot, overlay)
image_live = image_view.live_controller(frame).start()
next_frame = ImageFrame(
    owned_snapshot_from_arrays(schema=schema, values=values, revision=1),
    overlay,
)
image_live.publish(next_frame)
```

Image 的 equal-aspect 使用 canonical 单位的物理比例；例如 x 以 nm、y 以 µm 显示时，
屏幕上的同一物理半径仍是圆，而不是把两个显示数值强行当作同一尺度。drag、wheel zoom、
API viewport 和 reset 都只修改同一个 limits authority，不移动第二张缩略图或副本。
只有 x/y 量纲兼容时才应用物理比例；不兼容时回退 numeric aspect，并从该 session 的
catalogue 中过滤 radial fit。

PulseTimeline 的公开输入由 `PulseTimelineData` 组合 `PulseChannel`、`PulseBlock`、`PulseAnalogTrace`、`PulseScanRegion`、`PulseDacScanSegment` 和 `PulseRepeatMarker`。这些 records 分别描述 digital channels、digital blocks、analog traces、scan regions、DAC scan segments 和 repeat brackets。所有 timeline 时间必须非负；`PulseScanRegion.number` 和已提供的 DAC scan `number` 必须是全 timeline 内唯一的正整数。

Plot 只允许以下九个 named preset：

```text
1x2  2x2  4x2  1x4  2x4  4x4  4x8  8x4  8x8
```

Notebook 和 Qt 都消费同一个 `SurfacePlan`。宿主窗口或浏览器区域 resize 只改变外围留白；只有 `session.set_size(name)` 会改变 authored plot size、axes geometry 和 FacetGrid font tier。DPR 只增加对应 preset 的物理像素数，不改变逻辑布局或字体 tier。

## 持久化与应用边界

- `zlc_data.save_npz/load_npz` 持有科学数据 snapshot 的 NPZ 格式。
- 当前已经显示的 Edit-tab snapshot 直接使用 immutable `widget.presented_front`；它包含准确的 RGBA、surface identity 与 interaction transform，不触发重绘。需要独立交互或 local fit 的 Edit surface，则由应用用冻结的 `zlc_data.OwnedSnapshot`、原 `PlotSpec` 和应用持有的 authored parameters 创建另一个 `RasterPlotHost`。这与 live panel 隔离，且不需要复制运行中 session 或恢复异步句柄。
- plot specification、authored panel parameters、same-shot/EventRef、archive codec、文件路径、设备配置、设备调用关系、Logic route、实验 workflow 与项目文件均由上层应用持有；`zlc_plot` 不定义第二套项目格式。

API 与数据契约：

- [`docs/api.md`](docs/api.md)
- [`docs/data_contract.md`](docs/data_contract.md)
- [`docs/architecture.md`](docs/architecture.md)
