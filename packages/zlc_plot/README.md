# zlc-plot

`zlc_plot` 是单一 Zou Lab Control distribution 内的可视化层，不是独立 wheel。
科学数据对象由同一产品中的 `zlc_data` 角色轴层提供；两者保持依赖边界，但只安装
一个 `zou-lab-control` 产品。
`zlc_plot` 本身提供 plot specification、Matplotlib renderer、selector、fit、
固定尺寸、static/live session，以及 Notebook canvas / PyQt5 QImage adapter。

外部代码只通过公开 API 提交数据、修改显示参数、读取 selector/fit 结果或嵌入 canvas。Notebook 和 GUI 使用同一个 `PlotSession` 语义。
轴与semantic字段只认`AxisRef(domain, axis_id)`稳定key；axis label是显示文本，不能作为
持久化或表单identity。Scope使用tagged latest或tagged typed coordinate value；因此文本坐标
`"latest"`仍是普通坐标，不会被解释成selector。Figure recipe使用同一tagged grammar。
顶层 `zlc_plot` facade 只放常规使用路径；模型注册、参数 schema、底层 raster
mapping 等扩展接口分别位于 `zlc_plot.fit`、`zlc_plot.parameters`、
`zlc_plot.raster`、`zlc_plot.ui` 和 `zlc_plot.layout`，避免把扩展作者 API
与普通调用混成一个平面命名空间。

## 安装

从产品根目录按唯一constraints安装；Notebook能力使用同一distribution的extra：

```bash
python -m pip install -c constraints.txt -e ".[notebook]"
```

正式教程只有 `packages/zlc_workbench/notebooks/usage.ipynb`；release gate把tracked
document读入内存，并从checkout外的临时工作目录用fresh kernel执行。执行后的内存
document会被丢弃，不写回source。Notebook依赖惰性加载；普通`import zlc_plot`不会
加载Jupyter，Qt adapter只支持产品钉住的PyQt5。

## 创建数据与静态图

所有常规科学数据使用 `(R, P, *cell_shape)` shape。Repeat、Point和
Cell-data三个domain都直接拥有自己的axes；下面的例子是两个repeat、41个
Point rows和一个scalar cell：

```python
import numpy as np

from zlc_data import (
    AxisId, AxisSpec, DatasetSchema, DomainSpec,
    REPEAT, SCAN_POINT, COMPONENT, ValidityContract, ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_plot import AxisRef, PlotLabels, curve

x = np.linspace(-2.0e-3, 2.0e-3, 41)
scan = AxisSpec(AxisId("scan"), "scan", SCAN_POINT, 41, tuple(x), "V")
points = DomainSpec((41,), (scan,), (tuple(range(41)),))
repeat_axis = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
repeats = DomainSpec((2,), (repeat_axis,), ((0, 1),))
value_axis = AxisSpec(AxisId("value"), "value", COMPONENT, 1, (0,))
schema = DatasetSchema(
    repeats,
    points,
    DomainSpec((1,), (value_axis,)),
    ValueSchema(ValidityContract.value(), np.dtype("float64"), "V"),
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

## Data-backed Figure artifacts

`save_figure_artifact(...)` is the common TaskConsole/domain-report path. It
atomically writes a `zlc.figure` NPZ first, then renders the same-stem PNG
preview. The NPZ contains the typed Dataset, exact PlotSpec recipe, complete
normalized parameters, viewport, classifier/fit configuration, image overlay and causal
lineage. `read_figure_plot(...)` restores that typed input and recipe without
inferring a plot kind from array shape.

`build_figure_host`和`open_figure_host`是TaskConsole、
Panel Save与FigureViewer共享的host路径。FigureViewer因此重现保存时的语义，而不是
重新选择一个“看起来合适”的图。Reader只接受当前完整recipe，不接受alternate alias。

## Notebook 交互

Notebook 使用 `zlc_plot.raster.RasterFront` 与一个标准 `anywidget` DOM canvas，不生成额外的按钮、下拉框或参数面板。安装 `zou-lab-control[notebook]` 后，导入并构造 `zlc_plot.notebook.NotebookView` 时会自动加载这个 adapter，不需要运行 `%matplotlib widget` 或任何 IPython Matplotlib magic：

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

每种 selector 最多存在一个；再次设置同一种 kind 会原子更新它。Fit 的唯一默认范围优先级是 `AREA > X_RANGE > viewport > all`；`selector_kind=` 也可显式绑定 Area 或 X-range。Distribution 的 `threshold_classifier` 是独立的显示/分类功能：启用后自行求出初始最优 threshold，绘制左右 Gaussian、总和、可拖动 threshold 线以及当前 L/R population 和 fidelity。普通 `bimodal_gaussian` fit 不会创建、移动或清除 classifier，classifier 也不写普通 fit 状态。所有具有数值 fit 语义的 plot kind 共用一条 `FitSelection -> FitEngine -> overlay/presentation` 生命周期：Curve/Rolling 使用当前 DataView 的第一条 painted series，Histogram 使用当前 painted bin centers/counts，Image 使用当前 painted scalar field，FacetGrid 只把输入投影委托给当前 focused cell 的 Curve/Histogram/Image 语义。group、reduction 与 valid mask 只在 DataView 中评估一次，selector/viewport 随后筛选实际显示的 projection；Fit 不会另建 raw tensor mask/reduction 路径。Rolling 还会限制在当前可见 window。selector/viewport 只决定参数估计使用的样本，成功的拟合曲线仍覆盖当前完整显示域，并以当前显示单位写出公式、参数值和 `±` 不确定度。`FitResult` 保存 canonical 参数、covariance 和 data revision；live data的`FitEvent`在exact solve接受后即可发布给Rolling等数据消费者，不等待较慢的owner raster，但主Panel仍只把`data@N + fit@N`画进同一个atomic front；显式manual `fit()`仍在accepted overlay transaction后通知。Area/X 拖动中的 draft 默认每 30 ms 只更新视觉场景；selector、viewport、unit 和 resize 不会启动拟合，旧 overlay 保持稳定并在上下文变化时标为 lagging。只有显式调用 `fit()`，或 live fit 已 armed 且出现新的 data revision，才会求解并原子替换完整结果。Image 的 color-limit handles 属于显示色阶控制，不是 data selector；拖动时 handles、色阶与 raster preview 跟随指针节拍实时更新（只重着色图像像素，不触碰 chrome），释放时提交精确终值。删除由 `selector_kind=` 显式绑定的 selector 会取消并关闭该 live fit；自动选择的 live fit 则按相同优先级继续。Histogram 只对 count projection 执行 fit；启用 `density` 或 `cumulative` 时必须先切回 count。普通数据的 Selector API 提供 canonical/display range；只有显式调用 `selector_data()` 时才从调用当时的最新 snapshot 计算 mask、indices、坐标、数据值和 revision；crosshair 只返回显示坐标中距离光标最近的一个有效样本。PulseTimeline 返回 `PulseTimelineSelectionData`（与所选时间范围相交的 blocks、analog traces、scan regions、DAC segments 与 repeat markers）。selection event 本身不切片或缓存数据。

Panel的单行Fit表达式使用当前显示单位，参数名就是公式里印出来的符号（`FitModelSpec.symbols`）：exponential decay 画的是 $f(x)=A e^{-x/\tau}+B$，所以写 `A=2` 把参数精确固定并从优化自由度移除，`tau=guess(5)` 只替换初始猜测；省略参数即保持Auto。PanelState与Figure只保存canonical `fixed`/`initial` mappings。表达式无效时忽略这份optional override、继续同model自动fit并显示warning；fixed参数显示为`(fixed)`且没有估计误差。

`session.fit_models` 与 `plot_host.fit_models()` 只返回当前 plot 语义和坐标单位都兼容的模型，并把该语义的默认模型排在第一位。Curve/Rolling 提供 Lorentzian、Gaussian with offset、symmetric Lorentzian doublet、damped sine 和 exponential decay；Histogram 提供 bimodal 与 single Gaussian，以及 single/bimodal Poisson-Gaussian（`histogram_poisson_gaussian`、`bimodal_poisson_gaussian`：泊松律经 Γ 函数延拓到实数光子数 $p(u)=\lambda^u e^{-\lambda}/\Gamma(u+1)$、归一化后与高斯读出噪声卷积，$f(x)=\frac{A}{\sigma\sqrt{2\pi}\,\int p}\int_0^\infty p(u)\,e^{-\frac{1}{2}((x-u)/\sigma)^2}du$，是 x 的光滑函数，和其他模型一样直接在 bin 中心与计数上拟合，不问数据来源；负值是读出噪声的正常结果而不是非法输入；每类参数为 A（面积=计数×bin 宽）、λ、σ，bimodal 以 λ_L 与 δ=λ_R−λ_L 参数化，headline 为 δ 即 bright−dark contrast；λ 低于约 3 光子时延拓律的均值高于 λ、拟合值偏低，低于 1 光子它已不是光子计数律）；Image 仅在 x/y 坐标量纲兼容时提供 radial Gaussian center；PulseTimeline 不伪造可用的数值 fit。

Live fit 的唯一自动触发源是宿主的通用 indexed-derived signal。只有真实Rolling/Histogram等history consumer取得window lease后，Runtime才从当时的current event开始记录；lease区间内每个Measurement primary index都在同一个普通Dataset中有value或invalid cell，之前的shot不回填。`display_interval`只控制Surface deadline。Host只保留一个active pair和一个latest完整输入，中间输入不排FIFO；现有Raster worker的active deadline超过1秒会loud发布invalid、取消该solve并继续latest。任何window/history按lease内source index连续，cadence skip与solver failure都显示为invalid/NaN，但只有后者是错误。主Panel的commit仍把`data@N + fit@N`原子画进同一front。

Rolling 的静态快照也保留 R 轴的逐 shot 历史种子，因此 static 与 live 共用同一
projection 语义；Runtime以严格递增revision提交新快照。Notebook拖拽候选由
kernel烘焙进单一raster front。

唯一可执行产品教程是`packages/zlc_workbench/notebooks/usage.ipynb`；它把真实
virtual Camera Measurement publication交给普通Image `NotebookView`并完整关闭owner。
其余plot kind和交互contract由本层API文档及自动测试覆盖，不再维护第二本教程。

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
调用方提交的是authored target；`describe_display()`返回的`DisplayDescription.spec`才是
本次成功transaction实际接受的spec。Live、Frozen/Edit和FigureViewer都必须以这个accepted
spec判断classifier、selector、overlay、viewport与其它resolved capability，不能用尚未接受的
target猜当前pixels。

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

`zlc_plot`只负责把调用方提交的immutable revision投影、拟合并生成
`RasterFront`。完整run history、publication cadence、Stop和Final属于
`zlc_runtime`；plot层不再维护第二个live channel、controller或worker。
Workbench通过`RasterPlotHost.update_data()`提交Runtime已经选定的revision。
Pulse timeline直接以不可变`PulseTimelineData`更新同一个session。每个session
只拥有一个串行analysis executor，prepare、manual fit和live fit不并行执行。

鼠标 motion cadence、selector 命中半径和滚轮缩放倍率集中在
`DEFAULTS.interaction`。

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
Selector与viewport callback都携产生该observation的Dataset generation和revision；应用只有在
它仍匹配当前accepted presentation时才能接受。`zlc_plot`不持久化应用interaction truth，也不
把host-local selector revision当成跨Live/Frozen surfaces的共同时钟。

Pulse preview 或嵌套 scroll area 可调用 `widget.set_interaction_enabled(False)`；这只关闭 adapter input transport，不重建 Figure/host，也不停止 live、resize、DPR 或参数更新。需要 same-shot group display 时，各 widget 使用 `auto_present=False`，应用按自己的 causal shot identity 等齐各 host 返回的 `RasterOperation.front`，再在 Qt owner thread 调用每个 widget 的 `present_front(front)`。`zlc_plot` 不用恰好相等的 revision 猜测 same-shot；join 仍属于应用。`front.identity` 同时携带 dataset generation/revision 和 Image overlay revision 供应用核对。

`widget.presented_front.buffer.save(path)` 保存当前屏幕已经显示的准确物理 RGBA 像素，不触发重绘；`plot_host.save(path, dpi=...)` 则按指定 DPI 正式重绘导出。

必须在创建首个 `QApplication` 前调用 `ensure_qt5_application()`。它统一配置 PyQt5 High-DPI 属性并注册包内 Helvetica Light；固定 preset 的逻辑尺寸不变，而高 DPR 屏幕使用更多物理像素绘制同一 surface。首帧、Runtime提交的数据更新和preset切换都通过worker内session的同一重绘路径进入Qt QImage front。

常规热更新与 1024²/2048² monitor-camera live profiling 结果及复现命令见
[`docs/performance.md`](docs/performance.md)。

## Plot kinds 与固定尺寸

公开的六种 plot kind 是 Curve、Image、Histogram、Rolling、FacetGrid 和 PulseTimeline。Rolling 通过 `side_distribution` 参数选择是否显示 side distribution。FacetGrid 可沿Repeat、Point或Cell-data任一具名axis展开；同一个grid的cells使用同一种Curve、Image或Histogram kind。
一维 FacetGrid 使用 `facet_display_unit`；二维 rows×columns grid 分别使用
`facet_row_display_unit` 和 `facet_col_display_unit`，因此两个 facet 轴不会被错误地
强制共用单位。

未经 authoring 时每种 kind 显示什么，由 `zlc_plot/_kinds/defaults.py` 一张表决定；每个 kind 的 `default_spec`、FacetGrid 的 cell kind 选择和「从当前 plot 要一个 grid」都只是对这张表的读取。表按 `classify_axes` 得到的 axis family 分组，从不按 axis 名字特判：R（repeat）是统计量，只被 reduce 或被 Histogram pool；H（Runtime 的 primary index）除 Rolling 自己走它之外也是统计量，只在其它轴都没有结构时作 curve 最后的 x；S（scan axis，slowest first）是位置：最内层是 curve 的 x，两层是 heatmap，最外层是 grid 的 facet，无人认领的 scan 轴保持可编辑的 Reduced；E（`READOUT_EVENT` Point axis，如 camera frame、survival pair）是子测量的选择：grid 给每个 event 一个 cell，无 scan 的 curve 沿它走，其它情况显示 Latest scope；D（Cell-data payload）是内容：声明的 picture 或两条 content 轴成 image，剩下的一条 content 在 palette 能分辨时成 group，否则 reduce。size为1的axis仍是provenance；其中event axis即使只有一个坐标仍可标识一个cell。`tests/test_default_roles.py`枚举全表。Limit 类 display 字段（relim 与 x/y/color 范围）声明为 `portable=False`：panel identity 改变时它们随 semantic/fit 一起从新 vocabulary 重新开始，只有外观字段跨 kind 携带。

坐标标记不是单独的 plot kind。普通 Image 可叠加独立、可动态更新的
`ImagePointOverlay`；`coordinates` 是 canonical x/y 的 `N×2` 数组，ID、label
可选。手写或Calibration标记使用一个不可变的`static_statuses`向量：

```python
overlay = ImagePointOverlay(
    revision=0,
    coordinates=np.array([[-2.0e-3, 1.0e-3], [0.0, -1.0e-3]]),
    point_ids=("a", "b"),
    static_statuses=(PointStatus.EMPTY, PointStatus.OCCUPIED),
)
image_session.update_image_overlay(overlay)
image_session.set_parameter("show_point_labels", True)
```

动态点状态不是plugin专用对象：producer给numeric/bool Dataset声明
`IMAGE_POINT_OVERLAY_CONTRACT`，用`image_point_overlay_geometry(...)`记录image
axes、XY坐标和完整status data axis，再由
`image_point_overlay_from_signal(...)`构造同一个`ImagePointOverlay`。Dataset
values表达EMPTY/OCCUPIED，Dataset validity表达INVALID；future-invalid或一个
surface仍pool多个repeat/point cells时显示UNKNOWN，不发明跨cells共识。geometry
严格绑定status axis identity与canonical coordinates，所以同数量但顺序不同的
site vector也会被拒绝。点环尺寸和状态颜色由package style统一解析，不属于可变
显示参数。

需要operator校对一组稳定point identities时，`ImagePointReviewSurface`
直接复用现有Image host与`ImagePointOverlay`，只负责单点/框选gesture和状态
overlay。搜索、checkbox、批量exclude/restore、buttons与window全部属于
`zlc_ui.console.PointReviewView`；Plot不复制image数据、不创建第二套geometry，
也不拥有应用的modal lifecycle。

同一 shot 的 image 与动态坐标/状态必须作为一个 `ImageFrame` 发布；frame revision 只来自 snapshot，不建立第二个 frame 时钟。data 与 overlay 作为同一 frame 提交；live fit随后只针对该当前 revision运行。overlay 自身仍保持单调 revision；清空点层使用 `ImagePointOverlay.empty(revision)`。如果 frame 在后台准备期间点层被独立更新，旧 frame 的 CAS 会失败，而不是覆盖较新的点层：

```python
from zlc_data import owned_snapshot_from_arrays
from zlc_plot import ImageFrame

frame = ImageFrame(snapshot, overlay)
next_frame = ImageFrame(
    owned_snapshot_from_arrays(schema=schema, values=values, revision=1),
    overlay,
)
image_session.update_data(next_frame)
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
- 当前已经显示的 Edit-tab snapshot 直接使用 immutable `widget.presented_front`；它包含准确的 RGBA、surface identity 与 interaction transform，不触发重绘。需要独立交互或 local fit 的 Edit surface，则由应用用冻结的`zlc_data.OwnedSnapshot`和Live host已接受的`DisplayDescription.spec`、normalized parameters创建另一个`RasterPlotHost`；不得从未接受的authored target重猜。这与live panel隔离，且不需要复制运行中session或恢复异步句柄。
- `zlc_data`拥有Figure NPZ grammar，`zlc_durable`拥有原子路径发布；`zlc_plot`拥有
  exact Plot recipe与archive-first/render-second公共流程。设备配置、Logic route、
  实验workflow与项目文件仍由上层应用持有，不建立第二套项目格式。

API 与数据契约：

- [`docs/api.md`](docs/api.md)
- [`docs/data_contract.md`](docs/data_contract.md)
- [`docs/architecture.md`](docs/architecture.md)
