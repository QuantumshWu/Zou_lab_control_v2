# API guide

当前文档中的`(R, P, *data_dim)`数据对象来自同一产品的角色轴`zlc_data`层。
`zlc_plot`只消费`zlc_data.OwnedSnapshot`，不建立第二份数据模型。常用类型和构造器由稳定的顶层`zlc_plot` facade公开。模型注册、参数 schema、
底层 raster mapping 等扩展接口分别由 `zlc_plot.fit`、`zlc_plot.parameters`、
`zlc_plot.raster`、`zlc_plot.ui` 和 `zlc_plot.layout` 公开，不重复平铺到顶层。

## Top-level names

The supported convenience namespace is deliberately small. Its executable
source of truth is `zlc_plot.__all__`; the examples below import only from that
facade. Extension interfaces remain in their owning submodules.

Fit-engine result records, raster packets, semantic internals and the remaining exception types
stay available from their owning submodules and are intentionally absent from
the facade. `FitCancelled` and `BackendUnavailableError` are the only
individually catchable optional-runtime exceptions exported at the top level.

## Constructing `(R, P, *D)` data

```python
import numpy as np
from zlc_data import (
    AxisId, AxisSpec, DatasetSchema, GridTopology, PointColumn, PointTable,
    REPEAT, SCAN_POINT, COMPONENT, ValidityContract, ValueSchema,
    owned_snapshot_from_arrays,
)

x = np.linspace(-2.0e-3, 2.0e-3, 41)
y = np.linspace(-1.5e-3, 1.5e-3, 25)
x_rows = np.tile(x, y.size)
y_rows = np.repeat(y, x.size)
points = PointTable(1025, (
    PointColumn(AxisId("x"), "x", SCAN_POINT, PointColumn.NUMERIC, tuple(x_rows), "V"),
    PointColumn(AxisId("y"), "y", SCAN_POINT, PointColumn.NUMERIC, tuple(y_rows), "V"),
))
topology = GridTopology(
    (AxisId("y"), AxisId("x")), (tuple(y), tuple(x)), tuple(np.ndindex(y.size, x.size)),
)
site = AxisSpec(AxisId("site"), "site", COMPONENT, 4, (0, 1, 2, 3))
schema = DatasetSchema(
    AxisSpec(AxisId("repeat"), "repeat", REPEAT, 8, tuple(range(8))),
    points,
    topology,
    ValueSchema((site,), ValidityContract.value(), np.dtype("float64"), "V"),
)
profile = (
    0.35e-3
    + 2.0e-3 * np.exp(-0.5 * (x_rows / 0.65e-3) ** 2)
    * np.exp(-0.5 * (y_rows / 0.9e-3) ** 2)
)
values = (
    profile[None, :, None]
    + 0.02e-3 * np.arange(site.size)[None, None, :]
    + 0.03e-3 * np.sin(np.linspace(0.0, 2.0 * np.pi, 8))[:, None, None]
)
snapshot = owned_snapshot_from_arrays(schema=schema, values=values, revision=1)
next_snapshot = owned_snapshot_from_arrays(schema=schema, values=values + 0.01e-3, revision=2)
assert snapshot.block.values.shape == (8, 1025, 4)
```

GridTopology is optional. Supply it only when the producer owns an explicit row-to-cell mapping.

## Convenience constructors

普通 Notebook 调用可直接使用六个顶层构造器；它们只组装对应的 typed plot spec，
返回的仍是同一个 `PlotSession`：

```python
from zlc_plot import AxisRef, curve, histogram, image, rolling

curve_session = curve(snapshot, AxisRef.point_dimension("x"), size="2x2")
image_session = image(
    snapshot,
    AxisRef.point_dimension("x"),
    AxisRef.point_dimension("y"),
)
histogram_session = histogram(snapshot, bins=80)
rolling_session = rolling(snapshot, window=64, side_distribution=True)
```

Rolling 的 `trailing=N` 决定每个画出的点平均多少枪：`1`（默认）就是那一枪
自己，`N` 是最近 N 枪的均值，band 也随之变成这 N 枪的标准误。它读的是保留
下来的矩，因此一枪 pool 的样本越多权重越大——这是均值的算术，所以在非 MEAN
reduction 上它不起作用。窗口未填满时它就是「至今为止的均值」，不会有跳变。
`window=` 仍然只决定看得见多少枪、永不改变数字；当 `trailing` 比 `window`
更靠后时，runtime 的历史保留量按 `trailing` 计。

`curve(...)`、`image(...)` 与 `rolling(...)` 都通过 `reduction=` 暴露与各自
typed specification 相同的 reduction 选择；需要 topology dimension 时传入显式
`AxisRef.point_dimension(...)`，字符串只表示 PointTable coordinate。

需要组合 FacetGrid、自定义 reduction 或长期保存 plot specification 的代码，使用下面的
typed spec 接口。两种入口共享同一套 parameter schema、selector、fit、live 和 backend
实现。

## Plot specifications

```python
from zlc_plot import (
    AxisRef,
    CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot,
    PulseTimelinePlot, RollingPlot,
)

scan_x = AxisRef.point_dimension("x")
curve = CurvePlot(scan_x)
image = ImagePlot(AxisRef.point_dimension("x"), AxisRef.point_dimension("y"))
histogram = HistogramPlot()
rolling = RollingPlot()
grid = FacetGridPlot(AxisRef.data("site"), CurvePlot(scan_x))
pulse = PulseTimelinePlot()
```

Axis references are explicit:

- `AxisRef.repeat()`
- `AxisRef.point_rows()`
- `AxisRef.point("column")`
- `AxisRef.point_dimension("topology_dimension")`
- `AxisRef.data("data_axis")`

FacetGrid accepts a required `facet_rows` source and an optional `facet_cols`
source, producing a row-major two-dimensional grid of homogeneous
Curve/Image/Histogram cells. The row and column sources must differ from each
other and from every cell axis, group, or histogram sample.
One-dimensional facets expose `facet_display_unit`; two-dimensional facets
expose independent `facet_row_display_unit` and `facet_col_display_unit`
parameters so each facet axis can be converted and labelled separately.
Histogram `samples` is an explicit pool contract: every repeat, point or trailing
data axis that contributes observations must be listed (or consumed by the facet).
Rolling owns its horizontal history ordinal inside `PlotSession`; it has no dataset
`x` axis.  Change semantic roles on an existing surface with
`session.replace_spec(new_spec)`, and use `session.fit(..., fit_all_facets=True)`
for an ordered `zlc_plot.fit.FacetFitBatchResult`.

## Image point overlays and PulseTimeline inputs

Coordinate annotations are an independently revisioned overlay on an ordinary
Image. They do not replace the Image's `(R, P, *data_dim)` snapshot:

```python
import numpy as np
from zlc_plot import (
    ImageFrame, ImagePointOverlay, PointStatus,
)

overlay = ImagePointOverlay(
    revision=0,
    coordinates=np.array(
        ((-2.0e-3, 1.0e-3), (0.0, -1.0e-3), (1.5e-3, 1.2e-3))
    ),
    point_ids=("point-01", "point-02", "point-03"),
    labels=("A", "B", "C"),
    static_statuses=(
        PointStatus.EMPTY, PointStatus.OCCUPIED, PointStatus.INVALID
    ),
)
image_session.update_image_overlay(overlay)
image_session.set_parameter("show_point_labels", True)

# A causal data + point-status update is one immutable live payload.
next_frame = ImageFrame(next_snapshot, overlay)
image_session.update_data(next_frame)

# An overlay-only clear advances only the overlay clock.
image_session.update_image_overlay(ImagePointOverlay.empty(revision=1))
```

`coordinates` always contain canonical x then y. IDs and labels are optional
parallel metadata. `static_statuses` is the immutable marker form used by
hand-authored or calibration annotations.

A signal-driven layer instead stores one canonical bool/0-or-1 status
`OwnedSnapshot` in `ImagePointOverlay.status`. The producer declares
`IMAGE_POINT_OVERLAY_CONTRACT` and writes the strict image-axis, point-identity,
coordinate, and status-axis document returned by
`image_point_overlay_geometry(...)` into its run record. The application
validates and constructs the layer with
`image_point_overlay_from_signal(geometry, status, image, revision=...)`.
Status values express EMPTY/OCCUPIED and Dataset validity expresses INVALID.
The status Dataset must share the image's repeat/point topology and have one
complete trailing point axis. The renderer applies the Image/FacetGrid
PlotSpec scope and facet to those leading axes; it draws a judgement only for
one exact repeat/point cell, while a pooled surface remains UNKNOWN rather
than inventing a consensus.

An overlay-only update must have a strictly newer revision; it updates the
point artists without changing or reprojecting the image snapshot. Point ring
size is derived from canonical coordinate spacing and the immutable package
style. Occupied rings are the primary annotation; empty rings deliberately use
much lower opacity, and each optional ordinal is drawn at the ring's upper-left
with exactly the same colour and opacity as its ring. An `ImageFrame` always contains a
revisioned overlay, including an explicit empty layer. Reusing the identical
overlay revision in later frames is allowed; the same revision cannot identify
different content. A prepared frame also compares the point-layer authority at
commit, so it cannot overwrite an independent overlay update accepted while
the frame was being prepared.

PulseTimeline uses `PulseTimelineData` with public digital, analog, scan, DAC,
and repeat records:

```python
from zlc_plot import (
    PlotSession, PulseAnalogTrace, PulseBlock, PulseChannel,
    PulseDacScanSegment, PulseRepeatMarker, PulseScanRegion,
    PulseTimelineData, PulseTimelinePlot,
)

pulse_data = PulseTimelineData(
    channels=(
        PulseChannel("laser", "Laser"),
        PulseChannel("detect", "Detect"),
    ),
    blocks=(
        PulseBlock("laser", 0.0, 1.2, label="Init"),
        PulseBlock("detect", 7.0, 8.5, label="Count"),
    ),
    analog_traces=(
        PulseAnalogTrace(
            "bias", "Bias", -1.0, 1.0,
            starts=(0.0, 3.0, 6.0, 9.0),
            values=(0.0, -0.4, 0.5),
        ),
    ),
    scan_regions=(PulseScanRegion(3.0, 6.0, 1),),
    scan_dac_segments=(
        PulseDacScanSegment("bias", 3.0, 6.0, value=0.3, number=2),
    ),
    repeat_markers=(PulseRepeatMarker(1.5, 8.8, "×4"),),
    time_unit="us",
    total_duration=9.0,
)
pulse_session = PlotSession(pulse_data, PulseTimelinePlot(), size="4x2")
```

Pulse time coordinates are non-negative. `PulseScanRegion.number` is a positive
integer; `PulseDacScanSegment.number` is either a positive integer or `None` to
omit its badge. Every provided scan number must be unique across the timeline.
`pulse_session.set_time_unit("ns")` selects an explicit display unit; `None`
restores automatic ns/us/ms/s selection.

## PlotSession

```python
from zlc_plot import PlotSession

session = PlotSession(snapshot, curve, size="2x2")
session.set_parameters({"title": "Signal", "show_grid": True})
session.set_size("2x4")
session.configure(
    semantic=session.describe_semantics().values,
    parameters={"title": "Signal", "show_grid": True},
    size="2x4",
)
session.save("curve.png")
rgba = session.rgba()
```

`HistogramPlot` and `FacetGridPlot(cell=HistogramPlot())` expose an independent
`threshold_classifier` display parameter. Submit the complete target in one
call; an authored threshold sequence is optional and follows facet order:

```python
session.configure(
    parameters={"threshold_classifier": True},
    classifier_thresholds=(0.15, 0.21, 0.18),
)
```

The classifier owns its bimodal Gaussian classification fit, initial
fitted-population-weighted threshold, component/sum curves, movable threshold,
fitted population L/R percentages that sum to 100%, and population-weighted
fidelity readout. A coordinate-addressed threshold target may additionally
carry `gaussian_components` (left/right mean, sigma, and weight) so a saved
scientific Figure presents its authoritative model instead of refitting the
displayed bins; an explicit null keeps the final threshold line without
inventing a Gaussian fit. It is independent of the ordinary `fit()` state.

Use the context-manager form when practical:

```python
with PlotSession(snapshot, curve) as export_session:
    export_session.save("curve.png")
```

Only the nine named presets are accepted. Ordinary host-window resize does not call `set_size`.

### Display description and parameter editors

Each plot kind owns one canonical, toolkit-independent parameter schema. It
contains parameter names, Python types, defaults, ranges, finite choices and
the smallest render surface affected by an edit. Environment- or
data-dependent choices are delivered beside it in one immutable description:

```python
from zlc_plot import parameter_controls

description = session.describe_display()
controls = parameter_controls(
    description.parameter_schema,
    description.display_state.values,
    choice_overrides=description.parameter_choices,
)
print(description.kind, description.size, description.size_choices)
print(description.limits, description.viewport)
```

`parameter_choices` contains compatible units and the package's closed Image
colormap catalogue (`inferno`, `viridis`, `magma`, `plasma`, `gray`) when those
parameters apply. A frontend maps the returned
toolkit-neutral controls to widgets; it does not duplicate validation rules.
`zlc_plot.qt_controls.Qt5ParameterPanel` provides the PyQt5 mapping.

### Semantic edit description

Semantic roles are described by one registry-derived, frontend-neutral API:

```python
from zlc_plot import describe_semantics

semantics = describe_semantics(
    snapshot.schema, session.spec, layout=DEFAULTS.layout,
)
same_description = session.describe_semantics()
print(semantics.values)
```

`describe_semantics(schema, spec, *, layout=DEFAULTS.layout)` returns a
`zlc_plot.semantics.SemanticDescription`. Its `kind_choices` are the kinds whose registry
handler admits the schema; `axis_choices` is the stable ordered set of
`AxisRef` values declared by that schema; and `fields` contains the current
`kind`, `x`, `y`, `group`, `reduction`, `samples`, `facet_rows` and
`facet_cols` values that apply to the current kind. Every `zlc_plot.semantics.SemanticField` is
marked `rebuild=True`. A frontend that owns a complete form submits its whole
semantic/display/size/overlay/fit target once through `configure()`; `zlc_plot`
composes the typed spec and chooses the minimum render path. Code that already
owns a complete typed spec may call `replace_spec()` directly. `facet_cols` is optional, must differ
from `facet_rows`, and `facet_max_cells` is the layout-declared capacity for a
grid. Histogram `samples` is a multi-choice field. `zlc_plot.ui.semantic_controls()`
projects this exact description into the same toolkit-neutral
`zlc_plot.ui.ParameterControl` pipeline used by display controls; semantic controls carry
`semantic=True` and `rebuild=True`.

Frequently changed presentation state is edited in place:

```python
curve_session.set_labels(title="Signal", x="Delay", y="Response")
histogram_session.set_parameter("bin_count", 80)
image_session.set_parameter("show_colorbar", False)
curve_session.set_relim_mode("fixed")
curve_session.set_y_limits(0.0, 2.5)
image_session.set_color_limits(0.2, 1.8)
curve_session.set_x_limits(-1.0, 1.0)
curve_session.set_view_limits(x=(-1.0, 1.0), y=(0.0, 2.5))
```

Entering `fixed` through either `set_parameter()` or `set_relim_mode()`
atomically seeds missing y/color bounds from the currently painted limits.
Returning to `tight` or `normal` clears omitted authored bounds. Label and grid
edits retain their existing artists; they do not reproject data, reconstruct
the Figure or replace data artists. Title/x/y/value labels update those artists
in place and finish in the same complete-frame transaction. Grid and colorbar
visibility use the separate chrome lane. The parameter schema associates every
field with composable `RenderEffect` flags; one transaction resolves those
effects and performs one final draw (or the image-only axis path). For Histogram,
`normal` and `fixed` reuse the canonical bin domain and expand only a side breached by new live data;
`tight` intentionally recomputes the domain every revision. Changing plot
kind, semantic axes, reduction, grouping or facet structure is a rebuild on
the same session surface: `replace_spec()` retains only independently
revalidated display values, units and semantically compatible viewport state,
then clears selector and fit state atomically. The Figure and host identity
remain unchanged.

## Selectors and selected data

```python
from zlc_plot import NumericRange, SelectorKind

session.set_x_selector(-1.0, 1.0, display=True)
session.set_area_selector(
    NumericRange(-1.0, 1.0),
    NumericRange(0.0, 2.0),
    display=True,
)
session.set_threshold_selector(0.7, display=True)
session.set_crosshair_selector(0.0, 1.0, display=True)

canonical_state = session.selector_state(SelectorKind.X_RANGE)
display_state = session.selector_state(SelectorKind.X_RANGE, display=True)
selection = session.selector_data(SelectorKind.X_RANGE)
print(selection.flat_indices, selection.canonical_values, selection.display_values)

unsubscribe = session.subscribe_selection(
    lambda event: print(
        event.change,
        event.display_selector.value,
        event.data_revision,
    ),
    selector_kind=SelectorKind.X_RANGE,
)
session.set_selector_value(
    SelectorKind.X_RANGE,
    NumericRange(-0.8, 0.8),
    display=True,
)
unsubscribe()
for kind in (SelectorKind.AREA, SelectorKind.THRESHOLD, SelectorKind.CROSSHAIR):
    session.remove_selector(kind)
```

Selection event 只携带 change、canonical selector geometry、display selector
geometry 和 data revision；它不携带、缓存或预先计算选区数据。只有外部显式调用
`selector_data(kind)` 时，session 才会对调用当时最新的 immutable snapshot
执行切片。返回值包含 mask、flat indices、canonical/display values、相关坐标和
data revision。收到 `SelectionChange.REMOVED` 后该 kind 已不存在，不能再用该 kind
调用 `selector_data()`。

生命周期事件为 `ADDED`、`UPDATED`、`COMMITTED`、`REMOVED`。同一 selector kind 的
revision 严格递增，包括连续拖动、提交、删除后重新创建；event 仍只传 geometry，
不会因为订阅而提前计算 selected data。

每种 selector 在一个 session 中最多存在一个。任意主图左拖都会创建完整矩形 area selector；中心区域用于整体移动，四边和四角的八个 handle 用于改变范围。Curve、Rolling、Histogram、PulseTimeline 与 1D facet 以矩形的 x 范围选数据，Image 与 image facet 使用 x/y 范围。新 Area 在拖动期间只是 candidate；旧 committed Area 保留到非退化 geometry 校验并成功绘制后 controller 才原子 swap，失败或 cancel 直接恢复旧 state。同 kind 替换是 update/commit，不会制造一次假的 remove/add。指针拖动和 API 修改共享同一个 session state；`session.selectors` 返回 immutable state tuple，所有增删改仍经过 `PlotSession` API。

PulseTimeline returns an immutable typed result:

```python
pulse_session.set_x_selector(2.0, 5.0)
pulse_selection = pulse_session.selector_data(SelectorKind.X_RANGE)
print(
    pulse_selection.blocks,
    pulse_selection.analog_traces,
    pulse_selection.scan_regions,
    pulse_selection.scan_dac_segments,
    pulse_selection.repeat_markers,
)
```

`PulseTimelineSelectionData` returns every timeline record whose interval
intersects the selected source-time span. It carries canonical and display
selector values plus the source data revision. This result object is created
only by an explicit `selector_data(kind)` call.

## Units

```python
session.set_axis_unit(scan_x, "V")
session.set_value_unit("V")
```

Canonical arrays are unchanged. Ticks, labels, selector display values and fit presentation are converted together. Selector state and fit calculations remain canonical.
The aliases `"arb"` and `"1"` both resolve to the dimensionless unit; neither
adds a suffix to an axis or colorbar label. Passing `None` restores the
data-declared display unit.

Curve, Rolling and Histogram y limits use the current display-value/count unit.
Image color limits use the current displayed value unit. Normal mode keeps a
zero baseline for non-negative data; tight mode follows both bounds; fixed mode
is controlled through the public helpers:

```python
session.set_y_limits(0.0, 0.003, fixed=True)
session.reset_y_limits(mode="normal")
image_session.set_color_limits(0.2, 1.7, fixed=True)
print(image_session.resolved_color_limits(display=True))
image_session.reset_color_limits(mode="tight")
```

Image side-distribution 中始终只有一组 color-limit handles。它们是显示色阶的控制，
不属于 `SelectorKind`，不会出现在 `selectors` / `selector_state()` / fit scope 中；
空白处点击不会创建另一组。`resolved_color_limits(display=False)` 返回 canonical value
unit 下实际画入当前 frame 的范围，`display=True` 返回当前显示单位；
`RasterPlotHost.resolved_color_limits()` 提供相同的 worker-safe 读取。

## Fit scope and results

```python
from zlc_plot import FitTarget

available = curve_session.fit_models
assert all(FitTarget.SERIES in model.targets for model in available)
selection = curve_session.fit_selection("gaussian_offset")
result = curve_session.fit("gaussian_offset")
future = curve_session.fit_async("gaussian_offset")
```

`fit(..., fixed={"offset": 0.0}, initial={"sigma": 1.0})` uses two distinct
semantics: `fixed` parameters are removed from the optimizer exactly, while
`initial` only replaces the corresponding starting guess. Those mappings are
keyed by parameter *name*, which is also the published signal id and what a
saved layout stores.

The Panel editor projects the same contract as one display-unit expression,
written in the *symbols the model's formula prints* rather than those names --
`gaussian_offset` draws $f(x)=A e^{-(x-x_0)^2/2\sigma^2}+B$, so its box takes
`B=0, sigma=guess(1)`. `FitModelSpec.symbols` lists them and the editor shows
them in the field's own hint; `parameter_for_symbol` maps one back to the
parameter it names. Invalid optional expression text falls back to the selected
model's automatic fit and reports a warning; it is never persisted into the
canonical Panel/Figure fit target.

`FitModelSpec.targets` declares whether a model fits a `SERIES`, `HISTOGRAM` or
`IMAGE` projection, and `default_for` marks a target default. A registry permits
at most one default per target; the built-in registry supplies one for all three.
Every `FitModelSpec` also declares one required `headline` parameter name. The
name must be present in `parameter_names`; the renderer uses that declaration
for one compact parameter annotation in the upper-left of each FacetGrid
overview cell. The annotation uses the same value/uncertainty formatter as the
focused full annotation, including the parameter symbol, standard error and
display unit. Cell titles remain the facet identity only.
`FitModelSpec.capabilities` declares optional solver paths. The specialized
regular-image path is selected by capability bit — `regular_image_radial` for
the radial model, `regular_image_separable` for separable per-axis models such
as the built-in anisotropic Gaussian — never by model id or evaluator
identity, so a compatible registered model can provide the same specialized
contract explicitly. A model with neither capability takes the general
coordinate-expansion solver.
`session.fit_models` returns only models compatible with the current semantic
plot and coordinate units, with the default first. The same filtered tuple is
available asynchronously through `plot_host.fit_models()`; Notebook and GUI
must consume it instead of inferring compatibility from independent arity.

The built-in catalogue is Series: Lorentzian (default), Gaussian with offset,
symmetric Lorentzian doublet, damped sine and exponential decay; Histogram:
Bimodal Gaussian (default) and Single Gaussian; Image: Radial Gaussian center
(default, only for compatible x/y coordinate dimensions) plus Anisotropic
Gaussian center for independent x/y dimensions. PulseTimeline has no numeric
fit target.

For coordinate fits the scope precedence is:

```text
AREA > X_RANGE > viewport > all valid finite data
```

An explicitly requested `selector_kind=THRESHOLD` uses the Threshold instead
of this automatic precedence.

Fit does not call `selector_data()`. It resolves the current selector/viewport
range and reads the current immutable snapshot directly when the fit request is
evaluated. `zlc_plot._fit_projection.FitSelection` records the source `data_revision`, selected sample
count and canonical selector/viewport authority. The built-in radial Gaussian
path for a regular dense Image keeps its native 2-D observation view and may
leave `selected_indices` as `None`; selecting all 2048² pixels therefore does
not first allocate a 4,194,304-element index array. Material result arrays are
created only when the solver returns. Custom Image models use the general
coordinate expansion path unless they provide their own specialized solver.

每种 selector 最多存在一个。默认 fit 严格按 `AREA > X_RANGE > viewport > all` 解析；需要固定 live fit
authority 时可传入 `selector_kind=SelectorKind.AREA`、
`selector_kind=SelectorKind.X_RANGE` 或 `selector_kind=SelectorKind.THRESHOLD`。
Histogram 的 `threshold_classifier` 独立拥有 bimodal classification fit、按拟合population加权的初始 threshold、三条分类曲线，以及严格合计 100% 的 fitted-population L/R 百分比和加权 Fidelity 显示；coordinate-addressed threshold target可携带权威`gaussian_components`，此时直接显示该模型而不重拟合bins。普通 `bimodal_gaussian` fit 不创建、移动或清除 classifier，classifier 也不写 `fit_status`。只有调用方明确以 `selector_kind=THRESHOLD` 启动普通 fit 时，当前 threshold 才作为该普通 fit 的 scope。crosshair 不作为 fit scope；color limits 根本不属于 data selector。`session.fit_status` 在没有 overlay
时为 `None`；只有已接受结果同时匹配当前 data revision 和 fit-context generation
（canonical selector/viewport authority、facet、projection 与 fit request）时才为 `"current"`；
selector/viewport 改变后旧结果只标为 `"lagging"`，不会在 pointer motion 中启动新的 solver。
图内保留上一份稳定 overlay；下一条 data revision 才一次性替换 curve 与参数文字。

`zlc_plot.fit.FitResult` 包含 success/message、model id、参数值、covariance/error、selected indices 和 `source_revision`；facet batch 使用同名的 `source_revision`，而 `batch_revision` 只表示发布顺序。只有 source revision 仍然有效的结果才会显示到当前图。Curve、Rolling、Histogram、Image 与 FacetGrid 的三种 cell 语义共用同一 `zlc_plot._fit_projection.FitSelection`、solver、结果接受和 overlay presentation 生命周期；差异只在于从当前 painted payload 生成 series、bin counts 或 scalar-field solver input。FacetGrid 的 `fit(..., live=True)` 每个 data revision 对所有 cell 生成一个 `zlc_plot.fit.FacetFitBatchResult`，其 `overlays` 与 `results` 按 cell 同序；overview 画全部 cell 的 fit 曲线和每 cell 一个 headline 参数注释，focus 后显示所选 cell 的完整参数框。group、reduction 与 valid mask 只在 DataView 中评估一次，selector/viewport 随后筛选实际显示的 projection；fit 不会另建 raw tensor mask/reduction 路径。Rolling 还会把候选数据限制在当前可见 window。selector/viewport 决定参数估计样本，`zlc_plot.fit.FitResult.fitted` 与 residuals 也对应这些样本；图内 overlay 则用已接受参数覆盖当前完整显示域。`FitResult.selected_indices` 索引当前 fit projection（series、histogram bins 或扁平 image projection），不是原始 snapshot 的 flat indices；原始数据索引只由显式 `selector_data(kind)` 返回。成功结果同时显示使用当前显示单位的公式、参数值和 `±` 不确定度。

Facet批量结果直接公开与cell同序的数值列：

```python
batch.parameter_names
batch.parameter_units
batch.parameter_values
batch.parameter_errors
batch.parameter_error_validity
batch.success
batch.sample_coordinates
batch.sample_labels
batch.source_revision
batch.batch_revision
```

Scalar`FitResult`公开自己的参数向量、standard errors和source/batch revision；
不再构造一份重复table。Facet failure messages由`batch.failure_messages`提供。

`fit()` / `fit_async()`默认启用live fit。Runtime提交的每个data revision通过同一
prepare/solve/commit入口与匹配的fit原子呈现。selector、viewport、unit和resize
不会因为没有新data而自动求解；需要立刻按新选择重算时显式调用`fit()`。

`zlc_plot.fit.FitResult.parameters` 始终使用 canonical units。外部数据消费者若要取得exact solve的公式、参数、单位和uncertainty，应订阅`FitEvent`：

```python
def show_fit(event):
    print(event.formula)
    # For a FacetGrid batch, event.result is zlc_plot.fit.FacetFitBatchResult and
    # event.overlays contains one display overlay per facet cell.
    for parameter in event.display_parameters:
        print(parameter.name, parameter.value, parameter.standard_error, parameter.unit)

unsubscribe = session.subscribe_fit(show_fit)
```

`subscribe_fit(..., replay_current=True)` additionally delivers the currently
accepted immutable `FitEvent` once at subscription time. This is for a late
consumer such as FigureViewer's Runtime bridge; it does not run the solver or
renderer again. The default remains future events only.

Live data pair的event在solve结果通过request/projection identity检查后发布，因此Rolling可以与main raster并行准备；此时主图仍保持旧front，直到owner把`data@N + fit@N`一起commit/present。显式manual `fit()`则仍在accepted overlay transaction之后通知。两种路径都只发布一次同一source revision；event callback不得被当成“像素已呈现”的信号。

Histogram fit 使用 count projection；`density=True` 或 `cumulative=True` 时会明确拒绝 fit，切回两者均为 `False` 后再调用。Threshold classifier 同样要求 `cumulative=False`，但不依赖普通 fit 的启停或 model choice。

Fit annotation 使用固定 axes-fraction anchor，单图/focus 的 full annotation 为
3.25 pt，FacetGrid overview 的单行 headline annotation 为固定 3.5 pt；两者
均使用 package palette 的 fit text，z-order 高于 data/fit curve 且低于 selector。Selector geometry 变化只更新
内容或 fit result，不重新寻找 annotation anchor，因此框选、pan 或 live revision
不会让参数文字跳位。

Built-in models provide analytic residual Jacobians to SciPy's common
`least_squares` path; custom models may omit the declaration and retain
two-point numerical differentiation. A live fit request uses the last accepted
parameters for each cell as the next revision's initial seed, falling back to
the model initializer on the first frame or after a failed warm solve.

自定义模型可作为 `FitModelSpec` 直接传给 `fit()`，或在
`zlc_plot.fit.FitModelRegistry` 中注册后由 `zlc_plot.fit.FitEngine` 注入 session。参数的 canonical/display
单位关系由 `FitParameterSpec` 声明；可选的分量曲线、交点统计或 radial
center/radius glyph 由 `FitPresentationSpec` 声明。renderer 只消费已经生成的
通用 polyline、guide、marker、ellipse 和 annotation，不按 model id 复制公式或
单位转换逻辑。

## Live revisions

Runtime owns run history、publication cadence和Stop/Final。它通过
`RasterPlotHost.update_data()`提交immutable revision；host在自己的worker上执行
`prepare_live_frame → solve_live_frame → commit_live_frame`，并只发布匹配的
data/fit pair。Plot层不再提供第二个live controller或channel。

## Notebook

Install the optional stack:

```bash
python -m pip install -c constraints.txt -e ".[notebook]"
```

唯一产品教程是`packages/zlc_workbench/notebooks/usage.ipynb`；release gate从
checkout外的fresh kernel执行它，不修改`sys.path`。

```python
from zlc_plot import show

view = show(session)
```

`display()` is the only Notebook output path and is idempotent for one view.
The view retains one Figure, canvas and widget model for its complete lifetime;
the display handle is not exposed as a second rich-display path.
The display bundle carries the current front as a static `image/png` fallback
next to the widget-view mime, and `close()` replaces the live widget output in
place with the final front's PNG: closing a view (directly, through
`close_session_on_close`, or from a later cell) leaves the last frame visible
in the original cell instead of a blank output, and the frame survives
notebook save/reopen because a closed widget model cannot be resolved again.
`zlc_plot.notebook.NotebookView` 是 `RasterPlotHost` 的薄 `anywidget` adapter（底层仍兼容
ipywidgets 的 DOMWidget 协议）。它把一个完整的 `RasterFront` 作为单个
`frame_packet` buffer（包含 RGBA、尺寸、DPR 与 interaction map）原子交给浏览器
的单一 canvas；`SelectorScene` 的提交态与拖拽候选都由 kernel 的同一套 Matplotlib
renderer 烘焙，浏览器只负责 blit 与输入归一化。不加载 ipympl、不创建 Matplotlib
widget，也不要求任何 `%matplotlib` magic。固定 size 与 DPR 变化只发布新的完整
front，Notebook 与 Qt 使用同一套 host/session 协议。

调用 `NotebookView.display()` 时会自动创建唯一 widget 输出。View 不创建按钮、
下拉框、参数编辑器或状态面板；selector 与包提供的 zoom/pan 手势和 GUI 使用
同一套 raster pointer events 与 session state：

- 任意主图左拖直接创建一个完整矩形 area selector，无需 enable call；每种 selector
  在一个 session 中最多存在一个。中心区域移动矩形，四边和四角的八个 handle 改变范围。
- 1D、Rolling、Histogram、PulseTimeline 与 1D facet 使用矩形 x 范围；
  Image 与 image facet 使用 x/y 范围。
- Right-click places a crosshair and double-right-click clears it.
- The wheel zooms; middle-drag pans.  Double-middle-click zooms to the
  committed area/x-range selection (area wins; the selector is kept), or
  restores the complete home view when no range selection exists.
- Double-clicking a FacetGrid overview cell opens its focus view; double-click
  again or press Escape to return, with `session.show_facet_overview()` as the API equivalent.
- Escape or `session.cancel_interaction()` restores and releases an unfinished gesture.
- Image side distributions expose draggable low/high color limits.

```python
session.set_size("2x4")
session.set_parameter("show_grid", True)
session.set_x_selector(-1.0, 1.0, display=True)
result = session.fit("gaussian_offset")
```

For a new immutable revision, submit it through the existing session/host:

```python
session.update_data(next_snapshot)
# Hosted product code uses host.update_data(next_snapshot).
```

Run history、cadence和Stop由Runtime拥有；Notebook view只消费RasterHost front。
Pointer cadence、selector hit radius和wheel zoom factor位于`DEFAULTS.interaction`。

Missing dependencies raise `BackendUnavailableError` without making ordinary
package import depend on Jupyter.

## PyQt5

PyQt5是根产品的固定runtime依赖，不存在单独的Plot Qt extra。

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
    CurvePlot(AxisRef.point_dimension("x")),
    size="2x2",
)
widget = Qt5PlotWidget(plot_host)
description = plot_host.describe_display().result().value
parameters = Qt5ParameterPanel(description)
parameters.parameterEdited.connect(plot_host.set_parameter)
widget.show()
try:
    app.exec_()
finally:
    widget.close_adapter()
    plot_host.close()
```

Adapter 只支持 PyQt5 并惰性导入。必须由 `ensure_qt5_application()` 创建或取得首个
`QApplication`，以便在创建 application 前统一设置 High-DPI 属性。`Qt5PlotWidget`
只显示居中的固定尺寸 QImage front，不包含 application controls；外部 Qt 控件连接公开的
异步 host methods，不能取得 worker 内的 Matplotlib session：

```python
bins.valueChanged.connect(
    lambda value: widget.host.set_parameter("bin_count", int(value))
)
fit_button.clicked.connect(
    lambda _checked=False: widget.host.fit("gaussian_offset", live=True)
)
size.currentTextChanged.connect(
    lambda preset: widget.host.set_size(str(preset))
)
```

Runtime通过`plot_host.update_data(snapshot)`提交revision。事件订阅使用
`plot_host.subscribe_display/subscribe_selection/subscribe_fit`；fit callback表示exact analysis completion，display/front callback才表示raster presentation。回调若要修改 Qt 控件，
再经 `widget.dispatch` 回到 UI thread。Notebook 与 Qt 同时打开时，应基于同一 immutable
snapshot 分别创建 session。

`plot_host.fit_models()` 返回 worker session 的 fit catalogue；GUI 可用每个
`FitModelSpec.display_name` 显示选项，并把对应 `model_id` 传给 `plot_host.fit()`；
该 catalogue 已按当前 plot 的 `FitTarget` 与单位兼容性过滤，第一项是默认模型。

Selector geometry 与数据切片在 raster API 中也是两条明确的读取路径：

```python
states = plot_host.selectors().result().value
display_state = plot_host.selector_state(
    states[0].kind,
    display=True,
).result().value
selected = plot_host.selector_data(states[0].kind).result().value
```

`selectors()` 和 `selector_state()` 只读取 immutable geometry，不计算 mask 或复制
scientific data；只有显式 `selector_data()` 才按调用时的最新 snapshot 物化选区。
同一份已经画出的 geometry 也位于 `RasterFront.interaction.selectors`，供自定义
frontend 与它正在显示的像素 front 保持严格一致。

Qt 与 Notebook 使用同一个固定 `SurfacePlan`：DPR 增大时逻辑 preset、axes geometry
和 font tier 不变，但 backing store 使用更多物理像素。首帧、live revision、fit overlay
和 preset 切换都通过 session 的同一 surface redraw 路径更新。

`widget.set_interaction_enabled(False)` suspends only plot input, allowing an
outer preview/editor scroll area to receive wheel events while render/live/DPR
updates continue. For application-coordinated group presentation, construct
widgets with `auto_present=False`, join returned `RasterOperation.front` values
using the application's causal shot identity, then call `present_front(front)`
on the Qt owner thread. `RasterIdentity.data_generation/data_revision` and
`image_overlay_revision` are verification fields, not a replacement for the
application's causal join.

The exact already-visible physical pixels are available through
`widget.presented_front.buffer.save(path)`. `plot_host.save(path, dpi=...)`
remains the separate high-resolution rerendering route.

## Persistence boundary

`zlc_data` owns Dataset NPZ encoding and decoding. Atomic path publication
belongs to `zlc_durable`:

```python
from zlc_data import load_npz, save_npz
from zlc_durable import atomic_write_file

atomic_write_file("run.npz", lambda stream: save_npz(stream, snapshot))
restored_snapshot = load_npz("run.npz")
```

The already-presented Edit-tab snapshot is the immutable raster front:

```python
front = widget.presented_front
if front is not None:
    frozen_pixels = front.buffer
    source_identity = (
        front.identity.data_generation,
        front.identity.data_revision,
        front.identity.image_overlay_revision,
    )
```

This is a zero-render copy of the exact accepted physical pixels and their
interaction transform. An interactive Edit surface or local Fit uses another
`RasterPlotHost` created from the application's frozen `zlc_data.OwnedSnapshot`,
`PlotSpec`, and authored parameter state. Runtime sessions and asynchronous Fit
handles are deliberately not serialized or cloned. Archive manifests/paths,
causal EventRef, authored panel configuration, devices, Logic routes and
workflow persistence remain application responsibilities.

### Application integration ownership

Application surfaces map to the public boundaries below. The adapter translates
domain values once; it does not reproduce projection, selector, fit, raster, or
backend code.

| Application path | `zlc_plot` public boundary | Remains application-owned |
| --- | --- | --- |
| Plot panel / plot-kind switch | `PlotSpec`, `PlotSession`, `RasterPlotHost`, `describe_semantics()`, `replace_spec()` | Choosing a signal, kind and axes; authored panel state and placement; atomic semantic rebuild on the existing host |
| Same-shot group display | `RasterOperation.front`, `RasterIdentity`, `Qt5PlotWidget(auto_present=False)` and `present_front()` | The causal shot/event join and the decision that a cohort is complete |
| Pulse preview | `PulseTimelineData`, `PulseTimelinePlot`, interaction gate, fixed size and save APIs | Pulse document/compiler semantics, visible-row policy and conversion into immutable timeline records |
| Edit-tab snapshot | `widget.presented_front` for exact pixels, or a new host over a frozen snapshot/spec for independent interaction and fit | Which source revision and authored settings the Edit tab freezes; archive identity |
| Selector and Fit outputs | selector geometry/data methods, `SelectionEvent`, compatible `fit_models()`, `FitResult` and `FitEvent` | Wiring a result into another logic node, naming routes and deciding what is persisted |
| Data/project/device persistence | no plot API; `zlc_data.save_npz/load_npz` encodes and decodes scientific snapshots | Project files, device calls, Logic routes, causal IDs, paths and atomic publication through `zlc_durable` |

Coordinate/status maps use the ordinary Image boundary: the application performs
its exact same-shot image/occupancy join, then supplies one
`ImageFrame(snapshot, ImagePointOverlay(...))`. Point coordinates and statuses
may advance live without creating a second plot kind.
