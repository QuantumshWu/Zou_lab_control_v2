# Architecture

`zlc-plot` 是一个安装单元，依赖独立的角色轴 `zlc-data`。本仓只发行
`zlc_plot`；科学数据对象始终来自 `zlc_data`，绘图仓不再捆绑第二份数据模型。

`zlc_plot`：view specification、Matplotlib renderer、selector、fit、固定 surface
和 Notebook/PyQt5 adapter。

上层实验控制应用仍是第三个边界。它拥有设备配置、设备调用关系、Logic route、实验 workflow、项目文件和数据如何产生；这些状态不能进入 `zlc_plot`。

## Ownership

| Concern | Owner |
| --- | --- |
| `(R, P, *data_dim)` geometry, dtype, validity | `zlc-data` role-axis schema and `OwnedSnapshot` |
| PointTable rows and optional GridTopology | data producer + `zlc-data` validation |
| Canonical unit annotations | `zlc-data`; display conversion is `zlc_plot.units` |
| NPZ save/load | `zlc_data.save_npz` / `zlc_data.load_npz` |
| Plot kind, axes roles, reduction, facet choice | `zlc_plot` specifications |
| Runtime viewport, display parameters, selectors, fixed size | `PlotSession` |
| Parameter names/types/defaults/ranges/choices/render impact | core `ParameterSchema` + `RenderEffect` |
| Schema-to-control projection and PyQt5 widgets | `zlc_plot.ui` / `zlc_plot.qt_controls` |
| Concrete control layout and application workflow | application UI |
| Pulse payload revision envelope and latest-only handoff | `LiveDataRevision` / `LivePlotController` |
| Fit models and immutable fit results | `zlc_plot.fit` |
| Immutable rendered front and exact physical-pixel export | `RasterFront` / `RasterBuffer` |
| Same-shot causal join, panel document and PulseDocument projection | application layer |
| Device/Logic routes and acquisition orchestration | application layer |

The dependency direction is one way:

```text
application / acquisition
        │
        ▼
 zlc_data.OwnedSnapshot / ImageFrame ─┐
 PulseTimelineData ──────────────► zlc_plot.PlotSession
        │                     │              │
        │             LiveDataRevision       ├── headless Agg/export
        │                     │              ├── NotebookView (RasterFront + anywidget)
        │                     └─ latest-only └── RasterPlotHost worker
        │                                           │
        │                                           └── Qt5PlotWidget (QImage)
        │
        └── NPZ I/O (zlc_data only)
```

The data contract never imports Matplotlib or Qt. `zlc_plot` never imports device or Logic modules.

## Data geometry and point topology

Every scientific snapshot has fixed shape:

```text
(R, P, *data_dim)
```

- `R` is the repeat axis.
- `P` is the ordered PointTable row axis.
- `data_dim` contains dense per-point axes such as camera coordinates or site.

PointTable columns are row attributes; repeated/unique values do not prove a Cartesian scan. When a producer knows that rows represent, for example, `b_x × b_y × b_z`, it attaches a `GridTopology` containing dimension domains and `row_to_cell`. `zlc_plot` consumes that declaration but never reconstructs it heuristically.

FacetGrid may facet along repeat, point rows, a PointTable coordinate, a declared GridTopology dimension, or a data axis. `facet_rows` and optional
`facet_cols` form an explicit row-major grid; every cell in one FacetGrid uses
the same cell kind. A point-coordinate grouping without GridTopology remains
a group-by operation, not a claim of Cartesian geometry.

## Six formal plot kinds

The public `PlotKind` vocabulary is complete:

1. `Curve`
2. `Image`
3. `Histogram`
4. `Rolling`
5. `FacetGrid`
6. `PulseTimeline`

Rolling is one plot kind with `side_distribution=True/False`. FacetGrid cells are homogeneous Curve, Image, or Histogram views. Canonical point coordinates are an independently revisioned `ImagePointOverlay` on an ordinary Image, not another plot kind or data geometry. Pulse program conversion remains an application/domain responsibility; the plot package consumes `PulseTimelineData` with `PulseChannel`, `PulseBlock`, `PulseAnalogTrace`, `PulseScanRegion`, `PulseDacScanSegment`, and `PulseRepeatMarker`.

## Session state

`PlotSession` owns runtime view state around one immutable input payload:

- independent data, display and layout revisions;
- fixed named size preset and resolved `SurfacePlan`;
- display-unit overrides and viewport;
- immutable selector state and selection data API;
- fit request/result and overlay;
- private renderer surface; external code receives only session APIs or immutable raster fronts.

`FitProjection` is the sole owner of the current payload and data revision;
`PlotSession` delegates those reads and never keeps mirror fields. Display,
selector, viewport and fit generations cross the projection boundary as one
immutable `ProjectionContext`. A data update is fully projected first and then
installed through one presentation transaction; projection, Image overlay, fit
scene, facet state and layout are restored together if painting fails. Fit
polylines, markers, ellipses and annotations live in a backend-neutral scene
value. Only the session/renderer boundary resolves Matplotlib axes or pixel
transforms.

所有主图的左拖交互都写入同一个单例 area selector。renderer 始终显示完整矩形：中心区域移动整体，四边和四角的八个 handles 改变范围。进行中的 candidate 在显示层 shadow 旧 committed Area，但 authority 不会在 pointer-down 时删除；完整预校验和 presentation 成功后 controller 才原子 swap，失败或 cancel 直接恢复旧 state。selection data 层根据 plot geometry 解释范围：1D、Rolling、Histogram、PulseTimeline 与 1D facet 使用 x，Image 与 image facet 使用 x/y。API 创建的 x-range、threshold 和 crosshair 仍是独立的显式 selector 类型。

A complete experiment archive is composed by the application from its own schemas:

```text
data NPZ + application plot/control/workflow document
```

`zlc_plot` provides runtime plotting APIs, immutable raster fronts and figure
export, but deliberately does not become a panel document, project-file or
device-routing owner. An Edit tab can immediately hold the accepted
`RasterFront`; an independently interactive Edit surface is another host over
the application's frozen data and authored panel state, never a clone of
running worker/Future ownership.

## Display schema and UI boundary

Every semantic plot specification creates one immutable `ParameterSchema`.
This is the sole owner of parameter names, Python types, defaults, finite
ranges, enumerated choices, update impact, transition normalization and
complete-state invariants such as paired fixed limits. `PlotSession`
materializes context-dependent values, such as current visible limits when
switching to fixed mode, validates display units against current data and asks
the schema to normalize and validate the complete candidate. `DisplayStateStore`
only atomically commits that already-prepared immutable candidate.
`DisplayDescription` combines that schema with the accepted state,
fixed size choices, current axes limits, explicit viewport and dynamic
parameter choices such as compatible units or the closed package colormap set.

`zlc_plot.ui.parameter_controls()` converts the description into ordered,
toolkit-neutral editor records. The optional `qt_controls` module performs the
PyQt5 widget mapping and nothing in the core imports Qt. A concrete GUI decides
where controls appear and which plot-kind factory to select; it does not repeat
validation or renderer policy. Changing plot kind or any semantic role uses
the session's atomic `replace_spec` rebuild policy on the existing Figure and
host; independently valid display values and compatible viewport state are
retained, while selector and fit state are cleared. Ordinary parameter,
label, unit, viewport, selector and fit changes retain the existing session
and update persistent artists.

`RenderEffect` is a set of independent invalidation flags rather than a single
redraw severity. It distinguishes view/payload projection, base geometry/style,
axis transform, text/chrome, overlays, layout, fit selection/presentation and
interaction reprojection. `PlotSession` combines all accepted changes into one
immutable `RenderFrame`; the renderer updates only affected persistent artists
and presents the transaction once. This is why a label edit can repaint glyphs
without rebuilding data, while a unit edit atomically updates axes, selectors,
fit presentation and interaction transforms.

`RenderFrame.effects` is the complete internal invalidation contract. Text,
chrome, selector and fit changes mutate their persistent artists and end in one
complete Agg draw; there is no cached selector/text background that can be
restored over a newer fit. For an image payload-only update the renderer may
exclude the image artist from the axis draw and paint that artist once with the
axis blit API, but the resulting buffer is still one complete front. The
	NotebookView consumes that complete `RasterFront`; the browser only blits it
	and normalizes pointer input, so no second renderer or diff protocol can expose
	a partial frame. `CHROME` is separate because grid, colorbar visibility and axis
	presentation invalidate more than text glyphs.

## Static and live are the same model

Static 是只接收一个 immutable payload 的 session；live 使用同一个
session/render path 接收严格递增的 immutable revisions，因此 selector、viewport、
单位、显示参数和 fit 没有第二套实现。Dataset live 保持同一 schema/generation；
同一因果 image 与动态 point overlay 使用一个 `ImageFrame`，frame ordering 只由其
snapshot 决定；点层仍有单调 revision，空层也是有 revision 的值。`PulseTimelineData` 由 `LiveDataRevision` 在 transport boundary
携带 producer revision。Controller 展示 payload 时保留该 revision，使 session
state、selection events、selected data 和 fit inputs 指向同一 producer revision。

`LatestIngress` owns dataset replace/patch/rolling state. Dataset and specialised
plot transports share `LatestRevisionChannel` for the same cadence-free,
capacity-one semantics; `LivePlotController` is the public presentation facade.
Producers may publish faster than display, and a pending item is replaced by the
newest revision. GUI timers choose a configurable display cadence while ingress
never builds an unbounded queue.

`PlotSession.owner_dispatch` is the stable live-to-owner gateway. It resolves
the currently attached Notebook or raster marshal when invoked instead of
capturing whichever host existed when a controller was constructed. Headless
direct execution and host attach/release share one ownership gate, and every
concrete dispatcher returns one Future covering callback completion. This keeps
controller construction order independent from host attachment without allowing
a live worker to bypass an attached surface owner. Notebook and Qt owners retain
their queued completion Futures and cancel them during close; the controller's
active owner wait has a separate shutdown wake, so a stopped event loop cannot
strand the live worker.

Layout changes rebuild the complete named layout inside the accepted Figure and publish it only after successful drawing. Data/artist updates mutate the same persistent artists whenever layout topology is unchanged.

## Fixed surfaces and backend parity

Plots accept exactly nine presets:

```text
1x2  2x2  4x2  1x4  2x4  4x4  4x8  8x4  8x8
```

`resolve_surface` is the sole geometry resolver. It produces logical size, physical raster, DPI, normalized axes boxes and FacetGrid typography. Device-pixel ratio and export scale change physical pixels only.

Both Notebook and Qt consume `SurfacePlan.logical_size`. Resizing a browser cell or Qt host adds centered whitespace; it does not silently author another plot size. A user-visible size change must call `PlotSession.set_size(preset)`.

`SurfacePlan.raster_size` is the half-up physical backing size for the current
DPR. `resolve_surface()` derives the Figure inches from that raster size and the
physical DPI, so Matplotlib produces the exact backing dimensions even at a
fractional DPR. The named logical size, normalized axes and fixed data box remain
unchanged. Native Qt keeps the named logical QWidget size while its DPR-scaled
backing store matches the same physical plan.

Notebook adapter uses `anywidget` (not an output-cell AMD module injection), so JupyterLab registers its ESM view through the standard widget manager before creating the model. The complete `frame_packet` retains the atomic front protocol described below.

**One renderer draws every pixel on every frontend.** Transient gesture candidates — the area rectangle with its handles and live coordinate readout, threshold and color-limit drags — are baked into raster fronts by the same matplotlib artists that draw committed selectors (`preview_selector` composes a complete frame per pointer move, and `raster_generation` ties front publishing to actual pixel changes). Neither the Qt widget nor the browser view paints geometry of its own, so drag-time and committed appearance are pixel-identical by construction and cannot drift between frontends. Pointer moves coalesce in the host queue, so preview cadence degrades gracefully to render speed instead of lagging. Selector geometry is handed to the renderer in **painted space** — the axes' own data space: display units for DataView-backed kinds, source time units for pulse (whose axes convert only tick and label text, applying the display factor to readouts exactly once through `x_label_factor`). Surface observers are notified at the single present commit point inside `_update_renderer`, so no mutation path can forget to notify.

The browser view is a **frame blitter and input normalizer only**. It paints `frame_packet`, reports its environment over the widget comm (`{type: "environment", device_pixel_ratio}`, feeding `host.set_device_pixel_ratio` exactly as the Qt widget does on screen changes; the kernel remembers the last report so later views publish a crisp first front, and the view holds a mismatched-density first frame briefly), and forwards normalized pointer/wheel/key events — synthesizing Qt-style click-chain doubles because `PointerEvent.detail` carries no click count, and opting out of JupyterLab's context menu so right presses reach the crosshair gesture.

Two invariants keep that view alive in a real browser: presentation styling lives on a view-owned wrapper element, because ipywidgets asynchronously rewrites Layout-managed inline styles on the view root; and `RasterPlotHost` republishes a coalesced no-op PUBLISH after every session surface commit it does not own, because `attach_host` routes application-level session mutations through the CONTROL dispatch, which renders without capturing a front — a stale front fails the pointer layout gate and interaction dies silently.

Because closing a widget model removes every live view from the Notebook DOM, `NotebookView.display()` publishes through a recorded display id and `close()` swaps the widget output in place for the final front encoded as PNG; the plot cell keeps its last frame after close and across notebook save/reopen.

交互图形由 backend-neutral `SelectorScene` 唯一描述：scene builder 生成 immutable
line/marker/text primitives、实时数值和 z-order，唯一的消费者是 matplotlib artists
（`_update_selectors` 把 committed 与 candidate 状态一起物化进 raster）。前端不再
存在第二套绘制器，Notebook 与 GUI 天然共享同一套 Area/X/Cross/threshold/color-limit
几何与文字样式。

Qt 只支持 PyQt5。`ensure_qt5_application()` 必须在首个 `QApplication` 创建前设置 High-DPI 属性并注册 Helvetica Light；named preset 保持相同逻辑尺寸，DPR 越高，backing store 使用的物理像素越多。普通应用通过 `RasterPlotHost.from_plot()` 把 immutable data/spec 交给 worker；只有自定义 `PlotSession` 子类使用 raw factory constructor。host 不公开 session，所有 Matplotlib mutation、draw 与 RGBA capture 都留在 serial worker。worker-originated immutable front 通过 queued Qt signal 进入 owner thread，首帧、live revision、fit overlay、preset 切换与 Notebook 共用 session redraw path。

`Qt5PlotWidget.set_interaction_enabled(False)` 只关闭输入 transport，使 Pulse
Preview 或 Edit surface 可以把 wheel 留给外层 scroll area，而不停止 live、DPR
或参数更新。Same-shot group display 由应用持有 causal join：各 widget 使用
`auto_present=False` 暂存独立 host 的 `RasterOperation.front`，整组齐备后在 Qt
owner thread 调用 `present_front()`。`RasterIdentity` 的 generation/revision 用于
核对来源；Image front 还携带 overlay revision。它们不冒充跨 signal 的 same-shot key。

`RasterPlotHost` 的 worker adapter 为每个公开操作提供一个显式委托方法；提交模式
（CONTROL/PUBLISH/PRESENTATION）与 coalesce key 作为 dispatch 参数传入。查询和
prepare/finalize 不发布 front，普通热更新 latest-only 合并，selector/fit 保序，一次
live commit 则按 capture → promote → finalize 原子发布。宿主提交完整表单时只调用一次
`configure()`；semantic/display/size/Image overlay 的差异与 `RenderEffect` 合并都在
session 内完成，同步部分最多发布一张 front，宿主不循环调用单字段 setter。facade 保留显式公共签名，
没有动态属性转发或另一份 GUI dispatch switch。

六个 authored plot kind 通过 `_kinds` 的闭集 `KindHandler` 注册表连接 projection
payload、renderer 更新和 fit target。`PlotSession` 仍是唯一公开 facade；kind 模块只
拥有语义路由，不复制 DataView、fit solver 或 Matplotlib artist 实现。新增 kind 必须
同时注册完整的 spec type、payload builder、renderer handler 和 fit target，缺件会在
registry contract test 中失败。

Image limits and aspect also have one authority. Pan, wheel zoom, API viewport
and reset mutate the same canonical limits. Equal aspect is resolved from the
physical scale of compatible canonical x/y units, so different display prefixes
do not turn a physical circle into an ellipse; incompatible dimensions fall
back to numeric aspect and reject the radial fit model.

Histogram bins likewise have one session-owned canonical domain. `tight`
recomputes it for each revision; `normal` and `fixed` retain the existing edges
while data remain inside them and only expand a breached side. They never shrink
or recenter on alternating live frames, so the x limits do not oscillate.

PulseTimeline creates all dynamic lines, rectangles and annotations through
clipped artist factories. Pan/zoom/reset therefore cannot leave scan badges,
DAC segments or repeat brackets in the figure margin; axis-transform effects
always use the current limits and one complete redraw.

Fit 在 canonical data 上计算并保存 immutable result；显示层把 fitted coordinates、参数和 uncertainty 一起转换到当前 display units，在图内绘制 fit curve、公式与 `±` 不确定度。annotation 使用固定 axes-fraction anchor、独立 3.25 pt font tier 和高于 data/fit curve 的 z-order；selector 或 viewport 改变不再触发障碍物搜索，所以 drag 与 live 更新不会让文字位置抖动。每种 selector 最多存在一个，默认 fit authority 严格按 `AREA > X_RANGE > viewport > all` 解析；`selector_kind` 可显式绑定 area、x-range 或 threshold。Histogram threshold classifier 独立拥有自己的 bimodal classification result、threshold、严格合计 100% 的 fitted-population L/R 百分比和 Fidelity projection；普通 fit 不创建或修改 classifier，classifier 也不进入普通 fit 状态。只有显式以 threshold 为 selector 的普通 fit 才读取当前 threshold scope。authority 保存 canonical geometry，单纯切换显示单位不会把同一批样本误判为过期。Curve/Rolling/1D Facet 直接消费 DataView 的第一条 immutable projected series；group、reduction 与 valid mask 只在 DataView 中评估一次，selector/viewport 再筛选这条实际显示的 series，Fit 不维护第二条 raw mask/reduction 真相源；Rolling 还限制到当前可见 window。selector/viewport 只限定参数估计样本，accepted 参数生成的 overlay 覆盖完整显示域。每个 accepted result 同时保存 data revision 和 fit-context generation；只有两者都匹配才是 current。

Fit models use a semantic `FitTarget` catalogue rather than frontend arity guesses.
The registry permits at most one default per Series/Histogram/Image target; the
built-in registry supplies all three, and a session then filters its target by
coordinate-unit compatibility. The built-in radial Gaussian regular-image path
retains the native 2-D view and canonical authority without eagerly allocating
flattened coordinates or all-pixel indices. Its specialized solver uses
separable Gaussian terms and bounded stripes for exact all-pixel objectives,
robust losses and covariance, without a full meshgrid or dense Jacobian. Custom
Image models remain on the general expansion/solver path unless they provide
their own specialization.

Live data-fit 使用一条 capacity-one clock lane：owner 冻结 base data revision、display/selector/view authority 和 causal Image overlay authority，analysis worker 在 isolated projection context 中为 incoming immutable snapshot 构建 payload、FitSelection 和结果；owner 最后通过一次 CAS install 与一次 draw 提交 data+matching fit。这样 history-dependent histogram domain 不会从过期 base 提交，准备中的 ImageFrame 也不会倒退覆盖独立接受的新点层。Raster control dispatch 不发布 prepare/finalize 阶段，只有 commit 产生新 front。慢 fit 不暴露半帧，中间 ingress revision 由 latest-only mailbox 合并。selector、viewport、unit 和 resize 不启动独立 fit lane：它们提交自己的完整交互/显示 front，保留旧 fit；下一条 data revision 冻结当时的全部 authority，和对应 fit 一起提交。显式绑定 selector 被删除时，其 request generation 立即退休、solver cancellation event 被置位、overlay 清除；自动选择请求则下一条 data revision 使用剩余 authority。

Manual fit and data-frame fit completions use the same host presentation transaction:
accept and render, capture/promote the complete raster front, then publish the
`FitEvent` and resolve the logical fit Future. Selector motion has no fit
completion lane; it only updates the selector scene. This ordering also applies when
an analysis Future is already complete while a raster-worker task is still on
the stack; an event can never announce a fit that the current front does not
yet contain.

## Style and runtime configuration

`PlotLibraryDefaults` combines immutable `PlotStyleConfig`, `PlotLayoutConfig`, `LiveDefaults`, `RuntimeDefaults` and `InteractionDefaults`. rcParams, palette, artist tokens, font tiers, split ratios, preset geometry, refresh choices, pointer cadence, selector hit radius and wheel zoom factor each have one typed owner. Renderers and sessions consume those tokens instead of embedding backend-specific literals. Fixed geometry, immutable snapshots, monotonic revisions and latest-only capacity one are package invariants, not misleading switch-like fields.
