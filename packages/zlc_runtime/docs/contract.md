# zlc_runtime 当前 public API contract

> 本文件描述当前 monorepo 中 `zlc_runtime` 的公开边界，用于调用方与实现保持一致；它不是仓库 Goal，也不覆盖根目录 `ARCHITECTURE_DESIGN.md` 与 `IMPLEMENTATION_PLAN.md`。

## 顶层 facade（allow-list，≤23 名）

`BoardScheduler` `DatasetCoverage` `DatasetOutputDeclaration` `FinalDatasetOutput` `HarmonicClock` `LiveDatasetOutput` `MonitorCoverage` `OwnerChannels` `SurfaceBatchArbiter` `SurfaceUpdate` `SignalDataPlane` `SignalValue` `SignalPublication` `SignalDescription` `AcquisitionStream` `NodeHost` `SelectionBridge` `__version__` `SelectionChange` `SelectionRange` `SelectionState` `FitEventValue`

`MAX_PUBLIC_NAMES = 23` 是真实包级公开命名空间的机械上限；守卫从
`dir(zlc_runtime)` 中排除子模块对象后计数。`__version__` 是 facade 的版本探针，因
双下划线前缀不计入该公开命名空间计数；`MAX_PUBLIC_NAMES` 本身是守卫元数据，不在
用户 facade 的 `__all__` 清单中。

未从包顶层 re-export 的实现仍由其所属子模块提供：
`SignalFront`、`ExactReservation`、`MonitorTap`、`FollowTap`、`LiveDatasetPort`、
`Node` 与 `RunHandleLike`。

## SignalDataPlane(信号面门面)

宿主/应用可见面(节点宿主↔plane 的契约):

```
reserve(producer: SignalProducer) -> StreamGenerationId
retire(producer: SignalProducer) -> frozenset[str]
attach(producer: SignalProducer, live_slot: LiveDatasetPort) -> None
detach_live(producer: SignalProducer) -> None
mark_changed(producer: SignalProducer, live_slot: LiveDatasetPort) -> None
publish_final(
    producer: SignalProducer,
    outputs: Mapping[str, FinalDatasetOutput],
) -> Mapping[str, SignalValue]
latest_publication(signal_key: str) -> SignalPublication | None
set_front_signals(signal_names: Iterable[str]) -> None
attach_latest_only_processor(
    node: LatestProcessorControl,
    *,
    source_name: str,
    initial_publication: SignalPublication,
    coherent: bool = True,
) -> None
cancel_latest_only_processor(control: LatestProcessorControl) -> bool
withdraw_processor(control: LatestProcessorControl) -> None
publish_processor(
    control: LatestProcessorControl,
    outputs: Mapping[str, LiveDatasetOutput],
    *,
    source_publication: SignalPublication,
    trigger: tuple[str, int] | None = None,
) -> Mapping[str, SignalValue]                 # parents forced to (source,)
freeze() -> SignalFront                             # one coherent pull-model front
bind_owner_wake(callback: Callable[[], None]) -> object
unbind_owner_wake(token: object) -> None
close() -> None
```

The following narrow lineage and generation seams are also part of the plane
contract: `direct_parent_publications(publication) ->
tuple[SignalPublication, ...]` resolves the exact retained parent payloads;
`follower_edges() -> frozenset[tuple[str, str]]` lists the (source signal,
follower signal) pairs of live presentation-paced routes; and
`withdraw_processor(control) -> None` removes a
latest-only processor binding after its lane has acknowledged cancellation or
failure.  `set_front_signals` declares the signal family requested by the
consumer; same-shot fallback is enforced for that declared set.  The
`BoardScheduler` is the sole production declaration authority: on every tick
it projects its port list's signal names into `set_front_signals` before
freezing, so the coherent set always equals what the board shows and no
membership bookkeeping exists beside the ports.  An undeclared plane builds
no lineage components and every signal floats at its own latest publication
— the camera-ahead-of-its-derived skew this declaration exists to prevent.

`attach_latest_only_processor(..., coherent=False)` declares a
presentation-paced follower: a route whose publications advance only AFTER
its source was presented (a panel's accepted-fit signals; the lane's
recompute of such a route raises stale rather than recutting).  A follower
keeps full `direct_parent_refs` lineage but never joins its source's
same-shot component — holding the source's selection for it deadlocks the
component: the source waits for the follower that waits for the source's
next presentation.  Autonomous processors (occupancy, committed-selection
recuts) stay `coherent=True` and are exactly what same-shot holding is for.

A follower can PUBLISH trailing: a fit for shot N may be accepted when the
route is already at N+1.  Its exact parent is resolved by PROVENANCE, not
retention: the panel port that rendered the fit still holds publication N
(pending or presented — the accept fired inside revision N's own commit),
and `SelectionBridge(..., source_publication_for=resolver)` asks that port
by data revision.  Every accepted fit publishes with a truthful parent
(fit@N names camera@N); the plane keeps its weak payload-lifetime contract
with no retention window.  A fit signal's dataset revision is the bridge's
own strictly-increasing batch counter, so follower consumers (a rolling
trace) always see advancing revisions.  `publish_processor` accepts any
issued same-generation parent whose sequence does not regress.

Presentation is SHOT-COHORT based: one lineage group shows one shot number
on screen and flips as one.  Every display tick the scheduler stages each
due panel individually and stamps the staged batch with its shot-root set
(the parentless ancestors reached by walking
`direct_parent_publications`; unresolvable lineage presents solo).  The
arbiter assembles equal-root batches into one `_ShotCohort` and presents a
complete sealed cohort in ONE owner-thread accept pass — several views of
one signal, a camera and its derived panels, and a panel's own fit trace
all flip together.  A superseded member abandons the WHOLE cohort (the
group skips to the newest shot together, never half a board one shot
ahead); cohorts sharing a panel present strictly in formation order, and
cohorts with disjoint panels never wait on each other, so a slow solo
panel cannot throttle the board.  When both sides of a follower edge are
displayed, a cohort holds a join window for exactly those follower panels:
the follower's batch stages one tick after its source pair's commit, the
window closes the moment every follower joined (typical cross-panel phase
≤ 1 beat), and two tick boundaries after the cohort's own work completed
serve as the fallback for a follower that never stages (a not-due slow
panel must not hold its shot open).  `tick_boundary()` is called by the
scheduler at the end of every tick and is what advances these windows.

**核心不变量(派生族 same-shot,自动,非 API)**:任何 `SignalFront` 内,派生信号(ROI/fit/occupancy…)的值必源自当前 front 中其祖先的同一根 publication(沿 `direct_parent_refs` 追根,根集一致),传递到任意深度;族不齐整族回退上一完整拍,绝不撕裂。跨 producer 不承诺,无全局 shot counter。

值类型：`SignalPublication`（`event_ref`、`direct_parent_refs`、兄弟输出 Mapping）、`SignalValue`（`name`、immutable `snapshot`、`coverage`、`transient`、run-time `run_record`，并从 snapshot 投影 block/schema/values）、`SignalFront`（signals/publication_by_signal）。`EventRef = (stream_id, generation, sequence)` 是 frozen lineage 身份三元组。`direct_parent_refs` 是 lineage 身份，不属于设备回传对账协议。

## 流与消费口(zlc_runtime.streams)

```
AcquisitionStream.create(...) -> (stream, producer)     # producer 持排他写/终止权
stream.reserve(total_events) -> ExactReservation        # 首发布前接入;一 generation 恰一 formal exact
  reservation.activate() -> cursor; cursor.next() -> Delivery; acknowledge_delivery(...)  # ack 即水位,绝不丢,缺口=StreamGap 异常
stream.monitor() -> MonitorTap                          # next(timeout) 有序；latest() 只保留当时最新值，不计 loss
stream.follow() -> FollowTap                            # 可中途加入,无 latest(),不回放
异常族:StreamGap / StreamEndedEarly / SchemaChanged / SourceFailed
LiveDatasetPort:bind(dataset) / set_change_listener(cb) / updated() / freeze_current() -> snapshot / fail(...) / source_terminal(...) / close()
```

## 节点宿主（zlc_runtime.host）

```
Node 协议没有 role -> runtime-kind 映射：
  未绑定 source：worker，提供 execute(ctx)
  绑定恰好一个 source：processor，提供 evaluate(SignalValue) -> 非空 output Mapping
  processor 的 finite FollowTap / retained final one-shot / infinite latest 路径由 source extent 决定
ctx 能力 = generation / cancel_requested / start_and_wait(starter) /
           attach_live_outputs(...) / open_live_dataset(...) / open_exact_dataset(...) /
           publish_final(outputs) / warn(text)
NodeHost:start() / cancel(reason) / poll() / shutdown();observation 只读状态投影;
  声明了输出却没发布 = 硬失败;同 host 可重启(generation 计数防陈旧完成)
RunHandleLike 协议:snapshot() / cancel(reason) / result()      # 硬件执行引擎在域侧,本包只经此相望
```

## 呈现调度(zlc_runtime.presentation)

```
WakeSink 协议:request_owner_wake()                      # Qt 垫片(QtOwnerWake)在 zlc_ui
OwnerChannels:notify_lifecycle()/notify_surface()/activate_data(plane)/take()->OwnerTurn/close()
HarmonicClock:base_ms(最小允许interval) / advance()->elapsed / group_due(elapsed, intervals)->bool
SurfacePort 协议:panel_id/signal_name/display_interval_ms/presented_publication()/
  prepare(value, publication)->SurfaceUpdate|None /observe/can_accept/accept/reject/
  finish_unpresented/report_waiting(missing)
SurfaceBatchArbiter:enqueue_group(ports, front)->bool / drain(resolve) / cancel_all()
  # all-or-nothing 入批;整批 done 才上;任一失格整批弃;组级 owed 位:流拍后每 base tick 重试
BoardScheduler:on_tick()->SignalFront / on_owner_turn(poll_lifecycle)
```

## SelectionBridge (`zlc_runtime.selection_bridge`；核心类型同时位于顶层 facade)

`SelectionBridge` 与教程实际构造的四个纯数值载荷
`SelectionChange`/`SelectionRange`/`SelectionState`/`FitEventValue` 一起进入顶层
facade；协议与 `FacetCondition` 等扩展类型仍只从
`zlc_runtime.selection_bridge` 导入。它是 plot/session 与 signal plane 之间的纯数值
过桥，runtime 不 import `zlc_plot`。

```
SelectionChange = "added" | "updated" | "committed" | "removed"
SelectionRange(axis: str, lower: float, upper: float, coordinate_frame: str | None)
FacetCondition(axis: str, value: int | float | str)
SelectionState(
    plot_kind: "image" | "curve" | "histogram",
    selector_kind: "area" | "x_range",
    ranges: tuple[SelectionRange, ...],
    facets: tuple[FacetCondition, ...] = (),
    repeat_index: int | None = None,  # 结构性限定 repeat 轴到第 k 行：repeat 轴从不按名寻址
    revision: int = 0,
)
FitEventValue(
    parameter_names: tuple[str, ...],
    parameter_units: Mapping[str, str],
    parameter_values: Mapping[str, NDArray[float64]],
    parameter_errors: Mapping[str, NDArray[float64]],
    success: NDArray[bool],
    sample_axis_name: str,
    sample_coordinates: NDArray[float64],
    sample_unit: str,
    sample_labels: tuple[str, ...] | None,
    source_revision: int,
    batch_revision: int,
)
```

`FitEventValue` is one parameter table with an optional sample axis; there is
no separate scalar fit API.  Every parameter value/error array is float64 of
length `N`, every parameter key is present in both mappings, and `success` is
the value-dataset per-sample validity criterion.  Failed cells carry `NaN`
values and covariance-invalid errors carry `NaN` errors.  Value validity is
`success`; error validity is `success AND isfinite(error)`, and consumers use
the explicit validity masks rather than inferring validity from `NaN`.  An
empty unit string means dimensionless.  The sample axis has a name, `N`
float64 coordinates, and a unit.  Numeric facet coordinates are carried
directly; text coordinates use `0..N-1` coordinates and retain their labels
in `sample_labels`.  A scalar fit is the degenerate `N=1` table with
`sample_axis_name=""`, coordinates `[0.0]`, empty unit, and no labels.  A
one-cell facet is also legal and retains its non-empty sample-axis name.
`source_revision` identifies the source data used by the solver and remains
constant when the same source is fit again.  `batch_revision` identifies the
publication itself and strictly increases for every accepted fit batch.

The two injected protocols are:

```
SelectionEventSource:
  subscribe_selection(callback(change: SelectionChange, state: SelectionState)) -> unsubscribe
  subscribe_fit(callback(event: FitEventValue)) -> unsubscribe
SelectionDataReader:
  selector_data(kind: str) -> SelectionState
```

The concrete plot adapter converts its own event/state objects into these
numeric values before invoking the callbacks. It never passes plot, Qt, or
fit-engine objects into runtime. `selector_data(kind)` is used to obtain the
current canonical axis names/ranges and facet value; it does not transfer the
plot's sliced array—the bridge always slices the bound upstream signal
snapshot.

```
SelectionBridge(
    plane: SignalDataPlane,
    source_signal: str,
    selection_source: SelectionEventSource,
    selection_data: SelectionDataReader,
    *,
    bridge_id: str,
    source_publication_for: Callable[[int], SignalPublication | None] | None = None,
) -> SelectionBridge
start() -> None
close() -> None
```

The bridge binds `source_signal` at `start()` to the current publication and
subscribes to both event streams. `UPDATED` is ignored. `COMMITTED` performs
one immediate cut; each later source publication performs a latest-only
re-cut. `REMOVED` withdraws the derived generation; a later committed state
may create it again. Close unsubscribes and retires all bridge outputs.
`source_publication_for` resolves a fit's exact parent publication by data
revision: the panel port that rendered the fit is the deterministic causal
holder (the accept fires inside that revision's own commit), so the console
passes its port's `publication_for_revision`.  Without a resolver the bridge
accepts only a fit whose revision still matches the current publication; a
fit trailing a panel that already advanced is superseded flow control and is
dropped silently.

For an image `area`, the bridge publishes
`@logic/{bridge_id}/roi_frame` (the selected 2-D role-axis sub-box) and
`@logic/{bridge_id}/roi_value` (the finite mean scalar). For a curve or
histogram `area`/`x_range`, it publishes only `roi_value`. Fit completion
publishes `@logic/{bridge_id}/fit_{parameter}` and
`@logic/{bridge_id}/fit_{parameter}_error` for every parameter; each is a
vector Dataset with one point row per sample coordinate.  The value and error
columns use the corresponding parameter unit, text sample coordinates are
materialized as numeric indices with a second
`{sample_axis_name}_label` TEXT point column carrying the original labels,
and value/error validity are materialized independently from the event.
Their declaration contracts are respectively
`zlc.selection.fit.parameter.v2` and `zlc.selection.fit.error.v2`.
The scalar `N=1` table therefore remains bitwise equivalent to the old scalar
shape while using the one vector path.
Every derived publication has the exact source `SignalPublication` as its
direct parent, so the normal same-shot lineage rule remains automatic.

`publish_processor(..., trigger=...)` keeps the existing one-result-per-source
sequence rule when `trigger is None`. A non-`None` opaque `(kind, revision)`
trigger permits one publication for each distinct trigger on that same source
sequence; a duplicate trigger is rejected. SelectionBridge also drops a
late worker result whose trigger is older than the currently accepted control
event. This is the control-event seam used only by SelectionBridge for a
committed selection or accepted fit event.

## 负面清单(并行仓不得引用,本包永不提供)

- `SignalAssociationRequest` / `SignalEventAssociationCursor` / `SignalEventAssociationSource` / cause_digest 对账 / arm-bind-next-finish 四步协议 / event-derived 世代 —— 设备回传对账族,已判死。
- `bind_generation_source` / `release_generation_source` / `withdraw_dependency_closure` / `finish_dependency_retirement` / `require_active_generation` —— preemption 族,已删,显式 stop-then-start 替代。
- 全局 shot counter、任意跨 producer same-shot 组、PointExecutor/SampleCollector 类 scan 协议。
- 任何 Qt 类型;任何 zlc_plot/zlc_ui/设备类型。
