# Step 6-D：`zlc_runtime` / `zlc_plot` 遗漏符号、测试与产品边界清册

状态：完成（只读审计；没有修改代码、旧文档或硬件）
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：`packages/zlc_runtime/src/zlc_runtime`（16 个 source 文件，约 9,761 行）、`packages/zlc_plot/src/zlc_plot`（52 个 Python source 文件，约 32,427 行）及两包全部 69 个 Python test/helper 文件。
本报告是差集审计：`AUDIT/02-plot-fit-overlay-selector.md` 已深入 PlotSession/RasterHost/render/fit/overlay/selector；`AUDIT/03c-runtime-contract-prune.md` 已深入 runtime streams/dataset/live/host/plane。这里引用其结论，只补此前未逐文件、逐 public symbol、逐 test 归档的部分。

## 1. 结论先行

这两包不是都该推倒重来：当前产品确实需要 runtime 的 Host/Signal plane/front/presentation/selection 桥，以及 plot 的 data projection/spec/semantics/layout/render/session/raster/Qt 主链。问题是它们外面又包了三圈没有产品消费者、由 tests 自己证明自己的通用框架，并夹着少数真实会卡 UI 或破坏 owner 约束的实现。

本轮差集裁决：

| 区域 | 裁决 | 说明 |
|---|---|---|
| runtime `_failure.py`、`cleanup.py`、`_public.py` | `DELETE`（随 03c dead exact/live/run 框架） | 没有独立产品消费者；只被 dead framework 或 tests 保活。 |
| runtime `owner_mailbox.py` | `KEEP + SIMPLIFY` | Future queue/completion/wake 是真实 worker owner；`RunHandleLike`/handle 状态属于无人调用的 `start_and_wait()` 支线。 |
| runtime `front.py`、`presentation.py` | `KEEP + REDESIGN` | 是产品主链；本报告只补 public/debug surface 与 owner 重复，不重复 02/03 的 same-shot/阻塞问题。 |
| runtime 大型 streams/dataset/live/preview | 按 03c：`DELETE/MERGE` | 不再重复约 4,000 行 dead exact/monitor/builder/live-port 证据。 |
| plot `live.py` + `_live_channel.py` | 当前产品边界下 `DELETE` | 仓内 production 0 caller；在 RasterHost capacity-one 与 BoardScheduler cadence 外又造一条 ingress/cadence/thread pipeline。 |
| plot `notebook.py` + `api.show()` | `USER DECISION` | 当前产品 0 caller；若保留为正式 standalone backend，必须先修复直接访问 Host-owned PlotSession 的 owner 违规。 |
| plot `qt_controls.py` | `KEEP + REDESIGN` | FigureViewer 真用；Qt slot 内同步 `.result(timeout=10)` 可冻结 GUI。 |
| plot fit engine/models/radial solver | `KEEP + REDESIGN` | 算法主链真实；删 test-only numeric table/public mutable registry，并收回无 shutdown 的全局 4-thread pool。 |
| plot layout/ticks/units/style/specs/semantics | `PASS/KEEP` | 责任边界基本正确，没有发现第二套产品实现；mutable extension surface 是否保留取决于 D-012。 |
| 两包 root facade | `KEEP lazy mechanism + PRUNE exports` | `MAX_PUBLIC_NAMES` 是 test-only production 常量；public-surface tests 目前在给历史宽度背书，不是产品需求证据。 |

本轮新增的最高风险不是“代码太长”，而是以下四项：

1. FigureViewer 的 display editor 会在 Qt owner thread 上同步等 raster worker，最长 10 秒；
2. Notebook backend 绕过 RasterPlotHost，直接调用已由 worker 接管的 PlotSession；
3. regular-image fit 的 module-global 4-thread pool 永不 shutdown，并与每 panel 的两个 executors 叠加；
4. 没有产品 caller 的 `LivePlotController` 又增加一条 cadence thread 和 latest queue，模糊谁拥有 live 节流。

## 2. 审计判定规则

- `production consumer`：`zlc_atom/src`、`zlc_workbench/src` 或同包真实主链可到达；tests、examples、notebooks、旧文档不算保留理由。
- `PASS`：职责与层级清楚，无新增实质缺陷；不表示该文件永远不可简化。
- `KEEP`：当前产品需要；`INTERNALIZE` 表示实现要留，但不需要 root/public 承诺。
- `REDESIGN`：产品需要但契约或线程/性能语义不应继续。
- `DELETE`：当前产品没有消费者，或只服务另一段已判 dead 的实现。
- `USER DECISION`：取决于 `AUDIT/DECISIONS.md` D-011/D-012；不得由 tests 或旧 README 代替用户选择。

本轮使用静态 import/call-site 反查和逐 source/test 阅读；没有重新跑 02/03 已做过的性能 probe，也没有运行硬件或修改任何实现。

## 3. 新增高风险发现

### RP-01 — FigureViewer 的 Qt display editor 在 GUI slot 中同步等 worker

`zlc_plot/qt_controls.py:503-557` 的 `_Qt5PlotControls` 在构造、`parameterEdited`、`semanticEdited` 三条 Qt 同步路径都调用 `_described()`；后者在 `:529` 执行：

```python
pending.result(timeout=self._timeout).value
```

默认 timeout 是 10 秒。真实产品 `zlc_workbench/apps/figure_viewer.py:73` 通过 `edit_plot_display()` 打开它；Host 的 `describe_display()`、`set_parameter()`、`apply_semantic()` 都返回排入 raster worker 的 Future。于是 worker 正在 render、fit、处理旧 frame 或关闭时，GUI 事件循环会直接冻结；这也解释“编辑参数时卡住”可以发生在没有 acquisition 的离线 FigureViewer。

当前 `zlc_plot` tests 没有覆盖 bound controls 的异步完成，只锁了 public 名字和独立 ParameterPanel。裁决：`REDESIGN`。Qt slot 只提交操作，用 done callback/signal 回 owner 更新 accepted description/error；打开 dialog 时也不能在构造函数里阻塞。拒绝结果和 superseded operation 要有显式 UI 状态，不可吞成十秒无响应。

### RP-02 — NotebookView 破坏 RasterPlotHost 的唯一 owner

`NotebookView.__init__()` 在 `notebook.py:409` 通过 `RasterPlotHost.from_session(session)` 把 session 交给 raster worker；但随后：

- `describe_display()` 在 `:448-451` 直接调用 `self._session.describe_display()`；
- `describe_semantics()` 在 `:453-456` 直接调用 session；
- `replace_spec()` 在 `:458-467` 直接调用 session；
- pointer 路径反而正确地通过 Host queue。

同一个 mutable PlotSession 因而可能同时被 notebook/kernel thread 和 raster worker 操作，违背 RasterPlotHost 的单 worker owner 设计。当前 Workbench/Atom production 没有 NotebookView 或 `plot.show()` caller，所以不会解释现有 Workbench 卡顿；但若 D-012 选择正式支持 notebook，这是 release blocker。

裁决：`USER DECISION + REDESIGN`。保留时 NotebookView 只能调用 Host 的 Future API，并异步消费完成；不允许持有一条可变 session 旁路。删除 standalone/notebook 产品面时，一并删 `notebook.py`、`api.show()`、notebook availability/wake helpers 和对应 tests。

### RP-03 — `LivePlotController` 是没有产品消费者的第二套 live cadence owner

`live.py`（746 行）和 `_live_channel.py`（129 行）实现 revision channel、capacity-one latest、独立 `zlc-live-plot` thread、refresh interval、prepare/solve/commit/publish/abort、metrics/error/retry。`RasterPlotHost.live_controller()`/`NotebookView.live_controller()`只是 factory；反查 `zlc_atom/src`、`zlc_workbench/src` 没有一个 caller。

真实 Workbench live 链已经有：runtime slot/plane → BoardScheduler cadence → RasterPlotHost capacity-one queue → PlotSession live transaction。启用这个 controller 会在外面再加一次 latest 丢弃、一次 refresh cadence 和一个 thread，并使“runtime 还是 plot 决定何时显示”没有唯一答案。

裁决：当前产品边界 `DELETE`。若 D-012 选择把 zlc_plot 作为独立 live plotting library，必须把它与 RasterPlotHost ingress 合成一个 owner，而不是保留双 cadence；tests 需要从 fake session 自洽测试升级为一条真实 backend acceptance。

### RP-04 — regular-image fit 隐藏了一个永不关闭的全局线程池

`_fit_radial.py:79-96` 懒建 module-global `_STRIPE_POOL(max_workers=4)`，没有 shutdown/atexit/package lifecycle。每个 PlotSession 又在 `session.py:467-475` 创建 fit executor 与 live-prepare executor，RasterPlotHost 还有自己的 worker。多 panel 会形成“每 panel 固定线程 + 全进程隐藏 4 线程”的混合 ownership；关掉所有 panel 后全局 pool 仍在。

这不是要求删 regular separable image solver：它是大图 fit 的真实优化路径。裁决：`KEEP solver + REDESIGN executor ownership`。推荐由明确的 application/session service 提供有界共享 compute pool，统一 shutdown；或将 stripe 并行并入现有 fit executor，避免 nested oversubscription。不要再增加第三种 pool abstraction。

### RP-05 — runtime mailbox 把真实 Future owner 与 dead RunHandle compatibility 混在一起

`RunOwnerMailbox.submit()/drain_completions()/worker_idle/shutdown()` 被 worker、frozen processor、follow processor 真实使用，应保留。`begin_generation()/owner_reaped` 也用于这些真实 Future generation，不能整体误删。

但 `_public.RunHandleLike`、mailbox 的 `_handle`/`handle`/`set_handle()`，以及 Host 的 `NodeExecutionContext.start_and_wait()`/`_start_and_wait()` 没有 production caller；只有这一支需要 hardware-like handle 的 snapshot/cancel/result。它使 `NodeHost.cancel()/poll()` 和 mailbox 表面同时存在两套 worker 状态源。

裁决：`KEEP + SIMPLIFY`。删除 handle-only 支线；保留 generation、tracked futures、completion queue、owner-reaped lifecycle。`OwnerCompletion` 可变成私有内部 carrier，不应作为 package API。

### RP-06 — tests 正在把无消费者 surface 伪装成稳定产品 API

明确的 test-only/closed-world surface：

| 符号/模块 | production consumer | 当前保活来源 | 裁决 |
|---|---:|---|---|
| runtime `MAX_PUBLIC_NAMES` | 0 | `test_import_guards.py` | `DELETE` 常量；直接测试允许名单。 |
| plot `MAX_PUBLIC_NAMES` | 0 | public surface tests | 同上。 |
| `_failure.DetachedFailure`/helpers | 0 独立 | dead dataset/live paths + tests | `DELETE`。 |
| `cleanup.CleanupReport/run_cleanup_steps` | 0 | `test_cleanup.py` | `DELETE`。 |
| `FitNumericTable`、`FitResult.table`、`FacetFitBatchResult.table` | 0 | `test_fit_numeric_table.py` | `DELETE` surface；保留真实 FitResult fields。 |
| public mutable `FitModelRegistry` | 0 custom registration | fit-engine construction + tests | `INTERNALIZE/FREEZE` builtin catalog。 |
| public mutable `UnitRegistry` | 0 custom registration | package DEFAULT_UNITS + tests | D-012 决定；当前产品可 `INTERNALIZE/FREEZE`。 |
| `schema_dtype()` | 0 | 自己的 `__all__` | `DELETE`。 |
| `panel_kinds()` root export | 0 | tests；Workbench 自有较窄 catalog | `DELETE` root export，避免第二 catalog。 |
| `notebook_available()/qt5_available()` | 0 | backend tests | `DELETE` convenience；真实 construction 已报告明确异常。 |

tests 应验证被选定的产品契约，而不是“因为已经有 test，所以 production symbol 必须永存”。

### RP-07 — renderer 拆包停在半途，形成两个组织规则

`rendering.py` 仍有约 4,555 行，拥有 curve/image/histogram/facet 等全部 Matplotlib 更新；只有 pulse 被移到 `_rendering/pulse.py`（449 行），`_rendering/__init__.py` 为空。同时 `_kinds/*.py` 又是另一套 per-kind dispatch adapter，内部调用 renderer 私有方法并直接填 projection 私有字段。

这不是运行时 bug，也不建议再造新层。裁决：`KEEP WITH DEBT`，用户二选一：要么把 pulse 合回单一 renderer；要么利用现有 `_kinds` 边界完成各 kind rendering 的对称拆分并删掉旧 private cross-calls。默认偏向前者，除非后续修改频率证明按 kind 拆分能实质降低冲突。

## 4. `zlc_runtime` 逐文件/顶层符号清册

以下“03c”表示其内部逐函数证据已在 `03c-runtime-contract-prune.md`，这里不重复。

| 文件 | 顶层类/函数（private helper 合并列示） | 真实消费者/职责 | 裁决 |
|---|---|---|---|
| `__init__.py` | 23-name eager facade、`MAX_PUBLIC_NAMES` | Workbench/Atom 大量从 root 取核心 DTO/Host/plane/selection；也混入 dead `AcquisitionStream` | `KEEP + PRUNE`；删 test-only max 常量与 dead export。 |
| `_failure.py` | `DetachedFailure`、`safe_error_summary()`、`detach_failure()`、`record_secondary_failure()`、`_safe_text()` | 只服务 dead preview/builder/live-port 或 tests | `DELETE`。错误若跨线程，Future 已保存原 exception；产品 UI 只需 owner 层一次格式化。 |
| `_public.py` | `RunHandleLike` Protocol | 仅 Host 无 caller 的 `start_and_wait()` 与 mailbox handle 支线 | `DELETE`。 |
| `cleanup.py` | `CleanupReport`、`run_cleanup_steps()`、`join_worker()` | 前两者 tests-only；`join_worker()` 只被 dead `_ExactDeltaLivePort` 使用 | `DELETE`。真实 Host/mailbox shutdown 继续由 owner 自己负责。 |
| `dataset_output.py` | `DatasetOutputDeclaration`、`FinalDatasetOutput`、`LiveDatasetOutput`、两个 Protocol、`single_live_dataset_output()` | 三个 DTO 是所有 node/plane 真核心；Protocols 和 single helper 只连 dead LiveDatasetPort/无 caller | `KEEP + SIMPLIFY`；按 03c 合并重复 carrier，删 dead Protocol/helper。 |
| `dataset.py` | cell/schedule/edge、coverage、preview delta、seal/artifact、builder、monitor 与 helpers | 仅 `DatasetCoverage`/`MonitorCoverage` 在产品信号 extent 真用；其余无 caller | 按 03c `DELETE`，coverage `MOVE` 到信号 extent owner。 |
| `front.py` | `_values()`、`_state_for_signal()`、roots/ancestry helpers、`build_front()` | `SignalDataPlane.freeze()` 的真实 coherent-front 算法 | `KEEP`；245 行为同一纯算法责任，暂不再拆。same-shot/lineage问题见 02/03。 |
| `host.py` | `Node` Protocol、`NodeProgress`、`LogicNodeObservation`、`NodeExecutionContext`、`NodeHost`、内部 terminal/start helpers | Workbench logic lifecycle 和三类 processor 真核心 | `KEEP + REDESIGN`；删 exact/live builder 与 start-and-wait 支线；role、publication completeness、failure/Stop 见 03c/03b。 |
| `live_dataset.py` | `LiveDatasetPort`、`_ExactDeltaLivePort`、`_required_message()` | production 0 caller；真实 nodes 全用 plugin-specific live slots | 按 03c `DELETE`。 |
| `output_name.py` | `bare_output_name()` | 唯一 caller 是 `DatasetOutputDeclaration.__post_init__()` | `MERGE` 进 declaration owner 后删文件；单函数文件没有独立 domain。 |
| `owner_mailbox.py` | `OwnerCompletion`、`RunOwnerMailbox` | Host worker/frozen/follow 的真实 Future queue | `KEEP + SIMPLIFY`；见 RP-05。`OwnerCompletion` internalize。 |
| `plane.py` | Signal producer protocols/DTOs/front、generation state、latest lane、`SignalDataPlane` 与 validation helpers | runtime 数据主干 | `KEEP + REDESIGN`；materialization thread、global latest lane、generation contract、identity 等见 03c。内部 Protocol/state 不进 root。 |
| `presentation.py` | `WakeSink`、`OwnerTurn/Channels`、`HarmonicClock`、`SurfaceUpdate/Port`、cohort/arbiter、`BoardScheduler` | Workbench board 真主链 | `KEEP + REDESIGN`；`WakeSink/SurfacePort` internalize；`OwnerTurn`、`pending_cohorts`、`owed_groups`、`last_front` 是 test/debug surface，不需 public promise。OwnerChannels 可收敛 Workbench 的另一套 wake owner，见 02 PLOT-022。 |
| `preview.py` | `LiveDatasetViewSpec`、`ExactDatasetPreviewSpec/Port`、`FailureAwarePreviewPort`、`notify_preview_failure()` | 只服务 dead runtime live ports；不要与真实 descriptor `NodePreviewSpec` 混淆 | 按 03c `DELETE`。 |
| `selection_bridge.py` | statistics/catalog、Selection range/state/facet/fit DTO、source/reader Protocol、`_BridgeProcessor`、`SelectionBridge` | Workbench selection/derived output 真主链 | `KEEP + REDESIGN`；catalog 与 DTO 真用；`SelectionDataReader` 的二次读取/TOCTOU 和 committed event 自包含方案见 02。Protocols 留作 internal structural boundary，不应扩 root。 |
| `streams.py` | ids/refs/envelopes/errors、exact/readiness/cursor/monitor/follow/producer/stream | 真实产品只需 `EventRef` 与 future-publication follow 子集 | 按 03c `MERGE/DELETE`；不要让 `AcquisitionStream` root export 为 dead 泛型框架背书。 |

### 4.1 runtime public facade 的推荐最小方向

当前 root 的 product-real names 是 declarations/output DTO、coverage（迁移前）、Signal DTO/plane、NodeHost、presentation primitives、selection DTO/bridge/catalog。`AcquisitionStream` 没有包外 production caller；`MAX_PUBLIC_NAMES` 不是 API。

不建议继续用一个数字证明 facade “小”：数字允许用替换/重命名绕过，也不能判断名字是否属于同一层。用显式 `__all__` snapshot 足够；更重要的是只保留真实跨包 imports。runtime 目前是 eager facade，dead framework 删除后 import cost 自然下降，无需仿造 plot 的 lazy registry。

## 5. `zlc_plot` 逐文件/顶层符号清册

这里把大类的方法按同一职责族合并；每个 source 文件、每个顶层 class/function family 都有归宿。PlotSession/RasterHost/fit projection/selector 的逐方法行为和已确认 bugs 仍以 02 为准。

| 文件 | 顶层类/函数（private helper 合并列示） | 责任与产品消费者 | 裁决 |
|---|---|---|---|
| `__init__.py` | `_EXPORTS`、`__getattr__()`、`__dir__()`、61-name `__all__`、`MAX_PUBLIC_NAMES` | lazy import 机制确实避免无绘图场景加载 Matplotlib；exports 混合产品、standalone 与 test-only | `KEEP lazy + PRUNE facade`；删 max 常量，见 5.1。 |
| `_axis_transform.py` | `canvas_physical_size()`、`AxisTransform` | Qt/notebook pointer 到 data coordinate 的 immutable front contract | `PASS/KEEP INTERNAL`。 |
| `_fit_projection.py` | `FitScope`、history/authority/selection/histogram/context/projection DTO 与 `accumulate_history()`/budget helpers | Session fit、selector、rolling/facet 的中心投影 | `KEEP + REDESIGN`；复杂度真实，02 已列 authority/atomic/fit问题；不要再复制一套。 |
| `_fit_radial.py` | regular-image summary/status/kernel/context helpers、sampling/objective/information/loss/result、`fit_regular_separable_image()` | FitEngine 的大图 separable model 快路径 | `KEEP solver + REDESIGN pool`；RP-04。private numeric helpers留在同一算法模块合理。 |
| `_fit_scene.py` | `FitPolyline`、`FitEllipseGlyph`、`FitOverlay` | renderer/session 间 immutable fit scene | `PASS/KEEP INTERNAL`。 |
| `_gesture_engine.py` | hit/drag/pan helpers、selector/color/pan gesture states | backend-neutral interaction state machine | `PASS/KEEP INTERNAL`；02 的 gesture/front一致性问题由 Session owner修，不应回到 Qt。 |
| `_image_raster.py` | `ImageFrontPolicy`、`PreparedImageFront`、reduction/window helpers、`prepare_image_front()`、`ImageFrontStore` | 大图降采样/front cache | `KEEP`；单一 image-raster responsibility，现有 performance guards 有价值。 |
| `_live_channel.py` | `IngressMetrics`、`RevisionedItem`、`LatestRevisionChannel` | 只服务无 production caller 的 `LivePlotController` | 当前产品边界 `DELETE`。 |
| `_pulse_time.py` | `_normalized_unit()`、`pulse_content_bounds()`、`pulse_time_scale()` | Pulse Editor timeline 的真实 scale/bounds | `PASS/KEEP INTERNAL`。 |
| `_selector_scene.py` | scene enums/DTO/style/context/owner、label/selector/color primitive builders | Renderer 与 selector controller 间 scene authority | `KEEP`；02 已审状态同步。 |
| `_session_fit.py` | warm/live/deferred DTO、`FitSessionMixin` 全部 fit request/solve/commit methods | PlotSession 的真实 fit orchestration | `KEEP + REDESIGN`；data-first vs atomic、cancel/warm seed等见 02。mixin 是对 4k-line facade 的实质拆分。 |
| `_session_gesture.py` | `GestureSessionMixin` | PlotSession 的 backend-neutral pointer/viewport/selector routing | `PASS/KEEP`，与 `_gesture_engine` 分工为 orchestration vs state algorithm。 |
| `_session_live.py` | `LiveSessionMixin` | RasterHost/PlotSession 当前 prepare/solve/commit transaction 真用；不是 `LivePlotController` 专属 | `KEEP + REDESIGN`；删除 controller 时不能误删该 transaction。 |
| `_session_state.py` | `FitEvent` 与 live/pointer/fit/projection internal DTOs | async Session 状态 carriers | `KEEP INTERNAL`；`FitEvent` 跨 Workbench subscription 可 public，其余保持 private。 |
| `_validation.py` | `finite_real()`、`integer()`、text/optional text、`readonly_copy()` | 多 contract 共用的无状态 canonical validation | `PASS/KEEP INTERNAL`。 |
| `api.py` | `_axis()`/alias helper；`curve()`、`histogram()`、`image()`、`rolling()`、`facet_grid()`、`pulse_timeline()`、`show()` | production 仅 calibration report 调 `curve/image/facet_grid`；其余只属于 standalone/tests | 保留前三个 product constructors；其余由 D-012 决定。`show()` 与 Notebook 一起裁决。 |
| `assets/__init__.py` | Helvetica asset constants、`helvetica_light_path()`、`register_helvetica_light()` | style 与 Qt font registration 真用 | `PASS/KEEP`；package asset 单一 owner。 |
| `backends.py` | backend error/cleanup/gate、notebook/Qt availability与启动 helpers、lazy Qt modules/font/high-DPI、Qt5 widget factory/facade | `Qt5PlotWidget` 是 TaskConsole 真主链；notebook helpers 和 availability convenience 没产品 caller | `KEEP Qt path + PRUNE`；Notebook部分随 D-012；availability root surface删。 |
| `config.py` | validators、`LiveDefaults`、`RuntimeDefaults`、`InteractionDefaults`、`ProjectionDefaults`、`PlotLibraryDefaults`、`DEFAULTS` | DEFAULTS、interaction/projection/runtime 真主链；LiveDefaults 仅 `live.py` 使用 | `KEEP`；若删 LiveController，同删 LiveDefaults 与 refresh literals，避免幽灵配置。 |
| `data_contract.py` | snapshot/schema getters、shape/repeat/point/axis/unit/image/live-grid/equality、implicit coordinates、`AxisDescriptor` 与 descriptor/column/topology helpers | zlc_data → plot 的集中适配层 | `KEEP + INTERNALIZE`；`image_axes` 真跨包使用；`schema_dtype()` 0 caller可删，其余 implementation 不需 root。 |
| `data_view.py` | errors、quantity/coordinate/sample/axis/rolling/curve/image/histogram/facet DTOs、`DataView`、histogram/alignment/reduction helpers | 所有 kind 的 R/P/*D 纯投影核心 | `KEEP WITH DEBT`；约 1,964 行但同一纯领域。若拆，仅沿现有 kind 边界拆普通函数，不新增 registry/layer。 |
| `errors.py` | `ZLCPlotError`、`RevisionError`、`UnitError` | revision/unit contract 明确异常 | `PASS/KEEP`；base hierarchy虽主要给 tests/type grouping，成本很低。 |
| `fit.py` | fit errors/enums/spec DTO、model registry/options/numeric table/input/results/batch/engine；model/jacobian/seed/bounds/covariance；builtin/default registry | fit 数值核心是真实产品；table 和 custom registry surface无产品 caller | `KEEP engine/models + SIMPLIFY API`；删 numeric table，freeze/internalize registry；模型/Jacobian helpers正确归属。 |
| `kinds.py` | `PlotKind`、`AxisDomain`、`AxisRef` | Workbench/Atom/shared semantic identity | `PASS/KEEP PUBLIC`。 |
| `layout.py` | size/margin/preset/split/config/box/axes/facet/typography/surface DTO；facet/panel/text/split/resolve/recommend helpers | Workbench panel size、Pulse Editor preset与 renderer axes 真用 | `PASS/KEEP`；不是第二套 Workbench layout，Workbench只是消费结果。 |
| `live.py` | `LiveDataRevision`、ingress/session protocol、metrics/error、`LivePlotController`、refresh validation | production 0 caller；第二 cadence/thread | 当前产品 `DELETE`；独立库选择见 D-012/RP-03。 |
| `notebook.py` | front/selector serialization、widget class/JS、`NotebookView` lifecycle/pointer/environment/live bridge | production 0 caller；复用 RasterHost raster，但另有 comm/sender/UI transport | `USER DECISION + REDESIGN`；RP-02。不是整条 render duplicate，但 owner 旁路必须修。 |
| `parameters.py` | `RenderEffect`、`ParameterSpec`、`FrozenParameters`、`ParameterSchema`、type/bound helpers | Session/Qt/semantic parameter 唯一 schema | `PASS/KEEP`。 |
| `primitives.py` | point status/marker/image overlay/frame；pulse channel/block/scan/trace/repeat/DAC/timeline DTO 与 validators | Occupancy overlay、Pulse Editor、camera image payload 真用 | `KEEP PUBLIC`；overlay authority/lineage问题在 producer/Workbench，不应复制 DTO。 |
| `qt_controls.py` | lazy ParameterPanel/BoundControls factories、`edit_plot_display()`、lazy facade | FigureViewer display editor 真用；ParameterPanel direct root export只有 tests/standalone | `KEEP product editor + REDESIGN async`；RP-01。ParameterPanel implementation internalize，是否 direct public由 D-012。 |
| `raster.py` | axis helper、buffer/identity/interaction/front/operation DTO；dispatch/task/session adapter；`RasterPlotHost` | Workbench所有 plot surfaces 的唯一 worker/raster facade | `KEEP + REDESIGN`；capacity-one、front复制、multi-operation render、cleanup见 02。`live_controller()`删或内收。 |
| `rendering.py` | array/unit/label/limits/image/histogram/facet/fit annotation helpers、`RenderFrame`、`MatplotlibRenderer` | 所有 Matplotlib drawing 真核心 | `KEEP WITH DEBT`；约4,555行，RP-07。性能/overlay问题见02；private helpers不应 public。 |
| `selectors.py` | selector number formatting、range/point/kind/handle/state/snapshot/gesture DTO、drag/clamp/controller helpers | PlotSession与Workbench selection真核心 | `KEEP`；public只需 `NumericRange`/`SelectorKind`及跨包 DTO；controller internal。 |
| `semantics.py` | semantic field/description；axis/schema/fate/scope/choice；`updated_spec()`、`composed_spec()`、`describe_semantics()` | Workbench panel authoring与Qt controls的唯一 semantic composition | `PASS/KEEP`；`schema_summary()` standalone convenience可 internalize。 |
| `session_policy.py` | `ReplaceSpecInitialState`、viewport/parameter/label merge、`replace_spec_initial_state()` | replace-spec transaction 的纯 policy | `PASS/KEEP INTERNAL`；从 Session 拆出合理。 |
| `session.py` | revision/display/selection DTO与 subject/event/subscription；`PlotSession` | 核心 mutable plot owner | `KEEP + CONTINUE DECOMPOSITION ONLY ALONG EXISTING MIXINS`；02 已详审。不要建立平行 session facade。 |
| `specs.py` | reduction/relim enums、plot labels与六类 PlotSpec、semantic/default parameter schema/validation | Plot 的声明式 SSOT | `PASS/KEEP PUBLIC CORE`。kind-specific parameter builders保持 internal。 |
| `state.py` | `DisplayState`、`DisplayStateStore` | Session accepted display values/revision | `PASS/KEEP INTERNAL`。Workbench PanelState 是持久化 authoring state，职责不同；同步问题见02。 |
| `style.py` | rc/font/palette/line/point/artist/pulse/render/plot config、`build_plot_style()`、`style_context()` | renderer/Qt canonical visual style | `PASS/KEEP`；asset registration唯一；大但无第二 truth。 |
| `ticks.py` | `SmartOffsetLocator/Formatter`、`apply_smart_ticks()`、declared label helpers、`apply_declared_ticks()` | 数值轴和声明 categorical/facet axis 两种真实需求 | `PASS/KEEP`；两套算法处理不同输入，不是重复实现。 |
| `ui.py` | control kind/DTO、parameter/semantic control builders | Workbench TaskConsole form与 Qt panel共享 schema projection | `KEEP`；TaskConsole widget实现属于 zlc_ui，非第二规则引擎。 |
| `units.py` | `Unit`、mutable `UnitRegistry`、builtin catalog、`resolve_unit()`、`DEFAULT_UNITS` | projection/ticks/labels 真核心；仓内无 custom registration | `KEEP builtin conversion`；当前产品可 freeze/internalize registry，D-012 若承诺库扩展才保留 public mutation。 |

### 5.1 `_kinds` 与 `_rendering` 逐文件清册

| 文件 | 顶层 class/function | 责任/裁决 |
|---|---|---|
| `_kinds/__init__.py` | `handler_for()`、`default_spec()`、`fitting_spec()`、`panel_kinds()` | closed kind dispatch/default spec/fitting是真实；`panel_kinds()`与Workbench较窄catalog重复且0 caller，删该 public helper。 |
| `_kinds/base.py` | `KindHandler` | 六个 adapter 的统一静态协议；`KEEP INTERNAL`。它是 dispatch table row，不是第二 renderer。 |
| `_kinds/curve.py` | `render/build_payload/admits/validate/label_roles/default_spec` | curve adapter全组 `PASS/KEEP`。 |
| `_kinds/image.py` | 同上 | image adapter全组 `PASS/KEEP`。 |
| `_kinds/histogram.py` | 同上 | histogram adapter全组 `PASS/KEEP`。 |
| `_kinds/rolling.py` | 同上 | rolling adapter全组 `PASS/KEEP`。 |
| `_kinds/pulse_timeline.py` | 同上 | Pulse Editor 真用，`KEEP`；即使 TaskConsole过滤该 kind也不是 dead。 |
| `_kinds/facet_grid.py` | adapter组、data-axis/cell/default helpers、`cell_within_one_cell()` | facet composition真核心；`KEEP`。private cross-field mutations是 debt，不应再复制。 |
| `_rendering/__init__.py` | 无符号 | 空 package marker；若 pulse 合回 renderer则删；若完成对称拆分才保留。 |
| `_rendering/pulse.py` | artist sync、analog geometry、`update_pulse_timeline()` | Pulse timeline真实 renderer；`KEEP/MERGE` 取决于 RP-07 组织裁决。 |

### 5.2 plot facade：产品核心与 standalone 承诺不可混为一谈

当前 61-name root facade 可分三组：

1. **仓内 production 直接需要**：`PANEL_SIZE_NAMES`、`recommended_pulse_preset`、`AxisRef`、`DEFAULTS`、`PlotKind`、`fitting_spec`、`ImageFrame`、`ImagePointOverlay`、`PointStatus`、pulse primitives/timeline、`RasterPlotHost`、`NumericRange`、`SelectorKind`、`describe_semantics`、`updated_spec`、核心 PlotSpec/labels，以及 calibration report 使用的 `curve/image/facet_grid`。
2. **实现需要但无需 root promise**：Fit errors/model DTO/target、`ensure_qt5_application`、unit registry helpers、`schema_summary`、`Qt5ParameterPanel`、`LiveDataRevision` 等；它们可从 owner submodule internal 使用。
3. **只有 standalone/tests 支持**：`histogram()`、`rolling()`、`pulse_timeline()` constructors、`show()`、`LivePlotController`、`panel_kinds()`，以及部分 availability/convenience names。

推荐先由 D-012 裁决“zlc_plot 是当前应用内部包，还是正式独立 library”。在裁决前不要继续加 root names；裁决为内部包时将 facade 收到第一组，裁决为 library 时则给第二/三组各自建立真实跨-backend acceptance 与版本兼容承诺，而不是靠 `MAX_PUBLIC_NAMES <= 61`。

## 6. duplicate pipelines 与唯一 owner 对照

| 领域 | 当前真实产品 owner | 平行/重复实现 | 裁决 |
|---|---|---|---|
| live publication cadence | runtime `BoardScheduler` + per-panel interval | `LivePlotController` 自带 refresh thread | 删除后者，或在 standalone产品中只保留一个 cadence owner。 |
| latest/capacity-one | Signal plane latest semantics + RasterHost queued data frame | `_live_channel.LatestRevisionChannel` | 当前产品删除。明确每层丢的是 publication、render request 还是 front，不能都叫 latest。 |
| plot session mutation | `RasterPlotHost` worker | Notebook direct `self._session.*` | 所有 backend 通过 Host queue。 |
| Qt parameter form | zlc_plot schema/control projection；zlc_ui负责 TaskConsole widget | Qt5ParameterPanel 是另一 widget，但复用同一 schema | 两种 view 可以共存；规则必须只在 `parameters/specs/ui`，不要在 widget 重算。 |
| kind catalog | `_kinds` closed handler registry | Workbench `panel_catalog` product-filtered list | handler registry是能力SSOT，Workbench filter是产品策略；删无 caller的 `panel_kinds()`，保留显式过滤。 |
| renderer organization | `rendering.py` monolith | `_rendering/pulse.py` 单独一支 | RP-07 二选一，不新增第三层。 |
| fit workers | per-Session fit executor | global `_STRIPE_POOL` + live-prepare executor | 明确一个 compute pool owner和shutdown。 |
| owner wake | runtime `OwnerChannels` | Workbench 另有 wake/coalescing owner | 见02 PLOT-022；合并到一个线程安全owner，不在 plot再造。 |

## 7. `zlc_runtime` test/helper 逐文件清册

| test/helper | 实际锁定的契约 | 裁决 |
|---|---|---|
| `_snapshots.py` | runtime tests 的 OwnedSnapshot fixture helpers | `KEEP`，随留下的 plane/front/selection tests 收窄。 |
| `test_acceptance_fixtures.py` | 检查 usage notebook 文本和 headless demo 能退出，不是产品行为 acceptance | `DELETE/REPLACE`；若 D-011 删除 generic stream demo，一并删。真正 acceptance 应从 Workbench node 到 panel。 |
| `test_cleanup.py` | 只验证无产品 caller 的 `join_worker()` | 随 `cleanup.py` `DELETE`。 |
| `test_cross_repo_contract.py` | pinned digest/固定字符串 | `DELETE/MERGE`；历史 digest 不是行为契约，真实 cross-package tests 放 consumer repo。 |
| `test_generation_lifecycle.py` | plane generation reserve/begin/supersession | `KEEP`；迁移到精简 plane 后保持行为。 |
| `test_host.py` | worker terminal/Stop/generation、live slot、latest/follow/frozen processor | `KEEP + SPLIT`；删 start-and-wait/exact compatibility assertions，增加每个 declared output terminal completeness。 |
| `test_import_guards.py` | facade snapshot、MAX计数、模块import/dependency/thread side effect | 保留 import/dependency/no-thread；`REWRITE` facade snapshot，删 MAX 数字断言。 |
| `test_presentation.py` | wake、harmonic cadence、cohort all-or-nothing、same-shot、owed beat | `KEEP`；debug-only properties若删，改从可观察结果验证。 |
| `test_runtime_dataset_builder.py` | exact builder/schedule/edge/seal/monitor materializer | 按03c随 dead framework `DELETE`；coverage的小型验证移到 signal extent tests。 |
| `test_runtime_helpers.py` | failure/cleanup/name/preview/live port/mailbox混合 | `SPLIT`；只保留精简 mailbox 与 declaration name tests，其余随 dead modules删除。 |
| `test_runtime_streams.py` | 31个 exact/monitor/readiness/cursor tests + 少量 follow/EventRef | `PRUNE`；只迁移 EventRef 与真实 follow gap/terminal/order行为，其余随 generic stream删除。 |
| `test_selection_bridge.py` | ROI/facet/fit-derived outputs、lineage、stale fit、retained final | `KEEP + UPDATE`；committed event自包含后删二次 reader seam tests。 |
| `test_signal_front.py` | transitive front/fallback/weak parent retention | `KEEP`；直接覆盖真实产品语义。 |
| `test_signal_plane.py` | live/follow/freeze/retire/processor/fanout/contract | `KEEP + EXTEND`；03c 指出的 immutable generation declaration、stamp injectivity 尚缺。 |

## 8. `zlc_plot` test/helper 逐文件清册

| test/helper | 覆盖责任 | 裁决 |
|---|---|---|
| `conftest.py` | 全包 pytest fixtures/config | `KEEP`，随保留 tests 收窄。 |
| `data_factory.py` | schema/snapshot test factories | `KEEP`；不是 production surface。 |
| `test_aggregate_by_codes.py` | DataView categorical/group reduction | `KEEP`。 |
| `test_backends.py` | backend availability、Qt construction/interaction | `KEEP Qt`；Notebook/availability convenience 随 D-012；不能替代 RP-01 bound-controls test。 |
| `test_bimodal_collapse.py` | bimodal fit collapse/classifier | `KEEP`。 |
| `test_camera_cycle_image_pooling.py` | camera-cycle image projection/pooling | `KEEP`，与04b camera contract交叉。 |
| `test_compose_identity.py` | semantic compose identity | `KEEP`。 |
| `test_cross_repo_contract.py` | pinned cross-repo contract artifact | `DELETE/MERGE` 到 consumer behavior tests。 |
| `test_data_contract.py` | zlc_data adapter/schema helpers | `KEEP + PRUNE` `schema_dtype`。 |
| `test_embed_semantic_resilience.py` | embedded editor semantic rebuild/refusal | `KEEP`；新增异步 bound-controls 路径。 |
| `test_facet_auto_semantics.py` | facet default inference | `KEEP`。 |
| `test_facet_cell_ticks.py` | facet cell declared ticks | `KEEP`。 |
| `test_facet_cell_title_fit.py` | title sizing/layout | `KEEP`。 |
| `test_facet_dense_equivalence.py` | dense facet projection equivalence | `KEEP`。 |
| `test_facet_focus_compose.py` | focus + semantic composition | `KEEP`。 |
| `test_facet_focus_image_parity.py` | focused facet/image parity | `KEEP`。 |
| `test_facet_live_fit.py` | facet live fit transaction | `KEEP + REWRITE AFTER 02 DECISION`；不得把 data+fit atomic blocking 当永恒需求。 |
| `test_fit_contract_k.py` | model presentation/contract exactness | `KEEP`；只锁真正外部字段。 |
| `test_fit_engine.py` | FitEngine model solve/result/failure | `KEEP`。 |
| `test_fit_headline.py` | fit headline presentation | `KEEP`。 |
| `test_fit_jacobian.py` | analytic Jacobians | `KEEP`；高价值数值测试。 |
| `test_fit_numeric_table.py` | 无产品 caller 的 FitNumericTable API | 随 table surface `DELETE`；若某个保存/export消费者出现，再由该消费者定义格式。 |
| `test_fit_projection.py` | fit scope/selection/projection | `KEEP`。 |
| `test_fit_warm_start.py` | live warm-start identity/state | `KEEP`。 |
| `test_gesture_layer.py` | backend-neutral gesture routing | `KEEP`。 |
| `test_histogram_samples.py` | histogram sample semantics | `KEEP`；即使删 public constructor，kind仍是 TaskConsole能力。 |
| `test_image_fit_geometry.py` | regular/radial image fit geometry | `KEEP`；应加 executor lifecycle但不重复profile。 |
| `test_kind_registry.py` | closed handler registry/default specs | `KEEP`；删 `panel_kinds()` public-surface expectation。 |
| `test_label_carry.py` | label persistence on semantic replace | `KEEP`。 |
| `test_layout.py` | presets/facet/pulse/layout geometry | `KEEP`。 |
| `test_live_channel.py` | standalone LatestRevisionChannel | 当前产品随 `_live_channel.py` `DELETE`；library选择则保留并补整链。 |
| `test_live_controller.py` | fake-session standalone cadence/controller | 当前产品随 `live.py` `DELETE`；fake自洽不能证明WorkBench需要。 |
| `test_live_protocol.py` | PlotSession prepare/commit/abort transaction | `KEEP`；controller删除后仍为RasterHost真路径；atomic展示断言按02用户裁决改。 |
| `test_namespace_isolation.py` | lazy import不泄漏重依赖 | `KEEP intent`；不要用它保留无消费者 exports。 |
| `test_no_data_colour.py` | empty/invalid color semantics | `KEEP`。 |
| `test_notebook_raster.py` | NotebookView widget/front/pointer transport | `USER DECISION`；保留时必须新增单-owner并发断言，当前测试未抓到RP-02。 |
| `test_npz_io.py` | saved figure/archive round trip | `KEEP`；FigureViewer真产品链。 |
| `test_performance_guards.py` | isolated projection/render timing/copy budgets | `KEEP + EXTEND LATER`；02 已说明没覆盖Workbench+多panel+fit+overlay链，本轮不重复profile。 |
| `test_plot_session_golden.py` | session rendering golden/semantic state | `KEEP`。 |
| `test_projection_coverage.py` | validity/coverage across projections | `KEEP`。 |
| `test_public_api.py` | 宽 root 名单及可解析性 | `REWRITE` 为选定产品 surface；目前把 standalone/test-only 名字当需求。 |
| `test_public_surface.py` | public modules/exports/size guard | `REWRITE`；删 MAX 数字，保留显式 allowlist和lazy-import成本。 |
| `test_qt_widget.py` | Qt raster widget/parameter panel基础行为 | `KEEP + EXTEND`；加入 edit slot不阻塞、future成功/拒绝/supersede/close。 |
| `test_raster_host.py` | queue/dispatch/front/close/subscriptions | `KEEP`；删除无caller live-controller factory后更新。 |
| `test_replace_spec_transaction.py` | spec replace原子性/state retention | `KEEP`。 |
| `test_rolling_shot_axis.py` | rolling shot axis/history | `KEEP`。 |
| `test_selection_subject.py` | selection subscription ordering/lifetime | `KEEP`。 |
| `test_selectors.py` | range/box/crosshair/color/gesture state | `KEEP`。 |
| `test_semantic_feasibility.py` | schema→kind可行性 | `KEEP`。 |
| `test_semantic_spec_authority.py` | spec/semantic authority | `KEEP`。 |
| `test_semantic_ui.py` | semantic controls projection | `KEEP`。 |
| `test_semantics.py` | axis/scope/fate/composition | `KEEP`。 |
| `test_tick_labels.py` | smart numeric + declared categorical ticks | `KEEP`。 |
| `test_units.py` | builtin conversion/registry | `KEEP builtin`；custom registry tests由D-012决定。 |
| `test_validate_implies_build.py` | kind validate/build一致性 | `KEEP`；能阻止“菜单可选但构建失败”。 |

### 8.1 当前明显缺失的 tests

1. **Qt bound controls 非阻塞**：从真实 `RasterPlotHost` 返回未完成 Future，Qt event loop仍能处理事件；完成/拒绝/supersede/close分别更新UI。
2. **Notebook single owner**（仅在保留时）：所有 describe/replace/pointer 都进入Host worker，测试禁止直接 session mutation。
3. **线程 lifecycle**：创建/关闭多 PlotSession/RasterHost 后没有遗留 per-panel/global fit threads；不要求用脆弱的固定线程数，而验证owner service已shutdown。
4. **Facade consumer test**：以 Workbench/Atom真实 imports 为 allowlist，不以 `MAX_PUBLIC_NAMES` 数字为目标。
5. **完整产品链性能**：02 已登记；后续只需一条多panel+live fit+overlay+Qt acceptance，不再堆 isolated microbench。

## 9. 文档与当前代码/产品调用图的矛盾

旧文档在本审计中只作为“声称的设计”，不作为保留证据。

| 文档声称 | 当前证据 | 裁决 |
|---|---|---|
| runtime README 把 `streams.py`、`dataset.py` 列为组织中心，把 `live_dataset.py` 称为 ownership seam | 03c 从初始 monorepo history 到当前 HEAD 均找不到 exact builder/live port 的产品 caller；真实节点全用 plugin slot | 文档描述的是未落地架构。D-011 选择删除后重写 package map；不要为满足 README 保留代码。 |
| runtime README 称 Notebook 是 first-class acceptance fixture | `test_acceptance_fixtures.py` 只检查 notebook JSON含某段 flow、demo进程可退出；不经过真实 Workbench/node/panel | 不能称产品 acceptance；删除或降级为 example smoke。 |
| plot README 把 Notebook canvas 与 PyQt5 adapter 都写成正式能力，并给出 `LivePlotController` 作为 canonical live path | 当前 Atom/Workbench 0 caller；WorkBench走 runtime BoardScheduler + RasterHost | 这是 D-012 产品边界，不是 tests 可替用户做的决定。 |
| plot README `:143` 说 live data 必须等 fit 完成后成对原子发布 | 同 README `:250` 又说 data projection 立即提交、慢 fit 后发单独 overlay；当前代码实现前者 | 直接自相矛盾，已在02登记。先由用户裁决 data-first/atomic，随后只留一段。 |
| plot README 的 Qt 嵌入段说外部控件把修改“异步”提交 RasterHost | `qt_controls.py` 自带 bound controls 在 Qt slot 内同步 `.result(timeout=10)` | 文档与产品实现不符；RP-01 必修。 |
| plot README 强调 Notebook/Qt 共享 RasterPlotHost 协议、session在专用worker | Notebook 的 describe/replace 三条接口直接调用 `self._session` | 文档目标正确、实现违规；若保留 Notebook，按 RP-02 修。 |
| plot facade 注释称只放“常规使用路径” | root 仍导出无产品 caller的 live controller、show、panel_kinds、direct Qt panel、多个 standalone helpers | 以真实跨包 import重建 allowlist。 |

## 10. 需要用户裁决的事项

### U-06D-01 — `zlc_plot` 的产品边界（对应 D-012，最高优先）

二选一：

1. **推荐：当前 Zou Lab 应用内部 plotting package。** 保留 Workbench/Atom真实链、Qt widget、offline calibration report/FigureViewer；删除 NotebookView、LivePlotController、无caller constructors/extension surface和对应 tests/examples。
2. **正式独立 plotting library。** 保留 Notebook/live/standalone，但它们必须有 release-level owner/thread/cleanup acceptance、public compatibility policy，并修 RP-02；不能继续由 fake tests 与 README 宣称支持。

这项决定同时裁决 `show()`、Notebook assets/tests、`LiveDefaults`、`LivePlotController`、public mutable Fit/Unit registries、额外 API constructors 和 facade 宽度。

### U-06D-02 — runtime generic exact/builder/live framework（对应 D-011）

03c 与本轮 small-module consumer audit结论一致：推荐删除无人使用框架，只留 EventRef/minimal follow/coverage/Host/plane/presentation。若用户明确计划近期实现 lossless exact scan coordinator，需先给出一个真实 owner/caller 和端到端 acceptance；“未来也许需要”不足以维持约4,000行框架。

### U-06D-03 — renderer 文件组织

二选一：

1. **推荐默认：** `_rendering/pulse.py` 合回 `rendering.py`，删除空 package，接受单 renderer 大文件，先解决运行时问题；
2. 沿现有 `_kinds` 完成所有 kind 对称拆分，随后缩小 `rendering.py`，但必须删除 private cross-calls，不能同时保留两种组织规则。

这是维护性决策，不应阻塞 RP-01/RP-04 或 02/03 的 correctness 修复。

### U-06D-04 — public API 是否对仓外用户承诺兼容

若没有仓外消费者，推荐以当前 monorepo真实 imports 为最小 allowlist，不做 deprecation ceremony；若确有外部 notebook/scripts，用户需要提供样本或明确需兼容的名字。现有 tests 只能证明名字存在，不能证明有人使用。

## 11. 最终 PASS / REDESIGN / DELETE 清册

### PASS / KEEP

- runtime：`front.py`；精简后的 output declarations/DTO、Host、SignalDataPlane、presentation、selection bridge；EventRef/minimal follow/coverage。
- plot contracts：`kinds.py`、`parameters.py`、`specs.py`、`state.py`、`semantics.py`、`session_policy.py`、`primitives.py`、`units.py` builtin path。
- plot projection/render：`data_contract.py`（删一个dead helper）、`data_view.py`、`_fit_projection.py`、`_image_raster.py`、selector/gesture/session mixins、fit engine/models、layout/style/ticks、RasterHost/Qt widget。
- tests：所有直接验证上述产品行为、数值 Jacobian、semantic feasibility、layout/ticks、same-shot/front/plane、Host terminal、selection lineage 的 tests。

### REDESIGN / SIMPLIFY

- `qt_controls._Qt5PlotControls`：owner thread绝不阻塞等Future。
- NotebookView（仅保留时）：所有 session mutation/read走Host owner。
- `_fit_radial`：全局pool归入可关闭、有界的明确owner。
- `RunOwnerMailbox`/Host：保留Future generation lifecycle，删RunHandle支线。
- 两个 root facade：删MAX数字与test-only/standalone exports，按产品consumer重建。
- renderer文件组织：只保留一种规则。
- 02/03 已列的 data+fit发布、multi-front、RGBA、overlay、slot materialization、processor lane、generation truth 等实质问题继续保持原优先级。

### DELETE（当前产品边界）

- runtime：`_failure.py`、`_public.py`、`cleanup.py`、`live_dataset.py`、`preview.py`、`output_name.py`（函数合并后）；dataset builder/monitor/seal/exact；streams exact/readiness/cursor/monitor/general facade；dead output Protocol/helper。
- plot：`_live_channel.py`、`live.py`/Host live-controller factory；`FitNumericTable`和table properties；`schema_dtype()`；`panel_kinds()` root helper；`MAX_PUBLIC_NAMES`；availability convenience。
- tests：与上述 dead implementation一一对应的 self-tests、两包 pinned digest contract tests、只验证 MAX/public宽度的 assertions。
- Notebook/`show()`/standalone constructors不是无条件删除项：由 U-06D-01 决定。

## 12. 若获准修改时的最小执行顺序（本轮未执行）

1. 用户先裁决 U-06D-01/02/04；不先决定产品面就无法正确删 facade/tests。
2. 先修 RP-01 Qt owner blocking；它是当前真实产品可达的直接卡顿。
3. 精简 runtime dead framework与 mailbox handle支线，同时迁移保留的 follow/coverage tests。
4. 按 D-012 删除或正式修复 Notebook/live standalone；禁止只删 facade 留幽灵实现。
5. 收回 fit global pool并加 lifecycle test；随后再处理02的 data/fit性能架构。
6. 最后做 renderer文件整理和 facade清扫；这些不应抢占 correctness/owner修复。

## 13. 完成声明

- 已覆盖 runtime 16/16 source 文件、14/14 Python test/helper 文件。
- 已覆盖 plot 52/52 个主/子包 Python source（含 `_kinds`、`_rendering`、assets）与55/55 Python test/helper 文件。
- 没有修改 source、test、旧文档或硬件；本文件是唯一新增审计产物。
- 本报告不宣称旧 tests 全部正确；已明确区分产品行为证据、test-only seam 与需要用户裁决的库产品面。
