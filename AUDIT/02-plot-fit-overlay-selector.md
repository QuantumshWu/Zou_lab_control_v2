# 02 — Plot、fit、overlay、selector深审

状态：本阶段完成。全部结论来自当前HEAD的静态链路和隔离只读探针；未修改代码、未触碰硬件。

## 1. 当前真实链

```text
OwnedSnapshot
  -> SignalPublication / coherent SignalFront
  -> Workbench PlotPanelPort
  -> Workbench _project_panel_input（可能临时拼ImagePointOverlay）
  -> RasterPlotHost capacity-one queue
  -> PlotSession prepare projection
  -> armed live fit solve
  -> PlotSession commit/render
  -> Matplotlib RGBA buffer
  -> RasterFront
  -> runtime board-coherent batch
  -> Qt5PlotWidget/QImage
```

交互反向链：

```text
Qt pointer/wheel
  -> PanelCard event filter
  -> Qt5PlotWidget
  -> RasterPlotHost pointer queue
  -> PlotSession gesture/selector/viewport/focus
  -> new RasterFront
  -> Workbench SelectionBridge
  -> PanelState / producer draft / derived outputs
```

## 2. 已确认问题

### PLOT-001 — 一次PanelState投影产生2–6张front

`ConsolePresenter._match_host_to_panel()`依次调用：

1. `host.configure()`；
2. `apply_panel_fit()`；无model也调用`clear_fit()`；
3. selection setter；
4. viewport setter；
5. classifier threshold setter；
6. facet focus setter。

真实无硬件探针确认，plain state令RasterFront sequence从1变3，即一次投影发布两张front。rich state最多六个独立host操作。新host还会经历initial front、mount match和metadata settle match，普通新panel理论最少约五张front。

`clear_fit()`即使当前没有fit也执行一次render，因此无fit panel仍付出第二张front成本。

### PLOT-002 — live fit刻意阻塞data front

当前流程是：

```text
prepare data@N -> solve fit@N to completion -> commit data@N + fit@N atomically
```

慢fit直接决定数据刷新率；在途solve不被新frame取消，只替换等待中的下一frame。多项测试主动锁住atomic pair行为。

根架构要求data-first、fit后到；Plot README同时存在atomic pair和data-first两种说法；Checkpoint又包含相反历史裁决。此项进入用户决策账本。

### PLOT-003 — 全帧RGBA复制构成固定带宽成本

每张front最终执行`canvas.buffer_rgba() -> bytes`。DPR=3时：

| Preset | Physical raster | 每front约 |
|---|---:|---:|
| `2x2` | 1470×1071 | 6.01 MiB |
| `4x4` | 2478×1827 | 17.27 MiB |
| `8x8` | 4494×3339 | 57.24 MiB |

单个`2x2 @ 10Hz`仅最终bytes约60 MiB/s；多个panel和一次配置额外2–6 fronts会产生显著突发。现有performance tests没有覆盖Workbench+fit+overlay+Qt+多panel人类路径。

### PLOT-004 — Overlay不是自描述signal

Occupancy没有发布typed overlay sibling。Workbench直接import concrete Occupancy plugin，从active node对象、calibration和当前outputs临时组装`ImagePointOverlay`。

这导致：

- Workbench承载plugin-specific逻辑；
- terminal后node对象消失，retained data不能恢复overlay；
- notebook、console、save存在不同组装边界；
- overlay revision是panel本地计数，不是producer publication revision。

Runtime lineage对正常live camera/occupancy companion已有same-shot保证，不能误报为“简单拼两个latest”。结构问题仍成立。

### PLOT-005 — ROI/binning后的Overlay坐标错误

Occupancy分类使用`TrapCalibration.rebased()`后的runtime SiteMap；overlay却固定读取原始`node.calibration.site_map`。

Camera image轴使用sensor coordinates：

```text
sensor = new_origin + binning * local_pixel
```

当前overlay直接发布原始local center。非零ROI origin或binning后，分类box与显示圈属于不同坐标系。现有测试只覆盖origin=0、binning=1，并固化了这个特殊情况。

### PLOT-006 — Selectors Off语义互相矛盾

当前代码/tests规定Off只阻止左右键创建selector，wheel zoom、middle pan和double-click focus仍属于plot；因此鼠标在plot上时外层Monitor不能滚动。

根架构要求Off时wheel交给board scroll，On后才进入plot。产品需要裁决，或采用第三种明确手势，例如默认wheel滚board、Ctrl+wheel缩放plot。

### PLOT-007 — Panel authored state与运行state分散

`PanelState`保存semantic/display/fit/selector/classifier/focus；`PanelBinding`另存viewport、overlay revision、多个host配置future；`PlotSession`再持有同一组resolved/accepted状态。Viewport没有进入`PanelState.document()`，而selector/focus会持久化。当前没有一个明确表格说明哪些是authored、哪些是resolved、哪些是transient。

### PLOT-008 — 两套parameter UI路径

Standalone/FigureViewer使用`zlc_plot.Qt5ParameterPanel`；TaskConsole使用`ParameterControl -> Workbench dict -> zlc_ui FormFieldProps -> Qt form`。两套UI映射消费同一plot vocabulary，存在漂移风险。是否保留取决于`zlc_plot`是否继续作为独立通用库。

## 3. 测试缺口

- 没有断言一次PanelState投影的front总数。
- 没有断言无关display修改不重新solve/clear fit。
- UI交互测试使用standalone offscreen card，没有TaskConsole祖先scroll。
- 性能测试不覆盖真实multi-panel、Qt、fit和overlay路径，且部分阈值宽到数秒。
- Overlay测试没有非零ROI/binning。
- 大量测试直接访问private renderer/session状态，守实现形状多于产品结果。

### PLOT-009 — SignalPlane接受错误snapshot stamp

Plane只校验output vocabulary和schema，不校验同一signal generation内：

- `block_id`/dataset generation是否稳定；
- snapshot revision是否严格递增；
- 相同ref是否仍是同一immutable snapshot。

探针连续发布：

```text
publication seq   1, 2, 3
snapshot gen      wrong-A, wrong-B, wrong-B
snapshot rev      5, 5, 4
values            1, 2, 3
```

Plane全部接受；PlotPanelPort只把第一帧交给host，后两帧因revision没有增加而静默跳过。`finish_live()`也会把同ref不同内容误判为未变化。

这不是单个plugin bug，而是“新Logic Node忘推进revision后preview冻结”的框架根因。Plane必须在首publication冻结dataset family，并在后续publication强制单调内容revision。

### PLOT-010 — Companion-only更新永远不会调度panel

BoardScheduler只比较primary publication是否已经presented；primary不变就跳过panel，不检查overlay等companion publication。

探针确认main EventRef不变、overlay EventRef从1到2时projector仍只调用一次，屏幕保持旧overlay。panel currency必须是：

```text
(primary EventRef, every companion EventRef)
```

而不是primary publication或data revision。

### PLOT-011 — Restart/generation replacement绕过same-shot cohort

PlotPanelPort遇到primary generation变化时直接替换host、标为presented并返回`None`。新host initial front由Workbench单独mount，不产生`SurfaceUpdate`，因此不经过SurfaceBatchArbiter。

多个同根panel在Restart/schema边界可分别换屏；现有same-shot测试只覆盖steady generation。新host的initial render也应作为普通cohort member，accept后才mount/present。

### PLOT-012 — Overlay投影有副作用且发生在去重之前

`_project_panel_input()`每次调用均重建overlay并执行`binding.overlay_revision += 1`。相同publication在慢render期间每个beat都先投影、推进counter，随后才被pending check丢弃。live、Edit、frozen与save还共享同一counter。

projection必须是纯函数；应先计算完整input stamp，再对一个exact companion EventRef memoize一次overlay。

### PLOT-013 — Overlay候选没有真正验证lineage

Runtime `build_front()`对已有causal edge/sibling的component能正确维持same-shot；仅把两个独立signal名字放进`front_signals`不会创造因果关系。

当前overlay picker只按contract列出所有Occupancy signal，几乎不使用所选image publication。另一个camera/run的Occupancy可与当前image各取latest后被Workbench组合。候选必须共享sibling、ancestor或lineage root；不兼容时应在UI过滤或运行时明确拒绝。

### PLOT-014 — Selector恢复后可见但不驱动输出

恢复/generation replacement时先用`emit_change=False`把selector画到host，之后才创建SelectionBridge。新bridge既不读取PanelState已有selector，也没有事件replay。

结果是图上有selector，但bridge selection仍为`None`，ROI/fit-derived outputs不会恢复，直到用户重新拖一次。静态fit也可能在bridge订阅前完成而永不补发。

### PLOT-015 — 结构轴selector无法roundtrip

`panel_selection_document()`没有保存`SelectionRange.domain`。repeat/point-row selector使用空axis加`domain="repeat"|"point_row"`；恢复时domain回到named，空axis随即ValueError。现有测试只守live translation，没有JSON roundtrip。

### PLOT-016 — 保存图和重开图状态不一致

Archive写入完整`PanelState.document()`，FigureViewer却维护第二套残缺parser，只恢复semantic/display/fit/overlay signal，丢失selector、classifier threshold、focused cell和published outputs；viewport根本未写入archive。

因此同一文件旁的PNG与NPZ重开后可能不是同一张图。PanelState必须有唯一decoder，Layout和Viewer均调用它。

### PLOT-017 — Classifier threshold有两份不等价truth

PlotSession保存所有facet threshold vector；PanelState只保存最后一个`{value, facet_index}`，而apply helper又忽略facet index，只传value。

用户修改A、B两格后，两个live host可能暂时记得A+B；host replacement、layout restore或archive reopen后A丢失，B还可能落到错误cell。`PanelPlotAnnotations`可以存vector，但production保存路径从未传入非空值，目前属于test-only abstraction。

### PLOT-018 — Plot committed event缺少完整selection meaning

Workbench为了补facet scope直接读取：

```text
host._worker_adapter._session()._projection._data/_payload
```

这使Workbench依赖zlc_plot私有实现，也产生第二次回读truth。正确的committed event应自带source generation/revision、axis/domain/frame、facet/repeat scope与canonical/display bounds。

### PLOT-019 — Atomic fit的实际性能和队列行为

隔离profile数据：

- `2x2 DPR1` front为490×357、699,720 bytes；data-only update约6.62ms。
- no-op configure仍发布1张front，约1.93ms；已clear的`clear_fit()`仍发布1张front，约3.11ms。
- 人为250ms fit：50ms后仍显示旧revision；queued rev2被rev3取消，但in-flight rev1必须跑完并先显示；总计约541ms，只发布rev1与rev3。
- 同一update扇出1/4/8个panel约5.86/19.58/40.20ms；RGBA复制0.70/2.80/5.60MB。DPR2再乘4。
- formal TaskConsole + real Qt：普通title commit在owner thread同步约41ms、live host产生2 fronts；Edit打开后同一commit约38ms并产生live 2 + editor 3 fronts。
- fit armed后，无关title edit让live重solve一次、editor重solve两次；live新增2 fronts、editor新增4 fronts。
- 两个同camera image panel，其中一个fit延迟300ms：fast host在50ms已算完，但两个真实widget都保持旧帧，直到约333ms才由cohort一起显示。

即slow fit不仅阻塞自身data，还通过same-root cohort产生head-of-line blocking。

### PLOT-020 — Atomic pair内部仍重复render

`commit_live_frame()`先提交`accepted_fit=None`的数据projection，再接受fit并执行第二次overlay render，最终只capture一次。`_present_projection_transaction(... accepted_fit=...)`的参数全仓没有非None调用，说明当前pair路径存在历史残余或漏用。

若用户选择保留atomic pair，至少应一次transaction装入data+fit并只render一次；但slow-fit延迟仍是产品语义，无法靠小优化消除。

### PLOT-021 — 每个panel的线程/内存成本

每个PlotSession拥有：

- 1个Raster worker；
- lazy prepare executor；
- lazy fit executor。

活跃data panel通常约2线程，fit panel约3线程；每开一个Edit host再增加一套，Save Fig再创建临时host。换帧期host latest、widget old、pending cohort和Agg canvas可同时持有多张全帧buffer。

`LivePlotController`和RasterPlotHost还各实现一遍prepare/solve/commit pipeline，是明确重复。

### PLOT-022 — Owner wake存在无锁丢唤醒窗口

Workbench `OwnerWake._pending`由plot worker写、GUI thread read-then-clear，没有锁。worker set与GUI clear交错时可能丢wake，最坏退回下一次100ms beat，符合偶发“卡一下”。现有测试只有单线程burst。

## 4. Identity裁决

| Identity/类型 | 当前裁决 |
|---|---|
| Runtime `EventRef` | `KEEP`：causal event与same-shot DAG |
| `DatasetRevisionRef` | `REDESIGN`：materialization identity必要，但字段需精简 |
| `DataBlock.block_id/revision` | `MOVE/DELETE`候选：与OwnedSnapshot ref重复 |
| `OwnedSnapshot` | `KEEP`：应成为ref + immutable payload唯一组合owner |
| `ValueSchema.fingerprint` | `DELETE`候选：无production消费者 |
| `DatasetSchema.fingerprint` | 待archive兼容裁决 |
| `SignalFront.signals` | `REDESIGN`：可由publication map派生，当前双truth |
| `SignalDescription.revision` | `DELETE`候选：production无读取 |
| fit source/batch revision | `KEEP`概念：source数据与fit request是不同事实 |
| Workbench overlay revision | `REDESIGN`：从exact companion stamp翻译，不按调用次数 |
| Raster host sequence | `KEEP`：frontend呈现顺序 |
| display/layout revision | `KEEP`：surface currency，不是dataset identity |

Plane不应强制Processor snapshot generation等于Processor EventRef generation：derived snapshot可能有意继承父data generation/revision。它必须强制的是每个output generation内部dataset family稳定、内容revision单调，并明确producer/derived stamp规则。

## 5. 逐类/函数裁决

| 实体 | 裁决 |
|---|---|
| `RasterBuffer/RasterIdentity/RasterInteractionMap/RasterFront` | `PASS`，边界清楚 |
| Qt `present_front` host/surface/sequence guard | `PASS` |
| `MatplotlibRenderer` persistent artist/cache | `KEEP WITH DEBT`，职责过宽但不是首要根因 |
| `PlotSession` | `REDESIGN` facade内部职责；保留公开能力 |
| `LiveSessionMixin` atomic-pair语义 | `REDESIGN`，待data-first裁决 |
| `FitSessionMixin` pair-only helpers | data-first下`DELETE/MERGE`候选 |
| `RasterPlotHost` | `REDESIGN` pipeline/idempotence；保留worker/frontend seam |
| `_WorkerSessionAdapter` | `MERGE/DELETE`候选，约220行单消费者Session镜像 |
| `LivePlotController` | `SIMPLIFY`，只留capacity-one/cadence，不再复制pipeline |
| `SurfaceBatchArbiter/BoardScheduler` | `KEEP`，真实data sibling的same-shot需要；修composite stamp与generation replacement |
| `PlotPanelPort` | `REDESIGN` composite identity与host replacement |
| `PanelState` | `KEEP` authored truth；补唯一decoder和interaction durability裁决 |
| `PanelFrozenData` | `KEEP`，exact multi-signal frozen moment有真实用途 |
| `_match_host_to_panel` | `REDESIGN`为一次plot-owned transaction |
| `_offer_state_to_editor` | `DELETE/REDESIGN`，当前造成重复完整配置/solver |
| `ConsolePresenter._site_overlay` | `DELETE` |
| `overlay_signal_groups` | `REDESIGN`，按lineage+contract筛选 |
| `OccupancyProcessor` classification core | `KEEP` |
| `OccupancyResult arrays + artifacts` | `SIMPLIFY`，当前双表达 |
| `frame_judged` | `USER DECISION`；若保留必须继承源snapshot axes/validity |
| `site_overlay()`当前形态 | `MOVE/DELETE`，不能在presentation依赖live node |
| `ImagePointOverlay` | `REDESIGN`坐标、scope、identity |
| `_repeats_of()` | `REDESIGN`，不能提前不可逆压掉repeat scope |
| `SelectionBridge` | `KEEP capability / REDESIGN implementation` |
| `SelectionDataReader` | `DELETE`候选，committed event应自包含 |
| `PlotSelectionSource` | `KEEP adapter`，去除plot private introspection |
| `panel_selection_document/from_document` | `KEEP`唯一codec，补domain和roundtrip |
| `viewer._panel_state` | `DELETE`，改用PanelState唯一decoder |
| `PanelPlotAnnotations` | `DELETE`或正式接入；当前非空只由测试产生 |
| UI `draws_image_surfaces` mirror | `DELETE duplicate` |
| `PanelCardView` | `KEEP WITH DEBT`，wheel/selector政策待裁决 |
| `ConsoleBoardView` | `PASS`，外层scroll owner存在 |

此外已确认的test-only/public seam候选：`PlotSession.commit_selector`、`adopt_native_device_pixel_ratio`、`live_fit_enabled`以及ConsolePresenter的`retarget_panel/resize_panel/set_panel_interval/rename_panel`只被测试使用或完全无人使用。

## 6. 推荐目标设计

1. Live Monitor采用data-first：data@N先promotion并清除/标记旧fit；只拟合最新revision；fit@N仍current时二次promotion。
2. Occupancy在自己的publication内生成typed、自描述overlay sibling，使用runtime SiteMap与完整x/y/repeat/point scope。
3. Panel只从同一coherent front读取image+overlay，且显式验证lineage compatibility。
4. PlotPanelPort以primary+companions composite EventRef stamp判断更新。
5. generation replacement也作为SurfaceUpdate进入cohort。
6. Plot committed event携带完整source/facet meaning，Workbench不读取私有Session。
7. PanelState通过一次plot-owned transaction投影；no-op配置、已clear fit均0 render/0 front。
8. `PanelState.document/from_document`成为layout/archive/viewer唯一codec。
9. Layout只保存被用户裁决为可复用的authored interaction；exact figure archive保存重现PNG所需的完整view state。

Frozen/manual Save Fig仍可等待fit完成；data-first只改变live monitor，不牺牲保存图的完整性。

## 7. 测试缺口

- Plane dataset family稳定、revision单调与same-ref content一致性；
- companion-only advance；
- generation replacement cohort；
- independent-lineage overlay refusal；
- ROI/binning与spatial axes；
- selector structural-domain roundtrip；
- selector恢复后derived outputs立即恢复；
- classifier多facet持久化；
- archive reopen与PNG view一致；
- 一次PanelState投影front总数；
- unrelated display change不重fit；
- formal Qt multi-panel fit/overlay人类路径performance；
- OwnerWake真实双线程无丢wake。

当前大量测试正在主动保护atomic pair、Selectors-Off仍zoom和private introspection；用户裁决后必须替换为产品行为守卫，不能继续把测试当设计来源。
