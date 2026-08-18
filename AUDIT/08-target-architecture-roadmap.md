# 08 — 推荐目标架构与分阶段修复路线

状态：**推荐方案A草案，不是已裁决架构**；全量scope已闭合，待用户八项顶层gate后重写/定稿。
原则：保留已证明的owner与数据结构，删除平行真相和无人使用框架；不以“重写”代替契约修复。

本文为了让依赖关系可读，按审计推荐选项串成一条完整方案；其中data-first、push live、typed overlay、RepeatRegion语义、SLM phase authority、Task锁定范围、durable allocation等仍在`DECISIONS.md`标为OPEN。用户选择不同分支时，本文对应章节必须整体改写，不能直接据此写old-red或修改production。冲突与证据等级以`09-independent-crosscheck.md`为当前索引。

## 1. 推荐保留的八层边界

现有八层名称基本合理，问题主要在边界被局部绕过：

| 层 | 唯一职责 | 不应拥有 |
|---|---|---|
| `zlc_data` | immutable scientific values/schema/validity/selection projection与codec grammar | runtime generation、Qt、设备、落盘路径 |
| `zlc_durable` | workspace path、atomic write、并发安全的name allocation | science schema、figure semantics |
| `zlc_runtime` | node execution、transactional live/final publication、causal lineage、front/cohort scheduling | plugin physics、plot rendering、Qt |
| `zlc_plot` | snapshot -> semantic projection -> fit/overlay/selector -> raster front | signal registry、Task lifecycle、plugin-specific overlay science |
| `zlc_ui` | pure Qt views与plain view models | `zlc_plot`/runtime/device/domain ownership、blocking futures |
| `zlc_pulse` | pulse model、compile、wire/transport、sequencer execution evidence | Measurement shot policy、camera grouping、Workbench state |
| `zlc_atom` | apparatus/device plugins、science nodes、calibration、SLM/atom physics | Workbench composition、panel save、global UI lifecycle |
| `zlc_workbench` | composition、presenter、workspace/session、device claims、panel persistence | occupancy/calibration/SLM science algorithms、第二plot/runtime实现 |

不要求每层独立发布wheel。代码层边界与distribution边界是两件事；建议先保留代码边界，再由D-063决定是否收敛成一个installable产品。

## 2. 一条数据/呈现主链

推荐唯一链：

```text
Producer / Processor
  -> context.publish_live({signal: immutable snapshot}, progress, lifecycle)
  -> SignalDataPlane atomic commit + EventRef lineage
  -> build_front(primary + causally compatible companions)
  -> BoardScheduler same-shot cohort
  -> PlotPanelPort composite input stamp
  -> PlotHost data-first projection
  -> RasterFront
  -> Qt present on owner thread

Async fit:
  data@N visible immediately without stale fit
  -> latest fit request for N
  -> if generation/revision/request still current
  -> fit overlay@N second promotion
```

### 2.1 Runtime commit不变量

1. 一个producer step提交一个immutable output bundle；所有siblings同一EventRef/revision。
2. generation内output vocabulary、schema、snapshot stream generation固定，revision严格递增。
3. 同snapshot ref不得代表不同内容；Plane必须拒绝重复/倒退/漂移。
4. fixed finite geometry的future cells必须`validity=False`；coverage只描述written extent，不替代validity或lifecycle。
5. stopped partial是正式retained partial：固定schema、unfilled invalid、可Save、可one-shot Processor消费。
6. terminal结果不受UI是否freeze/是否有follower影响。
7. `freeze()`只读取Runtime已提交状态，不调用plugin materializer，不在UI线程执行science work。
8. exact/latest是input contract，不从coverage类型猜。

### 2.2 删除平行live框架

保留一个Host/Plane commit seam。候选删除/合并：

- `DatasetBuilder/MonitorDataset`旧岛；
- `LiveDatasetPort/_ExactDeltaLivePort/ExactDatasetPreview`无人使用链；
- Camera/Calibration/Scan三种私有slot；
- `LivePlotController/_live_channel`与RasterHost重复pipeline；
- RunHandle/test-only state混入owner mailbox的部分。

若大camera array无法每次构造完整immutable snapshot，允许一个Host-owned materialization lane接收tokenized raw state；仍不得让Plane/UI主动pull plugin，也不得每插件自造slot。

## 3. Same-shot、overlay与selector

### 3.1 Same-shot

- EventRef lineage只证明软件publication因果。
- Camera物理trigger属于Pulse/Camera contract，必须另外验证window count、ordinal continuity和必要时hardware marker。
- `SignalFront`只组合共享root/ancestor的companion；contract相同但lineage无关不能叠加。
- generation replacement也必须进入cohort arbiter，不能绕过原子呈现。

### 3.2 Occupancy overlay

Occupancy发布typed、自描述、可保存/重放的overlay sibling；payload携带canonical sensor coordinates/status/site identity。Workbench不import occupancy plugin，也不从active node临时重建。

ROI/binning projection只有一个owner：classifier与overlay消费同一个rebased spatial transform。Terminal/frozen figure不依赖live node对象。

### 3.3 Selector/classifier

- authored selection identity使用AxisId/coordinate frame，不用可重名label。
- selector visual state与derived output commit分开，但restore后必须显式重新提交或标stale，不能只恢复图形。
- PanelState只保存用户裁决允许的interaction scope；Figure archive保存exact frozen interaction/evidence。
- classifier threshold作用域由D-023确定，不能vector session/singular state两种truth。
- Selectors Off的wheel归属由D-003确定并用真实board嵌套QTest守住。

## 4. Plot与UI性能目标

### 4.1 Plot host

- `configure`, `clear_fit`, selector/viewport/focus setters必须idempotent；no-op=0 render/0 front。
- 一个PanelState transaction最多一次基础promotion；只有真正异步fit result可第二次promotion。
- panel title/layout等非plot字段不得re-arm fit。
- Edit/live host共享immutable authored PanelState，但各自有front lifecycle；不得在metadata settle时重复完整configure。
- capacity-one queue取消/忽略stale in-flight fit结果；不能让250ms旧fit先显示再处理latest。
- fit pool/executors有session close/shutdown owner；不得每panel无限叠线程。

### 4.2 UI

- Qt slot不调用`Future.result(timeout=10)`；所有worker完成通过owner wake/signal回到UI。
- `OwnerWake`线程安全且无lost wake；100ms beat只能作为刷新cadence，不是并发正确性fallback。
- `zlc_ui`只持views/view-models，Presenter状态parser/codec唯一在Workbench owner；FigureViewer不得维护第二套不完整PanelState decoder。
- ConsolePresenter按lifecycle拆成panel projection、logic execution、artifact/layout、device control等协作者，但保留一个composition owner，避免新framework。

## 5. Measurement/Task/preview contract

### 5.1 Descriptor默认契约

Output vocabulary只声明一次：Runtime-neutral `DatasetOutputDeclaration(name, contract)`作为typed owner，Atom descriptor直接引用；保留“static output spec”概念，不保留第二个同形`OutputSpec` DTO与互转。

Measurement/Task若声明可视output，Host必须验证：

- Measurement在first bounded cadence内publish live；
- Task publish progress与至少一个monitor preview，或明确声明无preview理由；
- resolved preview一定属于resolved output vocabulary；
- terminal success所需artifact存在且可读；
- progress在terminal清除/终结，UI不继续显示“Scanning”。

新增Logic Node的通用test必须走真实Host -> Plane -> Workbench preview，而不是只调用`execute()`。

### 5.2 Preview lifecycle

Preview的`transient/retained/final`是独立lifecycle字段，不用DatasetCoverage冒充。Panel只有在首个真实publication后才显示live；声明但未publish不能标live。

SLM phase应用后立即publish candidate phase；camera running mean按有界时间cadencepublish；uniformity curve只在candidate完成时增加一点。

Task active默认只禁止硬件冲突与当前Task draft mutation；Monitor观察/布局是否可编辑由D-010最终裁决。

## 6. Pulse/Camera execution contract

### 6.1 Vocabulary

```text
timeline RepeatRegion = pulse document内部循环
cycle/shot             = 一次完整trigger-role序列
sweep                  = scan table完整遍历
repeat                 = dataset统计重复轴
```

Measurement只能用一个finite execution入口表达`N cycles`；compiler与device不从RepeatRegion猜shots。

### 6.2 Start preflight

在任何hardware side effect前统一验证：

- target ABI、clock/time step、program/scan geometry；
- integer/bounds-safe sweeps/counts；
- compiled camera window roles/count/cadence；
- actual CameraWorkingPoint exposure不重叠；
- delay event FIFO worst-case capacity；
- dynamic tuned device claims；
- source与sequencer的apparatus association。

任何off-grid authored duration/scan值要么拒绝，要么同时保存authored与canonical actual；Dataset坐标使用actual。

### 6.3 DONE与safe

推荐finite completion包含所有delayed output tail。`wait_done()`返回后safe不得截断作者波形。若用户选择engine-only DONE，则公开tail state并禁止调用方误解。

Virtual Sequencer必须按逻辑cadence逐cycle推进、可Stop，并模拟camera busy；不能同步爆发全部trigger后只sleep DONE。

## 7. SimulationWorld边界

- 保留一个installation-owned world state owner；大类可按pure physics helper拆，但不复制state或hidden truth。
- geometry、apparatus physics与seed由一个`SimulationWorldConfig`组成，不寄生某个virtual camera config。
- device callback在world lock外执行；registration必须有unregister/close lifecycle，不能永久持有已关闭camera。
- mutable test knobs改为显式scenario override；hidden trap/aberration oracle只通过test diagnostics surface，普通device/node不能读取。
- Virtual与real adapter共享相同公开contract；Simulation只增加可验证diagnostics，不让production算法旁路physics。

## 8. SLM目标闭环

参见`05-slm-summary.md`。唯一推荐上下文：

```text
TargetArtifact(intensity, objective)
ScienceContext(pattern base, numeric pupil, operator wavefront)
DeviceMapping(profile, wavelength, orientation, correction revision)
CommandReceipt(outcome, gray/readback/transport evidence)
```

Feedback只更新Pattern/base，保留frozen wavefront；sparse WGS走完便宜数值gate；100 shots只coarse；controller使用uncertainty、step clip、trust/rollback；final独立adaptive CI validation。real device unknown state不允许伪装zero。

## 9. Data与durability

- `OwnedSnapshot`是数据materialization owner；DataBlock不再维护重复ref truth。
- `restrict_snapshot`必须同步投影coordinates、labels、frame、unit、validity。
- validity入口严格bool，不做truthiness conversion。
- selection按AxisId/typed coordinate解析，支持text并拒绝重名歧义。
- figure archive先规划唯一member namespace、strict metadata，再stream写durable temp；reader验证format/version/shape/duplicates/nonfinite。
- durable unique allocation与commit原子化，或明确single-owner；产品Task更适合前者。
- unknown metadata不自动`str()`。

## 10. 测试目标

测试金字塔应从“实现形状”移向“产品不变量”：

1. pure numeric/property tests；
2. package contract tests（行为而非Markdown allow-list）；
3. real Host/Plane/Plot integration；
4. offscreen Qt + actual board/container interaction；
5. Virtual full-chain deterministic evidence；
6. experiment-machine acceptance receipts。

每个P0至少要有一个old-red行为探针。删除source-token、private-state、duplicate-doc SHA和旧installed-package false-green。Mock tests明确标software-byte evidence，不冒充SDK/physical evidence。

## 11. 分阶段修复顺序

### Phase 0 — 用户裁决与冻结

- 裁决data-first、selector wheel、Task lock、deployment、Feedback observable/context/Stop。
- 暂停新增Logic Node与新public framework，直到live contract固定。

### Phase 1 — 数据损坏/硬件错误blockers

- figure archive namespace/reader；
- concurrent unique path；
- snapshot labels/validity/selection；
- dynamic form reconcile；唯一PanelState decoder；
- window close前hardware safe/claim释放、Qt atexit不绕guard、所有executor有owner shutdown；
- GUI线程不直接hardware tune或同步等待worker Future；
- duplicate `DeviceSpec.key`导致的设备泄漏、wheel缺scan templates；
- Calibration artifact strict version/JSON/unit与sample manifest；
- FPGA 50MHz主clock约束、destructive build containment、program/flash fail-closed target选择；
- hardware SAFE pin gate、board/pin ABI、delay FIFO capacity、clock/ABI/count preflight、physical DONE tail；
- Pulse remote owner-token/SAFE takeover/network admission，移除全COM主动probe与伪interprocess lease；
- camera cadence/window count；
- SLM unknown command/correction lease。

### Phase 2 — Runtime live transaction

- fixed schema + invalid future + partial terminal；
- push commit、pure freeze、processor execution policy；
- stamp invariants、companion updates、generation cohort；
-统一 preview lifecycle。

### Phase 3 — Plot/overlay/UI

- data-first fit；
- idempotent PanelState transaction；
- typed overlay + ROI coordinates；
- selector restore/persistence；
- remove Qt blocking future与duplicate viewer parser。

### Phase 4 — Pulse/Measurement science chains

- finite cycle API；
- Calibration/Temperature protocol；
- Calibration sample改为有界/streamed保存，Temperature artifact保留完整survival数据；
- Temperature每次execute重置run-local state，Scan明确restore/leave-at-last；
- scan claims/actual coordinates/provenance；
- bounded camera buffer与incremental processor。

### Phase 5 — SLM feedback/device

- science/device context artifacts；
- numerical gate、uncertainty controller、CI validation；
- Editor external state；
- X15213 true profile/ABI/orientation/settle acceptance；
- dense MRAF另行处理。

### Phase 6 — Delete/move/document

- 删除runtime/data/plot零consumer clusters与重复pipelines；
- 收敛package/distribution metadata；
- 历史docs迁移；
- 重写当前architecture/status/contracts；
- 最终全树测试与实验机验收。

## 12. 修复纪律

- 每个phase先写不变量和old-red，再改唯一owner。
- 一次不引入新framework；优先删平行路径。
- 不靠提高timeout、增加sleep、隐藏invalid、降低质量或增加GPU掩盖contract错误。
- 用户未裁决的产品语义保持OPEN，不由实现者选一个再把测试写死。
- 文档只在行为与证据成立后更新；不以“Checkpoint写完成”替代验收。
