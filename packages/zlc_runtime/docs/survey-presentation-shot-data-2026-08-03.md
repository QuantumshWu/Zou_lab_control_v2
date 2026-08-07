我已完成三个子任务的取证。以下为最终审查报告。

---

# 任务3审查报告:呈现运行时提炼 / same-shot 组订阅 / zlc_data 依赖缺口

审查基线:迁移分支 `Zou_lab_control_v1_claude/Zou_lab_control_v1`(信号权威 = `zlc_neutral_atom/processing/signal_plane.py` 的 SignalDataPlane);旧主线参考 = `Github/ZLC_main`。所有路径省略共同前缀 `C:/Users/eadri/Dropbox/WorkCode/Github/`。

## A. 呈现运行时:现状机制与零 Qt 提炼

### A.1 现状机制盘点(逐条对应源码)

**三通道 owner wake**(已经是"回调注入"形态,只差把壳剥掉):
- 单一 `QtOwnerWake` 绑定 `_owner_cycle`,三个布尔 pending 位在 `_owner_event_lock` 下:`Zou_lab_control_v1_claude/.../zlc_workbench/task_console/window.py:224-231`。
- lifecycle 通道 `request_owner_wake`(window.py:3204-3211),被注入给 host 工厂(window.py:2909-2915)与 `RunOwnerMailbox`;data 通道 `_request_data_owner_wake`(window.py:3213-3220),经 `_activate_data_owner_wake` 用 token 借给 plane 的 `bind_owner_wake`(window.py:3222-3235;plane 侧 signal_plane.py:594-614);surface 通道 `_surface_future_done` 挂在每个 worker future 的 done callback 上(window.py:2964-2969)。
- `_owner_cycle` 一次锁下取走三 flag 再分派:surface→`_drain_surface_batches`,lifecycle→`_poll_logic_nodes`,(lifecycle|data)→`freeze()`+promote(window.py:3237-3270)。
- 关键细节:plane 的 `mark_changed` 只在**存在 reactive 下游**(processor/continuous)时才触发 data wake(signal_plane.py:1096-1111)——纯显示消费靠 timer 拉,不靠事件推。这是设计,提炼时必须保留。
- Qt 垫片本体:`zlc_frontend/qt_widgets/owner_wake.py:27-` 的 `QtOwnerWake`(queued connection 合并 + 回放位)。

**分拍(谐波 update_ms)**:
- 谐波集合单源 `UPDATE_INTERVALS = DEFAULTS.live.refresh_intervals_ms`(`zlc_workbench/task_console/console_records.py:170-177`),PanelConfig.update_ms 校验入集(console_records.py:257-266)。
- 单 timer 基频 = min(所有面板 update_ms),重设相位(window.py:3285-3295);timer 仅在有可见未暂停面板时运行(`_sync_display_timer` window.py:3297-3312)。
- `_tick`:`elapsed = _tick_count * base`;每 tick 一次 `freeze()` 成本拍唯一 front;按 `front.continuous_group` 分组,组 due 判据 = `elapsed % max(组内 update_ms) == 0`;组内所有面板 presented publication 均为当前 → 跳过(window.py:3399-3425;分组 window.py:3384-3397)。

**板级原子批次(同一 SignalFront 整组上屏)**:
- `_enqueue_surface_batch`:all-or-nothing——组内任一信号缺 value/publication → 整组不提交并置 "waiting for X"(window.py:2920-2934);prepare 中途出错 → 已提交成员走 `finish_unpresented_surface_update` 回收(window.py:2940-2953);成批入 `_surface_batches` deque,每个 future 完成触发 surface wake(window.py:2958-2962)。
- `_drain_surface_batches`:批内**全部** future done 才处理;任一成员缺卡/结果为 None/`can_accept_surface_update` 失败 → **整批丢弃**,绝不上半个板(window.py:2971-3020)。
- PanelCard 协议(`zlc_workbench/task_console/panel_card.py`):`PanelSurfaceUpdate`(panel_card.py:74-93,已零 Qt);`prepare_surface_update` 以 publication 身份 + generation + schema 指纹判定结构性替换,替换走 pending host,否则 `host.update_data(snapshot)`(panel_card.py:334-405);`can_accept` 五重身份门(serial/publication is/host_id,panel_card.py:407-419);`accept` 落 presented 记账 + present_front(panel_card.py:421-449);`_activate_pending_host` 是纯 Qt 的 widget 换装(panel_card.py:451-481)。

**双 executor**:`zlc_workbench/window_runtime.py:13-35`——2 线程 bulk compute + 1 线程 latency-sensitive,`submit_compute(fn, latency_sensitive=)`;另有原子导出 `stage_and_replace_export`/`cancel_export_commits`(window_runtime.py:38-81),整个文件已零 Qt。

### A.2 零 Qt 提炼设计(类/方法签名级)

以下全部可进 `zlc_runtime`(建议子模块 `zlc_runtime.presentation`),依赖仅 stdlib + 本包 signal 类型:

```python
class WakeSink(Protocol):                      # zlc_ui 的 QtOwnerWake 即一个实现
    def request_owner_wake(self) -> None: ...

class OwnerTurn(NamedTuple):
    lifecycle: bool; data: bool; surface: bool

class OwnerChannels:                           # 收编 window.py:224-231/3204-3235/3242-3248
    def __init__(self, sink: WakeSink) -> None: ...
    def notify_lifecycle(self) -> None: ...    # = request_owner_wake 的纯部分
    def notify_surface(self) -> None: ...      # future.add_done_callback 目标
    def activate_data(self, plane: SignalDataPlane) -> None:  # 含 bind_owner_wake token 借用/回滚
    def deactivate_data(self) -> None: ...
    def take(self) -> OwnerTurn: ...           # 锁下取走三 flag(closed 后恒 False)
    def close(self) -> None: ...

class HarmonicClock:                           # 收编 window.py:3285-3295 + 3406-3422 的纯算术
    def __init__(self, intervals: Sequence[int], default_ms: int) -> None: ...
    def rebase(self, panel_intervals: Iterable[int]) -> int   # 返回新基频(供宿主设 timer)
    def advance(self) -> int                                  # tick 计数,返回 elapsed_ms
    def group_due(self, elapsed_ms: int, member_intervals: Iterable[int]) -> bool

@dataclass(frozen=True)
class SurfaceUpdate:                           # = PanelSurfaceUpdate(panel_card.py:74-93)原样进包
    panel_id: str; serial: int; publication: SignalPublication
    value: SignalValue; future: Future; replacement: bool
    # 注意:去掉 host: RasterPlotHost 字段类型,改 host_token: object——
    # 身份校验(host_id/revision)下沉到 SurfacePort 实现侧,仲裁器不 import zlc_plot

class SurfacePort(Protocol):                   # PanelCard 的协议面(zlc_ui 实现)
    @property
    def panel_id(self) -> str: ...
    @property
    def signal_name(self) -> str: ...
    @property
    def display_interval_ms(self) -> int: ...
    def presented_publication(self) -> SignalPublication | None: ...
    def prepare(self, value: SignalValue,
                publication: SignalPublication) -> SurfaceUpdate | None: ...
    def observe(self, update: SurfaceUpdate, operation: object) -> None: ...
    def can_accept(self, update: SurfaceUpdate, operation: object) -> bool: ...
    def accept(self, update: SurfaceUpdate, operation: object) -> bool: ...
    def reject(self, update: SurfaceUpdate, error: BaseException | None) -> None: ...
    def finish_unpresented(self, update: SurfaceUpdate) -> None: ...
    def report_waiting(self, missing_signal: str) -> None: ...   # 状态文本由 Qt 侧渲染

class SurfaceBatchArbiter:                     # 收编 window.py:2920-3020,operation 全程 object
    def __init__(self, channels: OwnerChannels) -> None: ...
    def enqueue_group(self, ports: Sequence[SurfacePort], front: SignalFront) -> bool
    def drain(self, resolve: Callable[[str], SurfacePort | None]) -> None
    def cancel_all(self) -> None               # = window.py:3030-3033 关闭路径

class BoardScheduler:                          # 收编 _tick 的纯部分(window.py:3399-3425)
    def __init__(self, plane: SignalDataPlane, clock: HarmonicClock,
                 arbiter: SurfaceBatchArbiter,
                 ports: Callable[[], Sequence[SurfacePort]]) -> None: ...
    def on_tick(self) -> SignalFront           # freeze→promote 回调→分组→due→enqueue
    def on_owner_turn(self, poll_lifecycle: Callable[[], None]) -> None
```

`window_runtime.py` 的 `submit_compute`/`stage_and_replace_export`/`cancel_export_commits` 三个函数**直接原样进包**(零 Qt、零域耦合)。

**留 zlc_ui/presenter 的部分**(清单):
- `QtOwnerWake`(owner_wake.py:27)= `WakeSink` 的 Qt 实现;QTimer 本体与 `_sync_display_timer` 的可见性判据(window.py:3297-3312,读 Qt 状态)。
- `PanelCard` 全部 QWidget 面:`_activate_pending_host` 换 widget(panel_card.py:451-481)、`Qt5PlotWidget.present_front`、`RasterPlotHost` 构造(prepare 里的 `RasterPlotHost.from_plot`,panel_card.py:379-389——这段属于 presenter 对 `SurfacePort.prepare` 的实现,zlc_plot 依赖留在 UI 侧)、`isinstance(operation, RasterOperation)` 检查从仲裁器(window.py:2989)移入 port 实现。
- 状态条/summary/board 截图(window.py:3438-3474, 3349-3382)。

### A.3 旧主线考古:欠拍公平未保留 —— 标注为回收项(需改造后回收)

- 旧主线:单共享 RenderLoop,busy 时 `submit` 拒绝即跳帧背压(`ZLC_main/Zou_lab_control/frontend/render_loop.py:15-17, 109-120`);`_beat_owed` 在"该拍撞上 busy/拖拽"时记欠,下一个空闲 tick 无视 modulo 立即补拍,防止快拍重面板相位锁死慢拍面板(`ZLC_main/.../task_console.py:1904-1908, 9076-9106`)。
- 迁移分支:`zlc_workbench` 全树 **grep 无 `_beat_owed`/owed**;`_tick` 只有 `elapsed % max(update_ms) != 0 → continue`(window.py:3422),无补拍机制。
- **评估**:旧饥饿的确切形态(共享单 worker 拒收 → 相位锁)在新架构下不存在——每面板有自己的 worker host,prepare 从不因 busy 拒绝,重复提交由 publication 身份门挡住(panel_card.py:349-350)。但出现了**新的欠拍模式**:组在 due 拍因"成员 value 未到"(window.py:2927-2933)或 prepare 异常(window.py:2940-2953)整组流拍后,下一次机会是**下一个整倍数拍**,而非下一个 base tick;update_ms=2000ms 的组一次流拍即黑 2 秒。
- **回收建议(清单项)**:在 `BoardScheduler` 加组级 `owed` 位——due 拍 enqueue 返回 False 时记欠,之后每个 base tick 重试直至成功,成功即清欠。这是 `_beat_owed` 语义在新架构下的正确移植,不是照抄旧判据(旧判据依赖 `render_loop.busy`,已无对应物)。按血训(memory:render-thread 轮"欠拍公平 `_beat_owed`,删 rotor 丢的保证"),此项当年就是对抗审查抓回来的,拆包时不能再丢第二次。

## B. same-shot 组订阅设计

### 现状事实

- plane 明文拒绝全局 shot counter(signal_plane.py:15-21),`_note_display_drops` 同样声明"no global shot clock"(window.py:3427-3436)。
- **路①的核心其实已存在**:`set_front_signals`(signal_plane.py:616-632)声明连通集;`_build_front_locked` 对每个连通组件做 root-set join——每个 leaf publication 沿 `direct_parent_refs` 回溯到根 `_publication_roots`(signal_plane.py:2100-2117),组件 coherent 判据 = 所有 leaf 的根集合相同 + 逐名 ancestry 唯一(signal_plane.py:2232-2265);**不齐时整组件回退到上一拍完整组**(signal_plane.py:2271-2286)——这正是"组不齐等下一拍"的显示语义。
- 但它只覆盖**同一 producer→Processor 因果链内**的组;两个独立 producer 根集合必然不同,永远不 coherent,这是刻意不承诺(SignalPublication docstring signal_plane.py:206-217)。
- 路②的物理机制已备:`SignalAssociationRequest`(cause_id/cause_digest/trigger_schedule_fingerprint,`zlc_neutral_atom/runtime/signal_source.py:151-220`)、`SignalEventAssociationCursor` 的 arm→bind→next→finish 四段(signal_source.py:373-431)、`SignalEventAssociationSource`(signal_source.py:433-440)、plane 侧 `has_event_association`(signal_plane.py:1570-1575)与 event-derived 世代(signal_plane.py:1834-1934)。

### 裁决:两条都要,分层

- **路①(lineage 共同根 join)= 组订阅的默认引擎**,零新机制:把 `_build_front_locked` 已有的 root-set join 从"display front 内部实现"升格为公开的 Group subscription API。适用:同链派生组(source+ROI+fit 等)。跨 producer 声明 lineage join 应在 subscribe 时 fail-fast(拓扑上无公共可达根 → 立即报错),否则"永远不齐"是静默陷阱。
- **路②(EventRef/association 显式关联)= 跨 producer same-shot 的唯一诚实出路**,但**不建议给 plane 加新机制**。最小实现:association-capable producer 在 arm 到的组内,把 `(cause_digest, 组内序号)` 作为一个 sibling 输出随数据**同 publication 原子发布**(atomic sibling bundle 已保证同事务,signal_plane.py:206-243);组订阅按 shot_key 值相等做 join。物理同 shot 的证据由既有 association 机制担保,plane 核心零改动,join 器纯消费层。

### 接口草案(进 zlc_runtime)

```python
@dataclass(frozen=True)
class SignalGroupSpec:
    names: frozenset[str]                      # 组员(qualified signal key)
    join: Literal["lineage", "shot-key"] = "lineage"
    shot_key_signal: Mapping[str, str] | None = None   # shot-key 模式:成员→其 key 信号名
    hold_previous: bool = True                 # 不齐时保持上一完整组(front 语义前例 :2271-2286)

class GroupGap(NamedTuple):
    missing: frozenset[str]                    # 本拍尚无成员
    behind: Mapping[str, int]                  # 逐信号 MonitorCoverage.missed_events 投影
                                               # (runtime/dataset.py:522-545)

@dataclass(frozen=True)
class GroupFront:
    publications: Mapping[str, SignalPublication]
    complete: bool
    joined_key: frozenset[EventRef] | object | None    # lineage 根集 或 shot_key 值
    gap: GroupGap | None

class SignalDataPlane:                         # 或独立的 GroupJoiner 消费层类
    def subscribe_group(self, spec: SignalGroupSpec) -> GroupSubscription: ...

class GroupSubscription:
    def latest(self) -> GroupFront             # freeze-latest,永不阻塞;
                                               # 不齐时按 hold_previous 返回上一完整组或 incomplete
    def wait_complete(self, timeout: float | None = None) -> GroupFront
                                               # headless/notebook:阻塞到"新的完整组"或超时;
                                               # 超时返回 incomplete + gap,不抛异常(gap 是数据不是错误)
    def close(self) -> None: ...
```

**等待/超时/gap 语义定死三条**:等待=保持上一完整组可见(与 display 一致);超时=返回 incomplete+gap,绝不发明全局 shot counter 去"算差几拍";掉队度量保持逐信号(`MonitorCoverage.missed_events/current_gap`),组级只报"齐不齐+谁缺"。

**实现警告**:plane 的 parent payload 保持是 `WeakKeyDictionary`(signal_plane.py:580-584),`direct_parent_publications` 在 payload 被回收后直接 raise(signal_plane.py:1249-1254)——join 器等待凑组期间必须自己对候选 publication 持强引用,凑齐或被更新代取代后释放。

## C. zlc_data 依赖审计

### 树内 zlc_data 被 runtime 候选模块实际使用的符号(AST 提取,逐模块)

| 模块 | 符号 |
|---|---|
| `processing/signal_plane.py:33-41` | BlockId, DataBlock, DatasetRevisionRef, DatasetSchema, **OwnedSnapshot**, StreamGenerationId, ValueSchema |
| `runtime/streams.py:14` | DataBlock, StreamGenerationId, Value |
| `runtime/live_dataset.py:13` | DatasetRevision |
| `runtime/dataset.py:20-40` | AxisSpec, BlockId, CellValidity, ComponentValidity, DataBlock, DatasetComponentValidity, DatasetRevision, DatasetRevisionRef, DatasetSchema, Invalid, OwnedSnapshot, PointTable, **REPEAT(角色轴)**, StreamGenerationId, Valid, ValidityMode, Value, ValueSchema, `zlc_data.value.expand_component_validity` |
| `runtime/signal_source.py:16-45` | 上表之外再加:AxisId, CommittedTransform, DataTransformSpec, INVALID, VALID, Selection, `zlc_data.codec.value_schema_from_tree/to_tree`, `zlc_data.transform.apply_transform/commit_transform/resolve_transformed_schema`, `zlc_data.transform_codec.committed_transform_from_tree/to_tree` |
| `dataset_output.py:13`(signal_plane 的直接依赖) | OwnedSnapshot |
| (参照)`zlc_workbench/task_console/window.py:69-75` | BlockId, DataTransformSpec, DatasetRevisionRef, StreamGenerationId, materialize_value_dataset |

即:**OwnedSnapshot 用了,角色轴用了(REPEAT/AxisSpec/AxisId/SITE 族),repeat_axis 以 `REPEAT` 角色常量形态用了**,外加整个 value/validity/transform/selection/codec 层。

### 对照独立仓 zlc_data(`zlc_plot/src/zlc_data/__init__.py:1-31`,名字轴版)

独立仓只导出:Axis(名字轴,schema.py:48)、DatasetSchema(schema.py:418,**同名不同物**)、DatasetSnapshot、PointTable(schema.py:177,**同名不同物**)、PointTopology、LatestIngress/LatestRevisionChannel/RevisionedItem/IngressMetrics、units、npz io、errors。

**缺口清单(zlc_runtime 依赖独立仓 zlc-data 时全部缺失)**:
1. `value.py` 整层:BlockId, DataBlock, DatasetRevision, DatasetRevisionRef, OwnedSnapshot, StreamGenerationId, Value, ValuePayloadContract, expand_component_validity —— 全缺,这是 signal plane 的地基。
2. `axis.py` 角色轴层:AxisId, AxisRoleId, AxisSpec, REPEAT, SITE, SPATIAL_X/Y, SCAN_POINT, READOUT_EVENT, HISTOGRAM_BIN 等 —— 全缺(独立仓是名字轴 Axis)。
3. `schema.py`:ValueSchema 缺;DatasetSchema/PointTable **名字冲突、定义不同** —— 不是可加性回植,是替换。
4. `validity.py`:CellValidity, ComponentValidity, DatasetComponentValidity, Valid/Invalid/VALID/INVALID, ValidityMode, ValidityContract —— 全缺。
5. `transform.py`/`transform_codec.py`/`codec.py`:DataTransformSpec, CommittedTransform, apply/commit/resolve, 树编解码 —— 全缺。
6. `selection.py`:Selection 族 —— 全缺(zlc_plot 侧 SelectionData 链也要它)。
7. `snapshot_projection.py`:materialize_value_dataset 等 —— 全缺(workbench 已在用)。

**建议(可直接进 goal 清单)**:
- 独立仓 zlc-data **不可回植、只能换血**:以树内角色轴版(`Zou_lab_control_v1/zlc_data/`,零外部依赖、仅 numpy)为新基准整体替换独立仓内容;DatasetSchema/PointTable 同名冲突意味着 zlc_plot 独立仓必须同步迁移或版本钉死,不能两版共存(memory 已实证双向分叉+影子 import)。
- **额外依赖必须裁决**:runtime 候选模块普遍 import `zlc_storage`(canonical_text/exact_mapping/finite_real/nonnegative_integer/positive_integer/sha256_text:streams.py:15-21、dataset.py:15、signal_source.py:52-56、signal_plane.py:56、run.py:12、preview.py:9)。"零 Qt、依赖 numpy+zlc-data"的宣称要成立,这些小验证器要么随拆迁入 zlc-data(建议,它们是数据契约级),要么 zlc-runtime 显式加第三个依赖 zlc-storage——不能装作没有。

### 进包/留域裁决表(runtime 候选模块)

| 模块 | 裁决 | 依据 |
|---|---|---|
| `processing/signal_plane.py` | **进包**(把对 `zlc_neutral_atom.dataset_output` 的 import 改为包内) | 零 Qt;仅依赖 zlc_data+zlc_storage+同层 runtime |
| `runtime/streams.py`, `runtime/dataset.py`, `runtime/signal_source.py`, `runtime/live_dataset.py`, `runtime/preview.py`, `runtime/run.py`, `runtime/owner_mailbox.py`, `runtime/_failure.py`, cancellation/cleanup/ports/resources | **进包** | 全零 Qt、零域实体;owner_mailbox 即"唤醒回调注入"的现成范式(owner_mailbox.py:21-32) |
| `dataset_output.py` + `output_name.py` | **进包**(随 signal_plane 迁) | 仅依赖 zlc_data+runtime.dataset+zlc_storage(dataset_output.py:13-21) |
| `runtime/hosted_run.py` (LogicNodeHost) | **换血后进包或留域侧** | 耦合域实体 catalog.DefinitionKey/logic_node/artifact_output/input_spec(hosted_run.py:9-18);节点宿主骨架(mailbox+run 生命周期)可抽,descriptor 语义留域 |
| window.py 呈现段(A.2 提炼物) | **换血后进包**(OwnerChannels/HarmonicClock/SurfaceBatchArbiter/BoardScheduler) | 现代码与 QWidget 混编,须按 A.2 切协议面 |
| `window_runtime.py` 三函数 | **进包**(原样) | 已零 Qt |
| QtOwnerWake / QTimer / PanelCard / RasterPlotHost 装配 | **留 zlc_ui/presenter** | Qt 与 zlc_plot 依赖 |
| 欠拍公平 | **回收项**(A.3,改造后进 BoardScheduler) | 迁移分支已丢,旧形态判据不可照抄 |