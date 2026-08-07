# signal_plane.py 拆包审查报告(zlc_runtime)

审查对象:`zlc_neutral_atom/processing/signal_plane.py`(2,418 行,单文件,零 Qt,零 numpy 直接 import,零物理域概念)。以下行号均指该文件,除非另注。

---

## 1. 内部结构地图

| 行号 | 块 | 职责 |
|---|---|---|
| 1-21 | module docstring | 拉模型宣言:freeze-latest 非总线;MonitorCoverage 取代全局 shot clock |
| 23-56 | imports | 见 §4 |
| 69-83 | `SignalProducer` Protocol | producer 路由契约:`instance_id` + `dataset_output_declarations` + `signal_key` |
| 85-107 | `LatestProcessorControl` Protocol | processor lane 的 owner 回调面(validate/evaluate/accept×3/wake) |
| 110-121 | `DerivedSignalOutput` | 派生值载体(OwnedSnapshot + preserve_source_coverage) |
| 124-203 | `SignalValue` | 单信号单不可变修订;read-through 属性(block/schema/values/dtype/unit/axes);`behind`(192-203)读 `MonitorCoverage.missed_events` |
| 206-246 | `SignalPublication` | **原子兄弟捆绑** = 因果单元;`event_ref: EventRef`;`_issuer` 哨兵防跨 plane 注入;`direct_parent_refs: tuple[EventRef, ...]` |
| 249-323 | `SignalFront` | 不可变前沿:signals/failures/publication_by_signal/连续组;`continuous_group()`(314-323) |
| 326-368 | 模块级校验 helper | `_declared_outputs` / `_require_published_declaration` / `_require_signal_producer` / `_node_instance_id` |
| 371-396 | `_GenerationState` | 唯一可变状态:kind ∈ {producer, processor, continuous, event};slot/source_*/publication/last_parent_sequence/published_schemas/terminal/retired |
| 398-571 | `_ProcessorEntry` + `_LatestOnlyProcessorLane` | 单 worker ThreadPoolExecutor(413);attach/route/drain/cancel;latest-only 跳版语义(452-475);done 回调经 `request_processor_owner_wake` 注入唤醒(559-561) |
| 574-2418 | `SignalDataPlane` | 见下细分 |

`SignalDataPlane` 内部细分:

- **577-614 owner wake**:`bind_owner_wake`/`unbind_owner_wake`,token 借用制,回调注入,零 Qt。
- **616-632** `set_front_signals`:连接的连续信号集(front 一致性范围)。
- **634-801 generation 注册 helper**:单 owner 归属检查(638-651)、`_install_state_locked`(705-779)、`_node_route_names`(781-801)。
- **803-848 producer 预约**:`reserve`(803-837,幂等重入)、`require_active_generation`(839-848)。
- **850-1037 依赖闭包 / preemption 机器**:`bind_generation_source`(850-909,含环检测 890-905)、`release_generation_source`(911-928)、`withdraw_dependency_closure`(930-1012)、`finish_dependency_retirement`(1014-1037)。
- **1039-1111 live 生命周期**:`attach`(1039-1078)、`mark_changed`(1080-1111,dirty + 有下游 processor/continuous 才 wake)。
- **1113-1231 查询与 processor 接入**:`latest_publication`、`direct_parent_publications`(1119-1126)、`publication_owner`(1128-1148)、`attach_latest_only_processor`(1150-1198)、`cancel_latest_only_processor`(1200-1203)、`bind_processor_event_source`(1205-1228)。
- **1233-1348 发布校验**:issuer 校验(1233-1240)、私有 parent 载荷解析(1242-1260)、route parent 校验(1262-1309)、**generation 内 schema/兄弟集稳定性**(1311-1348)。
- **1350-1513 发布核心**:`_publish_locked`(1350-1396,铸 EventRef、登记 parents 到 WeakKeyDictionary L1388、sequence 单调)、`publish_final`(1398-1448)、`publish_processor`(1450-1513,双检锁 + `last_parent_sequence` 拒过期父)。
- **1515-1650 event 能力 + continuous derived 绑定**:`signal_event_binding`(1530-1568)、`has_event_association`(1570-1575)、`bind_continuous_derived`(1577-1650,同绑定幂等)。
- **1652-1732 derived 值物化**:`_route_owned_snapshot`(1652-1684,把 DataBlock 头重绑到 plane 拥有的路由身份 `signal/{owner}/{output}`,防止前端成为平行 generation 权威)、`_derived_values`(1686-1732)。
- **1734-1934 derived 发布**:`publish_continuous_derived`(1734-1781)、`fail_continuous_derived`(1783-1810)、`continuous_needs_publication`(1812-1832)、`bind_event_derived`(1834-1880)、`publish_event_derived`(1882-1934,terminal)。
- **1936-2056 回收**:`withdraw_derived`、`_retirement_closure_locked`(1941-1972,传递闭包)、`_withdraw_owner`(1974-1986)、`_cleanup_retired_states`(1988-2012)、`retire`(2014-2020)、`detach_live`(2022-2056,FINAL 保留 vs 闭包撤回)。
- **2058-2098** `_freeze_one`:调 slot 的 `freeze_live_outputs()` 并按声明词表冻结。
- **2100-2155 lineage 行走**:`_publication_roots`(2100-2117)、`_collect_publication_ancestry`(2119-2140)、`_name_is_ancestor`(2142-2155)。
- **2157-2312** `_build_front_locked`:邻接图→连通分量→leaf 选取→**根集单一性检验(2238-2243)+ 逐名祖先合并(2244-2261)**→不一致则回退上一 front(2271-2286)→连续组(2298-2306)。
- **2314-2383** `freeze()`:drain lane → 冻结 dirty slot → 发布 → route 给 lane → 建 front。**这是拉模型的心脏。**
- **2385-2418** `close` / `__len__`。

---

## 2. 四分方案(3-5 模块)

建议切成 **4 个模块 + 1 个门面**,全部可独立测:

### M1 `zlc_runtime/values.py` — 值模型与契约(L69-368)
- 内容:`SignalProducer`、`LatestProcessorControl`、`DerivedSignalOutput`、`SignalValue`、`SignalPublication`、`SignalFront`、四个模块级 helper。
- 公共接口:即这些类本身(已是今日 `processing/__init__.py` 的导出面)。
- 依赖:仅 zlc-data(OwnedSnapshot/DatasetSchema)+ 契约类型(EventRef、Coverage、OutputDeclaration,见 §4)。不依赖任何其他模块。
- 测试:纯构造/校验测试,零线程。

### M2 `zlc_runtime/processor_lane.py` — latest-only 执行缝(L398-571)
- 内容:`_ProcessorEntry` + `_LatestOnlyProcessorLane`。
- 公共接口:`attach_processor / route / drain_processors / cancel_processor / active_processor_bindings / close`。
- 依赖:M1(`SignalPublication`、`LatestProcessorControl`)。
- 测试:用 fake node 独测跳版/取消/失败/wake 语义(现有 `tests/test_console_data_plane.py:492-637` 已按这个粒度测)。

### M3 `zlc_runtime/registry.py` — generation 注册与回收(L371-396 + 634-801 + 803-848 + 1039-1111 + 1936-2056)
- 内容:`_GenerationState`、install/reserve/attach/mark_changed、retirement 闭包、retire/detach_live、cleanup。
- 公共接口:对 plane 内部的 locked API;对外经门面转发。
- 依赖:M1;持 slot 协议(鸭子型 `freeze_live_outputs`/`close`)。
- 测试:生命周期状态机测试,无需线程。

### M4 `zlc_runtime/front.py` — lineage 与 front 构建(L2100-2312)
- 内容:三个 lineage 行走函数 + `_build_front_locked` 改写为**纯函数**:`build_front(states_view, front_signals, previous_front, resolve_parents) -> SignalFront`。它只读 state 的 7 个字段与 parent 解析回调,天然可抽纯。
- 依赖:M1。
- 测试:给定合成 states 图直接断言一致性/回退/连续组——这是全文件最值得独立测的算法块。

### M5 `zlc_runtime/plane.py` — `SignalDataPlane` 门面(其余部分)
- 内容:锁、wake 绑定、发布核心(1233-1513)、derived 绑定/发布(1515-1934)、`freeze()`(2314-2383)、`close`。组合 M2/M3/M4。
- 依赖:M1-M4。
- 说明:发布核心与 registry 共享锁与 `_states`,强行再切会制造跨模块锁协议;保留在门面里是诚实的边界。

依赖图:`plane → {values, lane, registry, front}`;`lane/registry/front → values`;无环。

---

## 3. 逐块裁决

### 照搬进包(资产)
| 块 | 行号 | 理由 |
|---|---|---|
| freeze-latest 拉模型(dirty 集 + 条件 wake + `freeze()`) | 1080-1111, 2314-2383 | 设计核心,GUI 侧 17 处调用(`zlc_workbench/task_console/window.py:249` 等) |
| `SignalPublication` 原子捆绑 + `_issuer` 哨兵 | 206-246, 1233-1240 | 因果单元;哨兵防跨 plane 注入,便宜且有效 |
| `SignalFront` + 分量一致性 + 回退上一 front + 连续组 | 249-323, 2157-2312 | "组件内一致、跨 producer 不承诺"的机械实现;`continuous_group` 已被 `window.py:3390` 消费 |
| MonitorCoverage 逐信号掉队(`SignalValue.behind`) | 192-203 | shot clock 的替代品,契约级 |
| `_route_owned_snapshot` 路由身份单源 | 1652-1684 | 防前端变换成为平行 generation 权威 |
| generation 内 schema/兄弟集稳定性校验 | 1311-1348 | 真不变量,非仪式 |
| processor lane 的 latest-only + 双检锁发布 + `last_parent_sequence` 单调 | 409-571, 1450-1513 | 锁窗 TOCTOU 处理正确,照搬 |

### 换血后进包(简化)
**preemption 自动依赖闭包回收 —— 核实结果:**
- 真实行数(plane 侧,共约 195 行):`bind_generation_source` 850-909(60 行)、`release_generation_source` 911-928(18)、`withdraw_dependency_closure` 930-1012(83)、`finish_dependency_retirement` 1014-1037(24)、`require_active_generation` 839-848(10)。
- 调用方:**全仓唯一调用方是 `Zou_lab_control/api/_application_services.py`**(release:277、require:327、withdraw:398、finish:489),对应约 110 行应用侧机器(`_retirement_for_blockers_locked` 368-421 + start 流 476-497)。
- **关键实证:`bind_generation_source` 全仓零调用、零测试**(grep 全树只命中定义处 850)。这意味着 producer 的 `source_owner_id` 恒为 None,`withdraw_dependency_closure` 的"闭包"退化为逐根的普通派生闭包——与 `_retirement_closure_locked`(1941-1972)语义重合。
- **裁决:采纳此前评审建议。** 删 `bind_generation_source`(死代码);删 `withdraw_dependency_closure`/`finish_dependency_retirement`/`release_generation_source`/`require_active_generation` 四个 preempt API,应用层改显式 stop-then-start(cancel blockers → wait_until_released → `retire()`,现有 `retire`/`detach_live` 已足够)。**保留** `_retirement_closure_locked`——它是 `withdraw_derived`/`retire`/`detach_live` 的正常生命周期路径,不是 preemption 专属。

**`publication_owner`(1128-1148)**:门面为组合层解析 node 引用,泄漏 `object` 出包边界;进包时收窄为返回 opaque handle 或留域侧适配。

### 防御仪式(删/降级)
- **isinstance 密度**:48 处。公共边界的保留;但内部 locked helper 的重复校验是仪式,典型:`_state_for_generation_ref_locked` 677-685 每次内部调用重验 tuple 形状;`publish_continuous_derived` 在两次持锁各跑一遍 `_require_route_parents_locked`(1754、1771)——锁窗重查 state 身份是必要的,**重跑类型 isinstance 是仪式**,第二遍只查 sequence/retired 即可。
- **`SignalFront.__post_init__` 全量自检(264-303)**:生产中唯一构造方是 `_build_front_locked`,即每个 tick 都在校验自己刚构出的东西(集合相等 + 逐名 publication 归属 + 组对称性,O(n) ×2)。降级为 debug 断言或仅在测试构造路径启用。
- **digest 类**:**此文件内不存在 digest 类**——最近的只有 L1672 读取 `schema.fingerprint`(来自 zlc-data,只读)与 27 处 `canonical_text`(来自 zlc_storage 的字符串校验器)。此前评审的"digest 类"指控对本文件不成立;真正的问题是为一个 10 行的字符串校验器背上整个 zlc_storage 依赖(见 §4)。

---

## 4. import 面裁决(L33-56,逐条)

| import | 内容 | 裁决 |
|---|---|---|
| `zlc_data`(33-41) | BlockId, DataBlock, DatasetRevisionRef, DatasetSchema, OwnedSnapshot, StreamGenerationId, ValueSchema | **本包正当依赖**,zlc-data 契约,照搬 |
| `zlc_neutral_atom.dataset_output`(42-46) | DatasetOutputDeclaration, FinalDatasetOutput, LiveDatasetOutput | **域渗透但内容纯净**:整文件 156 行零物理,只依赖 zlc_data + coverage + canonical_text(见 `dataset_output.py:1-96`)。**整体迁入 zlc_runtime**(producer 输出契约就是本包的公共词汇) |
| `zlc_neutral_atom.runtime.dataset`(47-50) | DatasetCoverage, MonitorCoverage | **切断整模块、抽出两类**:两 dataclass 仅 `runtime/dataset.py:509-555`(含 `_validate_cell_counts`),纯值对象;迁入 zlc_runtime(或 zlc-data)。**绝不拖入 1,920 行的 dataset.py 全文** |
| `zlc_neutral_atom.runtime.signal_source`(51-54) | SignalEventSource, SignalEventAssociationSource | **抽 Protocol 缝**:plane 只调 `value_schema()` 和 isinstance(见 1613、1575);两个 Protocol 在 `signal_source.py:364-370, 433-440` 共约 80 行,迁入 zlc_runtime;`StreamSignalEventSource` 等实现(443 起,绑 streams)留域侧 |
| `zlc_neutral_atom.runtime.streams`(55) | EventRef, StreamId | **抽 id 类型**:`streams.py:77-99` 两个小 frozen dataclass(+可选 `event_ref_to_tree/from_tree` 102-130,lineage 持久化要用);迁入 zlc_runtime 核心 id 模块。**绝不拖入 2,056 行的 streams.py**(monitor tap 机器留域侧) |
| `zlc_storage.canonical_text`(56) | 字符串校验器 | **切断**:27 处调用但功能是 ~10 行文本校验;在 zlc_runtime 内联或由 zlc-data 导出,不为它背 storage 包 |

结论:signal_plane 本体**没有一行域物理渗透**;所有"域依赖"都是契约类型放错了包的问题,拆包时把 6 个 import 收敛为 `zlc_data` + 包内自持,零 Qt 已达成(wake 是回调注入,`bind_owner_wake` 594-605)。

---

## 5. lineage 真实数据结构与 same-shot 可行性

**`direct_parent_refs` 存什么**:`tuple[EventRef, ...]`(L222),`EventRef` = frozen dataclass `(stream_id: StreamId(str), generation: StreamGenerationId, sequence: int)`(`runtime/streams.py:88-99`)。约束:唯一(240-241)、derived 场景按源 sequence 升序(1295-1297)。**它是 ref 级公共契约**;不可变载荷另有私有保留:`_publication_parents: WeakKeyDictionary[SignalPublication, tuple[SignalPublication, ...]]`(581-584,写入点 1388,解析点 1242-1260),parent 载荷随子 publication 存活而传递保活。

**两个 processor 输出能否机械追到共同采集根**:**能,已实现且已被生产路径使用。**证据链:
1. producer 发布永远 `parents=()`(freeze 路径 2356、`publish_final` 1443-1447)→ 采集 publication 就是根。
2. processor 发布强制 `parents=(source_publication,)`(1507-1511);continuous/event derived 记全 parents(1775-1779、1928-1933)。
3. `_publication_roots`(2100-2117)沿 parents 走到无父 ref 即得根集;两个 processor 输出的根 `EventRef` 三元组相等 ⇔ 消费了同一次采集事务。
4. **front 构建已经在做这件事**:根集单一性检验(2238-2243)+ 逐名祖先合并禁止同名双版本(2244-2261)——`SignalFront.continuous_group`(314-323)就是"same-shot 组"的现成消费面(`window.py:3390` 在用);处理器落后时回退整组旧 front(2271-2286),保证组内永不撕裂。

**可行性结论**:same-shot 组订阅**可以直接建在 lineage 上**,且不需要新机制——把 `_publication_roots`/`_collect_publication_ancestry`/组构建抽成 M4 纯函数即是该能力的包级 API。三条边界必须写进 goal:
- **跨 producer 无解且刻意无解**(docstring 15-20):两台独立相机各自成根,无共同祖先,不得为此发明全局 shot id。
- ref 级 lineage(EventRef)可序列化持久(`event_ref_to_tree`,streams.py:102-111);**载荷级** `direct_parent_publications`(1119-1126)是进程内、随 GC 失效的便利品(1250-1254 显式抛错),不得当持久契约。
- `bind_generation_source` 那条 generation 级依赖不产生 publication 级 parents——但该路径本就是零调用死代码(§3),删除后 lineage 语义反而更单纯:publication 图 = 唯一因果权威。