# 任务2审查报告:runtime/ 消费口与宿主(zlc_runtime 拆包底稿)

审查范围:`zlc_neutral_atom/runtime/` 全部文件,全程只读。所有行号基于当前工作树。裁决词汇:**进包** / **换血后进包** / **留域侧** / **拆分**。

---

## 1. 逐文件档案

### streams.py(2,056 行)— 进包,新包地基
- **职责**:单 generation 采集流。`AcquisitionStream.create()` 返回 (stream, producer) 对,producer 持排他写/终止权(`streams.py:1308-1327`)。三种消费口(见 §2)+ Envelope/EventRef/EventSpanRef 事件身份 + `ExactConsumerReadiness` 链式活性证明。
- **公共接口**:`__all__`(`streams.py:2028-2056`)——Stream/Producer/Cursor/Delivery/Envelope/EndOfStream/EventRef/EventSpanRef(含 to/from_tree codec)/ExactReservation/MonitorTap/FollowTap/异常族(StreamGap/StreamEndedEarly/SchemaChanged/SourceFailed)。
- **依赖面**:`zlc_data`(DataBlock/StreamGenerationId/Value,`streams.py:14`)+ `zlc_storage` 校验器(`streams.py:15-21`)。零 Qt、零设备、零 signal_plane。注意 `_contains_materialization`(`streams.py:49`)硬性禁止 DataBlock 作为 payload——流面与物化面的边界是包内不变量。
- **纯度**:纯(threading + weakref + 注入 token,无 I/O)。**唯一雷**:`zlc_storage` 是第三个包依赖,拆包 goal 必须裁决 canonical_text/finite_real 等校验器归 zlc-data 还是 zlc-storage 随行。

### signal_source.py(880 行)— 进包
- **职责**:在已运行的 producer 流上开"源中立信号事件"口。消费者只见声明的 `ValueSchema` 和新鲜 Value+溯源,永不见物理 payload(`signal_source.py:1-6`)。含权威投影(schema-committed transform)包装层。
- **公共接口**:`SignalEvent`/`SignalEventCursor`/`SignalEventSource`(Protocol)/`SignalEventAssociationCursor`+`Source`(Protocol,见 §3)/`StreamSignalEventSource`/`Authoritative*` 四件套 + `SignalProjectionAuthority` codec(`signal_source.py:862-880`)。
- **依赖面**:zlc_data 重度(transform/codec 子模块,`signal_source.py:16-45`)+ runtime.streams + zlc_storage。零设备零 Qt。
- **纯度**:纯。`_apply_signal_value_transform` 把单事件包成 (1,1,…) 合成 DataBlock 走 zlc_data.apply_transform(`signal_source.py:805-859`)——这是"信号投影复用数据面 transform 单源"的关键设计,平移时保持。

### live_dataset.py(477 行)— 换血后进包(一处域侧 import)
- **职责**:`LiveDatasetPort` = 一个 materializer 生命周期 + 合并式变更通知(`live_dataset.py:39-41`);`_ExactDeltaLivePort` = 内部逐 cell 拉 delta 的 worker(`live_dataset.py:234-244`),SignalDataPlane 是唯一 wake-coalescing 所有者。
- **公共接口**:`__all__ = ["LiveDatasetPort"]`(`live_dataset.py:477`);`_ExactDeltaLivePort` 私有,仅 hosted_run 用。
- **依赖面**:runtime.dataset/preview/_failure(包内)+ **`zlc_neutral_atom.dataset_output`**(`live_dataset.py:14-18`)。后者本身近乎纯(zlc_data + runtime.dataset + output_name,`dataset_output.py:13-21`),可整体随包迁入。
- **纯度**:纯逻辑;`_ExactDeltaLivePort` 自起 daemon 线程(`live_dataset.py:268-273`),逐 cell freeze 使 close 能在 cell 间撤销积压(`live_dataset.py:412-417`)——这是设计点,不是杂质。
- **换血项**:`dataset_output` 的三个 Protocol(LiveDatasetOutput/Owner/SnapshotSource)要么随迁,要么在 runtime 包内声明、域侧实现。

### hosted_run.py(693 行)— 拆分:骨架进包,绑定层留域侧(详见 §4)

### owner_mailbox.py(146 行)— 换血后进包
- **职责**:单 headless owner 的 worker 邮箱:submit→done 回调入 completions→`request_owner_wake()` 唤醒(`owner_mailbox.py:95-104`);generation 计数防陈旧完成;terminal job 单飞闸(`owner_mailbox.py:112-136`)。
- **依赖面**:唯一非标依赖 = `runtime.run.RunHandle`(`owner_mailbox.py:9`),且只作为 `set_handle`/`handle` 的类型标注,零方法调用。
- **纯度**:纯(ThreadPoolExecutor)。**换血项**:把 RunHandle 类型改成包内 `RunHandleLike` Protocol(snapshot/cancel/result 三方法)即零依赖进包。"唤醒走回调注入"的样板正是这里的 `request_owner_wake`。

### cancellation.py(84 行)— 进包
- 只读 token / 私有 source 分权:token 无 request 方法,不能偷取消所有权或抢跑硬件封存(`cancellation.py:30-37`);`checkpoint()`/`wait_requested()` 阻塞不轮询。依赖仅 zlc_storage.canonical_text。纯。

### _failure.py(117 行)— 进包
- 字符串化失败证据:`DetachedFailure` 剥离 traceback 帧与 capability 引用(`_failure.py:89-101`,显式清 `__traceback__/__cause__/__context__` 防对象图滞留)——与 streams 的 weakref 所有权纪律同一族。纯 stdlib。

### cleanup.py(45 行)— 进包
- `CleanupReport` 聚合 + `run_cleanup_steps` 全跑不短路(`cleanup.py:30-45`)。纯 stdlib,虽然 docstring 说"device-specific sessions",机制本身通用。

### resources.py(302 行)— 拆分
- **进包半**:`ResourceKey/ResourceClaim/ResourceBusy/ResourceLease/ResourceArbiter`——进程内原子资源仲裁,acquire_all 全有或全无、blockers 返回而非异常(`resources.py:216-238`)、`wait_until_released`(`resources.py:240-271`)。通用调度件,零设备依赖。
- **留域/随 ports 半**:`PhysicalDeviceIdentity/DeviceIdentityEvidenceKind/DeviceBindingStamp` 及其 codec(`resources.py:50-127`)——设备身份语义,是 ports.py 的词汇不是调度词汇。

### preview.py(85 行)— 进包
- 四个小 Protocol + `notify_preview_failure`(best-effort、显示失败绝不掩盖 run 失败,`preview.py:63-76`)。依赖 zlc_data.BlockId + runtime.dataset 类型。纯。

---

## 2. 消费 API 底稿:三种流口 + LiveDatasetPort

### 共同前提(拓扑冻结)
exact 预约、monitor 口**都必须在第一次发布前接入**:`reserve` 拒绝 `_next_sequence != 0`(`streams.py:1394-1397`),`monitor()` 同(`streams.py:1430-1433`),MonitorTap 消费者绑定同(`streams.py:1031-1034`)。只有 FollowTap 可中途加入(`streams.py:1437-1451`)。一个 generation **恰好一个 formal exact 消费者**(`streams.py:1386-1393`),零事件失败预检可重绑(`streams.py:1971-1985`)。

### ExactReservation(streams.py:394)— 不丢帧、ack 即水位
- 协议:`reserve(total_events)` → `activate()→AcquisitionCursor` → `bind_consumer(consumer, terminal=True | downstream=readiness+三回调)` → 逐条 `cursor.next()`(单条未 ack 在飞,`streams.py:891-892`)→ `validate/acknowledge_delivery` → `complete_consumer(eos)`。
- 背压:记录保留到 ack 才裁剪(`_trim_locked` 只裁最早未 ack 水位之下,`streams.py:2018-2025`);ack 严格有序(`streams.py:1950-1951`)。**producer 不被阻塞**——emit 无界追加,exact 落后=内存增长;处理链的真背压在 `ExactConsumerReadiness._await_source_ack`(emit 后等下游真 ack,带 deadline+checkpoint,`streams.py:788-845`)。
- 丢帧契约:**绝不丢**。历史缺口=`StreamGap` 异常(`streams.py:307-315`);emit 时强制"一个覆盖 next sequence 的 live bound 预约"否则拒绝发布(`streams.py:1477-1499`);formal consumer 未绑定前 exact 数据禁止发出(`streams.py:1500-1513`)。
- `ExactConsumerReadiness`(`streams.py:628-649`)= 源到唯一 terminal DatasetBuilder 的进程内活性证明,bind/prepare/start 三点复查,弱引用不延寿(`_ObjectReference`,`streams.py:572-597`)。**这是新包最独特的资产,整体平移。**

### MonitorTap(streams.py:998)— 有账丢帧、无背压
- `next(timeout)` 有序不丢;`latest()` 显式跳到队尾,跳数记入 `MonitorUpdate.missed`(`streams.py:1077-1083`,`streams.py:985-995`)——掉队是消费者的显式选择且有账(MonitorCoverage 的逐信号掉队度量以此为底,`dataset.py:523`)。
- 队列无界,producer 零背压。终止错误清队(`streams.py:1108-1112`);单消费者所有权可选绑定。

### FollowTap(streams.py:1123)— 订阅后无损、可中途加入
- 刻意既非 exact 也非 monitor:可加入运行中的 generation,绝不回放,**没有 latest()**——不能静默跳过已提交值(`streams.py:1126-1131`)。offer/consume 双侧 StreamGap 自检(`streams.py:1173-1178`, `streams.py:1201-1206`)。是 SignalEventCursor 的载体(`signal_source.py:477-484`)。

### LiveDatasetPort(live_dataset.py:39)
- 表面:`bind(dataset)` 一次 / `set_change_listener` 一次(claim 前的变更 pending 重放,`live_dataset.py:98-111`)/ `updated()` 合并通知 / `freeze_current()→MonitorDatasetSnapshot`(冻结后复查生命周期未结束,`live_dataset.py:155-164`)/ `freeze_live_outputs()` 委托域侧 owner / `fail`/`source_terminal`(retain_on_terminal 决定保留或撤回)/`close`。
- 背压契约:无推流——通知只置脏,消费端自拉快照;通知失败单独记账不污染数据失败(`notification_failure`,`live_dataset.py:138-153`)。

**接口摘要(可直接抄进 goal)**:zlc_runtime 消费面 = `reserve→bind_consumer→cursor(ack)` (exact, 不丢) ∥ `monitor→next/latest(missed 记账)` (lossy-by-choice) ∥ `follow→next` (join-late lossless) ∥ `LiveDatasetPort(bind/updated/freeze_current)` (拉模型快照口)。

---

## 3. EventRef / 事件关联机制

### 数据结构
- `EventRef = (StreamId, StreamGenerationId, sequence)`(`streams.py:88-99`)+ 规范 codec(`streams.py:102-130`);`EventSpanRef` 连续区间,end≥start 强制(`streams.py:133-154`)。
- `Envelope.direct_parent_refs`:去重的 EventRef 元组 = 溯源 DAG 边(`streams.py:212-217`);`join_key`:可选、必须可哈希、由流级 `JoinKeyContract` 冻结校验(`streams.py:218-222`, `streams.py:1463-1467`)。
- `SignalEvent` 把 (Value, event_ref, direct_parent_refs, captured_at) 原子携带过投影层(`signal_source.py:223-247`)。

### 发布/消费协议(pulse 关联)
- `SignalAssociationRequest`(`signal_source.py:151-220`):cause_id + cause_digest(sha256)+ expected_event_count + 触发调度指纹/总数/最小间隔/时钟——**因果凭证是编译产物指纹,不是时间戳**。
- 四步协议(`SignalEventAssociationCursor` Protocol,`signal_source.py:373-430`):`arm`(pre-FIRE 原子冻结下一组边界,是因果准入不是容量策略)→ `bind(token, terminal_evidence)`(绑定 producer 观测到的硬件终止)→ `next_associated_signal(token)`(只吐组内成员)→ `finish(token)`(物理对账通过才许成功)。
- Producer 侧实现(camera,`logic_nodes/camera_measurement/signal_source.py:90-312`):arm 时校验"已发布前沿 == 物理序号起点"(`:176-180`)、起点对齐 frames_per_cycle(`:171-175`);next 按 phase + delivered×frames_per_cycle 选帧,序号错组即抛(`:245-248`);finish 用 `stream.wait_until_sequence(physical_end)` 证明硬件 FIRE 之外无帧发布(`:266-273`)。真正的物理对账藏在 `CameraSignalAssociationAuthority` Protocol 后面(`:44-76`)——模拟器进程内观测两侧,真机用 E0 资格+序号+终止回读,**cursor 两路共用**。
- pulse_scan 消费(`logic_nodes/pulse_scan/application.py:538-559` 与 `:590-615`):arm → session.fire → session.complete → validate terminal ↔ artifact → bind → 逐 address collect → finish。源经 `signal_plane.signal_event_binding(key)` 发现并要求 isinstance AssociationSource(`logic_nodes/pulse_scan/logic_node.py:57-60`)。

### 能否一般化为"跨 producer same-shot 组"基础?
**数据基座可以,cursor 协议本身不够,但方向正确。** 判据:
1. 可平移的一般化原料已在包内:`EventRef`(全局唯一身份)、`direct_parent_refs`(跨流因果边)、`join_key + JoinKeyContract`(流级声明的分组键)、`SignalAssociationRequest` 以 cause_id/cause_digest 为键——同一 cause 可以分发给多个 producer 各自 arm,天然是 same-shot 组的组键。
2. 缺口:(a) 当前协议是**单 producer 单连续区间**——`EventSpanRef` 强制连续(`streams.py:147`),camera 实现绑定单流单 phase;(b) 没有 cause 域的组注册中心——SignalDataPlane 刻意不承诺跨 producer same-shot(组件内一致性是它的边界),没有任何对象持有"cause → 各 producer 的 span 集合";(c) `finish` 的对账语义是 producer-私有的(authority Protocol),跨 producer 的"组完成"需要各自 finish 后再做一次消费端 join。
3. **裁决建议**:EventRef/EventSpanRef/join_key/SignalAssociationRequest/四步 cursor 协议全部进包作为 same-shot 的**词汇层**;"跨 producer 组" = 新的可选组件(cause-scoped registry:N 个 producer 各交一个 EventSpanRef + terminal 凭证,按 cause_digest 对账),建包内新模块而非改现有协议——现协议的单 producer 不变量(连续、独占 token)是它可证明性的来源,别为组语义稀释它。

---

## 4. hosted_run.py:LogicNodeHost 解剖

### 生命周期(start/cancel/poll/shutdown 单一表面,`hosted_run.py:133-139`)
- **start**(`:345-355`):先 `_retire_plane_state` + `_reset_generation`(同 host 可重启,generation 计数防陈旧完成),按 kind 分派。finite:有 dataset 输出则 `data_plane.reserve(self)`(`:421-423`),owner.begin_generation,submit execute 闭包(闭包内先查 stop_event,赢 stop 则 `_StartSuppressed`,`:428-436`);processor:见下。
- **cancel**(`:357-369`):finite = 置 stop_event + 已有 RunHandle 则 `handle.cancel(reason)`;processor = `data_plane.cancel_latest_only_processor`(`:633-639`)。
- **poll**(`:371-374`):finite 路径刷 handle.snapshot + drain mailbox completions → `_finish_finite_success/_failure`;成功但声明了输出却没发布 = 硬失败(`:472-481`);live 开过则 `detach_live` 保留 FINAL front,否则 retire 撤回(`:463-489` 注释写明了为什么)。
- **shutdown**(`:376-388`):active 则 cancel+poll,仍 active 则拒关;owner.shutdown(mailbox 反过来要求 pending 清零 + handle 已收割,`owner_mailbox.py:138-143`);最后 `data_plane.detach_live`。
- **execution context**(`:81-130`):传给叶操作的唯一 runtime capability——`cancel_requested/start_and_wait/open_live_dataset/open_exact_dataset/publish_final/warn`。`start_and_wait`(`:504-523`)= starter() 返回 RunHandle、登记进 mailbox、竞态窗口内 stop 已置则立即 cancel、`handle.result()` 阻塞等待。

### processor kind 强制
- `create()` 里:kind=="processor" ⇒ input_specs 中**恰好 1 个** `DatasetInputSpec`,经 application_context.input 解析成一个信号键字符串(`:172-187`)。
- `_start_processor`(`:608-631`):`data_plane.latest_publication(signal)` 不存在即拒;`attach_latest_only_processor(initial_publication=…)` 挂到 plane 的共享 latest-only lane。求值回调链(plane 调回):`evaluate_processor`(`:649-657`,必须返回非空 Mapping)→ `accept_processor_result`(`:659-672`,经 `publish_processor` 带 source_publication 溯源)→ `accept_processor_failure/cancelled`。
- `validate_processor_source`(`:641-647`):输入必须是 SignalValue 且 coverage 是 **MonitorCoverage**——latest-only lane 的掉队记账契约在此闭环。
- finite 与 processor 的发布路径互斥由多处断言强制(`:529-530`, `:545-546`)。

### 耦合点清单(可平移 vs 留域侧)
**可平移进 zlc_runtime(机械改名即可)**:
- `LogicNodeObservation`(纯状态投影,`:50-78`)、phase 状态机、generation 重置纪律;
- `RunOwnerMailbox` 全部(见 §1);
- `LogicNodeExecutionContext` 的表面(六个方法就是最小 Node 能力面);
- live 附着纪律 `_attach_live`(一 generation 一 live、listener→`plane.mark_changed`、失败即 slot.close,`:539-566`);
- processor 求值回调协议(evaluate/accept_result/accept_failure/accept_cancelled 四件套)。

**必须留域侧 / 在缝上换血**:
- `LogicNodeDescriptor` 全体(`logic_node.py:254-279`):authoring_schema、build_request、**device_requirements**、ui_contributions、task_previews、catalog `DefinitionKey`——全是发现/授权/UI 词汇。宿主只需要 create() 后的产物:`(kind, operation, dataset_output_names, artifact_output_names, source_signal)`,即 `hosted_run.py:189-199` 传给 `__init__` 的那组参数——**那个 __init__ 参数表就是最小 Node 协议的现成定义**。
- `LogicNodeApplicationContext`(`logic_node.py:154-245`):paths/device_catalog/open_ui/artifact 记忆——组合根词汇。宿主真正消费的只有 `signal_plane` + `input(spec)` 两项(`hosted_run.py:158-187`)。顺带:`logic_node.py:186-187` 有叠写的双 `@property`(Protocol 上无运行时害,但是缝质量的信号)。
- 信号命名约定 `"@logic/{instance_id}/{name}"`(`:285`)——命名策略应注入,不该烧死在宿主里。
- `SignalDataPlane` 本身(`processing/signal_plane.py:574`):宿主对它是纯注入依赖(9 个方法:reserve/retire/attach/detach_live/mark_changed/publish_final/latest_publication/attach·cancel_latest_only_processor/publish·withdraw_processor)——plane 按用户裁决就是新包的信号权威,**随包走**;这 9 个方法就是宿主↔plane 的契约面,goal 里应显式列出。

**最小 Node 协议提案**(从证据直接导出):`kind ∈ {finite, reactive}`;finite = `execute(ctx) -> result`(ctx 提供六能力 + RunHandleLike);reactive = `evaluate(SignalValue) -> Mapping[str, LiveDatasetOutput]` + 恰一输入信号键。descriptor→协议的翻译层(bind_execute/outputs_for/input 解析)整体留 zlc-atom。

---

## 5. dataset.py / run.py / ports.py 快速裁决

### dataset.py(1,920 行)— **进包**
一句话:它就是数据面的物化核心,零域耦合。证据:import 面只有 numpy + zlc_data + zlc_storage + 包内 `_failure`/`streams`(`dataset.py:1-59`);grep `device|camera|pulse|Experiment|Qt` **零命中**。内容 = `DatasetBuilder`(exact 消费→SealedDatasetArtifact,`:963`)、`MonitorDataset`(MonitorTap 上的 keyed_cycle/latest_cell 物化,`:1362-1451`)、`DatasetCoverage/MonitorCoverage`(`:510/:523`)、`DatasetSealProvenance` codec、`ExactDatasetPreviewReader`(禁 pickle,`:1308-1312`)。它与 streams.py 是一对,拆开毫无意义。

### run.py(1,197 行)— **留域侧(或第三包 zlc-run),zlc_runtime 只留 RunHandle 协议**
一句话:这是硬件安全执行引擎,不是信号/调度面——`RunPlan` 直接携带 `bound_devices: tuple[BoundDevice]` 与 `SafetyInterrupt`,并在 __post_init__ 强制 device claim↔BoundDevice 一一对应(`run.py:85-131`);`RunContext` 拥有 `device()/cleanup_device()/_enable_hardware/_revoke_hardware/_run_interrupts`(`run.py:310-465` 段)。反证其纯度也高:零 zlc_data、零 numpy、零 Qt(`run.py:3-38`)——所以它是**独立自洽的第三关切**,与 admission(resources)+broker(ports)构成闭包,不该为凑"runtime"名字混进信号包。zlc_runtime 侧只需 `RunHandleLike`(snapshot/cancel/result)供 owner_mailbox/hosted_run 引用(见 §1、§4)。

### ports.py(734 行)— **留域侧,与 run.py 同归**
一句话:设备身份+会话+安全中断的经纪层,是 run.py 的直接支撑面。证据:`DeviceBroker.bind` 全 callback 注入(execute_command/capability_probe/close_session/interrupt_operations,`ports.py:396-463`)——**无任何具体驱动 import**,抽象质量很好,但词汇(VerifiedPhysicalDeviceIdentity/SessionCloseCommand/SafetyOperation,`ports.py:30-170`)整段是设备域;run.py 从它 import 七个名字(`run.py:17-24`)。两文件 + resources.py 的设备身份半边是一个不可再分的 seam。

### preview.py(85 行)— **进包**(§1 已述,依赖全在包内+zlc_data)。

---

## 6. 拆包清单(汇总)

| 文件 | 裁决 | 关键动作 |
|---|---|---|
| streams.py | 进包 | 裁决 zlc_storage 校验器归属 |
| dataset.py | 进包 | 原样 |
| signal_source.py | 进包 | 原样(含关联协议词汇层) |
| preview.py | 进包 | 原样 |
| cancellation.py / _failure.py / cleanup.py | 进包 | 原样 |
| resources.py | 拆分 | Arbiter/Key/Lease 进包;PhysicalDeviceIdentity/BindingStamp 随 ports 留域 |
| live_dataset.py | 换血后进包 | dataset_output 三 Protocol 随迁或包内重声明 |
| owner_mailbox.py | 换血后进包 | RunHandle → RunHandleLike Protocol |
| hosted_run.py | 拆分 | 宿主骨架+Observation+execution context+processor 回调协议进包;descriptor/application_context/命名约定留域;`__init__` 参数表(`hosted_run.py:201-213`)= 最小 Node 协议底稿 |
| run.py + ports.py | 留域侧 | 硬件安全执行引擎自成闭包;zlc_runtime 仅依赖 RunHandleLike |
| (processing/signal_plane.py) | 进包(用户已裁) | 宿主↔plane 9 方法契约面显式写进 goal |

**两条横切风险**:① zlc_storage 成为第三个必带依赖(streams/signal_source/resources/cancellation 都用),goal 必须先裁它;② `zlc_neutral_atom.dataset_output` 是 runtime↔域的双向缝(域 import runtime.dataset,runtime.live_dataset import 域)——不随迁就成环,建议整文件迁入新包。