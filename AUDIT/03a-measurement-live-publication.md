# 03a — Measurement / Processor live publication 深审

状态：本子阶段完成。
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：Camera Measurement、Stepped Scan、Seamless Scan、Occupancy，以及它们实际经过的 `NodeHost`、`SignalDataPlane`、coverage / terminal / FollowTap 路径。
约束：只读源码、Git 历史、测试和无硬件隔离探针；未修改 production、tests、旧文档或硬件。

## 1. 结论先行

当前的“Measurement 应持续 live 发布”只在表面成立，底层契约没有成立。最严重的不是少一个 preview，而是下面五件事同时发生：

1. finite Camera 用补零后的**完整未来容量**冒充当前 snapshot，只在 snapshot 外另放一个 coverage 计数；future cells 在 `OwnedSnapshot` 内仍是 `VALID`。
2. Occupancy 忽略 parent coverage 和 parent validity，处理全部补零 cells，并把自己的输出重新声明成 complete。
3. finite source 成功 terminal 时另发 `FinalDatasetOutput(coverage=None)`；正在 Follow 的 Processor 会把这份 terminal publication 当下一份 live Dataset，随后硬失败。
4. Camera/Scan 每个新点都重新冻结整块预分配数组；Occupancy 每次又从头分类整块，形成确定的 O(N²) copy/freeze/classify。
5. live slot 的“先改对象、再 mark dirty、以后 freeze”不是原子提交。并发 freeze/update 可以给**同一 snapshot revision 发出两个 publication sequence**。

因此当前以下产品事实都不可靠：

- 加上 Occupancy 不会改变 Camera 采集节奏；
- finite Camera live 与 terminal 是同一 Dataset 的两个阶段；
- Stop 后留下的数据与 UI 是否正好刷新无关；
- coverage 能阻止未采 cell 被当作科学事实；
- finite Processor 能从 live 一直跟到 terminal；
- 同一 `frames_per_cycle` 组一定来自连续物理 frame ordinal。

这是一个 `REDESIGN` 级问题，不适合继续在 Camera、Occupancy、Scan 各自补条件分支。

## 2. 当前真实链路

### 2.1 finite Camera，没有 follower

~~~text
camera worker
  -> FiniteCapture.next_cycle()
  -> cycles list append
  -> _CameraLiveSlot.update(all_cycles_so_far)
  -> SignalDataPlane.mark_changed(slot)
  -> UI/owner 某次 SignalDataPlane.freeze()
  -> _CameraLiveSlot.freeze_live_outputs()
       -> 给 future cycles 构造 CameraFrameRecord(blank)
       -> stack 整个 authored repeat 容量
       -> OwnedSnapshot 再冻结可变 ndarray
       -> DatasetCoverage(written, authored total)
  -> transient live SignalPublication
~~~

采集完成后同一 worker 又走：

~~~text
all actual cycles
  -> frames_snapshot(actual only)
  -> FinalDatasetOutput
  -> publish_final()
  -> terminal SignalPublication(coverage=None, transient=False)
~~~

这不是一个 authority 的一次 seal，而是 live slot 与 final publisher 两条并行物化路径。

### 2.2 finite Camera，有 Follow Processor 或 Scan follower

`follow_publications()` / `reserve_follow_processor()` 会给 source 建 publication stream。此后 `mark_changed()` 不再只是便宜的 dirty mark，而会在 **camera worker 自己的线程**同步执行：

~~~text
mark_changed
  -> _publish_followed
  -> slot.freeze_live_outputs   # 整块复制在采集线程发生
  -> publication stream emit
  -> unbounded FollowTap queue
~~~

因此“有没有下游 follower”会改变 producer 的工作量和 shot cadence。加一个 Logic Node 不是旁路观察，而会把完整 snapshot freeze 插进采集循环。

### 2.3 Scan

Stepped 与 Seamless 共享：

~~~text
source SignalValue
  -> ScanDatasetWriter.write(one point)
  -> writer.live_output()
       -> writer.snapshot()
       -> copy complete preallocated values buffer
       -> scan/compact complete validity buffer
  -> ScanLiveSlot.publish(immutable full front)
  -> mark_changed / freeze / FollowTap
~~~

Scan 的 unfilled cells 至少在 snapshot validity 中是 false，这比 Camera 正确；但每一点仍冻结全容量，性能是 O(N²)。

### 2.4 Occupancy

~~~text
SignalValue(snapshot + coverage)
  -> OccupancyProcessor.evaluate()
       -> discard coverage
       -> discard snapshot validity
       -> classify every repeat x point frame from scratch
       -> build five new full snapshots
       -> give every output DatasetCoverage(total, total)
  -> publish_processor(parent publication)
~~~

`valid` 当前只表达 calibration/model 是否可用，未表达“这个 frame 是否已经采到”或 parent data 是否 valid。

## 3. 已确认缺陷

### LIVE-001（P0）— 未采 Camera cell 被发布为有效科学数据

位置：

- `camera_measurement/measurement.py:334-372`
- `camera_measurement/measurement.py:351-355`
- `occupancy/processor.py:269-389`
- `occupancy/processor.py:391-419`

Camera future cycles 使用零图补齐，但 `frames_snapshot()` 没有收到 validity；整个 `OwnedSnapshot` 因而是 `VALID`。`DatasetCoverage` 只在 snapshot 外说 `written < total`。

Occupancy 只读取 `signal_value.snapshot`，完全不读取 `signal_value.coverage`，也不读取 `snapshot.expanded_validity()`。它对补零图做真实 threshold classification，并为五个输出构造 `DatasetCoverage(total,total)`。

隔离探针：parent 为 `coverage=(1,4)`，只有第一帧是真实亮帧，后三帧是 current Camera 路径的补零帧。结果：

~~~text
parent coverage       (1, 4)
Occupancy coverage    (4, 4)
Occupancy valid       [True, True, True, True]
Occupancy occupied    [True, False, False, False]
input snapshot valid  all True
~~~

后三个 `False` 不是“测得 empty”，但当前数据把它们写成了有效 empty。任何 rate、fit、overlay、feedback 或保存只要没有额外读 parent coverage，就会把未发生的 shot 当事实。

裁决：

- Camera future cells 必须在 Dataset validity 中 invalid；coverage 不能替代 data validity。
- Occupancy 的 counts/occupied/rate/frame_judged snapshot validity 必须继承或正确投影 parent validity。
- Occupancy 的布尔 `valid` 至少是 `parent cell present/valid AND site/model valid`；未采 cell 自身还应是 Dataset-invalid。
- Processor 输出 coverage 必须由 parent committed cells 经明确映射得到，不能无条件 `total,total`。

### LIVE-002（P0）— finite live Processor 在正常 source terminal 时失败

位置：

- `host.py:935-1019`
- `host.py:971-978`
- `plane.py:1481-1533`

Follow Processor 要求每一份 source value 都有 `DatasetCoverage`。finite Camera/Scan terminal 却通过 `publish_final()` 发出 `coverage=None` 的最后一份 publication。FollowTap 会先交付这份 terminal publication，再报告 EOS；`_run_follow_processor()` 因而在到达 EOS 分支之前就在 `_validate_follow_source()` 失败。

通用 runtime 探针：

~~~text
processor terminal = True
processor phase    = failed
processor error    = ValueError: Follow Processor input must have DatasetCoverage
source terminal coverage = None
~~~

真实 virtual Camera→Occupancy 探针（repeat=2）：

~~~text
first Camera live coverage    (1, 2)
first Occupancy coverage      (2, 2)
first Occupancy valid         [True, True]
Camera terminal               done
Occupancy terminal            failed
Occupancy error               Follow Processor input must have DatasetCoverage
~~~

现有测试只分别证明了三条孤立路径：

- live DatasetCoverage + `finish_live()`；
- 已完成 Final + frozen one-shot Processor；
- latest MonitorCoverage Processor。

没有测试 production 正在做的第四条路径：“先 DatasetCoverage live，后 Final terminal”。

裁决：source terminal 是生命周期事实，不能由 coverage 类型猜。Follow subscriber 应收到明确 terminal transition；若 terminal 带一份 full replacement snapshot，Processor 要么将它作为 seal/final reconcile 处理，要么明确跳过重复 full recompute，不能把它当另一份 live event。

### LIVE-003（P0）— 一次 UI freeze 改变 Camera Stop 后的数据结构

位置：

- `FiniteCapture.collect():514-558`
- `CameraMeasurementNode.execute():919-968`
- `NodeHost._end_run():591-624`
- `SignalDataPlane.finish_live():1811-1909`

同一个 `repeat=4` Camera run，采到一帧后 Stop。隔离探针唯一变量是 Stop 前有没有执行一次 `plane.freeze()`：

| Stop 前 | terminal shape | coverage | transient | Host phase |
|---|---:|---:|---:|---|
| 没有 freeze | `(1,1,96,128)` | `None` | `False` | `cancelled` |
| 有一次 freeze | `(4,1,96,128)` | `(1,4)` | `True` | `cancelled` |

原因：

1. `collect()` 在 cancel 后仍把已采 cycles 交给 `_publish_finite()`；
2. 如果 live schema 从未进入 plane，partial actual-only Final 可以成功发布；
3. 如果 UI beat 已发布固定 authored schema，partial actual-only Final 因 schema 不同失败；
4. Host 把该异常重标为 cancelled，再由 `finish_live(cut_short=True)` 保留 padded live snapshot。

结果是同一用户动作只因显示时序不同而得到两个互斥事实：

- 一个看起来是完整 Final，但 repeat 轴缩短且没有 partial marker；
- 一个保持 authored geometry，但仍标 `transient=True` 且 coverage 非空。

这也决定之后 Processor 能否启动：前者走 frozen one-shot，后者被误判成 live Follow并因 source 已 terminal 而拒绝。

裁决建议：finite Stop 后若已有 committed cell，应保留**同一个固定 authored schema**，unfilled cells invalid，publication 明确为 terminal-partial、retained、non-transient，并可被 Processor 一次性消费。UI 是否刷新过不得参与结果。

### LIVE-004（P0 性能）— Camera freeze 与 Occupancy classify 均为 O(N²)

Camera 每次 update 把 `tuple(cycles_so_far)` 重新构造一次；每次 freeze 又为所有 future cycles 构造 placeholder records，并 stack 整个 authored repeat 容量。`CameraFrameRecord` 的 `replace(record, image=blank)` 还会经 `__post_init__` 把每张 blank 再复制成 bytes-backed array。

virtual `96 x 128 uint16`、一帧/cycle 的无硬件探针：

| Repeat N | N 次 live update+freeze | 完整容量 payload 累计下界 |
|---:|---:|---:|
| 25 | 0.0245 s | 14.6 MiB |
| 50 | 0.0868 s | 58.6 MiB |
| 100 | 0.3547 s | 234.4 MiB |
| 200 | 1.4664 s | 937.5 MiB |

N 翻倍，时间约四倍。payload 只算 `N publications × N frames × frame bytes`，未计 placeholder、内外 `np.stack`、immutable boundary 和 FollowTap backlog 的额外副本。

若是 `2048 x 2048 uint16`：

- 一帧是 8 MiB；
- repeat=200 时一份 padded live snapshot 已约 1.56 GiB；
- 200 次 publication 的逻辑 payload 下界约 312.5 GiB；
- 三帧/cycle 再乘三。

Occupancy 每份 cumulative snapshot 都从第一帧重新做全部 site extraction。35 sites 探针：

| Publications | 每份 frames | site-frame evaluations | 时间 |
|---:|---:|---:|---:|
| 25 | 25 | 21,875 | 0.106 s |
| 50 | 50 | 87,500 | 0.441 s |
| 100 | 100 | 350,000 | 1.702 s |

这还只是 BOX、小图、一个 Processor。PSF、真实 ROI、多 follower 和 renderer 会叠加。

### LIVE-005（P1 性能）— Scan 每点冻结完整 values + validity

位置：`scan/dataset.py:273-343`。

`ScanDatasetWriter` 的 mutable buffer 是正确的单一积累位置，但 `live_output()` 每点调用 `snapshot()`；`owned_snapshot_from_arrays()` 必须把 mutable values 复制到不可变 bytes，并扫描/压缩完整 validity。N 个点因此是 N 次完整容量物化。

`64 x 64 uint16` 探针：

| Points N | N 次 `write+live_output` | values payload 下界 | validity 扫描下界 |
|---:|---:|---:|---:|
| 25 | 0.0021 s | 4.9 MiB | 2.4 MiB |
| 50 | 0.0046 s | 19.5 MiB | 9.8 MiB |
| 100 | 0.0379 s | 78.1 MiB | 39.1 MiB |
| 200 | 0.1906 s | 312.5 MiB | 156.2 MiB |

对历史验收中出现的 `9 x (1200 x 1920 uint16)` image scan，仅 live writer 逻辑下界约 values 356 MiB + validity 178 MiB；27 点则约 3.13 GiB + 1.56 GiB，尚未进入 plot projection/RGBA。

`SteppedScanMeasurement._capture_shot()` 还在每 shot 调 `len(self.plan.rows())`，重复重建完整笛卡尔积；应使用已有 `point_count` 或 execute 中 `rows` 长度。它不是主瓶颈，但属于明确可删的 O(N²) 边角。

### LIVE-006（P0 same-shot）— finite 与 monitor 没有共用 cycle assembler

设计文字要求 `frames_per_cycle` 只有一个共同 owner，并依 source ordinal 拒绝 gap/错位。实际：

- `MonitorCapture._accept_record()` 检查 ordinal 连续及 cycle 对齐；
- `FiniteCapture.next_cycle()` 只数“返回了几张”，不看 ordinal；
- `CameraCycleSource`（Temperature、SLM Feedback 的真实路径）直接复用 finite path。

隔离探针把 ordinal `(0,2)` 交给两帧 finite cycle，当前原样接受：

~~~text
accepted_source_ordinals = (0, 2)
~~~

这会把物理 gap 两侧的帧拼成同一个 cycle。对 Temperature 是 before/after 错配；对 SLM Feedback 是 candidate/shot 错配。

设备侧也不完全满足契约：

- DCAM 在 ring overrun 时拒绝，ordinal 来自 transfer count；
- Pylon 当前 `source_ordinal=self._grabbed-1`，没有读取 driver image/block/skipped counter；driver skip 后软件 ordinal 仍可能连续；
- finite Camera 的 `_publish_finite()` 不验证 `CameraCaptureTerminalRecord` 的 count/no-more/joined，而 `CameraCycleSource.close()` 只在完整读取后验证。

裁决：共同 assembler 必须在 Camera Measurement owner 内由 finite/monitor/cycle-source 三者复用；adapter 只能报告真实物理 ordinal/gap。无法取得物理 counter 的设备必须明确降低保证，不能用“收到的第几张”冒充物理连续。

### LIVE-007（P0 concurrency）— dirty/freeze split 可重复发布同一 revision

位置：

- `plane.py:966-1044`
- `plane.py:1947-2069`
- `_CameraLiveSlot` 无锁；`ScanLiveSlot` 虽有锁，也没有 change-token acknowledgement。

当前顺序是：slot 先改变，`mark_changed()` 把 owner id 放入 set，之后某个线程移除 dirty，再在 plane lock 外读取 slot。若读取期间又 update，第一次 freeze 可以读到新值，而新 dirty mark 仍留下；下一次 freeze 再发布同一个新值。

确定性 barrier 探针：

~~~text
first publication sequence   1
first snapshot revision      2
second publication sequence  2
second snapshot revision     2
slot freeze calls             2
~~~

同一个 immutable snapshot ref 被发成两次事件。FollowTap/Scan 会把它当两个 shot；plot 可能按 snapshot revision 丢第二次。`AUDIT/02` 已另行证明 plane 还接受倒退/重复 snapshot revision。

裁决：live 更新必须是带 commit token/revision 的原子 immutable bundle 提交，不能继续用“mutate slot -> dirty set -> later pull”。至少 plane 必须冻结 dataset family 并拒绝相同/倒退 content revision；更干净的是 producer commit 时交出 immutable event，presentation 只 coalesce 已提交 event。

### LIVE-008（P0 terminal）— stopped exact Dataset retained 但不能 replay

`finish_live(cut_short=True)` 当前把 partial exact source 留在 plane：

~~~text
is_generation_live = False
coverage           = DatasetCoverage(1, 4)
transient          = True
~~~

随后启动 Processor，`NodeHost._start_processor()` 只看 coverage 类型而选择 Follow；plane 又正确发现 generation 已 terminal，于是拒绝：

~~~text
RuntimeError: Follow Processor source generation is not live
~~~

所以“Stop 后 measured data 仍可供 panel/Edit/Save/下游使用”只实现了一半。它留在 registry 中，但下游把它误分类。

推荐 terminal matrix：

| Source | Success | Stop after data | Failure |
|---|---|---|---|
| finite exact | retained complete | retained partial；固定 schema；unfilled invalid；可 one-shot process/save | withdraw，除非节点有明确 last-good 语义 |
| infinite latest | 默认 withdraw | 默认 withdraw | withdraw |

coverage 继续描述 committed extent；terminal/outcome 单独描述生命周期；`transient` 不应在 terminal-retained value 上仍为 true。

### LIVE-009（P1 architecture）— 四套 live API 并存

当前 production 节点各自实现：

- `_CameraLiveSlot`
- `ScanLiveSlot`
- `CalibrationCapturePreviewSlot`
- SLM Feedback 持有的 `ScanLiveSlot`

runtime 同时维护：

- `LiveDatasetPort`
- `_ExactDeltaLivePort`
- `DatasetBuilder` / `ExactDatasetPreviewReader`
- `MonitorDataset`
- `NodeExecutionContext.open_live_dataset/open_exact_dataset`

静态 consumer 搜索显示后一整套没有任何 production Logic Node 使用；只有 runtime 自身和 tests 使用。Scan 另写 `ScanDatasetWriter`，Camera 又写 cumulative list/padded snapshot。

这不是“多种业务投影”，而是 parallel lifecycle / ownership / notification / terminal 机制。当前缺陷正出在这些机制的差异处。

需要执行 `DECISIONS.md` 的 D-007 与 D-011：

- 若保留 runtime builder/delta 体系，就必须让真实 Camera/Scan/Processor 端到端使用并删除 custom slot；
- 若产品不需要这套通用框架，就删除它，收敛成简单 Host-owned live commit API；
- 不能继续两套都保留、production 永远不用“通用”那套。

### LIVE-010（P1 coupling）— Scan source 为取一个值会 freeze 整个 plane

`PublishedSignalSource.discard_pending()` 与 `next_value()` 在 scan worker 内反复调用 `signal_plane.freeze()`。一次 freeze 会 drain 全局 latest Processor lane、freeze 所有 dirty producers、route 全部 publications 并重建 front。

因此一个 scan 等自己选中的 source 时，会替 UI/无关 producer 做全局工作。相反，当 source 有 follower 时，producer 又在自己线程同步 `_publish_followed()`。consumer topology 决定谁承担 materialization，线程 owner 不稳定。

建议：source commit 与 presentation freeze 分离。FollowTap 应跟随 producer 已提交的 immutable events，不应靠 scan worker 主动 pump 全局 display plane 才产生事件。

### LIVE-011（P1 safety）— Scan cleanup 可跳过 sequencer.safe

两条 scan 的 finally 都是：

~~~python
finally:
    self.source.close()
    self.sequencer.safe()
~~~

如果 `source.close()` 抛错，`safe()` 不执行。普通 `PublishedSignalSource.close()` 很少失败，但 `CameraCycleSource.close()` 会验证 terminal camera proof并可能抛错，它正被 Temperature/SLM Task 使用。

应使用嵌套 `try/finally` 保证 safe；若 close 与 safe 都失败，保留 primary error 并附 secondary。现有测试没有注入 source-close failure。

### LIVE-012（P1 truth）— generation/revision 至少四个 authority

同一链存在：

1. plane `_GenerationState.generation + next_sequence`；
2. `OwnedSnapshot.ref.stream_generation + revision`；
3. `CameraMeasurementNode._generation + _revision`；
4. `_CameraLiveSlot.revision`（递增但无人读取）；
5. scan writer `_written` 又兼作 snapshot revision 和 coverage 基数。

Camera 注释要求 revision 跨 run 永远递增；根架构却说 revision 只在同 generation 内比较，新 generation 可从1开始。测试 `test_live_plot_accepts_successive_shots.py` 又用同一个 plot session跨 generation，反向钉住全局递增。

plane 不校验 snapshot generation/revision 与 publication generation/sequence 的关系，race 探针已证明两套编号会分叉。

此项进入 D-018。建议：

- `EventRef` 唯一承担 run/causal publication identity；
- snapshot ref 只承担 dataset content identity，但在一个 publication generation 内严格单调且同 ref 必须同 bytes；
- sibling bundle 的 content stamp 有明确一致规则；
- 删除 `_CameraLiveSlot.revision`；
- 不用跨 generation 单调 revision 掩盖 plot 没正确处理 generation replacement。

## 4. 推荐的统一 live 契约

以下是需要先裁决的语义，不是要求立即新增一组类。

### 4.1 不变量

1. **commit 原子**：一个 producer revision 的 sibling outputs、coverage、run record 和 content stamp 一次提交；不再先改 slot、以后 pull。
2. **validity 在数据内**：unfilled/unobserved cell 必须 Dataset-invalid；coverage 只是 committed extent 摘要。
3. **lifecycle 独立**：`live / terminal-complete / terminal-partial / failed/withdrawn` 不由 `MonitorCoverage / DatasetCoverage / None` 猜。
4. **consumer mode 独立**：latest-coalescing 与 exact-every-commit 是 subscription 语义，不是 Dataset shape 或 coverage 类型。
5. **terminal 只 seal 一次**：同一 output 不再先 live materialize、再走另一份 final materializer；terminal snapshot 来自同一 accumulator。
6. **Stop 可复现**：有没有 panel、是否刚好有 beat、是否已有 Processor 都不能改变 terminal schema/content。
7. **Processor 投影 parent 状态**：coverage、validity、generation/parent EventRef 都有明确映射。
8. **producer cadence 不受 observer topology 改变**：加 panel/Processor/follower 不得把全量 copy 突然插入 camera read loop。

### 4.2 Camera 与 Scan 不必拥有相同 live 视图

“统一 API”不等于强迫所有 Measurement live 数据长一样。

- Camera 操作者通常需要**最新完整 cycle**；历史由 node-owned accumulator 收集，terminal 冻结完整 Dataset 一次。live copy/classify 可为 O(1) per cycle。
- Scan 的产品价值就是看 grid 逐点填充；它可保留 fixed-geometry growing view，但只在 presentation cadence 物化完整 view，Processor 消费新 cell/delta。

推荐对 D-006 的裁决因此是：Camera 选 latest-cycle live；Scan 保留 growing live。共同契约统一 commit、validity、terminal 和 lineage，不统一业务投影形状。

### 4.3 Processor API 的最小需求

~~~text
attach live exact:
  evaluate current committed state once
  then process each newly committed delta exactly once
  on source terminal: seal current output terminal

attach live latest:
  process newest committed value; busy时可coalesce

attach terminal retained:
  process retained complete/partial snapshot once
~~~

Occupancy 最小增量状态是记录已处理的 `(repeat, point)` cells，只对新有效 frame 调 `signals()`；future cells 保持 invalid。若不愿引入 stateful Processor，Camera live 就应只发布 latest cycle，并在 terminal full snapshot 做一次完整 reconcile；不能继续每次发全历史、每次从头算。

## 5. 逐文件 / 类 / 函数裁决

### 5.1 Camera Measurement

| 文件 / 符号 | 裁决 | 理由 |
|---|---|---|
| `camera_measurement/logic_node.py` | `PASS WITH DEBT` | descriptor、request authoring、ROI mapping owner正确；根文档frame sibling规格陈旧。 |
| `_validate_measurement`、ROI helpers | `PASS` | 本plugin直接authoring/selection规则。 |
| `_frames_preview` | `PASS` | 稳定 `frames` + facet grid，不随frame count改vocabulary。 |
| `CameraMeasurementRequest` | `PASS` | frozen per-run request owner正确。 |
| frame/axis helpers | `PASS` | schema/坐标集中且cache有真实高频consumer。 |
| `frames_snapshot` | `REDESIGN` | 双层stack+immutable copy；不表达future validity；只取image丢record ordinal/timestamps。 |
| `_camera_working_point_snapshot` | `PASS WITH DEBT` | 必要run provenance；nested mapping只是shallow-frozen，属runtime通用问题。 |
| `_CameraLiveSlot` | `DELETE after replacement` | parallel live framework；无锁；dead revision；fake records；O(N²)；terminal双truth。 |
| `CameraCycleSource` | `PASS WITH CRITICAL FIX` | Temperature/SLM真实第二消费者；必须复用共同ordinal assembler。 |
| `MeasurementResult` / `frames` | `PASS` | direct/notebook caller有真实消费。 |
| `FiniteCapture.collect` | `REDESIGN` | cancel仍尝试Final；partial依UI timing；不验证terminal proof；累计tuple边角O(N²)。 |
| `FiniteCapture.next_cycle` | `REDESIGN` | stop slicing正确；只数数量、不验证ordinal连续/对齐。 |
| `FiniteCapture.close` | `PASS WITH DEBT` | cleanup必要；重复close返回虚构zero terminal record，应保留真实terminal。 |
| `MonitorCapture` | `PASS WITH DEBT` | latest complete-cycle产品存在；assembler不应与finite分叉；依赖待删除slot。 |
| `CameraMeasurementNode` | `PASS WITH DEBT` | concrete node owner正确，没必要因LOC机械拆文件；删slot/双物化后会缩小。 |
| `read_records` | `PASS` | counts/photoelectron单一intake正确。 |
| `_next_publication_stamp` | `REDESIGN` | 跨generation counter与EventRef分叉。 |
| `prepare` / `monitor` | `PASS WITH FIX` | device configure/arm owner正确；direct arm失败需回收已reserve generation。 |
| `execute` | `REDESIGN` | repeat=0/finite分流合理；finite attach-live+publish-final双路径是P0根因。 |
| `_publish_finite` | `REDESIGN` | 应seal同一accumulator；当前不验证terminal proof并与live schema竞争。 |
| `self.producer` alias | `DELETE` | Camera/Scan只需 `instance_id`；alias无production consumer。 |

`CameraMeasurementNode` docstring 仍写“one signal per frame”，实现和tests已是一个 `frames` Dataset；应在 D-004 裁决后改文档，而不是反改代码迎合旧句子。

### 5.2 Scan shared library

| 文件 / 符号 | 裁决 | 理由 |
|---|---|---|
| `scan/plan.py` 的 plan/port/bind/load函数 | `PASS` | 两个scan node的共享事实，owner正确。 |
| `_unique_domain` | `PASS WITH DEBT` | 一次schema build可接受；list.index复杂度不是主瓶颈。 |
| `check_cancelled`、`wait_for_board` | `PASS` | 两scan/Task真实共享；有限wait有产品证据。 |
| `PublishedSignalSource` | `REDESIGN` | adapter必要，但不应由scan worker pump全局 `plane.freeze()`。 |
| `scan_dataset_schema` | `PASS` | 正确保留source schema、point columns、topology和scan coordinates。 |
| `ScanDatasetWriter` | `REDESIGN` | mutable accumulator是正确owner；每点full immutable snapshot/validity scan为O(N²)。 |
| `ScanDatasetWriter.write` | `PASS WITH DEBT` | address/validity写入正确；首点重复展开一次source validity。 |
| `ScanDatasetWriter.snapshot` | `PASS for terminal only` | terminal freeze需要；不应每point调用。 |
| `ScanDatasetWriter.live_output` | `REDESIGN` | presentation与exact Processor应分别看cadence view/delta。 |
| `ScanLiveSlot` | `DELETE after replacement` | 锁比Camera slot正确，但仍是第三套slot且无atomic token。 |
| `SeamlessScanMeasurement` | `PASS WITH DEBT` | board loop有第二消费者，放shared scan package正确。 |
| `_streamed_sequence`、`_wire_table` | `PASS` | board slot/table owner准确。 |
| `acquire` | `REDESIGN` | live full copy每点；cleanup必须保证safe；played-order assignment本身直接。 |
| `run_record`、`execute` | `PASS WITH FIX` | final意义正确；需和live seal统一，避免Follow terminal failure。 |

### 5.3 Stepped / Seamless descriptors and engines

| 文件 / 符号 | 裁决 | 理由 |
|---|---|---|
| `stepped_scan/logic_node.py` | `PASS` | authoring、device/source/resource wiring在正确owner；preview已声明。 |
| `_build` | `PASS WITH DEBT` | LIVE source generation在Start冻结正确；受底层lifecycle问题影响。 |
| `SteppedScanMeasurement` | `PASS WITH DEBT` | 单消费者loop留concrete plugin符合规则。 |
| `_split_row`、`_api_values`、`_apply` | `PASS WITH DEBT` | 职责清楚；settle不可取消，超长authored settle会拖Stop。 |
| `execute`、`_collect`、`_capture_shot` | `REDESIGN` | live writer O(N²)，cleanup safe问题，`plan.rows()`重复构造。 |
| `seamless_scan/logic_node.py` | `PASS` | board-only限制、descriptor、preview正确。 |

### 5.4 Occupancy

| 文件 / 符号 | 裁决 | 理由 |
|---|---|---|
| `occupancy/logic_node.py` | `PASS WITH DEBT` | model/artifact/source authoring正确；typed overlay缺失详见 `AUDIT/02`。 |
| `OccupancyResult` | `PASS WITH DEBT` | direct Task/notebook有真实消费者；artifacts dict仍可变不是本轮P0。 |
| `inherited_stamps` | `REDESIGN` | 用snapshot stamp模仿parent同步；真正causal truth是parent EventRef。 |
| `_source_point_column` | `REDESIGN` | Camera单列可用；拒绝多列PointTable，使image-valued scan虽source-neutral却无法处理。 |
| source run-record/unit validation | `PASS` | geometry/unit owner正确；runtime rebasing有真实需求。 |
| `process` | `REDESIGN` | 丢parent validity/coverage；每次全历史重算；五输出validity不真实。 |
| `_live_outputs` | `REDESIGN` | 无条件 `DatasetCoverage(total,total)` 是明确假truth。 |
| `evaluate` | `REDESIGN` | 必须把整个 `SignalValue` extent/validity/parent commit带入。 |
| `occupancy/overlay.py` | `REDESIGN` | 这里只登记跨链；active node临时组装、ROI rebase与terminal问题已在 `AUDIT/02`。 |

### 5.5 Runtime 接缝

| 文件 / 符号 | 裁决 | 理由 |
|---|---|---|
| `DatasetOutputDeclaration` | `PASS` | 稳定output/contract identity必要。 |
| `FinalDatasetOutput` / `LiveDatasetOutput` | `REDESIGN` | 载体必要；当前用两个类型间接表达lifecycle，且只核对coverage total。 |
| `DatasetCoverage` / `MonitorCoverage` | `PASS WITH DEBT` | extent summary必要；不能当live/exact/frozen dispatch tag。 |
| `DatasetBuilder` / `MonitorDataset` / delta types | `USER DECISION` | 设计完整但production零消费者；要么唯一实现，要么删除。 |
| `LiveDatasetPort` / `_ExactDeltaLivePort` | `USER DECISION` | 与custom slots平行；当前只被tests消费。 |
| `preview.py` protocols | `USER DECISION` | 只服务无人使用路径；随D-011裁决。 |
| `NodeExecutionContext.attach_live_outputs` | `REDESIGN` | 需要atomic commit/seal。 |
| `open_live_dataset/open_exact_dataset` | `MERGE OR DELETE` | production零调用；不能与custom slot并存。 |
| `NodeHost._start_processor` | `REDESIGN` | 按coverage class猜lifecycle，terminal partial误路由。 |
| frozen/follow/latest三套Processor execution | `MERGE` | 需要三种消费语义，但不应分成plane lane与Host mailbox两套lifecycle。 |
| `_run_follow_processor` | `REDESIGN` | 不理解terminal Final replacement；正常Camera/Scan结束会失败。 |
| `_end_run` / `_finish_worker_success` | `REDESIGN` | retain意图正确；Camera双路径使结果依UI timing。 |
| `mark_changed` / `_publish_followed` / `freeze` | `REDESIGN` | consumer-dependent同步工作、atomicity race、global pump耦合。 |
| `_publish_locked` | `REDESIGN` | lineage bundling必要；缺snapshot family/generation/revision校验。 |
| `finish_live` | `REDESIGN` | partial retention目标正确；留下transient值却不能frozen replay。 |
| `front.py::build_front` | `PASS` | parent lineage/coherent family算法不是本轮根因。 |
| `FollowTap` | `PASS WITH DEBT` | exact ordered follow必要；full cumulative snapshots使unbounded queue放大。 |
| `RunOwnerMailbox` | `PASS` | worker completion mailbox职责集中。 |

## 6. 测试逐项裁决

### 6.1 Camera / Occupancy

| 测试 | 裁决 |
|---|---|
| `test_a_node_host_runs_a_camera_measurement_to_completion` | `KEEP`；守complete final/run record/schema；未连接live Processor。 |
| `runs_and_stops_repeat_zero` | `KEEP`；infinite detach/disarm有效。 |
| `hosted_and_direct_paths_share_one_acquisition` | `DELETE/REPLACE`；源码substring，不证明行为等价；partial/live现已实证不等价。 |
| `finite_run_shows_its_dataset_filling` | `REDESIGN TEST`；钉住padded geometry但不查future validity、copy量或下游。 |
| `stops_within_a_read_slice` | `KEEP`；cancel latency有价值；需加“采到1帧后Stop”terminal不变量。 |
| monitor complete-cycle test | `KEEP`；monitor gap/alignment是有效纵向守卫。 |
| repeated-freezes zero-copy test | `REDESIGN TEST`；直接测 `snapshot_from_array(single view)`，没有走真实 `frames_snapshot` stack。 |
| finite external-trigger test | `KEEP WITH ADDITION`；只测count；需注入ordinal gap与terminal proof failure。 |
| `test_derivation_boundary.py` finished/frozen | `KEEP`；证明已完成Final one-shot；需补live→Final与partial terminal。 |
| `test_installation_and_nodes.py::...same_shot_front` | `RENAME/REPLACE`；FakePlane+direct `process()`，没有SignalPublication/front/lineage。 |
| `test_real_runtime_integration.py` | `PASS WITH DEBT`；runtime只用于monitor；Occupancy仍offline direct process。 |
| `test_live_plot_accepts_successive_shots.py` | `REDESIGN`；跨generation强制revision全局递增，与generation replacement冲突。 |
| Occupancy oracle/mutation tests | `KEEP`；数学有效；全部输入complete-valid，未守partial。 |

缺失的必红守卫：

1. parent coverage `1/N` 时 Occupancy 只能把已采cells写成事实；
2. parent snapshot validity false 必须传到五个输出；
3. finite Camera + live Occupancy 正常结束双方均done且retained；
4. Stop 前有/无 `plane.freeze()` terminal snapshot严格相同；
5. stopped partial terminal可启动Processor one-shot；
6. ordinal `(0,2)` 对两帧cycle必须拒绝；
7. terminal camera proof不完整不得发布成功Final；
8. 实际 `frames_snapshot` copy/bytes路径，而不是helper旁路。

### 6.2 Scan

| 测试 | 裁决 |
|---|---|
| Stepped gating/shot/repeat/device-port tests | `KEEP`；物理执行语义清楚。 |
| Stepped live fill assertion | `KEEP WITH FIX`；只收coverage，不查unfilled validity、terminal、copy量。 |
| Seamless table/order tests | `KEEP`；played row归属有效。 |
| Seamless live/Stop tests | `MISSING`；不观察live validity、partial retained或source-close安全失败。 |
| Scan writer unit tests | `MISSING`；没有直接测validity、partial seal或增量复杂度。 |

建议增加非wall-clock性能守卫：spy immutable materialization/processor extraction的cell数。N次新增1 cell时处理量应O(N)或受固定display cadence约束，不能N²。

### 6.3 Runtime

| 测试 | 裁决 |
|---|---|
| DatasetCoverage Follow Processor test | `KEEP BUT INSUFFICIENT`；source只 `finish_live()`，没有production `publish_final()` terminal replacement。 |
| frozen Final Processor test | `KEEP`；孤立frozen路径正确。 |
| latest-only tests | `KEEP`；busy coalesce/cancel有价值。 |
| followed slot every update test | `KEEP WITH RACE CASE`；顺序update有效；缺freeze读slot时并发update。 |
| generation lifecycle tests | `KEEP`；EventRef replacement有效；没有snapshot stamp invariant。 |
| builder/live port tests | `PASS AS UNIT, FAIL AS PRODUCT PROOF`；大量细测production零消费者的平行框架。 |

本轮运行的已有精确节点：

~~~text
test_a_finite_run_shows_its_dataset_filling_and_stops_when_asked
test_hosting_a_processor_on_a_finished_signal_derives_once
test_dataset_coverage_processor_follows_every_publication_then_finishes

3 passed in 1.01 s
~~~

三条都绿，同时本报告P0探针稳定失败，说明缺口在测试组合边界。

## 7. 文档—实现矛盾

| 文档说法 | 当前实现 / 证据 | 裁决 |
|---|---|---|
| `ARCHITECTURE_DESIGN.md:287-290`：Camera发布 `frame_0...frame_N` | 当前代码、tests、Implementation Plan W1b均为一个 `frames` + READOUT_EVENT axis | D-004；建议保留当前单signal。 |
| `ARCHITECTURE_DESIGN.md:330`：Stepped完成后只发布一个FINAL scan | 当前Scan每点full live，最终再发Final | 旧句不完整；live是用户当前明确要求。 |
| `ARCHITECTURE_DESIGN.md:157`：finite frozen replay | complete Final可replay；stopped partial不能启动Processor | 实现未满足。 |
| `ARCHITECTURE_DESIGN.md:291-305`：共同assembler按ordinal拒绝gap | 只有Monitor检查；finite/cycle-source只数张数；Pylon ordinal非physical counter | 实现违约。 |
| `ARCHITECTURE_DESIGN.md:167-169`：不新建第二套revision | 当前至少四套；plane允许重复/倒退snapshot ref | 目标未落地；D-018。 |
| Implementation Plan Phase 2：finite FollowTap、completed frozen once | 两条孤立成立；live→Final transition失败 | 计划把两端误当已通链。 |
| zlc_runtime README：small contract package | `plane.py`2095行、`host.py`1157行、`streams.py`1937行、`dataset.py`1809行；三套Processor执行和两套live框架 | “small”不可信。 |
| runtime contract：extent选择finite/frozen/latest | 当前用coverage class猜extent；terminal partial误识别为live | contract缺terminal维度。 |
| zlc_atom README：finite FollowTap或retained Final once | production finite Camera→Occupancy terminal失败 | README是期望，不是现状。 |
| `_CameraLiveSlot` 注释：coverage说明blank未测 | snapshot validity把blank写有效；Occupancy当测量 | 注释与数据事实相反。 |

## 8. 需要用户裁决

本报告不修改共享 `DECISIONS.md`；以下交总审计汇总。

1. **Camera finite live视图（D-006）**
   A. 最新完整cycle，terminal一次完整Dataset；
   B. fixed-geometry growing Dataset，但必须invalid future + 增量处理。
   审计建议Camera选A；Scan仍可用growing grid。

2. **统一API（D-007）**
   A. Host-owned atomic `publish_live(outputs)` / terminal seal；
   B. 保留slot，但slot提交immutable `(token, outputs)`，不能被plane以后pull。
   审计建议A；普通函数/现有Host owner足够，不需manager/registry。

3. **无人使用runtime builder体系（D-011）**
   A. 接入Camera/Scan并删custom slots；
   B. 删除builder/live port/exact preview平行体系，围绕产品写最小commit路径。
   审计倾向B，除非先证明现有delta体系能同时解决Processor delta和terminal seal。

4. **Stopped partial terminal**
   是否明确要求固定authored schema、unfilled invalid、retained/non-transient、可Save、可one-shot Processor？审计建议全部是。

5. **Processor live精度**
   finite exact source的Processor必须every-commit，还是允许display-latest？Occupancy/feedback建议exact；纯preview derivation可latest。应由input contract声明，不能从coverage猜。

6. **Generation/revision（D-018）**
   建议EventRef为causal identity，snapshot revision为content identity，plane强制各自不变量；删除跨generation全局revision要求。

7. **Pylon same-shot保证**
   若SDK可读image/block/skipped counter必须使用；若不可读，是否接受多帧cycle只是“按收到顺序推定”而非“已证明same-shot”？不能文档声明绝对保证、实现静默降级。

## 9. 推荐修复顺序（本轮未实施）

1. 先冻结 terminal/partial/coverage/validity 语义并裁决D-006/D-007/D-018。
2. 加四条P0 old-red：Stop是否freeze一致、finite live→Occupancy terminal、partial validity、finite ordinal gap。
3. 修runtime lifecycle dispatch：以plane state判断live/terminal，不用coverage class；terminal partial one-shot可用。
4. Camera只保留一个accumulator/seal truth；删除padded `CameraFrameRecord(blank)` 和dead slot revision。
5. finite/monitor/CameraCycleSource复用一个ordinal assembler；补Pylon physical counter或明确降级。
6. Occupancy传播parent validity/coverage并只处理新commit；不得先做UI/overlay补丁。
7. Scan live改为delta/exact与display cadence分离；terminal只freeze一次；cleanup确保safe。
8. 收敛runtime live API，删除未选中的平行体系。
9. 最后再跑真实multi-panel/fit/overlay性能验收；否则renderer优化会掩盖上游仍复制数GiB。

## 10. 探针边界

- 全部进程先导入并打印当前checkout的 `zou_lab_control_v2`、`zlc_atom`、`zlc_runtime` 路径。
- 未启动真实设备、未打开真机SDK、未改workspace data。
- 时间只确认增长阶与相对放大，不作为跨机器硬阈值。
- Pylon physical counter结论来自当前adapter源码；未连接真实Basler。SDK具体可用哪个counter留待真机/官方API阶段确认，但当前代码确实未使用任何counter。
