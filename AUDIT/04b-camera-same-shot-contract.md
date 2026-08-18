# 04b — Camera / trigger / same-shot contract 深审

状态：本子阶段完成
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：Camera contract、DCAM/Pylon/Virtual adapters、record/buffer/capture/grouping、sequencer trigger 与 virtual cadence、Camera Measurement、Calibration、Stepped/Seamless Scan、SLM Feedback、Temperature 的 cycle/shot/repeat 调用及直接测试。
约束：只读源码、Git 历史、既有文档和无硬件隔离探针；未修改 production、tests、旧文档或硬件。

关联报告：

- [03a-measurement-live-publication.md](03a-measurement-live-publication.md) 已证明 finite/monitor assembler 分裂、finite ordinal gap、O(N²) live copy 和 scan cleanup 问题；本文不重复其全部 runtime 细节，而是继续追到真实 adapter、pulse windows 和 virtual apparatus。
- [04a-pulse-api-semantics.md](04a-pulse-api-semantics.md) 审查 pulse/scan/repeat/wire 侧；本文从 camera 所见的 trigger/cycle 事实反向核对。

## 1. 结论先行

当前系统**不能证明“发布的一组 frames 来自同一个物理 shot”**。它能证明的最强事实只有：

1. 一张 `CameraFrameRecord.image` 是 adapter 在某次 read 中复制出的不可变图像；
2. DCAM 能发现已产生 frame 的 ring overwrite，Virtual 能保留自己生成的 trigger ordinal，Pylon 能发现 SDK 明报的 failed grab；
3. Monitor assembler 能检查 adapter **报告的** ordinal 连续且模 `frames_per_cycle` 对齐；
4. 一旦某个 cycle 已经被发布，Runtime 能证明 sibling/derived outputs 来自同一 publication lineage。

它不能证明：

- pulse 每个 logical cycle 实际产生了恰好 `frames_per_cycle` 个 camera edges；
- camera 忙时没有忽略 edge；
- adapter 的 ordinal 是物理 trigger ordinal而不是“软件收到的第几张”；
- 连续 stream 中没有用下一个 shot 的第一张补上上一个 shot 丢失的最后一张；
- generic Scan 的第 N 个 publication 就是 board 播放的第 N 个 row；
- virtual 的 buffer、cadence、Stop 和 trigger-loss 行为等价于真实设备。

这是 `REDESIGN` 级契约问题，不是给某个 Task 再加一个超时即可解决。最危险的当前状态是：**finite total count 有时最终报错，但在报错前已发布错误分组；额外 trigger 的情况下 Pylon 和 Virtual 还能以“exact terminal”成功结束。** `Repeat=0 + frames_per_cycle>1` 没有 terminal count，错位可以永久持续。

## 2. 当前真实链路与平行真相

~~~text
PulseSequence
  -> compile / scan table / repeat
  -> physical camera edges on one logical port
       (Task owns this truth, Camera Measurement normally does not)

CameraMeasurementRequest.frames_per_cycle
  -> adapter.arm(source_group_sizes=...)
  -> read_frame_records()
  -> CameraFrameRecord.source_ordinal
  -> FiniteCapture chunks by count / MonitorCapture chunks by modulo
  -> frames Dataset (record metadata discarded)
  -> Occupancy / Scan / Temperature / Feedback
~~~

并行维护的关键数字至少有：

| 名称 | 当前含义 | owner |
|---|---|---|
| pulse camera-window count | compiled program/table 实际 camera rising edges | `zlc_pulse` |
| `frames_per_cycle` | caller 期望每 cycle 的 frame 数 | Camera request / Task |
| `source_group_sizes` | 同时被用作 finite cardinality validation 与 Pylon arm mode暗号 | adapter SPI |
| `source_ordinal` | DCAM copied index、Pylon retrieved index、Virtual accepted-trigger index | 各 adapter，语义不等价 |
| `produced_count` | DCAM driver transfer count、Pylon retrieved count、Virtual rendered count | terminal record，语义不等价 |
| Scan `shots_per_point` | row在scan table重复或whole-pulse repeat | Scan engine |
| Scan `repeats` | 完整table sweeps | Scan engine / private streamer state |
| source Dataset repeat | `sweeps × shots × source repeat` 展平 | `ScanDatasetWriter` |

系统没有一个边界把前两项核对，也没有保存足够证据把后四项重新关联。`CompiledProgram.camera_window_count()` / `camera_window_exposures()` 仍然存在，但 production capture 不调用。

## 3. Same-shot 保证矩阵

| 路径 | 当前可以证明 | 不能证明 / 结论 |
|---|---|---|
| `Repeat=0, F=1`, Pylon | 每次返回一张 latest free-running frame | 不属于 pulse shot；允许SDK跳帧。应称 live image，不称 same-shot。 |
| `Repeat=0, F=1`, DCAM/Virtual qCMOS | 每个已收到 frame 是一次 accepted capture | arm mode仍是external；与Pylon同一request含义不同。 |
| `Repeat=0, F>1` | reported ordinal连续且起点 `% F == 0` | Pylon/DCAM ordinal不是sent-trigger ordinal；busy/lost edge可让后续shot补位。**不能证明same-shot。** |
| finite Camera Measurement | 最终可比较读取数与部分adapter terminal count | `FiniteCapture.next_cycle()`连reported ordinal都不检查；extra-window在Pylon/Virtual可exact成功。**不能证明。** |
| Calibration shipped default | 默认JSON当前有3个20.1 ms cadence windows | 任意参数修改/custom pulse/internal repeat未preflight；DCAM terminal与forever board stop有race。只能说默认模板在virtual理想条件下自洽。 |
| SLM Feedback shipped default | 每chunk arm exact total，safe先于source close | hard-code frame 1/三帧且不核pulse；extra windows可错误成功。不能推广到用户所选pulse。 |
| Temperature | virtual返回两帧并可形成curve | 默认trigger间隔违反自己的20 ms working point，第一帧还积分两段probe。**真实链当前没有有效证据。** |
| Seamless Scan + private `CameraCycleSource` | source按读取顺序交付 | 继承finite分组缺陷。 |
| Seamless Scan + ordinary LIVE signal | tap按publication顺序交付 | source可free-run、latest/coalesced或每cycle多发；publication ordinal不是board row。**不是by construction。** |
| Stepped `pulse_gated` | arm边界前清旧publication，随后按顺序取N份 | 没有causal id或lossless要求；只是“next event”假设。 |
| Runtime sibling/derived family | 同一publication/EventRef lineage | 只能证明软件publication同源，不能倒推物理shot正确。 |

## 4. 已确认缺陷

### CAM-001（P0）— pulse window count 与 `frames_per_cycle` 是两份无人核对的真相

位置：

- `nodes/camera_measurement/measurement.py:380-470`，尤其 `CameraCycleSource.arm()`；
- `nodes/calibration/pulse.py:45-61`；
- `nodes/calibration/task.py:956-1107`；
- `nodes/slm_feedback/task.py:487-624`；
- `nodes/temperature/task.py:184-219`。

`CameraCycleSource.arm(program, table)` 拿到了唯一同时可见 compiled pulse 与 camera request 的 seam，却明确丢弃二者核对。Git 历史显示 commit `6151872` 删除了原有 `program.camera_window_count(...) == frames_per_cycle` 检查，理由是 camera 不应读取不属于自己的文档。这个 ownership 判断过度了：普通 Camera Measurement 确实不拥有pulse；但 Calibration/Temperature/Feedback Task 明确同时拥有 pulse、sequencer 和 camera，必须在自己的owner内核对物理计划。

后果不只是“少一帧后超时”。隔离探针模拟每物理cycle有4个edges、request却写3帧，两个cycle共8个edges。Virtual finite arm只接受前6个；当前finite path得到：

~~~text
physical cycles    [[0,1,2,3], [4,5,6,7]]
accepted groups    [[0,1,2],   [3,4,5]]
accepted ordinals  [[0,1,2],   [3,4,5]]
terminal           produced_count=6, stopped=True,
                   no_more_frames=True, joined=True
~~~

第二组跨越两个shot，但所有“exact proof”均为真。Pylon `StartGrabbingMax(expected)`具有同样的封顶结构；DCAM反而可能观察到 `count > expected` 后失败。因此同一错误pulse会在adapter之间产生“成功但错组”与“失败”的不同结论。

明确设计结论：

- 普通camera-only monitor不能凭空知道pulse；不要让它宣称绝对same-shot。
- 同时拥有pulse和camera的Task必须在fire前检查**每个实际scan row / internal repeat**的window count与顺序；不能只检查总数。
- 还需要一个camera↔sequencer logical trigger port真相。当前只有Pylon物理`Line1`和virtual sequencer默认`emCCD`，没有installation-owned mapping。

### CAM-002（P0）— `source_ordinal` 不是统一的物理ordinal

位置：

- DCAM `dcam.py:636-691`：`source_ordinal = _copied_count`；
- Pylon `pylon.py:624-680`：`source_ordinal = _grabbed - 1`；
- Virtual `simulation/camera.py:277-357`：使用自己accept的trigger queue ordinal；
- finite `measurement.py:560-613`：完全不读ordinal；
- monitor `measurement.py:665-684`：只核reported ordinal。

DCAM的transfer count能证明“camera已产生多少frames”，ring逻辑能证明已产生的frame没被overwrite；它不能证明一个根本未被camera接受的TTL edge。Pylon甚至没有读取任何SDK frame identity/stamp，只按成功Retrieve次数重新编号。Camera忙而忽略一个edge后，下一shot的第一张仍会得到连续软件ordinal。

因此：

- `MonitorCapture` 的算法对**可靠物理ordinal**是正确的，但当前只有Virtual direct trigger较接近这个前提；
- finite path连这个较弱前提也没使用；
- qCMOS的`frame_stamp/camera_stamp/timestamp`被保存进record但从未检查其语义、单调性或与trigger的关系；
- 是否能从真实DCAM/Pylon取得cycle marker、accepted-trigger counter、可信timestamp pattern，必须查实验机/官方SDK并实测，不能由软件名字推定。

这扩展 [DECISIONS.md](DECISIONS.md) 的 D-014：如果硬件没有cycle evidence，产品必须选择“每cycle单独arm/fire/terminal”的慢而可证路径，或明确把连续多帧称为best-effort ordered frames，不能继续绝对命名same-shot。

### CAM-003（P0）— Temperature 默认pulse在真实qCMOS上时序不成立

位置：

- `nodes/scan/temperature_template.json`；
- `nodes/temperature/task.py:166-219`；
- `devices/simulation/world.py:1041-1077`。

Temperature默认继承Calibration的sensor integration；默认是20 ms。编译后模板的两个camera rising edges为：

~~~text
trigger starts       [20.000 ms, 25.020 ms]
trigger interval       5.020 ms
camera integration    20.000 ms
probe windows         [20.000,25.000] ms and [25.020,30.020] ms
~~~

第二edge在第一帧积分完成前约15 ms到达。DCAM working point已经公开`TIMING_MIN_TRIGGER_INTERVAL`，但没有任何preflight读取它；真实camera很可能忽略第二edge。更严重的是，第一帧的20 ms integration覆盖两段5 ms probe，virtual实际计算出的probe overlap是10 ms；第二帧只有5 ms。Temperature却用同一个Calibration short-readout threshold判断两帧。

隔离probe同时确认：Virtual working point自己声明required interval为20 ms，却仍接收5.02 ms间隔的两张record并返回exact terminal。真实 `test_temperature_chain` 仍然绿色（本审计运行：`1 passed in 16.31s`），所以这是**false-green end-to-end**，不是验收证据。

这需要用户决定科学时序：

1. 保持20 ms sensor integration，在release结束、第二probe之前增加足够的trap-on/recapture等待，使edge间隔满足actual camera interval且第一积分不覆盖第二probe；或
2. Temperature使用约5 ms sensor integration，但必须重新定义它与20 ms sensor / 5 ms probe下训练出的Calibration threshold是否可比；或
3. 使用与Temperature完全相同的双帧protocol另做readout calibration。

无论选哪项，当前默认不能继续宣称真实可运行。

### CAM-004（P0）— Calibration可调参数会破坏 shipped template 的固定cadence

位置：

- `nodes/calibration/logic_node.py:53-101`；
- `nodes/calibration/imaging_template.json`；
- `nodes/calibration/task.py:865-951`。

UI允许自由修改reference/readout“exposure”，Task把reference同时写入sensor integration和两个long pulse periods，把readout写入middle period；但template的`gap_0=0.1 ms`、`gap_1=15.1 ms`保持固定。默认20/5 ms时两次间隔都是20.1 ms；改成31/6 ms后实测为：

~~~text
trigger intervals    [31.1 ms, 21.1 ms]
sensor integration    31.0 ms
~~~

第三edge可能被忽略。现有`test_imaging_template_cadence.py`只检查原始JSON默认值，没有解析AuthoringSchema允许的其他值，也不比较CameraWorkingPoint。

另外，三个API slot由用户按序号选择，但实现不证明这些`PulseFieldRef`实际控制第0/1/2个camera window。`READOUT_FRAME_INDEX=1`始终固定；选错slot会静默把一个period变短，却仍拿middle frame训练readout。

`FrameContract`只正式保存sensor exposure/geometry/readout mode；真正决定threshold分布的probe gate只藏在`report.run_record.pulse.frame_exposures`。Temperature/Occupancy不核它，SLM Feedback也只核sensor working point。这里缺少的不是另一个fingerprint，而是明确回答：Calibration threshold适用条件是否包含pulse-side probe gate/事件角色。

### CAM-005（P0 real/virtual）— Virtual一次fire瞬间播放全部scan rows，且不模拟camera busy

位置：

- `devices/simulation/sequencer.py:59-86,124-169`；
- `devices/simulation/world.py:973-1079`；
- `devices/simulation/camera.py:371-388`。

`VirtualPulseStreamer.fire()`先同步调用`_fire_world()`遍历全部points，之后只用`_logical_deadline`延迟DONE。五个60.3 ms cycles的stub probe结果：

~~~text
logical run duration      301.5 ms
all 5 world callbacks span  ~2.1 us
fire() return               ~0.81 ms
~~~

这使virtual raw buffer在Task开始read前就可能收到整张table；finite adapters又为完整run分配buffer，恰好把burst隐藏。真实板则按物理cadence逐cycle触发，reader、UI、Stop和ring pressure并发发生。

VirtualCamera也不执行自己声明的`required_external_trigger_interval_seconds`：每个world edge都会render/queue；达到finite expected后，后续逐个`trigger(1)`只是静默return。因此virtual无法测试busy-trigger loss，反而会给CAM-001的extra-window错组返回成功。

`safe()`对world thread只join 2秒且不检查仍alive；若worker正在一次长`_fire_world(all points)`，`world.safe()`可能先执行，旧worker之后继续改变world/触发camera，下一次fire还可能与旧worker并存。

结论：virtual physics公式可以保留，但scheduler/trigger delivery必须按logical event time推进并在Stop时可中断；至少必须模拟camera accepted-trigger interval和buffer pressure。延迟DONE不等于模拟cadence。

### CAM-006（P1）— 同一个Camera request在三个adapter上选择不同acquisition mode

`arm(frames=None, source_group_sizes=None)`被Pylon解释成free-running `LatestImageOnly`；DCAM始终external triggered；Virtual模式由device创建时的`free_running`构造参数固定。`source_group_sizes=(F,)`又被Pylon当成“continuous external”暗号。

所以`Repeat=0, Frames per cycle=1`不是一个跨adapter契约：Pylon忽略pulse，qCMOS等待pulse，virtual qCMOS等待pulse，virtual MOT按自身时钟。run record会记录actual mode，但用户在Start前没有显式选择，也不能据同一draft预测行为。

`CameraCaptureSpec`本可承载mode，但它完全没有production consumer，而且不能表达continuous capture；当前SPI反而用group sizes偷渡mode。建议用户决定：

- authoring显式选择`free_running/external_triggered`，不支持者拒绝；或
- 产品明确规定`Repeat=0,F=1`只是“device-native live preview”，承认不同设备不对应同一shot语义。

### CAM-007（P0/P1）— Scan按stream顺序猜board point，且pulse repeat可再乘一层

`SeamlessScanMeasurement`接受任意当前LIVE Dataset signal，只检查generation。它的文档声称“fired cycle drives source, publications ARE rows by construction”，但代码不要求source是pulse-gated、lossless或一cycle一publication。合法候选包括free-running Camera、latest/coalescing Processor和每cycle多publication的producer。

`SteppedScan`至少让用户选择`pulse_gated/sw_gated`，但pulse-gated仍只是清掉旧queue后取接下来的N份publication，没有event↔fire causal identity。`sw_gated`本来就是明确的boundary heuristic，不应称same-shot。

另外，Seamless把`shots_per_point`编码成重复scan rows；若template自己已有whole/partial `RepeatRegion`，一个row可能产生多组camera windows，scan loop仍只消费一个value。Calibration/SLM Feedback也不拒绝internal repeat。相关pulse重复语义由04a和D-013继续裁决。

### CAM-008（P1 lifecycle）— Stop大体可响应，但terminal/safe边界仍不统一

做得正确、应保留：

- finite read按50 ms slices检查cancel，camera timeout仍作为per-frame deadline；
- DCAM finish先置event打断owner-lane read，SDK handle始终在同一lane释放；
- Monitor detach失败仍会执行camera finish；
- SLM Feedback用嵌套`try/finally`保证sequencer safe后再关camera source。

明确问题：

1. Stepped/Seamless的finally顺序是`source.close(); sequencer.safe()`；source close一旦抛错，safe被跳过。Temperature正使用会做terminal proof并能抛错的`CameraCycleSource`。
2. Calibration成功路径先`capture.close()`再`sequencer.safe()`；forever board在两者之间仍可送extra edge。Pylon/Virtual会封顶忽略，DCAM可能观察extra count并使同一run偶发失败。外层异常处理最终会safe，但结果不确定。
3. `CameraCaptureTerminalRecord.no_more_frames/joined`不是共同事实：DCAM stop后无条件true，Pylon不检查SDK queue且joined无worker，Virtual以内部queue是否空决定。跨adapter把四个bool都当“exact proof”没有相同含义。
4. Virtual world join timeout不验证terminal，见CAM-005。

### CAM-009（P1 memory/performance）— finite raw buffer等于整次run，数据随后被多次复制

所有adapter要求finite `buffer_frame_count == expected total`：

- Calibration默认 `200 × 3 = 600` frames；
- Temperature默认 `20 sweeps × 8 points × 2 = 320` frames；
- SLM Feedback按chunk限制为至多 `128 × 3 = 384` frames；
- ordinary finite Camera没有上限。

DCAM的`dcambuf_alloc(total)`和Pylon `MaxNumBuffer=total`直接把这个数交给SDK。若frame为`2048×2048 uint16`，Calibration 600帧仅raw容量约4.69 GiB。reader本来逐cycle并发copy，因此“整次run都留在driver”不是唯一正确实现；它只是用内存避免定义有界ring overrun策略。

复制链也很重：Pylon先`np.array(result.Array, copy=True)`，随后`CameraFrameRecord`再复制到bytes；`frames_snapshot`内外两层`np.stack`后，`OwnedSnapshot`再把mutable stack复制成bytes。Calibration preview在`update()`为验证先stack一次，freeze又stack并freeze一次。完整O(N²) live成本见03a LIVE-004。

### CAM-010（P1 provenance）— record收集的物理metadata在Dataset边界全部丢失

`frames_snapshot()`只stack image。`source_ordinal`、produced count、DCAM stamps、timestamps、buffer index和terminal record都不进入Dataset轴、run record或artifact。Calibration saved samples重读时按文件名重新合成ordinal `ordinal*3 + j`，把“保存顺序”伪装成原采集ordinal。

因此成功artifact事后只能证明“文件里有三张一组”，不能审计原始continuity、driver stamp或terminal evidence。无需把所有driver字段提升成公共科学axis，但至少应保存足以复核cycle grouping的canonical capture evidence；无法解释的vendor字段不应收集后立刻丢弃。

### CAM-011（P1 cleanup）— dead / duplicated camera contract surface

- `CameraCaptureSpec`：production零消费者；与真实`arm(...)`签名平行且不能表达continuous。`DELETE`或直接改成唯一arm spec，不能继续摆设。
- `camera.working_point` capability：binding时保存一次会随ROI/exposure/tune立刻stale，production零消费者；真实owner是adapter的`working_point()`。`DELETE advertised capability`。
- `DcamCameraAdapter.observed_produced_count()`：production零消费者，仅测试/诊断；若不成为正式proof应删除。
- 三个adapter重复验证frames/groups/buffer/timeout；`source_group_sizes`又不参与adapter grouping。应由一个明确arm contract说mode/cardinality，cycle assembler留在Camera Measurement。
- `CameraAdapter` Protocol反而不要求`close()`，binding只能`getattr`；共享device lifecycle应正式要求close。
- `CameraWorkingPoint.acquisition_mode`注解为`str`，Pylon返回Enum，DCAM/Virtual返回string。应统一。

## 5. Real / Virtual 差异清单

| 事实 | DCAM | Pylon | Virtual |
|---|---|---|---|
| mode选择 | 始终external | arm参数隐式切free/external | 构造时固定 |
| min trigger interval | 读SDK `TIMING_MIN_TRIGGER_INTERVAL` | 直接写成`exposure`，不是SDK readback | 直接写成`exposure` |
| integration start offset | 读SDK | 固定0 | 固定0且world不消费WorkingPoint offset |
| frame ordinal | copied transfer index | successful retrieve index | accepted virtual trigger index |
| acquired-frame overrun | transfer count/ring可证明 | MaxNumBuffer已设置，但不读skip/frame identity | deque丢旧帧并保留virtual ordinal |
| missing/busy edge | 总数不足时最终可能发现 | 总数不足时最终发现；cycle边界前不可见 | **从不模拟busy，全部接受** |
| extra pulse edges | capture继续，可能`>expected`失败 | `StartGrabbingMax`封顶 | 达expected后静默忽略 |
| scan cadence | 真实board时间 | 真实board时间 | fire内burst全部rows，只延迟DONE |
| terminal `produced_count` | driver produced | app retrieved | rendered/queued produced |

根README“Virtual and physical cameras differ only below adapter boundary”不成立；差异恰好穿透到Task科学结果。

## 6. 逐文件 / 类 / 函数裁决

### 6.1 Camera contract 与 adapters

| 文件 / 符号 | 裁决 | 说明 |
|---|---|---|
| `camera/contract.py::_pair` | `PASS` | 小而直接的geometry validation。 |
| `CameraAcquisitionMode` | `PASS WITH DEBT` | 必要词汇；必须跨adapter统一使用。 |
| `CameraCaptureSpec` | `DELETE / REDESIGN` | 死production DTO；若保留必须成为唯一arm输入并支持continuous。 |
| `CameraWorkingPoint` / `__post_init__` | `KEEP` | sensor geometry/exposure/interval/offset是真实必要事实；当前关键timing字段无人消费。 |
| `CameraFrameRecord` / `__post_init__` | `KEEP + REDESIGN ordinal` | immutable ownership正确；`source_ordinal`命名过度，metadata没有下游。 |
| `CameraCaptureTerminalRecord` | `REDESIGN` | 四字段在三个adapter含义不同。 |
| `CameraAdapter` | `KEEP + REDESIGN arm/close` | 唯一共同SPI正确；mode被groups偷渡且缺close。 |
| `_owner_lane.py::CameraSdkOwnerLane` 全部 | `PASS` | owner affinity、bounded close、reentrant call均有真实DCAM消费者。 |
| `_dcam_driver.py` ctypes structs / `_checked` / driver/device wrappers | `PASS WITH DEBT` | 正确位于hardware leaf；process-global init缺并发guard但非本链主因。 |
| `DcamCameraConfig`, `_snap_roi_axis`, settings/readback | `PASS` | ROI覆盖式snapping与actual readback职责准确。 |
| DCAM `_finite_groups`, `arm` | `REDESIGN` | group validation重复；full-run driver allocation不可扩展。 |
| DCAM `_observe_transfer_on_owner`, `_drain_snapshot_on_owner`, `_read_on_owner` | `KEEP` | 对已产生frame的count/ring proof扎实；不能升格为sent-trigger proof。 |
| DCAM `_finish_on_owner`, `close` | `PASS WITH DEBT` | failure cleanup认真；terminal flags仍需统一语义。 |
| DCAM `observed_produced_count` | `DELETE or formalize` | production零消费者。 |
| `PylonCameraConfig`, open/ROI/exposure/gain/trigger setup | `PASS WITH DEBT` | leaf职责正确；min interval当前为猜值。 |
| Pylon `arm` | `REDESIGN` | `source_group_sizes`暗定mode、finite full buffer。 |
| Pylon `read_frame_records` | `REDESIGN` | 不读frame identity，且Array发生可避免的第二次copy。 |
| Pylon `finish_record_capture` | `PASS WITH DEBT` | stop/restore强；produced/no-more只是retrieval侧事实。 |
| `simulation/camera.py::VirtualCamera` settings/tune | `PASS` | virtual leaf归属正确。 |
| Virtual `arm/_produce/read` | `PASS WITH CRITICAL DEBT` | bounded queue与immutable record可留；模式固定、busy缺失。 |
| Virtual `trigger/finish/close` | `REDESIGN` | extra edge静默忽略、join proof不足。 |
| `camera/binding.py::bind_camera` | `KEEP` | 单一binding正确；stale `camera.working_point` capability应删。 |
| real/simulation `device_types.py` camera factories | `KEEP + FIX contract` | named instances与world ownership正确；缺camera↔trigger-port mapping。 |

### 6.2 Capture / grouping / Dataset

| 符号 | 裁决 | 说明 |
|---|---|---|
| `_frame_point_column`, pixel-axis helpers | `PASS` | READOUT_EVENT与sensor coordinates owner准确。 |
| `frames_snapshot` | `KEEP + REDESIGN materialization/provenance` | 一个frames signal合理；多次copy且丢record evidence。 |
| `CameraMeasurementRequest` | `KEEP + USER DECISION` | camera/exposure/ROI/repeat/F必要；acquisition mode与trigger semantics缺失。 |
| `_CameraLiveSlot` | `REDESIGN` | future-valid/O(N²)/dirty race详见03a。 |
| `CameraCycleSource.open` | `PASS` | 总cycle cardinality核对有效。 |
| `CameraCycleSource.arm` | `REDESIGN` | 正是pulse-camera seam，却删除window/cadence检查。 |
| `CameraCycleSource.next_value` | `PASS WITH DEBT` | 私有per-cycle value必要；证据被snapshot丢弃。 |
| `CameraCycleSource.close` | `KEEP + FIX cleanup contract` | terminal compare必要，但proof字段不统一且可阻断safe。 |
| `MeasurementResult` / `frames` | `PASS` | direct finite API的合理结果。 |
| `FiniteCapture.collect` | `REDESIGN live/terminal` | partial/final问题见03a；cleanup基本正确。 |
| `FiniteCapture.next_cycle` | `REDESIGN` | 只按数量切组，是直接same-shot缺口。 |
| `FiniteCapture.close` | `PASS WITH DEBT` | 二次close返回伪造零terminal，不应当作原proof。 |
| `MonitorCapture.poll/_accept_record` | `KEEP + LOWER CLAIM` | 对reported ordinal算法正确；不能补adapter物理证据。 |
| `MonitorCapture.close` | `PASS` | detach失败仍disarm。 |
| `CameraMeasurementNode._configure/read_records/_freeze_working_point` | `PASS WITH DEBT` | 唯一转换/working-point路径正确；run record不含capture evidence。 |
| `prepare/monitor` | `REDESIGN arm spec/buffer` | finite全量buffer、mode隐式。 |
| `execute/_publish_finite` | `REDESIGN` | live/partial/follower问题见03a。 |
| `calibration/outputs.py::_cycle_array/cycle_snapshot/preview slot` | `KEEP + PERF FIX` | plugin-local preview owner正确；validation stack重复且metadata丢失。 |

### 6.3 Sequencer / Scan / Tasks

| 文件 / 符号 | 裁决 | 说明 |
|---|---|---|
| `zlc_pulse.schedule::trigger_times/trigger_windows/run_duration_seconds` | `PASS` | 已有的唯一compiled timing truth；Task没有使用是上层错误。 |
| `CompiledProgram.camera_window_count/exposures` | `KEEP AND USE` | 现成preflight能力，历史删除consumer不等于能力无用。 |
| `simulation/sequencer.py::VirtualPulseStreamer.fire/_fire_world/_repeat_world/safe` | `REDESIGN` | burst rows、Stop/join问题；只延迟DONE。 |
| `SimulationWorld.fire` | `KEEP PHYSICS + REDESIGN trigger delivery` | compiled program物理投影owner正确；应尊重camera interval/offset和logical time。 |
| `scan/source.py::PublishedSignalSource` | `KEEP` | follow generation/tap实现必要；它只承诺next publication，不能被scan包装成shot id。 |
| `wait_for_board` | `PASS WITH DEBT` | cancellable wait正确；丢弃terminal cursor/report evidence。 |
| `SeamlessScanMeasurement._streamed_sequence/_wire_table` | `KEEP + pulse fixes from 04a` | board-driven table方向正确；repeat/row count仍有平行语义。 |
| `SeamlessScanMeasurement.acquire` | `REDESIGN source contract/cleanup` | arbitrary LIVE order冒充row identity，source-close可跳过safe。 |
| `SteppedScanMeasurement._apply` | `USER DECISION` | whole-pulse RepeatRegion兼作shots，见D-013/04a。 |
| Stepped `_collect/_capture_shot` | `KEEP + LOWER CLAIM` | sw-gated是明确heuristic；pulse-gated需lossless/causal source contract。 |
| `ScanDatasetWriter` repeat flattening | `USER DECISION` | 数值不丢但sweep/shot/source repeat只靠run record反解；live output还不带该record。 |
| `CalibrationRequest` exposure/slot fields | `REDESIGN wording/validation` | sensor integration、probe gate、event role混在“exposure”。 |
| `calibration/pulse.py::_validate_calibration_sequence` | `REDESIGN` | 现在只拒scan slots；不验证camera port/windows/repeat/roles。 |
| `resolve_pulse/arm_sequencer` | `KEEP` | 单一路径编译/load正确；caller必须补preflight。 |
| `CalibrationTask._driven_*`, `_pulse_facts` | `KEEP + VERIFY role mapping` | 单一API值转换正确；slot与camera event的对应未证。 |
| `CalibrationTask._capture` | `REDESIGN preflight/end ordering/buffer` | shipped happy path可用；custom/changed pulse和forever terminal不可靠。 |
| `SampleWriter/read_saved_samples` | `KEEP + provenance fix` | 保存/重放有产品价值；当前合成ordinal。 |
| `SlmFeedbackTask._measure` | `PASS WITH CRITICAL FIX` | chunk/Welford/safe顺序良好；hard-coded3/frame1且无pulse contract。 |
| `TemperatureTask._judge/_survival/_pooled` | `PASS CONDITIONALLY` | pairing数学本身正确；输入cycle目前不能证明且默认timing无效。 |
| `TemperatureTask.execute` | `KEEP + terminal fixes` | live/final直接；依赖unsafe Seamless cleanup。 |
| `temperature_template.json` | `REDESIGN` | 5.02 ms trigger spacing与20 ms sensor不相容。 |
| `imaging_template.json` | `PASS only at default` | 默认cadence自洽；API duration变化不联动gaps。 |

## 7. 测试审查

本轮运行直接adapter/cadence/sequencer测试：`50 passed in 1.69s`。它们全绿但未守住上述物理契约。

| 测试 | 裁决 |
|---|---|
| `test_dcam_camera_adapter.py` | `KEEP`：owner lane、ring overwrite、finish recovery质量高；缺sent-trigger loss、stamp语义和大run buffer budget。 |
| `test_pylon_camera.py` | `KEEP + RENAME/EXTEND`：mode/rollback/failed grab有效；“one trigger, one frame, one shot strictly”测试只预塞frame queue，没有trigger identity，结论超出证据。 |
| `test_monitor_and_installation.py::repeat_zero...complete_cycle` | `KEEP`：证明Virtual visible ordinal gap/resync；不能代表真实busy edge。 |
| `test_hosted_nodes.py` finite/live/Stop | `KEEP`：产品worker和50 ms cancel有效；pulse与F始终手工匹配。 |
| `test_imaging_template_cadence.py` | `KEEP + PARAMETRIZE`：只守default JSON；必须覆盖schema允许的duration和actual min interval/offset。 |
| `test_virtual_physics.py::virtual_pulse_fire...` | `REDESIGN CLAIM`：只证明DONE被延迟；没有断言各row trigger的wall cadence。 |
| `test_virtual_physics.py::zero_slot_scan_sweeps...` | `KEEP physics, ADD cadence`：occupancy grouping正确，但全部sweeps在fire内burst。 |
| `test_slm_feedback_task.py::measurement_streams_bounded...` | `REPLACE fake proof`：fake sequencer直接执行`camera.trigger(sweeps*3)`，把应验证的3写进测试double，无法发现pulse mismatch。 |
| `test_temperature_chain.py` | `KEEP outputs, REJECT hardware acceptance`：真实descriptor/host/artifact链有价值；virtual不模拟busy且template时序无效，不能叫真实相机验收。 |
| Stepped/Seamless scan tests | `KEEP + adversarial source tests`：当前fake source恰好每row发一次；缺free-run、coalesced、extra/missing publication和source-close failure。 |

最少应有的纵向红灯：

1. 2/3/4 camera windows与`F=3`分别在fire前接受/拒绝；含scan rows和internal repeat。
2. finite/monitor/cycle-source共同assembler对gap、duplicate、out-of-order一致拒绝。
3. 一个lost edge后下个shot frames不得在live preview、Temperature或artifact中形成cycle。
4. Pylon/DCAM真实metadata qualification；拿不到证据时测试必须断言“保证降级”，而非伪ordinal。
5. CameraWorkingPoint min interval/offset对每个trigger schedule做preflight。
6. Temperature默认两个readout都与Calibration short-frame条件一致。
7. Virtual每row按logical cadence交付、Stop中断且buffer pressure与真实策略一致。
8. source.close抛错时sequencer仍safe。
9. Seamless拒绝或正确处理free-running/latest/coalesced source。
10. 有限capture在合理ROI/frame size下的driver buffer memory预算。

## 8. 文档 / 实现矛盾

| 文档声明 | 当前事实 |
|---|---|
| Architecture 8.2：adapter不得丢帧后重编号成无缺口序列 | DCAM用copied index，Pylon用retrieved index；都不是sent-trigger ordinal。 |
| Architecture 8.2：连续多帧不会跨shot混组 | 该节自己又承认没有可靠physical shot id；external ordered stream不足以证明。 |
| Architecture 8.2：`frame_0...N` sibling outputs | 当前W1代码是单一`frames + READOUT_EVENT`。见D-004。 |
| Architecture 3.3：Virtual finite fire尊重logical duration/cadence | Virtual在fire内burst所有rows，只延迟DONE。 |
| Temperature code comment：非2个probe windows会在arm时拒绝 | `CameraCycleSource.arm()`明确不检查program。 |
| Seamless doc：publication与played row相等是“by construction” | 任意LIVE signal都能绑定，无gating/lossless contract。 |
| Implementation Plan：`F>1` pressure下只发完整same-shot | 只对Virtual reported ordinal gap成立；finite不检查，Pylon无物理ordinal。 |
| Package README：virtual/physical只在adapter以下不同 | cadence、busy、mode、extra trigger和terminal语义均穿透Task。 |
| Temperature e2e被记录为完成科学链 | 测试在物理不可能的20 ms / 5.02 ms组合下绿色。 |

这些文档都不能作为“问题已解决”的证据。

## 9. 推荐的最小目标路径（设计，不实施）

不建议先造新的通用manager/coordinator。现有owner已经足够：

1. **Task-local preflight**：Calibration/Temperature/Feedback在自己已有方法中，用compiled program/table + actual `CameraWorkingPoint`核window count、event role、最小间隔和integration offset；失败发生在fire前。
2. **trigger wiring单一真相**：用户裁决固定logical port（例如`emCCD`）还是installation保存camera↔sequencer port mapping。不能让Virtual私有默认代替真实apparatus配置。
3. **一个cycle assembler**：Finite、Monitor、CameraCycleSource复用Camera Measurement中的同一普通函数；它只能根据adapter evidence给出对应强度的结论，不能把retrieval index包装成physical proof。
4. **finite scientific publish先seal**：没有hardware cycle marker时，Calibration/Temperature等finite Task可以live显示“provisional latest cycle”，但只有terminal exact + pulse preflight成功后形成最终科学artifact；错误run不得留下看似accepted的pair。
5. **infinite多帧诚实降级**：若实验机无法提供cycle marker，`Repeat=0,F>1`要么限制为明确受控的Task/cycle arm，要么UI/文档称“ordered groups (unproven same-shot)”。
6. **修正两个template**：Temperature先定科学时序；Calibration可调duration要么联动derived gaps，要么在preflight拒绝。不要让固定gap和可调exposure平行维护。
7. **Virtual按事件时间推进**：row/cycle逐个调度、camera busy拒绝、Stop可中断；再用同一纵向测试比较真实host+memory transport与virtual。
8. **有界finite buffer**：根据reader实际并发能力选择固定cycles ring并在overrun loud fail，或明确按chunk fire；不要默认把整次run塞进driver。
9. **保存最小capture evidence**：run record/artifact保存用于接受cycle的ordinal/stamp规则、terminal count和pulse window facts；不必保存无解释的全部vendor字段。

## 10. 交主线程登记的用户裁决

以下不是可以由审计者擅自决定的产品语义：

1. **D-014扩展：same-shot强度**——必须绝对硬件证明，还是允许连续stream best-effort；若绝对证明且SDK无marker，是否接受per-cycle arm/fire的性能代价。
2. **Trigger wiring authority**——固定canonical logical port，还是每个camera/sequencer pair在installation中声明mapping。
3. **Camera live mode**——`Repeat=0,F=1`显式选择free/external，还是device-native preview并承认跨adapter语义不同。
4. **Temperature protocol**——20 ms sensor + recapture gap、5 ms sensor、或专用双帧Calibration三者选择。
5. **Calibration API**——三个slot继续按序号自由选但必须验证其camera-event role；以及gap由pulse作者显式维护还是由Task派生。
6. **Seamless source资格**——只允许声明为pulse-gated/lossless的source，还是增加free-running采样语义；当前“任意LIVE”不可保留原绝对声明。
7. **finite buffer策略**——完整run driver buffer，或有界ring/chunk并以overrun失败；需要按真实qCMOS frame size/cadence定预算。
8. **capture provenance**——是否要求保存足以事后复核cycle grouping的metadata；审计建议要求。
9. **D-013扩展**——template internal repeat、shots per point、scan sweeps和Camera repeat是否继续叠加，还是在Task边界明确互斥/翻译。

## 11. 最终裁决

- CameraAdapter这一层**有存在必要且层级正确**；DCAM owner-lane/ring proof、Pylon配置/rollback、Virtual physics均有可保留核心。
- `Camera Measurement -> common cycle assembler`也是正确owner，但当前finite/monitor分裂且adapter evidence不满足它宣称的强度。
- Calibration/SLM/Temperature复用Camera Measurement是正确方向；错误在于为了避免跨owner读取pulse，连同时拥有两边的Task也不做preflight。
- Temperature当前默认真实采集链判定为`FAIL / REDESIGN`；virtual绿色不能覆盖。
- Generic Seamless Scan当前“row identity by publication order”判定为`REDESIGN`。
- `CameraCaptureSpec`、stale `camera.working_point` capability、test-only produced-count surface属于高置信删/合并候选。
- 在用户完成上述裁决前，任何文档或UI都不应再使用无条件“same-shot guaranteed”措辞。
