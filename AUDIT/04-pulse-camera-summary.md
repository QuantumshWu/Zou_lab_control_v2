# 04 — Pulse、Sequencer、Camera与same-shot综合结论

状态：阶段完成。详细证据见`04a`、`04b`和`04c`。

## 1. 当前核心问题

当前Pulse/Camera链同时混用了以下概念：

- timeline内部`RepeatRegion`；
- whole-pulse有限次数；
- `fire(forever)`；
- shots per point；
- scan table sweeps；
- plan repeats；
- camera repeat/cycles；
- SLM candidate chunks。

这些数量由不同节点分别相乘、覆盖或忽略，没有一个冻结的execution truth。同时Camera多帧分组只依据“收到的frame顺序”，而pulse window count、camera busy interval与physical trigger identity没有共同preflight。

交叉复核说明：`04a`把`CompiledProgram.camera_window_count/exposures`当前方法判为可删便利面，`04b`把同一能力判为必须保留使用。统一结论是：**compiled schedule的trigger-window projection必须保留并由Task preflight消费；唯一算法owner是`zlc_pulse.schedule.trigger_windows()`。** 两个camera-named方法可以只是薄投影或删除，不能另写第二套count算法；“保留能力”不等于必须保留当前方法名。

## 2. 无需用户裁决的P0错误

### PC-001 — Delay FIFO可静默丢真实波形

Host没有校验compiled TTL/DAC delayed event在任意delay window内的peak in-flight数量是否超过connected geometry的FIFO depth。

构造130个每tick翻转、delay=200 ticks、FIFO depth=64的合法序列：compile和pack均成功，RTL mirror相对理想delay line有33 ticks错误。实板可成功LOAD但静默丢toggle。

Compiler/load边界必须在寄存器写入前完成delay event capacity验证。

### PC-002 — Device load没有绑定正确board事实

当前`PulseStreamer.load()`不验证：

- `program.target_abi_fingerprint == connected target ABI`；
- `program.clock_hz == connected clock`；
- program使用的scan coefficient fraction/geometry与connected geometry相同。

探针确认三类mismatch均可成功load。相反，部分node用`resolved.target != board.target`全对象比较，修改无关display label也会被误拒。

唯一owner应是device load；node-local target比较删除。

### PC-003 — Delay scan slot是假功能

Model允许构造并上传delay `PulseSlot`，但compiler全部tick coefficient为零，scan row改变不会改变trigger timing。当前硬件只支持delay作为host-resolved API parameter，应在PulseSequence构造时拒绝delay scan slot。

### PC-004 — Count silent coercion与32-bit wrap

`sweeps=0/-3/1.9/True`均静默变1；大于32-bit的RepeatRegion count和scan count在host/run record保留大值、wire寄存器却wrap成小值。所有public count必须拒绝bool、非整数、`<1`及超硬件范围。

### PC-005 — Temperature默认真实时序不成立

默认Temperature pulse两次camera trigger相隔5.02ms；任务继承Calibration的20ms sensor integration。第二edge在第一帧结束前约15ms到达，第一帧还会覆盖两段probe。Virtual不模拟camera busy，现有Temperature端到端测试因此false green。

Temperature protocol必须在用户选择20ms+recapture gap、5ms sensor或专用双帧Calibration后重定；当前配置不能宣称真机可用。

### PC-006 — Calibration可调参数会破坏trigger cadence

Calibration把reference duration同时当sensor exposure和两个long pulse periods，readout控制middle period，但固定gaps不联动。默认20/5ms成立；改31/6ms后trigger intervals为31.1ms与21.1ms，第三edge可能被忽略。

Task必须在fire前用actual CameraWorkingPoint与compiled schedule验证window count、间隔、integration offset和事件角色。

### PC-007 — finite cycle grouping不检查ordinal gap

Monitor检查reported ordinal连续；FiniteCapture只数帧。两帧cycle收到ordinal`(0,2)`仍被接受。Temperature/Feedback的CameraCycleSource复用finite path。

Finite、Monitor、CameraCycleSource必须使用同一cycle assembler，并只声称adapter证据能证明的强度。

### PC-008 — Camera window count与frames-per-cycle可跨shot错组仍“exact success”

每物理cycle四个camera edges、request配置三帧时，两个cycles共8个edges会被封顶取前6张并分成：

```text
[0,1,2], [3,4,5]
```

第二组跨shot，但terminal四项均可为真。Task同时拥有pulse和camera时必须逐row、逐internal repeat核window count/order；普通camera-only monitor不能凭空做此保证。

### PC-009 — Dynamic device tuning绕过DeviceUseCoordinator

Stepped descriptor只claim sequencer EXCLUSIVE，却接收installation全部`tunable_devices`并可直接`tune(camera exposure等)`。测试甚至允许Camera Measurement独占camera时由Stepped无claim修改同一camera。

Start必须从ScanPlan解析dynamic claims；若被调device同时是source producer owner，应拒绝或取得明确协调，不能把`tunable_devices`字典当授权。

## 3. Real/Virtual与性能问题

### PC-010 — Virtual只延迟DONE，物理事件在fire内瞬间爆发

五个逻辑60.3ms cycles在约微秒量级全部进入world/camera，之后只等约301.5ms DONE。它不模拟camera busy、buffer pressure或Stop；大table还在`fire()`调用线程同步渲染全部frames。

Virtual必须按compiled event time逐row推进并可中断；极速离线模式若需要，必须显式独立，不能作为正式virtual acceptance。

### PC-011 — finite driver buffer等于整次run

Calibration默认600 frames，2048² uint16约4.69GiB raw driver容量；Temperature默认320 frames；SLM每chunk最多384 frames。Reader本来逐cycle消费，不需要默认保留全run。应使用有界ring/chunk并在overrun loud fail，预算由真实qCMOS尺寸/cadence决定。

### PC-012 — 多重全帧copy与metadata丢失

Pylon Array copy、CameraFrameRecord bytes copy、两层stack、OwnedSnapshot immutable copy叠加；DCAM stamps/timestamps/buffer index最终全部在Dataset边界丢失。至少要保存用于复核cycle grouping的canonical capture evidence。

### PC-013 — SLM每chunk重复load同一program

SLM Task只resolve/compile一次，但每个chunk仍safe→load identical program→write table→fire。Task独占sequencer且program不变，应在Task开始load一次，chunk只更新cycle count/table并fire。

### PC-014 — Scan科学坐标不是实际播放值

Off-grid authored values在wire crossing被round，Dataset/Temperature curve仍保存原float。探针中`0.004001ms`实际播放`0.004000ms`。科学坐标应使用wire-decoded actual，metadata同时保存authored intent。

## 4. 统一术语与推荐execution链

| 术语 | 唯一含义 |
|---|---|
| Internal repeat | Pulse timeline内部`RepeatRegion` |
| Cycle | PulseSequence start到end的一次outer执行 |
| Frames per cycle | 一个cycle的camera tuple大小 |
| Shots per point | 同一个plan row的独立cycles数 |
| Sweep | 完整plan走一遍 |
| Candidate shots | 某SLM candidate下cycles数 |
| Chunk | 有界camera/buffer拆分，不改变science geometry |

推荐finite执行顺序：

```text
resolve resource + API values
materialize canonical actual values
compile once per distinct program
sequencer.safe()
settle/tune only claimed devices
sequencer.load(program)        # 唯一ABI/clock/geometry/capacity gate
write rows + explicit finite cycle count
preflight pulse windows against actual CameraWorkingPoint
arm exact source for cycles * frames_per_cycle
fire(forever=False)
drain exact cycles through one assembler
wait_done in cancellable slices; reject fault/tail incomplete
finally:
    sequencer.safe()
    source.close()
```

节点映射：

- Calibration：finite cycle count=Samples，不再forever+camera截停；
- Stepped：每point cycle count=Shots/point，不改PulseSequence repeat；
- Seamless：rows按point/shot展开，sweeps=plan repeats；
- Temperature：Seamless、S=1；
- SLM：zero-slot cycle count=chunk、program只load一次；
- Camera Measurement：仍不拥有pulse，正确。

## 5. Pulse model/device裁决

### 保留

- PulseTarget/Port、Period、FieldRef、ApiParameter、AnalogStep；
- internal RepeatRegion结构本身；
- pure schedule、scan wire conversion；
- compiler核心IR；
- PulseStreamer transport/safe/load/fire/wait/refill骨架；
- DCAM owner lane与ring overwrite proof；
- CameraAdapter/WorkingPoint/FrameRecord核心。

### Redesign

- PulseSequence binding invariant、clock/count限制；
- compile delay capacity；
- CompiledProgram删除embedded rows/sweeps/forever test path；
- schedule显式接rows/sweeps并包含delay tail；
- PulseStreamer load严格硬件事实；
- AppliedState包含rows/sweeps/forever/actual form；
- PulseEditor Sync要么完整同步executable truth，要么降级只读摘要；
- Camera arm mode/cardinality/buffer contract；
- terminal proof跨adapter统一含义。

### 删除/合并候选

- dead `CameraCaptureSpec`或让它成为唯一arm spec，不能继续平行；
- `CompiledProgram.scan_points/scan_point_durations/repeat_forever`等无production producer字段；
- test-only`write_slots`若held-point产品不使用；
- compiler重复scan conversion死分支；
- `wire.py`内FPGA build/resource/CLI职责移入已有`fpga.py`；
- `arm_sequencer`一行跨plugin alias与dead ResolvedPulse metadata；
- Camera stale advertised `working_point` capability；
- engine_model中无测试/产品消费者的RTL mirror家族。

## 6. Same-shot保证边界

Runtime EventRef/lineage能证明“软件publication由哪份parent产生”，不能证明camera每个TTL edge属于哪一个物理shot。

- DCAM ordinal是copied transfer index；
- Pylon ordinal是successful retrieve index；
- Virtual ordinal是它自己接受的trigger index且不模拟busy。

如果真实SDK没有cycle marker/accepted-trigger counter，产品只能选择：

1. 每cycle单独arm/fire/terminal，牺牲性能换强证明；
2. 连续stream按顺序推定，并在UI/文档明确为best-effort，不再写绝对same-shot。

无论选择哪一个，Task-local pulse preflight都必须先完成。

## 7. 文档与测试裁决

- 当前50个直接camera/cadence tests和Temperature纵向仍全部绿，但没有覆盖camera busy、physical ordinal、window count、Virtual arrival cadence；它们不是物理验收。
- 现有Stepped tests用fake source按loaded loop count主动产生恰好S条，属于实现自证whole-bracket方案。
- Pulse contract仍写load会检查target ABI，代码没有。
- Engine/RTL注释声称host validator阻止delay FIFO overflow，代码没有。
- Virtual cadence文档声称按logical time，代码只延DONE。
- Temperature链被记录为完成，但默认真实时序无效。

必须增加少量纵向红灯：delay FIFO overflow、load三类mismatch、invalid count、window/cardinality、trigger interval、ordinal gap、Virtual frame arrival cadence、dynamic claims、actual scan coordinates和safe cleanup。

## 8. 待用户裁决

产品语义选项已登记到[DECISIONS.md](DECISIONS.md)。最关键的是whole bracket与shots是否分离、same-shot强度、trigger wiring、Temperature protocol、Camera acquisition mode、Seamless source资格、driver buffer策略、dynamic claims、actual scan coordinates、Applied Sync、delay DONE和offline Calibration claims。
