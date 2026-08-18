# 04c — Measurement / Task 对 Pulse API 的调用深审

状态：本子阶段完成。
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：Calibration、Camera Measurement、Stepped Scan、Seamless Scan、Temperature、SLM Feedback 的 pulse load/resolve/compile/table/fire/wait/safe，以及 shots/repeats/chunks/camera cycles 映射。
约束：只读源码、Git历史、tests与无硬件探针；只新增本报告，未修改代码、硬件或其他文档。

## 1. 结论先行

六种节点没有一套共同的“执行 N 个完整 pulse cycles”语义。同一物理意图至少有四种实现：

1. Calibration：`fire(forever=True)`，相机读够 R cycles 后 `safe()` 截停；
2. Stepped：把 `shots_per_point` 写成覆盖全pulse的 `RepeatRegion`；
3. Seamless / Temperature：重复scan rows，再用table sweeps表达计划重扫；
4. SLM Feedback：写一行零列empty scan table，以 `sweeps=chunk` 表达shots。

它们并不等价：

- 模板自身RepeatRegion有的保留、有的覆盖、有的删除；
- Camera cycle数有时由camera arm、有时由table count、有时由timeline loop决定；
- Calibration没有DONE/fault terminal proof；
- Stepped动态device port绕过 `DeviceUseCoordinator`；
- Seamless对任意LIVE signal声称“由fired pulse驱动”，实现没有因果证明；
- target compatibility在Calibration/Stepped过严、Seamless/Temperature缺失，device load又不校验ABI；
- scan/temperature artifact保存authored float，而板子实播tick-quantized value；
- virtual finite table在 `fire()` 内同步生成全部physics/triggers，`wait_done()`只延后DONE。

明显统一路径不是增加coordinator/manager，而是：

1. target ABI在sequencer `load`边界统一校验；
2. authored timeline RepeatRegion与measurement-level cycle count分开；
3. finite measurements统一为explicit cycle/table count与 `fire -> cancellable wait_done -> safe`；
4. camera/source arm count、table count、dataset count在Start前由一处算出并核对；
5. actual quantized rows、sweeps、pulse path/values进入run record；
6. device ports进入真实device claims，不能经 `tunable_devices`旁路。

## 2. 六节点调用总表

### 2.1 Resolve / compile / target

| Node | API parameter处理 | RepeatRegion处理 | target检查 | compile次数 |
|---|---|---|---|---:|
| Camera Measurement | 无pulse | 无 | 无 | 0 |
| Calibration | 取全部authored值，再按三个1-based slot覆盖duration | 原样保留 | target全对象比较 | 每run 1 |
| Stepped | 每point烘焙全部authored API，再覆盖row params | S=1删除whole；S>1新增/覆盖whole；partial+S>1拒绝 | target全对象比较 | R×P |
| Seamless | scanned API转PulseSlot；其他API用authored值 | 原样保留 | **无** | 每run 1 |
| Temperature | 复用Seamless，只扫 `t_off` | 原样保留 | **无** | 每run 1 |
| SLM Feedback | 使用当前pulse文件authored API values | 原样保留 | 经Calibration helper全对象比较 | 每Task 1 |

R=plan sweeps，P=plan points，S=shots per point。

### 2.2 Device calls

| Node | safe / settle | load/table | fire | source drain | wait | terminal |
|---|---|---|---|---|---|---|
| Camera | 不拥有sequencer | 无 | 无 | external timing下读C cycles | 无 | 只disarm camera |
| Calibration | load内部safe；无settle | load一次；无table | 一次forever | 读R个三帧cycle | **不wait** | camera close后safe |
| Stepped | 每point safe+settle | 每pointresolve/compile/load | 每point一次 | 每point取S publications | 每pointwait | run finally safe |
| Seamless | run开头safe+settle | load一次；rows×S，sweeps=R | run一次 | 取R×P×S publications | 一次wait | run finally safe |
| Temperature | fixed 50ms settle | 复用Seamless，S=1 | 一次 | R×P个双帧cycle | 一次 | 复用Seamless |
| SLM | 每chunk safe | 每chunk重复load同一program；empty row sweeps=chunk | 每chunk一次 | chunk个三帧cycle | 每chunkwait | safe后source close |

### 2.3 数量映射

| Node字段 | 实际含义 | Board表达 | Camera表达 | Dataset表达 |
|---|---|---|---|---|
| Camera `repeat=C` | C个external cycles；0=infinite | 不拥有 | repeat=C, frames/cycle=F | repeat axis=C |
| Calibration `Samples=R` | R个long/readout/long samples | forever后截停 | repeat=R, F=3 | samples=R |
| Stepped `Repeats=R` | 完整plan重扫R次 | host outer loop | 任意source | visit slow index |
| Stepped `Shots/point=S` | 同point完整cycle S次 | whole RepeatRegion=S | 期望S publications | 与R压入一个repeat axis |
| Seamless `Repeats=R` | table扫R遍 | table sweeps=R | R×P×S publications | 与S/source repeat压到一个axis |
| Seamless `Shots/point=S` | 同row完整cycle S次 | row重复S次 | S publications/row | 与R压入repeat axis |
| Temperature `Repeats=R` | t_off plan扫R遍 | table sweeps=R | repeat=R×P, F=2 | survival repeat=R |
| SLM `shots=Q` | candidate下Q个cycles | empty-row sweeps，按128切chunk | 每chunk repeat=chunk, F=3 | 在线Welford |

“repeat”同时表示camera cycles、plan sweeps、timeline loop、scan sweeps、Dataset axis。D-013是行为冲突，不是命名问题。

## 3. 已确认问题

### PULSE-CALL-001（P0）— authored RepeatRegion静默乘camera cycles

四路径处理同一模板repeat不同：

- Calibration / Seamless / Temperature / SLM原样保留；
- Stepped在shots=1删除whole repeat；
- Stepped在shots>1用shots覆盖whole repeat；
- partial repeat只在Stepped+shots>1拒绝。

给shipped pulse添加cover全timeline、count=2的RepeatRegion，纯编译探针：

~~~text
imaging_template:
  plain emCCD windows       3
  whole-repeat=2 windows    6
  node expected/cycle       3

temperature_template:
  plain emCCD windows       2
  whole-repeat=2 windows    4
  node expected/cycle       2
~~~

对Seamless/Temperature，一table row本应对应一camera cycle；whole repeat让第一row产生两个cycle。Camera仍只按R×P×S cycles arm，于是可能把第一row第二cycle记到下一row，后续真实row被忽略。这是silent point misassignment。

建议：

- measurement shots/cycles不再通过修改PulseSequence whole RepeatRegion表达；
- externally counted template若有whole repeat，Start前拒绝；
- partial RepeatRegion可保留，但exact camera cycle须验证F帧；
- 若用户坚持whole RepeatRegion就是shots，则六节点全部纳入它的count，不能半数覆盖、半数嵌套。

### PULSE-CALL-002（P0）— Calibration forever+camera截停丢board terminal proof

位置：`calibration/task.py:956-1107`。

Calibration先arm `3R` frames、load、`fire(forever=True)`，读R cycles后close camera并safe。它不调用 `wait_done`，构造器却要求这个未用方法。

后果：

- board没有精确R-cycle terminal，camera count决定切电；
- 无DoneReport，underflow/fault/cursor不参与成功；
- 第R cycle后到safe之间board可能进入下一cycle；
- camera先close、pulse后safe，中间仍可产生trigger；
- test主动断言fire(True)与零wait，误把实现钉成产品不变量。

SLM已证明zero-slot program可用empty-row sweeps执行有限Q cycles。Calibration可改为：

~~~text
load once
write cycle count R
arm camera 3R
fire finite
drain R cycles
wait_done with Stop/fault
safe
~~~

若不希望empty tuple成为应用API，应在 `zlc_pulse` 提供薄的cycle-count接口。

### PULSE-CALL-003（P0）— target compatibility没有唯一owner

当前：

- Calibration、Stepped、SLM做 `resolved.target != board.target`；
- Seamless、Temperature不检查；
- compiler只查geometry；
- `PulseStreamer.load`不查 `program.target_abi_fingerprint`。

只改一个port display label的探针：

~~~text
PulseTarget object equality   False
ABI fingerprint equality      True
~~~

前者会误拒ABI相同pulse；后两者可把ABI不同但geometry相同的program送进load。

正确owner是device load：

~~~text
program.target_abi_fingerprint == connected_target.abi_fingerprint
~~~

随后删除所有node-local target对象比较。

### PULSE-CALL-004（P0 ownership）— Stepped动态device port绕过claims

Stepped descriptor只声明sequencer EXCLUSIVE。Workbench另把installation全部 `tunable_devices`作为extras传入；`_split_row/_apply`直接 `device.tune`。Candidate claims只从static descriptor requirements生成。

因此plan可写 `device:mot_camera:exposure_seconds`，但Stepped没有camera claim。现有测试正是在Camera Measurement持有camera EXCLUSIVE并采集时，让Stepped无claim改同一camera exposure。

设计选项：

1. Start从plan解析dynamic device claims并纳入reservation；
2. 若被调device也是source producer owner，拒绝并要求own camera+sequencer的Task；
3. 建跨node tuning lease。

建议1+2；不建议3。`tunable_devices` dict不能当授权。

### PULSE-CALL-005（P0 same-shot）— Seamless的source并非by construction

Seamless文档说publication order与played row由构造保证。实际descriptor允许任意LIVE Dataset；`PublishedSignalSource.arm(program,table)`直接丢弃program/table，不验证source device、trigger或sequencer。

结果：

- free-running或另一producer也可选；
- unrelated publication会顺序记到scan row；
- `arm(program,table)`签名制造“已绑定”假象；
- Stepped至少有gating choice，Seamless没有。

Temperature不受影响：它own camera并用 `CameraCycleSource`。generic Seamless必须限制到明确pulse-cycle-coupled source、own exact source，或承认gating/关联不保证。

### PULSE-CALL-006（P1）— authored scan值与实播tick值分叉

`validate_scan_table`保留author-unit float；`scan_rows_to_wire`才round为wire ticks。Dataset/Temperature curve/run record仍保存author values。

20ns tick、Temperature `t_off` 探针：

~~~text
authored t_off        0.004001 ms
wire                  200 ticks
actual played         0.004000 ms
artifact/curve x      0.004001 ms
~~~

差值通常小，但它直接进入科学x轴。建议科学coordinate用 `scan_rows_from_wire(...)` actual；metadata同时保留authored intent、sweeps、program facts。

### PULSE-CALL-007（P1 provenance）— 多数节点丢pulse path/actual values

- Calibration保存path、parameter IDs/values/units/frame indices，最好；
- Stepped/Seamless只存sequence name、plan、R/S；
- Temperature构造时丢 `ResolvedWorkspaceResource.path`；
- SLM存pulse path，但不存resolved API values、actual table/chunks、DoneReport或camera snapshot。

同名JSON可修改，仅名字不能重建run。建议记录plain facts：

- resource path；
- resolved API values/units；
- target ABI；
- actual wire-decoded table/sweeps；
- compiled duration；
- DoneReport摘要；
- node的R/S/chunk映射。

### PULSE-CALL-008（P1 hierarchy）— Calibration helper被SLM跨plugin复用

SLM直接import `load_calibration_pulse_template`、`resolve_pulse`、`arm_sequencer`、`ResolvedPulse`，但它不运行Calibration protocol。

- `arm_sequencer`只是一行 `load`，名字虚称arm；
- `ResolvedPulse.metadata={"repeat_forever": ...}` production无人读且compile默认false；
- Calibration需三duration override，SLM只需authored defaults，语义已分叉。

建议删除one-line alias与dead metadata；两caller直接用 `zlc_pulse.authored_api_values/resolve_api_parameters/compile_sequence`。Calibration保留本地三参数规则，SLM保留imaging规则；不新增Resolver class/file。

### PULSE-CALL-009（P1 API identity）— Calibration按位置而非ID绑定

根架构明确三项duration绑定三个parameter IDs。当前UI保存1-based整数slot。API parameter插入/重排后，layout静默驱动另一字段；position不是identity。

代码只拒绝unit `value`，不要求 `field_ref.kind == duration`；time-valued delay也可能被当exposure。

建议UI动态列duration API parameters，以parameter_id持久化并验证三个ID存在、互异、kind=duration。

另有validation漂移：schema只拒readout > reference；`CalibrationRequest`拒readout >= reference。equal值Edit通过、build才失败。

### PULSE-CALL-010（P1 performance）— SLM每chunk重复load相同program

SLM execute只resolve/compile一次；`_measure`每chunk仍：

~~~text
safe -> load identical program -> write empty table -> arm -> fire
     -> read -> wait -> safe
~~~

`_SHOT_CHUNK=128`不来自board capacity，主要限制camera buffer。Task持sequencer EXCLUSIVE且pulse不变；`safe()`后software-loaded program仍可resident reload。

可task开始load一次；每chunk只改cycle count/table、arm、fire、wait、safe。chunk size应以camera/buffer policy命名并记录。现有bounded test守fires/sweeps/buffer，不守load count。

### PULSE-CALL-011（P1 simulation）— finite triggers在fire内同步爆发

`VirtualPulseStreamer.fire`同步遍历所有scan points调用 `world.fire`；`wait_done`只以logical deadline补墙钟。

隔离探针（5个20ms zero-slot cycles）：

~~~text
fire()返回约          1.6 ms
fire返回时world count 5
wait_done terminal    约100 ms
~~~

全部trigger/frame瞬时入队，随后只等DONE。影响：

- live progress不是真实cadence；
- camera buffer瞬时灌满；
- Stop在同步fire loop中不可见；
- 大table world render阻塞caller；
- 违背virtual finite cadence架构。

Calibration forever第一cycle也同步，后续thread按cadence；Stepped/Seamless/Temperature/SLM finite路径均受影响。修复必须在Virtual sequencer：physics callbacks与DONE共享logical schedule，不能node sleep。

### PULSE-CALL-012（P1 Stop/safe）— cleanup保证不一致

| Node | Stop观察点 | 不可取消区间 | cleanup |
|---|---|---|---|
| Camera | camera read slices | configure/arm | 只disarm camera；不碰external pulse，正确 |
| Calibration | 每cycle/read slice | load/fire、分析保存 | camera close→safe；外层异常再safe |
| Stepped | point/deadline/source/wait | settle sleep、tune/compile/load | source.close→safe |
| Seamless/Temperature | 每cycle/wait | settle；Virtual同步fire全table | source.close→safe |
| SLM | solver、chunk/shot/wait | phase apply、load/table/fire | safe→source.close |

Stepped/Seamless若source.close抛错会跳过safe；SLM nested finally更正确。settle应cancellable；Virtual同步fire在device层修。

### PULSE-CALL-013（P1 scope）— saved-frame Calibration仍强占硬件

`frame_source=folder` 路径不使用camera、sequencer或pulse；descriptor仍无条件要求camera+sequencer EXCLUSIVE和pulse resource，Task异常还会safe sequencer。

纯offline reanalysis会停止/阻塞bench节点。建议同plugin支持conditional device/resource claims，不新增node/file。

## 4. 推荐统一路径

### 4.1 术语

| 术语 | 唯一含义 |
|---|---|
| Pulse internal repeat | `RepeatRegion`，timeline内部循环，不等于measurement shots |
| Cycle | PulseSequence start到end的一次outer执行 |
| Frames per cycle | 一个cycle的camera tuple大小 |
| Shots per point | 同一plan row的独立cycles数 |
| Sweep | 完整plan走一遍 |
| Candidate shots | 某SLM candidate下cycles数 |
| Chunk | bounded camera/buffer拆分，不改变science geometry |

### 4.2 标准finite顺序

不新增executor class；node仍在owner内直写，但遵守：

~~~text
resolve resource + values
compile once per distinct program
sequencer.safe()
settle/tune only devices this node claims
sequencer.load(program)        # device统一ABI check
write rows / explicit cycle count
arm exact source for table_cycles * frames_per_cycle
fire(forever=False)
drain exactly table_cycles
wait_done in cancellable slices; reject fault
finally:
    sequencer.safe()
    source.close()
~~~

节点投影：

- Calibration：zero-slot cycle count=Samples；
- Stepped：每point cycle count=Shots/point，不改PulseSequence.repeat；
- Seamless：rows按point/shot展开，sweeps=Repeats；
- Temperature：Seamless的S=1；
- SLM：zero-slot cycle count=chunk，program只load一次；
- Camera：继续不拥有pulse。

### 4.3 Count invariant

Start前冻结：

~~~text
table_cycles
  == expected source publications
  == camera request.repeat

expected camera frames
  == table_cycles * frames_per_cycle
~~~

模板whole repeat存在时，要么拒绝，要么显式乘入全部三项；不能只让compiler知道。

### 4.4 不应统一

- Stepped每point safe/tune/settle；Seamless只table前一次；
- Calibration三duration protocol；SLM probe gate保持operator-authored；
- free-running deadline只属于Stepped；
- SLM phase rollback/artifact只属于SLM。

这些留plugin，不抽coordinator callbacks。

## 5. 逐文件 / 函数裁决

### 5.1 Calibration

| 文件 / 符号 | 裁决 | 理由 |
|---|---|---|
| `load_calibration_pulse_template` | `PASS WITH DEBT` | JSON single path正确；SLM复用造成owner命名错误。 |
| `_validate_calibration_sequence` | `REDESIGN` | 只拒slots；未处理whole repeat、required duration IDs/cycle contract。 |
| `ResolvedPulse` | `MERGE/DELETE` | metadata dead；program/sequence/path可局部持有。 |
| `arm_sequencer` | `DELETE` | one-line load alias且名字虚称arm。 |
| `resolve_pulse` | `PASS WITH DEBT` | default+owned override正确；target check下沉device；wrapper不跨plugin。 |
| `CalibrationRequest` slots | `REDESIGN` | exposure必要；1-based position不是API identity。 |
| `_driven_slots/_driven_values` | `REDESIGN` | unit conversion正确；改按duration parameter_id。 |
| `_pulse_facts` | `PASS WITH ADDITIONS` | 当前provenance最好；补actual count/table/DoneReport。 |
| `_capture` | `REDESIGN` | CameraMeasurement复用正确；forever/no wait错误。 |
| `_safe` | `PASS` | owner直接且幂等。 |
| `_replay_saved_frames` | `PASS` | offline科学路径必要；claims错误。 |
| `_run` | `PASS WITH DEBT` | direct flow清楚；offline异常也触硬件。 |
| `calibration/logic_node.py` | `REDESIGN` | parameter IDs、Samples default、conditional claims需裁决。 |

### 5.2 Camera Measurement

| 符号 | 裁决 | 理由 |
|---|---|---|
| descriptor无sequencer | `PASS` | external timing边界正确。 |
| `request.repeat` | `PASS WITH NAMING DEBT` | 实际是cycles，不是pulse repeat。 |
| `CameraCycleSource.open(cycles)` | `PASS` | Task own camera时exact count正确。 |
| `CameraCycleSource.arm(program,table)` | `REDESIGN` | 参数完全不用；删除或真实验证，不能伪arm。 |
| `next_value/close` | `PASS WITH FIX` | exact source必要；ordinal问题见03a。 |

### 5.3 Stepped

| 符号 | 裁决 | 理由 |
|---|---|---|
| descriptor/schema | `PASS WITH CRITICAL DEBT` | pulse/source/gating正确；dynamic claims缺失。 |
| `_build` | `REDESIGN` | tunable dict未形成authority。 |
| `_split_row` | `PASS WITH CLAIM FIX` | 分发清楚；调用前必须claim。 |
| `_api_values` | `MERGE` | 重写了已有 `authored_api_values`。 |
| `_apply` | `REDESIGN` | safe/tune正确；不应覆写whole repeat；每sweep重复compile。 |
| `_collect` | `PASS WITH DEBT` | absolute deadline正确；source关联保证有限。 |
| `execute` | `PASS WITH FIX` | outer flow清楚；precompile rows、cleanup safe。 |

### 5.4 Seamless / shared scan

| 符号 | 裁决 | 理由 |
|---|---|---|
| `_streamed_sequence` | `REDESIGN` | API→slot正确；缺whole-repeat与target guard。 |
| `_wire_table` | `PASS WITH DEBT` | row×shots正确；actual values未回写。 |
| `acquire` | `REDESIGN` | finite table正确；arbitrary source、cleanup、Virtual fire问题。 |
| `run_record` | `REDESIGN` | 缺path、actual table、DoneReport。 |
| `PublishedSignalSource` | `REDESIGN` | FollowTap必要；arm丢program/table。 |
| `wait_for_board` | `PASS` | cancellable/fault check集中正确。 |
| Seamless descriptor | `PASS WITH DEBT` | UI正确；input contract过宽。 |

### 5.5 Temperature

| 符号 | 裁决 | 理由 |
|---|---|---|
| `TemperatureTask.__init__` | `PASS WITH DEBT` | 复用Seamless/CameraCycleSource正确；pulse path丢失。 |
| t_off-only plan bind | `PASS` | plugin科学owner正确。 |
| camera repeat=R×P, F=2 | `PASS` | repeat-free模板与table一致。 |
| `execute` | `PASS WITH UPSTREAM FIX` | 继承Seamless repeat/ABI/quantization问题。 |
| point column/curve/run record | `REDESIGN` | x轴用authored非actual；缺pulse/table proof。 |
| descriptor claims | `PASS` | own camera+sequencer正确。 |

### 5.6 SLM Feedback

| 符号 | 裁决 | 理由 |
|---|---|---|
| pulse loader/import | `MOVE/DELETE WRAPPER` | 不应伪装Calibration helper。 |
| `resolve_pulse(api_values={})` | `PASS` | 当前裁决要求probe gate保持pulse-authored。 |
| `_measure` count/chunk | `PASS WITH DEBT` | exact Q shots、3-frame grouping、Welford正确；magic chunk、重复load、whole repeat缺口。 |
| empty-row table | `PASS AS BEHAVIOR / API DEBT` | finite cycles正确；需清晰cycle-count语义。 |
| execution finally | `PASS WITH PERFORMANCE FIX` | cleanup顺序最好；load移出chunk。 |
| `_READOUT_FRAME=1`, F=3 | `PASS WITH CONTRACT DEBT` | shipped protocol正确；loader未显式验证frame protocol。 |
| candidate/validation分开 | `PASS` | independent validation正确。 |
| metadata | `REDESIGN` | 补resolved values、actual chunks/table/DoneReport/camera snapshot。 |

### 5.7 zlc_pulse / device

| 符号 | 裁决 | 理由 |
|---|---|---|
| `authored_api_values / resolve_api_parameters` | `PASS` | 已有唯一覆盖路径。 |
| `compile_sequence` | `PASS WITH NOTE` | geometry验证正确；不代替connected ABI。 |
| `scan_rows_to_wire/from_wire` | `PASS` | 唯一转换已存在；应用未保存actual。 |
| `PulseStreamer.load` | `REDESIGN` | 统一校验target ABI。 |
| `write_scan_table` | `PASS WITH API DEBT` | table/sweeps正确；zero-column cycle count不清晰。 |
| `fire(forever)` | `PASS` | primitive合理；Measurement不应用forever模拟finite。 |
| `wait_done/safe` | `PASS` | 节点调用需统一。 |
| `AppliedState` | `REDESIGN` | 不保存sweeps，无法完整readback。 |
| `VirtualPulseStreamer.fire` | `REDESIGN` | physics triggers按logical cadence，不能同步burst。 |

## 6. 测试裁决

| 测试 | 裁决 |
|---|---|
| Camera leaf无sequencer test | `KEEP`。 |
| Calibration resolver roundtrip | `KEEP WITH ADDITIONS`；补API reorder/duration-kind/whole repeat。 |
| Calibration descriptor exercise | `REDESIGN TEST`；当前钉forever=True、wait=0。 |
| Calibration safe-on-fire-failure | `KEEP`。 |
| Stepped shots/repeats | `USER DECISION`；精确钉“shots=whole RepeatRegion”。 |
| Stepped gating/deadline | `KEEP`。 |
| Stepped camera exposure port | `REDESIGN TEST`；功能同时固化无claim并发改camera。 |
| Seamless table/order | `KEEP WITH ADDITIONS`；补whole-repeat old-red。 |
| Temperature end-to-end | `KEEP WITH ADDITIONS`；补actual tick、target mismatch、whole repeat。 |
| SLM bounded grouped | `KEEP WITH ADDITIONS`；补load count、repeat、DoneReport provenance。 |
| SLM Stop/terminal | `KEEP`。 |
| zlc_pulse table/bank | `KEEP`；不证明node count mapping。 |
| target ABI | `MISSING`。 |
| dynamic device claim | `MISSING`。 |
| Virtual trigger cadence | `MISSING`；不能只测DONE延时。 |

最小新增守卫：

1. repeat-bearing imaging/temperature template明确拒绝或count一致；
2. Calibration R samples为finite count，fire非forever，DoneReport无fault；
3. Stepped device port进入reservation；同camera source冲突拒绝；
4. Seamless unrelated/free-running source不能静默Start；
5. ABI check只在load：label-only accepted，ABI mismatch rejected；
6. Temperature artifact x等于wire-decoded actual；
7. SLM同pulse跨chunks只full-load一次；
8. source.close失败仍safe；
9. Virtual cycles的camera publications按logical cadence出现。

## 7. 文档冲突

| 文档 | 实现 | 结论 |
|---|---|---|
| Calibration durations绑定明确parameter IDs | UI/Request存1-based positions | 直接矛盾；建议ID。 |
| Calibration Samples默认300 | schema默认200 | 待用户裁决。 |
| Virtual finite cadence真实推进trigger | fire同步生成world events，只延DONE | 实现违约。 |
| Stepped shots用whole bracket | 当前符合；其他nodes另用外层count | 文档只覆盖一个node，缺统一。 |
| Seamless source由fired cycle保证 | generic source任意LIVE且arm不验证 | 实现无法证明。 |
| SLM probe gate保持authored API | 当前保留file defaults | 实现符合，不应改成Calibration values。 |
| run metadata含actual pulse/sequencer facts | 多数scan/task只存name/path/authored plan | 不完整。 |
| 同设备一个EXCLUSIVE owner | Stepped动态tune无claim | 实现违约。 |

## 8. 需要用户裁决

1. **D-013：whole-pulse RepeatRegion是否兼作shots？**
   A. 保持Stepped语义，其他节点纳入；
   B. RepeatRegion只属timeline，measurement shots统一cycle/table count。
   建议B。

2. **Calibration finite execution**
   是否从forever+camera截停改为finite R + DoneReport？建议是。

3. **Calibration API binding**
   parameter ID还是1-based位置？建议ID且只列duration fields。

4. **Seamless source contract**
   是否要求source声明由selected sequencer cycle驱动？否则需gating choice或限制own-camera Task。

5. **Stepped dynamic devices**
   若允许任意tunable device，必须dynamic claim；device也是source owner时建议拒绝。

6. **Actual scan coordinates**
   科学x轴用authored还是tick actual？建议actual，metadata同时留authored。

7. **Cycle-count API**
   正式接受 `write_scan_table(((),),sweeps=N)`，还是在 `zlc_pulse` 增薄 `cycle_count` API？建议后者。

8. **Saved-frame Calibration claims**
   是否真正offline不claim camera/sequencer？建议允许conditional claims，不新增node。

## 9. 推荐实施顺序（未实施）

1. 裁决D-013、Calibration finite count、Seamless source contract。
2. 在 `PulseStreamer.load`加ABI guard，删node-local target比较。
3. 加whole-repeat camera-window count old-red，冻结count invariant。
4. Calibration改finite count + wait/fault。
5. Stepped改dynamic claims；再迁移shots表达。
6. Seamless/Temperature记录actual wire rows并加强source contract。
7. SLM load移出chunk，记录chunk/table/DoneReport。
8. 修Virtual trigger cadence；不得node sleep。
9. 删除 `arm_sequencer`、dead metadata、重复参数构造。
10. 最后更新tests/旧文档。

## 10. 探针边界

- Python进程先打印当前checkout的 `zou_lab_control_v2`、`zlc_atom`、`zlc_pulse` 路径。
- Repeat/window与quantization探针只编译当前JSON，不连接硬件。
- Virtual cadence来自Memory transport+fake world，无真实camera/FPGA。
- 未写workspace pulse、未调用真实sequencer、未改production/tests。
