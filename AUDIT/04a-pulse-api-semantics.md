# 04a — Pulse API、重复语义与执行真相深审

状态：完成（只读审查；未修改 production/test/旧文档）
审查基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：`zlc_pulse` 的 model/binding/compile/schedule/scan/device/wire，Pulse Editor 的执行/同步边界，以及 Measurement/Task 对 pulse 的关键调用。Camera same-shot 的逐帧细节由 `04b`、逐 Logic Node 调用矩阵由 `04c`继续展开。

## 1. 结论先行

当前 pulse 栈不是“API 命名有点乱”这么简单，而是有四类彼此放大的问题：

1. **存在可直接导致实板静默错波形的缺口。** 延时 event FIFO 容量没有任何 host validator；target ABI、编译时钟和 scan fixed-point geometry 也没有在 `load()` 边界核对。
2. **没有一份完整的执行真相。** `CompiledProgram`、外置 scan rows、私有 `_scan_sweeps`、`fire(forever)`、`RepeatRegion` 和 Measurement 自己的 `repeats/shots` 分散持有运行含义；`AppliedState` 又漏掉 sweeps，无法重建真正播放次数。
3. **同名 repeat 在不同层表示不同物理动作。** bracket repeat、whole-pulse repeat、scan table sweeps、plan repeats、shots per point、Camera repeat 和 source repeat 被反复互相翻译；其中 stepped 与 seamless 对 `shots_per_point` 的实现不同，只有在额外的“一次顶层 pulse 恰好发布一次”假设下才等价。
4. **测试大量证明“当前实现与当前 fake 一致”，没有证明“当前实现与板卡/相机的唯一物理语义一致”。** 多个关键错误都能通过现有全套窄测试。

因此本阶段结论是 `REDESIGN`，不是继续在各 Measurement 中补乘法或加特判。优先级顺序应为：

1. 先封死实板静默错误：delay FIFO capacity、ABI/clock/geometry、32-bit count；
2. 再确定唯一重复术语与 execution truth；
3. 然后统一 Stepped/Seamless/Calibration/SLM 的调用；
4. 最后才清理历史字段、重复 helper 和文档。

## 2. 唯一术语：建议与当前实现对照

下表先只定义物理含义，不让某个现有类名反过来决定产品语义。

| 建议唯一术语 | 唯一含义 | 当前表示 | 当前问题 |
|---|---|---|---|
| Pulse period | `PulseSequence.periods` 中一段正时长、固定 digital state/analog action | `PulsePeriod`；UI 常称 step | 基本清楚；建议文档只把 step 当 UI 标签，不再产生第二个模型词 |
| Region repeat / bracket | 一个连续 period region 在**一次 program execution 内**重复 `C>=2` 次 | `RepeatRegion`、`loop_start/end/count` | 全幅 bracket 又被赋予“whole pulse/有限运行策略”，跨越模型与编排边界 |
| Program execution | 对一个 scan row 执行一次完整 compiled timeline；内部可含 region repeat | 没有一等名称 | 常被叫 cycle、shot、pulse repeat，导致乘法不透明 |
| Scan row / hardware point | 一组 runtime slot wire values；板卡消费一行后 cursor 前进一次 | `write_scan_table()` row、`SCAN_COUNT` | zero-width row 又被用作纯播放计数；合理但 API 名义仍叫 scan |
| Table sweep | 按原顺序完整遍历同一 row table 一次 | `write_scan_table(..., sweeps=R)` | `sweeps` 不在 RTL，不在 `AppliedState`，由 host 用 `SCAN_COUNT=N*R` 和 modulo refill 模拟 |
| Plan point | 科学 scan plan 的一个坐标组合 | `ScanPlan.rows()` 的一行 | 到 wire 前可能因 tick/DAC rounding 改值，但 dataset 仍保存 authored float |
| Plan repeat / full sweep repeat | 完整科学 plan 从头重扫一次 | Logic Node `repeats=R` | Seamless 翻译为 table sweeps；Stepped 是 Python 外层循环 |
| Shot per point | 相同 plan point 下的一次完整独立观测机会 | `shots_per_point=S` | Stepped 改写 whole bracket；Seamless 重复 scan row；两者不是无条件等价 |
| Camera cycle repeat | Camera Measurement 需要交付的完整 same-shot group 数；`0` 才是无限 | `CameraMeasurementRequest.repeat=N` | 与 pulse 的 program executions/trigger windows 没有共同 cardinality contract |
| Frames per camera cycle | 一个完整 camera cycle 中按物理顺序交付的 readout event 数 | `frames_per_cycle=F` | 目前由 Camera request 和 pulse windows分别声明，缺少统一核对 |
| Forever | 无限重复当前 program/table，直到显式 safe/stop | `fire(forever=True)` / `REPEAT_FOREVER` | `CompiledProgram.repeat_forever` 还有一份永远由 compiler 写 False 的历史镜像 |
| Source repeat | 被采样 Dataset 自己已有的 repeat axis | `source_schema.repeat_axis` | Scan 又把 plan repeats × shots × source repeats 全压回同一 repeat axis，物理差异只留在 metadata |

### 当前真实乘法链

令：

- `P = plan point_count`
- `S = shots_per_point`
- `R = plan repeats`
- `B = RepeatRegion.count`（无 bracket 时为 1）
- `W = 一次未展开 timeline 在所选 camera channel 上的窗口数`
- `F = frames_per_cycle`

当前实现如下：

| 路径 | 板卡层实现 | 代码期望 publication/cycles | 真实 pulse camera windows |
|---|---|---:|---:|
| Stepped | 每 point compile/load/fire 一次；把 whole bracket 改成 `S` | `R * P * S` | 通常为 `R * P * S * W`；模板 partial bracket 可改变 W，`S>1`时则直接拒绝 |
| Seamless | row table 先变成 `P*S` 行，再 `sweeps=R` | `R * P * S` | `R * P * S * windows(program,row)`；模板原有 repeat 完全保留 |
| SLM Feedback | zero-width row `((),)`，`sweeps=chunk` | `chunk`个三帧 cycle | 若模板自身 repeat 使窗口重复，会大于 `chunk*3` |
| Calibration | program `fire(forever=True)`；Camera finite repeat=N后主动 safe | N个三帧 cycle | 由 pulse 无限产生；N只属于 camera reader，不属于 pulse |
| Pulse Editor finite scan | rows=N，`sweeps=scan_repeats` | 无 Measurement accounting | `N * sweeps * W`，每 row 内再乘 bracket影响 |

这说明 `shots_per_point` 目前不是 pulse core 的一个事实，而是两个不同执行策略的同名参数。`B/W/F` 也没有在任何共同边界证明相容。

## 3. 明确错误与高风险问题

### PULSE-001 — P0：delay event FIFO capacity validator 完全缺失

证据：

- `wire.StreamerParams` 明确携带 `evt_fifo_depth` / `bus_evt_fifo_depth`；
- 当前 RTL/Python mirror 注释都写明 FIFO 满时会**丢弃** toggle/descriptor，并声称“host validator prevents”；
- `compile_sequence()`、`pack_program()` 和 `PulseStreamer.load()` 没有任何 in-flight edge/segment 计数；全仓 production 也没有相应 validator。

隔离探针构造了 130 个每 tick 翻转一次的合法 period、200 tick output delay、FIFO depth 64：

```text
compiled_and_packed: 132 edges / 546 words
rtl_drops_vs_reference: True
different_ticks: 33
```

也就是说当前 host 会成功编译、成功打包、成功 LOAD，而 frozen RTL 按自己明示的 overflow guard 静默丢边。这不是边缘 UI 问题，是硬件波形正确性 blocker。

结论：`compile_sequence`/最终 pack 边界必须用**实际 compiled edge/bus descriptors + connected geometry**验证每个延时窗口内的 peak in-flight 数；超限必须在任何寄存器写入前拒绝。只验证 delay 数值能放进 32 bit 远远不够。

### PULSE-002 — P0：compiled program 没有在设备边界绑定正确 board ABI/clock/scan geometry

三个独立缺口：

1. `CompiledProgram.target_abi_fingerprint` 已存在，但 `PulseStreamer.load()` 从不比较 `self._target.abi_fingerprint`；隔离探针确认 ABI 不同仍成功 applied。
2. `CompiledProgram.clock_hz` 与 `PulseStreamer.clock_hz` 不比较；`compile_sequence()`也不检查 `sequence.time_step_ns == 1e9/clock_hz`。20 ns 文档用 25 MHz 编译会成功并实际播放 40 ns。
3. scan coefficient 用编译 geometry 的 `coeff_frac_bits`缩放，但设备 pack/load 不比较 connected geometry；探针用 frac=8 编译、frac=7 的已握手 device 加载，成功接受 coefficient 256，host/board解释相差2倍。

当前 caller 的补丁也不正确：Calibration/Stepped 用 `resolved.target != board.target` 全对象相等；仅改变 display label 时 ABI fingerprint 不变，但对象不等，因而会误拒。Seamless则不比较。

唯一正确 owner 应是：

- server/installed sequencer 的 `BoardDescription` 提供当前 target/geometry/clock；
- compiler记录所依据的事实；
- **`PulseStreamer.load()` 是唯一强制边界**，在 physical SAFE 或写 image 前比较 target ABI、clock 和现有 `scan_coeff_frac_bits`；
- caller 不再各写一份全对象比较。

另需纠正文档措辞：当前 CTRL[63] 只证明 geometry fingerprint；target logical mapping/package pins 和 clock 并没有由 FPGA readback 证明，只是 server 侧 deployment XDC/config 的声明。

### PULSE-003 — P1：Delay scan slot 是“可构造、可上传、实际无作用”的假功能

`model.py` 自己已经给出相互矛盾的两份答案：

- `FIELD_BINDING_CYCLES[FIELD_DELAY] = (None, "api")`，明确 delay 没有 scan table column；
- `SLOT_KINDS` 却包含 `FIELD_DELAY`，`PulseSlot`/codec/compiler/device 全部接受 delay slot。

但 compiler 只把 duration slot用于 period starts、把 DAC slot用于 bus selector；`_delay_values()`完全忽略 slot binding。隔离探针结果：

```text
slot_kinds = ('delay',)
tick_slot_coeffs = ((0,), (0,), (0,))
trigger_times(row=0) == trigger_times(row=5) == [0]
```

因此该列会走完整 scan/wire path，却不能改变任何 channel/bus delay。负值还会在更晚的 unsigned wire packing 才失败。

结论：当前硬件/编译设计下 delay 只能是 API parameter，必须在 `PulseSequence` 构造时拒绝 delay `PulseSlot`；删除 compiler 中为它保留的死分支和相应 `CompiledProgram` kind。若未来真要 scan delay，需要先有可表达动态 delay 的 wire/RTL设计，不能只把枚举值加回去。

### PULSE-004 — P1：计数存在静默 coercion 与 32-bit wrap

`PulseStreamer.write_scan_table()` 使用：

```python
self._scan_sweeps = max(1, int(sweeps))
```

Remote server 与 `wire.scan_bank_words()`又各做一次同类 coercion。实测：`0`、`-3`、`1.9`、`True` 全部静默变成 1。

更严重的是 wire register 为 32 bit，host 未做上界校验：

```text
RepeatRegion.count = 2**32 + 2 -> model 4294967298, hardware LOOP_COUNT word 2
sweeps = 2**32 + 3          -> snapshot scan_count 4294967299, hardware SCAN_COUNT word 3
```

这会让 run record、timeout、camera expected count 和板卡真实播放次数完全分裂。

结论：public boundary 严格拒绝 bool、非整数、`<1` 和乘积超出 unsigned-32 的值；`RepeatRegion.count` 在 compile/pack 的硬件边界也必须 checked，不得依赖 `_write()`的 `& 0xFFFFFFFF`。

### PULSE-005 — P1：Virtual sequencer 只延迟 terminal，所有物理事件仍在 `fire()` 中爆发

`VirtualPulseStreamer.fire()`先同步执行 `_fire_world(applied, points)`，它逐 row 立即调用 `world.fire()`并渲染/trigger全部 camera frames；随后只设置 `_logical_deadline`，让 `wait_done()`晚一点返回。

隔离探针：5 个 20 ms zero-slot shots，逻辑总时长 0.1 s；`fire()`约 1.6 ms 返回时 `world.fire_count` 已经是 5，只有 `wait_done(0)`仍返回 None。

这直接违背 `ARCHITECTURE_DESIGN.md` 关于 virtual finite fire 不应把几十/几百次采集压进一个 Monitor refresh interval 的文字，也解释了以下现象：

- 大 scan/多 shots 在调用 `fire()`的 worker 内同步渲染全部帧，造成明显卡顿；
- camera raw buffer 被瞬间灌满，virtual 的 overrun/consumer pacing 与实机不同；
- UI随后可能成批消费早已生成的帧，看似 live，实际没有物理 cadence；
- 现有测试只测 `wait_done`墙钟时间，未测 frame arrival cadence，因而错误通过。

结论：产品 virtual 路径必须按 compiled per-point schedule 推进 world/camera event；若保留极速离线模式，必须是显式独立模式，不能伪装成正式 virtual apparatus acceptance。

### PULSE-006 — P1：`AppliedState` 不是完整 application truth，Pulse Editor Sync 会显示错误状态

设备真正持有：program、source、rows、sweeps、forever、当前 table form。`AppliedState`只保存 unique `scan_rows`，不保存 sweeps；sweeps藏在 `PulseStreamer._scan_sweeps`。具体后果：

- 外部 `write_scan_table(rows, sweeps=5)` 后，Pulse Editor Sync只能拿到 rows，保留自己旧的 `scan_repeats`；下一次 On 与当前板卡不同。
- 外部 `write_slots(values)` 后，`AppliedState.slot_values`有实际 row，但 `PulseEditor.sync_from_sequencer()`只读 `scan_rows`，完全忽略 slot values，UI显示 source nominal field。
- Sync也不使用 `AppliedState.forever`恢复 finite/infinite scan含义。
- `AppliedState.source` 是 resolved executable source；Pulse Editor On 会移除 API declarations，因此 Sync 会丢失原 authoring API binding，这一点必须被明确当作不可逆 executable view，而不是“完整原 pulse”。

结论有两个诚实选项：

1. 保留 Sync，则 application echo 必须完整包含 table rows、sweeps、forever和单-row form，Workbench逐项恢复；
2. 删除“从设备还原 authoring document”的承诺，只显示只读 executable/applied摘要。

当前“一半同步”最差。

### PULSE-007 — P1 / USER DECISION：`RepeatRegion` 同时拥有 bracket science 与 shots，存在双重 owner

`PulseSequence.whole_pulse_repeat`把“首 period 到末 period 的 bracket”解释成 whole-pulse有限次数，并进一步声称“无 bracket 默认 until-stopped”。但同一个 sequence：

- 在 Pulse Editor 中可能据此 finite/forever；
- 在 Stepped 中 whole bracket 被 `shots_per_point`覆盖，`S=1`时甚至删除；
- 在 Seamless/Calibration/SLM 中原 bracket 保留，可能额外乘 camera windows；
- `compile_sequence`本身始终生成 `repeat_forever=False`，真实 forever 又由 `fire()`参数决定。

Git历史还显示当前 Stepped语义在不到一小时内从“每 shot 完整 safe/apply/load/fire”改为“一个 fire 内 whole bracket”，说明这不是稳定物理契约，只是最近一次实现选择。测试随后同步改写，不能充当用户裁决。

审计建议（对应 `DECISIONS.md` 的 D-013）：

- `RepeatRegion`只表示 program 内 region bracket；`full_span_repeat_count`也只是其结构属性，不再隐式决定 forever。
- `shots_per_point`在 execution层统一翻译成相同 point 的顶层 program executions；Stepped可与Seamless一样用 repeated rows/zero-width rows，而不是改写文档 bracket。
- 对 scan/calibration/feedback template 明确禁止或显式乘入 full-span bracket；绝不 silent override。
- finite/forever是 execution request 的字段，不再藏在 bracket是否存在这一推断里。

若用户选择维持当前设计，则每个模板必须验证“每 whole iteration恰好产生一条所选 source publication”，并在 UI 明示模板 whole bracket 会被 shots覆盖；否则仍不可接受。

### PULSE-008 — P1：schedule API 无法表达完整 device execution

`trigger_times/trigger_windows/run_duration_seconds`只接 `program + table`：

- 不接 device `sweeps`；
- 不接 runtime `forever`；
- 默认还会读取历史 `program.scan_points`；
- `run_duration_seconds`不包含 output-delay done-tail；
- table值用 `int()`静默截断，而不是要求 wire int。

于是 common schedule并不是执行真相。Virtual靠读取私有 `_scan_sweeps`并自行展开；Measurement各自手算 `R*P*S`；Camera cardinality另算；AppliedState又不能补齐。

此外 frozen RTL 在 program final 即置 DONE，但 delayed TTL/DAC scheduler仍继续排空；`wait_done()`立刻返回后 caller常马上 `safe()`，可能截断作者期待的 delayed final edges。`DoneReport.tail_elapsed`名字像 tail事实，实际只是 `now - fire_started` 的总墙钟时间，remote还把它打印成 `tail_ms`。需要用户/硬件层明确裁决：DONE到底表示timeline结束还是所有物理延迟输出完成。当前名字和调用均不诚实。

最小方向是不新造大框架：让 pure schedule函数显式接 `rows`、`sweeps`，由一处展开有限 execution；device/applied提供这些同源参数。Delay tail另建立真实完成定义，不能继续用误名字段掩盖。

### PULSE-009 — P2：authored、canonical和actual数值被静默混用

`resolve_api_parameters()`/`replace_pulse_field()`会把 off-grid time自动 round到最近 tick，DAC则 `round`成整数；scan wire conversion也 round。隔离探针中请求 `1.000011 ms`，resolved source为 `1.00002 ms`。

但：

- `ScanPlan`和 dataset coordinate保留原 float；
- run record通常保存原 authored plan；
- Seamless table实际播放rounded wire row；
- Stepped source保存rounded value，但final dataset仍用原 plan rows。

因此 saved plot可宣称 x=1.000011 ms，实播却是1.00002 ms。Camera adapter已有 authored request + actual readback 的清楚区分，pulse也应采用同一原则。

建议：plan bind/start时一次 materialize canonical played values；Dataset科学坐标使用 actual/canonical，run metadata另存 authored。若产品希望严格，则直接拒绝 off-grid，不应在每层各 round 一次。

### PULSE-010 — P2：`CompiledProgram`保留第二条、测试驱动的 embedded-table路径

Compiler明确注释“table data不属于 compiled program”，且 `compile_sequence()`总是写：

- `scan_points=()`
- `scan_point_durations=()`
- `repeat_forever=False`

但 `CompiledProgram`仍持有这些字段，`device.load()`、wire pack和schedule仍支持 embedded table；production没有 compiler能产生它，主要由 `engine_model`/tests通过 `dataclasses.replace()`伪造。

同时：

- `scan_point_durations`没有production reader；
- `scan_enabled`没有消费者；
- `camera_window_count/exposures`没有production consumer（camera/world直接用schedule）；
- `slot_units/slot_ids`没有production reader；
- `repeat_forever`与 runtime `fire(forever)`重复并会在log/metadata中给出错误的静态 False。

这些字段让 program看似拥有table/run policy，实际上truth在device外部。建议删除 embedded table/repeat policy和测试专用便利方法；engine model测试应显式把 rows作为输入，不应靠伪造production dataclass维持第二执行路径。

### PULSE-011 — P2：scan conversion有重复实现和死分支

`compile._slot_row(sequence, values=...)`实现了一套 mapping/sequence -> ticks/DAC offset conversion，但 production只以 `values=None`调用它来生成 nominal validation row。真实 runtime conversion在 `scan.scan_rows_to_wire()`。

两套实现已经不一致：前者读 `PulseSlot.unit`，后者/编辑器按 physical field authored unit；前者允许field-name fallback，后者按column order。`PulseSlot.unit`因而也成为两份unit truth：scan catalog基本忽略它，compiler却把它存进无人读取的 `CompiledProgram.slot_units`。

结论：删除 `_slot_row(values)`死分支，只保留明确的 nominal row helper；scan table只由 `scan.py`转换。按当前产品文字，scan column永远使用field authored unit，则 `PulseSlot.unit`应删除；若要支持独立slot unit，就必须所有路径统一尊重它，由用户裁决。

### PULSE-012 — P2：`wire.py`层级错误且混有死/历史 API

`wire.py`同时承担：

- runtime register ABI/packing；
- config读取；
- FPGA part资源估算/容量求解；
- Verilog/TCL geometry emit；
- CLI；
- test-only unpacker。

已有 `fpga.py`宣称自己是build-time owner，却只是从 `wire.py`重导出，真实职责没有移动。`pack_scan_rows()`甚至定义在 `if __name__ == "__main__"`之后：import时可用，直接运行wire模块时该定义永远到不了，是明显历史拼接痕迹。

建议在现有文件内完成移动，不增层：runtime packing/config留 `wire.py`，resource/emit/CLI实现移入已有 `fpga.py`。`default_clock_hz()`全仓除定义外无消费者；`unpack_program()`只有测试使用，应删除或移到测试诊断，不作为runtime production seam。

## 4. 逐文件、类与函数裁决

### 4.1 `zlc_pulse/model.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| constants / `FIELD_BINDING_CYCLES` | PASS WITH DEBT | domain常量必要；delay cycle与`SLOT_KINDS`矛盾必须消除 |
| `cycle_binding_kind` | PASS | UI与model共享有限状态正确；它给出的delay不可scan答案应成为constructor invariant |
| `_text/_identifier/_number/_nonnegative_int/_unit` | PASS WITH DEBT | 集中validation合理；`_text`对空文本抛TypeError且后续ValueError分支不可达，需局部清理 |
| `_tick_ratio` / `exact_ticks` / `align_to_grid` | PASS WITH DEBT | Fraction-based grid数学正确；`align_to_grid`是authoring行为，不能让scientific actual metadata仍保留旧值 |
| `PulsePortSpec` / `signed_range` | PASS | logical port与DAC编码归属正确 |
| `PulseTarget` | PASS WITH DEBT | target/ABI owner合理；`lanes`alias和`lanes=`constructor兼容面无production消费者，可删；ABI必须在device load真正使用 |
| `PulseFieldRef` | PASS | 一份typed physical field identity |
| `PulseSlot` | REDESIGN | 必须拒绝delay；`unit`与field unit双源 |
| `PulseApiParameter` | PASS WITH DEBT | 与scan slot分开合理；resolution后的actual值记录不足 |
| `AnalogStep` / `PulsePeriod` | PASS | model层职责准确 |
| `OutputDelay` | PASS WITH DEBT | relative delay模型必要；FIFO capacity与done-tail契约缺失 |
| `RepeatRegion` | USER DECISION | bracket结构必要；不应再自动拥有run finite/forever与shots，见D-013 |
| `PulseSequence.__init__` | REDESIGN | binding唯一性检查很好；未拒delay slot、未限制hardware loop count；time_step与board clock双源 |
| `slot_count` / `period_by_id` / `field_unit` | PASS | 有真实消费者 |
| `slot_kinds` / `slot_by_id` / `api_parameter_by_id`及其cache | DELETE | 全仓production无消费者；后两者连测试也不读 |
| `whole_pulse_repeat` | USER DECISION | 改为纯结构名`full_span_repeat_count`或删除特殊语义；当前docstring跨层且对Measurement不成立 |

### 4.2 `zlc_pulse/binding.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| `pulse_field_value` | PASS | 唯一field readback conversion |
| `replace_pulse_field` | PASS WITH DEBT | owner正确；time/DAC静默canonicalization必须向caller返回/记录actual |
| `authored_api_values` | PASS | subset override的正确起点 |
| `resolve_api_parameters` | PASS WITH DEBT | exact key-set validation很好；其结果应成为run/dataset actual truth |
| `convert_time` / `_number_for` / `_check_inputs` | PASS | 小而直接；保留 |

### 4.3 `zlc_pulse/compile.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| `TargetBusDelay` / `TargetBusSegment` | PASS | wire-facing immutable IR必要 |
| `CompiledProgram.__post_init__` | REDESIGN | essential IR与embedded table/run policy/test fields混合；还会拒绝某些只有动态duration的all-low/analog-only合法程序，因为用base tick验证loop |
| `slot_count` | PASS | device/wire需要 |
| `scan_enabled` | DELETE | 无消费者，且延续embedded-table假象 |
| `camera_window_count` / `camera_window_exposures` | DELETE | 无production消费者；直接使用`schedule`即可，Camera cardinality不能靠便利方法暗示已验证 |
| `digest` | PASS | compiled wire-equivalent identity有真实GUI消费者 |
| `_resolve_slot_operand_width` / `slot_operand_width` / `narrow_slot_operand` / `evaluate_affine_tick` | PASS WITH DEBT | board math必须单源；load仍须绑定正确geometry |
| `_slot_index` / `_default_slot_value` | PASS WITH DEBT | nominal compile validation需要；删除delay case |
| `_slot_row(values=...)` | DELETE/MERGE | 非None分支无调用且重复`scan.py`；只保留nominal helper |
| `_table_rows` | RENAME/PRUNE | 不是table，只是一个nominal validation row |
| `_period_starts` / `_effective_rows` / `_bus_segments` / `_delay_values` | PASS WITH DEBT | compiler核心应保留；需接delay FIFO capacity验证、移除无效slot kind |
| `compile_sequence` | REDESIGN | 应检查sequence clock、slot capacity、loop count并产出能在load验证的ABI事实；当前缺delay physical validator |

### 4.4 `zlc_pulse/schedule.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| `trigger_times` / `trigger_windows` | PASS WITH DEBT | pure projection owner正确；需显式有限execution rows/sweeps并拒绝非wire int |
| `run_duration_seconds` | REDESIGN | 名称声称exact run，实际不含sweeps、runtime forever和delay tail |
| `_channel_transitions` / `_point_timing` | PASS WITH DEBT | region loop展开集中且必要；没有完整execution输入 |
| `_scan_points` | REDESIGN | 应只解析显式table；删除`prog.scan_points`第二truth与`int(float)`截断 |

### 4.5 `zlc_pulse/scan.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| `ScanColumnSpec` | PASS WITH DEBT | column conversion owner正确；名称同时服务API parameter，且constructor未验证finite/scale/limit完整性 |
| `_column_for_field` | PASS WITH DEBT | kind-specific范围必要；不存在port时回退`-512..511`会掩盖model破坏，应直接拒绝 |
| `scan_columns_for` | REDESIGN | 应只含duration/DAC；需裁决并消除`PulseSlot.unit`双源 |
| `api_parameter_columns_for` | PASS | ScanPlan复用API parameter范围合理 |
| `resolve_scan_point` | PASS WITH DEBT | held point静态化路径有真实消费者；canonical actual要可见 |
| `scan_rows_to_wire` / `scan_rows_from_wire` | PASS WITH DEBT | 唯一wire crossing应保留；public调用最好要求先validate或组合为一个入口 |
| `_longest_slot_ticks` / `_ticks_per` / `_quantum` | PASS | 物理limit helper合理；依赖frozen config应在文档说明 |
| `validate_scan_table` | PASS WITH DEBT | shape/finite/range核心正确；numeric strings/bool会经NumPy静默转型，off-grid值又不materialize actual |
| `scan_table_template`及nested `_sweep/_note` | PASS WITH DEBT | 编辑器生成器有真实消费者；无columns时虚构`s0`、未知kind默认为column stack应改为拒绝 |

### 4.6 `zlc_pulse/device.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| `DoneReport` / `fault` / readback properties | PASS WITH DEBT | terminal facts必要；`tail_elapsed`误名且不是tail completion |
| `SafeReadback` | PASS | 物理SAFE事实清楚 |
| `AppliedState` | REDESIGN | 漏sweeps；source不是可逆authoring；无法支撑其GUI sync承诺 |
| `BoardDescription` | PASS WITH DEBT | device description owner正确；target/clock不是CTRL handshake证明的板内readback |
| `PulseStreamer.__init__` | PASS WITH DEBT | single session owner正确；clock finite检查不完整（bool/NaN）；target/geometry只查count/width不等于load ABI检查 |
| `open` / `_check_register_layout_locked` | PASS | geometry handshake必要 |
| `check_register_layout` / `transport_self_test` | PASS WITH DEBT | 仅tests/人工诊断消费；若保留应明确非runtime主路径 |
| `close` / `safe` / `_enter_safe` / `_drive_physical_safe` | PASS WITH DEBT | safety顺序集中正确；worker join超时后仍清owner需另做lifecycle风险审查 |
| `load` | REDESIGN | 唯一应执行target/clock/geometry/capacity checks的边界，目前缺失 |
| `write_slots` | DELETE/MERGE候选 | 无product consumer，只有README/notebook/tests；是`write_scan_table((row,))`的重复API，且held-point产品已刻意不用它 |
| `write_scan_table` | REDESIGN | silent coercion、32-bit wrap、sweeps不进AppliedState；每row重复O(edges) validation |
| `fire` | PASS WITH DEBT | runtime forever owner应保留；与`CompiledProgram.repeat_forever`重复，delay tail语义不清 |
| `wait_done` | REDESIGN | DONE不代表delayed outputs完成；`tail_elapsed`错误；依赖caller调用才清`_firing` |
| `cursor` | PASS | observer cache ownership清楚 |
| `describe` | PASS WITH DEBT | 应如实叫deployment description，不宣称target/clock全由板证明 |
| `snapshot` / `applied` | REDESIGN | snapshot含total scan_count但无sweeps/rows；applied含rows但无sweeps，两个都不完整 |
| `_observe/_finish_observation/_record_observer_failure/_refill/_scan_bank_arming` | PASS WITH DEBT | observer/refill owner正确；依赖私有sweeps truth，极快point仍受host refill cadence约束 |
| `_validate_slot_row` | REDESIGN | affine edge检查必要；未按slot kind限制duration/DAC，未检查bus timing/delay FIFO capacity |
| `_await_loaded/_read/_write/_strobe/_fire_took_effect/_require_*` | PASS | 硬件命令/transport boundary整体直接且有真实测试 |
| `_stop_worker` / safe-readback cache helpers | PASS WITH DEBT | owner明确；join timeout后线程仍活的可能性需在总lifecycle审计统一处理 |

### 4.7 `zlc_pulse/wire.py`

| 符号组 | 裁决 | 说明 |
|---|---|---|
| `build_fingerprint` / `CtrlWords` | PASS WITH DEBT | register geometry ABI核心；不证明target/clock，文档不可越界声称 |
| `_shipped_config_params/_geom` / `StreamerParams` properties | PASS WITH DEBT | offline defaults有用途；import-time silent fallback与runtime config warning是两条配置路径 |
| `_ceil/_pow2_at_least/_addr_width/region_bases/build_ip_sizes` | PASS（`build_ip_sizes` MOVE） | runtime address math留wire；IP size投影移fpga owner |
| `_to_unsigned/_checked_unsigned/_checked_signed/_from_unsigned/_field_words/_unfield/_pack_coeffs/_unpack_coeffs/_is_pow2` | PASS | wire bit math集中正确；不得让上层count先绕过checked path |
| `check_rtl_assumptions` | PASS WITH DEBT | static geometry guard必要；名称不可让人误以为它验证program delay FIFO capacity |
| `_bus_mode_value/_raise_mode/_bus_mode_name` | PASS | 小型wire enum codec |
| `scan_bank_words` | REDESIGN | modulo sweep实现必要；silent `max(1,int)`与32-bit total缺口 |
| `pack_program` | REDESIGN | 最终wire gate应检查loop_count、program clock/fraction/ABI相关事实及delay FIFO capacity；当前embedded table路径应删 |
| `unpack_program` | DELETE/MOVE TEST | 仅一个test consumer，不是产品readback；返回的也只是初始resident chunks，不是完整application |
| `FpgaPartProfile/part_profile/SolvedCapacity/_edge_ramb/_scan_ramb/estimate_resources/solve_capacity` | MOVE | 必要build tooling，但真实owner应是已有`fpga.py` |
| `_config_search_paths/_default_config_path/params_from_config/load_streamer_config/default_params` | PASS WITH DEBT | runtime配置需要；cwd/env/package三候选与import-time defaults并存，要避免同进程不同答案 |
| `default_clock_hz` | DELETE | 全仓无消费者 |
| `default_coeff_frac_bits/default_slot_mul_width/check_config_capacity/format_capacity_report` | MOVE/PRUNE | build入口使用者留`fpga.py`；不要从wire承担 |
| `emit_geometry_vh/emit_geom_tcl/_main`及其常量表 | MOVE | 明确build/CLI职责 |
| `pack_scan_rows` | MERGE/MOVE | runtime必要但应紧邻`scan_bank_words`；当前位于module main guard之后是历史残余 |

## 5. 关键调用文件裁决

| 文件/实体 | 裁决 | Pulse语义问题 |
|---|---|---|
| `zlc_atom/devices/sequencer/device.py::SequencerDevice` | PASS | 薄转发正确；core修复后不应在wrapper重复验证 |
| `zlc_atom/devices/simulation/sequencer.py::VirtualPulseStreamer` | REDESIGN | 私读`_scan_sweeps`、同步爆发全部world points、只延迟terminal |
| `zlc_atom/nodes/scan/plan.py` | PASS WITH DEBT | API parameter -> plan port清楚；plan float不是played canonical truth |
| `zlc_atom/nodes/scan/source.py::PublishedSignalSource` | REDESIGN | `arm(program, table)`直接丢弃参数；任意LIVE publication并不能证明一条publication对应一条played row |
| `zlc_atom/nodes/scan/dataset.py` | USER DECISION | 将plan sweep、shot、source repeat压为一个repeat轴，承认物理不同却丢失可分析坐标；需决定保留flatten还是显式sweep/shot坐标 |
| `zlc_atom/nodes/scan/seamless.py` | REDESIGN | S通过复制rows、R通过device sweeps；不处理模板repeat/cardinality；多份table复制有性能债 |
| `zlc_atom/nodes/stepped_scan/measurement.py` | REDESIGN | S覆盖whole bracket；重复判断full-span而不用model property；target全对象误比；settle发生在device tune之前而非之后 |
| `zlc_atom/nodes/calibration/pulse.py` | PASS WITH DEBT | API resolve owner清楚；target应交core ABI check；resolved actual metadata需统一 |
| `zlc_atom/nodes/calibration/task.py` | REDESIGN（详见04b/04c） | forever pulse与finite camera各自计数；pulse窗口/相机能力没有共同arm contract |
| `zlc_atom/nodes/slm_feedback/task.py` | REDESIGN（详见04c/05） | zero-width scan row充当shot counter是可行wire技巧，但模板repeat与camera cycles未统一验证 |
| `zlc_workbench/pulse_state.py` | PASS WITH DEBT | editor-only state合理；`scan_repeats=0`运行策略不应与model whole bracket暗中竞争 |
| `zlc_workbench/pulse_editor.py::_prepare_execution/_load_prepared/fire` | REDESIGN | UI scan repeat、whole bracket、finite/forever组合由多个布尔推断；invalid值仍有max coercion |
| `zlc_workbench/pulse_editor.py::sync_from_sequencer` | REDESIGN | 忽略slot_values/sweeps/forever，无法兑现Sync语义 |

## 6. 测试逐文件审查

### `zlc_pulse/tests`

| 测试文件 | 裁决 | 能守住什么 / 漏掉什么 |
|---|---|---|
| `test_model_compile.py` | PASS WITH DEBT | 静态/slot/loop基本波形有价值；未测delay slot假功能、clock mismatch、geometry mismatch、FIFO capacity、loop count上界；whole bracket测试把待裁决语义写成既定事实 |
| `test_scan_model.py` | PASS WITH DEBT | column单位/range/table sweep/held point覆盖好；同文件一边断言delay不能cycle到scan，一边从不测试constructor拒绝delay slot |
| `test_wire_device.py` | PASS WITH DEBT | register region、bank refill、rearm、applied生命周期强；`zero_slot...sweeps=5`反而锁定AppliedState漏sweeps；无invalid sweeps/overflow/ABI/clock/frac/FIFO测试 |
| `test_command_strobe.py` | PASS | SAFE/LOAD/FIRE strobe与ack风险测试实质强；不覆盖program semantic ABI |
| `test_remote.py` | PASS WITH DEBT | RPC/lifecycle/ownership很全；client/server都做同一int/clamp，使错误被镜像自证；无跨ABI/count语义测试 |
| `test_transport.py` | PASS | transport基础契约；不承担pulse语义 |
| `test_uart_transport.py` | PASS | UART framing/retry；不承担pulse语义 |
| `test_contract.py` | PASS WITH DEBT | 只钉signature，不能证明physical contract；容易阻碍必要API收敛 |
| `test_public_surface.py` | PASS WITH DEBT | export边界检查有用；不能把历史surface存在当必要性证据 |
| `test_manifest.py` | PASS WITH DEBT | XDC解析/ABI fingerprint有价值；没有守`load()`必须使用fingerprint |
| `test_fpga_assets.py` | PASS WITH DEBT | build asset一致性；只检查FIFO depth是配置/asset，不检查program是否超capacity |
| `test_import_purity.py` | PASS | import边界，与本问题正交 |
| `test_launcher.py` | PASS | launcher基本行为，与本问题正交 |
| `test_notebook_coverage.py` | PASS WITH DEBT | 只证明notebook提到API；当前反而维持无product consumer的`write_slots` |

### 跨包关键测试

| 测试 | 裁决 | 问题 |
|---|---|---|
| `zlc_atom/tests/test_stepped_scan_node.py` | PASS WITH DEBT | ScriptedScanBench按loaded loop count主动发布恰好S条，因此自证whole-bracket方案；不证明真实camera/source contract |
| `zlc_atom/tests/test_seamless_scan_node.py` | PASS WITH DEBT | fake按`R*P*S`预先生成publication，恰好复述被测手算；virtual e2e有科学值但仍受burst simulation与camera宽松行为污染 |
| `zlc_atom/tests/test_virtual_physics.py` | PASS WITH DEBT | 测terminal墙钟和最终frame数量；未测frame arrival cadence，因此virtual burst通过 |
| `zlc_workbench/tests/test_pulse_editor.py` | PASS WITH DEBT | 当前UI规则覆盖很多；没有外部`sweeps>1`或`write_slots`后Sync还原真相的产品测试 |

必须新增的最小红测试集合（实施阶段，不在本审查修改）：

1. compile/pack拒绝TTL和DAC delay FIFO overflow；
2. load拒绝target ABI、clock、coeff fraction mismatch；
3. PulseSequence拒绝delay PulseSlot；
4. sweeps/count拒绝bool/float/<=0/>uint32；
5. Virtual camera frame arrival按logical cadence，而非只测DONE；
6. AppliedState/Sync round-trip涵盖slot row、rows、sweeps、forever；
7. off-grid plan的Dataset坐标等于actual played value；
8. 每个使用pulse的Measurement验证camera windows/cycle cardinality或明确拒绝无法证明的source。

## 7. 文档—实现矛盾

| 文档/注释 | 声称 | 实现事实 |
|---|---|---|
| `packages/zlc_pulse/docs/contract.md` device表 | `write_slots`是“api slot” | 同文后面又明确API parameters永不进入device slot/table；当前API先bake后compile |
| 同contract的模型词汇 | target ABI在connect时比较 | `PulseStreamer.load()`从未比较；caller还用错误的全对象相等或完全不比 |
| 同contract的PulseStreamer签名 | ctor只有transport/geom/clock | 当前还强制`target=`；`write_scan_table`也多了未记录的`sweeps` |
| `packages/zlc_pulse/README.md` | `applied()`是last load/table/fire原样echo，可供GUI sync | 漏sweeps；Workbench又忽略slot_values/forever，不能round-trip |
| `model.whole_pulse_repeat` docstring | 无whole bracket表示until-stopped | Stepped/Seamless/SLM finite fire并非如此；forever属于caller |
| `engine_model.py`/RTL注释 | host validator阻止delay FIFO overflow | 当前无validator，探针已证明compile+pack后mirror丢边 |
| `ARCHITECTURE_DESIGN.md` virtual cadence | finite fire尊重logical duration，不能压进一个Monitor interval | world/camera events在`fire()`中瞬间全部生成，只延迟wait_done |
| `fpga.py`模块说明 | build emitters/resource tooling以该模块为明确owner | 实现仍全部在`wire.py`，`fpga.py`只是re-export |
| 根架构Camera段与Implementation附录 | 前者仍写`frame_0...N`，后者/代码为单一`frames + READOUT_EVENT` | 属于04b的更大文档矛盾，也会影响“W/F如何计数”的pulse API设计 |

## 8. 需要用户裁决的问题

以下不是用“需裁决”回避明确bug；PULSE-001到006的安全/正确性缺口无需产品投票。需要裁决的是目标语义：

### U-PULSE-1 — Whole bracket是否继续兼作shots（已有D-013）

1. **推荐：分开。** `RepeatRegion`永远是timeline内部bracket；shots是execution层相同point的program execution count；full-span只是一种region，不隐式决定forever。
2. 保持当前Stepped覆盖语义，但UI明示并严格验证一iteration一publication。
3. 删除Logic Node的shots字段，要求作者完全在pulse中表达。

审计推荐1；它与Seamless row重复、SLM zero-row shots可以收敛为同一物理词汇，也不会覆盖模板science。

### U-PULSE-2 — Scan Dataset是否保留sweep与shot身份

1. **推荐：保留可分析坐标。** plan sweep成为显式scan coordinate/point维度，repeat轴只承载shot × source repeat，或至少为flatten repeat提供结构化 `(sweep, shot, source_repeat)` labels。
2. 继续全部flatten到repeat axis，只在run metadata保存R/S。

当前代码自己承认sweep与adjacent shot受drift不同，却选择丢掉区别；若实验要分析drift，选项2不够。

### U-PULSE-3 — Off-grid API/plan值如何处理

1. **推荐：Start时materialize canonical actual，Dataset坐标用actual，metadata同时保存authored。**
2. 严格拒绝任何off-grid值。
3. 维持静默round且Dataset写authored（不建议）。

### U-PULSE-4 — Applied Sync的产品承诺

1. **推荐：保留Sync但只同步完整executable application truth**（rows/sweeps/forever/actual values），明确不会恢复API authoring declarations。
2. 删除Sync对authoring的反写，只提供只读board applied摘要。

### U-PULSE-5 — Delay DONE语义

1. **推荐：finite completion直到所有delayed TTL/DAC tail完成；Stop/safe仍可主动截断。** host可用compiled最大delay保守等待，长期则需要硬件tail-complete事实。
2. DONE只表示timeline engine结束，caller自行决定何时safe；则必须改名并把tail状态公开，不能继续让`wait_done`看起来是物理完成。

## 9. 建议的最小目标架构（不实施）

无需新manager/DTO层，现有owner内即可收敛：

```text
PulseSequence
  periods + internal RepeatRegion
  scan slots (duration/DAC only)
  API parameters (host-resolved only)
        |
        | resolve API -> canonical actual sequence
        v
compile_sequence(sequence, connected geometry/clock)
  -> CompiledProgram
     carries target ABI + clock + scan fraction facts
     contains NO rows / sweeps / forever
        |
        | load: one mandatory ABI/capacity gate
        v
PulseStreamer
  applied program
  explicit rows + sweeps + forever
  schedule queries consume the same rows/sweeps
        |
        v
Measurement execution
  plan repeats / shots translated once
  camera cycle cardinality validated once
  Dataset coordinates use played canonical values
```

其中最重要的约束是：

- bracket不再被Measurement偷偷改写；
- API parameter不再被叫作slot；
- rows/sweeps/forever不再藏在三个对象和私有字段；
- board ABI只在device load强制一次；
- camera expected cycles只从同一execution truth推导，不由每个Task手乘。

## 10. 本审查的验证记录

只运行隔离内存/virtual探针，没有连接硬件、没有修改代码或旧文档。每个Python探针先打印了repository root与实际`zlc_pulse.__file__`，确认测试的是当前workspace包。

已复现：

- delay slot row改变无任何schedule效果；
- target ABI mismatch仍可load；
- label-only target变化会被caller全对象比较误拒；
- sequence tick与compile clock不一致会静默改变实际时间；
- API off-grid duration静默round；
- invalid sweeps静默变1；
- LOOP_COUNT/SCAN_COUNT超32bit静默wrap；
- compile geometry fraction与device geometry不同仍load；
- delay FIFO overflow compile/pack成功但RTL mirror相对reference丢33 ticks；
- Virtual 5×20ms points在约1.6ms内全部进入world，只延迟terminal。

这些探针足以证明根因，不需要全套测试来增加“通过数字”。
