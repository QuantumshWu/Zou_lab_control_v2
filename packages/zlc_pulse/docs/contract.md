# zlc_pulse 对外契约(跨仓唯一权威)

> 本文件是并行仓(zlc_atom 的 sequencer 侧等)写 fake 的唯一依据。**任何签名变更必须先改本文件**。签名来源:docs/survey-pulse-fpga-2026-08-02.md §3(最小 host API)。宪章:脉冲文档显式区分 scan slots 与 host API parameters；API 参数必须先解析为普通物理字段，设备只接收 resolved program 与 scan table。设备不替编排数拍子,但保存的最后应用原样记录可被主动询问用于 GUI sync。

## Runtime export surface

The package-level `zlc_pulse.__all__` is intentionally one-to-one with this
runtime list. FPGA emitters and capacity estimators live under `zlc_pulse.fpga`
and are not package-level runtime exports.

```
__all__ = (
    "PulseStreamer",
    "RemotePulseStreamer",
    "connect",
    "serve",
    "PulseSequence",
    "PulseApiParameter",
    "PulsePeriod",
    "AnalogStep",
    "PulsePortSpec",
    "PulseTarget",
    "PulseSlot",
    "PulseFieldRef",
    "OutputDelay",
    "MINIMUM_REPEAT_COUNT",
    "PULSE_TREE_FORMAT",
    "sequence_from_tree",
    "sequence_to_tree",
    "RepeatRegion",
    "compile_sequence",
    "pulse_target_from_xdc",
    "load_streamer_config",
    "UartRegisterTransport",
    "VivadoAxiRegisterTransport",
    "MemoryRegisterTransport",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT",
    "TIME_UNIT_CHOICES",
    "TIME_UNIT_TO_NS",
    "ANALOG_MODE_CHOICES",
    "align_to_grid",
    "cycle_binding_kind",
    "resolve_scan_point",
    "resolve_api_parameters",
    "pulse_field_value",
    "api_parameter_columns_for",
    "scan_columns_for",
    "scan_table_template",
    "validate_scan_table",
    "scan_rows_to_wire",
    "scan_rows_from_wire",
    "RemoteError",
    "UartError",
    "BackendResolutionError",
    "__version__",
)
```

Wire commands, status words, packing helpers, geometry fingerprints, and RTL
assumption checks remain available from `zlc_pulse.wire`; build emitters and
capacity tools remain under `zlc_pulse.fpga`. They are deliberately not device
package exports.

`StreamerParams`, `CompiledProgram`, `RegisterTransport`, `MemoryRegisterTransport`,
`AppliedState`, `DoneReport`, `SafeReadback`, and `trigger_times` remain real
implementation objects in their owning submodules. They are deliberately not
top-level user-construction or user-catch names; the package surface above is
the only package-level contract.

## X0.5 hardware protocol audit (migrated from v1)

The source of truth for this table is the read-only v1
`zlc_pulse/transport/session.py`.  The current device layer does not infer a
new protocol from the RTL: the command strobe, SAFE acknowledgement, image
ordering, and bank arming below are the migrated sequences.  `COMMAND=0` is
shown explicitly because the frozen RTL detects a rising edge and does not
self-clear the command register.

| v1 operation / hardware I/O | current `PulseStreamer` sequence | result |
|---|---|---|
| `start()` then `check_register_layout()`; read `CTRL[63]` | `open()` starts the transport and reads `CtrlWords.LAYOUT_ID`; `check_register_layout()` repeats the same fingerprint check on demand | same; the public diagnostic is now present without adding a remote RPC |
| `transport_self_test()`; write/read/clear CTRL scratch | `transport_self_test()` writes a bounded `0xC0DE0000 + index` pattern, reads it back, and clears it in `finally` | same scratch-only diagnostic; it is intentionally not run by `open()` because v1 did not make it part of startup and it is a write side effect |
| `prepare()` before image upload: if prior state is not SAFE, `_drive_physical_safe()` | `load()` proves cached SAFE or runs `_drive_physical_safe()` before the first image word | same physical precondition; no image is written after an unacknowledged SAFE |
| first SAFE: write `(STATUS, STATUS_ERROR)`, `(COMMAND, 0)`, `(COMMAND, CMD_SAFE)`; poll STATUS until two adjacent zero reads; retry the zero-plus-SAFE strobe once; timeout is `TimeoutError` | `_enter_safe()` uses that exact write order, adjacent-zero rule, one retry, and deadline | same; `safe()` no longer returns `SafeReadback(stable=False)` on failure |
| disable clock muxes: write every `CLK_ENABLE` word to zero, read every word back | `_drive_physical_safe()` writes and verifies every `CtrlWords.CLK_ENABLE + index` | same |
| final SAFE after clock disable | `_drive_physical_safe()` issues the same acknowledged SAFE sequence again | same; returned `SafeReadback.status_reads` is the final `(0, 0)` pair |
| upload and LOAD: ordered image, `BANK_READY=0b11`, `(COMMAND, 0)`, `(COMMAND, CMD_LOAD)`; wait for `STATUS_LOADED` or error/timeout | `load()` sends one ordered batch in that order and `_await_loaded()` polls `STATUS_LOADED`/`STATUS_ERROR` with the five-second loader budget | same hardware order and handshake; the local API has no v1 operation-cancellation token to pass through |
| `fire()`: every v1 shot follows a fresh `prepare()`/LOAD; then start the observer behind a gate, write `BANK_READY=0b11`, zero-plus-FIRE, and release the gate | The first shot uses `load()`'s LOADED state. After FIRE/DONE or SAFE has cleared it, `fire()` reissues only zero-plus-LOAD against the already-resident image, waits for `STATUS_LOADED`, then starts the gated observer and sends the ordered repeat/bank/zero-plus-FIRE write | same per-shot loader boundary and observer race protection without recompiling or retransmitting the edge/bus image; repeated table/fire points cannot be silently rejected by the RTL's LOADED gate |
| observer terminal reads: STATUS/CURSOR, then a second STATUS/CURSOR pair | the gated observer owns all post-FIRE STATUS/CURSOR reads and `wait_done()` returns the two-read facts | same read ownership and terminal double-read; JTAG/UART cadence remains transport-declared |
| v1 `clear_host_config()` before prepare | no separate public clear operation; `pack_program()` writes the complete host-owned image fields (including all delay and clock words), and `load()` first drives physical SAFE | not needed as a second operation: there is no partial-config API that can leave a previous program field behind |
| `safe_state()` / `close()` | `safe()` runs the full physical sequence; `close()` runs `safe()` before closing an opened transport | same safety-before-release rule; cached verified SAFE may be reused exactly as v1's cached safe readback |
| v1 optional batched reads and operation-stop tokens | local `RegisterTransport` requires ordered writes and single-word reads; safety writes/readbacks receive a deadline, while the minimal public API has no cancellation method | transport capability difference only; it does not change any register address, write order, acknowledgement rule, or read fact |

The write-sequence replay in `tests/test_command_strobe.py` and the RTL-gated
transport tests in `tests/test_wire_device.py` cover the zero-plus-command rule,
pre-LOAD SAFE, SAFE retry, clock readback, loader acknowledgement, per-shot
resident-image reload, and scan-bank re-arming. Any future hardware-facing
change must update those tests and this table together.

## 纯函数层(无 I/O)

```
config = load_streamer_config()
compile_sequence(sequence, config["params"], config["clock_hz"]) -> CompiledProgram
    # 产物含边表/仿射系数/总线段/延时/clk_enable + slot 声明;不含 scan 表、不含任何触发调度
pack_program(program, params=None) -> dict[int, int]             # (addr, word) 稀疏表; wire image serializer
pack_scan_rows(rows: ndarray, geom, bank, chunk, sweeps=1) -> dict[int, int]
trigger_times(prog, channel: str, table: ndarray | None) -> ndarray
    # 触发时刻推导纯函数;编排层自取自算,设备/协议零参与
```

## 设备层

```
class PulseStreamer:
    __init__(transport: RegisterTransport, geom: StreamerParams, clock_hz: float)
    open()                      # 读 CTRL[63] 比对 build_fingerprint(geom),不符即拒
    close()
    load(prog: CompiledProgram, *, source: PulseSequence | None = None)
                                # source 原样保存;上传边表/系数/总线/延时 + LOAD
    write_slots(values: Sequence[int])        # api slot:一行表写 bank0 行 0;不重编译不重传边表
    write_scan_table(rows: ndarray)           # scan slot:预载两 chunk,长表内部 observer 流式补给;换表不重编译
    fire(*, forever: bool = False)
    wait_done(timeout: float | None) -> DoneReport | None
    cursor() -> int | None                    # 非阻塞进度(scan 点游标)
    safe() -> SafeReadback                    # 三相 SAFE;{status==0 稳定两次, clk_enable 全 0} 读回
    snapshot() -> dict
    applied() -> AppliedState | None          # 主动询问的最后应用原样记录;不提供同步对账信息

DoneReport  = (status 双读, cursor 双读, underflow, tail_elapsed)   # 纯读回事实,非对账凭证
SafeReadback = 物理安全读回事实
AppliedState = (program, source, slot_values, scan_rows, forever, loaded_at)
    # frozen snapshot; load sets program/source/loaded_at, writes replace the active table form,
    # fire records forever, safe/wait_done preserve it, and close clears it.
RegisterTransport 协议 = 寄存器字读/写(axi / uart / fake 字典皆可实现)
```

## XDC board mapping

```python
pulse_target_from_xdc(path=None, config_path=None) -> PulseTarget
```

The default path is `fpga/board_config/board.xdc`. The parser preserves the
declaration order as `ch00`, `ch01`, …; each bare output is one digital port,
each `name[index]` family is one DAC port ordered by numeric index, and each
numbered `*_clk` output is the corresponding DAC latch clock. Structural
control outputs (`clk`, UART, reset/status, LEDs, and ground names) are not
pulse lanes. No board signal-name table is embedded in the package. The
returned target exposes `target.package_pins[lane]` for operator diagnostics.
Loading also checks lane count, DAC bus count, and DAC width against
`streamer_config.json` and raises a value-bearing error on any mismatch.

## 分离 remote(薄门面)

`RemotePulseStreamer(host, port)`（或 `connect(host, port)`）与本地 `PulseStreamer` 共享上面十一方法的
调用签名。server 进程只持有一个真实 `PulseStreamer`(UART/AXI/测试 memory
transport),逐一转发请求,不复制 applied/运行状态,不使用 rpyc。

协议是长度前缀 JSON 的一问一答平明 tree；`CompiledProgram`、`PulseSequence`
与 `AppliedState` 仅按公开字段重建,不经过握手 digest/canonical 家族。server
端 `wait_done` 永远是零等待探针,client 端用 `snapshot()`/`cursor()`/短
`wait_done` 轮询组合出同名 API,所以 `safe()` 可在 forever 运行中从同一连接立即到达。
TCP 连接承载单客户端所有权；断连自动 SAFE 但保留 server 进程内的 applied
记录,显式 `close()` 才清记录。server 重启后 applied 为 `None` 而寄存器可能仍
为 LOADED,处置是重新 `load`；文件持久化不在本契约内。
remote 客户端额外提供且仅提供 `{disconnect, __enter__, __exit__}` 三个连接管理方法；
它们不是设备方法集合的一部分。

`serve` is the server-side entry point; `PulseRemoteServer` remains its
implementation in `zlc_pulse.remote`. The alternate UART/AXI constructors are
the concrete backend-selection surfaces. The notebook uses those public
constructors only to show the local choices, and uses the real remote endpoint
for device operations. `RemotePulseStreamer` has the eleven device methods;
only `disconnect`, `__enter__`, and `__exit__` are remote-only lifecycle helpers.

## 模型词汇

`PulseSequence` 包含 periods(digital states + `AnalogStep`)、delays、repeat、scan-only `slots` 与 host-only `api_parameters`。`PulseApiParameter(parameter_id, field_ref, unit)` 与 `PulseSlot` 共用唯一 ID namespace；同一物理 field 只能属于其中一种。`resolve_api_parameters(sequence, values)` 将全部 API 参数写回普通物理字段并删除声明；省略 `values` 时使用文档当前值。`compile_sequence` 拒绝任何尚未解析的 API 参数。CompiledProgram carries the edge table, affine coefficients, bus descriptors, and scan-slot schema; table rows remain writable scan data.
`PulseTarget`:逻辑口→lane 映射 + `abi_fingerprint`(connect 时一次相等比较)。
`AppliedState.source` 是 owner 提供的 resolved `PulseSequence` 原样引用,设备不解释、不编排；`slot_values` 与 `scan_rows` 只描述 scan slots，API 参数永不进入设备 slot/table 宽度。

## 负面清单(永不提供)

- `trigger_schedules` / `expected_trigger_counts` / 任何"应收帧数"回执——设备不替编排数拍子。
- 逐点 fire API(`fire_point` 类)/ 逐点对账 / arm-bind-finish 协议。
- `PulseExecutionForm` 五态(有无表 × forever 两布尔足够)、`visible_ports`、`scan_sweep_count`、`scan_recipe`。
- session/epoch/superseded 仲裁层(单客户端所有权由连接生命周期承载)。
- rpyc/双连接(瘦 remote 门面后置,做时逐一转发无第二状态机)。
- `applied()` 不是设备回传 trigger 调度、应收帧数或逐点对账;它只回声设备保存的最后应用原样记录。
