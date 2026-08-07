复验全部完成,两仓确认只读未污染(zlc_pulse porcelain 0 行;参照树仅 1 个 2026-07-22 遗留 untracked 文件,非本次产生)。

# zlc_pulse R1-R6 返工复验报告(2026-08-03)

**结论:小修**(R1-R6 全部真实落地,六探针逐一变红,R1 跨树实跑逐字相等;仅剩 wire.py 三处一行级 stale 注释残留,属 R6 声称"清理完成"的漏网)

所有实验在临时副本 `scratchpad/zp_rv` 上进行(`zlc_pulse.__file__` 断言解析到副本);基线 **29 passed, 0.44s**;每个探针后还原并复绿。

## 1) R1 终审(实跑,EXP5/EXP2/EXP5b 复现)— PASS

脚本 `scratchpad/r1_final.py`,树侧 `image.py` 按旧模型手工独立推导 carrier(不抄新侧输出):

| 实验 | 结果 |
|---|---|
| EXP5:仅声明 `da0=-40ns` + 两条被驱动 TTL | `channel_delays=(2,2,0×11)`,`bus_delays=()` — 被驱动 TTL 全部 +2 tick、bus 0 ✓;与树侧 pack **逐字相等** |
| EXP2(上一轮 1 字差程序)重跑 | addr 38720 两侧均 `0x2`;118 字稀疏表**地址集对称差空、值差空**,上一轮 1 字差已消失 ✓ |
| EXP5b:`ttl_b` 声明 0 延时 vs 未声明 | `channel_delays`/`bus_delays` 与 EXP5 完全相同 — 相对对齐不再随"是否声明"漂移 ✓ |

修法与 R1 要求一致:`compile.py:350-372` 折叠集合 = 被驱动 lane ∪ 被驱动总线 ∪ 声明集(`raw_lane.setdefault(index, 0)`)。锁死测试真实:**E5 探针**(删掉 driven-fold 四行)→ 仅 `test_negative_bus_delay_shifts_every_driven_ttl_lane` 红(`tests/test_model_compile.py:84-98`,同时覆盖 EXP5b 半边)。

## 2) 六个绿探针复验(R4①-⑥ 对应 P1-P6)— 全部变红

| 探针 | 突变位置 | 结果 |
|---|---|---|
| P1 wrap 重启点 `nxt_idx = 1 if last` | `engine_model.py:596` | **红**:`test_scan_table_wrap_is_gapless_for_forever_program`(`test_model_compile.py:120-136`,`repeat_forever=True` 场景已真踩,:129) |
| P2 `scan_bank_words` 每值 +1 | `wire.py:535` | **红**:`test_pack_slot_scan_image_matches_frozen_byte_baseline`(`test_wire_device.py:84-99`,新增含 slot+scan 区的第二条冻结字节基准) |
| P3 `open()` word63 拒绝改 `if False` | `device.py:110` | **红**:`test_open_rejects_mismatched_word63`(`test_wire_device.py:177-183`,负路径 + `transport.closed` 断言) |
| P4 `_refill` 首行 raise | `device.py:374` | **红**:`test_observer_refills_a_freed_scan_bank`(`test_wire_device.py:225-242`,`_AdvancingMemoryTransport` cursor 真推进,断言 `(BANK0_CHUNK,2)` 补给写;`_refill` 不再是死代码) |
| P5 `fire()` 主线程加 STATUS 读 | `device.py:219` 前 | **红**:`test_wait_done_uses_observer_owned_terminal_double_reads`(`test_wire_device.py:186-204`,`read_log ==` 恰好 4 次 observer 读的**全量正面断言**,比上轮尾部检查强) |
| P6 删 SAFE 第一相 | `device.py:265` | **红**:`test_safe_readback_uses_stable_status_and_zero_clock_mask`(`test_wire_device.py:149-174`,`write_batches[-3:]` 锁三相**写序列**非仅终值) |

每个探针恰由其对应新增测试单独抓住(1 failed, 28 passed),还原后 29 复绿。

## 3) R3/R5/R6 核实

- **R3 import 守卫**:`glob`→`rglob`(`test_import_purity.py:58`);serial 处置 = `OPTIONAL_TOP_LEVEL={"serial"}` 豁免注记(:13-15)+ pyproject `uart = ["pyserial>=3.5"]`(`pyproject.toml:16`)。**E4 探针**:向 `transport/uart.py` 注入 `import rpyc` → 3 测试红,子包逃逸已封 ✓
- **R3 几何锚**:选了钉住测试路线 —— `test_default_geometry_is_pinned_to_deployed_word63`(`test_wire_device.py:64-66`,`== 0x5AFC7CFB`)。**E1 探针**:`evt_fifo_depth` 缺省 64→65 → 该测试点名红 ✓(未 vendor 部署 JSON,GOAL R3 写明"或至少",合规)
- **R5 写口边序校验**:已实做非豁免 —— `device.py:351-372` `_validate_slot_row`(`evaluate_affine_tick` 逐行严格递增 + loop 元数据),`write_slots`(:161)与 `write_scan_table`(:187)都调;**E2 探针**(改 no-op)→ `test_runtime_slot_rows_reject_colliding_affine_edges` 红(`test_wire_device.py:136-146`)✓
- **R6 `_refill` 异常入 observer except**:`device.py:316-321` `_refill` 在外层 try 内,失败 → `_record_observer_failure`(:324-329)→ 终态 `STATUS_ERROR`、`wait_done` 不再悬置;**E3 探针**(吞掉 refill 异常)→ `test_observer_refill_failure_becomes_terminal_error` 红(`test_wire_device.py:245-259`)✓
- **R6 stale 清理**:350930c 已清 CLI prog 名(`wire.py:1314` = `python -m zlc_pulse.wire`)、image.py 路径注释、跨包引用措辞。**残留 3 处(即小修清单)**。

## 4) R4① 树内 late-refill ScanUnderflow 回收 — 已回收

`test_scan_table_late_refill_raises_underflow`(`test_model_compile.py:139-159`):`refill_delay=20, raise_on_underflow=True` → `ScanUnderflow`,与树资产 `tests/test_zlc_pulse_trigger_schedule.py::test_streamed_scan_is_gapless_when_ready_and_reports_a_late_refill` 同参数同语义;gapless 半边在 :101-117,forever-wrap 半边在 :120-136。

## 5) 其余机械判据复核

- 负面 grep 8 模式(`trigger_schedule/expected_trigger_counts/visible_ports/scan_sweep_count/PulseExecutionForm/rpyc/sha256_text/evidence`)src/ 全 0;`zlc-data` 全仓 0(R2 依赖已删,`pyproject.toml` 仅 numpy+pyserial extra)。
- contract.md 参数名已改 `sequence/program`(`docs/contract.md:10,12`),`test_contract.py:8-12` 签名机械核对绿。
- src LOC 实测 **5,063** ≤ 6k。
- `notebooks/usage.ipynb` 副本逐 cell 复跑全 OK(cell 4:`done_status_reads=(4,4), safe_stable=True`)。

## 小修清单(精确到行,均一行级注释修补)

1. `src/zlc_pulse/wire.py:156` 与 `:183` — 引用不存在的 `test_streamer_params_defaults_match_config`;实际锚是 `test_default_geometry_is_pinned_to_deployed_word63`,改名或删引。误导维护者以为另有守卫,恰是 R6 想清的那类 stale。
2. `src/zlc_pulse/wire.py:541` — docstring "Pack a RuntimeSequenceProgram" 是旧树类型名,本仓类型为 `CompiledProgram`。
3. `src/zlc_pulse/wire.py:1147` — 注释仍指旧仓路径 `Zou_lab_control._streamer_geometry`。

**附注(非缺陷,记录在案)**:350930c 清理了 `emit_geometry_vh` 头部注释,新仓 .vh 输出与树内已提交 `fpga/pulse_streamer/zlc_geometry.vh` 不再逐字节相等 —— 实测差异仅 3 行注释,宏值(非注释行)全等,`geom.tcl` 仍逐字节相等,指纹不变 0x5AFC7CFB;将来若用本仓再生树内 .vh 会有注释级 churn。另:上轮债项 7(engine_model mirror 家族本仓零调用)不在 R1-R6 范围,仍未记入 GOAL 阻塞记录,可顺手补一行。