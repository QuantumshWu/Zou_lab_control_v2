# 任务3 验收报告:测试 / notebook 质量

**结论:小修**(核心纯函数面守卫真实有效,三处指定突变全被抓、notebook 全绿;但 Q7③"wrap"名不副实、word63 握手拒绝无测试——这两条直接触碰 GOAL 终态判据 3 的字面意图;device 动态路径大面积零覆盖)

所有实验均在临时副本 `scratchpad/zp_audit` 上进行(PYTHONPATH 指向副本 src,已验证 `zlc_pulse.__file__` 解析到副本;被验收仓与参照树零写入)。基线:20 passed(0.29s)。

---

## 1) Q7 三条等价性测试逐条审实质

**① write_slots 单行表 vs 静态编译 —— 存在,实质合格但样本极小**
- 位置:`tests/test_model_compile.py:70` `test_write_slots_single_row_matches_static_waveform`。
- 强度:`reference_play(runtime_row, 12) == reference_play(static, 12)` —— `reference_play` 返回逐 tick 的全通道 mask 整词(`engine_model.py:302-356`),即**逐周期、全 62 通道整词相等**,非抽样。合格。
- 弱点:被测程序只有 3 period/1 slot/duration kind;DAC 值 slot、delay slot、loop_count>1、channel_delays 均不在等价环内。且这是纯 engine_model 等价——**设备侧 `write_slots` 打包出的寄存器数值不在环内**(见 P2)。

**② 换 scan 表不重传边表且波形正确 —— 拆成两半,中间数值编码缝隙无守卫**
- "不重传边表":`tests/test_wire_device.py:92` `test_write_scan_table_changes_only_scan_regions`,断言写地址不落 tick/coeff/mask 区、落 scan 区。但断言与实现共用同一 `region_bases`(`wire.py:319`)——同源自证;M2 突变证实 region 漂移只有字节基准能抓。
- "波形正确":`tests/test_model_compile.py:82` 前半,`replace(program, scan_points=...)` 后 80 tick 逐周期比对 `reference_play`。强度够。
- 缝隙:两半之间的**scan 行数值编码**(`wire.py:507` `scan_bank_words` 的定点/窄化/word 布局)零测试——`test_pack_scan_rows_only_targets_the_requested_bank_chunk`(`test_wire_device.py:145`)只断言 `set(words)` 的**键**不断言值;冻结字节基准(`test_wire_device.py:59`)用的是无 slot 程序(`SCAN_COUNT==0`),scan 区不在基准内。**探针 P2:把 `scan_bank_words` 的每个值 +1(`wire.py:537`),20 测试全绿。**

**③ scan 无缝 wrap —— 名不副实,wrap 分支从未执行**
- 位置:`tests/test_model_compile.py:82` `test_scan_table_is_data_and_engine_wrap_is_gapless`。
- 事实:compile 恒产出 `repeat_forever=False`(`compile.py:435`),测试也未 replace 它;`streaming_scan_play` 的 wrap 分支(`engine_model.py:596` `nxt_idx = 0 if last`、`engine_model.py:600` `base ^ wrap_toggle`)只在 `repeat_forever` 下可达。5 点/bank 2 在末点直接 `running=False`(`engine_model.py:593-594`)。
- **探针 P1:把 wrap 重启点改坏(`nxt_idx = 1 if last`),20 测试全绿** —— 该测试实际证明的是一次 sweep 内 chunk 流式换页(0→1→2)无缝,即 docstring 里"CONTINUOUS CYCLIC PING-PONG"(`engine_model.py:519-526`)的 wrap 半边完全没踩到。断言形式本身(80 tick 逐周期 + `stalled`/`points_played`)是够的,但目标分支是死路径。
- 对照:旧树原资产 `tests/test_zlc_pulse_trigger_schedule.py:237` 同场景还带 late-refill `ScanUnderflow` 半边(`refill_delay=20, raise_on_underflow=True`),迁移时被丢弃。

## 2) device 测试覆盖核对

| 应测项 | 有无 | 证据 |
|---|---|---|
| open 握手拒绝(word63 不符) | **无** | 探针 P3:把 `device.py:110` 的拒绝改成 `if False:`,20 全绿。仅有指纹逐字段契约(`test_wire_device.py:48`),握手**拒绝**行为零测试——GOAL 判据 3 "word63 握手…在" 只满足一半 |
| load/fire/wait_done 全路径 | 有(静态半) | `test_wire_device.py:124`:open→load→fire→wait_done,断言 status/cursor 双读 + `read_log[-4:]` 尾序。但 `auto_done=True` 使 observer 首轮即终结:RUNNING 期、timeout→None、underflow、`fire(forever=True)` 全未踩 |
| 三相 SAFE | 有(终值半) | `test_wire_device.py:109` 断言 stable/(0,0)/clk 全零。STATUS 毒写设计让断言有实质(M3 被抓);但只锁终值不锁序列——**探针 P6:删掉第一相 `_safe_phase()`(`device.py:262`),20 全绿** |
| observer 所有权(FIRE 后主线程不碰 STATUS) | 部分 | 仅 `read_log[-4:]` 尾部间接约束终端双读来自 observer。**探针 P5:在 fire() 里主线程加一次 STATUS 读(`device.py:220` 前),20 全绿**。无"FIRE 至 done 期间主线程零 STATUS/CURSOR 读"的正面断言 |
| streaming 补给(_refill ping-pong) | **无** | **探针 P4:把 `device.py:342` `_refill` 首行改 raise,20 全绿**。memory transport cursor 恒 0 且 auto_done 秒终,`_refill` 在全套件是死代码 |

## 3) mutation 抽查 3 处(副本上,改后已对照还原,末次基线 20 passed 复绿)

| 突变 | 位置 | 结果 |
|---|---|---|
| M1 仿射求值符号翻转(`+` → `-`) | `engine_model.py:104` `effective_tick` | **红 ×2**:`test_slot_compile_changes_only_affine_data...` + `test_write_slots_single_row...`(2 failed, 18 passed)✓ |
| M2 pack region base 改坏(scan 基址 +1) | `wire.py:331` `region_bases` | **红 ×1**:仅 `test_pack_sparse_image_matches_frozen_byte_baseline`(sha256 不符)。两条 region 边界测试因与实现同源 `region_bases` 自洽保持绿——冻结字节基准是唯一防线,幸而它真咬人 ✓ |
| M3 SAFE 序列改坏(删 `CMD_SAFE` 写) | `device.py:338` `_safe_phase` | **红 ×1**:`test_safe_readback_uses_stable_status_and_zero_clock_mask`(毒写 STATUS 未被清)✓ |

三处指定突变全部被现有测试抓住;但补充探针 P1-P6 中 **6 个语义破坏突变全绿**(wrap、scan 值编码、open 拒绝、_refill、所有权、SAFE 第一相),这就是 20 测试对 5k 行"偏薄"的实底。

## 4) notebooks/usage.ipynb 实跑

- venv 无 jupyter/nbclient/nbconvert/matplotlib——ipynb 无法以 notebook 形式机械执行;我用忠实复现 Jupyter 语义(共享命名空间、尾表达式求值)的 runner 在副本上逐 cell 跑:**cell 1、3、5、7 全 OK,退出码 0**;cell 1 的 sys.path 自举正确使副本 src 优先。
- 输出:cell 3 `((0,0,1),(1,0,0),('duration',))`;cell 5 `first_value_waveform=[1,0,0,0,...]` vs `wider_value_waveform=[1,1,1,0,...]`;cell 7 `done_status_reads=(4,4), safe_stable=True`。
- "看波形+write_slots 改值再看"评定:**半满足**。"波形"=打印 mask 整数序列(无图,离线台架可接受);改值前后波形对比确实有(值 1 vs 3,cell 5)——但走的是 model `replace(scan_points=...)`;cell 7 的 `streamer.write_slots((3,))` 之后**没有从设备寄存器侧导出波形再看**,而设备打包数值恰是 P2 证实的无守卫缝。GOAL 判据 4 字面(顶到底无错)满足。

## 5) 按 GOAL 应有而缺的测试清单

GOAL 点名四项核对:
- pack 字节等价基准:**在**(`test_wire_device.py:59`,M2 证明有效)。缺口:基准只覆盖无 slot/无 scan 程序,slotted+scan 区编码需第二条基准。
- 指纹覆盖全几何字段契约:**在**(`test_wire_device.py:48`,逐字段扰动断言,强)。
- uart 帧协议:**半**。`uart_frame` 往返+CRC+coalescing 在(`test_transport.py:7,20`);`UartRegisterTransport`(220 行,`transport/uart.py:96`)零测试——缺假 `UartLink` 下 write_words→coalesce→帧→应答匹配全路径。
- 4KB burst 切分:**在**(`test_transport.py:30`,断言 0FFC/1000 切分)。

真正缺失(按危害排序):
1. `open()` word63 不符**拒绝**测试(P3;判据 3 字面要求)。
2. wrap 真覆盖:`repeat_forever=True` + `streaming_scan_play` 跨 sweep 边界逐周期比对(P1;判据 3 的 ③ 实质未成立)。
3. scan 行数值编码守卫:`pack_scan_rows`/`scan_bank_words` 值级字节基准或 pack→decode 往返(P2)。
4. `device._refill` 流式补给(cursor 推进的假 transport)+ underflow/timeout/forever 分支(P4)。
5. 三相 SAFE 序列结构断言(写序列而非仅终值,P6)+ observer 所有权正面断言(FIRE 期间主线程零 STATUS/CURSOR 读,P5)。
6. late-refill `ScanUnderflow`(旧资产 `test_zlc_pulse_trigger_schedule.py:250-257` 已有,迁移被丢)。
7. engine_model 迁入的 mirror 家族(`prefetch_play`/`rtl_mirror_play`/`rtl_mirror_play_stale_seed`/`bus_play`/`bus_value_at`/`rtl_bus_segment_delay_mirror`/delay-line 家族,约 700 行)在本仓**零调用**——旧树等价测试(`trigger_schedule.py:202/221/260/272`)未迁,这些函数 docstring 里"proven equal"的证明在本仓已无守卫。

**小修最小集建议**:补 1+2(判据 3 兑现)、3(② 的缝)、6;4/5/7 可作为后续债记录进 GOAL 阻塞记录。