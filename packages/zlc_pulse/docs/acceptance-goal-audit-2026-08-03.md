# zlc_pulse 验收审计报告(任务2:GOAL/宪章/契约逐项)

**结论:小修**(主体合格:20 测试独立复跑全绿、notebook 顶到底可执行、负面 grep 实测全零、LOC 5,019≤6k、宪章四能力与两条真机纪律均实在;但契约参数名漂移、import 守卫覆盖洞、死依赖三处需改,均为小时级修复)

复验方式:被验收仓与参照树只读;测试与 notebook 在系统临时目录副本上用仓内 venv 复跑(`PYTHONDONTWRITEBYTECODE=1`),`20 passed in 0.25s`;notebook 4 code cell 顺序 exec 全过。

---

## 1) Q0–Q8 逐项核对

| 项 | 实况 | 证据 |
|---|---|---|
| Q0 引导 | ✅ src 布局/pyproject(发行名 zlc-pulse, numpy+zlc-data, dev pytest)/import 纯度守卫/顶层 allow-list/contract.md 预写并校订 | `pyproject.toml:6-16`;`tests/test_import_purity.py:9-12,69-81`;`docs/contract.md` | 
| Q1 model.py | ✅ 统一 PulseSlot(kind/field_ref/unit,kind∈duration/dac/delay);PulseTarget(lanes/ports/abi_fingerprint);无 visible_ports/scan_sweep_count/编辑器默认常量(grep `MIN_REPEAT\|DEFAULT_` 零命中) | `model.py:275-303`(PulseSlot)、`169-247`(PulseTarget)、`30`(SLOT_KINDS) |
| Q2 compile.py | ✅ 边表/仿射系数/总线段/延时/repeat;`compile(sequence, geom, clock_hz)` 无 trigger_channels/grouping(grep 零);产物 `scan_points=()` 表恒为数据 | `compile.py:367-454`;`compile.py:442`;`tests/test_contract.py:9` |
| Q3 wire.py | ✅ image.py 并入(diff 实证:除 `_checked_unsigned/_checked_signed` 严格化、`pack`/`pack_scan_rows`/`StreamerGeometry` 追加、"final evidence"→"final note" 措辞外逐行同源);word63 build_fingerprint 原样、哈希单源结构注释在 | `wire.py:83-95`(fingerprint)、`1364`(别名)、`1367-1392`;严格化 diff 见 `wire.py:372-386,593,597` |
| Q4 device.py | ✅(详见 §2) | `device.py` 全文 394 行 |
| Q5 transport/ | ✅ RegisterTransport 协议+AXI+UART(3Mbaud 帧协议 uart_frame.py+CRC)+lease;**AXI 4KB 边界切分保留且有真守卫测试**(1023/1024 字地址切分断言);参照树 session.py(1,128 行状态机)正确未搬 | `transport/axi.py:20,172`;`tests/test_transport.py:30-44,47-61` |
| Q6 纯函数面 | ✅ `schedule.py::trigger_times` 独立模块、纯函数、device/transport 零引用(schedule 只 import numpy+compile;device.py 不 import schedule);enumerate_pulse_params 为条件项("若归属此包"),未落户,grep 全仓零引用,可接受 | `schedule.py:1-104` |
| Q7 免重编译三证明 | ✅ 三条测试存在且绿:① 单行表 vs 静态编译逐周期等价 `test_write_slots_single_row_matches_static_waveform`;② 换表只写 scan 区不碰 tick/coeff/mask 区 `test_write_scan_table_changes_only_scan_regions` + 波形正确 `test_scan_table_is_data_and_engine_wrap_is_gapless`;③ 无缝 wrap 同测试(bank_size=2、5 点,ping-pong 跨 wrap,stalled=False) | `tests/test_model_compile.py:70-98`;`tests/test_wire_device.py:92-106` |
| Q8 测试+notebook+README | ✅ 20 测试(1+3+5+4+7)复跑全绿;pack 字节基准 sha256 冻结 `test_pack_sparse_image_matches_frozen_byte_baseline`;指纹覆盖全几何字段 `test_build_fingerprint_covers_each_geometry_field_except_host_cap`(逐字段翻转断言,豁免 ttl_delay_max_ticks 与实现 `_FINGERPRINT_HOST_ONLY` 一致);usage.ipynb 4 cell 顶到底 exec 无错(编译→pack→engine_model 两值波形→device write_slots→fire→safe);README 与实况一致、无搬清单 | `tests/test_wire_device.py:48-72`;`wire.py:80` |

**commit 粒度与勾选:不对应(小修项)。** 全史仅 6 commit;Q0–Q7 几乎全部打包进首个 commit `4bf8681`(+5,390 行,含 model/compile/wire/device/engine_model/transport 一次落地),随后 `c8ae739`(对齐)、`5c5b370`(observer)、`dfd246f`(notebook)、`13ba001`(**GOAL.md 与两份"动工前必读"survey 倒数第二才入库**)、`2539a69`(收尾勾选)。"每主题小 commit"铁律与"每轮读 GOAL→选未勾项"仪式在历史上不可追溯;内容完备但过程凭证缺失。

## 2) 删除清单核实(Q4)

- **epoch/superseded/session 仲裁层:确无。** device.py 状态面仅 `_opened/_loaded/_firing/_forever` + scan 游标缓存(`device.py:82-99`);grep `epoch|session_id|superseded|to_tree|from_tree|APPROVED_DEPLOYED` 于 src 仅 1 命中=`engine_model.py:1130` 注释里 "superseded segment"(ramp 段物理语义,非仲裁层)。参照树 `transport/session.py` 1,128 行状态机整体未搬入。
- **FIRE 后单 I/O worker 所有权:真在。** `fire()` 起 `zlc-pulse-observer` 线程(`device.py:220-221`);`_observe` 是唯一读 STATUS/CURSOR 与 `_refill` 补给者(`device.py:292-320,342-359`);公开 `cursor()` 在 firing 时只回缓存(`device.py:252-256`);终读双读由 observer 完成(`_finish_observation`,`device.py:322-332`),`wait_done` 只消费缓存并有 `read_log` 尾序断言守卫(`tests/test_wire_device.py:124-142`)。
- **三相 SAFE:真在。** `safe()`=`_safe_phase`(STATUS_ERROR 哨兵+COMMAND 0→SAFE,双读)→ CLK_ENABLE 全 0 写+读回 → 再 `_safe_phase` 双读,`stable = 两读==0 且 clk 字全 0`(`device.py:258-276,334-340`),结构对齐参照 `session.py:704-772` 的 `_drive_physical_safe`;有测试(`test_safe_readback_uses_stable_status_and_zero_clock_mask`)。差异:参照版轮询等稳+重试+deadline,本版一次快照读回——契约写的是"读回",合规,但真机上健康设备可能偶发 `stable=False`(观察项,非缺陷)。
- **evidence 体系:确删。** `DoneReport` 是纯 frozen dataclass(6 字段+4 个只读派生 property,`device.py:29-52`),无证据链/内容寻址;`SafeReadback` 同(`device.py:55-63`)。grep `evidence` src 零(image.py 里唯一一处 "final evidence" 文案已改 "final note",`wire.py:1215`)。

## 3) model.py

- **slot 统一:是。** 单一 `PulseSlot`,无 api/scan 双命名空间(grep `api_slot|scan_slot` 零);"一行表 vs 多行表"差异落在 device 的 `write_slots`(bank0 行 0,`device.py:161-167`)vs `write_scan_table`,模型层无区分 ✅。
- **visible_ports/scan_sweep_count/编辑器常量:确无。** grep 零且 `test_negative_surface_is_absent` 机械守卫(`tests/test_import_purity.py:69-81`,rglob 全 src)。保留的 `TIME_UNIT_TO_NS/TIME_UNIT_CHOICES` 是单位换算事实非编辑器默认值,合理。

## 4) canonical.py 与 schedule.py

- **canonical.py:72 行(< ~150 预算),未泛化。** 全仓唯一调用方=`model.py:17,234`(PulseTarget ABI 指纹);未出现在 `__init__.__all__`,不对外开放 ✅。
- **schedule.py trigger_times:纯函数。** 输入 CompiledProgram+channel+table,输出 ndarray;模块 import 仅 numpy+compile,零 device/transport/协议引用;device.py 反向也不 import schedule(`schedule.py:1-8`;README 明言 "not sent to the device")✅。

## 5) 契约/README/守卫

- **contract.md 逐条:基本零漂移,一处参数名漂移(小修)。** `StreamerGeometry = StreamerParams` 别名 ✓(`wire.py:1364` vs `contract.md:8`);`pack` thin wrapper ✓(`wire.py:1390` vs `contract.md:12`);open 读 CTRL[63] ✓(`CtrlWords.LAYOUT_ID=63`,`wire.py:143`;`device.py:108-116`);write_slots/write_scan_table/fire/wait_done/cursor/safe/snapshot 签名与语义逐条对上;负面清单五条全部 grep 实证为零。**漂移**:contract 写 `compile(seq, …)`/`pack(prog, …)`(`contract.md:10,12`),实现与 `test_contract.py:9-10` 断言的是 `sequence`/`program`——契约是并行仓写 fake 的唯一依据,关键字调用会碎;两边取一改齐。
- **README:与实况一致。** 示例 import 全部真实存在;"no measurement/GUI/run-planning/synchronization layer" 与代码相符;RTL 冻结声明在。
- **顶层 allow-list 守卫:非空洞但有洞(小修)。** 有效性实证:allow-list 仅 `{zlc_pulse, numpy}`,任何 Qt/matplotlib/rpyc 进顶层模块即红;负面 token 测试 rglob 全 src 真扫描。**洞**:`tests/test_import_purity.py:54` 用 `SRC.glob("*.py")` 非递归,`transport/` 子包整体逃逸——实证 `transport/uart.py:37` 惰性 `import serial`(serial 不在 allow-list,pyproject 亦未声明 pyserial),守卫改 rglob 即红。需改 rglob 并把 serial 列为 uart 豁免(或声明依赖)。
- **word63 握手负路径无测试(小修)。** 终态判据3要求"word63 握手…契约测试在";现只有正路径隐式覆盖(memory transport 以 build_fingerprint 预置 LAYOUT_ID),`open()` 不符即拒(`device.py:110-116`)无 mismatch 拒绝测试。

## 6) LOC 与范围

实测 src 总计 **5,019 行**(与外报一致;tests 另 416):

| 模块 | 实测 | GOAL 预算 | |
|---|---|---|---|
| model.py | 524 | ~500 | ✓ |
| compile.py | 469 | ~900 | 远小于预算 ✓ |
| wire.py | 1,393 | ~1,300 | ✓ |
| device.py | 394 | ~800 | 远小于预算 ✓ |
| transport/ | 785 | ~1,000 | ✓ |
| engine_model.py | 1,165 | 1,159 | +6=空行;`--strip-trailing-cr` diff 实证仅 import 适配块(19→11 行)+周期性空行,逻辑零改 ✓ |
| canonical/schedule/__init__ | 72/104/113 | — | ✓ |

**范围外偷带:未发现。** 编辑器支撑(authoring/document/scan_program 等)与 rpyc/remote/server 全未出现;wire.py 内的容量求解器/估算 CLI(`solve_capacity/estimate_resources/_main`)属 GOAL Q3 "image.py 并入"的原样部分,非偷带——但其 docstring/CLI prog 名仍指旧世界(`python -m fpga.pulse_streamer.host.image`、`test_geometry_vh_matches_config`、`neutral_atom.devices...`,`wire.py:1270,1264,204`),为观察级文档漂移。

## 小修清单(按优先级)

1. contract.md 与实现参数名对齐(`seq/prog` vs `sequence/program`),契约先行改文件(`docs/contract.md:10,12`)。
2. `test_import_purity.py:54` `glob`→`rglob`,同时处置 `transport/uart.py:37` 的 serial(豁免+pyproject 声明,或延后导入注记)。
3. pyproject `zlc-data` 依赖零使用(全仓 grep 零 import)且与 allow-list 自相矛盾——删除或真用于 validation(`pyproject.toml:12`)。
4. 补 word63 mismatch 拒绝测试(open 负路径)。
5. (可选)`_observe` 内 `_refill` 写失败异常未兜(`device.py:318-319`):transport 抖动会杀死 observer 而无终读,`wait_done` 永远 None、`_firing` 悬置;把 `_refill` 纳入现有 except 路径。
6. (可选)清理 wire.py 指向旧仓的 stale docstring/CLI prog 名。

观察(不计修):`__init__.py:8-10` 安装路径断言只查目录名,树内同名影子包同样通过,防影子能力按 GOAL 字面完成但实际有限;safe() 稳定判定为快照非轮询(契约"读回"口径下合规)。