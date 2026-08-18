# 06-F — FPGA RTL、约束、构建、仿真与 Windows 入口审查

状态：本子阶段完成。
基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：`packages/zlc_pulse/fpga/` 下全部 Verilog/VH、Tcl、XDC、board/build 配置、testbench 与说明；补审根 `bin/*.bat` 及 `packages/zlc_workbench/bin/*.bat` 在 04-A/06-E 未逐项覆盖的 FPGA/重复入口部分。
限制：只读审查；未运行 Vivado、xsim、综合、实现、烧写或任何硬件操作；没有把本机 ignored build 产物当成当前源码的验收结果。只新增本报告。

## 1. 结论先行

当前 host 与 RTL 在默认冻结数值上的 **CTRL word、内存 region base、UART frame 常量和大部分 geometry 是对齐的**。核心 edge、ramp、scan、TTL/DAC delay 算法也不是应该因单文件较大而推倒重写的历史垃圾：它们共享一个 tick owner 和有限硬件资源，放在同一 engine 有真实理由。

但现有证据不能支持“已经通过物理 timing、安全和完整 waveform 验收”。有四个会直接改变实验输出、且软件 handshake 发现不了的 P0：

1. **50 MHz engine clock 没有 timing constraint**；构建脚本只要找到一条非负路径就可放行。现存旧 report 已实际证明它检查的是 JTAG TCK，而 16,824 个 engine register/latch pin 没有根时钟约束。
2. **`CMD_SAFE` 不是单命令物理安全态**：`CLK_ENABLE` mux 绕过 engine reset/running，SAFE 后仍可输出 `~clk`；新 image 上传时也会在 LOAD/FIRE 前提前开 DAC clock。
3. **DONE 早于 delayed output 排空**：host `wait_done()` 可返回后立即 SAFE/LOAD，从而截断仍在 FIFO 中的 TTL/DAC 尾部。
4. **物理 pin/lane ABI 不在硬件 fingerprint 中**：host 从 XDC 文本声明顺序推导 `ch00..`，top 却手写 index-to-port；只重排 XDC 就可静默把逻辑通道打到错误引脚，而 word 63 仍匹配。

此外，scan underflow 会消失、首 scan point 不验证 bank identity、delay FIFO overflow 静默丢事件、UART 截断帧可长期独占控制总线。这些都不是“多加几条测试”即可证明安全的边角；需要先由用户裁决终态契约，再改 RTL/ABI。

总体裁决：

- `zlc_edge_streamer`：`REDESIGN IN PLACE`，保留算法 owner，修 DONE/error/scan contract，不机械拆散 timeline。
- `zlc_pulse_streamer_top`：`REDESIGN`，尤其 pin-boundary SAFE、clock output、status 和 board ABI。
- `zlc_uart_bridge`：`PASS WITH REDESIGN DEBT`，保留 framing/CRC/commit，补 watchdog 与 bounds。
- XDC/timing/build qualification：`REDESIGN / BLOCK PHYSICAL QUALIFICATION`。
- testbench 资产：多数有诊断价值，但 **当前没有一条被普通测试自动编译或执行**；必须接入 runner 和可失败 oracle 才能称回归。
- `packages/zlc_workbench/bin/`：`DELETE/MERGE`，它是绕过 monorepo bootstrap 的第二套旧入口。

## 2. 审查与验证边界

本子阶段做了：

- 逐文件读取全部 tracked FPGA 非 Python 资产；
- 逐 module/function/task、Tcl proc、testbench 和 batch 入口建立裁决；
- 反查 host 的 `wire.py`、compiler、device observer、UART transport 和 XDC parser，以核对跨边界 ABI；
- 运行不接触硬件的三组静态/内存测试：

~~~text
packages/zlc_pulse/tests/test_fpga_assets.py
packages/zlc_pulse/tests/test_command_strobe.py
packages/zlc_workbench/tests/test_launchers.py
42 passed
~~~

这些测试只证明资产存在、generated header 文本、command-edge 源码形状、Python/memory twin 与 batch CRLF；它们 **没有编译或执行 Verilog**。

本机没有 `iverilog`、`verilator`；本阶段也按限制没有调用 Vivado/xsim。本机 `fpga/build/ps/...timing_summary_routed.rpt` 是 `.gitignore` 下的旧产物，header 指向旧 standalone checkout、Vivado 2019.1 和 2026-08-04。下文只用它证明“当前同形 Tcl gate 可 false-green”的既成机制，不把它当当前 HEAD 的 timing acceptance。

## 3. 当前确实对齐的数字 ABI，以及它没有覆盖的内容

| 契约 | 当前 owner/投影 | 当前结果 | 裁决 |
|---|---|---|---|
| CTRL words 0..20、63 | host `wire.CtrlWords`；top localparams | 默认值一致 | `PASS` |
| region bases | host `region_bases()`；top `R_TICK/R_COEFF/...` | 默认 geometry 下公式一致 | `PASS` |
| channel/slot/edge/bank/bus widths | `streamer_config.json` → generated `zlc_geometry.vh`/`geom.tcl` → host/top | 当前 checked-in 值一致 | `PASS WITH DEBT` |
| UART sync/op/count/CRC | `uart_frame.py`；`zlc_uart_bridge.v` | 常量与 CRC 算法一致 | `PASS` |
| LAYOUT_ID | host `build_fingerprint(StreamerParams)` → header/top word 63 | 能拒绝大部分 geometry drift | `PASS WITH ABI HOLES` |
| clock rate | JSON/host compiler；board oscillator/XDC/RTL | JSON 写 50 MHz，但 XDC 无 `create_clock` | `REDESIGN` |
| physical lanes/pins | host 按 XDC 声明顺序；top 手写 index mapping | 没有共同 machine-readable identity | `REDESIGN` |
| DAC safe value | public target/compiler 可携带；RTL固定 midpoint | custom value 没有 wire field | `REDESIGN API` |
| DONE | RTL logical table terminal；host理解为 run complete | delayed physical tail 不一致 | `REDESIGN` |

`LAYOUT_ID` 不是完整 build/behavior identity。它没有覆盖：

- `clock_hz`；
- XDC signal order、logical port grouping、package pin 和 target ABI；
- RTL 固定 `SLOT_MUL_WIDTH=25`、`RD_LAT`、`FIFO_DEPTH`、global-time width；
- clock mux phase、IP property 的真实 readback、Vivado/IP/toolchain version。

`slot_mul_width` 并非普通漏读：config loader 明确要求它等于冻结值 25，compiler/model/top 也各自使用 25。但它不属于 `StreamerParams`，所以不进 word 63；开发者若同时改 loader/RTL，仍可能产出同一个 layout ID。正确结论是“有 host guard，但不是硬件 ABI proof”。

## 4. 最高风险问题

### FPGA-001（P0 timing）— engine 主时钟未约束，当前 Tcl gate 可把 JTAG timing 当作通过

`board.xdc` 给 `clk` 分配 package pin/IOSTANDARD，但全仓没有 `create_clock`、`set_input_delay`、`set_output_delay` 或 `check_timing`。`create_project.tcl` 的终态门槛只是：

1. `get_timing_paths -delay_type max/min -max_paths 1`；
2. 确认至少有一条 constrained path；
3. 确认那一条 slack 非负。

这并不要求该 path 来自 pulse engine，也不要求零 unconstrained endpoint。ignored 旧 routed report 恰好显示：

- 16,824 个 register/latch pin “no clock driven by root clock pin: clk”；
- 60,923 个 pin 没有 maximum-delay constraint；
- 1 个 input 无 input delay，65 个 output 无 output delay；
- 唯一列出的 clock summary 是 BSCAN/JTAG TCK，period 33 ns；
- 同一 report 仍写 `All user specified timing constraints are met.`。

因此现有 build 可在 pulse core 完全未做 STA 的情况下生成 bitstream 并保存 source hash。“setup/hold slack pass”不是当前物理资格证据。

裁决：`REDESIGN / BLOCK BITSTREAM QUALIFICATION`。最小正确目标：

- 对 50 MHz board clock显式 `create_clock`；
- 明确 UART async/CDC 约束；
- 为 DAC data/latch clock 建外部 setup/hold/board contract，而不是把 `~clk` 经普通 LUT mux 当已证明的 generated clock；
- build hard-fail 于任何意外 unconstrained sequential endpoint/I/O；
- gate 针对预期 clock group逐项检查，而非任意一条 timing path。

### FPGA-002（P0 physical safety）— SAFE 与 LOAD 不能保证 pin 安全

top 的最终输出是：

~~~verilog
out_final[cmx] = clk_en[cmx] ? ~clk : out[cmx];
~~~

这个 mux 不看 `eng_reset`、`running` 或 SAFE 状态。SAFE command 只 assert engine reset并清 STATUS，不清 `CLK_ENABLE`。后果：

- 一个已 enable 的 DAC clock lane 在 SAFE 后仍连续跳变；
- host 为补洞不得不执行 `SAFE → 清 CLK_ENABLE words → readback → SAFE`；
- 若连接在第一次 SAFE 后断掉，STATUS 可以为零，而物理 clock 永久继续；
- `load()` 上传新 image 时，新的 CLK_ENABLE word 一写入 ctrl reg 就打开 clock，发生在 LOAD/FIRE 前，所谓 prepare/armed safe 不成立。

裁决：`REDESIGN`。推荐 pin boundary 有独立硬件 safe gate，单条 SAFE 必须在不依赖后续 host write 的情况下把所有 TTL/DAC data/clock 送到定义好的安全状态。需要同时设计 DONE 尾部的最后一次 DAC latch，不能只在某一周期粗暴截 clock。

### FPGA-003（P0 completion）— DONE 是 logical done，不是 physical done

finite engine 在 edge table terminal 时立即 `done<=1`、清 undelayed state；delay FIFO 随后仍在独立分支继续 advance/drain。top 同周期把它投影为 STATUS_DONE，host observer 一读到 DONE 就返回。compiler 的 duration 也没有加最大 output-delay tail。

因此实际状态序列是：

~~~text
logical table terminal → STATUS_DONE / wait_done returns
                       → TTL/DAC delayed FIFO 仍可能输出
                       → physical outputs finally drained/safe
~~~

调用者在 `wait_done()` 后 SAFE 或 LOAD 下一 program，会截断上一 shot 的物理尾部。现有 delay benches 大多用 `repeat_forever=1`，没有覆盖 finite DONE + nonzero delay tail。

裁决：`REDESIGN`。推荐 DONE 只在所有物理 scheduler empty 且 pin 已进入契约安全态后置位；若产品确实需要 logical terminal，则新增明确的 `LOGICAL_DONE` 与 `PHYSICAL_DONE`，host public API 不得再把前者叫 completed。`DoneReport.tail_elapsed` 当前只是 FIRE 后墙钟时间，不是 tail wait。

### FPGA-004（P0 board identity）— XDC 声明顺序是隐形 lane truth，硬件 handshake 不验证它

host parser 按 XDC 中非 system `PACKAGE_PIN` 声明的出现顺序生成 `ch00..ch61` 并组合 DAC family；top 则在源码尾部手工固定 `out_final[index] → named port`。word 63 只 hash `StreamerParams` 数字。

于是只要：

- 重排 XDC 声明；
- 改 logical signal family/name；
- 用 `ZLC_PS_XDC` build 另一份 pin map；

host lane mapping 就可能改变，而 top index mapping和 LAYOUT_ID 不变。server 读到 word 63 仍会认为兼容，随后把 mask 打到错误物理线。这直接否定 README 中“board describes itself”的强表述：board只回 geometry，不回 lane/pin ABI；target 是本机 XDC 推导出的。

裁决：`REDESIGN`。推荐用一个显式带 index 的 board manifest 同时生成 host target、top port mapping 和 XDC，另将 board/pin ABI ID 嵌入 bitstream readback。XDC 文本顺序不可继续充当不可见身份。

### FPGA-005（P1 scan correctness）— underflow 可被抹掉，首点与短帧没有闭合契约

三个独立问题：

1. bank 不 resident 时 engine 置 underflow；refill 后又清零，top 每周期覆盖 STATUS。慢 host polling 可永远看不到一次已经破坏绝对时间轴的 stall。
2. FIRE 直接使用 arm 时缓存的 `scan_first_values`，不验证 bank ready/chunk0 identity；resident check 从 point0→point1 才开始。旧 bank0 数据可被静默当首点播放。
3. scan BRAM 被强制 latency 2，但没有 host/RTL 最短 scan-frame ticks 契约。现有 bench 用 10 或 60 ticks，没有 1/2 tick point。极短合法 point 可能在 rdata 更新前切换而复用旧值。

裁决：`REDESIGN`。UNDERFLOW 必须 sticky 到 reset/下一 FIRE；scan-enabled FIRE 必须验证 point0；用户需决定是 host 明确拒绝低于 pipeline 下限的 frame，还是为了 1-tick scan 增加 prefetch FIFO。

### FPGA-006（P1 waveform integrity）— delay FIFO overflow 静默丢事件

TTL 和每 bus segment FIFO 都只在 not-full 时 push；full 时不产生错误。top 注释还明确 STATUS_ERROR 是 host-only。`tb_evt_depth` 甚至把 overflow 后少两个 toggle 当成预期行为。

静态 compiler validator 可以降低出现概率，但不是硬件 invariant；任何 host/RTL capacity model drift、future ABI 或受损 image 都会变成不可观测的物理 waveform corruption。

裁决：`REDESIGN`。overflow 应 latch hardware ERROR，并拒绝/终止 run；测试应验证 loud failure，而不是锁定 silent drop。

### FPGA-007（P1 UART liveness）— 截断帧可 wedge decoder并屏蔽 JTAG

UART decoder 从 opcode 起置 `u_active=1`，没有 inter-byte/frame timeout。若帧被截断但没有 framing error，它会一直等待剩余 payload/CRC；top 在 `u_active` 时优先 UART、屏蔽 JTAG write/read path。CMD_RESET/SAFE 不 reset UART FSM，只有 POR counter。

另有两个边界：

- RTL 不拒绝 `COUNT > FRAME_WORDS=256`，低位 index 会重用 buffer；host虽限制，RTL protocol 自身未闭合。
- READ tap 实际只能读 6-bit CTRL words，host `UartRegisterTransport.read_word()` 看起来却接受任意 flat address；超出 63 的 read 应由 host 直接拒绝，而不是隐式 modulo/返回 CTRL alias。

裁决：`PASS CORE / REDESIGN DECODER`。保留 CRC、commit-after-CRC、reply serializer；加 frame watchdog、count/address bounds、明确 error reply，并释放 `u_active`。

### FPGA-008（P1 API/RTL）— custom DAC safe value 被 public API 接受但 RTL 固定 512

`PulsePortSpec` 允许 DAC `safe_value`，compiler把它保存进 `CompiledProgram.bus_safe_values`；wire image没有对应字段或 packing consumer，RTL `BUS_SAFE_VALUE` 固定为 `1 << (BUS_WIDTH-1)`。因此非 midpoint safe value 能通过 model/codec/compile，却不会改变硬件 safe state。

裁决：`REDESIGN API`。最小方案二选一：

- 推荐短期：FPGA target/compiler明确拒绝所有非硬件 midpoint 的 DAC safe value；
- 若确有 per-bus safe 需求：将它纳入 wire ABI、fingerprint、loader和 RTL，并测试 SAFE/DONE/LOAD 全路径。

### FPGA-009（P0 destructive build）— `zlc_safe_project_dir` 的名字与实际保护不符，可递归删除任意短目录

`create_project.tcl` 接受 `ZLC_PS_PROJECT_DIR`，`zlc_safe_project_dir()` 只检查推导出的 debug temp path 不超过 146 字符；它没有验证：

- resolved path 位于预期 `fpga/build` 或明确批准的 build root 内；
- basename/结构确实是本工具创建的 `ps` project；
- 目录内存在可识别的 generated-project marker；
- target 不等于 checkout、workspace 或其他用户目录。

随后脚本对该路径执行 `file delete -force $project_dir`。环境变量误设为任意较短真实目录即可触发递归删除；注释“build must stay under fpga/build”不是可执行 invariant。

裁决：`P0 FIX BEFORE NEXT BUILD`。delete 前必须做 canonical containment、expected basename/marker、拒绝 broad/root/workspace target；override build root 时也应以显式批准的 root 为 containment owner，而不是仅看字符串长度。

### FPGA-010（P0 hardware targeting）— program/flash 都默认选第一个 target/device

`program_fpga.tcl` 与 `program_flash.tcl` 都取 `get_hw_targets` 的第 0 项，再取 `get_hw_devices` 的第 0 项；没有：

- 明确 selector 或 exactly-one 检查；
- device PART/IDCODE 与 build target核对；
- 多 JTAG target/device 时拒绝；
- flash 永久动作的二次确认；
- program/boot 后读取 layout/build/board identity 证明写入对象与 image。

在同时连接多板或 JTAG chain 含多个器件时，这可能把 volatile image 或永久 flash 写到错误设备。根 `build_and_program.bat` 无参数还默认 build + program，使 recovery-only 文案无法承担真正授权边界。

裁决：`P0 REDESIGN`。默认应 fail closed：要求 exactly one approved device，或显式 target/device selector；核对 part/IDCODE；flash 需要独立确认并在 reboot 后做 identity/readback。用户需裁决无参数是否仍允许改硬件；审计建议默认 build-only，program/flash 必须显式选项。

### FPGA-011（P1 canonical config）— build/server 对损坏 deployment manifest 会回退默认值

`load_streamer_config()` 为 offline 使用提供 built-in fallback并附 warning，这对离线 GUI/容量估计可以合理；但 geometry emitter 和 remote server/config check 没有把 `warnings/source` 升级为错误。canonical JSON 缺失、损坏或 schema 不合法时，build launcher仍可能用 fallback 生成 header/Tcl并返回成功。

这与 manifest 自称“single frozen deployment manifest；错误只会让 startup fail”相反，也让“一份 geometry truth”变成 JSON 与 Python literal fallback 两份可部署 truth。

裁决：`SPLIT POLICY`：

- offline editor/estimator可以明确展示 fallback warning；
- server、build、program前检查必须要求 canonical config 存在、严格有效、无 warning，并记录其 digest；
- fallback 不得静默成为可烧写 image 的 authority。

## 5. Board config、generated bridge 与约束逐文件裁决

| 文件/资产 | truth owner / 当前职责 | 裁决 | 主要债务或终态 |
|---|---|---|---|
| `board_config/streamer_config.json` | deployment geometry、part、clock、capacity target | `PASS WITH REDESIGN DEBT` | 当前可解析且与生成投影一致；不是 pin/behavior/build identity；critical 路径必须 fail-closed。 |
| `board_config/board.xdc` | package pin、电气属性；host又把文本顺序当 lane order | `REDESIGN` | 补 clock/I/O/CDC constraints；lane index显式化；不能声称唯一 mapping truth而 top 又手写第二份。 |
| `pulse_streamer/zlc_geometry.vh` | config → RTL parameter 的 generated projection | `PASS GENERATED` | 当前与 generator完全一致；继续保留 equality guard。扩大 ABI 时不要手改。 |
| `build/geom.tcl` | config → Vivado IP sizing 的 generated projection | `USER DECISION / DELETE TRACKED COPY` | 当前与 generator相等；batch每次重生成，direct Tcl又不会默认 source它，存在性测试是主要 tracked consumer。推荐生成到 build temp并作为同一事务；若坚持 tracked，则 create_project必须默认强制读取且测试精确 equality。 |
| `fpga/__init__.py`、`pulse_streamer/__init__.py` | Python package marker | `PASS / OUT OF NON-PYTHON SCOPE` | 无第二行为 owner。 |

### 5.1 XDC 电气边界还需硬件负责人裁决

`GND1..GND15` 是 LVCMOS33 FPGA output，由 top 驱低；它们不是物理 ground。如果实验接线真的把这些 IO 当 signal return/地，需要立即由硬件负责人确认是否安全。若它们只是 reserved-low/guard outputs，应改名和文档，避免使用者将普通 IO 当回流路径。

DAC `da_clk` 当前来自组合 `~clk` mux；`tb_da_clk_phase` 只模拟人为 bit skew，不能代替 ODDR/clock-forwarding 设计与板级 setup/hold 约束。实际 DAC 器件、走线、边沿和允许相位须成为明确 board contract。

## 6. RTL 逐 module/function/task 裁决

### 6.1 `zlc_pulse_streamer_top.v` — `REDESIGN`

文件有存在必要，也位于正确层级：它是 transport/IP/BRAM 与单一 engine 的硬件 composition root。问题是 pin、安全、status和 board ABI 都被混在一份“看似只是 glue”的源码中，必须明确它就是 physical boundary owner。

| 区域/符号 | 裁决 | 理由 |
|---|---|---|
| top parameters + region localparams | `PASS WITH ABI DEBT` | 当前 host formulas一致；部分“参数”实际只支持冻结 geometry。 |
| CTRL register projection/read tap | `PASS WITH DEBT` | words对齐；`C_MAGIC/C_BANK_SIZE/C_SLOT_COUNT`硬件不消费，属于冗余描述。 |
| UART/JTAG priority mux | `KEEP` | 单一 arbitration合理；必须由 UART watchdog保证不会永久占有。 |
| `out_final = clk_en ? ~clk : out` | `REDESIGN` | 绕过SAFE/running且没有物理 clock-forwarding timing proof。 |
| hardwired layout readback | `KEEP / EXPAND` | geometry拒错有价值；加入 board/build/behavior identity。 |
| BRAM/IP composition | `KEEP` | 层级正确；build需硬验证关键 IP property。 |
| bus image mini-loader | `KEEP WITH DEBT` | 单owner合理；coeff loader固定两words，暴露假参数化。 |
| command/status FSM | `REDESIGN` | SAFE、DONE、ERROR、UNDERFLOW语义不闭合。 |
| `bus_count_of()` | `PASS` | 小型 packed field accessor合理。 |
| `R_relbus()` | `PASS` | loader address计算单owner。 |
| `zlc_delay_identity_map()` | `PASS WITH PROFILE DECISION` | 当前固定 mapping正确；若冻结板型可变成明确常量生成物。 |
| hand-written `out_final[index] → port` | `REDESIGN/GENERATE` | 与 XDC parser形成第二 lane truth。 |
| GND outputs | `USER DECISION` | 技术上可驱低，但命名暗示物理地，硬件含义不清。 |

明确死/历史候选：

- `ldr_cmd_clear` 永远写 0、从不清 command：`DELETE` 或实现真正职责；当前测试反而用 regex 锁住它“不自清”的源码形状。
- `CMD_RESET` 没有真实 public caller，行为与 SAFE相同：下一 ABI `DELETE/MERGE` candidate。
- `C_MAGIC`、`C_BANK_SIZE`、`C_SLOT_COUNT` 不被 hardware execution消费：若只为 archive/diagnostic，移出 hardware control truth或明确只读 metadata；不要假装它们参与验证。

### 6.2 `zlc_edge_streamer.v` — `REDESIGN IN PLACE`

不建议因 1326 行机械拆文件。edge prefetch、affine tick、scan point、bus ramp 和 delayed physical scheduler共享同一 global time；随意拆成多个“职责漂亮”的模块会增加跨模块 latency/valid contract。正确做法是先固定行为语义和 error/status，再按可综合边界决定是否提取 helper。

| function/task/主要区 | 裁决 | 理由/修改方向 |
|---|---|---|
| `evt_ch_of()` | `PASS` | FIFO flat index解码清楚。 |
| `zlc_delay_ch_at()` | `PASS` | packed delay slice accessor。 |
| `zlc_delay_bus_at()` | `PASS` | 同上。 |
| `zlc_effective_tick()` | `PASS WITH HOST RANGE CONTRACT` | affine时间核心正确；依赖 host 对 overflow/slot范围的硬校验。 |
| `clamp3()` | `REDESIGN` | 名称误导且 seed/width 写法只真实支持当前 FIFO/prefetch profile；若冻结就命名为固定 profile，否则泛化。 |
| `zlc_bus_count_at()` | `PASS` | packed count accessor。 |
| `scan_addr_of()` | `PASS` | bank-local地址职责清楚。 |
| `bank_of()` | `DELETE` | 零调用。 |
| `chunk_of()` | `PASS` | chunk identity核心。 |
| `scan_point_resident()` | `KEEP WITH FIX` | 首点也必须调用；结果导致 sticky underflow/error。 |
| `zlc_bus_clear_runtime` | `PASS` | fire/reset runtime state唯一清理点。 |
| `zlc_bus_capture_safe_hold` | `KEEP` | 末尾 DAC safe/latch有价值；DONE必须等它完成。 |
| `zlc_delay_clear_runtime` | `PASS` | delay FIFO runtime清理。 |
| `zlc_bus_ramp_divmod` | `PASS WITH NUMERIC TEST DEBT` | ramp余数分配owner合理。 |
| `zlc_bus_apply_segment` | `KEEP WITH ERROR LATCH` | segment transition合理；FIFO full不可静默。 |
| `zlc_bus_tick` | `KEEP WITH ERROR LATCH` | 同一tick owner正确；补overflow/physical-done状态。 |
| `seed_from_edge0` | `KEEP WITH DEBT` | arm/first-edge pipeline必要；固定e0..e4展开显示假参数化。 |
| ARM/FIRE/prefetch主 FSM | `REDESIGN` | 修首scan bank、短frame、sticky underflow、DONE tail。 |
| TTL FIFO generate | `KEEP WITH ERROR LATCH` | per-channel scheduler设计合理；full必须失败。 |

明确死/历史候选：

- `bus_runtime_addr`：未使用，`DELETE`。
- `bnd_delay_advance`：赋值但注释已承认未使用，`DELETE`。
- `C_REPEAT_FROM_LOOP_START` 对应 engine branch：host固定写 `HOST_REPEAT_FROM_INDEX=0`，`CompiledProgram`不暴露；`DELETE NEXT ABI` candidate，除非用户确认要公开该模式。

“可参数化”目前是半真状态：shadow/arm arrays显式写死 `[0..3]`，edge seeds写死 e0..e4，若干 pointer宽度固定，top bus coeff loader固定两word。用户需二选一：

- 推荐：承认唯一冻结 board profile，把假参数化收成 generated constants + elaboration assertions；
- 若要任意 geometry：重写固定展开并对多个 geometry做真实 RTL elaboration/sim，而不是由 host guard替 RTL兜底。

### 6.3 `zlc_uart_bridge.v` — `PASS CORE / REDESIGN LIVENESS`

| module/符号 | 裁决 | 理由 |
|---|---|---|
| `zlc_uart_bridge` baud NCO / RX 8N1 | `PASS WITH CDC DEBT` | 实现可留；补 async synchronizer属性/constraint与误差范围证据。 |
| `crc_byte()` | `PASS` | 与 host CRC一致。 |
| decoder/HUNT FSM | `REDESIGN` | 加 timeout、count/address bounds、abort/error reply。 |
| commit-after-CRC | `PASS` | 坏 frame 不落寄存器是正确事务边界。 |
| CTRL read staging | `PASS NARROW CONTRACT` | 只支持 CTRL，应在 host/public protocol写明并拒绝其他地址。 |
| reply serializer | `PASS` | 当前不允许 pipelined READ，需继续显式。 |
| `zlc_uart_tx` | `PASS` | 小而独立的 serializer合理。 |
| `u_rd_req` | `DELETE CANDIDATE` | top不消费；若只为旧bench seam应移除。 |

源码注释引用不存在的 `host/uart_bridge_model.py`、`tests/test_uart_bridge_equivalence.py` 和 `tb_uart_bridge.v`。真实 benches 是 `tb_uart_pipeline.v`、`tb_uart_read_tap.v`，且没有自动 runner。注释不可作为 equivalence evidence。

## 7. Tcl 构建、诊断与烧写逐 proc/文件裁决

### 7.1 `create_project.tcl` — `REDESIGN`

它应该继续作为 Vivado project/IP composition owner，但不能继续同时依赖 fail-open property、未约束 timing 和危险 delete。

| proc/区域 | 裁决 | 理由/终态 |
|---|---|---|
| `env_or(name, default)` | `KEEP FOR PATH ONLY` | 对 XDC/project path normalize合理；scalar/name/url不能复用。 |
| `zlc_default_project_root()` | `MERGE VOCABULARY` | 三份 Tcl各复制且 batch另有 BUILD_ROOT/PROJECT_ROOT/PROJECT_DIR；定义一套路径语义。 |
| `zlc_debug_tmp_path()` | `PASS` | Windows Vivado深路径诊断有价值。 |
| `zlc_safe_project_dir()` | `REDESIGN P0` | 只验长度，不验delete containment/marker。 |
| `zlc_require_run_complete()` | `PASS` | synth状态 fail-closed。 |
| `zlc_try()` | `DELETE FOR CRITICAL SETTINGS` | 关键 IP property错误不可只打印 warning；仅可用于 debug probes等真正可选项。 |
| `zlc_dump_ip()` | `PASS DIAGNOSTIC` | 完整 dump有定位价值，但不是 machine gate。 |
| `zlc_force_latency2()` | `PASS` | 少数真正 set + readback + hard-fail 的 property owner；应扩展同一模式。 |
| `zlc_require_nonnegative_slack()` | `REDESIGN P0` | 当前可验证错 clock；改成 expected domains + zero unexpected unconstrained。 |
| direct literal geometry fallback | `DELETE DEPLOYMENT FALLBACK` | approved build必须读严格 canonical projection；direct Tcl无 Python不应制造另一合法 build truth。 |
| project/IP generation flow | `KEEP WITH HARD READBACK` | topology可保留；JTAG AXI、BRAM protocol/width/depth/enable/latency全部关键项需 hard readback。 |
| in-process opt/place/route | `PASS WITH TOOLCHAIN RECEIPT` | 规避旧并行run问题合理；需要记录 tool/IP版本、constraints digest。 |

目前大量 correctness-critical property 走 `zlc_try`：JTAG AXI protocol/data/address/ID、AXI BRAM controller protocol/ID/read latency，以及多个 BRAM 的 memory type、width、depth、enable/reset。resolver又倾向选择任意已安装的新 Vivado；property rename 正是会触发“继续build但语义变了”的场景。只有 edge/scan port-B latency与少量 depth被 hard readback，不足以证明 IP ABI。

脚本还打印固定“4096 edges/bank2048”等文案，即使 recovery geometry通过投影改变；这属于诊断漂移，不能当验证。

### 7.2 `diagnose_hw_target.tcl` — `PASS WITH DEBT`

| proc/行为 | 裁决 | 理由 |
|---|---|---|
| raw `env_or()` | `PASS/INLINE` | 对 URL 保留原值是正确语义；仅一次使用可inline。 |
| enumerate target/device | `KEEP` | 只读恢复诊断有必要。 |
| first bad target handling | `FIX` | 当前可能在第一个不可开 target停止，无法完整列出其余候选。 |

诊断脚本可以帮助用户选择设备，但 program scripts目前并不消费其输出形成显式 identity，二者尚未闭环。

### 7.3 `program_fpga.tcl` — `REDESIGN`

| proc/行为 | 裁决 | 理由 |
|---|---|---|
| `env_or()` | `SPLIT PATH/RAW` | 它对所有值 `file normalize`，会把 `host:3121` 变成文件路径。 |
| `zlc_default_project_root()` | `MERGE VOCABULARY` | 与 create/flash/batch重复。 |
| bit/ltx path resolution | `KEEP AFTER PATH OWNER` | 路径 fallback合理。 |
| target/device selection | `REDESIGN P0` | 不可默认第一个；核 part/IDCODE/exact selector。 |
| program + refresh + list hw_axi | `KEEP WITH POSTCHECK` | 编程流程合理；之后需读 board/build/layout identity并拒错。 |

`ZLC_PS_HW_SERVER_URL`/`ZLC_HW_SERVER_URL` 目前经过路径 normalize，documented remote hw_server override 实际失效。硬编码输出 `CHANNEL_COUNT=62 NUM_SLOTS=4` 也不是验证，只是陈旧文案。

### 7.4 `program_flash.tcl` — `REDESIGN HIGH RISK`

| proc/行为 | 裁决 | 理由 |
|---|---|---|
| `env_or()` | `SPLIT PATH/RAW` | 除 URL 外，还破坏 cfgmem part name与 numeric size。 |
| `zlc_default_project_root()` | `MERGE VOCABULARY` | 同上。 |
| `.mcs` generation | `KEEP AFTER STRICT INPUT CHECK` | 应绑定 approved bitstream/build receipt/flash geometry。 |
| cfgmem attach/program/verify | `KEEP WITH TARGET REDESIGN` | verify有价值；设备和 part选择不安全。 |
| immediate reboot | `USER DECISION` | 永久动作后自动boot合理，但必须显式授权并post-handshake。 |

三个 documented override 被错误 normalize：

- `ZLC_PS_HW_SERVER_URL`；
- `ZLC_PS_CFGMEM_PART`；
- `ZLC_PS_CFGMEM_SIZE_MB`。

flash脚本还默认第一个 target/device；在永久动作上这是不可接受的组合。

## 8. Build launcher、tool resolver 与重复 Windows 入口

### 8.1 `bin/build_and_program.bat` — `REDESIGN`

保留一个 recovery workflow有必要，但当前约 380 行 batch 同时拥有参数状态机、路径、config projection、source hash、tool invocation、build cache、program/flash授权，已经成为第二个 deployment controller。

| label/区域 | 裁决 | 主要问题/终态 |
|---|---|---|
| argument mode dispatch | `REDESIGN` | 无参数默认改硬件；推荐默认 build-only，program/flash显式。 |
| `:zlc_verify_sources` | `KEEP / MOVE TO PYTHON OWNER` | 资产完整性有价值；不宜在 batch复制 schema。 |
| `:zlc_resolve_part` | `MERGE WITH STRICT CONFIG` | canonical JSON坏时不可 fallback。 |
| `:zlc_emit_geom` | `MOVE TO ATOMIC PROJECTION` | header/Tcl pair需原子生成，不能半成功留下工作树变化。 |
| `:zlc_compute_src_hash` | `REDESIGN BUILD ID` | 漏 part override、Vivado/IP版本；又包含纯program Tcl导致无谓rebuild。 |
| `:zlc_check_prebuilt` | `KEEP AFTER RECEIPT` | skip rebuild有价值；必须验证 artifact/part/tool/config/source完整receipt。 |
| `:zlc_save_src_hash` | `MERGE AS BUILD RECEIPT` | 固定 ignored hash不足以证明 qualified image。 |
| `:zlc_print_capacity_estimate` | `PASS DIAGNOSTIC` | 估算不等于build/physical资格。 |
| `:zlc_default_paths` | `REDESIGN VOCABULARY` | BUILD_ROOT/PROJECT_ROOT/PROJECT_DIR三套重叠。 |
| `:zlc_run_tcl` | `PASS` | 薄工具调用可留。 |

具体 cache/identity 缺口：

- hash 不含 `ZLC_PS_FPGA_PART` override、Vivado/IP版本；
- `program_fpga.tcl` 属部署逻辑却被纳入 bitstream source hash；
- `ZLC_PS_GEOM_TCL` 可另指与刚生成 header不一致的 sizing Tcl，缺交叉验证；
- `ZLC_REPO_ROOT` 被设置但全仓无 consumer；
- fixed temp/hash文件不支持并行 build；
- `--program-only`/`--flash` 可部署已有 bitstream，却没有验证它与当前 config/source/target的 receipt相容。

geometry/config/hash/build receipt应由 `zlc_pulse.fpga` 的一个严格非硬件命令 owner完成；batch只解析Windows点击参数、找工具并调用，不应维护另一份状态机。

### 8.2 `packages/zlc_pulse/fpga/_resolve_tools.bat` — `MOVE/REDESIGN`

| label/行为 | 裁决 | 理由 |
|---|---|---|
| `:zlc_find_python` | `MOVE TO ROOT BOOTSTRAP` | 全产品解释器选择不应由 FPGA leaf拥有。 |
| `:zlc_python_try` | `KEEP AFTER VALIDATION FIX` | candidate验证思路有价值。 |
| `:zlc_python_conda_activate` | `KEEP WITH EXPLICIT ENV POLICY` | 激活行为应属于产品environment owner。 |
| `:zlc_python_executable` / `:zlc_python_found` | `MERGE` | 小状态机可简化。 |
| `:zlc_find_vivado` / `:zlc_vivado_found` | `REDESIGN TOOLCHAIN PIN` | 当前倾向任意最新安装版，与version-sensitive IP配置冲突。 |

明确历史 seam/bug：

- `.zlc_python_path` 只有 reader，全仓无 writer或文档：`DELETE` 或正式定义；
- 已定义 `ZLC_PY_CMD` 时不充分验证/quote，带空格路径风险；
- `ZLC_FPGA_PYTHON` 指向不存在文件时仍可无条件接纳；
- known-version loop/glob可能覆盖先前候选，没有 approved Vivado version。

### 8.3 root 普通 launchers补审

06-E 已给出产品入口总裁决；本节只记录本阶段新增的非 Python/重复入口事实。

| 文件 | 裁决 | 本阶段补充 |
|---|---|---|
| `bin/_launch.bat` | `PASS WITH DEBT` | 正确走 monorepo bootstrap；Python resolver不应位于FPGA leaf；delayed expansion会损坏含`!`参数。 |
| `bin/experiment.bat` | `PASS` | 薄转发，不复制composition。 |
| `bin/pulse_editor.bat` | `PASS` | 同上。 |
| `bin/figure_viewer.bat` | `PASS` | 同上。 |
| `bin/run_server.bat` | `PASS WITH DOC/DEAD LABEL DEBT` | 正式server入口；help仍写不存在的 `fpga\run_server.bat`；`:zlc_help_error` 无调用，`DELETE`。 |
| `bin/estimate_resources.bat` | `PASS WITH CONFIG DEBT` | 只读入口合理；必须明确 fallback warning，不得被误当build授权。 |
| `bin/install_requirements.bat` | `PASS WITH CONCURRENCY DEBT` | 固定 `%TEMP%\zlc_dependencies.txt` 可并发碰撞；其部署/lock问题见06-E。 |
| `bin/update.bat` | `PASS WITH DOC DEBT` | import/check并不证明已打开的窗口仍工作；environment mutation也不能靠checkout回滚。 |

### 8.4 `packages/zlc_workbench/bin/*` — `DELETE/MERGE`

| 文件 | 裁决 | 理由 |
|---|---|---|
| `_launch.bat` | `DELETE` | bare Python直接 `-m zlc_workbench.apps...`，不设置root path、不import bootstrap，可加载旧editable/sibling packages。 |
| `task_console.bat` | `DELETE/REDIRECT ROOT` | 第二入口，无独立产品必要。 |
| `device_manager.bat` | `DELETE/REDIRECT ROOT` | 与统一 experiment composition冲突。 |
| `pulse_editor.bat` | `DELETE/REDIRECT ROOT` | 与root wrapper重复且bootstrap语义不同。 |
| `figure_viewer.bat` | `DELETE/REDIRECT ROOT` | 同上。 |
| `README.md` | `DELETE/REWRITE AFTER ENTRY DECISION` | 声称会bootstrap当前 checkout，实际不会；还保留旧 Init后自动开 Pulse UI描述。 |

这不是“developer convenience”可以豁免的问题：同名入口在同一仓库里给出不同 import/environment truth，正是 stale standalone package重新进入产品的路径。若确需package-local双击文件，它们只能无逻辑地转发 root launcher。

## 9. Testbench、fixture 与自动化真实边界

### 9.1 没有 RTL simulator 被普通测试自动运行

仓库没有 tracked xsim/iverilog/verilator runner。普通 Python suite 对 FPGA 的自动保护只有：

- required asset `is_file()`；
- `zlc_geometry.vh` 与 generator文本相等；
- command rising-edge规则的源码 regex；
- Python/memory/cycle reference models；
- batch CRLF与少量 launcher调用。

它不检查 Tcl syntax、Vivado IP property、XDC clock、top/XDC lane mapping、Verilog compile、bench transcript、`$display("BAD")`，也不检查 replay fixture由当前 packer生成。`test_fpga_assets.py` 中“meaning is verified below”的措辞大于真实覆盖。

多数 bench 即使打印 `FAIL/BAD` 也只 `$finish`，process仍 exit 0；没有统一 `$fatal` 或 transcript parser。因此未来“把所有 .v 跑一遍”仍不够，必须先定义 process-failing oracle。

### 9.2 逐 testbench/module/helper 裁决

| testbench / module | helper task/function | 裁决 | 当前价值与盲点 |
|---|---|---|---|
| `tb_1tick.v::tb_1tick` | `pa` | `REDESIGN/MERGE`；helper `PASS HARNESS` | 统计 bad但无可靠pass/fail；只看变化时刻，不核完整mask/20 edges。 |
| `tb_gapsweep.v::tb_gapsweep` | `pa` | `KEEP/MERGE`；helper `PASS HARNESS` | 有 BAD 文本但无runner/oracle exit。与1tick可合成table-timing suite。 |
| `tb_loop.v::tb_loop` | `pa` | `KEEP WITH FIX` | 打印 expected count，却不在不符时 `$fatal`。 |
| `tb_edge_streamer.v::tb_edge_streamer` | 无独立helper | `DELETE/MERGE DIAGNOSTIC` | 旧 latency variant；behavioral extra pipeline不代表当前real IP，无self-check。 |
| `tb_real_engine.v::tb_real_engine` | `pa_write` | `KEEP DIAGNOSTIC ONLY` | real tick/mask BRAM接口有定位价值；无oracle，不得称 definitive。 |
| `tb_scan_wrap.v::tb_scan_wrap` | `load_chunk_into` | `KEEP WITH NEW CASES` | 自检较好；只测10-tick frame，缺首bank、1/2 tick、sticky underflow。 |
| `tb_ramp_scan.v::tb_ramp_scan` | `prog_seg`, `expect_v` | `KEEP WITH DEBT` | ramp/scan组合有价值；behavioral RAM并跳过尾2 ticks，不能证明real IP pipeline。 |
| `tb_delay_sched.v::tb_delay_sched` | inline stimulus | `KEEP WITH FINITE CASE` | 覆盖TTL scheduler；repeat-forever，缺 finite DONE/tail。 |
| `tb_delay_compact.v::tb_delay_compact` | inline stimulus | `KEEP` | compact mapping有价值；缺invalid/noneligible channel/error。 |
| `tb_evt_depth.v::tb_evt_depth` | inline stimulus | `REDESIGN` | 当前把silent overflow/drop锁成成功；应反向验证 sticky ERROR。 |
| `tb_bus_delay.v::tb_bus_delay` | `prog_seg` | `KEEP WITH FINITE CASE` | bus delay有价值；repeat-forever，缺physical safe tail。 |
| `tb_da_ttl_align.v::tb_da_ttl_align` | `prog_bus_seg`, `run_with_delay` | `KEEP MODEL` | 算法对齐有价值；不是full-top/physical IO proof。 |
| `tb_da_clk_phase.v::tb_da_clk_phase` | `prog_seg`, `capture` | `KEEP AS DESIGN MODEL ONLY` | 人为0.5–3.2 ns skew与固定latch edge，不能替代STA/板测。 |
| `tb_t_ff.v::jtag_axi_0` | 无 | `KEEP STUB` | full-top模拟替身合理；不是real JTAG IP。 |
| `tb_t_ff.v::axi_bram_ctrl_0` | `wr`, `cmd`, `upload`, `prepare_and_fire` | `KEEP STUB / REDESIGN ORACLE` | host sequence harness有价值；固定literal image且不验证真实AXI。 |
| `tb_t_ff.v::tb_t_ff` | `report_fire` | `REDESIGN` | 只比较同一 FIRE 内 F0/F1/F2，未比较两次 FIRE；最终总打印DONE。 |
| `tb_uart_pipeline.v::tb_uart_pipeline` | `send_byte`, `recv_byte` | `KEEP / MOVE UNDER sim` | 证明受限WRITE pipeline；缺CRC bad、截断、oversize、timeout。helpers可留。 |
| `tb_uart_read_tap.v::tb_uart_read_tap` | `send_byte`, `recv_byte`, `send_arr`, `check_reply` | `KEEP / MOVE UNDER sim` | read/reply自检有价值；缺address/count bounds与watchdog。helpers可留。 |

`tb_uart_pipeline.v`、`tb_uart_read_tap.v` 位于 source root 而非 `sim/`，sim README又没有完整列出它们；这是资产组织债，不表示它们被自动运行。推荐移入统一 simulator suite，runner显式列 top、sources、expected pass marker并将 `$fatal` 映为非零。

### 9.3 replay fixtures

| 文件 | 裁决 | 理由 |
|---|---|---|
| `sim/replay_t.vh` | `REDESIGN/GENERATE` | 手工 sparse words，无来源manifest或当前 packer parity。 |
| `sim/replay_t_frame.vh` | `REDESIGN/GENERATE` | 同上；只被 `tb_t_ff` include。 |
| `sim/.gitignore` | `PASS` | 隔离 simulator outputs正确。 |
| `sim/README.md` | `PASS HONEST LIMIT / REDESIGN CLAIMS` | 承认无runner是诚实的；却对`tb_t_ff`跨run determinism过度声明。 |

推荐由当前 host `pack_program()` 生成 deterministic fixture，并在 Python test逐word核对 source input、geometry和 expected digest；或者删掉 replay模式，bench内只写一个极小、可人工审阅的 image。不能继续保留无法追溯的 literal又称它代表 real host payload。

### 9.4 自动测试应新增的最小矩阵（设计建议，未实施）

1. lint/elaboration：默认冻结 profile完整编译，所有 warning allow-list；
2. engine RTL：1-tick、loop、scan、ramp、TTL/DAC delay，与 Python reference逐tick对比；
3. negative：first-bank missing、late refill sticky underflow、TTL/DAC FIFO overflow sticky ERROR；
4. completion：finite program + nonzero delay，DONE只能在 physical tail结束后；
5. SAFE：CLK_ENABLE=1时单命令立即到pin-safe，断开host也不恢复；
6. UART：bad CRC、truncated timeout、oversize count、illegal READ address、JTAG最终恢复；
7. full top：generated current host image，两次 FIRE跨run waveform相等；
8. build static gate：XDC包含expected clock，lane manifest与top/XDC生成投影相等；
9. Vivado-only recovery qualification：critical IP readback、zero unexpected unconstrained、setup/hold、artifact receipt；这类测试不混入普通无Vivado suite，但也不能省略。

## 10. 文档、注释与实现冲突

文档只作为冲突证据，不据此覆盖代码事实。

### 10.1 `packages/zlc_pulse/fpga/README.md` — `REDESIGN`

- 引用不存在的 `REAL_HARDWARE_BRINGUP_zh.md`、`MAINTAINER_NOTES.md`、`SYSTEM_ARCHITECTURE_DESIGN_zh.md`；
- 把 `build_and_program.bat`、`run_server.bat` 写成 FPGA目录内路径，实际在 root `bin/`；
- 引用不存在的 `pulse_gui.bat`；
- runtime chain只写JTAG，实际默认 transport为 UART-first auto；
- 声称 resolver包含 repo `.venv`，脚本反而明确拒绝 `.venv`；
- 把现有 build/timing/fingerprint描述成完整资格证据，实际缺clock/pin/build identity。

不能只修路径；必须在用户裁决 SAFE/DONE/board ABI/build identity后重写能力边界。

### 10.2 `pulse_streamer/README.md` — `REDESIGN`

- 同样引用三个不存在文档与错误 bat路径；
- Files清单漏 `zlc_uart_bridge.v`、`program_flash.tcl`；
- 称JTAG是current default，实际 auto优先UART；
- estimator数字已漂移：当前纯计算约 RAM 78%、LUT 85.2%、FF 27.6%、DSP 57.8%，不是文档LUT约97%、FF约22%；
- timing/real-engine措辞超出当前自动/物理证据。

### 10.3 `board_config/README.md` — `REDESIGN AFTER BOARD ABI DECISION`

- 声称 XDC 是 single runtime mapping source，但 top手写完整第二 mapping；
- `bin\run_server.bat` 中混入真实 CR (`0x0D`)，渲染成 `bin` + `un_server.bat`；
- 声称正常 server/client不会不同 mapping，却没有 hardware pin ABI handshake；
- 没有提醒 `GND*` 是普通被驱低IO而非物理地。

### 10.4 config/RTL inline comments — `FIX AFTER CONTRACT`

- `streamer_config.json` 说 clock必须等于 XDC `create_clock`，但 constraint不存在；
- config说除 host-only cap外每个 params field都进 fingerprint；`slot_mul_width` 位于 `params`却不属于 `StreamerParams`，只有冻结guard、不进word63；
- UART RTL注释引用不存在的 model/testbench；
- `create_project.tcl` 注释称 build必须留在 `fpga/build`，代码允许任意 override并delete；
- command strobe test锁定 `ldr_cmd_clear` 永远不清的实现形状，不是public outcome；
- sim README说 `tb_t_ff` 证明 repeated runs first frames identical，bench没有跨FIRE比较。

### 10.5 相邻 package/root docs

`packages/zlc_pulse/README.md` 多次引用不存在的 `fpga\run_server.bat`，并同时说“RTL/bitstream external，本仓不build/program”；实际仓库跟踪完整RTL/Tcl与root recovery launcher。正确边界应是“普通实验不构建，批准的 recovery workflow在本仓构建/烧写”，而非否认代码存在。

`packages/zlc_workbench/bin/README.md` 声称package wrappers bootstrap当前 checkout，实际 `_launch.bat` 直接 bare import；它还保留旧 Init后自动开 Pulse UI流程，与当前 on-demand Pulse Control冲突。

## 11. 建议的最小正确目标设计（供用户裁决，未实施）

### 11.1 三份 truth，各自只负责一种 identity

1. **Board manifest**：显式 `lane_index → logical signal → RTL port → package pin → electrical role`，含 oscillator、DAC timing和board ID；生成/验证 host `PulseTarget`、top mapping、XDC，不再解析文本出现顺序获得identity。
2. **Streamer geometry**：edge/scan/bus/delay数值与固定 profile constants；严格生成 host params、`zlc_geometry.vh`、Vivado sizing Tcl和 layout ID。build/server不允许 fallback。
3. **Firmware build receipt**：source/constraints/board/geometry digest、part、Vivado/IP版本、critical IP readback和STA summary；给bitstream一个可从硬件读取的 build ID。layout-compatible与qualified-build是两个不同判断。

不能把三者继续压成一个 word 63 CRC，也不能让 source hash同时假装是 board identity与qualification。

### 11.2 物理状态机

推荐明确以下状态/事件：

~~~text
SAFE_PINS
  → LOAD_IMAGE (pins仍safe；CLK_ENABLE只进入shadow)
  → ARMED (first scan bank/geometry/error checks完成)
  → RUNNING
  → LOGICAL_TERMINAL
  → DRAINING_DELAY_AND_FINAL_DAC_LATCH
  → PHYSICAL_DONE + SAFE_PINS
~~~

关键 invariant：

- 任意时刻收到 SAFE，硬件独立于host后续write进入 `SAFE_PINS`；
- LOAD 不可让physical clock提前输出；
- public DONE只代表 `PHYSICAL_DONE`；
- UNDERFLOW/OVERFLOW/PROTOCOL ERROR sticky到明确ack/reset；
- scan point0也必须先resident；
- hardware不支持的custom safe/geometry在compile/load前拒绝。

### 11.3 构建与烧写边界

- build工具先严格解析 canonical config与board manifest，原子生成所有 projection；
- destructive project cleanup只能发生在已验证 generated build root内；
- critical IP property全部 set/readback hard-fail；
- STA明确验证50 MHz engine、CDC/I/O contract和零意外unconstrained endpoint；
- artifact与receipt一起产生；program-only也必须读receipt；
- program/flash显式选择并验证target/device；flash另有确认；
- reboot后读 layout + board + build identity，再宣告成功。

## 12. 需要用户裁决

1. **DONE定义**：是否必须等所有TTL/DAC delayed output和末尾latch完成且safe？审计推荐“是”；若保留logical done，必须另命名且public wait默认等physical done。
2. **SAFE强度**：是否要求单条command、无后续host write也能物理safe？审计推荐“是”，这是硬件boundary invariant。
3. **lane/pin owner**：A由显式board manifest生成host/top/XDC；B以top固定lane ABI、XDC只管pin。推荐A；拒绝继续靠XDC行序+手写top混合。
4. **FPGA profile**：唯一冻结35T board profile，还是支持任意geometry？推荐先正式冻结，删除假参数化；真需要第二profile时再用多profile RTL tests证明。
5. **短scan point**：允许最短多少tick？推荐先声明并校验真实RD_LAT下限；若实验必须1-tick scan，批准增加scan prefetch/FIFO。
6. **FIFO overflow政策**：推荐sticky ERROR并中止/拒绝run；是否存在任何理由接受silent drop？审计认为没有。
7. **firmware identity**：只保证layout compatibility，还是硬件必须证明exact qualified build？推荐加入独立build ID/receipt，但这是新增ABI，需用户明确批准。
8. **program默认动作**：无参数 `build_and_program.bat` 是否仍直接program？推荐默认build-only，volatile program和永久flash都显式选择。
9. **多板选择**：由serial/target URL/IDCODE/board ID中的哪组字段选择？在答案前脚本应 exactly-one fail closed。
10. **`GND1..15`真实用途**：它们是reserved-low/guard，还是实验接线把FPGA IO当ground？必须由硬件负责人裁决。
11. **custom DAC safe value**：产品需要per-bus custom值，还是统一midpoint？推荐短期只允许硬件真实支持的midpoint并拒绝其他值。
12. **tracked `build/geom.tcl`**：删除并每次原子生成，还是让它成为 direct Tcl强制输入并有exact parity test？推荐前者。
13. **Vivado版本**：批准并固定哪个tool/IP版本？当前自动选“找到的新版本”与 fail-open property不可共存。
14. **UART信任边界**：是否永久只允许trusted host？即使是，watchdog仍需要；count/address error是hard reject还是协议reply需确定。

## 13. 推荐收口顺序

1. **先阻止危险恢复动作**：修 project delete containment、program/flash target选择；在主clock约束完成前不得把新bitstream标为qualified。
2. 用户裁决 SAFE、DONE、overflow、scan最短周期；把它们写成host/RTL共同可执行contract。
3. 建 explicit board manifest，消除 XDC行序与top手写mapping的双truth；处理 `GND*`硬件含义。
4. 补 hardware pin-safe gate、physical-done drain、sticky status、first-bank与UART watchdog/bounds。
5. strict config projection + build receipt + critical IP readback +真实STA gate；固定Vivado版本。
6. 将现有 benches接入统一nonzero-failure runner，先覆盖上述 P0/P1，再考虑清理诊断bench。
7. 删除 dead RTL seam、tracked/history fixture和package duplicate launchers。
8. 最后重写 README；不得先靠文案把未闭合的物理契约写成已解决。

## 14. 最终裁决摘要

当前设计不是“全部重写”的对象。应保留：单一tick owner、edge/ramp算法、ping-pong scan思路、per-signal delay scheduler、CRC后commit、top作为硬件composition root。真正需要重做的是边界语义和证据链：

- **物理安全边界**：SAFE/LOAD/clock mux；
- **完成与错误边界**：DONE/drain/sticky underflow/overflow；
- **身份边界**：geometry ≠ pin ABI ≠ qualified firmware；
- **构建边界**：constraint/IP/tool/target必须fail-closed；
- **测试边界**：diagnostic Verilog文件不等于自动RTL回归。

在这些边界闭合前，Python测试全绿只能说明host投影与静态资产仍能自洽，不能说明板上输出安全、时序正确或每个shot完整。
