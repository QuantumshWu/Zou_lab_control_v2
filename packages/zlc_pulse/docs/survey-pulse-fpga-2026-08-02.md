# 任务7 审查报告:zlc_pulse + fpga host 侧 vs "pulse device 极简化"目标

范围:`zlc_pulse`(31 文件 ≈14.7k 行)、`fpga/pulse_streamer/host`(≈2.8k 行)、`zlc_neutral_atom/devices/sequencer` + 相关消费方(≈2.5k 行)。全程只读。

---

## 1) 现状分层与链路

### 链路图(编译期 → 执行期)

```
[授权模型]  PulseDocument (document.py:392)
   │  周期表 periods(digital states + AnalogStep)+ delays + RepeatRegion
   │  + scan_parameters/FrozenScanTable/scan_recipe + api_parameters
   │  + target: PulseTarget (target.py:96, raw_lanes+逻辑口+ABI指纹)
   │  编辑操作全在 authoring.py(insert/move/remove period, cycle_field_binding,
   │  replace_pulse_field, freeze_scan_table, resolve_api_parameters...)
   ▼
[编译]  compile_pulse_document (compiler.py:78) ──→ TargetIR (ir.py:74)
   │      静态: _compile_static;扫描: _compile_scan(N-slot 仿射降低)
   │  compile_pulse_artifact (compiler.py:130)
   │      = 源文档 digest + TargetIR + PulseWireImage + trigger_schedules
   ▼
[打包]  pack_target_ir (fpga.py:67) → PulseWireImage(稀疏 (addr,word) 表)
   │      实际序列化在 fpga/pulse_streamer/host/image.py:517 pack_program
   │      几何单源 StreamerParams (image.py:178) / region_bases (:316)
   │      / CtrlWords 寄存器映射 (:109) / build_fingerprint (:83)
   ▼
[RPC]  CompiledPulseArtifact 二进制编码 ──rpyc──→
   ▼
[服务]  PulseExecutionService (server.py:287)
   │      状态机 IDLE/VALIDATING/PREPARING/PREPARED/FIRING/RUNNING/
   │      COMPLETING/DONE/TIMEOUT/FAILED/SAFE/INTERRUPTING/SAFE_FAILED
   │      单连接所有权 + connection_generation + PreparedPulseRef
   ▼
[后端]  DeployedStreamerSession (transport/session.py:81)
   │      自己的第二套状态机 + observer worker 线程(:850)
   │      ping-pong 扫描 bank 流式补给 (:961) + 三相 SAFE (:757)
   │      + 终态证据采集 (:1055)
   ▼
[传输]  RegisterTransport 协议 (session.py:54)
          ├ VivadoAxiRegisterTransport (transport/axi.py, JTAG/tcl 子进程)
          └ UartRegisterTransport (transport/uart.py, 3Mbaud 帧协议)
          + InterprocessDeviceLease (transport/lease.py)

[客户端]  RemotePulseExecutionClient (client.py:24)
          双 rpyc 连接(控制 + 中断)对接 server.py:932 的 RPyC 门面
   ▼
[zlc_neutral_atom 设备层] —— 第三套状态机
   remote_pulse.py:31 RemotePulseExecutionEndpoint
   → endpoint.py:52 _SequencerSessionOwner(session/epoch/superseded 再来一遍)
   → port.py:674 PulseSession + BoundPulsePort(命令/应答对象:Prepare/Fire/
     Complete + Ack,port.py:244-361)
   → application.py:578 prepare_pulse_execution(bind target → resolve api
     values → materialize_scan_sweeps → compile → RunPlan)
   → capture/binding.py:436 bind_triggered_camera_acquisition(把 artifact 的
     trigger_schedules 和相机 layout 对账)
```

**关键事实:`RuntimeSequenceProgram`/`RemoteSequencer` 在此分支已不存在**(旧 main 的概念),被 `TargetIR` + `PulseExecutionService`/`RemotePulseExecutionClient` 取代。`pack_program` 通过鸭子类型仍兼容旧程序对象(fpga.py:77 用 `SimpleNamespace` 拼 carrier)。

**依赖方向是干净的**:`zlc_pulse` 只 import 标准库、numpy、`zlc_storage`、`fpga.pulse_streamer.host`(实测 grep,无 GUI/measurement/neutral_atom 依赖)。问题不在 import 方向,而在**语义装了多少外界概念**(见 §2)。另外注意 `zlc_pulse` 依赖顶层独立包 `fpga` —— 若 zlc_pulse 要独立成仓,`fpga/pulse_streamer/host` 必须并入(它本来就是 host↔bitstream 契约的单源,见 §4)。

---

## 2) 违背"device 不知道外界"的逐处点名

按严重度排序:

### A. 编排/测量概念长进了 device 数据模型
1. **`PulseDocument.scan_sweep_count`(document.py:404)** —— "扫几遍"是 run 级编排量,却持久化在脉冲文档里;真正用它的是 `materialize_scan_sweeps`(scan_execution.py:72,把表复制 N 遍)——由 application.py:602 在编译前调用。sweep 应属编排层参数,不属 sequence 模型。
2. **`trigger_schedules` 长在 CompiledPulseArtifact 里(artifact.py:48)**,编译器收 `trigger_channels` 参数(compiler.py:141)。这是"某条数字通道是相机触发、每个 scan 点触发几次"的**测量层知识**。后果连锁:
   - compiler.py:185-204:因 output-delay 展开会破坏源循环分组,编译器要**用无延迟文档整个重编译一遍**再按序号拼回 trigger 分组 —— 纯粹为了给测量层保留 per-point/per-loop 归属;
   - artifact.py:89-100:artifact 构造时验证 schedule 是 IR 的确定性展开(又算一遍);
   - server.py:265 `PulseCompletion.expected_trigger_counts_from_completed_schedule`:**FPGA 服务器的完成回执里携带相机应收帧数** —— 硬件根本没数过触发,这是 host 编排知识借道 device 协议回传。
   触发时刻表本身是纯函数 `f(TargetIR, channel)`(schedule.py 的推导是确定性的),完全可以由编排层在 host 侧自行调用,device/artifact/server 三处都不需要知道。
3. **`physical.py` 的 "readout-context index"**(docstring L10-16 自述:为读出窗口取 held 数字/DAC 值)—— 读出测量的积分窗口语义放进了 pulse 包。窗口投影函数可以保留为纯查询,但命名与约束(拒绝 ramp 等)是替测量层做的决定。

### B. GUI/编辑器概念长进了包里
4. **`PulseDocument.visible_ports`(document.py:401, 488-497)** —— 编辑器"显示哪几行"的 UI 状态,参与文档指纹、参与 `bind_pulse_document_target` 的重映射(binding.py:75, 110-112)。纯 UI 态混入设备模型与内容寻址。
5. **`ScanRecipeProvenance`(document.py:330)+ `scan_program.py`** —— 文档里存 GUI 扫描程序的 **Python 源码文本** + normalizer id + 定义 digest;`evaluate_numeric_scan_program`(scan_program.py:64-95)在包内 `exec()` 用户代码。这是编辑器工作区职责。
6. **`scan_template.py` 全文件** —— 生成给人看的 numpy starter 源码字符串,`SWEEP_SCAN_SLOT/SWEEP_API_SLOT`(:25-26)明说"是前端表单的两个选项"。纯 GUI。
7. **`scan_columns.py:272 / :311-314` 的起始扫描范围启发式**(`max(nominal*2.0, 100.0*tick)`)、`template_description.py`(自述服务 "Pulse-slots form")、`authoring.py:80-113` 的 `PulseEditImpact/DestructivePulseEditError`(为 GUI 确认对话框准备的影响清单)—— 都是编辑器支撑代码。约 2000+ 行(authoring 1384 + scan_columns 326 + template 96 + scan_template 145)属于"授权/编辑器库",不属于 device。
8. **`document.py:43-44 DEFAULT_REPEAT_COUNT=2 / MIN_REPEAT_COUNT=2`、`DEFAULT_PERIOD_DURATION`** —— 编辑器默认值当作模型常量。

### C. 实验室部署事实硬编码进包
9. **`deployment.py:31 APPROVED_DEPLOYED_TARGET_ABI`** —— 一个具体实验室 target 的 sha256 写死在库常量里,`validate_deployed_target`(deployment.py:148-193)又按 StreamerParams 结构性复推一遍 lane 布局。同一事实三重表达(hash + 结构检查 + assets/deployed_target.json 打包资源,target.py:254-258)。
10. **`server_app.py` + RPyC + 双连接客户端** —— 远程部署形态(rpyc、interrupt 双连接、connection_generation)焊死在包 API 里。client.py:49-59 强制两条独立连接,原因是 rpyc ThreadedServer 单连接阻塞在 `complete()` 时无法响应 safe_state —— 传输框架缺陷倒灌成协议复杂度。

### D. 设备层之上的三重状态机(最大的偶然复杂度)
`DeployedStreamerSession`(session.py,自带 epoch/superseded/observer)之上是 `PulseExecutionService`(server.py,又一套 epoch/superseded/VALIDATING…)之上是 `_SequencerSessionOwner`(endpoint.py:52,第三套 session_id/run_id/operation_epoch/physical_operation_in_flight)再包一层 `PulseSession`(port.py:674)。**四层几乎同构的 prepare/fire/complete/safe 生命周期防御,每层各有锁、代际计数、"superseded"检查**,合计约 3000 行。这是"device 不知道外界"最直接的反面:run_id/session_id/binding_stamp/capability_fingerprint(endpoint.py:37-50, remote_pulse.py:79-105)全部是编排层身份,被迫穿透到设备栈每一层。

---

## 3) scan slot / api slot 硬件模型:资产提炼

### 值得保留的硬件模型(这是真资产)
- **N-slot 仿射时刻**:每条边 `有效tick = base + (Σ coeff_j · slot_j) >> frac_bits`,单公式单源 `evaluate_affine_tick`(ir.py:391-405);loop 端点同样仿射(`loop_end_slot_coeffs`)。
- **全局边沿表**:`ticks/masks` 每边一行,62 通道 mask,末 mask 强制 0(安全终态,ir.py:121)。
- **流式 scan 表**:scan 点 = 长度为 slot_count 的整数行;硬件持有双 bank(`bank_size`×2)ping-pong 窗口,host observer 经 `CURSOR/BANK_READY/BANK0_CHUNK/BANK1_CHUNK` 邮箱补给(image.py:123-128, session.py:961-1053),表长可超驻留窗口,FPGA 全程拥有点计时(无缝)。
- **DAC 总线段**:`TargetBusSegment` edge/ramp,`value_select` 指到 DAC slot ⇒ **DAC 值也可被 scan 表逐点驱动**;段延时、通道延时事件调度(32b/信号,R_DELAY 区)。
- **compact 有限 bracket**(loop_start/end/count)+ repeat_forever。
- 几何守卫 `check_rtl_assumptions`(image.py:406,拦截"能综合但会静默腐坏"的几何)。

### 关键观察:api slot 目前没有硬件通道
`resolve_api_parameters`(authoring.py:924)是**纯文档替换**:改一个 API 值 ⇒ 整文档重编译 ⇒ 整个 wire image 重上传(session.prepare 全量写,session.py:373-383)。而硬件的 slot 机制其实天然支持"编译一次、只改值":**把 api slot 编译成 slot 绑定 + 一行 scan 表,改值 = 重写一行 scan bank(几个字)**,边表/系数/总线段全部不动。当前唯一接近的形态是 `STATIC_REFERENCE_POINT` 执行形式,但它也走全量重编译。这是重写时把用户目标("便捷改值")和现有硬件资产(N-slot 仿射)对齐的最大机会。

同理 **scan 表换表不必重编译**:scan 区独立于边表区(region_bases,image.py:316-333),硬件本就支持只重写 scan 区;但现在 artifact 把表熔进 IR 指纹,换表必须全链路重来。

### 最小 host API(方法签名级)

```python
# ---- 纯函数层(无 I/O)----
def compile(seq: PulseSequence, geom: StreamerGeometry, clock_hz: float
            ) -> CompiledProgram:
    """seq 含 slot 声明(duration/DAC 字段 → slot_id);产物含边表/系数/
    总线段/延时/clk_enable + slot_kinds;不含 scan 表、不含触发调度。"""

def pack(prog: CompiledProgram, geom: StreamerGeometry) -> dict[int, int]
def pack_scan_rows(rows: np.ndarray, geom, bank: int, chunk: int
                   ) -> dict[int, int]
def trigger_times(prog: CompiledProgram, channel: str,
                  table: np.ndarray | None) -> np.ndarray   # 编排层自取,device 不管

# ---- 设备层 ----
class PulseStreamer:
    def __init__(self, transport: RegisterTransport,
                 geom: StreamerGeometry, clock_hz: float): ...
    def open(self) -> None            # 读 CTRL[63] 比对 build_fingerprint(geom),不符即拒
    def close(self) -> None
    def load(self, prog: CompiledProgram) -> None          # 上传边表/系数/总线/延时 + LOAD
    def write_slots(self, values: Sequence[int]) -> None   # api slot:一行表写入 bank0 行0
    def write_scan_table(self, rows: np.ndarray) -> None   # scan slot:预载前两 chunk,
                                                           # 长表由内部 observer 流式补给
    def fire(self, *, forever: bool = False) -> None
    def wait_done(self, timeout: float | None) -> DoneReport | None
                                       # DoneReport = {status双读, cursor双读, underflow, tail_elapsed}
    def cursor(self) -> int | None     # 非阻塞进度
    def safe(self) -> SafeReadback     # 三相 SAFE + {status==0, clk_enable全0} 读回
    def snapshot(self) -> dict
```

`PulseSequence` 侧(设备无关模型)保留:`periods(states, analog_steps)`、`delays`、`repeat`、`slot 声明(kind, field_ref, unit)`、`target(lanes/ports/ABI)`。删除:`visible_ports`、`scan_sweep_count`、`scan_recipe`、`api/scan 双命名空间之分`(统一为 slot;"api 用法"与"scan 用法"只是 host 端一行表 vs 多行表)。

---

## 4) 指纹/握手/校验:留什么删什么

### 必须保留(真 ABI 边界,有实战战绩)
| 机制 | 位置 | 理由 |
|---|---|---|
| **geometry fingerprint**(CTRL word 63 读回 vs `build_fingerprint(params)`) | image.py:55-95, session.py:222-231 | host↔bitstream 唯一运行时 ABI 证明;注释记录了它抓住的两起真实静默腐坏(CLK_ENABLE 移位、bus_seg_addr_width 平移 R_DELAY 896 字) |
| **StreamerParams 单源三投影**(`emit_geometry_vh`/`emit_geom_tcl`/pack 同源) | image.py:1209, 1244 | 几何改一处,RTL 头/Vivado tcl/host 打包全跟;指纹自动折入新字段(fail-safe 设计,:74-80) |
| **check_rtl_assumptions** | image.py:406-470 | 打包溢出/寻址别名的硬门 |
| **target ABI 指纹相等检查**(connect 时 server 声明 target,client 比 `abi_fingerprint`) | target.py:146-164, server.py:701 | 逻辑口→lane 映射是真 ABI;一次相等比较,便宜 |
| **SAFE 读回证明**(status==0 稳定两次 + clk_enable 全 0) | session.py:704-802 | 物理安全事实,非仪式 |
| **终态双读纪律**(STATUS×2、CURSOR×2、underflow) | session.py:1055-1090 | 寄存器读的抗毛刺基本功;但**保留读法、删掉证据对象体系**(见下) |

### 可以删(仪式性/多重冗余)
1. **内容寻址全家桶**:每个 dataclass 的 `to_tree/from_tree` 精确字段集校验 + schema 字符串 + canonical digest + "decode 后 re-encode 必须逐字节相等"(artifact.py:167-176)。document.py 约 500/1271 行、ir.py 约 230/657 行、server.py 约 350/1072 行是编解码仪式。设备协议用普通序列化 + 一个 payload sha256 足够。
2. **同一 artifact 的四重验证**:TargetIR `__post_init__` 逐 scan 点复验单调性/时长(ir.py:299-375,O(points×edges))→ compiler 再验 → `validate_target_ir_for_target/geometry` 三验 → server `_validate_artifact` **重打包 wire image 逐字比对**(server.py:708)→ session `_validate_artifact_against_bound_deployment` 又重打包(deployment.py:213-231)。`FrozenScanTable` 里的 `_validated_contract` 缓存黑魔法(document.py:264-326,frozen dataclass 上 object.__setattr__ 记忆化)就是这种重复验证代价的症状。保留一处(编译后),digest 传递信任即可。
3. **artifact 三表示同传**:源文档 digest + TargetIR + wire image 一起过 RPC(artifact.py:42-48)。选一个:传 IR 由 server pack(server 已有 geometry),或传 wire words + 少量元数据。
4. **evidence 体系**(evidence.py 380 行):`STATIC_STATUS_READ_RECIPE` 字符串、每份证据自带指纹、`PostTerminalTailEvidence` digest 链到 terminal 指纹、recipe id 校验。本质是 `wait_done() -> DoneReport(status, cursor, underflow, tail_ok)`,一个 dataclass 完事。
5. **connection_generation + PreparedPulseRef + 双连接中断通道**(server.py:121-129, client.py:49-59, 194-245):单客户端所有权可由 TCP 连接生命周期承载;"load 返回 digest、fire(digest)"即可;若换掉 rpyc(如自定义帧协议或 zmq),阻塞-complete 期间响应 abort 不需要第二条连接。
6. **APPROVED_DEPLOYED_TARGET_ABI 硬编码 hash**(deployment.py:31)与 `validate_deployed_target` 结构复推:二选一,建议留结构检查(它由 StreamerParams 推导,可移植),删实验室专属 hash 常量。
7. **`PulseExecutionForm` 五态**(artifact.py:33-38):可折成两个正交布尔(有无 scan 表 × repeat_forever);`STATIC_REFERENCE_POINT` 是文档变换(`nominal_scan_reference`)不是执行形式;由此 artifact.py:69-100 的形式-内容一致性检查全部消失。
8. **`bind_pulse_document_target` 的物理签名重映射**(binding.py:53-144):为"文档 target 键名与 live target 不同"服务。若规定键名稳定(ABI 指纹相等即通过,否则拒绝),此文件缩成 10 行。

---

## 5) 重写提案:独立 zlc_pulse 包

### 包边界(三个子域,一个包)

```
zlc_pulse/
  model.py      # PulseSequence + PulseTarget:设备无关授权模型(≈500 行)
  compile.py    # PulseSequence → CompiledProgram(N-slot 仿射降低,≈900 行,
                #   现 compiler.py 逻辑基本原样保留——它是好的)
  wire.py       # 现 fpga/pulse_streamer/host/image.py 并入:StreamerGeometry、
                #   CtrlWords、region_bases、pack、build_fingerprint、
                #   check_rtl_assumptions、emit_geometry_vh/emit_geom_tcl(≈1300 行)
  device.py     # PulseStreamer(§3 签名):现 DeployedStreamerSession 的
                #   物理协议部分(LOAD/FIRE/SAFE 三相/observer/bank 补给/终态双读),
                #   删 epoch/superseded 层(单线程外加一个 observer,≈800 行)
  transport/    # RegisterTransport 协议 + axi/uart/lease(现状基本可平移,≈1000 行)
  remote.py     # 可选:瘦 RPC 门面(load/write_slots/write_scan_table/fire/
                #   wait_done/cursor/safe/snapshot 逐一转发,无第二状态机,≈300 行)
```

预计 ≈5k 行,对现 ≈20k 行(zlc_pulse+fpga host+neutral_atom 设备层)。

### 明确移出包的
- 编辑器支撑:authoring 编辑操作、PulseEditImpact、scan_template、template_description、scan_columns 启发式、scan_program 的 exec → 随 pulse 编辑器走(它们消费方全在 `zlc_workbench/pulse_editor` 与 `logic_nodes/pulse_scan/ui`,实测 grep)。
- 触发归属:`trigger_times(prog, channel)` 纯函数保留(推导逻辑是资产),但从 artifact/server/completion 中全部拆出,由 capture/编排层调用。compiler.py:185-204 的 grouping 双编译随之消失。
- sweep 展开、api 值解析、run/session 身份:全归编排层。设备只见 `CompiledProgram` + 表 + 命令。
- timeline/simulation/physical 投影:保留为 `zlc_pulse.views`(可选模块)或随编辑器/虚拟设备走 —— 虚拟后端(`devices/simulation/sequencer_endpoint.py`)是 `build_pulse_playback` 的正当消费者,建议留 `simulation.py` 在包内(它只依赖 artifact/ir),删 physical.py 的 readout 专用索引。

### 与未来 runtime/scan 编排的边界(一句话协议)
编排层持有 `PulseSequence` 与 `PulseStreamer`;它 `compile` 一次,之后**改参数走 `write_slots`(一行表),扫描走 `write_scan_table`(多行表),两者都不触发重编译或边表重传**;相机同步由编排层用 `trigger_times` 自算;run_id/超时/重试/sweep 全在编排层;设备唯一向外暴露的事实是 `DoneReport/cursor/SafeReadback` 与 geometry/target 指纹。这与用户目标原话逐条对应,且比现状多兑现了一件事:api slot/scan table 的"免重编译改值"是现有 RTL 已经支持、host 栈尚未暴露的能力。

### 风险提示
- 上述 "write_slots 走一行 scan 表" 依赖 `SCAN_ENABLE=1` 路径对单行表的行为与静态路径完全一致(edge 系数为 0 的 slot 不受影响,仿射公式保证);建议重写时用 `fpga/pulse_streamer/host/engine_model.py`(现成的 RTL 周期级 host 模型,1159 行,测试资产,应保留)先行等价性验证。
- session.py 的并发防御虽冗余四层,但其中**单 I/O worker 所有权**(FIRE 后只有 observer 碰 STATUS/CURSOR/补给,session.py:461-466 注释)与**三相 SAFE**是从真机 bug 中长出来的,重写必须保留这两条纪律,删的只是层间重复的 epoch 仲裁。