# Step 6-H：`zlc_pulse` 剩余 Python source / tests / docs 全量只读审计

状态：完成（只读审计；仅新增本报告，不修改 production、tests、旧文档、RTL 或硬件）
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：`zlc_pulse/{__init__,canonical,codec,endpoint,engine_model,fpga,manifest,remote}.py`、`transport/*.py`、package tests、README/contract/notebook/metadata，以及根 launcher 对这些 Python 入口的调用。
分工：model/binding/compile/schedule/scan/device/wire 的 pulse 语义见 `04a`，Atom 调用见 `04c`；RTL/XDC/Tcl/Windows build/program 资产见 `06f`。本报告不审 RTL，只在 Python 与硬件 ABI/生命周期相交时引用 06f。

裁决词：`KEEP`、`REDESIGN`、`DELETE`、`MOVE TO TESTS`、`PASS WITH DEBT`、`USER DECISION`。

## 1. 结论先行

当前剩余 Python 面不是“remote/tests 很多所以大体安全”。本轮确认了四条新的硬件所有权 P0、两条操作安全 P0，以及一批明确 dead production seams：

1. Remote server 默认监听 `0.0.0.0`，协议没有认证、授权、TLS、allow-list 或 command integrity。任何能访问端口的 LAN client 都可成为 last-connected owner，执行 `load/fire/safe/close`。除非部署网络已有正式隔离/ACL 且被验收，否则这是直接的物理硬件控制入口。
2. `PulseRemoteServer.dispatch()` 不检查发请求的 client 仍是当前 owner。旧 handler 与新连接竞态时，旧 owner 在被替换并 SAFE 后仍可继续发命令。隔离探针已让 stale A 在 B 成为 owner 后成功执行 `close`，B 保持 owner 名义但 board session 已被 A 关闭。
3. takeover 会调用旧 owner 的 AUTO-SAFE，但 `claim_client()` 完全忽略 SAFE 的成功/失败返回。隔离注入 SAFE failure 后，新 client 仍被设为 owner，硬件保持 opened、物理状态未验证。
4. `InterprocessDeviceLease` 名称与行为相反：它只是 module-global dict；两个 Python process 对同一路径都能同时 acquire。更严重的是 UART/AXI transport 根本不使用它，现有测试只在同一进程自证。
5. Remote 默认 `auto` 会枚举并逐个打开所有 COM port，向每个端口发送 pulse UART probe。README 虽提醒“有其他仪器时传 `--uart-port`”，但危险动作仍是默认行为；在实验机上这可能向无关串口仪器发送协议字节。
6. Notebook 最后一个正式 cell 对真实 FPGA `fire(forever=True)`，让全部 TTL 以 1 µs 高/低交替、四路 DAC 在全幅范围运行，且 notebook 后面没有 finally/SAFE cell。README 声称五分钟 idle 自动 SAFE 和 `--client-idle-timeout`，当前实现没有该参数，tests 反而明确保证 quiet owner 永不释放。
7. `engine_model.py` 1158 行完全没有 production consumer。外部只剩 tests 使用 `effective_tick/reference_play/streaming_scan_play/ScanUnderflow`；其余约十余个 public mirror/function 连当前 tests 都无调用。它还重复定义 `_first_values`，并声称不存在的 host FIFO validator 已防 overflow。裁决：从 production package 移到 tests，只保留当前 tests 真正需要的独立 reference。
8. Pulse JSON tree codec 是唯一 model serializer，但不是严格 reader：未知键静默忽略、某些错误类型被 `str/int` 强制转换；实际文件 callers 使用普通 `json.loads`，会接受 duplicate keys 和 NaN。唯一 JSON 路径的设计意图尚未兑现。
9. `manifest.py` 仍从 XDC 文本声明顺序和命名启发式发明 physical target；没有精确匹配的 DAC latch clock 时会静默拿“第一个 clock”。这与 06f 的 pin/lane ABI P0 是同一根因。
10. AXI transport 接受负 word address 并 wrap 为 `0xFFFFFFFC`；timeout/Stop 后命令可能仍在 Vivado 中执行，close 不 join reader thread、kill 后不 wait；不能证明安全退役和零 worker/process。
11. `rewrite_scan_bank`（UART/AXI）、`read_words`（AXI）、transport-level `record_diagnostic` protocol、`DeviceLease` 等均无真实 caller，是 confirmed dead seams。

总体裁决：

- `canonical/codec/endpoint/manifest`：保留能力，收紧输入与 owner；
- `remote.py`：`REDESIGN IN PLACE`，先修 ownership/auth/safe，再谈拆文件；
- `transport/base|memory|uart|axi`：保留真实 transport，删除死缝并补真实生命周期；
- `transport/lease.py`：当前实现不能保留为“interprocess”；要么真正接入 OS lock，要么删除；
- `engine_model.py`：`MOVE TO TESTS / DELETE UNUSED CLUSTERS`；
- `fpga.py`：按 04a/06f，把 build implementation 从 `wire.py` 真正移回已有 owner；
- README/notebook/contract：当前有会导致 indefinite physical output 的直接矛盾，不能继续当操作手册。

## 2. 已复现的隔离证据

所有 Python 探针均先打印 repository root 与 `zlc_pulse.__file__`，确认解析到当前 checkout；未连接网络服务、串口、Vivado、FPGA 或其他硬件。

### 2.1 stale remote client 可在 replacement 后继续命令 board

Memory transport 上依次 claim A、claim B；B takeover 已对 A AUTO-SAFE。随后直接以 `client="A"` 调现有 `dispatch("close")`：

```text
owner_before_stale_dispatch=B opened=True
ZLC CLOSE client=A ... device_session=closed
owner_after_stale_dispatch=B opened=False
```

所以注释所称“旧 session 先结束，两者永不重叠”没有由代码实现。`dispatch` 只验证 method 名，不验证 owner；handler 在 socket 被 drop 前已经读到的 request 可以落入同一竞态。

### 2.2 takeover SAFE 失败仍授予新 owner

对 Memory streamer 注入 `safe()` failure：

```text
ZLC AUTO-SAFE FAILED client=A error="RuntimeError: injected SAFE failure"
owner_after_failed_takeover_safe=B opened=True
```

`client_disconnected()` 返回 `False`，但 `claim_client()` 丢弃返回值，继续设置 B。物理状态未知时不能发放新控制权；这是明确错误，不需要用户选择是否保留。

### 2.3 “Interprocess” lease 两个进程同时成功

两个独立 Python process 使用同一个临时 `pulse.lock` 路径：

```text
first=ACQUIRED
second_rc=0 second=ACQUIRED
```

文件从未创建/锁定；`_OWNERS` 只存在各自进程内。当前 UART/AXI constructors 也完全没有 lease caller。

### 2.4 codec 未知字段、强制转换、duplicate/NaN

对合法 `sequence_to_tree()` 结果加入 top-level 与 period 未知字段，并令 `period_id=None`：

```text
unknown_keys_accepted=True
coerced_period_id='None'
```

普通 caller 使用的 `json.loads`：

```text
{"format":"wrong","format":"zlc.pulse.v1","x":NaN}
duplicate_last_wins='zlc.pulse.v1'
nan_accepted=True
```

model constructors能抓部分非法数值，但抓不到被忽略的 typo，例如 `delays/repeat` 拼错后默认空/None。严格 artifact 不能把“模型最终还能构造”当 schema exactness。

### 2.5 canonical mapping key collision

```text
canonical_bytes({1:'lost', '1':'kept'}) == b'{"1":"kept"}'
canonical_digest({1:'lost','1':'kept'}) == canonical_digest({'1':'kept'})
```

当前真实 ABI inputs大多使用字符串键，因此这不是已发生的 target collision；但 public generic canonical helper 对不同 value 给出相同 digest。最小修复是只接受 string mapping keys并拒绝 collision，不增加新 hash。

### 2.6 Manifest 会猜错 latch clock

给 `_target_from_bindings` 一个 `foo[0:1]` DAC 和唯一但名称无关的 `unrelated_clk`：

```text
dac=foo silently_selected_latch=unrelated_clk
```

原因是没有 exact choice 时 `choices=[clocks[0]]`。物理 latch owner不能使用 fallback guess。

### 2.7 AXI negative address wrap

external-executor 隔离探针：

```text
write_words(((-1,1),)) -> -address FFFFFFFC
read_word(-1)          -> -address FFFFFFFC
```

UART codec会拒绝负 address；AXI把它乘4后 `& 0xFFFFFFFF`。两个 transport 对同一 RegisterTransport contract 不一致。

### 2.8 UART reply decoder 接受超 contract frame

`MAX_FRAME_WORDS=256`，手工构造合法 CRC 的 257-word reply：

```text
max=256 decoded_count=257 status=0
```

实时 `_extract_reply` 会在 decode 前拒绝，但 public `decode_reply` 与自己的 encoder/reply length contract 不对称，应在 codec owner拒绝。

## 3. P0/P1 问题账本

| ID | 优先级 | 事实 | 裁决 |
|---|---:|---|---|
| PULSE-PY-001 | P0 security/physical | wildcard bind + 无 auth/TLS/allow-list，任意可达client能drive board | `USER DECISION + BLOCK LAN DEPLOYMENT`：先明确可信网络边界；默认至少 localhost，远程用明确安全通道。 |
| PULSE-PY-002 | P0 ownership | stale handler dispatch不核当前owner | `FIX`：每个dispatch在同一owner临界区验证client，replacement后旧请求必须拒绝。 |
| PULSE-PY-003 | P0 safe takeover | SAFE失败仍授予new owner | `FIX`：未得到stable SAFE不得完成claim；应拒绝new client并保持明确fault。 |
| PULSE-PY-004 | P0 multi-process | Interprocess lease实际process-local且unused | `FIX OR DELETE`：若允许local hardware transports，必须用真实OS/device exclusive ownership；否则删虚假类/测试并只允许server owner。 |
| PULSE-PY-005 | P0 operation safety | notebook最后forever fire无finally SAFE；无idle release，文档声称有 | `REDESIGN NOTEBOOK`：hardware cell必须显式armed confirmation、try/finally、安全时限；默认finite。 |
| PULSE-PY-006 | P0/P1 lab integration | 默认auto probe所有COM，向无关仪器发送字节 | `USER DECISION`；推荐正式server要求显式UART port或approved allow-list，auto只作人工诊断。 |
| PULSE-PY-007 | P1 DoS | 8 MiB unauthenticated JSON、无限请求循环、ThreadingMixIn、server read无deadline/rate limit | `REDESIGN`：安全边界先定；限制active handlers、严格frame/tree depth/schema。quiet editor与半帧攻击不能用同一“永不超时”政策。 |
| PULSE-PY-008 | P1 shutdown | server close不drop active owner socket，daemon handler可继续阻塞；AXI reader不join、kill不wait | `REDESIGN`：close完成条件必须是safe、socket关闭、handler/reader/process有界归零。 |
| PULSE-PY-009 | P1 codec | pulse file tree非exact，file parsing不拒duplicate/nonfinite | `REDESIGN EXISTING CODEC`，不造第二serializer。 |
| PULSE-PY-010 | P1 ABI | XDC order/naming/clock fallback推导physical target | `REDESIGN / CROSS-REF 06f`：单一显式board manifest生成host/RTL/XDC，禁止clock guess。 |
| PULSE-PY-011 | P1 AXI | negative address wrap；first target/device/axis；timeout后hardware state uncertain | `REDESIGN`，先address bounds与identity，再定义timeout后recovery/safe gate。 |
| PULSE-PY-012 | P1 dead production | engine_model test-only且绝大多数零test consumer；transport专用helpers无caller | `MOVE/DELETE`。 |
| PULSE-PY-013 | P1 docs false safety | README/fpga README/batch help承诺5分钟idle SAFE与不存在参数 | `REWRITE NOW`，不能等待大架构重构。 |

## 4. `__init__.py` / public surface

| 符号/区域 | 裁决 | 依据 |
|---|---|---|
| `__version__` | `KEEP` | 顶层包身份事实；产品版本SSOT见06e。 |
| `_PACKAGE_DIR` name/package check | `DELETE` | 只证明目录恰叫`zlc_pulse`，不能证明当前checkout、wheel或ABI；zip/vendor等合法import形态反而可能被拒。 |
| model/compiler/binding/scan/codec exports | `KEEP` | Workbench与Atom有真实production consumers；04a逐项已审。 |
| `pulse_target_from_xdc/load_streamer_config` exports | `KEEP CURRENTLY + PACKAGE DECISION` | source-checkout正式路径真实使用；standalone wheel缺外部asset，见metadata。 |
| `MemoryRegisterTransport` top-level export | `KEEP WITH DEBT` | Atom virtual sequencer与standalone Pulse Editor virtual mode有真实consumer。 |
| `UartRegisterTransport/VivadoAxiRegisterTransport` top-level exports | `PRUNE TO SUBMODULE` | production server在remote内部从transport import；顶层外部consumer只有notebook/tests。本地backend是高级具体实现，不必污染普通pulse facade。 |
| `UartError/BackendResolutionError` top-level exports | `PRUNE` | Remote client收到的是`RemoteError`；这两个只对server/local backend有效。Notebook同时catch它们是假安全感。 |
| lazy `__getattr__` for remote | `KEEP` | 避免 `python -m zlc_pulse.remote` 重复import warning，并让普通model import不加载socket server。 |
| hand-maintained exact `__all__` test/doc | `PASS WITH DEBT` | 它能守意外增长，但当前test用历史说明为每次扩张辩护。应按真实production consumer收窄，不让notebook/test自己证明public必要。 |

## 5. `canonical.py` 逐函数裁决

| 函数/常量 | 裁决 | 说明 |
|---|---|---|
| `_normal` scalar/Enum/bytes/dataclass/ndarray | `KEEP + NARROW` | PulseTarget ABI与CompiledProgram digest真实依赖；支持已知plain/dataclass足够。任意`__dict__` fallback与generic object graph无真实consumer，应删。 |
| Mapping normalization | `REDESIGN` | 所有key先`str()`会collision；只收string keys并在转换前拒绝。 |
| ndarray normalization | `KEEP WITH VALIDATION` | dtype/shape/values进入digest合理；应明确只支持当前numeric dtypes并递归finite校验，不假装任意object/complex array都canonical。 |
| `canonical_bytes` | `KEEP` | stable compact JSON是现有ABI digest单一输入。 |
| `DIGEST_BITS=128` / `canonical_digest` | `KEEP` | 已部署ABI/content identity，不新增另一hash。与zlc_data位宽相同不表示两包digest可互换。 |

本报告不建议删除现有fingerprint；权威规则是“不新增第二套hash”，不是移除当前word63/target/program identity。

## 6. `codec.py` 逐函数裁决

| 函数/部分 | 裁决 | 说明 |
|---|---|---|
| `PULSE_TREE_FORMAT` | `KEEP` | `zlc.pulse.v1` 是唯一正式文档格式。 |
| `sequence_to_tree` | `KEEP` | 完整保存target、periods、slots、API parameters、delays、repeat；这是唯一serializer，未发现第二份science model。 |
| `sequence_from_tree` | `KEEP + STRICTEN` | 仍应把所有值交给model constructors，但首先对每层exact keys/list/mapping/scalar types做schema check；不要用`str/int`把JSON类型错误变合法值。 |
| top-level/target/port/period/field-ref/slot/API/delay/repeat readers | `REDESIGN AS ONE PATH` | 未知key当前忽略；optional字段必须明确列出而非`.get`掩盖typo。 |
| file bytes→tree | `MISSING OWNER` | 实际WorkBench/Atom各自普通`json.loads`，所以duplicate/NaN在进codec前已被接受。推荐在现有codec增加唯一strict JSON text/file入口，或复用一条已有strict reader；不得各caller再手写。 |

关键原则：strictness不是给artifact加防篡改/hash；它只保证操作者写下的键不会被静默丢掉。

## 7. `endpoint.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| `DEFAULT_PORT`、`DEFAULT_HOST`、connect/request timeouts | `KEEP` | server、client、device form共享一处默认值是正确SSOT。 |
| `DEFAULT_BIND_HOST="0.0.0.0"` | `USER DECISION / SECURITY GATE` | 对separated machine便利，但在无auth协议上等价于把物理board控制开放给全部可达接口。推荐默认localhost；显式`--host 0.0.0.0`必须伴随已验收的网络控制。 |

## 8. `manifest.py` 逐函数裁决

| 函数/规则 | 裁决 | 说明 |
|---|---|---|
| `_read_bindings` | `REDESIGN` | XDC parser当前只认单行PACKAGE_PIN/get_ports，duplicate signal静默skip，不验duplicate pin。它可做diagnostic parser，不能继续当physical ABI authority。 |
| `_is_system` | `REDESIGN/REMOVE HEURISTIC` | 任意未命中system name的XDC port会自动变pulse lane；debug/其他peripheral可误入。 |
| vector→DAC推断 | `REDESIGN` | 任意`name[index]`都当DAC，无显式port kind；这是假设当前board命名，不是通用manifest。 |
| clock matching | `FIX` | exact prefix/index无match时取第一个clock、多个match取第一个，均可错接latch。必须对0或>1 choice拒绝。 |
| declaration order→`chNN` | `REDESIGN / 06f` | 仅重排XDC就改变host lane，RTL手写mapping和word63不变。 |
| `read_xdc_pulse_lanes` | `KEEP DIAGNOSTIC` | tests/bring-up可读原始投影；不应被称完整ABI proof。 |
| `pulse_target_from_xdc` | `KEEP CAPABILITY, CHANGE SOURCE` | target projection有真实consumer；最终应读显式board manifest/生成物，并校验真实board ABI。 |
| lane/bus/width config checks | `KEEP BUT INSUFFICIENT` | 当前三项有价值；缺pin/order/clock/target ABI/clock_hz等见06f。 |
| `DEFAULT_XDC_PATH` | `SOURCE-CHECKOUT ONLY` | 指向`src`外package tree；installed wheel默认不存在。 |

## 9. `fpga.py`

裁决：`KEEP MODULE / MOVE IMPLEMENTATION FROM wire`。

当前71行只是从`wire.py`重导出build constants、capacity solver、emitters，并将CLI转发到private `_wire._main`。这与模块docstring“explicit home”矛盾，也让runtime wire owner继续承担build/resource/CLI。具体函数迁移表已在04a，非Python build/RTL风险在06f；本轮不重复。

`python -m zlc_pulse.fpga` 是根bootstrap调用的真实入口，应保留；但其默认assets是source-tree路径，wheel产品必须另行解决。

## 10. `engine_model.py` 逐簇裁决

### 10.1 consumer事实

production source对本模块 **零import**。当前外部tests只使用：

- `effective_tick`；
- `reference_play`；
- `streaming_scan_play`；
- `ScanUnderflow`。

`prefetch_play`、`rtl_mirror_play`、`rtl_mirror_play_stale_seed`、TTL/DAC delay mirror、bus replay/value evaluator、`PrefetchStall/DelayTooLargeError`等当前没有外部test consumer。注释所称“proven == reference/xsim”没有当前runner证据；06f确认普通suite不编译执行RTL。

### 10.2 逐符号簇

| 符号簇 | 裁决 | 说明 |
|---|---|---|
| `effective_tick` | `DELETE DELEGATE / TEST DIRECTLY` | 已只是`compile.evaluate_affine_tick`薄delegate；test若要独立oracle应在test写公式，调用同一production函数不构成独立证明。 |
| `EngineProgram.from_program` | `MOVE TO TESTS + STRICT` | 用大量getattr/default接受不完整对象，可能让错误CompiledProgram在mirror中“正常”。 |
| `reference_play` | `MOVE TO TESTS` | 少量compile/scan tests的独立逐tick oracle有价值，不属于installed runtime。 |
| `streaming_scan_play` / `ScanUnderflow` | `MOVE TO TESTS` | ping-pong/refill test有价值；不能冒充real RTL proof。 |
| `prefetch_play` / `PrefetchStall` | `DELETE OR RESTORE REAL TEST` | 当前零caller。 |
| `rtl_mirror_play` / stale-seed variant | `DELETE OR RESTORE REAL RTL PARITY TEST` | 当前零caller；名称“rtl mirror”不等于执行RTL。 |
| delay-line reference/mirror cluster | `MOVE MINIMAL REFERENCE / DELETE MIRRORS` | 06f已证明真实FIFO overflow validator缺失；注释“host validator prevents”不实。 |
| bus play/replay/value-at cluster | `DELETE OR TEST-OWNED` | 全部零caller，约数百行历史oracle；若未来接RTL cosim再从tests owner恢复最小必要函数。 |
| `_first_values` | `DELETE DUPLICATE` | 文件中连续定义两次，后者覆盖前者，是明确历史残余。 |
| import-time `load_streamer_config()` | `DELETE FROM PRODUCTION IMPORT` | test oracle import不应读取deployment manifest/fallback并把当前machine config变成module globals。 |

最终建议不是把1158行换一个production wrapper，而是：当前4个真实test需求移入tests，零consumer簇直接删除；未来RTL runner需要哪一小段再由test调用。

## 11. `remote.py` 逐类/函数簇裁决

### 11.1 framing/tree codec

| 符号 | 裁决 | 说明 |
|---|---|---|
| `MAX_FRAME_BYTES` | `KEEP + ADD RESOURCE LIMITS` | 8 MiB wire cap必要，但不能限制JSON nesting、dataclass expansion、thread/request rate。 |
| `REMOTE_METHODS` | `KEEP INTERNAL` | server/client单表正确；actual是12 methods（新增`describe`后），docs仍多处写11。 |
| `_TREE_TYPES`、`encode_tree/decode_tree` | `KEEP CAPABILITY + MAKE METHOD-SPECIFIC/STRICT` | client/server传CompiledProgram等需要；generic `__type__` mapping、stringified keys、unknown ordinary tree无exact schema扩大attack/input面。至少拒nonfinite、duplicate keys、nonstring collisions、excess depth，并让method决定期望类型。 |
| `_recv_exact/_send_frame/_recv_frame` | `REDESIGN SERVER DEADLINES` | client timeout合理；server端读header/payload无限等待。quiet editor与“宣称8 MiB后只发1 byte”的slow client应区分：idle无frame可静默，started frame必须有deadline。 |

### 11.2 logging/address helpers

`_log_fields/_server_log/_server_log_change/_forget_polls/_program_summary/_print_client_endpoints`：`KEEP WITH DOC FIX`。payload不全量dump、poll只在变化时记录、disconnect清poll cache均合理。非owner disconnect目前返回True并记录`outputs=SAFE`，其实该handler没有执行SAFE，日志应写`NOT_OWNER/NO_ACTION`。

### 11.3 backend resolver

| 符号 | 裁决 | 说明 |
|---|---|---|
| `BackendResolution/Error` | `KEEP SERVER-INTERNAL` | startup reason/attempts对operator有用；不是普通client top-level API。 |
| `_uart_candidates` | `REDESIGN POLICY` | port enumeration失败被降格并auto fallback；所有OS ports都进入probe。正式server推荐explicit port/approved identity。 |
| `_probe_uart_port` | `KEEP` | 复用PulseStreamer word63 open而非第二handshake正确；probe结束直接close transport避免第二hardware command。 |
| `_probe_failure_reason` | `KEEP WITH DEBT` | operator分类有用；用exception文字匹配会漂移，只用于说明不能成为安全决策。 |
| `resolve_backend` | `KEEP + CHANGE DEFAULT AUTHORITY` | explicit UART failure不fallback、explicit JTAG跳probe正确；auto探测所有仪器与任意failure转JTAG需用户裁决。 |
| `memory` backend | `KEEP DEV ONLY / LABEL HONESTLY` | 当前CLI会继续打印`HARDWARE CONNECTED`，虽CONFIG写memory mock。应明确`SIMULATION/NO HARDWARE`，避免远端client误认。 |

### 11.4 handler/server ownership

| 类/方法 | 裁决 | 说明 |
|---|---|---|
| `_RemoteHandler.handle` | `REDESIGN` | request schema不exact、无auth、started-frame无deadline；caught Exception回传内部type/message。核心P0是dispatch未绑定owner。 |
| `PulseRemoteServer.claim_client` | `FIX P0` | last-connect policy本身可裁决；无论选last-win或busy拒绝，必须在successful SAFE后原子转移，失败不能继续。 |
| `dispatch` | `FIX P0` | 必须验证current owner并与claim/replacement串行；现在stale client可执行全部12 methods。 |
| `client_disconnected` | `REDESIGN` | disconnect AUTO-SAFE正确；SAFE failure却release owner且调用方可忽略。失败应进入server fault/拒绝新命令。 |
| `owner_status/_release_client` | `KEEP` | 小状态足够；不新增session/epoch框架，只需让同一lock真正保护dispatch。 |
| `ThreadingMixIn daemon_threads/block_on_close=False` | `REDESIGN SHUTDOWN` | 方便进程退，但掩盖handler未归零；server close必须drop active socket并bounded join/confirm。 |

### 11.5 Remote client

| 方法 | 裁决 | 说明 |
|---|---|---|
| ctor/open/describe/load/table/fire/cursor/safe/snapshot/applied | `KEEP` | 与local device表面对齐有真实consumer。 |
| `wait_done` short polling | `KEEP` | 一次poll一请求、释放I/O lock让safe插入，设计正确；docs称每poll还调用snapshot/cursor，实际没有。 |
| `close` | `KEEP` | 显式device close再socket disconnect；server close应保证physical safe。 |
| `disconnect` / `__del__` | `PASS WITH DANGER` | 只drop socket，依赖server异步AUTO-SAFE；不能作为“safe已完成”证据。Control window关闭必须先显式safe/close并确认。 |
| `_call_locked` | `REDESIGN FAILURE STATE` | timeout/OSError会断开；invalid response shape/id/JSON未必断开。任何protocol desync都应废弃connection，并告知命令结果可能不确定。 |
| `RemoteError` | `KEEP` | client需要server failure类型；不要把未认证server全部内部路径/exception detail原样暴露。 |

### 11.6 server CLI/API

| 符号 | 裁决 |
|---|---|
| `serve/connect` | `KEEP`；分别是server API与client factory。 |
| `PulseRemoteServer` public export | `MAKE IMPLEMENTATION-PRIVATE`；仓外production consumer为零，tests可import owning private module。 |
| `build_arg_parser/_main` | `KEEP`；根launcher真实调用。补auth/bind/explicit UART policy后同步。 |
| `_main` initial open+stable SAFE before listen | `KEEP`；这是正确startup gate。config fallback/physical ABI不足见06f。 |
| `_main finally streamer.close` | `KEEP + REPORT FAILURE`；close失败不可被普通return遮住，必须非零并清晰说明physical state unknown。 |

## 12. transport 逐文件裁决

### 12.1 `transport/base.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| observer interval constants | `KEEP` | UART/memory 1 ms、JTAG 50 ms反映真实transport cost。 |
| `TransportAborted` | `KEEP` | Stop与deadline需要统一异常；Memory目前错误地抛普通RuntimeError，应对齐。 |
| `RegisterTransport` | `KEEP + NARROW` | `start/close/write_words/read_word`、transport id/cadence/lossy_line是真正device contract。 |
| `record_diagnostic` protocol method | `DELETE` | Device不调用；仅AXI自己内部使用。改AXI private helper，删Memory/UART空实现。 |

### 12.2 `transport/lease.py`

| 类 | 裁决 |
|---|---|
| `DeviceLease` | `DELETE`：普通class里两个ellipsis method，既非Protocol/ABC也无consumer，实例化后调用什么都不做。 |
| `InterprocessDeviceLease` | `DELETE CURRENT IMPLEMENTATION / USER DECISION REAL LOCK`：名称、测试和行为矛盾。若产品只允许remote server持真实transport，删它；若允许direct local UART/JTAG，使用现有文件实现真正OS lock并在transport start/close接入。 |

### 12.3 `transport/memory.py`

| 方法/状态 | 裁决 | 说明 |
|---|---|---|
| register/status/command-edge twin | `KEEP` | virtual/product rehearsal与device tests真实使用，能抓command strobe。 |
| `record_history` | `KEEP` | tests默认完整历史，三个长期product owner已显式False，避免无界增长。 |
| `start/close` | `KEEP + ENFORCE CONTRACT` | 目前read/write在未start或close后仍可直接执行；PulseStreamer挡住正常路径，但transport contract本身不一致。 |
| cancellation | `FIX` | 应抛`TransportAborted`而非generic RuntimeError。 |
| `record_diagnostic` | `DELETE DEAD`。 |

### 12.4 `transport/uart_frame.py`

| 函数 | 裁决 | 说明 |
|---|---|---|
| CRC/framing constants与`crc16_ccitt` | `KEEP` | 与RTL常量当前对齐，06f。 |
| `_unsigned/_span` | `KEEP` | address/count边界正确；caller先`int()`会让bool绕过bool拒绝，应停止预转换。 |
| encode write/read/reply | `KEEP` | 单一wire codec。 |
| `decode_reply` | `FIX` | 补count<=MAX_FRAME_WORDS；当前与encoder/_extract_reply不对称。 |
| `reply_frame_len/coalesce_runs` | `KEEP` | batching与4-byte words需要；coalesce只合相邻输入不重排，正确。 |

### 12.5 `transport/uart.py`

| 类/方法 | 裁决 | 说明 |
|---|---|---|
| `UartLink` | `KEEP` | 窄Protocol使PySerial与tests fake共用。 |
| `PySerialLink.open` | `KEEP + CLEANUP` | DTR/RTS关闭正确；若设置line或后续步骤失败应close刚开的handle。重复open也应拒绝/先close。 |
| `exchange/write_batch/_read_replies` | `KEEP` | reset buffer、deadline、stop、partial diagnostic清楚。 |
| `_extract_reply` | `KEEP` | damaged sync/count/CRC后逐byte重新hunt，真实line robustness。 |
| `UartRegisterTransport.start/close` | `KEEP + OWNER` | 重复start可覆盖serial handle；无cross-process lease。 |
| `write_words/_deliver/_classify` | `KEEP` | idempotent writes可retry、command strobe不blind retry、CRC-refused可retry的区分合理。 |
| `read_word/_read_with_retry` | `KEEP` | read幂等且SEQ/CRC分类合理。 |
| `rewrite_scan_bank` | `DELETE DEAD` | Device `_refill` 已一次`write_words`提交同一ordered rows；全仓零caller。 |
| `record_diagnostic` | `DELETE DEAD` | 全仓零caller。 |
| timeout budgeting | `KEEP WITH HARDWARE ACCEPTANCE` | 数学合理，但3 Mbaud/USB adapter真实可靠性仍需实验机，不由mock证明。 |

### 12.6 `transport/axi.py`

| 类/方法 | 裁决 | 说明 |
|---|---|---|
| `_default_vivado/_default_probes` | `KEEP + PIN DEPLOYMENT` | 自动发现便利；lexicographic newest tool不是qualified toolchain。06f已有build/toolchain gate。 |
| ctor/start | `KEEP + VALIDATE` | persistent Vivado owner是真实JTAG需求；start并发无lock、external executor只属test、state dir构造即写盘。 |
| `_init_tcl` | `REDESIGN IDENTITY` | 选择第一个target/device，`get_hw_axis`不要求唯一；多板/JTAG chain可控错硬件。06f同源。 |
| hw_server/probes Tcl interpolation | `REDESIGN ESCAPING` | env/path直接拼Tcl；至少用安全Tcl quoting并拒newline/brace escape。 |
| write/read address | `FIX` | 负/超范围当前wrap为32-bit；与UART contract不同。 |
| burst split | `KEEP` | insertion order、burst_max、4 KiB边界逻辑有测试。 |
| `read_words` | `DELETE DEAD` | 全仓零caller，所谓proof batch未被device使用。 |
| `rewrite_scan_bank` | `DELETE DEAD` | Device已把ordered bank rows交同一个write_words；全仓零caller。 |
| `_run_tcl/_execute/_read_until_marker` | `KEEP + DEFINE UNCERTAIN COMMAND` | deadline/stop有必要；Stop只停止等待，Tcl/hardware action可能继续。timeout杀Vivado后也无法用同transport证明SAFE。必须把这种结果标成physical state unknown并阻止普通继续。 |
| close | `REDESIGN` | 无I/O lock串行、reader thread不join、terminate/kill后最后不wait、fields不完整reset；不满足zero-worker/process。 |
| persistent log | `BOUND/ROTATE` | `vivado_axi_transport.log` append且逐行flush，长期server无界增长。 |
| `record_diagnostic` | `KEEP PRIVATE` | AXI内部failure/action evidence有用，不属于RegisterTransport。 |

### 12.7 `transport/__init__.py`

保留真实transport与Protocol/error exports；删除两个lease假面。具体UART/AXI transports可继续从`zlc_pulse.transport`显式import，不必同时占顶层`zlc_pulse` facade。

## 13. tests 逐文件裁决

| 测试文件 | 裁决 | 证据价值与缺口 |
|---|---|---|
| `test_remote.py` | `KEEP CORE + REWRITE OWNERSHIP TESTS` | local device parity、safe interrupt、logs、backend reason有价值；它明确锁死last-connect-wins、quiet forever永不释放、handler完全无deadline、probe所有OS ports，却没测stale dispatch、SAFE-failed takeover、auth、shutdown threads、slow partial frame。当前绿会保护危险政策。 |
| `test_transport.py` | `KEEP + DELETE FALSE LEASE TEST` | framing、AXI 4K split、modem lines、cadence、memory history有价值；lease test仅同process，给“Interprocess”假通过。 |
| `test_uart_transport.py` | `KEEP` | lost reply retry、strobe不双发、bad acknowledgement、CRC resync、retry latency均是高价值行为；补decode oversize、double-open/cleanup/cancel。 |
| `test_manifest.py` | `KEEP + ADD AMBIGUITY REJECTIONS` | default target、pin metadata、lane/bus/width drift有价值；缺duplicate signal/pin、多clock/no exact clock、multiline XDC、unrelated vector。 |
| `test_model_compile.py` | `KEEP SCIENCE, MOVE ORACLE` | 只有reference/streaming模型真实用到；把最小oracle移tests。不能用production `effective_tick` delegate自证compile函数。 |
| `test_public_surface.py` | `REDESIGN` | 防意外facade增长有用；当前以tests/notebook consumer为理由冻结UART/AXI/errors，不证明public产品必要。 |
| `test_notebook_coverage.py` | `REDESIGN P0` | offline prefix真实执行是好测试；hardware cells只用FakeRemote，且test明确要求最后cell forever fire、无safe afterwards。应改为安全finite/finally contract，并检查/清除stale saved outputs。 |
| `test_import_purity.py` | `KEEP + STRENGTHEN` | dependency allowlist/headless import有用；`test_package_import_is_pure`只看版本/目录名，不检查无filesystem/config import side effect。 |
| `test_launcher.py` | `KEEP / CROSS-REF 06f` | root bootstrap/shadow cwd/arg forwarding有用；help含不存在idle option未被拒。 |
| `test_fpga_assets.py` | `KEEP / 见06f` | source asset/header projection，不是wheel/RTL execution证明。 |
| `test_command_strobe.py` | `KEEP / 见04a,06f` | device command/safe/loader/UART strobe证据强；不覆盖remote auth/owner或physical RTL safe尾部。 |
| `test_wire_device.py` | `KEEP / 见04a,06f` | wire/device/memory twin；不是real transport/RTL proof。 |
| `test_contract.py` | `KEEP WITH DOC FIX` | signatures有用；contract仍漏`describe`/把12称11。 |
| `test_scan_model.py` | `KEEP / 见04a`。 |

本轮未跑package/full-tree suite；隔离探针已足以证明上述P0，不需要以大测试延迟落盘。

## 14. Notebook / README / contract / metadata / launchers

### 14.1 `notebooks/usage.ipynb`

裁决：`SPLIT OFFLINE TUTORIAL FROM HARDWARE ACCEPTANCE`。

已确认：

- offline前缀由test执行，model/compile示例有价值；
- saved output写`public package names: 22`，当前facade远大于22；
- saved real-server output仍出现旧`RemoteBusyError: server is busy`，当前政策是new client替换old；
- saved Stop cell报`RuntimeError remote PulseStreamer is not open`；
- 最终cell仍无条件连接localhost并forever fire真实全通道/全幅DAC；
- 最终cell之后没有SAFE/close cell；只能依赖用户另开cell或连接断开；
- 当前server明确没有idle timer，kernel保持连接就可无限firing。

这不是普通“saved output旧了”，而是操作安全说明与代码相反。推荐offline notebook只保留无hardware cells；真实scope/bring-up变成显式批准的有限命令或带确认、timeout、try/finally SAFE的独立操作步骤。

### 14.2 package `README.md`

必须修正：

- “same eleven methods”实际12（有`describe`）；
- 多次引用不存在的`fpga\run_server.bat`，正式入口在root `bin\run_server.bat`；
- 声称idle client五分钟AUTO-SAFE和`--client-idle-timeout`，代码/CLI没有，tests明确禁止idle timeout；
- 声称RTL/bitstream external且本仓不build/program，实际tracked RTL/Tcl和root recovery launcher，见06f；
- `0.0.0.0`+firewall提示不是authentication/security contract；
- remote ownership未说明任何LAN peer都可替换owner，也未说明SAFE-failure/stale-handler现状。

### 14.3 `docs/contract.md`

当前仍有这些实现冲突：

- device/remote称11 methods，实际含`describe`为12；
- PulseStreamer伪代码ctor未体现target等当前真实要求，详见04a；
- remote wait描述成snapshot/cursor/wait_done组合，client实际只短poll wait_done；
- connection ownership文字宣称单client生命周期足够，却没有owner check；
- XDC映射声称不同层不会silent drift，06f与manifest probe已反证。

它是历史冻结契约，不因名字“唯一权威”自动胜过当前代码/物理证据；需要在用户裁决security/ownership后重写。

### 14.4 `fpga/README.md`、`board_config/README.md`、历史docs

- fpga README同样声称5分钟idle release和不存在CLI参数；
- board_config README把XDC声明顺序/heuristic称为single runtime mapping且“cannot silently differ”，与word63 ABI hole矛盾；
- `acceptance-*`、`reacceptance-*`、`survey-*`、`goal-archive.md`均是2026-08-02/03历史证据，不是当前remote/transport验收；
- `GOAL.md`若为historical tombstone可保留，不用附加当前结论。

### 14.5 `pyproject.toml` / wheel

| 项 | 裁决 |
|---|---|
| numpy/pyserial dependencies | `KEEP`；pyserial虽lazy import，server UART是正式backend。 |
| no package scripts | `PASS FOR ROOT PRODUCT / USER DECISION STANDALONE`；当前正式入口经root bootstrap。 |
| `py.typed` | `KEEP + WHEEL TEST`；需实际build/install验证是否随包。 |
| `fpga/` assets outside `src` | `P0 IF INSTALLABLE PRODUCT`；manifest/wire/remote/fpga CLI默认路径在wheel不存在。见06e/06f/09。 |

### 14.6 root `bin/run_server.bat` / bootstrap

根bootstrap能确保当前checkout而非旧editable，这是正确的。剩余Python相关冲突：

- 默认host仍是0.0.0.0；
- help展示`--client-idle-timeout 300`，Python parser拒绝；
- 默认backend auto仍probe全部COM；
- memory backend被列作普通override；
- help路径仍写`fpga\run_server.bat`。

其余batch/build问题见06f，不重复。

## 15. 确认可删/移走的production surface

立即可裁为dead，无需为“未来可能”保留：

- `engine_model.py` 中除当前tests所需最小reference外的全部cluster；最小reference也移tests；
- duplicated `_first_values`；
- `transport.lease.DeviceLease`；
- 当前假的`InterprocessDeviceLease`（除非同一轮真正实现并接入）；
- UART/AXI `rewrite_scan_bank`；
- AXI `read_words`；
- RegisterTransport/UART/Memory的`record_diagnostic` seam；
- top-level facade的UART/AXI concrete classes与server-only error names；
- `PulseRemoteServer`、tree codec、backend resolution等作为public `remote.__all__`的test-driven承诺；实现可留module-private；
- `_PACKAGE_DIR`目录名guard。

必须保留：

- current target/program canonical digest；
- single PulseSequence tree codec；
- endpoint constants；
- client/local device同方法面与short wait poll；
- initial server open+SAFE gate；
- UART CRC/SEQ/retry distinction；
- persistent AXI owner（若JTAG backend继续正式支持）；
- Memory transport command-edge twin；
- XDC/board target capability，最终改为显式manifest owner。

## 16. 推荐修复顺序（本轮不改代码）

1. **立即关P0 ownership**：dispatch owner check/serialization；SAFE failure拒绝takeover；server shutdown drop sockets并归零handlers。
2. **确定remote security边界**：在此之前不要把wildcard server当已验收LAN设备。最小安全默认是localhost或受控tunnel；若直接LAN则需正式auth/integrity设计与部署证据。
3. **撤回危险操作文案/Notebook**：删除虚假idle承诺；hardware forever示例改finite/finally SAFE。
4. **决定UART discovery**：正式server explicit/allow-listed port；auto probe降为人工diagnostic。
5. **修real transport lifecycle**：AXI地址、target identity、uncertain timeout、reader/process join；UART double-open/cleanup；真实cross-process ownership。
6. **strict codec/manifest**：只加强现有single serializer/reader和board owner，不加第二格式/hash。
7. **删除test-only production**：engine model与dead transport helpers。
8. **最后收敛facade/docs/wheel**：按06e产品distribution裁决同步。

## 17. 需要用户裁决

### D6H-01：Remote部署的信任边界

推荐：server默认只bind localhost；跨机通过实验室受控tunnel/VPN或正式authenticated channel。若用户确认FPGA机所在LAN物理隔离且只受控主机可达，也必须把该假设写入deployment acceptance并由firewall/ACL实测，不可把README一句firewall提示当证据。

### D6H-02：第二client如何取得owner

推荐：authenticated client可显式takeover，但只有旧owner成功SAFE后才转移；旧handler全部请求失效。备选是busy拒绝直到旧client断开。当前“任何新TCP连接静默抢板且SAFE失败也继续”不是可保留选项。

### D6H-03：quiet editor与active forever的失联策略

推荐区分两种状态：无active command的quiet editor不因编辑沉默被踢；active forever必须有可验证connection/lease失联策略并在超时后SAFE。用户需给出合适timeout。当前docs说5分钟、tests说永不timeout，必须二选一并只保留一份truth。

### D6H-04：UART自动发现

推荐正式server要求明确`--uart-port`或apparatus config的approved port/identity；`auto`只在人工Scan/diagnostic中运行并显示将触碰的ports。若保留默认全probe，需要用户明确接受向其他COM仪器发送probe bytes的风险。

### D6H-05：local UART/JTAG是否允许绕过remote server

若不允许：删除Interprocess lease假面，正式产品只让server process持hardware transport。若允许：在现有transport owner实现真实OS/device lock，start前acquire、所有close/failure释放，并跨进程测试。当前两种路线各做一半不可接受。

### D6H-06：真实硬件Notebook的产品定位

推荐offline tutorial与硬件bring-up分离；bring-up要求显式危险确认、有限pulse默认、try/finally SAFE和独立物理验收记录。若用户坚持notebook最后forever fire，必须接受kernel/网络半断时可能长期输出，并明确停止机制；不能继续声称不存在的idle AUTO-SAFE。

### D6H-07：installable wheel还是source-checkout-only

沿用06e/09的顶层裁决。若installable，FPGA config/XDC/assets、py.typed、server/fpga entries必须由wheel实测；若source-only，删掉误导的standalone交付承诺并让root launcher成为唯一正式入口。

## 18. 最终判断

`zlc_pulse` 的 model/compiler/device 主链已有不少严肃的command/SAFE/UART测试，不能因本轮问题把整个包判作废。但真正硬件边界最危险的部分恰好位于此前未逐源审查的remote ownership与transport lifecycle：

- network owner不是device owner proof；
- AUTO-SAFE日志不是SAFE成功；
- 名叫Interprocess的dict不是跨进程锁；
- mock/同进程test不证明Vivado/serial process teardown；
- 一个最后cell的forever fire也不是教程验收。

因此在PULSE-PY-001～006关闭前，不能宣布Pulse硬件Python路径已完成或可安全LAN部署。正确方向是先收紧现有owner与直线流程，再删除test-only production和历史public seams；不需要新增session/epoch/registry框架，也不需要第二serializer或第二hash。
