# GOAL — zlc_pulse:极简 pulse 设备包

状态:**X0.5 → V/V1b → V2 → X → §API,21 项待办**(已完成条目已归档到 `docs/goal-archive.md`,不许回改;勾选状态现在是真的)
仓库:`C:\Users\eadri\Dropbox\WorkCode\Github\zlc_pulse`(所有工作只发生在这里;RTL/bitstream 冻结,一行不动)

> 用户裁决(宪章):pulse 设备只提供——① on/off 与基本点火;② 编辑用的 sequence 模型;③ `write_scan_table`(scan slots,无缝硬件扫);④ `write_slots`(api slots,免重编译改 period/DAC/delay);⑤ **applied 状态回读**(设备保存最后应用的原样记录供 GUI/编排 sync——被动回声,问了才答,用户 2026-08-03 裁决)。**设备不回传任何用于与外界同步的信息**(trigger 调度、应收帧数、逐点对账全部判死);设备不知道 measurement/GUI/run 的存在。目标 ~5k 行(现语料 ~20k)。
> ⚠️ 同名影子警示:迁移分支树内也有 `zlc_pulse` 包。本仓自首个 commit 起为唯一 owner;`__init__` 加 `__version__` + 安装路径断言;在 v1 树目录下起 Python 时注意 import 解析。
> 动工前必读随仓两份 survey(`docs/survey-pulse-fpga-2026-08-02.md` 链路图+逐处点名+最小 API 签名;`docs/survey-defenses-2026-08-02.md` 防御裁决表)。搬运源(只读):`..\Zou_lab_control_v1_claude\Zou_lab_control_v1\zlc_pulse` 与同树 `fpga/pulse_streamer/host`。

## 铁律 / 仪式 / 收尾

> 🔴 **永不建 venv**(用户 2026-08-05 明令):依赖一律全局 `pip install`;脚本里**不许**出现 .venv 探测/偏好分支。血训:zlc_pulse 的启动器曾优先用仓库 .venv,而那个 venv 没装 pyserial,导致串口枚举抛 ModuleNotFoundError、返回零候选口、UART 探测循环一次都不进,服务器**插不插线都无条件退回 JTAG**(实验机上=常驻 1-2 GB vivado.exe)。

> **🔴 铁律 0(2026-08-05 真机哑火之后加,优先于其他所有条目):跟硬件说话的那一层,只许迁移,不许重写。**
> 本仓的 host 在 `4bf8681` 是**从零重写**的,不是从 v1 `transport/session.py` 迁来的。结果:v1 每条命令前写的 `COMMAND=0` 归零丢了、fire 前的 bank 重新上膛丢了、LOAD 后的 `STATUS_LOADED` 握手丢了。RTL 的命令是**上升沿检测**且**硬件从不清 COMMAND**(`ldr_cmd_clear` 声明了却从未赋 1),于是连写两次同一命令 = 第二次静默丢弃 —— `write_slots`+`fire` 扫描循环**只有第一发出光**,而 host 每次都正常返回 DoneReport。真机上表现为"根本不输出",查了整整一轮才定位(修复:`232788f`)。
> **根因不是某个 bug,是"发明代替迁移"**:凡是 v1 已在真机上跑通的行为,一律以 v1 源码为底本逐条搬,**差异必须显式写理由**;靠读 RTL 重新推导等于把当年踩过的坑重踩一遍,而且是一次踩一个、修一个。
> **机械强制**:`tests/test_command_strobe.py` 把 RTL 的边沿规则本身钉住,并让 host 的写序列跑这条规则。改 host↔RTL 协议前先读它。

绝不 push;每主题小 commit;无后向兼容、干净删除;不加防御仪式;守卫自证非空洞;**不跑 Vivado build/program(xsim 可)**;参照树只读。每轮:读本文件全文→`pytest -q` 基线→选最靠前未勾项→确认判据再动手。收尾三式:完成勾选 commit / 受阻记录 / 全绿改 GOAL COMPLETE 只回终止陈述。

## §X0 开工前必读(2026-08-05)

> **勾选状态现在是真的**:已完成的条目(R7 全轮 + V/W 里已核实达成的)已移进 `docs/goal-archive.md`,本文件里**剩下的 21 项全部是真待办**。
> **后写的轮次覆盖先写的**(不许重复实现):**X1 是导出面的最终裁决,覆盖 W3.1 与 V3.2 / V3.3**——V3.2 的「构建期工具移出包级导出」已含在 X1 的名单里(名单上没有它们),V3.3 的「契约与导出面对齐」就是 X1 判据②;**V3.3 里「52 个导出」是过期数字,现在是 30**。V3.4(远端/本地方法面对等)是另一件事,仍要单独做。此外:**X2 覆盖 W4.1 与 V4.3——"每个导出名都要被使用"这条判据已作废**,正是它造成了 notebook 里的凑数断言与内部 API 露脸;X4 覆盖 W4.0 与 V4.1/V4.2 的真机段;X3 覆盖 V4.5;W4.2 与 V2.1 是同一件事,做一次。
> **优先级**:X0.5(host↔RTL 协议对账,铁律 0)> V1/V1b(启动链路与真机副作用)> V2(守卫补真牙)> X2-X5(notebook 教程化)> X1/V3/W3(导出面与契约)> 其余。

## V 轮:U 轮验收修复 + 公开面收敛(2026-08-05 三方审查 + 对抗复核;先读本节全文)

> **U 轮结论**:决议逻辑本身是真的(默认 auto 运行时可断言、`resolve_backend` 单源、指纹比对复用 `PulseStreamer.open()` 全仓唯一处、显式 uart 失败 raise+exit 2 不回退、八用例矩阵存在且三次突变可红)。**但启动路径实测是坏的**,而且"默认走 UART"在真机上会静默失效。以下每项都有实证,不是推测。
> **本节所有判断已机械化**:凡需取舍处都给了确定规则;本节未覆盖的取舍 → 记阻塞问用户,不要自行发挥。

### V1 启动链路修复(最高优先:**现在每次正常启动都起不来**)

- [ ] V1.5 **`--check-config` 分支会解析到别的 zlc_pulse(影子 import)**:主调用路径有 `pushd "%REPO_ROOT%"`,而 `--check-config` 分支没有,于是 `sys.path[0]` 是调用者的当前目录并**压过** bat 刚设好的 PYTHONPATH。对抗复核实测:在 `Zou_lab_control_v1` 树下执行 `fpga
un_server.bat --check-config`,`zlc_pulse` 解析到该树里 vendored 的那一份而不是本仓。本项目被同名影子包咬过不止一次,必须封死。
  判据:两条分支的工作目录/路径设置一致(或都显式用绝对路径调用);补测试:在另一个含同名包的目录下执行该分支,断言解析到本仓的 `__file__`。

### V1b 真机可用性与副作用(对抗复核发现,三份报告都漏了;**优先级等同 V1**)

> **可达性是有意设计,不动**:用户明确要让实验室所有人都能用,所以绑 `0.0.0.0`、播报 LAN 地址全部保留,不加鉴权、不加独占。本组解决的是**多人共用时的并发受理**与**真机副作用**。

- [ ] V1b.1 **一次一个客户端不变,但占用要说话、忘关要能自动释放**(用户 2026-08-05 裁决:**不需要支持同时连接**——`0.0.0.0` 可达性与"实验室所有人都能用"是有意设计,一律不动;本项只修"静默挂死"与"永久锁死"两个行为)
  实测现状:`PulseRemoteServer` 是非线程 `TCPServer` + `request_queue_size=1`,`_RemoteHandler.handle()` 用 `while True` 持有连接直到对端断开;服务端**没有任何 socket 超时或 keepalive**(`remote.py:762-763` 的 `settimeout` 是客户端侧 `request_timeout=30`)。两个后果:
  - 第二个客户端的握手被 backlog 接受,但 `serve_forever` 永远轮不到服务它 → **静默挂着**,既不知设备被占,也不知被谁占。
  - 持有者若是没关的 notebook kernel 或休眠的笔记本,**设备被无限期锁住**,只能重启服务器。
  修法(**保持"一次一个客户端"的语义**,不引入并发受理、不加鉴权、不加接管):
  ① **占用即回话**:第二个连接必须**立刻**收到一条明确回复——当前持有者地址 + 已持有时长——然后由服务端主动断开该连接。客户端把这条回复原样呈现给用户(不是裸超时)。实现上服务端需要能受理并回绝第二个连接(可用一个极薄的受理层专门回绝),**设备本身仍严格一次一个**。
  ② **服务端空闲超时**:持有者在 N 分钟内无任何请求则断开释放(N 作为具名常量,写进 README 与 `--help`);**释放前若设备处于发射状态,先驱动 SAFE**,并在日志记明"因空闲超时释放,已 SAFE"。
  ③ 保留并确保现有 `client=addr` 归属日志覆盖每个设备方法(用户点名"log 里显示谁在控制、控制了什么")。
  判据(memory 后端,不碰真机):甲持有时乙连接必须在 1 秒内收到含"持有者地址+持有时长"的占用回复而非超时;空闲超时到期后设备被释放,且释放前调用过 SAFE(用 spy 断言调用顺序);超时值可配置且默认值在 README 与 `--help` 中一致;每条设备操作日志含发起方地址。

- [ ] V1b.2 **auto 探针会向每个串口写数据**:`transport/uart.py:37` 用 `serial.Serial(port, 3_000_000, timeout=0.05, write_timeout=1.0)` 打开(Windows 下**默认拉高 DTR/RTS**,Arduino 类板卡会因此复位),随后 `exchange()` 执行 `reset_input_buffer()` + `write(frame)`。**实验机上挂着别的 COM 仪器时,每次启动服务器都会去戳它们一遍。**修法:① 打开时显式 `dsrdtr=False`、并在写入前把 DTR/RTS 置低;② 探针只在**未指定 `--uart-port`** 时枚举,且横幅打出"即将探测的端口清单"让运维可预见;③ 文档明写:已知有其他 COM 仪器时应显式传 `--uart-port` 跳过枚举。
  判据:探针打开串口的参数有测试钉死(不拉 DTR/RTS);横幅含待探测端口清单。
- [ ] V1b.3 **forever 运行时 observer 每 ~1ms 轮询,JTAG 回退下会把 Vivado 打满**:`device.py:365-388` 每轮两次寄存器读 + `time.sleep(0.001)`,直到停止。UART 3Mbaud 下很便宜,但 **JTAG-AXI 下每轮是两次 Vivado Tcl 事务**——notebook 的 `fire(forever=True)` 会让常驻 Vivado 进程持续高负载(正是内存吃紧的实验机上最不该发生的)。修法:轮询间隔按传输后端取值(UART 保持 1ms;JTAG 退避到 ≥50ms),间隔作为具名常量由 transport 声明。
  判据:两种后端的轮询间隔有测试钉死;间隔不是散落字面量。

### V2 守卫补真牙(现有守卫有两处是空的)

- [ ] V2.1 **`tests/test_notebook_coverage.py:24` 的切片 bug**:按字面量 `'DoneReport'` 切契约的 PulseStreamer 段,切点落在 `wait_done` 的返回类型注解上,导致 `cursor()`/`safe()`/`snapshot()`/`applied()` **四个方法从未被纳入强制集**。改为按方法名精确提取(或直接用导出面反射),使十一方法全部被钉。
  判据:临时从 notebook 删掉 `.applied()` 的调用,该守卫必红(现在是绿的);记 commit message。
- [ ] V2.2 **`test_uart_probe_reuses_pulse_streamer_word63_open` 是空洞守卫**:把探针里的 `streamer.open()` 整个删掉它仍然绿(它的 FakeTransport 只在 `read_word` 内部断言,open 不调用则断言永不执行)。改为断言"探针确实调用了 `PulseStreamer.open`"(可用 spy/计数),使删掉 open 必红。
  判据:突变实验(删 `streamer.open()`)必红,记 commit message。
- [ ] V2.4 **探针失败分类从未被真实异常执行**:八用例里六个注入假 `probe=`,`_probe_failure_reason`(`remote.py:332-351`)对**真实**异常类型的归类从未跑过(CRC 用例喂的是合成 `RuntimeError`,真实链路抛 `uart_frame.FrameError`)。补用例:用真实的 `uart_frame` 异常与真实的握手不符异常驱动分类,断言五类原因(无口/打不开/超时/CRC 错/指纹不符)各自归类正确。
- [ ] V2.5 **冻结资产守卫内部自相矛盾(死锁),必须改判据而不是改文件**:`test_frozen_fpga_asset_manifest_is_byte_exact` 比**原始字节**,它钉的 `zlc_geometry.vh` 头注释写着"由 `python -m fpga.pulse_streamer.host.image` 生成",而**同一个测试文件**又断言 `not (ROOT/'fpga/pulse_streamer/host').exists()`——即:用当前真实命令(`python -m zlc_pulse.wire --emit-geometry-vh`)重新生成必然改掉那行注释、必然破坏哈希,而按哈希把注释改回去又会引用一个测试断言不存在的路径。**这不是"用户误改、git checkout 即可"**(用户当前工作树的红确实来自真机构建,但根因是这个死锁)。
  修法:守卫改比 **宏值集合 + LAYOUT_FINGERPRINT**,注释头不参与;RTL 源文件(`.v`)继续比字节;并把 `.vh` 头注释更新为真实生成命令。
  判据:用真实命令重新生成 `.vh` 后守卫仍绿;人为改任一宏值或指纹必红(突变实验记 commit message);`.vh` 头注释里的命令可直接照抄执行。

- [ ] V2.6 **真实 CRC 故障的原因分类可能显示错**:设备侧返回 `ST_CRC_FAIL` 时 `transport/uart.py:155-156` 抛的是 `UartError('UART read reply was invalid')`,而 `_probe_failure_reason` 的 CRC 分支匹配的是别的形状,真实 CRC 故障会落进兜底类别 → 逐口原因显示错误的故障类型,把运维引向错误方向。判据:用真实的 `uart_frame`/`UartError` 异常驱动分类,断言五类原因各自归类正确(与 V2.4 合并验收)。

### V3 公开面收敛(宪章合规,但臃肿失真)

> **先说结论:宪章没有被违反。**八个禁词 grep 全零并有 `test_negative_surface_is_absent` 机械钉死;`trigger_times` 是 `schedule.py` 里的纯函数,被 device/wire/remote/compile 零引用(是调用者对自己编译的 program 求边沿,符合"scan 编排自己数点");`applied()` 全文三行、锁内返回已存快照,是真被动回声;设备层没有 fire_point、没有应收帧数、没有 arm-bind-finish。问题只在"臃肿与失真"。

- [ ] V3.2 **构建期工具移出包级导出**(**注意:先逐个确认是否真是构建期**,对抗复核已抓到一处误判——`check_rtl_assumptions` 被 `wire.py:548` 在**运行时路径**调用,**不是**构建期工具,不许移走)。确属构建期的:`emit_geometry_vh`(`wire.py:1251`)、`emit_geom_tcl`(`:1285`)、以及容量估算那一段;协议常量 `IMAGE_MAGIC`/`REGISTER_LAYOUT_ID`/9 个 `CMD_*`/`STATUS_*`/`CtrlWords` 属"调协议"而非"用设备"。移到 `zlc_pulse.fpga` 子模块(构建脚本从那里 import),包级 `__all__` 只保留运行时设备 API。**每移一个名字,先 grep 它在 src 内的调用点确认没有运行时路径依赖**,有则留下。
  判据:`zlc_pulse.__all__` 收敛到设备+模型+编译+传输;`fpga/*.bat` 与相关脚本改从新位置 import 且仍可运行(测试覆盖 import 路径)。
- [ ] V3.3 **契约与导出面对齐(现在双向漂移)**:`docs/contract.md` 自称唯一权威,但 52 个导出里 **35 个从未在契约中出现**;反向:`RegisterTransport` 写在契约设备层里、也是 `PulseStreamer.__init__` 的形参类型,却**不在** `__all__`。V3.1/V3.2 收敛后重写契约使之与最终导出面**一一对应**,并补一个**导出清单契约测试**(`__all__` 必须等于契约里列举的名字集合),让"零漂移"这条终态判据从此可机械验证。
- [ ] V3.4 **远端/本地方法面不对等(范围已收窄)**:对抗复核纠正——`contract.md:46` 只说远端"共享上面十一方法的**调用签名**",并未宣称独占;`tests/test_remote.py:434` 也已把转发的 RPC 面钉死为恰好这十一个。**真正措辞过头的只有 `README.md:40`**。裁决:承认差异而非抹平——十一方法设备面共享,连接生命周期(`disconnect` + 上下文管理器)是远端独有;**只改 README 使其与契约一致**,并补一条测试断言远端额外公开面恰为 `{disconnect, __enter__, __exit__}`。
### V4 notebook:从"能跑"到"真台架"

- [ ] V4.4 **不再依赖非导出 API 讲故事**:教程当前用了 `reference_play`、`MemoryRegisterTransport` 等不该出现在教程里的名字(**`load_streamer_config` 已在 X1 保留名单上,教程用它是对的;`MemoryRegisterTransport` 是测试替身,X1 已裁定撤下门面**)(其中 `load_streamer_config` 对硬件段是承重的)。要么导出并进契约,要么改用导出面表达。
### V 轮机械终态判据
1. `pytest -q` **全绿**(含 V2.5 修好的冻结资产守卫);V1.1+V1.2 的四种启动调用测试全在且**真正执行 bat**。
1b. `fpga
un_server.bat` 无参数运行能真正把服务器起起来(这是本轮的头号验收:现在 exit 255)。
2. V2.1/V2.2 的两处空洞守卫,突变实验必红并记 commit message。
3. `zlc_pulse.__all__` 内零别名、零 builtins 同名;导出清单契约测试绿。
4. notebook 在**无板卡**机器上执行到硬件段之前零错误,硬件段恰好一条含 `run_server.bat` 与端口的指引报错;带执行输出提交;覆盖测试绿;最后一个可执行格是停止格。
5. 行数守卫存在且绿(engine_model 显式除外并注明理由)。
6. `docs/contract.md` 与最终导出面一一对应;远端额外方法面已成文。
7. V1b 各项(并发受理+归属日志/串口探针不拉 DTR-RTS/JTAG 轮询退避)各有测试钉死。

## W 轮:板级映射回归 XDC 单源 + 导出面收敛(2026-08-05 用户实测 notebook 后提出)

> **根因(责任在 GOAL,不在实现)**:`fpga/board_config/board.xdc` 里逐行写着全部信号名与引脚(`cooling`/`repump`/`emCCD`/`microwave`… 以及 `da_dipole[0..9]` 这样的 DAC 位向量、`da_clk*` 锁存时钟)。v1 有 `zlc_pulse/manifest.py::read_xdc_pulse_lanes` + `pulse_target_manifest_from_xdc`,**从 XDC 派生 lane/signal/pin 并校验**;拆包时这套**一行未迁**(独立仓 grep `xdc` 零命中,无 `manifest.py`),而 `load_streamer_config()` 只读几何参数、不含任何端口名。于是 notebook 想连真机就**只能把 board.xdc 手抄成 Python 元组**(`hardware_ttl_names`、`hardware_dac_specs`、`hardware_lanes`)。这不是实现乱改,是能力缺失下的唯一出路;此前的 survey 与各轮 GOAL 都没发现,是审查失职。
> **用户裁决(2026-08-05)**:**XDC 是板级映射的唯一真相源**——不要 canonical JSON target 再加 XDC 校验那套(v1 做法),直接**从 XDC 全量派生**。换板只改 XDC 一处。
> **本节所有判断已机械化**;未覆盖的取舍记阻塞问用户。

### W1 板级映射单源(最高优先)

- [ ] W1.2 **与 `streamer_config.json` 的一致性硬校验**:派生出的 pulse lane 数必须等于 `channel_count`,DAC 总线数/位宽必须等于 `bus_count`/`bus_width`;不符 = **加载即硬报错**并指出两个文件里冲突的具体数值。(这是几何指纹握手同一条纪律在配置层的延伸:两份配置描述同一块板,不许静默分叉。)
  判据:构造一个 lane 数不符的临时 XDC,断言加载抛出含两个数值的清晰错误。
### W2 notebook 真机段改为派生

- [ ] W2.2 W1.1 的派生入口必须在**包级导出面**且写进契约——它是运维/编排的一等 API(notebook 与服务器都要用),不许再出现"承重却非导出"的情况(V4.4 抓过同类问题)。

### W3 导出面收敛(**V3.2 只做了一半,我上轮验收漏检**)

### W4 覆盖判据修正(**我上一条把分母算错了,在此更正**)

### W 轮机械终态判据
1. `pytest -q` 全绿;notebook 在无板卡机器上执行到硬件段之前零错误,硬件段恰好一条含 `run_server.bat` 与端口的指引报错。
2. W1.3 的硬编码信号名扫描零命中;W1.2 的不一致硬报错测试绿。
3. `zlc_pulse.__all__` 与 `docs/contract.md` 名字集合双向相等(W3.2 测试绿)。
4. W4.1 覆盖测试绿且突变自证必红,结果记 commit message。
4b. W4.0 的真机方法扫描绿(11 个设备方法在真机段各至少一次);回读字段打印齐全;forever 之后必有 safe。
5. 换一份假 XDC 能派生出不同的 target(证明零硬编码),该测试绿。

## X 轮:notebook 从"能跑的测试"改回"能学的教程" + 导出面收干净(2026-08-05 用户实测报告)

> **责任在 GOAL(W4.1)**:我写了"**每个导出名都要在 notebook 里被真实使用**"这条机械判据,于是出现了纯粹为满足它而存在的凑数代码——
> ```python
> package_version = __version__
> alternate_transports = (MemoryRegisterTransport, UartRegisterTransport, VivadoAxiRegisterTransport)
> transport_protocol = RegisterTransport
> assert issubclass(UartError, RuntimeError) and issubclass(TransportAborted, RuntimeError)
> assert transport_protocol and len(alternate_transports) == 3
> ```
> 这教不了任何人任何东西。**把代理指标当成目标了**(Goodhart)。判据本身作废,改成下面 X2 的形态。
> **用户裁决**:notebook 是**教程**——按功能分 cell、每格教一件事、用 `print` 展示结果让人看懂;**断言属于 `tests/`,不属于教程**。
> **W1/W2 已成功不要回改**:`pulse_target_from_xdc()` 已落地,真机段确实从 XDC 派生(手抄板级映射的问题已解决)。

- [ ] X0.5 **host↔RTL 协议逐条对账(铁律 0 的一次性清账,排在所有 X 项之前)**
  已修的三条(232788f)不代表清完。以 v1 `transport/session.py` 为底本,把它对硬件的**每一次写和每一次等**列成表,与本仓 `device.py` 并排,逐条给出"相同 / 差异+理由"。已知仍有差异、本轮需裁决的:① v1 的 `safe_state` 要求 **STATUS 连续两次读到 0** 才算确认,中途还会**重发一次 SAFE**,超时则 `TimeoutError`;本仓 `safe()` 只读两次并把结论塞进 `SafeReadback.stable`,**失败不报错**。② v1 `prepare` 在写镜像前会 `_drive_physical_safe`(先把板子驱到安全态),本仓 `load()` 不驱。③ v1 有 `check_register_layout` / `transport_self_test`,本仓只在 `open()` 比指纹。
  判据:对账表进 `docs/contract.md`;每条差异要么消除,要么写明"为什么本仓不需要";凡涉及写序列的,补进 `test_command_strobe.py` 的规则回放。

- [ ] X1 **导出面收到"用户真正要用的"为止 —— 名单已裁完,照做即可**
  > 我在 W3 写过"`CompiledProgram` 与 `RegisterTransport` 保留"——**收回,那是错的**,用户从不构造它们。
  > 判定只问一句:**用户会亲手构造它、亲手调用它、或亲手 `except` 它吗?** 否则一律进子模块。

  **保留(22)** —— 设备与连接:`PulseStreamer`、`RemotePulseStreamer`、`connect`、`serve`;用户手写的序列模型:`PulseSequence`、`PulsePeriod`、`AnalogStep`、`PulsePortSpec`、`PulseTarget`、`PulseSlot`、**`PulseFieldRef`(新增:`PulseSlot` 的构造参数就是它,不导出等于逼用户翻 `zlc_pulse.model`)**、`OutputDelay`、`RepeatRegion`;编译与板级事实:`compile_sequence`、`pulse_target_from_xdc`、`load_streamer_config`;两个真实传输:`UartRegisterTransport`、`VivadoAxiRegisterTransport`;用户真会捕获的错误:`RemoteError`、`UartError`、`BackendResolutionError`;`__version__`。

  **撤走(9)——只是从顶层门面拿下来,实现原样留在子模块,一行不删** —— 逐条理由与实测消费者:
  | 名字 | 为什么撤 | 包外消费者(实测) |
  |---|---|---|
  | `CompiledProgram` | `compile_sequence` 的返回、`load()` 的入参,用户不构造 | 零 |
  | `RegisterTransport` | Protocol 基类,用户传具体传输 | 零 |
  | `MemoryRegisterTransport` | **测试替身,绝不该是公开 API** | 零 |
  | `PulseRemoteServer` | `serve()` 才是入口 | 零 |
  | `StreamerParams` | 几何由 `load_streamer_config()["params"]` 给出(**用值不用名字**) | `zlc_atom/tests/test_contract_fakes.py:7` —— 本轮一并改掉 |
  | `AppliedState` / `DoneReport` / `SafeReadback` | 方法的返回类型,用户读属性不 import 类型 | 零 |
  | `trigger_times` | 分析辅助,非设备核心 | 零 |

  > ⚠️ 这 9 个名字在 `notebooks/usage.ipynb` 里各出现 2-11 次——那是我作废掉的"每个导出名都要被使用"判据造成的**凑数**,不是需求证据;X2 重写 notebook 时它们自然消失。

  判据:① 包级 `__all__` **== 上面 22 个名字**,并写成机械上限测试 **`MAX_PUBLIC_NAMES = 24`**(留 2 个余量,再加名字必须先改这个数字——这是让每次膨胀变成显式决定的唯一手段);② `__all__` 与 `docs/contract.md` 名字集合**双向相等**;③ 每撤一个先 grep 调用点、把 import 改成子模块路径(实现不动),类型注解用字符串或 `TYPE_CHECKING`;④ `MemoryRegisterTransport` 不再从顶层解析得到(它是测试替身,应只在 `zlc_pulse.transport` 里,`from zlc_pulse.transport import MemoryRegisterTransport` 仍可用);⑤ 每个保留的名字都在 X2 的教程里有**真实教学用途**(仅 import 不算)。

- [ ] X2 **教程判据取代覆盖判据(W4.1 作废)**:不再要求"每个导出名被使用",改为——
  ① **每个 cell 教一件事**,cell 内代码 **≤ 25 行**(现最长 67 行);
  ② **每个 cell 前有 markdown 说明"这一格教什么、为什么这样用"**;
  ③ **教程里 `assert` 数量为 0**(现 13 条);要断言就写进 `tests/`;
  ④ 每个 cell **用 `print` 展示结果**(现整份 notebook `print` 次数为 **0**),让人看得到 API 返回了什么;
  ⑤ **保留的每个公开名都要有一格真正的教学**(而不是被 import 一下)。
  判据:机械检查 ①③④(行数上限、assert 为 0、每个 code cell 至少一次 print);⑤ 由人工评审 + 覆盖检查共同保证。
- [ ] X3 **按功能补齐缺失的教学(用户点名)**:现在完全没有教 `write_slots` 与 `write_scan_table`,而这是这个包最有价值的两件事。至少要有:
  - **`write_slots` 一格**:讲清"什么是 api slot、为什么它能免重编译改 period/DAC/delay",演示改一个值、回读 `applied()` 看变化、再 fire;
  - **`write_scan_table` 一格**:讲清"scan slot 与 api slot 是同一机制的一行表 vs 多行表",构造一张小扫描表、fire、用 `cursor()` 看扫描点推进;
  - **`compile_sequence` → `CompiledProgram` 一格**:讲清编译产物里有什么(边沿表/掩码/总线段),用 print 展示;
  - **`wait_done` → `DoneReport` 一格**:把六个字段逐个打印并解释;
  - **`safe()` → `SafeReadback` 一格**:打印 status 与 clock_enable_words 并解释;
  - **`applied()` 一格**:讲清它是被动回声(问了才答),演示 GUI/编排如何靠它 sync。
- [ ] X4 **真机段拆成按功能的多格(W4.0 仍未做到)**:现在真机仍是**一个 67 行 8 断言的巨型 cell**,且只碰了 open/load/fire/snapshot 四个方法。按 W4.0 已定的安全顺序拆成多格,每格一个方法、每格 print 真实回读:open(打印指纹握手)→ load → applied 回读确认 → 有限次 fire + wait_done(打印 DoneReport 六字段)→ write_slots 改值再 fire(用回读证明免重编译)→ write_scan_table + cursor(看扫描点推进)→ fire(forever) + snapshot → **safe(打印 SafeReadback)→ close**。最后一格必须是停止格。
- [ ] X5 **把断言搬进 tests/**:X2③ 删掉的 13 条断言里,凡是有真实守卫价值的(如编译产物的 ticks/masks/bus_segments 形状)**移入 `tests/`** 成为真正的测试,不要直接丢弃。

### X 轮机械终态判据
1. `pytest -q` 全绿;notebook 在无板卡机器上执行到硬件段之前零错误。
2. notebook 中 `assert` 计数为 **0**;每个 code cell ≤ 25 行且至少一次 `print`;每个 code cell 前有 markdown。
3. `write_slots` / `write_scan_table` / `DoneReport` / `SafeReadback` / `applied` / `CompiledProgram` 各有专门一格教学。
4. 真机段按方法拆格,最后一格是 `safe()+close()`。
5. `__all__` 里每个名字都在教程中有真实教学用途;X1 撤走的名字已进子模块且调用方改好。

## 阻塞记录 / 开放问题



- U 轮实现与回归已完成；`pytest -q --ignore=tests\test_fpga_assets.py` 为 53 passed。完整 `pytest -q` 仍只剩本轮开始前就存在的 `fpga/pulse_streamer/zlc_geometry.vh` 冻结 SHA 不一致（当前用户工作区的 6 行生成注释改动），该用户改动按要求保留，未纳入本轮提交。
- (受阻时追加)
- 远程部署形态(瘦 remote 门面)后置,做时按 contract.md 逐一转发。
- 非缺陷备忘:350930c 清理 vh 头注释后,本仓再生 `.vh` 与树内已提交版仅有 3 行注释差异(宏值全等、指纹仍为 0x5AFC7CFB);将来用本仓再生树内 `.vh` 会产生注释 churn。
- 非缺陷备忘:engine_model 的 mirror 家族在本仓零调用(上轮债项 7),留档待将来裁决。
- Q9.3 环境阻塞(2026-08-04):当前机无 `xsim`/Vivado,未执行仿真或 build/program;待具备 xsim 的 FPGA 机按 `fpga/pulse_streamer/sim/README.md` 跑指定 bench 并记录。
