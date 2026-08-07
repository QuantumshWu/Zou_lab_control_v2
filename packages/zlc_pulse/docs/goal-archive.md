# GOAL 归档 — zlc_pulse 已完成条目

> 已完成并核实的条目原文(R7 全轮 + V/W 两轮里已勾的项),留作证据与追溯。**活的计划在 `GOAL.md`**。

## R7 复验收尾(2026-08-03 复验产出;勾完即改回 COMPLETE)

- [x] R7.1 wire.py 三处 stale 注释(均一行级):`:156` 与 `:183` 引用不存在的 `test_streamer_params_defaults_match_config`(实际锚是 `test_default_geometry_is_pinned_to_deployed_word63`,改名或删引);`:541` docstring "Pack a RuntimeSequenceProgram" 改 `CompiledProgram`;`:1147` 注释仍指旧仓路径 `Zou_lab_control._streamer_geometry`。
- [x] R7.2 阻塞记录补两行备忘(非缺陷):① 350930c 清理 vh 头注释后,本仓再生的 .vh 与树内已提交版有 3 行注释级差异(宏值全等、指纹不变 0x5AFC7CFB),将来用本仓再生树内 .vh 会有注释 churn;② engine_model 的 mirror 家族在本仓零调用(上轮债项 7),留档待将来裁决。

## V / W 两轮中已完成的条目

- [x] V1.1 **`run_server.bat` 第 51 行未转义的右括号提前闭合 `if` 块 —— 这是 BLOCKER,Python 根本没被调用**。`echo Other-computer client endpoint(s): printed by Python as CLIENT ENDPOINT` 位于 `if /I "%ZLC_PS_HOST%"=="0.0.0.0" (` 块内,`(s)` 的 `)` 被 cmd 当作块结束,余下 `: printed by...` 成为语法错误 → `': was unexpected at this time.'`,**exit 255**。而 `ZLC_PS_HOST` 默认就是 `0.0.0.0`,所以**每次正常启动必然命中**。(注:同一段第 49-50 行已正确写成 `^(` / `^|`,唯独这行漏了转义。)
  修法:转义为 `endpoint^(s^)` 或改写措辞去掉括号。
- [x] V1.2 **修好 V1.1 后紧接着的第二个坑:无参数时 `--inner` 会被透传给 Python**。第 5 行 `set "ZLC_FORWARD_ARGS=%*"` 在无参时**取消定义**该变量,第 18 行 `if not defined` 兜底命中,而 **cmd 的 `shift /1` 不影响 `%*`**,内层 `%*` 仍含 `--inner` → argparse `unrecognized arguments` 退出 2。
  判据(V1.1+V1.2 合并验收):新增**真正执行 bat** 的启动链路测试(用一个假 python 打印收到的 argv;现有测试只做源码 grep,永远抓不到这两类),覆盖四种调用——无参数 / `--backend jtag-axi` / `--backend uart --uart-port COMx` / `--help`——每种都必须把正确且仅正确的参数交给 Python,`--inner` 绝不出现。
- [x] V1.3 **解释器分裂 + pyserial 依赖形态**(对抗复核纠正:.venv 缺 pyserial 但**启动器根本不用 .venv**)。`fpga/_resolve_tools.bat` 用 `where python` 解析到系统 Python(该解释器**有** pyserial 3.5),而全部测试跑在 `.venv`(**没有** pyserial)——**服务器与测试跑在两个不同解释器上**,依赖集不同,这正是"pyserial 缺失"结论互相矛盾的根源。修法:① 启动器优先使用仓库 `.venv`(存在则用,否则退回 `where python` 并在横幅打出实际解释器路径);② `pyserial` 从 optional extra 提升为**必需依赖**(默认后端就是 UART,默认依赖不该可选);③ auto 因缺 pyserial 回退时,横幅打出可直接照抄的安装命令。
  判据:横幅含实际解释器绝对路径;`pyproject.toml` 主依赖含 pyserial;缺 pyserial 的回退分支有测试断言横幅含安装命令。

- [x] V1.4 **jtag-axi 回退被 U4 顺带砸了**:commit `f98ba21` 删掉了 `call "%FPGA_DIR%_resolve_tools.bat" vivado`(旧版会扫 `C:\Xilinx\Vivado\<ver>in` 并设 `ZLC_PS_VIVADO_BIN`),而 `transport/axi.py:22` 只会退化成裸 `vivado`。实测:Vivado 已装但不在 PATH 时,auto 与显式 jtag-axi **都 exit 3**。恢复该发现步骤(或等价物),使回退后端在 Vivado 已安装未入 PATH 时仍可用。
  判据:回退路径在 Vivado 不在 PATH 的环境下仍能定位到可执行文件(测试用假目录树验证发现逻辑,不跑真 Vivado)。
- [x] V2.3 **启动横幅零守卫**:`_main` 的横幅(最终后端 / 选择原因 / 逐口 attempts / jtag-axi 的 vivado 内存注记 / 显式失败 exit 2)全靠人眼。补 offscreen 测试钉住这五项文本要素。
- [x] V3.1 **删 6 个纯别名 + 1 个单行包装**(违"无后向兼容、干净删除"):`Slot`=`PulseSlot`、`PulsePort`=`PulsePortSpec`、`TargetIR`=`CompiledProgram`、`compile_sequence`=`compile`、`StreamerGeometry`=`StreamerParams`、`RemotePulseServer`=`PulseRemoteServer`、`pack`→`pack_program`。每组**只留一个名字**(其中 `TargetIR` 与 `compile_sequence` 在 src 内零调用)。**并且 `compile` 遮蔽 Python 内建,必须改名**——保留 `compile_sequence` 这一个名字,删 `compile`。
  **另有一处更硬的理由必须删 `compile`**:它把**同名子模块永久遮蔽**了——`__init__.py` 先 import 子模块(绑定 `zlc_pulse.compile` = 模块),又把同一属性重绑为函数,于是 `import zlc_pulse.compile; zlc_pulse.compile.CompiledProgram` 直接 `AttributeError`。契约是跨仓并行开发的依据,这种"按契约写却拿不到"的坑必须消灭。
  判据:`__all__` 内无两个名字指向同一对象(写成守卫测试,用 `id()` 比对但**要排除小整数驻留的假阳性**——`CMD_FIRE`=2 与 `STATUS_RUNNING`=2 是数值相同的不同概念,不算别名);`builtins` 同名导出为零;`import zlc_pulse.compile` 后能取到模块成员(守卫测试)。
- [x] V3.5 **删死形状 `repeat_from_index`**(**只删这一个**;对抗复核已证 `scan_points`/`scan_point_durations`/`repeat_forever` 是**承重字段**,调用方经 `dataclasses.replace` 设置,notebook 三处这么用,不许一并删):`CompiledProgram.repeat_from_index` 是恒返回 0 的 property,`wire.py:576` 还对它做防御式 `getattr(..., 0) or 0`。已查实:**参照树 `Zou_lab_control_v1/zlc_pulse/fpga.py:85` 同样硬编码 0**,所以不是迁移丢失,是 RTL 有能力而主机侧从未使用。删掉该 property 与防御式读取,改为在写控制字处用一个具名常量 0 并注明"RTL 支持 `repeat_from_loop_start` 回卷,当前无主机路径发射";`engine_model` 作为 RTL 孪生**保留**该寄存器建模(它模的是硬件,不是编译器)。在阻塞记录里登记"RTL 未暴露能力:循环回卷",供将来物理需要时专轮暴露。
- [x] V3.6 **行数判据失守且无守卫**:src 实测 **6,476 行**,越过 GOAL 自订 ≤6k 硬上限 476 行,而全仓没有任何行数守卫,这条判据一直靠目测。补守卫测试;**并裁决分母**:`engine_model.py`(1,165 行,占 18%)是 RTL 周期级仿真孪生、生产路径零调用、仅 tests 与 notebook 使用——**从"设备包行数"分母中显式除外**并在守卫里写明理由,余下部分守 ≤5.5k。

- [x] W1.1 **迁回 XDC→PulseTarget 派生**(铁律「迁移不是发明」适用:以 v1 `zlc_pulse/manifest.py`(294 行)为底本改造,**不要凭空重写**;参照树只读)。产出一个公开入口,从 `board.xdc` 派生完整 `PulseTarget`:
  ① **lane 顺序 = XDC 中的声明顺序**(v1 即 `ch{len(result):02d}`,保持);
  ② **端口分类规则必须从 XDC 自身结构推导,不许硬编码任何名字表**:裸名 = digital TTL 端口;位向量 `name[i]` = 一条 DAC 总线(位宽 = 该向量的位数,lane 按位序);`da_clk*` = 对应总线的 latch clock;`clk`/`uart_rx`/`uart_tx`/`led[*]` 等非脉冲信号排除。分类规则写进契约,**换板换名字不需要改代码**。
  ③ 派生结果同时携带 package pin(v1 的 manifest 就带,真机排障要用)。
- [x] W1.3 **删除任何"手写板级映射"的可能**:全仓(含 notebook、examples、tests)grep 不得出现 `'cooling'`/`'emCCD'`/`'da_dipole'` 之类的**硬编码信号名字面量**,除了 board.xdc 本身与专门测试解析器的夹具(夹具用假名字如 `sig_a`,不许复制真板名)。
  判据:守卫测试扫描上述位置,命中即红。

- [x] W3.1 **实测现状**:`fpga.py` 子模块**已建**(22 个构建/容量工具),但用户点名的这些**一个都没撤出包级 `__all__`**——`CtrlWords`、`CMD_FIRE/LOAD/RESET/SAFE`、`STATUS_*`(5 个)、`IMAGE_MAGIC`、`REGISTER_LAYOUT_ID`、`build_fingerprint`、`region_bases`、`scan_bank_words`、`exact_ticks`、`evaluate_affine_tick`、`check_rtl_assumptions`。它们是**线协议与打包内部件**,不是"用 pulse 设备"要用的东西,却和 `PulseStreamer.fire()` 平级摆在同一个 import 里。
  修法:包级 `__all__` 收敛到**运行时设备 API**——设备(`PulseStreamer`/`RemotePulseStreamer`/`DoneReport`/`SafeReadback`/`AppliedState`)、序列模型(`PulseSequence`/`PulsePeriod`/`AnalogStep`/`PulsePortSpec`/`PulseTarget`/`PulseSlot`/`OutputDelay`/`RepeatRegion` 等编辑时真要用的)、编译(`compile_sequence`/`CompiledProgram`——它是 `compile_sequence` 的返回类型与 `load()` 的入参,**保留**)、传输与远端(三个 transport + `RegisterTransport`(构造形参类型,契约点名,**必须留**)+ `serve`/`connect`/`PulseRemoteServer`)、W1.1 的派生入口、错误类型。其余按性质迁入 `zlc_pulse.wire`(线协议/打包)或已有的 `zlc_pulse.fpga`(构建期)。
  **每撤一个名字,先 grep 它在 src/tests/notebook 的使用点确认改到新位置**(V3.2 已抓到 `check_rtl_assumptions` 其实在运行时路径被调用的误判,别再犯)。
- [x] W3.2 **导出清单契约测试**:`zlc_pulse.__all__` 必须**等于** `docs/contract.md` 里列举的名字集合(双向,多一个少一个都红)。这条上轮已列 V3.3,本轮必须真正建立——它是防止导出面再次膨胀的唯一机械手段。
