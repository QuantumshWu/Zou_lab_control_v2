# GOAL — zlc_workbench:组合根与 presenter(收官仓)

状态:NOT STARTED(**最后开工**;前置=zlc_ui/zlc_runtime/zlc_atom/zlc_pulse 各自 GOAL COMPLETE 且 contract 冻结。此文件先立框架,开工前按各仓落地后的实际契约校订一轮——校订是誊抄不是重设计)
仓库:`C:\Users\eadri\Dropbox\WorkCode\Github\zlc_workbench`

> 定位:**唯一允许认识所有包的地方**。所有 presenter(把 zlc_ui 的哑视图接到 runtime/atom/plot)、组合根、启动脚本、跨包 E2E 都住这里;任何领域逻辑/渲染/信号机制**不得**在此新生——只准接线。范围随 atom 的最小骨架:三节点两设备(virtual 优先)。
> ⚠️ 同名影子警示:树内也有 `zlc_workbench`。同 zlc_pulse 处理:`__version__`+路径断言,本仓自首 commit 起唯一 owner。
> 参考:随仓 `docs/survey-workbench-2026-08-02.md`(window.py 三缝手术图)。

## 铁律 / 仪式 / 收尾

> 🔴 **永不建 venv**(用户 2026-08-05 明令):依赖一律全局 `pip install`;脚本里**不许**出现 .venv 探测/偏好分支。血训:zlc_pulse 的启动器曾优先用仓库 .venv,而那个 venv 没装 pyserial,导致串口枚举抛 ModuleNotFoundError、返回零候选口、UART 探测循环一次都不进,服务器**插不插线都无条件退回 JTAG**(实验机上=常驻 1-2 GB vivado.exe)。

同各仓(绝不 push/小 commit/干净删除/不加防御仪式/守卫非空洞/GUI 测试 offscreen 对象级、像素人审)。加一条:**presenter 全部 Qt-free 可无头测试**(Qt 只在 view 装配与 wake 垫片),这是 pulse_editor controller 已验证的形态。

## 已落成(2026-08-05,虚拟全链可跑;每项都有机械守卫与实测)

- [x] **W0 引导** —— git init + pyproject + `src/` + 影子路径断言;`python -m zlc_workbench.tools.check_environment` 断言八个包各自解析到自己的仓、三个旧名彻底消失。**卸掉了占名的 `zou-lab-control`,补装了从未安装的 zlc_ui**(它当时静默解析成空 namespace 包)。
- [x] **落盘包 `zlc_durable`** —— 原子写 / 目录 fsync / 逃逸安全路径(224+33 行逐字节迁自旧树,`cmp` 验证),外加新写的日期路由 `<save_root>/<YYYY_MM_DD>/`。**不含 canonical/digest**。
- [x] **W1 headless session 门面** —— `ExperimentSession`:从写下来的装置开机(或模板)、配脉冲、点火、存档、关闭。**notebook 与 GUI 调同一批方法**,不是两套实现。
- [x] **呈现接线** —— `PlotPanelPort` 把 runtime 的 `BoardScheduler` 接到 zlc_plot 的 `RasterPlotHost`;`LiveBoard` 管 tick/commit/wake 三方与线程归属,Qt 只出现在最底下那个换线程的垫片里。**这条缝设计出来后从没被连过**,现在有真 live 帧穿过它的测试。
- [x] **W2 console presenter** —— 接哑视图,自己不做任何决定;**从不 import Qt**(有守卫),所以可无头测试,也因此和 notebook 共用同一条路。
- [x] **存档与 provenance** —— 按日期分组的 npz,信息层单键 JSON **不 pickle**;`provenance` 每 run 抓一次、只走设备公开契约、派生向上游走并扁平。**守卫在全新解释器里打开存档,不 import 任何 zlc_\***。
- [x] **W5 入口脚本** —— `python -m zlc_workbench.apps.task_console --workspace ... --check` 装配全链并跑 beat;`--check` 是启动自检,被测试守着。
- [x] **notebook 教程** —— 22 格,按功能分,每格 print,零 assert,带执行输出提交,**从头跑通零错误**。

### 实测跑通的链路
装置写盘 → 脉冲自述三窗口 bracket → 一发 `(1,1,3,32,48)` → 连拍 revision 递增且 generation 各异 → 存进 `2026_08_05/` → **重开进程**读回当时的曝光/ROI/脉冲名。

## 仍未做(按价值排)

- [x] **虚拟↔真机参数化契约测试** —— `zlc_atom/tests/test_sequencer_contract.py`,8 用例 × {real via MemoryRegisterTransport, virtual}。真件 8/8 直接过,虚拟件先红 5 条(不建模 firing 状态);修完虚拟件后又揪出 4 个测试文件里 7 处 fire 不 wait。
- [x] **W2 SelectionRouter** —— `selection.py`;发布归 zlc_runtime.SelectionBridge,命名/语义归 zlc_atom,本仓只搬手势。顺带两个根修:zlc_plot 的 SelectionEvent 现在带 `subject`(哪根上游轴被切,事后再问会答成另一个 projection);zlc_runtime 的桥对**已终结的 generation** 改为一次性 publish_final(有限测量发布即终结,原本永远拒绝)。
- [x] **W2 信号拓扑投影** —— `topology.py` + `SignalDataPlane.describe_signals()`(只出拷贝)。Add Panel 从此有东西可加(原本连着空信号),Pause 可逆(原本每次都 emit True),Selectors 真断桥。
- [x] **W3 pulse editor presenter** —— `pulse_editor.py`:投影 + 编辑意图 + preview 时间线;合法性仍由 zlc_pulse 裁决(编辑造出非法序列就被模型拒绝,编辑器留住上一版)。Save 明确拒绝覆写 pulse 文件。
- [x] **W4 figure viewer** —— `viewer.py` 五页投影(Plot/Measurement/Device/Flow/Raw)+ `apps/figure_viewer.py`;存档现在带 dataset 身份(zlc_data 的 manifest 投影),存下的图能**重新打开**而不只是重新读数;绘图按**轴 role** 选,不按位置。
- [ ] **W6 真机接线** —— qCMOS 首光(DCAM SDK 未装,日历前置)、真 pylon、真 FPGA;混合装配已可用(逐设备 init,`hardware` 模板已加)。**唯一未做项,卡在硬件不在手上。**

## 阻塞记录

(受阻时追加)

## GUI 逐控件审计(2026-08-06 起,逐条打通不中断)

对账口径:zlc_ui 四个 pulse 页面共 **45 个外发信号 / 30 个 setter**;接线前我只应答了 19 个信号、只调了 6 个 setter。Scan 页与 Target 页**整页无人应答**。v1 参考=`ZLC_main/Zou_lab_control/frontend/pulse_gui.py`。

### A. 端口模型:DAC 是一个集合,clk 不单列
- [x] A1 pulse 表只列**可编程端口**(digital + DAC-as-bus);clock 端口不作独立行(v1 `_display_rows` 跳过 `PORT_CLOCK`)
- [x] A2 DAC 行=该总线**全部 lane + 它的 latch clk**,一个条目;lane/pin 进 tooltip
- [x] A3 clk 驱动的 lane 在 period 卡里锁定+置灰(v1:"wired to the FPGA clk (not engine-driven)")

### B. Target 页(整页未接)
- [x] B1 `set_ports(records, editable, status)`:硬件列=**固定 lane/pin**,右列=可编辑显示名
- [x] B2 `apply_requested` → 只改 label(不动拓扑/ABI 指纹)
- [x] B3 `set_width_rules` + `feedback_requested`

### C. Preview 页
- [x] C1 `set_size_names(PANEL_SIZES)`——Size 下拉现在是空的
- [x] C2 `size_committed` + 选过即 **pinned**(不再自动跟随内容)
- [x] C3 "Show off rows" 真的显示全部通道(含常关)
- [x] C4 `selectors_toggled`、`save_requested`(存图)

### D. 运行语义
- [x] D1 **默认整条 pulse repeat forever**(v1:On Pulse = 连续跑到 Stop,总是 `repeat_forever=True`)
- [x] D2 Stop = safe;运行中控件状态随之
- [x] D3 Sync = 只上传不发射

### E. 视图稳定性
- [x] E1 On Pulse 之后**不许跳回顶部**:整表重建必须保留滚动位置

### F. Scan 页(8 信号整页未接)
- [x] F1 repeats / hold / step / load_program / template / run / save_array / progress_refresh

### G. Edit 页剩余信号
- [x] G1 `binding_cycle_requested`(duration/DAC 字段 off→scan→api 循环)
- [x] G2 `scan_array_load_requested` / `scan_source_committed`
- [x] G3 `left_panels_collapsed` —— **view 自持**:Collapse 按钮已由 view 自己折叠并事后通知,presenter 无事可做。不假装接线。
- [x] G4 `feedback_requested`
- [x] G5 未调的 setter:`set_period`/`set_delay_row`/`set_port_label`/`set_visible_ports`/`set_scan_source`/`set_scan_busy`/`accept_local_commit`

### H. 编辑器外壳
- [x] H1 `close_requested`(已由 launcher 绑定,复核)/`save_path_requested`/`open_path_requested`
- [x] H2 `set_status_color`(状态点)/`set_capabilities`

### 值/结构分界(用户当场指出,已改)
- [x] 点一个 channel 不再重投整表:值编辑走 `set_period`/`set_delay_row`/`set_port_label`/`set_summary`,只有形状变更(增删/移动 period、换 target、换 pulse)才 `set_schedule`;`accept_local_commit` 残余已删。

### I. 其余 GUI 同样待审
- [x] I1 task_console —— panel 卡六控件全接(signal/size/interval/title/edit/remove,分组按 producer);
      Logic 页整页接通:catalog=`discover_logic_nodes()` 非自备清单、Add/Start/Stop/Edit/Remove 五路、
      按 descriptor 声明绑定设备与参数(缺件当场报原因,不留 idle 假象)、运行中拒改(记录不能撒谎)、
      Remove 先停后撤(还握着相机的行不能先消失);board 拖放次序回灌 presenter(存档次序=看见的次序)。
- [x] I2 figure_viewer —— 多 dataset 选择器 + Save image(只画第一个 = 看起来像只存了一个)。
- [x] I3 device_manager —— 新 presenter + app + `bin\device_manager.bat`;类型目录来自
      `discover_device_types()`,表单来自各类型自己的 AuthoringSchema(`authoring_form` 单源),
      换型保留共有设置,存盘合法性由 `InstallationConfig` 裁决而非二次判断。

### J. 全仓单一真相源审计(2026-08-06)
- [x] J1 `SelectionChange` 两份同枚举(zlc_plot + zlc_runtime)→ 落 zlc_data;原先只靠 str-enum 值相等碰巧互通。
- [x] J2 `finite_real`/`integer` 两份 → zlc_plot 委托 zlc_data(plot 只留 text/readonly_copy 等自己的)。
- [x] J3 `zlc_runtime/resources.py` 整模块零调用者(活的在 zlc_atom.execution,且边界禁止域侧 import runtime)→ 删。
- [x] J4 `zlc_atom/devices/sequencer/protocol.py` 手抄契约漂移(`SafeReadback` 三字段真板子从来没有;
      虚拟 sequencer snapshot 只答 3/11 键)→ 对齐 + zlc_workbench 加跨仓机械守卫。
- [x] J5 `zlc_ui` `SerialWorkerWindow`(第二套窗口模型,零调用者)→ 删;`QtOwnerWake` 保留(gallery 在用)。
- [x] J6 端口 18861 写在五处四包 → `zlc_pulse.remote.DEFAULT_PORT/DEFAULT_HOST`;装置表单那份加守卫比对。
- [x] J7 `project_authoring_form` 消费一套本仓无人生产的字段词汇(拆包前遗留)→ 删,换成真正需要的
      `zlc_ui.form.SettingsDialog`(spec 上的模态表单)。
- [x] J8 无界 join(`zlc_runtime/live_dataset.py`、相机 owner lane)→ `join_worker` 有界并报错。
- [x] J9 虚拟相机 producer 非守护线程 → 崩溃后进程永不退出、traceback 永不现身 → 改守护。
- [x] J10 第一轮 workflow(24 agent / 232 条原始 → 200 条确认)高危项逐条落地。
- [x] J11 第二轮 workflow(16 agent / 96 条原始 → 67 条确认)——schema 机制裁决 + 五个新角度。
      裁决:**摘要机制承重但 SHA-256 不承重**。承重的唯一理由是 `CommittedTransform`
      落盘且**不带 schema**,明天另一个进程只能靠名字认出它;`hash()` 每进程随机化,不行。
      不承重的是"拿名字核对手上就有的东西"——`schema_equal` 已改 `==`。全项目统一
      **BLAKE2b-128**(`DIGEST_BITS`),`sha256_text` → `digest_text`,测试宽度全部派生。
      **schema 本身承重**,但 zlc_plot 从不读 `AxisSpec.role`、改按 size/位置猜,
      导致两窗口 bracket 与 1×W ROI 被直接拒画——猜的那一方删掉,声明成为唯一来源。
- [x] J13 pulse editor 三症一根(用户报:delay 不载入 / scroll 无反应 / virtual-offline 语义)。
      **根因是两条平行投影**:`project_board` 是 `project_schedule` 的残缺副本,只建通道名列、
      从不建 delay 列,连"哪些口该有 delay 行"都写成两种拼法。合成一条,`refresh` 三路收成一路
      (顺带去掉重复调用的 `refresh_target`)。第二根:`_release` 只收 sequencer,把 `board`/`pins`
      留在原地 → 连过一次之后编辑器永远认为自己还接着,Target 页永远只读,**offline
      这个唯一以编辑拓扑为目的的模式再也编辑不了拓扑**。v1 语义已核对(offline=无板可编拓扑 /
      virtual=可发射的仿真板 / remote=真板;delay 永远来自 pulse 文档,缺省即 0)。
- [x] J14 滚动条:两栏各自一个滚动区、交叉连接、指定其一可见 —— "有没有超出显示范围"被回答了两遍,
      一高一矮时两答案不一致,当选的那根决定操作员能够到哪。连上板子还没建 pulse 时时间轴是空的,
      于是 22 通道那栏根本滚不动。新增可复用 `zlc_ui.fluent.LinkedScrollPanes`:一根条、
      范围取当前最深的那一栏、没人超出就自己隐藏。Target 页那根"常驻但拉不动"的也改按需。
- [x] J15 面板只会画 image:每个 panel 都硬写成 `ImagePlot(spatial-x, spatial-y)`,而派生的
      ROI 总量只有 `zlc_data.scalar` 一根轴;figure viewer 还有第三份自己的 image/curve 二分。
      每种 kind 本来就有 `default_spec(schema)`,缺的只是入口 → `zlc_plot.fitting_spec`。
- [x] J16 Add Logic 三处断线:选择器给每一行硬写 "available"(由唯一无从检查的地方声称);
      `build_arguments` 一直收 artifacts 而**从没人喂过**,所以 occupancy 无论标定跑没跑都建不出来;
      processor 从没被问过读哪个信号,于是失败信息是一句关于 runtime 的话。全部改为按声明驱动
      (契约 id 对接产物、`DatasetInputSpec` 决定要不要问)。
- [x] J17 `CalibrationTask` 只有 `run()` 而 NodeHost 走 `execute(ctx)` —— descriptor 声明了一个
      runtime 驱动不了的节点,一按 Start 就炸。补上 host 入口(领 generation、经 context 发布)
      + provenance 记录器;新守卫逐个检查每种 kind 的驱动入口,**并断言自己真的检查到了**
      (第一版因延迟标注是字符串,一个都没查到却全绿)。
- [x] J18 `resolve_pulse(search_paths=)` 名为"要搜的目录"实为"工作区根、内部再拼 pulses" ——
      唯一照字面理解的调用者得到 `pulses/pulses/<name>.py`。改为名副其实,目录布局只由 workspace 说一次。
- [x] J19 存档只记 session 自己的节点 → 窗口里跑出来的东西一律无溯源;面板卡的 100ms 是编的
      (真值 400)。前者改记"真正产出的节点",后者由 presenter 告知卡片。
- [x] J20 `zlc_runtime/signal_source.py`(655 行 + 179 行测试)删除:无人 import、不在 `__all__`、
      不在 contract,且它服务的 association 族已被 contract 判死。**同文件邻居看着一样"无人引用"
      却全是活的**(都在 `AcquisitionStream` 内部铸造/抛出/往返)——跨文件引用计数说明不了内聚模块。
## K. GUI 控制面机械台账(2026-08-06;不靠勾选框,靠数)

口径:枚举 zlc_ui 全部 `pyqtSignal` 与全部公开 `set_*`/`show_*`,逐个问"**定义它的类之外**有没有人应答"。
起始 **203 个控制面 / 24 个无人应答**;下表逐条裁决,现剩 19(全部为"故意保留"或"待接")。
脚本口径已验:五条用户点名项(DAC 成集合、on_pulse 不上跳、默认 repeat forever、preview size/show-all、
target pin)**实测全部成立**,是我上一版探针找错字段——不是代码问题。

- [x] K1 **点选决定编辑落点**(v1 核心交互,整体缺失)。`period_clicked` 发了没人听、`gap_clicked`
      **声明了从没 emit 过**、`_selected_before_id()` 是返回 None 的桩 → Add 只能追加、Remove 只能删最后。
      按 v1 复刻:选卡片=插其后/删它,选间隙=插那儿,两者互斥,再点取消;重建时选中项自裁(卡片没了就清,
      间隙超界就清)。`FluentGroupBox.set_outline` 无人调用的原因也在这——它当初就是为这个高亮迁来的。
      **三个"无人应答"其实是一个缺失特性。**
- [x] K2 **Hide Off 从来没隐藏过任何东西**:它读 `PortRowVM.active`,而 `project_ports` 硬写 `True`。
      这个标志天生会漂(通道开关是值级编辑,端口行只在结构变更时重推)→ 删标志,由**持有 periods 的
      view 现算**。全关时说明原因而不是把整块板子藏起来。
- [x] K3 **Show All 会崩掉编辑器**:`set_visible_ports` 不换 revision,Hide→Show 两个不同模型同 revision,
      view 正确拒绝,而拒绝发生在 Qt slot 里 = 进程 abort 无 traceback(看着像 segfault)。
      根修不是补一次 `+= 1`——**让 presenter 自己看**:`_push_schedule` 在交付处比较上一次模型,
      不同则进位。谁都不用再记得。
- [x] K4 `set_scan_busy` 零调用者 → 载入扫描表期间按钮不再可点(网络盘上第二次点击会和第一次抢 `_scan_rows`)。
- [x] K5 `PanelCardView.set_selectors_enabled` 零调用者 → 关掉 selectors 后卡片仍显示控件可用,
      而背后 bridge 已关:一个看起来能用、实际不能的控件。
- [x] K6 `set_close_guard` 零调用者 → **console 关窗顺序反了**:渲染线程/举着相机的节点原本在 `closed`
      (关闭已提交之后)才释放,中途失败就剩一台还举着相机的设备集和一个够不着它的进程。改为守卫式:
      放不干净就不关,操作员可以再试。
- [x] K7 `set_visible_ports`(view)在 `docs/pulse-views.md` 里是声明过的接口却无人调用 → 接上值路;
      **顺带挖出更根本的一条**:`_kept_scroll` 只包在 `set_schedule` 一个调用点上,而真正会重建的是
      `_reconcile` —— 所有值级 setter 都绕过它,于是 22 行的板子每次 Hide Off 都被扔回顶部。
      保位逻辑移进重建本身,并纳入操作员真正拖的那根共享滚动条。
- [x] K8 gallery 里那个可关标签页的 X **按下去什么也不会发生**(`tab_close_requested` 无人接)——
      一个演示不工作控件的演示。接上;并改掉 `FluentTabWidget` 描述"每面板一个 Edit 标签页"的 docstring:
      console 的 Edit 早已是 `zlc_plot.edit_plot_display` 独立窗口,那段描述本身是残余。
- [x] K9 `RepeatBracket.set_repeat_count` 删:与构造参数重复的第二条更新路,只会变陈旧。
- [x] K10 已裁"保留,不动"(各有其主,非漂移):`PulseScanView` 六个 setter(`pulse-views.md:65-69`
      声明的接口,由 `set_page(record)` 在 view 内扇出)· `PulseEditorView.warning_requested/show_warning`
      (view 自环,状态条不被下一次重绘冲掉)· `left_panels_collapsed`(G3 已裁)·
      `FlowGraphView.set_role_styles`(`test_controls_smoke.py:167` 在跑)· `FluentWindow.hidden`
      (notebook `%gui qt` 嵌入语义)· `QtOwnerWake.requested` / `keyed_choice_picked` /
      `action_clicked`(控件库自然 affordance,单行成本)。
      **台账口径:203 面 → 无人应答 24 → 现 15,余下全部为已裁"保留"。**:`FluentWindow.hidden/set_close_guard/set_hide_guard`、`QtOwnerWake.requested`、
      `FluentTabWidget.tab_close_requested`、`FluentTreeComboBox.keyed_choice_picked`、
      `FluentStatusStrip.action_clicked`、`FlowGraphView.set_role_styles`(通用件设施,判"留"还是"删");
      `PulseScanView` 六个 setter(经 `set_page(record)` 在 view 内扇出,判"合理聚合");
      `PulseEditorView.warning_requested/show_warning`(view 自持);`RepeatBracket.set_repeat_count`、
      `PulseScheduleView.set_visible_ports`(值级路径待接);`left_panels_collapsed`(G3 已裁"view 自持")。

## L. 其余 GUI 逐控件驱动(2026-08-06;真窗口、真台架,不看代码看行为)

方法:开真窗口,把每个控件按操作员的顺序点一遍,看它到底做没做、做对没做对。
先机械扫全仓 presenter/view 的**桩函数**(单语句 `pass`/`return 常量`)——只剩 2 个且都是有意为之,
说明 K1 那类"应答了但答的是常数"已清干净。

- [x] L1 **console 对第二个信号 Add Panel 直接失败**("raster plot host is closing")。
      根因在 zlc_plot:`AxisRef.point()` 要的是**坐标 id**,而 `curve.default_spec` 传的是**显示名**。
      两者常常相同,直到某个 producer 把列名起得和坐标不一样(标定任务正是如此)——于是 PlotSession
      构造期抛错,RasterPlotHost 把它变成永远 closing 的 host,报出来的是一句**关于 host 的话,
      回答的却是关于轴的问题**。`facet_grid` 继承同一推断,一起错。
      **更深一层:测试工厂造不出这个 bug**——它给每个点列的 id 恒等于名字,于是"查错了那一个"整类
      缺陷在这里天生测不出。已让工厂能表达差异 + 加一个两者不同的 schema 家族 + 守卫逐条解析
      每种 kind 默认规格点名的每根轴(走各 kind 自己的 `label_roles` 声明,新增 kind 自动纳入)。
      已验守卫会红。
- [x] L2 **figure viewer 用内部编号当数据集名**:存档里明明记着每个 dataset 的 panel 标题与信号,
      viewer 只把它读进信息行,选择器给的是 `panel-1`/`panel-2` —— 操作员得猜哪个是相机。
      改为 (key, label):key 存档用、按 item data 回传(和本项目其它选择器一致),label 给人看;
      图的标题同源。没有 panel 记录时回落到存档自己的名字——**标签只能是存档知道的,不能是编出来的**。
- [x] L3 **存图叫 `panel-1.png`**:一天的文件夹里三十张图没人分得清,而面板一直带着标题。
      改为按标题命名,`_file_stem` 只留文件系统到处都收的字符(信号键有 `/` 和 `@`,标题是操作员随手打的)。
- [x] L4 device_manager 全链驱动通过:添加 / 换型(`exposure_seconds` 跨 dcam→pylon 保住)/ 改角色 /
      存盘 / 重复角色被拒 / 删除。
- [x] L5 task_console 全控件驱动通过:pause·selectors·signal 选择·resize·interval·rename·reorder·
      save·save images·remove;Logic 行三阶段报述属实(idle/not started → running/starting → idle/done,
      附已发布信号与存活状态),Remove 先停后撤。

- [ ] J12 最终全项目审查(16 agent,含本轮全部改动的回归检查)——待跑。


# ═══════════════════════════════════════════════════════════════════════
# M 轮:长跑任务书(2026-08-06 定稿,可直接进 /loop 连续跑)
# ═══════════════════════════════════════════════════════════════════════

## M0 开工仪式(每次唤醒,先做完这四步再动手)

1. `cd C:/Users/eadri/Dropbox/WorkCode/Github` 并读**本文件的 M 节**——它是唯一计划权威,
   永不新建第二份计划文档。
2. 八仓 `git status --porcelain` 必须全干净;不干净先弄清是谁留下的,再决定提交或回退。
3. `tasklist | grep python` 必须为空——上一轮留下的 GUI 窗口先杀掉。
4. 挑 M 节里**第一个未打勾**的条目开工;一个条目做完就 commit(**永不 push**),并把
   本文件对应行改成 `[x]` + 一句根因。

## M∞ 三种合法收尾(只有这三种,其它一律继续做)

- **A 全清**:M 节所有条目 `[x]`,八仓绿,工作树干净 → 写一句终止陈述,停。
- **B 阻塞**:某条目需要我做不到的东西(硬件不在手上、需要你裁决的设计),把它标 `[?]`
  + 写清卡在哪、需要什么,然后**跳过它继续做下一条**;全部条目非 `[ ]` 时按 A 收尾。
- **C 增量归零**:连续两轮没有任何条目推进 → 写极简终止陈述,停(不要空转唤醒)。

## M-决策(2026-08-06 已确认,不要再问)

| 问题 | 裁决 |
|---|---|
| repeat 语义 | 编辑器**当前只允许一个 bracket**(未来可能多个)。「最外层 bracket」=包住**整条 pulse** 的那个;**它存在就 override 默认的 repeat-forever,改为按它指定的次数重复**。默认(无 bracket)=forever。 |
| v2 仓库形态 | **合并成一个**(单 git 仓 + 单发行物,内部保持 `zlc_*` 子包结构)。**现有散落的包一律不删**。 |
| bin 启动 | 只留 `pulse_editor.bat` + 一个 `experiment.bat` 做「device_manager 配置 → init → task_console + pulse editor」的编排。 |
| 消息 | **全部走 `fluent_message` 阻塞模态**,和 v1 一致。成功与失败都弹。**不许再自造 header/status strip 之类的新展示件。** |
| zlc_ui 顶层 API | ~~豁免~~ **2026-08-06 推翻:不但不豁免,而且最严**——见 MA 节。其余包一律不得跨包 import 子模块——要用就提到顶层 `__all__` + 契约 + tutorial。 |
| GUI seam 形状 | **A:不透明句柄,窗口归 zlc_ui**。`open_*() -> handle`,handle 上只有信号 + `set_*`/`show_*`,外部**一个 QWidget 类都不 import**。 |
| VM 数据类 | **公开**——它们就是连线契约,提到 zlc_ui 顶层 `__all__` + tutorial。 |
| zlc_plot widget | 绘图面板是它的产品,notebook 直接用。**但 widget 本身不许穿缝**:workbench 传的是 zlc_plot 的**宿主对象**(host/surface),zlc_ui 鸭子类型地问它要 widget 和 logical_size。workbench 可以持有 host,**但机械禁止它取 widget**。两条铁律都不破:zlc_ui 零 `zlc_plot` import,workbench 零 QWidget。 |
| zlc_atom | 豁免(logic_node / device 是 plug-in 形态)。 |
| v1 pulse JSON 兼容 | **不做**,除非另行通知。 |

## M-铁律(每条修改都要过一遍)

- **不打补丁**:先找本源、理解原理,再从高层/架构入手。性能问题必须 profiling 定位 + 交叉验证根因。
- **DRY / 唯一真相源**:同一逻辑散在多处就是缺陷本身,合并它。
- **不留历史残余**:不为迁移风险、不为迁就旧 test 留兼容。
- **接线不算修好**:每个按钮必须**真的从真入口跑通一遍**(`create_window` + 真台架),
  只连信号不验证路径,视同没做。这是 M1 整节存在的原因。
- **批判性继承 v1**:只继承最基本的思想,v1 明显错误或糟糕的设计不要抄。

---

## MA GUI 密封:zlc_ui 只暴露整体句柄(2026-08-06 加入,**最高优先**)

**为什么这条排在最前**:今天修的 M1-2 就是它的直接后果——presenter 越过整体视图去戳
`schedule_view` 的一个子面板,另一个子面板(delay 列)就漏了。**外部能拿到 widget,外部就会
自己攒一套 UI**,于是「谁负责显示」这件事又变成散在两边的两份。已量:workbench 里 22 处
跨包 import zlc_ui 子模块,`test_pulse_editor.py` 里 73 处直接戳子视图。

**规则**(zlc_ui 顶层 `__all__` 只许有这四类):

1. 每个 GUI 一个开窗入口 `open_pulse_editor / open_task_console / open_figure_viewer /
   open_device_manager(*, title=, window_ratio=, …) -> handle`。**窗口生命周期归 zlc_ui**。
2. handle 上的声明式端口:**信号**(外部往里接线)+ `set_*`/`show_*`(外部往里投喂)+
   `close()`/`closed`。**没有任何 QWidget、没有任何子视图属性。**
3. 连线词汇:VM 数据类(ScheduleVM/PeriodVM/FieldVM/PortRowVM/DelayRowVM/TargetPortRecord…)、
   FormSpec 家族、BoardMetrics。
4. `ensure_qt_app`(notebook 需要)+ `capture_window`(验收截图,改成接受 handle)。

其余全部私有:`fluent`、`pulse.*`、`console.*`、`device_manager.*`、`figure_viewer.*`。

**绘图怎么进来**(两条既有铁律的交点:zlc_ui 被机械禁止 import zlc_plot/numpy/matplotlib,
而面板天生是 QWidget):**传宿主对象,不传 widget**。workbench 拿着 zlc_plot 的 host,
`handle.show_preview(host)` / `handle.mount_panel(panel_id, host)`;zlc_ui 鸭子类型地向 host
索取 widget、logical_size、wheel target。workbench 可以持有 host,**机械禁止它自己取 widget**。

- [x] MA-1 契约从真实调用点机械抽出(AST 扫描),写进 `zlc_ui/docs/contract.md`;facade 7→20 名。
      **扁平化立刻逼出两处被页面形状藏住的重名**:schedule 与 scan 都叫 `run_requested`、
      schedule 与 preview 都叫 `save_requested`。
- [x] MA-2 pulse editor 已切 handle(`open_pulse_editor` → `PulseEditorHandle`,124 个成员)。
      workbench 零 QWidget;preview 改传 zlc_plot 的 **host**(新增 `qt_widget()`/`logical_size`)。
      `test_view_contracts` 改校验 handle 并当场抓出我偷懒的 `*args` 签名;三仓全绿,
      真入口实跑:1152×653、`is widget: False`、开得了关得掉。
- [x] MA-3 task_console 切 handle。**这一个的缝最糟也最说明问题**:外面被塞了一堆**工厂**去造
      widget,再把造好的递回来;每张卡六根线由造它的人重接一遍,漏一根=控件看着能配其实不能。
      卡片与逻辑行归窗口后暴露出一件隐含事实:「面板是这个顺序」和「卡片是这个顺序」变成两句话,
      而只有第一句在说 → 补 `set_panel_order`。绘图包自己的对话框以**函数**穿缝(`run_host_dialog`)。
- [x] MA-4 figure_viewer 切 handle。它还有第二条越墙路:presenter 被注入 `surface_of` 回调
      自己把 host 变成 QWidget——隔了一层的「组合根造 Qt 对象」,离墙一步都不远。已删。
- [x] MA-5 device_manager 切 handle。真机开窗时抓到一个**与 MA 无关的既有缺陷**:设备只存
      「被设过的」参数,类型一旦新增字段,之前存的 apparatus.json 就永远打不开(表单正确地拒绝
      不全的键)——编辑器打不开自己存的文件,而那正是它存在的理由。schema 说有哪些字段,文件说
      选了什么。
- [x] MA-6 launcher 退出跨包使用(`test_windows` 现在断言 apps/ 里**零** launcher 调用);
      `capture_window` 在包内解开句柄;`WINDOW_SCREEN_FRACTION`/`capture_window` 提到顶层。
- [x] MA-7 `tests/test_gui_seam.py` 三条:不许 import zlc_ui 子模块、不许 import PyQt5
      (只豁免 `board.py`——它要的是线程跳转和定时器,不造/不持/不显示 widget)、四个窗口
      都必须是「一次调用换一个句柄」。**跨包子模块 import 22 → 0**;顶层 25 名,契约与
      tutorial 同步,notebook 只教门面。
- [x] MA-8 台账已在新 seam 上重跑(枚举+驱动脚本按句柄改写,四个 GUI 全过一遍)。

## M1 用户当面报的故障(逐条真机验证)

- [x] M1-1 **Save / Sync 一按就崩进程**。根因不是 severity 写错——是我擅自加了一个 header
      `StatusStrip`(commit 4acc7da,你从未批准),又自造了它不认的 severity 词汇。
      **删掉那个 header,消息全部走 `fluent_message`**(见 M-决策)。
      ⚠ `zlc_ui/src/zlc_ui/pulse/editor_view.py` 已有一份未提交的改动做了这件事,已验 Save/Sync
      不再杀进程;开工先复核它是否符合「不许自造展示件」,再决定提交或重做。
- [x] M1-2 **Hide off 等操作没有改变 delay 那一栏**。根因=「哪些端口有行」写了两遍:名字列和
      period 卡过 `port.visible`,delay 行却是 target 直推、没过滤。已在视图里落一处交集,
      重建和单行推送都过它。顺带根治了 GUI 测试 1/4 概率的 access violation:**没人拥有关机**
      ——CPython 清 wrapper 与 Qt 析构赛跑;唯一入口现在在 atexit 里确定性拆窗口(zlc_ui 8d7bc11)。
- [x] M1-3 美术:对回 v1 的两根「无标题、卡片同高、Repeat 在表头行、次数在首控件行」立柱
      (PeriodCard 当时把 `panel_top_height()`/`period_control_width()` 手抄了一遍,所以柱子和卡片
      对不上)。语义根因=**「这个 bracket 是否取代外层」谁读谁自己算一遍**:preview 算了、
      一个失去调用者的 helper 算了、**真正 fire 的那条路没人算**。已落到文档自己身上
      (`PulseSequence.whole_pulse_repeat`);次数本来就随编译产物的唯一循环区进 FPGA,
      缺的只是别再把它包进 forever。删掉 `repeat_presentation.py`(无调用者的残余)。
- [x] M1-4 **逐控件真跑完毕**(机械枚举,不是手写清单——手写会漏)。pulse editor 437 个控件、
      console 16、device_manager 16、figure_viewer 7,全部从 `create_window` 真入口点下去,
      记录「说了什么 / 状态怎么变 / 有没有报错」。**跑出来六个真缺陷**:
      1. `slot ids must be unique` —— 名字只取了字段身份的一半,一个 period 里只能绑一个 DAC;
      2. Hold 被当成第三种模式送回模型 → 从 Qt slot 抛出 → 进程无声消失;
      3. viewer 原样存路径 → 相对路径进来后 Save image 抛异常关窗、不写文件;
      4. 教程存的是裸数组而非快照 → 它自己的 viewer 打不开(下一句还写着「两条路同一个 session」);
      5. device_manager 打不开自己存的 apparatus(类型新增字段后旧文件永远短一项);
      6. console 的 `set_close_guard` 装在句柄上没人接——**没有任何测试开过 console 窗口**。

## M2 顶层 API 收尾(zlc_data / zlc_runtime 已完成)

- [x] M2-1 zlc_pulse 8 名已抬(26→34),契约/tutorial/cap/行数预算全同步,每条写明为哪个缺失能力。
      `MemoryRegisterTransport` 顺带离开「留在子模块」明单:它被记成测试传输,而出厂的 virtual
      模式就建在它上面——那条裁定描述的是它还没进产品时的世界。tutorial 那格还硬写着
      `'127.0.0.1', 18861`,正是这两个常量存在的理由,一并改掉。
- [x] M2-2 zlc_plot 2 名已抬(59→61):面板可选尺寸、以及这条 pulse 该用哪个预设——
      两者都得重推 zlc_plot 的布局规则才能自己算,而自己算的宿主就是第二套布局引擎。
- [x] M2-3 机械核查复跑:**除 zlc_atom(豁免)外全零**。原有 45 处里,大多数名字本来就在门面上、
      只是走了后门(zlc_durable 4 / zlc_runtime 13 / zlc_data 13),已全部改走正门;
      zlc_data 另有 6 名真缺(选择词汇四件 + snapshot manifest 两件)已抬并进 tutorial。
      顺带抓到 `is_intrinsically_immutable_array`:**能 import、不在 `__all__`**——
      两头不靠,拥有方看不见自己的依赖。

## M3 其余 GUI 对 v1 逐条批判性审查(此前只驱动过控件,没做 v1 对照)

**对照表(2026-08-06 读 v1 源码得出;判断列=批判性继承的结论)**

### M3-1 task_console(v1 9461 行 / v2 console.py + 视图)

| v1 的做法 | v2 现状 | 判断 |
|---|---|---|
| Monitor/Logic 两个常驻页 | 有 | 已继承 |
| panel=纯视图,选到信号前什么都不显示,永不拥有测量 | 同 | 已继承 |
| 卡片边框兼作拖拽把手,落下吸附网格 | 有拖拽 | 已继承 |
| 逻辑节点**加进来是停的**,状态点 灰/绿/红 + Edit(参数表单 + Start/Stop) | 有 | 已继承 |
| **数据源是一行表达式**(`value = occupied - b_occupied`、`history('counts',200)`) | 只有信号选择器 | ⛔ **判定:不继承**(见下) |
| **布局(面板+逻辑节点+位置+尺寸+表达式+参数)存成一份可移植 JSON** | **完全没有**;v2 的 Save=存数据存档,Load=开已存的图 | 🔴 缺口:每天早上都要重搭板子 |

### M3-2 figure_viewer(v1 965 行 / v2 viewer.py + 视图)

| v1 的做法 | v2 现状 | 判断 |
|---|---|---|
| **不做专用查看器**:载入的图变成**一个 hub 信号**(`LoadedFigureNode`,声明与活产者相同的 output_specs),窗口就是一块**真的 TaskConsole 板**,用保存时的 kind+view 种下第一个面板 | **专用单图窗口** | 🔴 根本分歧:v1 白拿 Add Panel/选信号/改 kind/重接线/fit/重存;v2 一样都没有 |
| pulse 图走**完全相同**的路,无特例 | 不支持 | 🔴 同上 |
| Info 五页(Plot/Measurement/Device/Flow/Raw) | 有 | 已继承 |
| Browse 或敲一个合法路径**自动载入**,没有 Load 按钮 | 同 | 已继承 |

### M3-3 device_manager(v1 1234 行 / v2 device_manager.py + 视图)

| v1 的做法 | v2 现状 | 判断 |
|---|---|---|
| 左栏:按**设备域**分组,每条一张可折叠卡;类型下拉来自注册表并按域过滤 | 平铺,无分组 | 🟡 可继承(域分组) |
| **导入失败的类型置灰并写明原因**,绝不崩 | 未处理 | 🔴 缺口 |
| 类自己声明的参数表单(typed widgets,永不 eval) | 有(AuthoringSchema) | 已继承 |
| `$device:` 交叉引用渲染成「从本配置其它条目里选」 | 无 | 🟡 可继承 |
| 右栏:**Discovered**(总线扫描,每行「Add to config」)+ **Loaded**(活实例,每台 Snapshot 弹窗 + Open devices) | **整栏没有** | 🔴 缺口:装置管理器不能发现硬件 |
| Save / Save as… / Load / Apply;**无 session 打开时 Apply 变「Init devices」**,就地把窗口升级成有 session 的 | 只有 Save | 🔴 缺口 |
| 脏状态**事件驱动**:标题 `<config>[*]` + 状态点 灰=无 session / 绿=已同步 / 橙=有未应用的改动 | 无 | 🔴 缺口 |

**⛔ 表达式数据源:明确不继承,这是 v1 的错误设计**

v1 的面板在**绘制时**对 hub 里的当前值求值一行表达式。两个信号写在同一个表达式里,
就可能一个来自这一发、一个来自下一发——**没有任何东西保证它们同属一发**。这正是 v2 的
board-coherent tick 存在的理由(v2 自己的存档代码里写着:曾经按面板各自 freeze,结果一张
存图里半块板来自这一发、半块来自下一发)。照抄它等于把这个不一致重新引进来。

v2 的答案是:派生要成为一个**声明了输入的处理器节点**,于是那次派生有了 provenance、有了
generation 归属、能被存档解释。框架甚至机械地强制「处理器恰好一个数据集输入」
(`descriptor.py:123`),因为两个输入意味着要回答「哪一发配哪一发」——v1 从来没回答过这个问题。

⚠ 真实的人体工学缺口(记录,不在本轮修):v2 只有 occupancy 一个处理器,想「看一眼站点平均」
也得写一个节点。缺的是**一批通用单输入变换节点**,不是表达式框。

- [x] M3-1 task_console vs v1 —— 布局存/读已落地(见下方 commit);表达式源判定不继承并写明理由;
      顺带根治顶栏「加一个控件就撑破窗口」(`FluentFlowRow`,最小宽 907→148)。
- [x] M3-2 figure_viewer vs v1 —— **v1 的核心想法已继承一半并落地**:存档现在通过
      `LoadedFigure` 变成真信号(声明与活产者相同的输出),`ArchiveSession` 回答 console 要的
      两个问题,**console 已能在一个存档上无设备跑起来**(有测试)。窗口形态按你的裁决走 C:
      单图升级成一张**静态面板卡**(选择器/尺寸/标题/Edit;没有间隔与 Remove——静态图上那两个
      控件做不到自己承诺的事),viewer 自己的 dataset 下拉删掉(同一个问题不放两处)。
      ⚠ 未做(按 C 的代价):Add Panel / 多面板 / 板子级复用——那要走 A 或 B。
- [x] M3-3 device_manager vs v1 —— 三个 🔴 已落地,一个标阻塞,两个 🟡 判定如下:
      * 🔴 **导入失败置灰写原因**:`discover_device_types()` 原本裸调 `import_module`,
        任何一个设备族缺 SDK,**整个设备管理器就打不开**(这台机器 DCAM SDK 就没装)。
        改成报告而非抛出,选择器里灰着列出来并写明原因。
      * 🔴 **脏状态**:标题 `<file>[*]` + 状态点(灰=没存过/橙=有未写盘的改动/绿=一致),
        **每次投影重算**,不留 flag(留 flag 就会出现「显示已存、实际没存」)。
      * 🔴 **Apply/Init devices** → 按 v2 架构继承成 **Test devices**:把屏幕上的配置真开一遍、
        报告谁答应了、然后放手。v1 是留住设备并把窗口升级成 session;v2 里**设备归 session 所有**,
        窗口再持一份就是两个所有者。真窗口实测:`2 device(s) came up: camera, sequencer`。
      * [?] 🔴 **硬件发现栏(Discovered/Loaded + Snapshot)** —— **阻塞:需要硬件在手**。
        zlc_atom 目前没有总线扫描 API,而没接设备就无法验证扫描结果,写了也只能靠猜。
        与 W6 同一个前置条件。
      * 🟡 域分组:v2 目前只有 5 个类型,分组现在不产生可读性收益,不做。
      * 🟡 `$device:` 交叉引用:**不适用**——v2 没有任何授权 schema 声明设备交叉引用
        (`camera_key` 只出现在虚拟模板里,不是表单字段)。等真出现再说。
      三条统一口径:**先列表**(每个控件/功能一行:v1 怎么做、我怎么做、差在哪、判继承还是改),
      再逐行处理。**只继承最基本的思想,不抄 v1 的坏设计。**

## M4 合并成 Zou_lab_control_v2

- [ ] M4-1 `Github/Zou_lab_control_v2/`:单 git 仓 + 单发行物,内部保持 `zlc_*` 子包结构。
      **复制/整理进去,现有散落的包不删。** 建 git、首个 commit。
- [ ] M4-2 `bin/`:只留 `pulse_editor.bat` + `experiment.bat`(device_manager 配置 → init →
      task_console + pulse editor)。
- [ ] M4-3 notebook tutorial:批判性参考 v1 的 tutorial,覆盖对应 API 调用。
- [ ] M4-4 v2 内跑通全部测试 + 四个入口真开窗验收。

## M5 收尾

- [ ] M5-1 J12 最终全项目审查(含 M 轮全部改动的回归)。
- [ ] M5-2 W6 真机接线 —— **[?] 阻塞**:qCMOS DCAM SDK 未装、真 pylon、真 FPGA 都不在手上。
      按收尾规则 B 跳过,除非你说硬件已就位。
