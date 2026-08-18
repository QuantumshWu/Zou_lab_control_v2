# Step 6-G：全树测试证据架构、examples/notebooks/support 审计

状态：完成（只读审计；没有修改 production、tests、旧文档或硬件）
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：八包全部 `tests/test_*.py`、test support/fixtures/goldens、root pytest/bootstrap、全部 examples 与 notebooks；本报告审查“测试能证明什么”，不重复 02–06d 的包功能审查。
限制：未运行 package/full-tree suite、未 collection-import 全树、未操作真实设备。数字来自当前 checkout 的静态 AST/文本清点；1,346 是 test function definitions，parameterize 后的 pytest collected cases 可以更多。

## 1. 结论先行

当前测试体系数量很多、局部质量也不低，但证据形状明显失衡：

- 166 个 test 文件、1,346 个 test functions、约 55,304 行 test code；
- 直接数值/纯 contract tests 很多，Host、pulse wire、fit Jacobian、archive、selector、Qt interaction 等有实质回归价值；
- 同时有 116 项 shape/text 层测试、220 项访问 private、172 项使用 fake/mock/monkeypatch doubles、304 项位于 module-level offscreen 环境；
- 只有保守归类的 80 项 virtual vertical tests，**0 项正式 real-hardware automated test、0 项 committed CI workflow、0 项 fresh-kernel 全 notebook execution**；
- tests 会读错 checkout、只看旧 output 不执行 notebook、用 fake 复述被测公式、把 source token/doc SHA/public API 数量当行为、以及专门测试无人使用 test fake；这些都已经产生了可点名的 false green。

所以“1,346 tests 全绿”不能作为以下结论的证据：

1. 实验机安装/依赖/启动的一致性；
2. FPGA ABI、FIFO/timing closure、真实 pulse/camera trigger cadence；
3. DCAM/Pylon busy exposure、SLM vendor ABI/readback与真实光学方向；
4. 100-shot SLM feedback 的噪声统计、均匀性提升和总亮度；
5. 真实屏幕的DPR、窗口手感、不卡顿与安全退出；
6. Notebook从空kernel按当前source完整执行；
7. 被tests/public-surface guards保活的代码具有产品存在必要。

总裁决不是“删测试越多越好”，而是把证据重新分层：

| 证据层 | 当前状态 | 裁决 |
|---|---|---|
| pure numeric / immutable contract | 数量多，部分很强 | `KEEP`，减少private implementation lock。 |
| component behavior with real objects | 中等；plot/runtime/pulse较好 | `KEEP + 补 mutation/adversarial`。 |
| seam tests with doubles | 很多；有些double复述production假设 | `REDESIGN`，double只代替外部边，不能代替待证事实。 |
| virtual vertical | 有 Guard A/B/C、synthetic readout、virtual devices | `KEEP`，明确只叫virtual/synthetic evidence。 |
| process/install/notebook | 零散且存在wrong-checkout/stale output | `REDESIGN`为fresh env/fresh kernel。 |
| real-screen / experiment-machine / hardware | 只有手工脚本和历史文字，没有current-HEAD evidence bundle | `MISSING`，必须建立正式验收记录。 |

## 2. 全量静态统计

### 2.1 包级规模

| package | test files | test defs | test physical lines |
|---|---:|---:|---:|
| `zlc_atom` | 34 | 270 | 13,540 |
| `zlc_data` | 9 | 64 | 1,792 |
| `zlc_durable` | 3 | 20 | 385 |
| `zlc_plot` | 53 | 271 | 9,980 |
| `zlc_pulse` | 14 | 139 | 3,626 |
| `zlc_runtime` | 13 | 164 | 7,079 |
| `zlc_ui` | 14 | 93 | 4,707 |
| `zlc_workbench` | 26 | 325 | 14,195 |
| **total** | **166** | **1,346** | **55,304** |

### 2.2 行为层级（互斥、按优先级归类）

这是用于审计的保守 taxonomy，不是 pytest marker：先识别 source/doc/shape；再识别十个明确 virtual vertical 文件；再识别 subprocess/offscreen/cross-component；再识别 private/fake seam；剩余归 direct unit/contract。它用于暴露证据结构，不用于给单个测试自动判生死。

| 层级 | test defs | 比例 | 能证明 / 不能证明 |
|---|---:|---:|---|
| L0 source/doc/shape | 116 | 8.6% | 可守机械import/asset规则；不能证明runtime行为或存在必要。 |
| L1 direct unit/contract | 642 | 47.7% | 对纯函数、数值、immutable DTO最可靠；不能自动外推到线程/设备。 |
| L2 seam/private/double | 213 | 15.8% | 可隔离owner和failure；double若写入待证答案就会false green。 |
| L3 process/UI/component | 295 | 21.9% | 覆盖subprocess、Qt/offscreen和跨component；offscreen不等于human/physical。 |
| L4 virtual vertical | 80 | 5.9% | 覆盖真实产品composition的大段链；只证明simulation model。 |
| L5 real hardware / optical / experiment machine | 0 | 0% | 当前自动suite没有这一层。 |

L4 的保守集合是 Atom 的 `test_real_runtime_integration`、`test_hosted_nodes`、`test_live_plot_accepts_successive_shots`、`test_repeated_runs`、`test_virtual_physics`，以及 Workbench 的 `test_end_to_end`、Guard A/B/C、`test_task_console_app`。其它文件可能含小型纵向链，但未为了抬高数字而算入。

### 2.3 非互斥 evidence markers

每个数字表示“至少一次出现该证据形状的 test function”；同一测试可同时属于多列。`private`是private module/import/attribute访问的下限；`fake`识别fake/mock/stub/dummy/spy/monkeypatch命名，因此没有把所有 `_ConsoleView` 一类手写double都算入，实际double依赖只会更高。`offscreen`表示所在文件设置/声明offscreen并涉及Qt或subprocess，不等于每项都真正点击widget。

| package | private | monkeypatch/patch | source-token | doc-SHA | sleep/qWait | fake/mock lower bound | virtual | offscreen context | subprocess | golden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Atom | 47 | 58 | 8 | 0 | 17 | 94 | 57 | 17 | 2 | 0 |
| Data | 0 | 1 | 2 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| Durable | 1 | 3 | 2 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| Plot | 78 | 14 | 3 | 2 | 5 | 19 | 0 | 7 | 1 | 3 |
| Pulse | 17 | 13 | 11 | 0 | 3 | 25 | 1 | 0 | 1 | 0 |
| Runtime | 19 | 7 | 4 | 2 | 0 | 7 | 0 | 0 | 4 | 0 |
| UI | 0 | 0 | 5 | 0 | 3 | 9 | 3 | 82 | 13 | 0 |
| Workbench | 58 | 13 | 15 | 0 | 17 | 14 | 45 | 198 | 12 | 0 |
| **total** | **220** | **109** | **50** | **4** | **45** | **172** | **106** | **304** | **34** | **3** |
| **of 1,346** | **16.3%** | **8.1%** | **3.7%** | **0.3%** | **3.3%** | **12.8%** | **7.9%** | **22.6%** | **2.5%** | **0.2%** |

`sleep/qWait` 的45项中，多数是有deadline的线程/GUI polling，不能一律删除；但 UI 的多次固定 `QTest.qWait(300)`、virtual physics的真实时长sleep和若干“sleep后立即assert”仍是速度/flake来源。根dev依赖只有 `pytest,jupyter`，没有全局 `pytest-timeout`；一个没有自己timeout的hung worker可拖死整套。

## 3. 已确认的 false-green 分类

### TG-01 — wrong checkout：测试进程可验证另一个安装

06a 已做隔离实证：

- `zlc_data/test_package_guards.py::test_version_and_installation_path...` 的 in-process `zlc_data.__file__`来自本checkout，但 `importlib.metadata.version("zlc-data")`读取机器上另一份旧standalone distribution；两者版本碰巧相同所以绿。
- `test_zlc_data_kernel.py::test_import_is_headless...` 启动裸 `python -c "import zlc_data"`，未注入当前root/src，解析到旧checkout；它证明了旧包headless，不是当前代码。

根 conftest能保护当前pytest进程，却不能自动保护subprocess。当前34项subprocess tests里不少显式设置root/PYTHONPATH并且正确；规则应统一为：每个subprocess先打印/断言关键module `__file__`位于当前repo，或测试刚构建的wheel/venv，绝不能只断言命令返回0。

另一个已确认false green是06e的 `zlc_atom/test_declared_dependencies.py`：硬编码distribution mapping漏掉真实 `zlc_atom -> zlc_workbench` reverse import，测试仍绿。依赖边界必须从唯一product manifest与AST import graph求出，不能用一份漏项表检查另一份表。

### TG-02 — Notebook test声称“executes”，实际只读旧outputs/源码

当前工作目录有7个canonical `usage.ipynb`和5个被Git忽略的自动checkpoint残余；没有任何测试用fresh kernel从第一格完整执行任一本canonical notebook。

- Data `test_usage_notebook...executed_tutorial`只检查已保存 `execution_count/outputs`；06a已实证source与saved output不一致仍绿。
- Workbench `test_the_tutorial_executes_without_error`同样只搜索已保存outputs中的error，不执行source。
- Runtime只查marker字符串；Plot只查notebook中GUI lifecycle文字；UI只查cell调用名字；Atom notebook完全没有test引用。
- Pulse做得最好：会执行offline prefix，并用`FakeRemote`执行最后hardware cell；但中间硬件链没有完整执行。

更直接的反例：当前 `zlc_pulse/notebooks/usage.ipynb` 保存着2个error outputs：

1. `hardware-stop`：`RuntimeError: remote PulseStreamer is not open`；
2. `hardware-direct`：`RemoteBusyError ... current owner=127.0.0.1:58458`。

现有8个pulse notebook tests仍可全绿，因为它们检查cell AST/API文字并绕开这些saved errors。裁决：所有“执行”措辞改为真实fresh-kernel execution；hardware cells拆成明确manual section且saved output必须清空或成功，不允许错误输出当教程历史。

### TG-03 — fake把待证答案写进去，再由production读回来

已确认案例：

- `test_slm_feedback_task::measurement_streams_bounded...` 的fake sequencer直接执行 `camera.trigger(sweeps * 3)`；“3 frames per cycle”正是pulse/camera契约待证事实，fake先写死3再断言得到3，无法发现模板窗口变化或internal repeat（04b）。
- Stepped/Seamless `ScriptedScanBench`按被测代码同样的loop/table手算主动发布恰好数量，能证明consumer ordering，却不能证明真实pulse/source publication cardinality（04a/04c）。
- 大多数SLM feedback controller tests monkeypatch出zero SEM、exact unity vectors；它们证明控制流，却掩盖当前100-shot validation统计门在真实噪声下不可达（05b）。
- Pylon“one trigger one frame”测试预塞frame queue却没有trigger identity；DCAM/Pylon/Virtual tests都不能证明busy sensor是否接受/丢失physical edge（04b）。
- X15213 fake SDK同时定义本项目自己假定的ABI、mode reboot与readback语义；它只能证明Python两边一致，不能证明vendor DLL（05c）。

原则：double可以替代外部设备的**机制边界**，但待证的camera-window count、timing、ABI、noise distribution必须来自独立oracle/trace/real device或明显不同的模型，不能由double重述被测公式。

### TG-04 — tests专门保活dead production/test seams

已在06a/06b/06d确认的 production seams：runtime exact/builder/live-port、plot LivePlotController/FitNumericTable、UI FlowGraph/gallery-only public surface等，都有自己的完整tests/public allow-list；这些tests只能证明实现自洽，不能证明产品需要它们。

test-support内部还有更纯的例子：`zlc_atom/tests/fakes.py::FakeNodeHost`与`FakePulseStreamer`没有产品测试consumer，唯一caller是`test_contract_fakes.py`，即“测试fake的测试”。这两个double又重写Host/streamer lifecycle，随真实contract漂移也不影响产品测试。裁决：`DELETE`两个unused fake及对应self-tests；保留真正被使用、尽量subclass/delegate真实实现的`FakePlane`和ScriptedScanBench（后者重命名证据声明）。

### TG-05 — source-token/prose/doc-SHA把文字形状当架构

50项test直接检查AST/source token；其中import boundary、禁止Qt依赖、overloaded PyQt signal、XDC asset存在等机械约束有价值。但以下形状会false green或阻碍修复：

- `assert "某函数名" in source` 不能证明唯一owner、caller或行为；
- public name count/cap只能锁历史宽度，不能判断存在必要；
- `notebooks.md`要求“每个facade name都必须出现在教程”，反向促使教程替dead API凑数；
- Plot与Runtime各有2项 `test_cross_repo_contract.py`，用SHA钉住两份`fit-numeric-contract.md`，注释甚至要求“edit BOTH copies”；这是非唯一真相源的自动化保护。

裁决：保留AST import/dependency/asset guards，删除prose grep、arbitrary cap、duplicate doc-SHA；真正跨包契约应有一个typed owner和consumer行为test。

### TG-06 — virtual/offscreen测试正确，但测试名/结论外推到physical

04b已经复现：Temperature virtual端到端全绿，同时默认20 ms sensor exposure面对5.02 ms两trigger间隔；VirtualCamera仍交付两frame，真实sensor第一帧integration会覆盖两个probe windows并丢edge。这是“纵向链越长越可信”也会发生的false green：链真实，但plant model漏了关键物理。

类似边界：

- 304项位于offscreen context；它可证明widget state/event wiring，不能证明real-screen DPR、字体、窗口层级、图像手感与不卡顿。
- Pulse FPGA tests守Verilog/source/digest/register memory transport，不跑Vivado synth/timing、bitstream或真板。
- SLM virtual coherent plant对软件因果有价值，但没有真实orientation/LUT/correction/zero order/光学uniformity证据。
- synthetic readout fixture带latent occupancy，是比implementation-equivalence oracle更强的科学测试；它仍只覆盖一套模拟噪声/六site，不是实验机distribution。

测试名与交付报告必须带 `synthetic`、`virtual`、`offscreen` 或 `mocked SDK` 标签，不能简写为“hardware acceptance”。

### TG-07 — private-heavy tests会把重构失败误报成产品回归

至少220项直接访问private module/attribute；Plot 78、Workbench 58、Atom 47最集中。private断言有三种不同价值：

1. 数值算法内部不变量/Jacobian：可保留；
2. 为构造难到达failure的窄seam：可保留但应少；
3. 直接改 presenter/session `_state`、调用private handler、断言容器形状：应替换为public command/observable。

Workbench Pulse Editor的95项细行为、SLM Editor offscreen tests很强，但大量驱动private presenter fields；它们能让内部重构红，却漏真实device command subscription、owner shutdown和04a/05c硬件缺口。应保留少数单元seam，并把关键场景上移到真实button/public handle。

## 4. pytest/bootstrap/CI 架构

### 4.1 root collection的优点

- 一个root `pyproject.toml`列八层testpaths，`--import-mode=importlib`解决重复test basename；
- root先import `zou_lab_control_v2`，使in-process production imports指向当前monorepo；
- 没有默认触碰hardware，适合作为开发机回归。

### 4.2 root conftest污染所有test imports

为复用test doubles，root `conftest.py`把八个`tests/`目录全部插进全局`sys.path`。当前有20条bare `import/from test_*`，分布在14个test files，甚至包括Workbench从Runtime test模块拿private helper。

后果：

- package单跑与root跑的import语义不同；
- 任一新重名helper可让bare import指向另一层；
- test目录可以shadow普通top-level module；
- consumer test依赖另一个test file的private fixture，移动/删除一个test会连锁破坏无关suite。

裁决沿用06e ROOT-005：建立显式`tests_support` package（只放跨层少量factory/double），包内fixture使用相对/明确import；删除八tests目录全局注入。不要把166个test modules变成隐式公共库。

### 4.3 没有committed CI或全局hang边界

仓库没有`.github/workflows`或其它CI配置；README只有人工 `python -m pytest`。`IMPLEMENTATION_PLAN.md`记录很多历史focused/full-tree数字、真窗口描述和不同HEAD，但没有机器可重放的current-HEAD log/artifact/environment manifest，不能替代CI或正式验收记录。

同时根dev extras没有`pytest-timeout`，线程/Qt/socket测试没有全局timeout。多数subprocess和Future局部有timeout是好事，但一个漏掉的deadlock会无限挂suite。裁决：至少建立Windows software CI（fresh env、wheel/install、test path provenance、bounded suite）；hardware不放普通CI，而走受控实验机验收。

## 5. Notebooks：文件、执行证据与裁决

### 5.1 canonical notebooks

| canonical notebook | cells / code | saved executed / outputs / errors | 当前test实际做什么 | 裁决 |
|---|---:|---:|---|---|
| Atom `notebooks/usage.ipynb` | 11 / 7 | 7 / 6 / 0 | **没有任何test引用** | `REDESIGN`：fresh-kernel offline virtual flow；hardware另列manual。 |
| Data `notebooks/usage.ipynb` | 29 / 14 | 14 / 14 / 0 | 只检查saved output、cell短小、每个facade名出现 | `REDESIGN`；06a已证source/output stale。删“教每个export”。 |
| Plot `notebooks/usage.ipynb` | 36 / 18 | 17 / 17 / 0 | 只检查GUI lifecycle文字；NotebookView单元测试另跑 | D-012决定产品面；保留则fresh-kernel执行。 |
| Pulse `notebooks/usage.ipynb` | 49 / 25 | 20 / 20 / **2** | 执行offline prefix；最后hardware cell用FakeRemote；其它主要token检查 | `KEEP CONTENT + REDESIGN EVIDENCE`；先清错误outputs，hardware cells明确manual。 |
| Runtime `notebooks/usage.ipynb` | 31 / 15 | 14 / 14 / 0 | 只查`SignalDataPlane/NodeHost/AcquisitionStream/...`文字 | D-011后重写/删除dead API教程；真正执行保留flow。 |
| UI `notebooks/usage.ipynb` | 30 / 14 | 0 / 0 / 0 | 检查每个cell调用某demo并close；demos在subprocess另测 | `KEEP AS LAUNCH INDEX`或删除，不能称executed tutorial。 |
| Workbench `notebooks/usage.ipynb` | 24 / 11 | 10 / 10 / 0 | 查saved outputs无error、AST有print、步骤文字 | `REDESIGN`；test名声称executes但不执行。legacy convenience见06b。 |

### 5.2 ignored local checkpoints

Atom、Plot、Pulse、UI、Workbench工作目录各有一个`.ipynb_checkpoints/usage-checkpoint.ipynb`，5/5 source hash都与canonical不同。`git ls-files`确认它们均未tracked，`.gitignore`已经排除该目录，因此它们**不是仓库第二truth**，也不影响suite；只属于本机可选清理项，本审计未删除。

### 5.3 notebook目标测试法

1. 每本canonical notebook拆成`offline`与明确标注的`manual hardware`段；
2. CI在fresh kernel、当前wheel/checkout、临时workspace执行全部offline cells；
3. 禁止依赖committed execution_count证明成功；保存output可为教学展示，但CI比较source执行结果而不是相信它；
4. code cell发生error一律失败，除非该格明确教学expected error且test检查异常类型；
5. notebook只教真实产品路径，不承担“让所有public名字各出现一次”的API保活任务；
6. hardware段不在普通CI fake执行后冒充hardware；实验机运行时保存独立evidence bundle。

## 6. Examples：是否真的执行、是否构成产品证据

仓内14个Python examples：

| package / files | 当前自动证据 | 裁决 |
|---|---|---|
| Plot：`_role_data.py`、`camera_live_profile.py`、`live_simulation.py`、`notebook_data.py`、`pyqt5_embed.py`、`static_and_fit.py` | 6/6没有test直接执行 | D-012后处理：profile移tools；正式library则每个至少`--once`/import smoke；内部产品则删standalone dead examples。 |
| Runtime：`demo_signal_flow.py` | `test_acceptance_fixtures`真实subprocess执行`--once`，脚本自己bootstrap monorepo | `KEEP/REWRITE`随D-011；不要继续展示dead AcquisitionStream/exact框架。 |
| UI：`demo_console.py`、`demo_device_manager.py`、`demo_figure_viewer.py`、`demo_pulse_editor.py`、`gallery.py`、`synthetic_card.py` | 这些有subprocess/import/offscreen smoke | `KEEP DEV SMOKE`但不是production consumer；06b判dead Graph/gallery控件不能因此保留。 |
| UI：`capture_acceptance.py` | 没有自动test；入口刻意要求real screen | `MOVE/KEEP AS MANUAL TOOL`；这是正确的不自动offscreen替代，但必须产出current-HEAD证据记录。 |

examples的存在只证明“有人写了示例”；执行smoke只证明它能启动/退出。它不能为无产品caller的framework提供存在理由，也不能替代正式Workbench flow。

## 7. Test support、fixtures 与 goldens

### 7.1 support modules

| support | 消费与风险 | 裁决 |
|---|---|---|
| Atom `tests/__init__.py` | 让`from tests.fakes`成为generic top-level package名；可能与第三方/其它checkout的`tests`冲突 | `REDESIGN`为明确命名的monorepo test-support package或包内相对import。 |
| Atom `tests/fakes.py` | `FakePlane`是真plane instrumentation，广泛使用；`ScriptedScanBench`服务scan；`FakeNodeHost/FakePulseStreamer`只被`test_contract_fakes`自测 | 保留前两者并明确证据边界；删除后两者和self-tests。 |
| Atom `tests/pulse_fixture.py` | 直接resolve真实Calibration template，声明3 windows | `KEEP WITH DEBT`；04b指出duration变化可破坏cadence，不能把fixture常量3当camera contract。 |
| Workbench `tests/pulse_fixtures.py` | 写真实普通pulse workspace | `KEEP`；比手写compiled replacement强。 |
| Plot `tests/data_factory.py` | 38/53 test files使用；返回真实zlc_data对象，但增加兼容subclass/default roles/old ergonomic API | `KEEP UNIT FACTORY + LIMIT AUTHORITY`；关键cross-package cases必须消费真实Atom publications，不能只靠factory。 |
| Runtime `tests/_snapshots.py` | 共享真实OwnedSnapshot builder | `KEEP`，随精简runtime更新。 |
| Plot/UI `tests/conftest.py` | marker/Qt setup | `KEEP`；root path污染另修。 |

`data_factory.py`集中构造比38份手写schema好，但它也是一层normalizer：默认roles、scalar尾轴、identity命名可能把真实producer差异抹平。02/04发现的camera role/overlay问题说明unit factory green必须由少量真实producer→plot contract tests补强。

### 7.2 fixtures/goldens

| artifact | 价值 | 缺口/裁决 |
|---|---|---|
| Plot `fixtures/fit_anchors.json`（约42 KiB） | 8个model冻结曲线，mutation能使公式变化变红；优于用同一evaluator自产observations | `KEEP`；补生成来源/公式版本说明，不能替真实实验分布。 |
| Plot `goldens/{curve,histogram,image,facet_fit}.png` | 固定Matplotlib版下守RGBA几何/style；max delta + edge population容差合理 | `KEEP`；只证明像素稳定，不证明plot语义正确或GUI性能。 |
| Atom `fixtures/main_readout_oracle.npz/json`（约1.8 MiB） | synthetic frames + latent occupancy，按truth而非旧implementation judged；当前readout agreement与mutation guards价值高 | `KEEP AS SYNTHETIC TRUTH`；manifest缺generator commit/seed/noise model provenance，且6-site single distribution不能替实验数据。 |
| Atom `fixtures/hand_examples.json` | normal CDF与3x3 box独立手算 | `KEEP SUPPLEMENTAL`，名称已如实，不冒充main oracle。 |

仓内没有对应的真实camera timestamp/trigger trace、FPGA timing/ABI capture、SLM USB/DVI vendor readback、100-shot per-site raw statistics或真实screen capture manifest作为test fixture/evidence。不能把synthetic NPZ和golden PNG改名为“hardware oracle”填空。

## 8. 哪些正式证据当前缺失

### 8.1 software delivery evidence

- fresh Windows environment从root wheel/install启动四个console scripts；
- 所有关键subprocess断言module path/current HEAD，不回落已安装旧包；
- current HEAD的一次bounded full software suite log；
- fresh-kernel offline notebooks；
- close后无Python子进程、Qt top-level、executor/thread、device claim残留；
- no CI workflow意味着这些目前都依赖人工记忆/IMPLEMENTATION_PLAN文字。

### 8.2 real-screen evidence

现有`capture_acceptance`正确拒绝offscreen，但仓内没有与当前HEAD绑定的正式capture index。至少应记录：HEAD、OS/display/DPR/font、app/flow、logical/physical size、截图文件digest、关键human observations、close后资源状态。截图不是自动test替代，而是offscreen无法覆盖的证据层。

### 8.3 experiment-machine/hardware evidence

建议按设备独立记录，不建一个万能“hardware pytest”套件：

| domain | 最小原始证据 | 必须判定的门 |
|---|---|---|
| Pulse/FPGA | board identity、RTL/bitstream/build id、clock、geometry/layout ABI、program/table sizes、status/cursor/timing trace | FIFO容量、clock/frac/ABI一致、repeat/sweep cardinality、SAFE/LOAD/FIRE/underflow。 |
| Camera | adapter/model/firmware、working point、exposure/trigger timestamps、raw frame ordinals、drop/timeout records | 每edge是否接受、window grouping、busy behavior、cancel/finish、same-shot lineage。 |
| SLM USB/DVI | serial/profile/wavelength、vendor SDK/DLL version、correction map、command/readback bytes、reboot/mode state | orientation/LUT/correction sign/offset、exact readback、controller drive、settle。 |
| Optical/feedback | incoming/each candidate phase、每shot/site raw counts、dark subtraction、SEM/CI、accepted/rejected decision、total brightness | uniformity真实改善、噪声鲁棒性、validation可达、rollback/Stop、brightness不被牺牲。 |
| Full experiment flow | apparatus/pulse/artifact hashes、operator steps、UI screenshots、saved figure/archive、shutdown state | 同一HEAD从start到monitor/save/reopen/stop/close，无手工隐式修补。 |

每个evidence bundle应自带：Git HEAD、dirty status、environment/package paths、device IDs/versions、authored inputs、raw outputs、判定代码版本与pass/fail，不只保存一句“实验机已验收”。

## 9. 八包逐 test-file 裁决

以下每个`test_*.py`恰好归入一行；详细功能缺陷引用06a/06b/06d、04a–c、05a–c，不在此重写。

### 9.1 `zlc_data`（9）/ `zlc_durable`（3）

| package / verdict | files | 理由 |
|---|---|---|
| Data `KEEP` | `test_zlc_data_io.py`, `test_zlc_data_snapshot_builder.py`, `test_zlc_data_validation.py`, `test_zlc_data_validity.py` | direct schema/codec/validation behavior；见06a缺口扩展。 |
| Data `KEEP + REDESIGN` | `test_zlc_data_kernel.py`, `test_zlc_data_selection.py`, `test_zlc_data_snapshot_projection.py` | kernel裸subprocess错仓；selection覆盖窄；projection文件名与真实cutter覆盖不符。 |
| Data `REDESIGN` | `test_package_guards.py`, `test_usage_notebook.py` | metadata读旧distribution、三份allow-list/cap、notebook不执行且凑export。 |
| Durable `KEEP` | `test_durability.py` | fsync/replace/cleanup/Windows handle有真实价值；补post-replace失败/并发。 |
| Durable `REDESIGN` | `test_workspace.py` | 只证单线程命名，不能支撑“never overwrite”；见06a。 |
| Durable `KEEP MECHANICAL / PRUNE SHAPE` | `test_package_guards.py` | stdlib/import边界保留；prose/source唯一owner与public cap删除。 |

### 9.2 `zlc_runtime`（13）

| verdict | files | 理由 |
|---|---|---|
| `KEEP` | `test_generation_lifecycle.py`, `test_host.py`, `test_presentation.py`, `test_selection_bridge.py`, `test_signal_front.py`, `test_signal_plane.py` | 当前产品Host/plane/front/presentation/selection核心行为。按03c/06d补generation declaration、identity与terminal completeness。 |
| `KEEP SUBSET / PRUNE` | `test_runtime_streams.py`, `test_runtime_helpers.py`, `test_import_guards.py` | 只迁移EventRef/follow、精简mailbox/declaration、真实import/dependency/no-thread；删exact/live-port/MAX/public历史形状。 |
| `DELETE WITH DEAD FRAMEWORK` | `test_cleanup.py`, `test_runtime_dataset_builder.py` | 只服务06d已判dead cleanup/exact builder/monitor。 |
| `DELETE/REPLACE` | `test_cross_repo_contract.py`, `test_acceptance_fixtures.py` | duplicate doc SHA；notebook marker与dead framework demo不构成acceptance。保留demo时改真实behavior smoke。 |

### 9.3 `zlc_plot`（53）

| verdict | files | 理由 |
|---|---|---|
| `KEEP CORE` | `test_aggregate_by_codes.py`, `test_bimodal_collapse.py`, `test_camera_cycle_image_pooling.py`, `test_compose_identity.py`, `test_embed_semantic_resilience.py`, `test_facet_auto_semantics.py`, `test_facet_cell_ticks.py`, `test_facet_cell_title_fit.py`, `test_facet_dense_equivalence.py`, `test_facet_focus_compose.py`, `test_facet_focus_image_parity.py`, `test_fit_contract_k.py`, `test_fit_engine.py`, `test_fit_headline.py`, `test_fit_jacobian.py`, `test_fit_projection.py`, `test_fit_warm_start.py`, `test_gesture_layer.py`, `test_histogram_samples.py`, `test_image_fit_geometry.py`, `test_label_carry.py`, `test_layout.py`, `test_no_data_colour.py`, `test_npz_io.py`, `test_plot_session_golden.py`, `test_projection_coverage.py`, `test_replace_spec_transaction.py`, `test_rolling_shot_axis.py`, `test_selection_subject.py`, `test_selectors.py`, `test_semantic_feasibility.py`, `test_semantic_spec_authority.py`, `test_semantic_ui.py`, `test_semantics.py`, `test_tick_labels.py`, `test_units.py`, `test_validate_implies_build.py` | direct numerical/projection/layout/semantic/render behavior；private numeric/Jacobian seams可保留。 |
| `KEEP + REDESIGN CLAIM/BOUNDARY` | `test_backends.py`, `test_data_contract.py`, `test_facet_live_fit.py`, `test_kind_registry.py`, `test_live_protocol.py`, `test_namespace_isolation.py`, `test_performance_guards.py`, `test_public_api.py`, `test_public_surface.py`, `test_qt_widget.py`, `test_raster_host.py` | Qt/live atomic/facade/product边界见02/06d；保留真实behavior，删cap/test-only names并补bound controls/threads/full pipeline。 |
| `USER DECISION D-012` | `test_notebook_raster.py` | 保留Notebook产品才修single owner并fresh-kernel；否则随Notebook删除。当前还直接把host-attached session mutation当合法回归。 |
| `DELETE CURRENT PRODUCT` | `test_cross_repo_contract.py`, `test_fit_numeric_table.py`, `test_live_channel.py`, `test_live_controller.py` | duplicate doc SHA、test-only table、无product caller的第二live pipeline。若D-012选独立library，后两者须升级真实acceptance而非fake自测。 |

### 9.4 `zlc_pulse`（14）

| verdict | files | 理由 |
|---|---|---|
| `KEEP` | `test_command_strobe.py`, `test_fpga_assets.py`, `test_import_purity.py`, `test_launcher.py`, `test_manifest.py`, `test_model_compile.py`, `test_remote.py`, `test_scan_model.py`, `test_transport.py`, `test_uart_transport.py`, `test_wire_device.py` | compile/wire/strobe/transport/RPC/asset基础强；04a列出的ABI/clock/FIFO/sweeps/delay缺口仍需补，不能外推真板。 |
| `KEEP INTENT + REDESIGN` | `test_contract.py`, `test_public_surface.py` | signature/import surface守卫可留最小；不得锁错误repeat术语、dead write_slots或历史宽度。 |
| `REDESIGN` | `test_notebook_coverage.py` | offline execution有价值；token/API coverage保活dead API，且未拒绝当前2个saved hardware errors。 |

### 9.5 `zlc_atom`（34）

| verdict | files | 理由 |
|---|---|---|
| `KEEP DIRECT/SYNTHETIC` | `test_calibration_saved_frames.py`, `test_camera_and_execution.py`, `test_derivation_boundary.py`, `test_device_configuration.py`, `test_execution_safety.py`, `test_hosted_nodes.py`, `test_installation_and_nodes.py`, `test_installation_config.py`, `test_installation_guards.py`, `test_live_plot_accepts_successive_shots.py`, `test_monitor_and_installation.py`, `test_mutation_guards.py`, `test_photoelectron_units.py`, `test_physics.py`, `test_readout_against_known_truth.py`, `test_real_runtime_integration.py`, `test_repeated_runs.py`, `test_scan_plan.py`, `test_sequencer_contract.py`, `test_site_detection.py` | 真实objects、synthetic latent truth、terminal/lineage/device composition有价值；名字必须保留synthetic/virtual边界。 |
| `KEEP + REDESIGN CLAIM/ADVERSARY` | `test_dcam_camera_adapter.py`, `test_imaging_template_cadence.py`, `test_pylon_camera.py`, `test_seamless_scan_node.py`, `test_stepped_scan_node.py`, `test_temperature_chain.py`, `test_virtual_physics.py` | camera busy/cadence、free-run/extra/missing publication、burst simulation缺口见04a–c；Temperature不能再叫hardware acceptance。 |
| `KEEP + ADD PHYSICAL/STATISTICAL EVIDENCE` | `test_slm_editor.py`, `test_slm_feedback_task.py`, `test_slm_x15213.py` | software/UI/bytes/controller路径强；05a–c指出private/offscreen、zero-SEM fake、vendor ABI/reboot/readback/optics均未证。 |
| `KEEP MECHANICAL / REDESIGN SHAPE` | `test_import_boundaries.py`, `test_v3_architecture.py` | AST boundary与descriptor exercise保留；source token、wrapper形状、hardcoded catalogs不作产品truth。 |
| `REDESIGN FALSE-GREEN` | `test_declared_dependencies.py` | 06e已证漏真实reverse import仍绿；从manifest/import graph重建。 |
| `DELETE TEST-OF-TEST` | `test_contract_fakes.py` | 主要只验证unused FakeNodeHost/FakePulseStreamer及fake signature；不形成产品证据。 |

### 9.6 `zlc_ui`（14）

| verdict | files | 理由 |
|---|---|---|
| `KEEP` | `test_console_views.py`, `test_device_manager_view.py`, `test_figure_viewer.py`, `test_import_purity.py`, `test_modal_repaints_what_is_there.py`, `test_overloaded_signals.py`, `test_panel_card_plot_interaction.py`, `test_pulse_views.py`, `test_qt_app_single_entry.py`, `test_settings_layout.py` | Qt/QTest/layout/retirement/import行为真实；06b列出的defaults、hung worker、hardware shutdown、enabled graph与real-screen缺口仍需补。 |
| `KEEP DEV SMOKE / NOT PRODUCT PROOF` | `test_console_extension_cost.py`, `test_gallery.py` | example/gallery能启动有用，不为dead Graph/FormGrid提供存在理由。 |
| `PRUNE` | `test_controls_smoke.py` | 当前保护FlowGraph和未消费QtOwnerWake，反而掩盖production duplicate；保留真实control smoke。 |
| `REDESIGN FACADE` | `test_public_surface.py` | explicit allowlist/import purity留；删Graph/tool-only/API cap与notebook凑名。 |

### 9.7 `zlc_workbench`（26）

| verdict | files | 理由 |
|---|---|---|
| `KEEP` | `test_archive.py`, `test_auto_panel_kind.py`, `test_console_logic.py`, `test_console_presenter.py`, `test_device_manager.py`, `test_editor_named_behaviours.py`, `test_environment.py`, `test_guard_a_virtual_chain.py`, `test_guard_b_task_console_interaction.py`, `test_guard_c_save_semantics.py`, `test_panel_front_coherence.py`, `test_panel_spec.py`, `test_presentation.py`, `test_same_shot_presentation.py`, `test_selection.py`, `test_task_console_app.py`, `test_topology.py`, `test_windows.py` | 产品composition/Qt buttons/same-shot/save/lifecycle有价值；fake/private/offscreen/physical边界须按02–05与06b改名补证。 |
| `KEEP + REPLACE LEGACY/PRIVATE SEAM` | `test_end_to_end.py`, `test_view_contracts.py`, `test_viewer.py`, `test_pulse_editor.py` | legacy save fixture、test-module doubles、private presenter很多；保留场景，换formal Panel Save/public handles/real owner。 |
| `KEEP MECHANICAL` | `test_gui_seam.py`, `test_launcher_imports.py`, `test_launchers.py` | AST/launcher/CRLF有用；不证明domain object边界、argv fidelity、cleanup或实验机启动。 |
| `REDESIGN/USER DECISION` | `test_notebook.py` | 只读saved outputs/source且保护legacy conveniences；06b notebook产品边界裁决后fresh执行或删除。 |

## 10. 目标测试证据架构

### 10.1 不以数量、coverage或“全绿”作为单一门

每个重要产品不变量至少回答四个问题：

1. **谁拥有规则？** unit test在owner层直接验证；
2. **跨层时类型/identity有没有保持？** component test用真实相邻实现；
3. **真实composition能否走完？** virtual/offscreen vertical test；
4. **simulation省略的physical事实如何验？** real-screen或experiment-machine evidence。

同一项不需要四层都堆十个测试；关键是不能拿L1 fake test替L5 physical gate。测试数量下降但证据层补齐，比继续增加第1,347个同层test更可靠。

### 10.2 推荐 lanes

当前只有6个Plot tests标`@pytest.mark.gui`，0个hardware/integration/slow/acceptance/performance marker；其余数百Qt/vertical tests无法按证据层选择。推荐最小lanes：

| lane | 内容 | 默认何时运行 |
|---|---|---|
| `software` | pure/unit/component、无screen无hardware | 每次变更；fresh current checkout/wheel。 |
| `gui_offscreen` | Qt/QTest/raster、bounded subprocess | Windows CI；明确不是human acceptance。 |
| `virtual_vertical` | Guard A/B/C、synthetic readout、virtual camera/pulse/SLM | 相关变更 + release gate。 |
| `notebook_offline` | 7本选定产品notebook的offline cells fresh kernel | notebook/API变更 + release gate。 |
| `real_screen` | capture tool + human observations | UI release/重要交互变更；保存evidence bundle。 |
| `hardware` | device-specific runbooks | 对应device/pulse/science变更；受控实验机执行。 |

不建议建立复杂test manager：用pytest markers、少量明确scripts和一个evidence目录/manifest足够。hardware lane可以调用pytest节点，但原始设备trace与环境信息必须同时保存。

### 10.3 doubles规则

- double只实现被替代外部边的最小protocol，不重新实现NodeHost/scan/pulse算法；
- 待证值（window count、sweeps、timestamps、noise）从独立fixture/trace输入，不在fake中用同一公式算；
- 每个shared double必须有至少一个真实product consumer test；禁止test-of-test；
- 相邻包可用real in-memory implementation时优先用real，例如FakePlane继承SignalDataPlane；
- double contract改变时由typed protocol/consumer test报错，不另建signature digest表；
- synthetic plant必须列出明确省略的物理行为，并给相应hardware gate链接。

### 10.4 provenance规则

所有process/install/evidence tests先确认：

- module `__file__`、distribution root来自当前checkout/本次wheel；
- Git HEAD与dirty state；
- Python/OS/关键dependency版本；
- subprocess的cwd/PYTHONPATH由统一helper形成，不能各文件手写不同规则；
- fixture/golden的来源、seed、generator version或独立authority；
- result log与artifact绑定同一HEAD，旧IMPLEMENTATION_PLAN数字不能冒充current gate。

### 10.5 performance与concurrency

现有sleep/polling tests应逐步改为event/latch驱动；保留用真实deadline验证timeout/cadence的少数测试。对线程/Qt/socket建立suite级hang上限与close resource assertions。性能tests分两类：

- micro benchmark只守算法复杂度/固定budget；
- product profile守多panel+fit+overlay、100-shot feedback、save/archive等真实链。

不要在共享CI用过紧wall-time阈值；保存分阶段计时与输入规模，让回归能定位，而不是只得到“慢于1秒”。

## 11. 需要用户裁决

### U-06G-01 — 哪些notebooks/examples是正式产品

与D-012/06b notebook convenience联动。推荐：只保留当前操作者真正使用的Workbench/Atom/Pulse教程；Data作为底层开发教程可留；Plot standalone Notebook由D-012决定；Runtime在D-011删框架后重写；UI launch index可由README替代。正式保留的必须进入fresh-kernel lane，其余删除而不是继续shape-test保活。

### U-06G-02 — software CI落在哪里

仓库当前没有CI。推荐至少一个fresh Windows runner，因为PyQt5、batch launchers、directory flush和实验软件都以Windows为主；可用托管runner或用户控制的干净实验软件runner。若代码不能上传外部CI，需明确一个本地release command与不可编辑的result artifact owner。

### U-06G-03 — 实验机验收责任与频率

需要用户决定谁可操作FPGA/camera/SLM、哪些改动必须重验、evidence保存在哪里。推荐按domain gate而非每次全实验：pulse/wire改动验FPGA+camera cadence；camera adapter改动验对应相机；SLM profile/solver/feedback改动验device+optics+100-shot统计；UI-only改动走real-screen。

### U-06G-04 — 仓外public API/standalone distributions

如果没有外部users，删除API cap、doc-SHA、every-export notebook guards并按monorepo consumers收窄；如果确有外部脚本，用户需提供真实consumer样本/兼容名单，随后测试这些行为。当前手抄allow-list不能替代这个决定。

### U-06G-05 — golden/oracle更新政策

推荐：golden更新必须附“为什么产品结果应改变”的独立证据；禁止同一被测implementation自动重生expected后直接提交。Synthetic fixture可由独立generator产生，但manifest必须记录seed/model/version；实验fixture不得含敏感内容时至少保存脱敏raw statistics与采集provenance。

## 12. 优先级与最终清册

### P0 — 先消灭会说谎的绿灯

1. 统一subprocess provenance，修Data两项wrong-checkout与Atom dependency false-green；
2. Pulse notebook的2个error outputs不得继续绿；把“executes”tests改fresh-kernel或如实改名；
3. 删除duplicate doc-SHA、arbitrary API caps、unused FakeNodeHost/FakePulseStreamer self-tests；
4. 为current software gate建立Windows fresh-env lane与全局hang边界；
5. 所有报告区分unit/synthetic/virtual/offscreen/real-screen/hardware。

### P1 — 补关键纵向与physical evidence

1. pulse internal repeat/table sweeps → camera accepted-edge/timestamp/cardinality trace；
2. 100-shot SLM feedback raw per-site statistics、validation/rollback/brightness；
3. X15213 vendor ABI/readback/reboot/orientation/LUT/correction/settle；
4. real-screen TaskConsole/FigureViewer/Pulse/SLM flows与owner-thread不卡顿；
5. formal Panel Save→FigureViewer round-trip、fresh notebooks、wheel/install launch。

### P2 — 降维护噪声

- public handles替换private presenter/session mutation；
- events/latches替代固定sleep/qWait；
- `tests_support`替代14文件的bare sibling imports/root path injection；
- 保持checkpoint忽略规则；按需清理本机残余，并删除dead examples/dead-framework tests；
- 将prose/source scans收敛为少量AST semantic guards。

### PASS / KEEP

- pure numeric/Jacobian/validation、wire/strobe/transport、archive/durability、Host/plane/front/presentation、selector/semantic/layout/ticks；
- synthetic latent-truth readout与mutation guards；
- Guard A/B/C与真实public-button virtual flows，但名称保留virtual；
- offscreen Qt interaction/retirement；
- fit anchors和RGBA goldens（按更新政策）；
- 真正被消费、委托real implementation的test factories/doubles。

### REDESIGN

- wrong-checkout subprocess/metadata、hardcoded dependency mapping；
- notebook saved-output/token tests；
- fake复述window/sweep/noise/ABI的tests；
- private-heavy presenter/session tests；
- source-token/public-cap/every-export tutorial；
- root tests目录注入与bare test imports；
- offscreen/virtual tests的过度命名；
- 无CI、无global timeout、无lane markers、无current evidence artifact。

### DELETE（在对应产品裁决后）

- `.ipynb_checkpoints`无需代码删除：当前5个均为ignored本机残余；
- Plot/Runtime duplicate `test_cross_repo_contract.py`；
- Atom `test_contract_fakes.py`与unused test doubles；
- runtime exact/builder/live-port、plot test-only live/table、UI dead Graph等实现的self-tests；
- arbitrary public-size assertions与只grep prose的tests；
- D-011/D-012/06b裁决不保留的standalone examples/notebooks及其shape tests。

## 13. 完成声明

- 已静态覆盖166/166 test files、1,346/1,346 test function definitions及全部test support/fixtures/goldens。
- 已覆盖14/14 Python examples、7/7 canonical notebooks，并核实5/5 local checkpoints均为Git ignored残余。
- 没有运行大套件、collection import、GUI/hardware；没有修改production、tests、旧文档或硬件。
- 本报告只新增本文件；所有测试数字均说明了静态口径，没有把历史passed count或saved notebook output冒充current execution。
