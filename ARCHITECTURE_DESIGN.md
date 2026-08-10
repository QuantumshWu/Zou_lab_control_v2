# Zou_lab_control v2 目标架构设计（执行权威）

> 仓库绝对路径：`C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2`
> 本文绝对路径：`C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\ARCHITECTURE_DESIGN.md`
> 凡本文或用户明确要求“参考 v1”时，唯一允许读取的 v1 树是 `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v1_claude\Zou_lab_control_v1`。`ZLC_main`、`_reference\Zou_lab_control_v1` 和其他副本都不得代替它。这个路径裁决不把 v1 升格为产品规格；v1 只作为用户逐项点名的 Device Manager、TaskConsole/运行中 Task 操作面、Calibration report 和 virtual apparatus 默认值的行为参考，v2 package boundary 和唯一真相源仍只按用户裁决和本文实施。
> 设计基线：`0243aa6`。完成状态与实测证据以 `IMPLEMENTATION_PLAN.md` Checkpoint 为准；package/full-tree 测试不能替代真实实验入口和可见 GUI 的验收。
> 权威顺序：用户当前裁决 > 简单且可维护的边界 > v2 当前实现 > v1 参考。v1 不是规格。
> 根目录 `HANDOFF.md` 只是指向这两份权威文档的 historical/inactive pointer，不保存产品快照、Checkpoint 或验收结论，也不需要在上下文恢复时重读。

### 封闭的目标权威集合

Goal 启动后，仅下列两份磁盘文档定义目标和实施方向：

1. `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\ARCHITECTURE_DESIGN.md`
2. `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\IMPLEMENTATION_PLAN.md`

当前树的源代码、git diff 和实测结果是“现状证据”，不是“目标规格”。下列文件都是历史资料或旧 package 说明，可用于定位现有 API/过期行为，但不得作为 Goal，不得覆盖或重新解释上述两份权威文档。七份 `packages/*/GOAL.md` 已收缩为只指向这两份权威的 historical/inactive tombstone，不再保留 active 正文或另一份待办：

- 根目录 `HANDOFF.md` 和 `README.md`；
- `packages/*/GOAL.md`（当前存在 7 份）；
- `packages/*/docs/contract.md`；
- `packages/*/README.md` 及其他历史 design/goal 文档。

若这些文件与权威文档冲突，直接忽略冲突的旧说法并继续实施。若两份权威文档彼此冲突，执行者按本文末尾的自主决策顺序选择最优实现，同步修正两份文档和 Checkpoint，不停下询问。

## 1. 目标

首先跑通同一条真实产品路径，虚拟设备和真设备只在 adapter 底部分叉：

```text
Calibration Task
    -> repeat: camera measurement -> progress + current image -> Monitor panel
    -> finish loop: calculate SiteMap + box/psf/uniform_psf models once
    -> save that result as calibration JSON
    -> pass that same result to zlc_plot: site map + fidelity + three classifier grids + PSF kernels
Camera Measurement
    -> frames signal
Occupancy Processor(frames + calibration path + readout-model choice)
    -> occupancy data
Plot Panel
    -> Panel Edit Save Fig(image + data)
```

同时保证 TaskConsole 的 Add -> Edit -> Start/Restart、selector 反写 measurement draft、两种 header save 都符合产品语义。整个产品 UI 不提供 `Apply` 按钮。

## 2. 核心设计原则

1. 一个事实只有一个所有者。科学数据归 `zlc_data`，运行和 signal plane 归 `zlc_runtime`，中性原实验语义归 `zlc_atom`，绘图语义归 `zlc_plot`，widget 归 `zlc_ui`，跨层接线归 `zlc_workbench`。
2. UI 只提交意图和 draft，不直接读写硬件。阻塞 camera/sequencer 调用在 worker/session 侧执行。
3. exposure、ROI 是 Camera Measurement 的参数。UI 接线层负责表单同步和 selector 反写；Start/Restart 时 measurement request 将参数传给用户选中的 camera adapter。
4. 不增加 fingerprint、SHA/hash、防篡改协议、想象出的兼容层，或大量防御型包装。
5. latest buffer 只保存当前值，不记录、展示或存档丢了多少/哪些。
6. 测试守卫尽量修改现有测试。只增加少量能在原缺陷下变红的纵向行为测试。
7. 保持稳定的只有整体骨架：plugin discovery、descriptor/contract、NodeHost 生命周期、session/device ownership 和公共 signal/plot 能力。每个 Logic Node、device plugin 以及 Workbench 功能都在该骨架内用最短实现完成自己的业务；Workbench 只做基本逻辑和接线。单个 plugin 的需求不得升级成新的通用 registry/coordinator/transaction/DTO/adapter 层，现有单消费者框架应直接删除。
8. `zlc_atom` foundation 与 concrete plugin 的依赖边界必须分开：顶层基础模块、公共 contract、install/runtime glue 和 `nodes/_framework` 保持 headless，不依赖 Qt、`zlc_plot` 或 `zlc_ui`；具体 `nodes/<plugin>`、`devices/<plugin>` 可以在自己的目录内声明并实现本插件独有的 plot/UI，并调用 `zlc_plot`/`zlc_ui` 公共 API。该局部能力不得反向进入 foundation，也不得被 Workbench 收编成通用 plugin-specific 框架。

## 3. Experiment、GUI 和设备所有权

Notebook 创建的一个 `Experiment`/session 天然是共享底层。TaskConsole 和 Pulse Editor 都从它取同一组 named devices、同一个 virtual world 和同一个 sequencer。不新造第二个 session、IPC 或“两个 GUI 抢着 open 设备”问题。

### 3.0 正式实验入口与 Device Manager

1. 根目录 `bin\experiment.bat` 就是正式实验入口，等价于权威 v1 树中的 `task_console.bat`：它只启动一个统一的 TaskConsole experiment flow，不能串起多个 Python 进程，也不再增加第二个根入口。
2. 同一 Qt 进程先显示用户指定的 v1 Device Manager 操作面：`Config` tab 与 `Devices` header；左侧按 device domain 动态生成 Camera/Rf/Sequencer 等分组，每组包含具体 device cards 与 `Add device`；右侧是 `Discovered hardware`/`Scan hardware` 和 `Loaded session`；底部是 `New…`、`Load…`、`Save`、`Save as…`、`Init devices`。不得显示错误的 `Installation`、`Backend`、`Configured devices`、`Available`、`Cancel` 结构，也不得把 v2 内部 `InstallationConfig` 名称泄漏成用户界面。
3. `Init devices` 只从当前 device draft 创建并持有唯一 `ExperimentSession`；成功后隐藏 Device Manager，并在同一进程中同时显示 TaskConsole 和 Pulse UI。设备初始化不解析、不编译、不预载任何 pulse，也不以 `calibration`/`imaging_template` 是否存在作为成功条件。
4. 实验入口中的 Pulse UI 借用 session 的 sequencer authority；它不自行 dial 第二个设备，也不拥有或关闭底层 sequencer。Pulse UI 的当前/初始文档是自己的 editor 状态，不是 Device Manager 的初始化前置条件，也不是 Calibration Task 的隐式 pulse。独立 `pulse_editor.bat` 只作为单独的脉冲编辑/连接测试工具，不是正式 Experiment 入口。
5. 主 TaskConsole 关闭时按 Pulse controller -> TaskConsole nodes/workers 与 plot bindings -> session/devices -> Device Manager owner 的顺序有界清理；任何 ownership 尚未释放都不得伪装成成功退出。

### 3.1 两种访问状态

| 访问 | 语义 | 并发规则 |
|---|---|---|
| `OBSERVE` | 只读状态、snapshot、序列状态 | 可有多个 observer，也可与一个 exclusive owner 共存 |
| `EXCLUSIVE` | 驱动/配置/采集该底层设备 | 同一时刻只有一个正在运行的 Logic Node owner |

启动 Logic Node 的正常顺序是：

1. 先完整 validate/build 新 request；无效参数不停旧 node。
2. 查找与新 request 的 `EXCLUSIVE` claim 指向同一底层实例的运行中节点。
3. 只停止这些冲突节点，等它们退出并释放设备。
4. 启动新 node。不冲突的 node 和所有 observer 不受影响。

例如，启动 Calibration 会停止正在独占同一 camera 的 Camera Measurement；不会因为有窗口在只读显示 camera snapshot 而拒绝启动。

### 3.2 Sequencer 的特判

- Pulse Editor 是同一 session 上的 editor/controller，不登记为一个长期占用 sequencer 的 Logic Node。编辑和只读查看更不占用。
- Camera Measurement 独占 camera；它若需要 sequencer 状态，只以 `OBSERVE` 访问，不 prepare/fire pulse。
- Calibration 或真正执行 scan points 的 Logic Node 在运行期间可以 `EXCLUSIVE` 占用 sequencer。
- Pulse Editor 的普通 prepare/fire 使用同一 session 中的 sequencer 实例，不引入另一套 device owner 机制。当有长时间 exclusive scan/task 时，UI 按 session 当前状态禁用或拒绝冲突的驱动命令即可。

### 3.3 Simulation 设备与共同契约

1. 所有 virtual apparatus 实现只位于 `packages/zlc_atom/src/zlc_atom/devices/simulation/`：`camera.py`、`sequencer.py`、`world.py` 和 `device_types.py`。真相机/真 sequencer 目录只保留共同契约、binding 和硬件 leaf，不再寄存 virtual 类或 SimulationWorld。
2. Camera 的共同契约是 runtime-checkable `CameraAdapter` Protocol，不虚构另一个 `BaseCamera`。`VirtualCamera`、DCAM 和 Pylon 都通过同一 adapter/binding 契约，安装时校验该 Protocol。Sequencer 的共同契约是 nominal `SequencerDevice`，`VirtualSequencer` 必须继承它。
3. Virtual camera/sequencer 与硬件一样由 `DeviceTypeDescriptor -> InstalledLeaf -> binding` 组成，Logic Node 只按 capability 取设备，不写 `if virtual` 分支。
4. `SimulationWorld` 是 virtual 成像物理、site geometry、seed 和 trigger routing 的唯一所有者。默认 virtual apparatus 是 `5 x 7 = 35` sites 和 `96 x 128` image；这是模拟装置的可测真值，不能倒流成 Calibration request 的 grid/count 输入。
5. Virtual sequencer 的 finite `fire -> wait_done` 必须尊重 compiled pulse 的 logical duration；不能因为 memory transport 已立即给出 DONE 就把几十到几百次采集压进一个 Monitor refresh interval。`forever` 与 finite 都复用同一个 compiled duration，只是前者持续按 cadence 触发、后者到 logical terminal 才交付一次完成报告。

## 4. Package 责任边界

`zlc_workbench` 是 composition root，但不拥有科学或绘图规则。

| Package | 拥有 | 不拥有 |
|---|---|---|
| `zlc_data` | axis/schema/validity，immutable `OwnedSnapshot`，科学数据 NPZ 表达 | device、queue、GUI |
| `zlc_durable` | workspace path、唯一文件名、原子写 | 科学算法、hash/fingerprint |
| `zlc_runtime` | stream/tap/dataset builder、signal publication、node lifecycle、worker | camera/ROI/exposure、Qt、plot kind |
| `zlc_plot` | plot spec/projection/display/fit/selector/renderer/image export | device、TaskConsole layout、measurement form |
| `zlc_ui` | window/tab/form/widget、operator intent | device 访问、物理规则、runtime 调度 |
| `zlc_pulse` | pulse model/compiler/slot/scan、sequencer transport | measurement、TaskConsole 管线 |
| `zlc_atom` | foundation 拥有 device capability/adapter、logic contract/host-facing descriptor；具体 plugin 目录拥有自身物理、专有 plot/UI 声明与实现。Calibration Task 在自己的 plugin 目录调用公开 `zlc_plot` API 生成业务 report | foundation 不拥有 Qt/plot/UI；不拥有 TaskConsole panel/layout/archive 或 renderer 实现；plugin-local UI 不得反向进入 foundation |
| `zlc_workbench` | Experiment 接线、row draft、资源仲裁、Start/Restart、panel-producer 联动、save 组装 | 相机 SDK 分支、site 检测算法、plot 参数合法性 |

## 5. Logic 模型：role 与数据 extent 正交

| Role | 责任 | finite/infinite 语义 |
|---|---|---|
| Measurement | 使用设备并产生 dataset | 每次 run 可 finite 或 infinite |
| Task | 编排 measurement/分析，发布 progress/LIVE preview，完成后返回结果并保存 artifact | 不用 finite/infinite 给 Task 分类 |
| Processor | 消费上游 signal 并产生派生 dataset | 消费方式由当前上游决定 |

因此删除 `MEASUREMENT -> finite / TASK -> finite / PROCESSOR -> reactive` 的硬映射。

Task 运行中的 measurement 数据可以作为普通 signal 进入 Monitor；Task 完成结果则由调用方直接消费。Calibration 不把最终结果拆成一组为了 report 才存在的 signals，也不发布 report blob。

Camera Measurement 中：

- `Repeat > 0`：finite，收集确定数量的 frames；
- `Repeat = 0`：infinite，worker 持续采集，signal plane 保留 latest；
- UI 不再有另一个 finite/infinite mode 开关。

Finite 上游结束后的“frozen replay”只指：如果 Camera Measurement 已经采完并在 plane 中留下最终 immutable dataset，此后启动 Occupancy 可以对这份数据处理一次，不强迫重跑 camera。它不是磁盘 archive replay，也不保存 infinite 历史。

## 6. Signal 身份和 revision

| 概念 | 用途 |
|---|---|
| Logic/node id | TaskConsole 中的稳定节点身份 |
| Signal key | `node id + output name` 的稳定接线名；Restart 不改名 |
| Generation | 一次成功 Start/run；每次 Restart 更换 |
| Snapshot revision | 同一 generation 中 immutable snapshot 的单调内容次序 |

Revision 不是安全或文件版本。它唯一有用的场景是 plot/fit worker 异步完成时，防止旧 snapshot 的结果晚到后覆盖新图。当前 `OwnedSnapshot`/plot 已经有 revision，因此只复用它，不再新建一套 sequence/revision 机制。

比较时必须同时看 generation：两次 run 都可能有 `revision=1`。新 generation 到来时 panel 替换 plot host；同一 generation 内才用 revision 拒绝晚到的旧结果。Revision 不显示给用户，不进 layout，也不作为科学 provenance。

## 7. Measurement 参数、表单和 Restart

### 7.1 参数的所有权

`camera`、`mot_camera` 等是 Device Manager 安装的 named instances，它们可以是不同 `CameraAdapter` 实现。Camera Measurement/Calibration 只声明需要 `camera.adapter` capability，Workbench 在 Edit 中列出所有匹配实例供用户选择。

exposure、ROI、repeat 等是 measurement request 的参数：

```text
Logic/Panel Edit shared row draft
    -> build CameraMeasurementRequest(camera instance, exposure, ROI, repeat, ...)
    -> resource arbitration
    -> CameraMeasurement configures selected CameraAdapter
    -> adapter validates/snaps SDK constraints and returns actual readback
    -> acquisition worker starts
```

Workbench 只负责表单/draft/selector 接线，不代表 exposure/ROI 属于一个独立“device working-point”层。Device Manager 保存安装信息和默认值；每个 measurement run 的 exposure/ROI 以它自己冻结的 request 为准。

### 7.2 唯一 row draft

- 每个 Logic row 只有一份 draft，包含 node 参数、input binding 和 named device 选择。
- Logic Edit 和该 producer 所有 Panel Edit 投影同一份 draft。一处改动立即同步到其他打开表单，但不偷改当前运行中的 request。
- Logic Edit 只用 `Start`，已运行或已 terminal 时显示 `Restart`。
- Panel Edit 若投影 Producer 参数，只修改同一共享 draft，并复用同一个 `Start/Restart` action；不能换名再造 `Apply` 按钮或第二个 endpoint。

### 7.3 Start/Restart transaction

1. 校验整份 draft 并构建新 request。
2. request 合法后，停止当前同 row 的旧 run，以及与新 run 设备 claim 冲突的其他 Logic Node。
3. 等旧 owner 释放设备。
4. measurement 用 request 配置所选 device，读回 actual 参数，立即启动 worker。
5. 成功后创建新 signal generation；node id 和 signal key 不变。
6. 失败时明确显示 stopped/error，不发布伪成功数据，不猜测硬件 rollback。

一个已经 terminal 的 generation 必须在下一次 `Start/Restart` 时被新
generation 原子取代；它不能因为 FINAL 数据仍可读而阻止同一 row 再次运行。
`Remove` 后用同一 node id 重建也遵守同一规则。只有旧 generation 仍 LIVE
时才拒绝并发启动。

下游的 row 和 input binding 保留。原本 active 的 processor 在新 source generation 来时重新校验；不相容则显示 blocked/incompatible，不继续用旧 calibration 偷跑。

### 7.4 Selector 联动

- `zlc_plot` 发出带坐标语义的 committed selection 和 viewport；viewport 由 plot owner 同时给出 canonical data range 与 display range，Workbench 不重复推断轴或单位。
- `zlc_atom` 的 logic descriptor 声明哪类 selection 可以更新哪个 measurement 字段，例如 Image Area -> Camera Measurement ROI。映射是 data-only，不返回 QWidget 或 device instance。
- `zlc_workbench` 沿 panel -> signal -> producer row 把 selection/zoom/pan 路由到同一个 descriptor mapping，更新同一 measurement draft，并把同一 display viewport 投影给 live/Edit 两张 surface。这个 seam 也供以后的 measurement 使用，不在 Workbench 写死 camera `if`。
- Image Area 用当前 ROI origin/binning 把显示坐标转回 sensor 坐标，adapter 在 Start/Restart 时做硬件 increment 对齐。
- selector 只更新共享 draft；用户按同一个 `Restart` 后，才用新 ROI/exposure 重启 measurement。
- ROI 数据和 fit 参数同时是 data plane 中的普通 typed Dataset。Logic input 可以声明一个固定 contract，也可以显式声明 source-neutral；后一种由插件用实际 Dataset schema 和自己的动态 artifact/request 判断是否可用。Occupancy 属于后一种，因为可用 frame shape 由所选 Calibration 决定，Workbench 不得用 producer 名称或固定 `camera.frames` 字符串提前隐藏 ROI/fit signal。

## 8. 三个目标 Logic Node

### 8.1 Calibration Task

Calibration 不接受 grid rows、columns、site count 或预先 `SiteLayout`。Site 数量、位置和排序是 calibration 要测出的结果，不是输入真相。

建议的分析边界：

1. 按所选 calibration pulse/protocol 采集样本帧；“bracket”不作为 UI 概念或用户参数。
2. 从标定图像自动发现 site candidates，根据局部对比度/噪声、最小间距和 spot 尺度去重。
3. 精修每个 site 的 pixel center，生成稳定 site id/排序。若能从坐标推断拓扑，拓扑也是输出，不是必需输入。
4. 只生成一份 site labels 和 train/held-out split，以同一 site axis 同时训练三种 `ReadoutModelKind`：`box`、`psf` (per-site PSF) 和 `uniform_psf`。每个模型都保存自己的 integration feature/PSF、threshold、usable/quality，artifact 另存一个 `default_model_kind`。
5. 运行中通过 NodeHost 发布 progress 和当前 `capture_preview`。Preview 只投影最近一个完整 cycle 的最后一张二维 camera image，固定为 `R=1, P=1`；采集历史属于本次 Calibration result，不得把 `samples x 3 x Y x X` 累计数组冒充“当前图”反复交给 Monitor。循环完成后计算一次包含 SiteMap、三种模型及各模型诊断数据的 Calibration result。

Calibration 的 `Reference exposure` 与 `Readout exposure` 是显式 protocol 参数：相机 adapter 以 reference exposure 配置本次 run 的最大积分时间，编译后的 long/readout/long 外部门宽分别使用两项 authored 值。外部门宽可以按真实物理缩短某个 frame 的有效曝光，但不能把已配置的 camera exposure 延长；普通 Camera Measurement 仍只由自己的 request 配置 exposure，不能被一个无声明的 pulse metadata 替代。这样三帧继续共享同一 shot occupancy，同时 exposure 归属没有第二份隐式真相。

Calibration Edit 的 pulse/template 参数是一个以 project `pulses` 目录为起点的
文件选择控件，显示所选 JSON 的明确路径；不能用只显示裸文件名、看不出目录的
ComboBox。默认选择 `imaging_template.json`。该文件使用唯一的 `zlc_pulse`
tree 格式 `format: "zlc.pulse.v1"`：`slots` 只表示 scan 维度，三项
Calibration duration 则由显式 `PulseApiParameter/api_parameters` 分别绑定
`reference_probe_duration_before`、`readout_probe_duration`、
`reference_probe_duration_after`。只有用户 Start Calibration 时，Calibration
Logic 才通过 `sequence_from_tree()`、`resolve_api_parameters()` 和连接板卡的
`BoardDescription` 编译；未解析 API parameter 的 sequence 不能被 compile。
不得另造 `PulseDocument`、按名字前缀猜 API/scan、把 API parameter 塞进 scan
table，或保留第二套 pulse model。产品链不把 Python module 当 pulse 文档，也不
由 TaskConsole 启动或 Device Manager `Init devices` 偷偷预载 Calibration pulse。

Pulse 的唯一文件写路径是 `PulseEditorState -> state_to_tree() -> sequence_to_tree() -> write_readable_json()`，读路径严格反向。标量 list 保持紧凑并在可读宽度换行，带结构的 list 才展开。Shipped `imaging_template.json`、Pulse UI Save 和以后的 template generator 都必须经这一条序列化路径；不得手写第二套 `json.dump` 排版，也不得恢复 `.py` pulse 文档。

`SiteMap` 至少包含：

- stable site id/axis；
- sensor/image coordinate frame；
- center `(x, y)`；
- 可选的自动推断 topology/order；
- 每个 site 的 validity/quality。

SiteMap 只负责“site 在哪里、身份如何对齐”。科学积分框/PSF kernel 和 threshold 属于 readout models，不塞进 SiteMap。Calibration artifact 同时保存 SiteMap、与其 site axis 对齐的三种 readout models、`default_model_kind` 及实际 frame contract（camera、sensor shape、ROI、binning、exposure/readout mode）。Occupancy 显式选择 default 或其中一种模型提取各 site 读数并套该模型 threshold。

Calibration result 中的所有 site 数据共享 SiteMap 的实际 `site_ids` 和同一 pixel coordinate frame。每种模型都在这个结果中保存自己的 held-out samples、fidelity、thresholds 和 fit 所需数据；不能只保留 default model 后再重算另外两种模型。

循环结束后的代码路径保持直接：Calibration Task 先把该结果写成 calibration JSON，然后把同一个 Python result 交给现有 `zlc_plot` 公开 API，保存六张 report 图片：(1) site-map image + centers；(2) 三种 readout model 的 per-site held-out fidelity；(3–5) box、per-site PSF、uniform PSF 三个 classifier distribution grids；(6) per-site PSF kernel `FacetGrid[Image]`。三个 Distribution grid 都启用 plot-owned threshold classifier，并在同一次 `configure()` 中使用该 readout model 已算出的 per-site thresholds；不得再启用普通 `bimodal_gaussian` fit 伪装 classifier。JSON 和六张图是同一次 Calibration result 的两种文件投影，不是两次分析。TaskConsole 不显示、挂载或自动打开 report；Monitor 只显示循环中的 measurement-linked preview。Calibration Task 决定 report 内容，`zlc_plot` 实现绘图；Task 不持有 Qt、TaskConsole panel state 或另一个 renderer，也不为这一条路径引入通用 report registry/coordinator/transaction 框架。

`SiteMap` 不是 plot kind，UI 中也不存在 `SiteMap Plot`。它是 Calibration artifact 里的 domain data。Occupancy 在与 `frame_judged` 相同的 publication 中显式发布 typed `site_overlay` sibling：canonical site ids、人类短标签、pixel centers 和当前 status。固定的 `Image` panel 分别选择 Image signal 与可选 Overlay signal；Workbench 只核对二者属于同一 publication，不从 run record 偷读 calibration JSON，也不从 signal 名称猜 SiteMap。`zlc_plot` 只把这份通用 point-overlay data 画成 markers。绘图圈半径是 Image display 属性（可由 point spacing 自动给出），不与科学 integration half-width 混为一个参数。

输出文件由 Experiment workspace 选择目录和不重复文件名，例如 `calibration_20260808_153012_01.json`。这只是避免依赖 process cwd 和无提示覆盖，不是版本/安全机制。

### 8.2 Camera Measurement

Edit 中的理想参数：

| 参数 | 语义 |
|---|---|
| Camera | 从所有满足 runtime-checkable `CameraAdapter` 的 named instances 中选择 |
| Exposure | 本次 measurement 的 exposure，传给选中 camera |
| ROI | 本次 measurement 的 sensor ROI，可由 Image selector 反写 |
| Repeat | `0 = infinite`，正数为 finite cycles |
| Frames per cycle | 每个外部时序 cycle 期望的 frame 数 |

Camera Measurement 按本次 authored `Frames per cycle` 声明
`frame_0 ... frame_N`。每个输出都是一个普通二维 image Dataset signal；同一 cycle
内的不同 frame 不塞进一个额外 data axis，也不由 `zlc_plot` 解释 camera-specific
`frame_index`。signal chooser 逐项列出这些 signals，并在每项旁显示 dataset shape。
`frames_per_cycle` 的 shot 分组只有一个所有者：Camera Measurement 的共同采集实现。
所有 `CameraAdapter` 只交付按物理采集顺序编号的 `CameraFrameRecord`，不得在丢帧后
重新编号成无缺口序列；共同实现按连续且对齐的 source ordinal 组装完整 cycle，任何
缺帧、错位或 overrun 都不能跨 shot 补齐。只有完整 tuple 才一次发布全部 sibling，
`Repeat=0` 的 latest slot 也只以完整 tuple 为单位覆盖，因此慢 consumer 最多跳过完整
cycle，不会把不同 shot 的 frame 拼在一起。Virtual/DCAM/Pylon 只负责把各自 driver 的
frame counter、buffer overrun 和 failed grab 如实投影到该共同契约，不各自实现
`frames_per_cycle` 业务规则。Pylon 的 `LatestImageOnly` 只可用于真正的自由运行设备
预览；Camera Measurement 的 `Repeat=0` 必须使用 external-triggered ordered stream。
连续采集的内部 raw buffer 至少容纳一百二十八个完整 cycle，容量始终向
`frames_per_cycle` 的整数倍取整；这给短时 worker 调度、绘图和 UI 停顿留出读取
余量。该容量由 Camera Measurement 一次决定，Virtual deque、DCAM ring 和 Pylon
`MaxNumBuffer` 必须落实同一个帧数，不得只接收参数却继续使用 driver 默认值。它不是
用户参数，也不改变 latest-only 的发布语义。
等待/触发超时属于 adapter/session 内部采集策略，不作为普通用户 authoring 字段。

Camera Measurement 不驱动 pulse。它监听外部时序，独占 camera、最多只读 sequencer 状态。Driver/internal buffer 大小不进用户表单。

### 8.3 Occupancy Processor

Edit 中的理想参数：

| 参数 | 语义 |
|---|---|
| Frames signal | 显式选择一个 contract-compatible image/frame signal |
| Calibration file | 显式选择 calibration JSON path |
| Readout model | `default` / `box` / `psf` / `uniform_psf`；`default` 解析 artifact 的 `default_model_kind` |

没有 device、finite/infinite extent mode、buffer 或 loss 参数。Readout-model choice 是 Occupancy 必须显式拥有的科学算法模式，不是被删除的 extent mode。Start 时加载 calibration，只拒绝 frame shape、sensor、ROI、binning 等会破坏像素与 site 对齐的结构差异；exposure、camera id 和 readout mode 继续保存在 frame contract 作为 provenance，但不限制同一几何 calibration 的使用。随后按 SiteMap + 所选 readout model 的 feature/threshold 产生 per-site counts/occupied/rate 等 dataset。Finite source 顺序处理/可处理已完成的 frozen dataset；infinite source 只处理 latest。

## 9. TaskConsole Logic UI

### 9.1 Add/Edit 生命周期

- TaskConsole header 只有权威 v1 的一个 combined 下拉框和一个 `Add Panel` 按钮。下拉框依次列 `Plot`、`Measurement`、`Processor`、`Task`；不得新增独立 `Add Logic` 按钮或 modal logic chooser。
- 选中 Measurement/Processor/Task 后，同一个 `Add Panel` 按钮把 catalog `api_name` 送入唯一 `add_logic` endpoint，创建 stopped row，并立即切换到对应 Logic Edit tab。
- Logic Edit 实时编辑 row draft，包含本 node 的所有 measurement/task/processor 参数和 input binding。
- 按钮是 `Start/Restart`、`Stop`、`Remove`；产品 UI 没有 `Apply`。
- 新 Logic Node 启动时按第 3 节的 claim 规则停掉占用冲突设备的旧 node。

### 9.2 三个 node 在 Edit 中的字段

| Node | 字段 | 明确不显示 |
|---|---|---|
| Calibration | Camera instance；Sequencer instance；以 project `pulses` 为起点的 JSON file picker；Samples（默认 300）；Reference exposure；Readout exposure；Camera ROI；默认 readout model；box/PSF 训练参数；必要的高级检测参数；完成后只读 Detected sites | grid rows/columns/site count；`bracket`；timeout；用户填写的 output signal |
| Camera Measurement | Camera instance；Exposure；ROI；Repeat (`0=infinite`)；Frames per cycle | 独立 mode；user buffer；loss 计数；pulse drive；普通用户 timeout |
| Occupancy | Frames signal；Calibration file；Readout model (`default` / `box` / `psf` / `uniform_psf`)；只读输出摘要 | Device；finite/infinite extent mode；buffer；隐式“current calibration” |

Calibration 中所谓“必要的高级检测参数”只能是算法确实需要暴露的噪声门限、最小间距或 spot 尺度之类调整项；它们不能变相成“用户先告诉 site 数量/形状”。默认自动模式应当不需要用户调它们。

### 9.3 Task takeover、LIVE preview 与 terminal 清理

- Calibration 循环中的 `capture_preview` 是唯一自动加入 Monitor 的 measurement-linked panel；它显示最新一张二维 camera image，不携带累计采集维度。循环完成后，Calibration Task 直接把 result 交给 `zlc_plot` 保存 site map、fidelity、三个 classifier grids 和 PSF kernels 六张 report 图片。Workbench 只显示 Task 的进度和运行中 preview，不组装、不显示、不自动打开 report。
- Task active 时，TaskConsole header 的可操作区切换为唯一的 task status strip：显示当前阶段、进度和唯一 `Stop Task`。所有会改变系统状态的 header controls、logic rows/cards、Setting/Edit controls、Add/Start/Restart/Stop/Remove 都禁用，不能与 exclusive Task 并行改写 draft 或设备状态。
- Monitor 中 selector、zoom、pan、fit inspection 只允许 view-only；Task active 时 selector commit 不能回写任何 producer draft。
- 每个 task generation 的 LIVE preview 都带明确生命周期。它只在该 Task host 正在运行时允许自动创建；terminal、Stop、失败或取消统一移除 transient preview，后续 beat 即使仍能读到该 generation 的最后 publication 也不得重建。下一次 Restart 的新 generation 才能创建新的 preview。
- Monitor 左侧作为一个整体滚动区域响应鼠标滚轮；不能只有某个子控件吃掉滚动而使 logic/panel 列表无法上下移动。仅让 plot widget `ignore()` wheel 不足以证明这一点：Selectors 关闭时，Panel Card 必须把 surface 上的 Wheel 事件明确交给唯一的祖先 board scroll owner。

## 10. Plot Panel UI

### 10.1 Add Panel

TaskConsole 的 Plot catalog 顺序和标签固定为：`2D image` (`Image`)、`1D vector` (`Curve`)、`Rolling trace` (`Rolling`)、`Distribution` (`Histogram`)、`Site grid` (`FacetGrid`，cell kind 固定为 Curve)。`PulseTimeline` 属于 Pulse UI，不得出现在 TaskConsole catalog。

在 header combined 下拉框选中 `Plot: ...` 后，同一个 `Add Panel` 按钮只创建该固定 kind 的 blank panel；它不要求 signal 已存在，也不自动挑 signal。空卡显示 `Pick a signal in Setting`，随后由 Setting/Edit 选择 signal。Plot kind 一旦创建就固定；Setting/Edit 只读显示，需要另一种 kind 就新建 panel。

### 10.2 Setting frame

Setting 是 monitor board 上的完整初始配置面，而不是第一次修改后才补齐字段的简化表单。它从同一份 `zlc_plot` kind schema 构造，blank panel 创建时就显示全部 data-independent 参数：

- Plot kind（只读）；
- Signal；
- Panel size；
- Display interval：只用有限值 `100 / 200 / 400 / 800 ms` 的 ComboBox，默认 `400 ms`，不能用可输入非法值的 SpinBox；
- title/labels、limits 和当前 kind 声明的全部 data-independent display/interaction 参数；
- Edit / Remove。

Panel frame name 与图内 title 是两个事实：frame name 默认包含所选 signal 以便在 Monitor 中辨认，也可单独重命名；图内 title、axis labels、units 和 limits 仍是 `zlc_plot` 的 display 参数。凡 plot schema 声明 `None` 为 automatic 的字段，Setting 与 Panel Edit 都在编辑器右侧显示 `Auto` switch；开启时禁用手工 editor并使用 PlotSpec/data 默认，关闭时提交手工值。不得再由 Workbench 用 frame name 覆盖 plot title。

依赖 dataset schema 的 axis/reduction/group/facet choices 和依赖真实数据的 fit action 也按稳定位置显示，但在没有 compatible signal 时禁用并说明原因，不能因为 signal 为空而让其他设置消失。每个 signal label 同时显示人类可读名称和当前 dataset shape。Camera cycle 的各个 frame 已经是普通独立 signals，plot 参数层不再增加 camera-specific frame choice。Display interval 只控制 panel display scheduler；TaskConsole app beat 独立驱动 scheduler，二者不是同一个可编辑数值。

Setting 使用现有 `FluentPopup` 和 `FluentSettingsPopupAnchor` 锚在所属 panel 的 Setting 按钮旁，宽度约为 `2x2` panel 的一半；`Qt.Popup` 负责点击外部关闭，标题条允许拖动，内容放不下时在 popup 内滚动，外间距复用 Fluent popup gap。popup 从创建时就以 card 为 parent，form/button/control 从创建时就以 Setting body/popup 为 parent；只允许这个有身份的预期 popup 收到 top-level Show，禁止先显示无 parent 临时窗口再 reparent。Setting 没有 `Apply` 按钮：每个已完成编辑/choice commit 立即替换同一 `PanelState`；同一 signal/schema 的完整目标配置一次提交给当前 `zlc_plot` host，Display interval 也必须立即生效。

Monitor 的 `Selectors` 默认关闭，与 v1 一致。关闭时 plot widget 不消费 wheel，Panel Card 把 wheel 明确路由到外层 board scroll；打开后 wheel 才属于 plot 的 zoom/selection interaction。这个开关直接调用同一 plot widget 的 interaction gate，不重建 panel。

改 Signal 只换这个 panel 的绑定，不改 Occupancy 等 Logic Node 的 input binding，也不改 plot kind。

每个 panel 只有一份 Workbench-owned `PanelState`，其中包含 signal binding、size、update interval、plot semantic/display/fit 参数和固定 plot kind。Setting frame 和 Panel Edit 都是这一份 state 的 view/controller，不各自保存副本。

### 10.3 Panel Edit tab

Edit 是一个 tab，不是 modal。它包含：

- Plot kind（只读）、Signal、Panel size、有限值 Display interval ComboBox；后三个可编辑字段与 Setting 重复显示，两处直接绑定同一 `PanelState`；
- 当前图形和 `Refresh snapshot`；
- 完整 semantic/display/fit 参数和结果；
- selector/zoom/pan；
- direct producer 的完整 Logic parameter form，它是 producer row draft 的另一个投影；
- 与 producer row 共用的 `Start/Restart` action；
- `Save Fig`：保存这个 panel 当前图像和对应数据。

这种重复是有意的：用户在图上做 Area/range selection 或 zoom/pan 时，可以同时看到 ROI/range 等 producer 参数更新，然后调用同一个 `Restart`。

Setting 或 Edit 从任一边提交修改时，controller 立即替换同一 `PanelState`，两个 view 和 monitor panel 都收到同一次更新。不写“Setting -> Edit”和“Edit -> Setting”两套手工拷贝逻辑。Edit 中的 frozen data snapshot 与 `PanelState` 分开：参数始终同步；如果换了 signal，旧 frozen 图标为 stale，用户 Refresh 后取新 signal 的 snapshot。

Panel Setting/Edit 每次字段 commit 都把完整目标配置一次交给 `zlc_plot`：semantic mapping、整张 display parameter mapping、size、Image overlay 和 fit choice。Workbench 不判断哪一个字段能原位更新，也不循环调用单字段 setter；`zlc_plot` 用当前 `PlotSpec`、`ParameterSchema`、layout、overlay 和 fit 状态比较差异，合并需要的 render effects，并在同一个 worker job 中最多发布一张同步 front。只要 signal/schema 兼容，就保留同一个 host 和 Figure；只有 signal 改变、generation 改变或 schema 不兼容才替换 host。Fit 求解本身继续是异步科学计算，完成后再发布 fit overlay。owner thread 不调用 `.result()` 等待，旧的完整配置 job 由同一 coalescing key 淘汰。产品 UI 不存在 `Apply`；measurement 重配只走同一个 `Start/Restart` endpoint。

Live fit 不改变外部 data API：有无 fit 都只调用同一个 `RasterPlotHost.update_data()`。每个新 revision 先投影并发布 data front，立即撤下不再对应当前数据的旧 fit overlay；随后取消前一 solver，只在 `zlc_plot` 的既有 analysis worker 中拟合当前最新 revision。fit 完成后仍须通过当前 data revision/request generation 校验才可发布 overlay 和 `FitEvent`，所以晚到结果不能覆盖新图。`LivePlotController` 只提供 capacity-one/latest ingress 与 cadence，不拥有第二套 fit 状态机，也不要求 data 等 fit 完成后才显示。

Distribution 的 threshold classifier 是该 plot kind 自己的 boolean display 参数，和通用 fit 完全独立。打开后由 `zlc_plot` 自己执行 bimodal Gaussian classification fit，显示左右 Gaussian、总和、可拖动 threshold，以及当前 threshold 对应的 fitted population 左/右占比（两者严格合计 100%）和 balanced fidelity；初值是 equal-prior 最优 threshold。普通 fit 的启停、model、结果和状态不得创建、移动或清除 classifier，classifier 也不得写普通 `fit_status`。FacetGrid[Histogram] 对每个 cell 使用同一 classifier，overview 中保留三条曲线、threshold 和较小字号的三项数值；focus 后只把该 cell 的 threshold 变成可交互 selector。外部若已有模型 threshold，就把 toggle 与整组 canonical thresholds 放进同一次 `configure()`，不先画静态线再另跑普通 fit。

### 10.4 各 plot kind 的理想参数

| TaskConsole label / kind | schema 中的 semantic 参数 | schema 中的 display/interaction | Setting 初始 data-independent surface |
|---|---|---|---|
| `2D image` / Image | X/Y axes；Reduction；optional typed Overlay signal | colormap；color limits；interpolation；colorbar；optional small point labels；marker radius/style；empty/occupied/invalid colors；Area selector；2-D fit | title；colormap；color limits；interpolation；colorbar；all overlay/marker styling；selector display |
| `1D vector` / Curve | X axis；Group by；Reduction | labels/units；grid；limits；X-range selector；compatible fit | title；X/Y labels；units；grid；limits；selector display |
| `Distribution` / Histogram | value/reduction selection | bins；density；cumulative；log Y；range selector；独立 threshold-classifier switch/selector；compatible fit | title；bins；density；cumulative；log Y；classifier switch；range/limits |
| `Rolling trace` / Rolling | Group by；Reduction | window；Y limits；side distribution；X-range；compatible fit | title；window；Y limits；grid；side-distribution display；selector display |
| `Site grid` / FacetGrid[Curve] | Facet axis；fixed Curve cell semantic parameters | packing；focus cell；cell selector；compatible per-cell fit | title；facet unit；packing；focus/cell display；Curve display parameters |

Fit model 和参数兼容性由 `zlc_plot` 声明，UI 不写死列表。Overlay 不是 `zlc_plot` 的 Off/Centers/Occupancy mode 参数；它是 panel 的第二个显式 signal binding。producer 决定坐标、身份和状态，`zlc_plot` 只按 typed overlay contract 绘制，不从 grid shape 或 domain artifact 生成圈。
Site 旁边不得显示长 site id；若开启标签，最多在 marker 左上角显示小号 ordinal 数字。数字颜色和透明度跟对应 site 圈完全一致，使 empty/occupied/invalid 仍可由同一状态色区分；vacant/empty 必须明显淡于 occupied，不能成为画面的第一视觉层。

## 11. 三种 Save（必须分开）

| 用户动作 | 保存什么 | 不保存什么 |
|---|---|---|
| TaskConsole header `Save Layout` | pipeline/layout JSON：nodes、各 row draft、named device 选择、signal 接线、panels 的固定 plot kind/spec/size/interval/order | panel dataset、running state、generation/revision、device snapshot |
| TaskConsole header `Save Screenshot` | 整个当前 TaskConsole GUI 的一张普通图片 | layout JSON、科学数据、provenance |
| Panel Edit `Save Fig` | 该 panel 当前 frozen snapshot 的 image + data archive + 对应 plot state + 本次 run 调用链参数/device snapshots | 整个 monitor tab 或其他 panels 的数据 |

`Save Layout` 加载后恢复相同节点、`camera_measurement -> occupancy` 接线、panel signal/plot kind 和布局，但全部是 stopped draft。加载 layout 不打开或配置设备。

Panel `Save Fig` 只围绕当前 panel：

- 数据是 Edit tab 正在显示的同一 frozen snapshot，不在 Save 时又抓一份 latest；
- 调用链参数是该 run 真正启动时冻结的值：例如 camera instance/exposure/ROI/repeat，Occupancy 的 source/calibration path，以及上游 task/pulse 参数；
- device snapshot 是 run 配置/采集时从公开 adapter API 读回的 actual state，不是点 Save 时临时拍可能已经变化的状态；
- 如果该 `Image` panel 选择了 Overlay signal，重画所需的 typed coordinates/ids/labels/status 作为该 Image plot 的 data/annotation 保存；不创建 SiteMap plot kind，也不复制 calibration artifact；
- calibration 在调用链中按实际 `calibration_path` 记录，不内嵌另一份 calibration JSON，不考虑我之前臆造的“移动 panel 数据”场景；
- 不新增 fingerprint/hash。

Calibration Task 自身的 JSON/report 是第四个业务文件，但不是 TaskConsole Save 按钮：它在 Task 成功时由 workspace 自动写出。

## 12. 状态更换时的行为

| 动作 | 设备 | Signal | Panel | Processor |
|---|---|---|---|---|
| 修改 draft | 不变 | 不变 | 继续显示旧 run | 继续处理旧 run |
| Logic/Producer Restart | 停冲突 owner，用新 request 重配并启动 | key 不变，generation 更换 | 保留绑定，新数据到达时替换 host | active row 重新校验；不相容则 blocked |
| 新 Logic Node 占用同一 device | 旧冲突 node 被停止 | 旧 generation terminal | 可保留明确 frozen/stale 图 | 依赖该 source 的 active processor 停止/等新 source |
| Panel 换 Signal | 不变 | Logic signal 不变 | 该 panel 换绑 | 不变 |

## 13. 当前实现和验收状态

本文不作完成声明。此前基于六个 sites、旧 plot surface 和旧 Calibration artifact/report 的所谓正式 UI 验收已经撤销，不能作为当前树的产品证据；virtual apparatus 的默认验收目标是 `5 x 7 = 35` sites 和 `96 x 128` frames。

最新实现完成后必须从根 `bin\experiment.bat` 重新走一次真实按钮路径，而不是调用 presenter/private API 代替点击：Device Manager `Init devices` -> 同 session 同时出现 Pulse UI 和 TaskConsole -> Calibration Task takeover/progress/唯一 Monitor LIVE preview/Stop Task -> 确认 JSON 和六张 report 图片已写出且 UI 未创建 report panel/tab/window -> `Repeat=0` Camera Measurement -> Pulse JSON Load/On -> Occupancy readout-model choice -> 五种固定 panel kind 的 blank/full-schema/finite interval/即时 Setting commit -> signal/overlay/selector/Producer Restart -> 三种 Save -> Stop Pulse -> close。随后还必须验证窗口、session、workers、claims 和项目 Python 进程归零。当前真实按钮复验、性能复验、受影响测试和全树测试的未完成状态只记录在 `IMPLEMENTATION_PLAN.md` Checkpoint。

## 14. 本轮 review 的结论和非问题

以下已经定案，实施时不应再当成待决定项：

- Calibration 自动发现 sites；无 grid shape/count 输入。
- Calibration 用同一 labels/split 训练 `box`、per-site `psf`、`uniform_psf` 三种模型；Occupancy 显式选择 default/具体模型。
- Calibration 跑一次采集循环并计算一次结果；同一结果写 JSON 后直接交给 `zlc_plot` 保存 site-map、fidelity、三模型 classifier grids 和 PSF kernel grid 六张 report 图片。Monitor 只含循环中的 preview；Workbench 不显示 report。Task active 时 header takeover 并阻止所有状态改写，只保留 `Stop Task`。
- Virtual devices 只位于独立 simulation package，满足与真实设备相同的 `CameraAdapter`/`SequencerDevice` 契约，默认 `5 x 7 = 35` sites、`96 x 128` image。
- Pulse 文件只有 `zlc.pulse.v1` JSON 和一条 readable writer/readback 路径，不存在 `.py` pulse 或第二套排版。
- exposure/ROI 是 measurement 参数，由 UI 接线层维护共享 draft 并在 Start/Restart 时传给 device。
- `Repeat=0` 是 infinite。
- combined `Add Panel` 选择 Logic entry 后创建 stopped row 并自动进 Edit tab；没有独立 `Add Logic` 按钮或弹窗。
- Plot kind 在 Add Panel 时固定。
- TaskConsole 只提供五种固定 plot kind；Display interval 是有限 ComboBox，blank panel 初始即显示完整 schema，Setting 无 Apply、字段 commit 立即异步准备并拒绝 stale result。
- Panel Edit 重复显示 panel 参数和 direct producer 参数，selector 更新同一 measurement draft，Producer Restart 调用同一个 Start/Restart endpoint。
- Setting frame 和 Panel Edit 直接绑定同一 `PanelState`，对应参数天然双向同步；不存在两份 panel config。
- SiteMap 是 calibration domain data，不是 plot kind；Occupancy 显式发布同-publication typed overlay sibling，固定 `Image` panel 选择 optional Overlay signal，`zlc_plot` 只绘制通用 annotation。
- 设备可多方只读，只有 exclusive Logic Node 单占；新冲突 node 停旧 node。
- TaskConsole/Pulse Editor 使用同一 Experiment session，不引入 IPC。
- Header Layout Save、Header Screenshot 和 Panel Save Fig 三者语义分开。
- Panel archive 不复制 calibration JSON，不增 hash/fingerprint。

实施期间遇到任何未预见问题、新矛盾或现有代码与本架构冲突时，不停下询问、不把决策退回给用户。执行者必须按“用户已裁决的产品语义 > 本架构文档 > 整条科学数据链正确 > 最简单可维护的实现 > v2 现状 > v1 参考”自主作出最优决定，在实施记录中简要记录理由，然后继续运行直到交付定义全部满足。v1 或当前 v2 的错误实现不能重新打开本文已定案的决定。
