# Zou_lab_control v2 目标架构设计（执行权威）

> 仓库绝对路径：`C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2`
> 本文绝对路径：`C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\ARCHITECTURE_DESIGN.md`
> 设计基线：`0243aa6`。完成状态与实测证据以 `IMPLEMENTATION_PLAN.md` Checkpoint 为准；package/full-tree 测试不能替代真实实验入口和可见 GUI 的验收。
> 权威顺序：用户当前裁决 > 简单且可维护的边界 > v2 当前实现 > v1 参考。v1 不是规格。
> 根目录 `HANDOFF.md` 已在最初接手阶段完整读取，其有效要求和现状已吸收到本文与实施计划。Goal 启动后它只是历史输入，不是续跑权威，也不需要在每次上下文恢复时重读。

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
    -> calibration JSON (SiteMap + per-site readout model)
Camera Measurement
    -> frames signal
Occupancy Processor(frames + calibration path)
    -> occupancy data
Plot Panel
    -> Panel Edit Save Fig(image + data)
```

同时保证 TaskConsole 的 Add -> Edit -> Start/Restart、Panel Producer Apply、selector 反写 measurement 参数、两种 header save 都符合产品语义。

## 2. 核心设计原则

1. 一个事实只有一个所有者。科学数据归 `zlc_data`，运行和 signal plane 归 `zlc_runtime`，中性原实验语义归 `zlc_atom`，绘图语义归 `zlc_plot`，widget 归 `zlc_ui`，跨层接线归 `zlc_workbench`。
2. UI 只提交意图和 draft，不直接读写硬件。阻塞 camera/sequencer 调用在 worker/session 侧执行。
3. exposure、ROI 是 Camera Measurement 的参数。UI 接线层负责表单同步和 selector 反写；Start/Restart 时 measurement request 将参数传给用户选中的 camera adapter。
4. 不增加 fingerprint、SHA/hash、防篡改协议、想象出的兼容层，或大量防御型包装。
5. latest buffer 只保存当前值，不记录、展示或存档丢了多少/哪些。
6. 测试守卫尽量修改现有测试。只增加少量能在原缺陷下变红的纵向行为测试。

## 3. Experiment、GUI 和设备所有权

Notebook 创建的一个 `Experiment`/session 天然是共享底层。TaskConsole 和 Pulse Editor 都从它取同一组 named devices、同一个 virtual world 和同一个 sequencer。不新造第二个 session、IPC 或“两个 GUI 抢着 open 设备”问题。

### 3.0 正式实验入口与 Device Manager

1. 根目录 `bin\experiment.bat` 就是正式实验入口，等价于 v1 的 `task_console.bat`：它只启动一个统一的 TaskConsole experiment flow，不能串起多个 Python 进程，也不再增加第二个根入口。
2. 同一 Qt 进程先显示 v1 操作流程的 Device Manager：Config、状态与文档名、Installation/Configured devices、Available/Loaded，以及 New、Load、Save、Save as、Cancel、Init devices（活动后为 Shutdown/Shutdown for restart）。底层继续使用 v2 的 `InstallationConfig`/`apparatus.json` 和 named devices，不复制 v1 domain/storage。
3. `Init devices` 从当前 draft 创建并持有唯一 `ExperimentSession`；成功后隐藏 Device Manager，并在同一进程中同时显示 TaskConsole 和 Pulse UI。两者接收同一个 session、virtual world 和 sequencer 实例。
4. 实验入口中的 Pulse UI 借用 session 的 sequencer authority；它不自行 dial 第二个设备，也不拥有或关闭底层 sequencer。独立 `pulse_editor.bat` 只作为单独的脉冲编辑/连接测试工具，不是正式 Experiment 入口；它自行连接的 sequencer 也不代表 TaskConsole session。
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
| `zlc_atom` | device base capability/adapter，logic descriptor/request，calibration/site-map/occupancy 物理 | Qt、panel/layout/archive UI |
| `zlc_workbench` | Experiment 接线、row draft、资源仲裁、Start/Restart、panel-producer 联动、save 组装 | 相机 SDK 分支、site 检测算法、plot 参数合法性 |

## 5. Logic 模型：role 与数据 extent 正交

| Role | 责任 | finite/infinite 语义 |
|---|---|---|
| Measurement | 使用设备并产生 dataset | 每次 run 可 finite 或 infinite |
| Task | 编排 measurement/分析并生成 artifact/report | 不用 finite/infinite 给 Task 分类 |
| Processor | 消费上游 signal 并产生派生 dataset | 消费方式由当前上游决定 |

因此删除 `MEASUREMENT -> finite / TASK -> finite / PROCESSOR -> reactive` 的硬映射。

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

## 7. Measurement 参数、表单和 Producer Apply

### 7.1 参数的所有权

`camera`、`mot_camera` 等是 Device Manager 安装的 named instances，它们可以是不同 `BaseCamera` 子类。Camera Measurement/Calibration 只声明需要 camera capability，Workbench 在 Edit 中列出所有匹配实例供用户选择。

exposure、ROI、repeat 等是 measurement request 的参数：

```text
Logic/Panel Edit shared row draft
    -> build CameraMeasurementRequest(camera instance, exposure, ROI, repeat, ...)
    -> resource arbitration
    -> CameraMeasurement configures selected BaseCamera adapter
    -> adapter validates/snaps SDK constraints and returns actual readback
    -> acquisition worker starts
```

Workbench 只负责表单/draft/selector 接线，不代表 exposure/ROI 属于一个独立“device working-point”层。Device Manager 保存安装信息和默认值；每个 measurement run 的 exposure/ROI 以它自己冻结的 request 为准。

### 7.2 唯一 row draft

- 每个 Logic row 只有一份 draft，包含 node 参数、input binding 和 named device 选择。
- Logic Edit 和该 producer 所有 Panel Edit 投影同一份 draft。一处改动立即同步到其他打开表单，但不偷改当前运行中的 request。
- Logic Edit 没有另外的通用 `Apply`。它用 `Start`，已运行时为 `Restart`。
- Panel Edit 的 Producer 区有 `Apply`。它的语义就是“使用当前共享 draft 重新 Start 这个 measurement”，不是新机制。

### 7.3 Apply/Restart transaction

1. 校验整份 draft 并构建新 request。
2. request 合法后，停止当前同 row 的旧 run，以及与新 run 设备 claim 冲突的其他 Logic Node。
3. 等旧 owner 释放设备。
4. measurement 用 request 配置所选 device，读回 actual 参数，立即启动 worker。
5. 成功后创建新 signal generation；node id 和 signal key 不变。
6. 失败时明确显示 stopped/error，不发布伪成功数据，不猜测硬件 rollback。

下游的 row 和 input binding 保留。原本 active 的 processor 在新 source generation 来时重新校验；不相容则显示 blocked/incompatible，不继续用旧 calibration 偷跑。

### 7.4 Selector 联动

- `zlc_plot` 只发出带坐标语义的 committed selection；zoom/pan 不自动改 measurement 参数。
- `zlc_atom` 的 logic descriptor 声明哪类 selection 可以更新哪个 measurement 字段，例如 Image Area -> Camera Measurement ROI。映射是 data-only，不返回 QWidget 或 device instance。
- `zlc_workbench` 沿 panel -> signal -> producer row 路由 selection，更新同一 measurement draft。这个 seam 也供以后的 measurement 使用，不在 Workbench 写死 camera `if`。
- Image Area 用当前 ROI origin/binning 把显示坐标转回 sensor 坐标，adapter 在 Start/Apply 时做硬件 increment 对齐。
- 用户按 Panel Producer `Apply` 后，才用新 ROI/exposure 重启 measurement。

## 8. 三个目标 Logic Node

### 8.1 Calibration Task

Calibration 不接受 grid rows、columns、site count 或预先 `SiteLayout`。Site 数量、位置和排序是 calibration 要测出的结果，不是输入真相。

建议的分析边界：

1. 按所选 calibration pulse/protocol 采集样本帧；“bracket”不作为 UI 概念或用户参数。
2. 从标定图像自动发现 site candidates，根据局部对比度/噪声、最小间距和 spot 尺度去重。
3. 精修每个 site 的 pixel center，生成稳定 site id/排序。若能从坐标推断拓扑，拓扑也是输出，不是必需输入。
4. 以同一 site axis 为对齐键，另外构建 readout model：每个 site 的 integration feature/PSF、threshold、usable/quality。
5. 写出 plain calibration JSON/report，不发布 `calibration`/`report` signal。

Calibration 的 `Reference exposure` 与 `Readout exposure` 是显式 protocol 参数：相机 adapter 以 reference exposure 配置本次 run 的最大积分时间，编译后的 long/readout/long 外部门宽分别使用两项 authored 值。外部门宽可以按真实物理缩短某个 frame 的有效曝光，但不能把已配置的 camera exposure 延长；普通 Camera Measurement 仍只由自己的 request 配置 exposure，不能被一个无声明的 pulse metadata 替代。这样三帧继续共享同一 shot occupancy，同时 exposure 归属没有第二份隐式真相。

`SiteMap` 至少包含：

- stable site id/axis；
- sensor/image coordinate frame；
- center `(x, y)`；
- 可选的自动推断 topology/order；
- 每个 site 的 validity/quality。

SiteMap 只负责“site 在哪里、身份如何对齐”。科学积分框/PSF kernel 和 threshold 属于 readout model，不塞进 SiteMap。Calibration artifact 同时保存 SiteMap、与其 site axis 对齐的 readout model、及实际 frame contract（camera、sensor shape、ROI、binning、exposure/readout mode）。Occupancy 用 readout model 提取各 site 读数并套 threshold。

`SiteMap` 不是 plot kind，UI 中也不存在 `SiteMap Plot`。它是 Calibration artifact 里的 domain data。固定的 `Image` plot kind 有 `Site overlay` 参数，需要时用 SiteMap centers 画 site/occupancy markers。绘图圈半径是 Image display 属性（可由 site spacing 自动给出），不与科学 integration half-width 混为一个参数。

输出文件由 Experiment workspace 选择目录和不重复文件名，例如 `calibration_20260808_153012_01.json`。这只是避免依赖 process cwd 和无提示覆盖，不是版本/安全机制。

### 8.2 Camera Measurement

Edit 中的理想参数：

| 参数 | 语义 |
|---|---|
| Camera | 从所有 `BaseCamera`-compatible named instances 中选择 |
| Exposure | 本次 measurement 的 exposure，传给选中 camera |
| ROI | 本次 measurement 的 sensor ROI，可由 Image selector 反写 |
| Repeat | `0 = infinite`，正数为 finite cycles |
| Frames per cycle | 每个外部时序 cycle 期望的 frame 数 |
| Trigger/read timeout | 放在 Advanced，只在设备/调试需要时显示 |

Camera Measurement 不驱动 pulse。它监听外部时序，独占 camera、最多只读 sequencer 状态。Driver/internal buffer 大小不进用户表单。

### 8.3 Occupancy Processor

Edit 中的理想参数：

| 参数 | 语义 |
|---|---|
| Frames signal | 显式选择一个 contract-compatible image/frame signal |
| Calibration file | 显式选择 calibration JSON path |

没有 device、finite/infinite mode、buffer 或 loss 参数。Start 时加载 calibration，核对 frames 的 sensor/ROI/binning/exposure/readout contract，然后按 SiteMap + threshold 产生 per-site counts/occupied/rate 等 dataset。Finite source 顺序处理/可处理已完成的 frozen dataset；infinite source 只处理 latest。

## 9. TaskConsole Logic UI

### 9.1 Add/Edit 生命周期

- `Add Logic` 只选 node type，创建 stopped row，不弹 modal 追问 source。
- Add 成功后立即切换到对应 Logic Edit tab，与 v1 的交互一致。
- Logic Edit 实时编辑 row draft，包含本 node 的所有 measurement/task/processor 参数和 input binding。
- 按钮是 `Start/Restart`、`Stop`、`Remove`；没有空泛的 Logic `Apply`。
- 新 Logic Node 启动时按第 3 节的 claim 规则停掉占用冲突设备的旧 node。

### 9.2 三个 node 在 Edit 中的字段

| Node | 字段 | 明确不显示 |
|---|---|---|
| Calibration | Camera instance；Sequencer instance；Calibration pulse/protocol；Samples/repetitions；Reference exposure；Readout exposure；Camera ROI；threshold method；site integration half-width；按 method 显示的 reducer/PSF 参数；必要的高级检测参数；完成后只读 Detected sites | grid rows/columns/site count；`bracket`；output signal |
| Camera Measurement | Camera instance；Exposure；ROI；Repeat (`0=infinite`)；Frames per cycle；Advanced timeout | 独立 mode；user buffer；loss 计数；pulse drive |
| Occupancy | Frames signal；Calibration file；只读输出摘要 | Device；mode；buffer；隐式“current calibration” |

Calibration 中所谓“必要的高级检测参数”只能是算法确实需要暴露的噪声门限、最小间距或 spot 尺度之类调整项；它们不能变相成“用户先告诉 site 数量/形状”。默认自动模式应当不需要用户调它们。

## 10. Plot Panel UI

### 10.1 Add Panel

Add Panel 时选定 Signal 和 Plot kind。Plot kind 一旦创建就固定；Setting/Edit 只读显示，需要另一种 kind 就新建 panel。Facet Grid 的 cell kind 也在 Add 时固定。

### 10.2 Setting frame

Setting 是 monitor board 上的快速配置，包含：

- Plot kind（只读）；
- Signal；
- Panel size；
- Update interval；
- title/labels 和当前 kind 最常用的 display 参数；
- Edit / Remove。

改 Signal 只换这个 panel 的绑定，不改 Occupancy 等 Logic Node 的 input binding，也不改 plot kind。

每个 panel 只有一份 Workbench-owned `PanelState`，其中包含 signal binding、size、update interval、plot semantic/display/fit 参数和固定 plot kind。Setting frame 和 Panel Edit 都是这一份 state 的 view/controller，不各自保存副本。

### 10.3 Panel Edit tab

Edit 是一个 tab，不是 modal。它包含：

- Plot kind（只读）、Signal、Panel size、Update interval；这三个可编辑字段与 Setting 重复显示，两处直接绑定同一 `PanelState`；
- 当前图形和 `Refresh snapshot`；
- 完整 semantic/display/fit 参数和结果；
- selector/zoom/pan；
- direct producer 的完整 Logic parameter form，它是 producer row draft 的另一个投影；
- Producer `Apply`：就是用当前 draft 重启 producer measurement；
- `Save Fig`：保存这个 panel 当前图像和对应数据。

这种重复是有意的：用户在图上做 Area/range selection 时，可以同时看到 ROI/range 等 producer 参数更新，然后就地 Apply。

Setting 或 Edit 从任一边提交修改时，controller 替换同一 `PanelState`，两个 view 和 monitor panel 都收到同一次更新。不写“Setting -> Edit”和“Edit -> Setting”两套手工拷贝逻辑。Edit 中的 frozen data snapshot 与 `PanelState` 分开：参数始终同步；如果换了 signal，旧 frozen 图标为 stale，用户 Refresh 后取新 signal 的 snapshot。

### 10.4 各 plot kind 的理想参数

| Kind | Edit 中的 semantic 参数 | Edit 中的 display/interaction | Setting 中的快速子集 |
|---|---|---|---|
| Curve | X axis；Group by；Reduction | labels/units；grid；limits；X-range selector；compatible fit | title；X/Y labels；grid；limits |
| Image | X/Y axes；Reduction；`Site overlay = Off / Centers / Occupancy` | colormap；color limits；interpolation；colorbar；site labels；marker radius/style；empty/occupied/invalid colors；Area selector；2-D fit | title；colormap；color limits；colorbar；Site overlay on/off |
| Histogram | value/reduction selection | bins；density；cumulative；log Y；range selector；compatible fit | title；bins；density；log Y |
| Rolling | Group by；Reduction | window；Y limits；side distribution；X-range；compatible fit | title；window；Y limits；grid |
| Facet Grid | Facet axis；fixed cell-kind semantic parameters | packing；focus cell；cell selector；compatible per-cell fit | title；facet unit；packing |

Fit model 和参数兼容性由 `zlc_plot` 声明，UI 不写死列表。`Site overlay` 是 `Image` plot kind 的参数：`Centers` 使用 calibration SiteMap centers，`Occupancy` 再结合当前 occupied/valid 值绘制状态，不从 grid shape 生成圈。

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
- 如果该 `Image` panel 开启了 `Site overlay`，重画所需的 resolved centers/status 作为该 Image plot 的 data/annotation 保存；不创建 SiteMap plot kind；
- calibration 在调用链中按实际 `calibration_path` 记录，不内嵌另一份 calibration JSON，不考虑我之前臆造的“移动 panel 数据”场景；
- 不新增 fingerprint/hash。

Calibration Task 自身的 JSON/report 是第四个业务文件，但不是 TaskConsole Save 按钮：它在 Task 成功时由 workspace 自动写出。

## 12. 状态更换时的行为

| 动作 | 设备 | Signal | Panel | Processor |
|---|---|---|---|---|
| 修改 draft | 不变 | 不变 | 继续显示旧 run | 继续处理旧 run |
| Logic Restart / Producer Apply | 停冲突 owner，用新 request 重配并启动 | key 不变，generation 更换 | 保留绑定，新数据到达时替换 host | active row 重新校验；不相容则 blocked |
| 新 Logic Node 占用同一 device | 旧冲突 node 被停止 | 旧 generation terminal | 可保留明确 frozen/stale 图 | 依赖该 source 的 active processor 停止/等新 source |
| Panel 换 Signal | 不变 | Logic signal 不变 | 该 panel 换绑 | 不变 |

## 13. 当前实现状态

本文定义的产品链已经在当前树实现：role 与 extent 正交；finite `FollowTap`、retained final 和 infinite latest 三条 processor 路径均由 source 数据状态决定；Camera Measurement 支持 `Repeat=0` 以及 per-run exposure/ROI request；Calibration 自动发现 sites、只写 workspace JSON artifact；Occupancy 显式使用 frames 与 calibration path；Logic/Panel Edit、selector、Producer Apply、固定 plot kind、共享 `PanelState`、Image Site overlay 和三种 Save 语义均已闭环。

正式入口是根 `bin\experiment.bat`。它只启动一个 Qt/Python experiment flow：Device Manager `Init devices` 持有唯一 session，随后在同一进程中同时显示 TaskConsole 与 Pulse UI，二者使用同一个 sequencer/world；退出按 ownership 顺序清理，不留下第二个进程或设备 owner。实现与实测证据记录在 `IMPLEMENTATION_PLAN.md` 的 Checkpoint；本节不另立待办清单。

## 14. 本轮 review 的结论和非问题

以下已经定案，实施时不应再当成待决定项：

- Calibration 自动发现 sites；无 grid shape/count 输入。
- exposure/ROI 是 measurement 参数，由 UI 接线层维护共享 draft 并在 Start/Apply 时传给 device。
- `Repeat=0` 是 infinite。
- Add Logic 后自动进 Edit tab。
- Plot kind 在 Add Panel 时固定。
- Panel Edit 重复显示 panel 参数和 direct producer 参数，selector 更新同一 measurement draft，Producer Apply 立即重启 measurement。
- Setting frame 和 Panel Edit 直接绑定同一 `PanelState`，对应参数天然双向同步；不存在两份 panel config。
- SiteMap 是 calibration domain data，不是 plot kind；site/occupancy circles 是固定 `Image` kind 的 `Site overlay` 参数和 annotation。
- 设备可多方只读，只有 exclusive Logic Node 单占；新冲突 node 停旧 node。
- TaskConsole/Pulse Editor 使用同一 Experiment session，不引入 IPC。
- Header Layout Save、Header Screenshot 和 Panel Save Fig 三者语义分开。
- Panel archive 不复制 calibration JSON，不增 hash/fingerprint。

实施期间遇到任何未预见问题、新矛盾或现有代码与本架构冲突时，不停下询问、不把决策退回给用户。执行者必须按“用户已裁决的产品语义 > 本架构文档 > 整条科学数据链正确 > 最简单可维护的实现 > v2 现状 > v1 参考”自主作出最优决定，在实施记录中简要记录理由，然后继续运行直到交付定义全部满足。v1 或当前 v2 的错误实现不能重新打开本文已定案的决定。
