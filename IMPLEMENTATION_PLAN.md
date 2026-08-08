# Zou_lab_control v2 实施计划（Complete）

> 这是 [ARCHITECTURE_DESIGN.md](./ARCHITECTURE_DESIGN.md) 的实施顺序，不是已完成清单。
> 仓库绝对路径：`C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2`
> 本文绝对路径：`C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\IMPLEMENTATION_PLAN.md`
> 设计审查基线：`0243aa6`；实际执行 HEAD 以本文 Checkpoint 为准。先前完成声明曾因真实 `experiment.bat` 产品入口失败而撤销；Phase 12 已纠正入口并重新验收。
> 目标权威是封闭集合：只有本文和绝对路径 `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\ARCHITECTURE_DESIGN.md`。根 `HANDOFF.md`/`README.md`、七份 `packages/*/GOAL.md`、package contracts/README 以及其他旧 design/goal 文档都不是实施指令；冲突时忽略旧文档。

## 持续执行 Goal

> 在绝对路径 `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2` 的当前树上，严格按 `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\ARCHITECTURE_DESIGN.md` 和 `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\IMPLEMENTATION_PLAN.md` 持续实施，直到虚拟 Calibration -> Camera Measurement -> Occupancy -> Image/Panel Save Fig 通过正式 descriptor/catalog/NodeHost/TaskConsole 路径端到端跑通；同时完成 `Repeat=0` infinite、finite FollowTap/frozen final dataset 处理、Calibration 自动发现 sites 并写 workspace JSON、exposure/ROI measurement request 与 selector -> shared draft -> Producer Apply/Restart 闭环、OBSERVE/EXCLUSIVE 设备仲裁、Add Logic 后自动 Edit、固定 plot kind、Setting/Edit 共享 PanelState、Image Site overlay，以及 Header Save Layout / Header Save Screenshot / Panel Save Fig 三条互不混淆的保存路径。执行期间遇到任何未预见问题、新矛盾或现有代码与目标架构冲突时，不停下询问、不把决策退回给用户，而是按“用户已裁决的产品语义 > 本架构文档 > 整条科学数据链正确 > 最简单可维护实现 > v2 现状 > v1 参考”自主作出最优决定，简要记录理由并继续。除上述两份绝对路径文档外，不把任何 `GOAL.md`、HANDOFF、README、contract 或旧 design 当作目标规格；冲突时忽略旧文档。不增加 fingerprint/hash、loss telemetry、防御型框架或测试矩阵；三条新纵向 guard 必须先证明在原缺陷下变红。所有受影响 package tests、Guard A/B/C 和全树测试通过，两种 virtual run 可正常 stop/close 且无悬挂 worker/device claim，架构文档、实施计划 Checkpoint 和相关产品文档与实现一致时，才标记完成。

## 上下文压缩/自动续跑恢复协议

不依赖对话摘要保留文档细节。每一个 Goal 执行 turn（包括首次启动、自动 continuation、上下文压缩后、handoff 后和任何重新进入任务时），在修改代码前必须执行：

1. 将工作目录确认为 `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2`。
2. 从头到尾完整重读两份续跑权威：
   - `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\ARCHITECTURE_DESIGN.md`；
   - `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\IMPLEMENTATION_PLAN.md`。
3. 读取本文的“持久执行 Checkpoint”，再实测 `git status --short`、当前 HEAD、相关 diff 和最近测试结果。对话摘要与磁盘冲突时，以当前磁盘和这两份完整文档为准。`HANDOFF.md` 只是设计阶段的历史输入，不是续跑权威。
4. 从 Checkpoint 的下一个未完成步骤继续；不从头重做已完成 phase，不凭压缩后的记忆重构计划。
5. 任何 Python 验证仍必须遵守：脚本第一行 `import zou_lab_control_v2`，并在断言前打印被测生产模块 `__file__`。
6. 不在恢复时搜索或读取其他 `GOAL.md`/design/contract 来重建目标。若为定位现有 API 而读取它们，只把内容当历史现状证据，不接受其中与两份权威文档冲突的目标或限制。

执行者必须在每个 phase 完成后、开始长时间测试前，以及做出新架构裁决后，立即更新下面的 Checkpoint。这使得续跑不需依赖未压缩的对话历史。

## 持久执行 Checkpoint

> 该区块在 Goal 启动后由执行者持续更新，是续跑的磁盘事实，不是用户需要维护的表单。

- Goal status：`complete — product path, unified experiment entry, staged commits, clean-worktree verification and documentation are complete`
- Repository HEAD at checkpoint：`cb6ab87 Align pulse tests with held-point playback`
- Completed phases：`Phase 0 authority quarantine；Phase 1 role/extent；Phase 2 finite/infinite data path；Phase 3 Camera request + publication run record；Phase 4 Calibration artifact；Phase 5 Occupancy + Image overlay；Phase 6 headless Guard A；Phase 7 TaskConsole Logic Add/Edit/Start；Phase 8 PanelState/selector/Producer Apply；Phase 9 three Save paths/viewer；Phase 10 virtual product path；Phase 11 scoped code cleanup；Phase 12 operator entry/shared session/staged delivery implementation`
- Current phase：`complete`
- Last completed action：根 `bin\experiment.bat` 已恢复为唯一统一 experiment flow；v1 形态的 Device Manager `Init devices` 在同一进程/session 中打开 TaskConsole + Pulse UI，Pulse UI 使用 exact session sequencer。功能改动已经按依赖拆成五个提交：`2388fc4`、`3398139`、`451a64c`、`5951bd2`、`cb6ab87`。
- Last verified tests：当前主树先打印本树 root/atom/runtime/UI/Workbench/TaskConsole module 路径，随后全树 `1105 passed`；Guard A/B/C 单列 `3 passed`。Detached worktree 复验为：`2388fc4` Device Manager `21 passed`，`3398139` runtime `142 passed`，`451a64c` atom `132 passed`，最终 `cb6ab87` Pulse `137 passed`、Workbench `303 passed`。真实 batch 可见 smoke 确认第一屏只有一个 `Devices@Zou lab` Python GUI；Qt product-flow 测试确认 Init 后双窗口、同一 sequencer/world 的 On Pulse/Stop、close/recreate；所有临时 worktree 已移除，复核后为 `NO_ZLC_GUI_PROCESS`。
- Next action：`none within this Goal；真机/FPGA 现象需在硬件重新到手后另行实测`
- New decisions since architecture review：旧 `GOAL.md` 的误导风险不留到收尾处理；Phase 0 先隔离目标权威，Phase 11 按最终实现同步历史内容。Calibration 普通 measurement exposure 由 adapter request 决定；标定 long/readout/long 窗口由 Task 显式 protocol 参数处理。Panel archive 不持久化 schema 可导出的冗余 digest。Panel Edit 的操作图面是 Workbench 管理的独立 frozen plot host，不复用 monitor QWidget、不建立第二条 runtime derived signal。Layout 中 typed plot choice 以 plain JSON token 保存，并经当前 host 声明的 choices 恢复。Close 超时是 ownership 未释放而非可忽略错误，因此保留 window/bindings 并拒绝 session teardown。旧 FPGA asset SHA 清单因冻结 checkout 换行而无法在 clean worktree 重现，已改为资产存在性、解析后几何相等和既有 wire/launcher 语义验证，不再用字节 hash 代替行为。

## 1. 执行纪律

1. 任何临时 Python 脚本第一行必须是 `import zou_lab_control_v2`，并在断言前打印实际被测生产模块的 `__file__`。
2. 每个现状结论都在当前树重新实测，不用 pip 副本、记忆或 v1 代替现状。
3. 新守卫先在原缺陷下运行并记录失败，再修代码。无法在原树重现红色的测试不作为新守卫合入。
4. 默认改现有测试，不建立 device x field x plot-kind 矩阵。整个项目只计划增加三条产品级纵向守卫，见第 3 节。
5. 不添加 fingerprint、SHA/hash、content digest、防篡改协议或兼容 alias。
6. 不把 v1 的错误当规格。优先实现用户已经裁决的产品语义。
7. 工作树中无关的已有修改一律保留，不 revert、不借机整理。
8. 在已授权的 repo 范围内持续执行。遇到任何未预见问题、新缺陷、架构矛盾或当前/v1 代码与目标冲突，都按 Goal 中的优先级自主做出最优且可维护的决定，记录理由并继续到交付定义满足；不停下询问，不把决策退回给用户。
9. 不把任何 `packages/*/GOAL.md`、旧 design、README、contract 或 HANDOFF 当作实施目标。它们只能用来理解现状；一旦冲突，以本计划和 `ARCHITECTURE_DESIGN.md` 为准。分派任何子任务时也必须明示携带这个权威边界。

## 1.1 Phase 0：隔离错误目标入口（已完成）

在任何产品代码修改前，七份 `packages/*/GOAL.md` 的首屏均明确标成 historical/inactive，并指向两份绝对路径的执行权威；Phase 11 又把旧 active 正文全部收缩为 tombstone。它们不再保存路径、状态、冻结命令、TODO 或另一份设计，本 Goal 不允许重新扩写第三个目标入口。

## 2. 范围和优先级

### P0：虚拟链真正跑通

- Calibration 自动发现 sites，写出 SiteMap + readout model JSON/report，不发 signal。
- Camera Measurement 通过正式 descriptor/host 支持 finite 和 `Repeat=0` infinite，exposure/ROI 在 request 中传到 virtual camera。
- Occupancy 显式接收 frames signal + calibration path，finite 顺序，infinite latest。
- 同一 Experiment/session/SimulationWorld 中跑通 Calibration -> Camera Measurement -> Occupancy。

### P1：产品 UI 闭环

- Add Logic -> 自动 Edit tab -> Start/Restart。
- capability-filtered device selector，Camera Measurement 可选 `camera`、`mot_camera` 等实例。
- Panel Edit 共享 producer draft，selector 改 ROI/range，Producer Apply 立即重启 producer。
- Plot kind 在 Add Panel 时固定，Setting/Edit 参数表面完整。

### P2：Save 和边界收尾

- Header Save Layout、Header Save Screenshot、Panel Edit Save Fig 三条路径分开。
- 修正 provenance 捕获时机和内容，不扩展为 whole-board archive。
- 只清理本路径触及的过期 contract/Qt 泄漏。

## 3. 测试策略：只增加三条纵向守卫

### Guard A：headless virtual chain

从 descriptor/catalog 创建 Calibration、Camera Measurement 和 Occupancy，使用同一 virtual camera/sequencer/world，真正写/读 calibration JSON，最后得到 per-site occupancy。该 guard 同时守住 role/extent、Calibration artifact、finite processor 和 infinite measurement 四个已定案边界，不把每个断点又拆成一条新 E2E。

原树预期会红的直接原因：Camera Measurement descriptor 不接受 `Repeat=0`、Calibration 仍发 signal/不写 task artifact，Occupancy saved-reference 路径未接通。实施时仍必须先在当时树上实跑记录失败。

### Guard B：TaskConsole interaction chain

一条产品流覆盖：Add Camera Measurement -> 自动 Edit -> 选 camera/设 exposure+ROI -> Start -> Add Image Panel -> selector commit 更新同一 ROI draft -> 单次 Producer Apply -> 旧 measurement 停止且新 run 已启动。断言 signal key 不变、generation 更换。

这一条同时守 Add 后进 Edit、measurement 参数属性、selector 联动和 Producer Apply=Restart；不为每个字段各建 GUI test。

### Guard C：Save product semantics

同一 TaskConsole 含 camera -> occupancy + panel，一条流程验证：

- Header Save Layout 没有 dataset，load 后接线/plot kind 一致且 nodes stopped；
- Header Save Screenshot 只产生一张 GUI image；
- Panel Save Fig 只包含该 panel 的 image/data、run 参数和 actual device snapshots，不包含其他 panel arrays。

其余字段校验、adapter 转换、dataset builder 等继续使用或修改现有单元测试。

## 4. Phase 1：修正 Logic role/extent 和 host 分流

### 工作

1. 删除 `NodeRole -> _RUNTIME_KIND` 硬映射及其生产消费者。
2. 保留 `measurement/task/processor` role；measurement run 根据实际 request 决定 finite/infinite。
3. `camera_measurement.repeat` 许可 0，`0` 唯一表示 infinite。
4. Task 从 signal host 分支移出，只返回 artifact/report result。
5. Processor 绑定 source 后才根据 source extent 选消费路径。

### 完成标准

- Task 不再被询问 finite/reactive。
- 同一 Camera Measurement descriptor 能构建 `Repeat>0` 和 `Repeat=0` 两类 run。
- Processor 不因 `NodeRole.PROCESSOR` 被固定成 latest-only。

## 5. Phase 2：接通 finite/infinite 数据路径

### 工作

1. finite measurement 使用 exact reservation + dataset builder。
2. finite-source processor 接 `FollowTap`，按提交顺序无损处理。
3. source 已结束时，让 processor 可对 retained final `OwnedSnapshot` 处理一次，不重跑设备。
4. infinite Camera Measurement 在自己的 worker 上读 camera 并覆盖 latest slot；UI beat 只从 plane freeze。
5. infinite-source processor 只处理当前 latest，不追历史。
6. 删除 `missed_events/current_gap/behind/missed` 等 loss telemetry；保留 keyed sweep 断续时清 stale cells 的科学正确性规则。

### 完成标准

- finite occupancy 不会因 latest slot 覆盖丢格子。
- infinite 路径不暴露丢失数字给 UI/archive。
- 阻塞 device read 不会出现在 UI beat/freeze/render 路径。

## 6. Phase 3：设备访问与 Camera Measurement request

### 工作

1. 复用/补齐 session 的 `OBSERVE`/`EXCLUSIVE` 访问语义。多个 observer 可与一个 exclusive logic owner 共存。
2. Start 先 validate/build 新 request，再仅停止占用同一底层实例的冲突 Logic Nodes。
3. Camera Measurement 声明 Camera `EXCLUSIVE`；若需 sequencer 状态只声明 `OBSERVE`。Calibration/Scan 在真正驱动期间声明 camera/sequencer `EXCLUSIVE`。
4. Pulse Editor 继续使用 Experiment 中的同一 sequencer，不登记长期 Logic owner，不新建 IPC/session service。
5. Camera node descriptor 只按 `BaseCamera` capability 过滤 named instances，不硬绑实例名 `camera`。
6. 将 camera instance、exposure、ROI、repeat、frames per cycle 放进 `CameraMeasurementRequest`。
7. adapter 负责 exposure/ROI 合法性、increment snapping、SDK 写入和 actual readback；Workbench 不写 DCAM/Pylon/virtual 分支。
8. run 创建时冻结 authored request + actual device snapshot，为 Panel Save Fig 提供真正的调用链状态。

### 完成标准

- `camera` 和 `mot_camera` 都可作为 Camera Measurement 选项，非 Camera 不出现。
- observer 不阻止 measurement。启动 Calibration 会停掉占用同一 camera 的 Camera Measurement，而不停无关 node。
- virtual adapter 确实收到 exposure/ROI，数据 schema/frame contract 来自 actual readback。

## 7. Phase 4：重做 Calibration artifact，不接受 grid 真相

### 工作

1. 从 Calibration authoring、request、session resolver 和 algorithm 调用链中同时删除 grid rows/columns/site count。不只隐藏表单而在底层继续注入。
2. 使 detector 从 calibration image 自动输出 `N` 个 centers，`N=len(centers)`。去重/排序/质量评估不依赖“恰好 rows*columns 个峰”。
3. 定义简单 `SiteMap`：`site_ids`、`centers_xy`、`valid_sites`、`coordinate_frame`、可选自动 topology/order；无 `grid_shape`。
4. 定义与 `site_ids` 对齐的 readout model：integration features/PSF、thresholds、usable/quality。不把 integration 框和绘图圈半径塞进 SiteMap。
5. Calibration Edit 表单使用 camera、sequencer、pulse/protocol、samples、reference/readout exposure、camera ROI、threshold method、site integration half-width 和条件 reducer/PSF 参数。不显示 `bracket`。
6. artifact 保存 SiteMap + readout model + frame contract + report facts；删除 calibration/report signal `OutputSpec`。
7. Task 成功后由 workspace 选择目录和唯一文件名，原子写 plain JSON/report，不依赖 cwd，不静默覆盖。
8. 删除该路径上的 fingerprint/hash 生成和相容性分支。

### 完成标准

- 改变 virtual world 中的 site 数量时，不改 Calibration request，detected N 跟着图像变化。
- 输出 JSON 中 `centers[i]`、readout feature/threshold/validity 都由同一 `site_id` 对齐。
- signal catalog 中没有 calibration/report，workspace 中有新 artifact path。

## 8. Phase 5：Occupancy 正式输入、输出和 overlay

### 工作

1. Occupancy authoring 包含 frames stable signal key + calibration file path，删除 session 隐式“current calibration”注入。
2. 实现 saved calibration path resolver；Start 时加载 plain JSON，校验 frame contract 后才绑 source。
3. 使用 readout model 对每帧生成对齐的 `counts[N]`、`occupied[N]`、`valid[N]`，需要时生成 rate，并保留同一个 `frame_judged`。
4. 同一次 processor publication 中的 sibling outputs 共享直接 parent，不用全局 shot id 猜同步。
5. 不新增 SiteMap plot kind。固定 `Image` plot kind 的 `Site overlay` 参数决定 Off/Centers/Occupancy；Workbench/plot 用 `frame_judged` + SiteMap centers + occupied/valid status 组成 typed Image point overlay，绘制 empty/occupied/invalid markers。圈大小是 Image display 属性，不改科学 integration area。
6. finite 上游走 FollowTap/最终 frozen dataset，infinite 上游走 latest。用户不选该模式。

### 完成标准

- 无 calibration path 或 contract 不相容时 Occupancy 不发数据，显示可修复的 stopped/incompatible。
- 只有 JSON path 也可 Start，不需要 session 中刚跑过的 live Calibration object。
- Image panel 的 circles 来自 SiteMap centers，不来自 grid shape。

## 9. Phase 6：跑 Guard A，完成 headless P0

1. 在同一 Experiment/session 安装 virtual camera + virtual sequencer，确认两者指向同一 `SimulationWorld`。
2. 通过 descriptor/catalog/host 运行 Calibration，从图像自动得到 sites 并写 JSON。
3. 运行 Camera Measurement finite 和 `Repeat=0` infinite，每次都让 request 中 exposure/ROI 实际到达 virtual adapter。
4. 运行 Occupancy，显式传 frames key + JSON path，验证 counts/occupied/valid 与 SiteMap 对齐。
5. 停止所有 workers/nodes，释放 exclusive claims，关闭 session，不留阻塞读或后台线程。

P0 在 Guard A 通过且所有受影响 package 现有测试通过时结束。

## 10. Phase 7：TaskConsole Add/Edit/Start 产品路径

### 工作

1. `Add Logic` 只创建 stopped row，删除 Occupancy Add 的 modal signal picker。
2. Add 后立即打开/聚焦对应 Logic Edit tab。
3. 建立唯一 row draft，包含 node parameters、input binding 和 capability-filtered named device choices。
4. Logic Edit 只用 Start/Restart、Stop、Remove，不加通用 Apply。
5. Start/Restart 冻结当前 draft -> validate request -> 停冲突 Logic Nodes -> measurement/task 启动。
6. Occupancy signal selector 按 contract 过滤；在尚无 publication 时可保留 unresolved stable key，不强制 Add modal。

### 完成标准

- Add Calibration/Camera/Occupancy 都自动进各自 Edit。
- exposure/ROI 显示在 Camera Measurement/Calibration 参数中，不是另一份 Device Manager draft。
- 新 Calibration Start 会在验证成功后停掉占用同 camera 的 measurement。

## 11. Phase 8：Panel Setting/Edit、selector 和 generation replacement

### 工作

1. Add Panel 时固定 plot kind（Facet Grid 同时固定 cell kind），Setting/Edit 中只读，无任何切换路径。
2. 每个 panel 建立唯一 Workbench-owned `PanelState`，包含 signal/size/update interval/fixed kind/semantic/display/fit。Setting frame、Panel Edit 和 monitor panel 都订阅它，不保留独立 config 副本。
3. Setting frame 实现 Signal、Size、Update interval、title/labels 和 kind 常用 display 参数。
4. Panel Edit 作为 tab，重复显示 Signal/Size/Update interval，并提供完整 plot parameters、fit、selector、direct producer form、Producer Apply 和 Save Fig。两边的共同字段直接绑定同一 `PanelState`；任一边修改都由同一 controller 发布一次更新给所有 views。
5. Logic Edit 和同 producer 的 Panel Edit 共享同一 row draft；一处修改同步到其他打开投影。
6. 只处理 committed selection。Logic descriptor 用 data-only mapping 把 Image Area 等转成 typed measurement draft patch，Workbench 负责路由，不写 camera-specific branch。
7. Producer Apply 直接调同一 Start/Restart endpoint，一次按键完成停旧 run -> 配置 request -> 立即启动新 run。
8. node id/signal key 保持不变，成功启动创建新 generation。Panel 在 generation boundary 替换 plot host；同 generation 内复用 snapshot revision 拒绝晚到异步结果。不新建第二套 revision。
9. active downstream 保留 row/binding 并对新 source 重校验；ROI/exposure 使 calibration frame contract 不相容时显示 blocked。

### 完成标准

- selector -> ROI draft 在非零 origin/binning 下仍使用正确 sensor coordinates。
- 只按一次 Producer Apply 就已运行新 measurement，不再要用户按 Start。
- Guard B 通过，不增加字段级 GUI 测试矩阵。

## 12. Phase 9：实现三种 Save

### Header Save Layout

1. 保存 nodes/row drafts/named device choices/signal wiring 和 panels 的 signal/fixed kind/spec/size/interval/order。
2. 不 freeze dataset，不保存 running state、generation/revision、last error 或 device snapshot。
3. Load 恢复 stopped drafts 和接线，不打开/配置设备。缺少的 device/signal/path 保留为 unresolved 供用户修复。

### Header Save Screenshot

1. 截取整个当前 TaskConsole GUI 为普通 image。
2. 不保存 layout、NPZ、panel data 或 provenance。

### Panel Edit Save Fig

1. 对 Edit tab 正在显示的同一 frozen snapshot 保存 image + data archive；不在 Save 时另抓 latest。
2. archive 仅含该 panel 的 dataset/schema/validity、plot state/fit 和重画所需 overlay annotation，不含其他 panels。
3. 调用链 metadata 使用 run 时已冻结的 authored parameters、named devices、actual public device snapshots、pulse/sequencer snapshot（当该链确实使用时）和 Occupancy calibration path。
4. 不在 Save 点击时才首次捕获 device state。
5. 不内嵌 calibration JSON 副本，不增加 fingerprint/hash。如果该 `Image` panel 的 `Site overlay` 已开启，仅把重画当前 Image 所需的 resolved centers/status 当作该 panel 的 data/annotation 保存；不创建 SiteMap plot kind。

### 完成标准

- Guard C 通过。
- `numpy.load(..., allow_pickle=False)` 仍可读 Panel data archive，viewer 可用同一 plot state/overlay 重画。
- 没有 whole TaskConsole/monitor tab 的 panel 数据打包动作。

## 13. Phase 10：正式 TaskConsole 虚拟产品路径

在空 workspace 中用真 UI intent 执行：

1. `Add Calibration` -> 自动 Edit -> 选 virtual camera/sequencer/pulse，设 reference/readout exposure + camera ROI -> Start -> 自动发现 sites -> 得到唯一 calibration JSON。
2. `Add Camera Measurement` -> 自动 Edit -> 选 camera -> 设 exposure/ROI -> `Repeat=0` -> Start。
3. 从同一 Experiment 的 Pulse Editor 发出时序，Camera worker 持续产生 frames，GUI beat 不读 camera。
4. `Add Occupancy` -> 自动 Edit -> 选 frames signal + calibration path -> Start。
5. Add 固定 Image/Curve/Histogram 等 panels，验证 Image 用 SiteMap centers 画 occupancy circles。
6. Panel Edit 中 Area selector 改 Camera Measurement ROI draft -> 单次 Producer Apply -> 新 measurement 已运行，旧 calibration 不相容时 Occupancy 显示 blocked。
7. 分别执行 Header Save Layout、Header Save Screenshot、Panel Save Fig；Load Layout 恢复 stopped pipeline。
8. 启动新 Calibration，验证它只停掉占用同 camera/sequencer 的冲突 nodes，observer/无关 node 仍正常。
9. Stop/close，按 Pulse controller -> TaskConsole nodes/workers 与 plot bindings -> session/devices -> Device Manager owner 的所有权顺序有界清理；任一 ownership 未释放都拒绝伪装成成功退出。

这一流程用 Guard A/B/C 和现有 package tests 支撑，不再复制第四套 E2E。

## 14. Phase 11：窄范围收尾

1. 只清理本链接触的跨 package 私有 import、无消费者 `.v1/.v2` signal suffix 和过期 loss 文档。
2. 确保 Qt/widget 不泄漏给 runtime/atom，UI beat 不命中阻塞 device API。优先复用现有 seam test，不为 import 数量写装饰守卫。
3. Phase 0 将七份 package `GOAL.md` 标成 historical/inactive，当前又已把旧 active 正文收缩为 tombstone，并同步 root README、contracts 和 HANDOFF 中“task 发 signal”“infinite 记 loss”“grid shape 是 calibration 输入”“两 GUI 需 IPC”等错误现状说法。这些文件不得重新出现 active TODO/冻结命令来与两份执行权威竞争；文档同步也不得反过来影响已审查架构。
4. 先跑受影响 package tests，再跑 Guard A/B/C，最后跑全树。所有临时脚本都打印本树 module `__file__`。

## 15. Phase 12：真实实验入口与分阶段交付纠正

1. 既有 Device Manager/UI 测试先证明 flat editor + Test-and-release 旧流程会红，随后恢复 v1 可见结构；没有增加第四条 Guard。
2. `Init devices` 直接从当前 draft 建立并持有 session，不要求先 Save；成功后同一进程同时显示 TaskConsole 与 Pulse UI，并隐藏 Device Manager。
3. Pulse UI 借用该 session 的 exact sequencer。Qt product-flow 测试实际执行 On Pulse/Stop，证明它作用于同一 virtual world，且关闭窗口不会另开或关闭第二个 sequencer。
4. 根 `bin\experiment.bat` 本身是唯一正式实验入口（对应 v1 `task_console.bat`）；没有新增根 launcher。可见 smoke 确认第一屏只有一个 Device Manager/Python GUI；自动 product-flow 覆盖 Init 双窗、关闭清理和立即重建。
5. 功能已经按依赖提交：

   | commit | 阶段 |
   |---|---|
   | `2388fc4` | Device Manager 操作流程与 session lifecycle |
   | `3398139` | runtime source extent、FollowTap/frozen/latest |
   | `451a64c` | virtual camera、Calibration artifact、Occupancy |
   | `5951bd2` | TaskConsole/plot/UI/save/shared experiment flow |
   | `cb6ab87` | held-point Pulse 测试对齐、自包含 fixture 与语义化 FPGA asset 验证 |

6. 每次提交均显式审查 staged 文件和 `diff --check`；最终在 detached worktree 中打印被测 module `__file__` 后复验。当前主树全树为 `1105 passed`，Guard A/B/C 为 `3 passed`，临时 worktree 与 GUI 进程均已清零。

## 16. 整体交付定义

以下全部成立才算完成：

- Calibration 不知道 grid shape/count，从数据自动发现 sites，写 SiteMap/readout model JSON，不发 signal。
- Camera Measurement 支持 `Repeat=0` infinite，exposure/ROI 是 request 参数并真正传给所选 camera。
- Occupancy 只用显式 frames + calibration path，finite 无损，infinite latest，Image overlay 来自实测 SiteMap。
- 多个 read-only observer 可并存，一个 device 同时最多一个 exclusive Logic Node，新冲突 node 会停旧 node。
- TaskConsole/Pulse Editor 使用同一 Experiment/session/sequencer/world，Pulse Editor 不被虚构成长期 device owner。
- Add Logic 自动进 Edit；Logic 无空泛 Apply；Panel Producer Apply 就是修改参数后重启 measurement。
- Plot kind 在 Add Panel 时固定；Panel Edit 显示完整 panel + producer 参数并支持 selector 联动。Setting/Edit 共享唯一 `PanelState`，对应参数天然双向同步。
- SiteMap 不是 plot kind；site/occupancy markers 是固定 `Image` plot kind 的 `Site overlay` 参数。
- Header Layout、Header Screenshot、Panel Save Fig 三者分开；Panel data 包含正确调用链参数/device snapshots，不打包整个 monitor tab。
- 不记 buffer loss，不新增 fingerprint/hash，新增纵向守卫不超过 Guard A/B/C 三条且各自已证明原缺陷下会红。
- 七份旧 package `GOAL.md` 均保持 historical/inactive，且任何仍保留的产品说明与最终实测实现一致；仓库中不存在第三个看似 active 的目标入口。

## 17. 当前已知可复用与不可误判的部分

可复用：

- atomic NPZ/manifest 读写和基础 viewer；
- device public snapshot 的基础框架；
- runtime 已存在的 exact/follow/latest/dataset builder 机制；
- plot 中已存在的 typed image point overlay/empty-occupied-invalid 渲染能力；
- `OwnedSnapshot` 既有 revision，不需要第二套编号。

不能误判为已通：

- 当前 archive 基础测试通过不等于三种 Save 产品语义已通；
- calibration 表单隐藏 grid shape 不等于算法已自动发现 N；不允许 session 底层继续偷偷注入；
- `allow_saved_reference` 声明存在不等于 Occupancy path resolver 已接通；
- Monitor/stream/dataset 类存在不等于 Camera Measurement 正式 infinite 产品路径已接通。
