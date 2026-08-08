# Zou_lab_control v2 — 历史交接快照

> **HISTORICAL / INACTIVE。** 本文件原来记录 `99870de` 附近的接手现状；那些缺陷清单、状态和下一步不再是实施指令。
> 凡用户或权威计划明确要求参考 v1，唯一允许读取的树是 `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v1_claude\Zou_lab_control_v1`；其他副本不得代替。v1 不是 v2 的上位规格，目前只明确用于 Device Manager 可见 UI 等点名参考。
> 当前且唯一的目标权威是：
>
> 1. `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\ARCHITECTURE_DESIGN.md`
> 2. `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2\IMPLEMENTATION_PLAN.md`
>
> 以下内容只是当前树的产品快照，不能覆盖上述两份文档的设计、Checkpoint 或完成标准。

## 当前产品快照

- TaskConsole 与 Pulse Editor 是同一 `Experiment`/session 上的两个窗口，共用 named devices、virtual world 和 sequencer；没有第二个 session 或 IPC。
- 设备访问分为可并存的 `OBSERVE` 和同一底层实例只能有一个 Logic Node owner 的 `EXCLUSIVE`。新 request 先完整校验，再只停止与它争用同一实例的旧 node。
- Logic role 与数据 extent 正交。NodeHost 按“有无 source binding”托管 worker 或 processor，不再用 `_RUNTIME_KIND` 从 Measurement/Task/Processor 推出 finite/reactive。
- Camera Measurement 的 `Repeat > 0` 是 finite，`Repeat = 0` 是 infinite。finite processor 使用顺序 `FollowTap`，已结束的 finite source 可处理 retained final snapshot；infinite source 只处理 latest，不记录、展示或保存 missed/gap/loss telemetry。
- Calibration 是 artifact-only Task：它从图像自动发现 sites，不接受 grid rows、columns 或 site count，写出 workspace 唯一命名的 plain calibration JSON（SiteMap、readout model、frame contract、report facts），不发布 calibration/report signal。
- Camera Measurement 的 camera instance、exposure、ROI、repeat 和 frames-per-cycle 是每次 run 冻结的 measurement request；adapter 负责设备约束对齐和 actual readback。
- Occupancy 显式消费 frames stable signal key 与 calibration path，输出同一 parent publication 下对齐的 counts/occupied/valid/rate/frame_judged。SiteMap 是 calibration domain data，不是 plot kind；固定 Image kind 用 `Site overlay = Off / Centers / Occupancy` 显示实测 centers/status。
- TaskConsole header 只有一个 combined Plot/Measurement/Processor/Task 下拉框和一个 `Add Panel` 按钮。Logic entry 经同一按钮创建 stopped row 并自动进入 non-modal Logic Edit；没有独立 `Add Logic` 按钮或 modal chooser。Logic Edit 只有 Start/Restart、Stop、Remove，没有通用 Apply；Panel Edit 的 Producer Apply 使用共享 row draft 调用同一个 Start/Restart endpoint。
- Plot entry 经同一 `Add Panel` 按钮创建固定 kind 的 blank panel，不依赖 signal publication；signal 在 Setting/Edit 中接入。Setting frame 与 Panel Edit 绑定同一份 Workbench-owned `PanelState`；Panel Edit 的 selector 可通过 descriptor data-only mapping 更新 direct producer 的共享 measurement draft。

## 当前保存语义

| 用户动作 | 保存内容 |
|---|---|
| TaskConsole header `Save Layout` | stopped pipeline/layout JSON：row drafts、named device choices、signal wiring、panel binding/fixed kind/state/order；不含 dataset、running state 或 device snapshot |
| TaskConsole header `Save Screenshot` | 当前整个 TaskConsole GUI 的一张普通图片；不含 layout、NPZ 或 provenance |
| Panel Edit `Save Fig` | Edit 正在显示的同一 frozen panel snapshot 的图片、单 panel 数据、plot/overlay state，以及 run 时已经冻结的调用链参数和 actual public device snapshots；不含其他 panels |

Calibration JSON 是 Task 成功时自动产生的业务 artifact，不是上述三个 Save 动作。Panel archive 只记录实际 `calibration_path`，不内嵌 calibration JSON，也不增加 fingerprint/hash。

## 当前验证状态

最终产品证据以 `IMPLEMENTATION_PLAN.md` Checkpoint 为准。当前生产 HEAD `6ad2e70` 的全树回归为 `1114 passed`，Workbench 为 `319 passed`；该提交还通过独立 TEMP detached worktree 的路径探针与 Workbench `319 passed`，验证后 worktree 已删除。正式 `bin\experiment.bat` 的真实控件验收完成 `Init devices -> Calibration -> Repeat=0 Camera -> Pulse Load/On -> Occupancy -> blank Image -> frame_judged -> Occupancy site overlay -> Stop Pulse -> close`，显示六个实测 site circles；结束后两个 ZLC 窗口、session 和项目 Python 进程均无残留。本文件不维护另一份进度表。

## 历史真机观察（本轮未复测）

旧交接曾记录两项硬件相关现象：

- 真 FPGA 上 hold/step 时 DAC 不动而 scan 正常；当时的单点表/forever 假说没有在当前树和真板上重新验证。
- Pulse Editor 曾只能按 `.py` 查找、却把 pulse 保存为 `.json`；该旧现象没有在本轮当前树上重新验证。

它们是历史实验记录，不是当前 Goal、当前缺陷断言或实施待办；若以后重新进入真机范围，必须先在当时的树和硬件上重新实测。
