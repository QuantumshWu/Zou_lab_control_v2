# Zou_lab_control v2 — 交接文档

> **状态快照，不是计划书。** 记录截至 `99870de` 的实测状况、已定裁决与已知问题。
> 第 4 节的裁决是用户当面确认的，优先于仓库里任何历史文档。

## 0. 路径

| 什么 | 在哪 |
|---|---|
| **v2 工作树（唯一编辑处）** | `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v2` |
| **v1 单体（批判性参考基准）** | `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v1_claude\Zou_lab_control_v1` |
| 旁边八个 `zlc_*` 独立仓 | 历史，**不是第二个编辑处**；pip 装的是它们，会抢 import |

**本树不装 pip。** 靠 `import zou_lab_control_v2` 抢先把它放到路径最前。
**任何脚本第一行 import 它，并打印被测模块的 `__file__`** —— 否则你量的是 pip 装的另一份代码。

---

## 1. 目标

取代 v1 成为实验室日常控制软件，在真装置上产出带 provenance 的数据。
八层架构的价值全部兑现在「下一个新物理实验做起来便宜」—— v1 能用，但它的
task_console 长到 9461 行，每加一个实验代价越来越高。

**当前阶段：把虚拟链路整条跑通**（硬件不在手上）。
最小台架：virtual camera + virtual sequencer + 三个 logic node
（calibration / camera_measurement / occupancy）。

---

## 2. 八层

依赖自上而下，上层不知下层：

`zlc_data`（dataset 是什么：轴 / 有效性 / 快照 / manifest）
→ `zlc_durable`（原子写文件）
→ `zlc_runtime`（生成期、发布、采集流、节点托管）
→ `zlc_plot`（绘制、拟合、面板布局）
→ `zlc_ui`（窗口与控件，**Qt 不许漏出这层**）
→ `zlc_pulse`（序列模型、编译器、下板）
→ `zlc_atom`（物理：节点、设备、标定）
→ `zlc_workbench`（只做组合与接线）

两条机械强制：

* **每个窗口一次调用换一个不透明句柄**，外部零 QWidget（`test_gui_seam.py` 守）
* **每个包只走门面**，跨包子模块 import = 0

入口 `bin/experiment.bat` → device_manager 配置 → **task_console 与 pulse editor
两个独立进程**。

---

## 3. 数据与执行模型（地基，先读这节）

**signal 汇聚在 `SignalDataPlane`。UI 只读它（`freeze()`），永不读设备。**
实测：`beat()` 里零设备接触。

### 同发保证

刻意不设全局 shot 号 —— `plane.py` 模块头写着「重新引入一个会是虚构」。保证来自：

* 一次 `SignalPublication` = 一个原子兄弟包 → 包内信号天然同发
* 派生值带 `direct_parent_refs`，**精确指回**为产出它而消费的父事件
* 源与其活跃后代**一起替换** → 慢的 processor 不会把源 N 版与自己 N−1 版并排暴露
* **跨独立生产者的同发不被断言**（不同 run 各自推进）

### 三种接法

丢不丢由**接法**定，不由消费者能力定：

| 场景 | 接法 | 丢 |
|---|---|---|
| finite measurement：跑前建好 array，逐格填 | `ExactReservation` → `DatasetBuilder` | 不丢 |
| infinite measurement：维护 buffer，覆盖即丢 | `MonitorTap` → `MonitorDataset` | 丢，记 `missed_events` |
| processor 接 **finite** 上游：顺序处理不许丢 | **`FollowTap`** | 不丢 |
| processor 接 **infinite** 上游：处理 latest | `MonitorTap`（复用） | 丢在源头 |

四个格子、三种接法，无冗余。
**`FollowTap` 目前零调用者** —— 因为第三格从来没法表达，不是有人忘了接。

### 两种 coverage

* `DatasetCoverage(written_cells, total_cells)` = 预分配，有 `complete`
* `MonitorCoverage(+ missed_events, + current_gap)` = 滚动 buffer

多出的那两个字段就是「这是会丢的流」的定义。

---

## 4. 需求裁决（用户已定，勿再问）

1. **measurement** 调设备，**可以**驱动 pulse —— v1 `imaging_template` 形状：节点拿模板、
   按输入参数改 slot（曝光等）、通过 API apply。v2 已有此形状（calibration 的表单里有
   `AuthoringField("pulse", …, required=True)`）。
   **camera_measurement 例外**：它不调 pulse，把时序让给外部窗口。
2. **task** 是更高层，编排多个 measurement + 中途处理（calibration 是典型）。
   **task 不发布 signal**，产出 report / json 文件。**不用 finite/infinite 描述 task。**
3. **processor** 不碰设备，对**恰好一个**上游 signal 做处理并发布。
4. **finite/infinite 是 measurement 的属性**：finite = 跑前建好 array；infinite = 维护 buffer。
   **buffer 大小由写节点的人在代码里判断，不进用户表单**（pylon / qCMOS 这类允许 infinite）。
5. **processor 的形态继承上游**：上游 finite → 顺序处理**不许丢**；上游 infinite → 处理 latest。
6. **标定不是 signal**，是方法 / 参数。
   **occupancy 拿标定：只允许文件路径参数**，不做 session 隐式「当前标定」
   （牺牲便利，换存档能自证用了哪一份）。
7. **UI 不自造 v1 没有的新展示件**；v1 有的（header 等）可以用。
8. **两个 edit 界面必须是 tab 页面，不是弹窗**。v1 参考：
   `zlc_workbench/task_console/logic_node_editor.py`（"One generic TaskConsole Logic-node Edit page"）
   `zlc_workbench/task_console/panel_editor.py`（"Snapshot-scoped Plot Edit surface"，
   含 `FluentPlotFitPanel` / `FluentPlotParameterPanel`）

---

## 5. 完成情况（实测于 `1547f7b`）

| 项 | 状态 |
|---|---|
| calibration（task） | ✅ `phase=done`，产出 `centers=(6,2)` / `thresholds=(6,)` / `method=box` |
| camera_measurement（measurement） | ✅ `phase=done`，`frames` 有值（炮由 pulse 侧发） |
| save 数据 | ✅ 写出 `console-61.npz` |
| occupancy（processor） | ❌ `add_logic` 挂死，300 秒不返回 |
| 上板 | ❌ 任何信号都撞 `requires zlc_data.OwnedSnapshot` |
| save→重开读回 / 图片 fit / ROI 派生 / edit tab | ❌ 全部被上板挡住 |
| 真机接线 | ⛔ 阻塞：qCMOS DCAM SDK 未装、pylon 与 FPGA 不在手 |

**git**（工作区干净，无未提交改动）：

| commit | 内容 |
|---|---|
| `1547f7b` | 回滚点。此前 31 个 commit 被用户回滚，保在分支 `pre-rollback-2026-08-07` |
| `0a3ea89` | DAC 段延时全部当 0 送板 —— `getattr(bd,"delay",0)` 命中默认值，真字段是 `delay_ticks` |
| `5441541` | 本文件 |
| `99870de` | 绘图 host 启动失败时说出真原因，不再一律答「host is closing」 |

---

## 6. 显著问题

### ① `_RUNTIME_KIND` 硬映射与裁决逐行冲突（`descriptor.py:28`）

```
MEASUREMENT → finite     # 错：可 finite 可 infinite
TASK        → finite     # 错：task 不用这两个词描述
PROCESSOR   → reactive   # 错：形态应继承上游
```

根因：`finite` / `reactive` 一个词压了两件正交的事 ——
**谁驱动它**（自驱 / 被上游驱动，这个确实能从 NodeKind 推出来）
和**数据装在什么里**（预分配 / 滚动，这个推不出来）。

### ② calibration 发布了不该发布的 signal

两个 `OutputSpec` 让标定件进了信号盘，出现在「可画信号」清单里；选它建面板 →
`CurvePlot requires zlc_data.OwnedSnapshot`。绘图包没错，是它不该在信号盘上。
按裁决 2 应改为写文件。

### ③ occupancy 的 Add 弹模态问「读哪个信号」→ 挂死

接线是节点的一项设置，该在它自己的 edit tab 上、Apply 重建，
不是 Add 时问一次且永不可改。

### ④ `FollowTap` 零调用者

裁决 5 的「无损顺序」这条路机制已存在，但没有 dataset builder 绑定它，
processor 只能走会丢的 latest-only。

### ⑤ 没有任何 infinite measurement

`MonitorDataset` 在 `zlc_atom` 零引用 → plane 上没有持续活信号 → occupancy 无源。
正确闭环：infinite measurement 在自己的 worker 上读设备、发布到 plane；
UI 读 plane，两者永不接触。

### ⑥ 两个 edit 界面形态错

弹窗 / 实测 `edit_panel` 返回 `False` 什么都不开，与 v1 的 tab 页面差距很大；
task_console 整体 UI 与 v1 的差距尚未逐项对账。

### ⑦ 前后端分离靠自觉，无机械强制

没有任何测试会因为「有人在 GUI 线程上做阻塞事」变红。已知雷：
`MonitorCapture.poll()` 走 `read_frame_records(timeout=2.0)`，没帧时**整整阻塞 2 秒**；
曾有人把它挂进显示 beat，两个窗口全卡死
（实测最坏 2011 ms/拍，正常 0.17 ms/拍）。

### ⑧ logic_node 无法自带 UI

`LogicNodeDescriptor` 只有 `authoring_schema`（自动生成表单），
没有让节点提供自己控件的口子。plug-in 骨架成立，
但「需要特定 UI 时自己给」这一半不存在；要开这个口会撞上 `zlc_ui` 密封规则。

### ⑨ `.v1` 契约后缀 17 处，全项目无人解析或分支它

且 `selection_bridge.py:920-923` 已出现从没有过 v1 的 `…fit.parameter.v2`。
为不存在的第二版本预留的装饰。

---

## 7. 工作纪律

1. **`packages/zlc_workbench/GOAL.md` 是历史任务书，可能过期** —— 参考，不是无条件权威。
   与用户当面裁决冲突时以裁决为准。
2. **任何断言必须在当前树上实测**，并打印被测模块的 `__file__`。
   反复栽的根因就是拿别的树、或上一轮的记忆当现状。
3. **守卫必须验证「在原缺陷下会变红」**，否则它只是装饰。
   恢复突变探针**禁用 `git checkout --`**（会连带抹掉未提交的真修改，已栽两次）。
4. **v1 是批判性参考**：继承基本思想，不抄它明显错的设计
   （例如「绘制时对当前值求表达式」的数据源 —— 那会让同一张图里两个信号来自不同发）。
