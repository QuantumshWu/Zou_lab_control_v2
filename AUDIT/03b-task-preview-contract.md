# 03-B — Task / Measurement / Logic Node 可观察性与 Preview 契约审查

状态：只读深审完成；未修改 production、tests 或旧文档，未运行测试。
基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`（2026-08-17）。
范围：七个 discoverable Logic Node、descriptor/discovery、`NodeHost`、TaskConsole
自动 panel、Task takeover、progress、Stop、terminal artifact 及直接相关测试。文档只作
矛盾证据，不作真相源。

## 1. 结论先行

用户判断成立：新增 Logic Node 反复漏 live data、progress 或 preview，不是几个插件作者
“忘记写一行”的偶发现象，而是当前骨架允许四件互不约束的事分别存在：

1. descriptor 声明“这个 node 将来有这些 outputs”；
2. node 自己再次声明 runtime outputs，并手写 live slot；
3. `NodePreviewSpec` 再用字符串点名哪些 output 自动开 panel；
4. node 自愿调用 `report_progress()`，Workbench 再从 Host observation 猜状态。

任何一项遗漏都能通过 discovery；很多遗漏直到真实运行才表现为“Plot 开关不存在”、
“row 说 live 但 panel 不出现”、“运行结束才有图”或“Stop 后 artifact 找不到”。现有测试
分别守住若干已修过的节点，却没有一条通用门要求新 Measurement/Task 必须实时可观察。

更严重的是，当前 UI 还会主动制造两种假象：

- worker 一启动，row 就把尚未发布的所有声明 output 标成 `live`；
- terminal 后仍优先显示最后一条 progress，因此真实节点可在完成/取消后继续显示
  `Scanning n/n`、`Saving survival` 或最后一条 qCMOS 读数，而不是 `done/cancelled`。

所以本阶段结论是 **REDESIGN CONTRACT（收缩，不扩层）**，不是继续给每个新 node
补特殊分支。最小方向是：静态 output vocabulary、强制 Measurement/Task hosted-live、
Task 强制 progress、preview 必须引用同一 output declaration、terminal artifact 必须兑现，
Workbench 只显示 plane/Host 的真实状态。

## 2. 实际链路与目前的四份 truth

```text
LogicNodeDescriptor.outputs / node_previews / artifact_outputs
       │                 （静态承诺）
       ▼
ConsolePresenter._build_logic_candidate
       ├─ descriptor.instantiate() -> concrete node
       ├─ descriptor.previews_for() -> LogicBinding.preview_specs
       └─ make_host() -> descriptor outputs 再转换成 DatasetOutputDeclaration
                              │
                              ▼
NodeHost.start() -> node.execute(context)
       ├─ node 再暴露 dataset_output_declarations
       ├─ node 手工 attach_live_outputs(slot)
       ├─ slot 手工返回 name -> LiveDatasetOutput
       ├─ node 可选 report_progress()
       ├─ node 可选 publish_final()
       └─ execute return value 由 Workbench 反射 artifact path
                              │
                              ▼
ConsolePresenter.poll_logic()
       ├─ Host observation -> row/status strip
       ├─ plane.freeze() + preview output 字符串 -> 自动 panel
       └─ final_result + artifact output 字符串 -> artifact row
```

同一 output 名称/contract 最多同时存在于：

- plugin `DatasetOutputDeclaration`；
- descriptor `OutputSpec`；
- concrete node 的 `dataset_output_declarations`；
- live slot mapping key；
- `NodePreviewSpec.output_name`；
- Workbench stable signal key。

Hosted 路径以 descriptor 传给 Host 的 declarations 为准；notebook/direct 路径以 concrete
node 自己的 declarations 为准。二者不一致时 GUI 与 notebook 可以拥有不同 vocabulary，
而 discovery 不会报错。发布时也许才由 plane 拒绝。`occupancy` 最明显：processor.py 的
`_OUTPUT_DECLARATIONS` 与 logic_node.py 五个 `OutputSpec` 是两份逐字重写。

## 3. 高优先级发现

### TP-001 — P0 — live/preview/progress 不是 NodeKind 契约

证据：

- `LogicNodeDescriptor.__post_init__` 只要求 Processor 有一个 dataset input；Measurement 和
  Task 可以有 outputs 却完全没有 preview，也可以只在 terminal `publish_final()`。
- `NodeHost._finish_worker_success()` 接受“publish final”或“曾 attach live”二选一；不知道
  descriptor 的 Measurement/Task role，也不要求 Task 报过 progress。
- producer 的 publication 只要求是 declared outputs 的**非空 subset**：`publish_final()`与
  live `_freeze_one()`都允许漏掉其余声明名，Host只记一个`_final_published/_live_opened` bool。
  因此node可宣称三个outputs、只发布一个并正常done；row仍会把另两个假报live，preview则
  永久等待。
- `discover_logic_nodes()` 只 rglob/import/type-check/检查 `api_name` 唯一。
- `test_every_discovered_node_can_actually_be_driven_by_its_host` 只检查 class 有
  `execute/evaluate`，不检查运行中可观察性。

结果：README 的“创建 package + logic_node.py，framework/UI 无需修改”只对“菜单里能出现”
成立，不对“能按统一规范 live 展示”成立。

裁决：**REDESIGN**。Product-hosted Measurement 有 Dataset output 时应强制打开 live lane；
Task 有 Dataset output 时也应强制至少一个 live output，并至少报告一次 progress。若未来
确有瞬时、只产生 final 的 node，应明确成为例外，而不是现在的默认自由度。

### TP-002 — P0 — row 会把“声明了”伪装成“已 live 发布”

`ConsolePresenter._show_logic()` 在 plane 尚无 description 时，只要 `host.running` 就把
output lifecycle 写成 `live`。这正好覆盖最需要诚实展示的空窗期：

- Camera worker 整段运行都可能仍处于 Host 的 `starting` phase；
- SLM initial solve 和一个 candidate 的全部 shots 结束前没有 Dataset publication；
- 一个只在 terminal publish 的新 node 也会从 Start 起被 row 称为 live。

`published` 变量实际含“declared-but-not-published”条目，名称本身也误导。

裁决：**FIX**。row 至少应区分 `declared/waiting`、`live`、`held/final`、`failed`；只有
`SignalDataPlane.describe_signals()` 的当前 description 可以证明 live。Host running 不能
替 signal 作证。

### TP-003 — P0 — worker phase 与 terminal progress 产生错误状态

`NodeHost._reset_generation()` 把 phase 置为 `starting`，`_start_worker()` 提交成功后没有
改成 `running`。没有 progress 的 Camera Measurement 因此会在整次 live run 中显示
`starting`。

另一端，Host terminal 时故意保留 `NodeProgress`（runtime test 明确断言 done 后 progress
仍在）。这本身可以作为“最后一条 telemetry”，但 `ConsolePresenter._observation_status()`
在 terminal 仍优先返回 progress.text，导致：

| Node | terminal 后可能长期显示 |
|---|---|
| Stepped/Seamless Scan | `Scanning n/n` |
| Temperature | `Saving survival` |
| SLM Feedback | 最后一次 qCMOS ratio/shot 进度 |
| cancelled real Task | Stop 前最后一条 progress，而非 `cancelled` |

Workbench 的 takeover 测试没有发现它，因为 fake `TaskHost.poll()` 在 cancelled observation
中人为去掉了 progress；真实 `NodeHost._end_run()` 不去掉。

裁决：**FIX**，而且无需删 runtime 的最后 progress：

- worker submit 成功即 phase=`running`；
- presenter 对 terminal 先显示 `done/cancelled/failed`，若有价值可附带而不是替代 phase；
- Task status strip 不得把 progress 当 terminal truth。

### TP-004 — P1 — 自动 preview 对 developer error 静默降级且不重试

`_ensure_node_previews()` 有四个问题：

1. `NodePreviewSpec.plot_kind` 不校验是否属于 TaskConsole 五种 catalog；拼错会先 probe 失败，
   然后静默改成自动推断 kind，插件声明被无声忽略。
2. 明确声明的 kind 与真实 schema 不兼容时也静默 fallback；这与“node 最懂 physics，
   plot 不得猜”的 `NodePreviewSpec` 设计理由相反。
3. 在 `add_panel()` 成功前就把 output name 加入 `binding.previewed`；host/spec/panel 创建失败后
   本 generation 永不重试。
4. 该调用不在独立 error boundary 中；异常可从 `poll_logic()` 冒到 GUI beat。

动态 preview 还有一个静态缺口：`previews_for()` 只检查类型和 name 唯一，不检查 resolved
preview 是否属于同一次 `outputs_for(values)`。拼错 output name 的动态 preview 会永远等待，
不报错。

裁决：**FIX**。未知 kind 和不兼容的明确 kind 应显示一次 node-specific error，不得偷换；
只有 panel 真正建立后才记 `previewed`。在动态 resolver 删除前，至少在同一 authored values
上验证 preview subset。

### TP-005 — P1 — `resolve_outputs` / `resolve_node_previews` 是历史残余

全树事实：

- `resolve_outputs` 没有任何 concrete descriptor 使用；它来自旧的 camera `frame_0...N`
  动态 vocabulary。
- `resolve_node_previews` 只有 Camera Measurement 一个 consumer。
- 其 `_frames_preview(_values)` 已完全忽略 values，始终返回同一个
  `frames -> facet_grid`。

Git history 也吻合：动态 output resolver 为“每 frame 一个 signal”引入；camera 后来改回
唯一 `frames` signal，resolver 本体没有随旧模型退出。

裁决：**DELETE / SIMPLIFY**。当前七个节点 output/preview vocabulary 都是静态的；删除两种
resolver 与 `_frames_preview()`，Camera 直接声明静态 preview。不要为已消失的动态需求保留
额外 contract 分支。

### TP-006 — P1 — Task takeover 目前存在三套互相冲突的产品语义

三份证据各说不同的事：

1. `ConsolePresenter._task_command_blocked()` 的注释与 presenter test 说“Task owns the
   bench, not the window”，允许开/关 panel、加 row、改另一 row draft。
2. `ARCHITECTURE_DESIGN.md` / `IMPLEMENTATION_PLAN.md` 说 active Task 时除 view-only
   inspection 与 Stop Task 外，所有 state change 禁止。
3. 真实 Qt `TaskConsoleHandle.set_task_takeover()` 实际禁用所有 card settings、所有 logic
   rows/editors；`TaskConsoleView` 又禁用 Add/kind/layout，但仍留下 Pause、Selectors、Save
   screenshot。plot interaction 也没有停。

因此当前既不是“只锁 bench”，也不是“彻底 takeover”：

- 操作者无法在 Task 运行中手动 Add/retarget panel 来补看漏掉的 preview；
- card display settings 与 Edit 被禁用，monitor 能看但难以检查；
- `Selectors` header 仍可开；plot 的 area commit 最终仍可调用 `update_logic_draft()`。
  该 guard 只阻止 active Task 自己的 draft，另一 producer row 的 draft仍可能被改；
- 正式 Qt test 明确断言 Add Panel/card settings 被禁用，已把这套混合行为锁成“正确”。

这是**必须由用户裁决**的产品政策。推荐选择“bench-only admission”：

- 禁止任何 node/device Start、active Task draft mutation 和 device command；
- 允许 Add/retarget/display setting、zoom/pan/fit inspection、Save screenshot；
- area selector 可以只保存 view selection，但 Task active 时不回写 producer draft；
- 保留唯一 Stop Task。

这样与“运行中必须能 monitor”一致，也复用 DeviceUseCoordinator 的真实设备边界。若用户选择
旧文档的 total takeover，则必须把 Pause/Selectors/plot commit 等现有漏口也真正关掉，不能
继续保持现在的半锁定状态。

### TP-007 — P1 — Task progress 与普通 error 共用一个 last-writer status strip

`_project_task_takeover()` 每次 console projection 都无条件 `_report(active task status)`。
`beat()` 先 `_report_panel_errors()`，再 `poll_logic()`；后者刷新 projection 并立刻用 Task
progress 覆盖刚出现的 panel error。`StatusStrip` 又明确采用“最新消息是真相”，不保留高
severity。

结果是 Task progress 不是被动状态，而是每 beat 人工成为“最新事件”；运行期间其他错误
可能只闪一下或完全看不到。

裁决：**FIX**。最小方案是 row 持续显示 progress，status strip 只承担 event/error 与 Stop
action；至少也应只在 task status 真正变化时写 strip，并保证 error 不被同一 beat 的重复
progress 覆盖。

### TP-008 — P0 — Calibration “latest preview”被谎报成 exact finite Dataset

`CalibrationCapturePreviewSlot` 只保存 `_latest_cycle`，每次 update 覆盖上一 cycle；这在语义上
是 monitor/latest。但 `freeze_live_outputs()` 使用 `DatasetCoverage(3,3)`。Host 因而把它当
exact finite Dataset，在正常 terminal/Stop 后保留 generation。

实际后果：

- Workbench 只能用 `_transient_task_previews` 私有字典删自动 panel；
- `@logic/calibration/capture_preview` signal 本身仍作为 finished signal 留在 plane/chooser；
- real-app test 甚至明确断言 Task 完成后 plane 仍只含该 capture_preview，同时 panel稍后消失；
- `written_cycles` property 没有任何 consumer。

这不是一个“完整 3-cell Dataset”与“latest cycle”措辞差异：`DatasetCoverage` 决定 Host 是否
保留 generation，`MonitorCoverage` 才代表被替换的当前窗口。

裁决：**FIX / USER DECISION**。推荐把 Calibration capture preview 定义为真正 transient
Monitor：terminal 时 signal 与自动 panel一起退出；若用户希望保留最后一 cycle，就应将其
明确发布为 FINAL evidence output，不应把 latest monitor 伪装成 exact Dataset。

### TP-009 — P1 — SLM live preview 只在整批 shots 完成后发布

`SlmFeedbackTask._measure()` 每 shot 已维护在线 `readout_average` 并报告 progress，但直到
全部 `shots_per_candidate`（默认100）完成后，`execute()` 才调用 `_publish_candidate()`。
initial `solve_phase()` 前也没有 progress。因此实际人类时序是：

```text
Start -> “starting” + 三个 panel 都不存在
      -> initial solve 完成
      -> 100 shots 只有文字进度，无 readout/phase/history Dataset
      -> 一次性同时出现三个 preview
```

这不满足“Task 运行中持续把过程作为 preview 展示”的强含义。它只满足“每 candidate 一次”。
此外 SLM 借用了 `nodes.scan.ScanLiveSlot` 作为通用 dict slot；这是 scan-specific owner 被别的
plugin 当公共 runtime helper 使用的层级泄漏。

裁决：**REDESIGN WITHIN PLUGIN**。推荐在当前 candidate measurement 内按有界 cadence发布
running mean + 同一 applied phase；history 可以保持截至上一个完整 candidate，candidate完成
后再追加 ratio。不要按每个 raw frame强制 render，也不要等整个 candidate 才首次出现。
节流应由 producer/coalescing语义表达，不能靠 GUI 是否刚好 freeze。

### TP-010 — P1 — SLM Stop 是“cancelled”与“accept current best”的混合状态

SLM 的物理行为很明确：Stop 可重施加 best/latest durable candidate，并重发 matching
readout/phase/history；文件也存在。但 Host lifecycle 仍是 `cancelled`、
`final_result_resolved=False`。Workbench artifact capture 只接受 successful terminal result，
所以：

- device 正在使用 retained candidate；
- plane 有 retained candidate signals；
- candidate NPZ 在磁盘；
- row/artifact_results 却不承认有 terminal artifact；自动 preview panels 还会被移除。

这不是单纯 UI bug，而是 Stop 的产品含义未定。必须由用户二选一：

1. **推荐：Stop=取消**。恢复 incoming，不把 candidate称作正式 retained result；另设明确
   `Accept current best` 才提交 artifact/phase。
2. **Stop=提前接受 best**。则 lifecycle 必须有可消费的 terminal result/artifact，不能继续
   标成普通 cancelled 并丢 `final_result`。

当前混合态无法成为稳定 contract。

### TP-011 — P1 — Calibration/Temperature terminal side effect 没有与 Stop 排序

SLM 正确使用 `context.seal_terminal()`，把 Stop 与最终 apply/save 排序。另两个 Task 没有：

- Calibration 在 acquisition/replay 中能检查 Stop；进入 `writer.render()`、`_analyse()`、
  JSON save、六图 report 后不再检查，也不 seal。Stop 可以已被接受，Task仍长时间分析/写盘，
  最后 Host 将其标 cancelled并丢 final result；有效 JSON/report可能成为 UI 不可见的 orphan。
- Temperature 完成 scan 后计算、写 JSON，再 `publish_final()`；没有 terminal seal。Stop 若在
  save/final间获胜，同样可留下文件/输出与 cancelled lifecycle不一致。

裁决：**FIX**。在不可逆 terminal pair 前最后检查 cancel，然后 seal；seal 后 Stop不能重标，
异常仍是真失败。Calibration 的原始 sample folder 可按现有明确策略在取消时保留，但它不应
冒充成功 artifact。

### TP-012 — P1 — artifact declaration 只是一条“最好能找到”的反射提示

`_capture_artifact_results()` 对 descriptor 的每个 `ArtifactOutputSpec`：

- 从 result mapping/attribute 尝试取值；
- 缺失时静默 `continue`；
- 只 `Path.resolve()`，不检查文件存在、类型或 codec；
- extra result 不报；failure/cancelled 永不显示 artifact。

因此新 Task 可以声明 artifact却忘记 return，仍显示 done，只是 UI 没有 artifact。现有测试
只有 happy path。

裁决：**FIX**。successful Task 必须兑现每个 required artifact output，且路径至少必须存在；
否则这次 run 的 observable contract失败。若 artifact optional，应在 declaration中明确，不能
把所有 missing都当 optional。

## 4. 七个 Logic Node 逐一过关

| Node | 实际 live 时机 | Progress | Preview | Terminal / Stop | 裁决 |
|---|---|---|---|---|---|
| `camera_measurement` | repeat=0 每个完整 cycle 替换；finite 每个 cycle发布累计dataset | **无** | `frames/facet_grid` | finite FINAL；Stop read slice约50ms；monitor detach | **KEEP + FIX**：静态 preview、Host running phase；finite future zero validity另见总清册INV-004 |
| `stepped_scan` | 每 shot `slot.publish(scan)` | 每 shot | `scan/facet_grid` | FINAL；board wait可取消；authored `time.sleep(settle)`不可取消 | **PASS WITH DEBT**：live正确；settle改成可取消等待，terminal显示修正 |
| `seamless_scan` | 每 played point同 front发布 | 每 data point（shots聚合） | `scan/facet_grid` | FINAL；wait_done分片，但 `settle`与某些 adapter `fire()`可能不可取消 | **PASS WITH DEBT**：缺独立中途live/Stop guard |
| `occupancy` | runtime随 source：latest/follow/frozen | 无独立progress | **无** | source terminal后processor final | **USER DECISION**：processor auto preview是否仍可选；当前测试锁死“none” |
| `calibration` | 每完整 long/readout/long cycle替换latest | capture及后处理阶段 | `capture_preview/facet_grid` | artifact result；preview panel移除但signal误留；后处理Stop不安全 | **REDESIGN LIFECYCLE** |
| `temperature` | 每 release point同一 front发布 raw scan + survival + rate | scan + read/save stage | `survival_rate/curve` | 三个 FINAL + JSON；preview被promote后保留 | **KEEP + FIX**：terminal seal/complete status；第三个raw `scan` output需用户确认 |
| `slm_feedback` | 每完整 candidate或validation才发布三output | 每 shot + ratio | 两image + curve | success seal正确；Stop保留candidate却lifecycle cancelled | **REDESIGN OBSERVABILITY/STOP SEMANTICS** |

### 4.1 Camera Measurement

保留：

- `_CameraLiveSlot` 对 monitor与finite coverage作真实区分的方向；
- complete-cycle publication与50ms sliced read cancellation；
- 单一 `frames` signal + READOUT_EVENT point axis；这比动态 sibling vocabulary稳定。

问题：

- `_frames_preview()` 已是死的动态包装，应删；
- node不报 progress，暴露 Host `starting` defect；
- direct `.measure()` 只在最后 publish，只有 hosted `execute()`提供中途 live。需要确认“始终
  live”是 Product-hosted contract，还是 notebook direct API也必须流式；推荐前者，notebook
  若要观察中途应显式使用 capture/callback，不隐式创建后台 lifecycle。
- finite slot预分配未来 cycles为零但未在 Dataset validity中标 invalid；coverage虽说 partial，
  plot/processor读的snapshot仍可能把未来零当事实。该问题已在全局清册INV-004记录，属P0。

### 4.2 Stepped / Seamless Scan

两者均已做到核心期望：第一条完整 observation 到达后立即发布 growing scan，并最终把同一
科学 dataset转成 FINAL；preview声明也一致。

仍需过关：

- 两处 `time.sleep(settle_seconds)` 对任意 authored大值不可取消；
- seamless tests主要守 table/order/final，没有像 stepped long virtual test那样明确断言至少
  一个 partial live coverage；
- `ScanLiveSlot` 的实现是通用 latest-front容器，却位于 scan plugin并被 SLM借用。若保留共用，
  owner应是runtime公共signal能力；若不愿增加公共类，SLM应在自己owner内写最短slot，不能
  让一个concrete plugin成为另一个plugin的foundation。

### 4.3 Occupancy

Processor本身跟随live source发布，runtime extent分流正确；问题集中在声明与呈现：

- `_OUTPUT_DECLARATIONS` 与 descriptor outputs是两份truth；
- 没有 primary preview，UI甚至不显示 Plot switch；`test_auto_panel_kind` 明确把“processor不
  preview”锁为政策。

若用户只要求 Measurement/Task自动preview，Occupancy可保留无preview；若目标是“每个可视
Logic Node Start后都不沉默”，则应由Occupancy自己选择 `frame_judged`、`rate` 或其他primary
output，不能由shape猜。选哪一个是科学/产品决策，审查不代判。

### 4.4 Calibration

保留：

- acquisition期间每cycle publish与progress；
- current capture而非累计200份大数组；
- Task直接产生artifact/report，Workbench不组装科学report。

问题：

- 语义是latest却使用exact coverage（TP-008）；
- `CalibrationCapturePreviewSlot.written_cycles`零consumer；
- Task在一个类中同时acquire/replay/analyse/save/report，体积很大，但这些是同一 concrete
  Task的直线workflow，本审查不建议为此新增manager/coordinator；真正该切的是可取消边界和
  terminal commit，而不是再加层；
- preview到底是一张“last image”还是三张long/readout/long facet，代码/测试与权威文档相反，
  见§7，需要用户确认。按物理可解释性推荐保留当前三-frame facet。

### 4.5 Temperature

这是当前最接近目标契约的Task：每point在同一front发布frames/survival/rate，rate是普通curve，
最后publish FINAL并返回JSON artifact。

问题：

- code实际声明/发布三个outputs（含raw `scan` frames），根架构和Atom README都说只发布
  `survival`与`survival_rate`；raw scan没有production consumer或专门test；
- 最后一条progress是`Saving survival`，终态显示会说仍在保存；
- save/final没有terminal seal。

raw scan保留与否必须由用户按“可重分析证据 vs memory/contract最简”裁决。若保留，应更新
目标和测试；若删除，不能只删descriptor，必须让shared scan live front不再声称该output。

### 4.6 SLM Feedback

保留：

- 三个outputs在每次 `_publish_candidate()`共享generation/revision/run record；
- online mean/Welford progress；
- success的terminal seal；
- ordinary failure恢复incoming。

问题：

- initial solve无progress，candidate内无Dataset preview（TP-009）；
- `readout_average`/`candidate_phase`是replace-latest current candidate，却都用
  `DatasetCoverage(1,1)`；`uniformity_history`预分配max_updates并从第一次就声称
  `DatasetCoverage(max,max)`，未来invalid点被算作“已完整写入”。这是为了让Host早停成功
  时通过exact-complete检查，说明coverage/terminal模型与该Task不匹配；
- Stop hybrid（TP-010）；
- Workbench没有任何SLM real-button preview/artifact lifecycle test。

算法和科学鲁棒性由SLM专项审查处理；本节只裁可观察性/生命周期。

## 5. 文件 / 类 / 函数裁决清册

| 文件 / 符号 | 必要性与归属 | 裁决 |
|---|---|---|
| `nodes/_framework/descriptor.py::LogicNodeDescriptor` | shared skeleton，必要 | **KEEP, SIMPLIFY**；加role可观察性不变量，删无consumer动态resolver |
| `OutputSpec` | 与runtime `DatasetOutputDeclaration`重复name/contract | **MERGE/REFERENCE**；descriptor应引用同一declaration，不手抄 |
| `NodePreviewSpec` | node-owned physics kind必要，但output_name是第二拼写 | **KEEP SHAPE OR MERGE**；最好引用declaration对象；不再动态resolve |
| `ArtifactOutputSpec` | Workbench需要semantic contract | **KEEP + ENFORCE** missing/existence；明确optional与否 |
| `outputs_for()` | 所有调用者使用，但动态能力零consumer | **SIMPLIFY**为静态tuple；保留method仅为call-site稳定也可 |
| `previews_for()` | 同上 | **SIMPLIFY**；直接静态tuple |
| `_framework/discovery.py` | 自动收集必要 | **KEEP**；它不应运行硬件，但descriptor construction应在此暴露静态违约 |
| `zlc_runtime/host.py::NodeProgress/LogicNodeObservation` | UI-neutral telemetry必要 | **KEEP**；明确progress不是terminal phase |
| `NodeExecutionContext.attach_live_outputs/publish_final/report_progress/seal_terminal` | 当前生产路径必要 | **KEEP** |
| `open_live_dataset/open_exact_dataset`及Host helpers | 零production consumer，只由tests维持 | **DELETE CANDIDATE**；已在01清册，勿让其成为新的“统一实现”默认答案 |
| `NodeHost._start_worker/_finish_worker_success/_end_run` | lifecycle核心 | **FIX** running phase、live/progress policy与terminal语义；不增加parallel state machine |
| `zlc_workbench/logic.py::LogicBinding` | row/run binding必要 | **KEEP**；preview字段可收缩；无preview node不应保存无意义auto_preview truth |
| `LogicCandidate.previews` | 冻结run preview合理 | **KEEP**，静态后可直接来自descriptor |
| `make_host()` | descriptor→runtime唯一接线点 | **KEEP**；docstring仍说“三frame暴露三个signals”，与当前一signal实现矛盾，需改 |
| `ConsolePresenter._ensure_node_previews` | 自动panel产品能力必要 | **KEEP + FIX** error/retry/validation；不要再添加plugin分支 |
| `_transient_task_previews` / cleanup | panel lifecycle目前必要 | **REDESIGN/RENAME**；现在也记录Measurement；修coverage后只追真正Task auto-panels |
| `_show_logic/_observation_status/_project_task_takeover` | projection必要 | **FIX** truth与status冲突 |
| `TaskConsoleHandle.set_task_takeover` | composition投影必要 | **USER DECISION + FIX**，与presenter admission一致 |
| `TaskConsoleView.set_task_takeover` | view chrome必要 | **USER DECISION + FIX**，当前半锁定 |
| `LogicRowView.set_task_takeover/_project_commands` | 命令投影必要 | **FIX**：按per-command admission，不用一个bool关整窗 |
| camera `logic_node.py::_frames_preview` | 忽略参数、唯一动态consumer | **DELETE**；静态声明 |
| camera `_CameraLiveSlot` | capture-specific lazy materialization有真实职责 | **KEEP + FIX finite validity** |
| `ScanLiveSlot` | 实现通用，位置名义是scan | **MOVE/REPLACE DECISION**；不得继续被SLM当隐式framework |
| `CalibrationCapturePreviewSlot` | plugin-specific cycle materialization必要 | **KEEP + FIX coverage**；删dead `written_cycles` |
| `CalibrationTask._run/execute` | concrete workflow必要 | **KEEP DIRECT FLOW**；加cancel/seal边界，不增manager |
| `TemperatureTask.execute` | 简洁直接 | **KEEP + FIX seal/status/output裁决** |
| `SlmFeedbackTask._publish_candidate/_measure/execute` | plugin内必要 | **KEEP OWNER, REFACTOR FLOW**；in-candidate live与Stop结果语义需重定 |
| 七个 `logic_node.py::LOGIC_NODE` | discovery入口均必要 | **KEEP**；静态output/preview contract统一 |

## 6. 测试审查与缺口

### 已守住的真实行为

- Camera finite在运行中出现partial coverage、Stop在read slice内响应。
- Stepped长virtual path至少观察到partial live coverage。
- Calibration真实Qt按钮路径观察到多个preview sequence、Stop按钮和terminal panel cleanup。
- SLM unit/headless tests守candidate三output同front、history增长、terminal seal、Stop retained
  candidate与失败restore。
- Temperature真实virtual chain守FINAL/artifact/curve数值。

### 测试反而掩盖或锁死的问题

- runtime test明确锁住“done后progress仍存在”，但Workbench没有真实Host terminal status test。
- Task takeover presenter test使用不会保留cancel progress的fake，掩盖真实stale status。
- real Qt test明确断言active Task时Add Panel/card settings禁用，锁死与“monitor期间仍可操作
  view”的冲突政策。
- `test_auto_panel_kind`明确断言Occupancy没有preview；改变全node policy需用户先裁决。
- Calibration real-app test明确断言panel移除后capture_preview signal仍留plane，锁住exact/
  monitor混淆。

### 缺失的红灯

1. generic：每个Measurement/Task descriptor有Dataset output时必须声明至少一个合法preview；
2. generic：preview names是同一静态output declarations的subset，plot kind属于公共catalog；
3. generic/product：successful generation兑现完整declared output vocabulary；
4. generic/product：Measurement/Task不能只在terminal首次publish；Task至少有progress；
5. Workbench：真实Host done/cancelled后row/status strip显示terminal phase；
6. Workbench：SLM/Temperature/两种Scan真实auto-panel链，而不只是descriptor/headless；
7. SLM：第一candidate未完成时已有running average/phase preview；
8. Calibration/Temperature：Stop落在analysis/save/terminal边界的结果、文件和phase一致；
9. artifact：declared path缺失、不存在或extra时的明确失败；
10. Task active：真实plot area commit不能越过选定的takeover政策；
11. Seamless：中途partial live与Stop during settle/fire。

不建议为每个字段建矩阵；最有价值的是一个generic descriptor gate，加每种runtime语义各一条
真实产品纵向测试。

## 7. 文档—代码—测试矛盾矩阵

| 主题 | 文档说法 | 当前代码/测试 | 结论 |
|---|---|---|---|
| Camera outputs | `ARCHITECTURE_DESIGN`仍说`frame_0...frame_N`多个二维signals | 一个`frames` Dataset，frame在READOUT_EVENT point axis；测试守一signal | **文档过期**；推荐代码 |
| Calibration preview | 根架构说latest最后一张、R=1/P=1/image | 代码发布long/readout/long三frame并强制facet_grid；git/test刻意守住 | **需用户确认**；推荐代码的三frame物理语义 |
| Calibration transient | 文档说terminal统一移除transient preview | panel移除，但signal被DatasetCoverage保留；测试要求保留 | **实现/测试不符合文字** |
| Temperature outputs | 根架构/Atom README说两份survival/rate | descriptor/live/final另有raw `scan` | **需用户确认** |
| Task takeover | 根文档说所有state change禁用；presenter说Task不拥有window | Qt半锁定：有些全禁，有些仍开，selector commit可穿透到其他draft | **三方矛盾，必须裁决** |
| SLM observable | 根README/Atom README说occupied-only、空shot不入mean | code与最新架构明确所有finite valid shots进入、只减dark | **README严重过期** |
| SLM exponent | Atom README `0.45` | code/最新架构`0.25` | **README过期** |
| SLM geometry | Atom README只支持5x7 | code/最新架构支持trusted-prior任意sparse | **README过期** |
| SLM Stop | Atom README说Stop恢复incoming | code/最新架构保留best/latest durable | **README过期且生命周期仍混合** |
| Task preview naming | Atom README称“task preview declaration” | `NodePreviewSpec`也服务Measurements | **词汇过期** |
| make_host doc | “三frame暴露三个ordinary signals” | 当前一个frames signal | **代码内历史注释** |

特别说明：根 README 第20–22行和 Atom README 第138–148行描述的是已经被当前代码明确
否决的SLM算法，不只是措辞不同；在专项SLM审查收口后必须单独改写，不能继续作为用户说明。

## 8. 推荐的最简统一 contract（不实施）

### 8.1 推荐方案 A：收缩现有骨架，不新增manager/class

1. **Static output truth**
   - 删除 `resolve_outputs`、`resolve_node_previews`。
   - concrete plugin只定义一组 `DatasetOutputDeclaration`；descriptor引用同一对象/由它直接
     投影，不重写name/contract。
   - preview引用同一declaration而非裸字符串。

2. **Role observability gate**
   - Measurement + Dataset outputs：Product-hosted run必须attach live，首个完整科学observation
     到达即publish；finite terminal可以promote同一dataset为FINAL。
   - Task + Dataset outputs：必须attach live，且至少一个output是primary preview；Task还必须
     在第一个可能较长的步骤前report progress。
   - Processor：跟随source publish由runtime保证；是否auto preview由用户裁决。
   - artifact-only Task：允许无Dataset preview，但progress仍必需。

3. **Truthful UI**
   - row只按plane称live；未publish显示waiting。
   - running phase由Host保证，不靠每个node第一条progress修补。
   - terminal phase压过last progress；artifact另列。
   - explicit preview kind无效时一次明确报错，不fallback猜测。

4. **Terminal commit**
   - normal success：所有declared artifact都存在；缺失即contract failure。
   - 不可逆save/apply前最后check cancel并seal。
   - transient current preview用MonitorCoverage，terminal从plane退出；要保留的数据显式FINAL。

5. **Small generic guards**
   - discovery级静态contract一条；
   - real Product host级“partial live before terminal + progress + terminal phase/artifact”一条；
   - 不为每个plugin复制framework测试。

此方案主要是删动态分支、增加现有构造/terminal校验并修projection，不要求新registry、DTO、
preview manager或第二lifecycle。

### 8.2 备选方案 B：Host增加 `context.publish_live(outputs)`

让Host拥有一个通用latest-front slot，node只提交immutable output mapping，可删除/减少各plugin
重复的`set_change_listener/freeze_live_outputs/close`样板。优点是新node最难忘、API统一；缺点是
Camera/Calibration当前用lazy materialization避免producer每shot构造完整snapshot，直接替换会有
性能回退，最后仍可能保留custom slot双路径。

因此本审查不把B作为第一步。先按A收紧contract；只有profile证明slot样板本身是主要问题，
再由用户批准是否统一成Host-owned live publisher。

### 8.3 不推荐方案 C：继续加 `live_required` / `preview_policy` DTO/enum

在descriptor再声明一份live/final/retention policy，只会增加第五份truth；runtime实际publication
已经能证明这些事实。除非出现无法由actual coverage/final publication表达的真实产品需求，
不要增加这类平行状态。

## 9. 必须交给用户的裁决

1. **Task active UI**：采用推荐的“只锁bench、Monitor可操作”，还是旧文档的“除Stop外完全
   takeover”？当前半锁定不能保留。
2. **SLM Stop**：普通cancel并restore incoming，还是“提前接受current best”并返回正式artifact？
3. **Calibration preview**：保留当前long/readout/long三图facet（推荐），还是只显示最后一图？
4. **Calibration terminal preview**：真正transient、terminal时signal也退出（推荐），还是把最后
   cycle明确作为FINAL evidence保留？
5. **Temperature raw `scan`**：作为可重分析证据保留第三output，还是按目标文档删到只剩
   survival/rate？
6. **Processor auto preview**：Occupancy继续不自动开图，还是每个可视Logic Node都必须声明
   primary preview？若后者，primary选哪个output需要用户定。
7. **“Measurement始终live”边界**：推荐只约束Product-hosted run；direct notebook API按调用者
   选择iterator/callback。是否要求direct `.measure()`也隐式流式发布？
8. **SLM running preview cadence**：推荐有界coalesced running mean（不等完整candidate）；是按
   时间、shot chunk，还是每shot提交给latest slot由plane/UI合并，需结合真实shot cadence profile。

## 10. 最终裁决摘要

- **无需存在，应删除**：动态output resolver、动态preview resolver、Camera `_frames_preview()`、
  Calibration `written_cycles` dead property；runtime无人使用的两套open-live helper继续列删除候选。
- **层级错误**：SLM借用scan package的`ScanLiveSlot`；Task takeover admission在presenter与Qt
  handle各有一份不一致政策。
- **非唯一truth**：output declarations、preview output strings、artifact result reflection。
- **严重潜在问题**：假live、stale terminal status、Calibration latest/exact混淆、Task error被
  progress覆盖、Calibration/Temperature Stop与save竞态、SLM cancelled/artifact/device混合态。
- **当前做得正确，应保留**：Camera完整cycle组装；Scan逐point live + final；Temperature同front
  frames/survival/rate；SLM同candidate三output revision与success terminal seal；descriptor-owned
  physics plot kind而非shape猜测。

本报告没有实施任何修复。所有“推荐”都以当前用户裁决为最高权威；§9未决项在用户选择前
不应被旧architecture/checkpoint或现有测试替用户决定。
