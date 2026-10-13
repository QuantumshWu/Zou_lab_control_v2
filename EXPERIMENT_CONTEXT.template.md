# 实验背景与操作说明

这份说明补充软件无法推断的物理含义、实验方法和操作经验，供人和 MCP AI 共用。可以用中文或英文填写；将 `待填` 替换成实际信息，不适用的写 `不适用`。未知不能当成 0、关闭、无限制或已经校准。

填写后保存为实际运行 workspace 下的 `EXPERIMENT_CONTEXT.md`。AI 先用 `lab_state` 确认当前 workspace，再用 `lab_artifact(path="EXPERIMENT_CONTEXT.md")` 读取；若返回的 `next_offset < total_bytes`，继续读完。修改后重新读取即可，不需要重启实验。**现有 MCP 可以读取文件，但不会自动把它的内容注入每次对话。**

优先填写：实验目标、Pulse 通道含义、一个常用 Pulse 的阶段说明、操作限制。不必一次填完。

## 1 实验概况

- 实验装置名称与适用实验机：待填
- 最后核实日期、维护人：待填
- 原子种类、相关内态／跃迁、主要波长：待填
- 系统用途与当前实验目标：待填
- 常用术语或缩写的含义：待填
- 一句话描述实验链路，例如从装载到操控再到读出：待填

### 信息的分工

- 本文负责物理含义、选择理由、经过确认的限制和经验；当前设备值、Pulse 内容、Config、生效程序、运行状态及数据，以 MCP 读取的实际对象为准。
- 不在这里抄一套需要同步维护的当前参数。最优工作点引用对应 Config／Calibration／Science Context 文件；写出其适用条件。
- 文档与实际接线、通道身份或单位不一致时，只暂停受影响的操作并核对，不能凭名称猜极性或自行换算。
- AI 默认可以通过 MCP 执行当前任务内的常规操作；需要人工确认的具体例外写在第 8 节。本文不绕过框架的设备占用、限值或人工请求。

## 2 实验机与设备

设备 ID／Role 从 `lab_state` 和设备目录复制；不要仅凭型号识别同类设备。

| Role 与实际 device ID | 所在实验机 | 物理用途／连接对象 | 本地或 Remote 连接 | 需要注意的设置 |
| --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 | 待填 |

- Pulse 编译和控制所在机器、FPGA 所接机器：待填
- Camera 与 SLM 所接机器、GPU 计算所在机器：待填
- 外部触发链路（谁触发谁、边沿、频率或握手关系）：待填
- 手工开关、实验前需要确认但软件无法读取的状态：待填

只填写必要的地址和端口，不在此保存密码或 MCP token。

## 3 Pulse 通道的物理含义

用 `lab_editor_catalog`／详细 Pulse 文档核对实际 `port_key` 与显示名。**名字叫 shutter 不代表 High 一定开光；数字通道的 0、DAC 的 0 和物理零输出也不是同一回事。**

### TTL 与触发通道

| 实际 port_key／显示名 | 接到哪个器件、控制什么 | Low 电平与物理效果 | High 电平与物理效果 | 边沿、响应延迟与约束 |
| --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 | 待填 |

- 若器件有反相、开／关响应不同或机械动作，说明实际延迟及其测量来源：待填
- 相机触发是上升沿／下降沿／电平门控；曝光由相机还是 Pulse 决定：待填
- 必须同时或按顺序切换的通道，以及不可同时开启的组合：待填
- 未显式修改时的保持行为、Pulse 完成和 Stop／SAFE 后期望的物理状态：待填

### DAC 与模拟量

每个需要 AI 调整的模拟通道复制一份下面的小节。

#### 通道 待填

- 实际 port_key、显示名、连接对象：待填
- 控制的物理量及正负方向，例如磁场、失谐、光强：待填
- 数值链路：`DAC code → 输出电压 → 器件响应 → 实际物理量`；公式或标定文件：待填
- 换算中的单位、零点、偏置、符号、有效区间及校准日期：待填
- 可用范围与最大改变速度；是否必须 ramp、允许的跳变：待填
- Delay 的用途与参考事件；负 delay 是否已经计入实际编译时序：待填
- 安全输出、禁止组合、相关 Config 名称：待填

### RF 与光学参数的关系

| Device／字段 | 软件单位与器件输入 | 对应的光学或物理效果 | 换算与适用条件 |
| --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 |

如适用，写明 AOM 级次／单双程及移频符号、RF 到光功率的标定、阻抗与 Vpp／dBm 关系、切换后的真实稳定时间。不能直接把 RF 频率当光学失谐，或把 RF 功率当光功率。

## 4 常用 Pulse 的实验时序

每个常用 Pulse 复制本节。这里写“为什么这样做”；精确时长、重复、Config 覆盖和实际输出延迟仍读取 Pulse 与编译结果。

- Pulse 路径（相对 workspace）：待填
- 用途、前置实验条件、结束后期望状态：待填
- 此 Pulse 使用的 Config 文件与关键命名参数：待填
- Run repeat、Scan repeat、Bracket 和 Camera cycle 在这个实验中的对应关系：待填

| 阶段／Period 显示名 | 物理目的 | 关键通道及物理状态 | 与前后阶段的关系 | 是否拍照或触发其他设备 |
| --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 | 待填 |

- 一个 cycle 预期收到几帧；各触发属于哪个 cycle：待填
- 必须连续保持的光／磁场，允许中断的部分：待填
- SLM 修改或重排允许发生的时间窗口及结束条件：待填
- 最后一次核实时的编译时序或参考数据路径：待填

## 5 图像与信号应该怎样解释

**Frame 编号按实际数据的零基编号填写，不按文件名或肉眼的“第一张”猜测。**

| Frame 编号 | 拍摄阶段与物理含义 | 对应 Calibration／读出方式 | 后续用途 |
| --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 |

- Camera ROI、坐标方向、像素到物理距离的标定：待填
- SLM target／Fourier 像素与 Camera 像素的映射文件、适用光路：待填
- Site 编号依据；换 ROI／放大率／阵列后哪些标定会失效：待填
- Counts、occupied、survival 各自代表什么；survival 的条件、分子和分母：待填
- 背景扣除、无效点、未占据和丢帧怎样区分：待填
- 平均／pool 哪些轴，保留哪些轴；拟合是否使用权重或 SEM：待填
- 结果可信的判据及常见成像伪象：待填

## 6 实验方法的选择与执行

每一种常用测量复制本节，例如装载、寿命、温度、磁场优化或重排验收。方法名是示例，不代表系统已经配置完成。

### 方法 待填

- 要回答的物理问题；何时选它、何时不适用：待填
- 使用的 Layout、Pulse、Config、Calibration／Science Context 路径：待填
- 前置条件和必要人工准备：待填
- 采集与分析链（Logic 名称／信号引用／Plot 或 Fit）：待填
- 扫描哪个实际字段、单位、合理范围；哪些量必须固定：待填
- 重复次数／采样策略和停止判据：待填
- 拟合模型、固定参数、输出物理量及单位换算：待填
- 成功标准与可接受的不确定度：待填
- 校准结果应保存到哪里、更新哪个 Config 名称；是否需要人工确认：待填
- 结束后保留的状态或需恢复的工作点：待填

## 7 常用资源与参考结果

路径尽量相对 workspace。区分“已确认可用”与“历史对照”，不要仅凭最新文件名认定有效。

| 用途 | 文件路径 | 适用条件 | 核实日期／状态 |
| --- | --- | --- | --- |
| 默认 Layout | 待填 | 待填 | 待填 |
| 默认 Pulse 与 Config | 待填 | 待填 | 待填 |
| Occupancy Calibration | 待填 | 待填 | 待填 |
| 初始 SLM Science Context | 待填 | 待填 | 待填 |
| 正常实验的参考数据 | 待填 | 待填 | 待填 |

## 8 操作范围与异常处理

- AI 可以自行调整、扫描及保存的量：待填
- 必须先询问的具体操作：待填
- 禁止的范围、通道组合和操作顺序：待填
- 需要人工完成的物理动作：待填
- 出错后应保持还是停止哪些输出；正确的停止顺序：待填
- 可以重试的明确条件，以及不能盲目重试的操作：待填
- 发现无原子、异常亮度、失锁、帧数错误时，优先检查什么：待填

这些说明不是软件或硬件 interlock，也不意味着 AI 在断连后仍能执行紧急停止。不要把“工具请求被接受”当成设备已执行、光已稳定或实验已完成。

## 9 已知问题与实验经验

只写可追溯的现象和结论；将已确认事实与猜测分开。

| 现象 | 已确认原因或待验证假设 | 判断依据／参考数据 | 应对方法 |
| --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 |

## AI 开始实验前的阅读顺序

1. 读取本文，确认适用装置和核实日期；只对当前任务必需的缺失信息询问，不要求填完全部空白。
2. 已知且已授权的常规操作直接执行，不反复查目录或重新规划。未知字段才查 `lab_catalog`／`lab_api`；复用动作已经返回的状态，不另抄一套配置。
3. 操作 Pulse 前读取详细草稿与已加载 Config，必要时运行不触发硬件的 `validate`；检查编译结果是否仍匹配当前草稿／Config。
4. 按选定实验方法操作，执行后检查实际运行身份、完成状态、错误与数据有效性；结果不确定时先查状态，不自动重复 Fire／Start／Apply。

### 标准启动与准备顺序

这是一份准备检查表，不是自动执行脚本。Init 成功只表示设备初始化，不表示光路、相位和实验已经准备好。

1. 加载所选 apparatus JSON，再 Init 并等待实际就绪。Experiment 窗口自动打开，不另发“打开 Experiment”动作。顺序已经确定时可用通用 `lab_batch` 组合原操作，等待条件用 `lab_wait_for` 明确给出；服务端不内置这套实验流程。
2. **使用 SLM 光阱时，每次 Init／重新 Init 后都要加载本实验指定的初始相位或 Science Context，执行 Send，并确认设备命令成功。** 只在编辑器显示相位不算已应用；Adopt 也不能代替加载需要的相位。指定文件见第 7 节，未填时询问，不自行挑文件。
3. 按本次实验选择 Pulse、Config、Camera 设置和 Calibration。检查帧含义与资源适用条件；不要把上次实验的选择默认为这次所需。
4. 先准备 Pulse，再开始寻找相机视野；不要在 Init 时照抄历史小 ROI。载波、阵列或光路改变后先用较大视野定位，再收紧到完整阵列及所需背景／读出边界。只改 ROI 草稿不代表相机已经采用，须在下一次采集核对实际坐标。
5. 手动定位也须收齐所选 Pulse 的完整触发周期：例如一个周期有 3 次触发，就采 3 帧，不能因为只看第一帧而只采 1 帧。明确设置有限 `run_repeats`，Camera 本代 ready 后才 Fire；相机采完后还要等原 Pulse 有限播放生命周期结束。
6. 若 Calibration 等 Logic／Task 自己负责采集和 Fire，只 Start 一次，记录 generation 并等待这一代；不再额外 Fire。查看结果、错误和数据有效性，再进入下一步。不要在每个已经成功的同步操作后重复读取全套状态。

本装置需要增加或替换的准备步骤：待填

### MCP 调用与完成条件

以下沿用公共工具，不是新增工作流。将示例路径和设备 ID 换成本装置实际值；已知参数可用 `lab_batch` 组合，每批最多 16 步、30 秒。需要读图、选择 ROI 或判断科学结果时结束该批，再作决定。

**A · 初始化、SLM 准备。** 输入为本轮 apparatus 与 Science Context，前提是无冲突的活动任务。

```json
{"tool":"lab_call","arguments":{"handle":"devices","method":"load","arguments":{"path":"apparatus.json"}}}
{"tool":"lab_initialize","arguments":{}}
{"tool":"lab_wait_for","arguments":{"tool_name":"lab_state","arguments":{},"path":["session_open"],"equals":true}}
```

这些对象是 `lab_batch(calls=[...])` 的条目，不是一个合并 JSON 文件。检查等待结果中的 `result.device_failures` 为空，设备身份／连接正确、初始化不再 pending；`session_open=true` 本身不保证全部设备成功。

打开 SLM Control 后调用 `lab_editor_action(device_id=实际ID, action="context.load", arguments={"path":实际Context路径})`。使用下列等待条目，再检查其 `result.error` 为空且 `solved=true`、`send_enabled=true`，才发送：

```json
{"tool":"lab_wait_for","arguments":{"tool_name":"lab_editor_state","arguments":{"device_id":"实际SLM_ID"},"path":["pending"],"equals":false}}
{"tool":"lab_editor_state","arguments":{"device_id":"实际SLM_ID"},"expect":{"pending":false,"error":null,"solved":true,"send_enabled":true}}
{"tool":"lab_editor_action","arguments":{"device_id":"实际SLM_ID","action":"device.send","arguments":{}}}
```

Send 后再次等 `pending=false`，确认 `error=null`、`command_known=true`、`device_diverged=false`，记录本次 `command_revision`／receipt；旧的 known 状态或只显示 solved 不证明本次 Send 成功。软件命令确认也不等于光学稳定。

**B · 手动完整周期定位。** 输入为已准备的有限 Pulse、Camera 草稿和预计视野。

- `lab_action(action="logic.start", arguments={"node_id":相机节点}, expected_draft_revision=读到的修订)`；从返回的节点状态取 generation。
- `lab_wait(node_id=相机节点, generation=本代, until="ready")` 确认 `reached=true`、`still_current=true` 且无错误后，`lab_pulse(action="fire", device_id=实际sequencer_ID)`，只发一次。
- `lab_wait(..., until="terminal")` 等相机正常 done；再用 `lab_wait_for(tool_name="lab_pulse_state", arguments={"device_id":实际sequencer_ID}, path=["pending"], equals=false)`，随后等 `path=["completion_pending"], equals=false`。后者来自原有限播放生命周期，DONE 处理／释放完成后才清除；`running` 和 `board.firing` 都只是缓存的板状态。最终检查 `running=false`、`board.fault` 为空及 `last_warning` 无本次错误；停止／失败也可结束生命周期，不能只凭某个 false 宣布正常 DONE。
- 用 `lab_signal(signal=输出名, generation=本代)` 读取；分页的完整 `ref` 必须一致，不能只对 generation。Panel/Frozen 可能仍是旧图，截图／导出前 Refresh 并核对 publication；需要固定整份数据时读取该代保存的 Figure。

**C · Calibration。** 输入为本轮 Pulse／Config、正确帧角色和已核实的相机实际 ROI。只 Start 一次，记录 generation／run directory，沿同一 generation 用 `lab_wait` 观察。正常 done 后检查站点坐标、usable、loading、BOX bright−dark、实际与理论 fidelity；Task done 或理论 fidelity 高不能代替实际分类验证。正常批次按需求选择 `save_frames`，排查帧角色／漏点时开启原始帧留存。

**D · 位置比较。** 固定同一 Pattern／Target／Pupil、Pulse、曝光和重复数。前一 Calibration 与 Pulse 均结束后，只用 `operator.update` 的 `values` 修改载波，Send 并确认；保存该位置 Context、实际 ROI、command revision 与 run directory，再按 B/C 采集。最后复测起始位置区分空间差异和时间漂移，不自动重解 Pattern 或改变权重。

**异常出口统一：** `matched=false`／`reached=false` 是观察超时，不是没有执行；任何拒绝、非空 error、generation 变化或未知写入结果都停止后续组合，查原命令／原代。已完成步骤不回滚、不重放整个 batch。图像解释、ROI 是否包含全阵及是否继续实验仍由操作者／AI 根据实际证据判断，不由工具 `accepted` 替代。
