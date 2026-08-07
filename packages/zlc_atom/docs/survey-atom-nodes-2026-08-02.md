# zlc_neutral_atom 深读报告:device / logic_node 插件性审计

**审计范围澄清**:任务书里的 "session、operations/processors、core/signals" 是旧 main 布局的名字。迁移分支(`codex/system-architecture-migration`,HEAD=`7cf8b64 "Complete C3 generic Logic Node host migration"`)上对应物已迁位:session 门面 → `Zou_lab_control/api/facade.py`(Experiment,832 行);SignalHub → `zlc_neutral_atom/processing/signal_plane.py::SignalDataPlane`(2418 行);processors → `logic_nodes/`(kind="processor")。旧 `Zou_lab_control/` 只剩 `api/`+`workbench/` 共 2541 行。以下全部按现状实读。

---

## 1) 加一个新 device / logic_node 实际要碰几个文件

### 新 device:2–3 个文件(插件性已基本达标)

| 环节 | 要碰的文件 | 证据 |
|---|---|---|
| 定义+注册 | **1 个文件**:`devices/<子包>/device_types.py` 里加一条 `DeviceTypeDescriptor`(type_id、domain、AuthoringSchema、capabilities 元组、依赖声明、factory 闭包)。发现是 `walk_packages` 自动扫 `.device_types` 后缀模块,无注册表 | `device_types.py:96-120` |
| GUI 表单 | **0 个**。device_manager 完全由 `discover_device_types()` + `AuthoringSchema` 驱动,workbench 全树 grep 不到任何具体 type_id | `zlc_workbench/device_manager/editor_session.py:153-172`;grep 证实无硬编码 |
| session 接线 | **0 个**。`create_installation` 通用拓扑组合(Kahn 排序→factory→capability 校验→反向 closer),`Experiment` 只按 capability token 寻址 | `installation_runtime.py:232-318`、`facade.py` |
| 测试 | **1 处必改 + 1 个新文件**:`tests/test_device_package_discovery_current.py:34-43` 把 leaf 文件数(`== 3`)和 type_id 元组 pin 死(清单==套件是本仓库宪法),外加自己的行为测试 | 同文件 `:78-112` 用合成 descriptor 机械证明了"无 manager 门面、无 runtime 类型分支" |
| 可选 | `devices/<子包>/templates.py` 导出 `INSTALLATION_TEMPLATES`(仅当想要 `installation_template("xxx")` 快捷方式;也是自动发现) | `installation_config.py:196-218` |

全新硬件协议(如新相机驱动)额外加一个 adapter 模块(对标 `dcam.py` 587 行量级),但这属于设备本体而非骨架接线。

### 新 logic_node:2–3 个文件 + 行为测试,测试清单**自派生不用改**

| 环节 | 要碰的文件 | 证据 |
|---|---|---|
| 定义+注册 | `logic_nodes/<名>/logic_node.py` 导出 `LOGIC_NODE`(rglob 自动发现)+ 同包内的 request/执行体模块。最小实例 occupancy = `logic_node.py` 79 行 + `processor.py` 276 行,共 2 文件 | `logic_node.py:412-456`(discovery);`logic_nodes/readout/occupancy/` |
| exp.nodes API | **0 个**。`LogicNodeApis` 是 discovery 的通用投影,`exp.nodes.<api_name>` 自动出现 | `_logic_node_api.py:485-503` |
| task console GUI | **0 个**(标量字段情形)。参数面板由 `descriptor.authoring_schema`/`input_specs` 通用渲染;仅当声明 `structured` 字段才强制要一个同包 `ui/` 模块提供 `task_console_editor` | `logic_node_parameter_panel.py:148-170` |
| 测试 | `test_logic_node_discovery_current.py` 用 `descriptor 数 == rglob 文件数` 自派生(**不用改**);只需自己的行为测试 | `:12-27` |

**结论**:这一层的插件性是 v1 全仓库最成功的重构成果,两条"synthetic leaf 不碰骨架"的机械守卫测试值得原样带进新架构。真正的摩擦不在文件数,在第 4 节的耦合点。

---

## 2) 信号发布机制现状(SignalDataPlane)

**总设计:freeze-latest 拉模型,不是总线。** 核心类与职责:

- **`SignalDataPlane`**(`signal_plane.py:574`)— 唯一信号权威。producer 侧 `reserve→attach→mark_changed→publish_final/retire`;consumer 侧只有 `freeze()` 返回不可变 `SignalFront`。无回调订阅,GUI 唤醒走 `bind_owner_wake` 注册的粗粒度 wake 回调。
- **`SignalPublication`**(`:207`)— 因果单元:同一 producer 一次事务的**兄弟信号原子捆绑**,`direct_parent_refs` 记录精确消费的父事件(lineage)。
- **`SignalFront`**(`:250`)— 一次冻结的一致前沿。**同步语义明确分层**:同一 source→Processor 组件内,新 source 与其活跃派生一起替换(慢 Processor 不会把 source rev N 摆在自己 rev N-1 旁边);跨 producer **明确不做 same-shot 声明**(模块 docstring `:1-21`;`task_console/window.py:232` 同一注释)。same-shot 分组只在 producer 内部成立——camera_measurement 把同一原子相机周期的多帧放进一个 publication("One ordered frame from the same atomic camera cycle",`camera_measurement/logic_node.py:42`)。
- **buffer 在哪**:`runtime/streams.py::AcquisitionStream` 三种口——`ExactReservation`(无损游标、有背压)、`MonitorTap`(deque,`latest()` 弃旧并计 `missed`,`:998-1100`)、`FollowTap`(订阅后无损)。live 数据集积累在 `runtime/live_dataset.py::LiveDatasetPort`。
- **掉队度量**:旧 shot-clock 被 `MonitorCoverage.missed_events` 取代,per-signal 而非全局计数(`SignalValue.behind`,`:192-203`,docstring 直接解释了为什么全局 shot counter 是虚构)。
- **谁发布**:`LogicNodeHost`(`runtime/hosted_run.py`)是唯一发布者;processor 结果经 `publish_processor` 走同一 lane。
- **谁订阅**:task console 每 GUI tick `self._data.freeze()` 纯拉(`window.py:238,721,852...`)。
- **Processor 调度**:`_LatestOnlyProcessorLane`(`:409`)— 全部 processor 共享一个单 worker 线程,latest-only,慢了就跳。
- **x-y 事件关联**:`runtime/signal_source.py::SignalEventAssociationSource` + `EventRef` 提供事件级 join;pulse_scan 用它把 y 信号绑到扫描点(`pulse_scan/logic_node.py:57-61`)。

**评价**:概念集(publication=原子事务、front=冻结一致前沿、coverage 取代 shot-clock、事件关联独立于 latest 流)是这套系统里少见的想清楚了的部分,语义边界诚实(不承诺做不到的 same-shot)。**但 2418 行单文件同时管:generation 生命周期、依赖闭包回收、processor lane、derived 信号、front 构建、lineage 保留**——重写时应拆成 4 个可独立测的模块(见第 4 节)。

## 3) logic_node 抽象:本质 vs 过度设计

**现状**:`LogicNodeDescriptor` = 纯数据 + 两个回调(`build_request`、`bind_execute`),`LogicNodeHost` 是唯一通用宿主(start/cancel/poll/shutdown),kind 三分 task/measurement/processor(`catalog.py:76`)。

- **输入声明**:封闭和类型 `DatasetInputSpec | ArtifactInputSpec`(`input_spec.py:138`),按 `contract_id` 匹配,无 resolver/回调/GUI 元数据。
- **输出声明**:`DatasetOutputSpec`(name+contract_id);动态输出 `resolve_outputs` 与静态互斥(`logic_node.py:283-284`)。
- **repeat contract 已消失为结构事实**:repeat 是 `zlc_data.DatasetSchema` 的 `repeat_axis`(role=REPEAT,`runtime/dataset.py:217-245`),不再是节点级声明;camera `repeat==0` 即 live monitor 语义(`camera_measurement/logic_node.py:56`)。旧 `test_processor_repeat_contract` 那类契约测试无对应物、也不再需要。
- **reactive 边已消失**:旧 reactive ring 收敛为"processor kind = 恰好 1 个 Dataset 输入 + latest-only lane"(`hosted_run.py:172-187` 硬性强制)。

**本质、必须保留**:descriptor 数据化声明;单一通用宿主;kind 三分;contract_id 输入输出匹配;`device_requirements` 按 capability 声明并在 descriptor `__post_init__` 与 authoring schema 交叉校验(`logic_node.py:286-311`);latest-only processor lane;事件关联。

**过度设计、可删/可简**:
1. `LogicNodeApplicationContext` 15 个成员(五个 `*_root` 路径、artifact 记忆、open_ui、operation_guard…,`logic_node.py:127-221`)——docstring 自辩"不是 service locator"恰说明它在滑向 service locator。核心其实只有 `device()/input()/start_run()/signal_plane`。
2. `UiContribution` 的 module+symbol **字符串间接** + discovery 期的 ui-prefix 归属校验(`:441-447`)——为"域包不 headless import Qt"付出的仪式;一个 lazy callable 即可。
3. descriptor `__post_init__` 约 130 行防御校验、全库每层重复 isinstance——这是对抗 vibe-coding 回归的免疫系统,新架构在包边界收一次即可,不需要这个密度。
4. `operations` mapping + `NodeApi.__getattr__` 动态转发 + 保留字集合(`logic_node.py:366-386`、`_logic_node_api.py:450-457`)——为 pulse_scan 的 load/materialize 三个方法引入了一整套元机制。
5. `logic_node.py` 顶层 `import zlc_plot`(`:30-31`)只为 `TaskPreview` 声明——域包对画图包的硬依赖,应改为不透明 preview token。

---

## 4) 阻碍"插件式可拆卸"的前 5 个耦合点

1. **固定命名空间发现 + 清单 pin 测试(有意的封闭)**。`discover_logic_nodes`/`discover_device_types` 硬编码 `zlc_neutral_atom.logic_nodes` / `.devices` 包(`logic_node.py:416`、`device_types.py:99`),docstring 明写 "never plugins";device 测试 pin 死 type_id 元组。第三方包想带自己的 device/node 进来,发现机制本身要改。这是宪法级决策而非疏忽——拆包时必须显式重新裁决:**建议保留"编译期封闭清单"精神,但把清单从"扫固定包"改为"各包导出、composition root 汇总"**,让 `zlc_neutral_atom` 不再是唯一合法宿主。
2. **`LogicNodeApplicationContext` 的唯一实现长在应用层**。`_BoundLogicNodeContext` 在 `Zou_lab_control/api/_logic_node_api.py:41`,leaf 类型面(Protocol)在域包、实现和语义(default_artifact 解析顺序、workbench handle 注册)在应用包——节点包无法脱离 `Zou_lab_control.api` 单独跑通集成测试。重写时 context 实现应与宿主同包,应用层只注入原料。
3. **域包内嵌 Qt UI**。`logic_nodes/*/ui/*` 直接 `import PyQt5` + `zlc_frontend`(`pulse_scan/ui/task_console_parameter_form.py:9-17`),import-DAG 测试为此专门开洞(`test_architecture_import_dag.py:88-109`)。zlc_neutral_atom 因此永远不能宣称 headless。拆包方案二选一:UI leaf 随节点走但作为独立 optional 子包发布,或 UI 全部搬到 workbench 侧按 contract_id 认领。
4. **capability 值是无 schema 的 `object`**。factory 返回 `Mapping[str, object]`,`require_capability` 返回 `object`(`installation_runtime.py:114-126`),token 字符串→值类型的对应关系全靠散落的 isinstance(`BoundCapturePort`、`RemotePulseExecutionClient`…)。更糟的是虚拟设备用私有 capability 当设备间握手通道(`_VirtualSequencerConnection` 经 `simulation/device_types.py:287→300→326` 在同文件三个 factory 间传递)。新架构应有一张**机器可见的 capability→类型表**(哪怕就是 dict[str, type] 常量+契约测试),私有握手改走显式注入。
5. **camera/sequencer 的"插件缝"太厚**。名义缝是 `CameraAdapter` Protocol,但它约 10 个方法(arm/read_frame_records/finish_record_capture/capture_state…,`contract.py:1289`),外面还套 `CameraMonitorEndpoint` 1632 行、broker 绑定仪式(`verify_identity→bind→verify_capability`,`simulation/device_types.py:152-195` 每个设备重抄一遍 ~40 行)、`capture/` 4.5k 行 pipeline。sequencer 侧同构(endpoint 528 + port 829 行)。加一台"简单设备"(电源、波形计)按现在的仪式成本会显得荒谬——**新架构需要一个 10 行就能接入的 trivial-device 快速路径**,broker 仪式做成一个 helper 而不是每个 factory 的样板。

## 5) session 门面现状与重写建议

**现状清单**(`facade.py` 832 行 + `_application_services.py` 567 行,Experiment 实际管):生命周期状态机与 close 竞争仲裁(`close()` 约 140 行:双轮 GUI close、operations_drained、close_attempt 所有权);Run 准入控制平面(`application_start_run` 90 行:admission_lock、**preemption 依赖闭包回收** `_retirement_for_blockers_locked` 联动 signal_plane 的 `withdraw_dependency_closure`);GUI 句柄注册表(`open_workbench_handle`);default_artifacts 记忆;完整 `PulseFacade`;安装热替换(`_replace_installation` 90 行:冻结准入→close→重连→失败回滚旧配置);workspace 路径;nodes 投影。

**好的一面**:runtime(`_InstallationRuntime`:设备生命周期+Run 执行+反向 closer)与 services(准入+GUI+close)边界已经清楚,`_InstallationRuntime` 225 行小而正确,可近乎照搬。

**重写建议的职责切分**:
- **InstallationRuntime**(照搬):设备图组合、capability 寻址、Run 执行、shutdown。
- **Session(headless)**:准入 + signal plane + node host 注册 + default_artifacts。**GUI 句柄注册表和 close 里的 GUI 泵循环(`wait_for_close_attempt` 的 0.05s 轮询泵)移出去**——headless session 不应知道 Qt owner 线程的存在,这是现在 close 复杂度的主要来源。
- **Preemption 自动闭包回收降级**:它是全仓库最重的并发机制(signal_plane `:930-1037` 三个方法 + services 60 行 + admission 两段式重试),服务的场景只是"新 Run 顶掉旧 live monitor"。重写先做显式 stop-then-start,把"自动抢占"当作以后有实据再加的功能。
- **热替换简化为 close+connect+失败回滚**(它内部本来就是这个),不保留"保住同一个 Python 对象身份"的 `_binding` 偷换技巧(`facade.py:400`)。
- **PulseFacade 随 pulse 域走**:它的实体在 `devices/sequencer/application.py`(753 行 PulseApplicationOwner),不应由 session 门面代持。

## 6) 虚拟设备机制:值得照搬吗?

**机制**:虚拟设备是同一 `DeviceTypeDescriptor` 机制下的**平行 leaf 类型**(`sequencer.virtual`/`rf.virtual`/`camera.virtual_readout`/`camera.virtual_mot`),走同一 `create_installation`/broker/port/pipeline,全链路无一处 `if virtual` 分支(`test_virtual_template_composes_through_generic_capabilities` 机械守卫)。fake 面:

- **相机:铁律完全达标**。`VirtualCamera` duck-实现 `CameraAdapter` Protocol(`apparatus.py:1084`,注释明说只 fake 到 adapter 契约),往上 `CameraMonitorEndpoint`、`BoundCapturePort`、capture pipeline、全部 logic node **逐字节共享**。
- **sequencer:fake 面高一层**——`VirtualSequencerExecutionEndpoint` 实现的是 broker 命令端点而非硬件帧,但共享 `BoundPulsePort` 与整个 zlc_pulse 编译链。
- **物理因果保真**:`VirtualCamera` 从 in-process FIRE 回调按真实触发沿产帧(`simulation/device_types.py:251-253` 注释),虚拟 MOT 相机还读 DAC coil 口模拟磁场响应。

**照搬**:(a) 虚拟=平行 device 类型、零 runtime 分支;(b) fake 面选在窄 Protocol(相机模式),这是"virtual 走实机同一代码路径"铁律的正确实现形态;(c) 触发→帧的因果耦合;(d) 用合成 descriptor 的可组合性契约测试。

**不照搬**:(a) `_VirtualSequencerConnection`/`_VirtualRfConnection` 伪装成 capability 做设备间私有握手(见第 4 节第 4 点)——新架构给仿真世界一个显式 `SimulationWorld` 对象由 installation 组合注入;(b) 虚拟几何(5×7、96×128、9px 间距)硬编码在 `_imaging_geometry`(`simulation/device_types.py:84-100`)而真实相机同类事实走 authoring schema——两者应同源;(c) `_connect_sequencer` 依赖 `load_deployed_pulse_target/geometry`(`:267-268`)即部署态 fpga board_config 文件——**最小虚拟运行集被硬件描述文件拖住**,拆包时 virtual 应能用内置默认 target 起跑,把"与部署 bitstream 同几何"降为可选校验。

**总判断**:v1 的 device/logic_node 层已经把"插件性"做到了骨架级(定义即注册、GUI/session 零接触、机械守卫齐全),值得整体继承其**概念**;但发现机制的单包封闭、capability 无类型表、context 宽面、域包内嵌 Qt、设备接入仪式厚这五点,是拆成独立包前必须重切的边界。