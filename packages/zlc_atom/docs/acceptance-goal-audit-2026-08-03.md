# 任务2:GOAL 逐项符合性审计(zlc_atom)

**审计方式**:全量通读 zlc_atom src/tests/notebook/GOAL/README + 与参照树逐模块对照 + 在 scratchpad 副本实测(pytest、notebook 等价脚本、数值对比、bug 复现)。参照树与被验收仓均未写入。

## 总判决:**返工**(骨架概念面可保留,四处硬伤必须重做,详见文末范围)

实测基线:15/15 测试绿;notebook 等价脚本顶到底可跑;src+tests 共 3,924 行(GOAL 预估 15-20k,-75% 无任何说明)。

---

## A 半区逐项

### A0 引导 —【部分】
- ✅ git init、src 布局、pyproject 发行名 `zlc-atom`、numpy+zlc-data 依赖(pyproject.toml:6,11-14)、顶层 allow-list(`__init__.py:15-17` + test_import_boundaries.py:12-16)、import 纯度守卫禁 Qt/matplotlib/zlc_plot/zlc_ui/zlc_neutral_atom(test_import_boundaries.py:19-23)、A 半区零 runtime/pulse 引用(:25-37)。
- ❌ `zlc-data` 声明后**全仓零 import**(死依赖);"守卫按子包分白名单"只做了 A 半区半边,B 半区白名单无对应物。

### A1 物理数学迁入 —【部分,oracle 一项定性为缺失】
- ✅ bimodal 忠实迁移:逐函数对照树 `logic_nodes/readout/bimodal.py:43-254`,算法结构同构;scipy erf→math.erf、scipy minimize_scalar→黄金分割 96 轮(bimodal.py:124-143)。**实测数值等价**:同一双峰样本 Δthreshold=4.3e-7,fidelity 全同。
- ⚠️ calibration 数学=**重写非迁入**:树内 characterize_readout 链(analysis.py:596-1268 的 2D 高斯拟合/亚像素 refine/晶格正则化/train-test split/held-out 阈值/strict-consensus labels)全部无对应物;zlc_atom calibration.py:433-485 是自创的 130 行最小 calibrate()。psf 同理:测量核拟合链(analysis.py:936-1046)未搬,psf.py 只有解析核。
- ⚠️ 逐站点保真度=自创 `per_site_fidelity`(bimodal.py:257-271)+ report 里 `site_fidelity` 对 short_signals **再拟合 MODEL** 保真度(calibration.py:483),非树内 held-out 语义。
- ❌ **冻结 oracle 未随迁——自产 oracle 坐实**:`tests/fixtures/main_readout_oracle.json:3` 自述 `"authority": "frozen hand-authored physical quantities"`,内容仅 normal_cdf 三个标准点+3×3 box mean/sum(解析平凡值);树内同名 oracle(`tests/fixtures/main_readout_oracle.json` + `.npz`,format `main-readout-oracle`,authority_commit `6c337d49`,冻结帧+box/psf/阈值/labels/split 全覆盖)完全没搬,键名零重合。测试名 `test_frozen_readout_oracle_is_not_regenerated_from_a_reference_tree`(test_physics.py:12)是话术。scipy→黄金分割这类数值替换恰好落在 oracle 零覆盖区,"含数值算法必带冻结 oracle" 铁律实质未达。

### A2 相机 adapter —【部分,dcam 一项定性为缺失】
- ⚠️ "收窄到最小面,以三个真实现倒推"未发生:6 成员协议是树 `devices/camera/contract.py:1212-1276` **原样拷贝**(树上已是窄面)。结果可接受,叙述不实。
- ❌ **dcam(qCMOS)适配迁入=空壳**:zlc_atom dcam.py 共 98 行,driver 全 duck-type 注入(dcam.py:32-39),真 Hamamatsu 驱动(树 dcam.py 750 行 + _dcam_driver.py 511 行 ctypes 绑定)一行未搬,真机 qCMOS 路径不存在——这正是 ROADMAP 焦点。
- ⚠️ VirtualCamera:fake 面=adapter 协议 ✅、有界队列真丢帧 ✅(virtual.py:156-158)、触发→帧因果 ✅;但**无产帧线程**(树 apparatus.py:1128-1129 有 worker 线程),且有实测 bug(见文末);虚拟标定参数只保留 0.107/0.43/200,bg 300→350、sigma 0.7→1.2 被改,且 `test_calibration_report_and_noise` 派生关系测试不存在。
- ✅ pylon 后置已记 README(README.md:21-23)。

### A3 执行引擎 —【部分】
- ⚠️ run/ports/resources 概念面在(RunPlan/RunContext/SafetyInterrupt/RunHandleLike run.py:24-73;DeviceBroker 全回调注入 ports.py:62-156;PhysicalDeviceIdentity/DeviceBindingStamp resources.py:36-57),但 614 行 vs 树 1,931 行=重写非迁入,且**整个 run 引擎+ResourceArbiter 零测试覆盖、全仓零调用方**。
- ✅ broker 仪式 helper 化:`bind_verified_device`(ports.py:159-181),两个 device factory 均复用(camera device_types.py:26、sequencer device_types.py:17)。
- ❌ trivial device 快速路径:`bind_trivial_device` 存在(ports.py:183-200)但 **GOAL 点名的合成 device 测试缺失**(tests 全仓零引用)。
- ⚠️ 双 capability 表:`DeviceBroker.CAPABILITY_TYPES = {}`(ports.py:65)恒空,verify_capability 的类型检查永不触发,与 install/descriptors.py:15 的真表双源。

### A4 虚拟物理 —【部分】
- ✅ SimulationWorld 显式对象+显式触发路由(world.py:40-113),`_VirtualSequencerConnection` 式私有握手已消灭(机械 grep 守卫在 test_import_boundaries.py:21)。
- ❌ 但两个 factory 都留了兜底 `... if isinstance(context.world, SimulationWorld) else SimulationWorld()`(camera device_types.py:43、sequencer device_types.py:31)——miswire 时各设备静默各造一个世界、因果断裂,违"显式注入"与用户"哨兵绝不兜底"铁律。
- ⚠️ 几何同源:SimulationGeometry 单点(world.py:12-37)+工厂等值门(camera device_types.py:52-53)方向对;但 `VirtualCamera._default_frame`(virtual.py:103-119)保留了**第二套完整成像物理**(裸构造即启用,test_camera_and_execution.py:19 正在走这条),仍是两处硬编码。
- ✅ 内置默认 target:全仓无 board_config 引用(以"根本没有 target 概念"的方式满足)。

## B 半区逐项

### B0 契约 fake —【缺失(装饰性合规)】
两份 contract.md 已核实存在。但:
- `tests/fakes.py` 的 FakePlane/FakeNodeHost/FakePulseStreamer **零消费者**(grep 全 tests 无一 import)——为对上 GOAL 条文而写的摆设。
- 真正被节点消费的 plane 是 **src 内自研** `nodes/_framework/signal_plane.py`,与 zlc_runtime/docs/contract.md 多处签名冲突:`publish_final` 返回 SignalPublication 而非 `Mapping[str, SignalValue]`(contract.md:21-24 vs signal_plane.py:79-80);`reserve`/`retire` 返回 None 而非 StreamGenerationId/frozenset(contract.md:16-17 vs :96-101);`publish_processor(control, source_publication, outputs)` 位置参数 vs 契约 keyword-only(contract.md:32-37 vs :82);`SignalValue=(signal_key,value,publication,behind)` vs 契约 `(block/schema/values/behind)`;`attach_latest_only_processor`/`cancel_latest_only_processor` 缺失。**这就是 GOAL 第 9 行明令禁止的"自己发明变体"**,且放错位置(应为 tests/ 共享 fixture 唯一同步点)。B5 换真包时 measurement.py:41,90、processor.py:94 全部断。

### B1 descriptor 与发现 —【部分】
- ✅ 数据化声明/kind 三分/contract_id/processor 恰一 Dataset 输入(descriptor.py:98-99);rglob 发现(discovery.py:15);capability token→类型表+契约测试(descriptors.py:15-20 + test_installation_and_nodes.py:27-34);context 四件收缩成形(context.py:9-37);operations/`__getattr__` 元机制未带入;域包零 Qt。
- ❌ "合成叶子不碰骨架"**机械守卫缺失**:现有的只是清单 pin(test:37-41,`==3` + rglob 自派生),没有"注入合成叶子→发现自动收编→骨架零改动"的证明测试(树内有,survey 第 1 节点名要原样带走)。
- ❌ `ApplicationContext` 与 `NodeHost` 全仓**零实例化零测试**=死代码,节点从未在宿主/上下文下跑过。

### B2 sequencer 设备 —【部分】
- ✅ PulseStreamer 契约面与 zlc_pulse contract.md 逐方法一致(protocol.py:33-43,DoneReport 四元组 ✓);薄转发 SequencerDevice(virtual.py:109-145);虚拟孪生 fire→world→相机因果(virtual.py:74-76 → world.py:108-109);session/epoch/PulseFacade 未长回 ✅。
- ⚠️ 偏差:VirtualPulseStreamer 无 geom/clock_hz 参数、open() 无 fingerprint 比对(contract.md:22-23);transport 与内部状态脱钩(DictRegisterTransport 死件);SEQUENCER_SCHEMA 为空且 factory `del values`(device_types.py:30)→模板配置 `{"camera_key": "camera"}`(templates.py:8)是**从不校验的死配置**。

### B3 installation graph —【部分】
- ✅ 拓扑排序→factory→capability 类型校验→反向 closer(graph.py:114-134);模板只留 virtual 全链(templates.py:5-10)。
- ⚠️ "225 行小而正确近乎照搬"未发生(137 行重写);"故障隔离 open"变成了失败整体回滚(graph.py:130-133),树内 DeviceSet.open 的逐设备隔离语义没复刻;broker `_active` 恒空(ports.py:72,154)使 shutdown 检查空转。

### B4 三个 logic node —【部分】
- camera_measurement:same-shot 捆绑 ✅(measurement.py:90,一次 publication;粒度是整个 finite run 而非每 cycle);repeat==0 monitor ✅(:70-72);❌ **runtime exact/monitor 流口未接**(直接读 adapter,AcquisitionStream/ExactReservation/MonitorTap 全仓零引用);❌ capture pipeline "减肥迁入"实为弃迁+90 行重写,点名的 **LOC 报告不存在**。
- occupancy:reactive 样板+`evaluate(SignalValue)` 面 ✅(processor.py:79-82),lineage 断言有集成测试(test:56);"79+276 行近乎平移"未发生(重写 129 行)。
- calibration:task 存在但**不驱动设备采集**(task.py:23-34 吃现成帧数组),却声明 camera+sequencer device_requirements(logic_node.py:32-35)——声明与行为脱节;"吃 oracle"未做。
- "各自的行为测试":只有 2 条集成测试(test:44-80)兜底;**monitor 路径(repeat=0)、PSF 路径经 task、NodeHost 下运行,全部零测试**。

### B5 集成收尾 —【缺失】
- ❌ 真包替换 fake 未发生:zlc_runtime/zlc_pulse 未安装、未尝试,**阻塞记录空白**(GOAL.md:44 仍是模板行)——GOAL 明文"未装应记阻塞"。
- ⚠️ usage.ipynb 存在、实测顶到底可跑(等价脚本验证,rate 输出正常),但缺"拉 front"步骤(全程未 freeze());README 有叶子教程文字(README.md:13-19)但"写成可跑的合成叶子测试"缺失。
- ❌ "输出对 oracle" 未做(oracle 本身名存实亡)。

## 机械终态判据
1. ✅ 15/15 绿(scratchpad 副本实测);A 半区不需 runtime/pulse。
2. ✅ 禁词 grep 零且有机械测试(test_import_boundaries.py:19-23);`if.*virtual` runtime 分支形态实查为零。
3. ❌ 四件只有 1 件:capability 类型表契约 ✅;合成叶子守卫 ✗;virtual==real 契约测试 ✗(不存在该形态);oracle 测试形存实亡。
4. ❌ notebook ✅,但 **LOC 报告不存在**,3,924 行 vs 预估 15-20k 的 -75% 缺口零说明(缺口本质=大量"迁入"变成了最小重写)。

## 特别核对项汇总
- **A2 收窄来源**:非倒推,是树窄协议原样拷贝(可用但与 GOAL 叙述不符);dcam 真驱动未迁=真机路径缺失。
- **A3 合成 device 测试**:缺失(bind_trivial_device 零测试)。
- **A4**:显式注入被 else 兜底稀释;几何双源残留(virtual.py:103-119);默认 target ✅。
- **B1**:合成叶子守卫缺;token→类型表 ✅ 但 broker 侧第二张空表;context 四件=死代码。
- **B4**:见上,行为测试粒度不足、monitor 零覆盖。
- **B5**:notebook ✅ 可跑;README 教程文字有、可跑合成叶子测试无;集成未做且阻塞未记。
- **仪式违规**:GOAL 清单 **0/11 勾选**、状态仍 `IN PROGRESS`(GOAL.md:3)却对用户宣告"已结束";阻塞记录空白;commit 粒度=35 分钟内 3 个巨型主题混装 commit(2fe196a=A0-A4 全部,ffaa319=B1-B4 全部 1,367 行,af68067=全部测试+oracle+notebook 一把梭),非"每主题小 commit"。
- **范围纪律**:叶子/设备无偷带(4 device types/3 nodes,test:16-24 pin 住);但**偷带了一个自研 runtime 内核**(src 内 SignalDataPlane/NodeHost/ApplicationContext,应是按契约写的 tests fake);`"simulation.world"` capability 注册但无设备提供=死表项(descriptors.py:19)。README 声称大体符合实况,唯 "Runtime and pulse integration is contract-shaped"(README.md:10-11)对 runtime 侧不实。

## 实测新 bug(复现确认)
`VirtualCamera.read_frame_records(exact=True)`(virtual.py:166-173):等待循环条件是 `not self._queue`,**首帧一到即停止等待**,随后 `len(queue) < requested` 直接抛 TimeoutError——预算 1s、帧异步逐个到达时必炸(脚本复现:armed 2 帧、50ms/150ms 各到 1 帧、read(2, timeout=1.0) 抛错)。当前全绿只因虚拟链路同步产帧;真机异步节奏下 exact 采集即失败。

## 返工范围(建议保留 vs 必须重做)
**可保留**:physics/bimodal(数值忠实已验)、camera 协议面、sequencer protocol.py(契约一致)、install 发现/descriptor/类型表骨架、import 纯度守卫、authoring.py、SimulationWorld 触发路由思想。
**必须重做**:① oracle 从树 `tests/fixtures/main_readout_oracle.{json,npz}` 真随迁并接上 fit/otsu/threshold/psf 全覆盖;② 删 src 内自研 signal plane/host/context,按 zlc_runtime contract.md 签名重写为 **tests/ 共享 fake**(唯一同步点),节点改写到契约签名上,现有 fakes.py 摆设废除;③ dcam 真驱动迁入(树 dcam.py+_dcam_driver.py);④ 补四类缺失守卫(合成叶子、合成 trivial device、virtual==real 契约、虚拟标定参数派生)+ monitor/run 引擎测试;⑤ 修 VirtualCamera exact 读+产帧线程;⑥ B5 真集成或逐条记阻塞;⑦ LOC 报告+GOAL 清单如实勾选/记录;⑧ 清死件(zlc-data 死依赖、broker 空表、camera_key 死配置、simulation.world 死表项、工厂 else 兜底)。