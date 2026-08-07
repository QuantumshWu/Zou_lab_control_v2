# GOAL 归档 — zlc_atom 已完成轮次

> 已完成并验收的轮次原文(W0-W7 / M / v3 R1-R4 / v3.1 / v3.2),留作证据与追溯。**活的计划在 `GOAL.md`**。

## W0 基线与仪式修复

- [x] W0.1 完整读三份 `docs/acceptance-*-2026-08-03.md` 与两份契约(`..\zlc_runtime\docs\contract.md`、`..\zlc_pulse\docs\contract.md`);确认能读到参照树(读不到立刻记阻塞停工)。
- [x] W0.2 `zlc-data` 死依赖裁决:pyproject 声明了但全仓零 import——物理/节点数据面本应消费它(DataBlock/OwnedSnapshot/角色轴),二选一:真用起来,或从依赖删除并在 README 说明;不许挂空。
- [x] W0.3 守卫自证补课:`test_import_boundaries.py` 的 rglob 扫描加"文件数>0"断言(验收实证 count==0 时守卫 vacuous 绿);补 B 半区白名单(A 半区禁 runtime/pulse 已有,B 半区按契约允许);补 `if virtual` 运行时分支 grep 守卫。

## W1 冻结 oracle 真迁移(一切物理修复的前提)

- [x] W1.1 树内 `tests/fixtures/main_readout_oracle.json` + `main_readout_oracle.npz` **原样拷入**(main@6c337d49 冻结,10 个 authority 函数+完整合成帧 60×2×34×40);删除自产的同名 json(其中的手算例可改名 `hand_examples.json` 留作补充测试,不得再冒充 oracle)。
- [x] W1.2 参照树内 `test_zlc_readout_main_oracle.py`(760 行)为 zlc_atom 的 API 写映射消费断言(rtol=1e-12/atol=2e-12 量级):normal_cdf、box 四 reducer、psf 窗口(padding=3)、fit_bimodal 全组件+threshold、find_site_centers、classify、端到端 calibrate。**此项落地时大部分断言应当是红的——那是 W2 的工作清单,不许调松容差让它绿。**

## W2 物理修复(全部以 W1 断言锁死;证据:acceptance-fidelity §1)

- [x] W2.1 **calibrate() 补 short 帧表征链**(v1 致命伤:阈值拟在 reference 用在 short 帧,同输入错误率 29/360→122/360):共识标签(逐 shot (60,6) 语义,修掉现在 (6,) 的无意义 labels)→ 种子化 train/test split → 直方图经验阈值(bins=120)→ held-out confusion fidelity;以树内 `characterize_readout` 链为底本迁移。判据:oracle 同输入错误率==29/360。
- [x] W2.2 threshold 求解器回 bounded 最小化(黄金分割落零误差平台左缘,oracle 断言必挂;或黄金分割+平台中点规约,须过 oracle)。
- [x] W2.3 `find_site_centers` 迁树内实现(高斯平滑+maximum_filter+curve_fit 亚像素+格点修复;树内 admission 0.1px,现 0.455px 且私放容差 2.0)。
- [x] W2.4 psf:padding 贯通(现硬编码 2,oracle=3)+ 数据驱动 kernel 拟合链迁入(`_fit_psf_features`;现理想高斯与 oracle kernel 偏差 0.10≈峰值 60%)。
- [x] W2.5 `per_site_fidelity` 恢复 held-out 平衡 confusion 语义(现=全样本 accuracy,漂移 0.167;这是记忆点名的"数据驱动逐站点保真度"旗舰点)。

## W3 契约对齐(证据:acceptance-tests §3)

- [x] W3.1 `src/nodes/_framework/signal_plane.py` 影子变体**移出 src**,降级为 `tests/` 契约 fake,签名逐条对齐 runtime 契约:`reserve→StreamGenerationId`、`retire→frozenset[str]`、`mark_changed(producer, live_slot)`、`publish_final/publish_processor→Mapping[str,SignalValue]`、`publish_processor(control, outputs, *, source_publication)` keyword-only、`SignalValue=(block/schema/values/behind)`、补 `attach_latest_only_processor`/`cancel_latest_only_processor`。
- [x] W3.2 三个节点的调用面全部改向契约签名;`tests/fakes.py` 三件成为被真实消费的共享 fixture(FakePulseStreamer 补 `(transport, geom, clock_hz)` 构造与 open() 指纹比对),或删。
- [x] W3.3 occupancy 的 lineage 集成测试升级:counts/rate 断**数值**不只 shape(现被三处 mutation 全绿存活打脸);阈值不许硬编码 `full(6, 50.0)`,从标定产物来。

## W4 dcam 真迁移

- [x] W4.1 树内 `dcam.py`(750)+ `_dcam_driver.py`(511,ctypes DCAM-API)+ `_owner_lane.py`(80,SDK 属主线程)迁入 `devices/camera/`,现 98 行占位壳删除;属性编程/ROI/binning/trigger 配置/丢帧计数(现硬编码 0)恢复。无真机环境无法跑硬件测试=正常,迁移正确性以"代码底本一致+纯逻辑层测试"验收;确需整体后置必须用户批准并在此改写。

## W5 虚拟物理对齐(真机同路径资格;证据:acceptance-fidelity §2)

- [x] W5.1 标定参数对齐 qCMOS 冻结值并加派生关系守卫:σ=0.7px、bg=300、atom_rate=1100(信号=rate×曝光派生,非固定幅度);gain 0.107/read-noise 0.43/offset 200 保持。
- [x] W5.2 Poisson 光子统计恢复(现只有高斯背景近似);产帧 worker 线程真丢帧(现调用线程同步产帧);帧 dtype native 整型(现 float64,违相机链铁律)。
- [x] W5.3 `VirtualPulseStreamer.fire` 按已 load 程序的**基周期相机窗口数**产帧(现恒 1 帧不读程序;树内"每帧曝光判据"整套迁入)。
- [x] W5.4 删两个 device factory 的 `SimulationWorld()` 兜底(miswire 时各造世界因果断裂;哨兵绝不兜底)与 `VirtualCamera._default_frame` 第二套成像物理(几何单源)。

## W6 execution 安全最小集与杂项(证据:acceptance-fidelity §2 execution 行、goal-audit A3)

- [x] W6.1 以树内 run.py/ports.py 为底本补 fail-closed 最小集:取消即封存(`_seal_execution_on_cancel`/`_revoke_hardware` 语义)、run 域设备租约与吊销(现任何持 binding 者任意时刻可 execute)、身份凭证只能 broker 铸造(现 IdentityProof 裸 dataclass 可伪造)。
- [x] W6.2 修空洞与死件:`broker._active` 恒空使 shutdown 检查空转;`SafetyInterruptError` 定义未用;`DeviceBroker.CAPABILITY_TYPES={}` 与 install/descriptors.py 类型表双源合一。run 引擎补测试(现零覆盖零调用方)。
- [x] W6.3 守卫补强:合成叶子真守卫(注入假叶子证明"发现自动收编、骨架零改动");trivial device 合成测试(经 installation graph 全链);capability 实例校验(installation 产物 isinstance 声明类型)。

## W7 收尾(A/B 残余 + 集成)

- [x] W7.1 camera_measurement 的 monitor 路径(repeat==0)行为测试;install graph 恢复"故障隔离 open"语义(现失败整体回滚,树内是逐设备隔离)。
- [x] W7.2 B5 集成:zlc_runtime/zlc_pulse 可 editable 安装后,虚拟全链真跑(installation(virtual)→camera live→occupancy 派生→freeze 族一致→calibration task→对 oracle);未就绪**记阻塞停在此项**。
- [x] W7.3 `notebooks/usage.ipynb` 实况化:标定用足量帧(现 2+2 帧 fit 必不 ok、阈值全走兜底=仪式台架),带执行输出提交;README 定稿(叶子教程+LOC 报告:各模块行数 vs 树内底本,重写/减肥处逐项说明)。

## 机械终态判据(全绿才 GOAL COMPLETE)

1. 干净 venv `pytest -q` 全绿;W1 的 oracle npz 消费断言全部存在且绿(容差 1e-12 量级,不许放松)。
2. **同冻结输入端到端错误率 == 树内 29/360**(写成断言)。
3. 验收报告点名的三处 mutation(threshold+17.3 / classify 极性翻转 / occupancy rate 反相)逐一验证**必红**(临时副本实验,记录在 commit message)。
4. 节点调用面与 `zlc_runtime/docs/contract.md` 签名 diff 为零;src/ 内无任何 plane/host 影子实现(grep `class SignalDataPlane` 在 src 零命中)。
5. grep 为零(src/):`PyQt5`、`matplotlib`、`zlc_plot`、`zlc_ui`、`zlc_neutral_atom`、`if.*virtual` 分支、`SimulationWorld()`(工厂兜底形态)。
6. 守卫全部自证非空洞(空扫描红);清单勾选与 commit 一一对应;工作树干净。

## M 复验修复轮(2026-08-03 复验产出;先读 docs/reacceptance-2026-08-03.md;全部勾完并按判据验证后才改回 COMPLETE)

- [x] M1 **必修:occupancy 反相 mutation 必须能被杀死**(现 M3 存活双根因均已实证):① `tests/test_mutation_guards.py:50-53` 是空洞守卫——只在冻结数据自身上断言 `1-rate != rate`,从不调用 `OccupancyProcessor`,对任何实现变异永久绿——删除或重写为真调用实现的版本;② 唯一实现级断言取在简并点——虚拟世界默认占据每 shot 恰 3/6(`simulation/world.py:87`),`mean≡0.5` 时 `1-rate==rate` 不可辨。两路任选或都做:(a) 用 oracle 的 `runtime_probe_indices` 帧驱动 `OccupancyProcessor.process`,对 `runtime_rate_box` 断数值;(b) 集成测试用 `world.set_occupancy` 设非对称占据(如 1/6)离开简并点。修后在临时副本重放 M3(`nodes/occupancy/processor.py:138` 反相)确认必红,并按终态判据 3 把三处 mutation 实验结果记录进 commit message(现全部 31 个 commit 均无记录)。
- [x] M2 **R1 跟改(runtime 契约已于 201f624 校订)**:`tests/fakes.py:55-70` 的 `attach_latest_only_processor` override 与 `tests/test_contract_fakes.py:18-23` 的签名钉,从旧拼写 `(signal_key, control, initial_publication)` 改为契约现行拼写 `(node, *, source_name, initial_publication)`;src 三节点零改动(无调用方)。
- [x] M3 (建议,非阻塞)极性判决单源化:`calibrate()` 内联判决(`physics/calibration.py:880`)收敛到 `classify_threshold`,消除极性双实现(M2 mutation 只被两处直连测试抓住,occupancy 链无感的根源)。顺手删 `src/zlc_atom/nodes/_framework/__pycache__/` 里 W3.1 前的 stale .pyc。

---

# §v3 架构修正轮(2026-08-04 用户三裁决;动工前读完本节全文)

> **背景(用户拍板,不是建议)**:
> ① **measurement = 纯观测**:不得持有或操作 sequencer。激励源(装什么 pulse、何时 fire)是实验状态,归实验者(notebook 手动)或 task(编排)所有。现状违背:`nodes/camera_measurement/measurement.py:128-132` 构造强制要求 sequencer,`:176` 采集循环逐帧 `self.sequencer.fire()`——这是虚拟世界特化的假节拍,真机 pulse 常 repeat_forever 连续跑,逐帧 fire 不存在,违"虚拟真机同路径"。
> ② **task = 编排,默认包含 sequencer 操作**:默认从约定位置按名字 load pulse 定义,配置并 fire sequencer,调 measurement 采集,调数学计算,发布结果——全自动,允许参数覆盖 pulse。现状违背:`nodes/calibration/task.py` 是纯函数壳,要用户自己喂现成帧,声明了 camera+sequencer 设备需求却零设备操作。
> ③ **一切逻辑归 owner 文件夹**:每个 logic_node / device 的全部内容都在自己文件夹里。`physics/` 与 `simulation/` 是从 v1 `core/` 照搬的横切残余,解散归位。
>
> **v2 血训延续**:上一轮验收判据只压物理保真,没压"task 从 pulse 到 report 全自动跑通",于是编排层交付了壳。本轮判据以编排全链为核心。

## v3 补充铁律(叠加在顶部铁律之上)

- **V-A 验收判据即编排**:task 的"完成"定义是"虚拟设备下从 pulse 定义文件到 report 一条调用全自动跑通",数学函数能算不算完成。
- **V-B 声明=实现**:descriptor 里 DeviceRequirement 声明的每个设备,必须被机械证明真的被该节点使用(见 R2.4 守卫);声明不用=删声明,用了没声明=补声明。
- **V-C pulse 定义=Python 源文件**:zlc_pulse 没有文件存档 API(已核实:只有 compile()/device.load(prog)),不许发明新序列化格式。pulse 文件形态 = `pulses/<name>.py` 模块,暴露构建函数,文件即源。`src/zlc_atom` 包本体**不 import zlc_pulse**(依赖边界不变,pyproject 不加依赖);`pulses/` 目录里的定义文件与 tests 可以 import zlc_pulse(运行环境有);虚拟链可用满足 PulseStreamer.load(prog) 的任意 prog 对象。
- **V-D 迁移=搬家**:R3 的文件迁移保持内容逐字节等价(允许仅改 import 行);冻结 oracle 文件与全部断言值一个字符不动;迁移 commit 与行为改动 commit 分开。

## R1 measurement 纯观测化

- [x] R1.1 `nodes/camera_measurement/measurement.py`:删构造函数的 sequencer 参数与校验(:128-132),删采集循环内 `self.sequencer.fire()`(:174-176 一带)。帧→cycle 归属只由 `camera.arm(source_group_sizes=...)` 承诺 + `read_frame_records(exact=True)` 计数决定(现状机制保留)。
  判据:`grep -rn "sequencer" src/zlc_atom/nodes/camera_measurement/` 零命中;既有 oracle/契约测试全绿。
- [x] R1.2 `measure()` 现在一个方法按 repeat==0 返回两种类型(MeasurementResult | MonitorCapture)——拆成 `measure(repeat, frames_per_cycle) -> MeasurementResult`(repeat>=1)与 `monitor(buffer_frames) -> MonitorCapture` 两个方法,各自单一返回类型;调用方与测试同步更新。
  判据:`measure` 签名不再接受 repeat=0;monitor 行为测试(W7.1 的)迁到新方法上仍绿。
- [x] R1.3 `nodes/camera_measurement/logic_node.py` descriptor:device_requirements 只剩 camera。
  判据:R2.4 守卫对该节点绿。
- [x] R1.4 采集时序测试改写为用户手动流:测试先通过 sequencer(虚拟)load+fire 使 world 产帧,**再**调 measurement 纯收帧——fire 调用发生在 measurement 之外,测试代码结构直接证明观测/激励分离。
  判据:测试文件里 measurement 对象作用域内无 sequencer 引用。

## R2 task = 编排(默认 pulse + 设备 + 采集 + 计算 + 发布)

- [x] R2.1 **pulse 决议单源**:`nodes/_framework/pulse_source.py` 新增 `resolve_pulse(name, *, search_paths, override=None)`:override 非 None 直接用(已构建的 prog 或 (prog, meta));否则按 `pulses/<name>.py` 定位模块并调用其 `build()`;找不到/无 build = 带路径清单的清晰报错,绝不兜底。task 通过它拿 pulse,别处不许出现第二套决议逻辑。
  判据:决议逻辑 grep 全仓唯一;错误信息含搜索过的完整路径。
- [x] R2.2 **`pulses/` 目录 + 标定 pulse 定义**:仓库根建 `pulses/`,写 `calibration_reference.py` 与 `calibration_short.py` 两个真实定义(以参照树 readout/标定 pulse 为底本——铁律1"迁移不是发明"适用;树内读不到对应 pulse 则用最小两段曝光序列并在文件 docstring 声明底本缺失),虚拟 sequencer 必须能 load+fire 并驱动 world 产帧。
  判据:R2.3 全链用它们真跑。
- [x] R2.3 **CalibrationTask 重写为编排**(以参照树 `subsystems/readout` 编排链为底本):构造收 devices(camera+sequencer)+ 配置(grid_shape/method/roi_radius/reducer)+ pulse 名或 override(默认 ("calibration_reference","calibration_short"));`run()` = resolve pulse → seq.load → fire(真机语义:fire 后 measurement 纯收帧;forever 与逐 cycle 以树内底本为准)→ 用 CameraMeasurementNode 采 reference 与 short 两组帧 → `calibrate()` → 通过 signal plane 发布 calibration+report。异常路径必须 finish/safe 设备(不留 armed 残态)。
  判据:虚拟设备下 `CalibrationTask(camera=..., sequencer=..., signal_plane=...).run()` **一条调用**从 pulse 文件到 report,零手工喂帧;FakePulseStreamer/虚拟 streamer 记录到 load+fire 调用;report 数值对冻结 oracle(同输入错误率 29/360 判据继续有效)。
- [x] R2.4 **声明=实现守卫**:新契约测试遍历全部 LOGIC_NODE descriptor:对每个声明的 DeviceRequirement,用记录型 fake 构建并运行该节点,断言该设备被真实调用;并断言"未声明的设备构造参数不存在"。守卫必须自证非空洞:临时删除 CalibrationTask 里的 sequencer 使用,守卫必红(结果记入 commit message)。
- [x] R2.5 **occupancy 现状核查(小项,别扩大)**:occupancy 是 processor(接 frames 信号 + calibration 产物派生 counts/occupied/rate),不是 task,不碰 sequencer——现有职责正确。本轮只做:①R3 迁移后 import 改向;②补一条虚拟全链复验:calibration task 的 run() 产物直接喂 occupancy processor,数值对 oracle(runtime_rate_box);不重写既有数学与 M1 守卫。

## R3 physics / simulation 归 owner

- [x] R3.1 `physics/calibration.py`、`physics/bimodal.py`、`physics/psf.py` 迁入 `nodes/calibration/`(V-D:内容逐字节等价,只改 import;`physics/_readout_math.py` 4 行残根:读内容,纯转发则删);`nodes/calibration/__init__.py` 公开导出 TrapCalibration 等既有公开名。
- [x] R3.2 `simulation/world.py` 迁入 `devices/camera/`(虚拟相机的物理引擎归相机 device;sequencer 虚拟件对 world 的引用改 import 路径)。
- [x] R3.3 occupancy 改 `from zlc_atom.nodes.calibration import TrapCalibration`;新契约测试:AST/grep 扫 `nodes/` 之间的跨节点 import,白名单唯一一条 occupancy→calibration;`physics`、`simulation` 目录删除。
  判据:`test -d src/zlc_atom/physics || echo GONE` 与 simulation 同;`grep -rn "zlc_atom.physics\|zlc_atom.simulation" src tests notebooks` 零;pytest 全绿且 oracle 断言值 diff 为零。

## R4 教程与收尾

- [x] R4.1 `notebooks/usage.ipynb` 重排为两条路径并真执行提交:①手动路:用户自己 sequencer load/fire(pulses/ 定义),measurement 纯收帧;②自动路:CalibrationTask 一条调用全链 + occupancy 派生。删除与旧"measurement 自己 fire"相关的一切叙述。
- [x] R4.2 README 同步(编排分层一节:device / measurement=观测 / task=编排 / processor=派生);LOC 报告更新。

## v3 机械终态判据(全绿才把状态改回 GOAL COMPLETE)

1. 干净 venv `pytest -q` 全绿;v2 判据 1-6 依旧全绿(oracle 逐字节未动,29/360 断言在)。
2. `grep -rn "sequencer" src/zlc_atom/nodes/camera_measurement/` 零命中。
3. `src/zlc_atom/physics` 与 `src/zlc_atom/simulation` 目录不存在;全仓 grep 旧路径零命中。
4. 虚拟全链断言存在且绿:CalibrationTask 单条调用 pulse→report;其产物喂 occupancy 数值对 oracle。
5. R2.4 声明=实现守卫全绿,且突变实验(删 sequencer 使用)必红并记录在 commit message。
6. `src/zlc_atom` 包本体 grep `zlc_pulse` 零命中(pulses/ 与 tests 除外);pyproject 依赖未新增。
7. notebook 双路径带执行输出提交;勾选与 commit 一一对应;工作树干净。

---

# §v3.1 验收修正轮(2026-08-04 外部验收;仅两项,别扩大)

> v3 验收总体通过:task 真编排、观测/激励分离、守卫突变有记录、oracle 未动、notebook 双路径真跑。仅一处**规避性安置**必须纠正。

- [x] A1 **camera measurement 本体回自己的文件夹**:266 行 `nodes/_framework/camera_measurement.py` 是为了让跨节点 import 白名单测试(`edges == {("occupancy","calibration")}`)通过而挪进框架层的——CameraMeasurementNode 是具体节点不是框架件,此安置直接违背"每个 logic_node 全部内容在自己文件夹"的用户原则。修正:本体搬回 `nodes/camera_measurement/measurement.py`(删除 10 行转发壳,不留兼容 re-export);**白名单加一条真实边** `("calibration", "camera_measurement")`(编排调用下游节点是真实依赖,承认它,不伪装);calibration task 的 import 改回节点公开面。V-D 搬家纪律适用(逐字节等价,只改 import)。
  判据:`nodes/_framework/camera_measurement.py` 不存在;白名单断言 == {("occupancy","calibration"),("calibration","camera_measurement")};`pytest -q` 全绿。
- [x] A2 `src/zlc_atom.egg-info` 里 stale 的 physics/simulation 条目:重建 egg-info 或将其从版本控制/搜索面清掉,使"旧路径 grep 零"判据对构建产物也成立。

全部勾完且判据绿→状态改回 GOAL COMPLETE。

---

# §v3.2 标定 pulse 物理修正轮(2026-08-04 用户抓出;规格错误在 GOAL v3 R2.2 本身,不是实现走样)

> **物理事实(顶层裁决,以参照树为准,永不再违背)**:标定 bracket = 同一次 MOT 装载/冷却后的**三个相机窗口:long → short → long**(参照树 `logic_nodes/readout/calibration/task.py:138` 原文 "Imaging pulse for each long-short-long bracket")。前后两张 long 帧构成**共识标签对**——两帧一致才证明原子在整个 bracket 期间稳定存在,该 shot 该位点的 short 样本才有可信标签;short 夹在中间做阈值表征。这正是数学层已有的输入契约:`calibrate(reference_frames=(groups, 2, y, x), short_frames=(groups, y, x))`,oracle 帧形状 60×2×34×40 里的 2 就是每 bracket 的两张 long。v3 R2.2 的"两个独立 pulse"与本轮初稿的"两窗口 bracket"规格**都作废**——两次独立装载占据互不相关,两窗缺第二张 long 则共识机制整体失效、标签被污染。

- [x] P1 **单一标定 pulse**:删 `pulses/calibration_reference.py` 与 `calibration_short.py`,新建 `pulses/calibration.py`——单个序列 = MOT cooling 事件 → **long 窗 → short 窗 → long 窗**;以参照树 long-short-long bracket 为底本迁移(铁律1;JSON 配方无 loader 则 Python 迁移其完整三窗 bracket 结构,不许再删窗)。metadata 必须声明:`camera_windows=3`、逐窗曝光 `frame_exposures=(long_s, short_s, long_s)`(两 long 等曝光)、帧序语义(cycle 内 frame 0/2=reference 共识对,frame 1=short)。
- [x] P2 **task 单 pulse 单采集**:`CalibrationTask` 默认 pulse 名收敛为单个 `"calibration"`(pulse_names/pulse_overrides 二元组形态删除);`reference_repeats/short_repeats` 合并为一个 `repeats`(一个 bracket 同时产三帧);采集 = 一次 resolve+load → fire×repeats → `frames_per_cycle=camera_windows(3)` → 每 cycle 按帧序拆 `reference[(f0, f2)]`(形状 (repeats, 2, y, x))与 `short[f1]`((repeats, y, x))→ 直接喂现有 `calibrate()` 签名,数学零改动;short 曝光从 metadata `frame_exposures[1]` 取。
- [x] P3 **虚拟物理一致性**:虚拟链 fire 一次按窗口产 3 帧,**同 cycle 三帧必须取自同一原子占据状态**(同一次装载),信号按各窗曝光缩放(rate×exposure,W5.1 派生纪律);跨 cycle 占据可重新采样。守卫测试:同 cycle 三帧占据模式逐位点一致、两 long 帧信号统计等价、short/long 信号比≈曝光比;突变探针(帧间各自重采占据)必红,记 commit message。
- [x] P4 **收尾**:notebook 两条路径同步(手动路也用单 bracket pulse);oracle 端到端 29/360 判据保持(calibrate 数学不动);`grep -rn "calibration_reference\|calibration_short" src tests pulses notebooks` 零(oracle fixture 键名除外,如有则列明)。

### v3.2 机械终态判据
1. `pytest -q` 全绿;29/360 断言在;P3 守卫+突变记录在 commit message。
2. `pulses/` 只有 `calibration.py` 一个标定定义(camera_windows=3);P4 grep 零。
3. task 单条 `run()` 全链依旧零手工喂帧;声明=实现守卫依旧绿。

收尾证据: commits `1efe432`(P1/P2) 与 `5a63d5e`(P3);P3 突变实验使 short 帧标签准确率降至 5.6% 并失败。notebook 已执行，输出包含 `reference_frames=60`、`short_frames=30`，oracle 为 `29` 错误 / `360` 总样本；旧 pulse 名 grep 为零。
