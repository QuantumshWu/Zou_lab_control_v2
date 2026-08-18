# Zou Lab Control v2 — Approved Target Architecture

状态：`TARGET / NOT YET IMPLEMENTED`，除非`IMPLEMENTATION_PLAN.md`的当前Checkpoint明确标为完成。

本文只定义用户批准的产品不变量，不保存历史commit、旧测试数字或阶段日志。当前实现状态、验证证据和下一步只看`IMPLEMENTATION_PLAN.md`。完整实施范围与阶段顺序见根目录`ZLC_V2_IMPLEMENTATION_GOAL.md`。

## 1. Authority与原则

实施authority顺序：

1. 用户最新明确指令；
2. `ZLC_V2_IMPLEMENTATION_GOAL.md`；
3. 本文目标不变量；
4. `IMPLEMENTATION_PLAN.md`当前Checkpoint；
5. 当前代码与实验事实。

Package README、旧GOAL、survey、acceptance、历史contract和旧tests不是目标规格；它们在Milestone 1中删除或按当前产品重写。

总体原则：

- 保留八层骨架，删除平行truth和单消费者framework；
- 默认删，不为旧测试、兼容或“以后可能”保留历史路径；
- 每个事实只有一个owner；
- 优先扩展现有Data、Plane、Host、Session和device骨架，不新增manager/registry/base-class；
- 不用GPU、降采样、质量放宽、丢revision或增加timeout掩盖性能根因；
- 不增加密码、认证、TLS、权限系统或新的content hash体系；
- Domain validation、hardware acknowledgement、owner identity和strict format是功能正确性，不是防御性框架。

## 2. 八层职责

| Layer | 唯一职责 | 禁止 |
|---|---|---|
| `zlc_data` | Immutable scientific schema、values、validity、selection projection和codec grammar | Runtime、Qt、device、workspace路径 |
| `zlc_durable` | Atomic write、并发安全命名、workspace path | Science schema与figure语义 |
| `zlc_runtime` | Node lifecycle、canonical run accumulation、live/partial/final publication、causal identity、front scheduling | Plugin physics、plot rendering、Qt |
| `zlc_plot` | Snapshot projection、exact fit、overlay、selector、raster front | Signal registry、Task lifecycle、plugin science |
| `zlc_ui` | Qt views和plain view models | Plot/Runtime/device/domain ownership与blocking work |
| `zlc_pulse` | Pulse model、compile、wire、transport和execution evidence | Measurement shot policy与Workbench state |
| `zlc_atom` | Device plugins、science nodes、Calibration、SLM/atom physics | Workbench composition与panel-save truth |
| `zlc_workbench` | Composition、workspace/session、device claims、panel/layout persistence | Plugin science与第二Runtime/Plot实现 |

最终作为一个ZLC distribution安装；八层是代码依赖边界，不是八个standalone wheels。

## 3. Data与Durability

### 3.1 Scientific data

- `OwnedSnapshot`是外部不可变数据面；schema、coordinates、labels、units和validity共同定义truth。
- Snapshot restriction必须对values、validity、coordinates、labels和coordinate frame执行同一projection。
- Validity入口只接受明确bool contract，不做numeric truthiness转换。
- Selection按AxisId和typed coordinate唯一解析；重名或不可唯一映射必须拒绝。
- 同一run/content revision不可代表不同内容；EventRef只表达causal publication，不代替content identity。

### 3.2 Figure archive

- 一个writer、一个reader、一个format owner。
- Writer写入前规划全部member namespace并拒绝碰撞。
- Reader在解释内容前严格验证format/version、required members、shape、duplicates和non-finite metadata。
- 未知metadata类型拒绝，不自动字符串化。

### 3.3 Durable paths

- Unique name allocation与commit构成一个并发原子操作，多process不得取得同一目标。
- Atomic replace失败后的outcome必须诚实，不把可能已写入伪装成旧状态。
- 不新增content hash；使用run identity、受控path、shape/size和完成状态记录artifact集合。

## 4. Canonical Runtime Live Contract

### 4.1 One run-data owner

Logic Node只提交本次新增chunk/event；Runtime按run和signal identity累计唯一canonical dataset：

```text
Node new chunk
  -> Runtime append/commit
  -> immutable live view
  -> retained partial seal
  -> final seal
  -> Panel / Processor projections
```

- Camera、Scan、Calibration和Task preview不得自建parallel slot/history/terminal truth。
- Camera使用chunked append，避免每次复制全部历史；Scan按固定point geometry增长。
- 未写位置invalid；coverage只描述实际写入extent。
- UI freeze只读取已提交状态，不调用plugin materializer。
- Stop/Final不受Panel、freeze或Processor订阅影响。

### 4.2 Identity与processors

- Generation标识一次run/restart；generation内schema和stream generation固定。
- Revision严格递增，不接受重复、倒退或同ref不同内容。
- 一次commit的siblings共享revision、run record和causal parent。
- Exact scientific Processor逐event处理；pure display derivation可latest。策略由input contract声明，不从coverage猜。
- 不同Processor可并发，同一Processor保持有序。

### 4.3 Logic Node contract

- Measurement必须在bounded cadence内live commit。
- Task必须发布progress与声明preview，或显式声明无preview。
- 第一份真实publication前不显示live；terminal清除progress并seal/retire preview。
- Descriptor outputs、runtime declarations和preview references只有一份typed vocabulary。
- 通用discovery test必须走真实NodeHost、SignalDataPlane和preview contract。

### 4.4 Task运行中冻结

冻结：Add Logic Node、当前Node source/preview signal、overlay binding、scope/reduction/fate以及冲突硬件配置。

允许：其它Panel、当前Panel的样式/viewport等纯显示参数。Calibration继续显示long/readout/long三帧Grid/Figure。

## 5. Plot、Fit、Overlay与Selector

### 5.1 Exact Data/Fit pairing

- Fit armed后每个source revision都有一个ordered exact fit job；不丢revision、不只fit latest。
- Panel只原子呈现`data@N + fit@N`。
- FitResult携带source parent/revision，Rolling trace逐revision连续。
- Fit计算在后台worker；Qt owner thread不等待Future或执行fit。
- Backlog超过经profiling确定的支持预算时loud error，不跳revision；raw data仍可离线重算。

### 5.2 Performance与state

- PanelState一次应用是幂等transaction；no-op产生0 solve、0 render、0 front。
- Title/layout等非plot变化不得re-fit。
- 删除重复configure/clear/replay与多front handoff。
- 性能以真实TaskConsole、1/4/8 panels、fit+overlay、Setting/Edit和Qt owner latency为profile对象。

### 5.3 Overlay与selector

- Overlay是plugin发布的typed companion signal，不由Workbench重建science。
- Data、Fit和Overlay共同使用同一个scope/axis/fate projection；无法唯一对齐则拒绝。
- ROI/binning坐标只由一个transform owner处理。
- Selector Off时普通滚轮滚外层board；On时plot接管交互。

## 6. UI与Lifecycle

- `zlc_ui`不拥有domain parser、device state或plot lifecycle。
- Qt slot不得执行blocking I/O、device tune或`Future.result()`。
- Window只有在owned command、worker、executor和claim安全退出后才能消失。
- Pulse Stop UI立即进入Stopping；Stop/SAFE高优先级并可取消普通wait/transport，hardware ack后台完成。
- Timeout显示真实错误但不冻结UI；未确认前不能显示Safe。
- Form reconcile必须按当前schema重建dependency graph。
- PanelState decoder、owner wake和产品Figure save各只有一个实现。

## 7. Pulse、Camera、Remote与FPGA

### 7.1 Execution vocabulary

- `RepeatRegion`只表达timeline内部loop。
- Cycle/shot、scan sweep和Dataset repeat是独立事实。
- 一个finite execution入口表达N cycles；actual played values进入Dataset coordinates和run record。

### 7.2 Camera

- Same-shot保证采用continuous best-effort，不新增hardware marker或逐cycle arm/fire。
- 每个run核compiled trigger windows、frames-per-cycle和received ordinal；gap/cardinality错误立即失败。
- Temperature保留约20ms exposure并增加足够trigger/recapture gap，使用相容Calibration。
- Virtual按真实cadence逐cycle、支持Stop并模拟camera busy。

### 7.3 Remote

- 无密码、认证、TLS或权限UI。
- Second client默认last-client-wins；旧handler立即失效，takeover前旧active command必须成功Stop/SAFE。
- 正常连接无idle timeout；控制进程/socket/连接真正断开时自动SAFE。
- UART port显式配置，不扫描所有COM。
- 只有server process持hardware transport；不保留假的进程内Interprocess lease。

### 7.4 Host/RTL/build invariants

- Load前核target ABI、clock、geometry、counts、camera cadence和delay FIFO capacity。
- Count必须是合法hardware range内整数，不clamp/wrap。
- Hardware SAFE独立gate TTL/DAC data/clock；LOAD/FIRE前pins保持safe。
- Public DONE等待delay FIFOs和final DAC latch完成并进入安全态。
- Underflow、overflow和protocol error sticky且loud；scan point0必须resident。
- 50MHz engine有真实clock/STA constraints。
- Explicit board manifest统一生成host lanes、top mapping和XDC，不靠XDC行序。
- Build delete做真实path containment；program/flash exactly-one target fail closed，默认不自动flash。
- RTL tests自动compile/run并以nonzero failure/逐tickreference证明。

## 8. SLM

### 8.1 USB-only device

- 正式transport只有Hamamatsu USB SDK；DVI production/discovery/UI/tests/docs删除。
- Initial command state是unknown，只有成功write/display/readback/settle后才known。
- Side effect失败区分known-old、known-new和unknown outcome。
- Correction mutation取得同一DeviceUse claim并冻结mapping revision。
- Profile记录model、serial、wavelength、phase curve来源和settle语义；不新增hash。
- Editor明确区分authoring draft与device command；external Task后旧Send不得静默覆盖。

### 8.2 Context与artifacts

- Target保存intensity和objective。
- Science Context保存numeric pupil、Pattern/base、operator wavefront和system correction引用。
- Command receipt保存USB/profile/wavelength/orientation/correction/outcome。
- `SystemCorrectionArtifact`明确区分pupil phase map与target response map；不得把per-geometry site weights冒充通用wavefront correction。

### 8.3 Solver与Feedback

- 保留sparse WGS-Kim、fixed far-field phase、selected DFT和caller-owned optimizer state。
- Inner solve走到canonical numerical gate，不为省几十毫秒增加physical candidate。
- Current Feedback observable是all-shot fluorescence，不宣称trap depth。
- SLM Feedback复用canonical Camera Measurement `repeat=N`及同一Runtime dataset/projection；不另写camera average。
- Controller使用uncertainty、step clip、trust、rollback和invalid stop。
- 100 shots是coarse；final validation根据实测variance自适应并有最大time/shots，输出estimate、uncertainty或inconclusive。
- Stop接受并apply本run confidence-best phase；异常failure只在incoming known时恢复。
- Sparse-only contract明确；dense Gaussian/Flat Top先修算法定义和early stop，再profile CPU，不引GPU。

## 9. Calibration、Scan与Simulation

- 不重设计Calibration对外流程、主要artifact、默认raw policy或三帧report。
- 允许不改变外部行为的dependency解耦、明确corruption修复和内存优化。
- Scan正常完成、Stop或失败都默认restore pre-run device values。
- SimulationWorld保持一个类和一个state owner，不拆层。
- Simulation参数在init前通过单一API/immutable config确定；可读取workspace-local profile，Device Manager Init不运行时改写。
- Tests使用config override，不修改public mutable world attributes；hidden truth不泄漏给production算法。

## 10. Deployment、Evidence与Docs

- 一个可安装ZLC distribution，内部八层不独立发wheel。
- 一份product manifest、一份dependency lock、一组正式entrypoints。
- 正式evidence lanes：software、gui_offscreen、virtual_vertical、notebook_offline、real_screen和hardware runbooks。
- Mock/virtual/offscreen证据不得冒充真hardware/optical acceptance。
- Root Architecture只保存目标不变量；Implementation Plan只保存当前Checkpoint、milestone状态和最新证据。
- 旧package GOAL、survey、acceptance和历史contracts删除或重写；不在活文档尾部追加修补记录。

## 11. 当前实现状态

本文描述批准目标，不代表当前HEAD已实现。准确状态、已完成commit、测试和下一步见`IMPLEMENTATION_PLAN.md`的持久Checkpoint。
