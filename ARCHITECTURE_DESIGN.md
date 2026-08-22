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
- Workbench不得import或分支判断`zlc_atom.nodes.<concrete_leaf>`；它只消费discovered descriptor、Runtime signal与Data/Plot等中立层拥有的通用contract。新增/删除普通Logic Node的修改必须闭合在该leaf目录、资源与测试内；只有新增真正跨节点能力时，才先在中立层定义contract。
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
  -> immutable event view -> exact scientific Processor
  -> canonical current view -> Signal description / Panel / Edit / Save /
                               Selector / Overlay / display derivation
  -> retained partial seal
  -> final seal
```

- Camera、Scan、Calibration和Task preview不得自建parallel slot/history/terminal truth。
- Camera使用chunked append，避免每次复制全部历史；Scan按固定point geometry增长。
- 未写位置invalid；coverage只描述实际写入extent。
- Finite exact signal的event view只用于commit与exact Processor；所有UI/display consumer必须使用同一publication对应的canonical current view，从第一次publication起报告完整authored physical shape，未来位置invalid。
- 普通Monitor signal没有finite canonical extent，UI显示latest complete event；Processor不得仅因“derived”就增加科学轴。输出契约的`index_by_source`只声明history能力；只有真实consumer按window取得lease后，Runtime才从当时的current event开始建立带通用`primary-index`的bounded ordinary Dataset，并按全部active leases的最大window保留、在最后一个lease释放时立即归零。lease之前的event不回填、不伪造；所有Plot读取同一Dataset且不建立Plot-kind、Fit或Workbench专用history lane。Occupancy exact处理每个camera cycle，但其公开Monitor几何仍是当前cycle的`frame`，不得再叠一层source index。
- `scope/reduction/fate`只决定怎样投影canonical view，绝不决定选择event还是canonical；同一publication不能因Panel semantic不同代表两份不同数据truth。
- Incremental placement沿repeat与point rows；多维scan/grid通过point table与grid topology表达。一个cell payload原子完整发布，不新增cell-internal tile/slice streaming contract。
- Canonical display materialization只在实际display consumer到期时合并/cache，并在Qt owner thread之外执行；不得让producer每commit强制复制full prefix，也不得因Panel存在与否改变采集结果。
- Live Panel、Panel Edit/Refresh/Save、selector、fit input和overlay必须从同一accepted canonical presentation snapshot投影；无法唯一对齐即拒绝。
- Occupancy的SITE是每个`(repeat, point)` cell内原子完整的data axis；overlay不得另存site history。Occupancy只发布通用bool/numeric status signal，点是否可读由该Dataset自身validity表达；XY geometry与adapter contract由`zlc_plot`中立层拥有，Workbench只按contract路由signal且不得import Occupancy。只有scope/facet唯一选中一个cell时才能显示离散状态；对多个cells做reduce/pool时不得私自发明共识状态，未定义则显示UNKNOWN/隐藏。
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
- Concrete Logic Node不得要求Workbench识别其模块、output spelling或domain helper；通用显示/overlay/selection能力由中立层contract表达，Workbench只路由contract。
- 通用discovery test必须走真实NodeHost、SignalDataPlane和preview contract。

### 4.4 Task运行中冻结

冻结：Add Logic Node、当前Node source/preview signal、overlay binding、scope/reduction/fate以及冲突硬件配置。

允许：其它Panel、当前Panel的样式/viewport等纯显示参数。Calibration继续显示long/readout/long三帧Grid/Figure。

## 5. Plot、Fit、Overlay与Selector

### 5.1 Exact Data/Fit pairing

- `display_interval`只控制Surface刷新deadline，不决定active history lease内的Measurement primary index是否存在。Runtime只在lease起点之后为indexed-derived Dataset写value或invalid；昂贵Surface计算同一same-shot group只允许一个active，并在忙时只保留Plane latest完整输入，中间indices仍以invalid存在而不排完整frame。
- Panel只原子呈现`data@N + fit@N`。
- Fit selection唯一优先级是committed Area ROI（或显式X-range）→viewport→full range；FacetGrid selector必须保留所属focused cell identity，任何PanelState重放不得把ROI降级成viewport/full。
- FitResult携带source parent/revision；任何history/window投影按Measurement primary index连续，未计算、失败或timeout的位置invalid/NaN，window长度按source indices而非成功结果计数。
- Fit计算在后台worker；Qt owner thread不等待Future或执行fit。
- Active Fit超过1秒必须loud标记该source index invalid并从Plane latest继续；不得积累完整frame FIFO，也不得永久锁住Panel、Qt、Stop或close。普通cadence/backpressure跳过计算的indices同样invalid但不是solver failure；raw Runtime data始终完整。

### 5.2 Performance与state

- PanelState一次应用是幂等transaction；no-op产生0 solve、0 render、0 front。
- Title/layout等非plot变化不得re-fit。
- 删除重复configure/clear/replay与多front handoff。
- Histogram只有`bins`变更需要一次完整sample projection；`density`/`cumulative`只是已接受bins的representation，不得再扫描full payload。复用已settle tick unit时必须在枚举lattice前先核上界，不得因range大幅变化卡住UI。
- 正式96×128 Camera、小Area ROI、主图atomic fit、并行ROI image与一个fit-parameter Rolling Panel链路以100 ms作为profile警戒线；明显的额外cadence、HOL、错误串行和重复render必须删除。若剩余是必要fit/raster/Qt成本，只有能带来实质收益且不增加不相称复杂度的优化才实施。
- 性能以真实TaskConsole、1/4/8 panels、fit+overlay、Setting/Edit和Qt owner latency为profile对象。

### 5.3 Overlay与selector

- Overlay producer发布匹配中立Plot contract的numeric/bool companion signal，并在同一run record中携带该contract要求的geometry document；`zlc_plot`拥有通用adapter与renderer，Workbench只按contract路由，不import domain plugin，也不重建science。
- Data、Fit和Overlay共同使用同一个scope/axis/fate projection；无法唯一对齐则拒绝。
- ROI/binning坐标只由一个transform owner处理。
- Selector Off时plot不消费任何pointer gesture：不画selector、不zoom/pan，也不响应双击facet focus；普通滚轮继续滚外层board。
- Selector On时，FacetGrid overview只响应双击进入cell，不得在overview开始area selector；进入具体cell后，selector才按该cell的canonical projection工作。

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
- UART auto枚举COM、优先USB VID/PID，并只在word-63 fingerprint匹配后选用；
  显式port把探测限制为该端口。auto探测失败才回退JTAG，显式UART失败则报错。
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

### 8.1 Server-owned device

- server默认使用原本可用的DVI exact-raster presenter，不依赖vendor DLL；USB frame memory仅在显式`--transport usb`时使用。
- 和Pulse一样，真实SLM只有一个apparatus device type，其init参数是server host/port；只有server process持有DVI/USB输出与profile/correction。客户端通过bounded length-prefix、strict-JSON metadata和canonical `float32`相位payload做握手与command代理，不形成第二个hardware owner。普通state读取使用握手cache；apply携带expected command/mapping revision并拒绝stale writer，不确定transport outcome后必须采样真实hardware state再继续。
- SLM proxy无authentication/TLS，只能部署在trusted laboratory LAN，不得暴露到public Internet。
- Initial command state是unknown，只有成功write/display/readback/settle后才known。
- Side effect失败区分known-old、known-new和unknown outcome。
- Correction mutation取得同一DeviceUse claim并冻结mapping revision。
- Profile记录model、serial、wavelength、phase curve来源和settle语义；不新增hash。
- Editor明确区分authoring draft与device command；external Task后旧Send不得静默覆盖。

### 8.2 Context与artifacts

- Target v2保存intensity和objective，只是Editor authoring import/export artifact，不是run consumer的第二Target truth。
- Science Context v2保存run的唯一strict frozen Target、numeric pupil、Pattern/base、operator wavefront和system correction引用；Editor的v2 Load是一次atomic adopt。v1只允许显式migration为`target=None`的authoring input，Feedback必须拒绝。Candidate Context保存实际evolving Target，可直接作为下一run的唯一Context input。
- Command receipt保存USB/profile/wavelength/orientation/correction/outcome。
- `SystemCorrectionArtifact`明确区分pupil phase map与target response map；不得把per-geometry site weights冒充通用wavefront correction。

### 8.3 Solver与Feedback

- 保留sparse WGS-Kim、fixed far-field phase、selected DFT和caller-owned optimizer state。
- Inner solve走到canonical numerical gate，不为省几十毫秒增加physical candidate。
- Feedback mode是leaf-owned显式字段；当前唯一mode为`qcmos_bright_dark`。Pulse由operator显式选择，camera exposure由operator独立显式填写；Task不从Pulse或Calibration猜exposure，也不做Pulse/exposure科学“兼容性”判断。
- 当前mode复用canonical Camera Measurement `repeat=N`，每cycle严格一张camera frame；preview与估计读取同一sealed Dataset，不另写camera average或三帧reference判据。
- Calibration只提供Target→camera注册所需的site centers、BOX半宽/积分方式和frame坐标几何；Feedback不读取其dark/bright/threshold、exposure、photoelectron mode、camera identity或readout working-point provenance。实际camera requested/actual exposure、effective unit与conversion进入本run metadata；saturation只由本次actual raw integer maximum转换到本次effective unit判断。
- 每个site使用本candidate全部shot的raw BOX值拟合双高斯；observable是`bright_mean-dark_mean`，loading probability只作为mixture fraction。BIC、峰分离与simultaneous mean error共同决定valid/uncertain/censored；不把单峰硬拆成有效bright response。
- Controller保存并落盘每site的实际Target weight、bright-dark、误差、fit状态、动作与局部`d log C / d log weight`。有效site按实验已确认的单调方向更新：bright-dark偏大表示trap偏浅，因此增加Target；可信历史斜率优先，斜率不可辨识时使用固定`0.25`基准增益，单步最多1.5倍并保持总Target power。
- 已有双峰但fit error过大的site本轮hold。此前从未有效的单峰site按浅阱先验最多三次、每次1.4倍bootstrap；曾经有效且在增加weight后bright峰消失的site回到其上一有效weight，其余不可辨识site hold。mode、Pulse或exposure变化时不复用旧响应历史。
- 默认每candidate为500 shots、最多12次update；只有全部site valid时才以bright-dark ratio `<= 1.10`结束coarse。Normal terminal与Stop保留最后一个全site-valid controller state；无valid state保存Context-start candidate 0。Final validation默认最多3000 shots/300秒，对独立single-frame shots重新双高斯拟合，并按sites × maximum looks做simultaneous family correction，输出estimate、uncertainty或inconclusive。
- Feedback取得SLM后自己apply并确认frozen Science Context phase，再测baseline、更新Target、solve并继续camera闭环；Context receipt是provenance，不要求operator事先Send/Save或保持某条旧command。无valid candidate的normal terminal或Stop以该Context起始phase的candidate 0落盘，measurement为None；有valid state时保留并apply最后一轮，异常failure恢复该Context起始phase。
- Sparse-only contract明确；dense Gaussian/Flat Top先修算法定义和early stop，再profile CPU，不引GPU。

## 9. Calibration、Scan与Simulation

- 不重设计Calibration对外流程、主要artifact、默认raw policy或三帧report。
- 允许不改变外部行为的dependency解耦、明确corruption修复和内存优化。
- Calibration只产生与SLM无关的camera/readout artifact，UI和Task都不接受Science Context。SLM Feedback在同时拿到Calibration与Context后做Target X/Y→camera X/Y直接正向注册，并为未观测site生成predicted BOX；不枚举翻转、旋转或轴交换。
- BOX model仍为Calibration/Occupancy持久化自己的readout事实；Feedback只取BOX geometry。未观测Target site由注册产生predicted BOX，并与实测site一起接受本次run的双高斯估计，不伪造Calibration dark/bright样本。
- Scan正常完成、Stop或失败都默认restore pre-run device values。
- SimulationWorld保持一个类和一个state owner，不拆层。
- SimulationWorld的物理site只有当前SLM phase经共同pupil illumination、共同low-order wavefront aberration和FFT得到的dominant local peaks这一份动态roster；trap位置、强度、occupancy与Camera位置不得再拆成nominal/extra双状态。所有peaks经过同一个Fourier→camera affine；fluorescence imaging使用一个由共同imaging pupil/aberration生成的shared非对称PSF，不存在逐site随机gain/ellipse/angle/skew。Probe为红失谐，正的trap light-shift参数只把detuning进一步推红，因此occupied bright-dark随trap depth单调下降；loading probability随depth上升。Camera shot真实混合dark/bright population，Feedback不得读取hidden depth/occupancy truth。
- Apparatus root `simulation`是image/grid geometry、seed与profile的唯一持久化owner；virtual qCMOS只声明camera事实并消费world image geometry，virtual MOT保持独立的camera geometry。旧的camera-owned world字段不保留双owner或静默migration，必须loud refusal并给出root grammar。
- Simulation参数在init前通过单一API/immutable config确定；workspace-relative profile必须在任何device factory前解析且保持在workspace内，Device Manager Init不运行时改写。
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
