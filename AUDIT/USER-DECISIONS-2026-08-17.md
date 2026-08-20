# 用户架构裁决记录 — 2026-08-17

状态：已裁决；本文覆盖此前审计报告中的相反推荐。只记录产品行为和实施原则，不修改代码。

## 1. 明确错误的处理授权

用户同意修复`DECISIONS-USER-GUIDE.md`第三节列出的明确错误，包括：

1. Figure archive内部成员碰撞、Reader不严格验证format；
2. 未采future数据被当作有效事实、UI刷新影响Stop terminal；
3. duplicate device、window/worker/device claim关闭顺序；
4. FPGA SAFE/DONE/build/program边界；
5. remote owner正确性由实现者收口；安全认证不是产品目标，不加入密码流程；
6. real SLM unknown initial phase不再虚构zero。

额外不可妥协要求：Pulse Stop在任意时刻都必须立即响应，不得阻塞Qt界面或让用户等待一个不可取消的worker/transport调用。

产品语义：点击Stop后UI立即进入`Stopping`，高优先级取消/SAFE命令不排在普通工作后面；真正hardware ack可异步完成。若transport无响应，bounded timeout后显示明确错误，但UI仍可操作。不得在收到物理确认前伪称`Stopped/Safe`。

## 2. Gate 1 — 部署

裁决：采用方案A。

- 长期只有一个可安装ZLC产品distribution；
- 代码继续保持八层职责和依赖边界；
- 不再承诺八个standalone wheels；
- 一份dependency lock、一份product manifest、一组正式entrypoints；
- fresh install/CI/实验机receipt都指向同一产品版本；
- 旧source-checkout bootstrap、独立package metadata和重复launchers在迁移完成后删除，不保兼容层。

## 3. Gate 2 — Runtime统一live数据owner

### 用户需求

1. 每个Logic Node只负责提交当前新产生的数据，不在plugin内部维护另一份完整history/live slot真相；
2. Panel在任意时刻仍能访问截至当前的全部数据，并应用reduction、scope、某个axis point和其它semantic fate；
3. Camera、Scan、Calibration不得继续各写一套live实现；
4. 全系统只有一份可seal、可保存、可由Processor消费的run dataset truth。

### 目标设计

Runtime成为唯一累计owner：

```text
Logic Node提交新chunk/event
  -> Runtime按signal/run identity追加到canonical run dataset
  -> 同一owner生成live snapshot、partial seal和final seal
  -> Panel/Processor从这一份truth选择scope/reduction/fate
```

Node不发布“自己保存的全部历史”，只提交新增commit；Runtime内部可以使用chunked/append storage避免每次复制历史。对外仍提供immutable `OwnedSnapshot`/projection，不向Panel暴露plugin buffer。

- Camera raw frames按chunk增量存储；latest cycle和running reductions可以live投影，但完整截至当前的raw run仍由Runtime owner可访问/partial seal；
- Scan按固定point geometry逐步填充，future positions invalid；
- exact scientific Processor逐event增量处理，latest-only display derivation可以合并旧更新；
- Stop/Final与UI freeze无关；同一run只存在一条live→partial/final materialization路径。

这是替换现有多套slot/builder，不是再增加第五套抽象。优先复用`SignalDataPlane`、`OwnedSnapshot`和现有Host骨架，只增加它们缺失的单一append/commit责任。

## 4. Gate 3 — Data/Fit/Overlay/Selector

### Data与Fit

用户裁决：Data与Fit必须属于同一个source revision，并作为一个完整panel结果原子显示。不得先显示Data、稍后补Fit。2026-08-20用户进一步裁决连续源的过载恢复：正常负载保持逐revision exact；显示落后超过约1秒时必须报错但不得永久卡住流程，应丢弃尚未fit的中间revision并从当时latest继续。

原因：

- 图上的fit必须和当前data严格对应；
- 正常负载下Rolling trace消费每个data revision；过载resync产生的断点必须伴随明确错误，不能冒充连续结果；
- 若fit速度跟不上，必须明确报错并恢复到latest，不能静默skip、永久latch或显示不对应结果。

### 目标实现

- `display_interval`是Panel data+fit admission cadence；fit armed后，在1秒支持预算内每个已admit source revision形成一个exact fit job。区间内更高频且从未进入Panel的raw Monitor publication不是Rolling sample，也不伪造FitResult；
- jobs正常按顺序处理；只有最老pending超过1秒或pending arrays超过64 MiB时执行一次loud resync，取消未fit中间jobs并保留当时latest；
- Panel只有在`data@N + fit@N`都完成时才present revision N；
- Fit样本范围按committed Area ROI（或显式X-range）→viewport→full range；FacetGrid重建selector必须保留focused cell identity，不能让ROI derived正确而fit退回full frame；
- FitResult成为带source revision/parent identity的正式derived result，Rolling trace消费同一序列；
- acquisition data由Runtime继续可靠保存，不因为Panel fit慢而丢science data；
- 若fit backlog超过明确的容量/延迟预算，fit pipeline报告“无法跟上”并从latest继续；被丢弃revision不产生成功FitResult，而是发布带相同source identity的invalid gap，不显示unpaired data，不阻塞Qt/Stop/clear/close；保存的数据可离线重算；
- worker计算可在后台线程执行，“原子显示”不等于在Qt线程同步计算；Qt不得卡死。

性能优化必须先profile实际fit/model/render链，删除无关Setting触发的重复solve、重复render、多front handoff和无owner线程池。1秒resync是明确的连续源过载恢复，不得被滥用为正常负载latest-only策略。

2026-08-20追加验收：正式96×128 Camera、小Area ROI、fit直接运行在Camera主图、同一ROI另发布Image、fit参数进入一个Rolling Panel的完整链路，以100 ms作为profile警戒线。明显的额外cadence、HOL、错误串行或重复绘制必须修；若已无明显错误，不为百分之几或十几的边际收益引入大改或复杂调度。1秒只允许作为异常resync阈值，不能用来掩盖正常问题。

### Overlay

用户需求：Overlay与Data必须经过同一scope/axis/fate选择。Panel若选择frame 2、repeat 7或某个facet，Overlay必须显示那个相同语义位置的状态。

目标设计：

- Overlay是typed companion signal，带与Data可对齐的axis/site/coordinate identity和同源parent revision；
- projection一次解析Panel的scope/reduction/fate，再共同应用到Data、Fit和Overlay；
- 无法唯一对齐时明确拒绝，不显示“看起来像对上”的latest overlay；
- Workbench不重建Occupancy science，plugin发布完整overlay truth。

### Selector

裁决：采用A。

- Selector Off：普通滚轮交给外层TaskConsole/board滚动；
- Selector On：非Grid surface与已focused Grid cell由plot接管selector/viewport交互；FacetGrid overview只接受双击focus，不接受area、pan、zoom或滚轮。

## 5. Gate 4 — Logic Node contract与Task冻结范围

### 框架责任

用户要求把live、preview、progress、terminal等基本行为写入Logic Node的基础契约/Host验证，不再依赖每个实例作者记得实现。

目标设计：

- Descriptor/declaration构造时验证outputs与previews一致；
- Measurement在bounded cadence内必须live commit；
- Task必须提供progress与声明的preview，或者显式声明无preview；
- 第一份真实数据前不假报live；
- terminal清除progress并seal/retire preview；
- related data/fit/overlay同revision family由Runtime一起冻结；
- 通用contract test对每个discovered Logic Node走真实Host/Plane/preview链。

优先使用现有descriptor/NodeHost骨架，不新增通用BaseNode继承体系，除非现有composition无法表达强制contract。

### Task运行时冻结

用户裁决：

- 禁止Add新的Logic Node；
- 冻结正在运行Node的source/preview signal绑定及与该signal相关的overlay/derived binding；
- 禁止改变会影响当前Task硬件/数据身份的配置；
- 允许用户操作其它Panel；
- 允许当前Panel的纯显示参数，例如样式、viewport、非identity plot参数；
- 不允许通过“纯显示设置”偷偷改变source scope/reduction/fate；这些属于运行结果identity，运行期间冻结。

### Calibration preview

保持当前产品行为：Grid/Figure panel显示long/readout/long三个frame，不改。

### SLM Feedback中的Camera Measurement

此前“Task自己发布camera average”的设计判断被用户否决。

SLM Feedback必须调用/复用canonical Camera Measurement `repeat=N`采集和它的同一Runtime dataset：

- Camera Measurement中间过程按统一live contract发布；
- Preview Panel通过同一个projection系统选择Feedback所需readout event、repeat reduction、spatial/site fate；
- Feedback estimator消费Camera Measurement sealed result的同一投影，不另写camera accumulator/average算法；
- Preview显示的数值与Feedback真正用于更新phase的数值必须来自同一个projection request和同一dataset revision。

## 6. Gate 5 — Pulse/Camera/Remote

### Same-shot

裁决：采用第一档continuous best-effort，不要求hardware marker或逐cycle arm/fire证明。

实现仍需：

- 检查收到frame ordinal连续、compiled window count与frames-per-cycle一致；
- 一旦检测到gap/cardinality错误就loud fail，不能继续错位分组；
- artifact诚实记录保证等级为best-effort，不宣称absolute physical shot identity。

### Temperature

裁决：采用A。保留约20ms exposure，增加足够的trigger/recapture gap；使用与该working point相容的Calibration。不得用VirtualCamera宽松行为代替真机cadence preflight。

### Remote

用户不要求认证、密码或安全框架。

- 不增加登录、密码、token输入UI、TLS/权限系统；
- 保持last-client-wins：第二个client连接时默认抢占旧client；
- 抢占前旧active command必须成功SAFE或被硬件Stop；失败则明确报告takeover失败，不能同时存在两个有效owner；
- owner epoch/session ID只用于内部正确性，用户无需输入；旧handler的后续command必须失效；
- 正式UART port显式配置，避免向其它COM仪器发送probe，这属于设备正确性而非安全功能。

“失联”只指控制程序崩溃、TCP socket关闭或网线/网络中断，不是用户一段时间没有操作。正常连接可以无限期接受命令，不设idle timeout。

最终裁决：正常连接没有idle timeout；控制进程崩溃、socket关闭或连接真正断开时自动SAFE。该检测在后台完成，不需要密码或用户交互。

## 7. Gate 6 — SLM与未来可复用correction

### 当前Feedback

裁决：

- 当前优化observable就是all-shot fluorescence，不宣称trap depth；
- 100 shots用于coarse，但算法应使用当前可获得的最佳estimator/controller；
- final是否增加shots由算法根据实验数据uncertainty和产品预算决定；
- 用户按Stop表示提前接受本次run中最好phase；Task保存/apply best artifact；
- 用户若不喜欢，可在Editor/load history中主动切回旧phase；
- 非用户Stop的异常失败仍应在incoming phase known时恢复，unknown时不能伪恢复。

### Science Context

采用显式Science Context，但必须区分三种correction：

1. Vendor/device correction：X15213本机面形/LUT相关，adapter层应用；
2. Reusable optical/system correction：希望通过dense grid、wavefront measurement、atom fluorescence或其它校准得到的`my_correction`，可跨不同Target复用；
3. Per-geometry feedback weights/Pattern：当前构型下为均匀荧光调整的target weights和Pattern，不自动等于可复用pupil phase correction。

未来增加一个独立SLM calibration流程，输出`SystemCorrectionArtifact`：

- 明确它是pupil-plane phase map、target-plane response map还是二者；
- 保存适用wavelength、pupil、坐标、valid region和测量方法；
- 任意新Target先solve Pattern，再统一compose reusable system correction；
- 当前SlmFeedbackTask不假装从一组site brightness唯一反演完整wavefront。

这让用户未来实现“校准一次系统像差，不同geometry直接复用”，同时不混淆当前每构型荧光均匀化。

### USB与DVI

- USB：通过Hamamatsu USB SDK把灰度frame写入controller memory/display slot；
- DVI：把SLM当作第二显示器，通过视频线显示一个1280×1024全屏窗口。

实验机正式使用USB。USB成为唯一正式产品transport；删除DVI production/discovery/UI/tests/docs路径，不保留Experimental或兼容入口。USB仍必须完成真SDK、controller、orientation、correction和optical settle验收。

## 8. Gate 7 — Calibration与Scan

### Calibration

用户裁决：不进行此前建议的Calibration artifact/raw archive骨架重设计。

实施边界：

- 保持当前Calibration用户流程、主要artifact形式、默认raw保存策略和三帧report行为；
- 允许在不改变外部行为的前提下移除Atom→Workbench反向依赖、修明确codec/data corruption和降低不必要内存占用；
- 不新增deploy artifact + full run archive双体系；
- 不以“架构纯洁”为理由要求用户迁移现有Calibration工作流。

### Scan

裁决：默认restore pre-run device values。Stop、失败和正常完成都走同一restore owner；恢复失败必须明确报告，不伪装成功。

## 9. Gate 8 — SimulationWorld

用户裁决：SimulationWorld不分层、不拆成多个state owner。

目标设计：

- 保持一个`SimulationWorld`类/owner；
- 对外暴露确实需要的配置API，所有参数在world初始化前一次确定；
- 其余参数作为virtual device implementation内部默认；
- 可使用一个workspace/local simulation profile JSON供查看和人工编辑，但Device Manager Init不在运行时修改它；
- Virtual device init只读取已选择profile并构造immutable config；
- tests通过明确的constructor/config override建立scenario，不直接改world public mutable attributes；
- hidden truth不提供给production算法；必要测试diagnostics留在明确test seam。

推荐配置文件位于workspace/apparatus侧而不是Python package源码目录，避免Git pull覆盖本机simulation设置；UI可显示当前profile路径和只读摘要。

## 10. 实施原则

用户明确要求：

- 不做表面补丁；从root cause和唯一owner重构；
- 性能问题先阅读完整链路、profile并交叉验证，不靠猜；
- 遵守解耦、DRY、唯一真相源；
- 不为旧tests/兼容风险保留历史实现，不加compatibility shim；
- 不加密码、认证、安全框架或新的SHA256/content-hash体系；
- 不增加没有巨大收益的新抽象；优先收紧现有骨架；
- 中途不反复停下来请求普通实现确认，按已裁决方向自主完成；
- commit按阶段性完整milestone，不为每个小修改单独commit；
- 旧设计/计划/package docs随对应阶段完整重写或删除，不只在尾部追加新段落；
- AUDIT文件若被删除，可根据Git/当前证据重建，不让它们成为production依赖。

必要的domain validation、hardware ack、owner identity和数据格式检查属于功能正确性，不视为用户禁止的“防御性代码”。不新增与产品无关的security/crypto/冗余guard。

## 11. 删除优先的实施顺序

用户原则“先按审计删代码，再开始改”被采纳，但不允许把主干留在不可运行的中间状态。

### Milestone 1 — 纯删除与truth收缩

先删除没有production consumer、无需替代的内容：

- Runtime旧exact/builder/live-port/preview框架及self-tests；
- Plot无consumer LivePlotController、`_live_channel`、FitNumericTable等；
- Data零consumer clusters；
- Pulse test-only engine model移入tests、dead transport/lease/public seams；
- unused Atom descriptor/fakes/oracles；
- duplicate package launchers、duplicate docs/SHA tests、historical acceptance/survey；
- public facades与notebooks中只为dead API保活的入口。

同一milestone同步删错误public allow-list/tests/docs，不留compatibility alias。

### Milestone 2以后 — 每个domain先删旧owner，再收口唯一owner

例如Runtime live不能先删所有plugin slots后让Measurement完全失效；应在同一个阶段性commit中：

1. 让现有SignalDataPlane/Host获得唯一append/commit能力；
2. 切真实Camera/Scan/Calibration consumer；
3. 删除全部旧slot/builder路径；
4. 更新tests/docs；
5. 该milestone结束时主干可运行且只有一条路径。

这不是保留兼容，而是一次完成owner替换。其它阶段按Data/Durable → Runtime → Plot/UI → Pulse/Camera/FPGA → SLM顺序推进。

## 12. 裁决终态

- SLM正式transport：USB；
- Remote active forever：正常连接无idle timeout，真实断开自动SAFE；
- 当前不存在实施前仍需用户回答的架构问题。

自包含实施目标已写入仓库根目录：[ZLC_V2_IMPLEMENTATION_GOAL.md](../ZLC_V2_IMPLEMENTATION_GOAL.md)。只有用户以后明确发送该Goal要求执行时才开始代码修改。
