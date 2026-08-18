# 07 — 文档、代码与测试的authority冲突矩阵

状态：完成；已吸收`06a–06h`与`09`的package、FPGA、test/notebook及交叉复核结论。
基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`

## 1. 结论先行

当前文档问题不是“少更新几行”，而是没有区分四种不同用途：

1. 规范未来应当怎样；
2. 描述当前代码实际上怎样；
3. 记录某次迁移/验收当时怎样；
4. 作为测试可机器读取的API name list。

同一个文件经常同时承担两到三种角色。测试再读取旧contract/README/SHA，使历史文档重新成为事实authority；根文档却说package文档inactive。结果是：代码、测试和文档每一方都能找到一段支持自己的文字，任何修复都容易被另一段“权威”拉回去。

本轮用户已明确裁决：旧文档只作证据，不默认可信；真实冲突由本审计记录，最终由用户选择。该规则高于旧`AGENTS.md`/Checkpoint里“执行者自行裁决并同步权威”的流程。

## 2. 文档inventory与角色

### 2.1 应保留但必须重写/校准的现行文档

- 根`README.md`：操作者入口与部署说明；
- 根`ARCHITECTURE_DESIGN.md`：目标架构；
- 根`IMPLEMENTATION_PLAN.md`：当前状态/修复计划；
- package README与真正的technical guides；
- `zlc_plot/docs/api.md`、`architecture.md`、`data_contract.md`、`performance.md`；
- `zlc_pulse/docs/contract.md`与FPGA/board docs；
- `zlc_runtime/docs/contract.md`；
- `zlc_ui/docs/console-views.md`、`pulse-views.md`；
- package public contract docs，仅在用户确认各package仍需独立发布后保留。

这些文件不是当前可靠truth，只是有长期用途的载体。

### 2.2 明确历史但未被隔离的文档

以下命名族绝大多数描述2026-08-02至08-04的旧独立仓、迁移分支、旧LOC/测试数和已不存在路径：

- `docs/acceptance-*`
- `docs/reacceptance-*`
- `docs/survey-*`
- `docs/audit-*`
- `docs/goal-archive.md`
- `docs/loc-report*`
- `docs/semantic-edit-*`
- package `GOAL.md` tombstones
- 根`HANDOFF.md`

它们现在散落在active package docs旁。更严重的是，`goal-archive.md`多处仍写“活的计划在GOAL.md”，而所有package `GOAL.md`又明确是inactive tombstone并转指根architecture/plan。这是一个闭环失效的authority pointer。

裁决：`MOVE TO HISTORY OR DELETE`。若保留provenance，统一移入`docs/history/<date-topic>/`，首屏标明旧HEAD/旧repo/不参与测试与恢复；没有持久价值的验收聊天转录应删除。

### 2.3 精确重复

- `zlc_plot/docs/fit-numeric-contract.md`
- `zlc_runtime/docs/fit-numeric-contract.md`
- 两边`test_cross_repo_contract.py`
- `zlc_ui/docs/survey-workbench-ui-2026-08-02.md`
- `zlc_workbench/docs/survey-workbench-2026-08-02.md`

前两份contract还声明“两个repo必须逐字节同步”，当前已经是monorepo；SHA tests继续保护这个历史事实。裁决：一个owner、一份文档；consumer测试行为/结构，不测副本digest。

## 3. 已确认冲突矩阵

| 主题 | 文档A | 文档B/代码/测试 | 裁决 |
|---|---|---|---|
| live fit | 根Architecture/Plan要求data-first、fit later | 当前code/tests/package部分文档锁atomic `data@N+fit@N` | `USER DECISION`；审计推荐live data-first。 |
| fit同shot | Plan称atomic pair不得改 | same-shot真正需要的是source siblings；fit是异步analysis，当前慢fit阻塞全cohort | 文档把“无stale fit”扩写成“data必须等fit”。 |
| Qt async controls | Plot README称外部controls异步提交Host | FigureViewer bound controls在Qt slot里`Future.result(timeout=10)` | 直接GUI freeze风险；实现违背文档。 |
| Notebook owner | Plot docs称Notebook/Qt共享RasterHost worker协议 | Notebook describe/replace直接调用PlotSession | 若保留Notebook，必须统一owner；否则删除library surface。 |
| Runtime generic live | Runtime README把streams/dataset/live port称组织中心 | 从初始monorepo到当前无product caller，真实plugins全用私有slots | 文档描述未落地架构；D-011后删除或给真实consumer。 |
| Selector Off | 根文档说wheel回到board scroll | Qt实现/tests锁wheel仍zoom/pan | `USER DECISION`；不能两边都称contract。 |
| Overlay | 根文档要求Occupancy typed sibling | descriptor不发布；Workbench import插件并从active node临时组装 | 实现违反声明，且ROI坐标错误/terminal不可重放。 |
| Camera output | 根正文仍出现`frame_0...frame_N` | 当前单一`frames + READOUT_EVENT`；附录/代码用新语义 | 删除旧正文或明确迁移，不能保留两套public vocabulary。 |
| finite live | 文档把coverage当已写extent | Camera future zeros validity全真，Occupancy忽略coverage并标complete | 当前实现制造未来事实；docs/tests均未完整守住。 |
| Task takeover | Presenter注释/部分tests称只锁bench、window仍可用 | 真实Qt禁Add Panel、card/settings/logic rows | `USER DECISION`；审计推荐锁硬件/current draft，允许Monitor布局观察。 |
| Task preview | architecture要求Measurement live、Task可监视 | descriptor/Host均不强制；新node可success但从未publish preview | 框架contract缺失，不是逐plugin遗漏。 |
| Pulse repeat | 文档混用RepeatRegion、shots、sweeps、plan repeats | 各node有四种调用法；whole repeat会改变camera windows | 需统一execution vocabulary。 |
| same-shot | 文档称camera ordinal证明shot | DCAM/Pylon ordinal只是copied/retrieved index；trigger loss可不可见 | 若要求绝对保证，需hardware marker。 |
| Temperature | 文档/测试把双帧protocol视为成立 | 默认20ms exposure而edges只隔5.02ms；virtual false-green | 物理协议未成立。 |
| Pulse DONE | 文档暗示wait_done后可safe | delayed event tail可能仍排队，DONE在program final置位 | `USER DECISION`；推荐DONE含tail。 |
| FPGA timing | build/Tcl/README把timing pass写成资格证明 | 50MHz engine主clock没有create_clock，gate可能只检查JTAG TCK | 当前bitstream不能据此称timing-qualified。 |
| FPGA SAFE | host/docs称SAFE进入物理安全态 | clock mux不看SAFE/running，DAC clock可继续或LOAD时提前打开 | 单命令pin-safe是明确硬件缺陷。 |
| FPGA board identity | board README称XDC single mapping、word63证明兼容 | XDC出现顺序和top手写index是两份truth，fingerprint不含pin ABI | 需显式board manifest/build ID。 |
| FPGA build safety | Tcl helper名为safe project dir，launcher默认program | containment只查路径长度；program/flash默认第一个target/device | recovery工具本身有删错目录/写错板风险。 |
| Pulse remote trust | README/launcher把LAN endpoint当普通设备入口 | 默认`0.0.0.0`无认证；stale handler仍可command；SAFE失败仍授予新owner | 当前不可称安全LAN deployment。 |
| Interprocess ownership | 名称/测试暗示多进程device lease | 实现只是进程内dict且transport无人使用 | 明确false-green，删除或实现真实OS lock。 |
| UART auto discovery | docs把auto写成方便fallback | resolver会依次打开并向所有COM发送probe，包括其他实验仪器 | 正式路径必须explicit/allowlist。 |
| Pulse Notebook safety | 教程/README声称idle auto-safe/旧busy policy | 最后cell forever fire全TTL/DAC且无finally SAFE；保存output已过期 | hardware bring-up与offline教程必须分开。 |
| SLM target | 文档说objective是author intent | target JSON只存intensity，load回`auto` | artifact不满足文档。 |
| SLM context | 文档说Feedback保留Pattern/pupil/operator layers | Task用default hard pupil和full science warm start | 实现跨层断链。 |
| SLM real state | 文档称immutable last-commanded为device truth | X15213 init未send/read却报告zero；side-effect failure也可diverge | contract需known/unknown/outcome receipt。 |
| SLM correction | 文档称连续二维unwrap | 实现X后Y separable unwrap | 文字过度声明。 |
| SLM profile | 文档称serial-specific calibrated curve | JSON无raw/provenance/subtype/response；只能证明格式 | 应标unverified直到实验机验收。 |
| SLM feedback 1% | Plan记录100-shot defaults与fixed phase完成 | 100-shot SEM数个百分点，35-site extrema noise floor约1.2；validation几乎不可达 | “算法已完成”与统计可行性冲突。 |
| dense MRAF | Architecture称full-resolution MRAF | Gaussian无noise region、random phase、300轮、ratio无意义 | 实现不满足MRAF定义。 |
| package independence | root称one repo/one distribution，同时never installed | 八层仍独立pyproject/version/contract；root bootstrap靠path injection | 需选择部署模型。 |
| dependency pins | 根pyproject称保留子层pins | root unpinned NumPy/SciPy，install script只读root list | 文字与部署事实冲突。 |
| Atom headless boundary | Atom README称foundation/Calibration math不依赖Workbench | Calibration task、Sequencer/SLM controls反向import Workbench；subpackage eager import把依赖扩散 | 真实cycle且standalone import失败，守卫false-green。 |
| Atom wheel resources | source checkout tests认为scan templates存在 | atom package-data只列Calibration JSON与SLM profiles，漏两个scan JSON | wheel/install模式缺资源，现有tests未build wheel。 |
| Atom notebook | 文档/test名声称virtual installation匹配usage notebook | test不读取notebook；notebook API和保存output均已陈旧 | 要么CI执行短virtual notebook，要么移history。 |
| Calibration artifact | 文档暗示可重放/跨run证据 | 无format/version/strict JSON/unit，raw sample默认不保存且report可保留数GiB内存 | 需定义deploy calibration与full run archive。 |
| historical docs | root称package旧docs inactive | tests读取contract/public lists/SHA并让其成为机器authority | 必须取消测试对历史文本的依赖。 |
| RTL tests | FPGA docs/test名暗示assets有行为验证 | 普通suite不compile/run任何Verilog bench；多数bench打印BAD后仍exit0 | 必须建立自检runner与Vivado evidence lane。 |
| Notebooks | 多个tests命名为execute/coverage | 7本均未fresh-kernel完整执行，Pulse保存2个error output仍绿 | 正式notebook进独立lane，其余删。 |

## 4. `IMPLEMENTATION_PLAN.md`本身的结构问题

该文件同时是：

- 当前Checkpoint；
- 历史commit/test日志；
- 设计裁决；
- 性能数据库；
- 禁止回退规则；
- 下一步计划。

它按时间不断在顶部追加“superseded by later checkpoint”，导致同一主题保留多套数字与语义。搜索某个关键词会同时命中旧equal-loading、新depth-loading、旧5×7-only、新arbitrary sparse；读者必须自己重建时间顺序。测试pass/commit hash又很容易被误当当前代码正确性的证明。

裁决：`REDESIGN DOCUMENT ROLE`。建议拆成：

1. 当前短状态：只列open milestones、owner、blocker、最后验证基线；
2. ADR/decision log：一项产品裁决一份，不随实现细节重写；
3. benchmark/evidence：带environment、HEAD和日期；
4. history/changelog：旧commit与测试数；
5. architecture：只描述目标不变量，不写“某测试X passed”。

## 5. 测试不应把文档当第二实现

保留的测试：

- public facade真实import；
- serialization backward compatibility；
- dependency DAG；
- executable examples/notebooks的最小smoke；
- docs引用的代码片段若确实面向用户。

应删除/重做：

- 两份contract必须SHA相同；
- 从旧Markdown解析public allow-list再与`__all__`相等；
- source token/禁止词扫描替代行为测试；
- README字面出现某launcher/module名就认为产品链成立；
- 历史测试数/LOC/digest成为当前acceptance。

机器契约应来自代码中的typed declaration或唯一manifest；文档从它生成/引用。文档负责解释语义、边界和示例，不维护第二份字段/名字表。

## 6. 推荐目标文档树

```text
README.md                       operator quick start
docs/product/architecture.md    current normative invariants
docs/product/status.md          short current state only
docs/contracts/                 public semantic contracts, one owner each
docs/decisions/ADR-*.md         user-approved decisions
docs/operations/                experiment-machine / hardware procedures
docs/evidence/                  dated benchmark and acceptance evidence
docs/history/                   explicitly non-authoritative migration records
AUDIT/                          this read-only audit and decision backlog
```

package README可以保留本包用途与public examples，但不得再各自声明跨包全局架构或“另一repo必须同步”。

## 7. 需要用户裁决

1. 历史survey/acceptance/goal archive是删除还是统一移入history；审计建议只保留真正有provenance价值的少数，其余删除。
2. 是否继续把package contract Markdown当machine-readable public name authority；审计建议否，typed code/manifest唯一。
3. `IMPLEMENTATION_PLAN`是否允许拆为current status + ADR + evidence/history；审计建议拆。
4. 根Architecture的目标是描述理想终态，还是严格描述当前实现；建议目标不变量为主，并在未实现项显式标`NOT IMPLEMENTED`。
5. package独立发布需求是否仍成立；它决定package contracts/version/docs的保留范围。

## 8. 最终覆盖索引

逐文档/metadata裁决不在本文重复数百行；按owner落在：

- Data/Durable：`06a-data-durable.md` §8；
- UI/Workbench：`06b-ui-workbench.md` §7；
- Atom：`06c-atom-remaining.md` §12；
- Runtime/Plot：`06d-runtime-plot-remaining.md` §9；
- root/bootstrap：`06e-root-bootstrap-packaging.md` §3.7；
- FPGA/RTL/build docs：`06f-fpga-nonpython.md` §10；
- 166 tests、14 examples、7 notebooks：`06g-test-evidence-architecture.md`；
- Pulse Python/remote/transport docs：`06h-pulse-remaining-python.md` §14；
- 报告自身矛盾与scope复核：`09-independent-crosscheck.md`。

结论：当前没有一份可直接当作“code truth”的大文档；保留载体必须按用户八个gate重写。历史文件先隔离，current contracts从typed owner/实际behavior生成证据，最后才更新prose。`07`完成只表示冲突已清册，不表示用户已裁决或旧文档已修改。
