# Zou Lab Control — Current Product Architecture

状态：`CURRENT PRODUCT AUTHORITY`。Real-screen、camera、SLM、optical与FPGA board acceptance仍是明确的实验机runbook，不由software evidence代替。

本文只定义当前产品不变量。当前验证证据和未执行的实验机验收只看`IMPLEMENTATION_PLAN.md`。

## 1. Authority与原则

实施authority顺序：

1. 用户最新明确指令；
2. 本文产品不变量；
3. `IMPLEMENTATION_PLAN.md`当前实现状态与最新证据；
4. 当前代码与实验事实。

所有活文档只描述当前产品；Git记录不构成产品规格。

总体原则：

- 保留八层骨架，删除平行truth和单消费者framework；
- 默认删，不保留unsupported path或“以后可能”使用的抽象；
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
- Plot轴身份只用`AxisRef(domain, axis_id)`稳定key；label只用于显示，不进入
  semantic field identity。Scope在内存、PanelState与Figure recipe中都使用tagged
  `latest`或tagged typed coordinate value，文本坐标`"latest"`不得被当作控制字。
- 同一run/content revision不可代表不同内容；EventRef只表达causal publication，不代替content identity。

### 3.2 Figure archive

- 一个writer、一个reader、一个format owner。
- Writer写入前规划全部member namespace并拒绝碰撞。
- Reader在解释内容前严格验证format、required members、shape、duplicates和non-finite metadata。
- 未知metadata类型拒绝，不自动字符串化。
- Figure只使用稳定`zlc.figure`格式，无数字版本；reader只接受当前完整grammar，其它root或缺失字段均loud拒绝。
- Figure NPZ是可重绘的数据真相，包含typed Dataset、exact PlotSpec、完整normalized parameters、overlay、viewport、selectors、facet focus、classifier、fit和exact causal lineage graph；PNG只是同stem preview。
- FigureViewer按保存的recipe恢复typed plot input，并和TaskConsole Live/Frozen使用同一个Plot host/configure与accepted `DisplayDescription.spec` contract；不得按array shape重新猜plot kind。一个immutable archive可在同一Monitor board增加多个共享`PanelCardView`，默认panel恢复exact recipe，operator新增的其它plot kind只从同一typed Dataset schema重新compose。每个panel的Setting、可关闭Edit tab、size/signal/cell kind/display/fit均复用TaskConsole现有owner；saved/static Edit不显示live cadence、producer、snapshot refresh或第二套Save controls。Lineage以root、event nodes和direct parent IDs保存；Viewer把内部event ID只用作引用，Logic页显示有意义的Logic run snapshot，Devices页按实际device聚合working point，Flow把每个exact Logic event和共享Device投影为唯一node及显式edge。
- Panel Edit不得嵌入第二份Logic Editor；Direct Producer只显示稳定node identity并打开/聚焦现有Logic tab，draft、Start/Restart与device choices仍由唯一Logic Editor owner管理。正常首开只投影一次Editor Form；Editor host接受后只有accepted state或control surface真实变化才重放，关闭的Editor不得先构造projection再由View拒绝。
- Measurement worker若显式消费一个Dataset signal，必须在取出值的同一时刻把该exact source publication交给Runtime commit；Runtime是direct parent edge唯一owner。Scan不得只保留`SignalValue`后丢弃publication identity，也不得在Figure Save/Viewer中按`source_signal`反查latest补边。FigureViewer Flow只表达archive中真实的causal parent edge与Device-use edge，不重复parameters或device snapshot详情；这些分别由Logic与Devices页显示。Devices页用run record的stable role→instance mapping解释run/event record中的`device_snapshots`，同时读取`actual_devices`和lineage顶层仅对实际引用epoch展开的active override；不得猜`role == device key`，也不得拿override为空解释成run未使用device。Task生成而非Panel Save生成的normal/partial report Figure在source中保存该Task已经冻结的run record；Viewer可据此显示单个Task及其Device，但不得伪造Runtime event DAG。
- Dataset/Figure encoder只写caller-owned binary IO；路径原子发布唯一属于`zlc_durable`。

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
- 普通Monitor signal没有finite canonical extent，UI显示latest complete event；Processor不得仅因“derived”就增加科学轴。输出契约的`index_by_source`只声明history能力；只有真实consumer按window取得lease后，Runtime才从当时的current event开始建立带通用`primary-index`的bounded ordinary Dataset，并按全部active leases的最大window保留、在最后一个lease释放时立即归零。Runtime内部以绝对source ordinal排序、保留gap；materialized Dataset只暴露相对最新事件的普通整数坐标，最新固定为`0`、过去为`-1/-2/...`、缺失offset为invalid。lease之前的event不回填、不伪造；Runtime是唯一跨publication history owner，所有Plot读取同一Dataset且不建立Plot-kind、Fit或Workbench专用history lane。显式window只按该普通轴的相对坐标选择最后N个cell，不形成第二份history；除此之外Plot/Workbench不得识别primary-index为history或自动增加Latest scope，它与其他AxisRef使用完全相同的fate、selector、focus和viewport规则。history event/indexed表示切换即推进该signal的presentation epoch并使全部consumer重新投影，即使scientific publication未变；该epoch不冒充run generation或content revision。Occupancy exact处理每个camera cycle，但其公开Monitor几何仍是当前cycle的`frame`，processor自身不得在published geometry上再叠一层source index；这不禁止它像其他Monitor输出一样`index_by_source`声明history能力——那条history仍由Runtime在lease成立后用通用`primary-index`单独建立，几何不变。
- `scope/reduction/fate`只决定怎样投影canonical view，绝不决定选择event还是canonical；同一publication不能因Panel semantic不同代表两份不同数据truth。
- Incremental placement沿repeat与point rows；多维scan/grid通过point table与grid topology表达。一个cell payload原子完整发布，不新增cell-internal tile/slice streaming contract。
- Canonical display materialization只在实际display consumer到期时合并/cache，并在Qt owner thread之外执行；不得让producer每commit强制复制full prefix，也不得因Panel存在与否改变采集结果。
- Live Panel、Panel Edit/Refresh/Save、selector、fit input和overlay必须从同一accepted canonical presentation snapshot投影；无法唯一对齐即拒绝。
- Bounded indexed history已经淘汰的旧publication属于正常presentation过期，不是signal/Panel故障：Runtime必须以明确的expired/cancel结果拒绝该次materialization，Surface丢弃这次排队更新并保留上一完整front、host和control vocabulary；绝不能拿latest冒充旧publication，也不能让普通`ValueError`关闭host或清空Fit/Setting UI。
- Panel Edit的冻结数据是否落后于Live与冻结配置是否仍兼容是两个状态：同run coverage增长只标记`data advanced`，不得阻止对exact frozen snapshot做Fit、Refresh或Save；只有signal/spec/axis vocabulary真正不兼容才阻止保存。接受Display/Fit/Focus等配置时，PanelState、frozen target和两surface配置必须原子推进，不能先把自己标stale再由stale阻断同步。
- Panel的title shape与Setting semantic都只能读取同一publication的canonical current Dataset，不得读取最后event chunk冒充完整signal。title结构固定为`(repeat) × (point/scan geometry) × (cell payload)`三组；例如Survival field scan显示`(20) × (10×10×10) × (3×35)`。PanelCard以独立的accepted-data projection持有title structure/scope，不从Setting parameter surface读取；每次surface accept都直接更新该projection，即使不改变任何PanelState/control vocabulary。GridTopology已命名scan axes时不得再暴露flattened `point` ordinal；多维FacetGrid默认facet最外层真实scan axis，其余轴保持可编辑的Reduced。即使当前projection因FacetGrid 64-cell上限等原因拒绝，错误只标记不可用的presentation/fit，完整canonical scan-axis fate仍必须留在Setting中供operator修复。live publication未改变PanelState或authoring字段域时不得reconcile Setting form；Plot kind是Add Panel时确定的panel identity，不进入Setting通用表单，FacetGrid仅暴露可变的Cell kind。
- Panel host accept后的Setting/PanelState metadata必须继续读取该publication的`canonical_schema`，不得在像素已使用canonical Dataset后又用最后event chunk schema覆盖fate vocabulary。一个表单提交的全部fate rows是一次原子axis-role assignment：先确定全部x/y/group/facet目标，再处理Reduced/Pooled/scope及真正空缺的required role；结果不得依赖row迭代顺序或中间冲突。
- Fate vocabulary只由schema与plot kind声明：每个axis无条件列出该kind拥有的全部roles，UI不得运行candidate projection、cell-count、surface size、DPR或layout feasibility来删除选项。实际组合是否合法、Facet是否超过容量只由提交后的Plot/layout transaction判断并loud拒绝；拒绝不得改写或缩减Setting vocabulary。
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
- Hosted Task需要人工决定时，只能通过NodeHost同时公开一个带唯一request identity的operator-input request并在worker内等待；Workbench按request kind提供交互，response必须精确匹配当前request。Stop唤醒并取消等待，不使用轮询、第二条Task lifecycle或plugin-specific Workbench状态。未来人工Scan axis复用这个lifecycle，但其业务与UI不属于Calibration实现。

### 4.4 TaskRun与durable artifacts

- 只有实际`Start`进入NodeHost worker时才分配唯一run directory；打开Editor、draft validation或build failure不得留下空run。
- 每个run在任何不可逆工作前原子写`run.json`。它是run identity、normalized inputs、状态、latest progress、artifact inventory、terminal result和failure的唯一lifecycle记录。
- Task只保存由domain owner挑选的重要、可复算或不可替代artifact；Runtime不得自动dump live Dataset、全部shot或所有中间状态。
- Artifact必须先完整原子写入run directory，再按semantic contract注册；`run.json`只列已存在、已注册的文件。声明的final artifact未注册时Task不得成功。
- Run根保存`run.json`与summary；domain final进入`final/`，重要图进入`figures/`，精选candidate/site数据进入`data/`。
- Figure始终成对保存：同stem `zlc.figure` NPZ为primary data artifact，PNG为无science contract的preview。Calibration、Temperature和SLM Feedback遵守同一TaskRun规则。
- Stop保留已完成的精选artifact和明确partial状态；failure保留错误、last progress、已注册artifact与rollback outcome。进程异常终止留下非terminal `run.json`，不得清理或伪装成功。

### 4.5 Task运行中冻结

冻结：Add Logic Node、当前Node source/preview signal、overlay binding、scope/reduction/fate以及冲突硬件配置。

允许：其它Panel、当前Panel的样式/viewport等纯显示参数。Calibration继续显示long/readout/long三帧Grid/Figure。

## 5. Plot、Fit、Overlay与Selector

### 5.1 Exact Data/Fit pairing

- `display_interval`只控制Surface刷新deadline，不决定active history lease内的Measurement primary index是否存在。Runtime只在lease起点之后为indexed-derived Dataset写value或invalid；昂贵Surface计算同一same-shot group只允许一个active，并在忙时只保留Plane latest完整输入，中间indices仍以invalid存在而不排完整frame。
- Panel只原子呈现`data@N + fit@N`。
- Fit selection唯一优先级是committed Area ROI（或显式X-range）→viewport→full range；FacetGrid selector必须保留所属focused cell identity，任何PanelState重放不得把ROI降级成viewport/full。
- Fit参数编辑只有一个紧凑表达式：`name=value`表示该参数精确固定、从优化自由度中移除；`name=guess(value)`只替换初始猜测。表达式使用当前painted单位，PanelState、accepted description和Figure只保存canonical `fixed`/`initial` mappings，不保存第二份原始文本。语法/参数/domain错误只忽略该optional override并继续同model全自动fit，同时保留可修的临时draft和loud warning；不得用窄bounds伪装fixed。model切换清除旧model的fixed/initial。Curve、Rolling、Histogram、Image与Facet cells共用这一contract；fixed参数不报告估计误差，DOF/covariance只按free参数计算。
- FacetGrid overview每个cell显示哪个fit parameter是display state，不是solver request：Workbench把`Cell fit value`普通下拉放在Fit section的parameter expression正下方，但字段明确写回display owner；未选择fit时不显示。choices包含`Model headline`及当前model parameter identities，默认`Model headline`。修改它只重画annotation、不得re-fit；model切换仅在旧parameter不存在于新model时回到`Model headline`。focused cell仍显示完整formula与全部参数。
- FitResult携带source parent/generation/revision；任何history/window投影按Measurement primary index连续，未计算、失败或timeout的位置invalid/NaN，window长度按source indices而非成功结果计数。
- Fit计算在后台worker；Qt owner thread不等待Future或执行fit。
- Active Fit超过1秒必须loud标记该source index invalid并从Plane latest继续；不得积累完整frame FIFO，也不得永久锁住Panel、Qt、Stop或close。普通cadence/backpressure跳过计算的indices同样invalid但不是solver failure；raw Runtime data始终完整。

### 5.2 Performance与state

- PanelState一次应用是幂等transaction；no-op产生0 solve、0 render、0 front。
- `PanelState`是可编辑、可在拒绝后继续修复的authored target；只有Plot成功接受后返回的
  完整`DisplayDescription`才是当前Live/Frozen/Viewer pixels的accepted truth，其`spec`也是
  capability、selector、classifier、overlay和viewport判断的唯一依据。拒绝的target不得
  覆盖accepted truth。
- Plot selector与viewport observation都携产生它们的exact Dataset generation和revision；
  TaskConsole Console是唯一interaction owner，核对当前accepted publication/spec后才写入
  PanelState、发布derivation或镜像到另一host。Plot host只产生gesture observation，View只
  投影；retired/stale/non-classifier回调不得写回，selector不得跨semantic plot kind。
  Viewport identity还包含Dataset schema fingerprint、accepted spec、display coordinate units和
  focused cell；相同shape但不同axis roles绝不共享数值范围。Live configure接受的新viewport
  必须写回此identity，Frozen/Edit/Viewer只能重放仍匹配的范围。
- ImagePlot及FacetGrid的image cell统一使用`nearest`像素呈现；interpolation不是Parameter、PanelState、Figure recipe或UI字段，任何Logic/Task不得另行设置。
- Image/Heatmap的主显示框始终是固定正方形，layout、首帧、zoom、pan、Single/Facet/Focus切换均不得改变它；每个离散data point同时是正方形screen cell。规则grid以x/y cell pitch的唯一比例把canonical坐标归一为lattice geometry：canonical scan step只控制tick、selector、overlay与fit的坐标映射，不控制cell长宽。非方阵数据在square frame内居中letterbox，数据extent本身不被改写；zoom按两轴相同whole-cell span在固定square box内修改viewport，不得重新layout。50×50 scan必须完整填满square frame，即使两个scan轴步长不同。
- Single与Facet的规则tensor数据都先由同一个retained-axis projection一次归约，再只把结果包装成各自payload；不得为某个plot kind另建Facet数值kernel。Single、Facet overview和Focus的每个cell必须经过同一个kind preparation/render owner；Focus只选择同一个accepted cell并换layout/viewport，不重新解释数据、fit或annotation。Facet overview及steady Curve/Image/Fit/SEM保留native raster快路，但native与Agg只允许消费同一份prepared cell state；native拒绝必须整帧回到已准备好的公共draw path或保留上一完整front，不得出现partial/blank cells，也不得靠一次pointer materialization才能恢复。
- Curve prepared state同时拥有series、valid runs、SEM low/high、fit source presentation与style。Overview error bar必须保留每个独立stem/cap的几何并使用subpixel coverage（可在cell-local supersampled buffer绘制后area downsample），不得把多个bar按整数display column合并成min/max envelope。Facet pooled y范围必须包含finite SEM low/high。Fit annotation由公共Matplotlib MathText语义owner格式化；native可缓存MathText最终RGBA，但不得删除`$`、反斜杠或下标后用第二套plain glyph语法重画。
- Color-limit drag的每个accepted move是一个原子preview transaction：先更新candidate与native/Agg共享clim authority，再compose一次并发布该front；不得先发布旧颜色front，再在独立cadence分支recolor，release只负责提交最终DisplayState而不是第一次显示颜色变化。
- 同一shot的Surface仍原子accept；staging只把active-fit surface排在display-only sibling之前，使更深依赖链先启动，不改变panel cadence、cohort membership或accept顺序。
- RegularImage live batch即使具有完整warm seed也必须保留cold proxy竞争，再以选出的seed做full refinement；warm不能跳过cold证据、成为不可恢复的authority。
- Title/layout等非plot变化不得re-fit。
- 删除重复configure/clear/replay与多front handoff。
- Qt owner必须在RasterPlotHost第一次render前把当前screen DPR以plain scalar交给Plot；不得先按默认DPR生成front，再在Widget挂载后为同一data/state重画一次。Form consumer在FormSpec结构和实际Widget值均已匹配时只接受新metadata，不得reconcile；keyed runtime choice domain真实变化仍强制刷新。
- Fluent choice是`zlc_ui`唯一前端owner：collapsed控件、一个owned item model和operator信号在控件本身；flat/tree popup view只在operator第一次展开时建立，随后复用，Tree不得先造flat view再替换。popup QSS只有一个共享声明，数值/choice authority仍是typed model，Workbench、Plot和Logic不得感知popup、font metric或Qt私有view。popup先确定最终高度，再以真实delegate `sizeHintForColumn`及`view.width - viewport.width`实测frame/scrollbar chrome并迭代到几何稳定；不得手算字体/padding/native scrollbar metric。横向bar保持`AsNeeded`，只有popup达到active-screen宽度上限后内容仍溢出时才出现；Tree展开/折叠与open model变化重走同一owner。
- Semantic Fate中的Scope始终是一个popup action，不得因轴坐标数量隐藏能力，也不得把全部坐标展开成popup rows。选中后collapsed Fluent control显示`Scope: coordinate`；只有控件获得焦点且当前为Scope时，滚轮才按schema真实坐标顺序切换，未聚焦或其它fate时滚轮继续交给外层页面。PanelState、PlotSpec与Figure仍只保存完整tagged `scope_fate(coordinate)`，UI不保存第二份mode/coordinate状态。
- Histogram只有`bins`变更需要一次完整sample projection；`density`/`cumulative`只是已接受bins的representation，不得再扫描full payload。复用已settle tick unit时必须在枚举lattice前先核上界，不得因range大幅变化卡住UI。
- 正式96×128 Camera、小Area ROI、主图atomic fit、并行ROI image与一个fit-parameter Rolling Panel链路以100 ms作为profile警戒线；明显的额外cadence、HOL、错误串行和重复render必须删除。若剩余是必要fit/raster/Qt成本，只有能带来实质收益且不增加不相称复杂度的优化才实施。
- 性能以真实TaskConsole、1/4/8 panels、fit+overlay、Setting/Edit和Qt owner latency为profile对象。

### 5.3 Overlay与selector

- Overlay producer发布匹配中立Plot contract的numeric/bool companion signal，并在同一run record中携带该contract要求的geometry document；`zlc_plot`拥有通用adapter与renderer，Workbench只按contract路由，不import domain plugin，也不重建science。
- Data、Fit和Overlay共同使用同一个scope/axis/fate projection；无法唯一对齐则拒绝。
- ROI/binning坐标只由一个transform owner处理。
- Selector Off时plot不消费任何pointer gesture：不画selector、不zoom/pan，也不响应双击facet focus；普通滚轮继续滚外层board。
- Selector On时，FacetGrid overview只响应双击进入cell，不得在overview开始area selector；进入具体cell后，selector才按该cell的canonical projection工作。
- Curve的invalid位置切断line且绝不跨洞连接；standalone Curve与FacetGrid Curve cell中，每段仅一个valid点时用同series颜色/alpha/linewidth的短横线glyph显示，不建立scatter或第二series。Grouped Curve与Grouped Rolling共用series interaction：hover只轻微加粗命中的line/孤点glyph，其他lines保持正常alpha；click lock才加粗并压暗其余lines。Series文字固定在对应axes内部右上角、无背景框，locked文字以`* `开头；locked时滚轮按Group axis顺序切换line，未锁定时滚轮仍缩放viewport。

## 6. UI与Lifecycle

- `zlc_ui`不拥有domain parser、device state或plot lifecycle。
- Qt slot不得执行blocking I/O、device tune或`Future.result()`。
- Window只有在owned command、worker、executor和claim安全退出后才能消失。
- Device Manager的`instance_id`是稳定device identity，operator-facing role只是metadata；改role不得把同一硬件变成remove/add。Loaded card的Control与Close都只提交intent，不能由View直接关device。
- Active apparatus变更走同一个`ExperimentSession`内的差量reconcile：相同key/type/canonical parameters的leaf、SignalPlane、TaskConsole与Panel继续复用；新增只build新增leaf，remove/change/Close只处理受影响leaf、world-bound closure及factory dependants。只有完全相同的draft/live集合才把主按钮解释为Shutdown。
- Reconcile前以device-key maintenance barrier阻止新Logic/command，停止并等待受影响Logic lease，关闭对应Control；已有不可取消command时loud拒绝。partial close/factory cleanup失败后，所有仍open的leaf必须继续由Session或recovery owner强持有，effective live config与TaskConsole device projection同步后才允许下一次操作。
- Device operation或projection-refresh pending期间Control、Close、TaskConsole X和root close不得越过owner状态；失败保持window/session可达并提供只刷新projection的retry，不重复hardware work。
- Hosted Task可登记且只能登记一个domain-owned partial-exit writer；Runtime在worker线程、撤回Dataset及把TaskRun标为stopped/failed之前恰好调用一次。Writer只能从已经完成的数据原子写并登记checkpoint/process/Figure/preview/summary，不得制造required final；writer失败只附注原始错误，不能覆盖原始hardware/science failure。Calibration、Temperature与SLM Feedback都必须使用该边界保存各自可证明的partial报告。
- Device Control只显示adapter声明的`TunableField`：稳定表单metadata、authoritative current、当前是否live-write及dependency group。每行统一为Current、Desired、Live apply、Apply和Status；打开/显式Refresh及成功Apply后的readback只走session-owned串行device worker，Qt不碰SDK，也不做周期hardware polling。
- Logic在实际Start lease中声明protected fields；运行时才选出的device scan ports以nonexclusive resolved field claim加入同一lease。Device Manager按所有active claim与dependency closure锁字段，不按camera type或字段名特判；无owner时正常写，有owner时只有未claim且adapter确认live-safe的字段可在operator接受风险后写。
- 风险接受只绑定当前`device_session_id + device-specific owner revision`。owner或session变化立即失效；字段命令在DeviceUse同一原子锁内再次核revision/claim，active field command阻止新Logic Start。in-flight live edits只保留每字段最新值，owner变化后尚未执行的write取消。
- `device_session_id/settings_epoch`只在成功且effective值实际改变时推进；requested/effective/readback与active owners只在Logic运行期间的真实override中记录。Camera frame在adapter接受/复制边界冻结epoch，不能在publication时读取“当前epoch”倒填旧frame；无法证明边界的Pylon首批readback保守标为old/new mixed。Publication只带压缩epoch ranges，Figure只展开lineage实际引用的记录；idle调整不进入历史。
- Pulse Stop UI立即进入Stopping；Stop/SAFE高优先级并可取消普通wait/transport，hardware ack后台完成。
- Timeout显示真实错误但不冻结UI；未确认前不能显示Safe。
- Form reconcile必须按当前schema重建dependency graph。
- PanelState decoder只接受当前完整grammar；owner wake和产品Figure save各只有一个实现。
- FigureViewer与TaskConsole必须复用同一个`PanelCardView` frame owner、Monitor board、panel preset尺寸、title band、Setting按钮和body padding；Viewer右栏是白色Fluent work surface，global action bar固定自身高度，Panel在其下方top-align，不能把剩余窗口高度塞进action bar或让同一2x2 card漂到中部；card title读取当前archive dataset的operator label。左侧InfoPane宽度在window创建时一次确定，任何archive label/value不得改变window split；全部readout统一使用可选择、可自动换行的multiline control，并按Qt实际visual layout精确包住内容，一行保持紧凑、长无换行值不得cutoff、未设height cap时不得产生inner scroll range，整个表单只由外层Info tab滚动。Flow使用Fluent node-edge graph：Logic与Device节点不重叠，共享节点只出现一次，causal edge与device-use edge视觉区分；Workbench只给plain nodes/edges，Qt owner负责字体测量、布局、绘制和滚动。

## 7. Pulse、Camera、Remote与FPGA

### 7.1 Execution vocabulary

- `RepeatRegion`只表达timeline内部loop。
- Cycle/shot、scan sweep和Dataset repeat是独立事实。
- 一个finite execution入口表达N cycles；actual played values进入Dataset coordinates和run record。
- Hardware duration scan的绝对period由32-bit nominal tick base保存；25-bit signed slot只携相对nominal的delta，不得冒充约335 ms的绝对period上限。Host按整张scan table选择能覆盖它的最小整数tick scale，scale受现有signed Q8 coefficient约束且DAC恒为1；不为扩大范围修改RTL multiplier宽度。
- 同一次application的compiler、wire table、readback、Pulse Editor Run/Sync/Hold/Step与Seamless Dataset共用一个量化结果。Dataset coordinates和run record记录实际played values；若所需分辨率使两个不同authored points坍缩为同一点，必须在碰device前loud拒绝。

### 7.2 Camera

- Same-shot保证采用continuous best-effort，不新增hardware marker或逐cycle arm/fire。
- Camera Measurement只按自己的authored frames-per-cycle/repeat采集并核实际返回cardinality；Camera adapter不解析Pulse window数量，也不以exposure审查Pulse cadence。Adapter的source ordinal只编号实际采到的frames，必须从本次arm的0连续递增。
- qCMOS的ROI、exposure、trigger/readout各由adapter的单一working-point owner管理；未变化字段不得在每次Start整套重写。Measurement冻结设置操作返回的authoritative readback，不再为同一capture额外读取完整property surface；相同exposure/ROI的restart因此不支付冗余sensor reconfiguration。
- Camera auto Panel从canonical publication/preview signal建立；signal尚未publish时显示等待状态，但不得用重复device配置、额外generation或固定5秒轮询作为Panel接线条件。
- Camera settings provenance属于frame event而不是generation identity：`run_record`在一代内保持不变，frame冻结的小型`event_record`可变化；finite/scan前缀与有界indexed history按实际保留chunks合并epoch ranges，monitor只携带当前event。
- Temperature保留约20ms authored exposure；Pulse timing与camera exposure是各自owner的独立输入。
- Virtual sequencer按compiled wall cadence逐cycle并支持Stop；每个到达virtual camera的frame event都被采集，不根据Pulse时间或camera exposure私自skip、制造ordinal gap。

### 7.3 Remote

- 无密码、认证、TLS或权限UI。
- Second client默认last-client-wins；旧handler立即失效，takeover前旧active command必须成功Stop/SAFE。
- 正常连接无idle timeout；控制进程/socket/连接真正断开时自动SAFE。
- UART auto枚举COM、优先USB VID/PID，并只在word-63 fingerprint匹配后选用；
  显式port把探测限制为该端口。auto探测失败才回退JTAG，显式UART失败则报错。
- 只有server process持hardware transport；不保留假的进程内Interprocess lease。

### 7.4 Host/RTL/build invariants

- Load前核target ABI、clock、geometry、counts和delay FIFO capacity；不把camera exposure或frames-per-cycle反向解释进Pulse program。
- Count必须是合法hardware range内整数，不clamp/wrap。
- Hardware SAFE独立gate TTL/DAC data/clock；LOAD/FIRE前pins保持safe。
- Public DONE等待delay FIFOs和final DAC latch完成并进入安全态。
- Underflow与engine delay-FIFO overflow sticky且loud；scan point0必须resident。UART CRC/framing/address fault由framed reply与独立LINK_ERROR bit报告并由host retry，不能污染engine ERROR或使一次已正确执行的长pulse失败。Observer/transport异常必须保存真实exception与两次status/cursor readback，server只有真实DONE才打印DONE。
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

- Target使用稳定`zlc.slm.target` strict格式保存intensity和objective，只是Editor authoring import/export artifact，不是run consumer的第二Target truth。
- Science Context使用稳定`zlc.slm.science-context` strict格式，只持久化run的唯一frozen Target、测量前固化的16-bit circular Pattern模差分、pupil/operator语义参数、system correction引用与command receipt；numeric pupil、operator wavefront和composite phase由同一SLM核心公式重建，不重复保存全尺寸矩阵。16-bit固化发生在Editor Send或Feedback camera shot之前，已测candidate与可加载artifact逐元素一致；Editor Load直接atomic adopt并且不重新solve。Reader只接受当前完整Context，其它root或缺失字段均loud拒绝。
- Command receipt保存USB/profile/wavelength/orientation/correction/outcome。
- `SystemCorrectionArtifact`明确区分pupil phase map与target response map；不得把per-geometry site weights冒充通用wavefront correction。

### 8.3 Solver与Feedback

- 保留sparse WGS-Kim、fixed far-field phase、selected DFT和caller-owned optimizer state。
- Inner solve走到canonical numerical gate，不为省几十毫秒增加physical candidate。
- Feedback mode是leaf-owned显式字段；当前唯一mode为`qcmos_bright_dark`。Pulse由operator显式选择；camera exposure是独立、可见、可编辑的authored字段，默认`0.1 s`。Task不从Pulse或Calibration猜exposure，也不自动判断Pulse/exposure的科学一致性。
- 当前mode复用canonical Camera Measurement `repeat=N`，每cycle严格一张camera frame；同一逐帧publication经mean reduction实时显示，Feedback只把完整registered Target SiteMap写入该次camera run geometry，不发布第二份camera数据或三帧reference判据。
- Calibration只提供Target→camera注册所需的site centers、BOX半宽/积分方式和frame坐标几何；Feedback不读取其dark/bright/threshold、exposure、photoelectron mode、camera identity或readout working-point provenance。实际camera requested/actual exposure、effective unit与conversion进入本run metadata；saturation只由本次actual raw integer maximum转换到本次effective unit判断。
- 每个site仅使用本candidate完整一批authored shots的raw BOX值选择单高斯或双高斯；完整batch的受约束双高斯数值有效、满足基本分量/间距条件且full-data ΔBIC>0时，`bright_mean-dark_mean`才是observable；拟合正常但不满足为single，数值/采集失败为invalid。
- Controller保存每site的归一化Target share、正式double历史、probe方向与single/observable边界。没有可用正式double历史的single使用用户`probe_factors`做两侧诊断；产生double且最接近reference的winning effective share在`probe_combined`中直接采用，不再乘`feedback_gain`或普通`maximum_weight_change`，固定总功率不可同时满足时只向baseline缩到最大可行值且不反转方向。其余double允许共同缩放以提供该绝对share，但保留double内部的相对修正；若下一次完整测量中direct-adopted site仍single，或共同补偿使原baseline-double变成single，则从新的实际baseline重新probe这些矛盾site。两侧均未产生double且baseline未变时不得原地重复。已有正式double bracket的single沿边界继续。只有正式double更新使用以用户`feedback_gain`为初值的adaptive scalar：共同double sites连续两次显著改善后乘1.25，显著变差后乘0.5，不确定性内保持。invalid保持实际份额；Diagnostic probe不进入adaptive/formal history、best candidate或反馈指标。
- mode、Pulse、exposure或任一控制参数变化时丢弃prior response state；用户要求的dark增幅若在固定总功率下不可行，静默缩到本轮最大可行值。
- 默认每candidate为100 shots、12次formal update；每个phase严格一批shots，下一批前必须确认不同phase。`probe_combined`是正式update并计入上限，0.5/2等diagnostic probe candidates不计；最坏情况下每个formal update前都需要一组diagnostic probes，因此candidate容量为`1 + max_updates × (1 + len(probe_factors))`，两factor且上限15时为46，实际只在direct adoption被新测量反驳时才re-probe。正常运行完成全部authored updates后选择全site ratio最低的candidate；没有内置ratio停止阈值。Stop保留最佳已测candidate，置信区间只作记录，不触发额外采集。
- Feedback取得SLM后自己apply并确认frozen Science Context phase，并在shot前发布该phase；Context receipt是provenance，不要求operator事先Send/Save。normal terminal与Stop只从完整测量过的candidate中保留最佳/最可观测状态，异常failure恢复Context起始phase。
- Feedback run只保存精选candidate数据：stable site table、每candidate BOX shot×site samples、fit/classification、Target/control weights、update action、metrics、phase-change fact和command receipt；不保存raw camera frames。每个operator-visible candidate Context包含其可加载phase，final仍只有唯一selected Science Context。
- Feedback candidate是operator-visible、可直接Load Science/Send SLM/作为下一run Science Context的完整Context，保存在`candidates/candidate-XXXX.npz`；内部BOX/fit/decision数组单独保存在`data/measurements/measurement-XXXX.npz`，不得再用candidate命名冒充可加载Context。加载candidate仍使用既有Science Context输入且完全由operator手动选择；它只确定新run的起始光场，新run从candidate 1开始并执行本次authored `max_updates`，不自动查找旧run、续编号或继承旧run的update预算。
- Feedback summary同时提供机器可读JSON和人读文本，明确initial/selected uniformity、confidence、observable sites、common-site total brightness、selected candidate、Stop/failure与rollback。
- Feedback重要summary图固定为`uniformity_history`、`site_signal_evolution`、`weight_evolution`、`selected_site_histograms`、`camera_initial_selected`和`phase_initial_selected`；此外每个完整candidate（含diagnostic probe）各通过正式Figure API保存一份按site分cell的Histogram与bimodal Gaussian fit NPZ/PNG。它们只是run artifact，不形成Monitor preview；正常或Stop终态只写一个final Science Context。
- Feedback自动preview固定为带编号site map的实时Camera Measurement mean reduction、observable uniformity、site signal evolution与Target share evolution；phase仍发布且保存最终Figure，但不自动占用Monitor panel。
- Task preview只冻结运行中的signal/overlay/cell kind/semantic与publisher wiring；Selector、viewport、hover、line lock等Panel interaction始终由全局Selector toggle控制，Calibration、Feedback与普通Panel行为一致。Task锁只阻止selection反向改写正在运行的producer draft，不阻止本地交互状态。
- Task到达completed/stopped/failed terminal时移除该run自动创建的preview panels；用户手工创建的Panels不受影响。Panel header使用紧凑Setting与紧邻的`×`；`×`单击进入红色确认态，系统double-click interval内第二击才删除，超时恢复中性灰。
- Sparse-only contract明确；dense Gaussian/Flat Top先修算法定义和early stop，再profile CPU，不引GPU。

## 9. Calibration、Scan与Simulation

- Calibration保持既有科学流程、当前artifact和三帧preview。
- Calibration site detection只有两条并列证据：相邻reference frames的空间带通差值，以及全部reference frames的空间带通average。一个明显相邻帧变化即可保留single-loading possible site；steady/high-loading site由average保留。两条路径使用同一个authored `detection_sigma`下限并按全图/transition数量提高family-wise bar；site identity始终取完整average的局部峰。不得按奇偶/half分帧，不得用split consistency或全局saddle heuristic否决已经成立的证据。
- Calibration可由operator显式开启detected-site review：采集与site detection都只执行一次；检测完成后由同一run的短期companion producer发布reference average与candidate SiteMap，TaskConsole允许单点或框选排除高阶衍射/ghost site。确认后只用保留站点构造最终SiteMap并执行一次全部下游拟合；不重新采集、不重新检测、不二次确认。窗口壳、搜索、site checkbox、scroll、status与buttons全部由`zlc_ui` Fluent view拥有，`zlc_plot`只拥有Image surface的point/rectangle gesture与overlay，Workbench只连接两者。最终报告同时保存candidate/excluded/final identity映射和可由FigureViewer重开的`site_review` Figure/PNG；不开启时外部行为与artifact集合不变。
- 允许不改变外部行为的dependency解耦、明确corruption修复和内存优化。
- Calibration只产生与SLM无关的camera/readout artifact，UI和Task都不接受Science Context。SLM Feedback在同时拿到Calibration与Context后做Target X/Y→camera X/Y直接正向注册，并为未观测site生成predicted BOX；不枚举翻转、旋转或轴交换。
- BOX model仍为Calibration/Occupancy持久化自己的readout事实；Feedback只取BOX geometry。未观测Target site由注册产生predicted BOX，并与实测site一起接受本次run的双高斯估计，不伪造Calibration dark/bright样本。
- Calibration threshold method保留operator选择并默认`gaussian`：每个site/readout model只用全部finite short-shot signal做无标签双Gaussian mixture fit，按均值识别低/高分量并保留fit得到的population weights；threshold是两条实际加权分量曲线`w_dark N_dark(t)=w_bright N_bright(t)`在两均值之间、令拟合population总误判最小的解析交点。reference真实标签不得进入Gaussian参数、权重或threshold；只允许用于Empirical threshold及最终actual fidelity。Gaussian参数、population或相关解析根无效时该site使用全部有效labelled samples上令实际总正确率最大的empirical threshold；operator显式选择`empirical`时所有site都走该路径。Histogram竖线始终是最终写入Calibration并由`detect()`使用的threshold；Gaussian曲线必须复用Calibration保存的同一组参数与权重，不得由Plot二次拟合，fallback site不得伪造理论曲线。报告分别保存最终threshold在全部有效真实数据上的overall actual fidelity（另存dark/bright conditional值），以及Gaussian threshold按其fit population weights积分得到的theoretical fidelity；fit失败site没有theoretical值。
- Calibration只使用稳定`format="zlc.calibration.readout"`，无数字版本；reader只接受当前完整grammar，alternate root或缺失统计均loud拒绝。
- Calibration run保存final JSON、summary JSON/text及精选报告图；每张报告图都有可由FigureViewer重开的typed Figure NPZ，PNG仅为preview。默认不保存全部raw frames；operator显式请求时才保存采样数据。
- Temperature使用同一TaskRun lifecycle，保存final JSON、summary和生存率typed Figure/PNG，不建立第二套run管理。
- Scan正常完成、Stop或失败都默认restore pre-run device values。
- SimulationWorld保持一个类和一个state owner，不拆层。
- SimulationWorld的物理site只有当前SLM phase经共同pupil illumination、共同low-order wavefront aberration和FFT得到的dominant local peaks这一份动态roster；trap位置、强度、occupancy与Camera位置不得再拆成nominal/extra双状态。所有peaks经过同一个Fourier→camera affine；fluorescence imaging使用一个由共同imaging pupil/aberration生成的shared非对称PSF，不存在逐site随机gain/ellipse/angle/skew。Probe为红失谐，正的trap light-shift参数只把detuning进一步推红，因此occupied bright-dark随trap depth单调下降；loading probability随depth上升。Camera shot真实混合dark/bright population，Feedback不得读取hidden depth/occupancy truth。
- 默认plant的全部不均匀度必须来自FFT前同一个固定pupil amplitude/wavefront phase；该world wavefront与SLM command、Target和grid完全独立，并在每次propagation中始终相加。不得使用grid-resonant phase、target-specific correction或far-field site/field gain。默认nominal depth固定为520 µK；固定20 µK cooling温度下，低于500 µK的trap不load，超过阈值后按一个cooling-temperature尺度指数趋近全局loading ceiling。因nominal本身贴近实验loading edge，普通光学不均匀在不同grid中都会让至少约10% sites不可见，不得按某个grid反推nominal或由测试手改Target weight；`bright-dark`继续由现有probe参数决定。
- Apparatus root `simulation`是image/grid geometry、seed与profile的唯一持久化owner；virtual qCMOS只声明camera事实并消费world image geometry，virtual MOT保持独立的camera geometry。非当前grammar必须loud拒绝且不能形成第二owner。
- Simulation参数在init前通过单一API/immutable config确定；workspace-relative profile必须在任何device factory前解析且保持在workspace内，Device Manager Init不运行时改写。
- Tests使用config override，不修改public mutable world attributes；hidden truth不泄漏给production算法。

## 10. Deployment、Evidence与Docs

- 一个可安装`zou-lab-control` distribution，bootstrap package为`zou_lab_control`；内部八层不独立发wheel或维护版本。
- 所有checkout launcher（包括FPGA build/program与resource estimate）通过同一个Python
  环境owner激活当前tree的bootstrap；安装器是唯一installed-only路径。installed wheel在
  checkout外从distribution metadata解析同一组commands/layers，不保留第二入口名。
- 根`pyproject.toml`是唯一product manifest，`constraints.txt`是唯一resolved dependency surface，`zlc`是唯一console entry并从manifest加载commands/layers/evidence。
- Wheel必须包含bootstrap、八层、Calibration/Scan templates、SLM profile、Plot font及完整有效FPGA RTL/XDC/Tcl assets；installed environment check按distribution RECORD验证归属。
- 正式evidence lanes：software、gui_offscreen、virtual_vertical、notebook_offline、real_screen和hardware runbooks。
- Mock/virtual/offscreen证据不得冒充真hardware/optical acceptance。
- Root Architecture只保存目标不变量；Implementation Plan只保存当前实现状态和最新证据。
- 活文档保持current-only，不在尾部追加change log或修补记录。

## 11. 当前实现状态

当前tree正在按上述不变量完成无版本strict persistence与统一TaskRun收口；当前验证状态见`IMPLEMENTATION_PLAN.md`。任何未执行的real-screen/hardware/optical步骤必须继续标为`UNEXECUTED`。
