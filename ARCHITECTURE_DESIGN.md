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
- 公共restriction按已选行读取紧凑映射，连续局部范围保留slice；固定Scope跨revision只在原有projection cache中复用相同不可变plane和相同局部选区。候选缓存与已接受/Frozen状态隔离，成功后仅保留本次消费条目，取消限制或关闭时释放；schema、placement和ref始终来自本次snapshot，不以复用切片替代科学身份。
- Validity的存储与组装只沿schema声明的component轴，不为整cell判决展开逐像素mask再压回；需要逐像素布尔视图的数值消费者才广播。裁剪保留未改变的不可变Axis/Domain身份，不重建无变化的坐标与codes。
- Validity入口只接受明确bool contract，不做numeric truthiness转换。
- Selection按AxisId和typed coordinate唯一解析；重名或不可唯一映射必须拒绝。
- 数值坐标的精确定位由AxisSpec统一持有：有序数组只记方向，无序数组只记不可变排序索引；查询保持轴dtype并精确核对候选，不逐点装箱成Python字典。混合类型坐标保留原精确键语义。
- Plot轴身份只用`AxisRef(domain, axis_id)`稳定key；label只用于显示，不进入
  semantic field identity。Scope在内存、PanelState与Figure recipe中都使用tagged
  `latest`或tagged typed coordinate value，文本坐标`"latest"`不得被当作控制字。
- Dataset固定由Repeat、Point、Cell-data三个同级`DomainSpec`组成；每个domain都直接拥有
  零到多个`AxisSpec`，而不是把一个逻辑axis拆进两份schema。`DomainSpec(shape, axes,
  axis_codes)`只有一个contract：`axis_codes=None`表示axis与dense physical dimensions逐位对应、
  映射由shape/stride隐式给出；显式axis-major codes则把physical carrier element映射到每个
  logical coordinate domain。Repeat与Point当前是扁平carrier，Cell-data保留连续dense tensor；
  两百万像素图不物化广播coordinate codes。`ValueSchema`拥有dtype、unit、validity与可选数值name，不拥有
  第四份axis容器。不得再并行保存逐row coordinate与另一份domain/mapping，或向Plot暴露
  同一Point axis的两种身份。domain是数据归属，Plot fate只属于PanelState；
  UI不得把role/fate写回Dataset truth。
- 三域标题不会因为domain没有具名axis就消失：该域数量显示1、轴名显示“—”，不虚构axis；普通数值/有效性计数规则不变。Manual editor和Panel共用同一格式化函数。
- 同一run/content revision不可代表不同内容；EventRef只表达causal publication，不代替content identity。
- Runtime保留的不可变数据可按原提交块组成精确快照；窗口/有限范围不是必须复制的整张数组。块引用与只读数值origin/shape矩阵共同描述一个布局，不另存tuple-origin副本，也不让每个消费者逐块重解schema。读取范围、坐标、validity和sigma属于同一快照，连续数组仅由确实需要它的消费者显式物化。Fate、Scope、Group、Facet与Reduction变化只重新投影已有快照，不释放或重建科学history lease。
- 存储块不等于公开科学snapshot。整体视图只有一份schema/ref；内部每块仅持不可变values、紧凑bool mask（或统一True/False）、可选sigma及共同的typed placement。来源/回放记录仍由Runtime唯一保留，显式请求某条event时才恢复该条public snapshot；不能为了显示一个范围给每条记录重复构造schema/ref/DataBlock。增量计算与传输可以按持有中的不可变plane身份复用，但必须连同裁剪布局核对；旧身份的owner必须持有到本次复用结束，不能用悬空id猜相同数据。
- Axis坐标与Domain映射各只有一份不可变数值表示；坐标内容在Axis首次创建校验，Domain只核关系与映射。连续记录时间可用明确的record origins与sample offsets表示，不展开Python逐样本坐标或另建通用表达式系统。内部传输复用typed结构及既有input token，不通过文件codec反复编码/解码；文件边界仍严格验证。
- 临时restriction可以借用不可变parent的strided view；Runtime接纳长期保留的publication时，values、validity和sigma只能保留各自声明范围的紧凑bytes，不因小ROI的base引用钉住整个父图。恰好自有连续bytes直接复用；需要压紧的复制在Plane锁外完成，不按数据大小、设备或Plot kind设置阈值。

### 3.2 Figure archive

- 一个writer、一个reader、一个format owner。
- RenderFrame的公共present准备完整data、边框、刻度、标题及overlay后才选择输出；compose=False只省去屏幕像素合成，不能省去场景组成。Facet边框/刻度/标题在最终axes几何确定后统一准备，屏幕compose与文件save消费同一份；Save不得依赖此前显示过一帧、另建Calibration修补路径或补一次无用屏幕draw。
- Figure reader直接返回metadata、NPZ成员和已经完整验证的typed datasets；typed成员与Dataset共享同一不可变buffer，消费者不再次decode或copy。初始Host配置使用同一configure事务，在第一张front之前应用viewport/selectors/focus/classifier/fit。纯文件导出由已有save worker直接使用同一PlotSession/MatplotlibRenderer，按最终导出DPI准备数据和artist，不创建无人观看的RasterPlotHost、屏幕front或返回假的accepted description；规范化配置仍由同一Session与Figure codec保存。先写NPZ，再绘制真实文件，不恢复不存在的屏幕。已有交互Host保存仍恢复原屏幕。
- FigureViewer开图只创建真正的Monitor A Host，不在C先画一遍来取配置；首个真实accept才从其SelectionSubject恢复交互并同步Port/PanelState的规范化target。新图成功前保留旧板，失败或Close清理候选；A沿普通live fit契约首帧求解并继续处理新数据，C的静态保存策略不复制到A。
- 编辑任意archive Dataset的数据只需要typed数据和已有recipe，不能先创建隐藏Host求fit/description；包括非默认Dataset。修改后的实际Preview才进入同一个A接受流程，纯数据草稿不保存第二份display description。
- Writer写入前规划全部member namespace并拒绝碰撞。
- Reader在解释内容前严格验证format、required members、shape、duplicates和non-finite metadata。Figure与Dataset archive的每个member都按其物理ZIP名（`<key>.npy`）读取，不用NpzFile按逻辑名的猜测查找：`signal`与`signal.npy`是两个合法key，各自读回各自的数组；同名重复entry是含糊的archive，拒绝而不选一个。
- 未知metadata类型拒绝，不自动字符串化。
- Layout的Logic authoring容器由Layout codec递归编码：内存中的rows/numeric tuple写成JSON array，mapping写成object；标量不被猜测或字符串化，未知类型仍由strict writer拒绝。不是Derive专属的保存分支。
- Figure只使用稳定`zlc.figure`格式，无数字版本；reader只接受当前完整grammar，其它root或缺失字段均loud拒绝。
- Figure NPZ是可重绘的数据真相，包含typed Dataset、exact PlotSpec、完整normalized parameters、overlay、viewport、selectors、facet focus、classifier、fit和exact causal lineage graph；PNG只是同stem preview。
- Figure archive保持标准NPZ，但compression由唯一writer逐member决定：小member及采样后至少节省20%的结构化数组使用Deflate；大而低收益的camera-noise member使用ZIP Stored。不得为不同Task/Viewer复制压缩策略，也不得花秒级CPU只换取少量体积。
- FigureViewer把archive内每个typed Dataset作为sealed Runtime signal发布，再与TaskConsole Live/Frozen复用同一个Panel/SelectionBridge/Plot host/configure与accepted `DisplayDescription.spec` contract；不得按array shape重新猜plot kind，也不得保留第二套static panel owner。默认panel恢复exact recipe，且保存的`kind + cell_kind`必须在第一次semantic vocabulary投影前一起进入Panel identity；不得先按schema默认cell kind建立表单，再把另一cell kind的accepted fates写回。Add Panel只创建空的fixed-kind `panel-N`，operator随后在Setting选择任意archive或由ROI/Fit产生的派生signal。SelectionBridge晚于静态host挂载时必须显式replay当前accepted FitEvent，使Fit参数进入Runtime，但不得重跑solver或重画。每个panel的Setting、可关闭Edit tab、Frozen snapshot/Refresh、Interaction、Direct producer和Save figure均复用TaskConsole现有owner；`panel_only`只隔离Task lifecycle chrome，绝不删减Panel Edit能力。没有真实producer时同一控件自然disabled，不建立Viewer特判。Viewer文件选择默认从当前workspace当天data目录开始。Lineage以root、event nodes和direct parent IDs保存；Viewer把内部event ID只用作引用，Logic页显示有意义的Logic run snapshot，Devices页按实际device聚合working point，Flow把每个exact Logic event和共享Device投影为唯一node及显式edge；Logic行与Flow node共用同一个label（重名时才带sequence与generation），Flow node据此指回Logic/Devices页的行。
- FigureViewer的手工数据入口只编辑Dataset的数据结构和值，不编辑device、Logic run record、既有lineage、axis role/fate、coordinate labels或coordinate frame。Data tab固定按`Dataset -> Axes -> Data`三个全宽Fluent frame纵向排列。Axes只提供原子Add/Edit/Delete以及name、length、unit、domain（Repeat/Point/Cell-data）和该axis的typed coordinate values；axis values以一行横向virtual table显示，列号就是axis index，自有横向scroll，不再另设Coordinates/Label表或草稿，没有上下排序或第二套grid类型。Data顶部复用Panel title的三domain shape/axis摘要，operator显式选择一个Rows axis与一个Columns axis，其余每个axis使用与Setting一致的Scope模式并按真实coordinate value选择；二维虚拟table的行列header只读显示两个展开axis的coordinate values，cell只编辑Dataset data，Tab/Shift+Tab连续提交并进入下一/上一cell，非编辑态方向键移动current cell。两张table的模型长度只改变内部scroll range，绝不能撑大Data tab或外层window。大数组只按可见cell读取，绝不为每个value创建QWidget、展开大slice choice、每次编辑reset model或复制整张字符串表；axis length/coordinates一次批量调整并同步values/validity/sigma。Apply只用`zlc_data`唯一constructor生成canonical immutable Dataset并由Viewer现有私有producer发布为sealed Runtime signal；仅编辑data或axis metadata时必须保留已有Repeat/Point的物理carrier与`axis_codes`，只有Add/Delete、跨domain移动或length改变这种显式topology edit才生成新的dense authored map。已有archive编辑形成新working copy且不原位改变已加载snapshot，未Save前明确标脏。保存仍调用唯一Figure writer，并用真实manual publication event追加system-owned manual-create/manual-edit lineage node；旧lineage/device settings原样保留，用户只填写manual note。
- Axis Delete是显式的数据裁剪：直接删除任意domain中的目标axis并保留其当前Scope coordinate对应的slice；若该axis正作Rows/Columns则使用它最后保存的Scope（默认首项）。最后一个Repeat axis也可删除，留下合法的单row无具名axis domain。改name/unit且不换domain必须保留已有科学role、labels与frame；换domain才采用新domain的generic role。
- Panel Edit不得嵌入第二份Logic Editor；Direct Producer从唯一Logic草稿与descriptor的selection mappings投影会联动的参数，复用原字段控件并禁用编辑，同时可打开/聚焦现有Logic tab。它不拥有草稿、Start/Restart或device choices；联动或Logic草稿变化只更新已打开面板的只读投影，不构造隐藏Editor或周期重建表单。正常首开只投影一次Editor Form；Editor host接受后只有accepted state或control surface真实变化才重放，关闭的Editor不得先构造projection再由View拒绝。
- Figure导入同时保留原`source`与原DAG，冻结与再次保存走同一次精确publication溯源；原DAG为空时不得制造import或旧Task执行事件。Manual Apply通过Runtime的精确parent publication记录真实修改，不手工再拼一份来源链；普通Panel Save与Manual Save使用同一份冻结来源。Logic/Devices/Pulse按存在的source事实投影，不按Manual/ROI节点类型决定是否显示，已在DAG中的相同记录不重复。
- Measurement worker若显式消费一个Dataset signal，必须在取出值的同一时刻把该exact source publication交给Runtime commit；Runtime是direct parent edge唯一owner。Scan不得只保留`SignalValue`后丢弃publication identity，也不得在Figure Save/Viewer中按`source_signal`反查latest补边。FigureViewer Flow只表达archive中真实的causal parent edge与Device-use edge，不重复parameters或device snapshot详情；这些分别由Logic与Devices页显示。Devices页用run record的stable role→instance mapping解释run/event record中的`device_snapshots`，同时读取`actual_devices`和lineage顶层仅对实际引用epoch展开的active override；不得猜`role == device key`，也不得拿override为空解释成run未使用device。Task生成而非Panel Save生成的normal/partial report Figure在source中保存该Task已经冻结的run record；Viewer可据此显示单个Task及其Device，但不得伪造Runtime event DAG。
- Dataset/Figure encoder只写caller-owned binary IO；路径原子发布唯一属于`zlc_durable`。

### 3.3 Durable paths

- durable_mkdir确认目录与父目录项；已存在目录也可能是上次父flush失败后留下的，不能以exists代替确认。durable_makedirs只确认最近的existing anchor及其父项，再逐层创建缺失目录，不重刷完整祖先链。flush失败明确报告已可见published路径；再次Start/Save必须能够完成此前未完成的确认。文件自己的fsync、原子发布及父目录flush保持不变。

- Unique name allocation与commit构成一个并发原子操作，多process不得取得同一目标。
- Atomic replace失败后的outcome必须诚实，不把可能已写入伪装成旧状态。
- 不新增content hash；使用run identity、受控path、shape/size和完成状态记录artifact集合。

## 4. Canonical Runtime Live Contract

### 4.1 One run-data owner

Logic Node只提交本次新增chunk/event；Runtime按run和signal identity累计唯一canonical dataset：

```text
Node new chunk
  -> Runtime append/commit
  -> immutable event view / declared run or window view -> Processor
  -> canonical current view -> Signal description / Panel / Edit / Save /
                               Selector / Overlay / display derivation
  -> retained partial seal
  -> final seal
```

- Camera、Scan、Calibration和Task preview不得自建parallel slot/history/terminal truth。
- Camera使用chunked append，避免每次复制全部历史；Scan按固定point geometry增长。
- 未写位置invalid；coverage只描述实际写入extent。
- Finite exact signal的event view用于commit及默认的逐event Processor输入；Processor显式声明的Input range由Runtime在同一exact publication上选择，不能由delivery policy或coverage猜测。所有UI/display consumer必须使用同一publication对应的canonical current view，从第一次publication起报告完整authored physical shape，未来位置invalid。
- 普通Monitor signal没有finite canonical extent，UI显示latest complete event；Processor不得仅因“derived”就增加科学轴。输出契约的`index_by_source`只声明history能力；只有真实consumer按window取得lease后，Runtime才从当时的current event开始建立带通用`primary-index`的bounded ordinary Dataset，并按全部active leases的最大window保留、在最后一个lease释放时立即归零。Runtime内部以绝对source ordinal排序、保留gap；materialized Dataset只暴露相对最新事件的普通整数坐标，最新固定为`0`、过去为`-1/-2/...`、缺失offset为invalid。lease之前的event不回填、不伪造；Runtime是唯一跨publication history owner，所有Plot读取同一Dataset且不建立Plot-kind、Fit或Workbench专用history lane。materialized history的点表布局（shot 序列、每 shot 的 event 行数、随窗口滑动不变的 event）只由`zlc_data.snapshot_projection.indexed_history_layout`读一次并缓存在schema上：窗口选行掩码、rolling 的 shot 编码与 source index、`indexed_schemas_compatible`、标题里的 shot 计数全部读这一个对象，任何消费者不得再逐行扫描 primary-index 列或用对象数组做成员判断；不合契约的 shot index（正offset即绝对ordinal、乱序或非整数offset）直接拒绝而不是宽松读取。对该Dataset的任何restriction（Scope到过去的shot、显式window）保留源相对坐标：只留过去shot时最后一个offset为负，仍是同一history的合法裁剪，layout reader照常读取而不把它当坏producer；跨record样本选区可以留下不同数量的行，统一layout保留真实row codes，不重造矩形shot表。显式window只按该普通轴的相对坐标选择最后N个cell，不形成第二份history；除此之外Plot/Workbench不得识别primary-index为history或自动增加Latest scope，它与其他AxisRef使用完全相同的fate、selector、focus和viewport规则。history event/indexed表示切换即推进该signal的presentation epoch并使全部consumer重新投影，即使scientific publication未变；该epoch不冒充run generation或content revision。Occupancy exact处理每个camera cycle，但其公开Monitor几何仍是当前cycle的`frame`，processor自身不得在published geometry上再叠一层source index；这不禁止它像其他Monitor输出一样`index_by_source`声明history能力——那条history仍由Runtime在lease成立后用通用`primary-index`单独建立，几何不变。
- 信号的通用运算与衍生只有一个processor：`derive`。每行一个输出，由`Name`与可多行的`Python code`组成，Add逐行加入、行尾×移除；草稿仍保存普通`rows`（`name/code`），不另造程序格式。单个表达式走普通Python `eval`，多语句走`exec`并以`result`给出这一行的答案；`np`、同一输入bundle的`a.<signal>`和前面已命名输出可用，中间变量不发布。输出名、代码与Input range写入run record和存档lineage；descriptor按合法输出名声明`@logic/<node>/<name>`（contract `derive.<name>`），草稿检查名字和Python语法，不预先在schema上解释或限制一套DSL。代码在原Processor worker执行，普通执行异常按输出名报告；这不是安全沙箱，不保证中断无限循环或危险的原生调用。
- `Operand`只是完整`DatasetSchema + values + validity`的薄数值包装，始终保留Repeat × Point × Cell-data三domain，不按数组shape猜轴、不自动squeeze。`isel`按索引、`sel`按精确typed coordinate选择任意具名axis；标量选择移除该具名axis但保留所属domain，长度1列表保留该axis，无具名axis的domain仍有大小1。`mean/sum/count/any/all/min/max/std`可沿一个或多个任意domain中的具名axis归约，歧义用完整AxisId消除，轴名与ID区分大小写、示例须按实际输入轴名替换；`where`只限制validity，空组invalid，`std`为总体标准差（ddof=0），并为每个输出bucket选取该bucket自己的真实样本作锚点，再以两遍局部残差形成均值与平方和；不得用跨bucket全局中心或`E[x²]-E[x]²`，常量必须严格为零而大基线上的可分辨小spread必须保留。bool归约使用`count/any/all`，其`count`数有效True、numeric的`count`数有效样本。算术/比较/布尔运算要求相同几何并沿公共单位规则，不静默对齐或做笛卡尔积；原始NumPy可通过`.values/.valid/.masked`使用，同shape替换走`.with_values(...)`，高级变形必须显式构造带完整schema的`Operand`，不能直接发布一个猜不出domain的ndarray。
- Processor的Input range由Node的`dataset_input_view/dataset_input_window`声明，独立于exact/latest交付策略：`event`为当前publication的原始event；`run`为该publication对应的本次运行canonical Dataset，finite源包含尚未采集的invalid位置，且不受其它Panel的history lease影响；普通Monitor没有finite累计范围时，`run`就是它当前完整结果，不凭此无限积累。`window`为Runtime原有`index_by_source`能力下最近N个publication位置的bounded Dataset，沿原source ordinal保留gap/invalid与相对`primary-index`，不按Repeat长度猜事件数或只数成功样本；不支持history的源（包括无此能力的finite源）明确拒绝并提示使用`run`。Derive默认`event`、window默认50；显式范围在live与terminal求值一致，未声明该属性的既有Processor仍保持live-event/terminal-canonical规则。
- Input window的lease只在NodeHost运行期间对同一bundle所需成员一起取得，按同一exact publication与共同可保留窗口物化，不能拼出不同起点的siblings；开始前不可恢复的旧event不回填，结束、Stop、失败或关闭释放自身lease，不影响其它consumer的lease。Runtime继续独占history、event record与causal publication；Derive不存输入历史、不建第二份placement表。每次Derive求值的所有输出都是完整当前结果，以`MonitorCoverage`替换上次估计，不把任意归约结果按输入finite位置append，也不继承未归约输入的50-repeat geometry。输出保留本次实际parent与source primary index，声明`index_by_source`仅供真实下游按需取得history；终态保留完整结果并seal。
- History唯一保留上限是所有active有限正整数lease的最大window，不另以隐藏字节预算或固定shot数缩短用户请求。gap仍占其实际index位置并为invalid，最后lease释放立即归零；内存随真实requested window和retained buffers增长，不承诺任意大window均能装入物理内存。
- `scope/reduction/fate`只决定怎样投影Panel已经取得的canonical view，绝不替代Processor的Input range或决定读取event；同一publication不能因Panel semantic不同代表两份不同数据truth。
- Incremental placement沿Repeat/Point两个`DomainSpec`的物理rows；多维scan由Point domain内多个
  logical axes及其唯一`axis_codes`表达，不另存平行row/topology结构。一个cell payload
  原子完整发布，不新增cell-internal tile/slice streaming contract。
- Canonical display materialization只在实际display consumer到期时合并/cache，并在Qt owner thread之外执行；不得让producer每commit强制复制full prefix，也不得因Panel存在与否改变采集结果。
- Seal只结束写入并保留chunks、coverage和EOS，不替无人读取的finite信号预先物化完整数组。Processor Start只做元数据/输入能力与owner admission；真正的event/run/window输入在既有Processor worker内取得，终态输入同样如此，不在Start调用线程预取后丢弃。
- Live Panel、Panel Edit/Refresh/Save、selector、fit input和overlay必须从同一accepted canonical presentation snapshot投影；无法唯一对齐即拒绝。accepted surface的数据身份必须与其pixels一致：host持有什么由host自己的front回答（consumer Future被取消不等于host没有提交），port记住每次交给host的exact publication/PlotInput直到屏幕越过它；configure的front只能落在其数据身份命名的记录上（当前surface或已交付未呈现的render），身份未知时owe一次presentation pass而不是给旧surface换description；host已持有的revision再次offer时按host当前front重呈现而不是逐拍取消。
- 已选择共同显示的关联signal构成same-shot group，只有全部成员具有同一shot的publication才整体呈现。首发、Processor重启或本轮尚在计算都属于pending，不得把缺失成员排除后先呈现新图像；保留上一组完整accepted画面并显示等待状态。同shot已发布但validity为invalid的结果仍算本轮完成，只是不画对应标记，不得沿用旧判断。用户明确断开成员后才改变组的需求。
- 一个leaf的因果DAG可包含同名信号的旧世代：若存在唯一后代覆盖所有同名祖先，可见值取该后代，祖先继续保留在lineage及root计算。互不为祖先的同名分支仍冲突；不同可见leaves合并仍严格要求同一EventRef，不能借历史关系把旧overlay与新image混显示。
- Bounded indexed history已经淘汰的旧publication属于正常presentation过期，不是signal/Panel故障：`SignalDataPlane.retains(signal, publication)`必须把primary index早于history first index回答为False，materialization以明确的expired/cancel结果拒绝；Surface丢弃这次排队更新并保留上一完整front、host和control vocabulary。Frozen Edit仍以自身accepted snapshot完成selector/fit/producer映射，但不得把已淘汰parent交给Runtime SelectionBridge；它同时撤下该bridge之前的derived output，绝不能拿latest冒充旧publication或留下旧ROI信号，也不能让普通`ValueError`关闭host或清空Fit/Setting UI。
- Panel Edit的冻结数据是否落后于Live与冻结配置是否仍兼容是两个状态：同run内accepted surface所示Dataset revision推进（scan逐点增长或Monitor发布新值，比较的是exact revision ref而不是publication上不存在的coverage）只标记`data advanced`，不得阻止对exact frozen snapshot做Fit、Refresh或Save；只有signal/spec/axis vocabulary真正不兼容才阻止保存。Refresh只是请求：新的冻结候选留在editor entry里随Edit host一起交付，公开frozen record、`data advanced`与Save读取的都是Editor最后接受的完整画面，Editor接受新front时二者才一起推进；没有Edit surface时才直接冻结accepted Live。接受Display/Fit/Focus等配置时，PanelState、frozen target和两surface配置必须原子推进，不能先把自己标stale再由stale阻断同步。
- Panel的title shape与Setting semantic都只能读取同一publication的canonical current Dataset，不得读取最后event chunk冒充完整signal。title结构固定为`(repeat axes) × (point axes) × (cell-data axes)`三组；例如Survival field scan显示`(20) × (3×10×10×10) × (35)`：survival的pair是Point domain内的READOUT_EVENT axis，scan axes在同一Point domain按声明顺序追加，site留在Cell-data。PanelCard以独立的accepted-data projection持有title structure/scope，不从Setting parameter surface读取；每次surface accept都直接更新该projection，即使不改变任何PanelState/control vocabulary。存在具名Point axes时不得再发明flattened `point` ordinal；多维FacetGrid默认facet最外层真实scan axis，其余轴保持可编辑的Reduced。即使当前projection因FacetGrid 64-cell上限等原因拒绝，错误只标记不可用的presentation/fit，完整canonical scan-axis fate仍必须留在Setting中供operator修复。live publication未改变PanelState或authoring字段域时不得reconcile Setting form；Plot kind是Add Panel时确定的panel identity，不进入Setting通用表单，FacetGrid仅暴露可变的Cell kind。
- 未经authoring的默认投影只由`zlc_plot._kinds.defaults`一张表决定：每种plot kind的`default_spec`、FacetGrid的cell kind选择和「从当前plot要一个grid」都是对同一张表的读取，不得各自维护第二套推断。表按`classify_axes`得到的axis family分组，永不按axis name特判：R（Repeat domain axes）是统计量，只被reduce或被Histogram pool，只要还有别的轴有结构就不成为layout轴；H（Runtime的`primary-index`）同样是统计量，除Rolling自己走它之外只在其它轴都没有结构时作curve最后的x；S（Point domain中的scan axis，slowest first）是位置——最内层是一次sweep走的x，两层是heatmap，最外层是grid的facet，无人认领的scan轴保持可编辑的Reduced；E（Point domain中的`READOUT_EVENT`，如camera frame、survival pair）是子测量的选择——grid给每个event一个cell，无scan的curve沿它走，其它情况在构造默认spec时选择末项真实坐标的Scope，不得对不同frame取平均；D（Cell-data axes）是内容——声明的picture或两条content轴成image，剩下的一条content在palette能分辨时成group，否则reduce。size为1的degenerate轴是provenance不是结构：curve默认的x若只剩degenerate轴可选，则改走声明顺序里下一条有变化的轴（此时只可能是Repeat轴——别的family有变化的轴表已先选走），全都无变化才把首条degenerate轴画成一个点，一条轴都没有则该kind无默认（返回None，入口按「无法绘制」拒绝而不是抛异常）。这条规则只在这张表里，Workbench的`fitting_panel_spec`只定kind/cell kind，不得再改选任何轴。`packages/zlc_plot/tests/test_default_roles.py`枚举全表。Limit类display字段（relim与x/y/color范围）声明为non-portable：panel identity改变时它们随semantic/fit一起从新vocabulary重新开始，只有外观字段跨kind携带。
- Panel host accept后的Setting/PanelState metadata必须读取该次accepted surface的完整Dataset描述（含已物化窗口轴），不得再用原始event/canonical信号目录字段覆盖。首次挂载与普通更新共用端口呈现通知并接纳归一化target，不得只更新Console一侧而留下旧port target。一个表单提交的全部fate rows是一次原子axis-role assignment：先确定全部x/y/group/facet目标，再处理Reduced/Pooled/scope及真正空缺的required role；结果不得依赖row迭代顺序或中间冲突。
- Fate vocabulary只由schema与plot kind声明：每个axis无条件列出该kind拥有的全部roles，UI不得运行candidate projection、cell-count、surface size、DPR或layout feasibility来删除选项。实际组合是否合法、Facet是否超过容量只由提交后的Plot/layout transaction判断并loud拒绝；拒绝不得改写或缩减Setting vocabulary。这条法则覆盖表单全部section：**Setting表单的字段集合只由panel身份（signal/kind/cell kind解出的schema与描述）决定，数据与生命周期状态只能改字段的值、可用性与注记，永不能增删字段**——「标记不可用」不等于「移除字段」；host描述过的panel在任何degrade/报错路径都必须从其accepted描述重投完整字段集（含Fit），schema投影只服务从未描述过的panel。停止的run不是例外：sealed monitor publication由plane无条件保留（终态保留策略唯一属plane，节点不再逐个声明），其上的ROI/fit派生走terminal路径照常工作，「run no longer held」只在硬retire后出现且属于自清除的condition通道而非error。
- Panel surface在`board.commit`中首次accepted后，Derivation Bridge必须在同一个owner turn完成level reconcile，之后才能把交互权交回Qt；不得出现像素已可点但首个selector尚无Bridge而永久丢失的display-cadence窗口。
- Layout（`zlc.console-board`）恢复参数按当前声明取交集：存在的字段填保存值，缺失用当前默认，已不存在的字段忽略；Logic读取当前AuthoringSchema，Panel读取当前PanelState/Plot词汇，不为某两个废弃参数加兼容分支或版本表。字段本身的类型/数值校验仍保留；未完成的业务草稿可载入，完整性由原Start入口判断。动态semantic在真实schema可用时通过同一PanelState投影筛除不存在的键，不猜旧轴名称或翻译旧角色；底层Plot参数API和数据文件校验不放宽。结构身份仍明确：每个panel必须保存唯一非空`panel_id`与可识别kind，Logic必须可定位当前插件，不能按行号猜身份。Load先为本次加载铸造新panel id，再统一重映射`@logic/<保存的id>/...`；无法解析的连接置空并在status说明。Load/Clear仍是整板替换的同一commit，有运行节点时先确认、停止并等terminal，不从UI强关host。程序化`apply_layout`未授权`stop_running`仍拒绝覆盖运行节点；读取不自动改写原文件，保存仅写当前有效参数。
- Plot live revision identity始终读取底层Dataset snapshot的`stream_generation + revision`；`ImageFrame`只是overlay wrapper，不能隐藏新generation并把重置后的相同revision误判成stale。same-geometry新run复用host及交互订阅，geometry变化才replacement。
- Occupancy的SITE是每个`(repeat, point)` cell内原子完整的data axis；overlay不得另存site history。Occupancy只发布通用bool/numeric status signal，点是否可读由该Dataset自身validity表达；XY geometry与adapter contract由`zlc_plot`中立层拥有，Workbench只按contract路由signal且不得import Occupancy。动态status与图像都取其exact publication的canonical prefix，再按公共Scope/Last与facet定位一个Repeat/Point cell；site向量不能被图像pixel fate裁掉。多个cells的Mean或pool没有独立Boolean判决，不私造共识状态，无法唯一选定时显示UNKNOWN/隐藏。
- Processor可声明消费订阅锚点同一原子publication内的具名siblings；它们不形成独立source或异步join，exact replay与causal lineage由该publication提供。Derive用直接`a.<name>`引用规划所需siblings（加上订阅锚点）；动态使用`a`时保留该producer的全部成员，AST这里只规划依赖、不限制Python语法。所有成员统一使用同一个Input range，不能为其中一项读latest或另一份窗口；不支持跨独立producer的隐式join。原`occupancy_agreement`已退役，一致性掩码只是普通Python，例如`agree = a.occupied.isel(frame=0) == a.occupied.isel(frame=2)`，再以`.where(agree)`限制选定frame的counts；也可继续归约Repeat或其它命名轴，不限定为site-only操作，不重读camera/calibration或重新分类。
- Dataset输入可声明`select_bundle`：其界面按同一atomic producer的输出集合提供一项普通Fluent choice，显示producer和成员名，不让操作者在bundle内部选某个叶子。Runtime仍用一个成员signal作为既有订阅锚点，按上述依赖规则消费同publication的siblings，不引入bundle数据或另一份registry；不同atomic owner即使共用Panel标题也不得混成一个bundle。Derive使用这个入口，普通单signal输入继续使用现有树形选择器。
- UI freeze只读取已提交状态，不调用plugin materializer。
- Logic草稿的Start admission问题是待填写/修正提示，不是节点运行失败；保留原草稿且禁用Start，按用户可见field label与嵌套row位置说明，状态显示中性的`Draft: …`。只有明确Start失败、run error或既有draft_error才进入红色error状态。Derive所有控件、只读结构说明与短示例均使用英文。
- Stop/Final不受Panel、freeze或Processor订阅影响。

### 4.2 Identity与processors

- Occupancy对每个Repeat/Point cell的图像独立应用校准，只检查SPATIAL_Y/X图像及校准尺寸，不限制Point具名轴数量；frame、scan及其它领先轴连同codes/坐标/单位原样传递，只有图像Cell-data转为site判决/计数。Live仍只处理event，terminal处理完整保留数据。
- Frame Survival按唯一READOUT_EVENT定位frame轴，在每组其它Point坐标内部做forward pair；只把frame替换为pair，保留scan轴，不跨扫描点配对。finite coverage与placement按同一canonical frame-row映射转成pair-row，不把所有事件写到point origin 0。
- 不变的event/canonical frame拓扑各在现有Processor中规划一次；placement只定位本event覆盖的有序group范围，不每shot扫描整份canonical frame表。
- SignalDescription持有不可变canonical schema，physical shape由该schema派生。Logic/Panel Outputs与Source列表从Plot公共schema_structure读取三domain轴大小，只显示三组数字维度，不添加轴名或单位；长度1的轴也保留乘1，不再只打印扁平Point carrier长度。原signal name和Panel标题的轴名行不变，UI仍只接收投影后的文本。
- Signal目录只随其结构、生命周期与parent generation事实改变；值revision不属于菜单事实。Plane保留唯一不可变目录，Console从这一份目录产生并复用rows与overlay候选；无变更的idle/new-value拍不重建菜单。菜单家族匹配不是same-shot许可，实际呈现仍核exact publication。

- Scan数据的两条Repeat轴name与任务UI对齐为`repeat`、`shots per point`，不以Pulse硬件的Scan/Run repeats命名。轴顺序、坐标与稳定AxisId不随manual/device/pulse执行方式改变；Pulse本身的参数和执行字段保持原意。
- Generation标识一次run/restart；generation内schema和stream generation固定。
- Restart只等两件事：旧run真正停下（device lease随worker终态释放），以及仍要从Plane读旧generation的面板投影（含正在retarget的候选port）跑完。投影读完后surface拿着自己的副本走，渲染子进程里的绘制、cohort的成组、屏幕翻转都不参与等待；卡片说明在等哪一件。终态过渡无条件：worker完成一旦被取走，即使退休Plane generation失败，run也以failed结束并说明原因；display步骤失败不跳过节点轮询。
- Panel标题仅Repeat域显示条件写入数量，Point/Cell-data仍显示完整维度。对每个Repeat轴只放开自身，其余Repeat/Point轴固定于同一publication最后写入位置的canonical axis_codes；只数该切面已写入的不同Repeat坐标。Cell-data整块发布，site/pixel没有用于标题的当前坐标，科学validity（例如Survival分母资格）不能充当写入覆盖。Runtime复用唯一occupied_cells覆盖记录，在commit时把包括本次block的计数作为SignalValue.repeat_counts固定下来；Monitor的完整事件按实际Repeat carrier计数。旧publication、Stop后的数据与Frozen继续持有自己的整数，不读独立latest，不扫描像素或累计第二份进度。Panel只投影exact accepted publication的计数，不受Scope/Focus/Facet/Reduction影响，不产生min–max区间；计数之积不表示全Dataset样本总数。
- Producer与latest/frozen/follow Processor的Start共用同一终态世代交接：旧结果在结束/Shutdown后仍保留，直到下一次Start才退休旧owner及派生closure。cleanup在Plane锁外，最终source exact校验与新state安装在同一锁内，`_starting`由同一入口释放；仍active的owner不得被覆盖。不得用关闭时清数据或为Derive另建重启路径绕过。
- Revision严格递增，不接受重复、倒退或同ref不同内容。
- Selection revision属于用户的数值范围选择，不属于source generation。相同revision、相同数值几何和同source generation是幂等提交；旧revision或同revision改真实范围仍拒绝。兼容选区遇到新的accepted source generation时在原Bridge重新激活派生，不伪造一次用户编辑。仅视觉用的drawn不触发数值重算，Panel文档不另存revision，交回Bridge始终用原binding.selection_revision。
- 一次commit的siblings共享revision、run record和causal parent。
- 成功Dataset commit的通知只由Plane的publication订阅发出，worker与processor遵守同一入口；NodeHost不再为同一commit额外唤醒。进度、完成、交互及渲染完成仍沿各自原通知，不能为省唤醒把已到deadline的新数据强制推迟到下一display beat。Indexed history在原子commit预检一次，更新保留窗口不重复执行相同预检；lease变化仍在更新前校验。
- Run record属于generation声明，不属于每event数据：实际准备完成后只注册/冻结一次，后续commit引用Runtime拥有的不可变记录，不携整张计划反复比较。event record才携本事件变化的时间/设备事实，每次atomic commit只冻结一次；内部siblings/publication复用同一记录。finite物化共享已冻结条目，按需生成精确prefix记录，indexed只保留窗口范围。仅更换DataBlock身份不重扫未改变的数值内容。
- Finite保留在原generation owner内直接持有已验证的数值bytes与紧凑逐event事实，不逐shot常驻SignalValue/Snapshot/DataBlock及祖先Publication对象树。Boolean数值与mask无损按位保存、公开读取时仍恢复不可变数组；其它完整连续数值plane复用原不可变byte owner，不额外复制相机图像或sigma。祖先run声明共享，事件只记录其序号、parent关系和实际event record；相同非空record按内容共享，空record不建独立对象。普通数组物化不解码祖先；Exact replay按需重建公共对象，仍保留全部identity、validity、sigma、placement与来源事实。待消费replay只持所选signal的数据，不钉住退休node、其它siblings或整份materialized Dataset；已持有的真实parent payload仍沿原弱引用owner解析。内部紧凑编码不是新增磁盘格式。
- Boolean bytes直接存入同一append-only事件buffer，不逐plane留独立bytes头；materialization只在锁内截取已提交前缀的引用和范围，逐event拷贝/解码在锁外，不能因大pixel mask扩大公共锁内复制。来自本generation已有commit的parent复用其sequence/offset，不把过去整条祖先链在每个新event重复展开；读取用显式栈，不新增递归深度限制或永久EventRef索引。
- 数组读取与来源记录读取按真实需求分开、共用同一materialization owner：current_dataset只准备snapshot，不构造随即丢弃的记录；current_dataset_view才准备完整exact记录，并复用同一atomic bundle、同revision/范围的结果。公共锁只截取不可变输入及接纳结果，不合并时间索引或计算时间坐标；需要完整新平坦记录时的线性输出成本放在锁外，不能把“只增量放置数组”误称为所有metadata工作都O(增量)。
- 接收队列、未消费exact事件、用户科学history、Frozen快照和溯源元数据各有明确寿命；溯源的event identity/run record/parent关系不授予无限保留祖先像素的权力。当前科学求值、same-shot front和Frozen仍持所需真实payload；有限结果只永久保留自身数据及祖先元数据，回放按原chunks顺序按需读取，不在公共锁内构造全历史临时队列。普通Monitor不构造无消费者的finite replay副本。
- Exact订阅的live待处理队列同时按事件数与payload字节明确限额，默认1024事件、128MiB，可在订阅时显式调整；这是未消费缓冲而非history上限。满额或单事件超额只使该订阅明确失败，不丢旧/取latest、不阻塞其它消费者。原生不可暂停数据源不能承诺无限积压仍无损；有界故障必须携容量和积压信息。
- 运行后续失败与此前已commit数据有效性分开：保留已验证的partial prefix，同时向订阅者传播失败终态，不伪装完整或正常EOS。Stop继续保留数据；显式Remove/Clear释放被移除owner的Runtime保留，仍由实际Frozen/Save持有的不可变值不受影响。
- Exact scientific Processor逐publication有序处理；pure display derivation可latest。交付策略由input contract声明，不从coverage猜；同一交付publication的event/run/window输入范围是另一项显式选择，exact并不强制只读event chunk。
- 不同Processor可并发，同一Processor保持有序。
- Processor的Start是持续跟随意图，只有操作者Stop才取消；源暂时不满足计算条件（如Frame Survival少于两帧）保留原因并等待，不伪造结果。失败的同一源generation不反复执行；新generation有publication后复用原Start重新接入，包含已经sealed的新数据。实际尝试的输入publication由NodeHost唯一持有，Workbench不另存重试世代或判断插件帧数。

### 4.3 Logic Node contract

- Waveform Measurement消费adapter原生record FIFO，每record发布一次，保存全部sample；不按主机Read interval抽取latest或拼成另一种shot。记录数决定finite终点，Receive buffer只决定未消费记录可容纳多久，不是采样率/绘图刷新率。adapter arm成功必须已开始接收或真实启动硬件；Stop必须停止该adapter负责的采集，N100本身的持续广播不因停止一次capture被伪称关闭。公共队列满或capture期CRC/流水号/时间缺口明确失败，不覆盖旧记录。
- Waveform输出按声明的columns选择数据；连续列使用原不可变record的slice/transpose视图，非连续列才gather，最后都沿DataBlock同一不可变所有权入口，不先为连续内存而重复复制。协议CRC使用等价标准库原生算法时保留原多项式、初值与坏包拒绝，不以取消校验提速。
- Waveform time_basis只取device_clock、sample_clock或host_receive；设备运行时钟、采样计数推得时间与主机接收时刻不能互相冒充，started_at_ns不是设备时间到UTC的校准。每条record的原时钟/相对run时间/主机收到时间写入Runtime event_record.record_timing，物化只合并被选记录，不另建Measurement历史。
- 普通Stop先停源，再按原commit路径发布已经进入接收队列的完整record，最后封存；失败时不继续发布，清理失败不得遮蔽原始错误。finite记录数与Receive buffer容量互不替代；Monitor不无限积累，只保留既有consumer明确请求的窗口。
- AxisSpec.coordinate_of声明同一domain内同一物理行的替代坐标；主轴/替代坐标共用row codes而不形成笛卡尔积或重复fate。Setting以一行fate和坐标选择呈现，Scope切坐标按同一位置转换，Figure保存选择。连续源的SAMPLE_TIME位于Point，Runtime按记录时刻加样本偏移物化真实采样时间轴；Plot、选择、Fit与存档使用同一Dataset坐标，不在renderer临时变换科学时间。触发源的record time保留独立波形首样本相对时间；scope未知触发偏移时不得声称零点是trigger。
- 替代坐标只属于有显式行映射的Repeat/Point，不改变Cell-data的dense contract。Derive的scalar选择/归约移除整个物理坐标组，list选择同步全部坐标；Manual不把替代坐标展开成新维度，物理长度/域移动/删除同步整组，名称/单位/坐标值只编辑目标坐标。带替代坐标的整组移入Cell-data明确拒绝，不静默丢时间或拆成独立维度。
- Curve（含Facet Curve）与Histogram的Window默认1，复用同一Runtime history lease；Rolling仍默认100。保留多少记录与如何投影是两件事。Plot、Fit、selector和Runtime派生读取相同window；不得因其它Panel请求更长窗口而扩大本Panel的统计。时间横轴使用本run相对首record的真实时间，Rolling不再把最新时刻暗改为零；source index坐标仍以最新为0。
- Rolling的window就是本Panel请求和使用的历史范围；trailing mean不改变Runtime lease、不额外保留窗外记录。先在公共snapshot限制入口截取本window，再归约和平滑，绘制/Fit/selector/side histogram共用同一输入；其它消费者保留更长历史不得改变本窗结果。静态Repeat物理记录与live indexed记录遵守同一范围语义，不猜某个逻辑轴充当物理行；左端不足trailing时沿既有部分平均及有效样本数规则，gap仍占原记录位置，不额外补采。
- Rolling的横轴固定为记录序列，Index/Time只是这条轴的坐标表示，Setting只给X coordinate选择，不给记录族Fate，也不让其它轴抢占X；独立轴仍可Group/Scope/Reduce。数值入口在展开任何样本前拒绝记录轴及其别名重复Group，不能形成record×record平方桶。普通图种的坐标切换只重映射已有Fate，不偷偷分配X；完整坐标/Fate表一次性合成，拒绝态仍可修改恢复。

- Measurement必须在bounded cadence内live commit。
- Task必须发布progress与声明preview，或显式声明无preview。
- 第一份真实publication前不显示live；terminal清除progress并seal/retire preview。
- Measurement可显式声明`reports_ready`，并在worker真正完成设备准备后通过现有ExecutionContext报告本run ready；NodeHost独占该事实与等待，running、progress文字和首publication不能代替ready。Camera在完整arm返回后报告；未触发时仍可ready。
- Descriptor outputs、runtime declarations和preview references只有一份typed vocabulary。
- Concrete Logic Node不得要求Workbench识别其模块、output spelling或domain helper；通用显示/overlay/selection能力由中立层contract表达，Workbench只路由contract。
- 通用discovery test必须走真实NodeHost、SignalDataPlane和preview contract。
- Hosted Task需要人工决定时，只能通过NodeHost同时公开一个带唯一request identity的operator-input request并在worker内等待；Workbench按request kind提供交互，response必须精确匹配当前request。Stop唤醒并取消等待，不使用轮询、第二条Task lifecycle或plugin-specific Workbench状态。未来人工Scan axis复用这个lifecycle，但其业务与UI不属于Calibration实现。

### 4.4 TaskRun与durable artifacts

- 只有实际`Start`进入NodeHost worker时才分配唯一run directory；打开Editor、draft validation或build failure不得留下空run。
- 每个run只写两份记录，各自原子建立一次、从不替换：Start时在任何不可逆工作前写不可变的`start.json`（run identity、normalized inputs、started_at）；结束时写`run.json`（terminal状态、stop reason、last progress、artifact inventory、failure）。两者之间不写盘——progress与artifact registration只在进程内；每次都重写记录意味着一次长Calibration上百次fsync与`os.replace`落在别的句柄可能持有的路径上，Windows上就是最后一次写的PermissionError，而运行期间没有任何读者。
- Task只保存由domain owner挑选的重要、可复算或不可替代artifact；Runtime不得自动dump live Dataset、全部shot或所有中间状态。
- Artifact必须先完整原子写入run directory，再按semantic contract注册；`run.json`只列已存在、已注册的文件。声明的final artifact未注册时Task不得成功。
- Run根保存`run.json`与summary；domain final进入`final/`，重要图进入`figures/`，精选candidate/site数据进入`data/`。
- Figure始终成对保存：同stem `zlc.figure` NPZ为primary data artifact，PNG为无science contract的preview。Calibration和SLM Feedback遵守同一TaskRun规则。
- Stop保留已完成的精选artifact和明确partial状态；failure保留错误、last progress、已注册artifact与rollback outcome。进程异常终止只留下`start.json`而没有`run.json`——这就是「没有结束的run」，不得清理或伪装成功。

### 4.5 Task运行中冻结

冻结：Add Logic Node、当前Node source/preview signal、overlay binding、scope/reduction/fate以及冲突硬件配置。

允许：其它Panel、当前Panel的样式/viewport等纯显示参数。Calibration继续显示long/readout/long三帧Grid/Figure。

## 5. Plot、Fit、Overlay与Selector

### 5.1 Exact Data/Fit pairing

- 数值名称由生产该量的owner写入`ValueSchema.name`，不是Runtime地址或Plot猜测。普通named snapshot使用信号短名；Occupancy/FrameSurvival与用户Derive输出声明各自的新量名，Fit参数与ROI统计使用输出名。仅增加扫描/历史维度或裁剪ROI时保留输入value schema/name，Seamless接survival仍标survival而不是scan。Plot从同一QuantityArray.label读取：Curve/Rolling的Y、Histogram的X、Image/Heatmap/3D的色条或Z均用值名＋单位；Histogram的计数/密度轴仍由统计定义命名。显式Plot标签优先，真正无名Dataset才使用value占位。名称与Dataset/Figure同存同读，不另加标签注册表、来源追踪或图种/节点特判；Manual编辑已有Dataset保留值名，新建无名数据用其作者名称。
- Facet的layout只拥有title，轴/值的显式标签由现有cell.labels唯一保存。构造时将外层显式x/y/value优先规范到cell后清外层对应项；renderer、Focus、label carry均读同一semantic cell，不在每次渲染时重建spec。换轴删除失去角色的旧标签，不能被旧cell fallback重新带回。
- Pulse字段显示由field_label统一读取`period.name or period_id`：duration为`MOT.duration`，DAC为`MOT.<port label>`，delay为`<port label>.delay`。同一Pulse的有效Period显示名必须唯一，空Name允许使用唯一ID；错误输入只提示并恢复已有字段显示，不改执行程序。ScanColumn/ScanPort的name/port保留变量ID，label用于界面；扫描Dataset的AxisId继续由原port identity产生，Axis.name单独承接显示label，Period改名不得破坏Plan/selector接线或改写值/单位。
- Exponential的短区间数据只能提供初值，不能据其x/y跨度限制A/B或寿命tau；只保留tau>0的数学域及用户显式约束。固定B后可需要远大于观测窗口的tau，不能用自动上限排除合法更低损失解。fixed/free映射及通用solver停止容差不因此另建分支。
- Histogram的single/bimodal Gaussian及Poisson-Gaussian只包含其命名的概率分量，不默认添加每bin平底beta；已删除其参数、背景组件、gap-floor种子及从population/fidelity扣除K*beta的旁路。数值COUNT_FLOOR仍只为Poisson deviance计算服务，普通Series Gaussian的背景B保留。
- `release_recapture`是普通Series模型：`q=exp(-W0((2π f t)^2))`、`P=A[1-exp(-eta*q)]/[1-exp(-eta)]+B`，参数为amplitude/offset/eta/frequency，显示符号A/B/eta/f；eta无量纲、f为普通频率并复用sine的inverse-axis单位，时间原点固定为物理t=0而非选区起点。A/B可经现有表达式固定。模型/Jacobian/自动初值只有同一套Numba实现，single/batch复用通用TRF与现有协方差；继续使用Series数值/SEM拟合契约，不隐式构造binomial trial counts或另建温度Task。
- `display_interval`只控制Surface刷新deadline，不决定active history lease内的Measurement primary index是否存在。Runtime只在lease起点之后为indexed-derived Dataset写value或invalid；昂贵Surface计算同一same-shot group只允许一个active，并在忙时只保留Plane latest完整输入，中间indices仍以invalid存在而不排完整frame。
- Panel只原子呈现`data@N + fit@N`。
- Fit selection唯一优先级是committed Area ROI（或显式X-range）→viewport→full range。FacetGrid Focus只定位交互坐标变换与绘制cell，不为数值选区增加隐式facet Scope；同一canonical范围作用于全部facet，用户显式Scope/Last/window仍有效。切换Focus只迁移选区的绘制位置，不改选区范围或revision，任何PanelState重放不得把ROI降级成viewport/full；独立按cell配置的classifier阈值仍由其原owner管理。
- Fit参数编辑只有一个紧凑表达式：`name=value`表示该参数精确固定、从优化自由度中移除；`name=guess(value)`只替换初始猜测。表达式使用当前painted单位，按参数语义换算成canonical：位置类参数（中心、offset）走单位注册表的精确换算，dBm这类对数单位按其命名的功率跨越；宽度、幅度与标准误只有线性比例，同单位原样通过、跨对数单位没有值并loud拒绝，绝不用两点估一个倍率。PanelState、accepted description和Figure只保存canonical `fixed`/`initial` mappings，不保存第二份原始文本。语法/参数/domain错误只忽略该optional override并继续同model全自动fit，同时保留可修的临时draft和loud warning；不得用窄bounds伪装fixed。model切换清除旧model的fixed/initial。Curve、Rolling、Histogram、Image与Facet cells共用这一contract；fixed参数不报告估计误差，DOF/covariance只按free参数计算。
- MEAN uncertainty的唯一owner先取得每个输出bucket自己的mean，再以一次second pass同时累积`E[d]`与`E[d²]`（`d=x-mean_bucket`），由`E[d²]-E[d]²`恢复样本方差；不得以一条series的公共reference代替bucket中心。局部矩差若不大于本次浮点归约与相减的前向误差界，必须归零而不得冒充极小SEM；这个边界只消除无法与零区分的roundoff，不是数据阈值，因此`0±1e-3`与`1e6±1e-3`等bucket必须保留各自真实spread。Dense、axis、factored、generic、Facet及regular/irregular/indexed Rolling均调用同一contract。Rolling trailing不从远端whole-history prefix相减`N/S/Q`，而以逐shot `(n, mean, M2)`和span-block Chan/Welford前后缀形成窗口；总样本数为1时保留该sample声明的sigma。随后Fit对零/NaN SEM沿用统一的最小真实正SEM规则，不允许某个plot kind或模型另行修权重。
- Histogram除single/bimodal Gaussian外提供single/bimodal Poisson-Gaussian（`histogram_poisson_gaussian`、`bimodal_poisson_gaussian`）：泊松律经Γ函数延拓到实数光子数`p(u)=λ^u e^{-λ}/Γ(u+1)`（u≥0），按自身质量归一化后与高斯读出噪声卷积，`f(x)=A/(σ√2π·∫p)·∫p(u)exp(-(x-u)²/2σ²)du`——x的光滑函数，和其他模型走同一条evaluator/jacobian/initializer/bounds路子，直接在plot kind算好的bin中心与计数上拟合，不看数据来源与单位；负值是读出噪声的正常结果。每类参数A（面积=计数×bin宽，任何λ下都成立）、λ、σ；bimodal以λ_L与δ=λ_R−λ_L参数化，headline为δ（bright−dark contrast）。λ为NONNEGATIVE（零光子时密度就是读噪高斯本身），σ与Gaussian模型同为POSITIVE，沿用直方图对宽度的通用半bin下限，没有本模型专属的下限、capability或单位门。种子取直方图四分位间的质量加权均值与四分位距（不是矩，热像素尖峰会把矩种子推进平谷），σ种子为四分位宽度对λ的超出量且不低于一个bin。卷积无闭式，编译核用梯形积分：每光子n个节点，n=每σ与每个p尺度（λ≥1时√λ，否则1/(1+|ln λ|)）各十二个——节点数随参数变化处模型会跳一个积分误差、数值差分会把它除以步长，十二个而非六个把这一跳压到1e-9以下；u=0端点用Euler–Maclaurin修正到O(h⁶)；λ低于1e-150时延拓律比最细网格还窄，密度就取零光子高斯；p表从众数处n个直接值按`p(u+1)=p(u)λ/(u+1)`递推填满支撑区，高斯因子沿网格两乘递推，每bin每节点几次乘法、无超越函数；归一化质量与其λ导数用同一梯形规则。SciPy路径的evaluator与overlay调用同一编译核，冻结anchors用mpmath独立求积钉住它（相对1e-6）。延拓律不是格点律：λ低于约3光子时其均值高于λ、拟合值偏低约5%，低于1光子不再是光子计数律；σ只在低光子数区可辨识（方差=λ+σ²），高计数下误差棒如实变大；bin比读噪宽时σ停在半bin下限，要测读噪须用不宽于σ的bin。
- Pulse字段的默认值只属于原字段；唯一`PulseBinding`按稳定`PulseFieldRef`记录独立Scan能力与Default/API/Config三选一的基础来源，API和Config互斥，Scan可与任一来源共存。Scan/API不保存用户编号或自定义别名，用户按公共`field_label`选择物理字段，内部按稳定field identity寻址。执行值优先级为本次显式Scan值 > 本次API或已保存Config值 > 原字段默认值；未扫描的能力标记只在执行副本取消，不丢弃基础来源。解析/单位换算/编译由Pulse公共层统一完成，Editor和所有测量节点不得各自维护覆盖算法。
- 每个读取文件内容的表单字段（`resource`与`ArtifactInputSpec`）在Browse旁提供Refresh：decode结果按finalization key缓存而该key不含文件系统项，此前只能靠关闭重开Edit、按Start或改动别的字段才能重读。Refresh执行的就是打开Edit时那一套（force finalize + 重投影 + 刷新行），不另立第二条重读路径；`folder`字段只指地方不读内容，不提供。
- Calibration的Reference before、Readout、Reference after明确选择API时间字段，保存稳定字段身份，展示实际字段名；不得用列表第N项、显示编号或猜字段名定位。原有long/short/long帧角色及seconds到字段单位换算保持不变；Scan能力标记不妨碍非扫描测量消费其API默认值或覆盖值。
- 渲染进程边界的两个方向各只有一条写线程，由队列喂。`Connection.send_bytes` 会阻塞到对端读走且 Windows 上无超时，管道缓冲 8192 字节，而 front 消息随 cell 数增长（layout 允许的 64 cell 实测 9762 字节），所以「写要等」是常态不是异常。**任何需要读管道的线程都不得亲自执行写**：子进程 service loop 要发 input-ack 与拒绝，父进程读线程结算结果时要发 drop-input，两侧原先各有一把跨阻塞写持有的发送锁，于是 owner 一写、读线程停排空、对端写不完、对端停读，Qt 事件循环再不返回。一个方向一条写线程即按构造消除该环，顺序由队列保证（input 仍先于引用其 token 的 request 到达）；写线程退出前必须先关闭发送门，进入无人排空队列的消息等于永不失败也永不完成的请求。
- **published front 只写一次，写在会被读到的地方**：渲染子进程原先把一帧写进私有 publish block，再把整块拷进 frontend 映射的共享段——操作者密度下（1470×1071 DPR 3）6.3 MB 写两遍，每帧每面板。publish pool 现在是可安装的（`rendering.install_publish_pool`），子进程装的那个 pool 直接发共享段，`publish_front` 只交接租约，一次拷贝不再发生（实测 raster worker 自时间里 `publish` 12.5→0 ms/s，worker 自时间合计 108→85 ms/s）。**块的归还是显式的两手契约**，因为持有它的是两个进程而只有一个有解释器能说话：子进程侧沿用既有 weakref（渲染器自己那份只读视图消失时），frontend 侧是它释放租约，后到的那一手才把块放回自由表，谁都不许被当成最后一手。没人归还的块就是池再也见不到的块，下一次 `take` 另分配——每一种错法的代价都是多一个段，不是别人的像素。**两个池合并也要合并深度**：`FRONT_DEPTH=3`（一块在写、一块在屏上、一块在途中）乘以 live host 数才是自由表预算，只按 host 数留一块会让单面板窗口每帧新分配一个段，正是 pooling 本来要避免的那 6 ms 缺页。interaction map 与上一帧相同时以 `None` 过管道、父进程用缓存的那份（64 cell 一帧要 pickle 128 个 transform 才能说出「没变」），父子两侧的缓存都随 host 关闭和子进程重启清空。
- **同时画的面板要在不同进程里画**：live 面板的 renderer 原先全在一个渲染子进程里，编译核 `nogil` 能逃出 GIL，artist 更新、chrome 绘制和每帧 front 的 pickle 逃不掉，于是四块面板的 Python 一次只跑一块。`RenderProcessPool` 按面板分片，策略只有两条：**先铺开再共用**（成员未满先起新的，满了才挑 host 最少的那个——一上来就「挑最少」会把四块面板全塞进第一个，因为只有它存在时它就是最少的），以及**按需起**（只开一块面板的 console 仍然只有一个子进程，一个子进程在画任何东西之前就是两百多兆）。**数量是工作点的决定，不是默认值的**，两边都实测过：满速下四个子进程画出一个的 2.3–3.0 倍（4 面板 1470×1071 DPR 3：camera 4M 54.5→156.6 fps、facet64 image 35.1→104.6、heatmap 44.3→102.1、curve 65.8→108.2）；而 10 Hz 的 display beat 下没有人在满速——四面板只用三分之一个核、从没有人在等，于是多出来的子进程一帧都不多画，却让每次开 console 多等 **6.9 秒**（首块面板出图 12.5→19.4 s，全部是这些子进程自己 import matplotlib 和预热内核；实测池 size=1 是 12.46 s，与单进程的 12.50 s 无差别，所以这 7 秒是**数量**，不是起进程的方式，提前并发起也省不掉）、每个再占约 200 MB。所以默认 `size=1`（即今天的行为、今天的启动、今天的内存），beat 提上去、上 2048² 相机、或者手势要跟上整块板子时，由调用方把这个数字调大。上限之下仍是懒起：要了 4 只开一块面板，仍然只有一个子进程。
- **渲染子进程必须是 daemon**：`multiprocessing` 的退出钩子会 join 非 daemon 子进程，所以任何跳过 `close()` 的退出路径（脚本里逃出来的异常、崩溃）都会在 atexit 里永久挂住，而此时像素早已没人要。daemon 让同一个钩子改为 terminate；有序 close 仍会等 save worker 收尾。子进程收尾时告别消息（`stopped`）必须排在写线程哨兵**之前**，否则它进入无人排空的队列，父侧读线程只能等到 EOF 并把「按要求停下」记成失败。
- **渲染子进程一启动就把进程级的首次开销付掉，分两级**：磁盘 kernel cache 省的是编译，省不掉一个新进程第一次画图要付的那些——matplotlib 的 figure/axes/text 模块首次 import、第一次量文字要装字体、numba 刷新 typing context 并从磁盘读入每个 kernel 的机器码。这些与那块面板画什么无关，却全落在操作者的第一块面板上（DPR 3、4x4、3 格 image grid 实测：进程内冷 0.85 s / 热 0.22 s；console 第一块面板 0.9–2 s，之后 0.25 s）。所以 `_render_process_main` 一起来就在自己的线程上跑**一级** `_kernel_warm.warm_process()`——`representative_work` 的一小段、按 console 的尺寸画：相机帧的 grid（unsigned 与 float）、直方图 grid、数据轴曲线 grid，紧接着 64 个空 cell 的储备（`CELL_RESERVE`，任何 grid 都会取走它，所以必须排在这里画的最后一个 grid 之后），再单帧（三种 dtype）、由形状选出的签名、3D 场景——每画一张前先问 `proceed()`，第一个 create 请求一到就答否、预热停在下一张之前（请求与预热共享同一个解释器，预热多画一张就是面板多等一张；没预热到的由第一块需要它的面板照旧自己付）。**fit 不在一级里**：Windows spawn 没有 COW，子进程装进来的机器码全是私有内存，一级就把三族 fit 内核都装上的备用子进程为一块还没见到的面板多持约 70 MB，而它最终拿到的面板只用一族。所以**二级** `warm_fit(spec, storage=)` 只在这块面板**首帧发出之后、父进程请求出现第一个 ≥100 ms 空档时**（最迟首帧后 1 s 兜底）在 warm 线程上跑（不排到面板 worker：二级的 170–200 ms 几乎全是 numba 从磁盘缓存装内核，内核是进程级的，排在 worker 上落在其中的一帧要整整等 200–250 ms；在 warm 线程上只是 GIL 争用、重叠的那一帧多 5–45 ms，而等到空档就没有重叠——shot 间隔 ≥0.5 s，首帧后的空档装得下整个二级）、只按它自己的 kind、形状与存储 dtype（单图走串行求解器，grid 的 live fit 走 prange 批量，相机图走按 dtype 编译的条纹求解器）通过 `configure(fit=)` 画一张带 fit 的图——那正是面板自己会走的调用，直接调引擎会把 session 的请求机器留冷；备用子进程永远到不了二级。放在首帧之后而不是旁边，是因为装内核持有 numba 编译锁、并行入口首次分发会吃满机器。warm 线程与所有 ZLC worker 一样掩到 4 线程的工作组。实测（真实子进程池、DPR 3、5 种面板）：空闲备用子进程 commit 283→200 MB、独占 204→162；首帧/稳定帧/首 fit/稳定 fit 持平。预热失败写子进程 stderr，不结束子进程。守卫：`test_kernel_warm` 在新解释器里 `warm_process()` 之后读每个 dispatcher 自己装了的 overloads（不是 `cold_kernels`——那个连磁盘缓存一起数，缓存满时不预热也答空），断言求解器一个没装；再对每种面板（含各自的 grid）`warm_fit` 后按 console 的尺寸与 DPR 画并配置 fit，断言它们不再装任何 kernel；`test_raster_host` 在真实子进程里量首帧之后的第一次 fit。
- **只有 window 可以躲开 facet 容量上限**：shot 历史每个保留 shot 一个坐标，长度等于面板 window，是操作者旋钮而不是数据的尺寸，所以它超过 `facet_max_cells` 时默认 facet 不选它（grid 把整窗池成一格）。否则 window=1000 的 grid 默认就要 1000 个 cell，surface 拒绝面板唯一持有的 spec，host 根本起不来，操作者还没做任何选择就只剩一条报错和空面板。**结构轴（扫描轴、事件轴）反过来必须照旧被提供并被响亮拒绝**：65 点的扫描画成 64 个 cell 是另一张图，拒绝信息本身就写着两条出路（pin 一个轴或改一个 fate）。「这个 window 能不能当 facet」只有一个所有者 `_facetable_history()`。
- **面板编辑是一个事务，两侧同时回滚**：plot session 拒绝时已回滚自身十六个字段，console 侧必须同步回滚 `binding.state`、Runtime history lease 与 live port 的目标。否则面板永久声称一个从未画过的设置，且下一 shot 仍按被拒的 spec 投影。回滚基线是 host 最后**接受**的状态，不是上一个 authored 状态：被后继取代而从未执行的 configure 不能成为基线（它的基线顺延），已在 host 执行的前序按其真实结果结算后才是基线。
- **被拒的投影必须说出原因**：`project_panel_state` 抛出时 console 保留表单（表单就是修复面），但拒绝理由必须一起送到 `semantic_unavailable`，它渲染在 Edit 界面该段上方的说明标签里——正是操作者刚改 fate 的地方。理由被丢掉时，表单原样重绘，操作者看到的是「选了没反应」。
- **fate 交换读的是生效表而不是 authored 表**：一个轴可能因 kind 的默认而持有角色却从未写进 PanelState（相机信号的 frame 轴即如此）。只读 authored 表会看不见这个所有者，于是抢占它的角色时无人接手，操作者的选择看起来「选不动」。
- FacetGrid overview每个cell显示哪个fit parameter是display state，不是solver request：Workbench把`Cell fit value`普通下拉放在Fit section的parameter expression正下方，但字段明确写回display owner；未选择fit时不显示。choices包含`Model headline`及当前model parameter identities，默认`Model headline`。修改它只重画annotation、不得re-fit；model切换仅在旧parameter不存在于新model时回到`Model headline`。focused cell仍显示完整formula与全部参数。
- FitResult携带source parent/generation/revision；任何history/window投影按Measurement primary index连续，未计算、失败或timeout的位置invalid/NaN，window长度按source indices而非成功结果计数。
- Fit计算在后台worker；Qt owner thread不等待Future或执行fit。
- `saturation`是普通Series模型，`f(x)=(A*x+B)/(x+C)`，参数身份为`asymptote/numerator/shift`，显示A/B/C，headline为A。A为渐近值、单位y，B为分子常数、单位y*x，C为分母平移、单位x；三者可按既有表达式固定或给初值，不兼容旧参数名。只要求本次拟合域内`x+C>0`，允许负x与负C但不得跨极点，原点不随选区移动；不强加`A*C>B`的单调性硬门，增长与下降由数据决定。普通背景加饱和曲线是它的特例，例如旧`120*x/(x+2)+5`对应`(A,B,C)=(125,10,2)`。single/batch、evaluator/Jacobian、初值、covariance与预热复用现有compiled fit链，不另设物理拟合节点。
- Reduction `Last`是对所有Reduced轴按声明坐标顺序取末项的Scope便利写法，不是last-valid或最后到达的physical row。保留轴、Facet轴与Rolling本身的shot carrier不被折叠；末项invalid仍invalid，稀疏末坐标交集不存在时沿普通Scope的空选区语义。数值、Fit选区、selector subject和SEM读取同一restriction，复用既有归约核而不增加Last kernel。
- Histogram与Facet Histogram的Figure recipe必须保存`reduced`与`reduction`，重开后保留原来的Pool/Reduce/Last语义。此前遗漏这两字段的旧Histogram recipe不属于当前完整grammar，reader不得静默把它补成另一幅Mean/Pooled图。
- Active Fit超过1秒必须loud标记该source index invalid并从Plane latest继续；不得积累完整frame FIFO，也不得永久锁住Panel、Qt、Stop或close。普通cadence/backpressure跳过计算的indices同样invalid但不是solver failure；raw Runtime data始终完整。

### 5.2 Performance与state

- 数值显示单位仅在真实消费者需要的表示上转换；归约后绘图不得预先转换全量raw values，raw selector确实读取display时才按需取得。完整用户初值直接进入solver，不计算马上覆盖的自动初值；partial初值仍补自动值。compiled prepare的零行seed输出表示仅请求必要的自动bounds，模型内部只计算这些bounds真实依赖的统计，不产生弃置seed；普通cold/warm竞争不变。预热停止在构造下一个样例前生效，size/parameters按最终初态进入共享Host。
- 内置模型的值、Jacobian与single/batch求解共用同一逐点数学primitive；只有需要导数的求解/协方差消费者才请求Jacobian。只画曲线不生成N×P导数矩阵，也不为可写ABI复制一维坐标。已筛finite的数据在同一私有数值入口复用该事实，新的坐标变换/分箱、RegularImage原始masked输入仍检查实际有效性；不移除cold/warm不同初值竞争或custom fallback。
- 数值core只读取validity，其ABI接受readonly strided mask；已知全有效输入用一字节True广播，不分配B×N的全True矩阵。外部/RegularImage真实mask仍按其实际布局与有效性处理。configure中的spec替换仅修改状态，最外层统一生成一次最终description；public replace_spec仍返回完整真实描述。
- 显示用图像块平均统一float64累加、除完整有效样本数后才转换到输出dtype；NumPy参考与compiled采用同一数学，不为保留整数图像旧的float32中间舍入另建sum kernel或2^24分流。Histogram单组与Facet使用同一分箱kernel及真实边界修正规则，普通分布只是一组内部输出，不向Dataset伪造axis。入口统一的性能代价单独量化，不能当作提速。

- Display cadence按同一HarmonicClock的真实单调时间跨deadline判定；Qt延迟/合并回调时只欠一次最新呈现，不按回调次数再等待若干逻辑拍，也不补画已错过的帧。Pause、容量与same-shot接纳规则不变。
- 新数据已到但尚差少量时间才允许呈现时，原Scheduler从上次staging的同一deadline给出剩余时间，Qt owner重排现有同一个timer到期唤醒；不宽松提前接纳、不另建clock或高频轮询。无新数据、surface忙、Pause和关闭不安排这次deadline wake。

- PanelState一次应用是幂等transaction；no-op产生0 solve、0 render、0 front。
- 长期绘图服务不能用全局gc.freeze或提高回收阈值来永久保留正在使用的Figure。随viewport/time变化的共享刻度placement缓存有明确工作集上限；Host关闭释放其输入与结构依赖，已退休的共享像素segment在最后读者释放后解除映射，不积累空闲旧名字。进程可复用编译代码与有界缓存，但科学窗口固定时资源保留不得随运行时长无界增长。
- IPC静态Dataset结构沿现有input token及引用生命周期只传一次，后续revision只传结构引用和变化的数值/validity；进程重建重新安装，不传Python私有memo。Scope的坐标定位只持该AxisSpec，不通过绑定方法带入整张Domain坐标表；单次Freeze/Save共享同run的plain记录转换，磁盘JSON仍按当前公开grammar编码。
- Configure在最终绘制前被拒绝时只恢复旧字段及renderer准备态，保留原已接受front，不重新compose/发布；最终绘制已开始后失败则必须完整恢复像素，后续主动redraw同样只能呈现旧状态。
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
- Navigation viewport按X/Y分别记录可选范围；未操作的轴为None，继续由已有自动/Fixed参数决定。只缩放或平移X不得捕获当时Y。显式轴limits/mode编辑在同一Plot配置事务中撤回该轴旧navigation覆盖，保留另一轴；重放完整相同target不算重新编辑。普通Image的square/pixel-grid约束仍由原几何owner统一解析。Renderer、Fit、Live/Edit镜像及Figure读同一逐轴状态，不以完整矩形的旧Y覆盖新设置；真实Area selector仍必须是完整数据坐标矩形。
- TaskConsole与FigureViewer的显示执行固定为三个进程、一个Plot真相源：B是Qt主进程并继续拥有
  Runtime、Logic、device client、PanelState、SelectionBridge、LiveBoard与same-shot accept；A只承载
  全部Monitor card的`RasterPlotHost -> PlotSession -> DataView/Fit/Render/Compose`；C承载Panel Edit、
  point review与Figure archive/export。A和C运行同一个render-service实现，交互surface使用同一个
  `RasterPlotHost`；C纯文件save worker直接使用同一`PlotSession/MatplotlibRenderer`，无需屏幕Host。
  不得复制live/editor/export renderer，也不得在A/C失败时退回B进程内渲染。
  Edit的Qt表单和A/C输出的QImage-only frontend始终留在B；B只提交异步命令、接收完整Front和
  observation，任何Qt slot都不得同步等待IPC。同一application只创建一对A/C；TaskConsole与其
  打开的FigureViewer各持一个显式owner lease，任一窗口先关闭都不得终止另一窗口仍在使用的服务，
  最后一个owner才关闭进程。独立FigureViewer则拥有自己的一对A/C。
- Domain Task仍在B决定并写非Figure科学NPZ/JSON、选择artifact路径并向TaskRun登记完成文件；
  TaskArtifactContext不拥有Plot。Calibration与SLM Feedback的Figure archive/render/export
  由composition显式注入同一个C服务执行，Task worker只等待该Future，不得在B隐式构造本地Plot host。
- B向一个render service提交同一`DatasetRevisionRef`只建立一份transport value，由该service内
  全部Panel共享；不得按Panel重复发送或让Runtime感知Plot transport。A/C向B发布的RGBA使用
  有生命周期的共享buffer lease，只有B中最后一个Front/QImage/ndarray引用释放后才可复用。
  operation completion只引用同一host已经发布的front sequence，不得再次复制像素；进程断开时
  所有pending Future明确失败，B保留最后一张完整Front，并由现有Panel lifecycle从accepted
  PanelState与最新publication启动新service generation并原子替换surface。旧generation的共享
  Front可继续读取，其lease identity不得与新generation碰撞。
- ImagePlot及FacetGrid的image cell统一使用`nearest`像素呈现；interpolation不是Parameter、PanelState、Figure recipe或UI字段，任何Logic/Task不得另行设置。
- Image/Heatmap的主显示框始终是固定正方形，每个离散data point是等大的方形screen cell。格距由sample index定义，不要求科学坐标或显示单位换算后的数值等距。单调非等距扫描与非线性显示单位通过同一sample-index↔显示坐标映射摆放格子、刻度与overlay；canonical↔display继续使用现有Unit精确转换，pointer/selector/zoom/pan不得另作canonical分段近似。native和Matplotlib图像都在同一ordinal/affine空间贴RGBA，科学坐标/Fit输入不改写；原等距图保留原affine路径。非方阵在square frame内居中letterbox；zoom/pan只在该固定框内改变索引空间的范围，不重新layout。Single/Facet/Focus/Save共用此映射，不因三维scan或display unit另建图像路线。
- 3D height场景的刻度字符串只规划一次，真实字体宽高与tick/pad像素纳入同一scene fit的对称inset，再应用operator camera zoom；不靠固定几何百分比猜文字空间，不改outer Axes，不取消clip。Raster、id_plane与chrome读取同一scale/centre，orbit不因label位置换边而呼吸；主动zoom造成viewport裁剪仍是原行为。
- Single与Facet的规则tensor数据都先由同一个retained-axis projection一次归约，再只把结果包装成各自payload；不得为某个plot kind另建Facet数值kernel。Single、Facet overview和Focus的每个cell必须经过同一个kind preparation/render owner；Focus只选择同一个accepted cell并换layout/viewport，不重新解释数据、fit或annotation。Facet overview及steady Curve/Image/Fit/SEM保留native raster快路，但native与Agg只允许消费同一份prepared cell state；native拒绝必须整帧回到已准备好的公共draw path或保留上一完整front，不得出现partial/blank cells，也不得靠一次pointer materialization才能恢复。
- 存储分块不能变成逐块Python调度。公共归约直接读取一次typed布局，必要时一次打包计算scratch并批量调用原数值入口；不为每个块创建DataView/统计对象或调用一次kernel，不按大小或Plot kind另选特殊路线。无归约的结构identity可以借视图；真实交错布局必须保持FIRST及极值平局的全局顺序。只计算实际请求的统计，Plot结果不携无消费者的中间counts。
- Indexed history布局统一通过AxisSpec.coordinate_values批量读取已规范坐标，保留原整数/顺序/时间映射校验；每shot行数只在同一parser确定一次，结果对象不再次bincount。事件映射先判定全部轴是否重复，再只将最终保留的映射转成元组，不先装箱整窗后丢弃；不按设备、数据大小或Plot kind分流。
- DomainSpec的基础codes与inner/outer重复因子是唯一行映射；完整展开、按位置读取和布局验证都消费同一声明。历史布局只借用该Domain，不另存完整编码缓存；主索引/时间映射在基础序列上验证，事件是否重复由基础序列周期决定，任意非均匀裁剪仍保持原映射。布局和块边界查询不得先展开完整Point列。
- Plot的轴域直接消费该轴一维映射，不构造全样本positions再抽取representative；used-set只由基础codes确定，完整域及规则slice借原不可变坐标，同单位canonical/display共享数值。所有kind/Facet/Focus/拟合标签走同一轴域owner，删除平行flat/domain缓存。
- 整数频数窗口以持有中的不可变plane身份、局部Point范围和出现次数匹配贡献，窗口名字不作为跨revision复用许可；只打包并统计进出/变化范围，不物化前后整窗或读取无用sigma。源世代与Cell/value契约变化不复用；不消费该统计的Group/Reduced/其它kind与退场表示清理旧carry，不能长期持有旧大窗口。
- 同一次Plot输入的新增依赖沿现有input-token图按依赖顺序批量序列化，共用一次消息和传输allocation；不能每个存储块一条消息/一个共享内存映射。接收后每个数值plane仍独立拥有不可变bytes，单个小块不能钉住整批传输内存；引用释放同样批量发送，不新增第二份history或transport数据owner。
- 原始数值打包只由DataBlock.packed_planes处理布局、compact mask及按需sigma，不做finite过滤或统计；Plot复用该方法后再计算，完整物化仅在有空位时一次scatter到canonical范围。不得再为保存、Processor整数组读取或Plot各留一套逐块拼装。无归约的dense单块借同一数组；未请求sigma不能因共用入口而被计算。
- Indexed Rolling的既有归约结果只按持有中的不可变plane身份、真实布局及计算配置复用；旧结果整批搬运，新/改变的输入整批计算。缓存只在当前表示能消费时继承，完成复用后丢弃未消费的旧packed carry；缩小/关闭history不得因全部统计命中而钉住旧窗口。当前DataView只持当前source或本次复用所需的前一source，不链接全部revision。中心化SEM保持原统计定义，不改grouped/ungrouped trailing权重。
- FacetGrid的facet role可为空；为空不是semantic vacancy，也不允许UI或renderer伪造Dataset轴，而是唯一一个完整cell，标题为`Facet 1`。同一cell kind的projection、fit、selector、Focus和Figure grammar仍走普通Facet路径；给真实轴Facet fate后才扩为多cell。
- Curve prepared state同时拥有series、valid runs、SEM low/high、fit source presentation与style。每根SEM误差棒以stem与两cap的几何并集为一个工字形，subpixel coverage之后只应用一次alpha；不同误差棒仍独立混色，不能按整数display column合并为min/max envelope。Native、Agg与导出读取同一端点与屏幕尺度style。Facet pooled y范围必须包含finite SEM low/high。Fit annotation由公共Matplotlib MathText语义owner格式化；native可缓存MathText最终RGBA，但不得删除`$`、反斜杠或下标后用第二套plain glyph语法重画。
- 未声明coordinate labels的数值轴由共享SmartOffset/locator按空间决定ticks；一旦Dataset显式声明完整coordinate labels，每个label都必须在对应tick原样显示，不得为避免重叠静默抽稀、改写或省略。标签密度、Panel尺寸与zoom是operator明确authoring后的取舍。
- Tick label字号只有normal `6.5 pt`与compact `3.25 pt`，删除逐级×0.8和剩余空间比例缩字。保持对应tick的锚点/alignment；公共locator先在原字号选择可行数字刻度，最终真实text/data-frame/标题碰撞时先尝试更稀的合法lattice与formatter，再在确实无解时切compact；同轴至少两个数值刻度，不单独隐藏Text制造marks/导出分歧。可选计数侧栏zero优先由同一locator省略，色条两端和显式coordinate names不裁。Facet等价axes共同采用最终lattice，改变刻度时同步其native marks；布局变宽后可恢复刻度/字号，稳定布局不重复规划。3.25 pt仍拥挤时接受重叠，不移动锚点、不引入其它字号。
- Facet overview在最终实际cell几何上共同安排标题header和刻度gutter，同方向刻度与cell标题分别使用统一字号；像素取整不得让相同cell跨不同字号档，标题不得借用刻度空间。只在已有chrome布局签名变化时重新定价，复用同一文字度量/placement缓存，稳态数据更新不跑额外碰撞扫描；Focus仍用单图预算，屏幕与导出消费同一chrome。
- 刻度重新规划不等于全部静态背景失效：动态侧分布/色条范围变化仍由其原绘制层处理，坐标范围、layout和显式style的原owner继续负责失效；跨轴规划只为实际变化的字号补充对应轴失效，不因动态侧栏变化重画未变主图背景。
- Color-limit drag的每个accepted move是一个原子preview transaction：先更新candidate与native/Agg共享clim authority，再compose一次并发布该front；不得先发布旧颜色front，再在独立cadence分支recolor，release只负责提交最终DisplayState而不是第一次显示颜色变化。
- Staged Monitor widget的自动front与手势回复只能即时呈现当前已安装数据identity的preview；拖动不授予切换shot的权力。新数据仍由Board按same-shot group显式呈现，交互description若已带新数据则只请求已有presentation pass，不单独推进该成员。普通自动呈现的Notebook/Edit不受Monitor cohort门约束。
- History表示切换若删除退场轴的fate，后续configure与port必须继续使用同一归一化后的PanelState，不得回写调用前的旧candidate。已同identity描述的Panel在新surface接受前保留完整controls；schema-only fallback只用于尚无对应accepted description的面板，不能用空Fit词汇替换已显示词汇。
- Configure接纳与数据呈现共用已有presentation epoch：旧表示上完成的配置回复不得恢复退场轴或覆盖新target；沿现有presentation debt等待完整新surface，不按history轴名过滤回复。
- Pause时Edit/Refresh仍可采用已显示的完整快照，但不得隐式stage新的Live数据；尚欠的新数据刷新由Resume后的正常admission兑现。同数据上的clim、selector和viewport交互继续正常工作。
- 同一shot的Surface仍原子accept；staging只把active-fit surface排在display-only sibling之前，使更深依赖链先启动，不改变panel cadence、cohort membership或accept顺序。
- A内的Panel仍各自保留现有Raster/analysis worker、latest-only admission与串行Host状态机；它们
  共享A的一把Python GIL，所以该拓扑保证UI/Runtime与Save/Edit隔离，而不冒充Panel间Python
  绘制已经进程并行。A/C各自的Numba/OpenMP pool保留本机logical-CPU容量，但每个Raster/
  analysis worker默认只启用4-thread team并在空闲时sleep；大量互不重叠的Facet raster lanes
  可在本次kernel内临时扩到8并立即恢复。A与C的并发native team总预算不得超过本机容量，B不再
  初始化Plot kernel pool。operator显式环境设置仍优先；不得用跨host render/fit锁重新串行化A。
- RegularImage live batch即使具有完整warm seed也必须保留cold proxy竞争，再以选出的seed做full refinement；warm不能跳过cold证据、成为不可恢复的authority。
- RegularImage的single是同一批量数值流程的一条lane；不得按cell数量维护不同full-refinement算法。规则网格以明确的内部shape/axis包交给同一TRF，不展开重复的逐像素坐标，也不改变Dataset或Plot轴语义。现有数值context同时供prepare与objective使用，可保留每cell一次计算的中心化统计量；原始物理参数、bounds和损失/收敛含义保持不变。最终质量仍由真实数据的直接残差核对，不能用不稳定的大数相减或放松精度换取提速。
- 编译Fit中未启用权重时，由既有use_weights表示并传递空权重行，所有objective/finalizer只在启用时读取权重数据；不得为默认权重1建立完整B×N数组。前景仍由Agg/FreeType/MathText产生字形/覆盖率，现有compose按原顺序批量重放；未纳入批量覆盖的artist在原顺序位置保留既有draw，不另建科学数据路线。Image的备用像素在fallback/export消费时才由公共owner物化，普通native帧不重复生成一份未绘制RGBA。
- 编译、磁盘加载与数值执行都按本次实际请求的输出决定；RegularImage由自身信息矩阵收尾时，通用TRF不编译/加载未使用的普通finalizer/value-Jacobian，也不分配其弃置协方差/误差占位矩阵。全部参数固定的通用收尾只求模型值与质量，不求导数。需要普通收尾的消费者仍沿同一既有路径得到完整结果。
- 普通fit收尾的模型Jacobian是本次owned工作区；自由列选择与权重/robust缩放写入其前缀，不另建N×free矩阵。只有列重排存在写后读覆盖时才使用一行scratch；invalid行归零。通过Numba现有原生LAPACK Householder QR只取R，再对小R做SVD，保留奇异值/右向量与原秩阈值，不生成不被消费的Q及N×free左向量，也不改成平方条件数的JᵀJ求逆。成功路径直接使用模型返回的owned预测数组，不复制另一份fitted。秩与协方差始终来自最终参数的Jacobian，不读取上一trial的信息矩阵冒充最终导数；single/batch共用同一数学流程。
- Fit的warm记忆只保留当前request/model/cell最近一次成功参数tuple，失败清除；前次参数仅是与当前数据自动候选竞争的初值，不再通过半径/幅度/history chi-square阈值另设资格状态或扫描原图。RegularImage的线性least-squares proxy只负责寻找初值盆地，可使用与最终输出不同的收敛精度；robust loss仍保留原proxy精度。所有fresh正负候选仍参与，最终参数、残差与协方差必须继续来自完整数据及既有full-refinement精度，不能把proxy结果直接当成最终拟合。
- 通用compiled TRF的停止判据不得把受信赖域半径限制的小步当作函数收敛；参数步长收敛须每个自由参数各自满足相对变化条件，不将不同单位/数量级的参数合成一个范数，让大参数掩盖其它参数仍需调整的事实。保留原初始半径、梯度判据、误差权重、fixed/free约束与cold/warm候选竞争；不按模型、eta阈值或上一帧结果修饰答案。
- Title/layout等非plot变化不得re-fit。
- Histogram classifier先按distribution选择模型来源：调用方已提供Gaussian components就直接呈现该模型，显式空模型直接不画；仅未提供模型的分布自动求解。完整classifier初态必须先于Host首次计算传入，不能先fit再覆盖。拒绝overview/单series的line交互不得物化native artists。
- 删除重复configure/clear/replay与多front handoff。
- 交互配置的幂等判断比较renderer当前交互状态，不比较上次整帧快照（指针操作可在两帧之间提交）。Group/投影变化使锁定目标消失时，成功present后的规范状态回到同一DisplayState并统一通知；不得保存不可见锁或切回后自动复活。configure中的通知在最终render成功后一次合并，拒绝回滚不发布未接受状态。
- 显式series lock属于DisplayState.interaction，通过既有display通知、configure事务及Workbench同一namespace/revision/pending target规则同步Live与Frozen，不新增series专用订阅线路；状态只保存稳定group key及facet身份，不保存axes对象地址。PanelState/Layout/Figure保持同一interaction。指针hover不镜像到另一窗口，但Save在实际host内冻结当时可见的presentation/readout并写PNG及Figure recipe；重开复现该注释，真正鼠标move后再由本地命中更新，不能导出时主动清空已有文字。Rolling实时数值仍在主图框上方右侧，标题和数值在原有顶部预算内排布，复用原Text/动态合成/导出；同preset的Rolling与Image外尺寸相同，数据框尺寸也不改，不能为meter增加专属margin或在Qt裁切掩盖。Series交互文字仍留在框内右上角。
- Live/Edit之间的选区镜像也走同一个`configure`事务，以`selector_updates`只更新命名的kind，保留执行时其它选区；完整`selectors`替换后才应用同次patch，排队合并遵守同一顺序。事务返回同一front的完整`DisplayDescription`，不得把`SelectorState`交给configuration接受入口，也不得先安装像素再验证返回契约。Edit的交互与Refresh共用已有pending/accepted入口，Save在配置未接受时保存最后accepted frozen recipe，不复用正变化的host。
- Qt owner必须在RasterPlotHost第一次render前把当前screen DPR以plain scalar交给Plot；不得先按默认DPR生成front，再在Widget挂载后为同一data/state重画一次。Form consumer在FormSpec结构和实际Widget值均已匹配时只接受新metadata，不得reconcile；keyed runtime choice domain真实变化仍强制刷新。
- Fluent choice是`zlc_ui`唯一前端owner：collapsed控件、一个owned item model和operator信号在控件本身；flat/tree popup view只在operator第一次展开时建立，随后复用，Tree不得先造flat view再替换。popup QSS只有一个共享声明，数值/choice authority仍是typed model，Workbench、Plot和Logic不得感知popup、font metric或Qt私有view。popup几何是内容的纯函数：由可见行数、delegate的行高与`sizeHintForColumn`、view的frameWidth、collapsed控件宽度、锚点下方空间和active-screen宽度上限一次算出，并据此把纵/横scrollbar policy定为AlwaysOn/AlwaysOff再告知view；不得从scroll area读回`view.width - viewport.width`等lazy layout结果（那是上一次展开的残留，曾让同一picker在整洁与双滚动条之间交替），也不得手算字体/padding/native scrollbar metric。横向bar只在popup已达宽度上限而内容仍溢出时出现；Tree展开/折叠与open model变化重走同一owner。
- Semantic Fate中的Scope始终是一个popup action，不得因轴坐标数量隐藏能力，也不得把全部坐标展开成popup rows；只接受实际typed coordinate，不提供Latest。collapsed Fluent control显示`Scope: coordinate`，文字区单击激活/再次单击取消，激活以统一Fluent选中样式表示，右箭头独立打开fate菜单；点击其它位置、Esc、失焦或隐藏取消。只有当前Scope控件已激活时滚轮才按schema真实坐标顺序切换，否则交给外层页面；激活本身不修改参数/触发render，Live metadata更新保留同一控件的激活态。PanelState、PlotSpec与Figure仍只保存完整tagged `scope_fate(coordinate)`，UI不保存第二份mode/coordinate状态。
- Histogram只有`bins`变更需要一次完整sample projection；`density`/`cumulative`只是已接受bins的representation，不得再扫描full payload。复用已settle tick unit时必须在枚举lattice前先核上界，不得因range大幅变化卡住UI。
- Histogram的Group是普通axis fate；Single、Group、Facet及Facet×Group共用一次分组分箱，所有分布共享bin edges。Counts保留每组计数，Density每组独立归一，Cumulative沿各组bins累加。保留原来的无描边矩形bin样式，只改变分组颜色与透明度；hover/lock调整填充透明度，不增加step轮廓或不可见替身line。交互直接读取同一bin几何，Group不把科学数据伪装成Curve。逐组Fit保留真实sample axes，不混组、不悄悄取首组；既有fit结果契约以有序sample_axes承载Facet×Group，单fit是同一表的无轴一项。
- Series命中只读公共绘图owner实际画出的polyline/孤立点、fit源scatter或Histogram bins，不从raw重新构造一套全分辨率屏幕几何。现有命中缓存只保留当前几何及bounds，数据/viewport变化随原入口失效，不常驻每段方向等重复数组。Focus/Rolling复用同一native数据绘制，按真实zorder分批；手势只修改painted axes，accepted interaction map直接定位axes，不扫描隐藏Facet子树。有效hover/lock/wheel只经原OVERLAY入口compose一次，不投影、不重新fit；无变化不发新front。
- 正式96×128 Camera、小Area ROI、主图atomic fit、并行ROI image与一个fit-parameter Rolling Panel链路以100 ms作为profile警戒线；明显的额外cadence、HOL、错误串行和重复render必须删除。若剩余是必要fit/raster/Qt成本，只有能带来实质收益且不增加不相称复杂度的优化才实施。
- 性能以真实TaskConsole、1/4/8 panels、fit+overlay、Setting/Edit和Qt owner latency为profile对象。

### 5.3 Overlay与selector

- Overlay producer发布匹配中立Plot contract的numeric/bool companion signal，并在同一run record中携带该contract要求的geometry document；`zlc_plot`拥有通用adapter与renderer，Workbench只按contract路由，不import domain plugin，也不重建science。
- Data、Fit和Overlay共同使用同一个scope/axis/fate projection；动态Overlay读取其exact publication，并跟随主图已物化快照的范围：主图没有`DataBlock.window`时读取canonical prefix，不受其它Panel对companion的history lease影响，也不用最后event chunk覆盖finite前缀；主图有window时读取相同start/latest，不能拿另一个保留范围拼图。范围事实只由Runtime提供，不按axis名字猜测。公共`projection_scope`将`Last`化为各Reduced axis声明顺序的末coordinate，随后与显式Scope和facet走同一限制；不是最后valid值，不回退到之前已采位置。Overlay只借Repeat/Point确定对应采集cell，保留自身完整site向量，不把图像pixel axis当site axis。Mean没有另外一套Boolean归约/共识判断；scope后仍有多个Repeat/Point cells就不画离散判决。无法唯一对齐则拒绝。
- 图像数据更新是一份完整presentation输入；新数据未携overlay表示该帧没有overlay，直接更新与Host管线都必须清除旧层。同一数据上的显式overlay-only编辑仍是独立配置事务。动态status的invalid或无法唯一选定状态不画判断圈；静态Calibration/point-review显式标记不受该数据有效性规则影响。
- ROI/binning坐标只由一个transform owner处理。
- 分组颜色直接按源坐标序号读取公共line cycle，不跳过首槽：group[0]与未分组曲线均为原灰色#808080，其余依次使用共享色表。Curve/Rolling/Histogram、Facet overview/focus及meter共用同一映射；颜色不随当前可见子集重新编号。
- Producer联动只修改下一次运行的唯一Logic草稿，不改正在执行的运行。删除ROI时恢复联动前对应字段的草稿值，不从设备readback或Panel显示范围猜初值；必要的撤销记录留在现有authoring owner，不成为另一份可编辑草稿。每个字段只由最近写入它的Panel持有撤销权，多Panel不堆叠旧patch；用户之后手动改写的字段不被ROI撤销覆盖。运行一旦开始，它启动时用的值就是producer自己的值：撤销记录清空，之后删除ROI保留的是最近一次运行使用的值，不回到运行之前的草稿。数值控件的被动规范化沿同一draft patch标明normalized，更新唯一草稿但不取消撤销权，普通Form与Scan贡献编辑器相同；不能靠浮点容差或JSON文本顺序猜测用户意图。镜像删除使用公共panel_plot_selectors给出的绘制种类，不把Curve Area的x_range数值语义当成实际图形类型。没有producer参数语义的Histogram选区只限制数据，不联动producer。
- ROI统计按已启用输出准备计算：仅Mean/Sum不构造整数直方图，均值与总和共享一次累加；确实请求尾部统计时才用原计数路径，并保持全部结果一致。已知全有效的stacked结果不先分配随后丢弃的零矩阵。此规则不改变默认发布开关或推断订阅需求。
- Selector Off时plot不消费任何pointer gesture：不画selector、不zoom/pan，也不响应双击facet focus；普通滚轮继续滚外层board。
- Selector On时，FacetGrid overview只响应双击进入cell，不得在overview开始area selector；进入具体cell后，selector才按该cell的canonical projection工作。
- Area selector及已有Area handle/body的左键press只做命中并arm手势；它不得建立candidate、发布selection或渲染overlay。只有按键仍held且pointer坐标相对press确实变化的首个move才开始gesture并首次preview；原地release不得产生零面积Area。Qt/Notebook的double-click都走这一共享状态机，双击的首个普通press/release因此不能闪现或提交Area；空白单击清除既有Area仍是独立click语义，不得靠创建degenerate candidate实现。
- Curve的invalid位置切断line且绝不跨洞连接；standalone Curve与FacetGrid Curve cell中，每段仅一个valid点时用同series颜色/alpha/linewidth的短横线glyph显示，不建立scatter或第二series。Grouped Curve、Rolling与Histogram共用series interaction：Curve/Rolling的hover轻微加粗命中的line，Histogram的hover增强对应bin填充，其他series保持正常alpha；click lock才突出并压暗其余series。Series文字固定在对应axes内部右上角、无背景框，locked文字以`* `开头；locked时滚轮按Group axis顺序切换，未锁定时滚轮仍缩放viewport。颜色由公共Plot palette按源group坐标序号稳定分配，三类共享，不按当前可见子集重新编号。Rolling上方meter取锁定组的最新有效值并同色；未锁定显示首组且标明组名，hover不切meter，所选组无有效值显示破折号。Legend暂不实现。

- 字段旁的绑定入口只选择Scan开关与Default/API/Config来源，不循环点击、不显示编号、不在此命名Config key。Edge/Ramp数值可编辑；Hold沿唯一FieldVM.editable事实将数值框及其Slot入口整块灰显禁用，已开的绑定popup关闭，模式Combo仍可切回Edge/Ramp恢复编辑。小入口以S/A/C组合状态展示；Config命名及字段引用仍在Config tab。CPU/Qt只投影当前绑定和有效覆盖，不保存历史输入作为第二份真相。
- Config JSON只拥有`format`和命名`values`表，每个名称对应value/unit；无数字键语义、name/source/from-pulse/field来源信息，旧格式直接拒绝不兼容。多个Pulse字段可引用同一Config名称，字段到名称的引用保存在Pulse，不写入Config文件。未指定名称、未加载文件或名称无对应值使用原字段默认值并显示未覆盖；实际被消费的值遇到格式或单位错误则报告，不能假装应用成功。
- Sequencer Init可显式给出可选Config文件路径，并复用设备已有的load_config_file入口；留空不加载任何Config文件。Workspace只创建目录，不生成或自动读取current.json，Pulse Editor Connect不另选文件。Config tab的显式Load/Save仍使用同一设备配置入口；复用现有设备保留其已加载选择。已有文件不因初始化而修改或转换，显式给出的缺失/错误格式文件按原初始化错误通道报告。
- 旧Pulse/Config只通过操作者显式运行的离线迁移工具转换：先验证完整候选、备份原始字节，再原子替换；旧数字Config N在文件与Pulse引用中共同变为config_N。正式reader不兼容旧grammar，工具不改Figure或自动应用配置。

- Remote编码/解码按immutable dataclass类型复用字段metadata，JSON grammar与对象验证不变。Load可明确复用同连接已接受AppliedState中的未变authored_source，不反复传输/解码原稿；新实际source/program照常验证。原稿变化就完整传入，服务器无旧记录时拒绝复用，绝不把filled source当原稿。客户端在发送前清原有应用缓存，失败不缓存假成功；重连继续按原有applied查询恢复事实，不增加源缓存或后台同步。
- Config tab用独立草稿编辑命名值表并显示dirty，只有显式Save/Save as写入文件后才参与下一次Fire；不自动保存，不把未保存值送设备。Load绑定已保存文件，Refresh重读编辑文件（已经Unload的文件不被Refresh偷偷重新激活），Save使用草稿而不是Pulse默认值；Pulse Save只保存Pulse内容与引用。File读写沿唯一codec及原durable API，既有device load/fire在开始新执行前重读已绑定文件；运行中的Pulse不因编辑草稿或保存而改变。Config文件与Pulse引用的dirty互不混用；外部刷新不能静默覆盖未保存草稿。
- Config tab的Pulse bindings采用与Target一致的对齐Fluent行布局；Config name始终是可编辑文本，空白表示未指定，下拉只建议当前已加载Config文件中的名称，不加入Default、当前自定义名或历史名称。换文件和Unload只改变候选与覆盖状态，不删除已指定文本；未保存的Config草稿不冒充已加载文件的候选。真正使用Config覆盖的可编辑字段采用用户选择的C色（#D1E8E1）与深色文字，未覆盖的Config字段保留白底绿框；禁用字段沿普通Fluent灰显，不改变尺寸或字体区分状态。Calibration两个曝光字段后紧接三个API选择框，共用同一参数表单列宽及草稿通道，资源选择/Refresh沿原入口。

## 6. UI与Lifecycle

- TaskConsole与FigureViewer工具栏的Save image均保存当前整个GUI窗口，复用ConsolePresenter的截图动作、workspace当天目录与公共窗口capture；不依赖已打开的archive或某个active Panel，不在archive旁另存单Plot PNG。Edit的Save figure仍是数据和科学Figure归档，与GUI截图分开。
- 可编辑数值输入按“真实设备范围→限幅→可见精度舍入”的优先级处理：普通越界不是错误；舍入越界时取该精度小数网格内最靠近边界的合法数，不能仅因原边界尾数写不下就拒绝。显示数就是控件与提交草稿的数，不保留隐藏尾数，不自动改科学计数法、不扩宽；只有整个合法范围确实无可显示数才报错。disabled/read-only数值投影不拥有该值：它保留canonical数值，显示完整数字并由现有文本框处理超宽内容，不得因显示宽度设置`numericError`、舍入值或发送normalization；重新进入editable时才恰好重跑一次原可见精度规则。用户主动单位显示切换在成功前不提交候选ports/单位/值，拒绝时保持原整份显示；设备真实范围更新不能被显示失败回滚，必须保留新范围并明确标记当前输入不可用，不能把过时范围或未显示数值作为可Apply草稿。Decimal步进、步长对齐和单位换算继续共用原实现；同单位族前缀换算在公共单位owner做精确十进制缩放。单位、对应范围及数值在同一控件入口一起更新，不把新单位bounds先套在旧单位数值上。被动规范化只发`valueNormalized`回到草稿通道，不得先以普通`textChanged/valueChanged`冒充用户编辑或Apply；只读科学数据/设备读数不因此改写。Scan保存点列采用最终接受的限幅/舍入结果；浮点尾差不能将等距Range误标custom，真实非线性点列仍保留，Range与Values互不填充。
- Config与Manual Data等表格复用公共Fluent表格高度策略：不超过2000行时按实际行高纵向展开，只由外层Tab纵向滚动；超过2000行才保留有界的内部纵向滚动。横向溢出仍可滚动，虚拟model不因展开而复制数据或逐格构造控件。InfoTree的末行文字与动作按钮在单元格内保留上下间距；不得靠在外框与滚动条之间加padding掩盖行内容裁切。
- 滚动条轨道与滑块共用Fluent边框的圆角几何：位于边框内侧时扣除实际边框厚度，无框容器使用原圆角，避免贴边时两条圆弧圆心不一致留下空隙。公共样式统一处理，不按页面设置独立半径。
- Pulse绑定popup使用共同标签/内容列：Scan后的布尔check、Source后的Default/API/Config互斥三态胶囊开关，以及Config时只读的实际Config name。三态每段按自身文字加少量边距自然定宽，不横向拉伸。弹窗复用公共anchor toggle及点击重放处理：同一slot再点即关闭，不先关闭又重开。名称配置仍只在Config tab，不能把popup变成第二个Config编辑器或用操作指引冒充名称。

- TaskConsole与FigureViewer共用既有Qt回调异常边界：用户动作、初始打开、owner-turn和异步完成中的普通Python异常进入界面状态和日志，不退出GUI；直接程序化调用仍正常抛错。报告错误本身失败也不得逃出Qt。界面状态、回调与后台worker寿命仍由原owner管理。

- `zlc_ui`不拥有domain parser、device state或plot lifecycle。
- Frozen Edit的配置兼容性与数据年龄分开：新generation（包括revision重置）或旧source已退休时明确显示橙色旧代/不再current提示，尚无新首帧也不能漏提示；不重建表单、不替换旧Frozen图和数据。Save仍精确保存该Frozen，只有Refresh接受新front后才换快照；配置不兼容仍按原stale gate处理。Console与FigureViewer共用PanelEditorView的轻量状态更新。
- Derive的输入/输出信息以完整三domain逻辑结构为主，颜色复用Panel标题的`AXIS_GROUP_COLORS`；NumPy物理shape明确标为Storage辅助信息。多个逻辑Repeat/Point轴共享一个物理carrier是既有Dataset模型，不把逻辑轴丢弃或错误解释成同一轴。Signal列表仍只显示数字维度，不随此信息面板增加轴名。
- Derive的具体Logic Editor只组合普通Fluent控件：Input range与window数量、逐输出Name/多行Code、Add/Remove、只读输入/输出三domain结构与代码帮助。结构摘要读取Runtime已发布的schema/snapshot，未运行的代码不预猜输出；普通Python/NumPy、`isel/sel/where`、命名轴归约、有效性、显式schema及range边界都在同一帮助文本中说明。Qt只编辑草稿/投影metadata，不执行表达式或物化run/history；数值业务仍在Derive自己的owner。
- Qt slot不得执行blocking I/O、device tune或`Future.result()`。
- Window只有在owned command、worker、executor和claim安全退出后才能消失。
- 正式Board通过host的`qt_widget(auto_present=False)`挂载唯一缓存的Qt adapter，不另建未纳入host关闭流程的surface；普通Edit/standalone保持自动呈现。Card退场先以现有`set_surface(None)`解除Qt父子关系，host完成异步关闭后才结束adapter，不能由Card的deferred delete提前销毁仍接收结果的QObject。
- Device Manager的`instance_id`是稳定device identity，operator-facing role只是metadata；改role不得把同一硬件变成remove/add。所有设备UI的名称与choice label使用Role，choice value、保存引用与device ownership仍用稳定ID；Loaded/Control使用accepted apparatus的Role，未Init草稿显示自己的Role。Role在同一apparatus内不得重名；无效草稿在Init/Save前拒绝，不回退成旧名字后继续Init。Loaded card的Control与Close都只提交intent，不能由View直接关device。
- Active apparatus变更走同一个`ExperimentSession`内的差量reconcile：相同key/type/canonical parameters的leaf、SignalPlane、TaskConsole与Panel继续复用；新增只build新增leaf，remove/change/Close只处理受影响leaf、world-bound closure及factory dependants。只有完全相同的draft/live集合才把主按钮解释为Shutdown。
- Reconcile前以device-key maintenance barrier阻止新Logic/command，停止并等待受影响Logic lease，关闭对应Control；已有不可取消command时loud拒绝。partial close/factory cleanup失败后，所有仍open的leaf必须继续由Session或recovery owner强持有，effective live config与TaskConsole device projection同步后才允许下一次操作。
- Device operation或projection-refresh pending期间Control、Close、TaskConsole X和root close不得越过owner状态；失败保持window/session可达并提供只刷新projection的retry，不重复hardware work。
- Device Manager关闭失败必须展开ExceptionGroup里的实际原因并保留完整日志，不能只显示sub-exception数量。Vaunix LMS命令返回按厂商LVSTATUS最高位判错，不能把任意非零成功值当失败；Init/Close/frequency/power/RF开关共用同一校验。真实SDK错误仍保留installation的失败持有与显式重试语义，不自动重发或假称已关闭。
- Hosted Task可登记且只能登记一个domain-owned partial-exit writer；Runtime在worker线程、撤回Dataset及把TaskRun标为stopped/failed之前恰好调用一次。Writer只能从已经完成的数据原子写并登记checkpoint/process/Figure/preview/summary，不得制造required final；writer失败不能覆盖原始hardware/science failure：failure时附注在原始错误上（进入记录的traceback），Stop时成为observation的error与stopped记录的error而状态仍是stopped/cancelled——Stop不因保存失败变成failure，保存失败也不得被当作从未发生。Calibration与SLM Feedback都必须使用该边界保存各自可证明的partial报告。
- Device Control只显示adapter声明的`TunableField`：稳定表单metadata、authoritative current、当前是否live-write及dependency group。每行统一为Current、Desired、Live apply、Apply和Status；打开/显式Refresh及成功Apply后的readback只走session-owned串行device worker，Qt不碰SDK，也不做周期hardware polling。Generic Control的X在既有close guard放行后只隐藏，同一device session复用窗口与Desired/单位；隐藏时停止Live debounce、撤销尚未执行的字段写入并跳过周期UI投影，重开按保留单位读取current；device unload/rebuild或session shutdown才真正关闭并释放窗口与Qt连接。
- Device Control的表头与全部Fluent form rows共用一份列宽预算：只有Desired列伸缩，其余列按全表内容对齐；Desired内的单位选择器同宽，输入框右边缘一致。布尔开关保留自己的绘制/命中宽度，无Live能力的行保留空列。不得让各行按不同的两个stretch列独立分配宽度，也不单独给RF手写另一套表单。
- Scan Plan在原轴行owner内共享全表列宽：手动轴的提示与名字属于同一identity单元，普通/手动轴均保留单位位置；起终点等宽，单位选择器、点数、状态与删除列对齐，只有identity随窗口伸缩。单位控件只在自己的单元内替换，长状态使用现有ElidedLabel，不推动其它列；不得通过重建行或改写ScanPlan实现排版。
- 运行时设备字段的identity不编码单位：RF是`frequency/power`（多通道加`ch1_`等前缀），DG4000额外使用`ch1_fsk_hop_frequency`/`ch2_fsk_hop_frequency`，范围控制是`frequency_low/high`和`power_low/high`，Pylon是`gain`（dB），Virtual Camera是`exposure`（s）。单位只由metadata及显式请求携带；Control、claims、Remote、Scan port、派生scan axis和新保存引用使用同一identity，不在UI删后缀，也不保留旧runtime别名。固定单位的Config/Init字段、CameraWorkingPoint及SDK参数仍保留`_hz/_dbm/_seconds`等单位说明；RF范围控制与固定单位Init键在同一声明中明确对应。历史实验数据的已存字段不重写，旧authoring layout中的设备port和对应Panel fate轴引用需要同时更新。
- RF frequency/power policy window的四个edge是Init与Device Control共享的optional `TunableField`；`None`唯一表示该侧没有bench policy limit。Init省略全部edge不得移动硬件；Control可设置或清回`None`。仪器自身的frequency/power limits在Init连接时从设备读出，并以`TunableField.device_limits`只读投影给Device Control显示；DG4000的FSK hop frequency按每通道向仪器读取自己的current和waveform-dependent MIN/MAX，并与同一`frequency_low/high`安全窗口取交集，不能绕过carrier的bench频率政策。UI control、外部`tune`与Scan共用同一个有效范围＝policy window与device limits逐侧取更紧者，因此只要device limits存在knob就向Scan暴露有限range，缺失的policy edge不阻止扫描。window与任一受它约束的仪器范围无交集时Init/Control都loud拒绝；连接仍是只读观察，Init允许并显示policy外的真实carrier/FSK current，只有之后由Control收紧window会strand任一真实current时才拒绝。
- Logic在实际Start lease中声明protected fields；运行时才选出的device scan ports以nonexclusive resolved field claim加入同一lease。Device Manager按所有active claim与dependency closure锁字段，不按camera type或字段名特判；无owner时正常写，有owner时只有未claim且adapter确认live-safe的字段可在operator接受风险后写。
- 风险接受只绑定当前`device_session_id + device-specific owner revision`。owner或session变化立即失效；字段命令在DeviceUse同一原子锁内再次核revision/claim，active field command阻止新Logic Start。in-flight live edits只保留每字段最新值，owner变化后尚未执行的write取消。
- Control空闲beat只比较DeviceUse既有owner revision；没有变化不重新构建字段权限/依赖闭包或全表投影。本地编辑、风险确认和设备命令结果直接投影一次；该显示去重不参与实际command admission。
- `device_session_id/settings_epoch`只在成功且effective值实际改变时推进；requested/effective/readback与active owners只在Logic运行期间的真实override中记录。Camera frame在adapter接受/复制边界冻结epoch，不能在publication时读取“当前epoch”倒填旧frame；Pylon无法证明live tune前后的buffer边界，因此本次arm内tune之后的每个readback都保守标为old/new mixed，只有重新arm才回到单一epoch——一次read碰巧取走部分旧队列不是其余帧已是新设置的证据。Publication只带压缩epoch ranges，Figure只展开lineage实际引用的记录；idle调整不进入历史。
- Pulse Stop UI立即进入Stopping；Stop/SAFE高优先级并可取消普通wait/transport，hardware ack后台完成。
- Pulse Editor每个channel保留同一组编辑/单位/全开/全关列；DAC不支持全开时保留按钮但disabled，不隐藏列。全关仍可用。
- Pulse DAC模式Combo的宽度由公共FluentComboBox按实际字体、文字边距与箭头给出，不在Pulse复制估宽公式；模式与数值框之间保留行布局的显式逻辑像素间距，不能用过大的固定Combo宽度挤掉间距或数值输入空间。
- Pulse Period字段的无变化判断只读取该卡片当前accepted PeriodVM（名字、duration/unit、DAC mode/value），不保存上次发送intent的数值。Load/Sync/Clear及拒绝后的回显共用原投影；未被owner接受的输入不能被视作已提交，控件复用不得吞掉再次输入。
- Timeout显示真实错误但不冻结UI；未确认前不能显示Safe。
- Form reconcile只在schema改变时重建dependency graph；同schema成功adopt只更新值。隐藏Setting延后构造/度量到实际打开，Manual Data单格修改只通知实际变化的格子。Pulse显示切换保留原timeline/滚动与bracket对象，不拆装未变控件。
- 隐藏Setting在原Card保留待消费更新（包括runtime choice目录变化），显式重开或随父Tab重新Show时只消费一次；不得丢掉hidden force，也不得靠后续数据帧碰巧刷新。Pulse容器仍由原SetFixedSize layout定尺寸，真实LayoutRequest只同步已有gap指示线，不恢复多次scroll补偿。
- PanelState按当前字段定义恢复UI状态，未知字段不作为当前参数继续传递；owner wake和产品Figure save各只有一个实现。
- FigureViewer与TaskConsole必须复用同一个`PanelCardView` frame owner、Monitor board、panel preset尺寸、title band、Setting按钮和body padding；card是图的框不是preset的框：只有空卡按preset占位，挂了图的card尺寸只在图本身换了surface（首次画出、按新preset重画、自己加宽边距）的那一个事件里跟着变，操作员选Size不得先让card跳一次再等图，picture、frame与title band必须同帧变化；挂上一个已经带图的widget时card当场框住它；Viewer右栏是白色Fluent work surface，global action bar固定自身高度，Panel在其下方top-align，不能把剩余窗口高度塞进action bar或让同一2x2 card漂到中部；card title读取当前archive dataset的operator label。左侧InfoPane宽度在window创建时一次确定，任何archive label/value不得改变window split；每个信息页是一棵「名字 | 值」两列树：顶层行是该页的主题（一次run、一台device、文档的一个section），record的每个字段都是其下的一行、打开时整棵树默认全部展开，嵌套用每层的竖向guide线和可展开行的chevron画出而不只靠缩进；分支行的值列只用灰色墨写其下有几个字段，长数值列表按个数与范围读；值不换行、不cutoff、不用tooltip重复整页，过宽时整页横向滚动；每页顶部一个filter按子串同时找名字和值：命中的格子染橙色tint、树展开到命中处并滚到第一处、不命中的行隐藏；任一行Ctrl+C或右键菜单复制整个值或名字路径。Raw页就是文档本身按section嵌套，不再拍平成点分路径。Flow使用Fluent node-edge graph：Logic与Device节点不重叠，共享节点只出现一次，causal edge与device-use edge视觉区分；每个node携带它代表的行`(tab, label)`，点击card就切到该页并选中该行；Workbench只给plain nodes/edges，Qt owner负责字体测量、布局、绘制和滚动。
- Panel Setting是Panel page scroll viewport内的`FluentOverlayFrame`，不是top-level companion window；随page隐藏/恢复并被page边界裁剪。header显示固定Panel identity，例如`Setting · panel-3`，不得混入可编辑title、signal或structure。右侧紧凑`×`单击只隐藏Setting、不删除Panel；拖动与再次点击Setting切换仍由同一overlay owner处理。

## 7. Pulse、Camera、Remote与FPGA

### 7.1 Execution vocabulary

- Pulse编辑的Period与每个Bracket的一对post共用一份只读派生item order、一条drag/drop与插入目标通道；post的key是`<bracket id>:start|end`，同一gap内post由内向外排：先是在此关闭的Bracket的end（最内层先）、再是落在此gap的空Bracket（start紧接end）、最后是在此打开的Bracket的start（最外层先）——与model判定嵌套的规则相同，画面和loop表因此一致。重排/插入/删除在一次模型更新中同步period顺序和所有Bracket锚。Bracket可在编辑中为空且绝不自动删除；首/尾空边界允许缺少外邻锚，统一半开gap范围相等即空；空Bracket落在另一个Bracket的边界gap时算在它旁边而不是里面。Add Bracket按当前选择决定新Bracket的跨度：选中period就包住它、选中post就再包一层同跨度、选中gap就在该gap放一个空Bracket、无选择就包住整个Pulse；新id取`bracket<N>`首个空位。只有显式Delete某个post才移除那个Bracket。On Pulse、编译、Save Pulse/Preview及序列导出共用同一nonempty校验和提示，在设备/文件副作用前拒绝；普通Edit不弹错误或抛弃空Bracket。
- Period有两种`kind`：`period`（作者写的）与`spacer`（两个period之间留给慢设备缓冲的时间，由Add Spacer手动加入，永不自动生成）。spacer在模型层就是一个period：电平照常编辑、参与总时长与编译；但它hold每一路DAC（不得有analog step、DAC字段不可绑定），时长可取Config值却绝不scan、也没有API（模型拒绝这两种binding）。文件写`kind`字段，缺省为`period`。编辑器给spacer起名（`spacer1`、`spacer2`…从1数起）：名字显示在每张卡都有的name行里但line edit禁用；编号、`Periods:`汇总与`Period k/N`标题只数作者period，spacer另计（tooltip写「N period(s) and M spacer(s)」）。spacer卡更窄、用页面灰底加虚线边与灰色「Spacer」pill，通道行只画可勾选的圆圈（名字在tooltip），每个DAC是一个禁用的单项「Hold」combo，没有值框；新spacer的每路电平取两侧邻居的AND。
- Bracket post像卡片一样在标题pill里写自己的名字「Bracket N」，N是它在model里的序号（最外层先，随增删重编号，和「Period k/N」一样），左右两根post同名；post的标题文字与边框都用该Bracket的墨色，墨色按序号取自绘图调色板的`bracket_cycle`（`zlc_plot.bracket_color`是唯一来源），预览里该Bracket的loop也用这个墨色画，所以条上的框和时间线上的loop一眼能对上。post的第一行在end post写「Repeats」、start post留空。三种东西三种样子：period卡白底黑标题，spacer灰底虚线灰标题，Bracket post白底彩色标题加彩色边框。预览的loop标签只写「×N」（Run无限写「×∞」），哪个Bracket由墨色说，Run loop固定用`pulse_run`灰。loop立柱向外堆几层由**包含关系**定（`_loop_depths`）：里面什么都没有的贴着行，包住别的loop的比它里面最深的那个再高一层，互不相交的Bracket同高；同一跨度的两条loop按调用方给的先内后外顺序。
- 选中是填充不是描边：被选中的Period card/Bracket post正文填`ACCENT_TINT`、边线`SELECTION_EDGE`（1px），由`FluentGroupBox.set_selected`在paintEvent里画，不走stylesheet。按下一张未选中的卡片立刻点亮它（只改外观，`selection()`不变）；不动就松开是一次点击，经view的toggle成为真选中（再点已选中的卡片则清除）；拖动的第一下把点亮变成真选中，放下后的重排保留仍存在的选中项，所以被拖的那张在插入后仍然亮着，直到操作员主动点它取消。Add Period/Add Spacer之后刚加的那张卡片即为选中。这套语义对spacer与Bracket post同样成立。选中一个gap画一条与卡片同高的`ACCENT`圆角细杠：宽度按间隙宽度取（每侧留1px，最少2px），整数像素居中，不用固定宽度的杠——固定宽度在奇偶不同的间隙里必偏一侧。
- Pulse执行固定为三层且各有唯一owner：`Scan repeats -> scan point -> Run repeats -> Pulse timeline -> PulseBracket`。`PulseBracket`只表达timeline内一个连续period区间的内部loop，有自己的`bracket_id`，左右端点可放在任意合法gap并由一个count控制；即使覆盖整个Pulse也不得冒充Run repeats。一个Pulse可以有多个Bracket，两两要么不相交要么一个完全在另一个里面（部分重叠拒绝），数量与嵌套深度受几何`max_loops`/`loop_depth`（当前8/4）限制；model按最外层在前的规范顺序保存，compile把它们原样写进loop table。Bracket回绕对TTL与DAC是同一件事：RTL在回绕拍把loop-start那一行当作新进入的一行重新发出——它的TTL mask、它的DAC action（edge重新落码、ramp从carry值重新起跑）都重放，Bracket外的行绝不重放——preamble不属于Bracket，它的DAC码不得在任何一次重放中出现。ramp的目标值在下一行起始tick输出，回绕到loop-start行时也一样（回绕拍就是下一行），直接路径与延迟路径同一答案。`fpga/pulse_streamer/sim/diff/`是对旧引擎的差分仿真：两个build各用自己的host打包、各自的真实top在xsim里播同一批pulse，逐时钟比较25路TTL、4×10 DAC与4路da_clk；边沿表引擎唯一不同的是ramp恰好结束于回绕拍时少走最后一步（DAC停在倒数第二个值，其延迟路径却走完），那是旧引擎自相矛盾的缺陷，不是要保留的行为。换引擎必须重跑这个差分。
- `run_repeats`是Pulse文件的正式字段，UI默认`0 = infinite`，有限值为`1..2^32-1`。无scan时它控制整个Pulse（包含Bracket）执行次数；有scan时它控制同一个scan point保持不变并执行整个Pulse的次数，完成后scan cursor才前进。Task的`shots_per_point`只是本次execution对该值的显式immutable override，不修改保存的Pulse。
- `scan_repeats`保持独立，`0 = infinite`或有限完整table sweep数；Pulse Scan保留该字段，Seamless的`repeats`只是本次execution对它的override。它不改变Run repeats或Bracket。无scan时scan_repeats固定为1且不参与执行。
- Host与RTL不得再把三层flatten成一个`cycles`真相：loop table只属于Bracket，`RUN_REPEAT_COUNT`在同一row重复完整Pulse，`SCAN_COUNT`只表示一轮唯一row数，`SCAN_REPEAT_COUNT`控制table sweep。所有层间seam均在同一次FIRE内由FPGA推进；Host只提前补scan bank，补充不及时必须underflow并loud失败。
- run record里的pulse是操作员选的**文件**（`{"name": 文件名, "path": 路径}`），不是文档内部的name；sequencer的device snapshot除板子事实与config值外还携带实际播放的内容：编译程序的digest、总时长、loop起止与次数、scan表行、run/scan repeats，以及填好config与API值的完整pulse文档（每个period的时长与电平、slot、bracket）。FigureViewer的Devices页只列该文档的名字与period数，并为每个播过的pulse给一个Open动作：在右侧与Board/Edit同区打开「Pulse · 文件名」页，用Pulse Editor同一预览绘出该文档：同一控制行（off rows、Selectors、Size、Save Figure）只改画法不改文档，无scan表。pulse时序图（编辑器预览、console面板、viewer同一渲染器）在最上一行之上的一条带里按period印名字，并在每个period边界画一条淡虚线（`period_boundary_dash`），它与网格线的dash图样、墨色都不同，二者不得同形；spacer的跨度用斜线hatch铺满，名字照印；边界线与hatch的上下范围就是最内层括号立柱的范围（脚到顶杆），三者读作同一个框；名字与边界共用调色板的`pulse_period`墨色，不用色块内的白字。Bracket与最外层Run的括号只画两端的立柱与脚：脚长在draw时按坐标轴屏幕宽度的固定比例（`repeat_foot_axes_fraction`，1.2%）取、并以loop自身跨度的`repeat_max_foot_fraction`封顶，所以缩放进去脚不会横跨整个显示范围；嵌套loop的顶杆按点（`repeat_top_step_pt`）向上堆叠、标签随其顶杆一起抬升，y轴上限按屏幕像素换算为最外层的标签预留`repeat_ylim_room_pt`——按行堆叠时行一小，外层的线就会穿过内层的名字。period名与色块名只在其跨度在屏幕上放得下文字时印出：判定在draw时按当前坐标变换下的像素宽度做，缩放进去名字就出现，不按总时长的比例。
- Readout/Dataset/run record分别保存实际Run repeats、Scan repeats与scan coordinates；`shots_per_point`只是Run repeats的Task侧名称，不形成第四层。有限Task在开始前由两层repeat、唯一scan rows与source cardinality算出精确readout数。不得用context-sensitive的0、隐式改1、复制scan rows或改写Bracket模拟另一层repeat。
- 硬件是周期表：一个period一行，行里放32-bit绝对tick时长（或指向一个duration slot）、TTL mask和每路DAC的一条action（hold/edge/ramp，值是字面码或指向一个DAC slot）。Scan slot值就是板子播的绝对值——duration为`1..2^32-1` tick、DAC为`0..1023`码——没有nominal base、delta或tick scale；分辨率恒为一个tick（20 ns），量程恒为整个行计数器（约85 s）。行数、slot数、loop数与嵌套深度由几何决定，Bracket体只占一次行数。
- 同一次application的compiler、wire table、readback、Pulse Editor Run/Sync/Hold/Step与Seamless Dataset共用一个量化结果。Dataset coordinates和run record记录实际played values；若所需分辨率使两个不同authored points坍缩为同一点，必须在碰device前loud拒绝。

### 7.2 Camera

- Occupancy的每一帧可以用自己的calibration：`ArtifactInputSpec(per_frame=True)`让一个artifact输入除了给每帧兜底的那份路径外，再按`"<name>[<frame>]"`（帧从1数，操作员看到的编号）给某几帧各自一份；表单里只有源的一个周期多于一帧时才出现「· frame k」行（帧数先问源的生产者草稿`frames_per_cycle`，没有草稿再问plane最新publication里readout-event角色的Point轴），空行表示该帧读兜底那份；投影给表单的值恰好是表单的键（每行都有值，表单拒绝多一个少一个键），草稿里已命名的帧即使超过当前周期帧数也保留其行以便看见和清掉；finalization把同一文件只解码一次，build拿到`calibration`加`calibration_by_frame`。processor的规矩：每帧读的是同一组site（site_ids必须一致，否则拒绝），各帧用自己的阈值与usable集判定，ROI重定位对每份calibration一起做，帧号超过周期帧数或源没有帧轴却给了逐帧calibration都拒绝；run record只在有某帧自己的calibration时才写`calibration_paths_by_frame`，没有的run记录得和以前一模一样。
- 节点可以给操作员留空的字段自己定默认值：`LogicNodeDescriptor.resolve_defaults(values, resources)`在workspace资源解析之后、草稿投影之前被问，只填空字段，每次finalization重算，草稿本身不写入（默认跟着资源走）。Calibration用它把reference before/readout/reference after三个API字段默认成所选pulse按period顺序的前三个API参数（`zlc_pulse.api_bindings_in_period_order`：按period位置、同period先duration后DAC、delay最后），所以用为此写的模板时默认就是对的；操作员选过的字段不动。

- Same-shot保证采用continuous best-effort，不新增hardware marker或逐cycle arm/fire。
- Camera Measurement只按自己的authored frames-per-cycle/repeat采集并核实际返回cardinality；Camera adapter不解析Pulse window数量，也不以exposure审查Pulse cadence。Adapter的source ordinal只编号实际采到的frames，必须从本次arm的0连续递增。
- qCMOS的ROI、exposure、trigger/readout各由adapter的单一working-point owner管理；未变化字段不得在每次Start整套重写。Measurement冻结设置操作返回的authoritative readback，不再为同一capture额外读取完整property surface；相同exposure/ROI的restart因此不支付冗余sensor reconfiguration。
- qCMOS区分last-successful requested设置与actual working point；量化后的actual不覆盖requested，重复同请求不因此重写。成功setter及arm后的readback形成一份actual，普通working_point读取复用；失败清除请求成功事实，后续setter真正重试。arm后保留真实工作点读回与改变拒绝；不得要求读回之后transfer count仍为零——相机可能已接收本代首帧。本代copied/last count从0开始，首次读取按真实count/newest取回早到帧，负数、倒退、不一致、有限上限及copy-overrun检查不删除。
- Camera adapter独立持续搬运原生frame，read只消费有界FIFO，发布/投影不决定何时从SDK取帧；真实与Virtual都禁止静默丢旧和重新编号。接收容量是内部实现策略，不属于Measurement authoring/Request或用户run参数：Camera沿既有128MiB内部预算，按实际ROI/dtype分配SDK和应用FIFO（各半），另有单帧复制scratch及小型record开销；Waveform沿既有2秒内部容量预算按原生record时长计算。有限目标数、周期帧数与用户科学history/window不等同于接收容量，不增加高级表单。实际容量可记录为只读采集事实。完整周期放不进内部容量明确拒绝，溢出/可观测序号缺口仍报错并带容量/计数/时间信息。正常Stop先停接收，再发布已接收的完整周期；失败不继续发布，残缺周期不补造，Restart才重置该次采集。
- Camera/Waveform真实与Virtual adapter共用中立设备基础层的同一个RecordQueue（由原Waveform队列迁移，不保留两套实现）；容量、FIFO顺序、ordinal检查、失败优先与Stop后待消费尾部由它保证。插件仅产生已独占的原生record并执行自己的SDK start/read/stop；线程归属、触发方式及不可观测硬件gap不能由通用queue猜测，不把具体SDK依赖反向放入Runtime。
- Stop保留已接收尾部与最终Close释放资源是不同边界：硬件/生产线程确已关闭后，共同queue释放未消费记录及失败引用；SDK拒绝Close时仍保留句柄和数据供重试。持续广播设备的本次capture完整性错误只结束该次采集，接收线程继续idle排流，普通Restart重置本次ordinal/clock；真实通信I/O故障仍结束接收并明确要求重开设备，不自动吞错或继续失败capture。
- Pylon同样区分requested/actual；arm模式、restore及gain变化使工作点失效，不能复用旧mode/epoch。SDK frame在result仍有效时直接构造不可变CameraFrameRecord，再Release；不先复制一份随即丢弃的mutable整图。非连续输入直接打包C-order bytes，immutable ownership、frame ordinal及epoch事实不变。
- Camera auto Panel从canonical publication/preview signal建立；signal尚未publish时显示等待状态，但不得用重复device配置、额外generation或固定5秒轮询作为Panel接线条件。
- Scan绑定的是声明的Dataset输出，不以首个value或generation是否已出现判定contract兼容。已配置Panel Fit的参数由同一model词汇提供声明，禁用的输出不提供；无数据时可Start并在现有source owner等待首次真实publication，不创建假值；未显式选择Acquisition logic时不自动启动Camera。首次arrival接入现有有序tap，首绑后继续严格固定generation，停止时退订且不重放旧sealed值。
- Seamless Scan可显式选择一个`Acquisition logic`，只提供声明ready的Measurement，不硬编码Camera。每次Scan Start仅一次`Pulse SAFE → 原Logic Start/Restart → 本次host ready`，其后复用采集运行；manual Continue、device点和repeat均不重启。Seamless没有settle参数、UI、默认值或隐藏等待；设备写入后读取实际值，不比较与设定值相等，不把读回宣称为物理稳定。正常段尾以板端DONE为安全完成事实，不再追加SAFE；Stop/错误才发SAFE。程序只编译/完整load一次，后续Fire直接复用驻留程序，只重置计数、cursor及运行FIFO，不重抄DAC表、不清clock配置；唯一例外是`api:`轴：API参数是程序里的数，某个点改了它，就为该点重新编译并重新load，再fire。源tap/shot写入和进度仍走既有owner。
- ScanPlan自动将manual/device轴稳定移到board轴之前，不因添加顺序拒绝启动；host轴之间及board轴之间的原相对顺序、数值和单位保持不变。编辑器复用同一host-axis分类同步移动现有行，不重建控件、不留第二套排序规则；保存和执行读取同一个规范化Plan。
- Seamless Plan只需列出实际扫描的轴。未列出的Pulse Scan字段只在本次执行副本解除Scan标记，保留API/Config基础来源，按本次API、已保存Config或作者默认值固定并复用普通编译器生成常量；不改作者Pulse、Plan草稿、默认值，不发送重复常量列。仅计划中的字段占硬件scan列，manual/device-only计划也可使用带Scan能力的模板。输出Dataset只增加Plan显式轴，已有显式单点轴保留；固定字段由实际执行Pulse/run record保存，不伪造扫描维度或增加点数。未知显式port仍由同一bind_plan拒绝。
- Seamless允许仅manual/device轴且Pulse无scan slot。资源选择直接复用严格Pulse reader，不得额外要求slot存在；真实slot语法和Plan绑定仍严格校验。普通无slot Pulse只load一次（wire rows为空），每个Host点用Run repeats完成shots_per_point；整轮repeats由原Host循环推进，不重排采样顺序、不伪装成无表hardware scan_repeats。数据只包含真实扫描轴，不补虚构slot/轴；顶层ScanPlan至少有一条真实轴。
- Seamless的第三种host推进轴`api:<field id>`是pulse的一个API参数，像device轴一样由host在两次fire之间推进：把该点的值写进pulse、解析掉API来源、重新编译并load，然后fire一次（shots_per_point仍由Run repeats完成）；坐标就是写入的值，单位是参数声明的单位。只提供不是board slot的API参数，已是slot的由board从一次load里扫，不提供两次。被`api:`轴扫的参数在编辑器的API values表单里保留行但锁定并注明swept by the scan plan，直到该轴移出计划；run的`api_values`文本不携带它。api轴不占设备claim。
- Camera settings provenance属于frame event而不是generation identity：`run_record`在一代内保持不变，frame冻结的小型`event_record`可变化；finite/scan前缀与有界indexed history按实际保留chunks合并epoch ranges，monitor只携带当前event。
- Virtual sequencer按compiled wall cadence逐cycle并支持Stop；每个到达virtual camera的frame event都被采集，不根据Pulse时间或camera exposure私自skip、制造ordinal gap。

### 7.3 Remote

- 无密码、认证、TLS或权限UI。
- Second client默认last-client-wins；旧handler立即失效，takeover前旧active command必须成功Stop/SAFE。旧socket由server主动shutdown并close，不等待旧client再次发话；其退出不对新owner执行SAFE，日志明确标记已被替换的连接。
- 同一client的Stop不排队：其command lane正在等LOAD/FIRE的回复时，client用`open`回复里的cancel token在另一条自己的连接上发一次`cancel`，server执行与takeover/disconnect相同的第一步——command lane旁的SAFE，其stop event打断pending transport——而owner、epoch与command lane都不变；最终SAFE readback仍来自command lane上串行的`safe`。cancel连接一问一答后关闭、从不claim；token不是当前owner的按名拒绝且不碰板子；任何socket始终只有一个线程读写，timeout不因此缩短。
- 完成等待同样走不claim的旁路连接，以现有owner token和本次FIRE command_id绑定；server等待本run既有DONE Event，完成即回复，不占command lane或owner锁等待。有限timeout到期不消费结果；长等待以最长1秒的Event等待续接，不加客户端轮询sleep。Stop/Close唤醒旧Event，下一Fire更换Event；旧等待不能消费或覆盖下一run。两端握手确认completion_notifications，删除旧poll_interval参数。Pulse Editor复用独立后台worker在FIRE返回后立即发起等待，Qt仅通过完成投递更新；状态定时器不再取走有限run报告，Sync不取消本run等待，旧snapshot不得重新显示RUNNING。
- 正常连接无idle timeout；控制进程/socket/连接真正断开时自动SAFE。
- UART auto枚举COM、优先USB VID/PID，并只在word-63 fingerprint匹配后选用；
  显式port把探测限制为该端口。auto探测失败才回退JTAG，显式UART失败则报错。
- 只有server process持hardware transport；不保留假的进程内Interprocess lease。
- Device Manager的Remote把本机一个loaded device公布到bench fabric（generic tunable plane；自带协议的device只公布其server地址）。公布即交出：它在Session的DeviceUse里取该device的command claim——任何本地Logic/command占用该device时按名拒绝且不公布；已公布期间所有本地Logic、command、字段写入与rebuild都按名拒绝，直到Remote撤回。撤回先撤公告再释放claim；unload与Shutdown先撤回全部公布。排他是整个device而不是字段，没有第二张owner表。公布的是loaded device所来自的accepted apparatus，不是表单上未Apply的draft；远端proxy不缓存字段，每次Refresh都经fields RPC取当前完整字段投影（metadata/current/live_write/group/device_limits）。

### 7.4 Host/RTL/build invariants

- UART发送以完整write和本次匹配ACK为完成依据，不在等待ACK前调用Win32按50ms轮询的flush。接收在同一deadline内按本批待回复SEQ集合收齐；驱动队列长度只决定批量read大小，报0仍提交read(1)，单次阻塞以10ms和剩余deadline为界。旧/重复回复不占完成名额，CRC坏帧由同一extractor重同步；有效当前NAK、真正缺包、取消与总deadline仍归原Transport。重发计数只统计实际重发，另保留最后一次重发原因及固定大小读取摘要，不累计历史；缺CRC尾字节的诊断不补造字节、不放宽校验。DoneReport.elapsed_seconds在observer确认终态时固定；command_seconds是其内的命令确认区间，report_delay_seconds另列调用方取报告滞后，不能把GUI/采集等待冒充Pulse执行时间。
- Windows串口的运行期deadline调整只更新COMMTIMEOUTS，不通过pyserial属性setter重新下发波特率/DCB；完整硬件配置只在open完成。仍复用原pyserial读写，超时系统调用成功后才更新其现有属性状态。
- 板载CH340C的通信速率来自现有部署manifest的`uart_baud`，当前取厂商标注连续值460800；host默认、server CLI和生成RTL宏共用这一来源。它不改变Pulse的50MHz执行时钟或几何，修改后必须重新build/program FPGA并重启server；host单独改速率不是协商。显式CLI覆盖只用于匹配另一个已部署的速率。Pulse TCP命令连接两端使用TCP_NODELAY，避免低延迟RPC等待Nagle合包，不把UI取报告滞后计入板子执行时间。
- 扩展板配置为69物理lanes：25个TTL、4组10-bit DAC与4个clock。原19TTL保持物理引脚，只有F13名称由cooling_pgc改shutter_420；新增G16 push_shutter、G15 single_cooling_shutter、J14 cooling_pgc、H13 sweep_trig、K14 push_freq_switch、J16 pgc_1D_freq_switch。R17仍为trig；DAC物理引脚不变，raw lane编号后移6。旧Pulse仅通过显式离线迁移升级，按物理引脚保留状态、绑定和delay；F13旧波形归shutter_420，新TTL默认低，备份原件。正式reader不增加旧别名或自动改接线。
- Physical lane identity与周期表存储分离：每行`[时长][slot选择][25个TTL的mask][每路DAC: 值|slot选择|模式]`打成4个host word，只承载领先25个TTL；4路DAC的action在进入该行的tick生效，edge当拍落码，ramp用Bresenham从carry值（上一条仍在跑的ramp的目标）在该行时长内走到目标、下一行起始tick正好落在目标上。latch clock独立4位CTRL20使能并保留反相board clock、physical-active与SAFE最终锁存时序。R_DELAY只放TTL delay随后4个bus delay，不为DAC数据位保留假延时值。ABI packing版本进入几何fingerprint，host、MemoryTransport、RTL、生成IP几何使用同一布局，旧bitstream明确拒绝，不兼容误解码。
- 扩展版TTL待执行事件FIFO深度32，DAC待执行action FIFO 64；保持4 scan slots、512行周期表、8个loop深4层、2048点每bank、10bit DAC与50MHz。行表和scan表是真BRAM（`blk_mem_gen_rows`/`blk_mem_gen_scan`），取行侧用loop栈走表并预取到一个小FIFO，scan点只在两个bank都驻留时才被取走，取不到就underflow并loud失败——seamless scan不因周期表而变。Fire前按实际repeat/scan/delay校验容量，超限拒绝不丢事件。资源估算以周期表版本本机routed基线校准；估算不是新版本完成布线或硬件验收的证明。用户已授权本机build-only验证资源和时序；program/flash与实验板验收仍需用户另行授权。

- Load前核target ABI、clock、geometry与合法slot rows；delay FIFO capacity和循环接缝在Fire前按本次真实run/scan repeats验证，不先计算一个未请求的1×1执行。相同驻留程序与执行参数复用已验证结论；不把camera exposure或frames-per-cycle反向解释进Pulse program。
- Fire成功返回现有已确认AppliedState；调用方不为相同事实追加snapshot/applied查询。更新重复次数只校验计数并共享已验证program/rows，计数不变复用原对象。Remote以现有LOAD/FIRE确认及已接纳装载状态交付相同事实，只有真正变更才重新装载；同连接未变scan rows沿既有装载引用复用，不新增状态缓存。
- Count必须是合法hardware range内整数，不clamp/wrap。
- Hardware SAFE把TTL拉低、把DAC data置safe码，并让DAC latch strobe继续走足够长以把该safe码真正锁进外部转换器，之后才gate住strobe；只gate而不锁存等于把safe码摆上引脚却永远送不进DAC，模拟输出会保持run的最后一个edge/ramp值。LOAD/FIRE前pins保持safe。
- Public DONE等待delay FIFOs和final DAC latch完成并进入安全态。
- Underflow与engine delay-FIFO overflow sticky且loud；scan point0必须resident。UART CRC/framing/address fault由framed reply与独立LINK_ERROR报告，不能污染engine ERROR。命令使用独立32-bit ID；板端仅保留最后一次命令结果，重试同ID不得重复Fire。SAFE完成安全gating、LOAD完成装载、FIRE被接受后才返回完成/接受ACK；SAFE可抢占mini-loader。普通寄存器写ACK不充当命令完成证明。观察一次连续CTRL读取中的status/cursor；DONE后cursor固定，结果不是逐word读取的“原子快照”。Observer失败保存真实exception和已取得的状态，不虚构双读或board ERROR。新command ABI与server能力在既有握手严格检查。
- 50MHz engine有真实clock/STA constraints。
- Explicit board manifest统一生成host lanes、top mapping和XDC，不靠XDC行序。
- Build delete做真实path containment；program/flash exactly-one target fail closed，默认不自动flash。
- RTL tests自动compile/run并以nonzero failure/逐tickreference证明。

## 8. SLM

### 8.1 Server-owned device

- server默认使用原本可用的DVI exact-raster presenter，不依赖vendor DLL；USB frame memory仅在显式`--transport usb`时使用。
- 和Pulse一样，真实SLM有两个apparatus device type，物理adapter始终只归server process所有：`slm.hamamatsu_x15213`的init参数是server host/port，供另一台机器连接；`slm.hamamatsu_x15213_local`在插着SLM head的bench process内以server自己的表单（transport/profile/wavelength/correction/flips加serve port）启动server，bench自己的leaf再以loopback client接入，因此两种type装入的leaf都是同一remote proxy，DVI/USB输出与profile/correction只有server这一个hardware owner。客户端通过bounded length-prefix、strict-JSON metadata和canonical `float32`相位payload做握手与command代理，不形成第二个hardware owner。普通state读取使用握手cache；apply携带expected command/mapping revision并拒绝stale writer，不确定transport outcome后必须采样真实hardware state再继续。
- SLM proxy无authentication/TLS，只能部署在trusted laboratory LAN，不得暴露到public Internet。
- Initial command state是unknown，只有成功write/display/readback/settle后才known。
- Side effect失败区分known-old、known-new和unknown outcome。
- Correction mutation取得同一DeviceUse claim并冻结mapping revision。
- Profile记录model、serial、wavelength、phase curve来源和settle语义；不新增hash。
- Editor明确区分authoring draft与device command；external Task后旧Send不得静默覆盖。
- Editor的device状态问句（100 ms轮询与每次草稿变化）在Editor自己的串行command executor上问、在Qt线程上显示：一次只有一问在途，command进行中不问——command的交付本身带回它留下的device状态；Qt线程从不等在remote proxy的apply锁后面，远端慢apply只推迟状态行，不冻结event loop。

### 8.2 Context与artifacts

- Target使用稳定`zlc.slm.target` strict格式保存intensity和objective，只是Editor authoring import/export artifact，不是run consumer的第二Target truth。
- Science Context使用稳定`zlc.slm.science-context` strict格式，只持久化run的唯一frozen Target、测量前固化的16-bit circular Pattern模差分、pupil/operator语义参数、system correction引用与command receipt；numeric pupil、operator wavefront和composite phase由同一SLM核心公式重建，不重复保存全尺寸矩阵。16-bit固化发生在Editor Send或Feedback camera shot之前，已测candidate与可加载artifact逐元素一致；Editor Load直接atomic adopt并且不重新solve。Reader只接受当前完整Context，其它root或缺失字段均loud拒绝。
- Command receipt保存USB/profile/wavelength/orientation/correction/outcome。
- `SystemCorrectionArtifact`明确区分pupil phase map与target response map；不得把per-geometry site weights冒充通用wavefront correction。

### 8.3 Solver与Feedback

- Feedback独占run期间只装载一次已解析Pulse，每个candidate只执行一次用户指定的shot batch；正常DONE不追加SAFE，异常/Stop才SAFE。任何board fault或无法证明DONE都不能用camera帧数代替完成证明，也不能同phase自动重拍一整批。已完成candidate仍按现有partial出口保存，不改变控制权重算法。

- 保留sparse WGS-Kim、fixed far-field phase、selected DFT和caller-owned optimizer state。
- Inner solve走到canonical numerical gate，不为省几十毫秒增加physical candidate。
- Feedback mode是leaf-owned显式字段，两个mode：`qcmos_bright_dark`（观测量=每site双高斯拟合的bright_mean−dark_mean，trap越深occupied越暗，plant sign −1）与`qcmos_loading_rate`（观测量=同一拟合阈值判为bright的shot份额及其二项误差，loading随depth上升直到ceiling，plant sign +1）。两个mode的差别只是一条`FeedbackObservable`记录（键名、标签、历史字段名、plant sign）；分半收敛、pooled plant slope、share分配、probe/bracket都通过这条记录读观测量，不知道自己在哪个mode。分类用的是本batch自己拟出的两个population，不用Calibration的阈值——trap光变了荧光就变了；loading到ceiling时plant slope趋零、估计不可信则回落到假定斜率半增益，分半随即判出无可分辨的dispersion而停。Pulse由operator显式选择；camera exposure是独立、可见、可编辑的authored字段，默认`0.1 s`。Task不从Pulse或Calibration猜exposure，也不自动判断Pulse/exposure的科学一致性。
- 当前mode复用canonical Camera Measurement `repeat=N`，每cycle严格一张camera frame；同一逐帧publication经mean reduction实时显示，Feedback只把完整registered Target SiteMap写入该次camera run geometry，不发布第二份camera数据或三帧reference判据。
- Calibration只提供Target→camera注册所需的site centers、BOX半宽和frame坐标几何；Feedback每shot每site的bright/dark计数读出固定为BOX方法——在注册后的site centre按Calibration的BOX半宽对像素求和——因为只有BOX是真实光子计数；无论Calibration默认用哪种model读occupancy，其PSF/matched-filter权重都不进入Feedback frame，未观测site也不借用uniform PSF，没有BOX model的Calibration不能feedback。Feedback不读取其dark/bright/threshold、exposure、photoelectron mode、camera identity或readout working-point provenance。实际camera requested/actual exposure、effective unit与conversion进入本run metadata；saturation只由本次actual raw integer maximum转换到本次effective unit判断。
- 每个site仅使用本candidate完整一批authored shots经Calibration读出契约得到的site信号选择单高斯或双高斯；完整batch的受约束双高斯数值有效、满足基本分量/间距条件且full-data ΔBIC>10（决定性证据）时，`bright_mean-dark_mean`才是observable。这一判定是`fit_bimodal`自己的`decisive`（`ok`=两态分得开且都有人口，`decisive`=再加ΔBIC>10），校准给参考帧打标签用的是同一个`decisive`，而已知会load的site定读出阈值只要`ok`（六十发重叠的两个群体ΔBIC为负，它们的交点仍是最好的阈值）。候选对按似然减一项宽度比代价排序：代价2(ln r)²是一撮碰巧挤在一起的样本付不起、一个群体付得起的（它只防塌缩，不是边界，bright散粒噪声比dark读出噪声在qCMOS上就是十倍量级）；分量另有份额与宽度两个地板；拟合正常但不满足为single（未load），数值/采集失败为invalid。完全常量的样本没有任何合法切分，必须`ok=False/decisive=False`，不得以虚构的unit spread制造第二population；所有population width/separation先减各自真实样本作锚点，常量宽度严格为零、separation为未定义。ΔBIC>0曾把一个被拆成两半、相距1.7σ的单高斯当成loaded site（contrast 10.9，uniformity ratio读到116）。
- Controller保存每site的归一化Target share、正式double历史、bracket（方向＋最近一次single的share＋最近一次loaded的share，最新观测优先，因为loading edge随全阵漂移）与loading-edge标记。没有任何证据（无bracket、从未loaded）的single用用户`probe_factors`做一次两侧诊断，verdict给出方向：哪一侧loaded就往哪一侧，两侧都loaded取更近的，两侧都不loaded则往最深的dark share外推一个clamp；probe过的site永不再probe。有方向的dark site每candidate要一步：bracket宽于分辨率（2%，即识别excitation幅度）时向loaded share几何二分且不超过clamp，否则沿方向爬一个分辨率——从不整clamp外推（整clamp由全部loaded sites同时出资，4%余量的阵列被压暗29→23→26→25→32→22）。dark site要的share在share空间由本轮loop步不向上的loaded sites（loop送它向下、或对它无所求）出资（要减的dark site则把share交给loop步不向下的loaded sites；hold的site两边都不参与），每个出资site每candidate最多让出一个分辨率、不越过自己的bracket边界，总功率精确守恒；不足时按比例缩减、不反向。每个site的实际绝对份额只朝controller给它的方向走或不动：loop步向上的site绝不被出资压低，边界只能拦住一步、不能把它掉头，没有方向的site（invalid无verdict、等probe、hold）绝对份额分毫不动——旧的「每个loaded site乘同一公共因子」曾把loop刚送上去1%的site压低2%，公共因子无根时总量本身移动、所有hold的site随之改变份额。识别excitation（±2%、正负平衡的log图样）在被激励site子集内平移一个公共对数使其总份额不变，未激励site（invalid、unobservable、loading ramp上）的绝对份额不变；记录并用于plant slope的是实际施加的平移后图样。dark在loaded bound之外超过一个分辨率＝该bound已失效，丢弃并继续爬；方向只由probe verdict改变（按dark翻转方向曾让三个site在两个share间永久乒乓）。bright fraction是loading-margin观测量：低于全阵中位数一半的loaded site处于loading ramp上，其向浅的步被hold（`hold_loading_edge`）、不出资、不被识别excitation扰动。只有正式double更新使用`feedback_gain`（loop gain，除以实测plant slope幅度；未可信时假设单位slope半增益）；invalid保持实际份额；Diagnostic probe不进入formal history、best candidate或反馈指标。
- mode、板子播放的编译program（按`CompiledProgram.digest`比较；Pulse路径相同不是证据，同一文件会被原地编辑）、exposure或任一控制参数变化时丢弃prior response state；用户要求的dark增幅若在固定总功率下不可行，静默缩到本轮最大可行值。
- 默认每candidate为100 shots、12次formal update；每个phase严格一批shots，下一批前必须确认不同phase。`probe_combined`是正式update并计入上限，0.5/2等diagnostic probe candidates不计；每个site至多一次probe episode，candidate容量为`1 + max_updates × (1 + len(probe_factors))`。运行终止于连续3个formal candidate的split-half真方差与0不可分辨，或达到上限；split-half covariance在每半内先减一个真实log样本再中心化，完全均匀数组必须得到严格`variance=error=0`，不得让非二进制常量的舍入残差伪装成dispersion。选择split-half方差加其标准误最小的完整测量candidate（并列取最新；无全site结果时取最可观测者），没有内置ratio停止阈值。Stop保留最佳已测candidate，置信区间只作记录，不触发额外采集。
- Feedback取得SLM后自己apply并确认frozen Science Context phase，并在shot前发布该phase；Context receipt是provenance，不要求operator事先Send/Save。normal terminal、Stop与真实failure都封存同一选择——最佳已完整测量candidate——到SLM与`final/`，summary带failure的错误；只有封存本身写不出时才恢复Context起始phase。
- Feedback run只保存精选candidate数据：stable site table、每candidate BOX shot×site samples、fit/classification、Target/control weights、update action、metrics、phase-change fact和command receipt；不保存raw camera frames。每个operator-visible candidate Context包含其可加载phase，final仍只有唯一selected Science Context。
- Feedback candidate是operator-visible、可直接Load Science/Send SLM/作为下一run Science Context的完整Context，保存在`candidates/candidate-XXXX.npz`；内部BOX/fit/decision数组单独保存在`data/measurements/measurement-XXXX.npz`，不得再用candidate命名冒充可加载Context。加载candidate仍使用既有Science Context输入且完全由operator手动选择；它只确定新run的起始光场，新run从candidate 1开始并执行本次authored `max_updates`，不自动查找旧run、续编号或继承旧run的update预算。
- Feedback summary同时提供机器可读JSON和人读文本，明确initial/selected uniformity、confidence、observable sites、common-site total brightness、selected candidate、Stop/failure与rollback。
- Feedback重要summary图固定为`uniformity_history`、`site_signal_evolution`、`weight_evolution`、`selected_site_histograms`、`camera_initial_selected`和`phase_initial_selected`；每个完整candidate（含diagnostic probe）及selected Histogram都通过正式Figure API携带该candidate完整shots实际拟出的Gaussian分量和threshold，不在binned histogram上另跑一个优化器。无效site明确无模型/无threshold，仍保存其数据。classifier target中省略组件表示自动估计，显式null表示没有模型；value:null表示没有threshold，不以假数字代替，保存/重开保留这个区别。图只作run artifact，不形成Monitor preview；正常或Stop终态只写一个final Science Context。
- Feedback自动preview固定为带编号site map的实时Camera Measurement mean reduction、observable uniformity、site signal evolution与Target share evolution；phase仍发布且保存最终Figure，但不自动占用Monitor panel。
- Task preview只冻结运行中的signal/overlay/cell kind/semantic与publisher wiring；Selector、viewport、hover、line lock等Panel interaction始终由全局Selector toggle控制，Calibration、Feedback与普通Panel行为一致。Task锁只阻止selection反向改写正在运行的producer draft，不阻止本地交互状态。
- Task到达completed/stopped/failed terminal时移除该run自动创建的preview panels；用户手工创建的Panels不受影响。Panel header使用紧凑Setting与紧邻的`×`；`×`单击进入红色确认态，系统double-click interval内第二击才删除，超时恢复中性灰。
- Sparse-only contract明确；dense Gaussian/Flat Top先修算法定义和early stop，再profile CPU，不引GPU。

## 9. Calibration、Scan与Simulation

- Calibration保持既有科学流程、当前artifact和三帧preview。
- Calibration site detection只有两条并列证据：相邻reference frames的空间带通差值，以及全部reference frames的空间带通average。一个明显相邻帧变化即可保留single-loading possible site；steady/high-loading site由average保留。两条路径使用同一个authored `detection_sigma`下限并按全图/transition数量提高family-wise bar；site identity始终取完整average的局部峰。不得按奇偶/half分帧，不得用split consistency、全局saddle heuristic或单帧亮度bar（「该处是否曾在某一帧亮过」）否决已经成立的证据：亮邻居带通暗环里的弱trap在任何单帧都不高于背景，它仍是trap。difference证据只在变化本身成峰处成立：按求和后的变化幅度与外一圈（spot尺度）比较，trap自己的变化在邻居暗环之上再叠一个峰，而未加载lattice cell的中心是邻居变化的凹底（变化向制造它的邻居方向增长），不是site。
- Calibration可由operator显式开启detected-site review：采集与site detection都只执行一次；检测完成后由同一run的短期companion producer发布reference average与candidate SiteMap，TaskConsole允许单点或框选排除高阶衍射/ghost site。确认后只用保留站点构造最终SiteMap并执行一次全部下游拟合；不重新采集、不重新检测、不二次确认。窗口壳、搜索、site checkbox、scroll、status与buttons全部由`zlc_ui` Fluent view拥有，`zlc_plot`只拥有Image surface的point/rectangle gesture与overlay，Workbench只连接两者。最终报告同时保存candidate/excluded/final identity映射和可由FigureViewer重开的`site_review` Figure/PNG；不开启时外部行为与artifact集合不变。
- 允许不改变外部行为的dependency解耦、明确corruption修复和内存优化。
- Calibration只产生与SLM无关的camera/readout artifact，UI和Task都不接受Science Context。SLM Feedback在同时拿到Calibration与Context后做Target X/Y→camera X/Y直接正向注册，并为未观测site生成predicted BOX；不枚举翻转、旋转或轴交换。
- BOX model仍为Calibration/Occupancy持久化自己的readout事实；Feedback只取BOX geometry。未观测Target site由注册产生predicted BOX，并与实测site一起接受本次run的双高斯估计，不伪造Calibration dark/bright样本。
- BOX仅在每个实际区域累加为float64，不先把整帧转为float64。Camera组装cycle/repeat时只stack一次；不可变snapshot仍独占输入。Derive的numeric count只归约validity，其他归约只分配实际需要的累加量，保留全domain/空组/单位语义。成功Gaussian threshold不计算弃置的Empirical答案；Empirical模式仍保留Gaussian拟合用于独立理论报告。
- Calibration threshold method保留operator选择并默认`gaussian`：每个site/readout model只用全部finite short-shot signal做无标签双Gaussian mixture fit，按均值识别低/高分量并保留fit得到的population weights；threshold是两条实际加权分量曲线`w_dark N_dark(t)=w_bright N_bright(t)`在两均值之间、令拟合population总误判最小的解析交点。reference真实标签不得进入Gaussian参数、权重或threshold；只允许用于Empirical threshold及最终actual fidelity。Gaussian参数、population或相关解析根无效时该site使用全部有效labelled samples上令实际总正确率最大的empirical threshold；operator显式选择`empirical`时所有site都走该路径。Histogram竖线始终是最终写入Calibration并由`detect()`使用的threshold；Gaussian曲线必须复用Calibration保存的同一组参数与权重，不得由Plot二次拟合，fallback site不得伪造理论曲线。报告分别保存最终threshold在全部有效真实数据上的overall actual fidelity（另存dark/bright conditional值），以及Gaussian threshold按其fit population weights积分得到的theoretical fidelity；fit失败site没有theoretical值。
- Calibration只使用稳定`format="zlc.calibration.readout"`，无数字版本；reader只接受当前完整grammar，alternate root或缺失统计均loud拒绝。
- Calibration run保存final JSON、summary JSON/text及精选报告图；每张报告图都有可由FigureViewer重开的typed Figure NPZ，PNG仅为preview。默认不保存全部raw frames；operator显式请求时才保存采样数据。
- Scan正常完成、Stop或失败都restore pre-run device数值与单位：第一次移动该knob前从device读取原始数值/当前单位对及同单位bounds，确认可恢复后才写；不从dBm反算原Vpp读数。restore复用同一单位化写入，其拒绝在成功的run中就是run的失败、在失败的run中附注在原错误上，并继续恢复其他knob，不得被SAFE成功掩盖。
- Seamless写值及restore不比较设定值与回读值是否相等，也不设浮点容差。设备的实际tune返回值仍保留，仅核numeric/finite契约；设备自身拒绝与异常照常传播。扫描轴是用户设定坐标，不用设备量化后的回读替换它。
- ScanAxis的`unit`就是其`values`的单位，Plan、Layout、Seamless Dataset及run record保持该单位：135→247mVpp/10点直接存135…247与mVpp。设备轴的数值与单位原样传到设备层，bind和编辑器范围也取设备在所选单位下的投影；不在Scan先转dBm。Pulse编译仍在原边界换算，量化后的坐标转回同一author unit。切单位逐点转换已有列以保持物理扫描，显式编辑范围/点数才在选中单位内等分。
- Scan每轴可切换Range/Values，两套输入独立保存：authoring entry的`values`始终属于Range（包括已有非等距精确列表），`mode`默认`range`，独立`value_text`默认空字符串。Values只接收按顺序的逗号数值，不显示Points控件、不从Range自动填充；切换只隐藏原控件，不销毁或互相回填。Selector只更新Range及其原点数，即使当前隐藏在Values模式，也不得改Values文本。两套输入共用一个`unit`，切单位时分别换算各自已有数值，空Values仍为空。Layout保存raw plan以保留两套输入；Start仅将当前mode编译为原有不可变`ScanAxis(port, values, unit)`，空或非法的当前Values按行/port拒绝，不改变已冻结运行。执行Plan的`to_tree`仍只写实际port/values/unit，不增加第二套执行计划。
- 单位化设备边界由既有`tune_in_unit`、`read_tunable_in_unit`、`convert_tunable_value`共同承担，字段ID和author unit不变。RF的字段/单位投影只读有界会话latest facts；显式`refresh_tunable_fields`才更新设备真实读数，不保存epoch历史。常规carrier/FSK hop/幅度设置只发本字段设置与实际值查询，返回量化后的真实结果，不作相等检查；DG4000 hop使用`:SOURceN:MOD:FSKey:FREQuency`，不得顺带切换modulation type或打开modulation。carrier或hop频率改变都不额外读取幅度并自动回退；受影响幅度current/range失效，不能显示旧值为当前事实。epoch复用已知before与actual，不增加写前读。仅显示换单位不写设备；真正Apply才必要时切native UNIT并发送数值。未知负载/波形只在真实转换需要时查询，不能每个值重读或永久缓存已失效范围。Control Open/Refresh复用明确刷新入口，Apply及单位投影不整机查询；非RF adapter继续原接口，不多调用一次values。多channel共享功率policy仍不能冒充唯一电压换算。
- SimulationWorld保持一个类和一个state owner，不拆层。
- 真实camera site centers与固定成像条件未变时复用现有PSF spots；phase/intensity改变仍重新传播trap field，不裁PSF物理尾部、不替换像差模型、不重设随机序列。
- SimulationWorld的物理site只有当前SLM phase经共同pupil illumination、共同low-order wavefront aberration和FFT得到的dominant local peaks这一份动态roster；trap位置、强度、occupancy与Camera位置不得再拆成nominal/extra双状态。所有peaks经过同一个Fourier→camera affine；fluorescence imaging使用一个由共同imaging pupil/aberration生成的shared非对称PSF，不存在逐site随机gain/ellipse/angle/skew。Probe为红失谐，正的trap light-shift参数只把detuning进一步推红，因此occupied bright-dark随trap depth单调下降；loading probability随depth上升。Camera shot真实混合dark/bright population，Feedback不得读取hidden depth/occupancy truth。
- 默认plant的全部不均匀度必须来自FFT前同一个固定pupil amplitude/wavefront phase；该world wavefront与SLM command、Target和grid完全独立，并在每次propagation中始终相加。不得使用grid-resonant phase、target-specific correction或far-field site/field gain。默认nominal depth固定为520 µK；固定20 µK cooling温度下，低于500 µK的trap不load，超过阈值后按一个cooling-temperature尺度指数趋近全局loading ceiling。因nominal本身贴近实验loading edge，普通光学不均匀在不同grid中都会让至少约10% sites不可见，不得按某个grid反推nominal或由测试手改Target weight；`bright-dark`继续由现有probe参数决定。
- Apparatus root `simulation`是image/grid geometry、seed与profile的唯一持久化owner；virtual qCMOS只声明camera事实并消费world image geometry，virtual MOT保持独立的camera geometry。非当前grammar必须loud拒绝且不能形成第二owner。
- Simulation参数在init前通过单一API/immutable config确定；workspace-relative profile必须在任何device factory前解析且保持在workspace内，Device Manager Init不运行时改写。
- Tests使用config override，不修改public mutable world attributes；hidden truth不泄漏给production算法。

## 10. Deployment、Evidence与Docs

- `warm_numba_cache`沿原kernel discovery/cache owner按模块源码变更或机器码缺失选择预热组：render包含raster与3D，fit包含compiled solver与radial。组内保留完整production dtype/layout样本；无关组不运行。marker相同时仍检查实际缓存，Numba源码stamp/CPU/signature判定不被覆盖；日志分别显示新编译与磁盘加载。修改同一kernel源文件仍受Numba的整文件失效规则影响，跨文件依赖也不伪装成已经自动追踪。

- 一个可安装`zou-lab-control` distribution，bootstrap package为`zou_lab_control`；内部八层不独立发wheel或维护版本。
- 所有checkout launcher（包括FPGA build/program与resource estimate）通过同一个Python
  环境owner激活当前tree的bootstrap；安装器是唯一installed-only路径。installed wheel在
  checkout外从distribution metadata解析同一组commands/layers，不保留第二入口名。
- 根`pyproject.toml`是唯一product manifest，`constraints.txt`是唯一resolved dependency surface，`zlc`是唯一console entry并从manifest加载commands/layers/evidence。
- Wheel必须包含bootstrap、八层、Calibration/Scan templates、SLM profile、Plot font及完整有效FPGA RTL/XDC/Tcl assets；installed environment check按distribution RECORD验证归属。
- 正式evidence lanes：software、gui_offscreen、virtual_vertical、notebook_offline、real_screen和hardware runbooks。
- Evidence使用每个既有文件/层的普通pytest进程；TaskConsole用例中明确需要的fresh-process场景由用例自身隔离，不在外层再collect并给每个item各开一层Python。
- Mock/virtual/offscreen证据不得冒充真hardware/optical acceptance。
- Root Architecture只保存目标不变量；Implementation Plan只保存当前实现状态和最新证据。
- 活文档保持current-only，不在尾部追加change log或修补记录。

## 11. 当前实现状态

当前tree正在按上述不变量完成无版本strict persistence与统一TaskRun收口；当前验证状态见`IMPLEMENTATION_PLAN.md`。任何未执行的real-screen/hardware/optical步骤必须继续标为`UNEXECUTED`。
