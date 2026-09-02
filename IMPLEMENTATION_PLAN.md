# ZLC — Current Implementation Status

更新时间：2026-08-25

状态：`PLOT/RUNTIME/WORKBENCH CUT COMPLETE / OVERALL GOAL IN PROGRESS`

本文只记录当前tree已经完成的产品切面、最新验证和有效的实验机边界。
最终产品不变量见`ARCHITECTURE_DESIGN.md`。只有从当前tree重新取得的证据可以作为
完成证明。

## 1. 当前实施范围

- 所有项目内部持久化格式、Dataset contract和artifact contract改为稳定语义名；
  Figure、Calibration、Pulse、Target和Science Context不带数字版本。
- Reader只接受当前完整grammar；现有workspace不转换，文件可直接不受当前reader支持。
- 产品bootstrap为`zou_lab_control`；根`pyproject.toml`仍是唯一distribution manifest，
  八个`zlc_*`目录只是同一distribution的依赖边界。
- Science Context只保存16-bit circular Pattern模差分、Target和语义metadata；pupil/operator/composite均由SLM核心公式重建，Editor/Feedback在Send或shot前先固化Pattern。X15213 1024×1272 5×7同数据实测旧布局12.524→0.227 MiB（-98.19%），保存0.051 s、完整读取0.179 s；固化最大误差4.82e-5 rad，保存前后Pattern、composite phase和8-bit phase code逐元素一致，固化中位15.57 ms。
- Hosted Task只在NodeHost worker真正Start时分配run directory，并在任何不可逆工作前
  原子建立`run.json`。进度、artifact registration、Stop、failure和terminal result都写回
  同一个lifecycle truth。
- Runtime不自动dump live/intermediate Dataset。Calibration、Temperature和SLM Feedback
  由各自domain owner保存精选artifact，并通过ExecutionContext注册已完成文件。
- Figure NPZ是primary typed artifact，PNG只是preview。Figure保存exact Plot recipe、overlay、
  viewport、selectors、facet focus和causal lineage graph；FigureViewer与TaskConsole使用同一个Plot host路径。
- Figure NPZ唯一writer现按member做有界1MiB可压缩性probe：结构化/平滑数据继续Deflate，低收益大camera数组使用标准ZIP Stored。20×1200×1920 uint16 noise的Panel Save实测`4.23s→0.73s`，archive阶段`3.63s→0.18s`；92.16MB原始数据原压至79.12MB，现为92.16MB，明确以13MB换约3.45s。平滑1200×1920 float32仍Deflate为2.98MB、总时约0.33s。
- Plot axis/semantic identity已收口为`AxisRef(domain, axis_id)`稳定key；scope只接受tagged
  latest或tagged typed coordinate value，不再让display label或裸文本控制字进入truth。
- Fate Setting不再预跑candidate render/layout feasibility：所有axis始终列出plot kind声明的全部roles；
  64-cell等容量限制只在真实replace/layout transaction执行。旧semantic probe、cache和kind validate
  wrapper已删除，schema vocabulary不再随size、DPR或renderer可用性改变。
- Runtime是唯一跨publication history owner；active leases的signal-level event/indexed表示变化
  通过presentation epoch使同signal全部Panel重新投影，不增加scientific publication/revision。
  Runtime内部绝对ordinal在materialize时统一转换为以最新为0的相对primary-index；Plot与
  Workbench只把它当普通AxisRef，不自动scope或建立history专用interaction路径。
- `Occupancy Agreement`是普通Occupancy的纯数据下游：source picker选择`counts`，Runtime从同一原子publication交付其`occupied` sibling；默认frame `0/1/2`但三项均可编辑且可重复。一致的首/末occupancy保留共同bool和中间counts，不一致或任一所需值invalid则只通过Dataset validity标为invalid；不重新读取camera/calibration，不重新提取counts或分类，也不携带overlay知识。
- Panel window demand在authored state接受时先于Plot render同步；最后lease的`10→1`在调用
  返回前释放并切回event表示。当前host的Focus/Area/Crosshair按同generation与accepted轴词汇
  接受，owner落后一版不得否决indexed front，Facet只忽略其自身focus cell这一层subject差异。
- Panel title shape现由PanelCard独立accepted-data projection持有，不再从Setting parameter surface
  读取；ROI selector导致派生Image/Histogram换schema时，每次surface accept都直接重投影三段shape
  strip，PanelState/control相等不再阻止。FacetGrid新增display参数
  `facet_fit_parameter`，Workbench在fit model存在时把它放在Fit expression下方、通过通用
  `edit_section=display`写回display owner；普通下拉choices为`Model headline`加当前fit model
  parameters。切换仅重画cell annotation不重fit，model不兼容时回到`Model headline`。
- Rolling history投影在现有`DataView`内一次归约：规则repeat tensor直接沿非保留轴
  reduction，其余repeat与primary-index分别按`(repeat, group)`、`(source index, group)`
  联合bucket；旧`O(history × samples)`逐history mask循环已删除，Runtime history owner不变。
  2.04M MEAN compute为204.124→9.719 ms，真实Windows Rolling Host中位为
  264.37→58.85 ms、P90为286.34→66.89 ms；70组reduction/validity/group矩阵满足
  既有浮点数值等价与结构精确contract，聚焦回归63项通过。
- Facet/Single规则tensor投影已收敛到同一retained-axis reduction：一次保留`facet/x/y/group`真实tensor axes、一次归约其它轴，Curve/Image只包装不同payload；Histogram继续共用其批量分箱terminal。Curve/Image/Fit/SEM的native raster快路保留，并继续以完整差异像素而非阈值子集评价其Agg接近度；不得通过回退Agg把差异人为归零。RegularImage即使有完整warm seed也保留cold proxy竞争；Board的active-fit staging保持不变。
- FacetGrid现允许facet fate为空：DataView发布一个`Facet 1`完整cell，不创建phantom axis；真实facet被归约/移走时仍可画、fit和保存，重新赋予Facet fate后恢复普通多cell路径。
- 当前Render coherence Goal按以下顺序根修，全部在现有owner内完成且允许证据驱动调整实现细节：
  1. clim move合并为`candidate+clim mutation+compose+front`一次原子preview；
  2. indexed history旧publication改为正常expired cancellation，Panel保留最后完整front与Fit/Setting vocabulary；Edit拆开`data advanced`和真正configuration incompatibility，并让PanelState/frozen target原子同步；
  3. Image保留框架唯一固定square display frame，以x/y cell pitch归一canonical坐标并在frame内绘制square-cell footprint；非方阵数据居中letterbox，数据extent不变；canonical scan coordinate继续提供ticks、selector、overlay和fit，zoom按相同whole-cell span且不改layout；
  4. Single/Facet/Focus共用同一kind-prepared cell state，native/Agg只是两个consumer，删除`curve:native`/`facet:*_native`承担的平行science/presentation truth和无artist fallback空洞；
  5. Curve SEM保留独立stem/cap，Matplotlib artist继续拥有style/topology，native consumer读取其alpha/linewidth/capsize并以subpixel coverage绘制；删除整数列min/max envelope语义；公共ylim包含SEM bounds；Fit source line/scatter模式不再靠搜索现存Line2D决定；
  6. overview Fit文字恢复公共MathText，删除plain glyph parser/atlas及其warm signatures；
  7. 使用`workspace/layout.json`的50×50、4:1 scan step真实链验收square cells、固定zoom box、partial scan Curve持续显示、Fit立即line→scatter、history expiration不清UI、Edit Fit/Refresh/Save，并重新跑真实四Panel性能和全部像素差异矩阵。
- Render coherence后的正确语义性能cut已在同一真实Windows DPR3、2×2、MOT 40-shot四Panel链验证：Curve critical path从本cut前`166.6/185.8/202.1 ms`（P50/P90/max）降至重复真实run的`85.77–127.28/92.93–143.24 ms`，best`76.19–95.03 ms`；Windows混合核调度产生run间波动，帧序列无单调恶化，不用单次较好run冒充稳定值。Curve/SEM直接消费共享prepared state，grouped line+SEM一次批量transform；后者isolated render `24.48→12.56 ms`且与通用路径0差异像素。Fit动态数值不再触发第二次MathText grammar layout，overview以Matplotlib自己的最终MathText mask批量compose，完整公式、下标和抗锯齿保留。早期全局4/8-thread敏感性只是中间证据，最终process-pool/worker-team裁决见下；全局fit/render gate实测更慢并已删除。旧`7dce795`的`67.78 ms`依赖错误的SEM列envelope与plain glyph atlas，不能作为正确画面的等价下界。Image-Fit重复真实run为`112.77–139.80/156.05–163.01 ms`（P50/P90，best`86.88–109.23 ms`）；isolated正确cold-proxy＋full refinement约`28 ms fit + 19 ms render`，不得为复现旧`80.16 ms`而恢复warm跳过cold。
- Clim gesture现与live frame始终消费同一`image:prepared`并保持colorbar为唯一提交态chrome：相同clim press、move及中途live revision的colorbar区域逐像素不变，release才写最终ticks/state。真实TaskConsole MOT、DPR3、2×2、live clim手势中，steady picture gap由`18.12/21.32/28.82`降至`12.80/15.29/18.20 ms`（P50/P90/max），first move `25.32→15.05 ms`，720 moves的实际回答`558→719`；快速frame与完整compose逐像素一致。
- 四Panel剩余争抢的根因不是Panel线程数量，而是此前把整个进程Numba pool缩成4：四个独立Panel只能轮候同一小pool。现保留16-logical-core process pool，每个RasterHost/PlotSession analysis worker启动时mask为4；40个互不重叠SEM lanes仅在该kernel临时用8并恢复。真实DPR3 MOT四Panel的Curve-Fit为`84.64/93.44/101.82 ms`、Image-Fit为`92.45/97.66/102.68 ms`（P50/P90/max），均0 stalls；Image相对前一正确语义run的`112.77–139.80 ms`显著下降。把Image fit单独提到8 threads实测反而为`96.89/104.12/113.50 ms`，全局fit/render gate也更慢，二者均不保留。
- Panel Edit/Setting性能cut在同一真实Windows Camera Facet链上完成：Direct Producer不再嵌套
  LogicEditor而只打开已有Logic tab；Qt owner在Host首次render前传入screen DPR；正常已settle
  Edit首开`update_projection 3→1`、`refresh_panel_editor 3→0`、Form reconcile `19→4`、
  独立Form refresh `4→0`、renderer present `2→1`，Editor对象树由598/337/136个
  QObject/Widget/Layout降为500/280/115。相同FormSpec且Widget已显示目标值时Card只adopt metadata，
  relim因果front由138/135 ms降为113/99 ms。无方法hook Edit click P50由522.3降至487.6 ms，
  P95尾部约678→522 ms；尚未达到100 ms，剩余首轮4 Form约93–112 ms、add-tab约84–121 ms及
  单次正确DPR render约84–104 ms已明确记录为下一性能cut，不以已完成项掩盖。最终直接UI文件
  42项、跨层重点8项与Workbench相关六文件152项均通过；此前重负载组合中一次10秒settle长尾
  在最终tree文件级复跑消失，目标用例另连续3/3通过。
- Fluent Combo根修删除collapsed实例的逐控件QSS、反复全choice `sizeHint`扫描及所有首开popup子树；
  flat/tree共用一个owned model，第一次真实展开才分别建立唯一ListView/TreeView和FluentPopup。
  相对上一Edit cut，无hook三轮click P50 `487.6→287.2 ms`、首次Paint `414.8→254.8 ms`、
  interactive front `580.1→401.3 ms`；详细trace中Form `108.2→15.4 ms`、add-tab
  `116.8→46.7 ms`，对象树QObject/Widget/Layout `500/280/115→248/142/57`。首次展开成本没有
  隐藏：131-choice flat cold/warm P50约`23.5/6.6 ms`，Tree约`27.8/7.7 ms`；将首次Popup
  加回Editor首次Paint后仍比旧链快约135–142 ms。collapsed实屏抓图只在306/4203360像素的
  圆角抗锯齿边缘不同，popup抓图pixel-exact。
- Fluent Combo popup宽度根修：TaskConsole Add chooser恰有13项并首次触发vertical bar；旧实屏
  `popup/view/viewport/content=242/242/226/228 px`，手算native scrollbar chrome少2 px，产生
  `horizontal maximum=2`。现删除flat文字/QSS手算、Tree indentation手算及`_desired_popup_width`
  三个重复owner；唯一popup在最终高度后用delegate column hint与真实viewport chrome最多四轮收敛。
  同一实屏为`244/244/228/228 px`、horizontal maximum `0`、vertical正常；12/13行、Tree展开折叠、
  open model变宽及screen-cap真overflow均由现有Combo smoke覆盖，横向滚动没有被禁用或隐藏。
- PanelState只保存authored target；Live、Frozen和FigureViewer都以Plot成功返回的完整accepted
  `DisplayDescription`判断当前pixels、能力与交互。Selector/viewport observation携exact Dataset
  generation+revision，TaskConsole Console核对后才持久化、镜像或发布derivation。
- 大轴Scope不再受256项popup上限控制：Plot description携惰性真实coordinate domain，Setting/Edit
  共用的Fluent cycle choice只显示一个Scope action，focused wheel写回原有tagged scope fate；1024坐标
  轴的popup仍只有普通fate加一行Scope，未聚焦滚轮不改值。

## 2. 当前代码收口状态

### 2.1 Strict formats

- Figure根为`zlc.figure`，无numeric version；reader为strict current-only。
- Pulse根为`zlc.pulse`。
- Calibration根为`zlc.calibration.readout`。
- Target使用`zlc.slm.target`，Science Context使用`zlc.slm.science-context`。
- 纯内部Signal/Dataset/artifact contract只使用无数字后缀的稳定语义名。
- 外部SDK API identity、发行semver、FPGA hardware layout fingerprint与跨进程真实协议
  identity不属于本次删除范围。

### 2.2 TaskRun

- Run directory由Runtime在actual Start边界分配；Editor打开、draft validation或build失败
  不创建run。
- `run.json`包含run identity、normalized inputs、running/progress/terminal状态、artifact
  inventory和failure信息，并采用原子替换。
- Summary位于run根，domain final位于`final/`，Figure pair位于`figures/`，精选
  candidate/site数据位于`data/`。Figure NPZ contract为`zlc.figure`；同stem PNG只登记为preview。
- Task完成前必须注册所有声明的final artifacts；未注册、文件缺失或路径越出run root均失败。
- Stop和failure保留run directory与已注册artifact；异常进程退出留下非terminal记录，
  不清理、不伪装成功。

### 2.3 Figure与Viewer

- 公共Figure API严格编码/解码PlotSpec、parameters、size、viewport、selectors、facet focus、classifier、fit与
  typed image overlay；archive先发布，preview后渲染。
- Panel Save只是公共Figure API的adapter，不再维护第二套writer或restore grammar。
- FigureViewer把archive typed Dataset发布为sealed Runtime signals，默认panel从archive exact recipe恢复，且不按shape推断plot kind；保存spec的`kind + cell_kind`在Panel创建前经同一个catalog identity owner解析，semantic vocabulary随后才投影。Add Panel只建立空的fixed-kind `panel-N`，Signal/ROI/Fit派生及后续compose全部走与TaskConsole相同的ConsolePresenter、SelectionBridge和Plot host，不再保留static panel owner。静态host在Bridge订阅前已有accepted fit时，Fit subscription只replay该immutable FitEvent，不重复solve/render；因此ROI与Fit参数都继续发布给后续Panel。
- FigureViewer与TaskConsole Live/Frozen使用同一个accepted PlotSpec、parameter、selector/
  viewport capability contract；Viewer semantic edit同样只在host accept后更新surface，文件选择默认定位workspace当天data目录。
- `board.commit`首次接受Panel host后在同一owner turn幂等挂载Selection/Fit Bridge；真实Qt首个drag/release已验证可立即发布ROI，不等待下一display beat。
- SLM Feedback camera preview第一轮后停更的根因是`holds_live_revision`只识别裸`OwnedSnapshot`，带site overlay的`ImageFrame`把第二generation的revision 10误判为旧run的`10<=10`并cancel。现统一解包snapshot；真实两轮Camera Measurement→Panel从generation A前进至B、同host复用、无busy/error，Plot/Workbench seam各有回归。
- Lineage保存root、event nodes和direct parent IDs；Viewer验证引用、reachability和cycle后
  投影为tree。
- 显式消费Dataset的Measurement worker在每次保留值时将exact source publication随同一次Runtime commit提交；Scan Flow因此从Scan event沿真实parent回到Measurement/Processor，Save与Viewer不得按signal名查latest。Device tab从node run record显示baseline working point，并另列event ranges实际引用的active override epoch。

### 2.4 Domain Task artifacts

- Calibration：每run一个folder，final Calibration JSON、summary JSON/text、精选报告Figure
  NPZ与PNG。默认不保存全部raw frames。
- Calibration threshold method默认Gaussian、可显式选Empirical。Gaussian模式对每site全部finite short-shot values做无标签双Gaussian mixture fit，并在两均值间解析求拟合population-weighted分量曲线交点；真实reference labels不参与Gaussian fit/weight/threshold。fit或交点无效site才用全部有效labelled samples上最大化overall correct fraction的Empirical fallback；Empirical模式全部使用该empirical路径。Histogram线是最终classifier threshold，Gaussian曲线直接携带Calibration同一组分量而不在Plot二次拟合；fallback只有最终线而无伪造理论曲线。`actual_fidelity`是最终threshold在全部有效真实Calibration数据上的overall正确率，`gaussian_fidelity`是Gaussian threshold按其拟合population weights积分的理论正确率。
- Calibration SiteMap detector使用真实相邻frame difference＋完整average两条证据并集。difference逐transition按自身背景噪声标准化；重复中等变化使用run-measured binomial bar，单次明显变化使用pixel×transition family-wise bar；steady/high-loading由average保留。两条路径都不能低于authored `detection_sigma`，candidate identity只来自average local maxima。旧even/odd half state、split veto、absolute per-frame sighting与global saddle owner全部删除。
- Calibration新增默认关闭的`Review detected sites`。开启时capture与detect各执行一次，候选SiteMap通过`calibration/review` companion signal同时进入Monitor和modal point review；operator可单点、列表或框选排除零到多个ghost sites，确认后最终SiteMap重新连续编号并只运行一次全部下游分析。review使用`FluentDialogWindow`和完整Fluent control family；`zlc_ui`拥有view，`zlc_plot`只拥有Image point gesture/overlay，Workbench组合。candidate/excluded/final映射进入Calibration report与summary，`site_review.npz/png`保存候选和排除结果；取消等同Stop。Runtime的唯一operator request/response lifecycle为未来人工Scan axis保留复用边界，但当前没有Scan consumer或UI。
- Temperature：final JSON、summary和生存率Figure NPZ/PNG。
- SLM Feedback：保存输入摘要、stable site table和逐candidate精选BOX samples、fit、weights、
  actions、metrics、phase-change fact与command receipt；不保存raw camera frames。每个
  `candidates/candidate-XXXX.npz`都是可加载/续跑的Science Context，final仍只有唯一selected Context。
- Feedback报告固定包含uniformity history、site signal evolution、weight evolution、selected
  site histograms、initial/selected camera mean和initial/selected phase；每个完整candidate另存
  `candidate_site_fits/candidate-XXXX` Figure NPZ与PNG，使用真正的Histogram cell和Figure API
  的per-site bimodal Gaussian fit，不加入Monitor preview。normal或Stop只产生一个final Science Context。
- Feedback Monitor固定自动打开四张图：canonical Camera Measurement逐帧publication经mean reduction得到的带编号site map实时图、observable
  uniformity、site signal evolution和Target share evolution；phase保留为信号和最终Figure但不自动开panel。
- Feedback每site只在完整shot batch上做受约束双高斯与full-data ΔBIC>10判定。dark site按bracket（最新观测优先）向loaded share二分或沿方向逐分辨率爬行，由loaded sites以公共因子出资、每site每candidate至多一个分辨率；bright fraction低于全阵中位数一半的loaded site视为在loading ramp上：hold、不出资、不被识别excitation扰动；probe episode每site一次，方向只由verdict改变。formal-double使用loop gain除以实测plant slope（前6个ordinary update携带±2%零和excitation识别），无adaptive scalar。`probe_combined`计入`maximum feedback updates`，diagnostic candidates不计。
- Grouped Curve与Grouped Rolling共用hover/lock/wheel contract：hover仅轻微加粗，lock才压暗其它lines；无框标签固定axes右上角，lock加`* `并接管滚轮逐series移动。standalone/Facet Curve的孤立valid点使用同一Line2D短横线glyph；invalid仍断线。
- ImagePlot与FacetGrid image cell不再暴露interpolation参数；schema/style/Panel Setting/Edit均删除该字段，renderer唯一固定值为`nearest`。
- Feedback failure与normal/Stop封存同一选择（最佳已完整测量candidate到SLM与`final/`），summary
  记录错误；只有封存写不出时才restore起始phase。未测phase不得成为final。

### 2.5 Device Control与settings provenance

- Generic Device Control只消费adapter的`TunableField` contract，显示Current、Desired、Live apply、Apply、Status、Refresh及active owners；已删除旧的edit-immediate `field_committed/read_values/set_form`路径和demo残余。
- RF frequency/power四个policy edge已进入Rigol、Vaunix及Virtual RF的optional Init schema并复用同一Control tunable；空值表示无该侧policy、可随时清回空值。只有完整有限的low/high pair才形成Scan port范围，单侧edge只约束直接tune；全空Init不归一化或改写硬件当前值。
- Pylon公开`gain_db`的SDK bounds/current与grabbing-safe write；Virtual camera公开`exposure_seconds`。成功且effective实际改变才推进session-local epoch；Stepped Scan严格使用effective return，不能把hardware未接受的值写成Dataset coordinate。
- Logic静态requirements与Stepped Scan运行时选择的device ports都形成field claim。DeviceUse按device-specific owner revision原子核风险授权、dependency closure与pending write；字段命令期间不能进入新Logic，owner变化取消尚未执行的write。
- Device I/O只在现有串行worker/adapter command lane执行。Refresh去重合并且属于close guard；75 ms live input在相同policy projection及in-flight write期间保留每字段latest-only值，Qt owner只处理plain projection和已完成readback。
- CameraFrameRecord在adapter边界冻结settings session/epoch；Pylon无法证明live tune前后buffer边界时首个read batch明确携带old+new，Virtual在trigger时冻结。Runtime使用event-varying record并保持generation-stable run record，finite/scan/indexed保留范围合并为压缩epoch ranges。
- Figure lineage当前grammar包含每个event record及只解析实际引用epoch的device settings；FigureViewer Device tab读取同一事实。无active Logic的调整不写历史，完整参数状态不复制到每frame。

## 3. 当前验证状态

### 3.1 Fresh wheel与installed lanes

- Wheel：`zou_lab_control-2.0.0-py3-none-any.whl`，1,397,303 bytes，295 entries，
  SHA256 `5F4C8360F40A5068B2EB4F006FAAF5441D7DE246CFB157707289D684F871E6D2`。
- 全新venv按`constraints.txt`安装`[dev]`；`pip check`零问题；`zlc check`确认八层全部
  来自该venv中同一个`zou-lab-control` RECORD，且不存在第二bootstrap包。
- Installed `software: PASS`：1,614 passed、4 skipped；4个skip仅因本机无Icarus，
  Pulse其余139项全部通过，已有Vivado/xsim证据继续单独覆盖RTL。
- Installed `gui_offscreen: PASS`，包含UI、Plot Qt、SLM Editor、FigureViewer、Workbench
  presenter/device/Pulse Editor以及每个TaskConsole case的独立进程生命周期。
- Installed `virtual_vertical: PASS (9 passed)`；`notebook_offline: PASS`。
- Checkout bootstrap由共享Python resolver统一拥有；Experiment/Server/Viewer、FPGA
  build/program与resource estimate从任意工作目录都加载当前checkout的bootstrap和八层。
  `install_requirements.bat`显式清除source injection并验证installed distribution；同一新wheel
  的isolated install仍从site-packages加载bootstrap与Pulse server。该边界不改变science/runtime。
- Calibration七张report Figure都经FigureViewer current reader重开；SLM Feedback六张Figure
  均经formal `zlc figure_viewer --check`读取。上述证据属于此前冻结tree；本次Feedback/Curve
  最近一次pre-adaptive controller实测为6个probe candidate、22个总candidate、最佳34/35与ratio 1.1337。
  当前bracket/loading-edge controller在同一virtual lattice（25/35起始可见、4%loading余量）实测：2个probe candidate、20次formal update共23个candidate，第11次formal update起35/35并保持到结束，observable ratio 1.384→1.12；此前版本在32/35停滞，三个site在两个share间乒乓。
- Histogram新增`histogram_poisson_gaussian`/`bimodal_poisson_gaussian`（泊松光子计数⊛高斯读出噪声的精确卷积，格点和只有编译核一份实现，SciPy路径与overlay调用同一核；种子取加权中位数与四分位距）。`run_fit_models`同一直方图feed（两态8±2/30±4，读出宽度2–4格点，对该模型偏宽）：bimodal 3.91 ms对Gaussian bimodal 3.50 ms，single 5.45 ms（单态盖双态的misfit，宽度跑到12格点）对3.04 ms；光子计数直方图（0.7/6光子、读噪0.45、64 bins）编译求解0.7/1.0 ms对Gaussian 0.6/0.7 ms，两路径最优一致到3e-15。`run_mot_roi_chain --panel3 histogram`（新增：40个源索引Histogram cell，逐cell live bimodal fit，像素值故读出宽度数十格点=最差区）：fit_total 22.3对18.6 ms，numeric_fit_batch 13.0对13.7 ms，四面板临界路径97.2对94.4 ms、7.6对7.7 fps；首版259 ms的根因是NumPy孪生evaluator给40个overlay逐项求和，删孪生改调编译核后消失。engine测试的Poisson行用0.3光子读噪、四分之一光子bin（comb可见，宽度可辨识）与十倍amplitude（峰值约70 counts），二十光子的宽度是方差的3%、两求解器会停在不同点。
- Calibration weighted Gaussian/Empirical threshold当前worktree聚焦证据为`11 passed`：已知真值、label-invariance、窄bright＋尾部噪声、Empirical/fallback、Figure archive同模型重放、population-weighted Plot classifier、coordinate roundtrip与warm refresh。另对500组随机Gaussian参数核对解析交点，383组存在两均值间相关根，最大加权log-curve交点误差`3.37e-13`、相对数值最优误差`8.95e-17`。本cut未重跑正式Runtime/Workbench vertical。
- Calibration site review当前聚焦证据：Runtime精确operator request/response与Stop、saved-frame完整review链及全descriptor virtual Calibration保持通过。正式`zlc task_console --template virtual`science路径以8 samples检测24 sites，排除`site_0000`后最终Calibration为23 sites、terminal移除全部自动preview；`site_review.npz`42,208 bytes、PNG 145,791 bytes，FigureViewer current reader成功重开。此前`parent=None`测试遗漏了真实parent会把frameless QWidget降为child的错误；当前`FluentDialogWindow`以`Qt.Window + WindowModal`保留Fluent顶层身份，真实parent测试核对active modal、确认响应及title-bar关闭，QTimer/TaskConsole同类回调中的nested loop正常退出。`zlc_ui.capture_window`取得精确1152×653 shared-screen capture，Fluent title/body边界为32/32、无native dialog chrome，35-site状态与controls完整显示。
- SiteMap detector真实red为8帧50%-loading site在0/2/4/6出现：旧实现full-average `702.04σ`仍被split veto漏掉；新实现记录7个相邻变化、最大change `355.67σ`，定位误差0.030 pixel且同一35-site阵列全部找回。20个随机seed覆盖698个至少加载一次的sites，漏检0、spurious 0；完整site-detection `7 passed`，saved-frame review＋Workbench vertical chain `2 passed`。
- 后续adaptive gain/formal-update accounting与Curve hover/lock切分运行6个直接聚焦用例，结果`6 passed`；未重新运行100-shot验收。
- 紧凑Science Context当前证据：SLM Editor完整文件`22 passed`；strict Context与Feedback candidate/Stop/failure边界`10 passed`；最终三条直接边界`3 passed`。X15213全尺寸体积、Pattern/composite逐元素roundtrip和8-bit phase-code roundtrip均来自当前worktree；未运行100-shot。
- 固定nearest清理运行standalone/facet artist、Workbench parameter surface及Fluent Setting/Edit四个聚焦用例，结果`4 passed`。
- Device Control当前回归：Workbench完整`425 passed`；Runtime完整加Figure grammar `112 passed`；adapter/camera/scan受影响组`53 passed`；Device Control Qt、风险revision、refresh close guard、in-flight latest-only和demo直接证据均通过。Atom完整回归同时暴露并修复Temperature sibling event record、Feedback输出声明和三条terminal/Stop残余；100-shot virtual Feedback仍为既有`34/35`上限，未用放宽断言冒充通过。
- FigureViewer此前以formal launcher和`zlc_ui.capture_window`在真实Windows屏幕完成四条1152×653验收：current archive默认Image Monitor、点击Add panel新增Curve、从Setting点击Edit进入共享Fluent `PanelEditorView`、以及多层Flow展开树；四次均保持shared 90% window尺寸和固定左栏。右侧复用TaskConsole `ConsoleBoardView + PanelCardView`并置于白色work surface，支持每panel切saved dataset、alternate plot kind、Setting/remove/order与closable Edit；static Edit只保留Panel/Semantic/Display/Fit。用户当前重新裁决Info readout必须统一multiline并按实际visual layout紧包；旧的无换行单行分支会cutoff长内容且不能作为phantom inner-scroll的替代修复。固定Plot kind从Setting删除，动态Signal keyed-choice在reconcile写值前更新choice domain。
- FigureViewer Info readout当前根修：旧实现以单行控件掩盖multiline少算Qt block高度的问题，长无换行值因此cutoff，而multiline内部仍保留`maximum=1`的phantom scroll。InfoPane现统一复用`FluentReadoutMultiline`，唯一高度owner逐block读取`QTextLayout`实际像素并加document/widget margins；未设cap时horizontal/vertical range均为0，短值、显式两行和长无换行值实测为26/45/349 px且原文逐字符不变。现有UI用例扩展后相关文件`12 passed`；formal `capture_window`以1152×653真实FigureViewer重开Camera Figure，左栏一行readout保持紧凑。
- FigureViewer Logic/Devices/Flow当前根修：archive内部`event-N`只作parent引用，Logic页以真实Logic identity显示递归去除device字段后的run参数；Devices页用run record的stable role→instance映射解释run/event snapshots，按实际device聚合并给每项保留Logic、sequence与scope，缺映射/identity/device key一律拒绝而不猜。Flow原位删除QTree owner，Workbench只投影唯一Logic/Device nodes和causal/device edges；Qt以layered+barycentric布局、独立edge ports与long-edge lane绘制，典型100 nodes同步构建约6.5 ms，3-device、diamond、真实DFS汇合及10-node长链均无edge穿node，长链horizontal range为0。Calibration/Temperature normal与partial report、SLM candidate/report、Seamless/Stepped/Temperature live均保存实际用到的device facts；Feedback pre-shot只记录SLM，post-shot冻结同candidate三设备，failure rollback不改变已存candidate provenance；Stepped tunable以完整scan values及逐点readback等值contract记录，不复制event history。聚焦回归`67 passed`，另Console Logic`34 passed`；formal Windows real-screen capture为1152×653、DPR 3、3-device Flow无横向scroll且节点/箭头无重叠。
- 公共Panel Setting现复用master的page-local `FluentOverlayFrame` owner，并以固定identity（`Setting · panel-N`）作为可拖header，不读取可编辑title/signal/structure；右上角紧凑`×`只隐藏Setting。TaskConsole与FigureViewer因复用PanelCard同时获得该行为，Panel删除仍是card header的受保护命令。
- Exact Scan Panel恢复当前证据：真实event chunk为`1×1×3×5`、canonical为`2×(65×2×2)×3×5`的Signal经实际SignalPlane与Plot host由真实`field.x=65`触发>64拒绝；拒绝前后Setting均保留`field.x/y/z` fate且不再出现phantom `point`，独立Curve Panel title保持canonical axes，Fluent form在`fit_unavailable`同时仍含三个Semantic controls。精确目标`20×(10×10×10)×3×35`的title authority输出`(20)×(10×10×10)×(3×35)`。多维FacetGrid默认最外层真实scan axis，不再以flattened point rows制造1000 cells或phantom point-row restriction。相同live projection与仅title metadata变化均不reconcile Setting form；固定Plot kind不再进入Setting，FacetGrid只保留可编辑Cell kind；Facet默认、feasibility、真实拒绝与Fluent Setting聚焦证据`22 passed`。
- Exact Scan terminal/Frozen根修当前证据：真实`20×(10×10×10)×(3×35)`canonical Dataset从partial Live publication开始，原子提交`field.x→Facet, field.y→Y, field.z→X, pair/site→Reduced`后，Live、运行中Frozen及terminal seal后重新创建的Frozen host均保持同一schema fingerprint、物理shape `(20,1000,3,35)`、resolved roles和`[-0.5,9.5]×[-0.5,9.5]` limits。根因三处均删除：multi-fate逐行修复导致回退默认35×3、host accept后以1×1×3×35 event schema覆盖canonical surface、以及histogram threshold/shape-only viewport无条件重放到image。当前实现使用atomic fate assignment、canonical accept metadata、resolved capability interaction和schema/spec view identity；Plot semantic/feasibility/facet/threshold聚焦`52 passed`，Workbench canonical/Frozen/retarget/save交叉聚焦`10 passed`。
- Plot/Runtime/Workbench当前candidate直接回归：Plot `534 passed`、Runtime `107 passed`、Workbench `435 passed`；Atom对Figure/hosted-node新contract的direct用例`1 passed`。这些结果来自当前tree，不复用旧Exact Scan cut的计数。
- Seamless duration scan根修当前证据：绝对period保留32-bit nominal base，25-bit signed slot只承载delta；整张table自动选择最小整数tick scale，最大127且DAC恒为1，不改RTL/bitstream。实际量化rows统一进入compiler、wire、readback、Pulse Editor Run/Sync/Hold/Step、Seamless Dataset coordinates/run record与Temperature companion/artifact；distinct authored points若量化坍缩会在device前拒绝。Pulse全包`140 passed, 4 skipped`，Pulse Editor`98 passed`，Seamless＋Temperature直接纵向`10 passed`；三路冻结审查无第二量化owner、compat fallback或未闭合consumer。
- 通用Fit表达式减量候选：Panel Setting/Edit只提供单行`name=value`精确fixed与`name=guess(value)`初始猜测；fixed复用既有bounds请求通道的相等端点作为内部exact marker，但普通及regular-image solver都会把该维度真正移出optimizer，free-only计算DOF/Jacobian/covariance，all-fixed不启动optimizer。表达式按painted单位输入，PanelState/Figure只保存canonical fixed/initial；语法、unknown或domain错误只在DisplayDescription保留transient draft/warning并继续同model自动fit。Curve/Histogram/Image/Facet/Rolling共用FitSession请求，Console与Viewer共用同一Panel投影；fixed参数的误差publication为invalid。相对`17629d1`无新增production文件或类、production净增376行、test净增163行；此前临时`fit_target.py`与第二套canonical validator已删除。减量后直接聚焦`38 passed`，Plot全包`534 passed`、Console View全文件`31 passed`，另直接验证普通/regular all-fixed均返回`all parameters fixed`。Workbench全包在先运行36项后仍稳定暴露既有camera-restart selector顺序失败，目标test单独运行`1 passed`；该问题属于下一独立cut，不混入Fit提交。
- Camera restart selector顺序根修：`_refresh_signal_choices`原来把“首个surface尚未accept、因此`binding.host is None`”误当成“panel尚未mount”，在已有initial `PlotPanelPort`忙于首帧时又启动第二个retarget port；后完成的候选会关闭已接受crosshair的port。恢复路径现在只在唯一生命周期真相`binding.port is None`时创建port，Board继续独占已有port的首帧accept；没有新增状态、helper、selector/restart特判或测试函数。原必现的auto-inference→camera-restart顺序`2 passed`；Workbench全包该缺陷已消失，结果`436 passed`，唯一剩余失败是当前master新增`warm_numba_cache.bat`直接运行layer module的launcher约束，与本cut无关。
- 长Task partial artifacts：Runtime在worker failure/Stop边界调用domain writer；Feedback普通异常从最后完成candidate生成6组Figure后rollback，Temperature从已提交survival保存partial curve/Figure，Calibration从最新完整三帧cycle保存partial capture（分析完成则保存完整报告）。`run.json`只索引这些已完成文件，不再是失败run唯一内容。
- Feedback的`candidates/candidate-XXXX.npz`现为标准Science Context；operator可在既有Science Context输入中手动选择它作为新run起点。过程数组移至`data/measurements/measurement-XXXX.npz`。新run从candidate 1开始并使用本次authored update预算；没有resume输入、自动旧run查找、续编号或旧run预算继承。
- Pulse STATUS ABI当前为LOADED/RUNNING/DONE/ENGINE_ERROR/UNDERFLOW/LINK_ERROR；UART fault不再置engine ERROR，observer failure不再伪装成board error，Remote日志使用ERROR/DONE真实事件名并写status/cursor双读、observer exception与FIRE总elapsed。该ABI令layout fingerprint更新为`0x5A55DF95`，实验板必须重build/program。
- 第一次installed software尝试曾在重负载下出现一次本地SLM测试TCP connect timeout；
  同一wheel的精确case随后连续5/5通过，第二次完整installed software lane通过，因此没有
  用该不可复现事件改动产品remote timeout或server逻辑。

## 4. 仍有效的FPGA build/timing证据

以下证据描述已完成的硬件构建结果，不证明当前Python tree，也不代替实验板验收：

- Vivado 2019.1 fresh project完成全部IP、top synth、place/route、reports和bitstream。
- Routed setup WNS `+0.726 ns`、TNS `0`；hold WHS `+0.036 ns`、THS `0`。
- 12条bus-skew全部MET，0 violated，最差`+18.988 ns`。
- 资源：20075/20800 LUT、14053/41600 FF、76/90 DSP、40 RAMB36 + 2 RAMB18。
- Bitstream SHA256：`82A8E04DC3BD4F21E3ACED22D16E4544C7CBC3E9C7A642337F4D689702994A6C`。
- Engine/UART oracle、full-top FIRE和SAFE pin gate的已有结果仍是build/simulation evidence。

## 5. 明确未执行的实验机验收

以下均保持`UNEXECUTED`，不得由software/offscreen/virtual evidence冒充：

- real-screen：真实monitor、DPR、window interaction和capture receipt；
- camera：official DCAM/Pylon SDK/runtime、accepted-edge timestamp、首次auto Panel latency、
  exposure/busy/drop/cancel及raw/electron provenance；
- SLM：official SDK/header、serial/profile/correction、DVI/USB orientation/readback和optical settle；
- optical Feedback：per-site BOX samples、simultaneous CI、common-site total brightness、
  selected final Context和rollback；
- FPGA board：最终bitstream program/flash及外部DAC/TTL电气时序/波形。

这些步骤只按对应runbook由实验机operator显式执行。

## 6. 明确延后的GridPlot扩展

- 当前FacetGrid单surface仍以最大`8×8=64`个真实Matplotlib Axes为上限，不直接提高。
- 数百cells的后续方案是把Dataset全部`total_cell_count`与单页最多64个
  `visible_cell_count`分开；renderer只创建/复用当前页Axes，不先创建全部Axes再隐藏。
- 分页使用global cell/site identity，hover、selector、fit overlay、focus及跨页滚轮导航都不得
  把page-local index冒充global index；TaskConsole与FigureViewer复用同一机制。
- typed Figure仍保存全部cells；页面只是显示状态。导出提供当前页、全部分页PNG或多页PDF，
  不生成一个包含数百微小Axes的单张巨图。

## 7. 当前Goal进度（2026-08-27）

> 本节只记录**当前状态、裁决、和仍然开着的事**。
> 每一次尝试的经过、被实测否决的方案、以及我自己的测量失误，写在**做那件事的
> 那个 commit 的 message 里**——那才是不会过期的记录。计划文档不做 changelog；
> 曾经把它当 changelog 用，导致 586 行里 317 行是一次 Goal 的流水账，已清除。

### 七项清单：全部完成，不得重做

| 项 | 结论 | 定型提交 |
|---|---|---|
| 1 | `display__title` 切回 Auto 时 `ValueError` 穿出 Qt slot、进程 abort。根因＝**一个字段两件事**：`field.default`（当前值）被当成"允不允许空"的规则。改读 `field.required`，与标量的 `blank_allowed` 同一个所有者 | `90ee153` |
| 2 | 3D 拖动延迟。用户报的 93 ms/move 是夹具伪影（GUI 线程上跑阻塞生产者）；真实数字见下表 | `c166f42` 等 |
| 3 | selector 不响应/卡顿/跳变：image/curve/rolling **36/36 有画面、0 次静默、6 次拖动提交同一区域**。histogram 的区域会变，是因为它的 x 就是测量值、活数据值域在动 | — |
| 4 | 3D 柱数＝数据格数，LOD 全删；ROI 缩小时柱数跟着变（1150×1150 → 1150×790） | `14f140b` |
| 5 | home 相机四角 `far=a near=c left=d right=b`；墙＝ab/ad、轴＝cd/bc、z 轴在 d；轴与场景一起遮挡。`origin=lower` 时画面本身翻转，`far=d` 是同一关系 | `14f140b` `a15dd23` |
| 6 | rolling 的 x 是**相对最新一发**（区域实测为负 shot 号）；四种 kind 都能从区域派生：image/curve **切片**、histogram **值带置无效**、rolling **shot 窗口** | `f809e4d` `a359bd8` |
| 7 | 全 kind 矩阵、组合场景、逐环节 profiling，见下 | — |

### 当前实测（master，4x4 单面板，free-running 生产者，`chain/matrix.py`）

三列是三件不同的事：**live 帧**＝数据来了整幅重画；**手势帧**＝手势期间的帧；
**hand→picture**＝按下到画面出现。

| kind | live 帧 中位/p90 | 手势帧 中位/p90 | hand→picture 中位/p90 |
|---|---|---|---|
| image heatmap | 41.5 / 47.4 | 11.5 / 43.0 | **8.2 / 16.8** |
| image 3D bars | 85.8 / 89.9 | 35.8 / 86.5 | **34.7 / 135.3** |
| curve | 12.3 / 15.1 | 10.7 / 21.4 | **9.3 / 13.3** |
| histogram | 33.9 / 40.4 | 12.4 / 47.7 | **6.4 / 15.2** |
| rolling | 14.1 / 16.3 | 10.1 / 17.7 | **8.1 / 13.6** |
| facet grid | 20.7 / 23.7 | 11.1 / 45.3 | **8.6 / 14.4** |

组合场景（`chain/combos.py`）：四面板并存 2x2 手势 image 7.2 / 3D 30.9 / curve 6.3 /
rolling 6.3 ms；ROI 链 相机 9.0、ROI 3D 面板 38.5 ms；带 fit 的 rolling 6.3 ms。

**同一杆秤的对照**：heatmap 的选框手势走 overlay（只重画矩形）4.4–7.3 ms；
heatmap 的**中键 pan**（同样整幅重画）**30.3 ms**；静源 3D orbit **29.7 / 30.7 / 31.2 ms**。
3D 与 heatmap 的整幅帧持平——那是这套系统重画一幅画的地板。

### CPU 与内存（`chain/cpusplit2.py`，free-running）

| 阶段 | 修 `OMP_WAIT_POLICY` 前 | 后 |
|---|---|---|
| 只有生产者、零面板 | 1016% of one core | **82%** |
| 四个 2x2 面板 | 874% | **138%** |
| 单个 image 面板 4x4 | 846% | **115%** |

内存：四面板约 **360 MB**，稳定不涨（面板删除后回落）。

### 已做出的裁决（不再重开）

- **不改抗锯齿**。解析覆盖正是 3D 边缘能和 heatmap 在同一 DPR 下一样干净的原因。
- **`_stroke_rims` 的缝按设备像素收，不改成逻辑像素**。改了能省 3.7 ms，但稠密场景
  会整体变亮变平——那是改画面。
- **不给 `_reduce_blocks` 配浮点内核**。它是浮点输入下正确的退路；新内核＋新逐位契约
  测试换平均每帧约 1 ms，不成比例。
- **3D 场景的帧缓冲与 face-id 平面两块轮换**。多占一份缓冲，换掉每帧向 OS 买 21 MB
  新页面的 4.6 ms。安全前提是两块每帧都被完整写满。

### 仍然开着的

- **3D 的 live 帧 86 ms**，是它 p90 高的原因：一帧新数据要重算派生面（59 万格）再光栅化，
  一次 live 帧插进手势中间就把那一拍拉到 130 ms。手势本身（静源 30 ms）已与 heatmap 持平。
- **rolling 的区域切不到过去的 shot**：窗口是"距最新多少发"，而派生只看得到当下这一发，
  所以只有触到最新一发的窗口才有数据。现在会明说，不再发布一整帧无效数据（`ea462c6`）。
  要真的切到那些 shot，需要用面板已经租下的 indexed history 重新派生——那是能力，不是修补。
- **`_reduce_blocks` 4.2 ms 出现在半数 image 帧上**（裁决见上，记录在此备查）。
- **z 刻度标签被切**（"0.8" 印成 ".8"）：场景 fit 只留 4% 几何 margin。先于本轮存在。
- **`test_guard_c_save_semantics` 红**：保存面板图时 matplotlib mathtext
  `ParseException`。**在 master 上同样红**，与本轮无关。
- **`Github\zlc_*` 是拆包残留的旧副本**（`zlc_runtime/selection_bridge.py` 56KB vs 树内 96KB，
  8 月 3 日），pip editable 全部指向它们。走 `zou_lab_control` bootstrap 时不受影响
  （它把 checkout 置顶），但**裸 `import zlc_runtime` 会拿到旧副本**。删不删是用户的事。

### 已经查清、不是缺陷的

- **facet grid 先前那个 78 ms 是探针假象**：facet 的 overview 是"选择器"，
  按设计只认左双击进入单元格，别的手势一律忽略——探针在它上面拖，量到的是下一帧 live 到达。
  探针改成先进入单元格后，facet 是 **8.6 ms**，与其它 kind 同级。
- **live rolling 刚挂 fit 时的 `fit requires more finite observations than free parameters`**
  只出现一次：窗口里的点还少于自由参数的那一刻。随后正常求解。是正确反馈。

### 探针（`C:\Users\eadri\AppData\Local\Temp\claude\chain\`）

`matrix.py` 全 kind × 手势｜`combos.py` 多面板/ROI 链/fit｜`verify3.py` selector 三症状｜
`roikinds.py` 每种 kind 的区域派生｜`imgprof.py` live image 帧自时间｜
`orbitqt.py` 3D orbit 自时间｜`kernelcheck.py` 场景光栅与 numpy 规范的逐位对照。
`tap.py` 是它们共用的嵌套自时间探针（cProfile 会骗人）。
