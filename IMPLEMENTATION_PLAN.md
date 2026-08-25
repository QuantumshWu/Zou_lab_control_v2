# ZLC — Current Implementation Checkpoint

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
- Plot axis/semantic identity已收口为`AxisRef(domain, axis_id)`稳定key；scope只接受tagged
  latest或tagged typed coordinate value，不再让display label或裸文本控制字进入truth。
- Runtime是唯一跨publication history owner；active leases的signal-level event/indexed表示变化
  通过presentation epoch使同signal全部Panel重新投影，不增加scientific publication/revision。
- PanelState只保存authored target；Live、Frozen和FigureViewer都以Plot成功返回的完整accepted
  `DisplayDescription`判断当前pixels、能力与交互。Selector/viewport observation携exact Dataset
  generation+revision，TaskConsole Console核对后才持久化、镜像或发布derivation。

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
- FigureViewer必须从archive exact recipe恢复typed input，不按shape推断plot kind。
- FigureViewer与TaskConsole Live/Frozen使用同一个accepted PlotSpec、parameter、selector/
  viewport capability contract；Viewer semantic edit同样只在host accept后更新surface。
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
- Feedback每site只在完整shot batch上做受约束双高斯与full-data ΔBIC判定。winning probe effective share由`probe_combined`直接采用，普通gain/max-change不再缩小它；其余double共同补偿总功率并保留内部相对修正。direct-adopted site仍single或补偿后原double变single时，才从新baseline re-probe这些矛盾site；两侧无double且baseline未变不重复。formal-double gain从authored `feedback_gain`起步，连续两次显著改善乘1.25、显著变差乘0.5、不确定性内保持；diagnostic probe/single不参与adaptive。`probe_combined`计入`maximum feedback updates`，diagnostic candidates不计。
- Grouped Curve hover与lock已分离：hover仅轻微加粗，lock才压暗其它lines；无框标签固定axes右上角，lock加`* `并接管滚轮逐series移动。standalone/Facet Curve的孤立valid点使用同一Line2D的短横线glyph并共享hover/lock；invalid仍断线，Rolling不改变。
- ImagePlot与FacetGrid image cell不再暴露interpolation参数；schema/style/Panel Setting/Edit均删除该字段，renderer唯一固定值为`nearest`。
- Feedback failure记录last complete candidate与rollback receipt；Stop只从完整测量candidate
  选择结果，未测phase不得成为final。

### 2.5 Device Control与settings provenance

- Generic Device Control只消费adapter的`TunableField` contract，显示Current、Desired、Live apply、Apply、Status、Refresh及active owners；已删除旧的edit-immediate `field_committed/read_values/set_form`路径和demo残余。
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
- Calibration weighted Gaussian/Empirical threshold当前worktree聚焦证据为`11 passed`：已知真值、label-invariance、窄bright＋尾部噪声、Empirical/fallback、Figure archive同模型重放、population-weighted Plot classifier、coordinate roundtrip与warm refresh。另对500组随机Gaussian参数核对解析交点，383组存在两均值间相关根，最大加权log-curve交点误差`3.37e-13`、相对数值最优误差`8.95e-17`。本cut未重跑正式Runtime/Workbench vertical。
- Calibration weighted-threshold residual sweep：相对`68e2c4b`无新增文件、production class、兼容alias或第二fit owner；8个production文件净增334行，其中145行是Plot现有classifier owner对authoritative components的构造/重放，69行是既有bimodal owner的对称宽度先验与候选选择，其余120行是Calibration计算/report字段、target grammar和唯一stored tuple。新增production definitions只有selector validator与Plot现有mixin内两个直接方法；`_classifier_gaussian_components`只由同一PlotSession配置、rollback、refresh和recipe projection消费。三个既有test文件净增112行且没有新增test case：一个原threshold case改为label-invariance/weighted crossing/narrow-bright，一个既有artifact case增加exact component persistence，一个既有classifier case增加authoritative render；全部KEEP。active docs已同步，旧label-fit/equal-prior/balanced-fidelity语义搜索为零。唯一deferred是正式Runtime/Workbench vertical复跑。
- Calibration site review当前聚焦证据：Runtime精确operator request/response与Stop、saved-frame完整review链及全descriptor virtual Calibration保持通过。正式`zlc task_console --template virtual`science路径以8 samples检测24 sites，排除`site_0000`后最终Calibration为23 sites、terminal移除全部自动preview；`site_review.npz`42,208 bytes、PNG 145,791 bytes，FigureViewer current reader成功重开。此前`parent=None`测试遗漏了真实parent会把frameless QWidget降为child的错误；当前`FluentDialogWindow`以`Qt.Window + WindowModal`保留Fluent顶层身份，真实parent测试核对active modal、确认响应及title-bar关闭，QTimer/TaskConsole同类回调中的nested loop正常退出。`zlc_ui.capture_window`取得精确1152×653 shared-screen capture，Fluent title/body边界为32/32、无native dialog chrome，35-site状态与controls完整显示。
- SiteMap detector真实red为8帧50%-loading site在0/2/4/6出现：旧实现full-average `702.04σ`仍被split veto漏掉；新实现记录7个相邻变化、最大change `355.67σ`，定位误差0.030 pixel且同一35-site阵列全部找回。20个随机seed覆盖698个至少加载一次的sites，漏检0、spurious 0；完整site-detection `7 passed`，saved-frame review＋Workbench vertical chain `2 passed`。
- 后续adaptive gain/formal-update accounting与Curve hover/lock切分运行6个直接聚焦用例，结果`6 passed`；未重新运行100-shot验收。
- 紧凑Science Context当前证据：SLM Editor完整文件`22 passed`；strict Context与Feedback candidate/Stop/failure边界`10 passed`；最终三条直接边界`3 passed`。X15213全尺寸体积、Pattern/composite逐元素roundtrip和8-bit phase-code roundtrip均来自当前worktree；未运行100-shot。
- 固定nearest清理运行standalone/facet artist、Workbench parameter surface及Fluent Setting/Edit四个聚焦用例，结果`4 passed`。
- Device Control当前回归：Workbench完整`425 passed`；Runtime完整加Figure grammar `112 passed`；adapter/camera/scan受影响组`53 passed`；Device Control Qt、风险revision、refresh close guard、in-flight latest-only和demo直接证据均通过。Atom完整回归同时暴露并修复Temperature sibling event record、Feedback输出声明和三条terminal/Stop残余；100-shot virtual Feedback仍为既有`34/35`上限，未用放宽断言冒充通过。
- FigureViewer此前以formal launcher和`zlc_ui.capture_window`在真实Windows屏幕完成四条1152×653验收：current archive默认Image Monitor、点击Add panel新增Curve、从Setting点击Edit进入共享Fluent `PanelEditorView`、以及多层Flow展开树；四次均保持shared 90% window尺寸和固定左栏。右侧复用TaskConsole `ConsoleBoardView + PanelCardView`并置于白色work surface，支持每panel切saved dataset、alternate plot kind、Setting/remove/order与closable Edit；static Edit只保留Panel/Semantic/Display/Fit。用户当前重新裁决Info readout必须统一multiline并按实际visual layout紧包；旧的无换行单行分支会cutoff长内容且不能作为phantom inner-scroll的替代修复。固定Plot kind从Setting删除，动态Signal keyed-choice在reconcile写值前更新choice domain。
- FigureViewer冻结树残余审计：相对`80116b5`无新增production/test文件或production class；production净增562行，其中Viewer presenter +327为multi-panel prepare/configure/atomic swap/retirement，FigureViewer view +97、InfoPane +61、Handle +28为复用board/card/Edit/tree的直接UI owner，shared panel description projection +131由TaskConsole与Viewer两个consumer共同使用，PanelEditor/Form合计+16为static模式与dynamic keyed-choice正确reconcile，demo另净增18行；同时旧Console/card/modal path净删98行。Tests净增122行，只有一个新test function（用户明确报告的loaded-left-pane变宽），其余均扩展既有Viewer/Panel/Edit/atomic cases，全部KEEP且无重复fixture。`dataset_picked/set_datasets/set_figure_*/show_figure/run_host_dialog/kind_read_only`等旧API与ASCII Flow owner搜索为零；旧单panel `_host/dataset/recipe`镜像和unused candidate viewport已删除，新增状态只存在于唯一presenter `panels`、window cards/Edit tabs、固定pane width和tree widgets，不进入archive grammar或第二份plot lifecycle；无deferred残余。
- FigureViewer Info readout当前根修：旧实现以单行控件掩盖multiline少算Qt block高度的问题，长无换行值因此cutoff，而multiline内部仍保留`maximum=1`的phantom scroll。InfoPane现统一复用`FluentReadoutMultiline`，唯一高度owner逐block读取`QTextLayout`实际像素并加document/widget margins；未设cap时horizontal/vertical range均为0，短值、显式两行和长无换行值实测为26/45/349 px且原文逐字符不变。现有UI用例扩展后相关文件`12 passed`；formal `capture_window`以1152×653真实FigureViewer重开Camera Figure，左栏一行readout保持紧凑。
- FigureViewer Device/Flow当前根修：正常Camera working point早已保存在lineage node的`named_devices/device_snapshots`，旧Device页却只读active override epoch并把普通run显示为空；现同时投影baseline snapshot与实际引用的epoch delta。Seamless/Stepped Scan原先在`PublishedSignalSource`取到publication后只保留value，Runtime worker commit因此写`parents=()`；两个Scan现把仅被保留shot的exact publication随commit交给Runtime，discarded shot不进入chain。Viewer Flow用同一archive node展开parameters、pulse、plan/device call、device details与exact Upstream，不复制run record或查latest。Runtime全包`109 passed`、两个Scan文件`20 passed`、Viewer/Guard C直接`24 passed`、Console Logic`34 passed`、UI全包`87 passed`；Workbench全包本修改路径`436 passed`，唯一失败仍为当前master的`warm_numba_cache.bat`直接运行layer module。真实Camera→Seamless Scan Figure已验证parent为Camera Measurement、Device baseline非空，并以formal 1152×653截图显示完整Upstream树。
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

### 3.2 Frozen-tree residual sweep

- 本cut相对`1043aa7`的冻结统计为52 files：production 27 files
  `+4,046/-3,320`（净+726）、tests 23 files `+1,356/-800`（净+556）、
  active docs 2 files `+38/-24`（净+14）。无新增或删除文件。
- Plot production净删48行；增量集中在Runtime净+298的exact axis/generation、
  lease history与restart cleanup，以及Workbench净+476的staged/accepted surface、
  Live/Edit/Viewer原子替换和overlay lineage。最大正增文件为Plot `data_view.py`
  +174（`SelectionSubject`搬迁与统一axis projection）、Workbench `presentation.py` +171、
  Workbench `console.py` +166、Runtime `selection_bridge.py` +155、Runtime `plane.py`
  +140、Plot `figure_artifact.py` +106和Workbench `viewer.py` +88；全部是上述现有
  owner的直接实现，全部KEEP。
- Production class文本新增3、删除5，净删2。`ResolvedAxis`替换
  `AxisDescriptor`，`SelectionSubject`从`session.py`移到`data_view.py`，`_ProjectedAxis`
  替换`_ResolvedAxis`；同时删除`RollingHistoryPoint`和`PlotViewportObservation`。
  Runtime六个`indexed_*` dict合并为唯一`indexed_history`；PanelBinding的accepted/host/focus
  镜像与PlotPanelPort的presented/shown镜像已删除，当前pixels只由既有`_Prepared`
  记录的`_surface`表达。新增definition无0-consumer；Figure selector encode/decode对与
  Runtime锁内快照/锁外materialize分界是仅有的single-consumer helpers，均KEEP。
- Test function从369减到366（净-3）；无新test文件或fixture，fixture数保持10。
  7个新的直接用例全部KEEP，分别覆盖未上屏front手势拒绝、presentation debt、
  fit跨restart identity、lease window、atomic surface accept、generation+revision publication identity
  与Rolling viewport。旧label alias、`source_revisions`、revision-only和implementation-specific
  materializer用例已删除或合并，无剩余文件级MERGE/DELETE项。
- `AxisDescriptor`、`source_revisions`、`HistoryPool`、`AcceptedSurface`、
  `editor_accepted_display`、`PlotViewportObservation`、revision-only publication lookup、
  平行history owner、冲突标记与`git diff --check`残余均为0；workspace实验数据
  未转换、未删除、未修改。
- 下一未完成项是restart期间pending data surface不得覆盖较新的accepted selector/configuration；该顺序依赖将在独立worktree修复并重新跑Workbench全包。

### 3.3 Calibration detected-site review增量残余审计

- 相对父commit `4c0071b`冻结为32 files：production 17 files `+1373/-294`（净+1079），
  tests 7 files `+277/-47`（净+230），active docs 8 files `+89/-3`（净+86）。新增production集中在Runtime operator-input lifecycle、
  Calibration science/report、Plot point surface、Fluent point-review view/dialog和Workbench组合，
  来自用户明确批准的通用Task人工输入边界与完整Fluent交互，不是compatibility或第二套Task framework。
- Consumer graph只有五个owner且无重复：NodeHost唯一保存request/response/Stop；Calibration唯一
  决定candidate→excluded→final SiteMap和报告；`zlc_plot`只拥有Image point/rectangle gesture与
  overlay；`zlc_ui`只拥有Fluent modal/window/controls/local view state；Workbench只连接plain
  identities与两个完整surface。未来manual Scan axis没有consumer、业务字段或UI，仅明确复用
  NodeHost与FluentDialogWindow lifecycle。
- Material positive production全部KEEP：Calibration `task.py/outputs.py`拥有science flow与typed
  preview/report；Runtime `host.py`拥有精确阻塞/Stop；Plot `point_review.py`是唯一Image gesture
  surface；UI `point_review_view.py`与`FluentDialogWindow`是唯一Fluent view/modal；Workbench
  `console.py`与composition只路由request/publication/view。Calibration detector在
  `calibration.py`净删115行，只保留相邻difference＋average两条证据。`_SiteReviewPublisher`只是
  SignalDataPlane要求的短期companion identity，不保存第二份science或history。
- 新增四个直接测试均KEEP且不重复：Runtime同一Host覆盖精确response、stale拒绝、跨run identity
  与Stop；UI真实FluentDialogWindow覆盖全部Fluent control family和excluded result；Calibration
  saved-frame真实NodeHost/SignalPlane覆盖剔除、最终SiteMap、summary和Figure；Site detection覆盖
  single-frame、50%-loading、steady、dim、dense、border与pure-background。Plot层不保留平行dialog
  test；Architecture、Guard A与Workbench view double只合并现有contract断言。
- `dataset_output.py`、`plane.py`和`test_signal_plane.py`最终零diff；production无raw Qt review
  controls或Plot→UI反向依赖；detector无偶奇/half state、absolute-frame sighting或全局saddle owner。
  无旧API/alias、无第二个review owner、无Scan
  compatibility、无conflict marker、`git diff --check`为0。唯一明确延后项是未来Scan
  manual-value request的domain字段和业务UI，当前Calibration不包含它们。

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
