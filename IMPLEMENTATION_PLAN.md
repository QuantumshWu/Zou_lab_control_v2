# ZLC — Current Implementation Checkpoint

更新时间：2026-08-23

状态：`CURRENT SOFTWARE COMPLETE / RESIDUAL SWEEP COMPLETE`

本文只记录当前tree已经完成的产品切面、最新验证和有效的实验机边界。
最终产品不变量见`ARCHITECTURE_DESIGN.md`。只有从当前tree重新取得的证据可以作为
完成证明。

## 1. 当前实施范围

- 所有项目内部持久化格式、Dataset contract和artifact contract改为稳定语义名；
  Figure、Calibration、Pulse、Target和Science Context不带数字版本。
- Reader只接受当前完整grammar；现有workspace不转换，文件可直接不受当前reader支持。
- 产品bootstrap为`zou_lab_control`；根`pyproject.toml`仍是唯一distribution manifest，
  八个`zlc_*`目录只是同一distribution的依赖边界。
- Hosted Task只在NodeHost worker真正Start时分配run directory，并在任何不可逆工作前
  原子建立`run.json`。进度、artifact registration、Stop、failure和terminal result都写回
  同一个lifecycle truth。
- Runtime不自动dump live/intermediate Dataset。Calibration、Temperature和SLM Feedback
  由各自domain owner保存精选artifact，并通过ExecutionContext注册已完成文件。
- Figure NPZ是primary typed artifact，PNG只是preview。Figure保存exact Plot recipe、overlay、
  viewport和causal lineage graph；FigureViewer与TaskConsole使用同一个Plot host路径。

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

- 公共Figure API严格编码/解码PlotSpec、parameters、size、viewport、classifier、fit与
  typed image overlay；archive先发布，preview后渲染。
- Panel Save只是公共Figure API的adapter，不再维护第二套writer或restore grammar。
- FigureViewer必须从archive exact recipe恢复typed input，不按shape推断plot kind。
- Lineage保存root、event nodes和direct parent IDs；Viewer验证引用、reachability和cycle后
  投影为tree。

### 2.4 Domain Task artifacts

- Calibration：每run一个folder，final Calibration JSON、summary JSON/text、精选报告Figure
  NPZ与PNG。默认不保存全部raw frames。
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
- Feedback每site只在完整shot batch上做受约束双高斯与full-data ΔBIC判定。formal-double gain从authored `feedback_gain`起步，连续两次显著改善乘1.25、显著变差乘0.5、不确定性内保持；diagnostic probe/single不参与adaptive。`probe_combined`计入`maximum feedback updates`，diagnostic candidates不计。
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
- Calibration六张report Figure都经FigureViewer current reader重开；SLM Feedback六张Figure
  均经formal `zlc figure_viewer --check`读取。上述证据属于此前冻结tree；本次Feedback/Curve
  最近一次pre-adaptive controller实测为6个probe candidate、22个总candidate、最佳34/35与ratio 1.1337。
- 后续adaptive gain/formal-update accounting与Curve hover/lock切分运行6个直接聚焦用例，结果`6 passed`；未重新运行100-shot验收。
- 固定nearest清理运行standalone/facet artist、Workbench parameter surface及Fluent Setting/Edit四个聚焦用例，结果`4 passed`。
- Device Control当前回归：Workbench完整`425 passed`；Runtime完整加Figure grammar `112 passed`；adapter/camera/scan受影响组`53 passed`；Device Control Qt、风险revision、refresh close guard、in-flight latest-only和demo直接证据均通过。Atom完整回归同时暴露并修复Temperature sibling event record、Feedback输出声明和三条terminal/Stop残余；100-shot virtual Feedback仍为既有`34/35`上限，未用放宽断言冒充通过。
- FigureViewer/TaskConsole Panel parity已用同一current 2x2 Curve在两个真实窗口经Windows capture逐轮对照：两者card均为612×494逻辑像素、同一约50px title band与同一body frame；Viewer的dataset action bar改为固定高度，card从中部漂移改为紧随bar top-align，title从占位`figure`改为archive dataset label。zlc_ui完整回归`86 passed`，Viewer/Panel/Workbench contract聚焦`62 passed`。
- 长Task partial artifacts：Runtime在worker failure/Stop边界调用domain writer；Feedback普通异常从最后完成candidate生成6组Figure后rollback，Temperature从已提交survival保存partial curve/Figure，Calibration从最新完整三帧cycle保存partial capture（分析完成则保存完整报告）。`run.json`只索引这些已完成文件，不再是失败run唯一内容。
- Feedback的`candidates/candidate-XXXX.npz`现为标准Science Context；operator可在既有Science Context输入中手动选择它作为新run起点。过程数组移至`data/measurements/measurement-XXXX.npz`。新run从candidate 1开始并使用本次authored update预算；没有resume输入、自动旧run查找、续编号或旧run预算继承。
- Pulse STATUS ABI当前为LOADED/RUNNING/DONE/ENGINE_ERROR/UNDERFLOW/LINK_ERROR；UART fault不再置engine ERROR，observer failure不再伪装成board error，Remote日志使用ERROR/DONE真实事件名并写status/cursor双读、observer exception与FIRE总elapsed。该ABI令layout fingerprint更新为`0x5A55DF95`，实验板必须重build/program。
- 第一次installed software尝试曾在重负载下出现一次本地SLM测试TCP connect timeout；
  同一wheel的精确case随后连续5/5通过，第二次完整installed software lane通过，因此没有
  用该不可复现事件改动产品remote timeout或server逻辑。

### 3.2 Frozen-tree residual sweep

- 最终consumer graph只有三个新增共享owner：Runtime `TaskRun`被NodeHost和Calibration direct
  run使用；Plot Figure API被Panel Save、Calibration、Temperature、SLM Feedback与FigureViewer
  使用；Workbench lineage capture只连接Runtime publication与Viewer tree，不保存第二份science。
- Material positive production files全部KEEP：SLM Feedback task（精选candidate data、六张图、
  summary/rollback）、Plot Figure core（strict recipe/overlay/archive-first/shared host）、Runtime
  TaskRun与Host接入、Calibration/Temperature产物接入、Workbench Task input projection和UI lineage tree。
- Workbench重复archive wrapper、被忽略的Panel Save render callbacks、Figure/PanelState alternate readers、
  Calibration alternate readers、旧bootstrap和内部带版本contract全部删除；无compatibility alias。
- Tests净增13行。删除的`test_archive.py`能力已合并到Data Figure codec、Panel Save与Viewer tests；
  其它删除只针对不再存在的格式/兼容行为，没有通过删科学验收掩盖失败。
- 冻结统计为143 files、production `+2,599/-1,478`（净+1,121）、tests
  `+958/-945`（净+13）、docs `+351/-255`（净+96）。新增生产量集中在上述三个真实owner
  和三个domain Task；没有新增plugin-specific manager/registry或平行lifecycle。
- Unsupported format/alias、旧bootstrap/contract/API、重复owner、Markdown本地链接、JSON、
  conflict marker与`git diff --check`残余均为0；workspace实验数据未转换、未删除、未修改。

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
