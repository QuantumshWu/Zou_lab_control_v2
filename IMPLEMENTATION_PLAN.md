# ZLC — Current Implementation Status

更新时间：2026-08-25

状态：`PLOT/RUNTIME/WORKBENCH CUT COMPLETE / OVERALL GOAL IN PROGRESS`

本文只记录当前tree已经完成的产品切面、最新验证和有效的实验机边界。
最终产品不变量见`ARCHITECTURE_DESIGN.md`。只有从当前tree重新取得的证据可以作为
完成证明。

## 1. 当前实施范围

- 新世代自派生front根修：Manual新data+overlay共同parent旧同名bundle时，旧front算法直接永久pending。现只在leaf内撞名时从本次已读DAG选唯一因果后代，独立分支和跨leaf同shot判据不放松。原source-sibling案例扩展红绿、三个front直接case通过，不删除旧parent、不用latest冒充。

- 第二轮独立交叉检查修复U1/U2遗漏：隐藏Setting丢force/父Tab Show不刷新会留下旧fates和Signal目录，现延期事实在原Card由Show或open消费一次；Pulse容器已正确缩小，不恢复adjustSize，只修gap indicator 228/286高度不同为228/228。4个既有直接case在Windows Qt正文通过，隐藏0 reconcile、未变Show0重建，无长期GUI残留。

- Pylon复用未变成功请求/读回，mode与gain变更仍真实失效；gain.previous不多读一次。二轮删除SDK→mutable整图→immutable的中间副本，在Release前完成唯一ownership copy；CameraFrameRecord直接打包非连续数据。10个既有direct case通过，包含SDK Release后立刻覆写仍保留原图、负stride/大端、失败回滚、mode与mixed epoch。没有缩小finite ring。

- Fit数学owner收敛：13内置模型删NumPy/bulk重复公式，value-only不求Jacobian（2M点/4参数少生成64MB导数输出），一维坐标借只读view；已有finite筛选不重复执行。4个production文件净减127行；14项相关既有用例覆盖独立数学锚/差分、single/batch、Poisson、RegularImage、fixed/NaN、custom及cold/warm竞争，均通过。未把测试含编译耗时当fit性能，未做全量预热，实际缓存重复signature检查为空。

- Evidence去掉TaskConsole逐item第二层进程包装：当前完整test_task_console_app在同一pytest进程顺序28 passed/55.07s。发现的两处旧测试问题分别是未选择必填Pulse、关闭后访问已清owner；改为真实选择及检查原窗口handle，没有修改产品默认值或生命周期来迁就测试。测试内部真正需要的独立app进程仍保留，运行后无本worktree Python残留。

- UI复核发现Manual与Panel的公共标题函数会隐藏无具名axis的域，与固定三域contract冲突；现保留该组，显示(1)/(—)，不创建新axis。3个既有title直接用例通过。

- Simulation同site geometry不再重采样全部camera PSF；原35个plane重建变为复用同一不可变对象。原add/remove/move下一帧物理图像用例红绿通过，FFT/像差/PSF尾部/随机序列未改；未量化毫秒收益。

- DCAM整改：同ROI setter由22次SDK属性调用降为0；相同exposure请求不再因硬件量化重写，arm保留一次真实工作点读回，Monitor复用该结果。失败后只读可恢复actual但不伪造请求成功，下一setter重试；原量化、读回失败及arm变更拒绝用例通过。源码满幅/裁剪arm链约20/24次SDK属性调用，不是通信往返或实测时延；未操作实验机。

- Stepped重复操作已清：device-only只编译/LOAD一次，API两点只编译两次（首个实际点直接用于run记录）；settle改到写设备之后。原case从等待时看到旧值[0.25,1]变为[1,2]，Stop、恢复及第二点拒绝路径通过；未删Stepped authored settle或Temperature等待。

- History删除64MiB/100000的隐藏截短、错误nbytes预算和重复capacity状态，唯一保留量为max(active window)。3个原window/多lease/gap用例通过；两次带gap的真实publication A/B证明请求100001时旧路径只给100000并丢首valid，新路径完整100001（两端valid、中间gap invalid）。未执行100000 shots；大窗口内存成本由实际数据决定。

- Plot按需计算：8×400 V→mV归约绘图只转换400个输出，不再额外转换8×400原数据；raw selector按需转换仍正确。完整初值自动initializer为0次、partial仍1次；停止warmer不生成后续样例，初始size/parameters不再多轮绘制。3个既有单位/预热直接用例通过，无全量warm/GUI运行。

- Feedback报告已删除binned Histogram二次fit，candidate与selected都复用本次科学fit的分量/threshold；invalid显式无模型/阈值，公共classifier target与Figure/远程roundtrip不再把null变成自动fit。既有classifier case和失败后partial Figure/Context case通过，报告参数按site坐标逐项一致；首次测试误按target列表顺序配site，已改为按真实coordinate核对，未改生产数值掩盖测试。

- 信号目录必要性整改：Plane缓存未变目录、删除无消费者的description revision；Console一次生成rows/overlay offers且直接交View。真实Plane＋Console纯metadata探针中首次4panel/2signals只生成2 descriptors及1次rows，20次idle与普通数值更新后均0重建/0菜单push；Stop→Start的overlay候选失效与恢复正确，4个原目录/拓扑用例通过。此证据不宣称像素或GUI验收。

- Figure单次读取与初态切面：read_archive返回(info, arrays, datasets)，删除read_dataset重复解码，所有生产消费者统一复用已验证Dataset；raw typed成员复用其不可变bytes。36个原格式/命名空间用例和3个Viewer/Calibration入口通过。Host共用initial_configuration，首次呈现已有最终fit，临时export不恢复废弃display；4个原Plot直接case覆盖固定limits、初始fit、零额外present及真实远程初态。未改磁盘格式。

- Atom必要性整改：FrameSurvival的8cycle映射从16次规划变为event/canonical各1次；SLM command规范化从3次变1次，未变phase的idle状态不重扫像素。Feedback每run一次LOAD、每candidate一次Fire，删除同phase整批重拍与接受observer故障的旧策略；Pulse fault不再以observer_error遮蔽engine error。直接有限capture/fault矩阵、Task失败后Figure/Context保存、Survival 2/3/4frame及SLM stale/unknown用例通过。没有实验板/build或长100-shot验收。

- 必要性整改的目录切面：已有目录0次重复flush，缺失层逐层创建并flush新child/parent；创建flush失败携published路径，不自动修复旧失败。5个既有直接case通过，文件atomic write顺序未改；没有新增marker、回滚删除或重试机制。

- Device Control未变owner的idle beat为0次policy/form投影，本地编辑及命令完成继续事件更新；所有写入仍在DeviceUse锁内核最新权限。三个既有直接用例通过，含风险失效和pending写取消；没有额外硬件读取或权限缓存。

- 必要性整改的display clock：HarmonicClock改按monotonic elapsed跨deadline，保留harmonic周期、owed/debt及Pause；Qt采用不会提早唤醒的PreciseTimer。6个既有直接case通过，延迟跨过多个800ms周期只产生一次due，不补画漏帧。该问题只解释慢周期延迟，不能归因默认100ms四图的全部耗时。

- Pulse装载不再为未请求的1×1执行计算delay FIFO占用；Fire使用实际repeat数的原有检查和驻留验证复用。既有TTL/DAC溢出及repeat seam/terminal SAFE直接用例通过，溢出在FIRE命令前拒绝；没有RTL修改或FPGA build。

- 必要性整改的Atom数值切面：删除BOX整帧float64转换、Camera二层stack、numeric count弃置数值归约、成功Gaussian threshold的弃置Empirical计算。5个既有直接用例通过；Derive与233基线在105个dtype/归约/轴组合上schema及validity一致、数值等价，std明确在float64做subtract避免float32中间运算。无科学阈值/shot/模型政策变化。

- 必要性整改的UI切面：隐藏Setting按需prepare、Manual table只更新受影响cells、Pulse Scan与timeline保留未变控件、InfoPane复用未变度量、Form同schema adopt不重建依赖，删除无消费者choice序列化API。8个既有定向用例通过；实屏FigureViewer输入7.25、Preview、Save及NPZ读回确认完成，截图和过程证据不入Git，已关闭所开GUI与render子进程。

- 必要性整改的记录切面：generation run record与atomic event record各自只冻结一次，内部构造复用owned记录；finite物化按现有有序chunks取增量，indexed删除重复raw record；DataBlock身份重包保留数值验证事实。10个既有定向case通过；三siblings计数由每shot 10次deep-freeze变为首shot 2次、后续1次，四次finite物化的metadata输入由1/2/3/4变为1/2/2/2。未把计数换算为耗时承诺。

- 必要性整改的终态/输入切面：删除Seal的全量物化和sealing中间态，保留coverage校验、终态数据与EOS；删除Processor终态暂存与latest Start的弃置预取。4个既有测试入口（13个参数化case）通过，验证无人读取时Seal零物化、真实terminal输入在worker中取得且siblings/窗口语义保留。

- 必要性整改的classifier/交互切面：authored Gaussian/fallback初态贯通Figure、本地/远程Host，0自动fit；移除单cell组件只求解该cell。配置只产生一次最终description，拒绝overview/单series hover不物化native。4个既有定向case通过；新worktree首次缓存触发过超时，未修改timeout、只在缓存就绪后重跑受影响case。无GUI/硬件/build。

- 必要性整改的数据存储切面：finite extend/indexed hole/roll直接组装compact validity，VALUE和各component组合与233基线结果一致；公共restriction复用未变Axis/Domain。4个既有直接案例通过；独立四种validity的A/B证明输出相同且不再分配VALUE逐像素mask，未运行GUI或硬件。

- 2026-09-10本次验证：真实UART串行帧证明LOAD完成回复及SAFE抢占；真实top＋既有Xilinx BRAM行为模型证明首次装载4shots与SAFE后驻留重放4shots的18 TTL/40 DAC data逐tick一致、4 DAC clock工作、同ID不重复Fire。没有运行FPGA build/synthesis/program，旧时序报告不代表新ABI已通过。相关软件定向验证覆盖驻留重用、丢ACK、pending LOAD取消、新server握手、device/manual扫描及错误恢复。
- RF正常设频率/幅度为1 write＋1 query（两次发送、一个响应）；Control Apply与单位投影不额外读设备。没有未经厂商证实的复合SCPI；真实native UNIT切换另发一次必要写入。错误后的current/unit/range保持unknown直到必要操作或显式Refresh确认。Fabric与SLM remote各在原session内复用连接，断线不自动重放写入，关闭释放idle连接。
- 窗口首个Close保留关闭意图；device read/tune/init/discovery完成后由原Qt owner继续完整关闭，不要求再次点击、不加timer或平行生命周期。直接Qt验证覆盖有/无TaskConsole的pending关闭，测试窗口均已关闭。原始探针/日志只在ignored目录，不进入git。

- 2026-09-10通信收口：Seamless仅Start准备一次Acquisition；按最新裁决删除全部settle参数/UI/等待/记录。正常DONE后不追加SAFE。Pulse使用带command ID的完成握手，驻留Fire不重load/不清clock，软件与RTL同一新ABI；实验机需重启server并自行build/program。本次不执行FPGA build/program，软件与RTL仿真证据单独列出。
- Repeat标题已撤掉擅自添加的min/max统计及区间格式：其余所有轴固定于同一数据当前坐标，Cell-data也先选定坐标，再沿目标Repeat数valid；每轴只返回整数，Point/Cell尺寸不变，不受Plot Scope影响，也不改为采集次数。旧多context汇总分支删除，GUI探针同步同一标量契约。
- Pulse Bracket编辑已统一Period/post的光标、chrome命中、拖动、gap指示和Add目标；旧分立MIME/端点drag/只数Period的gap路径删除。结构编辑一次提交period与Bracket，移动原边界不再留旧锚造成逆序。空Bracket在首/中/尾均保留，可改count及重新插入修复；On Pulse、Save Pulse/Preview与compile/codec共用同一错误提示，未改RTL或有效文件格式。直接边界/空编辑用例和可见Qt事件链及截图验证，证据在ignored research，不入Git。
- Repeat标题统计已从全局any改为本次publication当前坐标的条件有效数，不受呈现fate/Scope/Focus影响。正式Qt Scan第一power50次、下一power7次显示50→7；同publication切Scope135/200均为7，Point/Cell维度不改。完整panel_data_shape每case300次：Site35 P50/P95/max=0.094/0.253/0.575ms，207万像素但compact有效性=0.086/0.111/0.349ms；不包含Qt paint，未展开像素mask，首次统计也<1ms。原始计时与截图只在ignored research。
- Restart后的Frozen Edit旧代提示已补齐，纯数据/年龄变化仅更新状态文字，不reconcile表单；FigureViewer同一路径。真实Runtime/Host/Save直接用例及正式Qt确认revision同为1而generation不同仍正确提示，Refresh前保存ref/values精确为旧快照，Refresh后才更新；所有测试GUI/渲染子进程已关闭。

- Scan Plan统一手动/device/pulse行的列结构与预算，单位slot固定、起终点等宽，保留原扫描/单位/草稿逻辑。正式Console真实Qt在正常和70%屏幕宽度下截图，原手动行起点错位50 logical px降为0，各行start/stop/points/remove位置与宽度一致；resize不改plan，单位切换通过，2项既有直接编辑/单位用例通过，测试窗口关闭。截图和探针留ignored research，不入Git。
- Scan Range/Values复用原轴行与Plan owner：Range数列和初始为空的Values文本独立保留，Values不显示Points；切换只隐藏原控件，selector始终只改Range（包括隐藏时），共享unit分别换算两套输入、不互相回填。`plan_input_rows`读取raw banks供编辑/selector使用，Layout原样保存raw plan；Start才按mode解析当前数列，执行`ScanAxis/ScanPlan`及plain port/values/unit输出不变。直接用例覆盖顺序、重复值、空/坏Values拒绝、隐藏Range更新、单位独立换算与执行快照不被后续草稿改动；正式TaskConsole实屏通过三类轴混排、真实按钮切换/输入/Save→Load，空Values和未显示Range均保留，输入/单位/mode/删除列位置宽度一致，Qt异常为0。窗口已关闭，截图与探针仅在ignored research。

- sealed Scan→Derive Run→Image改表达式再Start的`signal generation owner is already active`已复现并根修：Processor复用原Producer的终态世代替换，不改Shutdown保留数据政策和Input range。正式Qt二次Start为Done，新generation、同一exact Scan parent，下游Image逐像素为原值×2；11项直接生命周期验证含连续三次frozen运行、并发Start、active保护、未发布幂等通过。Derive结构说明复用三域颜色，以logical shape为主、Storage shape为辅。

- Scan→Occupancy已删除旧“Point必须只有frame一个轴”的否决，保留真实图像/校准检查并去掉重复验证。正式ScanDatasetWriter/Plane的50×3×72×92案例复现terminal两Point轴报错，修后逐event与完整terminal结果通过且保留frame3、power1/mVpp；FrameSurvival同步按READOUT_EVENT分组配对并保留其它Point轴及finite placement，2/3/4 frames与2×2 scan定向验证通过。信号列表改读canonical schema并复用公共逻辑轴摘要，正式Qt截图确认(1×50)×(3×1)×(72×92)及power轴，测试窗口全部关闭，未操作硬件。

- Derive界面统一英文，只读信息完整展开，短帮助不设内部scroll。空Code等未提交草稿显示中性`Draft: Signals row N: Enter Python code`，不标节点error；Start仍禁用，填完整后经原finalization解除。实际Qt输入→计算→Plot→Save Fig→FigureViewer流程通过，1×1×35结果与独立NumPy相同，所有窗口已关闭。
- Numba预热复用原样本，按模块变更/缓存缺失选择render（raster＋3D）或fit（compiled＋radial）组；匹配marker仍检查缺失机器码，日志区分production signatures新编译/磁盘加载。11项无编译定向验证通过，未清缓存或全量预热。同一kernel源文件内的Numba整文件失效仍存在，不声称本改动消除了该重编译。

- Seamless支持无scan slot的manual/device-only计划：无板内轴时加载普通Pulse和空wire，Run repeats采shots_per_point，完整repeats由已有Host循环推进；不虚构slot或Dataset轴，沿用采集准备/Stop/恢复。删除Node/Editor旧否决及空子ScanPlan，真实slot路径保持不变；硬件snapshot记录每次Fire实际scan_repeats。7项定向验证通过（无slot manual/device、原混合路线、排序和Editor）。2026-09-09实屏又定位到资源加载层残留的“必须有slot”门槛，现删除该wrapper，直接复用严格Pulse reader；既有文件加载用例红绿确认普通Pulse可选、真实slot及API保留、坏slot引用仍拒绝。未操作真实硬件。

- Device Control对齐在原Fluent form内收口：统一表头/全部行/单位选择器列宽，仅Desired伸缩，保留bool和无Live行的正确占位；使用正式Control opener和zlc_ui截图API完成默认、混合单位、拉宽窗口的Windows DPR3可见截图检查。截图API支持内容尺寸窗口而不强改窗口尺寸，原固定屏幕比例验证保留；证据留ignored research，测试窗口均已关闭。

- 2026-09-08 运行时TunableField名称去单位后缀：RF frequency/power及四个policy边界、Pylon gain、Virtual Camera exposure从设备定义统一到Control/claims/Remote/Scan与新保存引用；metadata补齐dB/s，固定单位Config/SDK接口不改，不加旧runtime别名。RF/Scan保存与单位写入/Control的10项及非RF metadata/claims的4项定向验证通过；历史实验记录不重写。

- 2026-09-08 设备单位写入收口：Scan和Control共用设备层的只读单位投影/转换与Apply，Rigol原生Vpp/Vrms/dBm仅必要时切UNIT，原始数值/单位对在退出时恢复。端口范围和单位转换离开Qt；pending单位请求不能误Apply旧单位，晚结果不覆盖新draft或碰已删除控件。Remote沿同一接口透传，canonical provenance与requested_unit分开。最终15个定向实例通过，包含非50Ω、原生写/失败恢复、整数前缀、Control只读换单位、占用权限、异步Scan Editor及Fit负B/C单位转换；仅模拟SCPI与Qt控件验证，未做真机/实屏全流程验收。

- 2026-09-09 Scan Editor单位异步回调修复两处：换port可撤掉unit picker，pending期间改为禁用/恢复同一行稳定的unit host；用户已改草稿而丢弃晚结果时，按当前mode刷新状态，清除过期的Converting提示。只在原owner改3行，保留新草稿与原单位转换流程。

- 2026-09-08 按最终用户裁决，扫描彻底采用author unit：Plan直接存135…247与mVpp，Seamless/Stepped输出同一单位，仅设备/编译边界换算；display_unit旧路径及8ULP/相等检查均删除。5个单位/Plan直接实例通过；真实Runtime的Seamless十点例在设备回读偏离设定时完成，Dataset coordinates逐位等于135→247的十点且unit为mVpp，run record一致；设备异常与restore传播仍保留。曾添加的独立readback event字段不符合现有merge grammar，已撤掉，不扩格式，设备原有tune回读路径保留。未做真实硬件验收。

- Seamless的Acquisition logic沿原Start/Restart与ready入口，后续所有points/repeats复用同一generation。之前包含10ms settle的GUI结果不作为当前无settle通信流程验收；当前测试删除被取消的等待断言，保留shot placement/Stop/恢复与一次准备。Temperature原有50ms等待留在自身Task，不借Seamless参数实现。

- 2026-09-08 当前worktree完成六项：Pulse DAC保留disabled全开按钮并实屏确认四列对齐；Layout递归编码authoring rows并完成真实Save/Load；Derive以普通Fluent下拉选择atomic producer bundle、不新增Runtime数据；声明过的Panel fit在无首帧/无reserved generation时允许Scan Start，首次arrival复用有序tap。真实Qt验证单点单shot，Scan输入与新一代首个Fit publication数值/validity相同，唯一根为重启后的Camera首event，未自动启动Camera。GUI及渲染children全部关闭；退出曾有Device Manager等待sequencer control关闭的短暂拒绝日志，最终正常退出，未冒称无日志。
- `Saturation`按用户新裁决改为`f(x)=(A*x+B)/(x+C)`，使用`asymptote/numerator/shift`（A/B/C），headline为asymptote；参数单位依次为y、y*x、x。C可负，仅限制拟合域`x+C>0`，不把增长条件`A*C>B`设成不可配置门槛；固定B/C、绝对坐标裁剪、single/batch与uncertainty继续共用现有fit机制。公式/Jacobian/负C/下降数据/固定参数、normal/hard single与B1/B8/B64、headline及复合单位的显示/表达式/派生Dataset共8项目标验证通过。B的单位复用通用乘积解析，同单位原样通过，前缀转换保留符号，非线性换算不得冒充乘积倍率。
- Reduction `Last`复用Scope restriction与既有统计核，包含sigma/SEM、sparse空交集、Rolling shot carrier、Histogram pooling和Figure往返的50项聚焦验证通过；另10项小dtype检查通过，不新增Last kernel。Histogram Figure encoder原来漏存reduced/reduction，现已补齐并严格读取；缺两字段的旧Histogram/FacetHistogram Figure需重新保存，不自动兼容。原始日志、截图、性能报告都在ignored目录，不进入Git。

- 2026-09-07 已完成拟合请求公共准备与逐帧临时闭包减量：Single/Facet共享同一选区、单位与输入准备语义，每个请求仅解析一次公共信息，保留初值竞争、最终精度、逐cell有效性及same-shot呈现；绘图遍历以同序迭代替代自引用递归闭包。不禁用垃圾回收、不调大阈值、不以手动回收移出计时窗口冒充改善；初始化/重布局图对象与稳态临时对象分别取证，不宣称所有长尾已消除。性能结果与报告只留本地ignored research，不纳入Git。
- 2026-09-07 当前性能worktree实施要求：RegularImage single/Bn收敛为同一数值流程，内部规则轴不再展开成逐像素坐标；现有context贯通prepare/objective，以稳定中心化统计量减少重复整图扫描，并直接核对最终残差。前景文字/边框仍由Matplotlib语义生成，在同一有序compose owner内批量重放。最终是否保留必须由同输入正确性、single/Bn代价和真实四Panel结果裁决，不能将尚未通过的候选记为性能完成。
- 该cut实现已提交为`971765ba`，并按用户裁决与master `d1b1d1e0`整合；保留master的候选裁决、有效性、交付顺序与公共bench机制。整合前四Panel Curve critical `89.85→84.69ms`、Image `98.08→83.27ms`；isolated B40 image fit `27.52→24.70ms`，single `19.32→26.70ms`是明确代价，不将这些时间冒充合并后的复测。无新production类/文件，性能cut本身净+71行（包括最后补齐的f32/f64预热样本）。带探针Curve的约205ms长尾在补测中定位到117.334ms gen2 GC，不归入renderer/solver正常耗时；无探针的发生频率尚未确定，不宣称物理极限或永不掉帧。细节及剩余大头以`research/FINAL_MAJOR_OPTIMIZATION_REPORT.md`为准。

- 所有项目内部持久化格式、Dataset contract和artifact contract改为稳定语义名；
  Figure、Calibration、Pulse、Target和Science Context不带数字版本。
- Reader只接受当前完整grammar；现有workspace不转换，文件可直接不受当前reader支持。
- 产品bootstrap为`zou_lab_control`；根`pyproject.toml`仍是唯一distribution manifest，
  八个`zlc_*`目录只是同一distribution的依赖边界。
- Science Context只保存16-bit circular Pattern模差分、Target和语义metadata；pupil/operator/composite均由SLM核心公式重建，Editor/Feedback在Send或shot前先固化Pattern。X15213 1024×1272 5×7同数据实测旧布局12.524→0.227 MiB（-98.19%），保存0.051 s、完整读取0.179 s；固化最大误差4.82e-5 rad，保存前后Pattern、composite phase和8-bit phase code逐元素一致，固化中位15.57 ms。
- Hosted Task只在NodeHost worker真正Start时分配run directory，并在任何不可逆工作前
  原子建立一次不可变的`start.json`；进度、artifact registration与Stop只在进程内，不写盘；
  terminal result与failure在结束时一次性建立`run.json`。两份记录都只创建、从不替换。
- Runtime不自动dump live/intermediate Dataset。Calibration、Temperature和SLM Feedback
  由各自domain owner保存精选artifact，并通过ExecutionContext注册已完成文件。
- Figure NPZ是primary typed artifact，PNG只是preview。Figure保存exact Plot recipe、overlay、
  viewport、selectors、facet focus和causal lineage graph；FigureViewer与TaskConsole使用同一个Plot host路径。
- Figure NPZ唯一writer现按member做有界1MiB可压缩性probe：结构化/平滑数据继续Deflate，低收益大camera数组使用标准ZIP Stored。20×1200×1920 uint16 noise的Panel Save实测`4.23s→0.73s`，archive阶段`3.63s→0.18s`；92.16MB原始数据原压至79.12MB，现为92.16MB，明确以13MB换约3.45s。平滑1200×1920 float32仍Deflate为2.98MB、总时约0.33s。
- Plot axis/semantic identity已收口为`AxisRef(domain, axis_id)`稳定key；scope只接受tagged
  latest或tagged typed coordinate value，不再让display label或裸文本控制字进入truth。
- Dataset三组结构现统一为同级`DomainSpec(shape, axes, axis_codes)`：Repeat/Point使用
  explicit row codes，Cell-data使用不物化pixel codes的dense implicit stride；`ValueSchema`
  只保留dtype/unit/validity。旧的平行row-coordinate/topology与Plot双身份路径整体删除；
  scan、history、selection、fit与Figure只读取同一axis domain/code truth。
- Fate Setting不再预跑candidate render/layout feasibility：所有axis始终列出plot kind声明的全部roles；
  64-cell等容量限制只在真实replace/layout transaction执行。旧semantic probe、cache和kind validate
  wrapper已删除，schema vocabulary不再随size、DPR或renderer可用性改变。
- Runtime是唯一跨publication history owner；active leases的signal-level event/indexed表示变化
  通过presentation epoch使同signal全部Panel重新投影，不增加scientific publication/revision。
  Runtime内部绝对ordinal在materialize时统一转换为以最新为0的相对primary-index；Plot与
  Workbench只把它当普通AxisRef，不自动scope或建立history专用interaction路径。
- 关联显示按完整same-shot group就绪；已选成员pending（含首发与重启）时整组保留上次完整画面，只有同shot明确invalid结果可呈现且不画无效标记。完整新图像输入不继承旧overlay；删除缺companion提前放行与同generation沿用旧层的旧行为。
- Derive现使用普通Python/NumPy：每行Name与多行Code，单表达式`eval`或多语句`exec`后取`result`；上一行输出可按名引用，中间变量不发布。草稿只验证名字与Python语法，不保留`.frame/site` DSL或预先schema typing。具体Logic Editor复用Fluent控件，显示Input range、window数量、输入/输出三domain摘要和同一份代码帮助；Qt不求值也不物化数据。代码不是安全沙箱，执行异常按输出名报告，无限循环或危险native调用不承诺可取消。
- `Operand`薄封装完整Repeat × Point × Cell-data schema、values和validity；任意命名axis可`isel/sel`、`where`与`mean/sum/count/any/all/min/max/std`归约，不自动squeeze。标量选择只移除指定axis，无轴domain保留大小1；std为总体标准差、空组invalid，bool count数有效True。NumPy同shape修改通过`with_values`，高级变形显式构造完整schema的Operand；不猜裸ndarray的轴或隐式对齐不同几何。原判决一致性只是`a.occupied.isel(frame=0) == a.occupied.isel(frame=2)`及`where`的普通代码，可接任意命名轴归约，不重读camera/calibration或重新分类。
- Input range明确为`event`（默认，当前原子事件）、`run`（同publication的本次canonical Dataset，finite未采位置invalid且跳过其它Panel history）或`window`（默认50个位置，Runtime已有index_by_source bounded history）。没有finite累计范围的Monitor在run模式只提供当前完整结果，不无限积累；不支持history的源明确拒绝window并提示run。NodeHost对订阅锚点及所需atomic siblings使用同一范围与共同可保留窗口，保留source ordinal gap、event record及exact parent；窗口lease只在运行时持有，Stop/失败/结束释放自身需求，之前未保留的event不回填。直接`a.<name>`规划成员，动态使用`a`保留同producer全部成员，不把AST变成Python白名单，也不跨producer拼latest。
- Derive每次发布完整当前结果，用`MonitorCoverage`替换上次估计，不继承输入finite placement或把归约后的1×1×35结果偷偷append进50×3×35。输出保留source primary index和exact publication parent、终态完整seal；其index_by_source只声明可由下游按需取得Runtime history，Derive自身不积历史。输入范围与exact/latest交付策略分开，显式range在live/terminal一致，未声明range的原Processor行为保留。
- 动态Overlay与图像都从各自exact publication取canonical prefix；状态再走公共Scope/Last和facet定位，Last是声明顺序末coordinate而非最后valid值。仅Repeat/Point定位采集cell，site向量保持原子完整；Mean或pool没有独立Boolean判决，多个cells不能伪造共识圈。invalid或不能唯一选择时隐藏，与原same-shot等待规则一致。
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
- 本性能worktree的后续收口：Session warm记忆只保存成功参数tuple，删除半径/幅度/旧chi阈值及其扫描，数值cold+warm竞争不变；RegularImage仅linear loss的proxy使用1e-5收敛容差，robust proxy及全部full refinement保留原精度，保留fresh正负候选，不以warm成功为由跳过cold。旧Front完成回调不再闭包捕获自身Future，共享像素free预算跟随service现有Host数，关闭时缩减且不复用leased块。Curve/Rolling/Facet共用prepared summary的一次valid/范围/孤点扫描；serial Numba核直接读只读stride view，不增加尺寸阈值、OpenMP或第二缓存owner。具体数值证据见Plot performance文档与当前worktree报告。
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
  三个重复owner；唯一popup在最终高度后由delegate column hint、frameWidth与共享scrollbar厚度一次定出宽度与scrollbar policy，不读回viewport chrome。
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
- `start.json`在Start时一次写入run identity、normalized inputs与started_at，之后不可变；
  `run.json`在结束时一次建立（terminal状态、stop reason、last progress、artifact inventory、
  failure）。两者都只创建、从不替换；运行期间不写盘。
- Summary位于run根，domain final位于`final/`，Figure pair位于`figures/`，精选
  candidate/site数据位于`data/`。Figure NPZ contract为`zlc.figure`；同stem PNG只登记为preview。
- Task完成前必须注册所有声明的final artifacts；未注册、文件缺失或路径越出run root均失败。
- run record的`pulse`是文件名与路径；`device_snapshots.sequencer`携带`program`（digest、时长、loop、scan表、repeats）与`pulse`（填好值的完整文档）。Pulse Editor保存时文档名跟随文件名。
- Stop和failure保留run directory与已注册artifact；Stop时partial-exit writer的失败进入
  observation与stopped记录的error而状态仍是stopped。异常进程退出留下`start.json`而无
  `run.json`，不清理、不伪装成功。

### 2.3 Figure与Viewer

- 公共Figure API严格编码/解码PlotSpec、parameters、size、viewport、selectors、facet focus、classifier、fit与
  typed image overlay；archive先发布，preview后渲染。
- Panel Save只是公共Figure API的adapter，不再维护第二套writer或restore grammar。
- FigureViewer把archive typed Dataset发布为sealed Runtime signals，默认panel从archive exact recipe恢复，且不按shape推断plot kind；保存spec的`kind + cell_kind`在Panel创建前经同一个catalog identity owner解析，semantic vocabulary随后才投影。Add Panel只建立空的fixed-kind `panel-N`，Signal/ROI/Fit派生及后续compose全部走与TaskConsole相同的ConsolePresenter、SelectionBridge和Plot host，不再保留static panel owner。静态host在Bridge订阅前已有accepted fit时，Fit subscription只replay该immutable FitEvent，不重复solve/render；因此ROI与Fit参数都继续发布给后续Panel。
- FigureViewer与TaskConsole Live/Frozen使用同一个accepted PlotSpec、parameter、selector/
  viewport capability contract以及完整Panel Edit：Frozen snapshot/Refresh、Interaction、
  Direct producer、Save figure均走同一ConsolePresenter owner；`panel_only`不再隐藏Panel能力。
  Viewer semantic edit同样只在host accept后更新surface，文件选择默认定位workspace当天data目录。
- `board.commit`首次接受Panel host后在同一owner turn幂等挂载Selection/Fit Bridge；真实Qt首个drag/release已验证可立即发布ROI，不等待下一display beat。
- FigureViewer同页Data authoring固定为Dataset、Axes、Data三个全宽Fluent frame。Axes只提供
  Add/Edit/Delete与name/length/unit/domain（Repeat/Point/Cell-data）及typed coordinate values；
  axis values使用一行横向virtual table与自身scroll，
  不编辑role/fate、coordinate labels/frame或axis顺序，也不保留独立Coordinates/Label表。
  Data显示与Panel title相同的三domain shape/axis摘要，显式选择Rows与Columns axes；其行列header
  只读显示对应axis values，其他axes用Setting式Scope按真实coordinate value选择。虚拟
  table按当前二维slice读取、支持整块复制粘贴，并让Tab/Shift+Tab连续进入相邻cell、方向键
  在非编辑态移动current cell；两张table的row/column数量只改变内部scroll range。普通cell
  修改不得reset model或复制整个slice；blank写入validity而不把空字符串冒充numeric value。
  axis length一次同步resize values/validity/sigma。Apply构造canonical `OwnedSnapshot`并通过
  现有sealed Viewer producer发布，自动交给普通Panel，但未Save前仍是unsaved working copy；
  Save Figure As成功后才清该状态，并继续调用公共Figure writer。已有archive的lineage/device
  事实只读保留，manual-create/edit从真实publication追加系统provenance，纯manual数据不伪造device。
- Manual Axis Delete直接保留当前Scope coordinate的slice并删除该axis；允许最后一个Repeat axis
  退化为单row无具名axis domain。同domain的name/unit/length编辑保留原role/labels/frame，只有
  显式换domain才换成该domain的generic role。
- Existing Figure只改data或axis metadata时原样保留Repeat/Point carrier与`axis_codes`；只有
  Add/Delete、跨domain移动或length改变才把受影响domain明确重建成dense authored map，不能把
  sparse/serpentine scan在普通Apply时静默膨胀成Cartesian product。
- Plot共享手势现要求Area press只arm：无move时0 candidate、0 selection callback、0 overlay render；首个真实held move才启动preview。Qt double-click的首个press/release和Notebook explicit double均不得生成Area，已有Area空白click清除语义保留但不再通过degenerate draft实现。
- GUI探索使用既有ConsoleBench、真实Qt控件事件和zlc_ui截图，记录可重放操作及front安装/paint/accepted状态；截图不能替代瞬态状态证据。所有运行结果保持在ignored目录。选区镜像统一提交partial configure并复用Edit配置接受入口，删除只呈现像素而不更新冻结描述的旁路。
- Numeric axis继续由SmartOffset/locator防重叠；既有Dataset的显式coordinate labels全部忠实
  保存与显示，不做renderer端抽稀。Manual Data editor允许编辑axis coordinate values但不提供
  labels authoring，因而不再维护partial-label草稿或补全规则。
- SLM Feedback camera preview第一轮后停更的根因是`holds_live_revision`只识别裸`OwnedSnapshot`，带site overlay的`ImageFrame`把第二generation的revision 10误判为旧run的`10<=10`并cancel。现统一解包snapshot；真实两轮Camera Measurement→Panel从generation A前进至B、同host复用、无busy/error，Plot/Workbench seam各有回归。
- Lineage保存root、event nodes和direct parent IDs；Viewer验证引用、reachability和cycle后
  投影为tree。
- 显式消费Dataset的Measurement worker在每次保留值时将exact source publication随同一次Runtime commit提交；Scan Flow因此从Scan event沿真实parent回到Measurement/Processor，Save与Viewer不得按signal名查latest。Device tab从node run record显示baseline working point，并另列event ranges实际引用的active override epoch。

### 2.4 Domain Task artifacts

- Calibration：每run一个folder，final Calibration JSON、summary JSON/text、精选报告Figure
  NPZ与PNG。默认不保存全部raw frames。
- Calibration threshold method默认Gaussian、可显式选Empirical。Gaussian模式对每site全部finite short-shot values做无标签双Gaussian mixture fit，并在两均值间解析求拟合population-weighted分量曲线交点；真实reference labels不参与Gaussian fit/weight/threshold。fit或交点无效site才用全部有效labelled samples上最大化overall correct fraction的Empirical fallback；Empirical模式全部使用该empirical路径。Histogram线是最终classifier threshold，Gaussian曲线直接携带Calibration同一组分量而不在Plot二次拟合；fallback只有最终线而无伪造理论曲线。`actual_fidelity`是最终threshold在全部有效真实Calibration数据上的overall正确率，`gaussian_fidelity`是Gaussian threshold按其拟合population weights积分的理论正确率。
- Calibration SiteMap detector使用真实相邻frame difference＋完整average两条证据并集。difference逐transition按自身背景噪声标准化；重复中等变化使用run-measured binomial bar，单次明显变化使用pixel×transition family-wise bar；steady/high-loading由average保留。两条路径都不能低于authored `detection_sigma`，candidate identity只来自average local maxima。旧even/odd half state、split veto、absolute per-frame sighting、单帧亮度gate（max level z与其pixel×frame family-wise cut）与global saddle owner全部删除。difference候选按summed change magnitude相对spot尺度外环判定为峰（`_candidate_peaks`）：邻居暗环凹底的未加载cell不被admit，暗环里的弱trap仍被admit。
- Calibration新增默认关闭的`Review detected sites`。开启时capture与detect各执行一次，候选SiteMap通过`calibration/review` companion signal同时进入Monitor和modal point review；operator可单点、列表或框选排除零到多个ghost sites，确认后最终SiteMap重新连续编号并只运行一次全部下游分析。review使用`FluentDialogWindow`和完整Fluent control family；`zlc_ui`拥有view，`zlc_plot`只拥有Image point gesture/overlay，Workbench组合。candidate/excluded/final映射进入Calibration report与summary，`site_review.npz/png`保存候选和排除结果；取消等同Stop。Runtime的唯一operator request/response lifecycle为未来人工Scan axis保留复用边界，但当前没有Scan consumer或UI。
- Temperature：final JSON、summary和生存率Figure NPZ/PNG。
- SLM Feedback：保存输入摘要、stable site table和逐candidate精选BOX samples、fit、weights、
  actions、metrics、phase-change fact与command receipt；不保存raw camera frames。每个
  `candidates/candidate-XXXX.npz`都是可加载/续跑的Science Context，final仍只有唯一selected Context。
- Feedback报告固定包含uniformity history、site signal evolution、weight evolution、selected
  site histograms、initial/selected camera mean和initial/selected phase；每个完整candidate另存
  `candidate_site_fits/candidate-XXXX` Figure NPZ与PNG，使用真正的Histogram cell和Figure API
  的per-site authored full-data mixture fit，不在分箱值上二次求解，不加入Monitor preview。normal或Stop只产生一个final Science Context。
- Feedback Monitor固定自动打开四张图：canonical Camera Measurement逐帧publication经mean reduction得到的带编号site map实时图、observable
  uniformity、site signal evolution和Target share evolution；phase保留为信号和最终Figure但不自动开panel。
- Feedback每shot每site的信号是BOX读出（Calibration BOX几何内像素求和的真实光子计数；不用Calibration的matched-filter权重，也不为未观测site借uniform PSF）；每site只在完整shot batch上做受约束双高斯与full-data ΔBIC>10判定。dark site按bracket（最新观测优先）向loaded share二分或沿方向逐分辨率爬行，在share空间由本轮loop步不与之相反的loaded sites（loop送它同向、或对它无所求；hold的site除外）出资（总功率精确守恒、每site每candidate至多一个分辨率、绝不把任何site压向其自身方向的反面、hold的site份额不动）；bright fraction低于全阵中位数一半的loaded site视为在loading ramp上：hold、不出资、不被识别excitation扰动；probe episode每site一次，方向只由verdict改变。formal-double使用loop gain除以实测plant slope（前6个ordinary update携带±2%识别excitation：正负平衡的log图样在被激励site内平移一个公共对数保持总份额，未激励site绝对份额不变），无adaptive scalar。`probe_combined`计入`maximum feedback updates`，diagnostic candidates不计。
- Grouped Curve与Grouped Rolling共用hover/lock/wheel contract：hover仅轻微加粗，lock才压暗其它lines；无框标签固定axes右上角，lock加`* `并接管滚轮逐series移动。standalone/Facet Curve的孤立valid点使用同一Line2D短横线glyph；invalid仍断线。
- ImagePlot与FacetGrid image cell不再暴露interpolation参数；schema/style/Panel Setting/Edit均删除该字段，renderer唯一固定值为`nearest`。
- Feedback failure与normal/Stop封存同一选择（最佳已完整测量candidate到SLM与`final/`），summary
  记录错误；只有封存写不出时才restore起始phase。未测phase不得成为final。

### 2.5 Device Control与settings provenance

- Generic Device Control只消费adapter的`TunableField` contract，显示Current、Desired、Live apply、Apply、Status、Refresh及active owners；已删除旧的edit-immediate `field_committed/read_values/set_form`路径和demo残余。
- Generic Control的X复用现有Fluent隐藏机制，保留同一device session的窗口、Desired和单位；隐藏时停止Live debounce、撤回未执行字段写入并跳过周期UI投影，重开只读刷新current，卸载/重建或session结束真正销毁窗口。现有正式flow的单个Qt生命周期用例红/绿通过，验证ms单位与草稿保留、隐藏无周期投影、重开读回及session shutdown释放；未做硬件测试。
- RF frequency/power四个policy edge已进入Rigol、Vaunix及Virtual RF的optional Init schema并复用同一Control tunable；空值表示无该侧policy、可随时清回空值。仪器自身limits在Init读出并以`TunableField.device_limits`只读投影；Scan port范围、Control与外部`tune`的有效范围都是policy与device limits逐侧取更紧者，缺失policy edge时该侧就是仪器limit；全空Init不归一化或改写硬件当前值。
- Pylon以运行时`gain`（dB）公开SDK bounds/current与grabbing-safe write；Virtual camera公开`exposure`（s）。固定单位Config/SDK参数名保留。epoch由设备owner报告，Control不比较不同单位或用浮点相等推断增量；Seamless/Stepped保留用户设定坐标，实际回读不替换扫描轴。
- Logic静态requirements与Stepped Scan运行时选择的device ports都形成field claim。DeviceUse按device-specific owner revision原子核风险授权、dependency closure与pending write；字段命令期间不能进入新Logic，owner变化取消尚未执行的write。
- Device I/O只在现有串行worker/adapter command lane执行。Refresh去重合并且属于close guard；75 ms live input在相同policy projection及in-flight write期间保留每字段latest-only值，Qt owner只处理plain projection和已完成readback。
- Device Manager的Remote公布是Session DeviceUse里该device的command claim：本地Logic/command占用时按名拒绝且不公布，已公布期间本地Logic、command、字段写入与rebuild按名拒绝直到撤回；公布的是accepted apparatus而非draft，远端proxy每次Refresh经fields RPC取当前完整字段投影、不缓存bounds。SLM Editor的device状态问句在其串行command executor上问、Qt线程只显示答案，command的交付带回它留下的状态；Qt从不等remote proxy的apply锁。
- CameraFrameRecord在adapter边界冻结settings session/epoch；Pylon无法证明live tune前后的buffer边界，tune之后直到本次arm结束的每个read都明确携带old+new（一次read取走部分旧队列不证明其余帧是新设置），重新arm才回到单一epoch；Virtual在trigger时冻结。Runtime使用event-varying record并保持generation-stable run record，finite/scan/indexed保留范围合并为压缩epoch ranges。
- Figure lineage当前grammar包含每个event record及只解析实际引用epoch的device settings；FigureViewer Device tab读取同一事实。无active Logic的调整不写历史，完整参数状态不复制到每frame。

### 2.6 Plot三进程边界

- TaskConsole/FigureViewer采用固定B/A/C拓扑：B拥有Qt、Runtime、Logic、device通信、PanelState、SelectionBridge与same-shot accept；单一A拥有全部Monitor的DataView/Fit/Render/Compose；单一C拥有Panel Edit、point review、Panel/FigureViewer Save以及Calibration/Temperature/SLM Feedback的Figure render/export。A/C复用同一`RasterPlotHost/PlotSession`，正式Workbench没有B进程内Plot fallback。
- 同一application只有一对A/C；TaskConsole打开的FigureViewer共享并分别持有owner lease，最后窗口关闭才shutdown。A/C崩溃由现有Panel replacement lifecycle恢复，旧完整Front继续可读，不能把partial frame或latest publication伪装成旧surface。
- B→A/C的同一Dataset revision每service只传一次并按host/pending引用计数；A/C→B的RGBA使用只读shared-memory lease，QImage不复制像素。父子消息统一使用owned `send_bytes(pickle.dumps)`/`pickle.loads(recv_bytes())`，避开Python3.13 `Connection.send`临时BytesIO export生命周期错误。
- Domain Task仍在B决定科学数据、路径及非Figure NPZ/JSON并register artifact；只把Figure执行能力由composition注入C。direct/notebook显式使用本地Plot，不把TaskArtifactContext或Runtime变成Plot owner。

## 3. 当前验证状态

- 2026-09-05 新增release–recapture四参数Series fit（A、B、eta、f）；f为普通频率，A=1/B=0可用现有表达式精确固定。解析Jacobian和Numba single/batch已接入，warmer新增3个模型callback。独立182个高精度锚最大函数误差2.22e-16、Jacobian绝对误差4.19e-15；完整FitEngine 32/128/512点中位0.85/2.00/7.24ms，64×128批量33.57ms，已有cache新进程首次fit343ms。详细口径与复现命令见Plot performance文档；这是SEM加权最小二乘，并非binomial MLE。
- 2026-09-04 same-shot/Occupancy显示修复：Runtime与Workbench组等待/重启/延迟提交`34 passed`，Camera＋Occupancy与camera restart`2 passed`，Plot完整输入四入口（含真实RenderProcess）、动态invalid、active-fit配置及真实camera呈现`16 passed`，跨进程front/Save与typed overlay保存重开`2 passed`。清圈和全invalid的完整RGBA与同源无overlay对照0像素差；生产代码净减39行，无新production文件/类。实验机原始偶发现象未在本机复现；已复现并修复的是违背same-shot的缺成员放行与旧overlay继承路径。

- 当前三进程worktree正式证据：process Host/selector/Fit/Fit-event/C Save、共享RGBA与input refcount通过；TaskConsole实际A崩溃自动remount通过；FigureViewer全文件`21 passed`，actual A/C archive/open/edit/Save image及TaskConsole/Viewer两种关闭顺序均通过且所有PID退出；三类Domain Task的existing artifact用例`3 passed`，注入Future writer等待`3/3`。最终owned-wire的15次2M Image stress与DPR3完整Edit/Refresh/C Save均0 BufferError/SharedMemory/resource-tracker warning。最终代码真实DPR3四Panel critical P50/P90/max为`142.80/161.93/188.31 → 89.34/100.30/103.65 ms`，零stall；Grid Save总等待`3895.1→843.8 ms`、GUI最长轮`366.1→6.9 ms`。完整表与内存代价见`packages/zlc_plot/docs/performance.md`。

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
- Histogram的`histogram_poisson_gaussian`/`bimodal_poisson_gaussian`是Γ延拓的连续泊松律（归一化）⊛高斯读噪的密度，A=面积；和其他模型同一条路子，在plot kind给的bin中心与计数上拟合，不看数据来源或单位，没有本模型专属的下限/capability/单位门。编译核：梯形积分、每σ与每个p尺度各十二个节点（节点数随参数变化处模型跳一个积分误差，十二个把它压到1e-9以下，数值差分看不见）、u=0端点Euler–Maclaurin修到O(h⁶)、p表从众数递推填满支撑区、高斯因子两乘递推每64节点重锚一次、质量与均值同表求得、λ<1e-150取零光子高斯；SciPy路径与overlay调用同一核，anchors是mpmath独立求积（1e-6）。核对mpmath：模型高于峰值1e-6处相对误差5e-9，解析雅可比与mpmath求导一致到1e-10；64 bin一次值+雅可比8–58 µs。曾试过按数据bin定网格以保证对参数连续：overlay的细显示网格把节点密度推到所需的40倍、MOT四十cell面板fit_total 124 ms，故改回按σ定。合成Poisson+Gaussian直方图（64 bins、5000样本）恢复：5光子起与格点模型一致，λ与δ和Gaussian的中心/劈裂同精度（30/σ3→29.9±0.09、3.06±0.13；双态60/210 σ3→δ150±0.33、σ 3.15±0.30/3.21±0.95；1000→1000±0.5）；2光子且bin细于一光子时数据带整数comb而本模型画不出（deviance 7.1对格点1.2，σ 0.62对0.3）；λ≲0.5时λ跑低直至零、A与σ吸收峰（0.1/σ0.3→λ 0.0004、σ 0.41；双态0.5/6→λ_L 0.07但δ 5.6±0.14）；16光子bin下σ停在8的半bin下限、Gaussian bimodal拟得更好（0.17对0.90）——用户决定bin选择带来的误差不管。`run_fit_models`（两态8±2/30±4）：single 5.43 ms、bimodal 4.19 ms对Gaussian 2.91/3.50（格点版5.45/3.91）；`run_mot_roi_chain --panel3 histogram`：fit_total 29.8/18.7 ms对Gaussian同场16.4/5.8（格点版22.3/13.9），numeric_fit_batch 14.5对11.9，四面板临界路径106.6对97.0 ms、7.5对7.7 fps，差额是四十个overlay在数十光子宽度下每bin数百节点。用户此前报告的「MOT ROI像素直方图两个fit全错」：single Gaussian的宽坡（中心−35、σ30）是单高斯对「尖峰+长尾」在Poisson deviance下的真最大似然，要拟尖峰须用X-range/Area selector或bimodal Gaussian（尖峰σ1.82+宽尾σ33）；损失函数保留Poisson deviance（bin计数是独立抽样的多项分布，幅度自由时deviance极小点就是多项极大似然，对任何样本分布成立；最小二乘在低计数区有偏）。
- Pulse的API slot值链已落地：`apply_api_values`取交集覆盖（zlc_pulse）；三个scan节点的`api_values` text字段由scan plan editor的表格编辑（只记与pulse不同的项，被plan扫到的slot不显示）。两层顺序：authored → 节点表单（scan table再逐点）。**用户裁决（09-04）：值集文件那套全部删除**——`zlc.pulse.api_values`语法、`<workspace>/api_values/`、Workspace的`api_values`属性与seed、Pulse Editor打开即套用/On Pulse前重读、节点表单的Load values按钮与「current.json在哪些参数上不同」提示，一并清掉；pulse的API参数只属于pulse文件，节点表单只做本次运行的覆盖。板子标定数留在`config_values`那条路上，且不在seamless scan里配置。
- 每个读文件内容的表单字段获得Refresh（`FluentPathEdit`单源，`FormFieldProps.refreshable`开关，`folder`字段不给），console侧`refresh_logic_files`复用打开Edit那条重读路径；Figure Viewer的路径条同样可Refresh（等价于重开同一路径）。实测：外部改写pulse后不刷新读到旧值，Refresh后读到新值。
- 绑定id可在Scan页重命名（slot与api parameter同一命名空间），重复/非identifier/不存在分别拒绝；已推送pulse文件的载入不受影响（codec无白名单，calibration按序号寻址）。顺带修掉scan plan editor的静默改绑：axis指向pulse已不提供的port时保留原port并标注，交给`bind_plan`按名字拒绝，而不是静默选中index 0。
- 多进程合并（18888d2 域重构 + 25944a8 渲染进程隔离）后 task_console 两个基本场景冻死，四个根因已根修（worktree `zlc_v2_mp`，分支 `mp-console-audit`，commit fcfc919）：①管道写死锁（两侧各一条写线程+队列，删掉两把发送锁）②默认 facet 取窗口大小导致 host 根本起不来（只有 shot 历史这一个候选让位，结构轴照旧响亮拒绝）③拒绝后 console 不回滚（configure 记录带回退，回滚 state/lease/port）④fate 交换只读 authored 表（改读生效表）。
  组合矩阵实测（offscreen 真 Qt + virtual MOT，`scratchpad/combo_matrix.py`）：整条链路 camera → Image+ROI → roi_frame → Histogram + FacetGrid，57 个操作组合（window 1/30/60/64/65/200/1000、每一行每一种 fate、四种尺寸、三种 cell kind、三种面板 kind、两个 fit 模型与清除、facet 全格拟合、编辑器开合与其中的编辑、Save Fig、三个面板上的指针拖拽/取消/滚动）：**0 抛异常、0 超时、最长 Qt 事件循环单次 137 ms**；唯一报错是四条正确的容量拒绝（要求 facet 一个 796/99/99/602 坐标的轴），拒绝后面板回滚且保留上一张画面。
  修复前实测对照：同一链路在 >64 行时 host 永不启动、探针在 `grid host` 上超时，进程 1.6 GB 且六分钟无输出。
- Calibration weighted Gaussian/Empirical threshold当前worktree聚焦证据为`11 passed`：已知真值、label-invariance、窄bright＋尾部噪声、Empirical/fallback、Figure archive同模型重放、population-weighted Plot classifier、coordinate roundtrip与warm refresh。另对500组随机Gaussian参数核对解析交点，383组存在两均值间相关根，最大加权log-curve交点误差`3.37e-13`、相对数值最优误差`8.95e-17`。本cut未重跑正式Runtime/Workbench vertical。
- Calibration site review当前聚焦证据：Runtime精确operator request/response与Stop、saved-frame完整review链及全descriptor virtual Calibration保持通过。正式`zlc task_console --template virtual`science路径以8 samples检测24 sites，排除`site_0000`后最终Calibration为23 sites、terminal移除全部自动preview；`site_review.npz`42,208 bytes、PNG 145,791 bytes，FigureViewer current reader成功重开。此前`parent=None`测试遗漏了真实parent会把frameless QWidget降为child的错误；当前`FluentDialogWindow`以`Qt.Window + WindowModal`保留Fluent顶层身份，真实parent测试核对active modal、确认响应及title-bar关闭，QTimer/TaskConsole同类回调中的nested loop正常退出。`zlc_ui.capture_window`取得精确1152×653 shared-screen capture，Fluent title/body边界为32/32、无native dialog chrome，35-site状态与controls完整显示。
- SiteMap detector真实red为8帧50%-loading site在0/2/4/6出现：旧实现full-average `702.04σ`仍被split veto漏掉；新实现记录7个相邻变化、最大change `355.67σ`，定位误差0.030 pixel且同一35-site阵列全部找回。20个随机seed覆盖698个至少加载一次的sites，漏检0、spurious 0；完整site-detection `7 passed`，saved-frame review＋Workbench vertical chain `2 passed`。
- 后续adaptive gain/formal-update accounting与Curve hover/lock切分运行6个直接聚焦用例，结果`6 passed`；未重新运行100-shot验收。
- 紧凑Science Context当前证据：SLM Editor完整文件`22 passed`；strict Context与Feedback candidate/Stop/failure边界`10 passed`；最终三条直接边界`3 passed`。X15213全尺寸体积、Pattern/composite逐元素roundtrip和8-bit phase-code roundtrip均来自当前worktree；未运行100-shot。
- 固定nearest清理运行standalone/facet artist、Workbench parameter surface及Fluent Setting/Edit四个聚焦用例，结果`4 passed`。
- Device Control当前回归：Workbench完整`425 passed`；Runtime完整加Figure grammar `112 passed`；adapter/camera/scan受影响组`53 passed`；Device Control Qt、风险revision、refresh close guard、in-flight latest-only和demo直接证据均通过。Atom完整回归同时暴露并修复Temperature sibling event record、Feedback输出声明和三条terminal/Stop残余；100-shot virtual Feedback仍为既有`34/35`上限，未用放宽断言冒充通过。
- FigureViewer此前以formal launcher和`zlc_ui.capture_window`在真实Windows屏幕完成四条1152×653验收：current archive默认Image Monitor、点击Add panel新增Curve、从Setting点击Edit进入共享Fluent `PanelEditorView`、以及多层Flow展开树；四次均保持shared 90% window尺寸和固定左栏。右侧复用TaskConsole `ConsoleBoardView + PanelCardView`并置于白色work surface，支持每panel切saved dataset、alternate plot kind、Setting/remove/order与closable Edit；Panel Edit现与TaskConsole完整共用Frozen snapshot/Refresh、Interaction、Direct producer和Save figure。用户当前重新裁决Info readout必须统一multiline并按实际visual layout紧包；旧的无换行单行分支会cutoff长内容且不能作为phantom inner-scroll的替代修复。固定Plot kind从Setting删除，动态Signal keyed-choice在reconcile写值前更新choice domain。
- FigureViewer Info页是树：InfoPane每页一棵两列`InfoTree`（名字 | 值），record逐层展开、分支行内联标量摘要、长数值列表按个数与范围显示、值由wrap-anywhere delegate在列内换行不cutoff；页顶filter同时匹配名字与值并展开到命中处；Ctrl+C与右键菜单复制整个值或名字路径；Raw页是文档四个section的嵌套树；Flow node携带`row=(tab, label)`，点击card切页并选中该行。`FluentReadoutMultiline`不再用于InfoPane。
- FigureViewer Logic/Devices/Flow当前根修：archive内部`event-N`只作parent引用，Logic页以真实Logic identity显示递归去除device字段后的run参数；Devices页用run record的stable role→instance映射解释run/event snapshots，按实际device聚合并给每项保留Logic、sequence与scope，缺映射/identity/device key一律拒绝而不猜。Flow原位删除QTree owner，Workbench只投影唯一Logic/Device nodes和causal/device edges；Qt以layered+barycentric布局、独立edge ports与long-edge lane绘制，典型100 nodes同步构建约6.5 ms，3-device、diamond、真实DFS汇合及10-node长链均无edge穿node，长链horizontal range为0。Calibration/Temperature normal与partial report、SLM candidate/report、Seamless/Stepped/Temperature live均保存实际用到的device facts；Feedback pre-shot只记录SLM，post-shot冻结同candidate三设备，failure rollback不改变已存candidate provenance；Stepped tunable以完整scan values及逐点readback等值contract记录，不复制event history。聚焦回归`67 passed`，另Console Logic`34 passed`；formal Windows real-screen capture为1152×653、DPR 3、3-device Flow无横向scroll且节点/箭头无重叠。
- 公共Panel Setting现复用master的page-local `FluentOverlayFrame` owner，并以固定identity（`Setting · panel-N`）作为可拖header，不读取可编辑title/signal/structure；右上角紧凑`×`只隐藏Setting。TaskConsole与FigureViewer因复用PanelCard同时获得该行为，Panel删除仍是card header的受保护命令。
- Exact Scan Panel恢复当前证据：真实event chunk为`1×1×3×5`、canonical为`2×(65×2×2)×3×5`的Signal经实际SignalPlane与Plot host由真实`field.x=65`触发>64拒绝；拒绝前后Setting均保留`field.x/y/z` fate且不再出现phantom `point`，独立Curve Panel title保持canonical axes，Fluent form在`fit_unavailable`同时仍含三个Semantic controls。精确目标`20×(10×10×10)×3×35`的title authority输出`(20)×(10×10×10)×(3×35)`。多维FacetGrid默认最外层真实scan axis，不再以flattened point rows制造1000 cells或phantom point-row restriction。相同live projection与仅title metadata变化均不reconcile Setting form；固定Plot kind不再进入Setting，FacetGrid只保留可编辑Cell kind；Facet默认、feasibility、真实拒绝与Fluent Setting聚焦证据`22 passed`。
- Exact Scan terminal/Frozen根修当前证据：真实`20×(10×10×10)×(3×35)`canonical Dataset从partial Live publication开始，原子提交`field.x→Facet, field.y→Y, field.z→X, pair/site→Reduced`后，Live、运行中Frozen及terminal seal后重新创建的Frozen host均保持同一schema fingerprint、物理shape `(20,1000,3,35)`、resolved roles和`[-0.5,9.5]×[-0.5,9.5]` limits。根因三处均删除：multi-fate逐行修复导致回退默认35×3、host accept后以1×1×3×35 event schema覆盖canonical surface、以及histogram threshold/shape-only viewport无条件重放到image。当前实现使用atomic fate assignment、canonical accept metadata、resolved capability interaction和schema/spec view identity；Plot semantic/feasibility/facet/threshold聚焦`52 passed`，Workbench canonical/Frozen/retarget/save交叉聚焦`10 passed`。
- Plot/Runtime/Workbench当前candidate直接回归：Plot `534 passed`、Runtime `107 passed`、Workbench `435 passed`；Atom对Figure/hosted-node新contract的direct用例`1 passed`。这些结果来自当前tree，不复用旧Exact Scan cut的计数。
- Pulse repeat三层根修：`PulseBracket/PulseSequence.bracket`只负责timeline内部连续区间；主界面新增持久化`run_repeats`（默认0=∞），Pulse Scan保留`scan_repeats`。无scan时Run repeats控制完整Pulse；有scan时每个row执行Run repeats次后才前进，整张table由Scan repeats重走。`shots_per_point`与Seamless `repeats`分别只做本次run_repeats/scan_repeats override，不改写Bracket或复制rows。Wire/RTL使用独立`LOOP_COUNT/RUN_REPEAT_COUNT/SCAN_COUNT/SCAN_REPEAT_COUNT`并在同一FIRE内完成全部seam，三层不再压平到一套execution count或保留旧兼容路径。Bracket左右post复用Schedule drag owner，可拖到任意合法gap。
- Seamless duration scan根修当前证据：绝对period保留32-bit nominal base，25-bit signed slot只承载delta；整张table自动选择最小整数tick scale，最大127且DAC恒为1。实际量化rows统一进入compiler、wire、readback、Pulse Editor Run/Sync/Hold/Step、Seamless Dataset coordinates/run record与Temperature companion/artifact；distinct authored points若量化坍缩会在device前拒绝。三层repeat改造后已重跑Pulse/Editor/Seamless/Temperature纵向与RTL oracle，并以新ABI完成Vivado纯build；结果见第4节，不复用旧bitstream证据。
- 通用Fit表达式减量候选：Panel Setting/Edit只提供单行`name=value`精确fixed与`name=guess(value)`初始猜测；fixed复用既有bounds请求通道的相等端点作为内部exact marker，但普通及regular-image solver都会把该维度真正移出optimizer，free-only计算DOF/Jacobian/covariance，all-fixed不启动optimizer。表达式按painted单位输入，PanelState/Figure只保存canonical fixed/initial；语法、unknown或domain错误只在DisplayDescription保留transient draft/warning并继续同model自动fit。Curve/Histogram/Image/Facet/Rolling共用FitSession请求，Console与Viewer共用同一Panel投影；fixed参数的误差publication为invalid。相对`17629d1`无新增production文件或类、production净增376行、test净增163行；此前临时`fit_target.py`与第二套canonical validator已删除。减量后直接聚焦`38 passed`，Plot全包`534 passed`、Console View全文件`31 passed`，另直接验证普通/regular all-fixed均返回`all parameters fixed`。Workbench全包在先运行36项后仍稳定暴露既有camera-restart selector顺序失败，目标test单独运行`1 passed`；该问题属于下一独立cut，不混入Fit提交。
- Camera restart selector顺序根修：`_refresh_signal_choices`原来把“首个surface尚未accept、因此`binding.host is None`”误当成“panel尚未mount”，在已有initial `PlotPanelPort`忙于首帧时又启动第二个retarget port；后完成的候选会关闭已接受crosshair的port。恢复路径现在只在唯一生命周期真相`binding.port is None`时创建port，Board继续独占已有port的首帧accept；没有新增状态、helper、selector/restart特判或测试函数。原必现的auto-inference→camera-restart顺序`2 passed`；Workbench全包该缺陷已消失，结果`436 passed`。
- 长Task partial artifacts：Runtime在worker failure/Stop边界调用domain writer；Feedback普通异常从最后完成candidate生成6组Figure后rollback，Temperature从已提交survival保存partial curve/Figure，Calibration从最新完整三帧cycle保存partial capture（分析完成则保存完整报告）。`run.json`只索引这些已完成文件，不再是失败run唯一内容。
- Feedback的`candidates/candidate-XXXX.npz`现为标准Science Context；operator可在既有Science Context输入中手动选择它作为新run起点。过程数组移至`data/measurements/measurement-XXXX.npz`。新run从candidate 1开始并使用本次authored update预算；没有resume输入、自动旧run查找、续编号或旧run预算继承。
- Pulse STATUS位仍为LOADED/RUNNING/DONE/ENGINE_ERROR/UNDERFLOW/LINK_ERROR，新增独立完成命令协议和CTRL22..25的command/ACK信息；fingerprint为`0x5A83C4CA`，旧板或旧server不进入新执行路径。SAFE由板端隔离输出且保留clock/program；Fire回复丢失用同command ID重试，不猜旧LOADED gate。运行/完成读取一次status/cursor块，日志记录单一结果与command ID，不保存或伪造两次读回。既有Remote cancel旁路保留，正常DONE无需额外SAFE命令。
- 第一次installed software尝试曾在重负载下出现一次本地SLM测试TCP connect timeout；
  同一wheel的精确case随后连续5/5通过，第二次完整installed software lane通过，因此没有
  用该不可复现事件改动产品remote timeout或server逻辑。

## 4. 仍有效的FPGA build/timing证据

以下证据来自2026-09-03在当前三层repeat tree上强制执行的Vivado 2019.1纯build，不代替实验板验收：

本节是旧命令ABI的历史build证据，不能用于确认当前`0x5A83C4CA`的资源或时序；当前版本由操作员在实验机build/program。

- Vivado 2019.1 fresh project完成全部IP、top synth、place/route、reports和bitstream。
- Routed setup WNS `+0.193 ns`、TNS `0`；hold WHS `+0.036 ns`、THS `0`；全部约束MET。
- 12条bus-skew全部MET，0 violated，最差实际skew `0.905 ns`、slack `+19.095 ns`。
- 资源：19632/20800 Slice LUT（94.38%）、14138/41600 FF（33.99%）、76/90 DSP（84.44%）、41/50 Block RAM tile（40 RAMB36 + 2 RAMB18，82.00%）。
- 新register layout fingerprint：`0x5A86511A`；Bitstream SHA256：`FD7BDF79A8865A6961BA536275AF070BA84D74B24B479248D2FDC2CA62D5656C`。
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

## 7. Plot交互性能：基线与裁决（2026-08-27）

> 本节只记录**基线、裁决、和仍然开着的事**。
> 每一次尝试的经过、被实测否决的方案、以及我自己的测量失误，写在**做那件事的
> 那个 commit 的 message 里**——那才是不会过期的记录。计划文档不做 changelog。

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

### 实测基线（2026-08-27，4x4 单面板，free-running 生产者）

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

组合场景：四面板并存 2x2 手势 image 7.2 / 3D 30.9 / curve 6.3 /
rolling 6.3 ms；ROI 链 相机 9.0、ROI 3D 面板 38.5 ms；带 fit 的 rolling 6.3 ms。

**同一杆秤的对照**：heatmap 的选框手势走 overlay（只重画矩形）4.4–7.3 ms；
heatmap 的**中键 pan**（同样整幅重画）**30.3 ms**；静源 3D orbit **29.7 / 30.7 / 31.2 ms**。
3D 与 heatmap 的整幅帧持平——那是这套系统重画一幅画的地板。

### CPU 与内存（free-running）

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
