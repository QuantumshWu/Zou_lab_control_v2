# 待用户裁决账本

状态：用户已完成顶层裁决；当前解释见[USER-DECISIONS-2026-08-17.md](USER-DECISIONS-2026-08-17.md)。本文细ID仅作traceability，尚未开始代码实施。

## D-001 — 文档冲突由谁裁决

事实：旧规则要求执行者自行选择；当前用户要求最终由用户裁决。
当前审计规则：以当前用户要求为准。
状态：已由用户裁决。

## D-002 — Live fit显示时序

选项：

1. `data@N + fit@N`原子一起显示，接受fit决定刷新率；
2. data-first，先显示data@N，fit@N完成后仅在仍匹配N时补overlay；
3. live数据不自动逐revision拟合，只在显式请求时fit。

待补：fit-derived signal和same-shot board在各选项下的准确行为、性能数据。
审计建议：live Monitor采用选项2；frozen/manual Save Fig仍等待fit完整。
状态：OPEN。

## D-003 — Selectors Off的交互含义

选项：

1. Off关闭plot全部交互，wheel滚Monitor；
2. Off只关闭selector creation，zoom/pan/focus常开；
3. selector开关只管selector，默认wheel滚Monitor，使用Ctrl+wheel或其他明确手势缩放plot。

状态：OPEN。

## D-004 — Camera多frame的数据真相

选项：独立`frame_0...N` sibling signals，或单一`frames + READOUT_EVENT axis`。
当前代码采用后者，根文档正文与部分历史测试仍采用前者。
状态：OPEN。

## D-005 — FacetGrid cell kind

选项：TaskConsole固定Curve，或允许automatic/Curve/Image/Histogram。
当前代码采用可选；根文档有互相冲突描述。
状态：OPEN。

## D-006 — finite Measurement的live truth

选项：

1. live只发布最新完整cycle，terminal发布完整Dataset；
2. live发布固定geometry的累计Dataset，但future cells必须invalid，并使用增量writer/processor避免O(N²)。

状态：OPEN。

## D-007 — 统一live发布API

选项：一个简单、线程安全、Host-owned的`publish_live(outputs)`入口；或保留slot对象但所有插件实现同一标准。当前三套plugin slot与一套无人使用的Runtime live框架并存。
与D-038为同一gate；D-038补充了Host-owned materialization lane细节，最终只保留一个裁决结果。
状态：OPEN。

## D-008 — Overlay真相源

选项：Occupancy发布typed、自描述、可保存重放的overlay sibling；或允许Workbench依赖active node临时组装。
审计初步推荐前者，但具体payload仍待深审。
状态：OPEN。

## D-009 — SiteMap canonical坐标

选项：

1. SiteMap永久保存sensor coordinates，分类时投影到当前ROI local pixel；
2. SiteMap保留calibration image-local coordinates，但任何消费者都必须携带并应用FrameContract transform。

当前实现混用两者。
状态：OPEN。

## D-010 — Task active期间可修改什么

选项：完全冻结所有状态修改，仅允许查看/Stop；或允许纯布局操作和准备其他stopped drafts，但禁止硬件与当前Task draft改变。当前Presenter和真实Qt UI语义相反。
状态：OPEN。

## D-011 — Runtime无人使用的通用框架

选项：按当前产品消费者删除exact/monitor/builder/live旧体系；或明确把`zlc_runtime`保留为未来通用库。后者与“默认删”原则冲突。
状态：OPEN。

## D-012 — `zlc_plot`产品边界

是否继续支持Notebook、standalone Qt parameter panel、PulseTimeline和独立LivePlotController。该决定会区分必要公共API与历史残余。
状态：OPEN。

## D-013 — Pulse重复术语与API

是否继续让whole-pulse `RepeatRegion`兼作shots，还是分开局部bracket、shots、table sweeps、plan repeats，只在硬件边界翻译。
审计建议：分开；RepeatRegion永远只表示timeline内部循环，shots属于execution层。
状态：OPEN。

## D-014 — 绝对same-shot的硬件证据

当前camera ordinal只能检测可见frame gap，无法检测“trigger丢失但camera counter仍连续”。若要求绝对same-shot，需要真实camera/pulse提供cycle marker、trigger counter或等价事实。
状态：OPEN。

## D-015 — SLM Feedback接管什么phase

选项：只更新Pattern base并显式保持Editor pupil/Zernike/steering；或允许Task接管完整science phase。当前Feedback只有target array，缺少Editor上下文。
与D-055共同构成G6：本项裁phase authority，D-055裁context传递方式。
状态：OPEN。

## D-016 — 100 shots的验收含义

选项：100 shots只用于快速coarse update，最终1%使用独立大样本；或要求100 shots直接证明1%。当前约10% per-site SEM与0.5% validation gate不相容。
状态：OPEN。

## D-017 — Feedback最终要均匀的observable

公开all-shot fluorescence均匀与trap depth均匀在depth-dependent loading和AC-Stark fluorescence下不是同一量。需明确当前Task只优化哪一个，以及未来独立trap-frequency/light-shift反馈边界。
状态：OPEN。

## D-018 — Generation/revision模型

选项：EventRef唯一管理run/causal identity、snapshot revision只管理数据内容；或继续允许每层独立编号。当前至少四套编号共同参与stale判断。
与D-066共同构成G2：本项裁两个identity域，D-066裁content ref injectivity。
状态：OPEN。

## D-019 — Package独立性还是单一产品distribution

当前根distribution捆绑八层，同时每层仍有独立pyproject/version/contract；代码已经出现未声明反向import和sys.path注入。需裁决是否真的要求每层可独立安装。
与D-063为同一部署gate；以后以D-063/G1表述为准。
状态：OPEN。

## D-020 — 历史文档保留策略

选项：删除已被新审计取代的survey/acceptance/goal archive；或移入明确的历史目录并保证测试不再把它们当权威。
状态：OPEN。

## D-021 — Overlay候选不兼容时在哪里拒绝

选项：UI只列出与image共享lineage的overlay；或UI按contract列出、绑定时由runtime明确拒绝。无论选择哪一种，都不能继续把两个独立latest静默组合。
状态：OPEN。

## D-022 — Layout持久化哪些interaction

选项：

1. layout保存可复用authored selector，viewport/focus只进exact figure archive；
2. selector/classifier/focus/viewport全部保存，但必须携带source/schema/facet identity并验证兼容；
3. layout完全不保存interaction，每次打开干净视图。

状态：OPEN。

## D-023 — Classifier threshold作用域

选项：按semantic facet coordinate保存完整override map；只允许一个全局threshold；或threshold纯临时不持久化。当前PlotSession vector和PanelState singular record不等价。
状态：OPEN。

## D-024 — Panel derived-output默认策略

当前`published_outputs={}`表示所有当前及未来outputs隐式开启，新增fit参数会自动发布，`roi_frame`也可能产生昂贵复制。选项：全部显式opt-in；只默认开启scalar summaries；或保持当前。
状态：OPEN。

## D-025 — Overlay的repeat语义

选项：保留每个`(repeat, point)`状态，只在真正pooled surface上聚合；或产品明确只支持repeat consensus并禁止逐repeat facet overlay。当前实现提前把分歧压成UNKNOWN。
状态：OPEN。

## D-026 — 历史archive的schema fingerprint兼容

若删除snapshot ref中的schema fingerprint，需要决定：继续读取并忽略旧字段；提供一次性迁移；或只保证当前新格式。该字段不是当前数据恢复必需，但已写入历史archive。
状态：OPEN。

## D-027 — Restart时多panel是否必须原子换代

当前steady-state按same-shot cohort原子呈现，但generation replacement绕过cohort。选项：严格原子换host；或接受Restart边界短暂不同步并明确降低保证。
状态：OPEN。

## D-028 — SLM Feedback按Stop后的terminal语义

当前Stop让Host标`cancelled`，但Task重新apply best/latest durable phase并保留candidate NPZ；Workbench又因cancelled没有正式artifact result。选项：

1. Stop是真cancel，恢复incoming；
2. Stop是“提前接受best”，应形成成功/partial-success terminal并正式返回artifact；
3. 提供独立“Accept current best”和“Cancel/restore”动作。

状态：OPEN。

## D-029 — Calibration自动preview显示什么

选项：显示完整long/readout/long三帧facet，或只显示最后一张二维图。当前代码和物理诊断更支持三帧；旧根文档写单图。
状态：OPEN。

## D-030 — Calibration preview终止后是否保留signal

选项：真正transient，terminal时panel和signal都退出；或将最后cycle显式发布为FINAL evidence。当前用DatasetCoverage把latest preview伪装exact，只删自动panel但保留signal。
状态：OPEN。

## D-031 — Temperature是否保留raw scan output

当前发布`scan + survival + survival_rate`三份数据；旧目标文档写只有后两份。保留raw scan有利于重分析，但增加live/存储成本。
状态：OPEN。

## D-032 — Processor是否自动开primary preview

Occupancy当前不自动开图。选项：Processor默认不preview、由用户手动选signal；或所有可视Logic Node都声明一个primary preview。若选择后者，Occupancy应展示rate、site grid还是judged frame需要定义。
状态：OPEN。

## D-033 — “Measurement始终live”是否约束direct notebook API

推荐只强制Product-hosted run；direct `.measure()`由调用者选择iterator/callback。若要求direct API也隐式发布，会把SignalPlane副作用带入纯调用路径。
状态：OPEN。

## D-034 — SLM candidate内部preview cadence

当前完整100 shots结束后才发布phase、camera average和curve。选项：按固定时间节流、按shot chunk、或每shot提交给latest coalescing seam。审计建议phase应用后立即可见，camera running mean按有界时间节流，curve只在candidate完成时增加点。
状态：OPEN。

## D-035 — Stopped finite partial的产品语义

是否明确要求：固定authored schema、unfilled invalid、retained、non-transient、可Save、可由Processor one-shot消费。审计建议全部满足；当前结果取决于UI freeze时序。
状态：OPEN。

## D-036 — Processor跟随策略由什么声明

finite exact scientific processor应处理每个commit；纯display derivation可latest。当前Host从`DatasetCoverage/MonitorCoverage/None`猜执行模式。选项：由input contract显式声明，或统一exact并在presentation层coalesce。
状态：OPEN。

## D-037 — Latest Processor并发

选项：每processor内部串行、不同processor在有界共享pool中可并发；或保留全局单worker并接受任意慢processor阻塞全部Occupancy/selection。审计建议前者。
状态：OPEN。

## D-038 — Runtime live提交方式

选项：

1. `context.publish_live(outputs)`直接push immutable bundle；
2. 统一Host-owned slot/materialization lane，producer提交带token的raw/immutable state，禁止Plane/UI pull plugin；
3. 保留各plugin自制slot。

审计排除选项3。1与2需根据真实camera snapshot materialization profile决定。
与D-007为同一gate；旧ID仅保留traceability。
状态：OPEN。

## D-039 — Camera↔Sequencer trigger wiring authority

选项：固定一个canonical logical port（如`emCCD`）；或在installation中为每个camera/sequencer pair声明mapping。Virtual私有默认不能代替真实apparatus配置。
状态：OPEN。

## D-040 — Camera repeat=0的acquisition mode

当前Pylon把某些参数解释为free-running，DCAM固定external，Virtual由构造参数决定。选项：UI显式选择`free_running/external_triggered`且不支持者拒绝；或定义为device-native preview并承认跨adapter不同。
状态：OPEN。

## D-041 — Temperature双帧protocol

当前20ms sensor与5.02ms trigger interval不相容。选项：

1. 保持20ms integration并增加足够recapture/等待gap；
2. 使用约5ms integration并重新确认Calibration threshold可比性；
3. 为Temperature相同双帧protocol单独Calibration。

状态：OPEN。

## D-042 — Calibration pulse API与gap ownership

三个duration绑定用parameter ID还是1-based位置；固定gap由pulse作者维护还是Task根据actual exposure派生。审计建议parameter ID，并在fire前核event role/cadence；gap策略需用户选择。
状态：OPEN。

## D-043 — Seamless Scan允许什么source

选项：只允许声明为pulse-gated/lossless并与selected sequencer有apparatus association的source；或增加明确free-running sampling/gating语义。当前任意LIVE按publication顺序冒充row不可保留。
状态：OPEN。

## D-044 — finite Camera buffer策略

选项：完整run driver buffer；或有界ring/chunk并在overrun时loud fail。需结合实验机qCMOS frame size、最大cadence和可接受内存预算。
状态：OPEN。

## D-045 — Capture provenance最低要求

是否要求artifact保存足以事后复核cycle grouping的ordinal/stamp规则、terminal count和pulse window facts。审计建议要求；不必保存无法解释的全部vendor字段。
状态：OPEN。

## D-046 — Scan Dataset是否保留sweep/shot身份

选项：sweep成为显式scan coordinate，repeat保留shot×source repeat或结构化labels；或继续flatten全部repeat、只在metadata保存R/S。若需要分析drift，后者不足。
状态：OPEN。

## D-047 — Off-grid pulse/scan值

选项：Start时materialize canonical actual并将其用于Dataset坐标，同时metadata保存authored；或严格拒绝off-grid。当前静默round且Dataset写authored不可保留。
状态：OPEN。

## D-048 — Pulse Editor Sync承诺

选项：完整同步executable application truth（rows、sweeps、forever、slot/actual values），明确不恢复API authoring declarations；或取消反写，仅显示只读applied摘要。
状态：OPEN。

## D-049 — Delay DONE语义

选项：finite completion直到所有delayed TTL/DAC tail完成；或DONE只表示timeline engine结束并公开tail状态、由caller决定何时safe。审计建议前者。
状态：OPEN。

## D-050 — Saved-frame Calibration是否claim硬件

当前纯offline folder reanalysis仍无条件claim camera+sequencer并加载pulse。是否允许conditional claims、不新增第二node。审计建议允许。
状态：OPEN。

## D-051 — Stepped dynamic device claim策略

ScanPlan使用的每个`device:<key>:<field>`必须进入DeviceUse reservation。若该device同时是source producer owner，是拒绝该组合，还是协调停止并由scan接管，需要用户裁决。
状态：OPEN。

## D-052 — Calibration finite执行

当前forever pulse由camera数够R cycles后截停，无DoneReport。是否改为显式finite R cycles并验证DoneReport。审计建议改。
状态：OPEN。

## D-053 — Calibration sample默认值

根文档写300，代码schema默认200。两者都不是物理常数，需要用户确定默认实验预算。
状态：OPEN。

## D-054 — Cycle count的public API

选项：正式接受`write_scan_table(((),), sweeps=N)`表达无slot的N cycles；或增加一个很薄的明确cycle-count入口。审计倾向明确入口，但不得再引入第二执行机制。
状态：OPEN。

## D-055 — Feedback science context来源

选项：从当前open Editor私有内存读取；使用显式science-context artifact；或bare target/full-phase takeover。审计推荐显式artifact，冻结objective、numeric pupil、Pattern authority与operator wavefront；UI不应成为Task truth owner。
状态：OPEN。

## D-056 — Real SLM未知初始command

X15213无法读回当前canonical phase时，选项是禁止Feedback直到一次known Send/artifact takeover，或Init主动发送一个明确phase。禁止继续把未发送的zero伪装成last-commanded truth。
状态：OPEN。

## D-057 — X15213 serial profile的位置与证据

选项：serial-specific phase response继续随Python package发布；或移到installation/workspace calibration artifact，保存device subtype、raw measurement/vendor provenance、温度/波长与response timing。审计推荐后者；当前`LSH0804382.json`只能标`UNVERIFIED DEVICE DATA`。
状态：OPEN。

## D-058 — Correction wavelength policy

选项：只允许与current wavelength一致的vendor map；或正式支持换波长并定义一般二维unwrap/residue/metadata/验收。当前X后Y separable unwrap不能被称为一般二维转换。
状态：OPEN。

## D-059 — DVI transport产品级别

选项：继续当正式X15213 transport；或在controller mode、GPU scaling/color/dither、active raster、gray drive、orientation、settle与close lifecycle完成真机验收前标Experimental。审计推荐后者。
状态：OPEN。

## D-060 — SLM command/candidate artifact复现等级

选项：只保存canonical phase；保存足以重建gray的profile/wavelength/orientation/correction revision；或同时保存exact gray sidecar与transport/readback/outcome receipt。审计建议至少第二层，并为accepted/Stop retained candidate保存command receipt。
状态：OPEN。

## D-061 — Dense Gaussian/Flat Top solver范围

是否把dense image solve列为近期产品性能目标。若是，应先修MRAF signal/noise region、初相、FOM和early stop，再决定CPU/GPU backend；不得用GPU加速当前无意义的全raster Gaussian约束。
状态：OPEN。

## D-062 — Feedback target范围

当前Task是否明确只接受与Calibration SiteMap一一对应的sparse support，还是要设计dense/continuous observable。审计推荐本Task明确sparse-only，dense均匀性另建测量与验收契约。
状态：OPEN。

## D-063 — 根产品部署模型

选项：真正单一installable distribution；或明确source-checkout-only产品并删除/降格半安装metadata。当前root bootstrap、root distribution与八个standalone distributions是三套truth。审计长期推荐单一installable/locked产品。
包含D-019；按`DECISIONS-PRIORITY.md`的G1一次裁决。
状态：OPEN。

## D-064 — 实验机Python依赖升级

选项：`update.bat`继续安装latest-compatible依赖；或使用唯一lock/constraints和environment receipt，源码更新与环境升级分开。审计推荐lock；当前根列表没有保留`zlc_atom`的NumPy/SciPy pins，且checkout rollback不能恢复被升级的环境。
状态：OPEN。

## D-065 — Site-drop诊断工具

保留该功能并把per-pixel gate evidence合入Calibration纯API，CLI只负责I/O；或删除一次性工具。审计推荐保留功能、删除复制的science detector实现。
状态：OPEN。

## D-066 — Dataset revision ref唯一性

选项：producer承诺同ref同内容并由runtime generation owner enforce；或把ref降级为排序hint且所有去重exact compare。审计推荐第一项，same-shot继续只由EventRef表达。新增values content digest与当前“不新增产品hash”裁决冲突，除非用户主动重新开放，否则不作为候选。
状态：OPEN。

## D-067 — `unique_path`并发承诺

当前32个并发调用全部取得同一路径。选项：将unique allocation与commit原子化，保证多process不覆盖；或明确降级为single-owner顺序执行下的available-name helper。审计推荐前者，因为Task/Save存在真实并发可能。
状态：OPEN。

## D-068 — Package facade策略

选项：所有跨包调用只许top-level facade；或允许明确的module public API，facade只保留高频核心。审计推荐后者，并删除Markdown/测试手抄完整API allow-list。
与D-078同属G1：本项决定facade形状，D-078只在确有仓外consumer时决定compatibility期限。
状态：OPEN。

## D-069 — Figure metadata admission

选项：只接受strict JSON tree，未知对象拒绝并由section owner显式投影；或继续把未知对象自动`str()`。审计推荐strict；当前字符串化会静默丢类型且archive member namespace还可互相覆盖。
状态：OPEN。

## D-070 — Test/notebook-only data APIs

`numeric.py`、`AxisSourceRef/ResolvedPointRows`、`ValuePayloadContract`等无production consumer。选项：按真实产品图删除；或明确把它们作为外部scientific library兼容surface并补真实usage/compatibility policy。审计推荐删除。
状态：OPEN。

## D-071 — `zlc_ui`与plot host边界

选项：UI接受窄structural host protocol（widget/size/wheel target），不import具体package；或Workbench取出QWidget再交UI。当前代码较接近前者，README却把两者都禁止。审计推荐前者并冻结窄surface。
状态：OPEN。

## D-072 — Saved layout向前兼容

选项：新增有default字段自动补齐；任何变化显式version migration；或明确旧layout不兼容。审计建议同format只补default，结构变化升format并提供直接migration。需确认历史layout是否必须打开。
状态：OPEN。

## D-073 — Notebook convenience API

`ExperimentSession.camera/sequencer/save_figure`与`create_console_window`主要由tests/notebook消费。选项：保留并明确为default-device legacy convenience；或删除，让Notebook走named installation capability与formal Panel Save。审计推荐删除第二产品路径。
状态：OPEN。

## D-074 — Interpreter退出时的硬件策略

选项：有界调用composition close/safe；只销毁UI并明确硬件状态unknown；或每个实验入口注册lab emergency policy。Sequencer与SLM物理close语义不同，不能继续用通用`sip.delete`绕过所有guard。
状态：OPEN。

## D-075 — Figure Viewer普通图片支持

若Viewer只浏览scientific archive，picker应只给NPZ；若也支持PNG/JPG，必须定义没有dataset/flow/device/fit时的降级UI。当前文案承诺但实现拒绝。
状态：OPEN。

## D-076 — Developer UI组件产品范围

Graph、gallery、acceptance capture、FluentFormGrid是否继续作为独立UI toolkit public surface。按“无consumer默认删”，审计建议删除Graph/FormGrid，acceptance移tools，gallery只留最小manual smoke。
状态：OPEN。

## D-077 — Plot renderer文件组织

选项：把唯一拆出的`_rendering/pulse.py`合回单一renderer；或沿`_kinds`边界把所有kind对称拆开并删除private cross-calls。审计默认推荐先合回，除非维护历史证明完整按kind拆分有实质收益；该决定不应阻塞正确性修复。
状态：OPEN。

## D-078 — 仓外public API兼容承诺

若没有真实仓外消费者，按monorepo实际imports收窄facade并删除dead seams；若需要兼容外部notebook/scripts，用户需提供样本或明确名单、版本与deprecation policy。现有“名字存在”tests不能替代消费者证据。
与D-068共同裁决，不再单独要求用户回答。
状态：OPEN。

## D-079 — Calibration report/sample presentation owner

选项：Atom输出typed snapshots、summary与report recipe，通用plot/data writer落盘，Workbench只负责交互Panel Save；或把Calibration整体降为Workbench内置plugin。审计推荐前者，消除Atom直接import composition root。
状态：OPEN。

## D-080 — Calibration artifact复现等级

推荐小而严格的deploy calibration加可选/默认完整run archive（raw samples或manifest/digest、algorithm/config/diagnostics）。需决定raw frames默认是否保存；当前默认False意味着以后不能用新算法重算，全部保留又可能约4.7GiB内存/存储。
状态：OPEN。

## D-081 — 旧Calibration缺value unit

选项：拒绝并通过显式migration让operator选择counts/photoelectrons；或默认counts。审计推荐显式迁移，错单位可能全site一致地给出错误结果而不报错。
状态：OPEN。

## D-082 — Temperature artifact范围

选项：保存完整survival、validity与played coordinate，JSON曲线只作summary；或只保summary并明确逐shot/site数据在session后不可恢复。审计推荐完整science result。
状态：OPEN。

## D-083 — Scan结束/Stop后的device状态

选项：恢复run前值；或leave-at-last并在UI/run record显示final applied values。当前隐式leave且无统一readback。审计默认推荐restore pre-run。
状态：OPEN。

## D-084 — Simulation参数作用域

哪些稳定apparatus physics进入`SimulationWorldConfig`，哪些可在Device Manager编辑，哪些只属于test scenario override。审计建议单一composition-owned config，禁止继续散落public mutable attributes。
状态：OPEN。

## D-085 — Virtual hidden oracle的产品可见性

选项：test-only `SimulationDiagnostics`访问面；或正式typed diagnostics signal供现场simulation panel使用。审计推荐前者，普通device consumer不得看到hidden truth或散落oracle properties。
状态：OPEN。

## D-086 — Runtime correction修改权限

选项：Editor在stopped状态取得DeviceUse claim后可临时load/toggle并写receipt；或correction只能由Device Manager/installation配置改变。无论选择哪项，active Task/Send期间mapping必须冻结且不能绕过claim。
状态：OPEN；并入G6。

## D-087 — SLM optical settle policy

选项：profile保存该head实测最坏settle；用户authoring固定值；或command-dependent/adaptive。任何选项都必须区分transport ack、controller update与optical settled，并先在实验机测rise/fall。审计默认推荐profile最坏值。
状态：OPEN；并入G6。

## D-088 — DVI session close语义

选项：接受销毁presenter即撤销DVI输入并明确device state unknown；或要求独立持久presenter/service在Editor/session关闭后继续drive。当前统一“close保持phase”承诺对DVI不成立。
状态：OPEN；并入G6。

## D-089 — FPGA lane/pin identity owner

选项：显式board manifest生成host target、top mapping与XDC；或top固定lane ABI、XDC只映pin。审计推荐manifest，拒绝继续用XDC文本行序加手写top双truth；bitstream需可读board/pin ABI ID。
状态：OPEN；并入G5。

## D-090 — FPGA geometry profile范围

选项：先冻结唯一35T board profile并删除假参数化；或正式支持多geometry并为每profile建立RTL/build/receipt矩阵。审计推荐先冻结，真实第二板出现再扩。
状态：OPEN；并入G5。

## D-091 — 最短scan point

需基于BRAM read latency决定host拒绝的最小tick；若实验必须1-tick scan，则批准增加prefetch/FIFO。当前1/2 tick没有闭合契约。
状态：OPEN；并入G5。

## D-092 — Firmware identity与receipt

选项：只证明layout compatibility；或硬件还必须读回exact qualified build ID并关联source/constraint/board/geometry/Vivado/IP/STA receipt。审计推荐后者，但它需要新增ABI。
状态：OPEN；并入G5/G7。

## D-093 — Build/program默认动作

无参数`build_and_program.bat`是否仍直接program。审计推荐默认build-only，volatile program和永久flash均显式选择；flash另加不可误触确认。
状态：OPEN；并入G5。

## D-094 — 多板选择

program/flash按serial、target URL、IDCODE、board ID中的哪组字段唯一选择。裁决前必须exactly-one fail closed，不能默认第一个target/device。
状态：OPEN；并入G5。

## D-095 — `GND1..15`物理用途

需硬件负责人确认它们是reserved-low/guard IO，还是实验接线把FPGA IO当ground。普通被驱低IO不能在文档中称物理地。
状态：OPEN；并入G5。

## D-096 — DAC safe value

选项：产品只允许RTL真实支持的统一midpoint并拒绝其他值；或把per-bus safe value加入wire ABI/fingerprint/loader/RTL。审计推荐短期统一midpoint。
状态：OPEN；并入G5。

## D-097 — tracked `build/geom.tcl`

选项：删除并每次从canonical config原子生成；或作为direct Tcl强制输入并有exact parity test。审计推荐生成，不在源码树维持第二projection。
状态：OPEN；并入G5。

## D-098 — Vivado/IP版本

批准并固定哪一个tool/IP版本。当前自动选择“找到的新版本”与fail-open property不能支撑qualified build。
状态：OPEN；并入G5。

## D-099 — UART信任与error policy

即使只允许trusted host，截断帧watchdog、count/address bounds仍必须有。需决定protocol error是hard reset/reject还是error reply，以及何时释放UART对JTAG的优先权。
状态：OPEN；并入G5。

## D-100 — 正式notebooks/examples集合

只保留操作者真实使用并进入fresh-kernel lane的教程；其余删除，不用shape tests保活。建议保留Workbench/Atom/Pulse与Data底层教程；Plot Notebook由G1决定，Runtime旧框架教程随删除重写。
状态：OPEN；并入G1。

## D-101 — Software CI落点

推荐至少一个fresh Windows runner，覆盖wheel/install、PyQt5、batch launchers与software lanes。若不能上传外部CI，需指定本地release command和不可编辑result artifact owner。
状态：OPEN；并入G1/G7。

## D-102 — 实验机验收责任与频率

需指定谁可操作FPGA/camera/SLM、哪些domain改动触发重验、evidence保存位置。审计建议按domain gate，不是每次全实验：pulse/wire→FPGA+camera cadence；adapter→对应设备；SLM→device+optics+statistics；UI→real-screen。
状态：OPEN；横跨G5–G7。

## D-103 — Golden/oracle更新政策

Golden变化必须附独立证据说明产品结果为何应变；禁止被测implementation自动重生expected后直接提交。Synthetic fixture记录seed/model/version，实验fixture记录脱敏raw statistics与采集provenance。
状态：OPEN；并入G7。

## D-104 — Pulse remote网络信任边界

选项：只允许loopback；运行在隔离control LAN并加认证/owner token；或继续信任任意LAN client。审计排除第三项：当前无认证的`0.0.0.0` server允许任意client接管/load/fire，且stale handler竞态仍可command。推荐隔离LAN+认证，最少也应默认loopback。
状态：OPEN；并入G5。

## D-105 — UART设备发现策略

选项：apparatus显式COM；serial/VID/PID allowlist；或扫描并向所有COM发probe。审计排除第三项，因为同机其他实验仪器可能收到非本协议bytes。推荐显式配置，allowlist仅作辅助发现。
状态：OPEN；并入G5。

## D-106 — Remote第二client取得owner

选项：authenticated explicit takeover且旧owner成功SAFE后才转移、所有旧handler失效；或busy拒绝直到旧client断开。当前任何新连接静默抢板且SAFE失败也继续不是可选项。审计推荐前者。
状态：OPEN；并入G5。

## D-107 — Quiet editor与active forever失联

无active command的quiet editor不应因编辑沉默被踢；active forever需要connection/lease失联timeout后SAFE。用户需给timeout。当前docs写5分钟、tests/code又删除idle timeout，必须只留一份truth。
状态：OPEN；并入G5。

## D-108 — Local transport能否绕过remote server

若不允许，正式产品只有server process持UART/JTAG，删除假的Interprocess lease；若允许，必须实现真实OS/device lock并跨进程验收。审计倾向single server owner，避免两条hardware入口。
状态：OPEN；并入G5。

## D-109 — Hardware Notebook定位

推荐offline tutorial与hardware bring-up分离；bring-up默认有限pulse、显式危险确认、`try/finally SAFE`和独立验收记录。当前最后cell forever fire且无SAFE不可继续作为普通教程。
状态：OPEN；并入G1/G5。
