# 给实验负责人的架构裁决说明

状态：用户已完成裁决；最终记录见[USER-DECISIONS-2026-08-17.md](USER-DECISIONS-2026-08-17.md)。本文保留为前因后果和术语说明。

这份文档不假设你熟悉当前代码。它先解释系统现在是什么、哪里出了问题，再说明不同选择会带来什么结果。技术缩写只在必要时出现，并在第一次出现时解释。

如果你信任审计结论，不需要阅读其他30份报告。其他报告只是这里每个判断背后的代码证据。

## 一、先区分三种事情

审计发现的问题分成两类。

### 第一类：明确错误，不需要你判断“要不要修”

例如保存文件可能自相覆盖、未拍摄的数据被当成真实数据、设备明明没有成功进入安全状态却被软件说成成功。这些不存在合理的产品偏好，应该直接修。

你只需要知道它们的影响，不需要决定具体代码怎么写。

### 第二类：产品行为有多种合理选择

例如：

- 新相机图像到了，是立即显示，还是必须等拟合完成才一起显示？
- Task运行时，只锁定会影响实验的操作，还是把整个窗口都锁住？
- Calibration是否默认保存全部原始帧？
- SLM Feedback按Stop时，是接受当前最好结果，还是恢复开始前状态？

这些才需要你裁决。

### 第三类：必须由实验机事实或测量回答

有些问题不是“喜欢方案A还是B”，代码审查也不能替实验机回答。例如：

- 你是否真的用两台电脑通过LAN控制FPGA；网络是否隔离；
- 实验机有几块FPGA板、正式UART是哪个COM；
- camera/FPGA是否已经有hardware shot marker；
- `LSH0804382`的phase curve究竟来自厂家文件还是本机测量；
- X15213真实optical settle需要多少毫秒；
- USB/DVI在实验机上是否完成灰度、方向和光学验收；
- 一次Calibration可接受多少磁盘空间；
- SLM final validation最多允许多少shots或多长时间。

这些需要你提供现状，或者后续安排测量。它们不能通过“接受推荐”变成已证明事实。

## 二、几个反复出现的词是什么意思

### Dataset / Snapshot

Dataset是一次测量的数据，包括数值、坐标轴、单位、哪些数值有效等信息。

Snapshot是Dataset在某个时刻的不可变快照。例如Camera Measurement计划拍100次：拍完第20次时，可以形成一个“已经写入20次、后面80次还没拍”的快照。

### Live publication / Preview

Live publication指Measurement或Task还在运行时，把中间数据发布给Monitor。

Preview是Monitor自动打开的图。例如SLM Feedback运行中显示当前相机平均图、当前相位图和均匀性曲线。

### Revision、Generation、Coverage、Terminal

- Revision：同一次run中数据每更新一次就增加的版本号，例如第20次live更新。
- Generation：一次新的run或Restart的身份；新run的revision可以重新从头开始。
- Coverage：计划写多少数据、目前真正写了多少，例如20/100。
- Terminal：这次运行已经结束，包括成功、失败或被取消。

这些词的目的只是防止“旧数据冒充新数据”以及说明partial result，不是要求你理解内部类。

### Running reduction、Retained partial、Seal、Worker

- Running reduction：随新shots不断更新的缩减结果，例如running average、site rate，而不是全部raw frames。
- Retained partial：Task被Stop后，已经采到的部分数据仍留在系统里，可以显示和保存。
- Seal：宣布“这份partial/final结果不再继续写”，形成一个稳定可读版本。
- Worker：框架管理的后台执行线程；它可以做复制、拟合或设备调用，但必须有明确关闭owner，不能卡住Qt界面。

### Owner、Device Claim、Stopped Draft

- Owner：某份状态或设备命令唯一由谁负责。
- Device claim：运行Task时对设备的独占使用权，防止另一个窗口同时改它。
- Stopped draft：已经在UI里配置好、但还没有Start的Measurement/Task草稿，不会控制硬件。

### Objective、Profile、Heartbeat

- Objective：SLM target的求解意图。`spots`表示只关心离散点，`image`表示关心一整块连续图像。
- Profile：某台具体设备的校准资料，例如serial、phase response和settle time，不是普通UI preset。
- Heartbeat：active remote command期间client定期发出的“连接仍活着”信号；丢失一定时间后server可SAFE。

### Fit、Overlay、Selector

- Fit：对数据拟合模型，例如Gaussian拟合。
- Overlay：盖在原图上的附加标记，例如Occupancy在原子位置画绿色/红色圆圈。
- Selector：用户在图上框选、拖动范围或选择某个区域的工具。

### Artifact / Archive

Artifact是一次实验或Task保存到硬盘、以后还能读取的结果文件。

Figure archive是“保存图”的科学文件。它不是普通截图，而是一个类似ZIP的`.npz`文件，里面有很多带名字的内容：

- 原始数值；
- validity mask，即每个数是否有效；
- 坐标和单位；
- 当前绘图设置；
- fit使用的模型、参数设置及可重算信息；当前格式并不保证保存每次exact numeric fit result；
- overlay和selector状态；
- run/device信息。

Writer是写这个文件的代码，Reader是以后把文件读回来的代码。Format是文件内部约定，例如版本号、每一项叫什么名字、shape应该是什么。

### Python package、distribution、wheel、source checkout

- Python package：代码里的一个模块，例如`zlc_plot`。
- Distribution：可以通过Python安装工具一次安装的产品包。
- Wheel：Distribution常用的安装文件，类似一个Python安装包。
- Source checkout：直接从Git仓库目录运行源码，不先安装产品。
- Lock：精确记录NumPy、SciPy、Matplotlib等依赖的版本，保证两台机器安装出相同环境。
- Receipt：一次安装或设备命令的可追溯记录，例如使用了哪个版本、哪个设备profile、发送了什么、设备是否确认成功。

当前项目有八个代码层，不等于必须做八个独立安装包。代码可以继续分层，但产品只安装一次。

### 其它硬件/交付缩写

- CI（持续集成）：在干净机器上自动安装并运行software tests，避免只在开发机环境里通过。
- ABI：软件和硬件对数据布局、寄存器和字段宽度的精确约定；ABI不一致时可能把正确数据写到错误位置。
- FIFO：FPGA内部暂存延迟事件的有限队列；满了以后不能静默丢事件。
- UART/JTAG：电脑连接/控制FPGA的两种通道。UART通常是串口，JTAG常通过Vivado访问。
- Manifest：一个run或文件集合的目录清单，说明有哪些文件、shape和完成状态。
- Provenance：结果来源记录，例如由哪个Git版本、配置、设备和测量生成。
- Mock：开发机上的假设备接口，只能证明本项目两端的调用/bytes自洽，不能替代真SDK或真硬件验收。

## 三、这些是明确错误，不需要你裁决

### 1. Figure archive为什么会“成员覆盖”

一个Figure archive内部像一个有很多抽屉的箱子，每个抽屉有名字。例如：

```text
data
data.validity
panel.state
fit.parameters
```

当前Writer允许另一项用户数据也叫`data.validity`。这样两个内容会使用同一个内部名字，后写入的内容覆盖先写入的真正validity mask。

文件在写入时可能没有报错，但Reader以后会发现shape不对，连本项目自己的Reader都打不开。

Reader“不验自己的format”是指：文件声称自己是某个版本，但Reader没有严格检查版本、必需字段、shape、重复名字和非法数字。于是损坏或旧格式文件可能先被接受，直到更深处以不容易理解的错误失败。

正确修法是：Writer在写之前先规划所有内部名字，任何碰撞立即拒绝；Reader在读取第一步严格检查版本和结构。这里不需要你选方案。

### 2. 未拍摄的数据被当成真实数据

假设Camera Measurement要拍100次，现在只拍了20次。

当前实现提前创建100个位置，后面80个位置填零，却把这些零标成“有效”。Occupancy Processor会把它们分类成“没有原子”，然后说整个100次数据已经完成。

这会让图、rate和下游Processor看到尚未发生的实验事实。

更严重的是：如果Stop发生时UI恰好刷新过一次，保存结果可能保留100个位置；如果UI没刷新，可能只保留20个位置。科学结果不应该取决于Monitor刷新时机。

正确修法是未拍摄位置明确标为invalid，而且Task terminal结果只由采集进度决定，不由UI决定。

### 3. Device和窗口关闭顺序错误

明确例子包括：

- 两个Device使用同一个key时，两台设备都会被创建，但系统只记住第二台，第一台永远不关闭。
- Standalone Pulse Editor先把窗口关掉，随后才尝试把FPGA设为safe。如果safe失败，操作界面已经消失。
- Python退出时有一条通用路径直接删除所有Qt窗口，绕过每种设备自己的关闭规则。
- Device Manager启动的部分线程池没有明确owner，也没有保证程序关闭时退出。

正确不变量应该是：窗口只有在它拥有的命令、线程、device claim都已经安全退出后才能消失。这个不需要你选择。

### 4. FPGA的SAFE、DONE和build流程没有达到名字承诺

- SAFE：软件发送SAFE后，某些DAC clock仍可能继续输出；LOAD新程序时也可能在FIRE前提前打开clock。
- DONE：当前只表示主时间表走完，延迟输出队列可能还没排空。此时立即SAFE或LOAD会截断波形尾部。
- Timing：50 MHz主逻辑没有被正确加入时序约束，Vivado报告可能只验证了JTAG时钟。
- Build目录：所谓safe delete只检查路径长度；环境变量误设时可能递归删除不属于build的目录。
- Program/flash：脚本可能默认选择发现的第一块板。多板或其它Vivado target存在时有写错板风险；flash还是永久操作。

这些名字属于物理安全承诺，不应该作为产品偏好。SAFE必须真的safe，public DONE必须代表物理输出完成，build/program必须fail closed。

### 5. Pulse remote当前没有可靠的控制所有权

当前server默认可以监听整个LAN，没有认证。新的TCP client连接时可以取得控制权；旧client的handler在竞态中仍可能继续下命令。更严重的是，接管前尝试SAFE，即使SAFE失败，代码仍可能把新client设为owner。

一个名叫`InterprocessDeviceLease`的对象实际只是当前Python进程里的字典，另一个进程完全看不到它，而且真实transport没有使用它。

这些是明确错误：每条命令必须证明client仍是owner，SAFE失败不能完成接管，多进程所有权不能用进程内字典冒充。

### 6. SLM real初始phase不能假设为零

程序刚连接真实X15213时，没有读到当前屏幕phase，也没有发送零phase。但代码把`last_commanded_phase`记成零。

如果Feedback失败并“恢复开始前phase”，它可能发送零，从而覆盖实验机原本显示的未知phase。

正确做法是状态明确标为unknown。只有本程序成功发送并确认过，或者用户明确加载可信artifact接管后，才能说当前phase是known。

## 四、Gate 1：这个项目以后怎样安装和更新

### 当前为什么有三套答案

当前代码同时表达了三种部署方式：

1. 正式batch launcher把Git checkout放到`PYTHONPATH`，直接从源码运行；
2. 根`pyproject.toml`又把项目描述成一个可以安装的distribution；
3. 八个代码层各自还有独立`pyproject.toml`、版本和独立wheel的历史承诺。

因此“当前运行的是哪一份代码、依赖按谁的版本安装、一个package能不能独立运行”没有一个答案。

具体例子：`zlc_atom`要求固定NumPy/SciPy版本，根安装脚本却只写宽泛的`numpy`和`scipy`。另一项测试甚至读取了电脑上旧的standalone `zlc_data`安装，而不是当前仓库，仍然通过。

### 方案A：一个可安装产品，内部仍保留八层（推荐）

含义：

- 八层仍然保留，用来约束依赖方向和职责；
- 但用户只安装一次，例如安装一个ZLC wheel或installer；
- 一份lock文件精确规定dependency版本；
- 每次安装生成receipt，记录Git版本和环境；
- 不再承诺八层分别作为公开library发布。

优点：

- 实验机和开发机更容易使用同一代码/环境；
- fresh install和CI能真正复现；
- 不需要维护九套版本和安装说明；
- 可以逐步删掉旧standalone兼容面。

代价：

- 需要整理现有bootstrap、package data和入口；
- 第一次迁移工作量较大；
- 如果实验室外真有人单独依赖某个`zlc_*`包，需要为他们安排迁移。

### 方案B：明确只从Git源码运行

含义：不再宣称项目可安装，不做root wheel；所有正式launcher只从指定checkout启动。删除或降格误导性的package metadata。

优点：短期改动少，符合目前实验机习惯。

代价：依赖环境和路径仍比较脆弱；reclone、新电脑、CI与回滚更难；容易再次加载旧editable package。

### 方案C：正式维护八个独立wheel

这要求每层有独立版本、兼容政策、build/install测试和发布流程。当前没有明确外部消费者，维护成本最大，我不推荐。

### 我的推荐思路

这是一个实验控制产品，不是八个准备分别发布到PyPI的通用library。代码分层很有价值，但发布分成八份没有给实验带来相应收益。因此推荐方案A。

### 你需要回答

1. 实验室外是否有真实脚本只安装并使用某个`zlc_*`包？
2. 是否接受长期做成“一次安装整个ZLC产品”？

如果没有外部用户，直接回答“Gate 1接受方案A”即可。

## 五、Gate 2：Measurement运行时怎样发布数据

### 当前在解决什么问题

Camera、Scan、Calibration各自写了一套live发布逻辑。Runtime里又有另一套从未被真实节点使用的通用框架。

结果是：

- 新增Logic Node时很容易忘记live或preview；
- Camera可能在第1次复制1帧、第2次复制2帧、一直重复复制全部历史，数据越多越慢；
- 是否有人订阅会改变数据在哪个线程构造；
- UI刷新可能影响Stop结果；
- 每层维护不同revision。

Camera和Scan不应该强行用完全相同的live形态。

### Camera有两个合理方案

#### Camera方案A：每次发布不断增长的完整raw stack

例：作者要求100 shots。从第一帧起，Dataset最终容量就是100；拍到第20帧时：

- 前20个位置有效；
- 后80个位置明确invalid；
- coverage写20/100；
- Stop后仍能保存这个partial stack。

优点：运行中能直接查看或保存全部已拍raw frames。

代价：2048×2048相机图很大。如果第1次复制1帧、第2次复制2帧、一直到第100次复制100帧，会重复搬运大量历史数据，越到后面越慢。

#### Camera方案B：live发布最新cycle和累计reduction，terminal保存完整raw stack（推荐）

运行中分别发布：

- 最新一组完整camera frames，用来肉眼monitor；
- running average、Occupancy rate等增量统计；
- 已完成shots数量。

完整raw stack由采集owner顺序积累，Stop/结束时一次形成retained Dataset；在线Processor每次只消费新增frame，不重新分析全部历史。

优点：仍然是live，而且不会每次复制全部raw历史。对于你常用的average/reduce preview更直接。

代价：运行中普通panel看到的是latest/running reduction，而不是一个可随时随机访问全部raw frames的巨大stack。若中途必须保存完整partial raw，需要采集owner提供partial seal。

### Scan更适合固定geometry逐步填充

Scan的每个point有明确坐标，二维/多维grid尚未扫描的位置可以invalid；已扫描位置逐步变valid。这样运行中能看出scan走到哪里，也能保留partial grid。它不像Camera raw stack那样每个point都是巨大的图像。

### Processor也要区分两种需求

- 科学上必须逐个处理、不能漏event的Processor使用incremental exact输入；
- 只为显示、旧结果可以被新结果替代的Processor可以latest-only。

这应该由Processor声明，不能根据coverage类型猜。

### “Push”是什么意思

Node完成一个版本时主动把结果交给Runtime，Runtime记录一个新的revision。UI只读取已经提交的结果，不能在刷新时进入Node内部临时构造数据。

大camera的复制可由统一的Runtime后台worker完成，但不允许Camera、Scan、Calibration各写一套互不兼容的live lifecycle。

### 我的推荐思路

- Camera选方案B：latest cycle + running reductions live，raw由采集owner增量保存并在partial/terminal seal；
- Scan使用固定geometry逐步填充；
- 所有未写位置invalid，Stop结果不受UI刷新影响。

### 你需要回答

运行中的Camera Measurement，你是否真的需要panel随时拿到“全部已拍raw frame stack”？

- A. 需要，接受更高内存/复制成本；
- B. 不需要，latest cycle和running average/rate已经足够（我的推荐）。

## 六、Gate 3：新数据、Fit和Overlay什么时候显示

### 当前为什么卡

当前live fit采用“data和fit必须同生”的规则。新相机图像N到达后，系统先等fit N算完，再一起显示。

如果fit要300ms：

- 300ms内连原始新图都看不到；
- 为了同一shot一起换屏而被分在同一组的其它快速panel，也可能一起等待这个慢fit；
- 新er frame只能排队或丢弃；
- 改一个无关title也可能重新启动fit；
- 应用一次panel设置会重复渲染并传递2到6张完整屏幕图像。

### 方案A：Data-first（推荐）

1. data N到达，立即显示；
2. 旧fit N-1立即隐藏，绝不叠到新data上；
3. 后台只拟合最新N；
4. fit N完成且N仍是current，第二次只更新fit overlay；
5. 如果N+1已经到了，丢弃旧fit N结果。

用户会短暂看到“新数据，fit尚未完成”，但不会看到错误的旧fit。

这更适合Monitor：首先让你看到实验发生了什么，再补分析。

Save Fig或frozen report仍可以等待fit完成，保证保存结果完整。

### 方案B：Atomic data+fit

新数据必须等对应fit完成才显示。

优点：屏幕每一帧都同时有对应fit。

代价：fit延迟就是data延迟；慢panel会阻塞整个same-shot展示。若你选择它，就必须接受这个产品行为和性能上限。

### Live fit是否对每个数据版本都自动运行

这与data-first/atomic是另一件事。有三种策略：

1. 每个数据revision都启动fit。更新最密，但高帧率时会浪费大量计算，很多fit刚开始就过期。
2. 按panel显示刷新节奏只拟合最新数据（推荐）。中间版本可以不拟合，屏幕每次真正准备更新时只处理最新版本。
3. 只在用户点击Fit/Refresh时运行。最省资源，但不再是自动live fit。

我推荐第2项：保留自动fit体验，同时避免为不会显示的中间帧计算。

### Overlay为什么要由Occupancy自己发布

现在Workbench知道Occupancy的内部类，拿当前Calibration和statuses临时画圈。这造成：

- 保存后没有live node就无法重建圈；
- ROI/binning改变后classifier用新坐标，圈仍可能用旧坐标；
- Workbench承载了plugin science；
- overlay自己变化但主image没变时，scheduler可能不更新。

推荐Occupancy发布一个正式overlay数据：每个site的sensor coordinate、状态、site ID。Plot只负责画，Workbench只负责连接。

### Selector Off时滚轮去哪里

当前Off只阻止选择，滚轮仍被plot用来zoom/pan。根设计又说Off时滚轮应该滚动TaskConsole board。

有三个合理选择：

- A. Off时普通滚轮滚外层页面；On时plot接管滚轮。
- B. 无论Selector是否On，普通滚轮滚页面，只有`Ctrl + 滚轮`缩放plot。这是常见的混合规则，我更推荐，误缩放最少。
- C. 保持当前行为：Selector Off仍由plot缩放/平移。

这里不应继续让“Off”的含义靠猜。

### 你需要回答

1. 新数据是否应立即显示，即使fit晚一点出现？我推荐“是”。
2. Live fit选择“每revision”“按显示节奏只fit最新”还是“手动”？我推荐按显示节奏只fit最新。
3. 滚轮选择A、B还是C？我推荐B：普通滚轮滚页面，`Ctrl + 滚轮`缩放图。

## 七、Gate 4：Task运行时锁什么、显示什么

### Measurement和Task的区别

- Measurement主要采集数据，通常应该持续live发布。
- Task会控制多个设备、执行较长流程，例如Calibration或SLM Feedback，应该显示progress和中间preview。

### 当前问题

- 每个Logic Node都有一份“会产生哪些数据、应显示哪些preview”的声明，但目前允许完全不写preview，运行框架也不检查；新增Node很容易运行成功但屏幕什么都没有。
- Logic row只因为“正在运行”就把所有声明output标成live，即使一份数据都没发。
- worker终态仍可能显示旧progress，例如Task已经结束还写着Scanning。
- Task运行时，真实Qt UI禁用了Add Panel、所有card/settings和所有logic rows；另一部分Presenter文档又说只锁bench，不锁window。

### 推荐行为

Task active时禁止：

- 改当前Task参数；
- 启动冲突设备操作；
- 修改当前Task占用的hardware配置。

Task active时仍允许：

- 查看Monitor；
- 滚动、缩放、改fit或plot layout；
- 新建不会启动hardware的stopped draft；
- 保存当前已经发布的数据。

新Node的框架要求：

- Measurement在限定时间内必须至少发布一次live；
- Task必须发布progress和至少一个preview，除非明确声明“本Task无可视preview”；
- 首个真实数据到达前不能显示live；
- terminal后progress结束；
- preview创建失败可以重试并显示错误。

### Calibration preview显示哪几张

Calibration一个cycle是long/readout/long三帧。显示三帧facet更便于确认背景、原子读出和第二张long是否正常；只显示最后一张会丢关键诊断。我推荐显示三帧。

### SLM Feedback preview cadence

- phase一发送成功就立即显示；
- camera average在100 shots中按时间节流更新，例如每0.5–1秒，而不是等全部拍完；
- uniformity curve每完成一个candidate增加一点。

### 你需要回答

1. Task运行时是否允许继续操作纯Monitor/layout？我推荐允许。
2. Calibration preview是否显示完整三帧？我推荐是。

## 八、Gate 5：Pulse、Camera、FPGA和远程控制

这个Gate包含物理时序，先解释几个词。

### Repeat、shot、cycle、sweep现在为什么混乱

- RepeatRegion：Pulse文档内部把一段timeline重复几次。
- Shot/cycle：完整执行一次实验流程，例如cooling→probe→三张camera frame。
- Sweep：Scan table从第一行播放到最后一行一次。
- Dataset repeat：统计数据中重复实验的轴。

当前这些概念在不同Node里互相代替，可能把三帧pulse重复成六帧，或让camera按错误数量分组。

推荐它们严格分开：Pulse document只定义一个cycle内部结构；Measurement明确请求N cycles；Scan明确请求M sweeps；Dataset保留shot/sweep身份。

### Camera “same-shot”能证明到什么程度

现在DCAM/Pylon的ordinal主要表示“复制/取回的第几张frame”。如果硬件trigger丢了一次，但camera计数继续连续，软件可能看不出来。

有三档保证：

#### 第1档：连续流best-effort

Camera持续armed，Pulse连续发很多cycles，软件按收到顺序每三张分一组。

优点：吞吐最高、额外arm开销最小。

缺点：能发现收到的frame有gap，但不一定知道丢的是哪个physical trigger；一旦错一张，后续分组可能整体错位。

适合普通Monitor或对绝对shot身份不敏感的预览。

#### 第2档：每个cycle或小chunk单独arm/fire并核数量

每次只允许一个明确数量的camera windows，拍完核对数量后再进入下一组。

优点：错误被限制在当前cycle/chunk，不能无限向后错位；无需立刻改FPGA ABI。

缺点：arm/handshake更频繁，实验wall time可能增加。chunk越大越快，边界越弱。

这是我对正式finite science Task的近期推荐。

#### 第3档：hardware marker

FPGA给每个cycle明确marker/trigger counter，camera frame也带可关联的计数或时间戳。artifact能证明“这张图来自cycle 37”。

优点：保证最强，可事后审计。

缺点：需要FPGA、camera adapter和artifact一起改，并在真机验收。

只有实验确实要求绝对shot provenance时才值得近期实施。

### Temperature当前为什么不成立

当前默认camera integration约20ms，而两次trigger只相隔约5.02ms。真实camera第一帧尚未结束，第二个trigger已经到来；可能丢帧或第一帧同时包含两段probe。

合理方案：

- A. 保留20ms exposure，把两次trigger间隔拉长到大于integration和recapture时间；优点是信号强、与现有Calibration更接近，缺点是shot慢。
- B. 把exposure降到约5ms，并为这个工作点重新确认/运行Calibration；优点是快，缺点是信号与threshold改变。
- C. Temperature使用独立Calibration和专用双帧protocol；最严谨，流程更多。

我倾向A作为最快的正确修复；如果你在意Temperature速度或5ms信号已经足够，再选B/C。

### Remote server怎样控制FPGA

你需要先告诉我是否真的使用“两台电脑”：一台实验控制机通过LAN连接一台FPGA机。

推荐规则：

- 默认只监听localhost；跨机时使用隔离实验LAN加认证，或VPN/SSH tunnel；
- 每条命令带当前owner token，也就是server在本次控制会话发出的不可冒用session ID；
- 旧connection的所有后续请求立即失效；
- 正式UART port在apparatus config中明确填写，不扫描所有COM；
- 默认只有server进程能直接打开UART/JTAG，避免第二个local程序绕过owner。

第二个client有两种合理政策：

- Busy reject（推荐默认）：旧owner未正常释放前，新client只能看到“设备忙”，不能接管。最简单、安全。
- Explicit takeover：经过认证的用户主动点击/请求Take over；server先成功SAFE旧command，再转移owner。适合旧UI崩溃后的远程恢复，但实现和审计更复杂。

可以默认busy reject，再提供一个明确的管理员takeover动作，而不是“任何新连接自动抢板”。

### Active forever失联怎么办

有两种不同状态：

- Editor只是打开但没有active output，长时间不操作不应被踢。
- FPGA正在forever pulse，network/client失联后不能无限输出。

需要一个active-command lease timeout，例如5–30秒，client必须heartbeat；超时硬件SAFE。具体时间取决于网络稳定性与实验流程。

### 你需要回答

1. Temperature选择A、B还是C？
2. 正式science Task目前采用第2档是否足够，还是近期必须做第3档hardware marker？我推荐先第2档。
3. Pulse remote是否跨两台电脑使用？网络是否物理隔离？
4. 第二client默认busy reject是否可以？是否需要管理员explicit takeover？
5. Active forever失联后多少秒SAFE？
6. 是否只有一块正式FPGA板、一个固定UART port？

SAFE必须物理安全、DONE必须等tail、不能静默丢FIFO事件、program/flash必须唯一选板，这些不需要你裁决。

## 九、Gate 6：SLM phase、Feedback和100 shots

### SLM内部几层phase是什么

1. Target：你希望焦平面出现的强度图，例如Grid、Text或Flat Top。
2. Pattern/base phase：solver为这个Target计算出的核心全息phase。
3. Input pupil：打在SLM上的Gaussian入射光斑模型。
4. Zernike/wavefront：你主动设置的倾斜、像差校正等operator层。
5. Science phase：最终发送前的`Pattern + wavefront`。
6. Vendor correction/LUT：adapter把science phase转换成这台X15213需要的灰度码。

### 当前Feedback为什么上下文不完整

Editor知道Target objective、Gaussian pupil和Zernike，但Feedback只拿Target array和device last phase：

- objective丢失，某些image load后可能变成spots算法；
- Feedback使用另一个默认hard circular pupil；
- device last phase已经含Zernike，却被当作Pattern warm start；
- 新candidate可能没有保留原来的wavefront层。

推荐建立显式Science Context artifact：Target（intensity+objective）、实际传给solver的二维入射振幅数组（numeric pupil）、当前Pattern/base，以及操作者设置的Zernike/steering校正层（operator wavefront）。Feedback只更新Pattern，再加回冻结的wavefront。

它不依赖某个Editor窗口是否正好开着。

### 当前Feedback到底均匀什么

三个量不同：

1. All-shot average fluorescence：所有shot的平均荧光，等于loading概率和occupied brightness共同作用。
2. Occupied single-atom fluorescence：先判断有原子，只平均occupied shots。
3. Trap depth：需要light shift、trap frequency或其它独立谱学测量。

当前Task实际最接近第1项。它不能直接宣称“trap depth达到1%”。你以前也已经明确：当前先把荧光均匀，未来再用trap-depth frequency等测量优化真实trap depth。因此推荐把当前Task诚实限定为第1项。

### 为什么100 shots不能证明1%

以下数字来自当前SimulationWorld和已有workspace artifacts，不是真实实验机噪声分布的替代品：100 shots后每个site mean的不确定度约4.4–5.1%，较早run可10–14%。在这个模型下，即使真实35个site完全相同，取最大/最小也会因为极值噪声得到大约1.2甚至更高。

它足以证明“当前软件不能预先把100 shots写成1%保证”，但最终需要在实验机上记录per-site variance/SEM，再决定实际需要多少shots。

因此100 shots适合快速判断“5倍变成1.3倍”之类的coarse进步，不可能可靠证明1.01。

推荐：

- 每轮100 shots用于coarse update；
- controller按不确定度降低噪声site权重，限制每步变化；
- 变差时rollback或减小gain；
- 到达noise floor停止；
- 最终对锁定的best candidate单独增加shots，用confidence interval判断是否达到目标；
- 达到用户给定的最大shots或最大时间仍无法证明时，返回“当前estimate + uncertainty，证据不足（inconclusive）”，而不是把最好结果丢掉、误判失败或无限运行。

建议的统计语义是95% simultaneous confidence：考虑35个site一起比较后，仍有至少95%把握认为真实max/min不超过目标。具体目标可以是1.01，也可以先用实验可达到的值。

### Stop应该是什么

当前一个Stop同时想表达两件相反的事：

- “我满意当前结果，提前接受最好phase”；
- “取消本次Task，恢复开始前phase”。

推荐拆成两个按钮：

- Accept best：保留有证据的最好candidate，写正式artifact；
- Cancel and restore：恢复known incoming phase。

如果incoming是unknown，不能提供虚假的restore。

### USB和DVI

开发机软件/mock证据表明USB路径可以生成原生1272×1024灰度并走readback API，所以它**更适合优先做实验机验收**；这不表示USB已经在真X15213上完成ABI、display slot、orientation和optical phase验收。

DVI还受Windows/GPU scaling、color management、dithering、窗口生命周期影响。关闭DVI presenter可能就撤销输入，不能统一承诺“关闭Editor仍保持phase”。在真机完成灰度和optical验收前，我建议把DVI标为Experimental。

### 还有哪些必须由实验机提供的事实

1. `LSH0804382`的256点phase curve来源：厂家文件、本机cross-polarizer测量，还是旧软件导出？
2. Vendor correction由谁、在什么时候允许load/toggle：只在Device Manager配置，还是Editor stopped状态也允许？
3. 该head在852nm下最坏optical settle是多少；当前50ms不是已验收常数。
4. DVI关闭窗口后真实SLM输出如何；是否必须在Editor关闭后继续保持DVI drive？
5. Gaussian/Flat Top等连续图像target近期是否真要用于实验，还是目前主要使用离散sites？这决定是否立即投入重做连续图像求解算法；它与已经较快的稀疏site算法是两件事。

### 你需要回答

1. 是否接受当前Task只优化all-shot average fluorescence，而不宣称trap depth？我推荐接受。
2. 是否接受100 shots只作coarse，final validation自动增加shots？我推荐接受。
3. Final validation最多允许多少shots或多少分钟？是否接受95% confidence以及“到上限仍inconclusive”的结果？
4. 是否拆成Accept best和Cancel/restore两个动作？我推荐拆。
5. Feedback是否只修改Pattern并保留pupil/Zernike？我推荐是。
6. 是否先验收USB、DVI暂标Experimental？我推荐是。
7. 请提供/确认phase profile来源、期望的correction修改权限、settle事实、DVI close需求和dense target优先级。

## 十、Gate 7：实验结果保存多少、Task结束后设备停在哪里

### Calibration为什么有“小文件”和“完整证据”两种需求

实验运行时只需要一个小的deploy calibration：site位置、threshold、模型参数、单位等。

但以后想用新算法重新分析，就需要原始frames或至少能找到原始frames的manifest。当前默认不保存raw，无法重算；另一条report路径又可能把200×3×2048²的完整帧一直留在内存，约4.7GiB。

正确设计应分开：

- Deploy artifact：小、严格version、快速加载；
- Run archive：raw frames以stream/chunk方式写盘，不在内存堆4.7GiB；带algorithm/config/provenance；
- Summary report：图片和人类可读统计，可重生，不作为唯一science truth。

### Raw默认策略

选项：

- A. 默认保存完整raw run archive。最可复现，但占磁盘。
- B. raw由camera acquisition系统存到外部chunk目录，Calibration artifact保存run ID、受控相对路径、文件清单、shape/size和写入完成状态。推荐，如果已有可靠数据目录和备份。当前设计明确不为了这个功能新增一套content hash。
- C. operator opt-in；默认只存deploy artifact。省空间，但很多run以后不能重算。

我推荐B；若当前没有可靠external raw store，则先A。C不适合仍在快速迭代算法的阶段。

### Temperature保存什么

应保存每个repeat/site的survival、validity和实际played coordinate。最终curve JSON只是summary。否则Task结束后无法检查异常site或重新聚合。

### Scan结束后device状态

当前通常停在最后一个scan point，但UI/run record没有统一说明。

选项：

- Restore：结束或Stop后恢复run前设备值。更安全、可预测，我推荐默认。
- Leave at last：适合用户故意找到最佳点后希望设备留在那里，但必须明确显示final applied value，并由用户选择。

Generic Scan只有“最后播放的point”，并不知道哪个point科学上最好。因此这里只提供显式“Leave at last”选项。某个专门的optimization Task若定义了best，应该由那个Task另行提供“Apply best”，不能混在通用Scan语义里。

### 你需要回答

1. Calibration raw默认选A、B还是C？
2. Scan默认Restore，还是Leave at last？我推荐Restore，并把Leave做显式选项。

## 十一、Gate 8：Simulation哪些参数给用户看

### 当前问题

SimulationWorld同时承担：

- 原子装载与释放；
- qCMOS和MOT图像；
- SLM coherent propagation；
- 磁场与pulse响应；
- 大量测试用hidden oracle和可变参数。

一个state owner是对的，但一些physics配置寄生在virtual camera config里，一些test直接改public attributes；camera callback还可能在world lock里执行且没有unregister。

### 推荐分层

1. Apparatus configuration：模拟“这台实验”的稳定参数，可保存并由用户设置。
2. Scenario override：测试某个异常场景的临时扰动，只给tests或高级diagnostic。
3. Hidden truth/diagnostics：真实算法不能读取，只给测试验证simulation是否按设计工作。

### 哪些参数可能适合普通UI

候选包括：

- random seed；
- base loading probability；
- camera background/read noise；
- atom fluorescence scale；
- MOT magnetic optimum；
- 是否启用某类模拟设备。

像hidden SLM wavefront、每site真实depth、预设答案等不应进入普通UI，否则production算法可能旁路实验过程。

### 我的推荐思路

普通Device Manager只显示你日常会主动改变、而且具有实验意义的少数参数。其余放高级Simulation Diagnostics或test fixture。不要因为“可能调试有用”就让几十个hidden knob进入产品UI。

### 你需要回答

请列出你希望普通Simulation UI可调的参数。若你没有明确需求，我建议先只保留：seed、base loading probability、camera noise/brightness scale；其它test-only。

## 十二、你最终真正需要回答的问题

你不需要回复一百个技术细节。按下面格式回答即可：

1. **安装方式**：是否有实验室外部用户单独使用某个`zlc_*`包？是否接受整个ZLC一次安装？
2. **Measurement live**：Camera是否必须在运行中暴露全部growing raw stack，还是latest cycle + running reductions足够？
3. **Plot**：新data是否立即显示？fit每revision、只fit latest还是手动？滚轮规则选A/B/C？
4. **Task UI**：Task运行时是否允许继续操作Monitor/layout？Calibration是否显示三帧？
5. **Temperature**：20ms+更长gap、约5ms+重Calibration，还是独立protocol？
6. **Pulse/Camera**：same-shot选择第1/2/3档？remote是否跨两台机器、网络是否隔离、第二client策略、active forever timeout、板和COM现状？
7. **SLM Feedback**：是否接受all-shot fluorescence + 100-shot coarse + final更多shots + Accept/Cancel分开 + 只更新Pattern？最大validation时间是多少？profile/correction/settle/DVI/dense现状是什么？
8. **Calibration raw**：默认完整保存、外部raw+manifest，还是operator opt-in？
9. **Scan结束**：默认restore还是留在last？
10. **Simulation UI**：哪些参数需要普通用户可调？

如果你大体接受推荐，可以这样回复：

```text
1. 整个ZLC一次安装；没有外部包用户。
2. Camera latest cycle + running reductions；Scan逐步增长。
3. data立即显示；fit只处理latest；普通滚轮滚页面、Ctrl+滚轮缩放。
4. Task期间Monitor可操作；Calibration三帧。
5. Temperature选___。
6. same-shot第___档；remote使用情况___；第二client___；失联___秒SAFE；板/COM情况___。
7. 接受SLM推荐方案；final最多___分钟/shots；profile来源___；correction/settle/DVI/dense情况___。
8. Calibration raw选___。
9. Scan默认restore。
10. Simulation UI保留___。
```

你的回答只决定产品行为。具体类怎么拆、锁怎么实现、文件字段叫什么，属于后续工程实现，不会再反过来要求你做代码级选择。
