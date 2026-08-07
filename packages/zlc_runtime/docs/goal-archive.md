# GOAL 归档 — zlc_runtime 已完成轮次

> 这里是**已完成并验收**的轮次原文(P0-P6 / R / S / T / U / §NB),留作证据与追溯。
> **活的计划在 `GOAL.md`**;跨仓数值契约已搬到 `docs/fit-numeric-contract.md`(两仓字节相同,有 SHA-256 守卫)。

## P0 仓库引导

- [x] P0.1 `git init`;src 布局 `src/zlc_runtime/`;pyproject(发行名 `zlc-runtime`,依赖 **numpy + zlc-data**,dev: pytest);`.gitignore`;README 骨架(写入上面四条裁决为宪章)。
- [x] P0.2 第一天三守卫:import 纯度测试(白名单=stdlib+numpy+zlc_data+zlc_runtime;**PyQt5/matplotlib/zlc_plot/zlc_ui/zlc_storage/zlc_neutral_atom/zlc_workbench 全禁**,逐模块真实 import,自证扫描非空);public-surface allow-list 测试(顶层 `__all__` 与冻结清单精确相等);`import zlc_runtime` 不起线程不建任何全局。
- [x] P0.3 **契约先行(并行仓的同步点)**:`docs/contract.md` **已预写**(评审誊抄自 survey),本项=校订并冻结它,使 allow-list 测试与之对齐;内容——冻结对外契约——顶层 allow-list(即 P6.1 草案)、plane 九方法 + freeze + derived 绑定族签名、streams 三口与 LiveDatasetPort 表面、Node 协议与 execution context 六能力、`RunHandleLike`、以及**已删除项负面清单**(association 四步族 / preemption 五 API / event-derived 世代——并行仓不得引用)。此文件是跨仓唯一契约权威:zlc_atom 等并行仓按它写 fake 先行开发;**任何签名变更必须先改此文件**(commit message 点名),实现跟随。签名来源=三份 survey,写文档是誊抄不是设计。

## P1 流与物化核心(搬运源:`zlc_neutral_atom/runtime/`;survey-streams-host 是底稿)

- [x] P1.1 `streams.py`(2,056 行)整体迁入 → `zlc_runtime.streams`:EventRef/EventSpanRef/Envelope/AcquisitionStream(create 返回 stream+producer 对)/三种消费口/ExactConsumerReadiness/异常族(StreamGap/StreamEndedEarly/SchemaChanged/SourceFailed)。`zlc_storage` 校验器 import 改 `zlc_data.validation`。消费契约不变:exact 绝不丢(ack 即水位, gap 响亮)、monitor latest() 跳版并记 missed、follow 可中途加入无 latest()。
- [x] P1.2 `dataset.py`(1,920 行)整体迁入 → `zlc_runtime.dataset`(与 streams 是一对):DatasetBuilder/MonitorDataset/DatasetCoverage/MonitorCoverage/ExactDatasetPreviewReader。
- [x] P1.3 小件迁入:`cancellation.py`(token/source 分权)、`_failure.py`(DetachedFailure 剥引用)、`cleanup.py`、`preview.py`、`owner_mailbox.py`(**RunHandle 类型改包内 `RunHandleLike` Protocol:snapshot/cancel/result 三方法**)、`resources.py` 只取 Arbiter 半边(ResourceKey/Claim/Busy/Lease/Arbiter + wait_until_released;**PhysicalDeviceIdentity/BindingStamp 设备身份半边留域侧**)、`dataset_output.py`+`output_name.py` 随迁(producer 输出契约是本包词汇)。
- [x] P1.4 每模块迁入即配测试(树内 `test_zlc_*` 对应测试按前缀迁入改造); exact/monitor/follow 三口各一组并发压测。
- [x] P1.5 **明确不搬**(README 记录去向+理由):`run.py`+`ports.py`+resources 设备身份半边(硬件安全执行引擎,自成闭包留域侧,本包只经 RunHandleLike 相望); association 四步协议族(裁决②)。

## P2 信号面(搬运源:`processing/signal_plane.py`;survey-signal-plane 的四分方案是底稿)

- [x] P2.1 按 M1-M5 切分迁入:`values.py`(SignalProducer/LatestProcessorControl/DerivedSignalOutput/SignalValue/SignalPublication/SignalFront)/ `processor_lane.py`(latest-only 单 worker 执行缝)/ `registry.py`(generation 生命周期+回收闭包)/ `front.py`(**lineage 行走 + front 构建改写为纯函数** `build_front(...)`——它是裁决①的引擎,也是全文件最值得独立测的算法)/ `plane.py`(门面:锁+发布核心+derived+freeze)。`dataset_output` import 改包内;`canonical_text` 改 `zlc_data.validation`。
  注:当前 `values.py`/`processor_lane.py`/`registry.py` 是兼容 shim，`plane.py` 仍是实现单体；功能无损，默认不做真实拆分。
- [x] P2.2 **删 preemption 机器**(实证 `bind_generation_source` 全仓零调用死代码):`bind_generation_source`/`release_generation_source`/`withdraw_dependency_closure`/`finish_dependency_retirement`/`require_active_generation` 五 API 全删;保留 `_retirement_closure_locked`(它是 withdraw_derived/retire/detach_live 的正常路径)。应用侧的显式 stop-then-start 由域仓自己改,不在本仓。
- [x] P2.3 防御收缩:内部 locked helper 的重复 isinstance 仪式删(锁窗第二遍只查 sequence/retired);`SignalFront.__post_init__` 全量自检降为 debug 断言;公共边界验证保留。
- [x] P2.4 **裁决①落成机械测试**(本包最重要的测试):合成祖→子→孙三级链 + 慢 processor 注入,断言每个 front 内族成员 `_publication_roots` 恒一致、不齐时整族回退上一完整拍、恢复后一起前进;processor 跳版时 coverage.missed 记账正确;`WeakKeyDictionary` parent 载荷在 front 存活期内不被回收(强引用持有责任测试)。
- [x] P2.5 宿主↔plane 契约面显式化:plane 公开面收敛为九方法(reserve/retire/attach/detach_live/mark_changed/publish_final/latest_publication/attach·cancel_latest_only_processor/publish_processor)+ freeze + derived 绑定族 + bind_owner_wake;`publication_owner` 返回值收窄为 opaque token。

## P3 信号事件投影层(搬运源:`runtime/signal_source.py`,只取投影半边)

- [x] P3.1 迁入:SignalEvent/SignalEventCursor/SignalEventSource(Protocol)/StreamSignalEventSource/Authoritative 投影四件套 + SignalProjectionAuthority codec(投影复用 zlc_data.transform 单源的设计保持)。
- [x] P3.2 **不迁**(裁决②):SignalAssociationRequest/SignalEventAssociationCursor/SignalEventAssociationSource 及 plane 侧 `bind_processor_event_source`/`signal_event_binding`/`has_event_association`/event-derived 世代(`bind_event_derived`/`publish_event_derived`)——event 关联整族不进包;P2.1 迁 plane 时同步删这些方法与 `_GenerationState` 的 event kind。README 记:将来 scan 编排若需逐点关联,由编排层数点,另行最小设计。

## P4 呈现调度(重建源:window.py 呈现段 + window_runtime.py;survey-presentation 的 A.2 签名是底稿)

- [x] P4.1 `zlc_runtime.presentation`:`WakeSink` Protocol / `OwnerChannels`(三通道 pending 位,data 通道含 bind_owner_wake token 借还)/ `HarmonicClock`(谐波 update_ms + rebase + group_due 纯算术)/ `SurfaceUpdate`(host 字段改 opaque token,不 import zlc_plot)/ `SurfacePort` Protocol / `SurfaceBatchArbiter`(all-or-nothing 入批、整批 done 才上、任一失格整批弃、绝不上半个板)/ `BoardScheduler`(freeze→分组→due→enqueue)。保留"mark_changed 只在有 reactive 下游才推 wake、纯显示靠 timer 拉"的设计。
- [x] P4.2 **欠拍公平回收**(迁移分支已把旧主线的 `_beat_owed` 丢了,这是当年对抗审查抓回来的保证,不能丢第二次;旧判据依赖已死的 render_loop.busy,不可照抄):BoardScheduler 加组级 owed 位——due 拍因成员缺值/prepare 失败流拍即记欠,此后每个 base tick 重试直至成功。测试:update_ms=2000 的组流拍一次,断言下一 base tick 即补,不黑 2 秒。
- [x] P4.3 `window_runtime.py` 三函数原样迁入(submit_compute 双 executor / stage_and_replace_export / cancel_export_commits);全部纯件配测试(HarmonicClock 纯函数表驱动;Arbiter 用假 port+假 future 全路径)。

## P5 节点宿主(重建源:`runtime/hosted_run.py`;survey-streams-host §4 是底稿)

- [x] P5.1 最小 Node 协议(从 hosted_run `__init__` 参数表直接导出):`kind ∈ {finite, reactive}`;finite=`execute(ctx)`(ctx 六能力:cancel_requested/start_and_wait/open_live_dataset/open_exact_dataset/publish_final/warn);reactive=`evaluate(SignalValue)→Mapping`+恰一输入信号键。
- [x] P5.2 `NodeHost` 骨架迁入:start/cancel/poll/shutdown 单表面、generation 重置纪律、LogicNodeObservation、live 附着纪律(一 generation 一 live)、processor 回调四件套(validate/evaluate/accept×3)、"声明了输出却没发布=硬失败"。信号命名策略(`@logic/{id}/{name}`)改注入参数。descriptor/ApplicationContext/device_requirements 全留域侧(README 记)。
- [x] P5.3 宿主测试:fake finite node(成功/失败/取消/声明未发布)、fake reactive node(跳版/失败/cancel)、shutdown 拒关条件、mailbox 陈旧完成防御。

## P6 有限 API + notebook 验收台架

- [x] P6.1 顶层 facade 定稿(≤15 名,建议:`SignalDataPlane`、`SignalFront`、`SignalValue`、`SignalPublication`、`AcquisitionStream`、`ExactReservation`、`MonitorTap`、`FollowTap`、`LiveDatasetPort`、`NodeHost`、`Node` 协议、`BoardScheduler`、`HarmonicClock`、`OwnerChannels`、`RunHandleLike`);扩展/内部件全部留子模块;allow-list 测试同步冻结。`live_dataset.py` 随 P1/P2 迁入后在此收口。
- [x] P6.2 `notebooks/usage.ipynb` 顶到底可执行(全假数据、零设备、零 Qt):起假相机 producer 线程(可调帧率)→ 挂两级假 processor(模拟 ROI→fit)→ `freeze()` 循环里**现场验证裁决①**(打印每拍族根 publication,注入慢 processor 亲眼看整族等拍回退)→ 三种消费口逐个演示(exact ack 水位、monitor missed 记账、follow 中途加入)→ HarmonicClock+BoardScheduler 假 tick 驱动 + 假 SurfacePort 打印批次。这个 notebook 就是用户验收台架。
- [x] P6.3 `examples/demo_signal_flow.py`:headless 脚本版时间线打印,`--once` 退出码 0;README 定稿(宪章、模块地图、不搬清单、与 zlc-data 的关系、"zlc_plot 集成 demo 待 zlc_data 和解后补"注记)。

## 机械终态判据(全绿才 GOAL COMPLETE)

1. 干净 venv(装 zlc-data + 本包)`pytest -q` 全绿;`import zlc_runtime` 轻量、不起线程。
2. grep 为零(src/):`PyQt5`、`matplotlib`、`zlc_storage`、`zlc_neutral_atom`、`zlc_workbench`、`zlc_plot`、`SignalAssociationRequest`、`bind_generation_source`、`withdraw_dependency_closure`。
3. 顶层 `__all__` ≤15 且 allow-list 测试冻结;三守卫全部自证非空洞。
4. 裁决①传递性测试、欠拍补拍测试、三口并发压测、宿主全路径测试存在且绿。
5. usage.ipynb 顶到底执行无错;demo_signal_flow.py `--once` 退出码 0;README/docs 零漂移;LOC 报告(语料 ~11k,预期 ~8-9k,超出逐项说明)。

## R 验收修复轮(2026-08-03 验收产出;先读 docs/acceptance-*-2026-08-03.md 三份;全部勾完并全绿才改 COMPLETE)

- [x] R1 **必修(跨仓事故引信)契约同步**——契约先行:先改 `docs/contract.md` 再核对实现,commit message 点名:① `attach_latest_only_processor` 契约写 `(signal_key, control, initial_publication)`,实现是 `(node, *, source_name=..., initial_publication=...)`(plane.py:924-930)——按 FROZEN 规则把契约校订为实现现状(实现已被测试锁定;zlc_atom 按旧契约写 fake 会直接 TypeError);② **补 `set_front_signals` 进契约**——族一致性只对该声明集生效,不调用则任何族都不受回退保护,并行仓 fake 缺它写不出正确行为;③ `publish_final` 形参名 `projected` 与契约 `outputs` 对齐(关键字调用会炸);④ `publication_owner`/`direct_parent_publications`/`withdraw_processor` 公共存在但契约未列——列入或显式标注包内私有,二选一。
- [x] R2 **裁决①守卫封洞**(证据 acceptance-plane §1.3):① fan-out 双 derived 分歧测试——camera→{roi, fit} 双派生,一快一慢,source 推进到 seq2 只有 roi 追上,断言 front 整族回退 {camera:1,roi:1,fit:1}、追上后一起前进(封死"根集检验+逐名合并同时禁用仍 110 全绿"的联合空洞;审查探针脚本即现成模板);② WeakKey 保活测试加强:states 推进一代、plane 不再强持 seq1 后再断言 parent 链可解析(现测试的 root 被 state 强持有,断言恒真)。
- [x] R3 **follow 口补测**(mutation D 存活实锤:把中途加入改成从 0 回放,110 全绿):中途 join 不回放、逐序无损、gap 响亮三断言;顺带把 `test_runtime_streams.py:146` 的 `cursor.next()` 加 timeout(探针 B 显示此类回归会把套件挂死而非干净红);P1.4 已勾的"三口各一组并发压测"实况未兑现——补 producer/consumer 双线程高吞吐压测,或在 GOAL P1.4 明记裁减理由。
- [x] R4 宿主补测与裁决项(证据 acceptance-presentation-host):① reactive 真跳版测试(gate 住 evaluate(rev1) 期间连发 rev2/rev3,放行后断言 evaluate 序列==[1,3] 且派生 parent==rev3);② cancel→restart 测试(cancel 至 terminal 后再 start,第二代正常跑完——变异 H3 存活的 stop_event 复位缺口);③ 同一代第二次 `open_live_dataset` 必 raise(host.py:565-566 无覆盖);④ **裁决项(默认树内语义,用户可否决)**:live-opened finite 成功但未 publish_final——树内=成功保 FINAL(hosted_run.py:463-485 有注释说明),现实现=failed;默认改回树内语义并补测试,用户明示要新语义则保留并记 README;⑤ 删死变量 `self._generation`(host.py:210/430)。
- [x] R5 死代码与小件:plane.py 五处零调用删除——`active_processor_bindings`(:519)、`_state_for_generation_ref_locked`(:671,preemption 遗孤)、`_publication_roots`(:1637)/`_collect_publication_ancestry`(:1656)/`_name_is_ancestor`(:1680)(后三者是 front.py 活体的死副本,违单源);LiveDatasetPort 的 `fail()`/`source_terminal()` 分支补测。
- [x] R6 簿记收口:P5.1-P5.3、P6.1-P6.3 按实况勾选(验收已确认完成于 264dd86/bf03a9d,勾选是簿记不是新工作);GOAL P2.1 的"四分"名实不副——values/processor_lane/registry 实为 shim、plane.py 仍单体,在 P2.1 后补一行注记(功能无损,默认不真拆);全部勾完跑终态判据,绿则改 COMPLETE。

## S 轮:SelectionBridge——ROI/fit 在图上自动衍生新信号(2026-08-04 用户裁决)

> **用户语义(唯一真相):可见即所得的范围切**——image 上的 area 选区=切出对应 2D 区域的子图块;curve/histogram 上的 area/x_range=切出对应 x 范围内的数据。fit 结果=发布参数标量信号。派生信号种类由选择器类型机械决定,不是用户配置。
> **无前置,纯数值过桥**:桥切的是**上游信号的 role-axis snapshot**(本仓信号面固有契约);从 plot 侧过桥的只有选区的纯数值(轴名字符串 + canonical 闭区间 + facet 轴值),Protocol 就按纯数值定义——不接收 zlc_plot 的任何对象类型,与 zlc_plot 的 I 轮(zlc-data 和解)完全并行、互不等待。
> **解耦硬约束**:本仓**不 import zlc_plot**(import 纯度守卫加 zlc_plot 进禁单)。桥依赖两个 Protocol:选区事件源(`subscribe_selection(callback)->unsubscribe`,回调收 (change, state))与选区数据读取(`selector_data(kind)` 鸭子面)——zlc_plot 的 PlotSession/RasterPlotHost 天然满足,单测用 fake,跨包集成测试放 notebook 验收台架(环境装了两包)。
> **裁决①自动满足**:派生族(roi_frame/roi_value/fit_*)经 processor 面发布,与源 publication 同拍;绝不发明对账。

- [x] S1 **契约先行**:`docs/contract.md` 增 SelectionBridge 面(构造参数、两个 Protocol、派生信号命名与 schema、生命周期),先 commit 契约再写实现;顶层 facade 名额检查(≤15 allow-list,超了记阻塞议替换)。(5c3e07f)
- [x] S2 **SelectionBridge 实现**:
  ① 订阅:COMMITTED → 读选区(canonical 范围+kind+facet 轴值)→ 由纯数值(轴名+范围)构造 `zlc_data.Selection`(area on image → `Selection.rectangle`(两轴坐标闭区间);x_range 或 curve/histogram 上的 area → 单轴 `coordinate_range`;facet cell 内选区追加 facet 轴条件);轴名→上游 snapshot 轴的解析失败=清晰报错不兜底;UPDATED(拖拽中)不触发;REMOVED → 派生信号 retire。
  ② 切片:`resolve_selection_indices` + `materialize_derived_dataset`(规则子盒→`roi_frame`)+ 均值标量(`materialize_scalar_dataset`→`roi_value`);切的是**上游信号的 snapshot**(bridge 构造时绑定上游 signal key),不是 plot 内部数据。
  ③ 重切触发:选区 COMMITTED 时一次 + 上游每个新 publication(经 `attach_latest_only_processor`,背压自然合并);发布带 `source_publication` 血缘。
  ④ fit 衍生:接 plot 的 fit 完成事件(同 Protocol 化:fit 事件源),把 FitResult 参数/误差经 `materialize_scalar_dataset` 发布为 `fit_<param>` 标量族(v1 FitProcessor 语义)。
- [x] S3 **测试(fake 事件源+真 zlc_data 切片,数值断言)**:image area→子盒逐元素等于 numpy 手切;curve x_range→行子集;facet cell 选区→含 facet 条件;上游新 revision→派生自动跟切;REMOVED→retire;fit 标量数值;守卫自证:把 Selection 构造改错(开区间/轴颠倒),对应数值断言必红(记 commit message)。(0150b62; 两个变异探针均被击杀)
- [x] S4 **notebook 验收台架**:usage notebook 增一节——真 zlc_plot 会话框选 ROI → bridge → 派生 roi_frame/roi_value 信号被下游消费显示;fit armed → fit_x0 标量流。真执行提交。(c7a8acc; nbclient/Agg 零错误)

### S 轮机械终态判据
1. `pytest -q` 全绿;S3 全矩阵在;守卫突变记录在 commit message。
2. import 纯度守卫含 zlc_plot 禁单且自证非空洞。
3. 契约文档与实现零漂移;facade allow-list 测试绿。
4. notebook 该节真执行零错误。

## T 轮:fit 派生推广到 facet 批量 + 修复跨仓台架(2026-08-05)

> **归属**:ROI/fit 的派生与发布是本仓职责(S 轮已交付**标量** fit 派生)。zlc_plot 只出纯数值(其 J2 轮补齐列式暴露面:逐 cell 值/标准误/单位/facet 坐标)。**本轮不需要 zlc_data 改动**——(1, N) 的 schema 用 zlc_data 现成的公开 API 构造,数据集用现成的 `materialize_derived_dataset` 物化。
> **本轮无跨仓前置**:T1 立即可做;T2/T3 的输入若暂缺(zlc_plot J2 未交付),先按契约用 fake 数值表开发与测试,真接线放 T5。

- [x] T1 **修复跨仓台架(先做,现已破,两个独立原因)**:`notebooks/usage.ipynb` S4 节 ① 第 315 行仍 `from zlc_plot._zlc_data import ...`,该私有包已在 zlc_plot I1 轮删除;② 第 409-417 行用 `event.result.parameters.items()` 与 `event.result.errors.get(name)`,而 `FacetFitBatchResult` 根本没有 `parameters` 属性(它有 `parameter_values`),`errors` 还是逐 cell 失败消息不是逐参数误差。两处都改对,重跑带执行输出提交。判据:notebook 全执行零错误。(e446678)
- [x] T2 **`FitEventValue` 推广为"带可选样本轴的参数表"**:参数值从单浮点推广为 N 长数组 + 逐样本 validity + 可选样本轴(名字 + N 个坐标值 + 坐标单位);**标量是 N=1 的退化情形,一条路,不留两套**。契约文档先改再实现。(176b55a; e446678)
### 跨仓数值契约(zlc_plot J2 产出 == zlc_runtime T2 消费;两仓逐字相同,任一方改动必须先改这段)

facet 批量拟合对外的**纯数值表**(不含任何 zlc_plot / zlc_runtime 对象):

- `parameter_names: tuple[str, ...]` —— 参数的程序名(非符号),顺序即列顺序。
- `parameter_units: Mapping[str, str]` —— 参数名 → **与 `parameter_values` 同一系统**的单位字符串;无量纲为 `""`。
  > **铁律:值与单位必须同系统。**要么两者都 canonical,要么两者都当前显示单位——**绝不允许值是 canonical 而单位报显示单位**(消费者会给 canonical 数值贴上显示单位标签,产生 10^n 倍的物理错误)。本契约选定:**两者都用 canonical**(派生数据进入信号面后由呈现层自行换算,与 zlc_data 的"unit 字段=canonical 注解"归属一致)。守卫:改变显示单位不得改变 `parameter_values` 与 `parameter_units` 中的任何一个。
- `parameter_values: Mapping[str, np.ndarray]` —— 参数名 → float64 数组,长度 = cell 数,失败 cell 为 NaN。
- `parameter_errors: Mapping[str, np.ndarray]` —— 同形状标准误;失败 cell 或协方差无效为 NaN。
- `success: np.ndarray` —— bool 数组,长度 = cell 数,表示**该 cell 的拟合是否成功**。
  > **值数据集**的 validity = `success`。**误差数据集**的 validity = `success AND isfinite(error)` —— 因为拟合可以成功而协方差无效(此时标准误为 NaN),把 NaN 写进标记为 VALID 的格子会让下游把 NaN 当成真误差用。两个数据集的 validity **由生产者显式给出,消费者不得用 `isnan` 反推**。
- `sample_axis_name: str` —— facet 轴的显示名。
- `sample_coordinates: np.ndarray` —— float64,长度 = cell 数。facet 坐标为数值时即其 canonical 值;为 TEXT 时为 `0..N-1` 序号。
- `sample_unit: str` —— 坐标单位;TEXT 坐标情形为 `""`。
- `sample_labels: tuple[str, ...] | None` —— 仅 TEXT 坐标情形非空,与坐标一一对应。
- `source_revision: int` —— 本批拟合所依据的**来源数据** revision(服务血缘与 same-shot 族;同一份数据重复拟合时不变)。
- `batch_revision: int` —— **本次发布自身**的单调计数器,**每产生一批就 +1**,与来源数据是否变化无关(下游 `update_data` 要求 schema 恒定且 revision 严格递增,靠的是这个)。
  > 这两个是不同的东西,**不许合并成一个字段**:前者回答"这批结果来自哪一拍数据",后者回答"这是第几批结果"。

单图拟合是 N=1 的退化情形:同样的字段,`sample_axis_name=""`、`sample_coordinates=[0.0]`、`sample_unit=""`。**两侧都只有一条代码路径处理这两种情形**——生产侧必须由同一个函数产出两种形态(不许单图与 facet 各写一份),消费侧必须能接受 N=1 且 `sample_axis_name` 为空的表。**N=1 的 facet(单 cell 网格)是合法输出,消费者不得拒绝。**

- [x] T3 **发布向量派生信号**:批量情形下 `fit_{parameter}` / `fit_{parameter}_error` 发布 (1, N_cells) 数据集——point 列 = 样本坐标(TEXT 坐标则用序号),value unit = 该参数单位,失败样本 **validity 标 invalid**(不用 NaN 混进 valid);经 `materialize_derived_dataset` 物化并带 `source_publication` 血缘;派生族 same-shot 不变量照旧自动满足。
  **revision 单调**:下游 `PlotSession.update_data` 要求 schema 指纹恒定**且 revision 严格递增**,派生信号必须满足,写成判据。
- [x] T4 **测试**:数值断言(逐 cell 值与输入参数表逐元素相等、失败样本 invalid、单位正确、TEXT 坐标降级为序号);标量退化情形与 S 轮既有断言**逐位等价**(证明没改坏);连续两批派生的 revision 严格递增;守卫突变(让批量路径绕开共用构造、或让失败样本仍为 valid)必红,记 commit message。(e446678)
- [x] T5 **真接线**:zlc_plot J2 交付后,notebook 增一节——真 facet 全 cell fit → 数值表 → bridge → `fit_center` 向量信号 → 被第二个 plot 会话消费画成"参数 vs facet 坐标"曲线。真执行提交;J2 未就绪则记阻塞停在此项。(e446678; zlc_plot J2 table tests 6/6)

### T 轮机械终态判据
1. `pytest -q` 全绿;T4 全矩阵在;突变记录在 commit message。
2. notebook 全执行零错误(含修好的 S4 与新增批量节)。
3. 契约文档与实现零漂移;标量路径行为逐位不变。

## U 轮:T 轮验收返工(2026-08-05 六路对抗验收产出)

> **验收结论**:功能实测是真的——notebook 从头跑通零错误(含真实 zlc_plot facet fit 过桥进第二个 `PlotSession`),`_materialize_fit_outputs` 确是无分支单路径,S 轮的 `FitParameter` API 已彻底删除,独立数值探针确认值/误差/单位/validity/point 列逐元素正确。**但有一条静默数据丢失、一条合法输入被拒、以及多处无守卫。**
> **两条根因在我钉的契约里**(契约已于本轮修正,见上方跨仓契约块):误差数据集的 validity、`revision` 一字段两语义。U1/U2 是照新契约返工。
> **本节所有判断已机械化**;未覆盖的取舍记阻塞问用户。

- [x] U1 **误差数据集的 validity(照新契约)**:当前值与误差两个数据集共用同一个 `success` 掩码(`selection_bridge.py:1242` `CellValidity(event.success.reshape(1, sample_count))`),而拟合可以成功但协方差无效(标准误 NaN)→ **NaN 落进标记 VALID 的格子**,下游按契约信任 validity 就会把 NaN 当真误差用。改为:值数据集 validity=`success`,误差数据集 validity=`success AND isfinite(error)`。
  判据:构造"成功但协方差无效"的样本,断言误差数据集该格 INVALID、值数据集该格 VALID;并断言 VALID 格内**永不含 NaN**(全参数全数据集通用断言)。
- [x] U2 **接受拆分后的 revision 两字段**:按新契约消费 `source_revision`(血缘)与 `batch_revision`(单调,驱动派生数据集自身的 revision)。判据:同一来源数据的连续两批派生,`batch_revision` 严格递增且能连续喂进要求单调的下游。
- [x] U3 **静默数据丢失(race,最高优先)**:来源 revision 跳变后紧接着的一批拟合**发布成功,随后被迟到的陈旧拟合撤回**——向量 fit 信号从 front 消失,而 `bridge.last_error` 仍是 `None`,**无任何错误痕迹**。修法:陈旧拟合的撤回不得移除更新的一批;若确需撤回必须记录可见错误。
  判据:复现该竞态的回归测试(来源 revision 跳变 → 新批发布 → 注入迟到的陈旧撤回),断言新批仍在 front;守卫突变必红。
- [x] U4 **合法输入被拒:单 cell facet**:生产侧 `FacetFitBatchResult.__post_init__` 强制非空轴名,于是 size-1 repeat 轴的 facet 会给出非空 `sample_axis_name`,而消费侧对该形态拒绝。按新契约"**N=1 的 facet(单 cell 网格)是合法输出,消费者不得拒绝**"修正消费侧。
  判据:1-cell facet 表能完整发布并被下游消费;与 N=1 单图表走同一条路。
- [x] U5 **`sample_labels` 被消费者完全忽略**:生产侧对 TEXT 坐标填充它,而 `_materialize_fit_outputs` 从不读取——point 列只用 `sample_name`,文本标签丢失。修法:TEXT 坐标情形把标签带进数据集(作为列标签或等价载体),使下游能显示 `alpha/beta` 而不只是序号。
  判据:TEXT 坐标的端到端断言——下游能取回原始文本标签。
- [x] U6 **补齐无守卫的不变量**:① T2 的"一条路"(标量是 N=1 退化)无机械守卫——插入一条并行分支测试仍全绿;② 样本坐标列无守卫,因为两个夹具的坐标**恰好都是 0..N-1**,坐标被换掉也抓不到(改夹具用非序号坐标);③ 契约文档与实现漂移(终态判据 3 明确要求零漂移),逐句校对并补导出/字段清单契约测试。每条突变自证必红并记 commit message。
- [x] U7 **流程纪律**:突变实验的**结果**(哪条测试变红)必须写进 commit message,不能只写探针主题。

### U 轮机械终态判据
1. `pytest -q` 全绿;notebook 全执行零错误。
2. U1 的"VALID 格内永不含 NaN"通用断言存在且绿。
3. U3 竞态回归测试存在,突变必红。
4. U4 单 cell facet 端到端通过;U5 文本标签端到端可取回。
5. U6 三条突变逐条自证必红,结果记 commit message。
6. 跨仓契约块与 zlc_plot 侧**仍逐字相同**(本轮已同步修正,不得单侧再改)。

## §NB 轮:notebook 改回教程(2026-08-05 用户裁决,六仓统一标准)

> **责任在 GOAL**:我此前在多个仓写过"notebook 必须覆盖全部公开 API / 必须真执行"这类**机械覆盖判据**,于是 notebook 被写成了"能跑通的测试脚本"——巨型 cell、成堆 `assert`、极少 `print`。**把代理指标当成了目标。**
> **用户裁决**:notebook 是**教程**——按功能分 cell、**每格教一件事**、用 `print` 展示结果让人看懂;**断言属于 `tests/`,不属于教程**。
> **本仓实测**:最严重:6 个 code cell 共 486 行,**最长一格 256 行**,19 条 assert。
> **参照标杆:`zlc_ui` 的 notebook**(11 个小 cell、17 次 print、**零 assert**)——六仓里唯一做对的,照它的形态改。

- [x] NB1 **拆格**:每个 code cell **≤ 25 行**,只教一件事;每个 code cell **前面有 markdown** 说明"这一格教什么、为什么这样用"。
- [x] NB2 **去断言**:notebook 中 `assert` 计数归 **0**;凡有真实守卫价值的断言**移入 `tests/`** 成为真正的测试(不要直接丢弃)。
- [x] NB3 **给结果**:每个 code cell 至少一次 `print`(或等价的可视输出),让读者看得到 API 返回了什么、字段是什么意思。
- [x] NB4 **按功能覆盖,而不是按名字覆盖**:废除"每个导出名都要被使用"这条判据;改为"**每个公开能力都有一格真正的教学**"。仅仅 import 一下、或写 `x = SomeClass` 这种凑数用法,一律不算。
- [x] NB5 **真执行**:带执行输出提交;无外部依赖(硬件/服务器)的部分必须在干净环境从头跑通零错误。

### §NB 机械终态判据
1. notebook `assert` 计数为 0;每个 code cell ≤ 25 行且至少一次 `print`;每个 code cell 前有 markdown。
2. 移入 `tests/` 的断言全部绿。
3. notebook 带执行输出提交且零 cell 错误。
