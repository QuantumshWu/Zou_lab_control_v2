# GOAL — zlc_runtime:信号数据面 + 消费口 + 节点宿主 + 呈现调度

状态:**COMPLETE**(P0-P6 / R / S / T / U / §NB 全部完成并归档到 `docs/goal-archive.md`;§API 轮门面重选已完成)
仓库:`C:\Users\eadri\Dropbox\WorkCode\Github\zlc_runtime`(所有工作只发生在这里)

> 依据:2026-08-03 三读手深读(随仓 `docs/survey-*.md` 三份,行号锚点齐全,动工前必读)。基准 = 迁移分支 SignalDataPlane 体系。抽取语料:`zlc_neutral_atom/processing/signal_plane.py`(2,418)+ `zlc_neutral_atom/runtime/` 精选 + `zlc_workbench` 呈现运行时段落。参照树只读:`..\Zou_lab_control_v1_claude\Zou_lab_control_v1`。
>
> **用户已拍板的四条裁决(不可违背)**:
> ① **派生族 same-shot 是自动不变量,不是 API**:一个信号(如相机帧)派生出的子/孙信号(ROI、fit、occupancy…)在任何 front 里必须源自同一次根 publication;族不齐整族回退上一完整拍,绝不撕裂。跨 producer 不承诺、不发明全局 shot counter。
> ② **设备回传对账机械整族淘汰**:pulse 不回信息保证同步。`SignalAssociationRequest`/`SignalEventAssociationCursor`(arm/bind/next/finish 四步)/`cause_digest` 物理对账协议**不进包**;将来 scan 编排自己数点。注意:`EventRef`(流,generation,序号)三元组是 lineage 身份词汇,**必须进包**——它服务裁决①,与被淘汰的对账协议是两回事。
> ③ **零 Qt**:唤醒全部回调注入;Qt 垫片在 zlc_ui。
> ④ **notebook 一等公民 + 有限 API**:顶层 facade 必须小,且用**带具名数字上限**的 allow-list 契约测试冻结;`notebooks/usage.ipynb` 是验收台架。(原文写的是 ≤15,当时守住了数量却没守住选谁——见下面 §API 轮的重选,初始上限 9；教程载荷整组加入后最终上限 13。)

## 铁律 / 四步开工仪式 / 三种合法收尾

> 🔴 **永不建 venv**(用户 2026-08-05 明令):依赖一律全局 `pip install`;脚本里**不许**出现 .venv 探测/偏好分支。血训:zlc_pulse 的启动器曾优先用仓库 .venv,而那个 venv 没装 pyserial,导致串口枚举抛 ModuleNotFoundError、返回零候选口、UART 探测循环一次都不进,服务器**插不插线都无条件退回 JTAG**(实验机上=常驻 1-2 GB vivado.exe)。

绝不 push;每主题小 commit;无后向兼容、干净删除;不加防御仪式(验证只在包公共边界);守卫自证非空洞;测试禁 sleep 轮询(条件变量/事件驱动 + deadline);参照树只读。每轮:读本文件全文→`pytest -q` 基线(红先修红)→选最靠前未勾项→确认判据再动手。收尾:完成勾选 commit / 受阻写 §阻塞记录 / 全绿改 GOAL COMPLETE 后只回终止陈述。

---

## 本仓血训(从已完成轮次里提炼,新轮开工前读这几条就够)

1. **契约先行是硬约束,不是礼节**:签名变更必须**先改 `docs/contract.md` 再改实现**,commit message 点名。R1 的四条事故引信全是"实现动了文档没动",并行仓照文档写 fake 会直接 TypeError。
2. **跨仓数值契约有独立的家**:`docs/fit-numeric-contract.md`,与 zlc_plot 逐字节相同,`tests/test_cross_repo_contract.py` 用 SHA-256 守着。改它必须两仓同时改。
3. **"一字段两语义"必炸**:`revision` 曾同时表示"来源数据版本"与"本批发布序号",拆成 `source_revision` / `batch_revision` 才收敛。字段命名时问一句"它回答几个问题"。
4. **validity 由生产者显式给,消费者不得用 `isnan` 反推**:值数据集用 `success`,误差数据集用 `success AND isfinite(error)`——拟合可以成功而协方差无效,NaN 落进 VALID 格会被下游当真误差用。
5. **守卫要能被突变杀死**:R2/R3 抓到的空洞(fan-out 分歧未测、follow 口从 0 回放仍 110 全绿)都是"测试存在但杀不死变异"。每条守卫都要有一次记录在案的突变实验。
6. **测试禁 sleep 轮询**,用条件变量/事件 + deadline;`cursor.next()` 这类阻塞调用必须带 timeout,否则回归会把套件挂死而不是干净变红。


## §API 轮:门面按"用户会碰什么"重选(2026-08-05 逐名裁决 + 对抗复核)

> **现状 15,裁定保留 7。** 本仓的 allow-list **带数字上限(≤15)**,是六仓里唯一守住的机械件——但守住的是**数量**,没守住**选谁**。
> **病灶一句话:门面是按"出现在方法签名里"挑的,不是按"用户会碰"挑的。** 最锋利的证据:`SelectionBridge` **不在顶层导出**,而 notebook 第 10 节亲手 `SelectionBridge(plane, "camera/frame", ...)` 构造它;与此同时门面上坐着 10 个用户**根本构造不出来**的类型——`ExactReservation`/`MonitorTap`/`FollowTap` 的构造函数直接 `raise PermissionError`(streams.py:407 等)。

**保留(7)**:`SignalDataPlane`(用户亲手构造,唯一被兄弟仓真 import 的名字)、`SignalValue`(reactive Node 契约唯一入参,**公开可构造的校验型 frozen dataclass**,写节点单测/fake 必须自己 build)、`SignalPublication`(**被保留的 `SignalDataPlane` 五个公开方法 isinstance 硬校验的入参**,plane.py:901/1168;zlc_atom 按名标注)、`AcquisitionStream`(用户调 `create()`)、`NodeHost`(生命周期缝,用户直接构造)、**`SelectionBridge`(补进来:用户亲手构造的旗舰类型,现在却在门面外)**、`__version__`。

**从顶层门面撤下(10)——实现留在子模块不动**:`SignalFront`/`ExactReservation`/`MonitorTap`/`FollowTap`/`LiveDatasetPort`(返回类型,后三者构造即 `PermissionError`)、`Node`/`RunHandleLike`(只用于注解的 Protocol,用户写鸭子类;唯一消费者自己抄了一份)、**`BoardScheduler`+`HarmonicClock`+`OwnerChannels` 整组撤下门面**(实现留在 `zlc_runtime.presentation`)(呈现调度全组跨仓零引用;审计原本想留 `HarmonicClock` 却撤 `BoardScheduler`,而**门面上再没有任何东西能吃一个 HarmonicClock**——留一半不成立。将来 zlc_ui/zlc_workbench 真接线时**整组一起加回并显式抬高上限**,这正是上限该起的作用)。

- [x] API1 按用户会碰到的 7 个核心名字重选 `__all__`，撤下 10 个实现仍保留在子模块；`__version__` 与具名上限守卫一并落地。
- [x] API2 `SelectionBridge` 进门面后，notebook 第 10 节及四个实际构造的纯数值载荷(`SelectionChange`/`SelectionRange`/`SelectionState`/`FitEventValue`)从顶层拿；载荷整组加入，`MAX_PUBLIC_NAMES` 一次性由 9 抬到 13。
- [x] API 守卫突变实证：上限错改、contract 清单漂移、撤出项重新 re-export 均使对应测试变红；恢复后 notebook 与全套 pytest 全绿。

### 通用机械判据(六仓一致;**这三条缺一条,收缩就是装饰**)
1. **上限断言在真实公开命名空间上,不只是 `__all__`** —— 具名常量 `MAX_PUBLIC_NAMES = <数字>`,断言对象是 `[n for n in dir(pkg) if not n.startswith("_")]` 减去子模块名。实测发现:zlc_data 的 `__all__` 是 70 而 `dir()` 公开项是 80,只查 `__all__` 的守卫**抓不到从别处漏出来的名字**。
2. **`__all__` 与 `docs/contract.md` 的名字集合双向相等** —— 多一个少一个都红。文档里没有名字清单的,先补清单再写这条。
3. **"撤走"= 从顶层门面拿下来,代码一行不删。** 具体是:名字从包级 `__all__` 移除、不再在 `pkg/__init__.py` 里 re-export;**实现原封不动留在它自己的子模块里,继续可以 `from pkg.submodule import Name` 用**。判据写成"顶层 `getattr(pkg, name)` 不再解析得到"(不是"包里搜不到"),同时补一条正面断言:**子模块导入路径仍然可用**。每撤一个先 grep 调用点改成子模块路径,跨仓调用点一并改;注解用字符串或 `TYPE_CHECKING`。**删代码不在本轮范围内。**
4. 每个保留的名字都要在 notebook 教程里有**真实教学用途**(仅 import、或 `x = SomeClass` 这种凑数一律不算)。
5. **`__version__` 六仓一律保留** —— 本项目被同名影子包咬过不止一次,它是唯一能写出"我 import 的是哪一份"守卫的探针。

## 阻塞记录

- 2026-08-03 / P1.1-P1.3 / 依赖门曾未满足: `zlc_data` D1 initially lacked `zlc_data.validation` and still imported `zlc_storage`; no compatibility bridge was added and the sibling was not modified. Workspace state later supplied an editable, purity-tested `zlc_data` with `validation.py`; P1 proceeded. Prior commits: `b821276` (P0.1), `1097117` (P0.2 + P0.3)。

## 开放问题(遇到记录勿擅决)

- **scan 编排已裁决为"不立引擎不立协议"**(用户 2026-08-03):硬件扫=write_scan_table+fire 一次点火;软件扫=measurement 节点里的普通 for 循环;帧→点归属=host 从编译产物纯函数算每点帧数后数数,数数安全由本包 exact 口的无损契约担保(丢帧即 StreamGap 响亮)。本包与将来任何包都**不要**出现 PointExecutor/SampleCollector 类协议;共用逻辑重复出现时提普通函数。
- zlc_plot 独立仓与新 zlc-data(角色轴)的和解(迁移或钉版)是 zlc_plot 侧的后续 cut;在此之前**同一环境不要同时 editable 安装 zlc-plot 与 zlc-data**,runtime 的 notebook/demo 不依赖 zlc_plot。
- 呈现调度与 zlc_ui 的合体集成 demo(三张假卡不同 update_ms 看分拍与整板同拍)放在 zlc_ui 或组合仓,不在本仓。
