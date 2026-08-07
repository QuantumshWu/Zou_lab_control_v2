# GOAL — zlc_data:数据契约包(角色轴版)

状态:**GOAL COMPLETE**(此前各轮全部 COMPLETE 并已归档到 `docs/goal-archive.md`;§API 轮已完成)
仓库:`C:\Users\eadri\Dropbox\WorkCode\Github\zlc_data`(所有工作只发生在这里)

> 背景与裁决:zlc_runtime 拆包审计(2026-08-03)实证——独立 zlc_plot 仓附带的名字轴版 zlc_data **缺整个 value/validity/transform/selection/codec 层**,无法支撑信号面;且它与树内角色轴版存在 `DatasetSchema`/`PointTable` **同名不同物**,两版共存必出影子 import。**基准 = 迁移分支树内角色轴版**(`..\Zou_lab_control_v1_claude\Zou_lab_control_v1\zlc_data`,14 文件 ~4,084 行,numpy-only),本仓即它的独立发行家,自首个 commit 起为唯一 owner。
> ⚠️ 冲突警示:独立 zlc_plot 仓也发行一个顶层包 `zlc_data`(名字轴版)。**同一环境绝不同时 editable 安装两者**;zlc_plot 的迁移/钉版是它自己的后续 cut,不在本 goal。

## 铁律 / 仪式 / 收尾

> 🔴 **永不建 venv**(用户 2026-08-05 明令):依赖一律全局 `pip install`;脚本里**不许**出现 .venv 探测/偏好分支。血训:zlc_pulse 的启动器曾优先用仓库 .venv,而那个 venv 没装 pyserial,导致串口枚举抛 ModuleNotFoundError、返回零候选口、UART 探测循环一次都不进,服务器**插不插线都无条件退回 JTAG**(实验机上=常驻 1-2 GB vivado.exe)。

绝不 push;每主题小 commit;无后向兼容、干净删除;不加防御仪式;守卫自证非空洞;参照树只读。每轮:读本文件全文→`pytest -q` 基线→选最靠前未勾项→确认判据再动手。收尾三式:完成勾选 commit / 受阻记录 / 全绿改 GOAL COMPLETE 只回终止陈述。

## 清单

## 机械终态判据

1. 全局环境(`pip install -e .`,**不建 venv**)`pytest -q` 全绿;`import zlc_data` 不拉起 numpy 以外任何第三方。
2. grep 为零(src/):`zlc_storage`、`PyQt5`、`matplotlib`、`zlc_plot`、`zlc_neutral_atom`。
3. usage.ipynb 顶到底执行无错;README 与实现零漂移。

## R 验收修复(2026-08-03 外部验收产出;做完全部勾选后方可改回 COMPLETE)

> 环境注记(非代码问题):本仓在 Dropbox 下,验收期间观测到同步中间态导致的瞬时测试失败(收敛后 20+ 次连跑全绿)。**任何验收/交付前先确认 Dropbox 同步完成**;git 提交内容是唯一真相。

## R7 复验收尾(2026-08-03 复验产出;勾完即改 COMPLETE)

## W 轮:zlc_plot 和解前置能力(2026-08-04;依据=zlc_plot/docs/zlc-data-reconciliation.md 三缺口,已由用户侧裁决归属)

> **裁决(不可违背)**:①拓扑维必须能脱离 point 列存在(V1 合法几何);②**display 单位换算=呈现层职责,不进本包**——本包的 `unit` 字符串是 canonical 单位注解,物理计算全程 canonical;③latest-only live ingress=呈现层输入管道,不进本包;本包补的是**不可变快照的直接构造便利**,让 plot 与 runtime 共讲同一快照对象。

### W 轮机械终态判据
1. `pytest -q` 全绿;W1/W2 守卫在且突变自证(放宽点改回强制=W1 用例红,记 commit message)。
2. 契约文档该节存在;公开面新增名字列入 allow-list 契约测试。

## §NB 轮:notebook 改回教程(2026-08-05 用户裁决,六仓统一标准)

> **责任在 GOAL**:我此前在多个仓写过"notebook 必须覆盖全部公开 API / 必须真执行"这类**机械覆盖判据**,于是 notebook 被写成了"能跑通的测试脚本"——巨型 cell、成堆 `assert`、极少 `print`。**把代理指标当成了目标。**
> **用户裁决**:notebook 是**教程**——按功能分 cell、**每格教一件事**、用 `print` 展示结果让人看懂;**断言属于 `tests/`,不属于教程**。
> **本仓实测**:5 个 code cell / 125 行,最长 44 行,**10 条 assert** 对 5 次 print。
> **参照标杆:`zlc_ui` 的 notebook**(11 个小 cell、17 次 print、**零 assert**)——六仓里唯一做对的,照它的形态改。

### §NB 机械终态判据
1. notebook `assert` 计数为 0;每个 code cell ≤ 25 行且至少一次 `print`;每个 code cell 前有 markdown。
2. 移入 `tests/` 的断言全部绿。
3. notebook 带执行输出提交且零 cell 错误。

## §API 轮:逐名裁决(2026-08-05 审计 + 对抗复核;审计砍过头,已按证据回补)

> **现状 70 → 裁定保留约 54,撤走约 16。** 本仓天然类型多(schema / validity / 角色常量 / selection / transform),不该按别仓的尺度一刀切;但 70 里确实有一批**只被读、不被构造**的东西。
> **本仓是"allow-list ≠ 上限"的活证据**:`tests/test_package_guards.py:183` 的 `test_top_level_public_api_is_the_explicit_allow_list` 只断言 `set(__all__) == EXPECTED_PUBLIC_API`(硬编码 frozenset)+ `hasattr`;**全文件 grep `len(` / MAX / LIMIT / CAP 零命中**——它把 70 个名字原样冻住了,并没有拦住膨胀。
> **对抗复核推翻了审计的 7 条撤销**(证据都是兄弟仓**生产代码**,不是测试):
> - `AxisRoleId` —— zlc_atom 生产代码从顶层 import,既做公开签名的参数类型又运行时 isinstance 校验。
> - `ValidityMode` —— 用户读到 `ValueSchema.validity_contract.mode` 之后**唯一能用来分支**的枚举;zlc_runtime 生产源码 5 处做 `is ValidityMode.VALUE` 身份比较。撤了等于逼下游改用字符串比较。
> - `CommittedTransform` —— zlc_runtime 把它存成 frozen dataclass 字段并在 `__post_init__` 里校验(对 notebook 是"只读属性",对真实兄弟包不是)。
> - `ValuePayloadContract` —— 用户亲手构造并作为**位置参数**传进 zlc_runtime 生产 API `AcquisitionStream.create`。
> - `ReductionSpec` / `ReductionMethod` / `AxisSourceRef` —— 三者同生共死:保留的 `DataTransformSpec.operations` union 是 `Selection | ReductionSpec | HistogramSpec`,而 `ReductionSpec.method` 硬 isinstance 拒绝字符串、`ReductionSpec.sources` 只收 `AxisSourceRef`。撤掉任一个,顶层用户就构造不出任何一个合法 reduction。
> 另回补 `__version__`(通用判据 5)。**`HistogramSpec` 与 `ReductionSpec` 在同一个 union 里、构造形态相同——按同一条证据链一并保留,并在 commit message 里记明这次复核。**

**从顶层门面撤下(约 16,逐条理由;实现全部留在原子模块,一行不删)**:`CoordinateRangeSelection`/`IndexSelection`/`IndexRangeSelection`(`Selection` 的变体,用户走 `Selection` 的工厂构造,这些是返回类型)、`CoordinateScalar`、`ResolvedPointRows`、`TransformedData`(返回类型)、`MissingPolicy`、`ValidityPolicy`(用户不传的策略件)、`HISTOGRAM_BIN`/`HISTOGRAM_BIN_AXIS_ID`/`SCALAR`/`SCALAR_AXIS`/`SPECTRAL`(用户从不传递的角色常量;**注意 `REPEAT`/`SCAN_POINT`/`SITE`/`SPATIAL_X`/`SPATIAL_Y`/`COMPONENT`/`READOUT_EVENT` 是用户建 schema 时真的要传的,保留**)、`dataset_cell_value`/`expand_component_validity`/`expand_value_validity`(内部工具)。

- [x] API1 按上表改 `__all__`,写 `MAX_PUBLIC_NAMES = 56`;**每撤一个先 grep 兄弟仓调用点并一并改**(zlc_plot / zlc_runtime / zlc_atom 都在用本包)。
- [x] API2 `docs/contract.md` 已补完整名字清单，并以双向集合相等测试锁定契约；撤下名保留在原子子模块且顶层不再解析。

### 通用机械判据(六仓一致;**这三条缺一条,收缩就是装饰**)
1. **上限断言在真实公开命名空间上,不只是 `__all__`** —— 具名常量 `MAX_PUBLIC_NAMES = <数字>`,断言对象是 `[n for n in dir(pkg) if not n.startswith("_")]` 减去子模块名。实测发现:zlc_data 的 `__all__` 是 70 而 `dir()` 公开项是 80,只查 `__all__` 的守卫**抓不到从别处漏出来的名字**。
2. **`__all__` 与 `docs/contract.md` 的名字集合双向相等** —— 多一个少一个都红。文档里没有名字清单的,先补清单再写这条。
3. **"撤走"= 从顶层门面拿下来,代码一行不删。** 具体是:名字从包级 `__all__` 移除、不再在 `pkg/__init__.py` 里 re-export;**实现原封不动留在它自己的子模块里,继续可以 `from pkg.submodule import Name` 用**。判据写成"顶层 `getattr(pkg, name)` 不再解析得到"(不是"包里搜不到"),同时补一条正面断言:**子模块导入路径仍然可用**。每撤一个先 grep 调用点改成子模块路径,跨仓调用点一并改;注解用字符串或 `TYPE_CHECKING`。**删代码不在本轮范围内。**
4. 每个保留的名字都要在 notebook 教程里有**真实教学用途**(仅 import、或 `x = SomeClass` 这种凑数一律不算)。
5. **`__version__` 六仓一律保留** —— 本项目被同名影子包咬过不止一次,它是唯一能写出"我 import 的是哪一份"守卫的探针。

## 阻塞记录

(受阻时追加)
