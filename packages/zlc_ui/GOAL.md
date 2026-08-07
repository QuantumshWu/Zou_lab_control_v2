# GOAL — zlc_ui:纯视图部件包

状态:**§API 轮 COMPLETE**(R / B / PE / FV 全部完成并归档到 `docs/goal-archive.md`,不许回改;本轮只做导出面收敛)
仓库:`C:\Users\eadri\Dropbox\WorkCode\Github\zlc_ui`(所有工作只发生在这里)

> 背景:首轮 GOAL 已完成并按规则自删。2026-08-03 外部验收(3 个对抗审查 agent + 全新环境实测)结论:**地基合格**(干净环境 26 测试全绿、import 纯度守卫非空洞、三层分离干净、无呈现运行时渗入),但有 5 项违规 + 一批小修,那几轮(R / B / PE / FV)都已完成,原文在 `docs/goal-archive.md`。
> 边界宪章以 README 为准(上轮已写入);本文件补充两条裁决:
> ① `PyQt5-Frameless-Window` 依赖例外**批准维持**(README 已记、白名单已机械强制),但门面不得转出 `FramelessWindow`/`StandardTitleBar` 等第三方类型;
> ② 出向信号命名三缀扩为四缀:`*_requested / *_picked / *_committed / *_toggled`(开关类用 toggled 是诚实的),写进 README 宪章。

## 铁律 / 四步开工仪式 / 三种合法收尾

> 🔴 **永不建 venv**(用户 2026-08-05 明令):依赖一律全局 `pip install`;脚本里**不许**出现 .venv 探测/偏好分支。血训:zlc_pulse 的启动器曾优先用仓库 .venv,而那个 venv 没装 pyserial,导致串口枚举抛 ModuleNotFoundError、返回零候选口、UART 探测循环一次都不进,服务器**插不插线都无条件退回 JTAG**(实验机上=常驻 1-2 GB vivado.exe)。

与上轮相同(README 宪章 + 以下不变式):绝不 push;每主题小 commit;无后向兼容、干净删除;不加防御仪式;测试全 offscreen、禁 sleep/固定 qWait;守卫自证非空洞;参考树只读。每轮:读本文件全文→`pytest -q` 基线(红先修红)→选最靠前未勾项→确认判据再动手。收尾:完成勾选并 commit / 受阻写 §阻塞记录 / 全绿改 GOAL COMPLETE 后只回终止陈述。

---

## 本仓血训 / 现存状态(新轮开工前读这几条)

1. **notebook 是本项目六仓里唯一做对的**(11 个小 cell、17 次 print、零 assert)——**它是其他五仓 §NB 教程改造的参照标杆**。教程按功能分格、每格教一件事、用 print 给结果;断言属于 `tests/`。
2. **纯视图边界**:视图只发信号、不做决策;出向信号四缀 `*_requested / *_picked / *_committed / *_toggled`。门面不得转出 `FramelessWindow`/`StandardTitleBar` 等第三方类型(`PyQt5-Frameless-Window` 的依赖例外已批准并被白名单机械强制)。
3. **导出面 9 个,靠纪律守着,没有机械件** —— 六仓横向对照的结论是:**allow-list 没有数字上限就只是把当天的膨胀冻住**(zlc_data 有 allow-list 却是 70 个)。本仓要补一条带具名数字的上限测试。
4. **2026-08-05 实证事故**:`src/zlc_ui/board/panel_geometry.py`、`pulse/binding_cycle.py`、`acceptance.py` 三个源文件**从未提交**,而已提交的 `board/__init__.py:12` 与 `pulse/__init__.py:4` 在 import 它们——新克隆 `import zlc_ui.board` 直接 ModuleNotFoundError(移开文件已实测复现)。已补交。**教训:每轮收尾必看 `git status --untracked-files=all`,未跟踪的源文件等于仓库是坏的。**


## §API 轮:门面重选(2026-08-05 逐名裁决 + 对抗复核)

> **头号发现:顶层门面的 9 个名字,全仓和五个兄弟仓加起来的真实使用次数是 0。** 所有人都走子模块路径——`from zlc_ui.qt import ensure_qt_app`(notebooks/usage.ipynb:2)、`from zlc_ui.form import FormFieldProps, FormSpec`(:310)、`from zlc_ui.board import BoardMetrics`、`from zlc_ui.graph import FlowGraph`;全树唯一的 `import zlc_ui` 在 tests/test_qt_app_single_entry.py:102,而它只是为了断言 `QApplication.instance() is None`,**一个导出名都没碰**。
> **裁定**:不是把门面删空,而是**让它成为唯一的路**——门面留下用户真正要构造/调用的名字,并把 notebook 与 examples 的 import 改到顶层,使"一个显然的路径"成立;子模块路径不再出现在教程里。
> **本仓今天没有任何导出面守卫**(grep `__all__` 在 tests/ 零命中),9 个名字是靠纪律。

**保留(7)**:`ensure_qt_app`(用户调,唯一 QApplication/HiDPI/Fluent 缩放入口,必须先跑)、`FormSpec`、`FormFieldProps`(`FormSpec.fields` 是 `tuple[FormFieldProps,...]`,不给就写不出 FormSpec)、`FormChoice`(`FormFieldProps` 对选项**硬 isinstance 拒绝**其他类型,没有替代写法)、**`FormRuntimeContext`(反驳者证明是错撤:README 有独立章节声明它是本包的回调注入接口)**、**`BoardMetrics`(错撤:`ConsoleBoardView` 的文档化构造契约,板级几何策略的唯一注入点)**、`__version__`。

**从顶层门面撤下(3)——实现留在 `zlc_ui.graph` 不动**:`FlowGraph`/`FlowGraphEdge`/`FlowGraphNode` —— 演示用的图特性,其渲染器 `FlowGraphView` 本身就不在门面上,没有任何视图消费它们;它们本来就住在 `zlc_ui.graph`,只是不再从顶层 re-export。

- [x] API1 按上表改 `__all__`,`MAX_PUBLIC_NAMES = 9`;本仓从零开始建这条守卫(今天什么都没有)。
- [x] API2 notebook 与 examples 的 import 全部改走顶层,使门面成为唯一路径;改完 grep `from zlc_ui\.\w+ import` 在 notebooks/ 与 examples/ 零命中(`zlc_ui.graph` 等确属演示的子模块用法,若保留必须在 GOAL 里逐条列明理由)。

API2 的例外记录(逐项):`zlc_ui.fluent` 提供具体 Fluent 控件与窗口壳;
`zlc_ui.console`、`zlc_ui.pulse`、`zlc_ui.figure_viewer`、
`zlc_ui.device_manager` 提供各自 demo 必须实例化的具体视图;
`zlc_ui.graph` 只服务 Gallery 的图演示;`zlc_ui.acceptance` 是真实屏幕
验收适配器;`zlc_ui.concurrency` 只提供 Gallery 的 owner-wake 示例;
`zlc_ui.form` 只提供 Gallery 的 Qt `FluentParameterForm` 投影。它们均
不是本轮冻结的七个 facade 名字，因此教程使用显式 module-style import，
而所有可复用合同名字都从 `zlc_ui` 根导入；直接 `from zlc_ui.<module>
import ...` 在 `examples/` 与 `notebooks/` 保持零命中。

### 通用机械判据(六仓一致;**这三条缺一条,收缩就是装饰**)
1. **上限断言在真实公开命名空间上,不只是 `__all__`** —— 具名常量 `MAX_PUBLIC_NAMES = <数字>`,断言对象是 `[n for n in dir(pkg) if not n.startswith("_")]` 减去子模块名。实测发现:zlc_data 的 `__all__` 是 70 而 `dir()` 公开项是 80,只查 `__all__` 的守卫**抓不到从别处漏出来的名字**。
2. **`__all__` 与 `docs/contract.md` 的名字集合双向相等** —— 多一个少一个都红。文档里没有名字清单的,先补清单再写这条。
3. **"撤走"= 从顶层门面拿下来,代码一行不删。** 具体是:名字从包级 `__all__` 移除、不再在 `pkg/__init__.py` 里 re-export;**实现原封不动留在它自己的子模块里,继续可以 `from pkg.submodule import Name` 用**。判据写成"顶层 `getattr(pkg, name)` 不再解析得到"(不是"包里搜不到"),同时补一条正面断言:**子模块导入路径仍然可用**。每撤一个先 grep 调用点改成子模块路径,跨仓调用点一并改;注解用字符串或 `TYPE_CHECKING`。**删代码不在本轮范围内。**
4. 每个保留的名字都要在 notebook 教程里有**真实教学用途**(仅 import、或 `x = SomeClass` 这种凑数一律不算)。
5. **`__version__` 六仓一律保留** —— 本项目被同名影子包咬过不止一次,它是唯一能写出"我 import 的是哪一份"守卫的探针。

## 阻塞记录

(受阻时追加:日期 / 清单项 / 原因 / 已 commit 的进展)

## 开放问题(遇到记录勿擅决)

- pulse 编辑器的 presenter(window.py 的 `_wire_ui`/`_commit_local_edit` 前滚账本/owner 循环/压平层)与 controller 对接是用户验收 PE 接口之后的下一个 cut,发生在领域侧,不在本仓。
- `ensure_qt_app` 与 zlc_plot `ensure_qt5_application` 的组合层归属,待组合仓裁决。
