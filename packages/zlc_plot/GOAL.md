# GOAL — zlc_plot:绘图与语义编辑面

状态:**GOAL COMPLETE**(§API 轮、L6 与两个 Notebook 演示项均已完成;此前各轮已归档,不许回改)
仓库:`C:\Users\eadri\Dropbox\WorkCode\Github\zlc_plot`(所有工作只发生在这里)

> 背景:修复轮 P0-P5 于 2026-08-04 外部验收,总体结论=**通过带小修**(P1/P2 PASS 且 4 mutation 探针全击杀、P3 anywidget 单链成立、P4 删除项全部实锤 grep 零、P5 注册表守卫有牙)。三份验收报告在 `docs/acceptance-*-2026-08-04.md`,**动工前必读**。本 goal = V 轮(验收修复,含一起正在发生的命名空间事故)+ S 轮(语义编辑面,用户 2026-08-04 拍板:reduction 走重画;facet 取 `facet_rows/facet_cols` 干净替换)。
> 用户裁决背景(S 轮的宪法):**语义字段(kind/x/group/reduction/facet)改了就重画,这是既定架构不是限制;缺的是"语义选择域"这个唯一真相源和 GUI 的门**。绝不把语义知识散进 GUI per-kind 分支。

## 铁律 / 仪式 / 收尾

> 🔴 **永不建 venv**(用户 2026-08-05 明令):依赖一律全局 `pip install`;脚本里**不许**出现 .venv 探测/偏好分支。血训:zlc_pulse 的启动器曾优先用仓库 .venv,而那个 venv 没装 pyserial,导致串口枚举抛 ModuleNotFoundError、返回零候选口、UART 探测循环一次都不进,服务器**插不插线都无条件退回 JTAG**(实验机上=常驻 1-2 GB vivado.exe)。

绝不 push;每主题小 commit;无后向兼容、干净删除;不加防御仪式;守卫自证非空洞;金样更新必须 commit message 逐条说明原因;**冻结锚(oracle)绝不用被测实现再生**;测试全 offscreen 禁 sleep。每轮:读本文件全文→`pytest -q` 基线→选最靠前未勾项→确认判据再动手。收尾三式:完成勾选 commit / 受阻记录 / 全绿改 GOAL COMPLETE 只回终止陈述。

---

## S 轮:语义编辑面(用户 2026-08-04 拍板;V1-V3 完成后开工)

## 待 codex 的演示/notebook 清单(F 轮之外的纯演示项)

- [x] notebook cell 13 过长拆分(语义编辑演示一格塞多个主题,阅读性差)。语义编辑现在独立成单一短 cell，并有独立 markdown 说明。
- [x] selector 标注与 fit 注释重叠(演示 fit + 框选时文字碰撞,只影响 demo 观感)。演示先在同一 Area scope 完成 fit，再移除 selector 后展示固定 fit annotation。

## L 轮:真机 GUI 五缺陷根修(2026-08-05 用户实测报告 + 并行根因定位,全部已复现)

> **共同病灶(先读这条)**:五个缺陷里有四个的守卫缺口是同一个——**GUI/host 异步路径整体没有测试**。`tests/test_raster_host.py` / `test_qt_widget.py` / `test_notebook_raster.py` 里 `FacetGridPlot` 出现 **0 次**,`test_raster_host.py` **从未调用 `.fit()`**;`grep aspect|get_xlim|set_aspect` 在 tests/ 命中 **0**;`grep perf_counter tests/` 命中 **0**。所以"session 层全绿"与"GUI 能用"之间从来没有测试连接。本轮每一项都必须补上走 host/renderer 真实路径的守卫,否则修完还会再坏。
> **本节所有判断已机械化**;未覆盖的取舍记阻塞问用户。

- [x] L6 **notebook 的 GUI 格根本弹不出窗口,却打印"connected"(用户实测)**
  实测(整份 notebook 全文计数):`%gui qt` **0 次**、`.show()` **0 次**、`ensure_qt5_application` **0 次**、`create_window` **0 次**。那一格做的是:
  ```python
  gui_widget = Qt5PlotWidget(gui_host)
  print('PyQt5 widget connected; fit parameters:', ...)
  ```
  三重原因叠加,**不可能显示任何东西**:① 无父窗口的 QWidget 在 `.show()` 之前不可见;② 没有 `%gui qt`,ipykernel 不驱动 Qt 事件循环,即使 show 了也不会绘制/响应;③ `gui_widget` 赋值后再无引用,随时可被 GC。**而它还打印成功消息——比报错更糟,因为它告诉用户"成了"。**
  **修法:不要手搓 widget,用仓库里已有的单源。**`examples/pyqt5_embed.py::create_window()` 的 docstring 明写它是"非阻塞入口,给 Qt 事件循环已经在跑的 notebook 用"——正是这个场景。GUI 格改为:
  ① 一格 `%gui qt`(单独一格并在 markdown 里解释:**这是让 ipykernel 驱动 Qt 事件循环的前提,不执行它窗口不会响应**);
  ② 一格 `handle = create_window()`,**把 handle 存在变量里保持引用**,并 print 出"窗口已弹出,可在其中切换数据源/改语义/框选/拟合";
  ③ 最后一格 `handle.close(wait=True)` 显式关闭(与 §NB 的教程判据一致:一格一件事)。
  **顺带**:该格现在还调了 `gui_host.fit(...)` 并把 `gui_fit.value.parameters` 打印出来——这是把 GUI 演示和拟合演示混在一格,按 §NB 拆开。
  判据:① notebook 中 `%gui` 出现且在 GUI 格之前;② GUI 格使用 `create_window()` 而非手搓 widget;③ 有显式关闭格;④ **机械守卫**:扫描 notebook,若出现构造 Qt widget 却既无 `.show()` 也无 `create_window()` 的格,即红(防止再次出现"构造了但不显示"的假成功)。

### L 轮机械终态判据
1. `pytest -q` 全绿（当前 184 passed）; `notebooks/usage.ipynb` 从头执行 0 error。
2. L1 的 host 层 facet live fit 与回调异常 Future 完成测试存在且绿;结果与 front 使用同一 `source_revision`。
3. L2 三种编辑和 20 拍 rolling 的性能守卫存在且绿;大数据复测记录在提交说明中,解析轴只物化一份 flat plane。
4. L3 的 front 提升测试逐像素比较六个 setter,并验证 pointer 仍绑定最新 front。
5. L4 非等量纲 image 使用 anisotropic fit 并恢复中心;L5 切换 x 显示单位后像素数组和 axes bbox 不变,非等量纲图不补白。
6. `test_raster_host.py`、`test_qt_widget.py`、`test_notebook_raster.py` 均经过 FacetGrid/host 路径覆盖。

## §API 轮:逐名裁决(2026-08-05 审计 + 对抗复核;审计砍到 36,复核回补 18)

> **现状 99(六仓最多)→ 裁定保留 55,从顶层门面撤下 45。**
> **本仓今天没有任何导出面守卫**:`tests/test_public_surface.py` **名不副实**——96 行、`__all__` 零命中,四条测试断的是 replace_spec 图复用、fit revision 排序、rolling 播种、facet 批量结果数,全是 session 行为。全 tests/ 唯一的 `__all__` 是 `data_factory.py:196` 自己的。所以本轮如果不把守卫建起来,**收缩的第二天就能重新膨胀**。
> **复核推翻了审计的 18 条撤销**,最硬的一条是运行时证出来的:`CurvePlot(AxisRef.point('x'), reduction='median')` 直接 `TypeError: CurvePlot.reduction must be Reduction` —— 撤掉 `Reduction` 等于把 MEDIAN/SUM/MIN/MAX/FIRST 从公开 API 里悄悄删掉。其余回补分四类:
> - **必需的参数值**:`Reduction`、`FitModelSpec` / `FitTarget`(`fit(model=...)` 声明就是 `str | FitModelSpec`)、`PlotSpec`(`PlotSession(spec=)` / `replace_spec` 的参数类型,`updated_spec` 的返回)。
> - **调用方必须 isinstance 才能解释的响应类型**:`SelectorData`、`PulseTimelineSelectionData`(`selector_data()` 的两支返回)、`PlotKind`(host 从控件拿回来是 `object`,必须判型)、`SelectionChange`(区分"选区被删"与"新增/移动"的唯一途径)、`FitEvent`(`subscribe_fit` 的载荷)。
> - **用户真会 `except` 的**:`FitCancelled` —— `replace_spec` / `clear_fit` / 新 fit / 关会话 / 删选区 都会把它设到用户手上的 `fit_async()` future 上。
> - **文档化的自定义单位路径整条**:`UnitRegistry` + `Unit`(没有工厂,必须自己构造)+ `resolve_unit` + `DEFAULT_UNITS`(是 `resolve_unit` 的必需第二参数);它们和已保留的 `DEFAULTS` 在同一段可复制粘贴的文档片段里。另加 `Qt5ParameterPanel`(与 `Qt5PlotWidget`/`ensure_qt5_application`/`RasterPlotHost` 同属文档里那**一个** `from zlc_plot import (...)` 块,留三个撤一个说不通)、`parameter_controls` / `describe_semantics` / `schema_summary`(非 Qt 前端渲染参数面板与切 plot kind 的唯一顶层路径)。
> **`__version__` 补进门面**:本仓其实**有** `__version__`(实测 1.1.0),但它**不在 `__all__` 里**——而参照树里正有一个同名 vendored 孪生,本项目被影子包咬过不止一次,这是唯一能写出"我 import 的是哪一份"守卫的探针。

**保留(55)**:`AxisRef`、`BackendUnavailableError`、`CurvePlot`、`DEFAULTS`、`DEFAULT_UNITS`、`FacetGridPlot`、`FitCancelled`、`FitEvent`、`FitModelSpec`、`FitTarget`、`HistogramPlot`、`ImageFrame`、`ImagePlot`、`ImagePointOverlay`、`LiveDataRevision`、`LivePlotController`、`NumericRange`、`PlotKind`、`PlotLabels`、`PlotSession`、`PlotSpec`、`PointStatus`、`PulseAnalogTrace`、`PulseBlock`、`PulseChannel`、`PulseDacScanSegment`、`PulseRepeatMarker`、`PulseScanRegion`、`PulseTimelineData`、`PulseTimelinePlot`、`PulseTimelineSelectionData`、`Qt5ParameterPanel`、`Qt5PlotWidget`、`RasterPlotHost`、`Reduction`、`RollingPlot`、`SelectionChange`、`SelectorData`、`SelectorKind`、`Unit`、`UnitRegistry`、`__version__`、`curve`、`describe_semantics`、`ensure_qt5_application`、`facet_grid`、`histogram`、`image`、`parameter_controls`、`pulse_timeline`、`resolve_unit`、`rolling`、`schema_summary`、`show`、`updated_spec`

**从顶层门面撤下(45)——实现全部留在原子模块,一行不删**:`ControlKind`、`CrosshairPoint`、`DisplayDescription`、`FacetFitBatchResult`、`FitDeadlineExceeded`、`FitEngine`、`FitModelRegistry`、`FitNumericTable`、`FitOptions`、`FitParameterDisplay`、`FitResult`、`FitScope`、`FitSelection`、`LivePlotMetrics`、`LiveUpdateError`、`NotebookView`、`ParameterControl`、`PlotLibraryDefaults`、`PointMarker`、`RasterBuffer`、`RasterFront`、`RasterIdentity`、`RasterOperation`、`RectangleRange`、`RegularImageFitInput`、`RelimMode`、`ReplaceSpecInitialState`、`RevisionError`、`SelectionData`、`SelectionEvent`、`SelectorState`、`SemanticChoice`、`SemanticDescription`、`SemanticFeasibility`、`SemanticField`、`SessionRevisions`、`UnitError`、`ZLCPlotError`、`axis_choices_for_schema`、`builtin_fit_models`、`default_spec`、`notebook_available`、`qt5_available`、`replace_spec_initial_state`、`semantic_controls`

- [x] API1 按上表改 `__all__`,写 `MAX_PUBLIC_NAMES = 57`(保留 55 + 2 余量);把 `tests/test_public_surface.py` 名副其实化(或另起一个文件,并给现在那四条 session 行为测试改个诚实的名字)。
- [x] API2 **异常族一致性(本轮唯一留给实现者的裁决)**:保留 `FitCancelled` / `BackendUnavailableError` 而撤下 `ZLCPlotError`(基类)/ `UnitError` / `RevisionError` / `LiveUpdateError` / `FitDeadlineExceeded`。采用“用户会分别捕获的异常才留在 facade”规则；其余实现异常保留在所属子模块，理由已写入 `docs/api.md` 与 `docs/contract.md`。
- [x] API3 `docs/api.md` 补名字清单,并由 `docs/contract.md` 与 `__all__` 双向机械校验。

### 通用机械判据(六仓一致;**这三条缺一条,收缩就是装饰**)
1. **上限断言在真实公开命名空间上,不只是 `__all__`** —— 具名常量 `MAX_PUBLIC_NAMES = <数字>`,断言对象是 `[n for n in dir(pkg) if not n.startswith("_")]` 减去子模块名。实测发现:zlc_data 的 `__all__` 是 70 而 `dir()` 公开项是 80,只查 `__all__` 的守卫**抓不到从别处漏出来的名字**。
2. **`__all__` 与 `docs/contract.md` 的名字集合双向相等** —— 多一个少一个都红。文档里没有名字清单的,先补清单再写这条。
3. **"撤走"= 从顶层门面拿下来,代码一行不删。** 具体是:名字从包级 `__all__` 移除、不再在 `pkg/__init__.py` 里 re-export;**实现原封不动留在它自己的子模块里,继续可以 `from pkg.submodule import Name` 用**。判据写成"顶层 `getattr(pkg, name)` 不再解析得到"(不是"包里搜不到"),同时补一条正面断言:**子模块导入路径仍然可用**。每撤一个先 grep 调用点改成子模块路径,跨仓调用点一并改;注解用字符串或 `TYPE_CHECKING`。**删代码不在本轮范围内。**
4. 每个保留的名字都要在 notebook 教程里有**真实教学用途**(仅 import、或 `x = SomeClass` 这种凑数一律不算)。
5. **`__version__` 六仓一律保留** —— 本项目被同名影子包咬过不止一次,它是唯一能写出"我 import 的是哪一份"守卫的探针。
6. Notebook 教程对 55 个保留名字均有真实代码用途；L6 由 `tests/test_notebook_raster.py::test_usage_notebook_uses_the_real_pyqt_window_lifecycle` 守卫 `%gui qt`、`create_window()`、持久 handle 和显式 close。
## 阻塞记录 / 开放问题

- (受阻时追加)
- §API + L6 + NB 演示项完成（2026-08-05；commit `e33a2be`）：facade 为 55 个名字，`MAX_PUBLIC_NAMES=57`；`docs/contract.md` 与 `__all__` 集合相等；45 个实现名从顶层撤下但子模块导入仍可用；异常策略只在顶层保留 `FitCancelled` / `BackendUnavailableError`。usage notebook 为 34 cells / 17 code cells，语义演示拆分、fit 展示避开 selector 文本；GUI 示例先启用 `%gui qt`，使用 `examples.pyqt5_embed.create_window()` 保存 `gui_handle`，并用独立格 `gui_handle.close(wait=True)` 收尾。`tests/test_public_api.py` 与 notebook GUI 生命周期守卫已加入。
- `python -m pytest -q` 为 `184 passed`；usage notebook 在 `QT_QPA_PLATFORM=offscreen` 下对临时副本从头执行 `34` cells、`0` error；`git diff --check` 通过，最终工作树仅由本条 GOAL 记录产生并随收尾提交归零。
- L 轮与 NB 轮完成（2026-08-05）：host/facet fit 统一 `source_revision` 并把跨线程回调异常转为 Future 异常；`DataView` 对解析轴缓存只读 flat plane，单轴 grouping 避免通用二维 inverse；semantic probe 键覆盖 spec、display state、viewport、size、DPR 且容量 256；所有 surface commit 由 host coalesced republish，notebook 六个 setter 以像素变化验收；增加各向异性 image Gaussian 与 canonical image geometry。大数据复测（repeat=5、point=2400、site=200）由 baseline 的 group→repeat 3.196s / group→site 8.709s 降至 0.569s / 3.072s；20 拍 rolling 约 0.35s。`pytest -q` 178 passed。
- Notebook 改回教程：`notebooks/usage.ipynb` 34 cells / 17 code cells，单格最多 16 行、零 assert、每格 markdown+print，离线从头执行 0 error，保存 anywidget raster 输出且不含旧 `zlc_plot_raster` 元数据。
- K 轮完成(2026-08-05; commits `78b7cca`, `5b17d55`):拟合数值表统一 canonical 值/单位,拆分 source 与 publication revision,显式发布误差 validity,单图/facet 共用 `_make_fit_numeric_table`,并让 warm-start 候选完整参与 RSS 选择且在异常/取消/超时路径失效。单位解析别名与下拉选择符号已分离。
- K7 六条变异守卫均有独立机械断言:混合 success/失败 NaN、sample unit、诊断 inset、HEADLINE/FULL、以及 `fit.py` 无 `zlc_data` 导入；K8 使用确定性噪声 facet 输入并在同一提交更新 `facet_fit.png`。
- 两个主题提交均先跑完整套件并保持绿色；最终 `python -m pytest -q` 为 167 passed。`notebooks/usage.ipynb` 用 nbconvert 无界面执行 23 cells、0 error，执行副本写在临时目录，仓库 notebook 未保存执行输出。
- J3b 按判据跳过(2026-08-04):J3a 后 64 cell × 41 point gaussian_offset 实测约 67.2–67.9ms,≤100ms 开工阈值;不引入第二套批量 LM。
- J3a/J3c 证据:解析 Jacobian commit `388789d` 记录 curve/radial-coordinate/facet 前后计时与 rtol=1e-6；live warm-start commit `5865662` 记录 direct gaussian 1.19–2.68ms→0.71–0.88ms、curve live-frame cold 8.08–9.91ms vs warm 8.17–9.00ms，并有 dense-facet 结果/误差等价性测试。
- I1 complete (2026-08-04; commits `62e1bb6`, `016f6c6`): migrated all plotting/fit/live ingress paths to real `zlc_data.OwnedSnapshot`, moved units to `zlc_plot.units`, moved latest-only transport to `zlc_plot._live_channel`, deleted the private data directory, added `zlc-data>=0.1.0`, and rewrote the namespace guard to exchange one real snapshot. `pytest -q` is 146 passed; the usage notebook executes with zero cell errors and no saved widget outputs; a fresh wheel contains neither `_zlc_data` nor a top-level `zlc_data`; goldens are unchanged.
- I2/I3 complete (2026-08-04): `5cfd34f` adds one batch overlay per FacetGrid cell, all-cell live revision fitting, atomic latest-batch presentation, and overview/focus rendering; `264f5b2` documents and guards required model headlines and the shared headline formatter. `pytest -q` is 146 passed and `notebooks/usage.ipynb` executes with nbconvert without cell errors.
- I1 reconciliation and the data-boundary document are now the terminal mapping; no private data migration remains.
- kind-isinstance 分派收敛(143 处)为后续独立轮,S5 只做"新增字段单模块"的机械证明。
- 远程 notebook 拖拽手感若不达标,按原设计加客户端 AxisTransform 预览(V6① 留的门)。
