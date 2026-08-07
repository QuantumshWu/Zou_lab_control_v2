# 任务3 验收报告:文档/实跑/遗留风险(zlc_plot 修复轮)

**结论:小修** —— 声称删除/拆分项全部实锤为真,实跑全链路通;但文档留有三处"已删机制"残句(上轮同病复发)、goldens 依赖未钉版导致干净 venv 非全绿、fit 模型数学测试实测为自洽简并,以及 zlc_data 同环境冲突已在用户全局环境**实际炸裂**。

---

## 1) README 与 docs 四篇零漂移校对

**anywidget 迁移主线已跟上**:README.md:92(anywidget DOM canvas、无需 `%matplotlib widget`)、docs/api.md:551(anywidget adapter)、docs/architecture.md:205-213(anywidget 注册、frame blitter、PNG 收尾)均为现状。ipympl 代码/pyproject 零残留(a69f2c2 起),docs 仅剩否定式陈述(api.md:555 "不加载 ipympl")——合格。**selector-fit lane 残句已清**:全文只剩否定式("selector/viewport/unit/resize 不启动拟合" README.md:120,124;architecture.md:271);"lane" 仅剩合法的 chrome lane(README.md:171, api.md:269)。

**三处漂移残句(bf68290 删掉的"浏览器本地绘制/overlay canvas"机制仍被描述)**:

| 位置 | 残句 | 实况 |
|---|---|---|
| README.md:100 | "Area 拖拽候选由浏览器根据 press 时冻结的 `AxisTransform` 在 overlay 本地绘制" | bf68290 "no browser overlay canvas" 已整链删除;notebook.py JS 只有单一 base canvas(src/zlc_plot/notebook.py:191-195),自述 "frame blitter and input normalizer, nothing more"(:175)。README 最后触碰于 da7487d,早于 bf68290 |
| docs/api.md:553-559 | "在透明 overlay canvas 绘制 `SelectorScene`……候选矩形由浏览器……overlay 上本地计算" | 同上;该文件最后触碰 58b402e(晚于 bf68290)但此段落漏改——是当前 HEAD 上的活漂移 |
| docs/architecture.md:151-153 | "NotebookView…paints only the transient `SelectorScene` on a browser overlay canvas" | 与同文件 L207-209("Neither the Qt widget nor the browser view paints geometry of its own")**自相矛盾** |

小项:performance.md:5 行数 26,077 vs 现树 25,900(审计有日期戳,后续 commit 又减了行,可容忍);data_contract.md 无 notebook 内容、与捆绑名字轴 zlc_data 一致(但和解时需整篇重审)。api.md:545-550 的"display bundle 带 `image/png` 静态回退、close() 换成 PNG"声称已被实跑证实(见下)。

## 2) 实跑(干净 venv,副本在系统临时目录,原仓未动)

- **安装**:`pip install ./repo`(Python 3.13.12)成功;Pillow/scipy/numpy/matplotlib 自动解析;`zlc-plot 1.0.2` 落地,wheel 同时装入顶层 `zlc_data` 包与 `zlc_plot/assets/helvetica-light-*.ttf`(实证)。
- **pytest**:🔴 首跑 **3 failed, 96 passed** —— `test_plot_session_golden.py:71` 三个 kind 全挂,delta 高达 255。根因:干净 venv 解析到 matplotlib **3.11.1**,goldens 是 3.10.8 像素;钉 `matplotlib==3.10.8` 后 **99 passed**。pyproject.toml:12 的 matplotlib 未钉版 → "99 绿"只在开发机版本上成立,不可移植。
- **examples**:`static_and_fit --headless` **exit 0**(fit_success=True);`live_simulation` offscreen+自动退出 **exit 0**(worker 清停闸门 examples/live_simulation.py:143-144 通过);`pyqt5_embed` offscreen **exit 0**。
- **usage.ipynb**:21 cell 逐 cell 执行 **0 错误、nbconvert exit 0**(kernel 已钉到 venv 解释器;⚠️ 注意默认 nbconvert 会解析到用户级 python3 kernelspec 跑在全局解释器上,首跑即中招)。**anywidget 无浏览器降级验证通过**:全部 9 处 widget display 均携带 `image/png` 静态回退,无一炸裂。小瑕:cell 16 收集 `rolling_live_metrics` 但不断言不打印,live 推进无自证。
- **mutation 探针(前车之鉴项,3 发)**:
  - 🔴 **P1 未被抓住**:把 `_gaussian_offset` 指数 -0.5→-0.25(fit.py:1073),**99/99 仍全绿**。根因是教科书自洽简并:tests/test_fit_engine.py:28 用同一个 `spec.evaluate` 合成观测再拟合,且 `initial=` 直接给真值——模型改了,合成数据同步改,永远"恢复成功"。全套件对 8 个内建模型的**曲线形状零独立锚点**。
  - ✅ P2 被抓:curve `fit_target`→None → 6 failed(kind registry 契约有牙)。
  - ✅ P3 被抓:双击中键缩放置空 → `test_notebook_raster.py::test_middle_double_click_zooms_to_area_selection_then_resets` 单点命中(58b402e 最新行为有专守卫)。

## 3) 🔴 zlc_data 和解状态

**现状:仍捆绑,且冲突已从"警示"变成"事故"。**

- pyproject.toml:37 `include = ["zlc_data*", "zlc_plot*"]` → 本仓发行版**继续安装顶层 `zlc_data`(名字轴版)**,venv 实证。
- 更糟:树内现有**两份 tracked 副本**——`src/zlc_data/`(被打包)与 `vendor/zlc_data/`(a52d868 新增的钉版快照),字节级只差 EOL。vendor/README.md:6-8 自称 "zlc_data lived under src/zlc_data until it was split" 与树实况不符(src 副本从未删除)。
- **同环境冲突已根治**:早期全局环境曾把两个不同数据模型都安装到顶层 `zlc_data`，导致 `zlc_plot` 导入失败；I1 迁移后 `zlc_plot` 依赖并直接交换角色轴 `OwnedSnapshot`，不再发行第二个顶层命名空间。
- **已有的和解动作**:vendor 钉版 + tests/conftest.py:9-11 与 usage.ipynb cell-1 的 sys.path 前插 + API 探针报错指路(计划记录=vendor/README.md "Delete this directory…once ported")。这只救了本仓测试与教程,**救不了任何普通 `import zlc_plot` 消费方**;主 README 安装节(:14-47)对冲突只字未提。
- **终态裁决**:直接迁移到角色轴 API，钉定 `zlc-data>=0.1.0`，并删除绘图仓内的名字轴副本；单位转换和 latest-only transport 留在 `zlc_plot` presentation 层。

## 4) 遗留清单勾稽

**确实完成(逐项 grep/实证)**:P0——build/dist 无、.gitignore 覆盖、egg-info 未 tracked;Pillow 已声明(pyproject.toml:14);版本 bump 至 1.0.2(18fe8c8)。P4 全子项——`supports_blit`/`_SessionCommand`/`except BaseException`/`ipympl`/`transform=None` src 全零(仅 notebook.py:524 注释提到 BaseException 概念,非 except 语句);`_kinds/` 每 kind 一模块(curve/histogram/image/rolling/facet_grid/pulse_timeline+base);session 拆出 `_session_gesture.py`/`_session_live.py`(另有 `_session_fit.py`/`_session_state.py`,fc6cd93/5ba653e/b1e6605);`FitEngine` 已导出(`__init__.py:16,101`)。P5.3 审计已记录(performance.md:3-20)。**字体确认未动**:assets 最后触碰是修复轮之前的 bc13ff4,package-data 照常打包、安装实证、`ensure_qt5_application` 注册链完好。

**声称完成但实况打折**:①"99 测试绿"不可移植(matplotlib 未钉+goldens 版本敏感,见 §2);② 测试含简并点(P1 fit 模型数学 mutation 全绿);③"文档已跟上"仅主线段落,三处删除机制残句在案(§1);④ anywidget 依赖加入(50e2405)晚于 1.0.2 bump(18fe8c8),**依赖集变了版本没再动**,且全文档无 editable 重装提示——旧 `-e .[notebook]` 环境缺 anywidget(runtime 有 `BackendUnavailableError: "NotebookView requires anywidget; install zlc-plot[notebook]"` notebook.py:132-133 兜底,算部分缓解);⑤ 工作树留有**未 commit 的功能改动**:backends.py +46 行 ipykernel wake-timer(实质性 notebook 活性修复)与 usage.ipynb 的 model_id 噪声——违反"每主题完成即 commit"常态(按约未动,如实上报;本次实跑覆盖的是含该补丁的工作树)。

**建议的小修清单(按序)**:清三处残句 → 钉 matplotlib(或 goldens 按 mpl 版本分档)→ 给 fit 模型加独立数值锚点(硬编码非 `spec.evaluate` 生成的观测样本)→ commit wake-timer 补丁 → bump 1.1.0 + README 冲突警示 + 消除 src/vendor 双副本 → 和解路线拍板(§3 ②推荐)。

**卫生确认**:被验收仓结束时 git status 与开工一致(M usage.ipynb / M backends.py / ?? .browser_check/ / ?? .claude/);参照树只读未动;全部实验在系统临时目录副本(含 mutation,均已恢复并复跑 99 passed)。
