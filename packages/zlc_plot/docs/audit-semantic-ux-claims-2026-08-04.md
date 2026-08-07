实验副本已清理干净(仅在系统临时目录做过变异)。以下为最终审计报告。

# zlc_plot 语义面体验审计(任务C:SE1-SE7 勾稽 + 设计一致性)

审计手段:全部结论都在真实路径上复现过——离屏实跑 `examples/pyqt5_embed.py`(notebook cell 20 唤起的就是它)、session 级重放 notebook cell 3/13 的语义链、临时副本 mutation 实测守卫。被审仓零修改(唯一未提交改动 `notebooks/usage.ipynb` 仅是执行输出/execution_count,无内容漂移)。

---

## 一、用户真实路径上仍然坏/怪/误导的每一处(按严重度)

### 1.【坏】语义编辑后所有文字标签变成谎言 —— "还是不对"的最大嫌疑
离屏实测 embed:语义 kind 从 Curve 切到 Histogram 后,`describe_display` 返回 `{'title': 'Live curve', 'x_label': 'Time', 'y_label': 'Signal'}` —— **直方图(x=信号幅值 mV)的横轴标着 "Time"**。session 级实测更直白:把 X 从 Time 换成 Field 后,轴标签渲染为 **`'Time (mV)'`** —— 陈旧文字 + 新单位的嵌合体。notebook 同病:cell 6 `set_labels(x='Bias')` 后,cell 13 的每一次语义替换(含 x 换轴)轴上永远是 `'Bias (mV)'`。
双根因:
- `examples/pyqt5_embed.py:789` —— kind 切换后 `candidate = replace(candidate, labels=current.labels)` 把旧 kind 的 authored labels 硬贴到新 kind;
- `src/zlc_plot/session_policy.py:76-95` —— `_retain_parameters` 无差别保留 `title/x_label/y_label/value_label`;viewport 有 `_viewport_signature`(:55-73)判定语义兼容才保留,**label 没有对应的失效签名**;
- `src/zlc_plot/rendering.py:1011-1032` —— explicit label 恒覆盖投影派生标签。
复现:embed → Semantics 组 kind 选 Histogram;或任何 session `set_labels` 后 `replace_spec` 换 x。

### 2.【坏】旗舰 schema 下 x/group/facet combo 出现肉眼可见的重复项
离屏实测 embed 的 X combo = `['Repeat', 'point row', 'Field (mV)', 'Time (ms)', 'Field (mV)', 'Time (ms)', 'Site']` —— "Field (mV)"、"Time (ms)" 各出现两次,一次是 point-table 列(逐行坐标),一次是 topology 维度(网格轴),标签完全无法区分;选错孪生项行为微妙不同(facet demo 实测 `x=point('time')` 被接受,渲染语义却与 dimension 不同)。SE2 只做了 **value 恒等去重**,label 层碰撞原样漏过。守卫打不中:`tests/test_semantic_ui.py:22-34` 与 `tests/test_semantics.py:26-34` 的 schema 都没有 topology,天然无碰撞。
根因:`src/zlc_plot/semantics.py:171-181`(`axis_choices_for_schema` 同时枚举 POINT_COORDINATE 与 POINT_DIMENSION)+ `:184-206`(`_axis_label` 不产生任何区分后缀)。
复现:跑 embed(或 notebook schema),展开 Semantics 组任一轴 combo。

### 3.【坏】kind 域"消失而非灰掉" + FacetGrid 单向门(SE7 与实现明确不符)
SE7 原文:"推不出默认的 kind 在语义组**灰掉**"。实现:`semantics.py:300` 用 `admits` 过滤直接**不列出**;只有"当前 spec 自身不可推默认"时才追加并禁用(:301-308)。离屏实测 curve demo 的 kind combo = `[Curve, Histogram, Rolling]` —— FacetGrid、Image 凭空消失。后果:FacetGrid 的 `default_spec` 恒 None → 从 facet demo 语义切走后,**语义组永远回不去 facet**(单向门);而 FacetGrid 在该 schema 上明明可渲染(demo 自己就是)。Image 的消失倒与投影层一致(实测 `ImagePlot` 在带 site 轴的 schema 上被 `DataViewError: image leaves data axis 'site' unreferenced` 拒绝),但 UI 无任何解释。
另:SE7 的契约测试 `tests/test_kind_registry.py:110-116` **同语反复**——每个 `_kinds/*` 的 `admits` 本身就定义为 `default_spec(schema) is not None`,两个声明是一个声明,永不可能漂移,mutation 什么也杀不了。单源本身是对的,但"契约测试守住漂移"这句声称是空的。

### 4.【坏】embed 顶部 demo combo 与真实 kind 脱钩,且点回去"没反应"
离屏实测:语义切到 Histogram 后,顶部 "Plot kind" combo 仍显示 "Curve";此时点顶部 combo 选 "Curve" **什么也不发生**(spec 保持 HISTOGRAM;实测确认)——因为 `examples/pyqt5_embed.py:884` 用 `active.definition.kind is definition.kind` 判重,而 `_definition_after_semantic`(:705)只 `replace(definition, spec=candidate)` 不更新 `kind` 字段。窗口里同时存在两个互相矛盾的 kind 指示器,其中一个还失灵——这是用户"没反应"最直接的复现器。
复现:embed → Semantics kind 选 Histogram → 顶部 combo 点 Curve。

### 5.【坏】kind 往返静默改语义:Curve→Histogram→Curve 回来 x 从 Time 变 Field
离屏实测:回程 `x = AxisRef(point_dimension('field'))` —— default 取 `dims[0]`(embed 的 topology 是 `(field, time)`;notebook 是 `(y, x)`,同病)。叠加第 1 条,回来的图是"signal vs Field,轴标 'Time (mV)'"。default 单源本身正确;缺的是 embed 层"回到原 kind 应恢复上一个同 kind 的 spec"的记忆。文件:`examples/pyqt5_embed.py:775-789`。

### 6.【怪】facet demo 的 x/group 语义域提供必然被拒的选项
实测:facet(site) demo 中 group/x combo 都列出 "Site",选了立刻红字 `facet source cannot also be a cell axis, group, or sample`。互斥只做了 facet_rows/cols 排除 cell 已用轴的方向(`semantics.py:318,368-391`),没做反向。文档 `docs/semantic-edit-2026-08-04.md:9` 声称返回 "admissible domains" —— 与事实不符。SE5 红字可见所以不崩,但"选择域=唯一真相源"的承诺在 facet 上破口。

### 7.【怪】notebook 的 median 演示永远不可见(SE4 修了数据,没修台架)
cell 13 在**同一个 cell 内**依次 `replace_spec(... MEDIAN)` → 紧接 `replace_spec(... group=repeat)`,median 投影只存在于 cell 执行中途,用户看到的最终帧永远是 group=repeat。数据去简并(mean≠median 断言)是真的(cell 3 与 embed `_definitions()` 均实测通过),但"reduction 切换肉眼可见"在 notebook 验收台架上仍然没有一帧可看。

### 8.【怪】SE5/语义应答链零测试;SE6 声称的"demo 非简并断言"不在 pytest 里
grep 全仓:`set_error`、`semanticEdited`、语义编辑→replace→set_description 回路没有任何测试;demo 数据非简并断言只活在 demo 运行时 assert(embed `_definitions()`、notebook cell 3),不在 offscreen 测试套件。SE6 勾选的覆盖面与事实有差。另:`qt_controls.py:317` `set_semantic_values` 全仓无调用 = 死码(密封面板上的死 API)。

### 9.【怪】kind 切换时旧面板吃到新 state,KeyError 被静默吞掉
时序:worker 端 `session.replace_spec` 尾部 `_notify_display`(session.py:1587)先于 future 完成回调到达 → 旧 Curve 面板 `set_values` 拿到缺 `x_display_unit` 的 Histogram state(实测两 schema 参数名集合确有差)→ `set_values` 抛 KeyError → `backends.py:722-727` 把异常塞进被 embed `_dispatch` 丢弃的 future,无声消失。当前无用户可见后果(随后 rebuild 覆盖),但这条吞异常链会掩盖未来的真错。

### 10.【可解释】每次语义编辑语义组全量重建 → 焦点丢失
`qt_controls.py:281-301` 语义签名含 `control.value`,任何被接受的语义编辑必然触发 `_rebuild`(销毁重建全部语义控件)。闪烁被 `setUpdatesEnabled(False)` 压住,信号回环不存在(CHOICE 先 addItem 后 connect,`qt_controls.py:388-413`;setter 全程 QSignalBlocker),但键盘焦点/下拉状态每次丢。值变即重建是可辩护的设计,体验可再优化,不算 bug。

---

## 二、SE1-SE7 逐条勾稽

| 条目 | 结论 | 证据 |
|---|---|---|
| SE1 标签单源 | **实现属实,但权威本身产出歧义标签**(见一.2)| label 只在 `semantics.py`(`_axis_label`/`_kind_label`/reduction 小写);Qt 侧 semantic 值绝不 str(`qt_controls.py:42-55`,`_set_choice` semantic 分支改为 raise `:520-525`);anywidget ESM 纯 frame blitter,无任何标签逻辑(notebook 无语义 GUI,"同权"仅指 `replace_spec` API)|
| SE2 combo 卫生 | **value 去重属实,label 碰撞漏网**(一.2)| `(none)` 显示 ✓;去重单源实际在 `_choice_pairs→_unique_values`,`SemanticField.__post_init__` 里那层是冗余防御(mutation 删掉它测试全绿,因上游已去重)|
| SE3 embed kind 真语义 | **属实**(ada7082 核过:走 `default_spec`+`replace_spec`,失败 set_error,不再跳 demo 页)| 残留:一.1 labels 硬贴、一.4 顶部 combo 脱钩、一.5 往返丢语义 |
| SE4 demo 去简并 | **数据属实**(embed/notebook 均有 mean≠median 运行时断言,实测通过)| 台架缺口:一.7 median 无可见帧 |
| SE5 错误可见性 | **实现属实**(红字 `_error_label` 就地显示,成功后清除)| 零测试(一.8);另:显示参数编辑成功不清语义错字,小残留 |
| SE6 回归守卫 | **半真**。mutation 实测:label→`str(value)` 双文件红(`test_semantics` + `test_semantic_ui` 各 1 failed)✓;admits 过滤断言在 ✓ | 但守卫 schema 无 topology,对真实 demo 的 label 碰撞永远绿(一.2);"demo 非简并"不在 pytest(一.8)|
| SE7 default_spec 单源 | **单源与消费属实**;契约测试**在但同语反复**(一.3)| "灰掉"未按文本实现:是"消失",仅当前 kind 特例禁用(`test_offscreen_semantic_kind_without_default_is_disabled` 只测了这个特例)|

## 三、describe_semantics 与 describe_display 的关系
- **一次往返,无双请求**:`DisplayDescription.semantics` 内嵌 `SemanticDescription`(session.py:155-181, 767-784),Qt 面板一次 `set_description` 同时刷两组;锁为 RLock,`replace_spec` 尾部内嵌 `describe_display`(:1584)无死锁。
- **刷新循环**:replace future accepted → `set_description` → 语义组必然全重建(焦点丢,一.10);display 组仅 schema 变时重建。重建期无假信号发射(核过 connect 顺序与 blocker)。真正的时序脏点是一.9 的静默 KeyError。

## 四、最终裁决建议
**混合:约半数适合 codex 精确清单,但用户"还是不对"的主体是三个需要架构裁决的模糊地带,先裁再修,否则 codex 会再修一轮形状。**

需要用户拍板的架构级问题(正对应一.1/2/3):
1. **label 失效政策**:哪些文本绑定哪个语义角色(x_label 绑 x,value/y 绑投影……),`replace_spec` 时按角色签名失效(仿 `_viewport_signature`);embed 的 `labels=current.labels` 随之删除。不裁这个,语义面永远"改了但看着没改/看着改错"。
2. **孪生轴的域语义**:point 列与 topology 维度是否都该出现在选择域(信息确实不同)——都留则标签必须可区分(如 "Time (ms) · grid"),不留则枚举排除被 topology 覆盖的列。
3. **kind 域的三态**:registry 全集中"可渲染但推不出默认 / 不可渲染 / 可默认"如何呈现(灰掉+原因 vs 消失),以及 FacetGrid 的进入路径(单向门必须解)。

codex 可直接清单化修复:一.4(demo combo 同步)、一.5(per-kind last-spec 记忆,若采纳)、一.6(facet 反向排除一行域计算)、一.7(notebook 拆 cell)、一.8(补 set_error/回路测试 + 非简并 pytest 断言 + 删 `set_semantic_values` 死码)、一.9(dispatch future 错误上报)、文档两处(`semantic-edit-2026-08-04.md` 的 "113 tests" 应为 117;"admissible domains" 措辞)。