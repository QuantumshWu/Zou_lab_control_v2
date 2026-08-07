审计完成。所有场景已在真 embed 窗口(offscreen)驱动完毕,关键根因已定位到具体行号并有运行时证据。以下为最终报告。

---

# zlc_plot 语义面体验审计(embed 全场景实测,2026-08-04)

**总判**:用户"语义面还是不对"不是玄学——embed 里存在一个**一击致死的静默死锁 bug**(#1)和一个**库级非事务 replace_spec**(#2)。用户在语义组里随手选一个不合法选项(而合法选项是极少数,见 #3)后,窗口从此对一切语义编辑无响应、live 冻结、且**没有任何可见报错**。117 绿测试全部绕过了 `_track` 这条真实 GUI 通路。

---

## 一、按严重度排序的问题清单

### 1.【坏·头号】被拒命令的错误回调必然 NameError,静默吞掉后整窗死锁
`examples/pyqt5_embed.py:675-684`:`_track.finished` 的 `report_error` 闭包引用 `except ... as error` 的 `error`;Python 在 except 块结束时删除该名字,而闭包是**之后**才在 Qt 线程异步执行 → `NameError("cannot access free variable 'error'")`。该异常又被 `Qt5PlotWidget.dispatch` 存进无人读取的 completion Future(`src/zlc_plot/backends.py:722-729`;`_dispatch` at pyqt5_embed.py:661-663 丢弃返回值)→ **零痕迹**。
后果链(全部实测):`rejected` 不执行 → `_switching` 永久 True → 语义面板一切编辑(:764 guard)、顶部演示页切换(:880 guard)、live 发布(:947 guard)全部静默死;红字/状态栏均无字。
**复现**:启动 embed → Curve 页语义组 x→"Repeat"(必被投影层拒绝)→ 从此点什么都没反应,live 曲线冻住。dispatch future 实测 `exc=NameError(...)`。
**这就是用户手动验收"没反应"的最短路径。**

### 2.【坏·库级】`replace_spec` 非事务:layout 阶段拒绝时状态已半提交,session 永久中毒
`src/zlc_plot/session.py:1526-1543`:`self._spec = spec` 等十余项突变发生在 `plan = self._resolve_plan()` **之前**,而 `_resolve_plan` 的异常(如 facet 超 64 cell,`layout.py:642-646`)不在 :1545-1574 的 try/rollback 覆盖内。docstring 自称 "Atomically replace"。
**实测**:FacetGrid 页 facet_cols→Repeat 被拒(ValueError)后:`describe_semantics` 已报 `facet_cols=repeat`(被拒 spec 成了语义真相);live presented 14→14 而 published 涨到 38(**渲染永久冻结**);随后连 `set_parameter("title")` 都报内部错 `"visible facet count is outside the rendered grid"`。GUI restore 之后 combo 如实显示半提交值 "Repeat",而 embed 的 `definition.spec` 还是旧值 → 三处真相(session/panel/embed)分叉。
注:投影阶段的拒绝(:1507 之前抛出)是干净的,curve 页实测回滚正确——只有 layout 阶段炸。

### 3.【坏】语义选择域是可行域的巨大超集,"门"没有真的把门
`src/zlc_plot/semantics.py:333,337-345` 把全部轴塞进 x/group choices,但投影层拒绝大多数组合。Curve demo 实测:**group 的 8 个选项只有 "Site" 一个能成功**(其余全报 "curve leaves data axis 'site' unreferenced"),x→Repeat 也被拒("point-row domain unreferenced")。GOAL S 轮宪法要的"语义选择域=唯一真相源+GUI 的门",现状是门形同虚设;**GOAL S4 的验收场景 `group=repeat` 在 embed demo 数据上根本不可能成功**。配合 #1,用户逐个试选项=逐个把窗口打死。

### 4.【坏】语义改了,标签不跟:渲染出 "Time (mV)" 这类焊接谎言
两层钉死:embed `_semantic_edited`(pyqt5_embed.py:789)kind 切换时 `labels=current.labels` 强行携带;`session_policy._retain_parameters`(session_policy.py:76-95)又把 title/x_label 当普通显示参数跨 kind 保留。
**截图级实测**:Curve→Histogram 后渲染标题仍 "Live curve",x 轴变成 **"Time (mV)"**(错误标签+value 单位焊接),y 轴 "Signal" 实为计数;Facet→Curve 后标题仍 "Live FacetGrid" 且 x_label "Time" 而 x 实际是 field;Image 回切后 "Position X (um)" 轴上画的是 image_y 数据。

### 5.【坏】Image kind 往返:图像转置 + 站点 overlay 永久消失
`src/zlc_plot/_kinds/image.py:53-60` default_spec 按 data_axes 顺序给 `x=image_y, y=image_x`,与 demo 原 spec 相反;embed `_definition_after_semantic`(pyqt5_embed.py:698-722)把 ImageFrame 降为 snapshot 后**切回 Image 不恢复**。
**实测截图对比**:基线有 A/B/C 三个站点圈、图像横向充满;Image→Histogram→Image 后三圈消失、图形转置、viewport 还是旧轴的 ±4/±3 → 黑边错位,轴标签照旧撒谎。唯一恢复手段是顶部 combo 换页再换回。

### 6.【坏】SE5 的"就地红字"位于面板最底,默认窗口下不可见
`src/zlc_plot/qt_controls.py:202-208`:error label 排在 Display+Semantics 两组之后。offscreen 默认窗口(939×593)实测:参数滚动区 viewport 高 **69px**,Semantics 组 top=403、error label top=596 → `in_viewport=False`;**连 Semantics 组本身默认都在折叠线以下**。用户实际只可能瞥到窗口底部 status 一行小字。SE5 判"已实现但看不见"。

### 7.【坏】1281-cell 的 1D facet 绕过容量守卫,报内部几何错
`layout.py:641-646` 的 facet_max_cells 检查只在 `shape is not None` 时执行;1D pack 路径绕过 → facet_rows→"point row"(1281)实测报 **"box requires 0 <= top < bottom <= 1"** ——内部不变量泄漏给用户,而不是"超出容量"。

### 8.【坏/怪】embed demo 上 1D→2D facet 完全不可达
`facet_max_cells=64`(layout.py:205-208,presets 最大 8×8);demo 轴 site4/repeat24/field21/time61 → 任意两轴组合最小 4×21=84>64。**facet_cols 全部 5 个选项实测被拒**(且每次触发 #2 的中毒)。GOAL S4 声称 PyQt 示例覆盖"真实 2×2 grid"——embed 交互路径到不了任何 2D;demo 即验收台架的原则(SE4)在 S2 成果上失守。

### 9.【怪】同名双轴:两个 "Field (mV)"、两个 "Time (ms)",无法区分
`semantics.py:184-206` 的 label 权威不含 domain → point coordinate 与 point dimension 同名同标。x/group/samples/facet combos 实测各出现两对重复标签。SE2 守卫(tests/test_semantic_ui.py:22-34)用无 topology schema,天然测不出——合成测试两端同错自洽的又一例。

### 10.【怪】kind 往返不还原 x
Curve(x=time)→Histogram→Curve 回来 x=field(curve.py:46-48 取 `dimensions[0]`)。配合 #4 的标签钉死:图变了、标签没变、用户没动 x。

### 11.【怪】构造级拒绝不回拨 combo(与 host 级不一致)
`_semantic_edited` 的 except 分支(pyqt5_embed.py:808-812)没有 restore → group==x 这类构造错后 combo 一直显示被拒值(实测显示 "Time (ms)" 而 spec 是 Site),直到下一次任意成功编辑才复位;host 级拒绝则有 describe restore。同一类用户错误两种收场。

### 12.【怪】in-flight 期间输入静默丢弃 + 顶部 combo 永久错位
(a) 语义编辑进行中的第二次语义编辑被 :764 guard 吞掉(实测 group→(none) 丢失,无任何提示);(b) 进行中点顶部演示页 combo 被 :880 guard 吞掉且**不回拨选择**——实测 combo 显示 "Image"、页面仍是 Curve,且因 currentIndexChanged 已消费,再点同项无效。

### 13.【怪】每次语义编辑整组摧毁重建控件
`qt_controls.py:281-301` semantic signature 把 `control.value` 计入 → 任何值变化都走 `_rebuild`(销毁全部语义 editors 再造),焦点/下拉状态丢失;display 组的 signature(:255-268)不含 value、走 `_sync_controls`,两组不对称。这也是审计脚本里 QComboBox 悬垂指针的直接来源。

### 14.【可解释·待裁决】SE7 "灰掉 vs 消失" 实测判定
实现语义(`semantics.py:300-308`):`admits` 过滤后**推不出默认的 kind 直接从列表消失**;只有"当前 spec 自身不可推"(authored FacetGrid)才保留+灰掉,守卫测试也只测这一种(test_semantic_ui.py:66-83)。Curve demo 实测 kind 列表=`['Curve','Histogram','Rolling']`,Image/FacetGrid 无踪。**与 GOAL SE7 文字"推不出默认的 kind 在语义组灰掉"不符**。副作用:facet 页 kind→Curve 后 "Facet grid" 从列表消失=**单向门**,语义面无法回到 facet(实测)。机制(disabled_values)存在,只是没按 GOAL 用——要么改文字要么改实现。

---

## 二、任务 B 场景逐项验收

| 场景 | 判定 | 实测描述 |
|---|---|---|
| 1a Curve: kind→Hist→Curve | 【坏】 | spec 真换、面板字段刷新为新 kind 字段集、像素变;但标题/轴标签钉死(渲染出 "Time (mV)"),回程 x 从 time 变 field |
| 1b x 换 repeat | 【坏】 | 被投影层拒绝;原版代码下=无提示+全窗死锁(#1);修复 _track 后红字正确、combo 回拨 |
| 1c group=repeat | 【坏】 | demo 数据上任何非 Site 的 group 全被拒;该验收场景在 embed 上不可能通过 |
| 1d reduction 全枚举 | 【好】* | (_track 修复后)六档全接受、每档像素实测不同(SE4 去简并生效);*原版代码一旦先踩过一次拒绝就全数无响应 |
| 2 FacetGrid: rows 改轴 | 【好】 | repeat(24)/time-pt(61) 正常重画;point row(1281) 报内部几何错(#7) |
| 2 cell 字段改 | 【好】 | cell x→repeat 正常替换重画 |
| 2 一维↔二维 | 【坏】 | 5/5 facet_cols 选项被拒(64 cap),且每次拒绝都使 session 半提交中毒、live 永久冻结(#2) |
| 3 错误红字位置 | 【坏】 | 红字在参数面板最底部,默认窗口 in_viewport=False,必看丢;status bar 有同文一行 |
| 4 live 中语义编辑 | 【好/坏】 | 设计=编辑期间暂停发布、接受后恢复(实测 presented 2→13 恢复推进);但任何一次被拒编辑在原版代码下=live 永久冻结 |
| 5 kind 消失 vs 灰掉 | 【可解释】 | 实现=消失(仅当前不可推 kind 灰掉);与 GOAL SE7 文字冲突,附带单向门效应(#14) |

## 三、附注
- 所有结论均来自真 embed 窗口 offscreen 驱动(`create_window()`+processEvents 泵),PNG/描述符/`describe_semantics` 三路交叉验证;审计对被审仓零修改(#1 的 `_track` 修复仅在驱动脚本内 monkeypatch,用于揭示修复后的应然行为)。
- `notebooks/usage.ipynb` 有未提交修改(+243 行,含 describe_semantics 语义 demo 与 facet-grid cell),未深审,如实记录。
- 测试为何全绿:`tests/test_semantic_ui.py`/`test_semantics.py` 直接驱动 `PlotSession`/`Qt5ParameterPanel`,不经过 embed 的 `_track`/`_dispatch` 异步链;SE2 守卫用无 topology schema 测不出重复标签;SE7 守卫只测"当前 kind 灰掉"分支。
- 修复优先级建议:#1(3 行:`err = error` 落地绑定)→ #2(把 `_resolve_plan()` 挪进 try 或提前到突变前)→ #3(把投影层可行性并进 describe_semantics 的 disabled/过滤)→ #4/#5(标签与 default_spec 轴序单源)。#1+#2 修完后,用户"没反应"的主观感受应当消失;#3 不修则"连环报错"会取而代之。