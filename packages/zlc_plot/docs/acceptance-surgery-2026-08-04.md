# 验收报告 · 任务2:notebook raster 化(P3)与内部手术(P4/P5)

**结论:小修**(P3 实、P4 大体实、P5 的注册表守卫实但"加 kind 只碰 1 模块+1 行"不成立;另有测试缺口与脏树待裁决)。
基线:scratchpad 副本跑全套 **99 passed (3.6s)**;原仓只读,结束时 `git status` 与开工一致(`M notebooks/usage.ipynb`、`M src/zlc_plot/backends.py`、`?? .browser_check/`、`?? .claude/`)。所有 mutation 实验均在临时副本完成并复原。

---

## 1) P3 架构核实 — **PASS(有一处设计改道需知情,两处小瑕疵)**

**形态**:anywidget 单 `Bytes` trait;kernel 端 `RasterPlotHost` 渲染完整 `RasterFront`,浏览器纯 blit。
- anywidget 注册:`src/zlc_plot/notebook.py:129-144`(`RasterWidget(anywidget.AnyWidget)`,`_esm` 内联)。
- 纯 blit:JS 只做 `putImageData`(notebook.py:314-315)+ 输入归一化(`_normalized`/`_send`,L317-334);`test_widget_esm_is_a_pure_frame_blitter_and_input_normalizer` 断言无 `fillText`/overlay/第二 canvas(tests/test_notebook_raster.py:69-102)。
- **JS 侧 246 行,内联在 `notebook.py:147-394` 的 `_WIDGET_ESM` 原始字符串里**,无构建步骤、无外部资产。

**⚠️ 拖拽预览与"评审定的设计"不一致(已改道,非未做)**:评审定的是"客户端用冻结 AxisTransform 本地画";实现先做了(ae46a14 "notebook-local area drag preview"),后被 a52d868/bf68290 **推翻为 kernel 烘焙**——每次 move 走 comm 往返,matplotlib 把 transient 候选烘进 front,浏览器零几何绘制(notebook.py:174-179 注释明说)。冻结语义保留在几何层:press 时捕获 `_gesture_front`/`_gesture_axes`(notebook.py:495-500),move/release 全程用该冻结 transform。延迟靠 JS 30ms 节流(L365)+ kernel move 合并(raster.py `_pointer_coalesce`)+ stale press 迁移到最新 front(raster.py:1441-1455,fe1ee30)兜住。单渲染器一致性上位、预览延迟受 RTT 影响是代价——**属有据可查的设计裁决,建议向用户确认追认,不算缺陷**。

**原子传输 + stale 丢弃**:实。单 `frame_packet` = 4 字节头长 + JSON 头 + RGBA(notebook.py:63-89);kernel 端 `_publish_front` 丢 `sequence <=`(L456),JS 端 `_acceptFrame` 丢 `sequence <`(L237)。
- **瑕疵 A(测试缺口,mutation 实证)**:删掉 kernel 端 stale 序列守卫后 **99 测试仍全绿**——该守卫零测试覆盖。JS 端守卫只有字符串 grep 测试(本质无法用 pytest 验证,可接受但零证明力)。
- **瑕疵 B(死载荷)**:packet 头里每帧序列化 `axes`/`selectors`(notebook.py:81-86),JS `_decodeFrame` 完全不读(实测 ESM 无 `axes`/`selector` 字样)——是客户端预览时代的残留,体积无害但语义上是死接口,连同 `test_front_selector_state_json_preserves_display_geometry_for_browser_preview`(名字仍说 "for browser preview")宜清理或改名。

**selector 在 axes 空间画(58b402e)**:实。几何以 painted(axes-data)空间进 renderer,pulse 源单位系数只乘一次;`test_pulse_selectors_paint_in_source_units_and_fit_catalogue_is_empty`(tests/test_notebook_raster.py:198)是真数值测试,且在探针 D 中确实翻红(对链路敏感)。

**单一交互链:成立**。全仓 `mpl_connect` 零命中;Qt(backends.py:1011)与 notebook 同走 `host._pointer_event` → `session._raster_pointer_event`(session.py:3086)→ 合成 MouseEvent → 唯一一套 `_on_button_press/_on_motion/...`(只定义在 `_session_gesture.py`)。gesture 代码里唯一 "raster" 字样是色标拖拽的节流 lane 名(_session_gesture.py:634),非双轨。

---

## 2) P4 逐项 — **大体 PASS,防御收缩幅度远低于口径**

| 项 | 实况 | 证据 |
|---|---|---|
| session 四分 | **真代码搬迁,非 shim**:gesture 874 / fit 744 / live 310 / state 149 行,方法只定义一处;session.py 5,187→3,287 行。但是 **mixin 拆分**(`PlotSession(FitSessionMixin, LiveSessionMixin, GestureSessionMixin)`,session.py:310),gesture mixin 引用 44 个 `self._*` 私有属性——是文件级解剖,不是边界级解耦,无法独立测试 | session.py:37-39,310 |
| 回滚层删除后的恢复路径 | 在:presentation 失败→`on_abort()`,`on_abort` 再失败→`redraw_surface()` 整帧重画(raster.py:836-843);`redraw_surface` = 全量 `renderer.draw()`(session.py:688-696)。d443dd1 净删 23 行 | |
| 防御收缩 | **幅度很小**:isinstance 全包 624→603(-3%),`raise TypeError/ValueError` 717→682(-5%)。dc2d595 实删 234 行冗余记录校验,但对照原审查"65-75% 可删"的口径,**基本没收缩** | 实测对比 746c7f3 vs HEAD |
| blit 两档降级 | **未做两档——直接整个删了**:7cced8e 从 rendering.py 删 618 行,六种背景缓存全灭,`copy_from_bbox/restore_region` 零命中;每次呈现=完整帧。降级机制换成 cadence lane(selector 几何逐 move、昂贵色标重映射按 `raster_preview_interval_ms` 节流,_session_gesture.py:633-641)。性能实测入档(2x2/DPR1 artist render 22ms p50,docs/performance.md)——**方案不同但自洽且有数** | |
| 拖拽几何双权威 | 合一:`_gesture_engine.py`(191 行,backend-neutral,c6c9948/90d0390)。**mutation 实证**:反转 pan 方向→`test_backend_neutral_gesture_geometry_has_one_authority` 翻红,且该测试是真数值断言(tests/test_selectors.py:98-124) | |
| radial 快路径 | 独立模块 `_fit_radial.py`(641 行)+ 能力位 `regular_image_radial`(声明 fit.py:1703,门 fit.py:731)。**mutation 实证**:摘掉能力位→`test_radial_regular_image_fast_path_matches_coordinate_path` 翻红(且是快/慢路径等价测试,正确的测法) | |
| `_inverse_unit_symbol` 下沉 | 完成:zlc_plot 零命中,语义变成 `Unit.inverse_dimension` 落在 zlc_data/units.py:28(52c2183) | |

另:`vendor/zlc_data` 与 `src/zlc_data` 实测**字节等价(仅 CRLF 差异)**,vendor/README.md 说明了 pin 的原因与删除条件——原审查的"影子 import 分叉"在本仓内已收敛为有据可查的 pin。

---

## 3) P5 — **注册表守卫实,核心承诺不成立,尺寸审计半诚实**

**穷尽契约(29f4695)是真守卫**,mutation 实证两次:
- 探针 A:给 `PlotKind` 加 `SCATTER` 不注册 handler → 2 测试翻红(`test_kind_registry_is_closed_and_complete` 等)。
- 探针 B:curve handler 的 `fit_target` 改成 `"bogus"` → 6 测试翻红(含下游行为测试)。
- 探针 D(链路):废掉拖拽 republish 信号 → notebook + Qt 侧 4 测试翻红。合成 kind 缺件必红,**成立**。

**"加一个 kind 只碰 1 个模块+1 行清单":不成立**。注册表(`_kinds/`,6 kind × 21-51 行)只接管了 **3 个消费点**(rendering.py:988 render 入口、_fit_projection.py:373 fit_target、:456 build_payload)。kind-isinstance 分派 **151 → 143 处**(仅 -8):rendering.py 51、session.py 39、_fit_projection.py 24、specs.py 16、data_view.py 7……典型如 `_effective_labels` 仍是六 kind 全展开 if-链 + 尾部 TypeError(rendering.py:1001-1102)。加新 kind 实际要动 rendering/session/_fit_projection 的全部链。a93b905 审计里"registry removes dispatch duplication"**言过其实**——它加了封闭清单守卫,没消除分派重复。

**P5.3 尺寸审计(a93b905)**:记录 26,077 行；本轮 V1--V3 后实测 `src/zlc_plot` = 25,012 行，vs 目标 ≤15k——**目标未达,审计如实承认**并把去路指向拆包(与用户已拍板的多仓路线一致,合理)。四大保留责任的逐项说明成立。但**净账没算清**:修复轮起点 25,479 → 25,854(**净增 375 行**)是历史账；本次命名空间迁移只搬移文件，不把该历史增减重新归因。registry 仍是封闭清单守卫而非分派收敛，143 处 kind-isinstance 分派仍在；防御收缩实测约 3--5%，不是此前暗示的 65--75%。

**测试质量旁证**:golden 是紧容差真 golden(逐像素 |Δ|≤2 且 >2 的像素 ≤0.5%,tests/test_plot_session_golden.py:70-73);raster 交互测试有真数值断言与非空帧断言(test_notebook_raster.py:479-516)。全套 1,869 行测试对 25.9k 行源码偏薄,但抽查未见简并测试;唯一"形状剧场"是 ESM 字符串 grep 两条(test_notebook_raster.py:57-102),属 JS 无法 pytest 化的合理妥协。

---

## 4) 脏工作树裁决

1. **`src/zlc_plot/backends.py`(+46 行,未提交)**:`_install_ipykernel_wake_timer`——50ms QTimer 定期 quit ipykernel 专属 QEventLoop,修"下一 cell 挂起 / 顶层 await 永不恢复"的真饥饿问题;有详细 docstring,只在专属 loop 存在时安装。是**实打实的功能修复,不是残渣**。裁决:**应单独 commit**(按用户铁律"每主题完成即 commit");无自动测试可以接受(依赖真 ipykernel),但 commit message 应写明手工验证方式。
2. **`notebooks/usage.ipynb`(9±9 行)**:纯重执行噪声——全部是 widget `model_id` 翻新,无像素/文本输出变化。裁决:**丢弃(checkout)**,不值一个 commit;若与上条一起重跑过 notebook 验证,也可并入该 commit,二选一,勿单独成 commit。
3. **`.browser_check/`(11.5MB,单个已执行 usage.ipynb 副本)**:浏览器验证的临时产物。裁决:**加入 `.gitignore` 并删除目录**(现 .gitignore 无此条目)。
4. `.claude/`(untracked,开工时已存在):本地会话配置,建议一并 gitignore,不提交。

---

## 汇总:需要"小修"的清单(按优先级)

1. 向用户确认追认 P3 拖拽预览从"客户端本地画"改道为"kernel 烘焙"的设计裁决(有实现、有文档、有测试,但与评审定案不同)。
2. commit backends.py 的 ipykernel wake timer;丢弃 usage.ipynb 噪声;gitignore `.browser_check/`。
3. 补 kernel 端 stale-front 序列守卫的测试(探针 C 实证零覆盖)。
4. 清理 frame packet 中浏览器不消费的 `axes`/`selectors` 死载荷及对应测试名。
5. 修正 a93b905 审计措辞:registry 是封闭清单守卫而非分派去重(143 处 kind-isinstance 仍在);如实补记"防御收缩仅 ~3-5%"与净增 375 行的账。
