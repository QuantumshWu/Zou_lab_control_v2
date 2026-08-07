核对完毕。原仓 git status 与开工一致(`M notebooks/usage.ipynb`、`M src/zlc_plot/backends.py`、`?? .browser_check/`、`?? .claude/` 均为开工即有);全部实验在临时副本完成并逐个还原(还原后与原仓逐字节一致,仅换行符伪差,复跑 99 绿)。

---

# 任务1 验收报告:测试地基(P1)与行为回流(P2)

**结论:PASS**(附 3 条小注,不阻验收)

基线:临时副本 `pytest tests -q` → **99 passed, 0 skipped, 4.9s**(与声称一致;Qt offscreen 冒烟真实跑过而非 skip)。

## 1) 测试清单逐项落地(15 项清单 → 21 文件 / 99 测试,全部落地)

| 清单项 | 落地文件(测试数) | 备注 |
|---|---|---|
| zlc_data 不变量 | `tests/test_data_contract.py`(7)+ `tests/test_ingress.py`(3) | 真实 `OwnedSnapshot` 的只读数组、形状/dtype/validity/revision 与 `GridTopology` 显式性 |
| LatestRevisionChannel 并发 | `tests/test_live_channel.py`(3) | 严格单调+coalesce 计数+close 唤醒+4 线程×250 压力(见小注②) |
| npz 往返 | `tests/test_npz_io.py`(3) | 字节级确定性编码往返 `exactly_equals`+缺 manifest 拒+多余成员拒 |
| selectors 性质 | `tests/test_selectors.py`(4) | 200 次随机拖拽永不越界(seeded property 风格)+revision 单调+cancel 还原+几何单权威 |
| fit 参数恢复 + radial 快路径等价 | `tests/test_fit_engine.py`(11) | 8 内建模型合成参数恢复 rtol 2e-3;`test_radial_regular_image_fast_path_matches_coordinate_path`:83 |
| FitProjection authority 优先级 | `tests/test_fit_projection.py`(2) | AREA→X_RANGE→viewport→ALL 全链,逐级断言 sample_count |
| 三 kind 金样 rgba 回归 | `tests/test_plot_session_golden.py`(3)+ `tests/goldens/{curve,histogram,image}.png` | 见下 |
| 四段协议无线程合同 | `tests/test_live_protocol.py`(2) | prepare/commit/abort/finalize 原子性+陈旧拒绝,调用侧单线程 |
| LivePlotController fake session | `tests/test_live_controller.py`(2) | latest-only pump+metrics+stop 取消在途 prepare |
| RasterPlotHost coalesce | `tests/test_raster_host.py`(3) | 同 key 合并 superseded.cancelled()+close 取消队列+press 重定位 |
| 单位往返 | `tests/test_units.py`(1) | selector display/canonical 双读+fit 参数随显示单位(cm/mV 数值断言) |
| layout 纯数值 | `tests/test_layout.py`(28) | 3 DPR×预设逻辑尺寸不变、raster=逻辑×DPR,无渲染 |
| offscreen Qt 冒烟 | `tests/test_qt_widget.py`(1) | QTest 真拖拽→front 出现 area selector |

清单外增量:`test_projection_coverage.py`(3)、`test_histogram_samples.py`(2)、`test_kind_registry.py`(3)、`test_public_surface.py`(4)、`test_notebook_raster.py`(14)。

**金样是真图像回归**:`test_plot_session_golden.py:60-73` 将 `session.rgba()`(固定 shape (357,480,4)、double-render 一致性断言)与 `goldens/*.png` 逐像素比较。容差=每通道 |Δ|≤2,外加">2 像素占比≤0.5%"条款。**字体确定性成立**:包内 `src/zlc_plot/assets/helvetica-light-587ebe5a59211.ttf` 单源,`assets/__init__.py:25-38` addfont+家族名校验,不依赖系统字体。

## 2) P2 行为回流 + public surface 修订

**三项 C0 证伪行为,与 vendored 参照(`Zou_lab_control_v1/zlc_plot/`)对照:**

- **Curve/Image 聚合未引用轴 — 已修,与参照一致**。`src/zlc_plot/data_view.py:1042 _validate_axis_coverage`:size>1 数据轴未显式引用→`DataViewError`;P>1 点行未覆盖→拒;仅 R 允许隐式归约——与参照 `data_view.py:696`("R is the only axis this plot kind may reduce implicitly")、`:1176-1201` 同一政策。curve/image 调用点 `data_view.py:449/526`。测试:`test_projection_coverage.py:26/33/39`(含 image 双普通点坐标需 GridTopology)。
- **Histogram flatten 全数组 — 已修,与参照一致**。显式 samples 契约 `data_view.py:867-899 _sample_positions`:repeat 未消费→拒、点行未消费→拒、数据轴未消费→拒;`HistogramPlot.samples` 默认 `(repeat, point_rows)` 与参照 `specs.py:200-215` 逐字段同。测试:`test_histogram_samples.py`(两条)。
- **Rolling 把 R 当 history — 核心已修,但含一处相对参照的有意扩展(见小注①)**。旧病(R 轴直接当历史轴、无 revision 账本)已根除:现为 session 持有的 per-revision 历史(`data_view.py:766 rolling_sample` 每 revision 一标量),`source_revisions` 精确记账。**分歧点**:target 新增 `rolling_history_samples`(`data_view.py:808-865`)把**初始静态快照的 R 轴种子化为逐 shot 历史**(R=3 → 首帧 3 点,账本 (0,0,0)),而参照 `data_view.py:494-499` 明文"emits exactly one sample for the current immutable source revision"(R 在 revision 内聚合为单点),参照的 `_fit_projection.py:524 _append_rolling_payload` 无任何种子路径。测试 `test_public_surface.py:63-79` 把种子行为写死并给出成文理由("A static snapshot is a complete shot record")。

**Public surface 修订五项——全部采纳:**

| 修订 | 落地 | 测试 |
|---|---|---|
| `HistogramPlot.samples` | `specs.py:192-207`(与参照同构) | `test_histogram_samples.py:37` |
| 删 `RollingPlot.x` | `specs.py:211-223` 仅 group/reduction/labels,无 x(与参照 219-231 同) | kind registry 构造覆盖 |
| `replace_spec` | `session.py:1428`,复用同一 figure | `test_public_surface.py:30`(断言 `_renderer.figure is figure`) |
| `source_revisions` | payload/front/fit 全链(`data_view.py:214...`、`raster.py:240-278` 非降序强制、`session.py:867`) | `test_public_surface.py:42-60` |
| `fit_all_facets` | `session.py` fit(...)→`FacetFitBatchResult` | `test_public_surface.py:82` |

## 3) Mutation 探针(临时副本,逐个还原,还原后复验 99 绿)

| # | 变异 | 结果 |
|---|---|---|
| ① | `_fit_projection.py:82` `_DEFAULT_FIT_SELECTOR_PRIORITY` AREA↔X_RANGE 互换 | **红** ✓:`test_fit_projection.py::test_curve_fit_selection_prefers_area_then_x_range_then_viewport_then_all`(1 failed/98 passed),断言直接抓到 selector_kind 错位 |
| ② | style token:`style.py:701` `line_single` #808080→#7A7A7A(Δ=6 灰阶) | **红** ✓:curve 金样失败(该 token 只影响 curve,histogram/image 不受影响属预期)。追加探针:`layout.py:229` panel margin 110→111(1 逻辑像素)→ **三 kind 金样全红** ✓。敏感性充分 |
| ③ | `_session_live.py` commit 双闸禁用(`data_base_current` 强制 True + 删 `revision <= self.data_revision → None`) | **红** ✓:`test_live_protocol.py:54 test_stale_revision_is_rejected_before_prepare_and_commit` |
| ④ | P4.2 press-superseded:**专项测试存在**(`test_raster_host.py:73 test_press_relocates_to_latest_front_after_live_revision`,用陈旧 identity 发 press,断言成功产出 candidate)。禁用 `raster.py:1441-1455` 重定位块 | **红** ✓:RuntimeError "the current pointer front is no longer session-compatible"(1 failed/98 passed) |

四个探针均单点击杀,无形状剧场;金样对 1px 几何与 6 阶灰度均敏感。前车之鉴中的"简并测试点"未在本任务范围内发现(每个探针只杀死其对应的语义测试,其余 98 项不连坐)。

## 小注(不阻验收)

1. **Rolling 初始种子化偏离 vendored 参照**(上文详述)。不是旧病复发——per-revision 账本与窗口语义与参照一致——但"静态快照 R→种子历史"是参照没有的扩展,GOAL 已自删无法核对是否为其明文裁决。建议用户确认一句:回看存档 rolling 显示逐 shot 历史是否即所要。
2. **`test_live_channel.py:40-63` 并发测试自带理论竞态**:`channel.publish` 在 `counter_lock` 之外(:51),线程 A 取号后被抢占、B 先发号即 RevisionError 自杀。实测 200 次零失败(CPython GIL 下窗口极窄),当前不 flaky,但 free-threaded Python 下会炸;把 publish 挪进锁内即根治。
3. **金样第二容差条款是死断言**:`test_plot_session_golden.py:71` `delta.max()<=2` 成立时 `:72-73` 的"0.5% 像素>2"恒真。实际容差=严格 max≤2(更严方向,无洞),仅建议删死码或改为二选一语义。
4. **既有脏项如实评估**(未动):`src/zlc_plot/backends.py` 未提交 +46 行=ipykernel 专用 QEventLoop 唤醒定时器(notebook asyncio 饿死 workaround),与 P0-P5 验收面无关、无测试覆盖;`notebooks/usage.ipynb` 同轮联动改;本验收 99 绿是对含此脏改的工作树的结论。

**关键文件**:`C:/Users/eadri/Dropbox/WorkCode/Github/zlc_plot/tests/test_plot_session_golden.py`、`tests/test_fit_projection.py`、`tests/test_live_protocol.py`、`tests/test_raster_host.py:73`、`src/zlc_plot/_fit_projection.py:82`、`src/zlc_plot/_session_live.py:202,238-240`、`src/zlc_plot/raster.py:1441-1455`、`src/zlc_plot/data_view.py:808-865,1042`;参照 `Zou_lab_control_v1_claude/Zou_lab_control_v1/zlc_plot/data_view.py:494-499,1176-1201`。
