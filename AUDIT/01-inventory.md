# 01 — 基线、清册与全局风险

状态：第一阶段完成。本文记录现状证据，不代表最终删除或重构裁决。

## 1. 基线

- HEAD：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
- 开工工作树：clean
- tracked Python：459个，AST均可解析
- Markdown：81份，约9,737行、719,363字符
- 测试函数：约1,346个

核心package的source/test规模：

| Package | Source Python | Source LOC | Test files | Test LOC | Test functions |
|---|---:|---:|---:|---:|---:|
| `zlc_data` | 15 | 3,747 | 9 | 1,792 | 64 |
| `zlc_durable` | 5 | 503 | 3 | 385 | 20 |
| `zlc_runtime` | 17 | 11,159 | 14 | 7,148 | 164 |
| `zlc_plot` | 52 | 36,055 | 55含support | 10,286 | 271 |
| `zlc_ui` | 56 | 18,188 | 15 | 4,716 | 93 |
| `zlc_pulse` | 22 | 9,016 | 14 | 3,626 | 139 |
| `zlc_atom` | 76 | 21,955 | 37 | 14,021 | 270 |
| `zlc_workbench` | 30 | 15,786 | 27 | 14,271 | 325 |

## 2. 复杂度热点

| 实体 | 规模 | 初步风险 |
|---|---:|---|
| `ConsolePresenter` | 约4,408行、150 methods | panel、logic、Task、overlay、selection、layout、save混合 |
| `PlotSession` | 约3,701行、144 methods | projection、display、fit、selector、viewport、renderer、lifecycle混合 |
| `MatplotlibRenderer` | 约3,982行、117 methods | 所有plot kind、artist、overlay、selector、fit、classifier混合 |
| `PulseEditorPresenter` | 约2,407行、98 methods | authoring、compile、board state、scan、preview混合 |
| `RasterPlotHost` | 约1,826行、89 methods | worker queue与PlotSession大面积转发 |
| `FitProjection` | 约1,652行、69 methods | 名称是fit，实际承担通用data/view/payload projection |
| `SignalDataPlane` | 约1,429行、47 methods | registry、publication、processor、lineage、front、cleanup混合 |
| `SelectionBridge` | 约1,117行、37 methods | selector、ROI、fit-derived outputs、processor lifecycle混合 |

最大单函数包括439行的`solve_phase`、366行的`ConsolePresenter.update_panel_state`、312行的`SlmFeedbackTask.execute`和288行的renderer `_update_facets`。

大文件不是自动错误；这些实体进入后续逐方法审查，必须证明其状态确实需要共同生命周期，否则列为拆责或删除候选。

## 3. Package依赖事实

声明方向大体是：

```text
zlc_data / zlc_durable
  -> zlc_runtime / zlc_plot
  -> zlc_atom concrete plugins
  -> zlc_workbench composition
```

当前存在明确反向环：

```text
zlc_atom.nodes.calibration.task
  -> zlc_workbench.panel_save / panel_state / task_console.build_panel_host
zlc_workbench
  -> zlc_atom
```

`zlc_atom/pyproject.toml`没有声明`zlc-workbench`，但`test_declared_dependencies.py`的硬编码distribution映射也没有`zlc_workbench`，所以名为“每个import均已声明”的测试会漏掉这条反向依赖并错误通过。

Calibration用Workbench的`PanelState/PanelFrozenData`和Panel Save生成sample图，意味着科学plugin依赖TaskConsole panel实现。该能力是否应成为`zlc_plot`公共保存入口，或由Calibration直接拥有最小保存流程，后续单独裁决。

## 4. 明确重复文件与历史多仓残余

当前存在逐字节相同的文件：

- `zlc_plot/docs/fit-numeric-contract.md`
- `zlc_runtime/docs/fit-numeric-contract.md`
- 两边对应的`test_cross_repo_contract.py`
- `zlc_ui/docs/survey-workbench-ui-2026-08-02.md`
- `zlc_workbench/docs/survey-workbench-2026-08-02.md`

两份cross-repo测试用SHA固定两份文档副本，并要求同步更新“两个repo”；当前已经是一个monorepo。这是高置信历史残余。

package contract/README虽被根文档声明为inactive，测试仍读取其中的public allow-list并强制代码匹配。因此旧文档仍通过测试成为事实权威，与根文档声明冲突。

## 5. 测试维持的production seam候选

以下实体当前没有一方production消费者，主要或完全由测试维持：

- `zlc_data.ValuePayloadContract`
- `zlc_data.save_npz/load_npz`独立单dataset入口；真实产品只复用其manifest helper写figure archive
- `zlc_runtime.LiveDatasetPort`
- `zlc_runtime._ExactDeltaLivePort`
- `ExactDatasetPreviewSpec`
- `DatasetBuilder` / `MonitorDataset`大部分路径
- `NodeExecutionContext.open_live_dataset/open_exact_dataset/start_and_wait`
- `zlc_atom.CameraCaptureSpec`
- `zlc_pulse.engine_model`中的多组RTL/bus mirror helper

它们不能只凭“当前无调用”立即删除：后续会区分真正公共库能力、测试资产错放production、以及已被新路径取代的历史框架。

测试形态初筛显示：

- `zlc_atom`测试中约195处monkeypatch、534处private attribute访问；
- `zlc_workbench`约278处private attribute访问、34处sleep；
- `zlc_plot`约284处private attribute访问；
- 多个package test直接解析源码或冻结文档/public allow-list。

这解释了“测试很多但人类路径仍反复出错”的风险：相当部分守卫保护实现形状，而非完整产品链。

## 6. 第一轮高置信问题

### INV-001 — `OwnedSnapshot`存在重复identity字段

`OwnedSnapshot.ref`重复保存`DataBlock`已经持有的block id、revision和schema fingerprint。Runtime另外拥有`SignalPublication.EventRef`。这些字段可能分别服务数据内容、来源run和causal publication，但当前plane不校验它们的关系。后续沿完整front链判断哪些是必要正交身份，哪些是重复真相。

### INV-002 — `unique_path()`不能保证并发唯一

它先检查路径不存在，再由调用方稍后写入。两个并发Task可取得同一路径；后续`os.replace`会让其中一个覆盖另一个。当前测试只覆盖串行调用，却把“不覆盖”写成承诺。

### INV-003 — 新Logic Node的live/preview不是框架契约

Descriptor允许完全不声明preview，Host不要求Measurement/Task在运行中发布，通用discovery测试也不检查实时可观察性。Camera、Calibration、Scan各自实现live slot，而Runtime另有一套无人使用的live dataset框架。

### INV-004 — finite live数据会制造未来事实

finite Camera预分配完整geometry，future cells写零但snapshot validity仍全真；只在coverage里标未写。Occupancy忽略source coverage、分类整个数组，再把自己的coverage标成完成，能把尚未拍摄的frame发布为empty事实。

### INV-005 — 累计live路径是O(N²)

Camera每次freeze重新stack全部已采cycle并重建blank；Scan每point复制完整预分配数组；FollowTap携带完整累计snapshot；Occupancy每次重新处理全部累计数据。N次更新重复复制/分析约`1+2+...+N`。

### INV-006 — Output vocabulary维护多份平行真相

同一输出同时出现在plugin `DatasetOutputDeclaration`、descriptor `OutputSpec`、node declarations、live slot mapping和preview output name中。Workbench再做一次类型转换；缺少构造时一致性校验。

### INV-007 — Node role与执行mode是两套判断

Descriptor声明Measurement/Task/Processor；NodeHost却靠对象是否存在`source_signal/input_signal(s)`属性推断worker或processor。新节点漏字段或误用同名字段即可走错执行路径。

### INV-008 — 文档权威规则与当前用户裁决冲突

旧规则要求执行者遇到矛盾时自行决定；当前用户明确要求记录矛盾、设计选项和影响，最终由用户裁决。本审计以当前用户裁决为准。

## 7. 下一步

第一轮只建立候选和证据，不做删除。下一阶段从`OwnedSnapshot -> SignalFront -> PlotSession -> RasterFront -> Qt`逐段审查，先解决用户已点名的data/fit/overlay/selector与性能问题。
