# Zou_lab_control_v2 全项目只读审计

状态：审计与用户裁决完成；实施Goal已就绪，代码实施尚未开始
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
开始日期：2026-08-16
代码修改：禁止
硬件操作：禁止

## 目标

对当前 HEAD 的一方代码、测试和文档做可追溯的全量审查。每个 production 文件、类、函数和测试最终都要得到以下结论之一：

- `PASS`：存在必要、owner 和职责准确；
- `PASS WITH DEBT`：应保留，但有明确局部债务；
- `MOVE`：必要，但位于错误层级；
- `MERGE`：与另一实现维护同一事实；
- `DELETE`：没有真实 production 消费者，或只被测试/历史兼容维持；
- `REDESIGN`：当前契约本身导致正确性、同步或性能问题；
- `USER DECISION`：存在真实产品语义取舍，必须由用户裁决。

审查维度包括必要性、package/layer归属、职责、唯一真相源、generation/revision、线程和生命周期、数据契约、性能、重复实现、死代码、历史残余、测试有效性、文档与实现矛盾。

## 权威规则

本审计不默认相信现有文档。

1. 用户在当前审计中的明确裁决最高。
2. 当前代码、Git历史、运行artifact和隔离探针是现状证据，不自动成为正确设计。
3. `ARCHITECTURE_DESIGN.md`、`IMPLEMENTATION_PLAN.md`、`AGENTS.md`、package README/contract/GOAL均属于待审查材料。
4. 遇到矛盾时，记录事实、方案、影响和审计建议，不由执行者擅自作最终产品裁决。

这条规则明确取代旧文档中“执行者遇到冲突自行选择并同步文档”的说法。

## 范围

包含：

- 当前 HEAD 的全部一方 production Python；
- 全部 tests/support；
- launchers、package metadata与一方配置；
- 根设计/计划及package文档；
- 与问题相关的Git历史和已有workspace artifact，只作为证据读取。

排除：

- generated cache；
- 第三方runtime/vendor二进制本体；
- 历史分支和旧仓的全面审计；
- 真硬件操作。

## 方法

审计并行执行两条线：

1. 全量清册：逐文件、逐符号、逐测试登记消费者、owner、状态和证据。
2. 端到端链路：沿真实数据流审查跨层不变量，避免逐文件检查漏掉系统性问题。

重点链路：

- `OwnedSnapshot -> SignalPublication/SignalFront -> plot projection -> fit/overlay/selector -> RasterFront -> Qt`；
- `LogicNodeDescriptor -> NodeHost -> live publication -> preview -> terminal artifact`；
- `Pulse authoring -> compile -> sequencer -> camera cycle -> shot/repeat/data point/coverage`；
- `SLM target -> solver -> science phase -> adapter -> qCMOS estimator -> feedback controller -> validation/artifact`。

## 报告索引

- [01-inventory.md](01-inventory.md)：基线、规模、依赖、复杂度热点和第一轮全局发现。
- [02-plot-fit-overlay-selector.md](02-plot-fit-overlay-selector.md)：Plot/fit/overlay/selector深审，完成。
- [03-runtime-measurement-task-summary.md](03-runtime-measurement-task-summary.md)：Runtime/Measurement/Task综合结论。
- [03a-measurement-live-publication.md](03a-measurement-live-publication.md)：Measurement live、coverage、partial terminal与性能。
- [03b-task-preview-contract.md](03b-task-preview-contract.md)：Task progress/preview/artifact与七个Logic Node逐一审查。
- [03c-runtime-contract-prune.md](03c-runtime-contract-prune.md)：Runtime真实消费者图与历史框架删减。
- [04-pulse-camera-summary.md](04-pulse-camera-summary.md)：Pulse/Camera/same-shot综合结论。
- [04a-pulse-api-semantics.md](04a-pulse-api-semantics.md)：Pulse model/compiler/device/wire逐符号审查。
- [04b-camera-same-shot-contract.md](04b-camera-same-shot-contract.md)：Camera物理cycle、adapter与real/virtual差异。
- [04c-measurement-pulse-calls.md](04c-measurement-pulse-calls.md)：六个节点的pulse调用与数量映射。
- [05-slm-summary.md](05-slm-summary.md)：SLM target、solver、device、Editor与Feedback综合结论。
- [05a-slm-truth-context.md](05a-slm-truth-context.md)：SLM target、pupil、wavefront、device command与Feedback上下文真相链。
- [05b-slm-solver-feedback-algorithm.md](05b-slm-solver-feedback-algorithm.md)：Sparse WGS、dense MRAF、荧光估计、控制律、噪声与validation。
- [05c-slm-device-editor.md](05c-slm-device-editor.md)：X15213/Virtual adapter、LUT/correction、Editor状态与真机验收边界。
- [06-symbol-coverage-index.md](06-symbol-coverage-index.md)：459个tracked Python与非Python资产的完整报告归属索引。
- [06a-data-durable.md](06a-data-durable.md)：`zlc_data`/`zlc_durable`逐符号、selection/projection/archive与并发写入。
- [06b-ui-workbench.md](06b-ui-workbench.md)：`zlc_ui`/`zlc_workbench`逐符号、Qt lifecycle、presenter/state/parser与device controls。
- [06c-atom-remaining.md](06c-atom-remaining.md)：`zlc_atom`剩余foundation/install/Calibration/Occupancy/Scan/Temperature/Simulation逐符号审查。
- [06d-runtime-plot-remaining.md](06d-runtime-plot-remaining.md)：`zlc_runtime`/`zlc_plot`剩余逐符号、standalone/notebook/dead pipeline与线程owner。
- [06e-root-bootstrap-packaging.md](06e-root-bootstrap-packaging.md)：根bootstrap、部署metadata、依赖、launchers、工具与测试引导。
- [06f-fpga-nonpython.md](06f-fpga-nonpython.md)：全部RTL/VH/Tcl/XDC/testbench/build与launcher的逐符号硬件审查。
- [06g-test-evidence-architecture.md](06g-test-evidence-architecture.md)：全树166个test文件、examples/notebooks/support与证据等级审查。
- [06h-pulse-remaining-python.md](06h-pulse-remaining-python.md)：Pulse remote/codec/manifest/engine model/transport剩余Python逐符号与网络所有权审查。
- [07-doc-code-test-conflicts.md](07-doc-code-test-conflicts.md)：文档角色、历史残余、代码/测试冲突与目标文档树。
- [08-target-architecture-roadmap.md](08-target-architecture-roadmap.md)：推荐目标架构、不变量、分层修复顺序与测试策略。
- [09-independent-crosscheck.md](09-independent-crosscheck.md)：独立交叉复核、报告矛盾、证据等级、scope缺口与priority纠偏。
- [10-final-audit-summary.md](10-final-audit-summary.md)：最终风险结论、保留/重设计/删除、操作建议与修复顺序。
- [DECISIONS-USER-GUIDE.md](DECISIONS-USER-GUIDE.md)：面向实验负责人的完整裁决说明，逐项解释名词、现状、前因后果和方案代价。
- [USER-DECISIONS-2026-08-17.md](USER-DECISIONS-2026-08-17.md)：本轮用户最终产品裁决与实施解释。
- [../ZLC_V2_IMPLEMENTATION_GOAL.md](../ZLC_V2_IMPLEMENTATION_GOAL.md)：用户稍后可直接提交执行的自包含实施Goal。
- [DECISIONS-PRIORITY.md](DECISIONS-PRIORITY.md)：把完整决策账本压成8个可按推荐整体采纳的顶层gate。
- [DECISIONS.md](DECISIONS.md)：细项裁决traceability账本。

所有结论均写入上述独立报告，没有向旧设计文档尾部追加。

## 进度

| 阶段 | 状态 |
|---|---|
| 基线、全量清册、依赖地图 | 完成 |
| Snapshot/plot/fit/overlay/selector | 完成 |
| Measurement/Task/signal/preview | 完成 |
| Pulse/sequencer/camera/same-shot | 完成 |
| SLM solver/device/feedback | 完成 |
| 剩余逐包符号和测试 | 完成 |
| 文档矛盾矩阵 | 完成 |
| 总结与目标架构选项 | 用户裁决完成；实施Goal就绪 |
