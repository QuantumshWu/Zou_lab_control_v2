# Zou_lab_control_v2 agent execution rules

These rules apply to every agent and sub-agent working in this repository.
They constrain implementation method; product and architecture truth remain in
`ARCHITECTURE_DESIGN.md` and `IMPLEMENTATION_PLAN.md`.

## 最高实施约束（用户裁决，不得重新解释）

- **只有整体骨架保持稳定。** 稳定范围仅包括 plugin discovery、
  descriptor/contract、NodeHost lifecycle、session/device ownership 和公共
  signal/plot capability。
- **骨架上的一切功能都取最简实现。** 每个 Logic Node、device plugin 和
  Workbench feature 只在自己的现有 owner 内写完成该业务所必需的直接逻辑；
  Workbench 只做基本组合、状态投影和接线，不承载 plugin 科学逻辑。
- **Atom foundation 与 concrete plugin 分开。** `zlc_atom` 的顶层基础模块、
  common contract、install/runtime glue 和 `nodes/_framework` 不得依赖 Qt、
  `zlc_plot` 或 `zlc_ui`。具体 `nodes/<plugin>`、`devices/<plugin>` 可以在自己
  的目录内声明并实现该插件独有的 plot/UI，并只调用公共 API；这种依赖不得反向
  进入 foundation，也不得被提升成 Workbench 通用框架。
- **默认删，不默认抽象。** 单消费者 helper/framework、plugin-specific
  registry/coordinator/transaction/adapter/DTO、平行 lifecycle/state 和防御型
  代码应直接删除。不得为了“以后可能复用”、兼容旧测试或防范假想误用而保留。
- **未经用户明确批准，不增加任何 production 文件、通用类或架构层。** 能在
  现有模块用普通函数和直线流程完成，就必须采用该方案。若现有设计文字诱导出
  更复杂实现，先把两份权威文档改回最简单的产品流程，不能照着过度设计实施。
- **每次设计和修复前先读权威。** 每个 agent/sub-agent 在修改前必须先读取
  `ARCHITECTURE_DESIGN.md` 和 `IMPLEMENTATION_PLAN.md` 的相关完整章节；所有
  分派任务必须原样携带本节约束。对话摘要、旧测试和既有复杂代码都不能覆盖它。

以上约束适用于本仓库此后的每一次任务和上下文恢复，除非用户本人明确修改。

## Read the v2 authority before designing

- For every defect, design conflict, performance problem, or implementation
  choice, first search both `ARCHITECTURE_DESIGN.md` and
  `IMPLEMENTATION_PLAN.md`, then read the complete relevant sections before
  proposing or editing code.
- If those documents already specify the solution, implement that solution
  inside the existing architecture. Do not replace it with a new abstraction,
  a v1-shaped implementation, or an agent-invented framework.
- If the two documents are silent, incomplete, or contradictory, report the
  exact gap before editing and choose the smallest solution in an existing
  owner. Reading v1 never overrides the v2 authority unless the user explicitly
  asks for a particular v1 behavior comparison.

## Simplicity is the default

1. Do not add a file unless the user explicitly requests that file, or the user
   first approves a concrete explanation of why no existing owner can contain
   the change. This `AGENTS.md` is the one explicitly requested exception.
2. Only the shared skeleton is architectural: plugin discovery,
   descriptor/contracts, NodeHost lifecycle, session/device ownership, and the
   common signal/plot capabilities. Every Logic Node, device plugin, and
   Workbench feature must be the smallest implementation on that skeleton.
   Never promote one plugin's needs into shared infrastructure.
3. Workbench is a thin composition layer for basic logic and wiring. It must
   not own plugin science or grow plugin-specific registries, coordinators,
   transactions, adapters, report frameworks, or parallel lifecycle state.
   Delete existing single-consumer abstractions instead of wrapping or
   preserving them.
4. Prefer the smallest change inside the existing architecture. Reuse the
   current owner, public API, asynchronous path, lifecycle, and data model.
5. Do not add a class, wrapper, DTO, enum, coordinator, manager, transaction,
   registry, provenance marker, authority token, sealed-plan mechanism, retry
   framework, compatibility layer, or test-only production seam unless the user
   explicitly approves it.
6. Do not write defensive code for hypothetical misuse. Enforce only a real
   product, physical, persistence, concurrency, or public-contract invariant
   demonstrated by an existing consumer or a reproducible failure.
7. Tests must exercise the production path. Do not create a second production
   abstraction merely to make a test convenient, and do not turn every edge
   case into a new guard framework.

## Mandatory stop conditions

8. Before implementation, state the root cause and why the existing owner can
   or cannot fix it. For performance work, profile the real human UI path first.
9. Stop and report before editing if the proposed cut would:
   - add any unrequested file;
   - add any new production class;
   - modify more than 8 files; or
   - add more than roughly 300 net production lines.
10. If a simple change starts requiring lifecycle machinery or parallel state,
   discard that direction and re-derive the solution from the existing path.

## Repository authority and verification

11. The only v1 reference is
   `C:\Users\eadri\Dropbox\WorkCode\Github\Zou_lab_control_v1_claude\Zou_lab_control_v1`.
   Use it only for behavior explicitly requested as a v1 reference.
12. Every Python verification process must first import
    `zou_lab_control_v2` and print the root and tested package `__file__` paths.
13. GUI acceptance uses the formal launcher/composition and real Qt or desktop
    button interaction. Direct presenter calls do not prove the human flow.
14. Keep one topic in flight, stage only its exact files, run the narrow red/green
    proof, then commit before starting another topic. A very small, local change
    that introduces no new behavior boundary runs no tests at all. A substantive
    defect fix runs only the directly relevant red/green tests. Package-wide,
    repository-wide, full-tree, and detached full suites are reserved for a
    genuinely major phase boundary or the end of the active Goal; never run them
    reflexively after a small edit.
15. After every context compaction, automatic continuation, or resumed turn,
    read the persistent Checkpoint before selecting work. An item recorded there
    as completed, settled, or "do not revisit" is closed and must not be
    re-investigated, redesigned, reimplemented, or retested unless the user
    explicitly reopens it or new current-state evidence directly contradicts it.
    Conversation-summary uncertainty is never such evidence.
16. Immediately after completing a step, write its exact decision, evidence,
    commit, and next unfinished action into the Checkpoint. Do not leave the
    status only in chat or memory; that is what causes completed work to be
    repeated after compaction.
17. A milestone cannot be marked complete or committed until a post-milestone
    residual sweep compares its start and candidate trees and closes all of
    the following: production consumer graph; obsolete names/APIs/aliases and
    duplicate owners; production/test/docs LOC by category with every material
    increase explained; tests classified KEEP/MERGE/DELETE with duplicate,
    white-box, self-test, and obsolete-behaviour tests removed; active docs
    reconciled to code; and every intentionally deferred item written into the
    Checkpoint. Passing tests alone never satisfies this gate. If later evidence
    reopens a completed milestone, finish this sweep and any resulting cleanup
    before starting the next milestone.
18. The residual sweep happens only after every implementation cut is merged and
    the candidate tree is frozen. Per-agent handoffs, focused green runs, no-new-
    class counts, and test-function counts are inputs, never substitutes for this
    final audit. The frozen-tree audit must rank positive-net production and test
    files, enumerate added definitions and stored state, count consumers of new
    private helpers/adapters, and measure test bodies/fixtures as well as test
    function counts. Every material positive-net file is KEEP, MERGE, DELETE, or
    explicitly user-deferred with exact reasoning. A known safe deletion may not
    be deferred while the milestone is marked sweep-complete.
