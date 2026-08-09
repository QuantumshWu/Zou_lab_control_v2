# Zou_lab_control_v2 agent execution rules

These rules apply to every agent and sub-agent working in this repository.
They constrain implementation method; product and architecture truth remain in
`ARCHITECTURE_DESIGN.md` and `IMPLEMENTATION_PLAN.md`.

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
    proof, then commit and detached-verify before starting another topic.
