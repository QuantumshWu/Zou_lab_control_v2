# LOC report — 2026-08-03

The current `src/zlc_runtime` tree contains 9,567 physical Python lines,
including module documentation and contract validation.  The selected
read-only reference files total 9,379 lines under the same physical-line
counting method.

| Area | Current lines | Reference / note |
| --- | ---: | --- |
| Streams | 2,056 | direct migration |
| Dataset materialization | 1,920 | direct migration |
| Signal plane split (`plane`, `front`, `values`, `registry`, `processor_lane`) | 2,089 | 2,418-line plane reduced by removing association and preemption machinery |
| Projection-only event layer | 655 | 880-line reference reduced by removing association wrappers |
| Node host | 700 | 693-line hosted-run skeleton, with domain binding removed |
| Presentation scheduler | 661 | rebuilt from the presentation survey, including owed-beat fairness |
| Window runtime helpers | 87 | dependency-free compute/export seam |
| Remaining ownership and contract helpers | 1,399 | live port, resources arbiter, outputs, mailbox, cleanup, preview, cancellation, facade, and small modules |

The result is slightly above the 8–9k target because the package keeps the
full streams/dataset migrations, adds the standalone presentation scheduler
and host seams, and retains explicit public-boundary validation and module
documentation.  The removed association/preemption/event-derived surfaces are
not carried as compatibility shims.
