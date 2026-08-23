# zlc_runtime

`zlc_runtime` owns ZLC node execution, signal publication, causal lineage,
future-publication following, coherent fronts, presentation scheduling, and
selection-derived signals. It is Qt-free and contains no plugin physics.

Logic Nodes submit only new immutable event chunks. `SignalDataPlane` assigns
their generation/revision identity and, for finite data, places them into one
canonical run dataset using the declared schema and `(repeat, point)` origin.
Exact scientific processors consume the immutable event; every display
consumer sees the same publication's canonical full geometry through
`current_dataset()`, with unwritten cells invalid. Ordinary Monitor outputs,
including Processor outputs, have no finite canonical extent and retain only
their latest event. `index_by_source` declares only that a display-derived
output is capable of history. Runtime exposes a byte-bounded ordinary Dataset
over a neutral `primary-index` only while a consumer holds a window lease;
retention begins at the current event, uses the largest active window, and is
dropped with the last lease. Missing computations inside that interval are
invalid cells and bounded window materialization is independent of run length. Display
materialization is presentation-paced, cached, and performed off the UI owner;
`freeze()` only reads committed state and never calls plugin science or a
plugin materializer.

Presentation cadence is a Surface deadline, not a Dataset-index filter. A busy
same-shot group does not enqueue another full frame; `BoardScheduler` records
admission debt and stages Plane latest on the processor/surface completion wake.
During an active history lease, Runtime accounts every intervening primary
index as valid or invalid. Completion
wakes are coalesced into one owner turn and do not advance the display clock.

`seal_committed()` closes that same run truth as complete or explicitly
partial, with unwritten cells remaining invalid. Save, late one-shot processing,
and full-data scope/reduction read the sealed/current `OwnedSnapshot`; terminal
does not publish a second replacement dataset. Scientific processors declare
exact delivery and consume every ordered event chunk once. Display derivations
declare latest delivery, coalesce while busy, and run concurrently with other
processors while remaining serial within one processor.

Publication roots preserve lineage through exact replay and derived/follower
routes. Accepted-fit outputs are presentation-paced followers of the exact
source publication: they keep its roots without joining a coherent component
that would wait on itself. A missing or trailing fit therefore remains a loud
gap for that source revision; Runtime never attaches the latest unrelated fit,
and terminal fit output is rejected if it trails the finished source generation.

`NodeHost` enforces the descriptor-selected worker/processor role, live commit,
Task progress, Stop, and terminal contracts. Every hosted Task allocates one
unique run directory only when `start()` actually begins it. One atomically
replaced `run.json` is the identity, normalized input summary, current status,
latest progress, explicit artifact inventory, and failure record. Runtime never
dumps live or intermediate data: a domain Task writes a selected complete file
inside its run directory and then registers it through the execution context.
Declared final artifacts must be registered with their semantic contract before
the Task can complete. Stop and failure keep the directory and every registered
file. A Task may explicitly accept Stop before its irreversible terminal work;
any later exception is still a failure.

The installed product is the repository-root ZLC distribution; this directory
is an internal dependency layer. Target invariants and current implementation
status are recorded in the root Architecture and Implementation Plan.
