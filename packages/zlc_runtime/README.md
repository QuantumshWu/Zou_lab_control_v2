# zlc_runtime

`zlc_runtime` owns ZLC node execution, signal publication, causal lineage,
future-publication following, coherent fronts, presentation scheduling, and
selection-derived signals. It is Qt-free and contains no plugin physics.

Logic Nodes submit only new immutable event chunks. `SignalDataPlane` assigns
their generation/revision identity and, for finite data, places them into one
canonical run dataset using the declared schema and `(repeat, point)` origin.
Exact scientific processors consume the immutable event; every display
consumer sees the same publication's canonical full geometry through
`current_dataset()`, with unwritten cells invalid. Monitor outputs have no
finite canonical extent and retain only their latest event. Display
materialization is presentation-paced, cached, and performed off the UI owner;
`freeze()` only reads committed state and never calls plugin science or a
plugin materializer.

`seal_committed()` closes that same run truth as complete or explicitly
partial, with unwritten cells remaining invalid. Save, late one-shot processing,
and full-data scope/reduction read the sealed/current `OwnedSnapshot`; terminal
does not publish a second replacement dataset. Scientific processors declare
exact delivery and consume every ordered event chunk once. Display derivations
declare latest delivery, coalesce while busy, and run concurrently with other
processors while remaining serial within one processor.

`NodeHost` enforces the descriptor-selected worker/processor role, live commit,
Task progress, required artifact, Stop, and terminal contracts. A Task may
explicitly accept Stop before its irreversible terminal work; any later
exception is still a failure.

The installed product is the repository-root ZLC distribution; this directory
is an internal dependency layer. Target invariants and current implementation
status are recorded in the root Architecture and Implementation Plan.
