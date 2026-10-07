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
output is capable of history. Runtime exposes a window-bounded ordinary Dataset
over a neutral `primary-index` only while a consumer holds a window lease;
retention begins at the current event, uses the largest active window, and is
dropped with the last lease. A reader that names a window (a Processor's input
window, a region's, an overlay's) reads exactly that many source positions
ending at its publication, from its first event on: a row the history does not
hold is invalid, so the window's shape never depends on when it was read. An
exact follower's lease holds the rows it has yet to read while it lags, so
each publication is evaluated over its full window; readers without a window
still see only the largest lease's. A window whose shots, times the newest event's
bytes, less what is already held for it, exceed the machine's free physical
memory is refused outright rather than truncated: the lease when it is taken or
grown (less its live history), the commit when a generation's first shot meets
a lease taken before it (a restored panel, a Stop and Start; less the window
the dropped previous generation held, which its panel draws until this
publication replaces it, so a restart is refused only by what its record grew).
Missing computations inside that interval are
invalid cells and bounded window materialization is independent of run length.
Runtime is the only owner of data accumulated across publications. Changing a
signal between latest-event and indexed-history representation advances that
signal's presentation epoch and invalidates every display consumer, even when
the scientific publication is unchanged; this representation epoch is neither
a run generation nor a content revision. Display materialization is
presentation-paced, cached, and performed off the UI owner;
`freeze()` only reads committed state and never calls plugin science or a
plugin materializer.

Run metadata is declared once, after preparation, through
`NodeExecutionContext.set_run_record(record)` (or `SignalDataPlane.set_run_record`
for a direct producer). `LiveDatasetOutput` carries only its event data and
event-varying metadata. A Processor can provide `describe_run(inputs)`; the Host
calls it once after the first successful evaluation. Runtime owns the frozen
declaration and rejects replacement; later commits do not re-compare a plan.

Exact delivery has separate, explicit pending-event and payload-byte limits
(by default 1024 events and the larger of 128 MiB and 32 times the first
event's bytes, ancestors pinned by an event counted with it); these do not
truncate scientific history. Overflow fails that consumer rather
than dropping data or blocking the producer. Finite replay reads existing
chunks lazily. Historical ancestry retains event identities and records, not
all ancestor arrays; active computations, coherent fronts and frozen snapshots
still own the exact payloads they need.

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

A later acquisition failure preserves the verified partial prefix and signals
failure to existing followers; it does not report normal end-of-stream.
Retiring a generation (a Restart, a replaced owner) is not a failure: its
followers see the stream end (`StreamEndedEarly`) and end cancelled.
Publication callbacks receive the names of the signals a commit published, so
a board can wake only for the signals it shows. Stop
retains data. Explicit node removal or board replacement retires that owner's
retention, without invalidating independently owned frozen snapshots.

A signal is spelled `@logic/<owner>/<output>` by `stable_signal_key` and read
back by `split_signal_key` (the owner is everything up to the last slash); no
other package spells or parses that grammar.

A region drawn on a Rolling panel publishes no signal: its shot bounds name no
axis (index or shot time), so Runtime cannot map them to history rows. It only
scopes that panel's own fit.

Publication roots preserve lineage through exact replay and derived/follower
routes. Accepted-fit outputs are presentation-paced followers of the exact
source publication: they keep its roots without joining a coherent component
that would wait on itself. A missing or trailing fit therefore remains a loud
gap for that source revision; Runtime never attaches the latest unrelated fit,
and terminal fit output is rejected if it trails the finished source generation.

`NodeHost` enforces the descriptor-selected worker/processor role, live commit,
Task progress, Stop, and terminal contracts. Every hosted Task allocates one
unique run directory only when `start()` actually begins it. A run writes two
records about itself, each created once and never replaced: `start.json` at
Start (identity, normalized input summary, `started_at`) before any irreversible
work, and `run.json` when the run is over (terminal state, stop reason, last
progress, explicit artifact inventory, failure record). Nothing is written in
between -- progress and artifact registration stay in the process -- so a run
directory holding `start.json` and no `run.json` is a run that did not finish.
Runtime never dumps live or intermediate data: a domain Task writes a selected
complete file inside its run directory and then registers it through the
execution context. Declared final artifacts must be registered with their
semantic contract before the Task can complete. Stop and failure keep the
directory and every registered file; a partial-exit writer that fails during
Stop is reported as the observation's and the stopped record's error while the
state stays stopped. A Task may explicitly accept Stop before its irreversible
terminal work; any later exception is still a failure.

A hosted Task may expose one active `OperatorInputRequest` through `NodeHost`
and wait in its worker without polling. The response names that exact request;
stale or duplicate responses are rejected, and Stop wakes the worker and
cancels the wait. Runtime owns only this toolkit-neutral lifecycle—the
Workbench chooses the UI for each request kind and plugin science remains in
the Task.

The installed product is the repository-root ZLC distribution; this directory
is an internal dependency layer. Target invariants and current implementation
status are recorded in the root Architecture and Implementation Plan.
