# zlc_workbench

The composition root: the only package allowed to know all the others.

Presenters live here, wiring `zlc_ui`'s mute views to `zlc_runtime`, `zlc_atom`
and `zlc_plot`. So do the application entry points and the cross-package
end-to-end tests.

Nothing else may. A domain rule, a rendering decision or a signal mechanism that
appears in this package is misplaced, and belongs to whichever package owns that
subject. This is a constitutional limit, not a style preference: the whole point
of the split was that each subject has exactly one home.

The notebook and the GUI drive the **same** session facade. If those two ever
grow separate paths, the split has failed.

TaskConsole and Pulse Editor are two windows over that same `Experiment`
session, named devices, virtual world, and sequencer; no second session or IPC
service is introduced. Workbench arbitrates multiple `OBSERVE` readers and one
`EXCLUSIVE` Logic Node owner per concrete device instance. It validates a new
request before stopping only the old nodes that conflict with that instance.

## TaskConsole product wiring

The formal virtual and physical path is Calibration artifact -> Camera frames
signal -> Occupancy processor -> fixed-kind Plot Panel. Add Logic creates a
stopped row and opens its non-modal Edit tab. Each row has one shared draft;
Logic Edit uses Start/Restart, while Panel Edit's Producer Restart sends that same
draft through the same restart endpoint. Setting and Panel Edit likewise
project one immutable/replace-style `PanelState`, so signal, size, interval and
plot parameters cannot drift between two copies.

Committed selectors are routed panel -> exact displayed signal -> direct
producer descriptor. The descriptor returns a data-only draft patch (for
example Image Area to sensor ROI); Workbench contains no camera-SDK coordinate
branch. A new run keeps its stable signal key, gets a new generation, and causes
the panel to replace its plot host.

Logic descriptors also declare typed `NodePreviewSpec` values: the exact output
declaration, plot kind, initial semantic projection, and optionally a stable
companion-producer suffix. Workbench opens no preview until that producer's
first real publication, then binds the panel to the declared signal rather than
reconstructing plugin data. For example, SLM Feedback's qCMOS panel binds to
the companion `camera/frames` output and applies the descriptor's
readout-frame/repeat-mean semantic; its candidate phase and history remain the
Task's own outputs. At Task terminal, auto-preview panels whose Monitor signals
retire are removed, while a retained Runtime camera dataset and the durable
Task artifact keep their respective data and product truth.

If a hosted Task publishes one active Runtime operator-input request,
TaskConsole opens the request-kind interaction only after its referenced
publication exists. Calibration's point-selection request uses the same image
host and stable point identities as its companion Monitor preview；`zlc_ui`
提供Fluent modal/view，`zlc_plot`提供point gesture/overlay，Workbench只连接plain
identities与两个完整owner。Cancelling the dialog is Task Stop, while
confirmation answers only that exact request. Workbench does not interpret
which sites are scientifically valid.

Finite exact signals have one presentation identity independent of Panel
semantics: Logic shape, live Panel, Edit/Refresh/Save, selector, fit and overlay
all use the accepted canonical full Dataset for that publication. Event chunks
remain internal to acquisition and exact processors. Scope/reduction/fate only
change how the canonical Dataset is drawn; they never switch the data source.
Panel semantic keys encode the exact `AxisRef` domain/id, never its display
label, and scoped coordinates remain tagged latest/value records so text such
as `"latest"` cannot change meaning.
Ordinary Monitor signals, including Occupancy results, remain latest-event
views and keep the camera cycle's frame geometry. A capable derived output
exposes a Runtime-owned source-index Dataset only while a panel's generic
fate/window target holds a bounded history lease. Recording starts at that
panel's current event, never backfills earlier shots, and stops when the last
such panel closes. Runtime is the only cross-publication history owner; an
event/indexed representation change advances one signal-level presentation
epoch, invalidates every Panel bound to that signal and remounts them from the
same current publication without inventing a scientific revision. Generic
primary-index fate/window chooses latest or history and every skipped/failed
source index remains invalid rather than disappearing. Canonical assembly and companion
projection run on the board-owned presentation worker at panel cadence, not in
the Qt owner callback.

`PanelState.interval_ms` is the Surface deadline for one atomic data+fit front.
When a same-shot group is still rendering, Workbench does not enqueue another
full frame; it keeps admission debt and stages Plane latest on completion. The
Runtime indexed Dataset, not Workbench, preserves every source primary index and
its validity during that lease. Processor/surface completion wakes spend already-due work without
waiting for another interval.

A standing Plot host receives both the complete coalescing-safe target and the
fields authored by the current edit; new/replacement/save hosts need only the
complete PanelState. This distinction lets Plot's one transition owner clear
Fixed color bounds when Tight/Normal is selected and prevents a window edit from
asking Runtime to rematerialize a publication whose generation has retired.

`PanelState` is the authored target; the exact `DisplayDescription.spec` returned
by a successful host transaction is the accepted Live/Frozen/Viewer contract.
Rejected targets remain repairable but never become capability or interaction
truth. Plot hosts emit selector/viewport observations carrying the exact Dataset
generation and revision that produced them. TaskConsole Console is the only
interaction owner: it verifies that identity and the accepted spec, advances the
panel-owned interaction sequence, then writes `PanelState`, publishes a
derivation or mirrors the state to the other host. Views only project this state,
and a standing host is never overwritten by the briefly stale owner mirror that
its own gesture is still updating. Fit selection has one order: committed Area
(or explicit x-range), then viewport, then the full range.

Classifier choices are stored together as coordinate-addressed targets
(`axis domain/id + canonical coordinate`, or structural repeat row), not a
facet-index vector or a last-edited annotation. Plot strictly remaps every
target to the accepted facets; duplicate targets are rejected, while targets
whose coordinates no longer exist after a legal semantic replacement are
cleared instead of being kept by position or blocking the edit. Live, Edit,
replacement hosts, layout restore and FigureViewer all consume the same
accepted Plot contract and plural `PanelState.classifier_thresholds` truth.

Image overlay candidates are selected only by the neutral
`IMAGE_POINT_OVERLAY_CONTRACT`. The geometry and numeric/bool status Dataset
are adapted by `zlc_plot`; Workbench never imports a concrete Logic Node or
reconstructs its domain result.

While a Task runs, Add Logic and that Task's source/preview identities and data
projection are frozen. Other panels and pure display controls remain usable;
Workbench does not duplicate Task lifecycle or scientific state to enforce
this.

## TaskRun wiring

Workbench supplies the workspace save root but does not allocate run folders
while an editor or draft is merely open. NodeHost allocates one unique folder
when the worker actually starts, establishes `run.json`, and gives the execution
context the only artifact-registration path. Workbench projects that lifecycle;
it does not maintain a second Task status or plugin-specific report manager.

Calibration, Temperature and SLM Feedback all use the same TaskRun contract.
Their domain owners select final JSON/NPZ, summaries and important typed Figure
artifacts. Stop and failure leave the run folder reachable with its truthful
partial inventory. Runtime never asks Workbench to dump every publication or
intermediate shot.

## Save boundaries

- Header **Save Layout** writes stopped node drafts, named-device choices,
  signal wiring and panel layout/state. It does not freeze datasets or save
  running state/device snapshots.
- Header **Save Screenshot** writes one ordinary image of the TaskConsole GUI,
  with no layout, data archive or provenance.
- Panel Edit **Save Fig** writes only that panel's frozen typed data, exact Plot
  recipe, overlay, viewport and causal lineage. The `zlc.figure` NPZ is primary;
  its same-stem PNG is a preview. It does not include another panel or the whole
  monitor board. A dedicated composition-owned worker publishes the archive
  first and then renders the preview through the Edit surface's own plot host
  when it already shows that exact freeze; otherwise through the same Plot
  host/configure path used by TaskConsole and FigureViewer. The Qt beat and
  Stop remain live; a second Save for the same panel is rejected rather than
  queued.

Domain final JSON/NPZ files remain separate Task artifacts inside their run
folders. A Figure records their actual lineage/path where relevant without
embedding them or adding fingerprint/hash.

## Qt and owner shutdown

Figure Viewer reads and fully prepares a candidate off the Qt owner, publishes
each saved typed Dataset as a sealed signal in its private Runtime plane, and
keeps the previous accepted archive on failure. The default card restores the
exact Figure recipe without shape inference; every later blank fixed-kind card,
signal choice, derived ROI/Fit signal, resize and edit uses the same
ConsolePresenter/SelectionBridge/Plot host path as TaskConsole. It projects the
saved direct-parent lineage as one node-edge Flow of unique Logic and Device
nodes. Save and shutdown are asynchronous. TaskConsole
keeps its lifecycle beat running while nodes, projections, plot hosts or Panel
Save retire; the window stays visible until every owner is actually stopped.
Session/device shutdown runs on the one flow-owned serial device worker used by
discover/init/tune/shutdown, while Panel Save keeps its independent I/O worker.
Completion callbacks return to the Qt owner and queue the final guarded close;
timeouts report what is still active but never claim the window or worker is
closed.

Pulse Editor never talks to the board on the Qt owner.  Its window has one
serial device worker and one SAFE worker; every conversation a control starts
-- the 100 ms "what is the board doing" poll, On Pulse, Hold, Step, Sync,
Connect -- runs on the device worker and is shown when it delivers.  A status
question is one at a time (a request while one is pending only queues its
follow-up), is not sent while a command is in progress (the command reports
the board it left), and an answer from before a command started is dropped.
On the experiment machine the pulse server answers only between UART
transactions, and the poll used to hold the GUI thread for that wait.  The
presenter's public methods (`refresh_run_state`, `connect_to`, `hold_scan_point`,
`sync_from_sequencer`, ...) still answer before returning: they are what a
notebook, a test and the app's start-up call.

## Check the environment first

```bash
zlc check
```

The command reads the installed manifest projection, prints all eight module
origins, verifies that each file belongs to `zou-lab-control`, and rejects the
retired monolith names. It is independent of the working directory.
