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

Finite exact signals have one presentation identity independent of Panel
semantics: Logic shape, live Panel, Edit/Refresh/Save, selector, fit and overlay
all use the accepted canonical full Dataset for that publication. Event chunks
remain internal to acquisition and exact processors. Scope/reduction/fate only
change how the canonical Dataset is drawn; they never switch the data source.
Monitor signals remain latest-event views. Canonical assembly and companion
projection run on the board-owned presentation worker at panel cadence, not in
the Qt owner callback.

`PanelState.interval_ms` is the admission cadence for one atomic data+fit pair.
When a panel is due it admits the then-latest coherent publication as one source
revision; higher-frequency raw Monitor publications are not relabeled as panel
revisions or fit gaps. If a due component is waiting for a derived sibling, its
owed admission is spent by the matching processor/surface completion wake rather
than another clock tick.

Plot hosts are the native event authority for committed Area, viewport,
coordinate threshold and facet focus. Their immutable event is queued to the
Workbench owner turn, which acknowledges it into `PanelState` and mirrors it to
the other host before later display/fit configuration may read that state. A
standing host is never overwritten by the briefly stale owner mirror that its
own gesture is still updating. Fit selection has one order: committed Area (or
explicit x-range), then viewport, then the full range.

Classifier choices are stored together as coordinate-addressed targets
(`axis domain/id + canonical coordinate`, or structural repeat row), not a
facet-index vector or a last-edited annotation. Plot strictly remaps every
target to the current facets; duplicate or missing coordinates are rejected
instead of guessed. Live, Edit, replacement hosts, layout restore and Figure
Viewer all consume that same plural `PanelState.classifier_thresholds` truth.

Image overlay candidates are selected only by the neutral
`IMAGE_POINT_OVERLAY_CONTRACT`. The geometry and numeric/bool status Dataset
are adapted by `zlc_plot`; Workbench never imports a concrete Logic Node or
reconstructs its domain result.

While a Task runs, Add Logic and that Task's source/preview identities and data
projection are frozen. Other panels and pure display controls remain usable;
Workbench does not duplicate Task lifecycle or scientific state to enforce
this.

## Save boundaries

- Header **Save Layout** writes stopped node drafts, named-device choices,
  signal wiring and panel layout/state. It does not freeze datasets or save
  running state/device snapshots.
- Header **Save Screenshot** writes one ordinary image of the TaskConsole GUI,
  with no layout, data archive or provenance.
- Panel Edit **Save Fig** writes only the frozen image/data currently shown by
  that panel, its plot/overlay state, and the run-time call chain and actual
  device snapshots already captured when the runs executed. It does not include
  another panel or the whole monitor board. At the click, Workbench freezes the
  exact state/data and the identity-matched display viewport; viewport belongs
  to this Figure archive's strict `view` section, not reusable Save Layout.
  A dedicated composition-owned worker streams the archive first, then performs
  one saved-host configure/render and image write. The Qt beat and Stop remain
  live; a second Save for the same panel is rejected rather than queued.

Calibration JSON is a separate Task artifact. Panel Save records its actual
path where relevant, without embedding the JSON or adding fingerprint/hash.

## Qt and owner shutdown

Figure Viewer reads and fully prepares a candidate off the Qt owner, atomically
mounts only a successful candidate, and keeps the previous accepted figure on
failure. Its resize/save/host retirement are likewise asynchronous. TaskConsole
keeps its lifecycle beat running while nodes, projections, plot hosts or Panel
Save retire; the window stays visible until every owner is actually stopped.
Session/device shutdown runs on the one flow-owned serial device worker used by
discover/init/tune/shutdown, while Panel Save keeps its independent I/O worker.
Completion callbacks return to the Qt owner and queue the final guarded close;
timeouts report what is still active but never claim the window or worker is
closed.

## Check the environment first

```bash
python -m zou_lab_control_v2 check
```

That root-bootstrap entry prints every resolved production module path before
reporting success, so the check cannot silently measure an installed sibling
copy.

Run it from anywhere except the workspace root. Three separate incidents in this
project came from an import that succeeded while the wrong code ran — a monolith
installed under the same names, an uninstalled package resolving to an empty
namespace because the current directory happened to sit beside it, an editable
install pointing at a deleted copy. None of them raised. This asserts that every
package resolves to the repo that owns it, and that the retired names are gone.
