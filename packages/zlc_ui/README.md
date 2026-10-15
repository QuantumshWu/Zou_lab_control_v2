# zlc_ui

`zlc_ui` owns the reusable, domain-independent Qt view layer. It owns no
laboratory, plotting, data, or storage MODEL and imports none of those
layers, with one named exception: the unit vocabulary `zlc_data.units`,
which every number on screen is formatted and parsed through so the view
never keeps a second prefix table.  Its only non-standard UI dependency
beyond PyQt5 is the reference `PyQt5-Frameless-Window` shell used by the
Fluent layer.

## Ownership boundary

1. A component enters this repository only when it depends on PyQt5, the
   reference Qt-only `PyQt5-Frameless-Window` shell, the Python standard
   library and the unit vocabulary above, and it owns no experiment,
   plotting or data computation: a `Dataset`, a `Signal`, a `Device` or a
   `Plot` reaches it only as plain projected values.  Feature views are
   named for what they show (`console`, `pulse`, `device_manager`,
   `figure_viewer`); the names are not the models.
2. This package owns the view layer only:
   - **Pure controls** (`fluent`, `form`, and `board`) are
     domain-independent building blocks.
   - **Pure views** (`console` and other view packages) expose operator intent
     through `*_requested`, `*_picked`, and `*_committed` signals and accept
     idempotent `set_*` inputs.  They do not own data, experiment, scheduling,
     or plotting logic.
   - Each feature view owns its Qt widget tree and widget-local interaction
     state. Its public handle owns Qt mount, visibility, and close operations.
     Presentation scheduling and surface arbitration belong to `zlc_runtime`;
     application/session state and wiring belong to `zlc_workbench`.
3. Public signal payloads and `set_*` arguments are restricted to `str`,
   `int`, `float`, `bool`, plain tuples, `QWidget`, and this package's own
   headless value types such as `FormSpec`.  Domain objects do not cross the
   boundary.
4. `zlc_ui` is the sole owner of product widgets and window handles. Hosts
   consume those handles and view-model inputs; they do not reach through a
   handle to assemble or mutate an internal widget subtree.
5. Importing the package creates no `QApplication` and opens no window.
   `ensure_qt_app()` is the one explicit application-lifecycle entry.
6. The package is independently testable and reviewable inside the monorepo;
   demos use fake data only. The interactive console demo echoes outgoing view
   intents in its in-window log and to stdout; the gallery is a compact control
   survey.

## Plot interaction and close contract

The TaskConsole Selectors switch is a global plot-pointer gate. Off means the
plot consumes no area, zoom, pan, hover or double-click focus gesture, and the
ordinary wheel remains with the outer board. On a FacetGrid overview, only a
left-button double click may focus one cell; an Area selector can start only on
that focused cell (or on a non-grid plot).

Views emit close intent; a composition-installed close guard decides when the
top-level window may actually disappear. Handles provide queued close retry so
completion callbacks never re-enter the current close event. The Qt owner does
not wait for device, node, archive or plot workers, and a refused close leaves
the window visible. Application shutdown is that close: Workbench must retire
its real owners before the guard accepts it. At interpreter exit Qt objects
are left to the operating system (`sip.setdestroyonexit(False)`): no state
lives in a Qt destructor, and C++ destructors run in Python's teardown order
would destroy widgets after their QApplication. Before Python finalizes, one
`atexit` hook releases every top-level window's native window and delivers
queued `deleteLater` deletions, without close events, so no close guard is
asked; left to Qt's static teardown they crashed the process after its last
line.

## Interaction cues

Every window says what a control does and what state it is in the same way,
and every cue comes from `zlc_ui.fluent`; no view picks a colour of its own.
A state is a fill, never an extra outline or a focus ring.

- **Primary** (dark blue, white words) is the one run or commit command on a
  surface: Start on a node's Edit page, Init devices, a dialog's OK.  A Save
  or Apply with something new to keep becomes primary and gains a `*`.
- **Commands** are light blue.  Under the pointer they step a shade deeper,
  and held down they are medium blue -- the colour of anything chosen.
- **Toggle commands** -- a command whose next press undoes the last one:
  Expand/Collapse, Pause/Resume, Hide/Show side panels, a button that opens
  a frame (Setting, Preset…, Edit…) -- stay medium blue while what they did
  is in effect.  When the label flips, it names what the next press will do.
  On/off settings are switches, not toggle commands, and a disclosure chevron
  (a device card's fold, a tree's triangle) shows its state by its direction.
- **Red takes something away or stops it**: Remove, Clear, Ungroup, Unload,
  Discard, Revert (a Logic Edit page's edits), Exclude selected, Reset (a
  point review's exclusions), Reset wavefront, a device's Close, Shut down
  devices, every Stop.  A labelled
  one has a pale red face; a one-glyph `×` repeated down rows rests as a grey
  ghost and turns red under the pointer.  It acts on one click.
- **Closing a view is not removing anything**: a tab's `×` and a Setting
  frame's `×` are quiet grey ghosts that never turn red.

Removing a **logic node** or a **plot panel** takes two clicks — a
double-click, or two clicks within the system double-click interval. The first
click turns the control into the white-on-red confirm; if the second click does
not come, it returns to rest. This applies to the node's × in the Logic list
and the Remove on its Edit page, and to the panel's × on its card and the
Remove in its Setting frame: every control that removes one of the two. Every
other remover in the application is red and acts on one click. Stop always
acts on one click. Commands that already ask in a dialog are not armed as well.

Status is one table of five words, the dot and the words beside it from the
same row: `idle` grey (nothing under way and nothing wrong, a draft still to
fill in), `running` green (something lasting is live), `task` a blue dot
(an operation under way or just done), `warning` orange (needs attention but
is not a failure), `error` red (failed or unreachable).

The API, one line each:

- `FluentButton.set_active(on)` — a toggle command's held-down face, projected
  by its owner; a `FluentSettingsPopupAnchor` sets it on its button while the
  frame is open.
- `FluentButton(..., confirm_twice="remove this node")` and its `confirmed`
  signal — the two-click remover; act on `confirmed`, never on `clicked`.
- `FluentStatusDot.set_state(word)` and `set_status_ink(label, word)` — a dot
  and its words (a label, or a copyable readout) from `STATUS_CUES`; a
  presenter says the word with the text, so a failure is always red.  A
  sentence under an editor's line takes the same table: a problem that
  blocks is `error`, a kept item this run does not use is `warning`, a
  parked helper is `idle`.
- `set_muted(label, muted)` — a row kept but not in use (a parked field, a
  mark that plays nothing) in grey words, as a selection grid's
  `muted_rows`.
- `fill_grouped_choice_combo(..., texts=...)` — what a closed tree picker
  reads for each key: a signal as the presenter spells it everywhere else
  (`<producer>/<output>`).

## Development

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
zlc evidence gui_offscreen --repo C:\path\to\Zou_lab_control
```

The stable reusable facade is intentionally small:

```python
from zlc_ui import (
    BoardMetrics,
    FormChoice,
    FormFieldProps,
    FormRuntimeContext,
    FormSpec,
    ensure_qt_app,
)
```

Use these names for the headless contracts and the Qt application lifecycle.
Concrete feature views stay in their explicit modules (`zlc_ui.console`,
`zlc_ui.pulse`, and so on), so the facade remains a discoverable, bounded API.
The executable facade source of truth is `zlc_ui.__all__`.

The desktop gallery and console demos live in `examples/`.  The offscreen
environment is only for object-level automated tests; it does not produce a
GUI acceptance image.  On a desktop, omit the environment variable to open
the interactive scrollable gallery.  Gallery is organized as `1. 基础组件`
(named controls), `2. 组合件` (named reusable views), and `3. 完整 GUI 示例`
(TaskConsole, PulseEditor, FigureViewer, and DeviceManager).  Scan slot/API
slot states are visible in the composite section and in PulseEditor itself.

For an offline walkthrough, use the one product tutorial at
`packages/zlc_workbench/notebooks/usage.ipynb`; it uses virtual devices and the
normal NotebookView, not fake UI windows. Human GUI inspection stays in the
separate real-screen lane below.

For a size-faithful desktop capture, run these commands without
`QT_QPA_PLATFORM=offscreen`:

```powershell
zlc capture --view console --template virtual
zlc capture --view figure --path D:\data\run.npz
zlc capture --view pulse --workspace D:\experiment --pulse imaging.json
zlc capture --view device --workspace D:\experiment
```

`--pulse` names a pulse the operator keeps in that workspace's `pulses/` folder.

The Workbench command is an adapter around the reusable `zlc_ui.acceptance.capture_window`
API.  That API calls `ensure_qt_app()` before the same public `create_window()`
entry a human uses, verifies the exact shared `WINDOW_SCREEN_FRACTION` size,
and writes both a physical-DPR window crop and a desktop-relative proportion
canvas.  A wrong launcher raises instead of being silently resized.  The
offscreen backend is rejected: it cannot reproduce the human monitor and is
not a UI acceptance result.  Use the resulting real-screen PNGs for manual UI
inspection at the same screen ratio; there is no fixed-size canvas, reference
fixture, or image-difference path.

## Module map

The package is organized as follows:

- `zlc_ui.fluent` — shared Fluent controls, styling, scale helpers, and the
  reusable `frameless_content_top_margin()` native-titlebar boundary metric.
- `zlc_ui.form` — Qt-free form specifications plus their Qt projection.
- `zlc_ui.board` — domain-independent card geometry.
- `zlc_ui.ensure_qt_app` — the single QApplication/HiDPI/Fluent-scale entry
  point; call it before constructing any zlc_ui widget or enabling `%gui qt`.
  It rejects a pre-existing QApplication without the required Qt5 High-DPI
  attributes instead of silently producing a differently scaled GUI.
- `zlc_ui.console` — pure task-console views; presentation runtime stays out.
  `PointReviewView`使用完整Fluent control family围绕caller提供的普通QWidget：
  `FluentDialogWindow`负责modal Fluent chrome/lifecycle，view负责搜索、site
  checkbox、scroll、status和actions；它不理解plot、SiteMap或Calibration。
- `zlc_ui.pulse` — pure pulse schedule, scan, Config, target, preview, and editor-shell
  views driven by frozen plain view models; controller and plot ownership stay
  outside this package. Binding popups select independent Scan capability and
  Default/API/Config source; Config names and saved-file editing have their own
  page. Views emit intents and receive accepted projections, never perform I/O.
- `zlc_ui.figure_viewer` — pure file/path/info shell and presenter-owned
  QWidget mount point; archive IO and plot rendering stay outside.
- `zlc_ui.acceptance` — the test-only real-screen UI acceptance helper
  `capture_window`. Product code should use `open_fluent_window`; it should
  not implement its own screenshot or resize logic.
- `zlc_ui.device_manager` — pure device-instance editor view; catalog and
  persistence stay with the host.

Detailed signatures are maintained in `docs/console-views.md` (the
`TaskConsoleHandle` port, its internal views and the device-manager contract)
and `docs/pulse-views.md` (the `PulseEditorHandle` port and its pages).

## FormRuntimeContext

Forms describe fields and validation without importing domain packages.
Dynamic choices or other host-owned values are supplied through the
`FormRuntimeContext` callback-injection interface (available from the
top-level facade).  The form engine never becomes a second owner of
application data.

## Domain boundaries

`zlc_plot` owns what a plot's parameters MEAN -- the semantic declarations,
their vocabularies and validation; `zlc_ui` owns every Qt widget that edits
them, including the Fluent choice controls and the Scope wheel, and the two
meet only through plain projections (`FormSpec`, field descriptors, typed
values).  The pulse document/controller belongs to `zlc_pulse`, and plot
rendering belongs to `zlc_plot`. `zlc_ui.pulse` contains only the Qt
projection and its plain view-model seam; presentation runtime logic remains
outside this package.
