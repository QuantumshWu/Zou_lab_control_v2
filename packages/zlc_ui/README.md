# zlc_ui

`zlc_ui` owns the reusable, domain-independent Qt view layer. It imports no
laboratory, plotting, data, or storage layer even though all eight layers ship
in one distribution. Its only non-standard UI dependency beyond PyQt5 is the reference
`PyQt5-Frameless-Window` shell used by the Fluent layer.

## Ownership boundary

1. A component enters this repository only when it depends on PyQt5, the
   reference Qt-only `PyQt5-Frameless-Window` shell, and the Python standard
   library, and its public vocabulary contains no experiment, plotting, or
   data concepts such as `Dataset`, `Signal`, `Device`, `Pulse`, `Plot`,
   `matplotlib`, or `numpy`.
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
the window visible. There is no raw `atexit` widget-deletion path standing in
for application shutdown: Workbench must retire its real owners before the
guard accepts the close.

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
zlc capture --view pulse --pulse imaging_template.json
zlc capture --view device --workspace D:\experiment
```

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
- `zlc_ui.pulse` — pure pulse schedule, scan, target, preview, and editor-shell
  views driven by frozen plain view models; controller and plot ownership stay
  outside this package. Its Scan/API dot controls emit click intent only;
  presenters ask the public `zlc_pulse.cycle_binding_kind()` domain API for the
  next legal state and project the resulting `FieldVM` back into the view.
- `zlc_ui.figure_viewer` — pure file/path/info shell and presenter-owned
  QWidget mount point; archive IO and plot rendering stay outside.
- `zlc_ui.acceptance` — the test-only real-screen UI acceptance helper
  `capture_window`. Product code should use `open_fluent_window`; it should
  not implement its own screenshot or resize logic.
- `zlc_ui.device_manager` — pure device-instance editor view; catalog and
  persistence stay with the host.

Detailed view signatures are maintained in `docs/console-views.md`, including
the device-manager contract.

## FormRuntimeContext

Forms describe fields and validation without importing domain packages.
Dynamic choices or other host-owned values are supplied through the
`FormRuntimeContext` callback-injection interface (available from the
top-level facade).  The form engine never becomes a second owner of
application data.

## Domain boundaries

Qt plot editors belong to `zlc_plot` because they edit plot-domain
specifications. The pulse document/controller belongs to `zlc_pulse`, and plot
rendering belongs to `zlc_plot`. `zlc_ui.pulse` contains only the Qt projection
and its plain view-model seam; presentation runtime logic remains outside this
package.
