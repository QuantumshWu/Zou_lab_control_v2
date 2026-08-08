# zlc_ui

`zlc_ui` is the monorepo owner of the reusable, domain-independent UI
layer split out of ZLC v1. It is deliberately usable without a v1 checkout
or any laboratory, plotting, data, or storage package installed.  Its only
non-standard UI dependency beyond PyQt5 is the reference
`PyQt5-Frameless-Window` shell used by the original Fluent layer.

## Boundary charter

1. A component enters this repository only when it depends on PyQt5, the
   reference Qt-only `PyQt5-Frameless-Window` shell, and the Python standard
   library, and its public vocabulary contains no experiment, plotting, or
   data concepts such as `Dataset`, `Signal`, `Device`, `Pulse`, `Plot`,
   `matplotlib`, or `numpy`.
2. This package owns the first two layers only:
   - **Pure controls** (`fluent`, `form`, `board`, `graph`, and
     `concurrency`) are domain-independent building blocks.
   - **Pure views** (`console` and other view packages) expose operator intent
     through `*_requested`, `*_picked`, and `*_committed` signals and accept
     idempotent `set_*` inputs.  They do not own data, experiment, scheduling,
     or plotting logic.
   - Presentation scheduling and surface arbitration belong to `zlc_runtime`;
     generation replacement and application wiring belong to `zlc_workbench`.
     Views provide the handle/mount intents needed by those owners without
     taking over runtime or plot state.
3. Public signal payloads and `set_*` arguments are restricted to `str`,
   `int`, `float`, `bool`, plain tuples, `QWidget`, and this package's own
   headless value types such as `FormSpec`.  Domain objects do not cross the
   boundary.
4. This package directory in the `Zou_lab_control_v2` monorepo is the sole
   owner of current UI code. The v1 `zlc_frontend` and the old standalone
   package repositories are historical references, not alternate edit trees.
5. The package name is intentionally `zlc_ui`, distinct from v1's
   `zlc_frontend`, so a shadow import cannot silently select a legacy copy.
6. The package is independently testable and reviewable inside the monorepo;
   demos use fake data only. The interactive console demo echoes outgoing view
   intents in its in-window log and to stdout; the gallery is a compact control
   survey.

## Development

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -c "import zou_lab_control_v2; import zlc_ui; print(zlc_ui.__file__)"
pytest -q
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
    __version__,
)
```

Use these names for the headless contracts and the Qt application lifecycle.
Concrete feature views stay in their explicit modules (`zlc_ui.console`,
`zlc_ui.pulse`, and so on), so the facade remains a discoverable, bounded API.
The current complete allow-list is recorded in
[`docs/contract.md`](docs/contract.md).

The desktop gallery and console demos live in `examples/`.  The offscreen
environment is only for object-level automated tests; it does not produce a
GUI acceptance image.  On a desktop, omit the environment variable to open
the interactive scrollable gallery.  Gallery is organized as `1. 基础组件`
(named controls), `2. 组合件` (named reusable views), and `3. 完整 GUI 示例`
(TaskConsole, PulseEditor, FigureViewer, and DeviceManager).  Scan slot/API
slot states are visible in the composite section and in PulseEditor itself.

For a human-readable walkthrough, install Jupyter in the notebook environment
and open [`notebooks/usage.ipynb`](notebooks/usage.ipynb).  It has independent
human-runnable cells for Gallery, TaskConsole, PulseEditor, FigureViewer, and
DeviceManager; it then demonstrates how an external presenter supplies plain
tuples and FormSpec values to `TaskConsoleView` and `DeviceManagerView`,
connects outgoing signals, replaces host-owned QWidget surfaces, and closes the
Qt windows.  In a Notebook use the non-blocking `create_window()` flow shown in
[`notebooks/usage.ipynb`](notebooks/usage.ipynb); it calls `ensure_qt_app()`
before any view is constructed and lets IPython install the Qt event-loop hook.
Do not run `%gui qt` before that entry point, and do not call `app.exec_()` from
a cell.

For a size-faithful desktop capture, run these commands without
`QT_QPA_PLATFORM=offscreen`:

```powershell
python examples/capture_acceptance.py --view console
python examples/capture_acceptance.py --view figure
python examples/capture_acceptance.py --view pulse
python examples/capture_acceptance.py --view device
python examples/capture_acceptance.py --view gallery
```

The script is an adapter around the reusable `zlc_ui.acceptance.capture_window`
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
- `zlc_ui.graph` — generic flow graph and text-shape helpers.
- `zlc_ui.concurrency` — reusable Qt owner-wake primitives.
- `zlc_ui.ensure_qt_app` — the single QApplication/HiDPI/Fluent-scale entry
  point; call it before constructing any zlc_ui widget or enabling `%gui qt`.
  It rejects a pre-existing QApplication without the required Qt5 High-DPI
  attributes instead of silently producing a differently scaled GUI.
- `zlc_ui.console` — pure task-console views; presentation runtime stays out.
- `zlc_ui.pulse` — pure pulse schedule, scan, target, preview, and editor-shell
  views driven by frozen plain view models; controller and plot ownership stay
  outside this package. Its Scan/API dot controls emit click intent; presenters
  can reuse `zlc_ui.pulse.cycle_binding_kind()` to cycle duration/DAC fields
  through `off -> scan -> api -> off` and output delays through `off -> api -> off`.
- `zlc_ui.figure_viewer` — pure file/path/info shell and presenter-owned
  QWidget mount point; archive IO and plot rendering stay outside.
- `zlc_ui.acceptance` — the test-only real-screen UI acceptance helper
  `capture_window`. Product code should use `open_fluent_window`; it should
  not implement its own screenshot or resize logic.
- `zlc_ui.device_manager` — pure device-instance editor view; catalog and
  persistence stay with the host.

Detailed view signatures are maintained in `docs/console-views.md`, including
the device-manager contract.
The source-size comparison is recorded in `docs/loc-report.md`.

## FormRuntimeContext

Forms describe fields and validation without importing domain packages.
Dynamic choices or other host-owned values are supplied through the
`FormRuntimeContext` callback-injection interface (available from the
top-level facade).  The form engine never becomes a second owner of
application data.

## Deliberately not moved

The Qt plot editors (`plot_parameters.py`, `plot_spec.py`, and `plot_fit.py`)
remain on the `zlc_plot` side because they edit plot-domain specifications.
The pulse document/controller and plot renderer remain in their domain-side
packages.  `zlc_ui.pulse` contains only the pure Qt projection and its plain
view-model seam; presentation runtime logic remains outside this package.
