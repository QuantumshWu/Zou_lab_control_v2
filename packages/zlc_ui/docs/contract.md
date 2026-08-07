# zlc_ui package contract

`zlc_ui` is a pure view-components package.  The package initializer is a
small, lazy facade: importing it does not create a `QApplication` or import a
view implementation.

## 顶层 facade(allow-list)

The stable top-level names are exactly these:

```python
__all__ = (
    "__version__",
    "BoardMetrics",
    "DelayRowVM",
    "FieldVM",
    "FormChoice",
    "FormFieldProps",
    "FormRuntimeContext",
    "FormSpec",
    "PeriodVM",
    "PortRowVM",
    "RepeatVM",
    "ScanPageRecord",
    "ScheduleVM",
    "TargetPortRecord",
    "TargetWidthRule",
    "VALIDATOR_FLOAT",
    "VALIDATOR_INT",
    "WINDOW_SCREEN_FRACTION",
    "capture_window",
    "cycle_binding_kind",
    "ensure_qt_app",
    "open_device_manager",
    "open_figure_viewer",
    "open_pulse_editor",
    "open_task_console",
)
```

Everything else in this package is private, and that is the point.  A host
used to be handed the window's own widgets, so it could reach a sub-view and
push a value into one panel of one page -- and the picture on screen ended up
with two owners.  The delay column went on showing rows that Hide Off had
already taken out of the cards beside it, because "which rows exist" had
quietly become two facts.  Whatever the outside can hold, the outside will
assemble, and assembling a UI is the one job a composition root does not have.

So the surface is of three kinds, and nothing else:

* **One entry per window.**  `open_pulse_editor` / `open_figure_viewer` / `open_device_manager` /
  `open_task_console` return a HANDLE: signals to
  hear, methods to call, and no way through to a QWidget.  The window's
  lifecycle -- the launcher, the shared screen-fit size, centring, retention
  and the close handshake -- belongs here too.
* **The wiring vocabulary.**  `ScheduleVM`, `PeriodVM`, `FieldVM`,
  `PortRowVM`, `DelayRowVM`, `RepeatVM`, `ScanPageRecord`,
  `TargetPortRecord`, `TargetWidthRule`, the two validator tokens and
  `cycle_binding_kind` are what a host builds to say WHAT to show.  They are
  named here because they are the contract; reaching into a submodule for
  them is how a host ends up importing views as well.
* **The bootstrap and the headless forms.**  `ensure_qt_app` is the single
  application entry; the form and board names are reusable headless
  contracts, `FormRuntimeContext` the callback injection for dynamic
  choices; `__version__` identifies which installed package was imported.

A drawn panel crosses as its HOST, never as a widget: this package may not
import the package that draws (a canvas would carry matplotlib into the GUI
layer), and the outside may not hold a widget, so the host is asked for its
widget on this side of the wall -- see `PulseEditorHandle.show_preview`.

`FlowGraph`, `FlowGraphEdge`, and `FlowGraphNode` are deliberately not
re-exported here.  Their implementation remains available from
`zlc_ui.graph`, which is the explicit path for the graph demonstration.
Concrete Qt views likewise remain in their feature submodules so the facade
does not become an unbounded view registry.
