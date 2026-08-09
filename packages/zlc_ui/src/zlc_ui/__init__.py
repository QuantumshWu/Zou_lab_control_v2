"""Pure PyQt5 controls and views.

The package initializer intentionally stays lightweight.  Importing
``zlc_ui`` does not create a QApplication or import optional view modules.
"""

from importlib import import_module as _import_module


_EXPORTS = {
    "ensure_qt_app": ("zlc_ui.qt", "ensure_qt_app"),
    "FormChoice": ("zlc_ui.form", "FormChoice"),
    "FormFieldProps": ("zlc_ui.form", "FormFieldProps"),
    "FormSpec": ("zlc_ui.form", "FormSpec"),
    "FormRuntimeContext": ("zlc_ui.form", "FormRuntimeContext"),
    "BoardMetrics": ("zlc_ui.board", "BoardMetrics"),
    # How big each named panel's picture is, which the DRAWING package
    # knows and this one may not import.  Supplied once by whoever
    # composes a window; the alternative was a second copy of the
    # geometry here, kept in step by a test.
    "use_panel_display_sizes": ("zlc_ui.board", "use_panel_display_sizes"),
    # One call per window.  What comes back is a handle -- see zlc_ui.windows
    # for why the outside is never handed a widget tree.
    # The one UI-acceptance capture, and the screen fraction every window is
    # sized by.  A CI script that wants a picture of a real window needs both,
    # and reaching a submodule for them is how a script ends up sizing its own.
    "capture_window": ("zlc_ui.acceptance", "capture_window"),
    "WINDOW_SCREEN_FRACTION": ("zlc_ui.fluent", "WINDOW_SCREEN_FRACTION"),
    "open_device_manager": ("zlc_ui.windows", "open_device_manager"),
    "open_figure_viewer": ("zlc_ui.windows", "open_figure_viewer"),
    "open_pulse_editor": ("zlc_ui.windows", "open_pulse_editor"),
    "open_task_console": ("zlc_ui.windows", "open_task_console"),
    # The wiring vocabulary: what a host builds to say WHAT to show.  These
    # are the contract, so they are named here rather than reached for in a
    # submodule, which is how a host ends up importing views as well.
    "ConnectionChoiceVM": ("zlc_ui.pulse.models", "ConnectionChoiceVM"),
    "ConnectionVM": ("zlc_ui.pulse.models", "ConnectionVM"),
    "DelayRowVM": ("zlc_ui.pulse.models", "DelayRowVM"),
    "FieldVM": ("zlc_ui.pulse.models", "FieldVM"),
    "PeriodVM": ("zlc_ui.pulse.models", "PeriodVM"),
    "PortRowVM": ("zlc_ui.pulse.models", "PortRowVM"),
    "RepeatVM": ("zlc_ui.pulse.models", "RepeatVM"),
    "ScanPageRecord": ("zlc_ui.pulse.models", "ScanPageRecord"),
    "ScheduleVM": ("zlc_ui.pulse.models", "ScheduleVM"),
    "TargetPortRecord": ("zlc_ui.pulse.models", "TargetPortRecord"),
    "TargetWidthRule": ("zlc_ui.pulse.models", "TargetWidthRule"),
    "VALIDATOR_FLOAT": ("zlc_ui.pulse.models", "VALIDATOR_FLOAT"),
    "VALIDATOR_INT": ("zlc_ui.pulse.models", "VALIDATOR_INT"),
}

__all__: tuple[str, ...] = ("__version__", *tuple(sorted(_EXPORTS)))
__version__ = "0.1.0"


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(_import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
