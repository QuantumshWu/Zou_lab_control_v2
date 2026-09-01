def test_every_window_is_nobody_s_property() -> None:
    """One window law, after two bench incidents proved the other one wrong.

    A window opened from another used to be Qt-parented to it with the
    Window flag applied afterwards, to keep the pair adjacent on the
    desktop.  But a frameless window realizes its native handle inside
    __init__, so the late flag re-created the handle underneath it -- and
    on the bench that construct twice cascaded closes the WRONG way:
    closing one owned log window froze its siblings, and closing one
    owned device control reached the console, whose close guard shut the
    running installation mid-UART-transfer.

    So: every launcher window is a plain top-level and no opener takes an
    owner.  The one surface that genuinely belongs elsewhere -- the panel
    Setting frame -- is not a window at all: a FluentOverlayFrame is an
    ordinary child of the panel page, so stacking, tab visibility and
    shared minimise/close all follow from parenthood (pinned by the
    console view tests).  Modal dialogs anchor where they appear without
    any native ownership either.
    """

    import inspect

    from zlc_ui import windows as window_module
    from zlc_ui.fluent import open_fluent_window
    from zlc_ui.fluent.fluent import FluentDialogWindow, FluentWindow, launch_fluent_window

    for opener in (
        window_module.open_device_control,
        window_module.open_pulse_editor,
        window_module.open_figure_viewer,
        window_module.open_device_manager,
        window_module.open_task_console,
        open_fluent_window,
        launch_fluent_window,
    ):
        signature = inspect.signature(opener)
        assert "owner" not in signature.parameters, (
            f"{opener.__name__} must not take an owner: it opens a window "
            "that belongs to nobody"
        )
    assert "parent" not in inspect.signature(FluentWindow.__init__).parameters
    assert "parent" not in inspect.signature(FluentDialogWindow.__init__).parameters


def test_window_flags_are_never_changed_after_construction() -> None:
    """The trap stays mechanically shut.

    qframelesswindow realizes the native handle during __init__, so any
    later setWindowFlag(s) on such a window destroys and lazily re-creates
    the handle with none of the frameless registrations reapplied -- the
    root of both bench cascade incidents.  The one permitted site is the
    rounded QMenu, whose native window does not exist until it first pops
    up, so its flags in __init__ land before realization.
    """

    import re
    from pathlib import Path

    import zlc_ui

    root = Path(zlc_ui.__file__).resolve().parent
    offenders: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"setWindowFlags?\(", line):
                offenders.append((path.name, line.strip()))
    assert len(offenders) == 1, offenders
    name, line = offenders[0]
    # The QMenu composes its flags on top of the existing ones, before its
    # native window exists (a QMenu realizes only when it first pops up).
    assert name == "fluent.py" and "windowFlags() | " in line, offenders
