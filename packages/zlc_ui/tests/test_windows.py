

def test_a_window_opened_from_another_belongs_to_it() -> None:
    """Ownership is what keeps a stack a stack.

    Two windows of one application with no relation between them are two
    unrelated top-levels, and the desktop may put anything -- a browser, a
    terminal -- BETWEEN them.  An operator then has a console underneath,
    somebody else's window in the middle, and the settings frame that
    belongs to the console on top, which is not a stack they asked for and
    not one they can fix by clicking.  A window opened FROM another is
    parented to it, and stays a real window while it is.
    """

    from PyQt5 import QtCore, QtWidgets

    from zlc_ui import ensure_qt_app
    from zlc_ui.fluent import open_fluent_window

    ensure_qt_app([])
    owner = open_fluent_window(
        lambda: QtWidgets.QLabel("owner"), title="owner"
    )
    try:
        owned = open_fluent_window(
            lambda: QtWidgets.QLabel("owned"), title="owned", owner=owner
        )
        try:
            assert owned.parent() is owner, (
                "a window opened from another must belong to it"
            )
            assert owned.isWindow(), (
                "owning it must not turn it into a child widget inside its "
                "owner -- it is still its own movable, closable window"
            )
        finally:
            owned.close()
    finally:
        owner.close()


def test_an_application_window_is_nobody_s_property() -> None:
    """Ownership is for a frame, never for an application.

    A frame has no life of its own: a console opens it to show the
    settings of a card it is already showing, so the two belong together.
    The pulse editor, the figure viewer, the device manager and the
    console each open on their own, outlive whatever launched them, and
    are raised, minimised and closed on their own.  Tying one to another
    means minimising a console takes an editor down with it -- a
    relationship neither of them has, and one an operator cannot undo.
    """

    import inspect

    from zlc_ui import windows as window_module

    signature = inspect.signature(window_module.open_device_control)
    assert "owner" in signature.parameters, (
        "a frame opened from a console must be able to say whose it is"
    )
    for name in (
        "open_pulse_editor",
        "open_figure_viewer",
        "open_device_manager",
        "open_task_console",
    ):
        signature = inspect.signature(getattr(window_module, name))
        assert "owner" not in signature.parameters, (
            f"{name} opens an application window: it belongs to nobody"
        )
