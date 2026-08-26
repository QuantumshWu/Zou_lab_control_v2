

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


def test_every_console_window_states_who_owns_it() -> None:
    """The openers cannot be used without answering the question.

    Ownership is not something a new window may quietly omit: every shared
    opener takes it, so adding one makes the author say who it belongs to.
    """

    import inspect

    from zlc_ui import windows as window_module

    for name in (
        "open_device_control",
        "open_pulse_editor",
        "open_figure_viewer",
        "open_device_manager",
        "open_task_console",
    ):
        signature = inspect.signature(getattr(window_module, name))
        assert "owner" in signature.parameters, name
