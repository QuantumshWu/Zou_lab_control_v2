"""Priority-aware status chrome for the console shell."""

from __future__ import annotations

from zlc_ui.fluent import FluentStatusStrip


class StatusStrip(FluentStatusStrip):
    """Show what just happened, coloured by how bad it was.

    It used to keep one message per severity and always show the worst, which
    reads well for a burst inside one action and is wrong for everything after
    it: nothing ever retired an error, so the first failure stayed on screen
    forever and every later success -- "saved 3 panel(s)", "camera started" --
    was written into a slot the strip would not display.  An operator fixed the
    problem, pressed the button, and saw the old error.

    A status line is a fact about the last thing that happened.  The newest one
    is the true one, and its severity says how to colour it.
    """


    _PRIORITY = {"idle": 0, "warning": 1, "task": 2, "error": 3}
    _FLUENT_SEVERITY = {"idle": "info", "warning": "warning", "task": "task", "error": "error"}

    def __init__(self, parent=None, *, action_text: str = "") -> None:
        super().__init__(parent, action_text=action_text)
        #: What an empty message falls back to.
        self._idle_text = ""
        self._current_severity = "idle"
        # ``FluentStatusStrip.__init__`` initializes its own empty message
        # before this subclass has the priority state.  Re-apply the public
        # idle path so a freshly constructed console has the same left-anchored
        # empty marker as the v1 status strip.
        self.show_status("", "idle")

    @property
    def current_severity(self) -> str:
        return self._current_severity

    def show_status(self, text: str, severity: str) -> None:
        severity = str(severity)
        if severity not in self._PRIORITY:
            raise ValueError("severity must be idle, warning, task, or error")
        value = str(text)
        # An empty message at any severity means "nothing to say", which is the
        # idle state rather than a reason to bring an older line back.
        if not value:
            severity, value = "idle", self._idle_text
        elif severity == "idle":
            self._idle_text = value
        self._current_severity = severity
        super().show_message(value, severity=self._FLUENT_SEVERITY[severity])


__all__ = ["StatusStrip"]
