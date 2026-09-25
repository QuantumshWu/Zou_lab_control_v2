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

    def __init__(self, parent=None, *, action_text: str = "") -> None:
        super().__init__(parent, action_text=action_text)
        #: What an empty message falls back to.
        self._idle_text = ""

    def show_status(self, text: str, severity: str) -> None:
        """Show one line; the strip itself refuses a word outside the vocabulary."""

        severity = str(severity)
        value = str(text)
        # An empty message at any severity means "nothing to say", which is the
        # idle state rather than a reason to bring an older line back.
        if not value:
            severity, value = "idle", self._idle_text
        elif severity == "idle":
            self._idle_text = value
        super().show_message(value, severity=severity)


__all__ = ["StatusStrip"]
