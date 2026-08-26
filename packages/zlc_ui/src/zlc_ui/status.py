"""The status vocabulary, owned apart from anything that draws it.

A status severity is a plain word, and the two sides that must agree on it
sit on opposite sides of the Qt boundary: the strip that colours it, and the
headless presenters that choose it.  It used to be neither's -- bare strings
written at every call site and checked only inside the widget, by raising --
so one mistyped word ("info", which is the Fluent layer's name for idle)
travelled from a panel edit into a Qt slot and took the whole console down
with it.  Here it belongs to the package both sides already depend on, in a
module that imports nothing, so referencing it costs no Qt.
"""

from __future__ import annotations

#: Every severity a console status line may carry, worst last.
STATUS_SEVERITIES: tuple[str, ...] = ("idle", "warning", "task", "error")

__all__ = ["STATUS_SEVERITIES"]
