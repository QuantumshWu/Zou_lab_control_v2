"""The composition root: the only package allowed to know all the others.

Presenters live here, wiring zlc_ui's mute views to the runtime, the domain and
the plotting session. So do the application entry points and the cross-package
end-to-end tests. Nothing else may: a domain rule, a rendering decision or a
signal mechanism that appears in this package is misplaced, and belongs to
whichever package owns that subject.

The notebook and the GUI drive the SAME session facade. If those two ever grow
separate paths, the whole point of splitting the packages is lost.
"""

from __future__ import annotations

__all__: list[str] = []
