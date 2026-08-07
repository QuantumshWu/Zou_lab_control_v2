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

from pathlib import Path as _Path

__version__ = "0.1.0"
_PACKAGE_DIR = _Path(__file__).resolve().parent
if _PACKAGE_DIR.name != "zlc_workbench" or __package__ != "zlc_workbench":
    raise ImportError(f"unexpected zlc_workbench installation path: {_PACKAGE_DIR}")

__all__ = ["__version__"]
