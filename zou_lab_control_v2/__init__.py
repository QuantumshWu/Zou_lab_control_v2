"""The one entry: import this, and this tree is the code that runs.

There is no install step and nothing on the machine changes.  A notebook, a
launcher or a script says ``import zou_lab_control_v2`` first, and every
``zlc_*`` import after it resolves inside this checkout.

That matters because the alternative has already cost this project a day.
Eight of these layers are also installed as editable distributions pointing at
their standalone repositories, so ``import zlc_data`` in a fresh interpreter
answers with THOSE -- and a change made here appears to do nothing, over and
over, with nothing on screen to say why.  The failure has no symptom except
that the work does not take effect.

So this module does two things, and the second is the important one:

* it puts every layer's ``src`` on ``sys.path``, in dependency order;
* it REFUSES, loudly, if any ``zlc_*`` module was already imported from
  somewhere else.  Arriving second is exactly the case that used to pass
  silently, and a wrong answer you can see beats a wrong answer you cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path


__all__ = ["LAYERS", "ROOT", "layer_source_paths"]

#: The repository root -- the folder holding ``packages/``.
ROOT = Path(__file__).resolve().parent.parent

#: Dependency order, which is also the order they may know each other in.
LAYERS = (
    "zlc_data",
    "zlc_durable",
    "zlc_runtime",
    "zlc_plot",
    "zlc_ui",
    "zlc_pulse",
    "zlc_atom",
    "zlc_workbench",
)


def layer_source_paths() -> tuple[Path, ...]:
    """Where each layer's package lives inside this checkout."""

    return tuple(ROOT / "packages" / layer / "src" for layer in LAYERS)


def _already_loaded_elsewhere() -> tuple[str, ...]:
    """Any zlc_* module already imported from outside this checkout."""

    here = str(ROOT)
    stale: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("zlc_") or "." in name:
            continue
        origin = getattr(module, "__file__", None)
        if origin and not str(Path(origin).resolve()).startswith(here):
            stale.append(f"{name} <- {origin}")
    return tuple(sorted(stale))


def _install_paths() -> None:
    missing = [path for path in layer_source_paths() if not path.is_dir()]
    if missing:
        raise ImportError(
            "this does not look like the Zou lab control checkout: "
            f"missing {', '.join(str(path) for path in missing)}"
        )
    stale = _already_loaded_elsewhere()
    if stale:
        raise ImportError(
            "a zlc_* package was already imported from outside this checkout, "
            "so importing this now would leave two copies in one process:\n  "
            + "\n  ".join(stale)
            + "\n\nImport zou_lab_control_v2 FIRST, before any zlc_* import. "
            "In a notebook that means the very first cell; a kernel that has "
            "already imported one has to be restarted."
        )
    # Reversed, so the first layer ends up first on the path.  Order does not
    # decide correctness here -- the names do not collide -- but a path that
    # reads in dependency order is one a person can check at a glance.
    for path in reversed(layer_source_paths()):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


_install_paths()
