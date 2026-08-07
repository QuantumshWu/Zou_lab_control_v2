"""One test run over eight layers, without renaming anybody's files.

Two things had to be true at once.  Six test filenames repeat across layers --
``test_public_surface.py`` exists in three of them, because each layer guards
its own surface and that is the right name for it -- and pytest's default
import mode derives a module name from the basename, so the second one to be
collected is refused as a mismatch.  Meanwhile a handful of suites import a
SIBLING test module by bare name (``import test_console_presenter``) to reuse
its doubles, which only works while that directory is on ``sys.path``.

So: importlib import mode, which names a module from its whole path and stops
the collision; and this file, which puts every layer's ``tests`` and ``src`` on
the path so the sibling imports still resolve.  Every bare-imported name is
unique across the tree -- checked, not assumed -- so nothing here is ambiguous.

The layers sit under ``packages/`` for a third reason: a folder named
``zlc_data`` at the repo root shadows the package of the same name inside it,
because the root is on ``sys.path`` and an empty directory is a valid namespace
package.  Every submodule import then fails with "No module named
zlc_data.axis".  That is the shadow-import trap this project has already been
bitten by once, and the fix is to not put the two names in the same place.

Renaming the duplicates was the other option.  It would have meant editing
files in eight repositories to work around a tool's default, and the names are
not the problem: ``test_public_surface`` is exactly what each of those is.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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

for _layer in LAYERS:
    for _part in ("src", "tests"):
        _path = ROOT / "packages" / _layer / _part
        if _path.is_dir() and str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
