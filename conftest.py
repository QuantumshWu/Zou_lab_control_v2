"""One test run over eight layers, without renaming anybody's files.

Two things had to be true at once.  Six test filenames repeat across layers --
``test_public_surface.py`` exists in three of them, because each layer guards
its own surface and that is the right name for it -- and pytest's default
import mode derives a module name from the basename, so the second one to be
collected is refused as a mismatch.  Meanwhile a handful of suites import a
SIBLING test module by bare name (``import test_console_presenter``) to reuse
its doubles, which only works while that directory is on ``sys.path``.

So: importlib import mode, which names a module from its whole path and stops
the collision; and this file, which brings in the one entry -- importing
``zou_lab_control_v2`` is what puts THIS checkout's layers on the path -- and
then adds the ``tests`` directories the sibling imports need.  Every
bare-imported name is unique across the tree: checked, not assumed.

The path list lives in that package and not here, because a second copy of
"where the layers are" is a second answer waiting to disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import zou_lab_control_v2  # noqa: E402  -- the import IS the path setup

for _layer in zou_lab_control_v2.LAYERS:
    _tests = zou_lab_control_v2.ROOT / "packages" / _layer / "tests"
    if _tests.is_dir() and str(_tests) not in sys.path:
        sys.path.insert(0, str(_tests))
