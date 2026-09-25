from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_reading_the_catalogue_does_not_import_the_solver() -> None:
    """What a model DECLARES is not what it takes to solve one.

    A task console lists the parameters a panel publishes -- names and
    symbols, straight out of the catalogue -- and never solves anything;
    the render children solve and never list.  Importing this module used
    to cost scipy.optimize and scipy.signal either way, 0.64 s, so the GUI
    learned a parameter's name by loading a least-squares solver.  The
    solvers are reached from inside the three functions that use them, and
    a child that will solve warms them on purpose instead.

    The named solver packages, not scipy as a whole: a model's compiled
    descriptor is part of what it declares, so the catalogue does import
    the kernels, and numba brings scipy's own core with it.  That is the
    floor -- 0.41 s, against 1.01 s with the solvers.
    """

    script = (
        "import sys" + chr(10)
        + "import zou_lab_control" + chr(10)
        + "from zlc_plot.fit import builtin_fit_models" + chr(10)
        + "models = builtin_fit_models()" + chr(10)
        + "assert models and all(model.parameters for model in models)" + chr(10)
        + "reached = sorted(" + chr(10)
        + "    name for name in sys.modules" + chr(10)
        + "    if name.split('.')[0] == 'scipy'" + chr(10)
        + "    and name.split('.')[1:2] in (['optimize'], ['signal'])" + chr(10)
        + ")" + chr(10)
        + "assert not reached, reached" + chr(10)
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
