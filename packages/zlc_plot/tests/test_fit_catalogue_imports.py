from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_reading_the_catalogue_does_not_import_the_solver() -> None:
    """What a model DECLARES is not what it takes to solve one.

    A task console lists the parameters a panel publishes -- names and
    symbols, straight out of the catalogue -- and restores a saved panel's
    fit against them, and never solves anything; the render children solve
    and never list.  Importing this module used to cost scipy.optimize and
    scipy.signal either way, 0.64 s, so the GUI learned a parameter's name
    by loading a least-squares solver.  The solvers are reached from inside
    the functions that use them, and a child that will solve warms them on
    purpose instead.

    That includes the compiled engine.  A model's compiled descriptor is
    part of what it declares, and while the catalogue held one it imported
    numba and llvmlite to be read: loading a board with one fitted panel put
    them in the console for the rest of its life.  The catalogue names the
    descriptor now, and the first solve builds it.
    """

    script = (
        "import sys" + chr(10)
        + "import zou_lab_control" + chr(10)
        + "from zlc_plot.fit import FitOptions, builtin_fit_models, default_fit_registry" + chr(10)
        + "models = builtin_fit_models()" + chr(10)
        + "assert models and all(model.parameters for model in models)" + chr(10)
        + "assert all(default_fit_registry().get(m.model_id).parameter_names for m in models)" + chr(10)
        + "FitOptions(loss='linear')" + chr(10)
        + "reached = sorted(" + chr(10)
        + "    name for name in sys.modules" + chr(10)
        + "    if name.split('.')[0] in ('numba', 'llvmlite')" + chr(10)
        + "    or name == 'zlc_plot._fit_compiled'" + chr(10)
        + "    or (name.split('.')[0] == 'scipy'" + chr(10)
        + "        and name.split('.')[1:2] in (['optimize'], ['signal']))" + chr(10)
        + ")" + chr(10)
        + "assert not reached, reached" + chr(10)
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
