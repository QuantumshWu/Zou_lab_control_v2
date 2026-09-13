from __future__ import annotations

from importlib import import_module
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from test_facet_live_fit import _facet_snapshot, _spec as facet_spec
from zlc_plot import AxisRef, CurvePlot, PlotSession
from zlc_plot.fit import (
    FitEngine,
    FitModelSpec,
    FitParameterSpec,
    FitTarget,
    VALUE,
)


def test_scalar_and_facet_fit_results_share_source_revision_accessor() -> None:
    scalar = PlotSession(_facet_snapshot(), CurvePlot(AxisRef.point("x")))
    facet = PlotSession(_facet_snapshot(), facet_spec())
    try:
        scalar_result = scalar.fit("gaussian_offset", live=False)
        facet_result = facet.fit("gaussian_offset", live=False)
        assert hasattr(scalar_result, "source_revision")
        assert hasattr(facet_result, "source_revision")
        assert scalar_result.source_revision == facet_result.source_revision == 0
        assert not hasattr(scalar_result, "data_revision")
    finally:
        scalar.close()
        facet.close()


def test_warm_start_is_extra_candidate_not_an_early_success_exit(monkeypatch) -> None:
    def evaluator(x, value):
        return np.asarray(x, dtype=float) * float(value)

    def initializer(_coordinates, _values):
        return (0.0,)

    def candidates(_coordinates, _values):
        return ((1.0,), (0.0,))

    model = FitModelSpec(
        "warm_candidate_test",
        "Warm candidate test",
        1,
        (FitParameterSpec("value", VALUE),),
        "value",
        evaluator,
        initializer,
        (FitTarget.SERIES,),
        candidate_initializer=candidates,
    )

    def fake_least_squares(_residual, x0, **_kwargs):
        x0 = np.asarray(x0, dtype=float)
        good = np.isclose(x0[0], 1.0)
        fun = np.zeros(5) if good else np.ones(5)
        # SciPy's result carries the minimised cost beside the residual;
        # candidates compete on it, so the double reports it the same way.
        return SimpleNamespace(
            success=True,
            message="ok",
            x=x0,
            fun=fun,
            cost=0.5 * float(np.dot(fun, fun)),
            jac=np.ones((5, 1)),
        )

    # Where the solver LIVES: the engine imports it inside the function
    # that solves, so that reading the catalogue costs no scipy, and there
    # is no copy bound onto the engine's module to replace instead.
    monkeypatch.setattr(
        import_module("scipy.optimize"), "least_squares", fake_least_squares
    )
    fit_module = import_module("zlc_plot.fit")
    registry = fit_module.FitModelRegistry((model,))
    result = FitEngine(registry).fit(
        model,
        (np.ones(5),),
        np.zeros(5),
        warm_start=(0.0,),
    )
    assert result.success
    assert np.allclose(result.parameter_values, [1.0])


def test_fit_module_keeps_the_data_contract_import_free() -> None:
    source = Path(__file__).parents[1] / "src" / "zlc_plot" / "fit.py"
    assert "import zlc_data" not in source.read_text(encoding="utf-8")


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
