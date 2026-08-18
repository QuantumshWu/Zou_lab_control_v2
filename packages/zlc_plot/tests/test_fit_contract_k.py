from __future__ import annotations

from importlib import import_module
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
        return SimpleNamespace(
            success=True,
            message="ok",
            x=x0,
            fun=np.zeros(5) if good else np.ones(5),
            jac=np.ones((5, 1)),
        )

    fit_module = import_module("zlc_plot.fit")
    monkeypatch.setattr(fit_module, "least_squares", fake_least_squares)
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
