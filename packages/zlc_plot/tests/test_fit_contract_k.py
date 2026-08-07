from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from test_facet_live_fit import _facet_snapshot, _spec as facet_spec
from test_units import _session as unit_session
from zlc_plot import AxisRef, CurvePlot, PlotSession
from zlc_plot.fit import (
    FacetFitBatchResult,
    FitEngine,
    FitModelSpec,
    FitParameterSpec,
    FitTarget,
    VALUE,
)


def test_fit_table_reuses_one_builder_for_scalar_and_facet(monkeypatch) -> None:
    fit_module = import_module("zlc_plot.fit")
    original = fit_module._make_fit_numeric_table
    calls: list[tuple[str, int]] = []

    def wrapped(**kwargs):  # type: ignore[no-untyped-def]
        calls.append((kwargs["sample_axis_name"], kwargs["batch_revision"]))
        return original(**kwargs)

    monkeypatch.setattr(fit_module, "_make_fit_numeric_table", wrapped)
    facet_session = PlotSession(_facet_snapshot(), facet_spec())
    scalar_session = PlotSession(_facet_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        facet_result = facet_session.fit("gaussian_offset", live=True)
        scalar_result = scalar_session.fit("gaussian_offset", live=False)
        assert isinstance(facet_result, FacetFitBatchResult)
        facet_result.table
        scalar_result.table
        assert calls == [("facet", 1), ("", 1)]
    finally:
        facet_session.close()
        scalar_session.close()


def test_repeated_fit_batches_have_stable_source_and_monotonic_publication_revision() -> None:
    session = PlotSession(_facet_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        results = [session.fit("gaussian_offset", live=False) for _ in range(3)]
        tables = [result.table for result in results]
        assert {table.source_revision for table in tables} == {0}
        assert [table.batch_revision for table in tables] == [1, 2, 3]
    finally:
        session.close()


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


def test_canonical_fit_values_and_units_ignore_display_unit_changes() -> None:
    session = unit_session()
    try:
        baseline = session.fit("gaussian_offset", live=False).table
        baseline_values = {
            name: values.copy() for name, values in baseline.parameter_values.items()
        }
        baseline_units = dict(baseline.parameter_units)
        for x_unit, value_unit in (("cm", "mV"), ("mm", "uV")):
            session.set_axis_unit(AxisRef.point("x"), x_unit)
            session.set_value_unit(value_unit)
            current = session.fit("gaussian_offset", live=False).table
            assert dict(current.parameter_units) == baseline_units
            for name in current.parameter_names:
                assert np.array_equal(current.parameter_values[name], baseline_values[name])
    finally:
        session.close()


def test_successful_fit_with_invalid_covariance_has_invalid_error_columns() -> None:
    session = unit_session()
    try:
        result = session.fit("gaussian_offset", live=False)
        invalid = replace(result, covariance_valid=False)
        table = invalid.table
        assert bool(table.success[0])
        for name in table.parameter_names:
            assert not bool(table.parameter_error_validity[name][0])
            assert np.isnan(table.parameter_errors[name][0])
    finally:
        session.close()


def test_failed_scalar_table_masks_values_and_errors() -> None:
    session = unit_session()
    try:
        result = session.fit("gaussian_offset", live=False)
        failed = replace(result, success=False)
        table = failed.table
        assert not bool(table.success[0])
        for name in table.parameter_names:
            assert np.isnan(table.parameter_values[name][0])
            assert np.isnan(table.parameter_errors[name][0])
            assert not bool(table.parameter_error_validity[name][0])
    finally:
        session.close()


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


def test_single_and_one_cell_facet_use_identical_numeric_columns() -> None:
    x = np.linspace(-3.0, 3.0, 21)
    values = 2.0 * np.exp(-0.5 * ((x - 0.2) / 1.0) ** 2) + 0.2
    scalar_schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x}),
        dtype=np.float64,
        generation="single-facet-equivalence",
    )
    facet_schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x, "facet": np.zeros(x.size)}),
        dtype=np.float64,
        generation="single-facet-equivalence",
    )
    scalar = PlotSession(
        DatasetSnapshot(scalar_schema, values[None, :], revision=0),
        CurvePlot(AxisRef.point("x")),
    )
    facet = PlotSession(
        DatasetSnapshot(facet_schema, values[None, :], revision=0),
        facet_spec(),
    )
    try:
        scalar_table = scalar.fit("gaussian_offset", live=False).table
        facet_result = facet.fit(
            "gaussian_offset",
            live=False,
            fit_all_facets=True,
        )
        assert isinstance(facet_result, FacetFitBatchResult)
        facet_table = facet_result.table
        assert np.array_equal(scalar_table.success, facet_table.success)
        for name in scalar_table.parameter_names:
            np.testing.assert_allclose(
                scalar_table.parameter_values[name],
                facet_table.parameter_values[name],
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                scalar_table.parameter_errors[name],
                facet_table.parameter_errors[name],
                rtol=1e-12,
                atol=1e-12,
            )
            assert scalar_table.parameter_units[name] == facet_table.parameter_units[name]
            assert np.array_equal(
                scalar_table.parameter_error_validity[name],
                facet_table.parameter_error_validity[name],
            )
        assert scalar_table.source_revision == facet_table.source_revision == 0
        assert scalar_table.sample_unit == facet_table.sample_unit == ""
    finally:
        scalar.close()
        facet.close()


def test_fit_module_keeps_the_data_contract_import_free() -> None:
    source = Path(__file__).parents[1] / "src" / "zlc_plot" / "fit.py"
    assert "import zlc_data" not in source.read_text(encoding="utf-8")
