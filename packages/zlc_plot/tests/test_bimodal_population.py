"""The bimodal histogram models in physical parameters, and the question a
two-population model asks of its data.

A two-population model always finds two populations, because it has the
parameters for them.  Whether the data has two is answered by fitting the
nested one-population model as well and weighing the evidence -- the BIC
gain of two over one, ``DECISIVE_BIC_GAIN`` by default -- and where it
falls short the nested answer stands, written in the wider model's own
parameters with the two components coinciding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_plot import HistogramPlot, PlotSession
from zlc_plot.fit import (
    DECISIVE_BIC_GAIN,
    FitEngine,
    FitOptions,
    _bimodal_classifier_metrics,
    builtin_fit_models,
)


def _histogram(samples: np.ndarray, bins: int = 60) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(samples, bins=bins)
    return 0.5 * (edges[1:] + edges[:-1]), counts.astype(np.float64)


def _shots(rng: np.random.Generator, count: int, bright_fraction: float) -> np.ndarray:
    bright = rng.random(count) < bright_fraction
    return np.where(bright, rng.normal(250.0, 25.0, count), rng.normal(100.0, 10.0, count))


def test_the_histogram_models_are_written_in_physical_parameters() -> None:
    """An amplitude (the shots times the bin), a centre, a width and a flat
    background; then an offset, a width and a share.  Nothing is a peak
    height or a midpoint."""

    models = {model.model_id: model for model in builtin_fit_models()}
    assert models["histogram_gaussian"].parameter_names == (
        "amplitude", "center", "sigma", "background"
    )
    assert models["bimodal_gaussian"].parameter_names == (
        "amplitude", "center", "sigma", "delta_center", "sigma_B", "ratio", "background"
    )
    assert models["histogram_poisson_gaussian"].parameter_names == (
        "amplitude", "rate", "sigma", "background"
    )
    assert models["bimodal_poisson_gaussian"].parameter_names == (
        "amplitude", "rate", "sigma", "delta_rate", "sigma_B", "ratio", "background"
    )
    for model_id in ("bimodal_gaussian", "bimodal_poisson_gaussian"):
        model = models[model_id]
        assert model.headline == "ratio"
        assert model.reduction is not None
        assert model.reduction.nested_model_id == model_id.replace(
            "bimodal_", "histogram_"
        )
        assert model.symbols == (
            "Nw", "lambda" if "poisson" in model_id else "x_0", "sigma", "delta", "sigma_B", "r", "beta"
        )


def test_a_single_gaussian_is_the_shots_times_the_bin() -> None:
    rng = np.random.default_rng(3)
    samples = rng.normal(100.0, 10.0, 2000)
    centres, counts = _histogram(samples, bins=50)
    result = FitEngine().fit("histogram_gaussian", (centres,), counts, data_revision=0)
    assert result.success
    assert result.fixed_parameter_names == ()
    bin_width = float(centres[1] - centres[0])
    assert result.parameters["amplitude"] == pytest.approx(2000.0 * bin_width, rel=0.03)
    assert result.parameters["center"] == pytest.approx(100.0, abs=1.0)
    assert result.parameters["sigma"] == pytest.approx(10.0, rel=0.05)
    # The formula states the model: one amplitude, the shots times the bin,
    # over a flat background that a clean histogram leaves at nothing.
    assert result.model.symbols == ("Nw", "x_0", "sigma", "beta")
    assert result.parameters["background"] < 0.5


def test_a_histogram_fit_stays_within_what_the_histogram_shows() -> None:
    # Two populations far apart, fitted as one.  The honest single Gaussian
    # is a slope, and with nothing to stop it the fit runs out along the
    # axis, its shots growing without bound (5e13 of them centred at -1.3e5,
    # from 120 shots binned between 0 and 4000): a tail is flatter across
    # the empty valley than any hump.  The histogram's edges and breadth
    # are where a population it can show ends.
    rng = np.random.default_rng(7)
    centres, counts = _histogram(_shots(rng, 120, 0.4), bins=40)
    step = float(centres[1] - centres[0])
    low_edge, high_edge = centres[0] - 0.5 * step, centres[-1] + 0.5 * step
    engine = FitEngine()
    for model, options in (
        ("histogram_gaussian", None),
        ("bimodal_gaussian", FitOptions(min_bic_gain=1.0e6)),
    ):
        result = engine.fit(model, (centres,), counts, options=options)
        assert result.success
        assert result.reduced is (model == "bimodal_gaussian")
        assert low_edge <= result.parameters["center"] <= high_edge
        assert 0.5 * step <= result.parameters["sigma"] <= high_edge - low_edge
        assert result.parameters["amplitude"] <= 10.0 * counts.sum() * step
    # A bound the caller asks for meets the histogram's: a tighter one is
    # kept, one with nothing inside the histogram's limit is refused.
    kept = engine.fit(
        "histogram_gaussian", (centres,), counts, bounds={"center": (150.0, 300.0)}
    )
    assert 150.0 <= kept.parameters["center"] <= 300.0
    with pytest.raises(ValueError, match="location on the histogram's axis"):
        engine.fit(
            "histogram_gaussian",
            (centres,),
            counts,
            bounds={"center": (high_edge + 10.0, high_edge + 20.0)},
        )
    with pytest.raises(ValueError, match="width the histogram resolves"):
        engine.fit(
            "histogram_gaussian",
            (centres,),
            counts,
            bounds={"sigma": (2.0 * (high_edge - low_edge), None)},
        )
    with pytest.raises(ValueError, match="length on the histogram's axis"):
        engine.fit(
            "bimodal_gaussian",
            (centres,),
            counts,
            bounds={"delta_center": (2.0 * (high_edge - low_edge), None)},
        )


def test_two_populations_are_reported_with_their_share_and_evidence() -> None:
    rng = np.random.default_rng(11)
    centres, counts = _histogram(_shots(rng, 400, 0.3))
    result = FitEngine().fit("bimodal_gaussian", (centres,), counts, data_revision=0)
    assert result.success and not result.reduced
    assert result.evidence > DECISIVE_BIC_GAIN
    values = result.parameters
    assert values["center"] == pytest.approx(100.0, abs=3.0)
    assert values["center"] + values["delta_center"] == pytest.approx(250.0, abs=8.0)
    assert values["sigma"] == pytest.approx(10.0, rel=0.25)
    assert values["sigma_B"] == pytest.approx(25.0, rel=0.25)
    assert values["ratio"] == pytest.approx(0.3, abs=0.05)
    assert values["amplitude"] == pytest.approx(400.0 * (centres[1] - centres[0]), rel=0.05)
    assert result.fixed_parameter_names == ()
    threshold, left, right, fidelity = _bimodal_classifier_metrics(result)
    assert threshold is not None and 120.0 < threshold < 220.0
    assert right == pytest.approx(values["ratio"], abs=0.02)
    assert fidelity > 0.99


def test_counts_between_the_peaks_are_background_not_a_population() -> None:
    """Sixty shots, four of them lost part-way through the exposure, and the
    two-population fit turned a population into a plateau across the axis:
    a count between the peaks was impossible to it.  The flat background,
    the mixture's noise component, gives those counts somewhere to be --
    and because the one-population model carries it too, they are no
    evidence of a second population on their own."""

    rng = np.random.default_rng(12)
    bright = rng.random(60) < 0.4
    shots = np.where(bright, rng.normal(3400.0, 90.0, 60), rng.normal(50.0, 40.0, 60))
    samples = np.concatenate([shots, rng.uniform(600.0, 2800.0, 4)])
    centres, counts = _histogram(samples, bins=40)
    engine = FitEngine()
    result = engine.fit("bimodal_gaussian", (centres,), counts)
    assert result.success and not result.reduced
    values = result.parameters
    assert values["sigma"] < 150.0 and values["sigma_B"] < 250.0
    assert values["center"] == pytest.approx(50.0, abs=40.0)
    assert values["center"] + values["delta_center"] == pytest.approx(3400.0, abs=100.0)
    assert values["ratio"] == pytest.approx(float(bright.sum()) / 60.0, abs=0.08)
    assert values["background"] > 0.0
    threshold, left, right, fidelity = _bimodal_classifier_metrics(result)
    assert threshold is not None
    assert values["center"] + 2.0 * values["sigma"] < threshold
    assert threshold < values["center"] + values["delta_center"] - 2.0 * values["sigma_B"]
    assert right == pytest.approx(values["ratio"], abs=0.03)
    assert fidelity > 0.99
    # One population and the same four stray counts: background, not a
    # second population.
    dark = np.concatenate([rng.normal(50.0, 40.0, 60), rng.uniform(600.0, 2800.0, 4)])
    centres, counts = _histogram(dark, bins=40)
    alone = engine.fit("bimodal_gaussian", (centres,), counts)
    assert alone.success and alone.reduced
    assert alone.parameters["background"] > 0.0
    assert _bimodal_classifier_metrics(alone)[0] is None


def test_one_population_stands_with_its_components_coinciding() -> None:
    """A single Gaussian of three hundred shots used to come back as a broad
    component with a narrow one on its tail, and a threshold through it."""

    rng = np.random.default_rng(11)
    centres, counts = _histogram(rng.normal(100.0, 10.0, 300))
    engine = FitEngine()
    result = engine.fit("bimodal_gaussian", (centres,), counts, data_revision=0)
    assert result.success and result.reduced
    assert result.evidence < DECISIVE_BIC_GAIN
    values = result.parameters
    assert values["delta_center"] == 0.0
    assert values["sigma_B"] == values["sigma"]
    assert values["ratio"] == 0.5
    single = engine.fit("histogram_gaussian", (centres,), counts, data_revision=0)
    for name in ("amplitude", "center", "sigma", "background"):
        assert values[name] == single.parameters[name]
        assert result.errors[name] == single.errors[name]
    np.testing.assert_array_equal(result.fitted_values, single.fitted_values)
    # The pinned parameters were fitted by nobody: no error bar stands on them.
    assert result.fixed_parameter_names == ("delta_center", "sigma_B", "ratio")
    assert not result.parameter_error_validity["ratio"]
    assert "one population" in result.message
    assert _bimodal_classifier_metrics(result)[0] is None


def test_the_evidence_the_operator_demands_decides() -> None:
    rng = np.random.default_rng(5)
    centres, counts = _histogram(_shots(rng, 400, 0.3))
    engine = FitEngine()
    weighed = engine.fit("bimodal_gaussian", (centres,), counts)
    assert not weighed.reduced and weighed.evidence > 100.0
    demanding = engine.fit(
        "bimodal_gaussian",
        (centres,),
        counts,
        options=FitOptions(min_bic_gain=weighed.evidence + 1.0),
    )
    assert demanding.reduced
    assert demanding.evidence == pytest.approx(weighed.evidence)
    unasked = engine.fit(
        "bimodal_gaussian", (centres,), counts, options=FitOptions(min_bic_gain=None)
    )
    assert not unasked.reduced and math.isnan(unasked.evidence)
    np.testing.assert_allclose(unasked.parameter_values, weighed.parameter_values)


def test_a_batch_of_cells_is_weighed_cell_by_cell() -> None:
    rng = np.random.default_rng(8)
    two = _histogram(_shots(rng, 400, 0.4))
    one = _histogram(rng.normal(100.0, 10.0, 400))
    results, failures = FitEngine().fit_batch(
        "bimodal_gaussian",
        ((two[0],), (one[0],)),
        (two[1], one[1]),
    )
    assert failures == (None, None)
    assert results[0] is not None and not results[0].reduced
    assert results[1] is not None and results[1].reduced


def test_a_one_state_photon_histogram_reduces_to_one_poisson_gaussian() -> None:
    rng = np.random.default_rng(6)
    photons = rng.poisson(6.0, 800) + rng.normal(0.0, 0.8, 800)
    centres, counts = _histogram(photons, bins=48)
    result = FitEngine().fit("bimodal_poisson_gaussian", (centres,), counts)
    assert result.success and result.reduced
    assert result.parameters["delta_rate"] == 0.0
    assert result.parameters["sigma_B"] == result.parameters["sigma"]
    assert result.parameters["ratio"] == 0.5
    assert result.parameters["rate"] == pytest.approx(6.0, abs=0.5)


def _bimodal_session(samples: np.ndarray, **parameters) -> PlotSession:
    schema = make_dataset_schema(
        repeat_domain(size=samples.size),
        mapped_domain_from_columns({"sample": (0.0,)}),
        dtype=np.float64,
    )
    return PlotSession(
        make_snapshot(schema, samples[:, None], revision=0),
        HistogramPlot(),
        parameters=parameters,
    )


def test_the_overlay_says_how_the_question_was_decided() -> None:
    rng = np.random.default_rng(9)
    session = _bimodal_session(rng.normal(100.0, 10.0, 300))
    try:
        result = session.fit("bimodal_gaussian", live=False)
        assert result.reduced
        overlay = session._projected._make_fit_overlay(
            result, session.fit_selection("bimodal_gaussian")
        )
        assert overlay.evidence.startswith("ΔBIC = ")
        assert overlay.evidence.endswith("one population")
        # The amplitude is counts times the value axis; a unitless axis
        # multiplies nothing (this printed "count*1").
        amplitude_row = next(
            row for row in overlay.parameter_display if row.name == "amplitude"
        )
        assert amplitude_row.unit == "count"
    finally:
        session.close()
    session = _bimodal_session(_shots(rng, 400, 0.5))
    try:
        result = session.fit("bimodal_gaussian", live=False)
        assert not result.reduced
        overlay = session._projected._make_fit_overlay(
            result, session.fit_selection("bimodal_gaussian")
        )
        assert overlay.evidence.endswith("two populations")
    finally:
        session.close()


def test_the_threshold_is_a_fit_setting_of_the_model_that_asks() -> None:
    rng = np.random.default_rng(10)
    session = _bimodal_session(_shots(rng, 400, 0.5))
    try:
        session.configure(fit={"model": "bimodal_gaussian", "min_bic_gain": 25.0}, fit_live=False)
        described = session.describe_display().fit
        assert described["min_bic_gain"] == 25.0
        request = session._accepted_fit.request
        assert request.options is not None and request.options.min_bic_gain == 25.0
        session.configure(fit={"model": "bimodal_gaussian"}, fit_live=False)
        assert session.describe_display().fit["min_bic_gain"] == DECISIVE_BIC_GAIN
        with pytest.raises(TypeError, match="has no min_bic_gain"):
            session.configure(
                fit={"model": "histogram_gaussian", "min_bic_gain": 5.0}, fit_live=False
            )
    finally:
        session.close()
