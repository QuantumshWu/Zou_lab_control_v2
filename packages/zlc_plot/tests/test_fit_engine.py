from __future__ import annotations

import json
import math
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from zlc_plot import FitCancelled, _fit_compiled
from zlc_plot.fit import (
    FitEngine,
    FitOptions,
    FitResult,
    RegularImageFitInput,
    _DeferredFitData,
    _FIT_RESULT_RAW,
    builtin_fit_models,
)


_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def _area(height: float, sigma: float) -> float:
    """The area of a Gaussian peak ``height`` counts per bin tall: the
    histogram models' amplitude, the shots times the bin."""

    return height * sigma * _SQRT_TWO_PI


PARAMETERS = {
    "lorentzian": (0.35, 1.2, 2.5, 0.2),
    "gaussian_offset": (2.0, 0.15, 0.9, -0.3),
    "histogram_gaussian": (_area(2.0, 0.9), 0.2, 0.9, 0.0),
    "bimodal_gaussian": (
        _area(1.2, 0.6) + _area(0.9, 0.8),
        -0.7,
        0.6,
        1.4,
        0.8,
        _area(0.9, 0.8) / (_area(1.2, 0.6) + _area(0.9, 0.8)),
        0.0,
    ),
    "symmetric_lorentzian_doublet": (0.1, 1.0, 1.5, 0.1, 1.2),
    "damped_sine": (1.2, 0.1, 1.4, 3.0, 0.2),
    "exponential_decay": (2.2, 0.1, 2.5),
    "radial_gaussian_center": (3.0, 0.2, 0.8, 0.4, -0.3),
    "histogram_poisson_gaussian": (2.0, 1.5, 0.6, 0.0),
    "bimodal_poisson_gaussian": (2.1, 0.8, 0.5, 3.2, 0.7, 0.9 / 2.1, 0.0),
}

_ANCHOR_PATH = Path(__file__).with_name("fixtures") / "fit_anchors.json"

# Models whose origin is the start of the window they are fitted over.
_ANCHORED_MODELS = ("damped_sine", "exponential_decay")
_GENERIC_WARM_MODELS = tuple(
    model for model in PARAMETERS if model != "radial_gaussian_center"
)


def _anchors() -> dict[str, object]:
    # The JSON is a checked-in oracle.  It is intentionally not produced by
    # importing FitModelSpec/evaluate; the evaluator test below is what makes
    # a model mutation fail against the frozen numbers.
    return json.loads(_ANCHOR_PATH.read_text(encoding="utf-8"))["models"]


def _anchor(model: str) -> tuple[tuple[np.ndarray, ...], np.ndarray, tuple[float, ...]]:
    item = _anchors()[model]
    axes = tuple(
        np.asarray(values, dtype=np.float64)
        for values in item["coordinates"].values()
    )
    coordinates = (
        tuple(axis.reshape(-1) for axis in np.meshgrid(*axes))
        if model == "radial_gaussian_center"
        else axes
    )
    observations = np.asarray(item["values"], dtype=np.float64)
    parameters = tuple(float(value) for value in item["parameters"].values())
    return coordinates, observations, parameters


_BUILTIN_MODEL_IDS = (
    "lorentzian",
    "gaussian_offset",
    "histogram_gaussian",
    "bimodal_gaussian",
    "symmetric_lorentzian_doublet",
    "damped_sine",
    "exponential_decay",
    "release_recapture",
    "anisotropic_gaussian_center",
    "radial_gaussian_center",
    "histogram_poisson_gaussian",
    "bimodal_poisson_gaussian",
    "saturation",
)
_HISTOGRAM_MODELS = frozenset((
    "histogram_gaussian",
    "bimodal_gaussian",
    "histogram_poisson_gaussian",
    "bimodal_poisson_gaussian",
))
_POISSON_MODELS = frozenset(
    ("histogram_poisson_gaussian", "bimodal_poisson_gaussian")
)
_BASE_PARAMETERS = {
    "lorentzian": (-0.4, 1.1, 2.2, 0.25),
    "gaussian_offset": (2.0, 0.2, 0.9, -0.3),
    "histogram_gaussian": (_area(90.0, 0.8), -0.3, 0.8, 0.5),
    "bimodal_gaussian": (
        _area(60.0, 0.55) + _area(45.0, 0.75),
        -1.2,
        0.55,
        2.4,
        0.75,
        _area(45.0, 0.75) / (_area(60.0, 0.55) + _area(45.0, 0.75)),
        0.4,
    ),
    "symmetric_lorentzian_doublet": (0.1, 0.8, 1.4, 0.2, 2.5),
    "damped_sine": (1.2, 0.2, 0.25, 6.0, -0.3),
    "exponential_decay": (1.6, 0.2, 3.0),
    "release_recapture": (0.8, 0.05, 6.0, 0.4),
    "anisotropic_gaussian_center": (3.0, 0.2, 0.9, 0.6, 0.35, -0.25),
    "radial_gaussian_center": (3.0, 0.2, 0.8, 0.35, -0.25),
    # Nw is the shots times the bin, the density's area: these put ~60
    # counts in the tallest bin like the Gaussian rows do, over a flat
    # background of a fraction of a count per bin.  The read noise is a fair
    # share of each state's variance (sigma^2 / (rate + sigma^2) of 26%, and
    # 45% / 32%): it is a resolved quantity, the optimum is sharp and two
    # solvers land on the same point.  At a 13% share, three outlier bins
    # were enough to trade the bright state's read noise into its rate
    # (the same total variance), leaving the width on its floor in a flat
    # valley two solvers stop in differently; at twenty photons a
    # 0.3-photon read noise would be a 0.5% share, unidentifiable outright.
    "histogram_poisson_gaussian": (410.0, 4.0, 1.2, 0.3),
    "bimodal_poisson_gaussian": (520.0, 1.0, 0.9, 6.0, 1.8, 320.0 / 520.0, 0.3),
    "saturation": (125.0, 10.0, 2.0),
}


def test_release_recapture_matches_lambert_reference_and_recovers_parameters() -> None:
    from scipy.special import lambertw
    from scipy.optimize._numdiff import approx_derivative

    engine = FitEngine()
    model = engine.registry.get("release_recapture")
    assert model.parameter_names == ("amplitude", "offset", "eta", "frequency")
    t = np.linspace(0.0, 200e-6, 128)
    truth = np.array((0.8, 0.05, 6.0, 40_000.0))

    def reference(parameters):
        amplitude, offset, eta, frequency = parameters
        z = (2.0 * np.pi * frequency * t) ** 2
        q = np.exp(-lambertw(z).real)
        return amplitude * (-np.expm1(-eta * q) / -np.expm1(-eta)) + offset

    observations = reference(truth)
    np.testing.assert_allclose(model.evaluate((t,), truth), observations, rtol=2e-14)
    np.testing.assert_allclose(
        model.evaluate_jacobian((t,), truth),
        approx_derivative(reference, truth, method="3-point"),
        rtol=2e-6, atol=2e-10,
    )
    np.testing.assert_array_equal(
        model.evaluate_jacobian((np.array((0.0,)),), truth),
        np.array(((1.0, 1.0, 0.0, 0.0),)),
    )
    small = truth.copy()
    small[2] = 1e-10
    q = np.exp(-lambertw((2.0 * np.pi * small[3] * t) ** 2).real)
    np.testing.assert_allclose(
        model.evaluate((t,), small), reference(small), rtol=2e-14,
    )
    np.testing.assert_allclose(
        model.evaluate_jacobian((t,), small)[:, 2],
        small[0] * q * (1.0 - q) / 2.0, rtol=1e-9, atol=1e-14,
    )

    fitted = engine.fit(model, (t,), observations)
    assert fitted.success and fitted.covariance_valid
    np.testing.assert_allclose(fitted.parameter_values, truth, rtol=2e-5, atol=1e-7)

    # A/B use the existing exact-fixed-parameter mechanism; the normalized
    # two-parameter model needs neither a second model nor extra options.
    normalized = reference((1.0, 0.0, truth[2], truth[3]))
    fixed = engine.fit(
        model, (t,), normalized,
        bounds={"amplitude": (1.0, 1.0), "offset": (0.0, 0.0)},
    )
    assert fixed.fixed_parameter_names == ("amplitude", "offset")
    np.testing.assert_allclose(fixed.parameter_values, (1.0, 0.0, *truth[2:]), rtol=2e-5)


def test_saturation_response_jacobian_and_fixed_parameters_share_compiled_fit() -> None:
    from scipy.optimize._numdiff import approx_derivative

    engine = FitEngine()
    model = engine.registry.get("saturation")
    assert model.compiled_descriptor is not None
    assert model.parameter_names == ("asymptote", "numerator", "shift")
    assert model.symbols == ("A", "B", "C")
    assert r"\frac" not in model.formula
    ordinary = np.array((125.0, 10.0, 2.0))
    x = np.linspace(0.0, 10.0, 65)
    np.testing.assert_allclose(
        model.evaluate((x,), ordinary), 120.0 * x / (x + 2.0) + 5.0, rtol=2e-15,
    )
    np.testing.assert_array_equal(
        model.evaluate_jacobian((np.array((0.0,)),), ordinary), ((0.0, 0.5, -2.5),)
    )
    assert model.evaluate((np.array((2.0,)),), ordinary)[0] == 65.0
    for x, truth in (
        (x, ordinary),
        (np.linspace(135.0, 247.0, 65), np.array((125.0, -16865.0, -133.0))),
        (x, np.array((5.0, 20.0, 2.0))),
    ):
        def reference(parameters):
            asymptote, numerator, shift = parameters
            return (asymptote * x + numerator) / (x + shift)

        expected = reference(truth)
        np.testing.assert_allclose(model.evaluate((x,), truth), expected, rtol=2e-15)
        np.testing.assert_allclose(
            model.evaluate_jacobian((x,), truth),
            approx_derivative(reference, truth, method="3-point", abs_step=(1e-4, 1e-3, 1e-5)),
            rtol=2e-7, atol=1e-9,
        )
        fitted = engine.fit(model, (x,), expected)
        assert fitted.success and fitted.covariance_valid
        assert np.all(x + fitted.parameters["shift"] > 0.0)
        np.testing.assert_allclose(fitted.parameter_values, truth, rtol=1e-7)
        direction = np.sign(truth[0] * truth[2] - truth[1])
        assert np.all(np.diff(fitted.fitted_values) * direction > 0.0)
        observations = expected + 0.02 * np.sin(np.arange(x.size))
        for fixed_names in (("numerator",), ("shift",), model.parameter_names):
            bounds = {
                name: (value, value)
                for name, value in zip(model.parameter_names, truth, strict=True)
                if name in fixed_names
            }
            single = engine.fit(model, (x,), observations, bounds=bounds)
            batch, failures = engine.fit_batch(
                model, ((x,), (x,)), (observations, observations), bounds=bounds
            )
            assert failures == (None, None)
            for result in batch:
                assert result is not None and result.success
                _assert_fit_equal(result, single)
                assert result.fixed_parameter_names == fixed_names
                assert np.all(x + result.parameters["shift"] > 0.0)
                for name in fixed_names:
                    assert result.parameters[name] == truth[model.parameter_names.index(name)]
        # Cropping changes the fit domain, never the absolute coordinate origin.
        kept = x > x[0] + 0.1 * (x[-1] - x[0])
        cropped = engine.fit(model, (x[kept],), expected[kept])
        assert cropped.success
        np.testing.assert_allclose(cropped.parameter_values, truth, rtol=1e-6)
    # Negative x is legal to the right of the pole; the pole and its other
    # side are invalid, not an alternative branch the solver may cross into.
    values = model.evaluate((np.array((-3.0, -2.0, -1.0)),), ordinary)
    assert np.isnan(values[:2]).all()
    assert values[2] == -115.0


def _coordinates(model_id: str) -> tuple[np.ndarray, ...]:
    if model_id == "saturation":
        return (np.linspace(0.0, 10.0, 112),)
    if model_id == "release_recapture":
        return (np.linspace(0.0, 3.0, 112),)
    if model_id in _POISSON_MODELS:
        return (np.linspace(-2.0, 16.0, 73),)
    if model_id == "symmetric_lorentzian_doublet":
        return (np.linspace(-6.0, 6.0, 128),)
    if model_id == "damped_sine":
        return (50.0 + np.linspace(0.0, 12.0, 160),)
    if model_id == "exponential_decay":
        return (100.0 + np.linspace(0.0, 10.0, 112),)
    if model_id in {"anisotropic_gaussian_center", "radial_gaussian_center"}:
        x = np.linspace(-2.0, 2.0, 21)
        y = np.linspace(-1.8, 2.2, 19)
        xx, yy = np.meshgrid(x, y)
        return xx.reshape(-1), yy.reshape(-1)
    return (np.linspace(-5.0, 5.0, 112),)


def _cell_parameters(model_id: str, cell: int) -> np.ndarray:
    parameters = np.asarray(_BASE_PARAMETERS[model_id], dtype=np.float64).copy()
    position = (cell - 3.5) / 3.5
    if model_id == "lorentzian":
        parameters[[0, 1, 2]] += (0.7 * position, 0.15 * position, 0.2 * position)
    elif model_id == "gaussian_offset":
        parameters[[0, 2, 3]] += (0.2 * position, 0.15 * position, 0.7 * position)
    elif model_id == "histogram_gaussian":
        parameters[[0, 1, 2]] += (24.0 * position, 0.6 * position, 0.12 * position)
    elif model_id == "bimodal_gaussian":
        parameters += np.asarray(
            (22.5, 0.4, 0.08, 0.25, -0.08, 0.05, 0.01)
        ) * position
    elif model_id == "symmetric_lorentzian_doublet":
        parameters[[0, 1, 2, 4]] += (
            0.5 * position,
            0.1 * position,
            0.15 * position,
            0.3 * position,
        )
    elif model_id == "damped_sine":
        parameters[[0, 2, 3, 4]] += (
            0.15 * position,
            0.025 * position,
            0.6 * position,
            0.35 * position,
        )
    elif model_id == "exponential_decay":
        parameters[[0, 2]] += (0.2 * position, 0.5 * position)
    elif model_id == "release_recapture":
        parameters += np.asarray((0.04, 0.01, 0.5, 0.03)) * position
    elif model_id == "saturation":
        parameters += np.asarray((12.0, 0.4, 0.5)) * position
    elif model_id == "histogram_poisson_gaussian":
        parameters[[0, 1, 2]] += (60.0 * position, 0.8 * position, 0.15 * position)
    elif model_id == "bimodal_poisson_gaussian":
        parameters += np.asarray(
            (50.0, 0.2, 0.08, 0.8, -0.08, 0.05, 0.01)
        ) * position
    elif model_id == "anisotropic_gaussian_center":
        parameters[[0, 2, 3, 4, 5]] += (
            0.3 * position,
            0.12 * position,
            -0.08 * position,
            0.45 * position,
            -0.35 * position,
        )
    else:
        parameters[[0, 2, 3, 4]] += (
            0.3 * position,
            0.12 * position,
            0.45 * position,
            -0.35 * position,
        )
    return parameters


def _fit_case(
    engine: FitEngine,
    model_id: str,
    difficulty: str,
    cell: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray | None, np.ndarray]:
    model = engine.registry.get(model_id)
    coordinates = _coordinates(model_id)
    parameters = _cell_parameters(model_id, cell)
    evaluated_coordinates = (
        (coordinates[0] - float(np.min(coordinates[0])),)
        if model_id in _ANCHORED_MODELS
        else coordinates
    )
    expected = model.evaluate(evaluated_coordinates, parameters).reshape(-1)
    model_index = _BUILTIN_MODEL_IDS.index(model_id)
    random = np.random.default_rng(
        91_000 + 1_000 * model_index + 100 * (difficulty == "hard") + cell
    )
    if model_id in _HISTOGRAM_MODELS:
        observations = random.poisson(np.maximum(expected, 0.01)).astype(np.float64)
        if difficulty == "hard":
            outliers = random.choice(
                observations.size,
                max(1, observations.size // 24),
                replace=False,
            )
            observations[outliers] += 0.2 * max(float(np.max(expected)), 1.0)
        sigma = None
    else:
        scale = max(float(np.ptp(expected)), 1.0)
        deviation = (0.003 if difficulty == "normal" else 0.03) * scale
        observations = expected + random.normal(0.0, deviation, expected.size)
        if difficulty == "hard":
            outliers = random.choice(
                observations.size,
                max(1, observations.size // 20),
                replace=False,
            )
            observations[outliers] += random.normal(
                0.0,
                0.18 * scale,
                outliers.size,
            )
        sigma = np.full(expected.size, deviation, dtype=np.float64)
    selected = np.arange(expected.size, dtype=np.int64) + cell * 10_000
    return coordinates, observations, sigma, selected


def _normalized_error(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.linalg.norm(np.asarray(actual) - np.asarray(expected))
    scale = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(difference / scale)


def _assert_fit_equal(
    actual: FitResult,
    expected: FitResult,
    *,
    exact_message: bool = False,
    quality_tolerance: float = 1e-10,
) -> None:
    assert actual.model.model_id == expected.model.model_id
    assert actual.source_revision == expected.source_revision
    assert actual.success == expected.success
    if exact_message:
        assert actual.message == expected.message
    assert actual.covariance_valid == expected.covariance_valid
    assert actual.fixed_parameter_names == expected.fixed_parameter_names
    if expected.covariance_valid:
        assert _normalized_error(
            actual.parameter_values,
            expected.parameter_values,
        ) <= 1e-7
    assert _normalized_error(
        actual.fitted_values,
        expected.fitted_values,
    ) <= 1e-7
    np.testing.assert_allclose(
        actual.standard_errors,
        expected.standard_errors,
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )
    # Elementwise to a millionth of the matrix's own scale: an off-diagonal
    # element a hundred-thousandth of the largest is rounding, not a
    # disagreement (the two solvers' jacobians agree to 1e-16; the inverse
    # of a matrix conditioned at 1e5 does not).
    covariance_scale = float(np.nanmax(np.abs(expected.covariance), initial=0.0))
    np.testing.assert_allclose(
        actual.covariance,
        expected.covariance,
        rtol=1e-6,
        atol=max(1e-6 * covariance_scale, 1e-10),
        equal_nan=True,
    )
    assert actual.reduced_chi_square == pytest.approx(
        expected.reduced_chi_square,
        rel=quality_tolerance,
        abs=1e-12,
    )
    np.testing.assert_array_equal(actual.selected_indices, expected.selected_indices)


@pytest.mark.parametrize("difficulty", ("normal", "hard"))
@pytest.mark.parametrize("model_id", _BUILTIN_MODEL_IDS)
def test_public_batch_matches_single_for_all_builtins_and_batch_sizes(
    model_id: str,
    difficulty: str,
) -> None:
    """SciPy is the oracle for compiled single and B1/B8/B64 results."""

    engine = FitEngine()
    reference_model = replace(
        engine.registry.get(model_id),
        compiled_descriptor=None,
    )
    cases = tuple(_fit_case(engine, model_id, difficulty, cell) for cell in range(8))
    scalar = tuple(
        engine.fit(
            reference_model,
            coordinates,
            observations,
            observation_sigma=sigma,
            selected_indices=indices,
            data_revision=200 + cell,
        )
        for cell, (coordinates, observations, sigma, indices) in enumerate(cases)
    )
    single = tuple(
        engine.fit(
            model_id,
            coordinates,
            observations,
            observation_sigma=sigma,
            selected_indices=indices,
            data_revision=200 + cell,
        )
        for cell, (coordinates, observations, sigma, indices) in enumerate(cases)
    )
    for result, expected in zip(single, scalar, strict=True):
        _assert_fit_equal(result, expected)

    for batch_size in (1, 8, 64):
        order = tuple(cell % len(cases) for cell in range(batch_size))
        results, failures = engine.fit_batch(
            model_id,
            tuple(cases[cell][0] for cell in order),
            tuple(cases[cell][1] for cell in order),
            observation_sigmas=tuple(cases[cell][2] for cell in order),
            selected_indices=tuple(cases[cell][3] for cell in order),
            data_revisions=tuple(200 + cell for cell in order),
        )
        assert failures == (None,) * batch_size
        for result, cell in zip(results, order, strict=True):
            assert result is not None
            _assert_fit_equal(result, scalar[cell])
            if batch_size == 1:
                assert result.message == single[cell].message


@pytest.mark.parametrize("all_fixed", (False, True), ids=("partial", "all"))
def test_public_batch_fixed_parameters_match_single(all_fixed: bool) -> None:
    engine = FitEngine()
    model = engine.registry.get("gaussian_offset")
    reference_model = replace(model, compiled_descriptor=None)
    cases = tuple(_fit_case(engine, model.model_id, "normal", cell) for cell in range(8))
    if all_fixed:
        bounds = {
            name: (value, value)
            for name, value in zip(
                model.parameter_names,
                _BASE_PARAMETERS[model.model_id],
                strict=True,
            )
        }
        initial = None
    else:
        bounds = {"offset": (0.2, 0.2), "center": (-1.0, 1.0)}
        initial = {"sigma": 0.8}

    expected = tuple(
        engine.fit(
            reference_model,
            coordinates,
            observations,
            initial=initial,
            bounds=bounds,
        )
        for coordinates, observations, _sigma, _indices in cases
    )
    results, failures = engine.fit_batch(
        model,
        tuple(case[0] for case in cases),
        tuple(case[1] for case in cases),
        initial=initial,
        bounds=bounds,
    )
    assert failures == (None,) * len(cases)
    fixed_names = tuple(
        name
        for name in model.parameter_names
        if name in bounds and bounds[name][0] == bounds[name][1]
    )
    fixed_indices = tuple(model.parameter_names.index(name) for name in fixed_names)
    for result, scalar in zip(results, expected, strict=True):
        assert result is not None
        _assert_fit_equal(result, scalar)
        assert result.fixed_parameter_names == fixed_names
        assert np.all(result.standard_errors[list(fixed_indices)] == 0.0)
        assert not any(result.parameter_error_validity[name] for name in fixed_names)
        if all_fixed:
            assert result.message == "all parameters fixed"
            assert np.count_nonzero(result.covariance) == 0


def _all_fixed_bounds(model) -> dict[str, tuple[float, float]]:
    return {
        name: (value, value)
        for name, value in zip(
            model.parameter_names, _BASE_PARAMETERS[model.model_id], strict=True
        )
    }


def test_all_fixed_fit_evaluates_the_full_curve_uncompressed() -> None:
    """An all-fixed expression on a long curve is one evaluation of every point.

    Compression decides where the solver iterates; with no free parameter
    nothing iterates.  Binning 9000 points into 4096 statistics anyway and
    then weighting the full-length residual by the bin weights raised a
    broadcast error for an expression the exact-point option answered.
    """

    engine = FitEngine()
    model = replace(engine.registry.get("gaussian_offset"), compiled_descriptor=None)
    x = np.linspace(-4.0, 4.0, 9000)
    observations = model.evaluate((x,), _BASE_PARAMETERS[model.model_id])
    result = engine.fit(model, (x,), observations, bounds=_all_fixed_bounds(model))
    assert result.success and result.message == "all parameters fixed"
    assert result.residuals.size == x.size
    assert result.reduced_chi_square == 0.0


@pytest.mark.parametrize("compiled", (False, True), ids=("generic", "compiled"))
def test_all_fixed_fit_honours_a_cancelled_request(compiled: bool) -> None:
    """A cancelled request does no work, the all-fixed evaluation included.

    The generic all-fixed shortcut evaluated the model and returned success
    without ever asking ``cancelled``: the cooperative checks lived only in
    the iterating path.
    """

    engine = FitEngine()
    model = engine.registry.get("gaussian_offset")
    if not compiled:
        model = replace(model, compiled_descriptor=None)
    x = np.linspace(-4.0, 4.0, 31)
    observations = model.evaluate((x,), _BASE_PARAMETERS[model.model_id])
    asked: list[bool] = []

    def cancelled() -> bool:
        asked.append(True)
        return True

    with pytest.raises(FitCancelled):
        engine.fit(
            model,
            (x,),
            observations,
            bounds=_all_fixed_bounds(model),
            cancelled=cancelled,
        )
    assert asked


@pytest.mark.parametrize("model_id", tuple(_ANCHORED_MODELS))
def test_public_batch_keeps_each_nonzero_coordinate_anchor(model_id: str) -> None:
    engine = FitEngine()
    model = engine.registry.get(model_id)
    relative = (
        np.linspace(0.0, 12.0, 160)
        if model_id == "damped_sine"
        else np.linspace(0.0, 10.0, 112)
    )
    parameters = np.asarray(_BASE_PARAMETERS[model_id], dtype=np.float64)
    observations = model.evaluate((relative,), parameters)
    origins = (17.5, 40.0, 101.25, 250.0, 1000.5, 2048.0, 4096.25, 8192.0)
    coordinates = tuple((relative + origin,) for origin in origins)
    results, failures = engine.fit_batch(
        model_id,
        coordinates,
        (observations,) * len(coordinates),
    )
    assert failures == (None,) * len(coordinates)
    first = results[0]
    assert first is not None
    for result, coordinate in zip(results, coordinates, strict=True):
        assert result is not None
        np.testing.assert_allclose(
            result.parameter_values,
            first.parameter_values,
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result.model.evaluate(coordinate, result.parameter_values),
            result.fitted_values,
            rtol=1e-12,
            atol=1e-12,
        )


def test_compiled_batch_reports_the_origin_it_subtracted() -> None:
    """Every cell of an anchored batch learns the window start it is relative to.

    The shared axis' origin was computed once and then reported as zero for
    every cell of a batch (only a single-cell solve kept it), so a caller
    placing the anchored decays would have put them at t=0 instead of at
    the window start.
    """

    engine = FitEngine()
    model = engine.registry.get("exponential_decay")
    descriptor = model.compiled_descriptor
    assert descriptor is not None
    relative = np.linspace(0.0, 10.0, 112)
    observations = model.evaluate((relative,), _BASE_PARAMETERS[model.model_id])
    lower = np.asarray([parameter.bounds[0] for parameter in model.parameters])
    upper = np.asarray([parameter.bounds[1] for parameter in model.parameters])
    single = _fit_compiled.solve_compiled_single(
        descriptor,
        (relative + 250.0,),
        observations,
        base_lower=lower,
        base_upper=upper,
    )
    batch = _fit_compiled.solve_compiled_batch(
        descriptor,
        (relative + 250.0,),
        np.stack([observations, observations]),
        base_lower=lower,
        base_upper=upper,
    )
    np.testing.assert_array_equal(single.coordinate_origins[:, 0], [250.0])
    np.testing.assert_array_equal(batch.coordinate_origins[:, 0], [250.0, 250.0])
    np.testing.assert_allclose(
        batch.parameters,
        np.stack([single.parameters[0]] * 2),
        rtol=1e-7,
    )


def test_damped_sine_context_stays_linear_in_the_sample_count() -> None:
    """The damped sine shares the series coordinate plan; no N-by-N table.

    The model seeds itself from the observations with a Goertzel scan inside
    its prepare callback, so a plan carrying an N-by-N trigonometric table
    is dead weight: 128 MiB at the 4096-point budget, copied once per cell
    of a batch and held in the engine's context cache.
    """

    descriptor = _fit_compiled.damped_sine_descriptor()
    small = descriptor.context_builder((np.linspace(0.0, 1.0, 64),))
    large = descriptor.context_builder((np.linspace(0.0, 1.0, 4096),))
    assert small.shape[0] == large.shape[0]
    assert large.nbytes == small.nbytes * 4096 // 64


def test_public_batch_sigma_weights_and_nan_filter_keep_original_indices() -> None:
    engine = FitEngine()
    model_id = "gaussian_offset"
    model = engine.registry.get(model_id)
    base_x = np.linspace(-5.0, 5.0, 120)
    clean = model.evaluate((base_x,), _BASE_PARAMETERS[model_id])
    coordinates = []
    observations = []
    sigmas = []
    indices = []
    finite_masks = []
    for cell in range(8):
        random = np.random.default_rng(70_000 + cell)
        x = base_x.copy()
        sigma = np.linspace(0.01, 0.05, x.size)
        values = clean + random.normal(0.0, sigma)
        rejected = np.arange(10 + cell, 11 + cell + cell % 3)
        if cell % 2:
            values[rejected] = np.nan
        else:
            x[rejected] = np.nan
        sigma[0] = 0.0
        sigma[1] = np.nan
        finite = np.isfinite(x) & np.isfinite(values)
        coordinates.append((x,))
        observations.append(values)
        sigmas.append(sigma)
        indices.append(np.arange(x.size, dtype=np.int64) + 1_000 * cell)
        finite_masks.append(finite)

    results, failures = engine.fit_batch(
        model_id,
        tuple(coordinates),
        tuple(observations),
        observation_sigmas=tuple(sigmas),
        selected_indices=tuple(indices),
    )
    assert failures == (None,) * len(coordinates)
    reference_model = replace(model, compiled_descriptor=None)
    for cell, result in enumerate(results):
        assert result is not None
        scalar = engine.fit(
            reference_model,
            coordinates[cell],
            observations[cell],
            observation_sigma=sigmas[cell],
            selected_indices=indices[cell],
        )
        _assert_fit_equal(result, scalar)
        finite = finite_masks[cell]
        np.testing.assert_array_equal(result.selected_indices, indices[cell][finite])
        used_sigma = sigmas[cell][finite]
        floor = float(
            np.min(used_sigma[np.isfinite(used_sigma) & (used_sigma > 0.0)])
        )
        bounded = np.where(
            np.isfinite(used_sigma) & (used_sigma > 0.0),
            used_sigma,
            floor,
        )
        expected_reduced = float(
            np.dot(result.residuals / bounded, result.residuals / bounded)
            / (result.residuals.size - len(model.parameters))
        )
        assert result.reduced_chi_square == pytest.approx(expected_reduced, rel=1e-10)


def test_public_batch_compacts_regular_image_masks() -> None:
    engine = FitEngine()
    model_id = "radial_gaussian_center"
    model = engine.registry.get(model_id)
    x = np.linspace(-2.0, 2.0, 24)
    y = np.linspace(-1.8, 2.2, 20)
    xx, yy = np.meshgrid(x, y)
    inputs = []
    for cell in range(8):
        parameters = _cell_parameters(model_id, cell)
        image = model.evaluate(
            (xx.reshape(-1), yy.reshape(-1)),
            parameters,
        ).reshape(y.size, x.size)
        mask = np.ones(image.shape, dtype=np.bool_)
        mask[: 1 + cell % 3, :] = False
        mask[:, -1 - cell % 2 :] = False
        mask[5 + cell, 7 + cell] = False
        inputs.append(RegularImageFitInput(x, y, image, valid_mask=mask))

    scalar = tuple(engine.fit(model_id, item) for item in inputs)
    results, failures = engine.fit_batch(
        model_id,
        tuple(inputs),
        (None,) * len(inputs),
    )
    assert failures == (None,) * len(inputs)
    for result, expected in zip(results, scalar, strict=True):
        assert result is not None
        _assert_fit_equal(result, expected)


def test_explicit_batch_bounds_replace_model_derived_bounds() -> None:
    engine = FitEngine()
    model = engine.registry.get("radial_gaussian_center")
    coordinates = _coordinates(model.model_id)
    observations = model.evaluate(coordinates, _BASE_PARAMETERS[model.model_id])
    defaults = model.bounds_initializer(coordinates, observations)
    assert defaults is not None and defaults["center_x"][1] < 2.1
    bounds = {"center_x": (2.1, 2.3)}
    results, failures = engine.fit_batch(
        model,
        (coordinates,) * 8,
        (observations,) * 8,
        bounds=bounds,
    )
    assert failures == (None,) * 8
    for result in results:
        assert result is not None
        assert 2.1 <= result.parameters["center_x"] <= 2.3


def test_public_batch_filters_a_nan_coordinate_after_temporaries_recycle_ids() -> None:
    """A cell's NaN coordinate is filtered whatever ids earlier cells freed.

    The batch remembers per axis OBJECT whether it is all-finite, so a tensor
    facet's shared axes are scanned once.  Eight cells whose observations
    are all NaN each proved a temporary axis finite, failed, and freed it;
    the next cell's axis landed on a recycled id, inherited the proof, and
    its NaN coordinate reached the evaluator -- "fixed fit evaluation is
    non-finite" for a cell whose own single fit succeeds on the 32 finite
    points.
    """

    engine = FitEngine()
    model = engine.registry.get("gaussian_offset")
    bounds = _all_fixed_bounds(model)
    x = np.linspace(-2.0, 2.0, 33)
    clean = model.evaluate((x,), _BASE_PARAMETERS[model.model_id])
    modes = (0,) * 8 + (2, 1, 2)
    coordinates = []
    observations = []
    for mode in modes:
        axis = x.copy()
        values = clean.copy()
        if mode == 0:
            values[:] = np.nan
        elif mode == 1:
            values[3] = np.nan
        else:
            axis[5] = np.nan
        coordinates.append((axis,))
        observations.append(values)
    results, failures = engine.fit_batch(
        model, tuple(coordinates), tuple(observations), bounds=bounds
    )
    for cell, mode in enumerate(modes):
        if mode == 0:
            assert results[cell] is None
            assert failures[cell] == (
                "fit requires more finite observations than free parameters"
            )
            continue
        assert failures[cell] is None, (cell, failures[cell])
        result = results[cell]
        assert result is not None
        single = engine.fit(model, coordinates[cell], observations[cell], bounds=bounds)
        _assert_fit_equal(result, single)
        assert result.selected_indices.size == x.size - 1


def test_invalid_public_batch_warm_start_raises() -> None:
    engine = FitEngine()
    cases = tuple(
        _fit_case(engine, "gaussian_offset", "normal", cell)
        for cell in range(8)
    )
    invalid = np.asarray((2.0, 0.2, np.nan, -0.3))
    with pytest.raises(ValueError, match="invalid parameter values"):
        engine.fit_batch(
            "gaussian_offset",
            tuple(case[0] for case in cases),
            tuple(case[1] for case in cases),
            warm_starts=(None, invalid, None, None, None, None, None, None),
        )


@pytest.mark.parametrize("loss", ("linear", "soft_l1", "huber", "cauchy", "arctan"))
def test_public_batch_all_supported_losses_match_single(loss: str) -> None:
    engine = FitEngine()
    reference_model = replace(
        engine.registry.get("gaussian_offset"),
        compiled_descriptor=None,
    )
    cases = tuple(
        _fit_case(engine, "gaussian_offset", "hard", cell)
        for cell in range(8)
    )
    options = FitOptions(loss=loss)
    scalar = tuple(
        engine.fit(
            reference_model,
            coordinates,
            observations,
            options=options,
        )
        for coordinates, observations, _sigma, _indices in cases
    )
    results, failures = engine.fit_batch(
        "gaussian_offset",
        tuple(case[0] for case in cases),
        tuple(case[1] for case in cases),
        options=options,
    )
    assert failures == (None,) * len(cases)
    for result, expected in zip(results, scalar, strict=True):
        assert result is not None
        _assert_fit_equal(
            result,
            expected,
            quality_tolerance=(1e-10 if loss == "linear" else 1e-9),
        )


@pytest.mark.parametrize("compiled", (False, True), ids=("generic", "compiled"))
def test_robust_candidates_compete_on_the_robust_cost(compiled: bool) -> None:
    """A soft-L1 fit keeps the candidate its own loss prefers.

    Two basins: the Gaussian at x=-1 and a lone spike of 50 at x=+1.  The
    spike wins by squared residuals (2435 against 2500) and loses by the
    soft-L1 sum (126 against 98); ranking the warm and authored candidates
    by squared residuals returned the spike's basin from a robust fit in
    both solver lanes.  The reported quality stays the raw residual sum of
    squares.
    """

    engine = FitEngine()
    model = engine.registry.get("gaussian_offset")
    if not compiled:
        model = replace(model, compiled_descriptor=None)
    x = np.linspace(-3.0, 3.0, 601)
    observations = np.exp(-0.5 * ((x + 1.0) / 0.1) ** 2)
    observations[np.argmin(np.abs(x - 1.0))] += 50.0
    result = engine.fit(
        model,
        (x,),
        observations,
        bounds={
            "amplitude": (1.0, 1.0),
            "offset": (0.0, 0.0),
            "sigma": (0.1, 0.1),
            "center": (-2.0, 2.0),
        },
        initial=(1.0, 0.0, 0.1, 1.0),
        warm_start=(1.0, 0.0, 0.1, -1.0),
        options=FitOptions(loss="soft_l1", max_exact_points=None),
    )
    assert result.success
    assert result.parameters["center"] == pytest.approx(-1.0, abs=1e-4)
    residuals = np.asarray(result.residuals)
    assert result.reduced_chi_square == pytest.approx(
        float(np.dot(residuals, residuals)) / (x.size - 1), rel=1e-9
    )


def test_rank_deficient_cell_stays_on_compiled_batch_without_scalar_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FitEngine()
    model = engine.registry.get("gaussian_offset")
    x = np.linspace(-5.0, 5.0, 112)
    observations = [np.full(x.size, 0.2)]
    observations.extend(
        model.evaluate((x,), _cell_parameters(model.model_id, cell))
        for cell in range(1, 8)
    )

    def forbidden_scalar(*_args, **_kwargs):
        raise AssertionError("compiled batch fell back to scipy least_squares")

    monkeypatch.setattr(
        import_module("zlc_plot.fit"),
        "least_squares",
        forbidden_scalar,
    )
    results, failures = engine.fit_batch(
        model,
        ((x,),) * len(observations),
        tuple(observations),
    )
    assert failures == (None,) * len(observations)
    flat = results[0]
    assert flat is not None and flat.success
    assert not flat.covariance_valid
    assert np.all(np.isnan(flat.standard_errors))


def test_compiled_batch_judges_finiteness_on_the_points_it_fitted() -> None:
    """A masked NaN observation leaves a cell's covariance valid.

    The compiled solve masks non-finite observations out of the fit, but its
    finalizer checked the residual of every point, masked or not, so one NaN
    observation made an otherwise clean cell's covariance invalid and its
    errors NaN.
    """

    engine = FitEngine()
    model = engine.registry.get("gaussian_offset")
    descriptor = model.compiled_descriptor
    assert descriptor is not None
    x = np.linspace(-5.0, 5.0, 112)
    rng = np.random.default_rng(23)
    clean = model.evaluate((x,), _BASE_PARAMETERS[model.model_id])
    clean = clean + rng.normal(0.0, 0.003, clean.size)
    holed = clean.copy()
    holed[10] = np.nan
    output = _fit_compiled.solve_compiled_batch(
        descriptor,
        (x,),
        np.stack([clean, holed]),
        base_lower=np.asarray([parameter.bounds[0] for parameter in model.parameters]),
        base_upper=np.asarray([parameter.bounds[1] for parameter in model.parameters]),
        # One plan for both cells, as the engine hands a bucket: a plan built
        # per cell from its finite points would differ in shape here.
        context=descriptor.context_builder((x,)),
    )
    assert output.success.tolist() == [True, True]
    assert output.covariance_valid.tolist() == [True, True]
    assert np.all(np.isfinite(output.standard_errors))
    np.testing.assert_allclose(output.parameters[1], output.parameters[0], rtol=1e-3)


@pytest.mark.parametrize("fallback", ("custom_model", "custom_engine"))
def test_public_batch_uses_per_cell_fit_route_for_explicit_customization(
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
) -> None:
    class RecordingEngine(FitEngine):
        def __init__(self) -> None:
            super().__init__()
            self.fit_calls = 0

        def fit(  # type: ignore[no-untyped-def]
            self,
            model,
            coordinates,
            observations=None,
            **kwargs,
        ):
            self.fit_calls += 1
            return super().fit(model, coordinates, observations, **kwargs)

    engine: FitEngine
    model: object
    recording: RecordingEngine | None = None
    calls: list[None] = []
    if fallback == "custom_engine":
        recording = RecordingEngine()
        engine = recording
        model = "gaussian_offset"
    else:
        engine = FitEngine()
        builtin = engine.registry.get("gaussian_offset")
        model = replace(
            builtin,
            model_id="custom_gaussian",
            compiled_descriptor=None,
        )
        original = engine.fit
        def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(None)
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, "fit", counted)

    cases = tuple(
        _fit_case(engine, "gaussian_offset", "normal", cell)
        for cell in range(3)
    )
    results, failures = engine.fit_batch(
        model,  # type: ignore[arg-type]
        tuple(case[0] for case in cases),
        tuple(case[1] for case in cases),
    )
    assert failures == (None,) * len(cases)
    assert all(result is not None for result in results)
    call_count = (
        recording.fit_calls
        if recording is not None
        else len(calls)
    )
    assert call_count == len(cases)


def test_frozen_anchors_cover_all_builtin_evaluators() -> None:
    from scipy.optimize._numdiff import approx_derivative
    from zlc_plot.fit import _compiled_model_input

    engine = FitEngine()
    anchors = _anchors()
    assert set(anchors) == set(PARAMETERS)
    for model, item in anchors.items():
        spec = engine.registry.get(model)
        axes = tuple(
            np.asarray(values, dtype=np.float64)
            for values in item["coordinates"].values()
        )
        coordinates = (
            tuple(axis for axis in np.meshgrid(*axes))
            if model == "radial_gaussian_center"
            else axes
        )
        parameters = tuple(
            float(item["parameters"][name]) for name in spec.parameter_names
        )
        expected = np.asarray(item["values"], dtype=np.float64)
        actual = spec.evaluate(coordinates, parameters)
        # The Poisson-Gaussian models are a numerical integral; their anchors
        # are an independent quadrature and hold the kernel to its own
        # accuracy, not to the closed forms' rounding.
        tolerance = 1e-6 if model in _POISSON_MODELS else 2e-12
        assert np.allclose(actual, expected, rtol=tolerance, atol=tolerance), model
        flat = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in coordinates)
        packed, values = _compiled_model_input(flat, parameters)
        predicted, no_jacobian = spec.compiled_descriptor.value_jacobian(packed, values, False)
        assert no_jacobian.shape == (0, len(parameters))
        np.testing.assert_array_equal(predicted, actual.reshape(-1))
        if len(flat) == 1:
            assert np.shares_memory(packed, flat[0])
        derivative = approx_derivative(
            lambda point: spec.evaluate(flat, point), np.asarray(parameters), method="3-point",
        )
        np.testing.assert_allclose(
            spec.evaluate_jacobian(flat, parameters), derivative, rtol=2e-5, atol=2e-7,
            err_msg=model,
        )


@pytest.mark.parametrize("model", tuple(PARAMETERS))
def test_every_builtin_model_recovers_synthetic_parameters(model: str) -> None:
    engine = FitEngine()
    spec = engine.registry.get(model)
    coordinates, observations, _ = _anchor(model)
    expected = tuple(
        float(_anchors()[model]["parameters"][name])
        for name in spec.parameter_names
    )
    # Recovery is asked of the model itself.  A two-population model also
    # asks whether the data has two populations, and an anchor of a few
    # dozen shots spread over eighty bins cannot show that it does; that
    # question has its own tests.
    options = FitOptions(min_bic_gain=None) if spec.reduction is not None else None
    result = engine.fit(
        model, coordinates, observations, data_revision=11, options=options
    )
    assert result.success
    assert result.source_revision == 11
    if model in _ANCHORED_MODELS:
        # These models are anchored to the window they are fitted over, so the
        # amplitude (and phase) are reported at the window start instead of at
        # x=0.  Everything else -- and the curve itself -- is unchanged.
        for name, value, truth in zip(
            spec.parameter_names, result.parameter_values, expected, strict=True
        ):
            if name in ("offset", "decay_time", "baseband_frequency"):
                assert np.isclose(value, truth, rtol=2e-3, atol=2e-3), name
        curve = result.model.evaluate(coordinates, result.parameter_values)
        assert np.allclose(curve, observations, rtol=2e-3, atol=2e-3)
    else:
        assert np.allclose(result.parameter_values, expected, rtol=2e-3, atol=2e-3)
    assert np.all(np.isfinite(result.standard_errors))


@pytest.mark.parametrize("model", _GENERIC_WARM_MODELS)
def test_globally_unbeatable_warm_seed_skips_redundant_cold_candidates(
    model: str,
) -> None:
    engine = FitEngine()
    coordinates, observations, _parameters = _anchor(model)
    cold = engine.fit(model, coordinates, observations)
    warm = engine.fit(
        engine.registry.get(model),
        coordinates,
        observations,
        warm_start=tuple(float(value) for value in cold.parameter_values),
    )
    for field in (
        "parameter_values",
        "standard_errors",
        "covariance",
        "fitted_values",
        "residuals",
        "selected_indices",
    ):
        assert np.array_equal(
            getattr(warm, field), getattr(cold, field), equal_nan=True
        ), field
    assert warm.model.model_id == cold.model.model_id
    assert warm.success == cold.success
    assert warm.reduced_chi_square == cold.reduced_chi_square
    assert warm.covariance_valid == cold.covariance_valid


def test_misleading_warm_seed_keeps_the_cold_winner() -> None:
    engine = FitEngine()
    coordinates, observations, _parameters = _anchor("lorentzian")
    cold = engine.fit("lorentzian", coordinates, observations)
    recovered = engine.fit(
        engine.registry.get("lorentzian"),
        coordinates,
        observations,
        warm_start=(2.5, 0.1, 0.1, 1.5),
    )
    _assert_fit_equal(recovered, cold)


def test_fit_bounds_are_enforced() -> None:
    model = "gaussian_offset"
    (x,), values, _ = _anchor(model)
    result = FitEngine().fit(
        model,
        (x,),
        values,
        bounds={"center": (0.0, 0.1)},
    )
    assert 0.0 <= result.parameters["center"] <= 0.1

    truth = _anchors()[model]["parameters"]
    fixed = {"sigma": truth["sigma"], "center": truth["center"]}
    result = FitEngine().fit(
        model,
        (x,),
        values,
        bounds={name: (value, value) for name, value in fixed.items()},
    )

    assert result.success
    assert result.fixed_parameter_names == tuple(fixed)
    assert result.parameters["sigma"] == truth["sigma"]
    assert result.parameters["center"] == truth["center"]
    fixed_indices = (2, 3)
    assert np.all(result.standard_errors[list(fixed_indices)] == 0.0)
    assert np.all(result.covariance[list(fixed_indices)] == 0.0)
    assert np.all(result.covariance[:, list(fixed_indices)] == 0.0)
    assert not result.parameter_error_validity["sigma"]
    assert not result.parameter_error_validity["center"]
    assert result.parameter_error_validity["amplitude"]


def test_fit_cancellation_is_checked_before_work() -> None:
    (x,), values, _ = _anchor("gaussian_offset")
    with pytest.raises(FitCancelled):
        FitEngine().fit(
            "gaussian_offset",
            (x,),
            values,
            cancelled=lambda: True,
        )


def test_reflected_trust_region_step_is_measured_to_the_sphere_ahead() -> None:
    """The reflected candidate walks forward to the trust-region boundary.

    From a point inside the sphere the reflected line meets it once behind
    and once ahead.  Taking the root behind made every reflected stride
    non-positive, so the solver never tried a reflection and, on this
    positive-definite bounded subproblem, settled for a step eleven times
    worse than the feasible reflected one.
    """

    forward = _fit_compiled._positive_intersection(
        np.array([0.0]), np.array([1.0]), 1.0
    )
    assert forward == pytest.approx(1.0)

    step = np.zeros(2)
    scaled_step = np.zeros(2)
    _fit_compiled._select_reflective_step(
        np.zeros(2),
        np.diag([1.0, 10.0]),
        np.array([-1.0, -20.0]),
        np.array([1.0, 2.0]),
        np.ones(2),
        3.0,
        np.array([-3.0, -3.0]),
        np.array([0.9, 3.0]),
        0.995,
        step,
        scaled_step,
    )

    def objective(point: np.ndarray) -> float:
        return 0.5 * ((point[0] - 1.0) ** 2 + 10.0 * (point[1] - 2.0) ** 2)

    # The bound at x=0.9 reflects the Newton step into direction (-1, 2);
    # the quadratic's minimum along that line sits 3.9/41 of the way.
    reflected = np.array([0.9, 1.8]) + (3.9 / 41.0) * np.array([-1.0, 2.0])
    np.testing.assert_allclose(step, reflected, rtol=1e-9)
    assert objective(step) < objective(0.995 * np.array([0.9, 1.8]))


def test_radial_regular_image_fast_path_matches_coordinate_path() -> None:
    engine = FitEngine()
    model = engine.registry.get("radial_gaussian_center")
    item = _anchors()["radial_gaussian_center"]
    x_axis = np.asarray(item["coordinates"]["x"], dtype=np.float64)
    y_axis = np.asarray(item["coordinates"]["y"], dtype=np.float64)
    flattened = np.asarray(item["values"], dtype=np.float64)
    xx, yy = np.meshgrid(x_axis, y_axis)
    image = flattened.reshape(y_axis.size, x_axis.size)
    generic = engine.fit(
        model,
        (xx.reshape(-1), yy.reshape(-1)),
        image.reshape(-1),
    )
    regular = engine.fit(
        model,
        RegularImageFitInput(x_axis, y_axis, image),
    )
    assert np.allclose(regular.parameter_values, generic.parameter_values, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("background", (1e4, 1e8))
def test_regular_image_quality_is_its_own_residual_sum_of_squares(
    background: float,
) -> None:
    """A fixed exact model on a bright background reports no misfit.

    The closed-form separable objective differences second moments of the
    image, each about N*B^2; on a 1e8 background an exact fit reported a
    reduced chi-square of 3 over residuals that were identically zero, while
    the same image in a two-cell batch reported 0.  The quality is
    accumulated from the per-pixel residuals the result itself hands back.
    """

    engine = FitEngine()
    model = engine.registry.get("radial_gaussian_center")
    x = np.linspace(-1.0, 1.0, 19)
    y = np.linspace(-1.0, 1.0, 17)
    xx, yy = np.meshgrid(x, y)
    truth = np.asarray((1.0, background, 0.7, 0.15, -0.1))
    image = model.evaluate((xx.reshape(-1), yy.reshape(-1)), truth).reshape(yy.shape)
    bounds = {
        name: (value, value)
        for name, value in zip(model.parameter_names, truth, strict=True)
    }
    single = engine.fit(model, RegularImageFitInput(x, y, image), bounds=bounds)
    pair, failures = engine.fit_batch(
        model,
        (RegularImageFitInput(x, y, image), RegularImageFitInput(x, y, image)),
        (None, None),
        bounds=bounds,
    )
    assert single.success and failures == (None, None)
    residuals = np.asarray(single.residuals)
    own_quality = float(np.dot(residuals, residuals)) / residuals.size
    assert single.reduced_chi_square == pytest.approx(own_quality, abs=1e-9)
    assert single.reduced_chi_square <= 1e-9
    assert pair[0].reduced_chi_square == pytest.approx(single.reduced_chi_square, abs=1e-9)


def test_regular_radial_fit_does_not_depend_on_which_axis_is_finer() -> None:
    """One radius, one floor: half the finer pitch of the two axes.

    A radial Gaussian sampled at 0.1 along x and 1.0 along y is resolved by
    the x samples; deciding the floor per axis let whichever axis came last
    win, so the same pixels fitted R=0.5 one way round and R=0.2 transposed.
    """

    engine = FitEngine()
    model = engine.registry.get("radial_gaussian_center")
    x = np.linspace(-2.0, 2.0, 41)
    y = np.linspace(-2.0, 2.0, 5)
    xx, yy = np.meshgrid(x, y)
    truth = np.asarray((1.0, 0.0, 0.2, 0.0, 0.0))
    image = model.evaluate((xx.reshape(-1), yy.reshape(-1)), truth).reshape(yy.shape)
    bounds = {
        name: (value, value)
        for name, value in zip(model.parameter_names, truth, strict=True)
        if name != "one_over_e_radius"
    }
    upright = engine.fit(
        model, RegularImageFitInput(x, y, image), bounds=bounds, initial=tuple(truth)
    )
    transposed = engine.fit(
        model,
        RegularImageFitInput(y, x, np.ascontiguousarray(image.T)),
        bounds=bounds,
        initial=tuple(truth),
    )
    assert upright.success and transposed.success
    assert upright.parameters["one_over_e_radius"] == pytest.approx(0.2, rel=1e-6)
    assert transposed.parameters["one_over_e_radius"] == pytest.approx(
        upright.parameters["one_over_e_radius"], rel=1e-6
    )


def _separable_image(
    *,
    radial: bool,
    size: int = 96,
    noise: float = 0.02,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, size)
    y = np.linspace(-1.5, 2.5, size)
    xx, yy = np.meshgrid(x, y)
    if radial:
        signal = 0.8 + 5.0 * np.exp(
            -(((xx - 0.31) ** 2) + (yy - 0.22) ** 2) / 0.6**2
        )
    else:
        signal = 0.8 + 5.0 * np.exp(
            -((xx - 0.31) ** 2 / 0.45**2 + (yy - 0.22) ** 2 / 0.85**2)
        )
    return x, y, signal + rng.normal(0.0, noise, size=signal.shape)


@pytest.mark.parametrize(
    ("model", "radii", "permissive_bounds"),
    (
        (
            "radial_gaussian_center",
            (18.0,),
            {"one_over_e_radius": (0.5, None)},
        ),
        (
            "anisotropic_gaussian_center",
            (24.0, 12.0),
            {"radius_x": (0.5, None), "radius_y": (0.5, None)},
        ),
    ),
)
def test_large_regular_image_defaults_keep_a_narrow_peak_in_bounds(
    model: str,
    radii: tuple[float, ...],
    permissive_bounds: dict[str, tuple[float | None, float | None]],
) -> None:
    """A camera-sized noise floor must not set a peak's minimum width."""

    height, width = 1200, 1920
    x = np.arange(width, dtype=float)
    y = np.arange(height, dtype=float)
    center_x, center_y = 0.5 * (width - 1), 0.5 * (height - 1)
    radius_x, radius_y = (radii * 2)[:2]
    x_profile = np.exp(-((x - center_x) / radius_x) ** 2)
    y_profile = np.exp(-((y - center_y) / radius_y) ** 2)
    rng = np.random.default_rng(20260820)
    image = rng.standard_normal((height, width), dtype=np.float32)
    image *= 1.5
    image += 7.0 + 90.0 * y_profile[:, None] * x_profile[None, :]
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    engine = FitEngine()
    full = RegularImageFitInput(x, y, image)
    roi = RegularImageFitInput(
        x[704:1216],
        y[344:856],
        image[344:856, 704:1216],
    )
    roi_result = engine.fit(model, roi)
    full_result = engine.fit(model, full)
    roi_seeded_full = engine.fit(
        model,
        full,
        initial=roi_result.parameter_values,
        bounds=permissive_bounds,
    )

    assert roi_result.success and full_result.success and roi_seeded_full.success
    assert (
        full_result.reduced_chi_square
        <= roi_seeded_full.reduced_chi_square * (1.0 + 1.0e-10)
    )
    np.testing.assert_allclose(
        full_result.parameter_values[2 : 2 + len(radii)],
        radii,
        rtol=0.03,
        atol=0.2,
    )


def test_anisotropic_regular_image_matches_coordinate_path() -> None:
    engine = FitEngine()
    x, y, image = _separable_image(radial=False)
    xx, yy = np.meshgrid(x, y)
    generic = engine.fit(
        "anisotropic_gaussian_center",
        (xx.reshape(-1), yy.reshape(-1)),
        image.reshape(-1),
    )
    regular = engine.fit(
        "anisotropic_gaussian_center",
        RegularImageFitInput(x, y, image),
    )
    assert generic.success and regular.success
    assert np.allclose(
        regular.parameter_values,
        generic.parameter_values,
        rtol=1e-6,
        atol=1e-9,
    )
    assert np.all(np.isfinite(regular.standard_errors))
    assert np.allclose(
        regular.standard_errors,
        generic.standard_errors,
        rtol=1e-4,
        atol=1e-12,
    )


def test_regular_image_rejects_models_without_the_capability() -> None:
    x, y, image = _separable_image(radial=True, size=24)
    with pytest.raises(ValueError, match="regular-image capability"):
        FitEngine().fit("lorentzian", RegularImageFitInput(x, y, image))


@pytest.mark.parametrize(
    ("model", "radial", "bound_name", "parameter_index"),
    (
        ("radial_gaussian_center", True, "one_over_e_radius", 2),
        ("anisotropic_gaussian_center", False, "radius_x", 2),
    ),
)
def test_regular_image_explicit_radius_bound_overrides_sampling_default(
    model: str,
    radial: bool,
    bound_name: str,
    parameter_index: int,
) -> None:
    x, y, image = _separable_image(radial=radial, size=48)
    result = FitEngine().fit(
        model,
        RegularImageFitInput(x, y, image),
        bounds={bound_name: (1.5, 1.7)},
    )
    assert 1.5 <= result.parameter_values[parameter_index] <= 1.7


def test_rectangular_mask_crops_to_the_closed_form_and_keeps_original_indices() -> None:
    engine = FitEngine()
    x, y, image = _separable_image(radial=True, size=64)
    mask = np.zeros(image.shape, dtype=bool)
    mask[10:52, 8:56] = True
    masked = engine.fit(
        "radial_gaussian_center",
        RegularImageFitInput(x, y, image, valid_mask=mask),
    )
    cropped = engine.fit(
        "radial_gaussian_center",
        RegularImageFitInput(x[8:56], y[10:52], image[10:52, 8:56]),
    )
    assert masked.success and cropped.success
    assert np.allclose(
        masked.parameter_values,
        cropped.parameter_values,
        rtol=1e-9,
        atol=1e-12,
    )
    # Deferred indices map back to the flat pixels of the original image.
    assert np.array_equal(
        masked.selected_indices,
        np.flatnonzero(mask.reshape(-1)),
    )
    predicted = masked.model.evaluate(
        (
            np.meshgrid(x, y)[0][mask],
            np.meshgrid(x, y)[1][mask],
        ),
        masked.parameter_values,
    )
    assert np.allclose(masked.fitted_values, predicted, rtol=1e-12, atol=1e-12)
    assert np.allclose(
        masked.residuals,
        image[mask] - predicted,
        rtol=1e-9,
        atol=1e-12,
    )


def test_regular_image_result_arrays_are_deferred_until_first_access() -> None:
    engine = FitEngine()
    x, y, image = _separable_image(radial=True, size=48)
    result = engine.fit(
        "radial_gaussian_center",
        RegularImageFitInput(x, y, image),
    )

    def raw(target, name):
        return _FIT_RESULT_RAW[name].__get__(target, type(target))

    assert isinstance(raw(result, "fitted_values"), _DeferredFitData)
    # Laziness survives the unit and batch-revision clones used on the
    # session accept path.
    stamped = result.with_batch_revision(7)
    united = stamped.with_parameter_units({"amplitude": ""})
    assert isinstance(raw(stamped, "fitted_values"), _DeferredFitData)
    assert isinstance(raw(united, "fitted_values"), _DeferredFitData)
    assert united.batch_revision == 7

    fitted = united.fitted_values
    assert isinstance(raw(united, "fitted_values"), np.ndarray)
    assert fitted.dtype == np.float64 and not fitted.flags.writeable
    assert united.residuals.shape == fitted.shape
    assert united.selected_indices.shape == fitted.shape
    assert not united.residuals.flags.writeable
    assert np.array_equal(
        united.selected_indices, np.arange(image.size, dtype=np.int64)
    )
    assert np.allclose(
        united.fitted_values + united.residuals,
        image.reshape(-1),
        rtol=1e-12,
        atol=1e-12,
    )
    # dataclasses.replace materializes through the lazy accessors and keeps
    # the documented field semantics.
    invalid = replace(result, covariance_valid=False)
    assert invalid.fitted_values.shape == fitted.shape
    assert np.all(np.isnan(invalid.standard_errors))


@pytest.mark.parametrize(
    ("model", "radial"),
    (("radial_gaussian_center", True), ("anisotropic_gaussian_center", False)),
)
def test_regular_image_warm_start_reproduces_the_cold_solution(
    model: str, radial: bool
) -> None:
    engine = FitEngine()
    x, y, image = _separable_image(radial=radial, size=96)
    data = RegularImageFitInput(x, y, image)
    cold = engine.fit(model, data)
    warm = engine.fit(
        engine.registry.get(model),
        data,
        warm_start=tuple(float(value) for value in cold.parameter_values),
    )
    assert cold.success and warm.success
    assert np.allclose(
        warm.parameter_values,
        cold.parameter_values,
        rtol=1e-6,
        atol=1e-9,
    )
    assert np.all(np.isfinite(warm.standard_errors))


def test_large_curves_solve_on_binned_statistics_and_report_full_data() -> None:
    """Compression decides where the solver ITERATES, never what is reported.

    A curve past ``max_exact_points`` iterates on x-binned means, yet the
    result's fitted values, residuals and indices stay per-point: they are
    what overlays and published outputs consume.  The parameters must agree
    with the exact solve far inside any physical error bar.
    """

    engine = FitEngine()
    rng = np.random.default_rng(17)
    n = 60_000
    x = np.linspace(-4.0, 4.0, n)
    y = 0.3 + 2.0 * np.exp(-0.5 * ((x - 0.4) / 0.9) ** 2)
    y = y + rng.normal(0.0, 0.03, n)
    exact = engine.fit(
        "gaussian_offset", (x,), y, options=FitOptions(max_exact_points=None)
    )
    binned = engine.fit("gaussian_offset", (x,), y)
    assert exact.success and binned.success
    for name, value in exact.parameters.items():
        assert abs(binned.parameters[name] - value) <= 1e-4 * max(
            1e-12, abs(value)
        )
    assert binned.fitted_values.shape == (n,)
    assert binned.residuals.shape == (n,)
    assert binned.selected_indices.shape == (n,)
    # The reported quality is the full data's, not the binned statistics'.
    assert binned.reduced_chi_square == pytest.approx(
        float(np.dot(binned.residuals, binned.residuals))
        / (n - len(binned.parameter_values)),
        rel=1e-12,
    )


def test_max_exact_points_none_solves_every_point() -> None:
    engine = FitEngine()
    rng = np.random.default_rng(23)
    n = 20_000
    x = np.linspace(0.0, 8.0, n)
    y = 0.1 + 2.5 * np.exp(-x / 1.7) + rng.normal(0.0, 0.02, n)
    first = engine.fit(
        "exponential_decay", (x,), y, options=FitOptions(max_exact_points=None)
    )
    second = engine.fit(
        "exponential_decay", (x,), y, options=FitOptions(max_exact_points=None)
    )
    assert first.success and second.success
    assert tuple(first.parameter_values) == tuple(second.parameter_values)


def test_small_curves_never_compress() -> None:
    """Below the threshold the solver sees every point, exactly as before."""

    engine = FitEngine()
    rng = np.random.default_rng(29)
    n = 4_000
    x = np.linspace(-4.0, 4.0, n)
    y = 0.3 + 2.0 * np.exp(-0.5 * ((x - 0.2) / 0.8) ** 2)
    y = y + rng.normal(0.0, 0.02, n)
    default = engine.fit("gaussian_offset", (x,), y)
    exact = engine.fit(
        "gaussian_offset", (x,), y, options=FitOptions(max_exact_points=None)
    )
    assert tuple(default.parameter_values) == tuple(exact.parameter_values)


def test_poisson_gaussian_recovers_two_state_photon_histograms() -> None:
    """Rates, splitting and read noise from synthetic two-state photon
    histograms (64 bins, 5000 shots), in the regimes the model is for: a
    few to a few hundred photons, read noise resolved by the bins.  The
    Gaussian bimodal reports the same splitting; only this model reports
    the rates and the read noise."""

    engine = FitEngine()
    rng = np.random.default_rng(7)
    n = 5000
    loaded = rng.random(n) < 0.5

    photons = np.where(loaded, rng.poisson(30.0, n), rng.poisson(3.0, n))
    counts, edges = np.histogram(photons + rng.normal(0.0, 1.0, n), bins=64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    result = engine.fit("bimodal_poisson_gaussian", (centres,), counts.astype(float))
    assert result.success
    assert 2.5 < result.parameters["rate"] < 3.3
    assert 26.5 < result.parameters["delta_rate"] < 27.6
    assert 0.8 < result.parameters["sigma"] < 1.4
    assert 0.5 < result.parameters["sigma_B"] < 2.5
    assert 0.45 < result.parameters["ratio"] < 0.55

    bright = np.where(loaded, rng.poisson(210.0, n), rng.poisson(60.0, n))
    counts, edges = np.histogram(bright + rng.normal(0.0, 3.0, n), bins=64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    result = engine.fit("bimodal_poisson_gaussian", (centres,), counts.astype(float))
    assert result.success
    assert 148.0 < result.parameters["delta_rate"] < 152.0
    assert 2.0 < result.parameters["sigma"] < 4.5
    assert 1.5 < result.parameters["sigma_B"] < 5.0
