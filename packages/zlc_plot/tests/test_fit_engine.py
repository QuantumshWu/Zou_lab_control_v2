from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from zlc_plot import FitCancelled
from zlc_plot.fit import (
    FitEngine,
    FitOptions,
    FitResult,
    RegularImageFitInput,
    _DeferredFitData,
    _FIT_RESULT_RAW,
    builtin_fit_models,
)


PARAMETERS = {
    "lorentzian": (0.35, 1.2, 2.5, 0.2),
    "gaussian_offset": (2.0, 0.15, 0.9, -0.3),
    "histogram_gaussian": (2.0, 0.2, 0.9),
    "bimodal_gaussian": (0.0, 1.4, 1.2, 0.6, 0.9, 0.8),
    "symmetric_lorentzian_doublet": (0.1, 1.0, 1.5, 0.1, 1.2),
    "damped_sine": (1.2, 0.1, 1.4, 3.0, 0.2),
    "exponential_decay": (2.2, 0.1, 2.5),
    "radial_gaussian_center": (3.0, 0.2, 0.8, 0.4, -0.3),
    "histogram_poisson_gaussian": (2.0, 1.5, 0.6),
    "bimodal_poisson_gaussian": (0.8, 3.2, 1.2, 0.5, 0.9, 0.7),
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
    "anisotropic_gaussian_center",
    "radial_gaussian_center",
    "histogram_poisson_gaussian",
    "bimodal_poisson_gaussian",
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
    "histogram_gaussian": (90.0, -0.3, 0.8),
    "bimodal_gaussian": (0.0, 2.4, 60.0, 0.55, 45.0, 0.75),
    "symmetric_lorentzian_doublet": (0.1, 0.8, 1.4, 0.2, 2.5),
    "damped_sine": (1.2, 0.2, 0.25, 6.0, -0.3),
    "exponential_decay": (1.6, 0.2, 3.0),
    "anisotropic_gaussian_center": (3.0, 0.2, 0.9, 0.6, 0.35, -0.25),
    "radial_gaussian_center": (3.0, 0.2, 0.8, 0.35, -0.25),
    # Amplitudes ten times the Gaussian models': here A multiplies the unit
    # lattice shape, whose peak is sigma / sqrt(rate + sigma^2) ~ 0.12, so
    # these put ~70 counts in the tallest bin like the Gaussian rows do.
    "histogram_poisson_gaussian": (900.0, 4.0, 0.3),
    "bimodal_poisson_gaussian": (1.0, 6.0, 600.0, 0.3, 450.0, 0.35),
}


def _coordinates(model_id: str) -> tuple[np.ndarray, ...]:
    if model_id in _POISSON_MODELS:
        # Photon counts on the integer lattice at a quarter-photon bin with a
        # 0.3-photon read noise: the comb is visible (its contrast is
        # exp(-2 pi^2 sigma^2), 17% here and 0.1% at sigma 0.58), so the
        # width is a resolved quantity, the optimum is sharp and two solvers
        # land on the same point.  At twenty photons the width would be a 3%
        # share of the variance -- physically unidentifiable, and a flat
        # valley two solvers stop in differently.
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
        parameters[[0, 1, 2]] += (12.0 * position, 0.6 * position, 0.12 * position)
    elif model_id == "bimodal_gaussian":
        parameters += np.asarray(
            (0.4, 0.25, 8.0, 0.08, -6.0, -0.08)
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
    elif model_id == "histogram_poisson_gaussian":
        parameters[[0, 1, 2]] += (120.0 * position, 0.8 * position, 0.03 * position)
    elif model_id == "bimodal_poisson_gaussian":
        parameters += np.asarray(
            (0.2, 0.8, 80.0, 0.02, -60.0, -0.02)
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
    np.testing.assert_allclose(
        actual.covariance,
        expected.covariance,
        rtol=1e-6,
        atol=1e-10,
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
        assert np.allclose(actual, expected, rtol=2e-12, atol=2e-12), model


@pytest.mark.parametrize("model", tuple(PARAMETERS))
def test_every_builtin_model_recovers_synthetic_parameters(model: str) -> None:
    engine = FitEngine()
    spec = engine.registry.get(model)
    coordinates, observations, _ = _anchor(model)
    expected = tuple(
        float(_anchors()[model]["parameters"][name])
        for name in spec.parameter_names
    )
    result = engine.fit(model, coordinates, observations, data_revision=11)
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
