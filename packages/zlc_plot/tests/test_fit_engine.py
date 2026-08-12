from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zlc_plot import FitCancelled
from zlc_plot.fit import (
    FitEngine,
    FitOptions,
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
}

_ANCHOR_PATH = Path(__file__).with_name("fixtures") / "fit_anchors.json"

# Models whose origin is the start of the window they are fitted over.
_ANCHORED_MODELS = ("damped_sine", "exponential_decay")


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


def test_regular_image_warm_start_reproduces_the_cold_solution() -> None:
    engine = FitEngine()
    x, y, image = _separable_image(radial=True, size=96)
    data = RegularImageFitInput(x, y, image)
    cold = engine.fit("radial_gaussian_center", data)
    warm = engine.fit(
        "radial_gaussian_center",
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
