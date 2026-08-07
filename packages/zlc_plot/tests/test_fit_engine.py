from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zlc_plot import FitCancelled
from zlc_plot.fit import FitEngine, FitOptions, RegularImageFitInput, builtin_fit_models


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
