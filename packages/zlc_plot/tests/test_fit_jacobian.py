from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from zlc_plot.fit import FitEngine, FitOptions, builtin_fit_models


def _coordinates(model_id: str) -> tuple[np.ndarray, ...]:
    x = np.linspace(-1.3, 1.4, 41)
    if model_id in {"radial_gaussian_center", "anisotropic_gaussian_center"}:
        y = np.linspace(-1.1, 1.2, 37)
        xx, yy = np.meshgrid(x, y)
        return xx.reshape(-1), yy.reshape(-1)
    return (x,)


def _parameters(model_id: str) -> np.ndarray:
    return {
        "lorentzian": np.array([0.2, 0.8, 1.4, 0.1]),
        "gaussian_offset": np.array([1.2, 0.1, 0.7, 0.2]),
        "histogram_gaussian": np.array([1.2, 0.2, 0.7]),
        "bimodal_gaussian": np.array([0.1, 0.8, 1.2, 0.4, 0.9, 0.5]),
        "symmetric_lorentzian_doublet": np.array([0.1, 0.7, 1.2, 0.1, 0.8]),
        "damped_sine": np.array([1.2, 0.1, 0.4, 1.8, 0.3]),
        "exponential_decay": np.array([1.2, 0.1, 1.4]),
        "anisotropic_gaussian_center": np.array([1.2, 0.1, 1.1, 0.8, 0.2, -0.1]),
        "radial_gaussian_center": np.array([1.2, 0.1, 1.1, 0.2, -0.1]),
    }[model_id].copy()


def test_declared_jacobians_match_five_numerical_derivatives() -> None:
    rng = np.random.default_rng(20260804)
    for model in builtin_fit_models():
        assert model.jacobian is not None
        coordinates = _coordinates(model.model_id)
        for _ in range(5):
            parameters = _parameters(model.model_id)
            parameters += rng.normal(0.0, 0.03, parameters.size)
            for index, spec in enumerate(model.parameters):
                if spec.domain.value in {"positive", "nonnegative"}:
                    parameters[index] = max(abs(parameters[index]), 0.05)
            analytic = model.evaluate_jacobian(coordinates, parameters)
            numeric = approx_derivative(
                lambda values: model.evaluate(coordinates, values).reshape(-1),
                parameters,
                method="3-point",
            )
            assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_analytic_and_numeric_fit_results_are_equivalent() -> None:
    options = FitOptions(max_nfev=2000)
    engine = FitEngine()
    for model in builtin_fit_models():
        coordinates = _coordinates(model.model_id)
        parameters = _parameters(model.model_id)
        observations = model.evaluate(coordinates, parameters)
        observations = observations + 0.001 * np.sin(np.arange(observations.size))
        analytic = engine.fit(
            model,
            coordinates,
            observations,
            initial=parameters,
            options=options,
        )
        numeric = engine.fit(
            replace(model, jacobian=None),
            coordinates,
            observations,
            initial=parameters,
            options=options,
        )
        assert analytic.success == numeric.success
        assert np.allclose(
            analytic.parameter_values,
            numeric.parameter_values,
            rtol=1e-6,
            atol=1e-8,
        )
        assert np.allclose(
            analytic.standard_errors,
            numeric.standard_errors,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        )
