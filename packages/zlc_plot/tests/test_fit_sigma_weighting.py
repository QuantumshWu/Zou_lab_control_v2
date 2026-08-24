"""Per-point sigma weighting: known uncertainty steers the solver.

The occupation-rate projection attaches a standard error to every MEAN
point; a present sigma weights the fit residuals by 1/sigma and the quality
report becomes the chi-square.  Zero-spread endpoints (a rate of exactly 0
or 1) take the smallest positive sigma present, and spreadless data falls
back to the ordinary unweighted fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from zlc_plot.fit import FitEngine


# The builtin lorentzian is (center, fwhm, amplitude, offset).
def _lorentzian(x, center, fwhm, amplitude, offset):
    half = fwhm / 2.0
    return offset + amplitude / (1.0 + np.square((x - center) / half))


TRUTH = {"center": 4.0, "fwhm": 1.2, "amplitude": 0.8, "offset": 0.1}


def _values(engine, result):
    names = engine.registry.get("lorentzian").parameter_names
    return dict(zip(names, map(float, result.parameter_values)))


def test_sigma_weighting_recovers_parameters_under_heteroscedastic_noise() -> None:
    """Points with tight sigma dominate; a wildly noisy flank cannot drag
    the recovered peak away."""

    engine = FitEngine()
    x = np.linspace(0.0, 8.0, 60)
    clean = _lorentzian(x, **TRUTH)
    # Tight measurement near the peak, 20x looser on the flanks -- the
    # occupation-rate regime (few shots on coarse points, many on fine).
    # NOTE a deliberate boundary: the multi-start initial-value search is
    # unweighted, so noise rivaling the signal can still pick a wrong basin
    # for BOTH fits; sigma weighting sharpens the solve, it does not replace
    # a sane measurement.
    sigma = np.where(np.abs(x - 4.0) < 2.0, 0.004, 0.08)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        observations = clean + rng.normal(0.0, sigma)
        weighted = _values(
            engine,
            engine.fit(
                "lorentzian", (x,), observations, observation_sigma=sigma
            ),
        )
        unweighted = _values(
            engine, engine.fit("lorentzian", (x,), observations)
        )
        weighted_total = sum(
            abs(weighted[name] - truth) for name, truth in TRUTH.items()
        )
        unweighted_total = sum(
            abs(unweighted[name] - truth) for name, truth in TRUTH.items()
        )
        assert weighted_total <= unweighted_total + 1e-9, seed
        assert abs(weighted["center"] - TRUTH["center"]) < 0.02, seed
        assert abs(weighted["fwhm"] - TRUTH["fwhm"]) < 0.05, seed


def test_zero_sigma_endpoints_take_the_smallest_positive_sigma() -> None:
    """A saturated rate point (sem == 0) must neither crash nor dominate."""

    x = np.linspace(0.0, 8.0, 30)
    observations = _lorentzian(x, **TRUTH)
    sigma = np.full_like(x, 0.02)
    sigma[0] = 0.0          # rate exactly 0 across all shots
    sigma[-1] = np.nan      # single-shot bucket: sem undefined
    engine = FitEngine()
    result = engine.fit("lorentzian", (x,), observations, observation_sigma=sigma)
    assert result.success
    values = _values(engine, result)
    for name, truth in TRUTH.items():
        assert np.isclose(values[name], truth, rtol=1e-3, atol=1e-3), name


def test_spreadless_sigma_degrades_to_the_unweighted_fit() -> None:
    x = np.linspace(0.0, 8.0, 30)
    observations = _lorentzian(x, 3.5, 1.0, 0.7, 0.1)
    engine = FitEngine()
    weighted = engine.fit(
        "lorentzian", (x,), observations, observation_sigma=np.zeros_like(x)
    )
    plain = engine.fit("lorentzian", (x,), observations)
    np.testing.assert_allclose(
        weighted.parameter_values, plain.parameter_values, rtol=1e-9
    )


def test_reported_quality_is_the_chi_square_when_sigma_is_present() -> None:
    """chi^2/dof ~ 1 when the noise matches the declared sigma."""

    rng = np.random.default_rng(9)
    x = np.linspace(0.0, 8.0, 200)
    sigma = np.full_like(x, 0.03)
    observations = _lorentzian(x, **TRUTH) + rng.normal(0.0, sigma)
    engine = FitEngine()
    result = engine.fit(
        "lorentzian", (x,), observations, observation_sigma=sigma
    )
    reduced = result.reduced_chi_square
    assert 0.6 < float(reduced) < 1.6


def test_histogram_targets_refuse_an_external_sigma() -> None:
    engine = FitEngine()
    values = np.linspace(-1.0, 1.0, 32)
    counts = np.exp(-0.5 * np.square(values / 0.3))
    with pytest.raises(ValueError, match="counts"):
        engine.fit(
            "histogram_gaussian",
            (values,),
            counts,
            observation_sigma=np.full_like(values, 0.1),
        )
