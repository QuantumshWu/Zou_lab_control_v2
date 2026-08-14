"""A rare state must not be fitted away, and a live fit must be able to learn.

Both failures were real and both were reported as success.  With counts
weighted by 1/sqrt(n), the best two-Gaussian fit of a rare state over a
common one is a spike one bin wide standing on the common peak: it is not a
local trap, it is the optimum of that objective (measured on the same 430
shots: 16.21 for the spike against 16.55 for the truth).  And a classifier
refresh handed its previous parameters in as ``initial``, which REPLACES the
model's own seed, with a fallback only when the solve RAISED -- so a
wrong-but-converged answer became every later revision's only starting point.
"""

from __future__ import annotations

import numpy as np

from zlc_plot.fit import FitEngine, _bimodal_classifier_metrics

BINS = 60
RANGE = (40.0, 260.0)
DARK = (100.0, 9.0)
BRIGHT = (180.0, 13.0)
#: Half a bin: the width floor, and the value a collapsed component sits at.
FLOOR = 0.5 * (RANGE[1] - RANGE[0]) / BINS


def _histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=BINS, range=RANGE)
    return 0.5 * (edges[:-1] + edges[1:]), counts.astype(float)


def _draw(rng: np.random.Generator, count: int, fraction: float) -> np.ndarray:
    bright = rng.random(count) < fraction
    return np.where(
        bright, rng.normal(*BRIGHT, count), rng.normal(*DARK, count)
    )


def _named(result) -> dict[str, float]:
    return {
        parameter.name: float(value)
        for parameter, value in zip(
            result.model.parameters, result.parameter_values, strict=True
        )
    }


def test_a_rare_state_is_found_and_not_fitted_away() -> None:
    """Two per cent bright: the case the old objective always lost."""

    rng = np.random.default_rng(11)
    centres, counts = _histogram(_draw(rng, 2000, 0.02))
    result = FitEngine().fit("bimodal_gaussian", (centres,), counts, data_revision=0)
    values = _named(result)
    low = values["center"] - 0.5 * values["center_splitting"]
    high = values["center"] + 0.5 * values["center_splitting"]

    assert abs(low - DARK[0]) < 2.0, values
    assert abs(high - BRIGHT[0]) < 5.0, values
    assert min(values["left_sigma"], values["right_sigma"]) > 2.0 * FLOOR, values
    threshold, _left, right, _fidelity = _bimodal_classifier_metrics(result, None)
    assert DARK[0] < threshold < BRIGHT[0]
    assert 0.005 < right < 0.10, right


def test_a_live_classifier_leaves_an_early_wrong_answer_behind() -> None:
    """Thirty shots cannot show a rare state; four hundred can.

    The whole live sequence, seeded exactly as the classifier seeds it.
    """

    rng = np.random.default_rng(11)
    values = _draw(rng, 30, 0.02)
    warm = None
    thresholds = []
    for step in range(11):
        if step:
            values = np.concatenate([values, _draw(rng, 40, 0.02)])
        centres, counts = _histogram(values)
        result = FitEngine().fit(
            "bimodal_gaussian",
            (centres,),
            counts,
            data_revision=step,
            warm_start=warm,
        )
        warm = tuple(float(value) for value in result.parameter_values)
        thresholds.append(_bimodal_classifier_metrics(result, None)[0])
        assert min(_named(result).values(), default=1.0) is not None

    assert DARK[0] < thresholds[-1] < BRIGHT[0], thresholds
    assert thresholds[-1] > thresholds[0] + 20.0, thresholds


def test_a_run_that_has_not_loaded_yet_publishes_no_threshold() -> None:
    """Twenty runs, three loading rates, eleven revisions each.

    A two-state fit of thirty shots with no atom in them still converges --
    one component covers the dark peak and the other sits exactly on a single
    stray count, because under a likelihood that IS worth something.  What
    must not happen is that such a fit reports a threshold and a fidelity.
    Measured here: 15 of 220 fits put a component on one bin, and every one
    of them holds fewer than four shots.
    """

    engine = FitEngine()
    spikes = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        fraction = (0.02, 0.1, 0.5)[seed % 3]
        values = _draw(rng, 30, fraction)
        warm = None
        for step in range(11):
            if step:
                values = np.concatenate([values, _draw(rng, 40, fraction)])
            centres, counts = _histogram(values)
            result = engine.fit(
                "bimodal_gaussian",
                (centres,),
                counts,
                data_revision=step,
                warm_start=warm,
            )
            warm = tuple(float(value) for value in result.parameter_values)
            named = _named(result)
            threshold = _bimodal_classifier_metrics(result, None)[0]
            if min(named["left_sigma"], named["right_sigma"]) <= FLOOR * 1.01:
                spikes += 1
                assert threshold is None, (seed, step, named)
        assert threshold is not None, (seed, named)
        assert DARK[0] + 15.0 < threshold < BRIGHT[0] - 10.0, (seed, threshold)
    assert spikes < 40, spikes


def test_a_spike_does_not_survive_being_seeded() -> None:
    """Started ON the old failure, the fit walks away from it.

    Under the old objective this was the global optimum, so nothing could
    walk away: seeded there or not, it stayed.
    """

    rng = np.random.default_rng(11)
    centres, counts = _histogram(_draw(rng, 430, 0.02))
    engine = FitEngine()
    truth = engine.fit("bimodal_gaussian", (centres,), counts, data_revision=0)

    spike = dict(_named(truth))
    step = centres[1] - centres[0]
    peak = float(centres[int(np.argmax(counts))])
    spike.update(
        {
            "center": peak,
            "center_splitting": 0.0,
            "left_sigma": 0.5 * step,
            "left_amplitude": float(np.max(counts)),
        }
    )
    seeded = engine.fit(
        "bimodal_gaussian",
        (centres,),
        counts,
        data_revision=0,
        initial=spike,
    )
    escaped = _named(seeded)
    assert min(escaped["left_sigma"], escaped["right_sigma"]) > 2.0 * FLOOR, escaped
    assert np.allclose(
        seeded.parameter_values, truth.parameter_values, rtol=1e-3, atol=1e-3
    ), (escaped, _named(truth))
