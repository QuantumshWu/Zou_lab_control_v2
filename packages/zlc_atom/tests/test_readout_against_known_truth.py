"""The readout, checked against a run whose occupancy is known.

The fixture is a SIMULATED run: sixty cycles of two reference frames and one
short frame over a six-site lattice, and the occupancy that produced them.
That last array is what makes this a test rather than a comparison -- the
question is whether the readout recovers the atoms that were there, not
whether it reproduces what some earlier implementation printed.

It replaces a frozen equivalence oracle: a hundred and ten arrays of one
implementation's output, compared to twelve decimal places.  Such a fixture
cannot survive an improvement to the thing it measures -- every one of these
tests failed the moment site detection got better -- and it cannot say whether
either version was RIGHT.  What is kept from it is the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zlc_atom.nodes.calibration.calibration import (
    FrameContract,
    ReadoutModelKind,
    SiteMap,
    _empirical_threshold,
    _fit_readout_model,
    calibrate,
    detect_sites,
    fit_bimodal,
)
from zlc_plot.fit import DECISIVE_BIC_GAIN
from zlc_atom.nodes.slm_feedback.task import _support

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUN = FIXTURES / "main_readout_oracle.npz"
MANIFEST = FIXTURES / "main_readout_oracle.json"

MODEL_NAMES = ("box", "psf", "uniform_psf")

#: Gaussian calibration is deliberately blind to the true labels.  On this
#: short, overlapping 60-shot run the unsupervised population fit measures
#: 0.869 / 0.908 / 0.908 agreement and 0.780 at the worst site; the labels are
#: used below only to evaluate those answers.
AGREEMENT_FLOOR = 0.85
SITE_FIDELITY_FLOOR = 0.75


def _run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(RUN, allow_pickle=False) as archive:
        return (
            np.array(archive["input_reference_frames"], copy=True),
            np.array(archive["input_short_frames"], copy=True),
            np.array(archive["input_latent_occupancy"], copy=True),
        )


def _calibration(threshold_method: str = "gaussian"):
    reference, short, truth = _run()
    result = calibrate(
        reference,
        short,
        frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
        threshold_method=threshold_method,
        box_half_width=1,
        psf_half_width=3,
        psf_padding=3,
    )
    return result, truth


def test_the_fixture_is_a_run_and_its_truth() -> None:
    """Only the data is frozen, and the manifest says what it is."""

    with np.load(RUN, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "input_reference_frames",
            "input_short_frames",
            "input_latent_occupancy",
        }
    reference, short, truth = _run()
    assert reference.shape == (60, 2, 34, 40)
    assert short.shape == (60, 34, 40)
    assert truth.shape == (60, 6) and truth.dtype == np.bool_
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["format"] == "readout-known-truth-run"
    assert manifest["input"]["reference_shape"] == [60, 2, 34, 40]
    assert manifest["input"]["sites"] == 6


def test_the_detector_finds_the_lattice_that_is_there() -> None:
    """Six traps, each on its own spot, none invented."""

    reference, _short, truth = _run()
    average = reference.reshape(-1, *reference.shape[2:]).astype(float).mean(0)
    observed = detect_sites(average)

    assert observed.n_sites == truth.shape[1]
    assert observed.site_ids == tuple(
        f"site_{index:04d}" for index in range(observed.n_sites)
    )
    centres = np.asarray(observed.centers_xy, dtype=float)
    # Every centre sits on its own spot: within half a pixel of the brightest
    # pixel round it, which is as close as a pixel grid can say.
    for x, y in centres:
        column, row = int(round(float(x))), int(round(float(y)))
        patch = average[row - 1 : row + 2, column - 1 : column + 2]
        offset = np.unravel_index(int(np.argmax(patch)), patch.shape)
        assert (row - 1 + offset[0], column - 1 + offset[1]) == (row, column)
    # And no trap is reported twice.
    for index, centre in enumerate(centres):
        others = np.delete(centres, index, axis=0)
        assert float(np.min(np.hypot(*(others - centre).T))) > 2.0


def test_every_model_recovers_the_occupancy_that_produced_the_frames() -> None:
    """The whole point of a readout, on a run where the answer is known."""

    result, truth = _calibration()
    assert tuple(model.kind.value for model in result.calibration.models) == MODEL_NAMES

    for name in MODEL_NAMES:
        report = result.report["models"][name]
        predicted = np.asarray(report["predictions"], dtype=bool)
        assert predicted.shape == truth.shape
        agreement = float((predicted == truth).mean())
        assert agreement >= AGREEMENT_FLOOR, (name, agreement)
        worst = float(np.nanmin(np.asarray(report["site_fidelity"], dtype=float)))
        assert worst >= SITE_FIDELITY_FLOOR, (name, worst)


def test_a_threshold_is_the_weighted_crossing_of_the_unlabelled_fit() -> None:
    """Gaussian calibration fits values only; labels evaluate its result."""

    result, _truth = _calibration()
    for model in result.calibration.models:
        report = result.report["models"][model.kind.value]
        assert model.threshold_method == "gaussian"
        assert not np.any(report["threshold_fallback"])
        np.testing.assert_allclose(
            model.thresholds, report["gaussian_thresholds"]
        )
        for site, threshold in enumerate(model.thresholds):
            dark_mean = float(report["gaussian_dark_mean"][site])
            dark_sigma = float(report["gaussian_dark_sigma"][site])
            dark_weight = float(report["gaussian_dark_weight"][site])
            bright_mean = float(report["gaussian_bright_mean"][site])
            bright_sigma = float(report["gaussian_bright_sigma"][site])
            bright_weight = float(report["gaussian_bright_weight"][site])
            assert dark_mean < threshold < bright_mean
            dark_log_curve = (
                np.log(dark_weight)
                - np.log(dark_sigma)
                - 0.5 * ((threshold - dark_mean) / dark_sigma) ** 2
            )
            bright_log_curve = (
                np.log(bright_weight)
                - np.log(bright_sigma)
                - 0.5 * ((threshold - bright_mean) / bright_sigma) ** 2
            )
            assert dark_log_curve == pytest.approx(bright_log_curve, abs=1e-10)

    # A visibly narrower bright population is allowed, and two remote tail
    # samples do not make the fitter replace it with one broad component.
    rng = np.random.default_rng(91)
    narrow_bright = fit_bimodal(
        np.concatenate(
            (
                rng.normal(0.0, 1.0, 100),
                rng.normal(5.0, 0.35, 100),
                [-4.5, 8.5],
            )
        )
    )
    assert narrow_bright.ok
    assert narrow_bright.bright_sigma < narrow_bright.dark_sigma
    assert narrow_bright.dark_mean == pytest.approx(0.0, abs=0.4)
    assert narrow_bright.bright_mean == pytest.approx(5.0, abs=0.2)

    # The widths are the sample's own.  Bright shot noise eight times the
    # dark read noise, fifty sigma of separation, both states heavily
    # populated: the fit reports that ratio (the dark width sits on the
    # one-percent-of-spread floor) and is valid.  A fixed width ratio once
    # floored the narrow state at a fifth of the wide one and ranked any
    # pair inside the bound above the likelier one on it; this site was
    # then reported to the SLM feedback loop as one that did not load.
    rng = np.random.default_rng(17)
    loaded = rng.random(120) < 0.45
    eightfold = fit_bimodal(
        np.where(
            loaded, rng.normal(130.0, 4.0, 120), rng.normal(5.9, 0.5, 120)
        )
    )
    assert eightfold.ok, (eightfold.dark_sigma, eightfold.bright_sigma)
    assert 5.0 < eightfold.bright_sigma / eightfold.dark_sigma < 10.0
    assert eightfold.dark_mean < eightfold.threshold < eightfold.bright_mean

    # Changing truth labels cannot move a Gaussian model or threshold.  It can
    # only change the empirical evaluation of that already chosen threshold.
    samples = np.concatenate(
        (rng.normal(0.0, 0.5, 80), rng.normal(4.0, 0.8, 120))
    )[:, np.newaxis]
    labels_a = np.arange(samples.shape[0])[:, np.newaxis] >= 80
    labels_b = ~labels_a
    fitted = []
    for labels in (labels_a, labels_b):
        fitted.append(
            _fit_readout_model(
                kind=ReadoutModelKind.BOX,
                site_map=SiteMap(
                    ("site_0000",), [[0.0, 0.0]], [True], [1.0]
                ),
                short_signals=samples,
                labels_occupied=labels,
                labels_valid=np.ones(samples.shape, dtype=bool),
                threshold_method="gaussian",
                model_parameters={"integration_half_width": 0},
            )
        )
    for field in (
        "gaussian_thresholds",
        "gaussian_dark_mean",
        "gaussian_dark_sigma",
        "gaussian_dark_weight",
        "gaussian_bright_mean",
        "gaussian_bright_sigma",
        "gaussian_bright_weight",
    ):
        np.testing.assert_allclose(fitted[0][1][field], fitted[1][1][field])
    assert fitted[0][1]["site_fidelity"][0] != fitted[1][1]["site_fidelity"][0]

    empirical, _truth = _calibration("empirical")
    for model in empirical.calibration.models:
        report = empirical.report["models"][model.kind.value]
        assert model.threshold_method == "empirical"
        assert not np.any(report["threshold_fallback"])
        assert np.all(np.isfinite(report["site_gaussian_fidelity"]))
        assert np.any(
            ~np.isclose(model.thresholds, report["gaussian_thresholds"])
        )

    fallback_model, fallback_report = _fit_readout_model(
        kind=ReadoutModelKind.BOX,
        site_map=SiteMap(("site_0000",), [[0.0, 0.0]], [True], [1.0]),
        short_signals=np.asarray([[0.0], [10.0]]),
        labels_occupied=np.asarray([[False], [True]]),
        labels_valid=np.ones((2, 1), dtype=bool),
        threshold_method="gaussian",
        model_parameters={"integration_half_width": 0},
    )
    np.testing.assert_allclose(fallback_model.thresholds, [5.0])
    np.testing.assert_array_equal(fallback_report["threshold_fallback"], [True])
    np.testing.assert_allclose(fallback_report["site_fidelity"], [1.0])
    assert np.isnan(fallback_report["site_gaussian_fidelity"][0])
    assert _empirical_threshold(
        [0.0, 10.0], [0.0, 10.0], bright_above=True
    ) == 5.0


def test_the_crossing_is_stable_when_the_two_widths_nearly_agree() -> None:
    """The threshold is where the weighted curves cross, whatever the widths.

    With nearly equal widths the quadratic's leading coefficient vanishes
    and the relevant root was formed by subtracting two nearly equal
    numbers: means 0 and 4, widths 1 and 1+1e-15, weights 0.4/0.6 answered
    1.80 where the curves cross at 1.8986 -- a threshold the caller accepted
    because it lay between the means -- while exactly equal widths took the
    linear branch and were right.  The crossing must vary continuously
    through equal widths.
    """

    from zlc_atom.nodes.calibration.bimodal import optimal_gaussian_threshold

    equal, above = optimal_gaussian_threshold(0.0, 1.0, 4.0, 1.0, 0.4, 0.6)
    assert above
    assert equal == pytest.approx(1.8986337229729588, abs=1e-12)
    for delta in (1e-10, 1e-12, 1e-14, 1e-15):
        threshold, above = optimal_gaussian_threshold(
            0.0, 1.0, 4.0, 1.0 + delta, 0.4, 0.6
        )
        assert above
        assert threshold == pytest.approx(equal, abs=1e-8), delta
        width = 1.0 + delta
        dark_log_curve = np.log(0.4) - 0.5 * threshold**2
        bright_log_curve = (
            np.log(0.6) - np.log(width) - 0.5 * ((threshold - 4.0) / width) ** 2
        )
        assert dark_log_curve == pytest.approx(bright_log_curve, abs=1e-9), delta


def test_a_crop_whose_boxes_are_wholly_readable_is_accepted() -> None:
    """Rebase asks the one box rule the readout itself asks.

    The readout rounds a centre to its pixel and reads the box round it;
    rebase judged the continuous centre against a continuous bound, so a
    site at x=0.8 with radius 1 -- whose 3x3 box, x=0..2, the readout reads
    in full -- was refused as uncovered by a crop that covers it.
    """

    from zlc_atom.nodes.calibration.calibration import (
        FrameContract,
        ReadoutModel,
        ReadoutModelKind,
        SiteMap,
        TrapCalibration,
        box_fits,
        extract_box_signals,
    )

    sites = SiteMap(("site_0000",), [[3.8, 4.0]], [True], [1.0])
    model = ReadoutModel(
        sites.site_ids, [5.0], [0.0], [10.0], [True], [1.0], integration_half_width=1
    )
    source = TrapCalibration(
        sites,
        (model,),
        ReadoutModelKind.BOX,
        FrameContract((10, 10), sensor_shape=(10, 10), roi_xywh=(0, 0, 10, 10)),
    )
    assert box_fits((0.8, 4.0), 1, (10, 7))
    assert extract_box_signals(np.ones((10, 7)), [[0.8, 4.0]], radius=1).tolist() == [9.0]
    moved = source.rebased((3, 0, 7, 10), (1, 1), (10, 7))
    assert moved.site_map.centers_xy.tolist() == [[pytest.approx(0.8), 4.0]]
    with pytest.raises(ValueError, match="does not cover"):
        source.rebased((4, 0, 6, 10), (1, 1), (10, 6))


def test_no_model_is_recommended_when_none_read_a_site() -> None:
    """A summary with nothing to compare names nothing.

    Two shots of zero signal give a BOX model with no usable site and no
    fidelity; the summary still crowned it ``best_model`` and starred it in
    the progress lines, which reads as advice to use it.
    """

    from zlc_atom.nodes.calibration.calibration import (
        CalibrationResult,
        FrameContract,
        ReadoutModelKind,
        SiteMap,
        TrapCalibration,
        _fit_readout_model,
    )
    from zlc_atom.nodes.calibration.summary import readout_summary, summary_lines

    sites = SiteMap(("site_0000",), [[1.0, 1.0]], [True], [1.0])
    occupied = np.array([[False], [True]])
    valid = np.ones((2, 1), dtype=bool)
    model, model_report = _fit_readout_model(
        kind=ReadoutModelKind.BOX,
        site_map=sites,
        short_signals=np.array([[0.0], [0.0]]),
        labels_occupied=occupied,
        labels_valid=valid,
        threshold_method="gaussian",
        model_parameters={"integration_half_width": 1},
    )
    assert not model.usable_sites.any()
    summary = readout_summary(
        CalibrationResult(
            TrapCalibration(sites, (model,), ReadoutModelKind.BOX, FrameContract((3, 3))),
            {
                "labels_occupied": occupied,
                "labels_valid": valid,
                "models": {"box": model_report},
            },
        )
    )
    assert summary["best_model"] is None
    assert summary["models"]["box"]["usable_sites"] == 0
    assert not any(line.startswith("*") for line in summary_lines(summary))


def test_a_psf_kernel_is_the_spot_it_was_measured_from() -> None:
    """Amplitude-scaled, peaked on the site, concentrated where the light is.

    Not non-negative: the kernel is a MEASURED difference image, so its wings
    carry the noise of the shots it was measured from, and a pixel where the
    atom happened to subtract a little is worth what it is worth.  What must
    be true is that the weight is where the atom is -- and that the filter
    answers in the site's own total counts: applied to its own unit-total
    shape it must report exactly 1, so sum(w * p) with p = w / sum(w) is 1
    (equivalently w = p / sum(p^2), the least-squares amplitude of the
    measured pattern; a flat pattern degenerates to the box sum).
    """

    result, _truth = _calibration()
    per_site = result.calibration.select_model(ReadoutModelKind.PER_SITE_PSF)
    uniform = result.calibration.select_model(ReadoutModelKind.UNIFORM_PSF)

    for kernels in (np.asarray(per_site.psf_weights), np.asarray(uniform.psf_weights)):
        assert kernels.ndim == 3
        totals = kernels.sum(axis=(1, 2))
        powers = np.square(kernels).sum(axis=(1, 2))
        np.testing.assert_allclose(powers / totals, 1.0, atol=1e-9)
        for kernel in kernels:
            peak = np.unravel_index(int(np.argmax(kernel)), kernel.shape)
            centre = (kernel.shape[0] // 2, kernel.shape[1] // 2)
            assert abs(peak[0] - centre[0]) <= 1 and abs(peak[1] - centre[1]) <= 1
            shape = kernel / float(kernel.sum())
            core = shape[
                centre[0] - 1 : centre[0] + 2, centre[1] - 1 : centre[1] + 2
            ]
            assert float(core.sum()) > 0.5, "most of the light sits on the spot"
            edge = float(np.max(np.abs(kernel[0]))), float(np.max(np.abs(kernel[-1])))
            assert max(edge) < float(kernel[centre]) / 5.0, "the wings are wings"

    boxes = np.asarray(per_site.psf_boxes, dtype=int)
    centres = np.asarray(result.calibration.site_map.centers_xy, dtype=float)
    for (x, y, width, height), (centre_x, centre_y) in zip(boxes, centres):
        assert x <= centre_x <= x + width - 1
        assert y <= centre_y <= y + height - 1


def test_weighting_beats_a_box_when_the_spot_is_wider_than_the_box() -> None:
    """Why the PSF models exist, on data where the difference can exist.

    A box collects what falls inside it; a PSF weighting also uses the light
    outside it and discounts the pixels that carry more noise than signal.  So
    the two can only differ when the spot is wider than the box -- on a run
    whose spots fit inside a 3x3 there is nothing left to gather, and the
    comparison has no physical readout difference to measure.

    Here the box holds a little over half the light.  Measured on the bench's
    own 35-trap run, where the same is true: 0.9986 per-site fidelity for the
    PSF against 0.9930 for the box, and a worst site of 0.95 against 0.90.
    """

    rng = np.random.default_rng(5)
    cycles, height, width = 200, 44, 44
    centres = [(11.0 + 11.0 * column, 11.0 + 11.0 * row) for row in range(3) for column in range(3)]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    spots = np.stack(
        [
            np.exp(-(((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * 1.5**2)))
            for x, y in centres
        ]
    )
    occupancy = rng.random((cycles, len(centres))) < 0.5
    reference = rng.normal(200.0, 12.0, size=(cycles, 2, height, width))
    short = rng.normal(200.0, 12.0, size=(cycles, height, width))
    for site in range(len(centres)):
        lit = occupancy[:, site]
        reference[lit, 0] += 900.0 * spots[site]
        reference[lit, 1] += 900.0 * spots[site]
        short[lit] += 260.0 * spots[site]

    result = calibrate(
        reference,
        short,
        frame_contract=FrameContract((height, width), exposure_seconds=0.005),
        box_half_width=1,
        psf_half_width=3,
        psf_padding=3,
        detection_spot_sigma=1.5,
    )
    assert result.calibration.site_map.n_sites == len(centres)

    fidelity = {
        name: np.nanmean(
            np.asarray(result.report["models"][name]["site_fidelity"], dtype=float)
        )
        for name in MODEL_NAMES
    }
    assert fidelity["psf"] >= fidelity["box"], fidelity
    assert fidelity["uniform_psf"] >= fidelity["box"], fidelity


def test_target_registration_keeps_a_never_loaded_site_as_unresolved(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(41)
    cycles, height, width = 100, 50, 50
    support_yx = [
        (row, column)
        for row in (4, 10, 16)
        for column in (3, 9, 15)
    ]
    support_yx[0] = (3, 2)
    camera_centers = [
        (10.0 + (5.0 / 3.0) * column, (25.0 / 3.0) + (5.0 / 3.0) * row)
        for row, column in support_yx
    ]
    missing = 4
    grid_y, grid_x = np.mgrid[:height, :width]
    spots = np.stack(
        [
            np.exp(
                -(
                    (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
                )
                / (2.0 * 1.1**2)
            )
            for center_x, center_y in camera_centers
        ]
    )
    occupied = rng.random((cycles, len(camera_centers))) < 0.5
    occupied[:, missing] = False
    reference = rng.normal(100.0, 3.0, (cycles, 2, height, width))
    short = rng.normal(100.0, 3.0, (cycles, height, width))
    for site, spot in enumerate(spots):
        if site == missing:
            continue
        reference[occupied[:, site], 0] += 800.0 * spot
        reference[occupied[:, site], 1] += 800.0 * spot
        short[occupied[:, site]] += 200.0 * spot

    target = np.zeros((20, 20), dtype=np.float32)
    for row, column in support_yx:
        target[row, column] = 1.0
    context_path = tmp_path / "contexts" / "nine.npz"
    receipt = {
        "identity": "slm",
        "mapping_revision": 7,
    }
    result = calibrate(
        reference,
        short,
        frame_contract=FrameContract((height, width), exposure_seconds=0.005),
        box_half_width=1,
        psf_half_width=2,
        psf_padding=2,
        detection_spot_sigma=1.1,
    )

    saved = result.calibration.save(tmp_path / "generic-calibration.json")
    calibration = type(result.calibration).load(saved)
    assert calibration.site_map.n_sites == 8
    rows, columns, registered = _support(
        target,
        calibration,
        box_half_width=1,
        science_context_path=context_path,
        command_receipt=receipt,
    )
    assert len(rows) == len(columns) == registered.n_sites == 9
    np.testing.assert_allclose(
        registered.centers_xy[missing], camera_centers[missing], atol=0.1
    )
    assert (rows[missing], columns[missing]) == (10, 9)
    assert not registered.topology["observed_sites"][missing]
    assert not registered.valid_sites[missing]


def test_the_two_state_fit_takes_the_likeliest_pair_whatever_their_widths() -> None:
    """A calibration figure showed, over a three-wide dark spike at 5 and a
    thirty-wide bright peak at 150, a forty-wide "dark" Gaussian and a
    threshold at 100: the likeliest pair sat outside a fixed width ratio and
    any pair inside it was preferred.  The fit ranks by likelihood alone,
    reports the widths the shots have, and its ``decisive`` is the evidence
    for two populations -- so a site that never loaded is not two states,
    and five loaded shots in a hundred are."""

    for seed in range(20):
        rng = np.random.default_rng(seed)
        dark = int(rng.binomial(100, 0.6))
        fit = fit_bimodal(
            np.concatenate(
                (rng.normal(5.0, 2.5, dark), rng.normal(150.0, 30.0, 100 - dark))
            )
        )
        assert fit.decisive, (seed, fit)
        assert fit.dark_mean == pytest.approx(5.0, abs=2.0)
        assert fit.dark_sigma < 5.0
        assert fit.bright_mean == pytest.approx(150.0, abs=15.0)
        assert 15.0 < fit.threshold < 100.0
        assert fit.bright_fraction == pytest.approx(1.0 - dark / 100.0, abs=0.03)
        assert fit.bic_gain > 10.0 * DECISIVE_BIC_GAIN

    rng = np.random.default_rng(5)
    narrow_dark = fit_bimodal(
        np.concatenate((rng.normal(5.0, 0.5, 60), rng.normal(150.0, 30.0, 40)))
    )
    assert narrow_dark.decisive
    assert narrow_dark.bright_sigma / narrow_dark.dark_sigma > 20.0

    one_state = fit_bimodal(rng.normal(5.0, 2.5, 100))
    assert not one_state.decisive
    assert one_state.bic_gain < DECISIVE_BIC_GAIN
    five_loaded = fit_bimodal(
        np.concatenate((rng.normal(5.0, 2.5, 95), rng.normal(150.0, 30.0, 5)))
    )
    assert five_loaded.decisive
    assert five_loaded.bright_fraction == pytest.approx(0.05, abs=0.02)
    two_loaded = fit_bimodal(
        np.concatenate((rng.normal(5.0, 2.5, 98), rng.normal(150.0, 30.0, 2)))
    )
    assert not two_loaded.ok

    # Two populations that overlap at sixty shots: ok, and not decisive.  Six
    # of them within half a unit are not a third state either -- the cost on
    # the width ratio is what a chance cluster cannot pay.
    rng = np.random.default_rng(11)
    overlapping = fit_bimodal(
        np.concatenate((rng.normal(92.0, 4.0, 30), rng.normal(104.0, 4.0, 30)))
    )
    assert overlapping.ok and not overlapping.decisive
    assert overlapping.dark_sigma > 1.5 and overlapping.bright_sigma > 1.5
    assert 94.0 < overlapping.threshold < 102.0
