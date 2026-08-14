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

from zlc_atom.nodes.calibration.calibration import (
    FrameContract,
    ReadoutModelKind,
    calibrate,
    detect_sites,
    fit_bimodal,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUN = FIXTURES / "main_readout_oracle.npz"
MANIFEST = FIXTURES / "main_readout_oracle.json"

MODEL_NAMES = ("box", "psf", "uniform_psf")

#: Every model must agree with the occupancy that produced the frames at least
#: this often, and no site may fall below the second floor.  Measured on this
#: run: 0.919 / 0.925 / 0.928 overall, and 0.833 for the worst site of any
#: model.  The floors sit below those because a readout is allowed to improve.
AGREEMENT_FLOOR = 0.90
SITE_FIDELITY_FLOOR = 0.80


def _run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(RUN, allow_pickle=False) as archive:
        return (
            np.array(archive["input_reference_frames"], copy=True),
            np.array(archive["input_short_frames"], copy=True),
            np.array(archive["input_latent_occupancy"], copy=True),
        )


def _calibration():
    reference, short, truth = _run()
    result = calibrate(
        reference,
        short,
        frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
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


def test_a_threshold_separates_the_two_populations_it_was_fitted_to() -> None:
    """A site's threshold lies between its dark and bright means, and sorts."""

    result, truth = _calibration()
    signals = np.asarray(result.report["reference_label_signals"], dtype=float)
    for site in range(truth.shape[1]):
        fit = fit_bimodal(signals[:, :, site].reshape(-1))
        assert fit.ok and fit.bright_above
        assert fit.dark_mean < fit.threshold < fit.bright_mean
        assert fit.dark_sigma > 0.0 and fit.bright_sigma > 0.0
        assert 0.0 <= fit.fidelity <= 1.0


def test_a_psf_kernel_is_a_normalised_weighting_of_its_own_box() -> None:
    """Per-site and uniform kernels alike: positive, normalised, centred."""

    result, _truth = _calibration()
    per_site = result.calibration.select_model(ReadoutModelKind.PER_SITE_PSF)
    uniform = result.calibration.select_model(ReadoutModelKind.UNIFORM_PSF)

    for kernels in (np.asarray(per_site.psf_weights), np.asarray(uniform.psf_weights)):
        assert kernels.ndim == 3
        assert np.all(kernels >= 0.0)
        np.testing.assert_allclose(kernels.sum(axis=(1, 2)), 1.0, atol=1e-9)
        for kernel in kernels:
            peak = np.unravel_index(int(np.argmax(kernel)), kernel.shape)
            centre = (kernel.shape[0] // 2, kernel.shape[1] // 2)
            assert abs(peak[0] - centre[0]) <= 1 and abs(peak[1] - centre[1]) <= 1

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
    comparison measures the test split rather than the readout.

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
