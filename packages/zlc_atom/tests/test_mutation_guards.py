"""Mutations of the readout must change what it answers.

A guard that only checks a formula against its own output proves nothing: it
passes for any implementation, right or wrong.  These take the readout as it
stands, break one thing on purpose, and require the answer to move -- and
where there is a truth to compare against, to move AWAY from it.

The run they use is the same one the readout is judged on: frames, and the
occupancy that produced them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zlc_atom.nodes.occupancy import OccupancyProcessor
from zlc_atom.nodes.calibration.calibration import (
    FrameContract,
    ReadoutModelKind,
    calibrate,
    classify_threshold,
)

from tests.fakes import camera_cycle_snapshot


RUN = Path(__file__).parent / "fixtures" / "main_readout_oracle.npz"


def _run() -> dict[str, np.ndarray]:
    with np.load(RUN, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _calibrate(data: dict[str, np.ndarray]):
    return calibrate(
        data["input_reference_frames"],
        data["input_short_frames"],
        frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
        box_half_width=1,
        box_reducer="mean",
    )


def test_moving_every_threshold_makes_the_readout_wrong() -> None:
    """A threshold is a real number, not a decoration."""

    data = _run()
    truth = np.asarray(data["input_latent_occupancy"], dtype=bool)
    result = _calibrate(data)
    box_report = result.report["models"]["box"]
    box_model = result.calibration.select_model(ReadoutModelKind.BOX)

    honest = np.asarray(box_report["predictions"], dtype=bool)
    mutated = classify_threshold(
        box_report["short_signals"], box_model.thresholds + 17.3
    )
    assert not np.array_equal(mutated, honest)
    assert float((np.asarray(mutated, dtype=bool) == truth).mean()) < float(
        (honest == truth).mean()
    )


def test_reading_the_threshold_the_wrong_way_round_makes_it_wrong() -> None:
    """Bright is above the threshold; the opposite must not also pass."""

    data = _run()
    truth = np.asarray(data["input_latent_occupancy"], dtype=bool)
    result = _calibrate(data)
    box_report = result.report["models"]["box"]
    box_model = result.calibration.select_model(ReadoutModelKind.BOX)

    honest = np.asarray(box_report["predictions"], dtype=bool)
    flipped = np.asarray(
        classify_threshold(
            box_report["short_signals"],
            box_model.thresholds,
            bright_above=False,
        ),
        dtype=bool,
    )
    assert not np.array_equal(flipped, honest)
    assert float((flipped == truth).mean()) < float((honest == truth).mean())


def test_the_published_rate_is_the_occupied_fraction_and_not_its_inverse() -> None:
    """The rate a panel plots is what the judgements say, counted."""

    data = _run()
    result = _calibrate(data)
    frames = data["input_short_frames"][:6].reshape(2, 3, 34, 40)

    occupancy = OccupancyProcessor(result.calibration).process(
        camera_cycle_snapshot(frames),
    )
    occupied = np.asarray(occupancy.occupied, dtype=bool)
    valid = np.asarray(
        occupancy.artifacts["occupied"].expanded_validity(),
        dtype=bool,
    )
    counted = np.where(valid, occupied, np.nan).astype(float)

    np.testing.assert_allclose(
        occupancy.rate, np.nanmean(counted, axis=-1), rtol=1e-12, atol=2e-12
    )
    np.testing.assert_array_equal(occupancy.frame_judged, frames)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(1.0 - occupancy.rate, np.nanmean(counted, axis=-1))
