from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zlc_atom.nodes.occupancy import OccupancyProcessor
from zlc_atom.nodes.calibration.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
    calibrate,
    classify_threshold,
)


ORACLE = Path(__file__).parent / "fixtures" / "main_readout_oracle.npz"


def _load() -> dict[str, np.ndarray]:
    with np.load(ORACLE, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def test_threshold_plus_17_3_is_rejected_by_frozen_predictions() -> None:
    oracle = _load()
    result = calibrate(
        oracle["input_reference_frames"],
        oracle["input_short_frames"],
        frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
        box_half_width=1,
        box_reducer="mean",
    )
    box_report = result.report["models"]["box"]
    box_model = result.calibration.select_model(ReadoutModelKind.BOX)
    mutated = classify_threshold(
        box_report["short_signals"],
        box_model.thresholds + 17.3,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(mutated, oracle["pred_box"])


def test_classify_polarity_flip_is_rejected_by_frozen_predictions() -> None:
    oracle = _load()
    mutated = classify_threshold(
        oracle["short_signals_box"],
        oracle["thresholds_box"],
        bright_above=False,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(mutated, oracle["pred_box"])


def test_occupancy_rate_inverse_is_rejected_by_frozen_rate() -> None:
    oracle = _load()
    site_ids = tuple(f"site_{index:04d}" for index in range(len(oracle["centers_row_major"])))
    calibration = TrapCalibration(
        SiteMap(
            site_ids,
            oracle["centers_row_major"],
            np.ones(len(site_ids), dtype=bool),
            np.ones(len(site_ids)),
        ),
        (
            ReadoutModel(
                site_ids,
                oracle["thresholds_box"],
                np.ones(len(site_ids), dtype=bool),
                np.ones(len(site_ids)),
                kind=ReadoutModelKind.BOX,
                integration_half_width=1,
                reducer="mean",
            ),
        ),
        ReadoutModelKind.BOX,
        FrameContract((34, 40), exposure_seconds=0.005),
    )
    probe_frames = oracle["input_short_frames"][oracle["runtime_probe_indices"]].reshape(2, 3, 34, 40)
    # Raw arrays from an oracle file, not a run, so the stamps are stated.
    occupancy = OccupancyProcessor(calibration).process(
        probe_frames, generation="mutation-guard", revision=1
    )
    np.testing.assert_allclose(occupancy.counts, oracle["runtime_signals_box"], rtol=1e-12, atol=2e-12)
    np.testing.assert_array_equal(occupancy.occupied, oracle["runtime_occupied_box"])
    np.testing.assert_array_equal(occupancy.valid, np.ones_like(occupancy.occupied))
    np.testing.assert_array_equal(occupancy.frame_judged, probe_frames)
    np.testing.assert_allclose(occupancy.rate[..., None], oracle["runtime_rate_box"], rtol=1e-12, atol=2e-12)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(1.0 - occupancy.rate[..., None], oracle["runtime_rate_box"])
