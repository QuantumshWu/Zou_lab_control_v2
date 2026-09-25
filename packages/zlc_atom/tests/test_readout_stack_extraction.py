"""A stack of frames reads exactly as the frames read one at a time."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.devices.camera.contract import CameraFrameRecord
from zlc_atom.nodes.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
    extract_box_signals,
    extract_psf_signals,
)
from zlc_atom.nodes.calibration.calibration import extract_psf_window
from zlc_atom.nodes.camera_measurement.measurement import frames_snapshot
from zlc_atom.nodes.occupancy.processor import OccupancyProcessor


def _reference_box(frame: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
    """The site-by-site reading the extractor used to make, kept as the reference."""

    output = np.full(len(centers), np.nan)
    for index, (x, y) in enumerate(centers):
        x, y = int(round(float(x))), int(round(float(y)))
        values = frame[y - radius : y + radius + 1, x - radius : x + radius + 1]
        finite = values if values.dtype.kind in "biu" else values[np.isfinite(values)]
        if finite.size:
            output[index] = float(np.sum(finite, dtype=np.float64))
    return output


def _centers(count: int, shape: tuple[int, int], margin: int) -> np.ndarray:
    rng = np.random.default_rng(count)
    return np.stack([
        rng.uniform(margin, shape[1] - 1 - margin, size=count),
        rng.uniform(margin, shape[0] - 1 - margin, size=count),
    ], axis=1)


@pytest.mark.parametrize("dtype", ["uint16", "float64"])
@pytest.mark.parametrize("radius", [0, 1, 3])
def test_box_windows_gathered_at_once_read_as_the_reference_did(dtype, radius) -> None:
    rng = np.random.default_rng(radius)
    shape = (60, 80)
    frames = rng.poisson(30.0, size=(5, *shape)).astype(dtype)
    if dtype == "float64":
        frames[1, 10:12, 10:14] = np.nan
        frames[2, 30, 40] = np.inf
        frames[3] = np.nan
    centers = _centers(37, shape, radius)
    # Sites at the very edge of what fits, including half-way centres that
    # round to even.
    centers[0] = (radius, radius)
    centers[1] = (shape[1] - 1 - radius, shape[0] - 1 - radius)
    centers[2] = (10.5, 11.5)
    stacked = extract_box_signals(frames, centers, radius=radius)
    assert stacked.shape == (5, 37)
    for index, frame in enumerate(frames):
        reference = _reference_box(frame, centers, radius)
        single = extract_box_signals(frame, centers, radius=radius)
        np.testing.assert_array_equal(single, reference)
        np.testing.assert_array_equal(stacked[index], reference)
    with pytest.raises(ValueError, match="lies outside image"):
        extract_box_signals(frames[0], [[radius - 1, 20.0]], radius=radius)
    assert extract_box_signals(frames[0], np.zeros((0, 2)), radius=radius).shape == (0,)


@pytest.mark.parametrize("background", ["annulus", "none"])
def test_psf_windows_gathered_at_once_read_as_the_frames_do(background) -> None:
    rng = np.random.default_rng(7)
    shape = (64, 96)
    radius, padding = 2, 3
    frames = rng.normal(100.0, 5.0, size=(4, *shape))
    frames[2, 20:23, 30:33] = np.nan
    count = 31
    centers = _centers(count, shape, radius + padding)
    size = 2 * radius + 1
    kernels = rng.uniform(0.0, 1.0, size=(count, size, size))
    rounded = np.rint(centers).astype(int)
    boxes = np.stack(
        [rounded[:, 0] - radius, rounded[:, 1] - radius, np.full(count, size), np.full(count, size)],
        axis=1,
    )
    stacked = extract_psf_signals(
        frames, centers, kernels=kernels, boxes_xywh=boxes, background=background,
        radius=radius, padding=padding,
    )
    assert stacked.shape == (4, count)
    for index, frame in enumerate(frames):
        single = extract_psf_signals(
            frame, centers, kernels=kernels, boxes_xywh=boxes, background=background,
            radius=radius, padding=padding,
        )
        # The scalar path, window by window: the reference the gather must match.
        reference = np.full(count, np.nan)
        for site, (box, kernel) in enumerate(zip(boxes, kernels)):
            x, y, width, height = (int(value) for value in box)
            cut = frame[y : y + height, x : x + width]
            if np.isfinite(cut).all():
                reference[site] = extract_psf_window(
                    frame, (x, y, width, height), kernel, background=background, padding=padding,
                )
        np.testing.assert_allclose(single, reference, rtol=0, atol=1e-9, equal_nan=True)
        np.testing.assert_array_equal(stacked[index], single)


def test_a_calibration_reads_a_stack_as_it_reads_each_frame_and_occupancy_uses_it() -> None:
    sites = tuple(f"s{i}" for i in range(9))
    centers = np.asarray([[2.0 + 3 * (i % 3), 2.0 + 3 * (i // 3)] for i in range(9)])
    usable = [True] * 9
    usable[4] = False
    calibration = TrapCalibration(
        SiteMap(sites, centers, [True] * 9, [1.0] * 9),
        (ReadoutModel(sites, [50.0] * 9, [0.0] * 9, [100.0] * 9, usable, [1.0] * 9),),
        ReadoutModelKind.BOX,
        FrameContract((12, 12)),
    )
    rng = np.random.default_rng(3)
    frames = rng.poisson(20.0, size=(6, 12, 12)).astype(np.uint16)
    stacked = calibration.signals_of_frames(frames)
    assert stacked.shape == (6, 9)
    for index, frame in enumerate(frames):
        np.testing.assert_array_equal(stacked[index], calibration.signals(frame))
    assert np.isnan(stacked[:, 4]).all(), "an unusable site reads NaN in every frame"
    with pytest.raises(ValueError, match="stack shaped"):
        calibration.signals_of_frames(frames[0])

    strict = TrapCalibration(
        SiteMap(sites, centers, [True] * 9, [1.0] * 9),
        (ReadoutModel(sites, [1e6] * 9, [0.0] * 9, [2e6] * 9, [True] * 9, [1.0] * 9),),
        ReadoutModelKind.BOX,
        FrameContract((12, 12)),
    )
    cycle = frames_snapshot(
        (tuple(CameraFrameRecord(frame, k) for k, frame in enumerate(frames)),),
        producer="camera", generation="g", revision=1, value_unit="count",
    )
    result = OccupancyProcessor(calibration, calibration_by_frame={3: strict}).process(cycle)
    counts = np.asarray(result.counts).reshape(6, 9)
    for index, frame in enumerate(frames):
        expected = (strict if index == 2 else calibration).signals(frame)
        np.testing.assert_array_equal(np.where(np.isfinite(counts[index]), counts[index], np.nan),
                                      np.where(np.isfinite(expected), expected.astype("<f4"), np.nan))
