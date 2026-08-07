from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zlc_atom.nodes.calibration import TrapCalibration, normal_cdf
from zlc_atom.nodes.calibration.calibration import FrameContract, extract_box_signals


def test_hand_examples_are_explicitly_supplemental() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "hand_examples.json").read_text(encoding="utf-8"))
    observed = normal_cdf(fixture["normal_cdf"]["x"], fixture["normal_cdf"]["mu"], fixture["normal_cdf"]["sigma"])
    np.testing.assert_allclose(observed, fixture["normal_cdf"]["expected"], rtol=0.0, atol=2e-15)


def test_trap_calibration_single_dispatch_supports_box() -> None:
    image = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    calibration = TrapCalibration(np.asarray([[1.0, 1.0]]), [5.0], FrameContract((3, 3)), roi_radius=1, reducer="mean")
    np.testing.assert_allclose(calibration.signals(image), [5.0])
    np.testing.assert_array_equal(calibration.detect(image), [False])


def test_psf_dispatch_is_explicit_and_not_a_name_substring() -> None:
    kernel = np.ones((3, 3), dtype=float) / 9.0
    calibration = TrapCalibration(np.asarray([[1.0, 1.0]]), [5.0], FrameContract((3, 3)), method="psf", psf_weights=kernel[None], psf_boxes=np.asarray([[0, 0, 3, 3]]))
    np.testing.assert_allclose(calibration.signals(np.arange(9, dtype=float).reshape(3, 3) + 1.0), [5.0])


def test_box_reducer_oracle_is_shared_by_public_function_and_calibration() -> None:
    image = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    centers = np.asarray([[1.0, 1.0]])
    np.testing.assert_allclose(extract_box_signals(image, centers, reducer="mean"), [5.0])
    np.testing.assert_allclose(extract_box_signals(image, centers, reducer="sum"), [45.0])
