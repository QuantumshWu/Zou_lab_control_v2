from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zlc_atom.nodes.calibration import (
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
    normal_cdf,
)
from zlc_atom.nodes.calibration.calibration import FrameContract, extract_box_signals


def _calibration(
    *,
    kind: ReadoutModelKind = ReadoutModelKind.BOX,
    psf_weights: np.ndarray | None = None,
    psf_boxes: np.ndarray | None = None,
    frame_contract: FrameContract | None = None,
) -> TrapCalibration:
    site_ids = ("site_0000",)
    return TrapCalibration(
        SiteMap(site_ids, np.asarray([[1.0, 1.0]]), [True], [1.0]),
        (
            ReadoutModel(
                site_ids,
                [5.0],
                [1.0],
                [9.0],
                [True],
                [1.0],
                kind=kind,
                integration_half_width=1,
                reducer="mean" if kind is ReadoutModelKind.BOX else None,
                psf_weights=psf_weights,
                psf_boxes=psf_boxes,
                background=None if kind is ReadoutModelKind.BOX else "none",
                psf_padding=None if kind is ReadoutModelKind.BOX else 3,
            ),
        ),
        kind,
        frame_contract or FrameContract((3, 3)),
    )


def test_hand_examples_are_explicitly_supplemental() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "hand_examples.json").read_text(encoding="utf-8"))
    observed = normal_cdf(fixture["normal_cdf"]["x"], fixture["normal_cdf"]["mu"], fixture["normal_cdf"]["sigma"])
    np.testing.assert_allclose(observed, fixture["normal_cdf"]["expected"], rtol=0.0, atol=2e-15)


def test_trap_calibration_single_dispatch_supports_box(tmp_path: Path) -> None:
    image = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    calibration = _calibration(
        frame_contract=FrameContract(
            (3, 3),
            sensor_shape=(6, 6),
            roi_xywh=(0, 0, 6, 6),
            binning_yx=(2, 2),
        )
    )
    np.testing.assert_allclose(calibration.signals(image), [5.0])
    np.testing.assert_array_equal(calibration.detect(image), [False])
    target = calibration.save(tmp_path / "calibration.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload) == {
        "site_map",
        "models",
        "default_model_kind",
        "frame_contract",
        "report",
    }
    loaded = TrapCalibration.load(target)
    assert loaded.frame_contract.binning_yx == (2, 2)
    assert loaded.select_model().threshold_method == "empirical"
    np.testing.assert_allclose(loaded.select_model().dark_mean, [1.0])
    np.testing.assert_allclose(loaded.select_model().bright_mean, [9.0])
    np.testing.assert_allclose(loaded.signals(image), [5.0])


def test_usable_readout_requires_a_finite_positive_response() -> None:
    with pytest.raises(ValueError, match="finite bright_mean > dark_mean"):
        ReadoutModel(("site_0000",), [5.0], [1.0], [np.nan], [True], [1.0])
    with pytest.raises(ValueError, match="finite bright_mean > dark_mean"):
        ReadoutModel(("site_0000",), [5.0], [2.0], [2.0], [True], [1.0])

    model = ReadoutModel(
        ("site_0000",), [np.nan], [np.nan], [np.nan], [False], [np.nan]
    )
    assert model.to_dict()["dark_mean"] == [None]
    assert model.to_dict()["bright_mean"] == [None]


def test_psf_dispatch_is_explicit_and_not_a_name_substring() -> None:
    kernel = np.ones((3, 3), dtype=float) / 9.0
    calibration = _calibration(
        kind=ReadoutModelKind.PER_SITE_PSF,
        psf_weights=kernel[None],
        psf_boxes=np.asarray([[0, 0, 3, 3]]),
    )
    np.testing.assert_allclose(calibration.signals(np.arange(9, dtype=float).reshape(3, 3) + 1.0), [5.0])


def test_box_reducer_oracle_is_shared_by_public_function_and_calibration() -> None:
    image = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    centers = np.asarray([[1.0, 1.0]])
    np.testing.assert_allclose(extract_box_signals(image, centers, reducer="mean"), [5.0])
    np.testing.assert_allclose(extract_box_signals(image, centers, reducer="sum"), [45.0])
