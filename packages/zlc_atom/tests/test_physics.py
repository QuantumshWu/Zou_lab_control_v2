from __future__ import annotations

from dataclasses import replace
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
                dark_sample_count=np.asarray([8]),
                dark_sample_variance=np.asarray([4.0]),
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
        "format",
        "site_map",
        "models",
        "default_model_kind",
        "frame_contract",
        "report",
    }
    assert payload["format"] == TrapCalibration.FORMAT
    loaded = TrapCalibration.load(target)
    assert loaded.frame_contract.binning_yx == (2, 2)
    assert loaded.select_model().threshold_method == "empirical"
    np.testing.assert_allclose(loaded.select_model().dark_mean, [1.0])
    np.testing.assert_allclose(loaded.select_model().bright_mean, [9.0])
    np.testing.assert_array_equal(loaded.select_model().dark_sample_count, [8])
    np.testing.assert_allclose(loaded.select_model().dark_sample_variance, [4.0])
    np.testing.assert_allclose(loaded.signals(image), [5.0])

    unformatted = calibration.to_dict()
    unformatted.pop("format")
    with pytest.raises(ValueError, match="TrapCalibration fields"):
        TrapCalibration.from_dict(unformatted)

    missing_statistics = calibration.to_dict()
    missing_statistics["models"][0].pop("dark_statistics")
    with pytest.raises(ValueError, match="missing ReadoutModel fields"):
        TrapCalibration.from_dict(missing_statistics)
    with pytest.raises(ValueError, match="invalid counts"):
        replace(
            calibration.select_model(),
            dark_sample_count=np.asarray([2**63], dtype=np.uint64),
        )
    overflow_payload = calibration.to_dict()
    overflow_payload["models"][0]["dark_statistics"]["sample_count"] = [2**63]
    with pytest.raises(ValueError, match="invalid counts"):
        TrapCalibration.from_dict(overflow_payload)

    malformed = calibration.to_dict()
    malformed["format"] = "other"
    with pytest.raises(ValueError, match="unsupported Calibration format"):
        TrapCalibration.from_dict(malformed)

    unknown = calibration.to_dict()
    unknown["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown TrapCalibration fields"):
        TrapCalibration.from_dict(unknown)


def test_calibration_document_is_actual_json_data_not_python_container_aliases() -> None:
    payload = replace(
        _calibration(),
        report={"nested": {"values": (np.float64(1.5), np.int64(2))}},
    ).to_dict()
    assert payload["frame_contract"]["image_shape"] == [3, 3]
    assert payload["report"] == {"nested": {"values": [1.5, 2]}}
    json.dumps(payload, allow_nan=False)


def test_calibration_owns_nested_topology_and_report_truth(tmp_path, monkeypatch) -> None:
    topology = {"grid": {"rows": [0, 1]}}
    report = {
        "run_record": {"request": {"photoelectrons": False}},
        "diagnostics": {"scores": np.asarray([1.0, 2.0])},
    }
    original = _calibration()
    with pytest.raises(TypeError, match="report must be a mapping"):
        replace(original, report=[])
    calibration = replace(
        original,
        site_map=replace(original.site_map, topology=topology),
        report=report,
    )

    topology["grid"]["rows"][0] = 99
    report["run_record"]["request"]["photoelectrons"] = True
    report["diagnostics"]["scores"][0] = 99.0
    assert calibration.site_map.topology["grid"]["rows"] == (0, 1)
    assert calibration.report["run_record"]["request"]["photoelectrons"] is False
    assert calibration.report["diagnostics"]["scores"] == (1.0, 2.0)

    with pytest.raises(TypeError):
        calibration.report["run_record"]["request"]["photoelectrons"] = True
    with pytest.raises(TypeError):
        calibration.site_map.topology["grid"]["rows"][0] = 99

    document = calibration.to_dict()
    document["report"]["run_record"]["request"]["photoelectrons"] = True
    document["site_map"]["topology"]["grid"]["rows"][0] = 99
    assert calibration.report["run_record"]["request"]["photoelectrons"] is False
    assert calibration.site_map.topology["grid"]["rows"] == (0, 1)

    monkeypatch.chdir(tmp_path)
    written = calibration.save("relative-calibration.json")
    assert written == (tmp_path / "relative-calibration.json").resolve()
    assert written.is_file()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(extra=True), "unknown TrapCalibration fields"),
        (
            lambda payload: payload["site_map"].update(extra=True),
            "unknown SiteMap fields",
        ),
        (
            lambda payload: payload["models"][0].update(extra=True),
            "unknown ReadoutModel fields",
        ),
        (
            lambda payload: payload["models"][0]["integration"].update(extra=True),
            "unknown ReadoutModel.integration fields",
        ),
        (
            lambda payload: payload["frame_contract"].update(extra=True),
            "unknown FrameContract fields",
        ),
        (
            lambda payload: payload["site_map"].update(valid_sites=[1]),
            "valid_sites.*boolean",
        ),
        (
            lambda payload: payload["models"][0].update(thresholds=["5.0"]),
            "thresholds.*number",
        ),
        (
            lambda payload: payload["models"][0]["integration"].update(half_width=1.0),
            "half_width.*integer",
        ),
    ),
)
def test_calibration_tree_rejects_unknown_fields_and_type_coercion(
    mutation,
    message: str,
) -> None:
    payload = _calibration().to_dict()
    mutation(payload)
    with pytest.raises((TypeError, ValueError), match=message):
        TrapCalibration.from_dict(payload)


@pytest.mark.parametrize(
    ("document", "message"),
    (
        (
            '{"site_map": {}, "site_map": {}, "models": [], '
            '"default_model_kind": "box", "frame_contract": {}, "report": {}}',
            "duplicate key.*site_map",
        ),
        (
            '{"site_map": {}, "models": [], "default_model_kind": "box", '
            '"frame_contract": {}, "report": {"bad": NaN}}',
            "non-finite JSON constant.*NaN",
        ),
    ),
)
def test_calibration_load_rejects_duplicate_keys_and_nonfinite_json(
    tmp_path: Path,
    document: str,
    message: str,
) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        TrapCalibration.load(path)


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
