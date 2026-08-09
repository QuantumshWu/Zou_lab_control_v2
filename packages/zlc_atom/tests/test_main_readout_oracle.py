from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from zlc_atom.nodes.calibration.bimodal import fit_bimodal, normal_cdf, per_site_fidelity
from zlc_atom.nodes.calibration.calibration import (
    FrameContract,
    ReadoutModelKind,
    calibrate,
    classify_threshold,
    detect_sites,
    extract_box_signals,
    extract_psf_signals,
    extract_psf_window,
)


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES / "main_readout_oracle.json"
ORACLE = FIXTURES / "main_readout_oracle.npz"

MODEL_NAMES = ("box", "psf", "uniform_psf")
MODEL_FIELDS = (
    "quick_thresholds",
    "short_signals",
    "thresholds",
    "pred",
    "site_fidelity",
    "site_model_fidelity",
    "site_fidelity_dark",
    "site_fidelity_bright",
    "site_mu_dark",
    "site_sigma_dark",
    "site_mu_bright",
    "site_sigma_bright",
    "site_bright_above",
    "site_n_test",
    "site_n_train_dark",
    "site_n_train_bright",
    "aggregate_fidelity",
    "global_threshold",
    "global_fidelity",
    "split_train",
    "split_test",
    "ablation_drop_k",
    "ablation_excluded",
    "ablation_fidelity",
    "ablation_errors",
    "ablation_n_valid",
    "runtime_signals",
    "runtime_occupied",
    "runtime_rate",
)
COMMON_FIELDS = {
    "input_reference_frames",
    "input_short_frames",
    "input_latent_occupancy",
    "reference_average",
    "centers_row_major",
    "centers_serpentine",
    "centers_column_major",
    "centers_column_serpentine",
    "box_boxes_xywh",
    "psf_boxes_xywh",
    "psf_kernels",
    "psf_fit_centers_xy",
    "psf_fit_sigma_xy",
    "psf_fit_ok",
    "uniform_boxes_xywh",
    "uniform_kernel",
    "reference_box_signals",
    "labels_occupied",
    "labels_dark",
    "labels_valid",
    "reference_fit_threshold",
    "reference_fit_fidelity",
    "reference_fit_dark_mean",
    "reference_fit_dark_sigma",
    "reference_fit_bright_mean",
    "reference_fit_bright_sigma",
    "reference_fit_bright_above",
    "reference_fit_ok",
    "runtime_probe_indices",
}
EXPECTED_FIELDS = COMMON_FIELDS | {
    f"{prefix}_{name}" for name in MODEL_NAMES for prefix in MODEL_FIELDS
}


def _load_oracle() -> dict[str, np.ndarray]:
    with np.load(ORACLE, allow_pickle=False) as archive:
        assert set(archive.files) == EXPECTED_FIELDS
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _assert_close(actual: object, expected: object) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-12, equal_nan=True)


def _centers_from_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    return boxes[:, :2] + (boxes[:, 2:] - 1.0) / 2.0


def test_frozen_main_oracle_manifest_and_bytes_are_authoritative() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["format"] == "main-readout-oracle"
    assert manifest["authority_commit"] == "6c337d49c7086fa0ff21f879cd159bdf0e753f51"
    assert manifest["input"]["reference_shape"] == [60, 2, 34, 40]
    assert manifest["input"]["short_shape"] == [60, 34, 40]
    assert manifest["settings"]["psf_background_padding"] == 3
    assert manifest["settings"]["histogram_bins"] == 120
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == "f2e6087446cb089a551ecfb57520ea46bbfb496981e81df26fac361bca1f1194"
    assert hashlib.sha256(ORACLE.read_bytes()).hexdigest() == "ec0194edbe0ea55cad64c70d780939c3cd5f4a3b419e20997e337359965386aa"
    oracle = _load_oracle()
    assert oracle["input_reference_frames"].shape == (60, 2, 34, 40)
    assert oracle["input_short_frames"].shape == (60, 34, 40)


def test_normal_cdf_supplemental_hand_example() -> None:
    fixture = json.loads((FIXTURES / "hand_examples.json").read_text(encoding="utf-8"))
    _assert_close(
        normal_cdf(fixture["normal_cdf"]["x"], fixture["normal_cdf"]["mu"], fixture["normal_cdf"]["sigma"]),
        fixture["normal_cdf"]["expected"],
    )


def test_main_oracle_box_reducers() -> None:
    oracle = _load_oracle()
    centers = _centers_from_boxes(oracle["box_boxes_xywh"])
    image = oracle["input_reference_frames"][0, 0]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for reducer in ("mean", "sum", "median", "max"):
        observed = extract_box_signals(image, centers, radius=1, reducer=reducer)
        _assert_close(observed, manifest["box_reducer_oracle"][reducer])


def test_main_oracle_psf_window_and_public_padding() -> None:
    oracle = _load_oracle()
    boxes = oracle["psf_boxes_xywh"]
    centers = _centers_from_boxes(boxes)
    expected = oracle["short_signals_psf"]
    direct = np.asarray(
        [
            [
                extract_psf_window(frame, tuple(box), kernel, background="annulus", padding=3)
                for box, kernel in zip(boxes, oracle["psf_kernels"], strict=True)
            ]
            for frame in oracle["input_short_frames"]
        ]
    )
    _assert_close(direct, expected)
    observed = np.asarray(
        [
            extract_psf_signals(
                frame,
                centers,
                kernels=oracle["psf_kernels"],
                boxes_xywh=boxes,
                background="annulus",
                radius=3,
                padding=3,
            )
            for frame in oracle["input_short_frames"]
        ]
    )
    _assert_close(observed, expected)


def test_main_oracle_data_driven_psf_fit_and_uniform_sibling() -> None:
    oracle = _load_oracle()
    kwargs = {
        "frame_contract": FrameContract((34, 40), exposure_seconds=0.005),
        "box_half_width": 1,
        "psf_half_width": 3,
        "psf_padding": 3,
    }
    result = calibrate(
        oracle["input_reference_frames"], oracle["input_short_frames"], **kwargs
    )
    assert tuple(model.kind.value for model in result.calibration.models) == MODEL_NAMES
    per_site = result.calibration.select_model(ReadoutModelKind.PER_SITE_PSF)
    uniform = result.calibration.select_model(ReadoutModelKind.UNIFORM_PSF)
    _assert_close(per_site.psf_weights, oracle["psf_kernels"])
    np.testing.assert_array_equal(per_site.psf_boxes, oracle["psf_boxes_xywh"])
    per_site_report = result.report["models"]["psf"]
    _assert_close(per_site_report["psf_fit_centers_xy"], oracle["psf_fit_centers_xy"])
    _assert_close(per_site_report["psf_fit_sigma_xy"], oracle["psf_fit_sigma_xy"])
    np.testing.assert_array_equal(per_site_report["psf_fit_ok"], oracle["psf_fit_ok"])
    _assert_close(uniform.psf_weights[0], oracle["uniform_kernel"])
    _assert_close(
        result.report["models"]["uniform_psf"]["uniform_kernel"],
        oracle["uniform_kernel"],
    )


def test_main_oracle_bimodal_components_and_threshold() -> None:
    oracle = _load_oracle()
    fits = [fit_bimodal(oracle["reference_box_signals"][:, :, site].reshape(-1)) for site in range(6)]
    for attribute, field in (
        ("threshold", "reference_fit_threshold"),
        ("fidelity", "reference_fit_fidelity"),
        ("dark_mean", "reference_fit_dark_mean"),
        ("dark_sigma", "reference_fit_dark_sigma"),
        ("bright_mean", "reference_fit_bright_mean"),
        ("bright_sigma", "reference_fit_bright_sigma"),
    ):
        _assert_close([getattr(fit, attribute) for fit in fits], oracle[field])
    np.testing.assert_array_equal([fit.bright_above for fit in fits], oracle["reference_fit_bright_above"])
    np.testing.assert_array_equal([fit.ok for fit in fits], oracle["reference_fit_ok"])


def test_main_oracle_detector_discovers_sites_without_grid_input() -> None:
    oracle = _load_oracle()
    observed = detect_sites(oracle["reference_average"])
    assert observed.n_sites == 6
    assert observed.site_ids == tuple(f"site_{index:04d}" for index in range(6))
    _assert_close(observed.centers_xy, oracle["centers_row_major"])


def test_main_oracle_classification() -> None:
    oracle = _load_oracle()
    observed = classify_threshold(oracle["short_signals_box"], oracle["thresholds_box"])
    np.testing.assert_array_equal(observed, oracle["pred_box"])


def test_main_oracle_grouped_calibration_end_to_end() -> None:
    oracle = _load_oracle()
    result = calibrate(
        oracle["input_reference_frames"],
        oracle["input_short_frames"],
        frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
        box_half_width=1,
        box_reducer="mean",
    )
    report = result.report
    box_model = result.calibration.select_model(ReadoutModelKind.BOX)
    box_report = report["models"]["box"]
    assert box_model.threshold_method == "empirical"
    assert box_report["threshold_method"] == "empirical"
    _assert_close(report["reference_average"], oracle["reference_average"])
    _assert_close(report["reference_label_signals"], oracle["reference_box_signals"])
    np.testing.assert_array_equal(report["labels_occupied"], oracle["labels_occupied"])
    np.testing.assert_array_equal(report["labels_dark"], oracle["labels_dark"])
    np.testing.assert_array_equal(report["labels_valid"], oracle["labels_valid"])
    _assert_close(box_report["short_signals"], oracle["short_signals_box"])
    _assert_close(box_model.thresholds, oracle["thresholds_box"])
    np.testing.assert_array_equal(box_report["predictions"], oracle["pred_box"])
    assert int(np.count_nonzero(box_report["predictions"] != oracle["input_latent_occupancy"])) == 29
    _assert_close(box_report["site_fidelity"], oracle["site_fidelity_box"])
    _assert_close(
        per_site_fidelity(
            box_report["short_signals"],
            report["labels_occupied"],
            box_model.thresholds,
            test_mask=report["split_test"],
            valid_mask=report["labels_valid"],
        ).balanced,
        oracle["site_fidelity_box"],
    )
    _assert_close(box_report["site_fidelity_dark"], oracle["site_fidelity_dark_box"])
    _assert_close(box_report["site_fidelity_bright"], oracle["site_fidelity_bright_box"])
    _assert_close(box_report["site_model_fidelity"], oracle["site_model_fidelity_box"])
    np.testing.assert_array_equal(report["split_train"], oracle["split_train_box"])
    np.testing.assert_array_equal(report["split_test"], oracle["split_test_box"])
    np.testing.assert_array_equal(box_report["site_n_test"], oracle["site_n_test_box"])
    np.testing.assert_array_equal(box_report["site_n_train_dark"], oracle["site_n_train_dark_box"])
    np.testing.assert_array_equal(box_report["site_n_train_bright"], oracle["site_n_train_bright_box"])

    gaussian = calibrate(
        oracle["input_reference_frames"],
        oracle["input_short_frames"],
        frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
        threshold_method="gaussian",
        box_half_width=1,
        box_reducer="mean",
    )
    gaussian_model = gaussian.calibration.select_model(ReadoutModelKind.BOX)
    gaussian_report = gaussian.report["models"]["box"]
    assert gaussian_model.threshold_method == "gaussian"
    assert gaussian_report["threshold_method"] == "gaussian"
    np.testing.assert_allclose(
        gaussian_model.thresholds,
        gaussian_report["gaussian_thresholds"],
        equal_nan=True,
    )


def test_a_site_whose_fit_says_bright_is_below_is_refused() -> None:
    """An atom scatters photons: bright is above dark.

    The direction used to be carried per site, so this loop classified one way
    while per_site_fidelity and every later TrapCalibration.detect() classified
    the other -- three answers to "is this site bright?", on the number a
    readout is judged by.
    """

    from zlc_atom.nodes.calibration.bimodal import optimal_gaussian_threshold

    _threshold, bright_above = optimal_gaussian_threshold(100.0, 5.0, 40.0, 5.0)
    assert bright_above is False, "the fit can report an inverted pair"
