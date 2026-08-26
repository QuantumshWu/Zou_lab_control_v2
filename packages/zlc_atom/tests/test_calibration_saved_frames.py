"""Frames saved as they arrive, and calibrated again without the bench."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from zlc_atom.install import create_installation
from zlc_atom.devices.camera.contract import CameraFrameRecord, CameraWorkingPoint
from zlc_atom.nodes.calibration.outputs import (
    CAPTURE_PREVIEW_DECLARATION,
    SITE_REVIEW_DECLARATION,
)
from zlc_atom.nodes.calibration.logic_node import LOGIC_NODE as CALIBRATION_LOGIC_NODE
from zlc_atom.nodes.calibration.task import (
    FRAMES_FROM_FOLDER,
    CalibrationRequest,
    CalibrationRunResult,
    SampleWriter,
    CalibrationTask,
    read_saved_samples,
)
from zlc_data import StreamGenerationId
from zlc_data.figure_archive import FIGURE_SCHEMA, read_archive, read_dataset
from zlc_runtime import TaskRun
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane
from zlc_plot import ImageFrame, PointStatus, read_figure_plot

from tests.fakes import FakePlane
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE
from test_installation_and_nodes import _calibration_request


def _task(request: CalibrationRequest) -> CalibrationTask:
    installation = create_installation("virtual")
    return CalibrationTask(
        camera=installation.device("camera"),
        sequencer=installation.device("sequencer"),
        request=request,
        pulse_sequence=IMAGING_PULSE_RESOURCE.value,
        pulse_path=IMAGING_PULSE_RESOURCE.path,
        signal_plane=FakePlane(),
    )


def _sample_writer(folder: Path, *, run: str, generation: str) -> SampleWriter:
    point = CameraWorkingPoint(
        "EXTERNAL_TRIGGERED",
        (2, 2),
        (2, 2),
        (0, 0),
        (2, 2),
        (1, 1),
        np.dtype("<u2"),
        "count",
        0.005,
        0.006,
        0.0,
        1.0,
        "default",
    )
    artifact_run = TaskRun(
        folder.parent,
        task_name="calibration",
        instance_id="calibration",
        input_summary={"run": run},
    )
    artifact_run.mark_running()
    return SampleWriter(
        folder,
        working_point=point,
        run_record={"run": run},
        generation=StreamGenerationId(generation),
        photoelectrons=False,
        artifact_context=artifact_run,
    )


def _sample_cycle(offset: int = 0) -> tuple[CameraFrameRecord, ...]:
    return tuple(
        CameraFrameRecord(np.full((2, 2), offset + index, dtype="<u2"), index)
        for index in range(3)
    )


def test_saved_sample_replay_rejects_gaps_and_mixed_runs(tmp_path: Path) -> None:
    folder = tmp_path / "frames"
    writer = _sample_writer(folder, run="one", generation="run-one")
    writer.write(0, _sample_cycle())
    writer.write(1, _sample_cycle(10))

    (folder / "sample_0000.npz").unlink()
    with pytest.raises(ValueError, match="contiguous from zero"):
        read_saved_samples(folder)

    writer.write(0, _sample_cycle())
    other = _sample_writer(folder, run="two", generation="run-two")
    other.write(1, _sample_cycle(20))
    with pytest.raises(ValueError, match="different saved capture"):
        read_saved_samples(folder)


def test_saved_samples_are_written_as_they_arrive_and_calibrate_again(
    tmp_path: Path,
) -> None:
    """The whole point: retune detection without holding the bench again.

    Every sample is written in the acquisition loop, not after it, so a run
    that is cancelled or that fails its analysis still leaves the frames it
    paid atoms for.  Each file uses the shared figure archive -- the picture
    and the typed numbers behind it -- so the same viewer opens them, and the
    run record inside is where the crop and the exposure are.
    """

    request = replace(_calibration_request(repeats=12), save_frames=True)
    first = _task(request).run(tmp_path)

    folder = first.artifact_path.parents[1] / "figures"
    archives = sorted(folder.glob("sample_*.npz"))
    pictures = sorted(folder.glob("sample_*.png"))
    assert len(archives) == 12
    assert len(pictures) == 12

    info, arrays = read_archive(archives[0])
    snapshot = read_dataset(info, arrays, "data")
    assert np.asarray(snapshot.block.values).shape[:2] == (1, 3)
    assert set(info["sections"]["plot"]) == {"data"}
    reopened, recipe = read_figure_plot(info, arrays, "data")
    assert reopened.block.schema == snapshot.block.schema
    assert recipe["spec"].kind.value == "facet_grid"
    assert info["sections"]["source"]["run_record"], (
        "the crop and exposure travel with the frames"
    )
    assert "panel" not in info["sections"], "Calibration must not copy Workbench panel state"

    # Read straight back: the same three frames per sample, in order.
    cycles, run_record = read_saved_samples(folder)
    assert len(cycles) == 12
    assert run_record["request"]["camera_key"] == "camera"
    np.testing.assert_array_equal(
        np.asarray(cycles[0][1].image),
        np.asarray(first.capture.cycles[0][1].image),
    )

    # And calibrated again from that folder alone.  Same frames, same
    # settings, same answer -- to the pixel.
    def replay(**changes: object):
        return _task(
            replace(
                request,
                save_frames=False,
                frame_source=FRAMES_FROM_FOLDER,
                saved_frames_path=str(folder),
                **changes,
            ),
        ).run(tmp_path)

    replayed = replay()
    assert replayed.calibration.site_map.n_sites == first.calibration.site_map.n_sites
    np.testing.assert_allclose(
        replayed.calibration.site_map.centers_xy,
        first.calibration.site_map.centers_xy,
        atol=1e-9,
    )

    # The reason to keep them: detection can be retuned on the SAME atoms,
    # which is what nobody could do without holding the bench for another run.
    relaxed = replay(detection_sigma=4.0)
    assert relaxed.calibration.site_map.n_sites >= replayed.calibration.site_map.n_sites
    # The geometry comes from the frames, not from whatever the camera is set
    # to now.
    assert (
        replayed.calibration.frame_contract.image_shape
        == first.calibration.frame_contract.image_shape
    )
    assert (
        replayed.calibration.frame_contract.exposure_seconds
        == first.calibration.frame_contract.exposure_seconds
    )


def test_a_replay_publishes_what_the_node_declares(tmp_path: Path) -> None:
    """Run the folder through the runtime, not just through the function.

    A node that declares Dataset outputs must publish them however it got its
    frames.  Published only on the camera branch, a replay ran to the end and
    then died at the runtime -- "declared Dataset outputs but did not publish
    final outputs" -- which no test that called ``run()`` could see, because
    that failure lives in the host and ``run()`` has no host.
    """

    request = replace(_calibration_request(repeats=6), save_frames=True)
    acquired = _task(request).run(tmp_path)
    folder = acquired.artifact_path.parents[1] / "figures"

    plane = SignalDataPlane()
    host = None
    try:
        task = _task(
            replace(
                request,
                save_frames=False,
                frame_source=FRAMES_FROM_FOLDER,
                saved_frames_path=str(folder),
            ),
        )
        task.signal_plane = plane
        host = NodeHost(
            task,
            plane,
            Event().set,
            instance_id="calibration-replay",
            kind="task",
            dataset_output_declarations=CALIBRATION_LOGIC_NODE.outputs,
            required_artifacts={
                "artifact_path": CALIBRATION_LOGIC_NODE.artifact_outputs[0].contract_id
            },
            task_name=CALIBRATION_LOGIC_NODE.api_name,
        )
        host.start(run_root=tmp_path, input_summary=request.to_dict())
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            host.poll()
            if host.observation.terminal:
                break
            time.sleep(0.01)
        observation = host.observation
        assert observation.phase == "done", (
            f"replay ended in {observation.phase}: {observation.error}"
        )
        key = host.signal_key(CAPTURE_PREVIEW_DECLARATION.name)
        # A successful Task proves every declared output committed at least
        # once.  This preview is deliberately Monitor/latest-only, so Runtime
        # retires it at terminal instead of preserving one arbitrary sample as
        # the calibration's scientific final dataset.
        assert plane.latest_publication(key) is None
    finally:
        if host is not None:
            host.shutdown()
        plane.close()


def test_site_review_filters_once_then_runs_the_complete_analysis(tmp_path: Path) -> None:
    acquired_request = replace(_calibration_request(repeats=8), save_frames=True)
    acquired = _task(acquired_request).run(tmp_path)
    folder = acquired.artifact_path.parents[1] / "figures"
    request = replace(
        acquired_request,
        save_frames=False,
        frame_source=FRAMES_FROM_FOLDER,
        saved_frames_path=str(folder),
        review_detected_sites=True,
    )
    plane = SignalDataPlane()
    task = _task(request)
    task.signal_plane = plane
    wake = Event()
    host = NodeHost(
        task,
        plane,
        wake.set,
        instance_id="calibration-review",
        kind="task",
        dataset_output_declarations=CALIBRATION_LOGIC_NODE.outputs,
        required_artifacts={
            "artifact_path": CALIBRATION_LOGIC_NODE.artifact_outputs[0].contract_id
        },
        task_name=CALIBRATION_LOGIC_NODE.api_name,
    )
    try:
        host.start(run_root=tmp_path, input_summary=request.to_dict())
        deadline = time.monotonic() + 60.0
        while (
            host.operator_request is None
            and not host.terminal
            and time.monotonic() < deadline
        ):
            host.poll()
            wake.wait(0.01)
            wake.clear()
        review = host.operator_request
        assert review is not None and review.kind == "point-selection", host.observation
        point_ids = tuple(review.payload["point_ids"])
        assert len(point_ids) > 1
        publication = plane.latest_publication(
            f"@logic/{host.instance_id}/review/{SITE_REVIEW_DECLARATION.name}"
        )
        assert publication is not None
        excluded = point_ids[-1:]
        host.submit_operator_input(
            review.request_id, {"excluded_point_ids": excluded}
        )
        while not host.terminal and time.monotonic() < deadline:
            host.poll()
            wake.wait(0.01)
            wake.clear()
        assert host.observation.phase == "done", host.observation
        result = host.final_result
        assert isinstance(result, CalibrationRunResult)
        assert result.calibration.site_map.n_sites == len(point_ids) - 1
        assert result.summary["site_review"] == {
            "detected_sites": len(point_ids),
            "excluded_site_ids": excluded,
            "retained_sites": len(point_ids) - 1,
        }
        run_root = result.artifact_path.parents[1]
        assert (run_root / "figures" / "site_review.npz").is_file()
        assert (run_root / "figures" / "site_review.png").is_file()
    finally:
        if host.running:
            host.cancel("test cleanup")
            host.poll()
        host.shutdown()
        plane.close()


def test_failed_calibration_analysis_saves_partial_capture_figure(
    tmp_path: Path, monkeypatch
) -> None:
    request = replace(_calibration_request(repeats=3), save_frames=True)
    acquired = _task(request).run(tmp_path)
    folder = acquired.artifact_path.parents[1] / "figures"
    task = _task(
        replace(
            request,
            save_frames=False,
            frame_source=FRAMES_FROM_FOLDER,
            saved_frames_path=str(folder),
        )
    )
    monkeypatch.setattr(
        task,
        "_analyse",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("calibration analysis failed")
        ),
    )
    plane = SignalDataPlane()
    task.signal_plane = plane
    host = NodeHost(
        task,
        plane,
        Event().set,
        instance_id="calibration-partial",
        kind="task",
        dataset_output_declarations=CALIBRATION_LOGIC_NODE.outputs,
        required_artifacts={
            "artifact_path": CALIBRATION_LOGIC_NODE.artifact_outputs[0].contract_id
        },
        task_name=CALIBRATION_LOGIC_NODE.api_name,
    )
    try:
        host.start(run_root=tmp_path, input_summary=request.to_dict())
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
            time.sleep(0.01)
        assert host.observation.phase == "failed"
        run_root = host.run_directory
        assert run_root is not None
        assert (run_root / "figures" / "partial_capture.npz").is_file()
        assert (run_root / "figures" / "partial_capture.png").is_file()
        partial_info, _partial_arrays = read_archive(
            run_root / "figures" / "partial_capture.npz"
        )
        assert set(
            partial_info["sections"]["source"]["run_record"]["actual_devices"]
        ) == {"camera", "sequencer"}
        summary = json.loads(
            (run_root / "partial-summary.json").read_text(encoding="utf-8")
        )
        assert summary["cycles_completed"] == 3
        assert summary["status"] == "failed"
    finally:
        host.shutdown()
        plane.close()


def test_nothing_is_written_unless_the_operator_asks(tmp_path: Path) -> None:
    task = _task(_calibration_request(repeats=8))
    assert list(tmp_path.iterdir()) == []
    result = task.run(tmp_path)
    run_root = result.artifact_path.parents[1]
    assert not tuple((run_root / "figures").glob("sample_*.npz"))

    run = json.loads((run_root / "run.json").read_text())
    assert run["status"]["state"] == "completed"
    assert next(
        item for item in run["artifacts"] if item["name"] == "artifact_path"
    ) == {
        "name": "artifact_path",
        "path": "final/calibration.json",
        "role": "final",
        "contract_id": CALIBRATION_LOGIC_NODE.artifact_outputs[0].contract_id,
        "size_bytes": result.artifact_path.stat().st_size,
    }
    summary = json.loads((run_root / "summary.json").read_text())
    assert set(summary) == {
        "format", "run", "models", "default_model", "best_model", "run_chain"
    }
    assert (run_root / "summary.txt").is_file()
    figures = run_root / "figures"
    for preview in sorted(figures.glob("*.png")):
        archive = preview.with_suffix(".npz")
        assert archive.is_file()
        info, arrays = read_archive(archive)
        assert info["schema"] == FIGURE_SCHEMA
        assert set(info["sections"]["plot"]) == {"data"}
        reopened, recipe = read_figure_plot(info, arrays, "data")
        snapshot = getattr(reopened, "snapshot", reopened)
        assert snapshot.block.values.size
        assert recipe["spec"].kind.value in {"curve", "image", "facet_grid"}

    box_info, box_arrays = read_archive(figures / "box.npz")
    _box_figure, box_recipe = read_figure_plot(box_info, box_arrays, "data")
    box_model = next(
        model for model in result.calibration.models if model.kind.value == "box"
    )
    np.testing.assert_allclose(
        [item["value"] for item in box_recipe["classifier_thresholds"]],
        box_model.thresholds[box_model.usable_sites],
    )
    box_report = result.report["models"]["box"]
    for target, site in zip(
        box_recipe["classifier_thresholds"],
        np.flatnonzero(box_model.usable_sites),
        strict=True,
    ):
        expected = np.asarray(
            [
                box_report["gaussian_dark_mean"][site],
                box_report["gaussian_dark_sigma"][site],
                box_report["gaussian_dark_weight"][site],
                box_report["gaussian_bright_mean"][site],
                box_report["gaussian_bright_sigma"][site],
                box_report["gaussian_bright_weight"][site],
            ],
            dtype=float,
        )
        components = target["gaussian_components"]
        if np.isfinite(expected).all():
            np.testing.assert_allclose(
                [
                    components["left_mean"],
                    components["left_sigma"],
                    components["left_weight"],
                    components["right_mean"],
                    components["right_sigma"],
                    components["right_weight"],
                ],
                expected,
            )
        else:
            assert components is None

    actual_info, actual_arrays = read_archive(figures / "actual_fidelity.npz")
    actual_figure, _actual_recipe = read_figure_plot(
        actual_info, actual_arrays, "data"
    )
    gaussian_info, gaussian_arrays = read_archive(
        figures / "gaussian_fidelity.npz"
    )
    gaussian_figure, _gaussian_recipe = read_figure_plot(
        gaussian_info, gaussian_arrays, "data"
    )
    model_names = tuple(model.kind.value for model in result.calibration.models)
    expected_actual = np.stack(
        [result.report["models"][name]["site_fidelity"] for name in model_names],
        axis=-1,
    )
    expected_gaussian = np.stack(
        [
            result.report["models"][name]["site_gaussian_fidelity"]
            for name in model_names
        ],
        axis=-1,
    )
    np.testing.assert_allclose(
        actual_figure.block.values[0, 0], expected_actual, equal_nan=True
    )
    np.testing.assert_allclose(
        gaussian_figure.block.values[0, 0], expected_gaussian, equal_nan=True
    )

    site_info, site_arrays = read_archive(figures / "site_map.npz")
    report_source = site_info["sections"]["source"]
    assert report_source["task"] == "calibration"
    assert set(report_source["run_record"]["actual_devices"]) == {
        "camera", "sequencer"
    }
    site_figure, _site_recipe = read_figure_plot(site_info, site_arrays, "data")
    assert isinstance(site_figure, ImageFrame)
    site_map = result.calibration.site_map
    contract = result.calibration.frame_contract
    roi = contract.roi_xywh
    origin_x, origin_y = (0, 0) if roi is None else (int(roi[0]), int(roi[1]))
    binning_y, binning_x = (int(value) for value in contract.binning_yx)
    expected_centers = np.asarray(site_map.centers_xy, dtype=float) * (
        binning_x,
        binning_y,
    ) + (origin_x, origin_y)
    np.testing.assert_allclose(site_figure.overlay.coordinates, expected_centers)
    assert site_figure.overlay.point_ids == site_map.site_ids
    assert site_figure.overlay.labels == tuple(
        str(index) for index in range(1, site_map.n_sites + 1)
    )
    assert site_figure.overlay.static_statuses == tuple(
        PointStatus.UNKNOWN if valid else PointStatus.INVALID
        for valid in site_map.valid_sites
    )

    registered = {item["path"]: item for item in run["artifacts"]}
    for preview in figures.glob("*.png"):
        assert registered[preview.relative_to(run_root).as_posix()]["contract_id"] == ""
        assert (
            registered[preview.with_suffix(".npz").relative_to(run_root).as_posix()][
                "contract_id"
            ]
            == FIGURE_SCHEMA
        )


def test_calibration_run_result_deep_owns_nested_plain_truth(tmp_path: Path) -> None:
    base = _task(_calibration_request(repeats=4)).run(tmp_path)
    report = {
        "nested": {"values": [1, 2]},
        "array": np.asarray([3.0, 4.0]),
    }
    pulse = {"program": {"slots": [1, 2, 3]}}
    run_record = {"request": {"photoelectrons": False}}
    summary = {"headline": {"fidelity": [0.9]}}
    result = CalibrationRunResult(
        base.artifact_path,
        base.calibration,
        report,
        base.capture,
        pulse,
        run_record,
        summary,
    )

    report["nested"]["values"][0] = 99
    report["array"][0] = 99.0
    pulse["program"]["slots"][0] = 99
    run_record["request"]["photoelectrons"] = True
    summary["headline"]["fidelity"][0] = 0.0
    assert result.report["nested"]["values"] == (1, 2)
    np.testing.assert_array_equal(result.report["array"], [3.0, 4.0])
    assert result.report["array"].flags.writeable is False
    assert result.pulse["program"]["slots"] == (1, 2, 3)
    assert result.run_record["request"]["photoelectrons"] is False
    assert result.summary["headline"]["fidelity"] == (0.9,)

    with pytest.raises(TypeError):
        result.report["nested"]["values"][0] = 99
    with pytest.raises(TypeError):
        result.pulse["program"]["slots"][0] = 99
    with pytest.raises(TypeError):
        result.run_record["request"]["photoelectrons"] = True
    with pytest.raises(TypeError, match="report must be a mapping"):
        replace(result, report=7)
    with pytest.raises(TypeError, match="summary must be a mapping"):
        replace(result, summary=[])


def test_a_folder_with_no_samples_says_so(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    request = replace(
        _calibration_request(repeats=4),
        frame_source=FRAMES_FROM_FOLDER,
        saved_frames_path=str(empty),
    )
    with pytest.raises(ValueError, match="no saved calibration samples"):
        _task(request).run(tmp_path)
    run = json.loads((tmp_path / "calibration" / "run.json").read_text())
    assert run["status"]["state"] == "failed"
    assert "no saved calibration samples" in run["error"]["message"]


def test_calibrating_from_a_folder_needs_the_folder() -> None:
    with pytest.raises(ValueError, match="needs the folder"):
        replace(
            _calibration_request(),
            frame_source=FRAMES_FROM_FOLDER,
            saved_frames_path="",
        )
