"""The one-Pulse camera/SLM flow and its operator-visible timing controls."""

from dataclasses import replace
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event
from types import SimpleNamespace
import time

import numpy as np
import pytest

from zlc_atom.devices.simulation.camera import VirtualCamera, VirtualCameraConfig
from zlc_atom.devices.simulation.sequencer import VirtualPulseStreamer
from zlc_atom.devices.slm.device import phase_from_codes
from zlc_atom.nodes._framework.descriptor import ResolvedWorkspaceResource
from zlc_atom.nodes.slm_rearrangement import task as task_module
from zlc_atom.nodes.slm_rearrangement.logic_node import LOGIC_NODE, SLM_REARRANGEMENT_SCHEMA
from zlc_atom.nodes.slm_rearrangement.task import SlmRearrangementTask, pulse_timing
from zlc_data.figure_archive import read_archive
from zlc_plot import read_figure_plot
from zlc_plot.primitives import image_point_overlay_geometry
from zlc_pulse import PulseBinding, PulseBracket, PulseFieldRef, PulsePeriod, PulseSequence, sequence_to_tree
from zlc_pulse.device import DoneReport
from zlc_pulse.wire import STATUS_DONE
from zlc_runtime import NodeHost, SignalDataPlane

from pulse_fixture import IMAGING_PULSE_RESOURCE
from test_slm_feedback_task import _Context, _Slm, _calibration_at, _science_context


def _sequence(gap_seconds=10., *, bindings=(), brackets=()):
    template = IMAGING_PULSE_RESOURCE.value
    states = tuple(0 for _ in template.target.raw_lanes)
    imaging_states = next(period.states for period in template.periods if period.period_id == "short")
    return PulseSequence(
        "operator_rearrangement", target=template.target, time_step_ns=template.time_step_ns,
        periods=(
            PulsePeriod("before", .01, "s", imaging_states, name="Before image"),
            PulsePeriod("gap", gap_seconds, "s", states, name="Transport"),
            PulsePeriod("after", .01, "s", imaging_states, name="Verify image"),
        ),
        bindings=bindings, brackets=brackets,
    )


class _Board:
    """Actual Pulse compile/load/Config, with external triggers under test control."""

    def __init__(self, camera, trace):
        self.device = VirtualPulseStreamer()
        self.device.open()
        self.camera, self.trace = camera, trace
        self.fires, self.loads, self.safe_calls = [], [], 0
        self.early_verification = False

    def describe(self):
        return self.device.describe()

    def compile_pulse(self, *args):
        return self.device.compile_pulse(*args)

    def load(self, program, *, source=None, rows=()):
        self.loads.append((program, source))
        return self.device.load(program, source=source, rows=rows)

    def applied(self):
        return self.device.applied()

    def fire(self, *, run_repeats, scan_repeats=1):
        self.fires.append((run_repeats, scan_repeats))
        self.trace.append("fire")
        self.camera.trigger(2 if self.early_verification else 1)
        return self.applied().with_repeats(run_repeats, scan_repeats)

    def wait_done(self, timeout=None):
        return DoneReport(STATUS_DONE, 0, False, .1, report_delay_seconds=.001)

    def safe(self):
        self.safe_calls += 1
        return self.device.safe()

    def close(self):
        self.device.close()


class _SequenceSlm(_Slm):
    def __init__(self, trace):
        super().__init__((8, 10))
        self.trace = trace
        self.preparations, self.plays, self.releases = [], 0, 0
        self.play_error = None
        self.cleanup_error = None
        self.cancel = Event()
        self.frames = None
        self.confirmed = 0

    def prepare_phase_sequence(self, codes, frame_interval_seconds, *, frame_count=None):
        self.trace.append("upload")
        values = np.empty((frame_count, *self.shape_yx), np.uint8) if codes is None else np.array(codes)
        self.preparations.append((values, frame_interval_seconds))
        self.frames = Queue(maxsize=2)
        self.confirmed = 0
        self.cancel.clear()
        if codes is not None:
            for index, frame in enumerate(values):
                self.submit_phase_frame(index, frame)
        return {"frame_count": len(values), "prepare_ms": .01, "streaming": codes is None}

    def submit_phase_frame(self, index, codes):
        self.preparations[-1][0][index] = codes
        while not self.cancel.is_set():
            try:
                self.frames.put((index, np.array(codes)), timeout=.01)
                return
            except Full:
                pass
        raise RuntimeError("the SLM stream was cancelled")

    def cancel_phase_sequence(self):
        self.cancel.set()

    def play_phase_sequence(self, stop_requested=None):
        self.trace.append("play")
        self.plays += 1
        codes = self.preparations[-1][0]
        try:
            if self.play_error is not None:
                raise self.play_error
            while self.confirmed < len(codes) and not self.cancel.is_set():
                if stop_requested is not None and stop_requested():
                    self.cancel.set()
                    break
                try:
                    index, frame = self.frames.get(timeout=.01)
                except Empty:
                    continue
                assert index == self.confirmed
                self.apply_phase(phase_from_codes(frame, self.shape_yx))
                self.confirmed += 1
        finally:
            result = {"frame_count": len(codes), "played_frames": self.confirmed,
                      "cancelled": self.cancel.is_set(), "final_settle_completed": self.confirmed == len(codes),
                      "physical_vblank_observed": False, "acknowledgment": "test device"}
            self.receipt_overrides["sequence"] = result
            self.cancel.set()  # Wake a producer on a device failure too.
        return {**result, "receipt": self.last_command_receipt}

    def release_phase_sequence(self):
        self.releases += 1
        self.cancel.set()
        self.frames = None
        if self.cleanup_error is not None:
            raise self.cleanup_error


class _RunContext(_Context):
    instance_id = "slm_rearrangement"
    generation = "rearrangement-test"

    def __init__(self, directory, camera, board, trace):
        directory.mkdir()
        super().__init__(directory)
        self.camera, self.board, self.trace = camera, board, trace

    def report_progress(self, message, *args, **kwargs):
        super().report_progress(message, *args, **kwargs)
        if message.startswith("SLM target held;") and not self.board.early_verification:
            # Delivery of the authored second external trigger is controlled
            # here so a slow numeric test never relies on arbitrary sleeps.
            self.trace.append("after_trigger")
            self.camera.trigger(1)


def _stub_figures(path, **kwargs):
    path = Path(path)
    path.write_bytes(b"test preview")
    archive = path.with_suffix(".npz")
    np.savez(archive, test_stub=np.asarray(1))
    return path, archive


@pytest.fixture
def experiment(tmp_path, monkeypatch, request):
    trace = []
    slm = _SequenceSlm(trace)
    source_xy = np.asarray(((2., 2.), (4., 2.), (7., 2.), (2., 5.), (4., 5.), (7., 5.)))
    camera_step = getattr(request, "param", 4) == "camera-step"
    affine_crop = getattr(request, "param", 4) in {"camera-affine-roi", "camera-step"}
    sensor_shape = (18, 24) if affine_crop else slm.shape_yx
    centers = source_xy @ np.asarray(((1.5 if camera_step else 1., 0.), (1/3., 1.))) + (13/3., 3.) if affine_crop else source_xy
    calibration = _calibration_at(centers, shape=sensor_shape)
    if affine_crop:
        calibration = replace(calibration, frame_contract=replace(calibration.frame_contract,
            sensor_shape=sensor_shape, roi_xywh=(0, 0, sensor_shape[1], sensor_shape[0])))
    usable = np.ones(len(centers), bool)
    usable[2] = False
    calibration = replace(calibration, models=(replace(
        calibration.select_model(), usable_sites=usable,
    ),))
    target = np.zeros(slm.shape_yx, np.float32)
    target[source_xy[:, 1].astype(int), source_xy[:, 0].astype(int)] = 1
    source_context = _science_context(slm, target=target)
    images = []
    for before in (True, False):
        image = np.zeros(sensor_shape, np.uint16)
        image[np.rint(centers[:, 1]).astype(int), np.rint(centers[:, 0]).astype(int)] = 10
        if not before:
            image[int(round(centers[1, 1])), int(round(centers[1, 0]))] = 0
        images.append(image)
    camera = VirtualCamera(VirtualCameraConfig(frame_shape_yx=sensor_shape),
                           frame_source=lambda exposure: images.pop(0))
    if affine_crop:
        set_roi = camera.set_roi
        # Exercise a camera's accepted crop using its public readback path;
        # Calibration.rebased remains the only calibration-coordinate owner.
        monkeypatch.setattr(camera, "set_roi", lambda _requested: set_roi((4, 3, 14, 10)))
        from zlc_atom.devices.simulation.camera import adapter as camera_module
        record_type = camera_module.CameraFrameRecord
        def sdk_timestamped_record(*args, **kwargs):
            record = record_type(*args, **kwargs)
            return replace(record, timestamp_seconds=100+4*record.source_ordinal,
                           timestamp_microseconds=10000+20000*record.source_ordinal)
        monkeypatch.setattr(camera_module, "CameraFrameRecord", sdk_timestamped_record)
    board = _Board(camera, trace)
    plane = SignalDataPlane()
    context = _RunContext(tmp_path / "run", camera, board, trace)
    state = SimpleNamespace(trace=trace, camera=camera, board=board, slm=slm,
                            context=context, plane=plane, available_indices=[], closed=[], images=images,
                            affine_crop=affine_crop, calibrated_centers=centers)

    def prepare(**kwargs):
        trace.append("prepare_gpu")
        state.prepare_arguments = kwargs
        return {"gpu_info": {"device_name": "test CUDA boundary"},
                "initial_phase": source_context["phase"],
                "close": lambda: state.closed.append(True)}

    def plan(prepared, available):
        trace.append("plan")
        indices = np.asarray(available)
        assert indices.dtype.kind in "iu" and indices.ndim == 1
        state.available_indices.append(indices.copy())
        n = min(len(indices), len(state.prepare_arguments["target_yx"]))
        destinations = np.asarray((1, 0, 3, 2)) if n == 4 else np.arange(n)
        starts = np.asarray(state.prepare_arguments["source_yx"])[indices[:n]]
        ends = np.asarray(state.prepare_arguments["target_yx"])[destinations]
        return {"assigned_source_indices": indices[:n], "assigned_target_indices": destinations,
                "removed_source_indices": indices[n:], "source_indices": indices[:n],
                "target_indices": destinations,
                "motion_yx": np.stack((starts, ends)), "fraction": np.asarray((0.,1.)),
                "target_filled": np.arange(len(state.prepare_arguments["target_yx"])) < n}

    def compute(prepared, planned, **kwargs):
        trace.append("compute")
        assert planned["assigned_source_indices"].dtype.kind in "iu"
        if board.early_verification:
            deadline = time.monotonic() + 1
            while camera._records.produced_count < 2 and time.monotonic() < deadline:
                time.sleep(.001)
            assert camera._records.produced_count == 2
        frames = kwargs["motion_frames"] if len(planned["source_indices"]) else 0
        codes = np.full((frames, *slm.shape_yx), 32, np.uint8)
        codes.flags.writeable = False
        for index, frame in enumerate(codes):
            kwargs["frame_ready"](index, frame)
        starts = np.asarray(state.prepare_arguments["source_yx"], dtype=np.float64)[planned["source_indices"]]
        points = starts.copy()
        points[:len(planned["assigned_source_indices"])] = state.prepare_arguments["target_yx"][planned["assigned_target_indices"]]
        samples = (np.r_[0.,0.,0.,kwargs["motion_fractions"]] if kwargs.get("motion_fractions") is not None
                   else np.linspace(0,1,kwargs["motion_frames"]+1))
        return {"phase_codes": codes, "converged": True,
                "target_synthesis_coefficients": None,
                "motion_yx": starts[None] + samples[:,None,None]*(points-starts)[None],
                "fraction": np.linspace(0, 1, kwargs["motion_frames"] + 1),
                "support_intensity_ratios": np.full(frames, 1.005),
                "brightness_minimum_to_initial": np.linspace(2.3, 3.3, frames),
                "brightness_mean_to_initial": np.linspace(2.31, 3.31, frames),
                "brightness_maximum_to_initial": np.linspace(2.32, 3.32, frames),
                "phase_step_max_rad": np.full(frames, .5),
                "pupil_phase_step_rms_rad": np.full(frames, 1.1),
                "background_intensity_ratios": np.zeros(frames), "timing_ms": {"total": .2}}

    monkeypatch.setattr(task_module, "prepare_rearrangement", prepare)
    monkeypatch.setattr(task_module, "plan_rearrangement", plan)
    monkeypatch.setattr(task_module, "compute_rearrangement", compute)
    state.compute = compute
    state.task_arguments = dict(
        camera=camera, camera_key="camera", sequencer=board, sequencer_key="pulse",
        slm=slm, slm_key="slm", signal_plane=plane,
        calibration=calibration, calibration_path=tmp_path / "calibration.json",
        science_context=source_context, science_context_path=tmp_path / "science_context.npz",
        target_rows=1 if getattr(request, "param", 4) == 2 else 2, target_columns=2,
        pulse_sequence=_sequence(), pulse_path=tmp_path / "operator.json",
        before_period="before", after_period="after", motion_frames=2,
        save_figure_artifact=_stub_figures,
    )
    state.task = SlmRearrangementTask(**state.task_arguments)
    try:
        yield state
    finally:
        state.task.close()
        camera.close()
        board.close()
        plane.close()


@pytest.mark.parametrize("experiment", [4, "camera-affine-roi"], indirect=True)
def test_one_authored_pulse_runs_photograph_compute_play_verify_and_reopen_figures(experiment):
    e = experiment
    authored = sequence_to_tree(e.task.sequence)
    # Integration is independent of the trigger Period's length, including
    # camera readback longer than the entire authored imaging Period.
    e.task.exposure_seconds = .02
    # This one case uses the real Figure writer and reader. The other cases
    # focus on interruption and do not pay for rendering the same pictures.
    e.task._save_figure_artifact = None
    result = e.task.execute(e.context)
    assert e.board.fires == [(1, 1)] and len(e.board.loads) == 1
    assert sequence_to_tree(e.task.sequence) == authored
    assert e.board.loads[0][1].periods == e.task.sequence.periods
    assert e.trace.index("fire") < e.trace.index("compute") < e.trace.index("upload") < e.trace.index("play") < e.trace.index("after_trigger")
    np.testing.assert_array_equal(e.available_indices[0], [0, 1, 3, 4, 5])
    np.testing.assert_array_equal(e.task.target_indices, [1, 2, 4, 5])
    assert {item.name for item in LOGIC_NODE.input_specs} == {"calibration_path", "science_context_path", "end_target_path"}
    assert e.slm.plays == 1 and e.closed == []
    assert e.context.terminal_sealed
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    assert summary["target_sites"] == 4 and summary["judged_target_sites"] == 3
    assert summary["filled_target_sites"] == 2
    assert summary["missing_target_indices"] == [0]
    assert summary["invalid_target_indices"] == [1]
    assert result["target_filling_fraction"] is None
    assert summary["verification_complete"] is False
    assert summary["judged_target_filling_fraction"] == pytest.approx(2/3)
    field = summary["computed_field_diagnostics"]
    assert "not measured optical response" in field["basis"]
    assert field["minimum_power_ratio"] == 2.3 and field["maximum_power_ratio"] == 3.32
    assert field["final_minimum_power_ratio"] == 3.3 and field["final_mean_power_ratio"] == 3.31
    assert field["final_maximum_power_ratio"] == 3.32
    assert field["maximum_site_phase_step_rad"] == .5
    assert field["maximum_pupil_phase_step_rms_rad"] == 1.1
    assert "Computed field final mean power ratio: 3.31" in (e.context.run_directory / "summary.txt").read_text()
    outcomes = summary["transport_outcomes"]
    assert outcomes["verification_accepted"]
    assert "not individual-atom identity" in outcomes["basis"]
    assert outcomes["moving"] == {"assigned": 2, "judged": 1, "occupied": 1}
    assert outcomes["stationary"] == {"assigned": 2, "judged": 2, "occupied": 1}
    assignments = outcomes["assignments"]
    assert [item["moving"] for item in assignments] == [True, False, True, False]
    assert assignments[0]["after_valid"] is False and assignments[0]["after_occupied"] is None
    assert assignments[1]["after_valid"] is True and assignments[1]["after_occupied"] is False
    assert assignments[0]["source_label"] == "1" and assignments[0]["target_label"] == "3"
    assert assignments[0]["source_index"] == 0 and assignments[0]["target_source_index"] == 2
    observed = summary["observed_timing"]
    assert observed["requested_interval_ms"] == pytest.approx(1000/60)
    assert observed["logical_steps"] == 2
    assert observed["command_interval_median_ms"] is None, "no fabricated cadence when the receipt has none"
    assert observed["online_ms"] == summary["timing_ms"]["online_rearrangement"]
    assert observed["playback_ms"] == summary["timing_ms"]["sequence_play_and_final_settle"]
    if e.affine_crop:
        assert observed["photo_camera_interval_ms"] == pytest.approx(4020.)
        assert e.task._records[0].image.dtype == np.dtype('uint16')
        assert e.task._records[0].image.shape == (10, 14)
    else:
        assert "photo_camera_interval_ms" not in observed
    assert [item["source_ordinal"] for item in summary["frame_records"]] == [0, 1]
    with np.load(result["artifact_path"], allow_pickle=False) as data:
        assert all(not data[key].dtype.hasobject for key in data.files)
        original_motion = data["motion_yx"].copy()
    before_device_facts = None
    before_parameters = None
    for name in ("before_frame", "after_frame", "phase", "trajectory_2d", "intensity_ratio"):
        archive = e.context.artifacts[name + "_figure"][0]
        assert archive.is_file()
        info, arrays, datasets = read_archive(archive)
        assert datasets
        loaded, recipe = read_figure_plot(info, arrays, datasets, next(iter(datasets)))
        assert loaded is not None and recipe["spec"] is not None
        if name == "before_frame":
            before_device_facts = info["sections"]["source"]["run_record"]["device_snapshots"]
            before_parameters = recipe["parameters"]
        if name == "trajectory_2d":
            path_device_facts = info["sections"]["source"]["run_record"]["device_snapshots"]
            assert path_device_facts["camera"] == before_device_facts["camera"]
            assert path_device_facts["slm"] == before_device_facts["slm"]
            selected = e.task._plan["source_indices"]
            before = e.task._snapshots[task_module.BEFORE_FRAME_OUTPUT.name]
            assert loaded.snapshot.block.schema == before.block.schema
            np.testing.assert_array_equal(loaded.snapshot.block.values, before.block.values)
            assert loaded.overlay.point_ids == tuple(e.task._overlay_geometry["point_ids"])
            assert loaded.overlay.labels == tuple(e.task._overlay_geometry["labels"])
            np.testing.assert_array_equal(loaded.overlay.status.block.values,
                e.task._overlays[task_module.BEFORE_FRAME_OUTPUT.name].status.block.values)
            np.testing.assert_array_equal(loaded.overlay.status.expanded_validity(),
                e.task._overlays[task_module.BEFORE_FRAME_OUTPUT.name].status.expanded_validity())
            assert loaded.overlay.paths_xy.shape == (6, 3, 2)
            unselected = np.setdiff1d(np.arange(6), selected)
            np.testing.assert_array_equal(loaded.overlay.paths_xy[unselected],
                np.repeat(loaded.overlay.coordinates[unselected, None, :], 3, axis=1))
            source_camera = e.calibrated_centers
            roi_shift = np.asarray((4., 3.)) if e.affine_crop else np.zeros(2)
            motion = original_motion
            camera_indices = (motion[..., ::-1] @ np.asarray(((1., 0.), (1/3., 1.))) + (13/3., 3.)
                              if e.affine_crop else motion[..., ::-1]) - roi_shift
            full_indices = np.repeat((source_camera-roi_shift)[:, None, :], 3, axis=1)
            full_indices[selected] = camera_indices.transpose(1, 0, 2)
            expected = image_point_overlay_geometry(before, source_camera-roi_shift,
                loaded.overlay.point_ids, status_axis=e.task._status_axis,
                labels=loaded.overlay.labels, coordinates_are_indices=True, paths_xy=full_indices)
            np.testing.assert_allclose(loaded.overlay.paths_xy, expected["paths_xy"], atol=1e-12)
            np.testing.assert_allclose(loaded.overlay.paths_xy[:, 0], loaded.overlay.coordinates, atol=1e-12)
            assert recipe["spec"].x.axis_id == str(before.block.schema.cell_domain.axes[1].axis_id)
            assert recipe["spec"].y.axis_id == str(before.block.schema.cell_domain.axes[0].axis_id)
            assert recipe["parameters"] == before_parameters
        from zlc_workbench.viewer import describe_archive
        description = describe_archive(info, arrays)
        assert e.task.instance_id in dict(dict(description.tabs)["Logic"])
    with np.load(result["artifact_path"], allow_pickle=False) as data:
        assert data["phase_codes"].shape == (2, 8, 10)
        expected_end = e.prepare_arguments["target_yx"][e.task._plan["target_indices"]]
        expected_start = e.task.points[0][e.task._plan["source_indices"]]
        np.testing.assert_array_equal(data["motion_yx"], expected_start[None]
            + np.linspace(0, 1, 3)[:, None, None]*(expected_end-expected_start)[None])
        np.testing.assert_allclose(data["motion_camera_xy"],
            np.asarray(expected["paths_xy"])[selected].transpose(1, 0, 2), atol=1e-12)
        np.testing.assert_array_equal(data["before_thresholds"], np.full(6, 5.))
        assert not data["after_target_valid"][1] and not data["after_target_occupied"][1]


@pytest.mark.parametrize("occupied_count", [0, 2])
def test_shortage_fills_a_subset_and_empty_input_holds_the_phase(experiment, occupied_count):
    e = experiment
    e.images[0][:] = 0
    for y, x in e.task.points[0][:occupied_count]:
        e.images[0][y, x] = 10
    result = e.task.execute(e.context)
    assert Path(result["artifact_path"]).is_file()
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["assigned_atoms"] == occupied_count
    assert len(summary["unfilled_target_indices"]) == 4 - occupied_count
    assert e.slm.plays == int(occupied_count > 0)
    assert e.board.fires == [(1, 1)]


@pytest.mark.parametrize("experiment", ["camera-step"], indirect=True)
def test_camera_step_uses_sensor_geometry_and_restarts_keep_only_gpu_preparation(experiment):
    e = experiment
    photos = [image.copy() for image in e.images]
    e.task.frame_mode = "camera_step"
    e.task.max_camera_step = .6
    assert len(e.task.dataset_output_declarations) == 5
    result = e.task.execute(e.context)
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    # Five Fourier pixels map to 7.5 sensor pixels through the real registered
    # affine/cropped camera geometry: ceil(7.5/.6) + two source-fade maps.
    assert summary["actual_motion_frames"] == 15
    assert summary["requested_motion_frames"] == 2 and summary["motion_frames"] is None
    assert summary["actual_maximum_camera_step"] <= .6 + 1e-12
    with np.load(result["artifact_path"], allow_pickle=False) as data:
        assert data["phase_codes"].shape[0] == 15
        assert np.max(np.linalg.norm(np.diff(data["motion_camera_xy"],axis=0),axis=-1)) <= .6 + 1e-12
    prepared = e.task._prepared
    fresh = SlmRearrangementTask(**(e.task_arguments | {
        "frame_mode": "camera_step", "max_camera_step": .6, "exposure_seconds": .03}))
    assert e.task.restart_from(fresh)
    assert e.task._prepared is prepared and e.task.exposure_seconds == .03
    assert not e.task._detections and not e.task._records
    again = _RunContext(e.context.run_directory.parent / "again", e.camera, e.board, e.trace)
    e.images.extend(photos)
    e.task.execute(again)
    repeated = json.loads((again.run_directory / "summary.json").read_text())
    assert repeated["gpu_preparation_reused"]
    assert e.trace.count("prepare_gpu") == 1 and e.closed == []
    changed_context = dict(e.task_arguments["science_context"])
    changed_context["phase"] = np.remainder(changed_context["phase"] + .1, 2*np.pi)
    changed = SlmRearrangementTask(**(e.task_arguments | {"science_context": changed_context}))
    assert not e.task.restart_from(changed), "same file path cannot hide changed optical data"
    assert e.task._prepared is prepared
    e.task.close()
    assert e.closed == [True]


def test_explicit_end_target_does_not_invent_calibration_for_a_new_position(experiment):
    e = experiment
    old = e.task
    target = np.zeros(e.slm.shape_yx, np.float32)
    target[1,1] = 1
    e.task = SlmRearrangementTask(
        camera=e.camera, camera_key=old.camera_key, sequencer=e.board, sequencer_key=old.sequencer_key,
        slm=e.slm, slm_key=old.slm_key, signal_plane=e.plane,
        calibration=old.calibration, calibration_path=old.calibration_path,
        science_context=old.science_context, science_context_path=old.context_path,
        target_intensity=target, target_path=old.context_path.parent / "end-target.json",
        pulse_sequence=old.sequence, pulse_path=old.pulse_path,
        before_period=old.before_period, after_period=old.after_period,
        motion_frames=2, save_figure_artifact=_stub_figures)
    result = e.task.execute(e.context)
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    assert summary["assigned_atoms"] == 1 and summary["removed_atoms"] == 4
    assert summary["invalid_target_indices"] == [0] and not summary["verification_complete"]
    assignment = summary["transport_outcomes"]["assignments"][0]
    assert assignment["target_source_index"] is None and assignment["target_label"] is None
    assert assignment["after_valid"] is False and assignment["after_occupied"] is None
    assert result["target_filling_fraction"] is None
    np.testing.assert_array_equal(e.task.points[1], [[1,1]])


@pytest.mark.parametrize("experiment", [2], indirect=True)
def test_hosted_rearrangement_keeps_frozen_vocabulary_shared_records_and_source_target_union(experiment, tmp_path, monkeypatch):
    e = experiment
    e.task._save_figure_artifact = None
    wake = Event()
    host = NodeHost(e.task, e.plane, wake.set, instance_id="slm_rearrangement",
                    kind="task", dataset_output_declarations=e.task.dataset_output_declarations,
                    required_artifacts={item.name: item.contract_id for item in LOGIC_NODE.artifact_outputs},
                    task_name=LOGIC_NODE.api_name)
    publications = []
    names = {declaration.name: host.signal_key(declaration.name) for declaration in e.task.dataset_output_declarations}

    def received(signals):
        if names["phase"] not in signals:
            return
        publication = e.plane.latest_publication(names["phase"])
        snapshots = {name: e.plane.current_dataset(signal, publication).materialize()
                     for name, signal in names.items()}
        publications.append((publication, snapshots))

    unsubscribe = e.plane.subscribe_publications(received)
    original_progress = host._report_progress

    def progress(value):
        original_progress(value)
        # Only the fake external trigger is delegated. Execution, publication,
        # cancellation, terminal sealing and TaskRun remain the actual host's.
        e.context.report_progress(value.message, current=value.current, total=value.total)

    monkeypatch.setattr(host, "_report_progress", progress)
    try:
        host.start(run_root=tmp_path, input_summary={"source_sites": 6, "target_sites": 2})
        deadline = time.monotonic() + 20
        while not host.terminal and time.monotonic() < deadline:
            host.poll()
            wake.wait(.01)
            wake.clear()
        observation = host.poll()
        assert observation.terminal and observation.phase == "done", observation
        assert observation.error is None and observation.progress is None
        assert e.board.fires == [(1, 1)] and e.slm.plays == 1
        assert e.closed == []
        assert len(publications) >= 4
        first, initial = publications[0]
        frozen = {name: snapshot.block.schema for name, snapshot in initial.items()}
        assert initial["phase"].expanded_validity().all()
        for name in set(names) - {"phase"}:
            assert not initial[name].expanded_validity().any(), name
        for publication, snapshots in publications:
            assert set(publication.signals) == set(names.values())
            assert all(value.event_record == publication.event_record for value in publication.signals.values())
            assert {name: snapshot.block.schema for name, snapshot in snapshots.items()} == frozen
            assert snapshots["before_occupied"].block.values.shape == (1, 1, 6)
            assert snapshots["after_occupied"].block.values.shape == (1, 1, 6)
        terminal, final = publications[-1]
        assert terminal.event_ref.generation == first.event_ref.generation
        assert terminal.event_ref.sequence > first.event_ref.sequence
        np.testing.assert_array_equal(final["before_occupied"].block.values[0, 0], [True, True, False, True, True, True])
        np.testing.assert_array_equal(final["before_occupied"].expanded_validity()[0, 0], [True, True, False, True, True, True])
        np.testing.assert_array_equal(final["after_occupied"].expanded_validity()[0, 0], [True, True, False, True, True, True])
        assert final["before_frame"].expanded_validity().all() and final["after_frame"].expanded_validity().all()
        assert terminal.event_record["capture_events"]["before_frame"]["source_ordinal"] == 0
        assert terminal.event_record["capture_events"]["after_frame"]["source_ordinal"] == 1
        assert not e.plane.is_generation_live(names["phase"])
        np.testing.assert_array_equal(final["phase"].block.values[0, 0], e.slm.last_commanded_phase)
        directory = host.run_directory
        run = json.loads((directory / "run.json").read_text())
        assert (directory / "start.json").is_file() and run["status"]["state"] == "completed"
        assert json.loads((directory / "summary.json").read_text())["target_sites"] == 2
        assert Path(host.final_result["artifact_path"]).is_file()
        assert all(artifact.path.is_file() for artifact in host.artifacts)
        for name in ("before_frame", "after_frame", "phase", "trajectory_2d", "intensity_ratio"):
            archive = directory / "figures" / f"{name}.npz"
            info, arrays, datasets = read_archive(archive)
            loaded, recipe = read_figure_plot(info, arrays, datasets, next(iter(datasets)))
            assert loaded is not None and recipe["spec"] is not None
    finally:
        unsubscribe()
        if not host.terminal:
            host.cancel("test cleanup")
            deadline = time.monotonic() + 5
            while not host.terminal and time.monotonic() < deadline:
                host.poll()
                wake.wait(.01)
                wake.clear()
        deadline = time.monotonic() + 5
        while not host.shutdown() and time.monotonic() < deadline:
            host.poll()
            wake.wait(.01)
            wake.clear()
        assert host.shutdown()


@pytest.mark.parametrize("failure", ["short-gap", "gpu-prepare", "after-arm"])
def test_short_authored_gap_rejects_before_fire_and_keeps_pulse_unchanged(experiment, monkeypatch, failure):
    e = experiment
    if failure == "short-gap":
        e.task.sequence = _sequence(.001)
        error_type, message = RuntimeError, "nominal playback needs"
    elif failure == "after-arm":
        prepare_outputs = e.task._prepare_outputs
        def refused(*_args):
            raise RuntimeError("injected post-arm failure")
        monkeypatch.setattr(e.task, "_prepare_outputs", refused)
        error_type, message = RuntimeError, "post-arm failure"
    else:
        previous = {"played_frames": 7, "play_ms": 321., "cancelled": False}
        e.slm.receipt_overrides["sequence"] = previous

        def unavailable(**_kwargs):
            raise RuntimeError("GPU preparation failed")

        monkeypatch.setattr(task_module, "prepare_rearrangement", unavailable)
        error_type, message = RuntimeError, "GPU preparation failed"
    authored = sequence_to_tree(e.task.sequence)
    with pytest.raises(error_type, match=message):
        e.task.execute(e.context)
    assert e.board.fires == ([(1, 1)] if failure == "short-gap" else []) and e.slm.plays == 0
    assert sequence_to_tree(e.task.sequence) == authored
    assert "partial_data" in e.context.artifacts
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    assert summary["playback"] is None
    if failure == "gpu-prepare":
        assert summary["device_snapshots"]["slm"]["command_receipt"]["sequence"] == previous
        assert e.slm.commands == [], "early failure must retain the preceding device command"
    if failure == "after-arm":
        # The operator changes exposure and starts again in the same session.
        # Hardware is closed, but the companion producer must be released too.
        monkeypatch.setattr(e.task, "_prepare_outputs", prepare_outputs)
        e.task.exposure_seconds = .001
        retry = _RunContext(e.context.run_directory.parent / "retry", e.camera, e.board, e.trace)
        result = e.task.execute(retry)
        assert Path(result["artifact_path"]).is_file()
        assert e.board.fires == [(1, 1)] and e.slm.plays == 1


def test_buffered_second_frame_is_not_accepted_from_its_late_callback(experiment):
    e = experiment
    e.board.early_verification = True
    with pytest.raises(RuntimeError, match="before SLM playback completed"):
        e.task.execute(e.context)
    assert e.board.fires == [(1, 1)] and e.slm.plays == 1
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["verification_complete"] is False
    assert summary["target_filling_fraction"] is None
    assert summary["judged_target_filling_fraction"] is None


def test_recording_collects_fifty_frames_while_compute_and_play_are_active_and_saves_them(
    experiment, monkeypatch,
):
    from zlc_atom.devices.simulation.camera import adapter as camera_module
    from zlc_atom.nodes.camera_measurement import CameraMeasurementNode

    e = experiment
    expected = []
    template = e.images[0].copy()
    for index in range(50):
        image = template.copy()
        image[0, 0] = 1000 + index  # Outside every calibrated site BOX.
        expected.append(image)
    e.images[:] = [image.copy() for image in expected]
    raw_record = camera_module.CameraFrameRecord

    def sdk_record(*args, **kwargs):
        record = raw_record(*args, **kwargs)
        return replace(record, timestamp_seconds=100 + record.source_ordinal // 20,
                       timestamp_microseconds=(record.source_ordinal % 20) * 50000)

    monkeypatch.setattr(camera_module, "CameraFrameRecord", sdk_record)
    base = _sequence()
    bright, dark = base.periods[0].states, base.periods[1].states
    periods = tuple(period for index in range(50) for period in (
        PulsePeriod(f"image_{index}", .001, "s", bright, name=f"Image {index+1}"),
        PulsePeriod(f"gap_{index}", .001, "s", dark),
    ))
    sequence = replace(base, periods=periods)
    authored = sequence_to_tree(sequence)
    arguments = dict(e.task_arguments, recording_frames=50, pulse_sequence=sequence,
                     before_period="image_0", exposure_seconds=.001)
    arguments.pop("after_period")
    e.task = SlmRearrangementTask(**arguments)
    # No synthetic second trigger from the old two-photograph progress helper.
    monkeypatch.setattr(e.context, "report_progress", lambda *args, **kwargs:
                        _Context.report_progress(e.context, *args, **kwargs))

    def fire(*, run_repeats, scan_repeats=1):
        e.board.fires.append((run_repeats, scan_repeats))
        e.trace.append("fire")
        e.camera.trigger(50)
        return e.board.applied().with_repeats(run_repeats, scan_repeats)

    monkeypatch.setattr(e.board, "fire", fire)
    collected, compute_started, compute_finished = Event(), Event(), Event()
    play_started, play_finished = Event(), Event()
    consumed = []
    commit = CameraMeasurementNode._commit_direct_cycle

    def commit_cycle(node, cycle, index):
        commit(node, cycle, index)
        consumed.extend(cycle)
        if index == 49:
            assert compute_started.is_set()
            assert not compute_finished.is_set() and not play_finished.is_set()
            collected.set()

    monkeypatch.setattr(CameraMeasurementNode, "_commit_direct_cycle", commit_cycle)

    def compute(*args, **kwargs):
        compute_started.set()
        result = e.compute(*args, **kwargs)
        deadline = time.monotonic() + 2
        while not collected.wait(.002):
            assert time.monotonic() < deadline, "camera collector blocked behind computation"
        compute_finished.set()
        return result

    play = e.slm.play_phase_sequence

    def delayed_play(stop_requested=None):
        play_started.set()
        deadline = time.monotonic() + 2
        while not collected.wait(.002):
            assert time.monotonic() < deadline, "camera collector blocked behind playback"
        try:
            return play(stop_requested)
        finally:
            play_finished.set()

    monkeypatch.setattr(task_module, "compute_rearrangement", compute)
    monkeypatch.setattr(e.slm, "play_phase_sequence", delayed_play)
    e.task.execute(e.context)
    assert collected.is_set() and compute_finished.is_set() and play_finished.is_set()
    assert e.board.safe_calls == 0
    assert len(consumed) == 50
    assert [record.source_ordinal for record in consumed] == list(range(len(consumed)))
    assert e.board.fires == [(1, 1)] and len(e.board.loads) == 1
    assert sequence_to_tree(sequence) == authored
    assert e.slm.releases == 1
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert not summary["transport_outcomes"]["verification_accepted"]
    from PIL import Image
    files = sorted((e.context.run_directory / "frames").glob("*.tif"))
    assert len(files) == 50
    for path, original in zip(files, expected, strict=True):
        with Image.open(path) as saved:
            np.testing.assert_array_equal(np.asarray(saved), original)


@pytest.mark.parametrize("ending", ["device", "stopped", "numeric"])
def test_failed_or_stopped_play_keeps_completed_before_photo(experiment, monkeypatch, ending):
    e = experiment
    stopped = ending == "stopped"
    phase_figures = []

    def figures(path, **values):
        if Path(path).stem == "phase":
            phase_figures.append((values["plot_input"], values["source"]["run_record"]["device_snapshots"]["slm"]))
        return _stub_figures(path, **values)

    e.task._save_figure_artifact = figures
    if stopped:
        original = e.compute

        def compute(*args, **kwargs):
            result = original(*args, **kwargs)
            e.context.cancelled = True
            raise RuntimeError("the run was cancelled")

        e.context.cancelled = False
        e.slm.play_error = None
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(task_module, "compute_rearrangement", compute)
            with pytest.raises(RuntimeError, match="cancelled"):
                e.task.execute(e.context)
    elif ending == "device":
        e.slm.play_error = RuntimeError("injected sequence failure")
        e.slm.cleanup_error = RuntimeError("injected release failure")
        original_play = e.slm.play_phase_sequence
        original_submit = e.slm.submit_phase_frame
        submit_blocked = Event()

        def blocked_submit(index, frame):
            if index == 0:
                return original_submit(index, frame)
            submit_blocked.set()
            assert e.slm.cancel.wait(2)
            raise RuntimeError("queue admission cancelled before accepting its frame")

        def partial_play(stop_requested=None):
            # A failed play can leave a later frame confirmed. That current
            # receipt must not retag the earlier source-phase figure.
            index, frame = e.slm.frames.get(timeout=2)
            assert index == 0
            e.slm.apply_phase(phase_from_codes(frame, e.slm.shape_yx))
            e.slm.confirmed = 1
            assert submit_blocked.wait(2), "display failure occurs during blocked submission, not its precheck"
            return original_play(stop_requested)

        monkeypatch.setattr(e.slm, "play_phase_sequence", partial_play)
        monkeypatch.setattr(e.slm, "submit_phase_frame", blocked_submit)
        with pytest.raises(RuntimeError, match="injected sequence failure") as failure:
            e.task.execute(e.context)
        assert any("queue admission cancelled" in note for note in failure.value.__notes__)
    else:
        original = e.compute

        def rejected(*args, **kwargs):
            ready = kwargs["frame_ready"]
            kwargs["frame_ready"] = lambda index, frame: ready(index, frame) if index == 0 else None
            result = original(*args, **kwargs)
            result["converged"] = False
            return result

        monkeypatch.setattr(task_module, "compute_rearrangement", rejected)
        with pytest.raises(RuntimeError, match="encoded-field quality checks"):
            e.task.execute(e.context)
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    assert summary["status"] == ("stopped" if stopped else "failed")
    assert summary["before_occupied"] == 5 and "filled_target_sites" not in summary
    assert e.board.fires == [(1, 1)] and e.board.safe_calls == 1
    assert "partial_data" in e.context.artifacts and "artifact_path" not in e.context.artifacts
    assert e.task._result is None
    assert len(phase_figures) == 1
    phase, phase_device = phase_figures[0]
    np.testing.assert_array_equal(phase.block.values[0, 0], e.task.science_context["phase"])
    assert phase_device["command_receipt"] == summary["capture_events"]["phase"]["device_snapshots"]["slm"]["command_receipt"]
    if ending == "device":
        assert phase_device["command_revision"] < summary["device_snapshots"]["slm"]["command_revision"]
        assert not np.array_equal(e.slm.last_commanded_phase, phase.block.values[0, 0])
    elif ending == "numeric":
        assert "encoded-field quality checks" in summary["error"]
        assert summary["playback"]["cancelled"], "a successful cancelled play must not mask numerical rejection"
        outcomes = summary["transport_outcomes"]
        assert not outcomes["verification_accepted"]
        assert outcomes["moving"]["judged"] is None and outcomes["stationary"]["occupied"] is None
        assert all(item["after_valid"] is None and item["after_occupied"] is None
                   for item in outcomes["assignments"])
    with np.load(e.context.artifacts["partial_data"][0], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["before_valid"], [True, True, False, True, True, True])
        assert "before_image" in data.files and "after_image" not in data.files


def test_period_budget_uses_applied_config_and_expands_loops(experiment):
    board = experiment.board.device
    sequence = _sequence(.1, bindings=(PulseBinding(
        PulseFieldRef("duration", "gap"), "s", source="config", config_key="transport_gap",
    ),), brackets=(PulseBracket("transport_loop", "gap", "gap", 3),))
    description = board.describe()
    resolved, program = board.compile_pulse(sequence, description.geometry, description.clock_hz)
    board.load_config_values({"transport_gap": (.2, "s")})
    board.load(program, source=resolved)
    actual = board.applied()
    timing = pulse_timing(actual.source, actual.program, actual.rows, "before", "after")
    assert timing["before_start_seconds"] == 0
    assert timing["available_gap_seconds"] == pytest.approx(.6)
    assert timing["after_start_seconds"] == pytest.approx(.61)
    assert timing["pulse_duration_seconds"] == pytest.approx(.62)
    repeated = replace(sequence, brackets=(PulseBracket("repeat_before", "before", "before", 1_000_000_000),))
    resolved, program = board.compile_pulse(repeated, description.geometry, description.clock_hz)
    with pytest.raises(ValueError, match="Before imaging Period must play exactly once"):
        pulse_timing(resolved, program, (), "before", "after")
    with pytest.raises(ValueError, match="absent"):
        pulse_timing(actual.source, actual.program, actual.rows, "missing", "after")


def test_real_period_form_preserves_choice_identity_and_updates_disabled_nominal_duration(tmp_path):
    from PyQt5 import QtCore, QtTest, QtWidgets
    from zlc_workbench.authoring_form import project_logic_schema

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    form = LOGIC_NODE.ui_contributions[0]()
    try:
        sequence = _sequence()
        resource = ResolvedWorkspaceResource(tmp_path / "pulse.json", "zlc.pulse/slm-rearrangement", sequence)
        values = SLM_REARRANGEMENT_SCHEMA.draft_values()
        camera_outputs = ("before_frame", "before_occupied", "after_frame", "after_occupied", "phase")
        for phase_method in ("iterative", "lpi"):
            assert tuple(output.name for output in LOGIC_NODE.outputs_for(
                {**values, "phase_method": phase_method}, {})) == camera_outputs + ("trajectory", "intensity_ratio")
            assert tuple(output.name for output in LOGIC_NODE.outputs_for(
                {**values, "phase_method": phase_method, "frame_mode": "camera_step"}, {})) == camera_outputs
            assert tuple(preview.output.name for preview in LOGIC_NODE.previews_for(
                {**values, "phase_method": phase_method, "frame_mode": "camera_step"}, {})) == (
                    "before_frame", "after_frame", "phase")

        def project():
            values.update(LOGIC_NODE.resolve_defaults(values, {"pulse_template": resource}))
            form.update_projection({"form_spec": project_logic_schema(LOGIC_NODE, workspace_root=str(tmp_path)),
                                    "form_values": values, "workspace_resources": {"pulse_template": resource}})
            app.processEvents()

        project()
        assert form.read_value("before_period") == "" and form.read_value("after_period") == ""
        before = form.widget_for("before_period")
        fields = {field.key: field for field in form.spec.fields}
        assert [(choice.label, choice.value) for choice in fields["before_period"].choices] == [
            ("Select Period", ""), ("Before image", "before"), ("Transport", "gap"), ("Verify image", "after")]
        assert form.read_value("nominal_playback_seconds") == pytest.approx(16/60)
        assert not form.widget_for("nominal_playback_seconds").isEnabled()
        assert form.read_value("phase_method") == "iterative"
        assert form.read_value("frame_mode") == "fixed"
        assert form.widget_for("motion_frames").isEnabled()
        assert not form.widget_for("max_camera_step").isEnabled()
        assert form.widget_for("intensity_error_percent").isEnabled()
        patches = []
        form.draft_changed.connect(patches.append)
        QtTest.QTest.keyClick(before, QtCore.Qt.Key_Down)
        app.processEvents()
        assert patches[-1]["values"]["before_period"] == "before"
        values.update(before_period="before", after_period="after", motion_frames=32)
        project()
        assert form.read_value("nominal_playback_seconds") == pytest.approx(32/60, abs=1e-6)
        frame_mode = form.widget_for("frame_mode")
        QtTest.QTest.keyClick(frame_mode, QtCore.Qt.Key_Down)
        app.processEvents()
        assert patches[-1]["values"] == {"frame_mode": "camera_step"}
        values.update(patches[-1]["values"], max_camera_step=.75)
        project()
        assert not form.widget_for("motion_frames").isEnabled()
        assert form.widget_for("max_camera_step").isEnabled()
        assert form.read_value("motion_frames") == 32
        assert form.read_value("max_camera_step") == .75
        assert form.read_value("nominal_playback_seconds") is None
        assert form.widget_for("nominal_playback_seconds").placeholderText() == "Pending occupancy"
        QtTest.QTest.keyClick(frame_mode, QtCore.Qt.Key_Up)
        app.processEvents()
        values.update(patches[-1]["values"])
        project()
        assert form.read_value("frame_mode") == "fixed"
        assert form.widget_for("motion_frames").isEnabled()
        assert not form.widget_for("max_camera_step").isEnabled()
        assert form.read_value("max_camera_step") == .75
        assert form.read_value("nominal_playback_seconds") == pytest.approx(32/60, abs=1e-6)
        phase_method = form.widget_for("phase_method")
        QtTest.QTest.keyClick(phase_method, QtCore.Qt.Key_Down)
        app.processEvents()
        assert patches[-1]["values"] == {"phase_method": "lpi"}
        values.update(patches[-1]["values"])
        project()
        assert form.widget_for("intensity_error_percent").isEnabled()
        assert form.read_value("intensity_error_percent") == 1.
        QtTest.QTest.keyClick(phase_method, QtCore.Qt.Key_Up)
        app.processEvents()
        values.update(patches[-1]["values"])
        project()
        assert form.widget_for("intensity_error_percent").isEnabled()
        with pytest.raises(ValueError, match="max_camera_step"):
            SLM_REARRANGEMENT_SCHEMA.draft_values({**values, "max_camera_step": 0.})
        resource = replace(resource, value=replace(sequence, periods=tuple(
            replace(period, name="Renamed first image") if period.period_id == "before" else period
            for period in sequence.periods)))
        project()
        assert form.read_value("before_period") == "before"
        values["before_period"] = "deleted_period"
        project()
        assert form.read_value("before_period") == "deleted_period"
        offered = next(field for field in form.spec.fields if field.key == "before_period").choices
        assert offered[-1].label == "Unavailable: deleted_period"
        from zlc_atom.nodes import discover_logic_nodes

        recording = next(node for node in discover_logic_nodes() if node.api_name == "slm_rearrangement_recording")
        recording_form = recording.ui_contributions[0]()
        try:
            recording_values = recording.authoring_schema.draft_values({"before_period": "before"})
            recording_values.update(recording.resolve_defaults(recording_values, {"pulse_template": resource}))
            recording_form.update_projection({
                "form_spec": project_logic_schema(recording, workspace_root=str(tmp_path)),
                "form_values": recording_values, "workspace_resources": {"pulse_template": resource},
            })
            app.processEvents()
            recording_fields = {field.key: field for field in recording_form.spec.fields}
            assert "after_period" not in recording_fields
            assert recording_fields["before_period"].label == "First imaging Period"
            assert recording_form.read_value("before_period") == "before"
            assert recording_form.read_value("recording_frames") == 50
            assert recording_form.widget_for("recording_frames").isEnabled()
            assert tuple(recording_form._forms) == tuple(form._forms)
            assert all(type(recording_form._forms[key]) is type(form._forms[key]) for key in form._forms)
            assert recording_form.read_value("phase_method") == "iterative"
            assert recording_form.read_value("frame_mode") == "fixed"
            assert tuple(output.name for output in recording.outputs_for(recording_values, {})) == camera_outputs
        finally:
            recording_form.close()
            recording_form.deleteLater()
    finally:
        form.close()
        form.deleteLater()
        app.processEvents()
