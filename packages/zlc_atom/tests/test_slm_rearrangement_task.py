"""The one-Pulse camera/SLM flow and its operator-visible timing controls."""

from dataclasses import replace
import json
from pathlib import Path
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

    def prepare_phase_sequence(self, codes, frame_interval_seconds):
        self.trace.append("upload")
        self.preparations.append((np.array(codes), frame_interval_seconds))
        return {"frame_count": len(codes), "prepare_ms": .01}

    def play_phase_sequence(self, stop_requested=None):
        self.trace.append("play")
        self.plays += 1
        if self.play_error is not None:
            raise self.play_error
        codes = self.preparations[-1][0]
        self.apply_phase(phase_from_codes(codes[-1], self.shape_yx))
        return {"played_frames": len(codes), "cancelled": False,
                "final_settle_completed": True, "physical_vblank_observed": False,
                "acknowledgment": "test device", "receipt": self.last_command_receipt}

    def release_phase_sequence(self):
        self.releases += 1
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
    centers = np.asarray(((2., 2.), (4., 2.), (7., 2.), (2., 5.), (4., 5.), (7., 5.)))
    calibration = _calibration_at(centers, shape=slm.shape_yx)
    usable = np.ones(len(centers), bool)
    usable[2] = False
    calibration = replace(calibration, models=(replace(
        calibration.select_model(), usable_sites=usable,
    ),))
    target = np.zeros(slm.shape_yx, np.float32)
    target[centers[:, 1].astype(int), centers[:, 0].astype(int)] = 1
    source_context = _science_context(slm, target=target)
    images = []
    for before in (True, False):
        image = np.zeros(slm.shape_yx, np.uint16)
        image[centers[:, 1].astype(int), centers[:, 0].astype(int)] = 10
        if not before:
            image[2, 4] = 0
        images.append(image)
    camera = VirtualCamera(VirtualCameraConfig(frame_shape_yx=slm.shape_yx),
                           frame_source=lambda exposure: images.pop(0))
    board = _Board(camera, trace)
    plane = SignalDataPlane()
    context = _RunContext(tmp_path / "run", camera, board, trace)
    state = SimpleNamespace(trace=trace, camera=camera, board=board, slm=slm,
                            context=context, plane=plane, available_indices=[], closed=[], images=images)

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
        return {"assigned_source_indices": indices[:n], "assigned_target_indices": np.arange(n),
                "removed_source_indices": indices[n:], "source_indices": indices[:n],
                "target_indices": np.arange(n),
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
        starts = np.asarray(state.prepare_arguments["source_yx"], dtype=np.float64)[planned["source_indices"]]
        points = starts.copy()
        points[:len(planned["assigned_source_indices"])] = state.prepare_arguments["target_yx"][planned["assigned_target_indices"]]
        return {"phase_codes": codes, "converged": True,
                "motion_yx": starts[None] + np.linspace(0,1,kwargs["motion_frames"]+1)[:,None,None]*(points-starts)[None],
                "fraction": np.linspace(0, 1, kwargs["motion_frames"] + 1),
                "support_intensity_ratios": np.full(frames, 1.005),
                "background_intensity_ratios": np.zeros(frames), "timing_ms": {"total": .2}}

    monkeypatch.setattr(task_module, "prepare_rearrangement", prepare)
    monkeypatch.setattr(task_module, "plan_rearrangement", plan)
    monkeypatch.setattr(task_module, "compute_rearrangement", compute)
    state.compute = compute
    state.task = SlmRearrangementTask(
        camera=camera, camera_key="camera", sequencer=board, sequencer_key="pulse",
        slm=slm, slm_key="slm", signal_plane=plane,
        calibration=calibration, calibration_path=tmp_path / "calibration.json",
        science_context=source_context, science_context_path=tmp_path / "science_context.npz",
        target_rows=1 if getattr(request, "param", 4) == 2 else 2, target_columns=2,
        pulse_sequence=_sequence(), pulse_path=tmp_path / "operator.json",
        before_period="before", after_period="after", motion_frames=2,
        save_figure_artifact=_stub_figures,
    )
    try:
        yield state
    finally:
        camera.close()
        board.close()
        plane.close()


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
    assert e.slm.plays == 1 and e.closed == [True]
    assert e.context.terminal_sealed
    summary = json.loads((e.context.run_directory / "summary.json").read_text())
    assert summary["target_sites"] == 4 and summary["judged_target_sites"] == 3
    assert summary["filled_target_sites"] == 2
    assert summary["missing_target_indices"] == [0]
    assert summary["invalid_target_indices"] == [1]
    assert result["target_filling_fraction"] is None
    assert summary["verification_complete"] is False
    assert summary["judged_target_filling_fraction"] == pytest.approx(2/3)
    assert [item["source_ordinal"] for item in summary["frame_records"]] == [0, 1]
    for name in ("before_frame", "after_frame", "phase", "trajectory_2d", "intensity_ratio"):
        archive = e.context.artifacts[name + "_figure"][0]
        assert archive.is_file()
        info, arrays, datasets = read_archive(archive)
        assert datasets
        loaded, recipe = read_figure_plot(info, arrays, datasets, next(iter(datasets)))
        assert loaded is not None and recipe["spec"] is not None
        from zlc_workbench.viewer import describe_archive
        description = describe_archive(info, arrays)
        assert e.task.instance_id in dict(dict(description.tabs)["Logic"])
    with np.load(result["artifact_path"], allow_pickle=False) as data:
        assert data["phase_codes"].shape == (2, 8, 10)
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
    assert result["target_filling_fraction"] is None
    np.testing.assert_array_equal(e.task.points[1], [[1,1]])


@pytest.mark.parametrize("experiment", [2], indirect=True)
def test_hosted_rearrangement_keeps_frozen_vocabulary_shared_records_and_source_target_union(experiment, tmp_path, monkeypatch):
    e = experiment
    e.task._save_figure_artifact = None
    wake = Event()
    host = NodeHost(e.task, e.plane, wake.set, instance_id="slm_rearrangement",
                    kind="task", dataset_output_declarations=LOGIC_NODE.outputs,
                    required_artifacts={item.name: item.contract_id for item in LOGIC_NODE.artifact_outputs},
                    task_name=LOGIC_NODE.api_name)
    publications = []
    names = {declaration.name: host.signal_key(declaration.name) for declaration in LOGIC_NODE.outputs}

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
        assert e.closed == [True]
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
        host.shutdown()


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
    assert json.loads((e.context.run_directory / "summary.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("stopped", [False, True])
def test_failed_or_stopped_play_keeps_completed_before_photo(experiment, monkeypatch, stopped):
    e = experiment
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
    else:
        e.slm.play_error = RuntimeError("injected sequence failure")
        e.slm.cleanup_error = RuntimeError("injected release failure")
        original_play = e.slm.play_phase_sequence

        def partial_play(stop_requested=None):
            # A failed play can leave a later frame confirmed. That current
            # receipt must not retag the earlier source-phase figure.
            e.slm.apply_phase(phase_from_codes(e.slm.preparations[-1][0][0], e.slm.shape_yx))
            return original_play(stop_requested)

        monkeypatch.setattr(e.slm, "play_phase_sequence", partial_play)
        with pytest.raises(RuntimeError, match="injected sequence failure"):
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
    if not stopped:
        assert phase_device["command_revision"] < summary["device_snapshots"]["slm"]["command_revision"]
        assert not np.array_equal(e.slm.last_commanded_phase, phase.block.values[0, 0])
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
        patches = []
        form.draft_changed.connect(patches.append)
        QtTest.QTest.keyClick(before, QtCore.Qt.Key_Down)
        app.processEvents()
        assert patches[-1]["values"]["before_period"] == "before"
        values.update(before_period="before", after_period="after", motion_frames=32)
        project()
        assert form.read_value("nominal_playback_seconds") == pytest.approx(32/60, abs=1e-6)
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
    finally:
        form.close()
        form.deleteLater()
        app.processEvents()
