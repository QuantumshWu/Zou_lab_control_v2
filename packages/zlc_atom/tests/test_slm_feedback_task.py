import inspect
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.devices.camera import CameraWorkingPoint
from zlc_atom.devices.simulation.camera import VirtualCamera, VirtualCameraConfig
from zlc_atom.devices.slm import canonical_phase
from zlc_atom.devices.slm.solver import load_phase
from zlc_atom.devices.slm.solver import preset_grid
from zlc_atom.install import create_installation
from zlc_atom.nodes import discover_logic_nodes
from zlc_atom.nodes._framework.descriptor import DeviceAccess
from zlc_atom.nodes.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.calibration.pulse import resolve_pulse
from zlc_atom.nodes.slm_feedback import task as feedback_module
from zlc_atom.nodes.slm_feedback.task import SlmFeedbackTask, _updated_target
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE


class _Slm:
    identity = "test-slm"

    def __init__(self, shape: tuple[int, int], incoming: float = 0.1) -> None:
        self.shape_yx = shape
        self._phase = canonical_phase(np.full(shape, incoming), shape)
        self.commands: list[np.ndarray] = []

    @property
    def last_commanded_phase(self) -> np.ndarray:
        return self._phase

    def apply_phase(self, radians: object) -> np.ndarray:
        self._phase = canonical_phase(radians, self.shape_yx)
        self.commands.append(self._phase)
        return self._phase

    def close(self) -> None:
        return None


class _Context:
    generation = "slm-feedback-test"

    def __init__(self, cancelled: bool = False) -> None:
        self.cancelled = cancelled
        self.terminal_sealed = False
        self.progress: list[tuple] = []

    def cancel_requested(self) -> bool:
        return self.cancelled

    def seal_terminal(self) -> None:
        if self.cancelled:
            raise RuntimeError("SLM feedback was cancelled")
        self.terminal_sealed = True

    def report_progress(self, *args, **kwargs) -> None:
        self.progress.append((args, kwargs))


def _wait_host(host: NodeHost, wake: Event):
    deadline = time.monotonic() + 2.0
    while not host.terminal and time.monotonic() < deadline:
        host.poll()
        wake.wait(0.01)
        wake.clear()
    observation = host.poll()
    assert observation.terminal
    return observation


def _calibration(
    shape: tuple[int, int] = (5, 7), *, camera_id: str | None = None
) -> TrapCalibration:
    centers = np.asarray(
        [(column, row) for row in range(5) for column in range(7)], dtype=float
    )
    ids = tuple(f"site_{index:04d}" for index in range(35))
    site_map = SiteMap(ids, centers, np.ones(35, bool), np.ones(35))
    model = ReadoutModel(
        ids,
        np.full(35, 5.0),
        np.zeros(35),
        np.full(35, 10.0),
        np.ones(35, bool),
        np.ones(35),
        kind=ReadoutModelKind.PER_SITE_PSF,
        integration_half_width=0,
        reducer=None,
        psf_weights=np.ones((35, 1, 1)),
        psf_boxes=np.asarray(
            [(column, row, 1, 1) for row in range(5) for column in range(7)]
        ),
        background="none",
        psf_padding=1,
    )
    return TrapCalibration(
        site_map,
        (model,),
        ReadoutModelKind.PER_SITE_PSF,
        # Sensor integration and authored probe gate are deliberately not the
        # same fact: the shipped pulse below keeps its 5 ms readout window.
        FrameContract(shape, exposure_seconds=0.020, camera_id=camera_id),
    )


def _grid_target(shape: tuple[int, int]) -> np.ndarray:
    target = np.zeros(shape, dtype=np.float32)
    rows = np.linspace(1, shape[0] - 2, 5, dtype=int)
    columns = np.linspace(1, shape[1] - 2, 7, dtype=int)
    target[np.ix_(rows, columns)] = 1.0
    return target


def _task(
    tmp_path: Path,
    *,
    slm: _Slm,
    camera: object,
    sequencer: object,
    plane: SignalDataPlane,
    target: np.ndarray,
    shots: int = 10,
    validation_shots: int = 20,
    updates: int = 3,
) -> SlmFeedbackTask:
    return SlmFeedbackTask(
        camera=camera,
        camera_key="qcmos",
        sequencer=sequencer,
        sequencer_key="sequencer",
        slm=slm,
        slm_key="slm",
        signal_plane=plane,
        calibration=_calibration(),
        calibration_path=tmp_path / "calibration.json",
        target=target,
        target_path=tmp_path / "target.json",
        pulse_sequence=IMAGING_PULSE_RESOURCE.value,
        pulse_path=IMAGING_PULSE_RESOURCE.path,
        shots_per_candidate=shots,
        validation_shots=validation_shots,
        max_updates=updates,
        artifact_directory=tmp_path,
    )


def test_descriptor_and_direct_update_keep_the_plugin_boundary() -> None:
    descriptor = {item.api_name: item for item in discover_logic_nodes()}[
        "slm_feedback"
    ]
    assert tuple(item.name for item in descriptor.input_specs) == (
        "calibration_path",
        "target_path",
    )
    assert tuple(item.contract_id for item in descriptor.input_specs) == (
        "calibration.readout.v1",
        "zlc.slm.target.v1",
    )
    assert tuple(item.field_name for item in descriptor.workspace_resources) == (
        "pulse_template",
    )
    assert tuple(
        (item.name, item.contract_id) for item in descriptor.artifact_outputs
    ) == (("artifact_path", "zlc.slm.phase.v1"),)
    assert tuple(
        (item.capability_token, item.access) for item in descriptor.device_requirements
    ) == (
        ("camera.adapter", DeviceAccess.EXCLUSIVE),
        ("sequencer.streamer", DeviceAccess.EXCLUSIVE),
        ("slm.phase", DeviceAccess.EXCLUSIVE),
    )
    source = inspect.getsource(feedback_module)
    assert "SimulationWorld" not in source and "devices.simulation" not in source

    target = _grid_target((17, 23))
    rows, columns = np.nonzero(target)
    fluorescence = np.linspace(0.6, 1.4, 35)
    updated = _updated_target(target, fluorescence, rows, columns)
    assert updated[rows[0], columns[0]] > updated[rows[-1], columns[-1]]
    np.testing.assert_allclose(np.sum(updated), np.sum(target), rtol=1e-6)


def test_measurement_streams_bounded_exact_grouped_qcmos_publications(
    tmp_path: Path, monkeypatch
) -> None:
    fluorescence = np.arange(6, 11, dtype=np.uint16).repeat(7)

    def frame_source(ordinal: int, exposure: float) -> np.ndarray:
        del exposure
        image = np.zeros((5, 7), dtype="<u2")
        if ordinal % 3 == 1:
            image[:] = fluorescence.reshape(5, 7)
        return image

    camera = VirtualCamera(
        VirtualCameraConfig(frame_shape_yx=(5, 7), exposure_seconds=0.005),
        frame_source=frame_source,
    )

    class Sequencer:
        def __init__(self) -> None:
            self.loaded = None
            self.fires: list[bool] = []
            self.scan_sweeps = 0
            self.scan_sweeps_history: list[int] = []

        def load(self, program, *, source=None) -> None:
            self.loaded = (program, source)

        def write_scan_table(self, rows, *, sweeps=1) -> None:
            assert tuple(tuple(row) for row in rows) == ((),)
            self.scan_sweeps = int(sweeps)
            self.scan_sweeps_history.append(self.scan_sweeps)

        def fire(self, *, forever=False) -> None:
            self.fires.append(bool(forever))
            camera.trigger(self.scan_sweeps * 3)

        def wait_done(self, timeout=None):
            del timeout
            return SimpleNamespace(fault=None)

        def safe(self):
            return None

    sequencer = Sequencer()
    armed_buffer_sizes: list[int] = []
    original_arm = camera.arm

    def record_arm(*args, **kwargs):
        armed_buffer_sizes.append(int(kwargs["buffer_frame_count"]))
        return original_arm(*args, **kwargs)

    monkeypatch.setattr(camera, "arm", record_arm)
    monkeypatch.setattr(feedback_module, "_SHOT_CHUNK", 4)
    plane = SignalDataPlane()
    slm = _Slm((5, 7))
    task = _task(
        tmp_path,
        slm=slm,
        camera=camera,
        sequencer=sequencer,
        plane=plane,
        target=np.ones((5, 7), dtype=np.float32),
    )
    try:
        from zlc_atom.install import create_installation

        installation = create_installation("virtual")
        try:
            pulse = resolve_pulse(
                IMAGING_PULSE_RESOURCE.value,
                path=IMAGING_PULSE_RESOURCE.path,
                board=installation.device("sequencer").describe(),
                api_values={
                    "reference_probe_duration_before": 0.02,
                    "readout_probe_duration": 0.005,
                    "reference_probe_duration_after": 0.02,
                },
            )
        finally:
            installation.close()
        original_evaluate = task._occupancy.evaluate

        def one_missing_site(value):
            outputs = original_evaluate(value)
            valid = np.array(outputs["valid"].snapshot.block.values, copy=True)
            valid[0, 1, 0] = False
            outputs["valid"] = SimpleNamespace(
                snapshot=SimpleNamespace(block=SimpleNamespace(values=valid))
            )
            return outputs

        monkeypatch.setattr(task._occupancy, "evaluate", one_missing_site)
        measured, error, saturated, missing = task._measure(
            pulse, _Context(), 0, shots=10
        )
        assert np.isnan(measured[0]) and np.isnan(error[0])
        np.testing.assert_allclose(measured[1:], fluorescence[1:] / 10.0)
        np.testing.assert_allclose(error[1:], 0.0, atol=1e-15)
        assert not saturated
        assert missing == (0,)
        assert sequencer.fires == [False, False, False]
        assert sequencer.scan_sweeps_history == [4, 4, 2]
        assert armed_buffer_sizes == [12, 12, 6]
        assert sequencer.loaded[1] is pulse.sequence
        assert next(
            period.duration
            for period in pulse.sequence.periods
            if period.period_id == "short"
        ) == 0.005
        assert not camera.capture_state()
    finally:
        plane.close()
        camera.close()


def test_direct_fluorescence_does_not_rethreshold_the_same_noisy_counts(
    tmp_path: Path, monkeypatch
) -> None:
    """Dark/bright-normalized intensity is the feedback observable itself."""

    fluorescence = np.arange(6, 11, dtype=np.uint16).repeat(7)

    def frame_source(ordinal: int, exposure: float) -> np.ndarray:
        del exposure
        image = np.zeros((5, 7), dtype="<u2")
        if ordinal % 3 == 1:
            image[:] = fluorescence.reshape(5, 7)
        return image

    camera = VirtualCamera(
        VirtualCameraConfig(frame_shape_yx=(5, 7), exposure_seconds=0.005),
        frame_source=frame_source,
    )

    class Sequencer:
        def wait_done(self, timeout=None):
            del timeout
            return SimpleNamespace(fault=None)

        def load(self, program, *, source=None) -> None:
            self.loaded = (program, source)

        def write_scan_table(self, rows, *, sweeps=1) -> None:
            assert tuple(tuple(row) for row in rows) == ((),)
            self.sweeps = int(sweeps)

        def fire(self, *, forever=False) -> None:
            assert not forever
            camera.trigger(self.sweeps * 3)

        def safe(self):
            return None

    sequencer = Sequencer()
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=_Slm((5, 7)),
        camera=camera,
        sequencer=sequencer,
        plane=plane,
        target=np.ones((5, 7), dtype=np.float32),
    )
    try:
        installation = create_installation("virtual")
        try:
            pulse = resolve_pulse(
                IMAGING_PULSE_RESOURCE.value,
                path=IMAGING_PULSE_RESOURCE.path,
                board=installation.device("sequencer").describe(),
                api_values={},
            )
        finally:
            installation.close()

        original_evaluate = task._occupancy.evaluate

        def deliberately_false_classifier(value):
            outputs = original_evaluate(value)
            occupied = np.zeros_like(
                outputs["occupied"].snapshot.block.values, dtype=bool
            )
            outputs["occupied"] = SimpleNamespace(
                snapshot=SimpleNamespace(block=SimpleNamespace(values=occupied))
            )
            return outputs

        monkeypatch.setattr(task._occupancy, "evaluate", deliberately_false_classifier)
        measured, error, saturated, missing = task._measure(
            pulse, _Context(), 0, shots=10
        )
        np.testing.assert_allclose(measured, fluorescence / 10.0)
        np.testing.assert_allclose(error, 0.0, atol=1e-15)
        assert not saturated and not missing
    finally:
        plane.close()
        camera.close()


def test_virtual_feedback_runs_repeated_real_qcmos_candidates_and_restores(
    tmp_path: Path,
) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {item.api_name: item for item in discover_logic_nodes()}
    camera = installation.device("camera")
    sequencer = installation.device("sequencer")
    slm = installation.device("slm")
    incoming = np.array(slm.last_commanded_phase, copy=True)
    try:
        calibration_node = descriptors["calibration"].instantiate(
            camera=camera,
            camera_key="camera",
            sequencer=sequencer,
            sequencer_key="sequencer",
            signal_plane=plane,
            pulse_resource=IMAGING_PULSE_RESOURCE,
            artifact_directory=tmp_path,
            repeats=30,
        )
        calibration = calibration_node.run().calibration
        context = _Context()
        task = SlmFeedbackTask(
            camera=camera,
            camera_key="camera",
            sequencer=sequencer,
            sequencer_key="sequencer",
            slm=slm,
            slm_key="slm",
            signal_plane=plane,
            calibration=calibration,
            calibration_path=calibration_node.result.artifact_path,
            target=preset_grid(slm.shape_yx, (5, 7)),
            target_path=tmp_path / "target.json",
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            shots_per_candidate=40,
            validation_shots=80,
            max_updates=2,
            artifact_directory=tmp_path,
        )
        with pytest.raises(RuntimeError, match="did not reach 1.01"):
            task.execute(context)
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        assert not camera.capture_state()
        ratio_messages = [
            args[0]
            for args, _kwargs in context.progress
            if args and str(args[0]).startswith("qCMOS fluorescence ratio")
        ]
        assert len(ratio_messages) == 2
        assert not tuple(tmp_path.glob("slm_feedback*.npz"))
    finally:
        plane.close()
        installation.close()


def test_success_reapplies_and_saves_the_independently_validated_best(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23))
    plane = SignalDataPlane()
    phases = iter(
        [
            np.full(slm.shape_yx, 0.25, dtype=np.float32),
            np.full(slm.shape_yx, 0.75, dtype=np.float32),
        ]
    )
    batches = iter(
        [
            np.linspace(0.8, 1.2, 35),
            np.ones(35),
            np.ones(35),
        ]
    )
    resolved_api_values: list[dict[str, float]] = []

    def resolve_without_reauthoring(*args, **kwargs):
        resolved_api_values.append(dict(kwargs["api_values"]))
        return SimpleNamespace(program=object())

    monkeypatch.setattr(feedback_module, "resolve_pulse", resolve_without_reauthoring)
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (next(phases), {"method": "test"}),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration, *, shots=None: (
            requested_shots.append(self.shots if shots is None else int(shots)),
            next(batches),
            np.zeros(35),
            (),
            (),
        )[1:],
    )
    requested_shots: list[int] = []
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
    )
    try:
        result = task.execute(_Context())
        saved, metadata = load_phase(result["artifact_path"])
        np.testing.assert_array_equal(saved, slm.last_commanded_phase)
        np.testing.assert_array_equal(saved, np.full(slm.shape_yx, 0.75, np.float32))
        assert metadata["best"]["validation"]["uniformity_ratio"] == 1.0
        assert requested_shots == [10, 10, 20]
        assert resolved_api_values == [{}]
        assert result["updates"] == 2
        assert np.array_equal(slm.commands[-1], saved)
    finally:
        plane.close()


def test_a_failed_validation_does_not_block_the_next_threshold_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23))
    plane = SignalDataPlane()
    first = np.full(slm.shape_yx, 0.25, dtype=np.float32)
    next_candidate = np.full(slm.shape_yx, 0.50, dtype=np.float32)
    phases = iter((first, next_candidate))
    measurements = iter(
        (
            (np.linspace(1.0, 1.009, 35), np.zeros(35), (), ()),
            (np.ones(35), np.full(35, 0.02), (), ()),
            (np.linspace(1.0, 1.0095, 35), np.zeros(35), (), ()),
            (np.ones(35), np.zeros(35), (), ()),
        )
    )
    requested_shots: list[int] = []
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (next(phases), {"method": "test"}),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration, *, shots=None: (
            requested_shots.append(self.shots if shots is None else int(shots)),
            next(measurements),
        )[1],
    )
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        updates=3,
    )
    try:
        result = task.execute(_Context())
        saved, _metadata = load_phase(result["artifact_path"])
        np.testing.assert_array_equal(saved, next_candidate)
        # The first candidate has a good point estimate but insufficient
        # statistics.  The next threshold candidate is independently
        # validated and becomes the terminal command.
        assert requested_shots == [10, 20, 10, 20]
    finally:
        plane.close()


def test_validation_refuses_a_point_estimate_with_wide_qcmos_uncertainty(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    plane = SignalDataPlane()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (
            np.full(slm.shape_yx, 0.5, dtype=np.float32),
            {"method": "test"},
        ),
    )
    batches = iter(
        (
            (np.ones(35), np.zeros(35), (), ()),
            (np.ones(35), np.full(35, 0.02), (), ()),
        )
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration, *, shots=None: next(batches),
    )
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        updates=1,
    )
    try:
        try:
            task.execute(_Context())
        except RuntimeError as error:
            assert "did not reach 1.01" in str(error)
        else:
            raise AssertionError("a noisy point estimate was accepted as 1% evidence")
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        assert not tuple(tmp_path.glob("slm_feedback*.npz"))
    finally:
        plane.close()


def test_stop_at_the_terminal_gate_restores_incoming_and_writes_no_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    plane = SignalDataPlane()
    wake = Event()
    validation_entered = Event()
    release_validation = Event()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (
            np.full(slm.shape_yx, 0.5, dtype=np.float32),
            {"method": "test"},
        ),
    )
    calls = 0

    def measure(self, pulse, run_context, iteration, *, shots=None):
        nonlocal calls
        calls += 1
        if shots is not None:
            validation_entered.set()
            assert release_validation.wait(2.0)
        return np.ones(35), np.zeros(35), (), ()

    monkeypatch.setattr(SlmFeedbackTask, "_measure", measure)
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        updates=1,
    )
    host = NodeHost(task, plane, wake.set)
    try:
        host.start()
        assert validation_entered.wait(2.0)
        # Stop wins the same lock immediately before the final apply/save
        # commit, after validation has already produced a passing result.
        host.cancel("before terminal commit")
        release_validation.set()
        observation = _wait_host(host, wake)
        assert observation.phase == "cancelled"
        assert not host.final_result_resolved
        assert calls == 2
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        assert not tuple(tmp_path.glob("slm_feedback*.npz"))
    finally:
        release_validation.set()
        if not host.terminal:
            _wait_host(host, wake)
        host.shutdown()
        plane.close()


def test_stop_after_terminal_commit_keeps_host_success_and_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    best = np.full(slm.shape_yx, 0.5, dtype=np.float32)
    plane = SignalDataPlane()
    wake = Event()
    save_entered = Event()
    release_save = Event()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (best, {"method": "test"}),
    )
    batches = iter(
        (
            (np.ones(35), np.zeros(35), (), ()),
            (np.ones(35), np.zeros(35), (), ()),
        )
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration, *, shots=None: next(batches),
    )
    original_save = feedback_module.save_phase

    def blocking_save(*args, **kwargs):
        save_entered.set()
        assert release_save.wait(2.0)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(feedback_module, "save_phase", blocking_save)
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        updates=1,
    )
    host = NodeHost(task, plane, wake.set)
    try:
        host.start()
        assert save_entered.wait(2.0)
        host.cancel("after terminal commit")
        release_save.set()
        observation = _wait_host(host, wake)
        assert observation.phase == "done"
        result = host.final_result
        assert isinstance(result, dict)
        saved, _metadata = load_phase(result["artifact_path"])
        np.testing.assert_array_equal(saved, best)
        np.testing.assert_array_equal(slm.last_commanded_phase, best)
    finally:
        release_save.set()
        if not host.terminal:
            _wait_host(host, wake)
        host.shutdown()
        plane.close()


@pytest.mark.parametrize("terminal_operation", ("apply", "save"))
def test_terminal_apply_or_save_failure_restores_incoming_and_fails_host(
    tmp_path: Path, monkeypatch, terminal_operation: str
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    best = np.full(slm.shape_yx, 0.5, dtype=np.float32)
    plane = SignalDataPlane()
    wake = Event()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (best, {"method": "test"}),
    )
    batches = iter(
        (
            (np.ones(35), np.zeros(35), (), ()),
            (np.ones(35), np.zeros(35), (), ()),
        )
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration, *, shots=None: next(batches),
    )
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        updates=1,
    )
    if terminal_operation == "apply":
        original_apply = task._apply_exact
        apply_calls = 0

        def fail_terminal_apply(phase):
            nonlocal apply_calls
            apply_calls += 1
            if apply_calls == 3:
                raise OSError("terminal apply failed")
            return original_apply(phase)

        monkeypatch.setattr(task, "_apply_exact", fail_terminal_apply)
    else:
        monkeypatch.setattr(
            feedback_module,
            "save_phase",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("terminal save failed")
            ),
        )
    host = NodeHost(task, plane, wake.set)
    try:
        host.start()
        observation = _wait_host(host, wake)
        assert observation.phase == "failed"
        assert observation.error == f"OSError: terminal {terminal_operation} failed"
        assert not host.final_result_resolved
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        assert not tuple(tmp_path.glob("slm_feedback*.npz"))
    finally:
        if not host.terminal:
            _wait_host(host, wake)
        host.shutdown()
        plane.close()


def test_missing_history_serializes_as_null_before_a_later_success(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    plane = SignalDataPlane()
    phase = np.full(slm.shape_yx, 0.25, dtype=np.float32)
    first = np.ones(35)
    first[4] = np.nan
    batches = iter(
        (
            (first, np.full(35, np.nan), (), (4,)),
            (np.ones(35), np.zeros(35), (), ()),
            (np.ones(35), np.zeros(35), (), ()),
        )
    )
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (phase, {"method": "test"}),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration, *, shots=None: next(batches),
    )
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        updates=2,
    )
    try:
        result = task.execute(_Context())
        saved, metadata = load_phase(result["artifact_path"])
        assert metadata["history"][0]["fluorescence"][4] is None
        assert metadata["history"][0]["standard_error"][4] is None
        np.testing.assert_array_equal(saved, slm.last_commanded_phase)
        np.testing.assert_array_equal(saved, phase)
    finally:
        plane.close()


def test_cancel_or_failure_restores_the_incoming_phase(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    plane = SignalDataPlane()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (
            np.full(slm.shape_yx, 0.5, dtype=np.float32),
            {"method": "test"},
        ),
    )

    def cancelled(self, pulse, context, iteration, *, shots=None):
        raise RuntimeError("SLM feedback was cancelled")

    monkeypatch.setattr(SlmFeedbackTask, "_measure", cancelled)
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
    )
    try:
        try:
            task.execute(_Context(cancelled=True))
        except RuntimeError as error:
            assert "cancelled" in str(error)
        else:
            raise AssertionError("cancelled feedback unexpectedly succeeded")
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        assert np.array_equal(slm.commands[-1], incoming)
        assert not tuple(tmp_path.glob("slm_feedback*.npz"))
    finally:
        plane.close()
