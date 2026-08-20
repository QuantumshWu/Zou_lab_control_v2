import inspect
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from zlc_plot import Reduction
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.devices.camera import CameraWorkingPoint
from zlc_atom.devices.simulation.camera import VirtualCamera, VirtualCameraConfig
from zlc_atom.devices.slm import canonical_phase
from zlc_atom.devices.slm.solver import load_phase
from zlc_atom.devices.slm.solver import preset_grid
from zlc_atom.install import create_installation
from zlc_atom.nodes import discover_logic_nodes
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
    instance_id = "slm_feedback"
    generation = "slm-feedback-test"

    def __init__(self, cancelled: bool = False) -> None:
        self.cancelled = cancelled
        self.terminal_sealed = False
        self.progress: list[tuple] = []
        self.commits: list[dict[str, object]] = []

    def cancel_requested(self) -> bool:
        return self.cancelled

    def seal_terminal(
        self,
        *,
        accept_stop: bool = False,
        partial: bool = False,
    ) -> None:
        del partial
        if self.cancelled and not accept_stop:
            raise RuntimeError("SLM feedback was cancelled")
        self.terminal_sealed = True

    def report_progress(self, *args, **kwargs) -> None:
        self.progress.append((args, kwargs))

    def commit_live(self, outputs):
        committed = dict(outputs)
        self.commits.append(committed)
        return committed


def _wait_host(host: NodeHost, wake: Event):
    deadline = time.monotonic() + 2.0
    while not host.terminal and time.monotonic() < deadline:
        host.poll()
        wake.wait(0.01)
        wake.clear()
    observation = host.poll()
    assert observation.terminal
    return observation


def _task_host(
    task: SlmFeedbackTask,
    plane: SignalDataPlane,
    wake: Event,
) -> NodeHost:
    return NodeHost(
        task,
        plane,
        wake.set,
        instance_id=task.instance_id,
        kind="task",
        dataset_output_declarations=task.dataset_output_declarations,
        required_artifact_names=("artifact_path",),
    )


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
        kind=ReadoutModelKind.BOX,
        integration_half_width=0,
        reducer="mean",
    )
    return TrapCalibration(
        site_map,
        (model,),
        ReadoutModelKind.BOX,
        # Sensor integration and authored probe gate are deliberately not the
        # same fact: the shipped pulse below keeps its 5 ms readout window.
        FrameContract(shape, exposure_seconds=0.020, camera_id=camera_id),
    )


def _calibration_at(
    centers_xy: np.ndarray, *, shape: tuple[int, int] = (64, 64)
) -> TrapCalibration:
    centers = np.asarray(centers_xy, dtype=float)
    count = len(centers)
    ids = tuple(f"site_{index:04d}" for index in range(count))
    site_map = SiteMap(ids, centers, np.ones(count, bool), np.ones(count))
    model = ReadoutModel(
        ids,
        np.full(count, 5.0),
        np.zeros(count),
        np.full(count, 10.0),
        np.ones(count, bool),
        np.ones(count),
        kind=ReadoutModelKind.BOX,
        integration_half_width=0,
        reducer="mean",
    )
    return TrapCalibration(
        site_map,
        (model,),
        ReadoutModelKind.BOX,
        FrameContract(shape, exposure_seconds=0.020),
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
    calibration: TrapCalibration | None = None,
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
        calibration=_calibration() if calibration is None else calibration,
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
    assert tuple((item.name, item.contract_id) for item in descriptor.outputs) == (
        ("candidate_phase", "slm-feedback.candidate-phase.v1"),
        ("uniformity_history", "slm-feedback.uniformity-history.v1"),
    )
    assert tuple(
        (item.output.name, item.plot_kind, item.producer)
        for item in descriptor.node_previews
    ) == (
        ("frames", "image", "camera"),
        ("candidate_phase", "image", ""),
        ("uniformity_history", "curve", ""),
    )
    camera_preview = descriptor.node_previews[0]
    assert camera_preview.semantic["fate:frame"] == feedback_module.READOUT_FRAME_COORDINATE
    assert camera_preview.semantic == {
        "fate:frame": feedback_module.READOUT_FRAME_COORDINATE,
        "fate:repeat": "reduce",
        "reduction": Reduction.MEAN,
    }
    assert tuple(
        item.capability_token for item in descriptor.device_requirements
    ) == (
        "camera.adapter",
        "sequencer.streamer",
        "slm.phase",
    )
    source = inspect.getsource(feedback_module)
    assert "SimulationWorld" not in source and "devices.simulation" not in source

    target = _grid_target((17, 23))
    rows, columns = np.nonzero(target)
    fluorescence = np.linspace(0.6, 1.4, 35)
    updated = _updated_target(target, fluorescence, rows, columns)
    assert updated[rows[0], columns[0]] > updated[rows[-1], columns[-1]]
    np.testing.assert_allclose(np.sum(updated), np.sum(target), rtol=1e-6)


def test_arbitrary_sparse_geometry_matches_calibration_sites_before_updating_target(
    tmp_path: Path, monkeypatch
) -> None:
    target = np.zeros((17, 23), dtype=np.float32)
    rows = np.asarray([2, 2, 5, 7, 10, 10])
    columns = np.asarray([3, 9, 5, 12, 4, 15])
    target[rows, columns] = 1.0
    nominal_centers = np.column_stack(
        (10.0 + 2.5 * (columns - 3), 12.0 + 3.0 * (rows - 2))
    )
    nominal_centers += np.asarray(
        [
            [0.0, 0.0],
            [0.10, -0.08],
            [-0.12, 0.05],
            [0.06, 0.11],
            [-0.04, -0.09],
            [0.0, 0.0],
        ]
    )
    calibration_order = np.asarray([3, 0, 5, 1, 4, 2])
    calibration = _calibration_at(nominal_centers[calibration_order])
    target_fluorescence = np.asarray([0.7, 1.4, 0.9, 1.2, 0.8, 1.1])
    measured = iter(
        (
            target_fluorescence[calibration_order],
            np.ones(len(rows)),
            np.ones(len(rows)),
        )
    )
    solved_targets: list[np.ndarray] = []

    def solve(candidate, **_kwargs):
        solved_targets.append(np.array(candidate, copy=True))
        return np.full(target.shape, 0.25 * len(solved_targets), dtype=np.float32), {}

    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(feedback_module, "solve_phase", solve)
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration, *, shots=None: (
            next(measured),
            np.zeros(len(rows)),
            (),
            (),
        ),
    )
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=_Slm(target.shape),
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=target,
        calibration=calibration,
        updates=2,
    )
    try:
        result = task.execute(_Context())
        expected = _updated_target(target, target_fluorescence, rows, columns)
        np.testing.assert_allclose(solved_targets[1], expected)
        _saved, metadata = load_phase(result["artifact_path"])
        np.testing.assert_allclose(
            metadata["history"][0]["fluorescence"],
            target_fluorescence[calibration_order],
        )
    finally:
        plane.close()


def test_uniformity_history_is_one_latest_curve_paired_with_candidate_phase(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23))
    plane = SignalDataPlane()
    context = _Context()
    measurements = iter(
        (
            np.concatenate(([4.0], np.ones(34))),
            np.concatenate(([2.0], np.ones(34))),
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
        lambda *args, **kwargs: (
            np.full(slm.shape_yx, 0.5, dtype=np.float32),
            {"method": "test"},
        ),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, run_context, iteration, *, shots=None: (
            next(measurements),
            np.zeros(35),
            (),
            (),
        ),
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
        with pytest.raises(RuntimeError, match="did not reach 1.01"):
            task.execute(context)
        output = context.commits[-1]["uniformity_history"]
        np.testing.assert_allclose(output.snapshot.block.values[0, :, 0], (4.0, 2.0))
        assert len(context.commits) == 2
        assert (
            context.commits[0]["uniformity_history"].snapshot.block.schema
            is context.commits[1]["uniformity_history"].snapshot.block.schema
        )
        assert set(context.commits[-1]) == {
            "candidate_phase",
            "uniformity_history",
        }
        column = output.snapshot.block.schema.point_table.columns[0]
        assert column.name == "candidate"
        assert tuple(column.values) == (1.0, 2.0)
    finally:
        plane.close()


@pytest.mark.parametrize("failure", ("count", "distortion", "ambiguous"))
def test_sparse_geometry_refuses_non_bijective_calibration(
    tmp_path: Path, failure: str
) -> None:
    target = np.zeros((17, 23), dtype=np.float32)
    rows = np.asarray([2, 2, 8, 8])
    columns = np.asarray([3, 12, 5, 15])
    target[rows, columns] = 1.0
    centers = np.column_stack((10.0 + 2.0 * columns, 15.0 + 3.0 * rows))
    if failure == "count":
        centers = centers[:-1]
    elif failure == "distortion":
        centers[2] += (20.0, 15.0)
    else:
        centers[1] = centers[0]
    plane = SignalDataPlane()
    try:
        with pytest.raises(ValueError, match="count|unmatched|ambiguous"):
            _task(
                tmp_path,
                slm=_Slm(target.shape),
                camera=object(),
                sequencer=object(),
                plane=plane,
                target=target,
                calibration=_calibration_at(centers),
            )
    finally:
        plane.close()


def test_sparse_geometry_refuses_a_large_global_shear(
    tmp_path: Path,
) -> None:
    target = np.zeros((19, 23), dtype=np.float32)
    rows = np.asarray([2, 2, 8, 8, 14, 14])
    columns = np.asarray([3, 15, 5, 17, 4, 16])
    target[rows, columns] = 1.0
    centers = np.column_stack(
        (
            5.0 + 1.5 * columns + rows,
            8.0 + 2.0 * rows,
        )
    )
    plane = SignalDataPlane()
    try:
        with pytest.raises(ValueError, match="apparatus orientation"):
            _task(
                tmp_path,
                slm=_Slm(target.shape),
                camera=object(),
                sequencer=object(),
                plane=plane,
                target=target,
                calibration=_calibration_at(centers),
            )
    finally:
        plane.close()


@pytest.mark.parametrize("orientation", ("row", "column"))
def test_axis_aligned_sparse_geometry_refuses_a_large_cross_axis_tilt(
    tmp_path: Path, orientation: str
) -> None:
    target = np.zeros((23, 25), dtype=np.float32)
    primary = np.asarray([3, 8, 14, 19])
    if orientation == "row":
        rows, columns = np.full(4, 6), primary
        centers = np.column_stack((2.0 * primary + 4.0, primary + 9.0))
    else:
        rows, columns = primary, np.full(4, 9)
        centers = np.column_stack((primary + 7.0, 2.0 * primary + 5.0))
    target[rows, columns] = 1.0
    plane = SignalDataPlane()
    try:
        with pytest.raises(ValueError, match="apparatus orientation"):
            _task(
                tmp_path,
                slm=_Slm(target.shape),
                camera=object(),
                sequencer=object(),
                plane=plane,
                target=target,
                calibration=_calibration_at(centers),
            )
    finally:
        plane.close()


@pytest.mark.parametrize(
    ("rows", "columns", "centers"),
    (
        ([7], [11], [[23.0, 31.0]]),
        (
            [6, 6, 6, 6],
            [3, 8, 14, 19],
            [[30.0, 22.1], [8.0, 22.0], [40.0, 21.9], [18.0, 22.0]],
        ),
        (
            [2, 7, 13, 18],
            [9, 9, 9, 9],
            [[27.1, 43.0], [27.0, 10.0], [26.9, 25.0], [27.0, 58.0]],
        ),
    ),
)
def test_sparse_geometry_accepts_single_site_and_collinear_support(
    tmp_path: Path,
    rows: list[int],
    columns: list[int],
    centers: list[list[float]],
) -> None:
    target = np.zeros((23, 25), dtype=np.float32)
    target[np.asarray(rows), np.asarray(columns)] = 1.0
    plane = SignalDataPlane()
    try:
        task = _task(
            tmp_path,
            slm=_Slm(target.shape),
            camera=object(),
            sequencer=object(),
            plane=plane,
            target=target,
            calibration=_calibration_at(np.asarray(centers)),
        )
        assert len(task._rows) == len(rows)
    finally:
        plane.close()


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
        measured, error, saturated, missing = task._measure(
            pulse, _Context(), 0, shots=10
        )
        np.testing.assert_allclose(measured, fluorescence)
        np.testing.assert_allclose(error, 0.0, atol=1e-15)
        assert not saturated
        assert not missing
        assert sequencer.fires == [False]
        assert sequencer.scan_sweeps_history == [10]
        assert armed_buffer_sizes == [30]
        signal = "@logic/slm_feedback/camera/frames"
        raw = plane.current_dataset(signal)
        assert raw.block.values.shape == (10, 3, 5, 7)
        np.testing.assert_allclose(
            raw.block.values[:, 1],
            np.broadcast_to(fluorescence.reshape(5, 7), (10, 5, 7)),
        )

        from zlc_plot import PlotKind, PlotSession
        from zlc_plot._kinds import default_spec
        from zlc_plot.semantics import composed_spec

        descriptor = {item.api_name: item for item in discover_logic_nodes()}[
            "slm_feedback"
        ]
        preview = descriptor.node_previews[0]
        spec = default_spec(raw.block.schema, PlotKind.IMAGE)
        assert spec is not None
        spec = composed_spec(raw.block.schema, spec, preview.semantic)
        session = PlotSession(raw, spec)
        try:
            np.testing.assert_allclose(
                np.asarray(session._payload.z.canonical),
                raw.block.values[:, 1].mean(axis=0),
            )
        finally:
            session.close()
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


def test_feedback_averages_dark_subtracted_brightness_over_all_shots(
    tmp_path: Path,
) -> None:
    """A brightness change must not disappear behind a stale occupancy threshold."""

    fluorescence = np.arange(1, 6, dtype=np.uint16).repeat(7)

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

        assert task.model.kind is ReadoutModelKind.BOX
        assert task.model.reducer == "mean"
        measured, error, saturated, missing = task._measure(
            pulse, _Context(), 0, shots=10
        )
        # Values 1..4 are below the calibration's occupied threshold of 5,
        # but remain valid contributions to the repeat-mean observable.
        np.testing.assert_allclose(measured, fluorescence)
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
        artifacts = tuple(tmp_path.glob("slm_feedback_candidate_*.npz"))
        assert len(artifacts) == 2
        assert all(load_phase(path)[1]["status"] == "measured" for path in artifacts)
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
    def measured_result(self, pulse, context, iteration, *, shots=None):
        requested_shots.append(self.shots if shots is None else int(shots))
        return next(measurements)

    monkeypatch.setattr(SlmFeedbackTask, "_measure", measured_result)
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
        artifacts = tuple(tmp_path.glob("slm_feedback_candidate_*.npz"))
        assert len(artifacts) == 1
        assert load_phase(artifacts[0])[1]["status"] == "validated"
    finally:
        plane.close()


def test_stop_during_failed_first_candidate_save_restores_incoming(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
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
        lambda *args, **kwargs: (
            np.full(slm.shape_yx, 0.5, dtype=np.float32),
            {"method": "test"},
        ),
    )

    def fail_first_save(path, phase, metadata):
        if metadata["status"] == "applied":
            save_entered.set()
            assert release_save.wait(2.0)
        raise OSError("first candidate artifact failed")

    monkeypatch.setattr(feedback_module, "save_phase", fail_first_save)
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
    )
    host = _task_host(task, plane, wake)
    try:
        host.start()
        assert save_entered.wait(2.0)
        host.cancel("while first candidate phase is not durable")
        release_save.set()
        observation = _wait_host(host, wake)
        assert observation.phase == "failed"
        assert "first candidate artifact failed" in (observation.error or "")
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        assert not tuple(tmp_path.glob("slm_feedback_candidate_*.npz"))
    finally:
        release_save.set()
        if not host.terminal:
            _wait_host(host, wake)
        host.shutdown()
        plane.close()


def test_stop_at_terminal_gate_accepts_best_before_transient_previews_retire(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    best = np.full(slm.shape_yx, 0.5, dtype=np.float32)
    plane = SignalDataPlane()
    wake = Event()
    validation_entered = Event()
    release_validation = Event()
    stopped_save_entered = Event()
    release_stopped_save = Event()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (
            best,
            {"method": "test"},
        ),
    )
    calls = 0

    def measure(self, pulse, run_context, iteration, *, shots=None):
        nonlocal calls
        calls += 1
        fluorescence = np.ones(35)
        if shots is not None:
            validation_entered.set()
            assert release_validation.wait(2.0)
            fluorescence[0] = 1.005
        return fluorescence, np.zeros(35), (), ()

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
    save = feedback_module.save_phase

    def blocking_stopped_save(path, phase, metadata):
        if metadata["status"] == "stopped":
            stopped_save_entered.set()
            assert release_stopped_save.wait(2.0)
        return save(path, phase, metadata)

    monkeypatch.setattr(feedback_module, "save_phase", blocking_stopped_save)
    host = _task_host(task, plane, wake)
    try:
        host.start()
        assert validation_entered.wait(2.0)
        # Stop wins immediately before the final apply/save commit, after
        # validation has produced a passing result.  That measured candidate
        # is committed to both transient previews before Task terminal retires
        # them, and remains on the device and in the durable artifact.
        host.cancel("before terminal commit")
        release_validation.set()
        assert stopped_save_entered.wait(2.0)
        phase_publication = plane.latest_publication(
            host.signal_key("candidate_phase")
        )
        curve_publication = plane.latest_publication(
            host.signal_key("uniformity_history")
        )
        assert phase_publication is not None and curve_publication is not None
        phase_value = phase_publication.value(host.signal_key("candidate_phase"))
        curve_value = curve_publication.value(host.signal_key("uniformity_history"))
        np.testing.assert_array_equal(phase_value.snapshot.block.values[0, 0], best)
        np.testing.assert_array_equal(
            curve_value.snapshot.block.values[0, :, 0], np.asarray([1.0])
        )
        assert phase_value.run_record == curve_value.run_record
        release_stopped_save.set()
        observation = _wait_host(host, wake)
        assert observation.phase == "done"
        assert host.final_result_resolved
        assert calls == 2
        np.testing.assert_array_equal(slm.last_commanded_phase, best)
        artifacts = tuple(tmp_path.glob("slm_feedback_candidate_*.npz"))
        assert len(artifacts) == 1
        saved, metadata = load_phase(artifacts[0])
        np.testing.assert_array_equal(saved, best)
        assert metadata["candidate"] == 1
        # Task previews are Monitor projections and retire at terminal; the
        # durable artifact and commanded SLM remain the accepted truth.
        assert plane.latest_publication(host.signal_key("candidate_phase")) is None
        assert plane.latest_publication(host.signal_key("uniformity_history")) is None
    finally:
        release_validation.set()
        release_stopped_save.set()
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

    def blocking_save(path, phase, metadata):
        if metadata["status"] == "accepted":
            save_entered.set()
            assert release_save.wait(2.0)
        return original_save(path, phase, metadata)

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
    host = _task_host(task, plane, wake)
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


def test_stop_accepted_then_apply_failure_is_failed_and_restores_incoming(
    tmp_path: Path,
    monkeypatch,
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    best = np.full(slm.shape_yx, 0.5, dtype=np.float32)
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
        lambda *args, **kwargs: (best, {"method": "test"}),
    )

    def measure(self, pulse, context, iteration, *, shots=None):
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
    apply = task._apply_exact
    calls = 0

    def fail_stop_apply(phase):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("accepted Stop apply failed")
        return apply(phase)

    monkeypatch.setattr(task, "_apply_exact", fail_stop_apply)
    host = _task_host(task, plane, wake)
    try:
        host.start()
        assert validation_entered.wait(2.0)
        host.cancel("accept best then fail apply")
        release_validation.set()
        observation = _wait_host(host, wake)
        assert observation.phase == "failed"
        assert observation.error == "OSError: accepted Stop apply failed"
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
    finally:
        release_validation.set()
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
        original_save = feedback_module.save_phase

        def fail_terminal_save(path, phase, metadata):
            if metadata["status"] == "accepted":
                raise OSError("terminal save failed")
            return original_save(path, phase, metadata)

        monkeypatch.setattr(feedback_module, "save_phase", fail_terminal_save)
    host = _task_host(task, plane, wake)
    try:
        host.start()
        observation = _wait_host(host, wake)
        assert observation.phase == "failed"
        assert observation.error == f"OSError: terminal {terminal_operation} failed"
        assert not host.final_result_resolved
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        artifacts = tuple(tmp_path.glob("slm_feedback_candidate_*.npz"))
        assert len(artifacts) == 1
        saved, metadata = load_phase(artifacts[0])
        np.testing.assert_array_equal(saved, best)
        assert metadata["status"] == "validated"
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


def test_stop_before_first_candidate_accepts_incoming_as_formal_artifact(
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
        context = _Context(cancelled=True)
        result = task.execute(context)
        assert context.terminal_sealed
        saved, metadata = load_phase(result["artifact_path"])
        assert metadata["status"] == "stopped"
        np.testing.assert_array_equal(saved, incoming)
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        assert len(context.commits) == 1
        phase = context.commits[0]["candidate_phase"]
        history = context.commits[0]["uniformity_history"]
        np.testing.assert_array_equal(phase.snapshot.block.values[0, 0], incoming)
        assert not np.any(history.snapshot.expanded_validity())
        assert np.array_equal(slm.commands[-1], incoming)
        assert tuple(tmp_path.glob("slm_feedback*.npz")) == (
            Path(result["artifact_path"]),
        )
    finally:
        plane.close()
