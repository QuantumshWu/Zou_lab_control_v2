import inspect
import time
from dataclasses import replace
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
from zlc_atom.devices.slm.solver import load_science_context, preset_grid, solve_phase
from zlc_atom.install import create_installation
from zlc_atom.nodes import discover_logic_nodes
from zlc_atom.nodes._framework.descriptor import ResolvedArtifact
from zlc_atom.nodes.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.calibration.pulse import resolve_pulse
from zlc_atom.nodes.slm_feedback import task as feedback_module
from zlc_atom.nodes.slm_feedback.task import (
    SlmFeedbackTask,
    _censored_sites,
    _ratio_interval,
    _updated_target,
)
from zlc_atom.nodes.calibration.calibration import (
    _register_target_sites,
    validate_target_registration,
)
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE


class _Slm:
    identity = "test-slm"

    def __init__(self, shape: tuple[int, int], incoming: float = 0.1) -> None:
        self.shape_yx = shape
        self._phase = canonical_phase(np.full(shape, incoming), shape)
        self._command_revision = 1
        self._mapping_revision = 0
        self.receipt_overrides: dict[str, object] = {}
        self.commands: list[np.ndarray] = []

    @property
    def last_commanded_phase(self) -> np.ndarray:
        return self._phase

    @property
    def command_revision(self) -> int:
        return self._command_revision

    @property
    def mapping_revision(self) -> int:
        return self._mapping_revision

    @property
    def last_command_receipt(self) -> dict[str, object]:
        return {
            "transport": "virtual",
            "identity": self.identity,
            "profile": "virtual", "wavelength_nm": None,
            "flip_x": False, "flip_y": False,
            "correction_path": "", "correction_enabled": False,
            "mapping_revision": self._mapping_revision, "outcome": "known-new",
            "command_revision": self._command_revision,
            **self.receipt_overrides,
        }

    def apply_phase(self, radians: object) -> np.ndarray:
        self._phase = canonical_phase(radians, self.shape_yx)
        self._command_revision += 1
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
        dark_sample_count=np.full(35, 100),
        dark_sample_variance=np.zeros(35),
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
        dark_sample_count=np.full(count, 100),
        dark_sample_variance=np.zeros(count),
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


def _asymmetric_target(shape: tuple[int, int] = (17, 23)) -> np.ndarray:
    target = _grid_target(shape)
    row, column = np.argwhere(target > 0.0)[0]
    target[row, column] = 0.0
    target[row - 1, column + 1] = 1.0
    return target


def _science_context(
    slm: _Slm,
    *,
    target: np.ndarray,
    pattern: np.ndarray | None = None,
    wavefront: np.ndarray | None = None,
    pupil: np.ndarray | None = None,
) -> dict[str, object]:
    incoming = np.asarray(slm.last_commanded_phase)
    pattern_phase = incoming if pattern is None else canonical_phase(pattern, slm.shape_yx)
    operator = (
        np.zeros(slm.shape_yx, dtype=np.float32)
        if wavefront is None
        else canonical_phase(wavefront, slm.shape_yx)
    )
    phase = canonical_phase(pattern_phase.astype(float) + operator.astype(float), slm.shape_yx)
    if pattern is None and wavefront is None:
        phase = slm.last_commanded_phase
    amplitude = (
        np.ones(slm.shape_yx, dtype=np.float32)
        if pupil is None
        else np.asarray(pupil, dtype=np.float32)
    )
    return {
        "phase": phase,
        "pattern_phase": pattern_phase,
        "operator_wavefront": operator,
        "pupil_amplitude": amplitude,
        "pupil_support": amplitude > 0.0,
        "target_intensity": np.array(target, copy=True),
        "objective_kind": "spots",
        "pupil": {
            "enabled": True,
            "center_xy": [(slm.shape_yx[1] - 1) / 2, (slm.shape_yx[0] - 1) / 2],
            "diameter_xy": [float(slm.shape_yx[1]), float(slm.shape_yx[0])],
        },
        "system_correction": None,
        "command_receipt": slm.last_command_receipt,
        "pattern_metadata": {},
        "operator_metadata": {},
    }


def _load_candidate(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    context = load_science_context(path)
    return context["phase"], context["pattern_metadata"]


def _calibration_with_unresolved_site(
    target: np.ndarray,
    *,
    missing: int,
) -> TrapCalibration:
    rows, columns = np.nonzero(target > 0.0)
    keep = np.arange(len(rows)) != int(missing)
    detected = SiteMap(
        tuple(f"detected_{index:04d}" for index in range(np.count_nonzero(keep))),
        np.column_stack((columns[keep], rows[keep])).astype(float),
        np.ones(np.count_nonzero(keep), dtype=bool),
        np.ones(np.count_nonzero(keep)),
    )
    site_map = detected
    usable = np.ones(site_map.n_sites, dtype=bool)
    model = ReadoutModel(
        site_map.site_ids,
        np.full(site_map.n_sites, 5.0),
        np.zeros(site_map.n_sites),
        np.full(site_map.n_sites, 10.0),
        usable,
        np.ones(site_map.n_sites),
        dark_sample_count=np.full(site_map.n_sites, 100),
        dark_sample_variance=np.zeros(site_map.n_sites),
        kind=ReadoutModelKind.BOX,
        integration_half_width=0,
        reducer="mean",
    )
    return TrapCalibration(
        site_map,
        (model,),
        ReadoutModelKind.BOX,
        FrameContract(target.shape, exposure_seconds=0.020),
    )


def _task(
    tmp_path: Path,
    *,
    slm: _Slm,
    camera: object,
    sequencer: object,
    plane: SignalDataPlane,
    target: np.ndarray | None = None,
    calibration: TrapCalibration | None = None,
    shots: int = 10,
    validation_shots: int = 20,
    updates: int = 3,
    science_context: dict[str, object] | None = None,
) -> SlmFeedbackTask:
    if science_context is None:
        if target is None:
            raise ValueError("test must supply a Target or Science Context")
        frozen_context = _science_context(slm, target=target)
    else:
        frozen_context = science_context
    selected_calibration = _calibration() if calibration is None else calibration
    context_path = tmp_path / "science_context.npz"
    return SlmFeedbackTask(
        camera=camera,
        camera_key="qcmos",
        sequencer=sequencer,
        sequencer_key="sequencer",
        slm=slm,
        slm_key="slm",
        signal_plane=plane,
        calibration=selected_calibration,
        calibration_path=tmp_path / "calibration.json",
        science_context=frozen_context,
        science_context_path=context_path,
        pulse_sequence=IMAGING_PULSE_RESOURCE.value,
        pulse_path=IMAGING_PULSE_RESOURCE.path,
        shots_per_candidate=shots,
        validation_shots=validation_shots,
        max_updates=updates,
        artifact_directory=tmp_path,
    )


def test_descriptor_and_direct_update_keep_the_plugin_boundary() -> None:
    descriptors = {item.api_name: item for item in discover_logic_nodes()}
    descriptor = descriptors["slm_feedback"]
    calibration_inputs = descriptors["calibration"].input_specs
    assert calibration_inputs == ()
    assert tuple(item.name for item in descriptor.input_specs) == (
        "calibration_path",
        "science_context_path",
    )
    assert tuple(item.contract_id for item in descriptor.input_specs) == (
        "calibration.readout.v1",
        "zlc.slm.science-context.v2",
    )
    assert tuple(item.field_name for item in descriptor.workspace_resources) == (
        "pulse_template",
    )
    assert tuple(
        (item.name, item.contract_id) for item in descriptor.artifact_outputs
    ) == (("artifact_path", "zlc.slm.science-context.v2"),)
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
    standard_error = 0.02 * fluorescence
    updated = _updated_target(
        target,
        fluorescence,
        standard_error,
        rows,
        columns,
        gain=0.25,
    )
    assert updated[rows[0], columns[0]] > updated[rows[-1], columns[-1]]
    np.testing.assert_allclose(np.sum(updated), np.sum(target), rtol=1e-6)
    multipliers = updated[rows, columns] / target[rows, columns]
    assert float(np.max(multipliers) / np.min(multipliers)) <= np.exp(0.4)
    estimate, lower, upper, max_relative_sem = _ratio_interval(
        fluorescence, standard_error
    )
    assert lower <= estimate <= upper
    assert estimate == pytest.approx(1.4 / 0.6)
    assert max_relative_sem == pytest.approx(0.02)
    _estimate3, lower3, upper3, _relative3 = _ratio_interval(
        fluorescence, standard_error, looks=3
    )
    assert lower3 < lower and upper3 > upper
    assert set(_censored_sites(fluorescence, standard_error)) <= set(
        _censored_sites(fluorescence, standard_error, looks=3)
    )


def test_feedback_freezes_science_context_and_solves_only_pattern_to_gate(
    tmp_path: Path, monkeypatch
) -> None:
    shape = (17, 23)
    slm = _Slm(shape)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    pattern = canonical_phase(np.broadcast_to(0.2 + xx / 31.0, shape), shape)
    wavefront = canonical_phase(np.broadcast_to(0.1 + yy / 29.0, shape), shape)
    pupil = np.asarray(
        np.exp(-((xx - 11.0) ** 2 + (yy - 8.0) ** 2) / 80.0),
        dtype=np.float32,
    )
    incoming = canonical_phase(pattern.astype(float) + wavefront.astype(float), shape)
    slm.apply_phase(incoming)
    frozen_target = _grid_target(shape)
    science_context = _science_context(
        slm,
        target=frozen_target,
        pattern=pattern,
        wavefront=wavefront,
        pupil=pupil,
    )
    solved_pattern = canonical_phase(pattern.astype(float) + 0.05, shape)

    def solve(target, **kwargs):
        assert kwargs["objective_kind"] == "spots"
        assert kwargs["iterations"] is None
        np.testing.assert_array_equal(kwargs["initial_phase"], pattern)
        np.testing.assert_array_equal(kwargs["pupil_amplitude"], pupil)
        return solved_pattern, {
            "iterations_run": 19,
            "support_intensity_ratio": 1.005,
            "diffraction_efficiency": 0.8,
            "transform": "selected-dft",
        }

    monkeypatch.setattr(feedback_module, "solve_phase", solve)
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration, *, shots=None: (
            np.ones(35),
            np.zeros(35),
            (),
            (),
        ),
    )
    plane = SignalDataPlane()

    def build(
        *,
        selected_context=science_context,
        selected_calibration=None,
    ):
        return _task(
            tmp_path,
            slm=slm,
            camera=object(),
            sequencer=SimpleNamespace(describe=lambda: object()),
            plane=plane,
            calibration=selected_calibration,
            science_context=selected_context,
        )

    unknown = {
        **science_context,
        "command_receipt": {
            **science_context["command_receipt"],
            "outcome": "unknown",
        },
    }
    with pytest.raises(ValueError, match="known incoming command receipt"):
        build(selected_context=unknown)
    with pytest.raises(ValueError, match="legacy Science Context"):
        build(
            selected_context={**science_context, "target_intensity": None}
        )
    generic = _calibration()
    build(selected_calibration=generic)
    legacy_dark = replace(
        generic,
        models=tuple(
            replace(
                model,
                dark_sample_count=None,
                dark_sample_variance=None,
            )
            for model in generic.models
        ),
    )
    with pytest.raises(ValueError, match="calibrated BOX site"):
        build(selected_calibration=legacy_dark)
    task = build()
    try:
        result = task.execute(_Context())
        artifact = load_science_context(result["artifact_path"])
        expected = canonical_phase(
            solved_pattern.astype(float) + wavefront.astype(float), shape
        )
        np.testing.assert_array_equal(artifact["pattern_phase"], solved_pattern)
        np.testing.assert_array_equal(artifact["operator_wavefront"], wavefront)
        np.testing.assert_array_equal(artifact["pupil_amplitude"], pupil)
        np.testing.assert_array_equal(artifact["target_intensity"], frozen_target)
        np.testing.assert_array_equal(artifact["phase"], expected)
        np.testing.assert_array_equal(slm.last_commanded_phase, expected)
        assert artifact["pattern_metadata"]["solver"]["iterations_run"] == 19
        assert artifact["command_receipt"]["outcome"] == "known-new"
        mapping_stale = _task(
            tmp_path,
            slm=slm,
            camera=object(),
            sequencer=SimpleNamespace(describe=lambda: object()),
            plane=plane,
            target=_grid_target(shape),
        )
        command_count = len(slm.commands)
        slm.receipt_overrides["correction_path"] = "different_mapping.bmp"
        with pytest.raises(RuntimeError, match="no longer matches"):
            mapping_stale.execute(_Context())
        assert len(slm.commands) == command_count
        slm.receipt_overrides.clear()
        stale = _task(
            tmp_path,
            slm=slm,
            camera=object(),
            sequencer=SimpleNamespace(describe=lambda: object()),
            plane=plane,
            target=_grid_target(shape),
        )
        external = canonical_phase(np.full(shape, 1.25), shape)
        slm.apply_phase(external)
        with pytest.raises(RuntimeError, match="no longer matches"):
            stale.execute(_Context())
        np.testing.assert_array_equal(slm.last_commanded_phase, external)
    finally:
        plane.close()


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
            target_fluorescence,
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
        expected = _updated_target(
            target,
            target_fluorescence,
            np.zeros(len(rows)),
            rows,
            columns,
            gain=0.25,
        )
        np.testing.assert_allclose(solved_targets[1], expected)
        candidates = {
            context["pattern_metadata"]["candidate"]: context
            for context in map(
                load_science_context,
                tmp_path.glob("slm_feedback_candidate_*.npz"),
            )
        }
        np.testing.assert_allclose(candidates[2]["target_intensity"], expected)
        _saved, metadata = _load_candidate(result["artifact_path"])
        np.testing.assert_allclose(
            metadata["history"][0]["fluorescence"],
            target_fluorescence,
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
            np.ones(35),
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
        result = task.execute(context)
        assert result["validation_status"] == "accepted"
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


@pytest.mark.parametrize("failure", ("distortion", "ambiguous"))
def test_sparse_geometry_refuses_distorted_or_ambiguous_calibration(
    tmp_path: Path, failure: str
) -> None:
    target = np.zeros((17, 23), dtype=np.float32)
    rows = np.asarray([2, 2, 8, 8])
    columns = np.asarray([3, 12, 5, 15])
    target[rows, columns] = 1.0
    centers = np.column_stack((10.0 + 2.0 * columns, 15.0 + 3.0 * rows))
    if failure == "distortion":
        centers[2] += (20.0, 15.0)
    else:
        centers[1] = centers[0]
    plane = SignalDataPlane()
    try:
        with pytest.raises(ValueError, match="geometry|ambiguous"):
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


def test_registration_refuses_colliding_predicted_site_boxes() -> None:
    target = np.zeros((16, 16), dtype=np.float32)
    target[(2, 2, 12, 2), (2, 12, 2, 7)] = 1.0
    detected = SiteMap(
        ("a", "b", "c"),
        np.asarray(((20.0, 20.0), (40.0, 20.0), (20.0, 40.0))),
        np.ones(3, dtype=bool),
        np.ones(3),
    )
    with pytest.raises(ValueError, match="collide|separation|overlap"):
        _register_target_sites(
            detected,
            target,
            {"science_context_path": "c", "command_receipt": {}},
            frame_shape=(64, 64),
            measurement_radius=6,
        )


def test_regular_nine_by_nine_grid_registers_directly_with_one_missing_site() -> None:
    coordinates = np.arange(3, 30, 3)
    target = np.zeros((33, 33), dtype=np.float32)
    target[np.ix_(coordinates, coordinates)] = 1.0
    rows, columns = np.nonzero(target)
    centers = np.column_stack((10.0 + 2.0 * columns, 12.0 + 2.0 * rows))
    missing = 40
    detected = SiteMap(
        tuple(f"site_{index}" for index in range(80)),
        np.delete(centers, missing, axis=0),
        np.ones(80, dtype=bool),
        np.ones(80),
    )

    registered = _register_target_sites(
        detected,
        target,
        {"science_context_path": "c", "command_receipt": {}},
        frame_shape=(80, 80),
        measurement_radius=0,
    )
    support, _provenance = validate_target_registration(
        registered,
        frame_shape=(80, 80),
        box_half_width=0,
    )

    assert len(support) == 81
    assert np.flatnonzero(~registered.valid_sites).tolist() == [missing]
    np.testing.assert_allclose(registered.centers_xy[missing], centers[missing])


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
        def __init__(self) -> None:
            self.loaded = None
            self.fires: list[int | None] = []

        def describe(self):
            return object()

        def load(self, program, *, source=None, rows=()) -> None:
            assert not rows
            self.loaded = (program, source)

        def fire(self, *, cycles=1) -> None:
            self.fires.append(cycles)
            camera.trigger(int(cycles) * 3)

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
    raw_calibration = replace(
        _calibration(),
        report={
            "run_record": {
                "request": {"photoelectrons": False},
                "actual_devices": {
                    "qcmos": {"dtype": "<u2", "count_unit": "count"}
                },
            }
        },
    )
    task = _task(
        tmp_path,
        slm=slm,
        camera=camera,
        sequencer=sequencer,
        plane=plane,
        target=np.ones((5, 7), dtype=np.float32),
        calibration=raw_calibration,
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
        current_dataset = plane.current_dataset
        lookup_count = 0

        def one_result_lookup(*args, **kwargs):
            nonlocal lookup_count
            lookup_count += 1
            if lookup_count > 1:
                raise AssertionError("Feedback re-looked up Plane state after MeasurementResult")
            return current_dataset(*args, **kwargs)

        monkeypatch.setattr(plane, "current_dataset", one_result_lookup)
        measured, error, saturated, missing = task._measure(
            pulse, _Context(), 0, shots=10
        )
        assert lookup_count == 1
        monkeypatch.setattr(plane, "current_dataset", current_dataset)
        np.testing.assert_allclose(measured, fluorescence)
        np.testing.assert_allclose(error, 0.0, atol=1e-15)
        assert not saturated
        assert not missing
        assert measured[17] == fluorescence[17]
        assert sequencer.fires == [10]
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

        sample = 0

        def partial_signals(image, centers, *, radius, reducer):
            nonlocal sample
            del image, centers, radius, reducer
            sample += 1
            values = np.ones(task.calibration.n_sites)
            if sample > 2:
                values[0] = np.nan
            return values

        monkeypatch.setattr(feedback_module, "extract_box_signals", partial_signals)
        partial_mean, partial_error, _saturated, partial_missing = task._measure(
            pulse, _Context(), 0, shots=10
        )
        assert partial_missing == (0,)
        assert np.isnan(partial_mean[0]) and np.isnan(partial_error[0])

        sample = 0

        def partial_validation(image, centers, *, radius, reducer):
            nonlocal sample
            del image, centers, radius, reducer
            sample += 1
            values = np.ones(task.calibration.n_sites)
            if sample > 12:
                values[0] = np.nan
            return values

        monkeypatch.setattr(
            feedback_module, "extract_box_signals", partial_validation
        )
        monkeypatch.setattr(feedback_module, "resolve_pulse", lambda *args, **kwargs: pulse)
        monkeypatch.setattr(
            feedback_module,
            "solve_phase",
            lambda *args, **kwargs: (
                np.full(task.slm.shape_yx, 0.5, dtype=np.float32),
                {"method": "test"},
            ),
        )
        result = task.execute(_Context())
        assert result["validation_status"] == "inconclusive"
        _phase, metadata = _load_candidate(result["artifact_path"])
        assert metadata["best"]["validation"]["reason"] == (
            "independent validation data were invalid"
        )
        assert not camera.capture_state()
    finally:
        plane.close()
        camera.close()


def test_electron_measurement_freezes_conversion_and_saturation(
    tmp_path: Path,
) -> None:
    target = np.ones((5, 7), dtype=np.float32)
    recorded_offset, recorded_scale = 10.0, 0.5

    def calibration() -> TrapCalibration:
        return replace(
            _calibration(),
            report={
                "run_record": {
                    "request": {"photoelectrons": True},
                    "actual_devices": {
                        "qcmos": {
                            "dtype": "<u2",
                            "count_unit": "count",
                            "offset_counts": recorded_offset,
                            "electrons_per_count": recorded_scale,
                        }
                    },
                }
            },
        )

    class Sequencer:
        def __init__(self, camera):
            self.camera = camera

        def describe(self):
            return object()

        def load(self, program, *, source=None, rows=()):
            return None

        def fire(self, *, cycles=1):
            self.camera.trigger(int(cycles) * 3)

        def wait_done(self, timeout=None):
            return SimpleNamespace(fault=None)

        def safe(self):
            return None

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

    def run(offset, scale, expected_error=None):
        def frame_source(ordinal, exposure):
            image = np.zeros((5, 7), dtype="<u2")
            if ordinal % 3 == 1:
                image[2, 3] = np.iinfo("<u2").max
            return image

        camera = VirtualCamera(
            VirtualCameraConfig(
                frame_shape_yx=(5, 7),
                exposure_seconds=0.02,
                offset_counts=offset,
                electrons_per_count=scale,
            ),
            frame_source=frame_source,
        )
        plane = SignalDataPlane()
        task = _task(
            tmp_path,
            slm=_Slm(target.shape),
            camera=camera,
            sequencer=Sequencer(camera),
            plane=plane,
            target=target,
            calibration=calibration(),
        )
        try:
            if expected_error is not None:
                with pytest.raises(ValueError, match=expected_error):
                    task._measure(pulse, _Context(), 0, shots=10)
                return
            _mean, _error, saturated, missing = task._measure(
                pulse, _Context(), 0, shots=10
            )
            assert saturated == (17,) and not missing
        finally:
            plane.close()
            camera.close()

    run(recorded_offset, recorded_scale)
    run(None, None, "effective photoelectron mode")
    run(recorded_offset, 0.6, "conversion differs")


def test_censored_site_uses_bounded_batches_and_bootstrap_boost(
    tmp_path: Path, monkeypatch
) -> None:
    target = _asymmetric_target()
    slm = _Slm(target.shape)
    context_mapping = _science_context(slm, target=target)
    calibration = _calibration_with_unresolved_site(
        target,
        missing=17,
    )
    requested: list[int] = []
    measurements = iter(
        [
            (np.where(np.arange(35) == 17, 0.0, 1.0), np.full(35, 0.1), (), ())
            for _ in range(3)
        ]
        + [
            (np.ones(35), np.zeros(35), (), ()),
            (np.ones(35), np.zeros(35), (), ()),
        ]
    )
    solved_targets: list[np.ndarray] = []

    def solve(candidate, **_kwargs):
        solved_targets.append(np.array(candidate, copy=True))
        return np.full(candidate.shape, 0.1 * len(solved_targets), np.float32), {}

    monkeypatch.setattr(feedback_module, "solve_phase", solve)
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )

    def measure(self, pulse, context, iteration, *, shots=None):
        requested.append(self.shots if shots is None else int(shots))
        return next(measurements)

    monkeypatch.setattr(SlmFeedbackTask, "_measure", measure)
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        calibration=calibration,
        science_context=context_mapping,
        updates=2,
    )
    try:
        result = task.execute(_Context())
        assert result["validation_status"] == "accepted"
        assert requested == [10, 10, 10, 10, 20]
        rows, columns = np.nonzero(target > 0.0)
        assert solved_targets[1][rows[17], columns[17]] > solved_targets[1][
            rows[0], columns[0]
        ]
        assert np.sum(solved_targets[1]) == pytest.approx(np.sum(target))
        support_values = solved_targets[1][rows, columns]
        assert float(
            np.max(support_values) / np.min(support_values)
        ) == pytest.approx(np.exp(0.2), rel=1e-6)
        candidate_context = load_science_context(result["artifact_path"])
        metadata = candidate_context["pattern_metadata"]
        assert metadata["history"][0]["censored_sites"] == [17]
        assert metadata["history"][0]["shots"] == 30
        descriptor = {item.api_name: item for item in discover_logic_nodes()}[
            "slm_feedback"
        ]
        reused = descriptor.instantiate(
            slm=slm,
            slm_key="slm",
            camera=object(),
            camera_key="qcmos",
            sequencer=SimpleNamespace(describe=lambda: object()),
            sequencer_key="sequencer",
            signal_plane=plane,
            calibration=ResolvedArtifact(
                tmp_path / "calibration.json",
                "calibration.readout.v1",
                calibration,
            ),
            science_context=ResolvedArtifact(
                Path(result["artifact_path"]),
                "zlc.slm.science-context.v2",
                candidate_context,
            ),
            pulse_resource=IMAGING_PULSE_RESOURCE,
            artifact_directory=tmp_path,
            max_updates=1,
        )
        np.testing.assert_array_equal(reused.target, candidate_context["target_intensity"])
    finally:
        plane.close()


def test_adaptive_pooling_keeps_frozen_dark_uncertainty_once(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23))
    target = _grid_target(slm.shape_yx)
    calibration = _calibration()
    model = replace(
        calibration.select_model(),
        dark_sample_count=np.full(35, 100),
        dark_sample_variance=np.full(35, 4.0),
    )
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=object(),
        plane=plane,
        target=target,
        calibration=replace(calibration, models=(model,)),
    )
    calls = 0

    def measure(pulse, context, iteration, *, shots=None):
        nonlocal calls
        calls += 1
        return np.full(35, 0.5), np.full(35, np.sqrt(0.05)), (), ()

    monkeypatch.setattr(task, "_measure", measure)
    try:
        _mean, error, _saturated, _missing, censored, _attempts, shots = (
            task._coarse_measure(object(), _Context(), 0)
        )
        assert calls == 3 and shots == 30 and censored
        expected = np.sqrt(0.04 + (3.0 * 0.01 * 10.0 * 9.0) / (29.0 * 30.0))
        np.testing.assert_allclose(error, expected)
    finally:
        plane.close()


@pytest.mark.parametrize("stop_after_first", (False, True))
def test_persistently_censored_bootstrap_preserves_incoming(
    tmp_path: Path, monkeypatch, stop_after_first: bool
) -> None:
    target = _asymmetric_target()
    slm = _Slm(target.shape)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    calibration = _calibration_with_unresolved_site(
        target,
        missing=17,
    )
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    context = _Context()
    solve_calls = 0

    def solve(candidate, **_kwargs):
        nonlocal solve_calls
        solve_calls += 1
        if stop_after_first and solve_calls == 2:
            context.cancelled = True
        return np.full(candidate.shape, 0.25, np.float32), {}

    monkeypatch.setattr(feedback_module, "solve_phase", solve)
    calls = 0

    def censored(self, pulse, context, iteration, *, shots=None):
        nonlocal calls
        calls += 1
        return (
            np.where(np.arange(35) == 17, 0.0, 1.0),
            np.full(35, 0.1),
            (),
            (),
        )

    monkeypatch.setattr(SlmFeedbackTask, "_measure", censored)
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=target,
        calibration=calibration,
        updates=10,
    )
    try:
        result = task.execute(context)
        artifact = load_science_context(result["artifact_path"])
        np.testing.assert_array_equal(artifact["phase"], incoming)
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        metadata = artifact["pattern_metadata"]
        assert metadata["candidate"] == 0
        assert metadata["measurement"] is None
        if stop_after_first:
            assert result["validation_status"] is None
            assert metadata["status"] == "stopped"
            assert calls == 3
        else:
            assert result["validation_status"] == "inconclusive"
            assert calls == 12
            assert metadata["best"]["validation"]["reason"] == (
                "censored sites remained after bounded bootstrap"
            )
            assert metadata["history"][-1]["bootstrap_updates"] == 3
    finally:
        plane.close()


def test_virtual_feedback_runs_repeated_real_qcmos_candidates_and_restores(
    tmp_path: Path,
) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {item.api_name: item for item in discover_logic_nodes()}
    camera = installation.device("camera")
    sequencer = installation.device("sequencer")
    slm = installation.device("slm")
    try:
        target = preset_grid(slm.shape_yx, (5, 7))
        support = np.argwhere(target > 0.0)
        target = np.array(target, copy=True)
        weak_row, weak_column = support[17]
        target[weak_row, weak_column] = 0.1
        pattern, _metadata = solve_phase(
            target, objective_kind="spots", iterations=None
        )
        slm.apply_phase(pattern)
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
        calibration_result = calibration_node.run()
        calibration = TrapCalibration.load(calibration_result.artifact_path)
        assert calibration.site_map.n_sites == 34
        assert calibration.site_map.topology is None
        box = calibration.select_model(ReadoutModelKind.BOX)
        assert np.all(box.dark_sample_count >= 2)
        assert np.all(np.isfinite(box.dark_sample_variance))
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
            calibration_path=calibration_result.artifact_path,
            science_context=_science_context(slm, target=target),
            science_context_path=tmp_path / "science_context.npz",
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            shots_per_candidate=40,
            validation_shots=80,
            max_updates=4,
            artifact_directory=tmp_path,
        )
        result = task.execute(context)
        saved, metadata = _load_candidate(result["artifact_path"])
        np.testing.assert_array_equal(slm.last_commanded_phase, saved)
        assert not np.array_equal(saved, pattern)
        assert result["validation_status"] in {"accepted", "inconclusive"}
        assert metadata["status"] == result["validation_status"]
        assert [item["valid"] for item in metadata["history"]] == [
            False, False, False, True
        ]
        assert metadata["candidate"] == 4 and metadata["measurement"]["valid"]
        assert not camera.capture_state()
        censored_messages = [
            args[0]
            for args, _kwargs in context.progress
            if args and "was censored" in str(args[0])
        ]
        assert len(censored_messages) == 3
        artifacts = tuple(tmp_path.glob("slm_feedback_candidate_*.npz"))
        assert len(artifacts) == 4
        assert { _load_candidate(path)[1]["status"] for path in artifacts } <= {
            "measured", "accepted", "inconclusive"
        }
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
        saved, metadata = _load_candidate(result["artifact_path"])
        np.testing.assert_array_equal(saved, slm.last_commanded_phase)
        np.testing.assert_array_equal(saved, np.full(slm.shape_yx, 0.75, np.float32))
        assert metadata["best"]["validation"]["uniformity_ratio"] == 1.0
        assert requested_shots == [10, 10, 20]
        assert resolved_api_values == [{}]
        assert result["updates"] == 2
        assert np.array_equal(slm.commands[-1], saved)
    finally:
        plane.close()


def test_non_improving_candidate_rolls_back_best_and_reduces_controller_gain(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23))
    plane = SignalDataPlane()
    phases = tuple(
        np.full(slm.shape_yx, value, dtype=np.float32)
        for value in (0.25, 0.50, 0.75)
    )
    phase_results = iter(phases)
    first_fluorescence = np.concatenate(([2.0], np.ones(34)))
    measurements = iter(
        (
            (first_fluorescence, np.zeros(35), (), ()),
            (
                np.concatenate(([1.8], np.ones(34))),
                np.full(35, 0.2),
                (),
                (),
            ),
            (np.concatenate(([1.5], np.ones(34))), np.zeros(35), (), ()),
            (np.ones(35), np.zeros(35), (), ()),
        )
    )
    requested_shots: list[int] = []
    solved_targets: list[np.ndarray] = []
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    def solve(target, **kwargs):
        solved_targets.append(np.array(target, copy=True))
        return next(phase_results), {"method": "test"}

    monkeypatch.setattr(feedback_module, "solve_phase", solve)
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
        saved, metadata = _load_candidate(result["artifact_path"])
        np.testing.assert_array_equal(saved, phases[2])
        assert requested_shots == [10, 10, 10, 20]
        assert metadata["history"][1]["rollback_to_candidate"] == 1
        assert metadata["history"][2]["controller_gain"] == pytest.approx(0.125)
        expected = _updated_target(
            _grid_target(slm.shape_yx),
            first_fluorescence,
            np.zeros(35),
            *np.nonzero(_grid_target(slm.shape_yx)),
            gain=0.125,
        )
        np.testing.assert_allclose(solved_targets[2], expected)
        np.testing.assert_array_equal(slm.commands[0], phases[0])
        np.testing.assert_array_equal(slm.commands[1], phases[1])
        np.testing.assert_array_equal(slm.commands[2], phases[0])
    finally:
        plane.close()


@pytest.mark.parametrize(
    ("validation_shots", "expected_shots"),
    ((20, [10, 20]), (101, [10, 99, 2])),
)
def test_validation_refuses_a_point_estimate_with_wide_qcmos_uncertainty(
    tmp_path: Path,
    monkeypatch,
    validation_shots: int,
    expected_shots: list[int],
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
    requested: list[int] = []
    batches = iter(
        [(np.ones(35), np.zeros(35), (), ())]
        + [
            (np.ones(35), np.full(35, 0.02), (), ())
            for _ in expected_shots[1:]
        ]
    )

    def measure(self, pulse, context, iteration, *, shots=None):
        requested.append(self.shots if shots is None else int(shots))
        return next(batches)

    monkeypatch.setattr(SlmFeedbackTask, "_measure", measure)
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        updates=1,
        validation_shots=validation_shots,
    )
    try:
        result = task.execute(_Context())
        assert result["validation_status"] == "inconclusive"
        assert result["best_uniformity"] == pytest.approx(1.0)
        assert result["validation_confidence_upper"] > 1.01
        artifacts = tuple(tmp_path.glob("slm_feedback_candidate_*.npz"))
        assert len(artifacts) == 1
        saved, metadata = _load_candidate(artifacts[0])
        np.testing.assert_array_equal(saved, slm.last_commanded_phase)
        assert metadata["status"] == "inconclusive"
        assert metadata["best"]["validation"]["reason"] == (
            "maximum validation shots reached"
        )
        assert requested == expected_shots
    finally:
        plane.close()


def test_validation_adapts_in_independent_batches_until_confidence_resolves(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23))
    plane = SignalDataPlane()
    requested: list[int] = []
    batches = iter(
        (
            (np.ones(35), np.zeros(35), (), ()),
            (np.ones(35), np.full(35, 0.002), (), ()),
            (np.ones(35), np.full(35, 0.002), (), ()),
        )
    )
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    solve_calls = 0

    def solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return np.full(slm.shape_yx, 0.5, dtype=np.float32), {"method": "test"}

    monkeypatch.setattr(feedback_module, "solve_phase", solve)

    def measure(self, pulse, context, iteration, *, shots=None):
        requested.append(self.shots if shots is None else int(shots))
        return next(batches)

    monkeypatch.setattr(SlmFeedbackTask, "_measure", measure)
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        updates=1,
        validation_shots=300,
    )
    try:
        result = task.execute(_Context())
        assert result["validation_status"] == "accepted"
        assert requested == [10, 100, 100]
        _saved, metadata = _load_candidate(result["artifact_path"])
        validation = metadata["best"]["validation"]
        assert validation["shots"] == 200
        assert validation["maximum_shots"] == 300
        assert validation["maximum_looks"] == 3
        assert validation["confidence_family_alpha"] == 0.05
        assert validation["uniformity_confidence_upper"] <= 1.01
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

    def fail_first_save(path, phase, **kwargs):
        if kwargs["pattern_metadata"]["status"] == "applied":
            save_entered.set()
            assert release_save.wait(2.0)
        raise OSError("first candidate artifact failed")

    monkeypatch.setattr(feedback_module, "save_science_context", fail_first_save)
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


def test_stop_at_terminal_gate_accepts_best_and_retains_final_previews(
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
    save = feedback_module.save_science_context

    def blocking_stopped_save(path, phase, **kwargs):
        if kwargs["pattern_metadata"]["status"] == "stopped":
            stopped_save_entered.set()
            assert release_stopped_save.wait(2.0)
        return save(path, phase, **kwargs)

    monkeypatch.setattr(
        feedback_module, "save_science_context", blocking_stopped_save
    )
    host = _task_host(task, plane, wake)
    try:
        host.start()
        assert validation_entered.wait(2.0)
        # Stop wins immediately before the final apply/save commit, after
        # validation has produced a passing result.  That measured candidate
        # is committed to both retained previews before Task terminal, and
        # remains there as well as on the device and in the durable artifact.
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
        saved, metadata = _load_candidate(artifacts[0])
        np.testing.assert_array_equal(saved, best)
        assert metadata["candidate"] == 1
        # Feedback previews are latest-value monitors while running, but their
        # final phase and convergence curve remain visible after terminal.
        retained_phase = plane.latest_publication(
            host.signal_key("candidate_phase")
        )
        retained_curve = plane.latest_publication(
            host.signal_key("uniformity_history")
        )
        assert retained_phase is not None and retained_curve is not None
        np.testing.assert_array_equal(
            retained_phase.value(host.signal_key("candidate_phase")).snapshot.block.values[0, 0],
            best,
        )
        np.testing.assert_array_equal(
            retained_curve.value(host.signal_key("uniformity_history")).snapshot.block.values[0, :, 0],
            np.asarray([1.0]),
        )
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
    original_save = feedback_module.save_science_context

    def blocking_save(path, phase, **kwargs):
        if kwargs["pattern_metadata"]["status"] == "accepted":
            save_entered.set()
            assert release_save.wait(2.0)
        return original_save(path, phase, **kwargs)

    monkeypatch.setattr(feedback_module, "save_science_context", blocking_save)
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
        saved, _metadata = _load_candidate(result["artifact_path"])
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
        original_save = feedback_module.save_science_context

        def fail_terminal_save(path, phase, **kwargs):
            if kwargs["pattern_metadata"]["status"] == "accepted":
                raise OSError("terminal save failed")
            return original_save(path, phase, **kwargs)

        monkeypatch.setattr(
            feedback_module, "save_science_context", fail_terminal_save
        )
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
        saved, metadata = _load_candidate(artifacts[0])
        np.testing.assert_array_equal(saved, best)
        assert metadata["status"] == "measured"
    finally:
        if not host.terminal:
            _wait_host(host, wake)
        host.shutdown()
        plane.close()


def test_persistent_missing_retries_same_candidate_once_then_stops_invalid(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    plane = SignalDataPlane()
    phase = np.full(slm.shape_yx, 0.25, dtype=np.float32)
    first = np.ones(35)
    first[4] = np.nan
    batches = iter(
        (
            (first, np.full(35, np.nan), (), (4,)),
            (first, np.full(35, np.nan), (), (4,)),
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
        with pytest.raises(RuntimeError, match="invalid after two measurements"):
            task.execute(_Context())
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        artifacts = tuple(tmp_path.glob("slm_feedback_candidate_*.npz"))
        assert len(artifacts) == 1
        saved, metadata = _load_candidate(artifacts[0])
        assert metadata["measurement"]["attempts"] == 2
        assert metadata["measurement"]["fluorescence"][4] is None
        assert metadata["measurement"]["standard_error"][4] is None
        assert "history" not in metadata
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
    solve_calls = 0

    def solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return np.full(slm.shape_yx, 0.5, dtype=np.float32), {"method": "test"}

    monkeypatch.setattr(feedback_module, "solve_phase", solve)

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
        assert solve_calls == 0
        assert context.terminal_sealed
        saved, metadata = _load_candidate(result["artifact_path"])
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
