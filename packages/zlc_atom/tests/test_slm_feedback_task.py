import inspect
import json
import time
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from zlc_data.figure_archive import read_archive
from zlc_pulse import PulseSequence
from zlc_plot import FacetGridPlot, HistogramPlot, Reduction, read_figure_plot
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.devices.camera import CameraWorkingPoint
from zlc_atom.devices.simulation import SimulationWorld, SimulationWorldConfig
from zlc_atom.devices.simulation.camera import VirtualCamera, VirtualCameraConfig
from zlc_atom.devices.slm import canonical_phase
from zlc_atom.devices.slm.solver import load_science_context, preset_grid, solve_phase
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
from zlc_atom.nodes.slm_feedback.task import (
    SlmFeedbackTask,
    _adapt_double_gain,
    _allocate_requested_shares,
    _control_weights,
    _fit_contrasts,
    _needs_probe,
    _probe_boundary,
    _relative_probe_target,
    _ratio_interval,
    _selected_probe_target,
    _single_bracket_step,
    _updated_single_bracket,
    _updated_target,
)
from zlc_atom.nodes.calibration.calibration import (
    _register_target_sites,
    validate_target_registration,
)
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE


_IMAGING_SEQUENCE = IMAGING_PULSE_RESOURCE.value
_FEEDBACK_PERIOD_IDS = {"load", "short", "gap_1"}
FEEDBACK_PULSE_SEQUENCE = PulseSequence(
    "single_frame_feedback",
    target=_IMAGING_SEQUENCE.target,
    time_step_ns=_IMAGING_SEQUENCE.time_step_ns,
    periods=tuple(
        period
        for period in _IMAGING_SEQUENCE.periods
        if period.period_id in _FEEDBACK_PERIOD_IDS
    ),
    api_parameters=tuple(
        parameter
        for parameter in _IMAGING_SEQUENCE.api_parameters
        if parameter.field_ref.period_id in _FEEDBACK_PERIOD_IDS
    ),
    delays=_IMAGING_SEQUENCE.delays,
)


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

    def __init__(self, run_directory: Path | None = None, cancelled: bool = False) -> None:
        self.cancelled = cancelled
        self.run_directory = run_directory
        self.terminal_sealed = False
        self.progress: list[tuple] = []
        self.commits: list[dict[str, object]] = []
        self.artifacts: dict[str, tuple[Path, str, str]] = {}
        self.partial_exit_writer = None

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

    def register_artifact(self, name, path, *, role, contract_id=""):
        selected = Path(path).resolve()
        assert self.run_directory is not None
        selected.relative_to(self.run_directory.resolve())
        assert selected.is_file()
        self.artifacts[str(name)] = (selected, str(role), str(contract_id))
        return SimpleNamespace(path=selected)

    def register_partial_exit_writer(self, writer) -> None:
        assert self.partial_exit_writer is None
        self.partial_exit_writer = writer


def _wait_host(host: NodeHost, wake: Event):
    deadline = time.monotonic() + 15.0
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
        required_artifacts={"artifact_path": "zlc.slm.science-context"},
        task_name="slm_feedback",
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


def _load_history(path: str | Path) -> list[dict[str, object]]:
    root = Path(path).resolve().parent.parent
    history: list[dict[str, object]] = []
    for checkpoint in sorted((root / "data" / "measurements").glob("measurement-*.npz")):
        with np.load(checkpoint, allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
            for name in archive.files:
                if name != "metadata":
                    metadata[name] = np.asarray(archive[name]).tolist()
        history.append(metadata)
    return history


def _fitted_result(
    contrast: object,
    *,
    valid: object | None = None,
    single_population: object | None = None,
    invalid: object | None = None,
    standard_error: object = 0.0,
) -> dict[str, np.ndarray]:
    values = np.asarray(contrast, dtype=float).reshape(-1)
    sites = len(values)
    fit_valid = (
        np.ones(sites, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool).reshape(sites)
    )
    fit_single = (
        ~fit_valid
        if single_population is None
        else np.asarray(single_population, dtype=bool).reshape(sites)
    )
    fit_invalid = (
        np.zeros(sites, dtype=bool)
        if invalid is None
        else np.asarray(invalid, dtype=bool).reshape(sites)
    )
    fit_valid &= ~fit_invalid
    fit_single &= ~fit_invalid
    error = np.broadcast_to(np.asarray(standard_error, dtype=float), (sites,)).copy()
    error[~fit_valid] = np.nan
    return {
        "contrast": values,
        "standard_error": error,
        "dark_mean": np.full(sites, 10.0),
        "dark_sigma": np.full(sites, 1.0),
        "dark_standard_error": np.full(sites, 0.1),
        "bright_mean": 10.0 + values,
        "bright_sigma": np.full(sites, 1.0),
        "bright_fraction": np.full(sites, 0.5),
        "threshold": 10.0 + 0.5 * values,
        "fidelity": np.full(sites, 0.999),
        "bic_gain": np.full(sites, 100.0),
        "single_mean": np.full(sites, 10.0),
        "single_sigma": np.full(sites, 0.1),
        "valid": fit_valid,
        "single_population": fit_single,
        "invalid": fit_invalid,
    }


def _mixture_samples(
    contrast: object, shots: int, *, noise_scale: float = 0.02
) -> np.ndarray:
    values = np.asarray(contrast, dtype=float).reshape(-1)
    shot = np.arange(int(shots), dtype=float)[:, None]
    site = np.arange(len(values), dtype=float)[None, :]
    occupied = (np.arange(int(shots)) % 2 == 0)[:, None]
    noise = float(noise_scale) * np.sin(1.7 * shot + 0.31 * site)
    return 10.0 + 0.02 * site + noise + occupied * values[None, :]


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
    updates: int = 3,
    probe_factors: tuple[float, ...] = (0.5, 2.0),
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
        pulse_sequence=FEEDBACK_PULSE_SEQUENCE,
        pulse_path=IMAGING_PULSE_RESOURCE.path,
        feedback_mode="qcmos_bright_dark",
        exposure_seconds=0.020,
        shots_per_candidate=shots,
        probe_factors=probe_factors,
        feedback_gain=0.25,
        maximum_weight_change=0.5,
        max_updates=updates,
    )


def test_descriptor_and_direct_update_keep_the_plugin_boundary() -> None:
    descriptors = {item.api_name: item for item in discover_logic_nodes()}
    descriptor = descriptors["slm_feedback"]
    defaults = {
        field.name: field.default for field in descriptor.authoring_schema.fields
    }
    assert defaults["shots_per_candidate"] == 100
    assert defaults["probe_factors"] == (0.5, 2.0)
    assert defaults["feedback_gain"] == pytest.approx(0.25)
    assert defaults["maximum_weight_change"] == pytest.approx(0.5)
    assert defaults["max_updates"] == 12
    assert defaults["feedback_mode"] == "qcmos_bright_dark"
    assert defaults["exposure_seconds"] == pytest.approx(0.1)
    assert defaults["pulse_template"] == ""
    with pytest.raises(ValueError, match="pulse_template"):
        descriptor.authoring_schema.project_values()
    pulse_only = descriptor.authoring_schema.project_values(
        {"pulse_template": "operator-selected.json"}
    )
    assert pulse_only["exposure_seconds"] == pytest.approx(0.1)
    assert pulse_only["probe_factors"] == (0.5, 2.0)
    projected = descriptor.authoring_schema.project_values(
        {
            "pulse_template": "operator-selected.json",
            "probe_factors": "0.25, 4",
        }
    )
    assert projected["probe_factors"] == (0.25, 4.0)
    for invalid in ((), (0.0, 2.0), (0.5, 1.0), (2.0, 2.0)):
        with pytest.raises(ValueError, match="probe"):
            descriptor.authoring_schema.project_values(
                {
                    "pulse_template": "operator-selected.json",
                    "probe_factors": invalid,
                }
            )
    authored = descriptor.authoring_schema.project_values(
        {
            "pulse_template": "operator-selected.json",
            "exposure_seconds": 0.1,
        }
    )
    assert authored["feedback_mode"] == "qcmos_bright_dark"
    assert authored["exposure_seconds"] == pytest.approx(0.1)
    calibration_inputs = descriptors["calibration"].input_specs
    assert calibration_inputs == ()
    assert tuple(item.name for item in descriptor.input_specs) == (
        "calibration_path",
        "science_context_path",
    )
    assert tuple(item.contract_id for item in descriptor.input_specs) == (
        "calibration.readout",
        "zlc.slm.science-context",
    )
    assert tuple(item.field_name for item in descriptor.workspace_resources) == (
        "pulse_template",
    )
    assert tuple(
        (item.name, item.contract_id) for item in descriptor.artifact_outputs
    ) == (("artifact_path", "zlc.slm.science-context"),)
    assert tuple((item.name, item.contract_id) for item in descriptor.outputs) == (
        ("candidate_phase", "slm-feedback.candidate-phase"),
        ("uniformity_history", "slm-feedback.uniformity-history"),
        (
            "observable_uniformity_history",
            "slm-feedback.observable-uniformity-history",
        ),
        ("site_signal_history", "slm-feedback.site-signal-history"),
        ("target_share_history", "slm-feedback.target-share-history"),
    )
    assert tuple(
        (item.output.name, item.plot_kind, item.producer)
        for item in descriptor.node_previews
    ) == (
        ("frames", "image", "camera"),
        ("observable_uniformity_history", "curve", ""),
        ("site_signal_history", "curve", ""),
        ("target_share_history", "curve", ""),
    )
    camera_preview = descriptor.node_previews[0]
    assert camera_preview.semantic == {
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
    contrast = np.linspace(0.6, 1.4, 35)
    updated, correction, slope, decision = _updated_target(
        target,
        contrast,
        np.zeros(35),
        np.ones(35, dtype=bool),
        rows,
        columns,
        reference_valid=np.ones(35, dtype=bool),
        previous_weights=np.full(35, np.nan),
        previous_contrast=np.full(35, np.nan),
        feedback_gain=0.25,
        maximum_weight_change=0.5,
    )
    assert updated[rows[0], columns[0]] < updated[rows[-1], columns[-1]]
    assert correction[0] < 0.0 < correction[-1]
    assert np.all(np.isnan(slope))
    assert set(decision) == {"feedback_assumed_slope"}
    multipliers = updated[rows, columns] / target[rows, columns]
    assert float(np.max(multipliers) / np.min(multipliers)) <= np.exp(0.4)
    standard_error = 0.02 * contrast
    estimate, lower, upper, max_relative_sem = _ratio_interval(
        contrast, standard_error
    )
    assert lower <= estimate <= upper
    assert estimate == pytest.approx(1.4 / 0.6)
    assert max_relative_sem == pytest.approx(0.02)


def test_double_gain_adapts_only_after_significant_formal_evidence() -> None:
    valid = np.ones(3, dtype=bool)
    zero = np.zeros(3)
    gain, streak, action, common, previous, current = _adapt_double_gain(
        (2.0, 1.0, 1.0), zero, valid,
        (1.6, 1.0, 1.0), zero, valid,
        gain=0.25,
        improvement_streak=0,
    )
    assert (gain, streak, action, common) == (
        0.25, 1, "hold_first_significant_improvement", 3
    )
    assert previous == pytest.approx(2.0)
    assert current == pytest.approx(1.6)
    gain, streak, action, *_ = _adapt_double_gain(
        (1.6, 1.0, 1.0), zero, valid,
        (1.3, 1.0, 1.0), zero, valid,
        gain=gain,
        improvement_streak=streak,
    )
    assert gain == pytest.approx(0.3125)
    assert streak == 1 and action == "increase_after_continuous_improvement"
    gain, streak, action, *_ = _adapt_double_gain(
        (1.3, 1.0, 1.0), zero, valid,
        (1.8, 1.0, 1.0), zero, valid,
        gain=gain,
        improvement_streak=streak,
    )
    assert gain == pytest.approx(0.15625)
    assert streak == 0 and action == "decrease_after_worsening"


def test_single_population_classification_and_baseline_relative_probe_selection() -> None:
    rng = np.random.default_rng(17)
    occupied = rng.random(500) < 0.45
    samples = np.column_stack(
        (
            np.where(
                occupied,
                rng.normal(32.0, 2.0, 500),
                rng.normal(10.0, 1.5, 500),
            ),
            rng.normal(10.0, 1.5, 500),
        )
    )
    fitted = _fit_contrasts(samples)
    assert fitted["valid"].tolist() == [True, False]
    assert fitted["single_population"].tolist() == [False, True]
    assert fitted["invalid"].tolist() == [False, False]
    assert fitted["bic_gain"][0] > 0.0
    assert fitted["bic_gain"][1] < 0.0
    assert fitted["contrast"][0] == pytest.approx(22.0, rel=0.04)
    assert np.isnan(fitted["contrast"][1])
    assert np.isnan(fitted["standard_error"][1])

    target = _grid_target((17, 23))
    rows, columns = np.nonzero(target)
    probe = np.zeros(35, dtype=bool)
    probe[17] = True
    requested = np.ones(35)
    requested[17] = 2.0
    higher, effective = _relative_probe_target(
        target,
        requested,
        probe,
        rows,
        columns,
    )
    before_share = target[rows[17], columns[17]] / np.sum(target[rows, columns])
    after_share = higher[rows[17], columns[17]] / np.sum(
        higher[rows, columns]
    )
    assert after_share / before_share == pytest.approx(2.0)
    assert effective[17] == pytest.approx(2.0)
    nonprobe_ratio = higher[rows[~probe], columns[~probe]] / target[
        rows[~probe], columns[~probe]
    ]
    assert np.ptp(nonprobe_ratio) == pytest.approx(0.0)
    assert np.sum(higher[rows, columns]) == pytest.approx(
        np.sum(target[rows, columns])
    )

    crowded = np.zeros(35, dtype=bool)
    crowded[:30] = True
    crowded_requested = np.ones(35)
    crowded_requested[crowded] = 2.0
    _crowded_target, crowded_effective = _relative_probe_target(
        target,
        crowded_requested,
        crowded,
        rows,
        columns,
    )
    assert 1.0 < crowded_effective[0] < 2.0
    assert np.all(crowded_effective[crowded] == crowded_effective[0])

    selected_sites = np.zeros(35, dtype=bool)
    selected_sites[:4] = True
    baseline_contrast = np.ones(35)
    baseline_valid = ~selected_sites
    lower_contrast = np.full(35, np.nan)
    lower_contrast[[0, 2]] = (0.8, 0.8)
    lower_valid = np.zeros(35, dtype=bool)
    lower_valid[[0, 2]] = True
    upper_contrast = np.full(35, np.nan)
    upper_contrast[[1, 2]] = (1.1, 1.3)
    upper_valid = np.zeros(35, dtype=bool)
    upper_valid[[1, 2]] = True
    combined, selected, decisions, observed_factors = _selected_probe_target(
        target,
        selected_sites,
        baseline_contrast,
        baseline_valid,
        [
            (0.5, lower_contrast, np.full(35, 0.01), lower_valid),
            (2.0, upper_contrast, np.full(35, 0.01), upper_valid),
        ],
        rows,
        columns,
    )
    np.testing.assert_allclose(
        selected[:4], (0.5, 2.0, 0.5, 1.0)
    )
    np.testing.assert_allclose(observed_factors[:4], (0.5, 2.0, 0.5, 1.0))
    assert decisions[:4].tolist() == [
        "probe_choose_lower_only",
        "probe_choose_upper_only",
        "probe_choose_lower_closest",
        "probe_hold_unobservable",
    ]
    assert not np.array_equal(combined, target)



    np.testing.assert_allclose(
        _probe_boundary(
            np.asarray((1.0, 2.0, 1.0)),
            np.asarray((2.0, 1.0, 1.0)),
            np.asarray((1.0, -1.0, 1.0)),
        )[:2],
        (np.sqrt(2.0), np.sqrt(2.0)),
    )
    tightened = _probe_boundary(
        np.asarray((1.2, 1.8)),
        np.asarray((1.8, 1.2)),
        np.asarray((1.0, -1.0)),
    )
    np.testing.assert_allclose(tightened, np.sqrt((2.16, 2.16)))

    first_midpoint = _probe_boundary(
        np.asarray((1.0, 2.0)),
        np.asarray((2.0, 1.0)),
        np.asarray((1.0, -1.0)),
    )
    first_step = _single_bracket_step(
        np.asarray((1.0, 2.0)),
        first_midpoint,
        np.asarray((1.0, -1.0)),
        0.5,
    )
    np.testing.assert_allclose(first_step, (np.log(np.sqrt(2.0)), -np.log(np.sqrt(2.0))))
    second_midpoint = _probe_boundary(
        first_midpoint,
        np.asarray((2.0, 1.0)),
        np.asarray((1.0, -1.0)),
    )
    second_step = _single_bracket_step(
        first_midpoint,
        second_midpoint,
        np.asarray((1.0, -1.0)),
        0.5,
    )
    assert second_step[0] > 0.0 > second_step[1]
    flipped_single, flipped_observable, flipped_direction, crossed = (
        _updated_single_bracket(
            np.asarray((2.5, 0.5)),
            np.ones(2, dtype=bool),
            np.asarray((1.0, 2.0)),
            np.asarray((2.0, 1.0)),
            np.asarray((1.0, -1.0)),
        )
    )
    assert crossed.tolist() == [True, True]
    np.testing.assert_allclose(flipped_single, (2.5, 0.5))
    np.testing.assert_allclose(flipped_observable, (2.0, 1.0))
    np.testing.assert_allclose(flipped_direction, (-1.0, 1.0))
    flipped_midpoint = _probe_boundary(
        flipped_single, flipped_observable, flipped_direction
    )
    flipped_step = _single_bracket_step(
        np.asarray((2.5, 0.5)),
        flipped_midpoint,
        flipped_direction,
        0.5,
    )
    assert flipped_step[0] < 0.0 < flipped_step[1]


def test_direction_preserving_share_allocator_moves_only_balanced_requested_power() -> None:
    shares = np.asarray((0.4, 0.3, 0.2, 0.1))
    requested = np.log((1.5, 0.5, 1.0, 1.0))
    allocated, transfer, increase_scale, decrease_scale = (
        _allocate_requested_shares(shares, requested)
    )
    np.testing.assert_allclose(allocated, (0.55, 0.15, 0.2, 0.1))
    assert transfer == pytest.approx(0.15)
    assert increase_scale == pytest.approx(0.75)
    assert decrease_scale == pytest.approx(1.0)
    assert np.sum(allocated) == pytest.approx(1.0)
    np.testing.assert_array_equal(
        np.sign(allocated - shares), np.sign(requested)
    )

    one_sided, transfer, *_scales = _allocate_requested_shares(
        shares, np.log((1.1, 1.2, 1.0, 1.0))
    )
    np.testing.assert_array_equal(one_sided, shares)
    assert transfer == 0.0

    bounded, transfer, *_scales = _allocate_requested_shares(
        np.asarray((0.5, 0.5)),
        np.log((2.0, 0.5)),
        upper=np.asarray((0.6, np.inf)),
    )
    np.testing.assert_allclose(bounded, (0.6, 0.4))
    assert transfer == pytest.approx(0.1)

    target = np.asarray(((0.4, 0.3), (0.2, 0.1)), dtype=np.float32)
    rows, columns = np.nonzero(target)
    restored, applied_step, *_details = _updated_target(
        target,
        np.full(4, np.nan),
        np.zeros(4),
        np.zeros(4, dtype=bool),
        rows,
        columns,
        reference_valid=np.zeros(4, dtype=bool),
        previous_weights=np.full(4, np.nan),
        previous_contrast=np.full(4, np.nan),
        feedback_gain=0.25,
        maximum_weight_change=0.5,
        directed_log_step=np.log((1.25, 0.5, 1.0, 1.0)),
        control_boundary=np.asarray((2.0, np.nan, np.nan, np.nan)),
        control_direction=np.asarray((1.0, 0.0, 0.0, 0.0)),
    )
    np.testing.assert_allclose(restored[rows, columns], (0.5, 0.2, 0.2, 0.1))
    np.testing.assert_allclose(
        np.exp(applied_step), (1.25, 2.0 / 3.0, 1.0, 1.0)
    )

    ordinary, ordinary_step, *_details = _updated_target(
        target,
        np.asarray((0.1, 10.0, 10.0, 10.0)),
        np.zeros(4),
        np.ones(4, dtype=bool),
        rows,
        columns,
        reference_valid=np.ones(4, dtype=bool),
        previous_weights=np.full(4, np.nan),
        previous_contrast=np.full(4, np.nan),
        feedback_gain=0.25,
        maximum_weight_change=0.5,
        control_boundary=np.asarray((1.4, np.nan, np.nan, np.nan)),
        control_direction=np.asarray((1.0, 0.0, 0.0, 0.0)),
    )
    assert ordinary_step[0] < 0.0
    assert ordinary[rows[0], columns[0]] == pytest.approx(0.35)

    rng = np.random.default_rng(91)
    for _ in range(100):
        current = rng.random(12)
        current /= np.sum(current)
        steps = rng.uniform(-0.4, 0.4, 12)
        steps[rng.random(12) < 0.25] = 0.0
        result, *_details = _allocate_requested_shares(current, steps)
        assert np.sum(result) == pytest.approx(1.0)
        assert np.all(result > 0.0)
        moved = result - current
        assert np.all((moved == 0.0) | (np.sign(moved) == np.sign(steps)))
        np.testing.assert_array_equal(result[steps == 0.0], current[steps == 0.0])


def test_probe_admission_excludes_observable_invalid_and_formal_history_sites() -> None:
    single = np.ones(4, dtype=bool)
    observable = np.asarray((False, True, False, False))
    acquisition_invalid = np.asarray((False, False, True, False))
    previous_weight = np.asarray((np.nan, np.nan, np.nan, 1.0))
    previous_contrast = np.asarray((np.nan, np.nan, np.nan, 2.0))
    assert _needs_probe(
        single,
        observable,
        acquisition_invalid,
        previous_weight,
        previous_contrast,
    ).tolist() == [True, False, False, False]


def test_feedback_applies_science_context_then_measures_before_solving_update(
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
    frozen_target = _grid_target(shape)
    science_context = _science_context(
        slm,
        target=frozen_target,
        pattern=pattern,
        wavefront=wavefront,
        pupil=pupil,
    )
    science_context = {
        **science_context,
        "command_receipt": {
            **science_context["command_receipt"],
            "outcome": "unknown",
        },
    }
    external = canonical_phase(np.full(shape, 1.25), shape)
    slm.apply_phase(external)
    command_count = len(slm.commands)
    solved_pattern = canonical_phase(pattern.astype(float) + 0.05, shape)
    first_contrast = np.concatenate(([2.0], np.ones(34)))
    expected_target, *_details = _updated_target(
        frozen_target,
        first_contrast,
        np.zeros(35),
        np.ones(35, dtype=bool),
        *np.nonzero(frozen_target),
        reference_valid=np.ones(35, dtype=bool),
        previous_weights=np.full(35, np.nan),
        previous_contrast=np.full(35, np.nan),
        feedback_gain=0.25,
        maximum_weight_change=0.5,
    )

    def solve(target, **kwargs):
        np.testing.assert_allclose(target, expected_target)
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
    fits = iter((_fitted_result(first_contrast), _fitted_result(np.ones(35))))
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: next(fits),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
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

    with pytest.raises(ValueError, match="no frozen Target"):
        build(
            selected_context={**science_context, "target_intensity": None}
        )
    generic = _calibration()
    build(selected_calibration=generic)
    calibration_without_dark_history = replace(
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
    build(selected_calibration=calibration_without_dark_history)
    task = build()
    try:
        result = task.execute(_Context(tmp_path))
        artifact = load_science_context(result["artifact_path"])
        expected = canonical_phase(
            solved_pattern.astype(float) + wavefront.astype(float), shape
        )
        np.testing.assert_array_equal(artifact["pattern_phase"], solved_pattern)
        np.testing.assert_array_equal(artifact["operator_wavefront"], wavefront)
        np.testing.assert_array_equal(artifact["pupil_amplitude"], pupil)
        np.testing.assert_allclose(artifact["target_intensity"], expected_target)
        np.testing.assert_array_equal(artifact["phase"], expected)
        np.testing.assert_array_equal(slm.last_commanded_phase, expected)
        assert artifact["pattern_metadata"]["solver"]["iterations_run"] == 19
        assert artifact["pattern_metadata"]["feedback_mode"] == "qcmos_bright_dark"
        assert artifact["pattern_metadata"]["probe_factors"] == [0.5, 2.0]
        assert artifact["pattern_metadata"]["feedback_gain"] == pytest.approx(0.25)
        assert artifact["pattern_metadata"]["maximum_weight_change"] == pytest.approx(0.5)
        assert artifact["pattern_metadata"]["exposure_seconds"] == pytest.approx(0.020)
        assert artifact["command_receipt"]["outcome"] == "known-new"
        np.testing.assert_array_equal(slm.commands[command_count], incoming)
        np.testing.assert_array_equal(slm.commands[command_count + 1], expected)
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
    target_contrast = np.asarray([0.7, 1.4, 0.9, 1.2, 0.8, 1.1])
    fits = iter(
        (_fitted_result(target_contrast), _fitted_result(np.ones(len(rows))))
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
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: next(fits),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            np.zeros((self.shots, len(rows))),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
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
        result = task.execute(_Context(tmp_path))
        expected, *_details = _updated_target(
            target,
            target_contrast,
            np.zeros(len(rows)),
            np.ones(len(rows), dtype=bool),
            rows,
            columns,
            reference_valid=np.ones(len(rows), dtype=bool),
            previous_weights=np.full(len(rows), np.nan),
            previous_contrast=np.full(len(rows), np.nan),
            feedback_gain=0.25,
            maximum_weight_change=0.5,
        )
        np.testing.assert_allclose(solved_targets[0], expected)
        history = _load_history(result["artifact_path"])
        np.testing.assert_allclose(
            history[1]["target_weight"], expected[rows, columns]
        )
        np.testing.assert_allclose(
            history[0]["bright_minus_dark"],
            target_contrast,
        )
    finally:
        plane.close()


def test_uniformity_history_is_one_latest_curve_paired_with_candidate_phase(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23))
    plane = SignalDataPlane()
    context = _Context(tmp_path)
    contrasts = iter(
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
    phases = iter((0.25, 0.5))
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (
            np.full(slm.shape_yx, next(phases), dtype=np.float32),
            {"method": "test"},
        ),
    )
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: _fitted_result(next(contrasts)),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, run_context, iteration: (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
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
        assert result["feedback_status"] == "completed"
        output = context.commits[-1]["uniformity_history"]
        np.testing.assert_allclose(output.snapshot.block.values[0, :2, 0], (4.0, 2.0))
        assert output.snapshot.block.values[0, 2, 0] <= 1.10
        assert len(context.commits) == 7
        assert not np.any(
            context.commits[0]["uniformity_history"]
            .snapshot.expanded_validity()
        )
        assert (
            context.commits[0]["uniformity_history"].snapshot.block.schema
            is context.commits[1]["uniformity_history"].snapshot.block.schema
        )
        assert set(context.commits[-1]) == {
            "candidate_phase",
            "uniformity_history",
            "observable_uniformity_history",
            "site_signal_history",
            "target_share_history",
        }
        column = output.snapshot.block.schema.point_table.columns[0]
        assert column.name == "candidate"
        assert tuple(column.values) == (1, 2, 3, 4, 5, 6, 7)
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
        del ordinal, exposure
        return fluorescence.reshape(5, 7)

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
            camera.trigger(int(cycles))

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
        calibration=_calibration(),
    )
    try:
        from zlc_atom.install import create_installation

        installation = create_installation("virtual")
        try:
            pulse = resolve_pulse(
                FEEDBACK_PULSE_SEQUENCE,
                path=IMAGING_PULSE_RESOURCE.path,
                board=installation.device("sequencer").describe(),
                api_values={},
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
        samples, saturated, missing, _mean = task._measure(pulse, _Context(), 0)
        assert lookup_count == 1
        monkeypatch.setattr(plane, "current_dataset", current_dataset)
        np.testing.assert_allclose(samples, np.broadcast_to(fluorescence, (10, 35)))
        assert not saturated
        assert not missing
        assert samples[0, 17] == fluorescence[17]
        assert task._actual_exposure_seconds == pytest.approx(0.020)
        assert camera.working_point().exposure_seconds == pytest.approx(0.020)
        assert sequencer.fires == [10]
        assert armed_buffer_sizes == [10]
        signal = "@logic/slm_feedback/camera/frames"
        raw = plane.current_dataset(signal)
        assert raw.block.values.shape == (10, 1, 5, 7)
        np.testing.assert_allclose(
            raw.block.values[:, 0],
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
                raw.block.values[:, 0].mean(axis=0),
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
        slm.apply_phase(np.full(slm.shape_yx, 0.25, dtype=np.float32))
        partial_samples, _saturated, partial_missing, _mean = task._measure(
            pulse, _Context(), 1
        )
        assert partial_missing == (0,)
        assert np.all(np.isfinite(partial_samples[:2, 0]))
        assert np.all(np.isnan(partial_samples[2:, 0]))
        assert not camera.capture_state()
    finally:
        plane.close()
        camera.close()


def test_electron_measurement_uses_current_conversion_and_saturation(
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
            self.camera.trigger(int(cycles))

        def wait_done(self, timeout=None):
            return SimpleNamespace(fault=None)

        def safe(self):
            return None

    installation = create_installation("virtual")
    try:
        pulse = resolve_pulse(
            FEEDBACK_PULSE_SEQUENCE,
            path=IMAGING_PULSE_RESOURCE.path,
            board=installation.device("sequencer").describe(),
            api_values={},
        )
    finally:
        installation.close()

    def run(offset, scale, effective_photoelectrons):
        def frame_source(ordinal, exposure):
            del ordinal, exposure
            image = np.zeros((5, 7), dtype="<u2")
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
            _samples, saturated, missing, _mean = task._measure(pulse, _Context(), 0)
            assert saturated == (17,) and not missing
            assert task._effective_photoelectrons is effective_photoelectrons
        finally:
            plane.close()
            camera.close()

    run(recorded_offset, recorded_scale, True)
    run(None, None, False)
    run(recorded_offset, 0.6, True)


def test_single_population_sites_probe_both_sides_then_measure_combined_target(
    tmp_path: Path, monkeypatch
) -> None:
    target = _asymmetric_target()
    slm = _Slm(target.shape)
    context_mapping = _science_context(slm, target=target)
    calibration = _calibration_with_unresolved_site(
        target,
        missing=17,
    )
    measured_phases: list[np.ndarray] = []
    solved_targets: list[np.ndarray] = []

    def solve(candidate, **_kwargs):
        solved_targets.append(np.array(candidate, copy=True))
        return np.full(
            candidate.shape, 0.1 * (len(solved_targets) + 1), np.float32
        ), {}

    monkeypatch.setattr(feedback_module, "solve_phase", solve)
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )

    valid = np.ones(35, dtype=bool)
    valid[[17, 18]] = False
    single = np.zeros(35, dtype=bool)
    single[[17, 18]] = True
    lower_valid = np.ones(35, dtype=bool)
    lower_valid[18] = False
    lower_single = np.zeros(35, dtype=bool)
    lower_single[18] = True
    upper_valid = np.ones(35, dtype=bool)
    upper_valid[17] = False
    upper_single = np.zeros(35, dtype=bool)
    upper_single[17] = True
    baseline_values = np.linspace(0.8, 1.2, 35)
    baseline_values[~valid] = np.nan
    lower_values = np.linspace(4.0, 2.0, 35)
    lower_values[~lower_valid] = np.nan
    upper_values = np.linspace(0.2, 0.4, 35)
    upper_values[~upper_valid] = np.nan
    fits = iter(
        (
            _fitted_result(
                baseline_values,
                valid=valid,
                single_population=single,
            ),
            _fitted_result(
                lower_values,
                valid=lower_valid,
                single_population=lower_single,
            ),
            _fitted_result(
                upper_values,
                valid=upper_valid,
                single_population=upper_single,
            ),
            _fitted_result(np.ones(35)),
        )
    )

    def measure(self, pulse, context, iteration):
        measured_phases.append(np.array(self.slm.last_commanded_phase, copy=True))
        return (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        )

    monkeypatch.setattr(SlmFeedbackTask, "_measure", measure)
    monkeypatch.setattr(
        SlmFeedbackTask, "_save_figures", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: next(fits),
    )
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        calibration=calibration,
        science_context=context_mapping,
        updates=1,
    )
    try:
        result = task.execute(_Context(tmp_path))
        assert result["feedback_status"] == "completed"
        assert len(measured_phases) == 4
        assert all(
            not np.array_equal(left, right)
            for left, right in zip(
                measured_phases[:-1], measured_phases[1:], strict=True
            )
        )
        rows, columns = np.nonzero(target > 0.0)
        baseline = target[rows, columns]
        np.testing.assert_allclose(
            solved_targets[0][rows[[17, 18]], columns[[17, 18]]] / baseline[[17, 18]],
            (0.5, 0.5),
        )
        np.testing.assert_allclose(
            solved_targets[1][rows[[17, 18]], columns[[17, 18]]] / baseline[[17, 18]],
            (2.0, 2.0),
        )
        requested = np.ones(35)
        requested[[17, 18]] = (0.5, 2.0)
        selected_target, _effective = _relative_probe_target(
            target,
            requested,
            single,
            rows,
            columns,
        )
        expected, *_details = _updated_target(
            selected_target,
            baseline_values,
            np.zeros(35),
            valid,
            rows,
            columns,
            reference_valid=valid,
            previous_weights=np.full(35, np.nan),
            previous_contrast=np.full(35, np.nan),
            feedback_gain=0.25,
            maximum_weight_change=0.5,
            directed_log_step=np.zeros(35),
        )
        np.testing.assert_allclose(solved_targets[2], expected)
        history = _load_history(result["artifact_path"])
        assert [item["candidate_kind"] for item in history] == [
            "baseline", "probe", "probe", "probe_combined"
        ]
        assert history[1]["probe_requested_factor"] == pytest.approx(0.5)
        assert history[1]["probe_group_effective_factor"] == pytest.approx(0.5)
        assert history[2]["probe_requested_factor"] == pytest.approx(2.0)
        assert history[2]["probe_group_effective_factor"] == pytest.approx(2.0)
        np.testing.assert_allclose(
            history[1]["probe_effective_factor"],
            solved_targets[0][rows, columns] / baseline,
        )
        np.testing.assert_allclose(
            history[2]["probe_effective_factor"],
            solved_targets[1][rows, columns] / baseline,
        )
        selected_formal = np.asarray(
            history[3]["probe_selected_formal_factor"], dtype=float
        )
        assert np.all(np.isnan(selected_formal[~single]))
        expected_factors = (
            _control_weights(expected[rows, columns])
            / _control_weights(target[rows, columns])
        )
        np.testing.assert_allclose(
            selected_formal[single], expected_factors[single]
        )
        assert selected_formal[17] < 1.0 < selected_formal[18]
        assert history[3]["probe_decision"][17] == "probe_choose_lower_only"
        assert history[3]["probe_decision"][18] == "probe_choose_upper_only"
        assert history[0]["decision"][17] == "hold_for_probe"
        assert history[0]["decision"][0] == "feedback_assumed_slope"
        assert np.isnan(history[3]["previous_double_control_weight"][17])
        assert history[3]["previous_double_control_weight"][0] == pytest.approx(1.0)
        _phase, metadata = _load_candidate(result["artifact_path"])
        assert metadata["candidate"] == 4
    finally:
        plane.close()


def test_baseline_single_with_formal_history_steps_to_bracket_midpoint_without_probe(
    tmp_path: Path, monkeypatch
) -> None:
    target = _grid_target((17, 23))
    rows, columns = np.nonzero(target > 0.0)
    slm = _Slm(target.shape)
    context_mapping = _science_context(slm, target=target)
    previous_weights = np.ones(35)
    previous_weights[17] = 1.2
    context_mapping["pattern_metadata"] = {
        "feedback_controller": "slm-feedback.qcmos-bright-dark",
        "feedback_mode": "qcmos_bright_dark",
        "pulse_path": str(Path(IMAGING_PULSE_RESOURCE.path).resolve()),
        "exposure_seconds": 0.020,
        "probe_factors": [0.5, 2.0],
        "feedback_gain": 0.25,
        "maximum_weight_change": 0.5,
        "measurement": {
            "previous_double_control_weight": previous_weights.tolist(),
            "previous_double_bright_minus_dark": np.ones(35).tolist(),
        },
    }
    valid = np.ones(35, dtype=bool)
    valid[17] = False
    single = np.zeros(35, dtype=bool)
    single[17] = True
    baseline_contrast = np.ones(35)
    baseline_contrast[0] = 0.1
    baseline_error = 0.24 * baseline_contrast
    baseline_error[0] = 0.0
    fits = iter((
        _fitted_result(
            np.where(valid, baseline_contrast, np.nan),
            valid=valid,
            single_population=single,
            standard_error=baseline_error,
        ),
        _fitted_result(np.ones(35)),
    ))
    solved_targets: list[np.ndarray] = []
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda candidate, **_kwargs: (
            solved_targets.append(np.array(candidate, copy=True))
            or np.full(candidate.shape, 0.2, dtype=np.float32),
            {},
        ),
    )
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: next(fits),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        SlmFeedbackTask, "_save_figures", lambda *args, **kwargs: None
    )
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        calibration=_calibration_at(
            np.column_stack((columns, rows)), shape=target.shape
        ),
        science_context=context_mapping,
        updates=1,
    )
    try:
        result = task.execute(_Context(tmp_path))
        history = _load_history(result["artifact_path"])
        assert [item["candidate_kind"] for item in history] == [
            "baseline", "ordinary"
        ]
        assert history[0]["probe_sites"] == []
        assert history[0]["decision"][17] == "single_bracket_midpoint"
        assert _control_weights(
            solved_targets[0][rows, columns]
        )[17] == pytest.approx(np.sqrt(1.2))
    finally:
        plane.close()


def test_probe_combined_counts_as_formal_update_and_reuses_episode_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    target = _asymmetric_target()
    slm = _Slm(target.shape)
    solved_targets: list[np.ndarray] = []
    solve_inputs: list[tuple[np.ndarray, dict[str, object]]] = []
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    def solve(candidate, **kwargs):
        solved_targets.append(np.array(candidate, copy=True))
        solve_inputs.append((
            np.array(kwargs["initial_phase"], copy=True),
            dict(kwargs["spot_optimizer_state"]),
        ))
        kwargs["spot_optimizer_state"]["marker"] = len(solved_targets)
        return np.full(
            candidate.shape, 0.1 * (len(solved_targets) + 1), np.float32
        ), {}

    monkeypatch.setattr(feedback_module, "solve_phase", solve)
    valid = np.ones(35, dtype=bool)
    valid[17] = False
    single = np.zeros(35, dtype=bool)
    single[17] = True
    baseline_contrast = np.ones(35)
    baseline_contrast[0] = 0.1
    baseline_error = 0.24 * baseline_contrast
    baseline_error[0] = 0.0
    single_fit = lambda: _fitted_result(
        np.where(valid, baseline_contrast, np.nan),
        valid=valid,
        single_population=single,
        standard_error=baseline_error,
    )
    combined_valid = np.array(valid, copy=True)
    combined_valid[18] = False
    combined_single = ~combined_valid
    formal_double_contrast = 100.0 * np.exp(np.linspace(-0.15, 0.15, 35))
    formal_double_contrast[17] = 80.0

    def after_double_single_fit() -> dict[str, np.ndarray]:
        current_valid = np.ones(35, dtype=bool)
        current_valid[17] = False
        current_single = ~current_valid
        return _fitted_result(
            np.where(current_valid, formal_double_contrast, np.nan),
            valid=current_valid,
            single_population=current_single,
        )

    fits = iter(
        (
            single_fit(),
            single_fit(),
            _fitted_result(np.ones(35)),
            _fitted_result(
                np.where(combined_valid, formal_double_contrast, np.nan),
                valid=combined_valid,
                single_population=combined_single,
            ),
                _fitted_result(formal_double_contrast),
                after_double_single_fit(),
                after_double_single_fit(),
                after_double_single_fit(),
                after_double_single_fit(),
                after_double_single_fit(),
            )
    )
    monkeypatch.setattr(
        feedback_module, "_fit_contrasts", lambda samples: next(fits)
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        SlmFeedbackTask, "_save_figures", lambda *args, **kwargs: None
    )
    plane = SignalDataPlane()
    science_context = _science_context(slm, target=target)
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=target,
        calibration=_calibration_with_unresolved_site(target, missing=17),
        science_context=science_context,
        updates=3,
    )
    try:
        result = task.execute(_Context(tmp_path))
        assert result["feedback_status"] == "completed"
        for initial_phase, state in solve_inputs[:3]:
            np.testing.assert_array_equal(
                initial_phase, science_context["pattern_phase"]
            )
            assert state == {}
        for initial_phase, state in solve_inputs[3:6]:
            np.testing.assert_array_equal(
                initial_phase, np.full(target.shape, 0.4, dtype=np.float32)
            )
            assert state == {"marker": 3}
        for initial_phase, state in solve_inputs[6:9]:
            np.testing.assert_array_equal(
                initial_phase, np.full(target.shape, 0.7, dtype=np.float32)
            )
            assert state == {"marker": 6}
        history = _load_history(result["artifact_path"])
        assert [item["candidate_kind"] for item in history] == [
            "baseline", "probe", "probe", "probe_combined",
            "probe", "probe", "probe_combined",
            "probe", "probe", "probe_combined",
        ]
        assert [item["formal_updates_applied"] for item in history] == [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3
        ]
        assert sum(item["candidate_kind"] == "probe" for item in history) == 6
        assert len(history) == 1 + task.max_updates * 3
    finally:
        plane.close()


def test_all_single_population_sites_stall_at_baseline_without_fake_probe(
    tmp_path: Path, monkeypatch
) -> None:
    target = _asymmetric_target()
    slm = _Slm(target.shape)
    measured: list[np.ndarray] = []
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: pytest.fail("relative all-site probe must not solve"),
    )
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples: _fitted_result(
            np.full(35, np.nan),
            valid=np.zeros(35, dtype=bool),
            single_population=np.ones(35, dtype=bool),
        ),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            measured.append(np.array(self.slm.last_commanded_phase, copy=True)),
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        )[1:],
    )
    monkeypatch.setattr(
        SlmFeedbackTask, "_save_figures", lambda *args, **kwargs: None
    )
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=target,
        calibration=_calibration_with_unresolved_site(target, missing=17),
        updates=1,
    )
    try:
        result = task.execute(_Context(tmp_path))
        assert result["feedback_status"] == "stalled"
        assert len(measured) == 1
        history = _load_history(result["artifact_path"])
        assert len(history) == 1
        assert history[0]["candidate_kind"] == "baseline"
        assert history[0]["next_phase_changed"] is False
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["outcome"]["reason"] == (
            "relative SLM probe has no non-probe power reservoir"
        )
        assert summary["probe_candidates"] == []
    finally:
        plane.close()


def test_unchanged_solved_phase_stops_without_a_second_shot_batch(
    tmp_path: Path, monkeypatch
) -> None:
    target = _asymmetric_target()
    slm = _Slm(target.shape)
    plane = SignalDataPlane()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda candidate, **kwargs: (
            np.array(slm.last_commanded_phase, copy=True),
            {},
        ),
    )
    valid = np.ones(35, dtype=bool)
    valid[17] = False
    single = np.zeros(35, dtype=bool)
    single[17] = True
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: _fitted_result(
            np.where(valid, 1.0, np.nan),
            valid=valid,
            single_population=single,
        ),
    )
    calls = 0

    def measure(self, pulse, context, iteration):
        nonlocal calls
        calls += 1
        return (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        )

    monkeypatch.setattr(SlmFeedbackTask, "_measure", measure)
    monkeypatch.setattr(
        SlmFeedbackTask, "_save_figures", lambda *args, **kwargs: None
    )
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=target,
        calibration=_calibration_with_unresolved_site(target, missing=17),
    )
    try:
        result = task.execute(_Context(tmp_path))
        assert calls == 1
        assert result["feedback_status"] == "stalled"
        _phase, metadata = _load_candidate(result["artifact_path"])
        assert "no different phase" in metadata["outcome"]["reason"]
    finally:
        plane.close()


def test_persistently_single_site_probes_both_sides_then_restores_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    target = _asymmetric_target()
    slm = _Slm(target.shape)
    calibration = _calibration_with_unresolved_site(
        target,
        missing=17,
    )
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    context = _Context(tmp_path)
    solved_targets: list[np.ndarray] = []

    def solve(candidate, **_kwargs):
        solved_targets.append(np.array(candidate, copy=True))
        return np.full(
            candidate.shape, 0.1 * (len(solved_targets) + 1), np.float32
        ), {}

    monkeypatch.setattr(feedback_module, "solve_phase", solve)
    measured_phases: list[np.ndarray] = []
    valid = np.ones(35, dtype=bool)
    valid[17] = False
    single = np.zeros(35, dtype=bool)
    single[17] = True
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: _fitted_result(
            np.where(valid, 1.0, np.nan),
            valid=valid,
            single_population=single,
        ),
    )

    def measure(self, pulse, run_context, iteration):
        measured_phases.append(np.array(self.slm.last_commanded_phase, copy=True))
        return (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        )

    monkeypatch.setattr(SlmFeedbackTask, "_measure", measure)
    monkeypatch.setattr(
        SlmFeedbackTask, "_save_figures", lambda *args, **kwargs: None
    )
    plane = SignalDataPlane()
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=target,
        calibration=calibration,
        updates=4,
    )
    try:
        result = task.execute(context)
        artifact = load_science_context(result["artifact_path"])
        metadata = artifact["pattern_metadata"]
        assert result["feedback_status"] == "stalled"
        assert len(measured_phases) == 3
        assert all(
            not np.array_equal(left, right)
            for left, right in zip(measured_phases, measured_phases[1:])
        )
        assert metadata["candidate"] == 1
        np.testing.assert_array_equal(artifact["phase"], measured_phases[0])
        np.testing.assert_array_equal(slm.last_commanded_phase, measured_phases[0])
        np.testing.assert_array_equal(artifact["target_intensity"], target)
        history = _load_history(result["artifact_path"])
        assert [item["candidate_kind"] for item in history] == [
            "baseline", "probe", "probe"
        ]
        assert metadata["outcome"]["reason"] == (
            "two-sided SLM probes supplied no observable direction; "
            "baseline restored"
        )
    finally:
        plane.close()


def test_virtual_feedback_recovers_missing_sites_and_retains_best_candidate(
    tmp_path: Path,
) -> None:
    plane = SignalDataPlane()
    descriptors = {item.api_name: item for item in discover_logic_nodes()}
    calibration_installation = create_installation("virtual")
    calibration_camera = calibration_installation.device("camera")
    calibration_sequencer = calibration_installation.device("sequencer")
    calibration_slm = calibration_installation.device("slm")
    installation = None
    try:
        target = preset_grid(calibration_slm.shape_yx, (5, 7))
        pattern, _metadata = solve_phase(
            target, objective_kind="spots", iterations=None
        )
        calibration_slm.apply_phase(pattern)
        calibration_node = descriptors["calibration"].instantiate(
            camera=calibration_camera,
            camera_key="camera",
            sequencer=calibration_sequencer,
            sequencer_key="sequencer",
            signal_plane=plane,
            pulse_resource=IMAGING_PULSE_RESOURCE,
            repeats=30,
        )
        calibration_result = calibration_node.run(tmp_path)
        calibration = TrapCalibration.load(calibration_result.artifact_path)
        assert calibration.site_map.n_sites == 25
        assert calibration.site_map.topology is None
        box = calibration.select_model(ReadoutModelKind.BOX)
        assert box.integration_half_width == 1 and box.reducer == "mean"
        installation = create_installation(
            "virtual",
            world=SimulationWorld(SimulationWorldConfig(loading_probability=0.5)),
        )
        camera = installation.device("camera")
        sequencer = installation.device("sequencer")
        slm = installation.device("slm")
        slm.apply_phase(pattern)
        context = _Context(tmp_path)
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
            pulse_sequence=FEEDBACK_PULSE_SEQUENCE,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            feedback_mode="qcmos_bright_dark",
            exposure_seconds=0.020,
            shots_per_candidate=100,
            probe_factors=(0.5, 2.0),
            feedback_gain=0.25,
            maximum_weight_change=0.5,
            max_updates=12,
        )
        result = task.execute(context)
        saved, metadata = _load_candidate(result["artifact_path"])
        np.testing.assert_array_equal(slm.last_commanded_phase, saved)
        assert not np.array_equal(saved, pattern)
        assert result["feedback_status"] in {"completed", "stalled"}
        assert metadata["status"] == result["feedback_status"]
        history = _load_history(result["artifact_path"])
        assert 5 <= len(history) < 40
        assert sum(item["candidate_kind"] == "probe" for item in history) < 18
        ratios = np.asarray(
            [item["uniformity_ratio"] for item in history], dtype=float
        )
        finite = ratios[np.isfinite(ratios)]
        progress = np.asarray(
            [item["observable_uniformity_ratio"] for item in history],
            dtype=float,
        )
        assert max(item["observable_sites"] for item in history) == 35
        assert float(progress[-1]) < float(progress[0])
        if len(finite):
            assert metadata["measurement"]["valid"]
            assert metadata["measurement"]["uniformity_ratio"] == pytest.approx(
                float(np.min(finite))
            )
        else:
            assert result["feedback_status"] in {"completed", "stalled"}
            assert result["terminal_uniformity"] is None
        assert not camera.capture_state()
        incomplete_messages = [
            args[0]
            for args, _kwargs in context.progress
            if args and "site fits valid" in str(args[0])
        ]
        assert incomplete_messages
        root = Path(result["artifact_path"]).parent.parent
        artifacts = tuple((root / "data" / "measurements").glob("measurement-*.npz"))
        assert len(artifacts) == len(history)
        assert len(tuple((root / "figures").glob("*.npz"))) == 6
        assert len(tuple((root / "figures").glob("*.png"))) == 6
    finally:
        plane.close()
        if installation is not None:
            installation.close()
        calibration_installation.close()


def test_completed_run_selects_best_candidate_without_extra_shots(
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
    fit_results = iter(
        [
            np.linspace(0.8, 1.2, 35),
            np.linspace(0.9, 1.1, 35),
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
    requested_shots: list[int] = []
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: _fitted_result(next(fit_results)),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            requested_shots.append(self.shots),
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        )[1:],
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
        result = task.execute(_Context(tmp_path))
        saved, metadata = _load_candidate(result["artifact_path"])
        np.testing.assert_array_equal(saved, slm.last_commanded_phase)
        np.testing.assert_array_equal(saved, np.full(slm.shape_yx, 0.75, np.float32))
        assert metadata["measurement"]["uniformity_ratio"] <= 1.10
        assert requested_shots == [10, 10, 10]
        assert resolved_api_values == [{}]
        assert result["updates"] == 3
        assert result["feedback_status"] == "completed"
        assert np.array_equal(slm.commands[-1], saved)
    finally:
        plane.close()


def test_valid_site_history_changes_the_next_target_correction(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23))
    incoming = np.array(slm.last_commanded_phase, copy=True)
    plane = SignalDataPlane()
    phases = tuple(
        np.full(slm.shape_yx, value, dtype=np.float32)
        for value in (0.25, 0.50, 0.75)
    )
    phase_results = iter(phases)
    first_contrast = np.concatenate(([2.0], np.ones(34)))
    fit_results = iter(
        (
            _fitted_result(first_contrast),
            _fitted_result(np.concatenate(([1.8], np.ones(34)))),
            _fitted_result(np.concatenate(([1.5], np.ones(34)))),
            _fitted_result(np.ones(35)),
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
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: next(fit_results),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            requested_shots.append(self.shots),
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        )[1:],
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
        result = task.execute(_Context(tmp_path))
        saved, metadata = _load_candidate(result["artifact_path"])
        np.testing.assert_array_equal(saved, phases[2])
        assert requested_shots == [10, 10, 10, 10]
        history = _load_history(result["artifact_path"])
        np.testing.assert_allclose(
            [item["double_feedback_gain"] for item in history],
            (0.25, 0.25, 0.3125, 0.390625),
        )
        assert history[0]["adaptive_gain_action"] == (
            "initialize_formal_double_history"
        )
        assert history[2]["adaptive_gain_action"] == (
            "increase_after_continuous_improvement"
        )
        base = _grid_target(slm.shape_yx)
        rows, columns = np.nonzero(base)
        first_target, *_details = _updated_target(
            base,
            first_contrast,
            np.zeros(35),
            np.ones(35, dtype=bool),
            rows,
            columns,
            reference_valid=np.ones(35, dtype=bool),
            previous_weights=np.full(35, np.nan),
            previous_contrast=np.full(35, np.nan),
            feedback_gain=0.25,
            maximum_weight_change=0.5,
        )
        second_target, *_details = _updated_target(
            first_target,
            np.concatenate(([1.8], np.ones(34))),
            np.zeros(35),
            np.ones(35, dtype=bool),
            rows,
            columns,
            reference_valid=np.ones(35, dtype=bool),
            previous_weights=base[rows, columns],
            previous_contrast=first_contrast,
            feedback_gain=0.25,
            maximum_weight_change=0.5,
        )
        np.testing.assert_allclose(solved_targets[0], first_target)
        np.testing.assert_allclose(solved_targets[1], second_target)
        assert history[1]["decision"][0] == "feedback_history_slope"
        np.testing.assert_array_equal(slm.commands[0], incoming)
        np.testing.assert_array_equal(slm.commands[1], phases[0])
        np.testing.assert_array_equal(slm.commands[2], phases[1])
        np.testing.assert_array_equal(slm.commands[3], phases[2])
    finally:
        plane.close()


def test_stop_during_failed_first_checkpoint_retains_measured_candidate(
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

    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: _fitted_result(np.ones(35)),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        ),
    )

    def fail_first_checkpoint(self, context, paths, **kwargs):
        save_entered.set()
        assert release_save.wait(2.0)
        raise OSError("first candidate checkpoint failed")

    monkeypatch.setattr(
        SlmFeedbackTask, "_save_candidate_checkpoint", fail_first_checkpoint
    )
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
        host.start(run_root=tmp_path, input_summary={})
        assert save_entered.wait(10.0)
        host.cancel("while first candidate checkpoint is not durable")
        release_save.set()
        observation = _wait_host(host, wake)
        assert observation.phase == "done"
        assert host.final_result["feedback_status"] == "stopped"
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        run_root = Path(host.final_result["artifact_path"]).parent.parent
        assert not tuple((run_root / "data" / "measurements").glob("measurement-*.npz"))
        assert (run_root / "summary.json").is_file()
    finally:
        release_save.set()
        if not host.terminal:
            _wait_host(host, wake)
        host.shutdown()
        plane.close()


def test_failure_after_a_completed_candidate_saves_figures_and_context(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    plane = SignalDataPlane()
    wake = Event()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: _fitted_result(
            np.linspace(1.0, 2.0, 35)
        ),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            _mixture_samples(np.linspace(1.0, 2.0, 35), self.shots),
            (),
            (),
            np.zeros(
                self.calibration.frame_contract.image_shape,
                dtype=np.float32,
            ),
        ),
    )
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("candidate solve failed")
        ),
    )
    task = _task(
        tmp_path,
        slm=slm,
        camera=object(),
        sequencer=SimpleNamespace(describe=lambda: object()),
        plane=plane,
        target=_grid_target(slm.shape_yx),
        shots=100,
        updates=2,
    )
    host = _task_host(task, plane, wake)
    try:
        host.start(run_root=tmp_path, input_summary={})
        observation = _wait_host(host, wake)
        assert observation.phase == "failed"
        assert "candidate solve failed" in (observation.error or "")
        run_root = host.run_directory
        assert run_root is not None
        assert len(tuple((run_root / "figures").glob("*.npz"))) == 6
        assert len(tuple((run_root / "figures").glob("*.png"))) == 6
        candidate_figures = run_root / "figures" / "candidate_site_fits"
        assert [path.name for path in candidate_figures.glob("*.npz")] == [
            "candidate-0001.npz"
        ]
        assert [path.name for path in candidate_figures.glob("*.png")] == [
            "candidate-0001.png"
        ]
        info, arrays = read_archive(candidate_figures / "candidate-0001.npz")
        _plot_input, recipe = read_figure_plot(info, arrays, "data")
        assert isinstance(recipe["spec"], FacetGridPlot)
        assert isinstance(recipe["spec"].cell, HistogramPlot)
        assert recipe["fit"] == {
            "model": "bimodal_gaussian",
            "fit_all_facets": True,
        }
        run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
        artifact_roles = {
            item["name"]: item["role"] for item in run["artifacts"]
        }
        assert artifact_roles["candidate_0001_site_fits_figure"] == "figure"
        assert artifact_roles["candidate_0001_site_fits_image"] == "figure"
        candidate = run_root / "candidates" / "candidate-0001.npz"
        loaded = load_science_context(candidate)
        assert loaded["pattern_metadata"]["status"] == "checkpoint"
        assert loaded["pattern_metadata"]["candidate"] == 1
        summary = json.loads(
            (run_root / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "failed"
        assert summary["selected_candidate"] == 1
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
    finally:
        host.shutdown()
        plane.close()


def test_stop_after_terminal_commit_keeps_host_success_and_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    first_phase = np.full(slm.shape_yx, 0.5, dtype=np.float32)
    best = np.full(slm.shape_yx, 0.75, dtype=np.float32)
    plane = SignalDataPlane()
    wake = Event()
    save_entered = Event()
    release_save = Event()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    phases = iter((first_phase, best))
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (next(phases), {"method": "test"}),
    )
    fit_results = iter(
        (
            _fitted_result(np.concatenate(([2.0], np.ones(34)))),
            _fitted_result(np.concatenate(([1.2], np.ones(34)))),
            _fitted_result(np.ones(35)),
        )
    )
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: next(fit_results),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        ),
    )
    original_save = feedback_module.save_science_context

    def blocking_save(path, phase, **kwargs):
        if kwargs["pattern_metadata"]["status"] == "completed":
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
        updates=2,
    )
    host = _task_host(task, plane, wake)
    try:
        host.start(run_root=tmp_path, input_summary={})
        assert save_entered.wait(10.0)
        host.cancel("after terminal commit")
        release_save.set()
        observation = _wait_host(host, wake)
        assert observation.phase == "done"
        result = host.final_result
        assert isinstance(result, dict)
        saved, _metadata = _load_candidate(result["artifact_path"])
        np.testing.assert_array_equal(saved, best)
        np.testing.assert_array_equal(slm.last_commanded_phase, best)
        run_root = host.run_directory
        assert run_root is not None
        run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
        assert run["status"]["state"] == "completed"
        assert {item["name"] for item in run["artifacts"]} >= {
            "artifact_path",
            "summary_json",
            "summary_text",
            "uniformity_history_figure",
            "uniformity_history_preview",
        }
        assert len(tuple((run_root / "figures").glob("*.npz"))) == 6
        assert len(tuple((run_root / "figures").glob("*.png"))) == 6
        candidate_figures = run_root / "figures" / "candidate_site_fits"
        assert [path.name for path in candidate_figures.glob("*.npz")] == [
            "candidate-0001.npz",
            "candidate-0002.npz",
            "candidate-0003.npz",
        ]
        summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
        assert summary["initial_observable_uniformity_ratio"] is not None
        assert summary["selected_observable_uniformity_ratio"] is not None
        assert summary["common_observable_sites"] == 35
        assert summary["rollback"] is None
    finally:
        release_save.set()
        if not host.terminal:
            _wait_host(host, wake)
        host.shutdown()
        plane.close()


def test_terminal_save_failure_restores_incoming_and_fails_host(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    first_phase = np.full(slm.shape_yx, 0.5, dtype=np.float32)
    best = np.full(slm.shape_yx, 0.75, dtype=np.float32)
    plane = SignalDataPlane()
    wake = Event()
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    phases = iter((first_phase, best))
    monkeypatch.setattr(
        feedback_module,
        "solve_phase",
        lambda *args, **kwargs: (next(phases), {"method": "test"}),
    )
    fit_results = iter(
        (
            _fitted_result(np.concatenate(([2.0], np.ones(34)))),
            _fitted_result(np.concatenate(([1.2], np.ones(34)))),
            _fitted_result(np.ones(35)),
        )
    )
    monkeypatch.setattr(
        feedback_module,
        "_fit_contrasts",
        lambda samples, **_kwargs: next(fit_results),
    )
    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        lambda self, pulse, context, iteration: (
            np.zeros((self.shots, 35)),
            (),
            (),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
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
    original_save = feedback_module.save_science_context

    def fail_terminal_save(path, phase, **kwargs):
        if kwargs["pattern_metadata"]["status"] == "completed":
            raise OSError("terminal save failed")
        return original_save(path, phase, **kwargs)

    monkeypatch.setattr(
        feedback_module, "save_science_context", fail_terminal_save
    )
    host = _task_host(task, plane, wake)
    try:
        host.start(run_root=tmp_path, input_summary={})
        observation = _wait_host(host, wake)
        assert observation.phase == "failed"
        assert observation.error == "OSError: terminal save failed"
        assert not host.final_result_resolved
        np.testing.assert_array_equal(slm.last_commanded_phase, incoming)
        run_root = host.run_directory
        assert run_root is not None
        artifacts = tuple((run_root / "data" / "measurements").glob("measurement-*.npz"))
        assert len(artifacts) == 3
        assert artifacts[-1].name == "measurement-0003.npz"
        assert not (run_root / "final" / "science-context.npz").exists()
        summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "failed"
        assert summary["candidate_count"] == 3
        assert summary["rollback"]["status"] == "restored"
    finally:
        if not host.terminal:
            _wait_host(host, wake)
        host.shutdown()
        plane.close()


def test_invalid_site_holds_weight_and_never_retries_the_same_phase(
    tmp_path: Path, monkeypatch
) -> None:
    slm = _Slm((17, 23), incoming=0.125)
    incoming = np.array(slm.last_commanded_phase, copy=True)
    plane = SignalDataPlane()
    first = _mixture_samples(np.ones(35), 10)
    first[:, 4] = np.nan
    monkeypatch.setattr(
        feedback_module,
        "resolve_pulse",
        lambda *args, **kwargs: SimpleNamespace(program=object()),
    )
    measured_phases: list[np.ndarray] = []

    def measure(self, pulse, context, iteration):
        measured_phases.append(np.array(self.slm.last_commanded_phase, copy=True))
        return (
            first,
            (),
            (4,),
            np.zeros(self.calibration.frame_contract.image_shape, dtype=np.float32),
        )

    monkeypatch.setattr(
        SlmFeedbackTask,
        "_measure",
        measure,
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
        result = task.execute(_Context(tmp_path))
        assert result["feedback_status"] in {"completed", "stalled"}
        assert measured_phases
        assert all(
            not np.array_equal(left, right)
            for left, right in zip(measured_phases, measured_phases[1:])
        )
        artifacts = tuple(
            (Path(result["artifact_path"]).parent.parent / "data" / "measurements").glob(
                "measurement-*.npz"
            )
        )
        assert len(artifacts) == len(measured_phases)
        history = _load_history(result["artifact_path"])
        assert all(item["invalid_sites"] == [4] for item in history)
        assert all(
            item["decision"][4] == "hold_invalid"
            for item in history
            if item["candidate_kind"] != "probe"
        )
        assert all(
            item["requested_log_correction"][4] == 0.0
            for item in history
        )
        assert all(
            item["target_weight"][4] == 1.0
            for item in history
            if item["candidate_kind"] != "probe"
        )
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

    def cancelled(self, pulse, context, iteration):
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
        context = _Context(tmp_path, cancelled=True)
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
        assert not slm.commands
        root = Path(result["artifact_path"]).parent.parent
        assert not tuple((root / "data" / "measurements").glob("measurement-*.npz"))
        assert Path(result["artifact_path"]) == root / "final" / "science-context.npz"
    finally:
        plane.close()
