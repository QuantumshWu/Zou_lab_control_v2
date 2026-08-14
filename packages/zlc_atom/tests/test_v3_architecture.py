from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import time

import pytest
from zlc_durable import readable_json_bytes
from zlc_pulse import (
    PULSE_TREE_FORMAT,
    PulseTarget,
    resolve_api_parameters,
    sequence_from_tree,
    sequence_to_tree,
)
from zlc_pulse.device import BoardDescription
from zlc_runtime import NodeHost, SignalDataPlane

import zlc_atom.nodes.calibration.pulse as calibration_pulse_module
from zlc_atom.devices.simulation import VirtualPulseStreamer
from zlc_atom.install import create_installation
from zlc_atom.nodes import (
    ArtifactInputSpec,
    DatasetInputSpec,
    NodeKind,
    calibration_pulse_template_bytes,
    discover_logic_nodes,
)
from zlc_atom.nodes.calibration import (
    CalibrationRequest,
    CalibrationTask,
    LOGIC_NODE as CALIBRATION_LOGIC_NODE,
    ReadoutModelKind,
)
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from tests.fakes import FakePlane, camera_cycle_snapshot
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE


ROOT = Path(__file__).parents[1]
PULSE_ROOT = ROOT / "src" / "zlc_atom" / "nodes" / "calibration"


class _CalibrationCoveragePlane(FakePlane):
    def __init__(self) -> None:
        super().__init__()
        self.calibration_coverages: list[tuple[int, int]] = []
        self.calibration_schema_fingerprints: list[str] = []
        self.calibration_preview_shapes: list[tuple[int, ...]] = []

    def mark_changed(self, producer: object, live_slot: object) -> None:
        output = live_slot.freeze_live_outputs()["capture_preview"]
        self.calibration_coverages.append(
            (output.coverage.written_cells, output.coverage.total_cells)
        )
        self.calibration_schema_fingerprints.append(
            output.snapshot.block.schema.fingerprint
        )
        self.calibration_preview_shapes.append(output.snapshot.block.values.shape)
        super().mark_changed(producer, live_slot)


class _RecordingCamera:
    def __init__(self, camera: object) -> None:
        self.camera = camera
        self.events: list[tuple[str, object]] = []

    @property
    def timeout(self) -> float:
        self.events.append(("timeout", None))
        return float(self.camera.timeout)  # type: ignore[attr-defined]

    @property
    def photoelectron_conversion(self):
        return self.camera.photoelectron_conversion  # type: ignore[attr-defined]

    def working_point(self):
        self.events.append(("working_point", None))
        return self.camera.working_point()  # type: ignore[attr-defined]

    def set_exposure_seconds(self, seconds):
        self.events.append(("set_exposure_seconds", seconds))
        return self.camera.set_exposure_seconds(seconds)  # type: ignore[attr-defined]

    def set_roi(self, roi_xywh):
        self.events.append(("set_roi", roi_xywh))
        return self.camera.set_roi(roi_xywh)  # type: ignore[attr-defined]

    def arm(self, frames, *, source_group_sizes, buffer_frame_count, timeout) -> None:
        self.events.append(("arm", (frames, source_group_sizes, buffer_frame_count)))
        self.camera.arm(  # type: ignore[attr-defined]
            frames,
            source_group_sizes=source_group_sizes,
            buffer_frame_count=buffer_frame_count,
            timeout=timeout,
        )

    def read_frame_records(self, n, *, timeout, exact):
        self.events.append(("read", (n, exact)))
        return self.camera.read_frame_records(n, timeout=timeout, exact=exact)  # type: ignore[attr-defined]

    def finish_record_capture(self):
        self.events.append(("finish", None))
        return self.camera.finish_record_capture()  # type: ignore[attr-defined]

    def capture_state(self):
        self.events.append(("state", None))
        return self.camera.capture_state()  # type: ignore[attr-defined]


class _RecordingSequencer:
    def __init__(self, sequencer: object, *, fail_on_fire: bool = False) -> None:
        self.sequencer = sequencer
        self.events: list[tuple[str, object]] = []
        self.fail_on_fire = fail_on_fire

    def load(self, program: object, *, source: object | None = None) -> None:
        self.events.append(("load", program))
        self.sequencer.load(program, source=source)  # type: ignore[attr-defined]

    def describe(self):
        self.events.append(("describe", None))
        return self.sequencer.describe()  # type: ignore[attr-defined]

    def fire(self, *, forever: bool = False) -> None:
        """Fire, and nothing else.

        This used to wait for the shot as well, which is what the code under
        test is supposed to do -- so the report was consumed here and the
        production wait got None.  A double that performs the behaviour under
        test hides whether the real code performs it.
        """

        self.events.append(("fire", forever))
        if self.fail_on_fire:
            raise RuntimeError("recording sequencer fire failure")
        self.sequencer.fire(forever=forever)  # type: ignore[attr-defined]

    def wait_done(self, timeout: float | None = None) -> object:
        self.events.append(("wait_done", timeout))
        return self.sequencer.wait_done(timeout)  # type: ignore[attr-defined]

    def safe(self) -> None:
        self.events.append(("safe", None))
        self.sequencer.safe()  # type: ignore[attr-defined]

    def snapshot(self):
        self.events.append(("snapshot", None))
        return self.sequencer.snapshot()  # type: ignore[attr-defined]

    @property
    def camera_trigger_channel(self) -> str:
        return self.sequencer.camera_trigger_channel  # type: ignore[attr-defined]

    @camera_trigger_channel.setter
    def camera_trigger_channel(self, value: str) -> None:
        self.sequencer.camera_trigger_channel = value  # type: ignore[attr-defined]


def _calibration_request(*, repeats: int = 30) -> CalibrationRequest:
    return CalibrationRequest(
        camera_key="camera",
        sequencer_key="sequencer",
        pulse_template="imaging_template.json",
        repeats=repeats,
        reference_exposure_seconds=0.02,
        readout_exposure_seconds=0.005,
        reference_before_slot=1,
        readout_slot=2,
        reference_after_slot=3,
        default_model_kind=ReadoutModelKind.BOX,
        threshold_method="empirical",
        box_half_width=1,
        box_reducer="mean",
        psf_half_width=3,
        psf_padding=3,
        detection_spot_sigma=1.0,
        detection_sigma=6.0,
    )


def test_measurement_leaf_has_no_sequencer_dependency_or_operation() -> None:
    files = tuple((ROOT / "src" / "zlc_atom" / "nodes" / "camera_measurement").rglob("*.py"))
    assert files
    assert all("sequencer" not in path.read_text(encoding="utf-8") for path in files)


def _descriptor_kind(path: Path) -> object:
    """The NodeKind a package declares, read from its own descriptor."""

    module = importlib.import_module(
        f"zlc_atom.nodes.{path.parent.name}.logic_node"
    )
    return module.LOGIC_NODE.kind


def test_node_cross_imports_have_only_owner_edges() -> None:
    """Only a TASK may reach into another node; a measurement may not.

    ``nodes/scan`` carries no ``logic_node.py``: it is not a node, it is what
    the scan nodes stand on -- the plan, the ports, the dataset, the editor,
    and the board-advanced loop, which moved there the day a second consumer
    appeared.  An edge into it is reuse.

    An edge into another NODE is a node reaching into another's science, and
    what it means depends on which node reaches.  A MEASUREMENT reaching
    sideways is duplication waiting to happen, so it may only consume what
    another node SAVED -- occupancy reads a calibration artifact through the
    codec calibration published it under, and imports none of its analysis.

    A TASK is the one thing on this bench that owns a whole experiment: the
    temperature Task holds the camera and the sequencer, drives the camera's
    own acquisition, judges every cycle with the occupancy processor, and adds
    only the pairing that is its own.  Those edges are the alternative to
    three copies of somebody else's science, so they are allowed BY KIND and
    still listed one by one -- a new one has to be argued for.
    """

    nodes_root = ROOT / "src" / "zlc_atom" / "nodes"
    files = tuple(nodes_root.rglob("*.py"))
    assert files
    node_owners = {path.parent.name for path in nodes_root.rglob("logic_node.py")}
    assert "scan" not in node_owners, "the scan package must not be a node"
    edges: set[tuple[str, str]] = set()
    for path in files:
        relative = path.relative_to(nodes_root)
        source_owner = relative.parts[0] if len(relative.parts) > 1 else None
        if source_owner in {None, "_framework"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
                module = node.module
            elif isinstance(node, ast.Import):
                module = next((alias.name for alias in node.names if alias.name.startswith("zlc_atom.nodes.")), None)
            if not module or not module.startswith("zlc_atom.nodes."):
                continue
            target_owner = module.split(".")[2]
            if target_owner != source_owner and target_owner != "_framework":
                edges.add((source_owner, target_owner))
    node_edges = {edge for edge in edges if edge[1] in node_owners}
    assert node_edges == {
        ("calibration", "camera_measurement"),
        ("occupancy", "calibration"),
        ("slm_feedback", "calibration"),
        ("slm_feedback", "camera_measurement"),
        ("slm_feedback", "occupancy"),
        ("temperature", "calibration"),
        ("temperature", "camera_measurement"),
        ("temperature", "occupancy"),
    }
    kinds = {
        path.parent.name: _descriptor_kind(path)
        for path in nodes_root.rglob("logic_node.py")
    }
    sideways = {
        source
        for source, target in node_edges
        if target != "calibration"
    }
    assert all(kinds[source] is NodeKind.TASK for source in sideways), (
        "only a Task may drive another node's engine; a measurement that "
        f"needs one is asking for a library: {sorted(sideways)}"
    )
    assert {edge for edge in edges if edge[1] not in node_owners} == {
        ("seamless_scan", "scan"),
        ("slm_feedback", "scan"),
        ("stepped_scan", "scan"),
        ("temperature", "scan"),
    }


def test_pulse_resolver_uses_the_project_json_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = PULSE_ROOT / "imaging_template.json"
    packaged = calibration_pulse_template_bytes()
    assert packaged == asset.read_bytes()
    tree = json.loads(packaged)
    assert packaged == readable_json_bytes(tree)
    assert tuple(tree) == (
        "format",
        "name",
        "time_step_ns",
        "target",
        "periods",
        "slots",
        "api_parameters",
        "delays",
        "repeat",
    )
    assert tree["format"] == PULSE_TREE_FORMAT == "zlc.pulse.v1"
    assert not {
        "schema",
        "PulseDocument",
        "fingerprint",
        "hash",
        "editor",
    }.intersection(tree)
    sequence = sequence_from_tree(tree)
    assert sequence_to_tree(sequence) == tree
    assert sequence.slots == ()
    assert tuple(parameter.parameter_id for parameter in sequence.api_parameters) == (
        "reference_probe_duration_before",
        "readout_probe_duration",
        "reference_probe_duration_after",
    )
    api_values = {
        "reference_probe_duration_before": 0.031,
        "readout_probe_duration": 0.006,
        "reference_probe_duration_after": 0.031,
    }
    explicit = resolve_api_parameters(sequence, api_values)
    assert explicit.api_parameters == ()
    assert explicit.slots == ()
    resource_spec = CALIBRATION_LOGIC_NODE.workspace_resources[0]
    resource = resource_spec.resolve(asset)
    assert sequence_to_tree(resource.value) == tree

    sequencer = VirtualPulseStreamer()
    sequencer.open()
    try:
        board = sequencer.describe()
        monkeypatch.setattr(
            calibration_pulse_module,
            "sequence_from_tree",
            lambda _tree: (_ for _ in ()).throw(
                AssertionError("resolve_pulse decoded its resource a second time")
            ),
        )
        resolved = resolve_pulse(
            resource.value,
            path=resource.path,
            board=board,
            api_values=api_values,
        )
        assert resolved.path == asset.resolve()
        assert set(resolved.metadata) == {"repeat_forever"}
        assert resolved.program.camera_window_exposures("emCCD") == pytest.approx(
            (0.031, 0.006, 0.031)
        )
        assert resolved.program.slot_count == 0
        assert resolved.program.clock_hz == board.clock_hz
        assert resolved.program.channels == board.target.raw_lanes

        incompatible_board = BoardDescription(
            target=PulseTarget(
                raw_lanes=tuple(reversed(board.target.raw_lanes)),
                ports=board.target.ports,
            ),
            geometry=board.geometry,
            clock_hz=board.clock_hz,
        )
        with pytest.raises(ValueError, match="target.*connected board"):
            resolve_pulse(
                resource.value,
                path=resource.path,
                board=incompatible_board,
                api_values=api_values,
            )

        with pytest.raises(ValueError, match="existing"):
            resource_spec.resolve(
                PULSE_ROOT / "missing.json"
            )
    finally:
        sequencer.close()


def test_discovered_descriptors_build_and_exercise_declared_devices(tmp_path: Path) -> None:
    installation = create_installation("virtual")
    plane = _CalibrationCoveragePlane()
    calibration_host = None
    try:
        descriptors = {descriptor.api_name: descriptor for descriptor in discover_logic_nodes()}
        camera = _RecordingCamera(installation.device("camera"))
        sequencer = _RecordingSequencer(installation.device("sequencer"))
        camera_node = descriptors["camera_measurement"].instantiate(
            camera=camera,
            camera_key="camera",
            signal_plane=plane,
            # The authored default repeat=0 is the infinite monitor mode;
            # a finite prepare() must author its acquisition explicitly.
            repeat=1,
            frames_per_cycle=3,
        )
        assert camera_node.camera is camera
        pulse = resolve_pulse(
            IMAGING_PULSE_RESOURCE.value,
            path=IMAGING_PULSE_RESOURCE.path,
            board=sequencer.describe(),
            api_values={
                "reference_probe_duration_before": 0.02,
                "readout_probe_duration": 0.005,
                "reference_probe_duration_after": 0.02,
            },
        )
        capture = camera_node.prepare()
        arm_sequencer(sequencer, pulse)
        assert sequencer.sequencer.applied().source == pulse.sequence
        sequencer.fire()
        sequencer.wait_done(1.0)
        manual_result = capture.collect()
        assert len(manual_result.frames) == 3
        assert [name for name, _ in camera.events].count("timeout") == 1
        assert any(name == "arm" for name, _ in camera.events)
        assert next(
            name
            for name, _ in camera.events
            if name in {"set_exposure_seconds", "arm"}
        ) == "set_exposure_seconds"

        calibration_node = descriptors["calibration"].instantiate(
            camera=camera,
            camera_key="camera",
            sequencer=sequencer,
            sequencer_key="sequencer",
            pulse_resource=IMAGING_PULSE_RESOURCE,
            signal_plane=plane,
            artifact_directory=tmp_path,
            repeats=30,
        )
        assert calibration_node.camera is camera
        loads_before_task = len([event for event, _ in sequencer.events if event == "load"])
        fires_before_task = len([event for event, _ in sequencer.events if event == "fire"])
        waits_before_task = len(
            [event for event, _ in sequencer.events if event == "wait_done"]
        )
        camera_events_before_task = len(camera.events)
        calibration_host = NodeHost(calibration_node, plane)
        calibration_host.start()
        deadline = time.monotonic() + 10.0
        while not calibration_host.observation.terminal and time.monotonic() < deadline:
            calibration_host.poll()
            time.sleep(0.005)
        observation = calibration_host.poll()
        assert observation.terminal and observation.phase == "done", observation
        assert observation.progress is not None
        assert observation.progress.text == "Calibration complete 30/30"
        task_result = calibration_node.result
        assert task_result is not None
        task_camera_events = camera.events[camera_events_before_task:]
        # One publication is one whole three-window cycle: the long, readout
        # and long frames are three POINTS of the acquisition, and the preview
        # used to publish only the last of them.
        assert plane.calibration_coverages == [(3, 3)] * 30
        assert plane.calibration_preview_shapes == [(1, 3, 96, 128)] * 30
        assert len(set(plane.calibration_schema_fingerprints)) == 1
        report_images = tuple(
            sorted(
                path.name
                for path in (
                    task_result.artifact_path.with_suffix("") / "report"
                ).glob("*.png")
            )
        )
        assert report_images == (
            "box.png",
            "fidelity.png",
            "psf.png",
            "psf_kernels.png",
            "site_map.png",
            "uniform_psf.png",
        )
        assert len(task_result.capture.frames) == 90
        assert sum(len(group) for group in task_result.reference) == 60
        assert len(task_result.short) == 30
        # Through the camera measurement, which points the camera at the
        # geometry this run asks for -- and it asks for the geometry that is
        # already there, so a tuned ROI survives a calibration.
        assert [event for event, _ in task_camera_events].count("set_roi") == 1
        assert [event for event, _ in task_camera_events].count(
            "set_exposure_seconds"
        ) == 1
        assert next(
            name
            for name, _ in task_camera_events
            if name in {"set_exposure_seconds", "arm"}
        ) == "set_exposure_seconds"
        assert ("arm", (90, (3,) * 30, 90)) in task_camera_events
        assert [value for name, value in task_camera_events if name == "read"] == [
            (3, True)
        ] * 30
        assert [name for name, _ in task_camera_events].count("finish") == 1
        assert len([event for event, _ in sequencer.events if event == "load"]) - loads_before_task == 1
        # One pulse for the whole run: the board repeats the cycle itself and
        # the camera reads all thirty from it.  Thirty fires with a DONE
        # handshake each was a round trip per repeat, serialised against the
        # camera's own transfer.
        assert [
            value for event, value in sequencer.events if event == "fire"
        ][fires_before_task:] == [True]
        assert len(
            [event for event, _ in sequencer.events if event == "wait_done"]
        ) == waits_before_task

        calibration_path = task_result.artifact_path
        calibration_output = descriptors["calibration"].artifact_outputs[0]
        assert getattr(task_result, calibration_output.name) == calibration_path
        calibration_input = descriptors["occupancy"].input_specs[1]
        assert isinstance(calibration_input, ArtifactInputSpec)
        occupancy_node = descriptors["occupancy"].instantiate(
            calibration=calibration_input.codec.resolve(calibration_path),
            source_signal=camera_node.signal_key("frames"),
            model_kind="uniform_psf",
        )
        assert not hasattr(occupancy_node, "signal_plane")
        assert occupancy_node.calibration_path == calibration_path.resolve()
        assert occupancy_node.model.kind is ReadoutModelKind.UNIFORM_PSF
        occupancy_result = occupancy_node.process(
            camera_cycle_snapshot([(record,) for record in task_result.short]),
            generation="calibration-task",
            revision=1,
        )
        assert occupancy_result.counts.shape == (30, 1, 35)
        assert occupancy_result.valid.shape == occupancy_result.counts.shape
        assert occupancy_result.frame_judged.shape == (30, 1, 96, 128)

        assert tuple(value.argument_name for value in descriptors["camera_measurement"].device_requirements) == ("camera",)
        assert tuple(value.argument_name for value in descriptors["calibration"].device_requirements) == ("camera", "sequencer")
        assert tuple(
            (value.name, value.contract_id)
            for value in descriptors["calibration"].outputs
        ) == (("capture_preview", "calibration.capture-preview.v1"),)
        assert tuple(
            (preview.output_name, preview.plot_kind)
            for preview in descriptors["calibration"].node_previews
        ) == (("capture_preview", "facet_grid"),)
        # A preview is not a Task privilege: an ordinary measurement names the
        # output an operator came to watch in exactly the same words -- and
        # names how to draw it, which for a camera is what it was asked to
        # take.  One frame per cycle IS a picture; three are three of them.
        camera_descriptor = descriptors["camera_measurement"]
        assert tuple(
            (preview.output_name, preview.plot_kind)
            for preview in camera_descriptor.previews_for({"frames_per_cycle": 1})
        ) == (("frames", "facet_grid"),)
        assert tuple(
            (preview.output_name, preview.plot_kind)
            for preview in camera_descriptor.previews_for({"frames_per_cycle": 3})
        ) == (("frames", "facet_grid"),)
        assert not hasattr(descriptors["calibration"], "task_reports")
        assert [
            (value.name, value.contract_id)
            for value in descriptors["calibration"].artifact_outputs
        ] == [("artifact_path", "calibration.readout.v1")]
        assert descriptors["calibration"].authoring_schema.field_names == (
            "pulse_template",
            "repeats",
            "reference_exposure_seconds",
            "readout_exposure_seconds",
            "reference_before_slot",
            "readout_slot",
            "reference_after_slot",
            "default_model_kind",
            "threshold_method",
            "box_half_width",
            "box_reducer",
            "psf_half_width",
            "psf_padding",
            "detection_spot_sigma",
            "detection_sigma",
            # Where the frames come from, and whether they are kept: a
            # calibration can be re-run on samples it saved earlier instead of
            # holding the bench for another few hundred.
            "frame_source",
            "saved_frames_path",
            "save_frames",
            # And whether the camera is read in photoelectrons: the
            # conversion is configured on the camera, so the choice is a
            # run's, not the analysis's.
            "photoelectrons",
        )
        assert descriptors["calibration"].authoring_schema.project_values({})[
            "repeats"
        ] == 300
        with pytest.raises(ValueError, match="cannot exceed"):
            descriptors["calibration"].authoring_schema.project_values(
                {
                    "reference_exposure_seconds": 0.005,
                    "readout_exposure_seconds": 0.02,
                }
            )
        assert descriptors["occupancy"].device_requirements == ()
        assert descriptors["occupancy"].authoring_schema.field_names == (
            "model_kind",
        )
        assert len(descriptors["occupancy"].input_specs) == 2
        assert isinstance(descriptors["occupancy"].input_specs[0], DatasetInputSpec)
        assert descriptors["occupancy"].input_specs[0].name == "frames"
        assert descriptors["occupancy"].input_specs[0].contract_id is None
        assert isinstance(descriptors["occupancy"].input_specs[1], ArtifactInputSpec)
        assert descriptors["occupancy"].input_specs[1].name == "calibration_path"
        assert descriptors["occupancy"].input_specs[1].contract_id == (
            "calibration.readout.v1"
        )
        assert tuple(output.name for output in descriptors["occupancy"].outputs) == (
            "counts",
            "occupied",
            "valid",
            "rate",
            "frame_judged",
        )
        with pytest.raises((TypeError, ValueError)):
            descriptors["camera_measurement"].instantiate(
                camera=camera,
                camera_key="camera",
                signal_plane=plane,
                sequencer=sequencer,
            )
        with pytest.raises(TypeError):
            descriptors["occupancy"].instantiate(
                calibration=task_result.calibration,
                source_signal=camera_node.signal_key("frames"),
            )
    finally:
        if calibration_host is not None:
            calibration_host.shutdown()
        plane.close()
        installation.close()


def test_calibration_task_safes_sequencer_when_capture_fails(tmp_path: Path) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        camera = _RecordingCamera(installation.device("camera"))
        sequencer = _RecordingSequencer(installation.device("sequencer"), fail_on_fire=True)
        task = CalibrationTask(
            camera=camera,
            sequencer=sequencer,
            request=_calibration_request(repeats=1),
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            signal_plane=plane,
            artifact_directory=tmp_path,
        )
        with pytest.raises(RuntimeError, match="fire failure"):
            task.run()
        assert [event for event, _ in sequencer.events].count("safe") == 1
        assert [event for event, _ in camera.events].count("finish") == 1
        assert installation.device("camera").capture_state() is False
    finally:
        plane.close()
        installation.close()
