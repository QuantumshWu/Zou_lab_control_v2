from __future__ import annotations

import ast
from pathlib import Path

import pytest
from zlc_runtime import SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes import discover_logic_nodes
from zlc_atom.nodes._framework.pulse_source import resolve_pulse
from zlc_atom.nodes.calibration import CalibrationTask


ROOT = Path(__file__).parents[1]


class _RecordingCamera:
    def __init__(self, camera: object) -> None:
        self.camera = camera
        self.events: list[tuple[str, object]] = []

    @property
    def timeout(self) -> float:
        self.events.append(("timeout", None))
        return float(self.camera.timeout)  # type: ignore[attr-defined]

    def capture_working_point(self):
        self.events.append(("working_point", None))
        return self.camera.capture_working_point()  # type: ignore[attr-defined]

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

    def load(self, program: object) -> None:
        self.events.append(("load", program))
        self.sequencer.load(program)  # type: ignore[attr-defined]

    def fire(self) -> None:
        """Fire, and nothing else.

        This used to wait for the shot as well, which is what the code under
        test is supposed to do -- so the report was consumed here and the
        production wait got None.  A double that performs the behaviour under
        test hides whether the real code performs it.
        """

        self.events.append(("fire", None))
        if self.fail_on_fire:
            raise RuntimeError("recording sequencer fire failure")
        self.sequencer.fire()  # type: ignore[attr-defined]

    def wait_done(self, timeout: float | None = None) -> object:
        self.events.append(("wait_done", timeout))
        return self.sequencer.wait_done(timeout)  # type: ignore[attr-defined]

    def safe(self) -> None:
        self.events.append(("safe", None))
        self.sequencer.safe()  # type: ignore[attr-defined]

    @property
    def camera_trigger_channel(self) -> str:
        return self.sequencer.camera_trigger_channel  # type: ignore[attr-defined]

    @camera_trigger_channel.setter
    def camera_trigger_channel(self, value: str) -> None:
        self.sequencer.camera_trigger_channel = value  # type: ignore[attr-defined]


def _short_stamps(result: object) -> dict[str, object]:
    """The run the short frames came from, which a derived signal inherits."""

    from zlc_atom.nodes.occupancy.processor import inherited_stamps

    publication = result.short.publication
    name = next(iter(publication.signals))
    return inherited_stamps(publication.signals[name].snapshot)


def test_measurement_leaf_has_no_sequencer_dependency_or_operation() -> None:
    files = tuple((ROOT / "src" / "zlc_atom" / "nodes" / "camera_measurement").rglob("*.py"))
    assert files
    assert all("sequencer" not in path.read_text(encoding="utf-8") for path in files)


def test_node_cross_imports_have_only_owner_edges() -> None:
    nodes_root = ROOT / "src" / "zlc_atom" / "nodes"
    files = tuple(nodes_root.rglob("*.py"))
    assert files
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
    assert edges == {("occupancy", "calibration"), ("calibration", "camera_measurement")}


def test_pulse_resolver_has_one_named_source_and_clear_missing_paths() -> None:
    resolved = resolve_pulse("calibration", search_paths=(ROOT / "pulses",))
    assert resolved.path == (ROOT / "pulses" / "calibration.py").resolve()
    assert resolved.metadata["camera_windows"] == 3
    assert resolved.metadata["frame_exposures"] == (0.02, 0.005, 0.02)

    override_program = object()
    overridden = resolve_pulse(
        "does_not_need_a_file",
        search_paths=(),
        override=(override_program, {"camera_windows": 2}),
    )
    assert overridden.program is override_program
    assert overridden.metadata["camera_windows"] == 2

    try:
        resolve_pulse("missing", search_paths=(ROOT / "pulses",))
    except FileNotFoundError as error:
        assert str(ROOT / "pulses" / "missing.py") in str(error)
    else:
        raise AssertionError("missing pulse unexpectedly resolved")


def test_discovered_descriptors_build_and_exercise_declared_devices() -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        descriptors = {descriptor.api_name: descriptor for descriptor in discover_logic_nodes()}
        camera = _RecordingCamera(installation.device("camera"))
        sequencer = _RecordingSequencer(installation.device("sequencer"))
        camera_node = descriptors["camera_measurement"].instantiate(
            camera=camera,
            signal_plane=plane,
        )
        assert camera_node.camera is camera
        pulse = resolve_pulse("calibration", search_paths=(ROOT / "pulses",))
        capture = camera_node.prepare(repeat=1, frames_per_cycle=3)
        sequencer.load(pulse.program)
        sequencer.fire()
        sequencer.wait_done(1.0)
        manual_result = capture.collect()
        assert len(manual_result.frames) == 3
        assert any(name == "arm" for name, _ in camera.events)

        calibration_node = descriptors["calibration"].instantiate(
            camera=camera,
            sequencer=sequencer,
            signal_plane=plane,
            pulse_search_paths=(ROOT / "pulses",),
            expected_centers_xy=installation.world.geometry.site_centers_xy,
            repeats=30,
        )
        assert calibration_node.camera is camera
        loads_before_task = len([event for event, _ in sequencer.events if event == "load"])
        fires_before_task = len([event for event, _ in sequencer.events if event == "fire"])
        task_result = calibration_node.run()
        assert len(task_result.capture.frames) == 90
        assert len(task_result.reference.frames) == 60
        assert len(task_result.short.frames) == 30
        assert len([event for event, _ in sequencer.events if event == "load"]) - loads_before_task == 1
        assert len([event for event, _ in sequencer.events if event == "fire"]) - fires_before_task == 30

        occupancy_node = descriptors["occupancy"].instantiate(
            calibration=task_result.calibration,
            signal_plane=plane,
        )
        assert occupancy_node.signal_plane is plane
        occupancy_result = occupancy_node.process(
            task_result.short.frames, **_short_stamps(task_result)
        )
        assert occupancy_result.counts.shape == (30, 6)

        assert tuple(value.device_key for value in descriptors["camera_measurement"].device_requirements) == ("camera",)
        assert tuple(value.device_key for value in descriptors["calibration"].device_requirements) == ("camera", "sequencer")
        assert descriptors["occupancy"].device_requirements == ()
        with pytest.raises((TypeError, ValueError)):
            descriptors["camera_measurement"].instantiate(camera=camera, signal_plane=plane, sequencer=sequencer)
        with pytest.raises(TypeError):
            descriptors["occupancy"].instantiate(calibration=task_result.calibration, camera=camera)
    finally:
        plane.close()
        installation.close()


def test_calibration_task_safes_sequencer_when_capture_fails() -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        camera = _RecordingCamera(installation.device("camera"))
        sequencer = _RecordingSequencer(installation.device("sequencer"), fail_on_fire=True)
        task = CalibrationTask(
            camera=camera,
            sequencer=sequencer,
            signal_plane=plane,
            repeats=1,
            pulse_search_paths=(ROOT / "pulses",),
        )
        with pytest.raises(RuntimeError, match="fire failure"):
            task.run()
        assert any(event == "safe" for event, _ in sequencer.events)
        assert installation.device("camera").capture_state() == (False, 0)
    finally:
        plane.close()
        installation.close()
