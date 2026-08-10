import zou_lab_control_v2

"""Guard A: the formal headless virtual Calibration -> frames -> Occupancy chain."""

import json
from pathlib import Path
import time

import numpy as np

from zlc_atom.install import create_installation
from zlc_atom.nodes import (
    ResolvedArtifact,
    ResolvedWorkspaceResource,
    calibration_pulse_template_bytes,
)
from zlc_atom.nodes.calibration import logic_node as calibration_logic_node
from zlc_atom.nodes.camera_measurement import logic_node as camera_logic_node
from zlc_atom.nodes.occupancy import logic_node as occupancy_logic_node
from zlc_runtime import MonitorCoverage, SignalDataPlane
import zlc_runtime.host as runtime_host
from zlc_workbench.logic import (
    LogicCatalog,
    LogicDraft,
    build_arguments,
    finalize_logic_draft,
    make_host,
    stable_signal_key,
)
import zlc_workbench.image_overlay as image_overlay_module
from zlc_workbench.image_overlay import image_frame_from_publication
from zlc_workbench.session import Workspace


def _one_camera_window_program():
    from zlc_pulse import (
        PulsePeriod,
        PulseSequence,
        compile_sequence,
        load_streamer_config,
        pulse_target_from_xdc,
    )

    config = load_streamer_config()
    target = pulse_target_from_xdc(config_path=config["source"])
    trigger_index = target.raw_lanes.index(target.by_key["emCCD"].lanes[0])
    high = [0] * len(target.raw_lanes)
    high[trigger_index] = 1
    sequence = PulseSequence(
        "one_camera_window",
        target,
        1e9 / config["clock_hz"],
        (
            PulsePeriod("expose", 0.005, "s", tuple(high)),
            PulsePeriod(
                "close",
                1e9 / config["clock_hz"],
                "ns",
                (0,) * len(high),
            ),
        ),
    )
    return compile_sequence(sequence, config["params"], config["clock_hz"])


def _wait_terminal(host: object, *, phase: str, timeout: float = 10.0) -> object:
    deadline = time.monotonic() + timeout
    observation = host.poll()
    while not observation.terminal and time.monotonic() < deadline:
        time.sleep(0.005)
        observation = host.poll()
    assert observation.terminal, observation
    assert observation.phase == phase, observation
    return observation


def _wait_armed(host: object, camera: object, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not camera.capture_state() and time.monotonic() < deadline:
        observation = host.poll()
        if observation.terminal:
            break
        time.sleep(0.005)
    assert camera.capture_state(), host.observation


def _wait_revision(
    host: object,
    plane: SignalDataPlane,
    signal: str,
    *,
    after: int,
    timeout: float = 5.0,
) -> tuple[object, object, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        host.poll()
        front = plane.freeze()
        value = front.value(signal)
        publication = front.publication(signal)
        if value is not None and publication is not None:
            revision = int(value.snapshot.ref.revision.value)
            if revision > after:
                return publication, value, revision
        if host.terminal:
            break
        time.sleep(0.005)
    raise AssertionError(f"{signal!r} did not advance beyond revision {after}")


def _cleanup_host(host: object) -> None:
    if host.running:
        host.cancel("Guard A cleanup")
        deadline = time.monotonic() + 5.0
        while host.running and time.monotonic() < deadline:
            host.poll()
            time.sleep(0.005)
    host.shutdown()


def test_guard_a_headless_virtual_chain(tmp_path: Path) -> None:
    """All P0 nodes run through catalog, descriptor, and NodeHost seams."""

    print(calibration_logic_node.__file__)
    print(camera_logic_node.__file__)
    print(occupancy_logic_node.__file__)
    print(runtime_host.__file__)
    print(image_overlay_module.__file__)

    catalog = LogicCatalog()
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    hosts: list[object] = []
    try:
        pulse_directory = tmp_path / "pulses"
        pulse_directory.mkdir()
        (pulse_directory / "imaging_template.json").write_bytes(
            calibration_pulse_template_bytes()
        )
        workspace = Workspace(tmp_path).prepare()
        assert {"calibration", "camera_measurement", "occupancy"} <= set(
            catalog.by_name
        )
        camera = installation.capability("camera.adapter", key="camera")
        sequencer = installation.device("sequencer")
        assert sequencer.world is installation.world

        calibration_descriptor = catalog.get("calibration")
        assert calibration_descriptor is not None
        calibration_finalization = finalize_logic_draft(
            calibration_descriptor,
            LogicDraft(
                values={
                    "pulse_template": "imaging_template.json",
                    "repeats": 30,
                    "reference_exposure_seconds": 0.02,
                    "readout_exposure_seconds": 0.005,
                },
                device_keys={"camera": "camera", "sequencer": "sequencer"},
            ),
            installation=installation,
            signal_plane=plane,
            workspace=workspace,
        )
        assert calibration_finalization.can_start, calibration_finalization.issues
        calibration_resource = calibration_finalization.resources["pulse_template"]
        assert isinstance(calibration_resource, ResolvedWorkspaceResource)
        assert calibration_resource.path == (
            pulse_directory / "imaging_template.json"
        ).resolve()
        calibration_arguments = build_arguments(
            calibration_descriptor,
            signal_plane=plane,
            finalization=calibration_finalization,
            extras={"artifact_directory": tmp_path},
        )
        calibration_node = calibration_descriptor.instantiate(
            **calibration_arguments
        )
        calibration_host = make_host(
            calibration_descriptor,
            calibration_node,
            authored_values=calibration_finalization.values,
            signal_plane=plane,
            instance_id="calibration",
        )
        hosts.append(calibration_host)

        calibration_host.start()
        _wait_terminal(calibration_host, phase="done")
        first_calibration = calibration_host.final_result
        assert first_calibration is not None
        assert first_calibration.artifact_path.parent == tmp_path.resolve()
        assert first_calibration.artifact_path.is_file()
        assert set(
            json.loads(first_calibration.artifact_path.read_text(encoding="utf-8"))
        ) == {
            "site_map",
            "models",
            "default_model_kind",
            "frame_contract",
            "report",
        }
        assert first_calibration.calibration.n_sites == len(
            installation.world.geometry.site_centers_xy
        )
        assert tuple(
            (output.name, output.contract_id)
            for output in calibration_descriptor.outputs
        ) == (("capture_preview", "calibration.capture-preview.v1"),)
        assert {
            path.name
            for path in (
                first_calibration.artifact_path.with_suffix("") / "report"
            ).iterdir()
        } == {
            "site_map.png",
            "fidelity.png",
            "box.png",
            "psf.png",
            "uniform_psf.png",
            "psf_kernels.png",
        }
        assert plane.freeze().signals == {}

        plane.retire(calibration_host)
        assert plane.freeze().signals == {}
        calibration_host.start()
        _wait_terminal(calibration_host, phase="done")
        second_calibration = calibration_host.final_result
        assert second_calibration is not None
        assert second_calibration.artifact_path.is_file()
        assert second_calibration.artifact_path != first_calibration.artifact_path
        assert {first_calibration.artifact_path.name, second_calibration.artifact_path.name} == {
            "calibration.json",
            "calibration-2.json",
        }
        assert {
            path.name
            for path in (
                second_calibration.artifact_path.with_suffix("") / "report"
            ).iterdir()
        } == {
            "site_map.png",
            "fidelity.png",
            "box.png",
            "psf.png",
            "uniform_psf.png",
            "psf_kernels.png",
        }
        assert plane.freeze().signals == {}

        one_window_program = _one_camera_window_program()
        sequencer.load(one_window_program)
        camera_descriptor = catalog.get("camera_measurement")
        assert camera_descriptor is not None
        finite_finalization = finalize_logic_draft(
            camera_descriptor,
            LogicDraft(
                values={
                    "exposure_seconds": 0.005,
                    "repeat": 3,
                    "frames_per_cycle": 1,
                },
                device_keys={"camera": "camera"},
            ),
            installation=installation,
            signal_plane=plane,
            workspace=workspace,
        )
        assert finite_finalization.can_start, finite_finalization.issues
        finite_arguments = build_arguments(
            camera_descriptor,
            signal_plane=plane,
            finalization=finite_finalization,
        )
        finite_node = camera_descriptor.instantiate(**finite_arguments)
        finite_host = make_host(
            camera_descriptor,
            finite_node,
            authored_values=finite_finalization.values,
            signal_plane=plane,
            instance_id="camera_measurement",
        )
        hosts.append(finite_host)
        finite_host.start()
        _wait_armed(finite_host, camera)
        for _ in range(3):
            sequencer.fire()
            assert sequencer.wait_done(1.0) is not None
        _wait_terminal(finite_host, phase="done")
        assert finite_host.final_result == {
            "cycles": 3,
            "signals": (stable_signal_key("camera_measurement", "frame_0"),),
        }
        assert camera.capture_state() is False

        frames_signal = finite_host.signal_key("frame_0")
        assert frames_signal == stable_signal_key("camera_measurement", "frame_0")
        finite_front = plane.freeze()
        frames_publication = finite_front.publication(frames_signal)
        frames_value = finite_front.value(frames_signal)
        assert frames_publication is not None and frames_value is not None
        assert frames_value.coverage is None
        assert frames_value.transient is False
        assert frames_value.shape[:2] == (3, 1)
        assert not plane.is_generation_live(frames_signal)

        occupancy_descriptor = catalog.get("occupancy")
        assert occupancy_descriptor is not None
        occupancy_finalization = finalize_logic_draft(
            occupancy_descriptor,
            LogicDraft(
                source_signal=frames_signal,
                artifact_inputs={
                    "calibration_path": str(first_calibration.artifact_path),
                },
            ),
            installation=installation,
            signal_plane=plane,
            workspace=workspace,
            source_options=(frames_signal,),
        )
        assert occupancy_finalization.can_start, occupancy_finalization.issues
        occupancy_arguments = build_arguments(
            occupancy_descriptor,
            signal_plane=plane,
            finalization=occupancy_finalization,
        )
        assert set(occupancy_arguments) == {
            "calibration",
            "source_signal",
            "model_kind",
        }
        assert isinstance(occupancy_arguments["calibration"], ResolvedArtifact)
        assert occupancy_arguments["calibration"].path == (
            first_calibration.artifact_path.resolve()
        )
        assert occupancy_arguments["model_kind"] == "default"
        assert occupancy_arguments["source_signal"] == frames_signal
        occupancy_node = occupancy_descriptor.instantiate(**occupancy_arguments)
        occupancy_host = make_host(
            occupancy_descriptor,
            occupancy_node,
            authored_values=occupancy_finalization.values,
            signal_plane=plane,
            instance_id="occupancy",
        )
        hosts.append(occupancy_host)
        occupancy_host.start()
        _wait_terminal(occupancy_host, phase="done")

        occupancy_front = plane.freeze()
        counts_signal = occupancy_host.signal_key("counts")
        occupancy_publication = occupancy_front.publication(counts_signal)
        assert occupancy_publication is not None
        expected_outputs = {
            occupancy_host.signal_key(output.name)
            for output in occupancy_descriptor.outputs
        }
        assert set(occupancy_publication.signals) == expected_outputs
        assert expected_outputs == {
            stable_signal_key("occupancy", "counts"),
            stable_signal_key("occupancy", "occupied"),
            stable_signal_key("occupancy", "valid"),
            stable_signal_key("occupancy", "rate"),
            stable_signal_key("occupancy", "frame_judged"),
            stable_signal_key("occupancy", "site_overlay"),
        }
        counts = occupancy_publication.value(counts_signal)
        occupied = occupancy_publication.value(
            occupancy_host.signal_key("occupied")
        )
        valid = occupancy_publication.value(occupancy_host.signal_key("valid"))
        rate = occupancy_publication.value(occupancy_host.signal_key("rate"))
        frame_judged = occupancy_publication.value(
            occupancy_host.signal_key("frame_judged")
        )
        site_overlay = occupancy_publication.value(
            occupancy_host.signal_key("site_overlay")
        )
        assert all(
            value is not None
            for value in (
                counts,
                occupied,
                valid,
                rate,
                frame_judged,
                site_overlay,
            )
        )
        n_sites = first_calibration.calibration.site_map.n_sites
        assert counts.shape == occupied.shape == valid.shape == (3, n_sites, 1)
        assert rate.shape == (3, 1, 1)
        np.testing.assert_array_equal(
            frame_judged.values,
            frames_value.values,
        )
        assert all(
            value.coverage is None and value.transient is False
            for value in occupancy_publication.signals.values()
        )
        assert plane.direct_parent_publications(occupancy_publication) == (
            frames_publication,
        )
        assert occupancy_publication.run_record["parameters"] == {
            "frames_signal": frames_signal,
            "calibration_path": str(first_calibration.artifact_path),
            "model_kind": "box",
        }
        overlay = image_frame_from_publication(
            frame_judged,
            occupancy_publication,
            overlay_signal=occupancy_host.signal_key("site_overlay"),
            overlay_revision=1,
        )
        np.testing.assert_allclose(
            overlay.overlay.coordinates,
            first_calibration.calibration.site_map.centers_xy,
        )
        assert overlay.overlay.point_ids == (
            first_calibration.calibration.site_map.site_ids
        )
        assert overlay.snapshot is frame_judged.snapshot

        occupancy_host.shutdown()
        finite_host.shutdown()
        plane.retire(finite_host)
        assert plane.freeze().signals == {}

        sequencer.load(one_window_program)
        infinite_finalization = finalize_logic_draft(
            camera_descriptor,
            LogicDraft(
                values={
                    "exposure_seconds": 0.005,
                    "repeat": 0,
                    "frames_per_cycle": 1,
                },
                device_keys={"camera": "camera"},
            ),
            installation=installation,
            signal_plane=plane,
            workspace=workspace,
        )
        assert infinite_finalization.can_start, infinite_finalization.issues
        infinite_arguments = build_arguments(
            camera_descriptor,
            signal_plane=plane,
            finalization=infinite_finalization,
        )
        infinite_node = camera_descriptor.instantiate(**infinite_arguments)
        infinite_host = make_host(
            camera_descriptor,
            infinite_node,
            authored_values=infinite_finalization.values,
            signal_plane=plane,
            instance_id="camera_measurement",
        )
        hosts.append(infinite_host)
        infinite_host.start()
        _wait_armed(infinite_host, camera)

        live_signal = infinite_host.signal_key("frame_0")
        previous_revision = 0
        live_generation = None
        for _ in range(2):
            sequencer.fire()
            assert sequencer.wait_done(1.0) is not None
            live_publication, live_value, revision = _wait_revision(
                infinite_host,
                plane,
                live_signal,
                after=previous_revision,
            )
            assert isinstance(live_value.coverage, MonitorCoverage)
            assert live_value.transient is True
            assert live_value.run_record["parameters"]["repeat"] == 0
            generation = live_publication.event_ref.generation
            if live_generation is None:
                live_generation = generation
            else:
                assert generation == live_generation
            previous_revision = revision

        infinite_host.cancel("Guard A repeat-zero stop")
        _wait_terminal(infinite_host, phase="cancelled")
        assert camera.capture_state() is False
        assert plane.latest_publication(live_signal) is None

        for host in reversed(hosts):
            host.shutdown()
        assert all(
            host.terminal and not host.running and host.worker_idle for host in hosts
        )
        assert plane.freeze().signals == {}
        plane.retire(calibration_host)
        assert plane.freeze().signals == {}
        assert camera.capture_state() is False
        assert sequencer.snapshot()["firing"] is False
    finally:
        for host in reversed(hosts):
            _cleanup_host(host)
        plane.close()
        installation.close()
