"""The stepped scan node: the HOST applies each point, then takes its shots.

The end-to-end test IS the goal this engine was built for: scan the three
bias DACs on the virtual bench and find the MOT optimum the simulation
planted, watching a free-running monitor.  Nothing here reads the world's
ground truth except to say where the answer should have landed.

The gating tests ask the other question this node owns -- which publications
it KEPT -- against a source whose every publication is named.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from zlc_pulse import compile_sequence, resolve_api_parameters, sequence_from_tree
from zlc_runtime import MonitorCoverage, NodeHost, SignalDataPlane, SignalValue

from zlc_atom.devices.simulation import DEFAULT_MOT_FIELD_OPTIMUM_DAC
from zlc_atom.install import create_installation, tunable_devices
from zlc_atom.nodes import (
    ResolvedWorkspaceResource,
    discover_logic_nodes,
    scan_pulse_template_bytes,
)
from zlc_atom.nodes.scan import (
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    SCAN_PULSE_CONTRACT,
    SCAN_OUTPUT,
    ScanAxis,
    ScanDatasetWriter,
    ScanPlan,
    ScanPort,
)
from zlc_atom.nodes.stepped_scan import GATING_MODES, STEPPED_SCAN_SCHEMA
from zlc_atom.nodes.stepped_scan.measurement import SteppedScanMeasurement

from tests.fakes import (
    SCRIPTED_SEED_VALUE,
    ScriptedScanBench,
    camera_cycle_snapshot,
)


TEMPLATE_NAME = "mot_field_template.json"
BIAS_PORTS = tuple(
    PULSE_PARAM_FAMILY + name
    for name in ("da_bias_x", "da_bias_y", "da_bias_z")
)
#: Long enough that no scheduling jitter could produce it, short enough to pay.
AUTHORED_SETTLE_SECONDS = 0.37


def _scan_host(node: object, plane: SignalDataPlane) -> NodeHost:
    return NodeHost(
        node,
        plane,
        instance_id=node.instance_id,
        kind="measurement",
        dataset_output_declarations=(SCAN_OUTPUT,),
        input_signal=node.source.signal_name,
        input_delivery="exact",
    )


def _source_value(size: int = 64) -> SignalValue:
    snapshot = camera_cycle_snapshot(
        ((np.ones((size, size), dtype=np.uint16),),),
        producer="scan-source",
        revision=1,
    )
    return SignalValue("@logic/source/frames", snapshot, MonitorCoverage(1, 1))


def _template_sequence():
    return sequence_from_tree(json.loads(scan_pulse_template_bytes().decode("utf-8")))


def _pulse_resource(sequence):
    return ResolvedWorkspaceResource(
        Path(TEMPLATE_NAME), SCAN_PULSE_CONTRACT, sequence
    )


def _seed_the_mot_monitor(sequencer, monitor, plane) -> str:
    """One fire at the authored zeros, then wait for the monitor's first frame."""

    sequence = _template_sequence()
    board = sequencer.describe()
    seeded = resolve_api_parameters(sequence)
    sequencer.load(
        compile_sequence(seeded, board.geometry, board.clock_hz), source=seeded
    )
    sequencer.fire()
    sequencer.wait_done(5.0)
    deadline = time.monotonic() + 10.0
    signal_name = ""
    while time.monotonic() < deadline and not signal_name:
        monitor.poll()
        # Publications materialise when the plane FREEZES -- in the product
        # that is the console beat's act; here the test plays it.
        plane.freeze()
        for name in plane.describe_signals():
            text = str(getattr(name, "name", name))
            if "/frame" in text:
                signal_name = text
        time.sleep(0.02)
    assert signal_name, "the MOT monitor never published a frame"
    return signal_name


def _scripted_run(
    *,
    gating: str,
    shots: int,
    settle: float,
    repeats: int = 1,
    values: tuple[float, ...] = (-256.0, 256.0),
) -> tuple[np.ndarray, ScriptedScanBench]:
    """Run the node over a source whose every publication is named.

    Returns the kept shots as (visit, plan row) values -- each cell is the
    index of the publication that landed in it -- and the bench that scripted
    them.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    bench = None
    host = None
    try:
        bench = ScriptedScanBench(
            installation.device("sequencer"),
            plane,
            # One fire plays the pulse's whole-span repeat.  A free-running
            # source hands over one boundary straddler plus S samples; a
            # pulse-driven source publishes exactly S pulse-gated samples.
            publications_per_fire=8 * shots if gating == "sw_gated" else shots,
            paced_by_cycle=True,
            publications_per_cycle=8 if gating == "sw_gated" else 1,
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        plan = ScanPlan((ScanAxis(BIAS_PORTS[0], values),))
        node = descriptors["stepped_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(_template_sequence()),
            plan=plan.to_tree(),
            repeats=repeats,
            shots_per_point=shots,
            settle_seconds=settle,
            gating=gating,
            free_run_delay_seconds=0.002 if gating == "sw_gated" else 0.0,
        )
        host = _scan_host(node, plane)
        host.start()
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal, (
            "the stepped scan never finished; it published "
            f"{bench.published} and kept waiting"
        )
        signal = host.signal_key("scan")
        publication = plane.latest_publication(signal)
        assert publication is not None
        (parent,) = plane.direct_parent_publications(publication)
        assert parent.value(bench.signal_name) is not None
        value = plane.current_dataset(signal)
        block = np.asarray(value.block.values, dtype=float)
        # (visit, plan row, y, x): every pixel of a scripted frame carries the
        # publication's index, so the cell mean IS the shot that landed there.
        return block.mean(axis=(2, 3)), bench
    finally:
        if host is not None and not host.observation.terminal:
            host.cancel("test cleanup")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not host.observation.terminal:
                host.poll()
        if host is not None:
            host.shutdown()
        if bench is not None:
            bench.close()
        plane.close()
        installation.close()


def test_gating_is_the_operators_declaration_and_the_old_fields_are_gone() -> None:
    """How freshness is taken is DECLARED on the node, never probed.

    The safe default skips the straddler (right for a free-running monitor
    like the MOT camera); a pulse-driven source keeps every publication.
    """

    field = next(f for f in STEPPED_SCAN_SCHEMA.fields if f.name == "gating")
    assert field.value_type == "choice"
    assert field.default == "sw_gated"
    assert {choice.value for choice in field.choices} == set(GATING_MODES)
    with pytest.raises(ValueError, match="must be one of"):
        STEPPED_SCAN_SCHEMA.project_values(
            {"pulse_template": "t.json", "plan": "{}", "gating": "guess"}
        )
    # The node that modelled two measurements through one field is gone, and
    # so are the words it used.
    assert "capture" not in STEPPED_SCAN_SCHEMA.field_names
    assert "advance" not in STEPPED_SCAN_SCHEMA.field_names


def test_each_resolved_point_is_preflighted_before_its_load(monkeypatch) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    bench = None
    host = None
    try:
        bench = ScriptedScanBench(
            installation.device("sequencer"),
            plane,
            publications_per_fire=1,
            paced_by_cycle=True,
            publications_per_cycle=1,
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        descriptor = {
            value.api_name: value for value in discover_logic_nodes()
        }["stepped_scan"]
        node = descriptor.instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(_template_sequence()),
            plan=ScanPlan((ScanAxis(BIAS_PORTS[0], (-1.0, 1.0)),)).to_tree(),
            repeats=1,
            shots_per_point=1,
            settle_seconds=0.0,
            gating="pulse_gated",
            free_run_delay_seconds=0.0,
        )
        validations = 0

        def validate(*_args, **_kwargs) -> None:
            nonlocal validations
            validations += 1
            if validations == 2:
                raise ValueError("invalid second camera cadence")

        monkeypatch.setattr(node.source, "validate", validate)
        host = _scan_host(node, plane)
        host.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
        assert "invalid second camera cadence" in str(host.observation.error)
        assert validations == 2
        assert bench.loads == 1, "the invalid second program reached LOAD"
        assert bench.fired_cycles == [1]
    finally:
        if host is not None:
            host.shutdown()
        if bench is not None:
            bench.close()
        plane.close()
        installation.close()


def test_shots_are_fire_cycles_and_repeats_rescan_the_plan() -> None:
    """The board executes finite cycles; the host reapplies each point once."""

    kept, bench = _scripted_run(
        gating="sw_gated", shots=2, repeats=2, settle=0.0
    )
    assert kept.shape == (4, 2)
    assert SCRIPTED_SEED_VALUE not in kept.reshape(-1).tolist()
    assert bench.published == [SCRIPTED_SEED_VALUE, *range(64)]
    assert bench.loads == 4
    assert bench.loaded_loop_counts == [1, 1, 1, 1]
    assert all(source.repeat is None for source in bench.loaded_sources)
    assert bench.fired_cycles == [2, 2, 2, 2]
    assert sum(kind == "fire" for kind, _when in bench.events) == 4
    assert len(bench.stop_intervals()) == 4


def test_pulse_gated_keeps_exactly_one_publication_per_fired_shot() -> None:
    """The fire's cycle count owns the shots; the host applies each point once."""

    kept, bench = _scripted_run(gating="pulse_gated", shots=2, settle=0.0)
    assert kept.tolist() == [[0.0, 2.0], [1.0, 3.0]]
    assert bench.published == [SCRIPTED_SEED_VALUE, 0, 1, 2, 3]
    assert bench.loads == 2
    assert bench.loaded_loop_counts == [1, 1]
    assert bench.fired_cycles == [2, 2]
    assert sum(kind == "fire" for kind, _when in bench.events) == 2


def test_the_authored_settle_time_stops_the_board_before_every_point() -> None:
    """The pulse is stopped, and stays stopped for the AUTHORED time."""

    _kept, bench = _scripted_run(
        gating="sw_gated", shots=1, settle=AUTHORED_SETTLE_SECONDS
    )
    intervals = bench.stop_intervals()
    assert len(intervals) == 2, (
        f"one stop per plan point was expected, got {intervals}"
    )
    for interval in intervals:
        assert interval >= AUTHORED_SETTLE_SECONDS, (
            f"the board was stopped for only {interval:.3f}s, less than the "
            f"authored {AUTHORED_SETTLE_SECONDS}s"
        )
        assert interval < AUTHORED_SETTLE_SECONDS + 1.0, (
            f"the stop of {interval:.3f}s is not the authored "
            f"{AUTHORED_SETTLE_SECONDS}s"
        )


def test_device_scan_refuses_an_effective_value_different_from_its_coordinate() -> None:
    class QuantizedDevice:
        def tune(self, _field: str, value: float) -> float:
            return float(value) + 0.5

    port_name = DEVICE_PARAM_FAMILY + "camera:gain_db"
    measurement = SteppedScanMeasurement(
        sequencer=SimpleNamespace(safe=lambda: None),
        source=object(),
        sequence=_template_sequence(),
        plan=ScanPlan((ScanAxis(port_name, (3.0,)),)),
        ports=(ScanPort(port_name, "camera.gain_db", "", 0.0, 24.0),),
        repeats=1,
        shots_per_point=1,
        settle_seconds=0.0,
        gating="pulse_gated",
        free_run_delay_seconds=0.0,
        tunables={"camera": QuantizedDevice()},
    )

    with pytest.raises(RuntimeError, match="applied 3.5, not the scan coordinate 3.0"):
        measurement._apply((3.0,), object())


def test_scanning_a_device_port_moves_the_camera_exposure() -> None:
    """End to end: a ``device:`` axis tunes the camera and the frames show it.

    The plan scans the MOT camera's exposure over a 4x range; the spot's
    photon count is proportional to exposure in the simulated world (the
    read-noise floor is not), so pooled above-floor brightness at the long
    exposure must clearly exceed the short one.  Red if the device family is
    never dispatched -- the exposure then never moves and the two points
    look alike.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {d.api_name: d for d in discover_logic_nodes()}
    sequencer = installation.device("sequencer")
    monitor = None
    host = None
    try:
        monitor_node = descriptors["camera_measurement"].instantiate(
            camera=installation.device("mot_camera"),
            camera_key="mot_camera",
            signal_plane=plane,
            repeat=0,
            # The MOT monitor is a machine-vision camera: it states no
            # conversion, so the run that watches it is in counts and says so.
            photoelectrons=False,
        )
        monitor = monitor_node.monitor()
        signal_name = _seed_the_mot_monitor(sequencer, monitor, plane)

        exposures = (0.02, 0.08)
        plan = ScanPlan(
            (
                ScanAxis(
                    DEVICE_PARAM_FAMILY + "mot_camera:exposure_seconds",
                    exposures,
                ),
            )
        )
        scan_node = descriptors["stepped_scan"].instantiate(
            sequencer=sequencer,
            signal_plane=plane,
            source_signal=signal_name,
            pulse_resource=_pulse_resource(_template_sequence()),
            plan=plan.to_tree(),
            shots_per_point=1,
            settle_seconds=0.02,
            tunable_devices=tunable_devices(installation),
        )
        (claim,) = scan_node.resolved_device_claims()
        assert claim.device_key == "mot_camera"
        assert claim.device is installation.device("mot_camera")
        assert claim.protected_fields == ("exposure_seconds",)
        host = _scan_host(scan_node, plane)
        host.start()
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline and not host.observation.terminal:
            monitor.poll()
            plane.freeze()
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal

        publication = plane.latest_publication(host.signal_key("scan"))
        assert publication is not None
        published = publication.value(host.signal_key("scan"))
        assert published is not None
        role = "tunable:mot_camera"
        assert published.run_record["named_devices"][role] == "mot_camera"
        assert published.run_record["device_snapshots"][role]["fields"][
            "exposure_seconds"
        ]["scan_values"] == exposures
        value = plane.current_dataset(host.signal_key("scan"))
        frames = np.asarray(value.block.values, dtype=float)
        assert frames.shape[1] == len(exposures)
        pooled = np.clip(frames - 12.0, 0.0, None)
        brightness = pooled.sum(
            axis=tuple(axis for axis in range(pooled.ndim) if axis != 1)
        )
        assert brightness[1] > 2.0 * brightness[0], (
            "a 4x exposure did not brighten the spot; the device port was "
            f"never applied (brightness={brightness.round(1).tolist()})"
        )
    finally:
        if host is not None and not host.observation.terminal:
            host.cancel("test cleanup")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not host.observation.terminal:
                host.poll()
        if host is not None:
            host.shutdown()
        if monitor is not None:
            monitor.close()
        installation.close()


def test_scanning_the_bias_dacs_finds_the_planted_mot_optimum() -> None:
    """The goal, end to end: a 3x3x3 field scan lands on the world's optimum.

    The grid is coarse on purpose -- the assertion is that the brightest scan
    point is the grid point NEAREST the planted optimum, computed from the
    ground truth rather than hard-coded, so re-planting the optimum moves the
    expectation with it.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {d.api_name: d for d in discover_logic_nodes()}
    sequencer = installation.device("sequencer")
    monitor = None
    host = None
    try:
        monitor_node = descriptors["camera_measurement"].instantiate(
            camera=installation.device("mot_camera"),
            camera_key="mot_camera",
            signal_plane=plane,
            repeat=0,
            # The MOT monitor is a machine-vision camera: it states no
            # conversion, so the run that watches it is in counts and says so.
            photoelectrons=False,
        )
        monitor = monitor_node.monitor()
        signal_name = _seed_the_mot_monitor(sequencer, monitor, plane)

        values = (-256.0, 0.0, 256.0)
        plan = ScanPlan(tuple(ScanAxis(port, values) for port in BIAS_PORTS))
        scan_node = descriptors["stepped_scan"].instantiate(
            sequencer=sequencer,
            signal_plane=plane,
            source_signal=signal_name,
            pulse_resource=_pulse_resource(_template_sequence()),
            plan=plan.to_tree(),
            shots_per_point=1,
            settle_seconds=0.02,
        )
        host = _scan_host(scan_node, plane)
        host.start()
        scan_signal = host.signal_key("scan")
        live_fill_levels: set[int] = set()
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline and not host.observation.terminal:
            monitor.poll()
            front = plane.freeze()
            # A measurement publishes LIVE while it runs: a panel attaching
            # mid-scan must see the growing dataset, not silence until the
            # end.  This guard is red under an implementation that only
            # publishes a FINAL result.
            live = front.value(scan_signal)
            if live is not None and live.coverage is not None:
                live_fill_levels.add(live.coverage.written_cells)
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal
        partial = {
            level
            for level in live_fill_levels
            if 0 < level < plan.point_count
        }
        assert partial, (
            "the scan never published a partially filled live dataset; "
            f"observed fill levels: {sorted(live_fill_levels)}"
        )

        value = plane.current_dataset(scan_signal)

        frames = np.asarray(value.block.values, dtype=float)
        # (repeat, scan points, event, y, x): one MOT cycle per grid point,
        # its frames on the READOUT_EVENT axis.
        assert frames.shape[1] == plan.point_count
        # Brightness above the read-noise floor; position-independent, so the
        # spot moving with the field cannot fool the metric.
        pooled = np.clip(frames - 12.0, 0.0, None)
        brightness = pooled.sum(
            axis=tuple(axis for axis in range(pooled.ndim) if axis != 1)
        )
        best = plan.rows()[int(np.argmax(brightness))]

        expected = tuple(
            min(values, key=lambda value: abs(value - optimum))
            for optimum in DEFAULT_MOT_FIELD_OPTIMUM_DAC
        )
        assert best == expected, (
            f"the scan says {best} but the planted optimum {DEFAULT_MOT_FIELD_OPTIMUM_DAC} "
            f"is nearest {expected}; brightness={brightness.round(1).tolist()}"
        )
    finally:
        if host is not None and not host.observation.terminal:
            host.cancel("test cleanup")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not host.observation.terminal:
                host.poll()
        if host is not None:
            host.shutdown()
        if monitor is not None:
            monitor.close()
        installation.close()


def test_scan_planner_keeps_chunk_storage_linear_for_50_and_100_points() -> None:
    source = _source_value()

    def committed_bytes(points: int) -> int:
        writer = ScanDatasetWriter(
            tuple((float(index),) for index in range(points)),
            (("x", ""),),
        )
        payload = 0
        for row in range(points):
            output = writer.write(source, row=row, visit=0)
            assert output.snapshot is source.snapshot
            assert output.cell_origin == (0, row)
            payload += output.snapshot.block.values.nbytes
        assert not hasattr(writer, "_values")
        assert not hasattr(writer, "snapshot")
        return payload

    fifty = committed_bytes(50)
    hundred = committed_bytes(100)
    assert hundred == 2 * fifty


def test_partial_scan_current_dataset_has_invalid_future_points() -> None:
    source = _source_value(size=8)
    writer = ScanDatasetWriter(
        ((0.0,), (1.0,), (2.0,), (3.0,)),
        (("x", ""),),
    )
    producer = SimpleNamespace(
        instance_id="partial-scan",
        dataset_output_declarations=(SCAN_OUTPUT,),
        signal_key=lambda name: f"@logic/partial-scan/{name}",
    )
    plane = SignalDataPlane()
    try:
        plane.begin_generation(producer)
        for row in range(2):
            plane.commit_live(
                producer,
                {SCAN_OUTPUT.name: writer.write(source, row=row, visit=0)},
            )
        assert plane.seal_committed(producer, cut_short=True)
        current = plane.current_dataset(producer.signal_key(SCAN_OUTPUT.name))
        valid = current.expanded_validity()
        assert valid[:, :2].all()
        assert not valid[:, 2:].any()
    finally:
        plane.close()
