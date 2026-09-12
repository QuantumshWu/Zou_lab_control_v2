"""The stepped scan node: the HOST applies each point, then takes its shots.

The end-to-end test IS the goal this engine was built for: scan the three
bias DACs on the virtual bench and find the MOT optimum the simulation
planted, watching a free-running monitor.  Nothing here reads the world's
ground truth except to say where the answer should have landed.

The gating tests ask the other question this node owns -- which publications
it KEPT -- against a source whose every publication is named.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from zlc_data import owned_snapshot_from_arrays
from zlc_pulse import (
    compile_sequence,
    load_streamer_config,
    resolve_api_parameters,
)
from zlc_pulse.device import BoardDescription, ConfigValueHolder
from zlc_runtime import MonitorCoverage, NodeHost, SignalDataPlane, SignalValue

from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.devices.simulation import DEFAULT_MOT_FIELD_OPTIMUM_DAC
from zlc_atom.install import create_installation, tunable_devices
from zlc_atom.nodes import (
    ResolvedWorkspaceResource,
    discover_logic_nodes,
)
from tests.pulse_fixture import pulse_sequence
from zlc_atom.nodes.scan import (
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    SCAN_PULSE_CONTRACT,
    SCAN_OUTPUT,
    ScanAxis,
    ScanDatasetWriter,
    ScanPlan,
    ScanPort,
    check_cancelled,
    settle,
)
from zlc_atom.nodes.scan.devices import ScanDeviceKnobs
from zlc_atom.nodes.stepped_scan import GATING_MODES, STEPPED_SCAN_SCHEMA
from zlc_atom.nodes.stepped_scan.measurement import SteppedScanMeasurement

from tests.fakes import (
    SCRIPTED_SEED_VALUE,
    ScriptedScanBench,
    camera_cycle_snapshot,
)
from test_scan_repeat_domain import _source_schema


class _FakeSequencer(ConfigValueHolder):
    """The board's surface with no board behind it: every command counted.

    ``on_safe`` runs inside SAFE, which is where an operator's Stop lands
    while the board is acknowledging: the one moment the engines used to
    read too late.
    """

    def __init__(self, sequence) -> None:
        self._init_config_values()
        settings = load_streamer_config()
        self.board = BoardDescription(
            sequence.target, settings["params"], settings["clock_hz"]
        )
        from zlc_pulse import authored_config_entries
        self.load_config_values(authored_config_entries(sequence))
        self.fires = 0
        self.safe_calls = 0
        self.on_safe = None

    def describe(self):
        return self.board

    def safe(self) -> None:
        self.safe_calls += 1
        if self.on_safe is not None:
            self.on_safe()

    def load(self, program, **_kwargs) -> None:
        self.program = program

    def fire(self, **_kwargs) -> None:
        self.fires += 1

    def wait_done(self, _timeout):
        return SimpleNamespace(fault=None)


class _FakeSource:
    """A point's value on demand; ``fail_at`` names the take that fails and
    ``on_take`` sees every take, so a test can press Stop at a moment."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.taken = 0
        self.fail_at = fail_at
        self.on_take = None

    def open(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def validate(self, *_args, **_kwargs) -> None:
        pass

    def arm(self) -> None:
        pass

    def discard_pending(self) -> None:
        pass

    def describe(self) -> dict:
        return {"source_signal": "fake"}

    def next_value(self, context):
        self.taken += 1
        check_cancelled(context)
        if self.taken == self.fail_at:
            raise RuntimeError("scripted source failed")
        schema = _source_schema(shots=1)
        snapshot = owned_snapshot_from_arrays(
            schema,
            np.zeros(schema.physical_shape),
            self.taken,
            stream_generation="fake-source",
        )
        if self.on_take is not None:
            self.on_take(self.taken)
        return SignalValue("fake-source", snapshot, MonitorCoverage(1, 1)), None


class _Knob:
    """One installed device with one field, remembering every tune.

    ``refuse_restore`` raises the device's refusal to accept the restore.
    """

    def __init__(self, level: float = 0.25, *, refuse_restore: bool = False) -> None:
        self.level = level
        self.tunes: list[float] = []
        self.refuse_restore = refuse_restore

    def tune(self, field: str, value: float) -> float:
        assert field == "level"
        self.tunes.append(float(value))
        if self.refuse_restore and value == 0.25:
            raise RuntimeError("scripted device refused restore")
        self.level = float(value)
        return self.level

    def tunable_fields(self) -> tuple[TunableField, ...]:
        return (
            TunableField(
                AuthoringField("level", "float", "level", 0.25, minimum=0.0, maximum=3.0),
                self.level,
                True,
                ("level",),
            ),
        )

    def tunable_values(self) -> dict:
        return {"level": self.level}

    def settings_provenance(self) -> dict:
        return {"device_session_id": "knob", "settings_epoch": 0}


class _Context:
    """The host's surface: Stop is a flag, commits are counted."""

    def __init__(self) -> None:
        self.cancelled = False
        self.commits = 0

    def cancel_requested(self) -> bool:
        return self.cancelled

    def commit_live(self, outputs, *, source_publication=None) -> None:
        del outputs, source_publication
        self.commits += 1

    def report_progress(self, *_args, **_kwargs) -> None:
        pass

    def current_dataset(self, name: str) -> str:
        return name


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
    return pulse_sequence("mot_field_template.json")


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
    sequencer.fire(run_repeats=1, scan_repeats=1)
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

    Returns the kept shots as (scan/run repeat, plan row) values -- each cell is the
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
        # (scan repeat x run repeat, plan row, y, x): every pixel of a
        # scripted frame carries the publication's index, so the cell mean IS
        # the shot that landed there.
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
        compilations = []
        original_compile = bench.compile_pulse
        def counted_compile(*args, **kwargs):
            compilations.append(True)
            return original_compile(*args, **kwargs)
        monkeypatch.setattr(bench, 'compile_pulse', counted_compile)

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
        assert len(compilations) == 2, 'a discarded default pulse was compiled before point one'
        assert bench.loads == 1, "the invalid second program reached LOAD"
        assert bench.fired_repeats == [(1, 1)]
    finally:
        if host is not None:
            host.shutdown()
        if bench is not None:
            bench.close()
        plane.close()
        installation.close()


def test_shots_are_run_repeats_and_repeats_rescan_the_plan() -> None:
    """The board executes finite Run repeats; the host reapplies each point once."""

    kept, bench = _scripted_run(
        gating="sw_gated", shots=2, repeats=2, settle=0.0
    )
    assert kept.shape == (4, 2)
    assert SCRIPTED_SEED_VALUE not in kept.reshape(-1).tolist()
    assert bench.published == [SCRIPTED_SEED_VALUE, *range(64)]
    assert bench.loads == 4
    assert bench.loaded_loop_counts == [1, 1, 1, 1]
    assert all(source.bracket is None for source in bench.loaded_sources)
    assert bench.fired_repeats == [(2, 1)] * 4
    assert sum(kind == "fire" for kind, _when in bench.events) == 4
    assert len(bench.stop_intervals()) == 1, "only Start needs an explicit SAFE"


def test_pulse_gated_keeps_exactly_one_publication_per_fired_shot() -> None:
    """The fire's Run-repeat count owns shots; the host applies each point once."""

    kept, bench = _scripted_run(gating="pulse_gated", shots=2, settle=0.0)
    assert kept.tolist() == [[0.0, 2.0], [1.0, 3.0]]
    assert bench.published == [SCRIPTED_SEED_VALUE, 0, 1, 2, 3]
    assert bench.loads == 2
    assert bench.loaded_loop_counts == [1, 1]
    assert bench.fired_repeats == [(2, 1)] * 2
    assert sum(kind == "fire" for kind, _when in bench.events) == 2


def test_the_authored_settle_time_is_observed_before_every_fire(monkeypatch) -> None:
    """The authored wait remains per point; DONE needs no repeated SAFE."""

    import zlc_atom.nodes.stepped_scan.measurement as stepped_module
    original_settle = stepped_module.settle
    intervals = []
    def timed_settle(context, seconds):
        started = time.monotonic()
        original_settle(context, seconds)
        intervals.append(time.monotonic() - started)
    monkeypatch.setattr(stepped_module, 'settle', timed_settle)
    _kept, bench = _scripted_run(
        gating="sw_gated", shots=1, settle=AUTHORED_SETTLE_SECONDS
    )
    assert len(intervals) == 2
    assert len(bench.stop_intervals()) == 1
    for interval in intervals:
        assert interval >= AUTHORED_SETTLE_SECONDS, (
            f"the board was stopped for only {interval:.3f}s, less than the "
            f"authored {AUTHORED_SETTLE_SECONDS}s"
        )
        assert interval < AUTHORED_SETTLE_SECONDS + 1.0, (
            f"the stop of {interval:.3f}s is not the authored "
            f"{AUTHORED_SETTLE_SECONDS}s"
        )


def test_device_scan_keeps_actual_readback_without_comparing_the_setpoint() -> None:
    class QuantizedDevice:
        def tune(self, _field: str, value: float) -> float:
            return float(value) + 0.5

        def tunable_fields(self) -> tuple[TunableField, ...]:
            return (
                TunableField(
                    AuthoringField("gain", "float", "gain", 0.0, minimum=0.0, maximum=24.0),
                    0.0,
                    True,
                    ("gain",),
                ),
            )

    port_name = DEVICE_PARAM_FAMILY + "camera:gain"
    tunables = {"camera": QuantizedDevice()}
    knobs = ScanDeviceKnobs(tunables)
    assert knobs.move(port_name, 3.0) == 3.5
    knobs.restore()  # A rounded restore readback is not a device refusal.

    from zlc_atom.nodes.scan.devices import tune_value

    requested = -12.83089524390248
    returned = -12.830898342271585
    device = SimpleNamespace(tune=lambda _field, _value: returned)
    assert tune_value(device, "ch1_power", requested) == returned


def _device_stepped(knob: _Knob, sequencer: _FakeSequencer, source: _FakeSource):
    port = ScanPort(DEVICE_PARAM_FAMILY + "knob:level", "knob.level", "1", 0.0, 3.0)
    return SteppedScanMeasurement(
        sequencer=sequencer,
        source=source,
        sequence=_template_sequence(),
        pulse_path=Path("scan_template.json"),
        plan=ScanPlan((ScanAxis(port.port, (1.0, 2.0)),)),
        ports=(port,),
        repeats=1,
        shots_per_point=1,
        settle_seconds=0.0,
        gating="pulse_gated",
        free_run_delay_seconds=0.0,
        tunables={"knob": knob},
    )


def test_every_knob_the_scan_moved_is_put_back_however_the_scan_ends(monkeypatch) -> None:
    """The bench is handed back as it was found: complete, stopped or failed.

    A device axis left the instrument standing at the last scan point --
    2.0 here, whatever the operator had set before -- after a finished
    scan, after Stop and after a source failure alike.  The pre-run value
    is read from the device before the first move and written back, through
    the same verified tune, when the scan ends; a refusal to go back is the
    run's error when everything else succeeded, and a note on the original
    error when it did not.
    """

    knob, sequencer, source = _Knob(), _FakeSequencer(_template_sequence()), _FakeSource()
    import zlc_atom.nodes.stepped_scan.measurement as stepped_module
    observed_settles = []
    original_settle = stepped_module.settle
    def observe_settle(context, seconds):
        observed_settles.append(knob.level)
        original_settle(context, seconds)
    monkeypatch.setattr(stepped_module, 'settle', observe_settle)
    compiled = []
    original_compile = sequencer.compile_pulse
    def compile_once(*args, **kwargs):
        compiled.append(True)
        return original_compile(*args, **kwargs)
    monkeypatch.setattr(sequencer, 'compile_pulse', compile_once)
    _device_stepped(knob, sequencer, source).execute(_Context())
    assert observed_settles == [1.0, 2.0], 'settle happened before the device move'
    assert len(compiled) == 1, 'a device-only scan compiled the unchanged pulse again'
    assert knob.tunes == [1.0, 2.0, 0.25]
    assert knob.level == 0.25
    assert sequencer.fires == 2
    assert sequencer.safe_calls == 1, "normal DONE is already safe"

    knob, sequencer = _Knob(), _FakeSequencer(_template_sequence())
    with pytest.raises(RuntimeError, match="scripted source failed") as failure:
        _device_stepped(knob, sequencer, _FakeSource(fail_at=2)).execute(_Context())
    assert knob.tunes == [1.0, 2.0, 0.25] and knob.level == 0.25
    assert not getattr(failure.value, "__notes__", []), "a clean restore adds nothing"

    knob, sequencer, source, context = (
        _Knob(), _FakeSequencer(_template_sequence()), _FakeSource(), _Context()
    )
    source.on_take = lambda taken: setattr(context, "cancelled", taken == 1)
    with pytest.raises(RuntimeError, match="cancelled"):
        _device_stepped(knob, sequencer, source).execute(context)
    assert knob.tunes == [1.0, 0.25] and knob.level == 0.25
    assert sequencer.fires == 1, "Stop after the first point applied no second"

    knob, sequencer = _Knob(refuse_restore=True), _FakeSequencer(_template_sequence())
    with pytest.raises(RuntimeError, match="scripted device refused restore"):
        _device_stepped(knob, sequencer, _FakeSource()).execute(_Context())
    assert sequencer.safe_calls == 1, "the completed pulse is safe before restore"

    knob, sequencer = _Knob(refuse_restore=True), _FakeSequencer(_template_sequence())
    with pytest.raises(RuntimeError, match="scripted source failed") as failure:
        _device_stepped(knob, sequencer, _FakeSource(fail_at=2)).execute(_Context())
    assert any(
        "restoring the scanned device fields also reported" in note
        and "'level' of 'knob' was not put back to its pre-run value 0.25" in note
        for note in failure.value.__notes__
    ), "the original failure stays the failure; the restore refusal rides on it"

    # A knob standing where no tune could put it back -- outside the range
    # its device says may be commanded -- is refused before it is moved.
    knob, sequencer = _Knob(level=5.0), _FakeSequencer(_template_sequence())
    with pytest.raises(ValueError, match=r"stands at 5.0.*outside \[0.0, 3.0\]"):
        _device_stepped(knob, sequencer, _FakeSource()).execute(_Context())
    assert knob.tunes == [] and sequencer.fires == 0


def test_a_stop_received_while_the_board_goes_safe_fires_nothing_more() -> None:
    """Stop is read before anything new is done to the bench.

    Stop arrived while the board was acknowledging SAFE before a point;
    the engine came back from SAFE, settled, moved the knob, loaded and
    FIRED the point, and noticed the flag only in the read-out loop.  The
    settle is now sliced with the flag read between slices, and the flag
    is read again before a knob moves and before a fire.
    """

    knob, sequencer, source, context = (
        _Knob(), _FakeSequencer(_template_sequence()), _FakeSource(), _Context()
    )
    sequencer.on_safe = lambda: setattr(context, "cancelled", True)
    with pytest.raises(RuntimeError, match="cancelled"):
        _device_stepped(knob, sequencer, source).execute(context)
    assert sequencer.fires == 0
    assert knob.tunes == [], "nothing moved after Stop"
    assert knob.level == 0.25

    class _StopOnThirdAsk:
        asked = 0

        def cancel_requested(self) -> bool:
            self.asked += 1
            return self.asked >= 3

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="cancelled"):
        settle(_StopOnThirdAsk(), 30.0)
    assert time.monotonic() - started < 1.0, "a settle stays stoppable while it sleeps"


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
                    DEVICE_PARAM_FAMILY + "mot_camera:exposure",
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
        assert claim.protected_fields == ("exposure",)
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
        # The file names the pulse; the board's snapshot carries the filled
        # template every point was compiled from.
        assert published.run_record["pulse"]["name"] == Path(TEMPLATE_NAME).stem
        document = published.run_record["device_snapshots"]["sequencer"]["pulse"]
        assert document["periods"], "the played timing travels with the record"
        assert published.run_record["device_snapshots"][role]["fields"][
            "exposure"
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
            output = writer.write(
                source,
                row=row,
                scan_repeat=0,
                run_repeat=0,
            )
            assert output.snapshot is source.snapshot
            assert output.cell_origin == (0, row)
            assert tuple(
                axis.name for axis in output.canonical_schema.repeat_domain.axes
            ) == ("repeat", "shots per point")
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
                {
                    SCAN_OUTPUT.name: writer.write(
                        source,
                        row=row,
                        scan_repeat=0,
                        run_repeat=0,
                    )
                },
            )
        assert plane.seal_committed(producer, cut_short=True)
        current = plane.current_dataset(producer.signal_key(SCAN_OUTPUT.name))
        valid = current.expanded_validity()
        assert valid[:, :2].all()
        assert not valid[:, 2:].any()
    finally:
        plane.close()
