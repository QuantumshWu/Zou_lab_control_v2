"""What the apparatus was doing when this data was taken.

A saved figure is worthless six months later if it is only arrays.  The record
that makes it readable is the device base state of the run plus what was asked
of it -- and it has to be captured the same way whether the camera was a real
qCMOS or its virtual twin, or the two benches stop being comparable.

Four rules, migrated from the pre-split tree where they were arrived at the hard
way:

ONCE PER RUN, not per shot.
    The devices' state is the *base* state of the run and is held constant
    across its shots.  A parameter that changes during the run IS the scan, and
    it is already in the data; re-snapshotting per shot would record the scan
    twice and disagree with itself.

THROUGH THE PUBLIC CONTRACT ONLY.
    Device state is read through ``snapshot()`` and nothing else.  Reaching into
    a simulator for ground truth would produce a record a real run cannot
    produce, and the virtual bench would stop being a rehearsal for the real one.

PROBED, NOT BRANCHED.
    Every section appears only if the generic probe finds it, so a measurement,
    a processor and a task each record what they actually have without anyone
    writing a per-node-type branch.

FLAT.
    A processor's record hoists its upstream measurement's devices to the top
    level rather than nesting them, so "which apparatus produced this" is in the
    same place whether the figure shows raw frames or something derived from
    them.
"""

from __future__ import annotations

from collections.abc import Mapping
import time

__all__ = ["ProvenanceRecorder", "collect_provenance", "hoist_upstream"]


#: How each kind of device answers "what is your state", in its own contract's
#: words.  A camera's state is its working point -- exposure, gain, ROI, binning,
#: readout mode -- and a sequencer's is its register snapshot.  Both are public
#: contract methods, which is what keeps the record identical for a real device
#: and its virtual twin.
_STATE_READERS = ("capture_working_point", "snapshot")


def _describe(device: object) -> object | None:
    """One device's state, through whichever public reader its contract offers."""

    for reader in _STATE_READERS:
        method = getattr(device, reader, None)
        if not callable(method):
            continue
        try:
            state = method()
        except Exception as error:  # a device that cannot answer says so plainly
            return {"error": f"{type(error).__name__}: {error}"}
        if isinstance(state, Mapping):
            return dict(state)
        fields = getattr(state, "__dataclass_fields__", None)
        if fields is not None:
            return {name: getattr(state, name) for name in fields}
        return state
    return None


def _device_snapshots(node: object) -> dict[str, object]:
    """Every device the node holds, by role, through its public contract."""

    snapshots: dict[str, object] = {}
    for role in ("camera", "sequencer"):
        state = _describe(getattr(node, role, None))
        if state is not None:
            snapshots[role] = state
    return snapshots


def _acquisition_parameters(node: object) -> dict[str, object]:
    """What was asked of the apparatus, as the node was configured to ask it."""

    parameters: dict[str, object] = {}
    for name in ("repeat", "frames_per_cycle", "timeout", "repeats", "reducer"):
        value = getattr(node, name, None)
        if value is None or callable(value):
            continue
        parameters[name] = list(value) if isinstance(value, tuple) else value
    return parameters


#: What a device reported is passed on AS REPORTED.
#:
#: There used to be a plainifier here that turned anything it did not recognise
#: into str(value), and it ran BEFORE the archive's own encoder -- which refuses
#: an ndarray with a message telling the author to pass it as an array instead.
#: So a device reporting an array had it silently stringified, truncated by
#: numpy's repr, and the refusal that would have said so could never fire.
#: One encoder, at the boundary that writes the file.


def collect_provenance(node: object) -> dict[str, object]:
    """Assemble the record for any node, by probing rather than by type."""

    record: dict[str, object] = {
        "node": str(getattr(node, "instance_id", type(node).__name__)),
        "layer": str(getattr(node, "kind", "")),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    devices = _device_snapshots(node)
    if devices:
        record["devices"] = devices

    # A derived node holds no device, so its record would otherwise be empty.
    # Naming what it consumed gives it substance and lets a reader walk upstream.
    consumes = getattr(node, "source_signal", None)
    if consumes:
        record["consumes"] = [str(consumes)]

    parameters = _acquisition_parameters(node)
    if parameters:
        record["acquisition_parameters"] = parameters

    return record


def hoist_upstream(record: Mapping[str, object], upstream: Mapping[str, object]) -> dict[str, object]:
    """Lift an upstream measurement's devices to the top level of a derived record.

    Kept FLAT on purpose: a reader looking for "which apparatus produced this"
    finds it in the same place whether the figure shows frames or something
    derived from them, instead of having to know how deep the derivation went.
    """

    merged = dict(record)
    devices = upstream.get("devices")
    if devices and "devices" not in merged:
        merged["devices"] = devices
    upstream_parameters = upstream.get("acquisition_parameters")
    if upstream_parameters:
        merged.setdefault("source_acquisition_parameters", upstream_parameters)
    return merged


class ProvenanceRecorder:
    """Holds one node's record, captured once and then constant.

    A node owns one of these.  The first read captures; every read after that
    returns the same record, so a run's provenance cannot drift shot to shot.
    """

    __slots__ = ("_record",)

    def __init__(self) -> None:
        self._record: dict[str, object] | None = None

    def capture(self, node: object) -> dict[str, object]:
        """Capture now, if this run has not been captured yet."""

        if self._record is None:
            self._record = collect_provenance(node)
        return dict(self._record)

    def reset(self) -> None:
        """A new run captures afresh."""

        self._record = None

    @property
    def captured(self) -> bool:
        return self._record is not None
