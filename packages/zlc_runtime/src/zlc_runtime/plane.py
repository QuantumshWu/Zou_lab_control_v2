"""Headless committed signal plane with producer-local causal coherence.

Each producer commits one immutable event transaction.  Runtime accumulates
finite datasets, seals their terminal state, and projects a coherent visible
front without calling back into plugin materializers.  Independent producers
advance independently.  Within one explicit source -> Processor component,
however, a newer source and its active descendants replace the previous
component together. A slow Processor therefore cannot expose source revision N
beside its own derived revision N-1.

A Monitor retains only its current event.  A capable derived output gains
source-index history only while at least one consumer holds a bounded lease;
retention begins at that lease's current event, follows the largest active
window, and disappears with the last lease. Runtime can then materialize an
ordinary Dataset in which every retained source index is present and an
uncomputed index is invalid. Signals from different runs still advance
independently; there is no cross-run global counter.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
import threading
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol, TypeAlias, runtime_checkable
import uuid
from weakref import WeakKeyDictionary

import numpy as np
from zlc_data import (
    PRIMARY_INDEX,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    GridTopology,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    owned_snapshot_from_arrays,
)
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from .dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from .dataset import (
    DatasetCoverage,
    MonitorCoverage,
)
from .streams import (
    AcquisitionStream,
    EventRef,
    FollowTap,
    SourceFailed,
    SourceGenerationEnded,
    StreamEndedEarly,
    StreamId,
)
from zlc_data import canonical_text

__all__ = [
    "IndexedHistoryLease",
    "LatestProcessorControl",
    "SignalDataPlane",
    "SignalFront",
    "SignalPublication",
    "SignalProducer",
    "SignalValue",
]


def _run_records_equal(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    try:
        return dict(left) == dict(right)
    except (TypeError, ValueError):
        return False


def _freeze_run_record_value(value: object, path: str) -> object:
    """Take one strict, recursively immutable run-record snapshot."""

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be text")
            frozen[key] = _freeze_run_record_value(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_run_record_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _freeze_run_record(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_run_record_value(value, "run_record")
    assert isinstance(frozen, Mapping)
    return frozen


@runtime_checkable
class SignalProducer(Protocol):
    """Stable routing contract implemented by a hosted producer.

    The plane receives an immutable producer identity and owner declarations;
    it never discovers behavior from a registry or uses Python object identity
    as a routing key.
    """

    instance_id: str
    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]: ...

    def signal_key(self, output_name: str) -> str: ...


@runtime_checkable
class LatestProcessorControl(SignalProducer, Protocol):
    """Public owner callbacks used by the shared latest-only lane."""

    def validate_processor_source(self, source: "SignalValue") -> None: ...

    def evaluate_processor(
        self,
        source: "SignalValue",
        source_publication: "SignalPublication",
    ) -> Mapping[str, LiveDatasetOutput]: ...

    def accept_processor_result(
        self,
        source: "SignalValue",
        source_publication: "SignalPublication",
        result: Mapping[str, LiveDatasetOutput],
    ) -> None: ...

    def accept_processor_failure(self, error: Exception) -> None: ...

    def accept_processor_cancelled(self) -> None: ...

    def request_processor_owner_wake(self) -> None: ...


@dataclass(frozen=True)
class SignalValue:
    """One signal at one producer-owned immutable revision."""

    name: str
    snapshot: OwnedSnapshot
    coverage: DatasetCoverage | MonitorCoverage | None
    run_record: Mapping[str, object] = field(default_factory=dict)
    canonical_schema: DatasetSchema | None = None
    cell_origin: tuple[int, int] | None = None
    primary_index: int | None = None
    event_record: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = canonical_text(self.name, "signal name")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("signal snapshot must be OwnedSnapshot")
        if self.coverage is not None and not isinstance(
            self.coverage,
            (DatasetCoverage, MonitorCoverage),
        ):
            raise TypeError("signal coverage has an unknown type")
        if (self.canonical_schema is None) != (self.cell_origin is None):
            raise ValueError(
                "signal canonical_schema and cell_origin must appear together"
            )
        if self.canonical_schema is not None:
            if not isinstance(self.canonical_schema, DatasetSchema):
                raise TypeError("signal canonical_schema must be DatasetSchema")
            origin = tuple(self.cell_origin)
            if len(origin) != 2 or any(type(value) is not int for value in origin):
                raise TypeError("signal cell_origin must contain two integers")
            object.__setattr__(self, "cell_origin", origin)
        if not isinstance(self.run_record, Mapping):
            raise TypeError("signal value run_record must be a mapping")
        if not isinstance(self.event_record, Mapping):
            raise TypeError("signal value event_record must be a mapping")
        primary_index = self.primary_index
        if primary_index is not None and (
            type(primary_index) is not int or primary_index < 0
        ):
            raise TypeError("signal primary_index must be a non-negative integer or None")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "run_record", _freeze_run_record(self.run_record))
        object.__setattr__(
            self, "event_record", _freeze_run_record(self.event_record)
        )

    # The block is the value; these read off it rather than copying, so two
    # consumers describing "the same signal" cannot describe different data.
    @property
    def block(self):
        """The snapshot's DataBlock -- shape/dtype/schema live here."""

        return self.snapshot.block

    @property
    def schema(self):
        return self.block.schema

    @property
    def values(self):
        """The block's array.  Read-only by ownership: never mutate a frozen block."""

        return self.block.values

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    @property
    def cell_schema(self):
        """The per-cell value schema -- where dtype / unit / data axes live.

        A DatasetSchema describes the DATASET (repeat axis, point axes, layout);
        what a cell actually holds is its ``cell_schema``.  Reading through it
        keeps every consumer on the description the producer declared.
        """

        return self.schema.cell_schema

    @property
    def dtype(self):
        return self.cell_schema.dtype

    @property
    def unit(self) -> str | None:
        return self.cell_schema.value_unit

    @property
    def axes(self) -> tuple:
        return tuple(self.cell_schema.data_axes)

@dataclass(frozen=True, slots=True)
class SignalDescription:
    """One signal as an outsider may know it: names and flags, no objects.

    Deliberately not a window onto the plane's state.  A view that held the
    real thing would read it at whatever moment it happened to paint, and show
    a mixture of two instants; a copy taken under the lock cannot.
    """

    name: str
    owner_id: str
    kind: str
    contract_id: str | None
    live: bool
    source_name: str | None
    revision: int
    shape: tuple[int, ...] | None
    failure: str | None = None

    @property
    def derived(self) -> bool:
        """Whether this signal is cut from another rather than acquired."""

        return self.source_name is not None


class IndexedHistoryLease:
    """One consumer's explicit demand for bounded source-index history.

    The output declaration says history *can* be built.  This lease says a
    current consumer actually needs it.  Release is idempotent; resizing never
    fabricates events that were not retained under an earlier demand.
    """

    __slots__ = (
        "_closed",
        "_acquisition_transition",
        "_plane",
        "_signal_name",
        "_token",
        "_window",
    )

    def __init__(
        self,
        plane: "SignalDataPlane",
        signal_name: str,
        token: object,
        window: int,
        transition: tuple[bool, bool, bool],
    ) -> None:
        self._plane = plane
        self._signal_name = signal_name
        self._token = token
        self._window = window
        self._acquisition_transition = transition
        self._closed = False

    @property
    def signal_name(self) -> str:
        return self._signal_name

    @property
    def window(self) -> int:
        return self._window

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def acquisition_transition(self) -> tuple[bool, bool, bool]:
        """The exact effect of creating this lease."""

        return self._acquisition_transition

    def resize(self, window: int) -> tuple[bool, bool, bool]:
        if self._closed:
            raise RuntimeError("indexed history lease is closed")
        selected, transition = self._plane._resize_indexed_history_lease(
            self._signal_name,
            self._token,
            window,
        )
        self._window = selected
        return transition

    def close(self) -> tuple[bool, bool, bool] | None:
        if self._closed:
            return None
        transition = self._plane._release_indexed_history_lease(
            self._signal_name,
            self._token,
        )
        self._closed = True
        return transition


@dataclass(frozen=True, eq=False)
class SignalPublication:
    """One exact immutable signal transaction.

    A publication is the causal unit.  Its signal mapping is one atomic sibling
    bundle and ``direct_parent_refs`` names only the exact events consumed to
    produce it.  The plane privately retains the corresponding immutable parent
    payloads while a child is live; that process-local retention is deliberately
    not part of the public lineage contract.  ``run_record`` is only the
    shallow, application-authored record frozen for this run; it is not Dataset
    schema, a second revision authority, or security provenance.
    """

    event_ref: EventRef
    signals: Mapping[str, SignalValue]
    _issuer: object = field(repr=False, compare=False)
    direct_parent_refs: tuple[EventRef, ...] = ()
    run_record: Mapping[str, object] = field(default_factory=dict)
    event_record: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_ref, EventRef):
            raise TypeError("signal publication event_ref must be EventRef")
        signals = dict(self.signals)
        if not signals:
            raise ValueError("signal publication requires an atomic sibling bundle")
        if any(
            not isinstance(name, str)
            or not isinstance(value, SignalValue)
            or value.name != name
            for name, value in signals.items()
        ):
            raise TypeError("signal publication mapping differs from its SignalValues")
        parents = tuple(self.direct_parent_refs)
        if any(not isinstance(value, EventRef) for value in parents):
            raise TypeError("signal publication parents must be EventRef values")
        if len(set(parents)) != len(parents):
            raise ValueError("signal publication parent refs must be unique")
        if not isinstance(self.run_record, Mapping):
            raise TypeError("signal publication run_record must be a mapping")
        if not isinstance(self.event_record, Mapping):
            raise TypeError("signal publication event_record must be a mapping")
        run_record = _freeze_run_record(self.run_record)
        event_record = _freeze_run_record(self.event_record)
        if any(
            not _run_records_equal(value.run_record, run_record)
            for value in signals.values()
        ):
            raise ValueError(
                "signal publication run_record differs from its sibling values"
            )
        if any(
            not _run_records_equal(value.event_record, event_record)
            for value in signals.values()
        ):
            raise ValueError(
                "signal publication event_record differs from its sibling values"
            )
        object.__setattr__(self, "signals", MappingProxyType(signals))
        object.__setattr__(self, "direct_parent_refs", parents)
        object.__setattr__(self, "run_record", run_record)
        object.__setattr__(self, "event_record", event_record)

    def value(self, name: str) -> SignalValue | None:
        return self.signals.get(str(name))


@dataclass(frozen=True)
class SignalFront:
    """Immutable front: coherent derived components, independent producers."""

    signals: Mapping[str, SignalValue]
    failures: Mapping[str, str]     # producer instance_id -> freeze failure
    publication_by_signal: Mapping[str, SignalPublication] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        signals = dict(self.signals)
        failures = dict(self.failures)
        publications = dict(self.publication_by_signal)
        if __debug__:
            assert all(isinstance(value, SignalValue) for value in signals.values()), (
                "SignalFront signals must contain SignalValue values"
            )
            assert all(isinstance(value, str) for value in failures.values()), (
                "SignalFront failures must contain strings"
            )
            assert set(publications) == set(signals), (
                "SignalFront publications must cover every signal"
            )
            for name, publication in publications.items():
                assert isinstance(publication, SignalPublication), (
                    "SignalFront contains another publication type"
                )
                assert publication.value(name) is signals[name], (
                    "SignalFront value is not owned by its exact publication"
                )
        object.__setattr__(self, "signals", MappingProxyType(signals))
        object.__setattr__(self, "failures", MappingProxyType(failures))
        object.__setattr__(
            self,
            "publication_by_signal",
            MappingProxyType(publications),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self.signals)

    def value(self, name: str) -> SignalValue | None:
        return self.signals.get(str(name))

    def publication(self, name: str) -> SignalPublication | None:
        return self.publication_by_signal.get(str(name))

def _declared_outputs(declarations) -> dict[str, DatasetOutputDeclaration]:
    """Return one producer's frozen Dataset output declarations."""

    values = tuple(declarations)
    if any(not isinstance(value, DatasetOutputDeclaration) for value in values):
        raise TypeError(
            "signal outputs must retain DatasetOutputDeclaration values"
        )
    result = {value.name: value for value in values}
    if len(result) != len(values):
        raise ValueError("signal output declarations contain duplicate names")
    return result


def _require_published_declaration(
    route_name: str,
    output: LiveDatasetOutput,
    declared: Mapping[str, DatasetOutputDeclaration],
) -> None:
    if output.name != route_name:
        raise ValueError("published output key differs from its owner declaration")
    expected = declared.get(route_name)
    if expected is None:
        raise ValueError(
            "published output is absent from the frozen producer vocabulary"
        )
    if output.declaration != expected:
        raise ValueError(
            "published output contract differs from the frozen owner declaration"
        )


def _shared_run_record(
    outputs: Mapping[
        str,
        LiveDatasetOutput | SignalValue,
    ],
) -> Mapping[str, object]:
    """Copy the one run record shared by an atomic sibling output bundle."""

    shared: dict[str, object] | None = None
    for output in outputs.values():
        if not isinstance(
            output,
            (LiveDatasetOutput, SignalValue),
        ):
            raise TypeError("run_record carrier has an unknown output type")
        record = {} if output.run_record is None else output.run_record
        if shared is None:
            shared = record
            continue
        if not _run_records_equal(record, shared):
            raise ValueError("sibling outputs must share one run_record")
    return {} if shared is None else shared


def _shared_event_record(
    outputs: Mapping[str, LiveDatasetOutput | SignalValue],
) -> Mapping[str, object]:
    """Copy the small event-varying record shared by one sibling bundle."""

    shared: Mapping[str, object] | None = None
    for output in outputs.values():
        record = {} if output.event_record is None else output.event_record
        if shared is None:
            shared = record
            continue
        if not _run_records_equal(record, shared):
            raise ValueError("sibling outputs must share one event_record")
    return {} if shared is None else shared


def _merge_event_records(
    records: Iterable[Mapping[str, object]],
) -> Mapping[str, object]:
    """Union compact device epoch ranges used by one materialized value."""

    merged: dict[str, object] = {}
    devices: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("event record must be a mapping")
        for key, value in record.items():
            if key == "device_settings":
                if not isinstance(value, Mapping):
                    raise TypeError("device_settings event record must be a mapping")
                for device_key, raw in value.items():
                    if not isinstance(raw, Mapping):
                        raise TypeError("device settings reference must be a mapping")
                    session_id = str(raw.get("device_session_id", "")).strip()
                    ranges = raw.get("epoch_ranges", ())
                    if not session_id or not isinstance(ranges, (tuple, list)):
                        raise ValueError("device settings reference is invalid")
                    target = devices.setdefault(
                        str(device_key),
                        {"device_session_id": session_id, "epoch_ranges": []},
                    )
                    if target["device_session_id"] != session_id:
                        raise RuntimeError(
                            "device session changed inside one materialized value"
                        )
                    target["epoch_ranges"].extend(ranges)
                continue
            previous_key = str(key)
            if previous_key in merged and not _run_records_equal(
                merged[previous_key], value
            ):
                raise ValueError(f"event record field {key!r} cannot be merged")
            merged[previous_key] = value
    if devices:
        compact: dict[str, object] = {}
        for device_key, raw in devices.items():
            ordered: list[tuple[int, int]] = []
            for bounds in raw["epoch_ranges"]:
                try:
                    start, stop = tuple(int(value) for value in bounds)
                except (TypeError, ValueError) as error:
                    raise ValueError("device epoch range is invalid") from error
                if start < 0 or stop < start:
                    raise ValueError("device epoch range is invalid")
                ordered.append((start, stop))
            ranges: list[list[int]] = []
            for start, stop in sorted(ordered):
                if ranges and start <= ranges[-1][1] + 1:
                    ranges[-1][1] = max(ranges[-1][1], stop)
                else:
                    ranges.append([start, stop])
            compact[device_key] = {
                "device_session_id": raw["device_session_id"],
                "epoch_ranges": ranges,
                "mixed": len(ranges) > 1 or bool(ranges and ranges[0][0] != ranges[0][1]),
            }
        merged["device_settings"] = compact
    return merged


def _require_signal_producer(node: object) -> SignalProducer:
    if not isinstance(node, SignalProducer):
        raise TypeError("signal producer must implement SignalProducer")
    return node


def _node_instance_id(node: object) -> str:
    """Return the stable producer identity required by the signal plane."""

    producer = _require_signal_producer(node)
    return canonical_text(producer.instance_id, "signal producer instance_id")


def _restamp_snapshot(
    snapshot: OwnedSnapshot,
    *,
    block_id: str,
    generation: StreamGenerationId,
    revision: int,
) -> OwnedSnapshot:
    """Give committed immutable bytes their Runtime-owned content identity."""

    block = DataBlock(
        BlockId(block_id),
        DatasetRevision(revision),
        snapshot.block.values,
        snapshot.block.validity,
        snapshot.block.schema,
    )
    return OwnedSnapshot(block.ref(generation), block)


_INDEXED_HISTORY_BYTES = 64 << 20
_INDEXED_HISTORY_COUNT = 100_000


def _indexed_capacity(snapshot: OwnedSnapshot) -> int:
    values = snapshot.block.values
    bytes_per_index = max(1, int(values.nbytes) + int(values.size))
    return max(
        1,
        min(_INDEXED_HISTORY_COUNT, _INDEXED_HISTORY_BYTES // bytes_per_index),
    )


def _indexed_schema(
    event_schema: DatasetSchema,
    indices: tuple[int, ...],
) -> DatasetSchema:
    point_count = event_schema.point_table.row_count
    if any(
        column.coordinate_id == PRIMARY_INDEX_AXIS_ID
        for column in event_schema.point_table.columns
    ):
        raise ValueError("processor event schema uses the reserved primary-index axis")
    primary = PointColumn(
        PRIMARY_INDEX_AXIS_ID,
        "source index",
        PRIMARY_INDEX,
        PointColumn.NUMERIC,
        tuple(index for index in indices for _row in range(point_count)),
    )
    columns = [primary]
    for column in event_schema.point_table.columns:
        columns.append(
            PointColumn(
                column.coordinate_id,
                column.name,
                column.role,
                column.value_kind,
                tuple(column.values) * len(indices),
                column.unit,
                column.coordinate_frame,
                None
                if column.coordinate_labels is None
                else tuple(column.coordinate_labels) * len(indices),
            )
        )
    topology = event_schema.grid_topology
    if topology is not None:
        topology = GridTopology(
            (PRIMARY_INDEX_AXIS_ID, *topology.dimension_ids),
            (indices, *topology.coordinate_domains),
            tuple(
                (index_position, *cell)
                for index_position in range(len(indices))
                for cell in topology.row_to_cell
            ),
        )
    return DatasetSchema(
        event_schema.repeat_axis,
        PointTable(point_count * len(indices), tuple(columns)),
        topology,
        event_schema.cell_schema,
    )


def _materialize_indexed_dataset(
    signal_name: str,
    generation: StreamGenerationId,
    revision: int,
    event_schema: DatasetSchema,
    events: tuple[tuple[int, OwnedSnapshot], ...],
    first_index: int,
    latest_index: int,
    capacity: int,
) -> OwnedSnapshot:
    start = max(first_index, latest_index - capacity + 1)
    indices = tuple(range(start, latest_index + 1))
    schema = _indexed_schema(event_schema, indices)
    values = np.zeros(schema.physical_shape, dtype=event_schema.cell_schema.dtype)
    validity = np.zeros(schema.physical_shape, dtype=np.bool_)
    point_count = event_schema.point_table.row_count
    trailing = (slice(None),) * len(event_schema.cell_schema.data_axes)
    for primary_index, snapshot in events:
        if not start <= primary_index <= latest_index:
            continue
        point_start = (primary_index - start) * point_count
        target = (
            slice(None),
            slice(point_start, point_start + point_count),
            *trailing,
        )
        values[target] = snapshot.block.values
        validity[target] = snapshot.expanded_validity()
    return owned_snapshot_from_arrays(
        schema,
        values,
        revision,
        validity=validity,
        block_id=BlockId(f"{signal_name}.indexed"),
        stream_generation=generation,
    )


_IndexedHistory: TypeAlias = tuple[
    dict[int, tuple[int, OwnedSnapshot, Mapping[str, object]]],
    int,
    int,
    tuple[int, OwnedSnapshot, Mapping[str, object]] | None,
]


def _validate_indexed_event(
    history: _IndexedHistory,
    event: OwnedSnapshot,
    primary_index: int,
) -> None:
    events, _first_index, _capacity, _materialized = history
    current_schema = next(iter(events.values()))[1].block.schema
    if event.block.schema != current_schema:
        raise ValueError(
            "indexed Processor event schema changed inside one generation"
        )
    if events and primary_index < next(reversed(events)):
        raise RuntimeError(
            "indexed Processor source primary index moved backwards"
        )


def _update_indexed_history(
    history: _IndexedHistory | None,
    value: SignalValue,
    sequence: int,
    demand: int,
    event_record: Mapping[str, object] | None = None,
) -> tuple[_IndexedHistory, bool]:
    primary_index = value.primary_index
    if primary_index is None:
        raise RuntimeError("indexed signal lost its source primary index")
    event = value.snapshot
    selected_record = value.event_record if event_record is None else event_record
    if history is None:
        return (
            (
                {primary_index: (sequence, event, selected_record)},
                primary_index,
                min(demand, _indexed_capacity(event)),
                None,
            ),
            True,
        )
    _validate_indexed_event(history, event, primary_index)
    events, first_index, _capacity, materialized = history
    current = events.get(primary_index)
    changed = current is None or current[0] != sequence
    if changed:
        events[primary_index] = (
            sequence,
            event,
            selected_record,
        )
    capacity = min(demand, _indexed_capacity(event))
    previous_first = first_index
    first_index = max(first_index, primary_index - capacity + 1)
    while events and next(iter(events)) < first_index:
        events.pop(next(iter(events)))
    changed = changed or first_index != previous_first
    return (
        (events, first_index, capacity, None if changed else materialized),
        changed,
    )


def _indexed_materialization_input(
    history: _IndexedHistory,
    *,
    signal_name: str,
    generation: StreamGenerationId,
    sequence: int,
    value: SignalValue,
) -> tuple[OwnedSnapshot, Mapping[str, object]] | tuple[tuple, Mapping[str, object]]:
    events, first_index, capacity, cached = history
    if cached is not None and cached[0] == sequence:
        return cached[1], cached[2]
    primary_index = value.primary_index
    if primary_index is None:
        raise RuntimeError("indexed signal lost its source primary index")
    if primary_index < first_index:
        raise ValueError("publication precedes retained indexed history")
    start = max(first_index, primary_index - capacity + 1)
    selected_events = []
    selected_records = []
    for index in range(start, primary_index + 1):
        if index == primary_index:
            selected_events.append((index, value.snapshot))
            held = events.get(index)
            selected_records.append(
                held[2]
                if held is not None and held[0] == sequence
                else value.event_record
            )
            continue
        held = events.get(index)
        if held is not None and held[0] <= sequence:
            selected_events.append((index, held[1]))
            selected_records.append(held[2])
    return (
        (
            signal_name,
            generation,
            sequence,
            next(iter(events.values()))[1].block.schema,
            tuple(selected_events),
            first_index,
            primary_index,
            capacity,
        ),
        _freeze_run_record(_merge_event_records(selected_records)),
    )


@dataclass(slots=True)
class _GenerationState:
    """The sole mutable state for one process-local signal generation."""

    owner_id: str
    generation: StreamGenerationId
    kind: str
    output_names: tuple[str, ...]
    bare_names: Mapping[str, str]
    declarations: Mapping[str, DatasetOutputDeclaration]
    node: object | None = None
    source_name: str | None = None
    #: Whether this route participates in same-shot front coherence.  False
    #: for presentation-paced followers (a panel's accepted-fit signals):
    #: their publications only advance AFTER their source presents, so
    #: letting them hold the source's front selection is a deadlock -- the
    #: source waits for the follower that waits for the source.
    coherent: bool = True
    source_owner_id: str | None = None
    source_generation: StreamGenerationId | None = None
    publication: SignalPublication | None = None
    last_parent_sequence: int = 0
    last_parent_trigger: tuple[str, int] | None = None
    published_names: tuple[str, ...] | None = None
    published_schemas: Mapping[str, DatasetSchema] | None = None
    next_sequence: int = 1
    failure: str | None = None
    terminal: bool = False
    retired: bool = False
    publication_stream: AcquisitionStream[SignalPublication] | None = None
    exact_outputs: frozenset[str] | None = None
    canonical_schemas: Mapping[str, DatasetSchema] = field(default_factory=dict)
    commit_chunks: dict[
        str,
        list[
            tuple[
                int,
                SignalValue,
                tuple[int, int],
                tuple[SignalPublication, ...],
            ]
        ],
    ] = field(default_factory=dict)
    occupied_cells: dict[str, np.ndarray] = field(default_factory=dict)
    materialized: dict[
        str,
        tuple[int, OwnedSnapshot, Mapping[str, object]],
    ] = field(default_factory=dict)
    indexed_history: dict[str, _IndexedHistory] = field(default_factory=dict)
    committed_run_record: Mapping[str, object] | None = None
    sealing: bool = False
    processor_cleanup_complete: bool = False
    publication_stream_cleaned: bool = False


@dataclass(slots=True)
class _ProcessorEntry:
    node: LatestProcessorControl
    source_name: str
    work_future: Future | None = None
    work_publication: SignalPublication | None = None
    pending_publication: SignalPublication | None = None
    last_publication: SignalPublication | None = None
    cancel_requested: bool = False
    paused: bool = False


class _LatestOnlyProcessorLane:
    """One private latest-only execution seam for reactive Processors.

    It keeps its own lock.  The plane deliberately calls this lane OUTSIDE the
    plane lock, so that a node callback never runs while the plane is held --
    which left the lane's own registry and entries mutated from whichever thread
    happened to call in: the display tick routes and drains, while a GUI thread
    attaches and cancels.  The plane's discipline was protecting the plane, and
    nothing was protecting this.

    The lock covers the registry and the entry fields, and is released before
    any node callback, for the same reason the plane releases its own.
    """

    def __init__(self, processor_cancelled: Callable[[object], None]) -> None:
        if not callable(processor_cancelled):
            raise TypeError("processor_cancelled must be callable")
        self._executor = ThreadPoolExecutor(
            thread_name_prefix="signal-latest-processor",
        )
        self._lock = threading.Lock()
        self._processors: dict[str, _ProcessorEntry] = {}
        self._processor_cancelled = processor_cancelled
        self._closed = False

    @staticmethod
    def _require_node(node: object) -> LatestProcessorControl:
        if not isinstance(node, LatestProcessorControl):
            raise TypeError("Processor must implement LatestProcessorControl")
        return node

    def attach_processor(
        self,
        node: LatestProcessorControl,
        source_name: str,
        initial_publication: SignalPublication,
        *,
        paused: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("continuous worker lane is closed")
        node = self._require_node(node)
        name = canonical_text(source_name, "processor source name")
        source = initial_publication.value(name)
        if source is None:
            raise ValueError("initial publication has no selected Processor source")
        key = _node_instance_id(node)
        with self._lock:
            if key in self._processors:
                raise RuntimeError("Processor node is already attached")
        node.validate_processor_source(source)
        entry = _ProcessorEntry(
            node=node,
            source_name=name,
            pending_publication=None if paused else initial_publication,
            last_publication=initial_publication,
            paused=paused,
        )
        with self._lock:
            if key in self._processors:
                raise RuntimeError("Processor node is already attached")
            self._processors[key] = entry
            failure = self._start_processor_locked(entry)
        if failure is not None:
            entry.node.accept_processor_failure(failure)

    def catch_up_processor(
        self,
        node: LatestProcessorControl,
        publication: SignalPublication,
    ) -> None:
        """Start one paused entry at the newest source publication it has seen."""

        node = self._require_node(node)
        key = _node_instance_id(node)
        with self._lock:
            entry = self._processors.get(key)
            if entry is None or entry.node is not node or entry.cancel_requested:
                raise RuntimeError("latest-only Processor is no longer attached")
            source = publication.value(entry.source_name)
            if source is None:
                raise ValueError("catch-up publication has no selected Processor source")
        node.validate_processor_source(source)
        with self._lock:
            entry = self._processors.get(key)
            if entry is None or entry.node is not node or entry.cancel_requested:
                raise RuntimeError("latest-only Processor is no longer attached")
            previous = entry.last_publication
            if (
                previous is None
                or publication.event_ref.sequence > previous.event_ref.sequence
            ):
                entry.last_publication = publication
                entry.pending_publication = publication
            entry.paused = False
            failure = self._start_processor_locked(entry)
        if failure is not None:
            entry.node.accept_processor_failure(failure)

    def route(self, publications: Mapping[str, SignalPublication]) -> None:
        if self._closed:
            return
        with self._lock:
            entries = tuple(self._processors.values())
        for entry in entries:
            if entry.cancel_requested:
                continue
            publication = publications.get(entry.source_name)
            if publication is None or publication is entry.last_publication:
                continue
            source = publication.value(entry.source_name)
            if source is None:
                self._fail_processor(
                    entry,
                    RuntimeError("Processor publication lost its selected signal"),
                )
                continue
            try:
                entry.node.validate_processor_source(source)
            except Exception as error:
                self._fail_processor(entry, error)
                continue
            with self._lock:
                if (
                    self._processors.get(_node_instance_id(entry.node)) is not entry
                    or entry.cancel_requested
                ):
                    continue
                entry.last_publication = publication
                entry.pending_publication = publication
                failure = self._start_processor_locked(entry)
            if failure is not None:
                entry.node.accept_processor_failure(failure)

    def drain_processors(self) -> None:
        with self._lock:
            entries = tuple(self._processors.values())
        for entry in entries:
            with self._lock:
                if self._processors.get(_node_instance_id(entry.node)) is not entry:
                    continue
                work = entry.work_future
                finished = work is not None and work.done()
                publication = entry.work_publication
                if finished:
                    entry.work_future = None
                    entry.work_publication = None
            if finished:
                if entry.cancel_requested:
                    self._cancelled_processor(entry)
                    continue
                try:
                    if publication is None:
                        raise RuntimeError("Processor lane lost its exact publication")
                    source = publication.value(entry.source_name)
                    if source is None:
                        raise RuntimeError("Processor publication lost its source")
                    result = work.result()
                    entry.node.accept_processor_result(
                        source,
                        publication,
                        result,
                    )
                except Exception as error:
                    self._fail_processor(entry, error)
                    continue
            failure = None
            with self._lock:
                cancelled = entry.cancel_requested and entry.work_future is None
                if not entry.cancel_requested:
                    failure = self._start_processor_locked(entry)
            if failure is not None:
                entry.node.accept_processor_failure(failure)
                continue
            if entry.cancel_requested:
                if cancelled:
                    self._cancelled_processor(entry)
                continue

    def cancel_processor(self, node: object) -> bool:
        with self._lock:
            entry = self._processors.get(_node_instance_id(node))
            if entry is None:
                return True
            entry.cancel_requested = True
            entry.pending_publication = None
            idle = entry.work_future is None or entry.work_future.done()
        if idle:
            self._cancelled_processor(entry)
        return idle

    def _start_processor_locked(self, entry: _ProcessorEntry) -> Exception | None:
        """Submit this entry's pending work.  Called with the lane lock held.

        Returns the failure instead of reporting it.  Reporting means calling
        the node, and a node callback must never run under this lock: the
        standard implementation reaches back into the plane, which takes the
        plane lock, while a thread holding the plane lock is waiting for this
        one.  The caller releases the lock and then reports.
        """

        if (
            entry.cancel_requested
            or entry.paused
            or entry.work_future is not None
            or entry.pending_publication is None
        ):
            return None
        publication, entry.pending_publication = entry.pending_publication, None
        source = publication.value(entry.source_name)
        try:
            if source is None:
                raise RuntimeError("Processor pending publication lost its source")
            future = self._executor.submit(
                entry.node.evaluate_processor,
                source,
                publication,
            )
        except Exception as error:
            self._processors.pop(_node_instance_id(entry.node), None)
            return error
        entry.work_publication = publication
        entry.work_future = future
        future.add_done_callback(
            lambda _future, current=entry.node: self._wake_processor(current)
        )
        return None

    def _fail_processor(self, entry: _ProcessorEntry, error: Exception) -> None:
        with self._lock:
            key = _node_instance_id(entry.node)
            if self._processors.get(key) is not entry:
                return
            self._processors.pop(key)
        entry.node.accept_processor_failure(error)

    def _cancelled_processor(self, entry: _ProcessorEntry) -> None:
        with self._lock:
            key = _node_instance_id(entry.node)
            if self._processors.get(key) is not entry:
                return
            self._processors.pop(key)
        try:
            entry.node.accept_processor_cancelled()
        finally:
            self._processor_cancelled(entry.node)

    def _wake_processor(self, node: object) -> None:
        if not self._closed:
            node.request_processor_owner_wake()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for entry in self._processors.values():
                entry.cancel_requested = True
                entry.pending_publication = None
            self._processors.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)


class SignalDataPlane:
    """One owner for signal generations, publications, and visible frontiers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation_ready = threading.Condition(self._lock)
        self._lane = _LatestOnlyProcessorLane(
            self._processor_cleanup_completed,
        )
        self._publication_issuer = object()
        self._publication_parents: WeakKeyDictionary[
            SignalPublication,
            tuple[SignalPublication, ...],
        ] = WeakKeyDictionary()
        self._states: dict[str, _GenerationState] = {}
        self._starting: set[str] = set()
        self._indexed_history_demands: dict[str, dict[object, int]] = {}
        self._front_signals: frozenset[str] = frozenset()
        self._membership_changed = False
        self._closed = False
        self._front = SignalFront({}, {})
        self._publication_callbacks: set[Callable[[], object]] = set()

    def subscribe_publications(
        self,
        callback: Callable[[], object],
    ) -> Callable[[], None]:
        """Wake an owner once after each atomic live publication."""

        if not callable(callback):
            raise TypeError("publication callback must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            self._publication_callbacks.add(callback)
        subscribed = True

        def unsubscribe() -> None:
            nonlocal subscribed
            if not subscribed:
                return
            subscribed = False
            with self._lock:
                self._publication_callbacks.discard(callback)

        return unsubscribe

    @staticmethod
    def _indexed_history_window(value: object) -> int:
        if type(value) is not int or value <= 0:
            raise TypeError("indexed history window must be a positive integer")
        return value

    def supports_indexed_history(self, signal_name: str) -> bool:
        """Whether this signal's owner permits consumer-demanded history."""

        name = canonical_text(signal_name, "signal name")
        with self._lock:
            state = self._state_for_signal_locked(name)
            if state is None or state.retired:
                return False
            declaration = state.declarations.get(name)
            return bool(
                declaration is not None and declaration.index_by_source
            )

    def indexed_history_active(self, signal_name: str) -> bool:
        """Whether any consumer currently owns this signal's indexed view."""

        name = canonical_text(signal_name, "signal name")
        with self._lock:
            return bool(self._indexed_history_demands.get(name))

    def acquire_indexed_history(
        self,
        signal_name: str,
        window: int,
    ) -> IndexedHistoryLease:
        """Start retaining this capable signal from its current event onward."""

        name = canonical_text(signal_name, "signal name")
        selected = self._indexed_history_window(window)
        token = object()
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._state_for_signal_locked(name)
            if state is None or state.retired:
                raise LookupError(f"signal {name!r} is not retained")
            declaration = state.declarations.get(name)
            if declaration is None or not declaration.index_by_source:
                raise ValueError(
                    f"signal {name!r} does not declare source-index history"
                )
            transition = self._set_indexed_history_demand_locked(
                name, token, selected
            )
        return IndexedHistoryLease(self, name, token, selected, transition)

    def _resize_indexed_history_lease(
        self,
        signal_name: str,
        token: object,
        window: int,
    ) -> tuple[int, tuple[bool, bool, bool]]:
        selected = self._indexed_history_window(window)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            demands = self._indexed_history_demands.get(signal_name)
            if demands is None or token not in demands:
                raise RuntimeError("indexed history lease is not active")
            transition = self._set_indexed_history_demand_locked(
                signal_name, token, selected
            )
        return selected, transition

    def _release_indexed_history_lease(
        self,
        signal_name: str,
        token: object,
    ) -> tuple[bool, bool, bool] | None:
        with self._lock:
            demands = self._indexed_history_demands.get(signal_name)
            if demands is None or token not in demands:
                return None
            return self._set_indexed_history_demand_locked(
                signal_name, token, None
            )

    def _indexed_history_demand_locked(self, signal_name: str) -> int | None:
        demands = self._indexed_history_demands.get(signal_name)
        return None if not demands else max(demands.values())

    def _set_indexed_history_demand_locked(
        self,
        signal_name: str,
        token: object,
        selected: int | None,
    ) -> tuple[bool, bool, bool]:
        demands = self._indexed_history_demands.setdefault(signal_name, {})
        previous = demands.get(token)
        old_demand = self._indexed_history_demand_locked(signal_name)
        if selected is None:
            demands.pop(token)
        else:
            demands[token] = selected
        if not demands:
            self._indexed_history_demands.pop(signal_name)
        new_demand = self._indexed_history_demand_locked(signal_name)
        state = self._state_for_signal_locked(signal_name)
        try:
            return self._update_indexed_history_locked(
                state, signal_name, old_demand, new_demand
            )
        except BaseException:
            restored = self._indexed_history_demands.setdefault(signal_name, {})
            if previous is None:
                restored.pop(token, None)
            else:
                restored[token] = previous
            if not restored:
                self._indexed_history_demands.pop(signal_name)
            self._update_indexed_history_locked(
                state, signal_name, new_demand, old_demand
            )
            raise

    def _update_indexed_history_locked(
        self,
        state: _GenerationState | None,
        signal_name: str,
        old_demand: int | None,
        new_demand: int | None,
    ) -> tuple[bool, bool, bool]:
        """Apply one effective lease transition without inventing old events."""

        old_active = bool(state and signal_name in state.indexed_history)
        if old_demand == new_demand:
            return old_active, False, False
        if state is None:
            return False, False, False
        declaration = state.declarations.get(signal_name)
        if (
            new_demand is None
            or declaration is None
            or not declaration.index_by_source
            or state.publication is None
        ):
            data_changed = state.indexed_history.pop(signal_name, None) is not None
        else:
            value = state.publication.value(signal_name)
            if value is None or not isinstance(value.coverage, MonitorCoverage):
                data_changed = (
                    state.indexed_history.pop(signal_name, None) is not None
                )
            else:
                history = state.indexed_history.get(signal_name)
                history, data_changed = _update_indexed_history(
                    history,
                    value,
                    state.publication.event_ref.sequence,
                    new_demand,
                )
                state.indexed_history[signal_name] = history
        new_active = signal_name in state.indexed_history
        return new_active, old_active != new_active, data_changed

    def set_front_signals(self, signal_names) -> None:
        """Set the connected continuous signal set whose front must be coherent."""

        names = frozenset(
            canonical_text(name, "connected signal name")
            for name in signal_names
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            if names != self._front_signals:
                self._front_signals = names
                self._membership_changed = True

    @staticmethod
    def _mint_generation_locked() -> StreamGenerationId:
        return StreamGenerationId(uuid.uuid4().hex)

    def _state_for_signal_locked(
        self,
        signal_name: str,
    ) -> _GenerationState | None:
        selected = None
        for state in self._states.values():
            if state.retired or signal_name not in state.output_names:
                continue
            if selected is not None:
                raise RuntimeError(
                    f"signal {signal_name!r} has more than one generation owner"
                )
            selected = state
        return selected

    def _assert_names_available_locked(
        self,
        owner_id: str,
        output_names: tuple[str, ...],
    ) -> None:
        for name in output_names:
            conflict = next(
                (
                    candidate
                    for candidate in self._states.values()
                    if not candidate.retired
                    and name in candidate.output_names
                    and candidate.owner_id != owner_id
                ),
                None,
            )
            if conflict is not None:
                raise RuntimeError(
                    f"signal {name!r} is already owned by {conflict.owner_id!r}"
                )

    @staticmethod
    def _generation_ref(
        state: _GenerationState,
    ) -> tuple[str, StreamGenerationId]:
        return state.owner_id, state.generation

    def _drop_state_locked(self, state: _GenerationState) -> None:
        if self._states.get(state.owner_id) is not state:
            return
        if (
            state.retired
            and state.kind == "processor"
            and state.node is not None
            and not state.processor_cleanup_complete
        ):
            return
        self._states.pop(state.owner_id)
        state.retired = True
        state.publication = None
        state.failure = None

    def _processor_cleanup_completed(self, node: object) -> None:
        """Release a retired route only after its lane entry is truly gone."""

        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if state is None or state.node is not node:
                return
            state.processor_cleanup_complete = True
            if state.retired and owner_id not in self._starting:
                self._drop_state_locked(state)
                self._membership_changed = True
            self._generation_ready.notify_all()

    def _wait_for_start_locked(self, owner_id: str) -> None:
        while owner_id in self._starting and not self._closed:
            self._generation_ready.wait()
        if self._closed:
            raise RuntimeError("signal data plane is closed")

    def _install_state_locked(
        self,
        *,
        owner_id: str,
        kind: str,
        output_names: tuple[str, ...],
        bare_names: Mapping[str, str],
        node: object | None = None,
        source_name: str | None = None,
        coherent: bool = True,
    ) -> _GenerationState:
        identity = canonical_text(owner_id, "signal generation owner_id")
        kind = canonical_text(kind, "signal generation kind")
        names = tuple(
            canonical_text(name, "signal generation output name")
            for name in output_names
        )
        if not names or len(set(names)) != len(names):
            raise ValueError("signal generation outputs must be non-empty and unique")
        if identity in self._states:
            raise RuntimeError("signal generation owner is already active")
        bare = dict(bare_names)
        if set(bare) != set(names):
            raise ValueError("signal generation bare names differ from its outputs")
        if any(
            not isinstance(value, str) or not value or value.strip() != value
            for value in bare.values()
        ):
            raise ValueError("signal generation bare names must be canonical text")
        if node is None:
            declarations: Mapping[str, DatasetOutputDeclaration] = MappingProxyType({})
        else:
            declared = _declared_outputs(
                _require_signal_producer(node).dataset_output_declarations
            )
            if set(declared) != set(bare.values()):
                raise ValueError(
                    "signal generation declarations differ from its output names"
                )
            declarations = MappingProxyType(
                {
                    qualified: declared[bare_name]
                    for qualified, bare_name in bare.items()
                }
            )
        source_state = None
        if source_name is not None:
            source_name = canonical_text(source_name, "signal route source name")
            source_state = self._state_for_signal_locked(source_name)
            if source_state is None:
                raise LookupError(
                    f"signal route source {source_name!r} is not active"
                )
        self._assert_names_available_locked(identity, names)
        state = _GenerationState(
            owner_id=identity,
            generation=self._mint_generation_locked(),
            kind=kind,
            output_names=names,
            bare_names=MappingProxyType(bare),
            declarations=declarations,
            node=node,
            source_name=source_name,
            coherent=bool(coherent),
            source_owner_id=(
                None if source_state is None else source_state.owner_id
            ),
            source_generation=(
                None if source_state is None else source_state.generation
            ),
        )
        self._states[identity] = state
        self._membership_changed = True
        return state

    @staticmethod
    def _ensure_publication_stream_locked(
        state: _GenerationState,
    ) -> AcquisitionStream[SignalPublication]:
        stream = state.publication_stream
        if stream is None:
            stream = AcquisitionStream.create(
                next_sequence=state.next_sequence,
            )
            state.publication_stream = stream
        return stream

    @staticmethod
    def _node_route_names(node: object) -> tuple[
        tuple[str, ...],
        Mapping[str, str],
    ]:
        producer = _require_signal_producer(node)
        declarations = _declared_outputs(producer.dataset_output_declarations)
        qualified = tuple(str(producer.signal_key(name)) for name in declarations)
        return (
            qualified,
            MappingProxyType(
                {
                    selected: bare
                    for selected, bare in zip(
                        qualified,
                        declarations,
                        strict=True,
                    )
                }
            ),
        )

    def reserve(self, node: object) -> StreamGenerationId:
        """Reserve one producer generation before its worker can publish.

        The composition owner performs this before submitting a run.  Every
        event then enters through ``commit_live``; publication can therefore
        never recreate a generation after retirement.
        """

        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            self._wait_for_start_locked(owner_id)
            existing = self._states.get(owner_id)
            if existing is not None:
                if (
                    not existing.retired
                    and not existing.terminal
                    and existing.kind == "producer"
                    and existing.node is node
                    and existing.output_names == output_names
                    and dict(existing.bare_names) == dict(bare_names)
                ):
                    return existing.generation
                raise RuntimeError("producer generation is already active")
            state = self._install_state_locked(
                owner_id=owner_id,
                kind="producer",
                output_names=output_names,
                bare_names=bare_names,
                node=node,
            )
            return state.generation

    def begin_generation(self, node: object) -> StreamGenerationId:
        """Start a producer generation, superseding a FINISHED predecessor.

        A generation belongs to one run, but the node that performs the run is a
        reusable object, so something has to decide when the previous retained
        generation is replaced.  It ends when the next run begins -- and that
        is this method.

        Use this to START a run.  ``reserve`` is the lower-level operation that
        refuses to touch an existing generation at all; a caller that reserves
        directly can never run the same node twice, because the first run leaves
        a terminal generation behind and the second reservation is rejected.

        A generation that is still LIVE is not superseded: two concurrent runs of
        one producer is a real error and still raises.
        """

        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        retired: tuple[_GenerationState, ...] = ()
        with self._lock:
            self._wait_for_start_locked(owner_id)
            existing = self._states.get(owner_id)
            if existing is not None and not (
                existing.retired or existing.terminal
            ):
                if (
                    existing.kind == "producer"
                    and existing.node is node
                    and existing.publication is None
                    and existing.output_names == output_names
                    and dict(existing.bare_names) == dict(bare_names)
                ):
                    return existing.generation
                raise RuntimeError("producer generation is already active")
            if existing is None:
                return self._install_state_locked(
                    owner_id=owner_id,
                    kind="producer",
                    output_names=output_names,
                    bare_names=bare_names,
                    node=node,
                ).generation
            retired = self._reserve_retirement_closure_locked(
                owner_id,
            )
            retiring_owners = tuple(state.owner_id for state in retired)
        try:
            errors = self._cleanup_retired_states(retired)
        except BaseException as error:
            errors = (error,)
        with self._lock:
            try:
                if errors:
                    self._membership_changed = True
                elif self._closed:
                    errors = (RuntimeError("signal data plane is closed"),)
                else:
                    for candidate in retired:
                        self._drop_state_locked(candidate)
                    state = self._install_state_locked(
                        owner_id=owner_id,
                        kind="producer",
                        output_names=output_names,
                        bare_names=bare_names,
                        node=node,
                    )
            finally:
                self._starting.difference_update(retiring_owners)
                self._generation_ready.notify_all()
        if errors:
            raise BaseExceptionGroup(
                "previous signal generation cleanup failed",
                list(errors),
            )
        return state.generation

    @staticmethod
    def _declarations_by_bare(
        state: _GenerationState,
    ) -> dict[str, DatasetOutputDeclaration]:
        return {
            state.bare_names[qualified]: declaration
            for qualified, declaration in state.declarations.items()
        }

    @staticmethod
    def _materialization_input_locked(
        state: _GenerationState,
        signal_name: str,
        sequence: int,
    ) -> tuple[
        DatasetSchema,
        StreamGenerationId,
        tuple[tuple[OwnedSnapshot, tuple[int, int]], ...],
    ]:
        schema = state.canonical_schemas.get(signal_name)
        if not isinstance(schema, DatasetSchema):
            raise RuntimeError("signal has no canonical Dataset schema")
        chunks = tuple(
            (value.snapshot, origin)
            for commit_sequence, value, origin, _parents in state.commit_chunks.get(
                signal_name, ()
            )
            if commit_sequence <= sequence
        )
        return schema, state.generation, chunks

    @staticmethod
    def _materialize_dataset(
        signal_name: str,
        sequence: int,
        schema: DatasetSchema,
        generation: StreamGenerationId,
        chunks: tuple[tuple[OwnedSnapshot, tuple[int, int]], ...],
    ) -> OwnedSnapshot:
        values = np.zeros(schema.physical_shape, dtype=schema.cell_schema.dtype)
        validity = np.zeros(schema.physical_shape, dtype=np.bool_)
        for chunk, origin in chunks:
            repeat_origin, point_origin = origin
            chunk_schema = chunk.block.schema
            repeat_stop = repeat_origin + chunk_schema.repeat_axis.size
            point_stop = point_origin + chunk_schema.point_table.row_count
            target = (
                slice(repeat_origin, repeat_stop),
                slice(point_origin, point_stop),
                *(slice(None) for _axis in schema.cell_schema.data_axes),
            )
            values[target] = chunk.block.values
            validity[target] = chunk.expanded_validity()
        return owned_snapshot_from_arrays(
            schema,
            values,
            sequence,
            validity=validity,
            block_id=BlockId(f"{signal_name}.run"),
            stream_generation=generation,
        )

    def commit_live(
        self,
        node: object,
        outputs: Mapping[str, LiveDatasetOutput],
        *,
        worker_source: tuple[str, SignalPublication] | None = None,
    ) -> Mapping[str, SignalValue]:
        return self._commit_outputs(
            node,
            outputs,
            worker_source=worker_source,
        )

    def commit_processor(
        self,
        node: object,
        outputs: Mapping[str, LiveDatasetOutput],
        *,
        source_publication: SignalPublication,
        trigger: tuple[str, int] | None = None,
        retain: bool = False,
    ) -> Mapping[str, SignalValue]:
        """Commit one derived bundle with its exact causal parent."""

        if not isinstance(source_publication, SignalPublication):
            raise TypeError("Processor commit requires its exact parent")
        if type(retain) is not bool:
            raise TypeError("retain must be bool")
        if trigger is not None:
            if (
                not isinstance(trigger, tuple)
                or len(trigger) != 2
                or type(trigger[0]) is not str
                or not trigger[0].strip()
                or type(trigger[1]) is not int
                or trigger[1] < 0
            ):
                raise TypeError(
                    "processor trigger must be (non-empty text, non-negative int)"
                )
            trigger = (trigger[0].strip(), trigger[1])
        selected = dict(outputs)
        if retain:
            for name, output in tuple(selected.items()):
                if not isinstance(output, LiveDatasetOutput):
                    raise TypeError("processor outputs must be LiveDatasetOutput")
                if isinstance(output.coverage, MonitorCoverage):
                    schema = output.snapshot.block.schema
                    total = (
                        schema.repeat_axis.size
                        * schema.point_table.row_count
                    )
                    selected[name] = LiveDatasetOutput(
                        output.declaration,
                        output.snapshot,
                        DatasetCoverage(total, total),
                        output.run_record,
                        schema,
                        (0, 0),
                        output.event_record,
                    )
        return self._commit_outputs(
            node,
            selected,
            source_publication=source_publication,
            trigger=trigger,
        )

    def _commit_outputs(
        self,
        node: object,
        outputs: Mapping[str, LiveDatasetOutput],
        *,
        source_publication: SignalPublication | None = None,
        worker_source: tuple[str, SignalPublication] | None = None,
        trigger: tuple[str, int] | None = None,
    ) -> Mapping[str, SignalValue]:
        """Commit one immutable sibling bundle into Runtime-owned run state."""

        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError("live commit outputs must be a non-empty mapping")
        owner_id = _node_instance_id(node)
        if source_publication is not None and worker_source is not None:
            raise ValueError("one commit cannot have processor and worker sources")
        kind = "producer" if source_publication is None else "processor"
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
            ):
                raise SourceGenerationEnded(
                    "the processor's generation was retired before its commit"
                )
            if (
                state.terminal
                or state.sealing
                or state.node is not node
                or state.kind != kind
            ):
                raise RuntimeError("live commit requires the reserved generation")
            if state.publication is not None and state.exact_outputs is None:
                raise RuntimeError("one generation cannot mix publication paths")
            route_source = None
            if source_publication is not None:
                route_source = self._require_route_parent_locked(
                    state, source_publication
                )
                source_sequence = source_publication.event_ref.sequence
                if source_sequence < state.last_parent_sequence or (
                    source_sequence == state.last_parent_sequence
                    and (trigger is None or trigger == state.last_parent_trigger)
                ):
                    raise RuntimeError(
                        "Processor result belongs to an obsolete parent"
                    )
            worker_parent = None
            worker_signal = None
            if worker_source is not None:
                worker_signal, worker_parent = worker_source
                worker_signal = canonical_text(worker_signal, "worker source signal")
                self._require_issued_publication_locked(worker_parent)
                if worker_parent.value(worker_signal) is None:
                    raise ValueError(
                        "worker source publication does not contain its signal"
                    )

            declared = self._declarations_by_bare(state)
            if set(outputs) != set(declared):
                raise ValueError(
                    "live commit must cover the complete frozen output vocabulary"
                )
            if any(
                not isinstance(output, LiveDatasetOutput)
                for output in outputs.values()
            ):
                raise TypeError("live commit values must be LiveDatasetOutput")
            exact_bare = frozenset(
                bare
                for bare, output in outputs.items()
                if isinstance(output.coverage, DatasetCoverage)
            )
            exact_qualified = frozenset(
                qualified
                for qualified in state.output_names
                if state.bare_names[qualified] in exact_bare
            )
            if (
                state.exact_outputs is not None
                and state.exact_outputs != exact_qualified
            ):
                raise ValueError("live extent kinds changed inside one generation")
            run_record = _freeze_run_record(_shared_run_record(outputs))
            event_record = _freeze_run_record(_shared_event_record(outputs))
            if (
                state.committed_run_record is not None
                and not _run_records_equal(state.committed_run_record, run_record)
            ):
                raise ValueError("run_record changed inside one generation")

            canonical_schemas = dict(state.canonical_schemas)
            occupied_cells = dict(state.occupied_cells)
            occupied_updates: list[
                tuple[str, np.ndarray, tuple[slice, slice]]
            ] = []
            origins: dict[str, tuple[int, int]] = {}
            values: dict[str, SignalValue] = {}
            sequence = state.next_sequence
            primary_index = (
                sequence
                if route_source is None
                else route_source.primary_index
                if route_source.primary_index is not None
                else source_publication.event_ref.sequence
            )
            indexed_updates: dict[str, tuple[OwnedSnapshot, int]] = {}

            for qualified in state.output_names:
                bare = state.bare_names[qualified]
                output = outputs[bare]
                if not isinstance(output, LiveDatasetOutput):
                    raise TypeError("live commit values must be LiveDatasetOutput")
                _require_published_declaration(bare, output, declared)
                event = _restamp_snapshot(
                    output.snapshot,
                    block_id=f"{qualified}.event",
                    generation=state.generation,
                    revision=sequence,
                )
                history_demand = self._indexed_history_demand_locked(qualified)
                if (
                    source_publication is not None
                    and output.declaration.index_by_source
                    and isinstance(output.coverage, MonitorCoverage)
                    and history_demand is not None
                ):
                    history = state.indexed_history.get(qualified)
                    if history is not None:
                        _validate_indexed_event(history, event, primary_index)
                    indexed_updates[qualified] = (event, history_demand)
                if qualified in exact_qualified:
                    schema = output.canonical_schema
                    origin = output.cell_origin
                    if not isinstance(schema, DatasetSchema) or origin is None:
                        raise ValueError(
                            "finite live commit requires canonical placement"
                        )
                    previous_schema = canonical_schemas.get(qualified)
                    if previous_schema is not None and previous_schema != schema:
                        raise ValueError(
                            "canonical Dataset schema changed inside one generation"
                        )
                    canonical_schemas[qualified] = schema
                    origins[qualified] = origin
                    mask = occupied_cells.get(qualified)
                    if mask is None:
                        mask = np.zeros(
                            (
                                schema.repeat_axis.size,
                                schema.point_table.row_count,
                            ),
                            dtype=np.bool_,
                        )
                    repeat_origin, point_origin = origin
                    event_schema = event.block.schema
                    target = (
                        slice(
                            repeat_origin,
                            repeat_origin + event_schema.repeat_axis.size,
                        ),
                        slice(
                            point_origin,
                            point_origin + event_schema.point_table.row_count,
                        ),
                    )
                    if bool(np.any(mask[target])):
                        raise ValueError("live commit overlaps already written cells")
                    previous = (
                        None
                        if state.publication is None
                        else state.publication.value(qualified)
                    )
                    written = (
                        0
                        if previous is None
                        else previous.coverage.written_cells
                    ) + (
                        + event_schema.repeat_axis.size
                        * event_schema.point_table.row_count
                    )
                    if output.coverage.written_cells != written:
                        raise ValueError(
                            "finite coverage does not equal committed cell extent"
                    )
                    occupied_updates.append((qualified, mask, target))
                values[qualified] = SignalValue(
                    name=qualified,
                    snapshot=event,
                    coverage=output.coverage,
                    run_record=run_record,
                    canonical_schema=output.canonical_schema,
                    cell_origin=output.cell_origin,
                    primary_index=primary_index,
                    event_record=event_record,
                )

            parent = source_publication if worker_parent is None else worker_parent
            parents = () if parent is None else (parent,)
            publication = self._publish_locked(
                state,
                values,
                parents=parents,
                notify=False,
            )
            replay_parents = (
                ()
                if parent is None
                else (
                    self._slim_publication_locked(
                        parent,
                        worker_signal or state.source_name,
                        {},
                    ),
                )
            )
            for qualified, mask, target in occupied_updates:
                mask[target] = True
                occupied_cells[qualified] = mask
            state.exact_outputs = exact_qualified
            state.canonical_schemas = MappingProxyType(canonical_schemas)
            state.occupied_cells = occupied_cells
            state.committed_run_record = run_record
            for qualified, (event, history_demand) in indexed_updates.items():
                history = state.indexed_history.get(qualified)
                state.indexed_history[qualified] = _update_indexed_history(
                    history,
                    publication.signals[qualified],
                    sequence,
                    history_demand,
                    None if history is None else event_record,
                )[0]
            if source_publication is not None:
                state.last_parent_sequence = source_publication.event_ref.sequence
                state.last_parent_trigger = trigger
            for qualified in exact_qualified:
                state.commit_chunks.setdefault(qualified, []).append(
                    (
                        sequence,
                        publication.signals[qualified],
                        origins[qualified],
                        replay_parents,
                    )
                )
            producer = state.publication_stream
            if producer is not None:
                producer.emit(
                    publication,
                    sequence=publication.event_ref.sequence,
                )
            result = publication.signals
            callbacks = tuple(self._publication_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # A presentation wake cannot roll back already-committed data.
                continue
        return result

    def current_dataset_view(
        self,
        signal_name: str,
        publication: SignalPublication | None = None,
    ) -> tuple[OwnedSnapshot, Mapping[str, object]]:
        """Materialize one Dataset and the exact event record it contains."""
        name = canonical_text(signal_name, "signal name")
        indexed_input = None
        finite_input = None
        materialized_record: Mapping[str, object] | None = None
        with self._lock:
            state = self._state_for_signal_locked(name)
            if state is None or state.retired:
                raise LookupError(f"signal {name!r} is not retained")
            selected = state.publication if publication is None else publication
            if selected is None:
                raise LookupError(f"signal {name!r} has no committed data")
            self._require_issued_publication_locked(selected)
            if (
                selected.event_ref.stream_id.value != state.owner_id
                or selected.event_ref.generation != state.generation
            ):
                raise ValueError(
                    "publication belongs to another signal generation"
                )
            value = selected.value(name)
            if value is None:
                raise ValueError("publication does not contain the selected signal")
            sequence = selected.event_ref.sequence
            history = state.indexed_history.get(name)
            if history is not None:
                indexed_result = _indexed_materialization_input(
                    history,
                    signal_name=name,
                    generation=state.generation,
                    sequence=sequence,
                    value=value,
                )
                if isinstance(indexed_result[0], OwnedSnapshot):
                    return indexed_result
                indexed_input, materialized_record = indexed_result
            elif state.exact_outputs is None or name not in state.exact_outputs:
                return value.snapshot, value.event_record
            else:
                if not any(
                    commit_sequence == sequence
                    for commit_sequence, _value, _origin, _parents in state.commit_chunks[
                        name
                    ]
                ):
                    raise ValueError("publication is not a canonical commit of this run")
                cached = state.materialized.get(name)
                if cached is not None and cached[0] == sequence:
                    return cached[1], cached[2]
                finite_input = self._materialization_input_locked(
                    state,
                    name,
                    sequence,
                )
                materialized_record = _freeze_run_record(
                    _merge_event_records(
                        value.event_record
                        for commit_sequence, value, _origin, _parents in (
                            state.commit_chunks[name]
                        )
                        if commit_sequence <= sequence
                    )
                )
        snapshot = (
            _materialize_indexed_dataset(*indexed_input)
            if indexed_input is not None
            else self._materialize_dataset(name, sequence, *finite_input)
        )
        with self._lock:
            if self._states.get(state.owner_id) is state and not state.retired:
                if indexed_input is not None:
                    history = state.indexed_history.get(name)
                    if history is not None:
                        events, first_index, capacity, current = history
                        if current is None or current[0] <= sequence:
                            assert materialized_record is not None
                            current = (
                                sequence,
                                snapshot,
                                materialized_record,
                            )
                        state.indexed_history[name] = (
                            events,
                            first_index,
                            capacity,
                            current,
                        )
                else:
                    cached = state.materialized.get(name)
                    # Keep one immutable prefix per signal, never a list of all
                    # prefixes.  A later materialization must not be displaced by
                    # an older request that happened to finish afterwards.
                    if cached is None or cached[0] <= sequence:
                        state.materialized[name] = (
                            sequence,
                            snapshot,
                            materialized_record,
                        )
        assert materialized_record is not None
        return snapshot, materialized_record

    def current_dataset(
        self,
        signal_name: str,
        publication: SignalPublication | None = None,
    ) -> OwnedSnapshot:
        """Materialize one exact finite prefix or active indexed Dataset."""

        return self.current_dataset_view(signal_name, publication)[0]

    def seal_committed(self, node: object, *, cut_short: bool = False) -> bool:
        """Seal one commit generation without publishing a duplicate full event."""

        if type(cut_short) is not bool:
            raise TypeError("cut_short must be bool")
        owner_id = _node_instance_id(node)
        producer = None
        state = None
        sequence = 0
        retain_latest_monitor = False
        materialized: dict[
            str,
            tuple[int, OwnedSnapshot, Mapping[str, object]],
        ] = {}
        pending: dict[
            str,
            tuple[
                tuple[
                    DatasetSchema,
                    StreamGenerationId,
                    tuple[tuple[OwnedSnapshot, tuple[int, int]], ...],
                ],
                Mapping[str, object],
            ],
        ] = {}
        with self._lock:
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.terminal
                or state.sealing
                or state.node is not node
                or state.exact_outputs is None
                or state.publication is None
            ):
                raise RuntimeError("committed generation is not active")
            exact_outputs = state.exact_outputs
            if exact_outputs:
                if not cut_short and not all(
                    isinstance(state.publication.signals[name].coverage, DatasetCoverage)
                    and state.publication.signals[name].coverage.complete
                    for name in exact_outputs
                ):
                    raise RuntimeError(
                        "exact committed terminal Dataset coverage is incomplete"
                    )
                state.sealing = True
                sequence = state.publication.event_ref.sequence
                for name in exact_outputs:
                    cached = state.materialized.get(name)
                    if cached is not None and cached[0] == sequence:
                        materialized[name] = cached
                    else:
                        pending[name] = (
                            self._materialization_input_locked(
                                state,
                                name,
                                sequence,
                            ),
                            _freeze_run_record(
                                _merge_event_records(
                                    value.event_record
                                    for commit_sequence, value, _origin, _parents in (
                                        state.commit_chunks[name]
                                    )
                                    if commit_sequence <= sequence
                                )
                            ),
                        )
            else:
                producer = state.publication_stream
                retain_latest_monitor = bool(state.publication.signals) and all(
                    isinstance(value.coverage, MonitorCoverage)
                    and value.coverage.retain_at_terminal
                    for value in state.publication.signals.values()
                )
                if retain_latest_monitor:
                    state.terminal = True
                    self._membership_changed = True
        if not exact_outputs:
            if producer is not None:
                producer.finish()
            if retain_latest_monitor:
                return True
            self._withdraw_owner(owner_id)
            return False
        try:
            for name, (inputs, event_record) in pending.items():
                materialized[name] = (
                    sequence,
                    self._materialize_dataset(name, sequence, *inputs),
                    event_record,
                )
        except BaseException:
            with self._lock:
                if self._states.get(owner_id) is state:
                    state.sealing = False
            raise
        with self._lock:
            if (
                self._states.get(owner_id) is not state
                or state.retired
                or not state.sealing
                or state.publication is None
                or state.publication.event_ref.sequence != sequence
            ):
                raise RuntimeError("committed generation changed while sealing")
            state.materialized = materialized
            state.sealing = False
            state.terminal = True
            self._membership_changed = True
            producer = state.publication_stream
        if producer is not None:
            producer.finish()
        return True

    def seal_processor(self, node: object) -> bool:
        """Seal derived canonical state after its exact source reached EOS."""

        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if state is None:
                # The retirement cascade DROPS the derived closure's states;
                # a running follower reaching its seal after that is the
                # source's lifecycle, not a misuse.
                raise SourceGenerationEnded(
                    "the processor's generation was retired before it sealed"
                )
            if state.kind != "processor" or state.node is not node:
                raise RuntimeError("Processor generation is not committed")
            if state.retired:
                # Retired mid-run means the bench moved on underneath this
                # follower -- a restart superseded its whole derived
                # closure.  That is the source's lifecycle, not this
                # processor's failure.
                raise SourceGenerationEnded(
                    "the processor's generation was retired before it sealed"
                )
            if (
                state.source_owner_id is None
                or state.exact_outputs is None
                or state.publication is None
            ):
                raise RuntimeError("Processor generation is not committed")
            source = self._states.get(state.source_owner_id)
            if source is None:
                raise RuntimeError("Processor source has not reached its exact terminal")
            if source.retired or source.generation != state.source_generation:
                raise SourceGenerationEnded(
                    "the processor's source generation ended before the seal"
                )
            if (
                not source.terminal
                or source.publication is None
                or source.publication.event_ref.sequence
                != state.last_parent_sequence
            ):
                raise RuntimeError("Processor source has not reached its exact terminal")
            cut_short = any(
                isinstance(state.publication.signals[name].coverage, DatasetCoverage)
                and not state.publication.signals[name].coverage.complete
                for name in state.exact_outputs
            )
        return self.seal_committed(node, cut_short=cut_short)

    def describe_signals(self) -> tuple["SignalDescription", ...]:
        """Every signal this plane currently carries, as plain values.

        The plane holds the only complete answer to "who is producing what",
        and until now the only way to see it was to hold a signal's name
        already.  Anything offering a choice -- a picker, a legend, a status
        line -- needs the list first, and must not be handed the plane's own
        state to rummage through: these are copies, ordered, and safe to keep.
        """

        with self._lock:
            states = tuple(self._states.values())
            descriptions = []
            for state in states:
                if state.retired:
                    continue
                for name in state.output_names:
                    declaration = state.declarations.get(name)
                    value = (
                        None
                        if state.publication is None
                        else state.publication.signals.get(name)
                    )
                    descriptions.append(
                        SignalDescription(
                            name=name,
                            owner_id=state.owner_id,
                            kind=state.kind,
                            contract_id=(
                                None
                                if declaration is None
                                else declaration.contract_id
                            ),
                            live=not state.terminal,
                            source_name=state.source_name,
                            revision=(
                                0
                                if state.publication is None
                                else state.publication.event_ref.sequence
                            ),
                            shape=(
                                None
                                if value is None
                                else (
                                    value.canonical_schema.physical_shape
                                    if value.canonical_schema is not None
                                    else value.shape
                                )
                            ),
                            failure=state.failure,
                        )
                    )
        return tuple(sorted(descriptions, key=lambda item: item.name))

    def is_generation_live(self, signal_name: str) -> bool:
        """Whether more publications can still arrive for one signal.

        A live generation can feed a processor; a finished one can only be
        derived from once, and the two produce different things -- a value that
        keeps up with the run, or a single answer about a run that is over.
        Anything that must choose needs the answer before it attaches, rather
        than as an exception raised afterwards.
        """

        name = canonical_text(signal_name, "signal name")
        with self._lock:
            state = self._state_for_signal_locked(name)
            return state is not None and not state.retired and not state.terminal

    def retains(self, signal_name: str) -> bool:
        """Whether this signal's data is still HERE to be derived from.

        ``is_generation_live`` answers two of the three states a signal can
        be in -- it says False both for a run that finished with its data
        retained and for one whose data is gone.  A consumer choosing how
        to derive reads that single False as "finished", takes the terminal
        path, and only discovers the difference when materializing raises
        ``LookupError``: a box drawn on a panel whose run has been retired
        took the console down that way.  The third state is a fact this
        plane owns, so it answers it here instead of by exception.
        """

        name = canonical_text(signal_name, "signal name")
        with self._lock:
            state = self._state_for_signal_locked(name)
            return state is not None and not state.retired

    def latest_publication(self, signal_name: str) -> SignalPublication | None:
        name = canonical_text(signal_name, "signal name")
        with self._lock:
            state = self._state_for_signal_locked(name)
            return None if state is None else state.publication

    def _follow_tap_locked(
        self,
        state: _GenerationState,
        signal_name: str,
        *,
        replay: bool,
    ) -> FollowTap[SignalPublication]:
        stream = self._ensure_publication_stream_locked(state)
        retained: list[tuple[int, SignalPublication]] = []
        if replay:
            committed = state.commit_chunks.get(signal_name, ())
            if committed:
                for sequence, value, _origin, parents in committed:
                    if (
                        state.publication is not None
                        and state.publication.event_ref.sequence == sequence
                    ):
                        publication = state.publication
                    else:
                        publication = SignalPublication(
                            EventRef(
                                StreamId(state.owner_id),
                                state.generation,
                                sequence,
                            ),
                            {signal_name: value},
                            self._publication_issuer,
                            direct_parent_refs=tuple(
                                parent.event_ref for parent in parents
                            ),
                            run_record=value.run_record,
                            event_record=value.event_record,
                        )
                        self._publication_parents[publication] = parents
                    retained.append((sequence, publication))
            elif state.publication is not None:
                retained.append(
                    (state.publication.event_ref.sequence, state.publication)
                )
        return stream.follow(retained)

    def follow_publications(
        self,
        signal_name: str,
        *,
        replay: bool = True,
    ) -> tuple[SignalPublication | None, FollowTap[SignalPublication]]:
        """Return the current event and an ordered replay/future payload tap.

        An ARMED generation that has not published yet is followable: an
        externally triggered camera chain publishes nothing until something
        fires its triggers, and a consumer that will itself cause the first
        frame must be able to subscribe to that silence.  The baseline is
        None exactly then.  A signal with no reserved generation at all is
        unknown here, and a finished one stays a loud
        ``SourceGenerationEnded``.
        """

        if type(replay) is not bool:
            raise TypeError("replay must be bool")

        name = canonical_text(signal_name, "signal name")
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._state_for_signal_locked(name)
            if state is None:
                raise LookupError(f"signal {name!r} has no reserved generation")
            if state.retired or state.terminal:
                raise SourceGenerationEnded(
                    f"signal {name!r} generation is not live"
                )
            return state.publication, self._follow_tap_locked(
                state,
                name,
                replay=replay,
            )

    def follower_edges(self) -> frozenset[tuple[str, str]]:
        """(source signal, follower signal) pairs of live presentation-paced routes.

        A follower (``coherent=False``, a panel's accepted-fit signals)
        publishes only AFTER its source presents, so a presentation layer
        that shows both sides must hold a join window open for the
        follower's batch.  This is the sole authority on which edges exist;
        the scheduler intersects it with the displayed set each tick.
        """

        with self._lock:
            edges: set[tuple[str, str]] = set()
            for state in self._states.values():
                if (
                    state.coherent
                    or state.retired
                    or state.terminal
                    or state.source_name is None
                ):
                    continue
                for name in state.output_names:
                    edges.add((state.source_name, name))
            return frozenset(edges)

    def direct_parent_publications(
        self,
        publication: SignalPublication,
    ) -> tuple[SignalPublication, ...]:
        """Resolve only one issued event's exact process-local parent payloads."""

        with self._lock:
            return self._resolved_direct_parents_locked(publication)

    def publication_roots(
        self,
        publication: SignalPublication,
    ) -> frozenset[EventRef]:
        """Return the parentless event refs in one publication's causal chain."""

        from .front import _publication_roots

        with self._lock:
            return _publication_roots(
                publication,
                self._resolved_direct_parents_locked,
            )

    def attach_latest_only_processor(
        self,
        node: object,
        *,
        source_name: str,
        initial_publication: SignalPublication,
        coherent: bool = True,
        paused: bool = False,
    ) -> None:
        """Attach one reactive latest-only Processor to a live source.

        ``coherent=False`` declares a presentation-paced follower: a route
        whose publications advance only AFTER its source was presented (a
        panel's accepted-fit signals).  Such a route keeps full lineage but
        never joins the same-shot front component of its source -- holding
        the source's selection for it would deadlock: the source waits for
        the follower that waits for the source's next presentation.

        ``paused=True`` reserves the derived generation against an earlier
        publication of the same live run without starting numeric work.  Its
        caller may first commit the exact answer already shown on screen, then
        call :meth:`catch_up_latest_only_processor` to join the current source
        and all later publications without changing the derived generation.
        """

        source_name = canonical_text(source_name, "processor source name")
        if not isinstance(initial_publication, SignalPublication):
            raise TypeError("Processor requires an exact SignalPublication")
        source = initial_publication.value(source_name)
        if source is None:
            raise ValueError("Processor publication has no selected signal")
        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            self._require_issued_publication_locked(initial_publication)
            source_state = self._state_for_signal_locked(source_name)
            if source_state is None or source_state.publication is None:
                raise RuntimeError("Processor source generation is not available")
            current_publication = source_state.publication
            same_generation = (
                initial_publication.event_ref.stream_id.value
                == source_state.owner_id
                and initial_publication.event_ref.generation
                == source_state.generation
            )
            if not same_generation or (
                initial_publication.event_ref.sequence
                > current_publication.event_ref.sequence
            ):
                raise RuntimeError(
                    "Processor source is not in the current live generation"
                )
            if not paused and current_publication is not initial_publication:
                raise RuntimeError(
                    "Processor source is not the exact current publication"
                )
            if source_state.terminal:
                raise RuntimeError("Processor source generation is not live")
            state = self._install_state_locked(
                owner_id=owner_id,
                kind="processor",
                output_names=output_names,
                bare_names=bare_names,
                node=node,
                source_name=source_name,
                coherent=coherent,
            )
        try:
            self._lane.attach_processor(
                node,
                source_name,
                initial_publication,
                paused=paused,
            )
        except BaseException:
            with self._lock:
                if self._states.get(owner_id) is state:
                    self._drop_state_locked(state)
                    self._membership_changed = True
            raise

    def catch_up_latest_only_processor(self, node: object) -> None:
        """Activate one paused route at its source run's current publication."""

        owner_id = _node_instance_id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.terminal
                or state.kind != "processor"
                or state.node is not node
                or state.source_owner_id is None
                or state.source_generation is None
            ):
                raise RuntimeError("latest-only Processor is no longer active")
            source_state = self._states.get(state.source_owner_id)
            if (
                source_state is None
                or source_state.retired
                or source_state.terminal
                or source_state.generation != state.source_generation
                or source_state.publication is None
            ):
                raise RuntimeError("latest-only Processor source is no longer live")
            publication = source_state.publication
        self._lane.catch_up_processor(node, publication)

    def reserve_frozen_processor(
        self,
        node: object,
        *,
        source_name: str,
        source_publication: SignalPublication,
    ) -> None:
        """Reserve a one-shot Processor derived from one retained FINAL source."""

        source_name = canonical_text(source_name, "processor source name")
        if not isinstance(source_publication, SignalPublication):
            raise TypeError("frozen Processor requires an exact SignalPublication")
        source = source_publication.value(source_name)
        if source is None:
            raise ValueError("frozen Processor publication has no selected signal")
        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            self._require_issued_publication_locked(source_publication)
            source_state = self._state_for_signal_locked(source_name)
            if (
                source_state is None
                or source_state.publication is not source_publication
            ):
                raise RuntimeError(
                    "frozen Processor source is not the exact current publication"
                )
            if not source_state.terminal:
                raise RuntimeError("frozen Processor source generation is still live")
            self._install_state_locked(
                owner_id=owner_id,
                kind="processor",
                output_names=output_names,
                bare_names=bare_names,
                node=node,
                source_name=source_name,
            )

    def reserve_follow_processor(
        self,
        node: object,
        *,
        source_name: str,
        source_publication: SignalPublication | None,
    ) -> FollowTap[SignalPublication]:
        """Bind one Processor to the current exact publication and its future events.

        ``source_publication`` None binds to an ARMED source that has not
        published yet: the replay starts empty (or from whatever committed
        in the race between the caller's snapshot and this bind -- still
        this generation's stream, delivered exactly) and the first commit
        is the first input.  Non-None stays the exactness anchor it was.
        """

        source_name = canonical_text(source_name, "processor source name")
        if source_publication is not None:
            if not isinstance(source_publication, SignalPublication):
                raise TypeError(
                    "Follow Processor requires an exact SignalPublication"
                )
            source = source_publication.value(source_name)
            if source is None:
                raise ValueError(
                    "Follow Processor publication has no selected signal"
                )
        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            if source_publication is not None:
                self._require_issued_publication_locked(source_publication)
            source_state = self._state_for_signal_locked(source_name)
            if source_state is None:
                raise SourceGenerationEnded(
                    "Follow Processor source is not live"
                )
            if (
                source_publication is not None
                and source_state.publication is not source_publication
            ):
                raise SourceGenerationEnded(
                    "Follow Processor source is not the exact current publication"
                )
            if source_state.terminal:
                raise SourceGenerationEnded(
                    "Follow Processor source generation is not live"
                )
            tap = self._follow_tap_locked(
                source_state,
                source_name,
                replay=True,
            )
            try:
                self._install_state_locked(
                    owner_id=owner_id,
                    kind="processor",
                    output_names=output_names,
                    bare_names=bare_names,
                    node=node,
                    source_name=source_name,
                )
            except BaseException:
                tap.close()
                raise
        return tap

    def cancel_latest_only_processor(self, node: object) -> bool:
        idle = self._lane.cancel_processor(node)
        self.withdraw_processor(node)
        return idle

    def withdraw_processor(self, node: object) -> None:
        self._withdraw_owner(_node_instance_id(node))

    def _require_issued_publication_locked(
        self,
        publication: SignalPublication,
    ) -> None:
        if not isinstance(publication, SignalPublication):
            raise TypeError("signal parent must be SignalPublication")
        if publication._issuer is not self._publication_issuer:
            raise ValueError("signal publication was not issued by this data plane")

    def _slim_publication_locked(
        self,
        publication: SignalPublication,
        selected_signal: str | None,
        memo: dict[SignalPublication, SignalPublication],
    ) -> SignalPublication:
        """Retain one causal route without retaining unconsumed siblings."""

        existing = memo.get(publication)
        if existing is not None:
            return existing
        if selected_signal is None:
            raise RuntimeError("derived publication has no selected source signal")
        value = publication.value(selected_signal)
        if value is None:
            raise RuntimeError("causal parent lost its selected source signal")
        parents = self._resolved_direct_parents_locked(publication)
        state = self._states.get(publication.event_ref.stream_id.value)
        parent_signal = (
            None
            if state is None or state.generation != publication.event_ref.generation
            else state.source_name
        )
        slim_parents = tuple(
            self._slim_publication_locked(parent, parent_signal, memo)
            for parent in parents
        )
        slim = SignalPublication(
            publication.event_ref,
            {selected_signal: value},
            self._publication_issuer,
            direct_parent_refs=tuple(parent.event_ref for parent in slim_parents),
            run_record=publication.run_record,
            event_record=publication.event_record,
        )
        memo[publication] = slim
        self._publication_parents[slim] = slim_parents
        return slim

    def _resolved_direct_parents_locked(
        self,
        publication: SignalPublication,
    ) -> tuple[SignalPublication, ...]:
        """Resolve private parent payloads without weakening public lineage."""

        self._require_issued_publication_locked(publication)
        try:
            parents = self._publication_parents[publication]
        except KeyError as error:
            raise RuntimeError(
                "signal publication parent payloads are no longer retained"
            ) from error
        if (
            tuple(parent.event_ref for parent in parents)
            != publication.direct_parent_refs
        ):
            raise RuntimeError("signal publication parent refs are inconsistent")
        return parents

    def _require_route_parent_locked(
        self,
        state: _GenerationState,
        publication: SignalPublication,
    ) -> SignalValue:
        self._require_issued_publication_locked(publication)
        source_name = state.source_name
        if source_name is None:
            raise RuntimeError("derived generation has no frozen source")
        if (
            publication.event_ref.stream_id.value != state.source_owner_id
            or publication.event_ref.generation != state.source_generation
        ):
            raise ValueError("signal parent belongs to another source generation")
        source = publication.value(source_name)
        if source is None:
            raise ValueError("signal parent lacks the frozen route source")
        return source

    def _validate_generation_values_locked(
        self,
        state: _GenerationState,
        values: Mapping[str, SignalValue],
        *,
        terminal: bool,
    ) -> None:
        names = tuple(values)
        if not names or not set(names).issubset(state.output_names):
            raise ValueError(
                "signal publication is outside its declared output vocabulary"
            )
        canonical_names = tuple(
            name for name in state.output_names if name in values
        )
        if not terminal:
            if state.published_names is None:
                state.published_names = canonical_names
            elif state.published_names != canonical_names:
                raise ValueError(
                    "live sibling bundle changed inside one generation"
                )
        schemas = {
            name: values[name].snapshot.block.schema
            for name in canonical_names
        }
        if any(not isinstance(schema, DatasetSchema) for schema in schemas.values()):
            raise TypeError("signal publication block must own a DatasetSchema")
        prior = {} if state.published_schemas is None else dict(
            state.published_schemas
        )
        for name, schema in schemas.items():
            if name in prior and prior[name] != schema:
                raise ValueError(
                    "signal publication schema changed inside one generation"
                )
            prior[name] = schema
        state.published_schemas = MappingProxyType(prior)

    def _publish_locked(
        self,
        state: _GenerationState,
        values: Mapping[str, SignalValue],
        *,
        parents: tuple[SignalPublication, ...] = (),
        terminal: bool = False,
        notify: bool = True,
    ) -> SignalPublication:
        if state.retired or self._states.get(state.owner_id) is not state:
            raise RuntimeError("signal generation is no longer active")
        if state.terminal:
            raise RuntimeError("signal generation has already published terminal")
        if len({id(parent) for parent in parents}) != len(parents):
            raise ValueError("signal publication parents must be unique")
        for parent in parents:
            self._require_issued_publication_locked(parent)
        frozen = MappingProxyType(
            {
                name: values[name]
                for name in state.output_names
                if name in values
            }
        )
        run_record = _shared_run_record(frozen)
        event_record = _shared_event_record(frozen)
        self._validate_generation_values_locked(
            state,
            frozen,
            terminal=terminal,
        )
        publication = SignalPublication(
            event_ref=EventRef(
                StreamId(state.owner_id),
                state.generation,
                state.next_sequence,
            ),
            signals=frozen,
            _issuer=self._publication_issuer,
            direct_parent_refs=tuple(parent.event_ref for parent in parents),
            run_record=run_record,
            event_record=event_record,
        )
        self._publication_parents[publication] = parents
        state.next_sequence += 1
        state.publication = publication
        state.failure = None
        state.terminal = terminal
        producer = state.publication_stream
        if notify and producer is not None:
            producer.emit(
                publication,
                sequence=publication.event_ref.sequence,
            )
            if terminal:
                producer.finish()
        self._membership_changed = True
        return publication

    def _reserve_retirement_closure_locked(
        self,
        root_owner_id: str,
    ) -> tuple[_GenerationState, ...]:
        """Atomically retire and reserve one owner plus all descendants."""

        while True:
            root = self._states.get(root_owner_id)
            if root is None:
                return ()
            selected = {(root.owner_id, root.generation)}
            changed = True
            while changed:
                changed = False
                for state in self._states.values():
                    reference = self._generation_ref(state)
                    source_ref = (
                        None
                        if state.source_owner_id is None
                        else (state.source_owner_id, state.source_generation)
                    )
                    if reference not in selected and source_ref in selected:
                        selected.add(reference)
                        changed = True
            states = tuple(
                state
                for state in self._states.values()
                if self._generation_ref(state) in selected
            )
            owners = {state.owner_id for state in states}
            if not self._starting.intersection(owners):
                self._starting.update(owners)
                for state in states:
                    state.retired = True
                self._membership_changed = True
                return states
            self._generation_ready.wait()
            if self._closed:
                raise RuntimeError("signal data plane is closed")

    def _withdraw_owner(self, owner_id: str) -> frozenset[str]:
        with self._lock:
            self._wait_for_start_locked(owner_id)
            states = self._reserve_retirement_closure_locked(owner_id)
            retiring_owners = tuple(state.owner_id for state in states)
        retired_names = frozenset(
            name for state in states for name in state.output_names
        )
        try:
            errors = self._cleanup_retired_states(states)
        except BaseException as error:
            errors = (error,)
        with self._lock:
            if not errors:
                for state in states:
                    self._drop_state_locked(state)
            self._starting.difference_update(retiring_owners)
            self._generation_ready.notify_all()
        if errors:
            raise BaseExceptionGroup(
                "signal generation cleanup failed",
                list(errors),
            )
        return retired_names

    def _cleanup_retired_states(
        self,
        states: tuple[_GenerationState, ...],
    ) -> tuple[BaseException, ...]:
        errors = []
        for state in states:
            producer = state.publication_stream
            if producer is not None and not state.publication_stream_cleaned:
                try:
                    producer.fail(SourceFailed("signal generation retired"))
                except (SourceFailed, StreamEndedEarly):
                    state.publication_stream_cleaned = True
                except BaseException as error:
                    errors.append(error)
                else:
                    state.publication_stream_cleaned = True
            if (
                state.kind == "processor"
                and state.node is not None
                and not state.processor_cleanup_complete
            ):
                # Routing retirement and execution retirement are distinct.
                # Keep a cancelled entry lane-owned until its prepare/work
                # Future completes so the node receives exactly one terminal
                # acknowledgement from ``_cancelled_processor``.
                try:
                    idle = self._lane.cancel_processor(state.node)
                except BaseException as error:
                    errors.append(error)
                else:
                    # An absent or idle entry is already terminal.  A busy
                    # entry remains a routing tombstone until the lane invokes
                    # ``_processor_cleanup_completed`` after the Future ends.
                    if idle:
                        state.processor_cleanup_complete = True
        return tuple(errors)

    def retire(self, node: object) -> frozenset[str]:
        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if state is not None and state.node is not node:
                raise RuntimeError("signal owner id belongs to another generation")
        return self._withdraw_owner(owner_id)

    def _build_front_locked(self) -> SignalFront:
        from .front import build_front

        return build_front(
            self._states,
            self._front_signals,
            self._front,
            self._resolved_direct_parents_locked,
        )

    def freeze(self) -> SignalFront:
        """Return the coherent front of already-committed publications."""

        self._lane.drain_processors()
        with self._lock:
            if not self._membership_changed:
                return self._front
            latest = {
                name: publication
                for state in self._states.values()
                if not state.retired and state.publication is not None
                for name, publication in (
                    (name, state.publication)
                    for name in state.publication.signals
                )
            }
            self._membership_changed = False
        self._lane.route(latest)
        with self._lock:
            front = self._build_front_locked()
            self._front = front
            # The first lock cleared the work it captured.  A commit while
            # ``route`` ran has since set this flag again; clearing it here
            # would lose that publication and its latest-only processor wake.
            return front

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            while self._starting:
                self._generation_ready.wait()
            self._closed = True
            states = tuple(self._states.values())
            self._states.clear()
            self._indexed_history_demands.clear()
            self._front_signals = frozenset()
            self._front = SignalFront({}, {})
            self._publication_parents.clear()
            self._publication_callbacks.clear()
        self._lane.close()
        errors = list(self._cleanup_retired_states(states))
        if errors:
            raise BaseExceptionGroup("signal data plane close failed", errors)
