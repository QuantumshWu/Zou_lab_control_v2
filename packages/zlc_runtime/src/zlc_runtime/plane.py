"""Headless signal plane with producer-local causal coherence.

A hosted producer publishes through a ``LiveDatasetPort``; this plane freezes
only sources that reported a new revision. Each source is one producer
transaction. Combining their latest immutable fronts for one consumer cycle
does not assert that independent producers observed the same physical event.

Freeze-latest, not a bus.  Each changed slot materialises its own atomic
transaction exactly once; unchanged slots reuse their immutable fronts.  Independent
producers still advance independently.  Within one explicit source -> Processor
component, however, a newer source and its active descendants replace the previous
component together. A slow Processor therefore cannot expose source revision N
beside its own derived revision N-1.

A monitor tap overwrites when its consumer cannot keep up rather than
back-pressuring acquisition.  It retains only the current value; it does not
record or expose a loss count.  There is no global shot counter to compare
against -- signals from different runs advance independently.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, runtime_checkable
import uuid
from weakref import WeakKeyDictionary

from zlc_data import (
    BlockId,
    DataBlock,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    StreamGenerationId,
)
from .dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from .dataset import (
    DatasetCoverage,
    MonitorCoverage,
)
from .streams import (
    AcquisitionProducer,
    AcquisitionStream,
    EventRef,
    FollowTap,
    SourceFailed,
    StreamEndedEarly,
    StreamId,
)
from zlc_data import canonical_text

__all__ = [
    "DerivedSignalOutput",
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


@dataclass(frozen=True, slots=True)
class DerivedSignalOutput:
    """One consumer-derived immutable value without presentation metadata."""

    snapshot: OwnedSnapshot
    preserve_source_coverage: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("derived signal snapshot must be OwnedSnapshot")
        if type(self.preserve_source_coverage) is not bool:
            raise TypeError("preserve_source_coverage must be bool")


@dataclass(frozen=True)
class SignalValue:
    """One signal at one producer-owned immutable revision."""

    name: str
    snapshot: OwnedSnapshot
    coverage: DatasetCoverage | MonitorCoverage | None
    transient: bool = False         # withdrawn with its live producer
    run_record: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = canonical_text(self.name, "signal name")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("signal snapshot must be OwnedSnapshot")
        if self.coverage is not None and not isinstance(
            self.coverage,
            (DatasetCoverage, MonitorCoverage),
        ):
            raise TypeError("signal coverage has an unknown type")
        if not isinstance(self.transient, bool):
            raise TypeError("signal transient flag must be bool")
        if not isinstance(self.run_record, Mapping):
            raise TypeError("signal value run_record must be a mapping")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "run_record",
            MappingProxyType(dict(self.run_record)),
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
        run_record = dict(self.run_record)
        if any(
            not _run_records_equal(value.run_record, run_record)
            for value in signals.values()
        ):
            raise ValueError(
                "signal publication run_record differs from its sibling values"
            )
        object.__setattr__(self, "signals", MappingProxyType(signals))
        object.__setattr__(self, "direct_parent_refs", parents)
        object.__setattr__(
            self,
            "run_record",
            MappingProxyType(run_record),
        )

    def value(self, name: str) -> SignalValue | None:
        return self.signals.get(str(name))


class _SignalPublicationPayloadContract:
    """Immutable payload contract for a generation's lossless FollowTap."""

    @staticmethod
    def snapshot(payload: SignalPublication) -> SignalPublication:
        return payload

    @staticmethod
    def validate(payload: SignalPublication) -> None:
        if not isinstance(payload, SignalPublication):
            raise TypeError("followed signal payload must be SignalPublication")


_SIGNAL_PUBLICATION_CONTRACT = _SignalPublicationPayloadContract()


@dataclass(frozen=True)
class SignalFront:
    """Immutable front: coherent derived components, independent producers."""

    signals: Mapping[str, SignalValue]
    failures: Mapping[str, str]     # producer instance_id -> freeze failure
    publication_by_signal: Mapping[str, SignalPublication] = field(
        default_factory=dict,
        repr=False,
    )
    _continuous_group_by_signal: Mapping[str, frozenset[str]] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        signals = dict(self.signals)
        failures = dict(self.failures)
        publications = dict(self.publication_by_signal)
        groups = {
            str(name): frozenset(members)
            for name, members in self._continuous_group_by_signal.items()
        }
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
            for name, members in groups.items():
                assert name in signals and name in members, (
                    "continuous group must contain its visible signal"
                )
                assert members and members.issubset(signals), (
                    "continuous group contains a non-visible signal"
                )
                assert all(groups.get(member) == members for member in members), (
                    "continuous signal groups must be symmetric"
                )
        object.__setattr__(self, "signals", MappingProxyType(signals))
        object.__setattr__(self, "failures", MappingProxyType(failures))
        object.__setattr__(
            self,
            "publication_by_signal",
            MappingProxyType(publications),
        )
        object.__setattr__(
            self,
            "_continuous_group_by_signal",
            MappingProxyType(groups),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self.signals)

    def value(self, name: str) -> SignalValue | None:
        return self.signals.get(str(name))

    def publication(self, name: str) -> SignalPublication | None:
        return self.publication_by_signal.get(str(name))

    def continuous_group(self, name: str) -> frozenset[str]:
        """Return the already-resolved visible causal group for one signal."""

        selected = str(name)
        if selected not in self.signals:
            return frozenset()
        return self._continuous_group_by_signal.get(
            selected,
            frozenset((selected,)),
        )


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
    output: FinalDatasetOutput | LiveDatasetOutput,
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
        FinalDatasetOutput | LiveDatasetOutput | SignalValue,
    ],
) -> Mapping[str, object]:
    """Copy the one run record shared by an atomic sibling output bundle."""

    shared: dict[str, object] | None = None
    for output in outputs.values():
        if not isinstance(
            output,
            (FinalDatasetOutput, LiveDatasetOutput, SignalValue),
        ):
            raise TypeError("run_record carrier has an unknown output type")
        record = (
            {}
            if output.run_record is None
            else dict(output.run_record)
        )
        if shared is None:
            shared = record
            continue
        if not _run_records_equal(record, shared):
            raise ValueError("sibling outputs must share one run_record")
    return {} if shared is None else shared


def _require_signal_producer(node: object) -> SignalProducer:
    if not isinstance(node, SignalProducer):
        raise TypeError("signal producer must implement SignalProducer")
    return node


def _node_instance_id(node: object) -> str:
    """Return the stable producer identity required by the signal plane."""

    producer = _require_signal_producer(node)
    return canonical_text(producer.instance_id, "signal producer instance_id")


@dataclass(slots=True)
class _GenerationState:
    """The sole mutable state for one process-local signal generation."""

    owner_id: str
    generation: StreamGenerationId
    kind: str
    output_names: tuple[str, ...]
    bare_names: Mapping[str, str]
    owner_token: object = field(default_factory=object, repr=False)
    node: object | None = None
    slot: object | None = None
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
    publication_producer: AcquisitionProducer[SignalPublication] | None = None


@dataclass(slots=True)
class _ProcessorEntry:
    node: LatestProcessorControl
    source_name: str
    work_future: Future | None = None
    work_publication: SignalPublication | None = None
    pending_publication: SignalPublication | None = None
    last_publication: SignalPublication | None = None
    cancel_requested: bool = False


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

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="signal-latest-processor",
        )
        self._lock = threading.Lock()
        self._processors: dict[str, _ProcessorEntry] = {}
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
            pending_publication=initial_publication,
            last_publication=initial_publication,
        )
        with self._lock:
            if key in self._processors:
                raise RuntimeError("Processor node is already attached")
            self._processors[key] = entry
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
            self._processors.pop(_node_instance_id(entry.node), None)
        entry.node.accept_processor_failure(error)

    def _cancelled_processor(self, entry: _ProcessorEntry) -> None:
        with self._lock:
            self._processors.pop(_node_instance_id(entry.node), None)
        entry.node.accept_processor_cancelled()

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
        self._lane = _LatestOnlyProcessorLane()
        self._publication_issuer = object()
        self._publication_parents: WeakKeyDictionary[
            SignalPublication,
            tuple[SignalPublication, ...],
        ] = WeakKeyDictionary()
        self._states: dict[str, _GenerationState] = {}
        self._dirty: set[str] = set()
        self._front_signals: frozenset[str] = frozenset()
        self._membership_changed = False
        self._closed = False
        self._request_owner_wake: Callable[[], None] | None = None
        self._owner_wake_token: object | None = None
        self._front = SignalFront({}, {})

    def bind_owner_wake(self, request_owner_wake: Callable[[], None]) -> object:
        if not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            if self._request_owner_wake is not None:
                raise RuntimeError("signal data plane owner wake is already bound")
            token = object()
            self._request_owner_wake = request_owner_wake
            self._owner_wake_token = token
            return token

    def unbind_owner_wake(self, token: object) -> None:
        """Release only the exact Workbench wake borrow that was admitted."""

        with self._lock:
            if token is not self._owner_wake_token:
                raise RuntimeError("signal data plane owner wake token is not current")
            self._request_owner_wake = None
            self._owner_wake_token = None

    def set_front_signals(self, signal_names) -> None:
        """Set the connected continuous signal set whose front must be coherent."""

        names = frozenset(
            canonical_text(name, "connected signal name")
            for name in signal_names
        )
        wake = None
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            if names != self._front_signals:
                self._front_signals = names
                self._membership_changed = True
                wake = self._request_owner_wake
        if wake is not None:
            wake()

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
            state = self._state_for_signal_locked(name)
            if state is not None and state.owner_id != owner_id:
                raise RuntimeError(
                    f"signal {name!r} is already owned by {state.owner_id!r}"
                )

    @staticmethod
    def _generation_ref(
        state: _GenerationState,
    ) -> tuple[str, StreamGenerationId]:
        return state.owner_id, state.generation

    def _drop_state_locked(self, state: _GenerationState) -> None:
        if self._states.get(state.owner_id) is not state:
            return
        self._states.pop(state.owner_id)
        state.retired = True
        state.publication = None
        state.failure = None
        self._dirty.discard(state.owner_id)

    def _install_state_locked(
        self,
        *,
        owner_id: str,
        kind: str,
        output_names: tuple[str, ...],
        bare_names: Mapping[str, str],
        node: object | None = None,
        slot: object | None = None,
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
            node=node,
            slot=slot,
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
            stream, producer = AcquisitionStream.create(
                StreamId(f"{state.owner_id}:signal-publications"),
                _SIGNAL_PUBLICATION_CONTRACT,
            )
            state.publication_stream = stream
            state.publication_producer = producer
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

        The composition owner performs this before submitting a run.  A later
        live attachment upgrades the same state in place; a FINAL-only run uses
        it directly.  Publication can therefore never recreate a generation
        after retirement.
        """

        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            existing = self._states.get(owner_id)
            if existing is not None:
                if (
                    not existing.retired
                    and not existing.terminal
                    and existing.kind == "producer"
                    and existing.node is node
                    and existing.output_names == output_names
                    and dict(existing.bare_names) == dict(bare_names)
                    and existing.slot is None
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
        reusable object, so something has to decide when the previous run's
        generation ends.  It cannot end at ``publish_final``: the result must
        stay readable afterwards, which is exactly why nobody retires it.  It
        ends when the next run begins -- and that is this method.

        Use this to START a run.  ``reserve`` is the lower-level operation that
        refuses to touch an existing generation at all; a caller that reserves
        directly can never run the same node twice, because the first run leaves
        a terminal generation behind and the second reservation is rejected.

        A generation that is still LIVE is not superseded: two concurrent runs of
        one producer is a real error and still raises.
        """

        owner_id = _node_instance_id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            existing = self._states.get(owner_id)
            finished = existing is not None and (existing.retired or existing.terminal)
        if finished:
            self._withdraw_owner(owner_id)
        return self.reserve(node)

    def attach(
        self,
        node,
        slot,
    ) -> None:
        if slot is None:
            raise ValueError("a monitor slot is required")
        if not callable(getattr(slot, "freeze_live_outputs", None)):
            raise TypeError(
                "live slot must expose application-owned freeze_live_outputs()"
            )
        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.terminal
                or state.kind != "producer"
                or state.node is not node
                or state.output_names != output_names
                or dict(state.bare_names) != dict(bare_names)
                or state.slot is not None
            ):
                raise RuntimeError(
                    "live output requires the producer's reserved generation"
                )
            state.slot = slot
            self._membership_changed = True

    def mark_changed(self, node, slot) -> None:
        """Dirty only the exact live slot that emitted this wake."""

        if slot is None:
            raise ValueError("changed live slot must not be None")
        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.terminal
                or state.node is not node
                or state.slot is not slot
            ):
                return
            self._dirty.add(owner_id)
            output_names = frozenset(state.output_names)
            wake = (
                self._request_owner_wake
                if any(
                    candidate.source_name in output_names
                    for candidate in self._states.values()
                    if (
                        not candidate.retired
                        and candidate.kind in {"processor", "continuous"}
                    )
                )
                else None
            )
        if wake is not None:
            wake()

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
                declarations = (
                    {}
                    if state.node is None
                    else _declared_outputs(
                        _require_signal_producer(
                            state.node
                        ).dataset_output_declarations
                    )
                )
                for name in state.output_names:
                    bare_name = state.bare_names.get(name)
                    declaration = declarations.get(bare_name)
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
                            shape=None if value is None else value.shape,
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

    def latest_publication(self, signal_name: str) -> SignalPublication | None:
        name = canonical_text(signal_name, "signal name")
        with self._lock:
            state = self._state_for_signal_locked(name)
            return None if state is None else state.publication

    def follow_publications(
        self,
        signal_name: str,
    ) -> tuple[SignalPublication, FollowTap[SignalPublication]]:
        """Return one live signal's exact current event and every future event."""

        name = canonical_text(signal_name, "signal name")
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._state_for_signal_locked(name)
            if state is None or state.publication is None:
                raise LookupError(f"signal {name!r} has no current publication")
            if state.retired or state.terminal:
                raise RuntimeError(f"signal {name!r} generation is not live")
            stream = self._ensure_publication_stream_locked(state)
            return state.publication, stream.follow()

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

    def publication_owner(self, publication: SignalPublication) -> object | None:
        """Return the exact active generation owner of one issued publication.

        Composition occasionally needs generation-static leaf facts that live
        on the admitted node (for example Calibration geometry).  Resolving
        those facts by scanning a current UI row with the same textual owner id
        can splice an older retained front into a restarted node.  The plane is
        already the sole generation owner, so it alone may resolve this narrow
        process-local reference.  No presentation state is stored here.
        """

        with self._lock:
            self._require_issued_publication_locked(publication)
            state = self._states.get(publication.event_ref.stream_id.value)
            if (
                state is None
                or state.retired
                or state.generation != publication.event_ref.generation
            ):
                return None
            return state.owner_token

    def attach_latest_only_processor(
        self,
        node: object,
        *,
        source_name: str,
        initial_publication: SignalPublication,
        coherent: bool = True,
    ) -> None:
        """Attach one reactive latest-only Processor to a live source.

        ``coherent=False`` declares a presentation-paced follower: a route
        whose publications advance only AFTER its source was presented (a
        panel's accepted-fit signals).  Such a route keeps full lineage but
        never joins the same-shot front component of its source -- holding
        the source's selection for it would deadlock: the source waits for
        the follower that waits for the source's next presentation.
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
            if (
                source_state is None
                or source_state.publication is not initial_publication
            ):
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
            )
        except BaseException:
            with self._lock:
                if self._states.get(owner_id) is state:
                    self._drop_state_locked(state)
                    self._membership_changed = True
            raise

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
        if source.coverage is not None:
            raise ValueError("frozen Processor source must be a FINAL signal")
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
        source_publication: SignalPublication,
    ) -> FollowTap[SignalPublication]:
        """Bind one Processor to the current exact publication and its future events."""

        source_name = canonical_text(source_name, "processor source name")
        if not isinstance(source_publication, SignalPublication):
            raise TypeError("Follow Processor requires an exact SignalPublication")
        source = source_publication.value(source_name)
        if source is None:
            raise ValueError("Follow Processor publication has no selected signal")
        if not isinstance(source.coverage, DatasetCoverage):
            raise ValueError("Follow Processor source must have DatasetCoverage")
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
                    "Follow Processor source is not the exact current publication"
                )
            if source_state.terminal:
                raise RuntimeError("Follow Processor source generation is not live")
            stream = self._ensure_publication_stream_locked(source_state)
            tap = stream.follow()
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

    def _require_route_parents_locked(
        self,
        state: _GenerationState,
        publications: tuple[SignalPublication, ...],
    ) -> tuple[SignalValue, ...]:
        parents = tuple(publications)
        if not parents or any(
            not isinstance(parent, SignalPublication) for parent in parents
        ):
            raise TypeError("derived result requires exact parent publications")
        if len({id(parent) for parent in parents}) != len(parents):
            raise ValueError("derived result parent publications must be unique")
        sequences = tuple(parent.event_ref.sequence for parent in parents)
        if sequences != tuple(sorted(sequences)):
            raise ValueError("derived result parents must follow source event order")
        values = tuple(
            self._require_route_parent_locked(state, parent) for parent in parents
        )
        return values

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
        )
        self._publication_parents[publication] = parents
        state.next_sequence += 1
        state.publication = publication
        state.failure = None
        state.terminal = terminal
        if terminal:
            self._dirty.discard(state.owner_id)
        producer = state.publication_producer
        if producer is not None:
            producer.emit(
                publication,
                captured_at=time.time(),
                direct_parent_refs=publication.direct_parent_refs,
            )
            if terminal:
                producer.finish()
        self._membership_changed = True
        return publication

    def publish_final(
        self,
        node,
        outputs: Mapping[str, FinalDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        if not isinstance(outputs, Mapping):
            raise TypeError("FINAL signals must be a mapping")
        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        declared = _declared_outputs(
            _require_signal_producer(node).dataset_output_declarations
        )
        if not outputs or not set(outputs).issubset(declared):
            raise ValueError(
                "FINAL publication must be a non-empty declared output subset"
            )
        values: dict[str, SignalValue] = {}
        by_bare = {bare: qualified for qualified, bare in bare_names.items()}
        run_record = _shared_run_record(outputs)
        for bare in declared:
            if bare not in outputs:
                continue
            output = outputs[bare]
            if not isinstance(output, FinalDatasetOutput):
                raise TypeError("FINAL values must be FinalDatasetOutput")
            _require_published_declaration(bare, output, declared)
            qualified = by_bare[bare]
            values[qualified] = SignalValue(
                name=qualified,
                snapshot=output.snapshot,
                coverage=None,
                transient=False,
                run_record=run_record,
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.node is not node
                or state.kind != "producer"
                or state.output_names != output_names
                or dict(state.bare_names) != dict(bare_names)
            ):
                raise RuntimeError("FINAL owner differs from its active generation")
            publication = self._publish_locked(
                state,
                values,
                terminal=True,
            )
        return publication.signals

    def publish_processor(
        self,
        node,
        outputs: Mapping[str, LiveDatasetOutput],
        *,
        source_publication: SignalPublication,
        trigger: tuple[str, int] | None = None,
    ) -> Mapping[str, SignalValue]:
        if not isinstance(outputs, Mapping):
            raise TypeError("processor outputs must be a mapping")
        if not isinstance(source_publication, SignalPublication):
            raise TypeError("Processor publication requires its exact parent")
        if trigger is not None:
            if (
                not isinstance(trigger, tuple)
                or len(trigger) != 2
                or not isinstance(trigger[0], str)
                or not trigger[0].strip()
                or isinstance(trigger[1], bool)
                or not isinstance(trigger[1], int)
                or trigger[1] < 0
            ):
                raise TypeError(
                    "processor trigger must be a (non-empty kind, non-negative revision) tuple"
                )
            trigger = (trigger[0].strip(), trigger[1])
        owner_id = _node_instance_id(node)
        declared = _declared_outputs(
            _require_signal_producer(node).dataset_output_declarations
        )
        if set(outputs) != set(declared):
            raise ValueError(
                "Processor publication must cover its complete frozen output vocabulary"
            )
        with self._lock:
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.kind != "processor"
                or state.node is not node
            ):
                raise RuntimeError("Processor generation is no longer active")
            self._require_route_parent_locked(state, source_publication)
            source_sequence = source_publication.event_ref.sequence
            if source_sequence < state.last_parent_sequence:
                raise RuntimeError("Processor result belongs to an obsolete parent")
            if (
                source_sequence == state.last_parent_sequence
                and (
                    trigger is None
                    or trigger == state.last_parent_trigger
                )
            ):
                raise RuntimeError("Processor result belongs to an obsolete parent")
            source_name = state.source_name
            bare_names = dict(state.bare_names)
        assert source_name is not None
        source = source_publication.value(source_name)
        if source is None:
            raise ValueError("Processor parent lacks its selected signal")
        by_bare = {bare: qualified for qualified, bare in bare_names.items()}
        values: dict[str, SignalValue] = {}
        run_record = _shared_run_record(outputs)
        for bare in declared:
            output = outputs[bare]
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError("processor outputs must be LiveDatasetOutput")
            _require_published_declaration(bare, output, declared)
            qualified = by_bare[bare]
            values[qualified] = SignalValue(
                name=qualified,
                snapshot=output.snapshot,
                coverage=output.coverage,
                transient=True,
                run_record=run_record,
            )
        with self._lock:
            if self._states.get(owner_id) is not state or state.retired:
                raise RuntimeError("Processor generation retired during publication")
            self._require_route_parent_locked(state, source_publication)
            source_sequence = source_publication.event_ref.sequence
            if source_sequence < state.last_parent_sequence:
                raise RuntimeError("Processor result belongs to an obsolete parent")
            if (
                source_sequence == state.last_parent_sequence
                and (
                    trigger is None
                    or trigger == state.last_parent_trigger
                )
            ):
                raise RuntimeError("Processor result belongs to an obsolete parent")
            publication = self._publish_locked(
                state,
                values,
                parents=(source_publication,),
            )
            state.last_parent_sequence = source_publication.event_ref.sequence
            state.last_parent_trigger = trigger
        return publication.signals

    def publish_terminal_processor(
        self,
        node: object,
        outputs: Mapping[str, LiveDatasetOutput],
        *,
        source_publication: SignalPublication,
    ) -> Mapping[str, SignalValue]:
        """Publish one terminal, retained Processor result with its exact parent."""

        if not isinstance(outputs, Mapping):
            raise TypeError("terminal Processor outputs must be a mapping")
        if not isinstance(source_publication, SignalPublication):
            raise TypeError("terminal Processor publication requires its exact parent")
        owner_id = _node_instance_id(node)
        declared = _declared_outputs(
            _require_signal_producer(node).dataset_output_declarations
        )
        if set(outputs) != set(declared):
            raise ValueError(
                "terminal Processor publication must cover its complete output vocabulary"
            )
        output_names, bare_names = self._node_route_names(node)
        by_bare = {bare: qualified for qualified, bare in bare_names.items()}
        values: dict[str, SignalValue] = {}
        run_record = _shared_run_record(outputs)
        for bare in declared:
            output = outputs[bare]
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError("terminal Processor outputs must be LiveDatasetOutput")
            _require_published_declaration(bare, output, declared)
            qualified = by_bare[bare]
            values[qualified] = SignalValue(
                name=qualified,
                snapshot=output.snapshot,
                coverage=None,
                transient=False,
                run_record=run_record,
            )
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
                or state.output_names != output_names
                or dict(state.bare_names) != dict(bare_names)
            ):
                raise RuntimeError("terminal Processor generation is no longer active")
            source = self._require_route_parent_locked(state, source_publication)
            if source.coverage is not None and not isinstance(
                source.coverage,
                DatasetCoverage,
            ):
                raise ValueError(
                    "terminal Processor parent must be FINAL or exact Dataset coverage"
                )
            source_state = self._states.get(state.source_owner_id or "")
            if (
                source_state is None
                or source_state.retired
                or not source_state.terminal
                or source_state.generation != state.source_generation
                or source_state.publication is not source_publication
            ):
                raise RuntimeError(
                    "terminal Processor parent is not the retained source terminal"
                )
            publication = self._publish_locked(
                state,
                values,
                parents=(source_publication,),
                terminal=True,
            )
            state.last_parent_sequence = source_publication.event_ref.sequence
        return publication.signals

    def bind_continuous_derived(
        self,
        owner_id: str,
        *,
        source_name: str,
        expected_source_generation: StreamGenerationId,
        output_names,
    ) -> StreamGenerationId:
        """Bind one derived sibling bundle to its direct source generation."""

        identity = canonical_text(owner_id, "derived route owner_id")
        source_name = canonical_text(source_name, "derived route source name")
        if not isinstance(expected_source_generation, StreamGenerationId):
            raise TypeError("expected_source_generation must be StreamGenerationId")
        names = tuple(
            canonical_text(name, "derived route output name")
            for name in output_names
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            source_state = self._state_for_signal_locked(source_name)
            if source_state is None:
                raise RuntimeError("derived route source generation is not active")
            if source_state.generation != expected_source_generation:
                raise RuntimeError("derived route source generation changed")
            existing = self._states.get(identity)
            if existing is not None and not existing.retired:
                same = (
                    existing.kind == "continuous"
                    and existing.source_name == source_name
                    and existing.source_generation == expected_source_generation
                    and existing.output_names == names
                )
                if same:
                    return existing.generation
                raise RuntimeError(
                    "derived generation changed; withdraw it before rebinding"
                )
            state = self._install_state_locked(
                owner_id=identity,
                kind="continuous",
                output_names=names,
                bare_names={name: name for name in names},
                source_name=source_name,
            )
            return state.generation

    @staticmethod
    def _route_owned_snapshot(
        state: _GenerationState,
        output_name: str,
        snapshot: OwnedSnapshot,
    ) -> OwnedSnapshot:
        """Bind one derived value to the plane-owned route generation.

        Materializers own values and schemas, but not live route identity.
        Assigning the Dataset ref here prevents frontend transforms, Fits, or
        payload hashes from becoming a parallel generation authority.  Values
        already crossed the immutable ownership boundary, so rebuilding the
        small DataBlock header retains their bytes-backed arrays without a
        scientific-data copy.
        """

        block_id = BlockId(f"signal/{state.owner_id}/{output_name}")
        ref = DatasetRevisionRef(
            block_id,
            state.generation,
            snapshot.block.schema.fingerprint,
            snapshot.ref.revision,
        )
        return OwnedSnapshot(
            ref,
            DataBlock(
                block_id,
                snapshot.block.revision,
                snapshot.block.values,
                snapshot.block.validity,
                snapshot.block.schema,
            ),
        )

    @staticmethod
    def _derived_values(
        state: _GenerationState,
        source_publication: SignalPublication,
        values: Mapping[str, DerivedSignalOutput],
        *,
        transient: bool,
    ) -> Mapping[str, SignalValue]:
        source_name = state.source_name
        if source_name is None:
            raise RuntimeError("derived generation has no source")
        source = source_publication.value(source_name)
        if source is None:
            raise ValueError("derived parent lacks its selected signal")
        if set(values) != set(state.output_names):
            raise ValueError(
                "derived publication differs from its frozen sibling vocabulary"
            )
        result = {}
        for name in state.output_names:
            value = values[name]
            if not isinstance(value, DerivedSignalOutput):
                raise TypeError("derived values must contain DerivedSignalOutput")
            result[name] = SignalValue(
                name=name,
                snapshot=SignalDataPlane._route_owned_snapshot(
                    state,
                    name,
                    value.snapshot,
                ),
                coverage=(
                    source.coverage if value.preserve_source_coverage else None
                ),
                transient=transient,
            )
        return MappingProxyType(result)

    def publish_continuous_derived(
        self,
        owner_id: str,
        generation: StreamGenerationId,
        source_publications: tuple[SignalPublication, ...],
        values: Mapping[str, DerivedSignalOutput],
    ) -> bool:
        identity = canonical_text(owner_id, "derived route owner_id")
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("derived generation must be StreamGenerationId")
        parents = tuple(source_publications)
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.retired
                or state.generation != generation
                or state.kind != "continuous"
            ):
                return False
            self._require_route_parents_locked(state, parents)
            latest_sequence = parents[-1].event_ref.sequence
            if latest_sequence <= state.last_parent_sequence:
                return False
        frozen = self._derived_values(
            state,
            parents[-1],
            values,
            transient=True,
        )
        with self._lock:
            if (
                self._states.get(identity) is not state
                or state.retired
                or state.generation != generation
            ):
                return False
            self._require_route_parents_locked(state, parents)
            latest_sequence = parents[-1].event_ref.sequence
            if latest_sequence <= state.last_parent_sequence:
                return False
            self._publish_locked(
                state,
                frozen,
                parents=parents,
            )
            state.last_parent_sequence = latest_sequence
            return True

    def fail_continuous_derived(
        self,
        owner_id: str,
        generation: StreamGenerationId,
        source_publications: tuple[SignalPublication, ...],
        error: Exception,
    ) -> bool:
        identity = canonical_text(owner_id, "derived route owner_id")
        parents = tuple(source_publications)
        if not isinstance(error, Exception):
            raise TypeError("derived failure must be an Exception")
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.retired
                or state.generation != generation
                or state.kind != "continuous"
            ):
                return False
            self._require_route_parents_locked(state, parents)
            latest_sequence = parents[-1].event_ref.sequence
            if latest_sequence <= state.last_parent_sequence:
                return False
            state.last_parent_sequence = latest_sequence
            state.failure = f"{type(error).__name__}: {error}"
            self._membership_changed = True
            return True

    def continuous_needs_publication(
        self,
        owner_id: str,
        generation: StreamGenerationId,
        source_publications: tuple[SignalPublication, ...],
    ) -> bool:
        """Whether one active route has not published this exact parent yet."""

        identity = canonical_text(owner_id, "derived route owner_id")
        parents = tuple(source_publications)
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.retired
                or state.kind != "continuous"
                or state.generation != generation
            ):
                return False
            self._require_route_parents_locked(state, parents)
            return parents[-1].event_ref.sequence > state.last_parent_sequence

    def withdraw_derived(self, owner_id: str) -> None:
        self._withdraw_owner(
            canonical_text(owner_id, "derived signal owner_id")
        )

    def _retirement_closure_locked(
        self,
        root_owner_id: str,
    ) -> tuple[_GenerationState, ...]:
        root = self._states.get(root_owner_id)
        if root is None or root.retired:
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
                if state.retired or reference in selected:
                    continue
                if source_ref in selected:
                    selected.add(reference)
                    changed = True
        states = tuple(
            state
            for state in self._states.values()
            if self._generation_ref(state) in selected
        )
        for state in states:
            self._drop_state_locked(state)
        self._membership_changed = True
        return states

    def _withdraw_owner(self, owner_id: str) -> frozenset[str]:
        with self._lock:
            states = self._retirement_closure_locked(owner_id)
        retired_names = frozenset(
            name for state in states for name in state.output_names
        )
        errors = self._cleanup_retired_states(states)
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
        slots = []
        producers = []
        for state in states:
            if state.kind == "processor" and state.node is not None:
                # Routing retirement and execution retirement are distinct.
                # Keep a cancelled entry lane-owned until its prepare/work
                # Future completes so the node receives exactly one terminal
                # acknowledgement from ``_cancelled_processor``.
                self._lane.cancel_processor(state.node)
            if state.slot is not None:
                slots.append(state.slot)
            if state.publication_producer is not None:
                producers.append(state.publication_producer)
        errors = []
        seen_producers = set()
        for producer in producers:
            if id(producer) in seen_producers:
                continue
            seen_producers.add(id(producer))
            try:
                producer.fail(SourceFailed("signal generation retired"))
            except StreamEndedEarly:
                pass
            except BaseException as error:
                errors.append(error)
        seen_slots = set()
        for slot in slots:
            if id(slot) in seen_slots:
                continue
            seen_slots.add(id(slot))
            try:
                slot.close()
            except BaseException as error:
                errors.append(error)
        return tuple(errors)

    def retire(self, node: object) -> frozenset[str]:
        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if state is not None and state.node is not node:
                raise RuntimeError("signal owner id belongs to another generation")
        return self._withdraw_owner(owner_id)

    def finish_live(self, node: object) -> bool:
        """Retain a completed exact live generation and finish its FollowTaps.

        Monitor/latest slots keep their existing detach behavior and return
        ``False``.  Exact Dataset slots return ``True`` after their final
        current snapshot is published and the generation becomes terminal.
        """

        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if state is None or state.retired or state.node is not node:
                raise RuntimeError("live owner differs from its active generation")
            publication = state.publication
            exact = bool(
                publication is not None
                and publication.signals
                and all(
                    isinstance(value.coverage, DatasetCoverage)
                    for value in publication.signals.values()
                )
            )
            slot = state.slot
        if not exact:
            self.detach_live(node)
            return False
        if slot is None:
            raise RuntimeError("exact live generation lost its output slot")

        values, warning = self._freeze_one(state)
        if warning is not None:
            raise RuntimeError(warning)
        if not all(
            isinstance(value.coverage, DatasetCoverage)
            for value in values.values()
        ):
            raise RuntimeError("exact live terminal changed its coverage extent")
        if not all(value.coverage.complete for value in values.values()):
            raise RuntimeError("exact live terminal Dataset coverage is incomplete")
        run_record = _shared_run_record(values)

        producer = None
        with self._lock:
            if (
                self._states.get(owner_id) is not state
                or state.retired
                or state.terminal
                or state.slot is not slot
            ):
                raise RuntimeError("exact live generation changed while finishing")
            current = state.publication
            same_current = bool(
                current is not None
                and set(current.signals) == set(values)
                and _run_records_equal(current.run_record, run_record)
                and all(
                    current.signals[name].snapshot.ref == value.snapshot.ref
                    and current.signals[name].coverage == value.coverage
                    for name, value in values.items()
                )
            )
            if not same_current:
                self._publish_locked(state, values)
            state.terminal = True
            state.slot = None
            self._dirty.discard(owner_id)
            self._membership_changed = True
            producer = state.publication_producer
        try:
            slot.close()
        finally:
            if producer is not None:
                producer.finish()
        return True

    def detach_live(self, node) -> None:
        owner_id = _node_instance_id(node)
        retained_slot = None
        retired_states = ()
        with self._lock:
            state = self._states.get(owner_id)
            if state is None:
                return
            if state.node is not node:
                raise RuntimeError("signal owner id belongs to another generation")
            retained_final = state.terminal or bool(
                state.publication is not None
                and all(
                    not value.transient
                    for value in state.publication.signals.values()
                )
            )
            if retained_final:
                retained_slot = state.slot
                state.slot = None
                self._dirty.discard(owner_id)
            else:
                # Withdraw this exact generation and its descendants before
                # any slow slot close/join.  A replacement may then reserve
                # safely; cleanup below carries the retired state objects, not
                # the reusable textual owner id.
                retired_states = self._retirement_closure_locked(owner_id)
        if retained_slot is not None:
            retained_slot.close()
        errors = self._cleanup_retired_states(retired_states)
        if errors:
            raise BaseExceptionGroup(
                "signal generation cleanup failed",
                list(errors),
            )

    def _freeze_one(
        self,
        state: _GenerationState,
    ) -> tuple[Mapping[str, SignalValue], str | None]:
        node = state.node
        slot = state.slot
        if node is None or slot is None:
            raise RuntimeError("live generation lost its producer slot")
        outputs = slot.freeze_live_outputs()
        if not isinstance(outputs, Mapping) or not outputs:
            raise ValueError("live output owner must return a non-empty mapping")
        declared = _declared_outputs(
            _require_signal_producer(node).dataset_output_declarations
        )
        if not set(outputs).issubset(declared):
            raise ValueError(
                "live publication contains an undeclared output"
            )
        by_bare = {
            bare: qualified
            for qualified, bare in state.bare_names.items()
        }
        frozen: dict[str, SignalValue] = {}
        run_record = _shared_run_record(outputs)
        for bare in declared:
            if bare not in outputs:
                continue
            output = outputs[bare]
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError("live output values must be LiveDatasetOutput")
            _require_published_declaration(bare, output, declared)
            qualified = by_bare[bare]
            frozen[qualified] = SignalValue(
                name=qualified,
                snapshot=output.snapshot,
                coverage=output.coverage,
                transient=True,
                run_record=run_record,
            )
        return (
            MappingProxyType(frozen),
            getattr(slot, "notification_failure", None),
        )

    def _build_front_locked(self) -> SignalFront:
        from .front import build_front

        return build_front(
            self._states,
            self._front_signals,
            self._front,
            self._resolved_direct_parents_locked,
        )

    def freeze(self) -> SignalFront:
        """Advance changed generations and return one immutable coherent front."""

        self._lane.drain_processors()
        with self._lock:
            if not self._dirty and not self._membership_changed:
                return self._front
            dirty = tuple(self._dirty)
            self._dirty.clear()
            states = {
                owner_id: self._states.get(owner_id)
                for owner_id in dirty
            }
            self._membership_changed = False

        for owner_id, state in states.items():
            if (
                state is None
                or state.retired
                or state.terminal
                or state.slot is None
            ):
                continue
            try:
                values, warning = self._freeze_one(state)
            except Exception as error:
                with self._lock:
                    if (
                        self._states.get(owner_id) is state
                        and not state.retired
                        and not state.terminal
                    ):
                        state.failure = f"{type(error).__name__}: {error}"
                        self._membership_changed = True
                continue
            try:
                with self._lock:
                    if (
                        self._states.get(owner_id) is state
                        and not state.retired
                        and not state.terminal
                    ):
                        self._publish_locked(state, values)
                        state.failure = warning
            except Exception as error:
                with self._lock:
                    if (
                        self._states.get(owner_id) is state
                        and not state.retired
                        and not state.terminal
                    ):
                        state.failure = f"{type(error).__name__}: {error}"
                        self._membership_changed = True

        with self._lock:
            latest = {
                name: publication
                for state in self._states.values()
                if not state.retired and state.publication is not None
                for name, publication in (
                    (name, state.publication)
                    for name in state.publication.signals
                )
            }
        self._lane.route(latest)
        with self._lock:
            front = self._build_front_locked()
            self._front = front
            self._membership_changed = False
            return front

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            states = tuple(self._states.values())
            self._states.clear()
            self._dirty.clear()
            self._front_signals = frozenset()
            self._request_owner_wake = None
            self._owner_wake_token = None
            self._front = SignalFront({}, {})
            self._publication_parents.clear()
        self._lane.close()
        errors = list(self._cleanup_retired_states(states))
        if errors:
            raise BaseExceptionGroup("signal data plane close failed", errors)

    def __len__(self) -> int:
        with self._lock:
            return sum(
                state.slot is not None and not state.retired
                for state in self._states.values()
            )
