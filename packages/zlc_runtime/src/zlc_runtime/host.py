"""Domain-neutral worker and processor hosting over the signal plane."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import threading
from typing import Protocol

from zlc_data import canonical_text

from ._public import RunHandleLike
from .dataset import DatasetCoverage, MonitorCoverage
from .dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
    LiveDatasetOutputOwner,
)
from .live_dataset import LiveDatasetPort, _ExactDeltaLivePort
from .owner_mailbox import RunOwnerMailbox
from .plane import SignalDataPlane, SignalPublication, SignalValue
from .preview import ExactDatasetPreviewSpec, LiveDatasetViewSpec
from .streams import FollowTap, StreamEndedEarly


__all__ = [
    "LogicNodeObservation",
    "Node",
    "NodeExecutionContext",
    "NodeHost",
]


_UNRESOLVED = object()


class _StartSuppressed(Exception):
    """Cancellation won before a worker operation started its work."""


class Node(Protocol):
    """Behavior object hosted according to whether it binds a source signal."""

    ...


@dataclass(frozen=True, slots=True)
class LogicNodeObservation:
    """Read-only lifecycle projection shared by workers and processors."""

    running: bool
    terminal: bool
    phase: str
    error: str | None = None
    warnings: tuple[str, ...] = ()
    run_snapshot: object | None = None

    def __post_init__(self) -> None:
        if type(self.running) is not bool or type(self.terminal) is not bool:
            raise TypeError("node observation flags must be bool")
        object.__setattr__(self, "phase", canonical_text(self.phase, "node observation phase"))
        if self.error is not None:
            object.__setattr__(self, "error", canonical_text(self.error, "node observation error"))
        warnings = tuple(self.warnings)
        if any(not isinstance(value, str) or not value.strip() for value in warnings):
            raise ValueError("node observation warnings must be non-empty text")
        object.__setattr__(self, "warnings", tuple(value.strip() for value in warnings))


class NodeExecutionContext:
    """What a worker operation is given: seven capabilities and its identity.

    ``generation`` is not a capability -- it performs nothing -- but a node that
    stamps its output has to know which run it is in, and the host is the only
    party that knows: it reserved the generation on the node's behalf.  Without
    it a hosted node either invents an identity the plane does not recognise or
    reserves a second one and collides with its own host.
    """

    __slots__ = ("_host",)

    def __init__(self, host: "NodeHost") -> None:
        self._host = host

    @property
    def generation(self) -> object:
        """The producer generation this run publishes into."""

        return self._host.generation

    def cancel_requested(self) -> bool:
        return self._host.cancel_requested

    def start_and_wait(self, starter: Callable[[], RunHandleLike]) -> object:
        return self._host._start_and_wait(starter)

    def attach_live_outputs(self, slot: object) -> None:
        """Attach one application-owned live output slot to this generation."""

        self._host._attach_live(slot)

    def open_live_dataset(
        self,
        spec: LiveDatasetViewSpec,
        *,
        output_owner: LiveDatasetOutputOwner,
        retain_on_terminal: bool = True,
    ) -> LiveDatasetPort:
        return self._host._open_live_dataset(
            spec,
            output_owner=output_owner,
            retain_on_terminal=retain_on_terminal,
        )

    def open_exact_dataset(
        self,
        spec: ExactDatasetPreviewSpec,
        *,
        projection: object,
    ) -> object:
        return self._host._open_exact_dataset(spec, projection=projection)

    def publish_final(
        self,
        outputs: Mapping[str, FinalDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        return self._host._publish_final(outputs)

    def warn(self, message: str) -> None:
        self._host._warn(message)


class NodeHost:
    """Host one worker or source-bound processor with one lifecycle surface."""

    def __init__(
        self,
        node: Node,
        data_plane: SignalDataPlane,
        request_owner_wake: Callable[[], None] | None = None,
        *,
        instance_id: str | None = None,
        dataset_output_declarations: Iterable[DatasetOutputDeclaration] | None = None,
        dataset_output_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        input_signal: str | None = None,
        signal_namer: Callable[[str, str], str] | None = None,
    ) -> None:
        if not callable(getattr(data_plane, "freeze", None)):
            raise TypeError("data_plane must provide the SignalDataPlane surface")
        if request_owner_wake is None:
            request_owner_wake = lambda: None
        if not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable")
        if signal_namer is None:
            signal_namer = lambda owner, name: f"@logic/{owner}/{name}"
        if not callable(signal_namer):
            raise TypeError("signal_namer must be callable")

        source_signal = self._resolve_source_signal(node, input_signal)
        mode = "processor" if source_signal is not None else "worker"

        identity = instance_id
        if identity is None:
            identity = getattr(node, "instance_id", None)
        if identity is None:
            raise ValueError("node instance_id is required")
        identity = canonical_text(identity, "node instance_id")

        declarations = self._resolve_declarations(
            node,
            identity,
            dataset_output_declarations,
            dataset_output_names if dataset_output_names is not None else output_names,
        )
        if mode == "processor" and not declarations:
            raise ValueError("processor must declare at least one Dataset output")

        self._node = node
        self._mode = mode
        self.instance_id = identity
        self._dataset_outputs = declarations
        self._source_signal = source_signal
        self._data_plane = data_plane
        self._request_owner_wake = request_owner_wake
        self._signal_namer = signal_namer
        self._execution_context = NodeExecutionContext(self)
        self._owner = (
            RunOwnerMailbox(
                request_owner_wake,
                thread_name_prefix=f"node-{identity}",
                max_workers=1,
            )
            if mode == "worker"
            else None
        )
        self._closed = False
        self._active = False
        self._terminal = False
        self._phase = "not started"
        self._error: str | None = None
        self._warnings: list[str] = []
        self._result: object = _UNRESOLVED
        self._handle: RunHandleLike | None = None
        self._snapshot: object | None = None
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()
        self._stop_reason = "Host requested stop"
        self._plane_state = False
        self._live_opened = False
        self._final_published = False
        self._processor_path: str | None = None
        self._source_publication: SignalPublication | None = None
        self._follow_tap: FollowTap[SignalPublication] | None = None

    @classmethod
    def create(cls, node: Node, **kwargs) -> "NodeHost":
        return cls(node, **kwargs)

    @staticmethod
    def _resolve_names(
        explicit: Iterable[str],
        fallback: Iterable[str],
        *,
        field: str,
    ) -> tuple[str, ...]:
        explicit_values = tuple(explicit)
        values = explicit_values if explicit_values else tuple(fallback)
        normalized = tuple(canonical_text(value, field) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{field}s must be unique")
        return normalized

    @classmethod
    def _resolve_declarations(
        cls,
        node: Node,
        instance_id: str,
        explicit: Iterable[DatasetOutputDeclaration] | None,
        names: Iterable[str] | None,
    ) -> tuple[DatasetOutputDeclaration, ...]:
        candidate = explicit
        if candidate is None:
            candidate = getattr(node, "dataset_output_declarations", None)
        if candidate is not None:
            values = tuple(candidate)
            if values and all(isinstance(value, DatasetOutputDeclaration) for value in values):
                if len({value.name for value in values}) != len(values):
                    raise ValueError("Dataset output declarations must be unique")
                return values
            if values and any(not isinstance(value, str) for value in values):
                raise TypeError("dataset_output_declarations must contain declarations")
            if values:
                names = values
            elif explicit is not None:
                names = () if names is None else names
        if names is None:
            names = getattr(node, "dataset_output_names", None)
        if names is None:
            names = getattr(node, "output_names", ())
        output_names = cls._resolve_names(names, (), field="Dataset output name")
        return tuple(
            DatasetOutputDeclaration(name, f"runtime.{instance_id}.{name}")
            for name in output_names
        )

    @staticmethod
    def _resolve_source_signal(node: Node, explicit: str | None) -> str | None:
        candidate: object = explicit
        if candidate is None:
            candidate = getattr(node, "input_signal", None)
        if candidate is None:
            candidate = getattr(node, "source_signal", None)
        if candidate is None:
            candidate = getattr(node, "input_signals", None)
        if candidate is None:
            return None
        if isinstance(candidate, Mapping):
            values = tuple(candidate.values())
        elif isinstance(candidate, (tuple, list, set, frozenset)):
            values = tuple(candidate)
        else:
            values = (candidate,)
        if len(values) != 1:
            raise ValueError("processor must declare exactly one input signal key")
        return canonical_text(values[0], "processor input signal")

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return self._dataset_outputs

    @property
    def node(self) -> Node:
        """The exact admitted node whose request this host is running."""

        return self._node

    @property
    def source_signal(self) -> str | None:
        return self._source_signal

    def signal_key(self, output_name: str) -> str:
        name = canonical_text(output_name, "node output name")
        if name not in {value.name for value in self._dataset_outputs}:
            raise KeyError(f"undeclared node output {name!r}")
        result = self._signal_namer(self.instance_id, name)
        return canonical_text(result, "signal key")

    def published_signals(self) -> tuple[str, ...]:
        return tuple(self.signal_key(value.name) for value in self._dataset_outputs)

    @property
    def running(self) -> bool:
        return self._active

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def handle(self) -> RunHandleLike | None:
        return self._handle

    @property
    def final_result(self) -> object | None:
        return None if self._result is _UNRESOLVED else self._result

    @property
    def final_result_resolved(self) -> bool:
        return self._result is not _UNRESOLVED

    @property
    def cancel_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def worker_idle(self) -> bool:
        return (
            True
            if self._owner is None
            else self._owner.worker_idle and self._owner.owner_reaped
        )

    @property
    def observation(self) -> LogicNodeObservation:
        return LogicNodeObservation(
            self._active,
            self._terminal,
            self._phase,
            self._error,
            tuple(self._warnings),
            self._snapshot,
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("NodeHost is closed")
        if self._active:
            return
        if self._owner is not None and not self.worker_idle:
            raise RuntimeError("previous node generation still has pending work")
        self._retire_plane_state()
        self._reset_generation()
        if self._mode == "processor":
            self._start_processor()
        else:
            self._start_worker()

    def _ensure_owner(self) -> RunOwnerMailbox:
        owner = self._owner
        if owner is None:
            owner = RunOwnerMailbox(
                self._request_owner_wake,
                thread_name_prefix=f"node-{self.instance_id}",
                max_workers=1,
            )
            self._owner = owner
        return owner

    def cancel(self, reason: str = "Host requested stop") -> None:
        if not self._active:
            return
        reason = canonical_text(reason, "cancellation reason")
        with self._start_lock:
            self._stop_reason = reason
            self._stop_event.set()
            handle = self._handle
        self._phase = "stopping"
        if self._mode == "processor" and self._processor_path == "latest":
            idle = self._data_plane.cancel_latest_only_processor(self)
            self._plane_state = False
            if idle and self._active:
                self.accept_processor_cancelled()
        elif self._mode == "processor":
            if self._follow_tap is not None:
                self._follow_tap.close()
            self._retire_plane_state()
        elif handle is not None:
            try:
                handle.cancel(reason)
            except BaseException as error:
                self._warnings.append(f"cancel warning: {type(error).__name__}: {error}")

    def poll(self) -> LogicNodeObservation:
        if self._mode == "worker":
            if self._active and self._handle is not None:
                self._snapshot = self._handle.snapshot()
                phase = getattr(self._snapshot, "phase", None)
                if isinstance(phase, str) and phase.strip():
                    self._phase = phase.strip()
            self._poll_worker()
        elif self._processor_path == "frozen":
            self._poll_frozen_processor()
        elif self._processor_path == "follow":
            self._poll_follow_processor()
        return self.observation

    def shutdown(self) -> None:
        if self._closed:
            return
        if self._active:
            self.cancel("Host is closing")
            self.poll()
            if self._active:
                raise RuntimeError("cannot close NodeHost before terminal")
        if (
            self._mode == "processor"
            and self._processor_path == "latest"
            and self._plane_state
        ):
            self._data_plane.withdraw_processor(self)
            self._plane_state = False
        elif self._plane_state:
            self._data_plane.detach_live(self)
            self._plane_state = False
        if self._owner is not None:
            self._owner.shutdown()
        self._closed = True

    @property
    def generation(self) -> object:
        """The producer generation reserved for the current run, if any."""

        return self._generation

    def _reset_generation(self) -> None:
        self._generation = None
        self._active = False
        self._terminal = False
        self._phase = "starting"
        self._error = None
        self._warnings.clear()
        self._result = _UNRESOLVED
        self._handle = None
        self._snapshot = None
        self._stop_event.clear()
        self._stop_reason = "Host requested stop"
        self._live_opened = False
        self._final_published = False
        self._processor_path = None
        self._source_publication = None
        self._follow_tap = None

    def _start_worker(self) -> None:
        assert self._owner is not None
        if self._dataset_outputs:
            self._generation = self._data_plane.reserve(self)
            self._plane_state = True
        self._active = True
        generation = self._owner.begin_generation()
        try:
            self._owner.submit(
                "execute",
                self._execute_worker,
                generation=generation,
            )
        except BaseException:
            self._active = False
            self._terminal = True
            self._phase = "failed"
            self._error = "worker could not be submitted"
            self._retire_plane_state()
            raise

    def _execute_worker(self) -> object:
        if self._stop_event.is_set():
            raise _StartSuppressed()
        execute = getattr(self._node, "execute", None)
        if not callable(execute):
            raise TypeError("worker must provide execute(ctx)")
        return execute(self._execution_context)

    def _poll_worker(self) -> None:
        assert self._owner is not None
        for completion in self._owner.drain_completions():
            if completion.generation != self._owner.generation:
                continue
            try:
                error = completion.future.exception()
            except BaseException as error:
                self._finish_worker_failure(error)
                self._owner.mark_owner_reaped()
                continue
            if error is None:
                result = completion.future.result()
                if self.cancel_requested:
                    self._finish_worker_cancelled()
                else:
                    self._result = result
                    self._finish_worker_success()
            else:
                self._finish_worker_failure(error)
            self._owner.mark_owner_reaped()

    def _finish_worker_cancelled(self) -> None:
        if self._live_opened:
            self._detach_plane_state()
        else:
            self._retire_plane_state()
        self._result = _UNRESOLVED
        self._phase = "cancelled"
        self._error = None
        self._active = False
        self._terminal = True

    def _finish_worker_success(self) -> None:
        if self._dataset_outputs and not self._final_published:
            if self._live_opened:
                try:
                    self._finish_live_plane_state()
                except BaseException as error:
                    self._finish_worker_failure(error)
                    return
            else:
                self._finish_worker_failure(
                    RuntimeError(
                        "node declared Dataset outputs but did not publish final outputs"
                    )
                )
                return
        elif self._live_opened:
            try:
                self._finish_live_plane_state()
            except BaseException as error:
                self._finish_worker_failure(error)
                return
        self._phase = "done"
        self._active = False
        self._terminal = True

    def _finish_worker_failure(self, error: BaseException) -> None:
        cancelled = isinstance(error, _StartSuppressed) or self.cancel_requested
        self._phase = "cancelled" if cancelled else "failed"
        self._error = None if cancelled else f"{type(error).__name__}: {error}"
        self._result = _UNRESOLVED
        self._active = False
        self._terminal = True
        self._retire_plane_state()

    def _warn(self, message: str) -> None:
        message = canonical_text(message, "node warning")
        if not self._active:
            raise RuntimeError("inactive node cannot publish a warning")
        with self._start_lock:
            self._warnings.append(message)
        self._request_owner_wake()

    def _start_and_wait(self, starter: Callable[[], RunHandleLike]) -> object:
        if not callable(starter):
            raise TypeError("Run starter must be callable")
        with self._start_lock:
            if self._stop_event.is_set():
                raise _StartSuppressed()
        handle = starter()
        if not all(callable(getattr(handle, name, None)) for name in ("snapshot", "cancel", "result")):
            raise TypeError("Run starter returned no RunHandleLike")
        with self._start_lock:
            self._handle = handle
            if self._owner is not None:
                self._owner.set_handle(handle)
            cancelled = self._stop_event.is_set()
            reason = self._stop_reason
        if cancelled:
            handle.cancel(reason)
        try:
            return handle.result()
        finally:
            self._snapshot = handle.snapshot()

    def _publish_final(
        self,
        outputs: Mapping[str, FinalDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        if self._mode != "worker":
            raise RuntimeError("processor outputs publish through their source publication")
        if not self._active or not self._plane_state:
            raise RuntimeError("node Dataset generation is not active")
        if self._final_published:
            raise RuntimeError("node final outputs were already published")
        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError("final outputs must be a non-empty mapping")
        if not set(outputs).issubset({value.name for value in self._dataset_outputs}):
            raise ValueError("final outputs contain an undeclared name")
        published = self._data_plane.publish_final(self, outputs)
        self._final_published = True
        return published

    def _attach_live(self, slot: object) -> None:
        if self._mode == "processor":
            raise RuntimeError("processor output is owned by the latest-only lane")
        if self._live_opened:
            raise RuntimeError("one Node generation may open one live Dataset")
        if not self._plane_state:
            raise RuntimeError("node Dataset generation is not reserved")
        if not callable(getattr(slot, "set_change_listener", None)):
            raise TypeError("live Dataset slot must support set_change_listener")
        if not callable(getattr(slot, "freeze_live_outputs", None)):
            raise TypeError("live Dataset slot must materialize typed outputs")
        try:
            slot.set_change_listener(lambda: self._data_plane.mark_changed(self, slot))
            self._data_plane.attach(self, slot)
        except BaseException:
            slot.close()
            raise
        self._live_opened = True

    def _open_live_dataset(
        self,
        spec: LiveDatasetViewSpec,
        *,
        output_owner: LiveDatasetOutputOwner,
        retain_on_terminal: bool,
    ) -> LiveDatasetPort:
        slot = LiveDatasetPort(
            spec,
            retain_on_terminal=retain_on_terminal,
            output_owner=output_owner,
        )
        self._attach_live(slot)
        return slot

    def _open_exact_dataset(self, spec: ExactDatasetPreviewSpec, *, projection: object) -> object:
        slot = _ExactDeltaLivePort(spec, projection)
        self._attach_live(slot)
        return slot

    def _retire_plane_state(self) -> None:
        if not self._plane_state:
            return
        if self._mode == "processor":
            self._data_plane.withdraw_processor(self)
        else:
            self._data_plane.retire(self)
        self._plane_state = False
        self._live_opened = False

    def _detach_plane_state(self) -> None:
        if not self._plane_state:
            return
        self._data_plane.detach_live(self)
        self._plane_state = False
        self._live_opened = False

    def _finish_live_plane_state(self) -> None:
        if not self._plane_state:
            return
        retained = self._data_plane.finish_live(self)
        self._plane_state = retained
        self._live_opened = False

    def _start_processor(self) -> None:
        assert self._source_signal is not None
        publication = self._data_plane.latest_publication(self._source_signal)
        if publication is None:
            self._phase = "failed"
            self._terminal = True
            self._error = f"processor input signal {self._source_signal!r} is not active"
            raise LookupError(self._error)
        source = publication.value(self._source_signal)
        if not isinstance(source, SignalValue):
            self._phase = "failed"
            self._terminal = True
            self._error = "processor publication lost its selected input signal"
            raise RuntimeError(self._error)
        if isinstance(source.coverage, MonitorCoverage):
            self._processor_path = "latest"
            self._start_latest_processor(publication)
            return
        if source.coverage is None:
            self._processor_path = "frozen"
            self._start_frozen_processor(publication, source)
            return
        if isinstance(source.coverage, DatasetCoverage):
            self._processor_path = "follow"
            self._start_follow_processor(publication, source)
            return
        raise TypeError("processor source has an unknown coverage type")

    def _start_latest_processor(self, publication: SignalPublication) -> None:
        assert self._source_signal is not None
        self.validate_processor_source(publication.value(self._source_signal))
        self._active = True
        try:
            self._data_plane.attach_latest_only_processor(
                self,
                source_name=self._source_signal,
                initial_publication=publication,
            )
            self._plane_state = True
            self._phase = "running"
        except BaseException as error:
            self._active = False
            self._terminal = True
            self._phase = "failed"
            self._error = f"{type(error).__name__}: {error}"
            raise

    def _start_frozen_processor(
        self,
        publication: SignalPublication,
        source: SignalValue,
    ) -> None:
        assert self._source_signal is not None
        if source.name != self._source_signal or source.coverage is not None:
            raise ValueError("frozen Processor requires its selected FINAL signal")
        self._data_plane.reserve_frozen_processor(
            self,
            source_name=self._source_signal,
            source_publication=publication,
        )
        self._plane_state = True
        self._source_publication = publication
        owner = self._ensure_owner()
        self._active = True
        self._phase = "running"
        generation = owner.begin_generation()
        try:
            owner.submit(
                "evaluate-frozen",
                self._evaluate_frozen_processor,
                generation=generation,
            )
        except BaseException as error:
            owner.mark_owner_reaped()
            self._active = False
            self._terminal = True
            self._phase = "failed"
            self._error = f"{type(error).__name__}: {error}"
            self._retire_plane_state()
            raise

    def _evaluate_frozen_processor(self) -> Mapping[str, LiveDatasetOutput]:
        if self._stop_event.is_set():
            raise _StartSuppressed()
        publication = self._source_publication
        source_name = self._source_signal
        if publication is None or source_name is None:
            raise RuntimeError("frozen Processor lost its exact source publication")
        source = publication.value(source_name)
        if source is None or source.coverage is not None:
            raise RuntimeError("frozen Processor source is not a FINAL signal")
        return self._evaluate_processor_outputs(source)

    def _poll_frozen_processor(self) -> None:
        owner = self._owner
        if owner is None:
            return
        for completion in owner.drain_completions():
            if completion.generation != owner.generation:
                continue
            try:
                error = completion.future.exception()
            except BaseException as error:
                self._finish_frozen_processor_failure(error)
                owner.mark_owner_reaped()
                continue
            if error is not None:
                self._finish_frozen_processor_failure(error)
                owner.mark_owner_reaped()
                continue
            outputs = completion.future.result()
            if self.cancel_requested:
                self._finish_frozen_processor_cancelled()
                owner.mark_owner_reaped()
                continue
            publication = self._source_publication
            if publication is None:
                self._finish_frozen_processor_failure(
                    RuntimeError("frozen Processor lost its exact source publication")
                )
                owner.mark_owner_reaped()
                continue
            try:
                self._data_plane.publish_terminal_processor(
                    self,
                    outputs,
                    source_publication=publication,
                )
            except BaseException as error:
                self._finish_frozen_processor_failure(error)
            else:
                self._active = False
                self._terminal = True
                self._phase = "done"
                self._error = None
            owner.mark_owner_reaped()

    def _finish_frozen_processor_cancelled(self) -> None:
        self._retire_plane_state()
        self._result = _UNRESOLVED
        self._active = False
        self._terminal = True
        self._phase = "cancelled"
        self._error = None

    def _finish_frozen_processor_failure(self, error: BaseException) -> None:
        if isinstance(error, _StartSuppressed) or self.cancel_requested:
            self._finish_frozen_processor_cancelled()
            return
        self._retire_plane_state()
        self._result = _UNRESOLVED
        self._active = False
        self._terminal = True
        self._phase = "failed"
        self._error = f"{type(error).__name__}: {error}"

    def _start_follow_processor(
        self,
        publication: SignalPublication,
        source: SignalValue,
    ) -> None:
        assert self._source_signal is not None
        self._validate_follow_source(source)
        tap = self._data_plane.reserve_follow_processor(
            self,
            source_name=self._source_signal,
            source_publication=publication,
        )
        self._plane_state = True
        self._source_publication = publication
        self._follow_tap = tap
        owner = self._ensure_owner()
        self._active = True
        self._phase = "running"
        generation = owner.begin_generation()
        try:
            owner.submit(
                "evaluate-follow",
                self._run_follow_processor,
                generation=generation,
            )
        except BaseException as error:
            tap.close()
            self._follow_tap = None
            owner.mark_owner_reaped()
            self._active = False
            self._terminal = True
            self._phase = "failed"
            self._error = f"{type(error).__name__}: {error}"
            self._retire_plane_state()
            raise

    def _validate_follow_source(self, source: SignalValue | None) -> SignalValue:
        if not isinstance(source, SignalValue):
            raise TypeError("Follow Processor input must be SignalValue")
        if source.name != self._source_signal:
            raise ValueError("Follow Processor received another input signal")
        if not isinstance(source.coverage, DatasetCoverage):
            raise ValueError("Follow Processor input must have DatasetCoverage")
        return source

    def _run_follow_processor(self) -> None:
        publication = self._source_publication
        tap = self._follow_tap
        source_name = self._source_signal
        if publication is None or tap is None or source_name is None:
            raise RuntimeError("Follow Processor lost its exact source binding")
        last_publication: SignalPublication | None = None
        last_outputs: Mapping[str, LiveDatasetOutput] | None = None
        try:
            while True:
                if self._stop_event.is_set():
                    raise _StartSuppressed()
                source = self._validate_follow_source(publication.value(source_name))
                outputs = self._evaluate_processor_outputs(source)
                if self._stop_event.is_set():
                    raise _StartSuppressed()
                self._data_plane.publish_processor(
                    self,
                    outputs,
                    source_publication=publication,
                )
                self._request_owner_wake()
                last_publication = publication
                last_outputs = outputs
                try:
                    publication = tap.next().payload
                except StreamEndedEarly:
                    if self._stop_event.is_set():
                        raise _StartSuppressed()
                    if last_publication is None or last_outputs is None:
                        raise RuntimeError(
                            "Follow Processor source ended without a publishable result"
                        )
                    self._data_plane.publish_terminal_processor(
                        self,
                        last_outputs,
                        source_publication=last_publication,
                    )
                    self._request_owner_wake()
                    return
        finally:
            tap.close()

    def _poll_follow_processor(self) -> None:
        owner = self._owner
        if owner is None:
            return
        for completion in owner.drain_completions():
            if completion.generation != owner.generation:
                continue
            self._follow_tap = None
            try:
                error = completion.future.exception()
            except BaseException as error:
                self._finish_follow_processor_failure(error)
                owner.mark_owner_reaped()
                continue
            if error is not None:
                self._finish_follow_processor_failure(error)
            elif self.cancel_requested:
                self._finish_follow_processor_cancelled()
            else:
                self._active = False
                self._terminal = True
                self._phase = "done"
                self._error = None
            owner.mark_owner_reaped()

    def _finish_follow_processor_cancelled(self) -> None:
        tap = self._follow_tap
        if tap is not None:
            tap.close()
            self._follow_tap = None
        self._retire_plane_state()
        self._result = _UNRESOLVED
        self._active = False
        self._terminal = True
        self._phase = "cancelled"
        self._error = None

    def _finish_follow_processor_failure(self, error: BaseException) -> None:
        if isinstance(error, _StartSuppressed) or self.cancel_requested:
            self._finish_follow_processor_cancelled()
            return
        tap = self._follow_tap
        if tap is not None:
            tap.close()
            self._follow_tap = None
        self._retire_plane_state()
        self._result = _UNRESOLVED
        self._active = False
        self._terminal = True
        self._phase = "failed"
        self._error = f"{type(error).__name__}: {error}"

    def validate_processor_source(self, source: SignalValue | None) -> None:
        """A processor subscribes to a LIVE signal, never to a finished one.

        Latest-only semantics means keeping only the signal's current value.
        Monitor coverage describes completeness of that visible value; it does
        not count values replaced before processing.  A finished measurement
        has no coverage because there is nothing left to keep up with --
        deriving from it is a one-shot computation, not a subscription.
        """

        if not isinstance(source, SignalValue):
            raise TypeError("processor input must be SignalValue")
        if source.name != self._source_signal:
            raise ValueError("processor received another input signal")
        if not isinstance(source.coverage, MonitorCoverage):
            raise ValueError(
                f"a processor subscribes to a live signal, but "
                f"{source.name!r} is a finished measurement (no monitor "
                f"coverage); derive from it directly instead of hosting a "
                f"processor on it"
            )

    def evaluate_processor(self, source: SignalValue) -> Mapping[str, LiveDatasetOutput]:
        self.validate_processor_source(source)
        return self._evaluate_processor_outputs(source)

    def _evaluate_processor_outputs(
        self,
        source: SignalValue,
    ) -> Mapping[str, LiveDatasetOutput]:
        evaluate = getattr(self._node, "evaluate", None)
        if not callable(evaluate):
            raise TypeError("processor must provide evaluate(SignalValue)")
        outputs = evaluate(source)
        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError("processor evaluate() must return a non-empty mapping")
        return dict(outputs)

    def accept_processor_result(
        self,
        source: SignalValue,
        source_publication: SignalPublication,
        outputs: Mapping[str, LiveDatasetOutput],
    ) -> None:
        if not self._active or self.cancel_requested:
            return
        self.validate_processor_source(source)
        self._data_plane.publish_processor(
            self,
            outputs,
            source_publication=source_publication,
        )

    def accept_processor_failure(self, error: Exception) -> None:
        if not self._active:
            return
        self._data_plane.withdraw_processor(self)
        self._plane_state = False
        self._active = False
        self._terminal = True
        self._phase = "failed"
        self._error = f"{type(error).__name__}: {error}"

    def accept_processor_cancelled(self) -> None:
        if not self._active:
            return
        self._plane_state = False
        self._active = False
        self._terminal = True
        self._phase = "cancelled"
        self._error = None

    def request_processor_owner_wake(self) -> None:
        self._request_owner_wake()
