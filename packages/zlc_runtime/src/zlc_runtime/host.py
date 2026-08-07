"""Domain-neutral finite and reactive node hosting over the signal plane."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import threading
from typing import Protocol, runtime_checkable

from zlc_data import canonical_text

from ._public import RunHandleLike
from .dataset import MonitorCoverage
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


__all__ = [
    "LogicNodeExecutionContext",
    "LogicNodeHost",
    "LogicNodeObservation",
    "Node",
    "NodeExecutionContext",
    "NodeHost",
]


_UNRESOLVED = object()


class _StartSuppressed(Exception):
    """Cancellation won before a finite operation started its work."""


@runtime_checkable
class Node(Protocol):
    """Minimal node identity; the selected kind supplies its one operation."""

    kind: str


@dataclass(frozen=True, slots=True)
class LogicNodeObservation:
    """Read-only lifecycle projection shared by finite and reactive nodes."""

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
    """What a finite node operation is given: six capabilities and its identity.

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
    """Host one finite or reactive node with a single lifecycle surface."""

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
        artifact_output_names: Iterable[str] = (),
        input_signal: str | None = None,
        signal_namer: Callable[[str, str], str] | None = None,
    ) -> None:
        if not isinstance(getattr(node, "kind", None), str):
            raise TypeError("node must declare kind")
        kind = canonical_text(node.kind, "node kind")
        if kind not in {"finite", "reactive"}:
            raise ValueError("node kind must be 'finite' or 'reactive'")
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
        artifact_names = self._resolve_names(
            artifact_output_names,
            getattr(node, "artifact_output_names", ()),
            field="artifact output name",
        )
        all_names = tuple(declaration.name for declaration in declarations) + artifact_names
        if len(set(all_names)) != len(all_names):
            raise ValueError("node output names must be unique")

        source_signal = self._resolve_source_signal(node, input_signal)
        if kind == "reactive" and source_signal is None:
            raise ValueError("reactive node requires exactly one input signal key")
        if kind == "reactive" and not declarations:
            raise ValueError("reactive node must declare at least one Dataset output")

        self._node = node
        self._kind = kind
        self.instance_id = identity
        self._dataset_outputs = declarations
        self._artifact_outputs = artifact_names
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
            if kind == "finite"
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
            raise ValueError("reactive node must declare exactly one input signal key")
        return canonical_text(values[0], "reactive input signal")

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return self._dataset_outputs

    @property
    def artifact_output_names(self) -> tuple[str, ...]:
        return self._artifact_outputs

    @property
    def source_signal(self) -> str | None:
        return self._source_signal

    def signal_key(self, output_name: str) -> str:
        name = canonical_text(output_name, "node output name")
        if name not in {
            *tuple(value.name for value in self._dataset_outputs),
            *self._artifact_outputs,
        }:
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
        if self._kind == "reactive":
            self._start_reactive()
        else:
            self._start_finite()

    def cancel(self, reason: str = "Host requested stop") -> None:
        if not self._active:
            return
        reason = canonical_text(reason, "cancellation reason")
        with self._start_lock:
            self._stop_reason = reason
            self._stop_event.set()
            handle = self._handle
        self._phase = "stopping"
        if self._kind == "reactive":
            idle = self._data_plane.cancel_latest_only_processor(self)
            self._plane_state = False
            if idle and self._active:
                self.accept_processor_cancelled()
        elif handle is not None:
            try:
                handle.cancel(reason)
            except BaseException as error:
                self._warnings.append(f"cancel warning: {type(error).__name__}: {error}")

    def poll(self) -> LogicNodeObservation:
        if self._kind == "finite":
            if self._active and self._handle is not None:
                self._snapshot = self._handle.snapshot()
                phase = getattr(self._snapshot, "phase", None)
                if isinstance(phase, str) and phase.strip():
                    self._phase = phase.strip()
            self._poll_finite()
        return self.observation

    def shutdown(self) -> None:
        if self._closed:
            return
        if self._active:
            self.cancel("Host is closing")
            self.poll()
            if self._active:
                raise RuntimeError("cannot close NodeHost before terminal")
        if self._kind == "reactive" and self._plane_state:
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

    def _start_finite(self) -> None:
        assert self._owner is not None
        if self._dataset_outputs:
            self._generation = self._data_plane.reserve(self)
            self._plane_state = True
        self._active = True
        generation = self._owner.begin_generation()
        try:
            self._owner.submit(
                "execute",
                self._execute_finite,
                generation=generation,
            )
        except BaseException:
            self._active = False
            self._terminal = True
            self._phase = "failed"
            self._error = "finite node could not be submitted"
            self._retire_plane_state()
            raise

    def _execute_finite(self) -> object:
        if self._stop_event.is_set():
            raise _StartSuppressed()
        execute = getattr(self._node, "execute", None)
        if not callable(execute):
            raise TypeError("finite node must provide execute(ctx)")
        return execute(self._execution_context)

    def _poll_finite(self) -> None:
        assert self._owner is not None
        for completion in self._owner.drain_completions():
            if completion.generation != self._owner.generation:
                continue
            try:
                error = completion.future.exception()
            except BaseException as error:
                self._finish_finite_failure(error)
                self._owner.mark_owner_reaped()
                continue
            if error is None:
                self._result = completion.future.result()
                self._finish_finite_success()
            else:
                self._finish_finite_failure(error)
            self._owner.mark_owner_reaped()

    def _finish_finite_success(self) -> None:
        if self._dataset_outputs and not self._final_published:
            if self._live_opened:
                self._detach_plane_state()
            else:
                self._finish_finite_failure(
                    RuntimeError(
                        "node declared Dataset outputs but did not publish final outputs"
                    )
                )
                return
        elif self._live_opened:
            self._detach_plane_state()
        self._phase = "done"
        self._active = False
        self._terminal = True

    def _finish_finite_failure(self, error: BaseException) -> None:
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
        if self._kind != "finite":
            raise RuntimeError("reactive outputs publish through their source publication")
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
        if self._kind == "reactive":
            raise RuntimeError("reactive output is owned by the latest-only lane")
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
        if self._kind == "reactive":
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

    def _start_reactive(self) -> None:
        assert self._source_signal is not None
        publication = self._data_plane.latest_publication(self._source_signal)
        if publication is None:
            self._phase = "failed"
            self._terminal = True
            self._error = f"reactive input signal {self._source_signal!r} is not active"
            raise LookupError(self._error)
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

    def validate_processor_source(self, source: SignalValue | None) -> None:
        """A reactive node subscribes to a LIVE signal, never to a finished one.

        Latest-only semantics means keeping up with a signal that is still
        moving, and skipping values honestly when it cannot: that is what
        monitor coverage records.  A finished measurement has no coverage
        because there is nothing left to keep up with -- deriving from it is a
        one-shot computation, not a subscription.
        """

        if not isinstance(source, SignalValue):
            raise TypeError("reactive input must be SignalValue")
        if source.name != self._source_signal:
            raise ValueError("reactive node received another input signal")
        if not isinstance(source.coverage, MonitorCoverage):
            raise ValueError(
                f"a reactive node subscribes to a live signal, but "
                f"{source.name!r} is a finished measurement (no monitor "
                f"coverage); derive from it directly instead of hosting a "
                f"reactive node on it"
            )

    def evaluate_processor(self, source: SignalValue) -> Mapping[str, LiveDatasetOutput]:
        self.validate_processor_source(source)
        evaluate = getattr(self._node, "evaluate", None)
        if not callable(evaluate):
            raise TypeError("reactive node must provide evaluate(SignalValue)")
        outputs = evaluate(source)
        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError("reactive node evaluate() must return a non-empty mapping")
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


LogicNodeHost = NodeHost
LogicNodeExecutionContext = NodeExecutionContext
