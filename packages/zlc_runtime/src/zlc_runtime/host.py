"""Domain-neutral worker and processor hosting over the signal plane."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import threading

from zlc_data import OwnedSnapshot, canonical_text

from .dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from .owner_mailbox import RunOwnerMailbox
from .plane import SignalDataPlane, SignalPublication, SignalValue
from .streams import FollowTap, StreamEndedEarly
from .task_run import TaskArtifact, TaskRun


__all__ = [
    "LogicNodeObservation",
    "NodeExecutionContext",
    "NodeHost",
    "NodeProgress",
    "TaskArtifact",
    "TaskRun",
]


_UNRESOLVED = object()
_WORKER_KINDS = frozenset(("measurement", "task"))
_NODE_KINDS = _WORKER_KINDS | {"processor"}
_INPUT_DELIVERIES = frozenset(("exact", "latest"))


class _StartSuppressed(Exception):
    """Cancellation won before a worker operation started its work."""


@dataclass(frozen=True, slots=True)
class NodeProgress:
    """One worker-owned progress fact, independent of any UI toolkit."""

    message: str
    current: int | None = None
    total: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", canonical_text(self.message, "node progress message"))
        if (self.current is None) != (self.total is None):
            raise ValueError("node progress current and total must be supplied together")
        if self.current is None:
            return
        current = int(self.current)
        total = int(self.total)
        if total <= 0 or current < 0 or current > total:
            raise ValueError("node progress must satisfy 0 <= current <= total")
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "total", total)

    @property
    def text(self) -> str:
        if self.current is None:
            return self.message
        return f"{self.message} {self.current}/{self.total}"


@dataclass(frozen=True, slots=True)
class LogicNodeObservation:
    """Read-only lifecycle projection shared by workers and processors."""

    running: bool
    terminal: bool
    phase: str
    error: str | None = None
    progress: NodeProgress | None = None

    def __post_init__(self) -> None:
        if type(self.running) is not bool or type(self.terminal) is not bool:
            raise TypeError("node observation flags must be bool")
        object.__setattr__(self, "phase", canonical_text(self.phase, "node observation phase"))
        if self.error is not None:
            object.__setattr__(self, "error", canonical_text(self.error, "node observation error"))
        if self.progress is not None and not isinstance(self.progress, NodeProgress):
            raise TypeError("node observation progress must be NodeProgress or None")


class NodeExecutionContext:
    """The runtime capabilities and generation identity given to one worker.

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

    @property
    def instance_id(self) -> str:
        """The descriptor-owned instance identity of this exact run."""

        return self._host.instance_id

    def cancel_requested(self) -> bool:
        return self._host.cancel_requested

    def seal_terminal(
        self,
        *,
        accept_stop: bool = False,
        partial: bool = False,
    ) -> None:
        """Atomically enter a worker's non-cancellable terminal commit.

        A Stop that was accepted first rejects the seal.  Once the seal
        returns, later Stop requests cannot relabel the worker's terminal
        side effects as cancellation; an exception still ends as failure.
        """

        if type(accept_stop) is not bool:
            raise TypeError("accept_stop must be bool")
        if type(partial) is not bool:
            raise TypeError("partial must be bool")
        self._host._seal_worker_terminal(
            accept_stop=accept_stop,
            partial=partial,
        )

    def commit_live(
        self,
        outputs: Mapping[str, LiveDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        """Commit one new immutable event bundle into this run's datasets."""

        return self._host._commit_live(outputs)

    def current_dataset(
        self,
        output_name: str,
        publication: SignalPublication | None = None,
    ) -> OwnedSnapshot:
        """Materialize one of this node's declared output prefixes."""

        return self._host._data_plane.current_dataset(
            self._host.signal_key(output_name),
            publication,
        )

    def report_progress(
        self,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        """Replace this run's current progress fact and wake its owner."""

        self._host._report_progress(NodeProgress(message, current, total))

    @property
    def run_directory(self) -> Path:
        """The directory durably allocated for this hosted Task run."""

        directory = self._host.run_directory
        if directory is None:
            raise RuntimeError("only a hosted Task has a run directory")
        return directory

    def register_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        role: str,
        contract_id: str = "",
    ) -> TaskArtifact:
        """Register one already-complete, run-contained domain artifact."""

        return self._host._register_task_artifact(
            name,
            path,
            role=role,
            contract_id=contract_id,
        )

class NodeHost:
    """Host one worker or source-bound processor with one lifecycle surface."""

    def __init__(
        self,
        node: object,
        data_plane: SignalDataPlane,
        request_owner_wake: Callable[[], None] | None = None,
        *,
        instance_id: str,
        kind: str,
        dataset_output_declarations: Iterable[DatasetOutputDeclaration],
        input_signal: str | None = None,
        input_delivery: str | None = None,
        required_artifacts: Mapping[str, str] | None = None,
        task_name: str | None = None,
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

        identity = canonical_text(instance_id, "node instance_id")
        normalized_kind = canonical_text(
            str(getattr(kind, "value", kind)),
            "node kind",
        )
        if normalized_kind not in _NODE_KINDS:
            raise ValueError(f"node kind must be one of {tuple(sorted(_NODE_KINDS))}")
        mode = "processor" if normalized_kind == "processor" else "worker"

        declarations = tuple(dataset_output_declarations)
        if any(
            not isinstance(value, DatasetOutputDeclaration)
            for value in declarations
        ):
            raise TypeError(
                "dataset_output_declarations must contain DatasetOutputDeclaration values"
            )
        if len({value.name for value in declarations}) != len(declarations):
            raise ValueError("Dataset output declarations must be unique")
        if mode == "processor" and not declarations:
            raise ValueError("processor must declare at least one Dataset output")

        source_signal = (
            None
            if input_signal is None
            else canonical_text(input_signal, "processor input signal")
        )
        delivery = (
            None
            if input_delivery is None
            else canonical_text(
                str(getattr(input_delivery, "value", input_delivery)),
                "processor input delivery",
            )
        )
        if mode == "processor":
            if source_signal is None:
                raise ValueError("processor input_signal is required")
            if delivery not in _INPUT_DELIVERIES:
                raise ValueError(
                    "processor input_delivery must be 'exact' or 'latest'"
                )
        elif (source_signal is None) != (delivery is None):
            raise ValueError(
                "worker input_signal and input_delivery must be supplied together"
            )
        elif delivery is not None and delivery not in _INPUT_DELIVERIES:
            raise ValueError("worker input_delivery must be 'exact' or 'latest'")

        if required_artifacts is None:
            required_artifacts = {}
        if not isinstance(required_artifacts, Mapping):
            raise TypeError("required_artifacts must be a mapping")
        artifacts = {
            canonical_text(name, "required artifact name"): canonical_text(
                contract_id,
                "required artifact contract",
            )
            for name, contract_id in required_artifacts.items()
        }
        selected_task_name = (
            identity
            if task_name is None
            else canonical_text(task_name, "Task run name")
        )

        self._node = node
        self._mode = mode
        self._kind = normalized_kind
        self.instance_id = identity
        self._dataset_outputs = declarations
        self._source_signal = source_signal
        self._input_delivery = delivery
        self._required_artifacts = artifacts
        self._task_name = selected_task_name
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
        self._progress: NodeProgress | None = None
        self._result: object = _UNRESOLVED
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()
        self._worker_stop_sealed = False
        self._worker_stop_accepted = False
        self._worker_partial_seal = False
        self._stop_reason = "Host requested stop"
        self._plane_state = False
        self._live_commit_count = 0
        self._progress_reported = False
        self._processor_path: str | None = None
        self._source_publication: SignalPublication | None = None
        self._terminal_source: SignalValue | None = None
        self._follow_tap: FollowTap[SignalPublication] | None = None
        self._task_run: TaskRun | None = None

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return self._dataset_outputs

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
    def final_result(self) -> object | None:
        return None if self._result is _UNRESOLVED else self._result

    @property
    def final_result_resolved(self) -> bool:
        return self._result is not _UNRESOLVED

    @property
    def run_directory(self) -> Path | None:
        run = self._task_run
        return None if run is None else run.directory

    @property
    def artifacts(self) -> tuple[TaskArtifact, ...]:
        run = self._task_run
        return () if run is None else run.artifacts

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
            self._progress,
        )

    def start(
        self,
        *,
        run_root: str | Path | None = None,
        input_summary: Mapping[str, object] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("NodeHost is closed")
        if self._active:
            return
        if self._owner is not None and not self.worker_idle:
            raise RuntimeError("previous node generation still has pending work")
        self._retire_plane_state()
        self._reset_generation()
        if self._mode == "processor":
            if run_root is not None or input_summary is not None:
                raise ValueError("only a Task start accepts run metadata")
            self._start_processor()
        else:
            if self._kind == "task":
                if run_root is None or input_summary is None:
                    raise ValueError("a hosted Task start requires run_root and input_summary")
                self._task_run = TaskRun.create(
                    run_root,
                    task_name=self._task_name,
                    instance_id=self.instance_id,
                    input_summary=input_summary,
                )
            elif run_root is not None or input_summary is not None:
                raise ValueError("only a Task start accepts run metadata")
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
            if self._mode == "worker" and self._worker_stop_sealed:
                return
            self._stop_reason = reason
            self._stop_event.set()
        self._phase = "stopping"
        if self._task_run is not None:
            try:
                self._task_run.mark_stopping(reason)
            except BaseException as record_error:
                self._error = (
                    "TaskRun stopping record failed: "
                    f"{type(record_error).__name__}: {record_error}"
                )
        if self._mode == "processor" and self._processor_path == "latest":
            idle = self._data_plane.cancel_latest_only_processor(self)
            self._plane_state = False
            if idle and self._active:
                self.accept_processor_cancelled()
        elif self._mode == "processor":
            if self._follow_tap is not None:
                self._follow_tap.close()
            self._retire_plane_state()

    def poll(self) -> LogicNodeObservation:
        if self._mode == "worker":
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
        self._progress = None
        self._result = _UNRESOLVED
        self._stop_event.clear()
        self._worker_stop_sealed = False
        self._worker_stop_accepted = False
        self._worker_partial_seal = False
        self._stop_reason = "Host requested stop"
        self._live_commit_count = 0
        self._progress_reported = False
        self._processor_path = None
        self._source_publication = None
        self._terminal_source = None
        self._follow_tap = None
        self._task_run = None

    def _start_worker(self) -> None:
        assert self._owner is not None
        if self._dataset_outputs:
            self._generation = self._data_plane.begin_generation(self)
            self._plane_state = True
        self._active = True
        generation = self._owner.begin_generation()
        try:
            if self._task_run is not None:
                self._task_run.mark_running()
            self._owner.submit(
                "execute",
                self._execute_worker,
                generation=generation,
            )
            self._phase = "running"
        except BaseException as error:
            self._active = False
            self._terminal = True
            self._phase = "failed"
            self._error = f"{type(error).__name__}: {error}"
            self._mark_task_run_failed(error)
            self._retire_plane_state()
            raise

    def _execute_worker(self) -> object:
        try:
            if self._stop_event.is_set():
                raise _StartSuppressed()
            execute = getattr(self._node, "execute", None)
            if not callable(execute):
                raise TypeError("worker must provide execute(ctx)")
            return execute(self._execution_context)
        finally:
            with self._start_lock:
                self._worker_stop_sealed = True

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
                if self.cancel_requested and not self._worker_stop_accepted:
                    self._finish_worker_cancelled()
                else:
                    self._finish_worker_success(result)
            else:
                self._finish_worker_failure(error)
            self._owner.mark_owner_reaped()

    def _finish_worker_cancelled(self) -> None:
        self._end_run("cancelled", None)

    def _end_run(self, phase: str, error: BaseException | None) -> None:
        """How a run that did not succeed leaves the bench.

        Cancelled and failed differ in one thing, and it is the thing that
        matters to whoever was watching: stopping a measurement ENDS it, it
        does not unmeasure it, so what the run published is kept -- partial,
        and said to be partial by its own coverage.  A failure stands behind
        nothing, so its generation is withdrawn.

        Both endings were written out separately, and the copy that runs when
        the worker was interrupted MID-STEP -- the common case, since Stop
        raises out of whatever the node was doing -- named the run "cancelled"
        and then disposed of it as a failure.  A stopped scan therefore
        vanished from the bench, taking the panel watching it, its Edit
        snapshot and its Save with it.
        """

        kept = False
        terminal_phase = phase
        terminal_exception = error
        terminal_error_text = (
            None if error is None else f"{type(error).__name__}: {error}"
        )
        if phase == "cancelled" and self._live_commit_count:
            try:
                self._seal_committed_plane_state(cut_short=True)
                kept = self._plane_state
            except BaseException as seal_error:
                kept = False
                terminal_phase = "failed"
                terminal_exception = RuntimeError(
                    "stopped partial Dataset could not be sealed: "
                    f"{type(seal_error).__name__}: {seal_error}"
                )
                terminal_error_text = str(terminal_exception)
        if not kept:
            self._retire_plane_state()
        self._result = _UNRESOLVED
        self._phase = terminal_phase
        self._error = terminal_error_text
        self._progress = None
        self._active = False
        self._terminal = True
        if self._task_run is not None:
            try:
                if terminal_phase == "cancelled":
                    self._task_run.mark_stopped()
                else:
                    assert terminal_exception is not None
                    self._task_run.mark_failed(terminal_exception)
            except BaseException as record_error:
                self._phase = "failed"
                self._error = (
                    "TaskRun terminal record failed: "
                    f"{type(record_error).__name__}: {record_error}"
                )

    def _finish_worker_success(self, result: object) -> None:
        try:
            self._validate_worker_terminal_contract(result)
            if self._live_commit_count:
                self._seal_committed_plane_state(
                    cut_short=self._worker_partial_seal
                )
            if self._task_run is not None:
                if self._worker_stop_accepted:
                    self._task_run.mark_stopped()
                else:
                    self._task_run.mark_completed()
        except BaseException as error:
            self._finish_worker_failure(error)
            return
        self._result = result
        self._phase = "done"
        self._error = None
        self._progress = None
        self._active = False
        self._terminal = True

    def _finish_worker_failure(self, error: BaseException) -> None:
        # An error raised out of a run the operator STOPPED is the stop: the
        # node was interrupted in the middle of a step, which is what Stop
        # does.  It ends the same way a stop caught between steps ends.
        if isinstance(error, _StartSuppressed) or (
            self.cancel_requested and not self._worker_stop_accepted
        ):
            self._end_run("cancelled", None)
            return
        self._end_run("failed", error)

    def _seal_worker_terminal(
        self,
        *,
        accept_stop: bool = False,
        partial: bool = False,
    ) -> None:
        with self._start_lock:
            if self._mode != "worker" or not self._active:
                raise RuntimeError("only an active worker can seal its terminal commit")
            stopped = self._stop_event.is_set()
            if stopped and not accept_stop:
                raise _StartSuppressed()
            self._worker_stop_accepted = stopped and accept_stop
            self._worker_partial_seal = partial
            self._worker_stop_sealed = True

    def _commit_live(
        self,
        outputs: Mapping[str, LiveDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        if self._mode != "worker":
            raise RuntimeError("processor outputs publish through their source publication")
        if not self._active or not self._plane_state:
            raise RuntimeError("node Dataset generation is not active")
        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError("live outputs must be a non-empty mapping")
        values = dict(outputs)
        if any(not isinstance(value, LiveDatasetOutput) for value in values.values()):
            raise TypeError("live output values must be LiveDatasetOutput")
        declared = {value.name for value in self._dataset_outputs}
        if not set(values).issubset(declared):
            raise ValueError("live outputs contain an undeclared name")
        published = self._data_plane.commit_live(self, values)
        with self._start_lock:
            self._live_commit_count += 1
        self._request_owner_wake()
        return published

    def _validate_worker_terminal_contract(self, result: object) -> None:
        declared = {value.name for value in self._dataset_outputs}
        if declared and not self._live_commit_count:
            raise RuntimeError(
                f"hosted {self._kind} finished without a live Dataset commit"
            )
        if self._kind == "task" and not self._progress_reported:
            raise RuntimeError("hosted Task finished without reporting progress")
        for name, contract_id in self._required_artifacts.items():
            run = self._task_run
            artifact = None if run is None else run.artifact(name)
            if artifact is None:
                raise RuntimeError(
                    f"Task did not register required final artifact {name!r}"
                )
            if artifact.role != "final":
                raise RuntimeError(
                    f"required artifact {name!r} must be registered as final"
                )
            if artifact.contract_id != contract_id:
                raise RuntimeError(
                    f"required artifact {name!r} contract differs from {contract_id!r}"
                )
            if not artifact.path.is_file():
                raise FileNotFoundError(
                    f"registered final artifact {name!r} is missing: {artifact.path}"
                )
            if artifact.path.stat().st_size != artifact.size_bytes:
                raise RuntimeError(
                    f"registered final artifact {name!r} changed after registration"
                )

    def _register_task_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        role: str,
        contract_id: str,
    ) -> TaskArtifact:
        run = self._task_run
        if self._kind != "task" or run is None or not self._active:
            raise RuntimeError("only an active hosted Task can register artifacts")
        return run.register_artifact(
            name,
            path,
            role=role,
            contract_id=contract_id,
        )

    def _mark_task_run_failed(self, error: BaseException) -> None:
        run = self._task_run
        if run is None:
            return
        try:
            run.mark_failed(error)
        except BaseException as record_error:
            self._error = (
                f"{self._error}; TaskRun failure record failed: "
                f"{type(record_error).__name__}: {record_error}"
            )

    def _retire_plane_state(self) -> None:
        if not self._plane_state:
            return
        if self._mode == "processor":
            self._data_plane.withdraw_processor(self)
        else:
            self._data_plane.retire(self)
        self._plane_state = False

    def _seal_committed_plane_state(self, *, cut_short: bool = False) -> None:
        if not self._plane_state:
            return
        retained = self._data_plane.seal_committed(
            self,
            cut_short=cut_short,
        )
        self._plane_state = retained

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
        if not self._data_plane.is_generation_live(self._source_signal):
            snapshot = self._data_plane.current_dataset(
                self._source_signal,
                publication,
            )
            self._terminal_source = SignalValue(
                self._source_signal,
                snapshot,
                None,
                run_record=publication.run_record,
                primary_index=source.primary_index,
            )
            self._processor_path = "frozen"
            self._start_frozen_processor(publication, self._terminal_source)
            return
        if self._input_delivery == "latest":
            self._processor_path = "latest"
            self._start_latest_processor(publication)
            return
        if self._input_delivery == "exact":
            self._processor_path = "follow"
            self._start_follow_processor(publication, source)
            return
        raise RuntimeError("processor input delivery is not configured")

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
        if source.name != self._source_signal:
            raise ValueError("frozen Processor received another input signal")
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
        source = self._terminal_source
        if publication is None or source is None:
            raise RuntimeError("frozen Processor lost its exact source publication")
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
                self._data_plane.commit_processor(
                    self,
                    outputs,
                    source_publication=publication,
                    retain=True,
                )
                self._data_plane.seal_processor(self)
            except BaseException as error:
                self._finish_frozen_processor_failure(error)
            else:
                self._active = False
                self._terminal = True
                self._phase = "done"
                self._error = None
                self._progress = None
            owner.mark_owner_reaped()

    def _finish_frozen_processor_cancelled(self) -> None:
        self._retire_plane_state()
        self._result = _UNRESOLVED
        self._active = False
        self._terminal = True
        self._phase = "cancelled"
        self._error = None
        self._progress = None

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
        self._progress = None

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
        return source

    def _run_follow_processor(self) -> None:
        tap = self._follow_tap
        source_name = self._source_signal
        if tap is None or source_name is None:
            raise RuntimeError("Follow Processor lost its exact source binding")
        last_publication: SignalPublication | None = None
        try:
            while True:
                if self._stop_event.is_set():
                    raise _StartSuppressed()
                try:
                    publication = tap.next()
                except StreamEndedEarly:
                    if self._stop_event.is_set():
                        raise _StartSuppressed()
                    if last_publication is None:
                        raise RuntimeError(
                            "Follow Processor source ended without a publishable result"
                        )
                    self._data_plane.seal_processor(self)
                    self._request_owner_wake()
                    return
                source = self._validate_follow_source(publication.value(source_name))
                outputs = self._evaluate_processor_outputs(source)
                if self._stop_event.is_set():
                    raise _StartSuppressed()
                self._data_plane.commit_processor(
                    self,
                    outputs,
                    source_publication=publication,
                )
                self._request_owner_wake()
                last_publication = publication
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
                self._progress = None
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
        self._progress = None

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
        self._progress = None

    def validate_processor_source(self, source: SignalValue | None) -> None:
        """Validate identity only; the descriptor owns exact/latest delivery."""

        if not isinstance(source, SignalValue):
            raise TypeError("processor input must be SignalValue")
        if source.name != self._source_signal:
            raise ValueError("processor received another input signal")

    def evaluate_processor(
        self,
        source: SignalValue,
        _source_publication: SignalPublication,
    ) -> Mapping[str, LiveDatasetOutput]:
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
        self._data_plane.commit_processor(
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
        self._progress = None

    def accept_processor_cancelled(self) -> None:
        if not self._active:
            return
        self._plane_state = False
        self._active = False
        self._terminal = True
        self._phase = "cancelled"
        self._error = None
        self._progress = None

    def request_processor_owner_wake(self) -> None:
        self._request_owner_wake()

    def _report_progress(self, progress: NodeProgress) -> None:
        if not isinstance(progress, NodeProgress):
            raise TypeError("progress must be NodeProgress")
        if not self._active:
            raise RuntimeError("inactive node cannot report progress")
        if self._task_run is not None:
            self._task_run.report_progress(
                progress.message,
                current=progress.current,
                total=progress.total,
            )
        with self._start_lock:
            self._progress = progress
            self._progress_reported = True
        self._request_owner_wake()
