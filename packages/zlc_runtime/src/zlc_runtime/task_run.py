"""Crash-honest durable records for hosted Task runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from threading import RLock
import traceback
from typing import TypeVar

from zlc_durable import unique_path, write_readable_json


__all__ = ["TaskArtifact", "TaskRun"]


_Result = TypeVar("_Result")


_TERMINAL_STATES = frozenset(("completed", "stopped", "failed"))
_ARTIFACT_ROLES = frozenset(
    ("checkpoint", "process", "final", "figure", "summary", "preview")
)
#: The two files a run writes about itself, each created once and never
#: replaced: the start record when the run begins, the run record when it
#: is over.  Neither is a Task artifact.
_START_RECORD = "start.json"
_RUN_RECORD = "run.json"


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _text(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty text")
    return result


def _plain_input(value: object, path: str) -> object:
    """Normalize one Task input into the plain JSON the run's records hold.

    What the records cannot hold -- a non-finite float, a non-text key, an
    arbitrary object -- is refused here, at the input, so that ``create``
    can refuse a run before its directory exists and the start record never
    meets a surprise at the moment the run begins.
    """

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path} keys must be text")
        return {
            key: _plain_input(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _plain_input(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class TaskArtifact:
    """One complete file explicitly selected by the domain Task."""

    name: str
    path: Path
    relative_path: str
    role: str
    contract_id: str = ""
    size_bytes: int = 0

class TaskRun:
    """The durable lifecycle and explicit artifact index of one Task run.

    A run writes two files about itself, each created once and never
    replaced.  ``start.json`` is written when the run begins -- its identity,
    its normalized input, when it started -- so a process that dies mid-run
    still leaves what the run WAS.  ``run.json`` is written when the run is
    over: how it ended, what it registered, what went wrong.  Nothing is
    written in between: progress and artifact registration stay in the
    process.  A record rewritten on every one of them is, for a two-hundred-
    repeat calibration, two hundred fsyncs and two hundred ``os.replace``
    calls over a path something else may hold open -- on Windows, a
    PermissionError waiting for the last write of a long run -- and nothing
    reads it while the run is going.  Liveness belongs to the process that
    has it; a "running" left in a file by a process that has since died is
    not stale information, it is false information.  So a run directory
    with a start record and no run record is a run that did not finish,
    which is exactly what it means.

    This owner never inspects or saves scientific data: a domain Task writes
    a complete chosen file and then registers that file here.
    """

    def __init__(
        self,
        directory: Path,
        *,
        task_name: str,
        instance_id: str,
        input_summary: Mapping[str, object],
    ) -> None:
        self.directory = directory.resolve()
        self.task_name = _text(task_name, "task name")
        self.instance_id = _text(instance_id, "task instance_id")
        frozen_input = _plain_input(input_summary, "Task input")
        assert isinstance(frozen_input, dict)
        self._input = frozen_input
        self._started_at = _now()
        self._state = "starting"
        self._progress: dict[str, object] | None = None
        self._stop_reason: str | None = None
        self._ended_at: str | None = None
        self._error: dict[str, object] | None = None
        self._artifacts: dict[str, TaskArtifact] = {}
        self._lock = RLock()

    @classmethod
    def create(
        cls,
        run_root: str | Path,
        *,
        task_name: str,
        instance_id: str,
        input_summary: Mapping[str, object],
    ) -> "TaskRun":
        """Allocate a run directory for input the run's records can hold.

        The input is normalized before the directory is taken: a run refused
        for its input must not leave an empty run directory behind, and the
        refusal belongs at Start, not at the start record's write.
        """

        if not isinstance(input_summary, Mapping):
            raise TypeError("Task input summary must be a mapping")
        name = _text(task_name, "task name")
        identity = _text(instance_id, "task instance_id")
        plain = _plain_input(input_summary, "Task input")
        root = Path(run_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Task run root does not exist: {root}")
        selected = unique_path(root, name, "")
        return cls(
            selected,
            task_name=name,
            instance_id=identity,
            input_summary=plain,
        )

    @property
    def artifacts(self) -> tuple[TaskArtifact, ...]:
        with self._lock:
            return tuple(self._artifacts.values())

    def artifact(self, name: str) -> TaskArtifact | None:
        with self._lock:
            return self._artifacts.get(str(name))

    def mark_running(self) -> None:
        """Begin the run: write its start record, once, then run.

        The record is written before the state changes, so a record that
        could not be written is a run that did not begin: the caller ends
        it as failed with the write error, and nothing irreversible has
        happened yet.
        """

        with self._lock:
            self._require("running", allowed=("starting",))
            write_readable_json(
                self.directory / _START_RECORD,
                {
                    "schema": "zlc.task-start",
                    "run_id": self.directory.name,
                    "task": {
                        "api_name": self.task_name,
                        "instance_id": self.instance_id,
                    },
                    "input": self._input,
                    "started_at": self._started_at,
                },
            )
            self._state = "running"

    def mark_stopping(self, reason: str) -> None:
        with self._lock:
            if self._state in _TERMINAL_STATES:
                return
            if self._state not in {"starting", "running", "stopping"}:
                raise RuntimeError(f"Task run cannot stop from {self._state}")
            self._state = "stopping"
            self._stop_reason = _text(reason, "Task stop reason")

    def mark_completed(self) -> None:
        with self._lock:
            self._require("completed", allowed=("running",))
            self._end_locked("completed", None)

    def mark_stopped(self, error: BaseException | None = None) -> None:
        """End the run as stopped, recording what the ending itself could not do.

        Stopping is not failing: the state is ``stopped`` whatever ``error``
        says.  ``error`` is the ending's own trouble -- the partial artifacts
        the Task's exit writer was asked to save and could not -- and a
        record that said nothing of it left the operator a run that looked
        cleanly stopped and a report that was never written.
        """

        if error is not None and not isinstance(error, BaseException):
            raise TypeError("Task stop error must be an exception")
        with self._lock:
            self._require("stopped", allowed=("starting", "running", "stopping"))
            self._end_locked("stopped", error)

    def mark_failed(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("Task failure must be an exception")
        with self._lock:
            if self._state in _TERMINAL_STATES:
                return
            self._end_locked("failed", error)

    def execute(self, work: Callable[["TaskRun"], _Result]) -> _Result:
        """Run direct/notebook Task work through this same durable lifecycle."""

        if not callable(work):
            raise TypeError("Task work must be callable")
        self.mark_running()
        try:
            result = work(self)
        except BaseException as error:
            self.mark_failed(error)
            raise
        self.mark_completed()
        return result

    def report_progress(
        self,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        with self._lock:
            if self._state in _TERMINAL_STATES:
                raise RuntimeError("terminal Task run cannot report progress")
            self._progress = {
                "message": _text(message, "Task progress message"),
                "current": current,
                "total": total,
            }

    def register_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        role: str,
        contract_id: str = "",
    ) -> TaskArtifact:
        selected_name = _text(name, "Task artifact name")
        selected_role = _text(role, "Task artifact role")
        if selected_role not in _ARTIFACT_ROLES:
            raise ValueError(
                f"Task artifact role must be one of {tuple(sorted(_ARTIFACT_ROLES))}"
            )
        selected_contract = str(contract_id).strip()
        resolved = Path(path).expanduser().resolve()
        try:
            relative = resolved.relative_to(self.directory)
        except ValueError as error:
            raise ValueError("Task artifacts must stay inside their run directory") from error
        relative_text = relative.as_posix()
        if relative_text in (_START_RECORD, _RUN_RECORD):
            raise ValueError(
                f"{relative_text} is the run's own record, not a Task artifact"
            )
        if not resolved.is_file():
            raise FileNotFoundError(f"Task artifact is not a file: {resolved}")
        artifact = TaskArtifact(
            selected_name,
            resolved,
            relative_text,
            selected_role,
            selected_contract,
            resolved.stat().st_size,
        )
        with self._lock:
            if self._state in _TERMINAL_STATES:
                raise RuntimeError("terminal Task run cannot register an artifact")
            existing = self._artifacts.get(selected_name)
            if existing is not None:
                if (
                    existing.path != artifact.path
                    or existing.role != artifact.role
                    or existing.contract_id != artifact.contract_id
                ):
                    raise ValueError(
                        f"Task artifact {selected_name!r} is already registered"
                    )
                if existing == artifact:
                    return existing
            self._artifacts[selected_name] = artifact
        return artifact

    def _require(self, state: str, *, allowed: tuple[str, ...]) -> None:
        if self._state not in allowed:
            raise RuntimeError(
                f"Task run cannot enter {state} from {self._state}"
            )

    def _end_locked(self, state: str, error: BaseException | None) -> None:
        """Enter a terminal state by writing the run record, once.

        The record is the state: a record that could not be written is a
        terminal state that was not entered, so the run is still open for
        the caller's ``mark_failed`` to end it with the write error as its
        error -- the outcome reaches the disk unless the disk refuses
        twice, and a refusal is never silent.  A terminal state is reached
        once, and its record is created, never replaced: there is no
        destination for an open handle to hold.
        """

        previous = (self._state, self._ended_at, self._error)
        self._state = state
        self._ended_at = _now()
        if error is not None:
            self._error = self._error_document(error)
        try:
            write_readable_json(
                self.directory / _RUN_RECORD,
                {
                    "schema": "zlc.task-run",
                    "run_id": self.directory.name,
                    "task": {
                        "api_name": self.task_name,
                        "instance_id": self.instance_id,
                    },
                    "input": self._input,
                    "status": {
                        "state": self._state,
                        "started_at": self._started_at,
                        "ended_at": self._ended_at,
                        "progress": self._progress,
                        "stop_reason": self._stop_reason,
                    },
                    "artifacts": [
                        {
                            "name": artifact.name,
                            "path": artifact.relative_path,
                            "role": artifact.role,
                            "contract_id": artifact.contract_id,
                            "size_bytes": artifact.size_bytes,
                        }
                        for artifact in self._artifacts.values()
                    ],
                    "error": self._error,
                },
            )
        except BaseException:
            self._state, self._ended_at, self._error = previous
            raise

    @staticmethod
    def _error_document(error: BaseException) -> dict[str, object]:
        return {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": "".join(
                traceback.TracebackException.from_exception(error).format()
            ),
            "exceptions": [
                TaskRun._error_document(child)
                for child in getattr(error, "exceptions", ())
            ],
        }
