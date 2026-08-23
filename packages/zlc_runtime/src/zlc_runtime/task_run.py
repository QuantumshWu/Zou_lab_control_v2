"""Crash-honest durable records for hosted Task runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
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
    if value is None or type(value) in (str, bool, int, float):
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

    ``run.json`` is the only metadata truth.  This owner never inspects or
    saves scientific data: a domain Task writes a complete chosen file and
    then registers that file here.
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
        self._write_locked()

    @classmethod
    def create(
        cls,
        run_root: str | Path,
        *,
        task_name: str,
        instance_id: str,
        input_summary: Mapping[str, object],
    ) -> "TaskRun":
        if not isinstance(input_summary, Mapping):
            raise TypeError("Task input summary must be a mapping")
        root = Path(run_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Task run root does not exist: {root}")
        selected = unique_path(root, _text(task_name, "task name"), "")
        return cls(
            selected,
            task_name=task_name,
            instance_id=instance_id,
            input_summary=input_summary,
        )

    @property
    def artifacts(self) -> tuple[TaskArtifact, ...]:
        with self._lock:
            return tuple(self._artifacts.values())

    def artifact(self, name: str) -> TaskArtifact | None:
        with self._lock:
            return self._artifacts.get(str(name))

    def mark_running(self) -> None:
        self._transition("running", allowed=("starting",))

    def mark_stopping(self, reason: str) -> None:
        with self._lock:
            if self._state in _TERMINAL_STATES:
                return
            if self._state not in {"starting", "running", "stopping"}:
                raise RuntimeError(f"Task run cannot stop from {self._state}")
            self._state = "stopping"
            self._stop_reason = _text(reason, "Task stop reason")
            self._write_locked()

    def mark_completed(self) -> None:
        self._transition("completed", allowed=("running",))

    def mark_stopped(self) -> None:
        self._transition("stopped", allowed=("starting", "running", "stopping"))

    def mark_failed(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("Task failure must be an exception")
        with self._lock:
            if self._state in _TERMINAL_STATES:
                return
            self._state = "failed"
            self._ended_at = _now()
            self._progress = None
            self._error = self._error_document(error)
            self._write_locked()

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
            self._write_locked()

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
        if relative_text == "run.json":
            raise ValueError("run.json cannot be registered as a Task artifact")
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
            self._write_locked()
        return artifact

    def _transition(self, state: str, *, allowed: tuple[str, ...]) -> None:
        with self._lock:
            if self._state not in allowed:
                raise RuntimeError(
                    f"Task run cannot enter {state} from {self._state}"
                )
            self._state = state
            if state in _TERMINAL_STATES:
                self._ended_at = _now()
                self._progress = None
            self._write_locked()

    def _write_locked(self) -> None:
        write_readable_json(
            self.directory / "run.json",
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
                    "updated_at": _now(),
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
