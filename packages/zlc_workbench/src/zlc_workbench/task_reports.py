"""UI-neutral Task FINAL-publication to ordinary plot-page composition.

A TaskConsole should not know which tasks own reports and a task adapter should
not know which Qt window presents one.  This module is the single boundary:
adapters register against a descriptor's declared output contracts, consume one
exact terminal sibling publication, and return pages that the common raster
plot host presents.  There is no task-name branch and no second renderer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
import time
from types import MappingProxyType

from zlc_runtime import SignalPublication

from .prepared_panel import (
    PreparedPanelPageSpec,
    PreparedPanelSurface,
    create_prepared_panel_surface,
)


__all__ = [
    "TaskReport",
    "TaskReportAdapter",
    "TaskReportCoordinator",
    "TaskReportEvent",
    "TaskReportExport",
    "TaskReportReady",
    "TaskReportRegistry",
    "default_task_report_registry",
]


@dataclass(frozen=True, slots=True)
class TaskReportExport:
    """Declared artifact anchor and exporter for one prepared report."""

    artifact_output_name: str
    artifact_contract_id: str
    write: Callable[[object, Sequence[PreparedPanelSurface]], object]

    def __post_init__(self) -> None:
        name = str(self.artifact_output_name).strip()
        contract = str(self.artifact_contract_id).strip()
        if not name or not contract:
            raise ValueError("Task report export requires an artifact output contract")
        if not callable(self.write):
            raise TypeError("Task report export writer must be callable")
        object.__setattr__(self, "artifact_output_name", name)
        object.__setattr__(self, "artifact_contract_id", contract)


@dataclass(frozen=True, slots=True)
class TaskReport:
    adapter_id: str
    title: str
    publication: SignalPublication
    pages: tuple[PreparedPanelPageSpec, ...]
    export: TaskReportExport | None = None

    def __post_init__(self) -> None:
        adapter_id = str(self.adapter_id).strip()
        title = str(self.title).strip()
        pages = tuple(self.pages)
        if not adapter_id or not title:
            raise ValueError("Task report adapter id and title are required")
        if not isinstance(self.publication, SignalPublication):
            raise TypeError("Task report source must be a SignalPublication")
        if not pages or any(
            not isinstance(page, PreparedPanelPageSpec) for page in pages
        ):
            raise TypeError("Task report pages must follow PreparedPanelPageSpec")
        keys = tuple(str(page.key) for page in pages)
        if len(set(keys)) != len(keys):
            raise ValueError("Task report page keys must be unique")
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "pages", pages)
        if self.export is not None and not isinstance(self.export, TaskReportExport):
            raise TypeError("Task report export must be TaskReportExport or None")


@dataclass(frozen=True, slots=True)
class TaskReportAdapter:
    """One declarative Task descriptor/publication to report-page adapter."""

    adapter_key: str
    title: str
    output_contracts: tuple[tuple[str, str], ...]
    build_pages: Callable[[SignalPublication], Sequence[PreparedPanelPageSpec]]
    export: TaskReportExport | None = None

    def __post_init__(self) -> None:
        adapter_key = str(self.adapter_key).strip()
        title = str(self.title).strip()
        contracts = tuple(
            (str(name).strip(), str(contract).strip())
            for name, contract in self.output_contracts
        )
        if not adapter_key or not title:
            raise ValueError("Task report adapter key and title are required")
        if not contracts or any(not name or not contract for name, contract in contracts):
            raise ValueError("Task report output contracts must be non-empty")
        if len({name for name, _contract in contracts}) != len(contracts):
            raise ValueError("Task report output names must be unique")
        if not callable(self.build_pages):
            raise TypeError("Task report build_pages must be callable")
        if self.export is not None and not isinstance(self.export, TaskReportExport):
            raise TypeError("Task report adapter export must be TaskReportExport or None")
        object.__setattr__(self, "adapter_key", adapter_key)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "output_contracts", contracts)

    def build(
        self,
        report_spec: object,
        descriptor: object,
        publication: SignalPublication,
    ) -> TaskReport:
        if str(getattr(report_spec, "adapter_key", "")) != self.adapter_key:
            raise ValueError("Task report adapter received another report spec")
        output_names = tuple(getattr(report_spec, "output_names", ()))
        expected_names = tuple(name for name, _contract in self.output_contracts)
        if output_names != expected_names:
            raise ValueError("Task report spec output order differs from its adapter")
        declared_by_name = {
            str(getattr(output, "name", "")): str(
                getattr(output, "contract_id", "")
            )
            for output in tuple(getattr(descriptor, "outputs", ()))
        }
        if tuple(
            (name, declared_by_name.get(name, "")) for name in output_names
        ) != self.output_contracts:
            raise ValueError("Task report output contracts differ from its adapter")
        if self.export is not None:
            declared_artifacts = {
                str(getattr(output, "name", "")): str(
                    getattr(output, "contract_id", "")
                )
                for output in tuple(getattr(descriptor, "artifact_outputs", ()))
            }
            if declared_artifacts.get(self.export.artifact_output_name) != (
                self.export.artifact_contract_id
            ):
                raise ValueError(
                    "Task report export artifact differs from its descriptor"
                )
        if not isinstance(publication, SignalPublication):
            raise TypeError("Task report requires an exact SignalPublication")
        bare_names = tuple(
            str(name).rsplit("/", 1)[-1] for name in publication.signals
        )
        if len(set(bare_names)) != len(bare_names) or set(bare_names) != set(
            expected_names
        ):
            raise ValueError(
                f"{self.adapter_key} publication is not its exact sibling bundle"
            )
        pages = tuple(self.build_pages(publication))
        unknown_page_signals = {
            str(page.signal) for page in pages
        } - set(expected_names)
        if unknown_page_signals:
            raise ValueError(
                "Task report pages use outputs outside their exact sibling bundle: "
                f"{sorted(unknown_page_signals)}"
            )
        return TaskReport(
            self.adapter_key,
            self.title,
            publication,
            pages,
            self.export,
        )


class TaskReportRegistry:
    """Replace-free registry used by the TaskConsole composition root."""

    def __init__(self) -> None:
        self._by_key: dict[str, TaskReportAdapter] = {}

    @property
    def adapters(self) -> Mapping[str, TaskReportAdapter]:
        return MappingProxyType(dict(self._by_key))

    def register(self, adapter: TaskReportAdapter) -> None:
        if not isinstance(adapter, TaskReportAdapter):
            raise TypeError("adapter must be TaskReportAdapter")
        if adapter.adapter_key in self._by_key:
            raise ValueError(
                f"Task report adapter {adapter.adapter_key!r} is already registered"
            )
        self._by_key[adapter.adapter_key] = adapter

    def build(
        self,
        report_spec: object,
        descriptor: object,
        publication: SignalPublication,
    ) -> TaskReport:
        key = str(getattr(report_spec, "adapter_key", ""))
        try:
            adapter = self._by_key[key]
        except KeyError as exc:
            raise KeyError(f"no Task report adapter registered for {key!r}") from exc
        return adapter.build(report_spec, descriptor, publication)


@dataclass(frozen=True, slots=True)
class TaskReportReady:
    """One fully-described report whose surface ownership can be transferred."""

    node_id: str
    report: TaskReport
    surfaces: tuple[PreparedPanelSurface, ...]
    artifact_path: Path | None


@dataclass(frozen=True, slots=True)
class TaskReportEvent:
    """UI-neutral outcome for a composition root to project."""

    message: str = ""
    severity: str = "task"
    panel_ids: tuple[str, ...] = ()
    panel_status: str = ""
    panel_error: bool = False


@dataclass(slots=True)
class _ActiveSurface:
    panel_id: str
    surface: PreparedPanelSurface
    complete_reported: bool = False
    reported_error: BaseException | None = None


@dataclass(slots=True)
class _ExportJob:
    ready: TaskReportReady
    node_id: str
    title: str
    destination: Path
    panel_ids: tuple[str, ...]
    done: Event = field(default_factory=Event)
    error: BaseException | None = None
    written: object | None = None
    thread: Thread | None = None


@dataclass(slots=True)
class _ExportOutcome:
    ready: TaskReportReady
    error: BaseException | None
    projected: bool = False


class TaskReportCoordinator:
    """Single owner of terminal-report preparation and background export.

    The console supplies an exact terminal publication and artifact results,
    then consumes ready groups and plain events.  This object alone retains
    publication identities, owns not-yet-mounted surfaces, and joins exporters.
    Mounted surfaces remain ordinary panel-owned surfaces.
    """

    def __init__(
        self,
        registry: TaskReportRegistry,
        *,
        surface_factory: Callable[..., PreparedPanelSurface] = (
            create_prepared_panel_surface
        ),
    ) -> None:
        if not isinstance(registry, TaskReportRegistry):
            raise TypeError("registry must be TaskReportRegistry")
        if not callable(surface_factory):
            raise TypeError("surface_factory must be callable")
        self._registry = registry
        self._surface_factory = surface_factory
        # Keeping the object itself prevents CPython from recycling its id for
        # another run while this console remains alive.
        self._publications: dict[tuple[str, str, int], SignalPublication] = {}
        self._pending: list[TaskReportReady] = []
        self._ready: list[TaskReportReady] = []
        self._active: dict[str, _ActiveSurface] = {}
        self._exports: list[_ExportJob] = []
        self._export_outcomes: dict[int, _ExportOutcome] = {}
        self._events: list[TaskReportEvent] = []

    @property
    def exporting(self) -> bool:
        return any(not job.done.is_set() for job in self._exports)

    @property
    def panel_ids(self) -> tuple[str, ...]:
        return tuple(self._active)

    def panel_locked(self, panel_id: str) -> bool:
        key = str(panel_id)
        return any(
            key in job.panel_ids and not job.done.is_set()
            for job in self._exports
        )

    def _export_job(self, ready: TaskReportReady) -> _ExportJob | None:
        return next(
            (job for job in self._exports if job.ready is ready),
            None,
        )

    @staticmethod
    def _artifact_path(
        report: TaskReport,
        artifact_results: Sequence[Mapping[str, object]],
    ) -> Path | None:
        export = report.export
        if export is None:
            return None
        for row in artifact_results:
            if (
                str(row.get("name", "")) == export.artifact_output_name
                and str(row.get("contract_id", ""))
                == export.artifact_contract_id
            ):
                value = str(row.get("path", "")).strip()
                if value:
                    return Path(value).expanduser().resolve()
        return None

    def observe(
        self,
        *,
        node_id: str,
        descriptor: object,
        report_spec: object,
        publication: SignalPublication,
        trigger_signal: str,
        artifact_results: Sequence[Mapping[str, object]],
        signal_for_output: Callable[[str], str],
        interval_ms: int,
    ) -> None:
        """Observe one exact trigger without blocking or rebuilding it twice."""

        if not isinstance(publication, SignalPublication):
            raise TypeError("Task report observation requires SignalPublication")
        trigger_value = publication.value(str(trigger_signal))
        if trigger_value is None:
            raise ValueError("Task report trigger is absent from its publication")
        # Preview and partial publications may share the stable trigger name;
        # only the exact, non-transient FINAL sibling bundle owns a report.
        if trigger_value.coverage is not None or trigger_value.transient:
            return
        adapter_key = str(getattr(report_spec, "adapter_key", ""))
        identity = (str(node_id), adapter_key, id(publication))
        if self._publications.get(identity) is publication:
            return

        try:
            report = self._registry.build(report_spec, descriptor, publication)
        except Exception as error:
            self._publications[identity] = publication
            self._events.append(
                TaskReportEvent(
                    f"{node_id} report could not be prepared: {error}",
                    "error",
                )
            )
            return

        artifact_path = self._artifact_path(report, artifact_results)
        # The signal plane can expose FINAL one poll before NodeHost exposes its
        # result object.  Do not mark this publication observed until its exact
        # declared artifact destination exists.
        if report.export is not None and artifact_path is None:
            return

        self._publications[identity] = publication
        surfaces: list[PreparedPanelSurface] = []
        try:
            for page in report.pages:
                surfaces.append(
                    self._surface_factory(
                        page,
                        signal=signal_for_output(str(page.signal)),
                        interval_ms=int(interval_ms),
                    )
                )
        except Exception as error:
            for surface in surfaces:
                surface.close()
            self._events.append(
                TaskReportEvent(
                    f"{node_id} report could not be prepared: {error}",
                    "error",
                )
            )
            return
        self._pending.append(
            TaskReportReady(
                str(node_id),
                report,
                tuple(surfaces),
                artifact_path,
            )
        )

    def _poll_pending(self) -> None:
        for pending in tuple(self._pending):
            failure = next(
                (
                    error
                    for surface in pending.surfaces
                    if (error := surface.failure()) is not None
                ),
                None,
            )
            if failure is not None:
                job = self._export_job(pending)
                if job is not None and not job.done.is_set():
                    # The exporter owns these same hosts until its worker has
                    # observed the asynchronous failure and unwound.
                    continue
                self._pending.remove(pending)
                for surface in pending.surfaces:
                    surface.close()
                self._events.append(
                    TaskReportEvent(
                        f"{pending.node_id} report could not be rendered: {failure}",
                        "error",
                    )
                )
                continue
            if all(surface.descriptions_ready for surface in pending.surfaces):
                self._pending.remove(pending)
                self._ready.append(pending)

    def take_ready(self) -> tuple[TaskReportReady, ...]:
        """Transfer every ready surface group to the composition root."""

        ready = tuple(self._ready)
        self._ready.clear()
        return ready

    def activate(
        self,
        ready: TaskReportReady,
        panel_ids: Sequence[str],
    ) -> None:
        """Observe mounted surfaces and queue canonical export immediately."""

        if not isinstance(ready, TaskReportReady):
            raise TypeError("ready must be TaskReportReady")
        keys = tuple(str(panel_id) for panel_id in panel_ids)
        if len(keys) != len(ready.surfaces) or len(set(keys)) != len(keys):
            raise ValueError("mounted report panels differ from their ready surfaces")
        if any(key in self._active for key in keys):
            raise ValueError("a Task report panel is already active")
        export = ready.report.export
        destination: Path | None = None
        if export is not None:
            if ready.artifact_path is None:
                raise RuntimeError(
                    "Task report export lost its declared artifact path"
                )
            destination = ready.artifact_path.with_suffix("") / "report"

        for key, surface in zip(keys, ready.surfaces, strict=True):
            self._active[key] = _ActiveSurface(key, surface)
        if export is None:
            return

        existing = self._export_job(ready)
        if existing is not None:
            existing.panel_ids = keys
            return
        outcome = self._export_outcomes.get(id(ready))
        if outcome is not None and outcome.ready is ready:
            error = outcome.error
            if error is not None:
                self._events.append(
                    TaskReportEvent(
                        f"{ready.node_id} report export failed: {error}",
                        "error",
                        keys,
                        f"Report export failed: {error}",
                        True,
                    )
                )
            return
        self._start_export(ready, keys, destination)

    def _start_export(
        self,
        ready: TaskReportReady,
        panel_ids: tuple[str, ...],
        destination: Path | None = None,
    ) -> None:
        export = ready.report.export
        if export is None:
            return
        if destination is None:
            if ready.artifact_path is None:
                raise RuntimeError(
                    "Task report export lost its declared artifact path"
                )
            destination = ready.artifact_path.with_suffix("") / "report"
        job = _ExportJob(
            ready,
            ready.node_id,
            ready.report.title,
            destination,
            panel_ids,
        )

        def write() -> None:
            try:
                job.written = export.write(destination, ready.surfaces)
            except BaseException as error:
                job.error = error
            finally:
                job.done.set()

        job.thread = Thread(
            target=write,
            name=f"zlc-report-export-{ready.node_id}",
            daemon=True,
        )
        self._exports.append(job)
        try:
            job.thread.start()
        except BaseException as error:
            self._exports.remove(job)
            self._export_outcomes[id(ready)] = _ExportOutcome(ready, error)
            self._events.append(
                TaskReportEvent(
                    f"{ready.node_id} report export could not start: {error}",
                    "error",
                    panel_ids,
                    f"Report export failed: {error}",
                    True,
                )
            )

    def release_panel(self, panel_id: str) -> None:
        """Stop observing one mounted surface without taking its ownership."""

        key = str(panel_id)
        if self.panel_locked(key):
            raise RuntimeError(
                f"{key} cannot be released while its report is exporting"
            )
        self._active.pop(key, None)

    def _poll_active(self) -> None:
        for active in tuple(self._active.values()):
            error = active.surface.failure()
            if error is not None:
                if active.reported_error is error:
                    continue
                active.reported_error = error
                self._events.append(
                    TaskReportEvent(
                        f"{active.panel_id}: {error}",
                        "error",
                        (active.panel_id,),
                        f"Report render failed: {error}",
                        True,
                    )
                )
                continue
            if active.surface.configured and not active.complete_reported:
                active.complete_reported = True
                self._events.append(
                    TaskReportEvent(
                        panel_ids=(active.panel_id,),
                        panel_status="Report ready",
                    )
                )

    def _poll_exports(self) -> None:
        for job in tuple(self._exports):
            if not job.done.is_set():
                continue
            self._exports.remove(job)
            self._export_outcomes[id(job.ready)] = _ExportOutcome(
                job.ready,
                job.error,
            )
            if job.error is not None:
                self._events.append(
                    TaskReportEvent(
                        f"{job.node_id} report export failed: {job.error}",
                        "error",
                        job.panel_ids,
                        f"Report export failed: {job.error}",
                        True,
                    )
                )
            else:
                self._events.append(
                    TaskReportEvent(
                        f"{job.title} saved to {job.destination}",
                        "task",
                    )
                )

    def poll(self) -> None:
        """Advance only already-completed work; never await in a GUI beat."""

        self._poll_exports()
        self._poll_pending()
        self._poll_active()

    def take_events(self) -> tuple[TaskReportEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        # Once the composition root drains events, any export error is visible
        # through its ordinary status projection and need not refuse a later
        # deliberate close a second time.
        for outcome in self._export_outcomes.values():
            if outcome.error is not None:
                outcome.projected = True
        return events

    def close(self, *, timeout: float) -> None:
        """Join exporters or fail closed while every source host stays alive."""

        # A close may arrive in the short interval between FINAL observation
        # and surface-description completion.  Queue export from those exact
        # already-created hosts before waiting; otherwise closing a pending
        # group would silently discard a successful Task's report artifacts.
        for ready in (*self._pending, *self._ready):
            if (
                ready.report.export is not None
                and self._export_job(ready) is None
                and not (
                    (outcome := self._export_outcomes.get(id(ready))) is not None
                    and outcome.ready is ready
                )
            ):
                self._start_export(ready, ())
        deadline = time.monotonic() + max(0.0, float(timeout))
        for job in tuple(self._exports):
            thread = job.thread
            if thread is not None:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._poll_exports()
        unfinished = tuple(
            str(job.destination)
            for job in self._exports
            if not job.done.is_set()
        )
        if unfinished:
            raise TimeoutError(
                "report export did not finish before close: "
                + ", ".join(unfinished)
            )
        unprojected_failures = tuple(
            outcome
            for outcome in self._export_outcomes.values()
            if outcome.error is not None
            and not outcome.projected
        )
        if unprojected_failures:
            details = "; ".join(
                f"{outcome.ready.node_id}: {outcome.error}"
                for outcome in unprojected_failures
            )
            raise RuntimeError(f"report export failed before close: {details}")
        # These have never transferred to PanelBinding.  Mounted surfaces are
        # deliberately untouched and will be released by the ordinary panel
        # lifecycle after this method returns successfully.
        for report in (*self._pending, *self._ready):
            for surface in report.surfaces:
                surface.close()
        self._pending.clear()
        self._ready.clear()
        self._active.clear()
        self._export_outcomes.clear()


def default_task_report_registry() -> TaskReportRegistry:
    """Build the application's explicit report-adapter composition."""

    from .calibration_report import calibration_task_report_adapter

    registry = TaskReportRegistry()
    registry.register(calibration_task_report_adapter())
    return registry
