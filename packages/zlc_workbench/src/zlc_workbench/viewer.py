"""Opening a saved figure and reading what it was.

An archive already carries everything: the datasets with their axes, the
apparatus state, what was asked of it, which pulse drove it, which panel showed
what.  Until now it carried them for nobody -- `read_archive` returned a nested
JSON document, and reading a run six months later meant reading that document.

Two jobs, kept apart:

* :func:`describe_archive` projects the document into labelled rows.  It is
  pure, takes no Qt and no session, and is what makes "what was the apparatus
  doing" answerable in a notebook as well as a window.
* :class:`FigureViewerPresenter` connects those rows and a rebuilt dataset to a
  mute view.

Both stop short of interpretation.  Rows are named for the subject they came
from and are not renamed into something friendlier, because a record whose
labels drift from the ones the producing package used is a record you cannot
grep for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from zlc_plot import DEFAULTS

from .archive import read_archive, read_dataset
from .panel_save import (
    restore_panel_plot_input,
    restore_panel_viewport,
)
from .panel_state import PanelState


__all__ = ["ArchiveDescription", "FigureViewerPresenter", "describe_archive"]


Rows = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ArchiveDescription:
    """One saved figure, as tabs of labelled rows."""

    name: str
    schema: str
    #: (title, rows) pairs, in the order the view shows them.
    tabs: tuple[tuple[str, Rows], ...]
    #: Which datasets can be reopened, as (key, label) in saved order.  The
    #: key is the archive's own name for it; the label is what it IS, taken
    #: from the panel record that was saved beside it.
    datasets: tuple[tuple[str, str], ...]

    @property
    def dataset_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _label in self.datasets)


def describe_archive(
    info: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> ArchiveDescription:
    """Project one archive's info document into rows a person can read."""

    sections = info.get("sections", {})
    provenance = sections.get("provenance", {})
    def _label(key: str) -> str:
        entry = _panel_record(sections, key)
        if not isinstance(entry, Mapping):
            return key
        title = str(entry.get("title") or "").strip()
        signal = str(entry.get("signal") or "").strip()
        if title and signal and title != signal:
            return f"{title} — {signal}"
        return title or signal or key

    keys = tuple(sections.get("dataset", {}))
    datasets = tuple((key, _label(key)) for key in keys)
    return ArchiveDescription(
        name=str(info.get("name", "")),
        schema=str(info.get("schema", "")),
        datasets=datasets,
        tabs=(
            ("Plot", _plot_rows(sections, arrays, keys)),
            ("Measurement", _measurement_rows(sections, provenance)),
            ("Device", _device_rows(provenance)),
            ("Flow", _flow_rows(sections, provenance)),
            ("Raw", _flatten(sections)),
        ),
    )


def _panel_record(
    sections: Mapping[str, Any],
    dataset: str,
) -> Mapping[str, Any] | None:
    """Return the panel record which explicitly names ``dataset``."""

    panel = sections.get("panel", {})
    if not isinstance(panel, Mapping):
        return None
    if str(panel.get("dataset", "")) == str(dataset):
        state = panel.get("state")
        return state if isinstance(state, Mapping) else panel
    record = panel.get(str(dataset))
    return record if isinstance(record, Mapping) else None


def _panel_state(
    sections: Mapping[str, Any],
    dataset: str,
) -> PanelState | None:
    panel = sections.get("panel", {})
    if not isinstance(panel, Mapping) or str(panel.get("dataset", "")) != str(dataset):
        return None
    document = panel.get("state")
    if not isinstance(document, Mapping):
        return None
    return PanelState.from_document(document)


def _plot_rows(
    sections: Mapping[str, Any],
    arrays: Mapping[str, Any],
    datasets: Sequence[str],
) -> Rows:
    """What is in the file, and what each panel was showing."""

    rows: list[tuple[str, str]] = []
    for name, array in sorted(arrays.items()):
        shape = "x".join(str(size) for size in getattr(array, "shape", ()))
        dtype = getattr(getattr(array, "dtype", None), "name", "")
        reopenable = "" if name in datasets else "  (array only)"
        rows.append((name, f"{shape} {dtype}{reopenable}".strip()))
    for dataset in datasets:
        entry = _panel_record(sections, dataset)
        if isinstance(entry, Mapping):
            title = entry.get("title") or dataset
            rows.append(
                (f"panel {dataset}", f"{title} — {entry.get('signal', '')}")
            )
    return tuple(rows)


def _measurement_rows(
    sections: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Rows:
    """What was asked of the apparatus, and what drove it."""

    rows: list[tuple[str, str]] = []
    for node, record in sorted(provenance.items()):
        if not isinstance(record, Mapping):
            continue
        rows.append((node, str(record.get("layer", ""))))
        captured = record.get("captured_at")
        if captured:
            rows.append((f"{node} captured", str(captured)))
        for key in ("acquisition_parameters", "source_acquisition_parameters"):
            for label, value in sorted(dict(record.get(key, {})).items()):
                rows.append((f"{node}.{label}", _text(value)))
    for label, value in sorted(dict(sections.get("pulse", {})).items()):
        rows.append((f"pulse.{label}", _text(value)))
    return tuple(rows)


def _device_rows(provenance: Mapping[str, Any]) -> Rows:
    """The apparatus, exactly as each device reported itself."""

    rows: list[tuple[str, str]] = []
    for node, record in sorted(provenance.items()):
        if not isinstance(record, Mapping):
            continue
        for role, state in sorted(dict(record.get("devices", {})).items()):
            rows.append((f"{node}.{role}", ""))
            if isinstance(state, Mapping):
                for label, value in sorted(state.items()):
                    rows.append((f"    {label}", _text(value)))
            else:
                rows.append(("    state", _text(state)))
    return tuple(rows)


def _flow_rows(
    sections: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Rows:
    """Who fed whom: the chain from apparatus to the panel on screen.

    This is the question a saved figure is least able to answer on its own and
    the one most often asked -- "where did this number come from" -- so it gets
    its own reading rather than being spelled out across three other tabs.
    """

    rows: list[tuple[str, str]] = []
    signal_to_panel: dict[str, str] = {}
    for panel, entry in sorted(sections.get("panel", {}).items()):
        if isinstance(entry, Mapping) and entry.get("signal"):
            signal_to_panel[str(entry["signal"])] = str(entry.get("title") or panel)

    for node, record in sorted(provenance.items()):
        if not isinstance(record, Mapping):
            continue
        upstream = record.get("consumes") or []
        devices = sorted(dict(record.get("devices", {})))
        source = ", ".join(str(item) for item in upstream) or ", ".join(devices) or "—"
        shown = [
            f"{signal}→{panel}"
            for signal, panel in signal_to_panel.items()
            if f"/{node}/" in signal
        ]
        rows.append((node, f"{source}  ⇒  {', '.join(shown) or 'not shown'}"))

    for signal, panel in sorted(signal_to_panel.items()):
        if not any(f"/{node}/" in signal for node in provenance):
            rows.append((panel, f"{signal}  (no record of what produced it)"))
    return tuple(rows)


def _flatten(value: Any, prefix: str = "") -> Rows:
    """Every leaf of the document, so nothing is hidden by the other tabs."""

    if isinstance(value, Mapping):
        rows: list[tuple[str, str]] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            rows.extend(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return tuple(rows)
    return ((prefix, _text(value)),)


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class FigureViewerPresenter:
    """Connects a saved-figure view to archives on disk."""

    def __init__(
        self,
        view: object,
        *,
        make_host: Callable[[Any, str, PanelState | None], Any],
        run_off_thread: Callable[
            [
                Callable[[], object],
                Callable[[object], None],
                Callable[[BaseException], None],
            ],
            None,
        ],
        close_worker: Callable[[], bool],
        request_close: Callable[[], None],
        edit_figure: Callable[[Any, str], object] | None = None,
    ) -> None:
        self.view = view
        self._make_host = make_host
        self._run_off_thread = run_off_thread
        self._close_worker = close_worker
        self._request_close = request_close
        #: Opening the plot's own controls is Qt work this asks for rather
        #: than does, the same way the console asks for it.
        self._edit_figure = edit_figure
        self.path: Path | None = None
        self.description: ArchiveDescription | None = None
        #: The archive as read, kept so switching datasets does not re-read the
        #: file: an archive is immutable once written.
        self._info: Mapping[str, Any] = {}
        self._arrays: Mapping[str, Any] = {}
        self.dataset = ""
        self.panel_state: PanelState | None = None
        self._host: Any = None
        self._retired_host: Any = None
        self._retirement_pending = False
        self._busy = False
        self._close_requested = False
        self._closed = False
        self.view.set_panel_sizes(
            DEFAULTS.layout.size_names, DEFAULTS.layout.default_preset
        )
        self._connect()

    def _connect(self) -> None:
        self.view.path_committed.connect(self.open)
        self.view.dataset_picked.connect(self.show_dataset)
        self.view.save_image_requested.connect(self.save_image)
        # The figure is a panel, so the decisions a panel carries are answered.
        self.view.figure_size_picked.connect(self.resize_figure)
        self.view.figure_edit_requested.connect(self.edit_figure)

    def open(self, path: str) -> None:
        """Submit one complete archive candidate without blocking the Qt owner.

        The accepted archive, path, dataset and host change together only after
        reading, rebuilding and configuring the candidate all succeed.  A bad
        path therefore cannot tear down the last figure that opened correctly.
        """

        requested = Path(path)

        def prepare() -> object:
            resolved = requested.resolve()
            info, arrays = read_archive(resolved)
            description = describe_archive(info, arrays)
            candidate = self._prepare_dataset(
                info,
                arrays,
                description,
                description.dataset_keys[0] if description.datasets else "",
            )
            return resolved, info, arrays, description, candidate

        self._submit(
            f"opening {requested.name}…",
            prepare,
            self._accept_archive,
            f"cannot open {requested.name}",
        )

    def show_dataset(self, name: str) -> None:
        """Submit another dataset from the already accepted immutable archive.

        Panel Save Fig writes one dataset; notebook-created figure archives may
        still contain several.  The viewer therefore keeps the dataset choice
        explicit without implying that TaskConsole saves the whole board.
        """

        if not self._info or not name:
            return
        info, arrays, description = self._info, self._arrays, self.description
        if description is None:
            return

        def prepare() -> object:
            return self._prepare_dataset(info, arrays, description, str(name))

        self._submit(
            f"opening {name}…",
            prepare,
            self._accept_dataset,
            f"cannot draw {name}",
        )

    def _prepare_dataset(
        self,
        info: Mapping[str, Any],
        arrays: Mapping[str, Any],
        description: ArchiveDescription,
        name: str,
    ) -> tuple[str, PanelState | None, object | None, Any]:
        if not name:
            return "", None, None, None
        label = dict(description.datasets).get(str(name), str(name))
        host: Any = None
        try:
            snapshot = read_dataset(info, arrays, str(name))
            plot_input = restore_panel_plot_input(info, arrays, str(name), snapshot)
            panel_state = _panel_state(info.get("sections", {}), str(name))
            viewport = (
                None
                if panel_state is None
                else restore_panel_viewport(info, str(name))
            )
            host = self._make_host(plot_input, label, panel_state)
            self._configure_host(host, panel_state, viewport)
            return str(name), panel_state, viewport, host
        except BaseException:
            if host is not None:
                self._close_host(host)
            raise

    def _accept_archive(self, result: object) -> bool:
        resolved, info, arrays, description, candidate = result
        previous = self._host
        try:
            self._show_candidate(candidate, description)
            self.view.set_title(description.name or resolved.stem)
            self.view.set_path(str(resolved))
            self.view.set_info(description.tabs)
            self.view.set_datasets(
                description.datasets,
                description.dataset_keys[0] if description.datasets else "",
            )
        except BaseException:
            self._restore_previous_surface(previous)
            self._discard_candidate(candidate)
            raise

        name, panel_state, _viewport, host = candidate
        self.path = resolved
        self.description = description
        self._info, self._arrays = info, arrays
        self._host = host
        self.dataset = str(name)
        self.panel_state = panel_state
        return self._retire_previous(previous)

    def _accept_dataset(self, candidate: object) -> bool:
        description = self.description
        if description is None:
            raise RuntimeError("viewer has no accepted archive description")
        previous = self._host
        try:
            self._show_candidate(candidate, description)
        except BaseException:
            self._restore_previous_surface(previous)
            self._discard_candidate(candidate)
            raise

        name, panel_state, _viewport, host = candidate
        self._host = host
        self.dataset = str(name)
        self.panel_state = panel_state
        return self._retire_previous(previous)

    def _show_candidate(
        self,
        candidate: object,
        description: ArchiveDescription,
    ) -> None:
        name, panel_state, _viewport, host = candidate
        self.view.show_figure(host)
        if panel_state is not None:
            self.view.set_figure_size(panel_state.size)
        if host is None:
            self.view.set_status(
                "opened; its arrays were saved without axes, so it cannot be replotted"
            )
            return
        total = len(description.datasets)
        position = "" if total <= 1 else f"  ({total} datasets in this file)"
        self.view.set_status(f"showing {name}{position}")

    def _restore_previous_surface(self, previous: object | None) -> None:
        try:
            self.view.show_figure(previous)
            description = self.description
            if description is not None:
                self.view.set_title(
                    description.name
                    or ("" if self.path is None else self.path.stem)
                )
                self.view.set_path("" if self.path is None else str(self.path))
                self.view.set_info(description.tabs)
                self.view.set_datasets(description.datasets, self.dataset)
            if self.panel_state is not None:
                self.view.set_figure_size(self.panel_state.size)
        except BaseException:
            # Preserve the original candidate refusal.  The previous presenter
            # state is still authoritative and a later owner turn can repaint it.
            return

    def _discard_candidate(self, candidate: object) -> None:
        host = candidate[3]
        if host is not None and host is not self._host:
            self._start_retirement(host, "cannot retire rejected figure")

    def _retire_previous(self, previous: object | None) -> bool:
        if previous is None or previous is self._host:
            return True
        self._start_retirement(
            previous,
            "cannot retire previous figure",
            finished=self._finish_operation,
        )
        return False

    def _submit(
        self,
        busy_status: str,
        work: Callable[[], object],
        accepted: Callable[[object], object],
        failure_prefix: str,
        *,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        if self._closed:
            return
        if self._busy:
            self.view.set_status("viewer is busy; wait for the current operation", error=True)
            return
        if self._retired_host is not None:
            self.view.set_status(
                "viewer is still retiring the previous figure", error=True
            )
            return
        self._busy = True
        self.view.set_status(busy_status)

        def report_failure(error: BaseException) -> None:
            if on_failure is None:
                self.view.set_status(f"{failure_prefix}: {error}", error=True)
            else:
                on_failure(error)

        def delivered(result: object) -> None:
            complete = True
            try:
                complete = accepted(result) is not False
            except BaseException as error:  # Qt/mount refusal is still visible
                report_failure(error)
            finally:
                if complete:
                    self._finish_operation()

        def failed(error: BaseException) -> None:
            report_failure(error)
            self._finish_operation()

        try:
            self._run_off_thread(work, delivered, failed)
        except BaseException as error:
            report_failure(error)
            self._finish_operation()

    def _finish_operation(self) -> None:
        self._busy = False
        if self._close_requested:
            self._request_close()

    @staticmethod
    def _await(operation: object) -> object:
        return operation.result() if hasattr(operation, "result") else operation

    @classmethod
    def _configure_host(
        cls,
        host: object,
        state: PanelState | None,
        viewport: object | None,
    ) -> None:
        """Apply the saved panel decisions through the plot host's public API."""

        # The host was built from this same record and already holds the
        # appearance it accepts; re-sending the whole saved bag is how names
        # authored under another kind reached a vocabulary that never
        # declared them.  Only what this call adds travels.
        configuration: dict[str, object] = {
            "viewport": viewport,
        }
        if state is not None:
            configuration["size"] = state.size
            configuration["classifier_thresholds"] = state.classifier_thresholds
            configuration["fit"] = dict(state.fit)
            configuration["fit_live"] = False
        cls._await(host.configure(**configuration))

    def resize_figure(self, size: str) -> None:
        """The card and the picture inside it have to agree.

        A card resized around a figure that stayed 2x2 is a big card with a
        small picture in it -- the same rule the console keeps for its panels.
        """

        if self._host is None:
            return
        host = self._host
        previous_size = (
            DEFAULTS.layout.default_preset
            if self.panel_state is None
            else self.panel_state.size
        )

        def resize() -> object:
            self._await(host.configure(size=str(size)))
            return str(size)

        def accepted(selected: object) -> None:
            resolved = str(selected)
            if self._host is host and self.panel_state is not None:
                self.panel_state = replace(self.panel_state, size=resolved)
            self.view.set_figure_size(resolved)
            self.view.set_status(f"resized to {resolved}")

        def rejected(error: BaseException) -> None:
            self.view.set_figure_size(previous_size)
            self.view.set_status(f"cannot resize: {error}", error=True)

        self._submit(
            f"resizing to {size}…",
            resize,
            accepted,
            "cannot resize",
            on_failure=rejected,
        )

    def edit_figure(self) -> object | None:
        """Open the plot's own controls, which belong to the plotting package."""

        if self._host is None or self._edit_figure is None:
            return None
        return self._edit_figure(self._host, self.dataset or "figure")

    def save_image(self) -> None:
        """Write the figure as it is drawn, beside the archive it came from."""

        from zlc_durable import unique_path

        if self._host is None or self.path is None:
            self.view.set_status("there is no figure to save", error=True)
            return
        save = getattr(self._host, "save", None)
        if not callable(save):
            self.view.set_status("this figure cannot save itself", error=True)
            return
        host, path, dataset = self._host, self.path, self.dataset

        def write_image() -> object:
            def write(temporary: Path) -> None:
                self._await(host.save(temporary))

            return unique_path(
                path.parent,
                f"{path.stem}-{dataset or 'figure'}",
                ".png",
                writer=write,
            )

        self._submit(
            "saving image…",
            write_image,
            lambda target: self.view.set_status(f"saved {target.name}"),
            "cannot save",
        )

    @classmethod
    def _close_host(cls, host: object) -> None:
        close = getattr(host, "close", None)
        if not callable(close):
            return
        stopped = cls._await(close())
        if stopped is False:
            raise RuntimeError("plot host did not stop")

    def _start_retirement(
        self,
        host: object,
        failure_prefix: str,
        *,
        finished: Callable[[], None] | None = None,
    ) -> None:
        if self._retired_host is None:
            self._retired_host = host
        elif self._retired_host is not host:
            raise RuntimeError("viewer already owns another retiring figure")
        if self._retirement_pending:
            return
        self._retirement_pending = True

        def retired(_result: object) -> None:
            self._retirement_pending = False
            if self._retired_host is host:
                self._retired_host = None
            if finished is not None:
                finished()
            elif self._close_requested:
                self._request_close()

        def failed(error: BaseException) -> None:
            self._retirement_pending = False
            self._busy = False
            self._close_requested = False
            self.view.set_status(f"{failure_prefix}: {error}", error=True)

        try:
            self._run_off_thread(
                lambda: self._close_host(host),
                retired,
                failed,
            )
        except BaseException as error:
            failed(error)

    def close(self) -> bool:
        """Close guard: keep the window until its host and worker are retired."""

        self._close_requested = True
        if self._closed:
            return True
        if self._busy:
            self.view.set_status("closing after the current operation…")
            return False
        if self._retired_host is not None:
            self.view.set_status("closing previous figure…")
            if self._retirement_pending:
                return False
            self._busy = True
            self._start_retirement(
                self._retired_host,
                "cannot close previous figure",
                finished=self._finish_operation,
            )
            return False
        if self._host is not None:
            host = self._host
            self._busy = True
            self.view.set_status("closing figure…")

            def retired(_result: object) -> None:
                if self._host is host:
                    self._host = None
                    self.panel_state = None
                    self.view.show_figure(None)
                self._finish_operation()

            def failed(error: BaseException) -> None:
                self._busy = False
                self._close_requested = False
                self.view.set_status(f"cannot close figure: {error}", error=True)

            try:
                self._run_off_thread(
                    lambda: self._close_host(host),
                    retired,
                    failed,
                )
            except BaseException as error:
                failed(error)
            return False
        if not self._close_worker():
            self._request_close()
            return False
        self._closed = True
        return True
