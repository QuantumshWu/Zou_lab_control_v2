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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zlc_plot import DEFAULTS, read_figure_plot

from zlc_data.figure_archive import read_archive


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
    lineage: tuple[tuple[str, tuple], ...]

    @property
    def dataset_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _label in self.datasets)


def describe_archive(
    info: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> ArchiveDescription:
    """Project one archive's info document into rows a person can read."""

    sections = info.get("sections", {})
    if not isinstance(sections, Mapping) or set(sections) != {
        "dataset", "plot", "lineage", "source"
    }:
        raise ValueError("FigureViewer requires dataset, plot, lineage, and source sections")
    source = sections["source"]
    if not isinstance(source, Mapping):
        raise TypeError("figure source section must be an object")
    keys = tuple(sections["dataset"])
    title = str(source.get("title") or "").strip()
    signal = str(source.get("signal") or "").strip()
    label = f"{title} — {signal}" if title and signal and title != signal else title or signal
    datasets = tuple((key, label or key) for key in keys)
    recipes = {key: read_figure_plot(info, arrays, key)[1] for key in keys}
    lineage = _lineage_tree(sections["lineage"])
    return ArchiveDescription(
        name=str(info.get("name", "")),
        schema=str(info.get("schema", "")),
        datasets=datasets,
        lineage=lineage,
        tabs=(
            ("Plot", _plot_rows(arrays, recipes)),
            ("Measurement", _lineage_rows(sections["lineage"])),
            ("Device", _device_rows(sections["lineage"])),
            ("Flow", ()),
            ("Raw", _flatten(sections)),
        ),
    )


def _plot_rows(
    arrays: Mapping[str, Any],
    recipes: Mapping[str, Mapping[str, object]],
) -> Rows:
    """What is in the file, and what each panel was showing."""

    rows: list[tuple[str, str]] = []
    for name, array in sorted(arrays.items()):
        shape = "x".join(str(size) for size in getattr(array, "shape", ()))
        dtype = getattr(getattr(array, "dtype", None), "name", "")
        reopenable = "" if name in recipes else "  (array only)"
        rows.append((name, f"{shape} {dtype}{reopenable}".strip()))
    for dataset, recipe in recipes.items():
        rows.append((f"plot {dataset}", f"{recipe['spec'].kind.value}, {recipe['size']}"))
    return tuple(rows)


def _lineage_nodes(value: object) -> tuple[str | None, dict[str, Mapping[str, Any]]]:
    if not isinstance(value, Mapping) or set(value) != {
        "root", "nodes", "device_settings"
    }:
        raise ValueError(
            "figure lineage must contain root, nodes, and device_settings"
        )
    root, raw_nodes = value["root"], value["nodes"]
    if not isinstance(value["device_settings"], list) or not all(
        isinstance(item, Mapping) for item in value["device_settings"]
    ):
        raise TypeError("figure device settings must be an array of objects")
    if root is not None and not isinstance(root, str):
        raise TypeError("figure lineage root must be text or null")
    if not isinstance(raw_nodes, list):
        raise TypeError("figure lineage nodes must be an array")
    nodes: dict[str, Mapping[str, Any]] = {}
    for node in raw_nodes:
        entry = node if isinstance(node, Mapping) else {}
        if set(entry) != {
            "id", "event", "parents", "signals", "record", "event_record"
        }:
            raise ValueError("figure lineage node fields differ")
        node_id = entry["id"]
        if not isinstance(node_id, str) or not node_id or node_id in nodes:
            raise ValueError("figure lineage node IDs must be unique text")
        event = entry["event"]
        if not isinstance(event, Mapping) or set(event) != {"stream", "generation", "sequence"}:
            raise ValueError("figure lineage event fields differ")
        if (
            not isinstance(event["stream"], str)
            or not event["stream"]
            or not isinstance(event["generation"], str)
            or not event["generation"]
            or isinstance(event["sequence"], bool)
            or not isinstance(event["sequence"], int)
            or event["sequence"] < 0
        ):
            raise TypeError("figure lineage event identity is malformed")
        if not isinstance(entry["parents"], list) or not all(isinstance(item, str) for item in entry["parents"]):
            raise TypeError("figure lineage parents must be text IDs")
        if len(set(entry["parents"])) != len(entry["parents"]):
            raise ValueError("figure lineage parents must be unique")
        if not isinstance(entry["signals"], list) or not all(isinstance(item, str) for item in entry["signals"]):
            raise TypeError("figure lineage signals must be text")
        if not isinstance(entry["record"], Mapping):
            raise TypeError("figure lineage record must be an object")
        if not isinstance(entry["event_record"], Mapping):
            raise TypeError("figure lineage event_record must be an object")
        nodes[node_id] = entry
    if root is None:
        if nodes:
            raise ValueError("empty figure lineage cannot contain nodes")
        return None, nodes
    if root not in nodes or any(parent not in nodes for node in nodes.values() for parent in node["parents"]):
        raise ValueError("figure lineage refers to an unknown node")
    return root, nodes


def _lineage_tree(value: object) -> tuple[tuple[str, tuple], ...]:
    root, nodes = _lineage_nodes(value)
    if root is None:
        return ()
    visiting: set[str] = set()
    visited: set[str] = set()
    def branch(node_id: str) -> tuple[str, tuple]:
        if node_id in visiting:
            raise ValueError("figure lineage contains a cycle")
        visiting.add(node_id)
        node = nodes[node_id]
        event = node["event"]
        signals = ", ".join(node["signals"]) or event["stream"]
        children = tuple(branch(parent) for parent in node["parents"])
        visiting.remove(node_id)
        visited.add(node_id)
        return f"{signals} @{event['sequence']}", children
    tree = (branch(root),)
    if visited != set(nodes):
        raise ValueError("figure lineage contains nodes outside the root graph")
    return tree


def _lineage_rows(value: object) -> Rows:
    _root, nodes = _lineage_nodes(value)
    return tuple((node_id, _text(node["record"])) for node_id, node in nodes.items())


def _device_rows(value: object) -> Rows:
    _lineage_nodes(value)
    assert isinstance(value, Mapping)
    return tuple(
        (
            f"{item.get('device_key', 'device')} epoch {item.get('settings_epoch', '?')}",
            _text(item),
        )
        for item in value["device_settings"]
    )


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
        make_host: Callable[[Any, str, Mapping[str, object]], Any],
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
        self.recipe: dict[str, object] | None = None
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
    ) -> tuple[str, dict[str, object] | None, object | None, Any]:
        if not name:
            return "", None, None, None
        label = dict(description.datasets).get(str(name), str(name))
        host: Any = None
        try:
            plot_input, recipe = read_figure_plot(info, arrays, str(name))
            host = self._make_host(plot_input, label, recipe)
            return str(name), recipe, recipe["viewport"], host
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
            self.view.set_lineage_tree(description.lineage)
            self.view.set_datasets(
                description.datasets,
                description.dataset_keys[0] if description.datasets else "",
            )
        except BaseException:
            self._restore_previous_surface(previous)
            self._discard_candidate(candidate)
            raise

        name, recipe, _viewport, host = candidate
        self.path = resolved
        self.description = description
        self._info, self._arrays = info, arrays
        self._host = host
        self.dataset = str(name)
        self.recipe = recipe
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

        name, recipe, _viewport, host = candidate
        self._host = host
        self.dataset = str(name)
        self.recipe = recipe
        return self._retire_previous(previous)

    def _show_candidate(
        self,
        candidate: object,
        description: ArchiveDescription,
    ) -> None:
        name, recipe, _viewport, host = candidate
        self.view.show_figure(host)
        if recipe is not None:
            self.view.set_figure_size(str(recipe["size"]))
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
                self.view.set_lineage_tree(description.lineage)
                self.view.set_datasets(description.datasets, self.dataset)
            if self.recipe is not None:
                self.view.set_figure_size(str(self.recipe["size"]))
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
            if self.recipe is None
            else str(self.recipe["size"])
        )

        def resize() -> object:
            self._await(host.configure(size=str(size)))
            return str(size)

        def accepted(selected: object) -> None:
            resolved = str(selected)
            if self._host is host and self.recipe is not None:
                self.recipe = {**self.recipe, "size": resolved}
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
                    self.recipe = None
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
