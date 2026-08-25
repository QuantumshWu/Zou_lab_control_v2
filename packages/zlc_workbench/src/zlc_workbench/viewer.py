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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from zlc_plot import (
    DEFAULTS,
    decode_plot_recipe,
    describe_semantics,
    encode_plot_recipe,
    paints_image_surface,
    read_figure_plot,
)
from zlc_plot.specs import semantic_spec

from zlc_data.figure_archive import read_archive
from .panel_catalog import GRID_CELL_KINDS, panel_kind_choices, task_console_fitting_spec
from .panel_state import (
    PanelState,
    panel_data_shape,
    panel_state_from_description,
    panel_surface_from_description,
    project_panel_state,
)


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
    ) -> None:
        self.view = view
        self._make_host = make_host
        self._run_off_thread = run_off_thread
        self._close_worker = close_worker
        self._request_close = request_close
        self.path: Path | None = None
        self.description: ArchiveDescription | None = None
        #: The archive as read, kept so switching datasets does not re-read the
        #: file: an archive is immutable once written.
        self._info: Mapping[str, Any] = {}
        self._arrays: Mapping[str, Any] = {}
        self.panels: dict[str, dict[str, object]] = {}
        self._panel_serial = 0
        self._active_panel_id = ""
        self._retired_host: Any = None
        self._retirement_pending = False
        self._busy = False
        self._close_requested = False
        self._closed = False
        self.view.set_panel_sizes(
            DEFAULTS.layout.size_names, DEFAULTS.layout.default_preset
        )
        self.view.set_panel_kinds(panel_kind_choices())
        self.view.set_grid_cell_kinds(
            tuple(kind.value for kind in GRID_CELL_KINDS)
        )
        self._connect()

    def _connect(self) -> None:
        self.view.path_committed.connect(self.open)
        self.view.add_panel_requested.connect(self.add_panel)
        self.view.panel_state_changed.connect(self.update_panel)
        self.view.panel_remove_requested.connect(self.remove_panel)
        self.view.panel_edit_requested.connect(self.edit_panel)
        self.view.panel_order_committed.connect(self.reorder_panels)
        self.view.panel_editor_closed.connect(self.close_panel_editor)
        self.view.save_image_requested.connect(self.save_image)

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

    def add_panel(self, kind: str) -> None:
        description = self.description
        if description is None or not description.datasets:
            self.view.set_status("open a Figure before adding a panel", error=True)
            return
        active = self.panels.get(self._active_panel_id)
        active_dataset = "" if active is None else str(active["dataset"])
        dataset = (
            active_dataset
            if active_dataset in description.dataset_keys
            else description.dataset_keys[0]
        )

        def prepare() -> object:
            return self._prepare_dataset(
                self._info,
                self._arrays,
                description,
                dataset,
                kind=str(kind),
            )

        def accepted(candidate: object) -> None:
            try:
                panel_id = self._install_panel_candidate(candidate, description)
            except BaseException:
                self._discard_candidate(candidate)
                raise
            self.view.set_status(f"added {panel_id} as {kind}")

        self._submit(
            f"adding {kind} panel…",
            prepare,
            accepted,
            f"cannot add {kind} panel",
        )

    def update_panel(self, panel_id: str, patch: object) -> None:
        key = str(panel_id)
        record = self.panels.get(key)
        if record is None or not isinstance(patch, Mapping):
            return
        state = record["state"]
        assert isinstance(state, PanelState)
        changes = dict(patch)
        if set(changes) <= {"title"}:
            state = replace(state, title=str(changes.get("title", state.title)))
            record["state"] = state
            self.view.set_panel_projection(key, state, record["surface"])
            return
        if set(changes) <= {"size"}:
            self.resize_panel(key, str(changes.get("size", state.size)))
            return
        if set(changes) <= {"signal"}:
            self._replace_panel_dataset(
                key, str(changes.get("signal") or state.signal), state.cell_kind
            )
            return
        if set(changes) <= {"cell_kind"} and state.kind == "facet_grid":
            self._replace_panel_dataset(
                key,
                state.signal,
                str(changes.get("cell_kind") or ""),
            )
            return
        if len(changes) == 1 and next(iter(changes)) in {
            "semantic",
            "display",
            "fit",
        }:
            section = next(iter(changes))
            updates = dict(changes[section])
            host = record["host"]

            def apply() -> object:
                operation = self._await(
                    host.configure(
                        **(
                            {"semantic": updates}
                            if section == "semantic"
                            else {"parameters": updates}
                            if section == "display"
                            else {"fit": updates, "fit_live": False}
                        )
                    )
                )
                return getattr(operation, "value", operation)

            def accepted(description: object) -> None:
                current = record["state"]
                assert isinstance(current, PanelState)
                values = dict(getattr(current, section))
                values.update(updates)
                candidate_state = replace(current, **{section: values})
                surface = panel_surface_from_description(
                    candidate_state,
                    description,
                    description.semantics,
                    description.fit_models,
                )
                candidate_state = panel_state_from_description(
                    candidate_state, surface
                )
                surface.update(
                    self._panel_surface(
                        record["plot_input"], candidate_state, surface
                    )
                )
                record["state"] = candidate_state
                record["surface"] = surface
                self.view.set_panel_projection(key, candidate_state, surface)
                self.view.update_panel_editor(
                    key, self._editor_projection(record)
                )

            self._submit(
                f"updating {key} {section}…",
                apply,
                accepted,
                f"cannot update {key} {section}",
            )
            return
        self.view.set_panel_status(
            key,
            "Use Edit for semantic, display, and fit changes.",
            error=True,
        )

    def _replace_panel_dataset(
        self, panel_id: str, dataset: str, cell_kind: str
    ) -> None:
        record = self.panels.get(str(panel_id))
        description = self.description
        if record is None or description is None:
            return
        state = record["state"]
        assert isinstance(state, PanelState)

        def prepare() -> object:
            return self._prepare_dataset(
                self._info,
                self._arrays,
                description,
                str(dataset),
                kind=state.kind,
                cell_kind=str(cell_kind),
                size=state.size,
            )

        def accepted(candidate: object) -> bool:
            previous = record["host"]
            try:
                self._install_panel_candidate(
                    candidate, description, panel_id=panel_id
                )
            except BaseException:
                self._discard_candidate(candidate)
                raise
            return self._retire_previous(previous)

        self._submit(
            f"rebuilding {panel_id}…",
            prepare,
            accepted,
            f"cannot rebuild {panel_id}",
        )

    def remove_panel(self, panel_id: str) -> None:
        if self._busy or self._retired_host is not None:
            self.view.set_status("viewer is busy; panel was not removed", error=True)
            return
        key = str(panel_id)
        record = self.panels.pop(key, None)
        if record is None:
            return
        self.view.remove_panel(key)
        self._active_panel_id = next(reversed(self.panels), "")
        self._busy = True
        self._start_retirement(
            record["host"],
            f"cannot remove {key}",
            finished=self._finish_operation,
        )

    def reorder_panels(self, order: object) -> None:
        wanted = [str(key) for key in tuple(order) if str(key) in self.panels]
        wanted += [key for key in self.panels if key not in wanted]
        self.panels = {key: self.panels[key] for key in wanted}
        self.view.set_panel_order(wanted)

    def edit_panel(self, panel_id: str) -> object | None:
        record = self.panels.get(str(panel_id))
        description = self.description
        if record is None or description is None:
            return None
        state = record["state"]
        assert isinstance(state, PanelState)
        projection = self._editor_projection(record)
        self.view.open_panel_editor(
            str(panel_id), projection, title=f"Edit · {state.title}"
        )
        self._active_panel_id = str(panel_id)
        return projection

    def _editor_projection(
        self, record: Mapping[str, object]
    ) -> dict[str, object]:
        description = self.description
        if description is None:
            raise RuntimeError("viewer has no accepted archive")
        state = record["state"]
        assert isinstance(state, PanelState)
        return {
            "state": state.document(),
            "parameter_surface": record["surface"],
            "signal_options": (
                (
                    "this archive",
                    tuple((label, key) for key, label in description.datasets),
                ),
            ),
            "overlay_signal_options": (),
            "live": False,
        }

    def close_panel_editor(self, panel_id: str) -> None:
        self.view.close_panel_editor(str(panel_id))

    def _prepare_dataset(
        self,
        info: Mapping[str, Any],
        arrays: Mapping[str, Any],
        description: ArchiveDescription,
        name: str,
        *,
        kind: str | None = None,
        cell_kind: str = "",
        size: str | None = None,
    ) -> tuple[
        str,
        dict[str, object] | None,
        Any,
        object | None,
        object | None,
    ]:
        if not name:
            return "", None, None, None, None
        label = dict(description.datasets).get(str(name), str(name))
        host: Any = None
        try:
            plot_input, saved_recipe = read_figure_plot(info, arrays, str(name))
            recipe = (
                saved_recipe
                if kind is None
                else self._alternate_recipe(
                    plot_input,
                    str(kind),
                    str(cell_kind),
                    size=size,
                )
            )
            host = self._make_host(plot_input, label, recipe)
            described = self._await(host.describe_display())
            described = getattr(described, "value", described)
            return (
                str(name),
                recipe,
                host,
                plot_input,
                described,
            )
        except BaseException:
            if host is not None:
                self._close_host(host)
            raise

    @staticmethod
    def _alternate_recipe(
        plot_input: object,
        kind: str,
        cell_kind: str = "",
        *,
        size: str | None = None,
    ) -> dict[str, object]:
        snapshot = getattr(plot_input, "snapshot", plot_input)
        schema = getattr(getattr(snapshot, "block", None), "schema", None)
        if schema is None:
            raise TypeError("saved dataset has no typed schema")
        spec = task_console_fitting_spec(schema, kind, cell_kind)
        if spec is None:
            raise ValueError(f"saved dataset cannot be drawn as {kind!r}")
        encoded = encode_plot_recipe(
            spec,
            parameters={},
            size=DEFAULTS.layout.default_preset if size is None else str(size),
        )
        recipe = decode_plot_recipe(encoded)
        recipe.pop("overlay")
        return recipe

    @staticmethod
    def _panel_state(
        panel_id: str,
        dataset: str,
        title: str,
        recipe: Mapping[str, object],
        plot_input: object,
    ) -> PanelState:
        spec = recipe["spec"]
        snapshot = getattr(plot_input, "snapshot", plot_input)
        semantics = describe_semantics(snapshot.block.schema, spec)
        kind = str(spec.kind.value)
        cell_kind = (
            semantic_spec(spec).kind.value
            if kind == "facet_grid"
            else ""
        )
        return PanelState(
            signal=str(dataset),
            kind=kind,
            cell_kind=cell_kind,
            size=str(recipe["size"]),
            interval_ms=400,
            title=str(title or panel_id),
            semantic={
                str(name): value
                for name, value in semantics.values.items()
                if str(name) != "kind"
            },
            display=dict(recipe.get("parameters", {})),
            fit=dict(recipe.get("fit", {})),
        )

    @staticmethod
    def _panel_surface(
        plot_input: object,
        state: PanelState,
        surface: Mapping[str, object],
    ) -> dict[str, object]:
        snapshot = getattr(plot_input, "snapshot", plot_input)
        schema = snapshot.block.schema
        base = task_console_fitting_spec(
            schema, state.kind, state.cell_kind
        )
        resolved = None
        if base is not None:
            resolved, _semantic, _display = project_panel_state(
                schema, base, state
            )
        return {
            **panel_data_shape(schema, surface),
            "science_locked": False,
            "paints_images": (
                False if resolved is None else paints_image_surface(resolved)
            ),
        }

    def _install_panel_candidate(
        self,
        candidate: object,
        description: ArchiveDescription,
        *,
        panel_id: str | None = None,
    ) -> str:
        name, recipe, host, plot_input, described = candidate
        if recipe is None or host is None or plot_input is None:
            raise ValueError("saved array has no plot recipe")
        if panel_id is None:
            self._panel_serial += 1
            panel_id = f"saved-panel-{self._panel_serial}"
        label = dict(description.datasets).get(str(name), str(name) or "figure")
        state = self._panel_state(
            panel_id, str(name), label, recipe, plot_input
        )
        previous = self.panels.get(panel_id)
        if previous is not None:
            previous_state = previous["state"]
            assert isinstance(previous_state, PanelState)
            state = replace(
                state,
                title=previous_state.title,
                size=previous_state.size,
            )
        if described is None:
            raise ValueError("saved plot host has no display description")
        surface = panel_surface_from_description(
            state,
            described,
            described.semantics,
            described.fit_models,
        )
        state = panel_state_from_description(state, surface)
        surface.update(self._panel_surface(plot_input, state, surface))
        if previous is None:
            self.view.add_panel(panel_id, state.title)
        try:
            self.view.set_panel_datasets(
                panel_id, description.datasets, str(name)
            )
            self.view.set_panel_projection(panel_id, state, surface)
            self.view.show_panel(panel_id, host)
        except BaseException:
            if previous is None:
                self.view.remove_panel(panel_id)
            else:
                self.view.set_panel_datasets(
                    panel_id,
                    description.datasets,
                    str(previous["dataset"]),
                )
                self.view.set_panel_projection(
                    panel_id, previous["state"], previous["surface"]
                )
                self.view.show_panel(panel_id, previous["host"])
            raise
        self.panels[panel_id] = {
            "dataset": str(name),
            "recipe": dict(recipe),
            "host": host,
            "plot_input": plot_input,
            "state": state,
            "surface": surface,
        }
        self._active_panel_id = panel_id
        self.view.set_panel_order(tuple(self.panels))
        return panel_id

    def _clear_panel_views(self) -> tuple[object, ...]:
        hosts = tuple(record["host"] for record in self.panels.values())
        for panel_id in tuple(self.panels):
            self.view.remove_panel(panel_id)
        self.panels.clear()
        self._active_panel_id = ""
        return hosts

    def _accept_archive(self, result: object) -> bool:
        resolved, info, arrays, description, candidate = result
        old_panels = dict(self.panels)
        previous = tuple(record["host"] for record in old_panels.values())
        self._panel_serial += 1
        new_panel_id = f"saved-panel-{self._panel_serial}"
        try:
            if candidate[2] is not None:
                self._install_panel_candidate(
                    candidate, description, panel_id=new_panel_id
                )
            # The new plot must mount before any visible archive identity is
            # replaced.  A rejected host therefore leaves both the old pixels
            # and the old File/info projection untouched.
            self.view.set_title(description.name or resolved.stem)
            self.view.set_path(str(resolved))
            self.view.set_info(description.tabs)
            self.view.set_lineage_tree(description.lineage)
        except BaseException:
            self.view.remove_panel(new_panel_id)
            self.panels.pop(new_panel_id, None)
            self._discard_candidate(candidate)
            raise

        for panel_id in old_panels:
            self.view.remove_panel(panel_id)
            self.panels.pop(panel_id, None)
        self.view.set_panel_order(tuple(self.panels))

        self.path = resolved
        self.description = description
        self._info, self._arrays = info, arrays
        if self.panels:
            self.view.set_status(
                f"showing {next(iter(self.panels.values()))['dataset']}"
            )
        else:
            self.view.set_status(
                "opened; its arrays were saved without axes, so they cannot be replotted"
            )
        return self._retire_previous(previous)

    def _discard_candidate(self, candidate: object) -> None:
        host = candidate[2]
        if host is not None and all(
            host is not record["host"] for record in self.panels.values()
        ):
            self._start_retirement(host, "cannot retire rejected figure")

    def _retire_previous(self, previous: object | None) -> bool:
        active_hosts = tuple(record["host"] for record in self.panels.values())
        if (
            previous is None
            or previous == ()
            or any(previous is host for host in active_hosts)
        ):
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

    def resize_panel(self, panel_id: str, size: str) -> None:
        record = self.panels.get(str(panel_id))
        if record is None:
            return
        host = record["host"]
        state = record["state"]
        assert isinstance(state, PanelState)
        previous_size = state.size

        def resize() -> object:
            self._await(host.configure(size=str(size)))
            return str(size)

        def accepted(selected: object) -> None:
            resolved = str(selected)
            if record.get("host") is not host:
                return
            recipe = {**dict(record["recipe"]), "size": resolved}
            state = replace(record["state"], size=resolved)
            record["recipe"] = recipe
            record["state"] = state
            self.view.set_panel_projection(
                str(panel_id), state, record["surface"]
            )
            self.view.set_status(f"resized {panel_id} to {resolved}")

        def rejected(error: BaseException) -> None:
            self.view.set_panel_projection(
                str(panel_id), record["state"], record["surface"]
            )
            self.view.set_status(f"cannot resize: {error}", error=True)

        self._submit(
            f"resizing to {size}…",
            resize,
            accepted,
            "cannot resize",
            on_failure=rejected,
        )

    def save_image(self) -> None:
        """Write the figure as it is drawn, beside the archive it came from."""

        from zlc_durable import unique_path

        record = self.panels.get(self._active_panel_id)
        if record is None or self.path is None:
            self.view.set_status("there is no figure to save", error=True)
            return
        host = record["host"]
        save = getattr(host, "save", None)
        if not callable(save):
            self.view.set_status("this figure cannot save itself", error=True)
            return
        path, dataset = self.path, str(record["dataset"])

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
        if isinstance(host, tuple):
            for item in host:
                cls._close_host(item)
            return
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
        if self.panels:
            hosts = self._clear_panel_views()
            self._busy = True
            self.view.set_status("closing saved panels…")
            self._start_retirement(
                hosts,
                "cannot close saved panels",
                finished=self._finish_operation,
            )
            return False
        if not self._close_worker():
            self._request_close()
            return False
        self._closed = True
        return True
