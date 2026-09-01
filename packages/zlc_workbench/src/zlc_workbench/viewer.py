"""Opening a saved figure and reading what it was.

An archive already carries everything: the datasets with their axes, the
apparatus state, what was asked of it, which pulse drove it, which panel showed
what.  Until now it carried them for nobody -- `read_archive` returned a nested
JSON document, and reading a run six months later meant reading that document.

Two jobs, kept apart:

* :func:`describe_archive` projects the document into labelled rows.  It is
  pure, takes no Qt and no session, and is what makes "what was the apparatus
  doing" answerable in a notebook as well as a window.
* :class:`FigureViewerPresenter` publishes the saved typed datasets into a
  private Runtime plane and connects those rows to the same Panel engine used
  by TaskConsole.

Both stop short of scientific interpretation.  Logic and Device facts are
separated into operator-facing projections, while Raw retains the archive's
exact spelling for forensic inspection.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from zlc_plot import read_figure_plot
from zlc_plot.specs import semantic_spec

from zlc_data.figure_archive import read_archive


__all__ = ["ArchiveDescription", "FigureViewerPresenter", "describe_archive"]


Rows = tuple[tuple[str, object], ...]


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
    #: Plain node/edge data for the one Flow projection owned by the UI.
    flow: Mapping[str, tuple[Mapping[str, object], ...]]

    @property
    def dataset_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _label in self.datasets)


class _ArchiveDatasetProducer:
    """Publish one saved typed Dataset and its saved overlay as one event."""

    def __init__(
        self,
        serial: int,
        index: int,
        dataset: str,
        plot_input: object,
        path: Path,
    ) -> None:
        from zlc_plot.primitives import ImageFrame
        from zlc_runtime import DatasetOutputDeclaration

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset)).strip("._")
        safe = safe or f"dataset-{index + 1}"
        self.instance_id = f"figure-{serial}-{index + 1}"
        self.dataset = str(dataset)
        self.plot_input = plot_input
        self.path = Path(path)
        self._data_signal = f"@figure/{serial}/{safe}"
        self._overlay_signal = f"{self._data_signal}/overlay"
        overlay = plot_input.overlay if isinstance(plot_input, ImageFrame) else None
        self._status = None if overlay is None else self._status_snapshot(plot_input)
        outputs = [DatasetOutputDeclaration("data", "figure.dataset")]
        if self._status is not None:
            from zlc_plot import IMAGE_POINT_OVERLAY_CONTRACT

            outputs.append(
                DatasetOutputDeclaration("overlay", IMAGE_POINT_OVERLAY_CONTRACT)
            )
        self.dataset_output_declarations = tuple(outputs)

    def signal_key(self, output_name: str) -> str:
        if output_name == "data":
            return self._data_signal
        if output_name == "overlay" and self._status is not None:
            return self._overlay_signal
        raise KeyError(output_name)

    @property
    def data_signal(self) -> str:
        return self._data_signal

    @property
    def overlay_signal(self) -> str:
        return self._overlay_signal if self._status is not None else ""

    @staticmethod
    def _status_snapshot(frame: object) -> object | None:
        import numpy as np

        from zlc_data import (
            AxisId,
            AxisSpec,
            DatasetSchema,
            SITE,
            ValidityContract,
            ValueSchema,
            owned_snapshot_from_arrays,
        )
        from zlc_plot import PointStatus

        overlay = frame.overlay
        if overlay.status is not None:
            return overlay.status
        if overlay.static_statuses is None:
            return None
        image = frame.snapshot
        count = len(overlay.static_statuses)
        site_axis = AxisSpec(
            AxisId("figure.overlay.site"),
            "Site",
            SITE,
            count,
            coordinates=tuple(range(1, count + 1)),
        )
        schema = DatasetSchema(
            image.block.schema.repeat_axis,
            image.block.schema.point_table,
            image.block.schema.grid_topology,
            ValueSchema(
                (site_axis,),
                ValidityContract.components(site_axis.axis_id),
                np.dtype("?"),
                "1",
            ),
        )
        occupied = np.asarray(
            tuple(status is PointStatus.OCCUPIED for status in overlay.static_statuses),
            dtype=np.bool_,
        )
        valid = np.asarray(
            tuple(
                status in (PointStatus.EMPTY, PointStatus.OCCUPIED)
                for status in overlay.static_statuses
            ),
            dtype=np.bool_,
        )
        shape = schema.physical_shape
        return owned_snapshot_from_arrays(
            schema,
            np.broadcast_to(occupied, shape),
            image.block.revision,
            validity=np.broadcast_to(valid, shape),
            stream_generation=image.ref.stream_generation,
        )

    def publish(self, plane: object) -> object:
        from zlc_plot import (
            IMAGE_POINT_OVERLAY_GEOMETRY_RECORD,
            image_point_overlay_geometry,
        )
        from zlc_plot.primitives import ImageFrame
        from zlc_runtime import LiveDatasetOutput, MonitorCoverage

        snapshot = getattr(self.plot_input, "snapshot", self.plot_input)
        run_record: dict[str, object] = {
            "node": self.instance_id,
            "parameters": {
                "archive": str(self.path),
                "dataset": self.dataset,
            },
        }
        if isinstance(self.plot_input, ImageFrame):
            overlay = self.plot_input.overlay
            status = self._status
            status_axis = (
                None
                if status is None
                else status.block.schema.cell_schema.data_axes[0]
            )
            if status_axis is not None:
                point_ids = tuple(
                    overlay.point_ids
                    or tuple(f"point-{index + 1}" for index in range(overlay.count))
                )
                labels = tuple(
                    point_id if label is None else str(label)
                    for point_id, label in zip(
                        point_ids,
                        overlay.labels or (None,) * len(point_ids),
                        strict=True,
                    )
                )
                run_record[IMAGE_POINT_OVERLAY_GEOMETRY_RECORD] = (
                    image_point_overlay_geometry(
                        snapshot,
                        overlay.coordinates,
                        point_ids,
                        status_axis=status_axis,
                        labels=labels,
                    )
                )
        outputs = {
            "data": LiveDatasetOutput(
                self.dataset_output_declarations[0],
                snapshot,
                MonitorCoverage(
                    snapshot.block.schema.repeat_axis.size
                    * snapshot.block.schema.point_table.row_count,
                    snapshot.block.schema.repeat_axis.size
                    * snapshot.block.schema.point_table.row_count,
                    retain_at_terminal=True,
                ),
                run_record,
            )
        }
        if self._status is not None:
            status = self._status
            outputs["overlay"] = LiveDatasetOutput(
                self.dataset_output_declarations[1],
                status,
                MonitorCoverage(
                    status.block.schema.repeat_axis.size
                    * status.block.schema.point_table.row_count,
                    status.block.schema.repeat_axis.size
                    * status.block.schema.point_table.row_count,
                    retain_at_terminal=True,
                ),
                run_record,
            )
        plane.begin_generation(self)
        plane.commit_live(self, outputs)
        plane.seal_committed(self)
        return plane.latest_publication(self.data_signal)


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
    flow = _lineage_graph(sections["lineage"], source=source)
    return ArchiveDescription(
        name=str(info.get("name", "")),
        schema=str(info.get("schema", "")),
        datasets=datasets,
        flow=flow,
        tabs=(
            ("Plot", _plot_rows(arrays, recipes)),
            ("Logic", _logic_rows(sections["lineage"], source=source)),
            ("Devices", _device_rows(sections["lineage"], source=source)),
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


def _logic_name(node: Mapping[str, Any]) -> str:
    """Return one operator-facing Logic identity, never an archive node ID."""

    record = node["record"]
    raw_explicit = record.get("node")
    if raw_explicit is not None and not isinstance(raw_explicit, str):
        raise TypeError("figure Logic node identity must be text")
    explicit = str(raw_explicit or "").strip()
    if raw_explicit is not None and explicit != raw_explicit:
        raise ValueError("figure Logic node identity must be canonical text")
    if explicit:
        return explicit
    stream = str(node["event"]["stream"])
    if stream.startswith("@logic/"):
        owner, _separator, _output = stream[len("@logic/") :].rpartition("/")
        if owner:
            return owner
    return stream


def _signal_name(value: object) -> str:
    text = str(value)
    return text.rpartition("/")[2] if text.startswith("@logic/") else text


def _source_run_record(
    source: Mapping[str, object] | None,
) -> tuple[str, str, Mapping[str, object]] | None:
    if not isinstance(source, Mapping):
        return None
    record = source.get("run_record")
    if not isinstance(record, Mapping):
        return None
    raw_source_task = source.get("task")
    raw_record_node = record.get("node")
    if raw_source_task is not None and not isinstance(raw_source_task, str):
        raise TypeError("Figure source task must be text")
    if raw_record_node is not None and not isinstance(raw_record_node, str):
        raise TypeError("Task Figure Logic identity must be text")
    source_task = str(raw_source_task or "").strip()
    record_node = str(raw_record_node or "").strip()
    if raw_source_task is not None and source_task != raw_source_task:
        raise ValueError("Figure source task must be canonical text")
    if raw_record_node is not None and record_node != raw_record_node:
        raise ValueError("Task Figure Logic identity must be canonical text")
    if source_task and record_node and source_task != record_node:
        raise ValueError("Figure source task differs from its run-record Logic")
    task = source_task or record_node
    if not task:
        raise ValueError("Task Figure run record has no Logic identity")
    raw_output = source.get("report") or source.get("artifact") or source.get("signal")
    if raw_output is not None and not isinstance(raw_output, str):
        raise TypeError("Task Figure saved-result identity must be text")
    output = str(raw_output or "").strip()
    if raw_output is not None and output != raw_output:
        raise ValueError("Task Figure saved-result identity must be canonical text")
    if not output:
        raise ValueError("Task Figure source does not name its saved result")
    return task, output, record


_DEVICE_RECORD_FIELDS = frozenset(
    {"named_devices", "device_snapshots", "actual_devices", "device_settings"}
)


def _logic_record(value: object) -> object:
    """Remove device truth recursively; the Devices tab is its sole owner."""

    if isinstance(value, Mapping):
        return {
            str(key): _logic_record(item)
            for key, item in value.items()
            if key not in _DEVICE_RECORD_FIELDS
        }
    if isinstance(value, list):
        return [_logic_record(item) for item in value]
    return value


def _logic_rows(
    value: object,
    *,
    source: Mapping[str, object] | None = None,
) -> Rows:
    """Project saved Logic run snapshots without exposing event-N internals."""

    _root, nodes = _lineage_nodes(value)
    if not nodes:
        saved = _source_run_record(source)
        if saved is None:
            return ()
        name, output, record = saved
        projected = _logic_record(record)
        assert isinstance(projected, Mapping)
        return ((
            name,
            {
                "outputs": [output],
                **{
                    str(key): item
                    for key, item in projected.items()
                    if key != "node"
                },
            },
        ),)
    names = [_logic_name(node) for node in nodes.values()]
    repeated = {name for name, count in Counter(names).items() if count > 1}
    rows: list[tuple[str, object]] = []
    for node in nodes.values():
        name = _logic_name(node)
        event = node["event"]
        label = (
            f"{name} · sequence {event['sequence']}"
            if name in repeated
            else name
        )
        projected = _logic_record(node["record"])
        assert isinstance(projected, Mapping)
        record = {
            str(key): item
            for key, item in projected.items()
            if key != "node"
        }
        rows.append(
            (
                label,
                {
                    "outputs": [_signal_name(item) for item in node["signals"]],
                    **record,
                },
            )
        )
    return tuple(rows)


def _lineage_graph(
    value: object,
    *,
    source: Mapping[str, object] | None = None,
) -> dict[str, tuple[Mapping[str, object], ...]]:
    """Project the exact saved DAG as unique Logic/Device nodes and edges."""

    root, nodes = _lineage_nodes(value)
    if root is None:
        saved = _source_run_record(source)
        if saved is None:
            return {"nodes": (), "edges": ()}
        name, output, record = saved
        graph_nodes: list[Mapping[str, object]] = [
            {
                "id": "logic:source",
                "kind": "logic",
                "title": name,
                "subtitle": str(output),
                "root": True,
                "tooltip": f"Logic: {name}\nSaved result: {output}",
            }
        ]
        graph_edges: list[Mapping[str, object]] = []
        source_devices: dict[str, list[str]] = {}
        for role, device_key in _record_device_associations(record):
            source_devices.setdefault(device_key, []).append(role)
        for device_key, roles in source_devices.items():
            graph_nodes.append(
                {
                    "id": f"device:{device_key}",
                    "kind": "device",
                    "title": device_key,
                    "subtitle": ", ".join(roles),
                    "root": False,
                    "tooltip": (
                        f"Device: {device_key}\nRoles: {', '.join(roles)}\n"
                        f"Used by: {name}"
                    ),
                }
            )
            graph_edges.append(
                {
                    "source": f"device:{device_key}",
                    "target": "logic:source",
                    "kind": "device",
                    "label": ", ".join(roles),
                }
            )
        return {"nodes": tuple(graph_nodes), "edges": tuple(graph_edges)}
    visiting: set[str] = set()
    visited: set[str] = set()

    order: list[str] = []

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("figure lineage contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        node = nodes[node_id]
        for parent in node["parents"]:
            visit(parent)
        visiting.remove(node_id)
        visited.add(node_id)
        order.append(node_id)

    visit(root)
    if visited != set(nodes):
        raise ValueError("figure lineage contains nodes outside the root graph")
    graph_nodes: list[Mapping[str, object]] = []
    graph_edges: list[Mapping[str, object]] = []
    devices: dict[str, dict[str, object]] = {}
    device_edges: set[tuple[str, str, str]] = set()
    logic_names = [_logic_name(nodes[node_id]) for node_id in order]
    repeated_logic = {
        name for name, count in Counter(logic_names).items() if count > 1
    }
    for node_id in order:
        node = nodes[node_id]
        event = node["event"]
        signals = [_signal_name(item) for item in node["signals"]]
        graph_nodes.append(
            {
                "id": f"logic:{node_id}",
                "kind": "logic",
                "title": _logic_name(node),
                "subtitle": (
                    f"{', '.join(signals) or _signal_name(event['stream'])}"
                    + (
                        f" · sequence {event['sequence']}"
                        if _logic_name(node) in repeated_logic
                        else ""
                    )
                ),
                "root": node_id == root,
                "tooltip": (
                    f"Logic: {_logic_name(node)}\n"
                    f"Outputs: {', '.join(node['signals']) or event['stream']}"
                ),
            }
        )
        for parent in node["parents"]:
            graph_edges.append(
                {
                    "source": f"logic:{parent}",
                    "target": f"logic:{node_id}",
                    "kind": "causal",
                    "label": "",
                }
            )
        record = node["record"]
        event_record = node["event_record"]
        named = _named_devices(record)
        event_named = _named_devices(event_record)
        for role, device_key in event_named.items():
            previous = named.setdefault(role, device_key)
            if previous != device_key:
                raise ValueError(
                    f"device role {role!r} changes inside one Logic event"
                )
        associations = list(named.items())
        for association in (
            *_record_device_associations(record),
            *_record_device_associations(event_record),
        ):
            if association not in associations:
                associations.append(association)
        for role, device_key in associations:
            device = devices.setdefault(
                device_key,
                {"roles": [], "logic": []},
            )
            roles = device["roles"]
            consumers = device["logic"]
            assert isinstance(roles, list) and isinstance(consumers, list)
            if role not in roles:
                roles.append(role)
            logic_name = _logic_name(node)
            if logic_name not in consumers:
                consumers.append(logic_name)
            edge = (device_key, node_id, role)
            if edge not in device_edges:
                device_edges.add(edge)
                graph_edges.append(
                    {
                        "source": f"device:{device_key}",
                        "target": f"logic:{node_id}",
                        "kind": "device",
                        "label": role,
                    }
                )
    for device_key, facts in devices.items():
        roles = tuple(str(item) for item in facts["roles"])
        consumers = tuple(str(item) for item in facts["logic"])
        graph_nodes.append(
            {
                "id": f"device:{device_key}",
                "kind": "device",
                "title": device_key,
                "subtitle": ", ".join(roles),
                "root": False,
                "tooltip": (
                    f"Device: {device_key}\nRoles: {', '.join(roles)}\n"
                    f"Used by: {', '.join(consumers)}"
                ),
            }
        )
    return {"nodes": tuple(graph_nodes), "edges": tuple(graph_edges)}


def _record_devices(
    record: Mapping[str, object],
    *,
    named_devices: Mapping[str, str] | None = None,
) -> tuple[tuple[str, str, Mapping[str, object]], ...]:
    named = dict(named_devices or {})
    local_named = _named_devices(record)
    for role, device_key in local_named.items():
        previous = named.setdefault(role, device_key)
        if previous != device_key:
            raise ValueError(
                f"device role {role!r} names two different devices"
            )
    snapshots = record.get("device_snapshots")
    result: list[tuple[str, str, Mapping[str, object]]] = []
    if snapshots is not None:
        if not isinstance(snapshots, Mapping):
            raise TypeError("device_snapshots must be a mapping")
        for role, snapshot in snapshots.items():
            if not isinstance(role, str) or not role or role.strip() != role:
                raise ValueError("device snapshot roles must be canonical text")
            if role not in named:
                raise ValueError(
                    f"device snapshot role {role!r} has no stable device mapping"
                )
            if not isinstance(snapshot, Mapping):
                raise TypeError("device snapshot must be an object")
            result.append((role, named[role], snapshot))
    actual = record.get("actual_devices")
    if actual is not None:
        if not isinstance(actual, Mapping):
            raise TypeError("actual_devices must be a mapping")
        for device_key, snapshot in actual.items():
            if (
                not isinstance(device_key, str)
                or not device_key
                or device_key.strip() != device_key
            ):
                raise ValueError("actual device keys must be canonical text")
            if not isinstance(snapshot, Mapping):
                raise TypeError("actual device snapshot must be an object")
            existing = next(
                (item for item in result if item[1] == device_key),
                None,
            )
            if existing is not None:
                if dict(existing[2]) != dict(snapshot):
                    raise ValueError(
                        f"device {device_key!r} has conflicting snapshots"
                    )
                continue
            result.append((device_key, device_key, snapshot))
    return tuple(result)


def _named_devices(record: Mapping[str, object]) -> dict[str, str]:
    raw = record.get("named_devices")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("named_devices must be a mapping")
    result: dict[str, str] = {}
    for role, device_key in raw.items():
        if (
            not isinstance(role, str)
            or not role
            or role.strip() != role
            or not isinstance(device_key, str)
            or not device_key
            or device_key.strip() != device_key
        ):
            raise ValueError("named device roles and keys must be canonical text")
        result[role] = device_key
    return result


def _record_device_associations(
    record: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    result = list(_named_devices(record).items())
    actual = record.get("actual_devices")
    if actual is not None:
        if not isinstance(actual, Mapping):
            raise TypeError("actual_devices must be a mapping")
        for key in actual:
            if not isinstance(key, str) or not key or key.strip() != key:
                raise ValueError("actual device keys must be canonical text")
            if not any(existing == key for _role, existing in result):
                result.append((key, key))
    return tuple(result)


def _device_rows(
    value: object,
    *,
    source: Mapping[str, object] | None = None,
) -> Rows:
    _root, nodes = _lineage_nodes(value)
    assert isinstance(value, Mapping)
    devices: dict[str, dict[str, object]] = {}

    def device(device_key: str) -> dict[str, object]:
        return devices.setdefault(
            str(device_key),
            {"roles": [], "used_by": [], "snapshots": [], "settings": []},
        )

    records: list[
        tuple[str, int | None, str, Mapping[str, object], Mapping[str, str]]
    ] = []
    for node in nodes.values():
        run_record = node["record"]
        event_record = node["event_record"]
        named = _named_devices(run_record)
        for role, device_key in _named_devices(event_record).items():
            previous = named.setdefault(role, device_key)
            if previous != device_key:
                raise ValueError(
                    f"device role {role!r} changes inside one Logic event"
                )
        logic = _logic_name(node)
        sequence = int(node["event"]["sequence"])
        records.extend(
            (
                (logic, sequence, "run", run_record, named),
                (logic, sequence, "event", event_record, named),
            )
        )
    if not nodes:
        saved = _source_run_record(source)
        if saved is not None:
            records.append(
                (saved[0], None, "task run", saved[2], _named_devices(saved[2]))
            )
    for logic, sequence, scope, record, named in records:
        for role, raw_key in _record_device_associations(record):
            facts = device(str(raw_key))
            roles = facts["roles"]
            used_by = facts["used_by"]
            assert isinstance(roles, list) and isinstance(used_by, list)
            if str(role) not in roles:
                roles.append(str(role))
            if logic not in used_by:
                used_by.append(logic)
        for role, device_key, snapshot in _record_devices(
            record,
            named_devices=named,
        ):
            facts = device(device_key)
            roles = facts["roles"]
            used_by = facts["used_by"]
            snapshots = facts["snapshots"]
            assert isinstance(roles, list)
            assert isinstance(used_by, list)
            assert isinstance(snapshots, list)
            if role not in roles:
                roles.append(role)
            if logic not in used_by:
                used_by.append(logic)
            candidate = {
                "logic": logic,
                "scope": scope,
                **({} if sequence is None else {"sequence": sequence}),
                "snapshot": dict(snapshot),
            }
            if candidate not in snapshots:
                snapshots.append(candidate)
    for item in value["device_settings"]:
        raw_key = item.get("device_key")
        if (
            not isinstance(raw_key, str)
            or not raw_key
            or raw_key.strip() != raw_key
        ):
            raise ValueError("device setting record has no canonical device_key")
        key = raw_key
        settings = device(key)["settings"]
        assert isinstance(settings, list)
        settings.append(dict(item))
    rows = []
    for device_key, facts in devices.items():
        snapshots = facts["snapshots"]
        settings = facts["settings"]
        assert isinstance(snapshots, list) and isinstance(settings, list)
        rows.append(
            (
                device_key,
                {
                    "roles": facts["roles"],
                    "used_by": facts["used_by"],
                    "snapshots": snapshots,
                    "settings_epochs": settings,
                },
            )
        )
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
        panel_presenter: object,
        signal_plane: object,
    ) -> None:
        self.view = view
        self._run_off_thread = run_off_thread
        self._close_worker = close_worker
        self._request_close = request_close
        self._panel_presenter = panel_presenter
        self._signal_plane = signal_plane
        self._archive_producers: tuple[_ArchiveDatasetProducer, ...] = ()
        self._archive_serial = 0
        self._runtime_closed = False
        self.timer: object | None = None
        self.path: Path | None = None
        self.description: ArchiveDescription | None = None
        self.panels = panel_presenter.panels
        self._active_panel_id = ""
        self._busy = False
        self._close_requested = False
        self._closed = False
        self._connect()

    def _connect(self) -> None:
        self.view.path_committed.connect(self.open)
        # ConsolePresenter remains the sole panel mutation owner.  Viewer only
        # remembers which card the operator touched so its one global Save
        # image action targets that card.
        self.view.add_panel_requested.connect(self._remember_added_panel)
        self.view.panel_state_changed.connect(self._remember_panel)
        self.view.panel_edit_requested.connect(self._remember_panel)
        self.view.panel_remove_requested.connect(self._remember_removed_panel)
        self.view.save_image_requested.connect(self.save_image)

    def _remember_added_panel(self, _kind: object) -> None:
        self._active_panel_id = next(reversed(self.panels), "")

    def _remember_panel(self, panel_id: str, *_unused: object) -> None:
        if str(panel_id) in self.panels:
            self._active_panel_id = str(panel_id)

    def _remember_removed_panel(self, panel_id: str) -> None:
        if self._active_panel_id == str(panel_id):
            self._active_panel_id = next(reversed(self.panels), "")

    def open(self, path: str) -> None:
        """Submit one complete archive candidate without blocking the Qt owner.

        The accepted archive, path, dataset and host change together only after
        reading, rebuilding and configuring the candidate all succeed.  A bad
        path therefore cannot tear down the last figure that opened correctly.
        """

        self._open_runtime(path)

    def _open_runtime(self, path: str) -> None:
        requested = Path(path)
        serial = self._archive_serial + 1

        def prepare() -> object:
            import zlc_plot

            resolved = requested.resolve()
            info, arrays = read_archive(resolved)
            description = describe_archive(info, arrays)
            loaded = []
            for index, key in enumerate(description.dataset_keys):
                plot_input, recipe = read_figure_plot(info, arrays, key)
                described = None
                if index == 0:
                    # Only the default card restores a saved DisplayDescription.
                    # Other datasets become ordinary Runtime signals and are
                    # composed only if the operator chooses them later.
                    host = zlc_plot.open_figure_host(
                        plot_input,
                        recipe,
                        device_pixel_ratio=float(self.view.device_pixel_ratio()),
                    )
                    try:
                        described = self._await(host.describe_display())
                        described = getattr(described, "value", described)
                    finally:
                        self._close_host(host)
                loaded.append((key, plot_input, recipe, described))
            return resolved, description, tuple(loaded), serial

        self._submit(
            f"opening {requested.name}…",
            prepare,
            self._accept_runtime_archive,
            f"cannot open {requested.name}",
        )

    def _accept_runtime_archive(self, result: object) -> None:
        from zlc_plot.specs import FacetGridPlot, semantic_spec

        resolved, description, loaded, serial = result
        panel_presenter = self._panel_presenter
        plane = self._signal_plane
        if panel_presenter is None or plane is None:
            raise RuntimeError("FigureViewer has no Runtime panel composition")
        previous_panels = tuple(panel_presenter.panels)
        producers: list[_ArchiveDatasetProducer] = []
        published: list[tuple[object, object, object, object, object]] = []
        new_panel_id = ""
        try:
            for index, (key, plot_input, recipe, described) in enumerate(loaded):
                producer = _ArchiveDatasetProducer(
                    serial,
                    index,
                    key,
                    plot_input,
                    resolved,
                )
                producers.append(producer)
                publication = producer.publish(plane)
                published.append(
                    (producer, plot_input, recipe, described, publication)
                )
            if published:
                producer, plot_input, _recipe, described, publication = published[0]
                spec = described.spec
                cell_kind = (
                    semantic_spec(spec).kind.value
                    if isinstance(spec, FacetGridPlot)
                    else ""
                )
                semantic = {
                    str(name): value
                    for name, value in described.semantics.values.items()
                    if str(name) != "kind"
                }
                label = dict(description.datasets).get(
                    producer.dataset,
                    producer.dataset,
                )
                binding = panel_presenter.add_panel(
                    producer.data_signal,
                    getattr(plot_input, "snapshot", plot_input),
                    title=label,
                    kind=spec.kind.value,
                    size=described.size,
                    semantic=semantic,
                    display=dict(described.display_state.values),
                    fit=dict(described.fit),
                    overlay_signal=producer.overlay_signal,
                    initial_publication=publication,
                )
                new_panel_id = binding.panel_id
                panel_presenter.restore_panel_description(
                    binding.panel_id,
                    described,
                )
                self._active_panel_id = binding.panel_id
        except BaseException:
            if new_panel_id:
                panel_presenter.remove_panel(new_panel_id)
            for producer in producers:
                try:
                    plane.retire(producer)
                except BaseException:
                    pass
            raise

        for panel_id in previous_panels:
            panel_presenter.remove_panel(panel_id)
        for producer in self._archive_producers:
            plane.retire(producer)
        self._archive_producers = tuple(producers)
        self._archive_serial = int(serial)
        self.path = resolved
        self.description = description
        self.view.set_title(description.name or resolved.stem)
        self.view.set_path(str(resolved))
        self.view.set_archive_info(description.tabs, description.flow)
        self.view.set_status(
            "opened; choose a signal in Setting"
            if not published
            else f"showing {published[0][0].data_signal}"
        )
        panel_presenter.beat()

    def beat(self) -> None:
        self._panel_presenter.beat()

    def commit_surfaces(self) -> None:
        self._panel_presenter.commit_surfaces()

    def add_panel(self, kind: str) -> None:
        binding = self._panel_presenter.add_selected_panel(str(kind))
        if binding is not None:
            self._active_panel_id = binding.panel_id

    def update_panel(self, panel_id: str, patch: object) -> None:
        self._active_panel_id = str(panel_id)
        self._panel_presenter.update_panel_state(str(panel_id), patch)

    def remove_panel(self, panel_id: str) -> None:
        self._panel_presenter.remove_panel(str(panel_id))
        self._active_panel_id = next(reversed(self.panels), "")

    def reorder_panels(self, order: object) -> None:
        self._panel_presenter.reorder_panels(tuple(order))

    def edit_panel(self, panel_id: str) -> object | None:
        self._active_panel_id = str(panel_id)
        return self._panel_presenter.edit_panel(str(panel_id))

    def close_panel_editor(self, panel_id: str) -> None:
        self._panel_presenter.close_panel_editor(str(panel_id))

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
        self._panel_presenter.update_panel_state(
            str(panel_id),
            {"size": str(size)},
        )

    def save_image(self) -> None:
        """Write the active shared Panel exactly as drawn beside its archive."""

        from zlc_durable import unique_path

        binding = self.panels.get(self._active_panel_id)
        if binding is None and self.panels:
            binding = next(reversed(self.panels.values()))
        if binding is None or self.path is None or binding.host is None:
            self.view.set_status("there is no figure to save", error=True)
            return
        host = binding.host
        save = getattr(host, "save", None)
        if not callable(save):
            self.view.set_status("this figure cannot save itself", error=True)
            return
        path = self.path
        dataset = binding.state.signal.rsplit("/", 1)[-1]

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

    def close(self) -> bool:
        """Close the shared Panel engine, archive signals, then IO worker."""

        self._close_requested = True
        if self._closed:
            return True
        if self._busy:
            self.view.set_status("closing after the current operation…")
            return False
        timer = self.timer
        if not self._panel_presenter.close():
            self._panel_presenter.beat()
            self.view.set_status("closing saved panels…")
            self._request_close()
            return False
        stop = getattr(timer, "stop", None)
        if callable(stop):
            stop()
        if not self._runtime_closed:
            for producer in self._archive_producers:
                try:
                    self._signal_plane.retire(producer)
                except (LookupError, RuntimeError):
                    pass
            self._archive_producers = ()
            self._signal_plane.close()
            self._runtime_closed = True
        if not self._close_worker():
            self._request_close()
            return False
        self._closed = True
        return True
