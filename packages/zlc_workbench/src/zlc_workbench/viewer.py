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
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from math import prod
from pathlib import Path
import re
from typing import Any

import numpy as np
from zlc_plot import read_figure_plot

from zlc_data.figure_archive import read_archive


__all__ = ["ArchiveDescription", "FigureViewerPresenter", "describe_archive"]


Rows = tuple[tuple[str, object], ...]


_MANUAL_DTYPES = tuple(
    np.dtype(value).str
    for value in (
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
)
_MANUAL_ROLE_CHOICES = (
    "primary-index",
    "scan-point",
    "readout-event",
    "spatial-x",
    "spatial-y",
    "spectral",
    "histogram-bin",
    "site",
    "component",
    "scalar",
)
_POINT_ROWS = "zlc_data.point-ordinal"


def _axis_coordinates(axis: object) -> object:
    if axis.coordinates is not None:
        return axis.coordinates
    return range(int(axis.index_origin), int(axis.index_origin) + int(axis.size))


def _manual_version() -> str:
    try:
        return version("zou-lab-control")
    except PackageNotFoundError:
        return "unknown"


def _new_manual_snapshot() -> object:
    """A useful Curve Dataset, not an empty shape the operator must repair."""

    from zlc_data import (
        AxisId,
        AxisSpec,
        DatasetSchema,
        GridTopology,
        PointColumn,
        PointTable,
        REPEAT,
        SCAN_POINT,
        ValueSchema,
        owned_snapshot_from_arrays,
    )

    count = 16
    x_id = AxisId("manual.x")
    x_coordinates = tuple(range(count))
    schema = DatasetSchema(
        AxisSpec(AxisId("manual.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(
            count,
            (
                PointColumn(
                    x_id,
                    "x",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    x_coordinates,
                ),
            ),
        ),
        GridTopology(
            (x_id,),
            (x_coordinates,),
            tuple((index,) for index in range(count)),
        ),
        ValueSchema.scalar(np.dtype("<f8")),
    )
    return owned_snapshot_from_arrays(
        schema,
        np.zeros(schema.physical_shape, dtype=schema.cell_schema.dtype),
        0,
        validity=np.ones(schema.physical_shape, dtype=np.bool_),
    )


def _draft_from_snapshot(
    snapshot: object,
    *,
    editor_id: str,
    name: str,
    note: str,
    source_text: str,
    source_path: Path | None,
    source_dataset: str,
    source_lineage: Mapping[str, object],
    source_document: Mapping[str, object],
    recipe: Mapping[str, object] | None,
    described: object | None,
    overlay: object | None,
) -> dict[str, object]:
    from zlc_data import expand_snapshot_validity

    schema = snapshot.block.schema
    selected_axis = (
        str(schema.grid_topology.dimension_ids[0])
        if schema.grid_topology is not None
        else str(schema.point_table.columns[0].coordinate_id)
        if schema.point_table.columns
        else str(
            next(
                (
                    axis.axis_id
                    for axis in schema.cell_schema.data_axes
                    if str(axis.role) != "scalar"
                ),
                schema.repeat_axis.axis_id,
            )
        )
    )
    draft = {
        "editor_id": str(editor_id),
        "name": str(name),
        "initial_name": str(name),
        "dtype": schema.cell_schema.dtype,
        "unit": schema.cell_schema.value_unit,
        "note": str(note),
        "initial_note": str(note),
        "source_text": str(source_text),
        "source_path": source_path,
        "source_dataset": str(source_dataset),
        "source_lineage": deepcopy(dict(source_lineage)),
        "source_document": deepcopy(dict(source_document)),
        "source_snapshot": snapshot,
        "source_overlay": overlay,
        "recipe": None if recipe is None else dict(recipe),
        "described": described,
        "repeat_axis": schema.repeat_axis,
        "point_columns": list(schema.point_table.columns),
        "grid_topology": schema.grid_topology,
        "cell_axes": list(schema.cell_schema.data_axes),
        "validity_contract": schema.cell_schema.validity_contract,
        "values": np.array(snapshot.block.values, copy=True),
        "validity": np.array(expand_snapshot_validity(snapshot), dtype=np.bool_, copy=True),
        "sigma": (
            None
            if snapshot.block.sigma is None
            else np.array(snapshot.block.sigma, dtype=np.float64, copy=True)
        ),
        "selected_axis": selected_axis,
        "slices": {},
        "grid_lookup": None,
        "grid_layout": None,
        "component": "values",
        "modified": source_path is None,
        "unsaved": False,
        "message": "",
        "producer_serial": None,
        "producer": None,
        "publication": None,
        "applied_snapshot": None,
        "applied_name": None,
        "applied_note": None,
        "lineage": None,
        "panel_id": "",
        "save_ready": False,
    }
    draft["coordinate_label_drafts"] = {}
    return draft


def _draft_schema(draft: Mapping[str, object]) -> object:
    """Build the one canonical schema represented by a mutable editor draft."""

    from zlc_data import DatasetSchema, PointTable, ValidityContract, ValueSchema

    _require_complete_coordinate_labels(draft, "applying this Dataset")
    axes = tuple(draft["cell_axes"])
    validity = np.asarray(draft["validity"], dtype=np.bool_)
    previous = draft["validity_contract"]
    if len(axes) == 1 and str(axes[0].role) == "scalar":
        contract = ValidityContract.value()
    else:
        required = {
            axis_id
            for axis_id in tuple(previous.component_axis_ids)
            if any(axis.axis_id == axis_id for axis in axes)
        }
        for offset, axis in enumerate(axes):
            array_axis = 2 + offset
            first = np.take(validity, 0, axis=array_axis)
            if not np.array_equal(
                validity,
                np.broadcast_to(
                    np.expand_dims(first, axis=array_axis), validity.shape
                ),
            ):
                required.add(axis.axis_id)
        ordered = tuple(axis.axis_id for axis in axes if axis.axis_id in required)
        contract = (
            ValidityContract.components(*ordered)
            if ordered
            else ValidityContract.value()
        )
    cell_schema = ValueSchema(
        axes,
        contract,
        np.dtype(draft["dtype"]),
        None if draft["unit"] is None else str(draft["unit"]),
    )
    return DatasetSchema(
        draft["repeat_axis"],
        PointTable(int(np.asarray(draft["values"]).shape[1]), tuple(draft["point_columns"])),
        draft["grid_topology"],
        cell_schema,
    )


def _manual_snapshot(draft: Mapping[str, object]) -> object:
    from zlc_data import owned_snapshot_from_arrays

    schema = _draft_schema(draft)
    return owned_snapshot_from_arrays(
        schema,
        np.asarray(draft["values"]),
        0,
        validity=np.asarray(draft["validity"], dtype=np.bool_),
        sigma=draft["sigma"],
    )


def _validate_draft_arrays(draft: Mapping[str, object]) -> object:
    schema = _draft_schema(draft)
    values = np.asarray(draft["values"])
    validity = np.asarray(draft["validity"])
    if values.shape != schema.physical_shape or values.dtype != schema.cell_schema.dtype:
        raise ValueError("data values differ from the authored Dataset shape or dtype")
    if validity.dtype != np.dtype(np.bool_) or validity.shape != schema.physical_shape:
        raise ValueError("data validity must be one bool for every value")
    sigma = draft["sigma"]
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.dtype != np.dtype(np.float64) or sigma.shape != schema.physical_shape:
            raise ValueError("data sigma must be float64 with the Dataset shape")
        finite = np.isfinite(sigma)
        if bool(np.any(finite & (sigma < 0.0))):
            raise ValueError("sample sigma must be non-negative")
    return schema


def _axis_id_set(draft: Mapping[str, object]) -> set[str]:
    topology = draft["grid_topology"]
    return {
        str(draft["repeat_axis"].axis_id),
        *(str(column.coordinate_id) for column in draft["point_columns"]),
        *(str(axis.axis_id) for axis in draft["cell_axes"]),
        *(
            ()
            if topology is None
            else (str(axis_id) for axis_id in topology.dimension_ids)
        ),
    }


def _new_axis_id(draft: Mapping[str, object], stem: str) -> object:
    from zlc_data import AxisId

    existing = _axis_id_set(draft)
    serial = 1
    while f"manual.{stem}.{serial}" in existing:
        serial += 1
    return AxisId(f"manual.{stem}.{serial}")


def _point_column(draft: Mapping[str, object], axis_id: str) -> tuple[int, object] | None:
    for index, column in enumerate(draft["point_columns"]):
        if str(column.coordinate_id) == str(axis_id):
            return index, column
    return None


def _grid_position(draft: Mapping[str, object], axis_id: str) -> int | None:
    topology = draft["grid_topology"]
    if topology is None:
        return None
    for index, candidate in enumerate(topology.dimension_ids):
        if str(candidate) == str(axis_id):
            return index
    return None


def _grid_value_kind(draft: Mapping[str, object], axis_id: str) -> str:
    found = _point_column(draft, axis_id)
    if found is not None:
        return str(found[1].value_kind)
    position = _grid_position(draft, axis_id)
    if position is None:
        raise KeyError(axis_id)
    domain = draft["grid_topology"].coordinate_domains[position]
    return (
        "NUMERIC"
        if all(type(value) in (int, float) for value in domain)
        else "TEXT"
    )


def _draft_axes(draft: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    from zlc_data import AxisRoleId, SCAN_POINT, point_domain_admits

    topology = draft["grid_topology"]
    grid_ids = set() if topology is None else set(topology.dimension_ids)

    def roles(domain: str, current: object) -> tuple[tuple[str, str], ...]:
        if domain == "repeat":
            return (("repeat", "Repeat"),)
        values = list(_MANUAL_ROLE_CHOICES)
        selected = str(current)
        if selected not in values and selected != "repeat":
            values.append(selected)
        if domain in {"point", "grid"}:
            values = [
                value
                for value in values
                if point_domain_admits(AxisRoleId(value))
            ]
        elif domain == "cell":
            values = [
                value
                for value in values
                if value not in {"primary-index", "scalar"}
            ]
        return tuple((value, value.replace("-", " ").title()) for value in values)

    def entry(
        *,
        axis_id: object,
        domain: str,
        name: str,
        role: object,
        unit: object,
        frame: object,
        value_kind: str = "",
    ) -> dict[str, object]:
        return {
            "id": str(axis_id),
            "domain": domain,
            "domain_label": {
                "repeat": "Repeat",
                "point": "Point",
                "grid": "Grid",
                "cell": "Cell",
            }[domain],
            "name": str(name),
            "size": len(_axis_values(draft, str(axis_id))),
            "role": str(role),
            "unit": "" if unit is None else str(unit),
            "coordinate_frame": "" if frame is None else str(frame),
            "role_choices": roles(domain, role),
            "value_kind": str(value_kind),
            "value_kind_choices": (
                (("NUMERIC", "Numeric"), ("TEXT", "Text"))
                if domain in {"point", "grid"}
                else ()
            ),
            "show_value_kind": domain in {"point", "grid"},
            "unit_enabled": domain not in {"point", "grid"} or value_kind == "NUMERIC",
        }

    repeat = draft["repeat_axis"]
    result = [
        entry(
            axis_id=repeat.axis_id,
            domain="repeat",
            name=repeat.name,
            role=repeat.role,
            unit=repeat.unit,
            frame=repeat.coordinate_frame,
        )
    ]
    if topology is not None:
        for position, axis_id in enumerate(topology.dimension_ids):
            found = _point_column(draft, str(axis_id))
            column = None if found is None else found[1]
            result.append(
                entry(
                    axis_id=axis_id,
                    domain="grid",
                    name=str(axis_id) if column is None else column.name,
                    role=SCAN_POINT if column is None else column.role,
                    unit=None if column is None else column.unit,
                    frame=None if column is None else column.coordinate_frame,
                    value_kind=_grid_value_kind(draft, str(axis_id)),
                )
            )
    for column in draft["point_columns"]:
        if column.coordinate_id in grid_ids:
            continue
        result.append(
            entry(
                axis_id=column.coordinate_id,
                domain="point",
                name=column.name,
                role=column.role,
                unit=column.unit,
                frame=column.coordinate_frame,
                value_kind=column.value_kind,
            )
        )
    for axis in draft["cell_axes"]:
        if str(axis.role) == "scalar":
            continue
        result.append(
            entry(
                axis_id=axis.axis_id,
                domain="cell",
                name=axis.name,
                role=axis.role,
                unit=axis.unit,
                frame=axis.coordinate_frame,
            )
        )
    return tuple(result)


def _axis_location(draft: Mapping[str, object], axis_id: str) -> tuple[str, int, object]:
    if str(draft["repeat_axis"].axis_id) == str(axis_id):
        return "repeat", 0, draft["repeat_axis"]
    grid = _grid_position(draft, axis_id)
    if grid is not None:
        return "grid", grid, draft["grid_topology"]
    point = _point_column(draft, axis_id)
    if point is not None:
        return "point", point[0], point[1]
    for index, axis in enumerate(draft["cell_axes"]):
        if str(axis.axis_id) == str(axis_id):
            return "cell", index, axis
    raise KeyError(f"unknown Dataset axis {axis_id!r}")


def _axis_values(draft: Mapping[str, object], axis_id: str) -> object:
    domain, position, value = _axis_location(draft, axis_id)
    if domain == "grid":
        return value.coordinate_domains[position]
    if domain == "point":
        return value.values
    return _axis_coordinates(value)


def _axis_labels(draft: Mapping[str, object], axis_id: str) -> tuple[str, ...] | None:
    domain, position, value = _axis_location(draft, axis_id)
    if domain == "grid":
        labels = value.coordinate_labels
        return None if labels is None else labels[position]
    return value.coordinate_labels


def _coordinate_label_texts(
    draft: Mapping[str, object], axis_id: str
) -> tuple[str, ...] | None:
    saved = draft.get("coordinate_label_drafts", {})
    if str(axis_id) in saved:
        return saved[str(axis_id)]
    return _axis_labels(draft, axis_id)


def _partial_coordinate_label_axes(
    draft: Mapping[str, object],
) -> tuple[str, ...]:
    partial = set(draft.get("coordinate_label_drafts", {}))
    if not partial:
        return ()
    return tuple(
        f"{axis['name']} ({axis['id']})"
        for axis in _draft_axes(draft)
        if str(axis["id"]) in partial
    )


def _require_complete_coordinate_labels(
    draft: Mapping[str, object], action: str
) -> None:
    partial = _partial_coordinate_label_axes(draft)
    if partial:
        raise ValueError(
            f"Coordinate labels for {', '.join(partial)} are incomplete; "
            f"fill every label or clear them all before {action}"
        )


def _array_insert(array: np.ndarray, axis: int, index: int, fill: object) -> np.ndarray:
    shape = list(array.shape)
    shape[axis] = 1
    addition = np.full(tuple(shape), fill, dtype=array.dtype)
    before = [slice(None)] * array.ndim
    after = [slice(None)] * array.ndim
    before[axis] = slice(0, index)
    after[axis] = slice(index, None)
    return np.concatenate((array[tuple(before)], addition, array[tuple(after)]), axis=axis)


def _insert_storage(draft: dict[str, object], axis: int, index: int) -> None:
    draft["values"] = _array_insert(np.asarray(draft["values"]), axis, index, 0)
    draft["validity"] = _array_insert(
        np.asarray(draft["validity"], dtype=np.bool_), axis, index, False
    )
    if draft["sigma"] is not None:
        draft["sigma"] = _array_insert(
            np.asarray(draft["sigma"], dtype=np.float64), axis, index, np.nan
        )


def _grow_storage(draft: dict[str, object], axis: int, count: int) -> None:
    if count <= 0:
        return
    for key, fill, dtype in (
        ("values", 0, np.asarray(draft["values"]).dtype),
        ("validity", False, np.dtype(np.bool_)),
        ("sigma", np.nan, np.dtype(np.float64)),
    ):
        if draft[key] is None:
            continue
        array = np.asarray(draft[key], dtype=dtype)
        shape = list(array.shape)
        shape[axis] = int(count)
        draft[key] = np.concatenate(
            (array, np.full(tuple(shape), fill, dtype=dtype)), axis=axis
        )


def _delete_storage(draft: dict[str, object], axis: int, indices: object) -> None:
    selected = tuple(sorted({int(index) for index in tuple(indices)}))
    draft["values"] = np.delete(np.asarray(draft["values"]), selected, axis=axis)
    draft["validity"] = np.delete(
        np.asarray(draft["validity"], dtype=np.bool_), selected, axis=axis
    )
    if draft["sigma"] is not None:
        draft["sigma"] = np.delete(
            np.asarray(draft["sigma"], dtype=np.float64), selected, axis=axis
        )


def _reorder_storage(draft: dict[str, object], axis: int, order: object) -> None:
    selected = np.asarray(tuple(int(index) for index in tuple(order)), dtype=np.intp)
    draft["values"] = np.take(np.asarray(draft["values"]), selected, axis=axis)
    draft["validity"] = np.take(
        np.asarray(draft["validity"], dtype=np.bool_), selected, axis=axis
    )
    if draft["sigma"] is not None:
        draft["sigma"] = np.take(
            np.asarray(draft["sigma"], dtype=np.float64), selected, axis=axis
        )


def _new_coordinate(values: tuple[object, ...], *, numeric: bool) -> object:
    numeric_values = [value for value in values if type(value) in (int, float)]
    if numeric:
        candidate: object = 0 if not numeric_values else max(numeric_values) + 1
        while candidate in values:
            candidate = int(candidate) + 1
        return candidate
    serial = len(values) + 1
    candidate = f"coordinate-{serial}"
    while candidate in values:
        serial += 1
        candidate = f"coordinate-{serial}"
    return candidate


def _extend_coordinates(
    values: tuple[object, ...], count: int, *, numeric: bool
) -> tuple[object, ...]:
    result = list(values)
    occupied = set(result)
    if numeric:
        numeric_values = [value for value in result if type(value) in (int, float)]
        candidate: object = 0 if not numeric_values else max(numeric_values) + 1
        for _index in range(int(count)):
            while candidate in occupied:
                candidate = int(candidate) + 1
            result.append(candidate)
            occupied.add(candidate)
            candidate = int(candidate) + 1
    else:
        serial = len(result) + 1
        for _index in range(int(count)):
            candidate = f"coordinate-{serial}"
            while candidate in occupied:
                serial += 1
                candidate = f"coordinate-{serial}"
            result.append(candidate)
            occupied.add(candidate)
            serial += 1
    return tuple(result)


def _parse_coordinate(text: object, *, numeric: bool, allow_none: bool) -> object:
    value = str(text).strip()
    if not value:
        if allow_none:
            return None
        raise ValueError("axis coordinates cannot be blank")
    if not numeric:
        return value
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{value!r} is not a numeric coordinate") from error
    if not np.isfinite(number):
        raise ValueError("coordinates must be finite")
    return int(number) if number.is_integer() else number


def _axis_spec_replacing(axis: object, **changes: object) -> object:
    from zlc_data import AxisSpec, CoordinateFrameId

    coordinates = changes.pop("coordinates", axis.coordinates)
    frame = changes.pop("coordinate_frame", axis.coordinate_frame)
    return AxisSpec(
        axis.axis_id,
        str(changes.pop("name", axis.name)),
        changes.pop("role", axis.role),
        int(changes.pop("size", axis.size)),
        coordinates,
        changes.pop("unit", axis.unit),
        (
            None
            if frame in (None, "")
            else frame
            if isinstance(frame, CoordinateFrameId)
            else CoordinateFrameId(str(frame))
        ),
        int(changes.pop("index_origin", 0 if coordinates is not None else axis.index_origin)),
        changes.pop("coordinate_labels", axis.coordinate_labels),
    )


def _point_column_replacing(column: object, **changes: object) -> object:
    from zlc_data import CoordinateFrameId, PointColumn

    frame = changes.pop("coordinate_frame", column.coordinate_frame)
    values = tuple(changes.pop("values", column.values))
    labels = changes.pop("coordinate_labels", column.coordinate_labels)
    return PointColumn(
        column.coordinate_id,
        str(changes.pop("name", column.name)),
        changes.pop("role", column.role),
        str(changes.pop("value_kind", column.value_kind)),
        values,
        changes.pop("unit", column.unit),
        (
            None
            if frame in (None, "")
            else frame
            if isinstance(frame, CoordinateFrameId)
            else CoordinateFrameId(str(frame))
        ),
        labels,
    )


def _logical_dimensions(draft: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Meaningful table dimensions; Point storage is replaced by Grid topology."""

    repeat = draft["repeat_axis"]
    result = []
    if int(repeat.size) > 1:
        result.append(
            {
                "axis_id": str(repeat.axis_id),
                "label": repeat.name,
                "position": 0,
                "kind": "repeat",
                "coordinates": _axis_coordinates(repeat),
                "labels": repeat.coordinate_labels,
            }
        )
    topology = draft["grid_topology"]
    complete = topology is not None and len(topology.row_to_cell) == prod(topology.logical_shape)
    if topology is None or not complete:
        rows = int(np.asarray(draft["values"]).shape[1])
        if rows > 1:
            columns = tuple(draft["point_columns"])
            coordinate_source = columns[0] if len(columns) == 1 else None
            result.append(
                {
                    "axis_id": _POINT_ROWS,
                    "label": "point",
                    "position": 1,
                    "kind": "point",
                    "coordinates": (
                        range(rows)
                        if coordinate_source is None
                        else coordinate_source.values
                    ),
                    "labels": (
                        None
                        if coordinate_source is None
                        else coordinate_source.coordinate_labels
                    ),
                }
            )
        cell_offset = 2
    else:
        for index, (axis_id, coordinates) in enumerate(
            zip(topology.dimension_ids, topology.coordinate_domains, strict=True)
        ):
            if len(coordinates) <= 1:
                continue
            found = _point_column(draft, str(axis_id))
            column = None if found is None else found[1]
            labels = (
                None
                if topology.coordinate_labels is None
                else topology.coordinate_labels[index]
            )
            result.append(
                {
                    "axis_id": str(axis_id),
                    "label": str(axis_id) if column is None else column.name,
                    "position": 1 + index,
                    "kind": "grid",
                    "grid_position": index,
                    "coordinates": coordinates,
                    "labels": labels,
                }
            )
        cell_offset = 1 + len(topology.dimension_ids)
    axes = tuple(draft["cell_axes"])
    for index, axis in enumerate(axes):
        if str(axis.role) == "scalar" or int(axis.size) <= 1:
            continue
        result.append(
            {
                "axis_id": str(axis.axis_id),
                "label": axis.name,
                "position": cell_offset + index,
                "kind": "cell",
                "cell_position": index,
                "coordinates": _axis_coordinates(axis),
                "labels": axis.coordinate_labels,
            }
        )
    return tuple(result)


def _grid_layout(draft: Mapping[str, object]) -> tuple[bool, bool]:
    topology = draft["grid_topology"]
    cached = draft.get("grid_layout")
    if isinstance(cached, tuple) and len(cached) == 3 and cached[0] is topology:
        return bool(cached[1]), bool(cached[2])
    if topology is None or len(topology.row_to_cell) != prod(topology.logical_shape):
        result = (False, False)
    else:
        flat = np.ravel_multi_index(
            tuple(topology.cell_indices[:, index] for index in range(len(topology.logical_shape))),
            topology.logical_shape,
        )
        result = (
            True,
            bool(np.array_equal(flat, np.arange(len(flat), dtype=flat.dtype))),
        )
    if isinstance(draft, dict):
        draft["grid_layout"] = (topology, *result)
    return result


def _grid_lookup(draft: Mapping[str, object]) -> np.ndarray | None:
    topology = draft["grid_topology"]
    if topology is None:
        return None
    complete, _canonical = _grid_layout(draft)
    if not complete:
        return None
    cached = draft.get("grid_lookup")
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] is topology:
        return cached[1]
    lookup = np.full(topology.logical_shape, -1, dtype=np.intp)
    cells = topology.cell_indices
    lookup[tuple(cells[:, index] for index in range(cells.shape[1]))] = np.arange(
        cells.shape[0], dtype=np.intp
    )
    lookup.setflags(write=False)
    if isinstance(draft, dict):
        draft["grid_lookup"] = (topology, lookup)
    return lookup


def _table_address(
    draft: Mapping[str, object],
) -> tuple[
    tuple[object, ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    dimensions = _logical_dimensions(draft)
    shown = dimensions[-2:]
    sliced = dimensions[:-2]
    selected_slices = dict(draft["slices"])

    def selected_index(axis_id: str, default: int = 0) -> object:
        for dimension in shown:
            if str(dimension["axis_id"]) == str(axis_id):
                return slice(None)
        return int(selected_slices.get(str(axis_id), default))

    repeat = draft["repeat_axis"]
    repeat_index = selected_index(str(repeat.axis_id)) if int(repeat.size) > 1 else 0
    topology = draft["grid_topology"]
    complete, _canonical = _grid_layout(draft)
    if topology is None or not complete:
        point_size = int(np.asarray(draft["values"]).shape[1])
        point_index = selected_index(_POINT_ROWS) if point_size > 1 else 0
    else:
        grid_indices = tuple(
            selected_index(str(axis_id)) if len(domain) > 1 else 0
            for axis_id, domain in zip(
                topology.dimension_ids, topology.coordinate_domains, strict=True
            )
        )
        lookup = _grid_lookup(draft)
        assert lookup is not None
        point_index = lookup[grid_indices]
    cell_indices = tuple(
        selected_index(str(axis.axis_id)) if int(axis.size) > 1 else 0
        for axis in draft["cell_axes"]
    )
    return (
        (repeat_index, point_index, *cell_indices),
        shown,
        sliced,
    )


def _table_projection(
    draft: Mapping[str, object],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    complete, canonical = _grid_layout(draft)
    topology = draft["grid_topology"]
    if complete and canonical and topology is not None:
        dimensions = _logical_dimensions(draft)
        shown = dimensions[-2:]
        sliced = dimensions[:-2]
        selected_slices = dict(draft["slices"])
        logical_shape = (
            int(draft["repeat_axis"].size),
            *topology.logical_shape,
            *(int(axis.size) for axis in draft["cell_axes"]),
        )
        indexer: list[object] = [0] * len(logical_shape)
        for dimension in dimensions:
            indexer[int(dimension["position"])] = (
                slice(None)
                if dimension in shown
                else int(selected_slices.get(str(dimension["axis_id"]), 0))
            )
        address = tuple(indexer)
        value_source = np.asarray(draft["values"]).reshape(logical_shape)
        validity_source = np.asarray(draft["validity"], dtype=np.bool_).reshape(logical_shape)
        sigma_source = (
            None
            if draft["sigma"] is None
            else np.asarray(draft["sigma"], dtype=np.float64).reshape(logical_shape)
        )
    else:
        address, shown, sliced = _table_address(draft)
        value_source = np.asarray(draft["values"])
        validity_source = np.asarray(draft["validity"], dtype=np.bool_)
        sigma_source = (
            None
            if draft["sigma"] is None
            else np.asarray(draft["sigma"], dtype=np.float64)
        )
    component = str(draft["component"])
    validity_plane = validity_source[address]
    if component == "validity":
        values = np.asarray(validity_plane)
        table_validity = None
    elif component == "sigma" and sigma_source is not None:
        values = sigma_source[address]
        table_validity = validity_plane & np.isfinite(values)
    else:
        component = "values"
        values = value_source[address]
        table_validity = validity_plane
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape((1, 1))
    elif values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
        if table_validity is not None:
            table_validity = np.asarray(table_validity).reshape(values.shape)
    elif values.ndim != 2:
        raise RuntimeError("manual data table did not reduce to two dimensions")
    def headers(dimension: Mapping[str, object]) -> object:
        return (
            dimension["coordinates"]
            if dimension["labels"] is None
            else dimension["labels"]
        )

    rows = headers(shown[0]) if shown else ("Value",)
    columns = headers(shown[1]) if len(shown) == 2 else ("Value",)
    slice_rows = []
    selected_slices = dict(draft["slices"])
    for dimension in sliced:
        coordinates = dimension["coordinates"]
        index = max(0, min(int(selected_slices.get(dimension["axis_id"], 0)), len(coordinates) - 1))
        labels = dimension["labels"]
        current_label = (
            str(labels[index])
            if labels is not None
            else str(coordinates[index])
        )
        slice_rows.append(
            {
                "axis_id": dimension["axis_id"],
                "label": dimension["label"],
                "size": len(coordinates),
                "index": index,
                "current_label": current_label,
            }
        )
    choices = [("values", "Values"), ("validity", "Validity")]
    if draft["sigma"] is not None:
        choices.append(("sigma", "Sigma"))
    grid_headers = None
    topology = draft["grid_topology"]
    complete, _canonical = _grid_layout(draft)
    if topology is not None and not complete:
        grid_headers = {
            "cell_indices": topology.cell_indices,
            "coordinates": topology.coordinate_domains,
            "labels": topology.coordinate_labels,
        }
    table = {
        "component": component,
        "component_choices": tuple(choices),
        "sigma_enabled": draft["sigma"] is not None,
        "shape": tuple(values.shape),
        "values": values,
        "validity": table_validity,
        "blank_help": {
            "values": "Blank deletes this sample and marks it invalid",
            "validity": "Blank means False",
            "sigma": "Blank means no stated sigma",
        }[component],
        "row_headers": rows,
        "column_headers": columns,
        "editable": True,
    }
    for shown_index, dimension in enumerate(shown):
        if dimension["axis_id"] == _POINT_ROWS and grid_headers is not None:
            table["row_header_grid" if shown_index == 0 else "column_header_grid"] = grid_headers
    return (
        table,
        tuple(slice_rows),
    )


def _table_index(draft: Mapping[str, object], row: int, column: int) -> tuple[int, ...]:
    dimensions = _logical_dimensions(draft)
    shown = dimensions[-2:]
    selected_slices = dict(draft["slices"])
    def selected_index(axis_id: str) -> int:
        for shown_index, dimension in enumerate(shown):
            if str(dimension["axis_id"]) == str(axis_id):
                return int(row if shown_index == 0 else column)
        return int(selected_slices.get(str(axis_id), 0))

    repeat_axis = draft["repeat_axis"]
    repeat = selected_index(str(repeat_axis.axis_id)) if int(repeat_axis.size) > 1 else 0
    topology = draft["grid_topology"]
    complete, canonical = _grid_layout(draft)
    if topology is None or not complete:
        point = (
            selected_index(_POINT_ROWS)
            if int(np.asarray(draft["values"]).shape[1]) > 1
            else 0
        )
    else:
        grid = tuple(
            selected_index(str(axis_id)) if len(domain) > 1 else 0
            for axis_id, domain in zip(
                topology.dimension_ids, topology.coordinate_domains, strict=True
            )
        )
        if canonical:
            point = int(np.ravel_multi_index(grid, topology.logical_shape))
        else:
            lookup = _grid_lookup(draft)
            assert lookup is not None
            point = int(lookup[grid])
    result = [repeat, point]
    for axis in draft["cell_axes"]:
        result.append(selected_index(str(axis.axis_id)) if int(axis.size) > 1 else 0)
    return tuple(result)


def _data_projection(draft: Mapping[str, object]) -> dict[str, object]:
    axes = _draft_axes(draft)
    selected = str(draft["selected_axis"])
    if not any(str(axis["id"]) == selected for axis in axes):
        selected = str(axes[0]["id"])
    coordinates = _axis_values(draft, selected)
    coordinate_labels = _coordinate_label_texts(draft, selected)
    table, slices = _table_projection(draft)
    can_apply = True
    validation_message = ""
    try:
        _validate_draft_arrays(draft)
    except (TypeError, ValueError) as error:
        can_apply = False
        validation_message = str(error)
    message = str(draft.get("message") or validation_message)
    source_path = draft["source_path"]
    suggested = (
        f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', str(draft['name'])).strip('.-') or 'figure'}.npz"
    )
    return {
        "dataset": {
            "name": str(draft["name"]),
            "dtype": np.dtype(draft["dtype"]).str,
            "unit": "" if draft["unit"] is None else str(draft["unit"]),
            "note": str(draft["note"]),
            "source": str(draft["source_text"]),
            "dtype_choices": tuple(
                (value, np.dtype(value).name) for value in _MANUAL_DTYPES
            ),
        },
        "domain_choices": (
            ("repeat", "Repeat", False),
            ("point", "Point column", True),
            ("grid", "Grid dimension", True),
            ("cell", "Cell data", True),
        ),
        "new_axis_domain": "cell",
        "axes": axes,
        "selected_axis": selected,
        "coordinates": {
            "shape": (len(coordinates), 2),
            "column_values": (coordinates, coordinate_labels),
            "row_headers": range(len(coordinates)),
            "column_headers": ("Coordinate", "Label"),
            "editable": True,
        },
        "slices": slices,
        "table": table,
        "dirty": bool(draft["modified"] or draft["unsaved"]),
        "can_apply": can_apply,
        "can_save": bool(draft["save_ready"] and not draft["modified"]),
        "save_suggested": str(
            (Path(source_path).parent / suggested) if source_path is not None else suggested
        ),
        "message": message,
    }


def _sequence_insert(values: tuple, index: int, value: object) -> tuple:
    return (*values[:index], value, *values[index:])


def _labels_for_values(column: object, values: tuple[object, ...]) -> tuple[str, ...] | None:
    if column.coordinate_labels is None:
        return None
    known: dict[object, str] = {}
    for value, label in zip(column.values, column.coordinate_labels, strict=True):
        known.setdefault(value, label)
    return tuple(
        known.get(value, "missing" if value is None else str(value))
        for value in values
    )


def _topology_replacing(
    topology: object,
    *,
    dimension_ids: object | None = None,
    coordinate_domains: object | None = None,
    row_to_cell: object | None = None,
    coordinate_labels: object = ...,
) -> object:
    from zlc_data import GridTopology

    labels = topology.coordinate_labels if coordinate_labels is ... else coordinate_labels
    return GridTopology(
        tuple(topology.dimension_ids if dimension_ids is None else dimension_ids),
        tuple(topology.coordinate_domains if coordinate_domains is None else coordinate_domains),
        tuple(topology.row_to_cell if row_to_cell is None else row_to_cell),
        labels,
    )


def _replace_point_columns_for_rows(
    draft: dict[str, object],
    order: tuple[int, ...],
) -> None:
    columns = []
    for column in draft["point_columns"]:
        values = tuple(column.values[index] for index in order)
        labels = (
            None
            if column.coordinate_labels is None
            else tuple(column.coordinate_labels[index] for index in order)
        )
        columns.append(
            _point_column_replacing(
                column,
                values=values,
                coordinate_labels=labels,
            )
        )
    draft["point_columns"] = columns


def _append_point_storage(draft: dict[str, object], count: int) -> None:
    if count <= 0:
        return
    values = np.asarray(draft["values"])
    shape = list(values.shape)
    shape[1] = int(count)
    draft["values"] = np.concatenate(
        (values, np.zeros(tuple(shape), dtype=values.dtype)), axis=1
    )
    validity = np.asarray(draft["validity"], dtype=np.bool_)
    draft["validity"] = np.concatenate(
        (validity, np.zeros(tuple(shape), dtype=np.bool_)), axis=1
    )
    if draft["sigma"] is not None:
        sigma = np.asarray(draft["sigma"], dtype=np.float64)
        draft["sigma"] = np.concatenate(
            (sigma, np.full(tuple(shape), np.nan, dtype=np.float64)), axis=1
        )


def _set_coordinate_cells(
    draft: dict[str, object],
    axis_id: str,
    cells: object,
) -> bool:
    domain, position, selected = _axis_location(draft, axis_id)
    values = list(_axis_values(draft, axis_id))
    original_values = tuple(values)
    original_label_texts = _coordinate_label_texts(draft, axis_id)
    if domain in {"point", "grid"}:
        value_kind = (
            selected.value_kind
            if domain == "point"
            else _grid_value_kind(draft, axis_id)
        )
        numeric = value_kind == "NUMERIC"
    else:
        numeric = all(type(value) in (int, float) for value in values)
    updates = []
    for row, column, text in tuple(cells):
        row = int(row)
        column = int(column)
        if column not in (0, 1) or not 0 <= row < len(values):
            raise IndexError("coordinate edit lies outside the selected axis")
        updates.append((row, column, text))
    edited_labels = None
    if any(column == 1 for _row, column, _text in updates):
        edited_labels = (
            [""] * len(values)
            if original_label_texts is None
            else list(original_label_texts)
        )
    for row, column, text in updates:
        if column == 0:
            values[row] = _parse_coordinate(
                text,
                numeric=numeric,
                allow_none=domain == "point",
            )
        else:
            assert edited_labels is not None
            edited_labels[row] = str(text).strip()
    resolved = tuple(values)
    label_texts = (
        original_label_texts
        if edited_labels is None
        else tuple(edited_labels) if any(edited_labels) else None
    )
    resolved_labels = label_texts if label_texts is not None and all(label_texts) else None
    if resolved_labels is not None:
        by_value: dict[object, str] = {}
        for value, label in zip(resolved, resolved_labels, strict=True):
            previous = by_value.setdefault(value, label)
            if previous != label:
                raise ValueError("equal coordinates must share one display label")
    if resolved == original_values and label_texts == original_label_texts:
        return False
    if domain == "repeat":
        draft["repeat_axis"] = _axis_spec_replacing(
            selected,
            coordinates=resolved,
            coordinate_labels=resolved_labels,
            index_origin=0,
        )
    elif domain == "cell":
        axes = list(draft["cell_axes"])
        axes[position] = _axis_spec_replacing(
            selected,
            coordinates=resolved,
            coordinate_labels=resolved_labels,
            index_origin=0,
        )
        draft["cell_axes"] = axes
    elif domain == "point":
        columns = list(draft["point_columns"])
        columns[position] = _point_column_replacing(
            selected,
            values=resolved,
            coordinate_labels=resolved_labels,
        )
        draft["point_columns"] = columns
    else:
        topology = selected
        if any(value is None for value in resolved) or len(set(resolved)) != len(resolved):
            raise ValueError("grid coordinates must be non-empty and unique")
        domains = list(topology.coordinate_domains)
        domains[position] = resolved
        topology_labels = topology.coordinate_labels
        if topology_labels is None and resolved_labels is not None:
            topology_labels = tuple(
                resolved_labels if dim == position else None
                for dim in range(len(topology.dimension_ids))
            )
        elif topology_labels is not None:
            topology_labels = tuple(
                resolved_labels if dim == position else item
                for dim, item in enumerate(topology_labels)
            )
        draft["grid_topology"] = _topology_replacing(
            topology,
            coordinate_domains=domains,
            coordinate_labels=topology_labels,
        )
        found = _point_column(draft, axis_id)
        if found is not None:
            column_index, column = found
            mapped = tuple(resolved[cell[position]] for cell in topology.row_to_cell)
            mapped_labels = (
                None
                if resolved_labels is None
                else tuple(resolved_labels[cell[position]] for cell in topology.row_to_cell)
            )
            columns = list(draft["point_columns"])
            columns[column_index] = _point_column_replacing(
                column,
                values=mapped,
                coordinate_labels=mapped_labels,
            )
            draft["point_columns"] = columns
    if label_texts is not None and not all(label_texts):
        draft["coordinate_label_drafts"][str(axis_id)] = label_texts
    else:
        draft["coordinate_label_drafts"].pop(str(axis_id), None)
    return True


def _insert_coordinate(draft: dict[str, object], axis_id: str, after: int) -> None:
    from zlc_data import GridTopology

    domain, position, selected = _axis_location(draft, axis_id)
    old_values = tuple(_axis_values(draft, axis_id))
    index = max(0, min(int(after) + 1, len(old_values)))
    numeric = (
        selected.value_kind == "NUMERIC"
        if domain == "point"
        else _grid_value_kind(draft, axis_id) == "NUMERIC"
        if domain == "grid"
        else all(type(value) in (int, float) for value in old_values)
    )
    added = _new_coordinate(old_values, numeric=numeric)
    new_values = _sequence_insert(old_values, index, added)
    labels = _axis_labels(draft, axis_id)
    new_labels = (
        None
        if labels is None
        else _sequence_insert(labels, index, str(added))
    )
    if domain in {"repeat", "cell"}:
        storage_axis = 0 if domain == "repeat" else 2 + position
        _insert_storage(draft, storage_axis, index)
        replacement = _axis_spec_replacing(
            selected,
            size=len(new_values),
            coordinates=new_values,
            coordinate_labels=new_labels,
            index_origin=0,
        )
        if domain == "repeat":
            draft["repeat_axis"] = replacement
        else:
            axes = list(draft["cell_axes"])
            axes[position] = replacement
            draft["cell_axes"] = axes
        return
    if domain == "point":
        if draft["grid_topology"] is not None:
            raise ValueError(
                "add grid cells through a Grid dimension, not an unrelated point column"
            )
        if len(tuple(draft["point_columns"])) != 1:
            raise ValueError(
                "a new point row is ambiguous while several point columns exist"
            )
        _insert_storage(draft, 1, index)
        columns = []
        for column_index, column in enumerate(draft["point_columns"]):
            value = added if column_index == position else None
            values = _sequence_insert(tuple(column.values), index, value)
            column_labels = column.coordinate_labels
            column_labels = (
                None
                if column_labels is None
                else _sequence_insert(
                    tuple(column_labels),
                    index,
                    "missing" if value is None else str(value),
                )
            )
            columns.append(
                _point_column_replacing(
                    column,
                    values=values,
                    coordinate_labels=column_labels,
                )
            )
        draft["point_columns"] = columns
        return

    topology = selected
    if len(topology.row_to_cell) != prod(topology.logical_shape):
        raise ValueError(
            "a Grid coordinate can be inserted only when the existing grid is complete"
        )
    topology_ids = set(topology.dimension_ids)
    if any(
        column.coordinate_id not in topology_ids
        for column in draft["point_columns"]
    ):
        raise ValueError(
            "a Grid coordinate cannot invent values for unrelated point columns"
        )
    domains = list(topology.coordinate_domains)
    domains[position] = new_values
    shifted = tuple(
        tuple(
            value + 1 if dim == position and value >= index else value
            for dim, value in enumerate(cell)
        )
        for cell in topology.row_to_cell
    )
    ranges = [range(len(domain_values)) for domain_values in domains]
    additions = []
    for rest in product(*(values for dim, values in enumerate(ranges) if dim != position)):
        cell = []
        iterator = iter(rest)
        for dim in range(len(domains)):
            cell.append(index if dim == position else next(iterator))
        candidate = tuple(cell)
        if candidate not in shifted:
            additions.append(candidate)
    mapping = (*shifted, *additions)
    topology_labels = topology.coordinate_labels
    if topology_labels is not None:
        edited_labels = list(topology_labels)
        selected_labels = edited_labels[position]
        if selected_labels is not None:
            edited_labels[position] = _sequence_insert(
                tuple(selected_labels), index, str(added)
            )
        topology_labels = tuple(edited_labels)
    replacement = GridTopology(
        topology.dimension_ids,
        tuple(domains),
        mapping,
        topology_labels,
    )
    _append_point_storage(draft, len(additions))
    columns = []
    topology_ids = tuple(replacement.dimension_ids)
    for column in draft["point_columns"]:
        if column.coordinate_id in topology_ids:
            dim = topology_ids.index(column.coordinate_id)
            values = tuple(
                replacement.coordinate_domains[dim][cell[dim]] for cell in mapping
            )
            domain_labels = (
                None
                if replacement.coordinate_labels is None
                else replacement.coordinate_labels[dim]
            )
            column_labels = (
                None
                if domain_labels is None
                else tuple(domain_labels[cell[dim]] for cell in mapping)
            )
        else:
            values = (*tuple(column.values), *(None for _ in additions))
            column_labels = _labels_for_values(column, values)
        columns.append(
            _point_column_replacing(
                column,
                values=values,
                coordinate_labels=column_labels,
            )
        )
    draft["point_columns"] = columns
    draft["grid_topology"] = replacement
    draft["message"] = f"Added {len(additions)} physical point row(s) for the new Grid coordinate"


def _remove_coordinates(
    draft: dict[str, object], axis_id: str, indices: object
) -> bool:
    selected_indices = tuple(sorted({int(value) for value in tuple(indices)}))
    values = _axis_values(draft, axis_id)
    if not selected_indices:
        return False
    if selected_indices[0] < 0 or selected_indices[-1] >= len(values):
        raise IndexError("coordinate removal lies outside the selected axis")
    if len(selected_indices) >= len(values):
        raise ValueError("a Dataset axis must retain at least one coordinate")
    domain, position, selected = _axis_location(draft, axis_id)
    keep = tuple(index for index in range(len(values)) if index not in selected_indices)
    remaining = tuple(values[index] for index in keep)
    labels = _axis_labels(draft, axis_id)
    remaining_labels = None if labels is None else tuple(labels[index] for index in keep)
    if domain in {"repeat", "cell"}:
        storage_axis = 0 if domain == "repeat" else 2 + position
        _delete_storage(draft, storage_axis, selected_indices)
        replacement = _axis_spec_replacing(
            selected,
            size=len(remaining),
            coordinates=remaining,
            coordinate_labels=remaining_labels,
            index_origin=0,
        )
        if domain == "repeat":
            draft["repeat_axis"] = replacement
        else:
            axes = list(draft["cell_axes"])
            axes[position] = replacement
            draft["cell_axes"] = axes
        return True
    if domain == "point":
        row_keep = tuple(
            index for index in range(len(values)) if index not in selected_indices
        )
        _delete_storage(draft, 1, selected_indices)
        _replace_point_columns_for_rows(draft, row_keep)
        topology = draft["grid_topology"]
        if topology is not None:
            draft["grid_topology"] = _topology_replacing(
                topology,
                row_to_cell=tuple(topology.row_to_cell[index] for index in row_keep),
            )
        return True

    topology = selected
    removed = set(selected_indices)
    row_keep = tuple(
        row
        for row, cell in enumerate(topology.row_to_cell)
        if cell[position] not in removed
    )
    if not row_keep:
        raise ValueError("removing those grid coordinates would remove every point row")
    rank_map = {
        old: new for new, old in enumerate(index for index in range(len(values)) if index not in removed)
    }
    mapping = tuple(
        tuple(
            rank_map[value] if dim == position else value
            for dim, value in enumerate(topology.row_to_cell[row])
        )
        for row in row_keep
    )
    domains = list(topology.coordinate_domains)
    domains[position] = remaining
    topology_labels = topology.coordinate_labels
    if topology_labels is not None:
        edited = list(topology_labels)
        if edited[position] is not None:
            edited[position] = remaining_labels
        topology_labels = tuple(edited)
    _delete_storage(
        draft,
        1,
        tuple(row for row in range(len(topology.row_to_cell)) if row not in row_keep),
    )
    _replace_point_columns_for_rows(draft, row_keep)
    draft["grid_topology"] = _topology_replacing(
        topology,
        coordinate_domains=domains,
        row_to_cell=mapping,
        coordinate_labels=topology_labels,
    )
    return True


def _move_coordinate(
    draft: dict[str, object], axis_id: str, index: int, delta: int
) -> bool:
    values = _axis_values(draft, axis_id)
    index = int(index)
    destination = index + int(delta)
    if not (0 <= index < len(values) and 0 <= destination < len(values)):
        return False
    order = list(range(len(values)))
    moved = order.pop(index)
    order.insert(destination, moved)
    order_tuple = tuple(order)
    reordered = tuple(values[position] for position in order_tuple)
    labels = _axis_labels(draft, axis_id)
    reordered_labels = None if labels is None else tuple(labels[position] for position in order_tuple)
    domain, position, selected = _axis_location(draft, axis_id)
    if domain in {"repeat", "cell"}:
        storage_axis = 0 if domain == "repeat" else 2 + position
        _reorder_storage(draft, storage_axis, order_tuple)
        replacement = _axis_spec_replacing(
            selected,
            coordinates=reordered,
            coordinate_labels=reordered_labels,
            index_origin=0,
        )
        if domain == "repeat":
            draft["repeat_axis"] = replacement
        else:
            axes = list(draft["cell_axes"])
            axes[position] = replacement
            draft["cell_axes"] = axes
        return True
    if domain == "point":
        _reorder_storage(draft, 1, order_tuple)
        _replace_point_columns_for_rows(draft, order_tuple)
        topology = draft["grid_topology"]
        if topology is not None:
            draft["grid_topology"] = _topology_replacing(
                topology,
                row_to_cell=tuple(topology.row_to_cell[row] for row in order_tuple),
            )
        return True
    topology = selected
    inverse = {old: new for new, old in enumerate(order_tuple)}
    mapping = tuple(
        tuple(
            inverse[value] if dim == position else value
            for dim, value in enumerate(cell)
        )
        for cell in topology.row_to_cell
    )
    domains = list(topology.coordinate_domains)
    domains[position] = reordered
    topology_labels = topology.coordinate_labels
    if topology_labels is not None:
        edited = list(topology_labels)
        if edited[position] is not None:
            edited[position] = reordered_labels
        topology_labels = tuple(edited)
    draft["grid_topology"] = _topology_replacing(
        topology,
        coordinate_domains=domains,
        row_to_cell=mapping,
        coordinate_labels=topology_labels,
    )
    return True


def _add_axis(draft: dict[str, object], domain: str) -> str:
    from zlc_data import (
        AxisSpec,
        COMPONENT,
        GridTopology,
        PointColumn,
        SCAN_POINT,
    )

    selected = str(domain)
    rows = int(np.asarray(draft["values"]).shape[1])
    if selected == "point":
        axis_id = _new_axis_id(draft, "point")
        columns = list(draft["point_columns"])
        columns.append(
            PointColumn(
                axis_id,
                f"point {len(columns) + 1}",
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(range(rows)),
            )
        )
        draft["point_columns"] = columns
    elif selected == "grid":
        axis_id = _new_axis_id(draft, "grid")
        topology = draft["grid_topology"]
        if topology is None:
            coordinates = tuple(range(rows))
            topology = GridTopology(
                (axis_id,),
                (coordinates,),
                tuple((index,) for index in range(rows)),
            )
            column_values = coordinates
        else:
            topology = GridTopology(
                (*topology.dimension_ids, axis_id),
                (*topology.coordinate_domains, (0,)),
                tuple((*cell, 0) for cell in topology.row_to_cell),
                (
                    None
                    if topology.coordinate_labels is None
                    else (*topology.coordinate_labels, None)
                ),
            )
            column_values = (0,) * rows
        draft["grid_topology"] = topology
        columns = list(draft["point_columns"])
        columns.append(
            PointColumn(
                axis_id,
                f"grid {len(topology.dimension_ids)}",
                SCAN_POINT,
                PointColumn.NUMERIC,
                column_values,
            )
        )
        draft["point_columns"] = columns
    elif selected == "cell":
        axis_id = _new_axis_id(draft, "component")
        new_axis = AxisSpec(axis_id, "component", COMPONENT, 1, (0,))
        axes = list(draft["cell_axes"])
        if len(axes) == 1 and str(axes[0].role) == "scalar":
            axes = [new_axis]
        else:
            axes.append(new_axis)
            draft["values"] = np.expand_dims(np.asarray(draft["values"]), -1)
            draft["validity"] = np.expand_dims(
                np.asarray(draft["validity"], dtype=np.bool_), -1
            )
            if draft["sigma"] is not None:
                draft["sigma"] = np.expand_dims(
                    np.asarray(draft["sigma"], dtype=np.float64), -1
                )
        draft["cell_axes"] = axes
    else:
        raise ValueError(f"cannot add a {selected!r} Dataset axis")
    draft["selected_axis"] = str(axis_id)
    return str(axis_id)


def _remove_axis(draft: dict[str, object], axis_id: str) -> None:
    from zlc_data.axis import SCALAR_AXIS

    domain, position, selected = _axis_location(draft, axis_id)
    if domain == "repeat":
        raise ValueError("the Dataset repeat axis cannot be removed")
    if domain == "point":
        columns = list(draft["point_columns"])
        columns.pop(position)
        draft["point_columns"] = columns
    elif domain == "grid":
        topology = selected
        ids = list(topology.dimension_ids)
        ids.pop(position)
        if not ids:
            draft["grid_topology"] = None
        else:
            domains = list(topology.coordinate_domains)
            domains.pop(position)
            mapping = tuple(
                tuple(value for dim, value in enumerate(cell) if dim != position)
                for cell in topology.row_to_cell
            )
            if len(set(mapping)) != len(mapping):
                raise ValueError(
                    "remove point rows that would overlap before removing this Grid dimension"
                )
            labels = topology.coordinate_labels
            if labels is not None:
                labels = tuple(value for dim, value in enumerate(labels) if dim != position)
            draft["grid_topology"] = _topology_replacing(
                topology,
                dimension_ids=ids,
                coordinate_domains=domains,
                row_to_cell=mapping,
                coordinate_labels=labels,
            )
        found = _point_column(draft, axis_id)
        if found is not None:
            columns = list(draft["point_columns"])
            columns.pop(found[0])
            draft["point_columns"] = columns
    else:
        axes = list(draft["cell_axes"])
        axis = axes[position]
        if str(axis.role) == "scalar":
            raise ValueError("the scalar carrier cannot be removed")
        if int(axis.size) != 1:
            raise ValueError("remove all but one coordinate before removing a Cell axis")
        axes.pop(position)
        if axes:
            draft["values"] = np.squeeze(np.asarray(draft["values"]), axis=2 + position)
            draft["validity"] = np.squeeze(
                np.asarray(draft["validity"], dtype=np.bool_), axis=2 + position
            )
            if draft["sigma"] is not None:
                draft["sigma"] = np.squeeze(
                    np.asarray(draft["sigma"], dtype=np.float64), axis=2 + position
                )
        else:
            axes = [SCALAR_AXIS]
        draft["cell_axes"] = axes
    draft["selected_axis"] = str(draft["repeat_axis"].axis_id)


def _move_axis(draft: dict[str, object], axis_id: str, delta: int) -> bool:
    domain, position, selected = _axis_location(draft, axis_id)
    destination = position + int(delta)
    if domain == "repeat":
        return False
    if domain == "point":
        columns = list(draft["point_columns"])
        topology = draft["grid_topology"]
        grid_ids = set() if topology is None else set(topology.dimension_ids)
        ordinary = [index for index, column in enumerate(columns) if column.coordinate_id not in grid_ids]
        ordinal = ordinary.index(position)
        target_ordinal = ordinal + int(delta)
        if not 0 <= target_ordinal < len(ordinary):
            return False
        target = ordinary[target_ordinal]
        item = columns.pop(position)
        columns.insert(target, item)
        draft["point_columns"] = columns
        return True
    if domain == "grid":
        topology = selected
        if not 0 <= destination < len(topology.dimension_ids):
            return False
        order = list(range(len(topology.dimension_ids)))
        moved = order.pop(position)
        order.insert(destination, moved)
        labels = topology.coordinate_labels
        draft["grid_topology"] = _topology_replacing(
            topology,
            dimension_ids=tuple(topology.dimension_ids[index] for index in order),
            coordinate_domains=tuple(topology.coordinate_domains[index] for index in order),
            row_to_cell=tuple(tuple(cell[index] for index in order) for cell in topology.row_to_cell),
            coordinate_labels=(
                None if labels is None else tuple(labels[index] for index in order)
            ),
        )
        return True
    axes = list(draft["cell_axes"])
    if not 0 <= destination < len(axes):
        return False
    item = axes.pop(position)
    axes.insert(destination, item)
    draft["cell_axes"] = axes
    for key in ("values", "validity", "sigma"):
        if draft[key] is not None:
            draft[key] = np.moveaxis(
                np.asarray(draft[key]), 2 + position, 2 + destination
            ).copy()
    return True


def _resize_axis(draft: dict[str, object], axis_id: str, size: object) -> bool:
    from zlc_data import GridTopology

    wanted = int(size)
    if isinstance(size, bool) or wanted < 1:
        raise ValueError("axis size must be a positive integer")
    domain, position, selected = _axis_location(draft, axis_id)
    current_values = tuple(_axis_values(draft, axis_id))
    current = len(current_values)
    if wanted == current:
        return False
    if domain == "point":
        if draft["grid_topology"] is not None or len(tuple(draft["point_columns"])) != 1:
            raise ValueError(
                "Point row count can be resized only for one column without Grid topology"
            )
    if domain == "grid":
        topology = selected
        complete, _canonical = _grid_layout(draft)
        topology_ids = set(topology.dimension_ids)
        if not complete or any(
            column.coordinate_id not in topology_ids
            for column in draft["point_columns"]
        ):
            raise ValueError(
                "Grid size can change only for a complete grid without unrelated point columns"
            )
    numeric = (
        selected.value_kind == "NUMERIC"
        if domain == "point"
        else _grid_value_kind(draft, axis_id) == "NUMERIC"
        if domain == "grid"
        else all(type(value) in (int, float) for value in current_values)
    )
    labels = _axis_labels(draft, axis_id)
    if wanted < current:
        if domain == "grid":
            return _remove_coordinates(draft, axis_id, range(wanted, current))
        remove = tuple(range(wanted, current))
        storage_axis = 0 if domain == "repeat" else 1 if domain == "point" else 2 + position
        _delete_storage(draft, storage_axis, remove)
        values = current_values[:wanted]
        resized_labels = None if labels is None else labels[:wanted]
    else:
        added = wanted - current
        values = _extend_coordinates(current_values, added, numeric=numeric)
        resized_labels = (
            None
            if labels is None
            else (*labels, *(str(value) for value in values[current:]))
        )
        if domain == "grid":
            topology = selected
            domains = list(topology.coordinate_domains)
            domains[position] = values
            ranges = [range(len(domain_values)) for domain_values in domains]
            additions = []
            for chosen in range(current, wanted):
                for rest in product(
                    *(values_range for dim, values_range in enumerate(ranges) if dim != position)
                ):
                    cell = []
                    iterator = iter(rest)
                    for dim in range(len(domains)):
                        cell.append(chosen if dim == position else next(iterator))
                    additions.append(tuple(cell))
            mapping = (*topology.row_to_cell, *additions)
            topology_labels = topology.coordinate_labels
            if topology_labels is not None:
                edited = list(topology_labels)
                if edited[position] is not None:
                    edited[position] = resized_labels
                topology_labels = tuple(edited)
            replacement = GridTopology(
                topology.dimension_ids,
                tuple(domains),
                mapping,
                topology_labels,
            )
            _append_point_storage(draft, len(additions))
            columns = []
            for column in draft["point_columns"]:
                dim = replacement.dimension_ids.index(column.coordinate_id)
                column_values = tuple(
                    replacement.coordinate_domains[dim][cell[dim]] for cell in mapping
                )
                domain_labels = (
                    None
                    if replacement.coordinate_labels is None
                    else replacement.coordinate_labels[dim]
                )
                column_labels = (
                    None
                    if domain_labels is None
                    else tuple(domain_labels[cell[dim]] for cell in mapping)
                )
                columns.append(
                    _point_column_replacing(
                        column,
                        values=column_values,
                        coordinate_labels=column_labels,
                    )
                )
            draft["point_columns"] = columns
            draft["grid_topology"] = replacement
            draft["message"] = (
                f"Resized Grid axis to {wanted}; added {len(additions)} physical point row(s)"
            )
            return True
        storage_axis = 0 if domain == "repeat" else 1 if domain == "point" else 2 + position
        _grow_storage(draft, storage_axis, added)

    if domain == "repeat":
        draft["repeat_axis"] = _axis_spec_replacing(
            selected,
            size=wanted,
            coordinates=values,
            coordinate_labels=resized_labels,
            index_origin=0,
        )
    elif domain == "point":
        columns = list(draft["point_columns"])
        columns[position] = _point_column_replacing(
            selected,
            values=values,
            coordinate_labels=resized_labels,
        )
        draft["point_columns"] = columns
    else:
        axes = list(draft["cell_axes"])
        axes[position] = _axis_spec_replacing(
            selected,
            size=wanted,
            coordinates=values,
            coordinate_labels=resized_labels,
            index_origin=0,
        )
        draft["cell_axes"] = axes
    return True


def _set_axis_field(
    draft: dict[str, object], axis_id: str, field: str, value: object
) -> bool:
    from zlc_data import AxisRoleId, PointColumn, SCAN_POINT, point_domain_admits

    domain, position, selected = _axis_location(draft, axis_id)
    field = str(field)
    if field not in {"name", "role", "unit", "coordinate_frame", "value_kind", "size"}:
        raise ValueError(f"unknown axis field {field!r}")
    if field == "size":
        return _resize_axis(draft, axis_id, value)
    if field == "value_kind":
        if domain not in {"point", "grid"}:
            raise ValueError("only Point/Grid coordinates have a value kind")
        kind = str(value)
        if kind not in {"NUMERIC", "TEXT"}:
            raise ValueError("coordinate type must be Numeric or Text")
        current_kind = (
            selected.value_kind
            if domain == "point"
            else _grid_value_kind(draft, axis_id)
        )
        if kind == current_kind:
            return False
        values = _axis_values(draft, axis_id)
        converted = tuple(
            None
            if item is None
            else _parse_coordinate(
                item,
                numeric=kind == "NUMERIC",
                allow_none=domain == "point",
            )
            for item in values
        )
        if domain == "grid" and len(set(converted)) != len(converted):
            raise ValueError("coordinate type conversion would merge Grid coordinates")
        if domain == "point":
            columns = list(draft["point_columns"])
            columns[position] = _point_column_replacing(
                selected,
                value_kind=kind,
                values=converted,
                unit=None if kind == "TEXT" else selected.unit,
                coordinate_labels=selected.coordinate_labels,
            )
            draft["point_columns"] = columns
        else:
            topology = selected
            domains = list(topology.coordinate_domains)
            domains[position] = converted
            draft["grid_topology"] = _topology_replacing(
                topology,
                coordinate_domains=domains,
            )
            found = _point_column(draft, axis_id)
            if found is not None:
                column_index, column = found
                mapped = tuple(converted[cell[position]] for cell in topology.row_to_cell)
                domain_labels = (
                    None
                    if topology.coordinate_labels is None
                    else topology.coordinate_labels[position]
                )
                mapped_labels = (
                    None
                    if domain_labels is None
                    else tuple(
                        domain_labels[cell[position]] for cell in topology.row_to_cell
                    )
                )
                columns = list(draft["point_columns"])
                columns[column_index] = _point_column_replacing(
                    column,
                    value_kind=kind,
                    values=mapped,
                    unit=None if kind == "TEXT" else column.unit,
                    coordinate_labels=mapped_labels,
                )
                draft["point_columns"] = columns
        return True
    if field == "name":
        normalized: object = str(value).strip()
        if not normalized:
            raise ValueError("axis name cannot be blank")
    elif field == "role":
        normalized = AxisRoleId(str(value))
        if domain == "repeat" and str(normalized) != "repeat":
            raise ValueError("the Dataset repeat role cannot change")
        if domain in {"point", "grid"} and not point_domain_admits(normalized):
            raise ValueError("that role is not valid in the point/grid domain")
    else:
        text = str(value).strip()
        normalized = None if not text else text
        if field == "unit" and domain in {"point", "grid"} and (
            (
                selected.value_kind
                if domain == "point"
                else _grid_value_kind(draft, axis_id)
            )
            == "TEXT"
        ):
            if normalized is not None:
                raise ValueError("Text coordinates cannot declare a unit")
    found = _point_column(draft, axis_id) if domain == "grid" else None
    metadata = found[1] if found is not None else selected
    current = (
        str(axis_id)
        if domain == "grid" and found is None and field == "name"
        else "scan-point"
        if domain == "grid" and found is None and field == "role"
        else None
        if domain == "grid" and found is None and field in {"unit", "coordinate_frame"}
        else getattr(metadata, field)
    )
    if normalized == current or str(normalized) == str(current):
        return False
    if domain in {"repeat", "cell"}:
        replacement = _axis_spec_replacing(selected, **{field: normalized})
        if domain == "repeat":
            draft["repeat_axis"] = replacement
        else:
            axes = list(draft["cell_axes"])
            axes[position] = replacement
            draft["cell_axes"] = axes
        return True
    found = _point_column(draft, axis_id)
    if found is None:
        topology = selected
        values = tuple(
            topology.coordinate_domains[position][cell[position]]
            for cell in topology.row_to_cell
        )
        column = PointColumn(
            topology.dimension_ids[position],
            str(topology.dimension_ids[position]),
            SCAN_POINT,
            (
                PointColumn.NUMERIC
                if all(type(item) in (int, float) for item in values)
                else PointColumn.TEXT
            ),
            values,
        )
        columns = list(draft["point_columns"])
        columns.append(column)
        draft["point_columns"] = columns
        found = (len(columns) - 1, column)
    column_index, column = found
    columns = list(draft["point_columns"])
    columns[column_index] = _point_column_replacing(
        column,
        **{field: normalized},
    )
    draft["point_columns"] = columns
    return True


def _set_dtype(draft: dict[str, object], value: object) -> bool:
    selected = np.dtype(str(value)).newbyteorder("<")
    if selected.str not in _MANUAL_DTYPES:
        raise ValueError(f"unsupported manual data type {selected}")
    if selected == np.dtype(draft["dtype"]):
        return False
    current = np.asarray(draft["values"])
    valid = np.asarray(draft["validity"], dtype=np.bool_)
    if bool(np.any(~valid)):
        current = np.array(current, copy=True)
        current[~valid] = 0
    selected_values = current[valid]
    real = np.real(selected_values)
    imaginary = np.imag(selected_values)
    if selected.kind != "c" and bool(np.any(imaginary != 0)):
        raise ValueError("that data type cannot represent complex values")
    if selected.kind == "b" and bool(
        np.any((real != 0) & (real != 1))
    ):
        raise ValueError("boolean data can represent only 0 and 1 exactly")
    if selected.kind in "iu":
        limits = np.iinfo(selected)
        if bool(
            np.any(~np.isfinite(real))
            or np.any(real != np.floor(real))
            or np.any(real < limits.min)
            or np.any(real > limits.max)
        ):
            raise ValueError(f"existing values lie outside exact {selected.name} values")
    if selected.kind == "f":
        limits = np.finfo(selected)
        finite = np.isfinite(real)
        if bool(np.any(np.abs(real[finite]) > limits.max)):
            raise ValueError(f"existing values lie outside {selected.name} range")
    converted = current.astype(selected)
    if bool(np.any(valid)):
        restored = converted.astype(current.dtype)
        if not np.array_equal(current[valid], restored[valid], equal_nan=True):
            raise ValueError("that data type would change existing valid values")
    draft["values"] = converted
    draft["dtype"] = selected
    return True


def _parse_value(text: object, dtype: np.dtype) -> object:
    value = str(text).strip()
    if dtype.kind == "b":
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"{value!r} is not a boolean value")
    try:
        return np.asarray(value, dtype=dtype).item()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{value!r} is not a {dtype.name} value") from error


def _set_table_cells(
    draft: dict[str, object], component: str, cells: object
) -> bool:
    selected = str(component)
    if selected not in {"values", "validity", "sigma"}:
        raise ValueError(f"unknown data table {selected!r}")
    if selected == "sigma" and draft["sigma"] is None:
        raise ValueError("enable sigma before editing it")
    parsed = []
    for row, column, text in tuple(cells):
        index = _table_index(draft, int(row), int(column))
        if any(
            value < 0 or value >= size
            for value, size in zip(index, np.asarray(draft["values"]).shape, strict=True)
        ):
            raise IndexError("data edit lies outside the visible table")
        raw = str(text).strip()
        if selected == "values":
            value = None if not raw else _parse_value(raw, np.dtype(draft["dtype"]))
        elif selected == "validity":
            value = False if not raw else bool(_parse_value(raw, np.dtype("?")))
        else:
            if not raw:
                value = np.nan
            else:
                value = float(raw)
                if value < 0.0:
                    raise ValueError("sample sigma must be non-negative")
        parsed.append((index, value))
    changed = False
    for index, value in parsed:
        if selected == "values":
            if value is None:
                changed = changed or bool(
                    np.asarray(draft["validity"])[index]
                    or np.asarray(draft["values"])[index] != 0
                    or (
                        draft["sigma"] is not None
                        and not np.isnan(np.asarray(draft["sigma"])[index])
                    )
                )
                np.asarray(draft["values"])[index] = 0
                np.asarray(draft["validity"])[index] = False
                if draft["sigma"] is not None:
                    np.asarray(draft["sigma"])[index] = np.nan
            else:
                changed = changed or bool(
                    not np.asarray(draft["validity"])[index]
                    or np.asarray(draft["values"])[index] != value
                )
                np.asarray(draft["values"])[index] = value
                np.asarray(draft["validity"])[index] = True
        elif selected == "validity":
            changed = changed or bool(np.asarray(draft["validity"])[index] != value)
            np.asarray(draft["validity"])[index] = value
        else:
            previous = np.asarray(draft["sigma"])[index]
            changed = changed or bool(
                not (
                    np.isnan(previous)
                    and np.isnan(value)
                    or previous == value
                )
            )
            np.asarray(draft["sigma"])[index] = value
    return changed


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
        *,
        owner_id: str = "",
        data_signal: str = "",
        run_record: Mapping[str, object] | None = None,
    ) -> None:
        from zlc_plot.primitives import ImageFrame
        from zlc_runtime import DatasetOutputDeclaration

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset)).strip("._")
        safe = safe or f"dataset-{index + 1}"
        self.instance_id = str(owner_id or f"figure-{serial}-{index + 1}")
        self.dataset = str(dataset)
        self.plot_input = plot_input
        self.path = Path(path)
        self._data_signal = str(data_signal or f"@figure/{serial}/{safe}")
        self._overlay_signal = f"{self._data_signal}/overlay"
        self._run_record = None if run_record is None else dict(run_record)
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
        run_record: dict[str, object] = (
            {
                "node": self.instance_id,
                "parameters": {
                    "archive": str(self.path),
                    "dataset": self.dataset,
                },
            }
            if self._run_record is None
            else dict(self._run_record)
        )
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
                ),
                run_record,
            )
        plane.begin_generation(self)
        plane.commit_live(self, outputs)
        plane.seal_committed(self)
        return plane.latest_publication(self.data_signal)


def _manual_lineage(
    signal_plane: object,
    publication: object,
    inherited: Mapping[str, object],
) -> dict[str, object]:
    """Append one real manual Runtime event to an existing Figure DAG."""

    from .panel_save import capture_run_chain

    if set(inherited) != {"root", "nodes", "device_settings"}:
        raise ValueError("inherited Figure lineage fields differ")
    nodes = [deepcopy(dict(node)) for node in tuple(inherited["nodes"])]
    if not isinstance(inherited["device_settings"], list):
        raise TypeError("inherited Figure device settings must be an array")
    captured = capture_run_chain(signal_plane, publication)
    current_nodes = tuple(captured["nodes"])
    if len(current_nodes) != 1 or current_nodes[0]["parents"]:
        raise RuntimeError("manual Dataset publication was not one root event")
    used = {str(node.get("id")) for node in nodes}
    serial = len(nodes) + 1
    node_id = f"manual-{serial}"
    while node_id in used:
        serial += 1
        node_id = f"manual-{serial}"
    node = deepcopy(dict(current_nodes[0]))
    node["id"] = node_id
    old_root = inherited["root"]
    node["parents"] = [] if old_root is None else [str(old_root)]
    nodes.append(node)
    return {
        "root": node_id,
        "nodes": nodes,
        "device_settings": deepcopy(list(inherited["device_settings"])),
    }


def _manual_plot_input(draft: Mapping[str, object], snapshot: object) -> object:
    """Keep an archived image overlay only while its coordinate contract matches."""

    from zlc_plot.primitives import ImageFrame

    overlay = draft["source_overlay"]
    source = draft["source_snapshot"]
    if overlay is None:
        return snapshot
    old = source.block.schema
    new = snapshot.block.schema
    compatible = bool(
        old.repeat_axis == new.repeat_axis
        and old.point_table == new.point_table
        and old.grid_topology == new.grid_topology
        and old.cell_schema.data_axes == new.cell_schema.data_axes
    )
    return ImageFrame(snapshot, overlay) if compatible else snapshot


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


def _manual_lineage_needs_saved_source(
    root: str | None,
    nodes: Mapping[str, Mapping[str, Any]],
) -> bool:
    if root is None:
        return False
    current = root
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        node = nodes.get(current)
        if node is None:
            return False
        record = node["record"]
        operation = record.get("operation") if isinstance(record, Mapping) else None
        if operation not in {"manual-create", "manual-edit"}:
            return False
        parents = tuple(node["parents"])
        if not parents:
            return operation == "manual-edit"
        if len(parents) != 1:
            return False
        current = str(parents[0])
    return False


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

    root, nodes = _lineage_nodes(value)
    rows: list[tuple[str, object]] = []
    saved = (
        _source_run_record(source)
        if not nodes or _manual_lineage_needs_saved_source(root, nodes)
        else None
    )
    if saved is not None:
        name, output, record = saved
        projected = _logic_record(record)
        assert isinstance(projected, Mapping)
        rows.append((
            f"{name} · saved source" if nodes else name,
            {
                "outputs": [output],
                **{
                    str(key): item
                    for key, item in projected.items()
                    if key != "node"
                },
            },
        ))
    if not nodes:
        return tuple(rows)
    names = [_logic_name(node) for node in nodes.values()]
    repeated = {name for name, count in Counter(names).items() if count > 1}
    for node in nodes.values():
        name = _logic_name(node)
        event = node["event"]
        label = (
            f"{name} · sequence {event['sequence']} · {str(event['generation'])[:8]}"
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
                        f" · sequence {event['sequence']} · {str(event['generation'])[:8]}"
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
    saved = (
        _source_run_record(source)
        if _manual_lineage_needs_saved_source(root, nodes)
        else None
    )
    if saved is not None:
        name, output, record = saved
        source_id = "logic:saved-source"
        graph_nodes.append(
            {
                "id": source_id,
                "kind": "logic",
                "title": name,
                "subtitle": f"{output} · saved source",
                "root": False,
                "tooltip": (
                    f"Logic: {name}\nSaved result: {output}\n"
                    "This Task record has no exact Runtime event edge in the archive."
                ),
            }
        )
        source_devices: dict[str, list[str]] = {}
        for role, device_key in _record_device_associations(record):
            source_devices.setdefault(device_key, []).append(role)
        existing_device_ids = {
            str(node["id"])
            for node in graph_nodes
            if str(node.get("kind")) == "device"
        }
        for device_key, roles in source_devices.items():
            device_id = f"device:{device_key}"
            if device_id not in existing_device_ids:
                graph_nodes.append(
                    {
                        "id": device_id,
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
                existing_device_ids.add(device_id)
            graph_edges.append(
                {
                    "source": device_id,
                    "target": source_id,
                    "kind": "device",
                    "label": ", ".join(roles),
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
    saved = (
        _source_run_record(source)
        if not nodes or _manual_lineage_needs_saved_source(_root, nodes)
        else None
    )
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
        self._archive_data: dict[str, dict[str, object]] = {}
        self._data_drafts: dict[str, dict[str, object]] = {}
        self._data_source_editors: dict[str, str] = {}
        self._archive_serial = 0
        self._data_serial = 0
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
        for signal_name, handler in (
            ("new_data_requested", self.new_data),
            ("edit_data_requested", self.edit_data),
            ("data_editor_intent", self.data_editor_intent),
            ("data_editor_closed", self.close_data_editor),
        ):
            signal = getattr(self.view, signal_name, None)
            if signal is not None:
                signal.connect(handler)
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
        if any(
            bool(draft["modified"] or draft["unsaved"])
            for draft in self._data_drafts.values()
        ):
            self.view.set_status(
                "Save or discard the open data working copy before opening another Figure",
                error=True,
            )
            return
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
            sections = info["sections"]
            return (
                resolved,
                description,
                tuple(loaded),
                serial,
                deepcopy(dict(sections["lineage"])),
                deepcopy(dict(sections["source"])),
            )

        self._submit(
            f"opening {requested.name}…",
            prepare,
            self._accept_runtime_archive,
            f"cannot open {requested.name}",
        )

    def _accept_runtime_archive(self, result: object) -> None:
        from .panel_catalog import task_console_panel_identity_for_spec

        resolved, description, loaded, serial, source_lineage, source_document = result
        panel_presenter = self._panel_presenter
        plane = self._signal_plane
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
                kind, cell_kind = task_console_panel_identity_for_spec(spec)
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
                    kind=kind,
                    cell_kind=cell_kind,
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
        for draft in self._data_drafts.values():
            producer = draft.get("producer")
            if producer is not None:
                try:
                    plane.retire(producer)
                except (LookupError, RuntimeError):
                    pass
            closer = getattr(self.view, "close_data_editor", None)
            if callable(closer):
                closer(str(draft["editor_id"]))
        self._archive_producers = tuple(producers)
        self._data_drafts.clear()
        self._data_source_editors.clear()
        self._archive_data.clear()
        labels = dict(description.datasets)
        source_title = str(source_document.get("title") or "").strip()
        for key, plot_input, recipe, described in loaded:
            snapshot = getattr(plot_input, "snapshot", plot_input)
            overlay = getattr(plot_input, "overlay", None)
            choice = f"archive:{key}"
            self._archive_data[choice] = {
                "name": source_title or key,
                "display_label": labels.get(key, key),
                "key": key,
                "snapshot": snapshot,
                "overlay": overlay,
                "recipe": dict(recipe),
                "described": described,
                "path": resolved,
                "lineage": deepcopy(dict(source_lineage)),
                "source": deepcopy(dict(source_document)),
            }
        self._archive_serial = int(serial)
        self.path = resolved
        self.description = description
        self.view.set_title(description.name or resolved.stem)
        self.view.set_path(str(resolved))
        self.view.set_archive_info(description.tabs, description.flow)
        self._project_data_choices()
        self.view.set_status(
            "opened; choose a signal in Setting"
            if not published
            else f"showing {published[0][0].data_signal}"
        )
        panel_presenter.beat()

    def beat(self) -> None:
        self._panel_presenter.beat()
        self._refresh_data_save_states()

    def commit_surfaces(self) -> None:
        self._panel_presenter.commit_surfaces()
        self._refresh_data_save_states()

    def _project_data_choices(self, current: str = "") -> None:
        setter = getattr(self.view, "set_editable_data_choices", None)
        if not callable(setter):
            return
        multiple_archive_datasets = len(self._archive_data) > 1
        rows = [
            (
                key,
                (
                    f"{entry['display_label']} · {entry['key']}"
                    if multiple_archive_datasets
                    else str(entry["display_label"])
                ),
            )
            for key, entry in self._archive_data.items()
        ]
        rows.extend(
            (f"manual:{editor_id}", str(draft["name"]))
            for editor_id, draft in self._data_drafts.items()
            if draft["publication"] is not None
        )
        setter(tuple(rows), current=str(current))

    def _show_data_draft(self, draft: Mapping[str, object]) -> None:
        editor_id = str(draft["editor_id"])
        projection = _data_projection(draft)
        if bool(getattr(self.view, "has_data_editor", lambda _key: False)(editor_id)):
            self.view.update_data_editor(editor_id, projection)
            self.view.focus_data_editor(editor_id)
        else:
            self.view.open_data_editor(
                editor_id,
                projection,
                title=f"Data · {draft['name']}",
            )

    def new_data(self) -> None:
        """Open one immediately useful canonical Curve working copy."""

        self._data_serial += 1
        editor_id = f"data-{self._data_serial}"
        snapshot = _new_manual_snapshot()
        draft = _draft_from_snapshot(
            snapshot,
            editor_id=editor_id,
            name=f"Manual data {self._data_serial}",
            note="",
            source_text="New manual Dataset",
            source_path=None,
            source_dataset="",
            source_lineage={"root": None, "nodes": [], "device_settings": []},
            source_document={},
            recipe=None,
            described=None,
            overlay=None,
        )
        self._data_drafts[editor_id] = draft
        self._show_data_draft(draft)

    def _open_archive_data_draft(self, source_key: str) -> None:
        source = self._archive_data[source_key]
        self._data_serial += 1
        editor_id = f"data-{self._data_serial}"
        draft = _draft_from_snapshot(
            source["snapshot"],
            editor_id=editor_id,
            name=str(source["name"]),
            note="",
            source_text=f"Copy of {Path(source['path']).name} · {source['key']}",
            source_path=Path(source["path"]),
            source_dataset=str(source["key"]),
            source_lineage=source["lineage"],
            source_document=source["source"],
            recipe=source["recipe"],
            described=source["described"],
            overlay=source["overlay"],
        )
        self._data_drafts[editor_id] = draft
        self._data_source_editors[source_key] = editor_id
        self._show_data_draft(draft)

    def edit_data(self, source_key: str) -> None:
        key = str(source_key)
        if key.startswith("manual:"):
            editor_id = key.split(":", 1)[1]
            draft = self._data_drafts.get(editor_id)
            if draft is not None:
                self._show_data_draft(draft)
            return
        existing = self._data_source_editors.get(key)
        if existing is not None and existing in self._data_drafts:
            self._show_data_draft(self._data_drafts[existing])
            return
        source = self._archive_data.get(key)
        if source is None:
            self.view.set_status(f"unknown editable Dataset {key!r}", error=True)
            return
        if source["described"] is not None:
            self._open_archive_data_draft(key)
            return

        def describe() -> object:
            import zlc_plot

            plot_input = (
                source["snapshot"]
                if source["overlay"] is None
                else zlc_plot.ImageFrame(source["snapshot"], source["overlay"])
            )
            host = zlc_plot.open_figure_host(
                plot_input,
                source["recipe"],
                device_pixel_ratio=float(self.view.device_pixel_ratio()),
            )
            try:
                result = self._await(host.describe_display())
                return getattr(result, "value", result)
            finally:
                self._close_host(host)

        def accepted(description: object) -> None:
            source["described"] = description
            self._open_archive_data_draft(key)

        self._submit(
            f"preparing {source['name']} for editing…",
            describe,
            accepted,
            f"cannot edit {source['name']}",
        )

    def close_data_editor(self, editor_id: str) -> bool:
        draft = self._data_drafts.get(str(editor_id))
        if draft is None:
            return False
        if bool(draft["modified"] or draft["unsaved"]):
            draft["message"] = "Save or discard this data working copy before closing"
            self._show_data_draft(draft)
            return False
        closer = getattr(self.view, "close_data_editor", None)
        closed = bool(callable(closer) and closer(str(editor_id)))
        if closed and draft["publication"] is None:
            self._data_drafts.pop(str(editor_id), None)
            self._data_source_editors = {
                key: value
                for key, value in self._data_source_editors.items()
                if value != str(editor_id)
            }
        return closed

    def _restore_data_draft(self, draft: dict[str, object]) -> None:
        snapshot = draft["applied_snapshot"] or draft["source_snapshot"]
        name = draft["applied_name"] or draft["initial_name"]
        note = draft["applied_note"] or draft["initial_note"]
        persistent = {
            key: draft[key]
            for key in (
                "editor_id",
                "source_text",
                "source_path",
                "source_dataset",
                "source_lineage",
                "source_document",
                "source_snapshot",
                "source_overlay",
                "recipe",
                "described",
                "producer_serial",
                "producer",
                "publication",
                "applied_snapshot",
                "applied_name",
                "applied_note",
                "lineage",
                "panel_id",
                "save_ready",
                "unsaved",
            )
        }
        restored = _draft_from_snapshot(
            snapshot,
            editor_id=str(draft["editor_id"]),
            name=str(name),
            note=str(note),
            source_text=str(draft["source_text"]),
            source_path=draft["source_path"],
            source_dataset=str(draft["source_dataset"]),
            source_lineage=draft["source_lineage"],
            source_document=draft["source_document"],
            recipe=draft["recipe"],
            described=draft["described"],
            overlay=draft["source_overlay"],
        )
        restored.update(persistent)
        restored["modified"] = False
        restored["message"] = "Edits discarded"
        draft.clear()
        draft.update(restored)

    def _discard_applied_data(self, draft: dict[str, object]) -> None:
        panel_id = str(draft["panel_id"])
        if panel_id:
            self._panel_presenter.remove_panel(panel_id)
        producer = draft["producer"]
        if producer is not None:
            self._signal_plane.retire(producer)
        for key, value in (
            ("producer", None),
            ("publication", None),
            ("applied_snapshot", None),
            ("applied_name", None),
            ("applied_note", None),
            ("lineage", None),
            ("panel_id", ""),
            ("save_ready", False),
            ("unsaved", False),
        ):
            draft[key] = value
        self._restore_data_draft(draft)
        draft["message"] = "Applied working copy discarded"

    def data_editor_intent(self, editor_id: str, intent: Mapping[str, object]) -> None:
        draft = self._data_drafts.get(str(editor_id))
        if draft is None:
            self.view.set_status(f"unknown data editor {editor_id!r}", error=True)
            return
        try:
            command = dict(intent)
            operation = str(command.pop("op"))
            draft["message"] = ""
            marks_dirty = True
            if operation in {
                "insert_coordinate",
                "remove_coordinates",
                "move_coordinate",
                "remove_axis",
            } or (
                operation == "set_axis_field"
                and str(command.get("field")) == "size"
            ):
                _require_complete_coordinate_labels(
                    draft, "changing coordinate structure"
                )
            if operation == "set_dataset_field":
                field = str(command["field"])
                value = command.get("value")
                if field == "dtype":
                    marks_dirty = _set_dtype(draft, value)
                elif field == "name":
                    selected = str(value).strip()
                    if not selected:
                        raise ValueError("Dataset name cannot be blank")
                    marks_dirty = selected != str(draft["name"])
                    draft["name"] = selected
                elif field == "unit":
                    text = str(value).strip()
                    selected = None if not text else text
                    marks_dirty = selected != draft["unit"]
                    draft["unit"] = selected
                elif field == "note":
                    selected = str(value).strip()
                    marks_dirty = selected != str(draft["note"])
                    draft["note"] = selected
                else:
                    raise ValueError(f"unknown Dataset field {field!r}")
            elif operation == "select_axis":
                _axis_location(draft, str(command["axis_id"]))
                draft["selected_axis"] = str(command["axis_id"])
                marks_dirty = False
            elif operation == "add_axis":
                _add_axis(draft, str(command["domain"]))
            elif operation == "remove_axis":
                _remove_axis(draft, str(command["axis_id"]))
            elif operation == "move_axis":
                marks_dirty = _move_axis(
                    draft, str(command["axis_id"]), int(command["delta"])
                )
            elif operation == "set_axis_field":
                marks_dirty = _set_axis_field(
                    draft,
                    str(command["axis_id"]),
                    str(command["field"]),
                    command.get("value"),
                )
            elif operation == "insert_coordinate":
                _insert_coordinate(draft, str(command["axis_id"]), int(command["after"]))
            elif operation == "remove_coordinates":
                marks_dirty = _remove_coordinates(
                    draft, str(command["axis_id"]), command["indices"]
                )
            elif operation == "move_coordinate":
                marks_dirty = _move_coordinate(
                    draft,
                    str(command["axis_id"]),
                    int(command["index"]),
                    int(command["delta"]),
                )
            elif operation == "set_coordinates":
                marks_dirty = _set_coordinate_cells(
                    draft, str(command["axis_id"]), command["cells"]
                )
            elif operation == "set_slice":
                axis_id = str(command["axis_id"])
                index = int(command["index"])
                dimension = next(
                    (
                        item
                        for item in _logical_dimensions(draft)
                        if str(item["axis_id"]) == axis_id
                    ),
                    None,
                )
                if dimension is None or not 0 <= index < len(dimension["coordinates"]):
                    raise IndexError("slice index lies outside its Dataset axis")
                draft["slices"][axis_id] = index
                marks_dirty = False
            elif operation == "set_component":
                component = str(command["component"])
                if component not in {"values", "validity", "sigma"}:
                    raise ValueError(f"unknown data component {component!r}")
                draft["component"] = component
                marks_dirty = False
            elif operation == "toggle_sigma":
                enabled = bool(command["enabled"])
                marks_dirty = enabled != (draft["sigma"] is not None)
                if enabled and draft["sigma"] is None:
                    draft["sigma"] = np.full(
                        np.asarray(draft["values"]).shape,
                        np.nan,
                        dtype=np.float64,
                    )
                elif not enabled:
                    draft["sigma"] = None
                    if draft["component"] == "sigma":
                        draft["component"] = "values"
            elif operation == "set_cells":
                marks_dirty = _set_table_cells(
                    draft, str(command["component"]), command["cells"]
                )
            elif operation == "discard":
                if bool(draft["modified"]):
                    self._restore_data_draft(draft)
                elif bool(draft["unsaved"]):
                    self._discard_applied_data(draft)
                marks_dirty = False
            elif operation == "apply_preview":
                draft["note"] = str(command.get("note", "")).strip()
                self._apply_data_draft(draft)
                marks_dirty = False
            elif operation == "save_as":
                draft["note"] = str(command.get("note", "")).strip()
                self._save_data_draft(draft, str(command["path"]))
                marks_dirty = False
            else:
                raise ValueError(f"unknown data edit operation {operation!r}")
            if marks_dirty:
                draft["modified"] = True
            self._show_data_draft(draft)
            self._project_data_choices(f"manual:{editor_id}")
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
            draft["message"] = str(error)
            self._show_data_draft(draft)
            self.view.set_status(f"cannot edit data: {error}", error=True)

    def _apply_data_draft(self, draft: dict[str, object]) -> None:
        from .panel_catalog import task_console_panel_identity_for_spec

        validity = np.asarray(draft["validity"], dtype=np.bool_)
        values = np.asarray(draft["values"])
        if bool(np.any(~validity)):
            values = np.array(values, copy=True)
            values[~validity] = 0
            draft["values"] = values
        snapshot = _manual_snapshot(draft)
        plot_input = _manual_plot_input(draft, snapshot)
        if draft["producer_serial"] is None:
            draft["producer_serial"] = self._data_serial
        serial = int(draft["producer_serial"])
        owner_id = f"manual-data-{serial}"
        signal = f"@figure/manual/{serial}/data"
        operation = "manual-create" if draft["source_path"] is None else "manual-edit"
        timestamp = datetime.now(timezone.utc).isoformat()
        record: dict[str, object] = {
            "node": owner_id,
            "operation": operation,
            "timestamp_utc": timestamp,
            "software": {
                "distribution": "zou-lab-control",
                "version": _manual_version(),
            },
            "dataset": {
                "name": str(draft["name"]),
                "schema_fingerprint": snapshot.block.schema.fingerprint,
                "shape": list(snapshot.block.values.shape),
                "dtype": snapshot.block.values.dtype.str,
            },
            "note": str(draft["note"]),
        }
        if draft["source_path"] is not None:
            record["input"] = {
                "archive": str(draft["source_path"]),
                "dataset": str(draft["source_dataset"]),
            }
        producer = _ArchiveDatasetProducer(
            serial,
            0,
            str(draft["name"]),
            plot_input,
            Path(draft["source_path"] or "manual"),
            owner_id=owner_id,
            data_signal=signal,
            run_record=record,
        )
        publication = producer.publish(self._signal_plane)
        draft["producer"] = producer
        draft["publication"] = publication
        draft["applied_snapshot"] = snapshot
        draft["applied_name"] = str(draft["name"])
        draft["applied_note"] = str(draft["note"])
        draft["lineage"] = _manual_lineage(
            self._signal_plane,
            publication,
            draft["source_lineage"],
        )
        draft["modified"] = False
        draft["unsaved"] = True
        draft["save_ready"] = False
        draft["message"] = "Applied; preparing the preview Panel"

        panel_id = str(draft["panel_id"])
        if panel_id and panel_id in self.panels:
            self._panel_presenter.update_panel_state(
                panel_id,
                {
                    "signal": signal,
                    "title": str(draft["name"]),
                    "overlay_signal": producer.overlay_signal,
                },
            )
            self._panel_presenter.board.owe_presentation((panel_id,))
        else:
            described = draft["described"]
            if described is None:
                binding = self._panel_presenter.add_panel(
                    signal,
                    snapshot,
                    title=str(draft["name"]),
                    kind="",
                    initial_publication=publication,
                )
            else:
                kind, cell_kind = task_console_panel_identity_for_spec(described.spec)
                semantic = {
                    str(name): value
                    for name, value in described.semantics.values.items()
                    if str(name) != "kind"
                }
                binding = self._panel_presenter.add_panel(
                    signal,
                    snapshot,
                    title=str(draft["name"]),
                    kind=kind,
                    cell_kind=cell_kind,
                    size=described.size,
                    semantic=semantic,
                    display=dict(described.display_state.values),
                    fit=dict(described.fit),
                    overlay_signal=producer.overlay_signal,
                    initial_publication=publication,
                )
                self._panel_presenter.restore_panel_description(
                    binding.panel_id,
                    described,
                )
            draft["panel_id"] = binding.panel_id
            self._active_panel_id = binding.panel_id
        self._panel_presenter.beat()

    def _save_data_draft(self, draft: dict[str, object], path: str) -> None:
        from dataclasses import replace
        from .panel_save import save_panel_figure

        if bool(draft["modified"]) or draft["publication"] is None:
            raise ValueError("Apply & preview the current data before saving")
        binding = self.panels.get(str(draft["panel_id"]))
        frozen = None if binding is None else binding.frozen_data
        if frozen is None or frozen.publication is not draft["publication"]:
            raise ValueError("the preview Panel has not accepted this data yet")
        selected = Path(path).expanduser()
        image_path = (
            selected.with_suffix(".png")
            if selected.suffix.lower() == ".npz"
            else selected
        )
        state = binding.state
        manual_frozen = replace(frozen, lineage=draft["lineage"])
        source = deepcopy(dict(draft["source_document"]))
        saved_publication = draft["publication"]

        def write() -> object:
            return save_panel_figure(
                image_path,
                state=state,
                frozen=manual_frozen,
                source=source,
            )

        def saved(written: object) -> None:
            current = draft["publication"] is saved_publication
            if current:
                draft["unsaved"] = False
            draft["message"] = (
                f"Saved {written.archive.name}"
                if current
                else f"Saved older preview to {written.archive.name}; current data is not saved"
            )
            self.view.set_status(
                f"saved {written.archive.name} and {written.image.name}"
            )
            if bool(getattr(self.view, "has_data_editor", lambda _key: False)(str(draft["editor_id"]))):
                self.view.update_data_editor(
                    str(draft["editor_id"]), _data_projection(draft)
                )

        self._submit(
            f"saving {draft['name']}…",
            write,
            saved,
            f"cannot save {draft['name']}",
        )

    def _refresh_data_save_states(self) -> None:
        for draft in self._data_drafts.values():
            binding = self.panels.get(str(draft["panel_id"]))
            frozen = None if binding is None else binding.frozen_data
            ready = bool(
                draft["publication"] is not None
                and frozen is not None
                and frozen.publication is draft["publication"]
            )
            if ready == bool(draft["save_ready"]):
                continue
            draft["save_ready"] = ready
            if ready and not bool(draft["modified"]):
                draft["message"] = "Preview ready"
            if bool(getattr(self.view, "has_data_editor", lambda _key: False)(str(draft["editor_id"]))):
                self.view.update_data_editor(
                    str(draft["editor_id"]), _data_projection(draft)
                )

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

        if self._closed:
            return True
        if any(
            bool(draft["modified"] or draft["unsaved"])
            for draft in self._data_drafts.values()
        ):
            self._close_requested = False
            self.view.set_status(
                "Save or discard the open data working copy before closing",
                error=True,
            )
            return False
        self._close_requested = True
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
            for draft in self._data_drafts.values():
                producer = draft.get("producer")
                if producer is not None:
                    try:
                        self._signal_plane.retire(producer)
                    except (LookupError, RuntimeError):
                        pass
            self._archive_producers = ()
            self._data_drafts.clear()
            self._signal_plane.close()
            self._runtime_closed = True
        if not self._close_worker():
            self._request_close()
            return False
        self._closed = True
        return True
