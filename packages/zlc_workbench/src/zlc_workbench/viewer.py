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
# The manual editor works on one dense logical tensor.  Repeat and Point are
# flattened only at the Dataset boundary; keeping them expanded here makes
# every axis obey the same Add/Edit/Delete and Rows/Columns/Scope rules.


def _manual_version() -> str:
    try:
        return version("zou-lab-control")
    except PackageNotFoundError:
        return "unknown"


def _axis_coordinates(axis: object) -> object:
    if axis.coordinates is not None:
        return tuple(axis.coordinates)
    return range(int(axis.index_origin), int(axis.index_origin) + int(axis.size))


def _parse_axis_value(text: object) -> object:
    from zlc_data import canonical_coordinate_scalar

    token = str(text).strip()
    if not token:
        raise ValueError("axis values cannot be blank")
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        value: object = token[1:-1]
    elif re.fullmatch(r"[+-]?\d+", token):
        value = int(token)
    else:
        try:
            value = float(token)
        except ValueError:
            value = token
    return canonical_coordinate_scalar(value, "axis value")


def _resized_coordinates(axis: object | None, length: int) -> tuple[object, ...] | None:
    if axis is None or axis.coordinates is None:
        return None
    wanted = int(length)
    values = list(axis.coordinates[:wanted])
    occupied = set(values)
    while len(values) < wanted:
        if values and all(type(value) in (int, float) for value in values):
            step = values[-1] - values[-2] if len(values) >= 2 else 1
            candidate: object = values[-1] + step
            while candidate in occupied:
                candidate = candidate + 1
        else:
            serial = len(values)
            candidate = f"{axis.name}-{serial}"
            while candidate in occupied:
                serial += 1
                candidate = f"{axis.name}-{serial}"
        values.append(candidate)
        occupied.add(candidate)
    return tuple(values)


def _mapped_domain(axes: tuple[object, ...]) -> object:
    from zlc_data import DomainSpec

    if not axes:
        return DomainSpec((1,), (), ())
    shape = tuple(int(axis.size) for axis in axes)
    size = int(np.prod(shape, dtype=np.int64))
    codes = np.indices(shape, dtype=np.int64).reshape((len(shape), size))
    return DomainSpec(
        (size,), axes, tuple(codes[position] for position in range(len(shape)))
    )


def _edited_mapped_domain(axes: tuple[object, ...], source: object) -> object:
    """Keep an existing carrier map until the operator changes its topology.

    Editing data, coordinates, names or units does not turn a sparse/serpentine
    acquisition into a Cartesian product.  Add/Delete, moving an axis between
    domains, or changing an axis length does change the topology and therefore
    intentionally starts a new dense authored map.
    """

    signature = tuple((axis.axis_id, int(axis.size)) for axis in axes)
    source_signature = tuple(
        (axis.axis_id, int(axis.size)) for axis in source.axes
    )
    if signature != source_signature:
        return _mapped_domain(axes)
    from zlc_data import DomainSpec

    return DomainSpec(tuple(source.shape), axes, source.axis_codes)


def _domain_flat_rows(domain: object) -> np.ndarray:
    if not domain.axes:
        return np.zeros(int(domain.size), dtype=np.intp)
    result = np.ravel_multi_index(
        tuple(domain.codes(axis.axis_id) for axis in domain.axes),
        domain.logical_shape,
    )
    if np.unique(result).size != result.size:
        raise ValueError("manual editing requires unique logical rows in each Dataset domain")
    return np.asarray(result, dtype=np.intp)


def _expand_snapshot_for_edit(snapshot: object) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    from zlc_data import expand_snapshot_validity

    schema = snapshot.block.schema
    repeat_shape = tuple(int(axis.size) for axis in schema.repeat_domain.axes)
    point_shape = tuple(int(axis.size) for axis in schema.point_domain.axes)
    cell_shape = tuple(schema.cell_domain.shape)
    dense_shape = (*repeat_shape, *point_shape, *cell_shape)
    repeat_size = int(np.prod(repeat_shape, dtype=np.int64)) if repeat_shape else 1
    point_size = int(np.prod(point_shape, dtype=np.int64)) if point_shape else 1
    values = np.zeros(dense_shape, dtype=schema.value_schema.dtype)
    validity = np.zeros(dense_shape, dtype=np.bool_)
    source_values = np.asarray(snapshot.block.values)
    source_validity = np.asarray(expand_snapshot_validity(snapshot), dtype=np.bool_)
    target_values = values.reshape((repeat_size, point_size, *cell_shape))
    target_validity = validity.reshape((repeat_size, point_size, *cell_shape))
    repeat_rows = _domain_flat_rows(schema.repeat_domain)
    point_rows = _domain_flat_rows(schema.point_domain)
    target_values[repeat_rows[:, None], point_rows[None, :]] = source_values
    target_validity[repeat_rows[:, None], point_rows[None, :]] = source_validity
    sigma = None
    if snapshot.block.sigma is not None:
        sigma = np.full(dense_shape, np.nan, dtype=np.float64)
        sigma.reshape((repeat_size, point_size, *cell_shape))[
            repeat_rows[:, None], point_rows[None, :]
        ] = np.asarray(snapshot.block.sigma, dtype=np.float64)
    return values, validity, sigma


def _visible_axes(draft: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        axis
        for axis in (
            *tuple(draft["repeat_axes"]),
            *tuple(draft["point_axes"]),
            *tuple(draft["cell_axes"]),
        )
        if str(axis.role) != "scalar"
    )


def _storage_axes(draft: Mapping[str, object]) -> tuple[object, ...]:
    return (
        *tuple(draft["repeat_axes"]),
        *tuple(draft["point_axes"]),
        *tuple(draft["cell_axes"]),
    )


def _normalize_table_axes(draft: dict[str, object]) -> None:
    ids = tuple(str(axis.axis_id) for axis in _visible_axes(draft))
    row = str(draft.get("row_axis") or "")
    column = str(draft.get("column_axis") or "")
    if row not in ids:
        row = ids[-2] if len(ids) >= 2 else ids[0] if ids else ""
    if column not in ids or column == row:
        column = ids[-1] if len(ids) >= 2 and ids[-1] != row else ""
    draft["row_axis"] = row
    draft["column_axis"] = column
    scopes = dict(draft.get("scopes", {}))
    for axis in _visible_axes(draft):
        axis_id = str(axis.axis_id)
        scopes[axis_id] = min(max(0, int(scopes.get(axis_id, 0))), int(axis.size) - 1)
    draft["scopes"] = {key: value for key, value in scopes.items() if key in ids}


def _new_manual_snapshot() -> object:
    """A useful Curve Dataset, not an empty shape the operator must repair."""

    from zlc_data import (
        AxisId,
        AxisSpec,
        DatasetSchema,
        DomainSpec,
        REPEAT,
        SCAN_POINT,
        SCALAR_DOMAIN,
        ValueSchema,
        owned_snapshot_from_arrays,
    )

    count = 16
    repeat = AxisSpec(AxisId("manual.repeat"), "repeat", REPEAT, 1, (0,))
    x = AxisSpec(AxisId("manual.x"), "x", SCAN_POINT, count, tuple(range(count)))
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((count,), (x,), (tuple(range(count)),)),
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("<f8")),
    )
    return owned_snapshot_from_arrays(
        schema,
        np.zeros(schema.physical_shape, dtype=schema.value_schema.dtype),
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
    schema = snapshot.block.schema
    values, validity, sigma = _expand_snapshot_for_edit(snapshot)
    axes = tuple(
        axis
        for axis in (
            *schema.point_domain.axes,
            *schema.cell_domain.axes,
            *schema.repeat_domain.axes,
        )
        if str(axis.role) != "scalar"
    )
    draft: dict[str, object] = {
        "editor_id": str(editor_id),
        "name": str(name),
        "initial_name": str(name),
        "dtype": schema.value_schema.dtype,
        "unit": schema.value_schema.value_unit,
        "note": str(note),
        "initial_note": str(note),
        "source_text": str(source_text),
        "source_path": source_path,
        "source_dataset": str(source_dataset),
        "source_lineage": deepcopy(dict(source_lineage)),
        "source_document": deepcopy(dict(source_document)),
        "source_snapshot": snapshot,
        "source_repeat_domain": schema.repeat_domain,
        "source_point_domain": schema.point_domain,
        "source_overlay": overlay,
        "recipe": None if recipe is None else dict(recipe),
        "described": described,
        "repeat_axes": list(schema.repeat_domain.axes),
        "point_axes": list(schema.point_domain.axes),
        "cell_axes": list(schema.cell_domain.axes),
        "validity_contract": schema.value_schema.validity_contract,
        "values": values,
        "validity": validity,
        "sigma": sigma,
        "selected_axis": "" if not axes else str(axes[0].axis_id),
        "row_axis": "",
        "column_axis": "",
        "scopes": {},
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
    _normalize_table_axes(draft)
    return draft


def _draft_schema(draft: Mapping[str, object]) -> object:
    from zlc_data import DatasetSchema, DomainSpec, ValidityContract, ValueSchema

    repeat_axes = tuple(draft["repeat_axes"])
    point_axes = tuple(draft["point_axes"])
    cell_axes = tuple(draft["cell_axes"])
    repeat_domain = _edited_mapped_domain(
        repeat_axes, draft["source_repeat_domain"]
    )
    point_domain = _edited_mapped_domain(
        point_axes, draft["source_point_domain"]
    )
    cell_domain = DomainSpec(
        tuple(int(axis.size) for axis in cell_axes), cell_axes
    )
    validity = _pack_manual_array(
        draft,
        np.asarray(draft["validity"], dtype=np.bool_),
        repeat_domain=repeat_domain,
        point_domain=point_domain,
    )
    previous = draft["validity_contract"]
    if len(cell_axes) == 1 and str(cell_axes[0].role) == "scalar":
        contract = ValidityContract.value()
    else:
        required = {
            axis_id
            for axis_id in tuple(previous.component_axis_ids)
            if any(axis.axis_id == axis_id for axis in cell_axes)
        }
        # A materialized Dataset always has one Repeat carrier dimension and
        # one Point carrier dimension before the dense Cell-data dimensions,
        # regardless of how many logical axes either mapped carrier declares.
        cell_offset = 2
        for offset, axis in enumerate(cell_axes):
            array_axis = cell_offset + offset
            first = np.take(validity, 0, axis=array_axis)
            if not np.array_equal(
                validity,
                np.broadcast_to(np.expand_dims(first, axis=array_axis), validity.shape),
            ):
                required.add(axis.axis_id)
        ordered = tuple(axis.axis_id for axis in cell_axes if axis.axis_id in required)
        contract = ValidityContract.components(*ordered) if ordered else ValidityContract.value()
    return DatasetSchema(
        repeat_domain,
        point_domain,
        cell_domain,
        ValueSchema(
            contract,
            np.dtype(draft["dtype"]),
            None if draft["unit"] is None else str(draft["unit"]),
        ),
    )


def _pack_manual_array(
    draft: Mapping[str, object],
    array: np.ndarray,
    *,
    repeat_domain: object,
    point_domain: object,
) -> np.ndarray:
    """Project the editor's logical tensor back onto the Dataset carriers."""

    repeat_axes = tuple(draft["repeat_axes"])
    point_axes = tuple(draft["point_axes"])
    cell_shape = tuple(int(axis.size) for axis in tuple(draft["cell_axes"]))
    repeat_size = (
        int(np.prod(tuple(int(axis.size) for axis in repeat_axes), dtype=np.int64))
        if repeat_axes
        else 1
    )
    point_size = (
        int(np.prod(tuple(int(axis.size) for axis in point_axes), dtype=np.int64))
        if point_axes
        else 1
    )
    logical = np.asarray(array).reshape((repeat_size, point_size, *cell_shape))
    repeat_rows = _domain_flat_rows(repeat_domain)
    point_rows = _domain_flat_rows(point_domain)
    return np.asarray(logical[repeat_rows[:, None], point_rows[None, :]])


def _logical_shape(draft: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(int(axis.size) for axis in _storage_axes(draft))


def _validate_draft_arrays(draft: Mapping[str, object]) -> None:
    """Cheap per-interaction validation; schema/codes are built only on Apply."""

    shape = _logical_shape(draft)
    values = np.asarray(draft["values"])
    validity = np.asarray(draft["validity"])
    if values.shape != shape or values.dtype != np.dtype(draft["dtype"]):
        raise ValueError("data values differ from the authored Dataset shape or dtype")
    if validity.dtype != np.dtype(np.bool_) or validity.shape != shape:
        raise ValueError("data validity must be one bool for every value")
    sigma = draft["sigma"]
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.dtype != np.dtype(np.float64) or sigma.shape != shape:
            raise ValueError("data sigma must be float64 with the Dataset shape")


def _manual_snapshot(draft: Mapping[str, object]) -> object:
    from zlc_data import owned_snapshot_from_arrays

    _validate_draft_arrays(draft)
    if draft["sigma"] is not None:
        sigma_values = np.asarray(draft["sigma"], dtype=np.float64)
        if bool(np.any(np.isfinite(sigma_values) & (sigma_values < 0.0))):
            raise ValueError("sample sigma must be non-negative")
    schema = _draft_schema(draft)
    sigma = draft["sigma"]
    values = _pack_manual_array(
        draft,
        np.asarray(draft["values"]),
        repeat_domain=schema.repeat_domain,
        point_domain=schema.point_domain,
    )
    validity = _pack_manual_array(
        draft,
        np.asarray(draft["validity"], dtype=np.bool_),
        repeat_domain=schema.repeat_domain,
        point_domain=schema.point_domain,
    )
    packed_sigma = (
        None
        if sigma is None
        else _pack_manual_array(
            draft,
            np.asarray(sigma, dtype=np.float64),
            repeat_domain=schema.repeat_domain,
            point_domain=schema.point_domain,
        )
    )
    return owned_snapshot_from_arrays(
        schema,
        values,
        0,
        validity=validity,
        sigma=packed_sigma,
    )


def _axis_id_set(draft: Mapping[str, object]) -> set[str]:
    return {str(axis.axis_id) for axis in _storage_axes(draft)}


def _new_axis_id(draft: Mapping[str, object], stem: str) -> object:
    from zlc_data import AxisId

    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(stem).strip()).strip(".-") or "axis"
    occupied = _axis_id_set(draft)
    serial = 1
    while True:
        candidate = f"manual.{base}" if serial == 1 else f"manual.{base}.{serial}"
        if candidate not in occupied:
            return AxisId(candidate)
        serial += 1


def _axis_location(draft: Mapping[str, object], axis_id: str) -> tuple[str, int, object]:
    for domain, key in (
        ("repeat", "repeat_axes"),
        ("point", "point_axes"),
        ("cell_data", "cell_axes"),
    ):
        for position, axis in enumerate(tuple(draft[key])):
            if str(axis.axis_id) == str(axis_id) and str(axis.role) != "scalar":
                return domain, position, axis
    raise KeyError(f"unknown Dataset axis {axis_id!r}")


def _domain_key(domain: str) -> str:
    selected = str(domain)
    if selected not in {"repeat", "point", "cell_data"}:
        raise ValueError(f"unknown Dataset domain {selected!r}")
    return {"repeat": "repeat_axes", "point": "point_axes", "cell_data": "cell_axes"}[selected]


def _domain_role(domain: str) -> object:
    from zlc_data import COMPONENT, REPEAT, SCAN_POINT

    return {"repeat": REPEAT, "point": SCAN_POINT, "cell_data": COMPONENT}[str(domain)]


def _axis_spec(
    axis_id: object,
    name: str,
    length: int,
    unit: str,
    domain: str,
    *,
    previous: object | None = None,
    preserve_role: bool = False,
) -> object:
    from zlc_data import AxisSpec

    coordinates = _resized_coordinates(previous, length)
    same_coordinates = previous is not None and int(previous.size) == int(length)
    return AxisSpec(
        axis_id,
        str(name).strip(),
        previous.role if preserve_role and previous is not None else _domain_role(domain),
        int(length),
        coordinates,
        str(unit).strip() or None,
        None if previous is None else previous.coordinate_frame,
        0,
        previous.coordinate_labels if same_coordinates else None,
    )


def _resize_storage_axis(draft: dict[str, object], dimension: int, size: int) -> None:
    old_size = np.asarray(draft["values"]).shape[int(dimension)]
    new_size = int(size)
    if old_size == new_size:
        return
    target_shape = list(np.asarray(draft["values"]).shape)
    target_shape[int(dimension)] = new_size
    common = min(old_size, new_size)
    source_index = [slice(None)] * len(target_shape)
    source_index[int(dimension)] = slice(0, common)
    target_index = tuple(source_index)
    values = np.zeros(target_shape, dtype=np.dtype(draft["dtype"]))
    validity = np.zeros(target_shape, dtype=np.bool_)
    values[target_index] = np.asarray(draft["values"])[target_index]
    validity[target_index] = np.asarray(draft["validity"])[target_index]
    sigma = None
    if draft["sigma"] is not None:
        sigma = np.full(target_shape, np.nan, dtype=np.float64)
        sigma[target_index] = np.asarray(draft["sigma"])[target_index]
    draft["values"], draft["validity"], draft["sigma"] = values, validity, sigma


def _insert_storage_axis(draft: dict[str, object], dimension: int, size: int) -> None:
    shape = list(np.asarray(draft["values"]).shape)
    shape.insert(int(dimension), int(size))
    values = np.zeros(shape, dtype=np.dtype(draft["dtype"]))
    validity = np.zeros(shape, dtype=np.bool_)
    target = [slice(None)] * len(shape)
    target[int(dimension)] = 0
    values[tuple(target)] = np.asarray(draft["values"])
    validity[tuple(target)] = np.asarray(draft["validity"])
    sigma = None
    if draft["sigma"] is not None:
        sigma = np.full(shape, np.nan, dtype=np.float64)
        sigma[tuple(target)] = np.asarray(draft["sigma"])
    draft["values"], draft["validity"], draft["sigma"] = values, validity, sigma


def _take_storage_axis(
    draft: dict[str, object], dimension: int, index: int = 0
) -> None:
    draft["values"] = np.take(
        np.asarray(draft["values"]), int(index), axis=int(dimension)
    )
    draft["validity"] = np.take(
        np.asarray(draft["validity"]), int(index), axis=int(dimension)
    )
    if draft["sigma"] is not None:
        draft["sigma"] = np.take(
            np.asarray(draft["sigma"]), int(index), axis=int(dimension)
        )


def _add_axis(
    draft: dict[str, object],
    *,
    name: str,
    length: int,
    unit: str,
    domain: str,
) -> str:
    key = _domain_key(domain)
    axis = _axis_spec(_new_axis_id(draft, name), name, int(length), unit, domain)
    axes = list(draft[key])
    if key == "cell_axes" and len(axes) == 1 and str(axes[0].role) == "scalar":
        axes[0] = axis
        draft[key] = axes
        _resize_storage_axis(draft, len(tuple(draft["repeat_axes"])) + len(tuple(draft["point_axes"])), int(length))
    else:
        insertion = {
            "repeat_axes": len(tuple(draft["repeat_axes"])),
            "point_axes": len(tuple(draft["repeat_axes"])) + len(tuple(draft["point_axes"])),
            "cell_axes": len(_storage_axes(draft)),
        }[key]
        _insert_storage_axis(draft, insertion, int(length))
        axes.append(axis)
        draft[key] = axes
    draft["selected_axis"] = str(axis.axis_id)
    _normalize_table_axes(draft)
    return str(axis.axis_id)


def _edit_axis(
    draft: dict[str, object],
    axis_id: str,
    *,
    name: str,
    length: int,
    unit: str,
    domain: str,
) -> bool:
    from zlc_data.axis import SCALAR_AXIS

    old_domain, position, previous = _axis_location(draft, axis_id)
    target_domain = str(domain)
    _domain_key(target_domain)
    replacement = _axis_spec(
        previous.axis_id,
        name,
        int(length),
        unit,
        target_domain,
        previous=previous,
        preserve_role=old_domain == target_domain,
    )
    old_storage = list(_storage_axes(draft))
    old_dimension = next(
        index for index, axis in enumerate(old_storage) if str(axis.axis_id) == axis_id
    )
    _resize_storage_axis(draft, old_dimension, int(length))
    if old_domain == target_domain:
        key = _domain_key(old_domain)
        axes = list(draft[key])
        axes[position] = replacement
        draft[key] = axes
        _normalize_table_axes(draft)
        return replacement != previous
    old_key = _domain_key(old_domain)
    target_key = _domain_key(target_domain)
    old_axes = list(draft[old_key])
    old_axes.pop(position)
    draft[old_key] = old_axes
    storage_ids = [str(axis.axis_id) for axis in old_storage]
    if old_key == "cell_axes" and not old_axes:
        draft[old_key] = [SCALAR_AXIS]
    target_axes = list(draft[target_key])
    if target_key == "cell_axes" and len(target_axes) == 1 and str(target_axes[0].role) == "scalar":
        scalar_dimension = storage_ids.index(str(target_axes[0].axis_id))
        _take_storage_axis(draft, scalar_dimension)
        storage_ids.pop(scalar_dimension)
        target_axes = []
    target_axes.append(replacement)
    draft[target_key] = target_axes

    desired = [str(axis.axis_id) for axis in _storage_axes(draft) if str(axis.role) != "scalar"]
    order = [storage_ids.index(axis_id_value) for axis_id_value in desired]
    for key in ("values", "validity", "sigma"):
        if draft[key] is not None:
            draft[key] = np.transpose(np.asarray(draft[key]), order)
    if any(str(axis.role) == "scalar" for axis in tuple(draft["cell_axes"])):
        _insert_storage_axis(draft, len(desired), 1)
    _normalize_table_axes(draft)
    return True


def _delete_axis(draft: dict[str, object], axis_id: str) -> None:
    from zlc_data.axis import SCALAR_AXIS

    domain, position, axis = _axis_location(draft, axis_id)
    storage = list(_storage_axes(draft))
    dimension = next(index for index, item in enumerate(storage) if str(item.axis_id) == axis_id)
    key = _domain_key(domain)
    axes = list(draft[key])
    axes.pop(position)
    draft[key] = axes
    kept_index = min(
        max(0, int(dict(draft.get("scopes", {})).get(axis_id, 0))),
        int(axis.size) - 1,
    )
    if domain == "cell_data" and not axes:
        draft[key] = [SCALAR_AXIS]
        _take_storage_axis(draft, dimension, kept_index)
        _insert_storage_axis(draft, len(_storage_axes(draft)) - 1, 1)
    else:
        _take_storage_axis(draft, dimension, kept_index)
    visible = _visible_axes(draft)
    draft["selected_axis"] = "" if not visible else str(visible[0].axis_id)
    draft["message"] = (
        f"Deleted {axis.name}; kept coordinate {axis.coordinate_at(kept_index)!r}"
    )
    _normalize_table_axes(draft)


def _set_axis_values(
    draft: dict[str, object], axis_id: str, cells: object
) -> bool:
    domain, position, axis = _axis_location(draft, axis_id)
    values = list(_axis_coordinates(axis))
    replacements = []
    for row, column, text in tuple(cells):
        index = int(column)
        if int(row) != 0 or not 0 <= index < int(axis.size):
            raise IndexError("axis value edit lies outside the selected axis")
        replacements.append((index, _parse_axis_value(text)))
    if not replacements:
        return False
    changed = False
    for index, value in replacements:
        changed = changed or values[index] != value
        values[index] = value
    if len(set(values)) != len(values):
        raise ValueError("axis values must be unique")
    if not changed:
        return False
    from zlc_data import AxisSpec

    replacement = AxisSpec(
        axis.axis_id,
        axis.name,
        axis.role,
        axis.size,
        tuple(values),
        axis.unit,
        axis.coordinate_frame,
        0,
        axis.coordinate_labels,
    )
    key = _domain_key(domain)
    axes = list(draft[key])
    axes[position] = replacement
    draft[key] = axes
    return True


def _set_table_axis(draft: dict[str, object], axis_id: str, mode: str) -> None:
    _axis_location(draft, axis_id)
    selected = str(mode)
    if selected not in {"rows", "columns", "scope"}:
        raise ValueError(f"unknown table axis mode {selected!r}")
    row = str(draft.get("row_axis") or "")
    column = str(draft.get("column_axis") or "")
    if selected == "rows":
        if axis_id == column:
            column = row if row and row != axis_id else ""
        row = axis_id
    elif selected == "columns":
        if axis_id == row:
            row = column if column and column != axis_id else ""
        column = axis_id
    else:
        if row == axis_id:
            row = ""
        if column == axis_id:
            column = ""
    draft["row_axis"], draft["column_axis"] = row, column
    _normalize_table_axes(draft)


def _table_index(draft: Mapping[str, object], row: int, column: int) -> tuple[int, ...]:
    row_axis = str(draft.get("row_axis") or "")
    column_axis = str(draft.get("column_axis") or "")
    scopes = dict(draft.get("scopes", {}))
    result = []
    for axis in _storage_axes(draft):
        axis_id = str(axis.axis_id)
        if str(axis.role) == "scalar":
            value = 0
        elif axis_id == row_axis:
            value = int(row)
        elif axis_id == column_axis:
            value = int(column)
        else:
            value = int(scopes.get(axis_id, 0))
        if not 0 <= value < int(axis.size):
            raise IndexError("data edit lies outside the selected Dataset slice")
        result.append(value)
    return tuple(result)


def _table_projection(draft: dict[str, object]) -> dict[str, object]:
    from zlc_plot.semantics import SemanticCycleChoices, axis_structure

    _normalize_table_axes(draft)
    row_axis = str(draft.get("row_axis") or "")
    column_axis = str(draft.get("column_axis") or "")
    scopes = dict(draft.get("scopes", {}))
    storage = _storage_axes(draft)
    indexer: list[object] = []
    remaining: list[str] = []
    for axis in storage:
        axis_id = str(axis.axis_id)
        if axis_id in {row_axis, column_axis}:
            indexer.append(slice(None))
            remaining.append(axis_id)
        else:
            indexer.append(0 if str(axis.role) == "scalar" else int(scopes.get(axis_id, 0)))
    component = str(draft["component"])
    validity_source = np.asarray(draft["validity"], dtype=np.bool_)
    if component == "validity":
        values = validity_source[tuple(indexer)]
        table_validity = None
    elif component == "sigma" and draft["sigma"] is not None:
        values = np.asarray(draft["sigma"], dtype=np.float64)[tuple(indexer)]
        table_validity = validity_source[tuple(indexer)]
    else:
        component = "values"
        values = np.asarray(draft["values"])[tuple(indexer)]
        table_validity = validity_source[tuple(indexer)]
    values = np.asarray(values)
    if remaining == [column_axis, row_axis]:
        values = values.T
        if table_validity is not None:
            table_validity = np.asarray(table_validity).T
    if values.ndim == 0:
        values = values.reshape((1, 1))
    elif values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
        if table_validity is not None:
            table_validity = np.asarray(table_validity).reshape(values.shape)
    elif values.ndim != 2:
        raise RuntimeError("manual data table did not reduce to two dimensions")
    by_id = {str(axis.axis_id): axis for axis in _visible_axes(draft)}
    row_headers = _axis_coordinates(by_id[row_axis]) if row_axis else ("Value",)
    column_headers = _axis_coordinates(by_id[column_axis]) if column_axis else ("Value",)
    axis_rows = []
    for axis in _visible_axes(draft):
        axis_id = str(axis.axis_id)
        mode = "rows" if axis_id == row_axis else "columns" if axis_id == column_axis else "scope"
        index = min(max(0, int(scopes.get(axis_id, 0))), int(axis.size) - 1)
        axis_rows.append(
            {
                "axis_id": axis_id,
                "name": axis.name,
                "size": int(axis.size),
                "unit": axis.unit,
                "mode": mode,
                "index": index,
                "scope_choices": SemanticCycleChoices(
                    _axis_coordinates(axis),
                    unit=axis.unit or "",
                    locate=axis.coordinate_position,
                ),
            }
        )
    choices = [("values", "Values"), ("validity", "Validity")]
    if draft["sigma"] is not None:
        choices.append(("sigma", "Sigma"))
    structure = axis_structure(
        draft["repeat_axes"],
        draft["point_axes"],
        draft["cell_axes"],
    )
    return {
        "component": component,
        "component_choices": tuple(choices),
        "sigma_enabled": draft["sigma"] is not None,
        "shape": tuple(values.shape),
        "values": values,
        "validity": table_validity,
        "finite_values": component == "sigma",
        "blank_help": {
            "values": "Blank deletes this sample and marks it invalid",
            "validity": "Blank means False",
            "sigma": "Blank means no stated sigma",
        }[component],
        "row_headers": row_headers,
        "column_headers": column_headers,
        "editable": True,
        "structure": structure,
        "axes": tuple(axis_rows),
    }


def _draft_axes(draft: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    labels = {"repeat": "Repeat", "point": "Point", "cell_data": "Cell data"}
    result = []
    for domain, key in (
        ("repeat", "repeat_axes"),
        ("point", "point_axes"),
        ("cell_data", "cell_axes"),
    ):
        for axis in tuple(draft[key]):
            if str(axis.role) == "scalar":
                continue
            result.append(
                {
                    "id": str(axis.axis_id),
                    "name": axis.name,
                    "size": int(axis.size),
                    "unit": "" if axis.unit is None else str(axis.unit),
                    "domain": domain,
                    "domain_label": labels[domain],
                }
            )
    return tuple(result)


def _data_projection(draft: dict[str, object]) -> dict[str, object]:
    axes = _draft_axes(draft)
    selected = str(draft.get("selected_axis") or "")
    if not axes:
        selected = ""
        draft["selected_axis"] = ""
    elif not any(str(axis["id"]) == selected for axis in axes):
        selected = str(axes[0]["id"])
        draft["selected_axis"] = selected
    can_apply = True
    validation_message = ""
    try:
        _validate_draft_arrays(draft)
    except (TypeError, ValueError) as error:
        can_apply = False
        validation_message = str(error)
    source_path = draft["source_path"]
    suggested = f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', str(draft['name'])).strip('.-') or 'figure'}.npz"
    coordinates = ()
    if selected:
        _domain, _position, selected_axis = _axis_location(draft, selected)
        coordinates = _axis_coordinates(selected_axis)
    return {
        "dataset": {
            "name": str(draft["name"]),
            "dtype": np.dtype(draft["dtype"]).str,
            "unit": "" if draft["unit"] is None else str(draft["unit"]),
            "note": str(draft["note"]),
            "source": str(draft["source_text"]),
            "dtype_choices": tuple((value, np.dtype(value).name) for value in _MANUAL_DTYPES),
        },
        "domain_choices": (
            ("repeat", "Repeat"),
            ("point", "Point"),
            ("cell_data", "Cell data"),
        ),
        "axes": axes,
        "selected_axis": selected,
        "axis_values": {
            "shape": (1, len(coordinates)),
            "values": (coordinates,),
            "row_headers": ("Value",),
            "column_headers": range(len(coordinates)),
            "editable": True,
            "blank_hint": "Axis values cannot be blank",
        },
        "table": _table_projection(draft),
        "dirty": bool(draft["modified"] or draft["unsaved"]),
        "can_apply": can_apply,
        "can_save": bool(draft["save_ready"] and not draft["modified"]),
        "save_suggested": str((Path(source_path).parent / suggested) if source_path is not None else suggested),
        "message": str(draft.get("message") or validation_message),
    }


def _set_dtype(draft: dict[str, object], value: object) -> bool:
    selected = np.dtype(value)
    if selected.str not in _MANUAL_DTYPES:
        raise ValueError(f"unsupported manual data type {selected.name!r}")
    current = np.asarray(draft["values"])
    if current.dtype == selected:
        return False
    valid = np.asarray(draft["validity"], dtype=np.bool_)
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


def _set_table_cells(draft: dict[str, object], component: str, cells: object) -> bool:
    selected = str(component)
    if selected not in {"values", "validity", "sigma"}:
        raise ValueError(f"unknown data table {selected!r}")
    if selected == "sigma" and draft["sigma"] is None:
        raise ValueError("enable sigma before editing it")
    parsed = []
    for row, column, text in tuple(cells):
        index = _table_index(draft, int(row), int(column))
        raw = str(text).strip()
        if selected == "values":
            value = None if not raw else _parse_value(raw, np.dtype(draft["dtype"]))
        elif selected == "validity":
            value = False if not raw else bool(_parse_value(raw, np.dtype("?")))
        else:
            value = np.nan if not raw else float(raw)
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
                    or draft["sigma"] is not None
                    and not np.isnan(np.asarray(draft["sigma"])[index])
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
            changed = changed or not (
                np.isnan(previous) and np.isnan(value) or previous == value
            )
            np.asarray(draft["sigma"])[index] = value
    return bool(changed)


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
            DomainSpec,
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
            image.block.schema.repeat_domain,
            image.block.schema.point_domain,
            DomainSpec((count,), (site_axis,)),
            ValueSchema(
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
                else status.block.schema.cell_domain.axes[0]
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
                    snapshot.block.schema.repeat_domain.size
                    * snapshot.block.schema.point_domain.size,
                    snapshot.block.schema.repeat_domain.size
                    * snapshot.block.schema.point_domain.size,
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
                    status.block.schema.repeat_domain.size
                    * status.block.schema.point_domain.size,
                    status.block.schema.repeat_domain.size
                    * status.block.schema.point_domain.size,
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
        old.repeat_domain == new.repeat_domain
        and old.point_domain == new.point_domain
        and old.cell_domain == new.cell_domain
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
        build_figure_host: Callable[..., object],
        save_figure_artifact: Callable[..., object],
        save_front: Callable[..., object],
        confirm_discard: Callable[[str], bool] | None = None,
    ) -> None:
        self.view = view
        # Asked before edits are thrown away, and answered by a person.
        # Without one, unsaved work is never discarded on this presenter's
        # own initiative -- a headless host keeps the old refusal rather
        # than silently losing a working copy.
        if confirm_discard is not None and not callable(confirm_discard):
            raise TypeError("confirm_discard must be callable or None")
        self._confirm_discard = confirm_discard
        #: The answer belongs to the close GESTURE, not to a pass of it.
        self._discard_agreed = False
        self._run_off_thread = run_off_thread
        self._close_worker = close_worker
        self._request_close = request_close
        self._panel_presenter = panel_presenter
        self._signal_plane = signal_plane
        if not callable(build_figure_host):
            raise TypeError("build_figure_host must be callable")
        if not callable(save_figure_artifact):
            raise TypeError("save_figure_artifact must be callable")
        if not callable(save_front):
            raise TypeError("save_front must be callable")
        self._build_figure_host = build_figure_host
        self._save_figure_artifact = save_figure_artifact
        self._save_front = save_front
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
        device_pixel_ratio = float(self.view.device_pixel_ratio())

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
                        device_pixel_ratio=device_pixel_ratio,
                        build_host=self._build_figure_host,
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
        from .panel_save import _IMPORTED_LINEAGE_KEY

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
                    run_record=(
                        {_IMPORTED_LINEAGE_KEY: deepcopy(dict(source_lineage))}
                        if isinstance(source_lineage.get("root"), str)
                        else None
                    ),
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
        device_pixel_ratio = float(self.view.device_pixel_ratio())

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
                device_pixel_ratio=device_pixel_ratio,
                build_host=self._build_figure_host,
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

    def _agrees_to_lose(self, what: str) -> bool:
        """Ask once whether unsaved edits may go, and take the answer.

        Refusing to close until the operator finds the Discard button is
        not safety, it is a door that only opens from the inside: the
        work is theirs, so they are asked, and a yes is honoured.  With
        no one to ask -- a headless host, a test -- nothing is discarded.
        """

        ask = self._confirm_discard
        if ask is None:
            return False
        return bool(
            ask(
                "%s has unsaved edits.  Close it and lose them?" % what
            )
        )

    def close_data_editor(self, editor_id: str) -> bool:
        draft = self._data_drafts.get(str(editor_id))
        if draft is None:
            return False
        if bool(draft["modified"] or draft["unsaved"]) and not self._agrees_to_lose(
            str(draft["name"])
        ):
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
                _add_axis(
                    draft,
                    name=str(command["name"]),
                    length=int(command["length"]),
                    unit=str(command.get("unit") or ""),
                    domain=str(command["domain"]),
                )
            elif operation == "edit_axis":
                marks_dirty = _edit_axis(
                    draft,
                    str(command["axis_id"]),
                    name=str(command["name"]),
                    length=int(command["length"]),
                    unit=str(command.get("unit") or ""),
                    domain=str(command["domain"]),
                )
            elif operation == "delete_axis":
                _delete_axis(draft, str(command["axis_id"]))
            elif operation == "set_axis_values":
                marks_dirty = _set_axis_values(
                    draft, str(command["axis_id"]), command["cells"]
                )
            elif operation == "set_table_axis":
                _set_table_axis(
                    draft, str(command["axis_id"]), str(command["mode"])
                )
                marks_dirty = False
            elif operation == "set_scope":
                axis_id = str(command["axis_id"])
                _domain, _position, axis = _axis_location(draft, axis_id)
                index = int(command["index"])
                if not 0 <= index < int(axis.size):
                    raise IndexError("scope position lies outside its Dataset axis")
                draft["scopes"][axis_id] = index
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
        editor_host = (
            binding.editor_host
            if binding.editor_host is not None
            and binding.editor_configuration is None
            and getattr(binding.editor_host, "startup_failure", None) is None
            and not bool(getattr(binding.editor_host, "closing", False))
            else None
        )

        def write() -> object:
            return self._await(
                save_panel_figure(
                    image_path,
                    state=state,
                    frozen=manual_frozen,
                    writer=self._save_figure_artifact,
                    source=source,
                    host=editor_host,
                )
            )

        def saved(written: object) -> None:
            image, archive = written.image, written.archive
            current = draft["publication"] is saved_publication
            if current:
                draft["unsaved"] = False
            draft["message"] = (
                f"Saved {archive.name}"
                if current
                else f"Saved older preview to {archive.name}; current data is not saved"
            )
            self.view.set_status(
                f"saved {archive.name} and {image.name}"
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
        front = binding.host.front
        if front is None:
            self.view.set_status("this figure has no complete image yet", error=True)
            return
        path = self.path
        dataset = binding.state.signal.rsplit("/", 1)[-1]

        def write_image() -> object:
            def write(temporary: Path) -> None:
                self._await(self._save_front(temporary, front))

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
        unsaved = tuple(
            str(draft["name"])
            for draft in self._data_drafts.values()
            if bool(draft["modified"] or draft["unsaved"])
        )
        if unsaved and not self._discard_agreed:
            # Closing takes several passes -- panels, then the IO worker --
            # and each returns False and asks to be called again.  Asked
            # inside that loop, the operator answered once per PASS: a
            # first close wanted two and asked twice, and after declining,
            # the pass already spent left the next close wanting one.  The
            # decision is made once and held for as long as this gesture
            # is in flight; declining ends the gesture and the next close
            # is a fresh question.
            if not self._agrees_to_lose(", ".join(unsaved)):
                self._close_requested = False
                self.view.set_status(
                    "Save or discard the open data working copy before closing",
                    error=True,
                )
                return False
            self._discard_agreed = True
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
