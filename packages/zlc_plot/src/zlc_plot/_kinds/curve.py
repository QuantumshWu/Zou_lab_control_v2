"""Curve kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import AxisRef, PlotKind
from zlc_data import DatasetSchema
from ..specs import CurvePlot, Reduction
from .base import KindHandler


def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str, **pooled: Any) -> None:
    renderer._update_curve(axes, payload, state, key, **pooled)


def build_payload(projection: Any, view: Any, _state: Any) -> None:
    spec = projection._spec
    group = () if spec.group is None else (spec.group,)
    projection._payload = view.curve(
        spec.x,
        group_by=group,
        aggregation=spec.reduction,
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def validate(view: Any, spec: Any) -> None:
    view.validate_curve(spec.x, group_by=() if spec.group is None else (spec.group,))


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """A curve's x names its x axis; its y always names the plotted value."""

    return (
        ("title", ("title",)),
        ("x", ("axis", spec.x)),
        ("y", ("value",)),
    )


def default_spec(schema: Any) -> CurvePlot | None:
    """Infer one unambiguous curve projection from a dataset schema.

    The fastest-varying scan coordinate is the per-trace x: in a Cartesian
    topology the last dimension is the innermost loop, so it is the axis one
    acquisition sweep actually walks.  If there is no scan topology, the
    first point-table coordinate is used; the point-row axis is the final
    fallback.  A single dense data axis is grouped so its structure stays
    visible; several dense axes have no privileged one, so they all collapse
    under the declared reduction — unambiguous, unlike picking a favourite.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    if schema.grid_topology is not None and schema.grid_topology.dimension_ids:
        x = AxisRef.point_dimension(str(schema.grid_topology.dimension_ids[-1]))
    elif schema.point_table.columns:
        # By coordinate ID, which is what AxisRef.point means and what the
        # resolver looks up.  A column's NAME is what an operator reads, and
        # the two are equal often enough that passing the name worked until a
        # producer named a column something other than its coordinate -- then
        # the spec named an axis the point table does not have, and the
        # session raised on construction, leaving the host permanently closing.
        x = AxisRef.point(str(schema.point_table.columns[0].coordinate_id))
    else:  # pragma: no cover - DatasetSchema requires a point column
        x = AxisRef.point_rows()
    data_axes = tuple(axis for axis in schema.cell_schema.data_axes if axis.size > 1)
    group = AxisRef.data(str(data_axes[0].axis_id)) if len(data_axes) == 1 else None
    return CurvePlot(x, group=group, reduction=Reduction.MEAN)


HANDLER = KindHandler(
    PlotKind.CURVE,
    "Curve",
    CurvePlot,
    "series",
    render,
    build_payload,
    ("kind", "x", "group", "reduction"),
    admits,
    default_spec,
    label_roles,
    validate,
)
