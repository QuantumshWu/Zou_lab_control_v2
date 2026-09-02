"""Curve kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import AxisRef, PlotKind
from ..specs import CurvePlot, Reduction
from .base import KindHandler
from . import defaults


def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str, **pooled: Any) -> None:
    renderer._update_curve(axes, payload, state, key, **pooled)


def build_payload(projection: Any, view: Any, state: Any) -> None:
    spec = projection._spec
    group = () if spec.group is None else (spec.group,)
    projection._payload = view.curve(
        spec.x,
        group_by=group,
        aggregation=spec.reduction,
        # The band is the operator's display switch; the standard error only
        # exists for a MEAN, so on any other reduction the switch is inert.
        uncertainty=(
            bool(state["uncertainty"]) and spec.reduction is Reduction.MEAN
        ),
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """A curve's x names its x axis; its y always names the plotted value."""

    return (
        ("title", ("title",)),
        ("x", ("axis", spec.x)),
        ("y", ("value",)),
    )


def default_spec(schema: Any) -> CurvePlot | None:
    """The curve this dataset shows unasked: one reading of the table.

    A curve walks position -- the innermost scan loop, else the events of
    a cycle, else its own data -- and groups the one content axis the
    palette can tell apart.  See :mod:`zlc_plot._kinds.defaults`.
    """

    return defaults.default_spec(schema, PlotKind.CURVE)


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
)
