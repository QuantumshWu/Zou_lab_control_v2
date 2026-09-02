"""Rolling kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import PlotKind
from ..specs import Reduction, RollingPlot
from .base import KindHandler
from . import defaults

def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str) -> None:
    renderer._update_rolling(axes, payload, state, key)


def build_payload(projection: Any, view: Any, state: Any) -> None:
    spec = projection._spec
    uncertainty = (
        bool(state["uncertainty"]) and spec.reduction is Reduction.MEAN
    )
    history = view.rolling_history(
        group=spec.group,
        aggregation=spec.reduction,
        uncertainty=uncertainty,
    )
    projection._payload = projection._rolling_payload(
        history,
        window=int(state["window"]),
        trailing=(
            int(state["trailing"]) if spec.reduction is Reduction.MEAN else 1
        ),
        uncertainty=uncertainty,
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """A rolling plot's x is the shot counter; its y names the value."""

    return (
        ("title", ("title",)),
        ("x", ("repeat",)),
        ("y", ("value",)),
    )


def default_spec(schema: Any) -> RollingPlot | None:
    """The rolling view this dataset shows unasked: one reading of the table.

    Rolling walks the shot history itself; the table groups the one content
    axis the palette can tell apart and shows the latest of any event
    sequence.  See :mod:`zlc_plot._kinds.defaults`.
    """

    return defaults.default_spec(schema, PlotKind.ROLLING)


HANDLER = KindHandler(
    PlotKind.ROLLING,
    "Rolling",
    RollingPlot,
    "series",
    render,
    build_payload,
    ("kind", "group", "reduction"),
    admits,
    default_spec,
    label_roles,
)
