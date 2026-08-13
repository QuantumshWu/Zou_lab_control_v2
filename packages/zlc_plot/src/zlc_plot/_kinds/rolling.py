"""Rolling kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import PlotKind
from zlc_data import DatasetSchema
from ..specs import Reduction, RollingPlot
from .base import KindHandler

def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str) -> None:
    renderer._update_rolling(axes, payload, state, key)


def build_payload(projection: Any, view: Any, state: Any) -> None:
    # Accumulation is a projection-layer concern shared with every other kind
    # that looks back.  Importing it lazily keeps the closed kind registry
    # independent of FitProjection's implementation.
    from .._fit_projection import accumulate_history

    spec = projection._spec
    # The retained history is capped by memory policy only; the display
    # window is applied inside ``_rolling_payload``.  Persisting the display
    # slice here once made shrinking the window destructive and enlarging it
    # inert, because the truncated cache doubled as the permanent record.
    projection._rolling_history_cache = accumulate_history(
        projection,
        view,
        group=spec.group,
        aggregation=spec.reduction,
    )
    projection._payload = projection._rolling_payload(
        projection._rolling_history_cache,
        window=int(state["window"]),
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def validate(view: Any, spec: Any) -> None:
    # Rolling reduces whole revisions; only an authored group can be invalid.
    view.validate_rolling(spec.group)


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """A rolling plot's x is the shot counter; its y names the value."""

    return (
        ("title", ("title",)),
        ("x", ("repeat",)),
        ("y", ("value",)),
    )


def default_spec(schema: Any) -> RollingPlot | None:
    """Infer the ungrouped rolling reduction for any dataset schema."""

    if not isinstance(schema, DatasetSchema):
        return None
    return RollingPlot(reduction=Reduction.MEAN)


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
    validate,
)
