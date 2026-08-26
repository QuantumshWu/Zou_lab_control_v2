"""Histogram kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import PlotKind
from zlc_data import DatasetSchema
from ..specs import HistogramPlot
from .base import KindHandler


def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str, **pooled: Any) -> None:
    renderer._update_histogram(axes, payload, state, key, **pooled)


def build_payload(projection: Any, view: Any, state: Any) -> None:
    window = int(state["window"])
    spec = projection._semantic_spec()
    collapsed = tuple(getattr(spec, "reduced", ()))
    aggregation = getattr(spec, "reduction", None)
    if window <= 1:
        # One shot: the distribution of what was just measured.  No history is
        # consulted, and none is kept for it.
        projection._payload = view.histogram(
            bins=projection._histogram_bins(view, state),
            reduce_axes=collapsed,
            aggregation=aggregation,
        )
        return

    # Runtime already owns and bounds cross-publication history.  DataView
    # selects the last N history cells and bins their values in one pass;
    # Plot never copies or retains per-shot raw pools.
    values, valid = view.history_values(window)
    projection._payload = view.histogram(
        bins=projection._histogram_bins(
            view,
            state,
            history_values=values,
            history_valid=valid,
        ),
        values=values,
        valid=valid,
        reduce_axes=collapsed,
        aggregation=aggregation,
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def validate(view: Any, spec: Any) -> None:
    # A histogram pools every acquired value; any dataset admits it.
    return None


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """A histogram's x names the plotted value; its y is always a count."""

    return (
        ("title", ("title",)),
        ("x", ("value",)),
        ("y", ("count",)),
    )


def default_spec(schema: Any) -> HistogramPlot | None:
    """A histogram needs no inference: it is the distribution of all values."""

    if not isinstance(schema, DatasetSchema):
        return None
    return HistogramPlot()


HANDLER = KindHandler(
    PlotKind.HISTOGRAM,
    "Histogram",
    HistogramPlot,
    "histogram",
    render,
    build_payload,
    ("kind", "reduction"),
    admits,
    default_spec,
    label_roles,
    validate,
)
