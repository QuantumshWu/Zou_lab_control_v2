"""Histogram kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import PlotKind
from ..specs import HistogramPlot
from .base import KindHandler
from . import defaults


def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str, **pooled: Any) -> None:
    renderer._update_histogram(axes, payload, state, key, **pooled)


def build_payload(projection: Any, view: Any, state: Any) -> None:
    window = int(state["window"])
    spec = projection._semantic_spec()
    collapsed = tuple(getattr(spec, "reduced", ()))
    aggregation = getattr(spec, "reduction", None)
    if not collapsed and spec.group is None and view.has_primary_index:
        # A window of an integer history is binned from its frequency table,
        # which the view carries from one revision to the next and moves by
        # the shots that entered and left, instead of recounting the window.
        frequency = view.window_frequency(window)
        if frequency is not None:
            payload = view.histogram_from_frequency(
                bins=projection._histogram_bins(view, state, frequency=frequency),
                frequency=frequency,
            )
            if payload is not None:
                projection._payload = payload
                return
    view._frequency_carry = None
    plan = view._histogram_plan(
        () if spec.group is None else (spec.group,), collapsed, aggregation, window,
    )
    projection._payload = view._histogram_from_plan(
        projection._histogram_bins(view, state, binned_values=plan.values, binned_valid=plan.valid),
        plan,
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """A histogram's x names the plotted value; its y is always a count."""

    return (
        ("title", ("title",)),
        ("x", ("value",)),
        ("y", ("count",)),
    )


def default_spec(schema: Any) -> HistogramPlot | None:
    """A histogram pools every value: the table's one pooling entry."""

    return defaults.default_spec(schema, PlotKind.HISTOGRAM)


HANDLER = KindHandler(
    PlotKind.HISTOGRAM,
    "Histogram",
    HistogramPlot,
    "histogram",
    render,
    build_payload,
    ("kind", "group", "reduction"),
    admits,
    default_spec,
    label_roles,
)
