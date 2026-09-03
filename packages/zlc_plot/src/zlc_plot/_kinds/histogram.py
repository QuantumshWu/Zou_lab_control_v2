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
    if not collapsed and view.has_primary_index:
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
    if window <= 1 and not view.has_primary_index:
        # One shot: the distribution of what was just measured.  No history is
        # consulted, and none is kept for it.
        values, valid = None, None
    else:
        # Runtime already owns and bounds cross-publication history.  If some
        # other consumer keeps the signal indexed, window=1 still selects the
        # ordinary axis coordinate 0 (latest) instead of pooling every retained
        # cell.  Plot never copies or retains per-shot raw pools.
        values, valid = view.history_values(window)

    # REDUCED ONCE.  The pool is what the domain must cover and what the bins
    # then count, so it is derived here and handed to both -- rather than
    # deriving it inside the binning, where the domain could not see it and
    # took its limits from the raw samples instead.
    pool, pool_valid = view.histogram_pool(
        values=values,
        valid=valid,
        reduce_axes=collapsed,
        aggregation=aggregation,
    )
    projection._payload = view.histogram(
        bins=projection._histogram_bins(
            view,
            state,
            binned_values=pool,
            binned_valid=pool_valid,
        ),
        values=pool,
        valid=pool_valid,
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
    ("kind", "reduction"),
    admits,
    default_spec,
    label_roles,
)
