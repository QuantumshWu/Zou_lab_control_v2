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
    if window <= 1:
        # One shot: the distribution of what was just measured.  No history is
        # consulted, and none is kept for it.
        projection._payload = view.histogram(
            bins=projection._histogram_bins(view, state),
        )
        return

    # A window pools the last n SHOTS -- the session's history of this signal,
    # not the repeats one publication happens to carry.  Accumulation belongs
    # to the projection layer, which every kind that looks back shares; the
    # values it stores are already restricted to whatever the panel is scoped
    # to, so pooling them needs no second restriction.
    import numpy as np

    from .._fit_projection import accumulate_history
    from ..specs import Reduction

    history = accumulate_history(
        projection, view, group=None, aggregation=Reduction.MEAN
    )
    projection._rolling_history_cache = history
    visible = history[-window:]
    pooled = [point.values for point in visible if point.values is not None]
    values = (
        np.concatenate(pooled) if pooled else np.asarray(view.pooled_values())
    )
    projection._payload = view.histogram_of(
        values,
        bins=projection._histogram_bins(view, state, pooled=values),
        # Which shots really went in.  The memory budget releases the oldest
        # pooled values first, so a window can reach further back than the
        # values it still has -- and a distribution that quietly pooled fewer
        # shots than it was asked for would look exactly like one that did.
        source_revisions=tuple(
            point.sample.revision
            for point in visible
            if point.values is not None
        ),
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
    ("kind",),
    admits,
    default_spec,
    label_roles,
    validate,
)
