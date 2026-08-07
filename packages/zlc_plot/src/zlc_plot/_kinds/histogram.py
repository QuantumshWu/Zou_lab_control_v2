"""Histogram kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import PlotKind
from zlc_data import DatasetSchema
from ..specs import HistogramPlot
from .base import KindHandler


def render(renderer: Any, payload: Any, state: Any) -> None:
    renderer._update_histogram(renderer.primary_axes, payload, state, "histogram")


def build_payload(projection: Any, view: Any, state: Any) -> None:
    projection._payload = view.histogram(
        bins=projection._histogram_bins(view, state),
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
