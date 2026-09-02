"""Image kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import AxisRef, PlotKind
from ..specs import ImagePlot
from .base import KindHandler
from . import defaults


def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str, **pooled: Any) -> None:
    renderer._update_image(axes, payload, state, key, **pooled)


def build_payload(projection: Any, view: Any, _state: Any) -> None:
    spec = projection._spec
    projection._payload = view.image(
        spec.x,
        spec.y,
        aggregation=spec.reduction,
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """Image x/y name their axes; the value slot names the color scale."""

    return (
        ("title", ("title",)),
        ("x", ("axis", spec.x)),
        ("y", ("axis", spec.y)),
        ("value", ("value",)),
    )


def default_spec(schema: Any) -> ImagePlot | None:
    """The image this dataset shows unasked, or None when it has none.

    A declared picture, two content axes, a two-dimensional scan, or one
    position axis against one content axis -- in that order.  See
    :mod:`zlc_plot._kinds.defaults`.
    """

    return defaults.default_spec(schema, PlotKind.IMAGE)


HANDLER = KindHandler(
    PlotKind.IMAGE,
    "Image",
    ImagePlot,
    "image",
    render,
    build_payload,
    ("kind", "x", "y", "reduction"),
    admits,
    default_spec,
    label_roles,
)
