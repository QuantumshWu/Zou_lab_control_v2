"""Image kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..data_contract import image_axes
from ..kinds import AxisRef, PlotKind
from zlc_data import DatasetSchema
from ..specs import ImagePlot, Reduction
from .base import KindHandler


def render(renderer: Any, payload: Any, state: Any) -> None:
    renderer._update_image(payload, state, "image")


def build_payload(projection: Any, view: Any, _state: Any) -> None:
    spec = projection._spec
    projection._payload = view.image(
        spec.x,
        spec.y,
        aggregation=spec.reduction,
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def validate(view: Any, spec: Any) -> None:
    view.validate_image(spec.x, spec.y)


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """Image x/y name their axes; the value slot names the color scale."""

    return (
        ("title", ("title",)),
        ("x", ("axis", spec.x)),
        ("y", ("axis", spec.y)),
        ("value", ("value",)),
    )


def default_spec(schema: Any) -> ImagePlot | None:
    """Infer a regular image only when the schema proves two image axes."""

    if not isinstance(schema, DatasetSchema):
        return None
    if schema.grid_topology is not None and len(schema.grid_topology.dimension_ids) >= 2:
        # Dense data axes collapse under the declared reduction; the scan
        # grid itself is the unambiguous image surface.
        # Axes are declared slowest-first (C order), so the last dimension is
        # the innermost scan loop: that is the horizontal image axis, and the
        # dimension above it is the vertical one.  Slowest-first here would
        # silently transpose every scan image.
        dimensions = schema.grid_topology.dimension_ids
        return ImagePlot(
            AxisRef.point_dimension(str(dimensions[-1])),
            AxisRef.point_dimension(str(dimensions[-2])),
            reduction=Reduction.MEAN,
        )
    # By role, through the one place that decides which axes are the image.
    # This used to pick by size and position, which refused a two-window
    # bracket outright and a one-pixel-tall ROI strip from the other side --
    # for datasets whose schema names their two spatial axes.  Point rows
    # beyond the image collapse under the declared reduction, the same fate
    # every unassigned axis has: a single-axis scan of frames averages its
    # scan points here, and shows them apart as a facet grid instead.
    pair = image_axes(schema)
    if pair is None:
        return None
    x_axis, y_axis = pair
    return ImagePlot(
        AxisRef.data(str(x_axis.axis_id)),
        AxisRef.data(str(y_axis.axis_id)),
        reduction=Reduction.MEAN,
    )


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
    validate,
)
