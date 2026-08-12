"""Closed plot-kind registry shared by projection and renderer layers."""

from __future__ import annotations

from typing import Any

from zlc_data import DatasetSchema

from ..data_contract import live_grid_dimensions
from .base import KindHandler
from .curve import HANDLER as CURVE_HANDLER
from .facet_grid import HANDLER as FACET_GRID_HANDLER
from .histogram import HANDLER as HISTOGRAM_HANDLER
from .image import HANDLER as IMAGE_HANDLER
from .pulse_timeline import HANDLER as PULSE_TIMELINE_HANDLER
from .rolling import HANDLER as ROLLING_HANDLER

HANDLERS: tuple[KindHandler, ...] = (
    CURVE_HANDLER,
    IMAGE_HANDLER,
    HISTOGRAM_HANDLER,
    ROLLING_HANDLER,
    FACET_GRID_HANDLER,
    PULSE_TIMELINE_HANDLER,
)
_BY_SPEC = {handler.spec_type: handler for handler in HANDLERS}


def handler_for(spec: Any) -> KindHandler:
    """Return the one semantic handler for an authored PlotSpec."""

    try:
        return _BY_SPEC[type(spec)]
    except KeyError as error:
        raise TypeError(f"unsupported plot specification {type(spec).__name__}") from error


def default_spec(schema: Any, kind: Any) -> Any:
    """Return the kind-owned default spec, or ``None`` when it is ambiguous.

    The registry deliberately owns this dispatch.  Embedders and notebook
    adapters only ask for a candidate and then submit it through the normal
    ``replace_spec`` command; they do not recreate kind-specific inference.
    """

    try:
        handler = next(item for item in HANDLERS if item.kind is kind)
    except StopIteration as error:
        raise ValueError(f"unregistered plot kind: {kind!r}") from error
    candidate = handler.default_spec(schema)
    if candidate is not None:
        if handler_for(candidate).kind is not kind:
            raise TypeError(
                f"default spec for {kind!r} has kind {handler_for(candidate).kind!r}"
            )
    return candidate


#: The order an unasked-for plot is inferred in, most specific first.
#:
#: Only kinds a dataset can PROVE it wants are here.  A curve fits nearly any
#: schema, so it goes last and acts as the fallback; an image must be probed
#: before it or every frame would come back as a curve.  Histogram and rolling
#: are absent on purpose: each is a decision about how to look at data that
#: could equally be shown another way, and inferring one would be guessing at
#: intent rather than reading structure.
#:
#: A facet grid is not in this list either, and for a different reason: it is
#: not one more candidate to try, it is the answer to a question none of these
#: can answer -- "is there structure here that a single flat plot would have
#: to average away".  :func:`_grid_reads_structure_a_flat_plot_would_pool`
#: asks it before the list is walked at all.
_INFERENCE_ORDER: tuple[KindHandler, ...] = (
    PULSE_TIMELINE_HANDLER,
    IMAGE_HANDLER,
    CURVE_HANDLER,
)


def _grid_reads_structure_a_flat_plot_would_pool(schema: Any) -> bool:
    """Whether one flat plot of this dataset would pool something real.

    A flat image IS one frame: every point row and every live scan dimension
    it carries is reduced into that one picture.  A flat curve walks ONE
    dimension: a second live dimension is pooled into it.  Where that
    happens, the structure being pooled is the measurement -- two frames of a
    cycle, seven points of a scan -- and the operator asked to see the
    signal, not its average.

    Degenerate axes are provenance, not structure, so they are invisible here
    exactly as they are everywhere else; a single-frame cycle and a
    one-dimensional scan of scalars keep the plain kind that already shows
    all of them.
    """

    if not isinstance(schema, DatasetSchema):
        return False
    live = live_grid_dimensions(schema)
    dense = tuple(axis for axis in schema.cell_schema.data_axes if axis.size > 1)
    if len(dense) >= 2:
        # A frame per point: any live point structure is a second frame.
        return bool(live) or schema.point_table.row_count > 1
    # A scalar or a single vector per point: the curve walks one dimension.
    return len(live) >= 2


def fitting_spec(schema: Any, kind: Any = None) -> Any:
    """A spec this dataset admits, or ``None`` if it admits none.

    With no ``kind`` the dataset's own structure decides -- the one answer to
    "just show me this signal".  Callers that hardwire a kind instead work
    until the first dataset that is not that kind, and then fail inside
    whoever asked, which for a GUI is a click that raises out of a slot.

    With a ``kind`` the question is the operator's -- "show me this AS a
    curve" -- and the answer is that kind's own default or None when this data
    cannot be drawn that way.  One entry point for both, because "can this be
    drawn like that" is one question however it is asked.
    """

    if kind is not None:
        return default_spec(schema, kind)
    if _grid_reads_structure_a_flat_plot_would_pool(schema):
        grid = FACET_GRID_HANDLER.default_spec(schema)
        if grid is not None:
            return grid
    for handler in _INFERENCE_ORDER:
        candidate = handler.default_spec(schema)
        if candidate is not None:
            return candidate
    return None


def panel_kinds() -> tuple[tuple[str, str], ...]:
    """Every plot kind as (value, display name), in registry order.

    One listing.  A window that types its own list offers kinds that may not
    exist and misses ones that do -- the console's kind picker held a single
    hardcoded entry while six kinds were registered.
    """

    return tuple((handler.kind.value, handler.display_name) for handler in HANDLERS)


__all__ = [
    "HANDLERS", "KindHandler", "default_spec", "fitting_spec", "handler_for",
    "panel_kinds",
]
