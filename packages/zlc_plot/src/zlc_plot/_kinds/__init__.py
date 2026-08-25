"""Closed plot-kind registry shared by projection and renderer layers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from zlc_data import LATEST_COORDINATE
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from ..kinds import PlotKind
from ..kinds import AxisRef
from .base import KindHandler
from .curve import HANDLER as CURVE_HANDLER
from .facet_grid import HANDLER as FACET_GRID_HANDLER, cell_within_one_cell
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


def _with_primary_index_default(schema: Any, candidate: Any) -> Any:
    """Default ordinary plots to the latest cell of an indexed Dataset."""

    if candidate is None or not hasattr(schema, "point_table"):
        return candidate
    if not any(
        column.coordinate_id == PRIMARY_INDEX_AXIS_ID
        for column in schema.point_table.columns
    ):
        return candidate
    from ..config import DEFAULTS
    from ..specs import parameter_schema_for

    if "window" in parameter_schema_for(candidate, style=DEFAULTS.style):
        return candidate
    primary = AxisRef.point(PRIMARY_INDEX_AXIS_ID.value)
    semantic = getattr(candidate, "cell", candidate)
    used = {
        getattr(semantic, "x", None),
        getattr(semantic, "y", None),
        getattr(semantic, "group", None),
        getattr(candidate, "facet", None),
    }
    scopes = tuple(getattr(candidate, "scope", ()))
    if primary in used or any(ref == primary for ref, _value in scopes):
        return candidate
    return replace(candidate, scope=(*scopes, (primary, LATEST_COORDINATE)))


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
    candidate = _with_primary_index_default(schema, handler.default_spec(schema))
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
#: A facet grid is not in this list either, and for the original reason: a
#: grid is a way of LOOKING at data rather than something the data proves,
#: so it is asked for -- by an operator choosing it, or by the node whose
#: Start opened the panel, which knows what it measured and says so in its
#: preview declaration.  Inferring one from shape alone is guessing at
#: intent, and a shape-reading rule here was exactly that.
_INFERENCE_ORDER: tuple[KindHandler, ...] = (
    PULSE_TIMELINE_HANDLER,
    IMAGE_HANDLER,
    CURVE_HANDLER,
)


def fitting_spec(schema: Any, kind: Any = None, *, cell: Any = None) -> Any:
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

    if cell is not None and kind is not PlotKind.FACET_GRID:
        raise ValueError("only a facet grid has a cell kind")
    if kind is not None:
        grid = default_spec(schema, kind)
        if cell is None or grid is None:
            return grid
        # Naming a cell kind changes the KIND of the cell, never the rule
        # for what one cell shows: that stays with the grid that composed it.
        inner = default_spec(schema, cell)
        if inner is None:
            return None
        inner = cell_within_one_cell(schema, grid.facet, inner)
        return None if inner is None else replace(grid, cell=inner)
    for handler in _INFERENCE_ORDER:
        candidate = _with_primary_index_default(
            schema, handler.default_spec(schema)
        )
        if candidate is not None:
            return candidate
    return None


__all__ = [
    "HANDLERS", "KindHandler", "default_spec", "fitting_spec", "handler_for",
]
