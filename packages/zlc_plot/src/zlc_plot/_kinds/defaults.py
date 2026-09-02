"""What an unauthored plot shows: ONE table, read by every kind.

A dataset's axes are grouped by what they ARE
(:func:`zlc_plot.data_contract.classify_axes`): R, the repeat axis; H, the
Runtime's shot index; S, the authored scan dimensions, slowest first; E,
the event sequences inside one cycle (a camera's frames, frame pairs); and
D, the data each point holds -- a declared picture and the other content
axes.  The table below says what each family is FOR, and it is the same
answer whichever kind is asked:

* R is a statistic.  It is reduced (the mean, with its standard error) or
  pooled by a histogram, and is never a layout axis on its own.
* H is a statistic too -- the mean over shots IS an occupation rate -- for
  every kind but the one that walks it, Rolling, and as a curve's x of
  last resort when nothing else has structure.
* S is position, and position is spent before content.  The innermost
  dimension is what one sweep walks (a curve's x); two of them are a
  heatmap, whatever content the point holds; the outermost is a grid's
  facet.
  A scan dimension nothing claims is reduced, and the fate table says so.
  Bare point columns with no topology are nested by their rows: the
  column that changes least often is the outermost.
* E is a choice of sub-measurement.  A grid gives each event a cell, a
  curve with no scan walks them, and anything else shows the LATEST one:
  the mean of two different frames is not a frame.
* D is the payload.  A declared picture, or two content axes, is an image;
  one content axis left over is a group when the palette can tell its
  members apart, and reduced otherwise.

Degenerate axes (one value) are provenance, not structure: they never
decide a layout, with one deliberate exception -- an event axis of one
(a one-frame cycle) still identifies a grid's cell, so the grid's meaning
does not change with the frame count.  No axis is ever chosen by its
name: only its role, its family and its place in the declaration order
decide.  Every kind's ``default_spec`` is a reading of this one table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zlc_data import LATEST_COORDINATE, DatasetSchema

from ..data_contract import AxisFamilies, classify_axes
from ..kinds import AxisRef, PlotKind
from ..specs import (
    GRID_CELL_KINDS,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    Reduction,
    RollingPlot,
    ScopeTerm,
)

_Entry = tuple[AxisRef, int]


def tellable_apart(count: int) -> bool:
    """Can a picture show this many lines AS separate lines?

    A group exists so its members can be told apart, and what tells them
    apart is their colour: past the end of the line cycle the colours
    repeat and the lines are a fog.  So the cycle is the bound, and it is
    the palette's to state -- not a number invented here.

    A camera frame is the case that made this matter: 1920 columns as x
    leaves 1200 rows as the single dense axis, which a rule once grouped.
    Measured, Matplotlib draws one noisy 1920-point line in 46 ms, so that
    default asked for a picture costing 24 SECONDS a frame and showing 1200
    indistinguishable lines.  Collapsed under the declared reduction it is
    one line, which is what the reduction is for; an operator who wants the
    rows apart still asks for them by name.
    """

    from ..config import DEFAULTS  # noqa: PLC0415

    return int(count) <= len(DEFAULTS.style.palette.line_cycle)


@dataclass(frozen=True)
class _Plan:
    """One cell's reading of the table: its roles, and what it consumed."""

    x: AxisRef | None = None
    y: AxisRef | None = None
    group: AxisRef | None = None
    #: Scan dimensions the cell walks, innermost first.
    scan_used: tuple[AxisRef, ...] = ()
    #: Whether the cell used an event axis as one of its roles.
    event_used: AxisRef | None = None

    @property
    def consumed(self) -> tuple[AxisRef, ...]:
        return tuple(
            ref
            for ref in (self.x, self.y, self.group, self.event_used)
            if ref is not None
        ) + self.scan_used


def default_spec(
    schema: Any, kind: PlotKind, *, cell_kind: PlotKind | None = None
) -> Any:
    """The specification ``kind`` shows for ``schema`` unasked, or ``None``.

    ``None`` means the kind has nothing to draw here (an image with nothing
    to image, a cell kind the data cannot fill).  A grid never answers
    ``None`` for want of a facet: with nothing left to face it is one cell.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    if cell_kind is not None and kind is not PlotKind.FACET_GRID:
        raise ValueError("only a facet grid has a cell kind")
    families = classify_axes(schema)
    if kind is PlotKind.FACET_GRID:
        return _grid(families, cell_kind)
    if kind is PlotKind.CURVE:
        return _curve_spec(families, _curve_plan(families, facet=None))
    if kind is PlotKind.IMAGE:
        plan = _image_plan(families)
        return None if plan is None else _image_spec(families, plan)
    if kind is PlotKind.HISTOGRAM:
        return HistogramPlot()
    if kind is PlotKind.ROLLING:
        return _rolling_spec(families)
    raise ValueError(f"{kind!r} has no table entry; its handler owns its default")


def chosen_spec(schema: Any, current: Any) -> FacetGridPlot | None:
    """A grid for an operator who ASKED for one, from the plot they had.

    The plot they were looking at becomes the cell kind when it can be
    one; a kind that cannot be a cell, or a cell the data cannot fill,
    lets the data decide the cell instead.  Either way the facet comes
    from the table, and the fate table is where the operator says what
    they actually meant.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    current_kind = getattr(current, "kind", None)
    if current_kind in GRID_CELL_KINDS:
        asked = default_spec(schema, PlotKind.FACET_GRID, cell_kind=current_kind)
        if asked is not None:
            return asked
    return default_spec(schema, PlotKind.FACET_GRID)


# --------------------------------------------------------------- the table
def _densest_cell_kind(families: AxisFamilies) -> PlotKind:
    """What one cell shows when nobody said: the densest structure there is."""

    if families.picture is not None or len(families.live_scan()) >= 2:
        # A declared picture, or two live scan dimensions: position is
        # spent before content, so a field map of a per-site quantity is
        # the scan's heatmap with the sites reduced, not a curve per site.
        return PlotKind.IMAGE
    if len(families.live_content()) >= 2:
        return PlotKind.IMAGE
    if (
        not families.live_content()
        and not families.live_scan()
        and not families.live_events()
        and families.history is None
        and families.repeat[1] > 1
        and families.has_point_columns
    ):
        # A scalar measured repeatedly at authored point coordinates is a
        # distribution per point.  A curve would consume that point axis
        # and leave only repeat, which must not become the automatic
        # facet; the histogram consumes values instead and leaves the
        # authored coordinate as the honest facet.
        return PlotKind.HISTOGRAM
    return PlotKind.CURVE


def _grid(families: AxisFamilies, cell_kind: PlotKind | None) -> FacetGridPlot | None:
    kind = _densest_cell_kind(families) if cell_kind is None else cell_kind
    if kind not in GRID_CELL_KINDS:
        raise ValueError(f"a grid cell cannot be a {kind.value}")
    if kind is PlotKind.HISTOGRAM:
        facet = _facet(families, _Plan())
        return FacetGridPlot(
            facet, HistogramPlot(), scope=_latest_events(families, (facet,))
        )
    if kind is PlotKind.IMAGE:
        plan = _image_plan(families)
        if plan is None:
            return None
        facet = _facet(families, plan)
        cell = _image_spec(families, plan, scoped=False)
        return FacetGridPlot(
            facet, cell, scope=_latest_events(families, plan.consumed + (facet,))
        )
    # A curve cell walks the innermost scan dimension itself only when a
    # dimension outside it is left for the grid to face; with one scan
    # dimension the curve IS the walk and the grid faces the events.  With
    # no scan the grid faces the events and the curve walks what is left
    # inside a cell -- the sites of one judged frame.
    live_scan = families.live_scan()
    if len(live_scan) >= 2:
        facet: AxisRef | None = live_scan[0][0]
    elif live_scan:
        facet = _first_live(families.live_events()) or _live_history(families)
        if facet is None:
            facet = _first_any(families.events)
    else:
        facet = _facet(families, _Plan())
    plan = _curve_plan(families, facet=facet)
    cell = _curve_spec(families, plan, scoped=False)
    return FacetGridPlot(
        facet, cell, scope=_latest_events(families, plan.consumed + (facet,))
    )


def _facet(families: AxisFamilies, plan: _Plan) -> AxisRef | None:
    """The first axis a grid faces that the cell did not consume.

    Live structure first, whichever family holds it -- a scan's outermost
    free dimension, then an event sequence, then the shot history -- and
    with none of those, an event axis of one: a one-frame cycle still
    names its cell, so the grid's meaning does not depend on the count.
    """

    consumed = set(ref.physical_identity for ref in plan.consumed)
    for ref, _size in families.live_scan():
        if ref.physical_identity not in consumed:
            return ref
    for ref, _size in families.live_events():
        if ref.physical_identity not in consumed:
            return ref
    history = _live_history(families)
    if history is not None and history.physical_identity not in consumed:
        return history
    if not families.topology:
        # Without a scan topology the point column IS the authored cell
        # identity, at one value as much as at three: a one-frame cycle,
        # a scalar measured at one authored point.  A scan's degenerate
        # dimension, by contrast, is invisible.
        for ref, _size in families.events + families.scan:
            if ref.physical_identity not in consumed:
                return ref
    return None


def _image_plan(families: AxisFamilies) -> _Plan | None:
    picture = families.picture
    if picture is not None:
        return _Plan(x=picture[0][0], y=picture[1][0])
    scan = families.live_scan()
    content = families.live_content()
    if len(scan) >= 2:
        # Position before content.  The last live dimension is the
        # innermost loop: the horizontal axis of the heatmap, exactly as a
        # curve walks it; whatever content the point holds is reduced.
        return _Plan(
            x=scan[-1][0], y=scan[-2][0], scan_used=(scan[-1][0], scan[-2][0])
        )
    if len(content) >= 2:
        # Data axes are declared slowest-first, so the last is horizontal.
        return _Plan(x=content[-1][0], y=content[-2][0])
    if len(content) == 1:
        # One coordinate from the points and one dense axis are still two
        # coordinates: a scan of a per-site quantity IS a map of site
        # against what was scanned.
        if scan:
            return _Plan(x=scan[-1][0], y=content[0][0], scan_used=(scan[-1][0],))
        events = families.live_events()
        if events:
            return _Plan(x=events[0][0], y=content[0][0], event_used=events[0][0])
        history = _live_history(families)
        if history is not None:
            return _Plan(x=history, y=content[0][0])
        rows = families.live_rows()
        if rows is not None:
            return _Plan(x=rows, y=content[0][0])
    return None


def _image_spec(
    families: AxisFamilies, plan: _Plan, *, scoped: bool = True
) -> ImagePlot:
    return ImagePlot(
        plan.x,
        plan.y,
        reduction=Reduction.MEAN,
        scope=_latest_events(families, plan.consumed) if scoped else (),
    )


def _curve_plan(families: AxisFamilies, *, facet: AxisRef | None) -> _Plan:
    """A curve walks position: the innermost scan loop, else the events,
    else the shot history, else its own data, else the bare point rows."""

    taken = set() if facet is None else {facet.physical_identity}
    scan = [entry for entry in families.live_scan() if entry[0].physical_identity not in taken]
    if scan:
        x = scan[-1][0]
        return _Plan(
            x=x,
            group=_group(families, (x, facet)),
            scan_used=(x,),
        )
    events = [entry for entry in families.live_events() if entry[0].physical_identity not in taken]
    if events:
        x = events[0][0]
        return _Plan(x=x, group=_group(families, (x, facet)), event_used=x)
    data_innermost_first = tuple(reversed(families.live_data()))
    if data_innermost_first:
        x = data_innermost_first[0][0]
        return _Plan(x=x, group=_group(families, (x, facet)))
    history = _live_history(families)
    if history is not None and history.physical_identity not in taken:
        return _Plan(x=history)
    rows = families.live_rows()
    if rows is not None:
        return _Plan(x=rows)
    # Nothing has structure: the default x must still be an axis, so the
    # first one there is, degenerate as it may be, is drawn as one point.
    return _Plan(x=_first_any(families.scan) or _first_any(families.events) or families.first_data_axis() or families.rows_or_none() or families.repeat[0])


def _curve_spec(
    families: AxisFamilies, plan: _Plan, *, scoped: bool = True
) -> CurvePlot:
    return CurvePlot(
        plan.x,
        group=plan.group,
        reduction=Reduction.MEAN,
        scope=_latest_events(families, plan.consumed) if scoped else (),
    )


def _rolling_spec(families: AxisFamilies) -> RollingPlot:
    return RollingPlot(
        group=_group(families, ()),
        reduction=Reduction.MEAN,
        scope=_latest_events(families, ()),
    )


def _group(
    families: AxisFamilies, taken: tuple[AxisRef | None, ...]
) -> AxisRef | None:
    """The one live data axis left over, when its members can be told apart."""

    used = {ref.physical_identity for ref in taken if ref is not None}
    free = [
        entry
        for entry in families.live_data()
        if entry[0].physical_identity not in used
    ]
    if len(free) == 1 and tellable_apart(free[0][1]):
        return free[0][0]
    return None


def _latest_events(
    families: AxisFamilies, consumed: tuple[AxisRef | None, ...]
) -> tuple[ScopeTerm, ...]:
    """Every live event axis with no role shows its latest event."""

    used = {ref.physical_identity for ref in consumed if ref is not None}
    return tuple(
        (ref, LATEST_COORDINATE)
        for ref, _size in families.live_events()
        if ref.physical_identity not in used
    )


def _first_live(entries: tuple[_Entry, ...]) -> AxisRef | None:
    return entries[0][0] if entries else None


def _first_any(entries: tuple[_Entry, ...]) -> AxisRef | None:
    return entries[0][0] if entries else None


def _live_history(families: AxisFamilies) -> AxisRef | None:
    history = families.history
    return None if history is None or history[1] <= 1 else history[0]
