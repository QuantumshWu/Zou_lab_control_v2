"""The default table, pinned: what every kind shows unasked, by axis family.

No axis is chosen by its name here, and none may be: the schemas below
carry roles (SCAN_POINT, READOUT_EVENT, PRIMARY_INDEX, SPATIAL_X/Y, SITE)
and sizes, and the expectations are the table in
:mod:`zlc_plot._kinds.defaults` read row by row.  A change to a default is
a change to this file, on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import (
    LATEST_COORDINATE,
    PRIMARY_INDEX,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_plot._kinds import default_spec, fitting_spec
from zlc_plot._kinds.defaults import chosen_spec
from zlc_plot.data_contract import classify_axes
from zlc_plot.kinds import AxisRef, PlotKind
from zlc_plot.specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    RollingPlot,
)


def _scan(dims: dict[str, int], *, repeats: int = 1, cell_axes=(), extra_columns=None, roles=None):
    """A Cartesian scan of the named dimensions, with optional cell axes."""

    rows = list(np.ndindex(*(size for size in dims.values())))
    columns = {name: [row[index] for row in rows] for index, name in enumerate(dims)}
    column_roles = {name: SCAN_POINT for name in dims}
    if extra_columns:
        columns.update(extra_columns)
    if roles:
        column_roles.update(roles)
    table = mapped_domain_from_columns(columns, roles=column_roles)
    return make_dataset_schema(
        repeat_domain(size=repeats),
        table,
        cell_axes=tuple(cell_axes),
    )


def _cycle(frames: int, *, repeats: int = 1, cell_axes=(), role=READOUT_EVENT):
    """One camera cycle: its frames are the point rows, under one Point domain."""

    return make_dataset_schema(
        repeat_domain(size=repeats),
        mapped_domain_from_columns({"frame": list(range(frames))}, roles={"frame": role}),
        cell_axes=tuple(cell_axes),
    )


def _picture(height: int = 4, width: int = 6):
    return (
        axis("y", size=height, role=SPATIAL_Y),
        axis("x", size=width, role=SPATIAL_X),
    )


def _sites(count: int = 7):
    return (axis("site", size=count, role=SITE),)


def _history(shots: int, frames: int = 1, *, cell_axes=()):
    """The Runtime's materialized shot history of a cycle."""

    columns = {
        "shot": [shot for shot in range(shots) for _frame in range(frames)],
        "frame": [frame for _shot in range(shots) for frame in range(frames)],
    }
    return make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns(columns, roles={"shot": PRIMARY_INDEX, "frame": READOUT_EVENT}),
        cell_axes=tuple(cell_axes),
    )


P = AxisRef.point
D = AxisRef.cell_data
DIM = AxisRef.point


def test_families_are_read_by_role_never_by_name() -> None:
    families = classify_axes(_scan({"a": 3, "b": 4}, repeats=6, cell_axes=_sites(5)))
    assert families.repeat == ((AxisRef.repeat("repeat"), 6),)
    assert families.scan == ((DIM("a"), 3), (DIM("b"), 4))
    assert families.events == ()
    assert families.history is None
    assert families.picture is None
    assert families.content == ((D("site"), 5),)

    cycle = classify_axes(_cycle(3, repeats=30, cell_axes=_picture()))
    assert cycle.events == ((P("frame"), 3),)
    assert cycle.picture == ((D("x"), 6), (D("y"), 4))
    assert cycle.content == ()

    history = classify_axes(_history(40, cell_axes=_picture()))
    assert history.history == (P("shot"), 40)
    assert history.events == ((P("frame"), 1),)


# ------------------------------------------------------------- the table
def test_a_three_dimension_scalar_scan_facets_the_outer_dimension_over_a_heatmap() -> None:
    schema = _scan({"a": 3, "b": 4, "c": 5})
    grid = default_spec(schema, PlotKind.FACET_GRID)
    assert grid == FacetGridPlot(DIM("a"), ImagePlot(DIM("c"), DIM("b")))
    assert default_spec(schema, PlotKind.IMAGE) == ImagePlot(DIM("c"), DIM("b"))
    assert default_spec(schema, PlotKind.CURVE) == CurvePlot(DIM("c"))


def test_a_two_dimension_scalar_scan_is_one_heatmap_whatever_the_repeats() -> None:
    for repeats in (1, 6):
        schema = _scan({"a": 3, "b": 4}, repeats=repeats)
        assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(None, ImagePlot(DIM("b"), DIM("a")))
        assert default_spec(schema, PlotKind.CURVE) == CurvePlot(DIM("b"))


def test_a_scanned_picture_facets_the_scan_and_reduces_the_repeats() -> None:
    schema = _scan({"a": 3}, repeats=4, cell_axes=_picture())
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(DIM("a"), ImagePlot(D("x"), D("y")))
    assert default_spec(schema, PlotKind.IMAGE) == ImagePlot(D("x"), D("y"))
    # A curve walks the scan; the picture's two axes are not one group.
    assert default_spec(schema, PlotKind.CURVE) == CurvePlot(DIM("a"))

    # A degenerate dimension is invisible; the live one faces the frames.
    schema = _scan({"a": 1, "b": 3}, cell_axes=_picture())
    assert default_spec(schema, PlotKind.FACET_GRID).facet == DIM("b")
    assert default_spec(schema, PlotKind.IMAGE) == ImagePlot(D("x"), D("y"))

    # Two live dimensions: the outermost faces, the inner one is reduced
    # and stays an edit away in the fate table.
    schema = _scan({"a": 2, "b": 3}, repeats=4, cell_axes=_picture())
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(DIM("a"), ImagePlot(D("x"), D("y")))

    # Nothing live to face: repeat is history, not a layout axis.
    schema = _scan({"a": 1}, repeats=4, cell_axes=_picture())
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(None, ImagePlot(D("x"), D("y")))


def test_a_cycle_gives_each_frame_a_cell_and_a_single_image_shows_the_latest() -> None:
    schema = _cycle(2, repeats=30, cell_axes=_picture())
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(P("frame"), ImagePlot(D("x"), D("y")))
    # One image cannot give two frames a role: it shows the latest, never
    # the mean of two different frames.
    assert default_spec(schema, PlotKind.IMAGE) == ImagePlot(
        D("x"), D("y"), scope=((P("frame"), LATEST_COORDINATE),)
    )
    # A curve with no scan walks the events.
    assert default_spec(schema, PlotKind.CURVE) == CurvePlot(P("frame"))

    # A one-frame cycle keeps the frame as the cell's identity: the panel
    # must not change shape with the frame count.
    one = _cycle(1, repeats=30, cell_axes=_picture())
    assert default_spec(one, PlotKind.FACET_GRID) == FacetGridPlot(P("frame"), ImagePlot(D("x"), D("y")))
    assert default_spec(one, PlotKind.IMAGE) == ImagePlot(D("x"), D("y"))


def test_judged_frames_facet_by_frame_and_walk_the_sites() -> None:
    schema = _cycle(3, repeats=20, cell_axes=_sites(35))
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(P("frame"), CurvePlot(D("site")))
    assert fitting_spec(schema, PlotKind.FACET_GRID, cell=PlotKind.HISTOGRAM) == FacetGridPlot(
        P("frame"), HistogramPlot()
    )
    # A single curve walks the frames; thirty-five sites are a fog, reduced.
    assert default_spec(schema, PlotKind.CURVE) == CurvePlot(P("frame"))
    # A single image is a map of frame against site.
    assert default_spec(schema, PlotKind.IMAGE) == ImagePlot(P("frame"), D("site"))
    # Rolling groups the sites when the palette can tell them apart.
    few = _cycle(3, repeats=20, cell_axes=_sites(5))
    assert default_spec(few, PlotKind.ROLLING) == RollingPlot(
        group=D("site"), scope=((P("frame"), LATEST_COORDINATE),)
    )


def test_frame_pairs_are_events_like_the_frames_they_came_from() -> None:
    """Survival: (cycles) x (pairs) x (sites) reads exactly like judged frames."""

    schema = make_dataset_schema(
        repeat_domain(size=20),
        mapped_domain_from_columns({"pair": [0, 1, 2]}, roles={"pair": READOUT_EVENT}),
        cell_axes=_sites(35),
    )
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(P("pair"), CurvePlot(D("site")))
    assert default_spec(schema, PlotKind.CURVE) == CurvePlot(P("pair"))


def test_a_site_resolved_scan_spends_position_before_content() -> None:
    # Two live scan dimensions are the heatmap; the sites are reduced.
    schema = _scan({"a": 3, "b": 4}, cell_axes=_sites(7))
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(
        None, ImagePlot(DIM("b"), DIM("a"))
    )
    assert default_spec(schema, PlotKind.IMAGE) == ImagePlot(DIM("b"), DIM("a"))
    # A curve still walks the innermost dimension and tells the sites apart.
    assert default_spec(schema, PlotKind.CURVE) == CurvePlot(DIM("b"), group=D("site"))
    # Three: the outermost is the facet of the inner two's heatmap.
    three = _scan({"a": 3, "b": 4, "c": 5}, cell_axes=_sites(35))
    assert default_spec(three, PlotKind.FACET_GRID) == FacetGridPlot(
        DIM("a"), ImagePlot(DIM("c"), DIM("b"))
    )
    assert fitting_spec(three, PlotKind.FACET_GRID, cell=PlotKind.CURVE) == (
        FacetGridPlot(DIM("a"), CurvePlot(DIM("c")))
    )

    # One scan dimension: the curve IS the walk and nothing is left to face.
    one = _scan({"t": 6}, repeats=10, cell_axes=_sites(7))
    assert default_spec(one, PlotKind.FACET_GRID) == FacetGridPlot(None, CurvePlot(DIM("t"), group=D("site")))


def test_a_shot_history_faces_the_shots_and_a_curve_walks_its_own_data() -> None:
    schema = _history(40, cell_axes=_picture(40, 500))
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(P("shot"), ImagePlot(D("x"), D("y")))
    # No scan and no live event: the curve walks the picture's own x, and
    # the shots are reduced -- the mean over shots is the statistic.
    assert default_spec(schema, PlotKind.CURVE) == CurvePlot(D("x"))
    # Nothing but shots to walk: they are the curve's x of last resort.
    scalar = _history(40)
    assert default_spec(scalar, PlotKind.CURVE) == CurvePlot(P("shot"))


def test_a_repeated_scalar_at_one_point_is_a_distribution_per_point() -> None:
    schema = make_dataset_schema(
        repeat_domain(size=20),
        mapped_domain_from_columns({"x": [0.0]}),
    )
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(None, HistogramPlot())
    flat = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": np.arange(4.0)}),
    )
    assert default_spec(flat, PlotKind.FACET_GRID) == FacetGridPlot(None, CurvePlot(P("x")))


def test_a_named_cell_kind_keeps_the_grids_facet_and_refuses_what_it_cannot_fill() -> None:
    schema = _cycle(3, repeats=20, cell_axes=_sites(35))
    assert fitting_spec(schema, PlotKind.FACET_GRID, cell=PlotKind.CURVE) == FacetGridPlot(
        P("frame"), CurvePlot(D("site"))
    )
    scalar = _scan({"a": 3})
    assert fitting_spec(scalar, PlotKind.FACET_GRID, cell=PlotKind.IMAGE) is None
    with pytest.raises(ValueError):
        fitting_spec(scalar, PlotKind.CURVE, cell=PlotKind.IMAGE)


def test_asking_for_a_grid_from_the_plot_in_hand_keeps_its_kind_as_the_cell() -> None:
    schema = _cycle(3, repeats=20, cell_axes=_sites(35))
    assert chosen_spec(schema, HistogramPlot()) == FacetGridPlot(P("frame"), HistogramPlot())
    # A kind that cannot be a cell, or a cell the data cannot fill, lets
    # the data decide.
    assert chosen_spec(schema, RollingPlot()) == FacetGridPlot(P("frame"), CurvePlot(D("site")))
    assert chosen_spec(_scan({"a": 3}), ImagePlot(DIM("a"), D("site"))) == FacetGridPlot(None, CurvePlot(DIM("a")))


def test_point_axes_keep_the_producers_declaration_order() -> None:

    import numpy as np
    schema = make_dataset_schema(
        repeat_domain(size=2),
        mapped_domain_from_columns(
            {"x": np.arange(4.0), "row": np.array([0.0, 0.0, 1.0, 1.0])}
        ),
        dtype=np.float64,
    )
    families = classify_axes(schema)
    assert [ref.axis_id for ref, _size in families.scan] == ["x", "row"]
    assert default_spec(schema, PlotKind.CURVE) == CurvePlot(P("row"))
    # Two live scan dimensions of a scalar are a heatmap, bare or not.
    assert default_spec(schema, PlotKind.FACET_GRID) == FacetGridPlot(
        None, ImagePlot(P("row"), P("x"))
    )
    assert fitting_spec(schema, PlotKind.FACET_GRID, cell=PlotKind.CURVE) == (
        FacetGridPlot(P("x"), CurvePlot(P("row")))
    )
