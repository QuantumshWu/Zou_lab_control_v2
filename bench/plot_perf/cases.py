"""The measurement matrix: every plot kind x its meaningful interactions.

Each case names a feed, a spec, the interactions that make sense on it,
and (optionally) a fit request.  The runner interprets interaction tags
against the painted front's axis roles, so a tag means the same gesture on
every kind that offers the role.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .common import SnapshotFeed, camera_feed, lattice_feed, pulse_feed


def open_session(case, feed):
    """The session a case means, built the one way.

    A case is a spec AND its parameters: ``presentation="height_bars"`` is
    what makes the two h3d cases 3D at all, and it is not part of the spec.
    Built by hand in each runner, the attribution runner left it out and
    profiled two ordinary heatmaps under the 3D cases' own names, with
    nothing in its report to say so.
    """

    from zlc_plot import PlotSession

    from .common import SIZE_PRESET

    # SIZE AT CONSTRUCTION, which is what the product does: build_host takes
    # the panel's size and the renderer lays out once.  Set afterwards, the
    # session lays out twice -- and on a sixty-four cell grid the second
    # layout throws away a hundred and twenty-eight Axes and builds them
    # again, so a bench that set it that way reported 1.43 s for a first
    # frame the product paints in 0.56.
    session = PlotSession(feed.next(), case.spec(), size=SIZE_PRESET)
    if case.parameters:
        session.set_parameters(dict(case.parameters))
    return session


@dataclass(frozen=True)
class Case:
    name: str
    feed: Callable[[], SnapshotFeed]
    spec: Callable[[], object]
    interactions: tuple[str, ...] = ()
    fit: dict | None = None
    #: Display parameters the case is measured UNDER (a presentation, a
    #: window): the same spec drawn a different way is a different cost.
    parameters: dict | None = None
    notes: str = ""


def _specs():
    from zlc_plot import (
        AxisRef,
        CurvePlot,
        FacetGridPlot,
        HistogramPlot,
        ImagePlot,
        RollingPlot,
    )

    return AxisRef, CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, RollingPlot


def catalog() -> tuple[Case, ...]:
    AxisRef, CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, RollingPlot = _specs()
    lattice_2m = lambda: lattice_feed()
    lattice_20m = lambda: lattice_feed(repeats=200, buffers=2)
    # A shot an operator actually takes early in a run: one repeat over a
    # small scan.  A cost that is invisible at two million points is the
    # whole of one here, and the panels these open on are the ones a
    # console is judged by.
    lattice_small = lambda: lattice_feed(
        repeats=1, rows=200, frames=1, sites=4, dims=(10, 20)
    )
    # A run still filling: a quarter of the repeats have landed and the
    # producer dropped shots inside them.  Every other feed here is wholly
    # valid, which is the cheap half of three forks the product takes --
    # masked reductions, the raster's masked path and its counts array,
    # and the curve's isolated-point channel for a series with holes.
    lattice_partial_2m = lambda: lattice_feed(landed=0.25, hole_stride=11)
    lattice_partial_small = lambda: lattice_feed(
        repeats=4, rows=200, frames=1, sites=4, dims=(10, 20),
        landed=0.5, hole_stride=7,
    )
    return (
        Case(
            "curve_2M",
            lattice_2m,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("hover_series", "click_series", "drag_main", "pan_drag", "wheel_main"),
        ),
        Case(
            "curve_tensor_group_2M",
            lattice_2m,
            lambda: CurvePlot(
                AxisRef.cell_data("frame"), group=AxisRef.cell_data("site")
            ),
            ("hover_series", "click_series", "drag_main"),
            notes="tensor x and tensor group",
        ),
        Case(
            "curve_point_group_2M",
            lattice_2m,
            lambda: CurvePlot(
                AxisRef.point("ax"),
                group=AxisRef.point("ay"),
            ),
            ("hover_series", "click_series", "drag_main"),
            notes="two point-topology roles",
        ),
        Case(
            "curve_20M",
            lattice_20m,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("hover_series", "drag_main"),
        ),
        Case(
            "hist_2M",
            lattice_2m,
            lambda: HistogramPlot(),
            ("drag_main", "drag_threshold", "wheel_main"),
        ),
        Case(
            "image_heatmap_2M",
            lattice_2m,
            lambda: ImagePlot(
                AxisRef.point("ax"), AxisRef.point("ay")
            ),
            ("drag_main", "drag_clim", "click_main", "pan_drag", "wheel_main"),
        ),
        Case(
            "image_mixed_axes_2M",
            lattice_2m,
            lambda: ImagePlot(
                AxisRef.point("ax"), AxisRef.cell_data("site")
            ),
            ("drag_main", "drag_clim", "click_main", "pan_drag", "wheel_main"),
            notes="one point-topology and one tensor image axis",
        ),
        Case(
            "image_repeat_data_2M",
            lattice_2m,
            lambda: ImagePlot(AxisRef.cell_data("site"), AxisRef.repeat("repeat")),
            ("drag_main", "drag_clim", "click_main", "pan_drag", "wheel_main"),
            notes="two tensor image axes, including repeat",
        ),
        Case(
            "image_camera_4M",
            camera_feed,
            lambda: ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
            ("drag_main", "drag_clim", "click_main", "pan_drag", "wheel_main"),
            notes="2048x2048 uint16 dense camera frame",
        ),
        Case(
            "h3d_bars_2M",
            lattice_2m,
            lambda: ImagePlot(
                AxisRef.point("ax"), AxisRef.point("ay")
            ),
            ("drag_orbit", "drag_clim", "click_main", "wheel_main"),
            parameters={"presentation": "height_bars"},
            notes="3D height bars over the 10x10 lattice",
        ),
        Case(
            "h3d_bars_dense_2M",
            lambda: lattice_feed(
                repeats=4, rows=40000, frames=1, sites=1, dims=(200, 200)
            ),
            lambda: ImagePlot(
                AxisRef.point("ax"), AxisRef.point("ay")
            ),
            ("drag_orbit",),
            parameters={"presentation": "height_bars"},
            notes="3D height bars over a 200x200 grid (pooled)",
        ),
        Case(
            "rolling_2M",
            lattice_2m,
            lambda: RollingPlot(),
            ("hover_series",),
        ),
        Case(
            "facet_frame3_2M",
            lattice_2m,
            lambda: FacetGridPlot(
                AxisRef.cell_data("frame"), CurvePlot(AxisRef.point("ax"))
            ),
            ("hover_series", "dclick_cell"),
        ),
        Case(
            "facet_scan10_2M",
            lattice_2m,
            lambda: FacetGridPlot(
                AxisRef.point("ax"), CurvePlot(AxisRef.point("ay"))
            ),
            ("dclick_cell",),
        ),
        Case(
            "facet64_curve_2M",
            lambda: lattice_feed(
                repeats=16, rows=2000, frames=1, sites=64, dims=(10, 10, 20)
            ),
            lambda: FacetGridPlot(
                AxisRef.cell_data("site"), CurvePlot(AxisRef.point("ax"))
            ),
            ("hover_series", "dclick_cell"),
            notes="64 curve cells",
        ),
        Case(
            "facet10_image_2M",
            lattice_2m,
            lambda: FacetGridPlot(
                AxisRef.point("ax"),
                ImagePlot(AxisRef.cell_data("site"), AxisRef.cell_data("frame")),
            ),
            ("dclick_cell",),
            notes="10 image cells (site x frame)",
        ),
        Case(
            "facet64_image_2M",
            lambda: lattice_feed(
                repeats=160, rows=100, frames=2, sites=64, dims=(10, 10)
            ),
            lambda: FacetGridPlot(
                AxisRef.cell_data("site"),
                ImagePlot(
                    AxisRef.point("ax"), AxisRef.point("ay")
                ),
            ),
            ("dclick_cell",),
            notes="64 heatmap cells (ax x ay per site)",
        ),
        Case(
            "facet64_histogram_2M",
            lambda: lattice_feed(
                repeats=16, rows=2000, frames=1, sites=64, dims=(10, 10, 20)
            ),
            lambda: FacetGridPlot(
                AxisRef.cell_data("site"), HistogramPlot()
            ),
            ("dclick_cell",),
            notes="64 pooled histogram cells",
        ),
        Case(
            "facet34_mixed_image_2M",
            lattice_2m,
            lambda: FacetGridPlot(
                AxisRef.cell_data("site"),
                ImagePlot(
                    AxisRef.point("ax"), AxisRef.cell_data("frame")
                ),
            ),
            ("dclick_cell",),
            notes="34 mixed-axis image cells",
        ),
        Case(
            "curve_small",
            lattice_small,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("hover_series", "drag_main"),
            notes="one repeat, 200 points",
        ),
        Case(
            "curve_plain_2M",
            lattice_2m,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("hover_series", "drag_main"),
            parameters={"uncertainty": False},
            notes="error bars OFF -- curve_2M is this same measurement with "
            "them on, which is the default, so the pair prices the band",
        ),
        Case(
            "curve_plain_small",
            lattice_small,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("hover_series",),
            parameters={"uncertainty": False},
            notes="error bars OFF on one repeat, against curve_small",
        ),
        Case(
            "hist_small",
            lattice_small,
            lambda: HistogramPlot(),
            ("drag_main", "drag_threshold"),
            notes="pooled histogram of a small shot",
        ),
        Case(
            "hist_classifier_2M",
            lattice_2m,
            lambda: HistogramPlot(),
            ("drag_threshold",),
            parameters={"threshold_classifier": True},
            notes="the classifier's threshold bar and its label",
        ),
        Case(
            "image_small",
            lattice_small,
            lambda: ImagePlot(AxisRef.point("ax"), AxisRef.point("ay")),
            ("drag_clim", "click_main"),
            notes="a 20x10 heatmap",
        ),
        Case(
            "rolling_plain_2M",
            lattice_2m,
            lambda: RollingPlot(),
            ("hover_series",),
            parameters={"uncertainty": False, "side_distribution": False},
            notes="band and side distribution OFF -- both DEFAULT to on, so "
            "rolling_2M was this same measurement under a second name",
        ),
        Case(
            "facet4_curve_small",
            lattice_small,
            lambda: FacetGridPlot(
                AxisRef.cell_data("site"), CurvePlot(AxisRef.point("ax"))
            ),
            ("dclick_cell",),
            notes="four curve cells on a small shot",
        ),
        Case(
            "facet4_histogram_small",
            lattice_small,
            lambda: FacetGridPlot(
                AxisRef.cell_data("site"), HistogramPlot()
            ),
            ("dclick_cell",),
            notes="four pooled histogram cells on a small shot",
        ),
        Case(
            "fit_curve_small",
            lattice_small,
            lambda: CurvePlot(AxisRef.point("ax")),
            (),
            fit={"model": "gaussian_offset"},
            notes="the first fit an operator asks for, on the first shot",
        ),
        Case(
            "fit_curve_2M",
            lattice_2m,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("drag_main",),
            fit={"model": "gaussian_offset"},
            notes="live gaussian fit re-solved per revision",
        ),
        Case(
            "curve_partial_2M",
            lattice_partial_2m,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("hover_series", "drag_main"),
            notes="a run still filling: a quarter landed, holes inside it. "
            "curve_2M is the same measurement wholly valid",
        ),
        Case(
            "image_partial_2M",
            lattice_partial_2m,
            lambda: ImagePlot(AxisRef.point("ax"), AxisRef.point("ay")),
            ("click_main", "wheel_main"),
            notes="the raster's masked path and its counts array, against "
            "image_heatmap_2M",
        ),
        Case(
            "facet64_histogram_partial_2M",
            lattice_partial_2m,
            lambda: FacetGridPlot(
                AxisRef.cell_data("site"), HistogramPlot()
            ),
            (),
            notes="masked reductions on sixty-four cells, against "
            "facet64_histogram_2M",
        ),
        Case(
            "curve_partial_small",
            lattice_partial_small,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("hover_series",),
            notes="few shots and holes: the curve's isolated-point channel "
            "is only reached by a series with gaps",
        ),
        Case(
            "fit_facet10_2M",
            lattice_2m,
            lambda: FacetGridPlot(
                AxisRef.point("ax"), CurvePlot(AxisRef.point("ay"))
            ),
            (),
            fit={"model": "gaussian_offset"},
            notes="facet batch fit, 10 cells",
        ),
        Case(
            "pulse_timeline_imaging",
            pulse_feed,
            lambda: _pulse_spec(),
            ("hover_series", "drag_main", "wheel_main"),
            notes="the editor's preview: the imaging template through the "
            "presenter's own projection; hover is the crosshair, the drag "
            "an x-range selector",
        ),
    )


def _pulse_spec():
    from zlc_plot import PlotLabels, PulseTimelinePlot

    return PulseTimelinePlot(PlotLabels())
