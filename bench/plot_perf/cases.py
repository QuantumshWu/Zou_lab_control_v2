"""The measurement matrix: every plot kind x its meaningful interactions.

Each case names a feed, a spec, the interactions that make sense on it,
and (optionally) a fit request.  The runner interprets interaction tags
against the painted front's axis roles, so a tag means the same gesture on
every kind that offers the role.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .common import SnapshotFeed, camera_feed, lattice_feed


@dataclass(frozen=True)
class Case:
    name: str
    feed: Callable[[], SnapshotFeed]
    spec: Callable[[], object]
    interactions: tuple[str, ...] = ()
    fit: dict | None = None
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
    return (
        Case(
            "curve_2M",
            lattice_2m,
            lambda: CurvePlot(AxisRef.point("ax")),
            ("hover_series", "click_series", "drag_main", "pan_drag", "wheel_main"),
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
                AxisRef.point_dimension("ax"), AxisRef.point_dimension("ay")
            ),
            ("drag_main", "drag_clim", "click_main", "pan_drag", "wheel_main"),
        ),
        Case(
            "image_camera_4M",
            camera_feed,
            lambda: ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
            ("drag_main", "drag_clim", "click_main", "pan_drag", "wheel_main"),
            notes="2048x2048 uint16 dense camera frame",
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
                AxisRef.data("frame"), CurvePlot(AxisRef.point("ax"))
            ),
            ("hover_series", "dclick_cell"),
        ),
        Case(
            "facet_scan10_2M",
            lattice_2m,
            lambda: FacetGridPlot(
                AxisRef.point_dimension("ax"), CurvePlot(AxisRef.point("ay"))
            ),
            ("dclick_cell",),
        ),
        Case(
            "facet64_curve_2M",
            lambda: lattice_feed(
                repeats=16, rows=2000, frames=1, sites=64, dims=(10, 10, 20)
            ),
            lambda: FacetGridPlot(
                AxisRef.data("site"), CurvePlot(AxisRef.point("ax"))
            ),
            ("hover_series", "dclick_cell"),
            notes="64 curve cells",
        ),
        Case(
            "facet10_image_2M",
            lattice_2m,
            lambda: FacetGridPlot(
                AxisRef.point_dimension("ax"),
                ImagePlot(AxisRef.data("site"), AxisRef.data("frame")),
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
                AxisRef.data("site"),
                ImagePlot(
                    AxisRef.point_dimension("ax"), AxisRef.point_dimension("ay")
                ),
            ),
            ("dclick_cell",),
            notes="64 heatmap cells (ax x ay per site)",
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
            "fit_facet10_2M",
            lattice_2m,
            lambda: FacetGridPlot(
                AxisRef.point_dimension("ax"), CurvePlot(AxisRef.point("ay"))
            ),
            (),
            fit={"model": "gaussian_offset"},
            notes="facet batch fit, 10 cells",
        ),
    )
