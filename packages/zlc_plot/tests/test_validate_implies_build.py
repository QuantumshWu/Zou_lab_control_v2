"""Passing ``validate_*`` IS being buildable -- one decision, not two.

``DataView.validate_curve`` promises it in its own docstring: "whatever
passes here projects, and whatever projects passed here".  That promise was
false: the real-numeric coordinate requirement lived at the BUILD sites
only, so a text point column sailed through validation and then raised on the
first draw.

The guard states the contract directly and mechanically: for a
representative set of schemas and specs, whichever way the ``validate_*``
authority rules, the DataView projection and a full ``PlotSession``
construction must rule the same way.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_plot import PlotSession
from zlc_plot.data_view import DataView, DataViewError
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    Reduction,
    RollingPlot,
)

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


#: Bin count is a DISPLAY parameter, not part of the histogram spec; the
#: projection needs one, so the guard picks a single value for every case.
_BINS = 8


def _snapshot(*, points, data_axes=(), repeats=2, seed=3):
    table = PointTable.from_columns(points)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        table,
        data_axes=tuple(data_axes),
        dtype=np.float64,
        generation="validate-implies-build",
    )
    rng = np.random.default_rng(seed)
    values = rng.normal(size=schema.shape)
    return DatasetSnapshot(schema, values, revision=0)


def _scan(**kwargs):
    return _snapshot(points={"scan": [0.0, 1.0, 2.0, 3.0]}, **kwargs)


def _text_scan(**kwargs):
    return _snapshot(points={"label": ["a", "b", "c", "d"]}, **kwargs)


def _camera(**kwargs):
    from zlc_data import SPATIAL_X, SPATIAL_Y

    return _snapshot(
        points={"frame": [0.0, 1.0, 2.0]},
        data_axes=(
            Axis.create("sy", values=[0.0, 1.0, 2.0], role=SPATIAL_Y),
            Axis.create("sx", values=[0.0, 1.0, 2.0, 3.0], role=SPATIAL_X),
        ),
        **kwargs,
    )


def _site_scan(**kwargs):
    return _snapshot(
        points={"scan": [0.0, 1.0, 2.0, 3.0]},
        data_axes=(Axis.create("site", size=3),),
        **kwargs,
    )


def _validate(view: DataView, spec) -> None:
    """Run the projection-admission authority for one spec, and only that."""

    if isinstance(spec, CurvePlot):
        view.validate_curve(
            spec.x, group_by=() if spec.group is None else (spec.group,)
        )
    elif isinstance(spec, ImagePlot):
        view.validate_image(spec.x, spec.y)
    elif isinstance(spec, FacetGridPlot):
        view.validate_facet(spec)
    elif isinstance(spec, RollingPlot):
        view.validate_rolling(spec.group)
    elif isinstance(spec, HistogramPlot):
        pass
    else:  # pragma: no cover - the parametrisation below is exhaustive
        raise AssertionError(f"unhandled spec: {spec!r}")


def _project(view: DataView, spec) -> None:
    """Run the corresponding build, which must agree with ``_validate``."""

    if isinstance(spec, CurvePlot):
        view.curve(
            spec.x,
            group_by=() if spec.group is None else (spec.group,),
            aggregation=spec.reduction,
        )
    elif isinstance(spec, ImagePlot):
        view.image(spec.x, spec.y, aggregation=spec.reduction)
    elif isinstance(spec, FacetGridPlot):
        view.facet(
            spec,
            bins=_BINS if isinstance(spec.cell, HistogramPlot) else None,
        )
    elif isinstance(spec, RollingPlot):
        view.rolling_history(group=spec.group, aggregation=spec.reduction)
    elif isinstance(spec, HistogramPlot):
        view.histogram(bins=_BINS)
    else:  # pragma: no cover
        raise AssertionError(f"unhandled spec: {spec!r}")


#: (id, snapshot factory, spec).  Deliberately spans every kind, both
#: outcomes, and the three axis domains: R (repeat), P (scan points, frame
#: points) and D (dense data axes, camera pixels).
CASES = [
    ("curve-on-scan", _scan, CurvePlot(AxisRef.point("scan"))),
    # P pooled away entirely: allowed, because pooling is one rule.
    ("curve-x-repeat-pools-the-scan", _scan, CurvePlot(AxisRef.repeat())),
    (
        "curve-grouped-by-a-data-axis",
        _site_scan,
        CurvePlot(AxisRef.point("scan"), group=AxisRef.data("site")),
    ),
    # A text coordinate cannot be plotted as a number -- by BOTH authorities.
    ("curve-on-text-x", _text_scan, CurvePlot(AxisRef.point("label"))),
    (
        "curve-grouped-by-text",
        _text_scan,
        CurvePlot(AxisRef.repeat(), group=AxisRef.point("label")),
    ),
    ("curve-on-an-undeclared-axis", _scan, CurvePlot(AxisRef.point("nope"))),
    (
        "image-on-camera-pixels",
        _camera,
        ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
    ),
    (
        "image-with-a-text-axis",
        _text_scan,
        ImagePlot(AxisRef.point("label"), AxisRef.repeat()),
    ),
    (
        "image-on-two-plain-point-columns",
        lambda: _snapshot(points={"a": [0.0, 1.0], "b": [0.0, 1.0]}),
        ImagePlot(AxisRef.point("a"), AxisRef.point("b")),
    ),
    (
        "facet-curve-cell-pooling-the-scan",
        _site_scan,
        FacetGridPlot(AxisRef.repeat(), CurvePlot(AxisRef.data("site"))),
    ),
    (
        "facet-image-cell-over-frames",
        _camera,
        FacetGridPlot(
            AxisRef.point("frame"),
            ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
        ),
    ),
    (
        "facet-curve-cell-on-text-x",
        _text_scan,
        FacetGridPlot(AxisRef.repeat(), CurvePlot(AxisRef.point("label"))),
    ),
    (
        "facet-histogram-cell",
        _scan,
        FacetGridPlot(AxisRef.repeat(), HistogramPlot()),
    ),
    ("rolling-ungrouped", _scan, RollingPlot()),
    ("rolling-grouped-by-repeat", _scan, RollingPlot(group=AxisRef.repeat())),
    ("histogram-pools-the-box", _camera, HistogramPlot()),
]


@pytest.mark.parametrize(
    "make_snapshot, spec",
    [(factory, spec) for _, factory, spec in CASES],
    ids=[case_id for case_id, _, _ in CASES],
)
def test_passing_validate_implies_the_projection_builds(make_snapshot, spec) -> None:
    snapshot = make_snapshot()
    view = DataView(snapshot)

    admitted = True
    try:
        _validate(view, spec)
    except (DataViewError, TypeError, ValueError):
        admitted = False

    if admitted:
        _project(view, spec)  # must not raise: admission IS buildability
        return
    with pytest.raises((DataViewError, TypeError, ValueError)):
        _project(view, spec)


@pytest.mark.parametrize(
    "make_snapshot, spec",
    [(factory, spec) for _, factory, spec in CASES],
    ids=[case_id for case_id, _, _ in CASES],
)
def test_passing_validate_implies_a_session_constructs(make_snapshot, spec) -> None:
    """The same contract one level up: the session builds and renders."""

    snapshot = make_snapshot()
    admitted = True
    try:
        _validate(DataView(snapshot), spec)
    except (DataViewError, TypeError, ValueError):
        admitted = False

    if not admitted:
        with pytest.raises((DataViewError, TypeError, ValueError)):
            PlotSession(make_snapshot(), spec).close()
        return
    session = PlotSession(snapshot, spec)
    try:
        assert session.rgba().size
    finally:
        session.close()


def test_a_text_point_column_is_refused_by_validation_not_by_the_draw() -> None:
    """The concrete regression, stated once without the parametrisation."""

    view = DataView(_text_scan())
    with pytest.raises(DataViewError, match="real numeric"):
        view.validate_curve(AxisRef.point("label"))
    with pytest.raises(DataViewError, match="real numeric"):
        view.validate_image(AxisRef.point("label"), AxisRef.repeat())
    with pytest.raises(DataViewError, match="real numeric"):
        view.validate_facet(
            FacetGridPlot(AxisRef.repeat(), CurvePlot(AxisRef.point("label")))
        )


def test_reduction_is_honoured_when_the_whole_scan_is_pooled() -> None:
    """Pooling P is a REDUCTION, not a fallback: the authored one applies."""

    snapshot = _scan()
    values = np.asarray(snapshot.block.values)
    view = DataView(snapshot)
    mean = view.curve(AxisRef.repeat(), aggregation=Reduction.MEAN)
    smallest = view.curve(AxisRef.repeat(), aggregation=Reduction.MIN)
    np.testing.assert_allclose(
        np.asarray(mean.series[0].y.canonical), values.mean(axis=1).ravel()
    )
    np.testing.assert_allclose(
        np.asarray(smallest.series[0].y.canonical),
        np.min(values.reshape(values.shape[0], -1), axis=1),
    )
