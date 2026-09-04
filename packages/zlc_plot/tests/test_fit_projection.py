from __future__ import annotations

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, REPEAT
from zlc_plot import DEFAULTS, AxisRef, CurvePlot, HistogramPlot
from zlc_plot._fit_projection import FitProjection, FitScope, ProjectionContext
from zlc_plot.selectors import NumericRange, RectangleRange, SelectorKind, SelectorSnapshot, SelectorState
from zlc_plot.specs import parameter_schema_for
from zlc_plot.state import DisplayStateStore
from zlc_plot.fit import FitEngine


def _snapshot() -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": np.arange(5, dtype=np.float64)}),
        dtype=np.float64,
    )
    return make_snapshot(schema, np.arange(5, dtype=np.float64).reshape(1, 5), revision=3)


def _projection(spec, *, selectors=(), viewport=None) -> FitProjection:
    snapshot = _snapshot()
    schema = parameter_schema_for(spec, style=DEFAULTS.style)
    display = DisplayStateStore(schema).state
    projection = FitProjection(
        data=snapshot,
            revision=snapshot.ref.revision.value,
        spec=spec,
        context=ProjectionContext(display, SelectorSnapshot(tuple(selectors)), viewport=viewport),
        unit_registry=None,
        defaults=DEFAULTS,
        histogram_projection=None,
    )
    projection._build_view_and_payload()
    return projection


def test_curve_fit_selection_prefers_area_then_x_range_then_viewport_then_all() -> None:
    spec = CurvePlot(AxisRef.point("x"))
    area = SelectorState(
        SelectorKind.AREA,
        RectangleRange(NumericRange(1, 4), NumericRange(2.5, 4.5)),
    )
    x_range = SelectorState(SelectorKind.X_RANGE, NumericRange(2, 4))
    viewport = RectangleRange(NumericRange(1, 3), NumericRange(-100, 100))
    model = FitEngine().registry.get("gaussian_offset")

    selected = _projection(spec, selectors=(area, x_range), viewport=viewport).fit_selection(model)
    assert selected.scope is FitScope.SELECTOR
    assert selected.selector_kind is SelectorKind.AREA
    # x in [1, 4] -- all four of them.  A box restricts the COORDINATE; the two
    # samples whose observation lies outside its vertical extent are still part
    # of the curve being fitted, and dropping them for their VALUE is how a box
    # that did not reach over the peak deleted the peak from the fit.
    assert selected.sample_count == 4

    selected = _projection(spec, selectors=(x_range,), viewport=viewport).fit_selection(model)
    assert selected.selector_kind is SelectorKind.X_RANGE
    assert selected.sample_count == 3

    selected = _projection(spec, viewport=viewport).fit_selection(model)
    assert selected.scope is FitScope.VIEWPORT
    assert selected.sample_count == 3

    selected = _projection(spec).fit_selection(model)
    assert selected.scope is FitScope.ALL
    assert selected.sample_count == 5


def test_histogram_fit_uses_painted_count_bins_only() -> None:
    spec = HistogramPlot()
    projection = _projection(spec)
    model = FitEngine().registry.get("histogram_gaussian")
    selection = projection.fit_selection(model)
    assert selection.coordinates[0].ndim == 1
    assert selection.observations.ndim == 1
    assert selection.selected_indices is not None
    assert selection.sample_count == selection.observations.size
    density_projection = projection._with_context(
        ProjectionContext(
            DisplayStateStore(
                parameter_schema_for(spec, style=DEFAULTS.style),
                {"density": True},
            ).state,
            SelectorSnapshot(()),
        )
    )
    density_projection._build_view_and_payload()
    with pytest.raises(ValueError, match="density=False"):
        density_projection.fit_selection(model)


def _pulse_timeline_data():
    """A kind with no run behind it: authored, not acquired."""

    from zlc_plot.primitives import PulseBlock, PulseChannel, PulseTimelineData

    data = PulseTimelineData(
        channels=(PulseChannel("laser", "Laser"), PulseChannel("probe", "Probe")),
        blocks=(
            PulseBlock("laser", 0.0, 4.0e-6, label="Init"),
            PulseBlock("probe", 4.0e-6, 8.0e-6, label="Read"),
        ),
        time_unit="s",
        total_duration=10.0e-6,
    )
    return data


def _pulse_timeline_session():
    from zlc_plot import PlotLabels
    from zlc_plot.api import pulse_timeline

    return pulse_timeline(_pulse_timeline_data(), labels=PlotLabels(title="Pulse"))


def test_a_kind_with_no_run_answers_none_rather_than_refusing() -> None:
    """"Which dataset is this frame from" has ONE answer, and None is legal.

    A pulse timeline is authored, not acquired: it has a revision but no run
    behind it.  The projection used to REFUSE the question, while the session
    answered None and the front's own field is ``str | None`` -- so every
    caller that knew the kind might have no generation wrote the refusal off
    as an absent attribute, ``getattr(projection, "data_generation", None)``,
    which is not what a raising property does.
    """

    session = _pulse_timeline_session()
    assert session.data_generation is None
    # The projection owns the data, so it owns the answer; the session says
    # the same thing because it asks the projection.
    assert session._projection.data_generation is None


def test_resizing_a_pulse_timeline_keeps_drawing_it() -> None:
    """The Pulse Editor's Size control, at the mechanism it actually drives.

    ``set_size`` then ``update_data`` is what the preview does, and it goes
    through ``commit_live_frame`` -- the one line that tolerated a kind with
    no generation, and could not.  The operator saw "cannot draw this pulse"
    and a half-painted canvas.
    """

    from zlc_plot import RasterPlotHost

    session = _pulse_timeline_session()
    host = RasterPlotHost.from_session(session)
    try:
        first = host.wait_for_front(timeout=5.0)
        assert first is not None
        for size in ("4x4", "8x8", "1x2"):
            host.set_size(size).result(timeout=5.0)
            host.update_data(_pulse_timeline_data()).result(timeout=5.0)
            front = host.wait_for_front(timeout=5.0)
            assert front is not None, f"no front came back at {size}"
            assert front.identity.data_generation is None
    finally:
        host.close()
