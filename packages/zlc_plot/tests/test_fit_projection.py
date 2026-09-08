from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, REPEAT, DomainSpec, LATEST_COORDINATE
from zlc_plot import DEFAULTS, AxisRef, CurvePlot, HistogramPlot, ImagePlot, FacetGridPlot, RollingPlot, Reduction
from zlc_plot._fit_projection import FitProjection, FitScope, ProjectionContext
from zlc_plot.data_contract import DEFAULT_UNITS
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


def _projection(
    spec, *, selectors=(), viewport=None, snapshot=None, display=None
) -> FitProjection:
    snapshot = _snapshot() if snapshot is None else snapshot
    schema = parameter_schema_for(spec, style=DEFAULTS.style)
    display = DisplayStateStore(schema, display).state
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


def test_release_recapture_units_and_fixed_expression_use_the_series_contract() -> None:
    from scipy.special import lambertw
    from zlc_plot import PlotSession

    t = np.linspace(0.0, 200e-6, 64)
    q = np.exp(-lambertw((2 * np.pi * 40_000 * t) ** 2).real)
    y = -np.expm1(-6.0 * q) / -np.expm1(-6.0)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"time": t}, units={"time": "s"}),
        value_unit="1",
    )
    session = PlotSession(
        make_snapshot(schema, y.reshape(1, -1), revision=1),
        CurvePlot(AxisRef.point("time")),
    )
    try:
        session.set_parameter("x_display_unit", "us")
        session.configure(fit={
            "model": "release_recapture",
            "expression": "A=1, B=0, eta=guess(5), f=guess(0.04)",
        })
        description = session.describe_display()
        assert description.fit["fixed"] == {"amplitude": 1.0, "offset": 0.0}
        assert description.fit["initial"]["frequency"] == pytest.approx(40_000.0)
        model = FitEngine().registry.get("release_recapture")
        assert session._projected._fit_parameter_units(model)["eta"] == ""
        assert session._projected._fit_parameter_units(model)["frequency"] == "Hz"
        assert session._projected._display_fit_parameter_value(model.parameters[3], 40_000.0)[0] == pytest.approx(0.04)
        assert session.rgba().size > 0
    finally:
        session.close()


@pytest.mark.parametrize("x_unit, x_display, y_unit, y_display, factor", (
    ("mVpp", "Vpp", "count", "count", .001),
    ("s", "ms", "count", "count", 1000.0),
    ("s", "ms", "V", "mV", 1e6),
))
def test_product_fit_parameter_keeps_units_sign_and_expression_roundtrip(
    x_unit, x_display, y_unit, y_display, factor,
) -> None:
    model = FitEngine().registry.get("saturation")
    parameter = next(item for item in model.parameters if item.name == "numerator")
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": np.arange(1.0, 5.0)}, units={"x": x_unit}),
        value_unit=y_unit,
    )
    snapshot = make_snapshot(schema, np.ones((1, 4)), revision=0)
    projection = _projection(CurvePlot(AxisRef.point("x")), snapshot=snapshot,
        display={"x_display_unit": x_display, "value_display_unit": y_display})
    canonical_unit = projection._fit_parameter_units(model)["numerator"]
    assert canonical_unit == f"{y_unit}*{x_unit}"
    value, unit = projection._display_fit_parameter_value(parameter, -3.0)
    assert value == pytest.approx(-3.0 * factor)
    assert unit == f"{y_display}*{x_display}"
    error, error_unit = projection._display_fit_parameter_value(parameter, .5, difference=True)
    assert error == pytest.approx(.5 * factor) and error_unit == unit
    target = projection.fit_expression_target(model, f"B={value!r}")
    assert target["fixed"]["numerator"] == pytest.approx(-3.0)
    assert float(projection.fit_expression_text(model, target).split("=")[1]) == pytest.approx(value)
    # Published Fit vectors store this exact canonical unit string; consuming
    # that Dataset uses the ordinary unit resolver, not a fit-only catalog.
    published_schema = replace(schema, value_schema=replace(schema.value_schema, value_unit=canonical_unit))
    published = make_snapshot(published_schema, np.full((1, 4), -3.0), revision=0)
    downstream = _projection(CurvePlot(AxisRef.point("x")), snapshot=published)
    assert downstream._value_quantity().canonical_unit == DEFAULT_UNITS.resolve(canonical_unit)


def _dbm_curve_snapshot() -> OwnedSnapshot:
    x = np.linspace(-6.0, 6.0, 9)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": x}, units={"x": "dBm"}),
        dtype=np.float64,
    )
    return make_snapshot(schema, np.exp(-x * x / 8.0).reshape(1, -1), revision=0)


def test_fit_expression_crosses_a_logarithmic_axis_as_the_unit_registry_does() -> None:
    """A centre typed in watts on a dBm axis is the power it names.

    dBm is a level: no scale turns it into watts.  Reading the crossing off
    two converted points treated it as affine, so ``x_0=0.002`` (2 mW) was
    fixed at 3.86 dBm -- 2.43 mW -- inside every bound and without a word,
    and read back as 0.00243.  A width on that axis crosses unchanged in the
    axis' own unit and has no value in watts at all, which is said aloud
    rather than guessed.
    """

    spec = CurvePlot(AxisRef.point("x"))
    snapshot = _dbm_curve_snapshot()
    model = FitEngine().registry.get("gaussian_offset")

    watts = _projection(spec, snapshot=snapshot, display={"x_display_unit": "W"})
    target = watts.fit_expression_target(model, "x_0=0.002")
    assert target["fixed"]["center"] == pytest.approx(
        float(DEFAULT_UNITS.convert(0.002, "W", "dBm"))
    )
    symbol, literal = watts.fit_expression_text(model, target).split("=")
    assert symbol == "x_0" and float(literal) == pytest.approx(0.002, rel=1e-12)
    with pytest.raises(ValueError, match="only a position crosses a logarithmic unit"):
        watts.fit_expression_target(model, "sigma=1.5")

    own = _projection(spec, snapshot=snapshot)
    assert own.fit_expression_target(model, "sigma=1.5") == {
        "model": model.model_id,
        "fixed": {"sigma": 1.5},
    }
    assert own.fit_expression_text(model, {"fixed": {"sigma": 1.5}}) == "sigma=1.5"


def test_fit_selection_seals_the_sigma_plane_with_the_others() -> None:
    """Every plane an accepted fit replays to its subscribers is read-only.

    Coordinates, observations and indices were sealed; the sigma plane came
    out of its advanced index writable -- the one plane through which what
    the solver had weighted by could be rewritten after the fact.
    """

    scan = axis("scan", values=[10.0, 20.0, 30.0])
    schema = make_dataset_schema(
        repeat_domain(size=6),
        mapped_domain_from_columns({"x": [0.0, 1.0]}),
        cell_axes=(scan,),
        dtype=np.float64,
    )
    values = np.random.default_rng(5).normal(size=schema.physical_shape)
    projection = _projection(
        CurvePlot(AxisRef.cell_data("scan")),
        snapshot=make_snapshot(schema, values, revision=0),
        display={"uncertainty": True},
    )
    selection = projection.fit_selection(FitEngine().registry.get("gaussian_offset"))
    assert selection.observation_sigma is not None
    planes = (
        *selection.coordinates,
        selection.observations,
        selection.selected_indices,
        selection.observation_sigma,
    )
    assert not any(plane.flags.writeable for plane in planes)


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


@pytest.mark.parametrize("kind", ("curve", "image", "histogram", "facet_curve", "facet_image", "facet_histogram", "rolling"))
@pytest.mark.parametrize("hole", (False, True))
def test_last_reduction_replays_the_same_scope_for_payload_and_selection(kind, hole) -> None:
    repeat, slow, x, y = (AxisRef.repeat("repeat"), AxisRef.point("slow"),
                           AxisRef.cell_data("x"), AxisRef.cell_data("y"))
    # Declared descending order; the last coordinate is physical row 1,
    # not the largest coordinate or the last row that arrived.
    schema = make_dataset_schema(
        repeat_domain(size=3),
        DomainSpec((3,), (axis("slow", values=(9.0, 5.0, 1.0)),), ((0, 2, 1),)),
        cell_axes=(axis("x", size=2), axis("y", size=2)),
    )
    values = np.arange(36.0).reshape(schema.physical_shape)
    valid = np.ones(values.shape, dtype=bool)
    valid[-1, 1] = not hole
    snapshot = make_snapshot(schema, values, revision=7, validity=valid, sigma=np.full(values.shape, 0.25))
    specs = {
        "curve": (CurvePlot(x, group=y, reduction=Reduction.LAST), (repeat, slow)),
        "image": (ImagePlot(x, y, reduction=Reduction.LAST), (repeat, slow)),
        "histogram": (HistogramPlot(reduced=(repeat, slow), reduction=Reduction.LAST), (repeat, slow)),
        "facet_curve": (FacetGridPlot(facet=x, cell=CurvePlot(y, reduction=Reduction.LAST)), (repeat, slow)),
        "facet_image": (FacetGridPlot(facet=slow, cell=ImagePlot(x, y, reduction=Reduction.LAST)), (repeat,)),
        "facet_histogram": (FacetGridPlot(facet=x, cell=HistogramPlot(reduced=(repeat, slow), reduction=Reduction.LAST)), (repeat, slow)),
        "rolling": (RollingPlot(group=x, reduction=Reduction.LAST), (slow, y)),
    }
    spec, reduced = specs[kind]
    scope = tuple((ref, LATEST_COORDINATE) for ref in reduced)
    expected_spec = (
        replace(spec, cell=replace(spec.cell, reduction=Reduction.MEAN,
                                  **({"reduced": ()} if isinstance(spec.cell, HistogramPlot) else {})), scope=scope)
        if isinstance(spec, FacetGridPlot) else replace(spec, reduction=Reduction.MEAN, scope=scope,
                                                      **({"reduced": ()} if isinstance(spec, HistogramPlot) else {}))
    )
    actual = _projection(spec, snapshot=snapshot, display={"uncertainty": True} if kind in {"curve", "facet_curve", "rolling"} else None)
    expected = _projection(expected_spec, snapshot=snapshot, display={"uncertainty": True} if kind in {"curve", "facet_curve", "rolling"} else None)
    assert actual.spec == spec
    assert actual._view._schema == expected._view._schema
    for left, right in ((actual._view.samples.value.canonical, expected._view.samples.value.canonical),
                        (actual._view.samples.valid_mask, expected._view.samples.valid_mask),
                        (actual._view.samples.sigma, expected._view.samples.sigma)):
        np.testing.assert_array_equal(left, right)
    assert actual._view.selection_subject(spec, actual.payload).scope == expected._view.selection_subject(expected_spec, expected.payload).scope
    from zlc_plot.semantics import describe_semantics
    from zlc_plot.figure_artifact import encode_plot_recipe, decode_plot_recipe
    assert Reduction.LAST in describe_semantics(schema, spec).field("reduction").choice_values
    assert decode_plot_recipe(encode_plot_recipe(spec, parameters={}, size="2x2"))["spec"] == spec
    if kind == "curve" and not hole:
        selected = actual.fit_selection(FitEngine().registry.get("gaussian_offset"))
        np.testing.assert_allclose(selected.observation_sigma, 0.25)
    actual_cells = tuple(cell.payload for cell in actual.payload.cells) if isinstance(spec, FacetGridPlot) else (actual.payload,)
    expected_cells = tuple(cell.payload for cell in expected.payload.cells) if isinstance(spec, FacetGridPlot) else (expected.payload,)
    for left, right in zip(actual_cells, expected_cells, strict=True):
        if hasattr(left, "series"):
            for a, b in zip(left.series, right.series, strict=True):
                np.testing.assert_array_equal(a.valid, b.valid)
                np.testing.assert_array_equal(a.counts, b.counts)
                np.testing.assert_array_equal(a.sem, b.sem)
                np.testing.assert_allclose(np.where(a.valid, a.y.canonical, np.nan), np.where(b.valid, b.y.canonical, np.nan))
        elif hasattr(left, "z"):
            np.testing.assert_array_equal(left.valid, right.valid)
            np.testing.assert_array_equal(left.z.canonical, right.z.canonical)
        else:
            np.testing.assert_array_equal(left.edges.canonical, right.edges.canonical)
            np.testing.assert_array_equal(left.counts, right.counts)


def test_last_on_a_missing_sparse_end_is_the_same_empty_scope() -> None:
    from zlc_data import EmptySelection
    from zlc_plot.data_view import DataView

    schema = make_dataset_schema(
        repeat_domain(size=1),
        DomainSpec((3,), (axis("a", size=2), axis("b", size=2)), ((0, 1, 0), (0, 0, 1))),
        cell_axes=(axis("x", size=2),),
    )
    snapshot = make_snapshot(schema, np.arange(6.0).reshape(schema.physical_shape), revision=0)
    spec = CurvePlot(AxisRef.cell_data("x"), reduction=Reduction.LAST)
    for build in (
        lambda: _projection(spec, snapshot=snapshot),
        lambda: DataView(snapshot).curve(spec.x, aggregation=Reduction.LAST),
        lambda: _projection(replace(spec, reduction=Reduction.MEAN, scope=(
            (AxisRef.point("a"), LATEST_COORDINATE), (AxisRef.point("b"), LATEST_COORDINATE))), snapshot=snapshot),
    ):
        with pytest.raises(EmptySelection):
            build()


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
