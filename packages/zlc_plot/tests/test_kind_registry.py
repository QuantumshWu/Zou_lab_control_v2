from __future__ import annotations

import numpy as np
import pytest

from zlc_plot._kinds import HANDLERS, handler_for
from zlc_plot.kinds import PlotKind
from zlc_plot.specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PulseTimelinePlot,
    RollingPlot,
)
from zlc_plot import AxisRef
from data_factory import Axis, DatasetSchema, PointTable, PointTopology


def _schema_families() -> tuple[DatasetSchema, ...]:
    point = PointTable.from_columns({"x": np.arange(4.0)})
    topology_points = PointTable.from_columns(
        {
            "x": np.repeat(np.arange(2.0), 3),
            "y": np.tile(np.arange(3.0), 2),
        }
    )
    topology = PointTopology.from_cartesian(
        (
            Axis.create("x", values=np.arange(2.0)),
            Axis.create("y", values=np.arange(3.0)),
        ),
        point_table=topology_points,
    )
    return (
        DatasetSchema.create(
            Axis.create("repeat", size=2), point, generation="kind-default-point"
        ),
        DatasetSchema.create(
            Axis.create("repeat", size=2),
            topology_points,
            point_topology=topology,
            generation="kind-default-topology",
        ),
        DatasetSchema.create(
            Axis.create("repeat", size=2),
            PointTable.from_columns({"sample": np.array([0.0])}),
            data_axes=(
                Axis.create("row", size=3),
                Axis.create("column", size=4),
            ),
            generation="kind-default-dense-image",
        ),
        # A column an operator reads as "site" that a resolver finds under a
        # different id.  Every other family here has the two equal, which is
        # why "looked it up by the wrong one" was invisible for so long.
        DatasetSchema.create(
            Axis.create("repeat", size=2),
            PointTable.from_columns(
                {"site": np.arange(6.0)}, ids={"site": "readout.site.v1"}
            ),
            generation="kind-default-renamed-point",
        ),
    )


def test_kind_registry_is_closed_and_complete() -> None:
    assert tuple(handler.kind for handler in HANDLERS) == tuple(PlotKind)
    assert len({handler.spec_type for handler in HANDLERS}) == len(HANDLERS)
    from zlc_plot.fit import FitTarget

    for handler in HANDLERS:
        # fit_target is either None (the kind has no fittable projection:
        # FacetGrid delegates to its cell, PulseTimeline has none) or a
        # member of the FitTarget contract; anything else would blow up the
        # first fit_models() call at runtime.
        if handler.fit_target is not None:
            assert FitTarget(handler.fit_target)
        else:
            assert handler.kind in {PlotKind.FACET_GRID, PlotKind.PULSE_TIMELINE}
        assert handler.spec_type.kind is handler.kind
        assert handler.display_name
        assert callable(handler.render)
        assert callable(handler.build_payload)
        assert handler.semantic_fields
        assert callable(handler.admits)


def test_registry_semantic_metadata_is_the_only_extension_point() -> None:
    """A new field is visible mechanically without GUI/per-kind branching."""

    from dataclasses import replace

    curve = next(handler for handler in HANDLERS if handler.kind is PlotKind.CURVE)
    extended = replace(curve, semantic_fields=(*curve.semantic_fields, "future_axis"))
    assert extended.semantic_fields[-1] == "future_axis"


def test_kind_registry_resolves_each_authored_spec_exactly_once() -> None:
    specs = (
        CurvePlot(AxisRef.point("x")),
        ImagePlot(AxisRef.point("x"), AxisRef.point("y")),
        HistogramPlot(),
        RollingPlot(),
        FacetGridPlot(AxisRef.point("facet"), CurvePlot(AxisRef.point("x"))),
        PulseTimelinePlot(),
    )
    handlers = tuple(handler_for(spec) for spec in specs)
    assert tuple(handler.kind for handler in handlers) == tuple(PlotKind)
    assert tuple(handler.spec_type for handler in handlers) == tuple(
        type(spec) for spec in specs
    )


def test_registry_rejects_an_unregistered_kind_without_a_fallback_path() -> None:
    with pytest.raises(TypeError, match="unsupported plot specification"):
        handler_for(object())


def test_a_kind_owned_default_implies_admission_never_the_reverse() -> None:
    """`default_spec` is the narrower question: a usable default requires the
    kind to admit the schema, but a kind may admit a schema (renderable with
    an authored spec) without any unambiguous default to switch to."""

    for index, schema in enumerate(_schema_families()):
        for handler in HANDLERS:
            inferred = handler.default_spec(schema)
            if inferred is not None:
                assert handler.admits(schema)
                assert handler_for(inferred).kind is handler.kind


def test_default_specs_put_the_innermost_scan_loop_on_x() -> None:
    """Axis order convention: dimensions are slowest-first, so the trailing
    dimension is the axis one sweep walks — curve x and image x."""

    _, topology_schema, dense_schema, _renamed = _schema_families()
    curve = next(h for h in HANDLERS if h.kind is PlotKind.CURVE)
    image = next(h for h in HANDLERS if h.kind is PlotKind.IMAGE)
    facet = next(h for h in HANDLERS if h.kind is PlotKind.FACET_GRID)

    inferred_curve = curve.default_spec(topology_schema)
    assert inferred_curve.x == AxisRef.point_dimension("y")

    inferred_image = image.default_spec(topology_schema)
    assert inferred_image.x == AxisRef.point_dimension("y")
    assert inferred_image.y == AxisRef.point_dimension("x")

    dense_image = image.default_spec(dense_schema)
    assert dense_image.x == AxisRef.data("column")
    assert dense_image.y == AxisRef.data("row")

    # FacetGrid on a scalar two-dimension scan has no automatic facet left:
    # the heatmap consumes both scan dimensions, while repeat is acquisition
    # history and may become a facet only through an explicit operator edit.
    inferred_facet = facet.default_spec(topology_schema)
    assert inferred_facet is None


def test_repeat_is_not_an_automatic_facet_but_remains_explicitly_valid() -> None:
    """A default never guesses repeat; an authored repeat facet still works."""

    points = PointTable.from_columns(
        {
            "x": np.repeat(np.arange(2.0), 3),
            "y": np.tile(np.arange(3.0), 2),
        }
    )
    topology = PointTopology.from_cartesian(
        (
            Axis.create("x", values=np.arange(2.0)),
            Axis.create("y", values=np.arange(3.0)),
        ),
        point_table=points,
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        points,
        point_topology=topology,
        generation="facet-default-no-repeat",
    )
    facet = next(h for h in HANDLERS if h.kind is PlotKind.FACET_GRID)
    image = next(h for h in HANDLERS if h.kind is PlotKind.IMAGE)
    curve = next(h for h in HANDLERS if h.kind is PlotKind.CURVE)
    assert facet.admits(schema)
    assert facet.default_spec(schema) is None
    heatmap = image.default_spec(schema)
    assert heatmap.x == AxisRef.point_dimension("y")
    assert heatmap.y == AxisRef.point_dimension("x")
    explicit = FacetGridPlot(AxisRef.repeat(), heatmap)
    from zlc_plot.data_view import DataView
    from data_factory import DatasetSnapshot

    DataView(
        DatasetSnapshot(schema, np.zeros(schema.shape, dtype=np.float64), revision=0)
    ).validate_facet(explicit)

    # A flat point table with a single repeat has nothing to face either: the
    # curve IS the picture, and the grid asked for holds it in one cell.
    flat = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": np.arange(4.0)}),
        generation="facet-default-flat",
    )
    assert facet.admits(flat)
    assert facet.default_spec(flat) is None
    explicit_flat = FacetGridPlot(AxisRef.repeat(), curve.default_spec(flat))
    DataView(
        DatasetSnapshot(flat, np.zeros(flat.shape, dtype=np.float64), revision=0)
    ).validate_facet(explicit_flat)


def test_a_default_spec_names_axes_the_data_view_can_actually_resolve() -> None:
    """A kind's inference must produce a spec that RESOLVES, not merely one that
    type-checks.

    curve's default named its x axis by the column's display NAME while the
    resolver looks the axis up by coordinate ID.  The two are equal often
    enough that it worked everywhere until a producer named a column something
    other than its coordinate -- and then PlotSession raised on construction,
    which a RasterPlotHost turns into a host stuck permanently "closing": the
    console's Add Panel failed with a sentence about the host rather than
    about the axis.
    """

    from data_factory import DatasetSnapshot
    from zlc_plot.data_view import DataView

    unresolved = []
    for index, schema in enumerate(_schema_families()):
        for handler in HANDLERS:
            spec = handler.default_spec(schema)
            if spec is None:
                continue
            snapshot = DatasetSnapshot(
                schema, np.zeros(schema.shape, dtype=np.float64), revision=0
            )
            view = DataView(snapshot)
            # The axes the kind itself says its spec names, resolved the way
            # the projection resolves them.
            for slot, role in handler.label_roles(spec):
                if not (isinstance(role, tuple) and role and role[0] == "axis"):
                    continue
                ref = role[1]
                if ref is None:
                    continue
                try:
                    view._resolve(ref)
                except Exception as error:  # noqa: BLE001 - any failure is it
                    unresolved.append(
                        f"{handler.kind.value}.{slot} on family {index}: "
                        f"{type(error).__name__}: {error}"
                    )
    assert not unresolved, unresolved
