from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotLabels,
    RollingPlot,
    decode_plot_recipe,
    encode_plot_recipe,
)
from zlc_plot.selectors import (
    CrosshairPoint,
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorState,
)


@pytest.mark.parametrize(
    "spec",
    (
        CurvePlot(AxisRef.point("x"), group=AxisRef.cell_data("component")),
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
        HistogramPlot(labels=PlotLabels(title="distribution")),
        RollingPlot(group=AxisRef.cell_data("site")),
        FacetGridPlot(
            AxisRef.cell_data("site"),
            HistogramPlot(labels=PlotLabels(x="signal", y="count")),
        ),
    ),
)
def test_plot_spec_recipe_round_trip_is_exact(spec) -> None:
    document = encode_plot_recipe(spec, parameters={}, size="2x2")
    assert decode_plot_recipe(document)["spec"] == spec


def test_plot_recipe_round_trip_keeps_view_and_rejects_unknown_fields() -> None:
    viewport = RectangleRange(NumericRange(1.0, 2.0), NumericRange(3.0, 4.0))
    selectors = (
        SelectorState(SelectorKind.CROSSHAIR, CrosshairPoint(1.5, 3.5)),
    )
    document = encode_plot_recipe(
        CurvePlot(AxisRef.point("x")),
        parameters={"show_grid": True},
        size="4x4",
        viewport=viewport,
        selectors=selectors,
    )
    decoded = decode_plot_recipe(document)
    assert decoded["viewport"] == viewport
    assert decoded["selectors"] == selectors
    assert decoded["parameters"]["show_grid"] is True
    assert set(decoded["parameters"]) > {"show_grid"}

    with pytest.raises(ValueError, match="plot recipe fields differ"):
        decode_plot_recipe({**document, "unexpected": True})


def test_saved_value_name_is_used_when_the_figure_is_redrawn(tmp_path) -> None:
    from data_factory import make_dataset_schema, make_snapshot, mapped_domain_from_columns, repeat_domain
    from zlc_data.figure_archive import read_archive
    from zlc_plot import PlotSession, read_figure_plot, save_figure_artifact

    schema = make_dataset_schema(repeat_domain(size=1), mapped_domain_from_columns({"x": [0, 1, 2]}))
    schema = replace(schema, value_schema=replace(schema.value_schema, name="Survival"))
    snapshot = make_snapshot(schema, np.array([[0.9, 0.8, 0.7]]), 1)
    image, archive = save_figure_artifact(
        tmp_path / "survival", plot_input=snapshot, spec=CurvePlot(AxisRef.point("x")),
        parameters={}, size="2x2",
    )
    info, arrays, datasets = read_archive(archive)
    restored, recipe = read_figure_plot(info, arrays, datasets, "data")
    assert image.is_file()
    assert restored.block.schema.value_schema.name == "Survival"
    np.testing.assert_array_equal(restored.block.values, snapshot.block.values)
    session = PlotSession(restored, recipe["spec"], parameters=recipe["parameters"], size=recipe["size"])
    try:
        assert session._renderer.primary_axes.get_ylabel() == "Survival"
    finally:
        session.close()
