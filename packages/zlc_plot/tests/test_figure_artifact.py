from __future__ import annotations

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
from zlc_plot.selectors import NumericRange, RectangleRange


@pytest.mark.parametrize(
    "spec",
    (
        CurvePlot(AxisRef.point("x"), group=AxisRef.data("component")),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        HistogramPlot(PlotLabels(title="distribution")),
        RollingPlot(group=AxisRef.data("site")),
        FacetGridPlot(
            AxisRef.data("site"),
            HistogramPlot(PlotLabels(x="signal", y="count")),
        ),
    ),
)
def test_plot_spec_recipe_round_trip_is_exact(spec) -> None:
    document = encode_plot_recipe(spec, parameters={}, size="2x2")
    assert decode_plot_recipe(document)["spec"] == spec


def test_plot_recipe_round_trip_keeps_view_and_rejects_unknown_fields() -> None:
    viewport = RectangleRange(NumericRange(1.0, 2.0), NumericRange(3.0, 4.0))
    document = encode_plot_recipe(
        CurvePlot(AxisRef.point("x")),
        parameters={"show_grid": True},
        size="4x4",
        viewport=viewport,
    )
    decoded = decode_plot_recipe(document)
    assert decoded["viewport"] == viewport
    assert decoded["parameters"]["show_grid"] is True
    assert set(decoded["parameters"]) > {"show_grid"}

    with pytest.raises(ValueError, match="plot recipe fields differ"):
        decode_plot_recipe({**document, "unexpected": True})
