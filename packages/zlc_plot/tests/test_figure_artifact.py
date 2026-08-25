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
    selectors = (
        SelectorState(SelectorKind.CROSSHAIR, CrosshairPoint(1.5, 3.5)),
    )
    document = encode_plot_recipe(
        CurvePlot(AxisRef.point("x")),
        parameters={"show_grid": True},
        size="4x4",
        viewport=viewport,
        selectors=selectors,
        fit={
            "model": "exponential_decay",
            "fixed": {"offset": 0.0},
            "initial": {"decay_time": 2.0},
        },
    )
    decoded = decode_plot_recipe(document)
    assert decoded["viewport"] == viewport
    assert decoded["selectors"] == selectors
    assert decoded["parameters"]["show_grid"] is True
    assert decoded["fit"] == {
        "model": "exponential_decay",
        "fixed": {"offset": 0.0},
        "initial": {"decay_time": 2.0},
    }
    assert set(decoded["parameters"]) > {"show_grid"}

    with pytest.raises(ValueError, match="plot recipe fields differ"):
        decode_plot_recipe({**document, "unexpected": True})
    with pytest.raises(ValueError, match="unknown canonical fit fields"):
        encode_plot_recipe(
            CurvePlot(AxisRef.point("x")),
            parameters={},
            size="2x2",
            fit={"model": "exponential_decay", "expression": "offset=0"},
        )
    invalid_fit = {
        **document,
        "fit": {
            "model": "exponential_decay",
            "expression": "offset=0",
        },
    }
    with pytest.raises(ValueError, match="unknown canonical fit fields"):
        decode_plot_recipe(invalid_fit)
