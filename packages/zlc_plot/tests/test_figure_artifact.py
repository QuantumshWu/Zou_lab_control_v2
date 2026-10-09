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
    SelectorKind,
    SelectorState,
)


@pytest.mark.parametrize(
    "spec",
    (
        CurvePlot(AxisRef.point("x"), group=AxisRef.cell_data("component")),
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
        HistogramPlot(group=AxisRef.cell_data("site"), labels=PlotLabels(title="distribution")),
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
    if isinstance(spec, HistogramPlot):
        document["spec"].pop("group")
        assert decode_plot_recipe(document)["spec"] == replace(spec, group=None)
        document["spec"]["unrecognized"] = None
        with pytest.raises(ValueError, match="histogram recipe fields differ"):
            decode_plot_recipe(document)


@pytest.mark.parametrize("viewport", (
    (NumericRange(1.0, 2.0), NumericRange(3.0, 4.0)),
    (NumericRange(1.0, 2.0), None),
    (None, NumericRange(3.0, 4.0)),
    (None, None),
    None,
))
def test_plot_recipe_round_trip_keeps_view_and_rejects_unknown_fields(viewport) -> None:
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
    assert decoded["viewport"] == (None if viewport == (None, None) else viewport)
    assert decoded["selectors"] == selectors
    assert decoded["parameters"]["show_grid"] is True
    assert set(decoded["parameters"]) > {"show_grid"}

    with pytest.raises(ValueError, match="plot recipe fields differ"):
        decode_plot_recipe({**document, "unexpected": True})


def test_the_uncertainty_switches_survive_the_archive_and_default_from_the_schema() -> None:
    """The band and the trailing span are panel parameters: they travel in
    the recipe's parameters block, and a recipe that omits one is completed
    from the parameter schema's own default by the writer, never by a second
    default kept here.

    The band is ON by default: a mean shown without its spread is a number
    presented as if it were exact.  A trailing span of one averages nothing,
    which is the off state a count-valued parameter has instead of a False.
    """

    from zlc_plot.config import DEFAULTS
    from zlc_plot.specs import Reduction, parameter_schema_for

    def declared(spec) -> dict:
        return dict(parameter_schema_for(spec, style=DEFAULTS.style).initial_values({}))

    def saved(spec, parameters) -> dict:
        document = encode_plot_recipe(spec, parameters=parameters, size="2x2")
        return decode_plot_recipe(document)["parameters"]

    curve = CurvePlot(AxisRef.point("x"))
    rolling = RollingPlot(reduction=Reduction.MEAN)
    assert declared(curve)["uncertainty"] is True
    assert declared(rolling)["trailing"] == 1
    for spec, name, authored in ((curve, "uncertainty", False), (rolling, "trailing", 50)):
        assert saved(spec, {})[name] == declared(spec)[name]
        assert saved(spec, {name: authored})[name] == authored


def test_saved_value_name_is_used_when_the_figure_is_redrawn(tmp_path) -> None:
    from data_factory import make_dataset_schema, make_snapshot, mapped_domain_from_columns, repeat_domain
    from zlc_data.figure_archive import read_archive
    from zlc_plot import PlotSession, read_figure_plot, save_figure_artifact

    schema = make_dataset_schema(repeat_domain(size=1), mapped_domain_from_columns({"x": [0, 1, 2]}))
    schema = replace(schema, value_schema=replace(schema.value_schema, name="Survival"))
    snapshot = make_snapshot(schema, np.array([[0.9, 0.8, 0.7]]), 1)
    viewport = (NumericRange(0.5, 1.5), None)
    image, archive = save_figure_artifact(
        tmp_path / "survival", plot_input=snapshot, spec=CurvePlot(AxisRef.point("x")),
        parameters={}, size="2x2", viewport=viewport,
    )
    info, arrays, datasets = read_archive(archive)
    restored, recipe = read_figure_plot(info, arrays, datasets, "data")
    assert image.is_file()
    assert restored.block.schema.value_schema.name == "Survival"
    np.testing.assert_array_equal(restored.block.values, snapshot.block.values)
    session = PlotSession(
        restored, recipe["spec"], parameters=recipe["parameters"], size=recipe["size"],
        initial_configuration={"viewport": recipe["viewport"]},
    )
    try:
        assert session.viewport == viewport
        assert session.describe_display().limits.x == viewport[0]
        assert session._renderer.primary_axes.get_ylabel() == "Survival"
        from PIL import Image

        session.save(tmp_path / "fast.png", restore_display=False)
        session.save(tmp_path / "compressed.png", restore_display=False,
                     pil_kwargs={"compress_level": 6})
        with Image.open(tmp_path / "fast.png") as fast, Image.open(tmp_path / "compressed.png") as compressed:
            np.testing.assert_array_equal(np.asarray(fast), np.asarray(compressed))
    finally:
        session.close()
