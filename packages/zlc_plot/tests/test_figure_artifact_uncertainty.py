"""Figure archives round-trip the uncertainty display parameters.

The band and the cumulative trace are panel parameters, so they travel in
the recipe's parameters block: a figure saved with the band on reopens with
it on, and archives from before the parameters existed complete to off.
"""
from __future__ import annotations

from zlc_plot import AxisRef, CurvePlot, RollingPlot
from zlc_plot.figure_artifact import decode_plot_recipe, encode_plot_recipe
from zlc_plot.specs import Reduction, parameter_schema_for
from zlc_plot.config import DEFAULTS


def _round_trip(spec, parameters):
    return decode_plot_recipe(
        encode_plot_recipe(
            spec,
            parameters=parameters,
            size="2x2",
            viewport=None,
            classifier_thresholds=(),
        )
    )


def test_uncertainty_parameter_survives_the_archive() -> None:
    recipe = _round_trip(CurvePlot(AxisRef.point("x")), {"uncertainty": True})
    assert recipe["parameters"]["uncertainty"] is True
    recipe = _round_trip(
        RollingPlot(reduction=Reduction.MEAN), {"trailing": 50}
    )
    assert recipe["parameters"]["trailing"] == 50


def test_archives_complete_absent_parameters_to_the_schema_default() -> None:
    """An absent parameter is completed from the schema, not from a second
    answer written here.  The band's default is ON; ``trailing`` has no
    boolean off state, so its default is the span that averages nothing."""

    recipe = _round_trip(CurvePlot(AxisRef.point("x")), {})
    assert recipe["parameters"]["uncertainty"] is True
    recipe = _round_trip(RollingPlot(), {})
    assert recipe["parameters"]["trailing"] == 1


def test_schema_declares_the_switches() -> None:
    """The panel contract itself offers them, and the band is ON.

    A mean shown without its spread is a number presented as if it were
    exact.  Where the statistic does not exist the switch is inert, so the
    default never draws a band that is not there.  A trailing span of one
    averages nothing, which is the off state a count-valued parameter has
    instead of a False."""

    curve_defaults = dict(
        parameter_schema_for(
            CurvePlot(AxisRef.point("x")), style=DEFAULTS.style
        ).initial_values({})
    )
    assert curve_defaults["uncertainty"] is True
    rolling_defaults = dict(
        parameter_schema_for(RollingPlot(), style=DEFAULTS.style).initial_values({})
    )
    assert rolling_defaults["trailing"] == 1
