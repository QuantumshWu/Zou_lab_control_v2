"""Figure archives round-trip the uncertainty switches.

A saved figure whose panel drew the standard-error band must reopen with
the band; an archive written before the field existed decodes to off; a
genuinely foreign field is still refused.
"""
from __future__ import annotations

import pytest

from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, RollingPlot
from zlc_plot.figure_artifact import _decode_plot_spec, _encode_plot_spec
from zlc_plot.specs import Reduction


@pytest.mark.parametrize(
    "spec",
    [
        CurvePlot(AxisRef.point("x"), uncertainty=True),
        CurvePlot(AxisRef.point("x")),
        RollingPlot(reduction=Reduction.MEAN, cumulative=True),
        RollingPlot(),
        FacetGridPlot(
            facet=AxisRef.data("site"),
            cell=CurvePlot(AxisRef.point("x"), uncertainty=True),
        ),
    ],
)
def test_specs_round_trip_exactly(spec) -> None:
    assert _decode_plot_spec(_encode_plot_spec(spec)) == spec


def test_archives_from_before_the_band_decode_to_off() -> None:
    document = _encode_plot_spec(CurvePlot(AxisRef.point("x")))
    del document["uncertainty"]
    assert _decode_plot_spec(document).uncertainty is False
    document = _encode_plot_spec(RollingPlot())
    del document["cumulative"]
    assert _decode_plot_spec(document).cumulative is False


def test_foreign_fields_are_still_refused() -> None:
    document = _encode_plot_spec(CurvePlot(AxisRef.point("x")))
    document["mystery"] = 1
    with pytest.raises(ValueError):
        _decode_plot_spec(document)
