from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, DEFAULT_UNITS, PlotSession, SelectorKind, resolve_unit


def _session() -> PlotSession:
    x = np.linspace(0.0, 3.0, 61)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x}, units={"x": "m"}),
        dtype=np.float64,
        canonical_unit="V",
        generation="unit-test",
    )
    model = PlotSession(
        DatasetSnapshot(schema, np.zeros((1, x.size), dtype=np.float64), 0),
        CurvePlot(AxisRef.point("x")),
    )
    values = model._fit_engine.registry.get("gaussian_offset").evaluate(
        (x,),
        (2.0, 0.1, 0.4, 1.5),
    )
    model.update_data(DatasetSnapshot(schema, values.reshape(1, -1), 1))
    return model


def test_selector_round_trip_and_fit_parameters_follow_display_units() -> None:
    session = _session()
    events = []
    release = session.subscribe_fit(events.append)
    try:
        session.set_axis_unit(AxisRef.point("x"), "cm")
        session.set_value_unit("mV")
        session.set_x_selector(100.0, 200.0, display=True)
        canonical = session.selector_state(SelectorKind.X_RANGE, display=False)
        displayed = session.selector_state(SelectorKind.X_RANGE, display=True)
        assert canonical.value.low == 1.0
        assert canonical.value.high == 2.0
        assert displayed.value.low == 100.0
        assert displayed.value.high == 200.0

        session.fit(
            "gaussian_offset",
            initial=(2.0, 0.1, 0.4, 1.5),
            live=False,
        )
        assert events
        parameters = {item.name: item for item in events[-1].display_parameters}
        assert parameters["center"].unit == "cm"
        assert abs(parameters["center"].value - 150.0) < 0.1
        assert parameters["amplitude"].unit == "mV"
        assert abs(parameters["amplitude"].value - 2000.0) < 0.1
    finally:
        release()
        session.close()


def test_unit_choice_symbols_are_unique_but_alias_input_resolution_is_unchanged() -> None:
    symbols = DEFAULT_UNITS.canonical_symbols()
    resolved = [resolve_unit(symbol) for symbol in symbols]
    assert len(symbols) == len(set(symbols))
    assert len(resolved) == len(set(resolved))
    assert resolve_unit("us") is resolve_unit("µs") is resolve_unit("μs")
    assert resolve_unit("deg") is resolve_unit("°")
    assert resolve_unit("pixel").dimension == "pixel"


def test_display_unit_choices_do_not_repeat_aliases() -> None:
    session = _session()
    try:
        choices = session.describe_display().parameter_choices["x_display_unit"]
        assert len(choices) == len({resolve_unit(choice) for choice in choices})
    finally:
        session.close()
