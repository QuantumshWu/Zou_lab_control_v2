from __future__ import annotations

import numpy as np

from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)

from zlc_data.units import DEFAULT_UNITS, resolve_unit
from zlc_plot import (
    AxisRef,
    CurvePlot,
    NumericRange,
    PlotSession,
    SelectorKind,
)

def _session(*, x_unit: str = "m", value_unit: str = "V") -> PlotSession:
    x = np.linspace(0.0, 3.0, 61)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": x}, units={"x": x_unit}),
        dtype=np.float64,
        value_unit=value_unit,
    )
    model = PlotSession(
        make_snapshot(schema, np.zeros((1, x.size), dtype=np.float64), 0),
        CurvePlot(AxisRef.point("x")),
    )
    values = model._fit_engine.registry.get("gaussian_offset").evaluate(
        (x,),
        (2.0, 0.1, 0.4, 1.5),
    )
    model.update_data(make_snapshot(schema, values.reshape(1, -1), 1))
    return model

def test_selector_round_trip_and_fit_parameters_follow_display_units() -> None:
    session = _session()
    events = []
    release = session.subscribe_fit(events.append)
    try:
        session.set_axis_unit(AxisRef.point("x"), "mm")
        session.set_value_unit("mV")
        session.set_x_selector(1000.0, 2000.0, display=True)
        canonical = session.selector_state(SelectorKind.X_RANGE, display=False)
        displayed = session.selector_state(SelectorKind.X_RANGE, display=True)
        assert canonical.value.low == 1.0
        assert canonical.value.high == 2.0
        assert displayed.value.low == 1000.0
        assert displayed.value.high == 2000.0

        session.fit(
            "gaussian_offset",
            initial=(2.0, 0.1, 0.4, 1.5),
            live=False,
        )
        assert events
        parameters = {item.name: item for item in events[-1].display_parameters}
        assert parameters["center"].unit == "mm"
        assert abs(parameters["center"].value - 1500.0) < 1.0
        assert parameters["amplitude"].unit == "mV"
        assert abs(parameters["amplitude"].value - 2000.0) < 0.1

        session.configure(
            fit={
                "model": "gaussian_offset",
                "expression": "A=2000, x_0=guess(1500)",
            },
            fit_live=False,
        )
        description = session.describe_display()
        assert description.fit["fixed"] == {"amplitude": 2.0}
        assert description.fit["initial"] == {"center": 1.5}
        assert description.fit_expression == (
            "A=2000.0, x_0=guess(1500.0)"
        )

        session.set_axis_unit(AxisRef.point("x"), "m")
        session.set_value_unit("V")
        assert session.describe_display().fit_expression == (
            "A=2.0, x_0=guess(1.5)"
        )
    finally:
        release()
        session.close()

def test_unit_choice_symbols_are_unique_but_alias_input_resolution_is_unchanged() -> None:
    symbols = DEFAULT_UNITS.distinct_symbols()
    resolved = [resolve_unit(symbol) for symbol in symbols]
    assert len(symbols) == len(set(symbols))
    assert len(resolved) == len(set(resolved))
    assert resolve_unit("us") == resolve_unit("µs") == resolve_unit("μs")
    assert resolve_unit("deg") == resolve_unit("°")
    assert resolve_unit("pixel").dimension == "pixel"

def test_display_unit_choices_do_not_repeat_aliases() -> None:
    session = _session()
    try:
        choices = session.describe_display().parameter_choices["x_display_unit"]
        assert len(choices) == len({resolve_unit(choice) for choice in choices})
    finally:
        session.close()


def test_selectors_on_an_axis_not_in_base_units_round_trip_exactly() -> None:
    """Canonical is the DATASET's unit, whatever the unit system's base is.

    Every selector on metres and volts round-tripped because those are base
    units, where "to base" and "to the dataset's unit" coincide.  A seamless
    scan writes its duration column in microseconds: the box drawn on it was
    stored a million times too small, the runtime found no coordinate inside
    it, and the box drawn back from the stored value had no width.  Painted
    values must come back in the column's own unit, and a display unit change
    must convert between the two units, never through the base alone.
    """

    session = _session(x_unit="us", value_unit="count")
    try:
        session.set_x_selector(0.5, 2.0, display=True)
        canonical = session.selector_state(SelectorKind.X_RANGE, display=False)
        assert (canonical.value.low, canonical.value.high) == (0.5, 2.0)
        assert session.selector_state(SelectorKind.X_RANGE, display=True).value == canonical.value

        session.set_area_selector(NumericRange(1.0, 2.5), NumericRange(3.0, 4.0), display=True)
        area = session.selector_state(SelectorKind.AREA, display=False)
        assert (area.value.x.low, area.value.x.high) == (1.0, 2.5)
        assert (area.value.y.low, area.value.y.high) == (3.0, 4.0)

        session.set_axis_unit(AxisRef.point("x"), "ms")
        shown = session.selector_state(SelectorKind.AREA, display=True)
        assert abs(shown.value.x.low - 0.001) < 1e-12 and abs(shown.value.x.high - 0.0025) < 1e-12
        session.set_area_selector(NumericRange(0.002, 0.003), NumericRange(3.0, 4.0), display=True)
        moved = session.selector_state(SelectorKind.AREA, display=False)
        assert abs(moved.value.x.low - 2.0) < 1e-9 and abs(moved.value.x.high - 3.0) < 1e-9

        session.set_viewport(NumericRange(0.001, 0.002), NumericRange(0.0, 1.0))
        assert abs(session.viewport.x.low - 0.001) < 1e-12
    finally:
        session.close()

