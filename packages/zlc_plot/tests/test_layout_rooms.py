"""An axes plan's room is the same whatever screen draws it."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from test_tick_labels import _surface_sessions


def _plans(session):
    plan = session._renderer.plan
    plans = list(plan.axes)
    if plan.facet_focus_axes is not None:
        plans.extend(plan.facet_focus_axes)
    return plans


@pytest.mark.parametrize("case", ("image-512", "rolling", "facet"))
def test_a_room_does_not_depend_on_the_screen(case: str) -> None:
    one = _surface_sessions(case, "2x2", 1.0)
    three = _surface_sessions(case, "2x2", 3.0)
    try:
        assert [item.room for item in _plans(one)] == [item.room for item in _plans(three)]
    finally:
        one.close()
        three.close()
