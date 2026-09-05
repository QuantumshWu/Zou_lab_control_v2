"""Every axes plan states its room, and neighbours split their gap."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from test_tick_labels import _CASES, _surface_sessions
from zlc_plot.layout import Room


def _plans(session):
    plan = session._renderer.plan
    plans = list(plan.axes)
    if plan.facet_focus_axes is not None:
        plans.extend(plan.facet_focus_axes)
    return plans


@pytest.mark.parametrize("preset", ("1x2", "2x2", "4x4", "8x8"))
@pytest.mark.parametrize("case", _CASES)
def test_every_axes_plan_states_a_room_inside_the_figure(case: str, preset: str) -> None:
    session = _surface_sessions(case, preset, 1.0)
    try:
        plans = _plans(session)
        assert plans
        for item in plans:
            room = item.room
            assert isinstance(room, Room)
            assert item.box.left - room.left >= -1e-9
            assert item.box.right + room.right <= 1.0 + 1e-9
            assert item.box.top - room.top >= -1e-9
            assert item.box.bottom + room.bottom <= 1.0 + 1e-9
        by_role = {item.role: item for item in plans}
        if "distribution" in by_role and "image" in by_role:
            image, rail, bar = by_role["image"], by_role["distribution"], by_role["colorbar"]
            assert image.room.right == rail.room.left == pytest.approx((rail.box.left - image.box.right) / 2)
            assert rail.room.right == bar.room.left == pytest.approx((bar.box.left - rail.box.right) / 2)
            assert bar.room.right == pytest.approx(1.0 - bar.box.right)
        if "history" in by_role and "distribution" in by_role:
            history, rail = by_role["history"], by_role["distribution"]
            assert history.room.right == rail.room.left == pytest.approx((rail.box.left - history.box.right) / 2)
        cells = [item for item in plans if item.role == "facet_cell"]
        if cells:
            assert len({item.room for item in cells}) == 1, "every cell reads against one frame"
    finally:
        session.close()


@pytest.mark.parametrize("case", ("image-512", "rolling", "facet"))
def test_a_room_does_not_depend_on_the_screen(case: str) -> None:
    one = _surface_sessions(case, "2x2", 1.0)
    three = _surface_sessions(case, "2x2", 3.0)
    try:
        assert [item.room for item in _plans(one)] == [item.room for item in _plans(three)]
    finally:
        one.close()
        three.close()
