"""Loop rails stack by what they contain, not by where a loop came in the list."""

from __future__ import annotations

from zlc_plot import PulseLoopMarker
from zlc_plot._rendering.pulse import _loop_depths


def test_disjoint_brackets_stand_at_one_level_under_the_run_loop() -> None:
    first = PulseLoopMarker(0.0, 1.0, "×2", series=0)
    second = PulseLoopMarker(2.0, 3.0, "×3", series=1)
    run = PulseLoopMarker(0.0, 4.0, "×∞")
    assert _loop_depths((second, first, run)) == (0, 0, 1)


def test_nested_brackets_stack_from_the_inside_out_whatever_the_list_order() -> None:
    inner = PulseLoopMarker(1.0, 2.0, "×5", series=1)
    outer = PulseLoopMarker(0.0, 3.0, "×2", series=0)
    run = PulseLoopMarker(0.0, 4.0, "×∞")
    assert _loop_depths((inner, outer, run)) == (0, 1, 2)
    assert _loop_depths((run, outer, inner)) == (2, 1, 0)


def test_two_loops_over_the_same_span_keep_the_callers_inner_then_outer_order() -> None:
    whole = PulseLoopMarker(0.0, 4.0, "×3", series=0)
    run = PulseLoopMarker(0.0, 4.0, "×7")
    assert _loop_depths((whole, run)) == (0, 1)
    beside = PulseLoopMarker(5.0, 6.0, "×2", series=1)
    assert _loop_depths((whole, run, beside)) == (0, 1, 0)
