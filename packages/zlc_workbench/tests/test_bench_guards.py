"""The bench's guards must actually refuse the mistakes they name.

Every guard in ``bench.plot_perf.guards`` exists because a measurement was
reported that described the harness rather than the product.  A guard that
cannot fail is decoration, so each one is exercised here on both sides.

This lives with zlc_workbench because the console layer of the bench is
composed from it -- if the product's own vocabulary moves, these fail here
rather than silently in a bench nobody runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.plot_perf import guards  # noqa: E402


def _renderer(*, ratio: float, width: float = 1470.0, height: float = 1071.0):
    return SimpleNamespace(
        figure=SimpleNamespace(bbox=SimpleNamespace(width=width, height=height)),
        plan=SimpleNamespace(device_pixel_ratio=ratio, dpi=210.0 * ratio),
    )


def test_a_low_density_surface_is_refused_by_name() -> None:
    """Offscreen Qt gives DPR 1 and one ninth of the pixels."""

    facts = guards.require_real_density(_renderer(ratio=3.0))
    assert facts["device_pixel_ratio"] == 3.0
    assert facts["figure_px"] == (1470, 1071)

    with pytest.raises(guards.HarnessError) as refused:
        guards.require_real_density(_renderer(ratio=1.0, width=826, height=609))
    assert "offscreen" in str(refused.value)
    # ...and a caller who means it can say so.
    guards.require_real_density(
        _renderer(ratio=1.0, width=826, height=609), minimum_ratio=1.0
    )


def test_a_second_panel_cannot_hide_in_the_measurement() -> None:
    """Class-level taps time every renderer; this makes the extra one visible."""

    console = SimpleNamespace(panels={"panel-1": object(), "panel-2": object()})
    assert guards.require_panels(console, 2) == ("panel-1", "panel-2")
    with pytest.raises(guards.HarnessError) as refused:
        guards.require_panels(console, 1)
    assert "2 panels" in str(refused.value)


def test_a_quiet_window_that_was_not_quiet_is_refused() -> None:
    """``repeat: 0`` keeps the virtual camera producing with no pulse fired."""

    quiet = {"signal": "s", "distinct_revisions": 1, "per_second": 0.2, "quiet": True}
    assert guards.require_quiet(quiet) is quiet
    busy = {"signal": "s", "distinct_revisions": 96, "per_second": 19.2, "quiet": False}
    with pytest.raises(guards.HarnessError) as refused:
        guards.require_quiet(busy)
    assert "meant to be quiet" in str(refused.value)


def test_a_gesture_that_never_landed_is_refused() -> None:
    """Synthesised pointer calls build the gesture and drop its moves."""

    turned = guards.require_effect((55.0, 30.0), (-146.6, 80.0), "the camera")
    assert turned == (-146.6, 80.0)
    with pytest.raises(guards.HarnessError) as refused:
        guards.require_effect((55.0, 30.0), (55.0, 30.0), "the camera")
    assert "not delivered" in str(refused.value)


def test_a_refused_state_change_is_not_read_as_a_product_defect() -> None:
    """``update_panel_state`` takes a fixed vocabulary and says so."""

    assert guards.applied(True, "size") is True
    with pytest.raises(guards.HarnessError):
        guards.applied(False, "selector")


def test_the_committed_region_is_read_from_the_panel_itself() -> None:
    """What a gesture owns, in numbers a before/after comparison can use."""

    panel = SimpleNamespace(
        state=SimpleNamespace(
            selector={
                "ranges": (
                    {"domain": "value", "lower": 6.5, "upper": 17.3},
                    {"domain": "shot", "lower": -40.0, "upper": -2.0},
                )
            }
        )
    )
    assert guards.committed_region(panel) == (
        ("value", 6.5, 17.3),
        ("shot", -40.0, -2.0),
    )
    assert guards.committed_region(SimpleNamespace(state=SimpleNamespace(selector={}))) == ()


def test_the_bench_benchmarks_one_size_and_says_which() -> None:
    """Three layers comparing different pictures is not a comparison."""

    from bench.plot_perf.common import SIZE_PRESET

    assert SIZE_PRESET == "2x2"


def test_a_console_bench_cannot_be_left_open() -> None:
    """The console layer opens a real window and non-daemon threads.

    A bench that only quiets the pulse leaves the panels' raster workers,
    the logic node, the save worker build_console attaches, the window and
    the session's device claims all standing -- so the process never exits
    and the console stays on screen until it is killed from the task list.
    That happened, repeatedly, which is why close() now runs the product's
    own shutdown and why the runner holds the bench in a ``with``.
    """

    from bench.plot_perf.run_console import ConsoleBench

    assert hasattr(ConsoleBench, "__enter__")
    assert hasattr(ConsoleBench, "__exit__")

    # Closing something that never started must not raise: a failure during
    # start() has to leave the ``with`` able to clean up after it.
    never_started = ConsoleBench.__new__(ConsoleBench)
    never_started.close()

    # And the survivor check counts what would actually hold the process
    # open -- not the main thread, and not daemons.
    import threading

    survivors = ConsoleBench.surviving_threads(never_started)
    assert threading.main_thread().name not in survivors
    assert all(
        not thread.daemon
        for thread in threading.enumerate()
        if thread.name in survivors
    )
