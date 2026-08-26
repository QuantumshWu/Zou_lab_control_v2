"""A rolling region is the panel's, even though nothing derives from it.

Marking a stretch of a rolling trace says "these shots, here".  Nothing
upstream is cut by it -- the x is the shot history the session accumulates,
not a row of the publication a derived signal would come from -- and that
used to be read as "did not happen": the region was dropped before it ever
reached the panel, so the card and the Setting editor showed different
marks and a new generation lost it entirely.
"""

from __future__ import annotations

import time

import pytest

from test_console_presenter import (  # noqa: F401 -- fixtures
    PULSE_NAME,
    _commit_area,
    _settle_panel_hosts,
    presenter,
    session,
)
from zlc_workbench.logic import stable_signal_key
from zlc_workbench.selection import panel_selection_derives_signal


def test_a_rolling_region_is_remembered_and_derives_nothing(
    presenter, session
) -> None:
    from zlc_runtime.selection_bridge import SelectionRange, SelectionState

    camera_id = presenter.add_logic(
        "camera_measurement",
        node_id="roll-cam",
        values={"exposure_seconds": 0.002, "repeat": 0, "frames_per_cycle": 1},
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(camera_id)
    signal = stable_signal_key(camera_id, "frames")
    publication = None
    deadline = time.monotonic() + 20.0
    while publication is None and time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        publication = session.signal_plane.latest_publication(signal)
        time.sleep(0.005)
    assert publication is not None

    binding = presenter.add_panel(
        signal, publication.value(signal).snapshot, kind="rolling"
    )
    _settle_panel_hosts(presenter, lambda: binding.host is not None)
    presenter.set_deriving(True)
    for _ in range(20):
        session.fire(shots=1)
        presenter.beat()
        time.sleep(0.01)

    assert binding.state.selector == {}, "nothing marked yet"
    _commit_area(binding.host, lower_fraction=0.3, upper_fraction=0.7)
    _settle_panel_hosts(presenter, lambda: bool(binding.state.selector))

    # The panel remembers it, in its own words.
    document = binding.state.selector
    assert document, "the rolling region was dropped instead of remembered"
    assert document["plot_kind"] == "rolling"
    domains = [str(item["domain"]) for item in document["ranges"]]
    assert domains == ["shot", "value"], domains

    # And nothing upstream is cut by it: a rolling region derives no signal.
    marked = SelectionState(
        plot_kind="rolling",
        selector_kind="x_range",
        ranges=(
            SelectionRange(
                axis="", lower=1.0, upper=5.0, domain="shot"
            ),
        ),
    )
    assert panel_selection_derives_signal(marked) is False


def test_the_reportable_kinds_are_a_superset_of_the_derivable_ones() -> None:
    """The two vocabularies are related, and the relation is stated once.

    Every kind that can derive must be reportable; the extra reportable
    kind is exactly the one whose region is the panel's own.
    """

    from zlc_runtime.selection_bridge import SELECTION_PLOT_KINDS
    from zlc_workbench.selection import _PLOT_KINDS

    assert set(_PLOT_KINDS) == set(SELECTION_PLOT_KINDS)
    assert "rolling" in SELECTION_PLOT_KINDS
