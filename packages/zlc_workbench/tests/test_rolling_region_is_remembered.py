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

from test_console_presenter import (  # noqa: F401 -- fixtures
    _commit_area,
    _settle_panel_hosts,
    _started_camera,
    presenter,
    session,
)


def test_a_rolling_region_is_remembered_and_derives_nothing(
    presenter, session
) -> None:
    _camera_id, signal, publication = _started_camera(
        presenter, session, node_id="roll-cam", timeout=20.0
    )

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
    # That a shot range names no axis, so it binds no revision and derives
    # nothing, is test_selection's rolling-viewport test, on a real gesture.
