"""The embed demo must survive a rejected semantic edit and stay responsive.

Regression for the semantic-UX audit finding #1: the error-report closure
referenced the freed ``except ... as error`` name, raised NameError inside a
discarded dispatch future, and left ``_switching`` latched — every later
semantic edit, page switch and live publish silently died with no visible
error.  This test drives the real ``_track``/``_dispatch`` chain offscreen.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from zlc_plot.specs import Reduction  # noqa: E402

_WAIT_SECONDS = 8.0


def _wait_until(app, predicate, message: str) -> None:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
    raise AssertionError(f"timed out waiting for: {message}")


@pytest.mark.gui
def test_rejected_semantic_edit_reports_and_stays_responsive() -> None:
    from examples.pyqt5_embed import create_window
    from zlc_plot import ensure_qt5_application

    app = ensure_qt5_application()
    handle = create_window()
    try:
        _wait_until(app, lambda: handle._active is not None, "demo becomes active")
        active = handle._active
        assert active is not None

        # Force the tracked future to fail exactly like a rejected replace.
        real_replace = active.host.replace_spec

        def failing_replace(candidate):
            future: Future = Future()
            future.set_exception(ValueError("synthetic rejection"))
            return future

        active.host.replace_spec = failing_replace
        try:
            handle._semantic_edited("reduction", Reduction.MEDIAN)
            _wait_until(
                app,
                lambda: "synthetic rejection" in active.panel._error_label.text(),
                "rejection reaches the panel error label",
            )
        finally:
            active.host.replace_spec = real_replace

        # The rejection must release the switch latch and be visible twice.
        _wait_until(app, lambda: not handle._switching, "switch latch releases")
        assert "synthetic rejection" in handle.status.text()

        # The window must remain fully responsive: a real semantic edit now
        # succeeds, clears the error and refreshes the semantic description.
        handle._semantic_edited("reduction", Reduction.MEDIAN)
        _wait_until(
            app,
            lambda: (
                not handle._switching
                and active.panel._error_label.text() == ""
                and active.semantics.reduction is Reduction.MEDIAN
            ),
            "follow-up semantic edit is accepted and clears the error",
        )
    finally:
        handle.close(wait=True)


@pytest.mark.gui
def test_kind_switch_keeps_the_page_data_and_repick_restores_pristine() -> None:
    """A semantic kind switch re-projects the page's own data, and re-picking
    the data source page restores its authored spec.

    The page combo selects data sources only — a kind is chosen exclusively
    in the semantic panel, so switching kinds never swaps the example data.
    """

    from examples.pyqt5_embed import create_window
    from zlc_plot import ensure_qt5_application
    from zlc_plot.kinds import PlotKind

    app = ensure_qt5_application()
    handle = create_window()
    try:
        _wait_until(app, lambda: handle._active is not None, "demo becomes active")
        scan = handle._definitions[0]
        assert handle._active.definition is scan
        authored_spec = scan.spec
        data_source = handle._active.definition.make_update

        handle._semantic_edited("kind", PlotKind.HISTOGRAM)
        _wait_until(
            app,
            lambda: handle._active.definition.spec.kind is PlotKind.HISTOGRAM
            and not handle._switching,
            "semantic kind switch to histogram completes",
        )
        # Same page, same data producer: the switch re-projects the scan
        # dataset instead of jumping to another page's example data.
        assert handle._active.definition.page_id == scan.page_id
        assert handle._active.definition.make_update is data_source
        assert handle.page.currentData() == scan.page_id

        # Re-picking the data source page is a same-index click; it emits
        # only ``activated`` and must restore the authored pristine spec.
        index = handle.page.currentIndex()
        handle.page.activated.emit(index)
        _wait_until(
            app,
            lambda: (
                handle._active is not None
                and handle._active.definition is scan
                and not handle._switching
            ),
            "same-index re-pick restores the pristine page",
        )
        assert handle._active.definition.spec == authored_spec
    finally:
        handle.close(wait=True)
