"""A vacant required role is a panel STATE, not a way to lose the console.

The operator gives the x axis another fate.  Three things must be true, and
all three were false: the fate they chose sticks, the panel simply draws
nothing until some axis takes the role, and the console goes on running.
What actually happened was that the vacancy branch reported at a severity
the real status strip does not have; the ValueError left a Qt slot, PyQt
called qFatal, and the whole task console died where it stood -- taking the
operator's fates with it, so the axis appeared to snap back to x.

These tests are written against the REAL view and the REAL app assembly,
because the console double used to accept the very severity that killed the
shipped application.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from pulse_fixtures import PULSE_NAME, write_ordinary_pulse
from test_task_console_app import workspace  # noqa: F401
from zlc_ui import STATUS_SEVERITIES


def _fate_rows(binding) -> dict[str, object]:
    return {
        str(entry["key"]): entry["value"]
        for entry in binding.parameter_surface["semantic"]
        if str(entry["key"]).startswith("fate:")
    }


@pytest.mark.gui
def test_vacating_a_required_role_keeps_the_fate_and_the_console(workspace) -> None:
    """The real console, the real gesture: vacate x on a mounted image."""

    from zlc_atom.nodes.camera_measurement.measurement import (
        CameraMeasurementNode,
        CameraMeasurementRequest,
    )
    from zlc_ui import ensure_qt_app
    from zlc_workbench.apps.task_console import build_console
    from zlc_workbench.session import ExperimentSession

    write_ordinary_pulse(workspace)
    application = ensure_qt_app(["vacant-role"])
    session = ExperimentSession.open(workspace, template="virtual")
    view = presenter = None
    try:
        session.load_pulse(PULSE_NAME)
        node = CameraMeasurementNode(
            camera=session.camera,
            request=CameraMeasurementRequest("camera", 0.02, None, 1, 3),
            signal_plane=session.signal_plane,
            producer="vacant-camera",
        )
        capture = node.prepare()
        session.fire(shots=1)
        capture.collect()
        view, presenter = build_console(session)

        def settle(predicate, seconds=20.0) -> bool:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                presenter.beat()
                application.processEvents()
                if predicate():
                    return True
                time.sleep(0.005)
            return False

        view.add_panel_requested.emit("image")
        panel_id = next(iter(presenter.panels))
        view.panel_state_changed.emit(
            panel_id, {"signal": node.signal_key("frames")}
        )
        binding = presenter.panels[panel_id]
        assert settle(
            lambda: binding.host is not None
            and binding.port.accepted_surface() is not None
        ), "the panel never mounted"

        rows = _fate_rows(binding)
        x_holder = next(key for key, value in rows.items() if value == "x")

        # THE gesture.  Before the fix this line took the process with it.
        view.panel_state_changed.emit(
            panel_id, {"semantic": {x_holder: "reduce"}}
        )
        for _turn in range(60):
            presenter.beat()
            application.processEvents()
            time.sleep(0.005)

        # 1. The operator's fate STICKS -- no silent snap back to x.
        assert _fate_rows(binding)[x_holder] == "reduce"

        # 2. The bench keeps running underneath a table that cannot draw:
        #    new publications arrive and are simply not drawn.
        for _shot in range(2):
            capture = node.prepare()
            session.fire(shots=1)
            capture.collect()
            for _turn in range(40):
                presenter.beat()
                application.processEvents()
                time.sleep(0.005)

        # 3. The console is alive and still holds the operator's table.
        assert _fate_rows(binding)[x_holder] == "reduce"
        assert presenter.panels[panel_id] is binding

        # 4. Giving the role back draws again.
        view.panel_state_changed.emit(panel_id, {"semantic": {x_holder: "x"}})
        assert settle(lambda: _fate_rows(binding)[x_holder] == "x")
    finally:
        if presenter is not None:
            presenter.close()
        if view is not None:
            view.close()
        session.close()


def test_every_authored_status_severity_exists() -> None:
    """No console line may name a severity the status strip does not have.

    A severity is a word two packages have to agree on, and the workbench
    is the side that writes it.  One that only the writer believes in
    reaches the operator as a dead application, so the agreement is checked
    here rather than at the far end of a Qt slot.
    """

    source_root = Path(__file__).resolve().parents[1] / "src"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"severity=\"([a-z_]*)\"", text):
            if match.group(1) not in STATUS_SEVERITIES:
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.name}:{line} -> {match.group(1)!r}")
    assert not offenders, (
        "these lines name a status severity that does not exist: "
        + "; ".join(offenders)
    )


def test_a_reporting_defect_cannot_take_the_console_down() -> None:
    """The diagnostic seam is total: it reports, it does not explode.

    Reporting is what the console does when something has already gone
    wrong.  A status line that can raise turns a small defect into a dead
    instrument, so an unknown severity is shown -- named, at the worst
    severity there is -- and never thrown.
    """

    from types import SimpleNamespace

    from zlc_workbench.console import ConsolePresenter

    shown: list[tuple[str, str]] = []
    presenter = SimpleNamespace(
        view=SimpleNamespace(
            show_status=lambda text, severity: shown.append((severity, text))
        )
    )
    ConsolePresenter._report(presenter, "something happened", severity="nope")
    assert shown and shown[0][0] == "error"
    assert "nope" in shown[0][1] and "something happened" in shown[0][1]
