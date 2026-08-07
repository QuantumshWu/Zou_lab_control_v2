from __future__ import annotations

from types import SimpleNamespace

import zlc_plot.backends as backends


class _Signal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback


class _Timer:
    created = []

    def __init__(self, parent) -> None:
        self.parent = parent
        self.interval = None
        self.timeout = _Signal()
        self.started = False
        self.__class__.created.append(self)

    def setInterval(self, interval) -> None:
        self.interval = interval

    def start(self) -> None:
        self.started = True


def test_ipykernel_wake_timer_uses_dedicated_qt_loop(monkeypatch) -> None:
    """The notebook liveness timer only quits ipykernel's private loop."""

    _Timer.created.clear()
    loop = SimpleNamespace(quit=lambda: setattr(loop, "quit_count", loop.quit_count + 1), quit_count=0)
    shell = SimpleNamespace(
        kernel=SimpleNamespace(app=SimpleNamespace(qt_event_loop=loop))
    )
    modules = SimpleNamespace(
        QtCore=SimpleNamespace(QTimer=_Timer),
        QtWidgets=SimpleNamespace(QApplication=SimpleNamespace(instance=lambda: object())),
    )
    monkeypatch.setattr(backends, "_IPYKERNEL_WAKE_TIMER", None)
    monkeypatch.setattr(backends, "_load_qt5_modules", lambda: modules)

    backends._install_ipykernel_wake_timer(shell)

    assert len(_Timer.created) == 1
    timer = _Timer.created[0]
    assert timer.interval == 50
    assert timer.started is True
    assert timer.timeout.callback is not None
    timer.timeout.callback()
    assert loop.quit_count == 1

    backends._install_ipykernel_wake_timer(shell)
    assert len(_Timer.created) == 1


def test_ipykernel_wake_timer_ignores_shell_without_private_loop(monkeypatch) -> None:
    monkeypatch.setattr(backends, "_IPYKERNEL_WAKE_TIMER", None)
    shell = SimpleNamespace(kernel=SimpleNamespace(app=SimpleNamespace()))
    monkeypatch.setattr(
        backends,
        "_load_qt5_modules",
        lambda: (_ for _ in ()).throw(AssertionError("Qt must not load")),
    )
    backends._install_ipykernel_wake_timer(shell)
    assert backends._IPYKERNEL_WAKE_TIMER is None
