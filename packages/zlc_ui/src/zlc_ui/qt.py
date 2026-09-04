"""The single QApplication entry point for zlc_ui."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from collections.abc import Sequence

from PyQt5 import QtCore, QtGui, QtWidgets

from .fluent.style import FONT, FONT_SIZE


_QT_ARGV: list[str] | None = None
_QT_APP: QtWidgets.QApplication | None = None
_QT_ICON: QtGui.QIcon | None = None
_QT_LOOP_ENABLED = False
_KERNEL_WAKE_TIMER: QtCore.QTimer | None = None
_KERNEL_WAKE_INTERVAL_MS = 50
_WINDOWS_FLUENT_FONT_FILES = (
    "segoeui.ttf",
    "segoeuib.ttf",
    "segoeuii.ttf",
    "segoeuiz.ttf",
    "segoeuil.ttf",
    "segoeuisl.ttf",
)


def _fluent_font_size() -> int:
    return max(8, int(round(FONT_SIZE)))


def _install_ipykernel_wake_timer(shell) -> None:
    """Bound how long ipykernel 7's idle Qt loop can starve the asyncio loop.

    ipykernel with a dedicated kernel QEventLoop (7.x, and 6.x since ~6.16)
    parks the idle kernel inside ``kernel.app.qt_event_loop``'s ``exec()``
    and only leaves it on new shell-socket activity.  Two stalls
    follow: an execute request that arrives while the loop is being re-entered
    can miss its edge-triggered wake (the next cell hangs before its body even
    starts), and a cell suspended in a top-level ``await`` never resumes
    because asyncio timers cannot fire while ``exec()`` blocks the asyncio
    loop.  Quitting the kernel's dedicated QEventLoop on a short interval
    caps both stalls at the timer period; ipykernel re-enters the loop about
    1 ms later after draining pending asyncio callbacks.  ipykernel builds
    without the dedicated loop (early 6.x) idle in ``QApplication.exec_()``
    directly, which must not be quit from a timer (that would tear down
    nested loops), so the timer is only installed when the dedicated kernel
    QEventLoop exists.
    """

    global _KERNEL_WAKE_TIMER
    if _KERNEL_WAKE_TIMER is not None:
        return
    kernel = getattr(shell, "kernel", None)
    if kernel is None:
        return
    qt_loop = getattr(getattr(kernel, "app", None), "qt_event_loop", None)
    if not callable(getattr(qt_loop, "quit", None)):
        return

    def wake_kernel() -> None:
        # Re-resolve each tick: %gui off/on replaces the kernel's QEventLoop.
        loop = getattr(getattr(kernel, "app", None), "qt_event_loop", None)
        if loop is not None:
            loop.quit()

    timer = QtCore.QTimer(QtWidgets.QApplication.instance())
    timer.setInterval(_KERNEL_WAKE_INTERVAL_MS)
    timer.timeout.connect(wake_kernel)
    timer.start()
    _KERNEL_WAKE_TIMER = timer


def _enable_ipython_qt_loop() -> None:
    """Install the IPython Qt hook when called from a Jupyter kernel."""

    global _QT_LOOP_ENABLED
    if _QT_LOOP_ENABLED:
        return
    try:
        ipython = __import__("IPython", fromlist=["get_ipython"])
        shell = ipython.get_ipython()
    except Exception:
        return
    if shell is None:
        return
    try:
        shell.run_line_magic("gui", "qt")
        _QT_LOOP_ENABLED = True
    except Exception:
        # A plain IPython shell may not have a display; the caller can still
        # use the normal Qt event loop in that case.
        return
    _install_ipykernel_wake_timer(shell)


def _ensure_offscreen_fluent_fonts(app: QtWidgets.QApplication) -> bool:
    """Register the desktop Fluent font for Windows' offscreen Qt backend."""

    if app.platformName().strip().lower() != "offscreen":
        return False
    if FONT in set(QtGui.QFontDatabase().families()):
        return False
    windows_root = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if not windows_root:
        return False
    font_directory = os.path.join(windows_root, "Fonts")
    added = False
    for filename in _WINDOWS_FLUENT_FONT_FILES:
        path = os.path.join(font_directory, filename)
        if os.path.isfile(path):
            added = QtGui.QFontDatabase.addApplicationFont(path) >= 0 or added
    return added


def _ensure_fluent_scale() -> None:
    """Initialize the shared Fluent scale before callers construct widgets."""

    from .fluent.fluent import ensure_fluent_scale

    ensure_fluent_scale()


def _missing_high_dpi_attributes(app: QtWidgets.QApplication) -> tuple[str, ...]:
    """Return Qt5 density attributes that were not enabled before app creation."""

    missing: list[str] = []
    for name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attribute = getattr(QtCore.Qt, name, None)
        if attribute is not None and not app.testAttribute(attribute):
            missing.append(name)
    return tuple(missing)


def ensure_qt_app(argv: Sequence[str] | None = None) -> QtWidgets.QApplication:
    """Return the owner-thread QApplication and configure Fluent once.

    This is the only QApplication constructor in the package.  It also
    applies the shared HiDPI flags, the reference Segoe UI font, Windows
    offscreen font registration, initializes the shared Fluent scale before
    callers construct widgets, performs owner-thread checks, and installs the
    optional IPython Qt event-loop hook.  It deliberately does not resize a
    window: QApplication owns the screen/DPR environment, while
    ``open_fluent_window(factory)`` (or the factory form of
    ``launch_fluent_window``) owns the shared screen-fit window size.  An
    existing QApplication must have been created
    with the same HiDPI attributes; a pre-existing non-HiDPI app is rejected
    because Qt cannot be repaired after construction.
    """

    global _QT_APP, _QT_ARGV
    app = QtWidgets.QApplication.instance()
    if app is not None:
        if QtCore.QThread.currentThread() != app.thread():
            raise RuntimeError("Qt UI operations must run on the QApplication owner thread")
        missing = _missing_high_dpi_attributes(app)
        if missing:
            raise RuntimeError(
                "the existing QApplication was created without Qt5 High-DPI "
                f"attributes ({', '.join(missing)}); restart the process or Jupyter "
                "kernel and call ensure_qt_app() before constructing QApplication"
            )
        _ensure_offscreen_fluent_fonts(app)
        app.setFont(QtGui.QFont(FONT, _fluent_font_size()))
        _ensure_fluent_scale()
        _QT_APP = app
        if app.windowIcon().isNull():
            _claim_taskbar_identity()
            app.setWindowIcon(app_icon())
        _enable_ipython_qt_loop()
        return app

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("QApplication must be created on the Python main thread")
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
    _QT_ARGV = list(sys.argv if argv is None else argv)
    _QT_APP = QtWidgets.QApplication(_QT_ARGV)
    _ensure_offscreen_fluent_fonts(_QT_APP)
    _QT_APP.setFont(QtGui.QFont(FONT, _fluent_font_size()))
    _ensure_fluent_scale()
    _claim_taskbar_identity()
    _QT_APP.setWindowIcon(app_icon())
    _enable_ipython_qt_loop()
    return _QT_APP


#: What Windows groups this program's taskbar button under.  Without an
#: explicit one a python.exe process is grouped -- and iconified -- as Python
#: itself, however many window icons it sets.
APP_USER_MODEL_ID = "ZouLab.ZouLabControl.Console.1"


def _claim_taskbar_identity() -> None:
    """Tell the Windows shell this process is its own program.

    setWindowIcon dresses the WINDOW.  The taskbar button is drawn for the
    application the shell thinks the process is, and for an unmarked
    interpreter that is Python -- which is why the launchers all sat under the
    Python icon no matter what the windows wore.
    """

    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID
        )
    except Exception:
        # An older shell, or a sandbox without shell32: the windows still get
        # their icon, only the grouping stays generic.
        pass


def app_icon() -> QtGui.QIcon:
    """The product's icon, loaded once.

    One accessor because it has two consumers that must not drift: the
    QApplication, which is what a taskbar shows, and every FluentWindow's own
    title bar.  A window wearing a different mark from its taskbar button reads
    as two programs.
    """

    global _QT_ICON
    if _QT_ICON is None:
        path = Path(__file__).with_name("assets") / "zlc.ico"
        _QT_ICON = QtGui.QIcon(str(path)) if path.is_file() else QtGui.QIcon()
    return _QT_ICON


__all__ = ["app_icon", "ensure_qt_app"]
