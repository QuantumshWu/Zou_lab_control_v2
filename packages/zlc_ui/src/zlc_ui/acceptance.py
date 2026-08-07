"""The one human-size Qt acceptance capture path.

This module is deliberately small.  It opens the same non-blocking window
factory that a human uses, checks the shared screen-fit geometry, and writes
the physical-DPR window capture plus an optional screen-sized proportion
canvas.  The resulting PNG and geometry report are the evidence used for
visual UI acceptance.

The offscreen Qt backend is rejected because it cannot reproduce the human
desktop that this API is meant to inspect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from .fluent import WINDOW_SCREEN_FRACTION, screen_fit_window_size
from .qt import ensure_qt_app


GeometryTuple = tuple[int, int, int, int]
SizeTuple = tuple[int, int]


@dataclass(frozen=True)
class AcceptanceCapture:
    """Immutable report for one validated human-size top-level capture.

    Geometry is in logical Qt units.  Bitmap sizes are physical pixels and
    therefore include the monitor device-pixel ratio.  ``desktop_output`` is
    a neutral screen-sized canvas containing the real window grab at its real
    screen-relative position; it proves proportion and placement, not that
    host desktop pixels were captured.
    """

    platform: str
    available_geometry: GeometryTuple
    screen_geometry: GeometryTuple
    screen_dpr: float
    target_size: SizeTuple
    window_size: SizeTuple
    frame_geometry: GeometryTuple
    window_fraction: tuple[float, float]
    grab_size: SizeTuple
    desktop_size: SizeTuple | None
    title_bar_size: SizeTuple | None
    content_top: int | None
    content_top_margin: int | None
    output: Path | None
    desktop_output: Path | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostic mapping."""

        return {
            "platform": self.platform,
            "available_logical": self.available_geometry,
            "screen_logical": self.screen_geometry,
            "screen_dpr": self.screen_dpr,
            "target_logical": self.target_size,
            "window_logical": self.window_size,
            "window_fraction": self.window_fraction,
            "window_frame": self.frame_geometry,
            "grab_physical": self.grab_size,
            "desktop_relative_physical": self.desktop_size,
            "title_bar_logical": self.title_bar_size,
            "content_top_logical": self.content_top,
            "content_top_margin": self.content_top_margin,
            "output": None if self.output is None else str(self.output),
            "desktop_relative_output": (
                None if self.desktop_output is None else str(self.desktop_output)
            ),
        }


def _rect(rect: QtCore.QRect) -> GeometryTuple:
    return (int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))


def _size(size: QtCore.QSize) -> SizeTuple:
    return (int(size.width()), int(size.height()))


def _top_level_widget(value: object) -> QtWidgets.QWidget:
    """The window behind whatever the entry answered with.

    Every window here is opened through a handle now, and a handle is
    deliberately not a QWidget -- that is the whole point of it.  Capture is
    inside this package, so it is allowed to look behind one; nobody outside
    is, which is why this unwrapping lives here and not at the call site.
    """

    inner = getattr(value, "_window", None)
    if inner is not None:
        value = inner
    if not isinstance(value, QtWidgets.QWidget):
        raise TypeError("window opener must return a QWidget or a window handle")
    window = value if value.isWindow() else value.window()
    if not isinstance(window, QtWidgets.QWidget):
        raise TypeError("window opener returned a QWidget without a top-level window")
    return window


def _settle(app: QtWidgets.QApplication, milliseconds: int) -> None:
    """Pump Qt events for a bounded render-settle interval."""

    duration = max(0, int(milliseconds)) / 1000.0
    deadline = time.monotonic() + duration
    while True:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if time.monotonic() >= deadline:
            return


def _desktop_relative_pixmap(
    screen: QtGui.QScreen,
    window: QtWidgets.QWidget,
    window_pixmap: QtGui.QPixmap,
) -> QtGui.QPixmap:
    """Compose the real window grab on a physical-pixel screen canvas."""

    dpr = float(screen.devicePixelRatio())
    screen_rect = screen.geometry()
    canvas = QtGui.QPixmap(
        max(1, round(screen_rect.width() * dpr)),
        max(1, round(screen_rect.height() * dpr)),
    )
    canvas.setDevicePixelRatio(1.0)
    canvas.fill(QtGui.QColor("#D9D9D9"))

    foreground = QtGui.QPixmap(window_pixmap)
    foreground.setDevicePixelRatio(1.0)
    frame = window.frameGeometry()
    x = round((frame.x() - screen_rect.x()) * dpr)
    y = round((frame.y() - screen_rect.y()) * dpr)
    painter = QtGui.QPainter(canvas)
    painter.drawPixmap(x, y, foreground)
    painter.end()
    return canvas


def capture_window(
    opener: Callable[[], QtWidgets.QWidget],
    *,
    ratio: float = WINDOW_SCREEN_FRACTION,
    output: str | Path | None = None,
    desktop_output: str | Path | None = None,
    settle_ms: int = 120,
    close: bool = True,
) -> AcceptanceCapture:
    """Capture one human-visible Qt window through its real entry point.

    ``opener`` must be the same non-blocking ``create_window`` callable a
    human uses, or a wrapper around it.  The function calls
    :func:`ensure_qt_app` before invoking the factory, pumps only the Qt event
    loop, and verifies that the returned top-level window is exactly
    ``screen_fit_window_size(ratio)``.  It never resizes a mismatching window:
    a wrong launcher fails loudly instead of producing a plausible but false
    screenshot.

    This is the only UI acceptance capture API.  The default backend must be
    a real Qt screen; offscreen is rejected because its virtual canvas cannot
    represent the human monitor.  ``output`` is the physical-DPR window crop;
    ``desktop_output`` is an optional neutral full-screen composition used to
    make the 90%-of-screen relationship visible.  For the shared
    :class:`~zlc_ui.fluent.FluentWindow` shell, the report also validates that
    the loaded body begins below the actual native title-bar bottom.
    """

    if not callable(opener):
        raise TypeError("opener must be a zero-argument create_window callable")
    if not 0 < float(ratio) <= 1:
        raise ValueError("ratio must be between 0 and 1")

    app = ensure_qt_app(["zlc-ui-acceptance"])
    window: QtWidgets.QWidget | None = None
    try:
        window = _top_level_widget(opener())
        if not window.isVisible():
            window.show()
        _settle(app, settle_ms)

        platform = str(app.platformName())
        if platform.strip().lower() == "offscreen":
            raise RuntimeError(
                "UI acceptance capture requires a real Qt screen; "
                "the offscreen backend cannot reproduce the human GUI"
            )

        screen = window.screen() or app.primaryScreen()
        if screen is None:
            raise RuntimeError("QApplication has no primary screen")
        available = screen.availableGeometry()
        target = screen_fit_window_size(float(ratio))
        actual = window.size()
        if actual != target:
            raise RuntimeError(
                "window opener did not use the shared screen-fit launcher: "
                f"target={target.width()}x{target.height()}, "
                f"window={actual.width()}x{actual.height()}"
            )

        title_bar = getattr(window, "titleBar", None)
        loaded = getattr(window, "loaded", None)
        title_bar_size = None
        content_top = None
        content_top_margin = None
        if (
            isinstance(title_bar, QtWidgets.QWidget)
            and isinstance(loaded, QtWidgets.QWidget)
            and title_bar.parentWidget() is window
            and loaded.parentWidget() is window
        ):
            title_bar_size = _size(title_bar.size())
            content_top = int(loaded.geometry().top())
            margins = window.layout().contentsMargins() if window.layout() else None
            content_top_margin = None if margins is None else int(margins.top())
            title_bar_bottom = int(title_bar.geometry().bottom())
            if content_top < title_bar_bottom + 1:
                raise RuntimeError(
                    "shared frameless shell overlaps native title bar: "
                    f"title_bar_bottom={title_bar_bottom}, content_top={content_top}"
                )

        pixmap = window.grab()
        if pixmap.isNull():
            raise RuntimeError("Qt returned a null window grab")

        output_path = None if output is None else Path(output).expanduser().resolve()
        desktop_path = (
            None
            if desktop_output is None
            else Path(desktop_output).expanduser().resolve()
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not pixmap.save(str(output_path), "PNG"):
                raise RuntimeError(f"could not save Qt window grab to {output_path}")

        desktop_pixmap = None
        if desktop_path is not None:
            desktop_pixmap = _desktop_relative_pixmap(screen, window, pixmap)
            desktop_path.parent.mkdir(parents=True, exist_ok=True)
            if not desktop_pixmap.save(str(desktop_path), "PNG"):
                raise RuntimeError(
                    f"could not save desktop-relative capture to {desktop_path}"
                )

        fraction = (
            actual.width() / available.width(),
            actual.height() / available.height(),
        )
        return AcceptanceCapture(
            platform=platform,
            available_geometry=_rect(available),
            screen_geometry=_rect(screen.geometry()),
            screen_dpr=float(screen.devicePixelRatio()),
            target_size=_size(target),
            window_size=_size(actual),
            frame_geometry=_rect(window.frameGeometry()),
            window_fraction=fraction,
            grab_size=(pixmap.width(), pixmap.height()),
            desktop_size=(
                None
                if desktop_pixmap is None
                else (desktop_pixmap.width(), desktop_pixmap.height())
            ),
            title_bar_size=title_bar_size,
            content_top=content_top,
            content_top_margin=content_top_margin,
            output=output_path,
            desktop_output=desktop_path,
        )
    finally:
        if close and window is not None:
            window.close()
            app.processEvents()


__all__ = ["AcceptanceCapture", "capture_window"]
