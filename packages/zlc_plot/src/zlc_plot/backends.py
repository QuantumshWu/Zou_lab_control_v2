"""Optional notebook and PyQt5 adapters for PlotSession.

Qt is loaded only when its adapter is materialised. Notebook views consume the
package raster front directly, so users need no backend magic and the package
does not alter unrelated pyplot figures.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
import importlib
import math
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable

from ._axis_transform import AxisTransform
from ._selector_scene import (
    ColorLimitCandidate,
    SceneKind,
)
from .assets import HELVETICA_LIGHT_FAMILY, helvetica_light_path
from .raster import (
    RasterFront,
    RasterPlotHost,
)
from .selectors import SelectorState

if TYPE_CHECKING:
    from .session import PlotSession


class BackendUnavailableError(RuntimeError):
    """An explicitly requested optional frontend is not installed/configured."""


def _complete_cleanup(callbacks: tuple[Callable[[], object], ...]) -> None:
    """Run every cleanup edge, then re-raise the first failure unchanged."""

    first_error: Exception | None = None
    for callback in callbacks:
        try:
            callback()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error



class _InteractionGate:
    """One adapter-owned switch and generation for native input transport."""

    __slots__ = ("_enabled", "_generation")

    def __init__(self) -> None:
        self._enabled = True
        self._generation = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def generation(self) -> int:
        return self._generation

    def set_enabled(self, enabled: bool) -> bool:
        if not isinstance(enabled, bool):
            raise TypeError("interaction enabled state must be bool")
        if enabled is self._enabled:
            return False
        self._enabled = enabled
        self._generation += 1
        return True


@dataclass(frozen=True, slots=True)
class _QtModules:
    QtCore: Any
    QtGui: Any
    QtWidgets: Any


_QT_MODULES: _QtModules | None = None
_QT_WIDGET_CLASS: type[Any] | None = None
_QT_APPLICATION: object | None = None
_QT_FONT_FAMILY: str | None = None
_IPYTHON_QT_LOOP_ENABLED = False
_IPYKERNEL_WAKE_TIMER: object | None = None
_IPYKERNEL_WAKE_INTERVAL_MS = 50
_QT_WINDOW_BIND_RETRY_INTERVAL_MS = 16
_QT_WINDOW_BIND_MAX_ATTEMPTS = 8


def _register_qt5_font(qt_gui: object) -> str:
    """Register and verify the package font once after QApplication exists."""

    global _QT_FONT_FAMILY
    if _QT_FONT_FAMILY is not None:
        return _QT_FONT_FAMILY
    expected = HELVETICA_LIGHT_FAMILY
    font_id = int(
        qt_gui.QFontDatabase.addApplicationFont(str(helvetica_light_path()))
    )
    if font_id < 0:
        raise RuntimeError("Qt could not register the packaged Helvetica Light font")
    families = tuple(qt_gui.QFontDatabase.applicationFontFamilies(font_id))
    if expected not in families:
        qt_gui.QFontDatabase.removeApplicationFont(font_id)
        raise RuntimeError(
            f"canonical Qt font asset identifies as {families!r}, not {expected!r}"
        )
    _QT_FONT_FAMILY = expected
    return expected


def _install_ipykernel_wake_timer(shell: object) -> None:
    """Bound how long ipykernel's idle Qt loop can starve the asyncio loop.

    ipykernel with a dedicated kernel QEventLoop (7.x, and 6.x since ~6.16)
    parks the idle kernel inside ``kernel.app.qt_event_loop``'s ``exec()``
    and only leaves it on new shell-socket activity.  Two stalls follow: an
    execute request that arrives while the loop is being re-entered can miss
    its edge-triggered wake (the next cell hangs before its body even
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

    global _IPYKERNEL_WAKE_TIMER
    if _IPYKERNEL_WAKE_TIMER is not None:
        return
    kernel = getattr(shell, "kernel", None)
    if kernel is None:
        return
    qt_loop = getattr(getattr(kernel, "app", None), "qt_event_loop", None)
    if not callable(getattr(qt_loop, "quit", None)):
        return
    modules = _load_qt5_modules()

    def wake_kernel() -> None:
        # Re-resolve each tick: %gui off/on replaces the kernel's QEventLoop.
        loop = getattr(getattr(kernel, "app", None), "qt_event_loop", None)
        if loop is not None:
            loop.quit()

    timer = modules.QtCore.QTimer(modules.QtWidgets.QApplication.instance())
    timer.setInterval(_IPYKERNEL_WAKE_INTERVAL_MS)
    timer.timeout.connect(wake_kernel)
    timer.start()
    _IPYKERNEL_WAKE_TIMER = timer


def _enable_ipython_qt_loop() -> None:
    """Install IPython's Qt pump once when a widget is created in a notebook."""

    global _IPYTHON_QT_LOOP_ENABLED
    if _IPYTHON_QT_LOOP_ENABLED:
        return
    if os.environ.get("QT_QPA_PLATFORM", "").strip().lower() in {
        "offscreen",
        "minimal",
    }:
        return
    try:
        ipython_module = importlib.import_module("IPython")
        get_ipython = getattr(ipython_module, "get_ipython")
        shell = get_ipython()
    except (AttributeError, ImportError, ModuleNotFoundError):
        return
    if shell is None:
        return
    run_line_magic = getattr(shell, "run_line_magic", None)
    if not callable(run_line_magic):
        return
    try:
        run_line_magic("gui", "qt5")
    except Exception:
        return
    _IPYTHON_QT_LOOP_ENABLED = True
    _install_ipykernel_wake_timer(shell)


def _configure_qt5_high_dpi(qt_core: object, qt_widgets: object) -> None:
    """Set process-wide Qt5 density attributes before QApplication exists."""

    application_type = getattr(qt_widgets, "QApplication", None)
    qt = getattr(qt_core, "Qt", None)
    if application_type is None or qt is None or application_type.instance() is not None:
        return
    for name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attribute = getattr(qt, name, None)
        if attribute is not None:
            application_type.setAttribute(attribute, True)


def _missing_qt5_high_dpi_attributes(qt_core: object, application: object) -> tuple[str, ...]:
    qt = getattr(qt_core, "Qt", None)
    test_attribute = getattr(application, "testAttribute", None)
    if qt is None or not callable(test_attribute):
        return ()
    return tuple(
        name
        for name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps")
        if getattr(qt, name, None) is not None
        and not bool(test_attribute(getattr(qt, name)))
    )


def _load_qt5_modules() -> _QtModules:
    global _QT_MODULES
    if _QT_MODULES is not None:
        return _QT_MODULES
    try:
        qt_core = importlib.import_module("PyQt5.QtCore")
        qt_gui = importlib.import_module("PyQt5.QtGui")
        qt_widgets = importlib.import_module("PyQt5.QtWidgets")
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailableError(
            "Qt5PlotWidget requires PyQt5; "
            "install them with `pip install zlc-plot[qt]`."
        ) from error
    _configure_qt5_high_dpi(qt_core, qt_widgets)
    _QT_MODULES = _QtModules(qt_core, qt_gui, qt_widgets)
    return _QT_MODULES


def ensure_qt5_application(
    argv: list[str] | tuple[str, ...] | None = None,
) -> object:
    """Return the owner-thread QApplication with Qt5 DPR support enabled.

    Density attributes must be selected before the first QApplication is
    created.  Calling this entry point instead of constructing QApplication
    directly gives both standalone programs and notebook-launched Qt windows
    the same per-monitor DPR contract.
    """

    global _QT_APPLICATION
    modules = _load_qt5_modules()
    QtCore, QtWidgets = modules.QtCore, modules.QtWidgets
    application = QtWidgets.QApplication.instance()
    if application is not None:
        if not isinstance(application, QtWidgets.QApplication):
            raise RuntimeError("the active Qt application is not a QApplication")
        if QtCore.QThread.currentThread() != application.thread():
            raise RuntimeError("QApplication must be accessed from its owner thread")
        missing = _missing_qt5_high_dpi_attributes(QtCore, application)
        if missing:
            raise RuntimeError(
                "the existing QApplication was created without Qt5 High-DPI "
                f"attributes ({', '.join(missing)}); restart the process and call "
                "ensure_qt5_application() before constructing QApplication"
            )
        _register_qt5_font(modules.QtGui)
        _QT_APPLICATION = application
        _enable_ipython_qt_loop()
        return application
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("QApplication must be created on the Python main thread")
    if argv is not None and not isinstance(argv, (list, tuple)):
        raise TypeError("QApplication argv must be a list, tuple, or None")
    selected_argv = sys.argv if argv is None else argv
    arguments = list(selected_argv)
    if any(not isinstance(value, str) for value in arguments):
        raise TypeError("QApplication arguments must be strings")
    _configure_qt5_high_dpi(QtCore, QtWidgets)
    _QT_APPLICATION = QtWidgets.QApplication(arguments)
    _register_qt5_font(modules.QtGui)
    _enable_ipython_qt_loop()
    return _QT_APPLICATION


def _qt5_plot_widget_class() -> type[Any]:
    """Lazily construct and cache the concrete PyQt5 QWidget subclass."""

    global _QT_WIDGET_CLASS
    if _QT_WIDGET_CLASS is not None:
        return _QT_WIDGET_CLASS
    modules = _load_qt5_modules()
    QtCore, QtWidgets = modules.QtCore, modules.QtWidgets
    pointer_cancel_events = frozenset(
        event_type
        for event_type in (
            getattr(QtCore.QEvent, "Hide", None),
            getattr(QtCore.QEvent, "WindowDeactivate", None),
            getattr(QtCore.QEvent, "ApplicationDeactivate", None),
            getattr(QtCore.QEvent, "UngrabMouse", None),
        )
        if event_type is not None
    )
    interaction_events = frozenset(
        event_type
        for event_type in (
            getattr(QtCore.QEvent, "MouseButtonPress", None),
            getattr(QtCore.QEvent, "MouseButtonRelease", None),
            getattr(QtCore.QEvent, "MouseButtonDblClick", None),
            getattr(QtCore.QEvent, "MouseMove", None),
            getattr(QtCore.QEvent, "Wheel", None),
            getattr(QtCore.QEvent, "KeyPress", None),
            getattr(QtCore.QEvent, "KeyRelease", None),
        )
        if event_type is not None
    )

    class _RasterPixelRatioObserver(QtCore.QObject):
        """Observe the actual DPR of one visible QWidget hierarchy."""

        def __init__(
            self,
            host: object,
            on_change: Callable[[float], None],
        ) -> None:
            super().__init__(host)
            self._host = host
            self._on_change: Callable[[float], None] | None = on_change
            self._window_handle = None
            self._observed_screen_ids: set[int] = set()
            self._observed_screens: list[object] = []
            self._last_ratio: float | None = None
            self._bind_scheduled = False
            self._bind_attempts = 0
            self._detached = False
            host.installEventFilter(self)

        @property
        def current_ratio(self) -> float:
            if self._detached or self._host is None:
                raise RuntimeError("pixel-ratio observer is detached")
            top = self._host.window()
            source = self._host if top is None else top
            ratio = float(source.devicePixelRatioF())
            return ratio if math.isfinite(ratio) and ratio > 0.0 else 1.0

        def refresh(self, *, force: bool = False) -> None:
            if self._detached:
                return
            ratio = self.current_ratio
            if not force and ratio == self._last_ratio:
                return
            self._last_ratio = ratio
            callback = self._on_change
            if callback is not None:
                callback(ratio)

        def schedule_bind(self, *, reset: bool = False, delay_ms: int = 0) -> None:
            if self._detached or self._bind_scheduled:
                return
            if reset:
                self._bind_attempts = 0
            self._bind_scheduled = True
            QtCore.QTimer.singleShot(delay_ms, self._bind)

        def _native_window_handle(self, host: object) -> object | None:
            top = host.window()
            if top is None:
                return None
            try:
                top.ensurePolished()
                if top.windowHandle() is None:
                    top.winId()
                return top.windowHandle()
            except RuntimeError:
                return None

        def _bind(self) -> None:
            self._bind_scheduled = False
            host = self._host
            if self._detached or host is None or not host.isVisible():
                return
            handle = self._native_window_handle(host)
            if handle is None:
                if self._bind_attempts >= _QT_WINDOW_BIND_MAX_ATTEMPTS:
                    return
                self._bind_attempts += 1
                self.schedule_bind(delay_ms=_QT_WINDOW_BIND_RETRY_INTERVAL_MS)
                return
            self._bind_attempts = 0
            if handle is not self._window_handle:
                previous = self._window_handle
                self._window_handle = handle
                if previous is not None:
                    try:
                        previous.screenChanged.disconnect(self._screen_changed)
                    except (RuntimeError, TypeError):
                        pass
                handle.screenChanged.connect(self._screen_changed)
            self._observe_screen(handle.screen())
            self.refresh()

        def _observe_screen(self, screen: object) -> None:
            if screen is None or id(screen) in self._observed_screen_ids:
                return
            self._observed_screen_ids.add(id(screen))
            self._observed_screens.append(screen)
            for name in (
                "logicalDotsPerInchChanged",
                "physicalDotsPerInchChanged",
            ):
                signal = getattr(screen, name, None)
                if signal is not None:
                    signal.connect(self._screen_metric_changed)

        def _screen_metric_changed(self, *_args: object) -> None:
            self.refresh()

        def _screen_changed(self, screen: object) -> None:
            if self._detached:
                return
            self._observe_screen(screen)
            QtCore.QTimer.singleShot(0, self.refresh)

        def eventFilter(self, watched: object, event: object) -> bool:
            if (
                not self._detached
                and watched is self._host
                and event.type() == QtCore.QEvent.Show
            ):
                self.schedule_bind(reset=True)
            return super().eventFilter(watched, event)

        def detach(self) -> None:
            if self._detached:
                return
            host = self._host
            if host is not None and QtCore.QThread.currentThread() != host.thread():
                raise RuntimeError("pixel-ratio observer is GUI-thread affine")
            self._detached = True
            self._bind_scheduled = False
            self._bind_attempts = 0
            handle, self._window_handle = self._window_handle, None
            if handle is not None:
                try:
                    handle.screenChanged.disconnect(self._screen_changed)
                except (RuntimeError, TypeError):
                    pass
            for screen in self._observed_screens:
                for name in (
                    "logicalDotsPerInchChanged",
                    "physicalDotsPerInchChanged",
                ):
                    signal = getattr(screen, name, None)
                    if signal is not None:
                        try:
                            signal.disconnect(self._screen_metric_changed)
                        except (RuntimeError, TypeError):
                            pass
            self._observed_screens.clear()
            self._observed_screen_ids.clear()
            if host is not None:
                try:
                    host.removeEventFilter(self)
                except RuntimeError:
                    pass
            self._host = None
            self._on_change = None

    class _Qt5PlotWidget(QtWidgets.QWidget):
        """QImage-only frontend over a serial worker-owned plot session."""

        _front_ready = QtCore.pyqtSignal()
        _gesture_ready = QtCore.pyqtSignal(object)
        _invoke_ready = QtCore.pyqtSignal(object)
        _error_ready = QtCore.pyqtSignal(str)
        errorOccurred = QtCore.pyqtSignal(str)
        surfaceChanged = QtCore.pyqtSignal(object)

        def __init__(
            self,
            host: RasterPlotHost,
            parent: object | None = None,
            *,
            auto_present: bool = True,
        ) -> None:
            if not isinstance(host, RasterPlotHost):
                raise TypeError(
                    "Qt5PlotWidget requires a RasterPlotHost constructed from "
                    "a worker-side session factory"
                )
            if not isinstance(auto_present, bool):
                raise TypeError("auto_present must be a boolean")
            ensure_qt5_application()
            _register_qt5_font(modules.QtGui)
            _enable_ipython_qt_loop()
            super().__init__(parent)
            self._host = host
            self._closed = False
            self._interaction_gate = _InteractionGate()
            self._dispatch_lock = threading.RLock()
            self._pending_dispatch: set[Future[None]] = set()
            self._front: RasterFront | None = None
            self._front_handoff_lock = threading.Lock()
            self._queued_front: RasterFront | None = None
            self._front_signal_pending = False
            self._prepared_image: tuple[bytes, object] | None = None
            self._interaction_window = None
            self._gesture_front: RasterFront | None = None
            self._gesture_axes: AxisTransform | None = None
            self._gesture_kind: SceneKind | None = None
            self._candidate: SelectorState | ColorLimitCandidate | None = None
            self._pointer_button: int | None = None
            self._requested_dpr: float | None = None
            self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setMouseTracking(True)
            self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._front_ready.connect(
                self._accept_queued_front,
                type=QtCore.Qt.QueuedConnection,
            )
            self._gesture_ready.connect(
                self._finish_pointer,
                type=QtCore.Qt.QueuedConnection,
            )
            self._invoke_ready.connect(
                self._invoke_callback,
                type=QtCore.Qt.QueuedConnection,
            )
            self._error_ready.connect(
                self._emit_error,
                type=QtCore.Qt.QueuedConnection,
            )
            self._unsubscribe: Callable[[], None] | None = None
            self._pixel_ratio_observer: _RasterPixelRatioObserver | None = None
            try:
                if auto_present:
                    self._unsubscribe = self._host.subscribe_front(self._on_front)
                initial_front = self._host.front
                if initial_front is not None:
                    self._install_front(initial_front)
            except Exception:
                self.close_adapter()
                raise

        @property
        def host(self) -> RasterPlotHost:
            return self._host

        @property
        def presented_front(self) -> RasterFront | None:
            """The immutable front currently visible in this widget."""

            return self._front

        @property
        def interaction_enabled(self) -> bool:
            """Whether this adapter forwards native input to the plot worker."""

            return self._interaction_gate.enabled

        def set_interaction_enabled(self, enabled: bool) -> None:
            """Enable or suspend plot input without rebuilding the raster view."""

            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError(
                    "set_interaction_enabled must run on the Qt widget owner thread"
                )
            if self._closed:
                raise RuntimeError("Qt5PlotWidget is closed")
            if not self._interaction_gate.set_enabled(enabled):
                return
            if enabled:
                self.setFocusPolicy(QtCore.Qt.StrongFocus)
            else:
                self._cancel_active_interaction()
                self.clearFocus()
                self.setFocusPolicy(QtCore.Qt.NoFocus)

        def present_front(self, front: RasterFront) -> bool:
            """Atomically install one compatible front on the Qt owner thread.

            A front from a different host is a wiring mistake and says so.  A
            front whose surface has MOVED is not: an operator zooming, a panel
            re-specified when its measurement restarted, anything that happens
            between staging a front and presenting it leaves the staged pixels
            describing a surface that no longer exists.  That is an ordinary
            race -- the host's own newer front paints instead -- so it is
            refused by returning False, the same answer a stale revision gets.
            Raising made every zoom-during-restart shout at the operator.

            Data revision may lag the worker's latest revision so an
            application can coordinate presentation across independent hosts.
            """

            if not isinstance(front, RasterFront):
                raise TypeError("front must be a RasterFront")
            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError(
                    "present_front must run on the Qt widget owner thread"
                )
            if self._closed:
                raise RuntimeError("Qt5PlotWidget is closed")
            if front.identity.host_id != self._host.host_id:
                raise ValueError("front belongs to a different RasterPlotHost")
            latest = self._host.front
            if latest is None:
                raise RuntimeError("RasterPlotHost has no current front")
            if not front.identity.same_surface(latest.identity):
                return False
            return self._install_front(front)

        def close_adapter(self) -> None:
            if self._closed:
                return
            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError("close_adapter must run on the Qt owner thread")
            with self._dispatch_lock:
                if self._closed:
                    return
                self._closed = True
                pending_dispatch = tuple(self._pending_dispatch)
                self._pending_dispatch.clear()
                self._interaction_gate.set_enabled(False)
            observer, self._pixel_ratio_observer = self._pixel_ratio_observer, None
            interaction_window, self._interaction_window = self._interaction_window, None
            unsubscribe, self._unsubscribe = self._unsubscribe, None
            for completion in pending_dispatch:
                completion.cancel()
            callbacks: list[Callable[[], object]] = [self._cancel_active_interaction]
            if observer is not None:
                callbacks.append(observer.detach)
            if interaction_window is not None and interaction_window is not self:
                callbacks.append(lambda: interaction_window.removeEventFilter(self))
            if unsubscribe is not None:
                callbacks.append(unsubscribe)
            try:
                _complete_cleanup(tuple(callbacks))
            finally:
                with self._front_handoff_lock:
                    self._queued_front = None
                    self._front_signal_pending = False
                self._front = None
                self._prepared_image = None

        def dispatch(self, callback: Callable[[], None]) -> Future[None]:
            """Marshal an application callback to the Qt owner thread."""

            if not callable(callback):
                raise TypeError("callback must be callable")
            completion: Future[None] = Future()
            with self._dispatch_lock:
                if self._closed:
                    completion.set_exception(RuntimeError("Qt5PlotWidget is closed"))
                    return completion
                self._pending_dispatch.add(completion)
            try:
                self._invoke_ready.emit((callback, completion))
            except Exception as error:
                with self._dispatch_lock:
                    self._pending_dispatch.discard(completion)
                    if not completion.done():
                        completion.set_exception(error)
            return completion

        @QtCore.pyqtSlot(object)
        def _invoke_callback(self, request: object) -> None:
            callback, completion = request
            with self._dispatch_lock:
                if self._closed:
                    self._pending_dispatch.discard(completion)
                    completion.cancel()
                    return
                if not completion.set_running_or_notify_cancel():
                    self._pending_dispatch.discard(completion)
                    return
                try:
                    callback()
                except Exception as error:
                    completion.set_exception(error)
                else:
                    completion.set_result(None)
                finally:
                    self._pending_dispatch.discard(completion)

        def _on_front(self, front: RasterFront) -> None:
            emit = False
            with self._front_handoff_lock:
                if self._closed:
                    return
                self._queued_front = front
                if not self._front_signal_pending:
                    self._front_signal_pending = True
                    emit = True
            if emit:
                self._front_ready.emit()

        @QtCore.pyqtSlot()
        def _accept_queued_front(self) -> None:
            with self._front_handoff_lock:
                front = self._queued_front
                self._queued_front = None
                self._front_signal_pending = False
            if front is not None:
                latest = self._host.front
                if (
                    latest is not None
                    and front.identity.host_id == self._host.host_id
                    and front.identity.same_surface(latest.identity)
                ):
                    self._install_front(front)

        def _install_front(self, front: RasterFront) -> bool:
            if self._closed:
                return False
            current = self._front
            if (
                current is not None
                and front.identity.sequence <= current.identity.sequence
            ):
                return False
            gesture_front = self._gesture_front
            gesture_axes = self._gesture_axes
            if (
                gesture_front is not None
                and front.interaction.facet_focus_index
                != gesture_front.interaction.facet_focus_index
            ):
                # A focus transition REPLACES the interactive surface while
                # the double-click that requested it is still in flight: the
                # axes the press captured are gone by design -- that is what
                # the layout change means -- so the focused front supersedes
                # the gesture's expectation instead of being dropped for
                # mismatching it.  On static data no further front would
                # ever arrive to break the deadlock.
                self._clear_interaction()
                return self._promote_front(front)
            if gesture_front is not None and gesture_axes is not None:
                surface_compatible = (
                    front.identity.kind == gesture_front.identity.kind
                    and front.identity.preset == gesture_front.identity.preset
                    and front.identity.display_revision
                    == gesture_front.identity.display_revision
                    and front.identity.layout_revision
                    == gesture_front.identity.layout_revision
                    and front.logical_size == gesture_front.logical_size
                    and math.isclose(
                        front.device_pixel_ratio,
                        gesture_front.device_pixel_ratio,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
                replacement_axes = next(
                    (
                        candidate
                        for candidate in front.interaction.axes
                        if candidate.role == gesture_axes.role
                        and candidate.cell_index == gesture_axes.cell_index
                    ),
                    None,
                )
                if not surface_compatible or replacement_axes is None:
                    self._cancel_active_interaction()
                elif replacement_axes != gesture_axes:
                    if self._pointer_button == 2 and self._candidate is None:
                        # Pan input remains mapped through the immutable press
                        # transform, while each transient viewport front is
                        # allowed to become visible immediately.
                        return self._promote_front(front)
                    return False
                elif isinstance(self._candidate, ColorLimitCandidate):
                    limits = front.interaction.color_limits
                    value = self._candidate.value
                    if limits is None or not (
                        math.isclose(
                            limits.low,
                            value.low,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-15,
                        )
                        and math.isclose(
                            limits.high,
                            value.high,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-15,
                        )
                    ):
                        return False
                if surface_compatible and replacement_axes is not None:
                    self._gesture_front = front
                    self._gesture_axes = replacement_axes
            return self._promote_front(front)

        def _promote_front(self, front: RasterFront) -> bool:
            current = self._front
            pixels = front.buffer.pixels
            image = modules.QtGui.QImage(
                pixels,
                front.buffer.width,
                front.buffer.height,
                front.buffer.width * 4,
                modules.QtGui.QImage.Format_RGBA8888,
            )
            if image.isNull():
                self.errorOccurred.emit("Qt rejected the immutable RGBA raster front")
                return False
            # QImage borrows the Python buffer.  Promote them as one owner so
            # replacing a front can never briefly leave the old QImage backed
            # by already-released bytes.
            self._prepared_image = (pixels, image)
            self._front = front
            width, height = front.logical_size
            if self.width() != width or self.height() != height:
                self.setFixedSize(width, height)
            if self._pixel_ratio_observer is None:
                self._requested_dpr = front.device_pixel_ratio
                self._pixel_ratio_observer = _RasterPixelRatioObserver(
                    self,
                    self._apply_device_pixel_ratio,
                )
                self._apply_device_pixel_ratio(
                    self._pixel_ratio_observer.current_ratio
                )
            self.update()
            surface_changed = (
                current is None
                or current.identity.kind != front.identity.kind
                or current.identity.preset != front.identity.preset
                or current.identity.layout_revision
                != front.identity.layout_revision
                or current.logical_size != front.logical_size
                or not math.isclose(
                    current.device_pixel_ratio,
                    front.device_pixel_ratio,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
            if surface_changed:
                self.surfaceChanged.emit(front.identity)
            return True

        def _track(self, future: Future[object]) -> None:
            def completed(value: Future[object]) -> None:
                try:
                    value.result()
                except CancelledError:
                    return
                except Exception as error:
                    if not self._closed:
                        self._error_ready.emit(str(error))

            future.add_done_callback(completed)

        @QtCore.pyqtSlot(str)
        def _emit_error(self, message: str) -> None:
            if not self._closed:
                self.errorOccurred.emit(message)

        def closeEvent(self, event: object) -> None:
            self.close_adapter()
            super().closeEvent(event)

        def showEvent(self, event: object) -> None:
            super().showEvent(event)
            self.update()
            window = self.window()
            if window is self._interaction_window:
                return
            previous, self._interaction_window = self._interaction_window, window
            if previous is not None and previous is not self:
                try:
                    previous.removeEventFilter(self)
                except RuntimeError:
                    pass
            if window is not self:
                window.installEventFilter(self)

        def event(self, event: object) -> bool:
            if event.type() in pointer_cancel_events:
                self._cancel_active_interaction()
            if (
                not self._interaction_gate.enabled
                and event.type() in interaction_events
            ):
                event.ignore()
                return False
            return super().event(event)

        def eventFilter(self, watched: object, event: object) -> bool:
            if (
                watched is self._interaction_window
                and event.type() in pointer_cancel_events
            ):
                self._cancel_active_interaction()
            return super().eventFilter(watched, event)

        def _apply_device_pixel_ratio(self, ratio: float) -> None:
            if self._closed:
                return
            selected = float(ratio)
            if self._requested_dpr == selected:
                return
            self._requested_dpr = selected
            if self._pointer_button is not None or self._gesture_front is not None:
                self._cancel_active_interaction()
            future = self._host.set_device_pixel_ratio(selected)
            if self._unsubscribe is None:
                # A staged widget has no front subscription, but a DPR change
                # is this widget's own surface event: hand the re-rendered
                # front through the queued handoff auto-present uses.  The
                # front carries the same data revision at the new backing
                # density, so cross-host shot coordination is unaffected.
                def hand_over(done: Future[object]) -> None:
                    if done.cancelled() or done.exception() is not None:
                        return
                    front = getattr(done.result(), "front", None)
                    if front is not None and not self._closed:
                        self._on_front(front)

                future.add_done_callback(hand_over)
            self._track(future)

        def paintEvent(self, event: object) -> None:
            painter = modules.QtGui.QPainter(self)
            try:
                painter.fillRect(self.rect(), modules.QtGui.QColor("white"))
                prepared = self._prepared_image
                if prepared is not None:
                    image = prepared[1]
                    painter.drawImage(
                        QtCore.QRectF(self.rect()),
                        image,
                        QtCore.QRectF(
                            0.0,
                            0.0,
                            float(image.width()),
                            float(image.height()),
                        ),
                    )
            finally:
                painter.end()

        def _normalized(self, event: object) -> tuple[float, float]:
            position = getattr(event, "position", None)
            if callable(position):
                point = position()
                x, y = float(point.x()), float(point.y())
            else:
                x, y = float(event.x()), float(event.y())
            return (
                x / max(1, self.width()),
                y / max(1, self.height()),
            )

        @staticmethod
        def _button_number(button: object) -> int | None:
            return {
                QtCore.Qt.LeftButton: 1,
                QtCore.Qt.MiddleButton: 2,
                QtCore.Qt.RightButton: 3,
            }.get(button)

        def _submit_pointer(
            self,
            action: str,
            event: object | None = None,
            *,
            button: int | None = None,
            double: bool = False,
            step: float = 0.0,
            key: str | None = None,
        ) -> None:
            if not self._interaction_gate.enabled:
                return
            gate_generation = self._interaction_gate.generation
            x, y = (0.0, 0.0) if event is None else self._normalized(event)
            source_front = (
                self._gesture_front
                if action in {"move", "release"}
                and self._gesture_front is not None
                else self._front
            )
            source_axes = (
                self._gesture_axes
                if action in {"move", "release"}
                and self._gesture_axes is not None
                else None
            )
            if source_axes is None and source_front is not None and event is not None:
                source_axes = next(
                    (
                        item
                        for item in source_front.interaction.axes
                        if item.bounds[0] <= x <= item.bounds[2]
                        and item.bounds[1] <= y <= item.bounds[3]
                    ),
                    None,
                )
            if action == "press":
                self._gesture_front = source_front
                self._gesture_axes = source_axes
            future = self._host._pointer_event(
                action,
                x,
                y,
                button=button,
                double=double,
                step=step,
                key=key,
                identity=(None if source_front is None else source_front.identity),
                axes=source_axes,
                interaction=(
                    None if source_front is None else source_front.interaction
                ),
            )

            def completed(value: Future[object]) -> None:
                try:
                    operation = value.result()
                except CancelledError:
                    return
                except Exception as error:
                    self._gesture_ready.emit(
                        (gate_generation, action, None, None, str(error))
                    )
                else:
                    self._gesture_ready.emit(
                        (
                            gate_generation,
                            action,
                            operation.value,
                            operation.front,
                            None,
                        )
                    )

            future.add_done_callback(completed)

        @QtCore.pyqtSlot(object)
        def _finish_pointer(self, result: object) -> None:
            gate_generation, action, state, operation_front, error = result
            if gate_generation != self._interaction_gate.generation:
                return
            if error is not None:
                active = (
                    self._pointer_button is not None
                    or self._gesture_front is not None
                    or self._candidate is not None
                )
                self._clear_interaction()
                if active:
                    self._track(
                        self._host._pointer_event(
                            "cancel",
                            0.0,
                            0.0,
                        )
                    )
                self.errorOccurred.emit(str(error))
                return
            candidate = state.candidate
            role = state.role
            cell_index = state.cell_index
            active_pan = state.active_pan
            if action in {"release", "cancel", "key"}:
                self._clear_interaction()
                if state.publish_front and operation_front is not None:
                    self._install_front(operation_front)
                return
            if candidate is not None:
                self._candidate = candidate
                self._gesture_kind = candidate.kind
            if state.publish_front and operation_front is not None:
                self._install_front(operation_front)
            if candidate is None or role is None:
                if action in {"release", "cancel", "key"} or not active_pan:
                    self._clear_interaction()
                elif action == "move":
                    # A pan front is already complete when it reaches this
                    # owner-thread callback.  ``update()`` may be deferred
                    # behind a burst of queued mouse events, making the
                    # visible change appear only at release.  Repaint only
                    # this cheap QImage promotion path; Matplotlib rendering
                    # remains on the raster worker and is not repeated here.
                    self.repaint()
                return
            front = self._gesture_front or self._front
            if front is None:
                self._clear_interaction()
                return
            axes = self._gesture_axes
            if (
                axes is None
                or axes.role != role
                or axes.cell_index != cell_index
            ):
                axes = next(
                    (
                        item
                        for item in front.interaction.axes
                        if item.role == role and item.cell_index == cell_index
                    ),
                    None,
                )
            if axes is None:
                self._cancel_active_interaction()
                return
            self._gesture_front = front
            self._gesture_axes = axes
            self._gesture_kind = candidate.kind
            if action == "move":
                # Candidate scenes are immutable and already resolved by the
                # shared PlotSession interaction engine.  Paint them in the
                # same owner callback as the corresponding pointer event so
                # notebook and Qt cannot diverge by one release event.
                self.repaint()
            else:
                self.update()

        def mousePressEvent(self, event: object) -> None:
            if self._closed:
                event.ignore()
                return
            button = self._button_number(event.button())
            if button is None:
                super().mousePressEvent(event)
                return
            self._pointer_button = button
            if button in (1, 2):
                self.grabMouse()
            self._submit_pointer("press", event, button=button)
            event.accept()

        def mouseDoubleClickEvent(self, event: object) -> None:
            if self._closed:
                event.ignore()
                return
            button = self._button_number(event.button())
            if button is None:
                super().mouseDoubleClickEvent(event)
                return
            self._pointer_button = button
            if button in (1, 2):
                self.grabMouse()
            self._submit_pointer(
                "press",
                event,
                button=button,
                double=True,
            )
            event.accept()

        def mouseMoveEvent(self, event: object) -> None:
            button = self._pointer_button
            if self._closed or button is None:
                super().mouseMoveEvent(event)
                return
            self._submit_pointer(
                "move",
                event,
                button=button,
            )
            event.accept()

        def mouseReleaseEvent(self, event: object) -> None:
            button = self._button_number(event.button())
            if self._closed or button is None:
                super().mouseReleaseEvent(event)
                return
            self._submit_pointer("release", event, button=button)
            self._pointer_button = None
            try:
                if self.mouseGrabber() is self:
                    self.releaseMouse()
            except RuntimeError:
                pass
            event.accept()

        def wheelEvent(self, event: object) -> None:
            if (
                self._closed
                or self._pointer_button is not None
                or not self._interaction_gate.enabled
            ):
                event.ignore()
                return
            delta = float(event.angleDelta().y())
            if delta == 0.0:
                event.ignore()
                return
            # Session scroll steps are wheel ticks; Qt reports eighths of a
            # degree (one notch = 120), so fractional trackpad deltas become
            # fractional zoom steps rather than full notches.
            self._submit_pointer("scroll", event, step=delta / 120.0)
            event.accept()

        def keyPressEvent(self, event: object) -> None:
            if event.key() == QtCore.Qt.Key_Escape:
                self._submit_pointer("key", key="escape")
                self._clear_interaction()
                event.accept()
                return
            super().keyPressEvent(event)

        def _clear_interaction(self) -> None:
            try:
                if self.mouseGrabber() is self:
                    self.releaseMouse()
            except RuntimeError:
                pass
            self._gesture_front = None
            self._gesture_axes = None
            self._gesture_kind = None
            self._candidate = None
            self._pointer_button = None
            self.update()

        def _cancel_active_interaction(self) -> None:
            active = (
                self._pointer_button is not None
                or self._gesture_front is not None
                or self._candidate is not None
            )
            self._clear_interaction()
            if active:
                self._track(
                    self._host._pointer_event(
                        "cancel",
                        0.0,
                        0.0,
                    )
                )

    _Qt5PlotWidget.__name__ = "Qt5PlotWidget"
    _Qt5PlotWidget.__qualname__ = "Qt5PlotWidget"
    _Qt5PlotWidget.__module__ = __name__
    _QT_WIDGET_CLASS = _Qt5PlotWidget
    return _QT_WIDGET_CLASS


def __getattr__(name: str) -> object:
    """Resolve the optional Qt widget as its real class on explicit access."""

    if name != "Qt5PlotWidget":
        raise AttributeError(name)
    widget_type = _qt5_plot_widget_class()
    globals()[name] = widget_type
    return widget_type


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"Qt5PlotWidget"})


__all__ = [
    "BackendUnavailableError",
    "Qt5PlotWidget",
    "RasterPlotHost",
    "ensure_qt5_application",
]
