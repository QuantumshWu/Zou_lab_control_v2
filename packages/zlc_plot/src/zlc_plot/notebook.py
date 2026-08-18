"""Notebook raster adapter.

The kernel owns a :class:`RasterPlotHost` running with baked transient scenes:
matplotlib renders committed state and in-flight gesture candidates into every
front, and the browser is a pure frame blitter plus input normalizer.  One
renderer produces every pixel, so drag-time and committed appearance cannot
diverge.  No notebook backend is imported by this module.
"""

from __future__ import annotations

import base64
from concurrent.futures import Future
import json
import math
import struct
import threading
from typing import Any, Mapping

from ._axis_transform import AxisTransform
from .backends import BackendUnavailableError
from .raster import RasterFront, RasterOperation, RasterPlotHost
from .selectors import (
    CrosshairPoint,
    NumericRange,
    RectangleRange,
    SelectorState,
)


def _axis_to_dict(axis: AxisTransform) -> dict[str, object]:
    return {
        "role": axis.role,
        "cell_index": axis.cell_index,
        "bounds": list(axis.bounds),
        "x_limits": list(axis.x_limits),
        "y_limits": list(axis.y_limits),
        "canonical_x_limits": list(axis.canonical_x_limits),
        "canonical_y_limits": list(axis.canonical_y_limits),
    }


def _selector_state_to_dict(state: SelectorState) -> dict[str, object]:
    value = state.value
    if isinstance(value, NumericRange):
        encoded: object = [value.low, value.high]
    elif isinstance(value, RectangleRange):
        encoded = {
            "x": [value.x.low, value.x.high],
            "y": [value.y.low, value.y.high],
        }
    elif isinstance(value, CrosshairPoint):
        encoded = [value.x, value.y]
    else:
        encoded = float(value)
    return {
        "kind": state.kind.value,
        "value": encoded,
        "facet_index": state.facet_index,
    }


def _front_packet(front: RasterFront) -> bytes:
    """Encode one complete raster front for one widget buffer transaction.

    A notebook comm does not guarantee that separate synced traits are observed
    together in one browser turn.  Keeping the dimensions, interaction map and
    RGBA bytes behind one ``Bytes`` trait gives the browser a single frame
    authority.
    """

    header = json.dumps(
        {
            "width": front.buffer.width,
            "height": front.buffer.height,
            "logical_width": front.logical_size[0],
            "logical_height": front.logical_size[1],
            "logical_dpi": front.logical_dpi,
            "device_pixel_ratio": front.device_pixel_ratio,
            "sequence": front.identity.sequence,
            "axes": [_axis_to_dict(axis) for axis in front.interaction.axes],
            "selectors": [
                _selector_state_to_dict(state)
                for state in front.interaction.selectors
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack("<I", len(header)) + header + front.buffer.pixels


_WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"


def _snapshot_display_data(
    front: RasterFront,
) -> tuple[dict[str, object], dict[str, object]]:
    """Encode one front as persistable raster display data.

    A closed widget model has no live views, and a reopened Notebook cannot
    resolve any widget model; only a raster snapshot keeps the frame visible
    in the saved output.
    """

    logical_width, logical_height = front.logical_size
    data: dict[str, object] = {
        "image/png": base64.b64encode(front.buffer.encode("PNG")).decode("ascii"),
        "text/plain": f"zlc_plot raster {logical_width}x{logical_height}",
    }
    metadata: dict[str, object] = {
        "image/png": {"width": logical_width, "height": logical_height},
    }
    return data, metadata


_WIDGET_CLASS: type[Any] | None = None

# Last browser-reported device pixel ratio.  A NotebookView cannot know its
# browser's density before its widget view renders, so later views adopt the
# most recent report to publish a crisp first front instead of a ratio-1
# frame that visibly re-renders once their own report arrives.
_LAST_DEVICE_PIXEL_RATIO: float | None = None

def _widget_class() -> type[Any]:
    global _WIDGET_CLASS
    if _WIDGET_CLASS is not None:
        return _WIDGET_CLASS
    try:
        import anywidget
        from traitlets import Bytes
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailableError(
            "NotebookView requires anywidget; install zlc-plot[notebook]."
        ) from error

    class RasterWidget(anywidget.AnyWidget):
        # AnyWidget owns the standard JupyterLab/Notebook module registry.  A
        # custom ``_view_module`` injected with ``Javascript`` is not enough
        # for JupyterLab 4, whose manager resolves views from its registry.
        _esm = _WIDGET_ESM
        frame_packet = Bytes(b"").tag(sync=True)

    _WIDGET_CLASS = RasterWidget
    return RasterWidget


_WIDGET_ESM = r"""
export default {
  render({ model, el }) {
    const view = new ZlcRasterView(model, el);
    view.render();
    return () => view.dispose();
  },
};

class ZlcRasterView {
  constructor(model, el) {
    this.model = model;
    this.el = el;
    this._listeners = [];
    this._cleanups = [];
  }
  _listen(name, callback) {
    this.model.on(name, callback);
    this._listeners.push([name, callback]);
  }
  dispose() {
    for (const [name, callback] of this._listeners) this.model.off(name, callback);
    this._listeners = [];
    for (const cleanup of this._cleanups.splice(0)) {
      try { cleanup(); } catch (_) {}
    }
  }
  render() {
    // This view is a frame blitter and input normalizer, nothing more.  The
    // kernel session renders EVERYTHING with matplotlib -- committed state
    // and transient gesture candidates alike (the host runs with
    // frontend_transient_scenes=False) -- so what the browser shows during a
    // drag and after a commit comes from one renderer and cannot diverge.
    this.el.classList.add('zlc-raster-root');
    // Opt out of JupyterLab's context menu so a right press reaches the
    // kernel crosshair gesture exactly as it reaches the Qt widget.
    this.el.setAttribute('data-jp-suppress-context-menu', '');
    // ipywidgets rewrites Layout-managed inline styles on `el`
    // asynchronously; all presentation styling lives on an owned wrapper
    // element that no LayoutView ever touches.
    this.surface = document.createElement('div');
    this.surface.style.position = 'relative';
    this.surface.style.display = 'inline-block';
    this.el.appendChild(this.surface);
    this.canvas = document.createElement('canvas');
    this.canvas.style.display = 'block';
    this.canvas.style.touchAction = 'none';
    this.canvas.style.userSelect = 'none';
    this.surface.appendChild(this.canvas);
    this._dragging = false;
    this._button = null;
    this._lastMoveSent = 0;
    this._pressChain = {time: 0, x: 0, y: 0, button: null};
    this._paintPending = false;
    this._frame = null;
    // Hold a mismatched-density first frame briefly: the kernel republishes
    // at the reported ratio moments after the environment message lands, and
    // painting the ratio-1 frame first reads as a blurry-to-sharp jump.
    this._expectedRatio = window.devicePixelRatio || 1;
    this._graceDeadline = performance.now() + 500;
    this._pendingFrame = null;
    this._graceTimer = null;
    this._bind();
    this._listen('change:frame_packet', () => {
      this._acceptFrame(this._decodeFrame(this.model.get('frame_packet')));
    });
    this._watchEnvironment();
    this._acceptFrame(this._decodeFrame(this.model.get('frame_packet')));
  }
  _watchEnvironment() {
    // Report the real pixel density so the kernel rasterizes at native
    // resolution, the same environment feed the Qt widget provides
    // through host.set_device_pixel_ratio on screen changes.
    const report = () => this.model.send({
      type: 'environment',
      device_pixel_ratio: window.devicePixelRatio || 1,
    });
    report();
    const arm = () => {
      if (!window.matchMedia) return;
      const query = window.matchMedia('(resolution: ' + window.devicePixelRatio + 'dppx)');
      const onChange = () => { this._envCleanup = null; report(); arm(); };
      query.addEventListener('change', onChange, {once: true});
      this._envCleanup = () => query.removeEventListener('change', onChange);
    };
    arm();
    this._cleanups.push(() => { if (this._envCleanup) this._envCleanup(); });
  }
  _acceptFrame(frame) {
    if (!frame) return;
    if (this._frame && frame.sequence < this._frame.sequence) return;
    if (!this._frame && frame.device_pixel_ratio !== this._expectedRatio &&
        performance.now() < this._graceDeadline) {
      this._pendingFrame = frame;
      if (!this._graceTimer) {
        this._graceTimer = setTimeout(() => {
          this._graceTimer = null;
          if (!this._frame && this._pendingFrame) {
            this._frame = this._pendingFrame; this._pendingFrame = null;
            this._schedulePaint();
          }
        }, Math.max(16, this._graceDeadline - performance.now()));
        this._cleanups.push(() => { if (this._graceTimer) clearTimeout(this._graceTimer); });
      }
      return;
    }
    this._pendingFrame = null;
    this._frame = frame;
    this._schedulePaint();
  }
  _schedulePaint() {
    if (this._paintPending) return;
    this._paintPending = true;
    const paint = () => { this._paintPending = false; this._paint(); };
    if (window.requestAnimationFrame) window.requestAnimationFrame(paint);
    else window.setTimeout(paint, 0);
  }
  _bytes(value) {
    // Jupyter's buffer serializer stores Bytes traits as DataView.  Keep
    // the exact byte window instead of coercing a DataView to zero bytes.
    if (value instanceof Uint8Array) return value;
    if (value instanceof ArrayBuffer) return new Uint8Array(value);
    if (ArrayBuffer.isView(value)) {
      return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    }
    if (typeof value === 'string') {
      const raw = atob(value); const result = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) result[i] = raw.charCodeAt(i);
      return result;
    }
    return new Uint8Array(value || []);
  }
  _decodeFrame(value) {
    const bytes = this._bytes(value);
    if (bytes.length < 4) return null;
    const headerSize = (bytes[0] | (bytes[1] << 8) |
      (bytes[2] << 16) | (bytes[3] << 24)) >>> 0;
    if (headerSize <= 0 || 4 + headerSize > bytes.length) return null;
    let header;
    try {
      const raw = new TextDecoder('utf-8').decode(bytes.slice(4, 4 + headerSize));
      header = JSON.parse(raw);
    } catch (_) { return null; }
    const width = Number(header.width), height = Number(header.height);
    const rgba = bytes.slice(4 + headerSize);
    if (!Number.isInteger(width) || !Number.isInteger(height) ||
        width <= 0 || height <= 0 || rgba.length !== width * height * 4) return null;
    return {
      width, height,
      logical_width: Number(header.logical_width),
      logical_height: Number(header.logical_height),
      device_pixel_ratio: Number(header.device_pixel_ratio) || 1,
      sequence: Number(header.sequence) || 0,
      rgba,
    };
  }
  _paint() {
    const frame = this._frame;
    if (!frame) return;
    const w = frame.width, h = frame.height;
    this.canvas.style.width = frame.logical_width + 'px';
    this.canvas.style.height = frame.logical_height + 'px';
    this.surface.style.width = frame.logical_width + 'px';
    this.surface.style.height = frame.logical_height + 'px';
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
    }
    this.canvas.getContext('2d').putImageData(
      new ImageData(new Uint8ClampedArray(frame.rgba), w, h), 0, 0);
  }
  _normalized(e) {
    // A zero-size rect (hidden panel, mid-layout) yields NaN, which JSON
    // serializes as null and the kernel cannot coerce.
    const rect = this.canvas.getBoundingClientRect();
    const rawX = ((e.clientX || 0) - rect.left) / (rect.width || 1);
    const rawY = ((e.clientY || 0) - rect.top) / (rect.height || 1);
    return {
      x: Math.max(0, Math.min(1, Number.isFinite(rawX) ? rawX : 0)),
      y: Math.max(0, Math.min(1, Number.isFinite(rawY) ? rawY : 0)),
    };
  }
  _send(action, e, fields) {
    const point = this._normalized(e);
    this.model.send(Object.assign({
      type: 'pointer', action: action, x: point.x, y: point.y,
      button: null, double: false, step: 0, key: null,
    }, fields || {}));
  }
  _isDouble(e, button) {
    // The kernel's timed click chain covers middle/right buttons only; a
    // LEFT double relies on the explicit flag (Qt's MouseButtonDblClick),
    // and PointerEvent.detail carries no click count, so the view
    // synthesizes the chain.
    const now = performance.now();
    const chain = this._pressChain;
    const double = chain.button === button &&
      now - chain.time < 400 &&
      Math.abs(e.clientX - chain.x) < 8 &&
      Math.abs(e.clientY - chain.y) < 8;
    this._pressChain = double
      ? {time: 0, x: 0, y: 0, button: null}
      : {time: now, x: e.clientX, y: e.clientY, button: button};
    return double;
  }
  _bind() {
    this.canvas.addEventListener('contextmenu', e => { e.preventDefault(); e.stopPropagation(); });
    this.canvas.addEventListener('pointerdown', e => {
      // The protocol speaks one primary pointer with buttons 1..3; extra
      // touches and X1/X2/eraser buttons are not part of any gesture.
      if (!e.isPrimary || this._dragging || e.button > 2) return;
      this._dragging = true;
      this._button = e.button + 1;
      try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
      this._send('press', e, {button: this._button, double: this._isDouble(e, this._button)});
    });
    this.canvas.addEventListener('pointermove', e => {
      if (!this._dragging || !e.isPrimary) return;
      const now = performance.now();
      if (now - this._lastMoveSent < 30) return;
      this._lastMoveSent = now;
      this._send('move', e, {button: this._button});
    });
    this.canvas.addEventListener('pointerup', e => {
      if (!this._dragging || !e.isPrimary) return;
      this._dragging = false;
      this._send('release', e, {button: this._button});
      this._button = null;
    });
    this.canvas.addEventListener('pointercancel', e => {
      if (!e.isPrimary) return;
      this._dragging = false;
      this._button = null;
      this._send('cancel', e, {});
    });
    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      // A wheel step during an active drag would cancel the kernel gesture
      // while the browser keeps dragging; horizontal scroll has no zoom.
      if (this._dragging || !e.deltaY) return;
      this._send('scroll', e, {step: e.deltaY < 0 ? 1 : -1});
    }, {passive: false});
    this.el.tabIndex = 0;
    this.el.addEventListener('keydown', e => {
      if (e.key === 'Escape') this._send('key', e, {key: 'Escape'});
    });
  }
}
"""


class NotebookView:
    """A thin anywidget adapter over the same RasterPlotHost used by Qt."""

    def __init__(self, session: "PlotSession", *, close_session_on_close: bool = False) -> None:
        from .session import PlotSession

        if not isinstance(session, PlotSession):
            raise TypeError("session must be PlotSession")
        if not isinstance(close_session_on_close, bool):
            raise TypeError("close_session_on_close must be bool")
        self._session = session
        self._close_session = close_session_on_close
        self._host = RasterPlotHost.from_session(session, close_session=close_session_on_close)
        self._pending_environment: Future[Any] | None = None
        if _LAST_DEVICE_PIXEL_RATIO is not None and _LAST_DEVICE_PIXEL_RATIO != 1.0:
            self._pending_environment = self._host.set_device_pixel_ratio(
                _LAST_DEVICE_PIXEL_RATIO
            )
        self._widget: Any | None = None
        self._display_handle: Any | None = None
        self._front: RasterFront | None = None
        self._front_release: Any | None = None
        self._closed = False
        self._gesture_front: RasterFront | None = None
        self._gesture_axes: AxisTransform | None = None
        self._pointer_serial = 0
        self._consumed_serial = -1
        self._kernel_lock = threading.RLock()
        # Latest-only front handoff: packet building and the ipywidgets comm
        # send cost milliseconds per multi-megabyte frame, and running them on
        # the raster worker (where subscribe_front callbacks execute) would
        # block the next render.  A dedicated sender drains the newest front;
        # a burst of promotions collapses to one send.
        self._send_condition = threading.Condition(threading.Lock())
        self._pending_front: RasterFront | None = None
        self._sender_stopped = False
        self._sender = threading.Thread(
            target=self._send_fronts,
            name="zlc-notebook-front-sender",
            daemon=True,
        )
        self._sender.start()

    @property
    def session(self) -> "PlotSession":
        return self._session

    @property
    def host(self) -> RasterPlotHost:
        return self._host

    def describe_display(self) -> object:
        """Return the same control-plane snapshot used by the Qt panel."""

        return self._session.describe_display()

    def describe_semantics(self) -> object:
        """Return registry-derived semantic editors for notebook controls."""

        return self._session.describe_semantics()

    def replace_spec(
        self,
        spec: "PlotSpec",
        *,
        parameters: Mapping[str, object] | None = None,
    ) -> object:
        """Replace semantic roles through the session's rebuild transaction."""

        return self._session.replace_spec(spec, parameters=parameters)

    @property
    def widget(self) -> Any:
        if self._widget is None:
            raise RuntimeError("call display() before accessing widget")
        return self._widget

    def _schedule(self, callback: Any) -> None:
        # Run publish work directly on the calling thread.  Routing through
        # ``kernel.io_loop.add_callback`` silently drops every callback on
        # ipykernel 7: the tornado wrapper still exists and its asyncio loop
        # reports running, but the anyio-driven kernel never services it, so
        # every front promotion and pointer completion dies unobserved.
        # Direct invocation is safe from any thread: ipykernel's iopub socket
        # serializes cross-thread sends through IOPubThread, `_kernel_lock`
        # guards this view's front state, and both the kernel-side and
        # browser-side frame guards drop out-of-order sequences.
        callback()

    def _publish_front(self, front: RasterFront) -> None:
        with self._kernel_lock:
            if self._closed:
                return
            if self._front is not None and front.identity.sequence <= self._front.identity.sequence:
                return
            self._front = front
            widget = self._widget
            if widget is None:
                return
            # The trait write stays inside the reentrant lock: publishes
            # arrive from the raster worker, the comm thread and the shell
            # thread, and interleaving would let an older frame land after a
            # newer one.
            widget.frame_packet = _front_packet(front)

    def _on_front(self, front: RasterFront) -> None:
        with self._send_condition:
            self._pending_front = front
            self._send_condition.notify()

    def _send_fronts(self) -> None:
        while True:
            with self._send_condition:
                while self._pending_front is None and not self._sender_stopped:
                    self._send_condition.wait()
                if self._sender_stopped:
                    return
                front = self._pending_front
                self._pending_front = None
            # ``_schedule`` documents why direct invocation is correct from
            # any thread; the sequence guard in ``_publish_front`` drops any
            # frame a newer send already superseded.
            self._schedule(lambda: self._publish_front(front))

    def _axis_for(self, front: RasterFront, x: float, y: float) -> AxisTransform | None:
        for axis in front.interaction.axes:
            left, top, right, bottom = axis.bounds
            if left <= x <= right and top <= y <= bottom:
                return axis
        return None

    def _pointer_message(self, content: Mapping[str, object]) -> None:
        with self._kernel_lock:
            return self._pointer_message_locked(content)

    def _pointer_message_locked(self, content: Mapping[str, object]) -> None:
        if self._closed or self._front is None:
            return
        self._pointer_serial += 1
        serial = self._pointer_serial
        action = str(content.get("action", "")).lower()
        # NaN coordinates serialize to JSON null; a malformed browser event
        # must not raise inside the comm callback, where ipywidgets swallows
        # the traceback and the gesture dies silently.
        raw_x, raw_y = content.get("x"), content.get("y")
        x = float(raw_x) if isinstance(raw_x, (int, float)) else 0.0
        y = float(raw_y) if isinstance(raw_y, (int, float)) else 0.0
        front = self._front
        if action == "press":
            self._gesture_front = front
            self._gesture_axes = self._axis_for(front, x, y)
        elif action in {"move", "release", "cancel"} and self._gesture_front is not None:
            front = self._gesture_front
        axes = self._gesture_axes if action in {"move", "release"} else self._axis_for(front, x, y)
        identity = front.identity if action != "cancel" else None
        interaction = front.interaction if action == "press" else None
        future = self._host._pointer_event(
            action,
            x,
            y,
            button=None if content.get("button") is None else int(content["button"]),
            double=bool(content.get("double", False)),
            step=float(content.get("step", 0.0)),
            key=None if content.get("key") is None else str(content["key"]),
            identity=identity,
            axes=axes,
            interaction=interaction,
        )
        future.add_done_callback(
            lambda completed: self._schedule(
                lambda: self._consume_pointer(completed, action, serial)
            )
        )

    def _consume_pointer(self, future: Future[Any], action: str, serial: int) -> None:
        if future.cancelled():
            # A coalesced (superseded) pointer move is routine, not a failed
            # gesture; CancelledError is a BaseException and must never reach
            # the future's callback machinery.
            return
        with self._kernel_lock:
            if serial < self._consumed_serial:
                return
            try:
                operation: RasterOperation[Any] = future.result()
            except Exception:
                if serial != self._pointer_serial:
                    return
                self._consumed_serial = serial
                self._gesture_front = None
                self._gesture_axes = None
                if self._host.front is not None:
                    self._publish_front(self._host.front)
                return
            self._consumed_serial = serial
            if operation.front is not None:
                self._publish_front(operation.front)
            if action in {"release", "cancel"}:
                self._gesture_front = None
                self._gesture_axes = None

    def _on_widget_message(self, _widget: Any, content: object, _buffers: object) -> None:
        if not isinstance(content, Mapping):
            return
        kind = content.get("type")
        if kind == "pointer":
            self._pointer_message(content)
        elif kind == "environment":
            self._environment_message(content)

    def _environment_message(self, content: Mapping[str, object]) -> None:
        """Apply the browser environment to the host, mirroring the Qt path.

        The browser is the only party that knows its device pixel ratio;
        without this report the kernel rasterizes at ratio 1 and the canvas
        upscales, which reads as blur next to a Qt window on the same
        screen.
        """

        if self._closed:
            return
        ratio = content.get("device_pixel_ratio")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            return
        selected = float(ratio)
        # Browsers report ratios in the low single digits; anything larger is
        # a malformed message that would drive enormous rasterization.
        if not math.isfinite(selected) or not (0.0 < selected <= 8.0):
            return
        global _LAST_DEVICE_PIXEL_RATIO
        _LAST_DEVICE_PIXEL_RATIO = selected
        self._host.set_device_pixel_ratio(selected)

    def display(self) -> None:
        if self._closed:
            raise RuntimeError("NotebookView is closed")
        if self._widget is not None:
            return None
        widget_class = _widget_class()
        try:
            from IPython.core.interactiveshell import InteractiveShell
            from IPython.display import display
        except (ImportError, ModuleNotFoundError) as error:
            raise BackendUnavailableError(
                "NotebookView requires IPython and ipywidgets"
            ) from error
        self._widget = widget_class()
        self._widget.on_msg(self._on_widget_message)
        initial_front = self._host.wait_for_front()
        if self._pending_environment is not None:
            # The adopted pixel ratio republishes; ship that crisp front (and
            # PNG fallback) instead of the ratio-1 startup frame.
            try:
                self._pending_environment.result(timeout=5.0)
            except Exception:
                pass
            self._pending_environment = None
            latest = self._host.front
            if latest is not None:
                initial_front = latest
        self._front_release = self._host.subscribe_front(self._on_front)
        # Let _publish_front install the first identity and send its complete
        # buffer.  Assigning _front before this call makes the monotonic guard
        # treat the initial frame as already published, leaving the widget at
        # its 1x1 empty default until the next live/fit promotion.
        self._publish_front(initial_front)
        if not InteractiveShell.initialized():
            # Without a running shell IPython's display() degrades to
            # print(); the raw bundle would dump the base64 PNG to stdout
            # there, so keep the compact widget repr on that path.
            display(self._widget)
            return None
        # Display the widget view mime directly instead of nesting the widget
        # in an Output widget: an Output nesting creates a second widget view
        # and the duplicate-output behavior this adapter is meant to avoid.
        # The bundle also carries a raster fallback so frontends without a
        # live widget model (a saved Notebook, an HTML export) show a frame,
        # and the display id lets close() replace the live view with the
        # final frame instead of leaving an empty output.
        data, metadata = _snapshot_display_data(initial_front)
        data[_WIDGET_VIEW_MIME] = {
            "version_major": 2,
            "version_minor": 0,
            "model_id": self._widget.model_id,
        }
        self._display_handle = display(
            data,
            raw=True,
            metadata=metadata,
            display_id=True,
        )
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._front_release is not None:
            self._front_release()
        self._front_release = None
        with self._send_condition:
            self._sender_stopped = True
            self._pending_front = None
            self._send_condition.notify()
        self._sender.join(timeout=5.0)
        if self._display_handle is not None and self._front is not None:
            # Closing the widget model removes every live view from the
            # Notebook, which blanks the cell output.  Replace the widget
            # output with the final frame first so the cell keeps its plot
            # after close() and across save/reopen.
            data, metadata = _snapshot_display_data(self._front)
            try:
                self._display_handle.update(data, raw=True, metadata=metadata)
            except Exception:
                # Teardown must complete even when the kernel session is
                # already shutting down and can no longer publish output.
                pass
        self._display_handle = None
        self._host.close()
        if self._widget is not None:
            close = getattr(self._widget, "close", None)
            if callable(close):
                close()
        self._widget = None

    def __enter__(self) -> "NotebookView":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


__all__ = ["NotebookView"]
