"""Browser pixels and input for the current experiment, independent of MCP."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import Future
from importlib.resources import files
from io import BytesIO
import json
import logging
import math
from queue import Empty, Queue
import secrets
import socket
import struct
import sys
from threading import Condition, Event, Thread
from time import monotonic
from urllib.parse import urlsplit

from PIL import Image


def _popup_images(description):
    images = []
    for popup in description["popups"]:
        buffer = BytesIO()
        Image.frombytes("RGBA", (popup["pixel_width"], popup["pixel_height"]), popup["rgba"]).save(
            buffer, format="PNG", compress_level=1)
        images.append(dict(id=popup["id"], rect=popup["rect"],
                           url="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")))
    return images


def start_server(flow, *, on_clients, host="0.0.0.0", port=80, fps=60):
    """Share experiment windows after local approval; return close with decide()."""
    import tornado.httpserver
    import tornado.ioloop
    import tornado.netutil
    import tornado.web
    import tornado.websocket
    import av
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
    from zlc_ui.windows import SharedWindows
    from ..board import attach_qt_owner_turn

    if not 0 <= int(port) <= 65535 or not 1 <= float(fps) <= 60:
        raise ValueError("sharing requires a valid port and 1–60 frames per second")
    if sys.platform == "win32":
        import winrt.windows.graphics.capture.interop
        import winrt.windows.graphics.directx.direct3d11.interop
    assets = {name: files("zlc_workbench").joinpath("web_share", name).read_bytes()
              for name in ("index.html", "client.js", "style.css")}
    sockets = tornado.netutil.bind_sockets(int(port), address=host)
    actual_port = sockets[0].getsockname()[1]
    surface = SharedWindows(lambda: (
        flow.devices, flow.console, *flow.device_controls.values(),
    ))
    assets["theme.css"] = (":root {" + "".join(
        f"--{key}:{value};" for key, value in surface.theme().items()
    ) + "}").encode("utf-8")
    inbox = Queue(maxsize=64)
    closed, ready = Event(), Event()
    clients = set()
    network = {}
    frame_ready = asyncio.Event()

    class WindowVideoTrack(MediaStreamTrack):
        """One capture device, one latest raw frame and one sender per browser."""

        kind = "video"

        def __init__(self):
            super().__init__()
            self.loop = asyncio.get_running_loop()
            self.condition = Condition()
            self.target = self.request = self.capture = self.delivered = None
            self.failure = None
            self.worker = None
            self.changed = asyncio.Event()

        def configure(self, description, rect, size, source, available):
            with self.condition:
                self.target = None if description is None else (description, rect, size, source, available)
                self.failure = None
                self.condition.notify()
            if self.capture is not None:
                self.capture.interrupt()
            self.changed.set()

        async def open(self):
            if sys.platform == "win32":
                self.worker = Thread(target=self._run, name="zlc-window-video", daemon=True)
                self.worker.start()

        def _finish(self, future, packet, source):
            if not future.done():
                future.set_result((packet, source))

        def _run(self):
            from fractions import Fraction
            from zlc_ui._window_capture import WindowCapture

            capture = codec = None
            active = failed = None
            codec_size = None
            first = False
            previous_pts = -1
            try:
                capture = self.capture = WindowCapture(fps=fps)
                while self.readyState == "live":
                    with self.condition:
                        while (self.readyState == "live" and self.target is active
                               and (self.target is None or self.request is None or self.target is failed)):
                            self.condition.wait()
                        if self.readyState != "live":
                            break
                        target, future = self.target, self.request
                    try:
                        if target is not active:
                            active, failed, first = target, None, True
                            if target is None:
                                capture.clear()
                                continue
                            description, rect, size, source, available = target
                            root, dpr = description["root_rect"], description["native_dpr"]
                            crop = tuple(round((rect[i] - root[i]) * dpr) for i in (0, 1))
                            crop += tuple(round(value * dpr) for value in rect[2:])
                            capture.select(description["id"], description["hwnd"], crop, available, size)
                            if size != codec_size:
                                codec = av.CodecContext.create("libx264", "w")
                                codec.width, codec.height = size
                                codec.pix_fmt = "nv12"
                                codec.time_base = Fraction(1, 90000)
                                codec.framerate = Fraction(int(fps), 1)
                                codec.bit_rate = 20_000_000
                                codec.gop_size, codec.max_b_frames = max(1, int(fps) // 2), 0
                                codec.options = {"preset": "ultrafast", "tune": "zerolatency",
                                                 "profile": "baseline", "x264-params": "repeat-headers=1"}
                                codec.colorspace, codec.color_range = 1, 1
                                codec.color_primaries, codec.color_trc = 1, 1
                                codec.open()
                                codec_size = size
                        if future is None:
                            continue
                        with capture.frame() as captured:
                            if captured is None or target is not self.target:
                                continue
                            pixels, pts, pixel_format = captured
                            if not first and pts // (90000 / float(fps)) <= previous_pts // (90000 / float(fps)):
                                continue
                            image = av.VideoFrame.from_numpy_buffer(pixels, format=pixel_format, width=size[0] if pixel_format == "nv12" else pixels.shape[1])
                            if pixel_format == "nv12":
                                image.colorspace, image.color_range = 1, 1
                            image = image.reformat(width=size[0], height=size[1], format="nv12",
                                                   dst_colorspace="ITU709", dst_color_range="MPEG")
                            image.colorspace, image.color_range = 1, 1
                            image.pts, image.time_base = max(previous_pts + 1, pts), Fraction(1, 90000)
                            if first:
                                image.pict_type = av.video.frame.PictureType.I
                            packets = codec.encode(image)
                            if len(packets) != 1:
                                raise RuntimeError("The zero-latency video encoder did not return one complete frame")
                            packet = packets[0]
                            previous_pts, first = image.pts, False
                        with self.condition:
                            if self.request is future:
                                self.request = None
                        self.loop.call_soon_threadsafe(self._finish, future, packet, source)
                    except Exception as error:
                        self.failure, failed = (target[3], str(error)), target
                        capture.clear()
            except Exception as error:
                source = self.target[3] if self.target is not None else None
                self.failure = (source, str(error))
                self.loop.call_soon_threadsafe(self.stop)
            finally:
                self.capture = None
                if capture is not None:
                    capture.close()
                codec = None

        async def recv(self):
            if sys.platform != "win32":
                import numpy as np
                from fractions import Fraction
                while self.target is None and self.readyState == "live":
                    self.changed.clear()
                    await self.changed.wait()
                if self.readyState != "live":
                    raise MediaStreamError
                description, rect, size, source, _available = self.target
                await asyncio.sleep(1 / float(fps))
                scale = min(description["native_dpr"], size[0] / rect[2], size[1] / rect[3])
                snapshot = await owner(surface.capture, description["id"], math.ceil(description["width"] * scale), True)
                pixels = np.frombuffer(snapshot["rgba"], np.uint8).reshape(snapshot["pixel_height"], snapshot["pixel_width"], 4)
                scale = snapshot["pixel_width"] / snapshot["width"]
                x, y = (round((rect[i] + description["origin"][i] - snapshot["origin"][i]) * scale) for i in (0, 1))
                width, height = (round(value * scale) for value in rect[2:])
                image = av.VideoFrame.from_ndarray(pixels[y:y + height, x:x + width], format="rgba").reformat(width=size[0], height=size[1])
                image.pts, image.time_base = int(monotonic() * 90000), Fraction(1, 90000)
                self.delivered = (source, image.pts)
                return image
            if self.readyState != "live":
                raise MediaStreamError
            future = self.loop.create_future()
            with self.condition:
                self.request = future
                self.condition.notify()
            packet, source = await future
            self.delivered = (source, int(packet.pts * packet.time_base * 90000))
            return packet

        def stop(self):
            super().stop()
            with self.condition:
                if self.request is not None and not self.request.done():
                    self.request.set_exception(MediaStreamError())
                self.condition.notify()
            if self.capture is not None:
                self.capture.interrupt()
            self.changed.set()

        async def close(self):
            self.stop()
            if self.worker is not None:
                await asyncio.to_thread(self.worker.join)
                self.worker = None

    def drain():
        while True:
            try:
                future, function, arguments = inbox.get_nowait()
            except Empty:
                return
            if closed.is_set():
                future.cancel()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*arguments))
            except BaseException as error:
                future.set_exception(error)
            if surface.control_enabled and not closed.is_set():
                if not inbox.empty():
                    wake()
                return

    if surface.control_enabled:
        surface.owner_turn_requested.connect(drain)
        wake = surface.wake_owner
    else:
        wake = attach_qt_owner_turn(drain)

    def submit(function, *arguments):
        if closed.is_set():
            raise RuntimeError("experiment sharing is closed")
        future = Future()
        inbox.put_nowait((future, function, arguments))
        wake()
        return future

    async def owner(function, *arguments):
        future = submit(function, *arguments)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), 5.)
        except TimeoutError:
            future.cancel()
            raise RuntimeError("The experiment UI did not respond; inspect it before repeating an action") from None

    def poll_windows(requested):
        catalog = surface.catalog()
        visible = {item["id"] for item in catalog["windows"] if item["visible"]}
        frames = {identity: surface.describe(identity) for identity in requested if identity in visible}
        return catalog, frames

    addresses = {"127.0.0.1"}
    try:
        addresses.update(item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    if host not in ("0.0.0.0", "::"):
        addresses = {"127.0.0.1" if host == "localhost" else host}
    allowed_hosts = addresses | {"localhost", socket.gethostname().lower(), socket.getfqdn().lower()}

    def notify_clients():
        if not closed.is_set():
            submit(on_clients, tuple(dict(
                id=client.identity, address=client.request.remote_ip,
                browser=client.request.headers.get("User-Agent", "Browser")[:256],
                status="connected" if client.approved else "pending",
            ) for client in sorted(clients, key=lambda item: item.identity)))

    def valid_host(request):
        return (urlsplit("//" + request.host).hostname or "").lower() in allowed_hosts

    class Page(tornado.web.RequestHandler):
        def get(self, name="index.html"):
            if not valid_host(self.request) or name not in assets:
                raise tornado.web.HTTPError(404)
            self.set_header("Content-Type", {"index.html": "text/html; charset=utf-8",
                "client.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8",
                "theme.css": "text/css; charset=utf-8"}[name])
            self.set_header("Cache-Control", "no-store")
            self.set_header("Referrer-Policy", "no-referrer")
            self.set_header("X-Content-Type-Options", "nosniff")
            hostname = urlsplit("//" + self.request.host).hostname
            authority = f"[{hostname}]" if ":" in hostname else hostname
            websocket_origin = f"ws://{authority}:{actual_port}"
            self.set_header("Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
                f"media-src 'self' blob:; connect-src 'self' {websocket_origin}; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.write(assets[name])

    class Browser(tornado.websocket.WebSocketHandler):
        def check_origin(self, origin):
            source = urlsplit(origin)
            return valid_host(self.request) and source.scheme == "http" and source.netloc == self.request.host

        def open(self):
            self.set_nodelay(True)
            self.identity = secrets.token_hex(12)
            self.approved = False
            self.selection = 0
            self.selected = None
            self.panel = None
            self.viewport = None
            self.peer = self.track = None
            self.opening = None
            self.media_id = self.source_key = self.media_description = self.media_rect = self.media_size = None
            self.source = self.boundary = self.media_message = None
            self.available = ()
            self.last_media = None
            self.writing = None
            self.last_state = None
            clients.add(self)
            self.write_message(dict(type="pending", message="Waiting for approval on the experiment computer."))
            notify_clients()

        def write_message(self, message, binary=False):
            if isinstance(message, dict) and message.get("type") in {"media", "result", "error"}:
                message = {"selection": self.selection, **message}
            return super().write_message(message, binary=binary)

        def clear_source(self):
            self.source_key = self.source = self.boundary = self.last_media = self.media_message = None
            if self.track is not None:
                self.track.configure(None, None, None, None, ())

        def send_stream(self, message, *, binary=False):
            self.writing = self.write_message(message, binary=binary)
            self.writing.add_done_callback(self.stream_written)

        def stream_written(self, future):
            if not future.cancelled():
                try:
                    future.result()
                except tornado.websocket.WebSocketClosedError:
                    self.close()
                except Exception:
                    logging.getLogger(__name__).exception("Experiment sharing write failed")
                    self.close(1011, "Window transmission failed; reconnect to continue viewing.")

        async def stop_media(self, *, failed=False):
            peer, track = self.peer, self.track
            opening, self.opening = self.opening, None
            self.peer = self.track = None
            self.media_id = self.last_media = None
            if not failed:
                self.source_key = None
            if opening is not None:
                opening.cancel()
            if track is not None:
                track.stop()
            if opening is not None:
                await asyncio.gather(opening, return_exceptions=True)
            if peer is not None:
                await peer.close()
            if track is not None:
                await track.close()
            self.clear_source()

        async def offer(self, message):
            identity = message.get("stream")
            if identity != self.media_id or self.media_description is None:
                return
            track = peer = None
            try:
                track = self.track = WindowVideoTrack()
                track.configure(self.media_description if self.source is not None else None,
                                self.media_rect, self.media_size, self.source, self.available)
                await track.open()
                if identity != self.media_id:
                    await track.close()
                    return
                peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
                self.peer = peer
                @peer.on("connectionstatechange")
                async def connection_changed():
                    if peer.connectionState == "failed" and self.peer is peer and self.media_id == identity:
                        selected = self.selected
                        await self.stop_media(failed=True)
                        if self.ws_connection is not None:
                            self.send_stream(dict(type="error", window=selected, stream=identity,
                                message="Window video connection failed. Reconnect to try again."))
                transceiver = peer.addTransceiver(track, direction="sendonly")
                transceiver.setCodecPreferences([codec for codec in RTCRtpSender.getCapabilities("video").codecs
                                                 if codec.mimeType == "video/H264"])
                sender = transceiver.sender
                send_rtp = sender.transport._send_rtp
                offset = None

                async def sent(data):
                    nonlocal offset
                    delivered = track.delivered
                    await send_rtp(data)
                    # aiortc's RTP origin is private; use the actual sender header, not Packet.pts.
                    if delivered is None or len(data) < 12 or 192 <= data[1] <= 223 or not data[1] & 0x80:
                        return
                    timestamp, ssrc = struct.unpack_from("!II", data, 4)
                    if ssrc != sender._ssrc:
                        return
                    source, pts = delivered
                    if offset is None:
                        offset = (timestamp - pts) & 0xffffffff
                    if (timestamp - offset) & 0xffffffff != pts & 0xffffffff:
                        return
                    if (self.media_id == identity and self.source == source and self.boundary is None
                            and self.media_message is not None and self.ws_connection is not None):
                        self.boundary = timestamp
                        await self.write_message(dict(self.media_message, rtp_timestamp=timestamp))

                sender.transport._send_rtp = sent
                await peer.setRemoteDescription(RTCSessionDescription(type=message["description_type"], sdp=message["sdp"]))
                await peer.setLocalDescription(await peer.createAnswer())
                if identity == self.media_id and self.approved:
                    await self.write_message(dict(type="answer", stream=identity,
                        description_type=peer.localDescription.type, sdp=peer.localDescription.sdp))
                    logging.getLogger(__name__).info("Shared window video: persistent capture, H264")
            except BaseException as error:
                if track is not None:
                    track.stop()
                if peer is not None:
                    await peer.close()
                if track is not None:
                    await track.close()
                if identity == self.media_id:
                    self.peer = self.track = None
                    if isinstance(error, Exception) and self.ws_connection is not None:
                        await self.write_message(dict(type="error", window=self.selected,
                                                      stream=identity, message=str(error)))
                if isinstance(error, asyncio.CancelledError):
                    raise

        async def update_view(self, description, available, encoded_popups=None):
            if encoded_popups is None:
                encoded_popups = {}
            if description is None:
                if self.source is not None:
                    self.clear_source()
                    self.send_stream(dict(type="media", window=self.selected, panel=self.panel,
                                            stream=self.media_id, source=None))
                return
            panel, selection = self.panel, self.selection
            root = description["root_rect"]
            view = [0, 0, description["width"], description["height"]]
            if panel is not None:
                item = next((item for item in description["panels"] if item["id"] == panel), None)
                if item is None or not item["available"]:
                    self.panel, self.viewport = None, None
                    self.clear_source()
                    self.send_stream(dict(type="result", window=description["id"], panel=panel,
                        return_panel=None, message=item["reason"] if item else
                        "The plot is no longer visible. Returning to the whole window."))
                    return
                root = item["rect"]
                rectangles = [root, *(popup["rect"] for popup in description["popups"])]
                x, y = (min(rect[i] for rect in rectangles) for i in (0, 1))
                view = [x, y, max(rect[0] + rect[2] for rect in rectangles) - x,
                        max(rect[1] + rect[3] for rect in rectangles) - y]
            if self.viewport is None:
                return
            if sys.platform != "win32":
                root = view
            local_rect = (root[0] - description["root_rect"][0], root[1] - description["root_rect"][1], *root[2:])
            key = (description["id"], description["hwnd"], local_rect, description["native_dpr"], self.viewport)
            new_source = False
            if self.source_key != key:
                scale = min(description["native_dpr"], self.viewport[0] / view[2], self.viewport[1] / view[3])
                size = tuple(max(2, math.ceil(value * scale / 2) * 2) for value in root[2:])
                new_source = self.source_key is None or self.source_key[:4] != key[:4] or size != self.media_size
                if new_source:
                    self.source, self.boundary = secrets.token_hex(12), None
                    self.media_id = self.media_id or secrets.token_hex(12)
                self.source_key = key
            else:
                size = self.media_size
            track = self.track
            if track is not None and track.failure is not None and track.failure[0] == self.source:
                source, detail = track.failure
                self.clear_source()
                self.source_key = key
                self.send_stream(dict(type="error", window=self.selected, stream=self.media_id,
                                        source=source, message="Window capture failed: " + detail))
                return
            if self.source is None:
                return
            self.media_description, self.media_rect, self.media_size = description, root, size
            self.available = available
            source = self.source
            popup_key = tuple((popup["id"], tuple(popup["rect"]), id(popup["rgba"])) for popup in description["popups"])
            signature = (self.source, description["epoch"], description["inputs"],
                         description["panels"], panel, root, view, popup_key)
            if signature == self.last_media:
                return
            if popup_key not in encoded_popups:
                encoded_popups[popup_key] = (await asyncio.to_thread(_popup_images, description)
                                            if sys.platform == "win32" else [])
            if self not in clients or not self.approved or self.selection != selection or self.source != source:
                return
            metadata = {key: description[key] for key in ("epoch", "width", "height", "origin", "inputs")}
            metadata.update(type="media", selection=selection, window=self.selected, stream=self.media_id,
                            source=self.source, root=root, panel=panel, view=view,
                            panels=description["panels"], popups=encoded_popups[popup_key])
            if self.boundary is not None:
                metadata["rtp_timestamp"] = self.boundary
            self.media_message = metadata
            self.send_stream(metadata)
            self.last_media = signature
            if new_source and self.track is not None:
                await self.writing
                if self.selection == selection and self.source == metadata["source"]:
                    self.track.configure(description, root, size, self.source, available)
                    if self.peer is not None and sys.platform != "win32":
                        self.peer.getSenders()[0]._send_keyframe()

        async def on_message(self, raw):
            try:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("expected an object")
                kind = message.get("type")
                if not self.approved:
                    await self.write_message(dict(type="pending", message="Waiting for approval on the experiment computer."))
                    return
                if kind != "select" and message.get("selection") != self.selection and kind != "offer":
                    return
                if kind == "select":
                    selection = message.get("selection")
                    if type(selection) is not int or selection <= self.selection:
                        return
                    self.selection = selection
                    selected = message.get("window")
                    panel = message.get("panel")
                    width, height = (int(value) for value in message["viewport"])
                    if width <= 0 or height <= 0:
                        raise ValueError("viewport dimensions must be positive")
                    self.selected, self.panel, self.viewport = selected, panel, None
                    self.clear_source()

                    def choose():
                        catalog = surface.catalog()
                        window = next((window for window in catalog["windows"] if window["id"] == selected), None)
                        if window is None:
                            raise ValueError("unknown experiment window")
                        if not window["visible"]:
                            return catalog, None
                        if catalog["controller"] == self.identity:
                            surface.take_control(self.identity, selected, activate=False)
                        return catalog, surface.describe(selected)

                    catalog, description = await owner(choose)
                    rejection = None
                    if panel is not None and description is not None:
                        item = next((item for item in description["panels"] if item["id"] == panel), None)
                        if item is None or not item["available"]:
                            rejection = (item["reason"] if item else "The plot is no longer visible. Returning to the whole window.")
                    self.panel = None if rejection else panel
                    if rejection:
                        await self.write_message(dict(type="result", window=selected, panel=panel,
                                                      return_panel=None, message=rejection))
                    else:
                        self.viewport = (width, height)
                        await self.update_view(description, tuple(window["id"] for window in catalog["windows"]))
                    frame_ready.set()
                elif kind == "viewport":
                    if message.get("window") != self.selected or message.get("panel") != self.panel:
                        return
                    width, height = int(message["pixel_width"]), int(message["pixel_height"])
                    if width <= 0 or height <= 0:
                        raise ValueError("viewport dimensions must be positive")
                    self.viewport = (width, height)
                    frame_ready.set()
                elif kind == "offer":
                    if self.media_id is not None and self.peer is None and message.get("stream") == self.media_id and (self.opening is None or self.opening.done()):
                        # Media startup must not hold this connection's input and window-selection messages.
                        self.opening = asyncio.create_task(self.offer(message))
                elif kind in ("control", "show", "input"):
                    if kind == "control":
                        result = (await owner(surface.take_control, self.identity, self.selected)
                                  if message.get("take") is True else await owner(surface.release_control, self.identity))
                    elif kind == "show":
                        result = await owner(surface.show, self.identity, message.get("window"))
                    else:
                        message = dict(message)
                        message["type"] = message.pop("event", "")
                        modifiers = message.get("modifiers") or {}
                        message["modifiers"] = [qt for dom, qt in (
                            ("ctrl", "Control"), ("shift", "Shift"), ("alt", "Alt"), ("meta", "Meta")
                        ) if modifiers.get(dom)]
                        if message["type"] == "wheel":
                            scale = 40 if message.get("deltaMode") == 1 else 800 if message.get("deltaMode") == 2 else 1
                            message["delta_x"] = float(message.get("deltaX", 0)) * scale
                            message["delta_y"] = float(message.get("deltaY", 0)) * scale
                        result = await owner(surface.input, self.identity, message)
                    if kind != "input":
                        await self.write_message(dict(type="result", **result))
                else:
                    raise ValueError("unknown sharing message")
            except tornado.websocket.WebSocketClosedError:
                pass
            except Exception as error:
                if self.ws_connection is not None:
                    await self.write_message(dict(type="error", selection=message.get("selection", self.selection), message=str(error)))

        def on_close(self):
            clients.discard(self)
            network["loop"].add_callback(self.stop_media)
            if getattr(self, "approved", False) and not closed.is_set():
                submit(surface.release_control, self.identity)
            notify_clients()

    async def decide_client(identity, allow):
        client = next((item for item in clients if item.identity == identity), None)
        if client is None or closed.is_set():
            return
        if allow:
            if client.approved:
                return
            try:
                await client.write_message(dict(type="hello", client=identity, controller=None,
                                                control_enabled=surface.control_enabled))
            except tornado.websocket.WebSocketClosedError:
                clients.discard(client)
                notify_clients()
                return
            client.approved = True
        else:
            was_approved = client.approved
            client.approved = False
            await owner(surface.release_control, identity)
            await client.stop_media()
            client.close(1000, "Disconnected by the experiment computer." if was_approved
                         else "Connection rejected by the experiment computer.")
            clients.discard(client)
        notify_clients()

    async def stream():
        while not closed.is_set():
            frame_ready.clear()
            connected = [client for client in clients if client.approved]
            writable = [client for client in connected if client.writing is None or client.writing.done()]
            if connected:
                try:
                    requested = {client.selected for client in writable if client.selected is not None}
                    catalog, descriptions = await owner(poll_windows, requested)
                    state = dict(type="state", control_enabled=surface.control_enabled, **catalog)
                    available = tuple(window["id"] for window in catalog["windows"])
                    encoded_popups = {}
                    for client in writable:
                        if client.last_state != state:
                            client.send_stream(state)
                            client.last_state = state
                        await client.update_view(descriptions.get(client.selected), available, encoded_popups)
                except tornado.websocket.WebSocketClosedError:
                    pass
                except Exception as error:
                    logging.getLogger(__name__).exception("Experiment sharing frame failed")
                    for client in writable:
                        try:
                            if client.ws_connection is not None:
                                client.send_stream(dict(type="error", message=str(error)))
                        except tornado.websocket.WebSocketClosedError:
                            pass
            try:
                await asyncio.wait_for(frame_ready.wait(), 1 / float(fps))
            except TimeoutError:
                pass

    def serve():
        loop = tornado.ioloop.IOLoop()
        try:
            application = tornado.web.Application([
                (r"/zlc/?", Page), (r"/zlc/(client\.js|style\.css|theme\.css)", Page), (r"/zlc/ws", Browser),
            ], websocket_max_message_size=65536, websocket_ping_interval=20,
               websocket_ping_timeout=20)
            server = tornado.httpserver.HTTPServer(application, xheaders=False)
            server.add_sockets(sockets)
            task = loop.asyncio_loop.create_task(stream())
            network.update(loop=loop, server=server, task=task)
            ready.set()
            loop.start()
        except BaseException as error:
            network["error"] = error
            ready.set()
        finally:
            loop.close(all_fds=True)

    thread = Thread(target=serve, name="zlc-share-http", daemon=True)

    async def input_failed(identity, message):
        for client in tuple(clients):
            if client.identity == identity and client.ws_connection is not None:
                await client.write_message(dict(type="error", message=message))
        frame_ready.set()

    surface.input_failed.connect(lambda identity, message:
        network["loop"].add_callback(input_failed, identity, message) if "loop" in network else None)

    def close():
        if closed.is_set():
            return
        closed.set()
        surface.close()
        drain()
        async def stop():
            remaining = tuple(clients)
            await asyncio.gather(*(client.stop_media() for client in remaining), return_exceptions=True)
            for client in remaining:
                client.close(1001, "experiment sharing stopped")
            network["server"].stop()
            network["task"].cancel()
            await asyncio.gather(network["task"], return_exceptions=True)
            await network["server"].close_all_connections()
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            network["loop"].stop()
        if "loop" in network:
            network["loop"].add_callback(stop)
        else:
            for item in sockets:
                item.close()

    thread.start()
    if not ready.wait(3) or "error" in network:
        close()
        raise RuntimeError(f"could not start experiment sharing: {network.get('error', 'startup timed out')}")
    suffix = "" if actual_port == 80 else f":{actual_port}"
    visible_addresses = sorted(address for address in addresses
                               if not address.startswith(("127.", "169.254."))) or ["127.0.0.1"]
    urls = [f"http://{'[' + address + ']' if ':' in address else address}{suffix}/zlc" for address in
            visible_addresses]
    close.connection_info = dict(url=urls[0], urls=urls, port=actual_port)
    close.decide = lambda identity, allow: network["loop"].add_callback(decide_client, identity, allow)
    return close
