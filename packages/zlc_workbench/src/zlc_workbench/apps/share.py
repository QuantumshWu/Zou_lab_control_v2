"""Browser pixels and input for the current experiment, independent of MCP."""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from concurrent.futures import Future
import hashlib
from importlib.resources import files
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
from queue import Empty, Queue
import secrets
import socket
import subprocess
import sys
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

from PIL import Image


def _media_executable():
    return Path(os.environ["LOCALAPPDATA"]) / "ZouLabControl" / "media" / "ffmpeg-9.0.2" / "ffmpeg.exe"


def install_media_runtime():
    """Install the pinned Windows media executable during product installation."""
    if sys.platform != "win32":
        return
    executable = _media_executable()
    if executable.is_file():
        print(f"Browser media runtime: {executable}", flush=True)
        return
    url = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip"
    digest = "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba"
    print("Downloading the browser media runtime...", flush=True)
    with urlopen(url, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != digest:
        raise ValueError("Browser media runtime download failed its SHA256 check")
    executable.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(archive)) as bundle:
        temporary = executable.with_suffix(".tmp")
        temporary.write_bytes(bundle.read("ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe"))
        temporary.replace(executable)
    print(f"Browser media runtime installed: {executable}", flush=True)


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
    executable = _media_executable() if sys.platform == "win32" else None
    if executable is not None and not executable.is_file():
        raise RuntimeError("Browser media runtime is missing. Run bin/install_requirements.bat, then start sharing again.")
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
    encoders = ["h264_nvenc", "h264_qsv", "libx264"]

    class WindowVideoTrack(MediaStreamTrack):
        """Pull encoded frames without a second unbounded media queue."""

        kind = "video"

        def __init__(self, description, rect, size, encoder):
            super().__init__()
            self.description, self.rect, self.size, self.encoder = description, rect, size, encoder
            self.process = self.container = self.reading = self.stderr_task = None
            self.closing = None
            self.errors = deque(maxlen=12)
            self.packets = None

        async def open(self):
            root, dpr = self.description["root_rect"], self.description["native_dpr"]
            left, top = (round((self.rect[i] - root[i]) * dpr) for i in (0, 1))
            right, bottom = (round((root[i] + root[i + 2] - self.rect[i] - self.rect[i + 2]) * dpr)
                             for i in (0, 1))
            # Capture above the output cadence; retain original frames in phase-aligned time buckets.
            capture = (f"gfxcapture=hwnd={self.description['hwnd']}:max_framerate={2 * float(fps):g}"
                       f":crop_left={left}:crop_top={top}:crop_right={right}:crop_bottom={bottom}"
                       f":capture_cursor=0:width={self.size[0]}:height={self.size[1]}:resize_mode=scale_aspect,"
                       f"select='isnan(prev_selected_t)+gt(floor((t-start_t)*{float(fps):g}+0.5),"
                       f"floor((prev_selected_t-start_t)*{float(fps):g}+0.5))'")
            command = [str(executable), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "warning"]
            if self.encoder == "h264_nvenc":
                command += ["-init_hw_device", "d3d11va=cap:,vendor_id=0x10de", "-filter_hw_device", "cap"]
                options = ["-preset", "p1", "-tune", "ull", "-rc", "cbr"]
            elif self.encoder == "h264_qsv":
                command += ["-init_hw_device", "d3d11va=cap:,vendor_id=0x8086", "-filter_hw_device", "cap"]
                capture += ",hwmap=derive_device=qsv,format=qsv,vpp_qsv=format=nv12:async_depth=0"
                options = ["-preset", "veryfast", "-async_depth", "1", "-look_ahead", "0"]
            else:
                capture += ",hwdownload,format=bgra,format=yuv420p"
                options = ["-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p"]
            command += ["-filter_complex", capture, "-an", "-c:v", self.encoder, *options,
                        "-profile:v", "baseline", "-bf", "0", "-g", str(max(1, int(fps) // 2)),
                        "-b:v", "20M", "-maxrate", "20M", "-bufsize", "2M",
                        "-flags", "-global_header", "-bsf:v", "dump_extra", "-fps_mode", "passthrough",
                        "-flush_packets", "1", "-f", "nut", "pipe:1"]
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            creationflags=subprocess.CREATE_NO_WINDOW)
            def errors():
                for line in self.process.stderr:
                    self.errors.append(line.decode("utf-8", errors="replace").strip())
            self.stderr_task = asyncio.create_task(asyncio.to_thread(errors))
            reading = self.reading = asyncio.create_task(asyncio.to_thread(av.open,
                SimpleNamespace(read=self.process.stdout.read1), format="nut", mode="r",
                options={"probesize": "32", "analyzeduration": "0"}))
            try:
                container = await asyncio.wait_for(asyncio.shield(reading), 4.)
                if self.readyState != "live":
                    raise MediaStreamError
                self.container = container
                self.packets = self.container.demux(video=0)
            finally:
                if self.reading is reading and reading.done() and self.readyState == "live":
                    self.reading = None

        async def recv(self):
            if self.readyState != "live":
                raise MediaStreamError
            if executable is None:
                import numpy as np
                from fractions import Fraction
                await asyncio.sleep(1 / float(fps))
                scale = min(self.description["native_dpr"], self.size[0] / self.rect[2], self.size[1] / self.rect[3])
                snapshot = await owner(surface.capture, self.description["id"], math.ceil(self.description["width"] * scale), True)
                pixels = np.frombuffer(snapshot["rgba"], np.uint8).reshape(snapshot["pixel_height"], snapshot["pixel_width"], 4)
                scale = snapshot["pixel_width"] / snapshot["width"]
                x, y = (round((self.rect[i] + self.description["origin"][i] - snapshot["origin"][i]) * scale)
                        for i in (0, 1))
                width, height = (round(value * scale) for value in self.rect[2:])
                image = av.VideoFrame.from_ndarray(pixels[y:y + height, x:x + width], format="rgba")
                image = image.reformat(width=self.size[0], height=self.size[1])
                image.pts, image.time_base = int(monotonic() * 90000), Fraction(1, 90000)
                return image
            reading = self.reading = asyncio.create_task(asyncio.to_thread(next, self.packets, None))
            packet = await asyncio.shield(reading)
            if self.reading is reading:
                self.reading = None
            if self.readyState != "live" or packet is None or not packet.size:
                self.stop()
                raise MediaStreamError
            return packet

        def stop(self):
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            super().stop()

        async def close(self):
            if self.closing is None:
                self.closing = asyncio.create_task(self._close())
            await asyncio.shield(self.closing)

        async def _close(self):
            self.stop()
            if self.process is not None:
                await asyncio.to_thread(self.process.wait)
            reading = self.reading
            if reading is not None:
                outcome = await asyncio.gather(reading, return_exceptions=True)
                if self.container is None and outcome and isinstance(outcome[0], av.container.InputContainer):
                    self.container = outcome[0]
                if self.reading is reading:
                    self.reading = None
            if self.container is not None:
                await asyncio.to_thread(self.container.close)
                self.container = None
            if self.stderr_task is not None:
                await self.stderr_task
            if self.process is not None:
                self.process.stdout.close()
                self.process.stderr.close()

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
            self.selected = None
            self.panel = None
            self.viewport = None
            self.peer = self.track = None
            self.media_id = self.source_key = self.media_description = self.media_rect = self.media_size = None
            self.last_media = None
            self.writing = None
            self.last_state = None
            clients.add(self)
            self.write_message(dict(type="pending", message="Waiting for approval on the experiment computer."))
            notify_clients()

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
            self.peer = self.track = None
            self.media_id = self.last_media = None
            if not failed:
                self.source_key = None
            if track is not None:
                track.stop()
            if peer is not None:
                await peer.close()
            if track is not None:
                await track.close()

        async def offer(self, message):
            identity = message.get("stream")
            if identity != self.media_id or self.media_description is None:
                return
            track = peer = None
            try:
                attempts = encoders if executable is not None else ["software"]
                failures = []
                for encoder in tuple(attempts):
                    track = WindowVideoTrack(self.media_description, self.media_rect, self.media_size, encoder)
                    self.track = track
                    try:
                        if executable is not None:
                            await track.open()
                        break
                    except Exception as error:
                        await track.close()
                        failures.append(f"{encoder}: {'; '.join(track.errors) or str(error)}")
                        track = None
                        if identity != self.media_id:
                            return
                if track is None:
                    self.track = None
                    raise RuntimeError("Could not start window video: " + " | ".join(failures))
                if identity != self.media_id:
                    await track.close()
                    return
                if executable is not None:
                    encoders.remove(track.encoder)
                    encoders.insert(0, track.encoder)
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
                await peer.setRemoteDescription(RTCSessionDescription(type=message["description_type"], sdp=message["sdp"]))
                await peer.setLocalDescription(await peer.createAnswer())
                if identity == self.media_id and self.approved:
                    await self.write_message(dict(type="answer", stream=identity,
                        description_type=peer.localDescription.type, sdp=peer.localDescription.sdp))
                    logging.getLogger(__name__).info("Shared window video: %s %sx%s", track.encoder, *self.media_size)
            except Exception:
                if track is not None:
                    track.stop()
                if peer is not None:
                    await peer.close()
                if track is not None:
                    await track.close()
                if identity == self.media_id:
                    self.peer = self.track = None
                    raise

        async def on_message(self, raw):
            try:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("expected an object")
                kind = message.get("type")
                if not self.approved:
                    await self.write_message(dict(type="pending", message="Waiting for approval on the experiment computer."))
                    return
                if kind == "select":
                    selected = message.get("window")
                    panel = message.get("panel")
                    catalog = await owner(surface.catalog)
                    if selected not in {window["id"] for window in catalog["windows"]}:
                        raise ValueError("unknown experiment window")
                    rejection = None
                    if panel is not None:
                        description = await owner(surface.describe, selected)
                        item = next((item for item in description["panels"] if item["id"] == panel), None)
                        if item is None or not item["available"]:
                            rejection = (item["reason"] if item else "The plot is no longer visible. Returning to the whole window.")
                    if self.selected != selected or self.panel != panel or rejection:
                        self.selected = selected
                        self.panel = None if rejection else panel
                        self.viewport = None
                        await self.stop_media()
                        if catalog["controller"] == self.identity:
                            await owner(surface.take_control, self.identity, selected)
                    if rejection:
                        await self.write_message(dict(type="result", window=selected, panel=panel,
                                                      return_panel=None, message=rejection))
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
                    await self.offer(message)
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
                    await self.write_message(dict(type="error", message=str(error)))

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
                    encoded_popups = {}
                    for client in writable:
                        if client.last_state != state:
                            client.send_stream(state)
                            client.last_state = state
                        description = descriptions.get(client.selected)
                        if description is None:
                            if client.media_id is not None:
                                selected = client.selected
                                await client.stop_media()
                                client.send_stream(dict(type="media", window=selected, stream=None))
                            continue
                        panel = client.panel
                        root = description["root_rect"]
                        view = [0, 0, description["width"], description["height"]]
                        if panel is not None:
                            item = next((item for item in description["panels"] if item["id"] == panel), None)
                            if item is None or not item["available"]:
                                client.panel, client.viewport = None, None
                                await client.stop_media()
                                client.send_stream(dict(type="result", window=description["id"], panel=panel,
                                    return_panel=None, message=item["reason"] if item else
                                    "The plot is no longer visible. Returning to the whole window."))
                                continue
                            root = item["rect"]
                            rectangles = [root, *(popup["rect"] for popup in description["popups"])]
                            x, y = (min(rect[i] for rect in rectangles) for i in (0, 1))
                            view = [x, y, max(rect[0] + rect[2] for rect in rectangles) - x,
                                    max(rect[1] + rect[3] for rect in rectangles) - y]
                        if client.viewport is None:
                            continue
                        if executable is None:
                            root = view
                        local_rect = (root[0] - description["root_rect"][0], root[1] - description["root_rect"][1], *root[2:])
                        key = (description["hwnd"], local_rect, description["native_dpr"], client.viewport)
                        if client.source_key != key:
                            scale = min(description["native_dpr"], client.viewport[0] / view[2], client.viewport[1] / view[3])
                            size = tuple(max(2, math.ceil(value * scale / 2) * 2) for value in root[2:])
                            if client.source_key is None or client.source_key[:3] != key[:3] or size != client.media_size:
                                viewport = client.viewport
                                await client.stop_media()
                                if client.selected != description["id"] or client.panel != panel or client.viewport != viewport:
                                    continue
                                client.media_id = secrets.token_hex(12)
                            client.source_key = key
                        elif client.peer is not None and client.track is not None and client.track.readyState == "ended":
                            identity, selected, track = client.media_id, client.selected, client.track
                            await client.stop_media(failed=True)
                            detail = "; ".join(track.errors)
                            client.send_stream(dict(type="error", window=selected, stream=identity,
                                message="Window video stopped. Reconnect to start a new stream."
                                        + (" " + detail if detail else "")))
                            continue
                        else:
                            size = client.media_size
                        client.media_description, client.media_rect, client.media_size = description, root, size
                        if client.media_id is None:
                            continue
                        popup_key = tuple((popup["id"], tuple(popup["rect"]), id(popup["rgba"])) for popup in description["popups"])
                        signature = (client.media_id, description["epoch"], description["inputs"],
                                     description["panels"], panel, root, view, popup_key)
                        if signature == client.last_media:
                            continue
                        if popup_key not in encoded_popups:
                            encoded_popups[popup_key] = (await asyncio.to_thread(_popup_images, description)
                                                        if executable is not None else [])
                        if client not in clients or not client.approved or client.selected != description["id"] or client.panel != panel:
                            continue
                        metadata = {key: description[key] for key in ("epoch", "width", "height", "origin", "inputs")}
                        metadata.update(type="media", window=client.selected, stream=client.media_id,
                                        root=root, panel=panel, view=view, panels=description["panels"],
                                        popups=encoded_popups[popup_key])
                        client.send_stream(metadata)
                        client.last_media = signature
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
