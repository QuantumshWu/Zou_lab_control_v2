"""Browser pixels and input for the current experiment, independent of MCP."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.resources import files
from io import BytesIO
import json
import logging
from queue import Empty, Queue
import secrets
import socket
import ssl
from threading import Event, Thread
from time import monotonic
from urllib.parse import urlsplit

import numpy as np
from PIL import Image


def _patch(frame, previous):
    """Encode the changed rectangle off the Qt thread, preserving exact pixels."""
    width, height = frame["pixel_width"], frame["pixel_height"]
    pixels = np.frombuffer(frame["rgba"], np.uint8).reshape(height, width, 4)
    reset = previous is None or any(frame[key] != previous[key] for key in
                                   ("id", "epoch", "pixel_width", "pixel_height"))
    x, y, w, h = 0, 0, width, height
    if not reset:
        old = np.frombuffer(previous["rgba"], np.uint8).reshape(pixels.shape)
        changed = np.any(pixels != old, axis=2)
        rows = np.flatnonzero(np.any(changed, axis=1))
        if not rows.size:
            return None
        columns = np.flatnonzero(np.any(changed, axis=0))
        x, y = int(columns[0]), int(rows[0])
        w, h = int(columns[-1]) - x + 1, int(rows[-1]) - y + 1
    buffer = BytesIO()
    Image.fromarray(pixels[y:y+h, x:x+w]).save(buffer, format="PNG", compress_level=1)
    metadata = {key: frame[key] for key in
                ("epoch", "sequence", "width", "height", "pixel_width", "pixel_height")}
    return dict(type="frame", window=frame["id"], x=x, y=y, w=w, h=h,
                reset=reset, **metadata), buffer.getvalue()


def start_server(flow, *, host="0.0.0.0", port=18865, token=None,
                 allow_control=True, fps=12, certfile=None, keyfile=None):
    """Start an explicitly enabled application-only viewer; return nonblocking close."""
    import tornado.httpserver
    import tornado.ioloop
    import tornado.netutil
    import tornado.web
    import tornado.websocket
    from zlc_ui.windows import SharedWindows
    from ..board import attach_qt_owner_turn

    if not 0 <= int(port) <= 65535 or not 1 <= float(fps) <= 30:
        raise ValueError("sharing requires a valid port and 1–30 frames per second")
    if bool(certfile) != bool(keyfile):
        raise ValueError("share certificate and private key must be supplied together")
    tls = None
    if certfile:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certfile, keyfile)
    token = token or secrets.token_urlsafe(32)
    if not isinstance(token, str) or len(token) < 16:
        raise ValueError("the sharing token must contain at least 16 characters")
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
    clients, latest = set(), {}
    network = {}

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

    addresses = {"127.0.0.1"}
    try:
        addresses.update(item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    if host not in ("0.0.0.0", "::"):
        addresses = {"127.0.0.1" if host == "localhost" else host}
    allowed_hosts = addresses | {"localhost", socket.gethostname().lower(), socket.getfqdn().lower()}
    scheme = "https" if tls else "http"

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
            websocket_origin = f"{'wss' if tls else 'ws'}://{authority}:{actual_port}"
            self.set_header("Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
                f"connect-src 'self' {websocket_origin}; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.write(assets[name])

    class Browser(tornado.websocket.WebSocketHandler):
        def check_origin(self, origin):
            source = urlsplit(origin)
            return valid_host(self.request) and source.scheme == scheme and source.netloc == self.request.host

        def open(self):
            self.identity = secrets.token_hex(12)
            self.authenticated = False
            self.selected = None
            self.frame = None
            self.awaiting = None
            self.last_state = None
            self.auth_deadline = tornado.ioloop.IOLoop.current().call_later(10, self.close, 1008, "authentication required")
            clients.add(self)

        async def on_message(self, raw):
            try:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("expected an object")
                kind = message.get("type")
                if not self.authenticated:
                    supplied = message.get("token")
                    if kind != "hello" or not isinstance(supplied, str) or not secrets.compare_digest(supplied, token):
                        self.close(1008, "invalid sharing token")
                        return
                    self.authenticated = True
                    tornado.ioloop.IOLoop.current().remove_timeout(self.auth_deadline)
                    await self.write_message(dict(type="hello", client=self.identity, controller=None,
                                                  control_enabled=bool(allow_control)))
                    return
                if kind == "select":
                    selected = message.get("window")
                    catalog = await owner(surface.catalog)
                    if selected not in {window["id"] for window in catalog["windows"]}:
                        raise ValueError("unknown experiment window")
                    if self.selected != selected:
                        if catalog["controller"] == self.identity:
                            await owner(surface.input, self.identity, {"type": "release"})
                        self.selected, self.frame = selected, None
                elif kind == "ack":
                    if self.awaiting is not None and message.get("sequence") == self.awaiting:
                        self.awaiting = None
                elif kind in ("control", "show", "input"):
                    if not allow_control:
                        raise ValueError("this shared experiment is read-only")
                    if kind == "control":
                        operation = surface.take_control if message.get("take") is True else surface.release_control
                        result = await owner(operation, self.identity)
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
            if getattr(self, "authenticated", False) and not closed.is_set():
                submit(surface.release_control, self.identity)

    async def stream(executor):
        while not closed.is_set():
            started = monotonic()
            connected = [client for client in clients if client.authenticated]
            watched = {client.selected for client in connected}
            for key in tuple(latest):
                if key not in watched:
                    latest.pop(key)
            if connected:
                try:
                    catalog = await owner(surface.catalog)
                    state = dict(type="state", control_enabled=bool(allow_control), **catalog)
                    visible = {item["id"] for item in catalog["windows"] if item["visible"]}
                    available = {item["id"] for item in catalog["windows"]}
                    for key in tuple(latest):
                        if key not in available:
                            latest.pop(key)
                    for client in connected:
                        if client.last_state != state:
                            await client.write_message(state)
                            client.last_state = state
                    pending = [client for client in connected if client.awaiting is None and client.selected in visible]
                    for identity in {client.selected for client in pending}:
                        frame = await owner(surface.capture, identity, 1600, identity not in latest)
                        if frame is not None:
                            latest[identity] = frame
                    encoded = {}
                    for client in pending:
                        frame = latest.get(client.selected)
                        if frame is None or frame is client.frame:
                            continue
                        key = (id(frame), id(client.frame))
                        if key not in encoded:
                            encoded[key] = await asyncio.get_running_loop().run_in_executor(executor, _patch, frame, client.frame)
                        update = encoded[key]
                        client.frame = frame
                        if update is not None:
                            metadata, png = update
                            client.awaiting = metadata["sequence"]
                            await client.write_message(metadata)
                            await client.write_message(png, binary=True)
                except tornado.websocket.WebSocketClosedError:
                    pass
                except Exception as error:
                    logging.getLogger(__name__).exception("Experiment sharing frame failed")
                    for client in connected:
                        try:
                            if client.ws_connection is not None:
                                await client.write_message(dict(type="error", message=str(error)))
                        except tornado.websocket.WebSocketClosedError:
                            pass
            await asyncio.sleep(max(.005, 1 / float(fps) - (monotonic() - started)))

    def serve():
        loop = tornado.ioloop.IOLoop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zlc-share-encode")
        try:
            application = tornado.web.Application([
                (r"/", Page), (r"/(client\.js|style\.css|theme\.css)", Page), (r"/ws", Browser),
            ], websocket_max_message_size=65536, websocket_ping_interval=20,
               websocket_ping_timeout=20)
            server = tornado.httpserver.HTTPServer(application, ssl_options=tls, xheaders=False)
            server.add_sockets(sockets)
            task = loop.asyncio_loop.create_task(stream(executor))
            network.update(loop=loop, server=server, task=task)
            ready.set()
            loop.start()
        except BaseException as error:
            network["error"] = error
            ready.set()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            loop.close(all_fds=True)

    thread = Thread(target=serve, name="zlc-share-http", daemon=True)

    def close():
        if closed.is_set():
            return
        closed.set()
        surface.close()
        drain()
        async def stop():
            for client in tuple(clients):
                client.close(1001, "experiment sharing stopped")
            network["server"].stop()
            network["task"].cancel()
            await asyncio.gather(network["task"], return_exceptions=True)
            await network["server"].close_all_connections()
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
    urls = [f"{scheme}://{'[' + address + ']' if ':' in address else address}:{actual_port}/#{token}" for address in
            sorted(addresses, key=lambda value: (value.startswith("127."), value))]
    close.connection_info = dict(url=urls[0], urls=urls, token=token, port=actual_port,
                                 addresses=sorted(addresses), encrypted=bool(tls))
    return close
