"""Optional MCP transport over the current experiment, never a device owner.

Enabled by ``zlc task_console --mcp-port PORT [--mcp-read-only]``. The official
SDK owns the protocol. Short intents/projections run on the existing Qt owner;
scientific materialization and protocol waits do not run on that owner.
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import Future
from functools import wraps
import json
import logging
from pathlib import Path
from queue import Empty, Queue
import secrets
import socket
from threading import Event, Thread
from time import monotonic, sleep
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ImageContent, TextContent, ToolAnnotations


def start_server(flow, *, host="127.0.0.1", port: int, token: str, allow_control=True):
    """Serve one explicitly enabled local endpoint; return its nonblocking close.

    The flow outlives this adapter. Disconnecting a client neither stops nor
    repeats a run. Closing admission cancels requests that have not started;
    already executed intents still belong to the normal experiment lifecycle.
    """
    import uvicorn
    from starlette.responses import PlainTextResponse

    from ..board import attach_qt_owner_turn
    from ..console import _control_plain, _error_text

    if host != "127.0.0.1" or not 1 <= port <= 65535 or not token:
        raise ValueError("MCP requires a loopback port and a nonempty ZLC_MCP_TOKEN")
    inbox = Queue(maxsize=32)
    closed = Event()
    last_owner_reply = [None]

    def drain():
        while True:
            try:
                future, action = inbox.get_nowait()
            except Empty:
                return
            if closed.is_set():
                future.cancel()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(action())
            except BaseException as error:
                future.set_exception(error)
            finally:
                last_owner_reply[0] = monotonic()

    wake = attach_qt_owner_turn(drain)

    async def owner(action):
        if closed.is_set():
            raise RuntimeError("Experiment MCP is closing")
        future = Future()
        inbox.put_nowait((future, action))
        wake()
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=5.)
        except TimeoutError:
            cancelled = future.cancel()
            raise RuntimeError(
                "GUI owner did not respond within 5 s. "
                + ("The queued request was cancelled before execution."
                   if cancelled or future.cancelled() else
                   "The request had started; its outcome is unknown. Inspect state; do not replay a write.")
            ) from None

    def presenter():
        if flow.console_presenter is None:
            raise RuntimeError("No experiment Session. Inspect Devices and initialize the authored apparatus first.")
        return flow.console_presenter

    def state(after_sequence=0, node_id=None, panel_id=None, details=False):
        manager = flow.devices.presenter if flow.devices is not None else None
        result = dict(workspace=str(flow.space.root), control_enabled=bool(allow_control),
                      session_open=flow.session is not None, closing=flow._closing_all)
        if manager is not None:
            result["devices"] = [dict(instance_id=item.instance_id, role=item.role,
                                      type_id=item.type_id,
                                      **({"parameters": _control_plain(item.parameters)} if details else {}))
                                 for item in manager.devices]
            result["device_operation_pending"] = manager.device_operation_active
            result["device_status"] = flow.devices.status_snapshot()
        if flow.session is not None:
            result["device_failures"] = {name: str(error) for name, error in flow.session.failures.items()}
        if flow.console_presenter is not None:
            result["console"] = flow.console_presenter.control_state(after_sequence=after_sequence,
                node_id=node_id, panel_id=panel_id, details=details)
        return result

    def workspace_path(path):
        root = flow.space.root.resolve()
        selected = Path(path)
        selected = (selected if selected.is_absolute() else root / selected).resolve()
        if not selected.is_relative_to(root):
            raise ValueError("MCP file access is limited to this experiment workspace")
        return selected

    mcp = MCPServer("ZLC experiment", log_level="WARNING", instructions=(
        "Operate the currently open experiment, not another device session. Start with lab_state and lab_catalog. "
        "Draft edits are not applied runs. Start acceptance is not ready or completion; use exact generation in lab_wait. "
        "Inspect errors and data validity; do not infer success from screenshots alone. "
        "Manual physical actions require the human operator. Never automatically retry Start, Fire or device Apply "
        "after an uncertain outcome. This is trusted experiment access, not a sandbox: "
        "plugin expressions retain their normal semantics. No direct shell tool is offered."))
    read = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

    def tool(*, annotations=None):
        def register(function):
            @wraps(function)
            async def report_error(*args, **kwargs):
                try:
                    return await function(*args, **kwargs)
                except Exception as error:
                    # The SDK intentionally hides ordinary exceptions. This
                    # authenticated local operator needs the actual reason to
                    # recover, including queued-vs-started timeout outcomes.
                    logging.getLogger(__name__).exception("MCP %s failed", function.__name__)
                    raise ToolError(_error_text(error)) from error
            return mcp.tool(annotations=annotations)(report_error)
        return register

    @tool(annotations=read)
    async def lab_health() -> dict[str, object]:
        """Protocol health, also available when the GUI owner is unresponsive. Not hardware readiness."""
        last = last_owner_reply[0]
        return dict(service="closing" if closed.is_set() else "running",
                    seconds_since_owner_response=None if last is None else monotonic()-last,
                    queued_owner_requests=inbox.qsize(), control_enabled=bool(allow_control))

    @tool(annotations=read)
    async def lab_state(after_sequence: int = 0, node_id: str | None = None,
                         panel_id: str | None = None, details: bool = False) -> dict[str, object]:
        """One compact experiment overview; filter by node/panel for fast follow-ups. Details adds authored values.
        Current errors, exact generations, pending/ready/terminal and messages since after_sequence are authoritative.
        """
        return await owner(lambda: state(after_sequence, node_id, panel_id, details))

    @tool(annotations=read)
    async def lab_catalog(api_name: str | None = None, node_id: str | None = None,
                           panel_id: str | None = None) -> dict[str, object]:
        """Compact available-node index by default. Supply api_name for one full parameter schema, or node_id/panel_id
        for that current editor's units, choices and parameters. Requires an initialized Session.
        """
        return await owner(lambda: presenter().control_catalog(api_name=api_name,node_id=node_id, panel_id=panel_id))

    @tool(annotations=read)
    async def lab_screenshot(window: Literal["console", "devices"] = "console",
                             device_id: str | None = None) -> CallToolResult:
        """View the current experiment window or its active modal dialog. On-demand PNG; no rerender or other apps."""
        def capture():
            handle = (flow.device_controls.get(device_id) if device_id is not None else
                      flow.console if window == "console" else flow.devices)
            if handle is None:
                raise RuntimeError(f"The {window} window is not open")
            return handle.screenshot_png(include_dialog=True)
        data = await owner(capture)
        return CallToolResult(content=[ImageContent(type="image", mime_type="image/png",
                                                    data=base64.b64encode(data).decode("ascii"))])

    @tool(annotations=read)
    async def lab_wait(node_id: str, generation: str, until: Literal["ready", "terminal"] = "terminal",
                       timeout_seconds: float = 10.) -> dict[str, object]:
        """Wait for one exact run, without blocking Stop or the GUI. Timeout does not stop or restart it."""
        if not 0 <= timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be between 0 and 30")
        def current():
            binding = presenter().logic[node_id]
            host = binding.host
            if host is None or host.generation is None or host.generation.value != generation:
                raise ValueError("The requested generation is no longer this node's current run")
            return host
        selected = await owner(current)
        wait = selected.wait_ready if until == "ready" else selected.wait_terminal
        reached = await asyncio.to_thread(wait, timeout_seconds)
        def outcome():
            binding = presenter().logic.get(node_id)
            return dict(observation=_control_plain(selected.observation),
                        still_current=binding is not None and binding.host is selected,
                        state=state(node_id=node_id))
        result = await owner(outcome)
        return dict(node_id=node_id, generation=generation, condition=until, reached=bool(reached), **result)

    @tool(annotations=read)
    async def lab_signal(signal: str, generation: str | None = None, offset: int = 0, limit: int = 256,
                         dataset: bool = False) -> dict[str, object]:
        """Read immutable numeric data plus validity. Default is the latest publication chunk; dataset=True materializes
        the current finite dataset/retained window. Offset/limit page flattened physical storage, not logical axes.
        The returned exact revision identifies this page; concurrent live pages may have different revisions.
        Physical indices and each axis's sample_codes/sample_coordinates align one-to-one with returned values.
        """
        if offset < 0 or not 1 <= limit <= 4096:
            raise ValueError("offset must be nonnegative and limit between 1 and 4096")
        def freeze():
            plane = presenter().session.signal_plane
            publication = plane.latest_publication(signal)
            if publication is None or signal not in publication.signals:
                raise LookupError("No publication for this signal yet")
            value = publication.signals[signal]
            if generation is not None and value.snapshot.ref.stream_generation.value != generation:
                raise ValueError("Signal generation changed; read current state before using this result")
            return plane, publication, value.snapshot
        plane, publication, snapshot = await owner(freeze)
        def read_data():
            import numpy as np
            from zlc_data.codec import dataset_revision_ref_to_tree
            selected = plane.current_dataset(signal, publication) if dataset else snapshot
            selected = selected.materialize()
            block = selected.block
            values = block.values.reshape(-1)[offset:offset+limit]
            valid = selected.expanded_validity().reshape(-1)[offset:offset+limit]
            flat_indices = (np.arange(offset, offset + values.size, dtype=np.int64)
                            if values.size else np.empty(0, dtype=np.int64))
            physical_indices = np.column_stack(np.unravel_index(flat_indices, block.schema.physical_shape))
            value_schema = block.schema.value_schema
            schema = {"value_schema": dict(dtype=str(value_schema.dtype), value_unit=value_schema.value_unit,
                                           name=value_schema.name,
                                           validity_contract=_control_plain(value_schema.validity_contract))}
            physical_start = 0
            for domain_name, domain in (("repeat_domain", block.schema.repeat_domain),
                                        ("point_domain", block.schema.point_domain),
                                        ("cell_domain", block.schema.cell_domain)):
                axes = []
                for axis_index, axis in enumerate(domain.axes):
                    # Explicit domains map every logical axis from their one
                    # physical carrier. Dense cell axes map their own dimension.
                    dimension = physical_start + (axis_index if domain.axis_codes is None else 0)
                    codes = domain.codes(axis.axis_id, physical_indices[:, dimension])
                    axes.append(dict(
                        axis_id=axis.axis_id.value, name=axis.name, role=axis.role.value,
                        unit=axis.unit, size=axis.size,
                        coordinate_frame=None if axis.coordinate_frame is None else axis.coordinate_frame.value,
                        coordinate_of=None if axis.coordinate_of is None else axis.coordinate_of.value,
                        sample_codes=codes.tolist(),
                        sample_coordinates=[axis.coordinate_at(int(code)) for code in codes],
                        sample_labels=None if axis.coordinate_labels is None else
                        [axis.coordinate_labels[int(code)] for code in codes],
                    ))
                schema[domain_name] = dict(shape=list(domain.shape), axes=axes)
                physical_start += len(domain.shape)
            def numbers(a):
                if np.iscomplexobj(a):
                    return dict(real=numbers(a.real), imaginary=numbers(a.imag))
                return [v.item() if np.isfinite(v) else None for v in a]
            return dict(signal=signal, scope="current_dataset" if dataset else "publication_chunk",
                        ref=dataset_revision_ref_to_tree(selected.ref), shape=list(block.schema.physical_shape),
                        schema=schema, flat_indices=flat_indices.tolist(), physical_indices=physical_indices.tolist(),
                        offset=offset, returned=int(values.size),
                        total=int(block.values.size), values=numbers(values), valid=valid.tolist(),
                        nonfinite_encoding="null", sigma=None if block.sigma is None else
                        numbers(block.sigma.reshape(-1)[offset:offset+limit]))
        return await asyncio.to_thread(read_data)

    @tool(annotations=read)
    async def lab_files(directory: str = ".", offset: int = 0, limit: int = 100) -> dict[str, object]:
        """Discover workspace files/directories without GUI file dialogs. Paths returned are workspace-relative.
        Node resource fields may instead resolve relative to their declared resource directory; consult lab_catalog.
        """
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("offset must be nonnegative and limit between 1 and 200")
        selected = workspace_path(directory)
        def listing():
            entries = sorted(selected.iterdir(), key=lambda item: item.name.casefold())
            result = []
            for entry in entries[offset:offset+limit]:
                if not entry.resolve().is_relative_to(flow.space.root.resolve()):
                    continue
                stat = entry.stat()
                result.append(dict(path=entry.relative_to(flow.space.root).as_posix(),
                                   directory=entry.is_dir(), bytes=stat.st_size,
                                   modified_ns=stat.st_mtime_ns))
            return dict(directory=directory, entries=result, offset=offset,
                        next_offset=min(len(entries), offset+limit), total=len(entries))
        return await asyncio.to_thread(listing)

    @tool(annotations=read)
    async def lab_artifact(path: str, offset: int = 0, limit: int = 16000) -> CallToolResult:
        """Read a workspace report or preview PNG/JPEG. Binary scientific files remain files, not giant JSON arrays."""
        selected = workspace_path(path)
        if offset < 0 or not 1 <= limit <= 64000:
            raise ValueError("Use a nonnegative byte offset and a limit of 1..64000")
        def read_file():
            size = selected.stat().st_size
            if selected.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                if size > 16*1024*1024:
                    raise ValueError("Preview exceeds 16 MiB; use the current window screenshot")
                mime = "image/png" if selected.suffix.lower() == ".png" else "image/jpeg"
                return CallToolResult(content=[ImageContent(type="image", mime_type=mime,
                    data=base64.b64encode(selected.read_bytes()).decode("ascii"))])
            if selected.suffix.lower() not in {".json", ".txt", ".md", ".log"}:
                return CallToolResult(content=[TextContent(type="text", text=json.dumps(
                    dict(path=str(selected), bytes=size, note="Binary artifact; use the framework reader for analysis.")))])
            with selected.open("rb") as stream:
                stream.seek(offset)
                raw = stream.read(limit)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(dict(
                path=str(selected), offset=offset, next_offset=offset+len(raw), total_bytes=size,
                text=raw.decode("utf-8", errors="replace")), ensure_ascii=False))])
        return await asyncio.to_thread(read_file)

    @tool(annotations=read)
    async def lab_device_state(device_id: str) -> dict[str, object]:
        """Inspect a Control's latest reading, permission and pending state; no hardware refresh is implied."""
        return await owner(lambda: flow.control_device("inspect", device_id))

    @tool(annotations=read)
    async def lab_pulse_state(device_id: str) -> dict[str, object]:
        """Inspect an already-open Pulse Control's cached board state, pending commands and last warning; no device I/O."""
        return await owner(lambda: flow.control_pulse("inspect", device_id))

    def editor(device_id):
        control = flow.device_controls.get(device_id)
        if control is None:
            raise RuntimeError("Open this device Control with lab_device_action first")
        return getattr(control, "presenter", control)

    @tool(annotations=read)
    async def lab_editor_catalog(device_id: str) -> dict[str, object]:
        """Discover an opened Pulse/SLM editor's actions, IDs, units and parameter choices from its actual owner."""
        return await owner(lambda: editor(device_id).control_catalog())

    @tool(annotations=read)
    async def lab_editor_state(device_id: str, details: bool = False) -> dict[str, object]:
        """Read the opened editor's draft, pending/error and accepted-device facts. Details includes the actual Pulse
        document/Config or SLM target summaries. Draft, compiled preview and applied evidence remain distinct.
        """
        return await owner(lambda: editor(device_id).control_state(details=details))

    if allow_control:
        @tool()
        async def lab_editor_action(device_id: str, action: str, arguments: dict[str, object],
                                    expected_revision: int | None = None) -> dict[str, object]:
            """Use an opened Pulse/SLM editor's discovered action, via the same authoring/worker as the GUI.
            Consult lab_editor_catalog; path arguments are workspace-relative or absolute inside the workspace.
            An empty path stays empty for actions that explicitly clear a file reference.
            Saving/applying/solving can be asynchronous; inspect pending/errors afterward. expected_revision rejects
            edits against a stale draft. This does not override normal device ownership or stale-device protections.
            """
            arguments = dict(arguments)
            if "path" in arguments and arguments["path"] != "":
                arguments["path"] = str(workspace_path(str(arguments["path"])))
            def submit():
                selected = editor(device_id)
                current = selected.control_state()
                if expected_revision is not None and current.get("revision") != expected_revision:
                    raise ValueError("Editor revision changed; inspect current state before editing")
                return selected.control_action(action, arguments)
            return await owner(submit)

        @tool()
        async def lab_initialize() -> dict[str, object]:
            """Initialize the apparatus currently authored in Devices. Never toggles an existing session off.
            This opens actual devices; inspect lab_state first. Acceptance is not completion.
            """
            def initialize():
                if flow.session is not None:
                    return dict(accepted=False, status="already_initialized", state=state())
                manager = flow.devices.presenter
                if manager.device_operation_active:
                    return dict(accepted=False, status="pending", state=state())
                accepted = manager.toggle_lifecycle()
                return dict(accepted=accepted, status="initializing" if accepted else "rejected", state=state())
            return await owner(initialize)

        @tool()
        async def lab_action(action: Literal["logic.add", "logic.update", "logic.validate", "logic.start",
                                            "logic.stop", "logic.remove", "panel.add", "panel.update",
                                            "panel.remove", "panel.edit", "panel.refresh", "panel.save",
                                            "view.monitor", "view.logic", "layout.save", "layout.load"],
                             arguments: dict[str, object], expected_draft_revision: int | None = None,
                             expected_generation: str | None = None) -> dict[str, object]:
            """Use the normal experiment action. See lab_catalog first. logic.add: api_name, optional values,
            source_signal, device_keys, artifact_inputs. Node edits: node_id plus values,
            source_signal, device_keys, artifact_inputs. Start/Stop: node_id. Panel add: kind; update: panel_id,patch.
            view.monitor/view.logic take empty arguments and navigate directly, without pixel clicks.
            layout.save/load take path (workspace JSON); load restores stopped drafts. Loading over active Logic
            requires explicit stop_running=true; poll state.layout_pending until the replacement completes.
            Panel kind must be one of lab_catalog.panel_kinds (for example image, not Image).
            Save: panel_id,selected workspace image path ending .png/.pdf/.svg; the standard matching science .npz
            is saved too. Saving is asynchronous: inspect subsequent reports/errors. Manual requests stay human.
            Use observed expected_draft_revision for Start and expected_generation for Stop to reject stale intents.
            """
            arguments = dict(arguments)
            if action == "panel.save":
                arguments["selected"] = str(workspace_path(arguments["selected"]))
            if action in {"layout.save", "layout.load"}:
                arguments["path"] = str(workspace_path(arguments["path"]))
            return await owner(lambda: presenter().control_action(action, arguments,
                expected_draft_revision=expected_draft_revision, expected_generation=expected_generation))

        @tool()
        async def lab_device_action(action: Literal["open", "refresh", "unit", "apply"], device_id: str,
                                    field: str | None = None, value: float | int | bool | str | None = None,
                                    unit: str | None = None) -> dict[str, object]:
            """Open/refresh/apply through the same Device Control worker and ownership policy. Open and inspect first.
            Apply changes hardware; its return is submission, not readback. Check lab_device_state afterward.
            """
            return await owner(lambda: flow.control_device(action, device_id, field=field, value=value, unit=unit))

        @tool()
        async def lab_pulse(action: Literal["open", "inspect", "fire", "stop"], device_id: str,
                            path: str | None = None) -> dict[str, object]:
            """Use the bound Pulse Editor's existing asynchronous actions. Open takes a workspace Pulse JSON.
            Fire uses exactly its authored repeats/config; acceptance is not completion. Inspect pending/errors.
            Never retry Fire on a timeout. No implicit camera arming or pulse editing is performed.
            """
            selected = None if path is None else str(workspace_path(path))
            return await owner(lambda: flow.control_pulse(action, device_id, path=selected))

    application = mcp.streamable_http_app(host=host, stateless_http=True, json_response=True,
                                         max_request_body_size=1024*1024)

    async def authorized(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", ()))
            supplied = headers.get(b"authorization", b"").decode("latin-1")
            if b"origin" in headers:
                await PlainTextResponse("Browser origins are not allowed", status_code=403)(scope, receive, send)
                return
            if not secrets.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
                await PlainTextResponse("Bearer token required", status_code=401)(scope, receive, send)
                return
            if closed.is_set():
                await PlainTextResponse("Experiment MCP is closing", status_code=503)(scope, receive, send)
                return
        await application(scope, receive, send)

    # Bind before returning so an occupied port fails visibly at startup rather
    # than silently killing a background thread. No extra device/session exists.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind((host, port))
        listener.listen(32)
        listener.setblocking(False)
        server = uvicorn.Server(uvicorn.Config(authorized, host=host, port=port,
            log_level="warning", access_log=False, timeout_graceful_shutdown=1))
        failure = []
        def serve():
            try:
                asyncio.run(server.serve(sockets=[listener]))
            except BaseException as error:
                failure.append(error)
            finally:
                listener.close()
        thread = Thread(target=serve,
                        name="zlc-mcp", daemon=True)
        thread.start()
        deadline = monotonic() + 5.
        while not server.started and thread.is_alive() and monotonic() < deadline:
            sleep(.01)
        if not server.started:
            server.should_exit = True
            raise RuntimeError(f"MCP transport did not start: {failure[0] if failure else 'startup timed out'}")
    except BaseException:
        listener.close()
        raise

    def close():
        closed.set()
        server.should_exit = True
        while True:
            try:
                future, _ = inbox.get_nowait()
            except Empty:
                break
            future.cancel()
        return not thread.is_alive()

    return close
