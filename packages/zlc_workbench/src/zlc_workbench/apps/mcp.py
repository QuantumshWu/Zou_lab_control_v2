"""MCP transport over the current experiment, never a device owner.

Enabled by the ordinary experiment launcher unless ``--no-mcp``. The official
SDK owns the protocol. Short intents/projections run on the existing Qt owner;
scientific materialization and protocol waits do not run on that owner.
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import Future
from functools import wraps
import json
import inspect
import logging
import os
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


_DEVELOPER_TOOLS = frozenset({"lab_handles", "lab_api", "lab_call", "lab_intent"})

_OPERATOR_INSTRUCTIONS = (
    "ZLC is a complete experiment application: device control, Pulse timing, Camera/Waveform acquisition, "
    "Calibration, Occupancy, Frame Survival, Derive, Seamless Scan, SLM rearrangement, live Plot/Facet/Fit, "
    "and reopenable Figure archives. Begin with lab_guide('overview') and lab_catalog(); neither requires Init. "
    "Use lab_guide(topic) for recommended native workflows, lab_recipes for saved Layout recipes, and the "
    "actual workspace EXPERIMENT_CONTEXT.md for apparatus-specific settings. Frame Survival already computes "
    "conditional survival; native panels already pool repeats, facet by site, fit Exponential and publish "
    "fit parameters. Use lab_results to read those results, not external statistics or plotting scripts. "
    "Configure the native pipeline and fit before Start so results update during acquisition; distinguish "
    "the first usable result from completing the whole plan. Use lab_estimate for Scan's timing projection. "
    "Save through lab_save for matching image + scientific Figure NPZ, readable with lab_artifact. "
    "Address Logic nodes by node_id; the operator calls them by lab_state logic[].name, which may change "
    "at any time. "
    "Reuse returned state, exact generations and known signatures; batch already-decided steps without "
    "intervening reasoning. Ready/Fire belongs to the selected execution node when it already manages it. "
    "Acceptance is not completion: inspect the original owner error/ready/terminal facts. Never replay an "
    "uncertain Start/Fire/Apply; manual physical requests still require the human. Developer fallback APIs "
    "are offered only with the developer tool profile, using the same owners, not another session."
)


def _step_succeeded(result):
    """The shared action/wait outcome, independent of additional expectations."""
    if not isinstance(result, dict):
        return result is not False
    if (any(result.get(key) is False for key in
            ("accepted", "matched", "reached", "still_current")) or result.get("error")):
        return False
    observation = result.get("observation")
    # Terminal was reached after Stop, but the experiment did not finish.
    if isinstance(observation, dict) and observation.get("phase") == "cancelled":
        return False
    # Generic owner calls and field-condition waits wrap the original result.
    return _step_succeeded(result["result"]) if "result" in result else True


def guide_document(topic="overview"):
    """Read the shipped operator guide without a device session."""
    from importlib.resources import files

    source = files("zlc_workbench").joinpath("guides/mcp.md").read_text(encoding="utf-8")
    sections = {}
    for section in source.split("\n## ")[1:]:
        name, _, body = section.partition("\n")
        sections[name.strip()] = body.strip()
    if topic not in sections:
        raise ValueError(f"Unknown guide topic {topic!r}; choose {', '.join(sections)} or node:<api_name>")
    return dict(topic=topic, content=sections[topic], topics=list(sections),
                source="zlc_workbench/guides/mcp.md")


def _utf8_page(raw: bytes, offset: int, limit: int):
    """Never split a multi-byte label across the returned continuation offset."""
    text = raw[:limit].decode("utf-8", errors="ignore")
    consumed = len(text.encode("utf-8"))
    if raw and not consumed:
        raise ValueError("limit is too small for the next UTF-8 character")
    return text, offset + consumed


def _batch_arguments(value, completed):
    """Bind an earlier returned JSON field; no expressions, branches or retries."""
    if isinstance(value, dict):
        if set(value) == {"$result"}:
            path = value["$result"]
            if (not isinstance(path, list) or not path or type(path[0]) is not int
                    or not 0 <= path[0] < len(completed)):
                raise ValueError("$result must name an earlier completed step and its JSON field path")
            result = completed[path[0]]["result"]
            try:
                for key in path[1:]:
                    result = result[key]
            except (KeyError, IndexError, TypeError):
                raise ValueError(f"Earlier result does not contain {path!r}; inspect it without replaying writes") from None
            return result
        return {key: _batch_arguments(item, completed) for key, item in value.items()}
    if isinstance(value, list):
        return [_batch_arguments(item, completed) for item in value]
    return value


def start_server(flow, *, host="127.0.0.1", port: int, token: str, allow_control=True):
    """Serve one local experiment endpoint; return its nonblocking close.

    The flow outlives this adapter. Disconnecting a client neither stops nor
    repeats a run. Closing admission cancels requests that have not started;
    already executed intents still belong to the normal experiment lifecycle.
    """
    import uvicorn
    from starlette.responses import PlainTextResponse

    from ..board import attach_qt_owner_turn
    from ..console import _control_plain, _error_text

    if host != "127.0.0.1" or not 0 <= port <= 65535 or not token:
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
        if selected.is_relative_to(root / ".mcp"):
            raise ValueError("MCP connection credentials are not experiment artifacts")
        return selected

    mcp = MCPServer("ZLC experiment", log_level="WARNING", instructions=_OPERATOR_INSTRUCTIONS +
                    " This is trusted experiment access, not a sandbox; plugin expressions retain their normal semantics.")
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
                    text = _error_text(error)
                    if isinstance(error, KeyError) and text == str(error):
                        # A bare lookup: its text is only the quoted name.
                        text = (f"{function.__name__}: unknown name {text}; "
                                + ("lab_handles and lab_api" if function.__name__ in _DEVELOPER_TOOLS
                                   else "lab_state and lab_catalog") + " list the valid names")
                    raise ToolError(text) from error
            return mcp.tool(annotations=annotations)(report_error)
        return register

    @tool(annotations=read)
    async def lab_health() -> dict[str, object]:
        """Protocol health, also available when the GUI owner is unresponsive. Not hardware readiness."""
        last = last_owner_reply[0]
        return dict(service="closing" if closed.is_set() else "running", pid=os.getpid(),
                    workspace=str(flow.space.root.resolve()),
                    seconds_since_owner_response=None if last is None else monotonic()-last,
                    queued_owner_requests=inbox.qsize(), control_enabled=bool(allow_control))

    @tool(annotations=read)
    async def lab_state(after_sequence: int = 0, node_id: str | None = None,
                         panel_id: str | None = None, details: bool = False) -> dict[str, object]:
        """One compact experiment overview; filter by node/panel for fast follow-ups. Details adds authored values.
        Current errors, exact generations, pending/ready/terminal and messages since after_sequence are authoritative.
        Each logic row has node_id (the stable address) and name (what the operator sees; may change any time).
        """
        return await owner(lambda: state(after_sequence, node_id, panel_id, details))

    @tool(annotations=read)
    async def lab_catalog(api_name: str | None = None, node_id: str | None = None,
                           panel_id: str | None = None, details: bool = False) -> dict[str, object]:
        """Discover native acquisition, processing, scan, plot and fit capabilities BEFORE Init.
        Defaults to concise purposes/input/output summaries. api_name gives a node's full schema;
        node_id/panel_id gives current bound units, choices, source and editor parameters (needs a Session).
        details=True includes full static fit formulas/parameter definitions. Static support is distinct
        from compatibility with a live source. See lab_guide for workflows.
        """
        if node_id is None and panel_id is None:
            def describe():
                from ..logic import LogicCatalog
                from ..console import control_capabilities
                return control_capabilities(LogicCatalog(), flow.space, api_name=api_name, details=details)
            return await asyncio.to_thread(describe)
        return await owner(lambda: presenter().control_catalog(api_name=api_name,node_id=node_id,
                                                               panel_id=panel_id, details=details))

    @tool(annotations=read)
    async def lab_guide(topic: str = "overview") -> dict[str, object]:
        """Learn native capabilities and workflows without reading source or initializing devices.
        Start with overview; topics: setup, acquisition, pulse, slm, calibration, processing, scan, plot,
        fit, files, recipes, lifetime, errors. node:<api_name> reads that installed node's purpose/schema.
        This is read-only help; it does not initialize, fire, fit or save anything.
        """
        if topic.startswith("node:"):
            return await lab_catalog(api_name=topic[5:])
        return await asyncio.to_thread(guide_document, topic)

    @tool(annotations=read)
    async def lab_screenshot(window: Literal["console", "devices"] = "console",
                             device_id: str | None = None) -> CallToolResult:
        """View the current experiment window or its active modal dialog. On-demand PNG; no rerender or other apps."""
        def capture():
            handle = (flow.device_controls.get(device_id) if device_id is not None else
                      flow.console if window == "console" else flow.devices)
            if handle is None:
                raise RuntimeError(
                    f"The {window} window is not open" if device_id is None else
                    f"Device Control {device_id!r} is not open; open it with lab_device_action(action='open')")
            return handle.screenshot_png(include_dialog=True)
        data = await owner(capture)
        return CallToolResult(content=[ImageContent(type="image", mime_type="image/png",
                                                    data=base64.b64encode(data).decode("ascii"))])

    @tool(annotations=read)
    async def lab_wait(node_id: str, generation: str, until: Literal["ready", "terminal"] = "terminal",
                       timeout_seconds: float = 10.) -> dict[str, object]:
        """Observe one exact run for up to 1800 s without repeated model turns; Stop/GUI stay available.
        Uses existing Runtime events, not device polling. Timeout only ends observation, never restarts
        the run. An operator request returns early with reached=False and the real request in state.
        The client tool timeout should exceed timeout_seconds by at least 10 seconds.
        """
        if not 0 <= timeout_seconds <= 1800:
            raise ValueError("timeout_seconds must be between 0 and 1800")
        def current():
            binding = presenter().logic.get(node_id)
            if binding is None:
                raise KeyError(f"unknown logic node {node_id!r}")
            host = binding.host
            if host is None or host.generation is None or host.generation.value != generation:
                raise ValueError("The requested generation is no longer this node's current run")
            return host
        selected = await owner(current)
        wait = selected.wait_ready if until == "ready" else selected.wait_terminal
        deadline = monotonic() + timeout_seconds
        reached = False
        while True:
            # Observe the original Host, not hardware. A review must release
            # this tool call so the operator can answer it before completion.
            if selected.operator_request is not None:
                break
            try:
                reached = await asyncio.to_thread(wait, min(.25, max(0., deadline-monotonic())))
            except (RuntimeError, InterruptedError):
                # Stopped, failed or finished before readiness: the outcome
                # below reports which, with the run's own observation.
                break
            if reached or monotonic() >= deadline:
                break
        def outcome():
            binding = presenter().logic.get(node_id)
            observation = _control_plain(selected.observation)
            return dict(observation=observation,
                        error=observation.get("error") if observation.get("phase") == "failed" else None,
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
    async def lab_results(panel_id: str, offset: int = 0, limit: int = 256) -> dict[str, object]:
        """Read native fit rows: coordinates, units, sigma, fixed/valid flags and exact source refs.
        Enable the panel's native fit first; reads never solve/re-fit or redraw. For per-site lifetime,
        read decay_time from the native Exponential Facet fit. status='current' means a fit matching
        the display (not 'ready'); source_is_latest shows whether newer source data exists. status='lagging'
        can still contain usable earlier rows; row.success/valid determine individual fit validity.
        Actual published output signal names are returned; do not guess them from parameter symbols.
        """
        future = await owner(lambda: presenter().control_results(panel_id, offset=offset, limit=limit))
        return await asyncio.wrap_future(future)

    @tool(annotations=read)
    async def lab_estimate(node_id: str) -> dict[str, object]:
        """Read the selected node's native workload/time projection without Start/Fire/Load.
        Scan distinguishes full sweeps, shots per point, first-sweep and total scheduled Pulse time.
        Unknown device/config facts and unestimated hardware/operator overhead remain explicit.
        """
        future = await owner(lambda: presenter().control_estimate(node_id))
        return await asyncio.wrap_future(future)

    @tool(annotations=read)
    async def lab_recipes(name: str | None = None) -> dict[str, object]:
        """Discover verified native Layout recipes and their input bindings before initializing devices.
        Select a name for required resources, native-field defaults, start owner, result panels and
        completion/save guidance. Workspace recipes override bundled examples. This never starts a run.
        """
        def describe():
            from ..console_layout import experiment_recipes, read_experiment_recipe
            names = [name] if name is not None else list(experiment_recipes(flow.space.root))
            recipes = []
            for selected in names:
                recipe, document, defaults = read_experiment_recipe(flow.space.root, selected)
                if name is None:
                    recipes.append({key: recipe[key] for key in ("name", "title", "description", "requirements")})
                else:
                    recipes.append(dict(recipe, defaults=defaults, layout=document.to_tree(),
                                        note="Defaults are read from the Layout; required parameters must be explicit."))
            return {"recipes": recipes}
        return await asyncio.to_thread(describe)

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
                if (not entry.resolve().is_relative_to(flow.space.root.resolve())
                        or entry.resolve().is_relative_to(flow.space.root.resolve() / ".mcp")):
                    continue
                stat = entry.stat()
                result.append(dict(path=entry.relative_to(flow.space.root).as_posix(),
                                   directory=entry.is_dir(), bytes=stat.st_size,
                                   modified_ns=stat.st_mtime_ns))
            return dict(directory=directory, entries=result, offset=offset,
                        next_offset=min(len(entries), offset+limit), total=len(entries))
        return await asyncio.to_thread(listing)

    @tool(annotations=read)
    async def lab_artifact(path: str, offset: int = 0, limit: int = 16000,
                           section: str | None = None) -> CallToolResult:
        """Read text/preview or a validated native Figure NPZ summary through the existing reader.
        NPZ returns dataset shapes/units/refs and bounded recipe/fit/lineage metadata; section selects
        one exact saved section from section_names. Numeric arrays stay in the native Figure.
        """
        selected = workspace_path(path)
        if offset < 0 or not 1 <= limit <= 64000:
            raise ValueError("Use a nonnegative byte offset and a limit of 1..64000")
        def read_file():
            size = selected.stat().st_size
            if selected.suffix.lower() == ".npz":
                from zlc_data.figure_archive import read_archive
                from zlc_data.codec import dataset_revision_ref_to_tree, dataset_schema_to_tree
                info, arrays, datasets = read_archive(selected)
                def compact(value, depth=0):
                    if isinstance(value, dict):
                        if depth >= 8:
                            return {"keys": list(value)}
                        return {key: compact(item, depth+1) for key, item in value.items()}
                    if isinstance(value, list) and len(value) > 40:
                        return {"count": len(value), "first": [compact(item, depth+1) for item in value[:8]]}
                    if isinstance(value, list):
                        return [compact(item, depth+1) for item in value]
                    return value
                sections = info["sections"]
                if section is not None:
                    if section not in sections:
                        raise ValueError(f"Unknown Figure section {section!r}; available: {', '.join(sections)}")
                    sections = {section: sections[section]}
                result = dict(path=str(selected), format=info["schema"], name=info["name"],
                    section_names=list(info["sections"]), sections=compact(sections),
                    datasets={key: {"shape": list(snapshot.block.values.shape),
                                   "dtype": str(snapshot.block.values.dtype),
                                   "schema": compact(dataset_schema_to_tree(snapshot.block.schema)),
                                   "ref": dataset_revision_ref_to_tree(snapshot.ref)}
                              for key, snapshot in datasets.items()},
                    array_members={key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                   for key, value in arrays.items()})
                encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                if offset or len(encoded) > limit:
                    page, end = _utf8_page(encoded[offset:], offset, limit)
                    result = dict(path=str(selected), format=info["schema"], name=info["name"],
                                  section_names=list(info["sections"]), offset=offset, next_offset=end,
                                  total_bytes=len(encoded), text=page,
                                  note="Paged native Figure metadata JSON, not numeric image samples.")
                return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False, separators=(",", ":")))],
                                      structured_content=result)
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
            page, next_offset = _utf8_page(raw, offset, limit)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(dict(
                path=str(selected), offset=offset, next_offset=next_offset, total_bytes=size,
                text=page), ensure_ascii=False))])
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

    def live_handles():
        # Resolve the existing objects each time; no second ownership registry.
        result = {"experiment": (flow, None)}
        if flow.devices is not None:
            result["devices"] = (flow.devices.presenter, flow.devices)
        if flow.console_presenter is not None:
            result["console"] = (flow.console_presenter, flow.console)
        for key, control in flow.device_controls.items():
            target = getattr(control, "presenter", None)
            if target is None and callable(getattr(control, "control_action", None)):
                target = control
            result[f"control:{key}"] = (target, getattr(target, "view", control))
        return result

    def public_methods(target):
        # Only Python APIs declared by these application owners, not QObject
        # internals, attributes, drivers, or arbitrary object traversal.
        return {
            name: getattr(target, name)
            for cls in reversed(type(target).__mro__) if target is not None
            for name, member in vars(cls).items()
            if not name.startswith("_") and inspect.isfunction(member)
            and name != "invoke_public"
        }

    def view_intents(view):
        from PyQt5 import QtCore

        if view is None or not isinstance(view, QtCore.QObject):
            return {}
        meta = view.metaObject()
        result = {}
        for index in range(QtCore.QObject.staticMetaObject.methodCount(), meta.methodCount()):
            method = meta.method(index)
            name = bytes(method.name()).decode()
            if method.methodType() == QtCore.QMetaMethod.Signal and name.endswith(
                ("_requested", "_committed", "_toggled", "_changed", "_edited", "_selected", "_picked")
            ):
                result[name] = {"signature": bytes(method.methodSignature()).decode(),
                                "argument_types": [bytes(t).decode() for t in method.parameterTypes()]}
        return result

    def handle_snapshot(handle, target):
        for name in ("control_state", "snapshot", "execution_snapshot"):
            if inspect.isfunction(getattr(type(target), name, None)):
                return _control_plain(getattr(target, name)())
        return state() if handle == "experiment" else None

    @tool(annotations=read)
    async def lab_handles() -> dict[str, object]:
        """Discover the live experiment/Devices/Console/open Control handles. No new session is opened.
        These are the actual application owners, not a second per-feature MCP action registry.
        """
        return await owner(lambda: {key: {"type": None if obj is None else type(obj).__name__,
                                         "api_methods": len(public_methods(obj)),
                                         "gui_intents": len(view_intents(view))}
                                    for key, (obj, view) in live_handles().items()})

    @tool(annotations=read)
    async def lab_api(handle: str, method: str | None = None) -> dict[str, object]:
        """Discover public API signatures/docstrings directly from a live owner, plus its real GUI intents.
        Start with handle='devices'; its load/save/describe_type APIs work before Init. With no method this is
        a compact method index; request a named method for arguments/docs. Parameters/schema come from the
        application, not a separate MCP whitelist. Private attributes, raw drivers and Qt internals are not APIs.
        """
        def describe():
            target, view = live_handles()[handle]
            methods = public_methods(target)
            if method is None:
                return dict(handle=handle, methods=sorted(methods), gui_intents=view_intents(view),
                            note="Use lab_call with explicit arguments; GUI intents may open dialogs. Prefer request_* asynchronous execution APIs.")
            selected = methods[method]
            signature = inspect.signature(selected)
            return dict(handle=handle, method=method, signature=str(signature),
                        doc=inspect.getdoc(selected) or "", parameters=[
                            dict(name=p.name, kind=p.kind.name, required=p.default is inspect.Parameter.empty,
                                 default=None if p.default is inspect.Parameter.empty else _control_plain(p.default),
                                 annotation=None if p.annotation is inspect.Parameter.empty else str(p.annotation))
                            for p in signature.parameters.values() if not p.name.startswith("_")
                        ])
        return await owner(describe)

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
        async def lab_review_points(node_id: str, generation: str, request_id: str,
                                    excluded_ids: list[str] | None = None) -> dict[str, object]:
            """Inspect or confirm the current scientific site-selection review in its original GUI.
            Omit excluded_ids to read the point coordinates, labels and current checks. Supply the complete
            list of excluded point IDs (empty keeps all) to press the normal Continue button. Use the exact
            generation/request_id from lab_state; stale or duplicate responses are rejected. This cannot
            answer physical manual-axis requests. logic.stop uses the original review's Stop path.
            """
            return await owner(lambda: presenter().control_review_points(
                node_id, generation, request_id, excluded_ids))

        @tool()
        async def lab_call(handle: str, method: str, arguments: dict[str, object],
                           expected_revision: int | None = None, include_state: bool = True) -> dict[str, object]:
            """Call a public method on the actual GUI/business owner; use lab_api only if its signature is unknown.
            This also reaches new public application methods with no MCP-specific tool changes. Explicit path
            arguments resolve inside this workspace; present=False is supplied when the method supports it.
            Prefer asynchronous request_* APIs, not blocking hardware calls. Returns the method's real result
            and available owner state, NOT a blanket success acknowledgement. None may mean only submission;
            check pending/error/terminal state afterward. Never replay an uncertain write.
            Set include_state=False when only the result is needed. Calls that already return the owner's
            snapshot do not append a duplicate state; use result instead.
            """
            values = dict(arguments)
            if any(name.startswith("_") for name in values):
                raise ValueError("Private parameters are not part of the public API")
            if "path" in values and values["path"] not in (None, ""):
                values["path"] = str(workspace_path(str(values["path"])))
            def invoke():
                target, _view = live_handles()[handle]
                selected = public_methods(target)[method]
                signature = inspect.signature(selected)
                if "present" in signature.parameters:
                    values["present"] = False
                signature.bind(**values)
                if expected_revision is not None:
                    current = handle_snapshot(handle, target)
                    if not isinstance(current, dict) or current.get("revision") != expected_revision:
                        raise ValueError("Owner revision changed or is unavailable; inspect state before editing")
                dispatcher = getattr(target, "invoke_public", None)
                result = dispatcher(method, values) if callable(dispatcher) else selected(**values)
                reply = dict(handle=handle, method=method, result=_control_plain(result))
                if include_state and method not in {"control_state", "snapshot", "execution_snapshot"}:
                    reply["state"] = handle_snapshot(handle, target)
                return reply
            return await owner(invoke)

        @tool()
        async def lab_intent(handle: str, intent: str, arguments: list[object]) -> dict[str, object]:
            """Fallback: emit an existing public GUI Handle intent on the Qt owner, as the human UI does.
            Discover actual signals/signatures through lab_api. This can open ordinary dialogs; prefer lab_call
            with explicit path/value arguments. It is not arbitrary QWidget manipulation or Python execution.
            Use lab_close for closing and Shut down devices so confirmation questions and choices return to the AI.
            Dispatch is not successful completion: inspect owner state/errors or the actual UI afterward.
            """
            def emit():
                target, view = live_handles()[handle]
                description = view_intents(view)[intent]
                if len(arguments) != len(description["argument_types"]):
                    raise ValueError(f"{intent} expects {len(description['argument_types'])} arguments")
                getattr(view, intent).emit(*arguments)
                return dict(dispatched=True, handle=handle, intent=intent,
                            state=handle_snapshot(handle, target))
            return await owner(emit)

        @tool()
        async def lab_editor_action(device_id: str, action: str, arguments: dict[str, object],
                                    expected_revision: int | None = None) -> dict[str, object]:
            """Use an opened Pulse/SLM editor's discovered action, via the same authoring/worker as the GUI.
            Consult lab_editor_catalog; path arguments are workspace-relative or absolute inside the workspace.
            An empty path stays empty for actions that explicitly clear a file reference; a null path is omitted.
            Saving/applying/solving can be asynchronous; inspect pending/errors afterward. expected_revision rejects
            edits against a stale draft. This does not override normal device ownership or stale-device protections.
            """
            arguments = dict(arguments)
            if arguments.get("path") is None:
                arguments.pop("path", None)
            elif arguments["path"] != "":
                arguments["path"] = str(workspace_path(str(arguments["path"])))
            def submit():
                selected = editor(device_id)
                current = selected.control_state()
                if expected_revision is not None and current.get("revision") != expected_revision:
                    raise ValueError("Editor revision changed; inspect current state before editing")
                return selected.control_action(action, arguments)
            return await owner(submit)

        @tool()
        async def lab_save(panel_id: str, path: str, refresh: bool = False,
                           layout_path: str | None = None) -> dict[str, object]:
            """Save a native Panel Figure (image plus NPZ), optionally also its reusable Layout.
            path is a workspace .png/.pdf/.svg destination. Saves the existing Frozen snapshot, exactly
            like Edit panel → Save figure. refresh=True explicitly freezes current data first; false never silently
            replaces an existing frozen view. To save the latest accepted results, use refresh=True;
            this updates the save snapshot, not a workaround to restart live fitting.
            Submission is asynchronous: read panel.saving and owner
            reports/errors afterward, then lab_artifact on the resulting NPZ. No external renderer.
            """
            if refresh:
                refreshed = await lab_action("panel.refresh", {"panel_id": panel_id})
                if not _step_succeeded(refreshed):
                    return refreshed
                deadline = monotonic() + 30.
                for field in ("refresh_pending", "configuration_pending"):
                    settled = await lab_wait_for("lab_state", {"panel_id": panel_id},
                        ["console", "panels", 0, field], False, max(0., deadline - monotonic()))
                    if not _step_succeeded(settled):
                        return dict(accepted=False, saved=False,
                                    error="Refresh has not completed; no Figure was saved", state=settled["result"])
                error = settled["result"]["console"]["panels"][0]["snapshot_error"]
                if error:
                    return dict(accepted=False, saved=False, error=f"Refresh failed; no Figure was saved: {error}",
                                state=settled["result"])
            result = await lab_action("panel.save", {"panel_id": panel_id, "selected": path})
            if _step_succeeded(result) and layout_path is not None:
                result["layout"] = await lab_action("layout.save", {"path": layout_path})
                if not _step_succeeded(result["layout"]):
                    result["error"] = "The Figure save was submitted, but Layout save failed; inspect layout.error"
                    result["accepted"] = False
            return result

        @tool()
        async def lab_devices(action: Literal["inspect", "load", "save"],
                              path: str | None = None) -> dict[str, object]:
            """Inspect or load/save the native apparatus draft without initializing hardware.
            Load requires an existing workspace apparatus JSON. Save uses the Devices codec.
            This never implicitly initializes devices or shuts down a Session; use lab_initialize.
            """
            selected = None if path is None else workspace_path(path)
            if action == "load" and (selected is None or not selected.is_file()):
                raise ValueError("Load requires an existing apparatus JSON file in this workspace")
            def command():
                if flow.devices is None:
                    raise RuntimeError("Devices window is unavailable")
                manager = flow.devices.presenter
                if action == "inspect":
                    return state()
                if manager.device_operation_active:
                    return dict(accepted=False, error="Device operation is still pending", state=state())
                result = manager.load(selected) if action == "load" else manager.save(selected, present=False)
                return dict(accepted=bool(result), result=_control_plain(result), state=state(),
                            error=None if result else "Devices rejected the file operation; inspect device_status")
            return await owner(command)

        @tool()
        async def lab_close(scope: Literal["session", "experiment", "editor"] = "session",
                            device_id: str | None = None,
                            decision: Literal["discard", "cancel"] | None = None) -> dict[str, object]:
            """Close the Session, application, or one editor through its existing lifecycle.
            First call without decision. Unsaved drafts return needs_confirmation with the actual
            editor question, draft paths and discard/cancel choices; nothing starts closing yet.
            Decide autonomously from the current task and draft context, then call the same scope/device
            with decision, or save the drafts before closing. This is a tool interaction, not a requirement
            to ask the human about routine task decisions. Cancel preserves the drafts and current Session.
            closing means asynchronous cleanup was accepted, not hardware shutdown completed.
            For Session shutdown observe lab_state.session_open=false; experiment close ends this endpoint.
            """
            return await owner(lambda: flow.control_close(scope, device_id=device_id, decision=decision))

        @tool()
        async def lab_run_recipe(name: str, parameters: dict[str, object],
                                 start: bool = True) -> dict[str, object]:
            """Apply a native Layout recipe; start its existing follower processors and one entry node.
            No independent recipe run exists: returned node_id/generation are the original Runtime run.
            start=False only prepares and validates. Existing active Logic is never stopped implicitly;
            no automatic Init, Calibration, SLM Send, retries or per-shot Fire. Read lab_recipes(name) first.
            A partial failure leaves already applied authoring/actions visible; inspect state, do not replay.
            """
            from ..console_layout import read_experiment_recipe

            recipe, document, _defaults = await asyncio.to_thread(
                read_experiment_recipe, flow.space.root, name, parameters)
            def prepare():
                control = presenter()
                before = set(control.panels)
                followers = tuple(recipe["followers"])
                if len(set(followers)) != len(followers) or recipe["entry_node"] in followers:
                    raise ValueError("A recipe starts each follower once, separately from its entry node")
                declared = {entry.node_id: entry for entry in document.logic}
                for node in followers:
                    if control.catalog.get(declared[node].api_name).kind.value != "processor":
                        raise ValueError("Recipe followers must be standing processors, not device executions")
                accepted = control.apply_layout(document, stop_running=False)
                return dict(accepted=bool(accepted), previous_panels=before)
            prepared = await owner(prepare)
            if not prepared["accepted"]:
                return dict(accepted=False, started=False, error="Layout was refused; inspect the existing owner reports",
                            state=await owner(state))
            settled = await lab_wait_for("lab_state", {}, ["console", "layout_pending"], False, 30.)
            if not _step_succeeded(settled):
                return dict(accepted=True, started=False, error="Layout replacement is not complete; inspect state without reloading",
                            state=settled["result"])
            def validate():
                control = presenter()
                ids = [key for key in control.panels if key not in prepared["previous_panels"]]
                if len(ids) != len(document.panel_ids):
                    raise RuntimeError("Recipe panel mapping changed during preparation; inspect the current board")
                panel_map = dict(zip(document.panel_ids, ids, strict=True))
                checks = [control.control_action("logic.validate", {"node_id": key})
                          for key in (*recipe["followers"], recipe["entry_node"])]
                revisions = {key: control.logic[key].draft_revision
                             for key in (*recipe["followers"], recipe["entry_node"])}
                return panel_map, checks, revisions
            panel_map, checks, revisions = await owner(validate)
            result = dict(recipe=name, prepared=True, started=False,
                          node_id=recipe["entry_node"],
                          result_panels=[panel_map[key] for key in recipe["result_panels"]],
                          validation=[{key: value for key, value in check.items() if key != "state"}
                                      for check in checks],
                          completion=recipe["completion"], save=recipe["save"])
            if not all(_step_succeeded(check) for check in checks):
                return dict(result, accepted=False, error="Recipe inputs are not valid; inspect validation reasons")
            result["estimate"] = await lab_estimate(recipe["entry_node"])
            if not start:
                return dict(result, accepted=True)
            started = []
            for node_id in (*recipe["followers"], recipe["entry_node"]):
                def submit():
                    control = presenter()
                    return control.control_action("logic.start", {"node_id": node_id},
                        expected_draft_revision=revisions[node_id])
                response = await owner(submit)
                started.append({key: value for key, value in response.items() if key != "state"})
                if not _step_succeeded(response):
                    return dict(result, accepted=False, actions=started,
                                state=response.get("state"),
                                error="A native Start was refused; earlier actions remain, do not replay the recipe")
            entry = next(row for row in response["state"]["logic"] if row["node_id"] == recipe["entry_node"])
            return dict(result, accepted=True, started=True, generation=entry.get("generation"),
                        status=entry["status"], actions=started)

        @tool()
        async def lab_initialize() -> dict[str, object]:
            """Initialize the apparatus currently authored in Devices. Never toggles an existing session off.
            This opens actual devices. Acceptance is not completion or experiment readiness.
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
                                            "logic.stop", "logic.revert", "logic.remove", "logic.rename",
                                            "panel.add", "panel.update",
                                            "panel.remove", "panel.edit", "panel.refresh", "panel.save",
                                            "view.monitor", "view.logic", "layout.save", "layout.load"],
                             arguments: dict[str, object], expected_draft_revision: int | None = None,
                             expected_generation: str | None = None) -> dict[str, object]:
            """Use the normal experiment action. Consult lab_catalog only for unknown fields. logic.add: api_name, optional values,
            source_signal, device_keys, artifact_inputs, name. logic.rename: node_id, name (text; "" is the
            node_id) -- changes only what the operator sees; node_id, signal keys and every reference stay.
            Node edits: node_id plus values,
            source_signal, device_keys, artifact_inputs (a path per input; a frame_claims input such as
            Occupancy's calibrations takes [{path, frames}] rows; a row with frames [] is kept on purpose and
            never needs deleting). Start/Stop: node_id. logic.start on a running,
            following or queued node is the GUI Restart: it stops the current run and queues the replacement; the reply
            still carries the old generation (pending.generation null), so read the new one from lab_state once
            pending is null. Editing a running, following or queued node changes only its draft;
            lab_state logic[].edited_since_start is true while that draft differs from the configuration the
            current run started with; lab_state(details=True) shows it as logic[].running_draft. logic.revert
            (node_id) makes that whole configuration the draft again without restarting; it is refused when there
            is no current run or nothing differs. Value names outside the node's schema are refused.
            Draft issues name fields by their on-screen labels; map them back through
            lab_catalog(node_id).node.source_label and form_spec. Panel add: kind; update: panel_id,patch.
            view.monitor/view.logic take empty arguments and navigate directly, without pixel clicks.
            layout.save/load take path (workspace JSON); load restores stopped drafts. Loading over active Logic
            requires explicit stop_running=true; poll state.layout_pending until the replacement completes.
            Panel kind must be one of lab_catalog.panel_kinds (for example image, not Image).
            For panel.update read lab_catalog(panel_id).panel.update for accepted patch fields and the
            current parameter_surface for nested keys. The catalog's state is a Layout record, not a patch.
            Save: panel_id,selected workspace image path ending .png/.pdf/.svg; the standard matching science .npz
            is saved too. Saving is asynchronous: inspect subsequent reports/errors. Manual requests stay human.
            Use observed expected_draft_revision for Start and expected_generation for Stop to reject stale intents.
            """
            arguments = dict(arguments)
            if action == "panel.save":
                if not arguments.get("selected"):
                    raise ValueError("panel.save needs selected (a workspace .png/.pdf/.svg path)")
                arguments["selected"] = str(workspace_path(arguments["selected"]))
            if action in {"layout.save", "layout.load"}:
                if not arguments.get("path"):
                    raise ValueError(f"{action} needs path (a workspace Layout JSON)")
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

    @tool(annotations=read)
    async def lab_wait_for(tool_name: str, arguments: dict[str, object], path: list[str | int],
                           equals: object, timeout_seconds: float = 30) -> dict[str, object]:
        """Wait for an explicit JSON field of an existing operator read-only tool to equal a supplied value.
        No experiment stages or readiness rules are inferred. path is a list of object keys/list indices.
        Uses the same SDK tool dispatch, never writes/retries an action; timeout only ends observation.
        Returns matched/timed_out and the last result. Other requests, including Stop, remain available.
        """
        if not 0 <= timeout_seconds <= 1800 or tool_name in {"lab_wait_for", "lab_wait"}:
            raise ValueError("Use timeout_seconds 0..1800 and a non-wait read-only tool")
        if tool_name in _DEVELOPER_TOOLS:
            raise ValueError("Waiting applies to operator read tools, not developer reflection")
        descriptor = next((item for item in await mcp.list_tools() if item.name == tool_name), None)
        if descriptor is None or descriptor.annotations is None or not descriptor.annotations.read_only_hint:
            raise ValueError("Waiting requires an existing read-only tool")
        deadline = monotonic() + timeout_seconds
        # Each read is a full owner projection on the GUI thread, which also
        # draws the live plots: re-read quickly at first, then back off.
        delay = .02
        while True:
            response = await mcp.call_tool(tool_name, arguments)
            if response.is_error:
                raise ValueError(str(response.content))
            result = response.structured_content
            if result is None:
                raise ValueError(f"{tool_name} returns no JSON fields to wait on")
            value = result
            try:
                for key in path:
                    value = value[key]
            except (KeyError, IndexError, TypeError):
                raise ValueError(f"field {path!r} is not in {tool_name}'s result") from None
            if value == equals:
                return dict(matched=True, timed_out=False, result=result)
            if monotonic() >= deadline:
                return dict(matched=False, timed_out=True, result=result)
            await asyncio.sleep(min(delay, max(0, deadline-monotonic())))
            delay = min(.25, delay*1.5)

    if allow_control:
        @tool()
        async def lab_batch(calls: list[dict[str, object]]) -> dict[str, object]:
            """Execute a supplied ordered list of already-decided tool calls without intervening model turns.
            Each item: tool, arguments, optional expect (exact top-level result fields). Uses the official SDK's
            existing registered tools and validation, not a separate action registry or experiment workflow.
            An argument {"$result":[0,"state","logic",0,"generation"]} reads that JSON field from step0's
            result. Only earlier completed steps; no expressions/conditions/loops or automatic retries.
            Include explicit waits for asynchronous work. Stops on an error/refusal/unmet expectation; previous
            effects remain and are never rolled back or replayed. No nested batches. At most 16 calls, 30 s total.
            Refusal, error, unmet wait, cancelled run or retired generation stops the list; expect adds conditions,
            never overrides those failures. No implicit retries or interpretation of scientific data.
            The caller must not batch steps that require interpreting new scientific data or human decisions.
            """
            if not 1 <= len(calls) <= 16:
                raise ValueError("Supply 1..16 tool calls")
            names = {item.name for item in await mcp.list_tools()}
            for item in calls:
                if (set(item) - {"tool", "arguments", "expect"} or item.get("tool") not in names
                        or item.get("tool") == "lab_batch" or not isinstance(item.get("arguments"), dict)
                        or not isinstance(item.get("expect", {}), dict)):
                    raise ValueError("Each call needs an existing non-batch tool, arguments, and optional expect object")
            results = []
            deadline = monotonic() + 30.
            for index, item in enumerate(calls):
                try:
                    arguments = _batch_arguments(item["arguments"], results)
                    response = await asyncio.wait_for(mcp.call_tool(item["tool"], arguments),
                                                      timeout=max(0., deadline-monotonic()))
                    result = response.structured_content
                    if result is None:
                        result = response.model_dump(mode="json")
                    ok = not response.is_error and _step_succeeded(result)
                    if isinstance(result, dict):
                        if item.get("expect"):
                            ok = ok and all(key in result and result[key] == value for key, value in item["expect"].items())
                    elif item.get("expect"):
                        ok = False
                    results.append(dict(tool=item["tool"], result=result, completed=ok))
                except Exception as error:
                    ok = False
                    note = ("Batch deadline expired; the last operation may have started. Inspect state before any retry."
                            if isinstance(error, TimeoutError) else _error_text(error))
                    results.append(dict(tool=item["tool"], error=note, completed=False))
                if not ok:
                    return dict(completed=False, stopped_at=index, results=results,
                                note="Earlier effects remain. Inspect the result; do not replay the entire batch.")
            return dict(completed=True, results=results)

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
        port = int(listener.getsockname()[1])
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

    close.connection_info = {"url": f"http://{host}:{port}/mcp", "port": port,
                             "workspace": str(flow.space.root.resolve()), "pid": os.getpid()}
    return close


async def _stdio(workspace: Path, *, tool_profile="operator") -> None:
    """Connect an AI client to an existing experiment; never launch devices."""
    import anyio
    import httpx2
    from urllib.parse import urlsplit
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server import Server, NotificationOptions
    from mcp.server.stdio import stdio_server
    from mcp_types import Tool, ListToolsResult

    workspace = workspace.resolve()
    directory = workspace / ".mcp"
    selected = None
    connection = None
    forwarding = 0
    switching = False

    def records():
        for path in sorted(directory.glob("connection-*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                url = urlsplit(record["url"])
                if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.path != "/mcp"
                        or Path(record["workspace"]).resolve() != workspace):
                    continue
                yield path.name, record
            except (OSError, ValueError, KeyError, TypeError):
                continue

    async def hold_connection(record, ready, closed, timeout):
        # AnyIO's transport/session contexts must enter and exit in this same
        # task. Only the transport and SDK schema caches persist, not lab state.
        try:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer " + record["token"]},
                                          timeout=httpx2.Timeout(timeout, read=None)) as http:
                async with streamable_http_client(record["url"], http_client=http) as (read, write, *_):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        if not ready.done():
                            ready.set_result(client)
                        await closed.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            raise

    async def open_connection(record, timeout=40):
        ready = asyncio.get_running_loop().create_future()
        closed = asyncio.Event()
        task = asyncio.create_task(hold_connection(record, ready, closed, timeout))
        try:
            return await ready, closed, task
        except BaseException:
            task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task, return_exceptions=True)
            raise

    async def close_connection(current):
        current[1].set()
        with anyio.CancelScope(shield=True):
            results = await asyncio.gather(current[2], return_exceptions=True)
        if isinstance(results[0], Exception):
            logging.getLogger(__name__).warning("MCP connection closed with error: %s", results[0])

    async def remote(record, name=None, arguments=None, params=None, timeout=40):
        nonlocal forwarding
        persistent = record == selected
        if persistent:
            if switching:
                raise RuntimeError("Experiment connection is changing; wait for lab_connect before issuing another operation")
            if connection is None or connection[2].done():
                raise RuntimeError("Experiment connection is closed; use lab_connect to reconnect, then inspect its state")
            current = connection
            forwarding += 1
        else:
            current = await open_connection(record, timeout)
        try:
            # No request lock: a long wait must not block Stop. No retry: a
            # failed response can mean the operation already started.
            if name is None:
                return await current[0].list_tools(params=params)
            if name in {"lab_wait", "lab_wait_for"}:
                timeout = max(timeout, float((arguments or {}).get("timeout_seconds", 30)) + 10.)
            return await current[0].call_tool(name, arguments or {}, read_timeout_seconds=timeout)
        finally:
            if persistent:
                forwarding -= 1
            else:
                await close_connection(current)

    async def available():
        async def check(filename, record):
            item = {k: v for k, v in record.items() if k != "token"}
            item["connection"] = filename
            try:
                if record == selected and connection is not None and connection[2].done():
                    # Explicit discovery may inspect a disconnected endpoint,
                    # but never replays the operation that lost its response.
                    probe = await open_connection(record, timeout=3)
                    try:
                        reply = await probe[0].call_tool("lab_health", {}, read_timeout_seconds=3)
                    finally:
                        await close_connection(probe)
                else:
                    reply = await remote(record, "lab_health", timeout=3)
                health = reply.structured_content or {}
                item["live"] = (not reply.is_error and health.get("service") == "running"
                                and health.get("pid") == record["pid"]
                                and Path(health.get("workspace", "")).resolve() == workspace)
            except Exception as error:
                item.update(live=False, error=str(error))
            return item, record
        return await asyncio.gather(*(check(name, record) for name, record in records()))

    async def connect(filename=None):
        nonlocal selected, connection, switching
        if switching:
            raise RuntimeError("Experiment connection is already changing; wait for lab_connect to finish")
        candidates = [(item, record) for item, record in await available()
                      if item["live"] and (filename is None or item["connection"] == filename)]
        if len(candidates) != 1:
            raise RuntimeError("No unique live experiment. Start experiment.bat, then use lab_connections and "
                               "lab_connect(connection=<filename>) to choose the intended window.")
        item, record = candidates[0]
        if switching:
            raise RuntimeError("Experiment connection is already changing; wait for lab_connect to finish")
        if record == selected and connection is not None and not connection[2].done():
            return item
        if forwarding:
            raise RuntimeError("Cannot change experiment while requests are in flight; wait for them to finish or Stop the current experiment first")
        switching = True
        try:
            replacement = await open_connection(record)
            try:
                if connection is not None:
                    await close_connection(connection)
            except BaseException:
                await close_connection(replacement)
                raise
            selected, connection = record, replacement
            return item
        finally:
            switching = False

    read_hint = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
    connection_tools = [
        Tool(name="lab_connections", description="List this workspace's local experiment endpoints and verify which are live. Never exposes credentials or starts hardware.",
             input_schema={"type": "object", "properties": {}, "additionalProperties": False}, annotations=read_hint),
        Tool(name="lab_connect", description="Attach this AI connection to one running experiment. With multiple windows choose the connection filename from lab_connections; no experiment is started or replayed.",
             input_schema={"type": "object", "properties": {"connection": {"type": "string"}}, "additionalProperties": False}),
    ]

    async def list_tools(ctx, params):
        try:
            if selected is None:
                await connect()
            reply = await remote(selected, params=params)
            offered = [item for item in reply.tools
                       if tool_profile == "developer" or item.name not in _DEVELOPER_TOOLS]
            return ListToolsResult(tools=[*connection_tools, *offered], next_cursor=reply.next_cursor)
        except Exception:
            return ListToolsResult(tools=connection_tools)

    async def call_tool(ctx, params):
        try:
            hidden_batch = params.name == "lab_batch" and any(
                isinstance(call, dict) and call.get("tool") in _DEVELOPER_TOOLS
                for call in (params.arguments or {}).get("calls", ())
            )
            if tool_profile != "developer" and (params.name in _DEVELOPER_TOOLS or hidden_batch):
                raise ValueError("Reflection is a developer fallback. Use native operator tools or explicitly connect with --tools developer.")
            if params.name == "lab_connections":
                result = {"workspace": str(workspace), "connections": [item for item, _ in await available()]}
            elif params.name == "lab_connect":
                result = await connect((params.arguments or {}).get("connection"))
                await ctx.session.send_tool_list_changed()
            else:
                if selected is None:
                    await connect()
                return await remote(selected, params.name, params.arguments)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
                                  structured_content=result)
        except Exception as error:
            return CallToolResult(is_error=True, content=[TextContent(type="text", text=
                f"{error}. No operation was replayed; inspect the experiment before retrying a write.")])

    server = Server("ZLC current experiment", on_list_tools=list_tools, on_call_tool=call_tool,
                    instructions=_OPERATOR_INSTRUCTIONS + " If no experiment is "
                    "running yet, keep this connection and use lab_connect after starting experiment.bat. "
                    + (" Developer profile includes lab_handles/lab_api/lab_call for diagnosis."
                       if tool_profile == "developer" else
                       " Operator profile exposes native actions; start with lab_guide, not reflection.") +
                    " Never replay an uncertain write.")
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options(NotificationOptions(tools_changed=True)))
    finally:
        if connection is not None:
            await close_connection(connection)


def client_config(workspace: Path, client: str, *, tool_profile: str = "operator") -> str:
    """Print a native client connection using this experiment's interpreter."""
    import sys
    import zou_lab_control

    command = str(Path(sys.executable).resolve())
    args = ["-m", "zou_lab_control", "mcp", "--workspace", str(workspace.resolve())]
    if tool_profile != "operator":
        args.extend(("--tools", tool_profile))
    # An editable checkout can launch experiment.bat without being installed in
    # site-packages. A client starts elsewhere: name this same product explicitly.
    environment = {"PYTHONPATH": str(Path(zou_lab_control.__file__).resolve().parent.parent)}
    if client == "claude":
        return json.dumps({"mcpServers": {"zlc": {"type": "stdio", "command": command,
                                                   "args": args, "env": environment}}}, indent=2, ensure_ascii=False)
    if client != "codex":
        raise ValueError("client must be codex or claude")
    return ("[mcp_servers.zlc]\n"
            f"command = {json.dumps(command, ensure_ascii=False)}\n"
            f"args = {json.dumps(args, ensure_ascii=False)}\n"
            "required = true\nstartup_timeout_sec = 30\ntool_timeout_sec = 1860\n"
            'default_tools_approval_mode = "approve"\n'
            "[mcp_servers.zlc.env]\n"
            f"PYTHONPATH = {json.dumps(environment['PYTHONPATH'], ensure_ascii=False)}\n")


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Connect an MCP client to an already-running experiment.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--tools", choices=("operator", "developer"), default="operator",
                        help="Operator tools by default; developer also exposes owner API reflection.")
    parser.add_argument("--client-config", choices=("codex", "claude"),
                        help="Print native client configuration, without changing any client settings.")
    arguments = parser.parse_args(argv)
    if arguments.client_config:
        print(client_config(arguments.workspace, arguments.client_config, tool_profile=arguments.tools))
        return 0
    asyncio.run(_stdio(arguments.workspace, tool_profile=arguments.tools))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
