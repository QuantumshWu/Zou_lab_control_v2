"""TaskConsole adapter for the shared data-backed Figure artifact core."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from .panel_state import PanelFrozenData, PanelState


__all__ = ["PanelFigureFiles", "capture_run_chain", "save_panel_figure"]


# A sealed Figure publication already owns a complete causal DAG.  The Runtime
# event used to expose it to Panel code is a transport boundary, not a new
# experiment, so lineage capture replaces that event with this inherited DAG.
_IMPORTED_LINEAGE_KEY = "zlc.figure.imported-lineage"


@dataclass(frozen=True, slots=True)
class PanelFigureFiles:
    image: Path
    archive: Path


def _panel_figure_files(written: object) -> PanelFigureFiles:
    if isinstance(written, PanelFigureFiles):
        return written
    if not isinstance(written, tuple) or len(written) != 2:
        raise TypeError("Figure writer must return image/archive paths")
    image, archive = written
    return PanelFigureFiles(Path(image), Path(archive))


def _typed_writer_result(written: object) -> PanelFigureFiles | Future:
    """Preserve the public result type across synchronous and C writers."""

    add_done = getattr(written, "add_done_callback", None)
    if not callable(add_done):
        return _panel_figure_files(written)

    result = Future()

    def completed(pending: object) -> None:
        try:
            cancelled = getattr(pending, "cancelled", None)
            if callable(cancelled) and cancelled():
                result.cancel()
                return
            value = pending.result()
            if not result.done():
                result.set_result(_panel_figure_files(value))
        except BaseException as error:
            if not result.done():
                result.set_exception(error)

    add_done(completed)
    return result


def _plain(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("lineage record keys must be text")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    raise TypeError(f"lineage contains unsupported {type(value).__name__}")


def _event_document(publication: object) -> dict[str, object]:
    ref = getattr(publication, "event_ref", None)
    if ref is None:
        raise TypeError("lineage publication has no event_ref")
    stream = getattr(getattr(ref, "stream_id", None), "value", None)
    generation = getattr(getattr(ref, "generation", None), "value", None)
    sequence = getattr(ref, "sequence", None)
    if not isinstance(stream, str) or not isinstance(generation, str) or type(sequence) is not int:
        raise TypeError("lineage event_ref is malformed")
    return {"stream": stream, "generation": generation, "sequence": sequence}


def capture_run_chain(
    signal_plane: object,
    publication: object | None,
    *,
    event_records: Mapping[object, object] | None = None,
    resolve_device_settings: object | None = None,
) -> dict[str, object]:
    """Capture the exact causal DAG rooted at one displayed publication."""

    if publication is None:
        return {"root": None, "nodes": [], "device_settings": []}
    parents_of = getattr(signal_plane, "direct_parent_publications", None)
    if not callable(parents_of):
        raise TypeError("signal plane cannot resolve exact parent publications")
    identities: dict[int, str] = {}
    nodes: dict[str, dict[str, object]] = {}
    exact_records = {} if event_records is None else dict(event_records)
    inherited_settings: list[object] = []
    visiting: set[int] = set()

    def imported(current: object) -> Mapping[str, object] | None:
        record = getattr(current, "run_record", {})
        if not isinstance(record, Mapping):
            return None
        candidate = record.get(_IMPORTED_LINEAGE_KEY)
        if candidate is None:
            return None
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "root",
            "nodes",
            "device_settings",
        }:
            raise ValueError("imported Figure lineage fields differ")
        if not isinstance(candidate["nodes"], (list, tuple)) or not isinstance(
            candidate["device_settings"], (list, tuple)
        ):
            raise TypeError("imported Figure lineage arrays are malformed")
        return candidate

    def visit(current: object) -> str:
        identity = id(current)
        existing = identities.get(identity)
        if existing is not None:
            return existing
        boundary = imported(current)
        if boundary is not None:
            root = boundary["root"]
            if not isinstance(root, str):
                raise ValueError("imported Figure lineage needs one root")
            for raw in boundary["nodes"]:
                if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
                    raise TypeError("imported Figure lineage node is malformed")
                node = _plain(raw)
                node_id = str(node["id"])
                previous = nodes.setdefault(node_id, node)
                if previous != node:
                    raise ValueError("imported Figure lineage node ids collide")
            if root not in nodes:
                raise ValueError("imported Figure lineage root is missing")
            inherited_settings.extend(_plain(boundary["device_settings"]))
            identities[identity] = root
            return root
        if identity in visiting:
            raise ValueError("Runtime causal publications contain a cycle")
        visiting.add(identity)
        parents = tuple(parents_of(current))
        parent_ids = [visit(parent) for parent in parents]
        visiting.remove(identity)
        serial = len(nodes) + 1
        node_id = f"event-{serial}"
        while node_id in nodes:
            serial += 1
            node_id = f"event-{serial}"
        identities[identity] = node_id
        record = dict(getattr(current, "run_record", {}))
        record.pop(_IMPORTED_LINEAGE_KEY, None)
        nodes[node_id] = {
            "id": node_id,
            "event": _event_document(current),
            "parents": parent_ids,
            "signals": [str(name) for name in getattr(current, "signals", {})],
            "record": _plain(record),
            "event_record": _plain(
                exact_records.get(
                    current,
                    getattr(current, "event_record", {}),
                )
            ),
        }
        return node_id

    root = visit(publication)
    if any(id(current) not in identities for current in exact_records):
        raise ValueError("exact event record lies outside the captured lineage")
    ordered = list(nodes.values())
    result: dict[str, object] = {
        "root": root,
        "nodes": ordered,
        "device_settings": inherited_settings,
    }
    if resolve_device_settings is not None:
        if not callable(resolve_device_settings):
            raise TypeError("resolve_device_settings must be callable or None")
        result["device_settings"] = [
            *inherited_settings,
            *_plain(
                resolve_device_settings(
                    tuple(node["event_record"] for node in ordered)
                )
            ),
        ]
    return result


def save_panel_figure(
    base_path: str | Path,
    *,
    state: PanelState,
    frozen: PanelFrozenData,
    writer: Callable[..., object],
    source: Mapping[str, object] | None = None,
    host: object | None = None,
) -> PanelFigureFiles | Future:
    """Submit one frozen panel through the caller's Figure writer.

    This module owns the Workbench-to-Figure payload projection.  The writer
    owns execution: TaskConsole and FigureViewer pass the dedicated Edit/Save
    render process here, so this adapter never imports or constructs a local
    plotting host.

    ``source`` is the caller's source document, carried into the archive
    with the frozen signal, title and overlay written over it.  ``host`` is
    the settled Edit surface when one exists.  If it does not, the injected
    C-process writer builds and closes its temporary host inside C.
    """

    if state.signal != frozen.signal:
        raise ValueError("Panel Save target differs from its frozen surface")
    if not callable(writer):
        raise TypeError("Panel Save writer must be callable")
    description = frozen.description
    source_document = dict(source or {})
    source_document.pop("overlay_signal", None)
    source_document.update(
        {
            "signal": frozen.signal,
            "title": state.title,
            **dict(frozen.overlay),
        }
    )
    return _typed_writer_result(
        writer(
            base_path,
            plot_input=frozen.plot_input,
            spec=description.spec,
            parameters=description.display_state.values,
            size=description.size,
            viewport=description.viewport,
            classifier_thresholds=description.classifier_thresholds,
            facet_focus=description.facet_focus,
            fit=description.fit,
            selectors=description.selectors,
            lineage=frozen.lineage,
            source=source_document,
            host=host,
        )
    )
