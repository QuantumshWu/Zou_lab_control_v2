"""TaskConsole adapter for the shared data-backed Figure artifact core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from zlc_plot import save_figure_artifact
from .panel_state import PanelFrozenData, PanelState


__all__ = ["PanelFigureFiles", "capture_run_chain", "save_panel_figure"]


@dataclass(frozen=True, slots=True)
class PanelFigureFiles:
    image: Path
    archive: Path


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

    def visit(current: object) -> str:
        identity = id(current)
        existing = identities.get(identity)
        if existing is not None:
            return existing
        node_id = f"event-{len(identities) + 1}"
        identities[identity] = node_id
        parents = tuple(parents_of(current))
        nodes[node_id] = {
            "id": node_id,
            "event": _event_document(current),
            "parents": [visit(parent) for parent in parents],
            "signals": [str(name) for name in getattr(current, "signals", {})],
            "record": _plain(getattr(current, "run_record", {})),
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
    ordered = [nodes[key] for key in identities.values()]
    result: dict[str, object] = {
        "root": root,
        "nodes": ordered,
        "device_settings": [],
    }
    if resolve_device_settings is not None:
        if not callable(resolve_device_settings):
            raise TypeError("resolve_device_settings must be callable or None")
        result["device_settings"] = _plain(
            resolve_device_settings(
                tuple(node["event_record"] for node in ordered)
            )
        )
    return result


def save_panel_figure(
    base_path: str | Path,
    *,
    state: PanelState,
    frozen: PanelFrozenData,
    source: Mapping[str, object] | None = None,
) -> PanelFigureFiles:
    """Adapt one frozen panel to the shared archive-first Figure writer."""

    if state.signal != frozen.signal:
        raise ValueError("Panel Save target differs from its frozen surface")
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
    image, archive = save_figure_artifact(
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
    )
    return PanelFigureFiles(image, archive)
