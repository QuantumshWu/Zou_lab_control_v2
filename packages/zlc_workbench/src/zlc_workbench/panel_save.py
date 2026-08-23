"""TaskConsole adapter for the shared data-backed Figure artifact core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from zlc_plot import save_figure_artifact
from zlc_plot.selectors import RectangleRange

from .panel_catalog import task_console_fitting_spec
from .panel_state import PanelFrozenData, PanelState, project_panel_state


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
) -> dict[str, object]:
    """Capture the exact causal DAG rooted at one displayed publication."""

    if publication is None:
        return {"root": None, "nodes": []}
    parents_of = getattr(signal_plane, "direct_parent_publications", None)
    if not callable(parents_of):
        raise TypeError("signal plane cannot resolve exact parent publications")
    identities: dict[int, str] = {}
    nodes: dict[str, dict[str, object]] = {}

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
        }
        return node_id

    root = visit(publication)
    return {"root": root, "nodes": [nodes[key] for key in identities.values()]}


def _recipe(
    state: PanelState,
    frozen: PanelFrozenData,
    viewport: RectangleRange | None,
) -> tuple[object, dict[str, object]]:
    plot_input = frozen.snapshot if frozen.plot_input is None else frozen.plot_input
    snapshot = getattr(plot_input, "snapshot", plot_input)
    spec = task_console_fitting_spec(
        snapshot.block.schema, state.kind, state.cell_kind
    )
    if spec is None:
        raise ValueError(f"{state.signal!r} cannot be drawn as {state.kind!r}")
    spec, _semantic, parameters = project_panel_state(
        snapshot.block.schema, spec, state
    )
    return plot_input, {
        "spec": spec,
        "parameters": parameters,
        "size": state.size,
        "viewport": viewport,
        "classifier_thresholds": state.classifier_thresholds,
        "facet_focus": state.focused_cell,
        "fit": state.fit,
    }


def save_panel_figure(
    base_path: str | Path,
    *,
    state: PanelState,
    frozen: PanelFrozenData,
    viewport: RectangleRange | None,
) -> PanelFigureFiles:
    """Adapt one frozen panel to the shared archive-first Figure writer."""

    plot_input, recipe = _recipe(state, frozen, viewport)
    image, archive = save_figure_artifact(
        base_path,
        plot_input=plot_input,
        lineage=frozen.lineage,
        source={
            "signal": state.signal,
            "title": state.title,
            **dict(frozen.overlay),
        },
        **recipe,
    )
    return PanelFigureFiles(image, archive)
