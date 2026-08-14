"""Panel Edit Save Fig: one frozen plot, one dataset, one run chain."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from zlc_plot.primitives import ImageFrame, ImagePointOverlay, PointStatus

from .archive import write_figure_file
from .panel_state import PanelFrozenData, PanelState
from .plot_annotations import PanelPlotAnnotations


__all__ = [
    "PanelFigureFiles",
    "capture_run_chain",
    "overlay_payload",
    "panel_plot_annotations_section",
    "restore_panel_plot_input",
    "restore_panel_plot_annotations",
    "save_panel_data",
    "save_panel_figure",
    "save_panel_image",
]


@dataclass(frozen=True, slots=True)
class PanelFigureFiles:
    image: Path
    archive: Path


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    return str(value)


def capture_run_chain(
    signal_plane: object,
    publication: object | None,
) -> tuple[Mapping[str, Any], ...]:
    """Copy exact parent records while the runtime still retains the chain."""

    if publication is None:
        return ()
    parents_of = getattr(signal_plane, "direct_parent_publications", None)
    if not callable(parents_of):
        raise TypeError("signal plane cannot resolve exact parent publications")
    visited: set[int] = set()
    records: list[Mapping[str, Any]] = []

    def visit(current: object) -> None:
        identity = id(current)
        if identity in visited:
            return
        visited.add(identity)
        for parent in parents_of(current):
            visit(parent)
        run_record = getattr(current, "run_record", {})
        entry = _plain(run_record)
        if not isinstance(entry, dict):
            raise TypeError("signal publication run_record must be a mapping")
        entry["signals"] = [str(name) for name in getattr(current, "signals", {})]
        records.append(entry)

    visit(publication)
    return tuple(records)


def overlay_payload(
    plot_input: object,
    annotation: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not isinstance(plot_input, ImageFrame):
        return {}, {}
    overlay = plot_input.overlay
    arrays: dict[str, np.ndarray] = {
        "overlay.coordinates": np.asarray(overlay.coordinates, dtype="<f8"),
    }
    section: dict[str, Any] = {
        "dataset": "data",
        "revision": int(overlay.revision),
        **_plain(annotation),
        "coordinates": "overlay.coordinates",
    }
    if overlay.point_ids is not None:
        arrays["overlay.point_ids"] = np.asarray(overlay.point_ids, dtype="U")
        section["point_ids"] = "overlay.point_ids"
    if overlay.labels is not None:
        arrays["overlay.labels"] = np.asarray(
            tuple("" if label is None else label for label in overlay.labels),
            dtype="U",
        )
        section["labels"] = "overlay.labels"
        section["blank_label_is_none"] = True
    if overlay.statuses is not None:
        # One row per picture the overlay describes: the facet coordinates it
        # names, then NaN for the whole-picture row.  Rings that differ from
        # cell to cell are the record; one flattened row would keep the
        # geometry and throw the measurement away.
        keys = tuple(overlay.statuses)
        arrays["overlay.status_facets"] = np.asarray(
            tuple(np.nan if key is None else float(key) for key in keys),
            dtype="<f8",
        )
        arrays["overlay.statuses"] = np.asarray(
            tuple(
                tuple(status.value for status in overlay.statuses[key])
                for key in keys
            ),
            dtype="U",
        ).reshape((len(keys), overlay.count))
        section["status_facets"] = "overlay.status_facets"
        section["statuses"] = "overlay.statuses"
    return arrays, section


def panel_plot_annotations_section(
    annotations: PanelPlotAnnotations,
    *,
    dataset: str,
) -> dict[str, object]:
    """Archive one generic producer-authored plot annotation bundle."""

    if not isinstance(annotations, PanelPlotAnnotations):
        raise TypeError("annotations must be PanelPlotAnnotations")
    document = dict(annotations.document())
    if not document:
        return {}
    return {"dataset": str(dataset), **document}


def restore_panel_plot_annotations(
    info: Mapping[str, Any],
    dataset: str,
) -> PanelPlotAnnotations:
    """Restore generic annotations without interpreting their producer."""

    section = info.get("sections", {}).get("plot_annotations")
    if not isinstance(section, Mapping) or section.get("dataset") != str(dataset):
        return PanelPlotAnnotations()
    return PanelPlotAnnotations.from_document(section)


def _await(operation: object) -> object:
    return operation.result() if hasattr(operation, "result") else operation


def _panel_paths(base_path: str | Path) -> tuple[Path, Path]:
    selected = Path(base_path).expanduser().resolve()
    if not selected.suffix:
        selected = selected.with_suffix(".png")
    if selected.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("panel image format must be PNG, PDF, or SVG")
    selected.parent.mkdir(parents=True, exist_ok=True)
    return selected, selected.with_suffix(".npz")


def save_panel_image(
    base_path: str | Path,
    *,
    state: PanelState,
    frozen: PanelFrozenData,
    make_host: Callable[[object, str, str, str], object],
    configure_host: Callable[[object, PanelState, ImagePointOverlay | None], None],
) -> Path:
    """Draw one panel's frozen input and write the picture.

    The expensive half.  A producer taking data one shot at a time writes the
    NUMBERS as they arrive and comes back for the pictures afterwards -- a
    render costs a few hundred milliseconds, which inside an acquisition loop
    is long enough for a camera's ring to overrun.
    """

    image_path, _archive_path = _panel_paths(base_path)
    plot_input = frozen.snapshot if frozen.plot_input is None else frozen.plot_input
    host = make_host(
        frozen.snapshot, state.signal, state.kind, state.cell_kind
    )
    try:
        configure_host(
            host,
            state,
            plot_input.overlay if isinstance(plot_input, ImageFrame) else None,
        )
        save = getattr(host, "save", None)
        if not callable(save):
            raise TypeError("panel plotting host cannot save an image")
        _await(save(image_path))
    finally:
        close = getattr(host, "close", None)
        if callable(close):
            close()
    return image_path


def save_panel_data(
    base_path: str | Path,
    *,
    state: PanelState,
    frozen: PanelFrozenData,
    annotations: PanelPlotAnnotations | None = None,
) -> Path:
    """Write one panel's numbers, and everything needed to explain them.

    The cheap half, and the one that cannot be recomputed: the dataset with
    its axes, the panel it was configured as, the run chain it came out of,
    the overlay and the annotations.  Drawing needs none of it, so a producer
    can secure the data the moment it exists.
    """

    _image_path, archive_path = _panel_paths(base_path)
    plot_input = frozen.snapshot if frozen.plot_input is None else frozen.plot_input
    overlay_arrays, overlay_section = overlay_payload(
        plot_input,
        frozen.overlay,
    )
    sections: dict[str, Any] = {
        "panel": {
            "dataset": "data",
            "state": state.document(),
        },
        "run_chain": [_plain(record) for record in frozen.run_chain],
    }
    if overlay_section:
        sections["overlay"] = overlay_section
    annotation_section = panel_plot_annotations_section(
        PanelPlotAnnotations() if annotations is None else annotations,
        dataset="data",
    )
    if annotation_section:
        sections["plot_annotations"] = annotation_section
    return write_figure_file(
        archive_path,
        name=archive_path.with_suffix(".png").name,
        arrays={"data": frozen.snapshot, **overlay_arrays},
        sections=sections,
    )


def save_panel_figure(
    base_path: str | Path,
    *,
    state: PanelState,
    frozen: PanelFrozenData,
    make_host: Callable[[object, str, str, str], object],
    configure_host: Callable[[object, PanelState, ImagePointOverlay | None], None],
    annotations: PanelPlotAnnotations | None = None,
) -> PanelFigureFiles:
    """Save the Edit tab's exact frozen input without asking the plane again.

    Both halves, for the caller that wants them together.  They are separately
    callable because a producer saving as it acquires wants them apart.
    """

    image_path = save_panel_image(
        base_path,
        state=state,
        frozen=frozen,
        make_host=make_host,
        configure_host=configure_host,
    )
    written = save_panel_data(
        base_path,
        state=state,
        frozen=frozen,
        annotations=annotations,
    )
    return PanelFigureFiles(image_path, written)


def restore_panel_plot_input(
    info: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    dataset_name: str,
    snapshot: object,
) -> object:
    """Rebuild saved Image annotation without reopening calibration JSON."""

    section = info.get("sections", {}).get("overlay")
    if not isinstance(section, Mapping) or section.get("dataset") != str(dataset_name):
        return snapshot
    coordinates = np.asarray(arrays[str(section["coordinates"])], dtype="<f8")

    point_ids = None
    if "point_ids" in section:
        point_ids = tuple(str(value) for value in arrays[str(section["point_ids"])])
    labels = None
    if "labels" in section:
        labels = tuple(
            None if str(value) == "" else str(value)
            for value in arrays[str(section["labels"])]
        )
    statuses = None
    if "statuses" in section:
        rows = np.atleast_2d(arrays[str(section["statuses"])])
        facets = np.asarray(arrays[str(section["status_facets"])], dtype="<f8")
        statuses = {
            (None if not np.isfinite(facet) else float(facet)): tuple(
                PointStatus(str(value)) for value in row
            )
            for facet, row in zip(facets, rows, strict=True)
        }
    overlay = ImagePointOverlay(
        int(section.get("revision", 0)),
        coordinates,
        point_ids,
        labels,
        statuses,
    )
    return ImageFrame(snapshot, overlay)
