"""Data-backed figures: one exact plot recipe, archive first, image second."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from zlc_data import (
    LATEST_COORDINATE,
    OwnedSnapshot,
    canonical_coordinate_scalar,
    snapshot_from_manifest,
    snapshot_manifest,
)
from zlc_data.figure_archive import read_dataset, write_figure_archive
from zlc_durable import atomic_write_file

from .config import DEFAULTS
from .kinds import AxisDomain, AxisRef, PlotKind
from .primitives import ImageFrame, ImagePointOverlay, PointStatus
from .selectors import NumericRange, RectangleRange
from .specs import (
    CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, PlotLabels,
    Reduction, RollingPlot, parameter_schema_for,
)


def _plain(value: object) -> object:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("figure metadata keys must be text")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    raise TypeError(f"figure metadata contains unsupported {type(value).__name__}")


def _keys(value: object, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields differ from {sorted(expected)!r}")
    return value


def _axis_document(axis: AxisRef | None) -> object:
    return None if axis is None else {"domain": axis.domain.value, "axis_id": axis.axis_id}


def _axis(value: object, name: str) -> AxisRef | None:
    if value is None:
        return None
    entry = _keys(value, {"domain", "axis_id"}, name)
    return AxisRef(AxisDomain(entry["domain"]), entry["axis_id"])


def _labels_document(labels: PlotLabels) -> dict[str, object]:
    return {name: getattr(labels, name) for name in ("title", "x", "y", "value")}


def _labels(value: object) -> PlotLabels:
    entry = _keys(value, {"title", "x", "y", "value"}, "plot labels")
    return PlotLabels(**dict(entry))


def _scope_document(scope: object) -> list[object]:
    return [
        {
            "axis": _axis_document(axis),
            "coordinate": (
                {"kind": "latest"}
                if coordinate is LATEST_COORDINATE
                else {"kind": "value", "value": coordinate}
            ),
        }
        for axis, coordinate in scope
    ]


def _scope(value: object) -> tuple[tuple[AxisRef, object], ...]:
    if not isinstance(value, list):
        raise TypeError("plot scope must be an array")
    terms = []
    for item in value:
        entry = _keys(item, {"axis", "coordinate"}, "plot scope term")
        axis = _axis(entry["axis"], "plot scope axis")
        if axis is None:
            raise ValueError("plot scope axis cannot be null")
        coordinate = entry["coordinate"]
        if not isinstance(coordinate, Mapping) or coordinate.get("kind") not in {
            "latest",
            "value",
        }:
            raise ValueError("plot scope coordinate is not tagged")
        if coordinate["kind"] == "latest":
            _keys(coordinate, {"kind"}, "latest plot scope coordinate")
            resolved = LATEST_COORDINATE
        else:
            _keys(coordinate, {"kind", "value"}, "plot scope coordinate")
            resolved = canonical_coordinate_scalar(
                coordinate["value"], "plot scope coordinate"
            )
        terms.append((axis, resolved))
    return tuple(terms)


def _encode_plot_spec(spec: object) -> dict[str, object]:
    if not isinstance(
        spec, (CurvePlot, ImagePlot, HistogramPlot, RollingPlot, FacetGridPlot)
    ):
        raise TypeError("unsupported data-backed plot specification")
    common = {"kind": spec.kind.value, "labels": _labels_document(spec.labels)}
    common["scope"] = _scope_document(spec.scope)
    if isinstance(spec, HistogramPlot):
        return common
    if spec.kind is PlotKind.FACET_GRID:
        return {
            **common,
            "facet": _axis_document(spec.facet),
            "cell": _encode_plot_spec(spec.cell),
        }
    common["reduction"] = spec.reduction.value
    if isinstance(spec, (CurvePlot, RollingPlot)):
        common["group"] = _axis_document(spec.group)
    if isinstance(spec, (CurvePlot, ImagePlot)):
        common["x"] = _axis_document(spec.x)
    if isinstance(spec, ImagePlot):
        common["y"] = _axis_document(spec.y)
    if not isinstance(spec, (CurvePlot, ImagePlot, RollingPlot)):
        raise TypeError("unsupported plot specification")
    return common


def _decode_plot_spec(value: object) -> object:
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        raise TypeError("plot recipe spec must be an object with text kind")
    kind = PlotKind(value["kind"])
    if kind is PlotKind.PULSE_TIMELINE:
        raise ValueError("Pulse timelines are not data-backed Figure artifacts")
    base = {"labels": _labels(value.get("labels"))}
    base["scope"] = _scope(value.get("scope"))
    if kind is PlotKind.HISTOGRAM:
        _keys(value, {"kind", "labels", "scope"}, "histogram recipe")
        return HistogramPlot(**base)
    if kind is PlotKind.FACET_GRID:
        _keys(value, {"kind", "labels", "scope", "facet", "cell"}, "facet recipe")
        return FacetGridPlot(
            facet=_axis(value["facet"], "facet axis"),
            cell=_decode_plot_spec(value["cell"]),
            **base,
        )
    expected = {"kind", "labels", "scope", "reduction"}
    arguments = {
        **base,
        "reduction": Reduction(value.get("reduction")),
    }
    if kind in {PlotKind.CURVE, PlotKind.ROLLING}:
        expected.add("group")
        arguments["group"] = _axis(value.get("group"), "plot group")
    if kind in {PlotKind.CURVE, PlotKind.IMAGE}:
        expected.add("x")
        arguments["x"] = _axis(value.get("x"), "plot x")
    if kind is PlotKind.IMAGE:
        expected.add("y")
        arguments["y"] = _axis(value.get("y"), "plot y")
    _keys(value, expected, f"{kind.value} recipe")
    factory = {PlotKind.CURVE: CurvePlot, PlotKind.IMAGE: ImagePlot, PlotKind.ROLLING: RollingPlot}.get(kind)
    if factory is None:
        raise ValueError(f"unsupported data-backed plot kind {kind.value!r}")
    return factory(**arguments)


def _viewport_document(value: RectangleRange | None) -> object:
    if value is None:
        return None
    return {"x": [value.x.low, value.x.high], "y": [value.y.low, value.y.high]}


def _viewport(value: object) -> RectangleRange | None:
    if value is None:
        return None
    entry = _keys(value, {"x", "y"}, "plot viewport")
    axes = []
    for name in ("x", "y"):
        bounds = entry[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"plot viewport {name} must contain two bounds")
        axes.append(NumericRange(*bounds))
    return RectangleRange(*axes)


def _overlay_payload(plot_input: object, prefix: str) -> tuple[dict[str, np.ndarray], object]:
    if not isinstance(plot_input, ImageFrame):
        return {}, None
    overlay = plot_input.overlay
    arrays = {f"{prefix}.coordinates": np.asarray(overlay.coordinates, dtype="<f8")}
    document: dict[str, object] = {"revision": overlay.revision, "coordinates": f"{prefix}.coordinates"}
    for name, values in (("point_ids", overlay.point_ids), ("labels", overlay.labels)):
        if values is not None:
            key = f"{prefix}.{name}"
            arrays[key] = np.asarray(tuple("" if item is None else item for item in values), dtype="U")
            document[name] = key
    if overlay.static_statuses is not None:
        key = f"{prefix}.static_statuses"
        arrays[key] = np.asarray(tuple(item.value for item in overlay.static_statuses), dtype="U")
        document["static_statuses"] = key
    if overlay.status is not None:
        stored: dict[str, np.ndarray] = {}
        manifest = snapshot_manifest(
            overlay.status, stored, values_key=f"{prefix}.status",
            validity_key=f"{prefix}.status.validity",
        )
        ref = dict(manifest["ref"])
        ref.pop("schema_fingerprint", None)
        manifest["ref"] = ref
        document["status"] = manifest
        arrays.update(stored)
    return arrays, document


def _restore_overlay(snapshot: OwnedSnapshot, arrays: Mapping[str, np.ndarray], value: object) -> object:
    if value is None:
        return snapshot
    allowed = {"revision", "coordinates", "point_ids", "labels", "static_statuses", "status"}
    if not isinstance(value, Mapping) or not {"revision", "coordinates"} <= set(value) or set(value) - allowed:
        raise ValueError("figure overlay fields differ")
    def strings(name: str) -> tuple[str, ...] | None:
        return None if name not in value else tuple(str(item) for item in arrays[str(value[name])])
    labels = strings("labels")
    return ImageFrame(snapshot, ImagePointOverlay(
        int(value["revision"]), np.asarray(arrays[str(value["coordinates"])], dtype="<f8"),
        strings("point_ids"), None if labels is None else tuple(None if item == "" else item for item in labels),
        None if "static_statuses" not in value else tuple(PointStatus(item) for item in strings("static_statuses")),
        None if "status" not in value else snapshot_from_manifest(value["status"], arrays, embedded=True),
    ))


def encode_plot_recipe(
    spec: object, *, parameters: Mapping[str, object], size: str,
    viewport: RectangleRange | None = None, classifier_thresholds: object = (),
    facet_focus: int | None = None, fit: Mapping[str, object] | None = None,
    overlay: object = None,
) -> dict[str, object]:
    complete_parameters = dict(
        parameter_schema_for(spec, style=DEFAULTS.style).initial_values(parameters)
    )
    selected_size = DEFAULTS.layout.validate_preset(size)
    return {
        "spec": _encode_plot_spec(spec), "parameters": _plain(complete_parameters), "size": selected_size,
        "viewport": _viewport_document(viewport),
        "classifier_thresholds": _plain(list(classifier_thresholds)), "facet_focus": facet_focus,
        "fit": _plain({} if fit is None else fit), "overlay": overlay,
    }


def decode_plot_recipe(value: object) -> dict[str, object]:
    entry = _keys(value, {"spec", "parameters", "size", "viewport", "classifier_thresholds", "facet_focus", "fit", "overlay"}, "plot recipe")
    if not isinstance(entry["parameters"], Mapping) or not isinstance(entry["fit"], Mapping):
        raise TypeError("plot recipe parameters and fit must be objects")
    if not isinstance(entry["size"], str) or not entry["size"]:
        raise ValueError("plot recipe size must be non-empty text")
    thresholds = entry["classifier_thresholds"]
    if not isinstance(thresholds, list):
        raise TypeError("plot recipe classifier_thresholds must be an array")
    focus = entry["facet_focus"]
    if focus is not None and (isinstance(focus, bool) or not isinstance(focus, int) or focus < 0):
        raise TypeError("plot recipe facet_focus must be a nonnegative integer or null")
    return {
        "spec": _decode_plot_spec(entry["spec"]), "parameters": dict(entry["parameters"]),
        "size": entry["size"], "viewport": _viewport(entry["viewport"]),
        "classifier_thresholds": tuple(thresholds), "facet_focus": focus,
        "fit": dict(entry["fit"]), "overlay": entry["overlay"],
    }


def read_figure_plot(
    info: Mapping[str, Any], arrays: Mapping[str, np.ndarray], dataset: str,
) -> tuple[object, dict[str, object]]:
    recipes = info.get("sections", {}).get("plot")
    datasets = info.get("sections", {}).get("dataset")
    if not isinstance(recipes, Mapping) or not isinstance(datasets, Mapping) or set(recipes) != set(datasets):
        raise ValueError("figure must carry one exact plot recipe per dataset")
    recipe = decode_plot_recipe(recipes[str(dataset)])
    snapshot = read_dataset(info, arrays, str(dataset))
    return _restore_overlay(snapshot, arrays, recipe.pop("overlay")), recipe


def build_figure_host(
    plot_input: object, spec: object, *, parameters: Mapping[str, object], size: str,
) -> object:
    """The shared TaskConsole/FigureViewer raster-host construction path."""

    from .raster import RasterPlotHost

    return RasterPlotHost.from_plot(plot_input, spec, size=size, parameters=parameters)


def open_figure_host(plot_input: object, recipe: Mapping[str, object]) -> object:
    """Build and configure the one raster host described by a decoded recipe."""

    expected = {
        "spec", "parameters", "size", "viewport", "classifier_thresholds",
        "facet_focus", "fit",
    }
    entry = _keys(recipe, expected, "decoded plot recipe")
    host = build_figure_host(
        plot_input, entry["spec"], size=entry["size"], parameters=entry["parameters"],
    )
    try:
        pending = host.configure(
            viewport=entry["viewport"],
            classifier_thresholds=entry["classifier_thresholds"],
            facet_focus=entry["facet_focus"],
            fit=entry["fit"],
            fit_live=False,
        )
        if hasattr(pending, "result"):
            pending.result()
    except BaseException:
        host.close()
        raise
    return host


def save_figure_artifact(
    base_path: str | Path, *, plot_input: object, spec: object,
    parameters: Mapping[str, object], size: str, viewport: RectangleRange | None = None,
    classifier_thresholds: object = (), facet_focus: int | None = None,
    fit: Mapping[str, object] | None = None, lineage: Mapping[str, object] | None = None,
    source: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    selected = Path(base_path).expanduser().resolve()
    image_path = selected if selected.suffix else selected.with_suffix(".png")
    if image_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("figure image format must be PNG, PDF, or SVG")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = image_path.with_suffix(".npz")
    snapshot = plot_input.snapshot if isinstance(plot_input, ImageFrame) else plot_input
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("data-backed figure requires an OwnedSnapshot")
    overlay_arrays, overlay = _overlay_payload(plot_input, "data.overlay")
    recipe = encode_plot_recipe(
        spec, parameters=parameters, size=size, viewport=viewport,
        classifier_thresholds=classifier_thresholds, facet_focus=facet_focus,
        fit=fit, overlay=overlay,
    )
    source_document = dict(source or {})
    title = getattr(getattr(spec, "labels", None), "title", None)
    if title is not None:
        source_document.setdefault("title", title)
    sections = {
        "plot": {"data": recipe},
        "lineage": _plain(
            {"root": None, "nodes": [], "device_settings": []}
            if not lineage
            else lineage
        ),
        "source": _plain(source_document),
    }
    atomic_write_file(
        archive_path,
        lambda stream: write_figure_archive(
            stream, image_path.name, arrays={"data": snapshot, **overlay_arrays}, sections=sections,
        ),
    )
    decoded = decode_plot_recipe(recipe)
    decoded.pop("overlay")
    host = open_figure_host(plot_input, decoded)
    try:
        saved = host.save(image_path)
        if hasattr(saved, "result"):
            saved.result()
    except Exception as error:
        raise RuntimeError(f"figure archive {archive_path} was saved, but image rendering failed: {error}") from error
    finally:
        host.close()
    return image_path, archive_path


__all__ = [
    "build_figure_host", "decode_plot_recipe",
    "encode_plot_recipe", "open_figure_host",
    "read_figure_plot", "save_figure_artifact",
]
