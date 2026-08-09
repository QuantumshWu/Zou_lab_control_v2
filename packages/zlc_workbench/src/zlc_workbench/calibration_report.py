"""Calibration FINAL outputs projected onto the one shared plot stack.

The neutral-atom package owns SiteMap, model features, thresholds and
held-out fidelity.  ``zlc_plot`` owns every rendered primitive.  This module
is the deliberately small composition seam between them: it validates one
complete FINAL sibling bundle and authors the same three report pages used by
the authoritative v1 workflow.  It contains no Qt and draws no Matplotlib
artist itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
import numpy as np

from zlc_atom.nodes.calibration import CALIBRATION_DATASET_DECLARATIONS
from zlc_data import (
    COMPONENT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    CoordinateFrameId,
    OwnedSnapshot,
    PointColumn,
    expand_snapshot_validity,
)
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImageFrame,
    ImagePlot,
    PlotLabels,
    PointStatus,
)
from zlc_plot.primitives import ImagePointOverlay, PointMarker
from zlc_plot.specs import PlotSpec
from zlc_runtime import FinalDatasetOutput, SignalPublication

from .archive import read_archive, read_dataset, write_figure_file
from .panel_save import (
    overlay_payload,
    panel_plot_annotations_section,
    restore_panel_plot_annotations,
    restore_panel_plot_input,
)
from .panel_state import PanelState
from .plot_annotations import PanelPlotAnnotations
from .prepared_panel import (
    PreparedPanelSurface,
    create_prepared_panel_surface,
)
from .task_reports import TaskReportAdapter, TaskReportExport


__all__ = [
    "CalibrationReportFiles",
    "CalibrationReportPage",
    "CalibrationReportSurface",
    "calibration_report_pages",
    "calibration_report_pages_from_publication",
    "calibration_task_report_adapter",
    "create_calibration_report_surface",
    "export_calibration_report",
    "export_calibration_report_surfaces",
    "load_calibration_report_page",
]


_PAGE_KEYS = ("site_map", "fidelity", "distribution")
_DECLARATIONS = {
    declaration.name: declaration
    for declaration in CALIBRATION_DATASET_DECLARATIONS
}


@dataclass(frozen=True, slots=True)
class CalibrationReportPage:
    """One ordinary plot page and the report-owned configuration it needs."""

    key: str
    title: str
    signal: str
    plot_input: OwnedSnapshot | ImageFrame
    spec: PlotSpec
    facet_thresholds: tuple[float | None, ...] = ()
    fit_model: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if key not in _PAGE_KEYS:
            raise ValueError(f"unknown Calibration report page {key!r}")
        title = str(self.title).strip()
        signal = str(self.signal).strip()
        if not title or not signal:
            raise ValueError("Calibration report page title and signal are required")
        if not isinstance(self.plot_input, (OwnedSnapshot, ImageFrame)):
            raise TypeError("report plot_input must be OwnedSnapshot or ImageFrame")
        if not isinstance(
            self.spec,
            (CurvePlot, ImagePlot, FacetGridPlot),
        ):
            raise TypeError("Calibration report uses Image, Curve or FacetGrid specs")
        thresholds = tuple(self.facet_thresholds)
        if any(
            value is not None and not np.isfinite(float(value))
            for value in thresholds
        ):
            raise ValueError("report facet thresholds must be finite or None")
        fit_model = None if self.fit_model is None else str(self.fit_model).strip()
        if key == "distribution":
            if not isinstance(self.spec, FacetGridPlot) or not isinstance(
                self.spec.cell,
                HistogramPlot,
            ):
                raise TypeError("distribution page must be FacetGrid[Histogram]")
            if not thresholds or fit_model is None:
                raise ValueError("distribution page requires thresholds and a fit model")
        elif thresholds or fit_model is not None:
            raise ValueError("only the distribution page owns thresholds and a fit")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "facet_thresholds", thresholds)
        object.__setattr__(self, "fit_model", fit_model)
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )

    @property
    def snapshot(self) -> OwnedSnapshot:
        return (
            self.plot_input.snapshot
            if isinstance(self.plot_input, ImageFrame)
            else self.plot_input
        )

    @property
    def fit_options(self) -> Mapping[str, object]:
        if self.fit_model is None:
            return MappingProxyType({})
        return MappingProxyType(
            {
                "model": self.fit_model,
                "live": False,
                "fit_all_facets": True,
            }
        )

    def panel_state(self, *, signal: str, interval_ms: int) -> PanelState:
        """Project this typed page into the Workbench's one panel state."""

        cell_kind = (
            self.spec.cell.kind.value
            if isinstance(self.spec, FacetGridPlot)
            else ""
        )
        return PanelState(
            str(signal),
            self.spec.kind.value,
            "4x4" if isinstance(self.spec, FacetGridPlot) else "2x2",
            int(interval_ms),
            self.title,
            cell_kind=cell_kind,
            semantic=_semantic_state(self),
            display=_display_state(self),
            fit=self.fit_options,
            site_overlay=(
                "centers" if isinstance(self.plot_input, ImageFrame) else "off"
            ),
        )


@dataclass(frozen=True, slots=True)
class CalibrationReportFiles:
    images: tuple[Path, ...]
    archives: tuple[Path, ...]


CalibrationReportSurface = PreparedPanelSurface


def _calibration_outputs(
    outputs: Mapping[str, FinalDatasetOutput],
) -> dict[str, FinalDatasetOutput]:
    values = dict(outputs)
    if set(values) != set(_DECLARATIONS):
        raise ValueError(
            "Calibration report requires the complete declared FINAL output bundle"
        )
    ordered: dict[str, FinalDatasetOutput] = {}
    generations = set()
    revisions = set()
    run_records: list[dict[str, object]] = []
    for name, declaration in _DECLARATIONS.items():
        value = values[name]
        if not isinstance(value, FinalDatasetOutput):
            raise TypeError("Calibration report outputs must be FinalDatasetOutput values")
        if value.declaration != declaration:
            raise ValueError(f"Calibration output declaration differs for {name!r}")
        generations.add(value.snapshot.ref.stream_generation)
        revisions.add(value.snapshot.ref.revision)
        run_records.append(dict(value.run_record or {}))
        ordered[name] = value
    if len(generations) != 1 or len(revisions) != 1:
        raise ValueError("Calibration report outputs do not belong to one FINAL revision")
    if any(record != run_records[0] for record in run_records[1:]):
        raise ValueError("Calibration report outputs carry different run records")
    return ordered


def _site_column(snapshot: OwnedSnapshot, name: str) -> PointColumn:
    columns = tuple(
        column
        for column in snapshot.block.schema.point_table.columns
        if column.role == SITE
    )
    if len(columns) != 1:
        raise ValueError(f"{name} must carry exactly one SITE point coordinate")
    column = columns[0]
    if any(value is None for value in column.values):
        raise ValueError(f"{name} SITE identity cannot contain missing values")
    return column


def _scalar_site_snapshot(
    snapshot: OwnedSnapshot,
    name: str,
    *,
    repeats: int | None,
) -> PointColumn:
    schema = snapshot.block.schema
    if repeats is not None and schema.repeat_axis.size != repeats:
        raise ValueError(f"{name} must contain exactly {repeats} repeat")
    if schema.cell_schema.is_scalar is not True:
        raise ValueError(f"{name} must contain one scalar value per SITE")
    return _site_column(snapshot, name)


def _image_axes(snapshot: OwnedSnapshot) -> tuple[object, object, CoordinateFrameId]:
    schema = snapshot.block.schema
    if schema.repeat_axis.size != 1 or schema.point_table.row_count != 1:
        raise ValueError("Calibration site_map must contain one image")
    by_role = {
        role: tuple(axis for axis in schema.cell_schema.data_axes if axis.role == role)
        for role in (SPATIAL_X, SPATIAL_Y)
    }
    if any(len(axes) != 1 for axes in by_role.values()) or len(
        schema.cell_schema.data_axes
    ) != 2:
        raise ValueError("Calibration site_map must declare one X and one Y image axis")
    x_axis = by_role[SPATIAL_X][0]
    y_axis = by_role[SPATIAL_Y][0]
    if (
        x_axis.coordinate_frame is None
        or x_axis.coordinate_frame != y_axis.coordinate_frame
    ):
        raise ValueError("Calibration site_map axes must share a coordinate frame")
    if x_axis.unit != "pixel" or y_axis.unit != "pixel":
        raise ValueError("Calibration site_map axes must be expressed in pixels")
    return x_axis, y_axis, x_axis.coordinate_frame


def _centers(
    snapshot: OwnedSnapshot,
    site_column: PointColumn,
    frame: CoordinateFrameId,
) -> tuple[np.ndarray, np.ndarray]:
    schema = snapshot.block.schema
    current_site = _site_column(snapshot, "fidelity_centers")
    if current_site != site_column:
        raise ValueError("Calibration centers and diagnostics use different SITE identity")
    axes = schema.cell_schema.data_axes
    if schema.repeat_axis.size != 1 or len(axes) != 1 or axes[0].role != COMPONENT:
        raise ValueError("Calibration centers must have one COMPONENT axis")
    component = axes[0]
    if component.size != 2 or tuple(
        component.coordinate_at(index) for index in range(2)
    ) != ("x", "y"):
        raise ValueError("Calibration center components must be authored as x,y")
    if (
        component.unit != "pixel"
        or component.coordinate_frame != frame
        or schema.cell_schema.value_unit != "pixel"
    ):
        raise ValueError("Calibration centers must use the site_map pixel frame")
    values = np.asarray(snapshot.block.values, dtype=float)
    expected = (1, len(site_column.values), 2)
    if values.shape != expected:
        raise ValueError(f"Calibration centers have shape {values.shape}, expected {expected}")
    if not np.isfinite(values).all():
        raise ValueError("Calibration centers must be finite")
    validity = np.asarray(expand_snapshot_validity(snapshot), dtype=bool)
    if validity.shape != expected:
        raise ValueError("Calibration center validity differs from its values")
    return values[0], np.all(validity[0], axis=1)


def _thresholds(
    snapshot: OwnedSnapshot,
    site_column: PointColumn,
) -> tuple[float | None, ...]:
    current_site = _scalar_site_snapshot(
        snapshot,
        "fidelity_threshold",
        repeats=1,
    )
    if current_site != site_column:
        raise ValueError("Calibration thresholds and samples use different SITE identity")
    values = np.asarray(snapshot.block.values, dtype=float).reshape(-1)
    validity = np.asarray(expand_snapshot_validity(snapshot), dtype=bool).reshape(-1)
    if values.shape != (len(site_column.values),) or validity.shape != values.shape:
        raise ValueError("Calibration thresholds do not follow the SITE coordinate")
    return tuple(
        float(value) if accepted and np.isfinite(value) else None
        for value, accepted in zip(values, validity, strict=True)
    )


def _site_map_page(
    snapshot: OwnedSnapshot,
    overlay: ImagePointOverlay,
) -> CalibrationReportPage:
    x_axis, y_axis, _frame = _image_axes(snapshot)
    spec = ImagePlot(
        AxisRef.data(str(x_axis.axis_id)),
        AxisRef.data(str(y_axis.axis_id)),
        labels=PlotLabels(
            title="Reference average | calibrated sites",
            x="X",
            y="Y",
            value="Counts",
        ),
    )
    return CalibrationReportPage(
        "site_map",
        "Site map",
        "site_map",
        ImageFrame(snapshot, overlay),
        spec,
        parameters={"show_point_labels": True, "site_overlay": "centers"},
    )


def _fidelity_page(
    snapshot: OwnedSnapshot,
    site_column: PointColumn,
) -> CalibrationReportPage:
    current_site = _scalar_site_snapshot(snapshot, "fidelity_site", repeats=1)
    if current_site != site_column:
        raise ValueError("Calibration fidelity and samples use different SITE identity")
    return CalibrationReportPage(
        "fidelity",
        "Fidelity",
        "fidelity_site",
        snapshot,
        CurvePlot(
            # v1 plots the stable SITE order as an ordinal curve.  The typed
            # SITE column remains the identity/alignment authority shared by
            # every report output; it is not coerced into a fake numeric
            # coordinate merely to satisfy Curve's continuous x contract.
            AxisRef.point_rows(),
            labels=PlotLabels(
                title="Per-site held-out fidelity",
                x="Site",
                y="Fidelity",
            ),
        ),
    )


def _distribution_page(
    snapshot: OwnedSnapshot,
    site_column: PointColumn,
    thresholds: tuple[float | None, ...],
) -> CalibrationReportPage:
    current_site = _scalar_site_snapshot(snapshot, "readout_samples", repeats=None)
    if current_site != site_column:
        raise ValueError("Calibration samples and diagnostics use different SITE identity")
    if snapshot.block.schema.repeat_axis.size < 1:
        raise ValueError("Calibration readout samples cannot be empty")
    if len(thresholds) != len(site_column.values):
        raise ValueError("Calibration thresholds must follow every SITE facet")
    return CalibrationReportPage(
        "distribution",
        "Distribution",
        "readout_samples",
        snapshot,
        FacetGridPlot(
            AxisRef.point(str(site_column.coordinate_id)),
            HistogramPlot(
                labels=PlotLabels(x="Readout signal", y="Count"),
            ),
            labels=PlotLabels(title="Per-site readout distributions"),
        ),
        thresholds,
        "bimodal_gaussian",
    )


def calibration_report_pages(
    outputs: Mapping[str, FinalDatasetOutput],
) -> tuple[CalibrationReportPage, ...]:
    """Author the three v1 report pages from one exact typed FINAL bundle."""

    values = _calibration_outputs(outputs)
    site_map = values["site_map"].snapshot
    _x_axis, _y_axis, frame = _image_axes(site_map)
    centers_snapshot = values["fidelity_centers"].snapshot
    site_column = _site_column(centers_snapshot, "fidelity_centers")
    centers, center_validity = _centers(centers_snapshot, site_column, frame)
    markers = tuple(
        PointMarker(
            str(site_id),
            float(center[0]),
            float(center[1]),
            PointStatus.UNKNOWN if accepted else PointStatus.INVALID,
            str(site_id),
        )
        for site_id, center, accepted in zip(
            site_column.values,
            centers,
            center_validity,
            strict=True,
        )
    )
    overlay = ImagePointOverlay.from_markers(
        int(site_map.ref.revision.value),
        markers,
    )
    thresholds = _thresholds(values["fidelity_threshold"].snapshot, site_column)
    pages = (
        _site_map_page(site_map, overlay),
        _fidelity_page(values["fidelity_site"].snapshot, site_column),
        _distribution_page(
            values["readout_samples"].snapshot,
            site_column,
            thresholds,
        ),
    )
    if tuple(page.key for page in pages) != _PAGE_KEYS:
        raise RuntimeError("Calibration report page order changed")
    return pages


def calibration_report_pages_from_publication(
    publication: SignalPublication,
) -> tuple[CalibrationReportPage, ...]:
    """Recover the typed FINAL bundle from one exact runtime publication."""

    if not isinstance(publication, SignalPublication):
        raise TypeError("publication must be SignalPublication")
    by_bare: dict[str, object] = {}
    for qualified, value in publication.signals.items():
        bare = str(qualified).rsplit("/", 1)[-1]
        if bare in by_bare:
            raise ValueError("Calibration publication contains duplicate bare outputs")
        by_bare[bare] = value
    if set(by_bare) != set(_DECLARATIONS):
        raise ValueError("publication is not one complete Calibration FINAL bundle")
    outputs: dict[str, FinalDatasetOutput] = {}
    for name, declaration in _DECLARATIONS.items():
        value = by_bare[name]
        if getattr(value, "coverage", object()) is not None or bool(
            getattr(value, "transient", True)
        ):
            raise ValueError("Calibration report requires terminal FINAL values")
        outputs[name] = FinalDatasetOutput(
            declaration,
            getattr(value, "snapshot"),
            publication.run_record,
        )
    return calibration_report_pages(outputs)


def calibration_task_report_adapter() -> TaskReportAdapter:
    """Declare Calibration's exact FINAL bundle to the generic report registry."""

    return TaskReportAdapter(
        "calibration.report.v1",
        "Calibration report",
        tuple(
            (declaration.name, declaration.contract_id)
            for declaration in CALIBRATION_DATASET_DECLARATIONS
        ),
        calibration_report_pages_from_publication,
        TaskReportExport(
            "artifact_path",
            "calibration.readout.v1",
            export_calibration_report_surfaces,
        ),
    )


def create_calibration_report_surface(
    page: CalibrationReportPage,
) -> CalibrationReportSurface:
    """Create and asynchronously configure one ordinary RasterPlotHost."""

    if not isinstance(page, CalibrationReportPage):
        raise TypeError("page must be CalibrationReportPage")
    return create_prepared_panel_surface(page)


def _semantic_state(page: CalibrationReportPage) -> dict[str, object]:
    spec = page.spec
    if isinstance(spec, ImagePlot):
        return {"x": spec.x, "y": spec.y, "reduction": spec.reduction}
    if isinstance(spec, CurvePlot):
        return {
            "x": spec.x,
            "group": spec.group,
            "reduction": spec.reduction,
        }
    if isinstance(spec, FacetGridPlot):
        return {"facet": spec.facet}
    raise TypeError(type(spec).__name__)


def _display_state(page: CalibrationReportPage) -> dict[str, object]:
    labels = (
        page.spec.cell.labels
        if isinstance(page.spec, FacetGridPlot)
        else page.spec.labels
    )
    values = {
        name: value
        for name, value in page.parameters.items()
        if name != "site_overlay"
    }
    values.update(
        {
            "title": page.spec.labels.title or labels.title or page.title,
            "x_label": labels.x,
            "y_label": labels.y,
            "value_label": labels.value,
        }
    )
    return {name: value for name, value in values.items() if value is not None}


def _write_page_archive(
    path: Path,
    page: CalibrationReportPage,
    *,
    state: PanelState,
    annotations: PanelPlotAnnotations,
) -> Path:
    arrays: dict[str, object] = {"data": page.snapshot}
    report: dict[str, object] = {
        "page": page.key,
        "title": page.title,
        "fit_model": page.fit_model,
    }
    overlay_arrays, overlay_section = overlay_payload(
        page.plot_input,
        {"resolved_mode": "centers", "source": "calibration_report"},
    )
    arrays.update(overlay_arrays)
    sections: dict[str, object] = {
        "panel": {"dataset": "data", "state": state.document()},
        "calibration_report": report,
    }
    if overlay_section:
        sections["overlay"] = overlay_section
    annotation_section = panel_plot_annotations_section(
        annotations,
        dataset="data",
    )
    if annotation_section:
        sections["plot_annotations"] = annotation_section
    return write_figure_file(
        path,
        name=page.key,
        arrays=arrays,
        sections=sections,
    )


def export_calibration_report(
    destination: str | Path,
    source: Mapping[str, FinalDatasetOutput] | Sequence[CalibrationReportPage],
) -> CalibrationReportFiles:
    """Export the same three configured pages as PNG and portable NPZ files."""

    pages = (
        calibration_report_pages(source)
        if isinstance(source, Mapping)
        else tuple(source)
    )
    if tuple(page.key for page in pages) != _PAGE_KEYS or any(
        not isinstance(page, CalibrationReportPage) for page in pages
    ):
        raise ValueError("Calibration report export requires its canonical three pages")
    surfaces: list[PreparedPanelSurface] = []
    try:
        for page in pages:
            surfaces.append(create_calibration_report_surface(page))
        return export_calibration_report_surfaces(destination, surfaces)
    finally:
        for surface in surfaces:
            surface.close()


def export_calibration_report_surfaces(
    destination: str | Path,
    surfaces: Sequence[PreparedPanelSurface],
) -> CalibrationReportFiles:
    """Write one report by reusing its already-configured ordinary hosts."""

    selected = tuple(surfaces)
    pages = tuple(surface.page for surface in selected)
    if tuple(getattr(page, "key", "") for page in pages) != _PAGE_KEYS or any(
        not isinstance(page, CalibrationReportPage) for page in pages
    ):
        raise ValueError("Calibration report export requires its canonical surfaces")
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    archives: list[Path] = []
    for surface, page in zip(selected, pages, strict=True):
        surface.wait()
        image_path = root / f"{page.key}.png"
        surface.host.save(image_path).result()
        front = surface.host.front
        if front is None:
            raise RuntimeError("Calibration report page has no raster front")
        archive_path = _write_page_archive(
            root / f"{page.key}.npz",
            page,
            state=surface.state,
            annotations=surface.annotations,
        )
        images.append(image_path)
        archives.append(archive_path)
    return CalibrationReportFiles(tuple(images), tuple(archives))


def load_calibration_report_page(path: str | Path) -> CalibrationReportPage:
    """Reload one report page from the common figure archive format."""

    info, arrays = read_archive(path)
    report = info.get("sections", {}).get("calibration_report")
    if not isinstance(report, Mapping):
        raise ValueError("figure archive is not a Calibration report page")
    key = str(report.get("page", ""))
    if key not in _PAGE_KEYS:
        raise ValueError("Calibration report archive has an unknown page")
    snapshot = read_dataset(info, arrays, "data")
    plot_input = restore_panel_plot_input(info, arrays, "data", snapshot)
    if key == "site_map":
        if not isinstance(plot_input, ImageFrame):
            raise ValueError("site_map archive lost its point overlay")
        return _site_map_page(snapshot, plot_input.overlay)
    site_column = _site_column(snapshot, key)
    if key == "fidelity":
        return _fidelity_page(snapshot, site_column)
    thresholds = restore_panel_plot_annotations(
        info,
        "data",
    ).facet_thresholds
    if not thresholds:
        raise ValueError("distribution archive lost its calibrated thresholds")
    page = _distribution_page(snapshot, site_column, thresholds)
    if str(report.get("fit_model", page.fit_model)) != page.fit_model:
        raise ValueError("distribution archive names another fit model")
    return page
