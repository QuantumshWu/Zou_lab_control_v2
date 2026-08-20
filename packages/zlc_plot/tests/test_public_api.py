from __future__ import annotations

import types

import zlc_plot


EXPECTED_NAMES = {
    "AxisRef",
    "BackendUnavailableError",
    "CurvePlot",
    "DEFAULTS",
    "DEFAULT_UNITS",
    "FacetGridPlot",
    "FitCancelled",
    "FitEvent",
    "FitModelSpec",
    "FitTarget",
    "HistogramPlot",
    "IMAGE_POINT_OVERLAY_CONTRACT",
    "IMAGE_POINT_OVERLAY_GEOMETRY_RECORD",
    "ImageFrame",
    "ImagePlot",
    "ImagePointOverlay",
    "NumericRange",
    "PlotKind",
    "PlotLabels",
    "PlotSession",
    "PlotSpec",
    "PointStatus",
    "PulseAnalogTrace",
    "PulseBlock",
    "PulseChannel",
    "PulseDacScanSegment",
    "PulseRepeatMarker",
    "PulseScanRegion",
    "PulseTimelineData",
    "PulseTimelinePlot",
    "PulseTimelineSelectionData",
    "Qt5ParameterPanel",
    "Qt5PlotWidget",
    "RasterPlotHost",
    "Reduction",
    "RollingPlot",
    "SelectionChange",
    "SelectorData",
    "SelectorKind",
    "Unit",
    "UnitRegistry",
    "__version__",
    "curve",
    "describe_semantics",
    "edit_plot_display",
    "ensure_qt5_application",
    "facet_grid",
    "fitting_spec",
    "histogram",
    "image_axes",
    "image_point_overlay_from_signal",
    "image_point_overlay_geometry",
    "image",
    "parameter_controls",
    "pulse_timeline",
    "resolve_unit",
    "rolling",
    "schema_summary",
    "show",
    "updated_spec",
    "PANEL_SIZE_NAMES",
    "recommended_pulse_preset",
}

def test_facade_names_are_exact_and_resolve() -> None:
    assert set(zlc_plot.__all__) == EXPECTED_NAMES
    assert len(zlc_plot.__all__) == len(EXPECTED_NAMES)
    public = [
        name
        for name in dir(zlc_plot)
        if not name.startswith("_")
        and not isinstance(getattr(zlc_plot, name), types.ModuleType)
    ]
    assert set(public) <= EXPECTED_NAMES - {"__version__"}
    assert all(hasattr(zlc_plot, name) for name in zlc_plot.__all__)
    assert zlc_plot.__version__ == "1.1.0"
