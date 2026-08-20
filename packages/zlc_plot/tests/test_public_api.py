from __future__ import annotations

from importlib import import_module
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

REMOVED_NAMES = {
    "ControlKind",
    "CrosshairPoint",
    "DisplayDescription",
    "FacetFitBatchResult",
    "FitDeadlineExceeded",
    "FitEngine",
    "FitModelRegistry",
    "FitOptions",
    "FitParameterDisplay",
    "FitResult",
    "FitScope",
    "FitSelection",
    "NotebookView",
    "ParameterControl",
    "PlotLibraryDefaults",
    "PointMarker",
    "RasterBuffer",
    "RasterFront",
    "RasterIdentity",
    "RasterOperation",
    "RectangleRange",
    "RegularImageFitInput",
    "RelimMode",
    "ReplaceSpecInitialState",
    "RevisionError",
    "SelectionData",
    "SelectionEvent",
    "SelectorState",
    "SemanticChoice",
    "SemanticDescription",
    "SemanticFeasibility",
    "SemanticField",
    "SessionRevisions",
    "UnitError",
    "ZLCPlotError",
    "axis_choices_for_schema",
    "builtin_fit_models",
    "default_spec",
    "replace_spec_initial_state",
    "semantic_controls",
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


def test_removed_names_are_not_facade_attributes_but_submodules_stay_usable() -> None:
    assert all(not hasattr(zlc_plot, name) for name in REMOVED_NAMES)
    paths = {
        "ControlKind": "zlc_plot.ui",
        "CrosshairPoint": "zlc_plot.selectors",
        "DisplayDescription": "zlc_plot.session",
        "FacetFitBatchResult": "zlc_plot.fit",
        "FitDeadlineExceeded": "zlc_plot.fit",
        "FitEngine": "zlc_plot.fit",
        "FitModelRegistry": "zlc_plot.fit",
        "FitOptions": "zlc_plot.fit",
        "FitParameterDisplay": "zlc_plot.fit",
        "FitResult": "zlc_plot.fit",
        "FitScope": "zlc_plot._fit_projection",
        "FitSelection": "zlc_plot._fit_projection",
        "NotebookView": "zlc_plot.notebook",
        "ParameterControl": "zlc_plot.ui",
        "PlotLibraryDefaults": "zlc_plot.config",
        "PointMarker": "zlc_plot.primitives",
        "RasterBuffer": "zlc_plot.raster",
        "RasterFront": "zlc_plot.raster",
        "RasterIdentity": "zlc_plot.raster",
        "RasterOperation": "zlc_plot.raster",
        "RectangleRange": "zlc_plot.selectors",
        "RegularImageFitInput": "zlc_plot.fit",
        "RelimMode": "zlc_plot.specs",
        "ReplaceSpecInitialState": "zlc_plot.session_policy",
        "RevisionError": "zlc_plot.errors",
        "SelectionData": "zlc_plot.session",
        "SelectionEvent": "zlc_plot.session",
        "SelectorState": "zlc_plot.selectors",
        "SemanticChoice": "zlc_plot.semantics",
        "SemanticDescription": "zlc_plot.semantics",
        "SemanticFeasibility": "zlc_plot.semantics",
        "SemanticField": "zlc_plot.semantics",
        "SessionRevisions": "zlc_plot.session",
        "UnitError": "zlc_plot.errors",
        "ZLCPlotError": "zlc_plot.errors",
        "axis_choices_for_schema": "zlc_plot.semantics",
        "builtin_fit_models": "zlc_plot.fit",
        "default_spec": "zlc_plot._kinds",
        "replace_spec_initial_state": "zlc_plot.session_policy",
        "semantic_controls": "zlc_plot.ui",
    }
    assert set(paths) == REMOVED_NAMES
    for name, module_name in paths.items():
        assert hasattr(import_module(module_name), name)
