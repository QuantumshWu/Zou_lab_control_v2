"""Small, stable facade for the zlc_plot plotting API.

The implementation remains available from its owning submodules.  Only the
names in ``__all__`` are the supported convenience surface; backend, raster,
fit-engine and semantic implementation details are intentionally not flattened
into this namespace.

The facade RESOLVES those names, it does not import them.  Importing every
submodule to publish a name pulled matplotlib -- and its font cache -- into
the cost of the word ``import zlc_plot``, so a window that had not drawn
anything yet, and a notebook cell that only wanted a constant, each paid about
1.9 s before doing anything at all.  Nothing here is optional-dependency
defensive: it is the difference between a package you may mention and a
package you must load.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    "LATEST_COORDINATE": ("zlc_data", "LATEST_COORDINATE"),
    "curve": ("zlc_plot.api", "curve"),
    "facet_grid": ("zlc_plot.api", "facet_grid"),
    "histogram": ("zlc_plot.api", "histogram"),
    "image": ("zlc_plot.api", "image"),
    "pulse_timeline": ("zlc_plot.api", "pulse_timeline"),
    "rolling": ("zlc_plot.api", "rolling"),
    "show": ("zlc_plot.api", "show"),
    "BackendUnavailableError": ("zlc_plot.backends", "BackendUnavailableError"),
    "ensure_qt5_application": ("zlc_plot.backends", "ensure_qt5_application"),
    "DEFAULTS": ("zlc_plot.config", "DEFAULTS"),
    "FitCancelled": ("zlc_plot.fit", "FitCancelled"),
    "FitModelSpec": ("zlc_plot.fit", "FitModelSpec"),
    "FitTarget": ("zlc_plot.fit", "FitTarget"),
    "build_figure_host": ("zlc_plot.figure_artifact", "build_figure_host"),
    "decode_plot_recipe": ("zlc_plot.figure_artifact", "decode_plot_recipe"),
    "encode_plot_recipe": ("zlc_plot.figure_artifact", "encode_plot_recipe"),
    "open_figure_host": ("zlc_plot.figure_artifact", "open_figure_host"),
    "read_figure_plot": ("zlc_plot.figure_artifact", "read_figure_plot"),
    "save_figure_artifact": ("zlc_plot.figure_artifact", "save_figure_artifact"),
    "AxisRef": ("zlc_plot.kinds", "AxisRef"),
    "PlotKind": ("zlc_plot.kinds", "PlotKind"),
    "GRID_CELL_KINDS": ("zlc_plot.specs", "GRID_CELL_KINDS"),
    "ImageFrame": ("zlc_plot.primitives", "ImageFrame"),
    "ImagePointOverlay": ("zlc_plot.primitives", "ImagePointOverlay"),
    "IMAGE_POINT_OVERLAY_CONTRACT": (
        "zlc_plot.primitives",
        "IMAGE_POINT_OVERLAY_CONTRACT",
    ),
    "IMAGE_POINT_OVERLAY_GEOMETRY_RECORD": (
        "zlc_plot.primitives",
        "IMAGE_POINT_OVERLAY_GEOMETRY_RECORD",
    ),
    "image_point_overlay_from_signal": (
        "zlc_plot.primitives",
        "image_point_overlay_from_signal",
    ),
    "image_point_overlay_geometry": (
        "zlc_plot.primitives",
        "image_point_overlay_geometry",
    ),
    "PointStatus": ("zlc_plot.primitives", "PointStatus"),
    "ImagePointReviewSurface": (
        "zlc_plot.point_review",
        "ImagePointReviewSurface",
    ),
    "PulseAnalogTrace": ("zlc_plot.primitives", "PulseAnalogTrace"),
    "PulseBlock": ("zlc_plot.primitives", "PulseBlock"),
    "PulseChannel": ("zlc_plot.primitives", "PulseChannel"),
    "PulseDacScanSegment": ("zlc_plot.primitives", "PulseDacScanSegment"),
    "PulseRepeatMarker": ("zlc_plot.primitives", "PulseRepeatMarker"),
    "PulseScanRegion": ("zlc_plot.primitives", "PulseScanRegion"),
    "PulseTimelineData": ("zlc_plot.primitives", "PulseTimelineData"),
    "fitting_spec": ("zlc_plot._kinds", "fitting_spec"),
    "PANEL_SIZE_NAMES": ("zlc_plot.layout", "PANEL_SIZE_NAMES"),
    "recommended_pulse_preset": ("zlc_plot.layout", "recommended_pulse_preset"),
    "image_axes": ("zlc_plot.data_contract", "image_axes"),
    "RasterPlotHost": ("zlc_plot.raster", "RasterPlotHost"),
    "NumericRange": ("zlc_plot.selectors", "NumericRange"),
    "SelectorKind": ("zlc_plot.selectors", "SelectorKind"),
    "normalize_classifier_threshold_targets": (
        "zlc_plot.selectors",
        "normalize_classifier_threshold_targets",
    ),
    "FitEvent": ("zlc_plot.session", "FitEvent"),
    "PlotSession": ("zlc_plot.session", "PlotSession"),
    "PulseTimelineSelectionData": ("zlc_plot.session", "PulseTimelineSelectionData"),
    "SelectionChange": ("zlc_plot.session", "SelectionChange"),
    "SelectionSubject": ("zlc_plot.data_view", "SelectionSubject"),
    "SelectorData": ("zlc_plot.session", "SelectorData"),
    "describe_semantics": ("zlc_plot.semantics", "describe_semantics"),
    "schema_summary": ("zlc_plot.semantics", "schema_summary"),
    "updated_spec": ("zlc_plot.semantics", "updated_spec"),
    "CurvePlot": ("zlc_plot.specs", "CurvePlot"),
    "accepts_classifier_thresholds": (
        "zlc_plot.specs",
        "accepts_classifier_thresholds",
    ),
    "FacetGridPlot": ("zlc_plot.specs", "FacetGridPlot"),
    "HistogramPlot": ("zlc_plot.specs", "HistogramPlot"),
    "history_window_requirement": (
        "zlc_plot.specs",
        "history_window_requirement",
    ),
    "ImagePlot": ("zlc_plot.specs", "ImagePlot"),
    "PlotLabels": ("zlc_plot.specs", "PlotLabels"),
    "PlotSpec": ("zlc_plot.specs", "PlotSpec"),
    "paints_image_surface": ("zlc_plot.specs", "paints_image_surface"),
    "PulseTimelinePlot": ("zlc_plot.specs", "PulseTimelinePlot"),
    "Reduction": ("zlc_plot.specs", "Reduction"),
    "RollingPlot": ("zlc_plot.specs", "RollingPlot"),
    "parameter_controls": ("zlc_plot.ui", "parameter_controls"),
    "DEFAULT_UNITS": ("zlc_plot.units", "DEFAULT_UNITS"),
    "Unit": ("zlc_plot.units", "Unit"),
    "UnitRegistry": ("zlc_plot.units", "UnitRegistry"),
    "resolve_unit": ("zlc_plot.units", "resolve_unit"),
}


def __getattr__(name: str) -> object:
    """Resolve one facade name, importing only what that name lives in.

    Every public name arrives through here, so the cost of a name is the cost
    of its own module: ``PANEL_SIZE_NAMES`` loads the layout rules and nothing
    else, while ``RasterPlotHost`` loads the drawing stack because drawing is
    what it does.  Resolved names are cached in ``globals()``, so this runs at
    most once each.
    """

    if name == "Qt5PlotWidget":
        from .backends import _qt5_plot_widget_class

        value = _qt5_plot_widget_class()
    elif name == "Qt5ParameterPanel":
        from .qt_controls import _qt5_parameter_panel_class

        value = _qt5_parameter_panel_class()
    elif name == "edit_plot_display":
        from .qt_controls import edit_plot_display as value
    else:
        where = _EXPORTS.get(name)
        if where is None:
            raise AttributeError(name)
        module, attribute = where
        value = getattr(_import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "accepts_classifier_thresholds",
    "PANEL_SIZE_NAMES",
    "recommended_pulse_preset",
    "AxisRef",
    "BackendUnavailableError",
    "CurvePlot",
    "build_figure_host",
    "decode_plot_recipe",
    "DEFAULTS",
    "DEFAULT_UNITS",
    "FacetGridPlot",
    "FitCancelled",
    "FitEvent",
    "FitModelSpec",
    "FitTarget",
    "fitting_spec",
    "HistogramPlot",
    "history_window_requirement",
    "ImageFrame",
    "ImagePlot",
    "ImagePointOverlay",
    "ImagePointReviewSurface",
    "LATEST_COORDINATE",
    "IMAGE_POINT_OVERLAY_CONTRACT",
    "IMAGE_POINT_OVERLAY_GEOMETRY_RECORD",
    "image_point_overlay_from_signal",
    "image_point_overlay_geometry",
    "NumericRange",
    "open_figure_host",
    "PlotKind",
    "GRID_CELL_KINDS",
    "PlotLabels",
    "PlotSession",
    "PlotSpec",
    "paints_image_surface",
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
    "edit_plot_display",
    "Qt5PlotWidget",
    "RasterPlotHost",
    "Reduction",
    "RollingPlot",
    "SelectionChange",
    "SelectionSubject",
    "SelectorData",
    "SelectorKind",
    "normalize_classifier_threshold_targets",
    "Unit",
    "UnitRegistry",
    "curve",
    "describe_semantics",
    "ensure_qt5_application",
    "encode_plot_recipe",
    "facet_grid",
    "histogram",
    "image_axes",
    "image",
    "parameter_controls",
    "pulse_timeline",
    "resolve_unit",
    "read_figure_plot",
    "rolling",
    "schema_summary",
    "save_figure_artifact",
    "show",
    "updated_spec",
]
