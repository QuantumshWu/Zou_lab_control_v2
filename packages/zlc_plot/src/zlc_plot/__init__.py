"""Small, stable facade for the zlc_plot plotting API.

The implementation remains available from its owning submodules.  Only the
names in ``__all__`` are the supported convenience surface; backend, raster,
fit-engine and semantic implementation details are intentionally not flattened
into this namespace.
"""

from .api import curve, facet_grid, histogram, image, pulse_timeline, rolling, show
from .backends import (
    BackendUnavailableError,
    ensure_qt5_application,
)
from .config import DEFAULTS
from .fit import FitCancelled, FitModelSpec, FitTarget
from .kinds import AxisRef, PlotKind
from .live import LiveDataRevision, LivePlotController
from .primitives import (
    ImageFrame,
    ImagePointOverlay,
    PointStatus,
    PulseAnalogTrace,
    PulseBlock,
    PulseChannel,
    PulseDacScanSegment,
    PulseRepeatMarker,
    PulseScanRegion,
    PulseTimelineData,
)
from ._kinds import fitting_spec, panel_kinds
# What sizes a panel may be, and which one suits a pulse of a given shape.
# The window offers the first in a box and the second is what it offers when
# nobody has chosen -- neither is something a host can work out for itself
# without re-deriving this package's layout rules.
from .layout import PANEL_SIZE_NAMES, recommended_pulse_preset
from .data_contract import image_axes
from .raster import RasterPlotHost
from .selectors import NumericRange, SelectorKind
from .session import (
    FitEvent,
    PlotSession,
    PulseTimelineSelectionData,
    SelectionChange,
    SelectorData,
)
from .semantics import describe_semantics, schema_summary, updated_spec
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotLabels,
    PlotSpec,
    PulseTimelinePlot,
    Reduction,
    RollingPlot,
)
from .ui import parameter_controls
from .units import DEFAULT_UNITS, Unit, UnitRegistry, resolve_unit


__version__ = "1.1.0"

# Keep a little headroom for an explicitly reviewed addition without letting
# accidental imports turn the facade back into a second implementation API.
# 59 -> 61 for the two the pulse editor was reaching into .layout for: which
# sizes a panel may be, and which one suits a pulse of a given shape.  A host
# cannot answer either without re-deriving this package's layout rules, and a
# host that re-derives them is a second layout engine.
MAX_PUBLIC_NAMES = 61


def __getattr__(name: str) -> object:
    """Load optional Qt classes only when the caller explicitly requests one."""

    if name == "Qt5PlotWidget":
        from .backends import _qt5_plot_widget_class

        value = _qt5_plot_widget_class()
    elif name == "Qt5ParameterPanel":
        from .qt_controls import _qt5_parameter_panel_class

        value = _qt5_parameter_panel_class()
    elif name == "edit_plot_display":
        from .qt_controls import edit_plot_display as value
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"MAX_PUBLIC_NAMES"})


__all__ = [
    "PANEL_SIZE_NAMES",
    "recommended_pulse_preset",
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
    "fitting_spec",
    "panel_kinds",
    "HistogramPlot",
    "ImageFrame",
    "ImagePlot",
    "ImagePointOverlay",
    "LiveDataRevision",
    "LivePlotController",
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
    "edit_plot_display",
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
    "ensure_qt5_application",
    "facet_grid",
    "histogram",
    "image_axes",
    "image",
    "parameter_controls",
    "pulse_timeline",
    "resolve_unit",
    "rolling",
    "schema_summary",
    "show",
    "updated_spec",
]
