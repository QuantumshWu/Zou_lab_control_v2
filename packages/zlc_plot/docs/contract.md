# zlc_plot public contract

This file is the machine-readable name contract for the top-level
`zlc_plot` facade. The package's `__all__` and this list must stay identical.
Implementation details remain importable from their owning submodules, but a
name is not part of the convenience API merely because a module happens to
load it.

<!-- zlc_plot-public-names:start -->
AxisRef
BackendUnavailableError
CurvePlot
DEFAULTS
DEFAULT_UNITS
FacetGridPlot
FitCancelled
FitEvent
FitModelSpec
FitTarget
HistogramPlot
ImageFrame
ImagePlot
ImagePointOverlay
LiveDataRevision
LivePlotController
NumericRange
PANEL_SIZE_NAMES
PlotKind
PlotLabels
PlotSession
PlotSpec
PointStatus
PulseAnalogTrace
PulseBlock
PulseChannel
PulseDacScanSegment
PulseRepeatMarker
PulseScanRegion
PulseTimelineData
PulseTimelinePlot
PulseTimelineSelectionData
Qt5ParameterPanel
Qt5PlotWidget
RasterPlotHost
Reduction
RollingPlot
SelectionChange
SelectorData
SelectorKind
Unit
UnitRegistry
__version__
```
```text
curve
describe_semantics
edit_plot_display
ensure_qt5_application
facet_grid
fitting_spec
histogram
image
image_axes
panel_kinds
parameter_controls
pulse_timeline
recommended_pulse_preset
resolve_unit
rolling
schema_summary
show
updated_spec
<!-- zlc_plot-public-names:end -->

`FitCancelled` and `BackendUnavailableError` are the two individually useful
exceptions on the facade. `FitCancelled` is raised through fit futures when a
request is superseded, cleared, or closed; `BackendUnavailableError` reports a
missing optional frontend. The internal `ZLCPlotError`, `UnitError`,
`RevisionError`, `FitDeadlineExceeded`, and `LiveUpdateError` remain available
from their owning submodules only. Callers should catch their standard bases
(`ValueError`, `TimeoutError`, or `RuntimeError`) when they need a broader
policy; the facade does not expose a partial exception hierarchy.

`MAX_PUBLIC_NAMES` is a guard constant, not a plotting API name. It bounds the
real non-module names visible on the package object at 57 while leaving two
reviewed slots above the 55-name contract.
