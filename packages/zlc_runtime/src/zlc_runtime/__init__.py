"""Finite public facade for the dependency-light runtime primitives."""

MAX_PUBLIC_NAMES = 23
__version__ = "0.1.0"

from .dataset import DatasetCoverage, MonitorCoverage
from .dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from .plane import (
    SignalDataPlane,
    SignalDescription,
    SignalPublication,
    SignalValue,
)
from .presentation import (
    BoardScheduler,
    HarmonicClock,
    OwnerChannels,
    SurfaceBatchArbiter,
    SurfaceUpdate,
)
from .selection_bridge import (
    FitEventValue,
    SelectionBridge,
    SelectionChange,
    SelectionRange,
    SelectionState,
)
from .host import NodeHost
from .streams import AcquisitionStream

__all__ = (
    # Declared because siblings use them.  Reaching into a submodule for a
    # name this list does not carry is the same surface with the declaration
    # skipped, and nobody can then see how wide it really is.
    "BoardScheduler",
    "DatasetCoverage",
    "DatasetOutputDeclaration",
    "FinalDatasetOutput",
    "HarmonicClock",
    "LiveDatasetOutput",
    "MonitorCoverage",
    "OwnerChannels",
    "SurfaceBatchArbiter",
    "SurfaceUpdate",
    "SignalDataPlane",
    "SignalValue",
    "SignalPublication",
    "SignalDescription",
    "AcquisitionStream",
    "NodeHost",
    "SelectionBridge",
    "__version__",
    "SelectionChange",
    "SelectionRange",
    "SelectionState",
    "FitEventValue",
)

# Importing focused submodules necessarily adds their module objects to the
# package namespace.  The facade is an allow-list, so keep those implementation
# paths available only through explicit imports such as ``zlc_runtime.host``.
_FACADE_METADATA = {"MAX_PUBLIC_NAMES"}
for _name in tuple(globals()):
    if (
        not _name.startswith("_")
        and _name not in __all__
        and _name not in _FACADE_METADATA
    ):
        del globals()[_name]
